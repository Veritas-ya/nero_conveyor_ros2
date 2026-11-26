"""Single-sender experimental prefix/feedback-gated descent or prechecked stop."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import GripConfirmation, RestArrival, TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .grip_recovery import ClosingGrip
from .hover_gate import SynchronizationGate, BranchDecision
from .hover_planning import prepare_hover, concatenate
from .planning import InterceptPlan
from .phase_timing import verify_plan
from .runtime_sweep import scene_snapshot, request_for, sweep
from .streaming import fresh_feedback, CheckedWindowExpired


def record_path(e, name, path, **extra):
    e._emit(name, dict(identity=path.identity, version=path.version,
        duration_s=path.duration, curve_c=path.curve.c.copy(), curve_x=path.curve.x.copy(),
        diagnostics=path.diagnostics, **extra))


def staged_stream(e, bundle, future, *, sender_factory=JointStreamSender):
    """No planner work or waiting in callbacks; all motion uses one JS sender."""
    if e.config.raw.get('brake_no_descent_rise_m',0.):
        raise PickRejected('分段悬停发送器尚无闭爪高度认证，不能执行该轨迹')
    cfg=e.config; identity=e.owner; version=e.version
    origin=bundle.old.reference.start_s; advance=cfg.raw['command_advance_s']
    gate=SynchronizationGate(cfg,bundle); decision=BranchDecision(bundle,cfg)
    stop_path=concatenate(bundle.prefix,bundle.stopping)
    current={'path':stop_path, 'candidate':None, 'failed':False, 'adopted':False,
             'closed':False, 'confirmation':None, 'held':False, 'endpoint_sent':False}
    arrival=RestArrival(origin+stop_path.duration,tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
    context=SimpleNamespace(env=SimpleNamespace(target_identity=e.trace.target_identity),
        _full_pick_request=SimpleNamespace(generation=bundle.prefix.version,
            request_id='hover_descent_staged', planned_target_time_s=None))
    e.stage='hover_descent_staged'
    e._emit('hover_stream_started',dict(origin_s=origin, switch_s=origin+bundle.switch_s,
        decision_deadline_s=decision.deadline, grasp_enabled=decision.grasp))

    def active(state,now):
        e._active(identity,version,identity[0])
        if e.planner.version != bundle.prefix.version or e.planner.scene_revision != bundle.scene_revision:
            raise PickRejected('分段执行期间目标/轨迹/场景版本改变')
        fresh_feedback(state,now,e.feedback_stamps)

    closing_grip = None

    def step(state, dispatch, dt):
        nonlocal arrival, closing_grip
        active(state,e.clock())
        elapsed=dispatch+advance-origin
        result=gate.observe(state,e.clock())
        before=decision.branch
        branch=decision.update(e.clock(),result,candidate_ready=current['candidate'] is not None,
                               candidate_failed=current['failed'])
        if before is None and result.get('reason')!='not_yet_synchronized':
            e._emit('hover_feedback_gate',result)
        if branch is not None and before is None:
            current['decision_reason']=decision.reason
            current['synchronization_verified']=bool(result.get('allowed'))
            if branch=='descent':
                current['path']=current['candidate']['combined']
                current['adopted']=True
                arrival=RestArrival(origin+current['path'].duration,
                    tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
            e._emit('hover_branch_selected',dict(branch=branch,reason=decision.reason,
                selected_s=e.clock(),deadline_s=decision.deadline,gate=result))
        path=current['path']
        if elapsed<0 or elapsed>path.duration+cfg.settings.move_timeout_s:
            raise PickRejected('分段发送时钟无效或终点未到位')
        q,qd,qdd=path.sample(elapsed)
        if e.command_history.q is None:
            e.command_history.validate_first(q,path.q0)
        joint_time=mapped_stamp(state,'joint')
        tracking=float(np.max(abs(np.asarray(state['joints'])-path.position(joint_time-origin))))
        if joint_time>=origin and tracking>.08:
            raise PickRejected('分段运动跟踪偏差过大，停止发送',details={'joint_error_rad':tracking})
        if not current['closed'] and abs(state['width_m']-cfg.settings.open_width_m)>.002:
            raise PickRejected('同步前段或下降期间夹爪开口异常')
        ref=current['candidate']['reference'] if current['adopted'] else None
        if ref is not None and not current['closed'] and elapsed>=bundle.switch_s+ref.duration_s:
            with e.command_lock:
                e._active(identity,version,identity[0])
                e.device.grip(0.,cfg.settings.force_n)
                closed_at=e.clock()
            current['closed']=True
            closing_grip=ClosingGrip(e.planner.target,closed_at,cfg.settings)
            current['confirmation']=GripConfirmation(cfg.settings.open_width_m,cfg.settings.min_force_n,closed_at)
            e._emit('close_during_follow',dict(close_command_completed_monotonic_s=closed_at,
                reference_time_s=elapsed-bundle.switch_s,closure_policy='scheduled_attempt',
                source='hover_descent', physical_closure_proven=False))
        if closing_grip is not None and not current['held']:
            closing_grip.observe(state)
        if (current['confirmation'] is not None and not current['held'] and
            current['confirmation'].observe(state['width_m'],state['force_n'],mapped_stamp(state,'gripper'))):
            current['held']=True
            e.holding.held_width=state['width_m']
            e.planner.attach_measured(state,joint_time)
            e._emit('hover_grip_confirmed',dict(joint_feedback_s=joint_time,
                width_m=state['width_m'],force_n=state['force_n'],physical_lift_proven=False))
        if current['held']:
            e.holding.check_holding(state)
        request=request_for(e.planner,state,dispatch=dispatch,origin=origin,advance=advance,
            control_hz=cfg.settings.control_hz,reference=None,attached=current['held'],
            smooth=True,scheduled_attempt=True,follow_end=path.duration,grip_confirmed=current['held'])
        # Every prefix/empty-stop sample includes the moving object cloud.
        request.update(moving_target=not current['held'],contact=bool(ref is not None and
            elapsed>=bundle.switch_s+ref.duration_s),allow_belt_support=bool(current['held']))
        sweep(e.planner,path,request,cancelled=e.cancelled.is_set)
        e.command_history.validate(q,dispatch)
        if elapsed>=path.duration and current['endpoint_sent'] and arrival.observe(
                state['joints'],path.q1,joint_time):
            return None
        e.trace.prepare_command(context,policy_time_s=elapsed,candidate=q,stage=e.stage,
            stream_id='hover_descent_staged',trajectory=path,edge_started=0.)
        return q,dict(reference_time_s=elapsed,prepared_from_monotonic_s=dispatch,
            prepared_until_monotonic_s=request['until_s'],trajectory_version=path.version,
            phase='hover_prefix' if elapsed<bundle.switch_s else ('descent_follow_brake' if current['adopted'] else 'hover_stop'),
            raw_velocity_rad_s=qd.tolist(),raw_acceleration_rad_s2=qdd.tolist())

    def validate(state,dispatch,dt,q,metadata):
        active(state,dispatch)
        if not metadata['prepared_from_monotonic_s']-1e-9<=dispatch<=metadata['prepared_until_monotonic_s']+1e-9:
            raise CheckedWindowExpired('分段指令已超出本次实际几何检查时窗')
        e.command_history.validate(q,dispatch)

    def finalize(state,dispatch,dt,q,metadata):
        validate(state,dispatch,dt,q,metadata)
        elapsed=dispatch+advance-origin
        q,qd,qdd=current['path'].sample(elapsed)
        e.command_history.validate(q,dispatch)
        metadata.update(reference_time_s=elapsed,raw_velocity_rad_s=qd.tolist(),raw_acceleration_rad_s2=qdd.tolist())
        e.trace.update_final_sample(elapsed,q,qd,qdd,dispatch)
        return q,metadata

    def sent(dispatch,dt,q,metadata):
        e.motion_started=True
        e.command_history.record(dispatch,dt,q,metadata)
        current['endpoint_sent']=metadata['reference_time_s']>=current['path'].duration

    processed=False; report=None
    try:
        e.sender=sender_factory(e.device,step,sent,control_hz=cfg.settings.control_hz,
            clock=e.clock,sleep=e.sleep,pause_requested=e.cancelled.is_set,
            initial_dispatch_time=origin-advance,finalize_send=finalize,validate_dispatch=validate).start()
        while not e.sender.wait_stopped(.005):
            e._active(identity,version,identity[0])
            if not processed and future.done():
                processed=True
                try:
                    result=future.result()
                    if not result['ok']:
                        raise PickRejected(result.get('error','后段计算失败'),details={'attempts':result.get('attempts',[])})
                    if (tuple(result['identity'])!=identity or result['version']!=bundle.prefix.version
                            or result['scene_revision']!=bundle.scene_revision):
                        raise PickRejected('后段结果属于旧目标/版本/场景')
                    verify_plan(cfg,result['path'],result['reference'],for_execution=True)
                    if abs(result['reference'].start_s-(origin+bundle.switch_s))>1e-9:
                        raise PickRejected('后段绝对接续时刻不一致')
                    result['combined']=concatenate(bundle.prefix,result['path'])
                    record_path(e,'hover_suffix_ready',result['path'],completed_s=result['completed_s'],
                        received_s=e.clock(),attempts=result['attempts'])
                    # Publication happens last; late results never change a chosen stop.
                    current['candidate']=result
                except Exception as error:
                    current['failed']=True
                    e._emit('hover_suffix_rejected',dict(error=str(error),details=getattr(error,'details',{})))
        e.sender.join(); report=e.sender.snapshot()
        e.trace.record_sender_events(getattr(report,'events',()),stage=e.stage)
        if report.exception is not None:
            raise report.exception
        e._active(identity,version,identity[0])
        if not current['endpoint_sent']:
            raise PickRejected('分段发送器未交付终点')
    except BaseException as error:
        if e.sender is not None:
            e.sender.request_stop(); e.sender.join()
        e.device.stop()
        if isinstance(error, PickRejected) and closing_grip is not None:
            closing_grip.mark_collision(error)
        raise
    finally:
        if e.sender is not None:
            e.sender.request_stop(); e.sender.join(); e.sender=None
    e.rest_anchor=current['path'].q1.copy()
    e._emit('hover_stream_finished',dict(branch=decision.branch,reason=decision.reason,
        commands=report.command_count,grip_confirmed=current['held'],
        scheduled_endpoint_s=origin+current['path'].duration,physical_success=None))
    return current,report.last_feedback


def execute_intercept(e,pool,target,state):
    if e.config.raw.get('_hover_descent_authorized') is not True:
        raise PickRejected('分段接续未从独立入口授权')
    service=getattr(e.planner,'hover_service',None)
    if service is None:
        raise PickRejected('分段规划进程未在启动前准备好')
    e.stage='planning_hover_prefix'
    archive=getattr(e.planner,'experiment_archive',None)
    if archive is not None:
        archive.offer_target(target)
    e._emit('hover_planning_submitted',dict(capture_s=target.capture_s,submission_s=e.clock(),
        identity=target.identity,start_state=state))
    bundle=e._prepare(pool.submit(prepare_hover,e.planner,target,state,cancelled=e.cancelled.is_set),
                      e.owner,e.version,target.epoch)
    record_path(e,'hover_prefix_ready',bundle.prefix,evidence=bundle.evidence)
    record_path(e,'hover_stop_ready',bundle.stopping,switch_s=bundle.switch_s)
    deadline=bundle.old.reference.start_s-e.config.raw['command_advance_s']
    decision_deadline=BranchDecision(bundle,e.config).deadline
    latest=e.device.snapshot()
    fresh_feedback(latest,e.clock(),e.feedback_stamps)
    e.planner.model.check_feedback(latest)
    if (np.max(abs(np.asarray(latest['joints'])-bundle.prefix.q0))>.005
            or abs(latest['width_m']-e.config.settings.open_width_m)>.002
            or latest['force_n']>=e.config.settings.min_force_n):
        raise PickRejected('前段规划期间等待位或夹爪状态改变')
    if e.clock()>=deadline-e.config.raw['hover_descent']['send_guard_s']:
        raise PickRejected('分段前段交付已过期，不补发历史指令')
    future=service.submit(scene_snapshot(e.planner,bundle.prefix),bundle,state,decision_deadline)
    begin=e.clock()
    try:
        with e.command_lock:
            e._active(e.owner,e.version,target.epoch)
            e.motion_started=True
            e.device.prepare_motion(e.config.settings.speed_percent,
                query_limits=e.config.settings.query_controller_limits,
                requested_velocity_limits=e.config.velocity_limits,
                requested_acceleration_limits=e.config.acceleration_limits)
        latest=e.device.snapshot()
        fresh_feedback(latest,e.clock(),e.feedback_stamps)
        if (e.clock()>=deadline or np.max(abs(np.asarray(latest['joints'])-bundle.prefix.q0))>.005
                or abs(latest['width_m']-e.config.settings.open_width_m)>.002
                or latest['force_n']>=e.config.settings.min_force_n):
            raise PickRejected('设备准备后前段已过期或起点改变')
        e.trace.policy_origin=bundle.old.reference.start_s
        current,final=staged_stream(e,bundle,future)
    finally:
        service.cancel.set()
        try:
            terminal=future.result(timeout=10.)  # Sender stopped; join before recovery/reuse.
        except TimeoutError as error:
            e.device.stop()
            raise PickRejected('后段规划进程未响应取消') from error
        except Exception as error:
            # A failed, finished future already selected the stop branch.
            # Preserve that outcome instead of hiding it with a second raise.
            e._emit('hover_suffix_terminal',dict(ok=False,error=str(error),error_type=type(error).__name__))
        else:
            e._emit('hover_suffix_terminal', {k:v for k,v in terminal.items()
                if k not in ('path','reference','combined')})
    if not current['adopted']:
        raise PickRejected('同步测试结束或后段未接入，已按认证退路停车',details={'reason':'hover_stopped',
            'decision_reason':current.get('decision_reason'),
            'synchronization_verified':current.get('synchronization_verified',False)})
    result=current['candidate']
    plan=InterceptPlan(result['path'],result['reference'],target,bundle.scene_revision,bundle.grasp,(),{})
    return plan,final,begin

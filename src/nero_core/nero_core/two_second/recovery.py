"""Abandon a task, retain its last target, then execute newly checked recovery routes."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD, OpenConfirmation
from .streaming import fresh_feedback
from .stationary_hold import StationaryHold
from .workpiece_contact import recovery_workpiece_contact


class RecoveryFailed(PickRejected):
    def __init__(self, original, recovery):
        super().__init__('本次抓取取消，但恢复等待位失败：'+str(recovery), details={
            'original_error': str(original), 'original_error_type': type(original).__name__,
            'original_details': getattr(original, 'details', {}),
            'recovery_error': str(recovery), 'recovery_error_type': type(recovery).__name__,
            'recovery_details': getattr(recovery, 'details', {})})


def _fk_mismatch(error):
    details=getattr(error,'details',None) or {}
    return (details.get('reason')=='feedback_fk_mismatch' or
        all(k in details for k in ('position_error_m','orientation_error_deg','position_tolerance_m','joint_source')))


def _recovery_feedback(e,*,epoch,failure=None):
    """Read only: require new measured feedback after a rejected FK comparison.

    The sender has already joined. Do not send recovery targets or open the
    gripper while waiting; do not accept the failed sample or widen FK limits.
    """
    began=e.clock();deadline=began+.2
    identity,version=e.owner,e.version
    last=failure if failure is not None and _fk_mismatch(failure) else None
    details=(getattr(last,'details',None) or {})
    blocked=(details.get('pose_joint_pairing') or {}).get('pose_stamp')
    if blocked is None and details.get('pose_packet_stamps_monotonic_s'):
        blocked=min(details['pose_packet_stamps_monotonic_s'])
    waited=False;checks=0;anchor_error=None
    state=e.device.snapshot_nowait()
    while True:
        e._active(identity,version,epoch)
        if state is not None:
            fresh_feedback(state,e.clock(),e.feedback_stamps)
            stamp=mapped_stamp(state,'pose')
            if blocked is None or stamp>blocked:
                try:
                    checks+=1
                    e.planner.model.check_feedback(state)
                except PickRejected as error:
                    if not _fk_mismatch(error):raise
                    last=error;blocked=stamp
                else:
                    last=None
                    anchor=e.command_history.q
                    anchor_error=(0. if anchor is None else
                        float(np.max(abs(np.asarray(state['joints'])-anchor))))
                    if anchor_error<=TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                        if waited:e._emit('recovery_feedback_rechecked',{
                            'waited_s':e.clock()-began,'new_pose_stamp':stamp,'fk_checks':checks,
                            'last_target_error_rad':anchor_error})
                        return state
                    blocked=stamp
        if not waited:
            e._emit('recovery_feedback_wait',{'last_fk_error':None if last is None else last.details,
                'requires_new_pose_after':blocked,'timeout_s':.2,'recovery_targets_sent':0,
                'last_target_error_rad':anchor_error})
            waited=True
        remaining=deadline-e.clock()
        if remaining<=0:
            raise PickRejected('恢复前等待新反馈与最后目标到位超时',details={
                'reason':'recovery_feedback_timeout','waited_s':e.clock()-began,
                'last_fk_error':None if last is None else last.details,'fk_checks':checks,
                'last_target_error_rad':anchor_error})
        e.sleep(min(.005,remaining))
        state=e.device.snapshot_nowait()


def _payload_state(e, state):
    width, force = float(state['width_m']), float(state['force_n'])
    opening, minimum_force = e.config.settings.open_width_m, e.config.settings.min_force_n
    if not np.isfinite([width, force]).all():
        raise PickRejected('恢复时夹爪反馈无效')
    if (force < minimum_force and
            (abs(width-opening) <= .002 or 0. <= width <= .003)):
        return False
    if (e.holding.held_width is not None and e.planner.payload_local is not None and
            .003 < width < opening-.004 and force >= minimum_force):
        e.holding.check_holding(state)
        return True
    raise PickRejected('恢复时无法确认是否带载；保持夹爪，等待人工处理')


def _settle_last_target(e, state, *, epoch, carrying, sender_factory):
    """Repeat the last issued position, without jumping to a new recovery target.

    This is a position hold, not a measured acceleration or a certified physical
    deceleration. Require new stable joint feedback before planning any route.
    """
    anchor = np.asarray(state['joints'] if e.command_history.q is None else e.command_history.q).copy()
    if np.max(abs(anchor-np.asarray(state['joints']))) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
        raise PickRejected('恢复时实际关节距离最后指令超过1度')
    actual = np.asarray(state['joints'])
    count = max(1, int(np.ceil(np.max(abs(anchor-actual))/.008)))
    for u in np.linspace(0., 1., count+1):
        e.planner.valid(actual+(anchor-actual)*u, state['width_m'], payload=carrying,
            stamp=None if carrying else e.clock(), contact=False, allow_belt_support=carrying)
    e.rest_anchor = anchor
    e.stage = 'recovery_settling'
    e.motion_started = True
    until = e.clock()+e.config.settings.move_timeout_s
    window, last_stamp = [], None
    # The hold sender and this observer obtain separate snapshots concurrently.
    # Compare timestamps in each observer's own acquisition order; sharing a
    # mutable high-water mark can mistake a delayed older snapshot for a clock
    # reset after the other thread has already processed a newer snapshot.
    observer_stamps = dict(e.feedback_stamps)
    with StationaryHold(e, sender_factory, epoch=epoch, carrying=carrying) as hold:
        while e.clock() < until:
            e._active(e.owner, e.version, epoch)
            hold.check()
            current = e.device.snapshot_nowait()
            if current is not None:
                fresh_feedback(current, e.clock(), observer_stamps)
                stamp = mapped_stamp(current, 'joint')
                if stamp != last_stamp:
                    last_stamp = stamp
                    q = np.asarray(current['joints'], float)
                    window.append((stamp, q.copy()))
                    if np.max(abs(q-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                        raise PickRejected('恢复保持期间实际关节距离最后指令超过1度')
                    if _payload_state(e, current) != carrying:
                        raise PickRejected('恢复保持期间夹持状态改变')
                    # Every sample in this fresh 0.1-second window must agree.
                    while len(window) > 1 and stamp-window[1][0] >= .1:
                        window.pop(0)
                    if stamp-window[0][0] >= .1-1e-9:
                        spread = np.ptp(np.array([x[1] for x in window]), axis=0)
                        if np.max(spread) <= .001:
                            return current
            e.sleep(.01)
    raise PickRejected('恢复保持后未获得连续稳定的关节反馈')


def _open_empty(e, state, *, epoch, sender_factory):
    if _payload_state(e, state):
        raise PickRejected('恢复回位后仍然带载，不能直接张爪')
    for width in np.linspace(state['width_m'], e.config.settings.open_width_m, 11):
        e.planner.valid(state['joints'], width, payload=False)
    e.stage = 'recovery_open_empty'
    with StationaryHold(e, sender_factory, epoch=epoch, carrying=False) as hold:
        with e.command_lock:
            e._active(e.owner, e.version, epoch)
            e.device.grip(e.config.settings.open_width_m, 1.)
            completed = e.clock()
        opening = OpenConfirmation(e.config.settings.open_width_m, not_before=completed)
        return e.holding.wait(lambda x: opening.observe(x['width_m'], mapped_stamp(x, 'gripper')),
            anchor=state['pose'], stable=0., heartbeat=hold.check)


def recover_attempt(e, target, failure):
    if e.cancelled.is_set():
        raise PickRejected('用户已取消，禁止自动恢复或继续下一件')
    if e.sender is not None:
        raise PickRejected('旧发送器尚未退出，禁止开始恢复')
    e.owner = target.identity
    e.version += 1
    e._active(e.owner, e.version, target.epoch)
    failed_stage = e.stage
    e.stage = 'recovery_checking'
    e._emit('attempt_cancelled', {'error':str(failure), 'error_type':type(failure).__name__,
        'details':getattr(failure, 'details', {}), 'failed_stage':failed_stage})
    state = _recovery_feedback(e,epoch=target.epoch,failure=failure)  # original CAN/enable/driver checks must pass again
    if e.planner.target is not target:
        e.planner.install(target, state)
    carrying = _payload_state(e, state)
    if not carrying:
        e.holding.held_width = None
        e.planner.payload_local = e.planner.payload_tree = None
    # Include the initial position hold: it is exactly where a missed, closed
    # jaw can still overlap the cancelled task's predicted point cloud.
    enabled = e.config.raw.get('recovery_allow_workpiece_contact', False)
    with recovery_workpiece_contact(e.planner, target, enabled=enabled) as allowed:
        e._emit('recovery_workpiece_contact_policy', {
            'allow_cancelled_workpiece_contact': allowed,
            'target_identity': target.identity,
            'scope': 'cancelled_task_workpiece_only',
            'self_environment_and_belt_checks_retained': True})
        return _return_to_wait(e, target, failure, state, carrying,
            allow_workpiece_contact=allowed)


def _return_to_wait(e, target, failure, state, carrying, *, allow_workpiece_contact):
    from .execution import JointStreamSender, validate_start_state
    from .release import release_at_rest
    e.planner.identity = target.identity
    e.planner.version += 1  # invalidate every old result before any recovery send
    e.motion_started = True
    e.device.prepare_motion(e.config.settings.speed_percent,
        query_limits=e.config.settings.query_controller_limits,
        requested_velocity_limits=e.config.velocity_limits,
        requested_acceleration_limits=e.config.acceleration_limits)
    state = _recovery_feedback(e,epoch=target.epoch)
    state = _settle_last_target(e, state, epoch=target.epoch, carrying=carrying,
        sender_factory=JointStreamSender)
    e.planner.model.check_feedback(state)
    if carrying:
        e.planner.reanchor_support(state)
    prepared_taught = bool(carrying and e.config.raw.get('prepared_taught_cycle_enabled', False)
        and e.config.raw.get('taught_playback_enabled', True) and e.config.raw.get('taught_segments'))
    e._emit('recovery_started', {'carrying':carrying,
        'payload_policy':'taught_drop_then_wait' if prepared_taught else 'configured_bin_then_wait',
        'allow_cancelled_workpiece_contact': allow_workpiece_contact})
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='recovery-planner') as pool:
        stages = []
        if prepared_taught:
            state = e._prepared_taught_cycle(pool, state, target, recovery=True)
        elif carrying:
            lift = e.planner.model.tcp().copy()
            lift[:3, 3] += e.planner.support*e.config.settings.lift_m
            stages.extend([('recovery_lift', lift, None, True),
                ('recovery_transfer', e.config.bins[e.config.default_bin], None, True)])
        if not prepared_taught:
            stages.append(('recovery_return', None, e.config.standby, False))
        for stage, goal, joints, loaded in stages:
            e._active(e.owner, e.version, target.epoch)
            e.stage = 'planning_'+stage
            path = e._route(pool, state, stage=stage, goal=goal, joints=joints,
                carrying=loaded, epoch=target.epoch, version=e.version)
            # Check actual feedback at handoff; retain the certified command anchor.
            current = e.device.snapshot()
            fresh_feedback(current, e.clock(), e.feedback_stamps)
            e.planner.model.check_feedback(current)
            if np.max(abs(np.asarray(current['joints'])-path.q0)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('恢复路径规划期间实际起点改变')
            # Recheck the new route against this target's predicted motion at
            # its actual reservation, including an empty return after a miss.
            origin = e.clock()+max(.5, e.config.raw.get('planning_reserve_s', 1.5))
            with StationaryHold(e, JointStreamSender, epoch=target.epoch, carrying=loaded) as hold:
                e.planner.check_path(path, current['width_m'], origin=origin, payload=loaded,
                    cancelled=e.cancelled.is_set)
                hold.check()
                if e.clock() >= origin-e.config.raw['command_advance_s']:
                    raise PickRejected('恢复路径检查已错过发送起点')
            state = e._stream(path, origin=origin,
                epoch=target.epoch, carrying=loaded)
            if stage == 'recovery_transfer':
                e.stage = 'recovery_release'
                state = release_at_rest(e, state, epoch=target.epoch, sender_factory=JointStreamSender)
    state = _open_empty(e, state, epoch=target.epoch, sender_factory=JointStreamSender)
    validate_start_state(e.config, state, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
    e.stage = 'waiting'
    e._emit('recovery_finished', {'returned_to_wait':True, 'carried_to_bin':carrying})
    e.owner = e.planner.identity = None
    e.planner.version += 1
    return {'status':'attempt_cancelled_returned_to_wait', 'error':str(failure),
        'error_type':type(failure).__name__, 'failure_details':getattr(failure, 'details', {}),
        'returned_to_wait':True, 'carried_to_bin':carrying,
        'recovery_allow_workpiece_contact':allow_workpiece_contact,
        'physical_grasp_success':None, 'physical_bin_placement_verified':False}

"""Full remaining certificates and future references; no camera, SDK or CAN."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_policy_moving_approach import policy
from test_planning_worker_scene_validator import model,make_engine
from test_full_pick_execution import setup
from mujoco_conveyor_sorting.controllers.full_pick_replanning import version
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import PolicyStage


def running(policy,model,monkeypatch,*,mode='continuous'):
    result,obs=setup(policy,model,monkeypatch,approach_mode=mode)
    policy.config=replace(policy.config,max_translation_step=.022)
    policy._full_pick_act(obs())
    policy._full_pick_act(obs(10.1))
    return result,obs


def changed_observation(policy,obs,now=12.2,*,delta=.002,anchor_contact=False):
    data=obs(now);parent=policy._full_pick_bundle
    velocity=np.array(parent.request.continuation_velocity_m_s)+[delta,0.,0.]
    stamp=data['predictions']['part']['observed_timestamp']
    if anchor_contact:
        captured=parent.contact_pose.copy();captured[:3]+=velocity*(stamp-parent.contact_time)
        current=captured.copy();current[:3]+=velocity*(now-stamp)
        data['predictions']['part'].update(observed_pose=captured,current_ee_grasp_pose=current)
    data['belt_velocity']=velocity
    return data


def submit(policy,obs,now=12.2,**kwargs):
    data=changed_observation(policy,obs,now,**kwargs)
    parent=policy._full_pick_bundle
    policy._full_pick_act(data)
    request=policy._full_pick_update_request
    assert request is not None,policy._full_pick_info()
    assert policy._full_pick_bundle is parent
    return parent,request,data


def solve(model,request):
    result=make_engine(model).plan(request,lambda:False)
    assert result.success,result.diagnostics
    return result


def updated_data(obs,parent,request,now):
    data=obs(now)
    v=np.array(request.continuation_velocity_m_s)
    stamp=data['predictions']['part']['observed_timestamp']
    point=parent.contact_pose.copy()
    # Same NEW constant motion anchored at the exposure used by this request.
    point[:3]=request.target_pose[:3]+v*(stamp-request.planned_target_time_s)
    current=point.copy();current[:3]+=v*(now-stamp)
    data['predictions']['part'].update(observed_pose=point,current_ee_grasp_pose=current)
    data['belt_velocity']=v
    q,qd,_=policy_sample(parent,now)
    data.update(joint_pos=q,joint_vel=qd)
    return data


def policy_sample(bundle,now):return bundle.sample_full(now)


def test_running_full_receives_complete_future_plan_and_exact_common_prefix(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,data=submit(policy,obs)
    np.testing.assert_allclose((request.start_q,request.continuation_start_velocity,
        request.continuation_start_acceleration),parent.sample_full(request.continuation_start_time_s),atol=1e-12)
    result=solve(model,request)
    policy._planning_worker.results.append(result)
    now=data['timestamp']+.001
    policy._full_pick_act(updated_data(obs,parent,request,now))
    child=policy._full_pick_bundle
    assert child is not parent and child.prefix is parent
    assert child.closing is not None and child.lift is not None
    assert child.contact_time>child.start and child.contact_time>=parent.contact_time
    for t in np.linspace(parent.start,child.start-1e-6,30):
        np.testing.assert_array_equal(child.sample_full(t),parent.sample_full(t))
        np.testing.assert_array_equal(child.sample_command_full(t),parent.sample_command_full(t))
    np.testing.assert_allclose(child.sample_full(child.start),parent.sample_full(child.start),atol=1e-8,rtol=0.)
    # No arm sample was authorized by polling alone; original feedback/geometry
    # still run when the first prepared command crosses the installed splice.
    at=updated_data(obs,parent,request,child.start+.01)
    q,qd,_=child.sample_full(at['timestamp']);at.update(joint_pos=q,joint_vel=qd,
        ee_pose=policy.env.ik.forward_pose(q))
    assert policy._arm_on_full_pick(at) is not None
    assert policy._full_pick_abort_reason is None
    assert policy._full_pick_splice_crossed==version(child)
    assert policy._full_pick_replanning_metrics['accepted_count']==1
    assert policy._full_pick_info()['execution_replan_count']==1
    assert policy._full_pick_info()['execution_strategy']=='versioned_complete_pick_with_future_splice'
    assert policy._grasp_confirmed_at is None


@pytest.mark.parametrize('delta',[0.,1e-5,1e-4])
def test_small_position_and_velocity_noise_does_not_submit(policy,model,monkeypatch,delta):
    _,obs=running(policy,model,monkeypatch)
    for now in np.arange(10.2,12.,.05):
        policy._full_pick_act(changed_observation(policy,obs,float(now),delta=delta))
    assert len(policy._planning_worker.requests)==1
    assert policy._full_pick_replanning_metrics['trigger_count']==0


@pytest.mark.parametrize('failure',['late','failed','incomplete','seam','old_request','old_task'])
def test_bad_update_retains_valid_parent_without_closure(policy,model,monkeypatch,failure):
    _,obs=running(policy,model,monkeypatch)
    parent,request,data=submit(policy,obs)
    result=solve(model,request)
    now=data['timestamp']+.001
    if failure=='late':now=request.continuation_start_time_s+.001
    elif failure=='failed':result=replace(result,error='actual replan failed',path=None,full_pick_plan=None)
    elif failure=='incomplete':result=replace(result,continuation_fully_validated=False)
    elif failure=='seam':
        # Wrong initial derivative must not receive an install even if a fake
        # backend claims the remaining geometry is completely checked.
        wrong=replace(result.full_pick_plan.request,continuation_start_velocity=(.5,)*7)
        from mujoco_conveyor_sorting.controllers.full_pick_planning import FullPickPlan
        original=result.full_pick_plan
        class BadCurve:
            def __getattr__(self,name):return getattr(original.approach,name)
            def sample_full(self,t):
                q,v,a=original.approach.sample_full(t);return q,v+.5,a
        result=replace(result,full_pick_plan=replace(original,approach=BadCurve()))
    elif failure=='old_request':result=replace(result,request_id=request.request_id-1)
    elif failure=='old_task':
        from mujoco_conveyor_sorting.runtime.types import TargetIdentity
        result=replace(result,target_identity=TargetIdentity.confirmed(100.,1))
    policy._planning_worker.results.append(result)
    policy._full_pick_act(updated_data(obs,parent,request,now))
    assert policy._full_pick_bundle is parent and policy._full_pick_abort_reason is None
    assert policy._grasp_confirmed_at is None and policy._stage!=PolicyStage.CLOSING
    assert policy._full_pick_replanning_metrics['accepted_count']==0
    if failure in {'old_request','old_task'}:assert policy._full_pick_update_request is request


def test_failed_update_cannot_keep_invalid_parent_moving(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,data=submit(policy,obs)
    policy._planning_worker.results.append(replace(solve(model,request),path=None,error='blocked'))
    invalid=updated_data(obs,parent,request,12.21)
    invalid['predictions']['part']['observed_pose'][0]+=.02
    policy._full_pick_act(invalid)
    assert policy._full_pick_abort_reason=='predicted_contact_left_certificate_tolerance'
    assert policy.stage==PolicyStage.RECOVERING


def test_current_full_failure_with_explicit_identity_is_not_treated_as_stale(model):
    from test_full_pick_planning import full_request
    from mujoco_conveyor_sorting.runtime.types import TargetIdentity
    identity=TargetIdentity.confirmed(100.,1)
    request=replace(full_request(),target_identity=identity)
    result=make_engine(model,lambda *_:False).plan(request,lambda:False)
    assert not result.success and result.target_identity==identity

def test_stationary_synthetic_input_can_replan_to_valid_small_measured_speed(policy,model,monkeypatch):
    import test_full_pick_execution as helpers
    original=helpers.full_request
    monkeypatch.setattr(helpers,'full_request',lambda **kw:original(continuation_velocity_m_s=(0.,0.,0.),**kw))
    _,obs=running(policy,model,monkeypatch)
    parent,request,data=submit(policy,obs)
    result=solve(model,request)
    policy._planning_worker.results.append(result)
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    assert policy._full_pick_bundle.prefix is parent
    assert policy._full_pick_bundle.request.continuation_velocity_m_s[0]==pytest.approx(.002)
    assert policy._full_pick_bundle.contact_time>parent.contact_time


@pytest.mark.parametrize('speed',[.1,.2])
def test_synthetic_higher_speed_keeps_original_fixed_closure_dynamics_rejection(model,speed):
    from test_full_pick_planning import full_request
    result=make_engine(model).plan(full_request(full_approach_mode='continuous',
        continuation_velocity_m_s=(speed,0.,0.)),lambda:False)
    assert not result.success
    assert result.error=='fixed closure window violates joint dynamics'


@pytest.mark.parametrize('phase',['closing','lift'])
def test_worker_rejects_changed_complete_remaining_phase_not_only_approach(policy,model,monkeypatch,phase):
    _,obs=running(policy,model,monkeypatch)
    parent,request,data=submit(policy,obs)
    result=make_engine(model,lambda req,q:req.full_phase!=phase).plan(request,lambda:False)
    assert not result.success
    policy._planning_worker.results.append(result)
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    assert policy._full_pick_bundle is parent and policy._full_pick_abort_reason is None


def test_actual_worker_wait_does_not_block_certified_parent_samples(policy,model,monkeypatch):
    from threading import Event
    from mujoco_conveyor_sorting.controllers.planning_worker import AsyncPlanningWorker
    entered,release=Event(),Event()
    engine=make_engine(model)
    class Engine:
        def plan(self,request,cancelled):
            entered.set();assert release.wait(3.)
            return engine.plan(request,cancelled)
    _,obs=running(policy,model,monkeypatch)
    worker=AsyncPlanningWorker(Engine());policy._planning_worker=worker
    try:
        data=changed_observation(policy,obs)
        parent=policy._full_pick_bundle
        policy._full_pick_act(data)
        assert entered.wait(3.) and worker.busy
        request=policy._full_pick_update_request
        # The real worker is deliberately blocked on the barrier. These are
        # real production policy/arm callbacks, not queued old setpoints.
        for now in [12.201,12.202,12.203]:
            live=updated_data(obs,parent,request,now)
            policy._full_pick_act(live)
            np.testing.assert_array_equal(policy._arm_on_full_pick(live),parent.sample_full(now)[0])
            assert policy._full_pick_abort_reason is None and worker.busy
        release.set()
        import time
        limit=time.monotonic()+3.
        while policy._full_pick_bundle is parent and time.monotonic()<limit:
            policy._full_pick_act(updated_data(obs,parent,request,12.204));time.sleep(.001)
        assert policy._full_pick_bundle.prefix is parent
    finally:release.set();worker.close()


@pytest.mark.parametrize('sent',[False,True])
def test_closing_stage_or_jaw_send_retires_pending_and_never_restarts_contact(policy,model,monkeypatch,sent):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    result=solve(model,request)
    if sent:policy.full_pick_closure_sent(12.21)
    else:
        policy._stage=PolicyStage.CLOSING
        policy._maintain_full_replanning(updated_data(obs,parent,request,12.21),12.21,
            policy._full_pick_observation(updated_data(obs,parent,request,12.21),12.21)[0])
    assert policy._full_pick_update_request is None and policy._full_pick_pending_bundle is None
    policy._planning_worker.results.append(result)
    assert policy._full_pick_bundle is parent
    assert len(policy._planning_worker.requests)==2


@pytest.mark.parametrize('obstacle',['old','new'])
def test_scene_monitor_distinguishes_active_obstacle_and_candidate_only_obstacle(policy,model,monkeypatch,obstacle):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    status={'failed':False,'validated_capture_monotonic_s':12.201,
        'checked_plan_identities':[id(parent)],'pending_failure_reason':None}
    policy.env._policy_clock_origin=0.
    policy.env.full_plan_scene_monitor=SimpleNamespace(snapshot=lambda:dict(status))
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    candidate=policy._full_pick_pending_bundle
    assert candidate is not None and policy._full_pick_bundle is parent
    status['pending_plan_identity']=id(candidate)
    if obstacle=='new':status['pending_failure_reason']='new_obstacle_on_remaining_trajectory'
    else:status.update(failed=True,failure_reason='new_obstacle_on_remaining_trajectory')
    policy._full_pick_act(updated_data(obs,parent,request,12.202))
    assert policy._full_pick_bundle is parent
    if obstacle=='old':assert policy.stage==PolicyStage.RECOVERING
    else:assert policy._full_pick_abort_reason is None and policy._full_pick_pending_bundle is None


def test_waiting_for_new_scene_review_never_installs_after_splice(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy.env._policy_clock_origin=0.
    policy.env.full_plan_scene_monitor=SimpleNamespace(snapshot=lambda:{'failed':False,
        'validated_capture_monotonic_s':12.201,'checked_plan_identities':[id(parent)]})
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    assert policy._full_pick_pending_bundle is not None
    now=request.continuation_start_time_s+.001
    policy._full_pick_act(updated_data(obs,parent,request,now))
    assert policy._full_pick_bundle is parent and policy._full_pick_pending_bundle is None
    assert policy._full_pick_replanning_metrics['missed_splice_count']==1


@pytest.mark.parametrize('kind',['tracking','task','clock','scene'])
def test_joined_math_does_not_authorize_bad_live_feedback_or_changed_owner(policy,model,monkeypatch,kind):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    child=policy._full_pick_bundle
    now=child.start+.01;data=updated_data(obs,parent,request,now)
    q,v,_=child.sample_full(now);data.update(joint_pos=q,joint_vel=v,ee_pose=policy.env.ik.forward_pose(q))
    if kind=='tracking':data['joint_pos']=q+np.array([.1,0,0,0,0,0,0]);data['ee_pose']=policy.env.ik.forward_pose(data['joint_pos'])
    elif kind=='task':
        from mujoco_conveyor_sorting.runtime.types import TargetIdentity
        policy.env.target_identity=TargetIdentity.confirmed(100.,1)
    elif kind=='scene':data['scene_epoch']=99
    else:
        policy._full_pick_bundle=replace(child,request=replace(child.request,full_observation_clock_epoch=1))
        data['predictions']['part']['clock_epoch']=2
    if kind in {'scene','clock','task'}:
        assert policy.full_pick_dispatch_reason(data,q,now) is not None
    else:
        policy._arm_on_full_pick(data)
        assert policy._full_pick_abort_reason=='full_pick_feedback_or_live_geometry_invalid'

@pytest.mark.parametrize('delta',[.006,-.006])
def test_multiple_accelerating_or_decelerating_updates_preserve_versioned_prefix(policy,model,monkeypatch,delta):
    _,obs=running(policy,model,monkeypatch)
    original,first,_=submit(policy,obs)
    first_result=solve(model,first)
    policy._planning_worker.results.append(first_result)
    policy._full_pick_act(updated_data(obs,original,first,12.201))
    parent=policy._full_pick_bundle
    now=parent.start+.02
    # Re-estimation changes velocity but preserves the prior contact point;
    # old close-window deviation remains below its original 4 mm envelope.
    data=updated_data(obs,original,first,now)
    v=np.array(parent.request.continuation_velocity_m_s)+[delta,0.,0.]
    stamp=data['predictions']['part']['observed_timestamp']
    captured=parent.contact_pose.copy();captured[:3]+=v*(stamp-parent.contact_time)
    current=captured.copy();current[:3]+=v*(now-stamp)
    data['predictions']['part'].update(observed_pose=captured,current_ee_grasp_pose=current)
    data['belt_velocity']=v
    policy._full_pick_act(data)
    second=policy._full_pick_update_request
    assert second is not None and second.full_parent_version==version(parent)
    policy._planning_worker.results[:]=[first_result,solve(model,second)]
    policy._full_pick_act(data)
    assert policy._full_pick_bundle is parent and policy._full_pick_update_request is second
    policy._full_pick_act(data)
    newest=policy._full_pick_bundle
    assert newest.prefix is parent and parent.prefix is original
    assert newest.request.request_id!=parent.request.request_id
    for t in [original.start,parent.start-.001,parent.start+.001,newest.start-.001]:
        np.testing.assert_array_equal(newest.sample_full(t),parent.sample_full(t))
    assert policy._full_pick_replanning_metrics['accepted_count']==2
    assert policy._full_pick_replanning_metrics['stale_result_reject_count']==1
    assert policy._full_pick_jaw_committed is None


def test_high_approach_splice_retains_nonzero_initial_derivatives(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch,mode='high')
    parent,request,_=submit(policy,obs,now=10.2,anchor_contact=True)
    result=solve(model,request)
    np.testing.assert_allclose(result.full_pick_plan.sample_full(request.continuation_start_time_s),
        parent.sample_full(request.continuation_start_time_s),atol=1e-8,rtol=0.)
    assert np.max(np.abs(request.continuation_start_velocity))>0
    assert result.full_pick_plan.descent is not None and result.full_pick_plan.lift is not None


def test_installed_full_certificate_finishes_grasp_and_lift_without_another_jaw_timeline(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    child=policy._full_pick_bundle
    for now in np.unique(np.r_[np.arange(child.start,child.end,.01),child.contact_time,child.lift_time,child.end]):
        data=updated_data(obs,parent,request,float(now))
        q,v,_=child.sample_full(now)
        data.update(joint_pos=q,joint_vel=v,ee_pose=policy.env.ik.forward_pose(q))
        policy._full_pick_act(data)
        assert policy._full_pick_abort_reason is None,policy._full_pick_info()
        if policy.stage==PolicyStage.CLOSING and policy._full_pick_jaw_committed is None:
            policy.full_pick_closure_sent(float(now))
        policy.full_pick_command_sent(float(now))
    assert policy._full_pick_finished and policy._full_pick_lift_admitted
    assert policy._full_pick_info()['complete_pick_schedule_completed']
    assert not policy._full_pick_info()['immutable_full_schedule_completed']
    assert policy._full_pick_jaw_committed is not None
    names=[e['event'] for e in policy._full_pick_events]
    assert names.count('full_replan_first_SDK_dispatch')==1
    assert names.count('grasp_result')==1
    assert len(policy._planning_worker.requests)==2


def test_actual_dispatch_clock_rejects_late_result_even_with_old_observation(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy.env.execution_trace=SimpleNamespace(mode='hardware',policy_origin=100.,
        clock=lambda:100.+request.continuation_start_time_s+.001)
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    assert policy._full_pick_bundle is parent and policy._full_pick_abort_reason is None
    assert policy._full_pick_replanning_metrics['missed_splice_count']==1


@pytest.mark.parametrize('restore',[False,True])
def test_final_reservation_crossing_splice_restores_parent_or_holds(policy,model,monkeypatch,restore):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    actual=[12.201];calls=[]
    policy.env.execution_trace=SimpleNamespace(mode='hardware',policy_origin=100.,clock=lambda:100.+actual[0])
    def reserve(*args,**kw):
        calls.append(kw)
        actual[0]=request.continuation_start_time_s+.001
        return {'accepted':len(calls)==1 or restore,'predicted_ee_grasp_pose':kw['requested_pose']}
    policy.env.refine_selected_grasp_plan=reserve
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    assert policy._full_pick_bundle is parent
    assert len(calls)==2 and calls[1]['requested_time']==parent.contact_time
    assert policy._full_pick_replanning_metrics['accepted_count']==0
    assert policy._full_pick_abort_reason==(None if restore else 'full_replan_old_reservation_no_longer_valid')


def test_splice_budget_includes_measured_scene_review_and_existing_cadence(policy,model,monkeypatch):
    from mujoco_conveyor_sorting.runtime.full_plan_scene_monitor import FullPlanSceneMonitor
    _,obs=running(policy,model,monkeypatch)
    parent=policy._full_pick_bundle
    policy.env._policy_clock_origin=0.
    policy.env.full_plan_scene_monitor=SimpleNamespace(period=FullPlanSceneMonitor.period,
        snapshot=lambda:{'failed':False,'checked_plan_identity':id(parent),
            'validated_capture_monotonic_s':12.17,'last_compute_end_monotonic_s':12.2})
    _,request,_=submit(policy,obs)
    event=[e for e in policy._full_pick_events if e['event']=='full_replan_requested'][-1]
    assert event['scene_review_forecast_s']==pytest.approx(.23)
    assert request.continuation_start_time_s==pytest.approx(12.2+event['compute_forecast_s']+.23+policy._execution_dt(obs(12.2)))


def test_pending_scene_review_can_authorize_only_exact_current_candidate(policy,model,monkeypatch):
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    status={'failed':False,'validated_capture_monotonic_s':12.201,'checked_plan_identities':[id(parent)]}
    policy.env._policy_clock_origin=0.
    policy.env.full_plan_scene_monitor=SimpleNamespace(snapshot=lambda:dict(status))
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    candidate=policy._full_pick_pending_bundle
    assert candidate is not None and policy._full_pick_bundle is parent
    status.update(checked_plan_identities=[id(parent),id(candidate)],pending_plan_identity=id(candidate))
    policy._full_pick_act(updated_data(obs,parent,request,12.202))
    assert policy._full_pick_bundle is candidate and candidate.prefix is parent


@pytest.mark.parametrize('failure',['normal','delay','missing','backward','mapping'])
def test_joined_plan_still_requires_original_hardware_snapshot_quality(policy,model,monkeypatch,failure):
    from test_nero_pick_hardware_snapshot import asynchronous_adapter
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import FeedbackError
    from mujoco_conveyor_sorting.runtime.feedback_clock import FeedbackTime
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    child=policy._full_pick_bundle;now=child.start+.01
    adapter=asynchronous_adapter(monkeypatch)
    adapter._feedback_clock=object()  # Enable the production mapped schema; no SDK construction.
    from mujoco_conveyor_sorting.runtime import nero_pick_hardware as module
    monkeypatch.setattr(module.time,'time',lambda:1000.+now)
    monkeypatch.setattr(module.time,'monotonic',lambda:100.+now)
    def decorate(message):
        message.timestamp=1000.+now
        message.feedback_time=FeedbackTime(message.timestamp,100.+now,100.+now,0,'mapped',1000.+now,100.+now,0.)
        return message
    for part in vars(adapter.robot._parser).values():decorate(part)
    for owner,names in [(adapter.robot,['get_joint_angles','get_flange_pose','get_arm_status','get_driver_states']),
                        (adapter.gripper,['get_gripper_status'])]:
        for name in names:
            original=getattr(owner,name)
            setattr(owner,name,lambda *a,_original=original:decorate(_original(*a)))
    if failure=='missing':adapter.robot._parser.joint_7=None
    elif failure!='normal':
        part=adapter.robot._parser.joint_7
        part.feedback_time=replace(part.feedback_time,
            mapped_monotonic=(100.+now-.201 if failure=='delay' else None),
            mapping_quality=('mapped' if failure=='delay' else 'out_of_order' if failure=='backward' else 'ambiguous_epoch'))
    if failure!='normal':
        with pytest.raises(FeedbackError):adapter.snapshot(wait_timeout_s=0.)
        assert policy._full_pick_splice_crossed is None
    else:
        state=adapter.snapshot(wait_timeout_s=0.)
        assert state['joint_stamp_monotonic']==pytest.approx(100.+now)
        q,v,_=child.sample_full(now)
        data=updated_data(obs,parent,request,now)
        data.update(joint_pos=q,joint_vel=v,ee_pose=policy.env.ik.forward_pose(q),feedback_time=state['feedback_time'])
        assert policy._arm_on_full_pick(data) is not None and policy._full_pick_abort_reason is None


def test_trace_aligns_feedback_to_each_packet_time_across_future_splice(policy,model,monkeypatch):
    from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
    _,obs=running(policy,model,monkeypatch)
    parent,request,_=submit(policy,obs)
    policy._planning_worker.results.append(solve(model,request))
    policy._full_pick_act(updated_data(obs,parent,request,12.201))
    child=policy._full_pick_bundle
    trace=ExecutionTrace(clock=lambda:100.+child.start+.02,policy_origin=100.)
    packets=100.+child.start+np.array([-.01,-.005,.005,.01])
    times=packets[[0,0,1,1,2,2,3]]-100.
    reference=np.array([child.sample_command_full(t)[0][j] for j,t in enumerate(times)])
    error=np.arange(1,8)*.0001
    state={'joints':reference+error,'joint_stamp':1000.+times.min(),
        'joint_stamps_monotonic':packets.tolist(),'feedback_time':{'joint_7':{'mapping_quality':'mapped'}}}
    trace.record_feedback(state,100.+child.start+.01,100.+child.start+.02)
    trace.prepare_command(policy,policy_time_s=child.start+.02,candidate=child.sample_full(child.start+.02)[0],stage='tracking_descent')
    arrays=trace.arrays()
    np.testing.assert_allclose(arrays['feedback_packet_aligned_error_rad'][0],error,atol=1e-12)
    np.testing.assert_allclose(arrays['feedback_reference_q_rad'][0],reference,atol=1e-12)
    assert trace.summary()['feedback_tracking']['valid_samples_per_axis']==[1]*7
    state['feedback_time']['joint_7']['mapping_quality']='ambiguous_epoch'
    trace.record_feedback(state,100.+child.start+.01,100.+child.start+.02)
    trace.prepare_command(policy,policy_time_s=child.start+.02,candidate=np.zeros(7),stage='tracking_descent')
    assert np.isnan(trace.arrays()['feedback_packet_aligned_error_rad'][1]).all()


@pytest.mark.parametrize('obstacle',['old','new','none'])
def test_real_scene_process_reviews_both_certificates_without_turning_candidate_failure_into_active_failure(monkeypatch,obstacle):
    from test_full_plan_scene_monitor import frame_monitor,bundle
    monitor,frame=frame_monitor(monkeypatch)
    old,new=bundle(),bundle()
    monitor.context=SimpleNamespace(policy=SimpleNamespace(_full_pick_bundle=old,_full_pick_pending_bundle=new,
        _full_pick_lift_admitted=False),_motion_snapshot=None,_template_local=np.empty((0,3)),
        _policy_clock_origin=99.)
    calls=[]
    def check(points,plan,t,**kw):
        calls.append((plan,t))
        return 'new_obstacle_on_remaining_trajectory' if (plan is old and obstacle=='old') or (plan is new and obstacle=='new') else None
    monitor.check_points=check
    monitor.process_frame(frame)
    status=monitor.snapshot()
    assert [c[0] for c in calls]==[old,new] and all(c[1]==1. for c in calls)
    assert status['failed']==(obstacle=='old')
    assert (id(new) in status['checked_plan_identities'])==(obstacle!='new')
    assert status['pending_plan_identity']==id(new)


@pytest.mark.parametrize('fault',['normal','joint','jaw'])
def test_runner_records_commit_only_after_successful_original_sdk_call(fault):
    from test_full_policy_runner import Clock,Context,Device,action,run
    clock=Clock();sent=[];closed=[]
    error=RuntimeError('injected SDK send fault')
    class Fake(Device):
        def move_joints(self,q):
            if fault=='joint':raise error
            super().move_joints(q)
        def grip(self,width,force):
            if fault=='jaw' and width<.099:raise error
            super().grip(width,force)
    device=Fake(clock);context=Context([action(gripper=-1.)])
    def arm_hook(t):
        assert device.calls[-1][0]=='move';sent.append(t)
    def jaw_hook(t):
        assert device.calls[-1][0]=='grip';closed.append(t)
    context.policy.full_pick_command_sent=arm_hook
    context.policy.full_pick_closure_sent=jaw_hook
    if fault=='normal':
        run(context,device,clock)
        assert sent and len(closed)==1
    else:
        with pytest.raises(RuntimeError) as caught:run(context,device,clock)
        assert caught.value is error and not closed
        if fault=='joint':assert not sent
        else:assert sent
        assert any(c[0]=='stop' for c in device.calls)

"""Full schedule ownership and fail-closed events; no physical success claim."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_policy_moving_approach import policy
from test_planning_worker_scene_validator import model, make_engine
from test_full_pick_planning import full_request
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import PolicyStage


def setup(policy,model,monkeypatch,*,approach_mode='high',contact_prefix=False):
    policy.config=replace(policy.config,planning_mode='full_preplanned',selection_policy='first_safe',
        replan_policy='event_triggered',max_translation_step=.05,minimum_downward_alignment=-1.,
        pregrasp_timeout=50.,attempt_timeout=60.,descent_timeout=30.,lift_timeout=30.,
        require_grasp_confirmation=True,moving_approach_mode=approach_mode)
    request=replace(full_request(full_contact_handoff_enabled=contact_prefix),generation=policy._planning_generation,scene_epoch=7,
                    full_approach_mode=approach_mode)
    engine=make_engine(model); result=engine.plan(request,lambda:False)
    assert result.success,result.error
    policy._safe_joint_path=[]
    policy._full_pick_request=request; policy._full_pick_hold_q=request.start_q.copy()
    policy._planning_worker.requests.append(request); policy._planning_worker.results.append(result)
    policy._full_pick_result=None; policy._stage_started_at=0.
    policy.env.ik=engine.ik
    monkeypatch.setattr(policy,'_global_edge_is_safe',lambda *_a,**_k:True)
    monkeypatch.setattr(policy,'_window_valid',lambda *_:True)
    monkeypatch.setattr(policy,'_update_grasp_force_preload',lambda *_:True)
    monkeypatch.setattr(policy,'_grasp_feedback',lambda _target,obs:obs['timestamp']>=13.05)
    monkeypatch.setattr(policy,'_bilateral_finger_contact',lambda *_:True)
    # Test only dispatch ownership here; original live closure predicate has
    # separate coverage and remains called through _admit_descent_closure.
    monkeypatch.setattr(policy,'_descent_closure_qualification',lambda _t,_o,now,_p:(now>=13.,None))
    target={'object_id':'part','confidence':1.,'grasp_window':{'end_time':100.}}
    monkeypatch.setattr(policy,'_find_target',lambda *_:target)
    policy._target=target
    def observation(now=9.,offset=0.,age=.02):
        bundle=result.full_pick_plan
        q,v,_=bundle.sample_full(max(bundle.start,now))
        pose=bundle.contact_pose.copy(); pose[:3]+=np.array(request.continuation_velocity_m_s)*(now-bundle.contact_time)
        pose[0]+=offset
        captured=pose.copy(); captured[:3]-=np.array(request.continuation_velocity_m_s)*age
        obs={'joint_pos':q,'joint_vel':v,'ee_pose':engine.ik.forward_pose(q),
             'timestamp':now,'scene_epoch':7,'control_dt':.01,
             'predictions':{'part':{'valid':True,'visible':True,'prediction_confidence':1.,
                'observed_timestamp':now-age,'observed_pose':captured,
                'current_ee_grasp_pose':pose}}, 'belt_velocity':np.array(request.continuation_velocity_m_s)}
        return obs
    return result,observation


def test_contact_prefix_handoff_is_not_grasp_confirmation_or_automatic_closure(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch,approach_mode='continuous',contact_prefix=True)
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    data=obs(result.full_pick_plan.contact_time-.005)
    data['dispatch_clock']={'scheduled_dispatch_policy_time_s':result.full_pick_plan.contact_time+.01}
    assert policy._full_pick_act(data) is None
    assert policy._full_pick_contact_handed_off
    assert policy._grasp_confirmed_at is None and not policy._full_pick_lift_admitted
    assert not policy._full_pick_finished
    assert policy.stage==PolicyStage.TRACKING_DESCENT
    assert result.full_pick_plan.closing is None and result.full_pick_plan.lift is None


@pytest.mark.parametrize('qualified',[False,True])
def test_prefix_closure_uses_original_measured_gate_before_handing_off(policy,model,monkeypatch,qualified):
    result,obs=setup(policy,model,monkeypatch,approach_mode='continuous',contact_prefix=True)
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    now=result.full_pick_plan.contact_time-.17
    monkeypatch.setattr(policy,'_descent_closure_qualification',lambda *_:(qualified,None))
    data=obs(now)
    action=policy._full_pick_act(data)
    assert policy._full_pick_contact_handed_off is qualified
    assert policy._grasp_confirmed_at is None and not policy._full_pick_lift_admitted
    assert not policy._full_pick_finished
    if qualified:
        assert action is None and policy.stage==PolicyStage.CLOSING
        assert policy._full_pick_bundle is result.full_pick_plan
        assert policy._arm_on_full_pick(data) is None
        assert policy._full_pick_events[-1]['meaning']=='original_measured_closure_gate_admitted_shared_closing'
    else:
        assert action is not None and policy.stage==PolicyStage.TRACKING_DESCENT


def test_changed_prediction_retires_only_unexecuted_prefix_and_keeps_fixed_hold(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch,approach_mode='continuous',contact_prefix=True)
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    hold=policy._full_pick_hold_q.copy()
    old_count=len(policy._planning_worker.requests)
    action=policy._full_pick_act(obs(9.1,offset=.005))
    assert policy._full_pick_abort_reason is None
    assert policy._full_pick_bundle is None and policy._full_pick_request is None
    assert policy._full_pick_pre_motion_replans==1
    np.testing.assert_array_equal(action['arm'],hold)
    policy._full_pick_last_planning_wall_s=.4
    policy._full_pick_act(obs(9.2,offset=.005))
    assert len(policy._planning_worker.requests)==old_count+1
    new=policy._planning_worker.requests[-1]
    previous=result.full_pick_plan.request
    assert (new.generation,new.request_id)!=(previous.generation,previous.request_id)
    assert new.full_plan_compute_budget_s==pytest.approx(.4*1.05+.08)
    assert new.intercept_request_time_s==pytest.approx(9.2)
    np.testing.assert_array_equal(policy._full_pick_hold_q,hold)
    assert policy._grasp_confirmed_at is None


def test_contact_prefix_prediction_change_after_motion_still_aborts(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch,approach_mode='continuous',contact_prefix=True)
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    policy._full_pick_act(obs(10.01))
    policy._full_pick_act(obs(10.02,offset=.005))
    assert policy._full_pick_abort_reason=='predicted_contact_left_certificate_tolerance'
    assert policy._full_pick_pre_motion_replans==0


def test_contact_prefix_cannot_refresh_after_dispatch_reaches_start(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch,approach_mode='continuous',contact_prefix=True)
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    data=obs(9.9,offset=.005)
    data['dispatch_clock']={'scheduled_dispatch_policy_time_s':result.full_pick_plan.start}
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason=='predicted_contact_left_certificate_tolerance'
    assert policy._full_pick_pre_motion_replans==0


def test_high_passage_and_complete_lift_never_request_another_plan(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    assert policy._full_pick_act(obs()) is not None
    bundle=result.full_pick_plan
    for now in np.unique(np.r_[np.arange(9.9,bundle.end,.01),bundle.high_time,
                                bundle.contact_time,bundle.lift_time,bundle.end]):
        policy._full_pick_act(obs(float(now)))
        assert policy._full_pick_abort_reason is None,policy._full_pick_info()
    assert len(policy._planning_worker.requests)==1
    names=[e['event'] for e in policy._full_pick_events]
    assert names.count('high_waypoint_passed')==names.count('descent_started')==1
    assert 'replan_triggered' not in names
    assert policy._full_pick_lift_admitted and policy._full_pick_finished

def test_opt_in_contact_handoff_retires_schedule_without_claiming_force_or_lift(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch,approach_mode='continuous')
    policy.config=replace(policy.config,full_plan_contact_handoff_enabled=True)
    policy._full_pick_act(obs())
    policy._full_pick_act(obs(13.))
    assert not policy._full_pick_contact_handed_off
    assert policy._full_pick_act(obs(13.06)) is None
    assert policy._full_pick_contact_handed_off
    assert not policy._full_pick_finished and not policy._full_pick_lift_admitted
    assert policy._grasp_confirmed_at is None
    assert policy._full_pick_bundle is result.full_pick_plan
    assert policy._arm_on_full_pick(obs(13.07)) is None
    assert policy.full_pick_dispatch_reason(obs(13.07),np.zeros(7),13.07) is None
    assert any(e['event']=='contact_handoff_to_shared_feedback_control' for e in policy._full_pick_events)

def test_contact_handoff_keeps_feedback_translation_envelope(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch,approach_mode='continuous')
    policy._full_pick_contact_handed_off=True
    policy.config=replace(policy.config,max_translation_step=.022)
    data=obs(13.06);q=data['joint_pos'].copy()
    np.testing.assert_array_equal(policy._guard_contact_handoff_command(data,q),q)
    changed=q.copy();changed[0]+=.023
    policy._guard_contact_handoff_command(data,changed)
    assert policy._full_pick_abort_reason=='contact_handoff_feedback_envelope_invalid'

def test_handoff_speed_hint_only_slows_loaded_local_steps(policy):
    policy.config=replace(policy.config,local_tcp_linear_speed_limit_enabled=True,cartesian_linear_speed=.28)
    observation={'control_dt':.02}
    policy._full_pick_contact_handed_off=True
    policy._full_pick_handoff_lift_speed=.12
    policy._stage=PolicyStage.CLOSING
    assert policy._local_translation_step_limit(observation)==pytest.approx(.0056)
    policy._stage=PolicyStage.LIFTING
    assert policy._local_translation_step_limit(observation)==pytest.approx(.0024)
    policy._full_pick_contact_handed_off=False
    assert policy._local_translation_step_limit(observation)==pytest.approx(.0056)


def test_only_blocked_handed_off_lift_proposes_original_downstream_corridor(policy,monkeypatch):
    pose=np.array([.16,.08,.68,0.,1.,0.,0.])
    policy._stage=PolicyStage.LIFTING
    policy._full_pick_contact_handed_off=True
    policy._full_pick_bundle=SimpleNamespace(request=SimpleNamespace(continuation_velocity_m_s=np.array([.2,0.,0.])))
    policy._lift_clearance_global_replan_active=True
    monkeypatch.setattr(policy,'_clearing_payload',lambda:True)
    monkeypatch.setattr(policy,'_belt_frame',lambda:(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.])))
    monkeypatch.setattr(policy,'_belt_coordinate',lambda p:float(p[0]))
    monkeypatch.setattr(policy,'_belt_bounds',lambda *a,**k:(-.15,.25))
    choices=policy._async_target_poses(pose)
    assert len(choices)==2
    np.testing.assert_array_equal(choices[1],pose)
    np.testing.assert_allclose(choices[0],pose+np.array([.08,0.,0.,0.,0.,0.,0.]))
    policy._lift_clearance_global_replan_active=False
    assert len(policy._async_target_poses(pose))==1

def test_loaded_handoff_progress_uses_remaining_feedback_envelope(policy,monkeypatch):
    policy.config=replace(policy.config,local_tcp_linear_speed_limit_enabled=True,
        cartesian_linear_speed=.28,max_translation_step=.022)
    policy._full_pick_contact_handed_off=True
    policy._full_pick_handoff_lift_speed=.12
    policy._stage=PolicyStage.LIFTING
    measured=np.array([0.,0.,.5,1.,0.,0.,0.])
    previous=measured.copy();previous[2]+=.020
    target=previous.copy();target[2]+=.1
    monkeypatch.setattr(policy,'_cartesian_servo_reference',lambda *a,**k:previous.copy())
    monkeypatch.setattr(policy,'_ee_pose',lambda *a:measured.copy())
    monkeypatch.setattr(policy,'_clearing_payload',lambda:True)
    result=policy._cartesian_servo_pose(target,{'control_dt':.02})
    assert result[2]-previous[2]==pytest.approx(.001)
    assert np.linalg.norm(result[:3]-measured[:3])<.022


def test_loaded_global_edge_worker_executor_share_slower_time_law(policy,model):
    engine=make_engine(model)
    request=full_request(tcp_speed_limits_enabled=False,maximum_tcp_linear_speed=.08)
    start=request.start_q.copy();goal=start.copy();goal[0]+=.15
    policy.env.ik=engine.ik
    policy.config=replace(policy.config,tcp_speed_limits_enabled=False,
        trajectory_minimum_duration=request.trajectory_minimum_duration)
    policy._stage=PolicyStage.LIFTING
    policy._full_pick_contact_handed_off=True
    policy._full_pick_handoff_lift_speed=.08
    worker=engine._edge_minimum_duration(request,start,goal)
    executor=policy._edge_minimum_duration(start,goal,global_path=True,stage=PolicyStage.LIFTING)
    assert worker==pytest.approx(executor)
    old=engine._edge_minimum_duration(replace(request,maximum_tcp_linear_speed=None),start,goal)
    assert worker>old
    curve=policy._build_global_edge_trajectory(start,goal,request.trajectory_velocity_limits,
        request.trajectory_acceleration_limits,stage=PolicyStage.LIFTING)
    assert curve.duration>=worker
    for t in np.linspace(0.,curve.duration,65):
        q,v,_=curve.sample_full(t)
        assert np.linalg.norm(engine.ik.compute_jacobian(q)[:3]@v)<=.08+1e-10
    policy._full_pick_contact_handed_off=False
    assert policy._edge_minimum_duration(start,goal,global_path=True)==old


def test_loaded_pure_rotation_worker_executor_use_same_response_time_law(policy,model):
    engine=make_engine(model)
    request=full_request(tcp_speed_limits_enabled=True,maximum_tcp_angular_speed=.5)
    start=request.start_q.copy();goal=start.copy();goal[5]+=.8
    policy.env.ik=engine.ik
    policy.config=replace(policy.config,tcp_speed_limits_enabled=False,
        trajectory_minimum_duration=request.trajectory_minimum_duration)
    policy._stage=PolicyStage.LIFTING
    policy._full_pick_contact_handed_off=True
    policy._full_pick_handoff_angular_speed=.5
    worker=engine._edge_minimum_duration(request,start,goal)
    executor=policy._edge_minimum_duration(start,goal,global_path=True,stage=PolicyStage.LIFTING)
    assert worker==pytest.approx(executor)
    old=engine._edge_minimum_duration(replace(request,tcp_speed_limits_enabled=False),start,goal)
    assert worker>old
    curve=policy._build_global_edge_trajectory(start,goal,request.trajectory_velocity_limits,
        request.trajectory_acceleration_limits,stage=PolicyStage.LIFTING)
    for t in np.linspace(0.,curve.duration,65):
        q,v,_=curve.sample_full(t)
        assert np.linalg.norm(engine.ik.compute_jacobian(q)[3:]@v)<=.5+1e-10
    policy._full_pick_contact_handed_off=False
    assert policy._edge_minimum_duration(start,goal,global_path=True)==old


def test_small_noise_keeps_schedule_large_error_aborts_with_fixed_hold(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    for now in np.arange(10.,11.,.02):
        policy._full_pick_act(obs(float(now),offset=.002))
    assert len(policy._planning_worker.requests)==1 and policy._full_pick_abort_reason is None
    action=policy._full_pick_act(obs(11.,offset=.005))
    assert policy._full_pick_abort_reason=='predicted_contact_left_certificate_tolerance'
    anchor=action['arm'].copy()
    changed=obs(11.1,offset=.01);changed['joint_pos']+=.03
    np.testing.assert_array_equal(policy._full_pick_abort(changed,'again')['arm'],anchor)
    assert policy.stage==PolicyStage.RECOVERING


@pytest.mark.parametrize('reason,age,valid,visible',[
    ('target_observation_expired',2.001,True,True),
    ('target_observation_expired',2.001,True,False),
    ('target_observation_invalid',.02,False,True),
    ('target_observation_invalid',.02,False,False)])
def test_invalid_prediction_or_old_exposure_cannot_run_schedule(policy,model,monkeypatch,reason,age,valid,visible):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    data=obs(10.1,age=age)
    data['predictions']['part'].update(valid=valid,visible=visible)
    q=policy._full_pick_bundle.sample_full(data['timestamp'])[0]
    assert policy.full_pick_dispatch_reason(data,q,data['timestamp'])==reason
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason==reason


def test_coasting_can_submit_one_plan_without_restamping_capture(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    request=policy._full_pick_request
    policy._full_pick_request=None
    policy._planning_worker.requests.clear();policy._planning_worker.results.clear()
    monkeypatch.setattr(policy,'_make_planning_request',lambda *a,**k:request)
    data=obs(9.,age=.3187683751239092)
    prediction=data['predictions']['part'];prediction['visible']=False
    stamp=prediction['observed_timestamp'];captured=prediction['observed_pose'].copy()
    velocity=data['belt_velocity'].copy()
    assert policy._full_pick_act(data) is not None
    assert policy._full_pick_abort_reason is None
    assert len(policy._planning_worker.requests)==1
    submitted=policy._planning_worker.requests[0]
    expected=captured.copy()
    expected[:3]+=velocity*(submitted.planned_target_time_s-stamp)
    np.testing.assert_allclose(submitted.target_pose,expected)
    assert policy._full_pick_events[0]['observed_capture_s']==stamp
    assert prediction['visible'] is False
    assert prediction['observed_timestamp']==stamp
    np.testing.assert_array_equal(prediction['observed_pose'],captured)
    np.testing.assert_array_equal(data['belt_velocity'],velocity)
    # Repeated decisions while the worker is pending keep the same request.
    data['timestamp']+=.01
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason is None
    assert len(policy._planning_worker.requests)==1


@pytest.mark.parametrize('age', [.319, .629915210440231, 1.999, 2.])
def test_coasting_keeps_certified_schedule_through_policy_and_dispatch(policy,model,monkeypatch,age):
    result,obs=setup(policy,model,monkeypatch)
    bundle=result.full_pick_plan
    # Each probe has a genuine latest exposure followed by a detector miss;
    # an unchanged old exposure is still rejected by the age test above.
    for now in (9.,bundle.start+.01,(bundle.high_time+bundle.contact_time)/2.):
        data=obs(now,age=age);data['predictions']['part']['visible']=False
        stamp=data['predictions']['part']['observed_timestamp']
        policy._full_pick_act(data)
        q=bundle.sample_full(now)[0]
        assert policy._full_pick_abort_reason is None
        assert policy.full_pick_dispatch_reason(data,q,now) is None
        assert policy._full_pick_bundle is bundle
        assert data['predictions']['part']['observed_timestamp']==stamp
    assert len(policy._planning_worker.requests)==1
    assert policy.stage==PolicyStage.TRACKING_DESCENT


@pytest.mark.parametrize('offset,reason',[
    (.002,None),(.005,'predicted_contact_left_certificate_tolerance')])
def test_observation_after_coasting_still_checks_same_contact_time(policy,model,monkeypatch,offset,reason):
    result,obs=setup(policy,model,monkeypatch)
    data=obs();data['predictions']['part']['visible']=False
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason is None
    # A new exposure, with its own timestamp, compares at the frozen contact
    # time; ordinary target motion alone must not count as a position error.
    data=obs(10.1,offset=offset)
    q=result.full_pick_plan.sample_full(data['timestamp'])[0]
    assert policy.full_pick_dispatch_reason(data,q,data['timestamp'])==reason
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason==reason
    assert len(policy._planning_worker.requests)==1


def test_coasting_does_not_close_or_lift_just_because_curve_reaches_contact(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    qualified=[]
    def not_ready(_target,_observation,now,_pose):
        qualified.append(now)
        return False,None
    monkeypatch.setattr(policy,'_descent_closure_qualification',not_ready)
    monkeypatch.setattr(policy,'_grasp_feedback',lambda *_:False)
    for now in (9.,result.full_pick_plan.contact_time):
        data=obs(now);data['predictions']['part']['visible']=False
        policy._full_pick_act(data)
        assert policy._full_pick_abort_reason is None
    assert qualified
    assert policy.stage==PolicyStage.TRACKING_DESCENT
    assert not any(e.name=='gripper_close_started' for e in policy._events)
    data=obs(result.full_pick_plan.lift_time);data['predictions']['part']['visible']=False
    policy._full_pick_act(data)
    assert policy._full_pick_abort_reason=='grasp_not_confirmed_in_certified_close_window'
    assert not policy._full_pick_lift_admitted


def test_late_delivery_and_no_grasp_never_start_loaded_curve(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs(10.01))
    assert policy._full_pick_abort_reason=='full_pick_delivery_missed_start'


def test_empty_grasp_close_window_aborts_instead_of_lifting(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    monkeypatch.setattr(policy,'_grasp_feedback',lambda *_:False)
    policy._full_pick_act(obs(result.full_pick_plan.contact_time))
    policy._full_pick_act(obs(result.full_pick_plan.lift_time))
    assert policy._full_pick_abort_reason=='grasp_not_confirmed_in_certified_close_window'
    assert not policy._full_pick_lift_admitted


def test_future_lift_sample_requires_current_confirmed_grasp_and_original_start_tolerance(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    bundle=result.full_pick_plan
    policy._full_pick_act(obs(bundle.contact_time))
    policy._full_pick_act(obs(bundle.contact_time+.06))
    data=obs(bundle.lift_time-.001)
    data['dispatch_clock']={'scheduled_dispatch_policy_time_s':bundle.lift_time+.001}
    policy._full_pick_act(data)
    assert policy._full_pick_lift_admitted
    assert policy._full_pick_abort_reason is None


def test_new_scene_drops_no_in_motion_replacement(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    data=obs(10.1);data['scene_epoch']=8
    policy.act(data)
    assert policy._full_pick_abort_reason=='full_pick_scene_changed'
    assert len(policy._planning_worker.requests)==1


def test_pending_scene_change_aborts_instead_of_resetting_and_resubmitting(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._planning_worker.results.clear()
    assert policy._full_pick_bundle is None and policy._full_pick_request is not None
    data=obs(9.);data['scene_epoch']=8
    policy.act(data)
    assert policy._full_pick_abort_reason=='full_pick_scene_changed'
    assert len(policy._planning_worker.requests)==1
    policy.act(obs(9.1))
    assert len(policy._planning_worker.requests)==1


def test_final_dispatch_uses_actual_clock_not_old_preparation_time(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    bundle=result.full_pick_plan
    when=(bundle.start+bundle.high_time)/2.
    q=bundle.sample_full(when)[0]
    assert policy.full_pick_dispatch_reason(obs(when),q,when) is None
    assert policy.full_pick_dispatch_reason(obs(when),q,when+.3)=='full_pick_dispatch_clock_drift'
    assert policy.full_pick_dispatch_reason(obs(when),q,bundle.end+.01)=='full_pick_dispatch_past_certificate'


def test_new_obstacle_event_stops_before_next_command(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    policy.env.full_plan_scene_monitor=SimpleNamespace(snapshot=lambda:{'failed':True,
        'failure_reason':'new_obstacle_on_remaining_trajectory'})
    policy._full_pick_act(obs(10.1))
    assert policy._full_pick_abort_reason=='new_obstacle_on_remaining_trajectory'


def test_missing_scene_review_cannot_start_the_curve(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._full_pick_act(obs())
    policy.env._policy_clock_origin=0.
    policy.env.full_plan_scene_monitor=SimpleNamespace(snapshot=lambda:{'failed':False,
        'validated_capture_monotonic_s':9.0,'checked_plan_identity':None})
    policy._full_pick_act(obs(10.1))
    assert policy._full_pick_abort_reason=='scene_review_expired_or_missing_for_plan'


def test_request_entry_uses_policy_selection_setting(policy,monkeypatch):
    policy.config=replace(policy.config,selection_policy='first_safe')
    monkeypatch.setattr(policy,'_global_goal_seeds',lambda q,*a,**k:(q.copy(),))
    from test_policy_moving_approach import observation,future_pose
    req=policy._make_planning_request(future_pose(),observation())
    assert req.selection_policy=='first_safe'
    assert not req.optimize_direct_path_eta


def test_waiting_and_aborted_hold_use_real_deferred_dispatch(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    policy._planning_worker.results.clear()
    data=obs(); data['_defer_arm_motion']=True
    anchor=policy._full_pick_hold_q.copy()
    pending=policy._full_pick_act(data)
    assert pending['_arm_motion_pending']
    np.testing.assert_array_equal(policy.finalize_pending_motion(data)['arm'],anchor)
    pending=policy._full_pick_abort(data,'test_abort')
    assert pending['_arm_motion_pending']
    anchor=policy._full_pick_hold_q.copy()
    data['joint_pos']=data['joint_pos']+.02
    np.testing.assert_array_equal(policy.finalize_pending_motion(data)['arm'],anchor)


def test_submission_reserves_complete_budget_and_late_abort_cannot_resubmit(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    req=policy._full_pick_request
    policy._full_pick_request=None
    monkeypatch.setattr(policy,'_make_planning_request',lambda *a,**k:req)
    captured=[]
    worker=SimpleNamespace(submit=lambda r:captured.append(r) or True)
    monkeypatch.setattr(policy,'_ensure_planning_worker',lambda:worker)
    data=obs(9.)
    visual,reason=policy._full_pick_observation(data,9.)
    assert reason is None
    assert policy._submit_full_pick(data,9.,visual) is None
    assert len(captured)==1
    submitted=captured[0]
    assert submitted.continuation_start_time_s>=9.+policy.config.full_plan_compute_budget_s
    assert submitted.full_plan_compute_budget_s==policy.config.full_plan_compute_budget_s
    # The earlier real certified result is now late. No second job is allowed.
    policy._full_pick_request=req
    policy._poll_full_pick(obs(10.01),10.01)
    assert policy._full_pick_abort_reason=='full_pick_delivery_missed_start'
    policy.act(obs(10.02))
    assert len(captured)==1


def test_full_submission_honors_disabled_tcp_speed_limit(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch,approach_mode='continuous')
    req=policy._full_pick_request;policy._full_pick_request=None
    policy.config=replace(policy.config,tcp_speed_limits_enabled=False)
    monkeypatch.setattr(policy,'_make_planning_request',lambda *a,**k:replace(req,tcp_speed_limits_enabled=False))
    submitted=[]
    monkeypatch.setattr(policy,'_ensure_planning_worker',lambda:SimpleNamespace(submit=lambda r:submitted.append(r) or True))
    data=obs(9.);visual,reason=policy._full_pick_observation(data,9.)
    assert reason is None
    assert policy._submit_full_pick(data,9.,visual) is None
    assert submitted[0].continuation_tcp_speed_limits is None


def test_budget_cancel_is_reported_and_never_adopted_as_certificate(policy,model,monkeypatch):
    result,obs=setup(policy,model,monkeypatch)
    request=policy._full_pick_request
    cancelled=replace(result,target_pose=request.target_pose,start_q=request.start_q,effective_request=None,
        path=None,cost=None,full_pick_plan=None,continuation_path=None,continuation_trajectory=None,
        path_fully_validated=False,continuation_fully_validated=False,cancelled=True,
        error='planning_candidate_budget_exceeded')
    policy._planning_worker.results.clear()
    policy._planning_worker.results.append(cancelled)
    policy._poll_full_pick(obs(9.),9.)
    assert policy._full_pick_abort_reason=='planning_candidate_budget_exceeded'
    assert policy._full_pick_bundle is None


def test_full_submission_tries_inside_window_when_geometric_estimate_is_late(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch,approach_mode='continuous')
    req=policy._full_pick_request;policy._full_pick_request=None
    monkeypatch.setattr(policy,'_make_planning_request',lambda *a,**k:req)
    monkeypatch.setattr('mujoco_conveyor_sorting.controllers.full_pick_execution.intercept_lead',
                        lambda *a,**k:100.)
    submitted=[]
    monkeypatch.setattr(policy,'_ensure_planning_worker',lambda:SimpleNamespace(
        submit=lambda r:submitted.append(r) or True))
    data=obs(9.);visual,reason=policy._full_pick_observation(data,9.)
    assert reason is None
    visual[0]['grasp_window']={'start_time':12.,'end_time':20.,'feasible':True}
    assert policy._submit_full_pick(data,9.,visual) is None
    plan=submitted[0]
    assert plan.planned_target_time_s==pytest.approx(20.-policy.config.intercept_deadline_margin)
    assert plan.full_contact_earliest_s==12.
    assert plan.full_contact_latest_s==pytest.approx(plan.planned_target_time_s)
    np.testing.assert_allclose(plan.target_pose[:3],visual[1][:3]+visual[3]*(plan.planned_target_time_s-visual[2]))


def test_pending_full_plan_is_cancelled_when_even_fastest_contact_misses_window(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch,approach_mode='continuous')
    policy._planning_worker.results.clear()
    target={'object_id':'part','confidence':1.,'grasp_window':{'end_time':9.5}}
    monkeypatch.setattr(policy,'_find_target',lambda *_:target)
    data=obs(9.)
    action=policy._full_pick_act(data)
    assert policy._full_pick_abort_reason=='full_pick_window_expired_during_planning'
    assert policy.stage==PolicyStage.RECOVERING
    assert policy._full_pick_bundle is None
    assert action is not None


def test_pre_motion_reappointment_adopts_effective_clock_with_one_worker_submission(policy,model,monkeypatch):
    _,obs=setup(policy,model,monkeypatch)
    original=policy._full_pick_request
    contact=original.target_pose.copy()
    contact[:3]+=np.array(original.continuation_velocity_m_s)*2.
    revised=replace(original,target_poses=(contact,),
        continuation_start_time_s=original.continuation_start_time_s+2.,
        planned_target_time_s=original.planned_target_time_s+2.)
    result=make_engine(model).plan(revised,lambda:False)
    assert result.success,result.error
    policy._planning_worker.results[:]=[result]
    anchor=policy._full_pick_hold_q.copy()
    action=policy._full_pick_act(obs())
    assert policy._full_pick_abort_reason is None
    assert policy._full_pick_bundle is result.full_pick_plan
    assert policy._full_pick_bundle.request is result.effective_request
    assert policy._full_pick_bundle.start==revised.continuation_start_time_s
    assert policy._pregrasp_capture_time_s==revised.planned_target_time_s
    assert len(policy._planning_worker.requests)==1
    np.testing.assert_array_equal(action['arm'],anchor)


@pytest.mark.parametrize('delta,allowed',[(.002,True),(.02,False)])
@pytest.mark.parametrize('replanning',[False,True])
def test_velocity_update_keeps_frozen_close_window_gate(policy,model,monkeypatch,delta,allowed,replanning):
    result,obs=setup(policy,model,monkeypatch)
    policy.config=replace(policy.config,full_plan_replanning_enabled=replanning)
    policy._full_pick_act(obs())
    bundle=result.full_pick_plan
    data=obs(10.1)
    velocity=np.array(bundle.request.continuation_velocity_m_s);velocity[0]+=delta
    stamp=data['predictions']['part']['observed_timestamp']
    # Latest velocity still predicts EXACTLY the certified contact position,
    # but may leave the original 4 mm envelope during the following closure.
    captured=bundle.contact_pose.copy();captured[:3]+=velocity*(stamp-bundle.contact_time)
    current=captured.copy();current[:3]+=velocity*(data['timestamp']-stamp)
    data['belt_velocity']=velocity
    data['predictions']['part'].update(observed_pose=captured,current_ee_grasp_pose=current)
    now=data['timestamp']
    q=bundle.sample_full(now)[0]
    dispatch=policy.full_pick_dispatch_reason(data,q,now)
    policy._full_pick_act(data)
    if allowed:
        assert policy._full_pick_abort_reason is None and dispatch is None
    else:
        assert dispatch==policy._full_pick_abort_reason=='predicted_close_window_left_certificate_tolerance'
    assert len(policy._planning_worker.requests)==(2 if replanning and allowed else 1)

@pytest.mark.parametrize('field,value', [('request_id',101),('generation',-1),
    ('stage','old-stage'),('target_id','old-object'),('scene_epoch',-1)])
def test_stale_full_result_preserves_current_request_and_stage(policy,model,monkeypatch,field,value):
    current,obs=setup(policy,model,monkeypatch)
    request=policy._full_pick_request;policy._planning_request=request
    policy._planning_worker.results[:]=[replace(current,**{field:value}),current]
    stage=policy.stage;stale=policy._planning_stale_results
    assert policy._poll_full_pick(obs(),9.) is None
    assert policy._full_pick_request is request and policy._planning_request is request
    assert policy.stage==stage and policy._full_pick_bundle is None
    assert policy._full_pick_abort_reason is None
    assert policy._planning_stale_results==stale+1
    assert policy._full_pick_events[-1]['event']=='stale_result_ignored'
    policy._poll_full_pick(obs(),9.)
    assert policy._full_pick_bundle is current.full_pick_plan
    assert policy._full_pick_request is None

@pytest.mark.parametrize('change', [dict(error='actual-planning-failure'),
    dict(path_fully_validated=False),dict(continuation_fully_validated=False)])
def test_current_full_failure_is_not_swallowed(policy,model,monkeypatch,change):
    current,obs=setup(policy,model,monkeypatch)
    policy._planning_worker.results[:]=[replace(current,**change)]
    policy._poll_full_pick(obs(),9.)
    assert policy._full_pick_abort_reason
    assert policy._full_pick_request is None and policy._full_pick_bundle is None
    assert policy.stage==PolicyStage.RECOVERING


def test_actual_worker_cancelled_results_cannot_erase_new_full_request(policy,model,monkeypatch):
    from threading import Event
    from mujoco_conveyor_sorting.controllers.planning_worker import AsyncPlanningWorker
    template,obs=setup(policy,model,monkeypatch)
    entered=[Event() for _ in range(4)];release=[Event() for _ in range(4)]
    requests=[replace(template.full_pick_plan.request,request_id=101+i,generation=i+1) for i in range(4)]
    class Engine:
        def plan(self,request,cancelled):
            i=request.request_id-101;entered[i].set()
            assert release[i].wait(3.)
            bundle=replace(template.full_pick_plan,request=request)
            return replace(template,request_id=request.request_id,generation=request.generation,
                full_pick_plan=bundle,cancelled=cancelled(),
                effective_request=request,error='cancelled-old-request' if cancelled() else None)
    worker=AsyncPlanningWorker(Engine());policy._planning_worker=worker
    try:
        for i,request in enumerate(requests):
            policy._planning_generation=request.generation
            policy._full_pick_request=policy._planning_request=request
            assert worker.submit(request)
            if i:release[i-1].set()
            assert entered[i].wait(3.)
            if i:
                policy._poll_full_pick(obs(),9.)
                assert policy._full_pick_request is request
                assert policy._full_pick_bundle is None and policy._full_pick_abort_reason is None
                assert policy.stage==PolicyStage.PREGRASP
            if i<3:worker.cancel_pending()
        release[-1].set()
        import time
        deadline=time.monotonic()+3.
        while policy._full_pick_bundle is None and time.monotonic()<deadline:
            policy._poll_full_pick(obs(),9.);time.sleep(.001)
        assert policy._full_pick_bundle.request is requests[-1]
        assert policy._planning_stale_results==3
        assert policy._grasp_confirmed_at is None
    finally:
        for event in release:event.set()
        worker.close()

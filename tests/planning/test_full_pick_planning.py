"""Native IK/edge certificates, no devices."""
from dataclasses import replace
import numpy as np
import pytest
from test_planning_worker_scene_validator import model, make_engine, make_request
from test_ranked_goal_routes import redundant_request
from mujoco_conveyor_sorting.runtime.single_pick import PickSettings


def test_contact_prefix_preserves_network_contact_limited_zero_closure_width(model):
    request=full_request(full_approach_mode='continuous',full_contact_handoff_enabled=True,
        full_grasp_width_m=0.)
    result=make_engine(model).plan(request,lambda:False)
    assert result.success,result.error
    assert result.full_pick_plan.request.full_grasp_width_m==0.
    assert result.full_pick_plan.closing is None and result.full_pick_plan.lift is None
    with pytest.raises(ValueError,match='positive'):
        full_request(full_approach_mode='continuous',full_grasp_width_m=0.)


def full_request(**kwargs):
    original = make_request()
    contact = original.target_pose.copy(); contact[2] = .15
    values = dict(planning_mode='full_preplanned', selection_policy='first_safe',
        replan_policy='event_triggered', continuous_intercept=True,
        target_poses=(contact,), continuation_start_time_s=10., planned_target_time_s=13.,
        timed_continuation_duration_s=3., intercept_request_time_s=0.,
        continuation_velocity_m_s=(.02,.01,0.), full_lift_m=.03,
        trajectory_jerk_limits=(6.,)*7, continuation_cartesian_step=.015,
        continuation_max_joint_step=.05, intercept_descent_duration_s=.8)
    values.update(kwargs)
    return replace(original, **values)


def test_explicit_contact_plan_certifies_only_executable_approach(model):
    request=full_request(full_approach_mode='continuous',full_contact_handoff_enabled=True)
    engine=make_engine(model)
    original=engine._state_validator
    def validator(req,q):
        if req.full_phase in {'closing','lift'}:return False
        return True if original is None else original(req,q)
    engine._state_validator=validator
    result=engine.plan(request,lambda:False)
    assert result.success,result.error
    b=result.full_pick_plan
    assert b.closing is None and b.lift is None and b.closing_pose is None and b.lift_pose is None
    assert b.end==b.contact_time
    assert {phase for phase,_,_ in b.checked_samples}=={'approach'}
    assert result.diagnostics['full_preplanned']['loaded_motion_not_part_of_this_certificate']
    np.testing.assert_allclose(b.sample_full(b.contact_time)[0],b.approach.q1)
    assert result.path_fully_validated and result.continuation_fully_validated
    assert not engine.plan(replace(request,full_contact_handoff_enabled=False),lambda:False).success


def test_first_safe_switches_real_seed_and_no_ranker(model, monkeypatch):
    request = replace(redundant_request(), selection_policy='first_safe',
        optimize_direct_path_eta=True, rrt_restarts_after_direct=9, full_rrt_candidates_after_direct=8)
    assert not request.optimize_direct_path_eta
    assert request.rrt_restarts_after_direct == request.full_rrt_candidates_after_direct == 0
    engine = make_engine(model); engine.rank_goal_routes = True
    monkeypatch.setattr(engine, '_ranked_goal_route', lambda *_a, **_k: pytest.fail('ranker called'))
    result = engine.plan(request, lambda: False)
    assert result.success and result.path_fully_validated and result.continuation_fully_validated
    np.testing.assert_array_equal(result.diagnostics['ik_attempts'][0]['seed_joints_rad'],request.start_q)
    assert result.diagnostics['ik_seeds_tried'] == 1
    assert result.diagnostics['extra_comparison_wall_s'] == 0.
    assert result.diagnostics['first_safe_route_wall_s'] > 0.
    assert result.diagnostics['route_source'] == 'direct'


def test_first_seed_bad_next_seed_available(model, monkeypatch):
    engine = make_engine(model); solve = engine._solve_goal_seed; called=[]
    def first_bad(req,pose,seed,cancelled):
        called.append(seed.copy())
        return None if len(called)==1 else solve(req,pose,seed,cancelled)
    monkeypatch.setattr(engine,'_solve_goal_seed',first_bad)
    request=replace(make_request(),selection_policy='first_safe')
    result=engine.plan(request,lambda:False)
    assert result.success and len(called)==2
    np.testing.assert_array_equal(called[0],request.start_q)
    np.testing.assert_array_equal(called[1],request.goal_seeds[0])


def test_static_first_safe_result_keeps_submitted_contract_after_seed_reordering(model):
    request=replace(make_request(),stage='lifting',selection_policy='first_safe')
    result=make_engine(model).plan(request,lambda:False)
    assert result.success and result.path_fully_validated
    assert result.effective_request is None
    assert result.request_id==request.request_id and result.generation==request.generation
    np.testing.assert_array_equal(result.start_q,request.start_q)
    assert any(np.array_equal(result.target_pose,p) for p in request.target_poses)


def test_first_safe_full_descent_and_collision_still_required(model):
    request=replace(redundant_request(),selection_policy='first_safe',rrt_restart_count=1)
    assert not make_engine(model,lambda _r,q:not .22<q[2]<.25).plan(request,lambda:False).success
    assert not make_engine(model,lambda _r,q:not .08<q[0]<.12).plan(request,lambda:False).success


def test_complete_c2_clock_contact_velocity_and_derivative_limits(model):
    request=full_request(); engine=make_engine(model)
    result=engine.plan(request,lambda:False)
    assert result.success,(result.error,result.diagnostics)
    assert result.path_fully_validated and result.continuation_fully_validated
    bundle=result.full_pick_plan
    assert bundle.request is result.effective_request
    assert bundle.contact_time==request.planned_target_time_s
    assert bundle.start==request.continuation_start_time_s
    assert bundle.high_time+bundle.descent.duration==pytest.approx(bundle.contact_time)
    for a,b in [(bundle.approach,bundle.descent),(bundle.descent,bundle.closing),(bundle.closing,bundle.lift)]:
        for left,right in zip(a.sample_full(a.duration),b.sample_full(0.)):
            np.testing.assert_allclose(left,right,atol=1e-7)
    q,v,_=bundle.descent.sample_full(bundle.descent.duration)
    np.testing.assert_allclose(engine.ik.compute_jacobian(q)@v,[.02,.01,0.,0.,0.,0.],atol=1e-8)
    assert set(p for p,_,_ in bundle.checked_samples)=={'approach','descent','closing','lift'}
    high_floor=min(engine.ik.forward_pose(request.start_q)[2],
                   engine.ik.forward_pose(bundle.approach.sample_full(bundle.approach.duration)[0])[2])-.012
    assert bundle.minimum_tcp_height_at(bundle.start)==pytest.approx(high_floor)
    assert bundle.minimum_tcp_height_at(bundle.contact_time)==request.minimum_tcp_height
    assert result.diagnostics['full_preplanned']['curve_checks'][0]['minimum_tcp_height_m']==pytest.approx(high_floor)
    for curve in [bundle.approach,bundle.descent,bundle.closing,bundle.lift]:
        for order,limits in [(1,request.trajectory_velocity_limits),(2,request.trajectory_acceleration_limits),(3,np.full(7,6.))]:
            lo,hi=curve.derivative_bounds(order)
            assert np.all(np.maximum(abs(lo),abs(hi))<=limits+1e-8)
    assert result.diagnostics['extra_comparison_wall_s']==0.


@pytest.mark.parametrize('phase',['descent','closing','lift'])
def test_full_curve_obstacle_rejected_in_every_phase(model,phase):
    def obstacle(req,q):
        return not(req.planning_mode=='full_preplanned' and req.full_phase==phase and req.full_sample_time_s is not None)
    result=make_engine(model,obstacle).plan(full_request(),lambda:False)
    assert not result.success and result.full_pick_plan is None
    assert result.diagnostics['rejection_counts']['full_pick_curve_geometry']>0


def test_bent_boundary_curve_is_not_authorized_by_old_chords(model):
    def obstacle(req,q):
        return not(req.full_phase=='descent' and req.full_sample_time_s is not None and q[1]<-.001)
    result=make_engine(model,obstacle).plan(full_request(continuation_velocity_m_s=(.02,.1,0.)),lambda:False)
    assert not result.success


def test_config_defaults_and_invalid_combination():
    old=PickSettings(); old.validate()
    assert old.planning_mode=='legacy' and old.selection_policy=='default'
    with pytest.raises(ValueError,match='requires first_safe'):
        replace(old,planning_mode='full_preplanned').validate()
    replace(old,planning_mode='full_preplanned',selection_policy='first_safe',replan_policy='event_triggered').validate()
    with pytest.raises(ValueError,match='unknown selection_policy'):
        replace(make_request(),selection_policy='fast_unsafe')


def test_first_safe_retains_real_rrt_detour(model):
    from test_ranked_goal_routes import obstacle
    request=redundant_request()
    request=replace(request,goal_seeds=(request.goal_seeds[1],),selection_policy='first_safe',max_goal_candidates=1)
    result=make_engine(model,obstacle).plan(request,lambda:False)
    assert result.success and len(result.path)>2
    assert result.diagnostics['route_source']=='rrt_route'
    for a,b in zip(result.path,result.path[1:]):
        for u in np.linspace(0.,1.,101):assert obstacle(request,a+(b-a)*u)


def test_geometry_safe_but_unreachable_appointment_never_returns_first_safe(model,monkeypatch):
    request=replace(redundant_request(),selection_policy='first_safe',
        intercept_velocity_m_s=np.array([.01,0.,0.]),intercept_motion_start_time_s=0.,
        planned_target_time_s=.1,intercept_descent_duration_s=.8)
    engine=make_engine(model)
    monkeypatch.setattr(engine,'_intercept_request',lambda r,*_a,**_k:r)
    result=engine.plan(request,lambda:False)
    assert not result.success
    assert result.diagnostics['rejection_counts']['first_safe_contact_deadline']>0


def test_initial_budget_predicts_endpoint_and_certifies_once(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(.005,0.,0.))
    engine=make_engine(model); certify=engine._full_certify_curve; seen=[]
    def slow(req,curve,start,phase,cancelled):
        seen.append((req,phase,start))
        out=certify(req,curve,start,phase,cancelled)
        clock[0]+=.3
        return out
    monkeypatch.setattr(engine,'_full_certify_curve',slow)
    result=engine.plan(request,lambda:False)
    assert result.success,result.error
    effective=result.effective_request
    assert effective.planned_target_time_s>request.planned_target_time_s
    np.testing.assert_allclose(effective.target_pose[:3],request.target_pose[:3]
        +np.array(request.continuation_velocity_m_s)*(effective.planned_target_time_s-request.planned_target_time_s))
    assert result.full_pick_plan.request is effective
    assert set(phase for r,phase,_ in seen if r is effective)=={'approach','descent','closing','lift'}
    assert result.full_pick_plan.start>clock[0]
    report=result.diagnostics['full_preplanned']
    assert report['time_recertification_wall_s']==0.
    assert len(report['time_refinements'])==1
    assert effective.continuation_start_time_s==pytest.approx(
        request.intercept_request_time_s+request.full_plan_compute_budget_s+request.trajectory_minimum_duration)


def test_second_late_certificate_is_discarded_without_chasing_more_appointments(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(0.,0.,0.))
    engine=make_engine(model); once=engine._plan_full_pick_once; calls=[]
    def late(req,*args):
        result=once(req,*args); calls.append(req); clock[0]+=10.**len(calls)
        return result
    monkeypatch.setattr(engine,'_plan_full_pick_once',late)
    result=engine.plan(request,lambda:False)
    assert not result.success and result.full_pick_plan is None
    assert not result.path_fully_validated and len(calls)==2
    assert result.error=='full_pick_delivery_missed_start'
    report=result.diagnostics['full_preplanned']
    assert report['feasible_candidates']==2
    assert report['appointment_changes_after_feasible']==1
    assert report['time_recertification_wall_s']==100.


def test_late_initial_pick_rechecks_all_phases_at_new_predicted_appointment(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(.005,0.,0.))
    engine=make_engine(model);certify=engine._full_certify_curve;seen=[]
    def slow(req,curve,start,phase,cancelled):
        seen.append((req,phase,start))
        out=certify(req,curve,start,phase,cancelled)
        clock[0]+=.75
        return out
    monkeypatch.setattr(engine,'_full_certify_curve',slow)
    result=engine.plan(request,lambda:False)
    assert result.success,result.error
    effective=result.effective_request;bundle=result.full_pick_plan
    assert effective is bundle.request
    assert bundle.start>=clock[0]+request.trajectory_minimum_duration-1e-9
    assert effective.continuation_start_time_s>seen[0][0].continuation_start_time_s
    assert effective.planned_target_time_s>seen[0][0].planned_target_time_s
    np.testing.assert_allclose(effective.target_pose[:3],request.target_pose[:3]
        +np.array(request.continuation_velocity_m_s)*(effective.planned_target_time_s-request.planned_target_time_s))
    assert set(phase for r,phase,_ in seen if r is effective)=={'approach','descent','closing','lift'}
    report=result.diagnostics['full_preplanned']
    assert report['appointment_changes_after_feasible']==1
    assert report['time_recertification_wall_s']==3.
    assert report['delivery_forecasts'][-1]['ik_success']
    assert report['original_weighted_validation_samples']>0
    assert len(report['time_refinements'])==2
    for phase,t,q in bundle.checked_samples:
        np.testing.assert_allclose(bundle.sample_full(t)[0],q)


def test_reappointment_with_more_samples_reserves_the_measured_full_job(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(0.,0.,0.))
    engine=make_engine(model);once=engine._plan_full_pick_once;work=engine._full_pick_validation_work
    calls=[]
    def timed(req,*args):
        result=once(req,*args);calls.append(req);clock[0]+=3.
        return result
    def twice(req,curves):
        n,counts=work(req,curves)
        return (2*n if len(calls)>0 and req.continuation_start_time_s>calls[0].continuation_start_time_s else n),counts
    monkeypatch.setattr(engine,'_plan_full_pick_once',timed)
    monkeypatch.setattr(engine,'_full_pick_validation_work',twice)
    result=engine.plan(request,lambda:False)
    assert result.success,result.error
    assert result.diagnostics['full_preplanned']['revised_compute_forecast_s']==6.
    assert result.full_pick_plan.start>=9.+request.trajectory_minimum_duration


def test_new_appointment_geometry_failure_never_returns_the_old_certificate(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(.005,0.,0.))
    engine=make_engine(model,lambda r,q:r.planned_target_time_s<6.)
    once=engine._plan_full_pick_once;calls=[]
    def slow(req,*args):
        result=once(req,*args);calls.append(req);clock[0]+=3.
        return result
    monkeypatch.setattr(engine,'_plan_full_pick_once',slow)
    result=engine.plan(request,lambda:False)
    assert len(calls)==2
    assert not result.success and result.full_pick_plan is None
    assert not result.path_fully_validated
    assert result.effective_request is calls[1]
    assert not result.diagnostics['full_preplanned']['certified']


def test_cancelled_reappointment_never_publishes_first_certificate(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as module
    clock=[0.];cancel=[False]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    engine=make_engine(model);once=engine._plan_full_pick_once;forecast=engine._full_pick_delivery_request
    def slow(*args):
        result=once(*args);clock[0]+=3.
        return result
    def stop(*args):
        revision=forecast(*args);cancel[0]=True
        return revision
    monkeypatch.setattr(engine,'_plan_full_pick_once',slow)
    monkeypatch.setattr(engine,'_full_pick_delivery_request',stop)
    result=engine.plan(full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True),lambda:cancel[0])
    assert result.cancelled and not result.success
    assert result.full_pick_plan is None and not result.path_fully_validated


def test_boundary_endpoints_are_exact_even_on_soft_limit():
    from mujoco_conveyor_sorting.controllers.continuous_joint_trajectory import BoundaryQuinticJointTrajectory
    a=np.array([.13,.02,-.57,.48,1.3,-.1,.19]);b=np.array([.128,.02,-.71,.83,2.,.8,-1.])
    curve=BoundaryQuinticJointTrajectory(a,b,np.ones(7),np.ones(7),allow_path_change=True)
    np.testing.assert_array_equal(curve.sample_full(0.)[0],a)
    np.testing.assert_array_equal(curve.sample_full(curve.duration)[0],b)
    np.testing.assert_array_equal(curve.sample_full(curve.duration)[1],np.zeros(7))


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_complete_compute_budget_is_rejected(value):
    with pytest.raises(ValueError):full_request(full_plan_compute_budget_s=value)
    with pytest.raises(ValueError):replace(PickSettings(),full_plan_compute_budget_s=value).validate()


def test_cancelled_complete_certificate_is_never_published(model,monkeypatch):
    engine=make_engine(model);once=engine._plan_full_pick_once;cancel=[False]
    def finish_then_cancel(*args):
        result=once(*args);assert result.success
        cancel[0]=True
        return result
    monkeypatch.setattr(engine,'_plan_full_pick_once',finish_then_cancel)
    result=engine.plan(full_request(),lambda:cancel[0])
    assert result.cancelled and not result.success and result.full_pick_plan is None
    assert not result.path_fully_validated
    assert not result.diagnostics['full_preplanned']['certified']


def test_infeasible_motion_can_retry_before_first_feasible_without_moving_start(model,monkeypatch):
    engine=make_engine(model);once=engine._plan_full_pick_once;seen=[]
    def first_infeasible(req,cancelled,started):
        seen.append(req)
        if len(seen)==1:
            engine._request_diagnostics['full_preplanned']['suggested_duration_s']=4.
            return engine._full_failure(req,'infeasible candidate timing',started)
        return once(req,cancelled,started)
    monkeypatch.setattr(engine,'_plan_full_pick_once',first_infeasible)
    req=full_request()
    result=engine.plan(req,lambda:False)
    assert result.success,result.error
    assert len(seen)==2
    assert seen[0].continuation_start_time_s==seen[1].continuation_start_time_s
    assert seen[1].planned_target_time_s==seen[0].planned_target_time_s+1.
    np.testing.assert_allclose(seen[1].target_pose[:3],req.target_pose[:3]+np.array(req.continuation_velocity_m_s))
    assert result.diagnostics['full_preplanned']['feasible_candidates']==1

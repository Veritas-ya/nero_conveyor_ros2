"""The forecast is a proposal; real contact must stay in its allowed window."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.nero_moving_context import NeroMovingPolicyContext
from test_planning_worker_scene_validator import model, make_engine
from test_full_pick_planning import full_request


def test_upstream_target_with_future_window_is_eligible_for_full_planning():
    prediction={'valid': True, 'grasp_window': {'feasible': True},
                'predicted_grasp_pose': np.array([-.4,0,0,1,0,0,0])}
    context=SimpleNamespace(_target={'object_id':'part','prediction':prediction},
        _released=False,policy=SimpleNamespace(config=SimpleNamespace(planning_mode='full_preplanned')),
        grasp_pose=np.zeros(7),s_range=(-.2,.28),_coordinate=lambda p:float(p[0]))
    assert NeroMovingPolicyContext.is_target_feasible(context,'part')
    context.policy.config.planning_mode='legacy'
    assert not NeroMovingPolicyContext.is_target_feasible(context,'part')
    context.policy.config.planning_mode='full_preplanned'
    prediction['grasp_window']['feasible']=False
    assert not NeroMovingPolicyContext.is_target_feasible(context,'part')


def test_full_planner_never_certifies_contact_after_window(model,monkeypatch):
    engine=make_engine(model)
    request=full_request()
    assert request.planned_target_time_s is not None
    constrained=replace(request,full_contact_latest_s=request.planned_target_time_s-.01)
    def forbidden(*args,**kwargs):
        pytest.fail('expensive certification must not run after window expiry')
    monkeypatch.setattr(engine,'_plan_full_pick_once',forbidden)
    result=engine.plan(constrained,lambda:False)
    assert not result.success and result.error=='full_pick_contact_outside_reachable_window'


def test_real_planner_accepts_a_valid_contact_within_window(model):
    engine=make_engine(model)
    request=full_request()
    contact=request.planned_target_time_s
    result=engine.plan(replace(request,full_contact_earliest_s=contact-.1,
                               full_contact_latest_s=contact+.1),lambda:False)
    assert result.success, result.error
    assert result.full_pick_plan.contact_time == pytest.approx(contact)
    assert result.path_fully_validated and result.continuation_fully_validated


def test_late_reappointment_cannot_recertify_after_contact_window(model,monkeypatch):
    import mujoco_conveyor_sorting.controllers.planning_worker as worker_module
    clock=[0.]
    monkeypatch.setattr(worker_module.time,'perf_counter',lambda:clock[0])
    request=full_request(continuation_start_time_s=.1,planned_target_time_s=3.1,
        intercept_account_for_planning_time=True,continuation_velocity_m_s=(0.,0.,0.),
        full_contact_latest_s=5.3)
    engine=make_engine(model); once=engine._plan_full_pick_once; calls=[]
    def late(req,*args):
        result=once(req,*args); calls.append(req); clock[0]+=10.
        return result
    monkeypatch.setattr(engine,'_plan_full_pick_once',late)
    result=engine.plan(request,lambda:False)
    assert not result.success and result.error=='full_pick_contact_outside_reachable_window'
    assert result.full_pick_plan is None and len(calls)==1


def test_invalid_contact_window_is_rejected():
    with pytest.raises(ValueError,match='inverted'):
        replace(full_request(),full_contact_earliest_s=5.,full_contact_latest_s=4.)

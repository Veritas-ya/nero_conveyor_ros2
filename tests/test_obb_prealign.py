"""Offline coverage for optional prealignment; never sends to hardware."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.reference import Intercept
from test_two_second_smoothing import fixture as planner_fixture


def reference(prealign):
    start_q = Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat()[[3, 0, 1, 2]]
    target_q = Rotation.from_euler('xyz', [-.2, .1, 1.5]).as_quat()[[3, 0, 1, 2]]
    return SmoothIntercept(1., 4.-prealign, 2.+prealign, [.1, .2, .03], [.05, 0., 0.],
                           [0., 0., .2], start_q, target_q, .8, 1.5, prealign)


def test_prealign_rotates_at_fixed_tcp_before_two_second_approach():
    r = reference(1.5)
    first = Rotation.from_quat(r.start_quaternion_wxyz[[1,2,3,0]])
    last = Rotation.from_quat(r.target_quaternion_wxyz[[1,2,3,0]])
    for t in np.linspace(0., 1.5, 9):
        p, q = r.sample(t)
        np.testing.assert_allclose(p, r.start_tcp_world, atol=1e-14)
        p2, v, a = r.position_derivatives(t)
        np.testing.assert_allclose(p2, p, atol=1e-14)
        np.testing.assert_allclose([v, a], np.zeros((2,3)), atol=1e-12)
        if t == 1.5:
            assert (Rotation.from_quat(q[[1,2,3,0]]).inv()*last).magnitude() < 1e-12
    _, qhalf = r.sample(.75)
    assert (first.inv()*Rotation.from_quat(qhalf[[1,2,3,0]])).magnitude() == pytest.approx((first.inv()*last).magnitude()/2)
    for t in [1.5, 2., 3.5]:
        assert (Rotation.from_quat(r.sample(t)[1][[1,2,3,0]]).inv()*last).magnitude() < 1e-12
    assert r.contact_s == 6.
    np.testing.assert_allclose(r.contact_xyz, reference(0).contact_xyz)
    for t in [0., .5, 1., 2.]:
        np.testing.assert_allclose(r.sample(1.5+t)[0], reference(0).sample(t)[0], atol=1e-14)
    for eps in [-1e-7, 1e-7]:
        p,v,a = r.position_derivatives(1.5+eps)
        np.testing.assert_allclose(p,r.start_tcp_world,atol=1e-12)
        np.testing.assert_allclose(v,0,atol=1e-11)
        np.testing.assert_allclose(a,0,atol=1e-5)


@pytest.mark.parametrize('value',[-1., float('nan'), 2.])
def test_invalid_prealign_rejected(value):
    with pytest.raises(ValueError):
        SmoothIntercept(1.,2.,2.,[0]*3,[0]*3,[0]*3,[1,0,0,0],[1,0,0,0],.8,1.5,value)


def test_planner_schedules_prealign_earlier_and_preserves_contact_time(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=1., prealign_orientation_s=1.)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs): return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    def install(target, state): planner.version += 1
    monkeypatch.setattr(planner, 'install', install)
    seen=[]
    monkeypatch.setattr(planner, 'check_smooth_intercept',lambda path,ref,**kwargs: seen.append(ref))
    result=planner.intercept(target, {'joints':np.zeros(7)})
    assert len(seen) == 1
    assert result.reference.start_s == pytest.approx(4.)
    assert result.reference.contact_s == pytest.approx(7.)
    assert result.reference.prealign_s == pytest.approx(1.)
    assert result.path.diagnostics['approach_s'] == 2.
    assert result.path.diagnostics['prealign_s'] == 1.


def test_collision_overlay_uses_checked_candidate_and_collision_timestamp():
    from nero_core.two_second.camera_view import collision_predicted_grasp
    pose=np.eye(4); pose[:3,3]=[.1,.2,.3]
    metadata={'geometry_context':{'input_grasp_index':2,'selected_grasp_tcp_base':pose.tolist()},
              'capture_monotonic_s':10.,'predicted_monotonic_s':12.,
              'target_velocity_base_m_s':[.05,-.02,0.]}
    actual=collision_predicted_grasp(metadata)
    np.testing.assert_allclose(actual[:3,3],[.2,.16,.3])
    np.testing.assert_array_equal(actual[:3,:3],pose[:3,:3])
    np.testing.assert_array_equal(metadata['geometry_context']['selected_grasp_tcp_base'],pose)
    assert collision_predicted_grasp({}) is None


def test_terminal_marks_prealignment_as_planned():
    from nero_core.two_second.console import ConsoleProgress
    output=[];console=ConsoleProgress(write=output.append)
    def send(event,data):
        console.observe({'event':'execution_event','monotonic_s':1.,'data':{'record':{'event':event,'data':data}}})
    start=np.eye(4);target=np.eye(4)
    target[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    send('planning_submitted',{'start_tcp_base':start.tolist()})
    send('planning_delivery',{'selected_grasp_tcp_base':target.tolist(),'planning_wall_s':1.,
                              'trajectory_diagnostics':{'prealign_s':1.5}})
    assert any('接近前计划原位旋转 90.0°' in line for line in output)
    send('stage_started',{'stage':'smooth_intercept_follow_brake','prealign_s':1.5})
    assert any('先转向再接近' in line for line in output)

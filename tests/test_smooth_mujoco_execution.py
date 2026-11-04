"""Feedback gates only; physical success is checked by the MuJoCo runner."""
import numpy as np
import pytest
from types import SimpleNamespace
from experiments.two_second_grasp.smooth_reference import SmoothDiagnosticReference
from experiments.two_second_grasp.smooth_execution import closure_gate


def reference():
    return SmoothDiagnosticReference(0., 1., 2., [-.3, 0., .53], [.1, 0., 0.],
                                     [0., 0., .7], [0., 1., 0., 0.], [0., 1., 0., 0.])


def feedback(ref, elapsed, error=None, velocity=None):
    now = ref.start_s+elapsed
    pose = np.r_[ref.capture_tcp_world+ref.velocity_world*(now-ref.capture_s),
                 ref.target_quaternion_wxyz]
    if error is not None:
        pose[:3] += error
    before = pose.copy()
    before[:3] -= (ref.velocity_world if velocity is None else velocity)*.02
    return closure_gate(ref, elapsed, pose, before, now-.02, now)


def test_feedback_matched_close_and_original_remaining_dwell():
    r = reference()
    assert feedback(r, 2.)[0]
    assert feedback(r, 2.2)[0]
    assert not feedback(r, 2.201)[0]
    assert not feedback(r, 1.99)[0]


@pytest.mark.parametrize('error', [[.023, 0., 0.], [0., 0., .013]])
def test_spatial_error_prevents_close(error):
    assert not feedback(reference(), 2., error=error)[0]


def test_old_pose_alone_is_insufficient_when_tcp_is_not_following():
    allowed, detail = feedback(reference(), 2., velocity=np.zeros(3))
    assert detail['horizontal_error_m'] == 0.
    assert not detail['drift_ok']
    assert not allowed


def test_orientation_error_prevents_close():
    r = reference()
    xyz, _ = r.sample(2.)
    pose = np.r_[xyz, [1., 0., 0., 0.]]
    previous = pose.copy()
    previous[:3] -= r.velocity_world*.02
    assert not closure_gate(r, 2., pose, previous, 2.98, 3.)[0]


def test_missing_or_reset_feedback_clock_prevents_close():
    r = reference()
    pose = np.r_[r.contact_xyz, r.target_quaternion_wxyz]
    assert not closure_gate(r, 2., pose, None, None, 3.)[0]
    assert not closure_gate(r, 2., pose, pose, 4., 3.)[0]


def post_grasp_fixture(monkeypatch, geometry):
    from experiments.two_second_grasp import placement, smooth_execution
    t = np.linspace(0., 1., 51)
    q = np.outer(3*t*t-2*t*t*t, [.1]*7)
    monkeypatch.setattr(placement, 'plan_return', lambda *a, **k:
                        {'times': t, 'samples': q, 'duration_s': 1.})
    monkeypatch.setattr(smooth_execution, 'check_curve_geometry', geometry)
    controller = SimpleNamespace(max_joint_velocity=np.ones(7),
        max_joint_acceleration=np.full(7, .15),
        command_joint_lower_limits=np.full(7, -1.), command_joint_upper_limits=np.ones(7))
    return SimpleNamespace(robot_controller=controller, arm_qpos_adrs=np.arange(7))


def test_post_grasp_uses_the_certified_retimed_curve_and_rest_handoff(monkeypatch):
    from experiments.two_second_grasp.smooth_execution import plan_smooth_leg
    from nero_core.two_second.trajectory import extrema
    checked = []
    def geometry(env, path, *a, **kw):
        checked.append(path)
        assert np.max(abs(np.asarray(extrema(path.curve, 2)))) <= .15
        return {'states': 101}
    env = post_grasp_fixture(monkeypatch, geometry)
    out = plan_smooth_leg(env, 'return', np.zeros(7), np.zeros(7), np.ones(7)*.1,
                          None, .02, lambda: False)
    assert out['path'] is checked[0]
    assert out['duration_s'] > 1.
    np.testing.assert_allclose(out['path'].q0, 0., atol=1e-12)
    np.testing.assert_allclose(out['path'].q1, .1, atol=1e-12)
    np.testing.assert_allclose(out['path'].sample(0.)[1], 0., atol=1e-12)
    np.testing.assert_allclose(out['path'].sample(out['duration_s'])[1], 0., atol=1e-12)


def test_post_grasp_geometry_failure_never_delivers_an_executable_curve(monkeypatch):
    from experiments.two_second_grasp.smooth_execution import plan_smooth_leg
    env = post_grasp_fixture(monkeypatch, lambda *a, **kw: {'error': 'payload_collision'})
    out = plan_smooth_leg(env, 'return', np.zeros(7), np.zeros(7), np.ones(7)*.1,
                          None, .02, lambda: False)
    assert out == {'error': 'payload_collision'}


def test_redundant_rrt_sampling_does_not_require_a_stop_at_every_sample():
    from experiments.two_second_grasp.placement import merge_collinear_waypoints
    dense = np.outer(np.linspace(0., 1., 40), np.arange(1., 8.)/10)
    path = merge_collinear_waypoints(dense)
    assert len(path) == 2
    np.testing.assert_allclose(path, dense[[0, -1]])


def test_rrt_merge_preserves_bends_and_reversals():
    from experiments.two_second_grasp.placement import merge_collinear_waypoints
    a = np.zeros(7)
    b = np.array([.1, 0., 0., 0., 0., 0., 0.])
    c = np.array([.1, .1, 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(merge_collinear_waypoints([a, b/2, b, c, b, a]),
                               [a, b, c, b, a])


def test_partial_run_is_not_reported_as_returned_or_physically_complete():
    from experiments.two_second_grasp.simulation_report import summarize
    result = summarize({'status': 'simulation_time_budget_exhausted',
                        'attempts': [{'return_verified': True}], 'command_rows': []})
    assert not result['returned_to_wait']
    assert not result['placement_stable']
    assert result['recorded_control_steps'] == 0

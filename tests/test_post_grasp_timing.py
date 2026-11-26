"""Timing changes preserve geometry, boundary conditions and task ownership."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.interpolate import BPoly, CubicSpline, PPoly

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.post_grasp_timing import accelerate_path, _split_law, _straight_return
from nero_core.two_second.taught_timing import compose_path
from nero_core.two_second.trajectory import TimedPath, extrema
from nero_core.two_second.prepared_cycle import direct_return
from test_prepared_taught_cycle import fixture
from test_continuous_carry import chain, compile


def peaks(curve, d):
    return np.max(abs(np.asarray(extrema(curve, d))), axis=0)


def assert_continuity(curve):
    knots = curve.x[1:-1]
    for d in (0, 1, 2):
        np.testing.assert_allclose(curve(knots, d), curve(np.nextafter(knots, -np.inf), d), atol=1e-8)
    for t in (curve.x[0], curve.x[-1]):
        for d in (1, 2):
            np.testing.assert_allclose(curve(t, d), 0., atol=1e-8)


def straight():
    cfg, p, q = fixture()
    cfg.velocity_limits = np.array([np.pi]*3+[1.25*np.pi]*4)
    cfg.acceleration_limits = np.array([8., 8., 8., 8., 10., 10., 8.])
    cfg.raw['taught_joint_jerk_limit_rad_s3'] = 500.
    curve = PPoly.from_bernstein_basis(BPoly.from_derivatives([0., 1.33],
        [[q(-1.08), q(0.), q(0.)], [q(1.08), q(0.), q(0.)]]))
    path = TimedPath(curve, p.identity, 7, 'return', 1.33, 0.)
    return p, path


def test_straight_return_faster_without_raising_any_axis_jerk():
    p, old = straight()
    result = accelerate_path(p, old, .1, payload=False, cancelled=lambda: False)
    report = result.diagnostics['post_grasp_timing']
    assert report['applied'] and result.duration < old.duration
    assert report['method'] == 'straight_joint_path_bounded_jerk'
    assert result.identity == old.identity and result.version == old.version
    assert_continuity(result.curve)
    assert np.all(peaks(result.curve, 3) <= peaks(old.curve, 3)+1e-8)
    assert np.all(peaks(result.curve, 2) <= p.config.acceleration_limits+1e-8)
    assert np.all(peaks(result.curve, 1) <= p.config.velocity_limits+1e-8)
    np.testing.assert_allclose(result.q0, old.q0, atol=1e-10)
    np.testing.assert_allclose(result.q1, old.q1, atol=1e-10)
    values = result.curve(np.linspace(0., result.duration, 1001))
    assert np.all(np.diff(values[:, 0]) >= -1e-10)
    assert np.max(abs(values[:, 1:])) == 0.
    assert ('check', 'return', False) in p.calls


def test_rescaled_endpoint_never_falls_outside_polynomial_domain():
    from nero_core.two_second.post_grasp_timing import _scale_curve
    duration = 1.89
    # Reproduce the 1.89 vs 1.8899999999999997 failure without recorded assets.
    original_end = next(float(x) for x in np.linspace(1., 2., 1001)
                        if x*(duration/x) < duration)
    curve = PPoly.from_bernstein_basis(BPoly.from_derivatives([0., original_end],
        [[np.zeros(7), np.zeros(7), np.zeros(7)],
         [np.ones(7)*.2, np.zeros(7), np.zeros(7)]]), extrapolate=False)
    scaled = _scale_curve(curve, duration)
    path = TimedPath(scaled, (0, 1, 1), 1, 'carry', duration, 0.)
    assert scaled.x[-1] == path.duration
    assert np.isfinite(path.position(duration)).all()
    np.testing.assert_allclose(path.q1, np.ones(7)*.2, atol=1e-12)
    times = np.linspace(0., duration, 1001)
    source_times = np.clip(times/(duration/original_end), 0., original_end)
    for d in (0, 1, 2):
        np.testing.assert_allclose(scaled(times, d),
            curve(source_times, d)/(duration/original_end)**d, atol=1e-12)
    assert not np.isfinite(scaled(np.nextafter(duration, np.inf))).any()


@pytest.mark.parametrize('velocity,acceleration,jerk', [(20., 20., 1.), (20., .2, 10.), (.1, 20., 10.)])
def test_straight_profile_handles_jerk_acceleration_and_cruise_limits(velocity, acceleration, jerk):
    p, path = straight()
    result = _straight_return(path.curve, np.full(7, velocity), np.full(7, acceleration), np.full(7, jerk))
    assert_continuity(result)
    np.testing.assert_allclose(result(0.), path.q0, atol=1e-8)
    np.testing.assert_allclose(result(result.x[-1]), path.q1, atol=1e-8)
    for d, limit in ((1, velocity), (2, acceleration), (3, jerk)):
        assert np.max(peaks(result, d)) <= limit+1e-8


def test_clock_split_preserves_bends_not_a_refit_of_sampled_joints():
    p, path, goal = chain()
    original = compile(p, path, goal).curve
    law = CubicSpline([0., .9, 2., 2.8], [0., .8, 2.2, 3.], bc_type='natural')
    result = compose_path(original, _split_law(original, law))
    t = np.linspace(0., 2.8, 2001)
    np.testing.assert_allclose(result(t), original(law(t)), atol=2e-11)
    np.testing.assert_allclose(result(t, 1), original(law(t), 1)*law(t, 1)[:, None], atol=2e-10)
    np.testing.assert_allclose(result(t, 2), original(law(t), 2)*law(t, 1)[:, None]**2+
                               original(law(t), 1)*law(t, 2)[:, None], atol=2e-9)
    assert_continuity(result)


def test_optimized_carry_keeps_required_point_and_maps_diagnostics_clock():
    p, path, goal = chain()
    old = compile(p, path, goal)
    result = accelerate_path(p, old, .04, payload=True, cancelled=lambda: False)
    assert result.diagnostics['post_grasp_timing']['applied']
    clock = result.diagnostics['post_grasp_timing']['source_clock']
    law = PPoly(clock['coefficients'], clock['knots_s'])
    times = np.linspace(0., result.duration, 1501)
    np.testing.assert_allclose(result.curve(times), old.curve(law(times)), atol=1e-9)
    np.testing.assert_allclose(result.curve(result.diagnostics['continuous_carry']['lift_end_s']),
                               old.curve(old.diagnostics['continuous_carry']['lift_end_s']), atol=1e-9)
    assert result.diagnostics['continuous_carry']['duration_s'] == result.duration
    assert result.diagnostics['pieces'][-1]['end_s'] == pytest.approx(result.duration)
    assert np.all(peaks(result.curve, 3) <= peaks(old.curve, 3)+1e-8)
    assert_continuity(result.curve)


def test_geometry_failure_keeps_original_curve_and_cancellation_propagates():
    p, path = straight()
    def reject(*a, **kw):
        raise PickRejected('current payload/scene rejected')
    p.check_path = reject
    result = accelerate_path(p, path, .1, payload=False, cancelled=lambda: False)
    assert result.curve is path.curve
    assert not result.diagnostics['post_grasp_timing']['applied']
    assert 'current payload' in result.diagnostics['post_grasp_timing']['reason']
    with pytest.raises(PickRejected, match='取消'):
        accelerate_path(p, path, .1, payload=False, cancelled=lambda: True)


def test_optimizer_checks_cancellation_and_cannot_change_noncollinear_return_to_line():
    p, path, goal = chain()
    curve = compile(p, path, goal).curve
    # Introduce an independent axis bend, which must not be shortcut.
    c = curve.c.copy(); c[-1, :, 1] = np.linspace(0., .1, c.shape[1])
    assert _straight_return(PPoly(c, curve.x), np.ones(7), np.ones(7), np.ones(7)) is None
    p, path = straight();calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 2
    with pytest.raises(PickRejected, match='取消'):
        accelerate_path(p, path, .1, payload=True, cancelled=cancel)


def test_return_template_key_includes_new_timing_and_rechecks_each_scene():
    cfg, p, q = fixture()
    cfg.raw['taught_return_strategy'] = 'direct_to_wait'
    original = direct_return(p, cfg, q(.3), .1, cancelled=lambda: False)
    cfg.raw['post_grasp_retime_enabled'] = True
    result = direct_return(p, cfg, q(.3), .1, cancelled=lambda: False)
    assert p.calls.count('return') == 2
    assert 'post_grasp_timing' in result.diagnostics
    p.identity = (1, 6, 8);p.scene_revision += 1
    again = direct_return(p, cfg, q(.3), .1, cancelled=lambda: False)
    assert p.calls.count('return') == 2
    assert again.identity == p.identity
    assert again.diagnostics['scene_revision'] == p.scene_revision
    assert again.diagnostics['route_template_reused']
    assert p.calls[-1] == ('check', 'return', False)

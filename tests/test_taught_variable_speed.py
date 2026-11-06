"""Path identity, continuous dynamics and complete polynomial constraints."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PPoly

from nero_core.two_second.trajectory import joint_curve, extrema, retime_samples
from nero_core.two_second.taught_timing import compose_path, _time_law, retime_taught_path
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def example():
    times = np.linspace(0., 3., 61)
    u = times/3.
    joints = np.zeros((len(times), 7))
    joints[:, 0] = 10*u**3-15*u**4+6*u**5
    joints[:, 1] = .3*np.sin(np.pi*u)**2
    joints[:, 2] = .001*np.sin(75*times)*np.exp(-((times-.3)/.2)**2)
    motion = SimpleNamespace(velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*2.,
                             settings=SimpleNamespace(control_hz=100.))
    return times, joints, motion


def test_composition_preserves_the_complete_original_curve_and_c2_motion():
    times, joints, motion = example()
    path = joint_curve(times, joints, zero_end_acceleration=True)
    law = _time_law(path, motion, lambda: False)
    emitted = compose_path(path, law)
    samples = np.sort(np.r_[law.x, np.random.default_rng(1).uniform(0., law.x[-1], 1000)])
    np.testing.assert_allclose(emitted(samples), path(law(samples)), atol=1e-11)
    np.testing.assert_allclose(emitted(samples, 1), path(law(samples), 1)*law(samples, 1)[:, None], atol=1e-10)
    expected = path(law(samples), 2)*law(samples, 1)[:, None]**2 + path(law(samples), 1)*law(samples, 2)[:, None]
    np.testing.assert_allclose(emitted(samples, 2), expected, atol=1e-9)
    for derivative in (0, 1, 2):
        left = emitted(np.nextafter(emitted.x[1:-1], -np.inf), derivative)
        right = emitted(emitted.x[1:-1], derivative)
        np.testing.assert_allclose(left, right, atol=1e-8)


def test_variable_speed_is_faster_without_raising_any_joint_limit():
    times, joints, motion = example()
    args = dict(identity=(0, 1, 2), version=3, stage='taught_transfer')
    uniform = retime_samples(times, joints, motion, -np.ones(7)*3, np.ones(7)*3, **args)
    result = retime_taught_path(times, joints, motion, -np.ones(7)*3, np.ones(7)*3, **args)
    assert result.duration < uniform.duration*.9
    assert result.diagnostics['original_path_preserved']
    for derivative, limit in ((1, motion.velocity_limits), (2, motion.acceleration_limits)):
        assert np.all(abs(np.asarray(extrema(result.curve, derivative))) <= limit+1e-9)
        np.testing.assert_allclose(result.curve([0., result.duration], derivative), 0., atol=1e-8)
    np.testing.assert_allclose(result.q0, joints[0], atol=1e-12)
    np.testing.assert_allclose(result.q1, joints[-1], atol=1e-12)


def test_variable_speed_cancellation_cannot_become_a_successful_fallback():
    times, joints, motion = example()
    with pytest.raises(PickRejected, match='取消'):
        retime_taught_path(times, joints, motion, -np.ones(7)*3, np.ones(7)*3,
            identity=(0, 1, 2), version=1, stage='taught_transfer', cancelled=lambda: True)


def test_extrema_keeps_left_limit_at_a_discontinuous_polynomial_knot():
    # The value 10 immediately before the middle knot is otherwise omitted.
    curve = PPoly(np.array([[[10.]*7, [0.]*7], [[0.]*7, [0.]*7]]), [0., 1., 2.])
    low, high = extrema(curve, 0)
    np.testing.assert_allclose(high, 10.)


def test_unknown_taught_timing_policy_is_rejected(site):
    import yaml
    from nero_core.two_second.config import CellConfig
    path, data = site
    data['taught_transfer_timing_policy'] = 'unchecked_fast'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='taught_transfer_timing_policy'):
        CellConfig.load(path)


@pytest.mark.parametrize('key', ['taught_transfer_max_duration_s', 'taught_transfer_jerk_limit_rad_s3'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_transfer_time_and_smoothing_limits_must_be_positive_finite(site, key, value):
    import yaml
    from nero_core.two_second.config import CellConfig
    path,data=site
    data[key]=value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match=key):
        CellConfig.load(path)


from test_two_second_hardware import site

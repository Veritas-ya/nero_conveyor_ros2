import numpy as np
from experiments.two_second_grasp.constant_visual_speed import ConstantVisualSpeed


def test_incremental_fit_matches_batch_and_is_causal():
    fit = ConstantVisualSpeed(min_samples=4, min_span_s=.3)
    identity = (0, 3, 1)
    times = np.array([10., 10.1, 10.23, 10.4, 10.7])
    positions = np.column_stack([2 + .15 * (times-10) + [0, .001, -.001, .002, 0],
                                 -1 - .02 * (times-10), np.ones(len(times))])
    for t, p in zip(times, positions):
        assert fit.update(identity, t, p)
    estimate = fit.estimate(identity)
    for axis in range(2):
        np.testing.assert_allclose(estimate['velocity_world_m_s'][axis],
                                   np.polyfit(times-times[0], positions[:, axis], 1)[0])
    assert not fit.update(identity, times[-1], positions[-1] + 100)
    np.testing.assert_allclose(fit.estimate(identity)['velocity_world_m_s'], estimate['velocity_world_m_s'])


def test_identity_and_clock_epoch_have_independent_observations():
    fit = ConstantVisualSpeed(min_samples=3, min_span_s=.2)
    for stamp in (0., .1, .2):
        fit.update((0, 1, 0), stamp, [stamp, 0., 0.])
    fit.update((1, 1, 0), 0., [100., 0., 0.])
    fit.update((0, 2, 0), 0., [100., 0., 0.])
    assert fit.estimate((1, 1, 0)) is None
    assert fit.estimate((0, 2, 0)) is None
    np.testing.assert_allclose(fit.estimate((0, 1, 0))['velocity_world_m_s'], [1., 0., 0.])

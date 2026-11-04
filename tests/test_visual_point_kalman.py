import numpy as np
from experiments.two_second_grasp.visual_point_kalman import VisualPointKalman


def test_kalman_uses_observed_positions_and_keeps_identities_separate():
    bank = VisualPointKalman(max_velocity=.3)
    key = (0, 1, 0)
    for t in np.linspace(0., 1., 31):
        assert bank.update(key, float(t), [1+.15*t, 2., 0.])
    estimate = bank.estimate(key)
    assert estimate is not None
    assert abs(estimate['velocity_world_m_s'][0]-.15) < .001
    assert estimate['accel_noise_std'] == .002
    assert bank.estimate((1, 1, 0)) is None
    assert not bank.update(key, 1., [99., 99., 0.])

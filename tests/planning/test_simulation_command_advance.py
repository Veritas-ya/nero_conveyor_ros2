"""Fixed 100 ms proposal is simulated, smooth and requires full certification."""
import numpy as np
import pytest
from test_planning_worker_scene_validator import model, make_engine
from test_full_pick_planning import full_request
from experiments.continuous_intercept_sim.command_advance import advance_curves, install_simulation_advance


class Curve:
    def __init__(self, duration, offset):
        self.duration, self.offset = duration, offset
    def sample_full(self, t):
        t = np.clip(t, 0., self.duration)
        return (np.full(7, self.offset + .1*t), np.full(7, .1), np.zeros(7))
    def derivative_bounds(self, order):
        a = np.full(7, .1 if order == 1 else 0.)
        if order == 0:
            return np.full(7, self.offset), np.full(7, self.offset+.1*self.duration)
        return a, a


def test_hundred_ms_proposal_is_smooth_and_fades_before_lift():
    req = full_request(full_approach_mode='continuous')
    approach, closing, lift = Curve(3., 0.), Curve(.6, .3), Curve(1., .36)
    commands, ratio = advance_curves(req, (approach, None, closing, lift), .1)
    assert np.isfinite(ratio)
    for actual, expected in zip(commands[0].sample_full(0.), approach.sample_full(0.)):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert commands[0].sample_full(1.5)[0][0] == pytest.approx(.16, abs=1e-5)
    for left, right in zip(commands[0].sample_full(3.), commands[2].sample_full(0.)):
        np.testing.assert_allclose(left, right, atol=1e-9)
    for actual, expected in zip(commands[2].sample_full(.6), closing.sample_full(.6)):
        np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert commands[3] is lift


def test_uncertified_advance_never_returns_nominal_success(model, monkeypatch):
    engine = make_engine(model)
    original = engine._full_certify_curve
    def reject_advanced(req, curve, start, phase, cancelled, **kwargs):
        if kwargs.get('command'):
            return None
        return original(req, curve, start, phase, cancelled, **kwargs)
    monkeypatch.setattr(engine, '_full_certify_curve', reject_advanced)
    install_simulation_advance(engine, .1)
    result = engine.plan(full_request(full_approach_mode='continuous'), lambda: False)
    assert not result.success and result.full_pick_plan is None
    assert result.error == 'simulation_advance_not_fully_certified'


def test_high_or_prefix_plan_never_generates_uncertified_advance():
    req = full_request(full_approach_mode='high')
    with pytest.raises(ValueError, match='complete continuous'):
        advance_curves(req, (Curve(3., 0.), None, Curve(.6, .3), Curve(1., .36)), .1)

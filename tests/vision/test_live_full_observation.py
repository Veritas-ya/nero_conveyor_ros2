"""Live observation trace uses network and vision inputs, never physical truth."""
from copy import deepcopy
from test_vision_full_bridge import data
from experiments.graspgenx_sim.live_full_observation import LiveFullObservation


def test_network_and_actual_visual_times_reach_live_bridge_and_cancel():
    inputs = data()
    trace = LiveFullObservation(max_observation_age_s=1., max_result_age_s=4.)
    assert trace.accept_network_result(inputs['provenance'], inputs['candidate'], clock_epoch=1)
    out = trace.update(inputs['raw_observation'])
    assert out is not None
    record = trace.events[-1]
    assert record['network_capture_sim_s'] == 2.
    assert record['visual_observed_sim_s'] == 2.8
    assert record['scope'] == 'planner_input_only_not_certified_or_sent'
    assert not trace.accept_network_result(inputs['provenance'], inputs['candidate'], clock_epoch=1)
    stale = deepcopy(inputs['raw_observation']); stale['scene_epoch'] += 1
    assert trace.update(stale) is None and trace.latest is None
    trace.cancel('clock_reset')
    assert trace.update(inputs['raw_observation']) is None


def test_unknown_clock_or_generation_is_never_promoted_to_a_task():
    inputs = data()
    trace = LiveFullObservation(max_observation_age_s=1., max_result_age_s=4.)
    assert not trace.accept_network_result(inputs['provenance'], inputs['candidate'], clock_epoch=None)
    provenance = dict(inputs['provenance'], track_generation=None)
    assert not trace.accept_network_result(provenance, inputs['candidate'], clock_epoch=1)
    assert trace.identity is None


def test_trace_error_is_recorded_and_does_not_replace_physics_action(monkeypatch):
    from types import SimpleNamespace
    from experiments.graspgenx_sim.nero7_full_observation_run import TracedAdapter
    from experiments.graspgenx_sim.continuous import ContinuousGraspAdapter
    def fail(_):
        raise RuntimeError('diagnostic failure')
    trace = SimpleNamespace(update=fail, latest='old', events=[])
    adapter = object.__new__(TracedAdapter)
    adapter.full_observation_trace = trace
    expected = {'arm': [1.], 'gripper': 1.}
    monkeypatch.setattr(ContinuousGraspAdapter, 'act', lambda self, obs: expected)
    assert adapter.act({'timestamp': 3.}) is expected
    assert trace.latest is None and trace.events[0]['event'] == 'trace_error'

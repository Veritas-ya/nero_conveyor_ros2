"""Nero7 + GraspGen-X + physical control with full-observation shadow trace.

This runs the existing physical/legacy controller. Full planning and certified
full trajectory execution are not connected by this diagnostic entry.
"""
from experiments.graspgenx_sim import nero7_run
from experiments.graspgenx_sim.continuous import ContinuousGraspAdapter
from experiments.graspgenx_sim.live_full_observation import LiveFullObservation


class TracedAdapter(ContinuousGraspAdapter):
    def on_network_result(self, provenance, candidate, *, clock_epoch):
        self.full_observation_trace.accept_network_result(provenance, candidate, clock_epoch=clock_epoch)

    def act(self, observation):
        try:
            self.full_observation_trace.update(observation)
        except Exception as error:
            # Diagnostic failure must not change physics ownership or silently
            # become a motion certificate. Keep the error for offline analysis.
            self.full_observation_trace.latest = None
            self.full_observation_trace.events.append({'event': 'trace_error',
                'sim_s': float(observation['timestamp']),
                'error': f'{type(error).__name__}: {error}'})
        return super().act(observation)

    def reset(self, seed=None):
        if hasattr(self, 'full_observation_trace'):
            self.full_observation_trace.cancel('controller_reset')
        return super().reset(seed=seed)


def _adapter(env, **kwargs):
    # Construct the historical policy with the trace-capable adapter.
    adapter = nero7_run._adapter(env, adapter_class=TracedAdapter, **kwargs)
    adapter.full_observation_trace = LiveFullObservation(
        max_observation_age_s=env.config.maximum_track_age_s,
        max_result_age_s=None)
    return adapter


def main(argv=None):
    return nero7_run.main(argv, adapter_type=_adapter)


if __name__ == '__main__':
    raise SystemExit(main())

"""Recent completed planning rounds, independent of device/observation clocks."""
from collections import Counter, deque
import numpy as np


class PlanningTimeHistory:
    def __init__(self, initial_s, *, size=8, quantile=.8):
        self.initial_s = float(initial_s)
        self.samples = deque(maxlen=size)
        self.quantile = quantile
        self.excluded = Counter()

    @property
    def estimate_s(self):
        if not self.samples:
            return self.initial_s
        return float(np.quantile([x['cold_wall_s'] for x in self.samples], self.quantile))

    def observe(self, cold_s, *, outcome, completed, cancelled, components=None):
        if cancelled or not completed:
            self.excluded['cancelled' if cancelled else outcome] += 1
            return False
        self.samples.append(dict(cold_wall_s=float(cold_s), outcome=outcome,
                                 components_s=dict(components or {})))
        return True

    def diagnostics(self):
        return dict(policy='recent_complete', sample_count=len(self.samples),
                    window_size=self.samples.maxlen, quantile=self.quantile,
                    estimate_s=self.estimate_s, samples=list(self.samples),
                    excluded=dict(self.excluded))


def history_for(planner):
    raw = planner.config.raw
    if raw.get('planning_time_estimator', 'maximum') != 'recent_complete':
        return None
    if not hasattr(planner, 'planning_time_history'):
        planner.planning_time_history = PlanningTimeHistory(
            getattr(planner, 'measured_planning_lead_s', raw.get('measured_planning_initial_s', 0.)),
            size=raw.get('planning_time_window_size', 8),
            quantile=raw.get('planning_time_quantile', .8))
    return planner.planning_time_history

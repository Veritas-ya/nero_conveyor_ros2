"""Local, fixed-deadline interception curves; callers certify their geometry.

Real route corners retain zero velocity/acceleration. Endpoint derivatives may
bend a segment, so the original route certificate alone never authorizes it.
"""
from __future__ import annotations

import numpy as np

from .continuous_joint_trajectory import BoundaryQuinticJointTrajectory


class InterceptTimingError(ValueError):
    def __init__(self, message: str, suggested_duration_s: float):
        super().__init__(message)
        # A feasible duration for the old endpoints, not an appointment for a
        # moving object. The caller must predict a new pose and certify again.
        self.suggested_duration_s = float(suggested_duration_s)


class ContinuousInterceptTrajectory:
    """Piecewise C2 quintics with an exact, immutable capture deadline."""

    backend = 'fixed_deadline_quintic'
    requires_recertification = True

    def __init__(self, path, velocity_limits, acceleration_limits, *, duration,
                 initial_velocity=None, initial_acceleration=None,
                 final_velocity=None, final_acceleration=None, jerk_limits=None):
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1] != 7 or not np.isfinite(points).all():
            raise ValueError('intercept path must contain finite seven-joint states')
        if len(points) < 2:
            raise ValueError('intercept path needs start and contact states')
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0.:
            raise ValueError('intercept duration must be finite and positive')
        self.path = tuple(q.copy() for q in points)
        self.q0, self.q1 = points[0].copy(), points[-1].copy()
        segments = []
        for index, (q0, q1) in enumerate(zip(points[:-1], points[1:])):
            segments.append(BoundaryQuinticJointTrajectory(
                q0, q1, velocity_limits, acceleration_limits,
                initial_velocity=initial_velocity if index == 0 else None,
                initial_acceleration=initial_acceleration if index == 0 else None,
                final_velocity=final_velocity if index == len(points)-2 else None,
                final_acceleration=final_acceleration if index == len(points)-2 else None,
                jerk_limits=jerk_limits,
                # A coincident endpoint with a nonzero boundary velocity
                # still needs finite time to brake/return. Seed that case
                # from the appointment instead of a near-zero chord ETA.
                minimum_duration=(duration/(len(points)-1)
                    if np.max(np.abs(q1-q0)) < 1e-12 else 0.),
                allow_path_change=True))
        # Preserve the chosen route's real corners. Allocate the absolute
        # available time proportionally to its bounded local segment times,
        # then verify the exact polynomials; no segment may stretch itself.
        weights = np.maximum([segment.duration for segment in segments], 1e-8)
        suggested = float(np.sum(weights))
        durations = duration * weights / np.sum(weights)
        durations[-1] = duration - float(np.sum(durations[:-1]))
        for segment, edge_duration in zip(segments, durations):
            coefficients = segment._coefficients_at_duration(float(edge_duration))
            if not segment._within_limits(coefficients, float(edge_duration)):
                raise InterceptTimingError(
                    'fixed capture time violates joint velocity, acceleration or jerk', suggested)
            segment.duration = float(edge_duration)
            segment._coefficients = coefficients
            # The worker checks the full new curve; never reuse a chord-only
            # geometry claim after changing the duration/boundary derivatives.
            segment.geometry_preserved = False
        self.segments = tuple(segments)
        self.edge_durations = tuple(float(value) for value in durations)
        self._ends = np.cumsum(durations)
        self.duration = duration
        self.geometry_preserved = False

    def sample_full(self, time):
        value = float(time)
        if not np.isfinite(value):
            raise ValueError('sample time must be finite')
        value = float(np.clip(value, 0., self.duration))
        index = min(int(np.searchsorted(self._ends, value, side='right')),
                    len(self.segments)-1)
        start = 0. if index == 0 else float(self._ends[index-1])
        return self.segments[index].sample_full(value-start)

    def sample(self, time):
        q, v, _ = self.sample_full(time)
        return q, v

    def derivative_bounds(self, order):
        bounds = [segment.derivative_bounds(order) for segment in self.segments]
        return (np.min([item[0] for item in bounds], axis=0),
                np.max([item[1] for item in bounds], axis=0))

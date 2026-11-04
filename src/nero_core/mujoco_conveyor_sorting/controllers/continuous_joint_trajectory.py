"""Local continuous trajectory candidates for an already checked joint path.

These are *candidates*, not collision certificates.  Changing a trajectory's
timing, boundary state, or geometry requires the caller's original certification
and moving-target appointment calculation.  The module never contacts hardware
or the Ruckig waypoint/cloud API.  Missing optional Ruckig uses a local quintic.

Exact corners and reversals retain zero velocity and acceleration.  Removing an
ordinary same-direction collinear sample does not require stopping there.  A
nonzero endpoint state can be retained along a chord only when its derivatives
are tangent to that chord; an explicit ``allow_path_change`` enables a general
seven-axis candidate that must receive a new geometric certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Sequence

import numpy as np

from .joint_trajectory import ARM_DOF, QuinticJointTrajectory, _joint_vector, _positive_limits


def _boundary(value: Any | None, name: str) -> np.ndarray:
    return np.zeros(ARM_DOF) if value is None else _joint_vector(value, name)


def _real_roots(coefficients: np.ndarray) -> list[float]:
    """Interior real roots of an ascending-order scalar polynomial."""
    values = np.trim_zeros(np.asarray(coefficients, dtype=float), "b")
    if len(values) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(values)
    return [float(r.real) for r in roots if abs(r.imag) <= 1e-9 and 0.0 < r.real < 1.0]


def _derivative(coefficients: np.ndarray, order: int) -> np.ndarray:
    return np.polynomial.polynomial.polyder(coefficients, m=order, axis=0)


def _extrema(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic polynomial extrema, not a sampled limit estimate."""
    lower, upper = [], []
    for joint in range(coefficients.shape[1]):
        column = coefficients[:, joint]
        candidates = [0.0, 1.0, *_real_roots(_derivative(column, 1))]
        values = np.polynomial.polynomial.polyval(candidates, column)
        lower.append(float(np.min(values)))
        upper.append(float(np.max(values)))
    return np.asarray(lower), np.asarray(upper)


def _tangent_scalar(vector: np.ndarray, delta: np.ndarray) -> float | None:
    squared = float(np.dot(delta, delta))
    if squared <= 1e-24:
        return 0.0 if np.max(np.abs(vector)) <= 1e-12 else None
    scalar = float(np.dot(delta, vector) / squared)
    return scalar if np.max(np.abs(vector - delta * scalar)) <= 2e-11 else None


class BoundaryQuinticJointTrajectory:
    """Quintic with specified q/v/a endpoints and exact derivative bounds.

    With stationary boundaries this is the existing synchronized quintic,
    optionally lengthened to enforce a supplied jerk limit.  No jerk value is
    invented when the hardware value is unknown.  General endpoint derivatives
    can bend the joint-space path, so the caller must certify that new curve.
    """

    backend = "native_quintic"
    requires_recertification = True

    def __init__(
        self, q0: Any, q1: Any, velocity_limits: Any, acceleration_limits: Any,
        *, initial_velocity: Any | None = None, initial_acceleration: Any | None = None,
        final_velocity: Any | None = None, final_acceleration: Any | None = None,
        jerk_limits: Any | None = None, minimum_duration: float = 0.0,
        allow_path_change: bool = False,
    ) -> None:
        self.q0, self.q1 = _joint_vector(q0, "q0"), _joint_vector(q1, "q1")
        self.velocity_limits = _positive_limits(velocity_limits, "velocity_limits")
        self.acceleration_limits = _positive_limits(acceleration_limits, "acceleration_limits")
        self.jerk_limits = None if jerk_limits is None else _positive_limits(jerk_limits, "jerk_limits")
        self.initial_velocity = _boundary(initial_velocity, "initial_velocity")
        self.initial_acceleration = _boundary(initial_acceleration, "initial_acceleration")
        self.final_velocity = _boundary(final_velocity, "final_velocity")
        self.final_acceleration = _boundary(final_acceleration, "final_acceleration")
        minimum = float(minimum_duration)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_duration must be finite and non-negative")
        for state, limit in ((self.initial_velocity, self.velocity_limits),
                             (self.final_velocity, self.velocity_limits),
                             (self.initial_acceleration, self.acceleration_limits),
                             (self.final_acceleration, self.acceleration_limits)):
            if np.any(np.abs(state) > limit + 1e-12):
                raise ValueError("boundary derivatives exceed supplied limits")
        self._delta = self.q1 - self.q0
        tangent = [_tangent_scalar(v, self._delta) for v in (
            self.initial_velocity, self.initial_acceleration,
            self.final_velocity, self.final_acceleration)]
        if not allow_path_change and any(v is None for v in tangent):
            raise ValueError("nonzero endpoint derivatives are not tangent to the checked chord")
        stationary_boundary = not any(np.any(v != 0.0) for v in (
            self.initial_velocity, self.initial_acceleration,
            self.final_velocity, self.final_acceleration))
        original = QuinticJointTrajectory(self.q0, self.q1, self.velocity_limits,
                                         self.acceleration_limits, minimum_duration=minimum)
        duration = original.duration
        if self.jerk_limits is not None:
            duration = max(duration, float(np.max(np.cbrt(60.0 * np.abs(self._delta) / self.jerk_limits))))
        if duration == 0.0 and stationary_boundary:
            self.duration = 0.0
            self._coefficients = np.vstack((self.q0, np.zeros((5, ARM_DOF))))
            self.geometry_preserved = True
            return
        duration = max(duration, 1e-8)
        for _ in range(160):
            coefficients = self._coefficients_at_duration(duration)
            if self._within_limits(coefficients, duration):
                minimum_position, maximum_position = _extrema(coefficients)
                on_chord = all(v is not None for v in tangent) and bool(
                    np.all(minimum_position >= np.minimum(self.q0, self.q1) - 1e-10)
                    and np.all(maximum_position <= np.maximum(self.q0, self.q1) + 1e-10))
                if allow_path_change or on_chord:
                    self.duration = duration
                    self._coefficients = coefficients
                    self.geometry_preserved = on_chord
                    return
            duration *= 1.075
        raise ValueError("no bounded local quintic found for these endpoint derivatives")

    def _coefficients_at_duration(self, duration: float) -> np.ndarray:
        c0 = self.q0
        c1 = self.initial_velocity * duration
        c2 = self.initial_acceleration * duration**2 / 2.0
        p = self.q1 - c0 - c1 - c2
        v = self.final_velocity * duration - c1 - 2.0 * c2
        a = self.final_acceleration * duration**2 - 2.0 * c2
        return np.vstack((c0, c1, c2, 10*p-4*v+a/2,
                          -15*p+7*v-a, 6*p-3*v+a/2))

    def _within_limits(self, coefficients: np.ndarray, duration: float) -> bool:
        for order, limit in ((1, self.velocity_limits), (2, self.acceleration_limits),
                             (3, self.jerk_limits)):
            if limit is None:
                continue
            lower, upper = _extrema(_derivative(coefficients, order) / duration**order)
            if np.any(np.maximum(np.abs(lower), np.abs(upper)) > limit + 1e-10):
                return False
        return True

    @property
    def is_stationary(self) -> bool:
        return bool(np.all(self._delta == 0.0) and not np.any(self.initial_velocity)
                    and not np.any(self.final_velocity) and not np.any(self.initial_acceleration)
                    and not np.any(self.final_acceleration))

    def sample_full(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamp = float(time)
        if not np.isfinite(timestamp):
            raise ValueError("sample time must be finite")
        if self.duration == 0.0:
            return self.q1.copy(), np.zeros(ARM_DOF), np.zeros(ARM_DOF)
        if timestamp <= 0.:
            return self.q0.copy(), self.initial_velocity.copy(), self.initial_acceleration.copy()
        if timestamp >= self.duration:
            return self.q1.copy(), self.final_velocity.copy(), self.final_acceleration.copy()
        u = float(np.clip(timestamp / self.duration, 0.0, 1.0))
        return tuple(np.asarray(np.polynomial.polynomial.polyval(
            u, _derivative(self._coefficients, order))) / self.duration**order
            for order in range(3))  # type: ignore[return-value]

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        position, velocity, _ = self.sample_full(time)
        return position, velocity

    def derivative_bounds(self, order: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact component bounds used to preserve a caller's sample density."""
        if order not in (0, 1, 2, 3):
            raise ValueError("derivative order must be 0, 1, 2 or 3")
        if self.duration == 0.:
            return (self.q0.copy(), self.q0.copy()) if order == 0 else (np.zeros(7), np.zeros(7))
        return _extrema(_derivative(self._coefficients, order) / self.duration**order)

    def time_at_progress(self, progress: float) -> float:
        """Inverse time law along a preserved chord, including zero endpoints."""
        if not self.geometry_preserved:
            raise ValueError("a bent trajectory has no single chord progress")
        value = float(progress)
        if not np.isfinite(value):
            raise ValueError("progress must be finite")
        if value <= 0.0 or self.duration == 0.0:
            return 0.0
        if value >= 1.0:
            return self.duration
        norm = float(np.dot(self._delta, self._delta))
        polynomial = self._coefficients @ self._delta / norm
        polynomial[0] -= float(np.dot(self.q0, self._delta)/norm) + value
        roots = _real_roots(polynomial)
        if not roots:
            raise ValueError("scalar trajectory does not cross the requested progress")
        return min(roots) * self.duration


class LocalRuckigJointTrajectory:
    """Local Ruckig state-to-state generation; no intermediate waypoint call."""

    backend = "ruckig_local"
    requires_recertification = True

    def __init__(
        self, q0: Any, q1: Any, velocity_limits: Any, acceleration_limits: Any,
        *, jerk_limits: Any, initial_velocity: Any | None = None,
        initial_acceleration: Any | None = None, final_velocity: Any | None = None,
        final_acceleration: Any | None = None, minimum_duration: float = 0.0,
        allow_path_change: bool = False,
    ) -> None:
        ruckig = importlib.import_module("ruckig")
        self.q0, self.q1 = _joint_vector(q0, "q0"), _joint_vector(q1, "q1")
        self.velocity_limits = _positive_limits(velocity_limits, "velocity_limits")
        self.acceleration_limits = _positive_limits(acceleration_limits, "acceleration_limits")
        self.jerk_limits = _positive_limits(jerk_limits, "jerk_limits")
        boundaries = [_boundary(value, name) for value, name in (
            (initial_velocity, "initial_velocity"), (initial_acceleration, "initial_acceleration"),
            (final_velocity, "final_velocity"), (final_acceleration, "final_acceleration"))]
        self._delta = self.q1 - self.q0
        tangents = [_tangent_scalar(state, self._delta) for state in boundaries]
        self._scalar = all(s is not None for s in tangents) and np.any(self._delta != 0.0)
        if not self._scalar and not allow_path_change:
            raise ValueError("endpoint derivatives require a newly checked joint-space curve")
        count = 1 if self._scalar else ARM_DOF
        inp = ruckig.InputParameter(count)
        inp.current_position = [0.0] if self._scalar else self.q0.tolist()
        inp.target_position = [1.0] if self._scalar else self.q1.tolist()
        for key, values, scalar in zip(("current_velocity", "current_acceleration",
                                       "target_velocity", "target_acceleration"), boundaries, tangents):
            setattr(inp, key, [float(scalar)] if self._scalar else values.tolist())
        active = np.abs(self._delta) > 1e-15
        for key, limit in (("max_velocity", self.velocity_limits),
                           ("max_acceleration", self.acceleration_limits),
                           ("max_jerk", self.jerk_limits)):
            value = [float(np.min(limit[active] / np.abs(self._delta[active])))] if self._scalar else limit.tolist()
            setattr(inp, key, value)
        minimum = float(minimum_duration)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_duration must be finite and non-negative")
        if minimum:
            inp.minimum_duration = minimum
        self._trajectory = ruckig.Trajectory(count)
        result = ruckig.Ruckig(count).calculate(inp, self._trajectory)
        if int(result) < 0:
            raise ValueError(f"local Ruckig could not generate this boundary state: {result}")
        self.duration = float(self._trajectory.duration)
        self.geometry_preserved = bool(self._scalar)
        if self._scalar:
            extrema = self._trajectory.position_extrema[0]
            self.geometry_preserved = extrema.min >= -1e-10 and extrema.max <= 1.0 + 1e-10
            if not self.geometry_preserved and not allow_path_change:
                raise ValueError("local Ruckig leaves the checked chord")
        self._rebuild = dict(q0=self.q0.copy(), q1=self.q1.copy(),
            velocity_limits=self.velocity_limits.copy(), acceleration_limits=self.acceleration_limits.copy(),
            jerk_limits=self.jerk_limits.copy(), initial_velocity=boundaries[0].copy(),
            initial_acceleration=boundaries[1].copy(), final_velocity=boundaries[2].copy(),
            final_acceleration=boundaries[3].copy(), minimum_duration=minimum,
            allow_path_change=allow_path_change)

    def __deepcopy__(self, memo: dict[int, Any]) -> "LocalRuckigJointTrajectory":
        # pybind's native Trajectory is intentionally not pickleable.  The
        # numeric sender owns a rebuilt local trajectory, not a shared native
        # object or an accidental cloud request.
        rebuilt = type(self)(**self._rebuild)
        for key in ("fallback_reason", "jerk_limits_source"):
            if hasattr(self, key):
                setattr(rebuilt, key, getattr(self, key))
        memo[id(self)] = rebuilt
        return rebuilt

    @property
    def is_stationary(self) -> bool:
        return self.duration == 0.0

    def sample_full(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamp = float(time)
        if not np.isfinite(timestamp):
            raise ValueError("sample time must be finite")
        p, v, a = self._trajectory.at_time(float(np.clip(timestamp, 0.0, self.duration)))
        if self._scalar:
            return self.q0 + self._delta*p[0], self._delta*v[0], self._delta*a[0]
        return np.asarray(p), np.asarray(v), np.asarray(a)

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        position, velocity, _ = self.sample_full(time)
        return position, velocity

    def time_at_progress(self, progress: float) -> float:
        """Invert Ruckig's own scalar time law instead of quintic smoothstep."""
        if not self.geometry_preserved:
            raise ValueError("a bent trajectory has no single chord progress")
        value = float(progress)
        if not np.isfinite(value):
            raise ValueError("progress must be finite")
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return self.duration
        result = self._trajectory.get_first_time_at_position(0, value)
        if result is not None and abs(self._trajectory.at_time(float(result))[0][0]-value) <= 1e-10:
            return float(result)
        # The 0.12.2 Python wheel's inverse binding can return 0 for an
        # interior position.  Invert its actual local time law instead.
        lower, upper = 0.0, self.duration
        for _ in range(48):
            middle = (lower+upper)/2
            if self._trajectory.at_time(middle)[0][0] < value:
                lower = middle
            else:
                upper = middle
        return (lower+upper)/2


@dataclass(frozen=True)
class TrajectoryBuildResult:
    trajectory: Any | None
    backend: str
    geometry_preserved: bool
    requires_recertification: bool
    fallback_reason: str | None = None


def build_joint_trajectory(
    q0: Any, q1: Any, velocity_limits: Any, acceleration_limits: Any, *,
    backend: str = "auto", jerk_limits: Any | None = None, **kwargs: Any,
) -> TrajectoryBuildResult:
    """Return a candidate or a diagnostic so callers retain their old flow.

    ``auto`` uses local Ruckig only with explicit jerk limits.  Without those
    limits it uses the existing quintic family; unknown limits never abort a
    mission and are never silently replaced by empirical constants.
    """
    if backend not in {"auto", "native", "ruckig"}:
        return TrajectoryBuildResult(None, "unavailable", False, False, "unknown trajectory backend")
    fallback = None
    if backend != "native" and jerk_limits is not None:
        try:
            trajectory = LocalRuckigJointTrajectory(q0, q1, velocity_limits,
                acceleration_limits, jerk_limits=jerk_limits, **kwargs)
            return TrajectoryBuildResult(trajectory, trajectory.backend,
                                          trajectory.geometry_preserved, True)
        except (ImportError, ValueError, RuntimeError) as error:
            fallback = str(error)
    elif backend == "ruckig":
        fallback = "jerk limits are unknown; keeping local quintic generation"
    try:
        trajectory = BoundaryQuinticJointTrajectory(q0, q1, velocity_limits,
            acceleration_limits, jerk_limits=jerk_limits, **kwargs)
        return TrajectoryBuildResult(trajectory, trajectory.backend,
                                      trajectory.geometry_preserved, True, fallback)
    except (TypeError, ValueError, RuntimeError) as error:
        reason = str(error) if fallback is None else f"{fallback}; {error}"
        return TrajectoryBuildResult(None, "unavailable", False, False, reason)


def build_certified_edge_trajectory(
    q0: Any, q1: Any, velocity_limits: Any, acceleration_limits: Any, *,
    minimum_duration: float = 0.0, backend: str = "quintic",
    jerk_limits: Any | None = None,
) -> Any:
    """Shared worker/executor factory for a newly certified straight edge.

    The name describes its intended caller: it does not certify collisions.
    A Ruckig candidate uses a single scalar along exactly the original chord.
    Unknown jerk or missing local dependency retains the original trajectory;
    metadata explains that fallback without creating a new execution gate.
    """
    if backend == "ruckig" and jerk_limits is not None:
        result = build_joint_trajectory(q0, q1, velocity_limits, acceleration_limits,
            minimum_duration=minimum_duration, backend="ruckig", jerk_limits=jerk_limits)
        if result.trajectory is not None and result.backend == "ruckig_local":
            result.trajectory.fallback_reason = result.fallback_reason
            result.trajectory.jerk_limits_source = "explicit_host_profile"
            return result.trajectory
        fallback = result.fallback_reason or "local Ruckig unavailable"
    elif backend == "ruckig":
        fallback = "jerk limits are unknown; retaining original quintic"
    elif backend == "quintic":
        fallback = None
    else:
        fallback = f"unknown optional backend {backend!r}; retaining original quintic"
    trajectory = QuinticJointTrajectory(q0, q1, velocity_limits, acceleration_limits,
                                         minimum_duration=minimum_duration)
    trajectory.backend = "quintic"
    trajectory.geometry_preserved = True
    trajectory.requires_recertification = True
    trajectory.fallback_reason = fallback
    trajectory.jerk_limits_source = "unknown"
    return trajectory


class ContinuousJointPathTrajectory:
    """Piecewise checked chords; true corners stop, collinear samples do not."""

    requires_recertification = True

    def __init__(self, path: Sequence[Any], velocity_limits: Any, acceleration_limits: Any,
                 *, jerk_limits: Any | None = None, initial_velocity: Any | None = None,
                 initial_acceleration: Any | None = None, final_velocity: Any | None = None,
                 final_acceleration: Any | None = None, minimum_edge_durations: Sequence[float] | None = None,
                 backend: str = "auto", allow_path_change: bool = False) -> None:
        states = np.asarray(path, dtype=float)
        if states.ndim != 2 or states.shape[1:] != (ARM_DOF,) or len(states) < 2 or not np.all(np.isfinite(states)):
            raise ValueError("path must contain at least two finite seven-joint states")
        mins = np.zeros(len(states)-1) if minimum_edge_durations is None else np.asarray(minimum_edge_durations, dtype=float)
        if mins.shape != (len(states)-1,) or np.any(mins < 0) or not np.all(np.isfinite(mins)):
            raise ValueError("minimum_edge_durations must match the original path edges")
        # Collinear compression only removes samples within floating point
        # roundoff.  No circular blend or direction tolerance changes a corner.
        indices = [0]
        for index in range(1, len(states)-1):
            incoming, outgoing = states[index]-states[indices[-1]], states[index+1]-states[index]
            norm = float(np.linalg.norm(incoming))
            if norm <= 1e-14:
                continue
            projection = float(np.dot(outgoing, incoming) / norm**2)
            residual = np.linalg.norm(outgoing-projection*incoming)
            if projection < 0.0 or residual > 2e-13 * max(1.0, norm, float(np.linalg.norm(outgoing))):
                indices.append(index)
        indices.append(len(states)-1)
        self.path = states[indices].copy()
        self.original_indices = tuple(indices)
        self.segments = []
        self.fallback_reasons = []
        for edge, (start, end) in enumerate(zip(self.path, self.path[1:])):
            result = build_joint_trajectory(start, end, velocity_limits, acceleration_limits,
                jerk_limits=jerk_limits, backend=backend,
                initial_velocity=initial_velocity if edge == 0 else None,
                initial_acceleration=initial_acceleration if edge == 0 else None,
                final_velocity=final_velocity if edge == len(self.path)-2 else None,
                final_acceleration=final_acceleration if edge == len(self.path)-2 else None,
                minimum_duration=float(np.sum(mins[indices[edge]:indices[edge+1]])),
                allow_path_change=allow_path_change)
            if result.trajectory is None:
                raise ValueError(result.fallback_reason)
            self.segments.append(result.trajectory)
            if result.fallback_reason:
                self.fallback_reasons.append(result.fallback_reason)
        self.edge_durations = tuple(float(t.duration) for t in self.segments)
        self._end_times = np.cumsum(self.edge_durations)
        self.duration = float(self._end_times[-1])
        self.q0, self.q1 = self.path[0].copy(), self.path[-1].copy()
        self.geometry_preserved = all(t.geometry_preserved for t in self.segments)
        self.backend = "+".join(sorted({t.backend for t in self.segments}))

    def sample_full(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamp = float(time)
        if not np.isfinite(timestamp):
            raise ValueError("sample time must be finite")
        edge = min(int(np.searchsorted(self._end_times, timestamp, side="right")), len(self.segments)-1)
        start = 0.0 if edge == 0 else self._end_times[edge-1]
        return self.segments[edge].sample_full(timestamp-start)

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        position, velocity, _ = self.sample_full(time)
        return position, velocity


def build_continuous_joint_trajectory(path: Sequence[Any], velocity_limits: Any,
                                      acceleration_limits: Any, **kwargs: Any) -> TrajectoryBuildResult:
    try:
        trajectory = ContinuousJointPathTrajectory(path, velocity_limits, acceleration_limits, **kwargs)
        return TrajectoryBuildResult(trajectory, trajectory.backend, trajectory.geometry_preserved, True,
                                     "; ".join(trajectory.fallback_reasons) or None)
    except (TypeError, ValueError, RuntimeError) as error:
        return TrajectoryBuildResult(None, "unavailable", False, False, str(error))


__all__ = ["BoundaryQuinticJointTrajectory", "ContinuousJointPathTrajectory", "LocalRuckigJointTrajectory",
           "TrajectoryBuildResult", "build_certified_edge_trajectory", "build_continuous_joint_trajectory", "build_joint_trajectory"]

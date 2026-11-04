"""Time-parameterized joint trajectories for seven-axis manipulators.

The trajectory uses the normalized quintic smoothstep

``s(u) = 10 u^3 - 15 u^4 + 6 u^5``

with ``u = t / duration``.  Position, velocity, and acceleration are continuous,
and velocity and acceleration are zero at both endpoints.  A single synchronized
duration is chosen analytically so every joint respects its configured velocity
and acceleration limits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


ARM_DOF = 7
_PEAK_NORMALIZED_VELOCITY = 15.0 / 8.0
_PEAK_NORMALIZED_ACCELERATION = 10.0 / math.sqrt(3.0)


def sampled_tcp_angular_minimum_duration(
    q0: Any,
    q1: Any,
    rotational_jacobian_at: Callable[[npt.NDArray[np.float64]], Any],
    maximum_angular_speed: float,
    *,
    samples: int = 33,
    safety_factor: float = 1.05,
) -> float:
    """Estimate a safe quintic duration from the tool angular Jacobian.

    For the synchronized joint interpolation ``q(u)=q0+dq*s(u)``, tool
    angular speed is ``Jw(q(u))*dq*s'(u)/T``.  Sampling that one-dimensional
    curve is far less conservative than throttling every joint independently,
    while a small multiplicative reserve covers curvature between samples.
    """

    start = _joint_vector(q0, "q0")
    end = _joint_vector(q1, "q1")
    try:
        speed_limit = float(maximum_angular_speed)
        reserve = float(safety_factor)
        count = int(samples)
    except (TypeError, ValueError) as error:
        raise ValueError("angular trajectory limits must be numeric") from error
    if not np.isfinite(speed_limit) or speed_limit <= 0.0:
        raise ValueError("maximum_angular_speed must be finite and positive")
    if not np.isfinite(reserve) or reserve < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    if count < 3:
        raise ValueError("samples must be at least three")

    delta = end - start
    if np.max(np.abs(delta)) <= 1e-12:
        return 0.0
    maximum_coefficient = 0.0
    for normalized_time in np.linspace(0.0, 1.0, count):
        u = float(normalized_time)
        smoothstep = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        smoothstep_velocity = 30.0 * u**2 * (1.0 - u) ** 2
        jacobian = np.asarray(
            rotational_jacobian_at(start + delta * smoothstep),
            dtype=float,
        )
        if jacobian.shape == (6, ARM_DOF):
            jacobian = jacobian[3:]
        if jacobian.shape != (3, ARM_DOF) or not np.all(np.isfinite(jacobian)):
            raise ValueError("rotational jacobian must have shape (3, 7) or (6, 7)")
        coefficient = float(
            np.linalg.norm(jacobian @ delta) * smoothstep_velocity
        )
        maximum_coefficient = max(maximum_coefficient, coefficient)
    return reserve * maximum_coefficient / speed_limit


def _joint_vector(value: Any, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape != (ARM_DOF,):
        raise ValueError(f"{name} must have shape ({ARM_DOF},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _positive_limits(value: Any, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(ARM_DOF, float(result), dtype=float)
    if result.shape != (ARM_DOF,):
        raise ValueError(
            f"{name} must be a scalar or have shape ({ARM_DOF},), "
            f"got {result.shape}"
        )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite, strictly positive values")
    return result.copy()


class QuinticJointTrajectory:
    """A synchronized point-to-point quintic trajectory for a 7-DoF arm.

    Args:
        q0: Initial joint positions, shape ``(7,)``.
        q1: Final joint positions, shape ``(7,)``.
        velocity_limits: Positive scalar or per-joint limits in rad/s.
        acceleration_limits: Positive scalar or per-joint limits in rad/s².
        minimum_duration: Optional lower bound on the generated duration.

    ``sample(t)`` clamps time to the trajectory interval and returns ``(q, qd)``.
    Use :meth:`sample_full` when acceleration is also useful.
    """

    def __init__(
        self,
        q0: Any,
        q1: Any,
        velocity_limits: Any,
        acceleration_limits: Any,
        *,
        minimum_duration: float = 0.0,
    ) -> None:
        self._q0 = _joint_vector(q0, "q0")
        self._q1 = _joint_vector(q1, "q1")
        self._velocity_limits = _positive_limits(
            velocity_limits, "velocity_limits"
        )
        self._acceleration_limits = _positive_limits(
            acceleration_limits, "acceleration_limits"
        )
        try:
            minimum = float(minimum_duration)
        except (TypeError, ValueError) as error:
            raise ValueError("minimum_duration must be a finite non-negative value") from error
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_duration must be a finite non-negative value")

        self._delta = self._q1 - self._q0
        absolute_delta = np.abs(self._delta)
        velocity_duration = (
            _PEAK_NORMALIZED_VELOCITY
            * absolute_delta
            / self._velocity_limits
        )
        acceleration_duration = np.sqrt(
            _PEAK_NORMALIZED_ACCELERATION
            * absolute_delta
            / self._acceleration_limits
        )
        self._duration = max(
            minimum,
            float(np.max(velocity_duration)),
            float(np.max(acceleration_duration)),
        )

    @property
    def duration(self) -> float:
        """Synchronized trajectory duration in seconds."""

        return self._duration

    @property
    def q0(self) -> npt.NDArray[np.float64]:
        return self._q0.copy()

    @property
    def q1(self) -> npt.NDArray[np.float64]:
        return self._q1.copy()

    @property
    def velocity_limits(self) -> npt.NDArray[np.float64]:
        return self._velocity_limits.copy()

    @property
    def acceleration_limits(self) -> npt.NDArray[np.float64]:
        return self._acceleration_limits.copy()

    @property
    def is_stationary(self) -> bool:
        """Whether initial and final joint configurations are identical."""

        return bool(np.all(self._delta == 0.0))

    def _normalized_time(self, time: float) -> float:
        try:
            value = float(time)
        except (TypeError, ValueError) as error:
            raise ValueError("sample time must be a finite scalar") from error
        if not np.isfinite(value):
            raise ValueError("sample time must be a finite scalar")
        if self._duration == 0.0:
            return 1.0
        return float(np.clip(value / self._duration, 0.0, 1.0))

    def sample(
        self, time: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return joint position and velocity at ``time``.

        Times before zero produce the initial stationary state; times after the
        duration produce the final stationary state.
        """

        position, velocity, _ = self.sample_full(time)
        return position, velocity

    def sample_full(
        self, time: float
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return joint position, velocity, and acceleration at ``time``."""

        normalized_time = self._normalized_time(time)
        if self._duration == 0.0:
            zeros = np.zeros(ARM_DOF, dtype=float)
            return self._q1.copy(), zeros.copy(), zeros

        u = normalized_time
        smoothstep = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        smoothstep_velocity = 30.0 * u**2 * (1.0 - u) ** 2
        smoothstep_acceleration = 60.0 * u - 180.0 * u**2 + 120.0 * u**3

        position = self._q0 + self._delta * smoothstep
        velocity = self._delta * smoothstep_velocity / self._duration
        acceleration = (
            self._delta
            * smoothstep_acceleration
            / (self._duration * self._duration)
        )
        return position, velocity, acceleration


# Descriptive aliases for callers that prefer capability-oriented names.
TimeParameterizedJointTrajectory = QuinticJointTrajectory
JointTrajectory7DoF = QuinticJointTrajectory


__all__ = [
    "ARM_DOF",
    "JointTrajectory7DoF",
    "QuinticJointTrajectory",
    "TimeParameterizedJointTrajectory",
    "sampled_tcp_angular_minimum_duration",
]

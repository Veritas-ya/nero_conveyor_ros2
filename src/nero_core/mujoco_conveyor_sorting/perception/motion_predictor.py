"""Deterministic motion-prediction baselines for a moving conveyor.

The predictors deliberately operate on timestamped position history rather than
MuJoCo state.  This makes it difficult to accidentally leak the current truth
pose into the delayed perception path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp, sqrt
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from .observation_buffer import ObservationBuffer, TimestampedState


class MotionModel(str, Enum):
    """Supported deterministic extrapolation models."""

    CONSTANT_VELOCITY = "constant_velocity"
    WINDOWED_VELOCITY = "windowed_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"

    @classmethod
    def parse(cls, value: "MotionModel | str") -> "MotionModel":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "cv": cls.CONSTANT_VELOCITY,
            "velocity": cls.CONSTANT_VELOCITY,
            "constant": cls.CONSTANT_VELOCITY,
            "sliding": cls.WINDOWED_VELOCITY,
            "sliding_window": cls.WINDOWED_VELOCITY,
            "window": cls.WINDOWED_VELOCITY,
            "ca": cls.CONSTANT_ACCELERATION,
            "acceleration": cls.CONSTANT_ACCELERATION,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(model.value for model in cls)
            raise ValueError(
                f"unknown motion model {value!r}; expected one of {choices}"
            ) from exc


@dataclass(frozen=True, slots=True)
class MotionSample:
    timestamp: float
    position: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite_scalar(self.timestamp, "timestamp"))
        object.__setattr__(self, "position", _vector3(self.position, "position"))


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Motion-prediction output expressed in world coordinates."""

    observed_timestamp: float
    decision_timestamp: float
    grasp_timestamp: float
    observed_position: npt.NDArray[np.float64]
    current_estimated_position: npt.NDArray[np.float64]
    predicted_position: npt.NDArray[np.float64]
    estimated_velocity: npt.NDArray[np.float64]
    estimated_acceleration: npt.NDArray[np.float64]
    uncertainty: npt.NDArray[np.float64]
    confidence: float
    model: MotionModel
    motion_compensation_enabled: bool
    in_grasp_window: bool | None = None
    object_id: str | None = None
    sample_count: int = 0
    fit_rmse: float = 0.0
    valid: bool = True
    reason: str = "ok"

    def __post_init__(self) -> None:
        for name in (
            "observed_position",
            "current_estimated_position",
            "predicted_position",
            "estimated_velocity",
            "estimated_acceleration",
            "uncertainty",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))
        object.__setattr__(
            self, "confidence", float(np.clip(self.confidence, 0.0, 1.0))
        )

    @property
    def horizon(self) -> float:
        return self.grasp_timestamp - self.observed_timestamp

    @property
    def arrival_horizon(self) -> float:
        return self.grasp_timestamp - self.decision_timestamp

    @property
    def predicted_grasp_pose(self) -> npt.NDArray[np.float64]:
        """Alias used by the environment observation schema."""

        return self.predicted_position

    @property
    def future_position(self) -> npt.NDArray[np.float64]:
        return self.predicted_position

    @property
    def current_position_estimate(self) -> npt.NDArray[np.float64]:
        return self.current_estimated_position

    @property
    def current_estimated_velocity(self) -> npt.NDArray[np.float64]:
        elapsed = self.decision_timestamp - self.observed_timestamp
        return self.estimated_velocity + self.estimated_acceleration * elapsed

    @property
    def is_within_grasp_window(self) -> bool | None:
        return self.in_grasp_window

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "observed_timestamp": self.observed_timestamp,
            "decision_timestamp": self.decision_timestamp,
            "grasp_timestamp": self.grasp_timestamp,
            "observed_position": self.observed_position.copy(),
            "current_estimated_position": self.current_estimated_position.copy(),
            "predicted_grasp_pose": self.predicted_position.copy(),
            "estimated_velocity": self.estimated_velocity.copy(),
            "estimated_acceleration": self.estimated_acceleration.copy(),
            "uncertainty": self.uncertainty.copy(),
            "confidence": self.confidence,
            "model": self.model.value,
            "motion_compensation_enabled": self.motion_compensation_enabled,
            "in_grasp_window": self.in_grasp_window,
            "sample_count": self.sample_count,
            "fit_rmse": self.fit_rmse,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PredictionError:
    """Error measured when the gripper actually closes."""

    predicted_position: npt.NDArray[np.float64]
    actual_position: npt.NDArray[np.float64]
    error_vector: npt.NDArray[np.float64]
    euclidean_error: float
    timestamp: float | None = None
    object_id: str | None = None

    @property
    def norm(self) -> float:
        return self.euclidean_error

    @property
    def error(self) -> float:
        return self.euclidean_error


@dataclass(slots=True)
class PredictionErrorStatistics:
    """Online prediction-error accumulator (all distances in metres)."""

    count: int = 0
    _sum: float = 0.0
    _sum_squared: float = 0.0
    _maximum: float = 0.0
    _sum_vector: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def reset(self) -> None:
        self.count = 0
        self._sum = 0.0
        self._sum_squared = 0.0
        self._maximum = 0.0
        self._sum_vector.fill(0.0)

    def update(
        self,
        predicted_position: Sequence[float] | npt.NDArray[np.floating[Any]],
        actual_position: Sequence[float] | npt.NDArray[np.floating[Any]],
        *,
        timestamp: float | None = None,
        object_id: str | None = None,
    ) -> PredictionError:
        predicted = _vector3(predicted_position, "predicted_position")
        actual = _vector3(actual_position, "actual_position")
        vector = predicted - actual
        distance = float(np.linalg.norm(vector))
        if timestamp is not None:
            timestamp = _finite_scalar(timestamp, "timestamp")
        self.count += 1
        self._sum += distance
        self._sum_squared += distance * distance
        self._maximum = max(self._maximum, distance)
        self._sum_vector += vector
        return PredictionError(
            predicted_position=predicted,
            actual_position=actual,
            error_vector=vector,
            euclidean_error=distance,
            timestamp=timestamp,
            object_id=object_id,
        )

    add = update

    @property
    def mean_error(self) -> float:
        return self._sum / self.count if self.count else 0.0

    @property
    def max_error(self) -> float:
        return self._maximum

    @property
    def average_error(self) -> float:
        return self.mean_error

    @property
    def rmse(self) -> float:
        return sqrt(self._sum_squared / self.count) if self.count else 0.0

    @property
    def mean_error_vector(self) -> npt.NDArray[np.float64]:
        return (
            self._sum_vector / self.count
            if self.count
            else np.zeros(3, dtype=np.float64)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_prediction_error": self.mean_error,
            "max_prediction_error": self.max_error,
            "prediction_rmse": self.rmse,
            "mean_prediction_error_vector": self.mean_error_vector.copy(),
        }


class _WindowLike(Protocol):
    def contains(
        self,
        timestamp: float,
        position: Sequence[float] | npt.NDArray[np.floating[Any]] | None = None,
    ) -> bool: ...


class MotionPredictor:
    """Configurable constant-velocity/windowed/acceleration predictor.

    ``system_latency`` is interpreted as the observation-to-decision pipeline
    latency.  If ``current_time`` already includes that latency, it is not
    counted twice::

        decision_time = max(current_time, observed_time + system_latency)
        grasp_time = decision_time + robot_arrival_time

    With compensation disabled, ``predicted_position`` is exactly the historical
    observed position.  Other estimates remain available for diagnostics.
    ``max_prediction_horizon=None`` disables age rejection while diagnostic
    uncertainty still grows with extrapolation time.
    """

    def __init__(
        self,
        model: MotionModel | str = MotionModel.WINDOWED_VELOCITY,
        *,
        window_size: int = 6,
        window_seconds: float | None = 0.5,
        belt_direction: Sequence[float] = (1.0, 0.0, 0.0),
        enable_motion_compensation: bool = True,
        max_prediction_horizon: float | None = 2.0,
        nominal_position_noise: float = 0.003,
        acceleration_limit: float | None = 2.0,
        use_acceleration: bool | None = None,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if window_seconds is not None and window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive or None")
        if max_prediction_horizon is not None:
            max_prediction_horizon = float(max_prediction_horizon)
        if max_prediction_horizon is not None and max_prediction_horizon <= 0.0:
            raise ValueError("max_prediction_horizon must be positive or None")
        if nominal_position_noise <= 0.0:
            raise ValueError("nominal_position_noise must be positive")
        if acceleration_limit is not None and acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be positive or None")
        direction = _vector3(belt_direction, "belt_direction")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("belt_direction must be non-zero")
        self.model = (
            MotionModel.CONSTANT_ACCELERATION
            if use_acceleration is True
            else MotionModel.parse(model)
        )
        self.window_size = int(window_size)
        self.window_seconds = window_seconds
        self.belt_direction = direction / norm
        self.enable_motion_compensation = bool(enable_motion_compensation)
        self.max_prediction_horizon = (
            None if max_prediction_horizon is None else float(max_prediction_horizon)
        )
        # Unlimited prediction removes age rejection, not uncertainty growth.
        # Keep the original default 2 s scale for diagnostic uncertainty/score.
        self._uncertainty_horizon_scale = (
            2.0 if self.max_prediction_horizon is None else self.max_prediction_horizon
        )
        self.nominal_position_noise = float(nominal_position_noise)
        self.acceleration_limit = acceleration_limit
        self.error_statistics = PredictionErrorStatistics()

    @classmethod
    def from_config(cls, source: Mapping[str, Any]) -> "MotionPredictor":
        """Load predictor settings from latency/full-project configuration."""

        root: Mapping[str, Any] = source
        for _ in range(2):
            nested = root.get("latency")
            if not isinstance(nested, Mapping) or nested is root:
                break
            root = nested
        prediction = root.get("prediction", root)
        if not isinstance(prediction, Mapping):
            prediction = {}
        use_acceleration = bool(prediction.get("use_acceleration", False))
        configured_model = prediction.get(
            "model",
            MotionModel.CONSTANT_ACCELERATION.value
            if use_acceleration
            else MotionModel.WINDOWED_VELOCITY.value,
        )
        return cls(
            model=configured_model,
            window_size=int(prediction.get("window_size", 6)),
            window_seconds=prediction.get(
                "velocity_window_s", prediction.get("window_seconds", 0.5)
            ),
            belt_direction=prediction.get("belt_direction", (1.0, 0.0, 0.0)),
            enable_motion_compensation=bool(
                root.get("enable_motion_compensation", True)
            ),
            max_prediction_horizon=(
                prediction.get(
                    "maximum_horizon_s",
                    prediction.get("max_prediction_horizon", 2.0),
                )
            ),
            nominal_position_noise=float(
                prediction.get("nominal_position_noise", 0.003)
            ),
            acceleration_limit=prediction.get("acceleration_limit", 2.0),
        )

    def reset(self) -> None:
        self.error_statistics.reset()

    def estimate_velocity(
        self,
        history: Iterable[Any] | ObservationBuffer[Any],
        *,
        belt_velocity: float | Sequence[float] | None = None,
        model: MotionModel | str | None = None,
    ) -> npt.NDArray[np.float64]:
        samples = self._select_window(_coerce_samples(history))
        velocity, _, _, _ = self._fit(
            samples,
            MotionModel.parse(model) if model is not None else self.model,
            _coerce_velocity(belt_velocity, self.belt_direction),
            supplied_velocity=None,
        )
        return velocity

    def predict(
        self,
        history: Iterable[Any] | ObservationBuffer[Any],
        *,
        current_time: float | None = None,
        system_latency: float = 0.0,
        robot_arrival_time: float = 0.0,
        time_to_reach: float | None = None,
        belt_velocity: float | Sequence[float] | None = None,
        estimated_velocity: float | Sequence[float] | None = None,
        enable_motion_compensation: bool | None = None,
        model: MotionModel | str | None = None,
        object_id: str | None = None,
        grasp_window: _WindowLike | None = None,
    ) -> PredictionResult:
        """Predict current and future object position.

        ``time_to_reach`` is accepted as a readable alias for
        ``robot_arrival_time``.
        """

        samples = _coerce_samples(history)
        if not samples:
            raise ValueError("motion prediction requires at least one sample")
        samples = self._select_window(samples)
        observed = samples[-1]
        now = (
            observed.timestamp
            if current_time is None
            else _finite_scalar(current_time, "current_time")
        )
        latency = _nonnegative(system_latency, "system_latency")
        if time_to_reach is not None:
            if robot_arrival_time != 0.0:
                raise ValueError(
                    "pass only one of robot_arrival_time and time_to_reach"
                )
            robot_arrival_time = time_to_reach
        arrival = _nonnegative(robot_arrival_time, "robot_arrival_time")
        if now + 1e-12 < observed.timestamp:
            raise ValueError(
                "current_time cannot precede the newest observation timestamp"
            )
        decision_time = max(now, observed.timestamp + latency)
        grasp_time = decision_time + arrival
        selected_model = MotionModel.parse(model) if model is not None else self.model
        fallback_velocity = _coerce_velocity(belt_velocity, self.belt_direction)
        supplied_velocity = _coerce_velocity(
            estimated_velocity, self.belt_direction
        )
        velocity, acceleration, residual_std, fit_rmse = self._fit(
            samples, selected_model, fallback_velocity, supplied_velocity
        )
        current_dt = decision_time - observed.timestamp
        future_dt = grasp_time - observed.timestamp
        current_estimate = (
            observed.position
            + velocity * current_dt
            + 0.5 * acceleration * current_dt * current_dt
        )
        extrapolated = (
            observed.position
            + velocity * future_dt
            + 0.5 * acceleration * future_dt * future_dt
        )
        compensation = (
            self.enable_motion_compensation
            if enable_motion_compensation is None
            else bool(enable_motion_compensation)
        )
        predicted = extrapolated if compensation else observed.position.copy()
        uncertainty = self._uncertainty(
            residual_std, future_dt, len(samples), selected_model
        )
        confidence = self._confidence(
            uncertainty, fit_rmse, future_dt, len(samples), fallback_velocity
        )
        valid = (self.max_prediction_horizon is None
                 or future_dt <= self.max_prediction_horizon + 1e-12)
        reason = "ok" if valid else "prediction_horizon_exceeded"
        if not valid:
            confidence = 0.0
        in_window: bool | None = False if not valid else None
        if grasp_window is not None and valid:
            try:
                in_window = bool(grasp_window.contains(grasp_time, predicted))
            except TypeError:
                in_window = bool(grasp_window.contains(grasp_time))
        return PredictionResult(
            object_id=object_id,
            observed_timestamp=observed.timestamp,
            decision_timestamp=decision_time,
            grasp_timestamp=grasp_time,
            observed_position=observed.position,
            current_estimated_position=current_estimate,
            predicted_position=predicted,
            estimated_velocity=velocity,
            estimated_acceleration=acceleration,
            uncertainty=uncertainty,
            confidence=confidence,
            model=selected_model,
            motion_compensation_enabled=compensation,
            in_grasp_window=in_window,
            sample_count=len(samples),
            fit_rmse=fit_rmse,
            valid=valid,
            reason=reason,
        )

    def predict_position(
        self,
        observed_position: Sequence[float],
        *,
        horizon: float,
        velocity: float | Sequence[float],
        acceleration: float | Sequence[float] | None = None,
        enable_motion_compensation: bool | None = None,
    ) -> npt.NDArray[np.float64]:
        """Small stateless helper useful in tests and external policies."""

        position = _vector3(observed_position, "observed_position")
        dt = _nonnegative(horizon, "horizon")
        vel = _coerce_velocity(velocity, self.belt_direction)
        assert vel is not None
        acc = _coerce_velocity(acceleration, self.belt_direction)
        if acc is None:
            acc = np.zeros(3, dtype=np.float64)
        enabled = (
            self.enable_motion_compensation
            if enable_motion_compensation is None
            else bool(enable_motion_compensation)
        )
        return (
            position + vel * dt + 0.5 * acc * dt * dt
            if enabled
            else position.copy()
        )

    predict_future_position = predict_position

    def record_actual(
        self,
        prediction: PredictionResult,
        actual_position: Sequence[float],
        *,
        timestamp: float | None = None,
    ) -> PredictionError:
        return self.error_statistics.update(
            prediction.predicted_position,
            actual_position,
            timestamp=timestamp,
            object_id=prediction.object_id,
        )

    def _select_window(self, samples: list[MotionSample]) -> list[MotionSample]:
        selected = samples[-self.window_size :]
        if self.window_seconds is not None and selected:
            minimum = selected[-1].timestamp - self.window_seconds
            time_selected = [
                sample for sample in selected if sample.timestamp >= minimum
            ]
            # Retain two samples when available even if the time window is tiny.
            if len(time_selected) < 2 and len(selected) >= 2:
                time_selected = selected[-2:]
            selected = time_selected
        return selected

    def _fit(
        self,
        samples: list[MotionSample],
        model: MotionModel,
        fallback_velocity: npt.NDArray[np.float64] | None,
        supplied_velocity: npt.NDArray[np.float64] | None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        zero = np.zeros(3, dtype=np.float64)
        if supplied_velocity is not None:
            return supplied_velocity, zero.copy(), zero.copy(), 0.0
        if len(samples) == 1:
            velocity = fallback_velocity if fallback_velocity is not None else zero
            return velocity.copy(), zero.copy(), zero.copy(), 0.0

        times = np.asarray(
            [sample.timestamp - samples[-1].timestamp for sample in samples],
            dtype=np.float64,
        )
        positions = np.stack([sample.position for sample in samples])
        if np.ptp(times) <= 1e-9:
            velocity = fallback_velocity if fallback_velocity is not None else zero
            return velocity.copy(), zero.copy(), zero.copy(), 0.0

        if model is MotionModel.CONSTANT_VELOCITY:
            dt = samples[-1].timestamp - samples[0].timestamp
            velocity = (samples[-1].position - samples[0].position) / dt
            fitted = positions[0] + np.outer(times - times[0], velocity)
            acceleration = zero.copy()
        elif model is MotionModel.WINDOWED_VELOCITY or len(samples) < 3:
            design = np.column_stack((np.ones(len(times)), times))
            coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
            velocity = coefficients[1]
            fitted = design @ coefficients
            acceleration = zero.copy()
        else:
            design = np.column_stack(
                (np.ones(len(times)), times, 0.5 * times * times)
            )
            coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
            velocity = coefficients[1]
            acceleration = coefficients[2]
            if self.acceleration_limit is not None:
                norm = float(np.linalg.norm(acceleration))
                if norm > self.acceleration_limit:
                    acceleration = acceleration * (self.acceleration_limit / norm)
                    fitted = (
                        coefficients[0]
                        + np.outer(times, velocity)
                        + 0.5 * np.outer(times * times, acceleration)
                    )
                else:
                    fitted = design @ coefficients
            else:
                fitted = design @ coefficients
        residual = positions - fitted
        residual_std = np.sqrt(np.mean(residual * residual, axis=0))
        fit_rmse = float(np.sqrt(np.mean(residual * residual)))
        return velocity, acceleration, residual_std, fit_rmse

    def _uncertainty(
        self,
        residual_std: npt.NDArray[np.float64],
        horizon: float,
        sample_count: int,
        model: MotionModel,
    ) -> npt.NDArray[np.float64]:
        base = np.maximum(residual_std, self.nominal_position_noise)
        sample_penalty = 1.0 + 1.0 / sqrt(max(sample_count, 1))
        horizon_penalty = 1.0 + max(horizon, 0.0) / self._uncertainty_horizon_scale
        if model is MotionModel.CONSTANT_ACCELERATION and sample_count < 4:
            horizon_penalty *= 1.25
        return base * sample_penalty * horizon_penalty

    def _confidence(
        self,
        uncertainty: npt.NDArray[np.float64],
        fit_rmse: float,
        horizon: float,
        sample_count: int,
        fallback_velocity: npt.NDArray[np.float64] | None,
    ) -> float:
        sample_score = min(1.0, sample_count / 4.0)
        if sample_count == 1 and fallback_velocity is not None:
            sample_score = 0.65
        horizon_score = exp(
            -max(0.0, horizon) / max(self._uncertainty_horizon_scale, 1e-9)
        )
        scale = max(self.nominal_position_noise * 5.0, 1e-9)
        fit_score = exp(-fit_rmse / scale)
        uncertainty_score = exp(-float(np.linalg.norm(uncertainty)) / (scale * 3.0))
        return float(
            np.clip(
                sample_score * horizon_score * sqrt(fit_score * uncertainty_score),
                0.0,
                1.0,
            )
        )


class ConstantVelocityPredictor(MotionPredictor):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model", None)
        super().__init__(model=MotionModel.CONSTANT_VELOCITY, **kwargs)


class SlidingWindowPredictor(MotionPredictor):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model", None)
        super().__init__(model=MotionModel.WINDOWED_VELOCITY, **kwargs)


class ConstantAccelerationPredictor(MotionPredictor):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model", None)
        super().__init__(model=MotionModel.CONSTANT_ACCELERATION, **kwargs)


def _coerce_samples(
    history: Iterable[Any] | ObservationBuffer[Any],
) -> list[MotionSample]:
    source: Iterable[Any] = history
    parsed: list[MotionSample] = []
    for entry in source:
        timestamp: Any
        state: Any
        if isinstance(entry, MotionSample):
            parsed.append(entry)
            continue
        if isinstance(entry, TimestampedState):
            timestamp, state = entry.timestamp, entry.state
        elif isinstance(entry, Mapping):
            timestamp = entry.get("timestamp", entry.get("time"))
            state = entry
        elif isinstance(entry, Sequence) and len(entry) == 2:
            timestamp, state = entry
        elif hasattr(entry, "timestamp"):
            timestamp, state = getattr(entry, "timestamp"), entry
        else:
            raise TypeError(
                "history entries must be MotionSample, TimestampedState, "
                "a mapping with timestamp/position, or (timestamp, position)"
            )
        if isinstance(state, Mapping):
            position = state.get(
                "position",
                state.get(
                    "pos",
                    state.get("translation", state.get("observed_position")),
                ),
            )
        elif hasattr(state, "position"):
            position = getattr(state, "position")
        elif hasattr(state, "pose"):
            pose = np.asarray(getattr(state, "pose"))
            position = pose[:3]
        else:
            position = state
        if timestamp is None or position is None:
            raise ValueError("each history entry requires timestamp and position")
        parsed.append(MotionSample(float(timestamp), position))
    parsed.sort(key=lambda sample: sample.timestamp)
    # Replace duplicate times with the last supplied measurement.
    deduplicated: list[MotionSample] = []
    for sample in parsed:
        if deduplicated and sample.timestamp == deduplicated[-1].timestamp:
            deduplicated[-1] = sample
        else:
            deduplicated.append(sample)
    return deduplicated


def _coerce_velocity(
    value: float | Sequence[float] | None,
    belt_direction: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        if not np.isfinite(array):
            raise ValueError("velocity must be finite")
        return belt_direction * float(array)
    return _vector3(array, "velocity")


def _vector3(value: Any, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _finite_scalar(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _nonnegative(value: Any, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar

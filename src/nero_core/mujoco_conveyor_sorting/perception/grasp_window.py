"""Dynamic grasp-window calculation for conveyor targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .motion_predictor import PredictionResult

PositionCheck = Callable[[npt.NDArray[np.float64]], bool]


@dataclass(frozen=True, slots=True)
class GraspWindowConfig:
    """Geometry and timing assumptions for a dynamic pickup."""

    workspace_min: Sequence[float] = (-0.30, -0.115, 0.19)
    workspace_max: Sequence[float] = (0.30, 0.115, 0.36)
    belt_direction: Sequence[float] = (1.0, 0.0, 0.0)
    belt_end_coordinate: float | None = 0.64
    default_planning_time: float = 0.10
    default_motion_time: float = 0.35
    default_approach_time: float = 0.10
    default_gripper_time: float = 0.06
    maximum_horizon: float = 5.0
    minimum_window_duration: float = 0.04
    minimum_confidence: float = 0.15
    uncertainty_margin_scale: float = 2.0
    preferred_approach_direction: Sequence[float] = (0.0, 0.0, -1.0)
    maximum_approach_angle_deg: float = 60.0
    reachability_samples: int = 7

    def __post_init__(self) -> None:
        lower = _vector3(self.workspace_min, "workspace_min")
        upper = _vector3(self.workspace_max, "workspace_max")
        if np.any(upper <= lower):
            raise ValueError("workspace_max must exceed workspace_min on every axis")
        direction = _unit(self.belt_direction, "belt_direction")
        approach = _unit(
            self.preferred_approach_direction, "preferred_approach_direction"
        )
        object.__setattr__(self, "workspace_min", lower)
        object.__setattr__(self, "workspace_max", upper)
        object.__setattr__(self, "belt_direction", direction)
        object.__setattr__(self, "preferred_approach_direction", approach)
        for name in (
            "default_planning_time",
            "default_motion_time",
            "default_approach_time",
            "default_gripper_time",
            "maximum_horizon",
            "minimum_window_duration",
            "uncertainty_margin_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_horizon <= 0.0:
            raise ValueError("maximum_horizon must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0.0 < self.maximum_approach_angle_deg <= 180.0:
            raise ValueError("maximum_approach_angle_deg must be in (0, 180]")
        if self.reachability_samples < 2:
            raise ValueError("reachability_samples must be at least 2")
        if self.belt_end_coordinate is not None and not np.isfinite(
            self.belt_end_coordinate
        ):
            raise ValueError("belt_end_coordinate must be finite or None")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "GraspWindowConfig":
        """Load the grasp region from a scene/full-project configuration."""

        root: Mapping[str, Any] = source
        if isinstance(root.get("scene"), Mapping):
            root = root["scene"]  # type: ignore[assignment]
        default = cls()
        workspace = root.get("workspace", {})
        if not isinstance(workspace, Mapping):
            workspace = {}
        region = workspace.get("grasp_region", root.get("grasp_region", {}))
        if not isinstance(region, Mapping):
            region = {}
        lower = np.asarray(default.workspace_min, dtype=np.float64).copy()
        upper = np.asarray(default.workspace_max, dtype=np.float64).copy()
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            bounds = region.get(axis_name)
            if bounds is not None:
                pair = np.asarray(bounds, dtype=np.float64)
                if pair.shape != (2,):
                    raise ValueError(
                        f"workspace.grasp_region.{axis_name} must contain 2 values"
                    )
                lower[axis_index], upper[axis_index] = pair
        conveyor = root.get("conveyor", {})
        if not isinstance(conveyor, Mapping):
            conveyor = {}
        timing = root.get("grasp_window", root.get("timing", {}))
        if not isinstance(timing, Mapping):
            timing = {}
        return cls(
            workspace_min=lower,
            workspace_max=upper,
            belt_direction=conveyor.get("direction", default.belt_direction),
            belt_end_coordinate=conveyor.get(
                "missed_x",
                conveyor.get("belt_end_coordinate", default.belt_end_coordinate),
            ),
            default_planning_time=timing.get(
                "planning_time", default.default_planning_time
            ),
            default_motion_time=timing.get(
                "motion_time", default.default_motion_time
            ),
            default_approach_time=timing.get(
                "approach_time", default.default_approach_time
            ),
            default_gripper_time=timing.get(
                "gripper_time", default.default_gripper_time
            ),
            maximum_horizon=timing.get(
                "maximum_horizon", default.maximum_horizon
            ),
            minimum_window_duration=timing.get(
                "minimum_window_duration", default.minimum_window_duration
            ),
            minimum_confidence=timing.get(
                "minimum_confidence", default.minimum_confidence
            ),
            uncertainty_margin_scale=timing.get(
                "uncertainty_margin_scale", default.uncertainty_margin_scale
            ),
            preferred_approach_direction=timing.get(
                "preferred_approach_direction",
                default.preferred_approach_direction,
            ),
            maximum_approach_angle_deg=timing.get(
                "maximum_approach_angle_deg",
                default.maximum_approach_angle_deg,
            ),
            reachability_samples=timing.get(
                "reachability_samples", default.reachability_samples
            ),
        )


@dataclass(frozen=True, slots=True)
class GraspWindow:
    """A time interval during which target closure is safe and reachable."""

    object_id: str | None
    current_time: float
    start_time: float
    end_time: float
    predicted_grasp_time: float
    entry_position: npt.NDArray[np.float64]
    exit_position: npt.NDArray[np.float64]
    predicted_grasp_position: npt.NDArray[np.float64]
    feasible: bool
    current_in_window: bool
    reason: str
    confidence: float
    uncertainty_margin: npt.NDArray[np.float64]
    planning_ready_time: float
    geometric_start_time: float
    geometric_end_time: float

    def __post_init__(self) -> None:
        for name in (
            "entry_position",
            "exit_position",
            "predicted_grasp_position",
            "uncertainty_margin",
        ):
            object.__setattr__(self, name, _vector3(getattr(self, name), name))

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def time_to_open(self) -> float:
        return max(0.0, self.start_time - self.current_time)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - self.current_time)

    @property
    def is_graspable(self) -> bool:
        return self.feasible and self.current_in_window

    @property
    def should_skip(self) -> bool:
        return not self.feasible

    def contains(
        self,
        timestamp: float,
        position: Sequence[float] | None = None,
    ) -> bool:
        """Return whether a closure time/optional position lies in this window."""

        time = float(timestamp)
        if not self.feasible or not self.start_time <= time <= self.end_time:
            return False
        if position is None:
            return True
        point = _vector3(position, "position")
        # Entry/exit form a conveyor line segment.  This broad AABB check is
        # orientation-independent and includes the configured uncertainty margin.
        lower = np.minimum(self.entry_position, self.exit_position)
        upper = np.maximum(self.entry_position, self.exit_position)
        tolerance = self.uncertainty_margin + 1e-9
        return bool(np.all(point >= lower - tolerance) and np.all(point <= upper + tolerance))

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "predicted_grasp_time": self.predicted_grasp_time,
            "entry_position": self.entry_position.copy(),
            "exit_position": self.exit_position.copy(),
            "predicted_grasp_position": self.predicted_grasp_position.copy(),
            "feasible": self.feasible,
            "is_graspable": self.is_graspable,
            "current_in_window": self.current_in_window,
            "should_skip": self.should_skip,
            "reason": self.reason,
            "confidence": self.confidence,
            "uncertainty_margin": self.uncertainty_margin.copy(),
            "planning_ready_time": self.planning_ready_time,
        }


# Descriptive alias used in some policy code.
DynamicGraspWindow = GraspWindow


class GraspWindowCalculator:
    """Intersect a target trajectory with reachable and belt-safe regions."""

    def __init__(
        self,
        config: GraspWindowConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ):
        if config is not None and overrides:
            raise ValueError("pass either config or keyword overrides, not both")
        if isinstance(config, Mapping):
            self.config = GraspWindowConfig.from_mapping(config)
        else:
            self.config = (
                config if config is not None else GraspWindowConfig(**overrides)
            )

    def calculate(
        self,
        position: Sequence[float],
        velocity: float | Sequence[float],
        current_time: float,
        *,
        acceleration: float | Sequence[float] = 0.0,
        object_id: str | None = None,
        planning_time: float | None = None,
        motion_time: float | None = None,
        approach_time: float | None = None,
        gripper_time: float | None = None,
        uncertainty: float | Sequence[float] = 0.0,
        confidence: float = 1.0,
        executor_busy: bool = False,
        approach_direction: Sequence[float] | None = None,
        collision_check: PositionCheck | None = None,
        reachability_check: PositionCheck | None = None,
        belt_end_coordinate: float | None = None,
        requested_grasp_time: float | None = None,
    ) -> GraspWindow:
        """Calculate the feasible gripper-closure interval.

        The region is shrunk by prediction uncertainty, then a continuous
        point/axis-aligned-box intersection supplies the geometric interval.
        Planning, motion, approach and gripper-close time determine the earliest
        actionable instant.  Failure reasons are stable machine-readable strings.
        """

        config = self.config
        now = _finite(current_time, "current_time")
        point = _vector3(position, "position")
        vector = _velocity3(velocity, config.belt_direction)
        acceleration_vector = _velocity3(
            acceleration, config.belt_direction
        )
        margin = _margin3(uncertainty) * config.uncertainty_margin_scale
        lower = np.asarray(config.workspace_min) + margin
        upper = np.asarray(config.workspace_max) - margin
        timings = [
            config.default_planning_time if planning_time is None else planning_time,
            config.default_motion_time if motion_time is None else motion_time,
            config.default_approach_time if approach_time is None else approach_time,
            config.default_gripper_time if gripper_time is None else gripper_time,
        ]
        timing_names = ("planning_time", "motion_time", "approach_time", "gripper_time")
        timings = [
            _nonnegative(value, name)
            for value, name in zip(timings, timing_names, strict=True)
        ]
        ready_time = now + sum(timings)
        confidence_value = float(confidence)
        if not np.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

        reason = "ok"
        interval = _trajectory_aabb_interval(
            point,
            vector,
            acceleration_vector,
            lower,
            upper,
            config.maximum_horizon,
        )
        if np.any(lower > upper):
            interval = None
            reason = "uncertainty_exceeds_workspace"
        if interval is None and reason == "ok":
            reason = "trajectory_outside_reachable_workspace"
        geometric_start_offset, geometric_end_offset = (
            interval if interval is not None else (0.0, 0.0)
        )

        end_coordinate = (
            config.belt_end_coordinate
            if belt_end_coordinate is None
            else _finite(belt_end_coordinate, "belt_end_coordinate")
        )
        if interval is not None and end_coordinate is not None:
            belt_interval = _halfspace_interval(
                point,
                vector,
                acceleration_vector,
                np.asarray(config.belt_direction),
                end_coordinate,
                config.maximum_horizon,
            )
            if belt_interval is None:
                interval = None
                reason = "object_past_belt_end"
            else:
                geometric_start_offset = max(
                    geometric_start_offset, belt_interval[0]
                )
                geometric_end_offset = min(geometric_end_offset, belt_interval[1])
                if geometric_end_offset < geometric_start_offset:
                    interval = None
                    reason = "leaves_belt_before_reachable"

        geometric_start = now + geometric_start_offset
        geometric_end = now + geometric_end_offset
        start = max(geometric_start, ready_time)
        end = geometric_end
        requested = (
            start
            if requested_grasp_time is None
            else _finite(requested_grasp_time, "requested_grasp_time")
        )
        predicted_time = max(start, requested)
        # Never silently clamp a user-requested time beyond the deadline; it
        # should produce a clear infeasibility reason.
        predicted_offset = max(0.0, predicted_time - now)
        predicted_position = (
            point
            + vector * predicted_offset
            + 0.5 * acceleration_vector * predicted_offset**2
        )
        entry_position = (
            point
            + vector * geometric_start_offset
            + 0.5 * acceleration_vector * geometric_start_offset**2
        )
        exit_position = (
            point
            + vector * geometric_end_offset
            + 0.5 * acceleration_vector * geometric_end_offset**2
        )

        feasible = interval is not None
        if feasible and confidence_value < config.minimum_confidence:
            feasible, reason = False, "prediction_confidence_too_low"
        if feasible and executor_busy:
            feasible, reason = False, "executor_busy"
        if feasible and end - start < config.minimum_window_duration:
            feasible, reason = False, "insufficient_time_window"
        if feasible and requested_grasp_time is not None and not start <= requested <= end:
            feasible, reason = False, "requested_time_outside_window"
        if feasible and not self._approach_is_valid(approach_direction):
            feasible, reason = False, "invalid_approach_direction"
        if feasible and not self._path_check(
            point,
            vector,
            acceleration_vector,
            start,
            end,
            now,
            reachability_check,
            expected=True,
        ):
            feasible, reason = False, "unreachable_trajectory"
        if feasible and not self._path_check(
            point,
            vector,
            acceleration_vector,
            start,
            end,
            now,
            collision_check,
            expected=False,
        ):
            feasible, reason = False, "predicted_collision"

        # "Current" refers to the target's geometric presence in the grasp
        # region.  ``start_time`` remains the earliest *actionable closure*
        # instant after planning/motion latency.
        current_in_window = (
            feasible and geometric_start <= now <= geometric_end
        )
        return GraspWindow(
            object_id=object_id,
            current_time=now,
            start_time=start,
            end_time=end,
            predicted_grasp_time=predicted_time,
            entry_position=entry_position,
            exit_position=exit_position,
            predicted_grasp_position=predicted_position,
            feasible=feasible,
            current_in_window=current_in_window,
            reason=reason,
            confidence=confidence_value,
            uncertainty_margin=margin,
            planning_ready_time=ready_time,
            geometric_start_time=geometric_start,
            geometric_end_time=geometric_end,
        )

    compute = calculate
    compute_window = calculate
    calculate_window = calculate

    def from_prediction(
        self,
        prediction: PredictionResult,
        *,
        planning_time: float | None = None,
        motion_time: float | None = None,
        approach_time: float | None = None,
        gripper_time: float | None = None,
        executor_busy: bool = False,
        collision_check: PositionCheck | None = None,
        reachability_check: PositionCheck | None = None,
    ) -> GraspWindow:
        # PredictionResult already includes its arm-arrival horizon.  Using it
        # as the default timing avoids counting the calculator's nominal
        # planning/motion delay a second time.
        if (
            planning_time is None
            and motion_time is None
            and approach_time is None
            and gripper_time is None
        ):
            planning_time = 0.0
            motion_time = max(0.0, prediction.arrival_horizon)
            approach_time = 0.0
            gripper_time = 0.0
        return self.calculate(
            position=prediction.current_estimated_position,
            velocity=prediction.current_estimated_velocity,
            acceleration=prediction.estimated_acceleration,
            current_time=prediction.decision_timestamp,
            object_id=prediction.object_id,
            planning_time=planning_time,
            motion_time=motion_time,
            approach_time=approach_time,
            gripper_time=gripper_time,
            uncertainty=prediction.uncertainty,
            confidence=prediction.confidence,
            executor_busy=executor_busy,
            collision_check=collision_check,
            reachability_check=reachability_check,
            requested_grasp_time=prediction.grasp_timestamp,
        )

    calculate_from_prediction = from_prediction

    def _approach_is_valid(self, direction: Sequence[float] | None) -> bool:
        if direction is None:
            return True
        approach = _unit(direction, "approach_direction")
        preferred = np.asarray(self.config.preferred_approach_direction)
        minimum_dot = cos(radians(self.config.maximum_approach_angle_deg))
        return float(np.dot(approach, preferred)) >= minimum_dot

    def _path_check(
        self,
        point: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        acceleration: npt.NDArray[np.float64],
        start: float,
        end: float,
        now: float,
        callback: PositionCheck | None,
        *,
        expected: bool,
    ) -> bool:
        if callback is None:
            return True
        if end < start:
            return False
        for timestamp in np.linspace(
            start, end, num=self.config.reachability_samples
        ):
            elapsed = timestamp - now
            candidate = (
                point
                + velocity * elapsed
                + 0.5 * acceleration * elapsed**2
            )
            try:
                result = bool(callback(candidate.copy()))
            except Exception as exc:
                raise RuntimeError("grasp-window callback failed") from exc
            if result is not expected:
                return False
        return True


def _trajectory_aabb_interval(
    point: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    acceleration: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    max_horizon: float,
) -> tuple[float, float] | None:
    valid: list[tuple[float, float]] = [(0.0, max_horizon)]
    for axis in range(3):
        axis_intervals = _quadratic_range_intervals(
            float(point[axis]),
            float(velocity[axis]),
            float(acceleration[axis]),
            float(lower[axis]),
            float(upper[axis]),
            max_horizon,
        )
        valid = _intersect_intervals(valid, axis_intervals)
        if not valid:
            return None
    return valid[0]


def _halfspace_interval(
    point: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    acceleration: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    upper_coordinate: float,
    max_horizon: float,
) -> tuple[float, float] | None:
    coordinate = float(np.dot(point, axis))
    speed = float(np.dot(velocity, axis))
    scalar_acceleration = float(np.dot(acceleration, axis))
    intervals = _quadratic_range_intervals(
        coordinate,
        speed,
        scalar_acceleration,
        float("-inf"),
        upper_coordinate,
        max_horizon,
    )
    return intervals[0] if intervals else None


def _quadratic_range_intervals(
    position: float,
    velocity: float,
    acceleration: float,
    lower: float,
    upper: float,
    max_horizon: float,
) -> list[tuple[float, float]]:
    """Intervals where ``p + v*t + .5*a*t²`` lies in [lower, upper]."""

    breakpoints = [0.0, float(max_horizon)]
    for boundary in (lower, upper):
        if not np.isfinite(boundary):
            continue
        constant = position - boundary
        if abs(acceleration) <= 1e-12:
            if abs(velocity) > 1e-12:
                root = -constant / velocity
                if 0.0 < root < max_horizon:
                    breakpoints.append(root)
            continue
        discriminant = velocity * velocity - 2.0 * acceleration * constant
        if discriminant < -1e-12:
            continue
        square_root = float(np.sqrt(max(0.0, discriminant)))
        for root in (
            (-velocity - square_root) / acceleration,
            (-velocity + square_root) / acceleration,
        ):
            if 0.0 < root < max_horizon:
                breakpoints.append(float(root))
    points = sorted(set(round(value, 14) for value in breakpoints))

    def inside(time: float) -> bool:
        value = position + velocity * time + 0.5 * acceleration * time * time
        return lower - 1e-12 <= value <= upper + 1e-12

    intervals: list[tuple[float, float]] = []
    for left, right in zip(points[:-1], points[1:], strict=True):
        if inside(0.5 * (left + right)):
            intervals.append((left, right))
    # Preserve a tangential, zero-duration contact for diagnostics.
    if not intervals:
        for point_time in points:
            if inside(point_time):
                intervals.append((point_time, point_time))
                break
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1e-10:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _intersect_intervals(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start <= end + 1e-12:
            result.append((start, end))
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return result


def _margin3(value: float | Sequence[float]) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        result = np.full(3, float(array), dtype=np.float64)
    elif array.shape == (3,):
        result = array.copy()
    else:
        raise ValueError(f"uncertainty must be scalar or shape (3,), got {array.shape}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("uncertainty must be finite and non-negative")
    return result


def _velocity3(
    value: float | Sequence[float],
    direction: Sequence[float],
) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        if not np.isfinite(array):
            raise ValueError("velocity must be finite")
        return np.asarray(direction, dtype=np.float64) * float(array)
    return _vector3(array, "velocity")


def _unit(value: Sequence[float], name: str) -> npt.NDArray[np.float64]:
    vector = _vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _vector3(value: Any, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result

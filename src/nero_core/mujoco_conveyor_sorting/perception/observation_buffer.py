"""Timestamped state buffers used to simulate camera/perception latency.

The important invariant in this module is that a delayed read returns a state
whose source time is in the past.  The buffer never substitutes the newest
ground-truth value when the requested historical sample is unavailable.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, is_dataclass, replace
from numbers import Number
from threading import RLock
from typing import Any, Generic, Iterable, Iterator, Literal, Mapping, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")
ReadMode = Literal["previous", "nearest", "linear"]


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    """Pipeline latency components in seconds."""

    camera_exposure: float = 0.020
    camera_capture: float = 0.0
    image_transport: float = 0.025
    perception_inference: float = 0.055
    trajectory_planning: float = 0.040
    control_communication: float = 0.015
    robot_response: float = 0.025

    def __post_init__(self) -> None:
        for name in (
            "camera_exposure",
            "camera_capture",
            "image_transport",
            "perception_inference",
            "trajectory_planning",
            "control_communication",
            "robot_response",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def observation_delay(self) -> float:
        return (
            self.camera_exposure
            + self.camera_capture
            + self.image_transport
            + self.perception_inference
        )

    @property
    def post_perception_delay(self) -> float:
        return (
            self.trajectory_planning
            + self.control_communication
            + self.robot_response
        )

    @property
    def total_delay(self) -> float:
        return self.observation_delay + self.post_perception_delay

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "LatencyBreakdown":
        root: Mapping[str, Any] = source
        for _ in range(2):
            nested = root.get("latency")
            if not isinstance(nested, Mapping) or nested is root:
                break
            root = nested
        default = cls()

        def component(name: str, *aliases: str) -> float:
            for key in (name, *aliases):
                if key in root:
                    value = float(root[key])
                    # Explicit ``*_ms`` aliases are converted below, while
                    # canonical fields are always seconds.
                    return value / 1000.0 if key.endswith("_ms") else value
            return float(getattr(default, name))

        return cls(
            camera_exposure=component(
                "camera_exposure", "exposure", "camera_exposure_ms"
            ),
            camera_capture=component("camera_capture", "capture"),
            image_transport=component(
                "image_transport", "image_transfer", "image_transfer_ms"
            ),
            perception_inference=component(
                "perception_inference", "inference", "perception_inference_ms"
            ),
            trajectory_planning=component(
                "trajectory_planning", "planning", "trajectory_planning_ms"
            ),
            control_communication=component(
                "control_communication",
                "communication",
                "control_communication_ms",
            ),
            robot_response=component(
                "robot_response", "actuation", "robot_response_ms"
            ),
        )

    def scaled(self, factor: float) -> "LatencyBreakdown":
        scale = float(factor)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("latency scale must be finite and non-negative")
        return LatencyBreakdown(
            **{
                name: getattr(self, name) * scale
                for name in self.__dataclass_fields__
            }
        )

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        } | {
            "observation_delay": self.observation_delay,
            "post_perception_delay": self.post_perception_delay,
            "total_delay": self.total_delay,
        }


@dataclass(frozen=True, slots=True)
class TimestampedState(Generic[T]):
    """One immutable timestamped sample.

    ``state`` is copied on insertion/read by default by :class:`ObservationBuffer`.
    The ``value`` property is an alias that makes the type convenient in generic
    time-series code.
    """

    timestamp: float
    state: T
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> T:
        return self.state

    def __iter__(self) -> Iterator[Any]:
        # Convenient, backwards-compatible tuple unpacking: ``t, state = sample``.
        yield self.timestamp
        yield self.state


@dataclass(frozen=True, slots=True)
class ObjectObservation:
    """Canonical truth/observation record for one conveyor object."""

    object_id: str
    category_id: int
    position: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]
    linear_velocity: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    timestamp: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(
            self, "quaternion", _vector(self.quaternion, 4, "quaternion")
        )
        quaternion_norm = float(np.linalg.norm(self.quaternion))
        if quaternion_norm <= 1e-12:
            raise ValueError("quaternion cannot be zero")
        object.__setattr__(self, "quaternion", self.quaternion / quaternion_norm)
        object.__setattr__(
            self,
            "linear_velocity",
            _vector(self.linear_velocity, 3, "linear_velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _vector(self.angular_velocity, 3, "angular_velocity"),
        )
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


class ObservationBuffer(Generic[T]):
    """Thread-safe, timestamp-ordered finite history.

    Parameters
    ----------
    max_history_seconds:
        Samples older than this duration relative to the newest sample are
        discarded.  ``None`` disables age-based pruning.
    maxlen:
        Hard bound on retained sample count.  ``None`` disables count pruning.
    default_delay:
        Delay used by :meth:`get_delayed` when no explicit delay is supplied.
    strict_timestamps:
        Reject out-of-order inserts.  Equal timestamps replace the newest
        sample, which is useful when a simulator updates a state more than once
        in a frame.
    copy_values:
        Protect truth state from mutation by consumers.
    """

    def __init__(
        self,
        max_history_seconds: float | None = 5.0,
        maxlen: int | None = 4096,
        default_delay: float = 0.15,
        *,
        strict_timestamps: bool = True,
        copy_values: bool = True,
        default_read_mode: ReadMode = "previous",
    ) -> None:
        if max_history_seconds is not None and max_history_seconds <= 0.0:
            raise ValueError("max_history_seconds must be positive or None")
        if maxlen is not None and maxlen <= 0:
            raise ValueError("maxlen must be positive or None")
        if default_delay < 0.0 or not np.isfinite(default_delay):
            raise ValueError("default_delay must be finite and non-negative")
        if default_read_mode not in ("previous", "nearest", "linear"):
            raise ValueError(f"unsupported default_read_mode: {default_read_mode!r}")
        self.max_history_seconds = max_history_seconds
        self.maxlen = maxlen
        self.default_delay = float(default_delay)
        self.strict_timestamps = strict_timestamps
        self.copy_values = copy_values
        self.default_read_mode = default_read_mode
        self._samples: list[TimestampedState[T]] = []
        self._lock = RLock()

    @classmethod
    def from_config(cls, source: Mapping[str, Any]) -> "ObservationBuffer[Any]":
        root: Mapping[str, Any] = source
        for _ in range(2):
            nested = root.get("latency")
            if not isinstance(nested, Mapping) or nested is root:
                break
            root = nested
        history = root.get("observation_history", {})
        if not isinstance(history, Mapping):
            history = {}
        latency = LatencyBreakdown.from_mapping(root)
        duration = float(history.get("duration_s", 5.0))
        period = float(history.get("sample_period_s", 0.02))
        maxlen = max(2, int(np.ceil(duration / max(period, 1e-9))) + 2)
        return cls(
            max_history_seconds=duration,
            maxlen=maxlen,
            default_delay=latency.observation_delay,
            default_read_mode=_configured_read_mode(
                history.get("interpolation", "previous")
            ),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[TimestampedState[T]]:
        with self._lock:
            snapshot = tuple(self._copy_sample(sample) for sample in self._samples)
        return iter(snapshot)

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        with self._lock:
            return np.asarray(
                [sample.timestamp for sample in self._samples], dtype=np.float64
            )

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def append(
        self,
        timestamp: float | TimestampedState[T],
        state: T | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TimestampedState[T]:
        """Append one sample and return the stored record.

        A :class:`TimestampedState` can be passed directly.  Otherwise both a
        numeric ``timestamp`` and ``state`` are required.
        """

        if isinstance(timestamp, TimestampedState):
            if state is not None or metadata is not None:
                raise ValueError(
                    "state/metadata must be omitted when appending TimestampedState"
                )
            sample = timestamp
        else:
            if state is None:
                raise ValueError("state is required")
            sample = TimestampedState(
                timestamp=float(timestamp),
                state=state,
                metadata={} if metadata is None else dict(metadata),
            )
        if not np.isfinite(sample.timestamp):
            raise ValueError("timestamp must be finite")
        stored = self._copy_sample(sample)
        with self._lock:
            if self._samples and sample.timestamp < self._samples[-1].timestamp:
                if self.strict_timestamps:
                    raise ValueError(
                        "out-of-order timestamp: "
                        f"{sample.timestamp} < {self._samples[-1].timestamp}"
                    )
                index = bisect_right(
                    [entry.timestamp for entry in self._samples], sample.timestamp
                )
                self._samples.insert(index, stored)
            elif self._samples and sample.timestamp == self._samples[-1].timestamp:
                self._samples[-1] = stored
            else:
                self._samples.append(stored)
            self._prune_unlocked()
        return self._copy_sample(stored)

    add = append
    record = append
    push = append
    add_state = append

    def extend(self, samples: Iterable[TimestampedState[T]]) -> None:
        for sample in samples:
            self.append(sample)

    def latest(self) -> TimestampedState[T] | None:
        with self._lock:
            if not self._samples:
                return None
            return self._copy_sample(self._samples[-1])

    def oldest(self) -> TimestampedState[T] | None:
        with self._lock:
            if not self._samples:
                return None
            return self._copy_sample(self._samples[0])

    def read_at(
        self,
        timestamp: float,
        *,
        mode: ReadMode = "previous",
        allow_extrapolation: bool = False,
    ) -> TimestampedState[T] | None:
        """Read the sample associated with ``timestamp``.

        ``previous`` is the correct mode for latency simulation: the returned
        source timestamp is never newer than the query.  ``linear`` returns an
        interpolated state stamped at the query time when two bracketing states
        exist.  Unsupported nested values fall back to the previous value.
        """

        query = float(timestamp)
        if not np.isfinite(query):
            raise ValueError("timestamp must be finite")
        if mode not in ("previous", "nearest", "linear"):
            raise ValueError(f"unsupported read mode: {mode!r}")
        with self._lock:
            if not self._samples:
                return None
            times = [sample.timestamp for sample in self._samples]
            right = bisect_right(times, query)
            if mode == "previous":
                if right == 0:
                    return (
                        self._copy_sample(self._samples[0])
                        if allow_extrapolation
                        else None
                    )
                return self._copy_sample(self._samples[right - 1])

            insertion = bisect_left(times, query)
            if mode == "nearest":
                if insertion == 0:
                    if query < times[0] and not allow_extrapolation:
                        return None
                    return self._copy_sample(self._samples[0])
                if insertion == len(times):
                    if query > times[-1] and not allow_extrapolation:
                        return self._copy_sample(self._samples[-1])
                    return self._copy_sample(self._samples[-1])
                before = self._samples[insertion - 1]
                after = self._samples[insertion]
                chosen = (
                    before
                    if query - before.timestamp <= after.timestamp - query
                    else after
                )
                return self._copy_sample(chosen)

            # Linear interpolation.
            if insertion < len(times) and times[insertion] == query:
                return self._copy_sample(self._samples[insertion])
            if insertion == 0:
                return (
                    self._copy_sample(self._samples[0])
                    if allow_extrapolation
                    else None
                )
            if insertion == len(times):
                return self._copy_sample(self._samples[-1])
            before = self._samples[insertion - 1]
            after = self._samples[insertion]
            span = after.timestamp - before.timestamp
            alpha = 0.0 if span <= 0.0 else (query - before.timestamp) / span
            state = _interpolate(before.state, after.state, alpha)
            metadata = dict(before.metadata)
            metadata["interpolated"] = True
            metadata["source_timestamps"] = (before.timestamp, after.timestamp)
            return TimestampedState(query, self._copy_value(state), metadata)

    at = read_at
    get_at = read_at
    get_state_at = read_at

    def get_delayed(
        self,
        current_time: float,
        delay: float | None = None,
        *,
        mode: ReadMode | None = None,
        allow_extrapolation: bool = False,
    ) -> TimestampedState[T] | None:
        """Return the state visible after a configured observation delay."""

        delay_seconds = self.default_delay if delay is None else float(delay)
        if delay_seconds < 0.0 or not np.isfinite(delay_seconds):
            raise ValueError("delay must be finite and non-negative")
        now = float(current_time)
        if not np.isfinite(now):
            raise ValueError("current_time must be finite")
        selected_mode = self.default_read_mode if mode is None else mode
        return self.read_at(
            now - delay_seconds,
            mode=selected_mode,
            allow_extrapolation=allow_extrapolation,
        )

    delayed = get_delayed
    read_delayed = get_delayed
    get_delayed_state = get_delayed

    def between(
        self, start_time: float, end_time: float
    ) -> tuple[TimestampedState[T], ...]:
        if end_time < start_time:
            raise ValueError("end_time must be >= start_time")
        with self._lock:
            times = [sample.timestamp for sample in self._samples]
            first = bisect_left(times, float(start_time))
            last = bisect_right(times, float(end_time))
            return tuple(
                self._copy_sample(sample) for sample in self._samples[first:last]
            )

    def _copy_value(self, value: T) -> T:
        if not self.copy_values:
            return value
        if isinstance(value, np.ndarray):
            return value.copy()  # type: ignore[return-value]
        return deepcopy(value)

    def _copy_sample(self, sample: TimestampedState[T]) -> TimestampedState[T]:
        return TimestampedState(
            timestamp=sample.timestamp,
            state=self._copy_value(sample.state),
            metadata=deepcopy(dict(sample.metadata))
            if self.copy_values
            else sample.metadata,
        )

    def _prune_unlocked(self) -> None:
        if not self._samples:
            return
        if self.max_history_seconds is not None:
            threshold = self._samples[-1].timestamp - self.max_history_seconds
            first_valid = bisect_left(
                [sample.timestamp for sample in self._samples], threshold
            )
            if first_valid:
                del self._samples[:first_valid]
        if self.maxlen is not None and len(self._samples) > self.maxlen:
            del self._samples[: len(self._samples) - self.maxlen]


class MultiObjectObservationBuffer:
    """Independent timestamp history per logical object ID.

    Keeping histories separate prevents a newly recycled MuJoCo body from
    contaminating the trajectory of the previous logical object that occupied
    the same body slot.
    """

    def __init__(
        self,
        max_history_seconds: float | None = 5.0,
        maxlen_per_object: int | None = 1024,
        default_delay: float = 0.15,
        default_read_mode: ReadMode = "previous",
    ) -> None:
        self.max_history_seconds = max_history_seconds
        self.maxlen_per_object = maxlen_per_object
        self.default_delay = default_delay
        self.default_read_mode = _configured_read_mode(default_read_mode)
        self._buffers: "OrderedDict[str, ObservationBuffer[Any]]" = OrderedDict()
        self._ended_at: dict[str, float] = {}
        self._lock = RLock()

    @classmethod
    def from_config(
        cls, source: Mapping[str, Any]
    ) -> "MultiObjectObservationBuffer":
        single = ObservationBuffer.from_config(source)
        return cls(
            max_history_seconds=single.max_history_seconds,
            maxlen_per_object=single.maxlen,
            default_delay=single.default_delay,
            default_read_mode=single.default_read_mode,
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffers)

    @property
    def object_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._buffers)

    def clear(self, object_id: str | None = None) -> None:
        with self._lock:
            if object_id is None:
                self._buffers.clear()
                self._ended_at.clear()
            else:
                self._buffers.pop(str(object_id), None)
                self._ended_at.pop(str(object_id), None)

    def append(
        self,
        object_id: str,
        timestamp: float,
        state: T,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TimestampedState[T]:
        key = str(object_id)
        if not key:
            raise ValueError("object_id cannot be empty")
        sample_time = float(timestamp)
        if not np.isfinite(sample_time):
            raise ValueError("timestamp must be finite")
        with self._lock:
            buffer = self._buffers.get(key)
            if buffer is None:
                buffer = ObservationBuffer[Any](
                    max_history_seconds=self.max_history_seconds,
                    maxlen=self.maxlen_per_object,
                    default_delay=self.default_delay,
                    default_read_mode=self.default_read_mode,
                )
                self._buffers[key] = buffer
            ended_at = self._ended_at.get(key)
            if ended_at is not None and sample_time >= ended_at:
                # The same logical ID should normally not be reused, but
                # explicit appends define a new visible segment if it is.
                self._ended_at.pop(key, None)
        return buffer.append(sample_time, state, metadata=metadata)

    add = append
    record = append

    def append_observation(
        self, observation: ObjectObservation | Mapping[str, Any]
    ) -> TimestampedState[Any]:
        if isinstance(observation, ObjectObservation):
            return self.append(
                observation.object_id, observation.timestamp, observation
            )
        object_id = observation.get("object_id", observation.get("id"))
        timestamp = observation.get("timestamp", observation.get("time"))
        if object_id is None or timestamp is None:
            raise ValueError("observation requires object_id and timestamp")
        return self.append(str(object_id), float(timestamp), dict(observation))

    def append_frame(
        self,
        timestamp: float,
        objects: Mapping[str, Any] | Iterable[Any],
        *,
        mark_missing: bool = True,
    ) -> dict[str, TimestampedState[Any]]:
        """Record all object truth states from one simulator frame."""

        result: dict[str, TimestampedState[Any]] = {}
        items: Iterable[tuple[Any, Any]]
        if isinstance(objects, Mapping):
            items = objects.items()
        else:
            normalized: list[tuple[Any, Any]] = []
            for state in objects:
                object_id = (
                    state.get("object_id")
                    if isinstance(state, Mapping)
                    else getattr(state, "object_id", None)
                )
                if object_id is None:
                    raise ValueError("each frame object requires object_id")
                normalized.append((object_id, state))
            items = normalized
        materialized_items = list(items)
        visible_ids = {str(object_id) for object_id, _ in materialized_items}
        if mark_missing:
            with self._lock:
                previously_visible = set(self._buffers) - set(self._ended_at)
            for missing_id in previously_visible - visible_ids:
                self.mark_disappeared(missing_id, timestamp)
        for object_id, state in materialized_items:
            result[str(object_id)] = self.append(
                str(object_id), timestamp, state
            )
        return result

    def mark_disappeared(self, object_id: str, timestamp: float) -> None:
        """Add a tombstone so delayed reads do not create ghost targets."""

        key = str(object_id)
        end_time = float(timestamp)
        if not np.isfinite(end_time):
            raise ValueError("timestamp must be finite")
        with self._lock:
            buffer = self._buffers.get(key)
            if buffer is None:
                return
            latest = buffer.latest()
            if latest is not None and end_time + 1e-12 < latest.timestamp:
                raise ValueError("disappearance timestamp precedes latest state")
            self._ended_at[key] = end_time

    end_track = mark_disappeared

    def history(self, object_id: str) -> ObservationBuffer[Any] | None:
        with self._lock:
            return self._buffers.get(str(object_id))

    get_history = history

    def latest(self, object_id: str) -> TimestampedState[Any] | None:
        buffer = self.history(object_id)
        return None if buffer is None else buffer.latest()

    def get_delayed(
        self,
        object_id: str,
        current_time: float,
        delay: float | None = None,
        *,
        mode: ReadMode | None = None,
    ) -> TimestampedState[Any] | None:
        key = str(object_id)
        buffer = self.history(key)
        if buffer is None:
            return None
        delay_seconds = self.default_delay if delay is None else float(delay)
        if not np.isfinite(delay_seconds) or delay_seconds < 0.0:
            raise ValueError("delay must be finite and non-negative")
        query_time = float(current_time) - delay_seconds
        with self._lock:
            ended_at = self._ended_at.get(key)
        if ended_at is not None and query_time >= ended_at:
            return None
        return buffer.get_delayed(current_time, delay, mode=mode)

    delayed = get_delayed

    def get_all_delayed(
        self,
        current_time: float,
        delay: float | None = None,
        *,
        mode: ReadMode | None = None,
        include_missing: bool = False,
    ) -> dict[str, TimestampedState[Any] | None]:
        with self._lock:
            items = tuple(self._buffers.items())
        result: dict[str, TimestampedState[Any] | None] = {}
        for object_id, _buffer in items:
            sample = self.get_delayed(
                object_id, current_time, delay, mode=mode
            )
            if sample is not None or include_missing:
                result[object_id] = sample
        return result

    delayed_objects = get_all_delayed

    def prune(self, current_time: float) -> tuple[str, ...]:
        """Remove tracks whose tombstones are older than retained history."""

        if self.max_history_seconds is None:
            return ()
        now = float(current_time)
        if not np.isfinite(now):
            raise ValueError("current_time must be finite")
        with self._lock:
            expired = tuple(
                object_id
                for object_id, ended_at in self._ended_at.items()
                if now - ended_at > self.max_history_seconds
            )
            for object_id in expired:
                self._ended_at.pop(object_id, None)
                self._buffers.pop(object_id, None)
        return expired


def _vector(value: Any, length: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _configured_read_mode(value: Any) -> ReadMode:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "step": "previous",
        "hold": "previous",
        "zero_order_hold": "previous",
        "none": "previous",
        "interpolate": "linear",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ("previous", "nearest", "linear"):
        raise ValueError(f"unsupported observation interpolation mode: {value!r}")
    return normalized  # type: ignore[return-value]


def _interpolate(before: Any, after: Any, alpha: float) -> Any:
    """Recursively interpolate common simulation-state containers."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        if before.shape == after.shape and np.issubdtype(before.dtype, np.number):
            return before + (after - before) * alpha
        return deepcopy(before)
    if (
        isinstance(before, Number)
        and isinstance(after, Number)
        and not isinstance(before, (bool, np.bool_))
    ):
        return before + (after - before) * alpha
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return {
            key: _interpolate(before[key], after[key], alpha)
            if key in after
            else deepcopy(before[key])
            for key in before
        }
    if is_dataclass(before) and type(before) is type(after):
        updates: dict[str, Any] = {}
        for data_field in before.__dataclass_fields__.values():
            name = data_field.name
            interpolated = _interpolate(
                getattr(before, name), getattr(after, name), alpha
            )
            if (
                name in ("quaternion", "quaternion_wxyz", "orientation")
                and isinstance(interpolated, np.ndarray)
                and interpolated.shape == (4,)
            ):
                norm = float(np.linalg.norm(interpolated))
                if norm > 1e-12:
                    interpolated = interpolated / norm
            updates[name] = interpolated
        try:
            return replace(before, **updates)
        except (TypeError, ValueError):
            return deepcopy(before)
    return deepcopy(before)

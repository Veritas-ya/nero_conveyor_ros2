"""Hardware-neutral state and command types for sorting runtimes.

All timestamps exposed to the runtime are seconds on one monotonic host clock.
Device timestamps may additionally be retained in ``source_time`` for latency
diagnostics, but must never be compared directly with host timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class BackendError(RuntimeError):
    """A device backend could not complete an operation."""


class BackendUnavailable(BackendError):
    """A backend or a fresh device sample is unavailable."""


class MotionRejected(BackendError):
    """The safety gate rejected a motion command."""


class RuntimeState(str, Enum):
    DISCONNECTED = "disconnected"
    SELF_TEST = "self_test"
    SAFE_OFF = "safe_off"
    READY = "ready"
    ARMED = "armed"
    RUNNING = "running"
    PROTECTIVE_STOP = "protective_stop"
    E_STOP = "e_stop"
    FAULT = "fault"
    CLOSED = "closed"


class StopLevel(str, Enum):
    NONE = "none"
    HOLD = "hold"
    CONTROLLED = "controlled"
    EMERGENCY = "emergency"


class GoalStatus(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


def _finite_scalar(value: float, name: str, *, non_negative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _readonly_vector(value: Any, size: int | None, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1).copy()
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values, got {result.size}")
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result


def _readonly_image(
    value: Any,
    *,
    name: str,
    ndim: int,
    dtype: Any | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.ndim}")
    if 0 in result.shape:
        raise ValueError(f"{name} cannot contain an empty dimension")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SampleHeader:
    """Timing and sequence metadata for a device sample."""

    sample_time: float
    received_time: float
    sequence: int
    source_time: float | None = None
    clock_epoch: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_time",
            _finite_scalar(self.sample_time, "sample_time", non_negative=True),
        )
        object.__setattr__(
            self,
            "received_time",
            _finite_scalar(self.received_time, "received_time", non_negative=True),
        )
        if self.source_time is not None:
            object.__setattr__(
                self,
                "source_time",
                _finite_scalar(self.source_time, "source_time", non_negative=True),
            )
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative")
        if int(self.clock_epoch) < 0:
            raise ValueError("clock_epoch must be non-negative")

    def age(self, now: float) -> float:
        """Return non-negative sample age on the shared monotonic clock."""

        current = _finite_scalar(now, "now", non_negative=True)
        return max(0.0, current - self.sample_time)


@dataclass(frozen=True)
class DeviceHealth:
    connected: bool
    ready: bool
    fault_code: str | None = None
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return bool(self.connected and self.ready and not self.fault_code)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = field(default_factory=tuple)
    distortion_model: str = "unknown"

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("camera width and height must be positive")
        for name in ("fx", "fy", "cx", "cy"):
            value = _finite_scalar(getattr(self, name), name)
            if name in {"fx", "fy"} and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not all(np.isfinite(float(item)) for item in self.distortion):
            raise ValueError("distortion must contain finite values")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=float,
        )


@dataclass(frozen=True)
class RGBDFrame:
    """A depth frame in metres, with optional RGB and cell extrinsics."""

    header: SampleHeader
    depth_m: np.ndarray
    depth_intrinsics: CameraIntrinsics
    rgb: np.ndarray | None = None
    color_intrinsics: CameraIntrinsics | None = None
    camera_to_world_cv: np.ndarray | None = None
    serial_number: str = ""
    calibration_id: str = ""
    source: str = "unknown"
    rgb_aligned_to_depth: bool = False
    timestamp_metadata: dict[str, Any] | None = None
    # Optional original stereo input from the same SDK frameset.
    # depth_m remains the device depth used by the existing velocity source.
    left_ir: np.ndarray | None = None
    right_ir: np.ndarray | None = None
    left_ir_intrinsics: CameraIntrinsics | None = None
    right_ir_intrinsics: CameraIntrinsics | None = None
    stereo_baseline_m: float | None = None
    stereo_calibration: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        depth = _readonly_image(self.depth_m, name="depth_m", ndim=2, dtype=np.float32)
        expected_shape = (self.depth_intrinsics.height, self.depth_intrinsics.width)
        if depth.shape != expected_shape:
            raise ValueError(
                f"depth_m shape {depth.shape} does not match intrinsics {expected_shape}"
            )
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise ValueError("depth_m must contain finite, non-negative metres")
        object.__setattr__(self, "depth_m", depth)
        if self.rgb is not None:
            rgb = _readonly_image(self.rgb, name="rgb", ndim=3, dtype=np.uint8)
            if rgb.shape[2] != 3:
                raise ValueError("rgb must have exactly three channels")
            if self.color_intrinsics is not None and rgb.shape[:2] != (
                self.color_intrinsics.height,
                self.color_intrinsics.width,
            ):
                raise ValueError("rgb shape does not match color_intrinsics")
            object.__setattr__(self, "rgb", rgb)
        for side in ("left", "right"):
            image = getattr(self, f"{side}_ir")
            if image is None:
                continue
            image = _readonly_image(image, name=f"{side}_ir", ndim=2, dtype=np.uint8)
            intrinsic = getattr(self, f"{side}_ir_intrinsics")
            if intrinsic is None or image.shape != (intrinsic.height, intrinsic.width):
                raise ValueError(f"{side}_ir shape must match its actual IR intrinsics")
            object.__setattr__(self, f"{side}_ir", image)
        if self.stereo_baseline_m is not None:
            baseline = _finite_scalar(self.stereo_baseline_m, "stereo_baseline_m")
            if baseline <= 0.0:
                raise ValueError("stereo_baseline_m must be positive metres")
        if self.camera_to_world_cv is not None:
            transform = np.asarray(self.camera_to_world_cv, dtype=float).copy()
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError("camera_to_world_cv must be a finite 4x4 transform")
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError("camera_to_world_cv must be a homogeneous transform")
            transform.setflags(write=False)
            object.__setattr__(self, "camera_to_world_cv", transform)

    @property
    def calibrated(self) -> bool:
        return bool(self.calibration_id and self.camera_to_world_cv is not None)


@dataclass(frozen=True)
class RobotState:
    header: SampleHeader
    health: DeviceHealth
    joint_names: tuple[str, ...]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_effort: np.ndarray | None = None
    ee_pose_wxyz: np.ndarray | None = None
    enabled: bool = False
    control_mode: str = "unknown"
    protective_stopped: bool = False
    emergency_stopped: bool = False

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if count == 0 or len(set(self.joint_names)) != count:
            raise ValueError("joint_names must be non-empty and unique")
        object.__setattr__(
            self,
            "joint_position",
            _readonly_vector(self.joint_position, count, "joint_position"),
        )
        object.__setattr__(
            self,
            "joint_velocity",
            _readonly_vector(self.joint_velocity, count, "joint_velocity"),
        )
        if self.joint_effort is not None:
            object.__setattr__(
                self,
                "joint_effort",
                _readonly_vector(self.joint_effort, count, "joint_effort"),
            )
        if self.ee_pose_wxyz is not None:
            object.__setattr__(
                self,
                "ee_pose_wxyz",
                _readonly_vector(self.ee_pose_wxyz, 7, "ee_pose_wxyz"),
            )


@dataclass(frozen=True)
class GripperState:
    header: SampleHeader
    health: DeviceHealth
    opening_width_m: float
    open_fraction: float
    moving: bool = False
    holding: bool = False
    left_contact: bool = False
    right_contact: bool = False
    effort_n: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opening_width_m",
            _finite_scalar(self.opening_width_m, "opening_width_m", non_negative=True),
        )
        fraction = _finite_scalar(self.open_fraction, "open_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("open_fraction must be in [0, 1]")
        if self.effort_n is not None:
            object.__setattr__(
                self,
                "effort_n",
                _finite_scalar(self.effort_n, "effort_n", non_negative=True),
            )


@dataclass(frozen=True)
class ConveyorState:
    header: SampleHeader
    health: DeviceHealth
    velocity_m_s: float
    position_m: float | None = None
    commanded_velocity_m_s: float | None = None
    running: bool = False
    jammed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "velocity_m_s", _finite_scalar(self.velocity_m_s, "velocity_m_s")
        )
        for name in ("position_m", "commanded_velocity_m_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_scalar(value, name))


@dataclass(frozen=True)
class SafetySignals:
    header: SampleHeader
    health: DeviceHealth
    emergency_stop: bool = False
    protective_stop: bool = False
    guard_closed: bool = True
    reset_required: bool = False
    automatic_mode: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def permits_arming(self) -> bool:
        return bool(
            self.health.healthy
            and not self.emergency_stop
            and not self.protective_stop
            and self.guard_closed
            and not self.reset_required
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    positions: np.ndarray
    time_from_start: float
    velocities: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = _readonly_vector(self.positions, None, "positions")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(
            self,
            "time_from_start",
            _finite_scalar(
                self.time_from_start, "time_from_start", non_negative=True
            ),
        )
        if self.velocities is not None:
            object.__setattr__(
                self,
                "velocities",
                _readonly_vector(self.velocities, positions.size, "velocities"),
            )


@dataclass(frozen=True)
class TrajectoryGoal:
    goal_id: str
    sequence: int
    scene_epoch: int
    created_at: float
    expires_at: float
    joint_names: tuple[str, ...]
    points: tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id cannot be empty")
        if int(self.sequence) < 0 or int(self.scene_epoch) < 0:
            raise ValueError("sequence and scene_epoch must be non-negative")
        created = _finite_scalar(self.created_at, "created_at", non_negative=True)
        expires = _finite_scalar(self.expires_at, "expires_at", non_negative=True)
        if expires <= created:
            raise ValueError("expires_at must be later than created_at")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if not self.points:
            raise ValueError("trajectory must contain at least one point")
        previous = -1.0
        for point in self.points:
            if point.positions.size != len(self.joint_names):
                raise ValueError("trajectory point size does not match joint_names")
            if point.time_from_start <= previous:
                raise ValueError("trajectory times must be strictly increasing")
            previous = point.time_from_start


@dataclass(frozen=True)
class CommandReceipt:
    command_id: str
    status: GoalStatus
    timestamp: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id cannot be empty")
        object.__setattr__(
            self,
            "timestamp",
            _finite_scalar(self.timestamp, "timestamp", non_negative=True),
        )


@dataclass(frozen=True)
class PlatformCommand:
    command_id: str
    created_at: float
    expires_at: float
    scene_epoch: int
    trajectory: TrajectoryGoal | None = None
    gripper_open_fraction: float | None = None
    conveyor_velocity_m_s: float | None = None

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("command_id cannot be empty")
        created = _finite_scalar(self.created_at, "created_at", non_negative=True)
        expires = _finite_scalar(self.expires_at, "expires_at", non_negative=True)
        if expires <= created:
            raise ValueError("expires_at must be later than created_at")
        if int(self.scene_epoch) < 0:
            raise ValueError("scene_epoch must be non-negative")
        if self.gripper_open_fraction is not None:
            fraction = _finite_scalar(
                self.gripper_open_fraction, "gripper_open_fraction"
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("gripper_open_fraction must be in [0, 1]")
        if self.conveyor_velocity_m_s is not None:
            object.__setattr__(
                self,
                "conveyor_velocity_m_s",
                _finite_scalar(
                    self.conveyor_velocity_m_s, "conveyor_velocity_m_s"
                ),
            )
        if (
            self.trajectory is None
            and self.gripper_open_fraction is None
            and self.conveyor_velocity_m_s is None
        ):
            raise ValueError("platform command contains no operation")


@dataclass(frozen=True)
class PlatformSnapshot:
    timestamp: float
    scene_epoch: int
    robot: RobotState
    gripper: GripperState
    conveyor: ConveyorState
    safety: SafetySignals
    camera: RGBDFrame | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp",
            _finite_scalar(self.timestamp, "timestamp", non_negative=True),
        )
        if int(self.scene_epoch) < 0:
            raise ValueError("scene_epoch must be non-negative")


@dataclass(frozen=True)
class SafetyDecision:
    state: RuntimeState
    motion_allowed: bool
    new_grasp_allowed: bool
    stop_level: StopLevel
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """One confirmed object's immutable identity and one task's geometry version.

    The first time is the actual accepted RGB-D exposure on host monotonic time,
    never inference completion. Peer track numbers are provenance, not Object IDs.
    """
    session_id: str
    object_id: str
    task_id: str
    first_valid_observed_time: float | None
    clock_epoch: int | None
    geometry_revision: int = 0
    track_id: int | None = None
    track_generation: int | None = None

    @classmethod
    def confirmed(cls, observed_time, clock_epoch, *, session_id=None):
        import math
        import uuid
        if observed_time is not None and not math.isfinite(observed_time):
            raise ValueError('first observation time must be finite')
        return cls(session_id or uuid.uuid4().hex, uuid.uuid4().hex,
                   uuid.uuid4().hex, None if observed_time is None else float(observed_time),
                   None if clock_epoch is None else int(clock_epoch))

    def new_task(self):
        from dataclasses import replace
        import uuid
        return replace(self, task_id=uuid.uuid4().hex)

    def same_task(self, other):
        return isinstance(other, TargetIdentity) and (
            self.session_id, self.object_id, self.task_id) == (
            other.session_id, other.object_id, other.task_id)


def identity_dict(identity):
    from dataclasses import asdict
    return None if identity is None else asdict(identity)


def task_lifecycle(stage, *, complete=False, failed=False):
    """Diagnostic view of the existing FSM; never a second decision machine."""
    if failed:return 'FAILED'
    if complete:return 'COMPLETED'
    value = getattr(stage, 'value', stage)
    return {'waiting':'DETECTED','idle':'DETECTED','searching':'DETECTED','selecting':'TRACKING',
        'pregrasp':'PLANNING','tracking_descent':'EXECUTING','closing':'EXECUTING',
        'lifting':'GRASPED','transferring':'GRASPED','placing':'GRASPED',
        'releasing':'EXECUTING','retreating':'EXECUTING','returning':'EXECUTING',
        'recovering':'FAILED','done':'COMPLETED'}.get(value, 'TRACKING')

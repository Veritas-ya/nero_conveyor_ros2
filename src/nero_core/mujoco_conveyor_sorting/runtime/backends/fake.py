"""Deterministic in-memory backends for contract and fault-injection tests."""

from __future__ import annotations

import time
import uuid
from typing import Callable

import numpy as np

from ..types import (
    BackendError,
    CameraIntrinsics,
    CommandReceipt,
    ConveyorState,
    DeviceHealth,
    GoalStatus,
    GripperState,
    RGBDFrame,
    RobotState,
    SafetySignals,
    SampleHeader,
    TrajectoryGoal,
)


class _FakeBase:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.connected = False
        self.ready = False
        self.fault_code: str | None = None
        self.detail = ""
        self.sequence = 0
        self.freeze_samples = False
        self._sample_time = float(clock())

    def connect(self) -> None:
        if self.fault_code:
            raise BackendError(self.fault_code)
        self.connected = True
        self.ready = True
        self._sample_time = float(self.clock())

    def health(self) -> DeviceHealth:
        return DeviceHealth(self.connected, self.ready, self.fault_code, self.detail)

    def set_fault(self, code: str, detail: str = "") -> None:
        self.fault_code = code
        self.detail = detail
        self.ready = False

    def clear_fault(self) -> None:
        self.fault_code = None
        self.detail = ""
        self.ready = self.connected

    def header(self) -> SampleHeader:
        now = float(self.clock())
        if not self.freeze_samples:
            self.sequence += 1
            self._sample_time = now
        return SampleHeader(self._sample_time, now, self.sequence)

    def close(self) -> None:
        self.connected = False
        self.ready = False


class FakeRobotBackend(_FakeBase):
    def __init__(
        self,
        joint_names: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8)),
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(clock=clock)
        self._joint_names = tuple(joint_names)
        self.position = np.zeros(len(joint_names), dtype=float)
        self.velocity = np.zeros(len(joint_names), dtype=float)
        self.effort = np.zeros(len(joint_names), dtype=float)
        self.enabled = False
        self.protective_stopped = False
        self.emergency_stopped = False
        self.active_goal_id: str | None = None
        self.hold_count = 0
        self.cancel_count = 0

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    def read_state(self) -> RobotState:
        return RobotState(
            self.header(),
            self.health(),
            self.joint_names,
            self.position,
            self.velocity,
            self.effort,
            enabled=self.enabled,
            control_mode="position",
            protective_stopped=self.protective_stopped,
            emergency_stopped=self.emergency_stopped,
        )

    def enable(self) -> None:
        if not self.health().healthy:
            raise BackendError("fake robot is not healthy")
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def submit_trajectory(self, goal: TrajectoryGoal) -> CommandReceipt:
        now = float(self.clock())
        if not self.enabled:
            return CommandReceipt(goal.goal_id, GoalStatus.REJECTED, now, "robot disabled")
        if goal.joint_names != self.joint_names:
            return CommandReceipt(goal.goal_id, GoalStatus.REJECTED, now, "joint mismatch")
        self.active_goal_id = goal.goal_id
        self.position = goal.points[-1].positions.copy()
        self.velocity = np.zeros_like(self.position)
        return CommandReceipt(goal.goal_id, GoalStatus.ACCEPTED, now)

    def cancel(self, goal_id: str | None = None) -> None:
        if goal_id is None or goal_id == self.active_goal_id:
            self.active_goal_id = None
        self.cancel_count += 1

    def hold(self) -> None:
        self.velocity[:] = 0.0
        self.hold_count += 1


class FakeGripperBackend(_FakeBase):
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock=clock)
        self.open_fraction = 1.0
        self.opening_width_m = 0.10
        self.hold_count = 0

    def read_state(self) -> GripperState:
        return GripperState(
            self.header(),
            self.health(),
            self.opening_width_m,
            self.open_fraction,
        )

    def command_open_fraction(self, fraction: float) -> CommandReceipt:
        if not self.health().healthy:
            raise BackendError("fake gripper is not healthy")
        self.open_fraction = float(fraction)
        self.opening_width_m = 0.10 * self.open_fraction
        return CommandReceipt(str(uuid.uuid4()), GoalStatus.ACCEPTED, float(self.clock()))

    def hold(self) -> None:
        self.hold_count += 1


class FakeConveyorBackend(_FakeBase):
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock=clock)
        self.velocity_m_s = 0.0
        self.position_m = 0.0
        self.jammed = False
        self.stop_count = 0

    def read_state(self) -> ConveyorState:
        return ConveyorState(
            self.header(),
            self.health(),
            self.velocity_m_s,
            self.position_m,
            self.velocity_m_s,
            running=abs(self.velocity_m_s) > 1e-6,
            jammed=self.jammed,
        )

    def command_velocity(self, velocity_m_s: float) -> CommandReceipt:
        if not self.health().healthy:
            raise BackendError("fake conveyor is not healthy")
        self.velocity_m_s = float(velocity_m_s)
        return CommandReceipt(str(uuid.uuid4()), GoalStatus.ACCEPTED, float(self.clock()))

    def stop(self) -> None:
        self.velocity_m_s = 0.0
        self.stop_count += 1


class FakeSafetyBackend(_FakeBase):
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(clock=clock)
        self.emergency_stop = False
        self.protective_stop = False
        self.guard_closed = True
        self.reset_required = False
        self.automatic_mode = True

    def read_state(self) -> SafetySignals:
        return SafetySignals(
            self.header(),
            self.health(),
            emergency_stop=self.emergency_stop,
            protective_stop=self.protective_stop,
            guard_closed=self.guard_closed,
            reset_required=self.reset_required,
            automatic_mode=self.automatic_mode,
            details={},
        )


class FakeCameraBackend(_FakeBase):
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        width: int = 64,
        height: int = 48,
        calibrated: bool = True,
    ) -> None:
        super().__init__(clock=clock)
        self.intrinsics = CameraIntrinsics(
            width, height, 60.0, 60.0, 0.5 * width, 0.5 * height
        )
        self.depth = np.ones((height, width), dtype=np.float32)
        self.calibrated = calibrated

    def start(self) -> None:
        self.connect()

    def read_latest(
        self, timeout_s: float = 0.0, *, after_sequence: int | None = None
    ) -> RGBDFrame:
        if not self.health().healthy:
            raise BackendError("fake camera is not healthy")
        header = self.header()
        if after_sequence is not None and header.sequence <= after_sequence:
            raise BackendError("no newer fake frame available")
        return RGBDFrame(
            header,
            self.depth,
            self.intrinsics,
            camera_to_world_cv=np.eye(4) if self.calibrated else None,
            serial_number="FAKE-ORBBEC",
            calibration_id="fake-calibration" if self.calibrated else "",
            source="fake",
        )

    def stop(self) -> None:
        self.close()

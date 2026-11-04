"""Fail-closed orchestration for physical sorting-cell backends."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .interfaces import (
    CameraBackend,
    ConveyorBackend,
    GripperBackend,
    RobotBackend,
    SafetyBackend,
)
from .safety import SafetyConfig, SafetySupervisor
from .types import (
    BackendError,
    CommandReceipt,
    GoalStatus,
    MotionRejected,
    PlatformCommand,
    PlatformSnapshot,
    RuntimeState,
)


class HardwareRuntime:
    """Compose real or fake backends behind one guarded runtime boundary.

    ``start`` connects devices and finishes in ``READY``.  It never enables
    motion.  The caller must explicitly call ``arm`` and ``start_auto``.
    """

    def __init__(
        self,
        *,
        robot: RobotBackend,
        gripper: GripperBackend,
        camera: CameraBackend,
        conveyor: ConveyorBackend,
        safety_io: SafetyBackend,
        safety_config: SafetyConfig | Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        camera_read_timeout_s: float = 0.25,
    ) -> None:
        if camera_read_timeout_s <= 0.0:
            raise ValueError("camera_read_timeout_s must be positive")
        self.robot = robot
        self.gripper = gripper
        self.camera = camera
        self.conveyor = conveyor
        self.safety_io = safety_io
        self.clock = clock
        self.camera_read_timeout_s = float(camera_read_timeout_s)
        if isinstance(safety_config, Mapping):
            safety_config = SafetyConfig.from_mapping(safety_config)
        self.supervisor = SafetySupervisor(safety_config, clock=clock)
        self.scene_epoch = 0
        self._started = False
        self._closed = False
        self._stop_enforced_for: RuntimeState | None = None

    @property
    def state(self) -> RuntimeState:
        return self.supervisor.state

    def start(self) -> PlatformSnapshot:
        """Connect and self-test all devices without enabling the robot."""

        if self._closed:
            raise RuntimeError("cannot start a closed runtime")
        if self._started:
            return self.snapshot()
        try:
            # Observe the external safety chain first and connect the robot last.
            self.safety_io.connect()
            self.conveyor.connect()
            self.gripper.connect()
            self.camera.start()
            self.robot.connect()
            self.supervisor.connected()
            snapshot = self._read_snapshot(evaluate=False)
            reasons = self._self_test_reasons(snapshot)
            self.supervisor.complete_self_test(not reasons, "; ".join(reasons))
            if reasons:
                self._enforce_stop()
                raise BackendError("hardware self-test failed: " + "; ".join(reasons))
            self.supervisor.mark_ready(snapshot.safety)
            self._started = True
            return snapshot
        except Exception:
            self._safe_close_backends()
            raise

    def arm(self) -> PlatformSnapshot:
        """Enable the robot only after a fresh, safe snapshot is available."""

        self._require_started()
        snapshot = self.snapshot()
        if self.state is not RuntimeState.READY:
            raise MotionRejected(f"runtime must be ready before arming, got {self.state.value}")
        self.supervisor.arm(snapshot.safety)
        try:
            self.robot.enable()
        except Exception as error:
            self.supervisor.trip_fault(f"robot enable failed: {error}")
            self._enforce_stop()
            raise BackendError(f"robot enable failed: {error}") from error
        # Re-read so a controller that rejected enable cannot be treated as armed.
        armed_snapshot = self._read_snapshot(evaluate=False)
        if not armed_snapshot.robot.enabled:
            self.supervisor.trip_fault("robot did not report enabled after enable request")
            self._enforce_stop()
            raise BackendError("robot did not report enabled after enable request")
        return armed_snapshot

    def start_auto(self) -> PlatformSnapshot:
        self._require_started()
        if self.state is not RuntimeState.ARMED:
            raise MotionRejected(
                f"runtime must be armed before automatic motion, got {self.state.value}"
            )
        snapshot = self.snapshot()
        # The transition itself is explicit; the next snapshot supplies the
        # continuously evaluated freshness and interlock gate.
        self.supervisor.start_auto()
        decision = self.supervisor.evaluate(snapshot)
        if not decision.motion_allowed:
            self._enforce_stop()
            raise MotionRejected("automatic start rejected: " + "; ".join(decision.reasons))
        return snapshot

    def snapshot(self) -> PlatformSnapshot:
        self._require_started()
        try:
            snapshot = self._read_snapshot(evaluate=False)
        except Exception as error:
            self.supervisor.trip_fault(f"state acquisition failed: {error}")
            self._enforce_stop(force=True)
            if isinstance(error, BackendError):
                raise
            raise BackendError(f"state acquisition failed: {error}") from error
        decision = self.supervisor.evaluate(snapshot)
        if decision.stop_level.value != "none":
            self._enforce_stop()
        else:
            self._stop_enforced_for = None
        return snapshot

    def submit(self, command: PlatformCommand) -> tuple[CommandReceipt, ...]:
        """Validate and dispatch one bounded command transaction."""

        snapshot = self.snapshot()
        self.supervisor.assert_motion_allowed(snapshot)
        now = float(self.clock())
        if now >= command.expires_at:
            raise MotionRejected(f"command {command.command_id!r} is expired")
        if command.scene_epoch != self.scene_epoch:
            raise MotionRejected(
                f"command scene epoch {command.scene_epoch} != runtime epoch {self.scene_epoch}"
            )
        if command.trajectory is not None:
            goal = command.trajectory
            if goal.scene_epoch != self.scene_epoch:
                raise MotionRejected(
                    f"trajectory scene epoch {goal.scene_epoch} != runtime epoch {self.scene_epoch}"
                )
            if now >= goal.expires_at:
                raise MotionRejected(f"trajectory {goal.goal_id!r} is expired")
            if goal.joint_names != self.robot.joint_names:
                raise MotionRejected("trajectory joint names do not match robot backend")
        receipts: list[CommandReceipt] = []
        try:
            if command.trajectory is not None:
                goal = command.trajectory
                receipt = self.robot.submit_trajectory(goal)
                if receipt.status is GoalStatus.REJECTED:
                    raise MotionRejected(
                        receipt.detail or f"trajectory {goal.goal_id!r} was rejected"
                    )
                receipts.append(receipt)
            if command.gripper_open_fraction is not None:
                receipts.append(
                    self.gripper.command_open_fraction(command.gripper_open_fraction)
                )
            if command.conveyor_velocity_m_s is not None:
                receipts.append(
                    self.conveyor.command_velocity(command.conveyor_velocity_m_s)
                )
        except MotionRejected:
            raise
        except Exception as error:
            self.supervisor.trip_fault(f"command dispatch failed: {error}")
            self._enforce_stop()
            raise BackendError(f"command dispatch failed: {error}") from error
        return tuple(receipts)

    def controlled_stop(self, reason: str = "operator stop") -> None:
        if self._closed:
            return
        self.supervisor.request_controlled_stop(reason)
        self._enforce_stop(force=True)

    def reset_safety(self) -> PlatformSnapshot:
        self._require_started()
        signals = self.safety_io.read_state()
        self.supervisor.reset_latch(signals)
        self.scene_epoch += 1
        self._stop_enforced_for = None
        self.supervisor.mark_ready(signals)
        return self.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.controlled_stop("runtime shutdown")
            try:
                self.robot.disable()
            except Exception:
                pass
        finally:
            self._safe_close_backends()
            self.supervisor.close()
            self._closed = True
            self._started = False

    def _read_snapshot(self, *, evaluate: bool) -> PlatformSnapshot:
        now = float(self.clock())
        camera = self.camera.read_latest(timeout_s=self.camera_read_timeout_s)
        snapshot = PlatformSnapshot(
            timestamp=now,
            scene_epoch=self.scene_epoch,
            robot=self.robot.read_state(),
            gripper=self.gripper.read_state(),
            conveyor=self.conveyor.read_state(),
            safety=self.safety_io.read_state(),
            camera=camera,
        )
        if evaluate:
            self.supervisor.evaluate(snapshot)
        return snapshot

    def base_observation(self, snapshot: PlatformSnapshot) -> dict[str, object]:
        """Build the hardware-neutral portion of the existing policy observation."""

        camera = snapshot.camera
        gripper = snapshot.gripper
        observation: dict[str, object] = {
            "joint_pos": snapshot.robot.joint_position.copy(),
            "joint_vel": snapshot.robot.joint_velocity.copy(),
            "ee_pose": (
                None
                if snapshot.robot.ee_pose_wxyz is None
                else snapshot.robot.ee_pose_wxyz.copy()
            ),
            "gripper": {
                "opening_width": gripper.opening_width_m,
                "open_fraction": gripper.open_fraction,
                "is_holding": gripper.holding,
                "left_finger_contact": gripper.left_contact,
                "right_finger_contact": gripper.right_contact,
                "moving": gripper.moving,
            },
            "gripper_opening": gripper.opening_width_m,
            "belt_speed": snapshot.conveyor.velocity_m_s,
            "belt_velocity": [snapshot.conveyor.velocity_m_s, 0.0, 0.0],
            "timestamp": snapshot.timestamp,
            "scene_epoch": snapshot.scene_epoch,
            "severe_collision": snapshot.robot.protective_stopped,
            "runtime_state": self.state.value,
        }
        if camera is not None:
            observation.update(
                {
                    "top_depth": camera.depth_m.copy(),
                    "top_rgb": None if camera.rgb is None else camera.rgb.copy(),
                    "camera_timestamp": camera.header.sample_time,
                    "top_depth_timestamp": camera.header.sample_time,
                    "top_rgb_aligned_to_depth": camera.rgb_aligned_to_depth,
                    "top_camera": {
                        "intrinsics": camera.depth_intrinsics.matrix,
                        "camera_to_world_opencv": (
                            None
                            if camera.camera_to_world_cv is None
                            else camera.camera_to_world_cv.copy()
                        ),
                        "camera_convention": "opencv_x_right_y_down_z_forward",
                        "width": camera.depth_intrinsics.width,
                        "height": camera.depth_intrinsics.height,
                        "serial_number": camera.serial_number,
                        "calibration_id": camera.calibration_id,
                    },
                }
            )
        return observation

    def _self_test_reasons(self, snapshot: PlatformSnapshot) -> list[str]:
        reasons: list[str] = []
        for name, health in (
            ("safety_io", self.safety_io.health()),
            ("conveyor", self.conveyor.health()),
            ("gripper", self.gripper.health()),
            ("camera", self.camera.health()),
            ("robot", self.robot.health()),
        ):
            if not health.healthy:
                reasons.append(
                    f"{name}: {health.fault_code or health.detail or 'not ready'}"
                )
        reasons.extend(self.supervisor.preflight_reasons(snapshot))
        return list(dict.fromkeys(reasons))

    def _enforce_stop(self, *, force: bool = False) -> None:
        state = self.state
        if not force and self._stop_enforced_for is state:
            return
        # These calls are idempotent by backend contract.  Continue through all
        # of them even if one device has already failed.
        failures: list[str] = []
        for name, operation in (
            ("robot cancel", lambda: self.robot.cancel(None)),
            ("robot hold", self.robot.hold),
            ("conveyor stop", self.conveyor.stop),
            ("gripper hold", self.gripper.hold),
        ):
            try:
                operation()
            except Exception as error:
                failures.append(f"{name}: {error}")
        if failures:
            self.supervisor.trip_fault(
                "stop enforcement incomplete: " + "; ".join(failures)
            )
        self._stop_enforced_for = state

    def _safe_close_backends(self) -> None:
        for backend in (
            self.robot,
            self.camera,
            self.gripper,
            self.conveyor,
            self.safety_io,
        ):
            try:
                backend.close()
            except Exception:
                pass

    def _require_started(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if not self._started:
            raise RuntimeError("runtime has not been started")

    def __enter__(self) -> "HardwareRuntime":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

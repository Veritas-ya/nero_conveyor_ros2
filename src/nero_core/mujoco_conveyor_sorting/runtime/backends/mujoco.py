"""Adapter exposing the existing MuJoCo environment through runtime states."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..safety import SafetyConfig, SafetySupervisor
from ..types import (
    CameraIntrinsics,
    ConveyorState,
    DeviceHealth,
    GripperState,
    MotionRejected,
    PlatformSnapshot,
    RGBDFrame,
    RobotState,
    RuntimeState,
    SafetySignals,
    SampleHeader,
)


class MujocoRuntime:
    """Safety-gated compatibility wrapper around ``ConveyorSortingEnv``.

    The wrapper preserves the existing Gym-like ``reset``/``step`` contract,
    while publishing the exact same typed snapshot schema used by hardware.
    It is intentionally an adapter, so the mature policy and planner do not
    need a disruptive rewrite before real backends are ready.
    """

    def __init__(
        self,
        env: Any,
        *,
        safety_config: SafetyConfig | None = None,
        auto_start: bool = True,
        close_environment: bool = True,
    ) -> None:
        self.env = env
        self.close_environment = bool(close_environment)
        self.supervisor = SafetySupervisor(
            safety_config or SafetyConfig(require_camera_calibration=True),
            clock=lambda: float(self.env.data.time),
        )
        self._sequence = 0
        self._observation = env.get_observation()
        self._closed = False
        if auto_start:
            self.start()

    @property
    def state(self) -> RuntimeState:
        return self.supervisor.state

    def start(self) -> PlatformSnapshot:
        if self._closed:
            raise RuntimeError("cannot start a closed MuJoCo runtime")
        if self.state is RuntimeState.RUNNING:
            return self.snapshot()
        if self.state is RuntimeState.DISCONNECTED:
            self.supervisor.connected()
            self.supervisor.complete_self_test(True)
        snapshot = self._snapshot_from_observation(self._observation)
        if self.state is RuntimeState.SAFE_OFF:
            self.supervisor.mark_ready(snapshot.safety)
        if self.state is RuntimeState.READY:
            self.supervisor.arm(snapshot.safety)
        if self.state is RuntimeState.ARMED:
            self.supervisor.start_auto()
        return self.snapshot()

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot reset a closed MuJoCo runtime")
        self._observation = self.env.reset(seed=seed, options=options)
        if self.state in {
            RuntimeState.PROTECTIVE_STOP,
            RuntimeState.E_STOP,
            RuntimeState.FAULT,
        }:
            signals = self._safety_signals(float(self._observation["timestamp"]))
            self.supervisor.reset_latch(signals)
        self.start()
        return self._observation

    def step(
        self, action: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("cannot step a closed MuJoCo runtime")
        snapshot = self.snapshot()
        self.supervisor.assert_motion_allowed(snapshot)
        result = self.env.step(action)
        self._observation = result[0]
        if bool(self._observation.get("severe_collision", False)):
            # Latch the stop before the policy can issue a recovery motion on
            # the next control tick.
            self.supervisor.trip_protective_stop("MuJoCo severe collision")
            self.controlled_stop("MuJoCo severe collision")
        return result

    def snapshot(self) -> PlatformSnapshot:
        if self._closed:
            raise RuntimeError("MuJoCo runtime is closed")
        snapshot = self._snapshot_from_observation(self._observation)
        self.supervisor.evaluate(snapshot)
        return snapshot

    def controlled_stop(self, reason: str = "operator stop") -> None:
        self.supervisor.request_controlled_stop(reason)
        observation = self._observation
        current = np.asarray(observation["joint_pos"], dtype=float)
        if self.env.robot_controller is not None:
            self.env.robot_controller.reset()
        self.env._action_queue.clear()
        self.env.data.ctrl[self.env.arm_actuator_ids] = current
        self.env._joint_target = current.copy()
        self.env._joint_target_velocity[:] = 0.0
        self.env.set_belt_speed(0.0)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.controlled_stop("runtime shutdown")
        finally:
            if self.close_environment:
                self.env.close()
            self.supervisor.close()
            self._closed = True

    def _snapshot_from_observation(
        self, observation: Mapping[str, Any]
    ) -> PlatformSnapshot:
        now = float(observation["timestamp"])
        self._sequence += 1
        header = SampleHeader(now, now, self._sequence)
        health = DeviceHealth(True, True, detail="MuJoCo simulation")
        controller = self.env.robot_controller
        joint_names = (
            tuple(controller.joint_names)
            if controller is not None
            else tuple(f"joint{i}" for i in range(1, 8))
        )
        gripper_document = dict(observation.get("gripper", {}))
        robot = RobotState(
            header,
            health,
            joint_names,
            np.asarray(observation["joint_pos"], dtype=float),
            np.asarray(observation["joint_vel"], dtype=float),
            ee_pose_wxyz=np.asarray(observation["ee_pose"], dtype=float),
            enabled=True,
            control_mode=str(self.env.action_type),
            protective_stopped=bool(observation.get("severe_collision", False)),
        )
        status = str(gripper_document.get("status", "unknown")).lower()
        gripper = GripperState(
            header,
            health,
            float(gripper_document.get("opening_width", 0.0)),
            float(gripper_document.get("open_fraction", 0.0)),
            moving=status in {"opening", "closing", "moving"},
            holding=bool(gripper_document.get("is_holding", False)),
            left_contact=bool(gripper_document.get("left_finger_contact", False)),
            right_contact=bool(gripper_document.get("right_finger_contact", False)),
        )
        speed = float(observation.get("belt_speed", 0.0))
        conveyor = ConveyorState(
            header,
            health,
            speed,
            commanded_velocity_m_s=speed,
            running=abs(speed) > 1e-9,
        )
        safety = self._safety_signals(now)
        camera = self._camera_frame(observation)
        return PlatformSnapshot(
            now,
            int(observation.get("scene_epoch", 0)),
            robot,
            gripper,
            conveyor,
            safety,
            camera,
        )

    def _safety_signals(self, now: float) -> SafetySignals:
        header = SampleHeader(now, now, self._sequence)
        return SafetySignals(
            header,
            DeviceHealth(True, True, detail="simulated safety chain"),
            protective_stop=bool(self._observation.get("severe_collision", False)),
            guard_closed=True,
            automatic_mode=True,
        )

    def _camera_frame(self, observation: Mapping[str, Any]) -> RGBDFrame:
        camera = dict(observation["top_camera"])
        intrinsics = np.asarray(camera["intrinsics"], dtype=float)
        width = int(camera["width"])
        height = int(camera["height"])
        calibration = CameraIntrinsics(
            width,
            height,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
            distortion_model="ideal_mujoco_pinhole",
        )
        capture_time = float(observation.get("top_depth_timestamp", observation["timestamp"]))
        header = SampleHeader(
            capture_time,
            float(observation["timestamp"]),
            self._sequence,
        )
        rgb = (
            np.asarray(observation["top_rgb"], dtype=np.uint8)
            if bool(observation.get("top_rgb_valid", False))
            else None
        )
        return RGBDFrame(
            header,
            np.asarray(observation["top_depth"], dtype=np.float32),
            calibration,
            rgb=rgb,
            color_intrinsics=calibration if rgb is not None else None,
            camera_to_world_cv=np.asarray(camera["camera_to_world_opencv"], dtype=float),
            serial_number="MUJOCO-TOP-CAMERA",
            calibration_id=f"mujoco-scene-epoch-{observation.get('scene_epoch', 0)}",
            source="mujoco_renderer",
            rgb_aligned_to_depth=True,
        )

    def __enter__(self) -> "MujocoRuntime":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

"""Fail-closed application safety gate for simulation and hardware runtimes.

This supervisor is an application-level interlock.  It deliberately does not
claim to replace a hard-wired emergency-stop circuit, STO, guarding, or a
certified safety PLC.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Mapping

from .types import (
    MotionRejected,
    PlatformSnapshot,
    RuntimeState,
    SafetyDecision,
    SafetySignals,
    StopLevel,
)


@dataclass(frozen=True)
class FreshnessLimits:
    robot_s: float = 0.10
    gripper_s: float = 0.20
    conveyor_s: float = 0.20
    camera_s: float = 0.25
    safety_io_s: float = 0.10

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    freshness: FreshnessLimits = FreshnessLimits()
    require_camera: bool = True
    require_camera_calibration: bool = True
    require_guard_closed: bool = True
    require_automatic_mode: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SafetyConfig":
        document = dict(value)
        freshness_keys = {
            "robot_state_max_age_s": "robot_s",
            "gripper_state_max_age_s": "gripper_s",
            "conveyor_state_max_age_s": "conveyor_s",
            "camera_frame_max_age_s": "camera_s",
            "safety_io_max_age_s": "safety_io_s",
        }
        freshness_values = {
            target: float(document.pop(source))
            for source, target in freshness_keys.items()
            if source in document
        }
        allowed = {
            "require_camera",
            "require_camera_calibration",
            "require_guard_closed",
            "require_automatic_mode",
        }
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise ValueError(f"unknown safety settings: {', '.join(unknown)}")
        return cls(
            freshness=FreshnessLimits(**freshness_values),
            **{name: bool(item) for name, item in document.items()},
        )


class SafetySupervisor:
    """Explicit, latched state machine guarding every platform command."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        *,
        clock: Callable[[], float],
    ) -> None:
        self.config = config or SafetyConfig()
        self._clock = clock
        self._state = RuntimeState.DISCONNECTED
        self._latched_reasons: tuple[str, ...] = ()
        self._lock = RLock()

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    @property
    def latched_reasons(self) -> tuple[str, ...]:
        with self._lock:
            return self._latched_reasons

    def connected(self) -> None:
        with self._lock:
            self._require_state(RuntimeState.DISCONNECTED)
            self._state = RuntimeState.SELF_TEST

    def complete_self_test(self, passed: bool, reason: str = "") -> None:
        with self._lock:
            self._require_state(RuntimeState.SELF_TEST)
            if passed:
                self._state = RuntimeState.SAFE_OFF
                self._latched_reasons = ()
            else:
                self._trip_locked(RuntimeState.FAULT, reason or "self-test failed")

    def mark_ready(self, signals: SafetySignals) -> None:
        with self._lock:
            self._require_state(RuntimeState.SAFE_OFF)
            reasons = self._signal_reasons(signals)
            if reasons:
                self._trip_locked(RuntimeState.FAULT, *reasons)
                raise MotionRejected("; ".join(reasons))
            self._state = RuntimeState.READY

    def arm(self, signals: SafetySignals) -> None:
        with self._lock:
            self._require_state(RuntimeState.READY)
            reasons = self._signal_reasons(signals)
            if reasons:
                self._trip_locked(RuntimeState.FAULT, *reasons)
                raise MotionRejected("; ".join(reasons))
            self._state = RuntimeState.ARMED

    def start_auto(self) -> None:
        with self._lock:
            self._require_state(RuntimeState.ARMED)
            self._state = RuntimeState.RUNNING

    def request_controlled_stop(self, reason: str = "operator stop") -> None:
        with self._lock:
            if self._state in {
                RuntimeState.CLOSED,
                RuntimeState.DISCONNECTED,
                RuntimeState.E_STOP,
                RuntimeState.FAULT,
                RuntimeState.PROTECTIVE_STOP,
            }:
                return
            self._state = RuntimeState.SAFE_OFF
            self._latched_reasons = (reason,) if reason else ()

    def trip_protective_stop(self, reason: str) -> None:
        with self._lock:
            self._trip_locked(RuntimeState.PROTECTIVE_STOP, reason)

    def trip_fault(self, reason: str) -> None:
        with self._lock:
            self._trip_locked(RuntimeState.FAULT, reason)

    def reset_latch(self, signals: SafetySignals) -> None:
        """Clear a software latch only after the external safety chain is safe."""

        with self._lock:
            if self._state not in {
                RuntimeState.PROTECTIVE_STOP,
                RuntimeState.E_STOP,
                RuntimeState.FAULT,
                RuntimeState.SAFE_OFF,
            }:
                raise MotionRejected(f"cannot reset safety latch from {self._state.value}")
            reasons = self._signal_reasons(signals)
            if reasons:
                raise MotionRejected("external safety chain is not reset: " + "; ".join(reasons))
            self._state = RuntimeState.SAFE_OFF
            self._latched_reasons = ()

    def evaluate(self, snapshot: PlatformSnapshot) -> SafetyDecision:
        """Latch new safety faults and return the current command gate."""

        now = float(self._clock())
        with self._lock:
            reasons = self._snapshot_reasons(snapshot, now)
            if self._state in {RuntimeState.ARMED, RuntimeState.RUNNING}:
                if not snapshot.robot.enabled:
                    reasons.append("robot is not enabled")
            if snapshot.safety.emergency_stop:
                self._trip_locked(RuntimeState.E_STOP, *reasons)
            elif snapshot.safety.protective_stop or snapshot.robot.protective_stopped:
                self._trip_locked(RuntimeState.PROTECTIVE_STOP, *reasons)
            elif reasons:
                self._trip_locked(RuntimeState.FAULT, *reasons)
            state = self._state
            motion_allowed = state is RuntimeState.RUNNING and not reasons
            return SafetyDecision(
                state=state,
                motion_allowed=motion_allowed,
                new_grasp_allowed=motion_allowed,
                stop_level=self._stop_level(state),
                reasons=self._latched_reasons,
            )

    def preflight_reasons(self, snapshot: PlatformSnapshot) -> tuple[str, ...]:
        """Return current gating failures without mutating the state machine."""

        return tuple(self._snapshot_reasons(snapshot, float(self._clock())))

    def assert_motion_allowed(self, snapshot: PlatformSnapshot) -> None:
        decision = self.evaluate(snapshot)
        if not decision.motion_allowed:
            reason = "; ".join(decision.reasons) or decision.state.value
            raise MotionRejected(f"motion rejected by safety supervisor: {reason}")

    def close(self) -> None:
        with self._lock:
            self._state = RuntimeState.CLOSED

    def _snapshot_reasons(self, snapshot: PlatformSnapshot, now: float) -> list[str]:
        reasons = self._signal_reasons(snapshot.safety)
        devices = (
            ("robot", snapshot.robot.health, snapshot.robot.header, self.config.freshness.robot_s),
            ("gripper", snapshot.gripper.health, snapshot.gripper.header, self.config.freshness.gripper_s),
            ("conveyor", snapshot.conveyor.health, snapshot.conveyor.header, self.config.freshness.conveyor_s),
            ("safety_io", snapshot.safety.health, snapshot.safety.header, self.config.freshness.safety_io_s),
        )
        for name, health, header, limit in devices:
            if not health.healthy:
                detail = health.fault_code or health.detail or "not ready"
                reasons.append(f"{name} unhealthy: {detail}")
            age = header.age(now)
            if age > limit:
                reasons.append(f"{name} stale: {age:.3f}s > {limit:.3f}s")
        if self.config.require_camera:
            if snapshot.camera is None:
                reasons.append("camera frame unavailable")
            else:
                age = snapshot.camera.header.age(now)
                if age > self.config.freshness.camera_s:
                    reasons.append(
                        f"camera stale: {age:.3f}s > {self.config.freshness.camera_s:.3f}s"
                    )
                if self.config.require_camera_calibration and not snapshot.camera.calibrated:
                    reasons.append("camera-to-world calibration unavailable")
        if snapshot.robot.emergency_stopped:
            reasons.append("robot emergency stop active")
        if snapshot.robot.protective_stopped:
            reasons.append("robot protective stop active")
        if snapshot.conveyor.jammed:
            reasons.append("conveyor jam detected")
        return list(dict.fromkeys(reasons))

    def _signal_reasons(self, signals: SafetySignals) -> list[str]:
        reasons: list[str] = []
        if not signals.health.healthy:
            reasons.append(
                "safety IO unhealthy: "
                + (signals.health.fault_code or signals.health.detail or "not ready")
            )
        if signals.emergency_stop:
            reasons.append("emergency stop active")
        if signals.protective_stop:
            reasons.append("protective stop active")
        if self.config.require_guard_closed and not signals.guard_closed:
            reasons.append("guard is open")
        if signals.reset_required:
            reasons.append("external safety reset required")
        if self.config.require_automatic_mode and not signals.automatic_mode:
            reasons.append("controller is not in automatic mode")
        return reasons

    def _trip_locked(self, state: RuntimeState, *reasons: str) -> None:
        if self._state is RuntimeState.CLOSED:
            return
        cleaned = tuple(dict.fromkeys(item for item in reasons if item))
        if self._state is RuntimeState.E_STOP and state is not RuntimeState.E_STOP:
            return
        self._state = state
        if cleaned:
            self._latched_reasons = tuple(dict.fromkeys((*self._latched_reasons, *cleaned)))

    def _require_state(self, expected: RuntimeState) -> None:
        if self._state is not expected:
            raise MotionRejected(
                f"safety transition requires {expected.value}, current state is {self._state.value}"
            )

    @staticmethod
    def _stop_level(state: RuntimeState) -> StopLevel:
        if state is RuntimeState.E_STOP:
            return StopLevel.EMERGENCY
        if state in {RuntimeState.FAULT, RuntimeState.PROTECTIVE_STOP}:
            return StopLevel.CONTROLLED
        if state in {RuntimeState.SAFE_OFF, RuntimeState.READY, RuntimeState.ARMED}:
            return StopLevel.HOLD
        return StopLevel.NONE

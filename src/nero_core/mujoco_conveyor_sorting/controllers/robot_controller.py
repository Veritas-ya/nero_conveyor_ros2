"""Configuration-driven seven-axis MuJoCo robot controller.

The controller sends ordinary actuator targets.  It never changes ``qpos`` and
never manipulates workpieces, so contacts remain the only default grasping
mechanism.  Position, velocity, and acceleration limits are applied to every
command before it reaches ``data.ctrl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol

import numpy as np

from ._math import (
    as_vector,
    compose_world_delta,
    config_from_mapping,
    normalized_quat,
)
from .ik_solver import DampedLeastSquaresIK, IKConfig, IKResult

try:
    import mujoco
except ImportError:  # pragma: no cover - policy-only installations.
    mujoco = None  # type: ignore[assignment]


class GripperLike(Protocol):
    def command(self, value: float) -> Any:
        """Apply a normalized gripper command."""


@dataclass(frozen=True)
class RobotControllerConfig:
    """Names and safety limits for a position-actuated arm."""

    joint_names: tuple[str, ...] = field(default_factory=tuple)
    actuator_names: tuple[str, ...] = field(default_factory=tuple)
    ee_site_name: str = "ee_site"
    action_mode: str = "joint_position"
    control_timestep: float | None = None
    joint_lower_limits: tuple[float, ...] = field(default_factory=tuple)
    joint_upper_limits: tuple[float, ...] = field(default_factory=tuple)
    # Keep commanded set-points away from the physical/configured hard stops.
    # The raw limits remain available separately to motion planners so this
    # execution guard does not compound a planner's own soft-limit margin.
    joint_command_margin: float = 0.055
    max_joint_velocity: float | tuple[float, ...] = 1.5
    max_joint_acceleration: float | tuple[float, ...] = 6.0
    max_cartesian_delta: float = 0.025
    max_rotation_delta: float = 0.12
    translation_action_scale: float = 1.0
    rotation_action_scale: float = 1.0
    # Fail closed by default.  A non-converged Cartesian solve is diagnostic
    # information, never a motion command for either backend.
    allow_unconverged_ik: bool = False
    ik: IKConfig | Mapping[str, Any] = field(default_factory=IKConfig)
    wait_joint_positions: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.action_mode not in {"joint_position", "ee_delta_pose"}:
            raise ValueError(
                "action_mode must be 'joint_position' or 'ee_delta_pose'"
            )
        if self.control_timestep is not None and self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")
        if self.max_cartesian_delta <= 0.0 or self.max_rotation_delta <= 0.0:
            raise ValueError("Cartesian action limits must be positive")
        if self.translation_action_scale <= 0.0 or self.rotation_action_scale <= 0.0:
            raise ValueError("action scales must be positive")
        if (
            not np.isfinite(self.joint_command_margin)
            or self.joint_command_margin < 0.0
        ):
            raise ValueError("joint_command_margin must be finite and non-negative")


@dataclass(frozen=True)
class RobotCommand:
    """The command that was actually written after safety limiting."""

    action_mode: str
    requested_arm: np.ndarray
    joint_target: np.ndarray
    joint_target_velocity: np.ndarray
    clipped: bool
    ik_result: IKResult | None = None
    # True only when a caller-supplied, already time-parameterized joint
    # sample passed the controller's independent discrete envelope checks and
    # was written without a second online target filter. A float32 rounding-
    # only projection can set both this marker and ``clipped``.
    time_parameterized: bool = False


def _require_mujoco() -> Any:
    if mujoco is None:
        raise ImportError(
            "MuJoCo is required for RobotController. Install the `mujoco` "
            "Python package first."
        )
    return mujoco


def _name2id(model: Any, object_type: Any, name: str) -> int:
    identifier = int(_require_mujoco().mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo object named {name!r} does not exist")
    return identifier


def _expanded_limit(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=float)
    result = result.reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or {size}-value vector")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must contain positive values")
    return result


def discrete_braking_displacement(
    velocity: np.ndarray, acceleration: np.ndarray, dt: float
) -> np.ndarray:
    """Signed travel while braking accepted velocity on controller ticks.

    The next accepted velocity is reduced before its position increment,
    matching ``rate_limited_target``'s semi-implicit Euler convention.  This
    describes the commanded trajectory, not the physical arm's stopping point.
    """

    speed = np.asarray(velocity, dtype=float)
    accel = np.broadcast_to(np.asarray(acceleration, dtype=float), speed.shape)
    if not (
        np.all(np.isfinite(speed)) and np.all(np.isfinite(accel))
        and np.all(accel > 0.0) and np.isfinite(dt) and dt > 0.0
    ):
        raise ValueError("braking prediction needs finite velocity, positive acceleration and dt")
    # Count strictly positive future velocities; the final tick reaches zero
    # and contributes no travel, including a fractional final decrement.
    steps = np.maximum(np.ceil(np.abs(speed) / (accel * dt)) - 1.0, 0.0)
    distance = dt * (
        steps * np.abs(speed) - .5 * accel * dt * steps * (steps + 1.0)
    )
    return np.sign(speed) * np.maximum(distance, 0.0)


def _discrete_stopping_speed(
    distance: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Maximum speed that can stop within ``distance`` on controller ticks.

    Commands are integrated with semi-implicit Euler steps,
    ``q[k + 1] = q[k] + v[k + 1] * dt``.  The familiar continuous bound
    ``sqrt(2 * a * distance)`` does not include the distance travelled on the
    current controller tick.  Close to a position limit it therefore brakes
    one tick too late and a subsequent position clip creates an acceleration
    spike.

    For a candidate speed ``v`` and ``n = ceil(v / (a * dt))``, the exact
    discrete stopping distance (including the current tick) is::

        dt * (n * v - a * dt * n * (n - 1) / 2)

    This function inverts that piecewise-linear expression.  It is vectorized
    because every joint has its own acceleration limit and remaining travel.
    """

    remaining = np.maximum(np.asarray(distance, dtype=float), 0.0)
    accel = np.asarray(acceleration, dtype=float)
    scaled_distance = remaining / (accel * dt * dt)
    intervals = np.maximum(
        1.0,
        np.ceil(0.5 * (np.sqrt(1.0 + 8.0 * scaled_distance) - 1.0)),
    )
    speed = (
        remaining / dt
        + 0.5 * accel * dt * intervals * (intervals - 1.0)
    ) / intervals
    return np.where(remaining > 0.0, speed, 0.0)


def rate_limited_target(requested, dt, last_target, last_velocity, lower, upper, max_velocity, max_acceleration):
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("controller dt must be finite and positive")
    limited_position = np.minimum(
        np.maximum(requested, lower),
        upper,
    )
    position_error = limited_position - last_target
    # Use the same discrete integration convention as the hard command
    # envelope.  The continuous sqrt(2*a*d) bound can cross a fixed target
    # by one controller tick, reverse on the next tick, and settle into a
    # permanent two-cycle.  That was especially harmful at certified-edge
    # handoffs, where the next stop-to-stop profile correctly waits for a
    # zero-velocity command history.  Recomputing this exact target bound
    # each tick remains suitable for moving IK endpoints while preventing
    # overshoot of a temporarily fixed endpoint.
    stopping_speed = _discrete_stopping_speed(
        np.abs(position_error),
        max_acceleration,
        dt,
    )
    desired_velocity = np.sign(position_error) * np.minimum(
        max_velocity,
        stopping_speed,
    )

    acceleration_step = max_acceleration * dt
    acceleration_lower = np.maximum(
        -max_velocity,
        last_velocity - acceleration_step,
    )
    acceleration_upper = np.minimum(
        max_velocity,
        last_velocity + acceleration_step,
    )

    # A command can satisfy the acceleration bound on the current tick and
    # still be impossible to stop before the next soft-limit clamp.  Limit
    # outward velocity by the exact multi-tick braking envelope at both
    # ends.  Once initialized inside this envelope, its intersection with
    # the acceleration interval remains non-empty on every subsequent
    # controller tick.
    lower_stopping_speed = _discrete_stopping_speed(
        last_target - lower,
        max_acceleration,
        dt,
    )
    upper_stopping_speed = _discrete_stopping_speed(
        upper - last_target,
        max_acceleration,
        dt,
    )
    admissible_lower = np.maximum(
        acceleration_lower,
        -lower_stopping_speed,
    )
    admissible_upper = np.minimum(
        acceleration_upper,
        upper_stopping_speed,
    )
    if np.any(admissible_lower > admissible_upper + 1e-12):
        # This requires externally replacing limiter history (or reducing
        # the acceleration limit instantaneously) with a state from which
        # position and acceleration constraints are mathematically
        # incompatible.  Do not silently emit a command that violates one
        # of the advertised safety limits.
        joints = np.flatnonzero(
            admissible_lower > admissible_upper + 1e-12
        ).tolist()
        raise RuntimeError(
            "joint command limiter state is outside the discrete braking "
            f"envelope for joints {joints}"
        )
    # Collapse harmless floating-point interval inversions at a boundary.
    interval_midpoint = 0.5 * (admissible_lower + admissible_upper)
    admissible_lower = np.minimum(admissible_lower, interval_midpoint)
    admissible_upper = np.maximum(admissible_upper, interval_midpoint)
    target_velocity = np.minimum(
        np.maximum(desired_velocity, admissible_lower),
        admissible_upper,
    )
    candidate = last_target + target_velocity * dt

    # Limit the command trajectory relative to the previous command, not
    # relative to the measured joint.  Capping position error to one
    # velocity step starves a position servo of torque whenever the plant
    # lags (for NERO joint6 it limited the error to 0.04 rad and the wrist
    # never reached its IK orientation).  The velocity/acceleration bounds
    # above already make this target continuous and safe.
    candidate = np.minimum(
        np.maximum(candidate, lower),
        upper,
    )
    actual_velocity = (candidate - last_target) / dt
    clipped = not np.allclose(candidate, requested, atol=1e-12, rtol=0.0)
    return candidate, actual_velocity, clipped


def timed_joint_step(requested, dt, last_target, last_velocity, lower, upper,
                     max_velocity, max_acceleration, actuator_lower, actuator_upper):
    """Pure timed acceptance/fallback, preserving the original discrete guard.

    Rejection bits describe the unprojected request: position, actuator,
    velocity, acceleration, stopping. ``accepted`` also covers a float32
    rounding-only projection into the unchanged envelope. Such a projection
    sets ``clipped`` but does not invoke the ordinary goal/braking filter.
    """
    if dt <= 0 or not np.isfinite(dt) or not np.isfinite(requested).all():
        raise ValueError("Require finite command and positive finite dt")
    limited = np.clip(requested, lower, upper)
    velocity = (limited - last_target) / dt
    acceleration = (velocity - last_velocity) / dt
    lower_speed = _discrete_stopping_speed(last_target - lower, max_acceleration, dt)
    upper_speed = _discrete_stopping_speed(upper - last_target, max_acceleration, dt)
    tolerance = 1e-9
    rejected = np.array([
        np.any((requested < lower) | (requested > upper)),
        np.any((requested < actuator_lower) | (requested > actuator_upper)),
        np.any(np.abs(velocity) > max_velocity + tolerance),
        np.any(np.abs(acceleration) > max_acceleration + tolerance),
        np.any((velocity < -lower_speed - tolerance) | (velocity > upper_speed + tolerance)),
    ], dtype=bool)
    accepted = not np.any(rejected)
    if accepted:
        target, clipped = requested.copy(), False
    else:
        # Acceleration within the float32 rounding allowance is a necessary
        # condition for the rounding-only branch below. Most model samples
        # fail it by far. Evaluate that same condition before building and
        # rechecking a projection which would inevitably be discarded.
        # This changes neither the allowance nor the fallback command.
        with np.errstate(over="ignore", invalid="ignore"):
            single = requested.astype(np.float32)
            widened = single.astype(np.float64)
            def half_ulp(values):
                encoded = np.asarray(values, dtype=np.float32)
                centre = encoded.astype(np.float64)
                below = np.nextafter(encoded, np.float32(-np.inf)).astype(np.float64)
                above = np.nextafter(encoded, np.float32(np.inf)).astype(np.float64)
                return .5 * np.maximum(centre - below, above - centre)
            current_error = half_ulp(requested)
            last_error = half_ulp(last_target)
            previous_error = half_ulp(last_target - dt * last_velocity)
        velocity_error = (current_error + last_error) / dt
        acceleration_error = (current_error + 2 * last_error + previous_error) / (dt * dt)
        if not np.all(np.abs(acceleration) <= max_acceleration + acceleration_error + tolerance):
            target, velocity, clipped = rate_limited_target(
                requested, dt, last_target, last_velocity, lower, upper, max_velocity, max_acceleration)
            return target, velocity, clipped, False, rejected
        # Learned commands and recorded actuator labels are float32, even
        # after an adapter widens them to float64. A half-ulp upward rounding
        # at a velocity boundary must not turn a timed sample into a new stop
        # goal: that would replace nanoradians of rounding with a full tick
        # of deceleration. Apply this only to exact float32-valued requests;
        # arbitrary float64 requests retain the original strict checks.
        # Difference checks also contain quantization of the preceding input
        # positions: their error budgets have coefficients (1,1) for velocity
        # and (1,2,1) for acceleration. These budgets only decide whether to
        # project; every output still passes the unchanged actual envelope.
        velocity_lower = np.maximum.reduce((
            -max_velocity, last_velocity - max_acceleration * dt, -lower_speed))
        velocity_upper = np.minimum.reduce((
            max_velocity, last_velocity + max_acceleration * dt, upper_speed))
        target_lower = np.maximum.reduce((
            lower, actuator_lower, last_target + dt * velocity_lower))
        target_upper = np.minimum.reduce((
            upper, actuator_upper, last_target + dt * velocity_upper))
        bad_axis = (
            (requested < lower) | (requested > upper)
            | (requested < actuator_lower) | (requested > actuator_upper)
            | (np.abs(velocity) > max_velocity + tolerance)
            | (np.abs(acceleration) > max_acceleration + tolerance)
            | (velocity < -lower_speed - tolerance)
            | (velocity > upper_speed + tolerance))
        projected = np.where(bad_axis,
            np.clip(requested, target_lower, target_upper), requested)
        # A previous-position error also perturbs the discrete stopping
        # speed, whose piecewise slope is at most 1/dt.
        stopping_error = (current_error + 2 * last_error) / dt
        changed = projected != requested
        rounding_only = bool(
            np.all(target_lower <= target_upper)
            and np.all(~changed | (np.isfinite(widened) & (widened == requested)))
            and np.all(requested >= lower - current_error)
            and np.all(requested <= upper + current_error)
            and np.all(requested >= actuator_lower - current_error)
            and np.all(requested <= actuator_upper + current_error)
            and np.all(np.abs(velocity) <= max_velocity + velocity_error + tolerance)
            and np.all(np.abs(acceleration) <= max_acceleration + acceleration_error + tolerance)
            and np.all(velocity >= -lower_speed - stopping_error - tolerance)
            and np.all(velocity <= upper_speed + stopping_error + tolerance)
            and np.all(np.abs(projected - requested)
                       <= current_error + 2 * last_error + previous_error))
        projected_velocity = (projected - last_target) / dt
        projected_acceleration = (projected_velocity - last_velocity) / dt
        # Recheck the result; the original numerical tolerance is unchanged.
        # In particular, projection never widens a position/actuator bound.
        projection_feasible = bool(
            np.all(projected >= lower) and np.all(projected <= upper)
            and np.all(projected >= actuator_lower) and np.all(projected <= actuator_upper)
            and np.all(np.abs(projected_velocity) <= max_velocity + tolerance)
            and np.all(np.abs(projected_acceleration) <= max_acceleration + tolerance)
            and np.all(projected_velocity >= -lower_speed - tolerance)
            and np.all(projected_velocity <= upper_speed + tolerance))
        if rounding_only and projection_feasible:
            target, velocity, clipped, accepted = projected, projected_velocity, bool(np.any(changed)), True
        else:
            target, velocity, clipped = rate_limited_target(
                requested, dt, last_target, last_velocity, lower, upper, max_velocity, max_acceleration)
    return target, velocity, clipped, bool(accepted), rejected


def _normalize_config_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept common robot-YAML aliases without coupling to a config package."""

    source: Mapping[str, Any] = config
    if isinstance(config.get("robot"), Mapping):
        robot_section = config["robot"]
        if isinstance(robot_section.get("robot"), Mapping):
            source = robot_section["robot"]  # type: ignore[assignment]
        else:
            source = robot_section  # type: ignore[assignment]
    result = dict(source)
    control = result.get("controller")
    if isinstance(control, Mapping):
        result.update(control)
    aliases = {
        "end_effector_site": "ee_site_name",
        "ee_site": "ee_site_name",
        "home_qpos": "wait_joint_positions",
        "home_joint_positions": "wait_joint_positions",
        "max_velocity": "max_joint_velocity",
        "velocity_limits": "max_joint_velocity",
        "max_acceleration": "max_joint_acceleration",
        "acceleration_limits": "max_joint_acceleration",
        "dt": "control_timestep",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result[new] = result[old]
    end_effector = result.get("end_effector")
    if isinstance(end_effector, Mapping) and "ee_site_name" not in result:
        if "site_name" in end_effector:
            result["ee_site_name"] = end_effector["site_name"]
    joint_limits = result.get("joint_limits")
    if isinstance(joint_limits, Mapping):
        if "lower" in joint_limits and "joint_lower_limits" not in result:
            result["joint_lower_limits"] = joint_limits["lower"]
        if "upper" in joint_limits and "joint_upper_limits" not in result:
            result["joint_upper_limits"] = joint_limits["upper"]
    return result


class RobotController:
    """Rate-limited joint and Cartesian-delta control for a MuJoCo arm."""

    SUPPORTED_ACTION_MODES = ("joint_position", "ee_delta_pose")

    def __init__(
        self,
        model: Any,
        data: Any,
        config: RobotControllerConfig | Mapping[str, Any] | None = None,
        *,
        joint_names: Iterable[str] | None = None,
        actuator_names: Iterable[str] | None = None,
        ee_site_name: str | None = None,
        gripper: GripperLike | None = None,
    ) -> None:
        mj = _require_mujoco()
        if isinstance(config, Mapping):
            config = _normalize_config_mapping(config)
        parsed = config_from_mapping(RobotControllerConfig, config)
        overrides: dict[str, Any] = {}
        if joint_names is not None:
            overrides["joint_names"] = tuple(joint_names)
        if actuator_names is not None:
            overrides["actuator_names"] = tuple(actuator_names)
        if ee_site_name is not None:
            overrides["ee_site_name"] = str(ee_site_name)
        if overrides:
            values = {
                field_name: getattr(parsed, field_name)
                for field_name in RobotControllerConfig.__dataclass_fields__
            }
            values.update(overrides)
            parsed = RobotControllerConfig(**values)

        self.model = model
        self.data = data
        self.config = parsed
        self.gripper = gripper
        self.joint_names = tuple(parsed.joint_names)
        if len(self.joint_names) != 7:
            raise ValueError(
                "RobotController requires exactly seven configured arm joints; "
                f"got {len(self.joint_names)}"
            )
        if len(set(self.joint_names)) != 7:
            raise ValueError("joint_names must be unique")

        self.joint_ids = np.asarray(
            [
                _name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ],
            dtype=int,
        )
        supported = (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
        if any(int(model.jnt_type[joint_id]) not in supported for joint_id in self.joint_ids):
            raise ValueError("all configured arm joints must be hinge or slide joints")
        self.qpos_indices = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=int)
        self.dof_indices = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=int)

        self.actuator_ids = self._resolve_actuators(parsed.actuator_names)
        self.actuator_names = tuple(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, int(identifier))
            or f"actuator_{identifier}"
            for identifier in self.actuator_ids
        )
        self.ee_site_id = _name2id(
            model, mj.mjtObj.mjOBJ_SITE, parsed.ee_site_name
        )
        self.joint_lower_limits, self.joint_upper_limits = self._joint_limits()
        (
            self.command_joint_lower_limits,
            self.command_joint_upper_limits,
        ) = self._command_joint_limits()
        self.max_joint_velocity = _expanded_limit(
            parsed.max_joint_velocity, 7, "max_joint_velocity"
        )
        self.max_joint_acceleration = _expanded_limit(
            parsed.max_joint_acceleration, 7, "max_joint_acceleration"
        )
        self.control_timestep = float(
            parsed.control_timestep
            if parsed.control_timestep is not None
            else model.opt.timestep
        )
        if self.control_timestep <= 0.0:
            raise ValueError("model.opt.timestep must be positive")

        self.ik_solver = DampedLeastSquaresIK(
            model,
            data,
            self.joint_names,
            parsed.ee_site_name,
            config=parsed.ik,
        )
        self.action_mode = parsed.action_mode
        self._last_joint_target = self.clip_joint_positions(
            self.get_joint_positions()
        )
        self._last_target_velocity = np.clip(
            self.get_joint_velocities(),
            -self.max_joint_velocity,
            self.max_joint_velocity,
        )
        self._last_command: RobotCommand | None = None
        self._write_joint_targets(self._last_joint_target)

    @property
    def dof(self) -> int:
        return 7

    @property
    def last_command(self) -> RobotCommand | None:
        return self._last_command

    @property
    def commanded_joint_positions(self) -> np.ndarray:
        """Return the actuator target used by the discrete limiter history."""

        return self._last_joint_target.copy()

    @property
    def commanded_joint_velocities(self) -> np.ndarray:
        """Return the velocity of the accepted actuator-target sequence."""

        return self._last_target_velocity.copy()

    def adopt_joint_dispatch_history(self, positions, velocities) -> None:
        """Import actual sends from an exclusive joint-stream owner.

        This changes command history, never measured qpos/qvel. The caller
        must obtain an atomic sender snapshot (and join it before resuming
        ordinary commands). Planned but unsent samples must not enter here.
        """
        q = as_vector(positions, 7, name="dispatched_joint_positions").copy()
        v = as_vector(velocities, 7, name="dispatched_joint_velocities").copy()
        self._write_joint_targets(q)
        self._last_joint_target = q
        self._last_target_velocity = v
        self._last_command = RobotCommand(
            action_mode="joint_position", requested_arm=q.copy(),
            joint_target=q.copy(), joint_target_velocity=v.copy(),
            clipped=False, time_parameterized=True,
        )

    def _resolve_actuators(self, configured_names: Iterable[str]) -> np.ndarray:
        mj = _require_mujoco()
        names = tuple(configured_names)
        if names:
            if len(names) != 7:
                raise ValueError("actuator_names must contain exactly seven names")
            return np.asarray(
                [
                    _name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                    for name in names
                ],
                dtype=int,
            )

        resolved: list[int] = []
        for joint_id, joint_name in zip(self.joint_ids, self.joint_names):
            candidates = np.flatnonzero(
                np.asarray(self.model.actuator_trnid[:, 0], dtype=int) == joint_id
            )
            if candidates.size != 1:
                raise ValueError(
                    f"expected one actuator for joint {joint_name!r}, found "
                    f"{candidates.size}; configure actuator_names explicitly"
                )
            resolved.append(int(candidates[0]))
        return np.asarray(resolved, dtype=int)

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(7, -np.inf, dtype=float)
        upper = np.full(7, np.inf, dtype=float)
        for index, joint_id in enumerate(self.joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                lower[index], upper[index] = self.model.jnt_range[joint_id]

        configured_lower = self.config.joint_lower_limits
        configured_upper = self.config.joint_upper_limits
        if bool(configured_lower) != bool(configured_upper):
            raise ValueError(
                "joint_lower_limits and joint_upper_limits must be provided together"
            )
        if configured_lower:
            requested_lower = as_vector(
                configured_lower, 7, name="joint_lower_limits"
            )
            requested_upper = as_vector(
                configured_upper, 7, name="joint_upper_limits"
            )
            if np.any(requested_lower >= requested_upper):
                raise ValueError("every lower joint limit must be below its upper limit")
            # Explicit safety limits may tighten but never widen model limits.
            lower = np.maximum(lower, requested_lower)
            upper = np.minimum(upper, requested_upper)
            if np.any(lower >= upper):
                raise ValueError("configured safety limits conflict with model limits")
        return lower, upper

    def _command_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the inward-buffered limits used only for actuator commands.

        ``joint_lower_limits`` and ``joint_upper_limits`` intentionally remain
        the physical/configured hard limits.  The baseline planner derives its
        own admissible region from those arrays, so replacing them with these
        buffered limits would apply the same margin twice.
        """

        margin = float(self.config.joint_command_margin)
        if margin == 0.0:
            return self.joint_lower_limits.copy(), self.joint_upper_limits.copy()

        lower = self.joint_lower_limits + margin
        upper = self.joint_upper_limits - margin
        # Rounding ``hard_limit + margin`` can land a fraction of an ulp closer
        # to the hard stop than the requested margin.  Bias the inclusive soft
        # boundary inward so telemetry never reports a sub-margin command.
        lower = np.nextafter(lower, np.inf)
        upper = np.nextafter(upper, -np.inf)
        if np.any(lower >= upper):
            raise ValueError(
                "joint_command_margin leaves no valid command range for at "
                "least one joint"
            )
        return lower, upper

    def _write_joint_targets(self, joint_target: np.ndarray) -> None:
        target = np.asarray(joint_target, dtype=float).copy()
        for index, actuator_id in enumerate(self.actuator_ids):
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
                target[index] = np.clip(target[index], ctrl_min, ctrl_max)
        self.data.ctrl[self.actuator_ids] = target

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.data.qpos[self.qpos_indices], dtype=float, copy=True)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array(self.data.qvel[self.dof_indices], dtype=float, copy=True)

    def get_joint_accelerations(self) -> np.ndarray:
        if hasattr(self.data, "qacc"):
            return np.array(self.data.qacc[self.dof_indices], dtype=float, copy=True)
        return np.zeros(7, dtype=float)

    def clip_joint_positions(self, positions: Any) -> np.ndarray:
        """Apply the buffered command-position limits without issuing a command."""

        values = as_vector(positions, 7, name="joint_positions")
        return np.minimum(
            np.maximum(values, self.command_joint_lower_limits),
            self.command_joint_upper_limits,
        )

    def clip_joint_velocities(self, velocities: Any) -> np.ndarray:
        values = as_vector(velocities, 7, name="joint_velocities")
        return np.clip(values, -self.max_joint_velocity, self.max_joint_velocity)

    def clip_joint_accelerations(self, accelerations: Any) -> np.ndarray:
        values = as_vector(accelerations, 7, name="joint_accelerations")
        return np.clip(
            values, -self.max_joint_acceleration, self.max_joint_acceleration
        )

    def get_ee_pose(self) -> np.ndarray:
        """Return ``[x, y, z, qw, qx, qy, qz]`` in world coordinates."""

        quat = np.empty(4, dtype=float)
        _require_mujoco().mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return np.r_[
            np.array(self.data.site_xpos[self.ee_site_id], dtype=float, copy=True),
            normalized_quat(quat),
        ]

    def get_state(self) -> dict[str, Any]:
        pose = self.get_ee_pose()
        state: dict[str, Any] = {
            "joint_pos": self.get_joint_positions(),
            "joint_vel": self.get_joint_velocities(),
            "joint_acc": self.get_joint_accelerations(),
            "ee_pose": pose,
            "ee_position": pose[:3].copy(),
            "ee_quat": pose[3:].copy(),
            "joint_target": self._last_joint_target.copy(),
            "joint_target_velocity": self._last_target_velocity.copy(),
            "action_mode": self.action_mode,
        }
        if self.gripper is not None and hasattr(self.gripper, "get_state"):
            state["gripper"] = self.gripper.get_state()
        return state

    # Observation builders in external environments use both spellings.
    read_state = get_state
    get_joint_state = get_state
    get_end_effector_pose = get_ee_pose

    def set_action_mode(self, action_mode: str) -> None:
        if action_mode not in self.SUPPORTED_ACTION_MODES:
            raise ValueError(
                f"unsupported action mode {action_mode!r}; expected one of "
                f"{self.SUPPORTED_ACTION_MODES}"
            )
        self.action_mode = action_mode

    def _rate_limited_target(self, requested, dt):
        return rate_limited_target(requested, dt, self._last_joint_target,
                                   self._last_target_velocity, self.command_joint_lower_limits,
                                   self.command_joint_upper_limits, self.max_joint_velocity,
                                   self.max_joint_acceleration)

    def set_joint_target(
        self, joint_positions: Any, *, dt: float | None = None
    ) -> RobotCommand:
        requested = as_vector(joint_positions, 7, name="joint_positions")
        step_dt = self.control_timestep if dt is None else float(dt)
        target, velocity, clipped = self._rate_limited_target(requested, step_dt)
        self._write_joint_targets(target)
        self._last_joint_target = target.copy()
        self._last_target_velocity = velocity.copy()
        command = RobotCommand(
            action_mode="joint_position",
            requested_arm=requested.copy(),
            joint_target=target.copy(),
            joint_target_velocity=velocity.copy(),
            clipped=clipped,
        )
        self._last_command = command
        return command

    def preview_joint_step(self, requested, last_target, last_velocity, dt):
        """Read-only envelope evaluation, shared by execution and queue preview."""
        actuator_lower = np.full(7, -np.inf)
        actuator_upper = np.full(7, np.inf)
        for i, actuator_id in enumerate(self.actuator_ids):
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                actuator_lower[i], actuator_upper[i] = self.model.actuator_ctrlrange[actuator_id]
        return timed_joint_step(
            np.asarray(requested, dtype=float), dt, np.asarray(last_target, dtype=float),
            np.asarray(last_velocity, dtype=float), self.command_joint_lower_limits,
            self.command_joint_upper_limits, self.max_joint_velocity,
            self.max_joint_acceleration, actuator_lower, actuator_upper)

    def preview_joint_trajectory(self, requested, *, dt=None):
        """Project future targets from current command history, without writing state.

        Returns targets, velocities, and per-step envelope rejection bits.
        Does not step physics or touch ctrl/qpos/history. Gripper is separate.
        """
        requested = np.asarray(requested, dtype=float)
        if requested.ndim != 2 or requested.shape[1] != 7 or not np.isfinite(requested).all():
            raise ValueError("Expected finite [horizon,7] joint targets")
        dt = self.control_timestep if dt is None else dt
        q, v = self._last_joint_target.copy(), self._last_target_velocity.copy()
        targets, velocities, reasons = [], [], []
        for sample in requested:
            q, v, _, _, rejected = self.preview_joint_step(sample, q, v, dt)
            targets.append(q.copy())
            velocities.append(v.copy())
            reasons.append(rejected)
        return (np.asarray(targets).reshape(-1, 7), np.asarray(velocities).reshape(-1, 7),
                np.asarray(reasons, dtype=bool).reshape(-1, 5))

    def set_time_parameterized_joint_target(
        self, joint_positions: Any, *, dt: float | None = None
    ) -> RobotCommand:
        """Accept one pre-timed sample only after discrete envelope checks.

        A stop-to-stop trajectory produced upstream already obeys the same
        joint velocity and acceleration limits.  Feeding each of its samples
        back through :meth:`_rate_limited_target` treats that sample as a new
        online goal and applies a second braking profile, making execution
        substantially slower than the deadline-certified ETA.  This method
        removes only that duplicate filter.  The command is accepted verbatim
        iff its buffered position, per-tick velocity, acceleration change and
        discrete stopping envelope are all valid. A request represented in
        float32 may instead be projected by the bounded quantization error of
        the position/velocity/acceleration differences onto that same envelope.
        Otherwise the ordinary limiter remains the
        fail-closed fallback; none of the physical limits are widened.
        """

        requested = as_vector(joint_positions, 7, name="joint_positions")
        step_dt = self.control_timestep if dt is None else float(dt)
        if step_dt <= 0.0 or not np.isfinite(step_dt):
            raise ValueError("controller dt must be finite and positive")

        target, velocity, clipped, accepted, _ = self.preview_joint_step(
            requested, self._last_joint_target, self._last_target_velocity, step_dt)

        self._write_joint_targets(target)
        self._last_joint_target = target.copy()
        self._last_target_velocity = velocity.copy()
        command = RobotCommand(
            action_mode="joint_position",
            requested_arm=requested.copy(),
            joint_target=target.copy(),
            joint_target_velocity=velocity.copy(),
            clipped=bool(clipped),
            time_parameterized=accepted,
        )
        self._last_command = command
        return command

    # Familiar aliases for scripts and gym-style environment wrappers.
    set_joint_positions = set_joint_target
    command_joint_positions = set_joint_target

    def apply_ee_delta(
        self, delta_pose: Any, *, dt: float | None = None
    ) -> RobotCommand:
        delta = as_vector(delta_pose, 6, name="ee_delta_pose")
        translation = delta[:3] * self.config.translation_action_scale
        rotation = delta[3:] * self.config.rotation_action_scale
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > self.config.max_cartesian_delta:
            translation *= self.config.max_cartesian_delta / translation_norm
        rotation_norm = float(np.linalg.norm(rotation))
        if rotation_norm > self.config.max_rotation_delta:
            rotation *= self.config.max_rotation_delta / rotation_norm

        current_pose = self.get_ee_pose()
        target_position = current_pose[:3] + translation
        target_quaternion = compose_world_delta(current_pose[3:], rotation)
        ik_result = self.ik_solver.solve(
            target_position,
            target_quaternion,
            qpos_seed=self.get_joint_positions(),
        )
        requested = ik_result.joint_positions
        if not ik_result.success and not self.config.allow_unconverged_ik:
            requested = self._last_joint_target.copy()
        step_dt = self.control_timestep if dt is None else float(dt)
        target, velocity, clipped = self._rate_limited_target(requested, step_dt)
        clipped = clipped or not np.allclose(
            delta, np.r_[translation, rotation], atol=1e-12, rtol=0.0
        )
        self._write_joint_targets(target)
        self._last_joint_target = target.copy()
        self._last_target_velocity = velocity.copy()
        command = RobotCommand(
            action_mode="ee_delta_pose",
            requested_arm=delta.copy(),
            joint_target=target.copy(),
            joint_target_velocity=velocity.copy(),
            clipped=clipped,
            ik_result=ik_result,
        )
        self._last_command = command
        return command

    command_ee_delta_pose = apply_ee_delta

    def apply_action(
        self,
        action: Mapping[str, Any] | Any,
        *,
        action_mode: str | None = None,
        dt: float | None = None,
    ) -> RobotCommand:
        """Apply ``{"arm": ..., "gripper": ...}`` or a bare arm vector."""

        mode = self.action_mode if action_mode is None else action_mode
        if mode not in self.SUPPORTED_ACTION_MODES:
            raise ValueError(f"unsupported action mode {mode!r}")
        if isinstance(action, Mapping):
            arm = action.get("arm")
            if arm is None:
                arm = (
                    self._last_joint_target
                    if mode == "joint_position"
                    else np.zeros(6, dtype=float)
                )
            gripper_command = action.get("gripper")
            time_parameterized = (
                action.get("arm_time_parameterized", False) is True
            )
        else:
            arm = action
            gripper_command = None
            time_parameterized = False

        if mode == "joint_position":
            result = (
                self.set_time_parameterized_joint_target(arm, dt=dt)
                if time_parameterized
                else self.set_joint_target(arm, dt=dt)
            )
        else:
            result = self.apply_ee_delta(arm, dt=dt)
        if gripper_command is not None:
            if self.gripper is None:
                raise RuntimeError(
                    "action contains a gripper command but no gripper controller "
                    "was attached"
                )
            self.gripper.command(float(gripper_command))
        return result

    command = apply_action
    step = apply_action

    def acknowledge_dispatch(self, command: RobotCommand, *, actual_dt: float,
                             planned_dt: float, previous_target: Any,
                             previous_velocity: Any) -> dict[str, Any]:
        """Record host send spacing after one successfully sent command.

        This updates derivative history only: no new target, actuator write,
        policy step, or physical-feedback estimate. A computation overrun can
        make actual command velocity lower than the planned velocity; carrying
        that fictitious faster velocity into the next limiter was incorrect.
        Unpredictable host stalls can still cause a derivative discontinuity,
        which is reported rather than hidden by clipping the measured spacing.
        """
        if command is not self._last_command:
            raise ValueError('dispatch acknowledgement must match the pending command')
        if not np.isfinite([actual_dt, planned_dt]).all() or min(actual_dt, planned_dt) <= 0.:
            raise ValueError('dispatch intervals must be finite and positive')
        before = as_vector(previous_target, 7, name='previous_target')
        before_velocity = as_vector(previous_velocity, 7, name='previous_velocity')
        actual_velocity = (command.joint_target-before)/actual_dt
        actual_acceleration = (actual_velocity-before_velocity)/actual_dt
        self._last_target_velocity = actual_velocity.copy()
        self._last_command = replace(command, joint_target_velocity=actual_velocity.copy())
        return {
            'clock': 'host_move_joints_call_start',
            'planned_interval_s': float(planned_dt), 'actual_interval_s': float(actual_dt),
            'planned_velocity_rad_s': command.joint_target_velocity.tolist(),
            'actual_velocity_rad_s': actual_velocity.tolist(),
            'actual_acceleration_rad_s2': actual_acceleration.tolist(),
            'actual_velocity_within_limits': bool(np.all(
                np.abs(actual_velocity) <= self.max_joint_velocity+1e-8)),
            'actual_acceleration_within_limits': bool(np.all(
                np.abs(actual_acceleration) <= self.max_joint_acceleration+1e-8)),
            'physical_feedback_velocity': False,
        }

    def hold(self, *, dt: float | None = None) -> RobotCommand:
        return self.set_joint_target(self.get_joint_positions(), dt=dt)

    def command_wait_pose(self, *, dt: float | None = None) -> RobotCommand:
        if not self.config.wait_joint_positions:
            raise RuntimeError("wait_joint_positions is not configured")
        return self.set_joint_target(self.config.wait_joint_positions, dt=dt)

    def reset(self, *, command_current_position: bool = True) -> None:
        """Reset limiter history without resetting/teleporting simulation state."""

        current = self.get_joint_positions()
        self._last_joint_target = current
        self._last_target_velocity = np.clip(
            self.get_joint_velocities(),
            -self.max_joint_velocity,
            self.max_joint_velocity,
        )
        self._last_command = None
        if command_current_position:
            self._write_joint_targets(current)

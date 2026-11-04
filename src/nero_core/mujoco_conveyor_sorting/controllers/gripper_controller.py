"""Parallel-jaw gripper actuator control and contact-based feedback.

No object pose or equality constraint is touched here.  ``is_holding`` is an
observation derived from two-finger contacts and jaw opening, not a grasp
shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np

from ._math import as_vector, config_from_mapping

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


class GripperStatus(str, Enum):
    OPEN = "open"
    OPENING = "opening"
    CLOSED = "closed"
    CLOSING = "closing"
    HOLDING = "holding"
    STOPPED = "stopped"


@dataclass(frozen=True)
class GripperConfig:
    """Names, calibration, and action convention for a parallel gripper."""

    actuator_names: tuple[str, ...] = field(default_factory=tuple)
    joint_names: tuple[str, ...] = field(default_factory=tuple)
    open_ctrl: tuple[float, ...] = field(default_factory=tuple)
    closed_ctrl: tuple[float, ...] = field(default_factory=tuple)
    open_joint_positions: tuple[float, ...] = field(default_factory=tuple)
    closed_joint_positions: tuple[float, ...] = field(default_factory=tuple)
    open_width: float = 0.08
    closed_width: float = 0.0
    action_open: float = 1.0
    action_closed: float = -1.0
    position_tolerance: float = 1e-3
    max_open_fraction_rate: float | None = None
    control_timestep: float | None = None
    left_finger_geoms: tuple[str, ...] = field(default_factory=tuple)
    right_finger_geoms: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.action_open == self.action_closed:
            raise ValueError("action_open and action_closed must differ")
        if self.open_width <= self.closed_width:
            raise ValueError("open_width must be larger than closed_width")
        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive")
        if (
            self.max_open_fraction_rate is not None
            and self.max_open_fraction_rate <= 0.0
        ):
            raise ValueError("max_open_fraction_rate must be positive")
        if self.control_timestep is not None and self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")


def _require_mujoco() -> Any:
    if mujoco is None:
        raise ImportError(
            "MuJoCo is required for GripperController. Install the `mujoco` "
            "Python package first."
        )
    return mujoco


def _name2id(model: Any, object_type: Any, name: str) -> int:
    identifier = int(_require_mujoco().mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo object named {name!r} does not exist")
    return identifier


def _normalize_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source: Mapping[str, Any] = config
    if isinstance(config.get("gripper"), Mapping):
        source = config["gripper"]  # type: ignore[assignment]
    elif isinstance(config.get("robot"), Mapping):
        robot_document = config["robot"]
        if isinstance(robot_document.get("gripper"), Mapping):
            source = robot_document["gripper"]  # type: ignore[assignment]
    result = dict(source)
    aliases = {
        "open_position": "open_ctrl",
        "closed_position": "closed_ctrl",
        "open_command": "open_ctrl",
        "closed_command": "closed_ctrl",
        "left_contact_geoms": "left_finger_geoms",
        "right_contact_geoms": "right_finger_geoms",
        "dt": "control_timestep",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            value = result[old]
            if old in {
                "open_position",
                "closed_position",
                "open_command",
                "closed_command",
            } and np.isscalar(value):
                value = (float(value),)
            result[new] = value
    if "actuator_names" not in result and "actuator_name" in result:
        result["actuator_names"] = (result["actuator_name"],)
    if "joint_names" not in result:
        finger_joints = [
            result.get("left_finger_joint"),
            result.get("right_finger_joint"),
        ]
        if all(finger_joints):
            result["joint_names"] = tuple(finger_joints)
    if "left_finger_geoms" not in result and "left_finger_geom" in result:
        result["left_finger_geoms"] = (result["left_finger_geom"],)
    if "right_finger_geoms" not in result and "right_finger_geom" in result:
        result["right_finger_geoms"] = (result["right_finger_geom"],)
    if "open_width" not in result and "maximum_inner_width" in result:
        result["open_width"] = result["maximum_inner_width"]
    if "closed_width" not in result and "minimum_inner_width" in result:
        result["closed_width"] = result["minimum_inner_width"]
    # The repository's tendon actuator controls the sum of two symmetric
    # finger coordinates.  Derive a useful feedback calibration when explicit
    # per-joint endpoints are absent.
    if (
        result.get("command_semantics") == "opening_width"
        and len(result.get("joint_names", ())) == 2
        and "open_joint_positions" not in result
        and "open_command" in result
        and "closed_command" in result
    ):
        result["open_joint_positions"] = (
            0.5 * float(result["open_command"]),
            0.5 * float(result["open_command"]),
        )
        result["closed_joint_positions"] = (
            0.5 * float(result["closed_command"]),
            0.5 * float(result["closed_command"]),
        )
    return result


class GripperController:
    """Normalized open/close control for one or more finger actuators.

    By default an action of ``+1`` means fully open and ``-1`` means fully
    closed.  The calibrated endpoints are configurable.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        config: GripperConfig | Mapping[str, Any] | None = None,
        *,
        actuator_names: Iterable[str] | None = None,
        joint_names: Iterable[str] | None = None,
    ) -> None:
        mj = _require_mujoco()
        if isinstance(config, Mapping):
            config = _normalize_config(config)
        parsed = config_from_mapping(GripperConfig, config)
        if actuator_names is not None or joint_names is not None:
            values = {
                name: getattr(parsed, name)
                for name in GripperConfig.__dataclass_fields__
            }
            if actuator_names is not None:
                values["actuator_names"] = tuple(actuator_names)
            if joint_names is not None:
                values["joint_names"] = tuple(joint_names)
            parsed = GripperConfig(**values)

        self.model = model
        self.data = data
        self.config = parsed
        if not parsed.actuator_names:
            raise ValueError("at least one gripper actuator name must be configured")
        self.actuator_names = tuple(parsed.actuator_names)
        self.actuator_ids = np.asarray(
            [
                _name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.actuator_names
            ],
            dtype=int,
        )
        self.joint_ids = self._resolve_joints(parsed.joint_names)
        self.joint_names = tuple(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(identifier))
            or f"joint_{identifier}"
            for identifier in self.joint_ids
        )
        self.qpos_indices = np.asarray(
            model.jnt_qposadr[self.joint_ids], dtype=int
        )
        self.open_ctrl, self.closed_ctrl = self._control_endpoints()
        (
            self.open_joint_positions,
            self.closed_joint_positions,
        ) = self._joint_endpoints()
        self.left_finger_geom_ids = self._resolve_geoms(
            parsed.left_finger_geoms
        )
        self.right_finger_geom_ids = self._resolve_geoms(
            parsed.right_finger_geoms
        )
        self._self_geom_ids = self._infer_self_geoms()
        self.control_timestep = float(
            parsed.control_timestep
            if parsed.control_timestep is not None
            else model.opt.timestep
        )
        self._target_open_fraction = self.open_fraction
        self._last_action = self._fraction_to_action(self._target_open_fraction)
        self._last_status = GripperStatus.STOPPED

    def _resolve_joints(self, configured_names: Iterable[str]) -> np.ndarray:
        mj = _require_mujoco()
        names = tuple(configured_names)
        if names:
            result = np.asarray(
                [
                    _name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
                    for name in names
                ],
                dtype=int,
            )
        else:
            result = np.asarray(
                self.model.actuator_trnid[self.actuator_ids, 0], dtype=int
            )
            # Multiple actuators can legitimately drive the same tendon/joint.
            result = np.asarray(list(dict.fromkeys(result.tolist())), dtype=int)
        if np.any(result < 0):
            raise ValueError(
                "gripper actuator transmission does not identify a joint; "
                "configure joint_names explicitly"
            )
        # MuJoCo 3.14 enums and NumPy integer tuple-membership can disagree
        # despite scalar equality. Compare plain integers on both sides.
        supported = (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
        if any(int(self.model.jnt_type[joint_id]) not in supported for joint_id in result):
            raise ValueError("gripper feedback joints must be hinge or slide joints")
        return result

    def _control_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.actuator_ids)
        if bool(self.config.open_ctrl) != bool(self.config.closed_ctrl):
            raise ValueError("open_ctrl and closed_ctrl must be provided together")
        if self.config.open_ctrl:
            open_ctrl = as_vector(self.config.open_ctrl, count, name="open_ctrl")
            closed_ctrl = as_vector(
                self.config.closed_ctrl, count, name="closed_ctrl"
            )
        else:
            if not np.all(self.model.actuator_ctrllimited[self.actuator_ids]):
                raise ValueError(
                    "open_ctrl/closed_ctrl are required for actuators without "
                    "MuJoCo control ranges"
                )
            ranges = np.asarray(
                self.model.actuator_ctrlrange[self.actuator_ids], dtype=float
            )
            closed_ctrl = ranges[:, 0].copy()
            open_ctrl = ranges[:, 1].copy()
        if np.allclose(open_ctrl, closed_ctrl):
            raise ValueError("open and closed actuator controls must differ")
        return open_ctrl, closed_ctrl

    def _joint_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.joint_ids)
        if bool(self.config.open_joint_positions) != bool(
            self.config.closed_joint_positions
        ):
            raise ValueError(
                "open_joint_positions and closed_joint_positions must be "
                "provided together"
            )
        if self.config.open_joint_positions:
            opened = as_vector(
                self.config.open_joint_positions,
                count,
                name="open_joint_positions",
            )
            closed = as_vector(
                self.config.closed_joint_positions,
                count,
                name="closed_joint_positions",
            )
            return opened, closed

        opened = np.zeros(count, dtype=float)
        closed = np.zeros(count, dtype=float)
        for index, joint_id in enumerate(self.joint_ids):
            if not bool(self.model.jnt_limited[joint_id]):
                raise ValueError(
                    "joint endpoint calibration is required for an unlimited "
                    f"gripper joint ({self.joint_names[index]!r})"
                )
            low, high = self.model.jnt_range[joint_id]
            closed[index], opened[index] = low, high
        return opened, closed

    def _resolve_geoms(self, names: Iterable[str]) -> frozenset[int]:
        mj = _require_mujoco()
        return frozenset(
            _name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            for name in names
        )

    def _infer_self_geoms(self) -> frozenset[int]:
        """Geoms on the jaws and their ancestor bodies are not workpieces."""

        finger_geoms = self.left_finger_geom_ids | self.right_finger_geom_ids
        bodies = {
            int(self.model.geom_bodyid[geom_id]) for geom_id in finger_geoms
        }
        frontier = list(bodies)
        while frontier:
            body_id = frontier.pop()
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id != body_id and parent_id not in bodies:
                bodies.add(parent_id)
                frontier.append(parent_id)
        return frozenset(
            geom_id
            for geom_id in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[geom_id]) in bodies
        )

    def _fraction_to_action(self, fraction: float) -> float:
        return float(
            self.config.action_closed
            + fraction * (self.config.action_open - self.config.action_closed)
        )

    def _action_to_fraction(self, action: float) -> float:
        return float(
            np.clip(
                (action - self.config.action_closed)
                / (self.config.action_open - self.config.action_closed),
                0.0,
                1.0,
            )
        )

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array(self.data.qpos[self.qpos_indices], dtype=float, copy=True)

    @property
    def open_fraction(self) -> float:
        current = self.joint_positions
        direction = self.open_joint_positions - self.closed_joint_positions
        denominator = float(direction @ direction)
        if denominator < 1e-16:
            return float(self._target_open_fraction) if hasattr(
                self, "_target_open_fraction"
            ) else 0.0
        fraction = float(
            (current - self.closed_joint_positions) @ direction / denominator
        )
        return float(np.clip(fraction, 0.0, 1.0))

    @property
    def opening_width(self) -> float:
        return float(
            self.config.closed_width
            + self.open_fraction
            * (self.config.open_width - self.config.closed_width)
        )

    def get_opening_width(self) -> float:
        return self.opening_width

    def get_open_fraction(self) -> float:
        return self.open_fraction

    @property
    def target_open_fraction(self) -> float:
        return self._target_open_fraction

    def set_open_fraction(
        self, fraction: float, *, dt: float | None = None
    ) -> np.ndarray:
        requested = float(np.clip(fraction, 0.0, 1.0))
        previous = self._target_open_fraction
        if self.config.max_open_fraction_rate is not None:
            step_dt = self.control_timestep if dt is None else float(dt)
            if step_dt <= 0.0:
                raise ValueError("gripper dt must be positive")
            max_change = self.config.max_open_fraction_rate * step_dt
            requested = float(np.clip(requested, previous - max_change, previous + max_change))

        control = self.closed_ctrl + requested * (self.open_ctrl - self.closed_ctrl)
        for index, actuator_id in enumerate(self.actuator_ids):
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                low, high = self.model.actuator_ctrlrange[actuator_id]
                control[index] = np.clip(control[index], low, high)
        self.data.ctrl[self.actuator_ids] = control
        self._target_open_fraction = requested
        self._last_action = self._fraction_to_action(requested)
        if requested > previous + 1e-9:
            self._last_status = GripperStatus.OPENING
        elif requested < previous - 1e-9:
            self._last_status = GripperStatus.CLOSING
        else:
            self._last_status = GripperStatus.STOPPED
        return control.copy()

    def command(self, action: float, *, dt: float | None = None) -> np.ndarray:
        """Apply a calibrated normalized command (default: -1 close, +1 open)."""

        if not np.isfinite(action):
            raise ValueError("gripper action must be finite")
        return self.set_open_fraction(self._action_to_fraction(float(action)), dt=dt)

    step = command

    def open(self, *, dt: float | None = None) -> np.ndarray:
        return self.set_open_fraction(1.0, dt=dt)

    def close(self, *, dt: float | None = None) -> np.ndarray:
        return self.set_open_fraction(0.0, dt=dt)

    def hold(self) -> np.ndarray:
        return self.set_open_fraction(self._target_open_fraction)

    def _contacting_other_geoms(
        self, finger_geoms: frozenset[int], object_geoms: set[int] | None
    ) -> set[int]:
        contacted: set[int] = set()
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in finger_geoms:
                other = geom2
            elif geom2 in finger_geoms:
                other = geom1
            else:
                continue
            if other in self._self_geom_ids:
                continue
            if object_geoms is None or other in object_geoms:
                contacted.add(other)
        return contacted

    def both_fingers_in_contact(
        self, object_geom_ids: Iterable[int] | None = None
    ) -> bool:
        """Whether both jaw groups touch the same requested/any external geom."""

        if not self.left_finger_geom_ids or not self.right_finger_geom_ids:
            return False
        requested = (
            None
            if object_geom_ids is None
            else {int(identifier) for identifier in object_geom_ids}
        )
        left = self._contacting_other_geoms(self.left_finger_geom_ids, requested)
        right = self._contacting_other_geoms(self.right_finger_geom_ids, requested)
        if requested is None:
            # Same-object confirmation avoids calling a table contact a grasp.
            return bool(left & right)
        return bool(left) and bool(right)

    # Alias reads naturally from baseline policies.
    has_bilateral_contact = both_fingers_in_contact

    def external_finger_contacts(self) -> tuple[bool, bool]:
        """Return target-agnostic left/right fingertip contact flags.

        A physical gripper can expose these flags through pad switches, force
        thresholds, or motor-current observers.  The MuJoCo implementation is
        deliberately equivalent: it only asks whether either configured
        finger geom touches a non-robot geom.  It does not identify the
        workpiece and therefore cannot leak simulator object identity or pose
        into the policy.
        """

        left = bool(
            self._contacting_other_geoms(self.left_finger_geom_ids, None)
        )
        right = bool(
            self._contacting_other_geoms(self.right_finger_geom_ids, None)
        )
        return left, right

    def external_contact_loads(self) -> dict[str, Any]:
        """Return pad loads without looking up an object's identity or pose.

        Forces are resolved in MuJoCo's contact frame: component zero is
        compression and components one/two span the tangent plane. Summing
        positive compression and tangent magnitudes per jaw mirrors separate
        finger force sensors; it deliberately does not identify the workpiece.
        """

        normal = np.zeros(2, dtype=float)
        tangential = np.zeros(2, dtype=float)
        counts = np.zeros(2, dtype=int)
        wrench = np.empty(6, dtype=float)
        groups = (self.left_finger_geom_ids, self.right_finger_geom_ids)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            for jaw, finger_geoms in enumerate(groups):
                other = second if first in finger_geoms else first if second in finger_geoms else None
                if other is None or other in self._self_geom_ids:
                    continue
                _require_mujoco().mj_contactForce(self.model, self.data, contact_index, wrench)
                normal[jaw] += max(0.0, float(wrench[0]))
                tangential[jaw] += float(np.linalg.norm(wrench[1:3]))
                counts[jaw] += 1
        return {
            "left_normal_force": float(normal[0]),
            "right_normal_force": float(normal[1]),
            "left_tangential_force": float(tangential[0]),
            "right_tangential_force": float(tangential[1]),
            "minimum_normal_force": float(np.min(normal)),
            "left_contact_count": int(counts[0]),
            "right_contact_count": int(counts[1]),
        }

    @property
    def is_open(self) -> bool:
        return self.open_fraction >= 1.0 - self.config.position_tolerance

    @property
    def is_closed(self) -> bool:
        return self.open_fraction <= self.config.position_tolerance

    @property
    def is_holding(self) -> bool:
        return self._holding_from_feedback(
            self.both_fingers_in_contact(), self.external_contact_loads()
        )

    def _holding_from_feedback(
        self, bilateral: bool, loads: Mapping[str, Any]
    ) -> bool:
        """Recognize a loaded jaw obstruction at any adaptive grip width.

        An inward position error supplies preload when an object prevents the
        jaws reaching their requested width. Requiring a fully-closed target
        incorrectly rejected every nonzero width command. Geometry contacts
        alone are insufficient: MuJoCo also reports unloaded margin contacts.
        This instantaneous feedback does not certify a slip-free future lift.
        """

        measured = self.open_fraction
        tolerance = self.config.position_tolerance
        return bool(
            measured > tolerance
            and self._target_open_fraction < measured - tolerance
            and bilateral
            and float(loads["minimum_normal_force"]) > 0.0
        )

    @property
    def status(self) -> GripperStatus:
        return self._status_from_feedback(self.is_holding)

    def _status_from_feedback(self, holding: bool) -> GripperStatus:
        if holding:
            return GripperStatus.HOLDING
        if self.is_open and self._target_open_fraction > 0.5:
            return GripperStatus.OPEN
        if self.is_closed and self._target_open_fraction < 0.5:
            return GripperStatus.CLOSED
        return self._last_status

    def get_state(
        self, object_geom_ids: Iterable[int] | None = None
    ) -> dict[str, Any]:
        left_contact, right_contact = self.external_finger_contacts()
        bilateral = self.both_fingers_in_contact(object_geom_ids)
        loads = self.external_contact_loads()
        holding = self._holding_from_feedback(bilateral, loads)
        status = self._status_from_feedback(holding)
        return {
            **loads,
            "opening": self.opening_width,
            "opening_width": self.opening_width,
            "open_fraction": self.open_fraction,
            "target_open_fraction": self._target_open_fraction,
            "command": self._last_action,
            "status": status.value,
            "is_open": self.is_open,
            "is_closed": self.is_closed,
            "left_finger_contact": left_contact,
            "right_finger_contact": right_contact,
            "any_external_finger_contact": left_contact or right_contact,
            "both_fingers_contact": bilateral,
            "is_holding": holding,
            "joint_pos": self.joint_positions,
        }

    read_state = get_state

    def reset(self, *, command_open: bool = False) -> None:
        """Reset command history; optionally request open without changing qpos."""

        self._target_open_fraction = self.open_fraction
        self._last_action = self._fraction_to_action(self._target_open_fraction)
        self._last_status = GripperStatus.STOPPED
        if command_open:
            self.open()

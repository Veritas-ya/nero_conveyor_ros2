"""Logical workpiece lifecycle management with a reusable MuJoCo body pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .object_spec import ObjectSpec, default_object_specs


class ObjectState(str, Enum):
    NEW = "NEW"
    TRACKING = "TRACKING"
    READY_TO_PLAN = "READY_TO_PLAN"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    GRASPED = "GRASPED"
    SORTED = "SORTED"
    SKIPPED = "SKIPPED"
    MISSED = "MISSED"
    FAILED = "FAILED"

    @property
    def terminal(self) -> bool:
        return self in {
            ObjectState.SORTED,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }

    @classmethod
    def parse(cls, value: "ObjectState | str") -> "ObjectState":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(f"unknown object state: {value!r}") from exc


ALLOWED_TRANSITIONS: Mapping[ObjectState, frozenset[ObjectState]] = {
    ObjectState.NEW: frozenset(
        {
            ObjectState.TRACKING,
            ObjectState.READY_TO_PLAN,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.TRACKING: frozenset(
        {
            ObjectState.READY_TO_PLAN,
            ObjectState.GRASPED,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.READY_TO_PLAN: frozenset(
        {
            ObjectState.TRACKING,
            ObjectState.PLANNING,
            ObjectState.GRASPED,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.PLANNING: frozenset(
        {
            ObjectState.TRACKING,
            ObjectState.EXECUTING,
            ObjectState.GRASPED,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.EXECUTING: frozenset(
        {
            ObjectState.TRACKING,
            ObjectState.GRASPED,
            ObjectState.SKIPPED,
            ObjectState.MISSED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.GRASPED: frozenset(
        {
            ObjectState.TRACKING,  # released/dropped back onto the belt
            ObjectState.SORTED,
            ObjectState.FAILED,
        }
    ),
    ObjectState.SORTED: frozenset(),
    ObjectState.SKIPPED: frozenset(),
    ObjectState.MISSED: frozenset(),
    ObjectState.FAILED: frozenset(),
}


class ObjectManagerError(RuntimeError):
    pass


class NoAvailableObjectSlot(ObjectManagerError):
    pass


class InvalidObjectTransition(ObjectManagerError):
    pass


@dataclass(frozen=True, slots=True)
class StateTransition:
    timestamp: float
    previous: ObjectState
    current: ObjectState
    reason: str | None = None


@dataclass
class ObjectInstance:
    """One logical workpiece occupying a reusable physical body slot."""

    object_id: str
    spec: ObjectSpec
    body_name: str
    joint_name: str
    state: ObjectState
    spawned_at: float
    updated_at: float
    state_changed_at: float
    position: npt.NDArray[np.float64]
    quaternion: npt.NDArray[np.float64]
    linear_velocity: npt.NDArray[np.float64]
    angular_velocity: npt.NDArray[np.float64]
    target_bin_id: int
    transitions: list[StateTransition] = field(default_factory=list)
    grasped_by: str | None = None
    sorted_bin_id: int | None = None
    failure_reason: str | None = None
    recycled_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category_id(self) -> int:
        return self.spec.category_id

    @property
    def template_object_id(self) -> str:
        """Static ID from ObjectSpec; distinct from this logical instance ID."""

        return self.spec.object_id

    @property
    def active(self) -> bool:
        return not self.state.terminal and self.recycled_at is None

    @property
    def terminal(self) -> bool:
        return self.state.terminal

    @property
    def status(self) -> ObjectState:
        """Alias used by controller/observation integrations."""

        return self.state

    @property
    def age(self) -> float:
        return self.updated_at - self.spawned_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "template_object_id": self.template_object_id,
            "category_id": self.category_id,
            "body_name": self.body_name,
            "joint_name": self.joint_name,
            "state": self.state.value,
            "status": self.state.value,
            "position": self.position.copy(),
            "quaternion": self.quaternion.copy(),
            "linear_velocity": self.linear_velocity.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "spawned_at": self.spawned_at,
            "updated_at": self.updated_at,
            "state_changed_at": self.state_changed_at,
            "target_bin_id": self.target_bin_id,
            "sorted_bin_id": self.sorted_bin_id,
            "grasped_by": self.grasped_by,
            "failure_reason": self.failure_reason,
            "recycled_at": self.recycled_at,
            **self.metadata,
        }


@dataclass(slots=True)
class _BodySlot:
    spec: ObjectSpec
    logical_object_id: str | None = None

    @property
    def available(self) -> bool:
        return self.logical_object_id is None


class ObjectManager:
    """Manage state transitions and map logical objects to MuJoCo body slots.

    ``model``/``data`` are optional.  Without them the manager is a pure Python
    state machine suitable for unit tests.  With them, body reads and free-joint
    writes use MuJoCo's named-accessor interface (with array fallbacks).

    Direct pose writes occur only in :meth:`spawn` and :meth:`recycle`, never in
    per-step :meth:`update_from_sim`.
    """

    def __init__(
        self,
        specs: Iterable[ObjectSpec] | None = None,
        *,
        model: Any | None = None,
        data: Any | None = None,
        belt_direction: Sequence[float] = (1.0, 0.0, 0.0),
        belt_end_coordinate: float | None = None,
        inactive_position: Sequence[float] = (0.0, 0.0, -2.0),
        logical_id_prefix: str = "obj",
        auto_forward: bool = True,
        forward_callback: Callable[[Any, Any], None] | None = None,
    ) -> None:
        parsed_specs = list(default_object_specs() if specs is None else specs)
        if not parsed_specs:
            raise ValueError("ObjectManager requires at least one ObjectSpec")
        category_ids = [spec.category_id for spec in parsed_specs]
        if len(set(category_ids)) != len(category_ids):
            raise ValueError("ObjectManager requires unique category body slots")
        bodies = [spec.body_name for spec in parsed_specs]
        if len(set(bodies)) != len(bodies):
            raise ValueError("ObjectManager requires unique body_name values")
        if (model is None) != (data is None):
            raise ValueError("model and data must be provided together")
        direction = _vector(belt_direction, 3, "belt_direction")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            raise ValueError("belt_direction cannot be zero")
        if belt_end_coordinate is not None and not np.isfinite(belt_end_coordinate):
            raise ValueError("belt_end_coordinate must be finite or None")
        self.specs = tuple(sorted(parsed_specs, key=lambda item: item.category_id))
        self.spec_by_category = {spec.category_id: spec for spec in self.specs}
        self.model = model
        self.data = data
        self.belt_direction = direction / direction_norm
        self.belt_end_coordinate = belt_end_coordinate
        self.inactive_position = _vector(
            inactive_position, 3, "inactive_position"
        )
        self.logical_id_prefix = str(logical_id_prefix)
        self.auto_forward = bool(auto_forward)
        self.forward_callback = forward_callback
        self._slots = {
            spec.body_name: _BodySlot(spec=spec) for spec in self.specs
        }
        self._instances: dict[str, ObjectInstance] = {}
        self._next_id = 0
        self._lock = RLock()
        self._parking_qpos: dict[str, npt.NDArray[np.float64]] = {}
        if self.model is not None and self.data is not None:
            for spec in self.specs:
                try:
                    self._parking_qpos[spec.freejoint_name] = (
                        self._read_joint_qpos(spec.freejoint_name)
                    )
                except (AttributeError, KeyError):
                    # A precise error will be raised if that slot is activated.
                    pass

    def __len__(self) -> int:
        return len(self.active_objects())

    @property
    def objects(self) -> Mapping[str, ObjectInstance]:
        """All logical instances in the current episode, including terminal."""

        with self._lock:
            return dict(self._instances)

    @property
    def active_by_body(self) -> Mapping[str, str]:
        with self._lock:
            return {
                body_name: slot.logical_object_id
                for body_name, slot in self._slots.items()
                if slot.logical_object_id is not None
            }

    def available_specs(
        self, category_ids: Iterable[int] | None = None
    ) -> tuple[ObjectSpec, ...]:
        allowed = None if category_ids is None else {int(item) for item in category_ids}
        with self._lock:
            return tuple(
                slot.spec
                for slot in self._slots.values()
                if slot.available
                and (allowed is None or slot.spec.category_id in allowed)
            )

    def spawn(
        self,
        category_id: int,
        *,
        timestamp: float,
        position: Sequence[float],
        quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        logical_object_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        physical_overrides: Mapping[str, Any] | None = None,
    ) -> ObjectInstance:
        """Activate a category's body slot as a new logical object."""

        now = _finite(timestamp, "timestamp")
        pos = _vector(position, 3, "position")
        quat = _quaternion(quaternion)
        linear = _vector(linear_velocity, 3, "linear_velocity")
        angular = _vector(angular_velocity, 3, "angular_velocity")
        try:
            spec = self.spec_by_category[int(category_id)]
        except KeyError as exc:
            raise KeyError(f"unknown category_id: {category_id}") from exc
        with self._lock:
            slot = self._slots[spec.body_name]
            if not slot.available:
                raise NoAvailableObjectSlot(
                    f"body {spec.body_name!r} is occupied by "
                    f"{slot.logical_object_id!r}"
                )
            if logical_object_id is None:
                while True:
                    candidate = f"{self.logical_id_prefix}_{self._next_id:06d}"
                    self._next_id += 1
                    if candidate not in self._instances:
                        logical_object_id = candidate
                        break
            logical_object_id = str(logical_object_id)
            if not logical_object_id:
                raise ValueError("logical_object_id cannot be empty")
            if logical_object_id in self._instances:
                raise ValueError(
                    f"logical_object_id already exists: {logical_object_id}"
                )
            try:
                self._activate_physical_slot(
                    spec,
                    pos,
                    quat,
                    linear,
                    angular,
                    physical_overrides or {},
                )
            except Exception as exc:
                raise ObjectManagerError(
                    f"failed to activate MuJoCo body {spec.body_name!r}"
                ) from exc
            instance = ObjectInstance(
                object_id=logical_object_id,
                spec=spec,
                body_name=spec.body_name,
                joint_name=spec.freejoint_name,
                state=ObjectState.NEW,
                spawned_at=now,
                updated_at=now,
                state_changed_at=now,
                position=pos,
                quaternion=quat,
                linear_velocity=linear,
                angular_velocity=angular,
                target_bin_id=spec.bin_id,
                metadata=dict(metadata or {}),
            )
            slot.logical_object_id = logical_object_id
            self._instances[logical_object_id] = instance
            return instance

    create = spawn
    spawn_object = spawn

    def get(self, object_id: str) -> ObjectInstance:
        try:
            return self._instances[str(object_id)]
        except KeyError as exc:
            raise KeyError(f"unknown logical object_id: {object_id!r}") from exc

    def get_by_body(self, body_name: str) -> ObjectInstance | None:
        with self._lock:
            slot = self._slots.get(str(body_name))
            if slot is None or slot.logical_object_id is None:
                return None
            return self._instances[slot.logical_object_id]

    def active_objects(
        self, states: Iterable[ObjectState | str] | None = None
    ) -> list[ObjectInstance]:
        allowed = (
            None
            if states is None
            else {ObjectState.parse(state) for state in states}
        )
        with self._lock:
            return [
                instance
                for instance in self._instances.values()
                if instance.recycled_at is None
                and not instance.state.terminal
                and (allowed is None or instance.state in allowed)
            ]

    get_active_objects = active_objects

    def terminal_objects(self) -> list[ObjectInstance]:
        with self._lock:
            return [
                instance
                for instance in self._instances.values()
                if instance.state.terminal
            ]

    def transition(
        self,
        object_id: str,
        new_state: ObjectState | str,
        *,
        timestamp: float,
        reason: str | None = None,
        force: bool = False,
    ) -> ObjectInstance:
        state = ObjectState.parse(new_state)
        now = _finite(timestamp, "timestamp")
        with self._lock:
            instance = self.get(object_id)
            previous = instance.state
            if state == previous:
                instance.updated_at = max(instance.updated_at, now)
                return instance
            if not force and state not in ALLOWED_TRANSITIONS[previous]:
                raise InvalidObjectTransition(
                    f"{object_id}: {previous.value} -> {state.value} is not allowed"
                )
            if now + 1e-12 < instance.state_changed_at:
                raise InvalidObjectTransition(
                    f"{object_id}: transition timestamp moved backwards"
                )
            instance.state = state
            instance.state_changed_at = now
            instance.updated_at = max(instance.updated_at, now)
            instance.transitions.append(StateTransition(now, previous, state, reason))
            if state is ObjectState.GRASPED:
                instance.grasped_by = (
                    str(reason) if reason and reason.startswith("gripper") else "gripper"
                )
            if state is ObjectState.TRACKING and previous is ObjectState.GRASPED:
                instance.grasped_by = None
            if state in {
                ObjectState.FAILED,
                ObjectState.MISSED,
                ObjectState.SKIPPED,
            }:
                instance.failure_reason = reason or state.value.lower()
            return instance

    set_state = transition

    def mark_tracking(self, object_id: str, timestamp: float) -> ObjectInstance:
        return self.transition(object_id, ObjectState.TRACKING, timestamp=timestamp)

    def mark_ready(self, object_id: str, timestamp: float) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.READY_TO_PLAN, timestamp=timestamp
        )

    def mark_planning(self, object_id: str, timestamp: float) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.PLANNING, timestamp=timestamp
        )

    def mark_executing(self, object_id: str, timestamp: float) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.EXECUTING, timestamp=timestamp
        )

    def mark_grasped(
        self,
        object_id: str,
        timestamp: float,
        *,
        gripper_id: str = "gripper",
        enable_assist: bool = False,
        update_assist_relpose: bool = True,
    ) -> ObjectInstance:
        instance = self.transition(
            object_id,
            ObjectState.GRASPED,
            timestamp=timestamp,
            reason=gripper_id,
        )
        instance.grasped_by = gripper_id
        if enable_assist and instance.spec.grasp_assist_weld_name:
            if update_assist_relpose:
                self._update_weld_relative_pose(
                    instance.spec.grasp_assist_weld_name
                )
            self._set_equality_active(instance.spec.grasp_assist_weld_name, True)
            self._forward_simulation()
        return instance

    def mark_sorted(
        self,
        object_id: str,
        timestamp: float,
        bin_id: int,
        *,
        expected_bin_id: int | None = None,
    ) -> ObjectInstance:
        instance = self.get(object_id)
        actual_bin = int(bin_id)
        expected_bin = (
            instance.target_bin_id
            if expected_bin_id is None
            else int(expected_bin_id)
        )
        target_state = (
            ObjectState.SORTED
            if actual_bin == expected_bin
            else ObjectState.FAILED
        )
        instance = self.transition(
            object_id,
            target_state,
            timestamp=timestamp,
            reason=None
            if target_state is ObjectState.SORTED
            else f"wrong_bin:{actual_bin},expected:{expected_bin}",
        )
        instance.sorted_bin_id = actual_bin
        instance.grasped_by = None
        if instance.spec.grasp_assist_weld_name:
            self._set_equality_active(instance.spec.grasp_assist_weld_name, False)
            self._forward_simulation()
        return instance

    mark_placed = mark_sorted

    def mark_missed(
        self, object_id: str, timestamp: float, reason: str = "belt_end"
    ) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.MISSED, timestamp=timestamp, reason=reason
        )

    def mark_skipped(
        self, object_id: str, timestamp: float, reason: str = "no_feasible_window"
    ) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.SKIPPED, timestamp=timestamp, reason=reason
        )

    def mark_failed(
        self, object_id: str, timestamp: float, reason: str
    ) -> ObjectInstance:
        return self.transition(
            object_id, ObjectState.FAILED, timestamp=timestamp, reason=reason
        )

    def release_to_belt(
        self,
        object_id: str,
        timestamp: float,
        *,
        dropped: bool = False,
    ) -> ObjectInstance:
        """Release grasp assist and either resume tracking or record a drop."""

        instance = self.get(object_id)
        if instance.spec.grasp_assist_weld_name:
            self._set_equality_active(instance.spec.grasp_assist_weld_name, False)
            self._forward_simulation()
        instance.grasped_by = None
        return self.transition(
            object_id,
            ObjectState.FAILED if dropped else ObjectState.TRACKING,
            timestamp=timestamp,
            reason="dropped" if dropped else "released_to_belt",
        )

    def update_from_sim(
        self,
        timestamp: float,
        *,
        auto_track_new: bool = True,
        mark_missed: bool = True,
        belt_end_coordinate: float | None = None,
    ) -> list[ObjectInstance]:
        """Refresh active records from MuJoCo and mark objects past belt end.

        No belt velocity or pose is imposed here.  Conveyor contact/force logic
        remains the environment's responsibility.
        """

        now = _finite(timestamp, "timestamp")
        end = (
            self.belt_end_coordinate
            if belt_end_coordinate is None
            else _finite(belt_end_coordinate, "belt_end_coordinate")
        )
        updated: list[ObjectInstance] = []
        with self._lock:
            active = self.active_objects()
            for instance in active:
                if now + 1e-12 < instance.updated_at:
                    raise ObjectManagerError(
                        f"{instance.object_id}: simulator timestamp moved backwards "
                        f"({now} < {instance.updated_at})"
                    )
                if self.model is not None:
                    try:
                        pos, quat, linear, angular = self._read_body_state(
                            instance.body_name, instance.joint_name
                        )
                    except Exception as exc:
                        raise ObjectManagerError(
                            f"failed reading body {instance.body_name!r}"
                        ) from exc
                    instance.position = pos
                    instance.quaternion = quat
                    instance.linear_velocity = linear
                    instance.angular_velocity = angular
                instance.updated_at = now
                if auto_track_new and instance.state is ObjectState.NEW:
                    self.transition(
                        instance.object_id,
                        ObjectState.TRACKING,
                        timestamp=now,
                    )
                if (
                    mark_missed
                    and end is not None
                    and instance.state is not ObjectState.GRASPED
                    and float(np.dot(instance.position, self.belt_direction)) > end
                ):
                    self.mark_missed(instance.object_id, now, "belt_end")
                updated.append(instance)
        return updated

    update = update_from_sim
    update_objects = update_from_sim

    def recycle(
        self,
        object_id: str,
        *,
        timestamp: float,
        park: bool = True,
        force: bool = False,
    ) -> ObjectInstance:
        """Release a body slot for reuse while retaining logical episode history."""

        now = _finite(timestamp, "timestamp")
        with self._lock:
            instance = self.get(object_id)
            if instance.recycled_at is not None:
                return instance
            if not instance.state.terminal and not force:
                raise ObjectManagerError(
                    f"cannot recycle non-terminal object {object_id!r}; "
                    "mark it sorted/skipped/missed/failed or pass force=True"
                )
            if not instance.state.terminal:
                self.transition(
                    object_id,
                    ObjectState.FAILED,
                    timestamp=now,
                    reason="forced_recycle",
                    force=True,
                )
            try:
                self._deactivate_physical_slot(instance.spec, park=park)
            except Exception as exc:
                raise ObjectManagerError(
                    f"failed to recycle MuJoCo body {instance.body_name!r}"
                ) from exc
            self._slots[instance.body_name].logical_object_id = None
            instance.recycled_at = now
            instance.updated_at = max(instance.updated_at, now)
            return instance

    recycle_object = recycle

    def recycle_terminal(
        self, timestamp: float, *, minimum_terminal_age: float = 0.0
    ) -> list[ObjectInstance]:
        now = _finite(timestamp, "timestamp")
        age = _nonnegative(minimum_terminal_age, "minimum_terminal_age")
        with self._lock:
            candidates = [
                instance.object_id
                for instance in self._instances.values()
                if instance.state.terminal
                and instance.recycled_at is None
                and now - instance.state_changed_at >= age
            ]
        return [
            self.recycle(object_id, timestamp=now) for object_id in candidates
        ]

    def reset(self, *, timestamp: float = 0.0, park: bool = True) -> None:
        """Recycle all bodies and clear logical IDs for a new episode."""

        now = _finite(timestamp, "timestamp")
        with self._lock:
            for instance in list(self._instances.values()):
                if instance.recycled_at is None:
                    try:
                        self._deactivate_physical_slot(instance.spec, park=park)
                    except Exception as exc:
                        raise ObjectManagerError(
                            f"failed to reset MuJoCo body {instance.body_name!r}"
                        ) from exc
                    instance.recycled_at = now
            self._instances.clear()
            self._next_id = 0
            for slot in self._slots.values():
                slot.logical_object_id = None

    reset_all = reset

    def _activate_physical_slot(
        self,
        spec: ObjectSpec,
        position: npt.NDArray[np.float64],
        quaternion: npt.NDArray[np.float64],
        linear_velocity: npt.NDArray[np.float64],
        angular_velocity: npt.NDArray[np.float64],
        physical_overrides: Mapping[str, Any],
    ) -> None:
        if self.model is None or self.data is None:
            return
        if spec.parking_weld_name:
            self._set_equality_active(spec.parking_weld_name, False)
        if spec.grasp_assist_weld_name:
            self._set_equality_active(spec.grasp_assist_weld_name, False)
        self._write_free_joint(
            spec.freejoint_name,
            position,
            quaternion,
            linear_velocity,
            angular_velocity,
        )
        self._apply_physical_overrides(spec, physical_overrides)
        self._forward_simulation()

    def _deactivate_physical_slot(self, spec: ObjectSpec, *, park: bool) -> None:
        if self.model is None or self.data is None:
            return
        if spec.grasp_assist_weld_name:
            self._set_equality_active(spec.grasp_assist_weld_name, False)
        if park:
            parking_qpos = self._parking_qpos.get(spec.freejoint_name)
            parking_position = (
                parking_qpos[:3]
                if parking_qpos is not None
                else self.inactive_position
            )
            parking_quaternion = (
                parking_qpos[3:7]
                if parking_qpos is not None
                else np.asarray([1.0, 0.0, 0.0, 0.0])
            )
            self._write_free_joint(
                spec.freejoint_name,
                parking_position,
                parking_quaternion,
                np.zeros(3),
                np.zeros(3),
            )
        if spec.parking_weld_name:
            self._set_equality_active(spec.parking_weld_name, True)
        self._apply_physical_overrides(
            spec,
            {"mass": spec.mass, "friction": spec.friction, "rgba": spec.rgba},
        )
        self._forward_simulation()

    def _read_body_state(
        self, body_name: str, joint_name: str | None = None
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        assert self.model is not None and self.data is not None
        try:
            body = self.data.body(body_name)
            position = np.asarray(body.xpos, dtype=np.float64).copy()
            quaternion = np.asarray(body.xquat, dtype=np.float64).copy()
            spatial = np.asarray(body.cvel, dtype=np.float64)
        except (AttributeError, KeyError):
            body_id = _name_to_id(self.model, "body", body_name)
            position = np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()
            quaternion = np.asarray(
                self.data.xquat[body_id], dtype=np.float64
            ).copy()
            spatial = np.asarray(self.data.cvel[body_id], dtype=np.float64)
        # A root free-joint's translational qvel is the most direct world-frame
        # belt velocity.  Fall back to cvel for generic bodies.
        if joint_name is not None:
            try:
                joint_velocity = np.asarray(
                    self.data.joint(joint_name).qvel, dtype=np.float64
                )
            except (AttributeError, KeyError):
                joint_id = _name_to_id(self.model, "joint", joint_name)
                dof_address = int(self.model.jnt_dofadr[joint_id])
                joint_velocity = np.asarray(
                    self.data.qvel[dof_address : dof_address + 6],
                    dtype=np.float64,
                )
            linear = joint_velocity[:3].copy()
            angular = joint_velocity[3:6].copy()
        else:
            # MuJoCo cvel ordering is [angular, linear].
            angular = spatial[:3].copy()
            linear = spatial[3:6].copy()
        return position, quaternion, linear, angular

    def _read_joint_qpos(
        self, joint_name: str
    ) -> npt.NDArray[np.float64]:
        assert self.model is not None and self.data is not None
        try:
            return np.asarray(
                self.data.joint(joint_name).qpos, dtype=np.float64
            ).copy()
        except (AttributeError, KeyError):
            joint_id = _name_to_id(self.model, "joint", joint_name)
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            return np.asarray(
                self.data.qpos[qpos_address : qpos_address + 7],
                dtype=np.float64,
            ).copy()

    def _write_free_joint(
        self,
        joint_name: str,
        position: npt.NDArray[np.float64],
        quaternion: npt.NDArray[np.float64],
        linear_velocity: npt.NDArray[np.float64],
        angular_velocity: npt.NDArray[np.float64],
    ) -> None:
        assert self.model is not None and self.data is not None
        qpos = np.concatenate((position, quaternion))
        qvel = np.concatenate((linear_velocity, angular_velocity))
        try:
            joint = self.data.joint(joint_name)
            joint.qpos[:] = qpos
            joint.qvel[:] = qvel
            return
        except (AttributeError, KeyError):
            joint_id = _name_to_id(self.model, "joint", joint_name)
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_address : qpos_address + 7] = qpos
            self.data.qvel[dof_address : dof_address + 6] = qvel

    def _set_equality_active(self, equality_name: str, active: bool) -> None:
        if self.model is None or self.data is None:
            return
        equality_id = _name_to_id(self.model, "equality", equality_name)
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[equality_id] = int(active)
        elif hasattr(self.model, "eq_active0"):
            # Older MuJoCo exposes only model.eq_active0.  This is safe at
            # spawn/reset boundaries but environments should call mj_forward.
            self.model.eq_active0[equality_id] = int(active)
        else:
            raise AttributeError("MuJoCo model/data has no equality-active array")

    def _update_weld_relative_pose(self, equality_name: str) -> None:
        """Set a weld's relpose to the bodies' current relative transform."""

        if self.model is None or self.data is None:
            return
        equality_id = _name_to_id(self.model, "equality", equality_name)
        if not all(
            hasattr(self.model, name)
            for name in ("eq_obj1id", "eq_obj2id", "eq_data")
        ):
            raise AttributeError("MuJoCo model lacks weld relative-pose arrays")
        first_body_id = int(self.model.eq_obj1id[equality_id])
        second_body_id = int(self.model.eq_obj2id[equality_id])
        first_position = np.asarray(
            self.data.xpos[first_body_id], dtype=np.float64
        )
        first_quaternion = np.asarray(
            self.data.xquat[first_body_id], dtype=np.float64
        )
        second_position = np.asarray(
            self.data.xpos[second_body_id], dtype=np.float64
        )
        second_quaternion = np.asarray(
            self.data.xquat[second_body_id], dtype=np.float64
        )
        inverse_first = first_quaternion.copy()
        inverse_first[1:] *= -1.0
        relative_position = _rotate_vector(
            inverse_first, second_position - first_position
        )
        relative_quaternion = _quaternion_multiply(
            inverse_first, second_quaternion
        )
        relative_quaternion /= np.linalg.norm(relative_quaternion)
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quaternion

    def _apply_physical_overrides(
        self, spec: ObjectSpec, values: Mapping[str, Any]
    ) -> None:
        if self.model is None or not values:
            return
        if "mass" in values:
            mass = float(values["mass"])
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError("randomized mass must be finite and positive")
            body_id = _name_to_id(self.model, "body", spec.body_name)
            previous_mass = float(self.model.body_mass[body_id])
            if previous_mass > 0.0 and hasattr(self.model, "body_inertia"):
                self.model.body_inertia[body_id] *= mass / previous_mass
            self.model.body_mass[body_id] = mass
        friction_value = values.get("friction")
        rgba_value = values.get("rgba", values.get("color"))
        for geom_name in spec.xml_geom_names:
            geom_id = _name_to_id(self.model, "geom", geom_name)
            if friction_value is not None:
                friction = np.asarray(friction_value, dtype=np.float64)
                if friction.ndim == 0:
                    friction = np.asarray([friction, 0.005, 0.0001])
                if friction.shape != (3,) or np.any(friction < 0.0):
                    raise ValueError("randomized friction must contain 3 values")
                self.model.geom_friction[geom_id] = friction
            if rgba_value is not None:
                rgba = np.asarray(rgba_value, dtype=np.float64)
                if rgba.shape != (4,):
                    raise ValueError("randomized rgba must contain 4 values")
                self.model.geom_rgba[geom_id] = np.clip(rgba, 0.0, 1.0)

    def _forward_simulation(self) -> None:
        """Synchronize derived poses after the allowed spawn/recycle writes."""

        if (
            not self.auto_forward
            or self.model is None
            or self.data is None
        ):
            return
        if self.forward_callback is not None:
            self.forward_callback(self.model, self.data)
            return
        try:
            import mujoco
        except ImportError:
            return
        # Keep pure-Python/mock model tests decoupled from MuJoCo.
        if isinstance(self.model, mujoco.MjModel) and isinstance(
            self.data, mujoco.MjData
        ):
            mujoco.mj_forward(self.model, self.data)


def _name_to_id(model: Any, kind: str, name: str) -> int:
    """Resolve a MuJoCo named item without forcing MuJoCo at import time."""

    accessor_names = {
        "body": ("body",),
        "joint": ("joint", "jnt"),
        "geom": ("geom",),
        "equality": ("equality", "eq"),
    }
    for accessor_name in accessor_names[kind]:
        accessor = getattr(model, accessor_name, None)
        if callable(accessor):
            try:
                result = accessor(name)
                return int(result.id)
            except (KeyError, AttributeError, TypeError):
                pass
    try:
        import mujoco

        object_type = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
        }[kind]
        item_id = int(mujoco.mj_name2id(model, object_type, name))
    except ImportError as exc:
        raise KeyError(
            f"cannot resolve {kind} {name!r}: model has no named accessor "
            "and mujoco is not installed"
        ) from exc
    if item_id < 0:
        raise KeyError(f"MuJoCo {kind} not found: {name!r}")
    return item_id


def _vector(value: Any, length: int, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result.copy()


def _quaternion(value: Sequence[float]) -> npt.NDArray[np.float64]:
    quaternion = _vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion cannot be zero")
    return quaternion / norm


def _quaternion_multiply(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _rotate_vector(
    quaternion: npt.NDArray[np.float64],
    vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    pure = np.asarray([0.0, *vector], dtype=np.float64)
    conjugate = quaternion.copy()
    conjugate[1:] *= -1.0
    return _quaternion_multiply(
        _quaternion_multiply(quaternion, pure), conjugate
    )[1:]


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

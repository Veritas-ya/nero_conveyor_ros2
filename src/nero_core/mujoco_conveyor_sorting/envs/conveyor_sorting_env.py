"""Continuous narrow-conveyor sorting environment.

Only spawning, recycling, and a full reset write an object's free-joint pose.
During normal simulation the conveyor is represented by a contact-gated
Cartesian force which tracks the configured belt velocity.  Grasped objects are
excluded from that force and remain governed by finger contact and gravity.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np

from ..config import PACKAGE_ROOT, deep_merge, load_project_config, load_yaml
from ..contact_metrics import ContactEpisodes, held_payload_rail_key
from ..metrics import SortingMetrics
from ..controllers.dynamic_intercept import intercept_lead


TERMINAL_OBJECT_STATES = {"SORTED", "SKIPPED", "MISSED", "FAILED"}
BELT_OBJECT_STATES = {
    "NEW",
    "TRACKING",
    "READY_TO_PLAN",
    "PLANNING",
    "EXECUTING",
    "CLOSING",
}


def _section(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Accept either a plain YAML mapping or a mapping wrapped by its filename."""

    value = document.get(name, document)
    return dict(value) if isinstance(value, Mapping) else {}


def _value(mapping: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        current: Any = mapping
        found = True
        for component in path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                found = False
                break
            current = current[component]
        if found:
            return current
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
    return quaternion


@dataclass
class RuntimeObject:
    slot: int
    body_name: str
    joint_name: str
    body_id: int
    joint_id: int
    geom_ids: tuple[int, ...]
    qpos_adr: int
    qvel_adr: int
    category_id: int
    bin_id: int
    logical_id: int | None = None
    state: str = "INACTIVE"
    spawned_at: float = 0.0
    state_changed_at: float = 0.0
    selected_at: float | None = None
    grasped_at: float | None = None
    released_at: float | None = None
    lost_contact_since: float | None = None
    stable_contact_steps: int = 0
    contact_gap_steps: int = 0
    prediction_at_grasp: np.ndarray | None = None
    actual_grasp_pose: np.ndarray | None = None
    planned_grasp_pose: np.ndarray | None = None
    planned_grasp_time: float | None = None
    compensation_enabled_at_selection: bool | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.logical_id is not None and self.state not in TERMINAL_OBJECT_STATES


@dataclass
class VisionTaskState:
    """Controller state keyed by an independently generated visual track ID.

    Simulator lifecycle state is kept in :class:`RuntimeObject` for physics
    and scoring.  Keeping policy planning state here prevents a visual track
    from inheriting a MuJoCo logical ID, category, or grasp target.
    """

    track_id: int
    category_id: int
    state: str = "TRACKING"
    selected_at: float | None = None
    planned_grasp_pose: np.ndarray | None = None
    planned_grasp_time: float | None = None
    compensation_enabled_at_selection: bool | None = None
    failure_reason: str | None = None
    category_observed_timestamp: float | None = None


class ConveyorSortingEnv:
    """MuJoCo environment exposing VLA-friendly observations and two action modes.

    Parameters are loaded from the five YAML files under ``configs``.  ``step``
    returns the Gymnasium-style five-tuple while ``reset`` returns the observation
    directly to keep the dependency surface small.
    """

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        render_mode: str | None = None,
        action_type: str | None = None,
        enable_motion_compensation: bool | None = None,
        grasp_assist: bool | None = None,
        seed: int | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        render_observations: bool = True,
        perception_mode: str | None = None,
        grasp_event_source: str = "policy_events",
    ) -> None:
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.config = load_project_config(config_dir, config_overrides)
        self.scene_config = _section(self.config.get("scene", {}), "scene")
        self.robot_document = dict(self.config.get("robot", {}))
        self.robot_config = _section(self.robot_document, "robot")
        self.gripper_config = _section(self.robot_document, "gripper")
        config_root = Path(str(self.config["config_dir"]))
        object_catalog = Path(
            str(
                _value(
                    self.scene_config,
                    "object_catalog_path",
                    default="objects.yaml",
                )
            )
        )
        if not object_catalog.is_absolute():
            object_catalog = (config_root / object_catalog).resolve()
        if not object_catalog.is_file():
            raise FileNotFoundError(
                f"Object catalog was not found: {object_catalog}"
            )
        self.object_config_path = object_catalog
        self.objects_document = load_yaml(object_catalog)
        object_overrides = (
            config_overrides.get("objects")
            if isinstance(config_overrides, Mapping)
            else None
        )
        if isinstance(object_overrides, Mapping):
            self.objects_document = deep_merge(
                self.objects_document, object_overrides
            )
        from ..tasks.task_config import effective_objects
        self.asset_objects_document = self.objects_document
        self.objects_document = effective_objects(self.scene_config, self.asset_objects_document)
        self.active_category_ids = tuple(s["category_id"] for s in self.objects_document["object_specs"])
        self.config["objects"] = self.objects_document
        self.objects_config = _section(self.objects_document, "objects")
        self.cameras_document = dict(self.config.get("cameras", {}))
        self.cameras_config = _section(self.cameras_document, "cameras")
        self.rendering_config = _section(self.cameras_document, "rendering")
        self.perception_config = _section(self.cameras_document, "perception")
        self.latency_config = _section(self.config.get("latency", {}), "latency")
        self.render_mode = render_mode
        self.render_observations = bool(render_observations)
        if grasp_event_source not in {"policy_events", "physical_contacts"}:
            raise ValueError("grasp_event_source must be 'policy_events' or 'physical_contacts'")
        # Learned policies emit joint/gripper commands without expert stage
        # callbacks. This opt-in path starts lifecycle tracking from measured
        # contacts; it never selects a target or generates control commands.
        self.grasp_event_source = grasp_event_source
        configured_perception_mode = str(
            _value(self.perception_config, "mode", default="rendered_rgbd")
        )
        # ``render_observations=False`` is an explicit request for the cheap
        # simulator-history regression path.  The normal viewer and RGB-array
        # entry points use the rendered RGB-D estimator and fail closed if
        # offscreen rendering is unavailable.
        self.perception_mode = (
            "simulator_history"
            if perception_mode is None and not self.render_observations
            else str(perception_mode or configured_perception_mode)
        )
        if self.perception_mode not in {"rendered_rgbd", "simulator_history"}:
            raise ValueError(
                "perception_mode must be 'rendered_rgbd' or 'simulator_history'"
            )
        if self.perception_mode == "rendered_rgbd" and not self.render_observations:
            raise ValueError(
                "rendered_rgbd perception requires render_observations=True"
            )

        configured_model = _value(
            self.scene_config,
            "model_path",
            "mjcf_path",
            "model.xml_path",
            default=str(PACKAGE_ROOT / "models" / "scene.xml"),
        )
        candidate = Path(model_path) if model_path else Path(str(configured_model))
        if not candidate.is_absolute():
            config_root = Path(str(self.config["config_dir"]))
            config_relative = (config_root / candidate).resolve()
            package_relative = (PACKAGE_ROOT / candidate).resolve()
            candidate = (
                config_relative if config_relative.exists() else package_relative
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"MuJoCo scene XML was not found: {candidate}")
        self.model_path = candidate
        self.model = mujoco.MjModel.from_xml_path(str(candidate))
        self.data = mujoco.MjData(self.model)

        self.control_dt = _as_float(
            _value(self.scene_config, "simulation.control_dt", "control_dt", default=0.02),
            0.02,
        )
        if self.control_dt < self.model.opt.timestep:
            raise ValueError("control_dt cannot be smaller than MuJoCo's timestep")
        self.frame_skip = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.control_dt = self.frame_skip * float(self.model.opt.timestep)
        self.action_type = action_type or str(
            _value(
                self.robot_config,
                "action_mode",
                "control.action_mode",
                default="joint_position",
            )
        )
        if self.action_type not in {"joint_position", "ee_delta_pose"}:
            raise ValueError("action_type must be 'joint_position' or 'ee_delta_pose'")
        configured_compensation = bool(
            _value(
                self.latency_config,
                "enable_motion_compensation",
                "motion_compensation.enabled",
                default=True,
            )
        )
        self.enable_motion_compensation = (
            configured_compensation
            if enable_motion_compensation is None
            else bool(enable_motion_compensation)
        )
        configured_assist = bool(
            _value(
                self.gripper_config,
                "grasp_assist.enabled_by_default",
                "grasp_assist",
                default=_value(
                    self.scene_config,
                    "object_pool.grasp_assist_enabled",
                    default=False,
                ),
            )
        )
        self.grasp_assist = configured_assist if grasp_assist is None else bool(grasp_assist)
        self.debug_visualization = bool(
            _value(self.scene_config, "debug.enabled", default=True)
        )

        configured_seed = _value(
            self.scene_config, "randomization.seed", default=None
        )
        effective_seed = (
            int(configured_seed)
            if seed is None and configured_seed is not None
            else seed
        )
        self._rng = np.random.default_rng(effective_seed)
        self._initial_seed = effective_seed
        self._episode_seed = effective_seed
        self._unseeded_reset_sequence = 0
        self.metrics = SortingMetrics()
        self._viewer: Any | None = None
        self._renderers: dict[tuple[int, ...], Any] = {}
        self._renderer_failed = False
        self._camera_frames: deque[dict[str, Any]] = deque(maxlen=48)
        # RGB and wrist products are display streams, not inputs to the
        # depth-geometry controller.  They are captured explicitly/on demand
        # and carry timestamps independent of the 15 Hz perception stream.
        self._display_camera_frames: deque[dict[str, Any]] = deque(maxlen=8)
        self._state_history: deque[tuple[float, dict[int, dict[str, Any]]]] = deque(
            maxlen=max(128, int(math.ceil(4.0 / self.model.opt.timestep)))
        )
        # Independent conveyor-encoder history. Only speed telemetry and the
        # simulation clock enter it; no object pose, velocity or contact does.
        self._belt_motion_history: deque[tuple[float, float]] = deque(
            maxlen=max(128, int(math.ceil(4.0 / self.model.opt.timestep)) + 1)
        )
        self._belt_encoder_distance = 0.0
        self._ee_trajectory: deque[np.ndarray] = deque(
            maxlen=int(
                _value(
                    self.scene_config,
                    "debug.ee_trajectory.max_points",
                    default=120,
                )
            )
        )
        self._action_queue: deque[tuple[float, dict[str, Any]]] = deque()
        # Number of dequeue batches in which more than one level command had
        # become due before physics could consume one.  Such a batch must not
        # be replayed through the controller: its intermediate set-points were
        # never applied to physics and therefore cannot form valid velocity or
        # acceleration history for a later command.
        self._action_queue_overruns = 0
        # Passive-viewer callbacks may run concurrently with the control loop.
        # They enqueue mutations only; the owner/control thread applies them
        # at a tick boundary before computing or consuming another action.
        self._viewer_command_queue: deque[str] = deque(maxlen=64)
        self._last_camera_capture = -np.inf
        self._last_display_camera_capture = -np.inf
        self._last_observation: dict[str, Any] | None = None
        self._video_writer: Any | None = None
        self._video_path: Path | None = None
        self.video_recording_requested = False
        self._closed = False
        self._planning_workers: list[Any] = []
        self._scene_epoch = 0

        self._read_model_names()
        self._read_runtime_parameters()
        self._build_research_modules(effective_seed)
        self._build_object_slots()
        self._payload_evaluation_geometry = {
            int(geom): runtime for runtime in self._objects for geom in runtime.geom_ids
        }
        self._payload_evaluation_rail_geoms = frozenset(
            geom for geom in range(self.model.ngeom)
            if any(token in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").lower()
                   for token in ("rail", "barrier"))
        )
        self._held_payload_rail_contacts = ContactEpisodes(gap_seconds=0.02)
        self._payload_contact_evaluation_events: list[dict[str, Any]] = []
        self._build_controllers()
        self.reset(seed=effective_seed)

    # ------------------------------------------------------------------
    # Model/configuration setup
    # ------------------------------------------------------------------
    def _name2id(self, object_type: mujoco.mjtObj, name: str, *, required: bool = True) -> int:
        identifier = int(mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0 and required:
            raise ValueError(f"Scene is missing required {object_type.name}: {name!r}")
        return identifier

    def _read_model_names(self) -> None:
        default_joints = [f"joint_{index}" for index in range(1, 8)]
        self.arm_joint_names = tuple(
            _value(
                self.robot_config,
                "joint_names",
                "arm.joint_names",
                default=default_joints,
            )
        )
        if len(self.arm_joint_names) != 7:
            raise ValueError(
                f"Exactly seven arm joints are required, got {len(self.arm_joint_names)}"
            )
        default_actuators = [f"{name}_actuator" for name in self.arm_joint_names]
        self.arm_actuator_names = tuple(
            _value(
                self.robot_config,
                "actuator_names",
                "arm.actuator_names",
                default=default_actuators,
            )
        )
        self.ee_site_name = str(
            _value(
                self.robot_config,
                "ee_site_name",
                "ee_site",
                "end_effector_site",
                "end_effector.site_name",
                default="ee_site",
            )
        )
        self.gripper_actuator_names = tuple(
            _value(
                self.gripper_config,
                "actuator_names",
                default=[
                    _value(
                        self.gripper_config,
                        "actuator_name",
                        default="gripper_actuator",
                    )
                ],
            )
        )
        self.gripper_joint_names = tuple(
            _value(
                self.gripper_config,
                "joint_names",
                default=[
                    _value(
                        self.gripper_config,
                        "left_finger_joint",
                        default="finger_left_joint",
                    ),
                    _value(
                        self.gripper_config,
                        "right_finger_joint",
                        default="finger_right_joint",
                    ),
                ],
            )
        )
        self.finger_geom_names = tuple(
            _value(
                self.gripper_config,
                "finger_geom_names",
                "contact_geoms",
                default=[
                    _value(
                        self.gripper_config,
                        "left_finger_geom",
                        default="left_finger_geom",
                    ),
                    _value(
                        self.gripper_config,
                        "right_finger_geom",
                        default="right_finger_geom",
                    ),
                ],
            )
        )
        self.belt_geom_name = str(
            _value(
                self.scene_config,
                "conveyor.surface_geom",
                "conveyor.geom_name",
                default="conveyor_surface",
            )
        )
        self.top_camera_name = str(
            _value(
                self.cameras_config,
                "top.name",
                "top_camera.name",
                default="top_camera",
            )
        )
        self.wrist_camera_name = str(
            _value(
                self.cameras_config,
                "wrist.name",
                "wrist_camera.name",
                default="wrist_camera",
            )
        )
        self.policy_external_camera_name = str(
            _value(
                self.cameras_config,
                "policy_external.name",
                default="policy_external_camera",
            )
        )

        self.arm_joint_ids = np.asarray(
            [
                self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.arm_joint_names
            ],
            dtype=int,
        )
        self.arm_qpos_adrs = np.asarray(
            self.model.jnt_qposadr[self.arm_joint_ids], dtype=int
        )
        self.arm_dof_adrs = np.asarray(
            self.model.jnt_dofadr[self.arm_joint_ids], dtype=int
        )
        self.arm_actuator_ids = np.asarray(
            [
                self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.arm_actuator_names
            ],
            dtype=int,
        )
        self.gripper_actuator_ids = np.asarray(
            [
                self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.gripper_actuator_names
            ],
            dtype=int,
        )
        self.gripper_joint_ids = np.asarray(
            [
                self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.gripper_joint_names
            ],
            dtype=int,
        )
        self.gripper_qpos_adrs = np.asarray(
            self.model.jnt_qposadr[self.gripper_joint_ids], dtype=int
        )
        self.ee_site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self.belt_geom_id = self._name2id(
            mujoco.mjtObj.mjOBJ_GEOM, self.belt_geom_name
        )
        self.finger_geom_ids = tuple(
            self._name2id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.finger_geom_names
        )
        configured_robot_bodies = {
            str(
                _value(
                    self.robot_config,
                    "base_body",
                    default="base_link",
                )
            ),
            *(
                str(name)
                for name in _value(
                    self.robot_config,
                    "link_body_names",
                    default=[],
                )
            ),
            str(
                _value(
                    self.gripper_config,
                    "parent_body",
                    default="gripper_base",
                )
            ),
            str(
                _value(
                    self.gripper_config,
                    "left_finger_body",
                    default="gripper_link1",
                )
            ),
            str(
                _value(
                    self.gripper_config,
                    "right_finger_body",
                    default="gripper_link2",
                )
            ),
        }
        self.robot_body_ids = frozenset(
            body_id
            for name in configured_robot_bodies
            if (
                body_id := self._name2id(
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                    required=False,
                )
            )
            >= 0
        )
        self.top_camera_id = self._name2id(
            mujoco.mjtObj.mjOBJ_CAMERA, self.top_camera_name
        )
        self.wrist_camera_id = self._name2id(
            mujoco.mjtObj.mjOBJ_CAMERA, self.wrist_camera_name
        )
        # Custom scenes without a policy view can still run the baseline.
        # The explicit policy capture API below requires this camera.
        self.policy_external_camera_id = self._name2id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.policy_external_camera_name,
            required=False,
        )
        self._base_camera_pos = self.model.cam_pos.copy()
        self._base_camera_quat = self.model.cam_quat.copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        self._base_light_ambient = self.model.light_ambient.copy()
        self._base_light_specular = self.model.light_specular.copy()
        self._base_light_dir = self.model.light_dir.copy()
        self._base_finger_friction = {
            geom_id: self.model.geom_friction[geom_id].copy()
            for geom_id in self.finger_geom_ids
        }

    def _read_runtime_parameters(self) -> None:
        self.randomization_enabled = bool(
            _value(self.scene_config, "randomization.enabled", default=True)
        )
        conveyor = dict(_value(self.scene_config, "conveyor", default={}))
        self.belt_length = _as_float(conveyor.get("length", 1.2), 1.2)
        self.belt_width = _as_float(conveyor.get("width", 0.25), 0.25)
        self.belt_height = _as_float(
            conveyor.get("surface_height", conveyor.get("height", 0.20)), 0.20
        )
        self.belt_center_x = _as_float(conveyor.get("center_x", 0.0), 0.0)
        self.belt_center_y = _as_float(conveyor.get("center_y", 0.0), 0.0)
        self.belt_min_x = self.belt_center_x - 0.5 * self.belt_length
        self.belt_max_x = self.belt_center_x + 0.5 * self.belt_length
        self.base_belt_speed = _as_float(
            conveyor.get("default_speed", conveyor.get("speed", 0.2)), 0.2
        )
        speed_range = conveyor.get("speed_range", [0.0, 1.0])
        self.belt_speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.belt_force_time_constant = _as_float(
            conveyor.get("velocity_time_constant", 0.06), 0.06
        )
        self.belt_force_limit = _as_float(conveyor.get("max_force", 12.0), 12.0)
        self.belt_speed_jitter = _as_float(
            _value(
                self.scene_config,
                "randomization.belt_speed_jitter.amplitude",
                "randomization.belt_speed_jitter.std",
                "conveyor.speed_jitter_std",
                "randomization.speed_jitter",
                default=0.01,
            ),
            0.01,
        )
        self.belt_speed_jitter_enabled = self.randomization_enabled and bool(
            _value(
                self.scene_config,
                "randomization.belt_speed_jitter.enabled",
                "randomization.enable_belt_speed_jitter",
                default=True,
            )
        )
        self.randomize_belt_speed = self.randomization_enabled and bool(
            _value(
                self.scene_config,
                "randomization.belt_speed.enabled",
                "randomization.enable_belt_speed",
                default=False,
            )
        )

        self.dynamic_intercept_enabled = bool(_value(self.scene_config, "workspace.dynamic_intercept.enabled", default=False))
        self.dynamic_intercept_x_range = tuple(float(x) for x in _value(self.scene_config, "workspace.dynamic_intercept.x_range", default=[0.02, 0.18]))
        if len(self.dynamic_intercept_x_range) != 2 or not np.isfinite(self.dynamic_intercept_x_range).all() or not 0 <= self.dynamic_intercept_x_range[0] < self.dynamic_intercept_x_range[1] <= 0.28:
            raise ValueError("dynamic_intercept.x_range must be inside the validated [0,0.28] m access corridor")
        self.belt_per_object_randomization = self.randomize_belt_speed and bool(_value(self.scene_config, "randomization.belt_speed.per_object", default=False))
        self.belt_speed_max_delta = float(_value(self.scene_config, "randomization.belt_speed.per_object_max_delta", default=0.025))
        self.belt_speed_max_acceleration = float(_value(self.scene_config, "randomization.belt_speed.max_acceleration", default=0.0))
        if not np.isfinite([self.belt_speed_max_delta, self.belt_speed_max_acceleration]).all() or self.belt_speed_max_delta < 0 or self.belt_speed_max_acceleration < 0:
            raise ValueError("Belt variation and acceleration must be finite and nonnegative")
        if self.belt_per_object_randomization and self.belt_speed_max_acceleration <= 0:
            raise ValueError("Per-object speed variation requires a positive slew limit")

        episode = dict(_value(self.scene_config, "episode", default={}))
        self.max_episode_time = _as_float(episode.get("max_time", 30.0), 30.0)
        self.max_episode_objects = int(episode.get("object_count", len(self.active_category_ids)))
        self.initial_object_count = int(episode.get("initial_objects", 2))
        spawn = dict(
            _value(
                self.scene_config,
                "spawning",
                "conveyor.spawn",
                default={},
            )
        )
        self.spawn_spacing_range = tuple(
            float(item)
            for item in spawn.get(
                "spacing_range",
                [spawn.get("min_spacing", 0.18), spawn.get("max_spacing", 0.32)],
            )
        )
        interval_seconds = spawn.get("interval_seconds")
        self.spawn_interval_seconds = (
            None
            if interval_seconds is None
            else _as_float(interval_seconds, -1.0)
        )
        if (
            self.spawn_interval_seconds is not None
            and self.spawn_interval_seconds <= 0.0
        ):
            raise ValueError("spawning.interval_seconds must be positive")
        interval_range = spawn.get("interval_range_seconds")
        self.spawn_interval_range_seconds = None if interval_range is None else tuple(float(x) for x in interval_range)
        if self.spawn_interval_range_seconds is not None and (len(self.spawn_interval_range_seconds) != 2 or not np.isfinite(self.spawn_interval_range_seconds).all() or not 0 < self.spawn_interval_range_seconds[0] <= self.spawn_interval_range_seconds[1]):
            raise ValueError("interval_range_seconds must be finite, positive and ordered")
        self.spawn_x = _as_float(
            spawn.get("x", self.belt_min_x + 0.06), self.belt_min_x + 0.06
        )
        self.spawn_clearance = _as_float(spawn.get("clearance", 0.12), 0.12)
        grasp_region = conveyor.get(
            "grasp_region",
            _value(self.scene_config, "workspace.grasp_region.x", default=[-0.28, 0.30]),
        )
        self.grasp_region = (float(grasp_region[0]), float(grasp_region[1]))
        if self.dynamic_intercept_enabled:
            dynamic_cfg = _value(self.scene_config, "workspace.dynamic_intercept", default={})
            for key, default in (("planning_search_allowance_s", 0.), ("candidate_step_m", .04),
                                 ("tracking_horizon_s", .35), ("servo_lookahead_s", .04),
                                 ("lift_vertical_distance_m", .06), ("lift_downstream_offset_m", .08)):
                value = float(dynamic_cfg.get(key, default))
                if not np.isfinite(value) or value < 0 or (key == "candidate_step_m" and value == 0):
                    raise ValueError(f"dynamic_intercept.{key} must be finite and non-negative (step positive)")
            tracking_range = np.asarray(dynamic_cfg.get("tracking_x_range", self.dynamic_intercept_x_range), dtype=float)
            if (tracking_range.shape != (2,) or not np.isfinite(tracking_range).all()
                or not self.grasp_region[0] <= tracking_range[0] < tracking_range[1] <= self.grasp_region[1]):
                raise ValueError("dynamic_intercept.tracking_x_range must lie inside the original grasp window")
        closing_exit_margin = max(
            0.0,
            _as_float(
                _value(
                    self.scene_config,
                    "workspace.closing_confirmation_exit_margin",
                    default=0.02,
                ),
                0.02,
            ),
        )
        access_gap_x = conveyor.get(
            "robot_side_access_gap_x",
            [self.grasp_region[0], self.grasp_region[1]],
        )
        access_gap_end_x = _as_float(access_gap_x[1], self.grasp_region[1])
        if access_gap_end_x < self.grasp_region[1]:
            raise ValueError(
                "conveyor.robot_side_access_gap_x must contain the grasp-region "
                "upper boundary"
            )
        # The grasp-region limit remains the hard deadline for starting a
        # closure.  A closure that has already started may use only this small
        # physical confirmation horizon, capped by both the robot-side rail
        # opening and the end of the conveyor.
        self.closing_grasp_limit_x = min(
            self.grasp_region[1] + closing_exit_margin,
            access_gap_end_x,
            self.belt_max_x,
        )
        self.reach_y = tuple(
            float(item)
            for item in conveyor.get(
                "reachable_y",
                _value(
                    self.scene_config,
                    "workspace.grasp_region.y",
                    default=[-0.18, 0.18],
                ),
            )
        )
        self.release_classification_timeout = max(
            0.0,
            _as_float(
                _value(
                    self.scene_config,
                    "workspace.release_classification_timeout",
                    default=0.75,
                ),
                0.75,
            ),
        )
        try:
            self.stable_contact_steps = int(
                _value(
                    self.gripper_config,
                    "stable_contact_steps",
                    default=1,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError("gripper.stable_contact_steps must be an integer") from error
        if self.stable_contact_steps < 1:
            raise ValueError("gripper.stable_contact_steps must be positive")
        self.close_contact_force_threshold = _as_float(
            _value(
                self.gripper_config,
                "close_contact_force_threshold",
                default=0.0,
            ),
            0.0,
        )
        if self.close_contact_force_threshold < 0.0:
            raise ValueError(
                "gripper.close_contact_force_threshold must be non-negative"
            )
        self.release_command_threshold = _as_float(
            _value(
                self.gripper_config,
                "grasp_assist.release_command_threshold",
                "release_command_threshold",
                default=0.047,
            ),
            0.047,
        )
        if self.release_command_threshold < 0.0:
            raise ValueError(
                "gripper.grasp_assist.release_command_threshold must be "
                "non-negative"
            )

        self.camera_period = 1.0 / max(
            1.0,
            _as_float(
                _value(
                    self.cameras_config,
                    "capture_hz",
                    "top.capture_hz",
                    default=15.0,
                ),
                15.0,
            ),
        )
        self.top_resolution = self._camera_resolution("top", (640, 480))
        self.wrist_resolution = self._camera_resolution("wrist", (320, 240))
        self.visual_latency = sum(
            self._latency(
                "camera_exposure",
                "camera_capture",
                "image_transport",
                "perception_inference",
                "exposure",
                "transport",
                "perception",
            )
        )
        self.planning_latency = self._latency_value(
            "trajectory_planning", "planning", default=0.04
        )
        self.control_latency = self._latency_value(
            "control_communication", "communication", default=0.015
        )
        self.robot_response_latency = self._latency_value(
            "robot_response", "mechanical_response", default=0.035
        )
        self.estimated_arm_arrival = _as_float(
            _value(
                self.latency_config,
                "estimated_arm_arrival",
                "prediction.estimated_arrival_time",
                default=0.65,
            ),
            0.65,
        )
        self.history_window = _as_float(
            _value(self.latency_config, "prediction.history_window", default=0.35),
            0.35,
        )
        self.predict_acceleration = bool(
            _value(
                self.latency_config,
                "prediction.use_acceleration",
                default=False,
            )
        )
        self._bins = self._read_bins()

    def _camera_resolution(
        self, camera: str, default: tuple[int, int]
    ) -> tuple[int, int]:
        resolution = _value(
            self.cameras_config,
            f"{camera}.resolution",
            f"{camera}_camera.resolution",
            default=list(default),
        )
        if isinstance(resolution, Mapping):
            width = int(resolution.get("width", default[0]))
            height = int(resolution.get("height", default[1]))
        else:
            width, height = (int(resolution[0]), int(resolution[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"{camera} camera resolution must be positive")
        return width, height

    def _latency_value(self, *names: str, default: float = 0.0) -> float:
        for name in names:
            candidates = [
                name,
                f"{name}_ms",
                f"components.{name}",
                f"components.{name}_ms",
                f"latencies.{name}",
                f"latencies.{name}_ms",
            ]
            if name == "image_transport":
                candidates.extend(("image_transfer", "image_transfer_ms"))
            for candidate in candidates:
                result = _value(self.latency_config, candidate, default=None)
                if result is None:
                    continue
                value = _as_float(result, default)
                if candidate.endswith("_ms") or value > 5.0:
                    value /= 1000.0
                return max(0.0, value)
        return max(0.0, float(default))

    def _latency(self, *names: str) -> list[float]:
        aliases = {
            "camera_exposure": ("camera_exposure", "exposure"),
            "camera_capture": ("camera_capture", "capture"),
            "image_transport": ("image_transport", "transport"),
            "perception_inference": ("perception_inference", "perception"),
        }
        canonical = names[:4]
        defaults = (0.02, 0.01, 0.025, 0.055)
        return [
            self._latency_value(*aliases.get(name, (name,)), default=default)
            for name, default in zip(canonical, defaults)
        ]

    def _read_bins(self) -> list[dict[str, Any]]:
        raw = _value(self.scene_config, "bins", "sorting_bins", default=[])
        result: list[dict[str, Any]] = []
        if isinstance(raw, Mapping):
            raw = list(raw.values())
        for index, item in enumerate(raw or []):
            if not isinstance(item, Mapping):
                continue
            center = item.get("center", item.get("position", [0.0, 0.48, 0.12]))
            size = item.get("inner_size", item.get("size", [0.22, 0.22, 0.16]))
            parsed = {
                "id": int(item.get("id", index)),
                "name": str(item.get("name", item.get("label", f"bin_{index}"))),
                "center": np.asarray(center, dtype=float),
                "size": np.asarray(size, dtype=float),
            }
            drop_quaternion = item.get(
                "drop_quaternion_wxyz",
                item.get("drop_quaternion"),
            )
            if drop_quaternion is not None:
                quaternion = np.asarray(drop_quaternion, dtype=float)
                if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
                    raise ValueError(
                        f"bins[{index}].drop_quaternion_wxyz must have four "
                        "finite values"
                    )
                norm = float(np.linalg.norm(quaternion))
                if norm < 1e-12:
                    raise ValueError(
                        f"bins[{index}].drop_quaternion_wxyz cannot be zero"
                    )
                parsed["drop_quaternion"] = quaternion / norm
            drop_joint_positions = item.get("drop_joint_positions")
            if drop_joint_positions is not None:
                joints = np.asarray(drop_joint_positions, dtype=float)
                if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                    raise ValueError(
                        f"bins[{index}].drop_joint_positions must have seven "
                        "finite values"
                    )
                parsed["drop_joint_positions"] = joints
            result.append(parsed)
        if not result:
            for index, x in enumerate((-0.35, 0.0, 0.35)):
                result.append(
                    {
                        "id": index,
                        "name": f"bin_{index}",
                        "center": np.array([x, 0.50, 0.12]),
                        "size": np.array([0.20, 0.20, 0.16]),
                    }
                )
        return result

    def _object_config_entries(self) -> list[Mapping[str, Any]]:
        if hasattr(self, "object_specs"):
            return [spec.as_dict() for spec in self.object_specs]
        raw: Any = self.objects_config
        if isinstance(raw, Mapping):
            raw = raw.get(
                "object_specs",
                raw.get("specs", raw.get("categories", raw.get("objects", raw))),
            )
        if isinstance(raw, Mapping):
            raw = [
                dict(value, category_id=int(key))
                for key, value in raw.items()
                if isinstance(value, Mapping) and str(key).lstrip("-").isdigit()
            ]
        return list(raw) if isinstance(raw, list) else []

    def _build_research_modules(self, seed: int | None) -> None:
        """Instantiate the reusable perception/task modules used by the env adapter."""

        from ..perception import (
            DepthGeometryVisionPipeline,
            GeometryPrototype,
            GraspWindowCalculator,
            LatencyBreakdown,
            MotionPredictor,
            MultiObjectObservationBuffer,
            RGBDPoseProvider,
        )
        from ..tasks import SortingTask, load_object_specs

        self.asset_object_specs = tuple(load_object_specs(
            self.asset_objects_document, base_dir=self.object_config_path.parent))
        self.object_specs = tuple(s for s in self.asset_object_specs if s.category_id in self.active_category_ids)
        self.sorting_task = SortingTask.from_config(
            self.config, specs=self.object_specs, seed=seed
        )
        self._object_spec_by_category = {
            int(spec.category_id): spec for spec in self.object_specs
        }
        self.latency_breakdown = LatencyBreakdown.from_mapping(self.config)
        self._base_latency_breakdown = self.latency_breakdown
        self.visual_latency = self.latency_breakdown.observation_delay
        self.planning_latency = self.latency_breakdown.trajectory_planning
        self.control_latency = self.latency_breakdown.control_communication
        self.robot_response_latency = self.latency_breakdown.robot_response
        self.motion_predictor = MotionPredictor.from_config(self.config)
        self.grasp_window_calculator = GraspWindowCalculator(self.config)
        self.observation_buffer = MultiObjectObservationBuffer(
            max_history_seconds=_as_float(
                _value(
                    self.latency_config,
                    "observation_history.duration_s",
                    default=2.0,
                ),
                2.0,
            ),
            default_delay=self.visual_latency,
        )
        self.perception_buffer = MultiObjectObservationBuffer(
            max_history_seconds=_as_float(
                _value(
                    self.latency_config,
                    "observation_history.duration_s",
                    default=2.0,
                ),
                2.0,
            ),
            default_delay=self.visual_latency,
        )
        minimum_mask_pixels = int(
            _value(self.perception_config, "min_mask_pixels", default=24)
        )
        self.rgbd_pose_provider = RGBDPoseProvider(
            minimum_mask_pixels=minimum_mask_pixels,
            full_confidence_pixels=max(100, minimum_mask_pixels * 8),
        )
        prototype_path = Path(
            str(
                _value(
                    self.perception_config,
                    "prototype_catalog_path",
                    default="tless_rgbd_prototypes.yaml",
                )
            )
        )
        if not prototype_path.is_absolute():
            prototype_path = (
                Path(str(self.config["config_dir"])) / prototype_path
            ).resolve()
        prototype_document = load_yaml(prototype_path)
        calibrated_entries = prototype_document.get("prototypes", [])
        calibrated_by_category = {
            int(item["category_id"]): item
            for item in calibrated_entries
            if isinstance(item, Mapping) and "category_id" in item
        }
        prototypes = []
        for spec in self.object_specs:
            calibrated = calibrated_by_category.get(int(spec.category_id), {})
            symmetry_kind = str(spec.symmetry.get("kind", "none"))
            prototypes.append(
                GeometryPrototype(
                    category_id=int(spec.category_id),
                    dimensions=spec.dimensions,
                    # PCA yaw is intentionally retained for two-fold parts;
                    # only continuously rotational parts have no useful yaw.
                    symmetric=symmetry_kind == "continuous_rotational",
                    feature_mean=(
                        None
                        if calibrated.get("feature_mean") is None
                        else np.asarray(calibrated["feature_mean"], dtype=float)
                    ),
                    feature_scale=(
                        None
                        if calibrated.get("feature_scale") is None
                        else np.asarray(calibrated["feature_scale"], dtype=float)
                    ),
                )
            )
        detection_roi = _value(
            self.perception_config, "detection_roi", default={}
        )
        classifier_config = _value(
            self.perception_config, "classifier", default={}
        )
        tracker_config = _value(
            self.perception_config, "tracker", default={}
        )
        self.vision_pipeline = DepthGeometryVisionPipeline(
            prototypes,
            support_plane_height=_as_float(
                _value(
                    self.perception_config,
                    "support_plane_z",
                    default=self.belt_height - 0.001,
                ),
                self.belt_height - 0.001,
            ),
            roi_x=_value(detection_roi, "x", default=[-0.58, 0.24]),
            roi_y=_value(detection_roi, "y", default=[-0.115, 0.115]),
            minimum_height=_as_float(
                _value(detection_roi, "minimum_height_m", default=0.004),
                0.004,
            ),
            maximum_height=_as_float(
                _value(detection_roi, "maximum_height_m", default=0.105),
                0.105,
            ),
            minimum_mask_pixels=minimum_mask_pixels,
            minimum_footprint=_as_float(
                _value(detection_roi, "minimum_footprint_m", default=0.012),
                0.012,
            ),
            maximum_footprint=_as_float(
                _value(detection_roi, "maximum_footprint_m", default=0.13),
                0.13,
            ),
            association_distance=_as_float(
                _value(
                    tracker_config,
                    "association_distance_m",
                    default=0.075,
                ),
                0.075,
            ),
            new_track_max_x=_as_float(
                _value(
                    tracker_config, "new_track_max_x", default=-0.20
                ),
                -0.20,
            ),
            trusted_measurement_max_x=_as_float(
                _value(
                    tracker_config,
                    "trusted_measurement_max_x",
                    default=-0.18,
                ),
                -0.18,
            ),
            max_track_age=_as_float(
                _value(self.perception_config, "max_track_age_s", default=2.40),
                2.40,
            ),
            coast_confidence_half_life=_as_float(
                _value(
                    self.perception_config,
                    "coast_confidence_half_life_s",
                    default=1.00,
                ),
                1.00,
            ),
            category_ema=_as_float(
                _value(classifier_config, "category_ema", default=0.65),
                0.65,
            ),
            category_evidence_mode=str(
                _value(classifier_config, "category_evidence_mode", default="ema")
            ),
            velocity_ema=_as_float(
                _value(tracker_config, "velocity_ema", default=0.55),
                0.55,
            ),
            max_supported_lateral_speed=_as_float(
                _value(self.perception_config,
                       "occlusion_filter.max_supported_lateral_speed_m_s",
                       default=0.001),
                0.001,
            ),
            category_temperature=_as_float(
                _value(classifier_config, "temperature", default=1.0),
                1.0,
            ),
            minimum_category_confidence=_as_float(
                _value(
                    classifier_config,
                    "minimum_confidence",
                    default=0.05,
                ),
                0.05,
            ),
            minimum_detection_confidence=_as_float(
                _value(
                    classifier_config,
                    "minimum_detection_confidence",
                    default=0.06,
                ),
                0.06,
            ),
        )
        self._vision_task_states: dict[int, VisionTaskState] = {}
        # This privileged association is used only by simulator physics and
        # metrics.  It is computed after visual inference and never supplies a
        # pose, category, mask, identity or object enumeration to the policy.
        self._vision_to_logical_evaluation: dict[int, int] = {}
        self._logical_to_vision_evaluation: dict[int, int] = {}

    def _build_object_slots(self) -> None:
        # Preserve the original ObjectSpec defaults/path normalization for all
        # physical slots; only active slots may enter the conveyor task.
        entries = [spec.as_dict() for spec in self.asset_object_specs]
        entry_by_category = {
            int(entry.get("category_id", index)): entry
            for index, entry in enumerate(entries)
        }
        self._objects: list[RuntimeObject] = []
        self._base_geom_properties: dict[int, dict[str, Any]] = {}
        self._base_body_properties: dict[int, tuple[float, np.ndarray]] = {}
        slot_count = int(self.scene_config["object_pool"]["count"])
        if set(entry_by_category) != set(range(slot_count)):
            raise ValueError("Asset catalog must cover the configured physical slots")
        for slot in range(slot_count):
            body_name = f"object_{slot:02d}"
            joint_name = f"{body_name}_freejoint"
            body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            geom_ids = tuple(range(geom_start, geom_start + geom_count))
            entry = dict(entry_by_category.get(slot, {}))
            runtime = RuntimeObject(
                slot=slot,
                body_name=body_name,
                joint_name=joint_name,
                body_id=body_id,
                joint_id=joint_id,
                geom_ids=geom_ids,
                qpos_adr=int(self.model.jnt_qposadr[joint_id]),
                qvel_adr=int(self.model.jnt_dofadr[joint_id]),
                category_id=slot,
                bin_id=int(entry["bin_id"]),
                metadata=entry,
            )
            self._objects.append(runtime)
            self._base_body_properties[body_id] = (
                float(self.model.body_mass[body_id]),
                self.model.body_inertia[body_id].copy(),
            )
            for geom_id in geom_ids:
                self._base_geom_properties[geom_id] = {
                    "contype": int(self.model.geom_contype[geom_id]),
                    "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                    "rgba": self.model.geom_rgba[geom_id].copy(),
                    "friction": self.model.geom_friction[geom_id].copy(),
                }
        self._logical_to_slot: dict[int, RuntimeObject] = {}
        self._next_logical_id = 0
        self._automatic_category_manifest = np.asarray(self.active_category_ids, dtype=int)
        self._parking_eq_ids = {
            slot: self._name2id(
                mujoco.mjtObj.mjOBJ_EQUALITY,
                f"parking_weld_{slot:02d}",
                required=False,
            )
            for slot in range(slot_count)
        }
        self._assist_eq_ids = {
            slot: self._name2id(
                mujoco.mjtObj.mjOBJ_EQUALITY,
                f"grasp_assist_weld_{slot:02d}",
                required=False,
            )
            for slot in range(slot_count)
        }

    def _build_controllers(self) -> None:
        from ..controllers.ik_solver import DampedLeastSquaresIK

        ik_config = _value(self.robot_config, "ik", default={})
        self.ik = DampedLeastSquaresIK(
            self.model,
            self.data,
            self.arm_joint_names,
            self.ee_site_name,
            ik_config,
        )
        self._path_validation_data = mujoco.MjData(self.model)
        self._path_validation_jacobian_position = np.zeros(
            (3, self.model.nv), dtype=float
        )
        self._path_validation_jacobian_rotation = np.zeros(
            (3, self.model.nv), dtype=float
        )
        robot_body_names = {
            str(_value(self.robot_config, "base_body", default="base_link")),
            *(
                str(name)
                for name in _value(
                    self.robot_config,
                    "link_body_names",
                    default=(),
                )
            ),
            "gripper_flange",
            "gripper_base",
            "gripper_link1",
            "gripper_link2",
        }
        self._robot_collision_body_ids = {
            self._name2id(
                mujoco.mjtObj.mjOBJ_BODY,
                name,
                required=False,
            )
            for name in robot_body_names
        }
        self._robot_collision_body_ids.discard(-1)
        self.robot_controller: Any | None = None
        self.gripper_controller: Any | None = None
        try:
            from ..controllers.gripper_controller import GripperController
            from ..controllers.robot_controller import RobotController

            self.gripper_controller = GripperController(
                self.model, self.data, self.robot_document
            )
            self.robot_controller = RobotController(
                self.model,
                self.data,
                self.robot_document,
                gripper=self.gripper_controller,
            )
        except (ImportError, TypeError, ValueError) as error:
            warnings.warn(
                f"Falling back to the environment's direct actuator adapter: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        velocity = _value(
            self.robot_config,
            "max_joint_velocity",
            "velocity_limits",
            "limits.velocity",
            default=[1.4] * 7,
        )
        acceleration = _value(
            self.robot_config,
            "max_joint_acceleration",
            "acceleration_limits",
            "limits.acceleration",
            default=[3.0] * 7,
        )
        self.max_joint_velocity = np.broadcast_to(
            np.asarray(velocity, dtype=float), (7,)
        ).copy()
        self.max_joint_acceleration = np.broadcast_to(
            np.asarray(acceleration, dtype=float), (7,)
        ).copy()
        self._nominal_max_joint_velocity = self.max_joint_velocity.copy()
        self._nominal_max_joint_acceleration = self.max_joint_acceleration.copy()
        self._motion_profile = "nominal"
        self._joint_target = self.data.qpos[self.arm_qpos_adrs].copy()
        self._joint_target_velocity = np.zeros(7, dtype=float)
        self._last_gripper_command = 1.0

    # ------------------------------------------------------------------
    # Reset, spawning, and object lifecycle
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")
        # A viewer-key reset may occur without calling policy.reset().  Stop
        # any request rooted in the pre-reset physics snapshot before mutating
        # live MuJoCo data; generation checks in the policy provide the second
        # stale-result barrier.
        for worker in self._planning_workers:
            worker.cancel_pending()
        self._scene_epoch += 1
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._episode_seed = int(seed)
            self._unseeded_reset_sequence = 0
        else:
            self._unseeded_reset_sequence += 1
        options = dict(options or {})
        mujoco.mj_resetData(self.model, self.data)
        self.metrics.reset()
        self._contact_closing_previous = False
        self._contact_attempt_active = False
        self._contact_attempt_succeeded = False
        self._held_payload_rail_contacts.reset()
        self._payload_contact_evaluation_events.clear()
        self._logical_to_slot.clear()
        self._next_logical_id = 0
        # Pre-sample the episode's category deck from the seeded RNG.  Picking
        # an index from the *currently inactive* slots made the same seed yield
        # a different third category when asynchronous planning shifted the
        # previous object's recycle tick.  A fixed permutation is independent
        # of lifecycle timing and still presents every T-LESS category once.
        manifest_seed = (
            np.random.SeedSequence(
                [int(self._episode_seed), 0x43415445]
            )
            if self._episode_seed is not None
            else np.random.SeedSequence(
                [0x43415445, self._unseeded_reset_sequence]
            )
        )
        self._automatic_category_manifest = np.asarray(
            np.random.default_rng(manifest_seed).permutation(self.active_category_ids), dtype=int
        )
        # Dedicated RNG keeps feed timing/speed independent of rendering and planning.
        self._dynamic_rng = np.random.default_rng(np.random.SeedSequence([int(self._episode_seed or 0), 0x44594E41]))
        self.dynamic_randomization_log: list[dict[str, Any]] = []
        self._last_spawn_encoder_distance = 0.0
        self._state_history.clear()
        self._belt_motion_history.clear()
        self._belt_encoder_distance = 0.0
        self._belt_motion_history.append((float(self.data.time), 0.0))
        self._ee_trajectory.clear()
        self._camera_frames.clear()
        self._display_camera_frames.clear()
        self._action_queue.clear()
        self._action_queue_overruns = 0
        self._last_camera_capture = -np.inf
        self._last_display_camera_capture = -np.inf
        self._last_observation = None
        self.observation_buffer.clear()
        self.perception_buffer.clear()
        self.rgbd_pose_provider.reset()
        self.vision_pipeline.reset()
        self._vision_task_states.clear()
        self._vision_to_logical_evaluation.clear()
        self._logical_to_vision_evaluation.clear()
        self.motion_predictor.reset()
        self.sorting_task.reset(seed)
        self.selected_target_id: int | None = None
        self._episode_started_at = float(self.data.time)
        self._completed_objects = 0
        self._severe_collision = False
        self._collision_active_last_step = False
        self._new_terminal_events: list[dict[str, Any]] = []
        self.set_motion_profile("nominal")

        speed = options.get("belt_speed")
        if speed is None and self.randomize_belt_speed:
            configured_range = _value(
                self.scene_config,
                "randomization.belt_speed.range",
                default=[0.15, 0.25],
            )
            speed = self._rng.uniform(float(configured_range[0]), float(configured_range[1]))
        self._commanded_belt_speed = self._clip_belt_speed(
            self.base_belt_speed if speed is None else float(speed)
        )
        self._belt_jitter_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._current_belt_speed = self._commanded_belt_speed
        self._apply_episode_randomization()

        for runtime in self._objects:
            self._deactivate_object(runtime, during_reset=True)

        home = np.asarray(
            _value(
                self.robot_config,
                "home_joint_positions",
                "wait_joint_positions",
                default=self.model.qpos0[self.arm_qpos_adrs],
            ),
            dtype=float,
        )
        if home.shape != (7,):
            raise ValueError("robot home_joint_positions must contain seven values")
        if bool(
            _value(
                self.robot_config,
                "initialize_at_wait_pose",
                default=True,
            )
        ):
            wait_position = np.asarray(
                _value(
                    self.scene_config,
                    "workspace.waiting_ee_position",
                    default=[-0.10, -0.10, 0.48],
                ),
                dtype=float,
            )
            wait_quaternion = np.asarray(
                _value(
                    self.scene_config,
                    "workspace.waiting_ee_quaternion_wxyz",
                    default=_value(
                        self.robot_config,
                        "end_effector.nominal_down_quaternion_wxyz",
                        default=[0.0, 1.0, 0.0, 0.0],
                    ),
                ),
                dtype=float,
            )
            wait_solution = self.ik.solve(
                wait_position, wait_quaternion, qpos_seed=home
            )
            if wait_solution.success:
                home = wait_solution.joint_positions
        self.data.qpos[self.arm_qpos_adrs] = home
        self.data.qvel[self.arm_dof_adrs] = 0.0
        self._joint_target = home.copy()
        self._joint_target_velocity[:] = 0.0
        self.data.ctrl[self.arm_actuator_ids] = home
        self._set_gripper_command(1.0)
        if self.robot_controller is not None:
            self.robot_controller.reset()

        mujoco.mj_forward(self.model, self.data)
        initial_count = int(options.get("initial_objects", self.initial_object_count))
        initial_count = max(0, min(initial_count, len(self.active_category_ids), self.max_episode_objects))
        initial_spacing = max(self.spawn_clearance, np.mean(self.spawn_spacing_range))
        for index in range(initial_count):
            self.spawn_object(x=self.spawn_x + index * initial_spacing)
        self._schedule_next_spawn()
        mujoco.mj_forward(self.model, self.data)
        self._ee_trajectory.append(self._ee_pose()[:3].copy())
        self._record_truth_state()
        self._capture_cameras(force=True)
        self._update_debug_markers()
        return self.get_observation()

    def _apply_episode_randomization(self) -> None:
        """Restore model baselines, then apply seeded episode-level perturbations."""

        self.model.cam_pos[:] = self._base_camera_pos
        self.model.cam_quat[:] = self._base_camera_quat
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.light_ambient[:] = self._base_light_ambient
        self.model.light_specular[:] = self._base_light_specular
        self.model.light_dir[:] = self._base_light_dir
        for geom_id, friction in self._base_finger_friction.items():
            self.model.geom_friction[geom_id] = friction

        latency_randomization = _value(
            self.latency_config, "randomization", default={}
        )
        latency_scale = 1.0
        if bool(_value(latency_randomization, "enabled", default=False)):
            bounds = _value(
                latency_randomization, "scale_range", default=[0.85, 1.15]
            )
            distribution = str(
                _value(
                    latency_randomization,
                    "distribution",
                    default="uniform",
                )
            )
            low, high = float(bounds[0]), float(bounds[1])
            if distribution == "truncated_normal":
                candidate = self._rng.normal(1.0, max((high - low) / 4.0, 1e-6))
                latency_scale = float(np.clip(candidate, low, high))
            else:
                latency_scale = float(self._rng.uniform(low, high))
        self.latency_breakdown = self._base_latency_breakdown.scaled(latency_scale)
        self.visual_latency = self.latency_breakdown.observation_delay
        self.planning_latency = self.latency_breakdown.trajectory_planning
        self.control_latency = self.latency_breakdown.control_communication
        self.robot_response_latency = self.latency_breakdown.robot_response
        self.observation_buffer.default_delay = self.visual_latency
        self.perception_buffer.default_delay = self.visual_latency

        extrinsics = _value(
            self.rendering_config, "extrinsic_randomization", default={}
        )
        if bool(_value(extrinsics, "enabled", default=False)):
            position_std = np.asarray(
                _value(extrinsics, "position_std_m", default=[0.002] * 3),
                dtype=float,
            )
            rotation_std = np.asarray(
                _value(extrinsics, "rotation_std_rad", default=[0.005] * 3),
                dtype=float,
            )
            for camera_id in (self.top_camera_id, self.wrist_camera_id):
                self.model.cam_pos[camera_id] += self._rng.normal(
                    0.0, position_std, size=3
                )
                rotvec = self._rng.normal(0.0, rotation_std, size=3)
                angle = float(np.linalg.norm(rotvec))
                if angle > 1e-12:
                    delta = np.r_[
                        math.cos(0.5 * angle),
                        math.sin(0.5 * angle) * rotvec / angle,
                    ]
                    result = np.empty(4)
                    mujoco.mju_mulQuat(
                        result, delta, self._base_camera_quat[camera_id]
                    )
                    self.model.cam_quat[camera_id] = result

        scene_randomization = _value(
            self.scene_config, "randomization", default={}
        )
        if not bool(_value(scene_randomization, "enabled", default=True)):
            return
        light_scale_config = _value(
            scene_randomization, "light_intensity_scale", default={}
        )
        if bool(_value(light_scale_config, "enabled", default=False)):
            bounds = _value(light_scale_config, "range", default=[0.85, 1.15])
            scale = float(self._rng.uniform(float(bounds[0]), float(bounds[1])))
            self.model.light_diffuse[:] = self._base_light_diffuse * scale
            self.model.light_ambient[:] = self._base_light_ambient * scale
            self.model.light_specular[:] = self._base_light_specular * scale
        light_direction = _value(
            scene_randomization, "light_direction", default={}
        )
        if bool(_value(light_direction, "enabled", default=False)):
            std = np.asarray(
                _value(light_direction, "std", default=[0.04, 0.04, 0.02]),
                dtype=float,
            )
            perturbed = self._base_light_dir + self._rng.normal(
                0.0, std, size=self._base_light_dir.shape
            )
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            self.model.light_dir[:] = perturbed / np.maximum(norms, 1e-9)
        finger_randomization = _value(
            scene_randomization, "gripper_friction_scale", default={}
        )
        if bool(_value(finger_randomization, "enabled", default=False)):
            bounds = _value(finger_randomization, "range", default=[0.95, 1.05])
            scale = float(self._rng.uniform(float(bounds[0]), float(bounds[1])))
            for geom_id, friction in self._base_finger_friction.items():
                self.model.geom_friction[geom_id] = friction * scale

    def _set_equality_active(self, equality_id: int, active: bool) -> None:
        if equality_id < 0:
            return
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[equality_id] = int(active)
        else:  # pragma: no cover - retained for older MuJoCo wheels.
            self.model.eq_active0[equality_id] = int(active)

    def _deactivate_object(
        self, runtime: RuntimeObject, *, during_reset: bool = False
    ) -> None:
        if runtime.logical_id is not None:
            logical_id = int(runtime.logical_id)
            visual_id = self._logical_to_vision_evaluation.pop(
                logical_id, None
            )
            if visual_id is not None:
                self._vision_to_logical_evaluation.pop(visual_id, None)
            self._logical_to_slot.pop(logical_id, None)
        self._set_equality_active(self._assist_eq_ids[runtime.slot], False)
        self._set_equality_active(self._parking_eq_ids[runtime.slot], True)
        start = runtime.qpos_adr
        self.data.qpos[start : start + 7] = self.model.qpos0[start : start + 7]
        self.data.qvel[runtime.qvel_adr : runtime.qvel_adr + 6] = 0.0
        for geom_id in runtime.geom_ids:
            base = self._base_geom_properties[geom_id]
            # Parked bodies are hidden and excluded from broadphase contacts.
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            self.model.geom_rgba[geom_id] = base["rgba"]
            self.model.geom_rgba[geom_id, 3] = 0.0
            self.model.geom_friction[geom_id] = base["friction"]
        base_mass, base_inertia = self._base_body_properties[runtime.body_id]
        self.model.body_mass[runtime.body_id] = base_mass
        self.model.body_inertia[runtime.body_id] = base_inertia
        runtime.logical_id = None
        runtime.state = "INACTIVE"
        runtime.failure_reason = None
        runtime.metadata.pop("_lane_y", None)
        runtime.metadata.pop("_supported_until", None)
        runtime.prediction_at_grasp = None
        runtime.actual_grasp_pose = None
        runtime.planned_grasp_pose = None
        runtime.planned_grasp_time = None
        runtime.compensation_enabled_at_selection = None
        runtime.lost_contact_since = None
        runtime.stable_contact_steps = 0
        runtime.contact_gap_steps = 0
        if not during_reset:
            mujoco.mj_forward(self.model, self.data)

    def _object_half_height(self, runtime: RuntimeObject) -> float:
        configured = runtime.metadata.get(
            "spawn_half_height",
            runtime.metadata.get("half_height", runtime.metadata.get("grasp_height")),
        )
        if configured is not None:
            value = float(configured)
            # grasp_height is commonly the full center height.
            if "grasp_height" in runtime.metadata and "spawn_half_height" not in runtime.metadata:
                value *= 0.5
            return float(np.clip(value, 0.008, 0.08))
        extent = 0.02
        for geom_id in runtime.geom_ids:
            geom_type = int(self.model.geom_type[geom_id])
            size = self.model.geom_size[geom_id]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                local = size[0]
            elif geom_type in {
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            }:
                local = size[1]
            else:
                local = size[2]
            extent = max(extent, float(local + abs(self.model.geom_pos[geom_id, 2])))
        return extent

    def _object_lowest_local_z(self, runtime: RuntimeObject) -> float:
        """Return the lowest collision point in the object's body frame.

        The bundled object bodies use their free-joint origin at the support
        plane, not at the centre of the bounding box.  Treating
        ``grasp_height / 2`` as a spawn half-height therefore dropped parts
        from 10--30 mm above the belt and made them bounce throughout the
        grasp window.
        """

        configured_lowest = runtime.metadata.get("lowest_local_z")
        if configured_lowest is not None:
            return float(configured_lowest)

        lowest = float("inf")
        for geom_id in runtime.geom_ids:
            geom_type = int(self.model.geom_type[geom_id])
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            rotation = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(rotation, self.model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                z_extent = float(size[0])
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                z_extent = float(
                    abs(rotation[2, 2]) * size[1] + size[0]
                )
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                axial = abs(float(rotation[2, 2]))
                z_extent = float(
                    axial * size[1]
                    + math.sqrt(max(0.0, 1.0 - axial * axial)) * size[0]
                )
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                z_extent = float(
                    np.linalg.norm(rotation[2, :3] * size[:3])
                )
            else:
                z_extent = float(
                    np.dot(np.abs(rotation[2, :3]), size[:3])
                )
            lowest = min(
                lowest,
                float(self.model.geom_pos[geom_id, 2]) - z_extent,
            )
        return 0.0 if not np.isfinite(lowest) else lowest

    def spawn_object(
        self,
        category_id: int | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> int | None:
        """Activate one predeclared free body at the upstream spawn line.

        Direct qpos writes are intentionally confined to this method and reset/
        recycle handling.
        """

        if self.metrics.total_objects >= self.max_episode_objects:
            return None
        inactive = [runtime for runtime in self._objects
                    if runtime.logical_id is None and runtime.category_id in self.active_category_ids]
        if not inactive:
            return None
        if category_id is None:
            category_randomization = self.randomization_enabled and bool(
                _value(
                    self.scene_config,
                    "randomization.object_category.enabled",
                    default=True,
                )
            )
            if category_randomization:
                manifest_index = int(self.metrics.total_objects)
                preferred_slot = int(
                    self._automatic_category_manifest[
                        manifest_index
                        % len(self._automatic_category_manifest)
                    ]
                )
                matching = [
                    runtime
                    for runtime in inactive
                    if runtime.slot == preferred_slot
                ]
                # The 10 s episode normally recycles a slot before its next
                # active-category deck use.  If a caller spawns faster and the
                # manifest slot is still active, fail closed rather than draw
                # a different category and destroy seeded reproducibility.
                if not matching:
                    return None
                runtime = matching[0]
            else:
                runtime = min(inactive, key=lambda item: item.slot)
        else:
            category = int(category_id)
            if category not in self.active_category_ids:
                raise ValueError(f"category_id must be active: {self.active_category_ids}")
            matching = [runtime for runtime in inactive if runtime.slot == category]
            if not matching:
                return None
            runtime = matching[0]

        spawn_randomization = _value(self.scene_config, "randomization", default={})
        randomization_active = self.randomization_enabled
        lateral_range = _value(
            spawn_randomization,
            "object_lateral_position.range",
            default=[
                self.belt_center_y - 0.30 * self.belt_width,
                self.belt_center_y + 0.30 * self.belt_width,
            ],
        )
        x_value = self.spawn_x if x is None else float(x)
        lateral_randomization = randomization_active and bool(
            _value(
                spawn_randomization,
                "object_lateral_position.enabled",
                default=True,
            )
        )
        y_value = (
            float(self._rng.uniform(float(lateral_range[0]), float(lateral_range[1])))
            if y is None and lateral_randomization
            else self.belt_center_y
            if y is None
            else float(y)
        )
        yaw_range = _value(
            spawn_randomization, "object_yaw.range", default=[-np.pi, np.pi]
        )
        yaw_randomization = randomization_active and bool(
            _value(spawn_randomization, "object_yaw.enabled", default=True)
        )
        yaw_value = (
            float(self._rng.uniform(float(yaw_range[0]), float(yaw_range[1])))
            if yaw is None and yaw_randomization
            else 0.0
            if yaw is None
            else float(yaw)
        )
        clearance = _as_float(
            _value(self.scene_config, "conveyor.spawn.z_clearance", default=0.004),
            0.004,
        )
        lowest_local_z = self._object_lowest_local_z(runtime)
        pose = np.r_[
            [
                x_value,
                y_value,
                self.belt_height + clearance - lowest_local_z,
            ],
            _quat_from_yaw(yaw_value),
        ]

        self._set_equality_active(self._parking_eq_ids[runtime.slot], False)
        self._set_equality_active(self._assist_eq_ids[runtime.slot], False)
        self.data.qpos[runtime.qpos_adr : runtime.qpos_adr + 7] = pose
        velocity = np.zeros(6, dtype=float)
        velocity[0] = self._current_belt_speed
        self.data.qvel[runtime.qvel_adr : runtime.qvel_adr + 6] = velocity
        for geom_id in runtime.geom_ids:
            base = self._base_geom_properties[geom_id]
            self.model.geom_contype[geom_id] = base["contype"]
            self.model.geom_conaffinity[geom_id] = base["conaffinity"]
            self.model.geom_rgba[geom_id] = base["rgba"]
            self.model.geom_friction[geom_id] = base["friction"]

        mass_scale = 1.0
        if randomization_active and bool(
            _value(
                spawn_randomization,
                "object_mass_scale.enabled",
                default=False,
            )
        ):
            bounds = _value(
                spawn_randomization, "object_mass_scale.range", default=[0.9, 1.1]
            )
            mass_scale = float(self._rng.uniform(float(bounds[0]), float(bounds[1])))
        base_mass, base_inertia = self._base_body_properties[runtime.body_id]
        requested_mass = runtime.metadata.get("mass")
        if requested_mass is not None:
            mass_scale *= float(requested_mass) / max(base_mass, 1e-9)
        self.model.body_mass[runtime.body_id] = base_mass * mass_scale
        self.model.body_inertia[runtime.body_id] = base_inertia * mass_scale

        if randomization_active and bool(
            _value(
                spawn_randomization,
                "object_friction_scale.enabled",
                default=False,
            )
        ):
            bounds = _value(
                spawn_randomization,
                "object_friction_scale.range",
                default=[0.9, 1.1],
            )
            scale = float(self._rng.uniform(float(bounds[0]), float(bounds[1])))
            for geom_id in runtime.geom_ids:
                self.model.geom_friction[geom_id] = (
                    self._base_geom_properties[geom_id]["friction"] * scale
                )
        if randomization_active and bool(
            _value(spawn_randomization, "object_color.enabled", default=False)
        ):
            color = self._rng.uniform(0.15, 0.95, size=3)
            for geom_id in runtime.geom_ids:
                self.model.geom_rgba[geom_id, :3] = color

        logical_id = self._next_logical_id
        self._next_logical_id += 1
        runtime.logical_id = logical_id
        runtime.state = "NEW"
        runtime.spawned_at = float(self.data.time)
        runtime.state_changed_at = float(self.data.time)
        runtime.selected_at = None
        runtime.grasped_at = None
        runtime.released_at = None
        runtime.lost_contact_since = None
        runtime.stable_contact_steps = 0
        runtime.contact_gap_steps = 0
        runtime.prediction_at_grasp = None
        runtime.actual_grasp_pose = None
        runtime.planned_grasp_pose = None
        runtime.planned_grasp_time = None
        runtime.compensation_enabled_at_selection = None
        runtime.failure_reason = None
        runtime.metadata.pop("_supported_until", None)
        runtime.metadata["_lane_y"] = y_value
        if self.belt_per_object_randomization and logical_id > 0:
            bounds = _value(self.scene_config, "randomization.belt_speed.range", default=[0.10, 0.20])
            previous_speed = self._commanded_belt_speed
            low = max(float(bounds[0]), previous_speed - self.belt_speed_max_delta)
            high = min(float(bounds[1]), previous_speed + self.belt_speed_max_delta)
            self._commanded_belt_speed = float(self._dynamic_rng.uniform(low, high))
        if self.dynamic_intercept_enabled:
            self.dynamic_randomization_log.append({
                "kind": "spawn", "timestamp": float(self.data.time), "object_id": logical_id,
                "category_id": runtime.category_id, "position": pose[:3].tolist(), "yaw": yaw_value,
                "belt_speed_at_spawn": self._current_belt_speed, "commanded_belt_speed": self._commanded_belt_speed,
                "encoder_feed_distance": self._belt_encoder_distance-self._last_spawn_encoder_distance,
                "mass_scale": mass_scale,
                "mass_kg": float(self.model.body_mass[runtime.body_id]),
                "geom_friction": self.model.geom_friction[list(runtime.geom_ids)].tolist(),
                "geom_rgba": self.model.geom_rgba[list(runtime.geom_ids)].tolist(),
            })
            self._last_spawn_encoder_distance = self._belt_encoder_distance
        self._logical_to_slot[logical_id] = runtime
        self.metrics.total_objects += 1
        self.metrics.valid_sorting_objects += 1
        mujoco.mj_forward(self.model, self.data)
        return logical_id

    def _schedule_next_spawn(self) -> None:
        if self.spawn_interval_range_seconds is not None:
            interval = float(self._dynamic_rng.uniform(*self.spawn_interval_range_seconds))
            self._next_spawn_time = float(self.data.time) + interval
            self.dynamic_randomization_log.append({"kind": "feed_schedule", "timestamp": float(self.data.time), "interval_seconds": interval, "next_spawn_time": self._next_spawn_time})
            return
        if self.spawn_interval_seconds is not None:
            self._next_spawn_time = (
                float(self.data.time) + self.spawn_interval_seconds
            )
            return

        spacing_randomization = self.randomization_enabled and bool(
            _value(
                self.scene_config,
                "randomization.object_spacing.enabled",
                default=True,
            )
        )
        configured_spacing = _value(
            self.scene_config,
            "randomization.object_spacing.value",
            default=np.mean(self.spawn_spacing_range),
        )
        spacing = (
            float(self._rng.uniform(*self.spawn_spacing_range))
            if spacing_randomization
            else float(configured_spacing)
        )
        speed = max(abs(self._commanded_belt_speed), 0.04)
        self._next_spawn_time = float(self.data.time) + spacing / speed

    def _try_periodic_spawn(self) -> None:
        if float(self.data.time) + 1e-12 < self._next_spawn_time:
            return
        upstream_occupied = any(
            runtime.active
            and self.data.xpos[runtime.body_id, 0] < self.spawn_x + self.spawn_clearance
            for runtime in self._objects
        )
        if not upstream_occupied:
            self.spawn_object()
        self._schedule_next_spawn()

    def recycle_terminal_objects(self, *, older_than: float = 0.4, retain_object_ids=()) -> None:
        for runtime in self._objects:
            if (
                runtime.logical_id is not None
                and runtime.state in TERMINAL_OBJECT_STATES
                and runtime.logical_id not in retain_object_ids
                and self.data.time - runtime.state_changed_at >= older_than
            ):
                self._deactivate_object(runtime)

    def set_object_state(
        self, object_id: int, state: str, *, reason: str | None = None
    ) -> bool:
        runtime = self._logical_to_slot.get(int(object_id))
        if runtime is None:
            return False
        normalized = str(getattr(state, "value", state)).upper()
        # Terminal lifecycle states are immutable.  Delayed policy events must
        # not turn one physical object into two completed/failed records.
        if runtime.state in TERMINAL_OBJECT_STATES or normalized == runtime.state:
            return False
        previous = runtime.state
        runtime.state = normalized
        runtime.state_changed_at = float(self.data.time)
        runtime.failure_reason = reason
        visual_id = (
            None
            if runtime.logical_id is None
            else self._logical_to_vision_evaluation.get(
                int(runtime.logical_id)
            )
        )
        if visual_id is not None and (
            normalized == "GRASPED" or normalized in TERMINAL_OBJECT_STATES
        ):
            task_state = self._vision_task_states.get(visual_id)
            if task_state is not None:
                task_state.state = normalized
                task_state.failure_reason = reason
        if normalized == "GRASPED":
            runtime.grasped_at = float(self.data.time)
            runtime.actual_grasp_pose = self.data.xpos[runtime.body_id].copy()
        if normalized in TERMINAL_OBJECT_STATES:
            self._completed_objects += 1
            event = {
                "timestamp": float(self.data.time),
                "object_id": runtime.logical_id,
                "category_id": runtime.category_id,
                "previous_state": previous,
                "state": normalized,
                "reason": reason,
            }
            if visual_id is not None:
                event["track_id"] = visual_id
            self._new_terminal_events.append(event)
            if normalized == "SORTED":
                self.metrics.correctly_sorted += 1
                if runtime.grasped_at is not None:
                    self.metrics.processing_times.append(
                        float(self.data.time - runtime.spawned_at)
                    )
            elif normalized == "MISSED":
                self.metrics.missed += 1
            elif normalized == "SKIPPED":
                self.metrics.skipped += 1
            else:
                self.metrics.failed += 1
                normalized_reason = (reason or "").lower()
                if "dropped" in normalized_reason:
                    self.metrics.dropped += 1
                if normalized_reason.startswith("wrong_bin"):
                    self.metrics.wrong_bin += 1
            assert runtime.logical_id is not None
            compensation_enabled = (
                self.enable_motion_compensation
                if runtime.compensation_enabled_at_selection is None
                else runtime.compensation_enabled_at_selection
            )
            self.metrics.record_outcome(
                runtime.logical_id,
                compensation_enabled=compensation_enabled,
                success=normalized == "SORTED",
            )
            selected_runtime_id = self._policy_id_for_runtime(runtime)
            if self.selected_target_id == selected_runtime_id:
                self.selected_target_id = None
        return True

    # ------------------------------------------------------------------
    # Physics and action application
    # ------------------------------------------------------------------
    @property
    def belt_speed(self) -> float:
        return float(self._current_belt_speed)

    def set_belt_speed(self, speed: float) -> float:
        self._commanded_belt_speed = self._clip_belt_speed(speed)
        return self._commanded_belt_speed

    def _record_belt_motion(self) -> None:
        """Integrate encoder speed over the physical interval just completed.

        This observes the commanded conveyor motion without modifying physics.
        The speed was sampled before mj_step and applied over this interval.
        """
        now = float(self.data.time)
        previous = self._belt_motion_history[-1][0]
        if now > previous:
            self._belt_encoder_distance += self._current_belt_speed * (now - previous)
            self._belt_motion_history.append((now, self._belt_encoder_distance))

    def get_belt_displacement(self, start_time: float, end_time: float) -> float | None:
        """Read encoder travel between covered timestamps; never extrapolate.

        A real adapter can supply cumulative encoder position with the same
        contract. Missing history returns None so callers keep their existing
        fallback rather than inventing zero travel.
        """
        start, end = float(start_time), float(end_time)
        history = self._belt_motion_history
        if (not np.all(np.isfinite([start, end])) or end < start
                or not history or start < history[0][0] - 1e-9
                or end > history[-1][0] + 1e-9):
            return None

        def distance_at(timestamp: float) -> float:
            later_time, later_distance = history[-1]
            for earlier_time, earlier_distance in reversed(history):
                if earlier_time <= timestamp:
                    span = later_time - earlier_time
                    if span <= 0.0:
                        return earlier_distance
                    fraction = (timestamp - earlier_time) / span
                    return earlier_distance + fraction * (later_distance - earlier_distance)
                later_time, later_distance = earlier_time, earlier_distance
            return history[0][1]

        return float(distance_at(end) - distance_at(start))

    def _clip_belt_speed(self, speed: float) -> float:
        if not np.isfinite(speed):
            raise ValueError("belt speed must be finite")
        return float(np.clip(speed, self.belt_speed_range[0], self.belt_speed_range[1]))

    def toggle_motion_compensation(self) -> bool:
        self.enable_motion_compensation = not self.enable_motion_compensation
        return self.enable_motion_compensation

    def _update_belt_speed(self) -> None:
        if self.belt_speed_jitter_enabled and self.belt_speed_jitter > 0.0:
            t = float(self.data.time)
            smooth = (
                0.72 * math.sin(2.0 * math.pi * 0.73 * t + self._belt_jitter_phase)
                + 0.28
                * math.sin(2.0 * math.pi * 1.91 * t + 0.37 * self._belt_jitter_phase)
            )
            jitter = self.belt_speed_jitter * smooth
        else:
            jitter = 0.0
        desired = self._clip_belt_speed(self._commanded_belt_speed + jitter)
        if self.belt_speed_max_acceleration > 0:
            delta = self.belt_speed_max_acceleration * float(self.model.opt.timestep)
            desired = float(self._current_belt_speed + np.clip(desired-self._current_belt_speed, -delta, delta))
        self._current_belt_speed = desired

    def _contact_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            pairs.add((min(first, second), max(first, second)))
        return pairs

    def _object_supported(
        self, runtime: RuntimeObject, pairs: set[tuple[int, int]]
    ) -> bool:
        touching = any(
            (min(geom_id, self.belt_geom_id), max(geom_id, self.belt_geom_id))
            in pairs
            for geom_id in runtime.geom_ids
        )
        if touching:
            # Keep the tangential belt servo active through the sub-millimetre
            # contact gaps produced by the compliant solver.  Without this
            # short grace period, light parts receive force on only isolated
            # 2 ms steps and stall despite resting visibly on the belt.
            runtime.metadata["_supported_until"] = float(self.data.time) + 0.08
            return True
        return float(self.data.time) <= float(
            runtime.metadata.get("_supported_until", -np.inf)
        )

    def _both_fingers_touch(
        self, runtime: RuntimeObject, pairs: set[tuple[int, int]] | None = None
    ) -> bool:
        if self.gripper_controller is not None:
            try:
                return bool(
                    self.gripper_controller.both_fingers_in_contact(runtime.geom_ids)
                )
            except (AttributeError, TypeError):
                pass
        contact_pairs = self._contact_pairs() if pairs is None else pairs
        left, right = self.finger_geom_ids[:2]
        object_geoms = set(runtime.geom_ids)
        left_contact = any(
            (min(left, geom), max(left, geom)) in contact_pairs for geom in object_geoms
        )
        right_contact = any(
            (min(right, geom), max(right, geom)) in contact_pairs
            for geom in object_geoms
        )
        return left_contact and right_contact

    def _finger_object_contact_forces(
        self, runtime: RuntimeObject
    ) -> tuple[float, float]:
        """Return summed normal contact force for each jaw on one object.

        The MJCF touch sites are useful observation channels, but their small
        volumes do not cover every point on the imported NERO finger meshes.
        Lifecycle confirmation therefore uses the same physical contacts as
        bilateral-contact detection and sums MuJoCo's normal force directly.
        """

        object_geoms = set(runtime.geom_ids)
        forces = [0.0, 0.0]
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            finger_index: int | None = None
            for index, finger_geom in enumerate(self.finger_geom_ids[:2]):
                if (geom1 == finger_geom and geom2 in object_geoms) or (
                    geom2 == finger_geom and geom1 in object_geoms
                ):
                    finger_index = index
                    break
            if finger_index is None:
                continue
            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, wrench
            )
            forces[finger_index] += abs(float(wrench[0]))
        return float(forces[0]), float(forces[1])

    def _gripper_opening_command(self) -> float:
        """Map the normalized last action back to the actuator command unit."""

        if self.gripper_controller is not None:
            config = self.gripper_controller.config
            denominator = float(config.action_open - config.action_closed)
            fraction = float(
                np.clip(
                    (self._last_gripper_command - config.action_closed)
                    / denominator,
                    0.0,
                    1.0,
                )
            )
            commands = (
                self.gripper_controller.closed_ctrl
                + fraction
                * (
                    self.gripper_controller.open_ctrl
                    - self.gripper_controller.closed_ctrl
                )
            )
            return float(np.max(commands))

        fraction = 0.5 * (self._last_gripper_command + 1.0)
        commands = []
        for actuator_id in self.gripper_actuator_ids:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            commands.append(low + fraction * (high - low))
        return float(max(commands, default=0.0))

    def _apply_conveyor_forces(self) -> None:
        self.data.xfrc_applied[:, :] = 0.0
        pairs = self._contact_pairs()
        for runtime in self._objects:
            if not runtime.active or runtime.state not in BELT_OBJECT_STATES:
                continue
            if not self._object_supported(runtime, pairs):
                continue
            # MuJoCo has no native endless-belt tangential surface velocity.
            # The explicit conveyor/object contact pairs use low tangential
            # friction, so this contact-gated free-joint velocity constraint
            # moves parts at the requested belt speed without injecting normal
            # contact energy.  All other DoFs and contacts remain physical.
            self.data.qvel[runtime.qvel_adr] = self._current_belt_speed
            lane_y = float(
                runtime.metadata.get("_lane_y", self.belt_center_y)
            )
            lateral_velocity = np.clip(
                3.0 * (lane_y - float(self.data.qpos[runtime.qpos_adr + 1])),
                -0.12,
                0.12,
            )
            self.data.qvel[runtime.qvel_adr + 1] = lateral_velocity

    def _queue_action(self, action: Mapping[str, Any] | None) -> None:
        if action is None:
            return
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping with 'arm' and 'gripper' keys")
        normalized = {
            "arm": np.asarray(action.get("arm"), dtype=float).copy()
            if action.get("arm") is not None
            else None,
            "gripper": (
                None
                if action.get("gripper") is None
                else float(action.get("gripper"))
            ),
            # This marker is emitted only by the policy after a cached global
            # trajectory sample has passed its live edge check.  The robot
            # controller independently verifies the discrete position,
            # velocity, acceleration and stopping envelope before accepting
            # it without a duplicate online target filter.
            "arm_time_parameterized": (
                action.get("arm_time_parameterized", False) is True
            ),
        }
        expected = 7 if self.action_type == "joint_position" else 6
        if normalized["arm"] is not None and normalized["arm"].shape != (expected,):
            raise ValueError(
                f"{self.action_type} arm action must have shape ({expected},), "
                f"got {normalized['arm'].shape}"
            )
        apply_at = float(self.data.time) + self.control_latency
        self._action_queue.append((apply_at, normalized))

    def _apply_due_actions(self) -> None:
        due_actions: list[dict[str, Any]] = []
        while (
            self._action_queue
            and self._action_queue[0][0] <= self.data.time + 1e-12
        ):
            _, action = self._action_queue.popleft()
            due_actions.append(action)
        if not due_actions:
            return

        # Arm and gripper actions are level commands, so the newest due value
        # supersedes every older value in an overrun batch.  Replaying the
        # intermediate values here would mutate RobotController's target
        # history several times without an intervening physics step, while
        # falsely assigning a complete control_dt to every mutation.  The
        # surviving command consequently cannot retain an upstream passthrough
        # request; the ordinary limiter must validate it from the last target
        # that was actually presented to physics.
        action = due_actions[-1]
        if len(due_actions) > 1:
            self._action_queue_overruns += 1
            action = dict(action)
            action["arm_time_parameterized"] = False

        if self.robot_controller is not None:
            self.robot_controller.apply_action(
                action, action_mode=self.action_type, dt=self.control_dt
            )
            if action.get("gripper") is not None:
                self._last_gripper_command = float(action["gripper"])
        else:
            self._apply_action_direct(action)

    def _apply_action_direct(self, action: Mapping[str, Any]) -> None:
        arm = action.get("arm")
        if arm is not None:
            if self.action_type == "ee_delta_pose":
                pose = self._ee_pose()
                target_position = pose[:3] + np.asarray(arm[:3], dtype=float)
                target_quaternion = pose[3:]
                rotation = np.asarray(arm[3:], dtype=float)
                angle = float(np.linalg.norm(rotation))
                if angle > 1e-12:
                    half = 0.5 * angle
                    delta_quat = np.r_[
                        math.cos(half), math.sin(half) * rotation / angle
                    ]
                    composed = np.empty(4)
                    mujoco.mju_mulQuat(composed, delta_quat, target_quaternion)
                    target_quaternion = composed
                result = self.ik.solve(
                    target_position,
                    target_quaternion,
                    qpos_seed=self.data.qpos[self.arm_qpos_adrs],
                )
                # The direct adapter is a simulation fallback, but it must
                # obey the same fail-closed contract as a physical backend.
                requested = (
                    result.joint_positions
                    if result.success
                    else self.data.qpos[self.arm_qpos_adrs].copy()
                )
            else:
                requested = np.asarray(arm, dtype=float)
            joint_ranges = self.model.jnt_range[self.arm_joint_ids]
            requested = np.clip(requested, joint_ranges[:, 0], joint_ranges[:, 1])
            desired_velocity = (requested - self._joint_target) / self.control_dt
            desired_velocity = np.clip(
                desired_velocity, -self.max_joint_velocity, self.max_joint_velocity
            )
            velocity_delta = np.clip(
                desired_velocity - self._joint_target_velocity,
                -self.max_joint_acceleration * self.control_dt,
                self.max_joint_acceleration * self.control_dt,
            )
            self._joint_target_velocity = np.clip(
                self._joint_target_velocity + velocity_delta,
                -self.max_joint_velocity,
                self.max_joint_velocity,
            )
            self._joint_target += self._joint_target_velocity * self.control_dt
            self._joint_target = np.clip(
                self._joint_target, joint_ranges[:, 0], joint_ranges[:, 1]
            )
            self.data.ctrl[self.arm_actuator_ids] = self._joint_target
        if action.get("gripper") is not None:
            self._set_gripper_command(float(action["gripper"]))

    def _set_gripper_command(self, command: float) -> None:
        command = float(np.clip(command, -1.0, 1.0))
        self._last_gripper_command = command
        if self.gripper_controller is not None:
            self.gripper_controller.command(command, dt=self.control_dt)
            return
        fraction = 0.5 * (command + 1.0)
        for actuator_id in self.gripper_actuator_ids:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = low + fraction * (high - low)

    def _enable_grasp_assist(self, runtime: RuntimeObject) -> None:
        equality_id = self._assist_eq_ids[runtime.slot]
        if equality_id < 0:
            return
        body1 = int(self.model.eq_obj1id[equality_id])
        body2 = int(self.model.eq_obj2id[equality_id])
        position1 = self.data.xpos[body1].copy()
        quaternion1 = self.data.xquat[body1].copy()
        position2 = self.data.xpos[body2].copy()
        quaternion2 = self.data.xquat[body2].copy()
        inverse_position = np.empty(3)
        inverse_quaternion = np.empty(4)
        relative_position = np.empty(3)
        relative_quaternion = np.empty(4)
        mujoco.mju_negPose(
            inverse_position, inverse_quaternion, position1, quaternion1
        )
        mujoco.mju_mulPose(
            relative_position,
            relative_quaternion,
            inverse_position,
            inverse_quaternion,
            position2,
            quaternion2,
        )
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quaternion
        self._set_equality_active(equality_id, True)

    def _disable_grasp_assist(self, runtime: RuntimeObject) -> None:
        self._set_equality_active(self._assist_eq_ids[runtime.slot], False)

    def _bin_containing(self, position: np.ndarray) -> int | None:
        for bin_spec in self._bins:
            center = bin_spec["center"]
            half = 0.5 * bin_spec["size"]
            if (
                abs(position[0] - center[0]) <= half[0]
                and abs(position[1] - center[1]) <= half[1]
                and position[2] <= center[2] + half[2] + 0.05
                and position[2] >= max(-0.05, center[2] - half[2] - 0.05)
            ):
                return int(bin_spec["id"])
        return None

    def _finish_contact_grasp_attempt(self) -> None:
        if self._contact_attempt_active and not self._contact_attempt_succeeded:
            self.metrics.empty_grasps += 1
        self._contact_attempt_active = False

    def _update_contact_grasp_attempt(self, closing: bool) -> None:
        if closing and not self._contact_closing_previous:
            if not any(runtime.state == "GRASPED" for runtime in self._objects):
                self.metrics.grasp_attempts += 1
                self._contact_attempt_active = True
                self._contact_attempt_succeeded = False
        elif not closing:
            self._finish_contact_grasp_attempt()
        self._contact_closing_previous = closing

    def _update_grasp_and_lifecycle(self) -> None:
        pairs = self._contact_pairs()
        closing = self._last_gripper_command < 0.75
        if self.grasp_event_source == "physical_contacts":
            self._update_contact_grasp_attempt(closing)
        for runtime in self._objects:
            if runtime.logical_id is None or runtime.state in TERMINAL_OBJECT_STATES:
                continue
            position = self.data.xpos[runtime.body_id].copy()
            bilateral = self._both_fingers_touch(runtime, pairs)

            if runtime.state == "NEW" and self.data.time - runtime.spawned_at > 0.04:
                self.set_object_state(runtime.logical_id, "TRACKING")
            if (
                runtime.state == "TRACKING"
                and self.grasp_region[0] - 0.10 <= position[0] <= self.grasp_region[1]
            ):
                self.set_object_state(runtime.logical_id, "READY_TO_PLAN")

            # Adaptive ObjectSpec grip-width commands may be above zero even
            # though they are well below the fully-open action (+1).  Treat
            # those commands as closing/holding so bilateral contact remains
            # the physical grasp confirmation.
            if self.grasp_event_source == "physical_contacts":
                if runtime.state == "CLOSING" and not closing:
                    self.set_object_state(runtime.logical_id, "TRACKING")
                elif (closing and runtime.state in BELT_OBJECT_STATES
                      and runtime.state != "CLOSING"
                      and self.grasp_region[0] <= position[0] <= self.grasp_region[1]):
                    touching = any(
                        (min(finger, geom), max(finger, geom)) in pairs
                        for finger in self.finger_geom_ids[:2] for geom in runtime.geom_ids
                    )
                    if touching:
                        self.set_object_state(runtime.logical_id, "CLOSING")
            contact_forces = (
                self._finger_object_contact_forces(runtime)
                if bilateral
                else (0.0, 0.0)
            )
            force_confirmed = min(contact_forces) >= (
                self.close_contact_force_threshold
            )
            acquiring = (
                runtime.state == "CLOSING"
                and bilateral
                and force_confirmed
                and closing
            )
            if acquiring:
                runtime.stable_contact_steps += 1
                runtime.contact_gap_steps = 0
            elif runtime.state == "CLOSING":
                # Mesh contacts can disappear for one or two 2 ms solver
                # frames while a rigid object remains physically captured.
                # Debounce those gaps, but discard the accumulated evidence
                # after a full configured confirmation window without valid
                # bilateral force.
                runtime.contact_gap_steps += 1
                if runtime.contact_gap_steps >= self.stable_contact_steps:
                    runtime.stable_contact_steps = 0
            elif runtime.state != "GRASPED":
                runtime.stable_contact_steps = 0
                runtime.contact_gap_steps = 0

            if (
                acquiring
                and runtime.stable_contact_steps >= self.stable_contact_steps
            ):
                self.metrics.successful_grasps += 1
                if self.grasp_event_source == "physical_contacts":
                    self._contact_attempt_succeeded = True
                policy_id = self._policy_id_for_runtime(runtime)
                prediction = (
                    None
                    if policy_id is None
                    else self.get_predicted_grasp_pose(policy_id)
                )
                runtime.prediction_at_grasp = (
                    None if prediction is None else np.asarray(prediction[:3]).copy()
                )
                self.set_object_state(runtime.logical_id, "GRASPED")
                if runtime.prediction_at_grasp is not None:
                    error = float(
                        np.linalg.norm(
                            runtime.prediction_at_grasp - runtime.actual_grasp_pose
                        )
                    )
                    self.metrics.prediction_errors.append(error)
                if self.grasp_assist:
                    self._enable_grasp_assist(runtime)

            if runtime.state == "GRASPED":
                if (
                    self._gripper_opening_command()
                    >= self.release_command_threshold
                ):
                    self._disable_grasp_assist(runtime)
                    if runtime.released_at is None:
                        runtime.released_at = float(self.data.time)
                bin_id = self._bin_containing(position)
                if bin_id is not None and runtime.released_at is not None:
                    if bin_id == runtime.bin_id:
                        self.set_object_state(runtime.logical_id, "SORTED")
                    else:
                        self.set_object_state(
                            runtime.logical_id,
                            "FAILED",
                            reason=f"wrong_bin:{bin_id}",
                        )
                    continue
                if runtime.released_at is not None:
                    # Once the jaws have opened, classify the placement on
                    # its own settling horizon.  Waiting for a later global
                    # "closing" command left out-of-bin parts in GRASPED for
                    # tens of seconds and blocked subsequent scheduling.
                    runtime.lost_contact_since = None
                    if (
                        self.data.time - runtime.released_at
                        >= self.release_classification_timeout
                    ):
                        self.set_object_state(
                            runtime.logical_id,
                            "FAILED",
                            reason="dropped_outside_bin",
                        )
                    continue
                if bilateral:
                    runtime.lost_contact_since = None
                elif closing and not self.grasp_assist:
                    if runtime.lost_contact_since is None:
                        runtime.lost_contact_since = float(self.data.time)
                    elif self.data.time - runtime.lost_contact_since > 0.35:
                        self.set_object_state(
                            runtime.logical_id, "FAILED", reason="dropped"
                        )
                        continue

            missed_x = _as_float(
                _value(
                    self.scene_config,
                    "conveyor.missed_x",
                    default=self.belt_max_x + 0.06,
                ),
                self.belt_max_x + 0.06,
            )
            if runtime.state != "GRASPED" and (
                position[0] > missed_x or position[2] < -0.15
            ):
                self.set_object_state(runtime.logical_id, "MISSED", reason="belt_end")
            elif runtime.state in BELT_OBJECT_STATES and position[0] > (
                self.closing_grasp_limit_x
                if runtime.state == "CLOSING"
                else self.grasp_region[1]
            ):
                # Planning and motion must finish inside the normal grasp
                # region.  Only a closure that already started there gets a
                # bounded confirmation horizon inside the physical rail gap;
                # it is still aborted at closing_grasp_limit_x.
                self.set_object_state(
                    runtime.logical_id,
                    "SKIPPED",
                    reason="grasp_window_expired",
                )
            elif (
                runtime.state in {"PLANNING", "EXECUTING"}
                and runtime.planned_grasp_time is not None
                and self.data.time
                > runtime.planned_grasp_time
                + _as_float(
                    _value(
                        self.scene_config,
                        "workspace.grasp_deadline_tolerance",
                        default=0.20,
                    ),
                    0.20,
                )
            ):
                self.set_object_state(
                    runtime.logical_id,
                    "SKIPPED",
                    reason="planned_grasp_time_expired",
                )

    def _detect_held_payload_rail_collisions(self) -> None:
        """Score carried-object impacts without altering control or physics.

        Ordinary belt support, bin placement and contacts after jaw release
        are excluded. Positive normal force is required; margin-only geometry
        contacts do not count. A 20 ms contact-gap debounce merges one impact.
        """

        forces: dict[tuple[int, int], float] = {}
        wrench = np.empty(6, dtype=float)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            key = held_payload_rail_key(
                int(contact.geom1), int(contact.geom2),
                self._payload_evaluation_geometry, self._payload_evaluation_rail_geoms,
            )
            if key is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            normal_force = float(wrench[0])
            if np.isfinite(normal_force) and normal_force > 0.0:
                forces[key] = forces.get(key, 0.0) + normal_force
        now = float(self.data.time)
        for event in self._held_payload_rail_contacts.update(now, forces):
            self._payload_contact_evaluation_events.append({
                "kind": "held_payload_rail_collision", "timestamp": now, **event,
            })

    def _detect_severe_collision(self) -> None:
        finger_ids = set(self.finger_geom_ids)
        severe = False
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in finger_ids or geom2 in finger_ids:
                continue
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            robot_contact = (
                body1 in self.robot_body_ids or body2 in self.robot_body_ids
            )
            workstation_contact = any(
                token in name1 or token in name2
                for token in ("conveyor", "bin_", "camera_stand", "safety", "robot_platform")
            )
            if not (robot_contact and workstation_contact):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if abs(force[0]) > 120.0:
                severe = True
                break
        if severe and not self._collision_active_last_step:
            self.metrics.collisions += 1
        self._collision_active_last_step = severe
        self._severe_collision = severe

    def step(
        self, action: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed environment")
        viewer_commands = self.process_pending_viewer_commands()
        if viewer_commands["reset"]:
            # A command computed from the pre-reset scene must never be queued
            # into the new epoch.  The returned observation lets the runner
            # reset its policy before producing the next command.
            action = None
        # Policy callbacks run between environment steps.  Preserve terminal
        # events they create so the following step reports and rewards them
        # exactly once alongside any terminal events produced by physics.
        pending_terminal_events = list(self._new_terminal_events)
        self._new_terminal_events = []
        self._payload_contact_evaluation_events = []
        self._queue_action(action)
        for _ in range(self.frame_skip):
            self._apply_due_actions()
            target_interpolator = getattr(self, '_physics_target_interpolator', None)
            if target_interpolator is not None:
                target_interpolator.apply()
            self._update_belt_speed()
            self._apply_conveyor_forces()
            observer = getattr(self, '_physics_substep_observer', None)
            before_qvel = self.data.qvel[self.arm_dof_adrs].copy() if observer is not None else None
            mujoco.mj_step(self.model, self.data)
            if observer is not None:
                observer(self.model, self.data, before_qvel)
            self._record_belt_motion()
            self._update_grasp_and_lifecycle()
            self._detect_held_payload_rail_collisions()
            self._detect_severe_collision()
            self._record_truth_state()
            self._try_periodic_spawn()
            self.recycle_terminal_objects()
            self._capture_cameras()
        self._ee_trajectory.append(self._ee_pose()[:3].copy())
        self._update_debug_markers()
        self._append_viewer_video_frame()
        if self.render_mode == "human":
            self.render()

        step_terminal_events = [
            *pending_terminal_events,
            *self._new_terminal_events,
        ]
        reward = -0.001
        for event in step_terminal_events:
            if event["state"] == "SORTED":
                reward += 10.0
            elif event["state"] == "SKIPPED":
                reward -= 0.2
            else:
                reward -= 2.0
        terminated = bool(
            self.metrics.total_objects >= self.max_episode_objects
            and self._completed_objects >= self.max_episode_objects
        )
        truncated = bool(self.data.time - self._episode_started_at >= self.max_episode_time)
        if (terminated or truncated) and self.grasp_event_source == "physical_contacts":
            self._finish_contact_grasp_attempt()
        observation = self.get_observation()
        info = {
            "metrics": self.get_metrics(),
            "events": step_terminal_events,
            # Evaluation truth stays outside the observation consumed by policy.
            "evaluation_events": list(self._payload_contact_evaluation_events),
            "motion_compensation": self.enable_motion_compensation,
            "action_type": self.action_type,
            "viewer_commands": viewer_commands["commands"],
            "viewer_reset": bool(viewer_commands["reset"]),
        }
        self._new_terminal_events = []
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Delayed observations and motion prediction
    # ------------------------------------------------------------------
    def _snapshot_runtime(self, runtime: RuntimeObject) -> dict[str, Any]:
        qpos = self.data.qpos[runtime.qpos_adr : runtime.qpos_adr + 7]
        qvel = self.data.qvel[runtime.qvel_adr : runtime.qvel_adr + 6]
        return {
            "object_id": int(runtime.logical_id) if runtime.logical_id is not None else None,
            "category_id": runtime.category_id,
            "bin_id": runtime.bin_id,
            "body_name": runtime.body_name,
            "position": qpos[:3].copy(),
            "quaternion": qpos[3:7].copy(),
            "linear_velocity": qvel[:3].copy(),
            "angular_velocity": qvel[3:6].copy(),
            "timestamp": float(self.data.time),
            "state": runtime.state,
        }

    def _record_truth_state(self) -> None:
        snapshot = {
            int(runtime.logical_id): self._snapshot_runtime(runtime)
            for runtime in self._objects
            if runtime.logical_id is not None
        }
        self._state_history.append((float(self.data.time), snapshot))
        self.observation_buffer.append_frame(
            float(self.data.time), snapshot, mark_missing=True
        )

    def _runtime_for_policy_object(
        self, object_id: int
    ) -> RuntimeObject | None:
        """Resolve a policy ID only at the simulator bookkeeping boundary."""

        identifier = int(object_id)
        if self.perception_mode == "simulator_history":
            return self._logical_to_slot.get(identifier)
        logical_id = self._vision_to_logical_evaluation.get(identifier)
        return (
            None
            if logical_id is None
            else self._logical_to_slot.get(int(logical_id))
        )

    def _policy_id_for_runtime(self, runtime: RuntimeObject) -> int | None:
        if runtime.logical_id is None:
            return None
        if self.perception_mode == "simulator_history":
            return int(runtime.logical_id)
        return self._logical_to_vision_evaluation.get(int(runtime.logical_id))

    def _vision_task_state(
        self, track_id: int, category_id: int | None = None,
        *, observed_timestamp: float | None = None, update_category: bool = True,
    ) -> VisionTaskState:
        identifier = int(track_id)
        task_state = self._vision_task_states.get(identifier)
        if task_state is None:
            if category_id is None:
                raise KeyError(f"visual track {identifier} has no category")
            task_state = VisionTaskState(
                track_id=identifier, category_id=int(category_id),
                category_observed_timestamp=(
                    float(observed_timestamp)
                    if update_category and observed_timestamp is not None else None
                ),
            )
            self._vision_task_states[identifier] = task_state
        elif (
            update_category
            and category_id is not None
            and task_state.state in {"NEW", "TRACKING", "READY_TO_PLAN", "PLANNING"}
        ):
            # Capture can be ahead of the delivered camera stream. Only a
            # newer, delivered classification may revise a reservation; an
            # older delayed reader must never roll it back. Direct untimed
            # initialization is retained only before any delivered sample.
            newer_measurement = bool(
                observed_timestamp is not None
                and np.isfinite(observed_timestamp)
                and (
                    task_state.category_observed_timestamp is None
                    or float(observed_timestamp)
                    > task_state.category_observed_timestamp + 1e-9
                )
            )
            untimed_initialization = bool(
                observed_timestamp is None
                and task_state.category_observed_timestamp is None
                and task_state.state != "PLANNING"
            )
            if newer_measurement or untimed_initialization:
                changed = task_state.category_id != int(category_id)
                task_state.category_id = int(category_id)
                if newer_measurement:
                    task_state.category_observed_timestamp = float(observed_timestamp)
                if changed and task_state.state == "PLANNING":
                    # The policy invalidates its old category-dependent route
                    # before its next action. Keep this reservation, but do
                    # not feed the old category's grasp geometry/time back to
                    # that fresh plan or retain an obsolete VLA instruction.
                    task_state.planned_grasp_pose = None
                    task_state.planned_grasp_time = None
                    if self.selected_target_id == identifier:
                        self.sorting_task.create_assignment(
                            task_state.category_id, object_id=str(identifier),
                            include_skip_rule=True,
                        )
                        # Mirror invalidation in evaluator bookkeeping only;
                        # no body pose/category is used to choose the class.
                        runtime = self._runtime_for_policy_object(identifier)
                        if runtime is not None:
                            runtime.planned_grasp_pose = None
                            runtime.planned_grasp_time = None
        return task_state

    def _associate_visual_tracks_for_evaluation(
        self, detections: Any
    ) -> None:
        """Associate visual tracks to physics bodies for scoring only.

        The matching runs *after* the vision pipeline has fixed every public
        output.  It uses only XY proximity, never category, and cannot alter a
        detection pose/class/confidence/ID.  A real robot adapter does not need
        this map; MuJoCo needs it solely to attribute contacts and terminal
        metrics to a physical body.
        """

        active_runtimes = [
            runtime
            for runtime in self._objects
            if runtime.logical_id is not None
            and runtime.state not in TERMINAL_OBJECT_STATES
        ]
        active_logical_ids = {
            int(runtime.logical_id) for runtime in active_runtimes
        }
        visible_track_ids = {int(item.track_id) for item in detections}
        for track_id, logical_id in tuple(
            self._vision_to_logical_evaluation.items()
        ):
            runtime = self._logical_to_slot.get(logical_id)
            association_is_pinned = bool(
                runtime is not None
                and (
                    self.selected_target_id == track_id
                    or runtime.state
                    in {
                        "PLANNING",
                        "EXECUTING",
                        "CLOSING",
                        "GRASPED",
                    }
                )
            )
            if logical_id not in active_logical_ids or (
                track_id not in visible_track_ids
                and not association_is_pinned
            ):
                self._vision_to_logical_evaluation.pop(track_id, None)
                self._logical_to_vision_evaluation.pop(logical_id, None)

        unmatched_runtimes = [
            runtime
            for runtime in active_runtimes
            if int(runtime.logical_id)
            not in self._logical_to_vision_evaluation
        ]
        unmatched_detections = [
            item
            for item in detections
            if int(item.track_id)
            not in self._vision_to_logical_evaluation
        ]
        pairs: list[tuple[float, int, RuntimeObject]] = []
        for detection in unmatched_detections:
            for runtime in unmatched_runtimes:
                distance = float(
                    np.linalg.norm(
                        np.asarray(detection.position[:2], dtype=float)
                        - self.data.xpos[runtime.body_id, :2]
                    )
                )
                pairs.append((distance, int(detection.track_id), runtime))
        used_tracks: set[int] = set()
        used_logical: set[int] = set()
        for distance, track_id, runtime in sorted(
            pairs, key=lambda item: item[0]
        ):
            logical_id = int(runtime.logical_id)
            if (
                distance > 0.12
                or track_id in used_tracks
                or logical_id in used_logical
                or track_id in self._vision_to_logical_evaluation
                or logical_id in self._logical_to_vision_evaluation
            ):
                continue
            self._vision_to_logical_evaluation[track_id] = logical_id
            self._logical_to_vision_evaluation[logical_id] = track_id
            used_tracks.add(track_id)
            used_logical.add(logical_id)

    def _spec_for_policy_object(self, object_id: int) -> Any | None:
        if self.perception_mode == "simulator_history":
            runtime = self._logical_to_slot.get(int(object_id))
            category_id = None if runtime is None else int(runtime.category_id)
        else:
            task_state = self._vision_task_states.get(int(object_id))
            category_id = (
                None if task_state is None else int(task_state.category_id)
            )
        return (
            None
            if category_id is None
            else self._object_spec_by_category.get(category_id)
        )

    def _state_at(self, timestamp: float) -> tuple[float, dict[int, dict[str, Any]]]:
        if not self._state_history:
            return float(self.data.time), {}
        selected = self._state_history[0]
        for entry in self._state_history:
            if entry[0] <= timestamp + 1e-12:
                selected = entry
            else:
                break
        return selected

    def _history_for_object(
        self, object_id: int, *, end_time: float, window: float | None = None
    ) -> list[dict[str, Any]]:
        start = end_time - (self.history_window if window is None else float(window))
        samples: list[dict[str, Any]] = []
        for timestamp, snapshot in self._state_history:
            if timestamp < start - 1e-12 or timestamp > end_time + 1e-12:
                continue
            state = snapshot.get(int(object_id))
            if state is not None:
                samples.append(state)
        return samples

    def _observed_state_at(
        self, object_id: int, timestamp: float
    ) -> tuple[float, dict[str, Any] | None]:
        """Read the policy-visible source selected for this environment.

        Rendered mode reads only pose measurements produced from a captured
        RGB-D/segmentation frame.  Simulator history remains an explicit
        regression/ablation mode and is never used as a vision fallback.
        """

        if self.perception_mode == "simulator_history":
            sample_time, snapshot = self._state_at(timestamp)
            return sample_time, snapshot.get(int(object_id))
        history = self.perception_buffer.history(str(object_id))
        if history is None:
            return float(timestamp), None
        sample = history.read_at(
            timestamp,
            mode="previous",
            allow_extrapolation=False,
        )
        if sample is None:
            return float(timestamp), None
        state = dict(sample.state)
        state.setdefault("timestamp", float(sample.timestamp))
        return float(sample.timestamp), state

    def _observation_history_for_object(
        self, object_id: int, *, end_time: float, window: float | None = None
    ) -> list[dict[str, Any]]:
        if self.perception_mode == "simulator_history":
            return self._history_for_object(
                object_id,
                end_time=end_time,
                window=window,
            )
        history = self.perception_buffer.history(str(object_id))
        if history is None:
            return []
        start = end_time - (
            self.history_window if window is None else float(window)
        )
        result: list[dict[str, Any]] = []
        for sample in history.between(start, end_time):
            state = dict(sample.state)
            state.setdefault("timestamp", float(sample.timestamp))
            result.append(state)
        return result

    def _estimate_motion(
        self,
        object_id: int,
        observed_time: float,
        observed: Mapping[str, Any],
        history_samples: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        samples = (
            self._observation_history_for_object(
                object_id, end_time=observed_time
            )
            if history_samples is None
            else history_samples
        )
        fallback_velocity = np.asarray(
            observed.get(
                "linear_velocity",
                [self._current_belt_speed, 0.0, 0.0],
            ),
            dtype=float,
        )
        if len(samples) < 2:
            return fallback_velocity, np.zeros(3), 0.25
        times = np.asarray([sample["timestamp"] for sample in samples], dtype=float)
        positions = np.asarray([sample["position"] for sample in samples], dtype=float)
        relative = times - times[-1]
        velocity = np.zeros(3, dtype=float)
        acceleration = np.zeros(3, dtype=float)
        residuals: list[float] = []
        order = 2 if self.predict_acceleration and len(samples) >= 4 else 1
        for axis in range(3):
            coefficients = np.polyfit(relative, positions[:, axis], order)
            if order == 2:
                acceleration[axis] = 2.0 * coefficients[0]
                velocity[axis] = coefficients[1]
            else:
                velocity[axis] = coefficients[0]
            fitted = np.polyval(coefficients, relative)
            residuals.extend((positions[:, axis] - fitted).tolist())
        residual_rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
        sample_factor = min(1.0, len(samples) / 6.0)
        confidence = float(
            np.clip(sample_factor * math.exp(-residual_rms / 0.012), 0.05, 1.0)
        )
        return velocity, acceleration, confidence

    def _dynamic_intercept_prediction_lead(self, position, velocity, ee_position, **timing):
        """Prediction ETA hook; the default retains the original two-leg model."""
        return intercept_lead(position, velocity, ee_position, **timing)

    def _prediction_record(
        self, object_id: int, *, observation_time: float | None = None
    ) -> dict[str, Any] | None:
        runtime = self._runtime_for_policy_object(int(object_id))
        if self.perception_mode == "simulator_history" and runtime is None:
            return None
        now = float(self.data.time)
        requested_time = (
            now - self.visual_latency if observation_time is None else observation_time
        )
        observed_timestamp, observed = self._observed_state_at(
            int(object_id), requested_time
        )
        if observed is None:
            return None
        if self.perception_mode == "rendered_rgbd":
            detected_category_id = int(observed["category_id"])
            task_state = self._vision_task_state(
                int(object_id), detected_category_id,
                observed_timestamp=observed_timestamp,
            )
            category_id = int(task_state.category_id)
            control_state = task_state.state
            planned_grasp_pose = task_state.planned_grasp_pose
            planned_grasp_time = task_state.planned_grasp_time
        else:
            assert runtime is not None
            category_id = int(runtime.category_id)
            task_state = None
            control_state = runtime.state
            planned_grasp_pose = runtime.planned_grasp_pose
            planned_grasp_time = runtime.planned_grasp_time
        spec = self._object_spec_by_category.get(category_id)
        if spec is None:
            return None
        object_metadata = dict(spec.metadata)
        observed_position = np.asarray(observed["position"], dtype=float)
        observed_quaternion = np.asarray(observed["quaternion"], dtype=float)
        age = max(0.0, now - observed_timestamp)
        maximum_track_age = _as_float(
            _value(
                self.perception_config,
                "max_track_age_s",
                default=0.60,
            ),
            0.60,
        )
        if (
            self.perception_mode == "rendered_rgbd"
            and age > maximum_track_age
        ):
            # A missed mask may coast briefly through an arm occlusion, but a
            # stale visual sample must not remain an immortal ghost target.
            return None
        measurement_confidence = float(
            np.clip(observed.get("confidence", 1.0), 0.0, 1.0)
        )
        if self.perception_mode == "rendered_rgbd":
            nominal_age = self.visual_latency + self.camera_period
            coast_age = max(0.0, age - nominal_age)
            half_life = max(
                _as_float(
                    _value(
                        self.perception_config,
                        "coast_confidence_half_life_s",
                        default=0.20,
                    ),
                    0.20,
                ),
                1e-6,
            )
            measurement_confidence *= math.exp(
                -math.log(2.0) * coast_age / half_life
            )
        history_samples = self._observation_history_for_object(
            int(object_id), end_time=observed_timestamp
        )
        minimum_history_samples = int(
            _value(
                self.latency_config,
                "observation_history.minimum_samples_for_velocity",
                default=3,
            )
        )
        history_ready = len(history_samples) >= max(1, minimum_history_samples)
        module_prediction: Any | None = None
        try:
            current_prediction = self.motion_predictor.predict(
                history_samples,
                current_time=now,
                system_latency=self.visual_latency,
                robot_arrival_time=0.0,
                belt_velocity=self._current_belt_speed,
                enable_motion_compensation=True,
                object_id=str(object_id),
            )
            estimated_velocity = current_prediction.estimated_velocity.copy()
            estimated_acceleration = current_prediction.estimated_acceleration.copy()
            current_estimate = current_prediction.current_estimated_position.copy()
            confidence = float(current_prediction.confidence)
        except (TypeError, ValueError):
            estimated_velocity, estimated_acceleration, confidence = self._estimate_motion(
                int(object_id),
                observed_timestamp,
                observed,
                history_samples,
            )
            current_estimate = (
                observed_position
                + estimated_velocity * age
                + 0.5 * estimated_acceleration * age * age
            )
        confidence *= measurement_confidence
        if control_state in BELT_OBJECT_STATES:
            # Before grasp, the support constraint is a better short-horizon
            # model than a high-order fit over only a handful of delayed
            # frames.  The latter can briefly jump by tens of centimetres and
            # invalidate an otherwise reachable plan.
            observed_velocity = np.asarray(
                observed.get("linear_velocity", estimated_velocity),
                dtype=float,
            )
            lateral_speed_limit = _as_float(
                _value(
                    self.perception_config,
                    "occlusion_filter.max_supported_lateral_speed_m_s",
                    default=0.001,
                ),
                0.001,
            )
            supported_speed = float(
                np.clip(
                    0.75 * observed_velocity[0]
                    + 0.25 * estimated_velocity[0],
                    0.0,
                    1.20 * self._current_belt_speed,
                )
            )
            current_estimate = observed_position.copy()
            encoder_travel = (
                self.get_belt_displacement(observed_timestamp, now)
                if self.perception_mode == "rendered_rgbd" else None
            )
            if encoder_travel is not None:
                # The visual tracker has already propagated its last clean
                # mask to the capture timestamp with this same encoder. Only
                # integrate the remaining camera/processing delay here. Fitting
                # synthetic coasted positions and mixing that derivative into
                # the current velocity double-counted belt-speed changes.
                current_estimate[0] += encoder_travel
                supported_speed = float(self._current_belt_speed)
            else:
                current_estimate[0] += supported_speed * age
            current_estimate[2] = self.belt_height - 0.001
            estimated_velocity = estimated_velocity.copy()
            estimated_velocity[0] = supported_speed
            estimated_velocity[1] = float(
                np.clip(
                    estimated_velocity[1],
                    -lateral_speed_limit,
                    lateral_speed_limit,
                )
            )
            estimated_velocity[2] = 0.0
            estimated_acceleration = estimated_acceleration.copy()
            estimated_acceleration[:] = 0.0
        grasp_height = float(spec.grasp_height)
        nominal_cartesian_speed = _as_float(
            _value(
                self.robot_config,
                "controller.nominal_cartesian_speed",
                default=0.22,
            ),
            0.22,
        )
        grasp_center_estimate = current_estimate + np.array(
            [0.0, 0.0, grasp_height]
        )
        approach_clearance = _as_float(
            _value(
                self.scene_config,
                "workspace.pregrasp_clearance",
                default=0.10,
            ),
            0.10,
        )
        preferred_grasp_x = _as_float(
            _value(
                self.scene_config,
                "workspace.preferred_grasp_x",
                default=0.08,
            ),
            0.08,
        )
        pregrasp_center_estimate = grasp_center_estimate.copy()
        if control_state in BELT_OBJECT_STATES and not self.dynamic_intercept_enabled:
            # The acquisition route goes to a fixed interception station, not
            # to the workpiece's current upstream location.  Measuring the arm
            # path to that moving observation made the estimated route longer
            # the earlier a fast object was detected.  At 0.30 m/s this could
            # push an otherwise valid x=0.08 m intercept beyond the x=0.28 m
            # hard boundary before the policy was even allowed to plan.
            pregrasp_center_estimate[0] = float(
                np.clip(
                    preferred_grasp_x,
                    self.grasp_region[0],
                    self.grasp_region[1],
                )
            )
        pregrasp_center_estimate[2] += approach_clearance
        distance_to_pregrasp = float(
            np.linalg.norm(
                self.data.site_xpos[self.ee_site_id]
                - pregrasp_center_estimate
            )
        )
        speed_floor = max(nominal_cartesian_speed, 1e-3)
        geometric_arrival = (
            distance_to_pregrasp / speed_floor
            + approach_clearance / speed_floor
        )
        maximum_horizon = _as_float(
            _value(
                self.latency_config,
                "prediction.maximum_horizon_s",
                default=3.5,
            ),
            3.5,
        )
        # Estimate the actual two-leg approach: move to the high pregrasp,
        # then descend.  Measuring the diagonal distance directly to the
        # workpiece and adding another full descent allowance counted the
        # same vertical travel twice and rejected feasible late targets.
        dynamic_arrival = np.clip(
            max(
                self.estimated_arm_arrival,
                geometric_arrival,
            ),
            0.04,
            max(0.04, maximum_horizon),
        )
        lead = (
            self.planning_latency
            + self.control_latency
            + self.robot_response_latency
            + float(dynamic_arrival)
        )
        if control_state in BELT_OBJECT_STATES and self.dynamic_intercept_enabled:
            # Reserve explicit background-search time before the geometric
            # approach. Outer lanes may require several fully checked IK /
            # descent alternatives. Starting only when the no-search ETA
            # reaches the corridor leaves no time to reject those branches.
            search_allowance = float(_value(
                self.scene_config, "workspace.dynamic_intercept.planning_search_allowance_s", default=0.0
            ))
            if not np.isfinite(search_allowance) or search_allowance < 0:
                raise ValueError("planning_search_allowance_s must be finite and non-negative")
            lead = self._dynamic_intercept_prediction_lead(current_estimate, estimated_velocity, self.data.site_xpos[self.ee_site_id],
                grasp_height=grasp_height, approach_clearance=approach_clearance,
                cartesian_speed=speed_floor, minimum_motion_time=self.estimated_arm_arrival,
                latency=self.planning_latency+self.control_latency+self.robot_response_latency+search_allowance,
                maximum_horizon=maximum_horizon)
            dynamic_arrival = max(0.04, lead-self.planning_latency-self.control_latency-self.robot_response_latency)
        if control_state in BELT_OBJECT_STATES and not self.dynamic_intercept_enabled:
            # Prefer a repeatable intercept station in the middle of the access
            # gap.  This gives the arm time to reach a high pregrasp and then
            # synchronise its descent, instead of sweeping open fingers through
            # the belt while chasing a continually moving endpoint.
            if (
                supported_speed > 1e-4
                and preferred_grasp_x > current_estimate[0]
            ):
                station_lead = (
                    preferred_grasp_x - current_estimate[0]
                ) / supported_speed
                lead = max(lead, min(station_lead, maximum_horizon))
        grasp_time = now + lead
        full_horizon = max(0.0, grasp_time - observed_timestamp)
        try:
            module_prediction = self.motion_predictor.predict(
                history_samples,
                current_time=now,
                system_latency=self.visual_latency,
                robot_arrival_time=lead,
                belt_velocity=self._current_belt_speed,
                enable_motion_compensation=self.enable_motion_compensation,
                object_id=str(object_id),
            )
            compensated_position = module_prediction.predicted_position.copy()
            estimated_velocity = module_prediction.estimated_velocity.copy()
            estimated_acceleration = module_prediction.estimated_acceleration.copy()
            confidence = (
                float(module_prediction.confidence)
                * measurement_confidence
            )
        except (TypeError, ValueError):
            compensated_position = (
                observed_position
                + estimated_velocity * full_horizon
                + 0.5 * estimated_acceleration * full_horizon * full_horizon
            )
        if control_state in BELT_OBJECT_STATES:
            # A supported workpiece follows the belt in-plane.  Do not
            # extrapolate the brief vertical settling velocity over the full
            # arm-arrival horizon; all bundled object frames place their
            # lowest collision geometry at local z≈0.
            #
            # The RGB-D stream is only 15 Hz.  Its first velocity fit includes
            # belt spin-up and otherwise under-predicts the intercept by many
            # centimetres.  Reapply the contact-supported belt prior after the
            # generic predictor (which returns its own fitted velocity), using
            # the measured current estimate as the integration origin.
            compensated_position = current_estimate.copy()
            compensated_position[0] += supported_speed * lead
            compensated_position[1] += float(
                np.clip(
                    estimated_velocity[1],
                    -lateral_speed_limit,
                    lateral_speed_limit,
                )
            ) * lead
            compensated_position[2] = self.belt_height - 0.001
            estimated_velocity = estimated_velocity.copy()
            estimated_velocity[0] = supported_speed
            estimated_velocity[1] = float(
                np.clip(
                    estimated_velocity[1],
                    -lateral_speed_limit,
                    lateral_speed_limit,
                )
            )
            estimated_velocity[2] = 0.0
        if (
            planned_grasp_pose is not None
            and control_state in {"PLANNING", "EXECUTING", "CLOSING", "GRASPED"}
        ):
            if planned_grasp_time is not None:
                grasp_time = planned_grasp_time
                remaining = max(0.0, grasp_time - now)
                # Keep the intercept time fixed, but continuously correct its
                # position from the newest delayed observation.  Freezing both
                # time and pose made the gripper wait at an obsolete point and
                # produced systematic empty grasps when belt slip accumulated.
                if control_state in BELT_OBJECT_STATES:
                    # Very short histories make acceleration extrapolation
                    # explode.  Supported objects are constrained by the belt,
                    # so use a bounded measured belt-axis velocity and only a
                    # small lateral drift term.
                    predicted_position = current_estimate.copy()
                    predicted_position[0] += supported_speed * remaining
                    predicted_position[1] += float(
                        np.clip(
                            estimated_velocity[1],
                            -lateral_speed_limit,
                            lateral_speed_limit,
                        )
                    ) * remaining
                    predicted_position[2] = self.belt_height - 0.001
                else:
                    predicted_position = (
                        current_estimate
                        + estimated_velocity * remaining
                        + 0.5
                        * estimated_acceleration
                        * remaining
                        * remaining
                    )
            else:
                predicted_position = compensated_position
        else:
            predicted_position = (
                compensated_position
                if self.enable_motion_compensation
                else observed_position.copy()
            )
        grasp_target_y_range = np.asarray(
            object_metadata.get(
                "grasp_target_y_range",
                _value(
                    self.scene_config,
                    "workspace.grasp_target_y_range",
                    default=self.reach_y,
                ),
            ),
            dtype=float,
        ).reshape(-1)
        if (
            grasp_target_y_range.shape == (2,)
            and np.all(np.isfinite(grasp_target_y_range))
            and grasp_target_y_range[0] <= grasp_target_y_range[1]
        ):
            # Keep the perceived object centre untouched.  Only bias the TCP
            # intercept slightly toward the belt centre when an outer-lane
            # centre target sits on NERO's collision-free IK boundary.  The
            # configured range is inside the physical jaw capture tolerance.
            predicted_position = predicted_position.copy()
            predicted_position[1] = float(
                np.clip(
                    predicted_position[1],
                    grasp_target_y_range[0],
                    grasp_target_y_range[1],
                )
            )
        truth_qpos = (
            self.data.qpos[runtime.qpos_adr : runtime.qpos_adr + 7].copy()
            if self.perception_mode == "simulator_history" and runtime is not None
            else None
        )
        gate_position = (
            truth_qpos[:3] if truth_qpos is not None else current_estimate
        )
        speed_x = max(float(estimated_velocity[0]), 1e-5)
        window_start = now + max(
            0.0, (self.grasp_region[0] - current_estimate[0]) / speed_x
        )
        window_end = now + max(
            0.0,
            min(
                (self.grasp_region[1] - current_estimate[0]) / speed_x,
                (self.belt_max_x - current_estimate[0]) / speed_x,
            ),
        )
        in_geometric_window = bool(
            self.grasp_region[0] <= gate_position[0] <= self.grasp_region[1]
            and self.reach_y[0] <= gate_position[1] <= self.reach_y[1]
        )
        intercept_ready = bool(
            np.linalg.norm(current_estimate[:2] - predicted_position[:2]) <= 0.022
        )
        in_window = in_geometric_window and (
            intercept_ready
            if planned_grasp_pose is not None
            else True
        )
        module_window: Any | None = None
        try:
            module_window = self.grasp_window_calculator.calculate(
                current_estimate,
                estimated_velocity,
                now,
                acceleration=estimated_acceleration,
                object_id=str(object_id),
                planning_time=self.planning_latency,
                motion_time=float(dynamic_arrival),
                approach_time=0.0,
                gripper_time=self.control_latency + self.robot_response_latency,
                confidence=confidence,
                requested_grasp_time=grasp_time,
            )
        except (TypeError, ValueError):
            module_window = None
        feasibility_position = (
            planned_grasp_pose[:3]
            if planned_grasp_pose is not None
            else predicted_position
        )
        feasible = bool(
            history_ready
            and gate_position[0] <= self.grasp_region[1]
            and (
                planned_grasp_time is None
                or control_state in {"CLOSING", "GRASPED"}
                or now
                <= planned_grasp_time
                + _as_float(
                    _value(
                        self.scene_config,
                        "workspace.grasp_deadline_tolerance",
                        default=0.20,
                    ),
                    0.20,
                )
            )
            and (
                planned_grasp_pose is not None
                or grasp_time <= window_end
            )
            and feasibility_position[0] >= self.grasp_region[0]
            and feasibility_position[0] <= self.grasp_region[1]
            and self.reach_y[0] <= feasibility_position[1] <= self.reach_y[1]
            and (
                confidence >= 0.05
                or planned_grasp_pose is not None
            )
        )
        if self.dynamic_intercept_enabled and planned_grasp_pose is None:
            feasible = feasible and self.dynamic_intercept_x_range[0] <= predicted_position[0] <= self.dynamic_intercept_x_range[1]
        prediction_error = None
        if (
            self.perception_mode == "simulator_history"
            and runtime is not None
            and runtime.prediction_at_grasp is not None
            and runtime.actual_grasp_pose is not None
        ):
            prediction_error = float(
                np.linalg.norm(
                    runtime.prediction_at_grasp - runtime.actual_grasp_pose
                )
            )
        result = {
            "object_id": int(object_id),
            "observed_timestamp": float(observed_timestamp),
            "observation_age": age,
            # These are properties of the delayed RGB-D track, not simulator
            # lifecycle/truth.  Policies need to distinguish a fresh mask from
            # a tracker prediction when the wrist occludes the workpiece.
            "visible": bool(observed.get("visible", True)),
            "track_observation_age": max(
                0.0,
                float(observed.get("track_observation_age", 0.0)),
            ),
            "observed_pose": np.r_[observed_position, observed_quaternion],
            "current_estimated_pose": np.r_[current_estimate, observed_quaternion],
            "predicted_grasp_pose": np.r_[predicted_position, observed_quaternion],
            "estimated_velocity": estimated_velocity,
            "estimated_acceleration": estimated_acceleration,
            "estimated_grasp_time": grasp_time,
            "prediction_confidence": confidence,
            "orientation_confidence": float(
                observed.get("orientation_confidence", 0.0)
            ),
            "history_sample_count": len(history_samples),
            "prediction_ready": history_ready,
            "motion_compensation_enabled": self.enable_motion_compensation,
            "prediction_model": (
                None
                if module_prediction is None
                else module_prediction.model.value
            ),
            "prediction_uncertainty": (
                np.zeros(3)
                if module_prediction is None
                else module_prediction.uncertainty.copy()
            ),
            "grasp_window": {
                "start_time": float(window_start),
                "end_time": float(window_end),
                "time_remaining": max(0.0, float(window_end - now)),
                "current_in_window": in_window,
                "is_graspable": in_window,
                "feasible": feasible,
                "reason": (
                    "ok" if module_window is None else module_window.reason
                ),
            },
        }
        if planned_grasp_pose is not None:
            # This is the policy's own previously admitted intercept.  Keeping
            # it alongside the continually corrected visual prediction gives
            # the controller an oracle-free reference during occlusion.
            result["planned_grasp_pose"] = np.asarray(
                planned_grasp_pose, dtype=float
            ).copy()
        if planned_grasp_time is not None:
            result["planned_grasp_time"] = float(planned_grasp_time)
        if truth_qpos is not None:
            result["current_ground_truth_pose"] = truth_qpos
            result["actual_grasp_pose"] = (
                None
                if runtime is None or runtime.actual_grasp_pose is None
                else runtime.actual_grasp_pose.copy()
            )
            result["prediction_error"] = prediction_error
        return result

    def _camera_array_zeros(self, *, depth: bool, resolution: tuple[int, int]) -> np.ndarray:
        width, height = resolution
        if depth:
            return np.zeros((height, width), dtype=np.float32)
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _renderer_for(
        self,
        resolution: tuple[int, int],
        *,
        offscreen_samples: int | None = None,
    ) -> Any | None:
        if self._renderer_failed or not self.render_observations:
            return None
        key = resolution if offscreen_samples is None else (*resolution, offscreen_samples)
        renderer = self._renderers.get(key)
        if renderer is not None:
            return renderer
        width, height = resolution
        original_samples = int(self.model.vis.quality.offsamples)
        try:
            # MjrContext allocates its sample buffers at construction.  Keep
            # the policy RGB context separate from depth/display contexts and
            # restore the live model's visual setting even if creation fails.
            # Like capture itself, construction runs on the simulation owner
            # thread.  No integration or derived-state update is performed.
            if offscreen_samples is not None:
                self.model.vis.quality.offsamples = offscreen_samples
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as error:  # OpenGL availability is platform-dependent.
            warnings.warn(
                f"Offscreen MuJoCo rendering is unavailable; returning zero-valued "
                f"camera arrays ({type(error).__name__}: {error})",
                RuntimeWarning,
                stacklevel=2,
            )
            self._renderer_failed = True
            return None
        finally:
            self.model.vis.quality.offsamples = original_samples
        self._renderers[key] = renderer
        return renderer

    def _render_camera(
        self,
        camera_name: str,
        resolution: tuple[int, int],
        *,
        depth: bool,
    ) -> np.ndarray:
        renderer = self._renderer_for(resolution)
        if renderer is None:
            return self._camera_array_zeros(depth=depth, resolution=resolution)
        renderer.update_scene(self.data, camera=camera_name)
        if depth:
            renderer.enable_depth_rendering()
            frame = renderer.render().copy()
            renderer.disable_depth_rendering()
            return np.asarray(frame, dtype=np.float32)
        renderer.disable_depth_rendering()
        return np.asarray(renderer.render().copy(), dtype=np.uint8)

    def _render_segmentation(
        self, camera_name: str, resolution: tuple[int, int]
    ) -> np.ndarray:
        renderer = self._renderer_for(resolution)
        width, height = resolution
        if renderer is None:
            return np.zeros((height, width, 2), dtype=np.int32)
        renderer.update_scene(self.data, camera=camera_name)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        try:
            frame = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
        return np.asarray(frame, dtype=np.int32)

    def _capture_pose_measurements(
        self,
        *,
        timestamp: float,
        rgb: np.ndarray | None,
        depth: np.ndarray,
        calibration: Mapping[str, Any],
    ) -> None:
        """Run the policy-visible pipeline using depth and capture calibration."""

        detections = self.vision_pipeline.process(
            rgb,
            depth,
            calibration["intrinsics"],
            calibration["camera_to_world"],
            capture_timestamp=float(timestamp),
            # This models a real conveyor encoder/telemetry input.  No object
            # qvel is read by the visual pipeline.
            belt_speed=float(self._current_belt_speed),
            belt_position=float(self._belt_encoder_distance),
        )
        self._associate_visual_tracks_for_evaluation(detections)
        frame_states: dict[str, dict[str, Any]] = {}
        for detection in detections:
            track_id = int(detection.track_id)
            task_state = self._vision_task_state(
                track_id, int(detection.category_id), update_category=False,
            )
            # Visual lifecycle promotion must use visual evidence, not the
            # evaluator's MuJoCo body position.  Two real RGB-D samples are
            # enough because the tracker is initialized with the belt encoder
            # velocity prior; waiting for the physics-side READY_TO_PLAN state
            # delayed selection near a busy-cycle deadline.
            if task_state.state == "NEW":
                task_state.state = "TRACKING"
            elif (
                task_state.state == "TRACKING"
                and bool(detection.visible)
                and float(detection.observation_age) <= self.camera_period + 1e-9
            ):
                task_state.state = "READY_TO_PLAN"
            state = {
                "object_id": track_id,
                "category_id": int(detection.category_id),
                "position": detection.position.copy(),
                "quaternion": detection.quaternion.copy(),
                "linear_velocity": np.array(
                    [self._current_belt_speed, 0.0, 0.0], dtype=float
                ),
                "angular_velocity": np.zeros(3, dtype=float),
                "timestamp": float(detection.timestamp),
                "confidence": float(detection.confidence),
                "category_confidence": float(
                    detection.category_confidence
                ),
                "mask_pixels": int(detection.mask_pixels),
                "valid_depth_pixels": int(detection.valid_depth_pixels),
                "orientation_confidence": float(
                    detection.orientation_confidence
                ),
                "visible": bool(detection.visible),
                "track_observation_age": float(
                    detection.observation_age
                ),
                "feature_vector": detection.feature_vector.copy(),
                "pose_source": "rgbd_depth_components_cad_geometry",
                "state": task_state.state,
            }
            frame_states[str(track_id)] = state
        # Tracker coasting is explicit in ``detections``.  Once its visual age
        # limit expires, mark_missing writes the tombstone without consulting
        # MuJoCo lifecycle state.
        self.perception_buffer.append_frame(
            float(timestamp), frame_states, mark_missing=True
        )

    def _capture_cameras(self, *, force: bool = False) -> None:
        """Capture the bounded-latency control/perception camera stream.

        The deployed baseline extracts connected components and CAD geometry
        exclusively from top-camera depth.  RGB and wrist products therefore
        belong to a separate display stream and must never lengthen or share a
        timestamp implicitly with this control capture.
        """

        if (
            not force
            and self.data.time - self._last_camera_capture
            < self.camera_period - 1e-12
        ):
            return
        top_depth = self._render_camera(
            self.top_camera_name, self.top_resolution, depth=True
        )
        # Never render MuJoCo geom labels on the policy path.  A compatible
        # two-channel array is populated later from visual tracker masks.
        top_segmentation = np.full(
            (*self.top_resolution[::-1], 2), -1, dtype=np.int32
        )
        camera_noise = _value(self.rendering_config, "camera_noise", default={})
        if bool(_value(camera_noise, "enabled", default=False)):
            depth_std = _as_float(_value(camera_noise, "depth_std_m", default=0.001), 0.001)
            top_depth = np.maximum(
                0.0,
                top_depth
                + self._rng.normal(0.0, depth_std, size=top_depth.shape),
            ).astype(np.float32)
        timestamp = float(self.data.time)
        top_calibration = self._camera_calibration(
            self.top_camera_id, self.top_resolution
        )
        if self.perception_mode == "rendered_rgbd":
            self._capture_pose_measurements(
                timestamp=timestamp,
                rgb=None,
                depth=top_depth,
                calibration=top_calibration,
            )
            visual_labels = self.vision_pipeline.last_instance_labels
            if visual_labels is not None:
                top_segmentation[..., 0] = visual_labels
                top_segmentation[..., 1] = np.where(
                    visual_labels >= 0, 0, -1
                )
        self._camera_frames.append(
            {
                "timestamp": timestamp,
                "top_depth": top_depth,
                "top_segmentation": top_segmentation,
                "top_instance_labels": top_segmentation[..., 0].copy(),
                "top_camera": top_calibration,
            }
        )
        self._last_camera_capture = timestamp

    def _capture_display_cameras(
        self,
        *,
        force: bool = False,
        include_wrist: bool = True,
    ) -> dict[str, Any]:
        """Capture optional RGB/wrist display products at their own timestamp.

        This method is never called by the policy observation path.  It is
        intentionally synchronous only at an explicit display/record request;
        callers can therefore distinguish display latency from control-camera
        latency and cannot mistake a stale placeholder for a depth-paired RGB
        image.
        """

        if (
            not force
            and self._display_camera_frames
            and self.data.time - self._last_display_camera_capture
            < self.camera_period - 1e-12
            and (
                not include_wrist
                or self._display_camera_frames[-1]["wrist_rgb"] is not None
            )
        ):
            return self._display_camera_frames[-1]

        timestamp = float(self.data.time)
        top_rgb = self._render_camera(
            self.top_camera_name, self.top_resolution, depth=False
        )
        wrist_rgb: np.ndarray | None = None
        wrist_depth: np.ndarray | None = None
        wrist_calibration: dict[str, Any] | None = None
        if include_wrist:
            wrist_rgb = self._render_camera(
                self.wrist_camera_name, self.wrist_resolution, depth=False
            )
            wrist_depth_enabled = bool(
                _value(
                    self.cameras_config,
                    "wrist.outputs.depth",
                    default=False,
                )
            )
            if wrist_depth_enabled:
                wrist_depth = self._render_camera(
                    self.wrist_camera_name,
                    self.wrist_resolution,
                    depth=True,
                )
            wrist_calibration = self._camera_calibration(
                self.wrist_camera_id, self.wrist_resolution
            )

        camera_noise = _value(self.rendering_config, "camera_noise", default={})
        if bool(_value(camera_noise, "enabled", default=False)):
            rgb_std = _as_float(
                _value(camera_noise, "rgb_std_255", default=1.0), 1.0
            )
            top_rgb = np.clip(
                top_rgb.astype(float)
                + self._rng.normal(0.0, rgb_std, size=top_rgb.shape),
                0,
                255,
            ).astype(np.uint8)
            if wrist_rgb is not None:
                wrist_rgb = np.clip(
                    wrist_rgb.astype(float)
                    + self._rng.normal(0.0, rgb_std, size=wrist_rgb.shape),
                    0,
                    255,
                ).astype(np.uint8)
            if wrist_depth is not None:
                depth_std = _as_float(
                    _value(camera_noise, "depth_std_m", default=0.001),
                    0.001,
                )
                wrist_depth = np.maximum(
                    0.0,
                    wrist_depth
                    + self._rng.normal(0.0, depth_std, size=wrist_depth.shape),
                ).astype(np.float32)

        frame = {
            "top_rgb": top_rgb,
            "top_rgb_timestamp": timestamp,
            "top_camera": self._camera_calibration(
                self.top_camera_id, self.top_resolution
            ),
            "wrist_rgb": wrist_rgb,
            "wrist_rgb_timestamp": timestamp if wrist_rgb is not None else None,
            "wrist_depth": wrist_depth,
            "wrist_depth_timestamp": (
                timestamp if wrist_depth is not None else None
            ),
            "wrist_camera": wrist_calibration,
        }
        self._display_camera_frames.append(frame)
        self._last_display_camera_capture = timestamp
        return frame

    def _latest_display_camera_frame(self) -> dict[str, Any] | None:
        return (
            self._display_camera_frames[-1]
            if self._display_camera_frames
            else None
        )

    def capture_display_observation(
        self,
        *,
        include_wrist: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Return an observation after an explicit RGB/wrist display capture.

        Normal policy observations stay on the bounded top-depth path.  Dataset
        collection or UI code that genuinely needs pixels can call this method;
        the returned RGB/wrist timestamps describe the display snapshot and do
        not overwrite ``camera_timestamp``/``top_depth_timestamp``.
        """

        if self._closed:
            raise RuntimeError("Cannot capture cameras from a closed environment")
        self._capture_display_cameras(
            force=force,
            include_wrist=include_wrist,
        )
        return self.get_observation()

    def capture_policy_cameras(
        self,
        *,
        resolution: tuple[int, int] = (224, 224),
    ) -> dict[str, Any]:
        """Capture fresh external/wrist RGB from one simulation state.

        ``top_rgb`` names the external policy view for dataset compatibility;
        ``camera_names`` identifies its actual MJCF camera. This view includes
        the whole task, whereas the expert's ``top_camera`` watches upstream.
        Both images and their calibration use the current simulation timestamp.
        Call on the simulation owner thread immediately before reading robot
        state and applying the corresponding demonstration action. No stepping,
        perception-buffer updates or cached display frames are involved.

        Rendering must succeed: a disabled/failed renderer raises rather than
        silently recording placeholder images. Simulator debug sites and geoms
        are excluded so target selection and ground truth overlays cannot leak
        into policy inputs. ``resolution`` is (width, height).

        The policy RGB context defaults to a single sample. On the tested
        NVIDIA/MuJoCo stack, multisampled color rendering can vary by one RGB
        level at a fixed complete simulation state. Disabling those buffers
        removes that source of observation variation without rounding pixels
        or caching observations. Set ``rendering.policy_rgb_offscreen_samples``
        to 4 to reproduce the older antialiasing profile; edge pixels differ
        between profiles. Depth/display rendering retains its own profile.
        """

        if self._closed:
            raise RuntimeError("Cannot capture cameras from a closed environment")
        if (
            len(resolution) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
                for value in resolution
            )
        ):
            raise ValueError("policy camera resolution must contain two positive integers")
        if not self.render_observations:
            raise RuntimeError("Policy camera collection requires render_observations=True")
        if self.policy_external_camera_id < 0:
            raise RuntimeError(
                f"Policy camera {self.policy_external_camera_name!r} is missing from the scene"
            )
        resolution = tuple(int(value) for value in resolution)
        samples = _value(self.rendering_config, "policy_rgb_offscreen_samples", default=0)
        if (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or samples < 0
        ):
            raise ValueError("policy_rgb_offscreen_samples must be a nonnegative integer")
        renderer = self._renderer_for(resolution, offscreen_samples=int(samples))
        if renderer is None:
            raise RuntimeError("Policy camera rendering is unavailable; no images were recorded")

        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[:] = 0
        scene_option.geomgroup[4:] = 0
        timestamp = float(self.data.time)
        frame: dict[str, Any] = {"timestamp": timestamp, "camera_names": {}}
        for prefix, camera_name, camera_id in (
            ("top", self.policy_external_camera_name, self.policy_external_camera_id),
            ("wrist", self.wrist_camera_name, self.wrist_camera_id),
        ):
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
            renderer.update_scene(self.data, camera=camera_name, scene_option=scene_option)
            frame[f"{prefix}_rgb"] = np.asarray(renderer.render(), dtype=np.uint8).copy()
            frame[f"{prefix}_rgb_timestamp"] = timestamp
            frame[f"{prefix}_camera"] = self._camera_calibration(camera_id, resolution)
            frame["camera_names"][f"{prefix}_rgb"] = camera_name
        if float(self.data.time) != timestamp:
            raise RuntimeError("Simulation advanced during policy capture; use the owner thread")
        return frame

    def _delayed_camera_frame(self) -> dict[str, Any]:
        if not self._camera_frames:
            self._capture_cameras(force=True)
        target = float(self.data.time) - self.visual_latency
        selected = self._camera_frames[0]
        for frame in self._camera_frames:
            if frame["timestamp"] <= target + 1e-12:
                selected = frame
            else:
                break
        return selected

    def _ee_pose(self) -> np.ndarray:
        position = self.data.site_xpos[self.ee_site_id].copy()
        quaternion = _quat_from_matrix(self.data.site_xmat[self.ee_site_id])
        return np.r_[position, quaternion]

    def _camera_calibration(
        self, camera_id: int, resolution: tuple[int, int]
    ) -> dict[str, Any]:
        width, height = resolution
        fovy = math.radians(float(self.model.cam_fovy[camera_id]))
        focal = 0.5 * height / math.tan(0.5 * fovy)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3)
        extrinsic[:3, 3] = self.data.cam_xpos[camera_id]
        # MuJoCo/OpenGL cameras look along local -Z with +Y up.  BOP and
        # OpenCV use +Z forward with image +Y down.  Publish both transforms
        # so downstream pose estimators cannot accidentally mirror the cloud.
        mujoco_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
        return {
            "intrinsics": np.array(
                [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]],
                dtype=float,
            ),
            "camera_to_world": extrinsic,
            "camera_to_world_opencv": extrinsic @ mujoco_to_opencv,
            "camera_convention": "mujoco_x_right_y_up_minus_z_forward",
            "width": width,
            "height": height,
            "fovy_deg": float(self.model.cam_fovy[camera_id]),
        }

    def _gripper_observation(self) -> dict[str, Any]:
        if self.gripper_controller is not None:
            return dict(self.gripper_controller.get_state())
        positions = self.data.qpos[self.gripper_qpos_adrs].copy()
        ranges = self.model.jnt_range[self.gripper_joint_ids]
        span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-9)
        fraction = float(np.mean((positions - ranges[:, 0]) / span))
        return {
            "opening": float(np.sum(positions)),
            "opening_width": float(np.sum(positions)),
            "open_fraction": fraction,
            "command": self._last_gripper_command,
            "status": "open" if fraction > 0.9 else "closed" if fraction < 0.1 else "moving",
            "is_open": fraction > 0.9,
            "is_closed": fraction < 0.1,
            "left_finger_contact": False,
            "right_finger_contact": False,
            "any_external_finger_contact": False,
            "is_holding": any(
                runtime.state == "GRASPED" for runtime in self._objects
            ),
            "joint_pos": positions,
        }

    def _object_observations(self) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        objects: list[dict[str, Any]] = []
        predictions: dict[int, dict[str, Any]] = {}

        if self.perception_mode == "rendered_rgbd":
            object_ids = [
                int(item)
                for item in self.perception_buffer.object_ids
                if str(item).lstrip("-").isdigit()
            ]
        else:
            object_ids = [
                int(runtime.logical_id)
                for runtime in self._objects
                if runtime.logical_id is not None
            ]

        for object_id in object_ids:
            prediction = self._prediction_record(object_id)
            if self.perception_mode == "rendered_rgbd":
                if prediction is None:
                    continue
                _, observed = self._observed_state_at(
                    object_id, float(prediction["observed_timestamp"])
                )
                if observed is None:
                    continue
                detected_category_id = int(observed["category_id"])
                task_state = self._vision_task_state(
                    object_id, detected_category_id,
                    observed_timestamp=float(prediction["observed_timestamp"]),
                )
                category_id = int(task_state.category_id)
                spec = self._object_spec_by_category.get(category_id)
                if spec is None:
                    continue
                estimated_pose = np.asarray(
                    prediction["current_estimated_pose"], dtype=float
                )
                current = {
                    "object_id": object_id,
                    "category_id": category_id,
                    "category_confidence": float(
                        observed.get("category_confidence", 0.0)
                    ),
                    "bin_id": int(spec.bin_id),
                    "position": estimated_pose[:3].copy(),
                    "quaternion": estimated_pose[3:7].copy(),
                    "linear_velocity": np.asarray(
                        prediction["estimated_velocity"], dtype=float
                    ).copy(),
                    "angular_velocity": np.zeros(3, dtype=float),
                    "timestamp": float(self.data.time),
                    "state": task_state.state,
                    "visible": bool(observed.get("visible", True)),
                    "pose_source": "rgbd_depth_components_cad_geometry",
                }
                metadata = dict(spec.metadata)
            else:
                runtime = self._logical_to_slot.get(object_id)
                if runtime is None:
                    continue
                spec = self._object_spec_by_category.get(runtime.category_id)
                if spec is None:
                    continue
                current = self._snapshot_runtime(runtime)
                current["pose_source"] = "simulator_history"
                metadata = runtime.metadata

            if prediction is not None:
                predictions[object_id] = prediction
                prediction_fields = {
                    "observed_pose": prediction["observed_pose"].copy(),
                    "observed_position": prediction["observed_pose"][:3].copy(),
                    "predicted_grasp_pose": prediction[
                        "predicted_grasp_pose"
                    ].copy(),
                    "predicted_position": prediction[
                        "predicted_grasp_pose"
                    ][:3].copy(),
                    "visible": bool(prediction.get("visible", True)),
                    "track_observation_age": float(
                        prediction.get("track_observation_age", 0.0)
                    ),
                    "estimated_velocity": prediction[
                        "estimated_velocity"
                    ].copy(),
                    "estimated_grasp_time": float(
                        prediction["estimated_grasp_time"]
                    ),
                    "prediction_uncertainty": np.asarray(
                        prediction["prediction_uncertainty"], dtype=float
                    ).copy(),
                    "prediction_confidence": prediction[
                        "prediction_confidence"
                    ],
                    "orientation_confidence": float(
                        prediction.get("orientation_confidence", 0.0)
                    ),
                    "grasp_window": dict(prediction["grasp_window"]),
                    "window_valid": prediction["grasp_window"]["feasible"],
                    "in_grasp_window": prediction["grasp_window"][
                        "current_in_window"
                    ],
                    "grasp_window_start": prediction["grasp_window"][
                        "start_time"
                    ],
                    "grasp_window_end": prediction["grasp_window"]["end_time"],
                }
                if "planned_grasp_pose" in prediction:
                    prediction_fields["planned_grasp_pose"] = prediction[
                        "planned_grasp_pose"
                    ].copy()
                if "planned_grasp_time" in prediction:
                    prediction_fields["planned_grasp_time"] = float(
                        prediction["planned_grasp_time"]
                    )
                if "current_ground_truth_pose" in prediction:
                    prediction_fields["current_ground_truth_pose"] = prediction[
                        "current_ground_truth_pose"
                    ].copy()
                current.update(prediction_fields)

            serialized_spec = spec.as_dict()
            current.update(
                {
                    "object_spec_id": int(spec.category_id),
                    "grasp_height": float(spec.grasp_height),
                    "recommended_grip_width": float(
                        spec.recommended_grip_width
                    ),
                    "grip_width_margin": float(
                        metadata.get("grip_width_margin", 0.040)
                    ),
                    "symmetry": dict(spec.symmetry),
                    "grasp_candidates": serialized_spec["grasp_candidates"],
                    "grasp_yaw_mode": metadata.get(
                        "grasp_yaw_mode", "fixed"
                    ),
                    "grasp_yaw_delta_range": list(
                        metadata.get("grasp_yaw_delta_range", [])
                    ),
                }
            )
            for optional_grasp_parameter in (
                "contact_hold_squeeze_fraction",
                "contact_hold_max_relaxation_fraction",
                "confirmed_grasp_settle_time",
                "closure_requires_arm_settle",
                "close_lead_time",
                "close_settle_time",
                "max_close_time",
                "closure_horizontal_tolerance",
                "closure_vertical_tolerance",
                "closure_orientation_tolerance",
                "allow_predictive_closure_outside_window",
            ):
                if optional_grasp_parameter in metadata:
                    current[optional_grasp_parameter] = metadata[
                        optional_grasp_parameter
                    ]
            objects.append(current)
        return objects, predictions

    def get_observation(self, *, include_camera_images: bool = True) -> dict[str, Any]:
        frame = self._delayed_camera_frame()
        display_frame = self._latest_display_camera_frame()
        if include_camera_images:
            top_rgb = (
                self._camera_array_zeros(depth=False, resolution=self.top_resolution)
                if display_frame is None
                else display_frame["top_rgb"]
            )
            wrist_rgb = None if display_frame is None else display_frame["wrist_rgb"]
            if wrist_rgb is None:
                wrist_rgb = self._camera_array_zeros(depth=False, resolution=self.wrist_resolution)
            wrist_depth = None if display_frame is None else display_frame["wrist_depth"]
            if wrist_depth is None:
                wrist_depth = self._camera_array_zeros(depth=True, resolution=self.wrist_resolution)
            segmentation = frame.get("top_segmentation")
            if segmentation is None:
                segmentation = np.zeros((*self.top_resolution[::-1], 2), dtype=np.int32)
            instance_labels = frame.get("top_instance_labels")
            if instance_labels is None:
                instance_labels = np.full(self.top_resolution[::-1], -1, dtype=np.int32)
            image_observation = dict(
                top_rgb=top_rgb.copy(), top_depth=frame["top_depth"].copy(),
                top_segmentation=segmentation.copy(), top_instance_labels=instance_labels.copy(),
                wrist_rgb=wrist_rgb.copy(), wrist_depth=wrist_depth.copy(),
            )
        else:
            image_observation = {}
        objects, predictions = self._object_observations()
        selected_prediction = (
            None
            if self.selected_target_id is None
            else predictions.get(self.selected_target_id)
        )
        gripper = self._gripper_observation()
        if self.robot_controller is not None:
            commanded_joint_positions = (
                self.robot_controller.commanded_joint_positions
            )
            commanded_joint_velocities = (
                self.robot_controller.commanded_joint_velocities
            )
        else:
            commanded_joint_positions = self._joint_target.copy()
            commanded_joint_velocities = self._joint_target_velocity.copy()
        observation: dict[str, Any] = {
            **image_observation,
            "joint_pos": self.data.qpos[self.arm_qpos_adrs].copy(),
            "joint_vel": self.data.qvel[self.arm_dof_adrs].copy(),
            # The global-path executor must join a new stop-to-stop profile to
            # the target sequence the actuator actually accepted, not merely
            # to a nearby measured state.  These are command-history values,
            # not additional ground-truth perception channels.
            "joint_target": commanded_joint_positions,
            "joint_target_velocity": commanded_joint_velocities,
            "ee_pose": self._ee_pose(),
            "gripper": gripper,
            "gripper_opening": float(gripper.get("opening_width", 0.0)),
            "belt_speed": float(self._current_belt_speed),
            "belt_velocity": np.array([self._current_belt_speed, 0.0, 0.0]),
            "timestamp": float(self.data.time),
            "scene_epoch": int(self._scene_epoch),
            # Backward-compatible name for the policy/perception capture.  It
            # is intentionally not an RGB/wrist timestamp.
            "camera_timestamp": float(frame["timestamp"]),
            "top_depth_timestamp": float(frame["timestamp"]),
            "top_rgb_timestamp": (
                None
                if display_frame is None
                else float(display_frame["top_rgb_timestamp"])
            ),
            "wrist_rgb_timestamp": (
                None
                if display_frame is None
                or display_frame["wrist_rgb_timestamp"] is None
                else float(display_frame["wrist_rgb_timestamp"])
            ),
            "wrist_depth_timestamp": (
                None
                if display_frame is None
                or display_frame["wrist_depth_timestamp"] is None
                else float(display_frame["wrist_depth_timestamp"])
            ),
            "top_rgb_valid": display_frame is not None,
            "wrist_rgb_valid": (
                display_frame is not None
                and display_frame["wrist_rgb"] is not None
            ),
            "wrist_depth_valid": (
                display_frame is not None
                and display_frame["wrist_depth"] is not None
            ),
            "observed_timestamp": (
                None
                if selected_prediction is None
                else float(selected_prediction["observed_timestamp"])
            ),
            "system_latency": {
                "visual": self.visual_latency,
                "planning": self.planning_latency,
                "control": self.control_latency,
                "robot_response": self.robot_response_latency,
                "total": (
                    self.visual_latency
                    + self.planning_latency
                    + self.control_latency
                    + self.robot_response_latency
                ),
            },
            "task_text": self._task_text(),
            "objects": objects,
            "predictions": predictions,
            "selected_target_id": self.selected_target_id,
            "perception": selected_prediction or {},
            "perception_mode": self.perception_mode,
            "perception_source": (
                "rgbd_depth_components_cad_geometry_tracker"
                if self.perception_mode == "rendered_rgbd"
                else "mujoco_simulator_history"
            ),
            "perception_inference_seconds": float(
                self.vision_pipeline.last_inference_seconds
                if self.perception_mode == "rendered_rgbd"
                else 0.0
            ),
            "top_camera": frame.get(
                "top_camera",
                self._camera_calibration(
                    self.top_camera_id, self.top_resolution
                ),
            ),
            "wrist_camera": (
                self._camera_calibration(
                    self.wrist_camera_id, self.wrist_resolution
                )
                if display_frame is None
                or display_frame["wrist_camera"] is None
                else display_frame["wrist_camera"]
            ),
            "bins": [
                {
                    "bin_id": item["id"],
                    "name": item["name"],
                    "position": item["center"].copy(),
                    "size": item["size"].copy(),
                }
                for item in self._bins
            ],
            "severe_collision": self._severe_collision,
            "motion_compensation_enabled": self.enable_motion_compensation,
        }
        if selected_prediction is not None:
            observation.update(
                {
                    key: value
                    for key, value in selected_prediction.items()
                    if key
                    in {
                        "observed_pose",
                        "current_ground_truth_pose",
                        "predicted_grasp_pose",
                        "actual_grasp_pose",
                        "prediction_error",
                    }
                }
            )
        self._last_observation = observation
        return observation

    # ------------------------------------------------------------------
    # Baseline-policy hooks and public state interfaces
    # ------------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adrs].copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_adrs].copy()

    def get_global_planning_snapshot(self) -> dict[str, np.ndarray]:
        """Copy the only live physics input allowed into a plan request."""

        return {
            "full_qpos": np.asarray(self.data.qpos, dtype=float).copy(),
            "arm_qpos": self.get_joint_positions(),
            "scene_epoch": np.asarray(self._scene_epoch, dtype=np.int64),
        }

    def get_future_open_planning_snapshot(
        self, arm_joint_positions: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build an immutable future-state snapshot with fully open fingers.

        Speculative post-release planning must not certify a route with the
        narrower, closed-jaw geometry that exists while the current part is
        still being carried.  This helper changes only a copied qpos vector:
        the live simulator, controllers and target reservation state are never
        touched.  The isolated worker performs its own ``mj_forward`` for every
        sampled state using this open-gripper snapshot.
        """

        arm = np.asarray(arm_joint_positions, dtype=float).reshape(-1)
        if arm.shape != (7,) or not np.all(np.isfinite(arm)):
            raise ValueError("future arm_joint_positions must contain seven values")
        full = np.asarray(self.data.qpos, dtype=float).copy()
        full[self.arm_qpos_adrs] = arm
        if self.gripper_controller is not None:
            opened = np.asarray(
                self.gripper_controller.open_joint_positions, dtype=float
            ).reshape(-1)
        else:
            opened = np.asarray(
                self.model.jnt_range[self.gripper_joint_ids, 1], dtype=float
            ).reshape(-1)
        if opened.shape != self.gripper_qpos_adrs.shape or not np.all(
            np.isfinite(opened)
        ):
            raise ValueError("open gripper calibration is invalid")
        full[self.gripper_qpos_adrs] = opened
        return {
            "full_qpos": full,
            "arm_qpos": arm.copy(),
            "scene_epoch": np.asarray(self._scene_epoch, dtype=np.int64),
        }

    def get_future_carry_planning_snapshot(
        self,
        arm_joint_positions: np.ndarray,
        gripper_action: float,
    ) -> dict[str, np.ndarray]:
        """Copy a canonical carry state without mutating live physics.

        The fixed high-corridor-to-bin leg is searched before the part has
        been grasped.  Its collision geometry must therefore use the exact
        policy-requested jaw opening, not the currently open fingers and not
        simulator object truth.  Dynamic object contacts remain governed by
        the existing planner contract; every command is revalidated against
        live geometry during execution.
        """

        arm = np.asarray(arm_joint_positions, dtype=float).reshape(-1)
        action = float(gripper_action)
        if arm.shape != (7,) or not np.all(np.isfinite(arm)):
            raise ValueError("future arm_joint_positions must contain seven values")
        if not np.isfinite(action):
            raise ValueError("future gripper_action must be finite")
        full = np.asarray(self.data.qpos, dtype=float).copy()
        full[self.arm_qpos_adrs] = arm
        if self.gripper_controller is None:
            closed = np.asarray(
                self.model.jnt_range[self.gripper_joint_ids, 0], dtype=float
            ).reshape(-1)
            opened = np.asarray(
                self.model.jnt_range[self.gripper_joint_ids, 1], dtype=float
            ).reshape(-1)
            fraction = float(np.clip(0.5 * (action + 1.0), 0.0, 1.0))
        else:
            controller = self.gripper_controller
            closed = np.asarray(
                controller.closed_joint_positions, dtype=float
            ).reshape(-1)
            opened = np.asarray(
                controller.open_joint_positions, dtype=float
            ).reshape(-1)
            action_closed = float(controller.config.action_closed)
            action_open = float(controller.config.action_open)
            denominator = action_open - action_closed
            if abs(denominator) <= 1e-12:
                raise ValueError("gripper action calibration is degenerate")
            fraction = float(
                np.clip((action - action_closed) / denominator, 0.0, 1.0)
            )
        joints = closed + fraction * (opened - closed)
        if joints.shape != self.gripper_qpos_adrs.shape or not np.all(
            np.isfinite(joints)
        ):
            raise ValueError("future gripper calibration is invalid")
        full[self.gripper_qpos_adrs] = joints
        return {
            "full_qpos": full,
            "arm_qpos": arm.copy(),
            "scene_epoch": np.asarray(self._scene_epoch, dtype=np.int64),
        }

    def create_global_planning_worker(self) -> Any:
        """Create one isolated-model worker owned by this environment.

        The cloned model and all planning ``MjData`` instances are constructed
        before the control loop starts.  The worker never reads ``self.data``.
        """

        from ..controllers.planning_worker import (
            AsyncPlanningWorker,
            MuJoCoGlobalPlanningEngine,
        )

        engine = MuJoCoGlobalPlanningEngine(
            self.model,
            joint_names=self.arm_joint_names,
            ee_site_name=self.ee_site_name,
            ik_config=_value(self.robot_config, "ik", default={}),
            robot_collision_body_ids=tuple(self._robot_collision_body_ids),
        )
        worker = AsyncPlanningWorker(engine)
        self._planning_workers.append(worker)
        return worker

    def create_prefetch_planning_worker(self) -> Any:
        """Create the independent future-retreat/PREGRASP planning channel."""

        from ..controllers.planning_worker import (
            AsyncPlanningWorker,
            MuJoCoGlobalPlanningEngine,
        )

        engine = MuJoCoGlobalPlanningEngine(
            self.model,
            joint_names=self.arm_joint_names,
            ee_site_name=self.ee_site_name,
            ik_config=_value(self.robot_config, "ik", default={}),
            robot_collision_body_ids=tuple(self._robot_collision_body_ids),
        )
        worker = AsyncPlanningWorker(engine, name="nero-prefetch-planner")
        self._planning_workers.append(worker)
        return worker

    def is_arm_configuration_safe(
        self,
        joint_positions: np.ndarray,
        *,
        approach_axis_local: np.ndarray = np.array([0.0, 0.0, 1.0]),
        minimum_downward_alignment: float = 0.0,
        minimum_tcp_height: float = 0.0,
        minimum_joint_margin: float = 0.0,
    ) -> bool:
        """Validate a sampled transfer state without mutating live physics."""

        positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        axis = np.asarray(approach_axis_local, dtype=float).reshape(-1).copy()
        if (
            positions.shape != (7,)
            or not np.all(np.isfinite(positions))
            or axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) < 1e-9
        ):
            return False
        axis /= np.linalg.norm(axis)
        scratch = self._path_validation_data
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.arm_qpos_adrs] = positions
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, scratch)
        return self._arm_state_data_is_safe(
            positions,
            scratch,
            approach_axis_local=axis,
            minimum_downward_alignment=minimum_downward_alignment,
            minimum_tcp_height=minimum_tcp_height,
            minimum_joint_margin=minimum_joint_margin,
            maximum_jacobian_condition=None,
        )

    def _arm_state_data_is_safe(
        self,
        positions: np.ndarray,
        state_data: mujoco.MjData,
        *,
        approach_axis_local: np.ndarray,
        minimum_downward_alignment: float,
        minimum_tcp_height: float,
        minimum_joint_margin: float,
        maximum_jacobian_condition: float | None,
    ) -> bool:
        """Check all state constraints from one already-forwarded ``MjData``."""

        ranges = self.model.jnt_range[self.arm_joint_ids]
        margins = np.minimum(
            positions - ranges[:, 0],
            ranges[:, 1] - positions,
        )
        if np.min(margins) < float(minimum_joint_margin) - 1e-9:
            return False

        if float(state_data.site_xpos[self.ee_site_id, 2]) < float(
            minimum_tcp_height
        ) - 1e-9:
            return False
        world_axis = (
            state_data.site_xmat[self.ee_site_id].reshape(3, 3)
            @ approach_axis_local
        )
        if -float(world_axis[2]) < float(minimum_downward_alignment) - 1e-9:
            return False

        if maximum_jacobian_condition is not None:
            condition_limit = float(maximum_jacobian_condition)
            if not np.isfinite(condition_limit) or condition_limit <= 0.0:
                return False
            mujoco.mj_jacSite(
                self.model,
                state_data,
                self._path_validation_jacobian_position,
                self._path_validation_jacobian_rotation,
                self.ee_site_id,
            )
            dofs = self.ik.dof_indices
            jacobian = np.vstack(
                (
                    self._path_validation_jacobian_position[:, dofs],
                    self._path_validation_jacobian_rotation[:, dofs],
                )
            )
            jacobian[3:] *= 0.30
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            condition = (
                float("inf")
                if singular_values[-1] <= 1e-12
                else float(singular_values[0] / singular_values[-1])
            )
            if condition > condition_limit:
                return False

        for contact in state_data.contact[: state_data.ncon]:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            robot1 = body1 in self._robot_collision_body_ids
            robot2 = body2 in self._robot_collision_body_ids
            if not (robot1 or robot2):
                continue
            other_body = body2 if robot1 else body1
            other_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    other_body,
                )
                or ""
            )
            # The carried/target workpiece is intentionally in contact with
            # the fingers.  Its pose is not rigidly propagated in this scratch
            # model, so object contacts are excluded from arm-path validity.
            if other_name.startswith("object_"):
                continue
            if float(contact.dist) <= 1e-6:
                return False
        return True

    def is_arm_edge_safe(
        self,
        start_joint_positions: np.ndarray,
        end_joint_positions: np.ndarray,
        *,
        normalized_resolution: float,
        maximum_joint_step: float | None = None,
        approach_axis_local: np.ndarray = np.array([0.0, 0.0, 1.0]),
        minimum_downward_alignment: float = 0.0,
        minimum_tcp_height: float = 0.0,
        minimum_joint_margin: float = 0.0,
        maximum_jacobian_condition: float,
    ) -> bool:
        """Densely validate a live joint edge with one forward pass per sample.

        This is the production fast path used by the 50 Hz policy.  It keeps
        collision, tool orientation/height, singularity and soft-limit checks
        coupled to the same sampled state.  The measured first sample reuses
        the already-forwarded live ``MjData`` when it exactly matches the
        supplied start; every predicted sample uses the isolated scratch data.
        """

        start = np.asarray(start_joint_positions, dtype=float).reshape(-1)
        end = np.asarray(end_joint_positions, dtype=float).reshape(-1)
        axis = np.asarray(approach_axis_local, dtype=float).reshape(-1).copy()
        try:
            resolution = float(normalized_resolution)
            joint_step = (
                None
                if maximum_joint_step is None
                else float(maximum_joint_step)
            )
        except (TypeError, ValueError):
            return False
        if (
            start.shape != (7,)
            or end.shape != (7,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            or axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) < 1e-9
            or not np.isfinite(resolution)
            or resolution <= 0.0
            or (
                joint_step is not None
                and (not np.isfinite(joint_step) or joint_step <= 0.0)
            )
        ):
            return False
        axis /= np.linalg.norm(axis)
        ranges = self.model.jnt_range[self.arm_joint_ids]
        span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
        normalized_distance = float(np.linalg.norm((end - start) / span))
        normalized_samples = (
            0
            if normalized_distance <= 1e-12
            else max(
                1,
                int(math.ceil(normalized_distance / resolution)),
            )
        )
        joint_samples = (
            0
            if joint_step is None
            else int(
                math.ceil(
                    float(np.max(np.abs(end - start))) / joint_step
                )
            )
        )
        samples = max(normalized_samples, joint_samples)

        live_positions = self.get_joint_positions()
        reuse_live_start = bool(
            np.max(np.abs(start - live_positions)) <= 1e-9
        )
        scratch = self._path_validation_data
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        delta = end - start
        for index in range(samples + 1):
            fraction = 0.0 if samples == 0 else index / samples
            positions = start + delta * fraction
            if index == 0 and reuse_live_start:
                state_data = self.data
            else:
                scratch.qpos[self.arm_qpos_adrs] = positions
                mujoco.mj_forward(self.model, scratch)
                state_data = scratch
            if not self._arm_state_data_is_safe(
                positions,
                state_data,
                approach_axis_local=axis,
                minimum_downward_alignment=minimum_downward_alignment,
                minimum_tcp_height=minimum_tcp_height,
                minimum_joint_margin=minimum_joint_margin,
                maximum_jacobian_condition=maximum_jacobian_condition,
            ):
                return False
        return True

    def set_motion_profile(self, profile: str) -> None:
        """Select nominal tracking or gentler object-carrying joint limits."""

        normalized = str(profile).lower()
        if normalized not in {"nominal", "carrying"}:
            raise ValueError("motion profile must be 'nominal' or 'carrying'")
        if normalized == self._motion_profile:
            return
        trajectory = _value(self.robot_config, "trajectory", default={})
        velocity_scale = (
            _as_float(
                _value(
                    trajectory,
                    "carrying_velocity_scale",
                    default=0.65,
                ),
                0.65,
            )
            if normalized == "carrying"
            else 1.0
        )
        acceleration_scale = (
            _as_float(
                _value(
                    trajectory,
                    "carrying_acceleration_scale",
                    default=0.50,
                ),
                0.50,
            )
            if normalized == "carrying"
            else 1.0
        )
        self.max_joint_velocity = (
            self._nominal_max_joint_velocity * velocity_scale
        )
        self.max_joint_acceleration = (
            self._nominal_max_joint_acceleration * acceleration_scale
        )
        if self.robot_controller is not None:
            self.robot_controller.max_joint_velocity = (
                self.max_joint_velocity.copy()
            )
            self.robot_controller.max_joint_acceleration = (
                self.max_joint_acceleration.copy()
            )
            if hasattr(self.robot_controller, "_last_target_velocity"):
                self.robot_controller._last_target_velocity = np.clip(
                    self.robot_controller._last_target_velocity,
                    -self.max_joint_velocity,
                    self.max_joint_velocity,
                )
        self._motion_profile = normalized

    def get_robot_state(self) -> dict[str, Any]:
        if self.robot_controller is not None:
            return dict(self.robot_controller.get_state())
        return {
            "joint_pos": self.get_joint_positions(),
            "joint_vel": self.get_joint_velocities(),
            "ee_pose": self._ee_pose(),
        }

    get_state = get_robot_state

    def solve_ik_result(
        self,
        position: np.ndarray,
        quaternion: np.ndarray | None = None,
        qpos_seed: np.ndarray | None = None,
        *,
        allow_multiseed: bool = True,
    ) -> Any:
        """Return the converged IK branch nearest a continuity reference.

        Multiple deterministic seeds improve reachability, but unlike the old
        bin-specific shortcut they do not override continuity.  Successful
        branches are ranked first by normalized distance from ``qpos_seed``
        (or the current arm state), then by Cartesian residual.
        """

        if quaternion is None:
            pose = np.asarray(position, dtype=float).reshape(-1)
            if pose.size != 7:
                raise ValueError("A single IK pose argument must contain 7 values")
            position, quaternion = pose[:3], pose[3:]
        target_position = np.asarray(position, dtype=float)
        target_quaternion = np.asarray(quaternion, dtype=float)
        current = self.get_joint_positions()
        reference = (
            current
            if qpos_seed is None
            else np.asarray(qpos_seed, dtype=float).reshape(-1)
        )
        if reference.shape != (7,) or not np.all(np.isfinite(reference)):
            raise ValueError("qpos_seed must contain seven finite joint values")
        joint_ranges = self.model.jnt_range[self.arm_joint_ids]
        reference = np.clip(reference, joint_ranges[:, 0], joint_ranges[:, 1])
        seeds = [reference]
        if not np.allclose(current, reference):
            seeds.append(current)
        for key in ("safe_joint_positions", "home_joint_positions"):
            configured = _value(self.robot_config, key, default=None)
            if configured is not None:
                seed = np.asarray(configured, dtype=float)
                if seed.shape == (7,) and not any(
                    np.allclose(seed, existing) for existing in seeds
                ):
                    seeds.append(seed)
        calibration_poses = _value(
            self.robot_config,
            "calibration_poses",
            default={},
        )
        if isinstance(calibration_poses, Mapping):
            for calibration in calibration_poses.values():
                if not isinstance(calibration, Mapping):
                    continue
                configured = calibration.get("joint_positions")
                if configured is None:
                    continue
                seed = np.asarray(configured, dtype=float)
                if seed.shape == (7,) and not any(
                    np.allclose(seed, existing) for existing in seeds
                ):
                    seeds.append(seed)
        reference_result = self.ik.solve(
            target_position,
            target_quaternion,
            qpos_seed=reference,
        )
        if not allow_multiseed:
            # The 50 Hz Cartesian servo must never expand one failed warm
            # solve into safe/home/calibration multi-seed IK on the caller's
            # thread.  Global branches are searched by the isolated planning
            # worker; the local loop holds or tries a smaller Cartesian step.
            return reference_result
        trajectory_config = _value(
            self.robot_config,
            "trajectory",
            default={},
        )
        minimum_margin = _as_float(
            _value(
                trajectory_config,
                "minimum_joint_limit_margin",
                default=0.0,
            ),
            0.0,
        )

        def joint_margin(item: Any) -> float:
            positions = np.asarray(item.joint_positions, dtype=float)
            return float(
                np.min(
                    np.minimum(
                        positions - joint_ranges[:, 0],
                        joint_ranges[:, 1] - positions,
                    )
                )
            )

        # A converged solution from the current configuration is already the
        # continuous branch by construction, provided it also stays outside
        # the configured soft-limit buffer.  A merely converged solution at a
        # wrist hard stop is not safe enough to execute.
        if (
            reference_result.success
            and joint_margin(reference_result) >= minimum_margin
        ):
            return reference_result
        results = [reference_result]
        results.extend(
            self.ik.solve(
                target_position,
                target_quaternion,
                qpos_seed=seed,
            )
            for seed in seeds[1:]
        )
        span = np.maximum(joint_ranges[:, 1] - joint_ranges[:, 0], 1e-6)
        successful = [item for item in results if item.success]
        safe_successful = [
            item for item in successful if joint_margin(item) >= minimum_margin
        ]
        candidates = safe_successful or successful or results
        if successful:
            result = min(
                candidates,
                key=lambda item: (
                    float(
                        np.linalg.norm(
                            (item.joint_positions - reference) / span
                        )
                    ),
                    -joint_margin(item),
                    item.position_error + 0.15 * item.orientation_error,
                ),
            )
        else:
            # With no converged branch, report the genuinely closest failed
            # solution instead of a continuous but geometrically unrelated one.
            result = min(
                candidates,
                key=lambda item: (
                    item.position_error + 0.15 * item.orientation_error,
                    float(
                        np.linalg.norm(
                            (item.joint_positions - reference) / span
                        )
                    ),
                ),
            )
        return result

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion: np.ndarray | None = None,
        qpos_seed: np.ndarray | None = None,
    ) -> np.ndarray:
        result = self.solve_ik_result(position, quaternion, qpos_seed)
        return result.joint_positions.copy()

    inverse_kinematics = solve_ik

    def get_object_state(self, object_id: int) -> dict[str, Any] | None:
        runtime = self._runtime_for_policy_object(int(object_id))
        if self.perception_mode == "rendered_rgbd":
            task_state = self._vision_task_states.get(int(object_id))
            # A terminal physical lifecycle fact is immutable and contains no
            # pose truth.  Preserve that state long enough for the policy to
            # prove that a released object is already sorted even after its
            # RGB-D track/prediction has correctly disappeared.  Non-terminal
            # planning data remains strictly rendered-RGB-D-only below.
            #
            # The evaluation-only track<->runtime association is deliberately
            # removed on the first camera update after a terminal transition,
            # while VisionTaskState remains keyed by the policy's track ID.
            # Read that persistent policy-visible lifecycle first; otherwise a
            # SORTED event occurring on the release/return boundary disappears
            # before the short open-jaw handoff can prove its terminal gate.
            if (
                task_state is not None
                and task_state.state in TERMINAL_OBJECT_STATES
            ):
                category_id = int(task_state.category_id)
                spec = self._object_spec_by_category.get(category_id)
                if spec is None:
                    return None
                return {
                    "object_id": int(object_id),
                    "category_id": category_id,
                    "bin_id": int(spec.bin_id),
                    "timestamp": float(self.data.time),
                    "state": str(task_state.state),
                    "failure_reason": task_state.failure_reason,
                    "pose_source": "terminal_lifecycle_only",
                }
            if runtime is not None and runtime.state in TERMINAL_OBJECT_STATES:
                return {
                    "object_id": int(object_id),
                    "category_id": int(runtime.category_id),
                    "bin_id": int(runtime.bin_id),
                    "timestamp": float(self.data.time),
                    "state": str(runtime.state),
                    "failure_reason": runtime.failure_reason,
                    "pose_source": "terminal_lifecycle_only",
                }
            prediction = self._prediction_record(int(object_id))
            if prediction is None:
                return None
            _, observed = self._observed_state_at(
                int(object_id), float(prediction["observed_timestamp"])
            )
            if observed is None:
                return None
            detected_category_id = int(observed["category_id"])
            task_state = self._vision_task_state(
                int(object_id), detected_category_id,
                observed_timestamp=float(prediction["observed_timestamp"]),
            )
            category_id = int(task_state.category_id)
            spec = self._object_spec_by_category.get(category_id)
            if spec is None:
                return None
            current_pose = np.asarray(
                prediction["current_estimated_pose"], dtype=float
            )
            result = {
                "object_id": int(object_id),
                "category_id": category_id,
                "category_confidence": float(
                    observed.get("category_confidence", 0.0)
                ),
                "bin_id": int(spec.bin_id),
                "position": current_pose[:3].copy(),
                "quaternion": current_pose[3:7].copy(),
                "linear_velocity": np.asarray(
                    prediction["estimated_velocity"], dtype=float
                ).copy(),
                "angular_velocity": np.zeros(3, dtype=float),
                "timestamp": float(self.data.time),
                "state": task_state.state,
                "pose_source": "rgbd_depth_components_cad_geometry",
            }
        else:
            prediction = self._prediction_record(int(object_id))
            if runtime is None:
                return None
            result = self._snapshot_runtime(runtime)
            result["pose_source"] = "simulator_history"
        if prediction is not None:
            result.update(prediction)
            result["window_valid"] = prediction["grasp_window"]["feasible"]
            result["in_grasp_window"] = prediction["grasp_window"]["current_in_window"]
        return result

    def select_target(
        self, observation: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | int | None:
        obs = self.get_observation() if observation is None else observation
        objects = list(obs.get("objects", []))
        if self.selected_target_id is not None:
            for item in objects:
                if item.get("object_id") == self.selected_target_id and item.get(
                    "state"
                ) not in TERMINAL_OBJECT_STATES:
                    return item
            self.selected_target_id = None
        candidates = [
            item
            for item in objects
            if item.get("state") in {"NEW", "TRACKING", "READY_TO_PLAN"}
            and bool(item.get("window_valid", False))
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                float(item.get("grasp_window_end", np.inf)),
                -float(np.asarray(item["position"])[0]),
                int(item["object_id"]),
            )
        )
        selected = candidates[0]
        self.selected_target_id = int(selected["object_id"])
        runtime = self._runtime_for_policy_object(self.selected_target_id)
        category_id = int(selected["category_id"])
        planned_pose = selected.get("predicted_grasp_pose")
        prediction = self._prediction_from_observation(
            self.selected_target_id, obs
        )
        if self.perception_mode == "rendered_rgbd":
            task_state = self._vision_task_state(
                self.selected_target_id, category_id
            )
            eligible = task_state.state in {
                "NEW",
                "TRACKING",
                "READY_TO_PLAN",
            }
            if eligible:
                task_state.selected_at = float(self.data.time)
                task_state.compensation_enabled_at_selection = bool(
                    self.enable_motion_compensation
                )
                task_state.planned_grasp_pose = (
                    None
                    if planned_pose is None
                    else np.asarray(planned_pose, dtype=float).copy()
                )
                task_state.planned_grasp_time = (
                    None
                    if prediction is None
                    else float(
                        prediction.get(
                            "estimated_grasp_time", self.data.time
                        )
                    )
                )
                task_state.state = "PLANNING"
        else:
            eligible = bool(
                runtime is not None
                and runtime.state in {"NEW", "TRACKING", "READY_TO_PLAN"}
            )

        if eligible:
            if runtime is not None:
                runtime.selected_at = float(self.data.time)
                runtime.compensation_enabled_at_selection = bool(
                    self.enable_motion_compensation
                )
                runtime.planned_grasp_pose = (
                    None
                    if planned_pose is None
                    else np.asarray(planned_pose, dtype=float).copy()
                )
                runtime.planned_grasp_time = (
                    None
                    if prediction is None
                    else float(
                        prediction.get(
                            "estimated_grasp_time", self.data.time
                        )
                    )
                )
                if runtime.logical_id is not None:
                    self.set_object_state(runtime.logical_id, "PLANNING")
            self.sorting_task.create_assignment(
                category_id,
                object_id=str(self.selected_target_id),
                include_skip_rule=True,
            )
            selected = dict(selected)
            selected["state"] = "PLANNING"
        return selected

    select_sorting_target = select_target

    def release_selected_target(self, object_id: int) -> bool:
        """Undo a planning reservation without declaring a workpiece failed.

        Deadline admission may reject a safe path solely because the arm is
        still returning from the previous bin.  The object remains visually
        tracked on the belt and may become feasible from the wait pose, so a
        pre-motion rejection must not be counted as a terminal skip.
        """

        track_id = int(object_id)
        if self.selected_target_id != track_id:
            return False
        runtime = self._runtime_for_policy_object(track_id)
        task_state = (
            self._vision_task_states.get(track_id)
            if self.perception_mode == "rendered_rgbd"
            else None
        )
        state = (
            task_state.state
            if task_state is not None
            else None if runtime is None else runtime.state
        )
        if state != "PLANNING":
            return False
        if task_state is not None:
            task_state.state = "READY_TO_PLAN"
            task_state.selected_at = None
            task_state.planned_grasp_pose = None
            task_state.planned_grasp_time = None
            task_state.failure_reason = None
        if runtime is not None and runtime.logical_id is not None:
            runtime.selected_at = None
            runtime.planned_grasp_pose = None
            runtime.planned_grasp_time = None
            self.set_object_state(int(runtime.logical_id), "READY_TO_PLAN")
        self.selected_target_id = None
        self.sorting_task.clear_target()
        return True

    release_target_reservation = release_selected_target

    def refine_selected_grasp_plan(
        self,
        object_id: int,
        required_motion_time: float,
        deadline_margin: float = 0.25,
        *,
        allow_retime: bool = True,
        final_tracking_correction_limit: float | None = None,
        requested_time: float | None = None,
        requested_pose: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Retime one reserved intercept from the robot's executable path ETA.

        The controller deliberately supplies only a duration.  Position and
        velocity come from the policy-visible perception record (RGB-D track
        in strict mode), so this hook cannot turn route timing into an object
        qpos/qvel backdoor.  The selected task/runtime state is updated
        atomically only after the refined intercept passes the hard window.
        """

        try:
            track_id = int(object_id)
            motion_time = float(required_motion_time)
            reserve = float(deadline_margin)
        except (TypeError, ValueError):
            return {"accepted": False, "reason": "invalid_intercept_timing"}
        if (
            track_id != self.selected_target_id
            or not np.isfinite(motion_time)
            or motion_time < 0.0
            or not np.isfinite(reserve)
            or reserve < 0.0
        ):
            return {"accepted": False, "reason": "invalid_intercept_timing"}

        runtime = self._runtime_for_policy_object(track_id)
        task_state = (
            self._vision_task_states.get(track_id)
            if self.perception_mode == "rendered_rgbd"
            else None
        )
        control_state = (
            task_state.state
            if task_state is not None
            else None if runtime is None else runtime.state
        )
        if control_state != "PLANNING":
            return {"accepted": False, "reason": "target_not_in_planning"}

        prediction = self._prediction_record(track_id)
        if prediction is None:
            return {"accepted": False, "reason": "prediction_unavailable"}
        now = float(self.data.time)
        current_pose = np.asarray(
            prediction["current_estimated_pose"], dtype=float
        ).copy()
        velocity = np.asarray(
            prediction["estimated_velocity"], dtype=float
        ).copy()
        if (
            current_pose.shape != (7,)
            or velocity.shape != (3,)
            or not np.all(np.isfinite(current_pose))
            or not np.all(np.isfinite(velocity))
        ):
            return {"accepted": False, "reason": "invalid_prediction"}

        earliest_time = now + motion_time
        preferred_x = _as_float(
            _value(
                self.scene_config,
                "workspace.preferred_grasp_x",
                default=0.08,
            ),
            0.08,
        )
        speed_x = float(velocity[0])
        if not self.dynamic_intercept_enabled and speed_x > 1e-5 and preferred_x > current_pose[0]:
            earliest_time = max(
                earliest_time,
                now + (preferred_x - current_pose[0]) / speed_x,
            )

        window = prediction.get("grasp_window", {})
        try:
            window_end = float(window["end_time"])
        except (KeyError, TypeError, ValueError):
            return {"accepted": False, "reason": "grasp_window_unavailable"}
        latest_intercept = window_end - reserve
        if earliest_time > latest_intercept + 1e-9:
            return {
                "accepted": False,
                "reason": "insufficient_intercept_time",
                "required_grasp_time": earliest_time,
                "latest_grasp_time": latest_intercept,
                "window_end": window_end,
            }

        # Independent prefetch B already certified a high Cartesian endpoint
        # and a continuation at one explicit policy-visible visual clock.
        # Bind that clock/body XY atomically instead of silently preserving
        # select_target()'s nominal preferred-time pose.  The requested body
        # pose uses the pregrasp XY but its z/quaternion are not authoritative;
        # support-plane z and the latest policy-visible object orientation are
        # retained here.  All workspace/deadline checks remain fail-closed.
        if requested_time is not None or requested_pose is not None:
            try:
                exact_time = float(requested_time)
                exact_pose = np.asarray(requested_pose, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return {
                    "accepted": False,
                    "reason": "invalid_requested_intercept",
                }
            if (
                exact_pose.shape != (7,)
                or not np.all(np.isfinite(exact_pose))
                or not np.isfinite(exact_time)
                or exact_time < earliest_time - 1e-9
                or exact_time > latest_intercept + 1e-9
            ):
                return {
                    "accepted": False,
                    "reason": "insufficient_intercept_time",
                    "required_grasp_time": earliest_time,
                    "latest_grasp_time": latest_intercept,
                    "window_end": window_end,
                }
            bound_pose = current_pose.copy()
            bound_pose[:2] = exact_pose[:2]
            spec = self._spec_for_policy_object(track_id)
            metadata = {} if spec is None else dict(spec.metadata)
            target_y_range = np.asarray(
                metadata.get("grasp_target_y_range", self.reach_y),
                dtype=float,
            ).reshape(-1)
            if (
                target_y_range.shape == (2,)
                and np.all(np.isfinite(target_y_range))
            ):
                bound_pose[1] = float(
                    np.clip(
                        bound_pose[1],
                        target_y_range[0],
                        target_y_range[1],
                    )
                )
            if not (
                self.grasp_region[0] <= bound_pose[0] <= self.grasp_region[1]
                and self.reach_y[0] <= bound_pose[1] <= self.reach_y[1]
            ):
                return {
                    "accepted": False,
                    "reason": "trajectory_outside_reachable_workspace",
                }
            if task_state is not None:
                task_state.planned_grasp_time = exact_time
                task_state.planned_grasp_pose = bound_pose.copy()
            if runtime is not None:
                runtime.planned_grasp_time = exact_time
                runtime.planned_grasp_pose = bound_pose.copy()
            return {
                "accepted": True,
                "reason": "ok",
                "changed": True,
                "estimated_grasp_time": exact_time,
                "predicted_grasp_pose": bound_pose.copy(),
                "window_end": window_end,
                "deadline_slack": float(latest_intercept - exact_time),
            }

        old_time = (
            task_state.planned_grasp_time
            if task_state is not None
            else None if runtime is None else runtime.planned_grasp_time
        )
        try:
            previous_time = float(old_time)
        except (TypeError, ValueError):
            previous_time = float("nan")
        previous_pose = (
            task_state.planned_grasp_pose
            if task_state is not None
            else None if runtime is None else runtime.planned_grasp_pose
        )
        if not allow_retime:
            accepted = bool(
                np.isfinite(previous_time)
                and earliest_time <= previous_time + 1e-9
                and previous_time <= latest_intercept + 1e-9
                and previous_pose is not None
            )
            # A background route can finish a few control ticks later than
            # the first fixed-point estimate while still retaining positive
            # hard-deadline slack.  Rejecting that route and immediately
            # reserving the same workpiece again caused a planning storm until
            # the real window expired.  Permit one *bounded* monotonic clock
            # correction only when the corresponding belt-plane displacement
            # fits inside the policy's calibrated vertical pregrasp corridor.
            # The delivered global path still terminates at its originally
            # validated high waypoint; the correction is consumed over one or
            # more bounded per-tick Cartesian updates, each of which repeats
            # IK, edge, collision, cone, singularity and soft-limit checks.
            bounded_pose: np.ndarray | None = None
            correction = float("inf")
            try:
                correction_limit = float(final_tracking_correction_limit)
            except (TypeError, ValueError):
                correction_limit = 0.0
            if (
                not accepted
                and previous_pose is not None
                and np.isfinite(previous_time)
                and previous_time < earliest_time
                and earliest_time <= latest_intercept + 1e-9
                and np.isfinite(correction_limit)
                and correction_limit > 0.0
            ):
                bounded_pose = current_pose.copy()
                bounded_pose[:3] += velocity * max(0.0, earliest_time - now)
                if control_state in BELT_OBJECT_STATES:
                    bounded_pose[2] = current_pose[2]
                previous_array = np.asarray(previous_pose, dtype=float).reshape(-1)
                if previous_array.shape == (7,) and np.all(
                    np.isfinite(previous_array)
                ):
                    correction = 0.0
            if bounded_pose is not None:
                spec = self._spec_for_policy_object(track_id)
                metadata = {} if spec is None else dict(spec.metadata)
                target_y_range = np.asarray(
                    metadata.get("grasp_target_y_range", self.reach_y),
                    dtype=float,
                ).reshape(-1)
                if (
                    target_y_range.shape == (2,)
                    and np.all(np.isfinite(target_y_range))
                ):
                    bounded_pose[1] = float(
                        np.clip(
                            bounded_pose[1],
                            target_y_range[0],
                            target_y_range[1],
                        )
                    )
                accepted = bool(
                    self.grasp_region[0]
                    <= bounded_pose[0]
                    <= self.grasp_region[1]
                    and self.reach_y[0]
                    <= bounded_pose[1]
                    <= self.reach_y[1]
                )
                if accepted:
                    previous_array = np.asarray(
                        previous_pose, dtype=float
                    ).reshape(-1)
                    correction = float(
                        np.linalg.norm(
                            bounded_pose[:3] - previous_array[:3]
                        )
                    )
                    accepted = bool(
                        correction <= correction_limit + 1e-9
                    )
            if accepted and bounded_pose is not None:
                if task_state is not None:
                    task_state.planned_grasp_time = float(earliest_time)
                    task_state.planned_grasp_pose = bounded_pose.copy()
                if runtime is not None:
                    runtime.planned_grasp_time = float(earliest_time)
                    runtime.planned_grasp_pose = bounded_pose.copy()
                previous_time = float(earliest_time)
                previous_pose = bounded_pose.copy()
            return {
                "accepted": accepted,
                "reason": "ok" if accepted else "insufficient_intercept_time",
                "required_grasp_time": earliest_time,
                "estimated_grasp_time": previous_time,
                "latest_grasp_time": latest_intercept,
                "window_end": window_end,
                "predicted_grasp_pose": (
                    None
                    if previous_pose is None
                    else np.asarray(previous_pose, dtype=float).copy()
                ),
                "tracking_correction": (
                    0.0 if bounded_pose is None else float(correction)
                ),
            }
        # Preserve an already executable intercept.  Re-solving every normal
        # target at the earliest possible instant changed validated IK
        # branches for no scheduling benefit.  Only a route-infeasible or
        # low-deadline-margin plan is moved; the latter is pulled upstream to
        # the earliest executable time instead of merely shaving the margin.
        keep_previous = bool(
            np.isfinite(previous_time)
            and earliest_time - 1e-9 <= previous_time
            and previous_time <= latest_intercept + 1e-9
        )
        requested_time = previous_time if keep_previous else earliest_time

        remaining = max(0.0, requested_time - now)
        if keep_previous and previous_pose is not None and not self.dynamic_intercept_enabled:
            refined_pose = np.asarray(previous_pose, dtype=float).copy()
        else:
            refined_pose = current_pose.copy()
            refined_pose[:3] += velocity * remaining
        # Supported workpieces remain on the measured support plane.  A noisy
        # vertical velocity must not be extrapolated through a multi-second
        # route estimate.
        if control_state in BELT_OBJECT_STATES:
            refined_pose[2] = current_pose[2]
        spec = self._spec_for_policy_object(track_id)
        metadata = {} if spec is None else dict(spec.metadata)
        target_y_range = np.asarray(
            metadata.get("grasp_target_y_range", self.reach_y), dtype=float
        ).reshape(-1)
        if target_y_range.shape == (2,) and np.all(np.isfinite(target_y_range)):
            refined_pose[1] = float(
                np.clip(refined_pose[1], target_y_range[0], target_y_range[1])
            )
        if not (
            self.grasp_region[0] <= refined_pose[0] <= self.grasp_region[1]
            and self.reach_y[0] <= refined_pose[1] <= self.reach_y[1]
        ):
            return {
                "accepted": False,
                "reason": "trajectory_outside_reachable_workspace",
            }

        if task_state is not None:
            task_state.planned_grasp_time = float(requested_time)
            task_state.planned_grasp_pose = refined_pose.copy()
        # Runtime is a private lifecycle/evaluation mirror.  It is never used
        # above to calculate the refined pose in strict RGB-D mode.
        if runtime is not None:
            runtime.planned_grasp_time = float(requested_time)
            runtime.planned_grasp_pose = refined_pose.copy()
        return {
            "accepted": True,
            "reason": "ok",
            "changed": bool(
                old_time is None
                or abs(float(old_time) - requested_time) > 1e-9
            ),
            "estimated_grasp_time": float(requested_time),
            "predicted_grasp_pose": refined_pose.copy(),
            "window_end": window_end,
            "deadline_slack": float(latest_intercept - requested_time),
        }

    retime_selected_grasp_plan = refine_selected_grasp_plan

    def validate_selected_grasp_plan(
        self,
        object_id: int,
        required_motion_time: float,
        deadline_margin: float = 0.25,
        final_tracking_correction_limit: float = 0.012,
    ) -> dict[str, Any]:
        """Validate the final route with only a bounded servo correction."""

        return self.refine_selected_grasp_plan(
            object_id,
            required_motion_time,
            deadline_margin,
            allow_retime=False,
            final_tracking_correction_limit=final_tracking_correction_limit,
        )

    def get_predicted_grasp_pose(
        self,
        object_id: int,
        observation: Mapping[str, Any] | None = None,
    ) -> np.ndarray | None:
        if observation is not None:
            predictions = observation.get("predictions")
            if isinstance(predictions, Mapping):
                result = predictions.get(object_id, predictions.get(str(object_id)))
                if isinstance(result, Mapping) and result.get("predicted_grasp_pose") is not None:
                    return np.asarray(result["predicted_grasp_pose"], dtype=float).copy()
        result = self._prediction_record(int(object_id))
        if result is None:
            return None
        return np.asarray(result["predicted_grasp_pose"], dtype=float).copy()

    predict_grasp_pose = get_predicted_grasp_pose

    def is_target_feasible(
        self,
        object_id: int,
        observation: Mapping[str, Any] | None = None,
    ) -> bool:
        result = self._prediction_from_observation(object_id, observation)
        return bool(result and result["grasp_window"]["feasible"])

    is_grasp_window_feasible = is_target_feasible

    def is_target_graspable(
        self,
        object_id: int,
        observation: Mapping[str, Any] | None = None,
    ) -> bool:
        result = self._prediction_from_observation(object_id, observation)
        return bool(result and result["grasp_window"]["current_in_window"])

    is_in_grasp_window = is_target_graspable

    def _prediction_from_observation(
        self,
        object_id: int,
        observation: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if observation is not None:
            predictions = observation.get("predictions")
            if isinstance(predictions, Mapping):
                value = predictions.get(object_id, predictions.get(str(object_id)))
                if isinstance(value, Mapping):
                    return value
        return self._prediction_record(int(object_id))

    def is_object_grasped(self, object_id: int) -> bool:
        runtime = self._runtime_for_policy_object(int(object_id))
        return bool(runtime is not None and runtime.state == "GRASPED")

    is_target_grasped = is_object_grasped

    def has_severe_collision(self) -> bool:
        return bool(self._severe_collision)

    def get_bin_drop_pose(self, bin_id: int) -> np.ndarray:
        matches = [item for item in self._bins if int(item["id"]) == int(bin_id)]
        if not matches:
            raise ValueError(f"Unknown bin id: {bin_id}")
        item = matches[0]
        position = item["center"].copy()
        # Return an above-rim transfer waypoint.  The baseline adds its own
        # configurable release clearance; keeping the diagonal approach high
        # prevents the carried object or forearm from clipping a bin wall.
        position[2] = max(
            position[2] + 0.5 * item["size"][2],
            # Preserve the original 140 mm belt-relative release clearance
            # when the installation height changes.
            self.belt_height + 0.14,
        )
        quaternion = np.asarray(
            item.get(
                "drop_quaternion",
                _value(
                    self.robot_config,
                    "end_effector.nominal_down_quaternion_wxyz",
                    default=[0.0, 1.0, 0.0, 0.0],
                ),
            ),
            dtype=float,
        )
        return np.r_[position, quaternion]

    get_bin_pose = get_bin_drop_pose

    def get_wait_pose(
        self, observation: Mapping[str, Any] | None = None
    ) -> np.ndarray:
        position = np.asarray(
            _value(
                self.scene_config,
                "workspace.waiting_ee_position",
                default=self._ee_pose()[:3],
            ),
            dtype=float,
        )
        quaternion = np.asarray(
            _value(
                self.scene_config,
                "workspace.waiting_ee_quaternion_wxyz",
                default=_value(
                    self.robot_config,
                    "end_effector.nominal_down_quaternion_wxyz",
                    default=[0.0, 1.0, 0.0, 0.0],
                ),
            ),
            dtype=float,
        )
        return np.r_[position, quaternion]

    get_home_pose = get_wait_pose

    def on_policy_event(self, event: Any, *args: Any) -> None:
        name = str(getattr(event, "name", event))
        target_id = getattr(event, "target_id", None)
        reason = str(getattr(event, "reason", args[-1] if args else ""))
        if target_id is None:
            target_id = self.selected_target_id
        runtime = (
            None
            if target_id is None
            else self._runtime_for_policy_object(int(target_id))
        )
        task_state = (
            None
            if target_id is None or self.perception_mode != "rendered_rgbd"
            else self._vision_task_states.get(int(target_id))
        )

        def update_state(state: str, failure_reason: str | None = None) -> None:
            if task_state is not None:
                task_state.state = state
                task_state.failure_reason = failure_reason
            if runtime is not None and runtime.logical_id is not None:
                self.set_object_state(
                    int(runtime.logical_id), state, reason=failure_reason
                )

        if name == "target_selected" and target_id is not None:
            self.selected_target_id = int(target_id)
            if task_state is not None:
                task_state.compensation_enabled_at_selection = bool(
                    self.enable_motion_compensation
                )
                if task_state.state in {
                    "NEW",
                    "TRACKING",
                    "READY_TO_PLAN",
                }:
                    update_state("PLANNING")
            elif runtime is not None:
                runtime.compensation_enabled_at_selection = bool(
                    self.enable_motion_compensation
                )
                if runtime.state in {"NEW", "TRACKING", "READY_TO_PLAN"}:
                    update_state("PLANNING")
        elif name == "descent_started" and (
            runtime is not None or task_state is not None
        ):
            update_state("EXECUTING")
        elif name == "gripper_close_started":
            self.metrics.grasp_attempts += 1
            if runtime is not None or task_state is not None:
                update_state("CLOSING")
        elif (
            name == "attempt_failed"
            and (runtime is not None or task_state is not None)
            and (
                task_state.state
                if task_state is not None
                else runtime.state
            )
            not in TERMINAL_OBJECT_STATES
        ):
            if "empty" in reason or "not_confirmed" in reason:
                self.metrics.empty_grasps += 1
            skipped_reasons = {
                "grasp_window_closed",
                "grasp_window_expired",
                "planned_grasp_time_expired",
                "insufficient_intercept_time",
                "target_left_grasp_region",
                "trajectory_outside_reachable_workspace",
            }
            terminal_state = "SKIPPED" if reason in skipped_reasons else "FAILED"
            update_state(terminal_state, reason)
        elif name in {"recovery_complete", "attempt_complete"}:
            state = (
                task_state.state
                if task_state is not None
                else None if runtime is None else runtime.state
            )
            if runtime is None or state in TERMINAL_OBJECT_STATES:
                self.selected_target_id = None

    handle_policy_event = on_policy_event

    def _task_text(self) -> str:
        if self.selected_target_id is None:
            return self.sorting_task.task_text
        spec = self._spec_for_policy_object(self.selected_target_id)
        if spec is None:
            return "跳过无法在时间窗内抓取的工件"
        return self.sorting_task.generate_instruction(
            int(spec.category_id),
            object_id=str(self.selected_target_id),
            include_skip_rule=True,
        )

    def get_metrics(self) -> dict[str, Any]:
        result = self.metrics.as_dict()
        result.update(
            {
                "simulation_time": float(self.data.time),
                "active_objects": sum(
                    runtime.active for runtime in self._objects
                ),
                "completed_objects": self._completed_objects,
                "belt_speed": float(self._current_belt_speed),
                "motion_compensation_enabled": self.enable_motion_compensation,
                "action_queue_overruns": int(self._action_queue_overruns),
                "held_payload_rail_collisions": len(self._held_payload_rail_contacts.episodes),
            }
        )
        return result

    # ------------------------------------------------------------------
    # Debug display, interactive rendering, and cleanup
    # ------------------------------------------------------------------
    def _set_mocap_marker(self, body_name: str, position: np.ndarray) -> None:
        body_id = self._name2id(
            mujoco.mjtObj.mjOBJ_BODY, body_name, required=False
        )
        if body_id < 0:
            return
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        self.data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)

    def _update_debug_markers(self) -> None:
        marker_names = _value(
            self.scene_config, "debug.mocap_bodies", default={}
        )
        if not isinstance(marker_names, Mapping):
            return
        hidden = np.asarray(
            _value(
                self.scene_config,
                "debug.hidden_position",
                default=[0.0, 0.0, -2.0],
            ),
            dtype=float,
        )
        positions: dict[str, np.ndarray] = {
            str(name): hidden for name in marker_names.values()
        }
        if self.debug_visualization and self.selected_target_id is not None:
            runtime = self._runtime_for_policy_object(
                self.selected_target_id
            )
            prediction = self._prediction_record(self.selected_target_id)
            if runtime is not None and prediction is not None:
                truth = self.data.xpos[runtime.body_id].copy()
                observed = np.asarray(prediction["observed_pose"][:3])
                predicted = np.asarray(prediction["predicted_grasp_pose"][:3])
                current_estimate = np.asarray(
                    prediction["current_estimated_pose"][:3],
                    dtype=float,
                )
                velocity = np.asarray(
                    prediction["estimated_velocity"],
                    dtype=float,
                )
                acceleration = np.asarray(
                    prediction["estimated_acceleration"],
                    dtype=float,
                )
                window = prediction["grasp_window"]
                now = float(self.data.time)

                def window_position(timestamp: float) -> np.ndarray:
                    delta = max(0.0, float(timestamp) - now)
                    position = (
                        current_estimate
                        + velocity * delta
                        + 0.5 * acceleration * delta * delta
                    )
                    if runtime.state in BELT_OBJECT_STATES:
                        position[2] = truth[2]
                    return position

                window_start_position = window_position(window["start_time"])
                window_end_position = window_position(window["end_time"])
                assignments = {
                    "selected_target": truth,
                    "ground_truth_pose": truth,
                    "observed_pose": observed,
                    "predicted_grasp_pose": predicted,
                    "actual_grasp_pose": (
                        hidden
                        if runtime.actual_grasp_pose is None
                        else runtime.actual_grasp_pose
                    ),
                    "grasp_window_start": window_start_position,
                    "grasp_window_end": window_end_position,
                    "pregrasp_pose": predicted
                    + np.array(
                        [
                            0.0,
                            0.0,
                            _as_float(
                                _value(
                                    self.scene_config,
                                    "workspace.pregrasp_clearance",
                                    default=0.10,
                                ),
                                0.10,
                            ),
                        ]
                    ),
                }
                for semantic_name, value in assignments.items():
                    body_name = marker_names.get(semantic_name)
                    if body_name is not None:
                        positions[str(body_name)] = np.asarray(value, dtype=float)
        for body_name, position in positions.items():
            self._set_mocap_marker(body_name, position)
        mujoco.mj_forward(self.model, self.data)

    def handle_key(self, keycode: int) -> None:
        try:
            key = chr(keycode)
        except (TypeError, ValueError):
            return
        if key == " ":
            self._paused = not getattr(self, "_paused", False)
        elif key.lower() == "d":
            self.debug_visualization = not self.debug_visualization
        elif key.lower() in {"r", "n", "c", "v"} or key in {
            "+", "=", "-", "_"
        }:
            self._viewer_command_queue.append(key)

    def process_pending_viewer_commands(self) -> dict[str, Any]:
        """Apply callback requests on the owner thread at a tick boundary."""

        commands: list[str] = []
        while self._viewer_command_queue and len(commands) < 64:
            commands.append(self._viewer_command_queue.popleft())
        reset_observation: dict[str, Any] | None = None
        for key in commands:
            lowered = key.lower()
            if lowered == "r":
                reset_observation = self.reset()
            elif lowered == "n":
                self.spawn_object()
            elif lowered == "c":
                self.toggle_motion_compensation()
            elif key in {"+", "="}:
                self.set_belt_speed(self._commanded_belt_speed + 0.05)
            elif key in {"-", "_"}:
                self.set_belt_speed(self._commanded_belt_speed - 0.05)
            elif lowered == "v":
                if self._video_writer is None:
                    self._start_viewer_video()
                else:
                    self._stop_viewer_video()
        return {
            "reset": reset_observation is not None,
            "observation": reset_observation,
            "commands": tuple(commands),
        }

    @property
    def paused(self) -> bool:
        return bool(getattr(self, "_paused", False))

    def _start_viewer_video(self) -> Path | None:
        """Start an MP4 stream of the top camera for the interactive V key."""

        if self._video_writer is not None:
            return self._video_path
        output = Path(
            str(
                _value(
                    self.scene_config,
                    "debug.video_output_dir",
                    default="recordings/viewer",
                )
            )
        ).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = (output / f"viewer_{timestamp}.mp4").resolve()
        try:
            import imageio.v2 as imageio

            self._video_writer = imageio.get_writer(
                path,
                fps=max(1, round(1.0 / self.control_dt)),
                codec="libx264",
                quality=8,
            )
        except Exception as error:
            self._video_writer = None
            self._video_path = None
            self.video_recording_requested = False
            warnings.warn(
                f"Could not start viewer video recording: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        self._video_path = path
        self.video_recording_requested = True
        print(f"Viewer video recording started: {path}")
        return path

    def _append_viewer_video_frame(self) -> None:
        if self._video_writer is None:
            return
        try:
            # Recording is an explicit consumer of the display stream.  Its
            # render timestamp remains separate from the depth frame consumed
            # by the policy; repeated frames are expected between 15 Hz RGB
            # captures and the 50 Hz video writer cadence.
            frame = self._capture_display_cameras(
                include_wrist=False
            )["top_rgb"]
            self._video_writer.append_data(np.asarray(frame, dtype=np.uint8))
        except Exception as error:
            warnings.warn(
                f"Viewer video frame could not be written: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._stop_viewer_video()

    def _stop_viewer_video(self) -> Path | None:
        path = self._video_path
        writer = self._video_writer
        self._video_writer = None
        self._video_path = None
        self.video_recording_requested = False
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                warnings.warn(
                    f"Could not finalize viewer video: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if path is not None:
            print(f"Viewer video recording saved: {path}")
        return path

    def _update_viewer_overlays(self) -> None:
        """Draw the recent EE path in the viewer's user scene."""

        if self._viewer is None:
            return
        points = list(self._ee_trajectory) if self.debug_visualization else []
        trajectory_config = _value(
            self.scene_config, "debug.ee_trajectory", default={}
        )
        width = _as_float(_value(trajectory_config, "line_width", default=2.0), 2.0)
        rgba = np.asarray(
            _value(
                trajectory_config,
                "rgba",
                default=[0.12, 0.92, 0.92, 0.72],
            ),
            dtype=np.float32,
        )
        if rgba.shape != (4,):
            rgba = np.array([0.12, 0.92, 0.92, 0.72], dtype=np.float32)
        with self._viewer.lock():
            debug_group = int(
                _value(
                    self.scene_config,
                    "debug.marker_group",
                    default=5,
                )
            )
            if 0 <= debug_group < len(self._viewer.opt.geomgroup):
                visible = int(self.debug_visualization)
                self._viewer.opt.geomgroup[debug_group] = visible
                self._viewer.opt.sitegroup[debug_group] = visible
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = int(
                self.debug_visualization
                and bool(
                    _value(
                        self.scene_config,
                        "debug.show_camera_frusta",
                        default=True,
                    )
                )
            )
            scene = self._viewer.user_scn
            scene.ngeom = 0
            for start, end in zip(points[:-1], points[1:]):
                if scene.ngeom >= scene.maxgeom:
                    break
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).reshape(-1),
                    rgba,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    width,
                    np.asarray(start, dtype=float),
                    np.asarray(end, dtype=float),
                )
                scene.ngeom += 1

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._capture_display_cameras(
                force=True, include_wrist=False
            )["top_rgb"].copy()
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.handle_key
            )
        if self._viewer.is_running():
            self._update_viewer_overlays()
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._closed:
            return
        for worker in self._planning_workers:
            try:
                worker.close()
            except Exception:
                pass
        self._planning_workers.clear()
        self._stop_viewer_video()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderers.values():
            try:
                renderer.close()
            except Exception:
                pass
        self._renderers.clear()
        self._closed = True

    def __enter__(self) -> "ConveyorSortingEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

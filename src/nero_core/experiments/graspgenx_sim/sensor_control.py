"""Sensor-only policy boundary for the independent conveyor experiment.

The renderer/physics/evaluator stay outside this object.  Policy inputs are
delayed visual-buffer measurements, robot and jaw encoders, command history,
an emulated conveyor encoder, and actuator effort.  No contact array, contact
force, runtime object, physics lifecycle or visual-to-physics association is
read.  Inferred motor obstruction is *not* evidence of which object is held.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SensorControlConfig:
    minimum_gap_m: float = .002
    minimum_position_error_m: float = .001
    maximum_jaw_velocity_m_s: float = .005
    minimum_motor_effort: float = .5
    obstruction_dwell_s: float = .06
    confirmed_load_loss_dwell_s: float = .06
    telemetry_gap_s: float = .05
    maximum_track_age_s: float = 4.8
    minimum_history_samples: int = 3
    coast_confidence_half_life_s: float = 2.


class MotorObstructionObserver:
    """Detect jaw obstruction from encoders and motor effort, without contacts.

    The evidence is stalled position actuators with a
    nonzero gap and sustained effort.  A table obstruction can also satisfy
    it; independent placement scoring must therefore remain outside control.
    """
    def __init__(self, config: SensorControlConfig):
        self.config = config
        self.stable_since = None
        self.previous_timestamp = None
        self.holding = False
        self.loss_since = None

    def reset(self):
        self.stable_since = self.previous_timestamp = None
        self.holding = False
        self.loss_since = None

    def update(self, *, timestamp, positions, velocities, efforts,
               closed_positions, open_positions, target_fraction,
               closed_width_m, open_width_m):
        arrays = [np.asarray(v, dtype=float).reshape(-1) for v in
                  (positions, velocities, efforts, closed_positions, open_positions)]
        q, qd, effort, closed, opened = arrays
        if (q.size != 2 or any(a.shape != q.shape for a in (qd, closed, opened))
                or effort.size not in {1, 2}):
            raise ValueError("AGX observer requires two jaw encoders and one coupled/two actuator efforts")
        now, target = float(timestamp), float(target_fraction)
        telemetry = np.concatenate(([now, target, closed_width_m, open_width_m], *arrays))
        if not np.isfinite(telemetry).all():
            self.reset()
            raise ValueError("nonfinite AGX sensor telemetry")
        span = opened - closed
        if np.any(abs(span) < 1e-9):
            raise ValueError("jaw encoder endpoint calibration is degenerate")
        fractions = (q - closed) / span
        fraction = float(np.clip(np.mean(fractions), 0., 1.))
        opening = float(closed_width_m + fraction * (open_width_m - closed_width_m))
        target_q = closed + np.clip(target, 0., 1.) * span
        errors = (q - target_q) * np.sign(span)
        common_motor = effort.size == 1
        inward = (np.repeat(effort[0] * np.sign(span[0]) < -self.config.minimum_motor_effort, 2)
                  if common_motor else effort * np.sign(span) < -self.config.minimum_motor_effort)
        stalled = abs(qd) <= self.config.maximum_jaw_velocity_m_s
        obstruction = (errors >= self.config.minimum_position_error_m) & inward & stalled
        instant = bool(opening > self.config.minimum_gap_m and np.all(obstruction))
        loaded = bool(np.all((errors >= self.config.minimum_position_error_m) & inward))
        opening_requested = bool(target >= .99 or np.any(errors < -self.config.minimum_position_error_m))
        empty_gap = bool(opening <= self.config.minimum_gap_m)
        telemetry_gap = bool(self.previous_timestamp is not None
                and (now < self.previous_timestamp
                     or now - self.previous_timestamp > self.config.telemetry_gap_s))
        if telemetry_gap:
            self.stable_since = None
            self.loss_since = None
            self.holding = False
        self.previous_timestamp = now
        if empty_gap or opening_requested:
            self.stable_since = None
            self.loss_since = None
            self.holding = False
            phase = "empty_gap" if empty_gap else "opening_requested"
        elif self.holding:
            # Once obstruction has survived the strict acquisition dwell,
            # loaded jaw motion alone is not evidence of an empty grasp. The
            # observed v4 transfer recovered/opened on a single velocity
            # spike while gap, inward encoder error and effort stayed loaded.
            # No contact oracle is consulted to retain this motor estimate.
            if loaded:
                self.loss_since = None
                phase = "retained_loaded"
            else:
                if self.loss_since is None:
                    self.loss_since = now
                if now - self.loss_since + 1e-9 >= self.config.confirmed_load_loss_dwell_s:
                    self.holding = False
                    self.stable_since = None
                    phase = "confirmed_load_lost"
                else:
                    phase = "load_loss_pending"
        else:
            self.loss_since = None
            if not instant:
                self.stable_since = None
            elif self.stable_since is None:
                self.stable_since = now
            self.holding = bool(instant and now - self.stable_since + 1e-9
                                >= self.config.obstruction_dwell_s)
            phase = "acquired" if self.holding else "acquiring" if instant else "unconfirmed"
        if telemetry_gap:
            phase = "telemetry_gap_reacquiring" if instant else "telemetry_gap"
        return {
            "joint_pos": q.copy(), "joint_vel": qd.copy(),
            "motor_telemetry_timestamp": now,
            "motor_effort": effort.copy(), "motor_position_error_m": errors,
            "opening": opening, "opening_width": opening,
            "open_fraction": fraction, "target_open_fraction": float(target),
            "is_open": bool(fraction >= .97), "is_closed": bool(fraction <= .03),
            "is_holding": self.holding,
            # Compatibility names consumed by the inherited policy.  They
            # explicitly describe inferred motor obstruction, not mj contacts.
            "left_finger_contact": bool(obstruction[0]),
            "right_finger_contact": bool(obstruction[1]),
            "any_external_finger_contact": bool(np.any(obstruction)),
            "both_fingers_contact": bool(np.all(obstruction)),
            "contact_flags_source": "inferred_motor_obstruction_not_contact_sensor",
            "motor_coupling": "single_symmetry_coupled_actuator" if common_motor else "two_actuators",
            "bilateral_contact_observed": False,
            "holding_source": "jaw_gap_velocity_motor_effort_dwell",
            "feedback_source": "jaw_gap_velocity_common_motor_effort_dwell_no_contact_truth",
            "status": "holding" if self.holding else "open" if fraction >= .97 else "moving",
            "obstruction_stable_since": self.stable_since,
            "holding_phase": phase, "confirmed_load_loss_since": self.loss_since,
            "instant_stall_obstruction": instant,
            "loaded_inward_motor_obstruction": loaded,
            "opening_requested": opening_requested,
        }


@dataclass
class VisualTask:
    track_id: int
    category_id: int
    state: str = "TRACKING"
    planned_time: float | None = None
    planned_pose: np.ndarray | None = None
    failure_reason: str | None = None
    last_record: dict = field(default_factory=dict)


class _GripperSensorView:
    def __init__(self, boundary):
        self._boundary = boundary

    def get_state(self):
        return deepcopy(self._boundary._jaw_state)

    read_state = get_state

    def __getattr__(self, name):
        if name in {"config", "open_joint_positions", "closed_joint_positions"}:
            return deepcopy(getattr(self._boundary._physics.gripper_controller, name))
        raise AttributeError(f"gripper sensor boundary blocks {name!r}")


class _RobotSensorView:
    def __init__(self, boundary):
        self._boundary = boundary

    def get_state(self):
        return self._boundary.get_robot_state()

    def __getattr__(self, name):
        if name in {"config", "max_joint_velocity", "max_joint_acceleration",
                    "commanded_joint_positions", "commanded_joint_velocities",
                    "joint_lower_limits", "joint_upper_limits"}:
            return deepcopy(getattr(self._boundary._physics.robot_controller, name))
        raise AttributeError(f"robot sensor boundary blocks {name!r}")


class SensorControlEnvironment:
    """Drop-in policy environment with a strictly separate local task ledger.

    ``get_observation`` reads public visual histories directly, never the
    simulator's observation/lifecycle helpers.  ``drain_events`` lets the owner
    mirror commands into *evaluation* bookkeeping after policy execution; no
    evaluator answer is imported back.  The object model in planner snapshots
    is frozen at qpos0, with only robot/jaw encoder fields filled in.
    """
    _STATIC_ATTRIBUTES = {
        "action_type", "robot_config", "scene_config", "arm_joint_names",
        "arm_joint_ids", "arm_qpos_adrs", "arm_dof_adrs", "ee_site_name",
        "ee_site_id", "gripper_qpos_adrs", "gripper_joint_ids", "control_dt",
        "model", "belt_height", "grasp_region", "reach_y", "belt_max_x",
    }
    _STATIC_METHODS = {"get_bin_drop_pose", "get_bin_pose", "get_wait_pose",
                       "get_home_pose", "set_motion_profile",
                       "create_global_planning_worker", "create_prefetch_planning_worker"}
    _DONE = {"FAILED", "SKIPPED", "MISSED", "RELEASE_COMMAND_COMPLETED"}
    perception_mode = "rendered_rgbd"
    grasp_assist = False

    def __init__(self, physics, config: SensorControlConfig | None = None):
        if physics.perception_mode != "rendered_rgbd" or physics.grasp_assist:
            raise ValueError("require rendered vision and physical non-assisted grasps")
        self._physics = physics
        self.config = config or SensorControlConfig()
        self.observer = MotorObstructionObserver(self.config)
        self.tasks: dict[int, VisualTask] = {}
        self.selected_target_id = None
        self._scene_epoch = None
        self._events: list[Any] = []
        self._jaw_state = {}
        self._last_observation = None
        from experiments.graspgenx_sim.motor_effort_contract_audit import audit_agx_motor_contract
        self._motor_contract = audit_agx_motor_contract(physics.model, physics.gripper_controller)
        self.gripper_controller = _GripperSensorView(self)
        self.robot_controller = _RobotSensorView(self)
        # Kinematics sees only a neutral model plus robot encoder values.
        import mujoco
        from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK
        self._kinematic_data = mujoco.MjData(physics.model)
        self._validation_data = mujoco.MjData(physics.model)
        self.ik = DampedLeastSquaresIK(physics.model, self._kinematic_data,
                                      physics.arm_joint_names, physics.ee_site_name,
                                      physics.robot_config.get("ik", {}))
        self.ik_solver = self.ik

    def __getattr__(self, name):
        if name in self._STATIC_ATTRIBUTES or name in self._STATIC_METHODS:
            return getattr(self._physics, name)
        if name == "data":
            return SimpleNamespace(time=float(self._physics.data.time),
                                   qpos=self._neutral_qpos())
        raise AttributeError(f"sensor-only control boundary blocks {name!r}")

    def provenance(self):
        return {"control_source": "delayed_rendered_visual_measurements_and_emulated_hardware",
                "object_ground_truth": False, "physics_lifecycle": False,
                "mj_contact_force": False, "true_contact_object_identity": False,
                "planner_nonrobot_qpos": "model.qpos0, never live object state",
                "gripper_feedback": "two coupled jaw encoders, velocities, common actuator effort, dwell",
                "grasp_identity_confirmation": False,
                "motor_effort_contract_audit": deepcopy(self._motor_contract),
                "release_confirmation": "command completed; correct bin verified only by evaluator",
                "config": asdict(self.config)}

    def _neutral_qpos(self):
        full = np.asarray(self.model.qpos0).copy()
        full[self.arm_qpos_adrs] = self.get_joint_positions()
        full[self.gripper_qpos_adrs] = self._physics.data.qpos[self.gripper_qpos_adrs]
        return full

    def get_joint_positions(self):
        return np.asarray(self._physics.data.qpos[self.arm_qpos_adrs]).copy()

    def get_joint_velocities(self):
        return np.asarray(self._physics.data.qvel[self.arm_dof_adrs]).copy()

    def get_robot_state(self):
        self._kinematic_data.qpos[:] = self._neutral_qpos()
        pose = self.ik.forward_pose(self.get_joint_positions())
        return {"joint_pos": self.get_joint_positions(),
                "joint_vel": self.get_joint_velocities(), "ee_pose": pose}

    get_state = get_robot_state

    def _read_jaws(self, now):
        controller = self._physics.gripper_controller
        joint_ids = np.asarray(controller.joint_ids, dtype=int)
        q_indices = self.model.jnt_qposadr[joint_ids]
        v_indices = self.model.jnt_dofadr[joint_ids]
        state = self.observer.update(
            timestamp=now, positions=self._physics.data.qpos[q_indices],
            velocities=self._physics.data.qvel[v_indices],
            efforts=self._physics.data.actuator_force[controller.actuator_ids],
            closed_positions=controller.closed_joint_positions,
            open_positions=controller.open_joint_positions,
            target_fraction=controller.target_open_fraction,
            closed_width_m=controller.config.closed_width,
            open_width_m=controller.config.open_width)
        state.update(self._motor_contract["telemetry_contract"])
        return state

    def get_observation(self, **_):
        now = float(self._physics.data.time)
        epoch = int(self._physics._scene_epoch)
        if epoch != self._scene_epoch:
            self.tasks.clear()
            self.selected_target_id = None
            self.observer.reset()
            self._scene_epoch = epoch
        self._jaw_state = self._read_jaws(now)
        robot = self.get_robot_state()
        predictions, records = {}, []
        # Public buffer is populated before evaluation association.  Preserve
        # its delayed-read tombstones: no immortal last-visible target fallback.
        buffer = self._physics.perception_buffer
        for identifier in buffer.object_ids:
            sample = buffer.get_delayed(str(identifier), now,
                                        float(self._physics.visual_latency), mode="previous")
            if sample is None:
                continue
            track_id = int(identifier)
            observed = dict(sample.state)
            stamp = float(sample.timestamp)
            age = max(0., now - stamp)
            visual_age_at_capture = float(observed.get("track_observation_age", 0.))
            track_age = max(0., visual_age_at_capture) + age
            if track_age > self.config.maximum_track_age_s:
                continue
            category = int(observed["category_id"])
            task = self.tasks.setdefault(track_id, VisualTask(track_id, category))
            if task.state in self._DONE:
                continue
            if task.state in {"TRACKING", "PLANNING"}:
                task.category_id = category
            position = np.asarray(observed["position"], dtype=float).copy()
            quat = np.asarray(observed["quaternion"], dtype=float).copy()
            if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[position, quat]).all():
                continue
            velocity = np.asarray(observed.get("linear_velocity", [0., 0., 0.]), dtype=float).copy()
            # Conveyor encoder integration compensates only post-capture age.
            travel = self._physics.get_belt_displacement(stamp, now)
            encoder_speed = float(self._physics._current_belt_speed)
            if travel is not None:
                position[0] += float(travel)
                velocity[0] = encoder_speed
            else:
                position += velocity * age
            velocity[2] = 0.
            history = buffer.history(str(identifier)).between(
                stamp - float(self._physics.history_window), stamp)
            confidence = float(observed.get("confidence", 1.))
            coast = max(0., age - float(self._physics.visual_latency) - float(self._physics.camera_period))
            confidence *= 2 ** (-coast / self.config.coast_confidence_half_life_s)
            window = self._visual_window(position, velocity, now, task)
            ready = len(history) >= self.config.minimum_history_samples
            window["feasible"] = bool(window["feasible"] and ready and confidence >= .05)
            predicted_time = task.planned_time
            if predicted_time is None:
                speed = max(float(velocity[0]), 0.)
                preferred = float(self.scene_config.get("workspace", {}).get("preferred_grasp_x", .08))
                predicted_time = now + ((max(0., preferred - position[0]) / speed) if speed > 1e-5 else 0.)
            future = np.r_[position + velocity * max(0., predicted_time - now), quat]
            future[2] = position[2]
            prediction = {"object_id": track_id, "observed_timestamp": stamp,
                "capture_timestamp": stamp, "publication_timestamp": now,
                "measurement_timestamp": stamp - max(0., visual_age_at_capture),
                "observation_age": age, "track_observation_age": track_age,
                "observed_pose": np.r_[np.asarray(observed["position"]), quat],
                "current_estimated_pose": np.r_[position, quat],
                "predicted_grasp_pose": future, "estimated_velocity": velocity,
                "estimated_acceleration": np.zeros(3),
                "estimated_grasp_time": float(predicted_time),
                "prediction_confidence": confidence, "prediction_uncertainty": np.zeros(3),
                "prediction_ready": ready, "visible": bool(observed.get("visible", True)),
                "orientation_confidence": float(observed.get("orientation_confidence", 0.)),
                "grasp_window": window, "history_sample_count": len(history),
                "motion_compensation_enabled": True,
                "prediction_model": "visual_measurement_plus_conveyor_encoder"}
            if task.planned_time is not None:
                prediction["planned_grasp_time"] = task.planned_time
                prediction["planned_grasp_pose"] = task.planned_pose.copy()
            # Category is a visual classifier output; bin mapping is static
            # task configuration. No category grasp template is copied here.
            spec = self._physics._object_spec_by_category.get(task.category_id)
            if spec is None:
                continue
            record = {**prediction, "object_id": track_id, "category_id": task.category_id,
                "category_confidence": float(observed.get("category_confidence", 0.)),
                "bin_id": int(spec.bin_id), "position": position, "quaternion": quat,
                "linear_velocity": velocity, "timestamp": now, "state": task.state,
                "pose_source": "delayed_rendered_vision_only",
                "window_valid": window["feasible"], "in_grasp_window": window["current_in_window"],
                "grasp_window_start": window["start_time"], "grasp_window_end": window["end_time"]}
            task.last_record = deepcopy(record)
            records.append(record)
            predictions[track_id] = prediction
        controller = self._physics.robot_controller
        selected = predictions.get(self.selected_target_id)
        result = {**robot, "joint_target": controller.commanded_joint_positions,
            "joint_target_velocity": controller.commanded_joint_velocities,
            "gripper": deepcopy(self._jaw_state), "gripper_opening": self._jaw_state["opening_width"],
            "belt_speed": float(self._physics._current_belt_speed), "timestamp": now,
            "scene_epoch": epoch, "objects": records, "predictions": predictions,
            "selected_target_id": self.selected_target_id, "perception": selected or {},
            "perception_mode": self.perception_mode,
            "perception_source": "rendered_visual_buffer_sensor_only_control",
            "system_latency": {name: float(getattr(self._physics, name + "_latency"))
                               for name in ("visual", "planning", "control", "robot_response")}}
        self._last_observation = result
        return deepcopy(result)

    def _observed_state_at(self, track_id, timestamp):
        """Read capture-time visual history for inference provenance only.

        The caller must separately match the returned timestamp/track to the
        rendered mask.  This never falls back to simulator object helpers.
        """
        history = self._physics.perception_buffer.history(str(int(track_id)))
        if history is None:
            return float(timestamp), None
        sample = history.read_at(float(timestamp), mode="previous", allow_extrapolation=False)
        if sample is None:
            return float(timestamp), None
        allowed = {"object_id", "track_id", "track_generation", "generation",
                   "position", "quaternion", "linear_velocity", "category_id",
                   "category_confidence", "confidence", "orientation_confidence",
                   "visible", "timestamp", "last_seen", "last_update", "source",
                   "mask_pixels", "valid_depth_pixels", "feature_vector", "track_observation_age"}
        state = {key: deepcopy(value) for key, value in dict(sample.state).items() if key in allowed}
        state["timestamp"] = float(sample.timestamp)
        state["pose_source"] = "rendered_visual_buffer_only"
        return float(sample.timestamp), state

    def _visual_window(self, position, velocity, now, task):
        speed = float(velocity[0])
        region = np.asarray(self.grasp_region)
        x, y = map(float, position[:2])
        in_window = bool(region[0] <= x <= region[1] and self.reach_y[0] <= y <= self.reach_y[1])
        if speed > 1e-5:
            start = now + max(0., (region[0] - x) / speed)
            end = now + max(0., (min(region[1], self.belt_max_x) - x) / speed)
        else:
            start, end = now, now + 60.
        feasible = bool(x <= region[1] and self.reach_y[0] <= y <= self.reach_y[1]
                        and (speed > 1e-5 or in_window))
        return {"start_time": start, "end_time": end, "time_remaining": max(0., end - now),
                "current_in_window": in_window, "is_graspable": in_window,
                "feasible": feasible, "reason": "ok" if feasible else "visual_workspace_gate"}

    def get_object_state(self, track_id, *_):
        # This hook reads our visual/action ledger, not the simulator's helper.
        task = self.tasks.get(int(track_id))
        if task is None or not task.last_record:
            return None
        present = any(int(item["object_id"]) == int(track_id)
                      for item in (self._last_observation or {}).get("objects", []))
        if not present and task.state not in self._DONE | {"CLOSING", "CARRYING_SENSOR_CONFIRMED"}:
            return None
        result = deepcopy(task.last_record)
        result["state"] = task.state
        if task.failure_reason is not None:
            result["failure_reason"] = task.failure_reason
        if task.state not in self._DONE and task.state not in {"CLOSING", "CARRYING_SENSOR_CONFIRMED"}:
            now = float(self._physics.data.time)
            if now - float(result["observed_timestamp"]) > self.config.maximum_track_age_s:
                return None
        return result

    get_target = get_object_state
    get_object = get_object_state

    def select_target(self, observation=None):
        obs = self.get_observation() if observation is None else observation
        candidates = [x for x in obs.get("objects", []) if x.get("state") in {"TRACKING", "NEW", "READY_TO_PLAN"}
                      and x.get("window_valid", False)]
        if self.selected_target_id is not None:
            existing = next((x for x in obs.get("objects", [])
                             if x.get("object_id") == self.selected_target_id
                             and x.get("state") not in self._DONE), None)
            if existing is not None:
                return deepcopy(existing)
        if not candidates:
            return None
        chosen = min(candidates, key=lambda x: (x["grasp_window_end"], -x["position"][0], x["object_id"]))
        track_id = int(chosen["object_id"])
        self.selected_target_id = track_id
        task = self.tasks[track_id]
        task.state = "PLANNING"
        task.planned_time = float(chosen["estimated_grasp_time"])
        task.planned_pose = np.asarray(chosen["predicted_grasp_pose"]).copy()
        return deepcopy(chosen)

    select_sorting_target = select_target

    def release_selected_target(self, *_):
        self.selected_target_id = None

    release_target_reservation = release_selected_target

    def get_predicted_grasp_pose(self, track_id, observation=None):
        obs = self._last_observation if observation is None else observation
        prediction = (obs or {}).get("predictions", {}).get(int(track_id))
        return None if prediction is None else np.asarray(prediction["predicted_grasp_pose"]).copy()

    predict_grasp_pose = get_predicted_grasp_pose

    def is_target_feasible(self, track_id, observation=None):
        obs = self._last_observation if observation is None else observation
        return bool((obs or {}).get("predictions", {}).get(int(track_id), {}).get("grasp_window", {}).get("feasible", False))

    is_grasp_window_feasible = is_target_feasible
    target_window_feasible = is_target_feasible

    def is_target_graspable(self, track_id, observation=None):
        obs = self._last_observation if observation is None else observation
        return bool((obs or {}).get("predictions", {}).get(int(track_id), {}).get("grasp_window", {}).get("current_in_window", False))

    is_in_grasp_window = is_target_graspable
    target_in_grasp_window = is_target_graspable

    def refine_selected_grasp_plan(self, track_id, required_motion_time,
                                   deadline_margin=.25, *, allow_retime=True,
                                   final_tracking_correction_limit=None,
                                   requested_time=None, requested_pose=None):
        task = self.tasks.get(int(track_id))
        prediction = (self._last_observation or {}).get("predictions", {}).get(int(track_id))
        if task is None or task.state != "PLANNING" or prediction is None or int(track_id) != self.selected_target_id:
            return {"accepted": False, "reason": "target_not_in_planning"}
        duration, reserve = float(required_motion_time), float(deadline_margin)
        if not np.isfinite([duration, reserve]).all() or min(duration, reserve) < 0:
            return {"accepted": False, "reason": "invalid_intercept_timing"}
        now = float(self._physics.data.time)
        earliest = now + duration
        end = float(prediction["grasp_window"]["end_time"])
        latest = end - reserve
        current = np.asarray(prediction["current_estimated_pose"]).copy()
        velocity = np.asarray(prediction["estimated_velocity"])
        dynamic = self.scene_config.get("workspace", {}).get("dynamic_intercept", {}).get("enabled", False)
        if not dynamic and velocity[0] > 1e-5:
            preferred = float(self.scene_config.get("workspace", {}).get("preferred_grasp_x", .08))
            earliest = max(earliest, now + max(0., preferred - current[0]) / velocity[0])
        if earliest > latest + 1e-9:
            return {"accepted": False, "reason": "insufficient_intercept_time",
                    "required_grasp_time": earliest, "latest_grasp_time": latest, "window_end": end}
        previous = task.planned_time
        keep = previous is not None and earliest - 1e-9 <= previous <= latest + 1e-9
        chosen_time = float(requested_time) if requested_time is not None else previous if keep else earliest
        if not np.isfinite(chosen_time) or not earliest - 1e-9 <= chosen_time <= latest + 1e-9:
            return {"accepted": False, "reason": "insufficient_intercept_time"}
        pose = current.copy()
        pose[:3] += velocity * max(0., chosen_time - now)
        pose[2] = current[2]
        if requested_pose is not None:
            supplied = np.asarray(requested_pose, dtype=float)
            if supplied.shape != (7,) or not np.isfinite(supplied).all():
                return {"accepted": False, "reason": "invalid_requested_intercept"}
            pose[:2] = supplied[:2]
        elif keep and task.planned_pose is not None and not dynamic:
            pose = task.planned_pose.copy()
        if not self.grasp_region[0] <= pose[0] <= self.grasp_region[1] or not self.reach_y[0] <= pose[1] <= self.reach_y[1]:
            return {"accepted": False, "reason": "trajectory_outside_reachable_workspace"}
        correction = 0. if task.planned_pose is None else float(np.linalg.norm(pose[:3] - task.planned_pose[:3]))
        if not allow_retime and not keep:
            limit = 0. if final_tracking_correction_limit is None else float(final_tracking_correction_limit)
            if task.planned_pose is None or not np.isfinite(limit) or correction > limit + 1e-9:
                return {"accepted": False, "reason": "insufficient_intercept_time", "tracking_correction": correction}
        task.planned_time, task.planned_pose = float(chosen_time), pose.copy()
        return {"accepted": True, "reason": "ok", "changed": previous != chosen_time,
                "estimated_grasp_time": float(chosen_time), "predicted_grasp_pose": pose,
                "required_grasp_time": earliest, "window_end": end,
                "latest_grasp_time": latest, "deadline_slack": latest - chosen_time,
                "tracking_correction": correction}

    retime_selected_grasp_plan = refine_selected_grasp_plan

    def validate_selected_grasp_plan(self, track_id, required_motion_time,
                                    deadline_margin=.25, final_tracking_correction_limit=.012):
        return self.refine_selected_grasp_plan(track_id, required_motion_time, deadline_margin,
            allow_retime=False, final_tracking_correction_limit=final_tracking_correction_limit)

    def is_object_grasped(self, *_):
        return bool(self.observer.holding)

    is_target_grasped = is_object_grasped

    def has_severe_collision(self, *_):
        # No simulator privileged collision flag. Static robot-self/scene
        # checks run on model FK below; unknown obstacles need a future vision
        # occupancy check rather than a contact oracle.
        return not self.is_arm_configuration_safe(self.get_joint_positions())

    collision_abort_required = has_severe_collision

    def on_policy_event(self, event, *_):
        self._events.append(event)
        name = str(getattr(event, "name", event))
        track_id = getattr(event, "target_id", self.selected_target_id)
        task = None if track_id is None else self.tasks.get(int(track_id))
        if task is None:
            return
        states = {"target_selected": "PLANNING", "descent_started": "EXECUTING",
                  "gripper_close_started": "CLOSING", "lift_started": "CARRYING_SENSOR_CONFIRMED",
                  "release_complete": "RELEASE_COMMAND_COMPLETED", "attempt_failed": "FAILED"}
        if name in states:
            task.state = states[name]
        if name == "attempt_failed":
            task.failure_reason = str(getattr(event, "reason", "attempt_failed"))
        if name in {"attempt_complete", "recovery_complete"}:
            self.selected_target_id = None

    handle_policy_event = on_policy_event

    def drain_events(self):
        events, self._events = self._events, []
        return events

    def get_global_planning_snapshot(self):
        return {"full_qpos": self._neutral_qpos(), "arm_qpos": self.get_joint_positions(),
                "scene_epoch": np.asarray(self._scene_epoch or 0, dtype=np.int64)}

    def candidate_snapshot(self):
        """Owner-thread clone; nonrobot qpos never comes from live physics."""
        import mujoco
        from experiments.graspgenx_sim.candidates import CandidateEnvironmentSnapshot
        neutral = mujoco.MjData(self.model)
        neutral.qpos[:] = self._neutral_qpos()
        neutral.time = float(self._physics.data.time)
        mujoco.mj_forward(self.model, neutral)
        fields = {name: getattr(self, name) for name in (
            "model", "belt_height", "scene_config", "robot_config", "arm_joint_names",
            "ee_site_name", "ee_site_id", "arm_joint_ids", "arm_qpos_adrs", "arm_dof_adrs",
            "gripper_joint_ids", "gripper_qpos_adrs", "ik")}
        fields.update(data=neutral, _scene_epoch=int(self._scene_epoch or 0),
                      _robot_collision_body_ids=set(self._physics._robot_collision_body_ids))
        return CandidateEnvironmentSnapshot(SimpleNamespace(**fields))

    def get_future_open_planning_snapshot(self, arm_joint_positions):
        result = self.get_global_planning_snapshot()
        arm = np.asarray(arm_joint_positions, dtype=float)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("future arm state must have seven finite joint values")
        result["full_qpos"][self.arm_qpos_adrs] = arm
        result["full_qpos"][self.gripper_qpos_adrs] = self._physics.gripper_controller.open_joint_positions
        result["arm_qpos"] = arm.copy()
        return result

    def get_future_carry_planning_snapshot(self, arm_joint_positions, gripper_action):
        result = self.get_future_open_planning_snapshot(arm_joint_positions)
        controller = self._physics.gripper_controller
        span = controller.config.action_open - controller.config.action_closed
        fraction = float(np.clip((gripper_action - controller.config.action_closed) / span, 0., 1.))
        joints = controller.closed_joint_positions + fraction * (controller.open_joint_positions - controller.closed_joint_positions)
        result["full_qpos"][self.gripper_qpos_adrs] = joints
        return result

    def solve_ik_result(self, *args, **kwargs):
        self._kinematic_data.qpos[:] = self._neutral_qpos()
        from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
        # This unbound function uses only our encoder reference/static model/
        # isolated IK; its multi-seed branch convention is unchanged.
        return ConveyorSortingEnv.solve_ik_result(self, *args, **kwargs)

    def solve_ik(self, *args, **kwargs):
        return self.solve_ik_result(*args, **kwargs).joint_positions.copy()

    inverse_kinematics = solve_ik

    def _safe_sample(self, positions, axis, minimum_downward_alignment,
                     minimum_tcp_height, minimum_joint_margin, condition):
        import mujoco
        scratch = self._validation_data
        scratch.qpos[:] = self._neutral_qpos()
        scratch.qpos[self.arm_qpos_adrs] = positions
        scratch.qvel[:] = 0.
        mujoco.mj_forward(self.model, scratch)
        return bool(self._physics._arm_state_data_is_safe(
            positions, scratch, approach_axis_local=axis,
            minimum_downward_alignment=minimum_downward_alignment,
            minimum_tcp_height=minimum_tcp_height, minimum_joint_margin=minimum_joint_margin,
            maximum_jacobian_condition=condition))

    def is_arm_configuration_safe(self, joint_positions, *, approach_axis_local=(0., 0., 1.),
                                  minimum_downward_alignment=0., minimum_tcp_height=0.,
                                  minimum_joint_margin=0.):
        q, axis = np.asarray(joint_positions, dtype=float), np.asarray(approach_axis_local, dtype=float)
        if q.shape != (7,) or axis.shape != (3,) or not np.isfinite(np.r_[q, axis]).all() or np.linalg.norm(axis) < 1e-9:
            return False
        axis = axis / np.linalg.norm(axis)
        return self._safe_sample(q, axis, minimum_downward_alignment, minimum_tcp_height,
                                 minimum_joint_margin, None)

    def is_arm_edge_safe(self, start_joint_positions, end_joint_positions, *, normalized_resolution,
                         maximum_joint_step=None, approach_axis_local=(0., 0., 1.),
                         minimum_downward_alignment=0., minimum_tcp_height=0.,
                         minimum_joint_margin=0., maximum_jacobian_condition):
        start, end, axis = map(lambda x: np.asarray(x, dtype=float),
                               (start_joint_positions, end_joint_positions, approach_axis_local))
        resolution = float(normalized_resolution)
        if start.shape != (7,) or end.shape != (7,) or axis.shape != (3,) or not np.isfinite(np.r_[start, end, axis, resolution]).all() or resolution <= 0. or np.linalg.norm(axis) < 1e-9:
            return False
        axis = axis / np.linalg.norm(axis)
        ranges = self.model.jnt_range[self.arm_joint_ids]
        count = int(np.ceil(np.linalg.norm((end - start) / np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)) / resolution))
        if maximum_joint_step is not None:
            step = float(maximum_joint_step)
            if not np.isfinite(step) or step <= 0.:
                return False
            count = max(count, int(np.ceil(np.max(abs(end - start)) / step)))
        return all(self._safe_sample(start + (end - start) * fraction, axis,
            minimum_downward_alignment, minimum_tcp_height, minimum_joint_margin,
            maximum_jacobian_condition) for fraction in np.linspace(0., 1., max(1, count) + 1))

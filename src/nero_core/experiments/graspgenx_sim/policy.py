"""Preserve NN candidate IK branches as seeds for the ordinary baseline.

These joints are numerical starting guesses, never certified endpoints or
paths.  The inherited solver, descent continuation, endpoint/edge checks and
live executor retain their complete original validation contracts.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import (
    BaselinePolicyConfig, BaselineSortingPolicy, PolicyStage,
)
from mujoco_conveyor_sorting.controllers._math import (
    compose_world_delta, quat_error_vector,
)
from mujoco_conveyor_sorting.controllers.visual_acquisition import VisualAcquisitionMixin
from .servo_response import VisualFollowingLeadAdapter


class NetworkSeedSortingPolicy(VisualAcquisitionMixin, BaselineSortingPolicy):
    """Try the selected candidate's pregrasp IK before the baseline seeds.

Construct with the original policy's ``config`` dataclass and the same env
hooks. Register ``candidate['pregrasp_ik']['joint_positions']`` for the visual
track and capture scene epoch. Each registration replaces that key's guess;
different tracks/epochs remain separate, and every reset clears the map.
Acquisition also uses bounded increments from measured TCP feedback, so the
live IK checks use the same small chords as nominal descent validation.
"""
    def __init__(self, config: BaselinePolicyConfig | Mapping[str, Any] | None = None,
                 *, enable_speculative_prefetch=False,
                 acquisition_translation_step_m=.010,
                 acquisition_command_integrator=False,
                 visual_acquisition_tracking=False,
                 minimum_motor_hold_effort=0.,
                 visual_servo_bias_compensation=False, **kwargs):
        step = float(acquisition_translation_step_m)
        if not np.isfinite(step) or not 0 < step <= .050:
            raise ValueError("acquisition_translation_step_m must be finite and in (0, .050]")
        self._acquisition_translation_step_m = step
        self._acquisition_command_integrator = bool(acquisition_command_integrator)
        self._visual_acquisition_tracking = bool(visual_acquisition_tracking)
        effort = float(minimum_motor_hold_effort)
        if not np.isfinite(effort) or effort < 0.:
            raise ValueError("minimum_motor_hold_effort must be finite and non-negative")
        self._minimum_motor_hold_effort = effort
        self._motor_preload_diagnostics = None
        self._visual_servo_bias_compensation = bool(visual_servo_bias_compensation)
        self._following_lead = None
        self._following_lead_diagnostics = None
        # Baseline __init__ invokes self.reset(), so this must exist first.
        self._network_pregrasp_seeds: dict[tuple[int, int], np.ndarray] = {}
        self.closure_diagnostics = []
        self._enable_network_speculative_prefetch = bool(enable_speculative_prefetch)
        super().__init__(config, **kwargs)
        self._following_lead = VisualFollowingLeadAdapter(
            initial_horizon_s=self.config.tracking_velocity_feedforward_time,
            maximum_horizon_s=self.config.tracking_synchronization_horizon,
            filter_time_s=.2)

    def _current_visual_acquisition(self, target, observation):
        """Return current *published* track position and velocity, not GT.

        The perception adapter has already propagated capture-time position
        to its publication time and has added the network's grasp offset.
        Adding track_observation_age here would count image latency twice.
        The separate capture age remains diagnostic evidence and is checked
        by input/lifecycle admission, rather than disguised as a fresh image.
        """
        if (not self._visual_acquisition_tracking
                or self._stage not in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
                or not isinstance(target, Mapping)):
            return None
        try:
            position = np.asarray(target.get("position"), dtype=float)
            velocity = np.asarray(target.get("estimated_velocity", target.get("linear_velocity")), dtype=float)
            publication = float(target.get("timestamp"))
            now = self._timestamp_without_advancing(observation)
        except (TypeError, ValueError):
            return None
        if (position.shape != (3,) or velocity.shape != (3,)
                or not np.isfinite(np.r_[position, velocity, publication, now]).all()
                or abs(now - publication) > 2. * self.config.control_timestep):
            return None
        return position.copy(), velocity.copy()

    def _grasp_target_pose(self, target, observation, *, remember_orientation=True):
        pose = super()._grasp_target_pose(
            target, observation, remember_orientation=remember_orientation)
        visual = self._current_visual_acquisition(target, observation)
        if pose is not None and visual is not None:
            # Preserve the network's height and quaternion. The global
            # PREGRASP route still ends at its certified future intercept;
            # only final acquisition becomes a current visual trajectory.
            pose = pose.copy()
            pose[:2] = visual[0][:2]
        return pose

    def _tracking_command_target(self, target_pose, observation, *, lead_time=None,
                                 target=None, synchronize_to_intercept=False):
        visual = self._current_visual_acquisition(target, observation)
        if visual is None:
            return super()._tracking_command_target(
                target_pose, observation, lead_time=lead_time, target=target,
                synchronize_to_intercept=synchronize_to_intercept)
        horizon = (self.config.tracking_velocity_feedforward_time
                   if lead_time is None else max(0., float(lead_time)))
        if self.config.dynamic_intercept_tracking and (lead_time is None or lead_time > 0.):
            dynamic = getattr(self.env, "scene_config", {}).get("workspace", {}).get("dynamic_intercept", {})
            horizon = float(dynamic.get("servo_lookahead_s", horizon))
        if self._visual_servo_bias_compensation and (lead_time is None or lead_time > 0.):
            current = self._ee_pose(observation)
            horizon, self._following_lead_diagnostics = self._following_lead.update(
                timestamp=self._timestamp_without_advancing(observation),
                contact_position=visual[0], feedback_position=current[:3], track_velocity=visual[1],
                acquisition_key=(self._observed_scene_epoch, self.target_id), fallback_horizon_s=horizon,
                near_capture_height=abs(current[2] - target_pose[2]) <= self.config.closure_vertical_tolerance)
            if self.closure_diagnostics and abs(self.closure_diagnostics[-1]["sim_s"] - self._timestamp_without_advancing(observation)) < 1e-9:
                self.closure_diagnostics[-1]["visual_servo_bias"] = deepcopy(self._following_lead_diagnostics)
        command = np.asarray(target_pose, dtype=float).copy()
        offset = visual[1][:2] * horizon
        command[:2] = visual[0][:2] + offset
        # This reference is independent of an old intercept clock and of a
        # low-margin pregrasp branch. Unsafe upstream chords are rejected by
        # the unchanged IK/edge checks; clamping time to zero silently left
        # the open finger downstream in the approaching object's path.
        if self.config.dynamic_intercept_tracking:
            dynamic = getattr(self.env, "scene_config", {}).get("workspace", {}).get("dynamic_intercept", {})
            bounds = dynamic.get("tracking_x_range", dynamic.get("x_range", [.02, .18]))
            command[0] = np.clip(command[0], *bounds)
        return self._guard_final_tracking_xy(
            command, target, observation, reference_offset_xy=offset)

    def _guard_final_tracking_xy(self, pose, target, observation, *, reference_offset_xy=None):
        if self._current_visual_acquisition(target, observation) is None:
            return super()._guard_final_tracking_xy(
                pose, target, observation, reference_offset_xy=reference_offset_xy)
        guarded = np.asarray(pose, dtype=float).copy()
        planned, planned_time, visible, age = self._planned_tracking_reference(target, observation)
        if planned is None or planned_time is None or visible or age <= 0.:
            return guarded
        # An occluded target may coast only within the original 35 mm
        # residual allowance, but its reference must move at encoder speed.
        # Signed elapsed time is essential before the future route intercept:
        # max(0, elapsed) would leave the anchor ahead of the current object.
        now = self._timestamp_without_advancing(observation)
        reference = planned[:2] + self._belt_velocity(observation)[:2] * (now - planned_time)
        if reference_offset_xy is not None:
            reference = reference + np.asarray(reference_offset_xy, dtype=float)
        guarded[:2] = self._bounded_planar_correction(
            guarded[:2], reference, self.config.coasted_tracking_xy_limit)
        # Do not freeze a conveyor trajectory on a leading-pad brush or on
        # initial stall. Existing confirmation/dwell gates freeze measured
        # TCP only when CLOSING actually transitions to LIFTING.
        return guarded

    def _cartesian_servo_pose(self, target_pose, observation, *, scale=1.0):
        if self._stage not in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}:
            return super()._cartesian_servo_pose(target_pose, observation, scale=scale)
        # The global planner certifies small descent chords. With the stock
        # TCP cap disabled, its servo otherwise requests the whole 100 mm
        # descent and may repeatedly reject a large redundancy jump. Start
        # each acquisition chord at measured feedback: an unexecuted command
        # integrator must not get ahead of the plant after a successful retry.
        # _arm_for_pose / _solve_pose still check the original single-seed IK,
        # branch jump, joint limits, conditioning and dense collision edges.
        # This is a command horizon, not a promise of per-tick plant travel.
        # Too short a measured-relative horizon starves the position servo of
        # tracking torque; the experimental config records the chosen horizon.
        current = self._ee_pose(observation)
        target = np.asarray(target_pose, dtype=float)
        if self._acquisition_command_integrator:
            # Retain the original per-command speed profile and numerical
            # retry increment, while bounding accumulated lead relative to
            # measured feedback. This permits a position servo to develop
            # torque without admitting an arbitrarily far-ahead IK chord.
            target = super()._cartesian_servo_pose(target, observation, scale=scale)
        translation = target[:3] - current[:3]
        translation_limit = self._acquisition_translation_step_m
        rotation_limit = self.config.max_rotation_step
        if (not self._acquisition_command_integrator
                and (self.config.tcp_speed_limits_enabled or self.config.dynamic_intercept_tracking)):
            translation_limit = min(translation_limit, self.config.max_translation_step,
                                    self.config.cartesian_linear_speed * self.config.control_timestep)
            rotation_limit = min(rotation_limit,
                                 self.config.cartesian_angular_speed * self.config.control_timestep)
        translation_limit *= float(scale)
        rotation_limit *= float(scale)
        distance = float(np.linalg.norm(translation))
        if distance > translation_limit:
            translation = translation * (translation_limit / distance)
        rotation = quat_error_vector(target[3:], current[3:])
        angle = float(np.linalg.norm(rotation))
        if angle > rotation_limit:
            rotation = rotation * (rotation_limit / angle)
        return np.r_[current[:3] + translation,
                     compose_world_delta(current[3:], rotation)]

    def _ensure_prefetch_worker(self):
        # The inherited speculative handoff assumes the next grasp is known
        # during return. Keep ordinary fully verified return/pregrasp routes
        # until the asynchronously supplied network pose has arrived.
        if not self._enable_network_speculative_prefetch:
            return None
        return super()._ensure_prefetch_worker()

    @staticmethod
    def _pregrasp_seed_key(track_id, scene_epoch):
        if (isinstance(track_id, (bool, np.bool_))
                or isinstance(scene_epoch, (bool, np.bool_))
                or not isinstance(track_id, Integral)
                or not isinstance(scene_epoch, Integral)
                or track_id < 0 or scene_epoch < 0):
            return None
        return int(track_id), int(scene_epoch)

    @staticmethod
    def _within_seed_limits(seed, lower, upper, margin=0.):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        return bool(
            seed.shape == lower.shape == upper.shape == (7,)
            and np.isfinite(np.r_[seed, lower, upper, margin]).all()
            and margin >= 0.
            and np.all(lower < upper)
            and np.all(seed >= lower + margin - 1e-9)
            and np.all(seed <= upper - margin + 1e-9)
        )

    def register_pregrasp_seed(self, track_id, joint_positions, *, scene_epoch) -> bool:
        """Copy a finite seven-joint guess within the env's hard joint limits.

        Return False for invalid keys, joints or unavailable limits. This is
        input validation only; it does not grant a collision/path certificate.
        The current planner margin is checked again when the seed is consumed.
        """
        key = self._pregrasp_seed_key(track_id, scene_epoch)
        if key is None:
            return False
        try:
            seed = np.asarray(joint_positions, dtype=float)
            limits = self._joint_safety_limits()
            if limits is None or not self._within_seed_limits(seed, *limits):
                return False
        except (TypeError, ValueError):
            return False
        stored = seed.copy()
        stored.setflags(write=False)
        self._network_pregrasp_seeds[key] = stored
        return True

    def remove_pregrasp_seed(self, track_id, *, scene_epoch) -> bool:
        """Remove a guess when its corresponding selected grasp is discarded."""
        key = self._pregrasp_seed_key(track_id, scene_epoch)
        return key is not None and self._network_pregrasp_seeds.pop(key, None) is not None

    def reset(self, seed=None):
        self._network_pregrasp_seeds.clear()
        self.closure_diagnostics.clear()
        self._motor_preload_diagnostics = None
        if self._following_lead is not None:
            self._following_lead.reset()
        self._following_lead_diagnostics = None
        super().reset(seed=seed)

    def _clear_target(self):
        self._motor_preload_diagnostics = None
        if self._following_lead is not None:
            self._following_lead.reset()
        self._following_lead_diagnostics = None
        return super()._clear_target()

    def _action(self, observation, *, target_pose, gripper):
        result = super()._action(observation, target_pose=target_pose, gripper=gripper)
        # Record issued joint FK against the actual requested target. This is
        # diagnostic only: it reveals successful IK backtracking whose first
        # rejected proposal no longer appears in last_error.
        now = self._timestamp_without_advancing(observation)
        if (target_pose is not None and self._stage in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
                and self.closure_diagnostics and abs(self.closure_diagnostics[-1]["sim_s"] - now) < 1e-9):
            forward = getattr(getattr(self.env, "ik", None), "forward_pose", None)
            q = np.asarray(result.get("arm"), dtype=float)
            if callable(forward) and q.shape == (7,) and np.isfinite(q).all():
                issued = np.asarray(forward(q), dtype=float)
                current = self._ee_pose(observation)
                requested = np.asarray(target_pose, dtype=float)[:3] - current[:3]
                denominator = float(requested @ requested)
                fraction = (None if denominator < 1e-12
                            else float((issued[:3] - current[:3]) @ requested / denominator))
                self.closure_diagnostics[-1].update(
                    requested_cartesian_pose=np.asarray(target_pose, dtype=float).copy(),
                    issued_joint_command_fk=issued,
                    admitted_translation_fraction=fraction,
                    admission_fraction_is_diagnostic_only=True)
        return result

    def _update_grasp_force_preload(self, observation, now, feedback):
        if self._minimum_motor_hold_effort <= 0.:
            return super()._update_grasp_force_preload(observation, now, feedback)
        # This is a measured motor-load contract for this simulator's declared
        # actuator transmission, NOT a synthesized pad-normal-force sensor.
        # Initial capture still requires the separate motor obstruction dwell.
        # Use the existing position preload rate, continuous settle time and
        # max-close watchdog. No class mass, geometry or contact truth enters.
        gripper = observation.get("gripper", {})
        compatible = isinstance(gripper, Mapping) and (
            gripper.get("motor_effort_units") == "simulated_actuator_force_N"
            and gripper.get("motor_force_mapping") == "single_joint_gear1_symmetry_coupled"
            and gripper.get("motor_coupling") == "single_symmetry_coupled_actuator")
        try:
            raw = np.asarray(gripper.get("motor_effort"), dtype=float)
            sign = float(gripper.get("motor_closing_effort_sign"))
            stamp = float(gripper.get("motor_telemetry_timestamp"))
            valid = (compatible and raw.shape == (1,) and np.isfinite(raw).all()
                     and sign in {-1., 1.} and np.isfinite(stamp)
                     and abs(float(now) - stamp) <= 1.5 * self.config.control_timestep)
        except (TypeError, ValueError, AttributeError):
            valid = False
        inward_effort = float(raw[0] * sign) if valid else None
        previous = self._grasp_force_last_update_at
        dt = (self.config.control_timestep if previous is None
              else min(self.config.control_timestep, max(0., float(now) - previous)))
        if previous is not None and (now < previous or now - previous > 1.5 * self.config.control_timestep + 1e-9):
            self._grasp_force_stable_since = None
        self._grasp_force_last_update_at = float(now)
        acquired = self._grasp_confirmed_at is not None
        above = bool(acquired and feedback is True and valid
                     and gripper.get("is_holding") is True
                     and inward_effort + 1e-9 >= self._minimum_motor_hold_effort)
        if not above:
            self._grasp_force_stable_since = None
            self._grasp_force_status = ("motor_telemetry_invalid" if not valid
                                       else "motor_preloading" if acquired else "waiting_for_motor_grasp")
            if acquired and valid and inward_effort < self._minimum_motor_hold_effort:
                span = self.config.gripper_open_action - self.config.gripper_close_action
                fraction = (self._grasp_gripper_action - self.config.gripper_close_action) / span
                fraction = float(np.clip(fraction - self.config.grasp_force_preload_rate * dt, 0., 1.))
                self._grasp_gripper_action = self.config.gripper_close_action + fraction * span
            ready = False
        else:
            if self._grasp_force_stable_since is None:
                self._grasp_force_stable_since = float(now)
            ready = now - self._grasp_force_stable_since + 1e-9 >= self.config.grasp_force_settle_time
            self._grasp_force_status = "motor_ready" if ready else "motor_settling"
        self._motor_preload_diagnostics = {
            "target_inward_motor_effort": self._minimum_motor_hold_effort,
            "motor_effort_units": "simulated_actuator_force_N",
            "measured_inward_motor_effort": inward_effort,
            "telemetry_valid": bool(valid), "grasp_acquired": acquired,
            "feedback_holding": feedback, "settle_since_s": self._grasp_force_stable_since,
            "status": self._grasp_force_status, "ready": bool(ready),
            "gripper_action_after_preload": self._grasp_gripper_action,
            "is_pad_normal_force_measurement": False,
        }
        if self.closure_diagnostics and abs(self.closure_diagnostics[-1]["sim_s"] - now) < 1e-9:
            self.closure_diagnostics[-1]["motor_preload"] = deepcopy(self._motor_preload_diagnostics)
        return bool(ready)

    def _visual_closure_intercept(self, target, observation, now):
        if self._visual_acquisition_tracking:
            result, evidence = self._relative_visual_closure_intercept(target, observation, now)
        else:
            result = super()._visual_closure_intercept(target, observation, now)
            evidence = {"contact_prediction_frame": "inherited_baseline"}
        self._record_execution_feedback(target, observation, now,
                                        gate_result=result, kind="visual_closure_gate")
        self.closure_diagnostics[-1].update(evidence)
        return result

    def _relative_visual_closure_intercept(self, target, observation, now):
        """Use target-to-TCP velocity whenever final visual following is on.

        The inherited gate applies this correction only for the separate
        dynamic-workspace planner. A static route can now also end in visual
        following: treating its TCP as stationary added object motion a
        second time and rejected a centred co-moving grasp indefinitely.
        This changes the reference frame, not the spatial capture envelope.
        """
        evidence = {"contact_prediction_frame": "relative_visual_minus_encoder_fk_tcp"}
        lead = self.config.closure_contact_lead_time
        visual = self._current_visual_acquisition(target, observation)
        if lead <= 0. or visual is None or self._grasp_pose is None:
            return None, evidence
        position, velocity = visual
        try:
            height = float(target.get("grasp_height"))
            tolerance = float(target.get("closure_horizontal_tolerance", self.config.closure_horizontal_tolerance))
        except (TypeError, ValueError):
            return None, evidence
        if not np.isfinite([height, tolerance]).all() or velocity[:2] @ velocity[:2] < 1e-6:
            # Preserve clock-based acquisition for a genuinely stopped belt.
            return None, evidence
        solver = getattr(self.env, "ik", getattr(self.env, "ik_solver", None))
        jacobian_fn = getattr(solver, "compute_jacobian", None)
        qd = np.asarray(observation.get("joint_vel", []), dtype=float)
        jacobian = (None if not callable(jacobian_fn)
                    else np.asarray(jacobian_fn(self._joint_positions(observation)), dtype=float))
        if (jacobian is None or jacobian.shape != (6, 7) or qd.shape != (7,)
                or not np.isfinite(jacobian).all() or not np.isfinite(qd).all()):
            evidence["contact_prediction_status"] = "encoder_fk_velocity_unavailable"
            # Once following is explicitly enabled, missing velocity is not
            # evidence for a stationary TCP and cannot bypass this gate.
            return (False, float(now) + lead), evidence
        tcp_velocity = jacobian[:3] @ qd
        relative_velocity = velocity - tcp_velocity
        contact = self._grasp_pose.copy()
        contact[:3] = position + relative_velocity * lead
        contact[2] = position[2] + height
        current = self._ee_pose(observation)
        ready = bool(np.linalg.norm(contact[:2] - current[:2]) <= min(.010, tolerance)
                     and self._closure_pose_reached(contact, observation, target)
                     and self._window_valid(target, observation))
        evidence.update({"tcp_velocity_world_m_s": tcp_velocity,
                         "closure_relative_velocity_world_m_s": relative_velocity,
                         "predicted_contact_pose": contact,
                         "contact_prediction_status": "valid"})
        return (ready, float(now) + lead), evidence

    def _grasp_feedback(self, target, observation):
        if self._stage not in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING} and self._following_lead is not None:
            self._following_lead.reset()
        result = super()._grasp_feedback(target, observation)
        self._record_execution_feedback(target, observation,
                                        self._timestamp_without_advancing(observation),
                                        grasp_feedback=result, kind="grasp_feedback")
        return result

    def _record_execution_feedback(self, target, observation, now, *,
                                   gate_result=None, grasp_feedback=None, kind):
        target = (target if isinstance(target, Mapping)
                  else self._target if isinstance(self._target, Mapping) else {})
        planned, planned_time, visible, track_age = self._planned_tracking_reference(target, observation)
        capture_timestamp = target.get("capture_timestamp", target.get("capture_sim_s"))
        if capture_timestamp is None:
            # Explicitly label this as derived, not an actual camera stamp.
            capture_timestamp = float(now) - track_age
        # Record the actual gate inputs/output, without running the gate again
        # or using object truth to influence the closing command.
        self.closure_diagnostics.append({
            "sim_s": float(now), "target_id": self.target_id,
            "diagnostic_kind": kind, "policy_stage": self._stage.value,
            "tcp_feedback": self._ee_pose(observation).copy(),
            "joint_feedback": np.asarray(observation.get("joint_pos", []), dtype=float).copy(),
            "joint_target_feedback": np.asarray(observation.get("joint_target", []), dtype=float).copy(),
            "joint_velocity": np.asarray(observation.get("joint_vel", []), dtype=float).copy(),
            "last_error_before_command": self._last_error,
            "visual_position": np.asarray(target.get("position", []), dtype=float).copy(),
            "visual_velocity": np.asarray(target.get("estimated_velocity", target.get("linear_velocity", [])), dtype=float).copy(),
            "visual_timestamp": target.get("timestamp"),
            "visual_visible": visible,
            "track_observation_age_s": track_age,
            "capture_timestamp": capture_timestamp,
            "capture_timestamp_source": ("camera" if "capture_timestamp" in target or "capture_sim_s" in target else "publication_minus_track_age"),
            "observation_timestamp": self._timestamp_without_advancing(observation),
            "planned_grasp_time": planned_time,
            "planned_grasp_pose": None if planned is None else planned.copy(),
            "planned_time_remaining_s": None if planned_time is None else planned_time - float(now),
            "grasp_window": target.get("grasp_window", target.get("window")),
            "gripper_feedback_source": observation.get("gripper", {}).get("feedback_source", "unspecified"),
            "gripper_motor_feedback": deepcopy(observation.get("gripper", {})),
            "grasp_feedback": grasp_feedback,
            "gripper_action_before_command": self._grasp_gripper_action,
            "motor_preload": deepcopy(self._motor_preload_diagnostics),
            "visual_acquisition_tracking": self._visual_acquisition_tracking,
            "visual_servo_bias": deepcopy(self._following_lead_diagnostics),
            "grasp_height_m": target.get("grasp_height"),
            "grasp_pose": None if self._grasp_pose is None else self._grasp_pose.copy(),
            "gate_result": gate_result,
        })

    def _pregrasp_goal_seeds(self, current, lower, upper, margin, *, target_id, scene_epoch):
        defaults = super()._pregrasp_goal_seeds(
            current, lower, upper, margin, target_id=target_id, scene_epoch=scene_epoch)
        key = self._pregrasp_seed_key(target_id, scene_epoch)
        preferred = self._network_pregrasp_seeds.get(key)
        if preferred is None or not self._within_seed_limits(preferred, lower, upper, margin):
            return defaults
        # No solver result or planner state is installed here. Retain every
        # inherited seed, including any certified branch and all fallbacks.
        return (preferred.copy(), *defaults)

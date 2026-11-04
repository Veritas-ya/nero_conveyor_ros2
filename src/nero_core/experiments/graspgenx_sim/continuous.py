"""Offline GraspGen-X adapters for the existing continuous sorting controller.

Only the evaluator reads simulator object truth.  The controller adapter takes
network poses and visual track positions supplied by the caller; it never uses
an object's MuJoCo pose to choose a grasp.  All motion remains ordinary actuator
control through BaselineSortingPolicy and ConveyorSortingEnv.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from experiments.graspgenx_sim.physical_containment import (
    CollisionGeometryBounds, catalog_bounds, containment_checks,
)


_POSE_KEYS = {
    "observed_pose", "current_estimated_pose", "predicted_grasp_pose",
    "planned_grasp_pose", "admitted_grasp_pose", "predicted_pose",
    "future_grasp_pose", "pose", "historical_pose",
}
_POSITION_KEYS = {
    "position", "observed_position", "predicted_position", "future_position",
}


@dataclass(frozen=True)
class GraspTrackOverride:
    track_id: int
    tcp_pose_world_wxyz: np.ndarray
    track_position_world: np.ndarray
    capture_timestamp: float
    closing_width_m: float
    score: float
    source: str = "graspgenx"

    def __post_init__(self):
        pose = np.asarray(self.tcp_pose_world_wxyz, dtype=float).copy()
        position = np.asarray(self.track_position_world, dtype=float).copy()
        if pose.shape != (7,) or position.shape != (3,):
            raise ValueError("TCP must be xyz+wxyz, track position must be xyz")
        if not np.isfinite(np.r_[pose, position, self.capture_timestamp,
                                  self.closing_width_m, self.score]).all():
            raise ValueError("grasp override values must be finite")
        norm = float(np.linalg.norm(pose[3:]))
        if norm < 1e-9 or not 0 <= self.closing_width_m <= 0.10:
            raise ValueError("require a valid quaternion and AGX width in [0, .10] m")
        pose[3:] /= norm
        pose.setflags(write=False)
        position.setflags(write=False)
        object.__setattr__(self, "tcp_pose_world_wxyz", pose)
        object.__setattr__(self, "track_position_world", position)

    @property
    def anchor_offset_world(self) -> np.ndarray:
        """Belt-plane displacement from visual track centre to network TCP."""
        result = self.tcp_pose_world_wxyz[:3] - self.track_position_world
        result = result.copy()
        result[2] = 0.0
        return result

    @property
    def grasp_height_m(self) -> float:
        return float(self.tcp_pose_world_wxyz[2] - self.track_position_world[2])


class _GraspEnvironmentProxy:
    """Translate the policy's anchor while preserving the raw visual tracker.

    Retiming hooks use the unmodified object's support-plane coordinates in
    the real environment.  Every returned pose is translated back to the
    network anchor.  This avoids bypassing motion compensation with a frozen
    exact_ee_pose and avoids overwriting a non-central grasp during closure.
    """
    def __init__(self, adapter: "ContinuousGraspAdapter"):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(self.adapter.env, name)

    def get_observation(self):
        return self.adapter.decorate_observation(self.adapter.env.get_observation())

    def get_object_state(self, track_id):
        value = self.adapter.env.get_object_state(track_id)
        return self.adapter.decorate_target(value, track_id) if value is not None else None

    get_target = get_object_state
    get_object = get_object_state

    def select_target(self, observation=None):
        raw = None if observation is None else self.adapter.undecorate_observation(observation)
        if raw is None:
            raw = self.adapter.undecorate_observation(self.get_observation())
        value = self.adapter.env.select_target(raw)
        if isinstance(value, Mapping):
            return self.adapter.decorate_target(value)
        return value

    select_sorting_target = select_target

    def get_predicted_grasp_pose(self, track_id, observation=None):
        raw = None if observation is None else self.adapter.undecorate_observation(observation)
        value = self.adapter.env.get_predicted_grasp_pose(track_id, raw)
        return self.adapter.shift_value(value, track_id) if value is not None else None

    predict_grasp_pose = get_predicted_grasp_pose

    def refine_selected_grasp_plan(self, track_id, *args, **kwargs):
        if kwargs.get("requested_pose") is not None:
            kwargs = dict(kwargs)
            kwargs["requested_pose"] = self.adapter.shift_value(
                kwargs["requested_pose"], track_id, direction=-1.0)
        result = self.adapter.env.refine_selected_grasp_plan(track_id, *args, **kwargs)
        return self.adapter.shift_mapping(result, track_id)

    retime_selected_grasp_plan = refine_selected_grasp_plan

    def validate_selected_grasp_plan(self, track_id, *args, **kwargs):
        result = self.adapter.env.validate_selected_grasp_plan(track_id, *args, **kwargs)
        return self.adapter.shift_mapping(result, track_id)

    def is_target_feasible(self, track_id, observation=None):
        raw = None if observation is None else self.adapter.undecorate_observation(observation)
        return self.adapter.env.is_target_feasible(track_id, raw)

    is_grasp_window_feasible = is_target_feasible
    target_window_feasible = is_target_feasible

    def is_target_graspable(self, track_id, observation=None):
        raw = None if observation is None else self.adapter.undecorate_observation(observation)
        return self.adapter.env.is_target_graspable(track_id, raw)


class ContinuousGraspAdapter:
    """Reuse continuous tracking, interception, planning and sorting stages.

    ``set_grasp`` is called by the inference owner with a visually associated
    track and a network grasp already converted to the model's AGX TCP frame.
    ``closing_width_m`` is an execution parameter, not an assumed extra model
    output: zero requests contact-limited full closure when the grasp network
    returns only SE(3) poses. Never use the template's open width as the close
    target. The existing contact preload/confirmation remains active.
    The network orientation remains fixed while the supported part translates
    along the belt.  Conveyor rotation/sliding rotation is outside this first
    adapter's contract and must be detected/reinferred by the caller.
    """
    def __init__(self, env, *, policy=None, compression_m=0.002,
                 require_network_grasp=True, enable_speculative_prefetch=False,
                 acquisition_translation_step_m=.010,
                 acquisition_command_integrator=False,
                 visual_acquisition_tracking=False,
                 minimum_motor_hold_effort=0.,
                 visual_servo_bias_compensation=False,
                 moving_approach_mode="high"):
        if getattr(env, "perception_mode", None) != "rendered_rgbd":
            raise ValueError("continuous grasp control requires rendered_rgbd visual tracks")
        if bool(getattr(env, "grasp_assist", False)):
            raise ValueError("grasp_assist must be disabled for physical grasp evaluation")
        self.env = env
        self.overrides: dict[int, GraspTrackOverride] = {}
        self.compression_m = float(compression_m)
        if not 0 <= self.compression_m <= .010:
            raise ValueError("compression_m must be between zero and 10 mm")
        self.require_network_grasp = bool(require_network_grasp)
        self.proxy = _GraspEnvironmentProxy(self)
        if policy is None:
            from scripts._common import make_policy
            from experiments.graspgenx_sim.policy import NetworkSeedSortingPolicy
            original = make_policy(self.proxy, compensation=True)
            original_config = original.config
            original.close()
            policy = NetworkSeedSortingPolicy(original_config, env=self.proxy,
                                              enable_speculative_prefetch=enable_speculative_prefetch,
                                              acquisition_translation_step_m=acquisition_translation_step_m,
                                              acquisition_command_integrator=acquisition_command_integrator,
                                              visual_acquisition_tracking=visual_acquisition_tracking,
                                              minimum_motor_hold_effort=minimum_motor_hold_effort,
                                              visual_servo_bias_compensation=visual_servo_bias_compensation)
        if moving_approach_mode not in {"high", "continuous"}:
            raise ValueError("moving_approach_mode must be high or continuous")
        if moving_approach_mode == "continuous":
            # Reuse the certified complete-curve controller already exercised
            # by continuous_intercept_sim; do not build a new local follower.
            policy.config = replace(policy.config, moving_approach_mode="continuous",
                dynamic_intercept_tracking=True, timed_descent_enabled=False,
                pregrasp_fixed_high_enabled=False)
            if hasattr(policy, "_enable_network_speculative_prefetch"):
                policy._enable_network_speculative_prefetch = False
            # A plain high-mode scene has no dynamic servo corridor. Keep
            # its existing grasp region instead of silently installing the
            # legacy dynamic profile's unrelated +20..+180 mm default when
            # the continuous curve hands off to visual closure tracking.
            workspace = getattr(env, "scene_config", {}).get("workspace", {})
            dynamic = workspace.setdefault("dynamic_intercept", {})
            if "tracking_x_range" not in dynamic and "x_range" not in dynamic:
                bounds = workspace.get("grasp_region", {}).get("x")
                if bounds is not None:
                    dynamic["tracking_x_range"] = list(bounds)
        self._moving_approach_mode = moving_approach_mode
        self.policy = policy
        self.policy.env = self.proxy
        self._scene_epoch = None
        self.updates: list[dict[str, Any]] = []

    @property
    def ready_track_ids(self):
        return tuple(sorted(self.overrides))

    def set_grasp(self, track_id, tcp_pose_world_wxyz, track_position_world,
                  capture_timestamp, closing_width_m, score, source="graspgenx",
                  *, replace_active=False) -> bool:
        identifier = int(track_id)
        # Freeze an admitted grasp to avoid invalidating an in-flight path.
        if (not replace_active and identifier in self.overrides
                and getattr(self.policy, "target_id", None) == identifier):
            return False
        value = GraspTrackOverride(identifier, tcp_pose_world_wxyz,
                                   track_position_world, float(capture_timestamp),
                                   float(closing_width_m), float(score), str(source))
        self.overrides[identifier] = value
        self.updates.append({"track_id": identifier, "capture_timestamp": value.capture_timestamp,
                             "tcp_pose_world_wxyz": value.tcp_pose_world_wxyz.tolist(),
                             "track_position_world": value.track_position_world.tolist(),
                             "closing_width_m": value.closing_width_m, "score": value.score,
                             "source": value.source})
        return True

    def remove_grasp(self, track_id):
        self.overrides.pop(int(track_id), None)

    def shift_value(self, value, track_id, direction=1.0):
        result = np.asarray(value, dtype=float).copy()
        override = self.overrides.get(int(track_id))
        if override is not None and result.shape in {(3,), (7,)}:
            result[:3] += direction * override.anchor_offset_world
        return result

    def shift_mapping(self, value, track_id, direction=1.0):
        result = dict(value)
        for key in _POSE_KEYS | _POSITION_KEYS:
            if result.get(key) is not None:
                result[key] = self.shift_value(result[key], track_id, direction)
        return result

    def decorate_target(self, target, track_id=None):
        if not isinstance(target, Mapping):
            return target
        identifier = int(target.get("object_id", target.get("track_id", track_id)))
        override = self.overrides.get(identifier)
        result = self.shift_mapping(target, identifier)
        if override is not None:
            for key in ("close_lead_time", "close_settle_time", "max_close_time",
                        "contact_hold_squeeze_fraction", "contact_hold_max_relaxation_fraction",
                        "confirmed_grasp_settle_time", "closure_requires_arm_settle",
                        "closure_horizontal_tolerance", "closure_vertical_tolerance",
                        "closure_orientation_tolerance", "allow_predictive_closure_outside_window"):
                # Network execution uses the common controller timing/gates.
                # A classifier category must not silently install old per-part
                # capture rules while the pose is advertised as generated.
                result.pop(key, None)
            result.update({
                "grasp_height": override.grasp_height_m,
                "recommended_grip_width": override.closing_width_m,
                "grip_width_margin": self.compression_m,
                "grasp_quaternion": override.tcp_pose_world_wxyz[3:].copy(),
                "grasp_yaw_mode": "network_6d",
                "grasp_candidates": [],
                "grasp_source": override.source,
                "grasp_network_score": override.score,
                "grasp_capture_timestamp": override.capture_timestamp,
            })
            # A caller may have cached exact poses; a stale exact world pose
            # would bypass the moving anchor and must not reach the policy.
            for key in ("predicted_ee_grasp_pose", "ee_grasp_pose", "gripper_grasp_pose"):
                result.pop(key, None)
        return result

    def decorate_observation(self, observation):
        result = dict(observation)
        result["objects"] = [self.decorate_target(item) for item in observation.get("objects", [])
                             if not self.require_network_grasp
                             or int(item["object_id"]) in self.overrides]
        for field in ("predictions", "target_predictions", "motion_predictions"):
            if isinstance(observation.get(field), Mapping):
                result[field] = {key: self.shift_mapping(value, int(key))
                                 if isinstance(value, Mapping) else self.shift_value(value, int(key))
                                 for key, value in observation[field].items()}
        if self._moving_approach_mode == "continuous":
            # The historical subclass already publishes CURRENT XY during
            # acquisition. An explicit current TCP avoids the generic planner
            # subtracting a future-prediction interval from it a second time.
            targets = {int(item["object_id"]): item for item in result["objects"]}
            for key, prediction in result.get("predictions", {}).items():
                target, grasp = targets.get(int(key)), self.overrides.get(int(key))
                if target is None or grasp is None or not isinstance(prediction, dict):
                    continue
                position = np.asarray(target["position"], dtype=float).copy()
                position[2] += grasp.grasp_height_m
                prediction["current_ee_grasp_pose"] = np.r_[position, grasp.tcp_pose_world_wxyz[3:]]
            result["dispatch_clock"] = {
                "scheduled_dispatch_policy_time_s": float(observation["timestamp"])}
        return result

    def undecorate_observation(self, observation):
        result = dict(observation)
        result["objects"] = [self.shift_mapping(item, int(item["object_id"]), -1.0)
                             for item in observation.get("objects", [])]
        for field in ("predictions", "target_predictions", "motion_predictions"):
            if isinstance(observation.get(field), Mapping):
                result[field] = {key: self.shift_mapping(value, int(key), -1.0)
                                 if isinstance(value, Mapping) else self.shift_value(value, int(key), -1.0)
                                 for key, value in observation[field].items()}
        return result

    def act(self, raw_observation):
        epoch = raw_observation.get("scene_epoch")
        if self._scene_epoch is not None and epoch != self._scene_epoch:
            self.reset()
        self._scene_epoch = epoch
        return self.policy.act(self.decorate_observation(raw_observation), self.proxy)

    def reset(self, seed=None):
        self.overrides.clear()
        self.updates.clear()
        self._scene_epoch = None
        self.policy.reset(seed=seed)

    def close(self):
        self.policy.close()


class ContinuousSortingEvaluator:
    """Physical placement metric independent of policy and lifecycle success.

    Ground-truth reads are confined to this class.  The denominator is every
    spawned object, including objects which never obtain a network candidate.
    Success requires a released part to remain inside its assigned bin below
    the rim, move slowly, and have no finger contact for the full settle time.
    """
    def __init__(self, env, *, settle_time_s=.5, maximum_speed_m_s=.08,
                 maximum_angular_speed_rad_s=1.0, terminal_retention_s=2.0,
                 retain_sorted_until_evaluated=False):
        self.env = env
        self.settle_time_s = float(settle_time_s)
        self.maximum_speed_m_s = float(maximum_speed_m_s)
        self.maximum_angular_speed_rad_s = float(maximum_angular_speed_rad_s)
        self.terminal_retention_s = max(float(terminal_retention_s), self.settle_time_s + .5)
        self.retain_sorted_until_evaluated=bool(retain_sorted_until_evaluated)
        self.records: dict[int, dict[str, Any]] = {}
        self._stable_since: dict[int, float] = {}
        self._object_specs = {int(spec.category_id): spec for spec in env.object_specs}
        self._bin_geoms = {int(box["id"]): tuple(
            index for index in range(env.model.ngeom)
            if (env.model.geom(index).name or "").startswith(f"bin_{int(box['id'])}_"))
            for box in env._bins}
        self._collision_bounds = CollisionGeometryBounds(env.model)
        self._physical_bin_bounds = {}
        # The ordinary environment recycles a lifecycle SORTED object after
        # 0.4 s, before the stricter independent settling test can finish.
        # Delay only that terminal parking operation in this isolated run;
        # do not weld, reposition, or otherwise help a nonterminal grasp.
        self._original_recycle = env.recycle_terminal_objects
        def delayed_recycle(*, older_than=.4):
            options={}
            if self.retain_sorted_until_evaluated:
                options['retain_object_ids']={int(r.logical_id) for r in env._objects
                    if r.logical_id is not None and r.state=='SORTED'
                    and not self.records.get(int(r.logical_id),{}).get('success',False)}
            return self._original_recycle(older_than=max(float(older_than), self.terminal_retention_s),**options)
        self._delayed_recycle = delayed_recycle
        env.recycle_terminal_objects = delayed_recycle

    def close(self):
        if self.env.recycle_terminal_objects is self._delayed_recycle:
            self.env.recycle_terminal_objects = self._original_recycle

    def update(self):
        now = float(self.env.data.time)
        pairs = None
        for runtime in self.env._objects:
            if runtime.logical_id is None:
                continue
            identifier = int(runtime.logical_id)
            record = self.records.setdefault(identifier, {
                "object_id": identifier, "category_id": int(runtime.category_id),
                "target_bin_id": int(runtime.bin_id), "spawned_at_s": float(runtime.spawned_at),
                "grasped": False, "released": False, "success": False,
                "placed_at_s": None, "stable_at_s": None,
            })
            record["grasped"] = bool(record["grasped"] or runtime.grasped_at is not None)
            record["released"] = bool(record["released"] or runtime.released_at is not None)
            visual_id = getattr(self.env, "_logical_to_vision_evaluation", {}).get(identifier)
            if visual_id is not None:
                record["visual_track_id"] = int(visual_id)
                visual_state = getattr(self.env, "_vision_task_states", {}).get(int(visual_id))
                if visual_state is not None:
                    visual_category = int(visual_state.category_id)
                    record["visual_category_id"] = visual_category
                    record["classification_correct"] = visual_category == int(runtime.category_id)
                    visual_spec = self._object_specs.get(visual_category)
                    if visual_spec is not None:
                        record["visual_assigned_bin_id"] = int(visual_spec.bin_id)
                    if runtime.grasped_at is not None:
                        record.setdefault("visual_category_at_grasp", visual_category)
            record["lifecycle_state"] = str(runtime.state)
            record["failure_reason"] = runtime.failure_reason
            if record["success"]:
                continue
            position = self.env.data.xpos[runtime.body_id].copy()
            record["last_coarse_bin_id"] = self.env._bin_containing(position)
            record["last_position_world_m"] = position.tolist()
            if not record["released"]:
                # A part cannot begin the continuous placement interval before
                # release. Keep lifecycle/classification/position current, but
                # defer hull transforms and contact diagnostics until release.
                record["placement_checks"] = {
                    "status": "pending_release", "geometry_evaluated": False,
                    "instantaneously_stable": False,
                }
                self._stable_since.pop(identifier, None)
                continue
            if pairs is None:
                pairs = self.env._contact_pairs()
            velocity = self.env.data.qvel[runtime.qvel_adr:runtime.qvel_adr + 3]
            angular_velocity = self.env.data.qvel[runtime.qvel_adr + 3:runtime.qvel_adr + 6]
            box = next(item for item in self.env._bins if int(item["id"]) == int(runtime.bin_id))
            spec = self._object_specs[int(runtime.category_id)]
            dimensions = np.asarray(spec.dimensions, dtype=float)
            rotation = self.env.data.xmat[runtime.body_id].reshape(3, 3)
            legacy_bounds = catalog_bounds(position, rotation, dimensions)
            part_bounds = self._collision_bounds.object_bounds(
                self.env.data, runtime.geom_ids, legacy_bounds)
            bin_id = int(runtime.bin_id)
            if bin_id not in self._physical_bin_bounds:
                self._physical_bin_bounds[bin_id] = self._collision_bounds.bin_interior(
                    self.env.data, box, self._bin_geoms[bin_id])
            interior = self._physical_bin_bounds[bin_id]
            geometry_checks = containment_checks(part_bounds, interior)
            inside_xy, below_rim, above_floor = (geometry_checks[key] for key in
                                                ("inside_xy", "below_rim", "above_floor"))
            bilateral = self.env._both_fingers_touch(runtime, pairs)
            touching_finger = any((min(f, g), max(f, g)) in pairs
                                  for f in self.env.finger_geom_ids for g in runtime.geom_ids)
            bin_geoms = self._bin_geoms[int(runtime.bin_id)]
            supported_by_bin = any((min(b, g), max(b, g)) in pairs
                                   for b in bin_geoms for g in runtime.geom_ids)
            stable = bool(record["released"] and inside_xy and below_rim and above_floor
                          and np.linalg.norm(velocity) <= self.maximum_speed_m_s
                          and np.linalg.norm(angular_velocity) <= self.maximum_angular_speed_rad_s
                          and supported_by_bin and not bilateral and not touching_finger)
            # Preserve the actual failed condition. A lifecycle SORTED label
            # alone cannot distinguish a wrong class, a protruding part or a
            # part which has not physically settled on the bin.
            record["placement_checks"] = {
                "status": "evaluated_after_release", "geometry_evaluated": True,
                "inside_xy": inside_xy, "below_rim": below_rim,
                "above_floor": above_floor, "supported_by_bin": supported_by_bin,
                "touching_finger": touching_finger, "bilateral_finger_contact": bilateral,
                "linear_speed_m_s": float(np.linalg.norm(velocity)),
                "angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
                "collision_bounds": part_bounds.diagnostics(),
                "target_bin_interior": interior.diagnostics(),
                "containment_tolerance_m": .003,
                "floor_tolerance_m": .006,
                # These legacy fields are diagnostics only. A rotated cuboid
                # overestimates a round part's support along world axes.
                "bbox_center_world_m": ((legacy_bounds.lower + legacy_bounds.upper) / 2).tolist(),
                "bbox_half_extents_world_m": ((legacy_bounds.upper - legacy_bounds.lower) / 2).tolist(),
                "catalog_bbox_diagnostic": {
                    **legacy_bounds.diagnostics(),
                    **containment_checks(legacy_bounds, interior),
                    "used_for_scoring": part_bounds.source == "catalog_bbox_compatibility_fallback",
                },
                "instantaneously_stable": stable,
            }
            if stable:
                since = self._stable_since.setdefault(identifier, now)
                record["longest_stable_duration_s"] = max(
                    float(record.get("longest_stable_duration_s", 0.)), now - since)
                if now - since >= self.settle_time_s - 1e-9:
                    record["success"] = True
                    record["placed_at_s"] = float(since)
                    record["stable_at_s"] = now
                    record["spawn_to_stable_s"] = now - record["spawned_at_s"]
            else:
                self._stable_since.pop(identifier, None)

    def summary(self, *, expected_objects=None):
        from scipy.stats import beta
        records = [dict(item) for item in self.records.values()]
        for item in records:
            reasons = []
            if not item["success"]:
                if not item["grasped"]:
                    reasons.append("not_grasped")
                elif not item["released"]:
                    reasons.append("not_released")
                else:
                    coarse_bin = item.get("last_coarse_bin_id")
                    if coarse_bin is not None and int(coarse_bin) != int(item["target_bin_id"]):
                        reasons.append("released_in_wrong_bin")
                    checks = item.get("placement_checks", {})
                    for check, failure in (("inside_xy", "collision_geometry_not_inside_target_bin"),
                                           ("below_rim", "collision_geometry_above_bin_rim"),
                                           ("above_floor", "collision_geometry_below_bin_floor"),
                                           ("supported_by_bin", "no_target_bin_contact")):
                        if not checks.get(check, False):
                            reasons.append(failure)
                    if checks.get("touching_finger", False):
                        reasons.append("still_touching_finger")
                    if checks.get("linear_speed_m_s", float("inf")) > self.maximum_speed_m_s:
                        reasons.append("linear_speed_above_settle_limit")
                    if checks.get("angular_speed_rad_s", float("inf")) > self.maximum_angular_speed_rad_s:
                        reasons.append("angular_speed_above_settle_limit")
                    if not reasons:
                        reasons.append("continuous_settle_interval_not_completed")
            item["physical_failure_reasons"] = reasons
        observed = len(records)
        denominator = max(observed, int(expected_objects or observed))
        successes = sum(bool(item["success"]) for item in records)
        lower = 0.0 if successes == 0 else float(beta.ppf(.05, successes, denominator - successes + 1))
        times = sorted(float(item["stable_at_s"]) for item in records if item["success"])
        completion_intervals = np.diff(times)
        end_time = float(self.env.data.time)
        return {
            "metric": "released_in_correct_bin_and_stable",
            "geometry_metric_version": "collision_hull_physical_bin_v1",
            "containment_tolerance_m": .003,
            "floor_tolerance_m": .006,
            "bin_interiors": {str(identifier): bounds.diagnostics()
                              for identifier, bounds in self._physical_bin_bounds.items()},
            "expected_objects": denominator, "spawned_objects": observed,
            "not_spawned_objects": denominator - observed,
            "successful_sorted_objects": successes,
            "failed_or_incomplete_objects": denominator - successes,
            "grasped_but_not_sorted": sum(bool(item["grasped"] and not item["success"]) for item in records),
            "ungrasped_objects": sum(not bool(item["grasped"]) for item in records),
            "classification_error_objects": sum(item.get("classification_correct") is False for item in records),
            "released_wrong_bin_objects": sum("released_in_wrong_bin" in item["physical_failure_reasons"] for item in records),
            "success_rate": successes / denominator if denominator else 0.0,
            "one_sided_95_percent_lower_bound": lower,
            "elapsed_sim_s": end_time,
            "throughput_sorted_per_minute": 60 * successes / end_time if end_time else 0.0,
            "mean_completion_interval_s": float(np.mean(completion_intervals)) if len(completion_intervals) else None,
            "p95_completion_interval_s": float(np.percentile(completion_intervals, 95)) if len(completion_intervals) else None,
            "settle_time_s": self.settle_time_s,
            "maximum_speed_m_s": self.maximum_speed_m_s,
            "maximum_angular_speed_rad_s": self.maximum_angular_speed_rad_s,
            "terminal_retention_s": self.terminal_retention_s,
            "retain_sorted_until_independently_evaluated":self.retain_sorted_until_evaluated,
            "objects": records,
        }

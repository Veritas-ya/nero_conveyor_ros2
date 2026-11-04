"""Measured-state adapter for the existing complete sorting policy.

No SDK, camera, simulator step, workpiece truth, or actuator transmission lives
here. MuJoCo is a kinematic/collision mirror; all state advances come from the
caller. Motor width/force is explicitly a grasp proxy, never pad-contact truth.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from ..calibration.geometry import pose_xyz_rpy, transform_distance
from ..controllers.baseline_sorting_policy import BaselinePolicyConfig, BaselineSortingPolicy
from ..controllers.ik_solver import DampedLeastSquaresIK
from ..controllers.planning_worker import AsyncPlanningWorker, MuJoCoGlobalPlanningEngine
from ..controllers.robot_controller import RobotController
from ..controllers.joint_response import JointResponseModel
from .feedback_clock import mapped_stamp
from .types import identity_dict, task_lifecycle
from .single_pick import PickRejected, PickSettings, RobotModel


def _pose(transform):
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("TCP transform must be a finite 4x4 matrix")
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return np.r_[matrix[:3, 3], quat]


def _feedback_stamp_bounds(state, stream):
    """Use recorded packet stamps only; a policy tick is not a sensor stamp."""
    suffix = '_stamps_monotonic' if 'feedback_time' in state else '_stamps'
    stamps = state.get(stream + suffix)
    if stamps is None:
        stamp = (mapped_stamp(state,stream) if 'feedback_time' in state
                 else state.get(stream + '_stamp'))
        if stamp is None:
            return None
        stamps = [stamp]
    try:
        values = np.asarray(stamps, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        return None
    return float(values.min()), float(values.max())


def _copy_geometry(source):
    """Share immutable hulls/point arrays, isolate every mutable MuJoCo state."""
    result = copy.copy(source)
    result.model = copy.copy(source.model)
    result.data = source.mj.MjData(result.model)
    result.data.qpos[:] = source.data.qpos
    result.ik = DampedLeastSquaresIK(result.model, result.data, source.names,
                                    'ee_site', source.ik.config)
    result.ik.lower_limits = source.ik.lower_limits.copy()
    result.ik.upper_limits = source.ik.upper_limits.copy()
    result.ik.center_positions = source.ik.center_positions.copy()
    result.scene = source.scene.copy(); result.scene.setflags(write=False)
    result.target = source.target.copy(); result.target.setflags(write=False)
    from scipy.spatial import cKDTree
    result.tree = cKDTree(result.scene)
    source.mj.mj_forward(result.model, result.data)
    return result


def prepare_planning_roadmap(geometry, settings, *, initial_state=None, cancelled=None):
    """Pay static PRM construction once, before image capture/recognition.

    No workpiece pose or previous executed trajectory is an input. The nominal
    joint model and 100 mm open-jaw snapshot define static route suggestions;
    each future worker request still validates its own scene/object geometry.
    This helper does not modify the live robot, geometry state or limits.
    """
    from .observed_convex_batch import prepare_observed_convex_batch
    geometry._observed_convex_batch_preparation = prepare_observed_convex_batch()
    if not getattr(settings, 'warm_roadmap_enabled', True):
        geometry._prepared_warm_roadmap = None
        geometry._warm_roadmap_preparation = {'enabled': False, 'reason': 'disabled_for_ablation'}
        return None
    existing = getattr(geometry, '_prepared_warm_roadmap', None)
    if existing is not None and existing.ready:
        return existing
    try:
        from ..controllers.warm_roadmap import prepare_mujoco_roadmap
        snapshot = geometry.data.qpos.copy()
        if initial_state is not None:
            snapshot[geometry.indices] = np.asarray(initial_state['joints'], dtype=float)
        gripper = tuple(int(geometry.model.joint(name).qposadr[0])
                        for name in ('gripper_joint1', 'gripper_joint2'))
        snapshot[list(gripper)] = (float(settings.open_width_m)/2., -float(settings.open_width_m)/2.)
        limits = np.asarray(geometry.nominal_limits, dtype=float)
        bodies = tuple(sorted({int(geometry.model.geom_bodyid[g]) for g in geometry.meshes}))
        roadmap = prepare_mujoco_roadmap(
            geometry.model, joint_names=geometry.names, full_qpos=snapshot,
            lower_limits=limits[:, 0], upper_limits=limits[:, 1],
            robot_collision_body_ids=bodies, ee_site_name='ee_site',
            gripper_qpos_indices=gripper, cancelled=cancelled,
        )
        geometry._prepared_warm_roadmap = roadmap if roadmap.ready else None
        geometry._warm_roadmap_preparation = {'enabled': True, **roadmap.diagnostics}
        return geometry._prepared_warm_roadmap
    except Exception as error:
        geometry._prepared_warm_roadmap = None
        geometry._warm_roadmap_preparation = {
            'enabled': True, 'ready': False, 'reason': f'{type(error).__name__}: {error}',
            'fallback': 'original_direct_and_rrt_planner',
        }
        return None


class _JawCommandMirror:
    """Record policy command history without claiming physical jaw movement."""
    def __init__(self, width):
        self.target_open_fraction = float(width) / .1
        self.measured_open_fraction = float(width) / .1

    def command(self, value):
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("gripper action must lie in [-1, 1]")
        self.target_open_fraction = .5 * (float(value) + 1.)

    def get_state(self):
        return {'open_fraction': self.measured_open_fraction,
                'target_open_fraction': self.target_open_fraction}


class NeroPolicyContext:
    """One stopped-belt target through the unmodified policy stage sequence."""
    def __init__(self, geometry: RobotModel, detection: Mapping,
                 initial_state: Mapping, settings: PickSettings):
        settings.validate()
        self.target_identity = detection.get('target_identity')
        self.settings = settings
        self.geometry = geometry
        geometry.scene_collision_backend = getattr(settings, 'scene_collision_backend', 'auto')
        geometry.simd_broadphase_enabled = getattr(settings, 'simd_broadphase_enabled', True)
        self._candidate_backends = []
        self._scene_indexes = []
        self.model, self.data, self.ik = geometry.model, geometry.data, geometry.ik
        geometry.check_feedback(initial_state)
        self.joint_limit_report = geometry.configure_start_joint_tolerance(
            initial_state['joints'], settings.initial_joint_tolerance_deg)
        geometry.check_state(initial_state['joints'], initial_state['width_m'],
                             allow_finger_contact=True)
        self.grasp_transform = np.asarray(detection['grasp'], dtype=float).copy()
        self.grasp_pose = _pose(self.grasp_transform)
        self.wait_pose = self.grasp_pose.copy(); self.wait_pose[2] += .10
        self.lift_pose = self.grasp_pose.copy(); self.lift_pose[2] += settings.lift_m
        self.scene_epoch = 0
        self.complete = False
        self.failed_reason = None
        self.events = []
        self._reserved = False
        self._released = False
        self._last_joint_stamp = None
        self._last_joint_position = None
        self._joint_velocity = np.zeros(7)
        self._last_gripper_stamp = None
        self._gripper_observed_at = None
        self._proxy_since = None
        self._proxy_confirmed = False
        self._last_observation = None
        self._edge_validation_cache = {}
        self.geometry_validation_stats = {'edge_requests': 0, 'edge_cache_hits': 0,
                                          'state_checks': 0}
        self.motion_validation_rejection = {}
        self._workers = []
        self.gripper_controller = _JawCommandMirror(initial_state['width_m'])
        self.robot_controller = RobotController(
            self.model, self.data,
            {'joint_names': tuple(geometry.names), 'ee_site_name': 'ee_site',
             'action_mode': 'joint_position',
             'control_timestep': 1. / settings.control_hz,
             'joint_command_margin': geometry.joint_margin_rad,
             'max_joint_velocity': settings.joint_velocity_limit,
             'max_joint_acceleration': settings.full_policy_joint_acceleration_limits},
            gripper=self.gripper_controller,
        )
        self.max_joint_velocity = self.robot_controller.max_joint_velocity.copy()
        self.max_joint_acceleration = self.robot_controller.max_joint_acceleration.copy()
        self.robot_config = {'gripper': {'open_width': .1, 'closed_width': 0.},
                             'calibration_poses': {}, 'trajectory': {}}
        self.scene_config = {'workspace': {'dynamic_intercept': {'enabled': False}}}
        self._validation_geometry = _copy_geometry(geometry)
        self._target = {'object_id': (self.target_identity.object_id if self.target_identity is not None
                                       else 'measured-stationary-part'), 'category_id': 0,
                        'bin_id': 0, 'status': 'READY_TO_PLAN',
                        'pose': self.grasp_pose.copy(), 'ee_grasp_pose': self.grasp_pose.copy(),
                        'position': self.grasp_pose[:3].copy(), 'velocity': np.zeros(3),
                        'confidence': 1., 'closure_requires_arm_settle': True,
                        'confirmation_source': 'measured_rgbd_stationary_target'}
        timeout = max(30., 6. / settings.joint_velocity_limit)
        self.policy = BaselineSortingPolicy(
            {'action_mode': 'joint_position',
             'enable_motion_compensation': False, 'dynamic_intercept_tracking': False,
             # When the moving adapter enables dynamic PREGRASP, its unloaded
             # certified high route uses the original joint velocity/acceleration
             # profile without an additional TCP angular-time lower bound.
             # The overall TCP limiter stays on for descent and payload motion.
             'dynamic_pregrasp_global_tcp_speed_limits_enabled': False,
             'control_timestep': 1. / settings.control_hz,
             'grasp_quaternion': tuple(self.grasp_pose[3:]),
             'place_quaternion': tuple(self.grasp_pose[3:]),
             'wait_pose': tuple(self.wait_pose), 'approach_height': .10,
             'place_clearance': 0., 'bin_poses': {0: tuple(self.grasp_pose)},
             'safe_transfer_height': float(self.lift_pose[2]),
             'lift_clearance_height': 0., 'release_retreat_distance': 0., 'prefetch_handoff_retreat_distance': 0.,
             'require_grasp_confirmation': True,
             'grasp_confirmation_source': 'width_and_force',
             'adaptive_grip_width': False, 'minimum_grasp_normal_force': 0.,
             'confirmed_grasp_settle_time': .4, 'max_close_time': 6.,
             'track_belt_during_close': False,
             'tracking_velocity_feedforward_time': 0., 'closing_tracking_lead_time': 0.,
             'enable_global_transfer_planner': True, 'enable_time_parameterization': True,
             'planning_mode': settings.planning_mode, 'selection_policy': settings.selection_policy,
             'replan_policy': settings.replan_policy,
             'full_plan_observation_max_age_s': settings.full_plan_observation_max_age_s,
             'full_plan_close_window_s': settings.full_plan_close_window_s,
             'full_plan_compute_budget_s': settings.full_plan_compute_budget_s,
             'joint_trajectory_backend': settings.joint_trajectory_backend,
             'trajectory_jerk_limits': (None if settings.joint_jerk_limit_rad_s3 is None
                                        else (settings.joint_jerk_limit_rad_s3,) * 7),
             'trajectory_velocity_limits': (settings.joint_velocity_limit,) * 7,
             'trajectory_acceleration_limits': settings.full_policy_joint_acceleration_limits,
             'gentle_carry_categories': (),
             'minimum_joint_limit_margin': (geometry.joint_margin_rad
                                            if self.joint_limit_report['expanded'] else .035),
             'pregrasp_tracking_joint_margin_reserve': (0.
                                            if self.joint_limit_report['expanded'] else .01),
             'position_tolerance': .002, 'orientation_tolerance': .012,
             'closure_horizontal_tolerance': .002, 'closure_vertical_tolerance': .002,
             'closure_orientation_tolerance': .012,
             # Match the observed-scene 8 mrad edge sampling in the shared
             # worker, so its certificate covers the same density as live checks.
             'planner_edge_resolution': .008 / float(np.max(geometry.limits[:, 1]-geometry.limits[:, 0])),
             'planner_corner_settle_tolerance': .003,
             # Use the shared 18 mrad range throughout unloaded dynamic
             # PREGRASP. Actual feedback connectors are still checked at each
             # corner; static/loaded paths retain their original setting.
             'pregrasp_endpoint_settle_tolerance': BaselinePolicyConfig().planner_corner_settle_tolerance,
             'pregrasp_corner_settle_tolerance': BaselinePolicyConfig().planner_corner_settle_tolerance,
             'carry_path_cross_track_enter_limit': .003, 'carry_path_cross_track_exit_limit': .002,
             'planner_waypoint_velocity_tolerance': .03,
             'acquisition_path_tracking_error_limit': .055,
             'global_path_tracking_error_limit': .055,
             'carry_path_tracking_error_limit': .055,
             'attempt_timeout': timeout * 6, 'pregrasp_timeout': timeout,
             'descent_timeout': timeout, 'lift_timeout': timeout,
             'transfer_timeout': timeout, 'return_timeout': timeout,
             'recovery_timeout': timeout,
             'cartesian_linear_speed': min(.10, settings.joint_velocity_limit * .15),
             'cartesian_angular_speed': min(.8, settings.joint_velocity_limit)}, env=self)

        # The model is immutable during a run. Unknown or diagnostic-only
        # fits leave the original reference in place; loading is not a new
        # task admission condition.
        self.joint_response_model = None
        self.joint_response_report = {'requested': settings.joint_response_model_path,
                                      'loaded': False, 'status': 'original_reference_no_model'}
        if settings.joint_response_model_path:
            try:
                with open(settings.joint_response_model_path, 'rb') as handle:
                    contents = handle.read()
                response = JointResponseModel.from_dict(json.loads(contents))
                if response.metadata.get('simulation_only', False):
                    raise ValueError('仿真响应模型不能用于现场机械臂')
                if len(response.axes) != 7:
                    raise ValueError('位置响应模型必须对应七个关节')
                self.joint_response_model = response
                self.joint_response_report.update(loaded=True,
                    status='loaded_reference_proposal_only',
                    runtime_eligible=bool(response.metadata.get('runtime_eligible', False)),
                    sha256=hashlib.sha256(contents).hexdigest(), metadata=response.metadata)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.joint_response_report.update(status='original_reference_model_unavailable',
                                                  reason=str(error))

    def observe(self, state: Mapping, timestamp: float, *, joint_velocity=None):
        # Proof reuse never crosses a new feedback/vision publication, even
        # when two incoming samples happen to have identical joint values.
        self._edge_validation_cache.clear()
        q = np.asarray(state['joints'], dtype=float)
        width, force = float(state['width_m']), float(state['force_n'])
        if q.shape != (7,) or not np.isfinite(np.r_[q, width, force, timestamp]).all():
            raise PickRejected('完整策略收到无效实测状态')
        try:
            sdk_flange = pose_xyz_rpy(state['pose'])
        except (KeyError, TypeError, ValueError):
            raise PickRejected('完整策略收到无效法兰反馈：须为六个有限值') from None
        joint_stamp = mapped_stamp(state, legacy=timestamp)
        if joint_stamp is None:
            raise PickRejected('完整策略收到无效反馈时间映射')
        joint_stamp = float(joint_stamp)
        if joint_velocity is not None:
            velocity = np.asarray(joint_velocity, dtype=float)
            if velocity.shape != (7,) or not np.isfinite(velocity).all():
                raise PickRejected('完整策略收到无效关节速度')
            self._joint_velocity = velocity.copy()
        elif self._last_joint_stamp is not None and joint_stamp > self._last_joint_stamp:
            self._joint_velocity = (q - self._last_joint_position) / (joint_stamp - self._last_joint_stamp)
        if self._last_joint_stamp is None or joint_stamp > self._last_joint_stamp:
            self._last_joint_stamp = joint_stamp
            self._last_joint_position = q.copy()
        # Initial/preflight checks retain strict model/FK consistency. During
        # motion, independent joint and flange CAN streams describe different
        # instants: their difference is evidence, not a synchronous fault test.
        self.geometry.set_state(q, width)
        position_error, orientation_error = transform_distance(self.geometry.flange(), sdk_flange)
        joint_bounds = _feedback_stamp_bounds(state, 'joint')
        pose_bounds = _feedback_stamp_bounds(state, 'pose')
        packet_span = stamp_delta = None
        if joint_bounds is not None and pose_bounds is not None:
            packet_span = max(joint_bounds[1], pose_bounds[1]) - min(joint_bounds[0], pose_bounds[0])
            stamp_delta = joint_bounds[0] - pose_bounds[0]
        feedback_diagnostics = {'flange_fk_position_error_m': position_error,
                                'flange_fk_orientation_error_deg': orientation_error,
                                'joint_pose_packet_span_s': packet_span,
                                'joint_pose_stamp_delta_s': stamp_delta,
                                'flange_fk_consistency_is_diagnostic': True}
        self.data.qvel[self.robot_controller.dof_indices] = self._joint_velocity
        self.data.time = float(timestamp)
        self.gripper_controller.measured_open_fraction = width / .1
        grip_stamp = mapped_stamp(state, 'gripper', legacy=timestamp)
        if grip_stamp is None:
            raise PickRejected('完整策略收到无效夹爪反馈时间映射')
        grip_stamp = float(grip_stamp)
        if self._last_gripper_stamp is None or grip_stamp != self._last_gripper_stamp:
            self._gripper_observed_at = float(timestamp)
        fresh = self._gripper_observed_at is not None and 0 <= timestamp-self._gripper_observed_at <= .2
        loaded = bool(fresh and .003 < width < .096 and force >= self.settings.min_force_n)
        if not loaded:
            self._proxy_since = None
            self._proxy_confirmed = False
        elif self._last_gripper_stamp is None or grip_stamp < self._last_gripper_stamp:
            self._proxy_since = grip_stamp
            self._proxy_confirmed = False
        elif grip_stamp - self._last_gripper_stamp > .2 and not self._proxy_confirmed:
            # A processing gap cannot establish the initial continuous dwell.
            # Once confirmed, however, a NEW loaded packet is evidence that
            # the hold persists; skipped processing is not an unloaded sample.
            self._proxy_since = grip_stamp
        elif grip_stamp > self._last_gripper_stamp and self._proxy_since is None:
            self._proxy_since = grip_stamp
        # A repeated timestamp cannot extend a continuous force dwell.
        self._last_gripper_stamp = grip_stamp
        holding = bool(loaded and self._proxy_since is not None and grip_stamp - self._proxy_since >= .4)
        self._proxy_confirmed = holding
        tcp = sdk_flange @ self.geometry.flange_to_tcp
        observation = {'joint_pos': q.copy(), 'joint_vel': self._joint_velocity.copy(),
                       'ee_pose': _pose(tcp), 'timestamp': float(timestamp),
                       'feedback_diagnostics': feedback_diagnostics,
                       'scene_epoch': self.scene_epoch, 'belt_speed': 0.,
                       'belt_velocity': np.zeros(3), 'severe_collision': False,
                       'joint_target': self.robot_controller.commanded_joint_positions,
                       'joint_target_velocity': self.robot_controller.commanded_joint_velocities,
                       'objects': [] if self._released else [self._target.copy()],
                       'gripper': {'opening_width': width, 'open_fraction': width / .1,
                                   'target_open_fraction': self.gripper_controller.target_open_fraction,
                                   'confirmation_source': 'width_and_force',
                                   'width_force_holding': holding,
                                   'motor_force_magnitude_n': force,
                                   'motor_force_signed_n': state.get('force_signed_n'),
                                   'feedback_stamp': grip_stamp}}
        self._last_observation = observation
        return observation

    def select_target(self, observation):
        if self._reserved or self._released:
            return None
        self._reserved = True; self._target['status'] = 'PLANNING'
        return self._target.copy()

    def get_object_state(self, object_id):
        return self._target.copy() if object_id == self._target['object_id'] else None

    def release_selected_target(self, object_id):
        # Single-attempt entry: release never silently retries a rejected branch.
        self._target['status'] = 'REJECTED'

    def get_wait_pose(self, observation=None):
        return self.wait_pose.copy()

    def get_lift_pose(self, observation=None):
        return self.lift_pose.copy()

    def get_bin_drop_pose(self, bin_index):
        return self.grasp_pose.copy()

    def on_policy_event(self, event):
        self.events.append(event)
        if event.name == 'release_complete':
            self._released = True; self._target['status'] = 'RELEASED'
        elif event.name == 'attempt_complete':
            self.complete = True
        elif event.name in {'attempt_failed', 'return_timeout', 'recovery_timeout'}:
            self.failed_reason = event.reason or event.name

    def set_motion_profile(self, profile):
        if profile not in {'nominal', 'carrying'}:
            raise ValueError('unknown motion profile')

    def get_joint_positions(self):
        return self.data.qpos[self.geometry.indices].copy()

    def get_global_planning_snapshot(self):
        full = self.data.qpos.copy()
        width = self._arm_validation_width()
        if self.policy.stage.value in {'waiting', 'pregrasp', 'returning'}:
            # Plan acquisition/return for the future fully open jaws even if an
            # opening command is still settling; do not mutate measured state.
            width = max(width, .1)
        full[self.model.joint('gripper_joint1').qposadr] = width / 2
        full[self.model.joint('gripper_joint2').qposadr] = -width / 2
        return {'full_qpos': full, 'arm_qpos': self.get_joint_positions(),
                'scene_epoch': self.scene_epoch}

    def get_future_open_planning_snapshot(self, arm_joint_positions):
        """Explicit future geometry for native post-release/look-ahead jobs."""
        arm = np.asarray(arm_joint_positions, dtype=float)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError('future arm state must contain seven finite joints')
        full = self.data.qpos.copy()
        full[self.geometry.indices] = arm
        full[self.model.joint('gripper_joint1').qposadr] = .05
        full[self.model.joint('gripper_joint2').qposadr] = -.05
        return {'full_qpos': full, 'arm_qpos': arm.copy(), 'scene_epoch': self.scene_epoch}

    def solve_ik_result(self, position, quaternion, qpos_seed, *, allow_multiseed=False):
        if allow_multiseed:
            raise ValueError('multi-seed search belongs to shared background worker')
        return self.ik.solve(position, quaternion, qpos_seed=qpos_seed)

    def _allow_target_finger_contact(self, geometry, q):
        # The selected workpiece is a grasp/contact object. Its uncertain
        # visible surface cannot forbid the finger contact needed to pick it.
        # RobotModel still checks these fingers against the separate scene and
        # checks every non-finger robot geometry against this target cloud.
        return True

    def _arm_validation_width(self, measured_width=None):
        """Use one jaw geometry for local planning and actual arm dispatch.

        An opening command can precede corresponding width feedback. Local IK
        and edge checks must already include that requested opening, just as
        the final dispatch check does. This affects only collision geometry;
        the measured qpos, width/force feedback and grasp proxy stay unchanged.
        """
        if measured_width is None:
            measured_width = float(
                self.data.qpos[self.model.joint('gripper_joint1').qposadr][0]) * 2
        return max(float(measured_width), .1 * self.gripper_controller.target_open_fraction)

    def _geometry_valid(self, geometry, q, width, *, allow_start_region=None,
                        rejection_details=None, scene_prevalidated=False):
        if rejection_details is not None:
            rejection_details.clear()
        if allow_start_region is None:
            allow_start_region = self.policy.stage.value in {'waiting', 'pregrasp'}
        if self.joint_limit_report['expanded'] and not allow_start_region:
            # The bounded extra interval is only for acquiring a normal IK
            # branch from measured start. All later stage paths stay normal.
            if np.any(np.asarray(q) < geometry.nominal_command_lower-1e-10) or np.any(
                    np.asarray(q) > geometry.nominal_command_upper+1e-10):
                if rejection_details is not None:
                    rejection_details.update(reason='nominal_joint_range',
                        joints_rad=np.asarray(q).tolist(),
                        lower_limits_rad=geometry.nominal_command_lower.tolist(),
                        upper_limits_rad=geometry.nominal_command_upper.tolist())
                return False
        try:
            # check_state sets q/width and calls mj_forward itself. Target
            # finger permission is semantic and does not inspect stale FK.
            options = {'allow_finger_contact': self._allow_target_finger_contact(geometry, q)}
            if scene_prevalidated:
                options['check_scene'] = False
            geometry.check_state(q, width, **options)
        except PickRejected as error:
            if rejection_details is not None:
                rejection_details.update(reason='scene_geometry', message=str(error),
                                         details=error.details)
            return False
        return True

    @staticmethod
    def _batch_scene_validation(geometry, states, check, rejected):
        """Native/self/target checks per pose, one static-scene batch query."""
        positions, rotations = [], []
        geoms = tuple(geometry.meshes)
        for q in states:
            if not check(q):
                return False
            positions.append(geometry.data.geom_xpos[list(geoms)].copy())
            rotations.append(geometry.data.geom_xmat[list(geoms)].reshape(-1, 3, 3).copy())
        margins = np.array([geometry.scene_collision_margin(g) for g in geoms])
        bounds = np.array([geometry.mesh_bounds[g, float(m)] for g, m in zip(geoms, margins)])
        found = geometry.scene_collision_index().query_batch(
            np.array(positions), np.array(rotations),
            np.array([geometry.collision_radii[g]+1e-9 for g in geoms]), margins, bounds)
        for state_index, collisions in enumerate(found):
            for geom in geoms:
                indices = collisions.get(geom)
                if indices is None or not len(indices):
                    continue
                name = geometry.model.geom(geom).name
                rejected.update(reason='scene_geometry',
                    message=f'路径碰到可见障碍点云：{name}', details={
                        'kind': 'scene_collision', 'geometry': name,
                        'point_count': len(indices),
                        'points_base_m': geometry.scene[indices[:20]].tolist(),
                        'joints_rad': np.asarray(states[state_index]).tolist(),
                        'margin_m': float(geometry.scene_collision_margin(geom)),
                        'validation': 'batched_observed_points_exact_confirmation'})
                return False
        return True

    def _make_candidate_path_backend(self):
        prepared = getattr(self.geometry, '_prepared_candidate_backend', None)
        if prepared is not None:
            if prepared not in self._candidate_backends:
                self._candidate_backends.append(prepared)
                try:
                    prepared.update_scene(self.geometry.scene)
                except Exception as error:
                    self._candidate_backend_fallback = str(error)
                    return None
            return prepared
        mode = getattr(self.settings, 'planning_backend', 'auto')
        if mode == 'auto':
            # Saved-scene full-route measurements favour CPU on this host.
            # cuRobo stays explicitly selectable; GPU use alone is not proof
            # of a shorter capture-to-motion delay.
            mode = os.environ.get('NERO_PLANNING_BACKEND', 'cpu')
            if mode == 'auto':
                mode = 'cpu'
        if mode == 'cpu':
            return None
        try:
            from ..controllers.curobo_planner import CuRoboCandidatePathBackend
            backend = CuRoboCandidatePathBackend(self.model,
                joint_names=self.geometry.names, ee_site_name='ee_site',
                position_tolerance=self.ik.config.position_tolerance,
                orientation_tolerance=self.ik.config.orientation_tolerance,
                scene_points=self.geometry.scene,
                collision_geom_ids=tuple(self.geometry.meshes))
            self._candidate_backends.append(backend)
            return backend
        except Exception as error:
            self._candidate_backend_fallback = str(error)
            return None

    def planning_acceleration_report(self):
        roadmap = self._make_warm_roadmap_backend()
        return {'requested_planning_backend': getattr(self.settings, 'planning_backend', 'auto'),
                'candidate_backends': [dict(backend.diagnostics) for backend in self._candidate_backends],
                'fallback_reason': getattr(self, '_candidate_backend_fallback', None),
                'scene_queries': [dict(index.diagnostics) for index in self._scene_indexes],
                'shared_state_evaluation': getattr(self.settings, 'shared_state_evaluation', True),
                'minimal_state_evaluation': getattr(self.settings, 'minimal_state_evaluation', True),
                'adaptive_segment_validation': getattr(self.settings, 'adaptive_segment_validation', True),
                'simd_broadphase_enabled': getattr(self.settings, 'simd_broadphase_enabled', True),
                'observed_convex_batch_preparation': getattr(
                    self.geometry, '_observed_convex_batch_preparation', None),
                'warm_roadmap': (dict(roadmap.diagnostics)
                    if roadmap is not None else
                    getattr(self.geometry, '_warm_roadmap_preparation', {
                        'enabled': getattr(self.settings, 'warm_roadmap_enabled', True),
                        'ready': False, 'reason': 'not_preheated_use_original_planner'}))}

    def _make_warm_roadmap_backend(self):
        if not getattr(self.settings, 'warm_roadmap_enabled', True):
            return None
        prepared = getattr(self.geometry, '_prepared_warm_roadmap', None)
        return prepared if prepared is not None and prepared.ready else None

    def _state_valid(self, geometry, q, *, approach_axis_local=(0., 0., 1.),
                     minimum_downward_alignment=0., minimum_tcp_height=-np.inf,
                     minimum_joint_margin=0., maximum_jacobian_condition=None):
        self.geometry_validation_stats['state_checks'] += 1
        self.motion_validation_rejection = {}
        rejected = self.motion_validation_rejection
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            rejected.update(reason='invalid_joint_input')
            return False
        bounds = self.geometry.limits
        margins = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
        if np.min(margins) < minimum_joint_margin - 1e-9:
            rejected.update(reason='joint_margin', joint_index=int(np.argmin(margins)),
                            actual_margin_rad=float(np.min(margins)),
                            required_margin_rad=float(minimum_joint_margin))
            return False
        width = getattr(self, '_command_validation_width', None)
        if width is None:
            width = self._arm_validation_width()
        if not self._geometry_valid(geometry, q, width, rejection_details=rejected):
            return False
        tcp = geometry.tcp()
        if tcp[2, 3] < minimum_tcp_height - 1e-9:
            rejected.update(reason='tcp_height', actual_height_m=float(tcp[2, 3]),
                            minimum_height_m=float(minimum_tcp_height))
            return False
        axis = np.asarray(approach_axis_local, dtype=float)
        alignment = -float((tcp[:3, :3] @ (axis / np.linalg.norm(axis)))[2])
        if alignment < minimum_downward_alignment - 1e-9:
            rejected.update(reason='downward_alignment', actual_alignment=alignment,
                            minimum_alignment=float(minimum_downward_alignment))
            return False
        if maximum_jacobian_condition is not None and np.isfinite(maximum_jacobian_condition):
            jac = geometry.ik.compute_jacobian(q); jac[3:] *= .30
            singular = np.linalg.svd(jac, compute_uv=False)
            if singular[-1] <= 1e-12 or singular[0] / singular[-1] > maximum_jacobian_condition:
                rejected.update(reason='jacobian_condition', singular_values=singular.tolist(),
                                maximum_condition=float(maximum_jacobian_condition))
                return False
        return True

    def is_arm_configuration_safe(self, q, **kwargs):
        return self._state_valid(self._validation_geometry, q, **kwargs)

    def is_arm_edge_safe(self, start, end, *, normalized_resolution,
                         maximum_joint_step=None, **kwargs):
        self.motion_validation_rejection = {}
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (7,) or end.shape != (7,) or normalized_resolution <= 0:
            self.motion_validation_rejection = {'reason': 'invalid_edge_input'}
            return False
        self.geometry_validation_stats['edge_requests'] += 1
        width = getattr(self, '_command_validation_width', None)
        if width is None:
            width = self._arm_validation_width()
        geometry = self._validation_geometry
        # This is an exact-input memo, not a reuse tolerance. Any new scene,
        # target sweep, stage, width, limits, sampling or constraint differs.
        key = (self.scene_epoch, self.policy.stage.value, id(geometry.scene),
               id(geometry.target), id(getattr(self, '_live_collision', None)),
               geometry.gripper_scene_margin_m,
               self.geometry.limits.tobytes(), geometry.command_lower.tobytes(),
               geometry.command_upper.tobytes(), float(width), start.tobytes(),
               end.tobytes(), float(normalized_resolution), maximum_joint_step,
               tuple((k, np.asarray(v).tobytes()) for k, v in sorted(kwargs.items())))
        cacheable = not geometry.scene.flags.writeable and not geometry.target.flags.writeable
        if cacheable and key in self._edge_validation_cache:
            self.geometry_validation_stats['edge_cache_hits'] += 1
            return True
        if np.array_equal(start, end):
            valid = self.is_arm_configuration_safe(start, **kwargs)
        else:
            span = self.geometry.limits[:, 1] - self.geometry.limits[:, 0]
            count = max(1, int(np.ceil(np.linalg.norm((end-start)/span) / normalized_resolution)),
                        int(np.ceil(np.max(np.abs(end-start)) / min(.008, maximum_joint_step or .008))))
            valid = True
            for u in np.linspace(0., 1., count+1):
                if not self.is_arm_configuration_safe(start + (end-start)*u, **kwargs):
                    self.motion_validation_rejection['edge_fraction'] = float(u)
                    valid = False
                    break
        if valid and cacheable:
            # Keep references alive: a later snapshot cannot recycle an id
            # while an earlier proof still exists in this observation.
            self._edge_validation_cache[key] = (
                geometry.scene, geometry.target, getattr(self, '_live_collision', None))
        return valid

    @property
    def workers(self):
        return tuple(self._workers)

    def validate_gripper_command(self, state, width):
        self.motion_validation_rejection = {}
        if not np.isfinite(width) or not 0. <= width <= .1:
            self.motion_validation_rejection = {'reason': 'invalid_gripper_width'}
            return False
        q = np.asarray(state['joints'], dtype=float)
        start_width = float(state['width_m'])
        samples = int(np.ceil(abs(width-start_width)/.01))
        for value in np.linspace(start_width, width, samples+1):
            if not self._geometry_valid(self._validation_geometry, q, value,
                                        rejection_details=self.motion_validation_rejection):
                self.motion_validation_rejection['sampled_width_m'] = float(value)
                return False
        return True

    def validate_command(self, state, command):
        """Validate arm motion at the wider measured/requested jaw opening."""
        self.motion_validation_rejection = {}
        self._command_validation_width = self._arm_validation_width(state['width_m'])
        try:
            return self._validate_arm_command(state, command)
        finally:
            self._command_validation_width = None

    def _validate_arm_command(self, state, command):
        """Validate the actual shared-controller output, including limiter bends."""
        start = np.asarray(state['joints'], dtype=float)
        end = np.asarray(command.joint_target, dtype=float)
        if self.policy._safe_joint_path:
            return self.policy._safe_live_command_edge_is_safe(start, end)
        target_pose = self.policy._cartesian_command_pose
        if target_pose is None:
            target_pose = self.policy._grasp_pose
        if target_pose is None:
            target_pose = self.wait_pose
        current_height = float(self.ik.forward_pose(start)[2])
        target_height = float(target_pose[2])
        return self.is_arm_edge_safe(
            start, end,
            normalized_resolution=self.policy.config.planner_edge_resolution,
            maximum_joint_step=.008,
            approach_axis_local=self.policy.config.approach_axis_local,
            minimum_downward_alignment=self.policy._local_downward_alignment_limit(),
            minimum_tcp_height=min(current_height, target_height)-.015,
            minimum_joint_margin=self.policy.config.minimum_joint_limit_margin,
            maximum_jacobian_condition=self.policy.config.singularity_hard_stop_condition,
        )

    def can_reuse_planning_certificate(self, request, result, observation):
        """The worker and live validator own the same frozen observed scene.

        Arm feedback may drift and is connected separately by the policy. Any
        changed jaw/non-arm geometry, version or limit setting invalidates
        this reuse; absence of this hook keeps other environments unchanged.
        """
        if request.scene_epoch != self.scene_epoch or result.scene_epoch != self.scene_epoch:
            return False
        if request.full_qpos.shape != self.data.qpos.shape:
            return False
        non_arm = np.ones(self.model.nq, dtype=bool)
        non_arm[self.geometry.indices] = False
        # Compare the geometry actually used by BOTH collision validators.
        # Feedback qpos remains measured; an opening request can be ahead of it.
        collision_qpos = self.data.qpos.copy()
        width = self._arm_validation_width()
        collision_qpos[self.model.joint('gripper_joint1').qposadr] = width / 2
        collision_qpos[self.model.joint('gripper_joint2').qposadr] = -width / 2
        if not np.array_equal(request.full_qpos[non_arm], collision_qpos[non_arm]):
            return False
        lower = self.robot_controller.joint_lower_limits
        upper = self.robot_controller.joint_upper_limits
        velocity, acceleration = self.policy._trajectory_limits()
        return bool(
            np.array_equal(request.lower_limits, lower)
            and np.array_equal(request.upper_limits, upper)
            and np.array_equal(request.trajectory_velocity_limits, velocity)
            and np.array_equal(request.trajectory_acceleration_limits, acceleration)
            and request.planner_config.normalized_edge_resolution
                <= .008 / float(np.max(upper-lower)) + 1e-15
        )

    def create_global_planning_worker(self):
        geometry = _copy_geometry(self.geometry)
        self._scene_indexes.append(geometry.scene_collision_index())
        self._scene_indexes[-1].prepare()
        def validator(request, q):
            width = float(request.full_qpos[geometry.model.joint('gripper_joint1').qposadr][0]) * 2
            validator.last_rejection = {}
            return self._geometry_valid(geometry, q, width,
                                        allow_start_region=request.stage == 'pregrasp',
                                        rejection_details=validator.last_rejection)
        def batch(request, states):
            validator.last_rejection = {}
            width = float(request.full_qpos[geometry.model.joint('gripper_joint1').qposadr][0])*2
            target_prepare = getattr(geometry, 'prepare_target_broadphase', None)
            if callable(target_prepare):
                target_prepare(states, width, geometry.target, .004)
            return self._batch_scene_validation(geometry, states,
                lambda q: self._geometry_valid(geometry, q, width,
                    allow_start_region=request.stage == 'pregrasp',
                    scene_prevalidated=True, rejection_details=validator.last_rejection),
                validator.last_rejection)
        validator.batch = batch
        def bind(evaluator):
            geometry.model, geometry.data = evaluator.model, evaluator.data
            geometry._state_evaluator = evaluator
        validator.bind_state_evaluator = bind
        validator.geometry = geometry
        validator.segment_validation_semantics = 'nero_observed_convex_v1'
        bodies = tuple({int(self.model.geom_bodyid[g]) for g in self.geometry.meshes})
        engine = MuJoCoGlobalPlanningEngine(self.model, joint_names=self.geometry.names,
                    ee_site_name='ee_site', ik_config=self.ik.config,
                    robot_collision_body_ids=bodies, state_validator=validator,
                    candidate_path_backend=self._make_candidate_path_backend(),
                    warm_roadmap_backend=self._make_warm_roadmap_backend(),
                    shared_state_evaluation=getattr(self.settings, 'shared_state_evaluation', True),
                    minimal_state_evaluation=getattr(self.settings, 'minimal_state_evaluation', True),
                    adaptive_segment_validation=getattr(self.settings, 'adaptive_segment_validation', True))
        if self.joint_limit_report['expanded']:
            # RRT may certify the small outside interval at the initial start;
            # every newly solved endpoint and continuation stays normal. The
            # worker intersects these normal bounds with each request's
            # accepted joint window, then restores them after every solve.
            engine.ik.lower_limits = geometry.nominal_command_lower.copy()
            engine.ik.upper_limits = geometry.nominal_command_upper.copy()
            engine.ik.center_positions = .5*(engine.ik.lower_limits+engine.ik.upper_limits)
        worker = AsyncPlanningWorker(engine)
        self._workers.append(worker)
        return worker

    @property
    def task_state(self):
        state = task_lifecycle(self.policy.stage, complete=self.complete, failed=bool(self.failed_reason))
        if state == 'DETECTED':
            snapshot = getattr(self,'_motion_snapshot',None)
            visible = getattr(snapshot, 'visible', None)
            if visible is not None:
                return 'TRACKING' if visible else 'LOST'
        if state == 'PLANNING' and getattr(self.policy, '_full_pick_bundle', None) is not None:
            return ('EXECUTING' if 'trajectory_execution_start' in
                getattr(self.policy,'_full_pick_phase_events',()) else 'SCHEDULED')
        return state

    def close(self):
        if not getattr(self, '_task_terminal_recorded', False):
            state = self.task_state
            terminal = getattr(self,'_task_outcome',None) or (
                state if state in {'COMPLETED','FAILED'} else 'CANCELLED')
            self.task_terminal = {'name':'task_finished','state':terminal,
                'target_identity':identity_dict(self.target_identity),
                'finished_monotonic':time.monotonic(), 'resource_release':'context_close',
                'reason':self.failed_reason}
            self._task_terminal_recorded = True
        monitor = getattr(self, 'full_plan_scene_monitor', None)
        if monitor is not None:
            monitor.close()
        try:
            self.policy.close()
        finally:
            for backend in self._candidate_backends:
                backend.close()
            for index in self._scene_indexes:
                index.close()


def build_policy_context(geometry, detection, initial_state, settings):
    return NeroPolicyContext(geometry, detection, initial_state, settings)

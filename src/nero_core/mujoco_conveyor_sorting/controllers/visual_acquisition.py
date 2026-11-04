"""October 9 visual acquisition, shared by hardware and network simulation.

Only the final acquisition reference and command origin differ from the
baseline. Global planning, IK and edge admission, joint-rate limits, capture
feedback, lift, release and return all remain in BaselineSortingPolicy.
Positions, plane normals and velocities use the robot base frame in metres.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ._math import belt_plane_motion_from_xy, compose_world_delta, quat_distance, quat_error_vector
from .baseline_sorting_policy import (
    BaselinePolicyConfig, BaselineSortingPolicy, PolicyStage,
)
from .servo_response import VisualFollowingLeadAdapter


_ACQUISITION_STAGES = {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}


class VisualAcquisitionMixin:
    """Use published visual positions and bounded accepted-command travel.

    ``_cartesian_command_pose`` is the previous Cartesian request accepted by
    the shared IK/edge checks. It is not encoder feedback or the FK of the
    controller's rate-limited joint target. This is the October 9 integrator's
    origin; measured TCP feedback separately bounds its accumulated lead.
    """

    def _acquisition_plane_normal(self):
        explicit = getattr(self, '_belt_plane_normal', None)
        if explicit is not None:
            return explicit.copy()
        geometry = getattr(self.env, 'geometry', None)
        plane = getattr(geometry, 'scene_region_plane_to_base', None)
        if plane is not None:
            plane = np.asarray(plane, dtype=float)
            if plane.shape == (4, 4) and np.isfinite(plane).all():
                normal = plane[:3, 2]
                norm = float(np.linalg.norm(normal))
                if norm > 1e-8:
                    return normal / norm
        # Original independent simulation uses the level world XY plane.
        # The hardware context supplies its measured support-plane normal.
        return np.array([0., 0., 1.])

    def _acquisition_tangent(self, value):
        value = np.asarray(value, dtype=float)
        normal = self._acquisition_plane_normal()
        return value - normal * float(value @ normal)

    def _retimed_grasp_position(self, previous, body_position):
        # The motion-derived belt direction can have Vz=0 (Nero7). Use the
        # measured support plane for height when retiming a reserved grasp.
        normal = self._acquisition_plane_normal()
        if abs(float(normal[2])) <= 1e-12:
            return super()._retimed_grasp_position(previous, body_position)
        previous = np.asarray(previous, dtype=float)
        delta = np.r_[np.asarray(body_position)[:2]-previous[:2], 0.]
        return previous + belt_plane_motion_from_xy(delta, normal)

    def _current_visual_acquisition(self, target, observation):
        """Read publication-time position without adding camera age twice."""
        if (not self._visual_acquisition_tracking
                or self._stage not in _ACQUISITION_STAGES
                or not isinstance(target, Mapping)):
            return None
        try:
            position = np.asarray(target.get('position'), dtype=float)
            velocity = np.asarray(target.get('estimated_velocity', target.get('linear_velocity')), dtype=float)
            publication = float(target.get('timestamp'))
            now = self._timestamp_without_advancing(observation)
        except (TypeError, ValueError):
            return None
        if (position.shape != (3,) or velocity.shape != (3,)
                or not np.isfinite(np.r_[position, velocity, publication, now]).all()
                or abs(now-publication) > 2.*self.config.control_timestep):
            return None
        return position.copy(), velocity.copy()

    def _refresh_pending_motion_target(self, target_pose, observation, intent):
        """Rebase an unsolved acquisition intent on the newest publication.

        The task decision and following-lead filter ran once in act(). Neither
        accepted Cartesian travel nor IK/controller history has advanced yet.
        Keep the once-updated lead horizon, then recompute its displacement
        using the latest visual velocity and position. The lead filter itself
        is not integrated again while a prepared intent is refreshed.
        The original IK, Cartesian profile and measured-feedback bounds run
        once afterwards in finalize_pending_motion().
        """
        target_pose = super()._refresh_pending_motion_target(
            target_pose, observation, intent)
        if target_pose is None or intent.get('stage') not in _ACQUISITION_STAGES:
            return target_pose
        previous = intent.get('reference_target')
        target = self._find_target(observation, intent.get('target_id'))
        visual = self._current_visual_acquisition(target, observation)
        if not isinstance(previous, Mapping) or visual is None:
            return target_pose
        try:
            old_position = np.asarray(previous.get('position'), dtype=float)
        except (TypeError, ValueError):
            return target_pose
        if old_position.shape != (3,) or not np.isfinite(old_position).all():
            return target_pose
        refreshed = np.asarray(target_pose, dtype=float).copy()
        horizon = (self.config.closing_tracking_lead_time
                   if intent.get('stage') == PolicyStage.CLOSING
                   else self.config.tracking_velocity_feedforward_time)
        prepared = getattr(self, '_prepared_visual_tracking_reference', None)
        if (isinstance(prepared, Mapping)
                and prepared.get('stage') == intent.get('stage')
                and prepared.get('publication_s') == previous.get('timestamp')
                and np.allclose(prepared['pose'], target_pose, atol=1e-12, rtol=0.)):
            horizon = prepared['horizon_s']
        elif self.config.dynamic_intercept_tracking and horizon > 0.:
            dynamic = getattr(self.env, 'scene_config', {}).get('workspace', {}).get('dynamic_intercept', {})
            horizon = float(dynamic.get('servo_lookahead_s', horizon))
        intent.setdefault('visual_horizon_s', horizon)
        dispatch_delay = 0.
        dispatch_clock = observation.get('dispatch_clock')
        if isinstance(dispatch_clock, Mapping):
            scheduled = float(dispatch_clock['scheduled_dispatch_policy_time_s'])
            dispatch_delay = max(0., scheduled-self._timestamp_without_advancing(observation))
        # Preserve the height above the measured belt and selected quaternion.
        # Predict from this publication to the scheduled send, then add the
        # original servo lead. Keep publication/exposure times untouched.
        # Closure rechecks use a new observation without dispatch_clock, so
        # they do not add this preparation wait again to the contact target.
        refreshed = self._visual_tracking_pose(
            refreshed, observation, target, visual,
            float(intent['visual_horizon_s'])+dispatch_delay)
        displacement = self._acquisition_tangent(visual[0]-old_position)
        capture_pose = intent.get('capture_pose')
        if capture_pose is not None:
            intent.setdefault('reference_capture_pose', np.asarray(capture_pose, dtype=float).copy())
            capture_pose = np.asarray(intent['reference_capture_pose'], dtype=float).copy()
            capture_pose[:3] += displacement
            intent['capture_pose'] = self._guard_final_tracking_xy(
                capture_pose, target, observation)
        reference = intent.get('tracking_reference_before_update')
        if reference is not None:
            intent.setdefault('reference_tracking_pose', np.asarray(reference, dtype=float).copy())
            reference = np.asarray(intent['reference_tracking_pose'], dtype=float).copy()
            reference[:3] += displacement
            intent['tracking_reference_before_update'] = reference
        intent['refreshed_target'] = target
        return refreshed

    def _grasp_target_pose(self, target, observation, *, remember_orientation=True):
        pose = super()._grasp_target_pose(
            target, observation, remember_orientation=remember_orientation)
        visual = self._current_visual_acquisition(target, observation)
        if pose is not None and visual is not None:
            pose = pose.copy()
            # Keep the selected tool height and quaternion; replace only the
            # coordinates along the measured belt plane with current vision.
            pose[:3] += self._acquisition_tangent(visual[0]-pose[:3])
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
            dynamic = getattr(self.env, 'scene_config', {}).get('workspace', {}).get('dynamic_intercept', {})
            horizon = float(dynamic.get('servo_lookahead_s', horizon))
        velocity = self._acquisition_tangent(visual[1])
        if self._visual_servo_bias_compensation and (lead_time is None or lead_time > 0.):
            current = self._ee_pose(observation)
            normal = self._acquisition_plane_normal()
            horizon, self._following_lead_diagnostics = self._following_lead.update(
                timestamp=self._timestamp_without_advancing(observation),
                contact_position=visual[0], feedback_position=current[:3], track_velocity=velocity,
                acquisition_key=(self._observed_scene_epoch, self.target_id), fallback_horizon_s=horizon,
                near_capture_height=abs(float((current[:3]-target_pose[:3]) @ normal))
                                    <= self.config.closure_vertical_tolerance)
        command = self._visual_tracking_pose(target_pose, observation, target, visual, horizon)
        if lead_time is None or lead_time > 0.:
            self._prepared_visual_tracking_reference = {
                'pose': command.copy(), 'horizon_s': float(horizon),
                'stage': self._stage, 'publication_s': target.get('timestamp')}
        return command

    def _visual_tracking_pose(self, target_pose, observation, target, visual, horizon):
        """Pure position/velocity projection with an already selected horizon."""
        velocity = self._acquisition_tangent(visual[1])
        command = np.asarray(target_pose, dtype=float).copy()
        offset = velocity * horizon
        command[:3] += self._acquisition_tangent(visual[0]-command[:3]) + offset
        if self.config.dynamic_intercept_tracking:
            bounds = self._belt_bounds(
                'tracking_s_range', 'tracking_x_range', 's_range', 'x_range', default=(.02, .18))
            coordinate = self._belt_coordinate(command[:3])
            command[:3] += (float(np.clip(coordinate, *bounds))-coordinate) * self._belt_frame()[1]
        # The historical keyword remains compatible; this implementation
        # accepts a full base XYZ offset rather than assuming the world XY.
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
        now = self._timestamp_without_advancing(observation)
        # Signed time includes the period before the future intercept. The
        # existing 35 mm coast allowance follows the belt instead of stopping
        # at that future position. Finger brushes do not freeze acquisition.
        reference = planned[:3] + self._belt_velocity(observation) * (now-planned_time)
        if reference_offset_xy is not None:
            offset = np.asarray(reference_offset_xy, dtype=float)
            reference += np.r_[offset, 0.] if offset.shape == (2,) else offset
        residual = self._acquisition_tangent(guarded[:3]-reference)
        distance = float(np.linalg.norm(residual))
        limit = self.config.coasted_tracking_xy_limit
        if distance > limit and distance > 1e-12:
            guarded[:3] -= residual * (1.-limit/distance)
        return guarded

    def _cartesian_servo_reference(self, observation, *, dynamic_acquisition):
        if (self._stage in _ACQUISITION_STAGES
                and self._acquisition_command_integrator
                and self._cartesian_command_pose is not None):
            return self._cartesian_command_pose.copy()
        return super()._cartesian_servo_reference(
            observation, dynamic_acquisition=dynamic_acquisition)

    def _cartesian_servo_pose(self, target_pose, observation, *, scale=1.):
        if self._stage not in _ACQUISITION_STAGES:
            return super()._cartesian_servo_pose(target_pose, observation, scale=scale)
        current = self._ee_pose(observation)
        target = np.asarray(target_pose, dtype=float)
        previous = self._cartesian_command_pose
        allocation = {'enabled': bool(getattr(self, '_acquisition_tangent_priority', False)),
                      'applied': False}
        if self._acquisition_command_integrator:
            # Reuse the shared speed profile and its numerical retry scaling.
            # The hook above changes only the origin to the accepted request.
            reference = self._cartesian_servo_reference(
                observation, dynamic_acquisition=self.config.dynamic_intercept_tracking)
            requested_translation = target[:3]-reference[:3]
            target = super()._cartesian_servo_pose(target, observation, scale=scale)
            if (allocation['enabled']
                    and (self.config.tcp_speed_limits_enabled
                         or self.config.dynamic_intercept_tracking or self._clearing_payload())):
                # A large descent error must not scale away the belt-following
                # increment. Keep the shared total speed/backtracking budget,
                # spend it on the measured belt plane first, and descend with
                # the orthogonal remainder. IK, joint limits and feedback lead
                # admission still run on the resulting request as before.
                normal = self._acquisition_plane_normal()
                normal_distance = float(requested_translation @ normal)
                tangent = requested_translation-normal*normal_distance
                tangent_distance = float(np.linalg.norm(tangent))
                budget = self._local_translation_step_limit(observation) * float(scale)
                tangent_step = tangent.copy()
                if tangent_distance > budget and tangent_distance > 1e-12:
                    tangent_step *= budget/tangent_distance
                normal_budget = float(np.sqrt(max(0., budget*budget-float(tangent_step @ tangent_step))))
                normal_step = float(np.clip(normal_distance, -normal_budget, normal_budget))
                isotropic = target[:3]-reference[:3]
                allocated = tangent_step+normal*normal_step
                target = target.copy()
                target[:3] = reference[:3]+allocated
                allocation.update(
                    applied=bool(np.linalg.norm(allocated-isotropic) > 1e-12),
                    translation_budget_m=budget,
                    requested_tangent_m=tangent.copy(), requested_normal_m=normal_distance,
                    isotropic_tangent_step_m=isotropic-normal*float(isotropic @ normal),
                    isotropic_normal_step_m=float(isotropic @ normal),
                    allocated_tangent_step_m=tangent_step.copy(), allocated_normal_step_m=normal_step)
        translation = target[:3]-current[:3]
        translation_limit = self._acquisition_translation_step_m
        rotation_limit = self.config.max_rotation_step
        if (not self._acquisition_command_integrator
                and (self.config.tcp_speed_limits_enabled or self.config.dynamic_intercept_tracking)):
            translation_limit = min(translation_limit, self._local_translation_step_limit(observation))
            rotation_limit = min(rotation_limit,
                                 self.config.cartesian_angular_speed*self._execution_dt(observation))
        translation_limit *= float(scale)
        rotation_limit *= float(scale)
        distance = float(np.linalg.norm(translation))
        if distance > translation_limit:
            translation = translation * (translation_limit/distance)
        rotation = quat_error_vector(target[3:], current[3:])
        angle = float(np.linalg.norm(rotation))
        if angle > rotation_limit:
            rotation = rotation * (rotation_limit/angle)
        bounded = np.r_[current[:3]+translation, compose_world_delta(current[3:], rotation)]
        self._visual_acquisition_diagnostics = {
            'command_reference': ('previous_ik_accepted_cartesian_request'
                                  if self._acquisition_command_integrator and previous is not None
                                  else 'measured_tcp_feedback'),
            'measured_feedback_pose': current.copy(),
            'previous_cartesian_request_pose': None if previous is None else previous.copy(),
            'proposed_pose': target.copy(), 'bounded_pose': bounded.copy(),
            'measured_lead_m': float(np.linalg.norm(translation)),
            'measured_lead_limit_m': translation_limit,
            'requested_pose_is_actual_feedback': False,
            'tangent_priority_allocation': allocation,
        }
        return bounded

    def _closure_pose_reached(self, target_pose, observation, target=None):
        if not self._visual_acquisition_tracking or self._stage not in _ACQUISITION_STAGES:
            return super()._closure_pose_reached(target_pose, observation, target)
        current = self._ee_pose(observation)
        error = np.asarray(target_pose[:3])-current[:3]
        target = target if isinstance(target, Mapping) else {}
        values = [target.get(name, getattr(self.config, name)) for name in
                  ('closure_horizontal_tolerance', 'closure_vertical_tolerance',
                   'closure_orientation_tolerance')]
        try:
            horizontal, vertical, orientation = (max(0., float(value)) for value in values)
        except (TypeError, ValueError):
            horizontal = self.config.closure_horizontal_tolerance
            vertical = self.config.closure_vertical_tolerance
            orientation = self.config.closure_orientation_tolerance
        normal = self._acquisition_plane_normal()
        return bool(np.linalg.norm(self._acquisition_tangent(error)) <= horizontal
                    and abs(float(error @ normal)) <= vertical
                    and quat_distance(target_pose[3:], current[3:]) <= orientation)

    def _relative_visual_closure_intercept(self, target, observation, now):
        evidence = {'contact_prediction_frame': 'relative_visual_minus_encoder_fk_tcp',
                    'object_ground_truth_used': False}
        lead = self.config.closure_contact_lead_time
        visual = self._current_visual_acquisition(target, observation)
        if lead <= 0. or visual is None or self._grasp_pose is None:
            return None, evidence
        position, velocity = visual
        velocity = self._acquisition_tangent(velocity)
        try:
            height = float(target.get('grasp_height'))
            tolerance = float(target.get('closure_horizontal_tolerance', self.config.closure_horizontal_tolerance))
        except (TypeError, ValueError):
            return None, evidence
        if not np.isfinite([height, tolerance]).all() or velocity @ velocity < 1e-6:
            return None, evidence
        solver = getattr(self.env, 'ik', getattr(self.env, 'ik_solver', None))
        jacobian_fn = getattr(solver, 'compute_jacobian', None)
        qd = np.asarray(observation.get('joint_vel', []), dtype=float)
        jacobian = (None if not callable(jacobian_fn)
                    else np.asarray(jacobian_fn(self._joint_positions(observation)), dtype=float))
        if (jacobian is None or jacobian.shape != (6, 7) or qd.shape != (7,)
                or not np.isfinite(jacobian).all() or not np.isfinite(qd).all()):
            evidence['contact_prediction_status'] = 'encoder_fk_velocity_unavailable'
            # Preserve the inherited acquisition timing when optional velocity
            # is unavailable; do not add a new hardware admission condition.
            return super()._visual_closure_intercept(target, observation, now), evidence
        tcp_velocity = jacobian[:3] @ qd
        relative_velocity = self._acquisition_tangent(velocity-tcp_velocity)
        contact = self._grasp_pose.copy()
        contact[:3] = position + relative_velocity*lead + self._acquisition_plane_normal()*height
        current = self._ee_pose(observation)
        ready = bool(np.linalg.norm(self._acquisition_tangent(contact[:3]-current[:3])) <= min(.010, tolerance)
                     and self._closure_pose_reached(contact, observation, target)
                     and self._window_valid(target, observation))
        evidence.update(tcp_velocity_base_m_s=tcp_velocity,
                        closure_relative_velocity_base_m_s=relative_velocity,
                        predicted_contact_pose=contact, contact_prediction_status='valid')
        return (ready, float(now)+lead), evidence

    def _visual_closure_intercept(self, target, observation, now):
        if not self._visual_acquisition_tracking:
            return super()._visual_closure_intercept(target, observation, now)
        result, self._visual_closure_diagnostics = self._relative_visual_closure_intercept(
            target, observation, now)
        return result

    def reset(self, seed=None):
        self._prepared_visual_tracking_reference = None
        lead = getattr(self, '_following_lead', None)
        if lead is not None:
            lead.reset()
        self._following_lead_diagnostics = None
        self._visual_acquisition_diagnostics = None
        self._visual_closure_diagnostics = None
        return super().reset(seed=seed)

    def _clear_target(self):
        self._prepared_visual_tracking_reference = None
        lead = getattr(self, '_following_lead', None)
        if lead is not None:
            lead.reset()
        self._following_lead_diagnostics = None
        self._visual_acquisition_diagnostics = None
        self._visual_closure_diagnostics = None
        return super()._clear_target()


class VisualAcquisitionSortingPolicy(VisualAcquisitionMixin, BaselineSortingPolicy):
    """Hardware entry for the historical visual following/command integrator.

    There is no simulator motor preload or object truth dependency. Width and
    force confirmation retain the hardware context's measured feedback.
    Belt-plane tangent-priority allocation is an optional experiment; the
    historical radial allocation remains the default.
    """
    def __init__(self, config: BaselinePolicyConfig | Mapping[str, Any] | None = None,
                 *, acquisition_translation_step_m=.050,
                 acquisition_command_integrator=True, visual_acquisition_tracking=True,
                 visual_servo_bias_compensation=True, belt_plane_normal=None,
                 acquisition_tangent_priority=False, **kwargs):
        step = float(acquisition_translation_step_m)
        if not np.isfinite(step) or not 0. < step <= .050:
            raise ValueError('acquisition_translation_step_m must be finite and in (0, .050]')
        self._acquisition_translation_step_m = step
        self._acquisition_command_integrator = bool(acquisition_command_integrator)
        self._acquisition_tangent_priority = bool(acquisition_tangent_priority)
        self._visual_acquisition_tracking = bool(visual_acquisition_tracking)
        self._visual_servo_bias_compensation = bool(visual_servo_bias_compensation)
        self._belt_plane_normal = None
        if belt_plane_normal is not None:
            normal = np.asarray(belt_plane_normal, dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-8:
                raise ValueError('belt_plane_normal must be a nonzero finite base XYZ vector')
            self._belt_plane_normal = normal.copy()/np.linalg.norm(normal)
        self._following_lead = None
        super().__init__(config, **kwargs)
        self._following_lead = VisualFollowingLeadAdapter(
            initial_horizon_s=self.config.tracking_velocity_feedforward_time,
            maximum_horizon_s=self.config.tracking_synchronization_horizon,
            filter_time_s=.2)

    def acquisition_diagnostics(self):
        """Expose visual/request/feedback evidence without influencing control."""
        return {'visual_acquisition': self._visual_acquisition_diagnostics,
                'pregrasp_execution_timing': dict(self._pregrasp_execution_timing),
                'visual_following_lead': self._following_lead_diagnostics,
                'visual_closure': self._visual_closure_diagnostics,
                'belt_plane_normal_base': self._acquisition_plane_normal(),
                'minimum_simulated_motor_hold_effort_used': False}

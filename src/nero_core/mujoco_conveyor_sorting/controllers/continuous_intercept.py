"""Certified continuous approach to a moving contact, without a high stop.

The worker owns the entire curve and its future capture pose. This module
only schedules immutable requests and sends samples through the existing
controller and measured-state geometry checks. Closure and loaded motion
remain in the shared policy. It is not a CTMP or MPC implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np

from ._math import quat_distance
from .trajectory_splicing import prefix_segment


@dataclass(frozen=True)
class InterceptPlan:
    request: object
    result: object
    start: float
    end: float
    prefix: InterceptPlan | None = None

    def segment_at(self, timestamp):
        return prefix_segment(self, timestamp)

    def sample(self, timestamp):
        segment = self.segment_at(timestamp)
        return segment.result.continuation_trajectory.sample_full(float(timestamp)-segment.start)


class ContinuousInterceptMixin:
    def _reset_continuous_intercept(self):
        self._continuous_plan = None
        self._continuous_request = None
        self._continuous_parent = None
        self._continuous_replacement = None
        self._continuous_hold_q = None
        self._continuous_compute_s = 0.
        self._continuous_compute_overrun_s = 0.
        self._continuous_duration_s = 0.
        self._continuous_finished = False
        self._continuous_armed = False
        self._continuous_events = []

    def _discard_continuous_intercept(self):
        # The caller invalidates the shared worker generation as well.
        self._continuous_plan = None
        self._continuous_request = None
        self._continuous_parent = None
        self._continuous_replacement = None
        self._continuous_hold_q = None
        self._continuous_finished = False
        self._continuous_armed = False

    def _continuous_intercept_enabled(self):
        return bool(self.config.planning_mode != 'full_preplanned'
                    and self.config.moving_approach_mode == 'continuous'
                    and self.config.dynamic_intercept_tracking
                    and self.config.action_mode == 'joint_position')

    def _intercept_note(self, name, now, **details):
        self._continuous_events.append({'event': name, 'timestamp': float(now), **details})
        self._emit(name, now, str(details.get('reason', details.get('request_id', ''))))

    def _continuous_intercept_info(self):
        plan = getattr(self, '_continuous_plan', None)
        request = getattr(self, '_continuous_request', None)
        return {'enabled': self._continuous_intercept_enabled(),
                'active': plan is not None,
                'closure_armed': getattr(self, '_continuous_armed', False),
                'pending_request_id': None if request is None else request.request_id,
                'start_s': None if plan is None else plan.start,
                'capture_s': None if plan is None else plan.end,
                'compute_forecast_s': getattr(self, '_continuous_compute_s', 0.),
                'compute_overrun_s': getattr(self, '_continuous_compute_overrun_s', 0.),
                'duration_forecast_s': getattr(self, '_continuous_duration_s', 0.),
                'events': list(getattr(self, '_continuous_events', []))}

    def _current_contact_pose(self, observation):
        """Read a publication-time grasp, preserving its measured timestamps."""
        target = self._find_target(observation, self._target_id)
        if target is None:
            return None
        prediction = self._prediction_from_mapping(observation, self._target_id)
        if isinstance(prediction, Mapping):
            # Generic perception's current_estimated_pose is an OBJECT pose;
            # only this explicit contract identifies a current TCP grasp pose.
            current = prediction.get('current_ee_grasp_pose')
            if current is not None:
                current = np.asarray(current, dtype=float)
                if current.shape == (7,) and np.isfinite(current).all():
                    return current.copy()
        pose = self._grasp_target_pose(target, observation)
        if pose is None:
            return None
        pose = np.asarray(pose, dtype=float).copy()
        # Generic adapters may expose only an explicitly timed future grasp.
        # Remove that forecast once before predicting the new capture time.
        stamp = prediction.get('estimated_grasp_time') if isinstance(prediction, Mapping) else None
        if stamp is None and isinstance(target, Mapping):
            stamp = target.get('estimated_grasp_time')
        if stamp is not None:
            pose[:3] -= self._belt_velocity(observation) * max(
                0., float(stamp)-self._timestamp_without_advancing(observation))
        return pose

    def _intercept_owner_valid(self, plan, observation):
        request = plan.request
        if (request.generation != self._planning_generation
                or request.scene_epoch != int(self._observed_scene_epoch or 0)
                or not self._same_target_id(request.target_id, self._target_id)):
            return False
        reuse = getattr(self.env, 'can_reuse_planning_certificate', None)
        return bool(not callable(reuse) or reuse(request, plan.result, observation))

    def _commit_intercept_plan(self, plan, observation):
        callback = getattr(self.env, 'refine_selected_grasp_plan', None)
        if callable(callback):
            value = callback(self._target_id,
                max(0., plan.end-self._timestamp_without_advancing(observation)),
                self.config.intercept_deadline_margin,
                requested_time=plan.end, requested_pose=plan.result.continuation_target_pose)
            if not isinstance(value, Mapping) or not value.get('accepted', False):
                return False
            # Generic simulator callbacks return an OBJECT pose under the
            # historical predicted_grasp_pose key. Only an explicit TCP
            # return contract can be compared with our certified TCP endpoint.
            returned_pose = value.get('predicted_ee_grasp_pose')
            if returned_pose is not None:
                returned_pose = np.asarray(returned_pose, dtype=float)
                endpoint = plan.result.continuation_target_pose
                if (returned_pose.shape != (7,) or not np.isfinite(returned_pose).all()
                        or np.linalg.norm(returned_pose[:3]-endpoint[:3])
                            > self.config.trajectory_replan_position_tolerance
                        or quat_distance(returned_pose[3:], endpoint[3:])
                            > self.config.trajectory_replan_orientation_tolerance):
                    return False
        self._continuous_plan = plan
        self._pregrasp_capture_time_s = plan.end
        self._continuous_duration_s = max(self._continuous_duration_s,
                                        plan.result.continuation_trajectory.duration)
        return True

    def _poll_intercept(self, observation, now):
        request = self._continuous_request
        if request is not None:
            result = self._planning_worker.poll()
            if result is not None:
                effective = self._completed_planning_request(result, request)
                if effective is None:
                    self._planning_stale_results += 1
                    return
                self._continuous_request = self._planning_request = None
                # Include preparation, worker queuing and receipt, not only
                # the engine's internal calculation timer. Both clocks are
                # the existing policy elapsed time; no observation is restamped.
                delivery_s = max(float(result.planning_time_s),
                    now-float(request.intercept_request_time_s))
                self._continuous_compute_s = max(self._continuous_compute_s, delivery_s)
                if request.continuation_compute_forecast_s > 0.:
                    self._continuous_compute_overrun_s = max(self._continuous_compute_overrun_s,
                        delivery_s-request.continuation_compute_forecast_s)
                diagnostics = (result.diagnostics or {}).get('continuous_intercept', {})
                self._continuous_duration_s = max(self._continuous_duration_s,
                    float(diagnostics.get('suggested_duration_s', 0.) or 0.))
                curve = result.continuation_trajectory
                if curve is not None:
                    # A fully solved but late route still measured the time
                    # required by that motion. Retrying must not shrink it.
                    self._continuous_duration_s = max(self._continuous_duration_s, curve.duration)
                valid = bool(effective is not None and result.success
                    and result.path_fully_validated and result.continuation_fully_validated
                    and curve is not None and result.continuation_target_pose is not None)
                if valid:
                    plan = InterceptPlan(effective, result,
                        float(effective.continuation_start_time_s),
                        float(effective.planned_target_time_s))
                    valid = abs(plan.start+curve.duration-plan.end) <= 1e-7
                if not valid:
                    self._intercept_note('continuous_intercept_rejected', now,
                        request_id=request.request_id, reason=result.error or 'incomplete_certificate',
                        planning_time_s=float(result.planning_time_s))
                    if self._continuous_plan is None:
                        self._record_motion_plan_failure()
                elif now > plan.start + 1e-9:
                    # A late curve is never replayed from its middle. A new
                    # request uses measured calculation cost and a NEW pose.
                    self._intercept_note('continuous_intercept_late', now,
                        request_id=request.request_id, late_s=now-plan.start)
                elif self._continuous_parent is not self._continuous_plan:
                    self._intercept_note('continuous_intercept_rejected', now,
                        request_id=request.request_id, reason='parent_changed')
                elif self._continuous_plan is None:
                    if self._intercept_owner_valid(plan, observation) and self._commit_intercept_plan(plan, observation):
                        self._record_motion_plan_success()
                        self._intercept_note('continuous_intercept_prepared', now,
                            request_id=request.request_id, start_s=plan.start, capture_s=plan.end,
                            duration_s=curve.duration, terminal_velocity_m_s=list(effective.continuation_velocity_m_s))
                    else:
                        self._intercept_note('continuous_intercept_rejected', now,
                            request_id=request.request_id, reason='reservation_or_owner_changed')
                else:
                    self._continuous_replacement = plan
        replacement = self._continuous_replacement
        parent = self._continuous_plan
        if replacement is None:
            return
        # Install a piecewise reference BEFORE its splice. Samples before
        # that clock still use the identical parent prefix. A non-grid splice
        # therefore needs no late change between two diverged references.
        if (now <= replacement.start+1e-9 and parent is not None
                and self._intercept_owner_valid(replacement, observation)
                and all(np.allclose(a, b, atol=1e-8, rtol=0.) for a, b in
                        zip(parent.sample(replacement.start), replacement.sample(replacement.start)))
                and self._commit_intercept_plan(replace(replacement, prefix=parent), observation)):
            self._intercept_note('continuous_intercept_splice_prepared', now,
                request_id=replacement.request.request_id, scheduled_s=replacement.start,
                prepared_before_splice_s=max(0., replacement.start-now))
        else:
            differences = None if parent is None else [
                float(np.max(np.abs(a-b)))
                for a, b in zip(parent.sample(replacement.start), replacement.sample(replacement.start))]
            self._intercept_note('continuous_intercept_rejected', now,
                request_id=replacement.request.request_id, reason='splice_state_or_owner_changed',
                scheduled_splice_s=replacement.start, late_s=max(0., now-replacement.start),
                reference_q_v_a_difference=differences)
        self._continuous_replacement = None

    def _submit_intercept(self, observation, now, contact):
        parent = self._continuous_plan
        dt = self._execution_dt(observation)
        compute = max(self._continuous_compute_s+self._continuous_compute_overrun_s,
                      self.config.trajectory_minimum_duration, dt)
        start = now + compute + dt
        velocity = self._belt_velocity(observation)
        duration = max(self.config.descent_lead_time, self._continuous_duration_s,
            float(np.linalg.norm(contact[:3]-self._ee_pose(observation)[:3])) /
            self.config.cartesian_linear_speed)
        end = start + duration
        if parent is not None:
            # Updates retain the parent's intercept clock. If too little time
            # remains, finish its complete curve and use original visual servo.
            if start >= parent.end:
                return
            end, duration = parent.end, parent.end-start
            q, v, a = parent.sample(start)
        else:
            if self._continuous_hold_q is None:
                self._continuous_hold_q = self._joint_positions(observation).copy()
            q, v, a = self._continuous_hold_q.copy(), np.zeros(7), np.zeros(7)
        capture = contact.copy()
        capture[:3] += velocity*(end-now)
        target = self._find_target(observation, self._target_id)
        window = target.get('grasp_window', {}) if isinstance(target, Mapping) else {}
        latest = float(window.get('end_time', float('inf')))-self.config.intercept_deadline_margin
        if end > latest:
            self._last_error = 'continuous capture lies beyond the existing grasp window'
            self._record_motion_plan_failure()
            return
        snapshot = self._future_open_planning_snapshot(q)
        request = self._make_planning_request(capture, observation,
            minimum_tcp_height=min(float(self._ee_pose(observation)[2]), float(capture[2]))-.012)
        if snapshot is None or request is None:
            return
        arm, full, epoch = snapshot
        request = replace(request, continuous_intercept=True, stage='pregrasp',
            start_q=arm, full_qpos=full, scene_epoch=epoch, target_poses=(capture,),
            continuation_target_poses=None, continuation_minimum_tcp_height=None,
            timed_continuation_duration_s=duration, fixed_high_continuation=False,
            continuation_start_time_s=start, planned_target_time_s=end,
            continuation_start_velocity=tuple(v), continuation_start_acceleration=tuple(a),
            continuation_velocity_m_s=tuple(velocity), intercept_velocity_m_s=None,
            intercept_request_time_s=now, intercept_account_for_planning_time=parent is None,
            intercept_descent_duration_s=duration+dt,
            continuation_compute_forecast_s=self._continuous_compute_s,
            route_waypoint_hints=(), route_hints_after_direct=False)
        worker = self._ensure_planning_worker()
        if worker is not None and worker.submit(request):
            self._continuous_request = self._planning_request = request
            self._continuous_parent = parent
            self._intercept_note('continuous_intercept_requested', now,
                request_id=request.request_id, start_s=start, capture_s=end,
                predicted_contact_pose=capture.tolist(), observed_contact_pose=contact.tolist(),
                velocity_m_s=velocity.tolist(), compute_forecast_s=compute,
                future_splice=parent is not None)

    def _arm_on_continuous_intercept(self, observation):
        if (not self._continuous_intercept_enabled() or self._stage.value != 'tracking_descent'
                or self._continuous_finished):
            return None
        now = self._timestamp_without_advancing(observation)
        clock = observation.get('dispatch_clock', {})
        send = max(now, float(clock.get('scheduled_dispatch_policy_time_s', now)))
        self._poll_intercept(observation, now)
        plan = self._continuous_plan
        contact = self._current_contact_pose(observation)
        if plan is not None and not self._intercept_owner_valid(plan, observation):
            self._last_error = 'continuous interception certificate changed'
            self._discard_continuous_intercept()
            self._cancel_async_plan()
            self._record_motion_plan_failure()
            return self._joint_positions(observation).copy()
        if plan is not None and now >= plan.end:
            self._continuous_finished = True
            self._continuous_plan = None
            if self._continuous_request is not None:
                self._cancel_async_plan()
                self._continuous_request = None
            self._cartesian_command_pose = None
            self._last_ik_solution = None
            self._intercept_note('continuous_intercept_visual_handoff', now, capture_s=plan.end)
            return None
        if contact is not None and self._continuous_request is None and self._continuous_replacement is None:
            correction = float('inf')
            if plan is not None:
                updated = contact[:3]+self._belt_velocity(observation)*(plan.end-now)
                correction = np.linalg.norm(updated-plan.result.continuation_target_pose[:3])
            if correction > self.config.trajectory_replan_position_tolerance:
                self._submit_intercept(observation, now, contact)
        plan = self._continuous_plan
        if plan is None or (send < plan.start and plan.prefix is None):
            if self._continuous_hold_q is None:
                self._continuous_hold_q = self._joint_positions(observation).copy()
            return self._continuous_hold_q.copy()
        q, _, _ = plan.sample(min(send, plan.end))
        current = self._joint_positions(observation)
        controller = getattr(self.env, 'robot_controller', None)
        preview = getattr(controller, 'preview_joint_step', None)
        passthrough = False
        if callable(preview):
            previous, velocity = self._joint_command_history(observation, current)
            q, _, _, _, _ = preview(q, previous, velocity, self._execution_dt(observation))
            q, _, _, passthrough, _ = preview(q, previous, velocity, self._execution_dt(observation))
        q = np.asarray(q, dtype=float)
        pose = np.asarray(self.env.ik.forward_pose(q), dtype=float)
        feedback = self._ee_pose(observation)
        request = plan.request
        valid = (np.max(np.abs(q-current)) <= self.config.max_ik_joint_step
            and np.linalg.norm(pose[:3]-feedback[:3]) <= getattr(
                self, '_acquisition_translation_step_m', self.config.max_translation_step)
            and quat_distance(pose[3:], feedback[3:]) <= self.config.max_rotation_step
            and self._global_edge_is_safe(current, q,
                minimum_tcp_height=request.minimum_tcp_height,
                minimum_joint_margin=request.minimum_joint_margin,
                minimum_downward_alignment=request.minimum_downward_alignment))
        if not valid:
            translation_limit = getattr(self, '_acquisition_translation_step_m', self.config.max_translation_step)
            # The reference clock is not proof that the physical arm followed
            # it. Forecast a slower COMPLETE future curve from the observed
            # envelope excess, retaining every original envelope value. The
            # changed endpoint/time will receive a new worker certificate.
            ratio = max(float(np.max(np.abs(q-current)))/self.config.max_ik_joint_step,
                float(np.linalg.norm(pose[:3]-feedback[:3]))/translation_limit,
                float(quat_distance(pose[3:], feedback[3:]))/self.config.max_rotation_step)
            if ratio > 1.:
                self._continuous_duration_s = max(self._continuous_duration_s,
                    plan.result.continuation_trajectory.duration*ratio)
            self._intercept_note('continuous_intercept_live_rejected', now,
                request_id=request.request_id,
                joint_feedback_lead_rad=float(np.max(np.abs(q-current))),
                tcp_feedback_lead_m=float(np.linalg.norm(pose[:3]-feedback[:3])),
                rotation_feedback_lead_rad=float(quat_distance(pose[3:], feedback[3:])),
                candidate_q=q.tolist(), feedback_q=current.tolist(),
                next_duration_forecast_s=self._continuous_duration_s)
            self._last_error = 'continuous interception left current geometry or feedback envelope'
            self._discard_continuous_intercept()
            self._continuous_hold_q = current.copy()
            self._cancel_async_plan()
            self._record_motion_plan_failure()
            return current.copy()
        self._last_ik_solution = q.copy()
        self._cartesian_command_pose = pose.copy()
        self._continuous_armed = True
        self._arm_command_is_time_parameterized = bool(passthrough)
        self._record_motion_plan_success()
        return q.copy()

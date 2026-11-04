"""Execute a fully checked descent while its replacement is computed.

Engineering adaptation of future-state splicing (ASAP-MPC) and timed dynamic
grasping (Grasping on Time), not an implementation of either paper's optimizer.
The existing planner certifies the entire curve; the existing controller and
live geometry checks still own every sent command. No device calls live here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._math import quat_distance


@dataclass(frozen=True)
class DescentPlan:
    request: Any
    result: Any
    start: float
    end: float
    capture_pose: np.ndarray
    feedback_start: bool = False

    def sample(self, timestamp):
        return self.result.continuation_trajectory.sample_full(timestamp - self.start)


class TimedDescentMixin:
    def _reset_timed_descent(self):
        self._descent_plan = None
        self._descent_update_request = None
        self._descent_update_parent = None
        self._descent_replacement = None
        self._descent_compute_s = 0.
        self._descent_duration_s = 0.
        self._descent_high_pose = None
        self._descent_rebook_pending = False
        self._descent_high_hold_q = None
        self._descent_feedback_request_id = None
        self._descent_trace = []

    def _timed_descent_enabled(self, observation):
        return bool(self.config.timed_descent_enabled
                    and self.config.dynamic_intercept_tracking
                    and not self.config.prepare_high_only
                    and isinstance(observation.get('dispatch_clock'), dict)
                    and callable(getattr(self.env, 'preview_fixed_high_grasp_plan', None)))

    def _descent_note(self, event, now, **details):
        self._descent_trace.append({'event': event, 'timestamp': float(now), **details})
        self._emit(event, now, str(details.get('reason', details.get('request_id', ''))))

    def _discard_timed_descent(self):
        if self._descent_update_request is not None:
            self._cancel_async_plan()
        self._descent_plan = None
        self._descent_update_request = None
        self._descent_update_parent = None
        self._descent_replacement = None
        self._descent_rebook_pending = False
        self._descent_high_hold_q = None
        self._descent_feedback_request_id = None

    def _request_descent_rebooking(self, plan, observation, remaining):
        """Retire an expired clock, not the already certified high route.

        The original fixed-high worker will certify a NEW future low pose.
        Cancelling an obsolete splice must not increment the route generation:
        the arm can still finish its unchanged high approach while we plan.
        """
        now = self._timestamp_without_advancing(observation)
        if self._descent_update_request is not None:
            worker = self._planning_worker
            if worker is not None:
                worker.cancel_pending()
            if self._planning_request is self._descent_update_request:
                self._planning_request = None
        self._descent_plan = None
        self._descent_update_request = None
        self._descent_update_parent = None
        self._descent_replacement = None
        self._descent_rebook_pending = True
        self._fixed_high_ready = False
        self._fixed_high_validated_low = None
        self._fixed_high_validated_capture_time = None
        self._descent_note('timed_descent_rebook_requested', now,
            previous_request_id=plan.request.request_id,
            previous_start_s=plan.start, previous_capture_s=plan.end,
            lateness_s=max(0., now-plan.start), remaining_high_s=remaining,
            descent_compute_forecast_s=self._descent_compute_s,
            high_route_generation_preserved=True)

    def _timed_descent_feedback_high_ready(self, observation):
        """A descent certified from measured joints owns its own high start.

        Do not require equality to the OLD nominal TCP waypoint as well. Its
        original joint/velocity tolerance and every live feedback-to-command
        geometry check still apply to this new, fully checked descent.
        """
        plan = self._descent_plan
        return bool(plan is not None and plan.feedback_start
            and self._stage.value == 'pregrasp'
            and self._fixed_high_ready
            and self._descent_owner_valid(plan, observation)
            and self._descent_high_joints_settled(observation, plan.request.start_q))

    def _descent_high_joints_settled(self, observation, anchor):
        return bool(np.max(np.abs(self._joint_positions(observation)-anchor))
            <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)
            and self._joint_velocity_is_settled(
                observation, self.config.planner_waypoint_velocity_tolerance))

    def _descent_high_remaining(self, observation):
        anchor = self._descent_high_hold_q
        if anchor is None:
            return float(self._safe_path_remaining_motion_time(observation) or 0.)
        if self._descent_high_joints_settled(observation, anchor):
            return 0.
        return self._edge_execution_duration(
            self._joint_positions(observation), anchor, stage='pregrasp'
        ) * max(1., self._pregrasp_motion_time_scale)

    def _arm_on_descent_high_hold(self, observation):
        """Keep ONE measured high target while its future descent is checked."""
        q = self._descent_high_hold_q
        if q is None or self._stage.value != 'pregrasp':
            return None
        current = self._joint_positions(observation)
        if not self._global_edge_is_safe(current, q,
                minimum_tcp_height=self._safe_path_minimum_tcp_height,
                minimum_joint_margin=self._safe_path_minimum_joint_margin,
                minimum_downward_alignment=self._safe_path_minimum_downward_alignment):
            self._last_error = 'measured high hold left current geometry'
            self._record_motion_plan_failure()
            return current.copy()
        # The ordinary controller admits this hold using its actual sent
        # q/v history. Reading feedback never resets that history.
        return q.copy()

    def _make_descent_plan(self, result, request, observation):
        curve = result.continuation_trajectory
        if (curve is None or result.continuation_path is None
                or not result.success or not result.continuation_fully_validated
                or request.planned_target_time_s is None):
            return None
        end = float(request.planned_target_time_s)
        start = end - curve.duration
        if request.continuation_start_time_s is not None:
            start = float(request.continuation_start_time_s)
            if abs(start + curve.duration - end) > 1e-7:
                return None
        low = result.continuation_target_pose.copy()
        # This reconstructs the request's capture pose; never relabel it as
        # the latest observation. Live closure uses the latest real target.
        low[:3] -= np.asarray(request.continuation_velocity_m_s) * self.config.tracking_velocity_feedforward_time
        return DescentPlan(request, result, start, end, low)

    def _accept_initial_descent(self, result, request, observation):
        if not self._timed_descent_enabled(observation):
            return
        self._descent_plan = self._make_descent_plan(result, request, observation)
        rebooked = self._descent_rebook_pending
        if (self._descent_plan is not None
                and request.request_id == self._descent_feedback_request_id
                and self._descent_high_hold_q is not None):
            self._descent_plan = replace(self._descent_plan, feedback_start=True)
        self._descent_high_pose = result.target_pose.copy()
        self._record_descent_computation(result, request)
        if self._descent_plan is not None:
            self._descent_rebook_pending = False
            self._descent_note('timed_descent_prepared',
                self._timestamp_without_advancing(observation),
                request_id=request.request_id, start_s=self._descent_plan.start,
                capture_s=self._descent_plan.end,
                duration_s=result.continuation_trajectory.duration)
            if rebooked:
                self._descent_note('timed_descent_rebooked',
                    self._timestamp_without_advancing(observation),
                    request_id=request.request_id, start_s=self._descent_plan.start,
                    capture_s=self._descent_plan.end,
                    duration_s=result.continuation_trajectory.duration,
                    from_measured_high=self._descent_plan.feedback_start)

    def _record_descent_computation(self, result, request):
        # Measure the continuation inside a full high search as well. Do not
        # charge RRT/high IK to a descent, or erase history on a new high route.
        timing = (result.diagnostics or {}).get('continuation_timing', {})
        measured = timing.get('max_check_s',
            float(result.planning_time_s) if request.fixed_high_continuation else 0.)
        if request.fixed_high_continuation:
            # A delivered descent may have needed its own timing revision.
            # Forecast the observed whole job, not only its shorter first pass.
            measured = max(float(measured), float(result.planning_time_s))
        self._descent_compute_s = max(self._descent_compute_s, float(measured))
        if result.continuation_trajectory is not None:
            self._descent_duration_s = max(self._descent_duration_s,
                                          result.continuation_trajectory.duration)

    def _new_descent_duration(self):
        """Forecast a new curve using the already measured approach slowdown.

        This is an arrival-time estimate, not motor identification. Use it
        only before certification: never stretch or relabel an active curve.
        Worker dynamics/geometry validation still determines feasibility.
        """
        scale = max(1., float(self._pregrasp_motion_time_scale))
        return max(self.config.descent_lead_time * scale, self._descent_duration_s)

    def _preview_new_descent(self, observation, high, remaining):
        """Reserve after BOTH concurrent high travel and descent calculation."""
        now = self._timestamp_without_advancing(observation)
        duration = self._new_descent_duration()
        proposal = self.env.preview_fixed_high_grasp_plan(
            self._target_id, high, remaining, duration,
            capture_high=self._pregrasp_capture_high_active)
        if not proposal.get('accepted', False):
            return proposal
        earliest = now + max(remaining, self._descent_compute_s) + self._execution_dt(observation) + duration
        if earliest > float(proposal['estimated_grasp_time']):
            proposal = self.env.preview_fixed_high_grasp_plan(
                self._target_id, high, remaining, duration,
                capture_high=self._pregrasp_capture_high_active, capture_time=earliest)
        return proposal

    def _descent_owner_valid(self, plan, observation):
        request = plan.request
        if (request.generation != self._planning_generation
                or request.scene_epoch != int(self._observed_scene_epoch or 0)
                or not self._same_target_id(request.target_id, self._target_id)):
            return False
        reuse = getattr(self.env, 'can_reuse_planning_certificate', None)
        return bool(not callable(reuse) or reuse(request, plan.result, observation))

    def _commit_descent_reservation(self, plan):
        # The plan owns its endpoint/time. New publications retain their own
        # position and timestamps and separately drive the next request.
        # During descent _grasp_pose is refreshed by the original visual
        # policy each tick. Overwriting it with the curve endpoint would
        # bias the original closure gate when motion stops (zero velocity
        # intentionally disables the separate moving-contact predicate).
        if self._stage.value == 'pregrasp':
            self._grasp_pose = plan.capture_pose.copy()
        self._pregrasp_capture_time_s = plan.end

    def _poll_descent_update(self, observation, now):
        request = self._descent_update_request
        if request is not None:
            result = self._planning_worker.poll()
            if result is not None:
                if self._completed_planning_request(result, request) is None:
                    self._planning_stale_results += 1
                    return
                self._record_descent_computation(result, request)
                self._descent_update_request = self._planning_request = None
                candidate = self._make_descent_plan(result, request, observation)
                if candidate is not None and self._descent_plan is not None:
                    candidate = replace(candidate,
                        feedback_start=self._descent_plan.feedback_start)
                if candidate is None:
                    self._descent_note('timed_descent_update_rejected', now,
                        request_id=request.request_id, reason=result.error or 'incomplete_or_cancelled')
                elif now > candidate.start + 1e-9:
                    self._descent_note('timed_descent_update_late', now,
                        request_id=request.request_id, deadline_s=candidate.start,
                        late_s=now-candidate.start)
                elif self._descent_update_parent is not self._descent_plan:
                    self._descent_note('timed_descent_update_rejected', now,
                        request_id=request.request_id, reason='executing_parent_changed')
                else:
                    self._descent_replacement = candidate
        candidate = self._descent_replacement
        if candidate is None:
            return
        parent = self._descent_plan
        stationary_prefix = (parent is not None and now < parent.start
            and candidate.start <= parent.start + 1e-9
            and all(np.allclose(a, b, atol=1e-8, rtol=0.)
                    for a, b in zip(parent.sample(now), candidate.sample(now))))
        if now < candidate.start and not stationary_prefix:
            return
        # C2 equality is checked at the same absolute clock, independently of
        # the current feedback. Feedback is checked on the actual sent chord.
        if (parent is not None and self._descent_owner_valid(candidate, observation)
                and all(np.allclose(a, b, atol=1e-8, rtol=0.)
                        for a, b in zip(parent.sample(candidate.start), candidate.sample(candidate.start)))):
            self._descent_plan = candidate
            self._commit_descent_reservation(candidate)
            self._descent_note('timed_descent_spliced', now, request_id=candidate.request.request_id,
                scheduled_s=candidate.start, dispatch_lateness_s=max(0., now-candidate.start),
                unchanged_stationary_prefix=stationary_prefix)
        else:
            self._descent_note('timed_descent_update_rejected', now,
                request_id=candidate.request.request_id, reason='splice_state_or_owner_changed')
        self._descent_replacement = None

    def _refresh_timed_descent(self, observation):
        """One background update; submission never erases the active plan."""
        if not self._timed_descent_enabled(observation):
            return False
        now = self._timestamp_without_advancing(observation)
        self._poll_descent_update(observation, now)
        plan = self._descent_plan
        if plan is None:
            return False  # Original fully checked path remains the fallback.
        if not self._descent_owner_valid(plan, observation):
            self._discard_timed_descent()
            self._fixed_high_ready = False
            return False
        high_stage = self._stage.value == 'pregrasp'
        remaining = self._descent_high_remaining(observation) if high_stage else 0.
        # A missed appointment needs a new future endpoint and full descent,
        # not a new high approach or a jump into the expired reference.
        if high_stage and now + remaining > plan.start + self._execution_dt(observation):
            self._request_descent_rebooking(plan, observation, remaining)
            return False
        self._commit_descent_reservation(plan)
        self._fixed_high_ready = True
        self._pregrasp_execution_timing = {
            'timestamp': now, 'mode': 'timed_descent', 'accepted': True,
            'reason': 'executing_plan_retained', 'descent_start_s': plan.start,
            'estimated_grasp_time_s': plan.end,
            'background_pending': self._descent_update_request is not None}
        if self._descent_update_request is not None or self._descent_replacement is not None or now >= plan.end:
            return True
        proposal = self.env.preview_fixed_high_grasp_plan(
            self._target_id, self._descent_high_pose, remaining,
            self.config.descent_lead_time, capture_time=plan.end)
        if not proposal.get('accepted', False):
            return True
        capture = np.asarray(proposal['predicted_grasp_pose'], dtype=float).copy()
        capture[3:] = plan.capture_pose[3:]
        low = capture.copy()
        low[:3] += self._belt_velocity(observation) * self.config.tracking_velocity_feedforward_time
        correction = float(np.linalg.norm(low[:3]-plan.result.continuation_target_pose[:3]))
        self._pregrasp_execution_timing['low_correction_m'] = correction
        if correction <= self.config.trajectory_replan_position_tolerance:
            return True
        # Use measured compute history as a forecast, never as a hard bound.
        # A late result is discarded; the active complete path is retained.
        budget = max(self._descent_compute_s, self._execution_dt(observation))
        splice = max(plan.start, now + budget + self._execution_dt(observation))
        if splice >= plan.end:
            return True
        q, v, a = plan.sample(splice)
        solver = getattr(self.env, 'ik', None)
        if solver is None or not callable(getattr(solver, 'forward_pose', None)):
            return True
        pose = np.asarray(solver.forward_pose(q), dtype=float)
        request = self._make_planning_request(pose, observation,
            minimum_joint_margin=plan.request.minimum_joint_margin,
            minimum_tcp_height=float(low[2])-.015,
            minimum_downward_alignment=plan.request.minimum_downward_alignment)
        snapshot = self._future_open_planning_snapshot(q)
        if request is None or snapshot is None:
            return True
        arm, full, epoch = snapshot
        request = replace(request, start_q=arm, full_qpos=full, scene_epoch=epoch,
            target_poses=(pose,), goal_seeds=(arm,), continuation_target_poses=(low,),
            continuation_minimum_tcp_height=float(low[2])-.015,
            fixed_high_continuation=True, intercept_velocity_m_s=None,
            planned_target_time_s=plan.end, intercept_account_for_planning_time=False,
            timed_continuation_duration_s=plan.end-splice, continuation_start_time_s=splice,
            continuation_start_velocity=tuple(v), continuation_start_acceleration=tuple(a))
        worker = self._ensure_planning_worker()
        if worker is not None and worker.submit(request):
            self._descent_update_request = self._planning_request = request
            self._descent_update_parent = plan
            self._descent_note('timed_descent_update_started', now,
                request_id=request.request_id, splice_s=splice, capture_s=plan.end,
                correction_m=correction, measured_compute_forecast_s=budget)
        return True

    def _arm_on_timed_descent(self, observation):
        plan = self._descent_plan
        if plan is None or self._stage.value != 'tracking_descent':
            return None
        now = self._timestamp_without_advancing(observation)
        clock = observation.get('dispatch_clock', {})
        send_time = max(now, float(clock.get('scheduled_dispatch_policy_time_s', now)))
        if send_time >= plan.end:
            # The complete path ends with v=a=0. Existing visual acquisition
            # handles final observed alignment/closure; it never closes merely
            # because the reference's capture clock has elapsed.
            self._discard_timed_descent()
            self._cartesian_command_pose = None
            self._last_ik_solution = None
            self._descent_note('timed_descent_visual_handoff', now, capture_s=plan.end)
            return None
        current = self._joint_positions(observation)
        q, _, _ = plan.sample(send_time)
        raw = q.copy()
        parameterized = False
        controller = getattr(self.env, 'robot_controller', None)
        preview = getattr(controller, 'preview_joint_step', None)
        if callable(preview):
            previous, velocity = self._joint_command_history(observation, current)
            limited, _, _, accepted, _ = preview(raw, previous, velocity, self._execution_dt(observation))
            q = np.asarray(limited).copy()
            # Irregular dispatch intervals need the original discrete limiter,
            # even for a smooth continuous reference. Preview has no side
            # effects; actual history is updated only by the real sender.
            checked, _, _, parameterized, _ = preview(
                q, previous, velocity, self._execution_dt(observation))
            if parameterized:
                q = np.asarray(checked).copy()
            else:
                # Do not assert passthrough when the shared controller cannot
                # provide it. The runner will validate its final ordinary output.
                q = raw.copy()
            if not accepted:
                self._descent_trace.append({'event': 'timed_descent_discrete_retime',
                    'timestamp': now, 'raw_reference_q': raw.tolist(),
                    'prepared_q': q.tolist(), 'passthrough': bool(parameterized),
                    'control_dt_s': self._execution_dt(observation)})
        request = plan.request
        pose = np.asarray(self.env.ik.forward_pose(q), dtype=float)
        feedback_pose = self._ee_pose(observation)
        translation_limit = getattr(self, '_acquisition_translation_step_m', self.config.max_translation_step)
        owner_valid = self._descent_owner_valid(plan, observation)
        joint_lead = float(np.max(np.abs(q-current)))
        translation_lead = float(np.linalg.norm(pose[:3]-feedback_pose[:3]))
        rotation_lead = float(quat_distance(pose[3:], feedback_pose[3:]))
        envelope_valid = (joint_lead <= self.config.max_ik_joint_step
            and translation_lead <= translation_limit
            and rotation_lead <= self.config.max_rotation_step)
        evidence = dict(request_id=request.request_id, reference_sample_s=send_time,
            joint_feedback_lead_rad=joint_lead, joint_limit_rad=self.config.max_ik_joint_step,
            tcp_feedback_lead_m=translation_lead, translation_limit_m=translation_limit,
            rotation_feedback_lead_rad=rotation_lead, rotation_limit_rad=self.config.max_rotation_step,
            candidate_q=q.tolist(), feedback_q=current.tolist(),
            candidate_tcp_pose=pose.tolist(), feedback_tcp_pose=feedback_pose.tolist(),
            candidate_sent=False)
        if owner_valid and not envelope_valid:
            # The physical arm did not follow the booked reference closely
            # enough. Do not send that sample, accumulate a fake IK failure,
            # or carry its old Cartesian origin into the following servo.
            # _arm_for_pose falls through in this SAME decision to the
            # existing measured-seed IK, 50 mm feedback bound, joint limits
            # and full actual-to-command geometry checks. Closure remains
            # gated by current vision/feedback, never by this handoff.
            self._discard_timed_descent()
            self._cartesian_command_pose = None
            self._last_ik_solution = None
            self._dynamic_vertical_retry = False
            self._arm_command_is_time_parameterized = False
            self._descent_note('timed_descent_feedback_handoff', now,
                reason='feedback_envelope', **evidence)
            return None
        valid = (owner_valid and self._global_edge_is_safe(current, q,
                minimum_tcp_height=request.continuation_minimum_tcp_height,
                minimum_joint_margin=request.minimum_joint_margin,
                minimum_downward_alignment=request.minimum_downward_alignment))
        if not valid:
            self._descent_note('timed_descent_live_rejected', now,
                reason='geometry' if owner_valid else 'certificate_owner', **evidence)
            self._discard_timed_descent()
            self._last_error = 'timed descent left current geometry or feedback envelope'
            self._record_motion_plan_failure()
            return current.copy()
        self._last_ik_solution = q.copy()
        self._cartesian_command_pose = pose.copy()
        self._arm_command_is_time_parameterized = bool(parameterized)
        self._record_motion_plan_success()
        return q.copy()

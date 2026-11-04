"""Opt-in, event-triggered execution of one complete immutable pick schedule.

Execution-time proposals use the original complete certificate and a future
C2 reference. Invalid parents still abort/hold. The runner's existing healthy-feedback failure handler owns the
fixed joint hold; it never releases a grasp because a schedule has expired.
"""
from dataclasses import replace
from collections.abc import Mapping
import json

import numpy as np

from ._math import quat_distance
from .dynamic_intercept import intercept_lead


def validate_full_pick_settings(config):
    if config.planning_mode not in {'legacy', 'full_preplanned'}:
        raise ValueError('planning_mode must be legacy or full_preplanned')
    if config.selection_policy not in {'default', 'first_safe'}:
        raise ValueError('selection_policy must be default or first_safe')
    if config.replan_policy not in {'legacy', 'event_triggered'}:
        raise ValueError('replan_policy must be legacy or event_triggered')
    if not np.isfinite(config.full_plan_observation_max_age_s) or config.full_plan_observation_max_age_s <= 0.:
        raise ValueError('full plan observation validity must be finite and positive')
    if not np.isfinite(config.full_plan_close_window_s) or config.full_plan_close_window_s <= 0.:
        raise ValueError('full plan closure window must be finite and positive')
    if not np.isfinite(config.full_plan_compute_budget_s) or config.full_plan_compute_budget_s <= 0.:
        raise ValueError('full plan compute budget must be finite and positive')
    if config.planning_mode == 'full_preplanned':
        if config.selection_policy != 'first_safe' or config.replan_policy != 'event_triggered':
            raise ValueError('full_preplanned requires first_safe and event_triggered')
        if getattr(config, 'action_mode', 'joint_position') != 'joint_position' or not getattr(config, 'enable_time_parameterization', True):
            raise ValueError('full_preplanned requires joint-position trajectories')
        if getattr(config, 'prepare_high_only', False):
            raise ValueError('full_preplanned cannot prepare high only')


from .full_pick_replanning import FullPickReplanningMixin


class FullPickExecutionMixin(FullPickReplanningMixin):
    def _reset_full_pick(self):
        self._full_pick_request = None
        self._full_pick_result = None
        self._full_pick_bundle = None
        self._full_pick_hold_q = None
        self._full_pick_abort_reason = None
        self._full_pick_events = []
        self._full_pick_phase_events = set()
        self._full_pick_confirmed_since = None
        self._full_pick_lift_admitted = False
        self._full_pick_metrics = {}
        self._full_pick_observation_trace = []
        self._full_pick_finished = False
        self._full_pick_contact_handed_off = False
        self._full_pick_handoff_lift_speed = None
        self._full_pick_handoff_angular_speed = None
        self._full_pick_pre_motion_replans = 0
        self._full_pick_last_planning_wall_s = None
        self._full_pick_catalog_quaternion = None
        self._reset_full_replanning()

    def _full_pick_enabled(self):
        return self.config.planning_mode == 'full_preplanned'

    def _full_pick_note(self, name, now, **details):
        self._full_pick_events.append({'event': name, 'timestamp': float(now), **details})
        self._emit(name, now, str(details.get('reason', '')))

    def _full_pick_abort(self, observation, reason):
        now = self._timestamp_without_advancing(observation)
        if self._full_pick_abort_reason is None:
            self._full_pick_abort_reason = str(reason)
            self._full_pick_hold_q = self._joint_positions(observation).copy()
            self._cancel_async_plan()
            self._full_pick_note('replan_triggered', now, replan_reason=str(reason),
                reason=str(reason), candidate_sent=False, action='abort_to_existing_fixed_position_hold',
                in_motion_replacement=False)
            self._full_pick_note('grasp_result', now, success=False,
                reason=str(reason), confirmation_source=self.config.grasp_confirmation_source)
            from .baseline_sorting_policy import PolicyStage
            self._stage = PolicyStage.RECOVERING
            self._last_reason = str(reason)
            self._last_error = str(reason)
        # The real runner inspects RECOVERING before dispatch and enters its
        # existing fixed hold. Standalone callers also receive ONE fixed target.
        return self._action(observation,
            target_pose=self.env.ik.forward_pose(self._full_pick_hold_q),
            gripper=float((self._last_action or {}).get('gripper', self.config.gripper_open_action)))

    def _full_pick_observation(self, observation, now):
        target = self._find_target(observation, self._target_id)
        if not isinstance(target, Mapping):
            return None, 'target_lost'
        prediction = self._prediction_from_mapping(observation, self._target_id)
        if not isinstance(prediction, Mapping):
            return None, 'target_prediction_unavailable'
        stamp = prediction.get('observed_timestamp', target.get('observed_timestamp'))
        if stamp is None or not np.isfinite(stamp):
            return None, 'capture_timestamp_unavailable'
        age = now-float(stamp)
        if age < -1e-6 or age > self.config.full_plan_observation_max_age_s:
            return None, 'target_observation_expired'
        # A detector miss is distinct from an invalid motion prediction. The
        # tracker retains the target-bound measured speed and original capture
        # time while coasting. Policy and final dispatch share this check;
        # age, confidence, frozen-plan tolerances and closure checks still apply.
        if prediction.get('valid') is False:
            return None, 'target_observation_invalid'
        confidence = float(prediction.get('prediction_confidence', target.get('confidence', 0.)))
        if not np.isfinite(confidence) or confidence < self.config.minimum_prediction_confidence:
            return None, 'target_velocity_confidence_invalid'
        contact = self._current_contact_pose(observation)
        velocity = self._belt_velocity(observation)
        if contact is None or not np.isfinite(np.r_[contact, velocity]).all():
            return None, 'target_motion_unavailable'
        # current_contact_pose is explicitly publication-time TCP. Recover the
        # exposure pose once, then keep exposure, publication and contact clocks
        # separate. No old speed or image is restamped by this calculation.
        measured = prediction.get('observed_pose')
        capture_pose = np.asarray(measured, dtype=float) if measured is not None else contact.copy()
        if measured is None:
            capture_pose[:3] -= velocity*age
        if capture_pose.shape != (7,) or not np.isfinite(capture_pose).all():
            return None, 'target_capture_pose_invalid'
        if self._full_pick_catalog_quaternion is not None:
            # This is the chosen grasp orientation, not a new measurement of
            # object rotation. Its original observation stays in the adapter.
            capture_pose=capture_pose.copy()
            capture_pose[3:]=self._full_pick_catalog_quaternion
        self._full_pick_metrics.update(observation_age_s=age,
            velocity_m_s=np.asarray(velocity).tolist(), velocity_confidence=confidence,
            capture_timestamp_s=float(stamp))
        return (target, capture_pose, float(stamp), velocity), None

    def _submit_full_pick(self, observation, now, visual):
        target, captured, stamp, velocity = visual
        q = self._joint_positions(observation)
        if not self._joint_velocity_is_settled(observation, self.config.planner_waypoint_velocity_tolerance):
            return self._full_pick_abort(observation, 'full_pick_start_not_stationary')
        if self._full_pick_hold_q is None:
            self._full_pick_hold_q = q.copy()
        dt = self._execution_dt(observation)
        compute_reserve = self.config.full_plan_compute_budget_s
        if (self.config.full_plan_contact_handoff_enabled and self._full_pick_pre_motion_replans
                and self._full_pick_last_planning_wall_s is not None):
            compute_reserve = min(compute_reserve,
                self._full_pick_last_planning_wall_s*self.config.prefetch_planning_wall_realtime_ratio
                + self.config.prefetch_intercept_clock_guard)
        start = now+compute_reserve+self.config.trajectory_minimum_duration+dt
        current = captured.copy(); current[:3] += velocity*(now-stamp)
        preparation=float(getattr(self.env,'contact_orientation_preparation_s',0.))
        direct = self.config.moving_approach_mode == 'continuous'
        if direct:
            # The endpoint moves during both computation and execution. Use
            # the existing first-reachable-time solver at the future low TCP,
            # rather than measuring a long trip to an upstream current pose.
            # This is only an ETA: the worker still solves IK/time laws and
            # certifies every changed endpoint and clock.
            horizon = max(7.5, start-now+self.config.descent_lead_time)
            lead = intercept_lead(current[:3], velocity, self._ee_pose(observation)[:3],
                grasp_height=0., approach_clearance=0.,
                cartesian_speed=self.config.cartesian_linear_speed,
                minimum_motion_time=self.config.descent_lead_time,
                latency=start-now, maximum_horizon=horizon)
            duration = max(self.config.descent_lead_time, lead-(start-now))+preparation
            station=getattr(self.env,'contact_station_x_m',None)
            if station is not None and abs(float(velocity[0]))>1e-6:
                station_lead=(float(station)-float(current[0]))/float(velocity[0])
                station_pose=current[:3]+np.asarray(velocity)*station_lead
                duration=max(self.config.descent_lead_time,
                    float(np.linalg.norm(station_pose-self._ee_pose(observation)[:3]))/self.config.cartesian_linear_speed)+preparation
                # Wait at the unchanged current joint target, then execute the
                # bounded motion. Extra waiting is not a stretched polynomial.
                start=max(start,now+station_lead-duration)
        else:
            high = current.copy(); high[2] += self.config.approach_height
            duration = max(self.config.descent_lead_time+self.config.trajectory_minimum_duration,
                (np.linalg.norm(high[:3]-self._ee_pose(observation)[:3])
                 + self.config.approach_height)/self.config.cartesian_linear_speed)
        end = start+duration
        # A predicted target may be upstream now but enter the permitted range
        # later. Select an appointment in that real window before solving the
        # complete path. Do not expand the workspace or certify an expired
        # target time. Missing windows preserve legacy callers' behaviour.
        window = target.get('grasp_window')
        earliest = latest = None
        if isinstance(window, Mapping):
            try:
                earliest = float(window.get('start_time', now))
                latest = float(window.get('end_time', float('inf'))) - self.config.intercept_deadline_margin
            except (TypeError, ValueError):
                return self._full_pick_abort(observation, 'full_pick_invalid_contact_window')
            if not np.isfinite(earliest) or np.isnan(latest) or earliest > latest:
                return self._full_pick_abort(observation, 'full_pick_invalid_contact_window')
            # The Cartesian ETA is only a heuristic, not proof that an
            # earlier joint path is impossible. When it overshoots, try the
            # latest allowed contact using the available full motion time.
            end=min(max(end, earliest), latest)
            duration=end-start
            if duration < self.config.descent_lead_time - 1e-9:
                return self._full_pick_abort(observation, 'full_pick_insufficient_intercept_window')
        capture = captured.copy(); capture[:3] += velocity*(end-stamp)
        if (self.config.full_plan_contact_handoff_enabled and direct and capture[1]>.06
                and target.get('grasp_quaternion') is None and target.get('grasp_orientation') is None
                and self._full_pick_catalog_quaternion is None):
            choices=self._async_target_poses(capture,target=target)
            preferred=next((p for p in choices
                if quat_distance(p[3:],capture[3:])>1e-6),None)
            if preferred is not None:
                capture[3:]=preferred[3:]
                self._full_pick_catalog_quaternion=capture[3:].copy()
                self._selected_grasp_quaternion=capture[3:].copy()
                self._full_pick_note('existing_far_rail_grasp_orientation_selected',now,
                    position_unchanged=True,grasp_quaternion=capture[3:].tolist(),
                    source='existing_async_target_poses',requires_full_certificate=True)
        self._grasp_pose = capture.copy()
        request = self._make_planning_request(capture, observation,
            minimum_tcp_height=min(float(self._ee_pose(observation)[2]), float(capture[2]))-.012)
        if request is None:
            return self._full_pick_abort(observation, 'full_pick_request_unavailable')
        settings = getattr(self.env, 'settings', None)
        response = getattr(self.env, 'joint_response_model', None)
        try:
            response_json = None if response is None else json.dumps(response.to_dict(), allow_nan=False)
        except (AttributeError, TypeError, ValueError):
            response_json = None
        closure_tolerances = {}
        controller=getattr(self.env,'robot_controller',None)
        command_lower=getattr(controller,'command_joint_lower_limits',None)
        command_upper=getattr(controller,'command_joint_upper_limits',None)
        for name in ('closure_horizontal_tolerance','closure_vertical_tolerance','closure_orientation_tolerance'):
            try:
                closure_tolerances[name] = max(0.,float(target.get(name,getattr(self.config,name))))
            except (TypeError,ValueError):
                closure_tolerances[name] = getattr(self.config,name)
        request = replace(request, planning_mode='full_preplanned',
            full_response_model_json=response_json,
            full_observed_timestamp_s=stamp,
            full_observation_clock_epoch=self._full_replan_clock(observation)[0],
            full_observation_track_generation=self._full_replan_clock(observation)[1],
            full_contact_handoff_enabled=self.config.full_plan_contact_handoff_enabled,
            full_orientation_preparation_s=preparation,
            full_orientation_preparation_tracking_entry=bool(getattr(self.env,'contact_orientation_tracking_entry',False)),
            full_contact_acceleration_m_s2=float(getattr(self.env,'contact_acceleration_m_s2',0.)),
            full_finger_entry_height_m=float(getattr(self.env,'contact_finger_entry_height_m',0.)),
            full_finger_entry_lead_s=float(getattr(self.env,'contact_finger_entry_lead_s',.8)),
            full_response_projection_fraction=float(getattr(self.env,'contact_response_projection_fraction',.25)),
            full_approach_response_enabled=bool(getattr(self.env,'contact_approach_response_enabled',True)),
            full_close_lead_s=self.config.close_lead_time,
            full_command_lower_limits=None if command_lower is None else tuple(command_lower),
            full_command_upper_limits=None if command_upper is None else tuple(command_upper),
            full_control_period_s=dt,
            full_dispatch_lead_s=max(0., float(observation.get('dispatch_clock', {}).get(
                'scheduled_dispatch_policy_time_s', now))-now),
            full_feedback_translation_limit_m=self.config.max_translation_step,
            full_feedback_rotation_limit_rad=self.config.max_rotation_step,
            acquisition_joint_limit_rad=self.config.acquisition_path_tracking_error_limit,
            full_closure_horizontal_tolerance_m=closure_tolerances['closure_horizontal_tolerance'],
            full_closure_vertical_tolerance_m=closure_tolerances['closure_vertical_tolerance'],
            full_closure_orientation_tolerance_rad=closure_tolerances['closure_orientation_tolerance'],
            full_approach_mode=self.config.moving_approach_mode,
            selection_policy='first_safe', replan_policy='event_triggered',
            continuous_intercept=True, target_is_contact=True, target_poses=(capture,),
            continuation_target_poses=None, continuation_minimum_tcp_height=None,
            fixed_high_continuation=False, intercept_velocity_m_s=None,
            continuation_start_time_s=start, planned_target_time_s=end,
            timed_continuation_duration_s=duration, continuation_velocity_m_s=tuple(velocity),
            continuation_start_velocity=(0.,)*7, continuation_start_acceleration=(0.,)*7,
            intercept_motion_start_time_s=start, intercept_request_time_s=now,
            intercept_account_for_planning_time=isinstance(observation.get('dispatch_clock'), Mapping),
            intercept_descent_duration_s=self.config.descent_lead_time,
            # Keep the selected closure width; native no-adaptive jaw command
            # is zero. The nominal held width is measured object thickness.
            full_grasp_width_m=float(getattr(self.env, 'full_plan_grasp_width_m', .04)),
            full_high_clearance_m=self.config.approach_height,
            full_close_window_s=self.config.full_plan_close_window_s,
            full_plan_compute_budget_s=compute_reserve,
            full_contact_earliest_s=earliest,
            full_contact_latest_s=latest if latest is not None and np.isfinite(latest) else None,
            full_lift_m=float(getattr(settings, 'lift_m', .03)),
            continuation_tcp_speed_limits=((self.config.cartesian_linear_speed,
                                            self.config.cartesian_angular_speed)
                                           if self.config.tcp_speed_limits_enabled else None),
            route_waypoint_hints=())
        worker = self._ensure_planning_worker()
        if worker is None or not worker.submit(request):
            return self._full_pick_abort(observation, 'full_pick_worker_unavailable')
        self._full_pick_request = self._planning_request = request
        self._full_pick_note('full_plan_start', now, request_id=request.request_id,
            observed_capture_s=stamp, requested_start_s=start, requested_contact_s=end,
            compute_budget_s=compute_reserve,
            configured_compute_budget_s=self.config.full_plan_compute_budget_s,
            pre_motion_replan_count=self._full_pick_pre_motion_replans,
            approach_mode=self.config.moving_approach_mode, mandatory_high_waypoint=not direct,
            delivery_policy='one_plan_with_one_pre_motion_delivery_correction')
        return None

    def _poll_full_pick(self, observation, now):
        request = self._full_pick_request
        if request is None:
            return None
        result = self._planning_worker.poll()
        if result is None:
            return None
        effective = self._completed_planning_request(result, request)
        if effective is None or not self._request_has_current_owner(request):
            self._planning_stale_results += 1
            self._full_pick_note('stale_result_ignored', now,
                request_id=result.request_id, generation=result.generation,
                current_request_id=request.request_id,
                current_generation=request.generation)
            return None
        self._full_pick_request = self._planning_request = None
        self._full_pick_note('full_plan_end', now, request_id=request.request_id,
            wall_s=float(result.planning_time_s), success=result.success)
        self._full_pick_last_planning_wall_s = float(result.planning_time_s)
        bundle = result.full_pick_plan
        if (effective is None or not result.success or bundle is None
                or not result.path_fully_validated or not result.continuation_fully_validated
                or bundle.request is not effective
                or bundle.contact_time != effective.planned_target_time_s):
            return self._full_pick_abort(observation, result.error or 'full_pick_certificate_incomplete')
        send = float(observation.get('dispatch_clock', {}).get('scheduled_dispatch_policy_time_s', now))
        if max(now, send) >= bundle.start:
            return self._full_pick_abort(observation, 'full_pick_delivery_missed_start')
        window = self._find_target(observation, self._target_id).get('grasp_window', {})
        if bundle.contact_time > float(window.get('end_time', float('inf')))-self.config.intercept_deadline_margin:
            return self._full_pick_abort(observation, 'full_pick_contact_outside_reachable_window')
        reserve = getattr(self.env, 'refine_selected_grasp_plan', None)
        if callable(reserve):
            accepted = reserve(self._target_id, max(0., bundle.contact_time-now),
                self.config.intercept_deadline_margin, requested_time=bundle.contact_time,
                requested_pose=bundle.contact_pose)
            self._full_pick_metrics['reservation_result'] = accepted
            pose = accepted.get('predicted_ee_grasp_pose') if isinstance(accepted, Mapping) else None
            if (not isinstance(accepted, Mapping) or not accepted.get('accepted', False)
                    or (pose is not None and (np.linalg.norm(np.asarray(pose)[:3]-bundle.contact_pose[:3])
                        > self.config.trajectory_replan_position_tolerance
                        or quat_distance(np.asarray(pose)[3:], bundle.contact_pose[3:])
                        > self.config.trajectory_replan_orientation_tolerance))):
                return self._full_pick_abort(observation, 'full_pick_reservation_changed')
        self._full_pick_result, self._full_pick_bundle = result, bundle
        self._pregrasp_capture_time_s = bundle.contact_time
        self._full_pick_metrics.update(first_full_planning_wall_s=float(result.planning_time_s),
            request_to_result_policy_s=now-request.intercept_request_time_s,
            **{key: (result.diagnostics or {}).get(key) for key in
                ('selection_policy', 'ik_seeds_tried', 'first_safe_route_wall_s',
                 'extra_comparison_wall_s', 'route_source')},
            **(result.diagnostics or {}).get('full_preplanned', {}))
        self._full_pick_note('full_plan_certified', now, start_s=bundle.start,
            high_s=bundle.high_time, contact_s=bundle.contact_time,
            lift_start_s=None if bundle.lift is None else bundle.lift_time, end_s=bundle.end,
            certificate_scope='unloaded_direct_approach_only' if bundle.closing is None else 'complete_fixed_pick',
            nominal_lift_requires_measured_grasp=True)
        return None

    def _refresh_unexecuted_contact_prefix(self, observation, now, bundle, error, angle):
        """Retire an unused certificate; a new immutable request owns every new curve."""
        send=max(now,float(observation.get('dispatch_clock',{}).get('scheduled_dispatch_policy_time_s',now)))
        if (not self.config.full_plan_contact_handoff_enabled or bundle.closing is not None
                or send>=bundle.start or 'trajectory_execution_start' in self._full_pick_phase_events
                or not self._joint_velocity_is_settled(observation,
                    self.config.planner_waypoint_velocity_tolerance)):
            return False
        self._cancel_async_plan()
        self._full_pick_request = self._full_pick_result = self._full_pick_bundle = None
        self._full_pick_pre_motion_replans += 1
        self._full_pick_note('unexecuted_contact_prefix_retired',now,
            previous_request_id=bundle.request.request_id,previous_start_s=bundle.start,
            previous_contact_s=bundle.contact_time,position_error_m=error,angle_error_rad=float(angle),
            replan_count=self._full_pick_pre_motion_replans,arm_motion_started=False,
            fixed_hold_q_rad=self._full_pick_hold_q.tolist(),new_certificate_required=True)
        # The fixed hold and controller command history stay unchanged. The
        # next control cycle submits from fresh feedback/visual timestamps.
        return True

    def _handoff_full_prefix(self, observation, now, *, meaning):
        bundle=self._full_pick_bundle
        self._full_pick_contact_handed_off=True
        try:
            from .joint_response import JointResponseModel
            from .full_pick_response import eligible_response
            model=JointResponseModel.from_dict(json.loads(bundle.request.full_response_model_json))
            if eligible_response(model):
                horizon=max(a.delay_s+a.time_constant_s for a in model.axes)
                horizon+=self._execution_dt(observation)+bundle.request.full_dispatch_lead_s
                self._full_pick_handoff_lift_speed=min(self.config.cartesian_linear_speed,
                    self.config.max_translation_step/horizon)
                self._full_pick_handoff_angular_speed=min(self.config.cartesian_angular_speed,
                    self.config.maximum_tcp_angular_speed,self.config.max_rotation_step/horizon)
        except (ValueError,TypeError,KeyError,AttributeError):pass
        self._invalidate_motion_plan()
        self._full_pick_note('contact_handoff_to_shared_feedback_control',now,meaning=meaning,
            loaded_tcp_speed_candidate_m_s=self._full_pick_handoff_lift_speed,
            loaded_angular_speed_candidate_rad_s=self._full_pick_handoff_angular_speed,
            old_bundle_retired=True,full_schedule_completed=False)

    def _full_pick_act(self, observation):
        from .baseline_sorting_policy import PolicyStage
        if (not self._full_pick_enabled() or self._full_pick_finished or self._full_pick_contact_handed_off
                or self._stage not in {PolicyStage.PREGRASP, PolicyStage.TRACKING_DESCENT,
                                      PolicyStage.CLOSING, PolicyStage.LIFTING}):
            return None
        now = self._timestamp_without_advancing(observation)
        timeout = self._timeout_reason(now, observation)
        if timeout is not None or self._severe_collision(observation):
            return self._full_pick_abort(observation, timeout or 'severe_collision')
        visual, reason = self._full_pick_observation(observation, now)
        if reason is not None and not self._full_pick_lift_admitted:
            return self._full_pick_abort(observation, reason)
        if self._full_pick_bundle is None and self._full_pick_request is None:
            failed = self._submit_full_pick(observation, now, visual)
            if failed is not None:
                return failed
        if self._full_pick_bundle is None and self._full_pick_request is not None:
            # Once even an immediately delivered result cannot fit the
            # existing minimum approach and deadline, cancel the worker.
            # This is a necessary time bound, never a path approval.
            window=visual[0].get('grasp_window', {})
            if not isinstance(window, Mapping):
                return self._full_pick_abort(observation,'full_pick_invalid_contact_window')
            try:
                deadline=float(window.get('end_time',float('inf')))-self.config.intercept_deadline_margin
            except (TypeError, ValueError):
                return self._full_pick_abort(observation,'full_pick_invalid_contact_window')
            earliest_contact=(now+self.config.trajectory_minimum_duration+self._execution_dt(observation)
                              +self.config.descent_lead_time)
            if np.isnan(deadline):
                return self._full_pick_abort(observation,'full_pick_invalid_contact_window')
            if earliest_contact > deadline+1e-9:
                return self._full_pick_abort(observation,'full_pick_window_expired_during_planning')
        failed = self._poll_full_pick(observation, now)
        if failed is not None:
            return failed
        bundle = self._full_pick_bundle
        if bundle is None:
            return self._action(observation,
                target_pose=self.env.ik.forward_pose(self._full_pick_hold_q),
                gripper=self.config.gripper_open_action)
        monitor = getattr(self.env, 'full_plan_scene_monitor', None)
        if monitor is not None:
            status = monitor.snapshot()
            self._full_pick_metrics['scene_monitor'] = status
            if status.get('failed'):
                return self._full_pick_abort(observation, status.get('failure_reason', 'new_scene_obstacle'))
            stamp = status.get('validated_capture_monotonic_s')
            age = float('inf') if stamp is None else now+float(self.env._policy_clock_origin)-float(stamp)
            if now >= bundle.execution_start and (age > self.config.full_plan_observation_max_age_s
                    or not self._full_scene_has_plan(status,bundle)):
                return self._full_pick_abort(observation, 'scene_review_expired_or_missing_for_plan')
        if (bundle.request.scene_epoch != int(observation.get('scene_epoch', self._observed_scene_epoch or 0))
                or bundle.request.target_id != self._target_id
                or not self._request_identity_is_current(bundle.request)
                or not self._full_replan_clock_valid(bundle.request,observation)):
            return self._full_pick_abort(observation, 'full_pick_scene_or_identity_changed')
        if not self._full_pick_lift_admitted:
            target, captured, stamp, velocity = visual
            predicted = captured.copy(); predicted[:3] += velocity*(bundle.contact_time-stamp)
            error = float(np.linalg.norm(predicted[:3]-bundle.contact_pose[:3]))
            closing_prediction = captured[:3]+velocity*(bundle.lift_time-stamp)
            closing_error = (None if bundle.closing is None else
                float(np.linalg.norm(closing_prediction-bundle.closing_pose[:3])))
            angle = quat_distance(predicted[3:], bundle.contact_pose[3:])
            self._full_pick_metrics.update(predicted_contact_error_m=error,
                predicted_close_window_error_m=closing_error,
                predicted_contact_angle_rad=float(angle), remaining_contact_s=bundle.contact_time-now,
                remaining_execution_s=max(0., bundle.end-now))
            self._full_pick_metrics['maximum_predicted_contact_error_m'] = max(error,
                self._full_pick_metrics.get('maximum_predicted_contact_error_m', 0.))
            self._full_pick_observation_trace.append({'policy_time_s': now,
                'capture_time_s': stamp, 'capture_pose': captured.tolist(),
                'prediction_velocity_m_s': np.asarray(velocity).tolist(),
                'predicted_contact_pose': predicted.tolist(), 'certified_contact_pose': bundle.contact_pose.tolist(),
                'contact_position_error_m': error, 'close_window_error_m': closing_error,
                'remaining_contact_s': bundle.contact_time-now})
            if (error > self.config.trajectory_replan_position_tolerance
                    or angle > self.config.trajectory_replan_orientation_tolerance):
                if self._refresh_unexecuted_contact_prefix(observation,now,bundle,error,angle):
                    return self._action(observation,
                        target_pose=self.env.ik.forward_pose(self._full_pick_hold_q),
                        gripper=self.config.gripper_open_action)
                return self._full_pick_abort(observation, 'predicted_contact_left_certificate_tolerance')
            # Frozen execution must remain valid throughout the close window.
            # For constant-velocity predictions, checking both endpoints bounds
            # their position difference inside the interval by the same 4 mm.
            if closing_error is not None and closing_error > self.config.trajectory_replan_position_tolerance:
                return self._full_pick_abort(observation, 'predicted_close_window_left_certificate_tolerance')
            if not self._window_valid(target, observation):
                return self._full_pick_abort(observation, 'target_outside_reachable_window')
        if not self._full_pick_lift_admitted:
            self._maintain_full_replanning(observation,now,visual)
            if self._full_pick_abort_reason is not None:
                return self._full_pick_abort(observation,self._full_pick_abort_reason)
            bundle=self._full_pick_bundle
        planned_send=max(now,float(observation.get('dispatch_clock',{}).get('scheduled_dispatch_policy_time_s',now)))
        if (bundle.closing is None and self.config.full_plan_contact_handoff_enabled
                and planned_send>=bundle.contact_time):
            if self._stage==PolicyStage.PREGRASP:
                self._stage=PolicyStage.TRACKING_DESCENT;self._stage_started_at=now
            self._handoff_full_prefix(observation,now,
                meaning='scheduled_contact_phase_only_not_grasp_or_force_confirmation')
            return None
        if now >= bundle.start and 'trajectory_execution_start' not in self._full_pick_phase_events:
            self._full_pick_phase_events.add('trajectory_execution_start')
            self._full_pick_note('trajectory_execution_start', now, scheduled_start_s=bundle.start)
        approach_start = bundle.start if bundle.descent is None else bundle.high_time
        approach_event = 'continuous_approach_started' if bundle.descent is None else 'high_waypoint_passed'
        if now >= approach_start and approach_event not in self._full_pick_phase_events:
            self._full_pick_phase_events.add(approach_event)
            self._full_pick_note(approach_event, now, scheduled_s=approach_start, new_planning_request=False)
            self._stage = PolicyStage.TRACKING_DESCENT
            self._stage_started_at = now
            if bundle.descent is not None:
                self._full_pick_note('descent_started', now, scheduled_s=approach_start)
        if now >= bundle.contact_time-self.config.close_lead_time and self._stage == PolicyStage.TRACKING_DESCENT:
            current = captured.copy(); current[:3] += velocity*(now-stamp)
            self._grasp_pose = current.copy()
            if bool(observation.get('_defer_arm_motion', False)):
                self._deferred_closure_context = {'capture_pose': current.copy(),
                    'tracking_reference_before_update': current.copy()}
            else:
                self._admit_descent_closure(target, observation, now, current, current, current)
        if self._stage == PolicyStage.CLOSING:
            if bundle.closing is None and self.config.full_plan_contact_handoff_enabled:
                # The original measured closure predicate has already admitted
                # CLOSING. Retire the unloaded approach before the jaws touch;
                # the ordinary controller owns feedback tracking and preload.
                # This admits neither grasp confirmation nor loaded lifting.
                self._handoff_full_prefix(observation,now,
                    meaning='original_measured_closure_gate_admitted_shared_closing')
                return None
            feedback = self._grasp_feedback(target, observation)
            source_ok = (self.config.grasp_confirmation_source == 'width_and_force'
                         or self._bilateral_finger_contact(observation))
            if (self.config.full_plan_contact_handoff_enabled and feedback is True and source_ok
                    and now >= bundle.contact_time-1e-9):
                # The free-workpiece prediction and frozen loaded branch end
                # here. Ordinary CLOSING owns contact capture, preload and all
                # stable dwells; its subsequent loaded motion is checked from
                # actual feedback. This is an explicit adaptive alternative,
                # not successful completion of the immutable full schedule.
                self._handoff_full_prefix(observation,now,
                    meaning='contact_only_force_confirmation_and_new_loaded_checks_pending')
                return None
            if (feedback is True and source_ok and now >= bundle.contact_time-1e-9
                    and self._grasp_confirmed_at is None):
                # Match ordinary CLOSING: valid centred contact starts the
                # existing preload helper. Force readiness cannot be required
                # to initialize the contact timestamp that preload itself uses.
                # The separate full-pick stable dwell still owns lift admission.
                self._grasp_confirmed_at = now
                self._full_pick_note('grasp_contact_feedback_received', now,
                    confirmation_source=self.config.grasp_confirmation_source,
                    meaning='contact_feedback_only_not_force_ready_or_lift_success')
            force_ok = self._update_grasp_force_preload(observation, now, feedback)
            if (feedback is True and source_ok and force_ok
                    and self._grasp_confirmed_at is not None):
                if self._full_pick_confirmed_since is None:
                    self._full_pick_confirmed_since = now
            else:
                self._full_pick_confirmed_since = None
        planned_send = max(now, float(observation.get('dispatch_clock', {}).get(
            'scheduled_dispatch_policy_time_s', now)))
        if planned_send >= bundle.lift_time and not self._full_pick_lift_admitted:
            dwell = max(0., float(target.get('confirmed_grasp_settle_time', self.config.confirmed_grasp_settle_time)))
            stable = (self._full_pick_confirmed_since is not None
                and now-self._full_pick_confirmed_since >= dwell)
            if not stable:
                return self._full_pick_abort(observation, 'grasp_not_confirmed_in_certified_close_window')
            q = self._joint_positions(observation)
            measured_width = observation.get('gripper', {}).get('opening_width')
            self._full_pick_metrics.update(measured_lift_start_width_m=measured_width,
                nominal_lift_width_m=bundle.request.full_grasp_width_m, loaded_width_tolerance_m=.003)
            if (measured_width is not None and abs(float(measured_width)-bundle.request.full_grasp_width_m) > .003):
                return self._full_pick_abort(observation, 'nominal_payload_width_changed')
            if (np.max(np.abs(q-bundle.lift.q0)) > self.config.planner_corner_settle_tolerance
                    or np.linalg.norm(self._ee_pose(observation)[:3]-bundle.closing_pose[:3])
                       > self.config.trajectory_replan_position_tolerance
                    or np.max(np.abs(np.asarray(observation.get('joint_vel', np.full(7, np.inf)))
                        - bundle.lift.sample_full(0.)[1])) > self.config.planner_waypoint_velocity_tolerance):
                return self._full_pick_abort(observation, 'nominal_loaded_lift_start_changed')
            self._secured_hold_pose = self._ee_pose(observation).copy()
            self._stage = PolicyStage.LIFTING
            self._stage_started_at = now
            self._full_pick_lift_admitted = True
            self._emit('lift_started', now, 'grasp_confirmed')
            self._full_pick_note('grasp_result', now, success=True,
                confirmation_source=self.config.grasp_confirmation_source,
                meaning='existing_grasp_feedback_confirmed_not_independent_physical_ground_truth')
        if self._full_pick_lift_admitted and self._grasp_feedback(self._target, observation) is not True:
            return self._full_pick_abort(observation, 'grasp_lost_during_loaded_lift')
        if now >= bundle.end:
            q = self._joint_positions(observation)
            if (np.max(np.abs(q-bundle.lift.q1)) > self.config.planner_corner_settle_tolerance
                    or not self._joint_velocity_is_settled(observation, self.config.planner_waypoint_velocity_tolerance)):
                return self._full_pick_abort(observation, 'loaded_lift_did_not_finish_on_schedule')
            self._lift_pose = bundle.lift_pose.copy()
            self._full_pick_finished = True
            # Subsequent sorting/placement uses the original loaded policy.
            return None
        pose = bundle.contact_pose.copy()
        if self._stage == PolicyStage.LIFTING:
            pose = bundle.lift_pose.copy()
        return self._action(observation, target_pose=pose,
            gripper=(self._grasp_gripper_action if self._stage in {PolicyStage.CLOSING, PolicyStage.LIFTING}
                     else self.config.gripper_open_action))

    def _arm_on_full_pick(self, observation):
        if not self._full_pick_enabled() or self._full_pick_finished or self._full_pick_contact_handed_off:
            return None
        if self._full_pick_abort_reason is not None:
            return self._full_pick_hold_q.copy()
        bundle = self._full_pick_bundle
        if bundle is None:
            return None if self._full_pick_hold_q is None else self._full_pick_hold_q.copy()
        now = self._timestamp_without_advancing(observation)
        send = max(now, float(observation.get('dispatch_clock', {}).get('scheduled_dispatch_policy_time_s', now)))
        self._activate_full_splice(observation,send)
        if self._full_pick_abort_reason is not None:return self._full_pick_hold_q.copy()
        if send < bundle.execution_start:
            return self._full_pick_hold_q.copy()
        if send >= bundle.lift_time and not self._full_pick_lift_admitted:
            self._full_pick_abort(observation, 'loaded_lift_not_admitted_before_dispatch')
            return self._full_pick_hold_q.copy()
        if send > bundle.end:
            self._full_pick_abort(observation, 'full_pick_dispatch_past_certificate')
            return self._full_pick_hold_q.copy()
        self._full_pick_sample_time_s = send
        q, v, a = bundle.sample_command_full(send)
        nominal = bundle.sample_full(send)[0]
        self._safe_edge_reference_diagnostics = dict(
            status='certified_full_pick_response' if bundle.command_curves else 'original_reference',
            applied=bundle.command_curves is not None, candidate_q_rad=q.tolist(),
            candidate_qd_rad_s=v.tolist(), candidate_qdd_rad_s2=a.tolist(),
            applied_reference_q_rad=q.tolist(), nominal_q_rad=nominal.tolist())
        current = self._joint_positions(observation)
        controller = getattr(self.env, 'robot_controller', None)
        preview = getattr(controller, 'preview_joint_step', None)
        if callable(preview):
            previous, velocity = self._joint_command_history(observation, current)
            limited, *_ = preview(q, previous, velocity, self._execution_dt(observation))
            if np.max(np.abs(np.asarray(limited)-q)) > 1e-7:
                self._full_pick_metrics.update(retimer_requested_q_rad=q.tolist(),
                    retimer_limited_q_rad=np.asarray(limited).tolist(),
                    retimer_previous_q_rad=previous.tolist(),retimer_previous_velocity_rad_s=velocity.tolist(),
                    retimer_preview_dt_s=self._execution_dt(observation))
                self._full_pick_abort(observation, 'controller_retimer_would_change_certified_clock')
                return self._full_pick_hold_q.copy()
        tracking = float(np.max(np.abs(current-q)))
        self._full_pick_metrics['joint_feedback_error_rad'] = tracking
        pose = np.asarray(self.env.ik.forward_pose(q))
        feedback_pose = self._ee_pose(observation)
        nominal_now = self.env.ik.forward_pose(bundle.sample_full(now)[0])
        self._full_pick_metrics.update(
            same_time_feedback_translation_error_m=float(np.linalg.norm(nominal_now[:3]-feedback_pose[:3])),
            prepared_command_feedback_distance_m=float(np.linalg.norm(pose[:3]-feedback_pose[:3])),
            reference_advance_s=send-now,
            response_compensation_applied=bundle.command_curves is not None)
        joint_envelope = (self.config.acquisition_path_tracking_error_limit
            if send < bundle.approach_end_time else self.config.max_ik_joint_step)
        if (tracking > joint_envelope
                or np.linalg.norm(pose[:3]-feedback_pose[:3]) > self.config.max_translation_step
                or quat_distance(pose[3:], feedback_pose[3:]) > self.config.max_rotation_step
                or not self._global_edge_is_safe(current, q,
                    minimum_tcp_height=bundle.minimum_tcp_height_at(send),
                    minimum_joint_margin=bundle.request.minimum_joint_margin,
                    minimum_downward_alignment=bundle.request.minimum_downward_alignment)):
            self._full_pick_abort(observation, 'full_pick_feedback_or_live_geometry_invalid')
            return self._full_pick_hold_q.copy()
        self._last_ik_solution = q.copy()
        self._cartesian_command_pose = pose.copy()
        self._arm_command_is_time_parameterized = True
        return q.copy()

    def full_pick_dispatch_reason(self, observation, joint_target, now):
        bundle = self._full_pick_bundle
        if (not self._full_pick_enabled() or self._full_pick_finished
                or self._full_pick_contact_handed_off or bundle is None):
            return None
        if self._full_pick_abort_reason is not None:
            return self._full_pick_abort_reason
        if (not self._request_identity_is_current(bundle.request)
                or bundle.request.scene_epoch != int(observation.get('scene_epoch',self._observed_scene_epoch or 0))
                or not self._full_replan_clock_valid(bundle.request,observation)):
            return 'full_pick_dispatch_owner_or_clock_changed'
        monitor = getattr(self.env, 'full_plan_scene_monitor', None)
        if monitor is not None:
            status = monitor.snapshot()
            if status.get('failed'):
                return status.get('failure_reason', 'new_scene_obstacle')
            stamp = status.get('validated_capture_monotonic_s')
            if now >= bundle.execution_start and (stamp is None
                    or now+float(self.env._policy_clock_origin)-float(stamp) > self.config.full_plan_observation_max_age_s
                    or not self._full_scene_has_plan(status,bundle)):
                return 'scene_review_expired_or_missing_for_plan'
        if now < bundle.execution_start:
            return None
        if now > bundle.end:
            return 'full_pick_dispatch_past_certificate'
        if now >= bundle.lift_time and not self._full_pick_lift_admitted:
            return 'loaded_lift_not_admitted_before_dispatch'
        expected = bundle.sample_command_full(now)[0]
        pose = np.asarray(self.env.ik.forward_pose(expected))
        sent_pose = np.asarray(self.env.ik.forward_pose(np.asarray(joint_target)))
        if (np.max(np.abs(expected-joint_target)) > self.config.planner_corner_settle_tolerance
                or np.linalg.norm(pose[:3]-sent_pose[:3]) > self.config.trajectory_replan_position_tolerance):
            return 'full_pick_dispatch_clock_drift'
        if not self._full_pick_lift_admitted:
            visual, reason = self._full_pick_observation(observation, now)
            if reason is not None:
                return reason
            _, captured, stamp, velocity = visual
            predicted = captured[:3]+velocity*(bundle.contact_time-stamp)
            if np.linalg.norm(predicted-bundle.contact_pose[:3]) > self.config.trajectory_replan_position_tolerance:
                return 'predicted_contact_left_certificate_tolerance'
            closing_prediction = captured[:3]+velocity*(bundle.lift_time-stamp)
            if (bundle.closing is not None and
                    np.linalg.norm(closing_prediction-bundle.closing_pose[:3]) > self.config.trajectory_replan_position_tolerance):
                return 'predicted_close_window_left_certificate_tolerance'
        return None

    def _guard_contact_handoff_command(self, observation, joint_target):
        """Keep the original full-pick feedback envelope after adaptive handoff."""
        if (not self._full_pick_contact_handed_off or self._full_pick_abort_reason is not None
                or self.config.action_mode!='joint_position'):
            return joint_target
        q=np.asarray(joint_target);actual=self._joint_positions(observation)
        pose=self.env.ik.forward_pose(q);measured=self._ee_pose(observation)
        distance=float(np.linalg.norm(pose[:3]-measured[:3]))
        angle=float(quat_distance(pose[3:],measured[3:]));joint=float(np.max(np.abs(q-actual)))
        self._full_pick_metrics.update(handoff_proposed_feedback_distance_m=distance,
            handoff_proposed_feedback_angle_rad=angle,handoff_proposed_feedback_joint_rad=joint)
        if (distance>self.config.max_translation_step or angle>self.config.max_rotation_step
                or joint>self.config.max_ik_joint_step):
            self._full_pick_abort(observation,'contact_handoff_feedback_envelope_invalid')
            return self._full_pick_hold_q.copy()
        return joint_target

    def _full_pick_info(self):
        replanning = (self.config.full_plan_replanning_enabled
            and self.config.replan_policy == 'event_triggered')
        complete = self._full_pick_finished and not self._full_pick_contact_handed_off
        return {'enabled': self._full_pick_enabled(), 'planning_mode': self.config.planning_mode,
            'selection_policy': self.config.selection_policy, 'replan_policy': self.config.replan_policy,
            'replanning': dict(self._full_pick_replanning_metrics),
            'pending_update_id':None if self._full_pick_update_request is None else self._full_pick_update_request.request_id,
            'event_response': ('certified_future_splice_while_parent_valid_otherwise_original_hold'
                if replanning else 'safe_abort_no_unverified_motion_replacement'),
            'abort_reason': self._full_pick_abort_reason, 'metrics': dict(self._full_pick_metrics),
            'contact_handed_off':self._full_pick_contact_handed_off,
            'contact_handoff_enabled':self.config.full_plan_contact_handoff_enabled,
            'pre_motion_replan_count':self._full_pick_pre_motion_replans,
            'lift_scope_completed':self._full_pick_finished,
            'complete_pick_schedule_completed':complete,
            'immutable_full_schedule_completed':complete and not self._full_pick_replanning_metrics['accepted_count'],
            'execution_strategy':('certified_contact_approach_then_shared_feedback'
                if self.config.full_plan_contact_handoff_enabled else
                'versioned_complete_pick_with_future_splice' if replanning else 'immutable_complete_pick'),
            'execution_replan_count': self._full_pick_replanning_metrics['reference_activation_count'],
            'events': list(self._full_pick_events),
            'prediction_diagnostics': list(self._full_pick_observation_trace)}

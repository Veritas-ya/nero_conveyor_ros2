"""Event-driven full certificates at a future state of the active reference.

This mixin schedules the existing worker. It never sends, interpolates, relaxes
an old certificate, or authorizes closure. Failed/late proposals leave a valid
parent alone; the caller still enforces all original parent validity checks.
"""
from dataclasses import replace
from collections.abc import Mapping
import numpy as np
from ._math import quat_distance
from .dynamic_intercept import intercept_lead
from .trajectory_splicing import splice_errors


def version(bundle):
    return (bundle.request.generation, bundle.request.request_id)


class FullPickReplanningMixin:
    def _reset_full_replanning(self):
        self._full_pick_update_request = None
        self._full_pick_update_parent = None
        self._full_pick_pending_bundle = None
        self._full_pick_pending_result = None
        self._full_pick_last_update_attempt = None
        self._full_pick_jaw_committed = None
        self._full_pick_splice_prepared = None
        self._full_pick_splice_crossed = None
        self._full_pick_splice_dispatched = None
        self._full_pick_update_delivery_s = 0.
        self._full_pick_replanning_metrics = dict(trigger_count=0, compute_times_s=[],
            certificate_pass_count=0, accepted_count=0, stale_result_reject_count=0,
            missed_splice_count=0, completed_result_count=0, reference_activation_count=0,
            sdk_dispatch_count=0, splice_errors=[],
            planned_contact_shifts_s=[], ready_latencies_s=[])

    def _full_replan_latest_time(self, observation, now):
        latest=max(now,float(observation.get('dispatch_clock',{}).get('scheduled_dispatch_policy_time_s',now)))
        trace=getattr(self.env,'execution_trace',None)
        # Only an explicitly shared real-time clock may advance this sample.
        # Accelerated/offline policy seconds are not host elapsed time.
        if trace is not None and getattr(trace,'mode',None)=='hardware' and trace.policy_origin is not None:
            latest=max(latest,float(trace.clock()-trace.policy_origin))
        return latest

    @staticmethod
    def _full_scene_has_plan(status, bundle):
        identities=status.get('checked_plan_identities')
        return id(bundle) in (identities if identities is not None else [status.get('checked_plan_identity')])

    def _full_replan_clock(self, observation):
        prediction = self._prediction_from_mapping(observation, self._target_id) or {}
        velocity = prediction.get('velocity_measurement', {})
        return prediction.get('clock_epoch'), velocity.get('track_generation')

    def _full_replan_clock_valid(self, request, observation):
        epoch, generation = self._full_replan_clock(observation)
        return ((request.full_observation_clock_epoch is None or epoch == request.full_observation_clock_epoch)
            and (request.full_observation_track_generation is None
                 or generation == request.full_observation_track_generation))

    def _full_replan_window_open(self, bundle, observation, now):
        send = self._full_replan_latest_time(observation,now)
        return bool(self.config.full_plan_replanning_enabled
            and self.config.replan_policy == 'event_triggered'
            and self.config.dynamic_intercept_tracking
            and self._stage.value in {'pregrasp', 'tracking_descent'}
            and self._full_pick_jaw_committed is None and not self._full_pick_lift_admitted
            and bundle.closing is not None and bundle.lift is not None
            and bundle.command_curves is None
            and send < min(bundle.approach_end_time, bundle.contact_time-self.config.close_lead_time))

    def _cancel_full_update(self, now, reason):
        if self._full_pick_update_request is not None:
            self._planning_worker.cancel_pending()
            if self._planning_request is self._full_pick_update_request:
                self._planning_request = None
            self._full_pick_note('full_replan_update_cancelled', now, reason=reason)
        self._full_pick_update_request = self._full_pick_update_parent = None
        self._full_pick_pending_bundle = self._full_pick_pending_result = None

    def _full_update_owner(self, request, parent, observation):
        return bool(parent is self._full_pick_bundle
            and request.full_parent_version == version(parent)
            and request.generation == self._planning_generation
            and request.scene_epoch == int(observation.get('scene_epoch', self._observed_scene_epoch or 0))
            and self._same_target_id(request.target_id, self._target_id)
            and self._request_identity_is_current(request)
            and self._full_replan_clock_valid(request, observation))

    def _poll_full_update(self, observation, now):
        request = self._full_pick_update_request
        if request is None:return
        result = self._planning_worker.poll()
        if result is None:return
        parent = self._full_pick_update_parent
        effective = self._completed_planning_request(result, request)
        metrics = self._full_pick_replanning_metrics
        if effective is None or not self._full_update_owner(request, parent, observation):
            self._planning_stale_results += 1
            metrics['stale_result_reject_count'] += 1
            self._full_pick_note('stale_result_ignored', now, request_id=result.request_id,
                current_request_id=request.request_id, scope='full_replan')
            return
        self._full_pick_update_request = None
        if self._planning_request is request:self._planning_request = None
        metrics['completed_result_count'] += 1
        metrics['compute_times_s'].append(float(result.planning_time_s))
        self._full_pick_update_delivery_s = max(self._full_pick_update_delivery_s,
            float(result.planning_time_s), now-request.intercept_request_time_s)
        bundle = result.full_pick_plan
        valid = (result.success and result.path_fully_validated and result.continuation_fully_validated
            and bundle is not None and bundle.request is effective
            and bundle.closing is not None and bundle.lift is not None and bundle.command_curves is None
            and bundle.prefix is None
            and effective.full_parent_version == version(parent)
            and bundle.start == request.continuation_start_time_s
            and bundle.contact_time == effective.planned_target_time_s
            and bundle.contact_time > bundle.start
            and self._full_replan_window_open(parent, observation, now))
        send = self._full_replan_latest_time(observation,now)
        if not valid:
            self._full_pick_note('full_replan_rejected', now, request_id=request.request_id,
                reason=result.error or 'remaining_full_certificate_or_stage_invalid')
            return
        metrics['certificate_pass_count'] += 1
        if send >= bundle.start:
            metrics['missed_splice_count'] += 1
            self._full_pick_note('full_replan_late', now, request_id=request.request_id,
                scheduled_splice_s=bundle.start, latest_dispatch_s=send)
            return
        errors = splice_errors(parent,bundle,bundle.start)
        command_errors = splice_errors(parent,bundle,bundle.start,sampler='sample_command_full')
        if max(*errors,*command_errors)>1e-8:
            self._full_pick_note('full_replan_rejected',now,request_id=request.request_id,
                reason='splice_q_v_a_changed',errors=list(errors),command_errors=list(command_errors))
            return
        # Keep the EXACT certified prefix and its samples. It is never retimed.
        combined = replace(bundle,prefix=parent,
            checked_samples=tuple(s for s in parent.checked_samples if s[1]<bundle.start)+bundle.checked_samples,
            checked_command_samples=tuple(s for s in parent.checked_command_samples if s[1]<bundle.start)
                +bundle.checked_command_samples)
        self._full_pick_pending_bundle, self._full_pick_pending_result = combined, result
        metrics['splice_errors'].append(list(errors))
        self._full_pick_note('full_replan_certified',now,request_id=request.request_id,
            parent_version=list(version(parent)),scheduled_splice_s=bundle.start,
            planned_contact_s=bundle.contact_time,reference_q_v_a_error=list(errors),
            certificate_scope='approach_descent_closing_conditional_lift',not_yet_installed=True)

    def _install_full_update(self, observation, now, visual):
        candidate = self._full_pick_pending_bundle
        if candidate is None:return
        parent = self._full_pick_update_parent
        request = candidate.request
        send = self._full_replan_latest_time(observation,now)
        reject = None
        if send >= candidate.start:
            self._full_pick_replanning_metrics['missed_splice_count'] += 1
            reject = 'missed_splice_while_waiting_for_scene'
        elif not self._full_update_owner(request,parent,observation):reject='parent_or_owner_changed'
        elif not self._full_replan_window_open(parent,observation,now):reject='closure_or_phase_committed'
        monitor=getattr(self.env,'full_plan_scene_monitor',None)
        if reject is None and monitor is not None:
            status=monitor.snapshot()
            if status.get('pending_plan_identity')==id(candidate) and status.get('pending_failure_reason'):
                reject=status['pending_failure_reason']
            elif not self._full_scene_has_plan(status,candidate):return
            elif (status.get('failed') or status.get('validated_capture_monotonic_s') is None
                    or now+float(self.env._policy_clock_origin)-status['validated_capture_monotonic_s']
                        >self.config.full_plan_observation_max_age_s):reject='scene_certificate_changed'
        target,captured,stamp,velocity=visual
        predicted=captured[:3]+velocity*(candidate.contact_time-stamp)
        close=captured[:3]+velocity*(candidate.lift_time-stamp)
        if (np.linalg.norm(predicted-candidate.contact_pose[:3])>self.config.trajectory_replan_position_tolerance
                or np.linalg.norm(close-candidate.closing_pose[:3])>self.config.trajectory_replan_position_tolerance
                or quat_distance(captured[3:],candidate.contact_pose[3:])>self.config.trajectory_replan_orientation_tolerance
                or candidate.contact_time>float(target.get('grasp_window',{}).get('end_time',float('inf')))
                    -self.config.intercept_deadline_margin):reject='new_contact_or_close_prediction_changed'
        reservation_changed=False
        reserve=getattr(self.env,'refine_selected_grasp_plan',None)
        if reject is None:
            reserve=getattr(self.env,'refine_selected_grasp_plan',None)
            if callable(reserve):
                value=reserve(self._target_id,max(0.,candidate.contact_time-now),self.config.intercept_deadline_margin,
                    requested_time=candidate.contact_time,requested_pose=candidate.contact_pose)
                reservation_changed=isinstance(value,Mapping) and value.get('accepted',False)
                pose=value.get('predicted_ee_grasp_pose') if isinstance(value,Mapping) else None
                if (not isinstance(value,Mapping) or not value.get('accepted',False)
                        or (pose is not None and (np.linalg.norm(np.asarray(pose)[:3]-candidate.contact_pose[:3])
                            >self.config.trajectory_replan_position_tolerance
                            or quat_distance(np.asarray(pose)[3:],candidate.contact_pose[3:])
                                >self.config.trajectory_replan_orientation_tolerance))):reject='reservation_changed'
        ready=self._full_replan_latest_time(observation,now)
        if ready>=candidate.start:
            self._full_pick_replanning_metrics['missed_splice_count']+=int(reject is None)
            reject=reject or 'missed_splice_during_final_reservation'
        if reject is not None and reservation_changed:
            restored=reserve(self._target_id,max(0.,parent.contact_time-now),self.config.intercept_deadline_margin,
                requested_time=parent.contact_time,requested_pose=parent.contact_pose)
            if not isinstance(restored,Mapping) or not restored.get('accepted',False):
                self._full_pick_abort(observation,'full_replan_old_reservation_no_longer_valid')
        if reject is not None:
            self._full_pick_note('full_replan_rejected',now,request_id=request.request_id,reason=reject)
        else:
            self._full_pick_bundle=candidate
            self._full_pick_result=self._full_pick_pending_result
            self._pregrasp_capture_time_s=candidate.contact_time
            self._full_pick_splice_prepared=(version(candidate),ready)
            metrics=self._full_pick_replanning_metrics
            metrics['accepted_count']+=1
            metrics['planned_contact_shifts_s'].append(candidate.contact_time-parent.contact_time)
            metrics['ready_latencies_s'].append(ready-request.full_observed_timestamp_s)
            self._full_pick_note('full_replan_splice_prepared',ready,request_id=request.request_id,
                scheduled_splice_s=candidate.start,planned_contact_s=candidate.contact_time,
                parent_version=list(version(parent)),trajectory_version=list(version(candidate)),
                common_prefix_preserved=True)
        self._full_pick_pending_bundle=self._full_pick_pending_result=None

    def _maintain_full_replanning(self, observation, now, visual):
        parent=self._full_pick_bundle
        if not self._full_replan_window_open(parent,observation,now):
            self._cancel_full_update(now,'outside_reversible_approach')
            return
        self._poll_full_update(observation,now)
        self._install_full_update(observation,now,visual)
        if self._full_pick_abort_reason is not None or self._full_pick_bundle is not parent:return
        if self._full_pick_update_request is not None or self._full_pick_pending_bundle is not None:return
        target,captured,stamp,velocity=visual
        horizon=parent.contact_time-parent.start
        delta_velocity=float(np.linalg.norm(velocity-np.asarray(parent.request.continuation_velocity_m_s)))*horizon
        # Position/orientation errors > original thresholds are still handled
        # by the caller's original abort, NOT by permission to keep that route.
        if delta_velocity<=self.config.trajectory_replan_position_tolerance:return
        endpoint=captured[:3]+velocity*(parent.contact_time-stamp)
        previous=self._full_pick_last_update_attempt
        if previous is not None:
            old_endpoint,old_velocity=previous
            if max(np.linalg.norm(endpoint-old_endpoint),np.linalg.norm(velocity-old_velocity)*horizon)\
                    <=self.config.trajectory_replan_position_tolerance:return
        self._full_pick_last_update_attempt=(endpoint.copy(),velocity.copy())
        self._submit_full_update(observation,now,visual,parent)

    def _submit_full_update(self, observation, now, visual, parent):
        target,captured,stamp,velocity=visual
        dt=self._execution_dt(observation)
        compute=max(self._full_pick_update_delivery_s,
            float(self._full_pick_last_planning_wall_s or 0.)*self.config.prefetch_planning_wall_realtime_ratio
                +self.config.prefetch_intercept_clock_guard,dt)
        send=self._full_replan_latest_time(observation,now)
        scene_forecast=0.
        monitor=getattr(self.env,'full_plan_scene_monitor',None)
        if monitor is not None:
            status=monitor.snapshot()
            end=status.get('last_compute_end_monotonic_s')
            capture=status.get('validated_capture_monotonic_s')
            if end is not None and capture is not None:scene_forecast=max(0.,end-capture)
            scene_forecast+=float(getattr(monitor,'period',0.))
        splice=max(send,now+compute)+scene_forecast+dt
        if splice>=min(parent.approach_end_time,parent.contact_time-self.config.close_lead_time):
            self._full_pick_note('full_replan_skipped',now,reason='insufficient_future_splice_window')
            return
        q,v,a=parent.sample_full(splice)
        current=captured.copy();current[:3]+=velocity*(now-stamp)
        lead=intercept_lead(current[:3],velocity,self.env.ik.forward_pose(q)[:3],
            grasp_height=0.,approach_clearance=0.,cartesian_speed=self.config.cartesian_linear_speed,
            minimum_motion_time=self.config.descent_lead_time,latency=splice-now,
            maximum_horizon=max(7.5,parent.contact_time-now))
        end=max(parent.contact_time,now+lead,splice+self.config.descent_lead_time)
        capture=captured.copy();capture[:3]+=velocity*(end-stamp)
        window=target.get('grasp_window',{})
        earliest=window.get('start_time')
        latest=float(window.get('end_time',float('inf')))-self.config.intercept_deadline_margin
        if end>latest or (earliest is not None and end<float(earliest)):return
        snapshot=self._future_open_planning_snapshot(q)
        base=self._make_planning_request(capture,observation,
            minimum_tcp_height=min(float(self.env.ik.forward_pose(q)[2]),float(capture[2]))-.012)
        if snapshot is None or base is None:return
        arm,full,epoch=snapshot
        clock,generation=self._full_replan_clock(observation)
        request=replace(parent.request,request_id=base.request_id,generation=self._planning_generation,
            target_identity=base.target_identity,stage='pregrasp',scene_epoch=epoch,start_q=arm,full_qpos=full,
            target_poses=(capture,),full_parent_version=version(parent),full_observed_timestamp_s=stamp,
            full_contact_earliest_s=None if earliest is None else float(earliest),
            full_contact_latest_s=latest if np.isfinite(latest) else None,
            full_observation_clock_epoch=clock,full_observation_track_generation=generation,
            continuation_start_time_s=splice,planned_target_time_s=end,timed_continuation_duration_s=end-splice,
            continuation_start_velocity=tuple(v),continuation_start_acceleration=tuple(a),
            continuation_velocity_m_s=tuple(velocity),intercept_request_time_s=now,
            intercept_motion_start_time_s=splice,intercept_account_for_planning_time=False,
            continuation_compute_forecast_s=compute,goal_seeds=base.goal_seeds,route_waypoint_hints=())
        if self._planning_worker.submit(request):
            self._full_pick_update_request=self._planning_request=request
            self._full_pick_update_parent=parent
            self._full_pick_replanning_metrics['trigger_count']+=1
            self._full_pick_note('full_replan_requested',now,request_id=request.request_id,
                parent_version=list(version(parent)),observed_timestamp_s=stamp,scheduled_splice_s=splice,
                planned_contact_s=end,observed_pose=captured.tolist(),velocity_m_s=velocity.tolist(),
                predicted_contact_pose=capture.tolist(),compute_forecast_s=compute,
                scene_review_forecast_s=scene_forecast,scheduler_latest_time_s=send)

    def _activate_full_splice(self, observation, send):
        bundle=self._full_pick_bundle
        if bundle.prefix is None or send<bundle.start or self._full_pick_splice_crossed==version(bundle):return
        if (self._full_pick_splice_prepared is None or self._full_pick_splice_prepared[0]!=version(bundle)
                or self._full_pick_splice_prepared[1]>=bundle.start
                or self._full_pick_jaw_committed is not None
                or not self._request_identity_is_current(bundle.request)
                or bundle.request.generation!=self._planning_generation
                or not self._full_replan_clock_valid(bundle.request,observation)):
            self._full_pick_abort(observation,'full_replan_activation_owner_or_commit_changed')
            return
        self._full_pick_splice_crossed=version(bundle)
        self._full_pick_replanning_metrics['reference_activation_count']+=1
        self._full_pick_note('full_replan_splice_reference_active',send,
            trajectory_version=list(version(bundle)),scheduled_splice_s=bundle.start,
            meaning='reference_sample_only_SDK_dispatch_recorded_separately')

    def full_pick_command_sent(self, policy_time):
        bundle=self._full_pick_bundle
        if bundle is not None and bundle.prefix is not None and policy_time>=bundle.start\
                and self._full_pick_splice_dispatched!=version(bundle):
            self._full_pick_splice_dispatched=version(bundle)
            self._full_pick_replanning_metrics['sdk_dispatch_count']+=1
            self._full_pick_note('full_replan_first_SDK_dispatch',policy_time,
                trajectory_version=list(version(bundle)),scheduled_splice_s=bundle.start)

    def full_pick_closure_sent(self, policy_time):
        if self._full_pick_enabled() and self._full_pick_jaw_committed is None:
            self._full_pick_jaw_committed=policy_time
            self._cancel_full_update(policy_time,'jaw_close_SDK_command_sent')

"""Optional complete pick certificates. No device access or command sending.

Every changed clock/pose is a new immutable request. Polynomial candidates
reuse the native state, edge, TCP and exact joint-derivative checks. A nominal
loaded lift is conditional on the execution-time grasp/scene/width checks.
"""
from dataclasses import dataclass, replace
import time
import json

import numpy as np

from .continuous_intercept_trajectory import ContinuousInterceptTrajectory, InterceptTimingError
from .continuous_joint_trajectory import BoundaryQuinticJointTrajectory
from .dynamic_intercept import pregrasp_high_pose
from .trajectory_splicing import prefix_segment
from .planning_progress import planning_stage


@dataclass(frozen=True)
class FullPickPlan:
    request: object
    approach: object
    descent: object
    closing: object
    lift: object
    contact_pose: np.ndarray
    closing_pose: np.ndarray
    lift_pose: np.ndarray
    checked_samples: tuple
    approach_minimum_tcp_height: float
    command_curves: tuple | None = None
    checked_command_samples: tuple = ()
    prefix: object | None = None

    def __post_init__(self):
        for name in ('contact_pose', 'closing_pose', 'lift_pose'):
            if getattr(self,name) is None:
                continue
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ('checked_samples', 'checked_command_samples'):
            samples = []
            for phase, stamp, q in getattr(self,name):
                value = np.array(q, dtype=float, copy=True); value.setflags(write=False)
                samples.append((phase, float(stamp), value))
            object.__setattr__(self, name, tuple(samples))

    def segment_at(self, timestamp):
        return prefix_segment(self, timestamp)

    @property
    def execution_start(self):
        return self.start if self.prefix is None else self.prefix.execution_start

    @property
    def start(self):
        return float(self.request.continuation_start_time_s)

    @property
    def high_time(self):
        return None if self.descent is None else self.approach_end_time

    @property
    def approach_end_time(self):
        return self.contact_time if self.descent is None else self.start+self.approach.duration

    @property
    def contact_time(self):
        return float(self.request.planned_target_time_s)

    @property
    def lift_time(self):
        return self.contact_time+(0. if self.closing is None else self.closing.duration)

    @property
    def end(self):
        return self.lift_time+(0. if self.lift is None else self.lift.duration)

    def phase(self, timestamp):
        if self.prefix is not None and timestamp < self.start:
            return self.prefix.phase(timestamp)
        if self.closing is None:
            return 'approach'
        if timestamp < self.approach_end_time:
            return 'approach'
        if timestamp < self.contact_time:
            return 'descent'
        if timestamp < self.lift_time:
            return 'closing'
        return 'lift'

    def sample_full(self, timestamp):
        if self.prefix is not None and timestamp < self.start:
            return self.prefix.sample_full(timestamp)
        phase = self.phase(timestamp)
        curve, start = {'approach': (self.approach, self.start),
            'descent': (self.descent, self.approach_end_time),
            'closing': (self.closing, self.contact_time),
            'lift': (self.lift, self.lift_time)}[phase]
        return curve.sample_full(float(np.clip(timestamp-start, 0., curve.duration)))

    def minimum_tcp_height_at(self, timestamp):
        if self.prefix is not None and timestamp < self.start:
            return self.prefix.minimum_tcp_height_at(timestamp)
        return (self.approach_minimum_tcp_height if timestamp < self.approach_end_time
                else self.request.minimum_tcp_height)

    def sample_command_full(self, timestamp):
        if self.prefix is not None and timestamp < self.start:
            return self.prefix.sample_command_full(timestamp)
        if self.command_curves is None:
            return self.sample_full(timestamp)
        phase = self.phase(timestamp)
        index = {'approach': 0, 'descent': 1, 'closing': 2, 'lift': 3}[phase]
        curve = self.command_curves[index]
        clock = (self.start, self.approach_end_time, self.contact_time, self.lift_time)[index]
        return curve.sample_full(float(np.clip(timestamp-clock, 0., curve.duration)))


class FullPickPlanningMixin:
    @planning_stage('time_parameterization')
    def _full_pick_curves(self, request, path, low_q, close_q, lift_q, *, nullspace_speed=0.):
        """Use the existing continuous curve, with an optional high descent."""
        duration = request.planned_target_time_s-request.continuation_start_time_s
        low_v = self._full_terminal_velocity(request, low_q, nullspace_speed=nullspace_speed)
        close_v = self._full_terminal_velocity(request, close_q, nullspace_speed=nullspace_speed)
        kwargs = dict(jerk_limits=request.trajectory_jerk_limits, allow_path_change=True)
        if request.full_approach_mode == 'continuous':
            descent = None
            approach = ContinuousInterceptTrajectory(path,
                request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                duration=duration, initial_velocity=request.continuation_start_velocity,
                initial_acceleration=request.continuation_start_acceleration,
                final_velocity=low_v, jerk_limits=request.trajectory_jerk_limits)
        else:
            descent = BoundaryQuinticJointTrajectory(path[-1], low_q,
                request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                final_velocity=low_v, minimum_duration=request.intercept_descent_duration_s, **kwargs)
            remaining = duration-descent.duration
            minimum_s = sum(BoundaryQuinticJointTrajectory(a, b,
                request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                jerk_limits=request.trajectory_jerk_limits).duration for a, b in zip(path, path[1:]))
            if remaining < minimum_s:
                raise InterceptTimingError('full approach and descent miss contact time',
                                           minimum_s+descent.duration)
            approach = ContinuousInterceptTrajectory(path,
                request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                duration=remaining, jerk_limits=request.trajectory_jerk_limits,
                initial_velocity=request.continuation_start_velocity,
                initial_acceleration=request.continuation_start_acceleration)
        closing = BoundaryQuinticJointTrajectory(low_q, close_q,
            request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
            initial_velocity=low_v, final_velocity=close_v,
            minimum_duration=request.full_close_window_s, **kwargs)
        if abs(closing.duration-request.full_close_window_s) > 1e-8:
            raise ValueError('fixed closure window violates joint dynamics')
        lift = BoundaryQuinticJointTrajectory(close_q, lift_q,
            request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
            initial_velocity=close_v, **kwargs)
        curves = [curve for curve in (approach, descent, closing, lift) if curve is not None]
        for a, b in zip(curves, curves[1:]):
            if not all(np.allclose(x, y, atol=1e-7, rtol=0.) for x, y in
                       zip(a.sample_full(a.duration), b.sample_full(0.))):
                raise ValueError('full pick q/v/a seam is discontinuous')
        return approach, descent, closing, lift

    def _full_pick_validation_work(self, request, curves):
        counts = {}
        for phase, curve in zip(('approach', 'descent', 'closing', 'lift'), curves):
            if curve is None:
                continue
            intervals = self._curve_validation_intervals(request, curve)
            counts[phase] = len(np.unique(np.r_[np.linspace(0., curve.duration, intervals+1),
                np.cumsum(getattr(curve, 'edge_durations', (curve.duration,)))]))
        # Closing certifies the existing eleven jaw apertures at every sample.
        work = sum(n*(11 if phase == 'closing' else 1) for phase, n in counts.items())
        return work, counts

    @planning_stage('delivery_rescheduling')
    def _full_pick_delivery_request(self, request, result, complete_s, started, cancelled):
        """Forecast ONE future static-start request; never certify by retiming."""
        bundle = result.full_pick_plan
        if bundle.closing is None:
            from .contact_approach_planning import delivery_request
            return delivery_request(self,request,result,complete_s,started,cancelled)
        curves = (bundle.approach, bundle.descent, bundle.closing, bundle.lift)
        old_work, old_counts = self._full_pick_validation_work(request, curves)
        computation = complete_s
        duration = request.timed_continuation_duration_s
        high_seed = bundle.approach.sample_full(bundle.approach.duration)[0]
        contact_curve = bundle.approach if bundle.descent is None else bundle.descent
        low_seed = contact_curve.sample_full(contact_curve.duration)[0]
        close_seed = bundle.closing.sample_full(bundle.closing.duration)[0]
        lift_seed = bundle.lift.sample_full(bundle.lift.duration)[0]
        start_z = float(self.ik.forward_pose(request.start_q)[2])
        forecasts = []
        report = self._request_diagnostics['full_preplanned']

        def proposed():
            start = max(request.continuation_start_time_s,
                (request.intercept_request_time_s or 0.)+time.perf_counter()-started
                +computation+request.trajectory_minimum_duration)
            capture = start+duration
            shift = np.asarray(request.continuation_velocity_m_s)*(capture-request.planned_target_time_s)
            poses = tuple(np.r_[pose[:3]+shift, pose[3:]] for pose in request.target_poses)
            floor_shift = min(start_z, poses[0][2])-min(start_z, request.target_pose[2])
            return replace(request, target_poses=poses,
                minimum_tcp_height=request.minimum_tcp_height+floor_shift,
                continuation_start_time_s=start, intercept_motion_start_time_s=start,
                planned_target_time_s=capture, timed_continuation_duration_s=duration,
                continuation_compute_forecast_s=computation,
                goal_seeds=(high_seed, *request.goal_seeds),
                route_waypoint_hints=tuple(result.path[1:-1]))

        for index in range(request.max_goal_candidates):
            current = proposed()
            if cancelled():
                break
            contact = current.target_pose
            direct = current.full_approach_mode == 'continuous'
            high = (contact if direct else pregrasp_high_pose(contact, current.continuation_velocity_m_s,
                approach_height=current.full_high_clearance_m,
                descent_lead_time=current.intercept_descent_duration_s))
            close = contact.copy()
            close[:3] += np.asarray(current.continuation_velocity_m_s)*current.full_close_window_s
            lift = close.copy(); lift[2] += current.full_lift_m
            self._diagnostic_context = {'phase': 'full_pick_delivery_forecast', 'iteration': index}
            poses = (contact, close, lift) if direct else (high, contact, close, lift)
            seeds = (low_seed, close_seed, lift_seed) if direct else (high_seed, low_seed, close_seed, lift_seed)
            qs = [self._solve_goal_seed(current, pose, seed, cancelled)
                  for pose, seed in zip(poses, seeds)]
            forecast = {'start_s': current.continuation_start_time_s,
                'capture_s': current.planned_target_time_s, 'duration_s': duration,
                'compute_forecast_s': computation, 'ik_success': all(q is not None for q in qs)}
            forecasts.append(forecast)
            if not forecast['ik_success'] or cancelled():
                break  # The complete planner still owns geometric feasibility.
            if direct:
                low_seed, close_seed, lift_seed = qs
                high_seed = low_seed
            else:
                high_seed, low_seed, close_seed, lift_seed = qs
            try:
                estimate = self._full_pick_curves(current, (*result.path[:-1], high_seed),
                                                  low_seed, close_seed, lift_seed)
            except InterceptTimingError as error:
                duration = max(duration, error.suggested_duration_s)
                forecast['next_duration_s'] = duration
                continue
            except ValueError as error:
                forecast['curve_forecast_error'] = str(error)
                break
            work, counts = self._full_pick_validation_work(current, estimate)
            next_compute = complete_s*max(1., work/max(1, old_work))
            forecast.update(validation_samples=counts, weighted_validation_samples=work,
                            next_compute_forecast_s=next_compute)
            previous = computation
            computation = next_compute
            if computation <= previous:
                break
        # Include the probe's elapsed time in the appointment. This final
        # proposal is unvalidated; ALL its endpoints/curves are checked again.
        revised = proposed()
        report.update(delivery_forecasts=forecasts, original_validation_samples=old_counts,
            original_weighted_validation_samples=old_work,
            complete_candidate_wall_s=complete_s,
            revised_compute_forecast_s=computation)
        return revised

    def _full_failure(self, request, error, started):
        from .planning_worker import PlanningResult
        return PlanningResult(target_identity=request.target_identity,
            request_id=request.request_id, generation=request.generation,
            stage=request.stage, target_id=request.target_id, scene_epoch=request.scene_epoch,
            target_pose=request.target_pose, start_q=request.start_q, error=error,
            planning_time_s=time.perf_counter()-started)

    def _full_terminal_velocity(self, request, q, *, nullspace_speed=0.):
        jacobian = self.ik.compute_jacobian(q)
        twist = np.r_[request.continuation_velocity_m_s, np.zeros(3)]
        velocity = np.linalg.lstsq(jacobian, twist, rcond=None)[0]
        if nullspace_speed:
            direction = np.linalg.svd(jacobian, full_matrices=True)[2][-1]
            direction *= 1. if direction[0] >= 0. else -1.
            velocity += float(nullspace_speed)*direction
        if (not np.isfinite(velocity).all()
                or np.linalg.norm(jacobian@velocity-twist) > 1e-6+1e-4*np.linalg.norm(twist)):
            raise ValueError('full pick cannot match target contact velocity')
        return velocity

    @planning_stage('response_branch_search')
    def _full_response_redundancy(self, request, path, low_q, closing_pose, lift_pose, curves, cancelled):
        """Additional response-aware joint branches; every returned curve is
        still subject to the ordinary complete geometry/dynamics certificate.
        Advance the closing IK seed along the candidate joint velocity so its
        free joint motion continues through closure instead of reversing.
        """
        from .joint_response import JointResponseModel
        from .full_pick_response import eligible_response,command_curves,forecast_tracking,command_within_limits
        self._full_response_candidate_cache=None
        if request.full_approach_mode!='continuous' or request.full_response_model_json is None:
            return curves
        if min(request.full_closure_horizontal_tolerance_m,request.full_closure_vertical_tolerance_m,
                request.full_closure_orientation_tolerance_rad)<=0.:
            return curves
        report=self._request_diagnostics['full_preplanned']
        try:
            model=JointResponseModel.from_dict(json.loads(request.full_response_model_json))
            if not eligible_response(model):return curves
            original=forecast_tracking(request,curves,curves,model,self.ik)
            if original['constraint_ratio']<=1.:return curves
            trials=report.setdefault('response_redundancy_candidates',[])
            qualified=[]
            for speed in (1.,1.25,1.125):
                if cancelled():return curves
                seed=low_q+self._full_terminal_velocity(request,low_q,nullspace_speed=speed)*request.full_close_window_s
                self._diagnostic_context={'phase':'full_preplanned','check':'response_closing_ik'}
                close_q=self._solve_goal_seed(request,closing_pose,seed,cancelled)
                if close_q is None:continue
                lift_q=self._solve_goal_seed(request,lift_pose,close_q,cancelled)
                if lift_q is None:continue
                try:
                    candidate=self._full_pick_curves(request,path,low_q,close_q,lift_q,nullspace_speed=speed)
                    # Retain the original nominal 4 mm closing accuracy before
                    # spending the full geometry budget on an extra branch.
                    closing=candidate[2]
                    error=max(float(np.linalg.norm(self.ik.forward_pose(closing.sample_full(t)[0])[:3]
                        -(request.target_pose[:3]+np.asarray(request.continuation_velocity_m_s)*t)))
                        for t in np.linspace(0.,closing.duration,33))
                    proposed=command_curves(request,candidate,model)
                    derivative_ratio=command_within_limits(request,proposed)
                    forecast=forecast_tracking(request,candidate,proposed,model,self.ik)
                    usable=(error<=request.intercept_position_tolerance_m and derivative_ratio<=1.+1e-9
                        and forecast['constraint_ratio']<=1.
                        and forecast['predicted_same_time_peak_m']<original['predicted_same_time_peak_m'])
                    trials.append(dict(nullspace_speed_rad_s=speed,nominal_closing_error_m=error,
                        derivative_ratio=derivative_ratio,forecast=forecast,usable=usable))
                    if usable:
                        qualified.append((forecast['constraint_ratio'],forecast['predicted_same_time_peak_m'],
                            speed,candidate,proposed,forecast,derivative_ratio))
                        if len(qualified)>=2:break
                except ValueError as exc:
                    trials.append(dict(nullspace_speed_rad_s=speed,usable=False,reason=str(exc)))
            if qualified:
                selected=min(qualified,key=lambda value:value[:2])
                _,_,speed,candidate,proposed,forecast,derivative_ratio=selected
                report['response_nullspace_speed_rad_s']=speed
                report['response_redundancy_selection']='best_forecast_of_up_to_two_qualified_branches_not_global_optimum'
                # Same immutable request and exact curve objects only. Cache
                # this prediction, never a collision/geometry conclusion.
                self._full_response_candidate_cache=(request,candidate,proposed,forecast,derivative_ratio)
                return candidate
        except (ValueError,TypeError,KeyError,AttributeError):
            pass
        return curves

    @planning_stage('curve_certification', phase_arg=3)
    def _full_certify_curve(self, request, curve, start, phase, cancelled, *, command=False):
        """Certify the actual polynomial, with timed moving-cloud snapshots.

        Adjacent states and their edges receive the original discrete grid or
        conservative adaptive proof. This is not a claim of continuous solid
        reconstruction from a single visible surface.
        """
        if phase == 'approach' and not command:
            # Preserve the original high-approach floor separately from the
            # lower descent/contact floor. A complete bundle is not permission
            # to dip below the former while routing to the high waypoint.
            floor = min(float(self.ik.forward_pose(curve.sample_full(0.)[0])[2]),
                        float(self.ik.forward_pose(curve.sample_full(curve.duration)[0])[2]))-.012
            request = replace(request, minimum_tcp_height=floor)
        intervals = self._curve_validation_intervals(request, curve)
        times = np.unique(np.r_[np.linspace(0., curve.duration, intervals+1),
            np.cumsum(getattr(curve, 'edge_durations', (curve.duration,)))])
        checked = []
        peak_tcp_ratio = 1.
        peak_linear = peak_angular = 0.
        # A curve is immutable during this certification. Reuse its exact
        # q/v/a samples across speed, batch and fallback collision checks;
        # sampling density and every geometric check remain unchanged.
        samples = []
        # Reject a dynamically impossible clock BEFORE expensive geometry.
        # A later candidate still receives all geometry checks from scratch.
        for offset in times:
            if cancelled():
                return None
            q, velocity, acceleration = curve.sample_full(float(offset))
            samples.append((q, velocity, acceleration))
            twist = self.ik.compute_jacobian(q)@velocity
            if phase == 'closing' and not command:
                actual = self.ik.forward_pose(q)
                expected = request.target_pose[:3]+np.asarray(request.continuation_velocity_m_s)*(start+float(offset)-request.planned_target_time_s)
                if np.linalg.norm(actual[:3]-expected) > request.intercept_position_tolerance_m:
                    self._record_rejection('full_pick_closing_contact_error',
                        position_error_m=float(np.linalg.norm(actual[:3]-expected)),
                        tolerance_m=request.intercept_position_tolerance_m)
                    return None
            limits = request.continuation_tcp_speed_limits
            ratio = max(float(np.linalg.norm(twist[:3]))/limits[0] if limits is not None else 0.,
                float(np.linalg.norm(twist[3:]))/limits[1] if limits is not None else
                (float(np.linalg.norm(twist[3:]))/request.maximum_tcp_angular_speed
                 if request.tcp_speed_limits_enabled else 0.))
            peak_tcp_ratio = max(peak_tcp_ratio, ratio)
            peak_linear = max(peak_linear, float(np.linalg.norm(twist[:3])))
            peak_angular = max(peak_angular, float(np.linalg.norm(twist[3:])))
        self._request_diagnostics['full_preplanned'].setdefault('curve_checks', []).append({
            'phase': phase, 'command_reference': command,
            'start_s': float(start), 'end_s': float(start+curve.duration),
            'minimum_tcp_height_m': request.minimum_tcp_height,
            'samples': len(times), 'tcp_linear_peak_m_s': peak_linear,
            'tcp_angular_peak_rad_s': peak_angular, 'tcp_speed_ratio': peak_tcp_ratio,
            'joint_derivative_absolute_bounds': {str(order): np.maximum(np.abs(bounds[0]),np.abs(bounds[1])).tolist()
                for order in (1,2,3) for bounds in (curve.derivative_bounds(order),)}})
        if peak_tcp_ratio > 1.+1e-9:
            self._record_rejection('full_pick_tcp_speed', full_phase=phase, peak_ratio=peak_tcp_ratio)
            if phase in {'approach', 'descent'} and not command:
                report = self._request_diagnostics['full_preplanned']
                report['suggested_duration_s'] = max(report.get('suggested_duration_s', 0.),
                    request.timed_continuation_duration_s*peak_tcp_ratio*1.01)
            return None
        if (phase in {'approach', 'descent', 'closing', 'lift'}
                and not (request.full_contact_handoff_enabled and phase=='approach')
                and not getattr(self._state_validator,'prefer_timed_edges',False)):
            # Same grid, one immutable conservative object sweep. Batching the
            # existing checks avoids rebuilding FK/scene indices at each pose.
            # If that enlarged obstacle rejects the route, fall back to exact
            # per-edge clock snapshots below; never reject just for batching.
            states = tuple(sample[0] for sample in samples)
            aperture_grid = (np.linspace(0., .1, 11) if phase == 'closing'
                else (request.full_grasp_width_m,) if phase == 'lift' else (None,))
            all_passed = True
            for width in aperture_grid:
                full = request.full_qpos.copy()
                if width is not None:
                    for name, sign in [('gripper_joint1', 1.), ('gripper_joint2', -1.)]:
                        try:full[int(self.model.joint(name).qposadr[0])] = sign*width/2.
                        except KeyError:pass
                swept = replace(request, full_qpos=full, full_phase=phase,
                    stage={'approach': 'pregrasp', 'descent': 'tracking_descent',
                           'closing': 'closing', 'lift': 'lifting'}[phase],
                    full_sample_start_time_s=start, full_sample_time_s=start+curve.duration)
                self._diagnostic_context = {'phase':'full_preplanned','check':phase+'_sweep_batch',
                    'planned_target_time_s':request.planned_target_time_s}
                planner = self._planner(swept, seed=request.planner_seed, cancelled=cancelled)
                if (not self._states_valid(swept, states, cancelled)
                        or not all(planner.edge_valid(a, b) for a, b in zip(states, states[1:]))
                        or cancelled()):
                    all_passed = False
                    break
            if all_passed:
                self._request_diagnostics['full_preplanned'].setdefault('phase_sweep_batch_passes', []).append(phase)
                return [(phase, start+float(t), q.copy()) for t, q in zip(times, states)]
            self._request_diagnostics['full_preplanned'].setdefault('phase_sweep_time_fallbacks', []).append(phase)
        previous = samples[0][0]
        previous_t = start
        for offset, (q, velocity, _) in zip(times, samples):
            if cancelled():
                return None
            timestamp = start+float(offset)
            full = request.full_qpos.copy()
            if phase in {'closing', 'lift'}:
                # Certify the swept jaw aperture as well as the nominal load.
                fraction = (float(offset)/curve.duration if phase == 'closing' else 1.)
                for name, sign in [('gripper_joint1', 1.), ('gripper_joint2', -1.)]:
                    try:
                        address = int(self.model.joint(name).qposadr[0])
                    except KeyError:  # Test models can omit gripper coordinates.
                        continue
                    full[address] += fraction*(sign*request.full_grasp_width_m/2.-full[address])
            if phase == 'closing' and not command:
                actual = self.ik.forward_pose(q)
                expected = request.target_pose[:3]+np.asarray(request.continuation_velocity_m_s)*(timestamp-request.planned_target_time_s)
                if np.linalg.norm(actual[:3]-expected) > request.intercept_position_tolerance_m:
                    self._record_rejection('full_pick_closing_contact_error',
                        position_error_m=float(np.linalg.norm(actual[:3]-expected)),
                        tolerance_m=request.intercept_position_tolerance_m)
                    return None
            closing_approach=(request.full_contact_handoff_enabled and phase=='approach'
                and timestamp>=request.planned_target_time_s-request.full_close_lead_s)
            aperture_grid = (np.linspace(0., .1, 11) if phase == 'closing' or closing_approach else (None,))
            for width in aperture_grid:
                sample_full = full.copy()
                if width is not None:
                    for name, sign in [('gripper_joint1', 1.), ('gripper_joint2', -1.)]:
                        try:sample_full[int(self.model.joint(name).qposadr[0])] = sign*width/2.
                        except KeyError:pass
                current = replace(request, full_qpos=sample_full, full_phase=phase,
                    stage={'approach': 'pregrasp', 'descent': 'tracking_descent',
                           'closing': 'closing', 'lift': 'lifting'}[phase],
                    full_sample_start_time_s=previous_t, full_sample_time_s=timestamp)
                self._diagnostic_context = {'phase': 'full_preplanned', 'check': phase,
                    'sample_absolute_time_s': timestamp, 'planned_target_time_s': request.planned_target_time_s}
                planner = self._planner(current, seed=request.planner_seed, cancelled=cancelled)
                if not self._state_valid(current, q, cancelled) or not planner.edge_valid(previous, q):
                    self._record_rejection('full_pick_curve_geometry', full_phase=phase, sample_time_s=timestamp)
                    return None
            checked.append((phase, timestamp, q.copy()))
            previous, previous_t = q, timestamp
        return checked

    @planning_stage('command_response_compensation')
    def _full_response_proposal(self, request, curves, report):
        from .joint_response import JointResponseModel
        from .full_pick_response import (eligible_response, command_curves,
            forecast_tracking, command_within_limits)
        if request.full_approach_mode != 'continuous' or request.full_response_model_json is None:
            report.setdefault('response', {'status': 'original_reference_no_applicable_model'})
            return None, 0.
        if min(request.full_closure_horizontal_tolerance_m,request.full_closure_vertical_tolerance_m,
               request.full_closure_orientation_tolerance_rad) == 0.:
            report['response'] = {'status':'original_reference_exact_closure_required','applied':False}
            return None, 0.
        began = time.perf_counter()
        diagnostic = {'status': 'original_reference', 'applied': False}
        report['response'] = diagnostic
        try:
            model = JointResponseModel.from_dict(json.loads(request.full_response_model_json))
            if not eligible_response(model):
                diagnostic['status'] = 'original_reference_model_unvalidated'
                return None, 0.
            diagnostic['model_metadata'] = model.metadata
            base = forecast_tracking(request, curves, curves, model, self.ik)
            cached=getattr(self,'_full_response_candidate_cache',None)
            if cached is not None and cached[0] is request and cached[1] is curves:
                _,_,proposal,predicted,derivative_ratio=cached
                diagnostic['reused_same_candidate_forecast']=True
            else:
                proposal = command_curves(request, curves, model)
                derivative_ratio = command_within_limits(request, proposal)
                predicted = forecast_tracking(request, curves, proposal, model, self.ik)
            diagnostic.update(nominal_forecast=base, proposed_forecast=predicted,
                command_derivative_ratio=derivative_ratio)
            # A forecast is an optimization hint, never a new execution gate.
            # Limits below are inherited, not larger response-specific margins.
            usable = (derivative_ratio <= 1.+1e-9 and predicted['constraint_ratio'] <= 1.
                and predicted['predicted_same_time_peak_m'] < base['predicted_same_time_peak_m'])
            if usable:
                diagnostic['status'] = 'candidate_requires_full_geometry_certificate'
                return proposal, 0.
            # Prefer the lower-error command, rather than giving up as soon
            # as the uncompensated forecast narrowly fits the old envelope.
            # Its own bounds determine the next appointment proposal.
            improving = predicted['predicted_same_time_peak_m'] < base['predicted_same_time_peak_m']
            ratio = (max(predicted['constraint_ratio'], derivative_ratio)
                     if improving else base['constraint_ratio'])
            if ratio > 1. and getattr(self, '_response_duration_proposals_allowed', False):
                # A changed appointment will recalculate the moving endpoint,
                # IK and all geometry. This is not a stretched old certificate.
                suggested = request.timed_continuation_duration_s * min(2.,max(1.05,ratio*1.01))
                diagnostic.update(status='propose_longer_new_appointment', suggested_duration_s=suggested)
                return None, suggested
            diagnostic['status'] = 'original_reference_no_qualified_improvement'
            return None, 0.
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            diagnostic.update(status='original_reference_model_unavailable', reason=str(exc))
            return None, 0.
        finally:
            diagnostic['forecast_wall_s'] = time.perf_counter()-began

    @planning_stage('full_pick_candidate')
    def _plan_full_pick_once(self, request, cancelled, started):
        if request.full_contact_handoff_enabled and request.full_approach_mode=='continuous':
            from .contact_approach_planning import plan_once
            return plan_once(self,request,cancelled,started)
        from .planning_worker import PlanningResult
        contact = request.target_pose
        duration = request.planned_target_time_s-request.continuation_start_time_s
        direct = request.full_approach_mode == 'continuous'
        high = (contact if direct else pregrasp_high_pose(contact, request.continuation_velocity_m_s,
            approach_height=request.full_high_clearance_m,
            descent_lead_time=request.intercept_descent_duration_s))
        error = 'no completely certified full pick'
        report = self._request_diagnostics.setdefault('full_preplanned', {
            'attempts': [], 'certified': False, 'descent_validation_wall_s': 0.,
            'time_recertification_wall_s': 0., 'nominal_lift_conditional': True})
        report['certified'] = False
        # Each seed must pass the WHOLE pick, not merely its high endpoint.
        candidates = []
        for seed in request.goal_seeds:
            if len(candidates) >= request.max_goal_candidates:
                break
            if cancelled():
                break
            self._diagnostic_context = {'phase': 'full_preplanned',
                'check': 'contact_ik' if direct else 'high_ik'}
            q_high = self._solve_goal_seed(request, high, seed, cancelled)
            if q_high is None:
                continue
            if any(np.max(np.abs(q_high-old)) < 1e-4 for old in candidates):
                continue
            candidates.append(q_high.copy())
            route_request = replace(request, planning_mode='legacy', continuous_intercept=False,
                target_is_contact=direct, target_poses=(high,), goal_seeds=(q_high,),
                timed_continuation_duration_s=None, continuation_start_time_s=None,
                intercept_velocity_m_s=None, intercept_account_for_planning_time=False,
                continuation_target_poses=None if direct else (contact,),
                minimum_tcp_height=min(float(self.ik.forward_pose(request.start_q)[2]), float(high[2]))-.012,
                continuation_minimum_tcp_height=None if direct else request.minimum_tcp_height)
            route = self._plan(route_request, cancelled)
            if not route.success or (not direct and not route.continuation_fully_validated):
                error = route.error or error
                continue
            low_q = route.path[-1] if direct else route.continuation_path[-1]
            self._diagnostic_context = {'phase': 'full_preplanned', 'check': 'closing_ik'}
            closing_pose = contact.copy()
            closing_pose[:3] += np.asarray(request.continuation_velocity_m_s)*request.full_close_window_s
            close_q = self._solve_goal_seed(request, closing_pose, low_q, cancelled)
            if close_q is None:
                continue
            lift_pose = closing_pose.copy()
            lift_pose[2] += request.full_lift_m
            lift_q = self._solve_goal_seed(request, lift_pose, close_q, cancelled)
            if lift_q is None:
                continue
            try:
                approach, descent, closing, lift = self._full_pick_curves(
                    request, route.path, low_q, close_q, lift_q)
            except InterceptTimingError as exc:
                report['suggested_duration_s'] = max(report.get('suggested_duration_s', 0.),
                    exc.suggested_duration_s)
                error = str(exc)
                self._record_rejection('full_pick_contact_deadline', message=error)
                continue
            except ValueError as exc:
                error = str(exc)
                self._record_rejection('full_pick_dynamics', message=error)
                continue
            curves = (approach, descent, closing, lift)
            curves = self._full_response_redundancy(request,route.path,low_q,closing_pose,lift_pose,curves,cancelled)
            approach,descent,closing,lift=curves
            commands, suggested = self._full_response_proposal(request, curves, report)
            if suggested > duration:
                report['suggested_duration_s'] = max(report.get('suggested_duration_s',0.),suggested)
                return self._full_failure(request, 'response_forecast_new_appointment_proposed', started)
            checked = []
            passed = True
            for phase, curve, clock in [('approach', approach, request.continuation_start_time_s),
                ('descent', descent, None if descent is None else request.planned_target_time_s-descent.duration),
                ('closing', closing, request.planned_target_time_s),
                ('lift', lift, request.planned_target_time_s+closing.duration)]:
                if curve is None:
                    continue
                begin = time.perf_counter()
                samples = self._full_certify_curve(request, curve, clock, phase, cancelled)
                elapsed = time.perf_counter()-begin
                if phase == 'descent':
                    report['descent_validation_wall_s'] += elapsed
                report['attempts'].append({'phase': phase, 'wall_s': elapsed,
                    'samples': 0 if samples is None else len(samples), 'safe': samples is not None})
                if samples is None:
                    passed = False
                    error = 'full pick '+phase+' failed complete geometry/TCP checks'
                    if report.get('suggested_duration_s', 0.) > duration:
                        return self._full_failure(request, error, started)
                    break
                checked.extend(samples)
            if not passed:
                continue
            command_checked = []
            if commands is not None:
                # Original nominal certification above is retained. The
                # command path is additional geometry, not a nominal contact
                # position. Actual contact still owns original jaw admission.
                for phase, curve, clock in [('approach', commands[0], request.continuation_start_time_s),
                        ('closing', commands[2], request.planned_target_time_s)]:
                    command_request = (replace(request, minimum_tcp_height=min(
                        float(self.ik.forward_pose(approach.sample_full(0.)[0])[2]),
                        float(self.ik.forward_pose(approach.sample_full(approach.duration)[0])[2]))-.012)
                        if phase == 'approach' else request)
                    command_started=time.perf_counter()
                    samples = self._full_certify_curve(command_request, curve, clock, phase, cancelled, command=True)
                    report.setdefault('command_certification',[]).append({'phase':phase,
                        'wall_s':time.perf_counter()-command_started,
                        'samples':0 if samples is None else len(samples),'safe':samples is not None})
                    if samples is None:
                        commands = None
                        report['response'].update(status='original_reference_command_geometry_rejected')
                        command_checked = []
                        break
                    command_checked.extend(samples)
                if commands is not None:
                    report['response'].update(status='fully_certified_response_command', applied=True)
            bundle = FullPickPlan(request, approach, descent, closing, lift,
                contact.copy(), closing_pose, lift_pose, tuple(checked),
                min(float(self.ik.forward_pose(approach.sample_full(0.)[0])[2]),
                    float(self.ik.forward_pose(approach.sample_full(approach.duration)[0])[2]))-.012,
                commands, tuple(command_checked))
            report.update(certified=True, trajectory_backend='fixed_deadline_C2_quintics',
                approach_mode=request.full_approach_mode, mandatory_high_waypoint=not direct,
                close_window_s=closing.duration, jaw_sweep_spacing_m=.01,
                start_s=bundle.start, high_s=bundle.high_time,
                capture_s=bundle.contact_time, lift_start_s=bundle.lift_time, end_s=bundle.end,
                q_v_a_continuous=True, first_candidate_returned=True)
            self._request_diagnostics['first_safe_route_wall_s'] = time.perf_counter()-self._request_started
            return replace(route, target_pose=contact, continuation_target_pose=contact,
                full_pick_plan=bundle, continuation_trajectory=approach if direct else descent,
                edge_durations=approach.edge_durations,
                continuation_path=tuple(q for phase, _t, q in checked
                    if phase == ('approach' if direct else 'descent')),
                cost=(duration, route.cost[1], len(route.path)),
                path_fully_validated=True, continuation_fully_validated=True)
        return self._full_failure(request, error, started)

    def _plan_full_pick_delivery(self, request, cancelled, started):
        """One executable pick; at most one static-start delivery correction.

        An infeasible candidate may require a longer motion BEFORE a complete
        solution exists. A late first solution estimates ONE future request;
        the changed target and all four curves require fresh certification.
        The compute allowance describes all four phases, not a short-circuited
        rejection from the previous candidate.
        """
        dispatch = request.trajectory_minimum_duration
        start = request.continuation_start_time_s
        if request.intercept_account_for_planning_time:
            start = max(start, (request.intercept_request_time_s or 0.)
                        + request.full_plan_compute_budget_s + dispatch)
        duration = request.timed_continuation_duration_s

        def candidate(duration):
            capture = start+duration
            shift = np.asarray(request.continuation_velocity_m_s)*(capture-request.planned_target_time_s)
            return replace(request,
                target_poses=tuple(np.r_[pose[:3]+shift, pose[3:]] for pose in request.target_poses),
                continuation_start_time_s=start, planned_target_time_s=capture,
                timed_continuation_duration_s=duration)

        current = candidate(duration)
        attempts = []
        report = self._request_diagnostics.setdefault('full_preplanned', {
            'attempts': [], 'certified': False, 'descent_validation_wall_s': 0.,
            'time_recertification_wall_s': 0., 'nominal_lift_conditional': True})
        report.update(delivery_policy='one_plan_with_one_pre_motion_delivery_correction',
            compute_budget_s=request.full_plan_compute_budget_s, fixed_start_s=start,
            time_recertification_wall_s=0., appointment_changes_after_feasible=0,
            feasible_candidates=0, time_refinements=attempts)
        for index in range(3):
            contact = float(current.planned_target_time_s)
            if ((request.full_contact_earliest_s is not None and contact < request.full_contact_earliest_s - 1e-9)
                    or (request.full_contact_latest_s is not None and contact > request.full_contact_latest_s + 1e-9)):
                report.update(certified=False, failure_reason='full_pick_contact_outside_reachable_window')
                return self._full_failure(current, 'full_pick_contact_outside_reachable_window', started)
            self._response_duration_proposals_allowed = index < 2
            attempt_started = time.perf_counter()
            result = self._plan_full_pick_once(current, cancelled, attempt_started)
            delivery = (request.intercept_request_time_s or 0.)+time.perf_counter()-started
            attempts.append({'index': index, 'wall_s': time.perf_counter()-attempt_started,
                'success': result.success, 'start_s': start,
                'capture_s': current.planned_target_time_s, 'delivery_s': delivery,
                'response': dict(report.get('response', {}))})
            if cancelled():
                report.update(certified=False, failure_reason='full_pick_cancelled')
                return replace(result, path=None, full_pick_plan=None, cancelled=True,
                    path_fully_validated=False, continuation_fully_validated=False,
                    effective_request=current, error='full pick cancelled')
            if result.success:
                report['feasible_candidates'] = 1
                if (not request.intercept_account_for_planning_time
                        or delivery+dispatch <= start+1e-9):
                    report.update(certified=True, delivery_s=delivery,
                        start_wait_s=max(0., start-delivery))
                    return replace(result, effective_request=current)
                # Only the initial stationary job can change its appointment.
                # An already moving reference must never jump to a new clock.
                stationary = not (np.any(np.asarray(request.continuation_start_velocity))
                    or np.any(np.asarray(request.continuation_start_acceleration)))
                if stationary:
                    revision = self._full_pick_delivery_request(current, result,
                        attempts[-1]['wall_s'], started, cancelled)
                    revision_contact=float(revision.planned_target_time_s)
                    if ((request.full_contact_earliest_s is not None and revision_contact < request.full_contact_earliest_s-1e-9)
                            or (request.full_contact_latest_s is not None and revision_contact > request.full_contact_latest_s+1e-9)):
                        report.update(certified=False, failure_reason='full_pick_contact_outside_reachable_window')
                        return self._full_failure(revision, 'full_pick_contact_outside_reachable_window', started)
                    revision_started = time.perf_counter()
                    self._response_duration_proposals_allowed = False
                    revised = self._plan_full_pick_once(revision, cancelled, revision_started)
                    revised_wall = time.perf_counter()-revision_started
                    delivery = (request.intercept_request_time_s or 0.)+time.perf_counter()-started
                    attempts.append({'index': index+1, 'wall_s': revised_wall,
                        'success': revised.success, 'start_s': revision.continuation_start_time_s,
                        'capture_s': revision.planned_target_time_s, 'delivery_s': delivery,
                        'kind': 'pre_motion_delivery_correction'})
                    report.update(appointment_changes_after_feasible=1,
                        time_recertification_wall_s=revised_wall,
                        effective_start_s=revision.continuation_start_time_s,
                        delivery_s=delivery, feasible_candidates=1+int(revised.success))
                    if cancelled():
                        report.update(certified=False, failure_reason='full_pick_cancelled')
                        return replace(revised, path=None, full_pick_plan=None, cancelled=True,
                            path_fully_validated=False, continuation_fully_validated=False,
                            effective_request=revision, error='full pick cancelled')
                    if not revised.success:
                        report.update(certified=False, failure_reason=revised.error)
                        return replace(revised, effective_request=revision)
                    current, result = revision, revised
                    start = current.continuation_start_time_s
                    if delivery+dispatch <= start+1e-9:
                        report.update(certified=True, start_wait_s=max(0., start-delivery))
                        return replace(result, effective_request=current)
                # A second late result is discarded. No unbounded chasing and
                # no old certificate receives a shifted clock.
                report.update(certified=False, delivery_s=delivery,
                    failure_reason='full_pick_delivery_missed_start', start_wait_s=0.)
                return replace(result, path=None, full_pick_plan=None,
                    path_fully_validated=False, continuation_fully_validated=False,
                    effective_request=current, error='full_pick_delivery_missed_start')
            suggested = float(report.get('suggested_duration_s', 0.))
            if suggested <= duration or index == 2:
                return replace(result, effective_request=current)
            # The candidate has NOT passed all constraints. Keep the initial
            # start and frozen velocity; certify the new future endpoint anew.
            # No live observation or completed certificate is reused here.
            duration = suggested
            current = candidate(duration)
        raise AssertionError('bounded full pick candidate search did not return')

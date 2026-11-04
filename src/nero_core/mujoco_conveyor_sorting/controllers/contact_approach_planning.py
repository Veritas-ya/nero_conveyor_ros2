"""Certified direct approach for the explicit feedback-closing alternative.

Only the approach is executable. A constant-velocity extension is used solely
to form/forecast the inverse-response spline near its endpoint; it is never
returned as a closing/lift certificate or sent to the robot.
"""
from dataclasses import replace
import json
import time
import numpy as np

from .continuous_intercept_trajectory import ContinuousInterceptTrajectory,InterceptTimingError
from .full_pick_response import eligible_response,command_curves,forecast_tracking,command_within_limits,tangent_projected_curves
from .joint_response import JointResponseModel


class ForecastExtension:
    def __init__(self,curve,duration):
        self.q0,self.velocity,self.acceleration=curve.sample_full(curve.duration)
        self.duration=duration
    def sample_full(self,t):
        t=float(t)
        return (self.q0+self.velocity*t+self.acceleration*t*t*.5,
                self.velocity+self.acceleration*t,self.acceleration.copy())


def terminal_acceleration(engine,request,q,speed):
    if not request.full_contact_acceleration_m_s2:return np.zeros(7)
    velocity=np.asarray(request.continuation_velocity_m_s)
    norm=float(np.linalg.norm(velocity))
    if norm<1e-6:raise ValueError('contact acceleration requires target motion')
    qd=engine._full_terminal_velocity(request,q,nullspace_speed=speed)
    h=1e-4
    jac=engine.ik.compute_jacobian(q)
    jdot=(engine.ik.compute_jacobian(q+h*qd)-engine.ik.compute_jacobian(q-h*qd))/(2*h)
    rhs=np.r_[velocity/norm*request.full_contact_acceleration_m_s2,np.zeros(3)]-jdot@qd
    qdd=np.linalg.lstsq(jac,rhs,rcond=None)[0]
    if not np.isfinite(qdd).all() or np.linalg.norm(jac@qdd-rhs)>1e-6:
        raise ValueError('contact acceleration cannot match target direction')
    return qdd


def approach(engine,request,path,low_q,speed,*,turn=None,cancelled=lambda:False,defer_entry=True):
    if request.full_orientation_preparation_s:
        from .orientation_approach import prepared_approach
        curve=prepared_approach(engine,request,path,low_q,speed,turn=turn,cancelled=cancelled)
    else:
        curve=ContinuousInterceptTrajectory(path,request.trajectory_velocity_limits,
            request.trajectory_acceleration_limits,duration=request.timed_continuation_duration_s,
            initial_velocity=request.continuation_start_velocity,
            initial_acceleration=request.continuation_start_acceleration,
            final_velocity=engine._full_terminal_velocity(request,low_q,nullspace_speed=speed),
            final_acceleration=terminal_acceleration(engine,request,low_q,speed),
            jerk_limits=request.trajectory_jerk_limits)
    if defer_entry and request.full_finger_entry_height_m:
        from .orientation_approach import deferred_finger_entry
        return deferred_finger_entry(engine,request,curve,cancelled)
    return curve


def candidates(engine,request,path,low_q,report,cancelled=lambda:False):
    turn=None
    if request.full_orientation_preparation_s:
        from .orientation_approach import preparation_curve
        turn=preparation_curve(engine,request,cancelled)
    reference=approach(engine,request,path,low_q,0.,turn=turn,cancelled=cancelled,defer_entry=False)
    fallback=(float('inf'),reference,None,{'status':'original_reference','applied':False},None)
    baseline=reference
    if request.full_finger_entry_height_m:
        from .orientation_approach import deferred_finger_entry
        try:baseline=deferred_finger_entry(engine,request,reference,cancelled)
        except ValueError as exc:report['finger_entry_candidate_unavailable']=str(exc)
    if not request.full_approach_response_enabled:return [fallback]
    model=None
    try:
        if request.full_response_model_json:
            model=JointResponseModel.from_dict(json.loads(request.full_response_model_json))
    except (ValueError,TypeError,KeyError,AttributeError):pass
    if (not eligible_response(model) or min(request.full_closure_horizontal_tolerance_m,
            request.full_closure_vertical_tolerance_m,request.full_closure_orientation_tolerance_rad)<=0.):
        return [fallback]
    return response_candidates(engine,request,path,low_q,report,model,baseline,turn,fallback,cancelled)


def response_candidates(engine,request,path,low_q,report,model,baseline,turn,fallback,cancelled):
    """Certify in priority order before forecasting a less preferred speed."""
    qualified_count=0
    report['response_candidate_priority']='least_redundant_velocity_then_own_command_forecast'
    trials=report.setdefault('response_redundancy_candidates',[])
    for speed in (0.,.8,1.,1.125,1.25,1.35):
        if cancelled():break
        try:
            curve=baseline if speed==0. else approach(engine,request,path,low_q,speed,turn=turn,cancelled=cancelled)
            nominal=(curve,None,ForecastExtension(curve,request.full_close_window_s),None)
            proposed=command_curves(request,nominal,model)
            ratio=command_within_limits(request,(proposed[0],None,None,None))
            nominal_ratio=command_within_limits(request,(curve,None,None,None))
            if max(ratio,nominal_ratio)>1.+1e-9:
                trials.append(dict(nullspace_speed_rad_s=speed,derivative_ratio=ratio,
                    nominal_command_ratio=nominal_ratio,usable=False,
                    reason='original_joint_derivative_limits_rejected_before_forecast'))
                continue
            forecast=forecast_tracking(request,nominal,proposed,model,engine.ik)
            base=forecast_tracking(request,nominal,nominal,model,engine.ik)
            usable=(max(ratio,nominal_ratio)<=1.+1e-9 and forecast['constraint_ratio']<=1.
                and forecast['predicted_same_time_peak_m']<base['predicted_same_time_peak_m'])
            diagnostic=dict(status='candidate_requires_full_approach_certificate',applied=False,
                model_metadata=model.metadata,nullspace_speed_rad_s=speed,
                command_derivative_ratio=ratio,nominal_command_ratio=nominal_ratio,
                finger_entry_shape=getattr(curve,'finger_entry_shape',None),
                nominal_forecast=base,proposed_forecast=forecast)
            trials.append(dict(nullspace_speed_rad_s=speed,derivative_ratio=ratio,
                nominal_command_ratio=nominal_ratio,forecast=forecast,usable=usable))
            if usable:
                qualified_count+=1
                options=[(forecast['constraint_ratio'],curve,proposed[0],diagnostic,proposed)]
                alternative=projected_candidate(engine,request,curve,proposed,diagnostic,report,cancelled)
                if alternative is not None:
                    command,details=alternative
                    options.append((details['proposed_forecast']['constraint_ratio'],curve,command,details,None))
                # Both commands have the same redundant endpoint velocity;
                # compare their own forecasts before certifying either one.
                yield from sorted(options,key=lambda item:item[0])
                if qualified_count>=2:break
        except ValueError as exc:
            trials.append(dict(nullspace_speed_rad_s=speed,usable=False,reason=str(exc)))
    yield fallback


def projected_candidate(engine,request,curve,raw_commands,diagnostic,report,cancelled):
    fraction=request.full_response_projection_fraction
    row=dict(nullspace_speed_rad_s=diagnostic.get('nullspace_speed_rad_s'),fraction=fraction)
    report.setdefault('response_projection_candidates',[]).append(row)
    began=time.perf_counter()
    try:
        if cancelled():return None
        if fraction==0.:
            row['status']='zero_projection_reuses_unprojected_command'
            return None
        nominal=(curve,None,ForecastExtension(curve,request.full_close_window_s),None)
        model=JointResponseModel.from_dict(json.loads(request.full_response_model_json))
        proposal,notes=tangent_projected_curves(request,nominal,raw_commands,engine.ik,fraction=fraction,cancelled=cancelled)
        ratio=command_within_limits(request,(proposal[0],None,None,None))
        forecast=forecast_tracking(request,nominal,proposal,model,engine.ik)
        row.update(projection=notes,derivative_ratio=ratio,forecast=forecast)
        if (ratio>1.+1e-9 or forecast['constraint_ratio']>1.
                or forecast['predicted_same_time_peak_m']>=diagnostic['nominal_forecast']['predicted_same_time_peak_m']):
            row['status']='model_or_original_command_limits_rejected';return None
        row.update(status='requires_full_geometry_certificate',checked_samples=0)
        return proposal[0],dict(diagnostic,projection=notes,
            projection_candidate_index=len(report['response_projection_candidates'])-1,
            unprojected_forecast=diagnostic['proposed_forecast'],proposed_forecast=forecast,
            command_derivative_ratio=ratio)
    except ValueError as error:
        row.update(status='projection_unavailable',reason=str(error));return None
    finally:row['wall_s']=time.perf_counter()-began


def plan_once(engine,request,cancelled,started):
    from .full_pick_planning import FullPickPlan
    report=engine._request_diagnostics.setdefault('full_preplanned',dict(
        attempts=[],certified=False,descent_validation_wall_s=0.,time_recertification_wall_s=0.))
    report.update(certificate_scope='unloaded_direct_approach_until_contact',
        feedback_closing_and_loaded_motion='original_shared_controller_rechecks_actual_feedback',
        nominal_lift_conditional=False)
    seen=[];error='no certified direct contact approach'
    for seed in request.goal_seeds:
        if cancelled() or len(seen)>=request.max_goal_candidates:break
        engine._diagnostic_context={'phase':'contact_approach','check':'contact_ik'}
        low_q=engine._solve_goal_seed(request,request.target_pose,seed,cancelled)
        if low_q is None or any(np.max(abs(low_q-old))<1e-4 for old in seen):continue
        seen.append(low_q.copy())
        route_request=replace(request,planning_mode='legacy',continuous_intercept=False,
            target_is_contact=True,goal_seeds=(low_q,),continuation_target_poses=None,
            timed_continuation_duration_s=None,continuation_start_time_s=None,
            intercept_velocity_m_s=None,intercept_account_for_planning_time=False)
        route=engine._plan(route_request,cancelled)
        if not route.success:
            error=route.error or error;continue
        try:options=candidates(engine,request,route.path,route.path[-1],report,cancelled)
        except InterceptTimingError as exc:
            report['suggested_duration_s']=max(report.get('suggested_duration_s',0.),exc.suggested_duration_s)
            error=str(exc);continue
        except ValueError as exc:error=str(exc);continue
        nominal_certificates={}
        for _,curve,command,diagnostic,raw_commands in options:
            if cancelled():break
            begin=time.perf_counter()
            reused=id(curve) in nominal_certificates
            if reused:nominal_samples=nominal_certificates[id(curve)]
            else:
                nominal_samples=engine._full_certify_curve(request,curve,request.continuation_start_time_s,
                    'approach',cancelled)
                nominal_certificates[id(curve)]=nominal_samples
            report['attempts'].append(dict(phase='approach',safe=nominal_samples is not None,
                samples=0 if nominal_samples is None else len(nominal_samples),wall_s=time.perf_counter()-begin,
                exact_nominal_curve_reused_within_request=reused))
            if nominal_samples is None:continue
            command_samples=()
            if command is not None:
                command_samples=engine._full_certify_curve(request,command,request.continuation_start_time_s,
                    'approach',cancelled,command=True)
                projection_index=diagnostic.get('projection_candidate_index')
                if projection_index is not None:
                    report['response_projection_candidates'][projection_index].update(
                        status='fully_certified' if command_samples is not None else 'geometry_rejected',
                        checked_samples=0 if command_samples is None else len(command_samples))
                if command_samples is None:
                    continue
                diagnostic.update(status='fully_certified_response_approach',applied=True)
            report['response']=diagnostic
            report['response_nullspace_speed_rad_s']=diagnostic.get('nullspace_speed_rad_s',0.)
            floor=min(float(engine.ik.forward_pose(curve.q0)[2]),float(request.target_pose[2]))-.012
            bundle=FullPickPlan(request,curve,None,None,None,request.target_pose,None,None,
                tuple(nominal_samples),floor,None if command is None else (command,None,None,None),
                tuple(command_samples))
            report.update(certified=True,start_s=bundle.start,capture_s=bundle.contact_time,
                high_s=None,lift_start_s=None,end_s=bundle.end,mandatory_high_waypoint=False,
                q_v_a_continuous_within_certified_approach=True,
                loaded_motion_not_part_of_this_certificate=True,jaw_sweep_spacing_m=.01)
            engine._request_diagnostics['first_safe_route_wall_s']=time.perf_counter()-engine._request_started
            return replace(route,full_pick_plan=bundle,effective_request=request,
                target_pose=request.target_pose,continuation_target_pose=request.target_pose,
                continuation_path=tuple(q for _,_,q in nominal_samples),continuation_trajectory=curve,
                path=tuple(curve.path),edge_durations=curve.edge_durations,path_fully_validated=True,continuation_fully_validated=True)
    return engine._full_failure(request,error,started)


def delivery_request(engine,request,result,complete_s,started,cancelled):
    """One bounded late-delivery correction, with new IK and recertification."""
    computation=complete_s;duration=request.timed_continuation_duration_s
    bundle=result.full_pick_plan
    old_work,_=engine._full_pick_validation_work(request,(bundle.approach,None,None,None))
    seed=bundle.approach.q1.copy();forecasts=[]
    def proposed():
        start=max(request.continuation_start_time_s,(request.intercept_request_time_s or 0.)
            +time.perf_counter()-started+computation+request.trajectory_minimum_duration)
        capture=start+duration
        shift=np.asarray(request.continuation_velocity_m_s)*(capture-request.planned_target_time_s)
        return replace(request,target_poses=tuple(np.r_[p[:3]+shift,p[3:]] for p in request.target_poses),
            continuation_start_time_s=start,intercept_motion_start_time_s=start,planned_target_time_s=capture,
            timed_continuation_duration_s=duration,goal_seeds=(seed,*request.goal_seeds),
            route_waypoint_hints=tuple(result.path[1:-1]))
    for _ in range(min(2,request.max_goal_candidates)):
        current=proposed()
        q=engine._solve_goal_seed(current,current.target_pose,seed,cancelled)
        if q is None or cancelled():break
        try:curve=approach(engine,current,(*result.path[:-1],q),q,0.)
        except InterceptTimingError as exc:duration=max(duration,exc.suggested_duration_s);continue
        except ValueError:break
        work,_=engine._full_pick_validation_work(current,(curve,None,None,None))
        next_compute=complete_s*max(1.,work/max(1,old_work))
        forecasts.append(dict(capture_s=current.planned_target_time_s,weighted_samples=work,
            compute_forecast_s=next_compute))
        if next_compute<=computation:break
        computation=next_compute;seed=q
    engine._request_diagnostics['full_preplanned']['delivery_forecasts']=forecasts
    return proposed()

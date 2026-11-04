"""Advisory servo forecasts and fully recertifiable direct-pick references.

No measured state is changed. The fitted model is not a collision certificate.
The caller retains the original nominal checks and certifies the proposed
command curves separately. Missing/unvalidated models leave ordinary execution
available. Loaded lift always retains its original command trajectory.
"""
from math import comb
from dataclasses import replace

import numpy as np
from scipy.interpolate import make_interp_spline, PPoly
from scipy.spatial.transform import Rotation,Slerp

from ._math import quat_distance
from .joint_response import predict_position_response


def eligible_response(model):
    if model is None or not model.metadata.get('runtime_eligible', False):
        return False
    if len(model.axes) != 7:
        return False
    return all(a is not None and a.status == 'identified'
        and np.isfinite([a.delay_s, a.time_constant_s, a.holdout_rmse_rad,
                         a.unmodeled_holdout_rmse_rad]).all()
        and a.holdout_samples >= 2 and a.time_constant_s > 0 and a.delay_s >= 0
        and a.holdout_rmse_rad <= a.unmodeled_holdout_rmse_rad
        for a in model.axes)


class ResponseCurve:
    """Slice of one C4 quintic spline; conservative Bernstein bounds per span."""
    backend = 'response_quintic_spline'

    def __init__(self, spline, begin, end):
        self.spline, self.begin, self.duration = spline, float(begin), float(end-begin)
        self.q0 = np.asarray(spline(begin)).copy()
        self.q1 = np.asarray(spline(end)).copy()
        knots = np.unique(np.r_[begin, spline.t[(spline.t > begin) & (spline.t < end)], end])
        self.edge_durations = tuple(np.diff(knots))
        self._bounds = {}
        self._power_basis = None

    def _derivative_bounds(self,order):
        if self._power_basis is None:
            self._power_basis=tuple(PPoly.from_spline((self.spline.t,self.spline.c[:,axis],self.spline.k))
                for axis in range(7))
        begin,end=self.begin,self.begin+self.duration
        if order not in self._bounds:
            lows, highs = [], []
            for axis in range(7):
                p = self._power_basis[axis].derivative(order)
                axis_lo, axis_hi = [], []
                for k, (left, right) in enumerate(zip(p.x, p.x[1:])):
                    if right <= begin or left >= end or right <= left:
                        continue
                    # Full knot-span bounds also bound a clipped boundary span.
                    powers = p.c[:, k][::-1] * (right-left)**np.arange(p.c.shape[0])
                    n = len(powers)-1
                    bernstein = [sum(powers[j]*comb(i,j)/comb(n,j) for j in range(i+1))
                                 for i in range(n+1)]
                    axis_lo.append(min(bernstein)); axis_hi.append(max(bernstein))
                lows.append(min(axis_lo)); highs.append(max(axis_hi))
            self._bounds[order] = np.array(lows), np.array(highs)

    def sample_full(self, timestamp):
        t = self.begin+float(np.clip(timestamp, 0., self.duration))
        return tuple(np.asarray(self.spline(t, nu=k)) for k in range(3))

    def sample(self, timestamp):
        return self.sample_full(timestamp)[:2]

    def derivative_bounds(self, order):
        if order not in range(4):raise KeyError(order)
        self._derivative_bounds(order)
        return tuple(x.copy() for x in self._bounds[order])


def command_curves(request, curves, model):
    """Smooth inverse-response proposal through approach and jaw closing only.

    The nominal contact/closing reference stays unchanged. The correction fades
    out over the existing closing window and matches original q/v/a at lift.
    No retiming, live gain or unverified per-cycle offset is applied downstream.
    """
    approach, descent, closing, lift = curves
    if descent is not None or not eligible_response(model):
        return None
    contact, end = approach.duration, approach.duration+closing.duration

    def nominal(t):
        return (approach.sample_full(t) if t < contact
                else closing.sample_full(min(t-contact, closing.duration)))

    # Knot spacing is a representation choice, not a controller period or gate.
    times = np.unique(np.r_[np.linspace(0., contact, max(8, int(np.ceil(contact/.08)))+1),
                            np.linspace(contact, end, max(8, int(np.ceil(closing.duration/.08)))+1)])
    positions = []
    for t in times:
        q = nominal(t)[0]
        u = float(np.clip((t-contact)/closing.duration, 0., 1.))
        fade = 1.-(10*u**3-15*u**4+6*u**5)
        proposed = np.array([nominal(min(end,t+a.delay_s))[0][i]
            + a.time_constant_s*nominal(min(end,t+a.delay_s))[1][i]
            for i,a in enumerate(model.axes)])
        positions.append(q+fade*(proposed-q))
    initial, final = nominal(0.), nominal(end)
    positions[0], positions[-1] = initial[0], final[0]
    spline = make_interp_spline(times, np.asarray(positions), k=5,
        bc_type=([(1,initial[1]),(2,initial[2])],[(1,final[1]),(2,final[2])]))
    return (ResponseCurve(spline,0.,contact), None,
            ResponseCurve(spline,contact,end), lift)


def tangent_projected_curves(request,nominal,raw,ik,*,fraction=.25,cancelled=lambda:False):
    """A separately certified candidate; only explicit contact prefixes use it.

    Retain the model's along-track lead, reduce transverse and orientation
    corrections at representation knots, then fit the original C4 time law.
    The unexecuted suffix fades the contact joint correction to zero so its
    original end q/v/a stay intact. It is not a payload motion certificate.
    """
    if not request.full_contact_handoff_enabled or request.full_approach_mode!='continuous':
        raise ValueError('tangent projection requires the explicit contact prefix')
    direction=np.asarray(request.continuation_velocity_m_s,dtype=float)
    length=float(np.linalg.norm(direction))
    if length<1e-6 or not np.isfinite(fraction) or not 0.<=fraction<=1.:
        raise ValueError('finite moving direction and projection fraction required')
    direction=direction/length
    import mujoco
    from .ik_solver import DampedLeastSquaresIK
    projector=DampedLeastSquaresIK(ik.model,mujoco.MjData(ik.model),ik.joint_names,ik.ee_site_name,
        replace(ik.config,position_tolerance=min(ik.config.position_tolerance,.0003),
            orientation_tolerance=min(ik.config.orientation_tolerance,.001),joint_centering_gain=0.))
    projector.lower_limits=np.array(ik.lower_limits if request.full_command_lower_limits is None
        else request.full_command_lower_limits,copy=True)
    projector.upper_limits=np.array(ik.upper_limits if request.full_command_upper_limits is None
        else request.full_command_upper_limits,copy=True)
    contact=nominal[0].duration;end=contact+nominal[2].duration
    times=np.unique(np.r_[0.,raw[0].spline.t,contact,end]);times=times[(times>=0.)&(times<=end)]
    def sample(curves,t):
        return curves[0].sample_full(t) if t<contact else curves[2].sample_full(t-contact)
    positions=[];delta=None;max_position=max_angle=0.
    for t in times:
        if cancelled():raise ValueError('tangent projection cancelled')
        q=sample(raw,t)[0]
        if t<=contact:
            cart=ik.forward_pose(q);reference=ik.forward_pose(sample(nominal,t)[0])
            correction=cart[:3]-reference[:3]
            desired_position=cart[:3]-fraction*(correction-direction*float(correction@direction))
            rotations=Rotation.from_quat(np.array([cart[3:],reference[3:]])[:,[1,2,3,0]])
            desired_quaternion=Slerp([0.,1.],rotations)([fraction]).as_quat()[0][[3,0,1,2]]
            solved=projector.solve(desired_position,desired_quaternion,qpos_seed=q)
            if not solved.success:raise ValueError('tangent projection IK '+solved.message)
            target=solved.joint_positions
            max_position=max(max_position,float(solved.position_error))
            max_angle=max(max_angle,float(solved.orientation_error))
            if abs(t-contact)<1e-9:delta=target-q
        else:
            u=(t-contact)/(end-contact);fade=1.-(10*u**3-15*u**4+6*u**5)
            target=q+fade*delta
        positions.append(target)
    initial=sample(nominal,0.);final=sample(nominal,end)
    positions[0]=initial[0];positions[-1]=final[0]
    spline=make_interp_spline(times,np.array(positions),k=5,
        bc_type=([(1,initial[1]),(2,initial[2])],[(1,final[1]),(2,final[2])]))
    return (ResponseCurve(spline,0.,contact),None,ResponseCurve(spline,contact,end),nominal[3]),dict(
        method='along_track_response_with_reduced_transverse_and_orientation_correction',
        fraction=float(fraction),direction_m_s_normalized=direction.tolist(),
        maximum_representation_ik_position_error_m=max_position,
        maximum_representation_ik_orientation_error_rad=max_angle,
        projection_at_representation_knots_not_an_exact_cartesian_curve_constraint=True)


def _forecast(request, nominal, commands, model, ik):
    approach, _, closing, _ = commands
    contact, end = approach.duration, approach.duration+closing.duration
    dt = request.full_control_period_s
    times = np.unique(np.r_[np.arange(0.,end,dt),contact,end])
    send_times = np.unique(np.r_[np.arange(0.,end,dt),end])

    def sample(curves,t):
        a,_,c,_ = curves
        return a.sample_full(t) if t < contact else c.sample_full(t-contact)

    sent = np.asarray([sample(commands,t)[0] for t in send_times])
    actual = np.column_stack([predict_position_response(send_times, sent[:,i], times,
        request.start_q[i], axis.delay_s,axis.time_constant_s) for i,axis in enumerate(model.axes)])
    aligned, prepared, angular, joint, closure_ratios, closing_error = [], [], [], [], [], []
    evaluation_end=contact if request.full_contact_handoff_enabled else end
    post_contact_prepared=[];post_contact_closure=[]
    for t,feedback in zip(times,actual):
        measured = ik.forward_pose(feedback)
        ref = ik.forward_pose(sample(nominal,t)[0])
        ahead = sample(commands,min(end,t+request.full_dispatch_lead_s))[0]
        command = ik.forward_pose(ahead)
        if t>evaluation_end+1e-12:
            # This suffix is retired only after genuine contact in adaptive
            # execution. Keep its forecast as diagnostics; it cannot choose
            # the timing of an approach which will not execute that suffix.
            post_contact_prepared.append(float(np.linalg.norm(measured[:3]-command[:3])))
            target=request.target_pose[:3]+np.asarray(request.continuation_velocity_m_s)*(t-contact)
            error=measured[:3]-target
            post_contact_closure.append(max(float(np.linalg.norm(error[:2]))/request.full_closure_horizontal_tolerance_m,
                abs(float(error[2]))/request.full_closure_vertical_tolerance_m,
                float(quat_distance(measured[3:],request.target_pose[3:]))/request.full_closure_orientation_tolerance_rad))
            continue
        aligned.append(float(np.linalg.norm(measured[:3]-ref[:3])))
        prepared.append(float(np.linalg.norm(measured[:3]-command[:3])))
        angular.append(float(quat_distance(measured[3:],command[3:])))
        envelope = request.acquisition_joint_limit_rad if t < contact else request.continuation_max_joint_step
        joint.append(float(np.max(np.abs(feedback-ahead)))/envelope)
        if t >= contact:
            target = request.target_pose[:3]+np.asarray(request.continuation_velocity_m_s)*(t-contact)
            error = measured[:3]-target
            closing_error.append(float(np.linalg.norm(error)))
            closure_ratios.append(max(float(np.linalg.norm(error[:2]))/request.full_closure_horizontal_tolerance_m,
                abs(float(error[2]))/request.full_closure_vertical_tolerance_m,
                float(quat_distance(measured[3:],request.target_pose[3:]))/request.full_closure_orientation_tolerance_rad))
    contact_index = int(np.searchsorted(times, contact))
    ratio = max(max(prepared)/request.full_feedback_translation_limit_m,
                max(angular)/request.full_feedback_rotation_limit_rad, max(joint),
                max(closure_ratios))
    return dict(predicted_same_time_peak_m=max(aligned),
        predicted_prepared_distance_peak_m=max(prepared),
        predicted_prepared_rotation_peak_rad=max(angular),
        predicted_joint_envelope_ratio=max(joint),
        predicted_contact_error_m=aligned[contact_index],
        predicted_closing_error_peak_m=max(closing_error), constraint_ratio=ratio,
        predicted_closure_envelope_ratio=max(closure_ratios),
        evaluation_scope=('approach_through_contact_for_explicit_handoff'
            if request.full_contact_handoff_enabled else 'full_approach_and_closing'),
        evaluation_end_offset_s=float(evaluation_end),
        retired_suffix_diagnostic={'prepared_distance_peak_m':max(post_contact_prepared,default=0.),
            'closure_envelope_ratio':max(post_contact_closure,default=0.)},
        sample_period_s=dt, samples=len(times),
        source='fitted_first_order_delay_model_not_physical_success')


def forecast_tracking(request, nominal, commands, model, ik):
    return _forecast(request, nominal, commands, model, ik)


def command_within_limits(request, curves):
    ratio = 1.
    for curve in curves:
        if curve is None:
            continue
        if request.full_command_lower_limits is not None and request.full_command_upper_limits is not None:
            lower=np.asarray(request.full_command_lower_limits);upper=np.asarray(request.full_command_upper_limits)
            if lower.shape!=(7,) or upper.shape!=(7,) or not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>=upper):
                raise ValueError('invalid inherited command position limits')
            center=(lower+upper)/2.;radius=(upper-lower)/2.
            lo,hi=curve.derivative_bounds(0)
            ratio=max(ratio,float(np.max(np.maximum(abs(lo-center),abs(hi-center))/radius)))
        for order,limit in ((1,request.trajectory_velocity_limits),
                            (2,request.trajectory_acceleration_limits),
                            (3,request.trajectory_jerk_limits)):
            if limit is None:
                continue
            lo,hi = curve.derivative_bounds(order)
            ratio = max(ratio,float(np.max(np.maximum(abs(lo),abs(hi))/limit))**(1./order))
    return ratio

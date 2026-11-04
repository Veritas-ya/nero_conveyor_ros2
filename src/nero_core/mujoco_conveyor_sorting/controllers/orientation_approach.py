"""Optional fully recertified turn at the current TCP before low interception."""
import numpy as np
from dataclasses import replace
from .continuous_intercept_trajectory import ContinuousInterceptTrajectory


class JoinedApproach:
    backend='fixed_deadline_quintic'
    requires_recertification=True
    geometry_preserved=False

    def __init__(self,first,second):
        if not all(np.allclose(a,b,atol=1e-7,rtol=0.) for a,b in zip(
                first.sample_full(first.duration),second.sample_full(0.))):
            raise ValueError('orientation approach seam must preserve q/v/a')
        self.first,self.second=first,second
        self.duration=first.duration+second.duration
        self.q0,self.q1=first.q0,second.q1
        self.path=(*first.path,*second.path[1:])
        self.edge_durations=(*first.edge_durations,*second.edge_durations)

    def sample_full(self,t):
        t=float(np.clip(t,0.,self.duration))
        return self.first.sample_full(t) if t<self.first.duration else self.second.sample_full(t-self.first.duration)

    def sample(self,t):return self.sample_full(t)[:2]

    def derivative_bounds(self,order):
        a,b=self.first.derivative_bounds(order),self.second.derivative_bounds(order)
        return np.minimum(a[0],b[0]),np.maximum(a[1],b[1])


def preparation_curve(engine,request,cancelled):
    current=engine.ik.forward_pose(request.start_q)
    goal=np.r_[current[:3],request.target_pose[3:]]
    if request.full_orientation_preparation_tracking_entry:
        remaining=request.timed_continuation_duration_s-request.full_orientation_preparation_s
        goal[:2]=np.asarray(request.target_pose[:2])-np.asarray(request.continuation_velocity_m_s[:2])*remaining*.5
    engine._diagnostic_context={'phase':'contact_approach','check':'orientation_preparation_ik'}
    q=engine._solve_goal_seed(request,goal,request.start_q,cancelled)
    if q is None:raise ValueError('orientation at original start position has no safe IK')
    return ContinuousInterceptTrajectory((request.start_q,q),duration=request.full_orientation_preparation_s,
        initial_velocity=request.continuation_start_velocity,initial_acceleration=request.continuation_start_acceleration,
        velocity_limits=request.trajectory_velocity_limits,
        acceleration_limits=request.trajectory_acceleration_limits,jerk_limits=request.trajectory_jerk_limits)


def prepared_approach(engine,request,path,low_q,speed,*,turn=None,cancelled=lambda:False):
    preparation=request.full_orientation_preparation_s
    remaining=request.timed_continuation_duration_s-preparation
    if remaining<=0.:raise ValueError('orientation preparation leaves no intercept motion time')
    if turn is None:turn=preparation_curve(engine,request,cancelled)
    q=turn.q1
    common=dict(velocity_limits=request.trajectory_velocity_limits,
        acceleration_limits=request.trajectory_acceleration_limits,jerk_limits=request.trajectory_jerk_limits)
    from .contact_approach_planning import terminal_acceleration
    low=ContinuousInterceptTrajectory((q,*path[1:]),duration=remaining,
        final_velocity=engine._full_terminal_velocity(request,low_q,nullspace_speed=speed),
        final_acceleration=terminal_acceleration(engine,request,low_q,speed),**common)
    return JoinedApproach(turn,low)


def deferred_finger_entry(engine,request,curve,cancelled):
    """Optional C2 geometric proposal; no intermediate arrival/holding state."""
    height=request.full_finger_entry_height_m
    if not height:return curve
    remaining=request.full_finger_entry_lead_s
    stamp=curve.duration-remaining
    if stamp<=0.:raise ValueError('finger entry shape leaves no initial approach time')
    oldq,oldv,olda=curve.sample_full(stamp)
    goal=engine.ik.forward_pose(oldq);goal[2]+=height
    engine._diagnostic_context={'phase':'contact_approach','check':'deferred_finger_entry_ik'}
    midq=engine._solve_goal_seed(request,goal,oldq,cancelled)
    if midq is None:raise ValueError('finger entry shape has no original safe IK')
    midj=engine.ik.compute_jacobian(midq)
    desired=engine.ik.compute_jacobian(oldq)@oldv
    midv=oldv+np.linalg.lstsq(midj,desired-midj@oldv,rcond=None)[0]
    final=curve.sample_full(curve.duration)
    common=dict(velocity_limits=request.trajectory_velocity_limits,
        acceleration_limits=request.trajectory_acceleration_limits,jerk_limits=request.trajectory_jerk_limits)
    first=ContinuousInterceptTrajectory((curve.q0,midq),duration=stamp,
        initial_velocity=curve.sample_full(0.)[1],initial_acceleration=curve.sample_full(0.)[2],
        final_velocity=midv,final_acceleration=olda,**common)
    second=ContinuousInterceptTrajectory((midq,curve.q1),duration=remaining,
        initial_velocity=midv,initial_acceleration=olda,
        final_velocity=final[1],final_acceleration=final[2],**common)
    joined=JoinedApproach(first,second)
    joined.finger_entry_shape=dict(height_increment_m=float(height),lead_s=float(remaining),
        join_offset_s=float(stamp),requested_pose=goal.tolist(),
        actual_ik_pose=engine.ik.forward_pose(midq).tolist(),intermediate_arrival_gate=False)
    return joined

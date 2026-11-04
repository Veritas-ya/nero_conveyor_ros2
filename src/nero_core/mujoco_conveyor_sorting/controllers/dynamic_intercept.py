"""Time-to-contact geometry from policy-visible tracks; no simulator object truth."""
from __future__ import annotations
import numpy as np


def pregrasp_high_pose(capture_pose, velocity, *, approach_height, descent_lead_time,
                       capture_high=False):
    """The same base-frame high endpoint for candidate checks and execution."""
    high = np.asarray(capture_pose, dtype=float).copy()
    if not capture_high:
        high[:3] -= np.asarray(velocity, dtype=float) * descent_lead_time
    high[2] += approach_height
    return high


def intercept_lead(position, velocity, ee_position, *, grasp_height: float,
                   approach_clearance: float, cartesian_speed: float,
                   minimum_motion_time: float, latency: float,
                   maximum_horizon: float) -> float:
    """Solve T = latency + max(T_min, two-leg Cartesian travel time to p+v*T).

    The safe-path planner subsequently replaces this geometric ETA with its
    executable joint-path ETA. No fixed x target or waiting-at-station term.
    Saturated horizons remain predictions, and must pass the caller's workspace
    and deadline checks before execution.
    """
    p=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float);ee=np.asarray(ee_position,dtype=float)
    values=np.r_[p,v,ee,grasp_height,approach_clearance,cartesian_speed,minimum_motion_time,latency,maximum_horizon]
    if p.shape!=(3,) or v.shape!=(3,) or ee.shape!=(3,) or not np.isfinite(values).all():
        raise ValueError('Expected finite 3-D poses and timing parameters')
    if cartesian_speed<=0 or min(grasp_height,approach_clearance,minimum_motion_time,latency)<0 or maximum_horizon<=0:
        raise ValueError('Invalid interception timing limits')
    def residual(t):
        high=p+v*t+np.array([0.,0.,grasp_height+approach_clearance])
        travel=np.linalg.norm(high-ee)/cartesian_speed+approach_clearance/cartesian_speed
        return latency+max(minimum_motion_time,travel)-t
    # Bracket the FIRST reachable horizon; valid also if speed >= nominal arm
    # speed, where a single global monotonicity assumption would be unsafe.
    grid=np.linspace(0.,maximum_horizon,65)
    previous=0.
    for candidate in grid[1:]:
        if residual(candidate)<=0:
            low,high=previous,float(candidate)
            for _ in range(22):
                mid=(low+high)/2
                if residual(mid)>0:low=mid
                else:high=mid
            return high
        previous=float(candidate)
    return float(maximum_horizon)

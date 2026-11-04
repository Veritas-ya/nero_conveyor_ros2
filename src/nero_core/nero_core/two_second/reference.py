"""Fixed-duration Cartesian interception, independent of full_preplanned.

The reference uses a visual capture and its original timestamp. This module
never reads object ground truth and has no hardware transport.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DEFAULT_FREEZE_UPSTREAM_DISTANCE_M = .35
APPROACH_DURATION_S = 2.


class EntryNotReady(ValueError):
    """An observation must remain in tracking rather than freeze a task yet."""
    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason

@dataclass(frozen=True)
class Intercept:
    capture_s: float
    start_s: float
    duration_s: float
    capture_tcp_world: np.ndarray
    velocity_world: np.ndarray
    start_tcp_world: np.ndarray
    start_quaternion_wxyz: np.ndarray
    target_quaternion_wxyz: np.ndarray

    def __post_init__(self):
        for name in ('capture_tcp_world','velocity_world','start_tcp_world'):
            value=np.asarray(getattr(self,name),dtype=float).copy()
            if value.shape!=(3,) or not np.isfinite(value).all(): raise ValueError('invalid '+name)
            value.setflags(write=False);object.__setattr__(self,name,value)
        for name in ('start_quaternion_wxyz','target_quaternion_wxyz'):
            value=np.asarray(getattr(self,name),dtype=float).copy()
            if value.shape!=(4,) or not np.isfinite(value).all() or np.linalg.norm(value)<1e-9: raise ValueError('invalid '+name)
            value/=np.linalg.norm(value);value.setflags(write=False);object.__setattr__(self,name,value)
        if not np.isfinite([self.capture_s,self.start_s,self.duration_s]).all() or self.duration_s<=0 or self.start_s<self.capture_s:
            raise ValueError('invalid intercept clock')

    @property
    def contact_s(self): return self.start_s+self.duration_s

    @property
    def contact_xyz(self): return self.capture_tcp_world+self.velocity_world*(self.contact_s-self.capture_s)

    def sample(self,elapsed_s):
        t=float(elapsed_s)
        if t<0: t=0.
        if t<=self.duration_s:
            u=t/self.duration_s
            xyz=self.start_tcp_world+(self.contact_xyz-self.start_tcp_world)*u
            slerp=Slerp([0.,1.],Rotation.from_quat(np.array([self.start_quaternion_wxyz,self.target_quaternion_wxyz])[:,[1,2,3,0]]))
            q=slerp([u]).as_quat()[0][[3,0,1,2]]
            return xyz,q
        return self.contact_xyz+self.velocity_world*(t-self.duration_s),self.target_quaternion_wxyz.copy()


def predict_entry_time(capture_s, center_x_m, speed_x_m_s, entry_x_m, now_s, reserve_s,
                       *, maximum_upstream_distance_m=None):
    """Schedule in the along-belt coordinate, including capture age.

    Simulation uses world X; a physical cell uses dot(p_base, belt_direction).
    Both callers supply the same spatial freeze window.
    """
    values=np.asarray([capture_s,center_x_m,speed_x_m_s,entry_x_m,now_s,reserve_s],dtype=float)
    if not np.isfinite(values).all() or speed_x_m_s<=0 or reserve_s<0 or capture_s>now_s:
        raise ValueError('invalid upstream observation or velocity')
    if center_x_m>=entry_x_m:
        raise EntryNotReady('observation is not upstream of the grasp region', 'already_at_entry')
    if maximum_upstream_distance_m is not None:
        maximum = float(maximum_upstream_distance_m)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError('maximum upstream distance must be positive')
        if center_x_m < entry_x_m-maximum:
            raise EntryNotReady('observation is too far upstream to freeze', 'too_far_upstream')
    entry=float(capture_s+(entry_x_m-center_x_m)/speed_x_m_s)
    if entry < now_s+reserve_s:
        raise EntryNotReady('insufficient time before predicted entry', 'insufficient_planning_time')
    return entry

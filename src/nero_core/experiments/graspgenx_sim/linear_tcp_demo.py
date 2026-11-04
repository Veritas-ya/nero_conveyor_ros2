"""Trapezoidal Cartesian reference for the MuJoCo-only motion demo."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearTCPReference:
    start_world_m: np.ndarray
    direction_world: np.ndarray
    distance_m: float
    cruise_speed_m_s: float
    ramp_s: float

    def __post_init__(self):
        start=np.asarray(self.start_world_m,dtype=float)
        direction=np.asarray(self.direction_world,dtype=float)
        if (start.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(np.r_[start,direction]).all()
                or not np.isfinite([self.distance_m,self.cruise_speed_m_s,self.ramp_s]).all()
                or min(self.distance_m,self.cruise_speed_m_s,self.ramp_s)<=0 or np.linalg.norm(direction)<1e-12
                or self.distance_m<self.cruise_speed_m_s*self.ramp_s):
            raise ValueError('invalid trapezoid geometry, speed or acceleration interval')
        start=start.copy();direction=direction/np.linalg.norm(direction)
        start.setflags(write=False);direction.setflags(write=False)
        object.__setattr__(self,'start_world_m',start)
        object.__setattr__(self,'direction_world',direction)

    @property
    def duration_s(self):
        return self.distance_m/self.cruise_speed_m_s+self.ramp_s

    def sample(self,t):
        t=float(t)
        if not np.isfinite(t):raise ValueError('reference time must be finite')
        t=float(np.clip(t,0.,self.duration_s));v=self.cruise_speed_m_s;r=self.ramp_s
        acceleration=v/r
        if t<r:
            distance=.5*acceleration*t*t; speed=acceleration*t; phase='accelerating'
        elif t<self.duration_s-r:
            distance=v*(t-r/2.); speed=v;phase='cruise'
        else:
            remaining=self.duration_s-t
            distance=self.distance_m-.5*acceleration*remaining*remaining
            speed=acceleration*remaining;phase='decelerating' if remaining>0 else 'stopped'
        return self.start_world_m+self.direction_world*distance, self.direction_world*speed, phase

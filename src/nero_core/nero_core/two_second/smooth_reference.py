"""Shared smooth interception for measured hardware and MuJoCo adapters."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation, Slerp
from .reference import Intercept


@dataclass(frozen=True)
class SmoothIntercept(Intercept):
    follow_s: float = .8
    brake_s: float = 1.5
    prealign_s: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if (not np.isfinite([self.follow_s, self.brake_s]).all()
                or self.follow_s < 0 or self.brake_s <= 0):
            raise ValueError('follow duration must be nonnegative and brake duration positive')
        if not np.isfinite(self.prealign_s) or not 0 <= self.prealign_s < self.duration_s:
            raise ValueError('prealignment must leave a positive approach duration')
        zero = np.zeros(3)
        approach = BPoly.from_derivatives([0., self.duration_s-self.prealign_s],
            [[self.start_tcp_world, zero, zero], [self.contact_xyz, self.velocity_world, zero]])
        end_follow = self.contact_xyz+self.velocity_world*self.follow_s
        braking = BPoly.from_derivatives([0., self.brake_s],
            [[end_follow, self.velocity_world, zero],
             [end_follow+self.velocity_world*self.brake_s/2, zero, zero]])
        orientations = Rotation.from_quat(np.array([
            self.start_quaternion_wxyz, self.target_quaternion_wxyz])[:, [1, 2, 3, 0]])
        object.__setattr__(self, '_approach', approach)
        object.__setattr__(self, '_braking', braking)
        object.__setattr__(self, '_orientations', Slerp([0., 1.], orientations))

    @property
    def follow_end_s(self):
        return self.duration_s+self.follow_s

    @property
    def full_duration_s(self):
        return self.follow_end_s+self.brake_s

    def position_derivatives(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        if t < self.prealign_s:
            return (self.start_tcp_world.copy(), np.zeros(3), np.zeros(3))
        if t <= self.duration_s:
            return tuple(self._approach(t-self.prealign_s, d) for d in (0, 1, 2))
        if t <= self.follow_end_s:
            return (self.contact_xyz+self.velocity_world*(t-self.duration_s),
                    self.velocity_world.copy(), np.zeros(3))
        return tuple(self._braking(t-self.follow_end_s, d) for d in (0, 1, 2))

    def sample(self, elapsed_s):
        t = float(np.clip(elapsed_s, 0., self.full_duration_s))
        if t < self.prealign_s:
            xyz = self.start_tcp_world.copy()
        elif t <= self.duration_s:
            xyz = self._approach(t-self.prealign_s, 0)
        elif t <= self.follow_end_s:
            xyz = self.contact_xyz+self.velocity_world*(t-self.duration_s)
        else:
            xyz = self._braking(t-self.follow_end_s, 0)
        rotation_end = self.prealign_s if self.prealign_s else self.duration_s
        if t >= rotation_end:
            return xyz, self.target_quaternion_wxyz.copy()
        u = float(np.clip(t/rotation_end, 0., 1.))
        phase = float(np.clip(10*u**3-15*u**4+6*u**5, 0., 1.))
        quat = self._orientations([phase]).as_quat()[0][[3, 0, 1, 2]]
        return xyz, quat

"""Separate a payload transport request from the robot's capability envelope."""
from dataclasses import dataclass
import numpy as np


def loaded_acceleration(raw, global_limits):
    key = 'loaded_joint_acceleration_limits_rad_s2'
    if key not in raw:
        return np.asarray(global_limits, float).copy(), 'inherit_global'
    requested = np.asarray(raw[key], float)
    source = raw.get('loaded_acceleration_limit_source')
    if (requested.shape != (7,) or not np.isfinite(requested).all() or
            np.any(requested <= 0) or np.any(requested > np.asarray(global_limits)+1e-10)):
        raise ValueError('载物加速度须为7个正有限数且不超过当前机械臂边界')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('载物运动参数必须说明来源')
    return requested.copy(), source


@dataclass(frozen=True)
class MotionProfile:
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    settings: object
    name: str
    source: str

    def diagnostics(self):
        return {'motion_profile': self.name, 'motion_profile_source': self.source,
                'joint_velocity_limits_rad_s': self.velocity_limits.tolist(),
                'joint_acceleration_limits_rad_s2': self.acceleration_limits.tolist()}


def select_motion_profile(config, *, payload=False):
    velocity = np.asarray(config.velocity_limits, float).copy()
    acceleration, source = (loaded_acceleration(config.raw, config.acceleration_limits) if payload else
                            (np.asarray(config.acceleration_limits, float).copy(), 'global'))
    velocity.setflags(write=False)
    acceleration.setflags(write=False)
    return MotionProfile(velocity, acceleration, config.settings,
                         'loaded' if payload else 'unloaded', source)

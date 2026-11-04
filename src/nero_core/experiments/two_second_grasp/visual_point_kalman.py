"""Independent simulation-only Kalman estimator for a visual position stream."""
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.kalman_motion import KalmanMotion


class VisualPointKalman:
    def __init__(self, max_velocity):
        self.max_velocity = float(max_velocity)
        self.tracks = {}

    def update(self, identity, stamp, position):
        identity = tuple(identity)
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(stamp):
            return False
        if identity not in self.tracks:
            self.tracks[identity] = KalmanMotion(point[:2], stamp, belt_direction=(1., 0.),
                                                max_velocity=self.max_velocity)
            return True
        return self.tracks[identity].update(point[:2], stamp)

    def estimate(self, identity):
        motion = self.tracks.get(tuple(identity))
        if motion is None or not motion.speed_valid:
            return None
        return {'velocity_world_m_s': np.r_[motion.state[2:4], 0.],
                'last_stamp_s': motion.observation_stamp,
                'sample_count': len(motion.accepted_stamps),
                'span_s': motion.accepted_stamps[-1]-motion.accepted_stamps[0],
                'velocity_std_m_s': float(np.sqrt(np.linalg.eigvalsh(motion.covariance[2:, 2:])[-1])),
                'accel_noise_std': motion.accel_noise_std}

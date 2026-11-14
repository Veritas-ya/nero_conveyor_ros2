"""Experimental Cartesian reference: continuous joining, then co-moving grasp.

The caller must separately solve and certify IK, time feasibility,
moving-target geometry and full execution.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation, Slerp

from .reference import Intercept, EntryNotReady
from .smooth_reference import SmoothIntercept


def freeze_window_for_speed(config, speed):
    """Bound the upstream freeze window using existing durations and speed.

    A whole maximum observation age is reserved between capture and planning
    submission. This is only an admission estimate: actual grasp geometry,
    reachability, trajectory certification and deadlines remain downstream.
    """
    raw = config.raw
    position = raw.get('freeze_position_upstream_m')
    if position is not None:
        position = float(position)
        tolerance = float(raw.get('freeze_position_tolerance_m', .02))
        if (not np.isfinite([speed, position, tolerance]).all() or speed <= 0
                or not 0 < tolerance < position):
            raise ValueError('invalid fixed freeze position')
        return {'distance_m': position, 'minimum_distance_m': position-tolerance,
                'required_m': position, 'capped': False, 'policy': 'fixed_capture_position',
                'capture_tolerance_m': tolerance}
    minimum = float(raw['freeze_upstream_distance_m'])
    if not raw.get('synchronized_adaptive_freeze', False):
        return {'distance_m': minimum, 'required_m': minimum, 'capped': False,
                'policy': 'fixed'}
    speed = float(speed)
    bounds = np.asarray(config.follow_bounds, float)
    maximum = float(raw['synchronized_max_freeze_upstream_distance_m'])
    if raw.get('planning_reserve_mode', 'fixed') == 'measured':
        if not np.isfinite([speed, minimum, maximum]).all() or speed <= 0 or maximum < minimum:
            raise ValueError('invalid adaptive freeze window')
        return {'distance_m': maximum, 'required_m': maximum, 'capped': False,
                'policy': 'earliest_configured_upstream_for_measured_planning'}
    horizon = (raw['maximum_observation_age_s'] + raw['planning_reserve_s'] +
        raw['command_advance_s'] + raw['prealign_orientation_s'] + raw.get('rotate_first_s',0.) +
        raw['synchronized_duration_s'] + config.settings.full_plan_close_window_s)
    if (not np.isfinite(np.r_[speed, bounds, minimum, maximum, horizon]).all()
            or speed <= 0 or bounds.shape != (2,) or bounds[0] >= bounds[1]
            or minimum <= 0 or maximum < minimum or horizon <= 0):
        raise ValueError('invalid adaptive freeze window')
    required = max(minimum, speed*horizon-(bounds[1]-bounds[0]))
    return {'distance_m': min(maximum, required), 'required_m': required,
            'capped': required > maximum, 'policy': 'speed_and_existing_time_budget',
            'budget_horizon_s': horizon}


def admit_synchronized_observation(capture_s, coordinate, speed, follow_bounds,
                                   now_s, maximum_upstream_distance_m, *, minimum_upstream_distance_m=None):
    """Admit a fresh track for exact scheduling; this is not a trajectory check."""
    bounds = np.asarray(follow_bounds, float)
    if (bounds.shape != (2,) or not np.isfinite(np.r_[capture_s, coordinate, speed,
            bounds, now_s, maximum_upstream_distance_m]).all() or speed <= 0
            or bounds[0] >= bounds[1] or capture_s > now_s
            or maximum_upstream_distance_m <= 0):
        raise ValueError('invalid synchronized observation')
    if coordinate < bounds[0]-maximum_upstream_distance_m:
        raise EntryNotReady('observation is too far upstream to freeze', 'too_far_upstream')
    if minimum_upstream_distance_m is not None:
        minimum = float(minimum_upstream_distance_m)
        if not np.isfinite(minimum) or not 0 <= minimum <= maximum_upstream_distance_m:
            raise ValueError('invalid minimum upstream freeze distance')
        # This bound is applied to the ORIGINAL exposure position, not a
        # projected/retimestamped point at the end of geometry processing.
        if coordinate > bounds[0]-minimum+1e-9:
            raise EntryNotReady('exposure has passed the fixed freeze position', 'past_fixed_freeze_position')
    if coordinate+speed*(now_s-capture_s) > bounds[1]:
        raise EntryNotReady('target has passed the follow region', 'past_follow_region')
    return float(capture_s+(bounds[0]-coordinate)/speed)


def schedule_forward_join(*, capture_s, grasp_position, velocity, standby_position,
                          direction, now_s, planning_reserve_s, advance_s,
                          join_s, synchronized_s, follow_s, follow_bounds, rotate_first_s=0.,
                          earliest_dispatch_s=None):
    """Find a same-direction join and closure window from the frozen target.

    Returns host/simulation clock times only. This does not prove reachability,
    freshness, actuator limits, collision freedom, or accuracy of prediction.
    """
    grasp = np.asarray(grasp_position, float)
    velocity = np.asarray(velocity, float)
    standby = np.asarray(standby_position, float)
    axis = np.asarray(direction, float)
    bounds = np.asarray(follow_bounds, float)
    if (grasp.shape != (3,) or velocity.shape != (3,) or standby.shape != (3,)
            or axis.shape != (3,) or bounds.shape != (2,)
            or not np.isfinite(np.r_[grasp, velocity, standby, axis, bounds,
                capture_s, now_s, planning_reserve_s, advance_s, join_s,
                synchronized_s, follow_s, rotate_first_s]).all() or np.linalg.norm(axis) < 1e-9
            or join_s <= 0 or synchronized_s < 0 or follow_s < 0
            or planning_reserve_s < 0 or advance_s < 0 or rotate_first_s < 0 or bounds[0] >= bounds[1]
            or capture_s > now_s):
        raise ValueError('invalid synchronized trajectory schedule')
    axis = axis/np.linalg.norm(axis)
    speed = float(velocity@axis)
    if speed <= 0:
        raise ValueError('target speed must point along belt direction')
    # With zero starting velocity and terminal speed v, this displacement
    # yields a monotonic quintic join in the belt direction.
    minimum_start = max(float(standby@axis)+speed*join_s/2, float(bounds[0]))
    earliest_join = now_s+planning_reserve_s+advance_s
    if earliest_dispatch_s is not None:
        if not np.isfinite(earliest_dispatch_s):
            raise ValueError('invalid earliest dispatch time')
        earliest_join = max(earliest_join, float(earliest_dispatch_s)+advance_s)
    earliest_sync = max(capture_s, earliest_join+rotate_first_s+join_s)
    sync_s = max(earliest_sync, capture_s+(minimum_start-float(grasp@axis))/speed)
    sync_coordinate = float(grasp@axis)+speed*(sync_s-capture_s)
    contact_coordinate = sync_coordinate+speed*synchronized_s
    follow_end_coordinate = contact_coordinate+speed*follow_s
    if sync_coordinate < bounds[0]-1e-9 or follow_end_coordinate > bounds[1]+1e-9:
        raise ValueError('synchronized segment and follow do not fit grasp interval')
    return {'join_start_s':sync_s-join_s-rotate_first_s, 'sync_start_s':sync_s,
            'translation_start_s':sync_s-join_s, 'rotate_first_s':float(rotate_first_s),
            'contact_s':sync_s+synchronized_s, 'sync_start_coordinate_m':sync_coordinate,
            'contact_coordinate_m':contact_coordinate, 'follow_end_coordinate_m':follow_end_coordinate,
            'speed_along_belt_m_s':speed}


@dataclass(frozen=True)
class SynchronizedIntercept(SmoothIntercept):
    """A join from rest followed by a same-velocity contact segment.

    ``rotate_first_s`` optionally holds TCP position while rotating. The join
    occupies the remainder of ``prealign_s``; ``duration_s - prealign_s`` is
    the synchronized duration. Contact occurs at its end. The default zero
    rotation lead retains the original concurrent translation and rotation.
    """

    rotate_first_s: float = 0.

    def __post_init__(self):
        Intercept.__post_init__(self)
        if (not np.isfinite([self.follow_s, self.brake_s, self.prealign_s]).all()
                or self.follow_s < 0 or min(self.brake_s, self.prealign_s) <= 0
                or self.prealign_s > self.duration_s or not np.isfinite(self.rotate_first_s)
                or not 0 <= self.rotate_first_s < self.prealign_s):
            raise ValueError('joining and brake must be positive; follow and synchronized durations cannot be negative')
        zero = np.zeros(3)
        sync_duration = self.duration_s-self.prealign_s
        sync_start = self.contact_xyz-self.velocity_world*sync_duration
        join = BPoly.from_derivatives([0., self.prealign_s-self.rotate_first_s],
            [[self.start_tcp_world, zero, zero], [sync_start, self.velocity_world, zero]])
        end_follow = self.contact_xyz+self.velocity_world*self.follow_s
        braking = BPoly.from_derivatives([0., self.brake_s],
            [[end_follow, self.velocity_world, zero],
             [end_follow+self.velocity_world*self.brake_s/2, zero, zero]])
        orientations = Rotation.from_quat(np.array([
            self.start_quaternion_wxyz, self.target_quaternion_wxyz])[:, [1, 2, 3, 0]])
        object.__setattr__(self, '_sync_start', sync_start)
        object.__setattr__(self, '_join', join)
        object.__setattr__(self, '_braking', braking)
        object.__setattr__(self, '_orientations', Slerp([0., 1.], orientations))

    def position_derivatives(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        if t < self.rotate_first_s:
            return self.start_tcp_world.copy(), np.zeros(3), np.zeros(3)
        if t < self.prealign_s:
            return tuple(self._join(t-self.rotate_first_s, order) for order in (0, 1, 2))
        if t <= self.follow_end_s:
            return (self._sync_start+self.velocity_world*(t-self.prealign_s),
                    self.velocity_world.copy(), np.zeros(3))
        return tuple(self._braking(t-self.follow_end_s, order) for order in (0, 1, 2))

    def sample(self, elapsed_s):
        t = float(np.clip(elapsed_s, 0., self.full_duration_s))
        xyz = self.position_derivatives(t)[0]
        orientation_duration = self.rotate_first_s or self.prealign_s
        if t >= orientation_duration:
            return xyz, self.target_quaternion_wxyz.copy()
        u = t/orientation_duration
        phase = 10*u**3-15*u**4+6*u**5
        return xyz, self._orientations([phase]).as_quat()[0][[3, 0, 1, 2]]

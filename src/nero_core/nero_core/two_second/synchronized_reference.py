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


def freeze_window_for_speed(config, speed, *, surface=None, belt_region=None):
    """Resolve the shared exposure admission policy for vision and planning.

    Detector-footprint mode uses live or frozen measured belt geometry.
    The legacy duration policy reserves a maximum observation age between capture and planning
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
    if raw.get('tracked_freeze_range') == 'detector_footprint':
        from .tracked_freeze import detector_freeze_window, native_exposure_window
        window=detector_freeze_window(config, surface, belt_region=belt_region)
        if native_exposure_window(config):
            window['minimum_distance_m']=raw['native_single_min_capture_upstream_m']
        return window
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


def native_observation_entry(capture_s, coordinate, speed, bounds, now, max_age):
    """Predict entry from an accepted native observation for the normal intercept solver."""
    lo,hi=bounds
    if not np.isfinite([capture_s,coordinate,speed,lo,hi,now,max_age]).all() or speed<=0 or lo>=hi:
        raise ValueError('invalid native observation for scheduling')
    if not 0 <= now-capture_s <= max_age:
        raise EntryNotReady('native exposure outside allowed age', 'observation_stale')
    if coordinate+speed*(now-capture_s)>hi:
        raise EntryNotReady('target has passed the follow region', 'past_follow_region')
    return float(capture_s+(lo-coordinate)/speed)


def schedule_forward_join(*, capture_s, grasp_position, velocity, standby_position,
                          direction, now_s, planning_reserve_s, advance_s,
                          join_s, synchronized_s, follow_s, follow_bounds, rotate_first_s=0.,
                          earliest_dispatch_s=None, from_bin=False, forward_ramp_s=None):
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
    minimum_start = float(bounds[0])
    if not from_bin:
        ramp = join_s if forward_ramp_s is None else min(join_s, forward_ramp_s)
        if not np.isfinite(ramp) or ramp <= 0:
            raise ValueError('invalid high-approach velocity ramp')
        minimum_start = max(float(standby@axis)+speed*ramp/2, minimum_start)
    earliest_join = now_s+planning_reserve_s+advance_s
    if earliest_dispatch_s is not None:
        if not np.isfinite(earliest_dispatch_s):
            raise ValueError('invalid earliest dispatch time')
        earliest_join = max(earliest_join, float(earliest_dispatch_s)+advance_s)
    earliest_sync = max(capture_s, earliest_join+rotate_first_s+join_s)
    sync_s = max(earliest_sync, capture_s+(minimum_start-float(grasp@axis))/speed)
    if from_bin:
        sync_s += 4*np.spacing(max(1., abs(sync_s)))  # Round into the admissible interval.
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
    bin_descent_s: float = 0.
    bin_clearance_m: float = 0.
    brake_rise_m: float = 0.
    height_lead_s: float = 0.
    height_normal_world: tuple = (0.,0.,1.)
    relative_descent_s: float = 0.
    relative_sync_s: float = 0.
    approach_clearance_m: float = 0.
    object_axes_world: tuple = ()

    @property
    def relative_alignment_s(self):
        return min(self.prealign_s, self.duration_s-self.height_lead_s)-self.relative_descent_s-self.relative_sync_s

    @property
    def approach_knots(self):
        return tuple(self._join.x + self.rotate_first_s)

    def __post_init__(self):
        Intercept.__post_init__(self)
        if (not np.isfinite([self.follow_s, self.brake_s, self.prealign_s]).all()
                or self.follow_s < 0 or min(self.brake_s, self.prealign_s) <= 0
                or self.prealign_s > self.duration_s or not np.isfinite(self.rotate_first_s)
                or not 0 <= self.rotate_first_s < self.prealign_s):
            raise ValueError('joining and brake must be positive; follow and synchronized durations cannot be negative')
        zero = np.zeros(3)
        sync_duration = self.duration_s-self.prealign_s
        tool_velocity = self.velocity_world.copy()
        guarded = self.brake_rise_m > 0
        if not np.isfinite([self.brake_rise_m,self.height_lead_s]).all() or self.brake_rise_m < 0:
            raise ValueError('invalid brake height profile')
        if guarded:
            # The frozen object still follows the measured plane. Only the
            # tool's vertical derivative changes, preserving the contact pose.
            n = np.asarray(self.height_normal_world,float)
            if n.shape!=(3,) or not np.isfinite(n).all() or n[2]<=0:
                raise ValueError('invalid height normal')
            gradient = -n[:2]/n[2]
            bias = 2*self.brake_rise_m/self.brake_s
            if self.relative_descent_s:
                n = n/np.linalg.norm(n)
                tool_velocity += n*max(bias-self.velocity_world@n,(bias-self.velocity_world[2])/n[2])
            else:
                tool_velocity[2] = max(0., gradient@tool_velocity[:2]) + bias
        sync_start = self.contact_xyz-tool_velocity*sync_duration
        if self.relative_descent_s:
            from .object_approach import join_curve
            join = join_curve(self, tool_velocity)
            object.__setattr__(self, 'object_axes_world', tuple(map(tuple,self.object_axes_world)))
        elif self.bin_descent_s:
            if (not 0 < self.bin_descent_s < self.prealign_s-self.rotate_first_s
                    or not np.isfinite(self.bin_clearance_m) or self.bin_clearance_m <= 0):
                raise ValueError('invalid bin approach descent')
            above = sync_start - self.velocity_world*self.bin_descent_s + [0., 0., self.bin_clearance_m]
            join = BPoly.from_derivatives([0., self.prealign_s-self.rotate_first_s-self.bin_descent_s,
                self.prealign_s-self.rotate_first_s], [[self.start_tcp_world, zero, zero],
                [above, self.velocity_world, zero], [sync_start, self.velocity_world, zero]])
        else:
            join = BPoly.from_derivatives([0., self.prealign_s-self.rotate_first_s],
                [[self.start_tcp_world, zero, zero], [sync_start, self.velocity_world, zero]])
        if guarded and not self.relative_descent_s:
            level = min(self.prealign_s, self.duration_s-self.height_lead_s)-self.rotate_first_s
            if level <= 0 or self.height_lead_s <= 0:
                raise ValueError('insufficient joining interval for early vertical transition')
            # Keep the horizontal polynomial exactly. Split at the transition
            # and replace only Z, matching position, velocity and acceleration.
            end = self.prealign_s-self.rotate_first_s
            if self.bin_descent_s and level<=end-self.bin_descent_s:
                raise ValueError('bin descent too short for no-descent transition')
            end_jet = np.array([sync_start[2], tool_velocity[2], 0.])
            start_jet = end_jet
            if end-level > 1e-12:
                # Positive Bernstein basis: choosing each vertical speed
                # coefficient above both zero and the local belt-height rate
                # protects both heights, even while XY is still accelerating.
                tail = BPoly.from_derivatives([level,end],
                    [[join(t,d)[:2] for d in range(3)] for t in (level,end)])
                speed_coeff = np.maximum(tail.derivative().c[:,0]@gradient,0.)+bias
                span = end-level
                start_jet = [sync_start[2]-span*speed_coeff.mean(),speed_coeff[0],
                             (len(speed_coeff)-1)*np.diff(speed_coeff)[0]/span]
            knots = np.unique(np.r_[join.x,level])
            values = []
            for t in knots:
                jet = np.stack([join(t,d) for d in range(3)])
                if t == level:
                    jet[:,2] = start_jet
                if t == end:
                    jet[:,2] = end_jet
                values.append(jet)
            join = BPoly.from_derivatives(knots, values)
        end_follow = self.contact_xyz+tool_velocity*self.follow_s
        braking = BPoly.from_derivatives([0., self.brake_s],
            [[end_follow, tool_velocity, zero],
             [end_follow+tool_velocity*self.brake_s/2, zero, zero]])
        orientations = Rotation.from_quat(np.array([
            self.start_quaternion_wxyz, self.target_quaternion_wxyz])[:, [1, 2, 3, 0]])
        object.__setattr__(self, '_sync_start', sync_start)
        object.__setattr__(self, '_tool_velocity', tool_velocity)
        object.__setattr__(self, '_join', join)
        object.__setattr__(self, '_braking', braking)
        object.__setattr__(self, '_orientations', Slerp([0., 1.], orientations))

    def _position_terms(self, t, *, derivatives):
        if t < self.rotate_first_s:
            xyz = self.start_tcp_world.copy()
            return (xyz, np.zeros(3), np.zeros(3)) if derivatives else (xyz,)
        if t < self.prealign_s:
            curve, local = self._join, t-self.rotate_first_s
        elif t <= self.follow_end_s:
            xyz = self._sync_start+self._tool_velocity*(t-self.prealign_s)
            return (xyz, self._tool_velocity.copy(), np.zeros(3)) if derivatives else (xyz,)
        else:
            curve, local = self._braking, t-self.follow_end_s
        return (curve(local), curve(local, 1), curve(local, 2)) if derivatives else (curve(local),)

    def position_derivatives(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        return self._position_terms(t, derivatives=True)

    def sample(self, elapsed_s):
        t = float(np.clip(elapsed_s, 0., self.full_duration_s))
        xyz, = self._position_terms(t, derivatives=False)
        orientation_duration = self.rotate_first_s or (self.relative_alignment_s if self.relative_descent_s else self.prealign_s-self.bin_descent_s)
        if t >= orientation_duration:
            return xyz, self.target_quaternion_wxyz.copy()
        u = t/orientation_duration
        phase = 10*u**3-15*u**4+6*u**5
        return xyz, self._orientations([phase]).as_quat()[0][[3, 0, 1, 2]]

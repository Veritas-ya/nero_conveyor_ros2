"""Timed seven-axis curves with polynomial extrema checks and task identity."""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline, PPoly
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def geometry_check_times(duration, check_hz, *boundaries):
    """Same geometry grid in real and simulated planners, retaining phase boundaries."""
    if not np.isfinite(check_hz) or check_hz <= 0:
        raise ValueError('几何检查频率须为正有限数')
    return sampling_times(duration, 1./check_hz, *boundaries)


def sampling_times(duration, dt, *boundaries):
    """Keep exact phase boundaries without near-duplicate floating-point knots."""
    if not np.isfinite([duration,dt,*boundaries]).all() or min(duration,dt) <= 0:
        raise ValueError('采样持续时间和周期须为正有限数')
    exact = np.unique(np.r_[0.,boundaries,duration])
    if exact[0] < 0 or exact[-1] > duration:
        raise ValueError('轨迹分段时刻必须位于持续时间内')
    grid = np.arange(0.,duration,dt)
    tolerance = 64*np.finfo(float).eps*max(1.,duration)
    merged = [exact[0]]
    for t in exact[1:]:
        if t-merged[-1] > tolerance:
            merged.append(t)
        elif t == duration:
            merged[-1] = t
    exact = np.asarray(merged)
    for t in exact:
        grid = grid[abs(grid-t) > tolerance]
    return np.sort(np.r_[grid,exact])


def extrema(curve, derivative):
    polynomial = curve.derivative(derivative)
    # Include both limits at knots, also for a piecewise time parameterization.
    candidates = list(np.r_[polynomial.x,
                           np.nextafter(polynomial.x[1:], polynomial.x[:-1])])
    if polynomial.c.shape[0] > 1:
        for roots in polynomial.derivative().roots(extrapolate=False):
            candidates.extend(np.asarray(roots)[np.isfinite(roots)].tolist())
    values = polynomial(np.asarray(candidates))
    return values.min(axis=0), values.max(axis=0)


@dataclass(frozen=True)
class TimedPath:
    curve: object
    identity: tuple
    version: int
    stage: str
    duration: float
    planning_s: float
    diagnostics: dict = field(default_factory=dict)

    def sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.duration))
        return tuple(np.asarray(self.curve(t, d)).copy() for d in (0, 1, 2))

    sample_full = sample

    def position(self, elapsed):
        """Evaluate only q when geometry validation does not need qd/qdd."""
        t = float(np.clip(elapsed, 0., self.duration))
        return np.asarray(self.curve(t, 0)).copy()

    @property
    def q0(self):
        return self.sample(0.)[0]

    @property
    def q1(self):
        return self.sample(self.duration)[0]


def joint_curve(times, joints, *, stationary_ends=True, zero_end_acceleration=False):
    if zero_end_acceleration:
        if not stationary_ends or len(times) < 4:
            raise ValueError('静止q/v/a端点需要至少四个样本和静止端点')
        zero = np.zeros(7)
        spline = make_interp_spline(times, joints, k=5,
            bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
        axes = [PPoly.from_spline((spline.t, spline.c[:, i], spline.k)) for i in range(7)]
        mask = np.diff(axes[0].x) > 0
        return PPoly(np.stack([p.c[:, mask] for p in axes], axis=-1),
                     np.r_[axes[0].x[:-1][mask], axes[0].x[-1]], extrapolate=False)
    boundary = ((1, np.zeros(7)), (1, np.zeros(7))) if stationary_ends else 'not-a-knot'
    return CubicSpline(times, joints, bc_type=boundary, extrapolate=False)


def certify_curve(times, joints, config, lower, upper, *, identity, version, stage,
                  planning_s=0., stationary_ends=True, zero_end_acceleration=False):
    times, joints = np.asarray(times, float), np.asarray(joints, float)
    if times.ndim != 1 or len(times) < 2 or joints.shape != (len(times), 7) or not np.isfinite(joints).all() or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('轨迹需要递增秒数及 N×7 有限关节位置')
    curve = joint_curve(times, joints, stationary_ends=stationary_ends,
                        zero_end_acceleration=zero_end_acceleration)
    for derivative, bound in ((0, (np.asarray(lower), np.asarray(upper))),
                               (1, (-config.velocity_limits, config.velocity_limits)),
                               (2, (-config.acceleration_limits, config.acceleration_limits))):
        low, high = extrema(curve, derivative)
        if np.any(low < bound[0]-1e-9) or np.any(high > bound[1]+1e-9):
            exceeded = (low < bound[0]-1e-9) | (high > bound[1]+1e-9)
            raise PickRejected('整条曲线超出关节位置/速度/加速度约束', details={
                'stage': stage, 'derivative': derivative, 'min': low.tolist(), 'max': high.tolist(),
                'limit_lower': bound[0].tolist(), 'limit_upper': bound[1].tolist(),
                'exceeded_joint_numbers': (np.flatnonzero(exceeded)+1).tolist(),
                'trajectory_duration_s': float(times[-1]),
                'acceleration_only_rest_to_rest_duration_lower_bound_s': (
                    float(np.max(2*np.sqrt(abs(joints[-1]-joints[0])/config.acceleration_limits)))
                    if stationary_ends else None)})
    return TimedPath(curve, identity, version, stage, float(times[-1]), planning_s)


def retime_samples(times, joints, config, lower, upper, *, identity, version, stage):
    """Uniformly retime one rest-to-rest curve; keep its checked geometry."""
    times, joints = np.asarray(times), np.asarray(joints)
    curve = joint_curve(times, joints, zero_end_acceleration=True)
    velocity_ratio = float(np.max(abs(np.asarray(extrema(curve, 1)))/config.velocity_limits))
    acceleration_ratio = float(np.sqrt(np.max(abs(np.asarray(extrema(curve, 2)))/config.acceleration_limits)))
    scale = max(1., velocity_ratio, acceleration_ratio)*1.02
    period = 1./config.settings.control_hz
    scale = np.ceil(times[-1]*scale/period)*period/times[-1]
    result = certify_curve(times*scale, joints, config, lower, upper,
        identity=identity, version=version, stage=stage, zero_end_acceleration=True)
    return TimedPath(result.curve, identity, version, stage, result.duration, result.planning_s,
                     {'time_scale': float(scale), 'rest_qva_endpoints': True})


def merge_collinear_waypoints(path):
    """Drop only exact collinear RRT densification, preserving bends/reversals."""
    kept = []
    for value in path:
        q = np.asarray(value, float).copy()
        if kept and np.max(abs(q-kept[-1])) < 1e-12:
            continue
        while len(kept) >= 2:
            chord = q-kept[-2]
            length2 = float(chord@chord)
            if length2 < 1e-24:
                break
            u = float((kept[-1]-kept[-2])@chord/length2)
            if not 0. < u < 1. or np.max(abs(kept[-1]-(kept[-2]+u*chord))) > 1e-10:
                break
            kept.pop()
        kept.append(q)
    return kept

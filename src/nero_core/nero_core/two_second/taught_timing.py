"""Change speed along the existing taught polynomial without changing its path.

The squared path speed x=(ds/dt)^2 makes each joint acceleration linear in
adjacent x values: q''(s)*x + q'(s)*dx/ds/2. Backward reachable intervals and a
forward pass provide a time law. Its C2 cubic interpolation is composed with
the original joint polynomial, then the complete resulting polynomials are
certified at their derivative extrema, not merely at the optimization grid.
"""
from dataclasses import replace
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import linprog

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .trajectory import TimedPath, extrema, joint_curve, retime_samples


class _CannotRetime(ValueError):
    pass


def limit_jerk(path, limit, control_hz):
    if limit is None:
        return path
    peak = float(np.max(abs(np.asarray(extrema(path.curve, 3)))))
    scale = max(1., np.cbrt(peak/limit)*1.02)
    duration = np.ceil(path.duration*scale*control_hz-1e-9)/control_hz
    scale = duration/path.duration
    coefficients = path.curve.c.copy()
    for i in range(len(coefficients)):
        coefficients[i] /= scale**(len(coefficients)-1-i)
    curve = PPoly(coefficients, path.curve.x*scale, extrapolate=False)
    return replace(path, curve=curve, duration=duration, diagnostics={**path.diagnostics,
        'joint_jerk_limit_rad_s3': limit, 'peak_joint_jerk_rad_s3': peak/scale**3,
        'jerk_time_scale': scale})


def compose_path(path, time_law):
    """Each time-law interval stays within one original polynomial interval."""
    blocks = []
    for i, source_start in enumerate(time_law(time_law.x[:-1])):
        # Cubic interpolation can round an exact knot a few ulps down.
        tolerance = 64*np.finfo(float).eps*max(1., abs(source_start))
        j = min(len(path.x)-2, np.searchsorted(path.x, source_start+tolerance, side='right')-1)
        warp = time_law.c[::-1, i].copy()
        warp[0] -= path.x[j]
        c = path.c[0, j][None, :].copy()
        for coefficient in path.c[1:, j]:
            product = np.zeros((len(c)+len(warp)-1, c.shape[1]))
            for k, value in enumerate(warp):
                product[k:k+len(c)] += c*value
            product[0] += coefficient
            c = product
        blocks.append(c[::-1])
    return PPoly(np.stack(blocks, axis=1), time_law.x, extrapolate=False)


def _time_law(path, motion, cancelled):
    knots = np.unique(path.x)
    uniform = np.linspace(knots[0], knots[-1], 257)
    tolerance = 64*np.finfo(float).eps*max(1., knots[-1])
    distances = np.min(abs(uniform[:, None]-knots[None, :]), axis=1)
    grid = np.sort(np.r_[knots, uniform[distances > tolerance]])
    count = len(grid)-1
    constraints = []
    # A finite path-rate ceiling also bounds constant/near-stationary pieces.
    squared_rate_cap = 100.
    for left, right in zip(grid[:-1], grid[1:]):
        if cancelled():
            raise PickRejected('示教轨迹时间参数化已取消')
        h = right-left
        rows, bounds = [], []
        for weight in (0., .5, 1.):
            source = left+weight*h
            v, a = path(source, 1), path(source, 2)
            coefficients = np.column_stack((a*(1-weight)-v/(2*h), a*weight+v/(2*h)))
            rows.extend(coefficients); bounds.extend(motion.acceleration_limits)
            rows.extend(-coefficients); bounds.extend(motion.acceleration_limits)
            cap = min(squared_rate_cap, float(np.min(
                (motion.velocity_limits/np.maximum(abs(v), 1e-9))**2)))
            rows.append([1-weight, weight]); bounds.append(cap)
        constraints.append((np.asarray(rows), np.asarray(bounds)))
    controllable = np.zeros(count+1)
    for i in range(count-1, -1, -1):
        if cancelled():
            raise PickRejected('示教轨迹时间参数化已取消')
        a, b = constraints[i]
        result = linprog([-1., 0.], A_ub=a, b_ub=b,
            bounds=[(0., squared_rate_cap), (0., controllable[i+1])], method='highs')
        if not result.success:
            raise _CannotRetime('不可行的路径速度区间')
        controllable[i] = result.x[0]
    squared = np.zeros(count+1)
    for i in range(count):
        if cancelled():
            raise PickRejected('示教轨迹时间参数化已取消')
        a, b = constraints[i]
        remainder, coefficient = b-a[:, 0]*squared[i], a[:, 1]
        positive, negative = coefficient > 1e-12, coefficient < -1e-12
        upper = min(controllable[i+1], float(np.min(remainder[positive]/coefficient[positive]))
                    if positive.any() else squared_rate_cap)
        lower = max(0., float(np.max(remainder[negative]/coefficient[negative])) if negative.any() else 0.)
        if lower > upper+1e-7:
            raise _CannotRetime('不一致的路径速度区间')
        squared[i+1] = max(0., upper*(1-1e-10))
    speed = np.sqrt(squared)
    sums = speed[:-1]+speed[1:]
    if np.any(sums <= 1e-12):
        raise _CannotRetime('路径速度不能前进')
    times = np.r_[0., np.cumsum(2*np.diff(grid)/sums)]
    law = CubicSpline(times, grid, bc_type=((1, 0.), (1, 0.)))
    velocity = law.derivative()
    roots = velocity.derivative().roots(extrapolate=False)
    minimum = float(np.min(velocity(np.r_[times, roots[np.isfinite(roots)]])))
    if minimum < -1e-10:
        raise _CannotRetime('平滑时间映射不单调')
    return law


def retime_taught_path(times, joints, motion, lower, upper, *, identity, version, stage,
                      cancelled=lambda: False, jerk_limit=None):
    baseline = retime_samples(times, joints, motion, lower, upper,
        identity=identity, version=version, stage=stage)
    baseline = limit_jerk(baseline, jerk_limit, motion.settings.control_hz)
    if cancelled():
        raise PickRejected('示教轨迹时间参数化已取消')
    path = joint_curve(times, joints, zero_end_acceleration=True)
    try:
        law = _time_law(path, motion, cancelled)
    except _CannotRetime as error:
        return replace(baseline, diagnostics={**baseline.diagnostics,
            'timing_policy': 'recorded_relative_time_uniformly_retimed_rest_to_rest',
            'variable_speed_fallback': str(error)})
    curve = compose_path(path, law)
    velocity_ratio = float(np.max(abs(np.asarray(extrema(curve, 1)))/motion.velocity_limits))
    acceleration_ratio = float(np.sqrt(np.max(abs(np.asarray(extrema(curve, 2)))/motion.acceleration_limits)))
    jerk_ratio = (0. if jerk_limit is None else
                  float(np.cbrt(np.max(abs(np.asarray(extrema(curve, 3))))/jerk_limit)))
    scale = max(1., velocity_ratio, acceleration_ratio, jerk_ratio)*1.02
    duration = np.ceil(law.x[-1]*scale*motion.settings.control_hz)/motion.settings.control_hz
    scale = duration/law.x[-1]
    coefficients = curve.c.copy()
    for i in range(len(coefficients)):
        coefficients[i] /= scale**(len(coefficients)-1-i)
    curve = PPoly(coefficients, curve.x*scale, extrapolate=False)
    for derivative, limits in ((0, (np.asarray(lower), np.asarray(upper))),
                               (1, (-motion.velocity_limits, motion.velocity_limits)),
                               (2, (-motion.acceleration_limits, motion.acceleration_limits))):
        lo, hi = extrema(curve, derivative)
        if np.any(lo < limits[0]-1e-9) or np.any(hi > limits[1]+1e-9):
            raise PickRejected('示教变速曲线超出关节约束', details={'derivative': derivative,
                'min': lo.tolist(), 'max': hi.tolist(), 'stage': stage})
    if cancelled():
        raise PickRejected('示教轨迹时间参数化已取消')
    if duration >= baseline.duration:
        return replace(baseline, diagnostics={**baseline.diagnostics,
            'timing_policy': 'recorded_relative_time_uniformly_retimed_rest_to_rest',
            'variable_speed_fallback': '原时间参数化更快'})
    return TimedPath(curve, identity, version, stage, duration, 0., {
        'timing_policy': 'path_preserving_variable_speed_c2',
        'rest_qva_endpoints': True, 'continuous_qva': True,
        'original_path_preserved': True, 'time_law_monotone': True,
        'uniform_duration_s': baseline.duration, 'speedup_over_uniform': baseline.duration/duration,
        'time_law_pieces': len(law.x)-1, 'certification': 'complete_polynomial_derivative_extrema',
        'joint_jerk_limit_rad_s3': jerk_limit,
        'peak_joint_jerk_rad_s3': float(np.max(abs(np.asarray(extrema(curve, 3)))))})

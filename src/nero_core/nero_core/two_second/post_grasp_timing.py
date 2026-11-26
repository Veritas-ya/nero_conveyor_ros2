"""Shorter carry/return timing with the same joint path and bounded jerk.

The small optimization only proposes a monotone C2 clock. Its composition
with the ORIGINAL polynomial is checked at complete derivative extrema and
then by the normal scene checker. No resampled joint interpolation is used.
"""
from dataclasses import replace
from math import factorial
import time
import warnings

import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq, minimize

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .motion_profile import select_motion_profile
from .taught_timing import compose_path
from .trajectory import extrema


def _peaks(curve, derivative):
    return np.max(abs(np.asarray(extrema(curve, derivative))), axis=0)


def _scale_curve(curve, duration):
    """Rescale coefficients and keep the advertised endpoint inside the domain."""
    scale = duration/curve.x[-1]
    coefficients = curve.c.copy()
    for i in range(len(coefficients)):
        coefficients[i] /= scale**(len(coefficients)-1-i)
    knots = curve.x*scale
    # x*(duration/x) can round one ULP below duration. With extrapolate=False
    # sampling that advertised endpoint then returns seven NaNs.
    knots[-1] = duration
    return PPoly(coefficients, knots, extrapolate=False)


def _inverse(law, value):
    if abs(value-float(law(law.x[0]))) < 1e-13:
        return float(law.x[0])
    if abs(value-float(law(law.x[-1]))) < 1e-13:
        return float(law.x[-1])
    return float(brentq(lambda t: law(t)-value, law.x[0], law.x[-1], xtol=1e-14))


def _split_law(path, law):
    # compose_path requires each clock piece to remain within one source
    # polynomial. Splitting preserves the cubic clock exactly, including C2.
    times = list(law.x)
    values = law(law.x)
    tolerance = 64*np.finfo(float).eps*max(1., path.x[-1])
    for value in path.x[1:-1]:
        if np.min(abs(values-value)) > tolerance:
            times.append(_inverse(law, value))
    times = np.unique(times)
    coefficients = np.array([law(times[:-1], d)/factorial(d) for d in (3, 2, 1, 0)])
    return PPoly(coefficients, times, extrapolate=False)


def _variable_clock(path, velocity, acceleration, jerk, cancelled):
    count = 6
    duration = path.x[-1]
    source = np.linspace(0., duration, count+1)
    derivatives = [path.derivative(d) for d in (1, 2, 3)]

    def constraints(intervals):
        if cancelled():
            raise PickRejected('搬运时间优化已取消')
        times = np.r_[0., np.cumsum(intervals)]
        law = CubicSpline(times, source, bc_type='natural')
        t = (times[:-1, None]+intervals[:, None]*np.linspace(0., 1., 65)).ravel()
        s, r, a, j = [law(t, d) for d in (0, 1, 2, 3)]
        q1, q2, q3 = [f(np.clip(s, 0., duration)) for f in derivatives]
        v = q1*r[:, None]
        acc = q2*r[:, None]**2+q1*a[:, None]
        change = q3*r[:, None]**3+3*q2*r[:, None]*a[:, None]+q1*j[:, None]
        return np.r_[1-np.max(abs(v), axis=0)/velocity,
                     1-np.max(abs(acc), axis=0)/acceleration,
                     1-np.max(abs(change), axis=0)/jerk, np.min(r)]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Values in x were outside bounds',
                                category=RuntimeWarning, module='scipy.optimize._slsqp_py')
        result = minimize(lambda d: np.sum(d), np.full(count, duration/count),
            method='SLSQP', bounds=[(.35*duration/count, 2*duration/count)]*count,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 50, 'ftol': 1e-6})
    # Even a grid-converged solution is not a certificate. An iteration-limited
    # proposal may be usable, but it receives the SAME exact checks below.
    law = CubicSpline(np.r_[0., np.cumsum(result.x)], source, bc_type='natural')
    rate = law.derivative()
    roots = rate.derivative().roots(extrapolate=False)
    if np.min(rate(np.r_[law.x, roots[np.isfinite(roots)]])) <= 0.:
        return None, None, {'optimizer_message': str(result.message)}
    return compose_path(path, _split_law(path, law)), law, {
        'optimizer_message': str(result.message), 'optimizer_iterations': int(result.nit)}


def _straight_return(path, velocity, acceleration, jerk):
    """Exact seven-phase jerk-limited timing, only for a monotone joint line."""
    q0, q1 = path(0.), path(path.x[-1])
    delta = q1-q0
    length2 = float(delta@delta)
    if length2 <= 1e-16:
        return None
    scalar_c = path.c@delta/length2
    scalar_c[-1] -= q0@delta/length2
    scalar = PPoly(scalar_c, path.x)
    residual = path.c-scalar_c[..., None]*delta
    residual[-1] -= q0
    if (np.max(_peaks(PPoly(residual, path.x), 0)) > 1e-9 or
            extrema(scalar, 1)[0] < -1e-9):
        return None
    moving = abs(delta) > 1e-10
    v = float(np.min(velocity[moving]/abs(delta[moving])))
    a = float(np.min(acceleration[moving]/abs(delta[moving])))
    j = float(np.min(jerk[moving]/abs(delta[moving])))
    tj = min(a/j, np.sqrt(v/j))
    ta = max(0., v/(j*tj)-tj)
    distance = j*tj*(ta+tj)*(ta+2*tj)
    if distance < 1.:
        cruise = (1.-distance)/v
    else:
        cruise = 0.
        tj = min(a/j, np.cbrt(1./(2*j)))
        ta = max(0., (-3*tj+np.sqrt(tj*tj+4./(j*tj)))/2.)
    coefficients, times = [], [0.]
    position = speed = accel = 0.
    for interval, change in zip((tj, ta, tj, cruise, tj, ta, tj), (j, 0., -j, 0., -j, 0., j)):
        if interval <= 1e-12:
            continue
        coefficients.append(np.array([delta*change/6., delta*accel/2.,
                                      delta*speed, q0+delta*position]))
        position += speed*interval+accel*interval**2/2.+change*interval**3/6.
        speed += accel*interval+change*interval**2/2.
        accel += change*interval
        times.append(times[-1]+interval)
    return PPoly(np.stack(coefficients, axis=1), times, extrapolate=False)


def _remap_diagnostics(path, curve, time_at):
    """Keep old intermediate calculation details explicitly in their old clock."""
    old = path.diagnostics
    out = {**old, 'source_timing_diagnostics': old}
    for key in ('route_waypoint_times_s', 'protected_waypoint_times_s'):
        if key in old:
            out[key] = [time_at(t) for t in old[key]]
    if 'pieces' in old:
        out['pieces'] = [{**p, 'start_s': time_at(p['start_s']), 'end_s': time_at(p['end_s']),
                         'diagnostics_time_base': 'before_post_grasp_retiming'} for p in old['pieces']]
    for key in ('corner_blend', 'continuous_carry', 'drop_orientation_schedule'):
        if key not in old:
            continue
        report = dict(old[key])
        for stamp in ('lift_end_s', 'terminal_start_s'):
            if stamp in report:
                report[stamp] = time_at(report[stamp])
        if 'duration_s' in report:
            report['duration_s'] = float(curve.x[-1])
        if 'joins' in report:
            report['joins'] = [{**join, 'time_s': time_at(join['time_s']),
                'max_joint_speed_rad_s': float(np.max(abs(curve(time_at(join['time_s']), 1))))}
                for join in report['joins']]
        for guides in ('rounded_guides', 'bin_approach_rounding'):
            if guides in report:
                report[guides] = [{k: time_at(v) for k, v in guide.items()} for guide in report[guides]]
        if key == 'corner_blend':
            report.update(peak_joint_velocity_rad_s=_peaks(curve, 1).tolist(),
                peak_joint_acceleration_rad_s2=_peaks(curve, 2).tolist(),
                peak_joint_jerk_rad_s3=float(np.max(_peaks(curve, 3))),
                time_scale=None, timing_policy='nonuniform_post_grasp_clock')
        out[key] = report
    return out


def accelerate_path(planner, path, width, *, payload, cancelled):
    """Return a faster fully checked path, or retain the existing checked one."""
    started = time.monotonic()
    if cancelled():
        raise PickRejected('搬运时间优化已取消')
    motion = select_motion_profile(planner.config, payload=payload)
    original_jerk = _peaks(path.curve, 3)
    configured = planner.config.raw.get('taught_joint_jerk_limit_rad_s3')
    jerk = np.minimum(original_jerk, float(configured)) if configured is not None else original_jerk
    info = {'applied': False, 'original_duration_s': path.duration,
            'original_peak_joint_jerk_rad_s3': original_jerk.tolist(),
            'jerk_ceiling_rad_s3': jerk.tolist(),
            'path_preserved': True, 'derivative_certification': 'complete_polynomial_extrema'}
    def unchanged(reason):
        wall = time.monotonic()-started
        return replace(path, planning_s=path.planning_s+wall,
            diagnostics={**path.diagnostics, 'post_grasp_timing': {
            **info, 'reason': reason, 'wall_s': wall}})
    if np.max(jerk) <= 1e-9:
        return unchanged('stationary_path')
    # Retain the existing 2% timing margin. No joint's peak jerk may exceed
    # its source value; a different joint must not consume unused headroom.
    velocity = motion.velocity_limits/1.02
    acceleration = motion.acceleration_limits/1.02**2
    jerk = np.maximum(jerk, 1e-9)/1.02**3
    law = None
    curve = None if payload else _straight_return(path.curve, velocity, acceleration, jerk)
    if curve is not None:
        info['method'] = 'straight_joint_path_bounded_jerk'
    else:
        curve, law, search = _variable_clock(path.curve, velocity, acceleration, jerk, cancelled)
        info.update(method='path_preserving_c2_clock', **search)
    if cancelled():
        raise PickRejected('搬运时间优化已取消')
    if curve is None:
        return unchanged('clock_not_monotone')
    # Grid checks only guide the search. Account for peaks BETWEEN samples.
    scale = max(1., float(np.max(_peaks(curve, 1)/velocity)),
                float(np.sqrt(np.max(_peaks(curve, 2)/acceleration))),
                float(np.cbrt(np.max(_peaks(curve, 3)/jerk))))
    duration = np.ceil(curve.x[-1]*scale*motion.settings.control_hz-1e-9)/motion.settings.control_hz
    if duration >= path.duration-1e-9:
        return unchanged('original_timing_faster')
    scale = duration/curve.x[-1]
    curve = _scale_curve(curve, duration)
    if law is None:
        # Monotone joint-line progress supplies the exact clock for metadata.
        delta = path.q1-path.q0
        axis = int(np.argmax(abs(delta)))
        time_at = lambda t: float(brentq(lambda x: curve(x)[axis]-path.curve(t)[axis],
            0., duration, xtol=1e-14)) if 0. < t < path.duration else (0. if t <= 0. else duration)
    else:
        time_at = lambda t: _inverse(law, t)*scale
        clock_knots = law.x*scale
        clock_knots[-1] = duration
        info['source_clock'] = {'knots_s': clock_knots.tolist(),
            'coefficients': (law.c/scale**np.arange(3, -1, -1)[:, None]).tolist()}
    diagnostics = _remap_diagnostics(path, curve, time_at)
    candidate = replace(path, curve=curve, duration=duration, diagnostics=diagnostics)
    # Explicit full derivative and endpoint checks also cover the new clock
    # implementation; scene checking below uses current payload/identity.
    for d, bounds in ((0, (planner.model.command_lower, planner.model.command_upper)),
                      (1, (-motion.velocity_limits, motion.velocity_limits)),
                      (2, (-motion.acceleration_limits, motion.acceleration_limits)),
                      (3, (-np.asarray(info['jerk_ceiling_rad_s3']), np.asarray(info['jerk_ceiling_rad_s3'])))):
        lo, hi = extrema(curve, d)
        if not np.isfinite([lo, hi]).all() or np.any(lo < bounds[0]-1e-8) or np.any(hi > bounds[1]+1e-8):
            return unchanged('complete_derivative_check_failed')
    for t, original in ((0., path.q0), (duration, path.q1)):
        if (not np.isfinite([curve(t, d) for d in (0, 1, 2)]).all() or
                np.max(abs(curve(t)-original)) > 1e-8 or
                any(np.max(abs(curve(t, d))) > 1e-7 for d in (1, 2))):
            return unchanged('rest_endpoint_check_failed')
    try:
        planner.check_path(candidate, width, payload=payload, cancelled=cancelled)
    except PickRejected as error:
        if cancelled():
            raise
        return unchanged(str(error))
    if cancelled():
        raise PickRejected('搬运时间优化已取消')
    info.update(applied=True, duration_s=duration, saved_s=path.duration-duration,
        continuous_qva=True, geometry_rechecked=True,
        peak_joint_velocity_rad_s=_peaks(curve, 1).tolist(),
        peak_joint_acceleration_rad_s2=_peaks(curve, 2).tolist(),
        peak_joint_jerk_rad_s3=_peaks(curve, 3).tolist(), wall_s=time.monotonic()-started)
    return replace(candidate, planning_s=path.planning_s+info['wall_s'],
                   diagnostics={**diagnostics, 'post_grasp_timing': info})

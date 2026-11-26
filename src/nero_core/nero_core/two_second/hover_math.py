"""Numerics for the opt-in hover/descent experiment; no device interfaces."""
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.interpolate import BPoly, PPoly, make_interp_spline
from scipy.spatial.transform import Rotation, Slerp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .precise_ik import PrecisePoseSolver
from .synchronized_reference import SynchronizedIntercept
from .trajectory import TimedPath, sampling_times, extrema
from .phase_timing import require_reference, polynomial_range

def fit_curve(times, joints, start_va, end_va):
    """Quintic interpolation with actual, potentially moving q/v/a endpoints."""
    times, joints = np.asarray(times), np.asarray(joints)
    if (len(times) < 4 or joints.shape != (len(times), 7) or
            not np.isfinite(times).all() or not np.isfinite(joints).all() or
            np.any(np.diff(times) <= 0)):
        raise ValueError('four or more seven-joint samples required')
    spline = make_interp_spline(times, joints, k=5,
        bc_type=([(1, start_va[0]), (2, start_va[1])],
                 [(1, end_va[0]), (2, end_va[1])]))
    axes = [PPoly.from_spline((spline.t, spline.c[:, i], spline.k)) for i in range(7)]
    mask = np.diff(axes[0].x) > 0
    return PPoly(np.stack([p.c[:, mask] for p in axes], axis=-1),
                 np.r_[axes[0].x[:-1][mask], axes[0].x[-1]], extrapolate=False)


def limits_report(curve, cfg, model):
    report = {'allowed': True, 'violations': []}
    for derivative, name, low, high in (
            (0, 'position', model.command_lower, model.command_upper),
            (1, 'velocity', -cfg.velocity_limits, cfg.velocity_limits),
            (2, 'acceleration', -cfg.acceleration_limits, cfg.acceleration_limits)):
        lo, hi = extrema(curve, derivative)
        report[name] = {'min': lo, 'max': hi}
        bad = (lo < low - 1e-9) | (hi > high + 1e-9)
        if bad.any():
            report['allowed'] = False
            report['violations'].append({'derivative': derivative,
                'joint_numbers': (np.flatnonzero(bad)+1), 'lower': low, 'upper': high})
    # Locate actual polynomial extrema, not just sampled acceleration peaks.
    acc = curve.derivative(2)
    candidates = list(np.r_[acc.x, np.nextafter(acc.x[1:], acc.x[:-1])])
    for roots in acc.derivative().roots(extrapolate=False):
        candidates.extend(np.asarray(roots)[np.isfinite(roots)].tolist())
    candidates = np.asarray(candidates)
    values = acc(candidates)
    peak_i = np.argmax(abs(values), axis=0)
    report['peak_acceleration_time_s'] = candidates[peak_i]
    report['peak_acceleration_abs'] = abs(values[peak_i, np.arange(7)])
    return report


def require_limits(curve, cfg, model):
    result = limits_report(curve, cfg, model)
    if not result['allowed']:
        raise PickRejected('hover q/v/a limits exceeded', details={'limits': result})
    return result


def path_for(curve, target, stage):
    return TimedPath(curve, target.identity, 1, stage, float(curve.x[-1]), 0.)


def jacobian(model, q, width):
    model.set_state(q, width)
    jp = np.zeros((3, model.model.nv)); jr = np.zeros_like(jp)
    model.mj.mj_jacSite(model.model, model.data, jp, jr, model.tcp_id)
    dofs = model.model.jnt_dofadr[model.ids]
    return np.vstack((jp[:, dofs], jr[:, dofs]))


def jacobian_rate(model, q, qd, width):
    h = 1e-5
    return (jacobian(model, q+h*qd, width)-jacobian(model, q-h*qd, width))/(2*h)


def endpoint_rates(model, q, width, velocity, acceleration):
    j = jacobian(model, q, width)
    qd = np.linalg.lstsq(j, np.r_[velocity, np.zeros(3)], rcond=None)[0]
    jd = jacobian_rate(model, q, qd, width)
    qdd = np.linalg.lstsq(j, np.r_[acceleration, np.zeros(3)]-jd@qd, rcond=None)[0]
    return qd, qdd


def cartesian_state(model, qva, width):
    q, qd, qdd = qva
    j = jacobian(model, q, width)
    jd = jacobian_rate(model, q, qd, width)
    model.set_state(q, width)
    tcp = model.tcp().copy()
    return tcp, j@qd, j@qdd+jd@qd


class StopReference:
    """Empty-jaw fallback, with the measured prefix endpoint derivatives."""
    def __init__(self, tcp, twist, acceleration, duration):
        self.duration_s = self.full_duration_s = self.prealign_s = duration
        self.follow_end_s = duration+1.
        self.rotate_first_s = 0.
        z = np.zeros(3)
        end = tcp[:3, 3]+twist[:3]*duration/2+acceleration[:3]*duration**2/12
        self.curve = BPoly.from_derivatives([0., duration],
            [[tcp[:3, 3], twist[:3], acceleration[:3]], [end, z, z]])
        self.rotation = Rotation.from_matrix(tcp[:3, :3])
        self.rotation_curve = BPoly.from_derivatives([0., duration],
            [[z, tcp[:3, :3].T@twist[3:], tcp[:3, :3].T@acceleration[3:]], [z, z, z]])

    def sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.duration_s))
        return self.curve(t), (self.rotation*Rotation.from_rotvec(
            self.rotation_curve(t))).as_quat()[[3, 0, 1, 2]]


@dataclass(frozen=True)
class MovingJoin(SynchronizedIntercept):
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        super().__post_init__()
        if self.rotate_first_s != 0:
            raise ValueError('hover moving join supports no separate rotation lead')
        for v in (self.linear_velocity, self.linear_acceleration,
                  self.angular_velocity, self.angular_acceleration):
            if np.shape(v) != (3,) or not np.isfinite(v).all():
                raise ValueError('finite moving boundary derivatives required')
        object.__setattr__(self, '_join', BPoly.from_derivatives([0., self.prealign_s],
            [[self.start_tcp_world, self.linear_velocity, self.linear_acceleration],
             [self._sync_start, self.velocity_world, np.zeros(3)]]))
        r0 = Rotation.from_quat(self.start_quaternion_wxyz[[1, 2, 3, 0]])
        r1 = Rotation.from_quat(self.target_quaternion_wxyz[[1, 2, 3, 0]])
        object.__setattr__(self, '_r0', r0)
        object.__setattr__(self, '_rotation', BPoly.from_derivatives([0., self.prealign_s],
            [[np.zeros(3), r0.inv().apply(self.angular_velocity),
              r0.inv().apply(self.angular_acceleration)],
             [(r0.inv()*r1).as_rotvec(), np.zeros(3), np.zeros(3)]]))

    def sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        quat = self.target_quaternion_wxyz if t >= self.prealign_s else (
            self._r0*Rotation.from_rotvec(self._rotation(t))).as_quat()[[3, 0, 1, 2]]
        return self.position_derivatives(t)[0], quat.copy()


def phase_times(reference, hz):
    """Distribute knots within phases instead of inserting tiny boundary intervals."""
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError('positive finite hover IK rate required')
    duration = reference.full_duration_s
    boundaries = sorted(set([0., duration]+[float(x) for x in (
        reference.prealign_s, reference.duration_s, reference.follow_end_s,
        getattr(reference, 'rotate_first_s', 0.), getattr(reference, 'wait_s', 0.),
        getattr(reference, 'rotation_end_s', 0.)) if 0 < x < duration]))
    tolerance = 64*np.finfo(float).eps*max(1., duration)
    merged = [boundaries[0]]
    for t in boundaries[1:]:
        if t-merged[-1] > tolerance:
            merged.append(t)
        elif t == duration:
            merged[-1] = t
    return np.concatenate(([0.], *[np.linspace(a, b, max(1, int(np.ceil((b-a)*hz)))+1)[1:]
                       for a, b in zip(merged[:-1], merged[1:])]))


def solve_curve(planner, reference, start_qva, *, hz, seed_fn=None, end_va=None):
    times = phase_times(reference, hz)
    if len(times) < 4:
        times = np.linspace(0., reference.full_duration_s, 4)
    q0, qd0, qdd0 = map(np.asarray, start_qva)
    planner.model.set_state(q0, planner.config.settings.open_width_m)
    solver = PrecisePoseSolver(planner.model)
    joints = [q0.copy()]
    previous_pose = reference.sample(0.)
    for i, t in enumerate(times[1:], 1):
        if getattr(planner, '_hover_cancel', lambda: False)():
            raise PickRejected('hover planning cancelled or deadline expired')
        pose = reference.sample(t)
        if seed_fn is None:
            # Differential continuation preserves the current redundant branch.
            # Secant extrapolation can accumulate an uncontrolled nullspace speed.
            r0, r1 = [Rotation.from_quat(np.asarray(p[1])[[1,2,3,0]])
                      for p in (previous_pose, pose)]
            delta = np.r_[pose[0]-previous_pose[0], (r1*r0.inv()).as_rotvec()]
            jac = solver.ik.compute_jacobian(joints[-1])
            step = np.linalg.lstsq(jac, delta, rcond=None)[0]
            bound = planner.config.velocity_limits*(t-times[i-1])
            seed = np.clip(joints[-1]+np.clip(step,-bound,bound),
                           planner.model.command_lower,planner.model.command_upper)
        else:
            seed = seed_fn(t)
        result = solver.solve(*pose, seed, cancelled=getattr(planner, '_hover_cancel', lambda: False))
        if not result.success:
            raise PickRejected('hover IK failed', details={'elapsed_s': t,
                'position_error': result.position_error, 'orientation_error': result.orientation_error})
        jump = abs(result.joint_positions-joints[-1])
        if np.any(jump > planner.config.velocity_limits*(t-times[i-1])+1e-7):
            raise PickRejected('hover IK branch exceeds neighboring velocity bounds',
                details={'elapsed_s':float(t), 'joint_step_rad':jump,
                         'sample_interval_s':float(t-times[i-1])})
        joints.append(result.joint_positions.copy())
        previous_pose = pose
    if callable(end_va):
        end_va = end_va(joints[-1])
    if end_va is None:
        end_va = (np.zeros(7), np.zeros(7))
    return fit_curve(times, joints, (qd0, qdd0), end_va)


def fallback(planner, old, state, prefix, delay):
    begun = time.perf_counter()
    qva = prefix.sample(delay)
    tcp, twist, acc = cartesian_state(planner.model, qva, state['width_m'])
    ref = StopReference(tcp, twist, acc, .4)
    bounds = polynomial_range(ref.curve)
    workspace = planner.config.workspace
    if np.any(bounds[:, 0] < workspace[:, 0]) or np.any(bounds[:, 1] > workspace[:, 1]):
        raise PickRejected('fallback reference outside workspace')
    accel = np.linalg.norm(np.max(abs(polynomial_range(ref.curve, 2)), axis=1))
    if accel > planner.config.raw['phase_timing']['max_tcp_acceleration_m_s2']+1e-9:
        raise PickRejected('fallback TCP acceleration bound exceeded')
    curve = solve_curve(planner, ref, qva, hz=50.)
    checks = require_limits(curve, planner.config, planner.model)
    path = path_for(curve, old.target, 'hover_empty_stop')
    geometry = planner.check_path(path, state['width_m'], origin=old.reference.start_s+delay, reference=ref, cancelled=getattr(planner, '_hover_cancel', lambda: False))
    axis_poly = PPoly.from_bernstein_basis(ref.curve)
    axis_poly = PPoly(axis_poly.c@planner.config.direction, axis_poly.x)
    roots = axis_poly.derivative().roots(extrapolate=False)
    positions = axis_poly(np.r_[axis_poly.x, roots[np.isfinite(roots)]])
    spec = planner.config.raw['phase_timing']
    stop_limits = spec['stop_interval_base_m']
    pad = float(old.target.velocity@planner.config.direction)*spec['residual_time_bound_s']+spec['position_margin_m']
    # The empty jaw can stop at the original waiting pose, which is upstream
    # of the padded grasp interval. The lower grasp bound applies to holding
    # the object, not to an empty-jaw cancellation. Retain the downstream
    # stop bound and all original robot/workspace/collision constraints.
    if positions.max() > stop_limits[1]-pad:
        raise PickRejected('fallback exceeds downstream stop bound')
    return path, {'allowed': True, 'wall_s': time.perf_counter()-begun,
        'stop_coordinate_m': float(axis_poly(.4)),
        'extra_forward_m': float(positions.max()-tcp[:3, 3]@planner.config.direction),
        'limits': checks, 'geometry': geometry}


class HoverPrefix:
    """Wait/ramp longitudinally; align lateral position and orientation in parallel.

    No constant-speed translation is assumed to remove an initial offset.
    The ramp is scheduled to end exactly above the moving grasp point.
    A smooth orientation transition runs from original start until alignment.
    After alignment the prefix keeps a constant vertical offset from the target.
    """
    def __init__(self, old, direction, ramp, hold_after, hover_height=None, *, alignment_fraction=1.,
                 bin_join_min_m=None):
        r = old.reference
        direction = np.asarray(direction)
        if abs(direction[2]) > 1e-10 or abs(np.linalg.norm(direction)-1) > 1e-8:
            raise ValueError('hover decomposition requires a horizontal unit belt axis')
        self.axis = direction
        side = np.cross(np.array([0., 0., 1.]), direction)
        basis = np.column_stack((direction, side, [0., 0., 1.]))
        p0 = r.start_tcp_world
        object0 = r.capture_tcp_world+r.velocity_world*(r.start_s-r.capture_s)
        speed = float(r.velocity_world@direction)
        lead = float((p0-object0)@direction)
        if min(speed, ramp) <= 0 or hold_after < 0:
            raise ValueError('positive speed/ramp required')
        if bin_join_min_m is None:
            wait = lead/speed-ramp/2
        else:
            # Travel upstream from the bin and arrive with the object's forward
            # velocity. Spend any excess time before the longitudinal approach.
            wait = max(0., (bin_join_min_m-float(object0@direction))/speed-ramp)
        if wait < 0:
            raise ValueError('ramp too long for a forward-only rendezvous at waiting coordinate')
        sync = wait+ramp
        self.sync_s, self.wait_s, self.ramp_s = sync, wait, ramp
        if not np.isfinite(alignment_fraction) or not 0 < alignment_fraction <= 1:
            raise ValueError('orientation alignment fraction must be in (0, 1]')
        self.rotation_end_s = sync*alignment_fraction
        self.alignment_fraction = alignment_fraction
        self.prealign_s = sync
        self.rotate_first_s = self.rotation_end_s
        self.duration_s = self.full_duration_s = sync+hold_after
        self.follow_end_s = self.full_duration_s+1.  # no attached payload in prefix
        object_sync = object0+r.velocity_world*sync
        hover = object_sync.copy()
        hover[2] = p0[2] if hover_height is None else object_sync[2]+hover_height
        self.height_m = float(hover[2]-object_sync[2])
        if self.height_m <= 0:
            raise ValueError('waiting TCP is not above grasp height')
        local0 = p0@basis; end = hover@basis; velocity = r.velocity_world@basis
        zero = np.zeros(2)
        transverse = BPoly.from_derivatives([0., sync],
            [[local0[1:], zero, zero], [end[1:], velocity[1:], zero]])
        longitudinal = BPoly.from_derivatives([0., ramp],
            [[local0[0], 0., 0.], [end[0], speed, 0.]])
        def state(t):
            if t >= sync:
                return (hover+r.velocity_world*(t-sync), r.velocity_world.copy(), np.zeros(3))
            result = []
            for d in (0, 1, 2):
                along = (local0[0] if d == 0 else 0.) if t <= wait else longitudinal(t-wait, d)
                result.append(basis@np.r_[along, transverse(t, d)])
            return tuple(result)
        times = np.unique([0., wait, sync, self.full_duration_s])
        self.curve = BPoly.from_derivatives(times, [state(t) for t in times])
        self.orientations = Slerp([0., 1.], Rotation.from_quat(np.array([
            r.start_quaternion_wxyz, r.target_quaternion_wxyz])[:, [1, 2, 3, 0]]))
        self.goal_quat = r.target_quaternion_wxyz.copy()
        self.velocity = r.velocity_world.copy()

    def position_derivatives(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        return tuple(self.curve(t, d) for d in (0, 1, 2))

    def sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.full_duration_s))
        if t >= self.rotation_end_s:
            quat = self.goal_quat.copy()
        else:
            u = t/self.rotation_end_s
            quat = self.orientations([10*u**3-15*u**4+6*u**5]).as_quat()[0][[3, 0, 1, 2]]
        return self.curve(t), quat


def prepare(planner, old, state, ref, *, ik_hz=50.):
    begun = time.perf_counter()
    cfg = planner.config; spec = cfg.raw['phase_timing']
    acceleration_bound = np.linalg.norm(np.max(abs(polynomial_range(ref.curve, 2)), axis=1))
    if acceleration_bound > spec['max_tcp_acceleration_m_s2']+1e-9:
        raise PickRejected('hover prefix TCP acceleration exceeds limit', details={'bound': acceleration_bound})
    ranges = polynomial_range(ref.curve)
    pad = abs(old.target.velocity)*spec['residual_time_bound_s']+spec['position_margin_m']
    if (np.any(ranges[:, 0] < cfg.workspace[:, 0]+pad)
            or np.any(ranges[:, 1] > cfg.workspace[:, 1]-pad)):
        raise PickRejected('hover prefix outside padded workspace')
    qva = (np.asarray(state['joints']), np.zeros(7), np.zeros(7))
    attempts = []
    for seed_name in (('sequential', 'old_branch') if hasattr(old, 'path') else ('sequential',)):
        try:
            # An alternative seed is allowed, but every resulting curve gets
            # identical extrema and geometry checks. Failed branches are kept.
            seed_fn = None if seed_name == 'sequential' else lambda t: old.path.position(
                min(t/ref.sync_s, 1.)*old.reference.prealign_s)
            curve = solve_curve(planner, ref, qva, hz=ik_hz, seed_fn=seed_fn,
                end_va=lambda q: endpoint_rates(planner.model, q, state['width_m'],
                                               old.target.velocity, np.zeros(3)))
            checks = require_limits(curve, cfg, planner.model)
            path = path_for(curve, old.target, 'hover_rendezvous_prefix')
            geometry = planner.check_path(path, state['width_m'], origin=old.reference.start_s, reference=ref, cancelled=getattr(planner, '_hover_cancel', lambda: False))
            return path, {'status': 'certified', 'wall_s': time.perf_counter()-begun,
                'limits': checks, 'geometry': geometry, 'tcp_acceleration_bound_m_s2': acceleration_bound,
                'seed': seed_name, 'ik_hz':ik_hz, 'interpolation':'phase_uniform_quintic',
                'continuation':'differential_pose_predictor', 'attempts': attempts}
        except PickRejected as error:
            attempts.append({'seed': seed_name, 'error': str(error), 'details': error.details})
    raise PickRejected('hover prefix has no certified branch', details={'attempts': attempts})


def descent(planner, old, state, prefix, switch, duration):
    begun = time.perf_counter()
    row = {'switch_after_original_start_s': switch, 'descent_s': duration, 'status': 'rejected', 'attempts': []}
    qva = prefix.sample(switch)
    tcp, twist, acceleration = cartesian_state(planner.model, qva, state['width_m'])
    r = old.reference
    object_at_switch = r.capture_tcp_world+r.velocity_world*(r.start_s+switch-r.capture_s)
    relative = tcp[:3, 3]-object_at_switch
    row.update(start_qva=qva, relative_xyz_at_switch_m=relative,
        velocity_error_at_switch_m_s=twist[:3]-r.velocity_world,
        angular_speed_at_switch_rad_s=float(np.linalg.norm(twist[3:])))
    # Numerical interpolation of IK points is checked, not silently interpreted
    # as exact Cartesian co-motion. These are preplanning admission tolerances.
    if np.linalg.norm(relative[:2]) > .0005 or np.linalg.norm(twist[:3]-r.velocity_world) > .005:
        row.update(error='prefix has not reached position/velocity synchronization', wall_s=time.perf_counter()-begun)
        return row, None
    ref = MovingJoin(capture_s=r.capture_s, start_s=r.start_s+switch,
        duration_s=duration, capture_tcp_world=r.capture_tcp_world,
        velocity_world=r.velocity_world, start_tcp_world=tcp[:3, 3],
        start_quaternion_wxyz=Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
        target_quaternion_wxyz=r.target_quaternion_wxyz, follow_s=r.follow_s,
        brake_s=r.brake_s, prealign_s=duration,
        linear_velocity=twist[:3], linear_acceleration=acceleration[:3],
        angular_velocity=twist[3:], angular_acceleration=acceleration[3:])
    row.update(contact_delay_from_old_s=ref.contact_s-r.contact_s,
        stop_space_consumed_m=float((ref.position_derivatives(ref.full_duration_s)[0]-
            r.position_derivatives(r.full_duration_s)[0])@planner.config.direction),
        stop_coordinate_m=float(ref.position_derivatives(ref.full_duration_s)[0]@planner.config.direction))
    try:
        row['phase'] = require_reference(planner.config, ref)
        for hz in (10., 50.):
            stage = 'ik'
            try:
                curve = solve_curve(planner, ref, qva, hz=hz)
                stage = 'polynomial_limits'
                checks = require_limits(curve, planner.config, planner.model)
                path = path_for(curve, old.target, 'hover_descent_follow_brake')
                stage = 'sampled_geometry'
                geometry = planner.check_smooth_intercept(path, ref, cancelled=getattr(planner, '_hover_cancel', lambda: False))
                continuity = [float(np.max(abs(curve(0., d)-qva[d]))) for d in (0, 1, 2)]
                if max(continuity) > 1e-7:
                    raise PickRejected('moving splice q/v/a discontinuity')
                row.update(status='certified', limits=checks, geometry=geometry,
                    ik_hz=hz, continuity_qva_max=continuity, wall_s=time.perf_counter()-begun)
                return row, path
            except PickRejected as error:
                row['attempts'].append({'stage': stage, 'hz': hz, 'error': str(error), 'details': error.details})
        row['error'] = 'descent/follow/brake has no certified joint curve'
    except PickRejected as error:
        row.update(error=str(error), details=error.details)
    row['wall_s'] = time.perf_counter()-begun
    return row, None

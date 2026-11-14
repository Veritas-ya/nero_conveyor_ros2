"""Opt-in, bounded discrete allocation of join / closure-follow / brake time.

Ranks analytic reference margins, then delegates certification to the existing
planner. This is not a global optimizer or a physical grasp/slip certificate.
"""
from collections import Counter
import hashlib
import itertools
import json

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import GRIP_CONFIRMATION_DWELL_S
from .synchronized_reference import SynchronizedIntercept, schedule_forward_join

KEY = 'phase_timing'
FIELDS = {'mode', 'join_candidates_s', 'follow_candidates_s', 'brake_candidates_s',
          'gripper_close_time_s', 'gripper_time_margin_s', 'residual_time_bound_s',
          'position_margin_m', 'stop_interval_base_m', 'max_tcp_acceleration_m_s2',
          'calibration_status', 'calibration_note'}


def validate_configuration(raw, workspace, direction):
    spec = raw[KEY]
    if (not isinstance(spec, dict) or not FIELDS <= set(spec)
            or set(spec)-FIELDS-{'closure_window_policy'}):
        raise ValueError('phase_timing requires exactly the documented fields')
    if (raw.get('trajectory_profile') != 'smooth'
            or raw.get('intercept_strategy') != 'synchronized'
            or raw.get('closure_policy') != 'scheduled_attempt'
            or raw.get('rotate_first_retime_enabled', False)):
        raise ValueError('phase_timing requires synchronized scheduled closure without rotation retiming')
    if 'execution_delay_window' in raw:
        raise ValueError('phase_timing and execution_delay_window cannot be combined (duplicate margins)')
    if spec['mode'] not in ('nominal', 'robust'):
        raise ValueError('phase_timing mode must be nominal or robust')
    if spec['calibration_status'] not in ('assumed_offline', 'measured', 'experimental_hardware'):
        raise ValueError('phase_timing requires an explicit calibration_status')
    closure = spec.get('closure_window_policy', 'confirm_before_brake')
    if closure not in ('confirm_before_brake', 'fixed_follow_experiment'):
        raise ValueError('unknown phase_timing closure_window_policy')
    if spec['calibration_status'] == 'experimental_hardware' and closure != 'fixed_follow_experiment':
        raise ValueError('experimental_hardware requires fixed_follow_experiment')
    if not isinstance(spec['calibration_note'], str) or not spec['calibration_note'].strip():
        raise ValueError('phase_timing requires a calibration_note')
    count = 1
    for name in ('join_candidates_s', 'follow_candidates_s', 'brake_candidates_s'):
        values = spec[name]
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)):
            raise ValueError(name+' must be a nonempty numeric list')
        a = np.asarray(values, float)
        if not np.isfinite(a).all() or np.any(a <= 0) or np.any(np.diff(a) <= 0):
            raise ValueError(name+' must be strictly increasing positive finite seconds')
        count *= len(a)
    if count > 256:
        raise ValueError('phase_timing grid must contain at most 256 timing combinations')
    if closure == 'fixed_follow_experiment' and len(spec['follow_candidates_s']) != 1:
        raise ValueError('fixed_follow_experiment requires exactly one follow duration')
    for name in ('gripper_close_time_s', 'gripper_time_margin_s', 'residual_time_bound_s',
                 'position_margin_m', 'max_tcp_acceleration_m_s2'):
        v = spec[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(name+' must be a nonnegative finite number')
        if name in ('gripper_close_time_s', 'max_tcp_acceleration_m_s2') and v <= 0:
            raise ValueError(name+' must be positive')
    bounds = np.asarray(spec['stop_interval_base_m'], float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('stop_interval_base_m must be an ordered finite pair')
    vertices = np.array(list(itertools.product(*np.asarray(workspace))))
    projected = vertices @ np.asarray(direction)
    if bounds[0] < projected.min()-1e-9 or bounds[1] > projected.max()+1e-9:
        raise ValueError('stop interval must be inside the projection of the existing workspace')


def polynomial_range(curve, derivative=0):
    """Exact per-coordinate extrema of a BPoly, including interior extrema."""
    power = PPoly.from_bernstein_basis(curve).derivative(derivative)
    result = []
    for axis in range(3):
        scalar = PPoly(power.c[..., axis], power.x)
        roots = scalar.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots)]
        samples = np.r_[power.x, roots]
        values = scalar(samples)
        result.append([float(np.min(values)), float(np.max(values))])
    return np.array(result)


def make_reference(cfg, target, start, grasp, join_s, follow_s, brake_s, *, now_s, reservation_s):
    rotation = cfg.raw.get('rotate_first_s', 0.)
    sync = cfg.raw.get('synchronized_duration_s', 0.)
    schedule = schedule_forward_join(capture_s=target.capture_s, grasp_position=grasp[:3, 3],
        velocity=target.velocity, standby_position=start[:3, 3], direction=cfg.direction,
        now_s=now_s, planning_reserve_s=reservation_s, advance_s=cfg.raw['command_advance_s'],
        join_s=join_s, rotate_first_s=rotation, synchronized_s=sync, follow_s=follow_s,
        follow_bounds=cfg.follow_bounds)
    ref = SynchronizedIntercept(target.capture_s, schedule['join_start_s'], join_s+rotation+sync,
        grasp[:3, 3], target.velocity, start[:3, 3],
        Rotation.from_matrix(start[:3, :3]).as_quat()[[3, 0, 1, 2]],
        Rotation.from_matrix(grasp[:3, :3]).as_quat()[[3, 0, 1, 2]],
        follow_s, brake_s, join_s+rotation, rotate_first_s=rotation)
    return ref


def assess_reference(cfg, ref, *, join_cache=None):
    spec = cfg.raw[KEY]
    robust = spec['mode'] == 'robust'
    dt = spec['residual_time_bound_s'] if robust else 0.
    pad = spec['position_margin_m'] if robust else 0.
    speed = float(ref.velocity_world @ cfg.direction)
    norm_speed = float(np.linalg.norm(ref.velocity_world))
    if speed <= 0:
        raise ValueError('phase_timing requires forward target motion')
    # Closure is requested at duration_s, NOT at the beginning of synchronization.
    # Do not credit the pre-closure segment or command advance as jaw closing time.
    closure = spec.get('closure_window_policy', 'confirm_before_brake')
    # Explicit hardware experiment: hold the requested follow duration, then
    # allow confirmation to continue during braking, as scheduled_attempt does.
    # This is a time allocation constraint, NOT proof of physical jaw closure.
    required = (spec['follow_candidates_s'][0] if closure == 'fixed_follow_experiment' else
                spec['gripper_close_time_s'] + GRIP_CONFIRMATION_DWELL_S
                + 1./cfg.settings.control_hz
                + (spec['gripper_time_margin_s']+dt if robust else 0.))
    sync_start = ref.position_derivatives(ref.prealign_s)[0]
    follow_end = ref.position_derivatives(ref.follow_end_s)[0]
    stop = ref.position_derivatives(ref.full_duration_s)[0]
    displacement_pad = speed*dt+pad
    follow_bounds = np.asarray(cfg.follow_bounds)
    stop_bounds = np.asarray(spec['stop_interval_base_m'])
    follow_margin = min(float(sync_start@cfg.direction)-follow_bounds[0]-displacement_pad,
                        follow_bounds[1]-float(follow_end@cfg.direction)-displacement_pad)
    stop_margin = min(float(follow_end@cfg.direction)-stop_bounds[0]-displacement_pad,
                      stop_bounds[1]-float(stop@cfg.direction)-displacement_pad)
    join_key = tuple(np.r_[ref.start_tcp_world, sync_start, ref.velocity_world,
                          ref.prealign_s-ref.rotate_first_s])
    cached = None if join_cache is None else join_cache.get(join_key)
    if cached is None:
        join_range = polynomial_range(ref._join)
        join_acc = float(np.linalg.norm(np.max(np.abs(polynomial_range(ref._join, 2)), axis=1)))
        if join_cache is not None: join_cache[join_key] = (join_range, join_acc)
    else:
        join_range, join_acc = cached
    brake_range = polynomial_range(ref._braking)
    ranges = np.stack([join_range, brake_range,
        np.column_stack((np.minimum(sync_start, follow_end), np.maximum(sync_start, follow_end)))])
    workspace_pad = np.abs(ref.velocity_world)*dt+pad
    workspace_margin = float(min(np.min(ranges[:, :, 0]-cfg.workspace[:, 0]-workspace_pad),
                                 np.min(cfg.workspace[:, 1]-ranges[:, :, 1]-workspace_pad)))
    # Norm of coordinatewise extrema is a conservative bound, not sampled peaks.
    brake_acc = 1.5*norm_speed/ref.brake_s
    limit = spec['max_tcp_acceleration_m_s2']
    margins = {'closure_time_s': ref.follow_s-required, 'follow_interval_m': follow_margin,
               'stop_interval_m': stop_margin, 'workspace_m': workspace_margin,
               'tcp_acceleration_m_s2': limit-max(join_acc, brake_acc)}
    reasons = [name for name, value in margins.items() if value < -1e-9]
    scales = [required, follow_bounds[1]-follow_bounds[0], stop_bounds[1]-stop_bounds[0],
              float(np.min(cfg.workspace[:, 1]-cfg.workspace[:, 0])), limit]
    # A fixed follow has identically zero time margin; it must remain a hard
    # constraint without flattening every candidate's ranking to zero.
    score = float(min(v/s for (name, v), s in zip(margins.items(), scales)
                      if closure != 'fixed_follow_experiment' or name != 'closure_time_s'))
    return {'mode': spec['mode'], 'allowed': not reasons, 'reasons': reasons,
        'closure_window_policy': closure,
        'confirmation_required_before_brake': closure == 'confirm_before_brake',
        'physical_closure_before_brake_guaranteed': False,
        'join_s': ref.prealign_s-ref.rotate_first_s, 'preclosure_sync_s': ref.duration_s-ref.prealign_s,
        'follow_s': ref.follow_s, 'brake_s': ref.brake_s,
        'required_closure_window_s': required, 'margins': margins, 'reference_margin_score': score,
        'join_acceleration_bound_m_s2': join_acc, 'brake_peak_acceleration_m_s2': brake_acc,
        'braking_distance_m': norm_speed*ref.brake_s/2,
        'stop_coordinate_m': float(stop@cfg.direction),
        'residual_time_bound_s': dt, 'position_margin_m': pad,
        'calibration_status': spec['calibration_status'],
        'physical_grasp_guaranteed': False, 'no_slip_guaranteed': False,
        'uncertain_collision_tube_certified': False}


def binding(cfg, ref):
    fields = ('capture_s', 'start_s', 'duration_s', 'capture_tcp_world', 'velocity_world',
              'start_tcp_world', 'start_quaternion_wxyz', 'target_quaternion_wxyz',
              'follow_s', 'brake_s', 'prealign_s', 'rotate_first_s')
    data = {'spec': cfg.raw[KEY], 'reference': {k: getattr(ref, k) for k in fields},
        'workspace': cfg.workspace, 'follow_bounds': cfg.follow_bounds, 'direction': cfg.direction,
        'advance': cfg.raw['command_advance_s'], 'control_hz': cfg.settings.control_hz,
        'closure_policy': cfg.raw.get('closure_policy'),
        'velocity_limits': cfg.velocity_limits, 'acceleration_limits': cfg.acceleration_limits}
    return hashlib.sha256(json.dumps(data, sort_keys=True,
        default=lambda v: np.asarray(v).tolist(), allow_nan=False).encode()).hexdigest()


def require_reference(cfg, ref):
    report = assess_reference(cfg, ref)
    spec = cfg.raw[KEY]
    for field, value in [('join_candidates_s', report['join_s']),
                         ('follow_candidates_s', ref.follow_s), ('brake_candidates_s', ref.brake_s)]:
        if not any(abs(value-v) < 1e-9 for v in spec[field]):
            report['reasons'].append('duration_outside_declared_grid')
            report['allowed'] = False
    if not report['allowed']:
        raise PickRejected('阶段时长不能满足夹持窗口/停止空间/运动约束',
                           details={'reason': 'phase_timing_rejected', KEY: report})
    report['binding_sha256'] = binding(cfg, ref)
    return report


def verify_plan(cfg, path, ref, *, for_execution=False):
    saved = path.diagnostics.get(KEY)
    report = require_reference(cfg, ref)
    if saved is None or saved.get('binding_sha256') != report['binding_sha256']:
        raise PickRejected('阶段时长认证记录缺失或已过期')
    if for_execution:
        spec = cfg.raw[KEY]
        experimental = (spec['calibration_status'] == 'experimental_hardware'
                        and spec.get('closure_window_policy') == 'fixed_follow_experiment'
                        and cfg.raw.get('_phase_timing_hardware_test_authorized') is True)
        if spec['calibration_status'] != 'measured' and not experimental:
            raise PickRejected('阶段时长参数仅为离线假设或未标定；实验参数只能从独立真机测试入口执行')
    return report


def authorize_hardware_test(cfg, *, enabled=False):
    """Runtime authorization from the dedicated entry, never from saved YAML."""
    cfg.raw.pop('_phase_timing_hardware_test_authorized', None)
    spec = cfg.raw.get(KEY)
    if spec is None:
        if enabled:
            raise ValueError('独立阶段时长入口需要 phase_timing 配置')
        return
    if spec['calibration_status'] == 'experimental_hardware':
        if not enabled or spec.get('closure_window_policy') != 'fixed_follow_experiment':
            raise ValueError('实验阶段时长配置只能通过独立真机测试入口 --execute 执行')
        cfg.raw['_phase_timing_hardware_test_authorized'] = True
    elif enabled:
        raise ValueError('独立测试入口需要明确标记 experimental_hardware，不能把假设冒充标定')


def candidate_options(cfg, target, start, grasps, *, now_s, reservation_s, cancelled=lambda: False):
    """Rank at most 512 grasp/timing pairs; recheck timing after actual IK work."""
    spec = cfg.raw[KEY]
    options, rows, join_cache = [], [], {}
    for join, follow, brake in itertools.product(spec['join_candidates_s'],
            spec['follow_candidates_s'], spec['brake_candidates_s']):
        for index, grasp in enumerate(grasps):
            if cancelled():
                raise PickRejected('阶段时长搜索预算耗尽或取消', details={'phase_timing_search': rows})
            row = {'grasp_index': index, 'join_s': join, 'follow_s': follow, 'brake_s': brake}
            try:
                ref = make_reference(cfg, target, start, grasp, join, follow, brake,
                                     now_s=now_s, reservation_s=reservation_s)
                report = assess_reference(cfg, ref, join_cache=join_cache)
                row.update(report)
                if report['allowed']:
                    row['completion_from_now_s'] = ref.start_s+ref.full_duration_s-now_s
                    options.append(row)
            except ValueError as error:
                row.update(allowed=False, reasons=['schedule'], error=str(error))
            rows.append(row)
    options.sort(key=lambda r: (-r['reference_margin_score'], r['completion_from_now_s'],
                               r['join_s'], r['follow_s'], r['brake_s'], r['grasp_index']))
    # Try each join/grasp family once before spending the deadline on another
    # follow/brake variation of the same short (possibly unreachable) join.
    families = {}
    for row in options:
        families.setdefault((row['join_s'], row['grasp_index']), []).append(row)
    options = [family[depth] for depth in range(max(map(len, families.values()), default=0))
               for family in families.values() if depth < len(family)]
    search = {'policy': 'margin_ranked_join_family_round_robin_then_first_fully_certified',
        'global_optimum_claimed': False, 'evaluated_pairs': len(rows), 'feasible_pairs': len(options),
        'rejection_counts': dict(Counter(reason for row in rows if not row['allowed']
                                        for reason in row['reasons'])), 'rows': rows}
    return options, search

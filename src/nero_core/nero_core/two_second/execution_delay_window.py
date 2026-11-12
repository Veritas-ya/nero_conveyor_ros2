"""Opt-in empirical residual-delay checks for a finite synchronized grasp region.

This is a conditional kinematic envelope, NOT a physical delay guarantee or a
collision/grasp certificate. Existing geometry and sender deadlines stay active.
The lag is host-send-to-feedback; controller sampling/return delay is unresolved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


PROFILE_SCHEMA = 'nero_empirical_execution_delay_v1'
KEY = 'execution_delay_window'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name+' must be numeric')
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name+' must be finite and nonnegative/positive')
    return value


class DelayWindowError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.details = {'reason': 'execution_delay_window_rejected', 'execution_delay_window': report}


def read_profile(path):
    profile = json.loads(Path(path).read_text())
    if (profile.get('schema') != PROFILE_SCHEMA
            or profile.get('measurement') != 'equivalent_host_send_to_joint_feedback_lag'
            or profile.get('coverage') != 'observed_range_plus_declared_margin_not_hardware_bound'
            or profile.get('phase') != 'near_contact'):
        raise ValueError('unsupported empirical execution-delay profile')
    bounds = profile.get('delay_bounds_s')
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError('delay_bounds_s must contain lower and upper bounds')
    low, high = [numeric(v, 'delay bound') for v in bounds]
    if low > high:
        raise ValueError('delay bounds must be ordered')
    n = profile.get('samples')
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError('delay profile needs at least three fitted cycles')
    if not isinstance(profile.get('condition_id'), str) or not profile['condition_id'].strip():
        raise ValueError('delay profile needs a condition identifier')
    return profile


def validate_configuration(raw, source):
    """Load and pin one immutable-in-process profile; no hot-path file reads."""
    if KEY not in raw:
        return
    spec = raw[KEY]
    keys = {'mode', 'profile_file', 'profile_sha256', 'condition_id', 'maximum_relative_error_m'}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError(KEY+' requires exactly '+', '.join(sorted(keys)))
    if spec['mode'] not in ('shadow', 'experimental'):
        raise ValueError('execution delay mode must be shadow or experimental')
    if (raw.get('trajectory_profile'), raw.get('intercept_strategy')) != ('smooth', 'synchronized'):
        raise ValueError('execution delay window requires smooth synchronized planning')
    numeric(spec['maximum_relative_error_m'], 'maximum_relative_error_m', positive=True)
    if not isinstance(spec['profile_file'], str) or not spec['profile_file'].strip():
        raise ValueError('execution delay profile_file must be a path')
    path = Path(spec['profile_file']).expanduser()
    path = (path if path.is_absolute() else Path(source).parent/path).resolve(strict=True)
    if sha256(path) != spec['profile_sha256']:
        raise ValueError('execution delay profile hash differs from pinned configuration')
    profile = read_profile(path)
    if spec['condition_id'] != profile['condition_id']:
        raise ValueError('execution delay condition label differs from profile')
    spec = dict(spec, profile_file=str(path))
    raw[KEY] = spec
    raw['_execution_delay_profile'] = profile


def envelope(config, velocity):
    spec = config.raw[KEY]
    profile = config.raw.get('_execution_delay_profile')
    if profile is None:
        raise ValueError('execution delay profile was not validated at configuration load')
    velocity = np.asarray(velocity, dtype=float)
    axis = np.asarray(config.direction, dtype=float)
    bounds = np.asarray(config.follow_bounds, dtype=float)
    if (velocity.shape != (3,) or axis.shape != (3,) or bounds.shape != (2,)
            or not np.isfinite(np.r_[velocity, axis, bounds]).all()
            or not np.isclose(np.linalg.norm(axis), 1., atol=1e-10, rtol=0)
            or bounds[0] >= bounds[1]):
        raise ValueError('invalid execution-delay geometry')
    speed = float(velocity@axis)
    if speed <= 0:
        raise ValueError('execution delay envelope requires forward belt velocity')
    advance = numeric(config.raw['command_advance_s'], 'command_advance_s')
    residual = np.asarray(profile['delay_bounds_s'], dtype=float)-advance
    # At nominal reference phase r, effective physical time is r+tau-advance.
    # Preserve BOTH the nominal TCP and the possible target coordinates in the
    # original interval. The 90 ms advance is subtracted exactly once.
    erosion = speed*np.array([max(0., -residual[0]), max(0., residual[1])])
    effective = bounds+erosion*np.array([1., -1.])
    error = float(np.linalg.norm(velocity)*max(abs(residual)))
    reasons = []
    if effective[0] >= effective[1]:
        reasons.append('empty_finite_grasp_interval')
    if error > spec['maximum_relative_error_m']+1e-12:
        reasons.append('relative_position_error_exceeds_declared_tolerance')
    return {'schema': 'nero_execution_delay_window_check_v1', 'mode': spec['mode'],
        'profile_sha256': spec['profile_sha256'], 'condition_id': spec['condition_id'],
        'absolute_delay_bounds_s': list(profile['delay_bounds_s']), 'command_advance_s': advance,
        'residual_delay_bounds_s': residual.tolist(), 'speed_along_belt_m_s': speed,
        'original_follow_bounds_m': bounds.tolist(), 'effective_follow_bounds_m': effective.tolist(),
        'upstream_erosion_m': float(erosion[0]), 'downstream_erosion_m': float(erosion[1]),
        'relative_position_error_bound_m': error,
        'maximum_relative_error_m': spec['maximum_relative_error_m'],
        'allowed': not reasons, 'reasons': reasons,
        'scope': 'conditional_constant_lag_sync_to_follow_target_envelope_only',
        'physical_delay_guaranteed': False, 'physical_grasp_verified': False,
        'gripper_closure_latency_modelled': False, 'uncertain_collision_tube_certified': False}


def planning_bounds(config, velocity):
    report = envelope(config, velocity)
    if config.raw[KEY]['mode'] == 'experimental':
        if not report['allowed']:
            raise DelayWindowError('实测残余时延无法满足有限抓取区间或位置容差', report)
        return np.asarray(report['effective_follow_bounds_m'])
    return config.follow_bounds


def _digest_report(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def assess_reference(config, reference):
    report = envelope(config, reference.velocity_world)
    velocity = np.asarray(reference.velocity_world)
    contact = np.asarray(reference.contact_xyz)
    sync = contact-velocity*(reference.duration_s-reference.prealign_s)
    end = contact+velocity*reference.follow_s
    residual = np.asarray(report['residual_delay_bounds_s'])
    nominal = np.array([sync@config.direction, end@config.direction])
    target = nominal[:, None]+report['speed_along_belt_m_s']*residual[None, :]
    interval = np.asarray(report['original_follow_bounds_m'])
    low, high = float(min(nominal.min(), target.min())), float(max(nominal.max(), target.max()))
    reasons = report['reasons']
    if low < interval[0]-1e-9 or high > interval[1]+1e-9:
        reasons.append('residual_delay_envelope_leaves_finite_grasp_interval')
    shifted = np.array([p+velocity*d for p in (sync, contact, end) for d in residual])
    workspace = np.asarray(config.workspace)
    if np.any(shifted < workspace[:, 0]-1e-9) or np.any(shifted > workspace[:, 1]+1e-9):
        reasons.append('residual_delay_target_reference_leaves_workspace')
    report.update(allowed=not reasons, nominal_sync_coordinate_m=float(nominal[0]),
        nominal_follow_end_coordinate_m=float(nominal[1]),
        envelope_minimum_coordinate_m=low, envelope_maximum_coordinate_m=high,
        upstream_margin_m=low-float(interval[0]), downstream_margin_m=float(interval[1])-high,
        binding={'capture_s': float(reference.capture_s), 'start_s': float(reference.start_s),
            'duration_s': float(reference.duration_s), 'prealign_s': float(reference.prealign_s),
            'follow_s': float(reference.follow_s), 'brake_s': float(reference.brake_s),
            'contact_xyz': contact.tolist(), 'velocity': velocity.tolist(),
            'start_xyz': np.asarray(reference.start_tcp_world).tolist(),
            'start_quaternion': np.asarray(reference.start_quaternion_wxyz).tolist(),
            'target_quaternion': np.asarray(reference.target_quaternion_wxyz).tolist(),
            'workspace': workspace.tolist()})
    report['certificate_sha256'] = _digest_report(report)
    return report


def require_reference(config, reference):
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    report = assess_reference(config, reference)
    if config.raw[KEY]['mode'] == 'experimental' and not report['allowed']:
        raise PickRejected('残余执行时延包络未通过抓取窗口检查', details={
            'reason': 'execution_delay_window_rejected', KEY: report})
    return report


def verify_plan(config, path, reference):
    """Fail closed on stale/missing opt-in metadata, before any stream starts."""
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    fresh = require_reference(config, reference)
    saved = path.diagnostics.get(KEY)
    if not isinstance(saved, dict) or saved != fresh:
        raise PickRejected('执行时延窗口记录缺失或与当前轨迹/配置不一致', details={
            'reason': 'execution_delay_window_binding_changed'})
    return fresh

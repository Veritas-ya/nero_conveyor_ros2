"""Opt-in timing ablations; omitted configuration leaves legacy paths intact.

No device access or file I/O. All groups use the same underlying smooth
planner and configured checks. Fixed reservation never learns from history.
"""
from dataclasses import replace
import math

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


GROUPS = {'A': ('fixed', 'reject'), 'B': ('fixed', 'replan'),
          'C': ('measured', 'replan')}


def validate_experiment(raw):
    if 'timing_experiment' not in raw:
        return
    spec = raw['timing_experiment']
    allowed = {'group', 'expiry_policy', 'max_planning_rounds', 'max_delivery_attempts'}
    if not isinstance(spec, dict) or set(spec)-allowed:
        raise ValueError('timing_experiment 必须为明确的实验配置，不能含未知字段')
    group = spec.get('group')
    if not isinstance(group, str) or group not in GROUPS:
        raise ValueError('timing_experiment.group 必须为 A B C')
    if (raw.get('planning_reserve_mode'), spec.get('expiry_policy')) != GROUPS[group]:
        raise ValueError('timing_experiment 组别与预约方式或过期策略不一致')
    if (raw.get('trajectory_profile') != 'smooth' or
            raw.get('intercept_strategy') != 'synchronized' or
            raw.get('freeze_position_upstream_m') is None):
        raise ValueError('timing_experiment 需要 smooth synchronized 和显式固定冻结位置')
    spec = dict(spec)
    for key in ('max_planning_rounds', 'max_delivery_attempts'):
        value = spec.get(key, 4)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
            raise ValueError(f'timing_experiment.{key} 必须为1到4的整数')
        spec[key] = value
    raw['timing_experiment'] = spec


def plan_experiment(planner, target, state, *, cancelled, planning_fn, deadline=None):
    """Reserve fixed/measured lead independently of reject/replan policy."""
    raw, clock = planner.config.raw, planner.clock
    spec = raw['timing_experiment']
    measured = raw['planning_reserve_mode'] == 'measured'
    retry = spec['expiry_policy'] == 'replan'
    started = clock()
    limit = started + raw['planning_budget_s']
    if deadline is not None:
        if not math.isfinite(deadline):
            raise ValueError('experiment deadline must be finite')
        limit = min(limit, deadline)
    previous = float(getattr(planner, 'measured_planning_lead_s', 0.))
    post = float(getattr(planner, 'measured_post_planning_s', 0.))
    lead = previous + post if measured else float(raw['planning_reserve_s'])
    rounds, observed = [], []
    for _ in range(spec['max_planning_rounds'] if retry else 1):
        if cancelled() or clock() >= limit:
            raise PickRejected('实验调度预算耗尽或任务取消', details={
                'timing_experiment': {**spec, 'rounds': rounds}})
        begin = clock()
        try:
            plan = planning_fn(planner, target, state, cancelled=cancelled,
                reservation_s=lead, measure_late=True, global_deadline=limit)
        except PickRejected as error:
            details = error.details or {}
            misses = [f for f in details.get('failures', [])
                      if f.get('details', {}).get('reason') == 'measured_deadline_missed']
            rounds.append({'reservation_s': lead, 'wall_s': clock()-begin,
                           'result': 'missed' if misses else 'rejected',
                           'failures': details.get('failures', []),
                           'stop_reason': details.get('stop_reason')})
            if not misses or not retry or cancelled() or clock() >= limit:
                raise PickRejected(str(error), details={**details,
                    'timing_experiment': {**spec, 'rounds': rounds}}) from error
            values = [float(f['details']['measured_candidate_wall_s']) for f in misses]
            observed.extend(values)
            if measured:
                # Match the legacy measured mode's clock-rounding allowance;
                # this is not a statistical timing margin.
                lead = max(lead, max(values)+post) + 4*math.ulp(max(abs(clock()), 1.))
            # Group B deliberately retains exactly its pilot-selected lead.
            continue
        elapsed = clock()-begin
        if cancelled() or clock() >= limit:
            rounds.append({'reservation_s': lead, 'wall_s': elapsed, 'result': 'budget_exhausted'})
            raise PickRejected('实验轨迹交付时已取消或耗尽总预算', details={
                'timing_experiment': {**spec, 'rounds': rounds,
                    'last_round_wall_s': elapsed}})
        rounds.append({'reservation_s': lead, 'wall_s': elapsed, 'result': 'delivered'})
        if measured:
            planner.measured_planning_lead_s = max(previous, elapsed, *observed)
        total = clock()-started
        timing = {**spec, 'reservation_mode': raw['planning_reserve_mode'],
            'rounds': rounds, 'total_wall_s': total, 'completed_monotonic_s': clock(),
            'history_before_s': {'planning': previous, 'preparation': post},
            'estimate_for_next_target_s': (planner.measured_planning_lead_s+post
                                         if measured else lead)}
        return replace(plan, path=replace(plan.path, planning_s=total,
            diagnostics={**plan.path.diagnostics, 'timing_experiment': timing}),
            timings_s={**plan.timings_s, 'timing_experiment': timing})
    raise PickRejected('实验调度重规划轮数耗尽', details={
        'timing_experiment': {**spec, 'rounds': rounds}})


def plan_record(config, plan, now, delivery_attempt):
    """Detached low-rate diagnostics; no new checks or command decisions."""
    ref = plan.reference
    contact = float(ref.contact_xyz @ config.direction)
    follow_end = float((ref.contact_xyz+ref.velocity_world*ref.follow_s) @ config.direction)
    sync_start = contact-float(ref.velocity_world @ config.direction)*float(
        ref.duration_s-ref.prealign_s)
    return {'identity': plan.target.identity, 'delivery_attempt': delivery_attempt,
        'group': config.raw['timing_experiment']['group'],
        'capture_monotonic_s': plan.target.capture_s, 'plan_ready_monotonic_s': now,
        'send_deadline_monotonic_s': ref.start_s-config.raw['command_advance_s'],
        'remaining_send_lead_s': ref.start_s-config.raw['command_advance_s']-now,
        'contact_monotonic_s': ref.contact_s,
        'follow_end_monotonic_s': ref.start_s+ref.follow_end_s,
        'contact_coordinate_m': contact, 'sync_start_coordinate_m': sync_start,
        'follow_end_coordinate_m': follow_end,
        'upstream_margin_m': sync_start-float(config.follow_bounds[0]),
        'downstream_margin_m': float(config.follow_bounds[1])-follow_end,
        'timing': plan.path.diagnostics['timing_experiment']}

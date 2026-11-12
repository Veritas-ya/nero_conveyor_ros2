"""Offline equivalent joint-response lag; never a pure actuator-delay measurement."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .execution_delay_window import numeric, sha256


def fit_delay(command_t, command_q, feedback_t, feedback_q, *, start_s, end_s):
    """Fit one common shift on a fixed observation set, never extrapolate.

    Same convention as the archived 2026-11-11 85–90 ms analysis:
    feedback q(t) ~= actual_sent_command(t-lag). No acceleration differences.
    """
    command_t, command_q = np.asarray(command_t), np.asarray(command_q)
    feedback_t, feedback_q = np.asarray(feedback_t), np.asarray(feedback_q)
    if (command_t.ndim != 1 or len(command_t) < 5 or command_q.shape != (len(command_t), 7)
            or feedback_t.ndim != 2 or feedback_t.shape[1] != 7 or feedback_q.shape != feedback_t.shape
            or not np.isfinite(command_t).all() or not np.isfinite(command_q).all()
            or not np.all(np.diff(command_t) > 0) or not start_s < end_s):
        raise ValueError('invalid or unordered command/feedback motion data')
    rows = []
    for axis in range(7):
        mask = (np.isfinite(feedback_t[:, axis]) & np.isfinite(feedback_q[:, axis])
            & (feedback_t[:, axis] >= max(start_s, command_t[0]+.25))
            & (feedback_t[:, axis] <= min(end_s, command_t[-1]-.02)))
        t, indices = np.unique(feedback_t[mask, axis], return_index=True)
        q = feedback_q[mask, axis][indices]
        if len(t) < 5:
            raise ValueError('insufficient non-extrapolated near-contact feedback')
        rows.append((axis, t, q))
    if max(float(np.ptp(q)) for _, _, q in rows) < .005:
        raise ValueError('motion excitation too small to identify a meaningful phase lag')
    def loss(delay):
        errors = [q-np.interp(t-delay, command_t, command_q[:, axis]) for axis, t, q in rows]
        return float(np.mean(np.concatenate(errors)**2))
    # A grid avoids relying on one local basin of the bounded scalar solver.
    grid = np.linspace(-.02, .25, 271)
    index = int(np.argmin([loss(d) for d in grid]))
    if index in (0, len(grid)-1):
        raise ValueError('delay optimum reaches search boundary; fit is not identified inside range')
    fit = minimize_scalar(loss, bounds=(grid[index-1], grid[index+1]), method='bounded',
                          options={'xatol': 1e-8})
    return {'delay_s': float(fit.x), 'rms_error_rad': float(np.sqrt(fit.fun)),
        'rms_error_without_delay_rad': float(np.sqrt(loss(0.))),
        'observations_per_axis': [len(t) for _, t, _ in rows],
        'constant_axis_bias_fitted': False, 'search_range_s': [-.02, .25]}


def fit_run(run, *, advance_s, window_start_s, window_end_s):
    """Retain incomplete/unfittable attempts separately; never relabel success."""
    advance = numeric(advance_s, 'advance_s')
    low = numeric(window_start_s, 'window_start_s')
    high = numeric(window_end_s, 'window_end_s', positive=True)
    if low >= high:
        raise ValueError('fit window must be ordered')
    # Verify sign and interpolation on a known delay before real-data fits.
    t = np.arange(0., 2.001, .002)
    q = np.column_stack([np.sin(t*(1.+j*.15)+j*.15) for j in range(7)])
    fq = np.column_stack([np.interp(t-.085, t, q[:, j]) for j in range(7)])
    check = fit_delay(t, q, np.tile(t[:, None], (1, 7)), fq, start_s=.3, end_s=1.7)
    if abs(check['delay_s']-.085) > 1e-5:
        raise ValueError('known-delay sign/interpolation check failed')
    run = Path(run).resolve(strict=True)
    summary = json.loads((run/'summary.json').read_text())
    hashes = summary.get('source_hashes', {})
    original_config = (summary.get('startup_return_wait') or {}).get('config_sha256')
    if original_config is None:
        original_config = next((v for p, v in hashes.items() if Path(p).name.startswith('two_second.hardware.')
            and p.endswith('.yaml')), None)
    if original_config is None:
        raise ValueError('recorded run lacks configuration provenance')
    fitted, unfitted, evidence = [], [], []
    for cycle in sorted(run.glob('cycle_*')):
        event_path, trace_path = cycle/'events.json', cycle/'execution_trace.npz'
        files = {p.name: sha256(p) for p in (event_path, trace_path) if p.exists()}
        evidence.append({'cycle': cycle.name, 'files_sha256': files})
        try:
            events = json.loads(event_path.read_text())
            deliveries = [e['data'] for e in events if e['event'] == 'planning_delivery']
            if len(deliveries) != 1:
                raise ValueError('need one recorded delivered plan for fitting')
            delivery = deliveries[0]
            with np.load(trace_path, allow_pickle=False) as trace:
                selected = (trace['sample_stage'] == 'smooth_intercept_follow_brake') & trace['dispatch_succeeded']
                ct, cq = trace['dispatch_start_monotonic_s'][selected], trace['dispatched_q_rad'][selected]
                if len(ct) < 5:
                    raise ValueError('insufficient successfully dispatched intercept commands')
                if len(np.unique(trace['stream_id'][selected])) != 1:
                    raise ValueError('multiple intercept streams must not be mixed')
                origins = (trace['final_sample_monotonic_s'][selected]+advance
                    -trace['policy_sample_time_s'][selected])
                if not np.isfinite(origins).all() or np.ptp(origins) > .001:
                    raise ValueError('inconsistent recorded trajectory origin or clock')
                origin = float(np.median(origins))
                planned = delivery['trajectory_diagnostics']['synchronized_schedule']['join_start_s']
                if abs(origin-planned) > .001:
                    raise ValueError('supplied command advance differs from recorded trajectory timing')
                feedback = trace['feedback_sample_stage'] == 'smooth_intercept_follow_brake'
                fit = fit_delay(ct, cq, trace['feedback_joint_axis_sample_monotonic_s'][feedback],
                    trace['feedback_q_rad'][feedback], start_s=origin+low, end_s=origin+high)
                fit['window_reference_elapsed_s'] = [low, high]
                fitted.append({'cycle': cycle.name, 'fits': {'near_contact': fit}, 'files_sha256': files})
        except (ValueError, KeyError, OSError, TypeError) as error:
            unfitted.append({'cycle': cycle.name, 'reason': str(error), 'files_sha256': files})
    return {'run': str(run), 'run_summary_sha256': sha256(run/'summary.json'),
        'config_sha256': original_config, 'command_advance_s': advance,
        'velocity_policy': summary.get('velocity_policy'), 'rows': fitted, 'attempts': len(fitted),
        'archived_cycles': len(evidence), 'unfitted_cycles': unfitted, 'all_cycle_source_hashes': evidence,
        'synthetic_known_delay_tests_passed': True,
        'method': 'common no-axis-bias shift on actual sends and per-axis host-receipt feedback',
        'limitations': ['Equivalent feedback lag, not pure mechanical delay.',
            'Near-contact window can include a short portion of braking; constant lag is an approximation.',
            'Incomplete/unfittable cycles retained; fit eligibility is not a physical-success label.'],
        'hardware_connections': 0, 'commands_sent': 0}

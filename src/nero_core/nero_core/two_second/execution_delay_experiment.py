"""Build a pinned empirical lag profile, config copies and recorded-window audits.

Offline only: no device SDK, sender or hardware process is imported/launched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .execution_delay_window import (KEY, PROFILE_SCHEMA, assess_reference,
    numeric, read_profile, sha256)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def build_profile(analysis_file, output, *, condition_id, margin_s, maximum_fit_rms_rad):
    """Use EVERY fitted cycle and preserve range tails; never select by success."""
    analysis_file = Path(analysis_file).resolve(strict=True)
    source = json.loads(analysis_file.read_text())
    margin = numeric(margin_s, 'margin_s')
    maximum_rms = numeric(maximum_fit_rms_rad, 'maximum_fit_rms_rad', positive=True)
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise ValueError('condition_id is required')
    if not source.get('synthetic_known_delay_tests_passed'):
        raise ValueError('source fit must have passed its known-delay sign/interpolation check')
    run = Path(source['run']).resolve(strict=True)
    if sha256(run/'summary.json') != source['run_summary_sha256']:
        raise ValueError('original run summary changed since delay analysis')
    rows, delays = source['rows'], []
    if len(rows) < 3 or len({r['cycle'] for r in rows}) != len(rows):
        raise ValueError('need at least three unique fitted cycles')
    if len(rows) != source['attempts']:
        raise ValueError('source fit cycle count is inconsistent')
    evidence = []
    for row in rows:
        fit = row['fits']['near_contact']
        delay = numeric(fit['delay_s'], 'fitted delay')
        rms = numeric(fit['rms_error_rad'], 'fit RMS')
        if fit.get('constant_axis_bias_fitted') is not False:
            raise ValueError('use the declared no-axis-bias near-contact fit consistently')
        if rms > maximum_rms or len(fit['observations_per_axis']) != 7 or min(fit['observations_per_axis']) < 5:
            raise ValueError('fit quality insufficient for '+row['cycle']+'; do not silently discard cycles')
        for name, expected in row['files_sha256'].items():
            path = run/row['cycle']/name
            if sha256(path) != expected:
                raise ValueError('raw source changed: '+str(path))
        delays.append(delay)
        evidence.append({'cycle': row['cycle'], 'delay_s': delay, 'fit_rms_rad': rms,
            'source_files_sha256': row['files_sha256']})
    profile = {'schema': PROFILE_SCHEMA,
        'measurement': 'equivalent_host_send_to_joint_feedback_lag', 'phase': 'near_contact',
        'coverage': 'observed_range_plus_declared_margin_not_hardware_bound',
        'condition_id': condition_id, 'condition_verification': 'operator_label_not_automatic_host_identification',
        'samples': len(delays), 'observed_delay_bounds_s': [min(delays), max(delays)],
        'delay_bounds_s': [max(0., min(delays)-margin), max(delays)+margin],
        'declared_extra_margin_s': margin, 'maximum_fit_rms_rad': maximum_rms,
        'source_analysis': str(analysis_file), 'source_analysis_sha256': sha256(analysis_file),
        'source_run': str(run), 'source_run_summary_sha256': source['run_summary_sha256'],
        'training_velocity_policy': source.get('velocity_policy'),
        'training_command_advance_s': source['command_advance_s'],
        'training_config_sha256': source['config_sha256'], 'cycles': evidence,
        'unfitted_source_cycles': source.get('unfitted_cycles', []),
        'fit_scope': 'cycles_with_identifiable_near_contact_motion_only',
        'limitations': ['Observed-cycle range, not a future coverage probability or worst-case bound.',
            'Host feedback receipt is not the encoder sampling instant.',
            'A constant phase lag approximates controller dynamics and feedback return together.',
            'Changing host, load, motion or feedback configuration requires fresh validation.',
            'Not a measured gripper closure delay or a physical grasp-success label.']}
    write_new(output, profile)
    read_profile(output)
    return profile


def config_copy(base, profile_file, output, *, mode, maximum_relative_error_m):
    from .config import CellConfig
    base = Path(base).resolve(strict=True)
    profile_file = Path(profile_file).resolve(strict=True)
    profile = read_profile(profile_file)
    raw = yaml.safe_load(base.read_text())
    if KEY in raw:
        raise ValueError('start from an original site configuration, not another delay-window copy')
    if mode not in ('shadow', 'experimental'):
        raise ValueError('choose shadow or experimental')
    numeric(maximum_relative_error_m, 'maximum_relative_error_m', positive=True)
    for key in ('camera_calibration_file', 'tcp_calibration_file', 'taught_transfer_file', 'taught_return_file'):
        if raw.get(key):
            path = Path(raw[key]).expanduser()
            raw[key] = str((path if path.is_absolute() else base.parent/path).resolve(strict=True))
    raw[KEY] = {'mode': mode, 'profile_file': str(profile_file), 'profile_sha256': sha256(profile_file),
        'condition_id': profile['condition_id'], 'maximum_relative_error_m': maximum_relative_error_m}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate a temporary copy before publishing. File contents remain YAML,
    # never the loaded runtime cache; no site or installed file is touched.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.execution-delay-', dir=output.parent) as temp:
        staged = Path(temp)/'config.yaml'
        staged.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
        CellConfig.load(staged)
        with output.open('x') as stream:
            stream.write(staged.read_text())
    return {'config': str(output.resolve()), 'sha256': sha256(output), 'mode': mode,
        'command_advance_s': raw['command_advance_s'], 'profile_sha256': sha256(profile_file),
        'maximum_relative_error_m': maximum_relative_error_m, 'hardware_connections': 0, 'commands_sent': 0}


def audit_recorded(config_file, run, output):
    from .config import CellConfig
    from .synchronized_reference import SynchronizedIntercept
    from scipy.spatial.transform import Rotation
    config = CellConfig.load(config_file)
    if KEY not in config.raw:
        raise ValueError('audit requires a delay-window configuration copy')
    run = Path(run).resolve(strict=True)
    summary = json.loads((run/'summary.json').read_text())
    rows = []
    for cycle in sorted(run.glob('cycle_*')):
        path = cycle/'events.json'
        if not path.exists():
            rows.append({'cycle': cycle.name, 'status': 'missing_events'}); continue
        events = json.loads(path.read_text())
        deliveries = [e['data'] for e in events if e['event'] == 'planning_delivery']
        if not deliveries:
            rows.append({'cycle': cycle.name, 'status': 'no_planning_delivery', 'events_sha256': sha256(path)})
            continue
        for data in deliveries:
            try:
                diagnostic = data['trajectory_diagnostics']
                schedule = diagnostic['synchronized_schedule']
                submitted = [e['data'] for e in events if e['event'] == 'planning_submitted'
                    and e['data']['identity'] == data['identity']
                    and abs(e['data']['capture_monotonic_s']-data['capture_monotonic_s']) < 1e-6]
                if len(submitted) != 1:
                    raise ValueError('recorded reference must match one exact-exposure submission')
                start = np.asarray(submitted[0]['start_tcp_base'])
                grasp = np.asarray(data['selected_grasp_tcp_base'])
                quat = lambda matrix: Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
                ref = SynchronizedIntercept(capture_s=data['capture_monotonic_s'],
                    start_s=schedule['join_start_s'],
                    duration_s=diagnostic['prealign_s']+diagnostic['synchronized_s'],
                    capture_tcp_world=grasp[:3, 3], velocity_world=np.asarray(data['velocity_base_m_s']),
                    start_tcp_world=start[:3, 3], start_quaternion_wxyz=quat(start), target_quaternion_wxyz=quat(grasp),
                    prealign_s=diagnostic['prealign_s'], follow_s=diagnostic['follow_s'],
                    brake_s=diagnostic['brake_s'], rotate_first_s=diagnostic['rotate_first_s'])
                report = assess_reference(config, ref)
                rows.append({'cycle': cycle.name, 'status': 'conditional_window_pass' if report['allowed'] else 'conditional_window_reject',
                    'events_sha256': sha256(path), 'check': report})
            except (ValueError, KeyError, TypeError) as error:
                rows.append({'cycle': cycle.name, 'status': 'input_error', 'error': str(error),
                    'events_sha256': sha256(path)})
    result = {'schema': 'nero_recorded_execution_delay_audit_v1', 'run': str(run),
        'run_summary_sha256': sha256(run/'summary.json'), 'audit_config_sha256': config.source_sha256,
        'profile_sha256': config.raw[KEY]['profile_sha256'],
        'recorded_source_hashes': summary.get('source_hashes'),
        'meaning': 'retrospective check of recorded reference, not replanning or an execution outcome',
        'profile_training_velocity_policy': config.raw['_execution_delay_profile'].get('training_velocity_policy'),
        'recorded_velocity_policy': summary.get('velocity_policy'),
        'operating_condition_transfer': 'not_validated_by_this_kinematic_audit',
        'grasp_success_inferred': False, 'hardware_connections': 0, 'commands_sent': 0, 'rows': rows}
    write_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('fit')
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--advance-s', type=float, required=True)
    a.add_argument('--window-start-s', type=float, required=True)
    a.add_argument('--window-end-s', type=float, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('profile')
    a.add_argument('--analysis', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--condition-id', required=True)
    a.add_argument('--margin-s', type=float, required=True)
    a.add_argument('--maximum-fit-rms-rad', type=float, required=True)
    a = sub.add_parser('config')
    a.add_argument('--base-config', type=Path, required=True)
    a.add_argument('--profile', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--mode', choices=['shadow', 'experimental'], default='shadow')
    a.add_argument('--maximum-relative-error-m', type=float, required=True)
    a = sub.add_parser('audit')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--run', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'fit':
            from .execution_delay_identification import fit_run
            result = fit_run(args.run, advance_s=args.advance_s,
                window_start_s=args.window_start_s, window_end_s=args.window_end_s)
            write_new(args.output, result)
        elif args.command == 'profile':
            result = build_profile(args.analysis, args.output, condition_id=args.condition_id,
                margin_s=args.margin_s, maximum_fit_rms_rad=args.maximum_fit_rms_rad)
        elif args.command == 'config':
            result = config_copy(args.base_config, args.profile, args.output, mode=args.mode,
                maximum_relative_error_m=args.maximum_relative_error_m)
        else:
            result = audit_recorded(args.config, args.run, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)); return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'error': str(error), 'hardware_connections': 0, 'commands_sent': 0},
                         ensure_ascii=False)); return 2


if __name__ == '__main__':
    raise SystemExit(main())

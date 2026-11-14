"""Offline preparation and paired full-planner replay. No execution entrypoint."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import yaml

from .config import CellConfig
from .diagnostics import json_value
from .phase_timing import KEY, assess_reference, verify_plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {str(path.relative_to(root)): digest(path)
            for package in ('nero_core', 'mujoco_conveyor_sorting')
            for path in sorted((root/package).rglob('*.py'))}


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(json_value(data), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(base, specification, output, *, fixed_times=None):
    """Copy the original plus B/C; optional fixed timings are NOT auto-tuned."""
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    CellConfig.load(base)
    raw = yaml.safe_load(base.read_text())
    if KEY in raw or 'execution_delay_window' in raw:
        raise ValueError('start from an unmodified base configuration')
    for key in ('camera_calibration_file', 'tcp_calibration_file', 'taught_transfer_file', 'taught_return_file'):
        if raw.get(key):
            p = Path(raw[key]).expanduser()
            raw[key] = str((p if p.is_absolute() else base.parent/p).resolve(strict=True))
    groups = {'current': deepcopy(raw)}
    if fixed_times is not None:
        join, follow, brake = map(float, fixed_times)
        if not np.isfinite([join, follow, brake]).all() or min(join, follow, brake) <= 0:
            raise ValueError('fixed timings must be three positive finite seconds')
        fixed = deepcopy(raw)
        fixed.update(prealign_orientation_s=join+raw.get('rotate_first_s', 0.),
                     synchronized_join_candidates_s=[join], brake_duration_s=brake)
        fixed.setdefault('pick_settings', {})['full_plan_close_window_s'] = follow
        groups['fixed'] = fixed
    for mode in ('nominal', 'robust'):
        groups[mode] = deepcopy(raw)
        groups[mode][KEY] = {**deepcopy(specification), 'mode': mode}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate every file before making the requested directory visible.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'configs'; stage.mkdir()
        hashes = {}
        for group, data in groups.items():
            path = stage/(group+'.yaml')
            path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
            CellConfig.load(path)
            hashes[group] = digest(path)
        write_new(stage/'manifest.json', {'base': str(base), 'base_sha256': digest(base),
            'groups': hashes, 'fixed_times': fixed_times, 'fixed_baseline_tuned': False,
            'physical_parameters_validated': specification['calibration_status'] == 'measured',
            'invariants': ['standby_joint_rad', 'follow_interval_base_m', 'workspace_base_m',
                'joint limits', 'command_advance_s', 'calibrations', 'gripper settings except follow duration'],
            'note': 'Current/fixed and adaptive feasibility are offline results, not success rates.'})
        stage.rename(output)
    return output


def load_groups(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory/'manifest.json').read_text())
    configs = {}
    for group, sha in manifest['groups'].items():
        path = directory/(group+'.yaml')
        if digest(path) != sha:
            raise ValueError('configuration changed after preparation: '+group)
        configs[group] = CellConfig.load(path)
    base = configs['current']
    allowed = {KEY, 'prealign_orientation_s', 'synchronized_join_candidates_s', 'brake_duration_s'}
    def common(cfg):
        raw = deepcopy(cfg.raw)
        for k in allowed: raw.pop(k, None)
        raw.setdefault('pick_settings', {}).pop('full_plan_close_window_s', None)
        return json.dumps(json_value(raw), sort_keys=True)
    if any(common(cfg) != common(base) for cfg in configs.values()):
        raise ValueError('groups differ outside declared phase timing settings')
    return configs, manifest


def replay(directory, runs, output):
    sources_before = source_hashes()
    from .paper_replay import restore_input, clone_input, input_fingerprint
    from .planning import MeasuredPlanner, calibrated_model
    from .taught import TaughtSegment
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    configs, manifest = load_groups(directory)
    for cfg in configs.values():
        if cfg.raw.get('taught_playback_enabled', True):
            cfg.raw['taught_segments'] = {kind: TaughtSegment.load(cfg.raw['taught_'+kind+'_file'], cfg, kind)
                                         for kind in ('transfer', 'return')}
    rows, inputs = [], []
    rng = random.Random(20261113)
    for run in map(Path, runs):
        try:
            target, state, info = restore_input(run, configs['current'])
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            rows.append({'run': str(run), 'status': 'input_error', 'error': str(error)})
            print(run.name, 'input_error', str(error), flush=True)
            continue
        inputs.append(info)
        order = list(configs); rng.shuffle(order)
        for group in order:
            cfg = configs[group]
            copied = clone_input(target)
            if input_fingerprint(copied, state) != info['input_sha256']:
                raise ValueError('replay input changed between groups')
            model = calibrated_model(cfg)
            origin = [None]
            clock = lambda: info['submission_s']+(0. if origin[0] is None else time.monotonic()-origin[0])
            planner = MeasuredPlanner(cfg, model=model, clock=clock)
            row = {'run': str(run), 'group': group, 'input_sha256': info['input_sha256']}
            origin[0] = time.monotonic()
            try:
                plan = planner.intercept(copied, deepcopy(state))
                if KEY in cfg.raw: verify_plan(cfg, plan.path, plan.reference)
                row.update(status='plan_ready', reference_diagnostics=plan.path.diagnostics,
                    selected_times_s={'join': plan.reference.prealign_s-plan.reference.rotate_first_s,
                        'preclosure_sync': plan.reference.duration_s-plan.reference.prealign_s,
                        'follow': plan.reference.follow_s, 'brake': plan.reference.brake_s},
                    remaining_send_lead_s=plan.reference.start_s-cfg.raw['command_advance_s']-clock(),
                    # Audit every group's plan against the SAME declared models.
                    nominal_assessment=assess_reference(configs['nominal'], plan.reference),
                    robust_assessment=assess_reference(configs['robust'], plan.reference))
            except (ValueError, RuntimeError) as error:
                row.update(status='planning_rejected', error=str(error), details=getattr(error, 'details', {}),
                           attempts=planner.attempt_diagnostics)
            row['planning_wall_s'] = time.monotonic()-origin[0]
            rows.append(json_value(row))
            print(run.name, group, row['status'], row.get('selected_times_s', row.get('error')), flush=True)
    sources_after = source_hashes()
    changed_sources = sorted(k for k in set(sources_before) | set(sources_after)
                             if sources_before.get(k) != sources_after.get(k))
    report = {'schema': 'nero_phase_timing_replay_v1', 'manifest': manifest, 'inputs': inputs, 'rows': rows,
        'source_sha256_before': sources_before, 'sources_changed_during_replay': changed_sources,
        'commands_sent': 0, 'camera_connections': 0, 'can_connections': 0,
        'physical_success_inferred': False,
        'limitations': ['Only the last frozen exposure per run is reconstructed, not all attempts.',
            'Recorded submission joints and velocity are identical across groups; current config is used.',
            'Open jaw width and model-derived flange are assumed; only measured belt scene is rebuilt.',
            'New code/current calibration settings on historical input, not historical execution replay.',
            'No gripper/contact/friction/controller dynamics simulation; assumed timings are uncalibrated.',
            'Fixed timings are an illustrative comparator, not a tuned publication baseline.',
            'Plan readiness is not physical grasp success or uncertain swept-volume certification.']}
    write_new(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--base', required=True); p.add_argument('--spec', required=True)
    p.add_argument('--output', required=True); p.add_argument('--fixed-times', nargs=3, type=float)
    p = commands.add_parser('replay')
    p.add_argument('--configs', required=True); p.add_argument('--run', action='append', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare(args.base, json.loads(Path(args.spec).read_text()), args.output, fixed_times=args.fixed_times)
            print(result)
        else:
            result = replay(args.configs, args.run, args.output)
            print(json.dumps({'rows': len(result['rows']), 'commands_sent': 0}))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()

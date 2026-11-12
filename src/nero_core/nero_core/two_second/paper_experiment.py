"""Offline preparation, annotation and export for conveyor timing experiments.

This command never imports a hardware SDK, opens a device, or launches motion.
"""
import argparse
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sqlite3
import tempfile

import yaml

from .experiment_records import Ledger, digest, export, finite, initialize, utc


def prepare_session(base, output, *, fixed_reserve_s, freeze_position_m,
                    repeats=5, objects=('W01', 'W02', 'W03'), seed=20261106,
                    phase='pilot', fixed_speed_m_s=None, reservation_note,
                    history_rule, freeze_source='tracked', groups=('A', 'B', 'C')):
    from .config import CellConfig
    from .main import provenance
    from .timing_experiment import GROUPS
    base = Path(base).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError('output already exists; never overwrite an experiment')
    reserve = finite(fixed_reserve_s, 'fixed_reserve_s', positive=True)
    position = finite(freeze_position_m, 'freeze_position_m', positive=True)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if (not objects or len(set(objects)) != len(objects) or
            any(not isinstance(v, str) or not v.strip() for v in objects)):
        raise ValueError('workpiece IDs must be unique nonempty strings')
    if phase not in ('pilot', 'formal'):
        raise ValueError('phase must be pilot or formal')
    if freeze_source not in ('tracked', 'single_exposure_native_depth'):
        raise ValueError('invalid freeze source')
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError('select unique groups from A B C')
    if not reservation_note.strip() or not history_rule.strip():
        raise ValueError('document baseline reservation selection and warm-up/restart rules')
    raw = yaml.safe_load(base.read_bytes())
    if not isinstance(raw, dict) or (raw.get('trajectory_profile'), raw.get('intercept_strategy')) != ('smooth', 'synchronized'):
        raise ValueError('base must already use smooth synchronized; no automatic strategy conversion')
    if 'timing_experiment' in raw:
        raise ValueError('use the original site configuration, not a previous experiment copy')
    # These paths are interpreted relative to the original configuration.
    for key in ('camera_calibration_file', 'tcp_calibration_file', 'taught_transfer_file', 'taught_return_file'):
        if raw.get(key) is not None:
            path = Path(raw[key]).expanduser()
            raw[key] = str((path if path.is_absolute() else base.parent/path).resolve(strict=True))
    raw['freeze_position_upstream_m'] = position
    if fixed_speed_m_s is not None:
        raw['velocity_source'] = 'operator_fixed_belt_speed'
        raw['fixed_belt_speed_m_s'] = finite(fixed_speed_m_s, 'fixed speed', positive=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'nero_timing_experiment_v1', 'created_utc': utc(),
        'base_config': str(base), 'base_sha256': digest(base), 'phase': phase,
        'fixed_reservation_selection': reservation_note, 'history_rule': history_rule,
        'seed': seed, 'repeats_per_workpiece_per_group': repeats, 'objects': list(objects),
        'active_groups': list(groups),
        'groups': {}, 'hardware_connections': 0, 'commands_sent': 0,
        'freeze_source': freeze_source,
        'trial_start_definition': 'Register before introducing the numbered workpiece; include detection misses.',
        'history_control': 'Manual protocol; begin/bind never restart a robot process or reset planner history.'}
    rng = random.Random(seed)
    blocks = [(obj, i) for obj in objects for i in range(1, repeats+1)]
    rng.shuffle(blocks)
    plan = []
    for obj, repeat in blocks:
        order = list(groups)
        rng.shuffle(order)
        for group in order:
            ordinal = len(plan)+1
            plan.append({'trial_id': f'{phase[0].upper()}{ordinal:04d}', 'ordinal': ordinal,
                'group': group, 'workpiece_id': obj, 'pair_id': f'{obj}-{repeat:03d}', 'phase': phase})
    with tempfile.TemporaryDirectory(prefix='.timing-prepare-', dir=output.parent) as temporary:
        stage = Path(temporary)/'session'
        (stage/'configs').mkdir(parents=True)
        for group, (mode, expiry) in GROUPS.items():
            config_raw = deepcopy(raw)
            config_raw.update(planning_reserve_mode=mode, planning_reserve_s=reserve if mode == 'fixed' else 0.,
                timing_experiment={'group': group, 'expiry_policy': expiry,
                    'max_planning_rounds': 4, 'max_delivery_attempts': 4})
            relative = f'configs/{group}.yaml'
            path = stage/relative
            path.write_text(yaml.safe_dump(config_raw, allow_unicode=True, sort_keys=False))
            config = CellConfig.load(path)  # Read-only validation; no camera/CAN imports.
            if freeze_source == 'single_exposure_native_depth' and config.raw['velocity_source'] != 'operator_fixed_belt_speed':
                raise ValueError('single exposure requires the existing fixed speed input')
            source_hashes = provenance(config)
            source_hashes[str(output/relative)] = source_hashes.pop(str(path))
            taught = {}
            if config.raw.get('taught_playback_enabled', True):
                from .taught import TaughtSegment
                for kind in ('transfer', 'return'):
                    key = 'taught_'+kind+'_file'
                    if config.raw.get(key):
                        segment = TaughtSegment.load(config.raw[key], config, kind)
                        taught[kind] = segment.sha256
            manifest['groups'][group] = {'config': relative, 'config_sha256': digest(path),
                'reservation_mode': mode, 'expiry_policy': expiry, 'source_hashes': source_hashes,
                'velocity_policy': {'source': config.raw['velocity_source'],
                    'fixed_belt_speed_m_s': config.raw.get('fixed_belt_speed_m_s')},
                'taught_sha256': taught, 'freeze_position_upstream_m': position,
                'freeze_position_tolerance_m': config.raw['freeze_position_tolerance_m'],
                'freeze_source': freeze_source}
        with (stage/'plan.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(plan[0]))
            writer.writeheader(); writer.writerows(plan)
        manifest['planned_trials'] = len(plan)
        (stage/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        initialize(stage, plan)
        # Exclusive creation cannot overwrite another user's session. Publish
        # the manifest last; Ledger cannot open a partially published session.
        output.mkdir(exist_ok=False)
        for child in stage.iterdir():
            if child.name != 'manifest.json':
                os.rename(child, output/child.name)
        os.rename(stage/'manifest.json', output/'manifest.json')
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare', help='生成独立A/B/C配置及随机试验计划，不启动设备')
    a.add_argument('--base-config', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--fixed-reserve-s', type=float, required=True)
    a.add_argument('--freeze-position-m', type=float, required=True)
    a.add_argument('--fixed-speed-m-s', type=float)
    a.add_argument('--objects', nargs='+', default=['W01', 'W02', 'W03'])
    a.add_argument('--groups', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    a.add_argument('--repeats', type=int, default=5)
    a.add_argument('--seed', type=int, default=20261106)
    a.add_argument('--phase', choices=['pilot', 'formal'], default='pilot')
    a.add_argument('--freeze-source', choices=['tracked', 'single_exposure_native_depth'], default='tracked')
    a.add_argument('--reservation-note', required=True)
    a.add_argument('--history-rule', required=True)
    for name in ('begin', 'bind', 'finish', 'status', 'export'):
        a = sub.add_parser(name)
        a.add_argument('--session', type=Path, required=True)
        if name in ('begin', 'bind', 'finish'):
            a.add_argument('--trial', required=True)
        if name == 'begin':
            a.add_argument('--video', required=True)
            a.add_argument('--video-offset-s', type=float, required=True)
        elif name == 'bind':
            a.add_argument('--run', type=Path, required=True)
            a.add_argument('--cycle', type=int, required=True)
        elif name == 'finish':
            a.add_argument('--status', choices=['success', 'failure', 'external_interruption'], required=True)
            for stage in ('grasp', 'place', 'returned'):
                a.add_argument('--'+stage, type=int, choices=[0, 1])
            a.add_argument('--reason', default='')
            a.add_argument('--evidence-note', required=True)
            a.add_argument('--measured-speed-m-s', type=float)
            a.add_argument('--cycle-time-s', type=float, help='从录像实测的成功完整周期时间')
            a.add_argument('--termination-time-s', type=float, help='从录像实测的失败或中断终止时间')
            a.add_argument('--correction-reason')
        elif name == 'export':
            a.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    ledger = None
    try:
        if args.command == 'prepare':
            result = prepare_session(args.base_config, args.output,
                fixed_reserve_s=args.fixed_reserve_s, freeze_position_m=args.freeze_position_m,
                fixed_speed_m_s=args.fixed_speed_m_s, objects=args.objects, repeats=args.repeats,
                phase=args.phase, seed=args.seed, reservation_note=args.reservation_note,
                history_rule=args.history_rule, freeze_source=args.freeze_source, groups=args.groups)
        else:
            ledger = Ledger(args.session)
            if args.command == 'begin':
                ledger.begin(args.trial, video_file=args.video, video_offset_s=args.video_offset_s)
                result = ledger._row(args.trial)
            elif args.command == 'bind':
                ledger.bind(args.trial, args.run, args.cycle)
                result = {'trial_id': args.trial, 'program_evidence_bound': True,
                          'physical_success_inferred': False}
            elif args.command == 'finish':
                ledger.finish(args.trial, status=args.status, grasp=args.grasp, place=args.place,
                    returned=args.returned, reason=args.reason, evidence_note=args.evidence_note,
                    measured_speed_m_s=args.measured_speed_m_s, correction_reason=args.correction_reason,
                    observed_cycle_time_s=args.cycle_time_s, observed_termination_time_s=args.termination_time_s)
                result = ledger._row(args.trial)
            elif args.command == 'export':
                result = export(ledger, args.output)
            else:
                from .experiment_records import summarize
                result = summarize(ledger.rows())
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        print(json.dumps({'error': str(error), 'hardware_connections': 0, 'commands_sent': 0}, ensure_ascii=False))
        return 2
    finally:
        if ledger is not None:
            ledger.close()


if __name__ == '__main__':
    raise SystemExit(main())

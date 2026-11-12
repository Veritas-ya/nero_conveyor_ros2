"""Paired offline timing replay of archived frozen exposures; no device entrypoint.

The latest-target preview is a convenience archive, not a complete trial log.
All selected inputs and all group outcomes, including reconstruction failures,
are retained. No outcome here is a physical grasp-success label.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .experiment_records import digest, utc


def write_json(path, value):
    from .diagnostics import json_value
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n')


def matched_event(run, metadata):
    """A frame number alone is insufficient: require identity AND exposure time."""
    matches = []
    for path in sorted(Path(run).glob('cycle_*/events.json')):
        for row in json.loads(path.read_text()):
            data = row.get('data', {})
            if row.get('event') == 'planning_submitted' and data.get('identity') == metadata['identity']:
                matches.append((path, row))
    if len(matches) != 1:
        raise ValueError('frozen exposure must match exactly one planning_submitted identity')
    path, row = matches[0]
    data = row['data']
    capture = float(metadata['capture_s'])
    submission = float(data['submission_monotonic_s'])
    if (not np.isfinite([capture, submission, data['capture_monotonic_s']]).all()
            or abs(float(data['capture_monotonic_s'])-capture) > 1e-6
            or submission < capture):
        raise ValueError('matched identity has inconsistent exposure/submission timestamps')
    return path, data


def input_fingerprint(target, state):
    """Digest the numeric planner input, independently of mutable native caches."""
    h = hashlib.sha256()
    scalars = {'identity': target.identity, 'capture': target.capture_s,
        'received': target.received_s, 'device': target.device_s,
        'speed_report': target.speed_report, 'state': state}
    from .diagnostics import json_value
    h.update(json.dumps(json_value(scalars), sort_keys=True, allow_nan=False).encode())
    arrays = [target.center, target.entry_position, target.velocity, target.points,
        np.asarray(target.grasps), target.object_pose,
        target.detection['scene_points'], target.detection['belt_region']['plane_to_base'],
        target.detection['belt_region']['bounds_xy_m']]
    for value in arrays:
        array = np.ascontiguousarray(value, dtype='<f8')
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def clone_input(target):
    """Copy numeric input without copying or sharing frame-cache thread locks."""
    frame = target.detection.get('frame')
    memo = {} if frame is None else {id(frame): replace(frame)}
    return deepcopy(target, memo)


def load_configs(session):
    """Freeze effective settings and reject unequal non-ablation parameters."""
    from .config import CellConfig
    from .main import provenance
    from .taught import TaughtSegment
    session = Path(session).resolve(strict=True)
    manifest = json.loads((session/'manifest.json').read_text())
    configs, common = {}, None
    for group in ('A', 'B', 'C'):
        spec = manifest['groups'][group]
        path = session/spec['config']
        if digest(path) != spec['config_sha256']:
            raise ValueError('experiment configuration changed after preparation: '+group)
        config = CellConfig.load(path)
        if config.raw.get('timing_experiment', {}).get('group') != group:
            raise ValueError('experiment group does not match manifest')
        comparable = deepcopy(config.raw)
        for key in ('timing_experiment', 'planning_reserve_mode', 'planning_reserve_s'):
            comparable.pop(key, None)
        comparable['experiment_retry_caps'] = {k: v for k, v in config.raw['timing_experiment'].items()
            if k not in ('group', 'expiry_policy')}
        # Loaded raw fields are JSON-compatible before taught objects are added.
        key = json.dumps(comparable, sort_keys=True, default=lambda x: np.asarray(x).tolist())
        if common is not None and common != key:
            raise ValueError('groups differ in non-ablation settings')
        common = key
        if config.raw.get('taught_playback_enabled', True):
            config.raw['taught_segments'] = {kind: TaughtSegment.load(
                config.raw['taught_'+kind+'_file'], config, kind)
                for kind in ('transfer', 'return') if config.raw.get('taught_'+kind+'_file')}
        configs[group] = config
    if configs['A'].raw['planning_reserve_s'] != configs['B'].raw['planning_reserve_s']:
        raise ValueError('A/B must use the same fixed reservation')
    current = provenance(configs['A'])
    # This replay records CURRENT sources. Formal ledger provenance stays separate.
    return configs, {'session': str(session), 'manifest_sha256': digest(session/'manifest.json'),
        'current_source_hashes': current,
        'config_hashes': {g: c.source_sha256 for g, c in configs.items()},
        'taught_hashes': {k: v.sha256 for k, v in configs['A'].raw.get('taught_segments', {}).items()}}


def restore_input(run, config):
    """Retain archived object/grasps; rebuild measured belt scene once per input."""
    from scipy.spatial.transform import Rotation
    from .collision_view import restore_frame
    from .measured_rgbd import fit_plane, green_belt_points
    from .planning import calibrated_model
    from .vision import MeasuredTarget
    from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
    from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
    from mujoco_conveyor_sorting.runtime.frame_products import world_points
    run = Path(run).resolve(strict=True)
    path = run/'preview/target.npz'
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        frame = restore_frame(meta, {'frame_rgb': data['frame_rgb'], 'frame_depth_m': data['frame_depth_m']})
        points, grasps = data['points'].copy(), data['grasps'].copy()
    event_path, event = matched_event(run, meta)
    # Tracked identities end in track_id; single-exposure identities end in
    # frame sequence. Never reinterpret a track ID as a camera sequence.
    frozen_sequence = meta.get('frozen_speed', {}).get('frame_sequence', frame.header.sequence)
    if tuple(meta['identity'])[0] != frame.header.clock_epoch or frozen_sequence != frame.header.sequence:
        raise ValueError('archive identity/frozen-speed metadata does not match camera epoch/sequence')
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    if (frame.calibration_id != calibration['calibration_id']
            or frame.serial_number != calibration['serial_number']
            or not np.allclose(frame.camera_to_world_cv, calibration['depth_to_base_cv'], atol=1e-12, rtol=0)):
        raise ValueError('recorded camera calibration differs from replay configuration')
    # Preview already stores calibrated native depth. Never apply D2C a second time.
    d2c = np.asarray(meta['depth_to_color_cv'])
    model = calibrated_model(config)
    model.set_state(event['start_joints_rad'], config.settings.open_width_m)
    flange = model.flange()
    state = {'joints': event['start_joints_rad'], 'width_m': config.settings.open_width_m,
        'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}
    if config.raw['environment_collision_scope'] != 'measured_belt_surface_only':
        raise ValueError('preview lacks original full-scene segmentation; only measured belt scope is supported')
    surface = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    if surface.plane['inlier_rms_m'] > .003 or surface.plane['tilt_from_base_xy_deg'] > 8:
        raise ValueError('reconstructed belt plane exceeds production noise/tilt thresholds')
    # The frozen object was already detected online. Re-detecting would condition
    # inclusion on a second detector and can crop out upstream freeze positions.
    # Rebuild only native belt points; no synthetic plane points or depth filling.
    region = surface.region_metadata()
    native = world_points(frame, d2c)
    local = (native-surface.plane_to_base[:3, 3])@surface.plane_to_base[:3, :3]
    bounds = np.asarray(region.get('scene_bounds_xy_m', region['bounds_xy_m']))
    mask = ((frame.depth_m > 0) & np.isfinite(native).all(axis=2)
        & (np.abs(local[..., 2]) < .005)
        & (local[..., :2] >= bounds[:, 0]).all(axis=2)
        & (local[..., :2] <= bounds[:, 1]).all(axis=2))
    observation = {'belt_region': region, 'plane': surface.plane,
        'scene_points': native[mask], 'scene_pixel_indices': np.flatnonzero(mask),
        'camera_origin_base_m': frame.camera_to_world_cv[:3, 3].copy()}
    plane = np.asarray(observation['belt_region']['plane_to_base'])
    # Archived OBB is in BASE; MeasuredTarget expects the local belt-plane frame.
    box = deepcopy(meta['obb'])
    box['axes_world_columns'] = plane[:3, :3].T@np.asarray(box['axes_world_columns'])
    box['center_world_m'] = plane[:3, :3].T@(np.asarray(box['center_world_m'])-plane[:3, 3])
    observation.update(frame=frame, depth_to_color_cv=d2c, scene_state=state, obb=box)
    target = MeasuredTarget(tuple(meta['identity']), float(meta['capture_s']), frame.header.received_time,
        frame.header.source_time, frame.header.clock_epoch, frame.header.sequence,
        event['target_center_base_m'], event['velocity_base_m_s'], tuple(grasps), points,
        observation, state, event['speed_report'], event['entry_observed_position_base_m'])
    submission = float(event['submission_monotonic_s'])
    if not target.received_s <= submission:
        raise ValueError('recorded submission precedes frame receipt')
    info = {'run': str(run), 'archive_sha256': digest(path), 'event_file': str(event_path),
        'event_sha256': digest(event_path), 'identity': list(target.identity),
        'input_sha256': input_fingerprint(target, state), 'capture_s': target.capture_s,
        'submission_s': submission, 'recorded_age_at_submission_s': submission-target.capture_s,
        'speed_source': target.speed_report['source'],
        'prediction_speed_m_s': float(np.linalg.norm(target.velocity)),
        'entry_coordinate_m': float(target.entry_position@config.direction),
        'target_point_count': len(target.points), 'scene_point_count': len(observation['scene_points'])}
    return target, state, info


def replay_one(config, target, state, submission, *, planner_factory=None, now=time.monotonic):
    from .planning import MeasuredPlanner, calibrated_model
    from .timing_experiment import plan_record
    # Configuration/model startup is deliberately outside the planning clock.
    # Fresh planner per group/exposure: no cross-input history or cache training.
    config = deepcopy(config)
    source = target.speed_report['source']
    config.raw['velocity_source'] = source
    if source == 'operator_fixed_belt_speed':
        config.raw['fixed_belt_speed_m_s'] = target.speed_report['configured_speed_m_s']
    if planner_factory is None:
        model = calibrated_model(config)
        planner_factory = lambda c, clock: MeasuredPlanner(c, model=model, clock=clock)
    holder = [None]
    clock = lambda: submission if holder[0] is None else submission+now()-holder[0]
    planner = planner_factory(config, clock)
    before = input_fingerprint(target, state)
    holder[0] = now()
    try:
        plan = planner.intercept(target, state)
        finished = clock()
        record = plan_record(config, plan, finished, 1)
        result = {'status': 'plan_ready' if record['remaining_send_lead_s'] >= 0 else 'send_deadline_missed',
            'planning_wall_s': finished-submission, 'plan': record,
            'rounds': record['timing']['rounds'], 'attempts': planner.attempt_diagnostics}
    except (ValueError, RuntimeError) as error:
        details = getattr(error, 'details', {}) or {}
        result = {'status': 'planning_rejected', 'error_type': type(error).__name__, 'error': str(error),
            'details': details, 'planning_wall_s': clock()-submission,
            'rounds': details.get('timing_experiment', {}).get('rounds', []),
            'attempts': planner.attempt_diagnostics}
    after = input_fingerprint(target, state)
    if before != after:
        raise RuntimeError('planner mutated paired input; replay is invalid')
    return {**result, 'input_sha256': before, 'age_at_submission_s': submission-target.capture_s,
        'input_unchanged': True, 'cross_input_history': 'reset_per_group_per_exposure',
        'physical_success': None, 'execution_performed': False}


def run_replay(session, runs, output, *, repeats=1, seed=20261106):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    paths = [Path(p).resolve() for p in runs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('select at least one unique run')
    configs, provenance = load_configs(session)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'nero_paired_offline_replay_v1', 'created_utc': utc(),
        'provenance': provenance, 'selected_runs': list(map(str, paths)),
        'repeats': repeats, 'seed': seed, 'history_rule': 'fresh planner for each group/exposure/repeat',
        'prediction_velocity': 'recorded event velocity for all groups; not session fixed-speed input',
        'scene': 'reconstructed once from frozen native RGB-D; original plane/scene not archived',
        'robot_state': 'recorded submission joints; assumed configured open width; model-derived flange',
        'selection_limit': 'one last frozen target per run; selected after detection, not all physical trials',
        'clock': 'recorded exposure-to-submission age plus current replay planning wall time',
        'excluded': ['live detection throughput', 'original capture-paired feedback',
            'live SDK preparation and command dispatch', 'grasp/place/return', 'cross-target learning'],
        'camera_connections': 0, 'can_connections': 0, 'commands_sent': 0, 'execution_performed': False,
        'physical_success_rate': None, 'status': 'running'}
    write_json(output/'manifest.json', manifest)  # Freeze selection BEFORE any result.
    rng, results = random.Random(seed), []
    for index, run in enumerate(paths, 1):
        input_id = f'I{index:03d}'
        try:
            target, state, info = restore_input(run, configs['A'])
            write_json(output/(input_id+'_input.json'), info)
            np.savez_compressed(output/(input_id+'_input.npz'), points=target.points,
                grasps=np.asarray(target.grasps), velocity=target.velocity,
                center=target.center, entry_position=target.entry_position,
                scene_points=target.detection['scene_points'],
                plane_to_base=target.detection['belt_region']['plane_to_base'],
                bounds_xy_m=target.detection['belt_region']['bounds_xy_m'])
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            for repeat in range(1, repeats+1):
                for group in ('A', 'B', 'C'):
                    row = {'input_id': input_id, 'run': str(run), 'repeat': repeat, 'group': group,
                        'status': 'input_rejected', 'error_type': type(error).__name__, 'error': str(error),
                        'physical_success': None, 'execution_performed': False}
                    results.append(row)
                    write_json(output/f'{input_id}_R{repeat:03d}_{group}.json', row)
            continue
        for repeat in range(1, repeats+1):
            order = list(configs); rng.shuffle(order)
            for position, group in enumerate(order, 1):
                # Separate copies prevent any transient native cache sharing.
                try:
                    copied = clone_input(target)
                    expected = info['input_sha256']
                    if input_fingerprint(copied, state) != expected:
                        raise RuntimeError('cloned group input differs from frozen input')
                    result = replay_one(configs[group], copied, deepcopy(state), info['submission_s'])
                    if result['input_sha256'] != expected:
                        raise RuntimeError('group result differs from paired input fingerprint')
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                    result = {'status': 'replay_error', 'error_type': type(error).__name__,
                        'error': str(error), 'physical_success': None, 'execution_performed': False}
                row = {'input_id': input_id, 'run': str(run), 'repeat': repeat, 'group': group,
                    'group_order': order, 'order_position': position, **result}
                results.append(row)
                write_json(output/f'{input_id}_R{repeat:03d}_{group}.json', row)
                print(json.dumps({k: row[k] for k in ('input_id', 'repeat', 'group', 'status')},
                                 ensure_ascii=False), flush=True)
    fields = ['input_id', 'run', 'repeat', 'group', 'order_position', 'status',
        'input_sha256', 'age_at_submission_s', 'planning_wall_s', 'error']
    with (output/'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    summary = {'selected_inputs': len(paths), 'rows': len(results),
        'groups': {g: dict(Counter(r['status'] for r in results if r['group'] == g)) for g in configs},
        'physical_success_rate': None, 'statistical_scope': 'exploratory paired planner replay only'}
    write_json(output/'summary.json', summary)
    manifest.update(status='completed', summary_sha256=digest(output/'summary.json'))
    write_json(output/'manifest.json', manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261106)
    args = parser.parse_args(argv)
    try:
        result = run_replay(args.session, args.runs, args.output, repeats=args.repeats, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'error': str(error), 'hardware_connections': 0, 'commands_sent': 0},
                         ensure_ascii=False)); return 2


if __name__ == '__main__':
    raise SystemExit(main())

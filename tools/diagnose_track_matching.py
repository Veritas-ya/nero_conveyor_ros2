#!/usr/bin/env python3
"""Replay recorded RGB-D through the production tracker, without devices.

This diagnoses tracking, not geometry freeze or trajectory delivery. A sparse
archive cannot reproduce a live 30 Hz tracking session; timestamps are kept.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.perception import make_meter
from nero_core.two_second.velocity_source import is_fixed, with_fixed_belt_speed
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def stats(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values:
        return None
    return {'count': len(values), 'mean': float(np.mean(values)),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-belt-speed-m-s', type=float,
                        help='Use the given fixed speed as the association prior, in m/s')
    parser.add_argument('--isolate-depth-gaps', action=argparse.BooleanOptionalAction, default=None,
                        help='Override the candidate isolation setting for offline comparison')
    parser.add_argument('--diagnostic-workspace-y-max', type=float,
                        help='Offline-only override of detector Y maximum, metres')
    args = parser.parse_args()
    paths = sorted((args.run / 'rgbd_frames').glob('*.npz'))
    if not paths:
        parser.error('No saved rgbd_frames found')
    config = CellConfig.load(args.config)
    if args.fixed_belt_speed_m_s is not None:
        config = with_fixed_belt_speed(config, args.fixed_belt_speed_m_s)
    if args.isolate_depth_gaps is not None:
        config = replace(config, raw={**config.raw, 'isolate_boundary_depth_gaps': args.isolate_depth_gaps})
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    first, d2c = read(paths[0], calibration)
    start = time.perf_counter()
    surface = BeltSurface.from_frame(first, d2c, fit_plane=fit_plane,
                                     green_belt_points=green_belt_points)
    surface_ms = 1000 * (time.perf_counter() - start)
    meter = make_meter(config, surface)
    if args.diagnostic_workspace_y_max is not None:
        lower = float(meter.parameters['workspace_y'][0])
        if not np.isfinite(args.diagnostic_workspace_y_max) or args.diagnostic_workspace_y_max <= lower:
            parser.error('diagnostic workspace Y maximum must be finite and greater than its minimum')
        meter.parameters['workspace_y'] = [lower, args.diagnostic_workspace_y_max]
    meter._geometry(first, d2c)
    tracker = meter._tracker
    original_update = tracker.update
    measured = {}

    def audited_update(detections, stamp):
        # Record the same physical/statistical gates used by the tracker,
        # without modifying detection, association, or clock values.
        pairs = []
        for tr in tracker.tracks:
            age = float(stamp - tr.stamp)
            if age < 0 or age > tracker.lost_timeout or tr.misses > tracker.lost_frames:
                continue
            physical_gate = tracker.gate_radius + tracker.max_vel * age
            for index, detection in enumerate(detections):
                pos = np.asarray(detection['centroid_base'], dtype=float)
                innovation = tr.motion.innovation(pos[:2], stamp)
                if innovation is None:
                    continue
                mahal, distance = map(float, innovation[:2])
                pairs.append({'track_id': int(tr.id), 'detection_index': index,
                    'distance_m': distance, 'distance_gate_m': physical_gate,
                    'innovation': mahal, 'innovation_gate': float(tr.motion.innovation_gate),
                    'within_gates': bool(distance <= physical_gate and mahal <= tr.motion.innovation_gate)})
        measured['pairs'] = pairs
        measured['detections'] = [np.asarray(d['centroid_base'], dtype=float).tolist() for d in detections]
        before = time.perf_counter()
        result = original_update(detections, stamp)
        measured['association_wall_ms'] = (time.perf_counter() - before) * 1000
        return result

    tracker.update = audited_update
    rows, identities = [], {}
    began = time.perf_counter()
    for path in paths:
        frame, d2c = read(path, calibration)
        measured.clear()
        t0 = time.perf_counter()
        report = meter.update(frame, d2c)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        peer = meter.latest_snapshot
        diag = (report or {}).get('diagnostics', {})
        tracks = []
        for t in (() if peer is None else peer.tracks):
            identity = f'{peer.clock_epoch}:{peer.track_generation}:{t.track_id}'
            row = {'identity': identity, 'confirmed': bool(t.confirmed),
                   'misses': int(t.misses), 'samples': int(t.sample_count),
                   'span_s': float(t.span_s), 'speed_valid': bool(t.speed_valid),
                   'velocity_base_m_s': t.velocity_base_m_s,
                   'observed_position_base_m': list(t.observed_position_base)}
            tracks.append(row)
            # Only current, genuinely observed tracks are evidence of detection.
            if peer.current and not t.misses:
                info = identities.setdefault(identity, {'first_sequence': int(frame.header.sequence),
                    'first_source_s': float(frame.header.source_time), 'first_confirmed_sequence': None,
                    'first_confirmed_source_s': None, 'observed_frames': 0})
                info['observed_frames'] += 1
                if t.confirmed and info['first_confirmed_sequence'] is None:
                    info['first_confirmed_sequence'] = int(frame.header.sequence)
                    info['first_confirmed_source_s'] = float(frame.header.source_time)
                    info['first_confirm_latency_s'] = info['first_confirmed_source_s'] - info['first_source_s']
        valid = [t for t in tracks if t['confirmed'] and not t['misses'] and
                 (is_fixed(config) or (t['speed_valid'] and t['velocity_base_m_s'] is not None
                   and t['span_s'] >= config.raw['minimum_speed_span_s']
                   and float(np.asarray(t['velocity_base_m_s']) @ config.direction) > 0))]
        if peer is None or not peer.current:
            gate = 'meter_not_current'
        elif not tracks:
            gate = 'no_tracks'
        elif len(valid) == 0:
            gate = 'no_qualified_tracks'
        elif len(valid) > 1:
            gate = 'multiple_qualified_tracks'
        else:
            gate = 'one_qualified_track'
        rows.append({'sequence': int(frame.header.sequence), 'source_s': float(frame.header.source_time),
            'current': bool(peer and peer.current), 'reason': None if peer is None else peer.reason,
            'gate': gate, 'qualified_count': len(valid), 'tracks': tracks,
            'detections_base_m': measured.get('detections', []), 'association_pairs': measured.get('pairs', []),
            'association_ambiguity': (report or {}).get('association_ambiguity'),
            'association_wall_ms': measured.get('association_wall_ms'), 'meter_wall_ms': elapsed_ms,
            'registration_ms': diag.get('registration_ms'), 'detection_ms': diag.get('detection_ms'),
            'tracking_ms': diag.get('tracking_ms'),
            'candidate_count': diag.get('candidate_count'),
            'depth_admission': diag.get('depth_admission'),
            'candidate_depth_diagnostics': diag.get('candidate_depth_diagnostics')})
    intervals = [b['source_s'] - a['source_s'] for a, b in zip(rows, rows[1:])
                 if b['source_s'] > a['source_s']]
    gaps = [b['sequence'] - a['sequence'] for a, b in zip(rows, rows[1:])]
    summary = {'mode': 'offline_camera_archive_replay', 'run': str(args.run.resolve()),
        'config': str(args.config.resolve()), 'velocity_source': config.raw.get('velocity_source'),
        'isolate_boundary_depth_gaps': config.raw.get('isolate_boundary_depth_gaps', False),
        'detector_workspace_y_m': meter.parameters['workspace_y'],
        'fixed_belt_speed_m_s': config.raw.get('fixed_belt_speed_m_s'),
        'camera_connections': 0, 'robot_connections': 0, 'robot_commands': 0,
        'frames': len(rows), 'sequence_increment': stats(gaps), 'source_interval_ms': stats([v*1000 for v in intervals]),
        'frame_reasons': dict(Counter(str(x['reason']) for x in rows)),
        'gate_counts': dict(Counter(x['gate'] for x in rows)),
        'meter_wall_ms': stats(x['meter_wall_ms'] for x in rows),
        'registration_ms': stats(x['registration_ms'] for x in rows),
        'detection_ms': stats(x['detection_ms'] for x in rows),
        'tracking_ms': stats(x['tracking_ms'] for x in rows),
        'association_wall_ms': stats(x['association_wall_ms'] for x in rows),
        'identity_confirm_latency_ms': stats(1000*x['first_confirm_latency_s'] for x in identities.values()
                                            if 'first_confirm_latency_s' in x),
        'identity_count': len(identities), 'confirmed_identity_count': sum('first_confirm_latency_s' in x for x in identities.values()),
        'surface_prepare_ms': surface_ms, 'replay_wall_s': time.perf_counter() - began,
        'limits': 'Track identities are not physical object labels. Capture gaps and changed calibration/config can alter association. Instrumentation adds timing overhead. Counts include empty-belt frames. No geometry freeze, robot feedback, planning or execution is replayed.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps({'summary': summary, 'identities': identities}, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'frames.jsonl').open('w') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

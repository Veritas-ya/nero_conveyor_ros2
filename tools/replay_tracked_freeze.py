#!/usr/bin/env python3
"""Replay tracked freeze geometry from recorded RGB-D without any devices.

Tracks are same-exposure results of diagnose_track_matching.py.  Production
geometry runs with explicitly simulated standby feedback and a reconstructed
host-monotonic clock; this is not evidence of hardware feedback or motion.
"""
import argparse
from concurrent.futures import Future
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.measured_rgbd import depth_capture_timing, fit_plane, green_belt_points
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.reference import EntryNotReady
from nero_core.two_second.velocity_source import with_fixed_belt_speed, prediction_velocity
from nero_core.two_second.vision import LiveVision
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True, help='Same-config per-frame tracking replay JSONL')
    parser.add_argument('--speed-m-s', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    config = with_fixed_belt_speed(CellConfig.load(args.config), args.speed_m_s)
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    paths = sorted((args.run / 'rgbd_frames').glob('*.npz'))
    if not paths:
        parser.error('no recorded frames')
    rows = [json.loads(line) for line in args.tracks.open()]
    by_sequence = {int(path.stem.split('_')[-1]): path for path in paths}
    first, d2c = read(paths[0], calibration)
    start = time.perf_counter()
    surface = BeltSurface.from_frame(first, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    surface_s = time.perf_counter() - start
    start = time.perf_counter()
    model = calibrated_model(config)
    model.set_state(config.standby, config.settings.open_width_m)
    flange = model.flange()
    state = {'joints': config.standby.tolist(), 'width_m': config.settings.open_width_m,
             'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}
    model_s = time.perf_counter() - start
    vision = LiveVision(None, calibration, config, model, lambda: state)
    vision.geometry_model = model
    attempts = []
    success = None
    try:
        for row in rows:
            if row['gate'] != 'one_qualified_track':
                continue
            candidates = [t for t in row['tracks'] if t['confirmed'] and not t['misses']]
            if len(candidates) != 1:
                continue
            source = candidates[0]
            position = np.asarray(source['observed_position_base_m'], dtype=float)
            velocity = prediction_velocity(config, None, surface.normal_base)
            original_frame, d2c = read(by_sequence[row['sequence']], calibration)
            timing = depth_capture_timing(original_frame)
            delivery_s = max(0., float(timing['depth_delivery_delay_estimate_s']))
            tracking_s = float(row['meter_wall_ms']) / 1000.
            receive_s = time.monotonic() - tracking_s
            capture_s = receive_s - delivery_s
            # Keep original device epoch/sequence; only rebase the monotonic
            # timestamps to the current process so freshness checks are real.
            header = replace(original_frame.header, sample_time=receive_s, received_time=receive_s)
            frame = replace(original_frame, header=header)
            track = SimpleNamespace(track_id=int(source['identity'].split(':')[-1]),
                observed_position_base=position, velocity_base_m_s=None, sample_count=source['samples'])
            peer = SimpleNamespace(clock_epoch=frame.header.clock_epoch,
                                   track_generation=int(source['identity'].split(':')[1]))
            upstream_m = float(config.follow_bounds[0] - position @ config.direction)
            try:
                vision.entry_time(capture_s, position, velocity, time.monotonic())
            except EntryNotReady:
                continue
            t0 = time.monotonic()
            result = vision._prepare_target(frame, d2c, peer, track, state, capture_s, 0,
                                            surface, .03)
            elapsed = time.monotonic() - t0
            progress = vision.geometry_progress.snapshot()
            # Exercise the actual result handoff and freshness/identity gate,
            # assuming the tracker generation did not change during geometry.
            if result:
                completed = Future()
                completed.set_result(result)
                vision.geometry_future = completed
                vision.clock_epoch = peer.clock_epoch
                vision.progress_snapshot = peer
                vision._finish_geometry()
            delivered = vision.poll() if result else None
            attempt = {'sequence': row['sequence'], 'identity': source['identity'],
                'position_base_m': position.tolist(), 'upstream_of_entry_m': upstream_m,
                'original_capture_monotonic_s': timing['depth_sample_monotonic_estimate'],
                'original_device_time_s': original_frame.header.source_time,
                'estimated_original_delivery_s': delivery_s, 'offline_tracking_wall_s': tracking_s,
                'geometry_wall_s': elapsed, 'reconstructed_observation_age_s': time.monotonic()-capture_s,
                'status': 'target_ready' if delivered is not None else 'not_ready',
                'last_blocker': None if delivered is not None else vision.geometry_progress.snapshot().get('last_blocker')}
            if delivered is not None:
                target = delivered
                attempt.update(point_count=len(target.points), grasp_candidates=len(target.grasps),
                    cloud_center_base_m=target.center.tolist(),
                    tracking_to_cloud_xy_m=float(np.linalg.norm(target.center[:2]-position[:2])),
                    grasp_positions_base_m=[g[:3, 3].tolist() for g in target.grasps])
                success = target
            attempts.append(attempt)
            print(json.dumps(json_value(attempt), ensure_ascii=False), flush=True)
            if success is not None:
                break
    finally:
        vision.geometry_pool.shutdown(wait=True)
    args.output.mkdir(parents=True)
    if success is not None:
        np.savez(args.output/'frozen_target.npz', points_base_m=success.points,
                 grasps_base=np.asarray(success.grasps), entry_position_base_m=success.entry_position,
                 velocity_base_m_s=success.velocity)
    report = {'mode': 'offline_tracked_freeze', 'config': str(args.config.resolve()),
        'config_sha256': config.source_sha256, 'run': str(args.run.resolve()),
        'track_replay': str(args.tracks.resolve()), 'speed_source': 'user_supplied_fixed_speed',
        'speed_m_s': args.speed_m_s, 'surface_preparation_wall_s': surface_s,
        'model_startup_wall_s': model_s, 'robot_state': 'configured_standby_model_only_not_recorded',
        'clock': 'original device time preserved; host monotonic rebased with recorded camera delivery and replay tracking times',
        'camera_connections': 0, 'can_connections': 0, 'commands_sent': 0,
        'physical_motion_tested': False, 'trajectory_planning_tested': False,
        'limits': 'Does not replay scheduling contention, actual robot feedback or asynchronous camera timing; tracking JSONL must come from the same current config and exposures.',
        'status': 'target_ready' if success is not None else 'no_target_ready', 'attempts': attempts}
    (args.output/'summary.json').write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2)+'\n')
    return 0 if success is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())

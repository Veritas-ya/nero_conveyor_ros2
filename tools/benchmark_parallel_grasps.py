"""Offline-only serial/four-worker OBB planning benchmark on recorded RGB-D."""
import argparse
from concurrent.futures import ThreadPoolExecutor, CancelledError
from dataclasses import replace
import json
from pathlib import Path
import resource
import threading
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.velocity_source import with_fixed_belt_speed, prediction_velocity, speed_report
from nero_core.two_second.vision import MeasuredTarget, obb_grasps
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.single_pick import detect_target


def setup_model(cfg):
    model = calibrated_model(cfg)
    model.set_state(cfg.standby, cfg.settings.open_width_m)
    flange = model.flange()
    state = {'joints': cfg.standby.tolist(), 'width_m': cfg.settings.open_width_m,
             'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}
    return model, state


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--recorded-standby-config', type=Path)
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--speed', type=float, required=True)
    p.add_argument('--mode', choices=('serial', 'thread4'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trajectory-npz', type=Path)
    a = p.parse_args()
    cfg = with_fixed_belt_speed(CellConfig.load(a.config), a.speed)
    # Evaluate exactly the same 2 s duration for each mode. Longer duration
    # fallbacks remain a separate, sequential scheduling decision.
    cfg = replace(cfg, raw={**cfg.raw, 'approach_duration_candidates_s': [2.0]})
    startup = time.monotonic()
    models_states = [setup_model(cfg) for _ in range(4 if a.mode == 'thread4' else 1)]
    model, state = models_states[0]
    pool = ThreadPoolExecutor(max_workers=4) if a.mode == 'thread4' else None
    if pool:
        ready = threading.Barrier(5)
        warms = [pool.submit(ready.wait) for _ in range(4)]
        ready.wait()
        for f in warms:
            f.result()
    startup_s = time.monotonic() - startup
    calibration = load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
    frames = sorted((a.run / 'rgbd_frames').glob('*.npz'))
    frame_path = next(x for x in frames if x.stem.endswith(f'_{a.frame:09d}'))
    first, d2c = read(frames[0], calibration)
    surface = BeltSurface.from_frame(first, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    frame, d2c = read(frame_path, calibration)
    capture = depth_capture_timing(frame)['depth_sample_monotonic_estimate']
    detection_model, detection_state = model, state
    if a.recorded_standby_config:
        detection_model, detection_state = setup_model(CellConfig.load(a.recorded_standby_config))
    t = time.monotonic()
    ob = detect_target(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points,
                       target_shape='simple', robot_model=detection_model, robot_state=detection_state,
                       measured_surface=surface, maximum_target_height_m=cfg.raw['maximum_target_height_m'])
    grasps, box = obb_grasps(ob['object_points'], ob['belt_region']['plane_to_base'], cfg.canonical_tcp)
    detection_s = time.monotonic() - t
    ob.update(frame=frame, depth_to_color_cv=d2c, scene_state=detection_state, obb=box)
    # Each planner owns its mutable robot/MuJoCo data; shared scene is only read.
    for key in ('scene_points', 'scene_pixel_indices'):
        if isinstance(ob.get(key), np.ndarray):
            ob[key].setflags(write=False)
    center = np.median(ob['object_points'], axis=0)
    velocity = prediction_velocity(cfg, None, ob['plane']['normal_base'])
    target = MeasuredTarget((frame.header.clock_epoch, 1, 1), capture, frame.header.received_time,
                            frame.header.source_time, frame.header.clock_epoch, frame.header.sequence,
                            center, velocity, tuple(grasps), ob['object_points'], ob, state,
                            speed_report(cfg, SimpleNamespace(sample_count=0), velocity), center)
    model.check_feedback(state)
    start_rotation = model.tcp()[:3, :3].copy()
    ranked = sorted(enumerate(target.grasps),
                    key=lambda g: Rotation.from_matrix(start_rotation.T @ g[1][:3, :3]).magnitude())
    assert len(ranked) == 4
    submission = capture + frame.header.received_time - capture + detection_s
    wall_origin = time.monotonic()
    clock = lambda: submission + time.monotonic() - wall_origin
    planners = [MeasuredPlanner(cfg, model=m, clock=clock) for m, _ in models_states]
    cancel = threading.Event()
    plan_start = time.monotonic()
    def attempt(rank):
        started = time.monotonic()
        planner = planners[rank]
        try:
            plan = planner.intercept(replace(target, grasps=(ranked[rank][1],)), state, cancelled=cancel.is_set)
            return {'rank': rank, 'original_index': ranked[rank][0], 'status': 'delivered',
                    'finished_s': time.monotonic()-plan_start, 'worker_wall_s': time.monotonic()-started,
                    'attempts': planner.attempt_diagnostics, 'plan': plan}
        except (ValueError, RuntimeError) as e:
            return {'rank': rank, 'original_index': ranked[rank][0], 'status': 'rejected',
                    'finished_s': time.monotonic()-plan_start, 'worker_wall_s': time.monotonic()-started,
                    'error': str(e), 'details': getattr(e, 'details', {}), 'attempts': planner.attempt_diagnostics}
    winner = None
    results = []
    selected_at = None
    if pool:
        futures = [pool.submit(attempt, i) for i in range(4)]
        # Resolve in existing priority order, never choose the fastest low-priority pose.
        for f in futures:
            result = f.result()
            if result['status'] == 'delivered':
                winner = result
                selected_at = time.monotonic()-plan_start
                cancel.set()
                break
        for f in futures:
            try:
                results.append(f.result())
            except CancelledError:
                pass
        drained_at = time.monotonic()-plan_start
        pool.shutdown(wait=True)
    else:
        planner = planners[0]
        try:
            plan = planner.intercept(target, state)
            selected_at = time.monotonic()-plan_start
            winner = {'status': 'delivered', 'plan': plan,
                      'rank': next(i for i, (_, g) in enumerate(ranked) if np.allclose(g, plan.selected_grasp)),
                      'finished_s': selected_at}
        except (ValueError, RuntimeError) as e:
            results = [{'status': 'rejected', 'error': str(e), 'details': getattr(e, 'details', {})}]
        drained_at = time.monotonic()-plan_start
        results += [{'status': x.get('status'), 'rank': x.get('candidate_index'),
                     'wall_s': x.get('wall_s'), 'failed_phase': x.get('failed_phase'),
                     'reason': x.get('reason')} for x in planner.attempt_diagnostics]
    plan = winner['plan'] if winner else None
    if plan is not None and a.trajectory_npz:
        times = np.linspace(0., plan.path.duration, 501)
        a.trajectory_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.trajectory_npz, times=times, qva=np.asarray([plan.path.sample(t) for t in times]))
    report = {'mode': a.mode, 'source_frame': str(frame_path), 'offline_only': True,
              'camera_connections': 0, 'can_connections': 0, 'commands_sent': 0,
              'robot_state': 'configured_standby_not_measured',
              'observation_age_at_submission_s': submission-capture,
              'detection_and_obb_wall_s': detection_s, 'model_and_worker_startup_s': startup_s,
              'duration_candidates_s': [2.0], 'configured_speed_m_s': a.speed,
              'priority_original_indices': [g[0] for g in ranked],
              'status': 'delivered' if winner else 'no_2s_candidate',
              'winning_priority_rank': winner['rank'] if winner else None,
              'selected_grasp': plan.selected_grasp if plan else None,
              'time_to_priority_selection_s': selected_at, 'time_until_workers_drained_s': drained_at,
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'attempts': [{k: v for k, v in result.items() if k != 'plan'} for result in results]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(json_value({k: v for k, v in report.items() if k != 'attempts'}), ensure_ascii=False))


if __name__ == '__main__':
    run()

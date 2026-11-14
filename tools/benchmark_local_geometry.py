#!/usr/bin/env python3
"""Paired saved-frame benchmark of full/local detection; never opens hardware."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/nero_core'))
from nero_core.two_second.collision_view import restore_frame
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.fixed_freeze import NativeDepthLocator, fixed_freeze_detection_surface
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.vision import obb_grasps
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration, apply_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
from mujoco_conveyor_sorting.runtime.single_pick import detect_target, PickRejected, TargetNotReady


def load_case(path):
    run = path.parents[2]
    cfg = CellConfig.load(run/'effective.yaml')
    # Compare both methods under the current independent interface size range.
    cfg.raw['maximum_target_length_m'] = .20
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        frame = restore_frame(meta, {k:archive[k] for k in ('frame_rgb','frame_depth_m')})
    d2c = np.asarray(meta['depth_to_color_cv'])
    if path.parent.name == 'rgbd_frames':
        frame = apply_runtime_depth_calibration(frame,
            load_runtime_depth_calibration(cfg.raw['camera_calibration_file']), d2c)
    surface = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    events = [e['data'] for f in sorted((run/'hardware').glob('cycle_*/events.json'))
              for e in json.loads(f.read_text()) if e['event']=='planning_submitted']
    matched = next((e for e in events if e['identity'][-1] == frame.header.sequence
                    and e['identity'][0] == frame.header.clock_epoch), None)
    state_source = 'matching_planning_submission' if matched is not None else 'nearest_planning_submission'
    submission = matched or min(events, key=lambda e:abs(e['identity'][-1]-frame.header.sequence))
    state = {'joints':submission['start_joints_rad'], 'width_m':cfg.settings.open_width_m}
    locator_model = calibrated_model(cfg)
    candidates, _ = NativeDepthLocator(cfg, surface, frame).locate(
        frame, robot_model=locator_model, robot_state=state)
    if len(candidates) != 1:
        raise ValueError(f'locator candidate count {len(candidates)}')
    kwargs = dict(fit_plane=fit_plane, green_belt_points=green_belt_points,
        target_shape='simple', maximum_target_height_m=cfg.raw['maximum_target_height_m'],
        maximum_target_length_m=.20, measured_surface=fixed_freeze_detection_surface(surface),
        target_seed_pixel_indices=candidates[0].seed_pixel_indices,
        robot_state=state, scene_scope='measured_belt_surface_only')
    return cfg, frame, d2c, kwargs, state_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--frames', type=Path, nargs='*')
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    if args.repeats < 1: parser.error('repeats must be positive')
    paths = args.frames or sorted((ROOT/'runs/phase_timing_test').glob('*/hardware/preview/target.npz')) + [
        ROOT/'runs/phase_timing_test/20261113_222700_233202/hardware/rgbd_frames/epoch_000_frame_000000137.npz',
        ROOT/'runs/phase_timing_test/20261113_221303_938167/hardware/rgbd_frames/epoch_000_frame_000000332.npz']
    report = {'hardware_connections':0,'commands_sent':0,'physical_success_inferred':False,
        'repeats':args.repeats,'rows':[], 'limitations':[
            'Local CPU timing, not historical online load. Whole-frame copies and setup excluded equally.',
            'Per-frame belt plane refitted; planning-submission joints and open jaw assumed for self filtering.',
            'Both groups use current 20cm detection range; no failed input is silently dropped.',
            'This measures detection plus required full scene reconstruction, not exposure-to-command latency.']}
    for path in paths:
        row = {'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            cfg, frame, d2c, kwargs, state_source = load_case(path)
            row.update(sequence=frame.header.sequence, state_source=state_source)
            models = {g:calibrated_model(cfg) for g in ('full','local')}
            timings = {g:[] for g in models}
            outputs, errors = {}, {}
            # One discarded warm-up per group; every exposure cache is fresh.
            for repeat in range(args.repeats+1):
                for group in (('full','local') if repeat%2 == 0 else ('local','full')):
                    fresh = replace(frame)
                    before = time.perf_counter()
                    try:
                        kw = {**kwargs,'robot_model':models[group]}
                        output = (detect_target(fresh,d2c,**kw) if group=='full' else
                                  detect_seed_local(fresh,d2c,detector=detect_target,**kw))
                        outputs[group] = output
                    except (PickRejected, TargetNotReady) as error:
                        errors[group] = {'type':type(error).__name__,'error':str(error),
                                         'details':getattr(error,'details',{})}
                    elapsed = time.perf_counter()-before
                    if repeat: timings[group].append(elapsed)
            row['timings_s'] = {g:{'samples':values,'median':float(np.median(values)),
                                   'max':max(values)} for g,values in timings.items()}
            row['speedup'] = row['timings_s']['full']['median']/row['timings_s']['local']['median']
            if errors:
                row.update(status='detector_rejected', errors=errors)
            else:
                a,b = outputs['full'],outputs['local']
                ga,_ = obb_grasps(a['object_points'],a['belt_region']['plane_to_base'],cfg.canonical_tcp)
                gb,_ = obb_grasps(b['object_points'],b['belt_region']['plane_to_base'],cfg.canonical_tcp)
                row.update(status='compared', local_processing=b['local_processing'],
                    full_points=len(a['object_points']),local_points=len(b['object_points']),
                    mask_different_pixels=int(np.count_nonzero(a['mask']!=b['mask'])),
                    object_points_identical=bool(np.array_equal(a['object_points'],b['object_points'])),
                    scene_pixels_identical=bool(np.array_equal(a['scene_pixel_indices'],b['scene_pixel_indices'])),
                    scene_points_identical=bool(np.array_equal(a['scene_points'],b['scene_points'])),
                    actual_grasps_max_abs_difference=float(np.max(np.abs(np.asarray(ga)-np.asarray(gb)))),
                    width_difference_m=float(b['width']-a['width']),height_difference_m=float(b['height']-a['height']),
                    footprint_difference_m=(np.asarray(b['footprint_m'])-a['footprint_m']).tolist())
        except Exception as error:
            row.update(status='input_reconstruction_failed',error=f'{type(error).__name__}: {error}')
        report['rows'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('file','sha256','timings_s')},default=json_value,ensure_ascii=False),flush=True)
    files = ['local_target_geometry.py','single_pick.py','visible_target.py']
    report['source_sha256'] = {f:hashlib.sha256((ROOT/'src/nero_core/mujoco_conveyor_sorting/runtime'/f).read_bytes()).hexdigest() for f in files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__': main()

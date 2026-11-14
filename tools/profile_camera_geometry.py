#!/usr/bin/env python3
"""Camera-only perception timing. No CAN, robot feedback, planner or executor."""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/nero_core'))


class Timings:
    """Nested timings: exclusive components add up to the enclosing total."""
    def __init__(self):
        self.values = {}
        self.stack = []

    @contextmanager
    def measure(self, name):
        item = [time.perf_counter(), time.thread_time(), 0., 0.]
        self.stack.append(item)
        try:
            yield
        finally:
            wall, cpu = time.perf_counter()-item[0], time.thread_time()-item[1]
            self.stack.pop()
            value = self.values.setdefault(name, dict(calls=0, wall_s=0., thread_cpu_s=0.,
                exclusive_wall_s=0., exclusive_thread_cpu_s=0.))
            value['calls'] += 1
            value['wall_s'] += wall
            value['thread_cpu_s'] += cpu
            value['exclusive_wall_s'] += wall-item[2]
            value['exclusive_thread_cpu_s'] += cpu-item[3]
            if self.stack:
                self.stack[-1][2] += wall
                self.stack[-1][3] += cpu

    def wrap(self, function, name):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured


@contextmanager
def geometry_instrumentation(timing, model):
    # Patches are private to this single-threaded profiling process and restored.
    from mujoco_conveyor_sorting.runtime import visible_target, shape_pick, frame_products
    with ExitStack() as stack:
        for module, attribute, label in (
            (visible_target, 'project_rgb_labels', 'rgb_projection'),
            (visible_target, 'visible_depth_mask', 'outline_check'),
            (shape_pick, 'filter_target_mask', 'target_filter'),
            (model, 'classify_robot_points', 'robot_point_filter'),
            (model, 'self_filter_bodies_for_state', 'robot_geometry_prepare')):
            stack.enter_context(patch.object(module, attribute,
                timing.wrap(getattr(module, attribute), label)))
        original = frame_products.FrameProducts.get
        def get(products, name, compute):
            label = name[0] if isinstance(name, tuple) else name
            if label in ('native_deprojection', 'belt_coordinates'):
                with timing.measure('coordinates'):
                    return original(products, name, compute)
            return original(products, name, compute)
        stack.enter_context(patch.object(frame_products.FrameProducts, 'get', get))
        yield


def geometry_sample(config, frame, d2c, kwargs, model, mode):
    from mujoco_conveyor_sorting.runtime.single_pick import detect_target
    from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
    from nero_core.two_second.vision import obb_grasps
    timing = Timings()
    row = dict(sequence=int(frame.header.sequence), status='geometry_rejected')
    try:
        with geometry_instrumentation(timing, model), timing.measure('geometry_total'):
            if mode == 'local':
                detection = detect_seed_local(frame, d2c, detector=detect_target,
                    robot_model=model, **kwargs)
            else:
                detection = detect_target(frame, d2c, robot_model=model, **kwargs)
            with timing.measure('grasp_generation'):
                points = detection['object_points']
                center = np.median(points, axis=0)
                obb_grasps(points, detection['belt_region']['plane_to_base'], config.canonical_tcp)
                seeds = kwargs['target_seed_pixel_indices']
                overlap = float(np.mean(detection['mask'].ravel()[seeds]))
                if overlap < .5:
                    raise ValueError('same exposure seed overlap below 0.5')
            row.update(status='geometry_ready', point_count=len(points), center=center.tolist(),
                seed_overlap=overlap, local_processing=detection.get('local_processing'))
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error),
            details=getattr(error, 'details', {}))
    row['stages'] = timing.values
    return row


def distribution(values):
    x = np.asarray(values, dtype=float)*1000
    return dict(n=len(x), mean_ms=float(x.mean()), p50_ms=float(np.percentile(x, 50)),
        p90_ms=float(np.percentile(x, 90)), p95_ms=float(np.percentile(x, 95)),
        p99_ms=float(np.percentile(x, 99)), max_ms=float(x.max())) if len(x) else None


def summarize(rows):
    groups = {}
    for status in sorted({r['status'] for r in rows}):
        selected = [r for r in rows if r['status'] == status]
        stages = {}
        for name in sorted({k for r in selected for k in r.get('stages', {})}):
            stages[name] = {metric:distribution([r['stages'][name][metric] for r in selected
                if name in r.get('stages', {})]) for metric in
                ('wall_s', 'thread_cpu_s', 'exclusive_wall_s', 'exclusive_thread_cpu_s')}
        groups[status] = dict(count=len(selected), stages=stages)
    return groups


class PassageTracker:
    """Observed candidate passages, separated by a sustained empty ROI."""
    def __init__(self, clear_s=.7):
        self.clear_s = clear_s
        self.active = None
        self.empty_since = None
        self.completed = []

    def update(self, candidate_count, stamp):
        closed = None
        if candidate_count is None:
            self.empty_since = None
            return self.active, closed
        if candidate_count > 0:
            self.empty_since = None
            if self.active is None:
                self.active = len(self.completed)+1
        elif self.active is not None:
            if self.empty_since is None:
                self.empty_since = stamp
            if stamp-self.empty_since >= self.clear_s:
                closed = self.active
                self.completed.append(closed)
                self.active = None
                self.empty_since = None
        return self.active, closed


def main(argv=None):
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.diagnostics import json_value
    from nero_core.two_second.planning import calibrated_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--mode', choices=('local','full'), default='local')
    parser.add_argument('--replay-frames', type=Path, nargs='+')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--preview', action='store_true', help='Display RGB; Q stops sampling')
    parser.add_argument('--passages', type=int, default=0, help='Stop after this many separated candidate passages; 0 uses duration')
    parser.add_argument('--declared-speed-cm-s', type=float, help='Operator-declared belt speed, metadata only')
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 600 or args.repeats < 1:
        parser.error('seconds must be in (0,600]; repeats must be positive')
    if args.passages < 0 or (args.replay_frames and args.passages):
        parser.error('passages must be nonnegative and live-camera only')
    if args.declared_speed_cm_s is not None and (not np.isfinite(args.declared_speed_cm_s) or args.declared_speed_cm_s <= 0):
        parser.error('declared speed must be finite and positive')
    if args.output.exists():
        parser.error('output directory already exists')
    args.output.mkdir(parents=True)
    rows, camera = [], None
    passages = PassageTracker()
    report = dict(mode='saved_frame_replay' if args.replay_frames else 'camera_only',
        geometry_mode=args.mode, robot_connections=0, can_connections=0, robot_commands_sent=0,
        camera_connections=0, status='starting', limitations=[
            'No robot feedback: live self filtering assumes configured standby joints and open gripper.',
            'Geometry computation only; no trajectory planning, deadline admission or robot execution.',
            'Frame counts are not independent workpiece counts; successes and rejections are separated.',
            'Nested wall timings overlap; only exclusive timings can be added within geometry_total.',
            'Thread CPU excludes other/native worker threads; wall minus thread CPU is not a precise scheduling delay.',
            'First geometry sample is included and identified; unchanged-state caching may be faster than live moving-arm feedback.'])
    report.update(requested_passages=args.passages, declared_speed_cm_s=args.declared_speed_cm_s,
        speed_source='operator_declared_not_measured',
        passage_rule='candidate occupancy separated by at least 0.7s observed empty ROI; not physical identity tracking')
    def record(row):
        row['sample_index'] = len(rows)
        row['first_geometry_sample'] = ('geometry_total' in row.get('stages', {}) and
            not any('geometry_total' in r.get('stages', {}) for r in rows))
        rows.append(row)
        with (args.output/'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(json_value(row), ensure_ascii=False)+'\n')
    try:
        if args.replay_frames:
            from benchmark_local_geometry import load_case
            for path in args.replay_frames:
                cfg, frame, d2c, kwargs, state_source = load_case(path.resolve())
                model = calibrated_model(cfg)
                for repeat in range(args.repeats):
                    row = geometry_sample(cfg, replace(frame), d2c, kwargs, model, args.mode)
                    row.update(source=str(path.resolve()), repeat=repeat,
                        state_source=state_source+'_assumed_open', first_for_model=repeat == 0)
                    record(row)
        else:
            import cv2
            from nero_core.two_second.fixed_freeze import NativeDepthLocator, fixed_freeze_detection_surface
            from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
            from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, BeltSurfaceNotReady
            from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
            from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration, apply_runtime_depth_calibration
            cfg = CellConfig.load(args.config)
            # Match the independent experiment's documented size range, private only.
            cfg.raw['maximum_target_length_m'] = .20
            report.update(config=str(cfg.source), source_config_sha256=cfg.source_sha256,
                effective_maximum_target_length_m=.20)
            model = calibrated_model(cfg)
            state = dict(joints=cfg.standby.copy(), width_m=cfg.settings.open_width_m)
            cal = load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
            settings = dict(cfg.raw['camera'])
            settings.update(enable_color=True, require_color=True, timestamp_source='device',
                serial_number=cal['serial_number'], calibration_id=cal['calibration_id'],
                camera_to_world_cv=cal['depth_to_base_cv'])
            camera = OrbbecCameraBackend(settings)
            camera.start()
            report['camera_connections'] = 1
            camera.read_latest(timeout_s=3., wait_for_fresh=True)
            d2c = camera.get_depth_to_color_cv()
            surface = locator = None
            sequence = None
            epoch = None
            started = time.monotonic()
            last_print = last_save = started-10
            print('CAMERA_ONLY_STARTED: no CAN or robot commands; sampling for',args.seconds,'seconds',flush=True)
            while time.monotonic()-started < args.seconds:
                timing = Timings()
                with timing.measure('camera_wait'):
                    raw = camera.read_latest(timeout_s=1., after_sequence=sequence, wait_for_fresh=True)
                sequence = raw.header.sequence
                if epoch != raw.header.clock_epoch:
                    epoch = raw.header.clock_epoch
                    surface = locator = None
                with timing.measure('depth_calibration'):
                    frame = apply_runtime_depth_calibration(raw, cal, d2c)
                row = dict(sequence=int(sequence), clock_epoch=int(epoch), status='no_single_candidate',
                    state_source='assumed_configured_standby_open_no_feedback')
                try:
                    if surface is None or not surface.matches_frame(frame, d2c):
                        with timing.measure('belt_initialization'):
                            surface = BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
                            locator = NativeDepthLocator(cfg,surface,frame)
                    with timing.measure('locator'):
                        candidates, diagnostics = locator.locate(frame,robot_model=model,robot_state=state)
                    row['candidate_count'] = len(candidates)
                    if len(candidates) == 1:
                        kwargs = dict(fit_plane=fit_plane,green_belt_points=green_belt_points,
                            target_shape='simple',measured_surface=fixed_freeze_detection_surface(surface),
                            robot_state=state,target_seed_pixel_indices=candidates[0].seed_pixel_indices,
                            maximum_target_height_m=cfg.raw['maximum_target_height_m'],
                            maximum_target_length_m=.20,scene_scope=cfg.raw['environment_collision_scope'])
                        row.update(geometry_sample(cfg,frame,d2c,kwargs,model,args.mode))
                    row['exposure_age_at_done_s'] = time.monotonic()-depth_capture_timing(frame)['depth_sample_monotonic_estimate']
                except BeltSurfaceNotReady as error:
                    row.update(status='belt_not_ready',error=str(error))
                row['stages'] = {**timing.values, **row.get('stages', {})}
                passage, closed = passages.update(row.get('candidate_count'), frame.header.received_time)
                row['passage_id'] = passage if row.get('candidate_count',0) else None
                row['received_monotonic_s'] = frame.header.received_time
                if row['passage_id'] is not None:
                    previous = [r for r in rows if r.get('passage_id') == passage]
                    if not previous:
                        print('PASSAGE_STARTED',passage,flush=True)
                        cv2.imwrite(str(args.output/f'passage_{passage:02d}_first.jpg'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                    if row['status'] == 'geometry_ready' and not any(r['status']=='geometry_ready' for r in previous):
                        cv2.imwrite(str(args.output/f'passage_{passage:02d}_recognized.jpg'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                record(row)
                if closed is not None:
                    print('PASSAGE_COMPLETED',closed,flush=True)
                if args.passages and len(passages.completed) >= args.passages:
                    report['stop_reason'] = 'requested_passages_completed'
                    break
                if time.monotonic()-last_save >= 5:
                    cv2.imwrite(str(args.output/'latest_rgb.jpg'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                    last_save = time.monotonic()
                if args.preview:
                    cv2.imshow('Camera timing only - no robot commands', cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1)&255 in (ord('q'),27):
                        break
                if time.monotonic()-last_print >= 5:
                    print(dict(samples=len(rows),statuses=dict(Counter(r['status'] for r in rows))),flush=True)
                    last_print = time.monotonic()
            report['sample_wall_s'] = time.monotonic()-started
        report['status'] = 'completed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception as error:
                report.update(status='failed', close_error=str(error))
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
        report.update(samples=len(rows), distributions=summarize(rows),
            steady_distributions=summarize([r for r in rows if not r.get('first_geometry_sample')
                and not r.get('first_for_model')]))
        report['completed_passages'] = passages.completed
        report['unfinished_passage'] = passages.active
        report['passage_results'] = [dict(passage_id=i, completed=i in passages.completed,
            distributions=summarize([r for r in rows if r.get('passage_id') == i]),
            ambiguous_multiple_candidates=any(r.get('candidate_count',0)>1 for r in rows if r.get('passage_id')==i))
            for i in sorted({r['passage_id'] for r in rows if r.get('passage_id') is not None})]
        if args.passages and len(passages.completed) < args.passages and report['status']=='completed':
            report.update(status='incomplete', stop_reason='duration_elapsed_before_requested_passages')
        (args.output/'summary.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status=report['status'],samples=len(rows),output=str(args.output),
            error=report.get('error')),ensure_ascii=False),flush=True)
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())

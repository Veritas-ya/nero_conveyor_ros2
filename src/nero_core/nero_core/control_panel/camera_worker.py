"""Headless host of the existing single-exposure camera/planning experiment."""
import argparse
import json
from pathlib import Path
import signal
from threading import Event
import time

import numpy as np


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez(stream, **arrays)
    path.with_suffix('.tmp').replace(path)


def publish_geometry(output, geometry, path, *, generation):
    from nero_core.two_second.collision_capture import frame_metadata
    from nero_core.two_second.diagnostics import json_value
    frame = geometry['frame']
    identity = [frame.header.clock_epoch, generation, frame.header.sequence]
    meta = frame_metadata(frame, geometry['d2c'])
    meta.update(identity=identity, obb=geometry['obb'], camera_only=True)
    atomic_npz(output/'preview/target.npz',
        metadata=np.asarray(json.dumps(json_value(meta), ensure_ascii=False)),
        frame_rgb=frame.rgb, frame_depth_m=frame.depth_m, points=geometry['points'],
        grasps=np.asarray(geometry['grasps']))
    if path:
        report = path['report']
        meta = {'identity': identity, 'frame_sequence': frame.header.sequence,
            'camera_only': True, 'path_duration_s': report['trajectory_duration_s'],
            'planning_wall_s': report['planning_wall_s'],
            'actual_remaining_send_lead_s': report['actual_remaining_send_lead_s'],
            'meaning': 'model_output_not_hardware_success'}
        atomic_npz(output/'preview/plan.npz', metadata=np.asarray(json.dumps(meta)),
            tcp_xyz_base_m=path['tcp_xyz'], contact_grasp_base=path['contact_grasp'],
            predicted_contact_points_base_m=path['predicted_object_points'])
    else:
        (output/'preview/plan.npz').unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--belt-speed-cm-s', type=float, required=True)
    parser.add_argument('--preset', choices=('single', 'coarse'), default='single')
    args = parser.parse_args(argv)
    stopped, rearm = Event(), Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stopped.set())
    signal.signal(signal.SIGUSR1, lambda *_: rearm.set())
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.diagnostics import json_value
    from nero_core.two_second.fixed_freeze_test import FixedFreezeWorker
    from nero_core.two_second.velocity_source import with_fixed_belt_speed
    from nero_core.two_second.visual_archive import VisualArchive
    from nero_core.two_second.taught import TaughtSegment
    from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
    config = with_fixed_belt_speed(CellConfig.load(args.config), args.belt_speed_cm_s/100.)
    config.raw['candidate_retry_policy'] = 'coarse_first' if args.preset == 'coarse' else 'per_grasp'
    taught = args.workspace/'taught/session_20261104_223416'
    config.raw['taught_transfer_file'] = str(taught/'transfer_trimmed_start025_20261105.json')
    config.raw['taught_return_file'] = str(taught/'planned_return/return_generated.json')
    config.raw['taught_segments'] = {kind: TaughtSegment.load(config.raw[f'taught_{kind}_file'], config, kind)
                                     for kind in ('transfer', 'return')}
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    worker = FixedFreezeWorker(config, calibration, args.output)
    visual = VisualArchive(args.output, lambda *a, **k: None, live=True, direction=config.direction)
    worker.start()
    token = None
    try:
        while worker.is_alive() and not stopped.wait(.2):
            if rearm.is_set():
                rearm.clear()
                worker.rearm.set()
            with worker.lock:
                snapshot = dict(worker.latest)
            if snapshot.get('frame') is not None:
                visual.offer_frame(snapshot['frame'], snapshot['d2c'])
            geometry = snapshot.get('live_geometry')
            path = snapshot.get('path_preview')
            current = (worker.generation, None if geometry is None else geometry['source_sequence'],
                       None if path is None else path['file'])
            if current != token:
                if geometry:
                    publish_geometry(args.output, geometry, path, generation=worker.generation)
                else:
                    for name in ('target', 'plan'):
                        (args.output/f'preview/{name}.npz').unlink(missing_ok=True)
                token = current
            status = {'status': snapshot.get('status', worker.status), 'camera_only': True,
                'camera_connections': worker.camera_connections, 'can_connections': 0,
                'robot_commands_sent': 0, 'generation': worker.generation,
                'freeze_upstream_cm': config.raw['freeze_position_upstream_m']*100,
                'tolerance_cm': config.raw['freeze_position_tolerance_m']*100,
                'attempts': len(worker.rows), 'last_attempt': worker.rows[-1] if worker.rows else None}
            temporary = args.output/'panel_status.tmp'
            temporary.write_text(json.dumps(json_value(status), ensure_ascii=False, allow_nan=False))
            temporary.replace(args.output/'panel_status.json')
    finally:
        worker.stop.set()
        worker.buffer.close()
        worker.join()
        visual.close()
    with worker.lock:
        snapshot = dict(worker.latest)
    error = snapshot.get('error')
    if error:
        (args.output/'panel_status.json').write_text(json.dumps({'error': error, 'status': 'stopped',
            'can_connections': 0, 'robot_commands_sent': 0}, ensure_ascii=False))
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())

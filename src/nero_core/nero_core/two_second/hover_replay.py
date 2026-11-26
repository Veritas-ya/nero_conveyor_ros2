"""Restore an actual staged-test exposure and its measured planning start."""
import json
from pathlib import Path

import numpy as np

from .collision_view import restore_frame
from .paper_replay import input_fingerprint
from .experiment_records import digest
from .vision import MeasuredTarget


def restore_hover_input(run, config, cycle_index=1):
    run = Path(run).resolve(strict=True)
    event_path = run/f'cycle_{cycle_index:03d}'/'events.json'
    events = json.loads(event_path.read_text())
    submissions = [r['data'] for r in events if r['event'] == 'hover_planning_submitted']
    if len(submissions) != 1:
        raise ValueError('expected exactly one hover planning submission per cycle')
    event = submissions[0]
    matches = []
    for path in sorted((run/'experiment').glob('*_target.npz')):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if (meta['identity'] == event['identity'] and
                    abs(meta['capture_s']-event['capture_s']) < 1e-6):
                matches.append(path)
    if len(matches) != 1:
        raise ValueError('hover identity/exposure must match exactly one archived target')
    path = matches[0]
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        frame = restore_frame(meta, {k:z[k].copy() for k in ('frame_rgb','frame_depth_m')})
        detection = dict(meta['detection'])
        detection.update({k:z[k].copy() for k in ('scene_points','scene_pixel_indices','mask') if k in z})
        detection.update(frame=frame, depth_to_color_cv=np.asarray(meta['depth_to_color_cv']),
                         scene_state=meta['capture_state'])
        target = MeasuredTarget(tuple(meta['identity']), meta['capture_s'], meta['received_s'],
            meta['device_s'], frame.header.clock_epoch, frame.header.sequence,
            meta['center'], meta['velocity'], tuple(z['grasps'].copy()), z['points'].copy(),
            detection, meta['capture_state'], meta['speed_report'], meta['entry_position'])
    from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
    cal = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    if (frame.calibration_id != cal['calibration_id'] or frame.serial_number != cal['serial_number']
            or not np.allclose(frame.camera_to_world_cv, cal['depth_to_base_cv'], atol=1e-12, rtol=0)):
        raise ValueError('archived hover exposure calibration differs from replay configuration')
    if (tuple(event['identity'])[0] != frame.header.clock_epoch or
            tuple(event['identity'])[2] != frame.header.sequence or
            not target.capture_s <= target.received_s <= event['submission_s']):
        raise ValueError('hover exposure identity or submission clock is inconsistent')
    state = event['start_state']
    info = dict(run=str(run), cycle_index=cycle_index, archive_sha256=digest(path),
        event_file=str(event_path), event_sha256=digest(event_path), identity=list(target.identity),
        input_sha256=input_fingerprint(target, state), capture_s=target.capture_s,
        submission_s=event['submission_s'], recorded_age_at_submission_s=event['submission_s']-target.capture_s,
        prediction_speed_m_s=float(np.linalg.norm(target.velocity)),
        target_point_count=len(target.points), scene_point_count=len(detection['scene_points']),
        input_source='full_archived_target_and_measured_planning_state')
    return target, state, info

"""Offline comparison of a selected target, feedback and a real camera image.

The image is evidence of physical geometry; FK alone is not a measured TCP.
Rendering and disk writes belong after motion, outside the dispatch loop.
"""
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from ..calibration.geometry import pose_xyz_rpy, rigid_transform, transform_distance
from .grasp_pose_display import _project_points


def feedback_pose_report(plan, state):
    target = rigid_transform(plan['grasp_tcp'])
    offset = rigid_transform(plan['flange_to_tcp_nominal'])
    actual = pose_xyz_rpy(state['pose']) @ offset
    distance, angle = transform_distance(target, actual)
    return {
        'planned_tcp_base_m': target.tolist(),
        'feedback_nominal_tcp_base_m': actual.tolist(),
        'delta_xyz_mm': ((actual[:3, 3]-target[:3, 3])*1000).tolist(),
        'translation_error_mm': distance*1000, 'rotation_error_deg': angle,
        'physical_tcp_measured': False,
        'note': 'SDK flange feedback times nominal tool transform; compare image outlines with real fingers independently.',
    }


def save_execution_evidence(directory, name, plan, state, frame, d2c, model):
    """Keep raw RGB-D and overlay two explicitly named model states.

    Latest-frame capture and feedback are not synchronized. Keep their original
    clocks and timestamps so a stale/differently posed image is never relabelled
    as a measured execution error. Does not change the selected plan or model.
    """
    if name not in ('grasp_arrival', 'grasp_contact', 'grasp_stopped'):
        raise ValueError('unknown evidence event')
    if not np.allclose(model.flange_to_tcp, plan['flange_to_tcp_nominal'], atol=1e-6, rtol=0):
        raise ValueError('recorded plan TCP differs from current model')
    directory = Path(directory)
    raw = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    overlay = raw.copy()
    qpos = model.data.qpos.copy()
    descend = next((stage for stage in plan['stages'] if stage['name'] == 'descend'), None)
    if descend is None and plan.get('preview_kind') != 'full_policy_target':
        raise ValueError('selected plan has no descent trajectory')
    model_states = []
    if descend is not None:
        model_states.append((descend['joints_rad'][-1], plan['settings']['open_width_m'], (220, 80, 220)))
    model_states.append((state['joints'], state['width_m'], (0, 210, 255)))
    try:
        for q, width, color in model_states:
            model.set_state(q, width)
            for g, vertices in model.mesh_vertices.items():
                body = model.model.body(model.model.geom_bodyid[g]).name
                if not body.startswith('gripper'):
                    continue
                points = vertices @ model.data.geom_xmat[g].reshape(3, 3).T+model.data.geom_xpos[g]
                _, _, uv, _, _, inside = _project_points(points, frame, d2c)
                if inside.all():
                    hull = cv2.convexHull(uv.astype('float32')).astype('int32')
                    cv2.polylines(overlay, [hull], True, color, 1, cv2.LINE_AA)
    finally:
        model.data.qpos[:] = qpos
        model.mj.mj_forward(model.model, model.data)
    caption = ('Yellow: feedback model / magenta: selected target / background: actual RGB'
               if descend is not None else 'Yellow: feedback model / no descent joint solution / background: actual RGB')
    cv2.putText(overlay, caption,
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 1, cv2.LINE_AA)
    for suffix, image in [('rgb', raw), ('overlay', overlay)]:
        if not cv2.imwrite(str(directory/f'{name}_{suffix}.png'), image):
            raise OSError('failed to save execution image')
    np.save(directory/f'{name}_depth_m.npy', frame.depth_m)
    report = {
        **feedback_pose_report(plan, state), 'event': name, 'robot_state': state,
        'selected_target_mesh_available': descend is not None,
        'header': asdict(frame.header), 'timestamp_metadata': frame.timestamp_metadata,
        'pairing': 'latest_frame_at_event; not hardware synchronized or stationary-bracketed',
        'depth_intrinsics': asdict(frame.depth_intrinsics),
        'color_intrinsics': asdict(frame.color_intrinsics),
        'depth_to_base_cv': frame.camera_to_world_cv.tolist(),
        'depth_to_color_cv': np.asarray(d2c).tolist(),
    }
    (directory/f'{name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {'event': name, 'json': f'{name}.json', 'rgb': f'{name}_rgb.png',
            'overlay': f'{name}_overlay.png', 'depth': f'{name}_depth_m.npy',
            'nominal_tcp_error_mm': report['translation_error_mm']}

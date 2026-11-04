"""Display a selected plan's nominal TCP without connecting or commanding hardware."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..calibration.geometry import inverse, rigid_transform


def _pose_fields(transform, prefix):
    rotation = Rotation.from_matrix(transform[:3, :3])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always', UserWarning)
        rpy = rotation.as_euler('xyz', degrees=True)
    return {
        f'{prefix}_xyz_m': transform[:3, 3].tolist(),
        f'{prefix}_xyz_mm': (transform[:3, 3]*1000).tolist(),
        f'{prefix}_rpy_xyz_deg': rpy.tolist(),
        f'{prefix}_quat_xyzw': rotation.as_quat().tolist(),
        f'{prefix}_rpy_gimbal_lock': any('Gimbal lock' in str(item.message) for item in recorded),
    }


def describe_grasp_pose(grasp_tcp, flange_to_tcp=None):
    """Describe the selected TCP and its derived flange in the robot base frame.

    RPY uses fixed XYZ, R = Rz(yaw) Ry(pitch) Rx(roll), matching pose_xyz_rpy.
    The quaternion remains available when Euler angles are not unique.
    """
    grasp = rigid_transform(grasp_tcp, 'grasp_tcp')
    report = {
        'frame': 'nero_base',
        'grasp_tcp': grasp.tolist(),
        'orientation_convention': 'fixed XYZ; R = Rz(yaw) Ry(pitch) Rx(roll)',
        'tcp_source': 'robot_model_nominal_tcp',
        'physical_tcp_calibrated': False,
        **_pose_fields(grasp, 'tcp'),
    }
    if flange_to_tcp is not None:
        offset = rigid_transform(flange_to_tcp, 'flange_to_tcp')
        flange = grasp @ inverse(offset)
        report.update(flange_to_tcp=offset.tolist(), target_flange=flange.tolist(),
                      **_pose_fields(flange, 'flange'))
    return report


def _project_points(points_base, frame, depth_to_color):
    to_depth = inverse(frame.camera_to_world_cv)
    d2c = rigid_transform(depth_to_color, 'depth_to_color')
    depth = points_base @ to_depth[:3, :3].T + to_depth[:3, 3]
    color = depth @ d2c[:3, :3].T + d2c[:3, 3]
    positive = (depth[:, 2] > 1e-9) & (color[:, 2] > 1e-9)
    uv = np.full((len(points_base), 2), np.nan)
    if positive.any():
        pixels, _ = cv2.projectPoints(
            color[positive], np.zeros(3), np.zeros(3), frame.color_intrinsics.matrix,
            np.asarray(frame.color_intrinsics.distortion, dtype=float))
        uv[positive] = pixels.reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    height, width = frame.rgb.shape[:2]
    inside = (positive & finite & (uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return depth, color, uv, positive, finite, inside


def _save_png(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'cannot save grasp pose image: {path}')


def save_grasp_pose_overlay(directory, plan, frame, d2c, flange_to_tcp=None, prefix='grasp_pose'):
    """Save the actual selected plan matrix, RGB axes and an unmodified RGB copy.

    A line is drawn only when both endpoints have positive depth in both
    camera frames and project inside the RGB image. Opening is a schematic
    span along TCP Y, not a measurement of the physical finger surfaces.
    Returned file paths are absolute strings.
    """
    if frame.rgb is None or frame.color_intrinsics is None:
        raise ValueError('grasp pose display requires recorded RGB and color intrinsics')
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError('grasp pose prefix must be a nonempty filename prefix')
    grasp = rigid_transform(plan['grasp_tcp'], 'selected plan grasp_tcp')
    if flange_to_tcp is None:
        flange_to_tcp = plan.get('flange_to_tcp_nominal')
    report = describe_grasp_pose(grasp, flange_to_tcp)
    yaw = plan.get('grasp_yaw_flip_deg')
    if yaw is not None:
        yaw = float(yaw)
        if not np.isfinite(yaw):
            raise ValueError('selected yaw flip must be finite')
    opening = float(plan.get('settings', {}).get('open_width_m', .10))
    if not np.isfinite(opening) or opening <= 0:
        raise ValueError('displayed opening width must be finite and positive')
    axis_length = .06
    names = ['tcp', 'x_axis_end', 'y_axis_end', 'z_axis_end', 'opening_minus_y', 'opening_plus_y']
    local = np.array([[0., 0., 0.], [axis_length, 0., 0.], [0., axis_length, 0.],
                      [0., 0., axis_length], [0., -opening/2, 0.], [0., opening/2, 0.]])
    points = local @ grasp[:3, :3].T + grasp[:3, 3]
    depth, color, uv, positive, finite, visible = _project_points(points, frame, d2c)
    records = []
    for i, name in enumerate(names):
        status = ('nonpositive_depth' if not positive[i] else 'nonfinite_projection'
                  if not finite[i] else 'visible' if visible[i] else 'offscreen')
        records.append({'name': name, 'base_m': points[i].tolist(),
                        'depth_camera_m': depth[i].tolist(), 'color_camera_m': color[i].tolist(),
                        'rgb_uv': uv[i].tolist() if finite[i] else None, 'status': status})
    projection = {
        'chain': 'nero_base -> depth_optical_cv -> factory_D2C -> color_optical_cv -> RGB_K_and_D',
        'total_points': len(points), 'visible_points': int(visible.sum()),
        'nonpositive_depth_points': int((~positive).sum()),
        'nonfinite_projection_points': int((positive & ~finite).sum()),
        'offscreen_points': int((positive & finite & ~visible).sum()),
        'points': records, 'drawn_axes': [], 'opening_span_drawn': False,
    }
    original = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    overlay = original.copy()
    # Clamp rounded coordinates only for a point already proven in-image.
    height, width = overlay.shape[:2]
    def pixel(index):
        return tuple(np.clip(np.rint(uv[index]), [0, 0], [width-1, height-1]).astype(int))
    if visible[4] and visible[5]:
        cv2.line(overlay, pixel(4), pixel(5), (230, 230, 230), 2, cv2.LINE_AA)
        for i in (4, 5):
            cv2.circle(overlay, pixel(i), 4, (230, 230, 230), 1, cv2.LINE_AA)
        projection['opening_span_drawn'] = True
    for index, label, bgr in [(1, 'X', (0, 0, 255)), (2, 'Y', (0, 220, 0)), (3, 'Z', (255, 0, 0))]:
        if visible[0] and visible[index]:
            cv2.arrowedLine(overlay, pixel(0), pixel(index), bgr, 3, cv2.LINE_AA, tipLength=.18)
            cv2.putText(overlay, label, pixel(index), cv2.FONT_HERSHEY_SIMPLEX, .6, bgr, 2, cv2.LINE_AA)
            projection['drawn_axes'].append(label)
    if visible[0]:
        cv2.circle(overlay, pixel(0), 5, (255, 255, 255), 2, cv2.LINE_AA)
    caption = f'SELECTED nominal TCP / X red Y green Z blue / opening {opening*1000:.0f}mm schematic'
    cv2.putText(overlay, caption, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, caption, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    report.update(source='selected_plan.grasp_tcp', selected_yaw_flip_deg=yaw,
                  axis_length_m=axis_length, gripper_open_width_m=opening,
                  opening_span_schematic=True, projection=projection,
                  note='Selected target from the plan; nominal model TCP, not a measured physical TCP calibration.')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory/f'{prefix}.json'
    overlay_path = directory/f'{prefix}_overlay.png'
    rgb_path = directory/f'{prefix}_rgb.png'
    _save_png(rgb_path, original)
    _save_png(overlay_path, overlay)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {'report': report, 'json': str(json_path), 'overlay_png': str(overlay_path), 'rgb_png': str(rgb_path)}

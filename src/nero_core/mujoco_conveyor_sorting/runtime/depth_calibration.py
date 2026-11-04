"""Use an explicitly selected runtime DEPTH transform, without inventing a PnP result."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import yaml

from ..calibration.geometry import rigid_transform


def load_runtime_depth_calibration(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    camera = yaml.safe_load(raw)['runtime']['camera']
    if not camera.get('serial_number') or not camera.get('calibration_id'):
        raise ValueError('runtime外参缺少相机序列号或calibration_id')
    transform = rigid_transform(camera['camera_to_world_cv'], 'runtime depth_to_base_cv')
    sizes = {side: [int(camera[f'{side}_width']), int(camera[f'{side}_height'])]
             for side in ('depth', 'color')}
    if any(size <= 0 for pair in sizes.values() for size in pair):
        raise ValueError('runtime相机分辨率必须为正数')
    return {'calibration_id': camera['calibration_id'], 'serial_number': camera['serial_number'],
            'camera_frame': 'depth_optical_cv', 'units': 'metres',
            'depth_to_base_cv': transform.tolist(), 'capture_profiles': sizes,
            'calibration_source': 'runtime_depth', 'intrinsics_source': 'current_camera_frame',
            'original_result_used': False, 'quality_report_used': False,
            'source_path': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest()}


def initial_camera_transform(calibration):
    # The historical result path is kept unchanged. Runtime explicitly names
    # native depth; this matrix must NOT be relabelled as camera_to_base_cv.
    key = ('depth_to_base_cv' if calibration.get('camera_frame') == 'depth_optical_cv'
           else 'camera_to_base_cv')
    return calibration[key]


def apply_runtime_depth_calibration(frame, calibration, depth_to_color):
    if frame.serial_number != calibration['serial_number']:
        raise ValueError('当前相机序列号与runtime外参不一致')
    if frame.rgb_aligned_to_depth:
        raise ValueError('runtime深度外参需要原生深度和彩色帧，不能使用已对齐帧')
    if frame.color_intrinsics is None or frame.rgb is None:
        raise ValueError('runtime深度外参需要当前相机RGB和真实彩色内参')
    for side, intrinsic in [('depth', frame.depth_intrinsics), ('color', frame.color_intrinsics)]:
        if [intrinsic.width, intrinsic.height] != calibration['capture_profiles'][side]:
            raise ValueError(f'{side}分辨率与runtime外参配置不一致')
    # D2C is still used by detection/projection. Validate the actual supplied
    # SDK transform, but never multiply it into an already-DEPTH-to-base map.
    rigid_transform(depth_to_color, 'SDK depth_to_color_cv')
    transform = rigid_transform(calibration['depth_to_base_cv'], 'depth_to_base_cv')
    metadata = dict(frame.timestamp_metadata or {})
    metadata['calibration_usage'] = {key: calibration[key] for key in (
        'calibration_source', 'intrinsics_source', 'source_path', 'source_sha256',
        'original_result_used', 'quality_report_used')}
    return replace(frame, camera_to_world_cv=transform,
                   calibration_id=calibration['calibration_id'], timestamp_metadata=metadata)

"""Opt-in native point selection for OBB; retain the full collision cloud."""
import time
import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady


def validate(raw):
    if not isinstance(raw.setdefault('depth_separation_enabled', False), bool):
        raise ValueError('depth_separation_enabled 必须为布尔值')
    if raw['depth_separation_enabled'] and raw.get('fast_stereo_enabled', False):
        raise ValueError('深度分离接口当前仅验证原生深度，不能同时启用 Fast Stereo')


def grasp_points(detection, frame, *, enabled=False):
    """Return base-frame measured points used by OBB for this exposure.

    Full object_points, scene_points, mask, and frame depth stay unchanged.
    The compact keep mask can reconstruct the OBB input from saved raw points.
    """
    points = detection['object_points']
    if not enabled:
        return points
    from mujoco_conveyor_sorting.runtime.depth_separation import separate
    started = time.monotonic()
    mask = detection['mask']
    indices = np.flatnonzero(mask)
    seeds = detection.get('object_seed_pixel_indices')
    if seeds is None or len(indices) != len(points):
        raise TargetNotReady('深度分离缺少同帧原生种子或像素对应', reason='depth_separation_input')
    pixels = np.column_stack(np.unravel_index(indices, mask.shape))
    plane_to_base = np.asarray(detection['belt_region']['plane_to_base'])
    plane = detection['plane']
    local = (points-plane_to_base[:3, 3]) @ plane_to_base[:3, :3]
    camera = (points-frame.camera_to_world_cv[:3, 3]) @ frame.camera_to_world_cv[:3, :3]
    spacing = float(np.median(camera[:, 2])/min(frame.depth_intrinsics.fx, frame.depth_intrinsics.fy))
    try:
        keep, stats = separate(local, pixels, np.isin(indices, seeds), spacing=spacing,
            plane_rms=plane['inlier_rms_m'], plane_band=plane['inlier_threshold_z_m'])
    except ValueError as error:
        raise TargetNotReady(str(error), reason='depth_separation_support') from error
    detection['grasp_point_keep_mask'] = keep
    detection['depth_separation'] = dict(stats, enabled=True, input_points=len(points),
        output_points=int(keep.sum()), frame_sequence=int(frame.header.sequence),
        capture_monotonic_s=float(frame.header.sample_time),
        wall_s=time.monotonic()-started, collision_cloud_preserved=True)
    return points[keep]

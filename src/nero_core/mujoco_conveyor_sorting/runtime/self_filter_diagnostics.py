"""Persist auditable self-filter labels without changing any scene points."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from ..calibration.geometry import inverse


LABEL_NAMES = {0: 'outside', 1: 'robot', 2: 'padding', 3: 'shadow_retained', 4: 'protected'}
DEPTH_LABEL_NAMES = {**LABEL_NAMES, 255: 'not_classified'}
COLORS_BGR = np.array([[0, 0, 0], [255, 255, 0], [0, 255, 255],
                       [255, 0, 255], [0, 255, 0]], dtype=np.uint8)


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _pixels(uv, positive, width, height):
    valid = positive & np.isfinite(uv).all(axis=1)
    safe = np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)
    pixels = np.rint(np.clip(safe, [-1, -1], [width, height])).astype(int)
    valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    return pixels, valid


def _save_image(path, pixels):
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f'cannot write {path}')


def _project_base_to_color(points, frame, d2c):
    to_depth = inverse(frame.camera_to_world_cv)
    depth = points @ to_depth[:3, :3].T + to_depth[:3, 3]
    color = depth @ np.asarray(d2c)[:3, :3].T + np.asarray(d2c)[:3, 3]
    if not len(color):
        return np.empty((0, 2)), np.empty(0, dtype=bool)
    uv, _ = cv2.projectPoints(color, np.zeros(3), np.zeros(3),
                             frame.color_intrinsics.matrix,
                             np.asarray(frame.color_intrinsics.distortion))
    return uv.reshape(-1, 2), color[:, 2] > 0


def _save_scene_region_diagnostics(directory, model, frame, d2c, prefix, state):
    region = getattr(model, 'scene_region_filter_result', None)
    voxel = getattr(model, 'scene_voxel_result', None)
    if region is None or voxel is None:
        return {}
    paths = {'scene_region_json': f'{prefix}_belt_scene.json',
             'scene_cloud_points': f'{prefix}_belt_scene_points.npz'}
    np.savez_compressed(
        directory/paths['scene_cloud_points'],
        roi_points_base_m=region.retained_points,
        roi_retained_indices_in_finite_scene=region.retained_indices,
        collision_points_base_m=model.scene,
        noise_retained_points_base_m=voxel.points,
        representative_indices_in_noise_retained=voxel.representative_indices,
        representative_counts=voxel.representative_counts,
        representative_max_errors_m=voxel.representative_max_errors_m,
        coverage_indices_in_noise_retained=voxel.coverage_indices)
    if frame.rgb is not None and frame.color_intrinsics is not None:
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        uv, positive = _project_base_to_color(region.retained_points, frame, d2c)
        pixels, valid = _pixels(uv, positive, width, height)
        marked = np.zeros((height, width), dtype=bool)
        marked[pixels[valid, 1], pixels[valid, 0]] = True
        image[marked] = np.rint(.7*image[marked]+.3*np.array([0, 255, 0])).astype(np.uint8)
        uv, positive = _project_base_to_color(model.scene, frame, d2c)
        pixels, valid = _pixels(uv, positive, width, height)
        image[pixels[valid, 1], pixels[valid, 0]] = (0, 165, 255)
        roi = region.statistics['roi']
        if roi['enabled']:
            bounds = np.asarray(roi['bounds_m'])
            # Binary vertex numbering: changing one coordinate gives an edge.
            local = np.array([[bounds[0, i], bounds[1, j], bounds[2, k]]
                              for i in range(2) for j in range(2) for k in range(2)])
            transform = np.asarray(roi['plane_to_base'])
            corners = local @ transform[:3, :3].T + transform[:3, 3]
            uv, positive = _project_base_to_color(corners, frame, d2c)
            valid = positive & np.isfinite(uv).all(axis=1)
            pixels = np.rint(np.clip(np.nan_to_num(uv), -1e6, 1e6)).astype(int)
            for first in range(8):
                for bit in (1, 2, 4):
                    second = first ^ bit
                    if first >= second or not (valid[first] and valid[second]):
                        continue
                    visible, begin, end = cv2.clipLine((0, 0, width, height),
                                                       tuple(pixels[first]), tuple(pixels[second]))
                    if visible:
                        cv2.line(image, begin, end, (255, 255, 0), 2, cv2.LINE_AA)
        text = (f'CYAN belt ROI / GREEN native ROI / ORANGE collision cloud '
                f'{len(region.points)} -> {len(region.retained_points)} -> {len(model.scene)}')
        for color, thickness in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
            cv2.putText(image, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, color, thickness, cv2.LINE_AA)
        paths['scene_region_rgb'] = f'{prefix}_belt_scene_rgb.png'
        _save_image(directory/paths['scene_region_rgb'], image)
    document = {
        'schema_version': 1, 'frame': prefix, 'robot_state': state,
        'configuration': model.belt_scene_filter_config.as_dict(),
        'pipeline_order': ['finite_scene', 'belt_roi', 'robot_self_filter',
                           'native_depth_edge_filter', 'native_scene_noise_filter', 'voxel_downsample'],
        'region_statistics': region.statistics,
        'downsample_statistics': voxel.statistics, 'files': paths,
        'notes': ['ROI is a geometric box in the measured belt-plane frame; all colors inside are eligible.',
                  'Target points are separate, checked for full ROI coverage, and unchanged by scene filters.',
                  'Representative indices and coverage indices refer to native points retained by the noise filter.',
                  'Representatives are actual observations, with at most 4 mm point-to-representative distance.',
                  ('Outside-ROI surfaces and camera-occluded space are not checked for collisions.'
                   if region.statistics['roi']['enabled'] else
                   'ROI is disabled; camera-occluded space is not certified clear.'),
                  'Original RGB-D files remain unchanged.']}
    (directory/paths['scene_region_json']).write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2, allow_nan=False)+'\n',
        encoding='utf-8')
    return paths


def _save_scene_depth_diagnostics(directory, model, frame, d2c, prefix, state):
    result = getattr(model, 'scene_depth_filter_result', None)
    if result is None:
        return {}
    removed = result.points[result.removed_mask]
    paths = {'scene_depth_json': f'{prefix}_scene_depth_filter.json',
             'scene_depth_points': f'{prefix}_scene_depth_removed_points.npz'}
    np.savez_compressed(directory/paths['scene_depth_points'], points_base_m=removed)
    if frame.rgb is not None and frame.color_intrinsics is not None:
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        uv, positive = _project_base_to_color(removed, frame, d2c)
        pixels, valid = _pixels(uv, positive, width, height)
        image[pixels[valid, 1], pixels[valid, 0]] = (0, 0, 255)
        label = 'RED robot depth-edge remnants removed (target unchanged)'
        for color, thickness in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
            cv2.putText(image, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, color, thickness, cv2.LINE_AA)
        paths['scene_depth_rgb'] = f'{prefix}_scene_depth_filter_rgb.png'
        _save_image(directory/paths['scene_depth_rgb'], image)
    document = {'schema_version': 1, 'frame': prefix, 'robot_state': state,
        'stage': 'after_robot_self_filter_before_density_filter',
        'statistics': result.statistics, 'components': result.components, 'files': paths,
        'notes': ['Requires native depth boundary evidence and neighboring removed robot pixels.',
                  'No interpolation or replacement of missing depth.',
                  'Original RGB-D and separately measured target points are unchanged.',
                  'A small real object touching the robot can share these single-view cues.']}
    (directory/paths['scene_depth_json']).write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2, allow_nan=False)+'\n',
        encoding='utf-8')
    return paths


def _save_scene_noise_diagnostics(directory, model, frame, d2c, prefix, state):
    result = getattr(model, 'scene_noise_filter_result', None)
    if result is None:
        return {}
    removed = result.points[result.removed_mask]
    paths = {'scene_noise_json': f'{prefix}_scene_noise_filter.json',
             'scene_noise_points': f'{prefix}_scene_noise_removed_points.npz'}
    if result.neighbor_counts is None and len(removed):
        from scipy.spatial import cKDTree
        # Detailed counts are only persisted for deleted samples. Reproduce
        # them exactly here, after delivery, rather than counting the dense
        # whole cloud on the decision path.
        removed_counts = np.asarray(cKDTree(result.points).query_ball_point(
            removed, result.statistics['parameters']['radius_m'], return_length=True), dtype=np.int64)-1
    else:
        removed_counts = (np.empty(0, dtype=np.int64) if result.neighbor_counts is None
                          else result.neighbor_counts[result.removed_mask])
    minimum_neighbors = result.statistics['parameters']['min_neighbors']
    reasons = np.where(removed_counts < minimum_neighbors, 1, 2).astype(np.uint8)
    np.savez_compressed(directory/paths['scene_noise_points'], points_base_m=removed,
                        neighbor_counts=removed_counts, removal_reason=reasons)
    to_depth = inverse(frame.camera_to_world_cv)
    depth_points = removed @ to_depth[:3, :3].T + to_depth[:3, 3]
    if frame.rgb is not None and frame.color_intrinsics is not None:
        color_points = depth_points @ np.asarray(d2c)[:3, :3].T + np.asarray(d2c)[:3, 3]
        if len(color_points):
            uv, _ = cv2.projectPoints(color_points, np.zeros(3), np.zeros(3),
                                     frame.color_intrinsics.matrix,
                                     np.asarray(frame.color_intrinsics.distortion))
            uv = uv.reshape(-1, 2)
        else:
            uv = np.empty((0, 2))
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        pixels, valid = _pixels(uv, color_points[:, 2] > 0, width, height)
        image[pixels[valid, 1], pixels[valid, 0]] = (0, 0, 255)
        cv2.putText(image, 'RED sparse scene points removed (target unchanged)',
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, 'RED sparse scene points removed (target unchanged)',
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        paths['scene_noise_rgb'] = f'{prefix}_scene_noise_filter_rgb.png'
        _save_image(directory/paths['scene_noise_rgb'], image)
    document = {'schema_version': 1, 'frame': prefix, 'robot_state': state,
                'stage': 'after_robot_self_filter_before_scene_voxels',
                'statistics': result.statistics, 'files': paths,
                'removal_reasons': {1: 'radius_outlier', 2: 'compact_small_component'},
                'notes': ['Neighbor counts exclude the point itself and use a single pass.',
                          'Removed points have insufficient local support, not certified empty space.',
                          'Thin or poorly observed obstacles can also have sparse depth.',
                          'Original RGB-D and the separate target cloud are unchanged.']}
    (directory/paths['scene_noise_json']).write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2, allow_nan=False)+'\n',
        encoding='utf-8')
    return paths


def save_self_filter_diagnostics(directory, model, frame, d2c, prefix, state=None):
    """Save native-depth labels, a factory-D2C RGB overlay and removal evidence.

    Padding is an uncertain model tolerance, and can include nearby obstacles.
    Shadow labels remain in the scene; they never certify empty space.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = Path(prefix).name
    result = model.self_filter_result
    points = np.asarray(result.points, dtype=float)
    labels = np.asarray(result.labels)
    removed = np.asarray(result.removed_mask, dtype=bool)
    geom_ids = np.asarray(result.geom_ids, dtype=int)
    if (points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),) or
            removed.shape != (len(points),) or geom_ids.shape != (len(points),) or
            not np.isfinite(points).all() or not np.isin(labels, list(LABEL_NAMES)).all()):
        raise ValueError('invalid self-filter diagnostic arrays')
    labels = labels.astype(np.uint8, copy=True)
    to_depth = inverse(frame.camera_to_world_cv)
    depth_points = points @ to_depth[:3, :3].T + to_depth[:3, 3]
    positive = depth_points[:, 2] > 0
    z = np.where(positive, depth_points[:, 2], 1.)
    k = frame.depth_intrinsics
    uv = np.column_stack((k.fx*depth_points[:, 0]/z+k.cx,
                          k.fy*depth_points[:, 1]/z+k.cy))
    pixels, valid = _pixels(uv, positive, k.width, k.height)
    depth_labels = np.full(frame.depth_m.shape, 255, dtype=np.uint8)
    # The source is native depth, so points normally map one-to-one. If points
    # share a projected pixel, let the nearest surface determine the label.
    order = np.argsort(depth_points[:, 2])
    order = order[valid[order]]
    _, nearest = np.unique(pixels[order, 1]*k.width+pixels[order, 0], return_index=True)
    order = order[nearest]
    depth_labels[pixels[order, 1], pixels[order, 0]] = labels[order]
    paths = {'json': f'{prefix}_self_filter.json',
             'depth_labels': f'{prefix}_self_labels_depth.png',
             'removed_points': f'{prefix}_self_removed_points.npz'}
    paths.update(_save_scene_region_diagnostics(directory, model, frame, d2c, prefix, state))
    paths.update(_save_scene_depth_diagnostics(directory, model, frame, d2c, prefix, state))
    paths.update(_save_scene_noise_diagnostics(directory, model, frame, d2c, prefix, state))
    _save_image(directory/paths['depth_labels'], depth_labels)
    np.savez_compressed(directory/paths['removed_points'], points_base_m=points[removed],
                        labels=labels[removed], geom_ids=geom_ids[removed])
    overlay_count = 0
    if frame.rgb is not None and frame.color_intrinsics is not None:
        color_points = depth_points @ np.asarray(d2c)[:3, :3].T + np.asarray(d2c)[:3, 3]
        if len(color_points):
            rgb_uv, _ = cv2.projectPoints(color_points, np.zeros(3), np.zeros(3),
                                           frame.color_intrinsics.matrix,
                                           np.asarray(frame.color_intrinsics.distortion))
            rgb_uv = rgb_uv.reshape(-1, 2)
        else:
            rgb_uv = np.empty((0, 2))
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        rgb_pixels, rgb_valid = _pixels(rgb_uv, color_points[:, 2] > 0, width, height)
        order = np.argsort(color_points[:, 2])
        order = order[rgb_valid[order]]
        _, nearest = np.unique(rgb_pixels[order, 1]*width+rgb_pixels[order, 0], return_index=True)
        order = order[nearest]
        order = order[labels[order] != 0]
        colors = image.copy()
        colors[rgb_pixels[order, 1], rgb_pixels[order, 0]] = COLORS_BGR[labels[order]]
        marked = np.zeros((height, width), dtype=bool)
        marked[rgb_pixels[order, 1], rgb_pixels[order, 0]] = True
        image[marked] = np.rint(.25*image[marked]+.75*colors[marked]).astype(np.uint8)
        overlay_count = int(marked.sum())
        cv2.putText(image, 'CYAN robot / YELLOW padding / MAGENTA shadow retained / GREEN protected',
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, 'CYAN robot / YELLOW padding / MAGENTA shadow retained / GREEN protected',
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
        paths['rgb_overlay'] = f'{prefix}_self_filter_rgb.png'
        _save_image(directory/paths['rgb_overlay'], image)
    document = {'schema_version': 1, 'frame': prefix, 'frame_header': asdict(frame.header),
                'robot_state': state, 'frame_metadata_file': f'{prefix}.json',
                'statistics': result.statistics,
                'configuration': result.statistics.get('parameters',
                                                        getattr(model, 'self_filter_config', None)),
                'labels': DEPTH_LABEL_NAMES, 'depth_label_domain': 'native_depth_pixels',
                'depth_label_shape': list(depth_labels.shape),
                'depth_projected_scene_points': int(valid.sum()),
                'rgb_overlay_pixels': overlay_count,
                'removed_point_count': int(removed.sum()),
                'padding_removed_points_base_m': points[(labels == 2) & removed],
                'files': paths,
                'notes': ['Model and padding labels are approximate, not exact physical segmentation.',
                          'Padding may include an obstacle touching or very close to the robot.',
                          'Shadow points are retained; shadow is not certified free space.',
                          'Depth label 255 means no scene point was classified at this pixel; targets and excluded depth stay unclassified.',
                          'Native depth labels use depth K. RGB overlay uses factory depth-to-color and RGB K/D.',
                          'Original RGB-D and target points are unchanged.']}
    (directory/paths['json']).write_text(
        json.dumps(_json_value(document), ensure_ascii=False, indent=2, allow_nan=False)+'\n',
        encoding='utf-8')
    return paths

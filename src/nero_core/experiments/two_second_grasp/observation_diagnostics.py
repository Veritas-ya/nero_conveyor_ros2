"""Simulation-only camera-observation diagnostics, never a truth-based control input."""
import numpy as np


def inspect_observation(frame, track_id):
    labels = np.asarray(frame['top_instance_labels'])
    depth = np.asarray(frame['top_depth'])
    rows, cols = np.nonzero(labels == int(track_id))
    values = depth[rows, cols]
    valid = np.isfinite(values) & (values > 0.)
    rows, cols, values = rows[valid], cols[valid], values[valid]
    if not len(values):
        return None
    cal = frame['top_camera']
    k = np.asarray(cal['intrinsics'], float)
    transform = np.asarray(cal['camera_to_world_opencv'], float)
    rays = np.column_stack(((cols-k[0, 2])/k[0, 0], (rows-k[1, 2])/k[1, 1], np.ones(len(rows))))
    points = (rays*values[:, None]) @ transform[:3, :3].T + transform[:3, 3]
    uv = np.array([np.mean(cols), np.mean(rows)])
    z = float(np.median(values))
    center_ray = np.array([(uv[0]-k[0, 2])/k[0, 0], (uv[1]-k[1, 2])/k[1, 1], 1.])
    center = transform[:3, :3] @ (center_ray*z) + transform[:3, 3]
    lo, hi = np.percentile(points, [5., 95.], axis=0)
    return {'pixel_center_uv': uv, 'median_depth_m': z, 'point_count': len(values),
            'depth_p05_p95_m': np.percentile(values, [5., 95.]),
            'deprojected_center_world_m': center,
            'point_cloud_mean_world_m': points.mean(axis=0),
            'point_cloud_median_world_m': np.median(points, axis=0),
            'point_cloud_percentile_box_center_world_m': (lo+hi)/2.,
            'pixel_bounds_uv': [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]}

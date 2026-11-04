"""Measured grasp geometry, independent of reference class or CAD dimensions."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


class ShapeGeometryNotReady(ValueError):
    """Measured target cannot supply a grasp; distinct from classifier faults."""


def make_shape_recognizer():
    from ..perception.shape_similarity import ShapeSimilarityMatcher
    from ..tasks import load_object_specs
    catalog = Path(__file__).resolve().parents[1]/'configs/tless_objects.yaml'
    return ShapeSimilarityMatcher(load_object_specs(catalog))


def exclude_green_belt(frame, depth_to_color, mask):
    """Reject known green pixels using D2C; unknown color retains its depth."""
    from .frame_products import green_rgb_mask
    from .visible_target import project_rgb_labels
    green = green_rgb_mask(frame, depth_to_color)
    projected_green = project_rgb_labels(frame, depth_to_color, green, pixel_mask=mask)
    # Unknown/out-of-view colour maps to False and keeps its native depth,
    # exactly as the original per-mask projection did.
    return np.asarray(mask, dtype=bool) & ~projected_green


def plane_pixel_steps(camera_to_plane, intrinsics, point):
    """Metric pixel Jacobian on a horizontal slice, including camera tilt."""
    origin = camera_to_plane[:3, 3]
    rotation = camera_to_plane[:3, :3]
    camera_point = (np.asarray(point)-origin) @ rotation
    if camera_point[2] <= 0:
        raise ValueError('目标在相机后方')
    ray = rotation @ (camera_point/camera_point[2])
    if abs(ray[2]) < .05:
        raise ValueError('相机视线过于贴近带面，无法可靠估计夹持宽度')
    distance = (point[2]-origin[2])/ray[2]
    du, dv = rotation[:, 0]/intrinsics[0, 0], rotation[:, 1]/intrinsics[1, 1]
    return np.array([distance*(du-ray*du[2]/ray[2]),
                     distance*(dv-ray*dv[2]/ray[2])])


def filter_target_mask(frame, depth_to_color, mask, points_plane, camera_to_plane):
    """Keep measured foreground with RGB and local metric-depth support.

    Reuse the shape measurement's five-point/4 mm/3.5-pixel support rule.
    There is no shape prior, replacement depth or erosion of every boundary.
    """
    clean = exclude_green_belt(frame, depth_to_color, mask)
    rows, cols = np.nonzero(clean)
    points = points_plane[clean]
    radius = None
    if len(points) >= 5:
        steps = plane_pixel_steps(camera_to_plane, frame.depth_intrinsics.matrix,
                                  np.median(points, axis=0))
        radius = max(.004, 3.5*float(np.max(np.linalg.norm(steps[:, :2], axis=1))))
        distances, _ = cKDTree(points).query(points, k=5)
        clean[rows[distances[:, -1] > radius], cols[distances[:, -1] > radius]] = False
    return clean, {'source': 'native_depth_rgb_and_metric_neighbor_support',
                   'input_points': int(np.count_nonzero(mask)),
                   'green_rejected': int(np.count_nonzero(mask))-len(points),
                   'unsupported_rejected': len(points)-int(np.count_nonzero(clean)),
                   'retained_points': int(np.count_nonzero(clean)),
                   'neighbor_count_including_self': 5, 'support_radius_m': radius,
                   'depth_modified': False}


def measured_grasp_geometry(points_plane, camera_to_plane, intrinsics):
    """Estimate an upper-body grasp using measured points, never a class size.

    Low boundary points do not shift the upper-body centre. The complete raw
    component remains available to the caller for collision checking.
    """
    points = np.asarray(points_plane, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('抓取测量需要Nx3点云')
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .005)]
    if len(points) < 40:
        raise ShapeGeometryNotReady('有效工件深度点不足，无法测量抓取位置')
    center_hint = np.median(points, axis=0)
    steps = plane_pixel_steps(camera_to_plane, intrinsics, center_hint)
    spacing = float(np.max(np.linalg.norm(steps[:, :2], axis=1)))
    neighbor_distances, _ = cKDTree(points).query(points, k=5)
    keep = neighbor_distances[:, -1] <= max(.004, 3.5*spacing)
    clean = points[keep]
    if len(clean) < 40 or keep.mean() < .60:
        raise ShapeGeometryNotReady('连续的有效深度点不足，无法测量夹持宽度')
    height = float(np.quantile(clean[:, 2], .98))
    upper = clean[clean[:, 2] >= .60*height]
    if len(upper) < 30:
        raise ShapeGeometryNotReady('工件上部有效点不足，无法定位抓取中心')
    covariance = np.cov(upper[:, :2].T)
    values, vectors = np.linalg.eigh(covariance)
    projected = upper[:, :2] @ vectors
    lo, hi = np.quantile(projected, [.01, .99], axis=0)
    center = (.5*(lo+hi)) @ vectors.T
    orientation_confidence = float((values[-1]-values[0])/max(values.sum(), 1e-12))
    yaw = float(np.arctan2(vectors[1, -1], vectors[0, -1])) if orientation_confidence >= .20 else 0.
    closing_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    # Existing stock-gripper convention: TCP 10 mm behind the fingertips.
    # Measure all visible points swept by the contact band, including shoulders.
    tcp_height = max(.020, height-.015)
    band_bottom = tcp_height-.010
    contact = clean[clean[:, 2] >= band_bottom]
    if len(contact) < 30 or height-band_bottom < .005:
        raise ShapeGeometryNotReady('可见夹持带高度或有效深度不足')
    steps = plane_pixel_steps(camera_to_plane, intrinsics, np.r_[center, height])
    padding = float(np.sum(np.abs(steps @ closing_axis)))
    closing = contact @ closing_axis
    closing_lower, closing_upper = np.quantile(closing, [.01, .99])
    width = float(closing_upper-closing_lower+padding)
    # Sparse but connected edge points can move a 1%/99% quantile between
    # frames. Use the central 90% only as a temporal consistency statistic.
    # Opening feasibility and collision geometry keep the wider envelope.
    stable_lower, stable_upper = np.quantile(closing, [.05, .95])
    stability_width = float(stable_upper-stable_lower+padding)
    center_projection = float(np.dot(np.r_[center, 0.], closing_axis))
    side_extents = [float(center_projection-closing_lower+padding/2),
                    float(closing_upper-center_projection+padding/2)]
    required_centered_opening = 2*max(side_extents)
    footprint = np.sort(hi-lo)[::-1]
    return {
        'source': 'measured_depth_contact_band', 'center_xy_m': center.tolist(),
        'height_m': height, 'yaw_rad': yaw, 'orientation_confidence': orientation_confidence,
        'grasp_width_m': width, 'tcp_height_m': tcp_height,
        'stability_width_m': stability_width,
        'stability_width_quantiles': [.05, .95],
        'required_centered_opening_m': required_centered_opening,
        'contact_side_extents_from_center_m': side_extents,
        'upper_visible_footprint_m': footprint.tolist(),
        'contact_band_min_height_m': band_bottom,
        'metric_pixel_padding_m': padding,
        'input_points': len(points), 'retained_points': len(clean),
        'isolated_points_removed': int((~keep).sum()),
        'upper_points': len(upper), 'contact_band_points': len(contact),
        'dimensions_note': 'visible point-cloud measurements; hidden surfaces are not measured',
    }


def shape_observation(frame, depth_to_color, component, points_base, points_plane,
                      plane_to_base, camera_to_plane, plane, hull, *,
                      recognizer=None, diagnostic=None):
    """Keep recognition and measured graspability as separate results."""
    clean_mask = exclude_green_belt(frame, depth_to_color, component.mask)
    measurement = measured_grasp_geometry(points_plane[clean_mask], camera_to_plane,
                                          frame.depth_intrinsics.matrix)
    if recognizer is None:
        recognizer = make_shape_recognizer()
    recognition = recognizer.match(frame.depth_m, frame.depth_intrinsics.matrix,
                                   camera_to_plane, clean_mask)
    # A category match supplies an identity only. Its CAD pose, scale and size
    # are deliberately not read by any part of the grasp calculation below.
    identified = bool(recognition.get('accepted')) and int(recognition.get('category_id', -1)) >= 0
    category = int(recognition['category_id']) if identified else -1
    object_id = recognition.get('object_id') if identified else None
    center_xy = np.asarray(measurement['center_xy_m'])
    center = plane_to_base[:3, :3] @ np.r_[center_xy, 0.] + plane_to_base[:3, 3]
    height, width = measurement['height_m'], measurement['grasp_width_m']
    shape = {'status': 'recognition_independent_of_metric_cad_fit',
             'note': 'class uses normalized shape; grasp uses actual depth'}
    if diagnostic is not None:
        diagnostic({'object_points': points_base[component.mask], 'mask': component.mask,
                    'plane': plane, 'height': height,
                    'footprint_m': component.feature_vector[:2].tolist(),
                    'shape_consistency': shape, 'cylinder_fit': {'accepted': False, 'reasons': ['not_requested']},
                    'recognition': recognition, 'grasp_measurement': measurement})
    # Physical limits concern this gripper and observed surroundings, not
    # membership of a CAD category or a print-to-CAD size tolerance.
    if not .020 <= height <= .12:
        raise ShapeGeometryNotReady(f'实测工件高度{height*1000:.1f} mm超出本抓取入口支持范围')
    centered_opening = measurement['required_centered_opening_m']
    if width <= 0 or centered_opening+.020 > .100:
        raise ShapeGeometryNotReady(f'实测夹持方向宽{width*1000:.1f} mm，围绕抓取中心需开口{centered_opening*1000:.1f} mm，'
                         '100 mm夹爪需每侧10 mm余量')
    if cv2.pointPolygonTest(hull, tuple(map(float, center_xy)), True) < .06:
        raise ShapeGeometryNotReady('工件靠近带面边缘，张开夹爪的边缘余量不足')
    yaw = measurement['yaw_rad']
    n = plane_to_base[:3, 2]
    x = plane_to_base[:3, :3] @ np.array([np.cos(yaw), np.sin(yaw), 0.])
    grasp = np.eye(4)
    grasp[:3, :3] = np.column_stack((x, np.cross(-n, x), -n))
    grasp[:3, 3] = center+n*measurement['tcp_height_m']
    valid = (frame.depth_m > 0) & np.isfinite(points_base).all(axis=2)
    return {
        'grasp': grasp, 'center': center, 'height': height, 'width': width,
        'footprint_m': component.feature_vector[:2].tolist(),
        'open_clearance_per_side_m': (.100-centered_opening)/2,
        'shape_consistency': shape, 'cylinder_fit': {'accepted': False, 'reasons': ['not_requested']},
        'target_geometry_source': 'measured_depth_contact_band',
        'grasp_measurement': measurement, 'recognition': recognition,
        'category_id': category, 'object_id': object_id,
        'category': object_id if identified else 'unknown; shape recognition unresolved',
        'sorting_destination': None,  # A separate application maps agreed classes to bins.
        'localization_points': measurement['upper_points'],
        'plane': plane, 'mask': component.mask, 'object_points': points_base[component.mask],
        'scene_points': points_base[valid & ~component.mask], 'points': points_base,
        'orientation_confidence': measurement['orientation_confidence'],
    }

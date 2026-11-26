"""Conservative validation against the same exposure's native target.

This bounds changes; it does not certify learned depth against physical truth.
No temporal fusion, depth replacement or inferred hidden surfaces are used.
"""
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree


def filter_reconstruction(native_points, native_mask, native_depth_m,
                          candidate_mask, candidate_points, candidate_depth_m,
                          plane_to_base, *, min_native_coverage=.80, max_center_shift_m=.006):
    started = time.monotonic()
    native = np.asarray(native_points, dtype=float)
    points = np.asarray(candidate_points, dtype=float)
    mask = np.asarray(candidate_mask, dtype=bool)
    seed = np.asarray(native_mask, dtype=bool)
    depths = np.asarray(candidate_depth_m, dtype=float)
    report = dict(policy='same_exposure_native_anchored_v1', accepted=False,
        native_reference_is_ground_truth=False, left_right_consistency_checked=False,
        limits=dict(same_pixel_depth_delta_m=.010, nearest_native_distance_m=.006,
                    completion_radius_pixels=2, minimum_points=70,
                    minimum_candidate_retained_fraction=.70, minimum_native_coverage_fraction=min_native_coverage,
                    maximum_extent_growth_m=.010, minimum_extent_fraction=.80,
                    maximum_center_shift_m=max_center_shift_m),
        input_points=len(points), native_points=len(native), reasons=[])
    clean = np.zeros(mask.shape, dtype=bool)
    def finish():
        report['wall_s'] = time.monotonic()-started
        return clean, report
    if (native.ndim != 2 or native.shape[1:] != (3,) or len(native)<70
            or not np.isfinite(native).all() or seed.shape != mask.shape
            or np.asarray(native_depth_m).shape != mask.shape
            or points.shape != (int(mask.sum()),3) or depths.shape != (len(points),)):
        report['reasons'].append('invalid_native_reference_or_candidate_grid')
        return finish()
    indices = np.flatnonzero(mask)
    reference_depth = np.asarray(native_depth_m).ravel()[indices]
    anchored = seed.ravel()[indices] & np.isfinite(reference_depth) & (reference_depth>0)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(depths) & (depths>0)
    agrees = ~anchored | (np.abs(depths-reference_depth)<=.010)
    near = np.zeros(len(points),dtype=bool)
    if finite.any():
        distance = cKDTree(native).query(points[finite],workers=1)[0]
        near[finite] = distance<=.006
    # Only tiny holes/edge gaps close to measured support may be completed.
    support = cv2.dilate(seed.astype(np.uint8),np.ones((5,5),np.uint8)).ravel()[indices]>0
    keep = finite & agrees & near & support
    report.update(nonfinite_points=int((~finite).sum()),
        same_pixel_depth_rejected=int((finite & ~agrees).sum()),
        native_distance_rejected=int((finite & ~near).sum()),
        unsupported_pixels=int((~support).sum()))
    clean.ravel()[indices[keep]] = True
    # Keep every component with native support: holes/occlusion can split a
    # real part into several visible regions. Discard only unsupported islands.
    count, labels = cv2.connectedComponents(clean.astype(np.uint8),connectivity=8)
    if count>1:
        support_counts = np.bincount(labels[clean & seed],minlength=count)
        support_counts[0] = 0
        before = int(clean.sum())
        clean &= (support_counts>0)[labels]
        report['disconnected_pixels_removed'] = before-int(clean.sum())
    selected = points[clean.ravel()[indices]]
    report['retained_points'] = len(selected)
    report['candidate_retained_fraction'] = len(selected)/max(1,len(points))
    if len(selected)<70:
        report['reasons'].append('insufficient_supported_points')
        return finish()
    coverage = float(np.mean(cKDTree(selected).query(native,workers=1)[0]<=.006))
    transform = np.asarray(plane_to_base)
    old = (native-transform[:3,3])@transform[:3,:3]
    new = (selected-transform[:3,3])@transform[:3,:3]
    old_bounds = np.quantile(old,[.01,.99],axis=0)
    new_bounds = np.quantile(new,[.01,.99],axis=0)
    old_extent = old_bounds[1]-old_bounds[0]
    new_extent = new_bounds[1]-new_bounds[0]
    center_shift = float(np.linalg.norm(new_bounds.mean(axis=0)-old_bounds.mean(axis=0)))
    report.update(native_coverage_fraction=coverage,center_shift_m=center_shift,
        center_definition='midpoint_of_1_to_99_percentile_belt_axis_bounds',
        native_extent_m=old_extent.tolist(),filtered_extent_m=new_extent.tolist())
    if report['candidate_retained_fraction']<.70:report['reasons'].append('excessive_candidate_loss')
    if coverage<min_native_coverage:report['reasons'].append('insufficient_native_shape_coverage')
    if np.any(new_extent>old_extent+.010):report['reasons'].append('shape_expanded')
    # A 3 mm allowance avoids ratio instability on thin surfaces.
    if np.any(new_extent+.003<.80*old_extent):report['reasons'].append('shape_collapsed')
    if center_shift>max_center_shift_m:report['reasons'].append('center_shifted')
    report['accepted'] = not report['reasons']
    return finish()

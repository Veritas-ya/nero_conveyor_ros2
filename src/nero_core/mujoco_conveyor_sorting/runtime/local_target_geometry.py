"""Seed-local workpiece processing with full native belt collision geometry.

The crop is a view of ONE calibrated exposure, not a new camera observation.
Ambiguous boundaries and unsupported modes use the original full detector.
"""
from dataclasses import replace
import time

import cv2
import numpy as np

from .belt_surface import BeltSurface, _geometry_fingerprint
from .frame_products import frame_products, world_points
from ..calibration.geometry import inverse


def seed_touches_boundary(foreground, seed_rows, seed_cols):
    """Only components joined to this seed can truncate this workpiece."""
    _, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    selected = np.unique(labels[seed_rows,seed_cols])
    selected = selected[selected != 0]
    edge_labels = np.r_[labels[:2].ravel(), labels[-2:].ravel(),
                        labels[:,:2].ravel(), labels[:,-2:].ravel()]
    return not len(selected) or bool(np.isin(edge_labels, selected).any())


def detect_seed_local(frame, depth_to_color, *, detector, **kwargs):
    from .single_pick import PickRejected, TargetNotReady
    started = time.monotonic()
    report = {'policy':'seed_local_with_full_native_belt_scene', 'applied':False,
              'native_depth_pixels':int(frame.depth_m.size), 'depth_modified':False}

    def fallback(reason):
        report.update(fallback_reason=reason, local_attempt_wall_s=time.monotonic()-started)
        try:
            result = detector(frame, depth_to_color, **kwargs)
        except (PickRejected, TargetNotReady) as error:
            error.details = {**error.details, 'local_processing':report}
            raise
        result['local_processing'] = report
        return result

    surface = kwargs.get('measured_surface')
    seeds = np.asarray(kwargs.get('target_seed_pixel_indices'))
    if (kwargs.get('target_shape') != 'simple'
            or kwargs.get('scene_scope') != 'measured_belt_surface_only'
            or not isinstance(surface, BeltSurface) or not surface.matches_frame(frame, depth_to_color)
            or frame.rgb_aligned_to_depth or frame.color_intrinsics is None
            or seeds.ndim != 1 or not seeds.size or seeds.dtype.kind not in 'iu'
            or np.any(seeds < 0) or np.any(seeds >= frame.depth_m.size)):
        return fallback('unsupported_input_or_mode')
    height, width = frame.depth_m.shape
    rows, cols = np.unravel_index(seeds, (height, width))
    # Context is not a maximum object size. Any truncation below falls back.
    halo = 32
    y0, y1 = max(0, int(rows.min())-halo), min(height, int(rows.max())+halo+1)
    x0, x1 = max(0, int(cols.min())-halo), min(width, int(cols.max())+halo+1)
    report.update(crop_yxyx=[y0,y1,x0,x1], local_depth_pixels=(y1-y0)*(x1-x0))
    if y0 == 0 or x0 == 0 or y1 == height or x1 == width:
        return fallback('crop_reaches_native_image_boundary')
    if (y1-y0)*(x1-x0) >= frame.depth_m.size/2:
        return fallback('crop_not_smaller_than_half_frame')

    # Full native geometry remains necessary for the existing belt collision
    # scope. Build it once; give the crop exact slices, preserving arithmetic.
    products = frame_products(frame, depth_to_color)
    points = world_points(frame, depth_to_color)
    transform = inverse(surface.plane_to_base)
    flat_key = ('belt_coordinates', surface.plane_to_base.tobytes())
    flat = products.get(flat_key, lambda: points @ transform[:3,:3].T + transform[:3,3])
    local_flat = flat[y0:y1,x0:x1]
    depth = frame.depth_m[y0:y1,x0:x1]
    maximum = kwargs.get('maximum_target_height_m')
    maximum = .12 if maximum is None else float(maximum)
    raw_foreground = ((depth > 0) & np.isfinite(local_flat).all(axis=2)
                      & (local_flat[...,2] >= .012) & (local_flat[...,2] <= maximum))
    # Check unfiltered foreground, before denoising could hide a cut edge.
    if seed_touches_boundary(raw_foreground, rows-y0, cols-x0):
        return fallback('foreground_touches_crop_boundary')
    k = frame.depth_intrinsics
    cropped = replace(frame, depth_m=depth, depth_intrinsics=replace(k,
        width=x1-x0, height=y1-y0, cx=k.cx-x0, cy=k.cy-y0))
    # Projection uses original absolute pixel arithmetic, including tie cases.
    object.__setattr__(cropped, '_depth_pixel_origin', (y0,x0))
    object.__setattr__(cropped, '_uncropped_depth_intrinsics', k)
    local_products = frame_products(cropped, depth_to_color)
    local_products.get('native_deprojection', lambda: points[y0:y1,x0:x1])
    local_products.get(flat_key, lambda: local_flat)
    local_surface = replace(surface, geometry_fingerprint=_geometry_fingerprint(cropped, depth_to_color))
    local_kwargs = {**kwargs, 'measured_surface':local_surface,
        'target_seed_pixel_indices':(rows-y0)*(x1-x0)+cols-x0}
    diagnostic_rows = []
    diagnostic = kwargs.get('diagnostic')
    if diagnostic is not None:
        local_kwargs['diagnostic'] = diagnostic_rows.append
    try:
        result = detector(cropped, depth_to_color, **local_kwargs)
    except (PickRejected, TargetNotReady) as error:
        return fallback('local_detector_rejected: '+str(error))

    def expand(mask):
        full = np.zeros((height,width), dtype=mask.dtype)
        full[y0:y1,x0:x1] = mask
        return full

    local_mask = result['mask']
    if (local_mask[:2].any() or local_mask[-2:].any()
            or local_mask[:,:2].any() or local_mask[:,-2:].any()):
        return fallback('completed_target_touches_crop_boundary')
    silhouette = result.get('visible_target_rgb_mask')
    if silhouette is None:
        # A cropped depth island alone cannot certify the complete outline.
        return fallback('no_closed_rgb_outline')
    from .visible_target import visible_depth_mask
    full_support, _ = visible_depth_mask(frame, depth_to_color, silhouette, flat, surface.plane)
    # Check ALL native rays on the selected RGB outline, including low edges
    # and disconnected depth patches outside the crop. Do not crop this proof.
    outside = full_support.copy()
    outside[y0:y1,x0:x1] = False
    if outside.any():
        return fallback('rgb_outline_has_depth_outside_crop')

    mask = expand(local_mask)
    rejected = expand(result.pop('_rejected_target_depth_mask'))
    valid = (frame.depth_m > 0) & np.isfinite(points).all(axis=2)
    scene_mask = valid & ~mask & ~rejected
    scene_input_count = int(scene_mask.sum())
    region = surface.region_metadata()
    bounds = np.asarray(region.get('scene_bounds_xy_m', region['bounds_xy_m']))
    epsilon = 1e-8
    scene_mask &= (np.abs(flat[...,2]) <= .005+epsilon)
    for axis in (0,1):
        scene_mask &= ((flat[...,axis] >= bounds[axis,0]-epsilon)
                       & (flat[...,axis] <= bounds[axis,1]+epsilon))
    result.update(mask=mask, object_points=points[mask], points=points, points_plane=flat,
        scene_points=points[scene_mask], scene_pixel_indices=np.flatnonzero(scene_mask),
        scene_input_count_before_prefilter=scene_input_count, _frame_products=products,
        _rejected_target_depth_mask=rejected)
    report.update(applied=True, full_rgb_outline_checked=True, full_belt_scene_preserved=True,
                  wall_s=time.monotonic()-started)
    result['local_processing'] = report
    if diagnostic is not None:
        for row in diagnostic_rows:
            row = dict(row)
            if 'mask' in row: row['mask'] = expand(row['mask'])
            if 'candidate_masks' in row: row['candidate_masks'] = [expand(m) for m in row['candidate_masks']]
            diagnostic({**row, 'local_processing':report})
    return result

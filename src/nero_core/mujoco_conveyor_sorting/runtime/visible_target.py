"""Complete a selected visible part using its observed RGB silhouette.

The ordinary depth detector still selects the target. Its tall foreground is
a seed, not the object's outline. No hidden geometry or replacement native
depth is generated here. Estimated stereo depth may use the same RGB outline.
"""
from __future__ import annotations

from contextlib import contextmanager

import cv2
import numpy as np

from .depth_diagnostics import measured_stage


@contextmanager
def _timed(timings, name):
    if timings is None:
        yield
        return
    with measured_stage('visible_' + name, timings):
        yield


def _opencv_projection(color, matrix, distortion, timings_s):
    with _timed(timings_s, 'cv_project_points'):
        return cv2.projectPoints(color, np.zeros(3), np.zeros(3), matrix,
                                 distortion)[0].reshape(-1, 2)


def _polynomial_distortion(distortion):
    """Use the polynomial model for rational OpenCV coefficients only when exact.

    Orbbec commonly supplies 8 coefficients with k4/k5/k6 exactly zero.
    The rational denominator is then exactly one; any nonzero denominator
    coefficient retains the full OpenCV/fallback path.
    """
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if len(coefficients) == 8 and np.all(coefficients[5:] == 0.):
        return coefficients[:5]
    return coefficients


def _project_points(color, matrix, distortion, backend, timings_s=None, diagnostics=None):
    """The existing OpenCV pinhole formula, without requesting its Jacobian.

    This path does not reinterpret distortion_model (the old call always used
    cv2.projectPoints). 8/12/14 coefficients retain that exact OpenCV call.
    Nonfinite/extreme values and rounding ties also retain OpenCV arithmetic.
    """
    if backend not in ('auto', 'opencv'):
        raise ValueError('projection backend must be auto or opencv')
    coefficients = _polynomial_distortion(distortion)
    supported = (len(coefficients) in (0, 4, 5) and np.isfinite(coefficients).all()
                 and np.isfinite(matrix).all())
    if backend == 'opencv' or not supported:
        if diagnostics is not None:
            diagnostics['projection_backend'] = 'opencv'
        return _opencv_projection(color, matrix, distortion, timings_s)
    with _timed(timings_s, 'no_jacobian_projection'):
        k = np.zeros(5)
        k[:len(coefficients)] = coefficients
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            reciprocal = np.ones(len(color))
            np.divide(1., color[:, 2], out=reciprocal, where=color[:, 2] != 0)
            x, y = color[:, 0] * reciprocal, color[:, 1] * reciprocal
            r2 = x*x + y*y
            r4 = r2*r2
            r6 = r4*r2
            a1, a2, a3 = 2*x*y, r2 + 2*x*x, r2 + 2*y*y
            radial = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
            xd = x*radial + k[2]*a1 + k[3]*a2
            yd = y*radial + k[2]*a3 + k[3]*a1
            uv = np.column_stack((xd*matrix[0, 0] + matrix[0, 2],
                                  yd*matrix[1, 1] + matrix[1, 2]))
            # Keep OpenCV handling of nonfinite identity-transform products,
            # overflow, and values within its rounding-boundary error budget.
            tolerance = 64*np.finfo(float).eps*np.maximum(1., np.abs(uv))
            ties = np.any(np.abs((uv-.5) - np.rint(uv-.5)) <= tolerance, axis=1)
            fallback = (ties | ~np.isfinite(color[:, 0]) | ~np.isfinite(color[:, 1])
                        | ~np.isfinite(color[:, 2])
                        | ~np.isfinite(uv).all(axis=1)
                        | (np.maximum(np.abs(x), np.abs(y)) > 2.))
    if diagnostics is not None:
        diagnostics.update(projection_backend='no_jacobian',
                           opencv_fallback_points=int(fallback.sum()))
    if fallback.any():
        uv[fallback] = _opencv_projection(color[fallback], matrix, distortion, timings_s)
    return uv


class _RoiUnavailable(Exception):
    """An optimization certificate failed; use the original full image."""


def _deproject_pixels(depth, valid, rows, cols, intrinsic):
    """Same three separate float64 operations; reuse the output allocation."""
    z = depth[valid]
    camera = np.empty((len(rows), 3), dtype=float)
    for output, pixels, center, focal in (
            (camera[:, 0], cols, intrinsic.cx, intrinsic.fx),
            (camera[:, 1], rows, intrinsic.cy, intrinsic.fy)):
        np.subtract(pixels, center, out=output)
        np.multiply(output, z, out=output)
        np.divide(output, focal, out=output)
    camera[:, 2] = z
    return camera


def _conservative_rgb_roi(color, matrix, distortion, labels, diagnostics):
    """Keep every point that could round to any nonzero RGB label.

    Bound the absolute radial/tangential displacement over ALL current rays,
    then expand the label box by that bound, 0.5 for np.rint, and 1 pixel for
    arithmetic roundoff. No inverse distortion, nominal depth or seed crop.
    Unsupported / numerically unbounded cases request full-image fallback.
    """
    coefficients = _polynomial_distortion(distortion)
    if (len(coefficients) not in (0, 4, 5) or not np.isfinite(coefficients).all()
            or np.any(np.abs(coefficients) > 1.) or not np.isfinite(matrix).all()
            or np.any(np.abs(matrix) > 1e5)):
        raise _RoiUnavailable('unsupported_or_unbounded_camera')
    if (not np.isfinite(color[:, 0]).all() or not np.isfinite(color[:, 1]).all()
            or not np.isfinite(color[:, 2]).all() or np.any(color[:, 2] < 1e-9)):
        raise _RoiUnavailable('nonfinite_or_near_camera_plane')
    reciprocal = 1./color[:, 2]
    x, y = color[:, 0]*reciprocal, color[:, 1]*reciprocal
    X, Y = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    if max(X, Y) > 2.:
        raise _RoiUnavailable('normalized_ray_outside_certificate')
    rows = np.flatnonzero(np.any(labels, axis=1))
    cols = np.flatnonzero(np.any(labels, axis=0))
    if not len(rows):
        return np.zeros(len(color), bool)
    k = np.zeros(5)
    k[:len(coefficients)] = np.abs(coefficients)
    # Positive sums/products, rounded outward with a further relative guard.
    r2 = X*X + Y*Y
    radial = k[0]*r2 + k[1]*r2*r2 + k[4]*r2*r2*r2
    dx = abs(matrix[0, 0])*(X*radial + k[2]*2*X*Y + k[3]*(r2+2*X*X))
    dy = abs(matrix[1, 1])*(Y*radial + k[2]*(r2+2*Y*Y) + k[3]*2*X*Y)
    dx = np.nextafter(dx*(1+64*np.finfo(float).eps), np.inf)
    dy = np.nextafter(dy*(1+64*np.finfo(float).eps), np.inf)
    # Under the gates above all formula terms are <1.3e8 pixels; even a
    # 128-operation IEEE-754 error bound is <1e-5 pixel. One pixel dominates
    # it, including cancellation and either dot-product summation order.
    halo = 1.5
    u, v = x*matrix[0, 0]+matrix[0, 2], y*matrix[1, 1]+matrix[1, 2]
    keep = ((u >= cols.min()-.5-dx-1.) & (u <= cols.max()+.5+dx+1.)
            & (v >= rows.min()-.5-dy-1.) & (v <= rows.max()+.5+dy+1.))
    if diagnostics is not None:
        diagnostics.update(roi_source='current_depth_forward_distortion_bound',
            rgb_label_bbox_xyxy=[int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
            distortion_displacement_bound_px=[float(dx), float(dy)],
            rounding_and_numeric_halo_px=halo,
            roi_input_points=len(color), roi_projected_points=int(keep.sum()))
    return keep


def project_rgb_labels(frame, depth_to_color, labels, *, depth=None, pixel_mask=None,
                       timings_s=None, projection_backend='auto', target_roi=False,
                       height_mask=None, diagnostics=None):
    from .frame_products import frame_products
    with frame_products(frame, depth_to_color).lock:
        return _project_rgb_labels(frame, depth_to_color, labels, depth=depth,
            pixel_mask=pixel_mask, timings_s=timings_s, projection_backend=projection_backend,
            target_roi=target_roi, height_mask=height_mask, diagnostics=diagnostics)


def _project_rgb_labels(frame, depth_to_color, labels, *, depth=None, pixel_mask=None,
                       timings_s=None, projection_backend='auto', target_roi=False,
                       height_mask=None, diagnostics=None):
    """Look up color labels at genuine native or reconstructed depth pixels."""
    with _timed(timings_s, 'valid_pixels'):
        depth = np.asarray(frame.depth_m if depth is None else depth)
        valid = np.isfinite(depth) & (depth > 0)
        if pixel_mask is not None:
            valid &= np.asarray(pixel_mask, dtype=bool)
        rows, cols = np.nonzero(valid)
        result = np.zeros(depth.shape, dtype=labels.dtype)
    if not len(rows):
        return result
    projection_cache = None
    requested_rows, requested_cols = rows, cols
    if (not target_roi and depth is frame.depth_m
            and not depth.flags.writeable and projection_backend == 'auto'
            and labels.shape == (frame.color_intrinsics.height, frame.color_intrinsics.width)):
        from .frame_products import frame_products
        products = frame_products(frame, depth_to_color)
        projection_cache = products.get('native_rgb_pixels', lambda: {
            'computed': np.zeros(depth.shape, bool),
            'pixels': np.zeros((*depth.shape, 2), np.int32),
            'inside': np.zeros(depth.shape, bool)})
        # The caller owns this exposure's work; cache updates are serialized.
        missing = ~projection_cache['computed'][rows, cols]
        rows, cols = rows[missing], cols[missing]
        valid = np.zeros(depth.shape, bool)
        valid[rows, cols] = True
        if not len(rows):
            inside = projection_cache['inside'][requested_rows, requested_cols]
            pixels = projection_cache['pixels'][requested_rows[inside], requested_cols[inside]]
            result[requested_rows[inside], requested_cols[inside]] = labels[pixels[:, 1], pixels[:, 0]]
            return result
    with _timed(timings_s, 'deprojection'):
        y0, x0 = getattr(frame, '_depth_pixel_origin', (0,0))
        intrinsic = getattr(frame, '_uncropped_depth_intrinsics', frame.depth_intrinsics)
        camera = _deproject_pixels(depth, valid, rows+y0, cols+x0, intrinsic)
    with _timed(timings_s, 'd2c'):
        color = camera @ depth_to_color[:3, :3].T + depth_to_color[:3, 3]
    matrix = frame.color_intrinsics.matrix
    distortion = np.asarray(frame.color_intrinsics.distortion)
    if diagnostics is not None:
        diagnostics['valid_projection_input_points'] = len(color)
    if target_roi:
        with _timed(timings_s, 'roi_selection'):
            # Preserve the original full-batch deprojection and D2C arithmetic
            # before selecting points. Changing GEMM row count can change its
            # reduction kernel / last bits at rounded pixel boundaries.
            if height_mask is not None:
                eligible = height_mask[valid]
                rows, cols, color = rows[eligible], cols[eligible], color[eligible]
                if not len(color):
                    return result
            keep = _conservative_rgb_roi(color, matrix, distortion, labels, diagnostics)
            rows, cols, color = rows[keep], cols[keep], color[keep]
        if not len(color):
            return result
    uv = _project_points(color, matrix, distortion, projection_backend, timings_s, diagnostics)
    with _timed(timings_s, 'pixel_round_label'):
        h, w = labels.shape
        pixels = np.rint(np.nan_to_num(np.clip(uv, -1, max(h, w)), nan=-1)).astype(int)
        inside = (np.isfinite(uv).all(axis=1) & (color[:, 2] > 0)
                  & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
        result[rows[inside], cols[inside]] = labels[pixels[inside, 1], pixels[inside, 0]]
        if projection_cache is not None:
            projection_cache['computed'][rows, cols] = True
            projection_cache['pixels'][rows, cols] = pixels
            projection_cache['inside'][rows, cols] = inside
            known_inside = projection_cache['inside'][requested_rows, requested_cols]
            known_pixels = projection_cache['pixels'][requested_rows[known_inside], requested_cols[known_inside]]
            result[requested_rows[known_inside], requested_cols[known_inside]] = labels[
                known_pixels[:, 1], known_pixels[:, 0]]
    return result


def selected_rgb_silhouette(frame, depth_to_color, seed_mask):
    """Recover closed non-green RGB components touched by the selected seed.

    Use exactly the existing belt HSV definition. An open/background component
    is only a failed completion hint; the original depth target is retained.
    """
    from .frame_products import green_rgb_mask
    green = green_rgb_mask(frame, depth_to_color)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~green).astype('uint8'), connectivity=8)
    seed_labels = project_rgb_labels(frame, depth_to_color, labels,
                                     pixel_mask=seed_mask)[np.asarray(seed_mask, bool)]
    indices = np.unique(seed_labels[seed_labels != 0])
    info = {'source': 'selected_depth_seed_connected_non_green_rgb',
            'rgb_component_ids': indices.tolist(), 'hidden_surface_added': False,
            'native_depth_modified': False}
    if not len(indices):
        return None, {**info, 'available': False, 'reason': 'no_color_component_for_selected_seed'}
    h, w = green.shape
    for index in indices:
        x, y, width, height, _ = stats[index]
        if x == 0 or y == 0 or x+width == w or y+height == h:
            return None, {**info, 'available': False, 'reason': 'rgb_component_open_to_image_boundary'}
    mask = np.isin(labels, indices)
    return mask, {**info, 'available': True, 'rgb_visible_pixels': int(mask.sum()),
                  'rgb_component_boxes_xywh': stats[indices, :4].tolist()}


def visible_depth_mask(frame, depth_to_color, silhouette, flat_points, plane, *, depth=None,
                       timings_s=None, projection_backend='auto', conservative_roi=True,
                       diagnostics=None, pixel_mask=None):
    """Select depth on this observed outline, including low shoulders/base.

    The lower band is the original measured plane's 4 mm RANSAC uncertainty,
    not a new foreground admission rule. The original tall seed selected the
    part already; points remain bound to its observed RGB outline.
    """
    source_depth = frame.depth_m if depth is None else depth
    uncertainty = float(plane.get('inlier_threshold_z_m', .004))
    with _timed(timings_s, 'height_filter'):
        height_valid = (np.isfinite(flat_points[..., 0]) & np.isfinite(flat_points[..., 1])
                        & np.isfinite(flat_points[..., 2])
                        & (flat_points[..., 2] >= -uncertainty)
                        & (flat_points[..., 2] <= .12))
        if pixel_mask is not None:
            if np.shape(pixel_mask) != np.shape(source_depth):
                raise ValueError('pixel_mask must match native depth grid')
            height_valid &= np.asarray(pixel_mask, bool)
    with _timed(timings_s, 'project_rgb_labels'):
        try:
            support = project_rgb_labels(frame, depth_to_color, np.asarray(silhouette, np.uint8),
                depth=source_depth, timings_s=timings_s, projection_backend=projection_backend,
                height_mask=height_valid if conservative_roi else None,
                pixel_mask=pixel_mask, target_roi=conservative_roi, diagnostics=diagnostics) != 0
        except _RoiUnavailable as error:
            if diagnostics is not None:
                diagnostics['roi_fallback_reason'] = str(error)
            support = project_rgb_labels(frame, depth_to_color, np.asarray(silhouette, np.uint8),
                depth=source_depth, timings_s=timings_s, projection_backend=projection_backend,
                pixel_mask=pixel_mask, diagnostics=diagnostics) != 0
    support &= height_valid
    return support, -uncertainty

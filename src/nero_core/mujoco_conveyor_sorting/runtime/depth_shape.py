"""Offline upper-contour diagnostics under an upright-cylinder shape prior.

Input points are already in the support-plane frame (z is height in metres).
This module has no camera or robot IO. A consistent circular upper contour is
evidence for a cylinder candidate, not proof of the hidden or complete sidewall.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _sample_polygon(vertices, spacing):
    """Sample perimeter, so four rectangle corners cannot fit a perfect circle."""
    samples = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        count = max(2, min(512, int(np.ceil(np.linalg.norm(b-a)/spacing))))
        samples.append(a + np.arange(count)[:, None]/count * (b-a))
    return np.concatenate(samples)


def _fit_slice(points, threshold, pixel):
    selected = points[points[:, 2] >= threshold, :2]
    result = {'height_threshold_m': float(threshold), 'selected_count': len(selected),
              'accepted': False, 'reasons': []}
    if len(selected) < 50:
        result['reasons'].append('上层有效点不足50个')
        return result
    # Isolated flying pixels should not become the convex hull. Dense abnormal
    # clusters remain visible to the shape/consistency checks, rather than being
    # silently trimmed until a circle appears.
    distances, _ = cKDTree(selected).query(selected, k=5)
    keep = distances[:, -1] <= max(3.5*pixel, .003)
    clean = selected[keep]
    result.update(retained_count=len(clean), isolated_points_removed=int((~keep).sum()),
                  isolated_excluded_fraction=float((~keep).mean()))
    if len(clean) < 50:
        result['reasons'].append('去除稀疏孤立点后有效点不足50个')
        return result
    try:
        hull = ConvexHull(clean)
    except QhullError:
        result['reasons'].append('上层轮廓退化，无法形成二维外边界')
        return result
    boundary = clean[hull.vertices]
    contour = _sample_polygon(boundary, max(pixel*.5, .00025))
    initial_center = np.mean(boundary, axis=0)
    initial_radius = float(np.mean(np.linalg.norm(boundary-initial_center, axis=1)))
    if initial_radius <= pixel or initial_radius >= .25:
        result['reasons'].append('圆轮廓尺寸不足以分辨或超出诊断范围')
        return result
    fit = least_squares(
        lambda value: np.linalg.norm(contour-value[:2], axis=1)-value[2],
        np.r_[initial_center, initial_radius], loss='soft_l1', f_scale=max(pixel, .0005),
        bounds=([-np.inf, -np.inf, pixel], [np.inf, np.inf, .25]), max_nfev=200)
    center, radius = fit.x[:2], float(fit.x[2])
    residual = np.linalg.norm(contour-center, axis=1)-radius
    rms = float(np.sqrt(np.mean(residual**2)))
    p95 = float(np.quantile(np.abs(residual), .95))
    circularity = float(4*np.pi*hull.volume / (hull.area*hull.area))
    # Coverage is computed from measured outer-hull vertices, never invented
    # perimeter samples. A large unseen angular gap makes the fit ambiguous.
    angles = np.sort(np.mod(np.arctan2(boundary[:, 1]-center[1],
                                     boundary[:, 0]-center[0]), 2*np.pi))
    coverage = float(1 - np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))/(2*np.pi))
    residual_limit = max(1.5*pixel, .0015)
    result.update(center_xy_m=center.tolist(), diameter_m=2*radius,
                  residual_rms_m=rms, residual_p95_m=p95,
                  circularity=circularity, angular_coverage_fraction=coverage,
                  hull_vertex_count=len(boundary), hull_area_m2=float(hull.volume),
                  contour_sample_count=len(contour), residual_limit_m=residual_limit)
    if not fit.success:
        result['reasons'].append('稳健圆拟合未收敛')
    if circularity < .90:
        result['reasons'].append('外轮廓圆度低于0.90，不能判为圆柱候选')
    if coverage < .75:
        result['reasons'].append('外轮廓可观测角覆盖不足75%')
    if rms > residual_limit or p95 > 2.5*residual_limit:
        result['reasons'].append('外轮廓径向残差过大')
    if result['isolated_excluded_fraction'] > .2:
        result['reasons'].append('上层孤立深度点超过20%')
    result['accepted'] = not result['reasons']
    return result


def fit_upright_cylinder(points_plane, pixel_size_m=.002):
    """Return a JSON-safe cylinder-candidate diagnostic, without any IO.

    ``pixel_size_m`` is the approximate metric spacing of one native depth
    pixel, e.g. median(depth)/sqrt(fx*fy); it is NOT a two-side diameter padding.
    ``diameter_m`` is the raw fitted contour diameter. The separate
    ``measurement_margin_m`` describes a two-pixel diagnostic uncertainty and
    must not be interpreted as an independently calibrated accuracy bound.

    Three nested upper-point sets are checked. These are height-threshold
    contours, not independently observed complete cylindrical cross-sections.
    Failure retains all diagnostics and never chooses a fit using true size.
    """
    result = {'accepted': False, 'center_xy_m': None, 'diameter_m': None,
              'reasons': [], 'slices': [],
              'shape_prior': 'upright circular cylinder; upper contour evidence only',
              'scope': 'does not prove hidden sidewall, grasp friction, or physical dimensions',
              'parameters': {'height_fractions': [.55, .65, .75],
                             'minimum_circularity': .90, 'minimum_angular_coverage': .75,
                             'minimum_slice_points': 50, 'height_quantile': .98}}
    points = np.asarray(points_plane, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        result['reasons'].append('输入需要Nx3带面坐标系点云')
        return result
    if not math.isfinite(pixel_size_m) or not .00005 <= pixel_size_m <= .02:
        result['reasons'].append('单像素米制间距无效')
        return result
    pixel = float(pixel_size_m)
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid = points[finite]
    result.update(input_point_count=len(points), valid_point_count=len(valid),
                  invalid_points_removed=int((~finite).sum()), pixel_size_m=pixel,
                  measurement_margin_m=2*pixel)
    if len(valid) < 80:
        result['reasons'].append('有效正高度点不足80个')
        return result
    height = float(np.quantile(valid[:, 2], .98))
    result['height_q98_m'] = height
    if height < .01:
        result['reasons'].append('目标高度不足10mm')
        return result
    slices = [_fit_slice(valid, height*f, pixel) for f in [.55, .65, .75]]
    result['slices'] = slices
    # Report every threshold's exclusion, including the low contour points
    # which motivated this diagnostic. They are not silently discarded.
    for item in slices:
        item['height_excluded_count'] = len(valid)-item['selected_count']
        item['height_excluded_fraction'] = 1-item['selected_count']/len(valid)
    failed = [i+1 for i, item in enumerate(slices) if not item['accepted']]
    if failed:
        result['reasons'].append(f'上层轮廓质量检查未通过：{failed}')
    if any('center_xy_m' not in s for s in slices):
        return result
    centers = np.asarray([s['center_xy_m'] for s in slices])
    diameters = np.asarray([s['diameter_m'] for s in slices])
    center_spread = float(np.max(np.linalg.norm(centers[:, None]-centers[None, :], axis=-1)))
    diameter_spread = float(np.ptp(diameters))
    center_limit, diameter_limit = max(2*pixel, .003), max(3*pixel, .004)
    result['quality'] = {'center_spread_m': center_spread, 'diameter_spread_m': diameter_spread,
                         'center_spread_limit_m': center_limit,
                         'diameter_spread_limit_m': diameter_limit,
                         'max_residual_rms_m': max(s['residual_rms_m'] for s in slices),
                         'min_circularity': min(s['circularity'] for s in slices),
                         'min_angular_coverage': min(s['angular_coverage_fraction'] for s in slices)}
    result['quality']['candidate_center_xy_m'] = np.median(centers, axis=0).tolist()
    result['quality']['candidate_diameter_m'] = float(np.median(diameters))
    if center_spread > center_limit:
        result['reasons'].append('不同上层阈值的拟合中心不一致')
    if diameter_spread > diameter_limit:
        result['reasons'].append('不同上层阈值的拟合直径不一致')
    if result['reasons']:
        return result
    result.update(accepted=True, center_xy_m=np.median(centers, axis=0).tolist(),
                  diameter_m=float(np.median(diameters)),
                  fit_residual_rms_m=float(np.median([s['residual_rms_m'] for s in slices])))
    return result

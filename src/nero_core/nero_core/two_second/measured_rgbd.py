"""Measured native RGB-D helpers migrated verbatim; see deployment provenance."""

import numpy as np
import cv2
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurfaceNotReady

class _ClockReferenceSchedulingDelay(PickRejected):
    pass


def depth_rays(frame):
    if np.any(np.abs(frame.depth_intrinsics.distortion) > 1e-10):
        raise ValueError('existing detector uses pinhole depth; nonzero depth D needs explicit handling')
    y, x = np.indices(frame.depth_m.shape)
    k = frame.depth_intrinsics
    return np.stack(((x-k.cx)/k.fx, (y-k.cy)/k.fy, np.ones_like(x)), axis=-1)

def project_to_rgb(points_depth, frame, depth_to_color):
    """Project native points without OpenCV's full-image Jacobian allocation.

    The common 0/4/5-coefficient OpenCV distortion model is evaluated in
    arrays. Extended distortion models retain the original OpenCV path.
    UVs also include points at zero or negative color-camera Z; the separate
    positive-Z flag lets callers retain their depth and image bounds checks.
    """
    shape = points_depth.shape[:-1]
    points_color = points_depth.reshape(-1, 3) @ depth_to_color[:3, :3].T + depth_to_color[:3, 3]
    intrinsics = frame.color_intrinsics.matrix
    original_distortion = np.asarray(frame.color_intrinsics.distortion)
    distortion = original_distortion
    # The camera's eight-term rational model is exactly the five-term model
    # when its three denominator coefficients are all zero. Keep every other
    # extended model on OpenCV, including arbitrarily small nonzero terms.
    zero_rational = distortion.size == 8 and np.all(distortion.reshape(-1)[5:] == 0.)
    if zero_rational:
        distortion = distortion.reshape(-1)[:5]
    if (distortion.size in (0, 4, 5) and points_color.size
            and points_color.dtype in (np.float32, np.float64)):
        coefficients = np.zeros(5, dtype=np.float64)
        coefficients[:distortion.size] = distortion.reshape(-1)
        k1, k2, p1, p2, k3 = coefficients
        color = points_color.astype(np.float64, copy=False)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # OpenCV uses reciprocal 1 when Z is exactly zero. Preserve that
            # projection as well as negative Z; validity is the caller's job.
            reciprocal = np.ones(len(color), dtype=np.float64)
            np.divide(1., color[:, 2], out=reciprocal, where=color[:, 2] != 0)
            x, y = color[:, 0]*reciprocal, color[:, 1]*reciprocal
            r2 = x*x+y*y
            r4, r6 = r2*r2, r2*r2*r2
            radial = 1+k1*r2+k2*r4+k3*r6
            xy2 = 2*x*y
            u = (x*radial+p1*xy2+p2*(r2+2*x*x))*intrinsics[0, 0]+intrinsics[0, 2]
            v = (y*radial+p1*(r2+2*y*y)+p2*xy2)*intrinsics[1, 1]+intrinsics[1, 2]
            uv = np.column_stack((u, v)).astype(points_color.dtype, copy=False)
        exceptional = ~np.isfinite(uv).all(axis=1)
        if zero_rational:
            # A pixel rounding tie may change label when arithmetic differs
            # in the last bits. Preserve the original 8-term OpenCV result.
            tolerance = 1e-7
            exceptional |= np.any(np.abs((uv-.5)-np.rint(uv-.5)) <= tolerance, axis=1)
        if exceptional.any():
            # Preserve OpenCV's NaN/Inf behavior for exceptional input or
            # overflow without slowing down the ordinary full-image case.
            uv[exceptional] = cv2.projectPoints(
                points_color[exceptional], np.zeros(3), np.zeros(3),
                intrinsics, original_distortion)[0].reshape(-1, 2)
    else:
        uv = cv2.projectPoints(points_color, np.zeros(3), np.zeros(3),
                               intrinsics, distortion)[0].reshape(-1, 2)
    return uv.reshape(*shape, 2), points_color[:, 2].reshape(shape) > 0

def green_belt_points(frame, depth_to_color):
    from mujoco_conveyor_sorting.runtime.frame_products import frame_products
    return frame_products(frame, depth_to_color).get('green_belt_extraction', lambda:
        _green_belt_points(frame, depth_to_color))

def _green_belt_points(frame, depth_to_color):
    from mujoco_conveyor_sorting.runtime.frame_products import green_rgb_mask, world_points
    green = np.uint8(green_rgb_mask(frame, depth_to_color))*255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green)
    if count < 2:
        raise BeltSurfaceNotReady('no green belt found; waiting for another measured frame',
            reason='green_belt_not_visible', details={'green_rgb_pixels': int(np.count_nonzero(green))})
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    green = cv2.erode(np.uint8(labels == selected), np.ones((15, 15), np.uint8))
    rays = depth_rays(frame)
    uv, positive = project_to_rgb(rays * frame.depth_m[..., None], frame, depth_to_color)
    h, w = green.shape
    # Invalid depths may project far outside the color image. Clamp before
    # integer conversion, but still reject them using their original UVs.
    pixels = np.rint(np.nan_to_num(np.clip(uv, -1, max(h,w)), nan=-1)).astype(int)
    inside = positive & (frame.depth_m > 0) & (pixels[..., 0] >= 0) & (pixels[..., 0] < w)
    inside &= (pixels[..., 1] >= 0) & (pixels[..., 1] < h)
    mask = np.zeros(frame.depth_m.shape, dtype=bool)
    mask[inside] = green[pixels[..., 1][inside], pixels[..., 0][inside]] > 0
    world = world_points(frame, depth_to_color)
    if mask.sum() < 2000:
        raise BeltSurfaceNotReady('too few green belt depth points for diagnostic plane fitting',
            reason='insufficient_green_depth', details={
                'green_rgb_pixels_after_erosion': int(np.count_nonzero(green)),
                'valid_green_depth_points': int(mask.sum()),
                'required_green_depth_points': 2000,
                'valid_native_depth_pixels': int(np.count_nonzero(np.isfinite(frame.depth_m) & (frame.depth_m > 0)))})
    return world, mask, green

def fit_plane(points, seed=42):
    """Fit z=a*x+b*y+c using reproducible RANSAC then inlier least squares."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=float)
    if len(points) > 12000:
        points = points[rng.choice(len(points), 12000, replace=False)]
    a = np.c_[points[:, :2], np.ones(len(points))]
    best = np.zeros(len(points), dtype=bool)
    for _ in range(160):
        chosen = rng.choice(len(points), 3, replace=False)
        if np.linalg.cond(a[chosen]) > 1e5:
            continue
        coefficients = np.linalg.solve(a[chosen], points[chosen, 2])
        residual = np.abs(a @ coefficients - points[:, 2])
        mask = residual < .004
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < len(points) * .5:
        raise BeltSurfaceNotReady('green belt points do not support a dominant plane',
            reason='belt_plane_not_dominant', details={'sampled_points': len(points),
                                                      'best_inlier_points': int(best.sum())})
    for _ in range(3):
        coefficients = np.linalg.lstsq(a[best], points[best, 2], rcond=None)[0]
        best = np.abs(a @ coefficients - points[:, 2]) < .004
    residual = a @ coefficients - points[:, 2]
    normal = np.r_[-coefficients[:2], 1.0]
    normal /= np.linalg.norm(normal)
    return coefficients, {
        'equation': 'z_base = a*x_base + b*y_base + c; metres',
        'abc': coefficients.tolist(), 'normal_base': normal.tolist(),
        'tilt_from_base_xy_deg': float(np.degrees(np.arccos(normal[2]))),
        'inlier_fraction': float(best.mean()),
        'inlier_rms_m': float(np.sqrt(np.mean(residual[best]**2))),
        'ransac_seed': seed, 'inlier_threshold_z_m': .004,
    }

def depth_capture_timing(frame):
    """Validate the SDK global depth-time estimate against the host clocks.

    Global time is a software fit, not hardware exposure synchronization. The
    receipt-anchored runtime sample_time cannot establish capture freshness.
    """
    metadata = frame.timestamp_metadata
    if not isinstance(metadata, dict) or metadata.get('selected_source') != 'depth_device':
        raise PickRejected('相机配对需要原始深度device/global时间及主机时钟参考')
    try:
        depth = metadata['depth']; host = metadata['host_clock_reference']
        device = float(depth['device_time_s']); global_time = float(depth['global_time_s'])
        system = float(depth['system_time_s']); source = float(frame.header.source_time)
        wall = float(host['unix_s']); monotonic = float(host['monotonic_s'])
        span = float(host['read_span_s']); received = float(frame.header.received_time)
    except (KeyError, TypeError, ValueError) as error:
        raise PickRejected('缺少深度global/system/device时间或主机时钟参考') from error
    if (not np.isfinite([device, global_time, system, source, wall, monotonic, span, received]).all()
            or min(device, global_time, system, wall, monotonic, received) <= 0
            or span < 0 or abs(device-source) > 1e-6):
        raise PickRejected('深度时间或主机时钟读取误差无效')
    offset = monotonic-wall
    captured = global_time+offset
    system_monotonic = system+offset
    evidence={'depth_device_s':device,'depth_global_s':global_time,'depth_system_s':system,
        'host_received_monotonic_s':received,'estimated_capture_monotonic_s':captured,
        'estimated_system_monotonic_s':system_monotonic,'reference_span_s':span,
        'estimated_delivery_delay_s':received-captured,'global_minus_system_s':global_time-system}
    if global_time > system:
        raise PickRejected('深度SDK全局时间估计超前或传输延迟超过250 ms',details={**evidence,'reason':'global_after_system'})
    if span > .005:
        raise _ClockReferenceSchedulingDelay('相机主机时钟参考读取期间发生调度延迟',details={**evidence,'reason':'host_reference_span'})
    if (captured > received+.001 or system_monotonic > received+.001
            or not -.001 <= received-captured <= .25):
        raise PickRejected('深度SDK全局时间估计超前或传输延迟超过250 ms',details={**evidence,'reason':'capture_or_system_outside_host_window'})
    return {'depth_sample_monotonic_estimate': captured,
            'depth_global_time_s': global_time, 'depth_device_time_s': device,
            'depth_system_time_s': system, 'host_wall_to_monotonic_offset_s': offset,
            'host_clock_read_span_s': span,
            'depth_delivery_delay_estimate_s': received-captured,
            'hardware_synchronized': False,
            'time_estimate_limitation': 'SDK global time is a software clock fit; '
                                       'stationary-window checks do not prove hardware exposure timing.'}

"""Selected unchanged saved-frame helpers; origin documented in migration report."""
import json
from pathlib import Path
import cv2
import numpy as np
from mujoco_conveyor_sorting.runtime.types import RGBDFrame, CameraIntrinsics, SampleHeader
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurfaceNotReady
from mujoco_conveyor_sorting.calibration.geometry import rigid_transform


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
    distortion = np.asarray(frame.color_intrinsics.distortion)
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
        if exceptional.any():
            # Preserve OpenCV's NaN/Inf behavior for exceptional input or
            # overflow without slowing down the ordinary full-image case.
            uv[exceptional] = cv2.projectPoints(
                points_color[exceptional], np.zeros(3), np.zeros(3),
                intrinsics, distortion)[0].reshape(-1, 2)
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


def load_frame(directory, selected):
    names = sorted(p.name.removesuffix('_depth_m.npy')
                   for p in directory.glob('frame_*_depth_m.npy'))
    if not names:
        raise ValueError(f'没有保存的 frame_*_depth_m.npy：{directory}')
    name = names[-1] if selected is None else Path(selected).name.removesuffix('.json').removesuffix('_depth_m.npy')
    if name.isdigit():
        name = f'frame_{int(name):06d}'
    if name not in names:
        raise ValueError(f'没有保存的帧：{name}')
    metadata = json.loads((directory/f'{name}.json').read_text(encoding='utf-8'))
    transform = metadata.get('depth_to_base_cv', metadata.get('camera_to_world_cv'))
    if transform is None or metadata.get('depth_to_color_cv') is None:
        raise ValueError('保存帧必须包含深度到基座外参和工厂深度到RGB变换')
    rgb = cv2.imread(str(directory/f'{name}_rgb.png'))
    if rgb is None:
        raise ValueError(f'不能读取 {name}_rgb.png')
    stereo = metadata.get('stereo_capture') or {}
    stereo_options = {}
    depth = np.load(directory/f'{name}_depth_m.npy', allow_pickle=False)
    timestamp_metadata = metadata.get('timestamp_metadata')
    if stereo.get('available'):
        # Sidecar names are determined by this saved capture, never by the
        # current camera or the last successful reconstruction.
        try:
            for side in ('left', 'right'):
                path = directory/f'{name}_{side}_ir.npy'
                if path.is_file():
                    infrared = np.load(path, allow_pickle=False)
                    if infrared.shape != depth.shape or infrared.dtype != np.uint8:
                        raise ValueError(f'{side} IR sidecar must match native Y8 depth grid')
                    stereo_options[side+'_ir'] = infrared
                intrinsic = stereo.get(side+'_ir_intrinsics')
                if intrinsic is not None:
                    stereo_options[side+'_ir_intrinsics'] = CameraIntrinsics(**intrinsic)
            stereo_options['stereo_baseline_m'] = stereo.get('baseline_m')
            stereo_options['stereo_calibration'] = stereo.get('calibration')
        except (OSError, ValueError, TypeError, EOFError) as error:
            # Optional reconstruction attachments cannot invalidate the
            # original RGB-D recording. Preserve the reason for diagnostics.
            stereo_options = {}
            timestamp_metadata = dict(timestamp_metadata or {},
                stereo_replay_error=f'{type(error).__name__}: {error}')
    frame = RGBDFrame(
        header=SampleHeader(**metadata['header']),
        depth_m=depth,
        depth_intrinsics=CameraIntrinsics(**metadata['depth_intrinsics']),
        rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
        color_intrinsics=CameraIntrinsics(**metadata['color_intrinsics']),
        camera_to_world_cv=rigid_transform(transform),
        serial_number=metadata.get('serial_number', ''),
        calibration_id=metadata.get('calibration_id', ''),
        source='recorded_pick_test', rgb_aligned_to_depth=False,
        timestamp_metadata=timestamp_metadata,
        **stereo_options,
    )
    return name, frame, rigid_transform(metadata['depth_to_color_cv'])

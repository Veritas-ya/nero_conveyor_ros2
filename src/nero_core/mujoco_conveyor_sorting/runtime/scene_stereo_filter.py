"""Recheck native scene samples against same-capture stereo image evidence.

Small isolated components are only an inexpensive scope for investigation.
Deletion additionally needs bidirectional learned stereo to put the visible
background behind the native sample, and original IR Census matching to prefer
that background unambiguously over the native disparity. Neither network
agreement nor missing depth alone establishes empty space. No surface is added
and all masks refer to the original input point order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import time

import os

import numpy as np

from .artifact_writer import background_artifact
from scipy.ndimage import binary_dilation

from .grasp_depth import _camera_geometry, _json, target_stereo_crop
from .scene_depth_filter import _component_candidates, _native_pixels


@dataclass(frozen=True)
class SceneStereoFilterConfig:
    enabled: bool = True
    connection_radius_m: float = .005
    max_component_points: int = 160
    max_component_extent_m: float = .04
    left_right_tolerance_px: float = 1.
    minimum_background_gap_m: float = .03
    minimum_disparity_separation_px: float = 2.
    maximum_disparity_px: int = 192
    maximum_roi_pairs: int = 2
    preferred_shape: tuple[int, int] = (384, 640)
    census_radius: int = 2
    census_ambiguity_radius_px: float = 1.

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        for name in ('connection_radius_m', 'max_component_extent_m',
                     'left_right_tolerance_px', 'minimum_background_gap_m',
                     'minimum_disparity_separation_px', 'census_ambiguity_radius_px'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self, name, float(value))
        for name in ('max_component_points', 'maximum_disparity_px',
                     'maximum_roi_pairs', 'census_radius'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self, name, int(value))
        shape = tuple(self.preferred_shape)
        if (len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral)
                                   or v < 32 for v in shape)):
            raise ValueError('preferred_shape must contain two integer pixel sizes >=32')
        object.__setattr__(self, 'preferred_shape', tuple(map(int, shape)))

    def as_dict(self):
        return asdict(self)


@dataclass
class SceneStereoFilterResult:
    points: np.ndarray
    removed_mask: np.ndarray
    statistics: dict
    components: list[dict]

    @property
    def retained_points(self):
        return self.points[~self.removed_mask]


def _sample_horizontal(image, x, y):
    """Linear horizontal sampling, requiring only samples with nonzero weight."""
    image = np.asarray(image)
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y))
    usable = np.isfinite(x) & np.isfinite(y)
    usable &= ((x >= 0) & (x <= image.shape[1]-1)
               & (y >= 0) & (y < image.shape[0]) & (y == np.floor(y)))
    safe_x = np.where(usable, x, 0.)
    safe_y = np.where(usable, y, 0).astype(np.int64)
    lo = np.floor(safe_x).astype(np.int64)
    hi = np.ceil(safe_x).astype(np.int64)
    weight = safe_x-lo
    a = image[safe_y, lo].astype(float)
    b = image[safe_y, hi].astype(float)
    usable &= np.isfinite(a) & np.isfinite(b)
    value = (1.-weight)*np.where(np.isfinite(a), a, 0.)+weight*np.where(np.isfinite(b), b, 0.)
    return np.where(usable, value, np.nan), usable


def stereo_free_space_evidence(native_depth_m, left_disparity_px, right_disparity_px,
                               uv, *, focal_baseline_m_px, config=None):
    """Return geometric eligibility, not a final obstacle-removal decision.

    Both disparity maps use original, unmirrored image coordinates. ``uv`` is
    N x 2 integer (u,v), native depths are the N actual corresponding samples.
    One pixel of LR eligibility tolerance is also used as a conservative
    disparity allowance when estimating the nearest possible background. This
    allowance is a configured safeguard, not a measured depth-error bound.
    """
    config = SceneStereoFilterConfig() if config is None else config
    left, right = np.asarray(left_disparity_px), np.asarray(right_disparity_px)
    uv = np.asarray(uv)
    native = np.asarray(native_depth_m, dtype=float)
    if (left.ndim != 2 or right.shape != left.shape or uv.ndim != 2 or uv.shape[1] != 2
            or uv.dtype.kind not in 'iu' or native.shape != (len(uv),)):
        raise ValueError('stereo evidence requires aligned depth samples and original disparity grids')
    fb = float(focal_baseline_m_px)
    if not np.isfinite(fb) or fb <= 0:
        raise ValueError('focal_baseline_m_px must be finite and positive')
    dl, valid_left = _sample_horizontal(left, uv[:, 0], uv[:, 1])
    valid_left &= (dl > 0) & (dl < config.maximum_disparity_px)
    xr = uv[:, 0]-dl
    dr, valid_right = _sample_horizontal(right, xr, uv[:, 1])
    valid_right &= (dr > 0) & (dr < config.maximum_disparity_px)
    error = np.abs(dl-dr)
    consistent = valid_left & valid_right & (error <= config.left_right_tolerance_px)
    valid_native = np.isfinite(native) & (native > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        estimated_depth = fb/dl
        conservative_depth = fb/(dl+config.left_right_tolerance_px)
        native_disparity = fb/native
    depth_gap = conservative_depth-native
    disparity_gap = native_disparity-dl
    eligible = (consistent & valid_native & (depth_gap >= config.minimum_background_gap_m)
                & (disparity_gap >= config.minimum_disparity_separation_px))
    return {'valid_left': valid_left, 'valid_right': valid_right,
            'right_coordinate_px': xr, 'left_disparity_px': dl,
            'right_disparity_px': dr, 'native_disparity_px': native_disparity,
            'left_right_error_px': error, 'left_right_consistent': consistent,
            'new_depth_m': estimated_depth, 'conservative_background_depth_m': conservative_depth,
            'estimated_depth_gap_m': estimated_depth-native, 'depth_gap_m': depth_gap,
            'disparity_gap_px': disparity_gap, 'free_space_evidence': eligible,
            'stereo_supported_background_contradiction': eligible}


def _patches(image, uv, disparities, radius):
    """Sample native IR patches at possibly fractional horizontal shifts."""
    offset_y, offset_x = np.mgrid[-radius:radius+1, -radius:radius+1]
    offsets_x, offsets_y = offset_x.ravel(), offset_y.ravel()
    x = uv[:, 0, None, None]-disparities[:, :, None]+offsets_x[None, None, :]
    y = uv[:, 1, None, None]+offsets_y[None, None, :]
    return _sample_horizontal(image, x, y)


def _census_cost(left_patches, right_patches):
    center = left_patches.shape[-1]//2
    keep = np.arange(left_patches.shape[-1]) != center
    a = left_patches[..., keep] >= left_patches[..., center, None]
    b = right_patches[..., keep] >= right_patches[..., center, None]
    return np.count_nonzero(a != b, axis=-1)


def census_ir_free_space_support(left_ir, right_ir, uv, native_disparity_px,
                                 estimated_disparity_px, *, config=None):
    """Require original IR to prefer a unique background correspondence.

    Census compares relative patch intensity to its centre (24 binary tests for
    default 5x5 patches). Hamming costs are measured in differing bits. A native
    match that is equally good protects the sample. An equally good alternative
    anywhere in the model's integer search range, outside the configured +/-1
    px neighbourhood of the estimated match, marks ambiguity and protects it.
    No absolute brightness/noise cutoff is fitted to the pictured component.
    """
    config = SceneStereoFilterConfig() if config is None else config
    left, right = np.asarray(left_ir), np.asarray(right_ir)
    uv = np.asarray(uv)
    native, estimated = np.asarray(native_disparity_px, float), np.asarray(estimated_disparity_px, float)
    if (left.ndim != 2 or right.shape != left.shape or uv.shape != (len(native), 2)
            or estimated.shape != native.shape or native.ndim != 1):
        raise ValueError('Census requires original IR grids and aligned point disparities')
    lpatch, lvalid = _patches(left, uv, np.zeros((len(uv), 1)), config.census_radius)
    # Native depth is quantized and a true narrow foreground can lie at the
    # neighbouring integer correspondence. Any of these native matches protects
    # it; a bad interpolated patch alone is not evidence against the obstacle.
    matches = np.column_stack((native, np.floor(native), np.ceil(native), estimated))
    rpatch, rvalid = _patches(right, uv, matches, config.census_radius)
    costs = _census_cost(lpatch, rpatch).astype(float)
    match_valid = lvalid.all(axis=(1, 2))[:, None] & rvalid.all(axis=2)
    costs[~match_valid] = np.inf
    # A flat/one-sided comparison pattern is not enough image evidence for
    # deletion, even if a learned model outputs two agreeing disparities.
    center = lpatch.shape[-1]//2
    textured_left = ((lpatch < lpatch[..., center, None]).any(axis=(1, 2))
                     & (lpatch > lpatch[..., center, None]).any(axis=(1, 2)))
    textured_right = ((rpatch[:, 3] < rpatch[:, 3, center, None]).any(axis=1)
                      & (rpatch[:, 3] > rpatch[:, 3, center, None]).any(axis=1))
    # Include both search boundaries as ambiguity evidence. A zero/maximum
    # disparity need not be a valid network depth to protect a competing raw
    # image correspondence.
    search = np.arange(config.maximum_disparity_px+1, dtype=float)
    all_disparities = np.broadcast_to(search, (len(uv), len(search)))
    alternatives, alternatives_valid = _patches(right, uv, all_disparities, config.census_radius)
    alternative_cost = _census_cost(lpatch, alternatives).astype(float)
    outside = np.abs(all_disparities-estimated[:, None]) > config.census_ambiguity_radius_px
    alternative_cost[~(outside & alternatives_valid.all(axis=2))] = np.inf
    best_alternative = np.min(alternative_cost, axis=1)
    native_cost = costs[:, :3].min(axis=1)
    background_cost = costs[:, 3]
    valid_patches = match_valid[:, :3].any(axis=1) & match_valid[:, 3]
    better_native = background_cost < native_cost
    unique = background_cost < best_alternative
    support = (valid_patches & textured_left & textured_right & better_native & unique)
    return {'native_census_cost_bits': native_cost, 'background_census_cost_bits': background_cost,
            'best_alternative_census_cost_bits': best_alternative,
            'valid_patches': valid_patches,
            'textured': textured_left & textured_right,
            'background_better_than_native': better_native,
            'background_unique_outside_neighborhood': unique,
            'original_ir_support': support}


def _result_disparity(result, crop, shape, capture):
    if result.provenance.get('capture_metadata') != capture:
        raise ValueError('fast_result_capture_or_reference_mismatch')
    if tuple(result.crop_xyxy) != tuple(crop):
        raise ValueError('fast_result_crop_mismatch')
    x0, y0, x1, y1 = crop
    values = np.asarray(result.disparity_px)
    if values.shape != (y1-y0, x1-x0):
        raise ValueError('fast_result_disparity_grid_mismatch')
    output = np.full(shape, np.nan, dtype=np.float32)
    output[y0:y1, x0:x1] = values
    return output


@background_artifact
def _save(result, directory, frame, *, stem_name='scene_stereo'):
    if directory is None:
        return
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        epoch = frame.header.clock_epoch if frame is not None else 'unknown'
        sequence = frame.header.sequence if frame is not None else 0
        stem = directory / f'{stem_name}_e{epoch}_s{sequence:06d}'
        ordinal = 1
        while stem.with_suffix('.json').exists():
            stem = directory / f'{stem_name}_e{epoch}_s{sequence:06d}_{ordinal:02d}'
            ordinal += 1
        np.save(str(stem)+'_before.npy', result.points)
        if not result.removed_mask.any():
            try:
                os.link(str(stem)+'_before.npy', str(stem)+'_retained.npy')
            except OSError:
                np.save(str(stem)+'_retained.npy', result.retained_points)
        else:
            np.save(str(stem)+'_retained.npy', result.retained_points)
        np.save(str(stem)+'_removed.npy', result.points[result.removed_mask])
        np.save(str(stem)+'_removed_mask.npy', result.removed_mask)
        result.statistics['artifact_prefix'] = str(stem)
        clouds = {'before': result.points, 'retained': result.retained_points,
                  'removed': result.points[result.removed_mask]}
        result.statistics['artifacts'] = {
            name: {'file': stem.name+'_'+name+'.npy', 'shape': list(values.shape),
                   'stored_dtype': str(values.dtype), 'sha256_dtype': 'float64_c_order',
                   'sha256': hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()}
            for name, values in clouds.items()}
        result.statistics['artifacts']['removed_mask'] = {
            'file': stem.name+'_removed_mask.npy', 'shape': list(result.removed_mask.shape),
            'stored_dtype': 'bool',
            'sha256': hashlib.sha256(np.ascontiguousarray(result.removed_mask).tobytes()).hexdigest()}
        stem.with_suffix('.json').write_text(json.dumps(_json({
            'statistics': result.statistics, 'components': result.components}),
            ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except Exception as error:
        result.statistics['artifact_error'] = f'{type(error).__name__}: {error}'
    return {key: result.statistics[key] for key in ('artifact_prefix', 'artifact_error')
            if key in result.statistics}


def filter_scene_stereo(points, *, frame=None, protected_mask=None, backend=None,
                        config=None, output_directory=None, emit=None):
    """Recheck at most two small isolated scene ROIs; preserve on uncertainty.

    This function neither connects devices nor modifies frame/time, target
    geometry, model padding or collision parameters. It only removes original
    scene indices supported by all geometric and original-image comparisons.
    """
    started = time.perf_counter()
    config = SceneStereoFilterConfig() if config is None else config
    if not isinstance(config, SceneStereoFilterConfig):
        raise ValueError('scene stereo filter requires SceneStereoFilterConfig')
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf'
            or not np.isfinite(points).all()):
        raise ValueError('scene points must be a finite numeric N by 3 array')
    points = points.copy()
    result = SceneStereoFilterResult(points, np.zeros(len(points), dtype=bool), {
        'algorithm': 'same_capture_bidirectional_stereo_and_original_ir_recheck',
        'evidence_source': 'learned_stereo_evidence_with_original_ir_relative_matching',
        'input_points': len(points), 'removed_points': 0, 'retained_points': len(points),
        'reason': 'evaluated', 'parameters': config.as_dict(), 'roi_pairs': 0,
        'backend_calls': 0, 'removed_components': 0, 'native_depth_modified': False,
        'timestamp_modified': False, 'added_surface_points': 0,
        'uncertainty_policy': 'retain_unverified_samples',
        'census_descriptor_bits': (2*config.census_radius+1)**2-1,
        'depth_allowance_is_measured_accuracy': False,
        'temporal_occupancy_fusion': False}, [])
    stats = result.statistics

    def finish(reason=None):
        if reason is not None:
            stats['reason'] = reason
        stats['removed_points'] = int(result.removed_mask.sum())
        stats['retained_points'] = len(points)-stats['removed_points']
        stats['removed_components'] = sum(c['removed_points'] > 0 for c in result.components)
        stats['processing_s'] = time.perf_counter()-started
        _save(result, output_directory, frame)
        if emit is not None:
            emit('scene_stereo_filter', {**stats, 'components': result.components})
        return result

    if not config.enabled:
        return finish('disabled')
    if not len(points):
        return finish('empty_cloud')
    if frame is None:
        return finish('missing_native_depth_frame')
    stats['capture_header'] = asdict(frame.header)
    stats['serial_number'] = frame.serial_number
    stats['calibration_id'] = frame.calibration_id
    if backend is None:
        return finish('stereo_backend_unavailable')
    try:
        matrix, baseline = _camera_geometry(frame)
        transform = np.asarray(frame.camera_to_world_cv, dtype=float)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or np.any(np.abs(frame.depth_intrinsics.distortion) > 1e-10)):
            return finish('invalid_native_camera_to_base')
        np.linalg.inv(transform)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        return finish(str(error) or 'invalid_native_stereo_frame')
    protected = np.asarray(protected_mask)
    if protected.shape != frame.depth_m.shape or protected.dtype.kind != 'b':
        return finish('missing_or_invalid_protected_target_mask')
    stats['protected_target_mask_sha256'] = hashlib.sha256(
        np.ascontiguousarray(protected, dtype=bool).tobytes()).hexdigest()
    stats['camera_to_base_sha256'] = hashlib.sha256(
        np.ascontiguousarray(transform, dtype=np.float64).tobytes()).hexdigest()
    # As in the existing depth-edge filter, a one-pixel target boundary is
    # protected as well: segmentation-border pixels are not free-space proof.
    protected = binary_dilation(protected, structure=np.ones((3, 3), dtype=bool))
    candidates, summary = _component_candidates(points.astype(float, copy=False), config)
    stats.update(summary)
    # Larger reviewable components first, deterministic tie-breaking. Budget
    # exhaustion retains all remaining samples; it is never a motion rejection.
    candidates.sort(key=lambda item: (-len(item[0]), int(item[0][0])))
    pair_cache = {}
    failed_crops = set()
    height, width = frame.depth_m.shape
    for selected, extent, linked in candidates:
        component = {'point_count': len(selected), 'extent_m': extent,
                     'original_point_indices': selected.tolist(), 'removed_points': 0,
                     'reason': 'evaluated', 'removed': False,
                     'bbox_base_m': [points[selected].min(axis=0).tolist(), points[selected].max(axis=0).tolist()]}
        result.components.append(component)
        if linked:
            component['reason'] = 'connected_to_other_scene_points'
            continue
        uv, reliable = _native_pixels(points[selected], frame.depth_m,
                                       frame.depth_intrinsics, transform)
        if not reliable.all() or len(np.unique(uv, axis=0)) != len(uv):
            component['reason'] = 'unreliable_native_pixel_mapping'
            continue
        component['native_uv'] = uv.tolist()
        if protected[uv[:, 1], uv[:, 0]].any():
            component['reason'] = 'protected_target_or_boundary_pixels'
            continue
        mask = np.zeros((height, width), dtype=bool)
        mask[uv[:, 1], uv[:, 0]] = True
        crop = target_stereo_crop(mask, maximum_disparity=config.maximum_disparity_px,
                                  preferred_shape=config.preferred_shape)
        component['crop_xyxy'] = list(crop)
        if crop in failed_crops:
            component['reason'] = 'same_capture_roi_failed'
            continue
        if crop not in pair_cache:
            if stats['roi_pairs'] >= config.maximum_roi_pairs:
                component['reason'] = 'roi_compute_budget_exhausted'
                continue
            stats['roi_pairs'] += 1
            try:
                capture = {'capture_header': asdict(frame.header),
                           'timestamp_metadata': frame.timestamp_metadata,
                           'serial_number': frame.serial_number,
                           'calibration_id': frame.calibration_id,
                           'reference': 'left_native_ir'}
                stats['backend_calls'] += 1
                left_result = backend.refine(frame.left_ir, frame.right_ir, matrix, baseline,
                                             crop, capture_metadata=copy.deepcopy(capture))
                dl = _result_disparity(left_result, crop, (height, width), capture)
                mirrored_matrix = matrix.copy()
                mirrored_matrix[0, 2] = width-1-matrix[0, 2]
                x0, y0, x1, y1 = crop
                mirrored_crop = (width-x1, y0, width-x0, y1)
                right_capture = {**capture, 'reference': 'right_ir_horizontal_mirror'}
                stats['backend_calls'] += 1
                right_result = backend.refine(np.ascontiguousarray(frame.right_ir[:, ::-1]),
                                              np.ascontiguousarray(frame.left_ir[:, ::-1]),
                                              mirrored_matrix, baseline, mirrored_crop,
                                              capture_metadata=copy.deepcopy(right_capture))
                dr_mirrored = _result_disparity(right_result, mirrored_crop,
                                               (height, width), right_capture)
                pair_cache[crop] = (dl, dr_mirrored[:, ::-1], {
                    'left_worker_artifact_prefix': left_result.artifact_prefix,
                    'right_worker_artifact_prefix': right_result.artifact_prefix,
                    'left_worker_timings_s': left_result.timings_s,
                    'right_worker_timings_s': right_result.timings_s,
                    'right_mirrored_crop_xyxy': list(mirrored_crop),
                    'right_mirrored_cx_px': float(mirrored_matrix[0, 2])})
            except Exception as error:
                failed_crops.add(crop)
                component.update(reason='stereo_recheck_failed',
                                 error=f'{type(error).__name__}: {error}')
                continue
        try:
            dl, dr, pair_metadata = pair_cache[crop]
            component.update(pair_metadata)
            native_depth = frame.depth_m[uv[:, 1], uv[:, 0]].astype(float)
            evidence = stereo_free_space_evidence(native_depth, dl, dr, uv,
                focal_baseline_m_px=matrix[0, 0]*baseline, config=config)
            # Image matching can only further protect a geometrically eligible
            # sample. Skip its full disparity search for samples that already
            # must be retained; record the omission rather than claim support.
            eligible = np.flatnonzero(evidence['stereo_supported_background_contradiction'])
            ir = {key: np.full(len(selected), np.nan) for key in (
                'native_census_cost_bits', 'background_census_cost_bits',
                'best_alternative_census_cost_bits')}
            ir.update({key: np.zeros(len(selected), dtype=bool) for key in (
                'valid_patches', 'textured', 'background_better_than_native',
                'background_unique_outside_neighborhood', 'original_ir_support',
                'census_evaluated')})
            if len(eligible):
                measured_ir = census_ir_free_space_support(frame.left_ir, frame.right_ir, uv[eligible],
                    evidence['native_disparity_px'][eligible], evidence['left_disparity_px'][eligible],
                    config=config)
                for key, values in measured_ir.items():
                    ir[key][eligible] = values
                ir['census_evaluated'][eligible] = True
            remove = evidence['stereo_supported_background_contradiction'] & ir['original_ir_support']
            result.removed_mask[selected[remove]] = True
            component.update(reason=('stereo_supported_background_contradiction' if remove.any()
                                     else 'no_joint_stereo_and_original_ir_evidence'),
                             removed=bool(remove.any()), removed_points=int(remove.sum()),
                             left_right_consistent_points=int(evidence['left_right_consistent'].sum()),
                             background_geometry_eligible_points=int(evidence['free_space_evidence'].sum()),
                             census_evaluated_points=len(eligible),
                             original_ir_support_points=int(ir['original_ir_support'].sum()),
                             per_point_evidence=_json({**evidence, **ir, 'native_depth_m': native_depth,
                                                       'removed': remove}))
        except Exception as error:
            component.update(reason='stereo_evidence_failed', error=f'{type(error).__name__}: {error}')
    return finish('evaluated' if result.components else 'no_small_isolated_components')

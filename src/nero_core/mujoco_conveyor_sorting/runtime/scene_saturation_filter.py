"""Reject clipped, ambiguous native stereo samples beside robot image edges.

This marks a native depth measurement invalid. It does not infer free space,
replace depth, enlarge the robot, remove a component solely for its size, or
claim the surrounding scene was observed empty. The clipping value 255 is the
native Y8 format's maximum, not a fitted brightness threshold.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import time

import os

import numpy as np

from .artifact_writer import background_artifact
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from .scene_workspace import spatial_tree

from .grasp_depth import _camera_geometry, _json
from .scene_depth_filter import _component_candidates, _native_pixels
from .scene_stereo_filter import _patches, _census_cost


@dataclass(frozen=True)
class SceneSaturationFilterConfig:
    enabled: bool = True
    connection_radius_m: float = .005
    max_component_points: int = 160
    max_component_extent_m: float = .04
    maximum_disparity_px: int = 192
    census_radius: int = 2
    census_ambiguity_radius_px: float = 1.

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be boolean')
        for name in ('connection_radius_m', 'max_component_extent_m', 'census_ambiguity_radius_px'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        for name in ('max_component_points', 'maximum_disparity_px', 'census_radius'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    def as_dict(self):
        return asdict(self)


@dataclass
class SceneSaturationFilterResult:
    points: np.ndarray
    removed_mask: np.ndarray
    statistics: dict
    components: list

    @property
    def retained_points(self):
        return self.points[~self.removed_mask]


def saturated_ambiguous_native_samples(left_ir, right_ir, uv, native_disparity_px, *, config=None):
    """Per-sample clipping and matching ambiguity, without scene assumptions."""
    config = SceneSaturationFilterConfig() if config is None else config
    left, right = np.asarray(left_ir), np.asarray(right_ir)
    uv = np.asarray(uv)
    disparity = np.asarray(native_disparity_px, dtype=float)
    if (left.ndim != 2 or right.shape != left.shape or left.dtype != np.uint8
            or right.dtype != np.uint8 or uv.shape != (len(disparity), 2)
            or uv.dtype.kind not in 'iu' or disparity.ndim != 1):
        raise ValueError('native saturation requires original Y8 grids and aligned integer pixels')
    n = len(uv)
    in_image = ((uv[:, 0] >= 0) & (uv[:, 0] < left.shape[1])
                & (uv[:, 1] >= 0) & (uv[:, 1] < left.shape[0]))
    valid = in_image & np.isfinite(disparity) & (disparity > 0) & (disparity < config.maximum_disparity_px)
    xr = uv[:, 0]-disparity
    valid &= np.isfinite(xr) & (xr >= 0) & (xr <= left.shape[1]-1)
    safe_uv = np.where(in_image[:, None], uv, 0)
    safe_xr = np.where(valid, xr, 0.)
    floor_x, ceil_x = np.floor(safe_xr).astype(int), np.ceil(safe_xr).astype(int)
    saturated = (valid & (left[safe_uv[:, 1], safe_uv[:, 0]] == 255)
                 & (right[safe_uv[:, 1], floor_x] == 255)
                 & (right[safe_uv[:, 1], ceil_x] == 255))
    answer = {'left_saturated_y8': in_image & (left[safe_uv[:, 1], safe_uv[:, 0]] == 255),
              'both_native_right_samples_saturated_y8': saturated,
              'native_disparity_px': disparity, 'native_right_coordinate_px': xr,
              'native_census_cost_bits': np.full(n, np.nan),
              'best_competing_census_cost_bits': np.full(n, np.nan),
              'census_evaluated': np.zeros(n, bool), 'no_local_texture': np.zeros(n, bool),
              'native_matching_ambiguous': np.zeros(n, bool),
              'invalid_saturated_native_measurement': np.zeros(n, bool)}
    indices = np.flatnonzero(saturated)
    if not len(indices):
        return answer
    pixels, d = uv[indices], disparity[indices]
    lp, lv = _patches(left, pixels, np.zeros((len(indices), 1)), config.census_radius)
    native_matches = np.column_stack((d, np.floor(d), np.ceil(d)))
    rp, rv = _patches(right, pixels, native_matches, config.census_radius)
    patch_valid = lv.all(axis=(1, 2)) & rv.all(axis=(1, 2))
    native_cost = _census_cost(lp, rp).astype(float).min(axis=1)
    search = np.broadcast_to(np.arange(config.maximum_disparity_px+1, dtype=float),
                             (len(indices), config.maximum_disparity_px+1))
    alternatives, av = _patches(right, pixels, search, config.census_radius)
    costs = _census_cost(lp, alternatives).astype(float)
    competing = np.abs(search-d[:, None]) > config.census_ambiguity_radius_px
    costs[~(competing & av.all(axis=2))] = np.inf
    best = costs.min(axis=1)
    # Fully constant patches carry no local intensity pattern at all. A
    # saturated centre alone does not imply a constant or ambiguous patch.
    flat = ((np.ptp(lp[:, 0], axis=1) == 0) | (np.ptp(rp, axis=2) == 0).all(axis=1))
    ambiguous = best <= native_cost
    for key, values in [('native_census_cost_bits', native_cost),
                        ('best_competing_census_cost_bits', best),
                        ('census_evaluated', patch_valid), ('no_local_texture', flat),
                        ('native_matching_ambiguous', ambiguous)]:
        answer[key][indices] = values
    answer['invalid_saturated_native_measurement'][indices] = patch_valid & (flat | ambiguous)
    return answer


@background_artifact
def _save(result, output_directory, frame):
    if output_directory is None:
        return
    try:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        epoch, sequence = (frame.header.clock_epoch, frame.header.sequence) if frame is not None else ('unknown', 0)
        stem = directory/f'scene_saturation_e{epoch}_s{sequence:06d}'
        ordinal = 1
        while stem.with_suffix('.json').exists():
            stem = directory/f'scene_saturation_e{epoch}_s{sequence:06d}_{ordinal:02d}'
            ordinal += 1
        clouds = {'before': result.points, 'retained': result.retained_points,
                  'removed': result.points[result.removed_mask]}
        artifacts = {}
        for name, values in clouds.items():
            path = Path(str(stem)+'_'+name+'.npy')
            if name == 'retained' and not result.removed_mask.any():
                try:
                    os.link(str(stem)+'_before.npy', path)
                except OSError:
                    np.save(path, values)
            else:
                np.save(path, values)
            artifacts[name] = {'file': path.name, 'shape': list(values.shape),
                'stored_dtype': str(values.dtype), 'sha256_dtype': 'float64_c_order',
                'sha256': hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()}
        path = Path(str(stem)+'_removed_mask.npy')
        np.save(path, result.removed_mask)
        artifacts['removed_mask'] = {'file': path.name, 'shape': list(result.removed_mask.shape),
            'stored_dtype': 'bool', 'sha256': hashlib.sha256(result.removed_mask.tobytes()).hexdigest()}
        result.statistics.update(artifact_prefix=str(stem), artifacts=artifacts)
        stem.with_suffix('.json').write_text(json.dumps(_json({'statistics': result.statistics,
            'components': result.components}), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    except Exception as error:
        result.statistics['artifact_error'] = f'{type(error).__name__}: {error}'
    return {key: result.statistics[key] for key in ('artifact_prefix', 'artifact_error')
            if key in result.statistics}


def filter_scene_saturation(points, *, frame=None, protected_mask=None, self_filter_result=None,
                             config=None, output_directory=None):
    """Drop only clipped ambiguous native samples in isolated robot-edge groups."""
    started = time.perf_counter()
    config = SceneSaturationFilterConfig() if config is None else config
    if not isinstance(config, SceneSaturationFilterConfig):
        raise ValueError('scene saturation filter requires SceneSaturationFilterConfig')
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf'
            or not np.isfinite(points).all()):
        raise ValueError('scene points must be finite numeric N by 3')
    points = points.copy()
    result = SceneSaturationFilterResult(points, np.zeros(len(points), bool), {
        'algorithm': 'native_y8_saturation_and_census_ambiguity_at_robot_edge',
        'parameters': config.as_dict(), 'reason': 'evaluated', 'input_points': len(points),
        'removed_points': 0, 'retained_points': len(points), 'removed_components': 0,
        'classification': 'invalid_native_measurement_only', 'free_space_inferred': False,
        'native_depth_modified': False, 'added_surface_points': 0, 'timestamp_modified': False,
        'robot_padding_modified': False, 'clipping_value_y8': 255,
        'census_descriptor_bits': (2*config.census_radius+1)**2-1}, [])
    stats = result.statistics
    def finish(reason=None):
        if reason is not None:
            stats['reason'] = reason
        stats.update(removed_points=int(result.removed_mask.sum()),
                     retained_points=int((~result.removed_mask).sum()),
                     removed_components=sum(c['removed_points'] > 0 for c in result.components),
                     processing_s=time.perf_counter()-started)
        _save(result, output_directory, frame)
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
    try:
        matrix, baseline = _camera_geometry(frame)
        transform = np.asarray(frame.camera_to_world_cv, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return finish('invalid_native_camera_to_base')
        protected = np.asarray(protected_mask)
        if protected.shape != frame.depth_m.shape or protected.dtype.kind != 'b':
            return finish('missing_or_invalid_protected_target_mask')
        stats['protected_target_mask_sha256'] = hashlib.sha256(
            np.ascontiguousarray(protected, dtype=bool).tobytes()).hexdigest()
        stats['camera_to_base_sha256'] = hashlib.sha256(
            np.ascontiguousarray(transform, dtype=np.float64).tobytes()).hexdigest()
        if self_filter_result is None:
            return finish('missing_self_filter_result')
        original = np.asarray(self_filter_result.points)
        self_removed = np.asarray(self_filter_result.removed_mask)
        if (original.ndim != 2 or original.shape[1] != 3 or self_removed.shape != (len(original),)
                or self_removed.dtype.kind != 'b'):
            return finish('invalid_self_filter_result')
        robot_uv, robot_reliable = _native_pixels(original[self_removed], frame.depth_m,
                                                  frame.depth_intrinsics, transform)
        if not robot_reliable.any():
            return finish('no_reliable_robot_depth_pixels')
        robot = np.zeros(frame.depth_m.shape, bool)
        robot_uv = robot_uv[robot_reliable]
        robot[robot_uv[:, 1], robot_uv[:, 0]] = True
        adjacent = binary_dilation(robot, structure=np.ones((3, 3), bool)) & ~robot
        uv, reliable = _native_pixels(points, frame.depth_m, frame.depth_intrinsics, transform)
        seeds = np.flatnonzero(reliable & adjacent[np.maximum(uv[:, 1], 0), np.maximum(uv[:, 0], 0)])
        stats.update(robot_native_samples=len(robot_uv), robot_adjacent_scene_seeds=len(seeds))
        if not len(seeds):
            return finish('no_robot_adjacent_scene_pixels')
        # A reviewable component's bbox diagonal is <=40 mm, so every point in
        # it lies within 40 mm of its robot-adjacent seed. This bounds graph work
        # only; exact radius isolation still uses every original support point.
        distances, _ = spatial_tree(points[seeds]).query(points,
            distance_upper_bound=config.max_component_extent_m+2e-6)
        scope = np.flatnonzero(np.isfinite(distances))
        candidates, summary = _component_candidates(points[scope].astype(float, copy=False), config,
            full_support=points, support_indices=scope)
        stats.update(summary, graph_scope='component_extent_bound_around_robot_adjacent_native_pixels',
                     graph_scope_points=len(scope), full_support_points=len(points),
                     graph_scope_radius_m=config.max_component_extent_m+2e-6)
        protected = binary_dilation(protected, structure=np.ones((3, 3), bool))
        for local, extent, linked in candidates:
            selected = scope[local]
            component = {'point_count': len(selected), 'extent_m': extent,
                'original_point_indices': selected.tolist(), 'removed_points': 0,
                'reason': 'evaluated', 'removed': False}
            result.components.append(component)
            if linked:
                component['reason'] = 'connected_to_other_scene_points'
                continue
            pixels = uv[selected]
            if not reliable[selected].all() or len(np.unique(pixels, axis=0)) != len(pixels):
                component['reason'] = 'unreliable_native_pixel_mapping'
                continue
            if not adjacent[pixels[:, 1], pixels[:, 0]].any():
                component['reason'] = 'no_robot_image_boundary'
                continue
            if protected[pixels[:, 1], pixels[:, 0]].any():
                component['reason'] = 'protected_target_or_boundary_pixels'
                continue
            depth = frame.depth_m[pixels[:, 1], pixels[:, 0]].astype(float)
            evidence = saturated_ambiguous_native_samples(frame.left_ir, frame.right_ir, pixels,
                matrix[0, 0]*baseline/depth, config=config)
            removed = evidence['invalid_saturated_native_measurement']
            result.removed_mask[selected[removed]] = True
            component.update(reason=('saturated_ambiguous_native_measurements' if removed.any()
                                      else 'native_measurements_retained'),
                native_uv=pixels.tolist(), removed=bool(removed.any()), removed_points=int(removed.sum()),
                robot_boundary_pixel_count=int(adjacent[pixels[:, 1], pixels[:, 0]].sum()),
                per_point_evidence=_json({**evidence, 'native_depth_m': depth, 'removed': removed}))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        # Incomplete optional provenance preserves every measurement. A normal
        # unavailable image does not create another motion rejection.
        result.removed_mask[:] = False
        for component in result.components:
            component.update(removed=False, removed_points=0,
                             reason='retained_after_invalid_optional_provenance')
        return finish(str(error) or 'invalid_native_stereo_provenance')
    return finish('evaluated')

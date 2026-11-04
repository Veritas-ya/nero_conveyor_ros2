"""Independent belt-region cropping and bounded observed-point voxel sampling."""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/pick_scene_region_filter.yaml'


def _bounds(value, name):
    if value is None:
        return None
    try:
        values = tuple(value)
    except TypeError as error:
        raise ValueError(f'{name} must contain two finite increasing bounds') from error
    if (len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, Real)
                                or not np.isfinite(v) for v in values)
            or values[0] >= values[1]):
        raise ValueError(f'{name} must contain two finite increasing bounds')
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class SceneRegionFilterConfig:
    # An enabled ROI requires explicit bounds and a calibrated belt frame.
    enabled: bool = False
    x_bounds_m: tuple[float, float] | None = None
    y_bounds_m: tuple[float, float] | None = None
    z_bounds_m: tuple[float, float] | None = None
    voxel_size_m: float = .005

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        for name in ('x_bounds_m', 'y_bounds_m', 'z_bounds_m'):
            object.__setattr__(self, name, _bounds(getattr(self, name), name))
        if self.enabled and any(getattr(self, name) is None for name in
                                ('x_bounds_m', 'y_bounds_m', 'z_bounds_m')):
            raise ValueError('enabled scene ROI requires x, y, and z bounds')
        if (isinstance(self.voxel_size_m, bool) or not isinstance(self.voxel_size_m, Real)
                or not np.isfinite(self.voxel_size_m)
                or not .001 <= self.voxel_size_m <= .005):
            raise ValueError('voxel_size_m must be finite and between 0.001 and 0.005 metres')
        object.__setattr__(self, 'voxel_size_m', float(self.voxel_size_m))

    def as_dict(self):
        return {'enabled': self.enabled,
                'x_bounds_m': None if self.x_bounds_m is None else list(self.x_bounds_m),
                'y_bounds_m': None if self.y_bounds_m is None else list(self.y_bounds_m),
                'z_bounds_m': None if self.z_bounds_m is None else list(self.z_bounds_m),
                'voxel_size_m': self.voxel_size_m}


def load_scene_region_filter_config(path=None):
    path = DEFAULT_CONFIG if path is None else Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:
        raise ValueError(f'invalid scene-region filter YAML: {path}') from error
    if not isinstance(document, dict):
        raise ValueError('scene-region filter YAML must be a mapping')
    unknown = set(document)-{'enabled', 'x_bounds_m', 'y_bounds_m', 'z_bounds_m',
                            'voxel_size_m'}
    if unknown:
        raise ValueError(f'unknown scene-region filter settings: {sorted(unknown, key=str)}')
    return SceneRegionFilterConfig(**document)


def _rigid_transform(plane_to_base):
    try:
        source = np.asarray(plane_to_base)
        if source.dtype.kind not in 'iuf':
            raise ValueError('transform coordinates must be real numeric values')
        matrix = source.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError('plane_to_base must be a finite rigid 4 by 4 transform') from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('plane_to_base must be a finite rigid 4 by 4 transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-9)
            or not np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-6)):
        raise ValueError('plane_to_base must contain a proper rigid rotation')
    return matrix


@dataclass
class SceneRegionFilterResult:
    points: np.ndarray
    removed_mask: np.ndarray
    retained_indices: np.ndarray
    statistics: dict

    @property
    def retained_points(self):
        return self.points[self.retained_indices]

def _observed_voxels(points, voxel_size_m):
    """Pick a real point closest to each native bounding-box centre.

    The inverse maps retained-native rows to lexicographically sorted cells.
    Ties use the earliest original observation. Audited errors are distances
    from the selected observation to ALL native observations in that cell.
    """
    if not len(points):
        return (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float))
    with np.errstate(over='ignore', invalid='ignore'):
        cells = np.floor(points/voxel_size_m)
    if (not np.isfinite(cells).all() or np.any(cells < -(2.**63))
            or np.any(cells >= 2.**63)):
        raise ValueError('scene coordinates exceed the voxel index range')
    # Sort once, then reduce contiguous cells. This preserves native-row
    # ties and the original lexicographic cell order without scatter passes.
    cells = cells.astype(np.int64)
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    starts = np.r_[0, 1+np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1))]
    keys = sorted_cells[starts]
    counts = np.diff(np.r_[starts, len(points)])
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[order] = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
    grouped = points[order]
    lower = np.minimum.reduceat(grouped, starts, axis=0)
    upper = np.maximum.reduceat(grouped, starts, axis=0)
    centers = lower+(upper-lower)*.5
    distances_squared = np.einsum('ij,ij->i', points-centers[inverse], points-centers[inverse])
    closest = np.minimum.reduceat(distances_squared[order], starts)
    selected = distances_squared == closest[inverse]
    representatives = np.minimum.reduceat(np.where(selected[order], order, len(points)), starts)
    errors = np.linalg.norm(points-points[representatives[inverse]], axis=1)
    max_errors = np.maximum.reduceat(errors[order], starts)
    return representatives, keys, inverse, counts, max_errors


def filter_scene_region(points, plane_to_base=None, config=None):
    """Crop base-frame points by explicit belt-local bounds, without sampling.

    ``plane_to_base`` maps belt-local coordinates INTO robot-base coordinates.
    Cropping includes every point in the 3D region, regardless of image colour
    or object labels. Disabling ROI permits an absent frame and preserves all
    native points. The ROI mask always uses input indices. Call voxel_downsample
    separately AFTER any native-density filtering, so sampling cannot alter
    the density filter's neighbourhood support.
    """
    config = SceneRegionFilterConfig() if config is None else config
    if not isinstance(config, SceneRegionFilterConfig):
        raise ValueError('scene-region filter requires SceneRegionFilterConfig')
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf':
        raise ValueError('scene point cloud must be a numeric N by 3 array')
    if (points.dtype.kind in 'iu' and
            (np.any(points > 2**53) or (points.dtype.kind == 'i' and np.any(points < -(2**53))))):
        raise ValueError('integer scene coordinates exceed exact float64 precision')
    numeric_points = points.astype(float, copy=False)
    if not np.isfinite(numeric_points).all():
        raise ValueError('scene point cloud must contain finite coordinates')
    points = points.copy()
    removed = np.zeros(len(points), dtype=bool)
    frame = None
    local_bounds = None
    if config.enabled:
        frame = _rigid_transform(plane_to_base)
        # Use the supplied matrix's actual inverse, including accepted roundoff
        # in its rotation, rather than assuming a numerically perfect R.T.
        with np.errstate(over='ignore', invalid='ignore'):
            local = np.linalg.solve(frame[:3, :3], (numeric_points-frame[:3, 3]).T).T
        if not np.isfinite(local).all():
            raise ValueError('scene coordinates overflow the belt-frame transform')
        local_bounds = np.array([config.x_bounds_m, config.y_bounds_m, config.z_bounds_m])
        # Only compensate roundoff in a rigid transform at inclusive bounds.
        # This picometre tolerance is not an obstacle/region safety margin.
        removed = np.any((local < local_bounds[:, 0]-1e-12)
                         | (local > local_bounds[:, 1]+1e-12), axis=1)
    retained_indices = np.flatnonzero(~removed)
    statistics = {
        'algorithm': 'belt_local_scene_roi',
        'input_points': len(points), 'removed_points': int(removed.sum()),
        'retained_points': len(retained_indices),
        'parameters': config.as_dict(),
        'roi': {'enabled': config.enabled, 'bounds_inclusive': True,
                'boundary_roundoff_tolerance_m': 1e-12,
                'frame_convention': 'plane_to_base_maps_local_to_base',
                'plane_to_base': None if frame is None else frame.tolist(),
                'bounds_m': None if local_bounds is None else local_bounds.tolist()},
    }
    return SceneRegionFilterResult(points, removed, retained_indices, statistics)


@dataclass
class VoxelDownsampleResult:
    points: np.ndarray
    representative_indices: np.ndarray
    representative_counts: np.ndarray
    representative_max_errors_m: np.ndarray
    coverage_indices: np.ndarray
    voxel_keys: np.ndarray
    voxel_inverse: np.ndarray
    voxel_counts: np.ndarray
    voxel_primary_indices: np.ndarray
    voxel_primary_max_errors_m: np.ndarray
    statistics: dict

    @property
    def retained_indices(self):
        return self.representative_indices

    @property
    def retained_points(self):
        return self.points[self.representative_indices]


def voxel_downsample(points, voxel_size_m=.005, max_error_m=.004):
    """Keep real observations with an audited input-to-output distance budget.

    Each base-frame voxel selects one observed point near its native bounding
    box centre. Every observation farther than max_error_m from that primary
    is additionally retained and represents itself. Thus EVERY input point is
    represented by a real retained point within the budget, including thin
    structures and opposite voxel corners. No centroid is introduced.

    coverage_indices maps each input row to its real representative's INPUT
    index. Representative counts/errors describe this mapping after extra
    retention; voxel_primary_max_errors_m audits the initial per-cell choice.
    """
    if (isinstance(voxel_size_m, bool) or not isinstance(voxel_size_m, Real)
            or not np.isfinite(voxel_size_m) or not .001 <= voxel_size_m <= .005):
        raise ValueError('voxel_size_m must be finite and between 0.001 and 0.005 metres')
    if (isinstance(max_error_m, bool) or not isinstance(max_error_m, Real)
            or not np.isfinite(max_error_m) or not 0. < max_error_m <= .004):
        raise ValueError('max_error_m must be finite, positive, and at most 0.004 metres')
    voxel_size_m = float(voxel_size_m)
    max_error_m = float(max_error_m)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf':
        raise ValueError('scene point cloud must be a numeric N by 3 array')
    if (points.dtype.kind in 'iu' and
            (np.any(points > 2**53) or (points.dtype.kind == 'i' and np.any(points < -(2**53))))):
        raise ValueError('integer scene coordinates exceed exact float64 precision')
    numeric_points = points.astype(float, copy=False)
    if not np.isfinite(numeric_points).all():
        raise ValueError('scene point cloud must contain finite coordinates')
    points = points.copy()
    primary, keys, inverse, counts, primary_errors = _observed_voxels(numeric_points, voxel_size_m)
    coverage = primary[inverse]
    input_errors = np.linalg.norm(numeric_points-numeric_points[coverage], axis=1)
    extra = np.flatnonzero(input_errors > max_error_m)
    coverage[extra] = extra
    input_errors[extra] = 0.
    representatives = np.unique(np.concatenate((primary, extra)))
    assignment = np.searchsorted(representatives, coverage)
    representative_counts = np.bincount(assignment, minlength=len(representatives))
    representative_errors = np.zeros(len(representatives))
    np.maximum.at(representative_errors, assignment, input_errors)
    maximum_error = float(representative_errors.max()) if len(representatives) else 0.
    statistics = {
        'algorithm': 'observed_voxel_primary_with_bounded_error_retention',
        'input_points': len(points), 'output_points': len(representatives),
        'removed_points': len(points)-len(representatives),
        'frame': 'base', 'voxel_size_m': voxel_size_m, 'max_error_budget_m': max_error_m,
        'voxel_count': len(keys), 'extra_retained_points': len(extra),
        'representative': 'observed_point_nearest_native_bbox_center',
        'tie_break': 'earliest_input_index', 'count_source': 'input_native_points',
        'represented_native_points': int(representative_counts.sum()),
        'max_representative_error_m': maximum_error,
        'max_primary_error_m': float(primary_errors.max()) if len(primary_errors) else 0.,
        'theoretical_primary_max_error_m': float(np.sqrt(3)*voxel_size_m),
        'coverage_within_budget': bool(maximum_error <= max_error_m),
    }
    return VoxelDownsampleResult(points, representatives, representative_counts,
                                 representative_errors, coverage, keys, inverse, counts,
                                 primary, primary_errors, statistics)

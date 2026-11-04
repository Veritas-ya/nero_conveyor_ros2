"""Radius outliers and optional verified isolated components in an observed scene."""
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .scene_workspace import spatial_tree
import yaml

from .scene_depth_filter import DepthEdgeTailConfig


DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/pick_scene_noise_filter.yaml'


@dataclass(frozen=True)
class SceneNoiseFilterConfig:
    enabled: bool = True
    radius_m: float = .005
    min_neighbors: int = 3
    min_cluster_points: int = 0
    max_cluster_extent_m: float = .010
    depth_edge_tail: DepthEdgeTailConfig = field(default_factory=DepthEdgeTailConfig)

    def __post_init__(self):
        depth_config = self.depth_edge_tail
        if isinstance(depth_config, Mapping):
            try:
                depth_config = DepthEdgeTailConfig(**depth_config)
            except TypeError as error:
                raise ValueError(f'invalid depth_edge_tail settings: {error}') from error
        if not isinstance(depth_config, DepthEdgeTailConfig):
            raise ValueError('depth_edge_tail must be a depth edge filter configuration')
        object.__setattr__(self, 'depth_edge_tail', depth_config)
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        if (isinstance(self.radius_m, bool) or not isinstance(self.radius_m, Real)
                or not np.isfinite(self.radius_m) or not .002 <= self.radius_m <= .02):
            raise ValueError('radius_m must be a finite number between 0.002 and 0.02 metres')
        if (isinstance(self.min_neighbors, bool) or not isinstance(self.min_neighbors, Integral)
                or not 1 <= self.min_neighbors <= 20):
            raise ValueError('min_neighbors must be an integer between 1 and 20')
        if (isinstance(self.min_cluster_points, bool)
                or not isinstance(self.min_cluster_points, Integral)
                or not 0 <= self.min_cluster_points <= 100):
            raise ValueError('min_cluster_points must be an integer between 0 and 100')
        if (isinstance(self.max_cluster_extent_m, bool)
                or not isinstance(self.max_cluster_extent_m, Real)
                or not np.isfinite(self.max_cluster_extent_m)
                or not .002 <= self.max_cluster_extent_m <= .03):
            raise ValueError('max_cluster_extent_m must be a finite number between 0.002 and 0.03 metres')
        object.__setattr__(self, 'radius_m', float(self.radius_m))
        object.__setattr__(self, 'min_neighbors', int(self.min_neighbors))
        object.__setattr__(self, 'min_cluster_points', int(self.min_cluster_points))
        object.__setattr__(self, 'max_cluster_extent_m', float(self.max_cluster_extent_m))

    def as_dict(self):
        return {'enabled': self.enabled, 'radius_m': self.radius_m,
                'min_neighbors': self.min_neighbors,
                'min_cluster_points': self.min_cluster_points,
                'max_cluster_extent_m': self.max_cluster_extent_m,
                'depth_edge_tail': asdict(self.depth_edge_tail)}


def load_scene_noise_filter_config(path=None):
    path = DEFAULT_CONFIG if path is None else Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:
        raise ValueError(f'invalid scene-noise filter YAML: {path}') from error
    if not isinstance(document, dict):
        raise ValueError('scene-noise filter YAML must be a mapping')
    unknown = set(document)-{'enabled', 'radius_m', 'min_neighbors', 'min_cluster_points',
                            'max_cluster_extent_m', 'depth_edge_tail'}
    if unknown:
        raise ValueError(f'unknown scene-noise filter settings: {sorted(unknown, key=str)}')
    return SceneNoiseFilterConfig(**document)


@dataclass
class SceneNoiseFilterResult:
    points: np.ndarray
    removed_mask: np.ndarray
    neighbor_counts: np.ndarray | None
    statistics: dict

    @property
    def retained_points(self):
        return self.points[~self.removed_mask]


def _small_cluster_mask(points, radius_m, min_cluster_points, max_cluster_extent_m):
    """Use a bounded KNN graph for candidates, then verify exact-radius isolation.

    Every graph edge is within the radius, but the nearest-neighbour graph can
    omit other radius links. A small component is removable only after querying
    ALL its points at the exact radius and finding no link to a different graph
    label. Missing graph edges therefore preserve extra points, never split a
    continuous surface into removable components.
    """
    tree = spatial_tree(points)
    k = min(4, len(points))
    distances, indices = tree.query(points, k=k)
    distances = np.asarray(distances).reshape(len(points), k)
    indices = np.asarray(indices).reshape(len(points), k)
    sources = np.broadcast_to(np.arange(len(points))[:, None], indices.shape)
    valid = (indices != sources) & (indices < len(points)) & (distances <= radius_m)
    # With duplicate coordinates the query point need not appear in the first
    # four tied results. Still cap the graph at three OTHER neighbours per row.
    valid &= np.cumsum(valid, axis=1) <= 3
    sources = sources[valid]; targets = indices[valid]
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)),
                       shape=(len(points), len(points))).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    lower = np.full((component_count, 3), np.inf)
    upper = np.full((component_count, 3), -np.inf)
    np.minimum.at(lower, labels, points)
    np.maximum.at(upper, labels, points)
    # The native-point bounding-box diagonal bounds every point-to-point
    # distance from above. A long thin component is kept despite a low count.
    extents = np.linalg.norm(upper-lower, axis=1)
    small = component_sizes < min_cluster_points
    compact = extents <= max_cluster_extent_m
    candidates = small & compact
    candidate_points = np.flatnonzero(candidates[labels])
    external_link = np.zeros(component_count, dtype=bool)
    # Bound temporary ragged neighbour lists; full-cloud query_pairs would
    # materialize millions of unnecessary edges on a dense support surface.
    for begin in range(0, len(candidate_points), 2048):
        selected = candidate_points[begin:begin+2048]
        neighborhoods = tree.query_ball_point(points[selected], radius_m)
        for index, neighbors in zip(selected, neighborhoods):
            if np.any(labels[neighbors] != labels[index]):
                external_link[labels[index]] = True
    removed_components = candidates & ~external_link
    removed = removed_components[labels]
    return removed, {'component_count': int(component_count),
                     'small_component_count': int(small.sum()),
                     'candidate_component_count': int(candidates.sum()),
                     'exact_checked_points': len(candidate_points),
                     'compact_components_removed': int(removed_components.sum()),
                     'extent_protected_small_components': int((small & ~compact).sum()),
                     'radius_link_protected_small_components': int((candidates & external_link).sum())}


def filter_scene_noise(points, config=None, *, collect_neighbor_counts=True):
    """Count neighbours once on the original input; exclude each point itself.

    ``points`` and ``removed_mask`` retain the original indexing. No second pass
    recomputes density after neighbours are classified. An optional component
    pass uses the radius-retained native points and never repeats density tests.
    Disabled filtering preserves every value, dtype, and input order.
    """
    config = SceneNoiseFilterConfig() if config is None else config
    if not isinstance(config, SceneNoiseFilterConfig):
        raise ValueError('scene-noise filter requires SceneNoiseFilterConfig')
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf':
        raise ValueError('scene point cloud must be a numeric N by 3 array')
    numeric_points = points.astype(float, copy=False)
    if not np.isfinite(numeric_points).all():
        raise ValueError('scene point cloud must contain finite coordinates')
    points = points.copy()
    removed = np.zeros(len(points), dtype=bool)
    counts = None
    radius_removed = 0; cluster_removed = 0
    cluster_statistics = {'component_count': 0, 'small_component_count': 0,
                          'candidate_component_count': 0, 'exact_checked_points': 0,
                          'compact_components_removed': 0, 'extent_protected_small_components': 0,
                          'radius_link_protected_small_components': 0}
    if config.enabled:
        if len(points):
            # query_ball_point includes the query point. Coincident OTHER
            # points remain neighbours; subtract only the point's own index.
            tree = spatial_tree(numeric_points)
            if collect_neighbor_counts:
                counts = np.asarray(tree.query_ball_point(
                    numeric_points, config.radius_m, return_length=True), dtype=np.int64)-1
            else:
                # Need >=m OTHER points: the (m+1)th nearest including self
                # suffices, even with duplicate coordinates/tied self index.
                # Exact query, inclusive radius; ambiguous floating boundary
                # points retain the original query_ball_point predicate.
                distances, _ = tree.query(numeric_points, k=[config.min_neighbors+1])
                furthest = distances[:, 0]
                removed = furthest > config.radius_m
                edge = np.isfinite(furthest) & (np.abs(furthest-config.radius_m)
                    <= 16*np.finfo(float).eps*max(1., config.radius_m))
                if edge.any():
                    exact = np.asarray(tree.query_ball_point(numeric_points[edge],
                        config.radius_m, return_length=True))-1
                    removed[edge] = exact < config.min_neighbors
        else:
            counts = np.empty(0, dtype=np.int64)
        if collect_neighbor_counts or not len(points):
            removed = counts < config.min_neighbors
        radius_removed = int(removed.sum())
        if config.min_cluster_points:
            retained_indices = np.flatnonzero(~removed)
            if len(retained_indices):
                small, cluster_statistics = _small_cluster_mask(
                    numeric_points[retained_indices], config.radius_m, config.min_cluster_points,
                    config.max_cluster_extent_m)
                removed[retained_indices[small]] = True
                cluster_removed = int(small.sum())
    statistics = {
        'algorithm': 'single_pass_radius_outlier_filter',
        'input_points': len(points), 'removed_points': int(removed.sum()),
        'radius_removed_points': radius_removed, 'cluster_removed_points': cluster_removed,
        'retained_points': int((~removed).sum()),
        'parameters': config.as_dict(), 'neighbors_exclude_self': True,
        'support_cloud': 'original_input', 'passes': 1 if config.enabled else 0,
        'cluster_filter': {'enabled': config.enabled and config.min_cluster_points > 0,
                           'connectivity': 'knn_candidates_with_exact_radius_isolation',
                           'max_nearest_neighbors': 3,
                           'isolation_radius_m': config.radius_m,
                           'size_source': 'radius_retained_native_points',
                           'extent_metric': 'native_point_bbox_diagonal',
                           'max_extent_m': config.max_cluster_extent_m,
                           **cluster_statistics},
    }
    return SceneNoiseFilterResult(points, removed, counts, statistics)

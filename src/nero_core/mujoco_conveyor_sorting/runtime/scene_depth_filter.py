"""Identify small robot-edge depth tails using the unchanged native image.

This is a narrowly scoped single-frame test, not a general classifier of real
objects versus depth artifacts. A component needs same-depth robot support and
an otherwise missing/deep-background boundary. Small detached obstacles and
continuous surfaces are retained. No depth values or robot padding are changed.
"""
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .scene_workspace import spatial_tree


@dataclass(frozen=True)
class DepthEdgeTailConfig:
    enabled: bool = True
    connection_radius_m: float = .005
    max_component_points: int = 160
    max_component_extent_m: float = .04
    robot_depth_tolerance_m: float = .01
    minimum_background_jump_m: float = .03
    minimum_unsupported_boundary_fraction: float = .9
    minimum_background_pixels: int = 3

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        for name in ('connection_radius_m', 'max_component_extent_m',
                     'robot_depth_tolerance_m', 'minimum_background_jump_m'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be a finite positive metric number')
            object.__setattr__(self, name, float(value))
        for name in ('max_component_points', 'minimum_background_pixels'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
            object.__setattr__(self, name, int(value))
        fraction = self.minimum_unsupported_boundary_fraction
        if (isinstance(fraction, bool) or not isinstance(fraction, Real)
                or not np.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError('minimum_unsupported_boundary_fraction must be in (0, 1]')
        object.__setattr__(self, 'minimum_unsupported_boundary_fraction', float(fraction))

    def as_dict(self):
        return asdict(self)


@dataclass
class DepthEdgeTailResult:
    points: np.ndarray
    removed_mask: np.ndarray
    statistics: dict
    components: list[dict]

    @property
    def retained_points(self):
        return self.points[~self.removed_mask]


def _component_candidates(points, config, *, full_support=None, support_indices=None):
    """Bound the graph, then verify every candidate's full-radius isolation.

    KNN can miss radius edges. A candidate with any such edge is retained, so a
    continuous surface cannot be split into a removable tail by this shortcut.
    Optional scope reduction changes only graph construction. Isolation still
    queries the entire original post-self-filter cloud before density removal.
    """
    tree = spatial_tree(points)
    k = min(4, len(points))
    distances, indices = tree.query(points, k=k)
    distances = np.asarray(distances).reshape(len(points), k)
    indices = np.asarray(indices).reshape(len(points), k)
    sources = np.broadcast_to(np.arange(len(points))[:, None], indices.shape)
    valid = (indices != sources) & (distances <= config.connection_radius_m)
    valid &= np.cumsum(valid, axis=1) <= 3
    graph = coo_matrix((np.ones(int(valid.sum()), dtype=np.uint8),
                       (sources[valid], indices[valid])),
                      shape=(len(points), len(points))).tocsr()
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    lower = np.full((count, 3), np.inf)
    upper = np.full((count, 3), -np.inf)
    np.minimum.at(lower, labels, points)
    np.maximum.at(upper, labels, points)
    extents = np.linalg.norm(upper-lower, axis=1)
    small = sizes <= config.max_component_points
    candidates = np.flatnonzero(small & (extents <= config.max_component_extent_m))
    if full_support is None:
        support_tree = tree
        support_labels = labels
    else:
        full_support = np.asarray(full_support)
        support_indices = np.asarray(support_indices)
        if (support_indices.shape != (len(points),) or support_indices.dtype.kind not in 'iu'
                or np.any(support_indices < 0) or np.any(support_indices >= len(full_support))
                or len(np.unique(support_indices)) != len(points)
                or not np.array_equal(full_support[support_indices], points)):
            raise ValueError('scoped component points must map exactly to full support')
        # A -1 label marks all points outside the graph scope. Any radius edge
        # reaching those points protects the candidate just like a different
        # graph component, so cropping cannot sever a real surface or wire.
        support_labels = np.full(len(full_support), -1, dtype=labels.dtype)
        support_labels[support_indices] = labels
        support_tree = spatial_tree(full_support) if len(candidates) else None
    output = []
    for component in candidates:
        selected = np.flatnonzero(labels == component)
        linked = any(np.any(support_labels[neighbors] != component)
                     for neighbors in support_tree.query_ball_point(
                         points[selected], config.connection_radius_m))
        output.append((selected, float(extents[component]), linked))
    return output, {
        'graph_components': int(count), 'candidate_components': len(output),
        'point_count_protected_components': int((~small).sum()),
        'extent_protected_components': int((small & (extents > config.max_component_extent_m)).sum()),
        'exact_radius_link_protected_components': sum(item[2] for item in output),
        'connectivity': 'bounded_knn_with_exact_radius_isolation',
        'support_cloud': 'original_post_self_filter_before_density_removal',
    }


def _robot_candidate_scope(points, robot_points, robot_depths, depth_shape,
                           intrinsics, camera_to_base, config):
    """Bound every removable component using its required robot pixel edge.

    Adjacent pixel rays differ by at most sqrt(1/fx^2 + 1/fy^2). For a component
    edge C and an adjacent robot pixel R, C = zC*rC and R = zR*rR. Their distance
    is at most dz*max_norm(rC) + max(zR)*norm(rC-rR). Every other component point
    lies within its allowed bounding-box diagonal of C. Thus all points of a
    removable component are inside the resulting radius of a robot sample.
    The transform operator norm also handles non-unit rotation inputs; 2 um
    accounts for the two already permitted native reconstruction errors.
    """
    height, width = depth_shape
    u = np.array([0., width-1., 0., width-1.])
    v = np.array([0., 0., height-1., height-1.])
    rays = np.column_stack(((u-intrinsics.cx)/intrinsics.fx,
                            (v-intrinsics.cy)/intrinsics.fy, np.ones(4)))
    maximum_ray_norm = float(np.linalg.norm(rays, axis=1).max())
    pixel_ray_step = float(np.hypot(1./intrinsics.fx, 1./intrinsics.fy))
    maximum_robot_depth = float(np.max(robot_depths))
    transform_norm = float(np.linalg.norm(camera_to_base[:3, :3], ord=2))
    robot_edge_bound = transform_norm*(config.robot_depth_tolerance_m*maximum_ray_norm
                                      + maximum_robot_depth*pixel_ray_step)
    radius = config.max_component_extent_m+robot_edge_bound+2e-6
    distances, _ = spatial_tree(robot_points).query(points, distance_upper_bound=radius)
    selected = np.flatnonzero(np.isfinite(distances))
    return selected, {
        'graph_scope': 'robot_adjacent_native_depth_bound',
        'graph_scope_points': len(selected), 'full_support_points': len(points),
        'graph_scope_radius_m': radius, 'robot_pixel_edge_distance_bound_m': robot_edge_bound,
        'maximum_camera_ray_norm': maximum_ray_norm,
        'maximum_robot_native_depth_m': maximum_robot_depth,
        'camera_to_base_operator_norm': transform_norm,
        'scope_reconstruction_allowance_m': 2e-6,
    }


def _native_pixels(points, depth, intrinsics, camera_to_base):
    from .scene_workspace import native_pixels
    key = (id(depth), depth.shape, intrinsics.fx, intrinsics.fy, intrinsics.cx,
           intrinsics.cy, np.asarray(camera_to_base).tobytes())
    return native_pixels(points, key, lambda:
        _compute_native_pixels(points, depth, intrinsics, camera_to_base))


def _compute_native_pixels(points, depth, intrinsics, camera_to_base):
    """Require projection onto an actual native sample, within numerical error.

    A one-micrometre reconstruction allowance covers array precision only; it
    does not admit synthetic, moved, downsampled-centroid or stale scene points.
    """
    inverse = np.linalg.inv(camera_to_base)
    camera = points @ inverse[:3, :3].T + inverse[:3, 3]
    reliable = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    uv = np.full((len(points), 2), -1, dtype=np.int64)
    projected = np.zeros((len(points), 2), dtype=float)
    usable = np.flatnonzero(reliable)
    projected[usable] = np.column_stack((
        camera[usable, 0]*intrinsics.fx/camera[usable, 2]+intrinsics.cx,
        camera[usable, 1]*intrinsics.fy/camera[usable, 2]+intrinsics.cy))
    reliable &= np.isfinite(projected).all(axis=1)
    reliable &= ((projected[:, 0] >= -.5) & (projected[:, 0] < depth.shape[1]-.5)
                 & (projected[:, 1] >= -.5) & (projected[:, 1] < depth.shape[0]-.5))
    usable = np.flatnonzero(reliable)
    uv[usable] = np.rint(projected[usable]).astype(np.int64)
    native_depth = depth[uv[usable, 1], uv[usable, 0]].astype(float)
    native_camera = np.column_stack((
        (uv[usable, 0]-intrinsics.cx)*native_depth/intrinsics.fx,
        (uv[usable, 1]-intrinsics.cy)*native_depth/intrinsics.fy, native_depth))
    reconstructed = native_camera @ camera_to_base[:3, :3].T+camera_to_base[:3, 3]
    reliable[usable] &= (np.isfinite(native_depth) & (native_depth > 0)
                         & (np.linalg.norm(reconstructed-points[usable], axis=1) <= 1e-6))
    return uv, reliable


def filter_depth_edge_tails(points, *, frame=None, self_filter_result=None,
                           protected_mask=None, config=None):
    """Return a mask over original native post-self-filter points, in base metres.

    Missing provenance preserves points and records a skip reason. This filter
    never substitutes synthetic depth or new robot padding for image evidence.
    """
    config = DepthEdgeTailConfig() if config is None else config
    if not isinstance(config, DepthEdgeTailConfig):
        raise ValueError('depth-edge filter requires DepthEdgeTailConfig')
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'iuf':
        raise ValueError('scene point cloud must be a numeric N by 3 array')
    if not np.isfinite(points).all():
        raise ValueError('scene point cloud must contain finite coordinates')
    points = points.copy()
    removed = np.zeros(len(points), dtype=bool)
    components = []
    statistics = {'algorithm': 'native_depth_robot_edge_tail_filter',
                  'input_points': len(points), 'removed_points': 0,
                  'retained_points': len(points), 'parameters': config.as_dict(),
                  'reason': 'evaluated', 'candidate_components': 0,
                  'removed_components': 0, 'native_reconstruction_tolerance_m': 1e-6}
    result = DepthEdgeTailResult(points, removed, statistics, components)
    def skip(reason):
        statistics['reason'] = reason
        return result
    if not config.enabled:
        return skip('disabled')
    if not len(points):
        return skip('empty_cloud')
    if frame is None:
        return skip('missing_native_depth_frame')
    if self_filter_result is None:
        return skip('missing_self_filter_result')
    if protected_mask is None:
        return skip('missing_protected_target_mask')
    try:
        depth = np.asarray(frame.depth_m)
        intrinsics = frame.depth_intrinsics
        camera_to_base = np.asarray(frame.camera_to_world_cv, dtype=float)
        valid_frame = (depth.ndim == 2 and camera_to_base.shape == (4, 4)
                       and np.isfinite(camera_to_base).all()
                       and np.isfinite([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]).all()
                       and intrinsics.fx > 0 and intrinsics.fy > 0)
        if not valid_frame:
            return skip('invalid_native_depth_frame')
        if np.any(np.abs(intrinsics.distortion) > 1e-10):
            return skip('unsupported_depth_distortion')
        np.linalg.inv(camera_to_base)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return skip('invalid_native_depth_frame')
    protected = np.asarray(protected_mask)
    if protected.shape != depth.shape or protected.dtype.kind != 'b':
        return skip('invalid_protected_target_mask')
    try:
        self_points = np.asarray(self_filter_result.points, dtype=float)
        self_removed = np.asarray(self_filter_result.removed_mask)
        if (self_points.ndim != 2 or self_points.shape[1] != 3
                or self_removed.shape != (len(self_points),) or self_removed.dtype.kind != 'b'):
            return skip('invalid_self_filter_result')
        robot_points = self_points[self_removed]
    except (AttributeError, TypeError, ValueError):
        return skip('invalid_self_filter_result')
    if not len(robot_points):
        return skip('no_filtered_robot_samples')
    robot_uv, robot_reliable = _native_pixels(robot_points, depth, intrinsics, camera_to_base)
    statistics['robot_native_samples'] = int(robot_reliable.sum())
    statistics['unreliable_robot_samples'] = int((~robot_reliable).sum())
    if not robot_reliable.any():
        return skip('no_reliable_robot_depth_pixels')
    robot_mask = np.zeros(depth.shape, dtype=bool)
    mapped_robot = robot_uv[robot_reliable]
    robot_mask[mapped_robot[:, 1], mapped_robot[:, 0]] = True
    scope, scope_statistics = _robot_candidate_scope(
        points, robot_points[robot_reliable], depth[mapped_robot[:, 1], mapped_robot[:, 0]],
        depth.shape, intrinsics, camera_to_base, config)
    statistics.update(scope_statistics)
    if not len(scope):
        return skip('no_robot_adjacent_scene_points')
    candidates, summary = _component_candidates(
        points[scope].astype(float, copy=False), config,
        full_support=points, support_indices=scope)
    statistics.update(summary)
    for selected, extent, linked in candidates:
        selected = scope[selected]
        component = {'point_count': len(selected), 'extent_m': extent,
                     'original_point_indices': selected.tolist(),
                     'bbox_base_m': [points[selected].min(axis=0).tolist(), points[selected].max(axis=0).tolist()],
                     'removed': False, 'reason': 'evaluated'}
        components.append(component)
        if linked:
            component['reason'] = 'connected_to_other_scene_points'
            continue
        uv, reliable = _native_pixels(points[selected], depth, intrinsics, camera_to_base)
        if not reliable.all() or len(np.unique(uv, axis=0)) != len(uv):
            component['reason'] = 'unreliable_native_pixel_mapping'
            component['unreliable_points'] = int((~reliable).sum())
            continue
        lower = uv.min(axis=0); upper = uv.max(axis=0)
        component['bbox_depth_uv'] = [lower.tolist(), upper.tolist()]
        if np.any(lower == 0) or upper[0] == depth.shape[1]-1 or upper[1] == depth.shape[0]-1:
            component['reason'] = 'incomplete_image_boundary'
            continue
        if protected[uv[:, 1], uv[:, 0]].any():
            component['reason'] = 'protected_target_pixels'
            continue
        # Only the candidate's local image patch is allocated; no full-image
        # graph or repeated global dilation is needed for a handful of tails.
        u0, v0 = lower-1; u1, v1 = upper+2
        patch = depth[v0:v1, u0:u1]
        mask = np.zeros(patch.shape, dtype=bool)
        mask[uv[:, 1]-v0, uv[:, 0]-u0] = True
        boundary = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & ~mask
        ys, xs = np.nonzero(boundary)
        # At a boundary pixel use the closest depth among its adjacent native
        # component pixels; a background label thus requires every adjacency
        # to have the configured positive depth jump.
        delta = np.empty(len(ys), dtype=float)
        for index, (y, x) in enumerate(zip(ys, xs)):
            ya, yb = max(0, y-1), min(mask.shape[0], y+2)
            xa, xb = max(0, x-1), min(mask.shape[1], x+2)
            differences = patch[y, x]-patch[ya:yb, xa:xb][mask[ya:yb, xa:xb]]
            delta[index] = differences[np.argmin(np.abs(differences))]
        valid_depth = np.isfinite(patch[ys, xs]) & (patch[ys, xs] > 0)
        robot = robot_mask[v0:v1, u0:u1][ys, xs]
        same_robot = robot & valid_depth & (np.abs(delta) <= config.robot_depth_tolerance_m)
        other = ~robot
        missing = ~valid_depth & other
        background = other & valid_depth & (delta >= config.minimum_background_jump_m)
        denominator = int(other.sum())
        unsupported = int(missing.sum()+background.sum())
        fraction = unsupported/denominator if denominator else 0.
        component.update(boundary_pixels=len(ys), robot_boundary_pixels=int(robot.sum()),
                         same_depth_robot_boundary_pixels=int(same_robot.sum()),
                         nonrobot_boundary_pixels=denominator,
                         missing_boundary_pixels=int(missing.sum()),
                         background_boundary_pixels=int(background.sum()),
                         supported_nonrobot_boundary_pixels=int((other & valid_depth & ~background).sum()),
                         unsupported_boundary_fraction=fraction)
        if protected[v0:v1, u0:u1][ys, xs].any():
            component['reason'] = 'protected_target_boundary'
        elif not same_robot.any():
            component['reason'] = 'no_same_depth_robot_boundary'
        elif int(background.sum()) < config.minimum_background_pixels:
            component['reason'] = 'insufficient_deep_background_boundary'
        elif fraction < config.minimum_unsupported_boundary_fraction:
            component['reason'] = 'supported_scene_boundary'
        else:
            removed[selected] = True
            component['removed'] = True
            component['reason'] = 'robot_edge_depth_tail'
    statistics['removed_points'] = int(removed.sum())
    statistics['retained_points'] = int((~removed).sum())
    statistics['removed_components'] = sum(item['removed'] for item in components)
    return result

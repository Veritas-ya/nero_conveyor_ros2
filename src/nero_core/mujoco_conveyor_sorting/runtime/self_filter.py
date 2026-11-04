"""Model-based robot self filtering without ROS or hardware dependencies.

Like robot_body_filter, this separates containment from sensor-ray shadows.
Padding is an explicit model uncertainty band, not proof of free space. Shadow
points are labelled but retained. Convex collision bodies approximate the robot;
neither concavities nor objects touching its padding band can be resolved here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from scipy.spatial import HalfspaceIntersection, cKDTree
from .scene_workspace import spatial_tree
import yaml

OUTSIDE, ROBOT, PADDING, SHADOW, PROTECTED, INVALID = range(6)
LABEL_NAMES = {OUTSIDE: 'outside', ROBOT: 'robot_model', PADDING: 'padding_band',
               SHADOW: 'shadow_retained', PROTECTED: 'protected', INVALID: 'invalid'}
LINK_NAMES = {'base_link', *(f'link{i}' for i in range(1, 8)), 'gripper_flange',
              'gripper_base', 'gripper_link1', 'gripper_link2'}
DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/pick_self_filter.yaml'


def _bounded(value, label, upper):
    if isinstance(value, (bool, str)):
        raise ValueError(f'{label} must be a finite metric number')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label} must be a finite metric number') from error
    if not np.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(f'{label} must be between 0 and {upper} metres')
    return value


@dataclass(frozen=True)
class SelfFilterConfig:
    padding_m: float = .015
    link_padding_m: Mapping[str, float] = field(default_factory=lambda: {'link2': .022})
    shadow_threshold_m: float = .003

    def __post_init__(self):
        object.__setattr__(self, 'padding_m', _bounded(self.padding_m, 'padding_m', .03))
        object.__setattr__(self, 'shadow_threshold_m',
                           _bounded(self.shadow_threshold_m, 'shadow_threshold_m', .05))
        if not isinstance(self.link_padding_m, Mapping):
            raise ValueError('link_padding_m must be a mapping of robot link names')
        unknown = set(self.link_padding_m)-LINK_NAMES
        if unknown:
            raise ValueError(f'unknown self-filter links: {sorted(unknown, key=str)}')
        object.__setattr__(self, 'link_padding_m',
                           {name: _bounded(value, f'link_padding_m.{name}', .03)
                            for name, value in self.link_padding_m.items()})

    def padding_for(self, name):
        return self.link_padding_m.get(name, self.padding_m)

    def as_dict(self):
        return {'padding_m': self.padding_m, 'link_padding_m': dict(self.link_padding_m),
                'shadow_threshold_m': self.shadow_threshold_m}


def load_self_filter_config(path=None):
    path = DEFAULT_CONFIG if path is None else Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:
        raise ValueError(f'invalid self-filter YAML: {path}') from error
    if not isinstance(document, dict):
        raise ValueError('self-filter YAML must be a mapping')
    unknown = set(document)-{'padding_m', 'link_padding_m', 'shadow_threshold_m'}
    if unknown:
        raise ValueError(f'unknown self-filter settings: {sorted(unknown, key=str)}')
    return SelfFilterConfig(**document)


@dataclass
class ConvexBody:
    geom_id: int
    name: str
    equations: np.ndarray
    vertices: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    padding_m: float

    def __post_init__(self):
        self.equations = np.asarray(self.equations, dtype=float).copy()
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.position = np.asarray(self.position, dtype=float)
        self.rotation = np.asarray(self.rotation, dtype=float)
        if (self.equations.ndim != 2 or self.equations.shape[1] != 4
                or self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or self.position.shape != (3,) or self.rotation.shape != (3, 3)
                or len(self.vertices) < 4):
            raise ValueError('invalid convex robot geometry dimensions')
        if not all(np.isfinite(a).all() for a in
                   (self.equations, self.vertices, self.position, self.rotation)):
            raise ValueError('robot geometry must be finite')
        if (not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-6)):
            raise ValueError('robot geometry rotation must be rigid')
        lengths = np.linalg.norm(self.equations[:, :3], axis=1)
        if not len(lengths) or np.any(lengths < 1e-12):
            raise ValueError('robot hull normals must be nonzero')
        self.equations /= lengths[:, None]
        self.padding_m = _bounded(self.padding_m, 'body padding_m', .03)
        self.padded_equations = self.equations.copy()
        self.padded_equations[:, 3] -= self.padding_m
        # Offsetting halfspaces can extend a corner by more than padding.
        # Bound the ACTUAL offset polytope, not nominal_radius + padding.
        interior = self.vertices.mean(axis=0)
        self.radius = float(np.linalg.norm(self.vertices, axis=1).max())
        if self.padding_m:
            padded_vertices = HalfspaceIntersection(self.padded_equations, interior).intersections
            self.padded_radius = float(np.linalg.norm(padded_vertices, axis=1).max())
        else:
            self.padded_radius = self.radius


def _inside(local_points, equations):
    return np.all(local_points @ equations[:, :3].T+equations[:, 3] <= 1e-10, axis=1)


def _ray_entry(directions, origin, body):
    """Positive first intersection with nominal convex body, in metric range."""
    local_origin = (origin-body.position) @ body.rotation
    numerator = -(body.equations[:, :3] @ local_origin+body.equations[:, 3])
    denominator = (directions @ body.rotation) @ body.equations[:, :3].T
    crossings = np.divide(numerator, denominator, out=np.zeros_like(denominator),
                          where=np.abs(denominator) > 1e-12)
    entry = np.max(np.where(denominator < -1e-12, crossings, -np.inf), axis=1)
    exit_ = np.min(np.where(denominator > 1e-12, crossings, np.inf), axis=1)
    parallel_outside = np.any((np.abs(denominator) <= 1e-12) & (numerator < 0), axis=1)
    valid = ~parallel_outside & (entry > 0) & np.isfinite(entry) & (exit_ >= entry)
    return np.where(valid, entry, np.inf)


@dataclass
class SelfFilterResult:
    points: np.ndarray
    labels: np.ndarray
    removed_mask: np.ndarray
    geom_ids: np.ndarray
    statistics: dict


class ModelSelfFilter:
    """Classify raw points; keep the camera-shadow label out of the removal mask."""
    def __init__(self, config=None):
        self.config = SelfFilterConfig() if config is None else config
        if not isinstance(self.config, SelfFilterConfig):
            raise ValueError('self filter requires SelfFilterConfig')

    def classify(self, points, bodies, camera_origin=None, protected_mask=None):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('point cloud must be N by 3 in base-frame metres')
        bodies = list(bodies)
        if len({body.geom_id for body in bodies}) != len(bodies):
            raise ValueError('self-filter geometry IDs must be unique')
        valid = np.isfinite(points).all(axis=1)
        protected = np.zeros(len(points), bool) if protected_mask is None else np.asarray(protected_mask)
        if protected.shape != (len(points),) or protected.dtype != bool:
            raise ValueError('protected_mask must be a boolean value for each point')
        labels = np.full(len(points), OUTSIDE, np.uint8)
        labels[~valid] = INVALID
        labels[protected & valid] = PROTECTED
        geom_ids = np.full(len(points), -1, dtype=np.int32)
        eligible = np.flatnonzero(valid & ~protected)
        tree = spatial_tree(points[eligible])
        for body in bodies:
            ids = eligible[tree.query_ball_point(body.position, body.padded_radius+1e-9)]
            for begin in range(0, len(ids), 512):
                selected = ids[begin:begin+512]
                local = (points[selected]-body.position) @ body.rotation
                nominal = _inside(local, body.equations)
                padded = _inside(local, body.padded_equations)
                padding_ids = selected[padded & (labels[selected] == OUTSIDE)]
                labels[padding_ids] = PADDING
                geom_ids[padding_ids] = body.geom_id
                # Nominal containment takes precedence over another link's pad.
                labels[selected[nominal]] = ROBOT
                geom_ids[selected[nominal]] = body.geom_id
        if camera_origin is not None:
            origin = np.asarray(camera_origin, dtype=float)
            if origin.shape != (3,) or not np.isfinite(origin).all():
                raise ValueError('camera origin must be 3 finite base-frame metres')
            outside = np.flatnonzero(labels == OUTSIDE)
            delta = points[outside]-origin
            ranges = np.linalg.norm(delta, axis=1)
            directions = np.divide(delta, ranges[:, None], out=np.zeros_like(delta),
                                   where=ranges[:, None] > 1e-12)
            first = np.full(len(outside), np.inf)
            first_geom = np.full(len(outside), -1, np.int32)
            for body in bodies:
                center = body.position-origin
                projection = directions @ center
                perpendicular_sq = np.dot(center, center)-projection**2
                ids = np.flatnonzero((perpendicular_sq <= body.radius**2+1e-10)
                                     & (projection+body.radius > 0))
                for begin in range(0, len(ids), 512):
                    selected = ids[begin:begin+512]
                    entry = _ray_entry(directions[selected], origin, body)
                    closer = entry < first[selected]
                    first[selected[closer]] = entry[closer]
                    first_geom[selected[closer]] = body.geom_id
            shadow = np.isfinite(first) & (ranges > first+self.config.shadow_threshold_m)
            labels[outside[shadow]] = SHADOW
            geom_ids[outside[shadow]] = first_geom[shadow]
        removed = (labels == ROBOT) | (labels == PADDING)
        per_link_removed = dict.fromkeys((body.name for body in bodies), 0)
        for body in bodies:
            per_link_removed[body.name] += int((removed & (geom_ids == body.geom_id)).sum())
        statistics = {
            'algorithm': 'convex_model_contains_and_ray_shadow',
            'input_points': len(points), 'robot_points_removed': int(removed.sum()),
            'nominal_robot_points': int((labels == ROBOT).sum()),
            'padding_band_points': int((labels == PADDING).sum()),
            'shadow_points_retained': int((labels == SHADOW).sum()),
            'outside_points': int((labels == OUTSIDE).sum()),
            'protected_points': int((labels == PROTECTED).sum()),
            'invalid_points': int((labels == INVALID).sum()),
            'parameters': self.config.as_dict(), 'shadow_test_enabled': camera_origin is not None,
            'per_link_removed': per_link_removed,
            'labels': LABEL_NAMES,
            'limitation': 'Padding is a model uncertainty band: close obstacles in this band '
                          'can be removed. Convex bodies approximate concavities. '
                          'Shadow points are retained and never certify hidden free space.',
        }
        return SelfFilterResult(points, labels, removed, geom_ids, statistics)

"""Configuration for the observed belt region used by the single-pick planner."""
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

from .scene_region_filter import SceneRegionFilterConfig


DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/pick_belt_scene.yaml'


@dataclass(frozen=True)
class BeltSceneFilterConfig:
    enabled: bool = True
    xy_margin_m: float = .075
    z_bounds_m: tuple = (-.08, .25)
    voxel_size_m: float = .005

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        for name, lower, upper in [('xy_margin_m', 0., .2), ('voxel_size_m', .001, .005)]:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not lower <= value <= upper):
                raise ValueError(f'{name} must be a finite number between {lower} and {upper} metres')
            object.__setattr__(self, name, float(value))
        if not isinstance(self.z_bounds_m, (tuple, list)) or len(self.z_bounds_m) != 2:
            raise ValueError('z_bounds_m must be [lower, upper] in metres relative to the belt plane')
        if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v)
               for v in self.z_bounds_m):
            raise ValueError('z_bounds_m must contain two finite numbers')
        lower, upper = map(float, self.z_bounds_m)
        if not -.25 <= lower <= 0. < upper <= .5:
            raise ValueError('z_bounds_m must span the belt plane within [-0.25, 0.5] metres')
        object.__setattr__(self, 'z_bounds_m', (lower, upper))

    def as_dict(self):
        return {'enabled': self.enabled, 'xy_margin_m': self.xy_margin_m,
                'z_bounds_m': list(self.z_bounds_m), 'voxel_size_m': self.voxel_size_m}

    def resolve_region(self, belt_region=None):
        if not self.enabled:
            return SceneRegionFilterConfig(enabled=False, voxel_size_m=self.voxel_size_m), None
        if not isinstance(belt_region, dict):
            raise ValueError('传送带点云范围已启用，但缺少实测带面坐标和边界')
        try:
            # Scene coverage uses the visible belt outline when available;
            # old saved regions keep their original native-depth quantiles.
            bounds = np.asarray(belt_region.get('scene_bounds_xy_m', belt_region['bounds_xy_m']),
                                dtype=float)
            transform = np.asarray(belt_region['plane_to_base'], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('实测传送带范围须包含 plane_to_base 和 bounds_xy_m') from error
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])):
            raise ValueError('实测传送带 bounds_xy_m 须为两个有限有序区间')
        margin = self.xy_margin_m
        region = SceneRegionFilterConfig(
            enabled=True, x_bounds_m=(bounds[0, 0]-margin, bounds[0, 1]+margin),
            y_bounds_m=(bounds[1, 0]-margin, bounds[1, 1]+margin), z_bounds_m=self.z_bounds_m,
            voxel_size_m=self.voxel_size_m)
        return region, transform


def load_belt_scene_filter_config(path=None):
    path = DEFAULT_CONFIG if path is None else Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as error:
        raise ValueError(f'invalid belt-scene YAML: {path}') from error
    if not isinstance(document, dict):
        raise ValueError('belt-scene YAML must be a mapping')
    unknown = set(document)-{'enabled', 'xy_margin_m', 'z_bounds_m', 'voxel_size_m'}
    if unknown:
        raise ValueError(f'unknown belt-scene settings: {sorted(unknown, key=str)}')
    return BeltSceneFilterConfig(**document)

"""Identify known shapes without treating their CAD size as measured geometry.

Reference renders use the physical camera viewpoint, then both observations
and references are independently centred and normalized. All match distances
are dimensionless. A category supplies a label, never a grasp pose or size.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ..calibration.geometry import rigid_transform


@dataclass(frozen=True)
class ShapeMatchConfig:
    minimum_mask_pixels: int = 40
    minimum_valid_depth_fraction: float = .55
    normalized_image_size: int = 64
    yaw_step_deg: float = 30.
    maximum_shape_distance: float = 1.15
    minimum_score_gap: float = .06

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError('shape settings must be finite')
        for key in ('minimum_mask_pixels', 'normalized_image_size'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 8:
                raise ValueError(f'{key} must be an integer >= 8')
        if not 0 < self.minimum_valid_depth_fraction <= 1:
            raise ValueError('minimum_valid_depth_fraction must be in (0, 1]')
        if not 5 <= self.yaw_step_deg <= 90:
            raise ValueError('yaw_step_deg must be between 5 and 90')
        if self.maximum_shape_distance <= 0 or self.minimum_score_gap < 0:
            raise ValueError('invalid shape distance thresholds')


def _points(depth, k, transform):
    rows, columns = np.indices(depth.shape)
    optical = np.stack(((columns-k[0, 2])*depth/k[0, 0],
                        (rows-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    return optical @ transform[:3, :3].T + transform[:3, 3]


def _descriptor(mask, points, size):
    """Isotropically normalized silhouette, relative heights and proportions."""
    yy, xx = np.nonzero(mask)
    measured = points[mask]
    height = max(float(np.quantile(measured[:, 2], .95)), 1e-9)
    xy = measured[:, :2]
    _, axes = np.linalg.eigh(np.cov(xy.T))
    projected = xy @ axes
    extents = np.diff(np.quantile(projected, [.025, .975], axis=0), axis=0)[0]
    major, minor = np.maximum(np.sort(extents)[::-1], 1e-9)
    ratios = np.array([minor/major, height/major])
    crop_mask = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    crop_height = np.where(mask, points[..., 2]/height, 0)[yy.min():yy.max()+1, xx.min():xx.max()+1]
    factor = (size-8)/max(crop_mask.shape)
    width, height_pixels = max(1, round(crop_mask.shape[1]*factor)), max(1, round(crop_mask.shape[0]*factor))
    foreground = cv2.resize(crop_mask.astype(np.uint8), (width, height_pixels), interpolation=cv2.INTER_NEAREST)
    relative_heights = cv2.resize(crop_height.astype(np.float32), (width, height_pixels), interpolation=cv2.INTER_NEAREST)
    normalized = np.zeros((size, size), dtype=np.uint8)
    height_map = np.zeros((size, size), dtype=np.float32)
    x0, y0 = (size-width)//2, (size-height_pixels)//2
    normalized[y0:y0+height_pixels, x0:x0+width] = foreground
    height_map[y0:y0+height_pixels, x0:x0+width] = np.clip(relative_heights, 0, 1.3)
    edge = normalized & ~cv2.erode(normalized, np.ones((3, 3), np.uint8))
    distances = cv2.distanceTransform((1-edge).astype(np.uint8), cv2.DIST_L2, 3)/size
    return {'mask': normalized.astype(bool), 'height': height_map, 'edge': edge.astype(bool),
            'distance': distances, 'ratios': ratios,
            'height_quantiles': np.quantile(measured[:, 2]/height, [.25, .5, .75])}


def _distance(observed, reference):
    overlap = observed['mask'] & reference['mask']
    union = observed['mask'] | reference['mask']
    iou = float(overlap.sum()/max(1, union.sum()))
    chamfer = .5*(float(observed['distance'][reference['edge']].mean()) +
                  float(reference['distance'][observed['edge']].mean()))
    height_error = float(np.mean(np.abs(observed['height'][overlap]-reference['height'][overlap]))) if overlap.any() else 1.
    ratio_error = float(np.mean(np.abs(np.log(observed['ratios']/reference['ratios']))))
    profile_error = float(np.mean(np.abs(observed['height_quantiles']-reference['height_quantiles'])))
    score = 1.5*(1-iou) + 6*chamfer + .6*height_error + .8*ratio_error + .3*profile_error
    return {'score': float(score), 'shape_similarity': float(np.exp(-score)),
            'silhouette_iou': iou, 'normalized_contour_distance': chamfer,
            'relative_height_difference': height_error,
            'shape_proportion_difference': ratio_error}


class ShapeSimilarityMatcher:
    """Classify isolated upright parts by normalized shape, at arbitrary size.

    The supplied mask must be on the native depth grid. ``camera_to_plane_cv``
    maps optical coordinates to a support frame with plane Z=0. Reference
    scale only sets rendering resolution; it is neither bounded around CAD
    dimensions nor exposed as a fitted object size.
    """

    def __init__(self, specs, config=None):
        self.specs = tuple(specs)
        if not self.specs or len({s.category_id for s in self.specs}) != len(self.specs):
            raise ValueError('supply a nonempty shape catalog with unique category IDs')
        self.config = config or ShapeMatchConfig()

    def match(self, depth_m, intrinsics, camera_to_plane_cv, component_mask):
        depth = np.asarray(depth_m, dtype=float)
        mask = np.asarray(component_mask, dtype=bool)
        k = np.asarray(intrinsics, dtype=float)
        transform = rigid_transform(camera_to_plane_cv, 'camera_to_plane_cv')
        if depth.ndim != 2 or mask.shape != depth.shape:
            raise ValueError('depth and component mask must share the native HxW grid')
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-9):
            raise ValueError('intrinsics must be a finite zero-skew pinhole matrix')
        cfg = self.config
        result = {'accepted': False, 'category_id': -1, 'object_id': None,
                  'pose_object_to_plane': None, 'nominal_dimensions_m': None,
                  'fitted_dimensions_m': None, 'scale': None, 'candidates': [], 'reasons': [],
                  'method': 'scale_normalized_shape_similarity', 'recognition_mode': 'shape_similarity',
                  'parameters': asdict(cfg), 'candidate_catalog_size': len(self.specs),
                  'dimension_source': 'none; grasp geometry must be measured separately',
                  'motion_control': False}
        original_count = int(mask.sum())
        if original_count < cfg.minimum_mask_pixels:
            result['reasons'] = ['insufficient_shape_pixels']
            return result
        mask = mask & np.isfinite(depth) & (depth > 0)
        result['valid_depth_fraction'] = float(mask.sum()/original_count)
        if mask.sum() < cfg.minimum_mask_pixels or result['valid_depth_fraction'] < cfg.minimum_valid_depth_fraction:
            result['reasons'] = ['insufficient_valid_shape_depth']
            return result
        if np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
            result['reasons'] = ['shape_truncated_by_depth_image']
            return result
        safe_depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        points = _points(safe_depth, k, transform)
        mask &= np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-5)
        if mask.sum() < cfg.minimum_mask_pixels:
            result['reasons'] = ['insufficient_above_plane_shape']
            return result
        measured = points[mask]
        observed_height = float(np.quantile(measured[:, 2], .95))
        observed = _descriptor(mask, points, cfg.normalized_image_size)
        # The centre only chooses the physical viewpoint for rendering; image
        # centring happens independently, without fitting a grasp translation.
        center = np.mean(np.quantile(measured[:, :2], [.05, .95], axis=0), axis=0)
        minimum_reference_height = max(0., float(np.quantile(measured[:, 2], .01))*.75)
        yy, xx = np.nonzero(mask)
        padding = max(24, int(max(np.ptp(xx), np.ptp(yy))))
        x0, x1 = max(0, int(xx.min())-padding), min(depth.shape[1], int(xx.max())+padding+1)
        y0, y1 = max(0, int(yy.min())-padding), min(depth.shape[0], int(yy.max())+padding+1)
        crop_k = k.copy()
        crop_k[:2, 2] -= [x0, y0]
        plane_to_camera = np.linalg.inv(transform)
        from .cad_rendering import CadDepthRenderer
        rows = []
        with CadDepthRenderer(self.specs, crop_k, x1-x0, y1-y0) as renderer:
            for spec in self.specs:
                reference_scale = observed_height/float(spec.dimensions[2])
                symmetric = (spec.symmetry.get('kind') == 'continuous_rotational' and
                             np.allclose(spec.symmetry.get('axis', [0, 0, 1]), [0, 0, 1]))
                yaws = [0.] if symmetric else np.deg2rad(np.arange(0, 360, cfg.yaw_step_deg))
                choices = []
                def evaluate(yaw):
                    pose = np.eye(4)
                    c, s = np.cos(yaw), np.sin(yaw)
                    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                    pose[:2, 3] = center
                    rendered = renderer.render(spec.category_id, plane_to_camera @ pose, scale=reference_scale)
                    reference_points = _points(rendered, crop_k, transform)
                    foreground = (rendered > 0) & (reference_points[..., 2] > minimum_reference_height)
                    if foreground.sum() < 8:
                        return
                    reference = _descriptor(foreground, reference_points, cfg.normalized_image_size)
                    metrics = _distance(observed, reference)
                    choices.append((metrics['score'], float(yaw), metrics))
                for yaw in yaws:
                    evaluate(yaw)
                if choices and not symmetric:
                    best_yaw = min(choices, key=lambda value: value[0])[1]
                    for offset in [-cfg.yaw_step_deg/3, cfg.yaw_step_deg/3]:
                        evaluate(best_yaw+np.deg2rad(offset))
                if choices:
                    _, _, best = min(choices, key=lambda value: value[0])
                    rows.append({'category_id': int(spec.category_id), 'object_id': spec.object_id, **best})
        rows.sort(key=lambda row: row['score'])
        result['candidates'] = rows
        if not rows:
            result['reasons'] = ['no_visible_shape_references']
            return result
        gap = rows[1]['score']-rows[0]['score'] if len(rows) > 1 else None
        result['score_gap_to_second_category'] = gap
        if rows[0]['score'] > cfg.maximum_shape_distance:
            result['reasons'].append('shape_not_similar_to_catalog')
        if gap is not None and gap < cfg.minimum_score_gap:
            result['reasons'].append('ambiguous_shape_categories')
        result['accepted'] = not result['reasons']
        if result['accepted']:
            result['category_id'] = rows[0]['category_id']
            result['object_id'] = rows[0]['object_id']
        return result

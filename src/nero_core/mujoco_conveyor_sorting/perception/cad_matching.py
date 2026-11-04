"""Fit known upright CAD parts to depth from the calibrated physical viewpoint.

The support plane fixes roll/pitch/z; only XY, yaw and a bounded uniform scale
are fitted. This is a CPU diagnostic for separated, upright conveyor parts,
not an arbitrary 6-D pose estimator or a certified dimensional measurement.
No simulation state, object labels or robot interfaces are read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CadMatchConfig:
    # Initial engineering tolerances, to be calibrated on labelled real parts.
    surface_tolerance_m: float = .003
    scale_tolerance: float = .03
    minimum_height_m: float = .012
    minimum_mask_pixels: int = 70
    yaw_step_deg: float = 30.
    minimum_observed_inlier_fraction: float = .65
    minimum_model_inlier_fraction: float = .60
    minimum_valid_model_fraction: float = .60
    minimum_score_gap: float = .12
    maximum_score: float = 1.8
    refine_categories: int = 4
    refinement_evaluations: int = 180

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all():
            raise ValueError('CAD matching settings must be finite')
        if not .0001 <= self.surface_tolerance_m <= .02:
            raise ValueError('surface_tolerance_m must be between 0.1 and 20 mm')
        if not 0 <= self.scale_tolerance <= .15:
            raise ValueError('scale_tolerance must be between 0 and 0.15')
        if not .001 <= self.minimum_height_m <= .05:
            raise ValueError('minimum_height_m must be between 1 and 50 mm')
        if not 5 <= self.yaw_step_deg <= 90:
            raise ValueError('yaw_step_deg must be between 5 and 90')
        for name in ('minimum_observed_inlier_fraction', 'minimum_model_inlier_fraction',
                     'minimum_valid_model_fraction'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('minimum_mask_pixels', 'refine_categories', 'refinement_evaluations'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.minimum_score_gap < 0 or self.maximum_score <= 0:
            raise ValueError('invalid score thresholds')


def _rigid(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    r = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1]) or
            not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError(f'{name} must be a proper rigid transform')
    return value


def _pose(parameters):
    x, y, yaw, _ = parameters
    c, s = np.cos(yaw), np.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[:2, 3] = [x, y]
    return value


def _depth_points(depth, k, camera_to_plane):
    rows, columns = np.indices(depth.shape)
    points = np.stack(((columns-k[0, 2])*depth/k[0, 0],
                       (rows-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    return points @ camera_to_plane[:3, :3].T + camera_to_plane[:3, 3]


def depth_agreement(observed, mask, rendered, rendered_foreground, config):
    """Bidirectional depth support; missing sensor pixels are never inliers.

    Trim only the depth residual term. Report support against all component
    pixels and model pixels with valid observations; separately constrain the
    valid fraction of the entire model so a tiny patch cannot explain it.
    """
    observed = np.asarray(observed)
    mask = np.asarray(mask, dtype=bool)
    predicted = np.asarray(rendered_foreground, dtype=bool)
    valid = np.isfinite(observed) & (observed > 0)
    mask = mask & valid
    known_model = predicted & valid
    overlap = mask & predicted
    n_observed, n_model = int(mask.sum()), int(predicted.sum())
    n_known, n_overlap = int(known_model.sum()), int(overlap.sum())
    tolerance = config.surface_tolerance_m
    residuals = np.abs(observed[overlap]-rendered[overlap])
    inliers = residuals <= tolerance
    # A finite penalty keeps reports JSON-safe even for a disjoint candidate.
    trimmed = np.sort(np.minimum(residuals, 4*tolerance))
    trimmed = trimmed[:max(1, int(math.ceil(.8*len(trimmed))))]
    depth_loss = float(np.mean(trimmed)/tolerance) if len(trimmed) else 4.
    score = depth_loss + 1.5*(1-n_overlap/max(1, n_observed)) + 1.5*(1-n_overlap/max(1, n_known))
    return {
        'score': float(score), 'overlap_pixels': n_overlap,
        'observed_pixels': n_observed, 'rendered_pixels': n_model,
        'observed_inlier_fraction': float(inliers.sum()/max(1, n_observed)),
        'model_inlier_fraction': float(inliers.sum()/max(1, n_known)),
        'valid_model_fraction': float(n_known/max(1, n_model)),
        'median_depth_residual_m': float(np.median(residuals)) if len(residuals) else None,
        'p90_depth_residual_m': float(np.quantile(residuals, .9)) if len(residuals) else None,
        'trimmed_depth_residual_m': depth_loss*tolerance,
    }


class CadDepthMatcher:
    """Render each CAD at the actual K/T, then fit and reject ambiguous matches.

    ``component_mask`` is on the native depth grid. ``camera_to_plane_cv``
    maps CV optical coordinates to a support plane frame with z=0. CAD object
    origins must use the existing support-plane convention. Results keep CAD
    model dimensions distinct from visible depth extents.
    """
    def __init__(self, specs, config=None, *, renderer_factory=None):
        self.specs = tuple(specs)
        if not self.specs or len({s.category_id for s in self.specs}) != len(self.specs):
            raise ValueError('supply a nonempty CAD catalog with unique category IDs')
        self.config = config or CadMatchConfig()
        self.renderer_factory = renderer_factory

    def match(self, depth_m, intrinsics, camera_to_plane_cv, component_mask):
        started = time.perf_counter()
        result = self._match(depth_m, intrinsics, camera_to_plane_cv, component_mask)
        result['inference_seconds'] = time.perf_counter()-started
        return result

    def _match(self, depth_m, intrinsics, camera_to_plane_cv, component_mask):
        depth = np.asarray(depth_m, dtype=float)
        mask = np.asarray(component_mask, dtype=bool)
        k = np.asarray(intrinsics, dtype=float)
        transform = _rigid(camera_to_plane_cv, 'camera_to_plane_cv')
        if depth.ndim != 2 or mask.shape != depth.shape:
            raise ValueError('depth and component mask must share the native HxW grid')
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or
                k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or
                abs(k[0, 1])+abs(k[1, 0]) > 1e-9):
            raise ValueError('intrinsics must be a finite zero-skew pinhole matrix')
        cfg = self.config
        mask = mask & np.isfinite(depth) & (depth > 0)
        result = {'accepted': False, 'category_id': -1, 'object_id': None,
                  'pose_object_to_plane': None, 'scale': None,
                  'nominal_dimensions_m': None, 'fitted_dimensions_m': None,
                  'candidates': [], 'reasons': [], 'parameters': asdict(cfg),
                  'method': 'calibrated_view_cad_depth_registration',
                  'assumptions': 'isolated upright CAD part on measured support plane',
                  'dimension_source': 'bounded CAD fit; not independent dimensional metrology',
                  'motion_control': False}
        if mask.sum() < cfg.minimum_mask_pixels:
            result['reasons'] = ['insufficient_component_depth']
            return result
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            result['reasons'] = ['component_truncated_by_depth_image']
            return result
        points = _depth_points(depth, k, transform)
        mask &= points[..., 2] >= cfg.minimum_height_m
        if mask.sum() < cfg.minimum_mask_pixels:
            result['reasons'] = ['insufficient_above_plane_depth']
            return result
        measured = points[mask]
        height = float(np.quantile(measured[:, 2], .95))
        upper = measured[measured[:, 2] >= .7*height]
        center = np.mean(np.quantile(upper[:, :2], [.05, .95], axis=0), axis=0)
        result['observed_height_q95_m'] = height
        result['initial_center_plane_m'] = center.tolist()
        # Keep a generous fixed crop for the whole optimization; do not resize
        # depth or alter focal lengths. The crop only shifts the principal point.
        yy, xx = np.nonzero(mask)
        padding = max(24, int(np.ceil(.06*max(k[0, 0], k[1, 1])/np.median(depth[mask]))))
        x0, x1 = max(0, int(xx.min())-padding), min(depth.shape[1], int(xx.max())+padding+1)
        y0, y1 = max(0, int(yy.min())-padding), min(depth.shape[0], int(yy.max())+padding+1)
        observed, observed_mask = depth[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        crop_k = k.copy(); crop_k[0, 2] -= x0; crop_k[1, 2] -= y0
        result['crop_xyxy'] = [x0, y0, x1, y1]
        # Known upper surfaces must be compatible with CAD height plus print
        # tolerance. A mismatch is reported rather than scaling arbitrarily.
        plausible = [s for s in self.specs if abs(float(s.dimensions[2])-height) <=
                     float(s.dimensions[2])*cfg.scale_tolerance + max(.008, 3*cfg.surface_tolerance_m)]
        plausible_ids = {s.category_id for s in plausible}
        result['height_gated_category_ids'] = [int(s.category_id) for s in self.specs if s.category_id not in plausible_ids]
        if not plausible:
            result['reasons'] = ['no_cad_height_within_tolerance']
            return result
        factory = self.renderer_factory
        if factory is None:
            from .cad_rendering import CadDepthRenderer
            factory = CadDepthRenderer
        plane_to_camera = np.linalg.inv(transform)
        best_by_category = []
        with factory(plausible, crop_k, x1-x0, y1-y0) as renderer:
            def evaluate(spec, parameters, details=False):
                rendered = renderer.render(spec.category_id, plane_to_camera @ _pose(parameters), scale=float(parameters[3]))
                rendered_points = _depth_points(rendered, crop_k, transform)
                foreground = (rendered > 0) & (rendered_points[..., 2] >= cfg.minimum_height_m)
                metrics = depth_agreement(observed, observed_mask, rendered, foreground, cfg)
                # Prior only breaks close ties; scale is never used to rescale
                # the input camera depth or change its calibration.
                prior = .05*abs(float(parameters[3])-1)/max(cfg.scale_tolerance, 1e-9)
                metrics['score'] += prior
                if details:
                    metrics.update(category_id=int(spec.category_id), object_id=spec.object_id,
                                   scale=float(parameters[3]), yaw_rad=float(parameters[2]),
                                   pose_object_to_plane=_pose(parameters).tolist(),
                                   nominal_dimensions_m=spec.dimensions.tolist(),
                                   fitted_dimensions_m=(spec.dimensions*parameters[3]).tolist(),
                                   symmetry=dict(spec.symmetry))
                    return metrics
                return metrics['score']

            for spec in plausible:
                initial_scale = float(np.clip(height/spec.dimensions[2], 1-cfg.scale_tolerance, 1+cfg.scale_tolerance))
                symmetric = spec.symmetry.get('kind') == 'continuous_rotational'
                yaws = [0.] if symmetric else np.deg2rad(np.arange(0, 360, cfg.yaw_step_deg))
                choices = []
                for yaw in yaws:
                    parameters = np.r_[center, yaw, initial_scale]
                    choices.append((evaluate(spec, parameters), parameters))
                score, parameters = min(choices, key=lambda pair: pair[0])
                best_by_category.append((score, spec, parameters))
            best_by_category.sort(key=lambda value: value[0])
            rows = []
            for index, (score, spec, parameters) in enumerate(best_by_category):
                if index < cfg.refine_categories:
                    symmetric = spec.symmetry.get('kind') == 'continuous_rotational'
                    # Optimize millimetres/degrees/percent to balance parameters.
                    start = np.r_[parameters[:2]*1000, np.rad2deg(parameters[2]), parameters[3]*100]
                    bounds = [(center[0]*1000-20, center[0]*1000+20),
                              (center[1]*1000-20, center[1]*1000+20),
                              (start[2]-cfg.yaw_step_deg, start[2]+cfg.yaw_step_deg),
                              ((1-cfg.scale_tolerance)*100, (1+cfg.scale_tolerance)*100)]
                    active = [0, 1] + ([] if symmetric else [2]) + ([] if cfg.scale_tolerance == 0 else [3])
                    def unpack(value):
                        scaled = start.copy(); scaled[active] = value
                        return np.r_[scaled[:2]/1000, np.deg2rad(scaled[2]), scaled[3]/100]
                    opt = minimize(lambda value: evaluate(spec, unpack(value)), start[active],
                                   method='Powell', bounds=[bounds[i] for i in active],
                                   options={'maxfev': cfg.refinement_evaluations, 'xtol': .04, 'ftol': .002})
                    refined = unpack(opt.x)
                    if evaluate(spec, refined) < score:
                        parameters = refined
                row = evaluate(spec, parameters, details=True)
                row['refined'] = index < cfg.refine_categories
                rows.append(row)
        rows.sort(key=lambda row: row['score'])
        result['candidates'] = rows
        best = rows[0]
        gap = rows[1]['score']-best['score'] if len(rows) > 1 else None
        result['score_gap_to_second_category'] = gap
        result['candidate_catalog_size'] = len(self.specs)
        reasons = []
        for key, threshold in [('observed_inlier_fraction', cfg.minimum_observed_inlier_fraction),
                               ('model_inlier_fraction', cfg.minimum_model_inlier_fraction),
                               ('valid_model_fraction', cfg.minimum_valid_model_fraction)]:
            if best[key] < threshold:
                reasons.append('insufficient_'+key)
        if best['score'] > cfg.maximum_score:
            reasons.append('depth_fit_residual_too_large')
        if gap is not None and gap < cfg.minimum_score_gap:
            reasons.append('ambiguous_cad_categories')
        result['reasons'] = reasons
        result['accepted'] = not reasons
        if result['accepted']:
            for key in ('category_id', 'object_id', 'pose_object_to_plane', 'scale',
                        'nominal_dimensions_m', 'fitted_dimensions_m'):
                result[key] = best[key]
        return result

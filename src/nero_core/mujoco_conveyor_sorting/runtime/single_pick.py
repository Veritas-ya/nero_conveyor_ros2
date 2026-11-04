"""One stationary-belt pick from measured current state; no hardware imports.

This is a supervised prototype, not the production sorting controller. Geometry
comes from the existing NERO model and live RGB-D; unseen obstacles and physical
TCP error cannot be ruled out by these checks. No home pose is used.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import os
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from .feedback_clock import mapped_stamp
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares

from ..calibration.geometry import inverse, pose_xyz_rpy, transform_distance
from ..controllers.ik_solver import DampedLeastSquaresIK, IKConfig, IKResult
from ..controllers.pick_trajectory import SmoothPickTrajectory
from ..perception import DepthGeometryVisionPipeline, GeometryPrototype, deproject_depth_to_world
from .self_filter import ConvexBody, ModelSelfFilter, load_self_filter_config
from .scene_noise_filter import filter_scene_noise, load_scene_noise_filter_config
from .scene_depth_filter import filter_depth_edge_tails
from .scene_saturation_filter import SceneSaturationFilterConfig, filter_scene_saturation
from .scene_stereo_filter import SceneStereoFilterConfig, filter_scene_stereo
from .scene_region_filter import filter_scene_region, voxel_downsample
from .belt_scene_config import load_belt_scene_filter_config
from .belt_surface import BeltSurface, BeltSurfaceNotReady


class PickRejected(RuntimeError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {} if details is None else details


class TargetNotReady(PickRejected):
    """This image has no usable single target; a warmed acquisition may retry."""
    def __init__(self, message, *, reason='target_geometry', details=None):
        super().__init__(message, details={**(details or {}), 'reason': reason})


@dataclass(frozen=True)
class PickSettings:
    lift_m: float = .03
    speed_percent: int = 3
    force_n: float = 2.
    open_width_m: float = .10
    min_force_n: float = .2
    move_timeout_s: float = 12.
    control_hz: int = 100
    initial_joint_tolerance_deg: float | None = None
    query_controller_limits: bool = False
    high_joint_stream_enabled: bool = True
    startup_gripper_force_threshold_n: float = .2
    planning_mode: str = 'legacy'
    moving_approach_mode: str = 'high'
    selection_policy: str = 'default'
    replan_policy: str = 'legacy'
    full_plan_close_window_s: float = .6
    full_plan_observation_max_age_s: float = 2.0
    full_plan_compute_budget_s: float = 2.0
    planning_backend: str = 'auto'
    scene_collision_backend: str = 'auto'
    warm_roadmap_enabled: bool = True
    shared_state_evaluation: bool = True
    minimal_state_evaluation: bool = True
    adaptive_segment_validation: bool = True
    simd_broadphase_enabled: bool = True
    joint_trajectory_backend: str = 'quintic'
    joint_jerk_limit_rad_s3: float | None = None
    joint_response_model_path: str | None = None

    def validate(self):
        values = [value for name, value in vars(self).items()
                  if name not in {'planning_backend', 'scene_collision_backend',
                                  'joint_trajectory_backend', 'joint_response_model_path',
                                  'planning_mode', 'selection_policy', 'replan_policy', 'moving_approach_mode'}
                  and not (name in {'initial_joint_tolerance_deg', 'joint_jerk_limit_rad_s3'} and value is None)]
        if not np.isfinite(values).all():
            raise ValueError('参数必须为有限数值')
        if not .01 <= self.lift_m <= .05 or not isinstance(self.speed_percent, int) or not 1 <= self.speed_percent <= 100:
            raise ValueError('抬升须为10–50 mm，速度须为1–100%')
        if not .5 <= self.force_n <= 5 or not 0 < self.min_force_n < self.force_n:
            raise ValueError('夹持力须为0.5–5 N，接触门槛须小于夹持力')
        if self.startup_gripper_force_threshold_n <= 0:
            raise ValueError('启动夹爪受力阈值必须为正数')
        if self.open_width_m != .10 or not 3 <= self.move_timeout_s <= 30:
            raise ValueError('本入口仅支持原装100 mm开口夹爪及3–30秒等待')
        if not isinstance(self.control_hz, int) or not 50 <= self.control_hz <= 100:
            raise ValueError('轨迹发送频率须为50–100 Hz的整数')
        if self.initial_joint_tolerance_deg is not None and self.initial_joint_tolerance_deg < 0:
            raise ValueError('当前位姿关节容差须非负；不指定时按实测起点适配本次路径范围')
        if not isinstance(self.query_controller_limits, bool):
            raise ValueError('控制器限速查询开关须为布尔值')
        if not isinstance(self.high_joint_stream_enabled, bool):
            raise ValueError('高位轨迹独立发送开关须为布尔值')
        for name in ('warm_roadmap_enabled', 'shared_state_evaluation',
                     'minimal_state_evaluation', 'adaptive_segment_validation',
                     'simd_broadphase_enabled'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name}须为布尔值')
        from ..controllers.full_pick_execution import validate_full_pick_settings
        validate_full_pick_settings(self)
        if self.moving_approach_mode not in {'high', 'continuous'}:
            raise ValueError('接近方式须为high或continuous')
        if self.planning_backend not in {'auto', 'cpu', 'curobo'}:
            raise ValueError('规划后端须为auto、cpu或curobo')
        if self.scene_collision_backend not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('场景查询后端须为auto、cpu或cuda')
        if self.joint_trajectory_backend not in {'quintic', 'ruckig'}:
            raise ValueError('关节时间律须为quintic或ruckig')
        if self.joint_jerk_limit_rad_s3 is not None and self.joint_jerk_limit_rad_s3 <= 0:
            raise ValueError('显式上位机jerk须为正有限数')

    @property
    def joint_velocity_limit(self):
        return .02*self.speed_percent

    @property
    def joint_acceleration_limit(self):
        # Preserve the original standalone PickCycle profile. The complete
        # shared policy uses the per-joint simulation profile below instead.
        return min(.05*self.speed_percent, 2.)

    @property
    def full_policy_joint_acceleration_limits(self):
        # Match the successful simulation's host trajectory/controller limits
        # at 100%. This does not change or measure the firmware acceleration.
        return tuple(value * self.speed_percent / 100.
                     for value in (8., 8., 9., 9., 10., 10., 12.))


def build_stage_trajectory(start_joints, stage, cfg):
    """Reconstruct a timed curve; collision validation belongs to planning."""
    if not stage['joints_rad']:
        raise PickRejected('运动阶段没有规划路径点')
    trajectory = SmoothPickTrajectory(
        [list(start_joints), *stage['joints_rad']],
        cfg.joint_velocity_limit, cfg.joint_acceleration_limit)
    samples = trajectory.verification_samples(max_joint_step=.004)
    return trajectory, samples


def stage_trajectory_key(start_joints, stage, cfg):
    """Cache by actual nodes and limits, never by a stage name alone."""
    nodes = np.asarray([list(start_joints), *stage['joints_rad']], dtype=np.float64)
    return (stage['name'], nodes.shape, nodes.tobytes(),
            cfg.joint_velocity_limit, cfg.joint_acceleration_limit)


def prepare_stage_trajectory(model, start_joints, stage, cfg):
    """Validate the continuous curve once, while generating the plan."""
    trajectory, samples = build_stage_trajectory(start_joints, stage, cfg)
    contact = stage['name'] not in ('raise_current', 'above_object')
    # Cubic blending may depart from the original chords: validate this curve
    # itself against the unchanged model, target and observed scene margins.
    for a, b in zip(samples[:-1], samples[1:]):
        model.check_edge(a, b, cfg.open_width_m, allow_finger_contact=contact)
    if len(samples) == 1:
        model.check_edge(samples[0], samples[0], cfg.open_width_m,
                         allow_finger_contact=contact)
    cache = getattr(model, '_pick_trajectory_cache', None)
    if cache is not None:
        retained = samples.copy(); retained.setflags(write=False)
        cache[stage_trajectory_key(start_joints, stage, cfg)] = (trajectory, retained)
    return trajectory, samples


def frame_points(frame):
    if np.any(np.abs(frame.depth_intrinsics.distortion) > 1e-10):
        raise PickRejected('现有深度反投影只支持零畸变深度流，不能直接忽略非零D')
    return deproject_depth_to_world(frame.depth_m, frame.depth_intrinsics.matrix,
                                   frame.camera_to_world_cv, camera_convention='opencv')


def detect_target(frame, depth_to_color, *, fit_plane, green_belt_points, diagnostic=None,
                  target_shape='geometry', shape_recognizer=None,
                  robot_model=None, robot_state=None, maximum_target_height_m=None,
                  measured_surface=None, scene_scope='observed_scene'):
    """Reuse the existing foreground component detector; do not guess a class.

    A measured belt search region is reused within the same camera session;
    simple mode also excludes green foreground. Every exposure still supplies
    its own native metric depth, transformed into the measured plane frame.
    """
    surface = (measured_surface if measured_surface is not None
               else getattr(robot_model, 'measured_belt_surface', None))
    cached_surface = isinstance(surface, BeltSurface) and surface.matches_frame(frame, depth_to_color)
    if not cached_surface:
        try:
            surface = BeltSurface.from_frame(
                frame, depth_to_color, fit_plane=fit_plane,
                green_belt_points=green_belt_points)
        except BeltSurfaceNotReady as error:
            raise TargetNotReady(str(error), reason='belt_surface_unavailable',
                                 details={'belt_surface_reason': error.reason,
                                          **error.details}) from error
    plane = surface.plane
    if plane['inlier_rms_m'] > .003 or plane['tilt_from_base_xy_deg'] > 8:
        raise PickRejected('带面拟合噪声/倾斜过大，检查相机外参及传送带')
    if robot_model is not None and not cached_surface:
        robot_model.measured_belt_surface = surface
    # Reuse measured static geometry only. The workpiece and all scene points
    # continue to come from this exposure, never from a projected plane image.
    from .frame_products import frame_products, world_points
    products = frame_products(frame, depth_to_color)
    points = world_points(frame, depth_to_color)
    n = np.asarray(plane['normal_base'])
    plane_to_base = surface.plane_to_base
    base_to_plane = inverse(plane_to_base)
    flat = products.get(('belt_coordinates', plane_to_base.tobytes()), lambda:
        points @ base_to_plane[:3, :3].T + base_to_plane[:3, 3])
    hull = surface.hull_xy
    lower, upper = surface.bounds_xy_m.T
    belt_region = surface.region_metadata()
    belt_region['reused_measured_surface'] = bool(cached_surface)
    maximum_height = .12 if maximum_target_height_m is None else float(maximum_target_height_m)
    if not np.isfinite(maximum_height) or maximum_height <= .012:
        raise ValueError('目标检测高度上限须高于原12mm检测下限')
    pipeline = DepthGeometryVisionPipeline(
        [GeometryPrototype(category_id=0, dimensions=(.05, .05, .05), symmetric=True)],
        support_plane_height=0, roi_x=[lower[0], upper[0]], roi_y=[lower[1], upper[1]],
        minimum_height=.012, maximum_height=maximum_height, minimum_mask_pixels=70,
        minimum_footprint=.012, maximum_footprint=.16)
    clean_mask = None
    rejected_target_depth = np.zeros(frame.depth_m.shape, dtype=bool)
    target_filter = None
    target_self_filter = None
    if target_shape == 'simple' or robot_model is not None:
        # Evaluate the unchanged foreground predicates once per exposure.
        # Scalar axis comparisons avoid allocating HxWx2 temporary arrays.
        finite_flat = np.isfinite(flat).all(axis=2)
        foreground_height = ((frame.depth_m > 0) & finite_flat
                      & (flat[..., 2] >= pipeline.minimum_height)
                      & (flat[..., 2] <= pipeline.maximum_height))
        in_search_xy = ((flat[..., 0] >= lower[0]) & (flat[..., 0] <= upper[0])
                        & (flat[..., 1] >= lower[1]) & (flat[..., 1] <= upper[1]))
        foreground = foreground_height & in_search_xy
    if robot_model is not None:
        # Use the exposure-paired state, never the model's last planning pose.
        # Mask only target segmentation; preserve native RGB-D and scene points.
        classified = robot_model.classify_robot_points(
            points[foreground], robot_state, camera_origin=frame.camera_to_world_cv[:3, 3])
        clean_mask = foreground.copy()
        clean_mask[foreground] = ~classified.removed_mask
        foreground = clean_mask
        target_self_filter = {
            **classified.statistics, 'scope': 'target_foreground_before_components',
            'state_source': 'capture_paired_robot_state', 'depth_modified': False}
        if diagnostic is not None:
            original_diagnostic = diagnostic
            diagnostic = lambda data: original_diagnostic(
                {**data, 'target_self_filter': target_self_filter})
    if target_shape == 'simple':
        # The moving GraspGen-X entry uses this mode. Filter BEFORE component
        # counting so belt-coloured depth islands cannot become extra parts.
        from .shape_pick import filter_target_mask
        clean_mask, target_filter = filter_target_mask(
            frame, depth_to_color, foreground, flat,
            base_to_plane @ frame.camera_to_world_cv)
        rejected_target_depth = foreground & ~clean_mask
        # The green-belt quantiles select where to find a part, not where
        # that part ends. Recover only complete foreground components that
        # contain a supported search seed; unrelated outside components stay
        # in the scene. Do not run the metric density test on the whole room.
        if clean_mask.any():
            from .shape_pick import exclude_green_belt
            complete_foreground = foreground_height.copy()
            complete_foreground = exclude_green_belt(frame, depth_to_color, complete_foreground)
            _, full_labels = cv2.connectedComponents(
                complete_foreground.astype(np.uint8), connectivity=8)
            seed_labels = np.unique(full_labels[clean_mask])
            seed_labels = seed_labels[seed_labels != 0]
            complete = np.isin(full_labels, seed_labels) & complete_foreground
            added = complete & ~foreground
            if added.any():
                if robot_model is not None:
                    extension_self = robot_model.classify_robot_points(
                        points[added], robot_state,
                        camera_origin=frame.camera_to_world_cv[:3, 3])
                    complete[added] = ~extension_self.removed_mask
                    target_filter['extension_self_filter'] = extension_self.statistics
                    # Removing a robot bridge may disconnect other surfaces.
                    _, full_labels = cv2.connectedComponents(
                        complete.astype(np.uint8), connectivity=8)
                    seed_labels = np.unique(full_labels[clean_mask])
                    complete &= np.isin(full_labels, seed_labels[seed_labels != 0])
                extended_mask, extension_filter = filter_target_mask(
                    frame, depth_to_color, complete, flat,
                    base_to_plane @ frame.camera_to_world_cv)
                target_filter['complete_component_filter'] = extension_filter
                target_filter['search_seed_points'] = int(clean_mask.sum())
                target_filter['initial_filter_scope'] = 'search_roi_seed'
                rejected_target_depth |= complete & ~extended_mask
                foreground |= complete
                clean_mask = extended_mask
    components = pipeline.extract_components(None, frame.depth_m, frame.depth_intrinsics.matrix,
                                             base_to_plane @ frame.camera_to_world_cv,
                                             camera_convention='opencv', foreground_mask=clean_mask,
                                             roi_selects_components=target_shape == 'simple',
                                             points_world=flat)
    candidates = [c for c in components if cv2.pointPolygonTest(
        hull, tuple(map(float, c.position[:2])), True) > .025]
    if len(candidates) != 1:
        if diagnostic is not None:
            mask = np.logical_or.reduce([c.mask for c in candidates]) if candidates else np.zeros_like(frame.depth_m, dtype=bool)
            diagnostic({'object_points': points[mask], 'mask': mask, 'plane': plane,
                        'candidate_masks': [c.mask for c in candidates],
                        'candidate_count': len(candidates), 'target_cloud_filter': target_filter})
        raise TargetNotReady(f'带面内检测到{len(candidates)}个候选，需保留一个独立工件',
                             reason='no_workpiece' if not candidates else 'multiple_candidates',
                             details={'candidate_count': len(candidates), 'target_cloud_filter': target_filter,
                                      'target_self_filter': target_self_filter})
    c = candidates[0]
    if target_filter is not None:
        # Only rejected depth belonging to THIS target's original component
        # is removed from collision input. Other components remain obstacles,
        # even if their colour/density disqualified them as a grasp target.
        _, original_labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
        label = original_labels.flat[np.flatnonzero(c.mask)[0]]
        rejected_target_depth &= original_labels == label
        target_filter['selected_target_points'] = int(np.count_nonzero(c.mask))
        target_filter['scene_rejected_target_depth_points'] = int(np.count_nonzero(rejected_target_depth))
        target_filter['roi_selects_complete_components'] = True
        target_filter['selected_points_outside_search_roi'] = int(np.count_nonzero(
            c.mask & ~in_search_xy))
    if target_shape not in ('shape', 'simple', 'geometry', 'auto', 'cylinder'):
        raise ValueError('target_shape must be shape, simple, geometry, auto or cylinder')
    if target_shape == 'shape':
        from .shape_pick import shape_observation, ShapeGeometryNotReady
        try:
            observation = shape_observation(
                frame, depth_to_color, c, points, flat,
                plane_to_base, base_to_plane @ frame.camera_to_world_cv,
                plane, hull, recognizer=shape_recognizer, diagnostic=diagnostic)
            observation['belt_region'] = belt_region
            observation['target_self_filter'] = target_self_filter
            return observation
        except ShapeGeometryNotReady as error:
            raise TargetNotReady(str(error), reason='shape_geometry') from error
        except ValueError as error:
            raise PickRejected(str(error)) from error
    object_points = points[c.mask]
    heights = flat[c.mask, 2]
    height = float(np.quantile(heights, .98))
    extent = np.asarray(c.feature_vector[:2])
    # Compare the same horizontal axes at different heights. A large extra
    # span confined to low boundary points is NOT evidence of a long object;
    # it can reflect mixed depths or an unmodelled varying cross-section.
    median_depth = float(np.median(frame.depth_m[c.mask]))
    pixel_allowance = median_depth/frame.depth_intrinsics.fx + median_depth/frame.depth_intrinsics.fy
    yaw_pca = float(c.raw_yaw)
    axes = np.array([[np.cos(yaw_pca), -np.sin(yaw_pca)],
                     [np.sin(yaw_pca), np.cos(yaw_pca)]])
    projections = flat[c.mask, :2] @ axes
    q = pipeline.robust_extent_quantile
    full_span = np.diff(np.quantile(projections, [q, 1-q], axis=0), axis=0)[0] + pixel_allowance
    upper_mask = heights >= .75*height
    shape = {'full_span_along_pca_axes_m': full_span.tolist(),
             'upper_slice_min_height_m': .75*height,
             'upper_slice_points': int(upper_mask.sum()),
             'status': 'insufficient_upper_points',
             'note': 'depth-contour consistency only; not a shape classification or measured diameter'}
    inconsistent = False
    if target_shape == 'simple':
        shape = {'status': 'not_required_in_simple_mode'}
    elif upper_mask.sum() >= pipeline.minimum_mask_pixels:
        upper_span = np.diff(np.quantile(projections[upper_mask], [q, 1-q], axis=0), axis=0)[0] + pixel_allowance
        inconsistent = bool(np.any((full_span-upper_span > .020) & (full_span > 1.30*upper_span)))
        shape.update(upper_span_along_same_axes_m=upper_span.tolist(),
                     status='inconsistent' if inconsistent else 'no_large_boundary_extension')
    cylinder = {'accepted': False, 'reasons': ['not_requested']}
    if target_shape in ('auto', 'cylinder'):
        from .depth_shape import fit_upright_cylinder
        # Depth geometry supplies dimensions; RGB only rejects points that
        # project onto the GREEN belt instead of the current visible part.
        to_depth = inverse(frame.camera_to_world_cv)
        depth_pts = object_points @ to_depth[:3, :3].T + to_depth[:3, 3]
        color_pts = depth_pts @ depth_to_color[:3, :3].T + depth_to_color[:3, 3]
        uv, _ = cv2.projectPoints(color_pts, np.zeros(3), np.zeros(3),
                                  frame.color_intrinsics.matrix, np.asarray(frame.color_intrinsics.distortion))
        uv = uv.reshape(-1, 2)
        pix = np.rint(np.nan_to_num(np.clip(uv, -1, max(frame.rgb.shape[:2])), nan=-1)).astype(int)
        inside = (color_pts[:, 2] > 0) & np.isfinite(uv).all(axis=1)
        inside &= (pix[:, 0] >= 0) & (pix[:, 0] < frame.rgb.shape[1])
        inside &= (pix[:, 1] >= 0) & (pix[:, 1] < frame.rgb.shape[0])
        keep = inside.copy()
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        green_rgb = cv2.inRange(hsv, np.array([35, 40, 35]), np.array([95, 255, 255])) > 0
        keep[inside] &= ~green_rgb[pix[inside, 1], pix[inside, 0]]
        cylinder = fit_upright_cylinder(flat[c.mask][keep], pixel_size_m=pixel_allowance/2)
        cylinder.update(rgb_depth_points_retained=int(keep.sum()),
                        rgb_depth_points_rejected=int((~keep).sum()),
                        rgb_check='project native depth through factory D2C; reject green belt pixels')
    if diagnostic is not None:
        diagnostic({'object_points': object_points, 'mask': c.mask, 'plane': plane,
                    'height': height, 'footprint_m': extent.tolist(), 'shape_consistency': shape,
                    'cylinder_fit': cylinder, 'target_cloud_filter': target_filter})
    if target_shape == 'cylinder' and not cylinder['accepted']:
        raise TargetNotReady('深度圆柱拟合未通过：'+', '.join(cylinder['reasons']),
                             reason='cylinder_fit')
    if inconsistent and not cylinder['accepted']:
        raise TargetNotReady(f'深度轮廓随高度明显变化：全轮廓{np.round(full_span*1000,1)} mm，'
                             f'上层{np.round(upper_span*1000,1)} mm；请核对保存的目标叠加图，不能将全轮廓直接当实物尺寸',
                             reason='inconsistent_contour')
    # The entry is intentionally limited to small, upright parts. No T-LESS
    # identity or hidden dimensions are invented from the prototype above.
    # Long dimension is independent of jaw opening. Opening is checked below
    # along the chosen TCP Y axis, not by applying one bound to both dimensions.
    if not .035 <= height <= .11 or not .015 <= min(extent) <= max(extent) <= pipeline.maximum_footprint:
        raise TargetNotReady(f'工件尺寸不适合本测试：高{height*1000:.1f} mm，外形{extent*1000} mm',
                             reason='target_size', details={'height_m': height,
                                 'footprint_m': extent.tolist(), 'height_range_m': [.035, .11]})
    center_plane = np.asarray(cylinder['center_xy_m']) if cylinder['accepted'] else c.position[:2]
    # Simple test: use the upper depth points for location, so low boundary
    # tails do not shift the target. No circle fit or shape-quality gate.
    # Keep the COMPLETE component for collision checking below.
    localization_points = object_points
    localization_confidence = float(c.orientation_confidence)
    localization_yaw = float(c.raw_yaw)
    if target_shape == 'simple':
        select = heights >= .60*height
        if select.sum() >= 20:
            localization_points = object_points[select]
            xy = flat[c.mask][select, :2]
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
            projected = xy @ eigenvectors
            lo, hi = np.quantile(projected, [q, 1-q], axis=0)
            center_plane = (.5*(lo+hi)) @ eigenvectors.T
            localization_confidence = float((eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(), 1e-12))
            localization_yaw = float(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    if target_shape != 'simple' and cv2.pointPolygonTest(hull, tuple(map(float, center_plane)), True) < .06:
        raise TargetNotReady('工件离带面边缘太近，原装张开夹爪的边缘余量不足', reason='target_edge')
    center = plane_to_base[:3, :3] @ np.r_[center_plane, 0.] + plane_to_base[:3, 3]
    z = -n
    # Near-circular parts have no observable yaw: choose fixed base-projected X.
    yaw = localization_yaw if localization_confidence >= .2 and not cylinder['accepted'] else 0.
    x = plane_to_base[:3, :3] @ np.array([np.cos(yaw), np.sin(yaw), 0])
    rotation = np.column_stack((x, np.cross(z, x), z))
    # Stock fingers open along local +/-Y. For low-confidence yaw (fixed X)
    # or unusual shapes, the PCA short dimension is not necessarily jaw width.
    closing_projection = localization_points @ rotation[:, 1]
    lower_width, upper_width = np.quantile(closing_projection, [q, 1-q])
    grasp_width = float(upper_width-lower_width+pixel_allowance)
    if cylinder['accepted']:
        grasp_width = float(cylinder['diameter_m']+cylinder['measurement_margin_m'])
    opening = PickSettings().open_width_m
    side_clearance = .010
    if grasp_width + 2*side_clearance > opening:
        raise TargetNotReady(f'实际夹持方向宽{grasp_width*1000:.1f} mm，'
                             f'夹爪开口{opening*1000:.0f} mm，需每侧保留{side_clearance*1000:.0f} mm余量',
                             reason='grasp_width')
    grasp = np.eye(4); grasp[:3, :3] = rotation
    # Nominal TCP is 10 mm behind fingertips. Keep tips >=10 mm above belt.
    grasp[:3, 3] = center + n * max(.02, height-.015)
    target_mask = c.mask
    visible_rgb_mask = None
    visible_completion = None
    if target_shape == 'simple':
        from .visible_target import selected_rgb_silhouette, visible_depth_mask
        visible_rgb_mask, visible_completion = selected_rgb_silhouette(frame, depth_to_color, c.mask)
        if visible_rgb_mask is not None:
            complete, lower_support = visible_depth_mask(
                frame, depth_to_color, visible_rgb_mask, flat, plane)
            complete, support_filter = filter_target_mask(
                frame, depth_to_color, complete, flat, base_to_plane @ frame.camera_to_world_cv)
            added = complete & ~c.mask
            if robot_model is not None and added.any():
                self_check = robot_model.classify_robot_points(points[added], robot_state,
                    camera_origin=frame.camera_to_world_cv[:3, 3])
                complete[added] = ~self_check.removed_mask
                visible_completion['extension_self_filter'] = self_check.statistics
            target_mask = c.mask | complete
            visible_completion.update(
                seed_depth_pixels=int(c.mask.sum()), completed_native_pixels=int(target_mask.sum()),
                added_native_pixels=int(np.count_nonzero(target_mask & ~c.mask)),
                lower_support_height_m=lower_support, support_filter=support_filter,
                applies_to='visible_geometry_after_original_target_selection',
                original_foreground_selection_height_m=pipeline.minimum_height)
            object_points = points[target_mask]
            if target_filter is not None:
                target_filter['visible_target_completion'] = visible_completion
            if diagnostic is not None:
                diagnostic({'object_points': object_points, 'mask': target_mask, 'plane': plane,
                    'height': height, 'footprint_m': extent.tolist(), 'shape_consistency': shape,
                    'cylinder_fit': cylinder, 'target_cloud_filter': target_filter})
    valid = (frame.depth_m > 0) & np.isfinite(points).all(axis=2)
    scene_mask = valid & ~target_mask & ~rejected_target_depth
    scene_input_count = int(np.count_nonzero(scene_mask))
    if scene_scope == 'measured_belt_surface_only':
        # Conservative broad phase in the already-computed plane grid. The
        # planner still performs its exact, same-frame belt predicate before
        # publishing the voxel cloud and KD tree. The epsilon covers float
        # roundoff between the two algebraically equivalent transforms.
        scene_bounds = np.asarray(belt_region.get('scene_bounds_xy_m', belt_region['bounds_xy_m']))
        epsilon = 1e-8
        scene_mask &= (np.abs(flat[..., 2]) <= .005 + epsilon)
        scene_mask &= (flat[..., 0] >= scene_bounds[0, 0]-epsilon) & (flat[..., 0] <= scene_bounds[0, 1]+epsilon)
        scene_mask &= (flat[..., 1] >= scene_bounds[1, 0]-epsilon) & (flat[..., 1] <= scene_bounds[1, 1]+epsilon)
    elif scene_scope != 'observed_scene':
        raise ValueError('unknown scene scope')
    scene_pixel_indices = np.flatnonzero(scene_mask)
    return {'grasp': grasp, 'center': center, 'height': height, 'width': grasp_width,
            'footprint_m': extent.tolist(), 'open_clearance_per_side_m': (opening-grasp_width)/2,
            'shape_consistency': shape,
            'cylinder_fit': cylinder,
            'target_geometry_source': ('upper_depth_geometry' if target_shape == 'simple' else
                                       'depth_cylinder_candidate' if cylinder['accepted'] else 'depth_foreground_envelope'),
            'localization_points': len(localization_points),
            'plane': plane, 'belt_region': belt_region, 'mask': target_mask, 'object_points': object_points,
            'scene_points': points[scene_mask], 'scene_prefilter_scope': scene_scope,
            'scene_input_count_before_prefilter': scene_input_count, 'points': points,
            'points_plane': flat, '_frame_products': products,
            'scene_pixel_indices': scene_pixel_indices,
            'visible_target_rgb_mask': visible_rgb_mask, 'visible_target_completion': visible_completion,
            'target_cloud_filter': target_filter,
            'target_self_filter': target_self_filter,
            'camera_origin_base_m': frame.camera_to_world_cv[:3, 3].copy(),
            'orientation_confidence': 0. if cylinder['accepted'] else localization_confidence,
            'category': 'unknown; geometry-only single-part pick'}


def stable_target(observations):
    if len(observations) < 3:
        raise PickRejected('需要至少三次独立的相机观测')
    centers = np.array([s['center'] for s in observations])
    heights = np.array([s['height'] for s in observations])
    widths = np.array([s['width'] for s in observations], dtype=float)
    measurements = [s.get('grasp_measurement', {}) for s in observations]
    measured_contact_bands = all(isinstance(item, dict) and
                                item.get('source') == 'measured_depth_contact_band'
                                for item in measurements)
    core_widths = []; invalid_core = []
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, dict) or 'stability_width_m' not in measurement:
            core_widths.append(None)
            continue
        value = measurement['stability_width_m']
        valid = (isinstance(value, (int, float, np.integer, np.floating)) and
                 not isinstance(value, (bool, np.bool_)) and
                 np.ndim(value) == 0 and np.isfinite(float(value)) and value > 0)
        core_widths.append(float(value) if valid else None)
        if not valid and measurement.get('source') == 'measured_depth_contact_band':
            invalid_core.append(index)
    complete_core = measured_contact_bands and all(value is not None for value in core_widths)
    core_span = float(np.ptp(core_widths)) if complete_core else None
    width_span = core_span if complete_core else float(np.ptp(widths))
    center_change = float(np.max(np.linalg.norm(centers-centers[-1], axis=1)))
    height_span = float(np.ptp(heights))
    stability = {'kind': 'geometry_stability',
                 'selected_width_metric': 'stability_width_m' if complete_core else 'width',
                 'grasp_widths_m': widths.tolist(), 'grasp_width_span_m': float(np.ptp(widths)),
                 'stability_widths_m': core_widths, 'stability_width_span_m': core_span,
                 'width_span_m': width_span, 'width_span_limit_m': .004,
                 'center_max_displacement_m': center_change, 'center_limit_m': .003,
                 'height_span_m': height_span, 'height_span_limit_m': .004,
                 'direction_max_delta_deg': None, 'direction_limit_deg': 5.,
                 'invalid_stability_width_indices': invalid_core}
    if invalid_core:
        raise PickRejected('工件轮廓测量不稳定：接触带核心宽度须为有限正数，'
                           f'原夹宽跨度{stability["grasp_width_span_m"]*1000:.3f} mm', details=stability)
    if center_change > .003 or height_span > .004:
        raise PickRejected('物体/带面不稳定，请停止传送带后重试', details=stability)
    if width_span > .004:
        raise PickRejected(f'工件轮廓测量不稳定：{stability["selected_width_metric"]}跨度'
                           f'{width_span*1000:.3f} mm超过4.000 mm；'
                           f'原夹宽跨度{stability["grasp_width_span_m"]*1000:.3f} mm', details=stability)
    if len({s.get('target_geometry_source', 'depth_foreground_envelope') for s in observations}) > 1:
        raise PickRejected('多帧形状判断不一致，不能切换几何模型继续抓取', details=stability)
    if all('grasp' in s for s in observations):
        ref = observations[-1]['grasp']
        angles = []
        for ob in observations:
            # Parallel jaws are equivalent under a 180-degree local Z turn.
            alternate = ob['grasp'].copy()
            alternate[:3, :3] = alternate[:3, :3] @ np.diag([-1., -1., 1.])
            angles.append(min(transform_distance(ob['grasp'], ref)[1],
                              transform_distance(alternate, ref)[1]))
        stability['direction_max_delta_deg'] = float(max(angles))
        if stability['direction_max_delta_deg'] > 5:
            raise PickRejected('抓取方向估计不稳定', details=stability)
    latest = observations[-1].copy()
    latest['geometry_stability'] = stability
    return latest


def _first_native_penetrating_contact(data):
    """First negative MuJoCo contact, preserving the original contact order."""
    negative = np.flatnonzero(data.contact.dist[:data.ncon] < 0)
    if not len(negative):
        return None
    contact = data.contact[int(negative[0])]
    return int(contact.geom1), int(contact.geom2), float(contact.dist)


class RobotModel:
    """Private existing model in robot BASE frame, never the simulation world."""
    def __init__(self, self_filter_config=None, scene_noise_filter_config=None,
                 belt_scene_filter_config=None, *, gripper_scene_margin_m=0.,
                 simd_broadphase_enabled=True, simd_broadphase_backend='auto'):
        import mujoco as mj
        from scipy.spatial import HalfspaceIntersection
        self.mj = mj
        package = Path(__file__).resolve().parents[1]
        root = ET.fromstring((package/'models/robot.xml').read_text())
        ET.SubElement(root, 'compiler', angle='radian', meshdir=str(package/'assets'))
        # Standalone robot.xml normally inherits this class from scene.xml.
        cls = ET.SubElement(root.find('default'), 'default', {'class': 'visual_only'})
        ET.SubElement(cls, 'geom', contype='0', conaffinity='0', density='0')
        base = root.find(".//body[@name='base_link']")
        base.set('pos', '0 0 0')
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mj.MjData(self.model)
        self.names = [f'joint{i}' for i in range(1, 8)]
        self.ids = [self.model.joint(n).id for n in self.names]
        self.indices = self.model.jnt_qposadr[self.ids]
        self.limits = self.model.jnt_range[self.ids]
        self.nominal_limits = self.limits.copy()
        self.joint_margin_rad = .02
        self.command_lower = self.limits[:, 0]+self.joint_margin_rad
        self.command_upper = self.limits[:, 1]-self.joint_margin_rad
        self.nominal_command_lower = self.command_lower.copy()
        self.nominal_command_upper = self.command_upper.copy()
        self._start_joint_tolerance_configuration = None
        self.flange_id = self.model.body('link7').id
        self.tcp_id = self.model.site('ee_site').id
        self.fingers = {self.model.geom(n).id for n in ('left_finger_geom', 'right_finger_geom')}
        self.gripper_geometries = self.fingers | {
            self.model.geom('nero_gripper_base_collision').id,
            self.model.geom('nero_gripper_flange_collision').id,
        }
        # Scene checks use the actual gripper hull. The explicit parameter is
        # retained for saved-scene comparisons with the former 8 mm envelope.
        # A zero margin still runs the complete hull intersection check.
        self.gripper_scene_margin_m = float(gripper_scene_margin_m)
        if not np.isfinite(self.gripper_scene_margin_m) or self.gripper_scene_margin_m < 0:
            raise ValueError('夹爪场景外扩距离须为非负有限米值')
        self.ik = DampedLeastSquaresIK(self.model, self.data, self.names, 'ee_site',
                                     IKConfig(max_iterations=200, position_tolerance=.0003,
                                              orientation_tolerance=.010, joint_centering_gain=0))
        # The private IK and path validator must use exactly the same bounds.
        # Previously IK used nominal ranges and could return a pose that the
        # subsequent 20 mrad execution margin immediately rejected.
        self.ik.lower_limits = self.command_lower.copy()
        self.ik.upper_limits = self.command_upper.copy()
        self.ik.center_positions = .5*(self.command_lower+self.command_upper)
        self.bounded_ik_fallbacks = 0
        self._pick_trajectory_cache = {}
        self.meshes = {}
        self.mesh_vertices = {}
        self.collision_radii = {}
        self.mesh_bounds = {}
        self.simd_broadphase_enabled = bool(simd_broadphase_enabled)
        self.simd_broadphase_backend = str(simd_broadphase_backend)
        self._official_target_index_cache = None
        self._target_broadphase_kernels = {}
        self._target_broadphase_batch_cache = None
        self._target_broadphase_bounds_cache = None
        self.simd_broadphase_stats = dict(batch_calls=0, batch_states=0,
            batch_cache_hits=0, single_state_calls=0, target_geometry_tests=0,
            target_geometry_proven_separated=0)
        self.self_filter = ModelSelfFilter(load_self_filter_config() if self_filter_config is None
                                         else self_filter_config)
        self.self_filter_result = None
        self.scene_noise_filter_config = (load_scene_noise_filter_config()
                                         if scene_noise_filter_config is None
                                         else scene_noise_filter_config)
        self.scene_noise_filter_result = None
        self.scene_depth_filter_result = None
        self.scene_saturation_filter_result = None
        self.scene_stereo_filter_result = None
        self.scene_background_reference = None
        self.scene_background_filter_result = None
        self.belt_scene_filter_config = (load_belt_scene_filter_config()
                                        if belt_scene_filter_config is None
                                        else belt_scene_filter_config)
        self.scene_region_filter_result = None
        self.scene_voxel_result = None
        self.scene_region_plane_to_base = None
        for g in range(self.model.ngeom):
            if not self.model.geom_contype[g] and not self.model.geom_conaffinity[g]:
                continue
            mid = int(self.model.geom_dataid[g])
            if mid < 0:
                raise PickRejected('碰撞几何必须是已知机器人网格')
            begin, count = int(self.model.mesh_vertadr[mid]), int(self.model.mesh_vertnum[mid])
            vertices = np.array(self.model.mesh_vert[begin:begin+count])
            self.meshes[g] = (ConvexHull(vertices).equations, np.max(np.linalg.norm(vertices, axis=1)))
            self.mesh_vertices[g] = vertices
            # A halfspace-offset corner may extend beyond radius+margin.
            # Also used by the separate target-workpiece checks (4/6 mm).
            # Keep their conservative bound when gripper scene padding is 0.
            self.collision_radii[g] = ConvexBody(
                g, self.model.body(self.model.geom_bodyid[g]).name,
                self.meshes[g][0], vertices, np.zeros(3), np.eye(3),
                max(.008, self.scene_collision_margin(g))).padded_radius
            # Bound the actual offset halfspaces, including their extended
            # corners. Expanding the nominal mesh AABB by margin is insufficient.
            # This is only a broad phase; every surviving point still receives
            # the original complete halfspace test with the unchanged margin.
            equations = self.meshes[g][0]
            self.mesh_bounds[g, 0.] = (vertices.min(axis=0)-1e-9,
                                      vertices.max(axis=0)+1e-9)
            for margin in {.002, .004, .006, .008, self.scene_collision_margin(g)} - {0.}:
                padded = equations.copy(); padded[:, 3] -= margin
                envelope = HalfspaceIntersection(padded, vertices.mean(axis=0)).intersections
                self.mesh_bounds[g, margin] = (envelope.min(axis=0)-1e-9,
                                               envelope.max(axis=0)+1e-9)
        self._target_broadphase_geom_ids = tuple(self.meshes)
        if self.simd_broadphase_enabled:
            # Compile/load the small optional SIMD kernel during model setup,
            # before the first workpiece route is submitted. No points/FK are
            # calculated here and unavailable native code uses NumPy.
            self._target_broadphase_kernel(.002)
            self._target_broadphase_kernel(.004)
            self._target_broadphase_kernel(.006)
            if self._target_broadphase_mode() == 'capt':
                from .official_capt import _load_native
                _load_native()
        self.scene = np.empty((0, 3)); self.target = np.empty((0, 3))
        self.tree = cKDTree(self.scene)
        self.set_state(np.zeros(7), .1)
        self.flange_to_tcp = inverse(self.flange()) @ self.tcp()

    def scene_collision_margin(self, geometry):
        return self.gripper_scene_margin_m if geometry in self.gripper_geometries else .008

    def configure_start_joint_tolerance(self, q, tolerance_deg=None,
                                        feedback_slack_rad=.003):
        """Include the measured start in this run's private path bounds.

        Only the side exceeded by the captured measured start is extended in
        this private model. IK endpoints retain the original command bounds;
        the complete policy can therefore connect this start to a normal IK
        branch without clipping the initial controller command. This is not a
        firmware-limit update or a calibration of the physical joint limits.
        The default accepts the finite measured start without a fixed angular
        offset gate. A caller can still explicitly request a tolerance in degrees.

        An identical second call is idempotent. A different second call is
        rejected so subsequent feedback cannot progressively grow the envelope.
        """
        try:
            start = np.asarray(q, dtype=float)
            tolerance = None if tolerance_deg is None else float(tolerance_deg)
            slack = float(feedback_slack_rad)
        except (TypeError, ValueError, OverflowError) as error:
            raise PickRejected('当前位姿关节容差配置须为有限数值') from error
        if start.shape != (7,) or not np.isfinite(start).all():
            raise PickRejected('当前位姿须为7个有限弧度值')
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise PickRejected('显式指定的当前位姿关节容差须为非负有限值')
        if not np.isfinite(slack) or not 0 <= slack <= .003:
            raise PickRejected('当前位姿反馈余量须为0–0.003 rad')
        captured = self._start_joint_tolerance_configuration
        if captured is not None and not (
                np.array_equal(captured[0], start)
                and captured[1] == tolerance and captured[2] == slack):
            raise PickRejected('本次运行的起始关节容差已固定，不能继续扩大')

        lower, upper = self.nominal_command_lower, self.nominal_command_upper
        below = start < lower-1e-10
        above = start > upper+1e-10
        excess = np.maximum.reduce((lower-start, start-upper, np.zeros(7)))
        rejected = (np.zeros(7, dtype=bool) if tolerance is None else
                    (below | above) & (excess > np.deg2rad(tolerance)+1e-10))
        if np.any(rejected):
            detail = '；'.join(
                f'{self.names[i]}={np.rad2deg(start[i]):.4f}°，'
                f'超过原规划边界{np.rad2deg(excess[i]):.4f}°'
                for i in np.flatnonzero(rejected))
            raise PickRejected(f'当前位姿超过{tolerance:g}°起始关节容差：{detail}')

        path_limits = self.nominal_limits.copy()
        path_limits[below, 0] = start[below]-slack-self.joint_margin_rad
        path_limits[above, 1] = start[above]+slack+self.joint_margin_rad
        changed = np.flatnonzero(below | above)
        actuator_ids = []
        # Resolve every actuator before changing any range, retaining atomic
        # rejection for an inconsistent private model.
        for index in changed:
            candidates = np.flatnonzero(
                self.model.actuator_trnid[:, 0] == self.ids[index])
            if len(candidates) != 1:
                raise PickRejected(f'{self.names[index]}缺少唯一的关节执行器')
            actuator_ids.append(int(candidates[0]))
        if captured is None:
            for index, actuator_id in zip(changed, actuator_ids):
                if below[index]:
                    self.model.jnt_range[self.ids[index], 0] = path_limits[index, 0]
                    self.model.actuator_ctrlrange[actuator_id, 0] = path_limits[index, 0]
                if above[index]:
                    self.model.jnt_range[self.ids[index], 1] = path_limits[index, 1]
                    self.model.actuator_ctrlrange[actuator_id, 1] = path_limits[index, 1]
            self.limits = self.model.jnt_range[self.ids]
            self.command_lower = self.limits[:, 0]+self.joint_margin_rad
            self.command_upper = self.limits[:, 1]-self.joint_margin_rad
            self._start_joint_tolerance_configuration = (start.copy(), tolerance, slack)

        return {
            'expanded': bool(len(changed)),
            'expanded_joints': [self.names[i] for i in changed],
            'initial_joints_rad': start.tolist(),
            'mode': 'measured_start' if tolerance is None else 'explicit_tolerance',
            'tolerance_deg': tolerance, 'feedback_slack_rad': slack,
            'observed_excess_deg': np.rad2deg(excess).tolist(),
            'nominal_limits_rad': self.nominal_limits.tolist(),
            'nominal_command_limits_rad': np.column_stack((lower, upper)).tolist(),
            'path_limits_rad': self.limits.tolist(),
            'path_command_limits_rad': np.column_stack(
                (self.command_lower, self.command_upper)).tolist(),
        }

    def set_state(self, q, width):
        shared = getattr(self, '_state_evaluator', None)
        if shared is not None:
            full = self.data.qpos.copy()
            full[self.indices] = q
            full[self.model.joint('gripper_joint1').qposadr] = width/2
            full[self.model.joint('gripper_joint2').qposadr] = -width/2
            shared.prepare(full)
            return
        self.data.qpos[self.indices] = q
        self.data.qpos[self.model.joint('gripper_joint1').qposadr] = width/2
        self.data.qpos[self.model.joint('gripper_joint2').qposadr] = -width/2
        self.mj.mj_forward(self.model, self.data)

    def flange(self):
        t = np.eye(4); t[:3, :3] = self.data.xmat[self.flange_id].reshape(3, 3)
        t[:3, 3] = self.data.xpos[self.flange_id]; return t

    def tcp(self):
        t = np.eye(4); t[:3, :3] = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        t[:3, 3] = self.data.site_xpos[self.tcp_id]; return t

    def check_feedback(self, state):
        paired = state.get('joints_at_pose_stamp')
        if paired is not None:
            q = np.asarray(paired, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise PickRejected('用于法兰时间对齐的关节反馈无效')
            if state.get('joints_at_pose_stamp_time') != mapped_stamp(state, 'pose'):
                raise PickRejected('法兰与对齐关节的反馈时间不同')
            pairing = (state.get('feedback_timing') or {}).get('pose_joint_pairing') or {}
            if 'joint_receive_offset_s' in pairing:
                pose_stamp = mapped_stamp(state, 'pose')
                joint_stamp = state.get('joints_at_pose_joint_receive_time')
                offset = pairing['joint_receive_offset_s']
                if (joint_stamp is None or not np.isfinite([pose_stamp, joint_stamp, offset]).all()
                        or abs(joint_stamp-(pose_stamp+offset)) > 1.e-8
                        or pairing.get('extrapolated') is not False):
                    raise PickRejected('关节反馈配对时间或插值依据无效')
        self.set_state(state['joints'] if paired is None else paired, state['width_m'])
        compared_flange = self.flange()
        d, a = transform_distance(compared_flange, pose_xyz_rpy(state['pose']))
        if paired is not None:
            # Collision checks and planning always retain the newest joints.
            self.set_state(state['joints'], state['width_m'])
        if d > .005 or a > 1.:
            timing=state.get('feedback_timing') or {}
            # Preserve the actual inputs of the rejected comparison.
            raise PickRejected(f'当前关节FK与法兰反馈不符：{d*1000:.2f} mm / {a:.2f}°',details={
                'reason':'feedback_fk_mismatch',
                'position_error_m':float(d),'orientation_error_deg':float(a),
                'position_tolerance_m':.005,'orientation_tolerance_deg':1.,
                'joint_source': ('interpolated_with_receive_offset' if paired is not None and
                    (timing.get('pose_joint_pairing') or {}).get('joint_receive_offset_s', 0.) else
                    'interpolated_at_pose_receive_time' if paired is not None else 'latest_joint_feedback'),
                'pose_packet_span_s':timing.get('pose_packet_span_s'),
                'joint_packet_span_s':timing.get('joint_packet_span_s'),
                'pose_joint_pairing':timing.get('pose_joint_pairing'),
                'pose_grouping':timing.get('pose_grouping'),
                'feedback_flange_xyz_rpy_m_rad':list(state['pose']),
                'feedback_joints_rad':list(state['joints']),
                'compared_joints_rad':list(state['joints'] if paired is None else paired),
                'fk_flange_transform':compared_flange.tolist(),
                'feedback_flange_transform':pose_xyz_rpy(state['pose']).tolist(),
                'pose_packet_stamps_monotonic_s':state.get('pose_stamps_monotonic'),
                'time_limitation':'host CAN receive timestamps are not hardware sample timestamps'})

    def inside_geom(self, points, g, margin):
        equations, _ = self.meshes[g]
        local = (points-self.data.geom_xpos[g]) @ self.data.geom_xmat[g].reshape(3, 3)
        bounds = self.mesh_bounds.get((g, margin))
        if bounds is None:
            return np.all(local @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)
        lower, upper = bounds
        selected = np.flatnonzero(np.all((local >= lower) & (local <= upper), axis=1))
        result = np.zeros(len(local), dtype=bool)
        if not len(selected):
            return result
        values = local[selected] @ equations[:, :3].T + equations[:, 3]
        # Changing matrix shape can change floating-point rounding at an exact
        # face. Preserve the original full-cloud calculation for that rare case;
        # this tolerance never expands the final collision/contact margin.
        roundoff = 32*np.finfo(float).eps*(1+np.sum(np.abs(local[selected]), axis=1)
                    + np.max(np.abs(equations[:, 3])) + abs(margin))
        if np.any(np.min(np.abs(values-margin), axis=1) <= roundoff):
            return np.all(local @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)
        result[selected] = np.all(values <= margin, axis=1)
        return result

    def _target_broadphase_kernel(self, margin):
        from .simd_broadphase import ConvexBroadPhase
        margin = float(margin)
        ids = tuple(self.meshes)
        key = (ids, margin)
        kernel = self._target_broadphase_kernels.get(key)
        if kernel is None:
            # Only use envelopes of the original offset halfspaces. A scalar
            # AABB expansion would miss the extended corners of sharp hulls.
            if not all((g, margin) in self.mesh_bounds for g in ids):
                return None
            kernel = ConvexBroadPhase(np.array([self.mesh_bounds[g, margin] for g in ids]),
                                      geometry_ids=ids)
            self._target_broadphase_kernels[key] = kernel
        return kernel

    def _target_bounds(self, points):
        from .simd_broadphase import bounds_of_points
        cache = self._target_broadphase_bounds_cache
        immutable = not points.flags.writeable
        if immutable and cache is not None and cache[0] is points:
            return cache[1]
        bounds = bounds_of_points(points)
        # Keep a strong reference: Python cannot recycle a former cloud id
        # while a cached box still refers to that immutable publication.
        self._target_broadphase_bounds_cache = (points, bounds) if immutable else None
        return bounds

    def _target_broadphase_mode(self):
        mode = getattr(self, 'simd_broadphase_backend', 'auto')
        return os.environ.get('NERO_TARGET_BROADPHASE_BACKEND', 'aabb') if mode == 'auto' else mode

    def _official_target_index(self, points):
        """Official CAPT is optional and keeps every unsupported candidate."""
        if points.flags.writeable:
            return None
        cache = self._official_target_index_cache
        if cache is not None and cache[0] is points:
            return cache[1]
        from .official_capt import OfficialCAPTIndex
        maximum_radius = max(float(self.collision_radii[g]) for g in self.meshes)
        index = OfficialCAPTIndex(points, maximum_radius)
        # The immutable publication is retained; a replacement never closes an
        # index still referenced by another copied geometry/worker.
        self._official_target_index_cache = (points, index)
        return index

    def _target_batch_candidates(self, kernel, positions, rotations, points):
        """Proposals only: official sphere and/or AABB conservative pruning."""
        if self._target_broadphase_mode() == 'capt':
            index = self._official_target_index(points)
            if index is not None and index.available:
                radii = np.array([self.collision_radii[g] for g in kernel.geometry_ids])
                possible = index.query(positions.reshape(-1, 3),
                    np.broadcast_to(radii, positions.shape[:2]).ravel()).reshape(positions.shape[:2])
                # CAPT sphere flags can still be ambiguous near the gripper;
                # the unchanged enclosing AABB provides another safe prune.
                boxes = kernel.world_aabbs(positions, rotations)
                return boxes, possible & kernel.object_candidates(boxes, self._target_bounds(points))
        boxes = kernel.world_aabbs(positions, rotations)
        return boxes, kernel.object_candidates(boxes, self._target_bounds(points))

    def target_broadphase_world_bounds(self, margin):
        """Padded hull world boxes[G,2,3] from the already prepared FK."""
        kernel = self._target_broadphase_kernel(margin)
        if kernel is None:
            return None
        shared = getattr(self, '_state_evaluator', None)
        features = None if shared is None else shared.current
        key = (id(self.model), id(kernel))
        cached = None if features is None else getattr(features, '_target_world_boxes', {}).get(key)
        if cached is not None:
            return cached
        ids = list(kernel.geometry_ids)
        boxes = kernel.world_aabbs(self.data.geom_xpos[ids][None],
            self.data.geom_xmat[ids].reshape(1, len(ids), 3, 3))[0]
        boxes.setflags(write=False)
        if features is not None:
            # These are geometry facts only; no cloud/scene verdict is saved
            # in the request-independent shared state evaluator.
            if not hasattr(features, '_target_world_boxes'):
                features._target_world_boxes = {}
            features._target_world_boxes[key] = boxes
        return boxes

    def prepare_target_broadphase(self, states, width, points, margin):
        """Batch target pruning from cached FK, without computing FK again.

        The batch cache lasts only until the next call and retains the exact
        immutable target publication, full qpos and collision margin. The
        original target narrow phase still checks every surviving geometry.
        """
        self._target_broadphase_batch_cache = None
        shared = getattr(self, '_state_evaluator', None)
        points = np.asarray(points)
        if (not self.simd_broadphase_enabled or shared is None or points.flags.writeable
                or not len(points)):
            return False
        kernel = self._target_broadphase_kernel(margin)
        if kernel is None:
            return False
        ids = list(kernel.geometry_ids)
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.indices):
            return False
        keys, features = [], []
        full = self.data.qpos.copy()
        full[self.model.joint('gripper_joint1').qposadr] = width/2
        full[self.model.joint('gripper_joint2').qposadr] = -width/2
        for q in states:
            full[self.indices] = q
            key = full.tobytes()
            feature = shared._states.get(key)
            if feature is None:
                # This batch has not been prepared by the shared native
                # checks; ordinary per-state pruning will keep working.
                return False
            keys.append(key)
            features.append(feature)
        if not features:
            return False
        positions = np.array([feature.poses['geom_xpos'][ids] for feature in features])
        rotations = np.array([feature.poses['geom_xmat'][ids].reshape(len(ids), 3, 3)
                              for feature in features])
        boxes, possible = self._target_batch_candidates(kernel, positions, rotations, points)
        boxes.setflags(write=False)
        candidates = {key: frozenset(g for g, value in zip(kernel.geometry_ids, mask) if value)
                      for key, mask in zip(keys, possible)}
        self._target_broadphase_batch_cache = (self.data, points, float(margin), candidates,
                                              self._target_broadphase_mode())
        for feature, row in zip(features, boxes):
            if not hasattr(feature, '_target_world_boxes'):
                feature._target_world_boxes = {}
            feature._target_world_boxes[id(self.model), id(kernel)] = row
        stats = self.simd_broadphase_stats
        stats['batch_calls'] += 1
        stats['batch_states'] += len(states)
        stats['target_geometry_tests'] += possible.size
        stats['target_geometry_proven_separated'] += int(np.count_nonzero(~possible))
        return True

    def target_broadphase_candidates(self, points, margin):
        """Original geom ids that still need exact target-point checking."""
        ids = tuple(self.meshes)
        if not self.simd_broadphase_enabled:
            return frozenset(ids)
        points = np.asarray(points)
        if not len(points):
            return frozenset()
        cache = self._target_broadphase_batch_cache
        if (cache is not None and cache[0] is self.data and cache[1] is points
                and cache[2] == float(margin) and not points.flags.writeable
                and cache[4] == self._target_broadphase_mode()):
            candidates = cache[3].get(self.data.qpos.tobytes())
            if candidates is not None:
                self.simd_broadphase_stats['batch_cache_hits'] += 1
                return candidates
        kernel = self._target_broadphase_kernel(margin)
        if kernel is None:
            return frozenset(ids)
        boxes = self.target_broadphase_world_bounds(margin)
        possible = kernel.object_candidates(boxes[None], self._target_bounds(points))[0]
        if self._target_broadphase_mode() == 'capt':
            index = self._official_target_index(points)
            if index is not None and index.available:
                radii = np.array([self.collision_radii[g] for g in kernel.geometry_ids])
                possible &= index.query(self.data.geom_xpos[list(kernel.geometry_ids)], radii)
        stats = self.simd_broadphase_stats
        stats['single_state_calls'] += 1
        stats['target_geometry_tests'] += possible.size
        stats['target_geometry_proven_separated'] += int(np.count_nonzero(~possible))
        return frozenset(g for g, value in zip(kernel.geometry_ids, possible) if value)

    def _self_filter_bodies(self, data):
        return [ConvexBody(
            g, self.model.body(self.model.geom_bodyid[g]).name, equations, self.mesh_vertices[g],
            data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy(),
            self.self_filter.config.padding_for(self.model.body(self.model.geom_bodyid[g]).name))
            for g, (equations, _) in self.meshes.items()]

    def self_filter_bodies_for_state(self, state):
        """Build private FK bodies from captured feedback, preserving planning state."""
        # Only robot geometry is reusable. Point classification / visibility
        # is always rerun for the current cloud and camera. Model placements,
        # hulls, padding and measured joint/gripper state all identify a body.
        key = (np.asarray(state['joints'], dtype=float).tobytes(), float(state['width_m']),
            self.model.qpos0.tobytes(), self.model.body_pos.tobytes(), self.model.body_quat.tobytes(),
            self.model.geom_pos.tobytes(), self.model.geom_quat.tobytes(),
            tuple((g, equations.tobytes(), self.mesh_vertices[g].tobytes(),
                self.self_filter.config.padding_for(self.model.body(self.model.geom_bodyid[g]).name))
                for g, (equations, _) in self.meshes.items()))
        previous = getattr(self, '_captured_self_bodies', None)
        if previous is not None and previous[0] == key:
            return previous[1]
        data = self.mj.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        data.qpos[self.indices] = state['joints']
        data.qpos[self.model.joint('gripper_joint1').qposadr] = state['width_m']/2
        data.qpos[self.model.joint('gripper_joint2').qposadr] = -state['width_m']/2
        self.mj.mj_forward(self.model, data)
        bodies = self._self_filter_bodies(data)
        for body in bodies:
            for name in ('equations', 'vertices', 'position', 'rotation', 'padded_equations'):
                value = getattr(body, name).copy()
                value.setflags(write=False)
                setattr(body, name, value)
        bodies = tuple(bodies)
        self._captured_self_bodies = (key, bodies)
        return bodies

    def classify_robot_points(self, points, state, *, camera_origin=None):
        """Apply the scene's self filter at a measured pose without changing it."""
        return self.self_filter.classify(points, self.self_filter_bodies_for_state(state), camera_origin)

    def install_scene(self, points, target, state, *, camera_origin=None, belt_region=None,
                      depth_frame=None, target_mask=None, scene_stereo_backend=None,
                      scene_filter_output_directory=None, scene_filter_evidence=None,
                      scene_saturation_enabled=True, scene_background_reference=None,
                      scene_background_enabled=True, gripper_nearby_scene_ignore=False,
                      scene_pixel_indices=None):
        from .scene_workspace import scene_workspace
        with scene_workspace() as workspace:
            started = time.perf_counter()
            result = self._install_scene(points, target, state, camera_origin=camera_origin,
                belt_region=belt_region, depth_frame=depth_frame, target_mask=target_mask,
                scene_stereo_backend=scene_stereo_backend,
                scene_filter_output_directory=scene_filter_output_directory,
                scene_filter_evidence=scene_filter_evidence,
                scene_saturation_enabled=scene_saturation_enabled,
                scene_background_reference=scene_background_reference,
                scene_background_enabled=scene_background_enabled,
                gripper_nearby_scene_ignore=gripper_nearby_scene_ignore)
            result['dataflow_timings_s'] = dict(workspace.timings_s,
                install_compute_and_submit_s=time.perf_counter()-started)
            result['dataflow_calls'] = dict(workspace.calls)
            # Every layer indexes its immediate predecessor. Preserve that
            # explicit composition to the original caller indices/pixels.
            source = self._finite_scene_source_indices
            self.scene_source_indices = {}
            for name, layer in (('region', self.scene_region_filter_result),
                    ('self', self.self_filter_result), ('saturation', self.scene_saturation_filter_result),
                    ('depth', self.scene_depth_filter_result), ('background', self.scene_background_filter_result),
                    ('stereo', self.scene_stereo_filter_result), ('noise', self.scene_noise_filter_result)):
                self.scene_source_indices[name+'_removed'] = source[layer.removed_mask]
                source = source[~layer.removed_mask]
                self.scene_source_indices[name+'_retained'] = source
            self.scene_source_indices['before_gripper_ignore'] = source
            if scene_pixel_indices is not None:
                pixel_ids = np.asarray(scene_pixel_indices)
                if pixel_ids.shape != (len(points),) or pixel_ids.dtype.kind not in 'iu':
                    raise ValueError('scene_pixel_indices must map each original scene point')
                self.scene_native_pixel_indices = {name: pixel_ids[index]
                    for name, index in self.scene_source_indices.items()}
            return result

    def _install_scene(self, points, target, state, *, camera_origin=None, belt_region=None,
                      depth_frame=None, target_mask=None, scene_stereo_backend=None,
                      scene_filter_output_directory=None, scene_filter_evidence=None,
                      scene_saturation_enabled=True, scene_background_reference=None,
                      scene_background_enabled=True, gripper_nearby_scene_ignore=False):
        from .scene_workspace import scene_call
        scene_call('robot_state_fk', self.set_state, state['joints'], state['width_m'])
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('场景点云须为N×3基座坐标，单位米')
        finite = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) < 1.5)
        self._finite_scene_source_indices = np.flatnonzero(finite)
        points = points[finite]
        region_config, plane_to_base = self.belt_scene_filter_config.resolve_region(belt_region)
        self.scene_region_plane_to_base = plane_to_base
        # Crop all colors in 3D; green points define only the measured bounds.
        # Density remains native until self/noise filtering has completed.
        self.scene_region_filter_result = scene_call('scene_region', filter_scene_region, points, plane_to_base, region_config)
        self.scene_region_filter_result.statistics['belt_boundary'] = (
            dict(belt_region.get('scene_boundary_estimation', {}))
            if isinstance(belt_region, dict) else {})
        points = self.scene_region_filter_result.retained_points
        if region_config.enabled and len(target):
            target_region = scene_call('scene_region', filter_scene_region, target, plane_to_base, region_config)
            if target_region.removed_mask.any():
                flat = np.linalg.solve(plane_to_base[:3, :3],
                    (np.asarray(target)-plane_to_base[:3, 3]).T).T
                bounds = np.array([region_config.x_bounds_m, region_config.y_bounds_m,
                                   region_config.z_bounds_m])
                raise PickRejected('传送带点云范围未完整覆盖工件，请调整范围后重试', details={
                    'kind': 'target_outside_belt_scene_roi',
                    'target_points': len(target),
                    'outside_points': int(target_region.removed_mask.sum()),
                    'below_axis_counts': (flat < bounds[:, 0]-1e-12).sum(axis=0).tolist(),
                    'above_axis_counts': (flat > bounds[:, 1]+1e-12).sum(axis=0).tolist(),
                    'max_excess_m': np.maximum(np.maximum(bounds[:, 0]-flat.min(axis=0),
                                                         flat.max(axis=0)-bounds[:, 1]), 0).tolist(),
                    'roi': target_region.statistics['roi'],
                    'boundary_estimation': dict(belt_region.get('scene_boundary_estimation', {})),
                })
        bodies = scene_call('robot_geometry_prepare', self.self_filter_bodies_for_state, state)
        # Classify native depth points BEFORE voxel sampling, so an arbitrary
        # voxel representative cannot hide a robot edge or a nearby obstacle.
        self.self_filter_result = scene_call('scene_self', self.self_filter.classify, points, bodies, camera_origin)
        points = points[~self.self_filter_result.removed_mask]
        # Saturated stereo pixels can have unstable, ambiguous depths even
        # when an isolated component is dense. Invalidate those original
        # measurements only; retain the physical robot model and target.
        from .scene_filter_evidence import restore_scene_filter_evidence
        evidence = scene_filter_evidence or {}
        output = None if scene_filter_output_directory is None else Path(scene_filter_output_directory)
        self.scene_saturation_filter_result = restore_scene_filter_evidence(
            points, depth_frame, evidence.get('scene_saturation_filter'), target_mask)
        if not self.scene_noise_filter_config.enabled or not scene_saturation_enabled:
            self.scene_saturation_filter_result = None
        if self.scene_saturation_filter_result is None:
            self.scene_saturation_filter_result = scene_call('scene_saturation', filter_scene_saturation, 
                points, frame=depth_frame, self_filter_result=self.self_filter_result,
                protected_mask=target_mask,
                config=SceneSaturationFilterConfig(enabled=self.scene_noise_filter_config.enabled
                                                   and scene_saturation_enabled),
                output_directory=None if output is None else output/'saturation')
        points = self.scene_saturation_filter_result.retained_points
        # Native depth evidence distinguishes a dense gripper-edge remnant from
        # a sparse outlier. Inspect it before density filtering can cut a thin
        # connection to a genuine surface. Raw RGB-D and target stay unchanged.
        depth_config = self.scene_noise_filter_config.depth_edge_tail
        if not self.scene_noise_filter_config.enabled:
            depth_config = replace(depth_config, enabled=False)
        depth_filter_started = time.perf_counter()
        self.scene_depth_filter_result = scene_call('scene_depth', filter_depth_edge_tails, 
            points, frame=depth_frame, self_filter_result=self.self_filter_result,
            protected_mask=target_mask, config=depth_config)
        self.scene_depth_filter_result.statistics['processing_s'] = time.perf_counter()-depth_filter_started
        points = points[~self.scene_depth_filter_result.removed_mask]
        from .scene_background_filter import filter_scene_background, state_digest, geometry_digest
        background_saved = evidence.get('scene_background_filter')
        self.scene_background_filter_result = None
        if (scene_background_enabled and self.scene_noise_filter_config.enabled
                and isinstance(background_saved, dict)
                and background_saved.get('robot_state_sha256') == state_digest(state)
                and (scene_background_reference is None or background_saved.get('reference_sha256')
                     == scene_background_reference.manifest_sha256)):
            self.scene_background_filter_result = restore_scene_filter_evidence(
                points, depth_frame, background_saved, target_mask,
                expected_metadata={'robot_state_sha256': state_digest(state),
                    'occlusion_geometry_sha256': geometry_digest(bodies),
                    'reference_sha256': background_saved.get('reference_sha256')
                        if scene_background_reference is None else scene_background_reference.manifest_sha256})
        if self.scene_background_filter_result is None:
            self.scene_background_filter_result = scene_call('scene_background', filter_scene_background, 
                points, frame=depth_frame, protected_mask=target_mask,
                reference=scene_background_reference, bodies=bodies, state=state,
                depth_edge_result=self.scene_depth_filter_result, depth_config=depth_config,
                ray_margin_m=self.self_filter.config.shadow_threshold_m,
                enabled=scene_background_enabled and self.scene_noise_filter_config.enabled,
                output_directory=None if output is None or scene_background_reference is None
                else output/'background')
        points = self.scene_background_filter_result.retained_points
        # GPU rechecking is explicit: the saved real-scene comparison did not
        # justify adding its latency to every recognition. Missing evidence or
        # a failed optional worker preserves all remaining native samples.
        self.scene_stereo_filter_result = restore_scene_filter_evidence(
            points, depth_frame, evidence.get('scene_stereo_filter'), target_mask)
        if not self.scene_noise_filter_config.enabled:
            self.scene_stereo_filter_result = None
        if self.scene_stereo_filter_result is None:
            self.scene_stereo_filter_result = scene_call('scene_stereo', filter_scene_stereo, 
                points, frame=depth_frame, protected_mask=target_mask, backend=scene_stereo_backend,
                config=SceneStereoFilterConfig(enabled=self.scene_noise_filter_config.enabled),
                output_directory=None if output is None or scene_stereo_backend is None else output/'stereo')
        points = self.scene_stereo_filter_result.retained_points
        # Evaluate density on native points once, before voxel representatives
        # make every surface artificially sparse. Targets stay separate below.
        self.scene_noise_filter_result = scene_call('scene_noise', filter_scene_noise, 
            points, self.scene_noise_filter_config, collect_neighbor_counts=False)
        points = self.scene_noise_filter_result.retained_points
        # The temporary experimental region is frozen at this capture, before
        # any candidate IK mutates model.data. Density has already been checked
        # so this removal cannot change density decisions outside the region.
        from .gripper_scene_ignore import ignore_captured_gripper_scene
        before_ignore = points
        points, ignored, self.gripper_nearby_scene_ignore_report = ignore_captured_gripper_scene(
            points, bodies, state, enabled=gripper_nearby_scene_ignore)
        if gripper_nearby_scene_ignore and output is not None:
            destination = output/'gripper_nearby'
            destination.mkdir(parents=True, exist_ok=True)
            header = getattr(depth_frame, 'header', None)
            stem = f'e{getattr(header, "clock_epoch", 0)}_s{getattr(header, "sequence", 0):06d}'
            removed_path = destination/f'{stem}_removed.npy'
            np.save(removed_path, before_ignore[ignored])
            self.gripper_nearby_scene_ignore_report['removed_points_file'] = str(removed_path)
            (destination/f'{stem}.json').write_text(
                json.dumps(self.gripper_nearby_scene_ignore_report, ensure_ascii=False, indent=2)+'\n')
        # Keep actual samples as representatives, with at most 4 mm coverage
        # error. Farther samples remain so diagonal cells cannot lose a corner.
        self.scene_voxel_result = scene_call('voxel_downsample', voxel_downsample, 
            points, voxel_size_m=self.belt_scene_filter_config.voxel_size_m, max_error_m=.004)
        self.scene = self.scene_voxel_result.retained_points
        # Target geometry is separate from scene points and never self-filtered.
        self.target = np.asarray(target)[::3]
        self.tree = scene_call('collision_tree', cKDTree, self.scene)
        return {'visible_scene_points': len(self.scene),
                'collision_margins_m': {'gripper_scene': self.gripper_scene_margin_m,
                                        'arm_scene': .008},
                'robot_points_removed': self.self_filter_result.statistics['robot_points_removed'],
                'self_filter': self.self_filter_result.statistics,
                'scene_region_filter': self.scene_region_filter_result.statistics,
                'scene_depth_filter': {**self.scene_depth_filter_result.statistics,
                                       'components': self.scene_depth_filter_result.components},
                'scene_background_filter': {**self.scene_background_filter_result.statistics,
                                            'components': self.scene_background_filter_result.components},
                'scene_saturation_filter': {**self.scene_saturation_filter_result.statistics,
                                            'components': self.scene_saturation_filter_result.components},
                'scene_stereo_filter': {**self.scene_stereo_filter_result.statistics,
                                        'components': self.scene_stereo_filter_result.components},
                'scene_noise_filter': self.scene_noise_filter_result.statistics,
                'gripper_nearby_scene_ignore': self.gripper_nearby_scene_ignore_report,
                'scene_downsample': self.scene_voxel_result.statistics,
                'coverage': ('visible surfaces inside the measured belt ROI only; outside ROI and occluded space '
                             'are not collision checked' if region_config.enabled else
                             'one camera visible surfaces only; occluded space is NOT certified clear')}

    def scene_collision_index(self):
        """Keep one immutable observed-point index for many sampled states."""
        mode = getattr(self, 'scene_collision_backend', 'auto')
        index = getattr(self, '_observed_scene_index', None)
        if index is None or index.backend != mode or not np.array_equal(index.points, self.scene):
            from .gpu_scene_collision import ObservedPointCollisionIndex
            index = ObservedPointCollisionIndex(self.scene,
                {g: equations for g, (equations, _) in self.meshes.items()},
                backend=mode, tree=self.tree)
            self._observed_scene_index = index
        return index

    def check_state(self, q, width, *, allow_finger_contact=False, check_scene=True):
        q = np.asarray(q)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise PickRejected('关节输入须为7个有限弧度值')
        bad = (q < self.command_lower-1e-10) | (q > self.command_upper+1e-10)
        if bad.any():
            details = [f'{self.names[i]}={np.rad2deg(q[i]):.4f}°，'
                       f'允许[{np.rad2deg(self.command_lower[i]):.4f}, {np.rad2deg(self.command_upper[i]):.4f}]°'
                       for i in np.flatnonzero(bad)]
            raise PickRejected('关节角超出规划范围：'+'；'.join(details))
        self.set_state(q, width)
        shared = getattr(self, '_state_evaluator', None)
        prevalidated = getattr(shared, 'native_contacts_prevalidated', None)
        native_proven = callable(prevalidated) and prevalidated(self.data.qpos)
        if not native_proven and shared is None:
            # MuJoCo exposes one contiguous distance view. Inspect every
            # native contact in its original order, constructing a Python
            # contact object only for the first negative-distance pair.
            contact = _first_native_penetrating_contact(self.data)
            if contact is not None:
                geom1, geom2, _ = contact
                raise PickRejected(f'模型自碰撞：{self.model.geom(geom1).name}/'
                                   f'{self.model.geom(geom2).name}')
        elif not native_proven:
            for geom1, geom2, distance in shared.contacts():
                if distance < 0:
                    raise PickRejected(f'模型自碰撞：{self.model.geom(geom1).name}/{self.model.geom(geom2).name}')
        target_candidates = (self.target_broadphase_candidates(self.target, .004)
                             if len(self.target) else frozenset())
        for g, (_, radius) in self.meshes.items():
            margin = self.scene_collision_margin(g)
            ids = (self.tree.query_ball_point(self.data.geom_xpos[g], self.collision_radii[g]+1e-9)
                   if check_scene else [])
            if ids:
                nearby = self.scene[ids]
                colliding = nearby[self.inside_geom(nearby, g, margin)]
                if len(colliding):
                    raise PickRejected(f'路径碰到可见障碍点云：{self.model.geom(g).name}', details={
                        'kind': 'scene_collision', 'geometry': self.model.geom(g).name,
                        'point_count': len(colliding), 'points_base_m': colliding[:20].tolist(),
                        'joints_rad': q.tolist(), 'width_m': float(width), 'margin_m': margin,
                    })
            if (not (allow_finger_contact and g in self.fingers) and len(self.target)
                    and g in target_candidates):
                if self.inside_geom(self.target, g, .004).any():
                    raise PickRejected(f'机器人会提前碰到工件：{self.model.geom(g).name}')

    def check_edge(self, a, b, width, *, allow_finger_contact=False):
        # Sample each short joint-space chord. Runtime also monitors each
        # joint's segment bounds; this is not a proof of the SDK interpolator.
        steps = max(2, int(np.ceil(np.max(np.abs(np.asarray(b)-a))/.008)))
        for u in np.linspace(0, 1, steps+1):
            self.check_state((1-u)*np.asarray(a)+u*np.asarray(b), width,
                             allow_finger_contact=allow_finger_contact)

    def solve_pose(self, position, quaternion, seed, *, cancelled=None):
        if cancelled is not None and cancelled():
            raise PickRejected('逆解已取消')
        solved = (self.ik.solve(position, quaternion, qpos_seed=seed) if cancelled is None else
                  self.ik.solve(position, quaternion, qpos_seed=seed, cancelled=cancelled))
        if solved.success:
            return solved
        # Clipping a DLS step at a wrist bound can stall although the other
        # joints can still satisfy the pose. Solve the SAME pose and bounds
        # with a bounded trust-region method rather than expanding a limit.
        scratch = self.mj.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        target_r = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        position = np.asarray(position)
        def residual(q):
            if cancelled is not None and cancelled():
                raise PickRejected('有界逆解已取消')
            scratch.qpos[self.indices] = q
            self.mj.mj_kinematics(self.model, scratch)
            actual_r = scratch.site_xmat[self.tcp_id].reshape(3, 3)
            rot_error = Rotation.from_matrix(target_r @ actual_r.T).as_rotvec()
            return np.r_[scratch.site_xpos[self.tcp_id]-position,
                         self.ik.config.orientation_weight*rot_error]
        fit = least_squares(residual,
                            np.clip(seed, self.nominal_command_lower, self.nominal_command_upper),
                            bounds=(self.nominal_command_lower, self.nominal_command_upper),
                            max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(fit.x)
        d = float(np.linalg.norm(error[:3]))
        angle = float(np.linalg.norm(error[3:])/self.ik.config.orientation_weight)
        success = d <= self.ik.config.position_tolerance and angle <= self.ik.config.orientation_tolerance
        self.bounded_ik_fallbacks += 1
        full = scratch.qpos.copy(); full[self.indices] = fit.x
        return IKResult(fit.x.copy(), full, success, fit.nfev, d, angle,
                        f'bounded IK: position error={d:.6g}m, rotation error={angle:.6g}rad')

    def segment(self, q, start_tcp, end_tcp, width, allow_contact=False):
        distance, angle = transform_distance(start_tcp, end_tcp)
        count = max(1, int(np.ceil(distance/.008)), int(np.ceil(angle/2.)))
        rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([start_tcp[:3, :3], end_tcp[:3, :3]])))
        path = []
        for u in np.linspace(0, 1, count+1)[1:]:
            position = start_tcp[:3, 3]*(1-u) + end_tcp[:3, 3]*u
            xyzw = rotations(u).as_quat(); wxyz = xyzw[[3, 0, 1, 2]]
            solved = self.solve_pose(position, wxyz, q)
            if not solved.success:
                raise PickRejected(f'当前路径逆解未通过误差门槛：{solved.message}')
            target = solved.joint_positions
            if np.max(np.abs(target-q)) > .20:
                raise PickRejected('局部逆解关节跳变过大，停止规划')
            split = max(1, int(np.ceil(np.max(np.abs(target-q))/.035)))
            for next_q in np.linspace(q, target, split+1)[1:]:
                self.check_edge(q, next_q, width, allow_finger_contact=allow_contact)
                path.append(next_q.tolist()); q = next_q
        return q, path

    def plan(self, state, observation, cfg):
        self._pick_trajectory_cache.clear()
        cfg.validate(); self.check_feedback(state)
        start = self.tcp().copy()
        grasp = np.asarray(observation['grasp'], dtype=float)
        alternate = grasp.copy()
        # Stock parallel jaws have the same contact geometry after a local-Z
        # half-turn. Prefer the smaller wrist rotation, validating the entire
        # path for each candidate, including the asymmetric arm/gripper base.
        alternate[:3, :3] = alternate[:3, :3] @ np.diag([-1., -1., 1.])
        candidates = sorted([(0, grasp), (180, alternate)],
                            key=lambda item: transform_distance(start, item[1])[1])
        failures = []
        for yaw, candidate in candidates:
            try:
                plan = self._plan_grasp(state, candidate, cfg)
                plan['grasp_yaw_flip_deg'] = yaw
                plan['grasp_candidates_rejected'] = failures
                return plan
            except PickRejected as error:
                failures.append({'yaw_flip_deg': yaw, 'error': str(error), 'details': error.details})
        raise PickRejected('两个等价夹取方向均未通过：'+'；'.join(
            f'{item["yaw_flip_deg"]}°：{item["error"]}' for item in failures),
            details={'kind': 'grasp_candidates_rejected', 'candidates': failures})

    def _plan_grasp(self, state, grasp, cfg):
        self.set_state(state['joints'], state['width_m'])
        start = self.tcp().copy()
        clearance = max(start[2, 3]+.03, grasp[2, 3]+.10)
        if clearance > .70 or np.linalg.norm(start[:3, 3]-grasp[:3, 3]) > .75:
            raise PickRejected('当前位姿到目标超出本测试路径范围')
        raised = start.copy(); raised[2, 3] = clearance
        # The pregrasp height belongs to the OBJECT, not to a possibly high
        # starting pose. Forcing the wrist to traverse the whole belt at the
        # initial height can unnecessarily push a joint against its limit.
        # The interpolated approach still receives every existing edge and
        # collision check; no waypoint is accepted merely for being lower.
        above = grasp.copy(); above[2, 3] += .10
        lifted = grasp.copy(); lifted[2, 3] += cfg.lift_m
        # No home waypoint. First state is exactly the received q7.
        q = np.array(state['joints']); origin_q = q.copy()
        try:
            for opening in np.linspace(state['width_m'], cfg.open_width_m, 10):
                self.check_state(q, opening)
        except PickRejected as error:
            raise PickRejected(f'当前位姿检查：{error}', details=error.details) from error
        stages = []
        for name, a, b, contact in [('raise_current', start, raised, False),
                                   ('above_object', raised, above, False),
                                   ('descend', above, grasp, True),
                                   ('lift', grasp, lifted, True),
                                   ('lower', lifted, grasp, True),
                                   ('retreat', grasp, above, True)]:
            try:
                stage_start = q.copy()
                q, path = self.segment(q, a, b, cfg.open_width_m, contact)
                stage = {'name': name, 'joints_rad': path, 'target_tcp': b.tolist()}
                trajectory, samples = prepare_stage_trajectory(self, stage_start, stage, cfg)
                stage['trajectory'] = {
                    'algorithm': 'cubic_joint_path_with_quintic_time_scaling',
                    'duration_s': trajectory.duration,
                    'control_hz': cfg.control_hz,
                    'joint_velocity_limit_rad_s': cfg.joint_velocity_limit,
                    'joint_acceleration_limit_rad_s2': cfg.joint_acceleration_limit,
                    'validation_samples': len(samples),
                    'validation_max_joint_step_rad': .004,
                    'arrival_wait': 'stage_endpoint_only',
                }
            except PickRejected as error:
                raise PickRejected(f'规划阶段 {name}：{error}', details=error.details) from error
            stages.append(stage)
        # Check the observed object's vertical swept points as well as the arm.
        # Support-plane contact is intentional at the zero-height endpoint.
        if len(self.target) and len(self.scene):
            for dz in np.linspace(.012, cfg.lift_m, max(2, int(cfg.lift_m/.004))):
                distance, _ = self.tree.query(self.target + [0, 0, dz])
                if np.min(distance) < .008:
                    raise PickRejected('抬起工件的扫掠体靠近可见障碍')
        return {'start_joints_rad': origin_q.tolist(), 'start_flange_m_rad': state['pose'],
                'start_tcp': start.tolist(), 'grasp_tcp': grasp.tolist(), 'stages': stages,
                'flange_to_tcp_nominal': self.flange_to_tcp.tolist(),
                'joint_command_limits_rad': np.column_stack((self.command_lower, self.command_upper)).tolist(),
                'bounded_ik_fallbacks': self.bounded_ik_fallbacks,
                'gripper': 'user-confirmed stock AGX; model nominal TCP, not physical TCP calibration',
                'source': 'fresh current state -> vertical clearance -> above object -> grasp',
                'path_checks': 'joint limits, sampled IK and smoothed curve, model self collision, visible point cloud'}


class PickCycle:
    def __init__(self, hardware, model, cfg, emit, *, clock=time.monotonic, sleep=time.sleep,
                 heartbeat=lambda: None):
        cfg.validate(); self.hw, self.model, self.cfg, self.emit = hardware, model, cfg, emit
        self.clock, self.sleep = clock, sleep
        self.heartbeat = heartbeat
        self.sent = False; self.held_width = None; self.force_lost_since = None

    def check_holding(self, state):
        if state['width_m'] < .003 or abs(state['width_m']-self.held_width) > .003:
            raise PickRejected('夹持开口异常；保持夹爪并停止机械臂', details={
                'reason': 'grip_width_out_of_range',
                'width_m': float(state['width_m']), 'confirmed_width_m': float(self.held_width),
                'width_change_m': float(state['width_m']-self.held_width),
                'minimum_width_m': .003, 'maximum_width_change_m': .003,
                'force_n': float(state['force_n'])})
        if state['force_n'] < self.cfg.min_force_n:
            if self.force_lost_since is None: self.force_lost_since = self.clock()
            if self.clock()-self.force_lost_since > .25:
                raise PickRejected('夹持力持续丢失，停止机械臂')
        else:
            self.force_lost_since = None

    def wait(self, predicate, *, timeout=6, stable=.15, holding=False, anchor=None, heartbeat=None):
        until = self.clock()+timeout; since = None; last = None
        while self.clock() < until:
            self.heartbeat()
            if heartbeat is not None:
                heartbeat()
            s = self.hw.snapshot(); self.emit('feedback', s)
            if anchor is not None:
                d, a = transform_distance(pose_xyz_rpy(s['pose']), pose_xyz_rpy(anchor))
                if d > .002 or a > .7:
                    raise PickRejected('夹爪动作/停留期间机械臂发生偏移')
            if holding:
                self.check_holding(s)
            stamps = (mapped_stamp(s, 'pose'), mapped_stamp(s, 'gripper'))
            if last is None or all(a > b for a, b in zip(stamps, last)):
                last = stamps
                if predicate(s):
                    if since is None: since = self.clock()
                    if self.clock()-since >= stable: return s
                else: since = None
            self.sleep(.02)
        raise PickRejected('反馈未按时到位；停止，不继续后续动作')

    def follow(self, stage, holding, *, trajectory=None, reference_points=None):
        self.emit('stage', {'name': stage['name']})
        before = self.hw.snapshot(); start = np.asarray(before['joints'])
        self.model.check_feedback(before)
        if trajectory is None:
            trajectory, reference_points = prepare_stage_trajectory(self.model, start, stage, self.cfg)
        elif reference_points is None:
            reference_points = trajectory.verification_samples(max_joint_step=.004)
        reference_points = np.asarray(reference_points)
        expected_start = trajectory.sample(0.)
        if np.max(np.abs(start-expected_start)) > .005:
            raise PickRejected('实际关节已偏离阶段起点')
        path_tree = cKDTree(reference_points)

        def check_path(state):
            q = np.asarray(state['joints'])
            if q.shape != (7,) or not np.isfinite(q).all():
                raise PickRejected('关节反馈无效')
            if hasattr(self.model, 'command_lower') and (
                    np.any(q < self.model.command_lower-1e-10) or
                    np.any(q > self.model.command_upper+1e-10)):
                bad = np.flatnonzero((q < self.model.command_lower-1e-10) |
                                     (q > self.model.command_upper+1e-10))
                violations = [
                    {'joint': f'joint{i+1}', 'angle_rad': float(q[i]),
                     'lower_rad': float(self.model.command_lower[i]),
                     'upper_rad': float(self.model.command_upper[i])} for i in bad]
                text = '；'.join(
                    f'{v["joint"]}={np.rad2deg(v["angle_rad"]):.6f}°，'
                    f'允许[{np.rad2deg(v["lower_rad"]):.6f}, '
                    f'{np.rad2deg(v["upper_rad"]):.6f}]°' for v in violations)
                raise PickRejected('实际关节超出规划限位：'+text, details={
                    'kind': 'feedback_joint_limit', 'stage': stage['name'],
                    'violations': violations, 'joints_rad': q.tolist()})
            distance, _ = path_tree.query(q, p=np.inf)
            if distance > .015:
                raise PickRejected('关节运动偏离已检查的平滑路径')
            return q

        period = 1./self.cfg.control_hz
        started = next_deadline = self.clock()
        last_send = None; previous_target = expected_start
        command_count = 0; max_gap = 0.
        while True:
            remaining = next_deadline-self.clock()
            if remaining > 0: self.sleep(remaining)
            self.heartbeat()
            state = self.hw.snapshot(); self.emit('feedback', state)
            q = check_path(state)
            self.model.check_feedback(state)
            if holding: self.check_holding(state)
            dispatch_time = self.clock()
            if dispatch_time-next_deadline > .05+1e-9:
                raise PickRejected('轨迹发送调度延迟超过50 ms，停止运动', details={
                    'kind': 'trajectory_deadline_missed', 'stage': stage['name'],
                    'lateness_s': dispatch_time-next_deadline,
                    'control_hz': self.cfg.control_hz})
            if last_send is not None:
                gap = dispatch_time-last_send
                max_gap = max(max_gap, gap)
                if gap > .05+1e-9:
                    raise PickRejected('轨迹发送间隔超过50 ms，停止运动', details={
                        'kind': 'trajectory_deadline_missed', 'stage': stage['name'],
                        'gap_s': gap, 'control_hz': self.cfg.control_hz})
            elapsed = min(dispatch_time-started, trajectory.duration)
            target = trajectory.sample(elapsed)
            if hasattr(self.model, 'command_lower') and (
                    np.any(target < self.model.command_lower-1e-10) or
                    np.any(target > self.model.command_upper+1e-10)):
                raise PickRejected('平滑轨迹指令超出规划限位')
            if np.max(np.abs(q-target)) > .055:
                raise PickRejected('实际关节跟随平滑轨迹滞后过大')
            if np.max(np.abs(target-previous_target)) > .035:
                raise PickRejected('定时轨迹相邻指令跨度过大')
            self.hw.move_joints(target.tolist())
            self.emit('joint_command', {'stage': stage['name'],
                      'joints_rad': target.tolist(), 'trajectory_time_s': elapsed,
                      'dispatch_monotonic_s': dispatch_time})
            command_count += 1; previous_target = target; last_send = dispatch_time
            if elapsed >= trajectory.duration: break
            # Use deadlines without accumulating ordinary work time. If a
            # tick was late, resume one period later; never send catch-up bursts.
            next_deadline += period
            now = self.clock()
            if next_deadline <= now: next_deadline = now+period
        measured_duration = last_send-started
        self.emit('trajectory_timing', {
            'stage': stage['name'], 'target_hz': self.cfg.control_hz,
            'commands': command_count, 'duration_s': measured_duration,
            'observed_hz': (command_count-1)/measured_duration if measured_duration > 0 else 0.,
            'max_gap_s': max_gap,
        })
        end = trajectory.sample(trajectory.duration)
        def arrived(state):
            q = check_path(state)
            return mapped_stamp(state, 'pose') > mapped_stamp(before, 'pose') and np.max(np.abs(q-end)) < .003
        self.wait(arrived, timeout=self.cfg.move_timeout_s, stable=.1, holding=holding)
        final = self.hw.snapshot(); self.model.check_feedback(final)
        actual = pose_xyz_rpy(final['pose']) @ self.model.flange_to_tcp
        d, a = transform_distance(actual, np.array(stage['target_tcp']))
        if d > .002 or a > .7:
            raise PickRejected(f'末端未到位：{d*1000:.2f} mm / {a:.2f}°')
        return final

    def run(self, plan, *, execute=False, deadline=None):
        if not execute:
            return {'status': 'plan_only', 'motion_commands_sent': False}
        if deadline is not None and not np.isfinite(deadline):
            raise ValueError('cycle deadline must be finite')
        s = self.hw.snapshot()
        if np.max(np.abs(np.array(s['joints'])-plan['start_joints_rad'])) > .005:
            raise PickRejected('规划期间机械臂已移动，请重新运行获取当前状态')
        if s['force_n'] >= self.cfg.min_force_n:
            raise PickRejected('夹爪可能已经夹持物体，不能开始自动张开')
        # The planner already checked opening, each path and the smoothed
        # curves. Reconstruct their timing only; do not repeat scene collision
        # queries or opening sweeps in the execution path.
        prepared = []; reused = 0
        cache = getattr(self.model, '_pick_trajectory_cache', {})
        start = np.asarray(plan['start_joints_rad'])
        for stage in plan['stages']:
            cached = cache.get(stage_trajectory_key(start, stage, self.cfg))
            if cached is None:
                trajectory, samples = build_stage_trajectory(start, stage, self.cfg)
            else:
                trajectory, samples = cached; reused += 1
            prepared.append((trajectory, samples))
            start = trajectory.sample(trajectory.duration)
        self.emit('trajectory_ready', {
            'collision_validation': 'during_planning_only',
            'stages_reused_from_planning': reused,
            'control_hz': self.cfg.control_hz,
            'stages': [{'name': stage['name'], 'duration_s': trajectory.duration}
                       for stage, (trajectory, _) in zip(plan['stages'], prepared)],
            'joint_velocity_limit_rad_s': self.cfg.joint_velocity_limit,
            'joint_acceleration_limit_rad_s2': self.cfg.joint_acceleration_limit,
        })
        self.heartbeat()
        current = self.hw.snapshot()
        if np.max(np.abs(np.asarray(current['joints'])-plan['start_joints_rad'])) > .005:
            raise PickRejected('执行准备期间机械臂已移动，请重新运行')
        if abs(current['width_m']-s['width_m']) > .002:
            raise PickRejected('执行准备期间夹爪开口改变，请重新运行')
        self.model.check_feedback(current)
        if current['force_n'] >= self.cfg.min_force_n:
            raise PickRejected('执行准备后夹爪可能已经夹持物体，不能自动张开')
        if deadline is not None and self.clock()+sum(t.duration for t, _ in prepared) > deadline:
            raise PickRejected('执行前全周期时间下界已超过预算；未发送动作')
        s = current
        try:
            self.sent = True
            controller_limits = self.hw.prepare_motion(self.cfg.speed_percent)
            if controller_limits is not None:
                self.emit('controller_motion_limits', controller_limits)
            current = self.hw.snapshot()
            if (np.max(np.abs(np.asarray(current['joints'])-s['joints'])) > .005 or
                    abs(current['width_m']-s['width_m']) > .002 or current['force_n'] >= self.cfg.min_force_n):
                raise PickRejected('读取运动参数期间状态改变；未张爪')
            if deadline is not None and self.clock()+sum(t.duration for t, _ in prepared) > deadline:
                raise PickRejected('读取运动参数后全周期时间下界已超过预算；未张爪')
            s = current
            self.emit('open_initial', {'width_m': self.cfg.open_width_m})
            self.hw.grip(self.cfg.open_width_m, 1.)
            self.wait(lambda x: abs(x['width_m']-self.cfg.open_width_m)<.002,
                      anchor=s['pose'])
            for stage, (trajectory, samples) in zip(plan['stages'], prepared):
                name = stage['name']
                state = self.follow(stage, holding=name in ('lift', 'lower'),
                                    trajectory=trajectory, reference_points=samples)
                if name == 'descend':
                    self.emit('grasp_arrival', state)
                    self.emit('close', {'force_n': self.cfg.force_n})
                    self.hw.grip(0., self.cfg.force_n)
                    held = self.wait(lambda x: mapped_stamp(x, 'gripper')>mapped_stamp(state, 'gripper') and
                                     .003<x['width_m']<self.cfg.open_width_m-.004 and
                                     x['force_n']>=self.cfg.min_force_n,
                                     stable=.4, anchor=state['pose'])
                    self.held_width = held['width_m']
                    self.emit('grasp_contact', held)
                if name == 'lift':
                    self.wait(lambda x: True, timeout=2, stable=.8, holding=True, anchor=state['pose'])
                if name == 'lower':
                    # Settled back at observed support position BEFORE release.
                    self.wait(lambda x: True, stable=.3, holding=True, anchor=state['pose'])
                    self.emit('open_release', {'width_m': self.cfg.open_width_m})
                    self.hw.grip(self.cfg.open_width_m, 1.)
                    self.wait(lambda x: abs(x['width_m']-self.cfg.open_width_m)<.002,
                              anchor=state['pose'])
            return {'status': 'cycle_completed_from_feedback', 'motion_commands_sent': True,
                    'note': '夹爪反馈不能独立证明物体已抬起；请核对现场及录像。'}
        except BaseException:
            try:
                self.hw.stop()  # Keep gripper closed on an interrupted carry.
                self.emit('stopped', {'automatic_gripper_release': False})
            except BaseException as stop_error:
                print(f'软件停机失败，请使用实体急停：{stop_error}', flush=True)
            raise

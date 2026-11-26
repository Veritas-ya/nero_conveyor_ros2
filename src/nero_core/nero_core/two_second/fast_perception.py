"""Reuse the historical same-exposure, target-only Fast stereo pipeline.

The native detector still selects the object. Reconstruction runs in the
geometry worker, never in the robot sender, and retains native scene depth.
"""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np

from mujoco_conveyor_sorting.runtime.fast_stereo import FastStereoProcessBackend
from mujoco_conveyor_sorting.runtime.depth_diagnostics import measured_stage
from mujoco_conveyor_sorting.runtime.grasp_depth import (TARGET_GEOMETRY_FIELDS, _points_sha, fast_target_pixels,
    reconstruct_target_depth, refine_grasp_observation)
from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
from .fast_quality import filter_reconstruction
from .fast_cloud_delivery import FastCloudDelivery
from .grasp import MIN_VISIBLE_POINTS


def validate(raw):
    if not isinstance(raw.setdefault('fast_stereo_enabled', False), bool):
        raise ValueError('fast_stereo_enabled 必须为布尔值')
    if raw.setdefault('fast_cloud_delivery_mode', 'quality_gated') not in ('quality_gated', 'all_reconstructed', 'direct_to_planner'):
        raise ValueError('fast_cloud_delivery_mode 须为 quality_gated、all_reconstructed 或 direct_to_planner')
    for key, default, upper in (('fast_min_native_coverage_fraction', .80, 1.),
                                ('fast_max_center_shift_m', .006, float('inf'))):
        value = raw.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f'{key} 配置值无效')


def camera_settings(config):
    result = dict(config.raw['camera'])
    if config.raw.get('fast_stereo_enabled', False):
        result.update(capture_stereo_ir=True, enable_frame_sync=True)
    return result


def _direct_observation(observation, backend, delivery):
    """Pass this request's delivered points and ordered pixel map straight to OBB generation."""
    frame = observation['frame']
    native = observation.get('native_object_points', observation['object_points'])
    mask = np.asarray(observation['mask'], dtype=bool)
    meta = dict(applied=False, requested_source='fast_foundationstereo', source='unavailable', grasp_input='direct_to_planner',
                capture_header=asdict(frame.header), timestamp_metadata=frame.timestamp_metadata,
                native_target_points=len(native), native_target_points_sha256=_points_sha(native),
                scene_source='native_device_depth', native_depth_modified=False,
                new_capture=False, hidden_surface_added=False, input_resized=False, timings_s={},
                quality_check=dict(evaluated=False, accepted=None, enforced_as_delivery_gate=False))
    out = dict(observation, grasp_depth_refinement=meta)
    try:
        stage = partial(measured_stage, timings=meta['timings_s'])
        depth = reconstruct_target_depth(frame, mask, backend, meta, stage)
        points, pixels = fast_target_pixels(frame, depth, mask)
        if not len(points):
            raise ValueError('no_finite_positive_fast_depth_on_target')
        if not np.isfinite(points).all():
            raise ValueError('nonfinite_fast_points_in_base_frame')
        cloud = delivery.publish(frame, points, pixels, meta['worker_artifact_prefix'])
        meta.update(cloud_delivery=cloud, source=cloud['source'], refined_target_points=len(points),
                    refined_target_pixels=len(pixels), refined_target_points_sha256=_points_sha(points))
        if len(points) < MIN_VISIBLE_POINTS:
            raise ValueError(f'工件点云有 {len(points)} 点，抓姿 OBB 至少需要 {MIN_VISIBLE_POINTS} 点')
        grasp_mask = np.zeros(mask.shape, dtype=bool)
        grasp_mask.flat[pixels] = True
        grasp_mask.setflags(write=False)
        # Native geometry remains attributable to its source. The production
        # OBB step fills in the reconstructed dimensions and chosen grasp.
        native_geometry = observation.get('native_target_geometry')
        measured = {key: out.pop(key) for key in TARGET_GEOMETRY_FIELDS if key in out}
        for key in ('obb', 'grasp_point_keep_mask', 'grasp_target_cloud_filter', 'depth_separation'):
            out.pop(key, None)
        out.update(object_points=points, grasp_target_mask=grasp_mask, native_object_points=native,
                   native_target_geometry=measured if native_geometry is None else native_geometry,
                   target_geometry_source=cloud['source'])
        meta.update(applied=True, reason='applied', geometry_seed_source='delivered_fast_target_cloud')
    except Exception as error:
        meta.update(reason=str(error), error_type=type(error).__name__)
    return out


class FastTargetPerception:
    """One prewarmed backend per run. A failed refinement rejects this target."""
    def __init__(self, config, output, *, backend=None, retain_artifacts=True):
        self.config = config
        self.output = Path(output)
        self.backend = backend or FastStereoProcessBackend(self.output, request_timeout_s=120., retain_artifacts=retain_artifacts)
        self.quality_filter = partial(filter_reconstruction,
            min_native_coverage=config.raw.get('fast_min_native_coverage_fraction', .80),
            max_center_shift_m=config.raw.get('fast_max_center_shift_m', .006))
        mode = config.raw.get('fast_cloud_delivery_mode', 'quality_gated')
        self.direct_to_planner = mode == 'direct_to_planner'
        self.cloud_delivery = (FastCloudDelivery(self.output/'clouds' if retain_artifacts else None)
            if mode in ('all_reconstructed', 'direct_to_planner') else None)
        self.event_sink = None
        self.requests = 0
        self.completed = 0
        self.last = None

    def start(self):
        try:
            self.backend.start()
            self.backend.warmup((384, 640))
            # First-use CUDA compilation belongs to startup, before any devices.
            # Once warm, cap an individual reconstruction independently.
            self.backend.request_timeout_s = 2.
            if self.cloud_delivery is not None:
                self.cloud_delivery.emit = self.event_sink
                self.cloud_delivery.start()
        except BaseException:
            self.backend.close()
            raise
        return self

    def close(self):
        try:
            self.backend.close()
        finally:
            if self.cloud_delivery is not None:
                self.cloud_delivery.close()

    def refine(self, detection, frame, d2c, *, robot_model=None, robot_state=None, surface=None):
        self.requests += 1
        started = time.monotonic()
        if hasattr(self.backend,'health') and not self.backend.health().get('warmed'):
            raise TargetNotReady('Fast预热进程不可用，请检查worker.log后重新启动感知',
                                 reason='fast_stereo_worker_unavailable')
        observation = dict(detection, frame=frame, depth_to_color_cv=d2c, scene_state=robot_state)
        if self.direct_to_planner:
            refined = _direct_observation(observation, self.backend, self.cloud_delivery)
        else:
            refined = refine_grasp_observation(observation, self.backend, robot_model=robot_model,
                                              output_directory=None, quality_filter=self.quality_filter,
                                              compact_native_geometry=True,
                                              publish_cloud=self.cloud_delivery.publish if self.cloud_delivery else None,
                                              enforce_quality=self.cloud_delivery is None)
        metadata = refined['grasp_depth_refinement']
        metadata['velocity_source'] = self.config.raw.get('velocity_source','native_nero7_rgbd')
        report = dict(metadata, perception_wall_s=time.monotonic()-started)
        self.last = report
        cloud = metadata.get('cloud_delivery')
        if cloud is not None and self.event_sink is not None and not self.direct_to_planner:
            quality = metadata.get('quality_check', {})
            self.event_sink('fast_cloud_quality', cloud_id=cloud['cloud_id'], frame_sequence=frame.header.sequence,
                quality_passed=quality.get('accepted'), reasons=quality.get('reasons', []),
                enforced_as_delivery_gate=False,
                retained_points=quality.get('retained_points'), native_coverage_fraction=quality.get('native_coverage_fraction'),
                center_shift_m=quality.get('center_shift_m'), geometry_applied=metadata['applied'],
                geometry_reason=metadata.get('reason'))
        if not metadata.get('applied'):
            if cloud is not None:
                raise TargetNotReady('Fast点云已交付，抓姿几何未通过：'+str(metadata.get('reason')),
                    reason='fast_grasp_geometry_rejected', details={'fast_stereo':report})
            if self.cloud_delivery is not None and self.event_sink is not None:
                self.event_sink('fast_cloud_unavailable', frame_sequence=frame.header.sequence,
                                reason=metadata.get('cloud_unavailable_reason', metadata.get('reason')))
            raise TargetNotReady('Fast工件重建未成功，本帧不交付抓姿：'+str(metadata.get('reason')),
                                 reason='fast_stereo_reconstruction_failed', details={'fast_stereo':report})
        # Recheck the reconstructed centre against the same detector boundary.
        # The historical refiner preserves native mask/scene, but changes target geometry.
        from .fixed_freeze import fixed_freeze_detection_surface
        from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, near_camera_belt_clearance
        region = detection['belt_region']
        transform = np.asarray(region['plane_to_base'])
        points = np.asarray(refined['object_points'])
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise TargetNotReady('Fast重建工件点云无效',reason='fast_stereo_target_height')
        local = (points-transform[:3,3]) @ transform[:3,:3]
        center = np.median(local, axis=0)
        native_surface = surface or getattr(robot_model, 'measured_belt_surface', None)
        hull = region.get('hull_xy_m')
        if isinstance(native_surface, BeltSurface):
            surface = fixed_freeze_detection_surface(native_surface)
        elif hull is not None:
            surface = BeltSurface(transform,np.asarray(region['bounds_xy_m']),np.asarray(hull,np.float32),detection['plane'])
        else:
            surface = None
        if surface is not None:
            distance = cv2.pointPolygonTest(surface.hull_xy,tuple(map(float,center[:2])),True)
            margin = self.config.raw.get('perception_belt_edge_margin_m',.025)
            near = self.config.raw.get('perception_belt_near_camera_margin_m',0.)
            if distance <= margin or (near > 0 and near_camera_belt_clearance(
                    surface,center[:2],frame.camera_to_world_cv[:3,3],self.config.direction) <= near+1e-9):
                raise TargetNotReady('Fast重建目标中心超出带边检测范围',reason='fast_stereo_belt_edge')
        if np.max(local[:,2]) > self.config.raw['maximum_target_height_m']:
            raise TargetNotReady('Fast重建工件高度或点云无效',reason='fast_stereo_target_height')
        if self.direct_to_planner:
            refined['height'] = float(np.quantile(local[:, 2], .98))
        refined['fast_stereo'] = report
        self.completed += 1
        return refined

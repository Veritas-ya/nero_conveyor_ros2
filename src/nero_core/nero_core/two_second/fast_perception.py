"""Reuse the historical same-exposure, target-only Fast stereo pipeline.

The native detector still selects the object. Reconstruction runs in the
geometry worker, never in the robot sender, and retains native scene depth.
"""
from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np

from mujoco_conveyor_sorting.runtime.fast_stereo import FastStereoProcessBackend
from mujoco_conveyor_sorting.runtime.grasp_depth import refine_grasp_observation
from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
from .fast_quality import filter_reconstruction
from .fast_cloud_delivery import FastCloudDelivery


def validate(raw):
    if not isinstance(raw.setdefault('fast_stereo_enabled', False), bool):
        raise ValueError('fast_stereo_enabled 必须为布尔值')
    if raw.setdefault('fast_cloud_delivery_mode', 'quality_gated') not in ('quality_gated','all_reconstructed'):
        raise ValueError('fast_cloud_delivery_mode 须为 quality_gated 或 all_reconstructed')
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


class FastTargetPerception:
    """One prewarmed backend per run. A failed refinement rejects this target."""
    def __init__(self, config, output, *, backend=None, retain_artifacts=True):
        self.config = config
        self.output = Path(output)
        self.backend = backend or FastStereoProcessBackend(self.output, request_timeout_s=120., retain_artifacts=retain_artifacts)
        self.quality_filter = partial(filter_reconstruction,
            min_native_coverage=config.raw.get('fast_min_native_coverage_fraction', .80),
            max_center_shift_m=config.raw.get('fast_max_center_shift_m', .006))
        self.cloud_delivery = (FastCloudDelivery(self.output/'clouds' if retain_artifacts else None)
            if config.raw.get('fast_cloud_delivery_mode') == 'all_reconstructed' else None)
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
        refined = refine_grasp_observation(observation, self.backend, robot_model=robot_model,
                                          output_directory=None, quality_filter=self.quality_filter,
                                          compact_native_geometry=True,
                                          publish_cloud=self.cloud_delivery.publish if self.cloud_delivery else None,
                                          enforce_quality=self.cloud_delivery is None)
        metadata = refined['grasp_depth_refinement']
        metadata['velocity_source'] = self.config.raw.get('velocity_source','native_nero7_rgbd')
        self.last = dict(metadata, perception_wall_s=time.monotonic()-started)
        cloud = metadata.get('cloud_delivery')
        if cloud is not None and self.event_sink is not None:
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
                    reason='fast_grasp_geometry_rejected', details={'fast_stereo':self.last})
            if self.cloud_delivery is not None and self.event_sink is not None:
                self.event_sink('fast_cloud_unavailable', frame_sequence=frame.header.sequence,
                                reason=metadata.get('cloud_unavailable_reason', metadata.get('reason')))
            raise TargetNotReady('Fast工件重建未成功，本帧不交付抓姿：'+str(metadata.get('reason')),
                                 reason='fast_stereo_reconstruction_failed', details={'fast_stereo':self.last})
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
        refined['fast_stereo'] = self.last
        self.completed += 1
        return refined

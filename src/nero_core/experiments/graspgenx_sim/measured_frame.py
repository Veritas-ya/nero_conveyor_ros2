"""Translate one simulated RGB-D/Nero7 frame into the measured planning input.

Only pixels, camera/belt calibration, Nero7 reports and robot encoder feedback
are accepted. Simulation object bodies, contacts and belt speed are not inputs.
The caller supplies a qualified identity and the original velocity report.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .cached_projection import CachedDepthProjector
from mujoco_conveyor_sorting.runtime.types import RGBDFrame, CameraIntrinsics, SampleHeader, TargetIdentity


@dataclass(frozen=True)
class MeasuredFrame:
    detection: dict
    velocity_measurement: dict
    capture_s: float
    sequence: int
    clock_epoch: int


class MeasuredFrameBuilder:
    """Deproject the full image once, cache immutable camera rays across frames."""
    def __init__(self):
        self.projector = CachedDepthProjector()

    def build(self, visual_frame, *, task_id, identity, world_to_base, belt_region, include_scene=True,
              allow_unqualified_speed=False):
        if not isinstance(identity, TargetIdentity):
            raise ValueError('confirmed task identity required')
        meta = visual_frame['nero_capture']
        stamp = float(visual_frame['timestamp'])
        seq, epoch = int(meta['sequence']), int(meta['clock_epoch'])
        if (identity.clock_epoch != epoch or abs(stamp-float(meta['device_time_s'])) > 1e-9):
            raise ValueError('frame identity or device time mismatch')
        item = meta['measurements'].get(int(task_id))
        if item is None or item['actual_peer_track_id'] != identity.track_id or item['track_generation'] != identity.track_generation:
            raise ValueError('Nero7 task/peer/generation mismatch')
        velocity = deepcopy(item['velocity_measurement'])
        qualified = velocity.get('speed_valid') and velocity.get('motion_source') == 'nero7_xy_kalman'
        if not qualified and not allow_unqualified_speed:
            raise ValueError('qualified Nero7 XY Kalman speed required')
        if qualified and (velocity.get('track_id') != identity.track_id or
                velocity.get('track_generation') != identity.track_generation or
                velocity.get('clock_epoch') != epoch or velocity.get('geometry_sequence') != seq or
                abs(float(velocity.get('geometry_host_capture_timestamp'))-stamp) > 1e-6 or
                abs(float(velocity.get('geometry_device_timestamp'))-stamp) > 1e-6):
            raise ValueError('qualified speed does not belong to this geometry capture')
        if not qualified:
            velocity = None  # Geometry is usable for preparation; never invent a velocity.
        sample = visual_frame['capture_robot_feedback']
        if abs(float(sample['sample_time_s'])-stamp) > 1e-9:
            raise ValueError('robot scene state must match exposure')
        q = np.asarray(sample['joints'], dtype=float)
        width = float(sample['width_m'])
        if q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(width):
            raise ValueError('invalid robot encoder state')
        depth = np.asarray(visual_frame['top_depth'], dtype=np.float32)
        labels = np.asarray(visual_frame['top_instance_labels'])
        rgb = np.asarray(visual_frame['top_rgb'], dtype=np.uint8)
        if depth.ndim != 2 or labels.shape != depth.shape or rgb.shape != (*depth.shape, 3):
            raise ValueError('RGB, depth and visual labels must share a pixel grid')
        calibration = visual_frame['top_camera']
        matrix = np.asarray(calibration['intrinsics'], dtype=float)
        T = np.asarray(world_to_base, dtype=float) @ np.asarray(calibration['camera_to_world_opencv'], dtype=float)
        intr = CameraIntrinsics(depth.shape[1], depth.shape[0], matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2])
        header = SampleHeader(stamp, float(meta['publication_time_s']), seq, stamp, epoch)
        frame = RGBDFrame(header=header, depth_m=depth, depth_intrinsics=intr, rgb=rgb,
            color_intrinsics=intr, camera_to_world_cv=T, serial_number='sim-render-camera',
            calibration_id='sim-native-pinhole', source='sim_rendered_rgbd', rgb_aligned_to_depth=True)
        valid = np.isfinite(depth) & (depth > 0.)
        mask = (labels == int(task_id)) & valid
        if not np.any(mask):
            raise ValueError('no measured target pixels')
        # On scene installation one full deprojection is shared by target and
        # background. Later target-only frames project only the measured mask;
        # they never repeat a full-frame projection or reduce the target mask.
        points = None
        if include_scene:
            points = self.projector.project(depth, matrix, T, camera_convention='opencv')
            target = points[mask]
        else:
            rows, cols = np.nonzero(mask)
            z = depth[rows, cols].astype(float)
            local = np.column_stack(((cols-matrix[0,2])*z/matrix[0,0],
                                     (rows-matrix[1,2])*z/matrix[1,1], z))
            target = local @ T[:3,:3].T + T[:3,3]
        plane = np.asarray(belt_region['plane_to_base'], dtype=float)
        if plane.shape != (4,4) or not np.isfinite(plane).all():
            raise ValueError('measured belt frame required')
        local = (target-plane[:3,3]) @ plane[:3,:3]
        height = float(np.max(local[:,2]))
        width_m = float(max(np.ptp(local[:,0]), np.ptp(local[:,1])))
        if height <= 0. or width_m <= 0.:
            raise ValueError('visible target has no measurable positive height/width')
        # Provisional anchor exists only to associate observations; the network
        # must replace it before any full-planner context can be constructed.
        grasp = np.eye(4); grasp[:3,3] = np.median(target, axis=0)
        detection = {'target_identity': identity, 'frame': frame, 'mask': mask,
            'grasp': grasp, 'object_points': target, 'height': height, 'width': width_m,
            'belt_region': deepcopy(belt_region), 'scene_state': {'joints': q.copy(), 'width_m': width},
            'target_geometry_source': 'sim_rendered_observed_depth_pending_network_grasp'}
        if include_scene:
            scene = valid & ~mask
            detection['scene_points'] = points[scene]
            detection['scene_pixel_indices'] = np.flatnonzero(scene)
        return MeasuredFrame(detection, velocity, stamp, seq, epoch)

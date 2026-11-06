"""Camera SDK input adapter for Nero7 visual speed and occlusion handling.

The peer detector, association and XY Kalman filter live in ``vendor``.  This
module replaces their ROS image/TF transport with native Gemini frames and the
local calibration, and selects the measured belt's image ROI.  It does not feed
our footprint centres or optical flow into the peer filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from vendor.nero7_speed_reference.nero7_perception.block_tracker import STATUS_LOST
from .nero7_occlusion_tracker import OcclusionAwareBlockTracker
from vendor.nero7_speed_reference.nero7_perception.conveyor_model import BeltModel, BeltPlane
from vendor.nero7_speed_reference.nero7_perception.detection import (
    detect_legacy, detect_blocks, load_detection_params,
)
from vendor.nero7_speed_reference.nero7_perception.geometry import CameraGeometry, check_transform
from vendor.nero7_speed_reference.nero7_perception.projection import project_cloud
from vendor.nero7_speed_reference.nero7_perception.speed_overlay import draw_speed_overlay


REFERENCE = Path(__file__).resolve().parents[2] / 'vendor/nero7_speed_reference'


def _compact_cv_distortion(coefficients):
    """Use the equivalent five-term model only when the rational terms are zero."""
    coefficients = tuple(coefficients)
    if len(coefficients) == 8 and all(value == 0.0 for value in coefficients[5:]):
        return coefficients[:5]
    return coefficients


@dataclass(frozen=True)
class Nero7TrackSnapshot:
    """Detached peer observations for binding one target across worker threads.

    All times here are camera source seconds. Observed and Kalman-predicted
    positions have separate stamps; prediction is not another observation.
    Tuples deliberately avoid mutable vendor Track/NumPy references.
    """
    track_id: int
    status: int
    observed_position_base: tuple[float, float, float]
    filtered_position_base: tuple[float, float, float]
    velocity_base_m_s: tuple[float, float, float] | None
    speed_valid: bool
    observation_source_time: float
    motion_source_time: float
    sample_count: int
    span_s: float
    speed_std_m_s: float
    velocity_covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    misses: int
    area_px: float
    confirmed: bool
    velocity_source: str = 'nero7_xy_kalman'
    legacy_velocity_base_m_s: tuple[float, float, float] | None = None
    speed_position_base_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Nero7FrameSnapshot:
    """One completed meter update, independent of its display selection."""
    source_time: float
    sequence: int
    clock_epoch: int
    track_generation: int
    current: bool
    tracks: tuple[Nero7TrackSnapshot, ...]
    reason: str


class SDKCameraGeometry(CameraGeometry):
    """Same RGB projection without OpenCV's unused per-point Jacobian.

    The 0/4/5-coefficient formula is shared with this project's existing
    ``tools/diagnose_real_perception.project_to_rgb``.  Unsupported distortion
    shapes retain the original peer implementation.  Points next to a rounded
    pixel boundary use OpenCV too, preserving the peer z-buffer's tie behavior.
    """

    def project(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        if (not self.D or not any(self.D) or len(self.D) not in (4, 5)
                or self.distortion_model not in ('plumb_bob', 'rational_polynomial')):
            return super().project(points)
        coefficients = np.zeros(5)
        coefficients[:len(self.D)] = self.D
        k1, k2, p1, p2, k3 = coefficients
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            reciprocal = np.ones(len(points))
            np.divide(1., points[:, 2], out=reciprocal, where=points[:, 2] != 0)
            x, y = points[:, 0] * reciprocal, points[:, 1] * reciprocal
            r2 = x*x + y*y
            r4, r6 = r2*r2, r2*r2*r2
            radial = 1 + k1*r2 + k2*r4 + k3*r6
            xy2 = 2*x*y
            u = (x*radial + p1*xy2 + p2*(r2 + 2*x*x))*self.K[0, 0] + self.K[0, 2]
            v = (y*radial + p1*(r2 + 2*y*y) + p2*xy2)*self.K[1, 1] + self.K[1, 2]
            boundary = np.abs((u-.5) - np.rint(u-.5)) < 1e-7
            boundary |= np.abs((v-.5) - np.rint(v-.5)) < 1e-7
            pixels = np.column_stack((u, v))
        fallback = boundary | ~np.isfinite(u) | ~np.isfinite(v)
        if fallback.any():
            pixels[fallback] = super().project(points[fallback])
        return pixels


@dataclass
class _CroppedCamera:
    camera: CameraGeometry
    u0: int
    v0: int
    width: int | None = None
    height: int | None = None

    def rays(self, u, v):
        return self.camera.rays(np.asarray(u)+self.u0, np.asarray(v)+self.v0)

    @property
    def frame_id(self):return self.camera.frame_id

    @property
    def K(self):
        value=self.camera.K.copy();value[:2,2]-=[self.u0,self.v0]
        return value

    def check_grid(self, image, frame_id):
        if image.shape[:2]!=(self.height,self.width) or frame_id!=self.frame_id:
            raise ValueError('Cropped registered RGB-D grid mismatch')

    def grid_rays(self):
        return self.camera.grid_rays()[self.v0:self.v0+self.height,self.u0:self.u0+self.width]


@dataclass
class _CroppedPlane:
    plane: BeltPlane
    full_depth: np.ndarray
    full_roi: list
    u0: int
    v0: int

    @property
    def coeffs(self):
        return self.plane.coeffs

    def fit(self, _depth, roi=None):
        # Preserve original pixels, RANSAC samples, coefficients and RNG state.
        return self.plane.fit(self.full_depth, roi=self.full_roi)

    def plane_z(self, u, v):
        return self.plane.plane_z(np.asarray(u)+self.u0, np.asarray(v)+self.v0)


class Nero7BeltVelocityMeter:
    """Run the peer's ``legacy_depth -> BlockTracker -> KalmanMotion`` chain.

    ``frame.camera_to_world_cv`` maps native depth optical coordinates to the
    robot base.  ``d2c`` maps native depth optical to color optical coordinates.
    Source device timestamps, rather than processing times, drive the filter.
    """

    def __init__(self, surface, window_s=.6, belt_direction_base=None, *, maximum_measured_speed_m_s=None,
                 detection_mode='legacy_depth', plane_mode='ransac3d', green_belt_extractor=None,
                 tracking_entry_coordinate_m=None, pointcloud_speed=False):
        self.surface = surface
        self.pointcloud_speed=bool(pointcloud_speed)
        self.parameters = load_detection_params(REFERENCE / 'upstream_perception.yaml')
        # The upstream six-frame bound expires after only 0.2 s at 30 Hz,
        # before its own 0.5 s lost timeout. Keep the original time limit and
        # allow enough missing frames for camera rates up to 120 Hz.
        self.parameters['tracker_lost_frames'] = max(60, self.parameters['tracker_lost_frames'])
        self.parameters['speed_window_s'] = float(window_s)
        if detection_mode not in ('legacy_depth','depth_plane','rgb','rgb_depth'):
            raise ValueError('Unknown Nero7 perception detection mode')
        self.parameters['detection_mode']=detection_mode
        self._fused=None
        if detection_mode!='legacy_depth':
            from .nero7_fused_detection import Nero7FusedPlane
            self._fused=Nero7FusedPlane(self.parameters,plane_mode,green_belt_extractor)
        if maximum_measured_speed_m_s is not None:
            maximum=float(maximum_measured_speed_m_s)
            if not np.isfinite(maximum) or maximum<=0.:
                raise ValueError('maximum measured speed must be finite and positive')
            # Explicit operating-range configuration, never a clipped or
            # fabricated measurement. Default source/filter parameters stay
            # byte-for-byte unchanged, including the original association gate.
            self.parameters['kalman_max_velocity_m_s']=maximum
        self._requested_direction = belt_direction_base
        if tracking_entry_coordinate_m is not None and not np.isfinite(tracking_entry_coordinate_m):
            raise ValueError('上游检测入口坐标须为有限米数')
        self.tracking_entry_coordinate_m=tracking_entry_coordinate_m
        self.tracking_rgb_roi=None
        self.candidate_filter=None
        self._direction = None
        self._tracker = None
        self._next_id = 0
        self._generation = 0
        self._camera = self._depth_camera = None
        self._camera_key = self._depth_key = None
        self._depth_rays = None
        self._color_rays = self._d2c_key = None
        self._native_crop_key = self._native_crop = None
        self.roi_px = None
        self.reset()

    def reset(self, *, calibration=False):
        """Clear measured history without reusing an earlier object's ID."""
        if self._tracker is not None:
            self._next_id = self._tracker._next_id
        self._tracker = None
        self._plane = BeltPlane(distance=self.parameters['plane_ransac_distance'],
                                min_ratio=self.parameters['plane_ransac_min_ratio'])
        self._frame_count = 0
        self._sequence = self._stamp = self._epoch = None
        self._selected_id = None
        self._detections = []
        self._candidate_depth_diagnostics = {}
        self._mask = None
        self._latest_snapshot = None
        self._point_speed = None
        self._generation += 1
        if calibration and self._fused is not None:self._fused.reset()

    @property
    def latest_snapshot(self) -> Nero7FrameSnapshot | None:
        """Immutable all-track report; only the meter owner calls ``update``.

        A worker may publish this value after an update and retain old values
        safely. It must bind a target by measured position and timestamp, not
        by the independent viewer's largest-area display selection.
        """
        return self._latest_snapshot

    @staticmethod
    def _intrinsics_key(intrinsics):
        return (intrinsics.width, intrinsics.height,
                tuple(np.asarray(intrinsics.matrix).ravel()), tuple(intrinsics.distortion))

    def _geometry(self, frame, d2c):
        color_key = self._intrinsics_key(frame.color_intrinsics)
        depth_key = self._intrinsics_key(frame.depth_intrinsics)
        if color_key != self._camera_key:
            k = frame.color_intrinsics
            self._camera = SDKCameraGeometry(k.matrix, k.width, k.height, 'color_optical',
                                             _compact_cv_distortion(k.distortion))
            self._camera_key = color_key
            self.roi_px = None
            self._native_crop_key = None
        if depth_key != self._depth_key:
            k = frame.depth_intrinsics
            self._depth_camera = CameraGeometry(k.matrix, k.width, k.height, 'depth_optical',
                                                tuple(k.distortion))
            self._depth_rays = self._depth_camera.grid_rays()
            self._depth_key = depth_key
            self._d2c_key = None
            self._native_crop_key = None
        d2c_key = tuple(np.asarray(d2c).ravel())
        if d2c_key != self._d2c_key:
            self._color_rays = self._depth_rays @ np.asarray(d2c)[:3, :3].T
            self._d2c_key = d2c_key
            self._native_crop_key = None
        self._camera.check_grid(frame.rgb, self._camera.frame_id)
        self._depth_camera.check_grid(frame.depth_m, self._depth_camera.frame_id)
        color_to_base = check_transform(frame.camera_to_world_cv) @ np.linalg.inv(check_transform(d2c))
        if self.roi_px is None:
            hull = np.asarray(self.surface.hull_xy, float).reshape(-1, 2)
            flat = np.column_stack((hull, np.zeros(len(hull))))
            plane_to_base = np.asarray(self.surface.plane_to_base, float)
            base = flat @ plane_to_base[:3, :3].T + plane_to_base[:3, 3]
            optical = (base - color_to_base[:3, 3]) @ color_to_base[:3, :3]
            projected = self._camera.project(optical)
            projected = projected[np.isfinite(projected).all(axis=1) & (optical[:, 2] > 0)]
            if not len(projected):
                raise ValueError('Measured belt ROI is outside the color camera view')
            lower = np.maximum(np.floor(projected.min(axis=0)).astype(int), [0, 0])
            upper = np.minimum(np.ceil(projected.max(axis=0)).astype(int) + 1,
                               [self._camera.width, self._camera.height])
            self.roi_px = [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]
            if self._fused is not None and self.tracking_entry_coordinate_m is not None:
                from .nero7_tracking_roi import upstream_rgb_roi
                self.tracking_rgb_roi=upstream_rgb_roi(self.surface,self._camera,color_to_base,
                    self._requested_direction,self.tracking_entry_coordinate_m,
                    self.parameters['block_height_max'])
                rows,cols=np.nonzero(self.tracking_rgb_roi)
                if len(rows):
                    # Include the whole height prism rather than truncating a
                    # tall part to the belt-plane rectangle.
                    self.roi_px=[int(cols.min()),int(rows.min()),int(cols.max()+1),int(rows.max()+1)]
            self.parameters['roi_px'] = self.roi_px.copy()
        if self._direction is None:
            self._direction = self._belt_direction(color_to_base)
        if self._tracker is None:
            self._make_tracker()
        if self.pointcloud_speed and self._point_speed is None:
            from .pointcloud_speed_tracker import PointCloudSpeedTracker
            self._point_speed=PointCloudSpeedTracker(self.parameters,self._direction)
        return color_to_base

    def _native_roi_slice(self, minimum_depth, d2c):
        """Cache a conservative native rectangle for the current RGB ROI.

        For a rotated depth ray r and translation t, color x/z equals
        r.x/r.z + (t.x - r.x/r.z*t.z)/z (likewise y). Bound that interval
        for every positive depth in this frame, then bound Brown distortion
        by the triangle inequality. Only impossible ROI projections are cut.
        A power-of-two *lower* depth bound lets ordinary depth noise reuse
        the rectangle; a nearer point immediately widens/rebuilds it.
        """
        rays = self._color_rays
        if not np.isfinite(minimum_depth) or np.any(rays[..., 2] <= 0):
            return np.s_[:, :]
        translation = np.asarray(d2c)[:3, 3]
        minimum_z = max(.05, minimum_depth * float(rays[..., 2].min()) + translation[2])
        lower_z = float(2. ** np.floor(np.log2(minimum_z)))
        key = (tuple(self.roi_px), lower_z)
        if key != self._native_crop_key:
            far = rays[..., :2] / rays[..., 2, None]
            near = far + (translation[:2] - far * translation[2]) / lower_z
            lo, hi = np.minimum(far, near), np.maximum(far, near)
            absolute = np.maximum(np.abs(lo), np.abs(hi))
            x, y = absolute[..., 0], absolute[..., 1]
            coeffs = np.zeros(5)
            if any(self._camera.D):
                coeffs[:len(self._camera.D)] = self._camera.D
            k1, k2, p1, p2, k3 = np.abs(coeffs)
            r2 = x*x + y*y
            radial = k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            dx = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x) + 1e-9
            dy = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y + 1e-9
            u0, v0, u1, v1 = self.roi_px
            k = self._camera.K
            candidate = ((hi[..., 0]+dx >= (u0-.5-k[0, 2])/k[0, 0])
                & (lo[..., 0]-dx <= (u1-.5-k[0, 2])/k[0, 0])
                & (hi[..., 1]+dy >= (v0-.5-k[1, 2])/k[1, 1])
                & (lo[..., 1]-dy <= (v1-.5-k[1, 2])/k[1, 1]))
            rows, cols = np.flatnonzero(candidate.any(axis=1)), np.flatnonzero(candidate.any(axis=0))
            self._native_crop = (np.s_[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
                                 if len(rows) else np.s_[0:0, 0:0])
            self._native_crop_key = key
        return self._native_crop

    def _native_selection(self, depth, d2c):
        """Discard only points that cannot project into the measured RGB ROI.

        Bounds include the full distortion displacement of every valid point,
        obtained from the triangle inequality of the same Brown polynomial.
        No approximate inverse-distortion corner rectangle is used.  The
        half-pixel extension includes both possible rounding ties.  The peer
        projector subsequently performs the exact projection and z-buffer.
        """
        valid = np.isfinite(depth) & (depth > 0)
        camera = self._camera
        if (camera.D and any(camera.D) and
                (len(camera.D) not in (4, 5) or camera.distortion_model not in (
                    'plumb_bob', 'rational_polynomial'))):
            return valid
        minimum = float(np.min(depth, where=valid, initial=np.inf))
        crop = self._native_roi_slice(minimum, d2c)
        full_valid = np.zeros(depth.shape, bool)
        depth, valid, rays = depth[crop], valid[crop], self._color_rays[crop]
        translation = np.asarray(d2c)[:3, 3]
        optical_z = depth * rays[..., 2] + translation[2]
        valid &= np.isfinite(optical_z) & (optical_z > .05)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            x = (depth * rays[..., 0] + translation[0]) / optical_z
            y = (depth * rays[..., 1] + translation[1]) / optical_z
        valid &= np.isfinite(x) & np.isfinite(y)
        coefficients = np.zeros(5)
        if camera.D and any(camera.D):
            coefficients[:len(camera.D)] = camera.D
        k1, k2, p1, p2, k3 = np.abs(coefficients)
        u0, v0, u1, v1 = self.roi_px
        lower_x, upper_x = (np.array([u0-.5, u1-.5]) - camera.K[0, 2]) / camera.K[0, 0]
        lower_y, upper_y = (np.array([v0-.5, v1-.5]) - camera.K[1, 2]) / camera.K[1, 1]
        absolute_x, absolute_y = np.abs(x), np.abs(y)
        # Each pass only removes geometrically impossible pixels; recomputing
        # the polynomial bound over that subset sharpens the next safe bound.
        for _ in range(3):
            max_x = float(np.max(absolute_x, where=valid, initial=0.))
            max_y = float(np.max(absolute_y, where=valid, initial=0.))
            radius_sq = max_x**2 + max_y**2
            radial_error = k1*radius_sq + k2*radius_sq**2 + k3*radius_sq**3
            delta_x = max_x*radial_error + 2*p1*max_x*max_y + p2*(radius_sq + 2*max_x**2)
            delta_y = max_y*radial_error + p1*(radius_sq + 2*max_y**2) + 2*p2*max_x*max_y
            # Roundoff allowance for I/O exclusion, never an observation gate.
            delta_x += 1e-9
            delta_y += 1e-9
            valid &= ((x >= lower_x-delta_x) & (x <= upper_x+delta_x)
                      & (y >= lower_y-delta_y) & (y <= upper_y+delta_y))
        full_valid[crop] = valid
        return full_valid

    def _registered_depth(self, depth, d2c, selected):
        """Peer-equivalent nearest-optical-Z registration within its image ROI.

        The original ``project_cloud`` stays unchanged in vendor as the test
        reference.  Candidate points are already finite/positive; repeated
        full-array checks and allocation of an unused Jacobian are omitted.
        Sorted pixel runs reduce all colliding depths to their same minimum.
        """
        indices = np.flatnonzero(selected)
        points = self._depth_rays.reshape(-1, 3)[indices] * depth.ravel()[indices, None]
        optical = points @ np.asarray(d2c)[:3, :3].T + np.asarray(d2c)[:3, 3]
        projection = self._camera.project(optical)
        u0, v0, u1, v1 = self.roi_px
        possible = (np.isfinite(projection[:, 0]) & np.isfinite(projection[:, 1])
                    & (projection[:, 0] >= u0-.5) & (projection[:, 0] <= u1-.5)
                    & (projection[:, 1] >= v0-.5) & (projection[:, 1] <= v1-.5))
        u = np.rint(projection[possible, 0]).astype(np.int64)
        v = np.rint(projection[possible, 1]).astype(np.int64)
        z = optical[possible, 2]
        inside = (u >= u0) & (u < u1) & (v >= v0) & (v < v1) & (z > .05)
        indices = v[inside] * self._camera.width + u[inside]
        z = z[inside]
        aligned = np.full((self._camera.height, self._camera.width), np.nan, np.float32)
        if len(indices):
            order = np.argsort(indices, kind='stable')
            sorted_indices = indices[order]
            starts = np.r_[0, np.flatnonzero(sorted_indices[1:] != sorted_indices[:-1]) + 1]
            aligned.ravel()[sorted_indices[starts]] = np.minimum.reduceat(z[order], starts)
        return aligned

    def _belt_direction(self, color_to_base):
        if self._requested_direction is not None:
            value = np.asarray(self._requested_direction, float).reshape(-1)
            if len(value) not in (2, 3) or not np.isfinite(value).all():
                raise ValueError('belt_direction_base requires two or three finite components')
            xy = value[:2].copy()
        else:
            hull = np.asarray(self.surface.hull_xy, float).reshape(-1, 2)
            centered = hull - hull.mean(axis=0)
            _, eigenvectors = np.linalg.eigh(centered.T @ centered)
            direction = np.asarray(self.surface.plane_to_base)[:3, :2] @ eigenvectors[:, -1]
            origin = (np.asarray(self.surface.plane_to_base)[:3, :2] @ hull.mean(axis=0)
                      + np.asarray(self.surface.plane_to_base)[:3, 3])
            endpoints = np.asarray([origin - .05 * direction, origin + .05 * direction])
            optical = (endpoints - color_to_base[:3, 3]) @ color_to_base[:3, :3]
            pixels = self._camera.project(optical)
            if pixels[1, 0] < pixels[0, 0]:
                direction = -direction
            xy = direction[:2]
        norm = float(np.linalg.norm(xy))
        if norm <= 0:
            raise ValueError('The peer XY estimator needs a nonzero XY belt direction')
        return xy / norm

    def _detect(self, rgb, depth, color_to_base):
        """Call the unchanged peer detector on its ROI with original coordinates.

        Two background pixels preserve the 3x3 morphology at ROI edges. Even
        offsets preserve Python's half-to-even rounding for the belt sample.
        Camera rays and plane fitting still use full-image pixel coordinates.
        """
        u0, v0, u1, v1 = self.roi_px
        x0, y0 = max(0, u0-2) & ~1, max(0, v0-2) & ~1
        x1, y1 = min(depth.shape[1], u1+2), min(depth.shape[0], v1+2)
        crop = np.s_[y0:y1, x0:x1]
        self._candidate_depth_diagnostics = {}
        parameters = dict(self.parameters, roi_px=[u0-x0, v0-y0, u1-x0, v1-y0],
                          _candidate_depth_diagnostics=self._candidate_depth_diagnostics)
        camera=_CroppedCamera(self._camera,x0,y0,x1-x0,y1-y0)
        color=cv2.cvtColor(rgb[crop],cv2.COLOR_RGB2BGR)
        if self.tracking_rgb_roi is not None:
            color[~self.tracking_rgb_roi[crop]]=0
        if self._fused is None:
            detections,mask=detect_legacy(color,depth[crop],camera,color_to_base,
                _CroppedPlane(self._plane,depth,self.roi_px,x0,y0),parameters,
                refit=self._frame_count%self.parameters['plane_refresh_every']==0)
        else:
            if parameters.get('isolate_boundary_depth_gaps', False):
                from .nero7_boundary_depth import isolate_boundary_depth_gaps
                color, skipped = isolate_boundary_depth_gaps(color, depth[crop], parameters,
                    camera=camera, color_to_base=color_to_base, surface=self.surface)
                skipped = [
                    {**item, 'bbox_px':[item['bbox_px'][0]+x0, item['bbox_px'][1]+y0,
                                        *item['bbox_px'][2:]]} for item in skipped]
                self._candidate_depth_diagnostics['skipped_boundary_components'] = [
                    item for item in skipped if item['reason']=='roi_boundary']
                self._candidate_depth_diagnostics['skipped_measured_background_components'] = [
                    item for item in skipped if item['reason']=='measured_outside_belt']
            detections,mask=detect_blocks(color,depth[crop],camera,color_to_base,
                                         self._fused.plane,parameters)
        for detection in detections:
            detection['u'] += x0
            detection['v'] += y0
        full_mask = np.zeros(depth.shape, np.uint8)
        full_mask[crop] = mask
        return detections, full_mask

    def _make_tracker(self):
        p = self.parameters
        speed_cfg = dict(window_s=p['speed_window_s'], min_samples=p['speed_min_samples'],
                         min_span=p['speed_min_span'])
        keys = ('measurement_std', 'accel_noise_std', 'cross_accel_noise_std',
                'initial_velocity_std', 'innovation_gate', 'max_velocity_std',
                'observation_max_age', 'qualification_gap')
        motion_cfg = {key: p['kalman_' + key] for key in keys}
        if 'kalman_max_velocity_m_s' in p:
            motion_cfg['max_velocity']=p['kalman_max_velocity_m_s']
        self._tracker = OcclusionAwareBlockTracker(
            BeltModel(d_hat=self._direction), speed_cfg,
            max_vel=p['tracker_max_vel'], gate_radius=p['tracker_gate_radius'],
            confirm_frames=p['tracker_confirm_frames'], lost_frames=p['tracker_lost_frames'],
            lost_timeout=p['tracker_lost_timeout'], motion_cfg=motion_cfg)
        self._tracker._next_id = self._next_id

    def _report(self, stamp, *, reason='', timings=None, current=False, sequence=None):
        tracks = [] if self._tracker is None else self._tracker.tracks
        def measured_motion(track):
            if not self.pointcloud_speed:
                return track.motion, None
            entry=None if self._point_speed is None else self._point_speed.get(track.id)
            return (track.motion if entry is None else entry['motion']), entry
        def is_valid(track, motion, entry):
            valid=bool(current and track.status != STATUS_LOST and track.speed_valid)
            if self.pointcloud_speed:
                valid=bool(valid and entry is not None and motion.speed_valid and
                           abs(float(motion.observation_stamp)-float(stamp))<1e-9)
            return valid
        selected = next((track for track in tracks if track.id == self._selected_id), None)
        live = [track for track in tracks if track.status != STATUS_LOST]
        if selected is None or (selected.status == STATUS_LOST and live):
            pool = [track for track in live if track.confirmed] or live
            selected = max(pool, key=lambda track: track.area_px) if pool else selected
            if selected is not None:
                self._selected_id = selected.id
        selected_motion,selected_entry=(measured_motion(selected) if selected is not None else (None,None))
        valid = bool(selected is not None and is_valid(selected,selected_motion,selected_entry))
        status = ('measured' if valid else 'lost' if selected is not None and selected.status == STATUS_LOST
                  else 'warming_up' if selected is not None else 'no_target')
        velocity = (np.r_[selected_motion.state[2:4],0.] if valid and self.pointcloud_speed
                    else selected.velocity.copy() if valid else None)
        speed = None if velocity is None else float(np.linalg.norm(velocity[:2]))
        motion = selected_motion
        stamps = [] if motion is None else motion.accepted_stamps
        track_diagnostics = []
        track_snapshots = []
        for track in tracks:
            track_motion,point_entry=measured_motion(track)
            covariance = track_motion.covariance[2:, 2:]
            track_valid = is_valid(track,track_motion,point_entry)
            accepted_stamps = track_motion.accepted_stamps
            track_velocity=(np.r_[track_motion.state[2:4],0.] if self.pointcloud_speed else track.velocity)
            track_snapshots.append(Nero7TrackSnapshot(
                track_id=int(track.id), status=int(track.status),
                observed_position_base=tuple(map(float, track.observed_centroid)),
                filtered_position_base=tuple(map(float, track.centroid)),
                velocity_base_m_s=tuple(map(float, track_velocity)) if track_valid else None,
                speed_valid=track_valid, observation_source_time=float(track.stamp),
                motion_source_time=float(track_motion.state_stamp), sample_count=len(accepted_stamps),
                span_s=float(accepted_stamps[-1]-accepted_stamps[0]) if len(accepted_stamps) > 1 else 0.,
                speed_std_m_s=float(np.sqrt(np.linalg.eigvalsh(covariance)[-1])),
                velocity_covariance_xy=tuple(tuple(map(float, row)) for row in covariance),
                misses=int(track.misses), area_px=float(track.area_px), confirmed=bool(track.confirmed),
                velocity_source='pointcloud_xy_kalman' if self.pointcloud_speed else 'nero7_xy_kalman',
                legacy_velocity_base_m_s=tuple(map(float,track.velocity)) if track.speed_valid else None,
                speed_position_base_m=None if point_entry is None else tuple(map(float,point_entry['point']))))
            track_diagnostics.append({
                'id': track.id, 'status': track.status,
                'speed_valid': track_valid,
                'velocity_base_m_s': track_velocity.tolist() if track_valid else None,
                'legacy_velocity_base_m_s':track.velocity.tolist() if track.speed_valid else None,
                'speed_position_base_m':None if point_entry is None else point_entry['point'].tolist(),
                'speed_position_point_count':None if point_entry is None else point_entry['point_count'],
                'speed_std_m_s': float(np.sqrt(np.linalg.eigvalsh(covariance)[-1])),
                'accepted_count': track_motion.accepted_count,
                'rejected_count': track_motion.rejected_count,
                'sample_count': len(track_motion.accepted_stamps),
                'observation_stamp': track.stamp, 'motion_stamp': track_motion.state_stamp,
                'misses': track.misses,
            })
        self._latest_snapshot = Nero7FrameSnapshot(
            source_time=float(stamp), sequence=int(self._sequence if sequence is None else sequence),
            clock_epoch=int(self._epoch), track_generation=self._generation,
            current=bool(current), tracks=tuple(track_snapshots), reason=reason)
        return {
            'status': status, 'reason': reason, 'timestamp': stamp,
            'association_ambiguity': None if self._tracker is None else {
                'frames': self._tracker.ambiguous_frames,
                'last': self._tracker.last_ambiguous},
            'clock_epoch': self._epoch, 'track_id': None if selected is None else selected.id,
            'position_base': None if selected is None else selected.observed_centroid.copy(),
            'velocity_base_m_s': velocity, 'speed_m_s': speed,
            'direction_base': None if speed is None or speed <= 1e-12 else velocity / speed,
            'sample_count': len(stamps),
            'span_s': float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.,
            'motion_source': 'pointcloud_xy_kalman' if self.pointcloud_speed else 'nero7_xy_kalman',
            'speed_valid': valid,
            'signed_belt_speed_m_s': None if not valid else float(velocity[:2]@self._direction),
            'speed_std_m_s': (None if motion is None else
                             float(np.sqrt(np.linalg.eigvalsh(motion.covariance[2:, 2:])[-1]))),
            # Duplicate/rejected images must not keep an earlier live overlay.
            'peer_tracks': tracks if current else [],
            'roi_px': None if self.roi_px is None else self.roi_px.copy(),
            'parameters': dict(self.parameters),
            'diagnostics': {'track_generation': self._generation,
                            'candidate_count': len(self._detections),
                            'processed_frames': self._frame_count,
                            'tracking_roi_px': None if self.roi_px is None else self.roi_px.copy(),
                            'tracking_limit_coordinate_m': self.tracking_entry_coordinate_m,
                            'belt_surface_capture_sequence': (getattr(self.surface, 'boundary_diagnostics', {}) or {}).get('capture_sequence'),
                            'belt_direction_xy': None if self._direction is None else self._direction.tolist(),
                            'tracks': track_diagnostics, **(timings or {})},
        }

    def update(self, frame, d2c, *, candidate_filter=None):
        # An exception must not leave a previous frame's successful snapshot
        # available as though it described the attempted update.
        self._latest_snapshot = None
        if candidate_filter is None:candidate_filter=self.candidate_filter
        started = time.perf_counter()
        stamp = float(frame.header.source_time)
        sequence, epoch = int(frame.header.sequence), int(frame.header.clock_epoch)
        if not np.isfinite(stamp):
            raise ValueError('A finite camera device timestamp is required')
        reason = ''
        if self._epoch is not None and epoch != self._epoch:
            self.reset(calibration=True); reason = 'camera_clock_reset'
        elif self._stamp is not None and stamp < self._stamp:
            self.reset(calibration=True); reason = 'camera_timestamp_reset'
        elif self._sequence is not None and sequence < self._sequence:
            self.reset(calibration=True); reason = 'camera_sequence_reset'
        if self._sequence == sequence or self._stamp == stamp:
            return self._report(stamp, reason='duplicate_frame', sequence=sequence)
        self._sequence, self._stamp, self._epoch = sequence, stamp, epoch
        color_to_base = self._geometry(frame, d2c)
        if self._fused is not None:
            if not self.surface.matches_frame(frame,d2c):
                raise ValueError('融合测速标定上下文变化，需要重新获取带面')
            if not self._fused.observe(frame,d2c,self._camera,color_to_base):
                return self._report(stamp,reason='calibrating_measured_green_belt_plane',
                    timings={'plane_calibration':self._fused.diagnostics()},sequence=sequence)
        source_depth = np.asarray(frame.depth_m)
        valid = self._native_selection(source_depth, d2c)
        before_registration = time.perf_counter()
        aligned_depth = self._registered_depth(source_depth, d2c, valid)
        before_detection = time.perf_counter()
        admission = None
        try:
            self._detections, self._mask = self._detect(frame.rgb, aligned_depth, color_to_base)
            if candidate_filter is not None:
                self._detections,admission=candidate_filter(self._detections,self._mask,
                    aligned_depth,self._camera,color_to_base,frame,self.parameters)
            if (not self._detections and
                self._candidate_depth_diagnostics.get('skipped_boundary_components')):
                raise ValueError('RGB object candidate has insufficient interior depth; frame is ambiguous')
        except ValueError as exc:
            self._detections = []
            # Match the ROS node: an invalid image is not a new empty-belt observation.
            return self._report(stamp, reason=str(exc), timings={
                'processing_ms': (time.perf_counter() - started) * 1000,
                'registration_ms': (before_detection - before_registration) * 1000,
                'candidate_depth_diagnostics': self._candidate_depth_diagnostics})
        before_tracking = time.perf_counter()
        self._frame_count += 1
        self._tracker.update(self._detections, stamp)
        if self.pointcloud_speed:
            self._point_speed.update(self._tracker.tracks,self._detections,stamp)
        finished = time.perf_counter()
        return self._report(stamp, reason=reason, current=True, timings={
            'processing_ms': (finished - started) * 1000,
            'registration_ms': (before_detection - before_registration) * 1000,
            'detection_ms': (before_tracking - before_detection) * 1000,
            'tracking_ms': (finished - before_tracking) * 1000,
            'native_valid_points': int((np.isfinite(source_depth) & (source_depth > 0)).sum()),
            'native_roi_candidate_points': int(valid.sum()),
            'registered_valid_pixels': int(np.isfinite(aligned_depth).sum()),
            'depth_admission':admission,
            'candidate_depth_diagnostics':self._candidate_depth_diagnostics,
            'detection_mode':self.parameters['detection_mode'],
            'tracking_roi_source':'upstream_measured_belt_height_prism' if self.tracking_rgb_roi is not None else 'measured_belt_rectangle',
            'plane_calibration':None if self._fused is None else self._fused.diagnostics(),
        })

"""Measured belt coordinates and explicitly synthetic offline motion input.

The replay source never opens a device or reads a wall clock. Its input is a
saved detector result and a declared constant velocity; all later samples are
labelled as synthetic, including when the shared predictor fits their history.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from threading import RLock
from typing import Any

import numpy as np

from .nero_moving_perception import (
    MovingTargetSnapshot, MovingTargetTracker, _direction, _readonly, _scalar,
    _transform, infer_belt_direction_base,
)


class NominalMotionReplaySource:
    """Advance declared uniform motion using the caller's virtual clock.

    The saved grasp is at the policy clock origin. Three preceding synthetic
    samples initialize the existing velocity fit without a wall-clock delay.
    ``set_policy_clock_origin`` starts a new replay at the same saved grasp;
    this is not an actual camera-clock reset or an observation of conveyor speed.
    """
    synthetic = True
    measured = False
    velocity_source = 'offline_nominal_motion'

    def __init__(self, detection: Mapping, *, belt_direction: Sequence[float],
                 nominal_speed_m_s: float = .1, sample_period_s: float = .1):
        self.belt_direction = _direction(belt_direction, 'belt_direction')
        self.nominal_speed_m_s = _scalar(nominal_speed_m_s, 'nominal_speed_m_s')
        if self.nominal_speed_m_s < 0:
            raise ValueError('nominal_speed_m_s must be nonnegative')
        self.sample_period_s = _scalar(sample_period_s, 'sample_period_s', positive=True)
        self.velocity = _readonly(self.belt_direction * self.nominal_speed_m_s,
                                  (3,), 'nominal velocity')
        self._grasp = _transform(detection['grasp'], 'saved grasp')
        self._points = _readonly(detection.get('object_points', np.empty((0, 3))),
                                 None, 'saved object points')
        if self._points.ndim != 2 or self._points.shape[1:] != (3,):
            raise ValueError('saved object points must have shape (N, 3)')
        self._detection = {'grasp': self._grasp, 'object_points': self._points,
                           'height': detection['height'], 'width': detection['width'],
                           'belt_region': detection.get('belt_region')}
        self._lock = RLock()
        self._origin = None
        self._latest_index = 0
        self._epoch = -1
        self._closed = False
        # The horizon bounds an offline policy run, not a real unobserved target.
        # Uniform synthetic samples remain available throughout that run.
        self.tracker = MovingTargetTracker(
            nominal_speed_m_s=self.nominal_speed_m_s,
            initial_direction_base=self.belt_direction,
            object_id='offline-nominal-motion-part',
            minimum_velocity_span_s=min(.1, 2 * self.sample_period_s),
            history_window_s=max(.6, 8 * self.sample_period_s),
            fresh_age_s=max(.25, 2 * self.sample_period_s),
            max_coast_s=max(.6, 3 * self.sample_period_s),
            max_prediction_horizon_s=300.)

    def _sample(self, index: int) -> None:
        elapsed = index * self.sample_period_s
        displacement = self.velocity * elapsed
        grasp = self._grasp.copy()
        grasp[:3, 3] += displacement
        observation = {**self._detection, 'grasp': grasp,
                       'object_points': self._points + displacement}
        accepted = self.tracker.update(observation,
            timestamp=self._origin + elapsed, sequence=index + 2, clock_epoch=self._epoch)
        if not accepted:
            raise RuntimeError('synthetic replay sample was not accepted by the shared tracker')

    def set_policy_clock_origin(self, origin_abs: float) -> None:
        origin = _scalar(origin_abs, 'policy clock origin')
        with self._lock:
            self._closed = False
            self._origin = origin
            self._epoch += 1
            self._latest_index = 0
            self.tracker.reset()
            for index in (-2, -1, 0):
                self._sample(index)

    def snapshot(self, now_abs: float, *, robot_arrival_time: float = 0.) -> MovingTargetSnapshot:
        now = _scalar(now_abs, 'now_abs')
        with self._lock:
            if self._closed:
                raise RuntimeError('nominal replay source is closed')
            if self._origin is None:
                self.set_policy_clock_origin(now)
            elapsed = now - self._origin
            if elapsed < -1e-12:
                raise ValueError('replay clock precedes its policy origin')
            index = int(np.floor((elapsed + 1e-12) / self.sample_period_s))
            # Large virtual jumps need only the recent fitting window. These are
            # generated nominal samples, so filling that window claims no unseen
            # camera measurements and avoids work proportional to elapsed time.
            first = max(self._latest_index + 1, index - 7)
            for next_index in range(first, index + 1):
                self._sample(next_index)
            self._latest_index = max(self._latest_index, index)
            result = self.tracker.snapshot(now, robot_arrival_time=robot_arrival_time)
            return replace(result, measured=False,
                           velocity_source=self.velocity_source,
                           reason=self.velocity_source if result.valid else result.reason)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def _measured_region_corners(detection: Mapping) -> np.ndarray:
    region = detection.get('belt_region') or {}
    transform = _transform(region['plane_to_base'], 'measured belt plane')
    bounds = _readonly(region['bounds_xy_m'], (2, 2), 'measured belt bounds')
    local = np.array([[x, y, 0.] for x in bounds[0] for y in bounds[1]])
    return local @ transform[:3, :3].T + transform[:3, 3]


def _image_tangent_prior(points, camera, normal, image_direction):
    """Image-coordinate gradient projected into the measured support plane."""
    image_axis = {'right': (1., 0.), 'left': (-1., 0.),
                  'down': (0., 1.), 'up': (0., -1.)}
    if image_direction not in image_axis:
        raise ValueError('image_direction must be right, left, up or down')
    center = np.mean(points, axis=0)
    rotation = camera[:3, :3].T
    q = rotation @ (center - camera[:3, 3])
    # Calibrated depth points normally lie in front of the camera. For a saved
    # region without that provenance, the camera-axis tangent remains a labelled
    # geometric prior; it does not fabricate a measured direction.
    if q[2] > 1e-12:
        jacobian = np.array([[1 / q[2], 0., -q[0] / q[2] ** 2],
                             [0., 1 / q[2], -q[1] / q[2] ** 2]]) @ rotation
        tangent = jacobian.T @ image_axis[image_direction]
    else:
        tangent = camera[:3, :3] @ [*image_axis[image_direction], 0.]
    tangent -= normal * np.dot(normal, tangent)
    if np.linalg.norm(tangent) <= 1e-12:
        # A requested image axis perpendicular to the visible support plane is
        # unobservable. Keep a measured plane axis as an explicit prior rather
        # than introducing a PCA confidence threshold that aborts preparation.
        _, vectors = np.linalg.eigh((points - center).T @ (points - center))
        tangent = vectors[:, -1]
        tangent -= normal * np.dot(normal, tangent)
    return _direction(tangent, 'calibrated image tangent prior')


def measured_belt_motion_region(
    detection: Mapping, frame: Any, d2c: Any, green_belt_points,
    *, direction: Sequence[float] | None = None, image_direction: str = 'right',
) -> dict:
    """Build longitudinal belt coordinates from measured native 3D surfaces.

    Actual visual velocity is authoritative when supplied as ``direction``.
    Geometric PCA and calibrated image direction are only initial priors. An
    ambiguous PCA reports its reason and falls back to the image tangent rather
    than preventing the complete policy from running.
    """
    diagnostic = None
    point_source = 'native_green_belt_depth'
    try:
        from .frame_products import frame_products
        products = frame_products(frame, d2c)
        def extract():
            world, mask, _ = green_belt_points(frame, d2c)
            points = np.asarray(world, dtype=float)[np.asarray(mask, dtype=bool)].reshape(-1, 3)
            return points[np.isfinite(points).all(axis=1)]
        points = products.get(('measured_green_points', green_belt_points), extract)
        if not len(points):
            raise ValueError('no finite green belt depth points')
    except (ValueError, IndexError) as error:
        # The detector already stores a measured plane and ROI. Reuse that
        # evidence when an optional repeated green extraction is unavailable.
        points = _measured_region_corners(detection)
        diagnostic = f'{type(error).__name__}: {error}'
        point_source = 'saved_measured_belt_region_corners'
    origin = np.mean(points, axis=0)
    plane = detection.get('plane') or {}
    normal = plane.get('normal_base')
    if normal is None and detection.get('belt_region') is not None:
        normal = np.asarray(detection['belt_region']['plane_to_base'])[:3, 2]
    if normal is None:
        _, vectors = np.linalg.eigh((points - origin).T @ (points - origin))
        normal = vectors[:, 0]
    normal = _direction(normal, 'measured belt normal')
    # frame is calibrated native DEPTH -> base, d2c is DEPTH -> COLOR. The user
    # gives a COLOR-image direction; factory D2C must participate in its sign.
    camera = _transform(frame.camera_to_world_cv, 'depth camera to base') @ np.linalg.inv(
        _transform(d2c, 'depth to color'))
    if direction is not None:
        belt_direction = _direction(direction, 'belt direction')
        direction_source = 'supplied_base_direction'
    else:
        try:
            belt_direction = infer_belt_direction_base(points, camera,
                image_direction=image_direction, plane_normal_base=normal)
            direction_source = 'measured_belt_pca_with_calibrated_image_sign_prior'
        except ValueError as error:
            diagnostic = '; '.join(filter(None, (diagnostic, str(error))))
            belt_direction = _image_tangent_prior(points, camera, normal, image_direction)
            direction_source = 'calibrated_image_tangent_prior'
    projected = (points - origin) @ belt_direction
    bounds = [float(np.min(projected)), float(np.max(projected))]
    # Preserve the complete observed extent for interception and tracking; a
    # second narrower arbitrary closing window would discard usable belt area.
    return {'belt_direction': belt_direction.tolist(), 'belt_origin': origin.tolist(),
            's_range': bounds, 'tracking_s_range': bounds.copy(),
            'metadata': {'point_source': point_source, 'native_point_count': len(points),
                'direction_source': direction_source, 'image_direction_prior': image_direction,
                # A supplied direction can be visual velocity or manual CLI
                # input; its measurement provenance belongs to the caller.
                'direction_measured_from_motion': None if direction is not None else False,
                'direction_prior_only': None if direction is not None else True,
                'diagnostic': diagnostic,
                'bounds_quantiles': [0., 1.], 'frame': 'robot_base',
                'coordinate': 'dot(point - belt_origin, belt_direction)',
                'scope': 'observed green belt extent; robot reachability uses the complete planner'}}

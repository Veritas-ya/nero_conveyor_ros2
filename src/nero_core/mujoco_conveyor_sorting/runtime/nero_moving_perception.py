"""Timestamped visual motion input for the existing conveyor sorting policy.

All poses and vectors use the robot base frame.  Camera capture timestamps and
``snapshot(now)`` must use the same host monotonic clock; a processing tick is
never inserted into the measured history. An optional configured speed is an
initial prior, not an encoder measurement. Live grasping consumes only a
target-bound Nero7 XY velocity with separate raw-device and host capture times.
The existing position fitter remains available for offline/synthetic inputs.

This module neither sends actuator commands nor implements a grasp policy.  Its
optional worker performs one ordinary detector call per fresh camera frame and
publishes an immutable snapshot independently of the arm control loop.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic
from types import MappingProxyType
from typing import Any

import numpy as np

from ..perception.motion_predictor import MotionPredictor, MotionSample
from ..controllers._math import belt_plane_motion_from_xy


def _scalar(value: float, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return result


def _readonly(value: Any, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    result.setflags(write=False)
    return result


def _direction(value: Sequence[float], name: str) -> np.ndarray:
    vector = _readonly(value, (3,), name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f'{name} must be nonzero')
    return _readonly(vector / norm, (3,), name)


def _transform(value: Any, name: str) -> np.ndarray:
    result = _readonly(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError(f'{name} must be a rigid homogeneous transform')
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly(value, None, 'metadata array')
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def infer_belt_direction_base(
    belt_points_base: Any,
    camera_to_base_cv: Any,
    *,
    image_direction: str = 'right',
    plane_normal_base: Sequence[float] | None = None,
) -> np.ndarray:
    """Estimate a belt-axis prior, then resolve its sign in the camera image.

    PCA uses measured belt points rather than a base X/Y assumption.  The sign
    uses the *perspective image derivative at the belt centroid*, which also
    works when camera axes and base axes differ.  A nearly square observed ROI
    cannot identify a belt axis and is reported as ambiguous; visual motion may
    still be estimated without using this optional geometric prior.
    """
    points = _readonly(belt_points_base, None, 'belt_points_base')
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        raise ValueError('belt_points_base must contain at least three 3D points')
    camera = _transform(camera_to_base_cv, 'camera_to_base_cv')
    if image_direction not in {'right', 'left', 'up', 'down'}:
        raise ValueError('image_direction must be right, left, up or down')
    center = np.mean(points, axis=0)
    centered = points - center
    if plane_normal_base is not None:
        normal = _direction(plane_normal_base, 'plane_normal_base')
        centered = centered - np.outer(centered @ normal, normal)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    if eigenvalues[-1] <= 1e-12 or eigenvalues[-1] <= 1.2 * eigenvalues[-2]:
        raise ValueError('measured belt ROI has no unambiguous long axis')
    direction = eigenvectors[:, -1]
    base_to_camera_rotation = camera[:3, :3].T
    center_camera = base_to_camera_rotation @ (center - camera[:3, 3])
    direction_camera = base_to_camera_rotation @ direction
    if center_camera[2] <= 0:
        raise ValueError('belt centroid lies behind the calibrated camera')
    derivative = ((direction_camera[:2] * center_camera[2]
                   - center_camera[:2] * direction_camera[2]) / center_camera[2] ** 2)
    image_axis = {'right': np.array([1., 0.]), 'left': np.array([-1., 0.]),
                  'down': np.array([0., 1.]), 'up': np.array([0., -1.])}[image_direction]
    signed = float(derivative @ image_axis)
    if abs(signed) <= 1e-9:
        raise ValueError('belt axis has no observable component in requested image direction')
    return _readonly(direction * (1. if signed > 0 else -1.), (3,), 'belt direction')


@dataclass(frozen=True, slots=True)
class MovingTargetSnapshot:
    """Past measured geometry plus timestamped current/future estimates.

    ``visible`` and ``measured`` describe the latest processed observation of
    this target, not sensor truth at ``decision_timestamp``. In particular,
    ``measured`` labels ``observed_grasp``; ``current_grasp`` and ``grasp`` remain
    motion estimates. ``fresh`` reports capture age independently, and an
    explicit detector miss sets ``coasting``. Finite-age configurations can
    expire observations; the live unlimited mode keeps the last measured speed
    without turning an old capture into a new measurement.
    """
    object_id: str
    observed_timestamp: float
    decision_timestamp: float
    prediction_timestamp: float
    sequence: int | None
    clock_epoch: int | None
    observed_grasp: np.ndarray
    current_grasp: np.ndarray
    grasp: np.ndarray
    object_points: np.ndarray
    current_object_points: np.ndarray
    height: float
    width: float
    belt_region: Mapping | None
    history: tuple[MotionSample, ...]
    velocity: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    fit_rmse: float
    sample_count: int
    observation_age_s: float
    valid: bool
    fresh: bool
    visible: bool
    measured: bool
    coasting: bool
    velocity_source: str
    reason: str
    velocity_sample_count: int = 0
    velocity_span_s: float = 0.
    velocity_speed_std_m_s: float | None = None
    velocity_track_id: int | None = None
    velocity_binding_id: int | None = None
    velocity_track_generation: int | None = None
    velocity_device_timestamp: float | None = None
    velocity_observed_timestamp: float | None = None
    # velocity retains the source's raw result (Nero7 XY, Vz=0). Only this
    # motion-model vector includes the Z inferred from the measured belt plane.
    prediction_velocity: np.ndarray | None = None
    target_identity: Any = None


class MovingTargetTracker:
    """Associate a single measured part and publish its 3D motion estimate.

    No automatic identity switch occurs when a different part appears.
    ``max_coast_s=None`` retains the last qualified external speed across
    missing/unqualified updates; live use also disables the prediction horizon.
    A camera clock reset discards the old fit and needs new measurements,
    without comparing different epochs.
    """
    def __init__(self, *, nominal_speed_m_s: float | None = .1,
                 initial_direction_base: Sequence[float] | None = None,
                 plane_normal_base: Sequence[float] | None = None,
                 object_id: str = 'measured-moving-part',
                 fresh_age_s: float = .25, max_coast_s: float | None = .6,
                 minimum_velocity_span_s: float = .1,
                 association_distance_m: float = .04,
                 dimension_tolerance_m: float = .025,
                 history_size: int = 8, history_window_s: float = .6,
                 max_prediction_horizon_s: float | None = 3.,
                 external_velocity_only: bool = False):
        speed = (None if nominal_speed_m_s is None else
                 _scalar(nominal_speed_m_s, 'nominal_speed_m_s'))
        if speed is not None and speed < 0:
            raise ValueError('nominal_speed_m_s must be nonnegative')
        self.initial_direction_base = (None if initial_direction_base is None else
                                       _direction(initial_direction_base, 'initial_direction_base'))
        self.plane_normal_base = (None if plane_normal_base is None else
                                  _direction(plane_normal_base, 'plane_normal_base'))
        if self.initial_direction_base is not None and self.plane_normal_base is not None:
            tangent = self.initial_direction_base - self.plane_normal_base * np.dot(
                self.initial_direction_base, self.plane_normal_base)
            self.initial_direction_base = _direction(tangent, 'belt tangent direction')
        self.nominal_speed_m_s = speed
        self.external_velocity_only = bool(external_velocity_only)
        self.object_id = str(object_id)
        self.target_identity = None
        self.fresh_age_s = _scalar(fresh_age_s, 'fresh_age_s', positive=True)
        self.max_coast_s = (None if max_coast_s is None else
                            _scalar(max_coast_s, 'max_coast_s', positive=True))
        if self.max_coast_s is not None and self.max_coast_s < self.fresh_age_s:
            raise ValueError('max_coast_s must be at least fresh_age_s')
        self.minimum_velocity_span_s = _scalar(minimum_velocity_span_s,
                                               'minimum_velocity_span_s', positive=True)
        self.association_distance_m = _scalar(association_distance_m,
                                              'association_distance_m', positive=True)
        self.dimension_tolerance_m = _scalar(dimension_tolerance_m,
                                             'dimension_tolerance_m', positive=True)
        self.predictor = MotionPredictor(window_size=history_size, window_seconds=history_window_s,
                                         max_prediction_horizon=max_prediction_horizon_s)
        self._history: deque[MotionSample] = deque(maxlen=history_size)
        self._lock = RLock()
        self._measurement: Mapping | None = None
        self._sequence: int | None = None
        self._clock_epoch: int | None = None
        self._visible = False
        self._missing_timestamp: float | None = None
        self._reason = 'no_measurement'
        self._external_velocity: np.ndarray | None = None
        self._external_host_timestamp: float | None = None
        self._external_device_timestamp: float | None = None
        self._external_track_id: int | None = None
        self._external_binding_id: int | None = None
        self._external_track_generation: int | None = None
        self._external_sample_count = 0
        self._external_span_s = 0.
        self._external_speed_std_m_s: float | None = None
        self._external_reason = 'no_measurement'

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._measurement = None
            self._sequence = self._clock_epoch = None
            self._visible = False
            self._missing_timestamp = None
            self._reason = 'no_measurement'
            self._external_velocity = None
            self._external_host_timestamp = None
            self._external_device_timestamp = None
            self._external_track_id = None
            self._external_binding_id = None
            self._external_track_generation = None
            self._external_sample_count = 0
            self._external_span_s = 0.
            self._external_speed_std_m_s = None
            self._external_reason = 'no_measurement'

    def _velocity(self, now: float | None = None) -> tuple[np.ndarray, str, bool]:
        if self.external_velocity_only:
            if (self._external_velocity is not None
                    and (now is None or self._accepts_age(now-self._external_host_timestamp))):
                return self._external_velocity.copy(), 'nero7_xy_kalman', True
            return np.zeros(3), 'unavailable', False
        span = self._history[-1].timestamp - self._history[0].timestamp if self._history else 0.
        measured = len(self._history) >= 2 and span + 1e-12 >= self.minimum_velocity_span_s
        if measured:
            velocity = self.predictor.estimate_velocity(self._history)
            if self.plane_normal_base is not None:
                velocity -= self.plane_normal_base * np.dot(velocity, self.plane_normal_base)
            return velocity, 'vision', True
        if self.initial_direction_base is not None and self.nominal_speed_m_s is not None:
            return self.initial_direction_base * self.nominal_speed_m_s, 'initial_prior', False
        return np.zeros(3), 'unavailable', False

    def _accepts_age(self, age_s: float) -> bool:
        return self.max_coast_s is None or age_s <= self.max_coast_s + 1e-12

    def _prediction_velocity(self, velocity):
        return (belt_plane_motion_from_xy(velocity, self.plane_normal_base)
                if self.external_velocity_only else np.asarray(velocity).copy())

    @staticmethod
    def _parse_external(measurement: Mapping | None, *, timestamp: float,
                        sequence: int | None, clock_epoch: int | None,
                        source_timestamp: float | None):
        if measurement is None:
            return None, None, None, None, None, 0, 0., None, 'no_nero7_report', None
        if not isinstance(measurement, Mapping):
            raise ValueError('velocity_measurement must be a Nero7 report mapping')
        if sequence is None or clock_epoch is None or source_timestamp is None:
            raise ValueError('Nero7 requires geometry frame sequence, epoch and device timestamp')
        source = _scalar(source_timestamp, 'geometry device timestamp')
        if (int(measurement['clock_epoch']) != clock_epoch
                or int(measurement['geometry_sequence']) != sequence
                or abs(float(measurement['geometry_device_timestamp'])-source) > 1e-6
                or abs(float(measurement['geometry_host_capture_timestamp'])-timestamp) > 1e-6):
            raise ValueError('Nero7 query does not belong to the geometry frame')
        generation = measurement.get('track_generation')
        generation = None if generation is None else int(generation)
        if not measurement.get('speed_valid', False):
            return (None, None, None, None, generation, 0, 0., None,
                    str(measurement.get('reason', 'unavailable')), None)
        if measurement.get('motion_source') != 'nero7_xy_kalman':
            raise ValueError('control velocity must come from Nero7 XY Kalman')
        velocity = _readonly(measurement['velocity_base_m_s'], (3,), 'Nero7 velocity')
        if abs(velocity[2]) > 1e-9:
            raise ValueError('Nero7 visual velocity must have zero Z component')
        device = _scalar(measurement['device_timestamp'], 'Nero7 device timestamp')
        host = _scalar(measurement['host_capture_timestamp'], 'Nero7 host capture timestamp')
        if (int(measurement['sequence']) > sequence or device > source + 1e-9
                or host > timestamp + 1e-9):
            raise ValueError('Nero7 velocity is later than the geometry capture')
        track_id = int(measurement['track_id'])
        sample_count = int(measurement['sample_count'])
        span = _scalar(measurement['span_s'], 'Nero7 sample span')
        speed_std = _scalar(measurement['speed_std_m_s'], 'Nero7 speed covariance diagnostic')
        if sample_count < 1 or span < 0 or speed_std < 0:
            raise ValueError('Nero7 speed metadata is invalid')
        binding_id = int(measurement.get('binding_id', track_id))
        return velocity, host, device, track_id, generation, sample_count, span, speed_std, 'ok', binding_id

    def update(self, observation: Mapping, *, timestamp: float,
               sequence: int | None = None, clock_epoch: int | None = None,
               source_timestamp: float | None = None,
               velocity_measurement: Mapping | None = None) -> bool:
        """Accept one genuine detector result; return False for a rejected track update."""
        stamp = _scalar(timestamp, 'capture timestamp')
        grasp = _transform(observation['grasp'], 'grasp')
        points = _readonly(observation.get('object_points', np.empty((0, 3))), None, 'object_points')
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError('object_points must have shape (N, 3)')
        height = _scalar(observation['height'], 'height', positive=True)
        width = _scalar(observation['width'], 'width', positive=True)
        if velocity_measurement is None:
            velocity_measurement = observation.get('velocity_measurement')
        if self.external_velocity_only:
            external = self._parse_external(velocity_measurement, timestamp=stamp,
                sequence=sequence, clock_epoch=clock_epoch, source_timestamp=source_timestamp)
        elif velocity_measurement is not None:
            raise ValueError('Nero7 velocity requires external_velocity_only=True')
        with self._lock:
            if self._clock_epoch is not None and clock_epoch is not None and clock_epoch != self._clock_epoch:
                self.reset()
                self._reason = 'camera_clock_reset'
            if self.external_velocity_only:
                generation = external[4]
                if generation is not None and generation != self._external_track_generation:
                    self._external_track_generation = generation
                    self._external_track_id = None
                    self._external_binding_id = None
                    self._external_velocity = None
                    self._external_host_timestamp = None
            if self._history:
                if stamp <= self._history[-1].timestamp:
                    return False
                if sequence is not None and self._sequence is not None and sequence <= self._sequence:
                    return False
                # Slow grasp geometry may arrive after the independent Nero7
                # reader has already qualified this very part. Use that real,
                # causally earlier velocity for association of this capture;
                # otherwise the old 40 mm geometric gate could reject a real
                # part before the tracker ever adopts its first speed.
                incoming_velocity, incoming_host = external[:2] if self.external_velocity_only else (None, None)
                incoming_usable = (incoming_velocity is not None
                    and self._accepts_age(stamp-incoming_host)
                    and (self._external_binding_id is None or external[9] == self._external_binding_id))
                if incoming_usable:
                    velocity, measured_velocity = incoming_velocity, True
                else:
                    velocity, _, measured_velocity = self._velocity(stamp)
                elapsed = stamp - self._history[-1].timestamp
                expected = self._history[-1].position + self._prediction_velocity(velocity) * elapsed
                if (self.max_coast_s is None and incoming_usable
                        and self._external_velocity is not None):
                    # A newly observed zero/reversed speed does not describe
                    # the whole unseen interval. Also match the position we
                    # have actually been predicting with the retained speed,
                    # using the same identity and original distance gate.
                    retained_expected = (self._history[-1].position
                        + self._prediction_velocity(self._external_velocity) * elapsed)
                    if (np.linalg.norm(grasp[:3, 3]-retained_expected)
                            < np.linalg.norm(grasp[:3, 3]-expected)):
                        expected = retained_expected
                # Until two real captures determine velocity, a geometric
                # direction prior must not prevent learning the actual axis.
                # Match displacement from the last measured position with the
                # configured speed allowance, even if the PCA/image prior is
                # wrong. That allowance does not create a motion sample/fit.
                association_distance = self.association_distance_m
                if not measured_velocity:
                    expected = self._history[-1].position
                    if self.nominal_speed_m_s is not None:
                        association_distance += self.nominal_speed_m_s * elapsed
                size_ok = (abs(height - self._measurement['height']) <= self.dimension_tolerance_m
                           and abs(width - self._measurement['width']) <= self.dimension_tolerance_m)
                if np.linalg.norm(grasp[:3, 3] - expected) > association_distance or not size_ok:
                    self.mark_missing(stamp, reason='association_mismatch')
                    return False
                # A new matching RGB-D measurement restores the track even
                # after coasting expired. The coast bound constrains prediction
                # without measurements, not acceptance of a real new sample.
            self._history.append(MotionSample(stamp, _readonly(grasp[:3, 3], (3,), 'position')))
            # MotionSample copies its vector; lock down that final copy as well.
            self._history[-1].position.setflags(write=False)
            self._measurement = MappingProxyType({'grasp': grasp, 'object_points': points,
                                                   'height': height, 'width': width,
                                                   'belt_region': _freeze(observation.get('belt_region'))})
            self._sequence, self._clock_epoch = sequence, clock_epoch
            if self.external_velocity_only:
                (velocity, host, device, track_id, _generation,
                 count, span, speed_std, reason, binding_id) = external
                if (velocity is not None and self._external_binding_id is not None
                        and binding_id != self._external_binding_id):
                    velocity, host, device, count, span, speed_std = None, None, None, 0, 0., None
                    reason = 'nero7_track_identity_changed'
                retain = (self.max_coast_s is None and self._external_velocity is not None
                          and reason != 'nero7_track_identity_changed'
                          and (velocity is None or host < self._external_host_timestamp))
                if retain:
                    # A new position or an unqualified speed is not a new
                    # velocity measurement. Preserve value AND provenance.
                    self._external_reason = 'retained_measured_speed'
                else:
                    if velocity is not None:
                        self._external_track_id = track_id
                        self._external_binding_id = binding_id
                    self._external_velocity = velocity
                    self._external_host_timestamp = host
                    self._external_device_timestamp = device
                    self._external_sample_count = count
                    self._external_span_s = span
                    self._external_speed_std_m_s = speed_std
                    self._external_reason = reason
            self._visible, self._missing_timestamp, self._reason = True, None, 'ok'
            return True

    def mark_missing(self, timestamp: float, *, reason: str = 'not_visible') -> None:
        stamp = _scalar(timestamp, 'missing timestamp')
        with self._lock:
            if self._history and stamp < self._history[-1].timestamp:
                return
            self._visible = False
            self._missing_timestamp = stamp
            self._reason = str(reason)

    def replace_geometry(self, observation: Mapping, *, timestamp: float,
                         sequence: int | None, clock_epoch: int | None) -> bool:
        """Change the TCP/shape template at the latest existing observation.

        This is not a new observation or velocity measurement. Re-express the
        same historical positions using the changed constant TCP offset, and
        preserve all timestamps, speed provenance, identity and missing state.
        The caller owns capture identity; the latest observation must match.
        """
        stamp = _scalar(timestamp, 'capture timestamp')
        grasp = _transform(observation['grasp'], 'grasp')
        points = _readonly(observation.get('object_points', np.empty((0, 3))), None, 'object_points')
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError('object_points must have shape (N, 3)')
        height = _scalar(observation['height'], 'height', positive=True)
        width = _scalar(observation['width'], 'width', positive=True)
        with self._lock:
            if (not self._history or stamp != self._history[-1].timestamp
                    or sequence != self._sequence or clock_epoch != self._clock_epoch):
                return False
            offset = grasp[:3, 3]-self._measurement['grasp'][:3, 3]
            history = deque(maxlen=self._history.maxlen)
            for previous in self._history:
                sample = MotionSample(previous.timestamp, previous.position+offset)
                sample.position.setflags(write=False)
                history.append(sample)
            self._history = history
            self._measurement = MappingProxyType({'grasp': grasp, 'object_points': points,
                'height': height, 'width': width, 'belt_region': _freeze(observation.get('belt_region'))})
            return True

    def snapshot(self, now: float, *, robot_arrival_time: float = 0.) -> MovingTargetSnapshot | None:
        stamp = _scalar(now, 'now')
        arrival = _scalar(robot_arrival_time, 'robot_arrival_time')
        if arrival < 0:
            raise ValueError('robot_arrival_time must be nonnegative')
        with self._lock:
            if not self._history:
                return None
            history = tuple(self._history)
            latest = history[-1]
            if stamp < latest.timestamp - 1e-12:
                raise ValueError('snapshot clock precedes the latest capture timestamp')
            velocity, source, measured_velocity = self._velocity(stamp)
            prediction_velocity = self._prediction_velocity(velocity)
            # Keep the existing fit diagnostics. In external mode only Z is
            # inferred from the measured plane; raw Nero7 XY is unchanged.
            # estimated_velocity bypasses the position fitter entirely;
            # its fit_rmse=0 is a model diagnostic, not speed accuracy.
            prediction = self.predictor.predict(history, current_time=stamp,
                                                robot_arrival_time=arrival,
                                                belt_velocity=(None if self.external_velocity_only else
                                                               self.initial_direction_base * self.nominal_speed_m_s
                                                               if self.initial_direction_base is not None
                                                               and self.nominal_speed_m_s is not None else None),
                                                estimated_velocity=(prediction_velocity if self.external_velocity_only else None),
                                                object_id=self.object_id)
            age = stamp-latest.timestamp
            current_position = latest.position + prediction_velocity * age
            future_position = current_position + prediction_velocity * arrival
            observed = self._measurement['grasp']
            current_grasp = observed.copy(); current_grasp[:3, 3] = current_position
            future_grasp = observed.copy(); future_grasp[:3, 3] = future_position
            points = self._measurement['object_points']
            current_points = points + (current_position-latest.position)
            fresh = age <= self.fresh_age_s + 1e-12
            # RGB-D processing takes time even when it successfully observes
            # the part. Age is a latency diagnostic, not evidence of a detector
            # miss; propagation already compensates from the genuine capture
            # timestamp. Only an actual miss/error starts occlusion coasting.
            coasting = not self._visible
            velocity_available = measured_velocity or source == 'initial_prior'
            valid = velocity_available and prediction.valid and self._accepts_age(age)
            velocity_expired = (self.external_velocity_only
                                and self._external_velocity is not None
                                and not self._accepts_age(stamp-self._external_host_timestamp))
            reason = ('velocity_observation_expired' if velocity_expired else
                      (self._external_reason if self.external_velocity_only
                       and self._external_reason != 'ok' else 'velocity_not_observed')
                      if not velocity_available else
                      'observation_expired' if not self._accepts_age(age) else
                      prediction.reason if not prediction.valid else
                      self._reason if not self._visible else
                      'initial_prior' if source == 'initial_prior' else
                      'capture_to_decision_latency' if not fresh else 'ok')
            return MovingTargetSnapshot(
                object_id=self.object_id, target_identity=self.target_identity, observed_timestamp=latest.timestamp,
                decision_timestamp=stamp, prediction_timestamp=stamp+arrival,
                sequence=self._sequence, clock_epoch=self._clock_epoch,
                observed_grasp=observed,
                current_grasp=_readonly(current_grasp, (4, 4), 'current_grasp'),
                grasp=_readonly(future_grasp, (4, 4), 'predicted_grasp'),
                object_points=points,
                current_object_points=_readonly(current_points, None, 'predicted points'),
                height=self._measurement['height'], width=self._measurement['width'],
                belt_region=self._measurement['belt_region'], history=history,
                velocity=_readonly(velocity, (3,), 'velocity'),
                prediction_velocity=_readonly(prediction_velocity, (3,), 'prediction_velocity'),
                uncertainty=_readonly(prediction.uncertainty, (3,), 'uncertainty'),
                confidence=prediction.confidence if valid else 0., fit_rmse=prediction.fit_rmse,
                sample_count=len(history), observation_age_s=age, valid=valid, fresh=fresh,
                visible=self._visible, measured=self._visible,
                coasting=coasting, velocity_source=source, reason=reason,
                velocity_sample_count=self._external_sample_count if measured_velocity and
                    self.external_velocity_only else 0,
                velocity_span_s=self._external_span_s if measured_velocity and
                    self.external_velocity_only else 0.,
                velocity_speed_std_m_s=self._external_speed_std_m_s if measured_velocity and
                    self.external_velocity_only else None,
                velocity_track_id=self._external_track_id if measured_velocity and
                    self.external_velocity_only else None,
                velocity_binding_id=self._external_binding_id if measured_velocity and
                    self.external_velocity_only else None,
                velocity_track_generation=self._external_track_generation if measured_velocity and
                    self.external_velocity_only else None,
                velocity_device_timestamp=self._external_device_timestamp if measured_velocity and
                    self.external_velocity_only else None,
                velocity_observed_timestamp=self._external_host_timestamp if measured_velocity and
                    self.external_velocity_only else None)


class MovingPerceptionWorker:
    """Optional asynchronous single-frame detection loop with injectable I/O.

    ``read_frame`` must wait for a *new* frame or return ``None``.  Its camera
    timestamps and ``clock`` share one host monotonic domain.  A detector miss
    or backend exception does not produce a made-up measurement; it is recorded
    and the tracker can coast for its bounded horizon.
    """
    def __init__(self, tracker: MovingTargetTracker, *, read_frame: Callable[[], Any],
                 detect_frame: Callable[[Any], Mapping | None],
                 measurement_timestamp: Callable[[Any], float] | None = None,
                 clock: Callable[[], float] = monotonic, interval_s: float = .05,
                 on_error: Callable[[str], None] | None = None):
        self.tracker, self.read_frame, self.detect_frame = tracker, read_frame, detect_frame
        self.measurement_timestamp = measurement_timestamp or (lambda frame: frame.header.sample_time)
        self.clock = clock
        self.interval_s = _scalar(interval_s, 'interval_s')
        if self.interval_s < 0:
            raise ValueError('interval_s must be nonnegative')
        self.on_error = on_error
        self.last_error: str | None = None
        self.processed_frames = 0
        self._last_frame_key: tuple[int | None, int] | None = None
        self._stop = Event()
        self._thread: Thread | None = None

    def process_once(self) -> bool:
        frame = self.read_frame()
        if frame is None:
            return False
        header = getattr(frame, 'header', None)
        sequence = getattr(header, 'sequence', None)
        epoch = getattr(header, 'clock_epoch', None)
        if sequence is not None:
            key = (epoch, sequence)
            if key == self._last_frame_key:
                return False
            self._last_frame_key = key
        capture_time = _scalar(self.measurement_timestamp(frame), 'capture timestamp')
        observation = self.detect_frame(frame)
        self.processed_frames += 1
        if observation is None:
            self.tracker.mark_missing(capture_time)
            return False
        return self.tracker.update(observation, timestamp=capture_time,
                                   sequence=sequence, clock_epoch=epoch,
                                   source_timestamp=getattr(header, 'source_time', None),
                                   velocity_measurement=observation.get('velocity_measurement'))

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self.clock()
            try:
                self.process_once()
                self.last_error = None
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
                self.tracker.mark_missing(self.clock(), reason='perception_error')
                if self.on_error is not None:
                    try:
                        self.on_error(self.last_error)
                    except Exception:
                        pass
            self._stop.wait(max(0., self.interval_s - (self.clock()-started)))

    def start(self) -> 'MovingPerceptionWorker':
        if self._thread is not None:
            raise RuntimeError('moving perception worker already started')
        self._thread = Thread(target=self._run, name='nero-moving-perception', daemon=True)
        self._thread.start()
        return self

    def close(self, *, timeout_s: float = 2.) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def snapshot(self, now: float, *, robot_arrival_time: float = 0.) -> MovingTargetSnapshot | None:
        return self.tracker.snapshot(now, robot_arrival_time=robot_arrival_time)

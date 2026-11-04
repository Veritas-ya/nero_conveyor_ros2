"""Move a captured grasp with the peer detector's actual object observations.

Nero7 supplies both position and qualified velocity. The captured shape and NN
orientation are templates: translating them is not a new surface measurement.
This adapter has no camera reader, fitting algorithm, planner or actuator.
"""
from __future__ import annotations

from threading import RLock
from dataclasses import replace
from .types import identity_dict
import numpy as np

from ..controllers._math import belt_plane_motion_from_xy


class Nero7GraspTracking:
    def __init__(self, tracker, velocity_source, observation, *, capture_timestamp, on_update=None):
        self.tracker, self.source = tracker, velocity_source
        self.on_update = on_update
        self._lock = RLock()
        self._closed = False
        self.processed_frames = 0
        self.last_error = None
        self.bind_grasp(observation, capture_timestamp=capture_timestamp)

    def bind_grasp(self, observation, *, capture_timestamp):
        """Replace shape/grasp at its original capture, preserving a live anchor.

        Initial geometry is attached before NN selection. When the selected NN
        grasp arrives for that same image, only its relative pose changes; the
        current workpiece position is not reset to the old image position.
        """
        with self._lock:
            header = observation['frame'].header
            key = (header.clock_epoch, header.sequence, header.source_time)
            same_capture = key == getattr(self, '_capture_key', None)
            identity = getattr(self.tracker, 'target_identity', None)
            incoming = observation.get('target_identity')
            if identity is not None and incoming is not None and not identity.same_task(incoming):
                self.last_error = 'old_task_geometry_ignored'
                return False
            if identity is not None and not same_capture:
                # A new image must match the existing real object's predicted
                # centre under the peer's original association distance.
                if getattr(self, '_latest_record', None) is None:
                    self.last_error = 'identity_continuity_unconfirmed'
                    return False
                elapsed = float(capture_timestamp)-self._latest_record.host_capture_time
                center = np.median(np.asarray(observation['object_points']),axis=0)
                expected = self._last_position.copy()
                if self._velocity is not None:expected += self._velocity*elapsed
                gate = float(self.source.meter.parameters['tracker_gate_radius'])
                if (elapsed < 0 or header.clock_epoch != self._capture_key[0]
                        or np.linalg.norm(center[:2]-expected[:2]) > gate):
                    self.last_error = 'identity_continuity_ambiguous'
                    return False
            begin = getattr(self.source, 'begin_target_binding', None)
            if callable(begin):
                binding = begin(geometry_points_base=observation.get('object_points'),
                    capture_device_time=header.source_time, capture_host_time=capture_timestamp,
                    clock_epoch=header.clock_epoch, sequence=header.sequence)
                self.binding_begin_metadata = binding
                if binding.get('binding_reset', False):
                    same_capture = False
                if not binding.get('accepted', True):
                    # A late old-image network result must not replace current
                    # geometry; the live reader keeps collecting actual frames.
                    self.last_error = binding.get('reason', 'geometry_binding_not_updated')
                    return False
            if identity is None:
                identity = incoming
            elif (not same_capture or not np.array_equal(observation['grasp'],self._template['grasp'])
                    or not np.array_equal(observation['object_points'],self._template['object_points'])):
                identity = replace(identity, geometry_revision=identity.geometry_revision+1)
            self.tracker.target_identity = identity
            if identity is not None:
                self.tracker.object_id = identity.object_id
                observation['target_identity'] = identity
            self._template = {key: observation[key] for key in ('width', 'height')}
            self._template.update({key: np.asarray(observation[key]).copy()
                                   for key in ('grasp', 'object_points')})
            self._template['belt_region'] = observation.get('belt_region')
            self._capture_key = key
            self._capture_host = float(capture_timestamp)
            self._capture_center = np.median(np.asarray(observation['object_points']), axis=0)
            if not same_capture:
                preceding = [r for r in self.source.captured_reports()
                    if r.peer.clock_epoch == header.clock_epoch
                    and r.peer.sequence <= header.sequence
                    and r.peer.source_time <= header.source_time
                    and r.host_capture_time <= self._capture_host]
                self._capture_preceding_generation = (None if not preceding else
                    preceding[-1].peer.track_generation)
                self._anchor = None
                self._peer_id = self._generation = None
                self._last_position = self._last_stamp = self._velocity = None
                self._last_sequence = None
                self._latest_record = self._latest_track = None
                self.anchor_metadata = None
                self.last_error = None
                self._geometry_obsolete = False
                self.tracker.reset()
            elif self._latest_record is not None and not self._geometry_obsolete:
                # NN selection changed only this capture's relative TCP/shape.
                # Do not clear a qualified speed, re-query an old report, or
                # publish an old sequence as another visual observation.
                updated, _, _ = self._observation_at_position(self._last_position)
                self.tracker.replace_geometry(updated,
                    timestamp=self._latest_record.host_capture_time,
                    sequence=self._latest_record.peer.sequence,
                    clock_epoch=self._latest_record.peer.clock_epoch)
            return True

    @staticmethod
    def _observed(record):
        return [t for t in record.peer.tracks if not t.misses
                and t.observation_source_time == record.peer.source_time]

    def _observed_anchor_candidates(self, reports, gate):
        """Prefer an exposure match, then real observations of each identity."""
        epoch, sequence, device = self._capture_key
        candidates = []
        for record in reports:
            if record.peer.source_time != device or record.peer.sequence != sequence:
                continue
            for track in self._observed(record):
                center = np.asarray(track.observed_position_base, float)
                if np.linalg.norm(center[:2]-self._capture_center[:2]) <= gate:
                    candidates.append((record, track, center, {
                        'method': 'same_capture', 'bracket_sequences': [sequence, sequence]}))
        if candidates:
            # In particular, a same-exposure ambiguity is not resolved by a
            # later velocity: both identities were plausible at this capture.
            return candidates

        before, after = {}, {}
        for record in reports:
            if (record.peer.source_time <= device and record.peer.sequence <= sequence
                    and self.source.accepts_age(self._capture_host-record.host_capture_time)):
                for track in self._observed(record):
                    before[track.track_id] = (record, track)
            if record.peer.source_time >= device and record.peer.sequence >= sequence:
                for track in self._observed(record):
                    after.setdefault(track.track_id, (record, track))
        for identity in before.keys() & after.keys():
            first, track = before[identity]
            last, later = after[identity]
            span = last.peer.source_time-first.peer.source_time
            if span <= 0.:
                continue
            fraction = (device-first.peer.source_time)/span
            center = np.asarray(track.observed_position_base, float)
            center = center + fraction*(np.asarray(later.observed_position_base)-center)
            if np.linalg.norm(center[:2]-self._capture_center[:2]) <= gate:
                candidates.append((first, track, center, {
                    'method': 'bracketed_observed_centres',
                    'bracket_sequences': [first.peer.sequence, last.peer.sequence]}))
        return candidates

    def _backprojected_anchor_candidates(self, reports, gate):
        """Use qualified current motion only to align the captured template."""
        if not reports:
            return []
        record = reports[-1]
        if (self._capture_preceding_generation is not None
                and record.peer.track_generation != self._capture_preceding_generation):
            return []
        _, sequence, device = self._capture_key
        elapsed = record.peer.source_time-device
        host_elapsed = record.host_capture_time-self._capture_host
        if (record.peer.sequence < sequence
                or elapsed < 0. or not self.source.accepts_age(elapsed)
                or host_elapsed < 0. or not self.source.accepts_age(host_elapsed)):
            return []
        candidates = []
        for track in self._observed(record):
            if not track.speed_valid or track.velocity_base_m_s is None:
                continue
            velocity = np.asarray(track.velocity_base_m_s, float)
            position = np.asarray(track.observed_position_base, float)
            if (velocity.shape != (3,) or position.shape != (3,)
                    or not np.isfinite(velocity).all() or not np.isfinite(position).all()):
                continue
            motion = belt_plane_motion_from_xy(velocity, self.tracker.plane_normal_base)
            center = position-motion*elapsed
            distance = float(np.linalg.norm(center[:2]-self._capture_center[:2]))
            if distance <= gate:
                candidates.append((record, track, center, {
                    'method': 'constant_velocity_backprojection',
                    'observation_sequence': record.peer.sequence,
                    'observation_device_timestamp': record.peer.source_time,
                    'observation_host_capture_timestamp': record.host_capture_time,
                    'capture_to_observation_device_s': elapsed,
                    'capture_to_observation_host_s': host_elapsed,
                    'maximum_age_s': self.source.max_age_s,
                    'observed_position_base_m': position.tolist(),
                    'measured_velocity_base_m_s': velocity.tolist(),
                    'backprojection_velocity_base_m_s': motion.tolist(),
                    'backprojection_motion_model': 'constant_XY_velocity_with_belt_plane_height',
                    'center_distance_xy_m': distance, 'matching_gate_m': gate,
                    'first_published_sequence': record.peer.sequence,
                    'backprojected_center_is_observation': False}))
        return candidates

    def _find_anchor(self, reports):
        epoch, sequence, device = self._capture_key
        gate = float(self.source.meter.parameters['tracker_gate_radius'])
        generation = self.source.diagnostics().get('geometry_binding_generation')
        if (generation is not None and reports
                and (reports[-1].peer.clock_epoch != epoch
                     or reports[-1].peer.track_generation != generation)):
            self.last_error = 'camera_or_tracker_clock_reset_requires_new_geometry'
            return False
        eligible = tuple(r for r in reports if r.peer.current
            and r.peer.clock_epoch == epoch and r.peer.track_generation == generation)
        candidates = self._observed_anchor_candidates(eligible, gate)
        if not candidates:
            candidates = self._backprojected_anchor_candidates(eligible, gate)
        if len(candidates) != 1:
            self.last_error = 'capture_target_ambiguous' if candidates else 'awaiting_capture_target_match'
            return False
        record, track, center, metadata = candidates[0]
        self._anchor = center.copy()
        self._peer_id = track.track_id
        self._generation = record.peer.track_generation
        self._last_position = center.copy()
        self._last_stamp = device
        if metadata['method'] == 'constant_velocity_backprojection':
            # Older reports can contain the expired startup identity. Begin at
            # the actual observation that justified this new binding, never at
            # a synthetic capture-time sample or an unrelated older identity.
            self._last_sequence = record.peer.sequence-1
        self.anchor_metadata = {**metadata, 'track_id': track.track_id,
            'shape_capture_sequence': sequence, 'shape_capture_timestamp': self._capture_host,
            'shape_capture_device_timestamp': device, 'peer_center_at_shape_capture_m': center.tolist(),
            'clock_epoch': epoch, 'track_generation': record.peer.track_generation}
        self.last_error = None
        return True

    def _select_track(self, record):
        retained = next((t for t in record.peer.tracks if t.track_id == self._peer_id), None)
        if retained is not None:
            # The peer owns this identity until it expires, including misses.
            return retained if retained in self._observed(record) else None
        elapsed = record.peer.source_time-self._last_stamp
        gate = float(self.source.meter.parameters['tracker_gate_radius'])
        matches = []
        for track in self._observed(record):
            velocity = self._velocity
            # An unqualified startup ID has no motion binding to preserve.
            # A newly qualified candidate can explain its actual displacement
            # using its own real velocity, as the downstream tracker already
            # does. Do not create a nominal speed for the missing interval.
            if velocity is None and track.speed_valid and track.velocity_base_m_s is not None:
                velocity = np.asarray(track.velocity_base_m_s)
            expected = self._last_position.copy()
            if velocity is not None:
                expected += velocity*elapsed
            if np.linalg.norm(np.asarray(track.observed_position_base)[:2]-expected[:2]) <= gate:
                matches.append(track)
        if len(matches) != 1:
            self.last_error = 'target_track_ambiguous' if matches else 'target_not_observed'
            return None
        return matches[0]

    def _observation_at_position(self, position):
        """Use the same measured displacement for reports and NN-only updates."""
        raw_displacement = position-self._anchor
        # Nero7 measures a visible surface centre. Occlusion can change its Z
        # without the part leaving the belt. Preserve the selected grasp's
        # height and the measured XY displacement, using the same belt-plane
        # motion model as all subsequent time predictions.
        displacement = belt_plane_motion_from_xy(
            raw_displacement, self.tracker.plane_normal_base)
        grasp = np.asarray(self._template['grasp']).copy()
        grasp[:3, 3] += displacement
        observation = {**self._template, 'grasp': grasp,
                       'object_points': np.asarray(self._template['object_points'])+displacement}
        return observation, raw_displacement, displacement

    def _publish(self, record, track):
        position = np.asarray(track.observed_position_base, float)
        measurement = self.source.query(position,
            capture_device_time=record.peer.source_time,
            capture_host_time=record.host_capture_time,
            clock_epoch=record.peer.clock_epoch, sequence=record.peer.sequence)
        # The source owns qualification; a new position alone cannot refresh
        # a speed timestamp. The tracker retains the last qualified speed.
        observation, raw_displacement, displacement = self._observation_at_position(position)
        accepted = self.tracker.update(observation, timestamp=record.host_capture_time,
            source_timestamp=record.peer.source_time, sequence=record.peer.sequence,
            clock_epoch=record.peer.clock_epoch, velocity_measurement=measurement)
        if not accepted:
            self.last_error = 'target_position_update_rejected'
            return False
        if self.tracker.target_identity is not None:
            self.tracker.target_identity = replace(self.tracker.target_identity, track_id=track.track_id,
                track_generation=record.peer.track_generation)
        previous = self._peer_id
        self._peer_id = track.track_id
        self._last_position, self._last_stamp = position.copy(), record.peer.source_time
        self._latest_record, self._latest_track = record, track
        if measurement['speed_valid']:
            self._velocity = np.asarray(measurement['velocity_base_m_s']).copy()
        self.processed_frames += 1
        self.last_error = None
        if self.on_update is not None:
            self.on_update({'target_identity':identity_dict(self.tracker.target_identity), 'sequence': record.peer.sequence, 'clock_epoch': record.peer.clock_epoch,
                'device_timestamp': record.peer.source_time,
                'host_capture_timestamp': record.host_capture_time,
                'position_source': 'nero7_observed_position',
                'geometry_source': 'captured_shape_translated_by_nero7_observation',
                'shape_capture_sequence': self._capture_key[1],
                'shape_capture_timestamp': self._capture_host,
                'track_id': track.track_id, 'previous_track_id': previous,
                'reassociation_source': None if previous == track.track_id else 'peer_observation_and_predicted_motion',
                'observed_position_base': position.tolist(), 'translation_base_m': displacement.tolist(),
                'raw_observed_displacement_base_m': raw_displacement.tolist(),
                'translation_model': ('nero7_xy_with_measured_belt_plane_z'
                    if self.tracker.plane_normal_base is not None else 'nero7_observed_xyz'),
                'position_published_without_new_qualified_speed': not measurement['speed_valid'],
                'grasp_tcp': observation['grasp'].tolist(), 'velocity_measurement': measurement,
                'anchor': self.anchor_metadata})
        return True

    def process_available(self):
        """Consume each real peer observation once, independent of NN/point clouds."""
        with self._lock:
            if self._closed or self._geometry_obsolete:
                return False
            reports = self.source.captured_reports()
            if self._anchor is None and not self._find_anchor(reports):
                return False
            changed = False
            for record in reports:
                if (record.peer.clock_epoch != self._capture_key[0]
                        or record.peer.track_generation != self._generation):
                    # A reset cannot give a new timestamp to the old template.
                    if record is reports[-1]:
                        self.tracker.reset()
                        self._geometry_obsolete = True
                        self.last_error = 'camera_or_tracker_clock_reset_requires_new_geometry'
                    continue
                if (record.peer.source_time < self._capture_key[2]
                        or self._last_sequence is not None and record.peer.sequence <= self._last_sequence):
                    continue
                self._last_sequence = record.peer.sequence
                track = self._select_track(record)
                if track is None:
                    self.tracker.mark_missing(record.host_capture_time, reason='nero7_target_not_visible')
                    continue
                changed = self._publish(record, track) or changed
            return changed

    def start(self):
        self.process_available()
        return self

    def snapshot(self, now, *, robot_arrival_time=0.):
        # The source callback consumes updates. Snapshot only reads completed
        # observations so a later camera frame cannot race ahead of `now`.
        with self._lock:
            if self._geometry_obsolete:
                return None
            if self._latest_record is not None and now < self._latest_record.host_capture_time:
                return None
            return self.tracker.snapshot(now, robot_arrival_time=robot_arrival_time)

    def close(self):
        with self._lock:
            self._closed = True

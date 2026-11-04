"""Independent cached-frame reader for Nero7 velocity during live grasping.

The Orbbec backend owns the only SDK capture thread. This reader consumes its
latest immutable RGB-D cache on a separate thread from grasp geometry. Nero7
continues to fit motion on raw device timestamps; host depth-capture estimates
are retained separately for control-time age checks. No part of this module
sends actuator commands or changes the peer detector/filter parameters.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import signal
from types import SimpleNamespace
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any

import numpy as np

from .nero7_belt_velocity import Nero7BeltVelocityMeter, Nero7FrameSnapshot


def _initialize_meter(meter):
    # Spawn owns only this CPU estimator, never camera/CAN handles or callbacks.
    # The parent handles Ctrl+C as pause. Ignore it before releasing the mask
    # inherited at spawn, including any Ctrl+C delivered during module imports.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'pthread_sigmask'):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    global _process_meter
    _process_meter = meter


def _meter_ready():
    return True


def _measure_in_process(metadata, memory_name, rgb_shape, depth_shape, depth_offset, depth_to_color_cv):
    memory = SharedMemory(name=memory_name)
    try:
        rgb = np.ndarray(rgb_shape, dtype=np.uint8, buffer=memory.buf)
        depth = np.ndarray(depth_shape, dtype=np.float32, buffer=memory.buf, offset=depth_offset)
        rgb.setflags(write=False)
        depth.setflags(write=False)
        frame = SimpleNamespace(**metadata, rgb=rgb, depth_m=depth)
        _process_meter.update(frame, depth_to_color_cv)
        return _process_meter.latest_snapshot
    finally:
        memory.close()


class _MeterProcess:
    """One in-flight cached frame; exact pixels, no image pickle/queue backlog."""
    def __init__(self, meter):
        self.executor = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_meter, initargs=(meter,))
        self.memory = None
        self._shape = None

    def start(self):
        # Complete spawn/imports before selecting the first camera frame. The
        # process startup must not become a gap between the first two captures.
        # On Linux, a child inherits the spawning thread's signal mask. Protect
        # imports/unpickling until the initializer above installs SIG_IGN. Only
        # this thread is masked; the parent's main thread still handles pause.
        previous_mask = (signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
                         if hasattr(signal, 'pthread_sigmask') else None)
        try:
            ready = self.executor.submit(_meter_ready)
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        ready.result()

    def update(self, frame, d2c):
        shape = (frame.rgb.shape, frame.depth_m.shape)
        offset = (frame.rgb.size+7)//8*8
        if shape != self._shape:
            self._release_memory()
            self.memory = SharedMemory(create=True, size=offset+frame.depth_m.size*4)
            self._shape = shape
        rgb = np.ndarray(shape[0], dtype=np.uint8, buffer=self.memory.buf)
        depth = np.ndarray(shape[1], dtype=np.float32, buffer=self.memory.buf, offset=offset)
        np.copyto(rgb, frame.rgb)
        np.copyto(depth, frame.depth_m)
        metadata = {key: getattr(frame, key) for key in
                    ('header', 'color_intrinsics', 'depth_intrinsics', 'camera_to_world_cv')}
        # Wait for the report before this shared slot can hold another frame.
        return self.executor.submit(_measure_in_process, metadata, self.memory.name,
                                    *shape, offset, d2c).result()

    def _release_memory(self):
        if self.memory is not None:
            self.memory.close()
            self.memory.unlink()
            self.memory = None

    def close(self):
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._release_memory()


@dataclass(frozen=True, slots=True)
class _CapturedReport:
    peer: Nero7FrameSnapshot
    host_capture_time: float


class Nero7VelocitySource:
    """Publish causal, target-bound speed from the peer's original XY filter.

    ``read_frame(after_sequence)`` should call the *same* camera backend's
    ``read_latest`` cache. ``capture_monotonic`` must map the depth device time
    to its original host capture estimate, never to image-processing completion.
    The geometry thread queries a report at or before its own capture and only
    speed first confirms the target. Expired peer IDs can be reassociated using
    the same geometry gate and the target's last measured motion.
    """

    def __init__(self, meter: Nero7BeltVelocityMeter, *,
                 read_frame: Callable[[int | None], Any],
                 calibrate_frame: Callable[[Any], Any],
                 capture_monotonic: Callable[[Any], float],
                 depth_to_color_cv: Any,
                 initial_sequence: int | None = None,
                 max_age_s: float | None = .6,
                 max_rate_hz: float = 30.,
                 process_isolation: bool = False,
                 on_report: Callable[[dict], None] | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_progress: Callable[[dict], None] | None = None):
        if not isinstance(meter, Nero7BeltVelocityMeter):
            raise TypeError('meter must be a Nero7BeltVelocityMeter')
        self.meter = meter
        self.read_frame = read_frame
        self.calibrate_frame = calibrate_frame
        self.capture_monotonic = capture_monotonic
        self.depth_to_color_cv = np.asarray(depth_to_color_cv, dtype=float).copy()
        if self.depth_to_color_cv.shape != (4, 4) or not np.isfinite(self.depth_to_color_cv).all():
            raise ValueError('depth_to_color_cv must be a finite 4x4 transform')
        # None retains a qualified measurement until replaced or rebound;
        # capture timestamps remain the original measurement timestamps.
        self.max_age_s = None if max_age_s is None else float(max_age_s)
        self.max_rate_hz = float(max_rate_hz)
        if self.max_age_s is not None and (not np.isfinite(self.max_age_s) or self.max_age_s <= 0):
            raise ValueError('max_age_s must be None or finite and positive')
        if not np.isfinite(self.max_rate_hz) or self.max_rate_hz <= 0:
            raise ValueError('max_rate_hz must be finite and positive')
        self.on_report, self.on_error = on_report, on_error
        self.on_progress = on_progress
        self.process_isolation = bool(process_isolation)
        self._executor = None
        self._last_sequence = initial_sequence
        self._last_key: tuple[int, int, float] | None = None
        self._epoch: int | None = None
        self._generation: int | None = None
        self._bound_track_id: int | None = None
        self._binding_id: int | None = None
        self._bound_anchor = None
        self._candidate_track_id: int | None = None
        self._last_reassociation = None
        self._geometry_binding_capture = None
        self._geometry_binding_host = None
        self._geometry_binding_generation = None
        self._geometry_binding_serial = 0
        self._last_binding_begin = None
        self._reports: deque[_CapturedReport] = deque(maxlen=256)
        self._qualified_report: _CapturedReport | None = None
        self._lock = RLock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._processed = self._current = self._duplicates = self._errors = 0
        self._processing_total_s = 0.
        self.last_error: str | None = None
        self._last_query_reason = 'no_geometry_query'

    def begin_target_binding(self, *, geometry_points_base, capture_device_time,
                             capture_host_time, clock_epoch, sequence) -> dict:
        """Start one new measured-geometry task without restarting perception.

        Only a strictly newer real exposure may clear the target association.
        A network candidate for the same capture is an idempotent update; old
        geometry cannot overwrite the newer task. KF, reports, timestamps and
        the one reader/process are retained. Incomplete legacy metadata keeps
        the old behavior and records a diagnostic rather than adding a gate.
        """
        try:
            points = np.asarray(geometry_points_base, dtype=float)
            device, host = float(capture_device_time), float(capture_host_time)
            epoch, seq = int(clock_epoch), int(sequence)
            complete = (points.ndim == 2 and points.shape[1:] == (3,) and len(points)>0
                and np.isfinite(points).all() and np.isfinite([device, host]).all()
                and epoch == clock_epoch and seq == sequence)
        except (TypeError, ValueError, OverflowError):
            complete = False
        with self._lock:
            if not complete:
                result = {'accepted': True, 'binding_reset': False,
                    'reason': 'legacy_geometry_identity_incomplete_no_reset',
                    'binding_serial': self._geometry_binding_serial}
                self._last_binding_begin = result
                return dict(result)
            key = (epoch, seq, device)
            previous = self._geometry_binding_capture
            reason = None
            latest_host = self._reports[-1].host_capture_time if self._reports else None
            # Host capture time continues across a device clock reset. A new
            # epoch exposure may precede its estimator report; stale old-epoch
            # geometry has an earlier host capture and must not take over.
            if self._epoch is not None and epoch != self._epoch:
                if latest_host is not None and host <= latest_host:
                    reason = 'geometry_clock_epoch_mismatch'
            generation_changed = (self._geometry_binding_generation is not None
                and self._generation is not None
                and self._geometry_binding_generation != self._generation)
            if reason is None and previous == key and not generation_changed:
                result = {'accepted': True, 'binding_reset': False,
                    'reason': 'same_capture_preserves_target_binding',
                    'binding_serial': self._geometry_binding_serial,
                    'capture_key': key}
                self._last_binding_begin = result
                return dict(result)
            if reason is None and previous is not None:
                if generation_changed:
                    if self._geometry_binding_host is not None and host <= self._geometry_binding_host:
                        reason = 'geometry_generation_requires_new_capture'
                elif epoch == previous[0]:
                    if seq <= previous[1] or device <= previous[2]:
                        # A genuine newer host exposure can arrive before the
                        # meter reports a same-epoch sequence/device reset.
                        # A duplicate source exposure with only a new sequence
                        # is still the same image, not a new geometry task.
                        pending_reset_capture = (self._geometry_binding_host is not None
                            and host > self._geometry_binding_host
                            and latest_host is not None and host > latest_host
                            and device != previous[2]
                            and (seq < previous[1] or device < previous[2]))
                        if not pending_reset_capture:
                            reason = 'geometry_capture_not_newer'
                    elif self._geometry_binding_host is not None and host < self._geometry_binding_host:
                        reason = 'geometry_capture_host_time_not_newer'
                elif self._geometry_binding_host is not None and host <= self._geometry_binding_host:
                    reason = 'geometry_clock_epoch_mismatch'
            if reason is not None:
                result = {'accepted': False, 'binding_reset': False, 'reason': reason,
                    'binding_serial': self._geometry_binding_serial, 'capture_key': key,
                    'current_capture_key': previous}
                self._last_binding_begin = result
                return dict(result)
            previous_track = self._bound_track_id
            retained = len(self._reports)
            self._bound_track_id = self._binding_id = self._candidate_track_id = None
            self._bound_anchor = self._last_reassociation = None
            self._qualified_report = None
            self._geometry_binding_capture, self._geometry_binding_host = key, host
            # The latest completed peer may precede this real exposure. Its
            # generation cannot describe a reset that the meter has not yet
            # processed. Resolve that provenance only when processing catches
            # up to the geometry's original host capture time.
            self._geometry_binding_generation = (self._generation
                if epoch == self._epoch and latest_host is not None and latest_host >= host
                else None)
            self._geometry_binding_serial += 1
            self._last_query_reason = 'awaiting_new_geometry_target_query'
            result = {'accepted': True, 'binding_reset': True,
                'reason': 'new_measured_geometry_target', 'capture_key': key,
                'capture_host_time': host, 'binding_serial': self._geometry_binding_serial,
                'previous_bound_track_id': previous_track,
                'geometry_center_base_m': np.median(points, axis=0).tolist(),
                'geometry_point_count': len(points), 'retained_reports': retained,
                'track_generation': self._geometry_binding_generation,
                'meter_and_report_history_preserved': True}
            self._last_binding_begin = result
            return dict(result)

    @staticmethod
    def _plain_report(peer: Nero7FrameSnapshot, host_capture_time: float) -> dict:
        return {
            'device_timestamp': peer.source_time,
            'host_capture_timestamp': host_capture_time,
            'sequence': peer.sequence, 'clock_epoch': peer.clock_epoch,
            'track_generation': peer.track_generation, 'current': peer.current,
            'reason': peer.reason, 'motion_source': 'nero7_xy_kalman',
            'track_measurements': [
                {'track_id': track.track_id, 'status': track.status,
                 'observed_position_base': list(track.observed_position_base),
                 'velocity_base_m_s': (None if track.velocity_base_m_s is None else
                                       list(track.velocity_base_m_s)),
                 'speed_valid': track.speed_valid, 'sample_count': track.sample_count,
                 'span_s': track.span_s, 'speed_std_m_s': track.speed_std_m_s,
                 'observation_device_timestamp': track.observation_source_time,
                 'misses': track.misses}
                for track in peer.tracks],
        }

    def _read_next(self):
        try:
            return self.read_frame(self._last_sequence)
        except Exception:
            # If an active backend's source clock epoch changes, its latest
            # cached frame may need inspection independently of the sequence.
            if self._last_sequence is None:
                raise
            frame = self.read_frame(None)
            if self._epoch is not None and frame.header.clock_epoch != self._epoch:
                return frame
            raise

    def process_once(self) -> bool:
        """Process at most one new cached frame; usable without starting a thread."""
        if self.process_isolation and self._executor is None:
            started_process = monotonic()
            if self.on_progress is not None:
                self.on_progress({'status': 'starting'})
            self._executor = _MeterProcess(self.meter)
            self._executor.start()
            if self.on_progress is not None:
                self.on_progress({'status': 'ready',
                                  'duration_s': monotonic()-started_process})
        raw = self._read_next()
        started = monotonic()
        header = raw.header
        source_time = float(header.source_time)
        host_time = float(self.capture_monotonic(raw))
        if not np.isfinite([source_time, host_time]).all():
            raise ValueError('Nero7 requires finite device and host capture timestamps')
        key = (int(header.clock_epoch), int(header.sequence), source_time)
        with self._lock:
            if key == self._last_key:
                self._duplicates += 1
                return False
            self._last_key = key
            self._last_sequence = key[1]
        calibrated = self.calibrate_frame(raw)
        if self.process_isolation:
            peer = self._executor.update(calibrated, self.depth_to_color_cv)
        else:
            self.meter.update(calibrated, self.depth_to_color_cv)
            peer = self.meter.latest_snapshot
        if peer is None or (peer.clock_epoch, peer.sequence, peer.source_time) != key:
            raise RuntimeError('Nero7 report does not belong to the cached RGB-D frame')
        elapsed = monotonic() - started
        with self._lock:
            if peer.clock_epoch != self._epoch or peer.track_generation != self._generation:
                self._reports.clear()
                self._bound_track_id = None
                self._binding_id = self._candidate_track_id = None
                self._bound_anchor = self._last_reassociation = None
                self._qualified_report = None
            self._epoch, self._generation = peer.clock_epoch, peer.track_generation
            if (self._geometry_binding_capture is not None
                    and self._geometry_binding_capture[0] == peer.clock_epoch
                    and self._geometry_binding_generation is None
                    and host_time >= self._geometry_binding_host):
                self._geometry_binding_generation = peer.track_generation
            self._processed += 1
            self._processing_total_s += elapsed
            self.last_error = None
            if peer.current:
                self._reports.append(_CapturedReport(peer, host_time))
                self._current += 1
        if self.on_report is not None:
            self.on_report(self._plain_report(peer, host_time))
        return peer.current

    def _run(self) -> None:
        interval = 1. / self.max_rate_hz
        while not self._stop.is_set():
            started = monotonic()
            try:
                self.process_once()
            except Exception as error:
                message = f'{type(error).__name__}: {error}'
                with self._lock:
                    self._errors += 1
                    self.last_error = message
                if self.on_error is not None:
                    try:
                        self.on_error(message)
                    except Exception:
                        pass
            self._stop.wait(max(0., interval - (monotonic() - started)))

    def start(self) -> 'Nero7VelocitySource':
        if self._thread is not None:
            raise RuntimeError('Nero7 velocity source already started')
        self._thread = Thread(target=self._run, name='nero7-live-velocity', daemon=True)
        self._thread.start()
        return self

    def close(self, *, timeout_s: float = 2.) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        if self._executor is not None:
            self._executor.close()
            self._executor = None

    def diagnostics(self) -> dict:
        with self._lock:
            latest = self._reports[-1] if self._reports else None
            span = (latest.host_capture_time-self._reports[0].host_capture_time
                    if len(self._reports) > 1 else 0.)
            tracks = latest.peer.tracks if latest is not None else ()
            bound = next((t for t in tracks if t.track_id == self._bound_track_id), None)
            candidate = next((t for t in tracks if t.track_id == self._candidate_track_id), None)
            return {'processed_frames': self._processed, 'current_frames': self._current,
                    'duplicate_frames': self._duplicates, 'errors': self._errors,
                    'last_error': self.last_error, 'clock_epoch': self._epoch,
                    'track_generation': self._generation,
                    'bound_track_id': self._bound_track_id,
                    'bound_track_sample_count': None if bound is None else bound.sample_count,
                    'bound_track_speed_valid': bool(bound is not None and bound.speed_valid),
                    'target_binding_id': self._binding_id,
                    'geometry_binding_serial': self._geometry_binding_serial,
                    'geometry_binding_capture': self._geometry_binding_capture,
                    'geometry_binding_generation': self._geometry_binding_generation,
                    'last_target_binding_begin': self._last_binding_begin,
                    'candidate_track_id': self._candidate_track_id,
                    'candidate_track_sample_count': None if candidate is None else candidate.sample_count,
                    'last_reassociation': self._last_reassociation,
                    'qualified_tracks': sum(t.speed_valid for t in tracks),
                    'last_query_reason': self._last_query_reason,
                    'effective_rate_hz': (len(self._reports)-1)/span if span > 0 else None,
                    'retained_reports': len(self._reports),
                    'retained_report_capacity': self._reports.maxlen,
                    'maximum_qualified_velocity_age_s': self.max_age_s,
                    'mean_processing_ms': (1000*self._processing_total_s/self._processed
                                           if self._processed else None),
                    'max_rate_hz': self.max_rate_hz,
                    'process_isolation': self.process_isolation}

    def captured_reports(self) -> tuple[_CapturedReport, ...]:
        """Detached immutable observations; reading never waits for new RGB-D."""
        with self._lock:
            return tuple(self._reports)

    def accepts_age(self, age_s: float) -> bool:
        """Age admission only; causal ordering is checked separately."""
        return self.max_age_s is None or age_s <= self.max_age_s + 1e-12

    @staticmethod
    def _unavailable(reason: str, *, capture_device_time: float,
                     capture_host_time: float, clock_epoch: int, sequence: int,
                     track_generation: int | None = None) -> dict:
        return {'speed_valid': False, 'velocity_base_m_s': None,
                'motion_source': 'unavailable', 'track_id': None,
                'device_timestamp': None, 'host_capture_timestamp': None,
                'clock_epoch': clock_epoch, 'sequence': None,
                'sample_count': 0, 'span_s': 0., 'speed_std_m_s': None,
                'track_generation': track_generation, 'reason': reason,
                'geometry_device_timestamp': capture_device_time,
                'geometry_host_capture_timestamp': capture_host_time,
                'geometry_sequence': sequence}

    def query(self, target_position_base: Any, *, capture_device_time: float,
              capture_host_time: float, clock_epoch: int, sequence: int) -> dict:
        """Return only an already observed, causally target-bound peer speed."""
        target = np.asarray(target_position_base, dtype=float)
        device = float(capture_device_time)
        host = float(capture_host_time)
        if (target.shape != (3,) or not np.isfinite(target).all()
                or not np.isfinite([device, host]).all()):
            raise ValueError('target position and capture times must be finite')
        epoch, seq = int(clock_epoch), int(sequence)
        def missing(reason):
            self._last_query_reason = reason
            return self._unavailable(reason, capture_device_time=device,
                capture_host_time=host, clock_epoch=epoch, sequence=seq,
                track_generation=self._generation)
        with self._lock:
            if self._epoch is not None and epoch != self._epoch:
                return missing('camera_clock_epoch_mismatch')
            geometry = self._geometry_binding_capture
            if (geometry is not None and self._geometry_binding_generation is not None
                    and self._generation != self._geometry_binding_generation):
                return missing('geometry_tracker_generation_changed')
            if geometry is not None and (epoch != geometry[0]
                    or seq < geometry[1] or device < geometry[2]-1e-9
                    or host < self._geometry_binding_host-1e-9):
                return missing('query_precedes_current_geometry_binding')
            records = list(self._reports)
            if (self.max_age_s is None and self._qualified_report is not None
                    and not any(r is self._qualified_report for r in records)):
                # Pin one actually qualified, target-bound report beyond the
                # rolling camera history. Never republish it as a new frame.
                records.append(self._qualified_report)
                records.sort(key=lambda r: r.peer.sequence)
            causal = [record for record in reversed(records)
                      if record.peer.clock_epoch == epoch
                      and record.peer.source_time <= device + 1e-9
                      and record.host_capture_time <= host + 1e-9
                      and record.peer.sequence <= seq
                      and self.accepts_age(host-record.host_capture_time)]
            if not causal:
                return missing('no_causal_nero7_report')
            gate = float(self.meter.parameters['tracker_gate_radius'])
            # A confirmed target keeps a logical identity across peer track
            # expiry. Never replace a track that is still retained by the peer.
            bound_present = any(t.track_id == self._bound_track_id for t in causal[0].peer.tracks)
            reidentifying = self._bound_track_id is not None and not bound_present
            expected_xy = None
            if reidentifying and self._bound_anchor is not None:
                position, velocity, stamp = self._bound_anchor
                if device >= stamp:
                    expected_xy = position + velocity*(device-stamp)
            matched = None
            matched_index = None
            for index, record in enumerate(causal):
                candidates = []
                for track in record.peer.tracks:
                    if track.misses or track.observation_source_time != record.peer.source_time:
                        continue
                    if (self._bound_track_id is not None and not reidentifying
                            and track.track_id != self._bound_track_id):
                        continue
                    predicted_xy = np.asarray(track.observed_position_base[:2], float).copy()
                    if track.speed_valid and track.velocity_base_m_s is not None:
                        predicted_xy += np.asarray(track.velocity_base_m_s[:2]) * (
                            device-track.observation_source_time)
                    if (reidentifying and track.track_id != self._bound_track_id
                            and (expected_xy is None or np.linalg.norm(predicted_xy-expected_xy) > gate)):
                        continue
                    if np.linalg.norm(predicted_xy-target[:2]) <= gate:
                        candidates.append(track)
                if len(candidates) > 1:
                    return missing('target_track_ambiguous')
                if candidates:
                    matched = candidates[0]
                    matched_index = index
                    break
                if (self._bound_track_id is not None
                        and any(track.track_id == self._bound_track_id and not track.misses
                                for track in record.peer.tracks)):
                    return missing('target_track_mismatch')
            if matched is None:
                return missing('target_track_unmatched')
            self._candidate_track_id = matched.track_id
            # A more recent peer frame that failed geometry association cannot
            # donate its speed merely because it carries the same numeric ID.
            for record in causal[matched_index:]:
                track = next((item for item in record.peer.tracks
                              if item.track_id == matched.track_id and item.speed_valid
                              and item.velocity_base_m_s is not None and not item.misses
                              and item.observation_source_time == record.peer.source_time), None)
                if track is None:
                    continue
                velocity = np.asarray(track.velocity_base_m_s, dtype=float).copy()
                velocity.setflags(write=False)
                previous = self._bound_track_id
                if previous is None:
                    self._binding_id = matched.track_id
                elif previous != matched.track_id:
                    self._last_reassociation = {
                        'previous_track_id': previous, 'track_id': matched.track_id,
                        'binding_id': self._binding_id, 'geometry_sequence': seq,
                        'geometry_device_timestamp': device,
                        'method': 'same_geometry_and_predicted_motion',
                    }
                self._bound_track_id = matched.track_id
                self._bound_anchor = (np.asarray(matched.observed_position_base[:2], float).copy(),
                                      velocity[:2].copy(), matched.observation_source_time)
                if (self.max_age_s is None and (self._qualified_report is None
                        or record.host_capture_time >= self._qualified_report.host_capture_time)):
                    self._qualified_report = record
                self._last_query_reason = 'ok' if record == causal[0] else 'recent_measured_speed'
                return {'speed_valid': True, 'velocity_base_m_s': velocity,
                        'motion_source': 'nero7_xy_kalman', 'track_id': track.track_id,
                        'binding_id': self._binding_id,
                        'device_timestamp': record.peer.source_time,
                        'host_capture_timestamp': record.host_capture_time,
                        'clock_epoch': record.peer.clock_epoch, 'sequence': record.peer.sequence,
                        'sample_count': track.sample_count, 'span_s': track.span_s,
                        'speed_std_m_s': track.speed_std_m_s,
                        'track_generation': record.peer.track_generation,
                        'reason': 'ok' if record == causal[0] else 'recent_measured_speed',
                        'geometry_device_timestamp': device,
                        'geometry_host_capture_timestamp': host,
                        'geometry_sequence': seq}
            return missing('nero7_speed_not_qualified')

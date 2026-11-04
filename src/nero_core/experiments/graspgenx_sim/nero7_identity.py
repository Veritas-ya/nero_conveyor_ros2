"""Keep a task ID using the deployed Nero7 position/velocity binding rule."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
from mujoco_conveyor_sorting.runtime.nero7_grasp_tracking import Nero7GraspTracking
from mujoco_conveyor_sorting.runtime.nero7_velocity_source import Nero7VelocitySource, _CapturedReport
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter


class _HistorySource(Nero7VelocitySource):
    """Use original query over real peer reports, without reader/thread/device.

    Detection is owned by the simulation worker. This meter supplies only the
    exact original query parameters; its update method is never called here.
    """
    def __init__(self, parameters, max_age_s):
        meter = Nero7BeltVelocityMeter(surface=None)
        meter.parameters.update(parameters)
        super().__init__(meter, read_frame=lambda sequence: None,
            calibrate_frame=lambda frame: frame,
            capture_monotonic=lambda frame: float(frame.header.source_time),
            depth_to_color_cv=np.eye(4), max_age_s=max_age_s)

    def append_peer(self, peer):
        self._epoch = peer.clock_epoch
        self._generation = peer.track_generation
        self._reports.append(_CapturedReport(peer, float(peer.source_time)))


class _Binding:
    _observed = staticmethod(Nero7GraspTracking._observed)
    _select_track = Nero7GraspTracking._select_track

    def __init__(self, track, record, parameters, max_age_s):
        self.source = _HistorySource(parameters, max_age_s)
        self.source.append_peer(record.peer)
        self.last_error = None
        self._velocity = None
        self.geometry_begin = None
        self.accept(track, record, self.query(track, record))

    def bind_measured_geometry(self, points_base, peer):
        if self.geometry_begin is not None:
            return self.geometry_begin
        begin = self.source.begin_target_binding(geometry_points_base=points_base,
            capture_device_time=peer.source_time, capture_host_time=peer.source_time,
            clock_epoch=peer.clock_epoch, sequence=peer.sequence)
        if begin['accepted'] and begin['binding_reset']:
            self.geometry_begin = begin
        return begin

    def query(self, track, record):
        peer = record.peer
        return self.source.query(track.observed_position_base,
            capture_device_time=peer.source_time, capture_host_time=peer.source_time,
            clock_epoch=peer.clock_epoch, sequence=peer.sequence)

    def accept(self, track, record, measurement):
        self._peer_id = track.track_id
        self._last_position = np.asarray(track.observed_position_base, float).copy()
        self._last_stamp = float(record.peer.source_time)
        self.measurement = measurement
        if measurement['speed_valid']:
            self._velocity = np.asarray(measurement['velocity_base_m_s'], float).copy()


class Nero7StableIdentityMap:
    """Peer selects identities; active task retains priority over provisional IDs.

    Reassociation executes the hardware function itself, with its existing
    30mm position/velocity gate and unique-candidate rule. Object truth and
    nominal belt speed are never inputs. New clock epochs get fresh task IDs.
    """
    def __init__(self, *, maximum_velocity_age_s=1.):
        self.maximum_velocity_age_s = float(maximum_velocity_age_s)
        self.bindings = {}
        self.next_task_id = 0
        self.clock_key = None

    def update(self, peer, parameters, *, active_task_id=None):
        record = SimpleNamespace(peer=peer)
        clock_key = (peer.clock_epoch, peer.track_generation)
        reset = self.clock_key is not None and clock_key != self.clock_key
        if reset:
            self.bindings.clear()
        self.clock_key = clock_key
        for binding in self.bindings.values():
            binding.source.append_peer(peer)
        observed = Nero7GraspTracking._observed(record)
        assignments, reasons, dropped = {}, {}, []
        task_ids = sorted(self.bindings, key=lambda k: (k != active_task_id, k))
        claimed = set()
        for task_id in task_ids:
            if task_id not in self.bindings:
                continue
            binding = self.bindings[task_id]
            previous_id = binding._peer_id
            track = binding._select_track(record)
            if track is None or track.track_id in claimed:
                reasons[task_id] = binding.last_error or 'retained_peer_not_observed'
                continue
            measurement = binding.query(track, record)
            if (track.track_id != previous_id and binding._velocity is not None
                    and not measurement['speed_valid']):
                # Same retained-qualified-identity condition as hardware
                # _publish: a new unqualified peer must not replace it yet.
                reasons[task_id] = measurement['reason']
                continue
            competitors = [k for k, b in self.bindings.items()
                           if k != task_id and b._peer_id == track.track_id]
            if task_id == active_task_id:
                dropped.extend(competitors)
                for competitor in competitors:
                    self.bindings.pop(competitor, None)
            elif competitors:
                continue
            binding.accept(track, record, measurement)
            assignments[task_id] = track
            claimed.add(track.track_id)
            reasons[task_id] = ('same_peer_id' if previous_id == track.track_id
                                else 'peer_observation_and_predicted_motion')
        for task_id in dropped:
            assignments.pop(task_id, None)
        for track in observed:
            if track.track_id in claimed:
                continue
            task_id = self.next_task_id
            self.next_task_id += 1
            self.bindings[task_id] = _Binding(track, record, parameters, self.maximum_velocity_age_s)
            assignments[task_id] = track
            reasons[task_id] = 'new_peer_observation'
        return assignments, {'task_reassociation_reasons': reasons,
            'dropped_provisional_task_ids': sorted(set(dropped)), 'clock_or_generation_reset': reset,
            'task_to_actual_peer_id': {task: track.track_id for task, track in assignments.items()}}

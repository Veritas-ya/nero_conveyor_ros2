"""One RGB-D camera, original Nero7 XY Kalman, same-exposure measured OBB."""
from dataclasses import dataclass, field
from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.calibration.geometry import rigid_transform
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, BeltSurfaceNotReady
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.single_pick import detect_target, PickRejected, TargetNotReady
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.feedback_pairing import capture_joint_pairing_valid
from .grasp import MIN_VISIBLE_POINTS, VisibleOBBError, visible_belt_aligned_obb
from .measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
from .reference import predict_entry_time, EntryNotReady
from .synchronized_reference import admit_synchronized_observation, freeze_window_for_speed
from .tracked_freeze import native_exposure_window, pairing_deadline, upstream_at
from .diagnostics import VisionProgress
from .velocity_source import KALMAN, POINT_KALMAN, FIXED, is_fixed, prediction_velocity, speed_report as make_speed_report, policy as velocity_policy


def offset_grasp_tcp_z(grasps, box, offset_m):
    """Shift candidate TCPs in base Z; measured object geometry stays fixed."""
    if (isinstance(offset_m, bool) or not isinstance(offset_m, (int, float))
            or not np.isfinite(offset_m) or abs(offset_m) > .05):
        raise ValueError('grasp_tcp_z_offset_m 须为-0.05至0.05米的有限数')
    adjusted = [np.array(grasp, dtype=float, copy=True) for grasp in grasps]
    for grasp in adjusted:
        grasp[2, 3] += offset_m
    return adjusted, dict(box, grasp_tcp_offset_base_m=[0., 0., float(offset_m)])


def obb_grasps(points_base, plane_to_base, canonical_tcp, *, tcp_z_offset_m=0.):
    """Same visible rectangle as simulation, in the MEASURED belt frame.

    TCP origins are the visible box centre plus the configured base-Z offset.
    Tool axes use the supplied physical
    canonical-to-TCP calibration; no simulation gripper assets are read.
    """
    plane = rigid_transform(plane_to_base)
    tool = rigid_transform(canonical_tcp)
    local = (np.asarray(points_base) - plane[:3, 3]) @ plane[:3, :3]
    box = visible_belt_aligned_obb(local)
    axes = np.asarray(box['axes_world_columns'])
    extents = np.asarray(box['extent_m'])
    result = []
    # Keep only the short closing axis and its finger-symmetric 180-degree
    # equivalent. A failed short-axis plan must never switch to the long axis.
    for axis in (int(np.argmin(extents[:2])),):
        for flip in (1., -1.):
            x = axes[:, axis] * flip
            z = np.array([0., 0., -1.])
            grasp = np.eye(4)
            grasp[:3, :3] = plane[:3, :3] @ np.column_stack((x, np.cross(z, x), z)) @ tool[:3, :3]
            grasp[:3, 3] = plane[:3, :3] @ np.asarray(box['center_world_m']) + plane[:3, 3]
            result.append(grasp)
    return offset_grasp_tcp_z(result, box, tcp_z_offset_m)


def record_grasp_box(detection, box, grasps, center):
    detection['obb'] = box
    if detection.get('target_geometry_source') == 'fast_depth_at_target_pixels':
        footprint = sorted(box['extent_m'][:2], reverse=True)
        detection.update(center=center.copy(), grasp=grasps[0].copy(), footprint_m=footprint,
                         width=footprint[1], localization_points=box['visible_point_count'])


@dataclass(frozen=True)
class MeasuredTarget:
    identity: tuple
    capture_s: float
    received_s: float
    device_s: float
    epoch: int
    sequence: int
    center: np.ndarray
    velocity: np.ndarray
    grasps: tuple
    points: np.ndarray
    detection: dict
    state: dict
    speed_report: dict
    entry_position: np.ndarray | None = None
    object_pose: np.ndarray = field(init=False)
    grasps_object: tuple = field(init=False)

    def __post_init__(self):
        if (len(self.identity) != 3 or self.identity[0] != self.epoch or
            not np.isfinite([self.capture_s,self.received_s,self.device_s]).all() or
            self.capture_s > self.received_s):
            raise ValueError('观测时间或目标身份无效')
        for name, shape in (('center', (3,)), ('velocity', (3,))):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid measured '+name)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        entry = self.center if self.entry_position is None else np.asarray(self.entry_position, dtype=float)
        if entry.shape != (3,) or not np.isfinite(entry).all():
            raise ValueError('invalid entry observation position')
        entry = entry.copy()
        entry.setflags(write=False)
        object.__setattr__(self, 'entry_position', entry)
        points = np.asarray(self.points, dtype=float).copy()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < MIN_VISIBLE_POINTS or not np.isfinite(points).all():
            raise ValueError('需要单帧有限目标点云')
        points.setflags(write=False)
        object.__setattr__(self, 'points', points)
        poses = []
        for pose in self.grasps:
            value = rigid_transform(pose)
            value.setflags(write=False)
            poses.append(value)
        source = self.speed_report.get('source')
        if not poses or source not in (KALMAN, POINT_KALMAN, FIXED):
            raise ValueError('必须提供 OBB 候选及明确的视觉/用户设定速度来源')
        if source == FIXED:
            speed = self.speed_report.get('configured_speed_m_s')
            if (speed is None or not np.isfinite(speed) or speed <= 0
                    or not np.isclose(np.linalg.norm(self.velocity), speed, rtol=0, atol=1e-10)
                    or self.speed_report.get('visual_velocity_used') is not False
                    or self.speed_report.get('measured_velocity_base_m_s') is not None):
                raise ValueError('用户固定带速报告与实际预测速度不一致，不能标为视觉实测')
        object.__setattr__(self, 'grasps', tuple(poses))
        plane = np.asarray(self.detection['belt_region']['plane_to_base'])
        box = self.detection['obb']
        object_pose = np.eye(4)
        object_pose[:3,:3] = plane[:3,:3] @ np.asarray(box['axes_world_columns'])
        object_pose[:3,3] = plane[:3,:3] @ np.asarray(box['center_world_m']) + plane[:3,3]
        object_pose = rigid_transform(object_pose, 'measured visible OBB frame')
        object_pose.setflags(write=False)
        local = []
        for grasp in poses:
            relative = np.linalg.inv(object_pose) @ grasp
            relative.setflags(write=False)
            local.append(relative)
        object.__setattr__(self, 'object_pose', object_pose)
        object.__setattr__(self, 'grasps_object', tuple(local))


class LiveVision:
    """Device lifecycle is owned by the caller; this thread only reads cache.

    It never starts a second camera, sends CAN, saves large arrays, or loads a
    network per target. Grasp source is explicitly OBB, not GraspGen-X.
    """
    def __init__(self, camera, calibration, config, model, read_feedback):
        self.camera, self.calibration, self.config = camera, calibration, config
        self.model, self.read_feedback = model, read_feedback
        self.stop = Event()
        self.lock = Lock()
        from .frozen_queue import FrozenTargetQueue
        self.frozen_queue=FrozenTargetQueue(config.raw.get('tracked_frozen_queue_capacity',1))
        self.latest = None
        self.error = None
        self.progress = VisionProgress()
        self.geometry_progress=VisionProgress()
        self.geometry_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='measured-grasp-geometry')
        self.geometry_future=None
        self.geometry_generation=0
        self.geometry_attempts={}
        self.geometry_failure_archives={}
        self.geometry_model=None
        self.fast_stereo=None
        self.fast_stereo_lock = Lock()
        self.progress_snapshot = None
        self.perception_diagnostics={}
        self.progress_entry = None
        self.clock_epoch = None
        self.consumed = set()
        self.freeze_position_missed = set()
        self.active_task = None
        self.lookahead_enabled = False
        self.recent_targets = []
        self.minimum_capture_s = float('-inf')
        self.reset_tracking = False
        self.thread = Thread(target=self._run, name='measured-two-second-vision')
        self.meter = None
        self.camera_consumption={'frames':0,'capture_sequence_skips':0,'max_device_gap_s':0.}
        self._previous_camera_frame=None
        self.match_in_flight = None
        self.native_band_only = (config.raw.get('freeze_source') == 'single_exposure_native_depth'
            and config.raw.get('tracked_freeze_publish_before_upstream_m') is not None)
        self.single_frame_backup = None
        self.tracked_feedback_worker = None
        if config.raw.get('tracked_freeze_publish_before_upstream_m') is not None and is_fixed(config):
            from .capture_feedback_worker import CaptureFeedbackWorker
            from .tracked_single_exposure import SingleFrameBackup
            if not self.native_band_only:
                self.tracked_feedback_worker = CaptureFeedbackWorker(self, self.geometry_pool, self.geometry_progress,
                                                                    lambda: time.monotonic())
            self.single_frame_backup = SingleFrameBackup(self)

    def start(self):
        if not self.native_band_only:
            self.geometry_future=self.geometry_pool.submit(self._warm_geometry_model)
        if self.single_frame_backup is not None:
            self.single_frame_backup.start()
        self.thread.start()
        return self

    @property
    def latest(self):
        return self.frozen_queue.head

    @latest.setter
    def latest(self,item):
        # Compatibility with the single-exposure entry and explicit resets.
        self.frozen_queue.clear()
        if item is not None:
            from .frozen_queue import FrozenEntry
            self.frozen_queue.entries.append(FrozenEntry(item,time.monotonic()))

    def _prune_frozen_locked(self,now,*,include_single=False):
        if self.frozen_queue.capacity==1 and not include_single:return
        for entry in tuple(self.frozen_queue.entries):
            item=entry.target
            reason=entry.invalid_reason(self.config,self.clock_epoch,now)
            if (reason is None and self.frozen_queue.capacity == 1
                    and now - item.capture_s > self.config.raw['maximum_observation_age_s']):
                reason = 'stale_at_handoff'
            if reason:
                self.frozen_queue.remove(item.identity,reason)
                self.consumed.add(item.identity)
                sink=getattr(self,'event_sink',None)
                if sink is not None:
                    sink('frozen_target_expired', identity=item.identity,
                        frame_sequence=item.sequence, capture_monotonic_s=item.capture_s,
                        reason=reason, observation_age_s=now-item.capture_s,
                        coordinate_m=float((item.entry_position+item.velocity*(now-item.capture_s))
                                           @ self.config.direction))

    def pending_admission(self,item,now):
        """Re-use an exposure accepted fresh; never retimestamp a queued target."""
        with self.lock:
            return (item.epoch==self.clock_epoch and item.identity not in self.consumed
                and self.frozen_queue.admitted(item,now,self.config.raw['maximum_observation_age_s'])
                and (self.frozen_queue.capacity>1 or now-item.capture_s<=self.config.raw['maximum_observation_age_s'])
                and float((item.entry_position+item.velocity*(now-item.capture_s))@self.config.direction)
                    <=self.config.follow_bounds[1])

    def poll(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.active_task is not None:
                return None
            item = self.latest
            if item is not None and item.identity not in self.consumed:
                return item
        return None

    def consume(self, identity):
        with self.lock:
            self._consume_locked(identity)

    def _consume_locked(self, identity):
        item=self.frozen_queue.remove(identity,'consumed')
        if (self.lookahead_enabled or self.single_frame_backup is not None) and item is not None:
            self.recent_targets = (self.recent_targets+[item])[-4:]
        if self.frozen_queue.capacity==1:self.geometry_generation+=1
        self.consumed.add(identity)
        self.active_task = identity

    def claim_frozen(self, *, released_owner=None):
        """Atomically take a valid exposure from idle or the confirmed release owner."""
        from .lookahead_selection import latest_dispatch
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.active_task != released_owner:
                return None
            if released_owner is not None and (
                    not self.lookahead_enabled or released_owner[0] != self.clock_epoch):
                return None
            self._prune_frozen_locked(time.monotonic(), include_single=True)
            entry = min(self.frozen_queue.entries, default=None,
                key=lambda e: (latest_dispatch(e.target, self.config), e.target.capture_s))
            if entry is not None:
                self._consume_locked(entry.target.identity)
            return entry

    def resume_targets(self):
        """Rearm after return, preserving exposures when acquisition stayed active."""
        with self.lock:
            if self.lookahead_enabled:
                # Preserve the next exposure and its original capture clock.
                self.active_task = None
                return
            self.geometry_generation+=1
            self.latest = self.active_task = None
            self.minimum_capture_s = time.monotonic()
            # Only the camera worker may mutate the meter/tracker.
            self.reset_tracking = True
            self.progress_snapshot = self.progress_entry = None

    def enable_lookahead(self):
        with self.lock:
            self.lookahead_enabled = True

    def capture_feedback(self, captured):
        reader = (getattr(self, 'capture_state_reader', None)
                  if self.lookahead_enabled or self.single_frame_backup is not None else None)
        return self.read_feedback() if reader is None else reader(captured)

    def _paired_state(self, captured, *, progress=None):
        progress = self.progress if progress is None else progress
        progress.enter('pairing_feedback')
        progress.annotate_frame(exposure_age_at_pairing_s=time.monotonic()-captured)
        with progress.measure('feedback_pairing'):
            state = self.capture_feedback(captured)
        if state is None:
            progress.block('waiting_feedback', 'feedback_missing', capture_monotonic_s=captured)
            return None
        return self._validate_capture_state(captured, state, progress)

    def _first_paired_capture(self, requests, progress, wakeup):
        batch_reader = getattr(self, 'capture_states_reader', None)
        if batch_reader is not None:
            pair, armed = batch_reader(tuple(req.captured for req in requests), wakeup=wakeup)
            if pair is None:
                return None, armed
            stamp, state = pair
            req = next(req for req in requests if req.captured == stamp)
            if self._validate_capture_state(stamp, state, progress) is not None:
                return (req, state), False
            return None, False
        for req in requests:
            state = self.capture_feedback(req.captured)
            if state is not None and self._validate_capture_state(req.captured, state, progress) is not None:
                return (req, state), False
        return None, False

    def _validate_capture_state(self, captured, state, progress):
        stamps = [mapped_stamp(state, kind) for kind in ('joint', 'pose')]
        progress.annotate_frame(joint_feedback_monotonic_s=stamps[0],
            pose_feedback_monotonic_s=stamps[1])
        if 'capture_joint_pairing' in state or 'capture_state_role' in state:
            if not capture_joint_pairing_valid(state, captured):
                progress.block('waiting_feedback', 'capture_joint_pairing_invalid',
                    capture_monotonic_s=captured, pairing=state.get('capture_joint_pairing'))
                return None
            progress.annotate_frame(capture_feedback_basis='measured_historical_joint_brackets',
                capture_joint_pairing=state['capture_joint_pairing'])
        else:
            if any(stamp is None or abs(stamp-captured) > .2 for stamp in stamps):
                progress.block('waiting_feedback',
                    'feedback_time_missing' if any(s is None for s in stamps) else 'feedback_frame_skew',
                    capture_monotonic_s=captured, feedback_monotonic_s=stamps, maximum_skew_s=.2,
                    joint_monotonic_s=stamps[0], pose_monotonic_s=stamps[1],
                    feedback_minus_capture_s=[None if s is None else s-captured for s in stamps])
                return None
            progress.annotate_frame(capture_feedback_basis='current_feedback_with_200ms_limit')
        return state

    def poll_pending(self):
        """Read one next exposure without changing the execution owner."""
        with self.lock:
            if self.error is not None:
                raise self.error
            self._prune_frozen_locked(time.monotonic())
            item = self.latest
            return item if item is not None and item.identity not in self.consumed else None

    def pending_candidates(self):
        """Immutable exposures available to the idle private planning worker."""
        with self.lock:
            if self.error is not None:
                raise self.error
            self._prune_frozen_locked(time.monotonic())
            return tuple(e.target for e in self.frozen_queue.entries
                         if e.target.identity not in self.consumed)

    def reserve_pending(self, target):
        with self.lock:
            self._prune_frozen_locked(time.monotonic())
            return target.identity not in self.consumed and self.frozen_queue.promote(target)

    def pending_status(self, target, now):
        """Classify expiry before pruning can turn it into an identity mismatch."""
        with self.lock:
            if target.epoch != self.clock_epoch or now < target.capture_s:
                return 'clock_invalidated'
            coordinate = float((target.entry_position+target.velocity*(now-target.capture_s))
                               @ self.config.direction)
            if coordinate > self.config.follow_bounds[1]:
                return 'past_follow_region'
            if target.identity in self.consumed:
                return 'consumed_or_cancelled'
            if not any(e.target is target for e in self.frozen_queue.entries):
                return 'exposure_replaced'
            return None

    def discard_pending(self, identity):
        with self.lock:
            self.consumed.add(identity)
            item=self.frozen_queue.remove(identity,'discarded')
            if item is not None:
                if self.lookahead_enabled:
                    self.recent_targets = (self.recent_targets+[item])[-4:]
                if self.frozen_queue.capacity==1:self.geometry_generation += 1

    def repeats_recent_target(self, position, captured):
        """Suppress one physical piece across exposure IDs or tracker ID churn."""
        with self.lock:
            return self._repeats_locked(position, captured)

    def _repeats_locked(self, position, captured, *, frame_sequence=None, mask=None, seed_indices=None):
        recent = tuple(self.recent_targets)+tuple(e.target for e in self.frozen_queue.entries)
        for item in recent:
            if item.epoch != self.clock_epoch:
                continue
            dt = captured-item.capture_s
            hybrid = self.single_frame_backup is not None
            if dt < 0 and not hybrid:
                continue
            if hybrid and frame_sequence == item.sequence:
                other = item.detection.get('mask')
                if other is not None and seed_indices is not None:
                    if 2*np.count_nonzero(other.ravel()[seed_indices]) >= len(seed_indices):
                        return True
                    continue
                if other is not None and mask is not None and mask.shape == other.shape:
                    shared = np.count_nonzero(mask & other)
                    if shared and 2*shared >= min(np.count_nonzero(mask), np.count_nonzero(other)):
                        return True
                    continue
            if hybrid:
                # A long narrow part and a neighbour beside it can be closer
                # than half its diagonal. Compare in its measured footprint.
                box = item.object_pose
                local = (np.asarray(position)-item.velocity*dt-box[:3,3]) @ box[:3,:2]
                half_extent = np.asarray(item.detection['obb']['extent_m'])[:2]/2
                if np.all(np.abs(local) <= half_extent + 1e-9):
                    return True
                continue
            predicted = item.entry_position+item.velocity*dt
            radius = max(.025, float(np.linalg.norm(np.ptp(item.points, axis=0)[:2]))*.5)
            if np.linalg.norm(np.asarray(position)[:2]-predicted[:2]) <= radius:
                return True
        return False

    def diagnostics(self):
        """Small detached progress report; never used to authorize movement."""
        with self.lock:
            peer, entry = self.progress_snapshot, self.progress_entry
            consumed = set(self.consumed)
            active_task = self.active_task
            missed_count = len(self.freeze_position_missed)
            frozen_queue = self.frozen_queue.snapshot()
        tracks = [] if peer is None else [{
            'id': t.track_id, 'confirmed': t.confirmed, 'speed_valid': t.speed_valid,
            'samples': t.sample_count, 'span_s': round(float(t.span_s), 3),
            'misses': t.misses, 'velocity_base_m_s': t.velocity_base_m_s,
            'observed_position_base_m': t.observed_position_base,
            'speed_std_m_s': t.speed_std_m_s,
            'velocity_covariance_xy': t.velocity_covariance_xy,
            'rejection_reasons': self._track_rejections(t, peer, consumed),
            'along_belt_m_s': None if t.velocity_base_m_s is None else
                float(np.asarray(t.velocity_base_m_s) @ self.config.direction),
        } for t in peer.tracks]
        return {**self.progress.snapshot(), 'entry_gate': entry,
                'frozen_queue':frozen_queue,
                'single_frame_detection': (self.single_frame_backup.progress.snapshot()
                    if self.native_band_only and self.single_frame_backup is not None else None),
                'single_frame_fallback_last_rejection': (self.single_frame_backup.last_rejection
                    if self.single_frame_backup is not None else None),
                'capture_feedback_wait': (dict(tracked=(self.tracked_feedback_worker.diagnostics()
                    if self.tracked_feedback_worker is not None else None),
                    single_frame=self.single_frame_backup.feedback_worker.diagnostics())
                    if self.single_frame_backup is not None else None),
                'grasp_geometry':self.geometry_progress.snapshot(),
                'geometry_in_flight':self.geometry_future is not None,
                'perception_detection_mode':self.config.raw.get('perception_detection_mode','rgb_depth'),
                'perception':self.perception_diagnostics,
                'velocity_policy':velocity_policy(self.config),
                'active_task': active_task,
                'target_processing_paused': active_task is not None and not self.lookahead_enabled,
                'freeze_position_missed_targets': missed_count,
                'meter_current': None if peer is None else peer.current,
                'meter_frame_sequence': None if peer is None else peer.sequence,
                'meter_reason': None if peer is None else peer.reason, 'tracks': tracks,
                'worker_alive': self.thread.is_alive(), 'stop_requested': self.stop.is_set(),
                'camera_capture':(self.camera.get_capture_diagnostics()
                    if callable(getattr(self.camera,'get_capture_diagnostics',None)) else None),
                'camera_consumption':dict(self.camera_consumption),
                'thresholds': {key: self.config.raw.get(key) for key in (
                    'minimum_speed_span_s', 'maximum_observation_age_s',
                    'tracked_freeze_upstream_limit_m', 'tracked_freeze_publish_before_upstream_m',
                    'freeze_upstream_distance_m', 'freeze_position_upstream_m',
                    'freeze_position_tolerance_m', 'planning_reserve_s', 'maximum_measured_speed_m_s')}}

    def _track_rejections(self, track, peer, consumed):
        reasons = []
        fixed = is_fixed(self.config)
        identity = (peer.clock_epoch, peer.track_generation, track.track_id)
        consumed_reason = ('past_fixed_freeze_position' if identity in self.freeze_position_missed
                           else 'already_consumed')
        for rejected, reason in (
            (not track.confirmed, 'track_unconfirmed'),
            (not fixed and not track.speed_valid, 'speed_invalid'),
            (bool(track.misses), 'track_missed'),
            (not fixed and track.velocity_base_m_s is None, 'velocity_missing'),
            (identity in consumed, consumed_reason),
            (not fixed and not track.span_s >= self.config.raw['minimum_speed_span_s'], 'speed_span_too_short')):
            if rejected:
                reasons.append(reason)
        if not fixed and track.velocity_base_m_s is not None and not (
                np.asarray(track.velocity_base_m_s) @ self.config.direction > 0):
            reasons.append('nonpositive_belt_velocity')
        return reasons

    def _select_geometry_track(self,frame,peer,valid,captured):
        age=time.monotonic()-captured
        if age>self.config.raw['maximum_observation_age_s']:
            self.progress.block('waiting_fresh_observation','observation_stale_before_detection',
                observation_age_s=age,maximum_age_s=self.config.raw['maximum_observation_age_s'])
            return None
        # A failed downstream component must not monopolize the worker. Give
        # untried pieces a turn, then retry oldest attempts; first ties use urgency.
        ordered=sorted(valid,key=lambda t:(
            self.geometry_attempts.get((peer.clock_epoch,peer.track_generation,t.track_id),-1),
            -float(np.asarray(t.observed_position_base)@self.config.direction)))
        for track in ordered:
            identity=(peer.clock_epoch,peer.track_generation,track.track_id)
            position=np.asarray(track.observed_position_base,dtype=float)
            velocity=prediction_velocity(self.config,track.velocity_base_m_s,
                self.model.measured_belt_surface.normal_base)
            self.progress.annotate_frame(track_id=track.track_id,
                track_generation=peer.track_generation,observed_position_base_m=position.tolist(),
                velocity_base_m_s=velocity.tolist(),samples=track.sample_count,span_s=track.span_s)
            try:
                self.entry_time(captured,position,velocity,time.monotonic())
            except EntryNotReady as error:
                if error.reason=='past_fixed_freeze_position':
                    with self.lock:
                        self.freeze_position_missed.add(identity)
                        self.consumed.add(identity)
                continue
            seed=None
            if self.config.raw.get('tracked_local_geometry',False):
                with self.progress.measure('matched_geometry_seed'):
                    seed=self.meter.geometry_seed_rgb(frame,track)
            if self.frozen_queue.capacity>1 and len(peer.tracks)>1 and seed is None:
                self.progress.block('waiting_target_association','missing_multi_target_seed',
                    track_id=track.track_id)
                continue
            return track,seed
        return None

    def entry_time(self, capture_s, position, velocity, now_s, *, progress=None, belt_region=None):
        """Same upstream admission as simulation, projected into the cell belt axis."""
        coordinate = float(np.asarray(position) @ self.config.direction)
        entry_coordinate = float(self.config.follow_bounds[0])
        along = float(np.asarray(velocity) @ self.config.direction)
        predicted = capture_s+(entry_coordinate-coordinate)/along if along > 0 else None
        reserve = self.config.raw['planning_reserve_s'] + self.config.raw.get('prealign_orientation_s', 0.)
        detail = {'coordinate_m': coordinate, 'distance_to_entry_m': entry_coordinate-coordinate,
            'along_belt_m_s': along, 'observation_age_s': now_s-capture_s,
            'scheduled_start_monotonic_s': None if predicted is None else predicted-self.config.raw.get('prealign_orientation_s', 0.),
            'remaining_lead_s': None if predicted is None else predicted-now_s,
            'required_lead_s': reserve,
            'maximum_upstream_distance_m': self.config.raw['freeze_upstream_distance_m']}
        try:
            if self.config.raw.get('intercept_strategy', 'classic') == 'synchronized':
                freeze_window = freeze_window_for_speed(self.config, along,
                    surface=getattr(self.model, 'measured_belt_surface', None), belt_region=belt_region)
                detail['freeze_window'] = freeze_window
                detail['maximum_upstream_distance_m'] = freeze_window['distance_m']
                detail['minimum_upstream_distance_m'] = freeze_window.get('minimum_distance_m')
                scheduled = admit_synchronized_observation(capture_s, coordinate, along,
                    self.config.follow_bounds, now_s, freeze_window['distance_m'],
                    minimum_upstream_distance_m=freeze_window.get('minimum_distance_m'))
                cutoff = self.config.raw.get('tracked_freeze_publish_before_upstream_m')
                if (cutoff is not None and not native_exposure_window(self.config)
                        and upstream_at(self.config, position, velocity, capture_s, now_s) < cutoff):
                    raise EntryNotReady('物块已越过冻结发布截止线', 'freeze_publication_deadline')
                detail['admission_rule'] = ('fixed_capture_position_band; exact_schedule_after_obb'
                    if freeze_window['policy'] == 'fixed_capture_position' else
                    'synchronized_follow_window; exact_schedule_after_obb')
            else:
                scheduled = predict_entry_time(capture_s, coordinate,
                    along, entry_coordinate,
                    now_s, reserve,
                    maximum_upstream_distance_m=self.config.raw['freeze_upstream_distance_m'])
        except EntryNotReady as error:
            with self.lock:
                self.progress_entry = {'reason': error.reason, **detail}
            (self.progress if progress is None else progress).block('waiting_upstream_window', error.reason, **detail)
            raise
        with self.lock:
            self.progress_entry = {'reason': 'eligible', **detail}
        return scheduled

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=3.)
        if self.thread.is_alive():
            raise RuntimeError('视觉线程尚未退出')
        if self.tracked_feedback_worker is not None:
            self.tracked_feedback_worker.close()
        self.geometry_pool.shutdown(wait=True,cancel_futures=True)
        if self.single_frame_backup is not None:
            self.single_frame_backup.close()

    def _finish_geometry(self):
        # Both the camera loop and completion callback may drain this job.
        # Claim, validate and publish atomically so a second drain cannot race
        # a new submission or republish a consumed/cancelled result.
        with self.lock:
            future=self.geometry_future
            if future is None or not future.done():return
            self.geometry_future=None
            if self.match_in_flight is not None and self.match_in_flight[-1] is future:
                self.match_in_flight = None
            try:result=future.result()
            except BaseException as error:
                self.error=error
                self.geometry_progress.block('geometry_error','geometry_exception',
                    error_type=type(error).__name__,error=str(error))
                return
            if result is None:return
            generation,tracking_generation,item=result
            publication = self._accept_geometry_locked(generation, tracking_generation, item)
        self._notify_frozen(item, publication)

    def _accept_geometry_locked(self, generation, tracking_generation, item):
        peer = self.progress_snapshot
        if (self.stop.is_set() or self.error is not None or generation != self.geometry_generation or self.clock_epoch != item.epoch
                or (self.active_task is not None and not self.lookahead_enabled)
                or item.identity in self.consumed or item.capture_s < self.minimum_capture_s
                or (tracking_generation is not None and (peer is None or peer.track_generation != tracking_generation))):
            return None
        self._prune_frozen_locked(time.monotonic())
        if self.single_frame_backup is not None and self._repeats_locked(
                item.center, item.capture_s, frame_sequence=item.sequence, mask=item.detection.get('mask')):
            return None
        delivered = time.monotonic()
        if not 0 <= delivered-item.capture_s <= self.config.raw['maximum_observation_age_s']:
            self.geometry_progress.block('waiting_fresh_observation','observation_stale_at_delivery',
                source_sequence=item.sequence)
            return None
        cutoff = self.config.raw.get('tracked_freeze_publish_before_upstream_m')
        capture_distance = float(self.config.follow_bounds[0] - item.entry_position @ self.config.direction)
        distance = upstream_at(self.config, item.entry_position, item.velocity, item.capture_s, delivered)
        if cutoff is not None:
            # Use the more downstream of the track observation and segmented cloud.
            distance = min(distance, upstream_at(self.config, item.center, item.velocity, item.capture_s, delivered))
            center_at_capture = float(self.config.follow_bounds[0] - item.center @ self.config.direction)
            too_early = max(capture_distance, center_at_capture) > self.config.raw['tracked_freeze_upstream_limit_m'] + 1e-9
            exposure_gate = native_exposure_window(self.config)
            reason = None
            if too_early:
                reason = 'too_far_upstream'
            elif exposure_gate:
                if min(capture_distance, center_at_capture) < self.config.raw['native_single_min_capture_upstream_m']-1e-9:
                    reason = 'exposure_past_native_capture_band'
                elif distance < self.config.follow_bounds[0]-self.config.follow_bounds[1]:
                    reason = 'past_follow_region'
            elif distance < cutoff:
                reason = 'freeze_publication_deadline'
            if reason:
                details = dict(source_sequence=item.sequence, upstream_distance_m=capture_distance,
                               estimated_upstream_distance_m=distance, minimum_upstream_distance_m=(
                                   self.config.raw['native_single_min_capture_upstream_m'] if exposure_gate else cutoff))
                timings = item.detection.get('freeze_delivery', {})
                if 'capture_feedback_retried' in timings:
                    details.update(capture_feedback_retried=timings['capture_feedback_retried'],
                                   capture_feedback_wait_s=timings['capture_feedback_wait_s'])
                self.geometry_progress.block('waiting_upstream_window', reason, **details)
                sink = getattr(self, 'event_sink', None)
                if sink is not None:
                    sink('target_freeze_rejected', reason=reason, identity=item.identity,
                         freeze_source=item.speed_report.get('acquisition','tracked'), **details)
                return None
        if not self.frozen_queue.offer(item, delivered, self.config.raw['maximum_observation_age_s']):
            return None
        timings = item.detection.setdefault('freeze_delivery', {})
        timings.update(published_monotonic_s=delivered, exposure_to_ready_s=delivered-item.capture_s,
            geometry_done_to_publish_s=delivered-timings.get('geometry_done_monotonic_s',delivered))
        return dict(active_task=self.active_task, published_monotonic_s=delivered,
            freeze_gate='native_exposure_band' if native_exposure_window(self.config) else 'configured_window',
            upstream_distance_m=capture_distance, published_upstream_distance_m=distance,
            freeze_source=getattr(item,'speed_report',{}).get('acquisition','tracked'), freeze_delivery=dict(timings))

    def _notify_frozen(self, item, publication):
        if publication is None:
            return
        self.progress.mark_ready()
        sink=getattr(self,'event_sink',None)
        if sink is not None:
            sink('target_frozen',identity=item.identity,frame_sequence=item.sequence,
                capture_monotonic_s=item.capture_s,**publication)
        next_pick_sink = getattr(self, 'next_pick_sink', None)
        if next_pick_sink is not None:
            next_pick_sink(publication['active_task'], item)

    def _geometry_completed(self, future):
        self._finish_geometry()

    def _warm_geometry_model(self):
        if self.geometry_model is None:
            self.geometry_progress.enter('preparing_geometry_model')
            with self.geometry_progress.measure('model_startup'):
                from .planning import calibrated_model
                self.geometry_model=calibrated_model(self.config)

    def _refine_stereo(self, *args, **kwargs):
        with self.fast_stereo_lock:
            return self.fast_stereo.refine(*args, **kwargs)

    def _record_geometry_failure(self,error,frame,d2c,state,identity,surface,indices):
        """Preserve the exact failed exposure and subfilter decision, off-thread IO."""
        detail=dict(error.details)
        self.geometry_progress.block('waiting_target_geometry','detection_'+detail.get('reason','no_candidates'),
            error=str(error),candidate_count=detail.get('candidate_count'),details=detail)
        event=getattr(self,'event_sink',None)
        if event is not None:
            event('target_geometry_rejected',identity=identity,
                frame_sequence=frame.header.sequence,
                capture_monotonic_s=self.geometry_progress.snapshot()['frame']['capture_monotonic_s'],
                error=str(error),details=detail)
        sink=getattr(self,'geometry_rejection_sink',None)
        # One representative RGB-D frame per identity/subfilter outcome. Keep
        # every small decision event, but never copy megapixel arrays every retry.
        stages=detail.get('candidate_filter_stages',{})
        key=(identity,detail.get('reason'),tuple(sorted(stages.items())))
        if sink is not None and key not in self.geometry_failure_archives:
            self.geometry_failure_archives[key]=frame.header.sequence
            if len(self.geometry_failure_archives)>64:
                self.geometry_failure_archives.pop(next(iter(self.geometry_failure_archives)))
            sink(frame,d2c,state,identity,surface,indices,detail)

    def _prepare_target(self, frame, d2c, peer, track, state, captured, generation, surface, gate,
                        *, seed_rgb_mask=None, submitted_s=None):
        """A single bounded job with its own robot model and exposure identity."""
        started=time.monotonic()
        progress=self.geometry_progress
        progress.frame(sequence=frame.header.sequence,clock_epoch=peer.clock_epoch,
            track_id=track.track_id,capture_monotonic_s=captured)
        if self.stop.is_set():return
        self._warm_geometry_model()
        progress.enter('detecting_target')
        indices=None
        try:
            with progress.measure('target_detection'):
                kwargs=dict(fit_plane=fit_plane,green_belt_points=green_belt_points,
                    retain_native_seed=self.config.raw.get('depth_separation_enabled',False),
                    rgb_cloud_completion=self.config.raw.get('rgb_cloud_completion_enabled',False),
                    target_shape='simple',robot_model=self.geometry_model,robot_state=state,
                    measured_surface=surface,maximum_target_height_m=self.config.raw['maximum_target_height_m'],
                    maximum_target_length_m=self.config.raw.get('maximum_target_length_m', .16),
                    belt_edge_margin_m=self.config.raw.get('perception_belt_edge_margin_m', .025),
                    belt_near_camera_margin_m=self.config.raw.get('perception_belt_near_camera_margin_m', 0.),
                    belt_direction_base=self.config.direction,
                    scene_scope=self.config.raw.get('environment_collision_scope', 'observed_scene'))
                if seed_rgb_mask is not None and self.config.raw.get('tracked_local_geometry',False):
                    from mujoco_conveyor_sorting.runtime.visible_target import project_rgb_labels
                    from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
                    from mujoco_conveyor_sorting.runtime.frame_products import world_points
                    points=world_points(frame,d2c)
                    heights=(points-surface.plane_to_base[:3,3])@surface.normal_base
                    foreground=(heights>=.012)&(heights<=self.config.raw['maximum_target_height_m'])
                    seed=project_rgb_labels(frame,d2c,seed_rgb_mask,pixel_mask=foreground)
                    indices=np.flatnonzero(seed)
                    progress.annotate_frame(geometry_seed_pixels=len(indices),
                        geometry_seed_source='matched_current_rgbd_component')
                    if len(indices):
                        kwargs['target_seed_pixel_indices']=indices
                        detection=detect_seed_local(frame,d2c,detector=detect_target,**kwargs)
                    else:
                        detection=detect_target(frame,d2c,**kwargs)
                else:
                    detection=detect_target(frame,d2c,**kwargs)
        except TargetNotReady as error:
            self._record_geometry_failure(error,frame,d2c,state,
                (peer.clock_epoch,peer.track_generation,track.track_id),surface,indices)
            return
        except PickRejected as error:
            if error.details.get('candidate_count')==0:
                self._record_geometry_failure(error,frame,d2c,state,
                    (peer.clock_epoch,peer.track_generation,track.track_id),surface,indices)
                return
            raise
        with self.lock:
            if self.stop.is_set() or generation!=self.geometry_generation:return
        progress.annotate_frame(component_extraction=detection.get('component_extraction'))
        if self.fast_stereo is not None:
            try:
                with progress.measure('fast_stereo_reconstruction'):
                    detection=self._refine_stereo(detection,frame,d2c,
                        robot_model=self.geometry_model,robot_state=state,surface=surface)
                progress.annotate_frame(fast_stereo=detection['fast_stereo'])
            except TargetNotReady as error:
                self._record_geometry_failure(error,frame,d2c,state,
                    (peer.clock_epoch,peer.track_generation,track.track_id),surface,indices)
                return
        center=np.median(detection['object_points'],axis=0)
        with self.lock:
            repeated = (self.lookahead_enabled or self.single_frame_backup is not None) and self._repeats_locked(
                center, captured, frame_sequence=frame.header.sequence, mask=detection.get('mask'))
        if repeated:
            progress.block('waiting_new_target', 'same_physical_workpiece')
            return
        observed=np.asarray(track.observed_position_base,dtype=float)
        distance=float(np.linalg.norm(center[:2]-observed[:2]))
        if distance>gate:
            progress.block('waiting_target_association','detection_track_mismatch',
                distance_xy_m=distance,maximum_distance_m=gate)
            return
        progress.enter('generating_obb')
        try:
            from .depth_separation import grasp_points
            from .tracked_freeze import check_near_camera_footprint
            check_near_camera_footprint(self.config, surface, frame, detection)
            if self.config.raw.get('depth_separation_enabled', False):
                with progress.measure('depth_separation'):
                    obb_points = grasp_points(detection, frame, enabled=True)
            else:
                obb_points = detection['object_points']
            with progress.measure('obb_grasp'):
                grasps,box=obb_grasps(obb_points,detection['belt_region']['plane_to_base'],
                                    self.config.canonical_tcp,
                                    tcp_z_offset_m=self.config.raw.get('grasp_tcp_z_offset_m', 0.))
        except TargetNotReady as error:
            self._record_geometry_failure(error,frame,d2c,state,
                (peer.clock_epoch,peer.track_generation,track.track_id),surface,indices)
            return
        except VisibleOBBError as error:
            progress.block('waiting_target_geometry', 'obb_invalid_visible_points',
                error=str(error), point_count=len(detection['object_points']))
            return
        velocity=prediction_velocity(self.config,track.velocity_base_m_s,detection['plane']['normal_base'])
        age=time.monotonic()-captured
        if age>self.config.raw['maximum_observation_age_s']:
            progress.block('waiting_fresh_observation','observation_stale_after_detection',
                observation_age_s=age,maximum_age_s=self.config.raw['maximum_observation_age_s'])
            return
        try:self.entry_time(captured,observed,velocity,time.monotonic(),progress=progress,
                            belt_region=detection['belt_region'])
        except EntryNotReady:return
        record_grasp_box(detection, box, grasps, center)
        detection.update(frame=frame,depth_to_color_cv=d2c,scene_state=state)
        item=MeasuredTarget((peer.clock_epoch,peer.track_generation,track.track_id),captured,
            float(frame.header.received_time),float(frame.header.source_time),int(peer.clock_epoch),
            int(frame.header.sequence),center.copy(),velocity,tuple(grasps),
            np.asarray(detection['object_points']).copy(),detection,state,
            make_speed_report(self.config,track,velocity),entry_position=observed)
        done=time.monotonic()
        detection['freeze_delivery']=dict(capture_monotonic_s=captured,
            received_monotonic_s=float(frame.header.received_time),
            geometry_submitted_monotonic_s=submitted_s,geometry_started_monotonic_s=started,
            geometry_done_monotonic_s=done,geometry_wall_s=done-started,
            geometry_queue_s=None if submitted_s is None else started-submitted_s)
        progress.annotate_frame(freeze_delivery=detection['freeze_delivery'],
            local_processing=detection.get('local_processing'))
        progress.mark_ready()
        return generation,peer.track_generation,item

    def _schedule_geometry(self, frame, d2c, peer, captured):
        with self.lock:
            self.progress_snapshot = peer
            self.progress_entry = None
            consumed = set(self.consumed)
        self._finish_geometry()
        with self.lock:
            self._prune_frozen_locked(time.monotonic())
            queued={e.target.identity for e in self.frozen_queue.entries}
            queue_full=self.frozen_queue.full
        if queue_full:
            return None
        if self.tracked_feedback_worker is not None:
            active_match = self.tracked_feedback_worker.active_match()
            if active_match is not None:
                return active_match
        elif self.geometry_future is not None:
            return self.match_in_flight
        valid = [] if peer is None or not peer.current else [t for t in peer.tracks
            if not self._track_rejections(t, peer, consumed)
            and (peer.clock_epoch,peer.track_generation,t.track_id) not in queued]
        multiple_enabled=self.frozen_queue.capacity>1 and self.config.raw.get('tracked_local_geometry',False)
        if not valid or (len(valid)>1 and not multiple_enabled):
            reason = ('meter_not_current' if peer is None or not peer.current else
                'no_tracks' if not peer.tracks else
                'multiple_qualified_tracks' if len(valid) > 1 else 'no_qualified_tracks')
            self.progress.block('waiting_qualified_speed', reason,
                qualified_count=len(valid), track_count=0 if peer is None else len(peer.tracks),
                meter_reason=None if peer is None else peer.reason,
                track_rejections=[] if peer is None else [
                    {'id': t.track_id, 'reasons': self._track_rejections(t, peer, consumed),
                     'samples': t.sample_count, 'span_s': t.span_s,
                     'speed_std_m_s': t.speed_std_m_s,
                     'velocity_base_m_s': t.velocity_base_m_s}
                    for t in peer.tracks])
            return None
        selected=self._select_geometry_track(frame,peer,valid,captured)
        if selected is None:return None
        track,seed_rgb=selected
        identity=(peer.clock_epoch,peer.track_generation,track.track_id)
        with self.lock:
            if identity in self.consumed:return None
            generation=self.geometry_generation
        surface = self.model.measured_belt_surface
        gate = float(self.meter.parameters['tracker_gate_radius'])
        track_gate = (np.asarray(track.observed_position_base), gate)
        d2c = np.array(d2c, copy=True)
        submitted = time.monotonic()

        def prepare(state):
            return self._prepare_target(frame, d2c, peer, track, deepcopy(state), captured, generation,
                                        surface, gate, seed_rgb_mask=seed_rgb, submitted_s=submitted)

        if self.tracked_feedback_worker is None:
            state = self._paired_state(captured)
            if state is None:
                return None
            job = self.geometry_pool.submit(prepare, state)
        else:
            distance = float(self.config.follow_bounds[0]-track_gate[0]@self.config.direction)
            deadline = pairing_deadline(self.config, captured, surface, distance)

            def calculate(state, timing):
                result = prepare(state)
                if result is not None:
                    item = result[2]
                    item.detection['freeze_delivery'].update(timing)
                    with self.lock:
                        publication = self._accept_geometry_locked(*result)
                    self._notify_frozen(item, publication)
            job = self.tracked_feedback_worker.submit(captured, frame.header.clock_epoch, generation,
                                                      deadline, calculate, identity=identity, track_gate=track_gate)
            if job is None:
                return self.tracked_feedback_worker.active_match() or self.match_in_flight
        self.geometry_attempts.pop(identity,None)
        self.geometry_attempts[identity]=frame.header.sequence
        if len(self.geometry_attempts)>64:
            self.geometry_attempts.pop(next(iter(self.geometry_attempts)))
        match = (captured, *track_gate, job)
        with self.lock:
            self.geometry_future = job
            self.match_in_flight = match
        job.add_done_callback(self._geometry_completed)
        return match

    def _run(self):
        decision = None
        sequence = None
        try:
            d2c = self.camera.get_depth_to_color_cv()
            while not self.stop.is_set():
                self._finish_geometry()
                self.progress.enter('waiting_rgbd')
                with self.progress.measure('camera_wait'):
                    reader=getattr(self.camera,'read_next',None) or self.camera.read_latest
                    options={}
                    if self.config.raw.get('tracked_frame_max_backlog_s') is not None:
                        options['max_backlog_s']=self.config.raw['tracked_frame_max_backlog_s']
                    frame = reader(timeout_s=.5, after_sequence=sequence, wait_for_fresh=True,**options)
                previous=self._previous_camera_frame
                self.camera_consumption['frames']+=1
                if previous is not None and previous.header.clock_epoch==frame.header.clock_epoch:
                    self.camera_consumption['capture_sequence_skips']+=max(0,frame.header.sequence-previous.header.sequence-1)
                    self.camera_consumption['max_device_gap_s']=max(self.camera_consumption['max_device_gap_s'],
                        frame.header.source_time-previous.header.source_time)
                self._previous_camera_frame=frame
                sequence = frame.header.sequence
                sink = getattr(self, 'frame_sink', None)
                if sink is not None:
                    sink(frame, d2c)
                with self.lock:
                    if self.clock_epoch is not None and self.clock_epoch!=frame.header.clock_epoch:
                        self.geometry_generation+=1
                        self.latest=None
                    self.clock_epoch=frame.header.clock_epoch
                self.progress.frame(sequence=int(sequence), clock_epoch=int(self.clock_epoch),
                    source_device_s=float(frame.header.source_time),
                    received_monotonic_s=float(frame.header.received_time),
                    receive_to_processing_s=max(0.,time.monotonic()-frame.header.received_time))
                with self.lock:
                    active_task = self.active_task
                if active_task is not None and not self.lookahead_enabled:
                    # This strategy uses the accepted exposure and frozen
                    # velocity. Keep the single camera receiving and epoch
                    # cancellation live, without competing with planning or
                    # the 100 Hz sender for unused perception calculations.
                    self.progress.enter('task_active_camera_only')
                    continue
                self.progress.enter('applying_calibration')
                with self.progress.measure('calibration'):
                    frame = apply_runtime_depth_calibration(frame, self.calibration, d2c)
                self.progress.enter('validating_capture_time')
                try:
                    timing = depth_capture_timing(frame)
                except PickRejected as error:
                    # Discard this exposure and all outstanding results based
                    # on it. A later valid exposure starts a fresh track; no
                    # timestamp is guessed and no rejected frame is consumed.
                    with self.lock:
                        self.geometry_generation+=1
                        self.latest=None
                        self.progress_snapshot=self.progress_entry=None
                        self.reset_tracking=True
                    self.progress.block('waiting_valid_timestamp','invalid_capture_time',
                        error=str(error),timestamp_details=error.details)
                    continue
                captured = timing['depth_sample_monotonic_estimate']
                self.progress.annotate_frame(capture_monotonic_s=captured, capture_timing=timing)
                with self.lock:
                    minimum_capture_s = self.minimum_capture_s
                    reset_tracking = self.reset_tracking
                    if captured >= minimum_capture_s:
                        self.reset_tracking = False
                if captured < minimum_capture_s:
                    self.progress.enter('waiting_new_exposure')
                    continue
                if reset_tracking and self.meter is not None:
                    self.meter.reset()
                surface=getattr(self.model,'measured_belt_surface',None)
                if (surface is None or (not self.native_band_only and self.meter is None)
                        or (isinstance(surface,BeltSurface) and not surface.matches_frame(frame,d2c))):
                    with self.lock:
                        self.geometry_generation+=1
                        self.latest=None
                    self.progress.enter('fitting_belt_surface')
                    try:
                        with self.progress.measure('belt_surface_fit'):
                            surface = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
                    except BeltSurfaceNotReady as error:
                        self.progress.block('fitting_belt_surface', 'belt_surface_not_ready',
                            error=str(error), belt_surface_reason=error.reason)
                        continue
                    self.model.measured_belt_surface = surface
                    if not self.native_band_only:
                        from .perception import make_meter
                        self.meter=make_meter(self.config,surface,previous=self.meter)
                decision = None
                if self.single_frame_backup is not None:
                    decision = self.single_frame_backup.offer(frame, d2c, surface, captured)
                if self.native_band_only:
                    self.progress.enter('single_frame_depth_location')
                    continue
                self.progress.enter('tracking_and_speed')
                with self.progress.measure('tracking_and_speed'):
                    speed_report = self.meter.update(frame, d2c)
                if speed_report is not None:
                    diagnostics=speed_report.get('diagnostics',{})
                    self.perception_diagnostics={k:diagnostics[k] for k in (
                        'detection_mode','plane_calibration','processing_ms','registration_ms',
                        'detection_ms','tracking_ms','tracking_roi_source','tracking_roi_px',
                        'detector_workspace_y_m',
                        'tracking_limit_coordinate_m','belt_surface_capture_sequence',
                        'native_roi_candidate_points','registered_valid_pixels','candidate_count',
                        'depth_admission','candidate_depth_diagnostics',
                        'association_prediction_source','association_velocity_base_m_s') if k in diagnostics}
                peer = self.meter.latest_snapshot
                speed_sink = getattr(self,'speed_sink',None)
                if speed_sink is not None:
                    display_report = speed_report
                    if is_fixed(self.config):
                        display_report = {**(speed_report or {}),
                            'control_velocity_source':FIXED,
                            'fixed_control_speed_m_s':self.config.raw['fixed_belt_speed_m_s']}
                    speed_sink(frame,d2c,peer,display_report,capture_s=captured)
                match = self._schedule_geometry(frame, d2c, peer, captured)
                if decision is not None:
                    decision.set_result(match)
        except BaseException as error:
            if not self.stop.is_set():
                self.progress.block('vision_error', 'vision_exception',
                    error_type=type(error).__name__, error=str(error),
                    details=getattr(error,'details',{}),
                    failed_stage=self.progress.snapshot()['stage'])
                with self.lock:
                    self.error = error
        finally:
            self.stop.set()
            if decision is not None and not decision.done():
                decision.set_result(None)
            if self.single_frame_backup is not None:
                self.single_frame_backup.close()
            if self.tracked_feedback_worker is not None:
                self.tracked_feedback_worker.close()
            self.geometry_pool.shutdown(wait=True,cancel_futures=True)
            self.geometry_progress.finish()
            self.progress.finish()

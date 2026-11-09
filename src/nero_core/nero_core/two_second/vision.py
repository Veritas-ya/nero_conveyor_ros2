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
from .grasp import VisibleOBBError, visible_belt_aligned_obb
from .measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
from .reference import predict_entry_time, EntryNotReady
from .synchronized_reference import admit_synchronized_observation, freeze_window_for_speed
from .diagnostics import VisionProgress
from .velocity_source import KALMAN, POINT_KALMAN, FIXED, is_fixed, prediction_velocity, speed_report as make_speed_report, policy as velocity_policy


def obb_grasps(points_base, plane_to_base, canonical_tcp):
    """Same visible rectangle as simulation, in the MEASURED belt frame.

    TCP origins are the visible box centre. Tool axes use the supplied physical
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
    return result, box


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
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 21 or not np.isfinite(points).all():
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
        self.latest = None
        self.error = None
        self.progress = VisionProgress()
        self.geometry_progress=VisionProgress()
        self.geometry_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='measured-grasp-geometry')
        self.geometry_future=None
        self.geometry_generation=0
        self.geometry_model=None
        self.progress_snapshot = None
        self.perception_diagnostics={}
        self.progress_entry = None
        self.clock_epoch = None
        self.consumed = set()
        self.freeze_position_missed = set()
        self.active_task = None
        self.minimum_capture_s = float('-inf')
        self.reset_tracking = False
        self.thread = Thread(target=self._run, name='measured-two-second-vision')
        self.meter = None
        self.camera_consumption={'frames':0,'capture_sequence_skips':0,'max_device_gap_s':0.}
        self._previous_camera_frame=None

    def start(self):
        self.geometry_future=self.geometry_pool.submit(self._warm_geometry_model)
        self.thread.start()
        return self

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
            self.geometry_generation+=1
            self.consumed.add(identity)
            self.latest = None
            self.active_task = identity

    def resume_targets(self):
        """After a completed return, require a fresh exposure and new speed history."""
        with self.lock:
            self.geometry_generation+=1
            self.latest = self.active_task = None
            self.minimum_capture_s = time.monotonic()
            # Only the camera worker may mutate the meter/tracker.
            self.reset_tracking = True
            self.progress_snapshot = self.progress_entry = None

    def diagnostics(self):
        """Small detached progress report; never used to authorize movement."""
        with self.lock:
            peer, entry = self.progress_snapshot, self.progress_entry
            consumed = set(self.consumed)
            active_task = self.active_task
            missed_count = len(self.freeze_position_missed)
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
                'grasp_geometry':self.geometry_progress.snapshot(),
                'geometry_in_flight':self.geometry_future is not None,
                'perception_detection_mode':self.config.raw.get('perception_detection_mode','rgb_depth'),
                'perception':self.perception_diagnostics,
                'velocity_policy':velocity_policy(self.config),
                'active_task': active_task,
                'target_processing_paused': active_task is not None,
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

    def entry_time(self, capture_s, position, velocity, now_s, *, progress=None):
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
                freeze_window = freeze_window_for_speed(self.config, along)
                detail['freeze_window'] = freeze_window
                detail['maximum_upstream_distance_m'] = freeze_window['distance_m']
                detail['minimum_upstream_distance_m'] = freeze_window.get('minimum_distance_m')
                scheduled = admit_synchronized_observation(capture_s, coordinate, along,
                    self.config.follow_bounds, now_s, freeze_window['distance_m'],
                    minimum_upstream_distance_m=freeze_window.get('minimum_distance_m'))
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
        self.geometry_pool.shutdown(wait=True,cancel_futures=True)

    def _finish_geometry(self):
        future=self.geometry_future
        if future is None or not future.done():return
        self.geometry_future=None
        try:result=future.result()
        except BaseException as error:
            self.geometry_progress.block('geometry_error','geometry_exception',
                error_type=type(error).__name__,error=str(error))
            raise
        if result is None:return
        generation,tracking_generation,item=result
        if time.monotonic()-item.capture_s>self.config.raw['maximum_observation_age_s']:
            self.geometry_progress.block('waiting_fresh_observation','observation_stale_at_delivery',
                source_sequence=item.sequence)
            return
        with self.lock:
            peer=self.progress_snapshot
            if (self.stop.is_set() or generation!=self.geometry_generation or
                    self.clock_epoch!=item.epoch or self.active_task is not None or
                    item.identity in self.consumed or item.capture_s<self.minimum_capture_s or
                    peer is None or peer.track_generation!=tracking_generation):
                return
            self.latest=item
        self.progress.mark_ready()

    def _warm_geometry_model(self):
        if self.geometry_model is None:
            self.geometry_progress.enter('preparing_geometry_model')
            with self.geometry_progress.measure('model_startup'):
                from .planning import calibrated_model
                self.geometry_model=calibrated_model(self.config)

    def _prepare_target(self, frame, d2c, peer, track, state, captured, generation, surface, gate):
        """A single bounded job with its own robot model and exposure identity."""
        progress=self.geometry_progress
        progress.frame(sequence=frame.header.sequence,clock_epoch=peer.clock_epoch,
            track_id=track.track_id,capture_monotonic_s=captured)
        if self.stop.is_set():return
        self._warm_geometry_model()
        progress.enter('detecting_target')
        try:
            with progress.measure('target_detection'):
                detection=detect_target(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
                    target_shape='simple',robot_model=self.geometry_model,robot_state=state,
                    measured_surface=surface,maximum_target_height_m=self.config.raw['maximum_target_height_m'],
                    scene_scope=self.config.raw.get('environment_collision_scope', 'observed_scene'))
        except TargetNotReady as error:
            progress.block('waiting_target_geometry','detection_'+error.details['reason'],
                error=str(error),candidate_count=error.details.get('candidate_count'))
            return
        except PickRejected as error:
            if error.details.get('candidate_count')==0:
                progress.block('waiting_target_geometry','detection_no_candidates',error=str(error))
                return
            raise
        with self.lock:
            if self.stop.is_set() or generation!=self.geometry_generation:return
        progress.annotate_frame(component_extraction=detection.get('component_extraction'))
        center=np.median(detection['object_points'],axis=0)
        observed=np.asarray(track.observed_position_base,dtype=float)
        distance=float(np.linalg.norm(center[:2]-observed[:2]))
        if distance>gate:
            progress.block('waiting_target_association','detection_track_mismatch',
                distance_xy_m=distance,maximum_distance_m=gate)
            return
        progress.enter('generating_obb')
        try:
            with progress.measure('obb_grasp'):
                grasps,box=obb_grasps(detection['object_points'],detection['belt_region']['plane_to_base'],
                                    self.config.canonical_tcp)
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
        try:self.entry_time(captured,observed,velocity,time.monotonic(),progress=progress)
        except EntryNotReady:return
        detection.update(frame=frame,depth_to_color_cv=d2c,scene_state=state,obb=box)
        item=MeasuredTarget((peer.clock_epoch,peer.track_generation,track.track_id),captured,
            float(frame.header.received_time),float(frame.header.source_time),int(peer.clock_epoch),
            int(frame.header.sequence),center.copy(),velocity,tuple(grasps),
            np.asarray(detection['object_points']).copy(),detection,state,
            make_speed_report(self.config,track,velocity),entry_position=observed)
        progress.mark_ready()
        return generation,peer.track_generation,item

    def _run(self):
        sequence = None
        try:
            d2c = self.camera.get_depth_to_color_cv()
            while not self.stop.is_set():
                self._finish_geometry()
                self.progress.enter('waiting_rgbd')
                with self.progress.measure('camera_wait'):
                    reader=getattr(self.camera,'read_next',None) or self.camera.read_latest
                    frame = reader(timeout_s=.5, after_sequence=sequence, wait_for_fresh=True)
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
                    received_monotonic_s=float(frame.header.received_time))
                with self.lock:
                    active_task = self.active_task
                if active_task is not None:
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
                if self.meter is None or (isinstance(surface,BeltSurface) and not surface.matches_frame(frame,d2c)):
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
                    from .perception import make_meter
                    self.meter=make_meter(self.config,surface,previous=self.meter)
                self.progress.enter('tracking_and_speed')
                with self.progress.measure('tracking_and_speed'):
                    speed_report = self.meter.update(frame, d2c)
                if speed_report is not None:
                    diagnostics=speed_report.get('diagnostics',{})
                    self.perception_diagnostics={k:diagnostics[k] for k in (
                        'detection_mode','plane_calibration','processing_ms','registration_ms',
                        'detection_ms','tracking_ms','tracking_roi_source','tracking_roi_px',
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
                with self.lock:
                    self.progress_snapshot = peer
                    self.progress_entry = None
                    consumed = set(self.consumed)
                self._finish_geometry()
                if self.geometry_future is not None or self.latest is not None:
                    continue
                valid = [] if peer is None or not peer.current else [t for t in peer.tracks
                    if not self._track_rejections(t, peer, consumed)]
                if len(valid) != 1:
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
                    continue
                track = valid[0]
                identity = (peer.clock_epoch, peer.track_generation, track.track_id)
                # Eye-to-hand speed is independent of arm feedback. Only a
                # geometry job needs this fresh encoder sample for self filtering.
                self.progress.enter('pairing_feedback')
                with self.progress.measure('feedback_pairing'):
                    state=self.read_feedback()
                if state is None:
                    self.progress.block('waiting_feedback','feedback_missing',capture_monotonic_s=captured)
                    continue
                packet_times=[mapped_stamp(state,k) for k in ('joint','pose')]
                self.progress.annotate_frame(joint_feedback_monotonic_s=packet_times[0],
                    pose_feedback_monotonic_s=packet_times[1])
                if any(stamp is None or abs(stamp-captured)>.2 for stamp in packet_times):
                    self.progress.block('waiting_feedback',
                        'feedback_time_missing' if any(t is None for t in packet_times) else 'feedback_frame_skew',
                        capture_monotonic_s=captured,joint_monotonic_s=packet_times[0],
                        pose_monotonic_s=packet_times[1],maximum_skew_s=.2,
                        feedback_minus_capture_s=[None if t is None else t-captured for t in packet_times])
                    continue
                entry_position = np.asarray(track.observed_position_base, dtype=float)
                velocity = prediction_velocity(self.config, track.velocity_base_m_s, self.model.measured_belt_surface.normal_base)
                self.progress.annotate_frame(track_id=track.track_id,
                    track_generation=peer.track_generation, observed_position_base_m=entry_position.tolist(),
                    velocity_base_m_s=velocity.tolist(), samples=track.sample_count, span_s=track.span_s)
                age = time.monotonic()-captured
                if age > self.config.raw['maximum_observation_age_s']:
                    self.progress.block('waiting_fresh_observation', 'observation_stale_before_detection',
                        observation_age_s=age, maximum_age_s=self.config.raw['maximum_observation_age_s'])
                    continue
                try:
                    self.entry_time(captured, entry_position, velocity, time.monotonic())
                except EntryNotReady as error:
                    if error.reason == 'past_fixed_freeze_position':
                        with self.lock:
                            self.freeze_position_missed.add(identity)
                            self.consumed.add(identity)
                    continue  # Upstream observations keep tracking; missed IDs stay skipped.
                with self.lock:
                    if identity in self.consumed:
                        self.progress.block('waiting_new_target', 'already_consumed')
                        continue
                with self.lock:
                    generation=self.geometry_generation
                self.geometry_future=self.geometry_pool.submit(self._prepare_target,frame,
                    np.array(d2c,copy=True),peer,track,deepcopy(state),captured,generation,
                    self.model.measured_belt_surface,float(self.meter.parameters['tracker_gate_radius']))
        except BaseException as error:
            if not self.stop.is_set():
                self.progress.block('vision_error', 'vision_exception',
                    error_type=type(error).__name__, error=str(error),
                    details=getattr(error,'details',{}),
                    failed_stage=self.progress.snapshot()['stage'])
                with self.lock:
                    self.error = error
        finally:
            self.geometry_pool.shutdown(wait=True,cancel_futures=True)
            self.geometry_progress.finish()
            self.progress.finish()

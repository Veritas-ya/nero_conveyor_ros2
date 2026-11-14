"""Periodic measured-to-path geometry monitoring outside the device sender.

The sender already owns a complete, offline-checked continuous curve. It
interpolates that curve at 100 Hz; this process only monitors measured geometry
at a lower rate. A recent monitor result is NOT a proof of every intervening
feedback state. Expiry, failure, scene changes and cancellation remain gates.
No device, SDK, camera, logger or executor is passed to the child process.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


@dataclass(frozen=True)
class SweepSettings:
    mode: str = 'inline'
    hz: float = 20.
    max_age_s: float = .2

    @classmethod
    def from_raw(cls, raw, control_hz=100.):
        mode = raw.get('runtime_sweep_mode', 'inline')
        if mode not in ('inline', 'process_periodic'):
            raise ValueError('runtime_sweep_mode 必须为 inline 或 process_periodic')
        values = []
        for key, default in (('runtime_sweep_hz', 20.), ('runtime_sweep_max_age_s', .2)):
            value = raw.get(key, default)
            if isinstance(value, (bool, str)) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f'{key} 须为有限数值')
            values.append(float(value))
        hz, age = values
        if not 1. <= hz <= control_hz:
            raise ValueError('runtime_sweep_hz 须在1 Hz和发送频率之间')
        if not 2./hz <= age <= .2:
            raise ValueError('runtime_sweep_max_age_s 须至少覆盖两次监测周期，且不超过0.2秒')
        return cls(mode, hz, age)


def geometry_config(config):
    """Only values needed by the kinematic mirror, never taught/camera objects."""
    keys = ('joint_position_margins_rad', 'workpiece_collision_margin_m',
            'grasp_workpiece_collision_check_enabled', 'payload_belt_clearance_m',
            'environment_collision_scope', 'joint_names', 'base_frame',
            'experimental_direct_drop_guard')
    return SimpleNamespace(raw={k: config.raw[k] for k in keys if k in config.raw},
        settings=SimpleNamespace(open_width_m=config.settings.open_width_m),
        workspace=np.array(config.workspace), standby=np.array(config.standby), tcp=np.array(config.tcp),
        source_sha256=getattr(config, 'source_sha256', None))


def scene_snapshot(planner, path):
    """Copy already filtered geometry; do not re-segment a frozen exposure."""
    from .workpiece_contact import recovery_contact_allowed
    target = planner.target
    return dict(token=uuid4().hex, path=path, identity=planner.identity,
        scene_revision=planner.scene_revision,
        scene=np.array(planner.model.scene), model_target=np.array(planner.model.target),
        command_lower=planner.model.command_lower.copy(), command_upper=planner.model.command_upper.copy(),
        gripper_scene_margin_m=planner.model.gripper_scene_margin_m,
        plane=planner.plane.copy(), belt_bounds=planner.belt_bounds.copy(),
        scene_report=planner.scene_report,
        points=np.array(target.points), velocity=np.array(target.velocity), capture_s=target.capture_s,
        target_identity=getattr(target,'identity',planner.identity),
        target_epoch=getattr(target,'epoch',planner.identity[0]),
        target_sequence=getattr(target,'sequence',0),
        target_depth_to_color_cv=np.array(target.detection.get('depth_to_color_cv',np.eye(4))),
        target_state={k:v for k,v in getattr(target,'state',{}).items()
                      if k in ('joints','pose','width_m','force_n')},
        geometry_context=dict(planner.geometry_context),
        released_points=None if planner.released_points is None else planner.released_points.copy(),
        recovery_workpiece_contact=recovery_contact_allowed(planner))


_planner = _path = _token = _cancel = None


def _initialize(config, cancel):
    global _planner, _cancel
    from .planning import MeasuredPlanner
    _planner, _cancel = MeasuredPlanner(config), cancel


def _ping():
    return {'pid': os.getpid(), 'hardware_connections': 0}


def _install(snapshot):
    global _path, _token
    restore_scene(_planner, snapshot)
    _path, _token = snapshot['path'], snapshot['token']


def restore_scene(p, snapshot):
    """Restore immutable geometry into a private model, without device state."""
    p.model.scene = snapshot['scene']
    p.model.tree = cKDTree(p.model.scene)
    p.model.target = snapshot['model_target']
    p.model.command_lower = snapshot['command_lower']
    p.model.command_upper = snapshot['command_upper']
    p.model.gripper_scene_margin_m = snapshot['gripper_scene_margin_m']
    p.identity, p.scene_revision = snapshot['identity'], snapshot['scene_revision']
    p.target = SimpleNamespace(points=snapshot['points'], velocity=snapshot['velocity'], capture_s=snapshot['capture_s'],
        identity=snapshot['target_identity'], epoch=snapshot['target_epoch'], sequence=snapshot['target_sequence'],
        state=snapshot['target_state'], detection={'depth_to_color_cv':snapshot['target_depth_to_color_cv']})
    p.target_tree = cKDTree(p.target.points)
    p.plane, p.belt_bounds = snapshot['plane'], snapshot['belt_bounds']
    p.support, p.support_origin = p.plane[:3, 2], p.plane[:3, 3]
    p.scene_report = snapshot['scene_report']
    p.released_points = snapshot['released_points']
    p.released_tree = None if p.released_points is None else cKDTree(p.released_points)
    p._recovery_workpiece_contact_target = p.target if snapshot['recovery_workpiece_contact'] else None
    p.version = snapshot['path'].version
    p.geometry_context = snapshot['geometry_context']
    # A child cannot emit parent's visualization/logging callbacks.
    p.collision_sink = None


def sweep(planner, path, request, *, cancelled=lambda: False):
    """Same three time cuts / .008 rad chords, run at the monitoring rate."""
    checked = set()
    actual = request['joints']
    for stamp in np.linspace(request['dispatch_s'], request['until_s'], 3):
        target = path.position(stamp+request['advance_s']-request['origin_s'])
        count = max(1, int(np.ceil(np.max(abs(target-actual))/.008)))
        for u in np.linspace(0., 1., count+1):
            if cancelled():
                return False
            q = actual+(target-actual)*u
            target_stamp = (request['pose_s']+(stamp+request['advance_s']-request['pose_s'])*u
                            if request['moving_target'] else None)
            key = (q.tobytes(), target_stamp)
            if key in checked:
                continue
            planner.valid(q, request['width_m'], stamp=target_stamp,
                contact=request['contact'], payload=request['payload'],
                allow_belt_support=request['allow_belt_support'])
            checked.add(key)
    return True


def _job(snapshot, request):
    started = time.monotonic()
    collisions = []
    try:
        if snapshot is not None:
            _install(snapshot)
        if request['token'] != _token:
            raise PickRejected('后台碰撞监测任务代次已改变')
        _planner.collision_sink = collisions.append
        if request['payload']:
            if request['payload_local'] is None or request['capture_tcp'] is None:
                raise PickRejected('后台碰撞监测缺少当前载物几何')
            _planner.payload_local = request['payload_local']
            _planner.payload_tree = cKDTree(_planner.payload_local)
            _planner.capture_tcp = request['capture_tcp']
        ok = sweep(_planner, _path, request, cancelled=_cancel.is_set)
        return dict(token=_token, ok=ok, cancelled=not ok, source_s=request['source_s'],
            started_s=started, completed_s=time.monotonic(), pid=os.getpid())
    except Exception as error:
        return dict(token=request['token'], ok=False, cancelled=False, source_s=request['source_s'],
            started_s=started, completed_s=time.monotonic(), pid=os.getpid(),
            error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', {}),
            collision_packets=collisions)


class SweepService:
    """One warmed geometry process shared across this run's motion stages."""
    def __init__(self, config):
        context = mp.get_context('spawn')  # Never fork live CAN/camera threads.
        self.cancel = context.Event()
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=context,
            initializer=_initialize, initargs=(geometry_config(config), self.cancel))
        try:
            self.worker = self.pool.submit(_ping).result(timeout=15.)
        except BaseException:
            self.close()
            raise

    def submit(self, snapshot, request):
        return self.pool.submit(_job, snapshot, request)

    def close(self):
        self.cancel.set()
        self.pool.shutdown(wait=True, cancel_futures=True)


class PeriodicSweep:
    """The sender only publishes a small request and reads an atomic result.

    pump()/prime()/close() belong to the supervising thread. In particular,
    serialization and ProcessPool.submit never execute in the sender callback.
    One pending job and one latest request bound the work and memory backlog.
    """
    def __init__(self, service, snapshot, settings, *, clock=time.monotonic,
                 collision_sink=None, collision_frame=None, feedback_reader=None):
        self.service, self.snapshot, self.settings, self.clock = service, snapshot, settings, clock
        self.token = snapshot['token']
        self.latest_request = self.result = self.fault = self.future = None
        self.next_due = -np.inf
        self.closed = False
        self.collision_sink, self.collision_frame = collision_sink, collision_frame
        # The supervising thread reads geometry feedback independently of the
        # sender. In particular, an absolute future send reservation must not
        # turn its one initial snapshot into the only monitor input.
        self.feedback_reader = feedback_reader
        self.last_submitted_source_s = -np.inf
        self.feedback_refreshes = self.duplicate_requests_skipped = 0
        self.diagnostic_errors = []
        self.submitted = self.completed = 0
        self.max_result_age_s = self.max_job_wall_s = 0.
        self.max_round_trip_s = 0.
        self.pending_since = None
        self.completion_times = []

    def offer(self, request):
        if not self.closed:
            self.latest_request = {**request, 'token': self.token}

    def prime(self, request):
        self.service.cancel.clear()
        self.offer(request)
        self.pending_since = self.clock()
        self.future = self.service.submit(self.snapshot, self.latest_request)
        self.last_submitted_source_s = self.latest_request['source_s']
        self.submitted += 1
        self.next_due = self.clock()+1./self.settings.hz
        # Motion has not started. Require one actual measured geometry check.
        self._accept(self.future.result(timeout=5.))
        self.future = None
        self.guard(self.clock())

    def _accept(self, result):
        if self.closed or result['token'] != self.token:
            return
        if self.pending_since is not None:
            self.max_round_trip_s = max(self.max_round_trip_s, self.clock()-self.pending_since)
        if not result['ok']:
            for packet in result.get('collision_packets', ()):
                if self.collision_sink is not None:
                    packet['frame'] = self.collision_frame
                    try:
                        self.collision_sink(packet)
                    except Exception as error:
                        self.diagnostic_errors.append(f'{type(error).__name__}: {error}')
            self.fault = PickRejected(result.get('error', '后台碰撞监测已取消'), details={
                **result.get('details', {}), 'runtime_sweep': True,
                'worker_error_type': result.get('error_type'), 'source_feedback_s': result['source_s']})
        else:
            self.result = result
            self.completed += 1
            self.completion_times.append(self.clock())
            self.max_job_wall_s = max(self.max_job_wall_s, result['completed_s']-result['started_s'])

    def pump(self):
        if self.closed:
            return
        if self.future is not None and self.future.done():
            try:
                self._accept(self.future.result())
            except Exception as error:
                self.fault = PickRejected('后台碰撞监测进程失效', details={
                    'runtime_sweep': True, 'error': str(error), 'error_type': type(error).__name__})
            self.future = None
        now = self.clock()
        if self.fault is None and self.future is None and now >= self.next_due and self.latest_request is not None:
            self.next_due = now+1./self.settings.hz
            state = None if self.feedback_reader is None else self.feedback_reader()
            # Read the template AFTER acquiring feedback: the sender may have
            # published a complete payload attachment while that read ran.
            # Only the sender changes these phase/attachment flags. Never
            # overwrite its newer publication with a supervisor's older copy.
            request = self.latest_request
            if state is not None:
                dispatch = max(self.clock(), request['origin_s']-request['advance_s'])
                request = refresh_request(request, state, dispatch)
            if request['source_s'] <= self.last_submitted_source_s:
                self.duplicate_requests_skipped += 1
                return
            if state is not None:
                self.feedback_refreshes += 1
            # Latest actual feedback wins; do not recheck one old packet over
            # and over, or count completed old jobs as renewed freshness.
            self.pending_since = now
            self.future = self.service.submit(None, request)
            self.last_submitted_source_s = request['source_s']
            self.submitted += 1

    def guard(self, now):
        if self.closed:
            raise PickRejected('后台碰撞监测已关闭')
        if self.fault is not None:
            raise self.fault
        result = self.result
        age = np.inf if result is None else now-result['source_s']
        if not 0 <= age <= self.settings.max_age_s:
            raise PickRejected('后台碰撞监测结果过期，取消当前轨迹', details={
                'runtime_sweep': True, 'feedback_age_s': float(age), 'maximum_age_s': self.settings.max_age_s,
                'source_feedback_s': None if result is None else result['source_s'],
                'guard_monotonic_s': float(now), 'submitted': self.submitted, 'completed': self.completed,
                'pending': self.future is not None, 'feedback_refreshes': self.feedback_refreshes})
        self.max_result_age_s = max(self.max_result_age_s, age)
        return result['source_s']+self.settings.max_age_s

    def close(self):
        if self.closed:
            return
        self.service.cancel.set()
        try:
            if self.future is not None:
                # No sender is active now; join before reusing the model.
                # A rejection already completed must not disappear merely
                # because the sender reached its endpoint at the same time.
                result = self.future.result(timeout=5.)
                if not result.get('cancelled'):
                    self._accept(result)
                self.future = None
        finally:
            self.closed = True
        if self.fault is not None:
            raise self.fault

    def diagnostics(self):
        return dict(mode='process_periodic', check_hz=self.settings.hz,
            maximum_feedback_age_s=self.settings.max_age_s,
            submitted=self.submitted, completed=self.completed,
            independent_feedback_refreshes=self.feedback_refreshes,
            duplicate_requests_skipped=self.duplicate_requests_skipped,
            maximum_observed_result_age_s=self.max_result_age_s,
            maximum_job_wall_s=self.max_job_wall_s,
            maximum_monitor_round_trip_s=self.max_round_trip_s,
            measured_completion_hz=((len(self.completion_times)-1)/
                (self.completion_times[-1]-self.completion_times[0]) if len(self.completion_times)>1 else None),
            worker_pid=self.service.worker['pid'],
            diagnostic_errors=list(self.diagnostic_errors),
            spatial_chord_rad=.008, time_cuts_per_check=3,
            interpolation='original_offline_checked_continuous_curve',
            coverage='periodic_measured_geometry_monitor_not_every_intermediate_state')


def request_for(planner, state, *, dispatch, origin, advance, control_hz,
                reference, attached, smooth, scheduled_attempt, follow_end, grip_confirmed=False):
    elapsed = dispatch+advance-origin
    pose_s = mapped_stamp(state, 'pose')
    until = dispatch+1./control_hz
    if reference is not None and not grip_confirmed and elapsed < follow_end:
        until = min(until, origin-advance+follow_end)
    return dict(joints=np.array(state['joints']), width_m=float(state['width_m']),
        dispatch_s=dispatch, until_s=until, pose_s=pose_s,
        source_s=min(pose_s, mapped_stamp(state, 'joint')),
        origin_s=origin, advance_s=advance,
        control_period_s=1./control_hz,
        contact_dispatch_s=None if reference is None else origin-advance+reference.duration_s,
        unconfirmed_follow_end_dispatch_s=(None if reference is None or grip_confirmed
                                           else origin-advance+follow_end),
        moving_target=reference is not None and not attached and not scheduled_attempt,
        contact=reference is not None and elapsed >= reference.duration_s,
        payload=attached, allow_belt_support=smooth and attached,
        payload_local=getattr(planner, 'payload_local', None) if attached else None,
        capture_tcp=getattr(planner, 'capture_tcp', None) if attached else None)


def refresh_request(template, state, dispatch):
    """Refresh packet clocks/positions, retaining a fully published phase.

    The sender may not have started yet, so advance the checking clock to the
    reserved start while retaining the current REAL feedback timestamps.
    Nothing here sends a target or invents feedback or a payload attachment.
    """
    pose_s = mapped_stamp(state, 'pose')
    until = dispatch+template['control_period_s']
    boundary = template['unconfirmed_follow_end_dispatch_s']
    if boundary is not None and dispatch < boundary:
        until = min(until, boundary)
    contact_at = template['contact_dispatch_s']
    return {**template, 'joints': np.array(state['joints']), 'width_m': float(state['width_m']),
        'dispatch_s': dispatch, 'until_s': until, 'pose_s': pose_s,
        'source_s': min(pose_s, mapped_stamp(state, 'joint')),
        'contact': contact_at is not None and dispatch >= contact_at}

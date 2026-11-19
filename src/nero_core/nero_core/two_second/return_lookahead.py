"""One pending exposure and a private planner; this worker has no device API."""
from dataclasses import dataclass
from copy import deepcopy
from threading import Event, RLock, Thread
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .clock_recovery import WaitClockConfirmation
from .prepared_cycle import nominal_state
from .lookahead_selection import latest_dispatch, pending_error


@dataclass(frozen=True)
class PreparedIntercept:
    target: object
    planner: object
    plan: object
    start_state: dict
    return_owner: tuple
    ready_s: float
    submitted_s: float | None = None

    def validate(self, config, epoch, now):
        plan, target, planner = self.plan, self.target, self.planner
        deadline = plan.reference.start_s-config.raw['command_advance_s']
        if (epoch != target.epoch or plan.target is not target
                or plan.path.identity != target.identity or planner.identity != target.identity
                or plan.path.version != planner.version or plan.scene_revision != planner.scene_revision
                or plan.reference.capture_s != target.capture_s):
            raise PickRejected('预取轨迹的目标、曝光、版本或场景已失效')
        if not np.isfinite(deadline) or now >= deadline:
            raise PickRejected('预取模型路径已生成，但原发送预约已过期', details={
                'reason': 'lookahead_send_deadline_missed', 'model_path_generated': True,
                'send_deadline_monotonic_s': deadline, 'send_margin_s': deadline-now})


class ReturnLookahead:
    """Acquire and plan immediately; only execution waits for confirmed return.

    The active planner is never touched until its owner has completed return.
    A future standby is a model waypoint, never fabricated device feedback.
    No worker callback may acquire a device, change the sender or move the arm.
    """
    def __init__(self, config, vision, spare_planner, *, cancelled, emit,
                 clock=time.monotonic, offer_target=None, offer_plan=None):
        self.config, self.vision, self.planner = config, vision, spare_planner
        self.cancelled, self.emit, self.clock = cancelled, emit, clock
        self.offer_target, self.offer_plan = offer_target, offer_plan
        self.lock = RLock()
        self.stop = Event()
        self.thread = Thread(target=self._run, name='next-pick-private-planner')
        self.owner = None
        self.available_s = None
        self.return_confirmed = False
        self.target = self.ready = self.error = None
        self.status = 'idle'
        self.rejected = 0
        self.started = False
        self.schedule_revision = 0
        self.attempted_revision = None
        self.deferred = 0
        self.wait_state = None
        self.wait_feedback_stamps = {}
        self.wait_clock_confirmation = None
        self.wait_feedback_error = None
        self.planning_start_state = None
        self.deferred_revisions = {}
        self.multiple_pending = config.raw.get('tracked_frozen_queue_capacity', 1) > 1

    def start(self):
        self.vision.enable_lookahead()
        self.thread.start()
        self.started = True
        return self

    def begin_cycle(self, identity):
        with self.lock:
            self.owner = identity
            self.available_s = None
            self.return_confirmed = False
            self.target = self.ready = None
            self.schedule_revision += 1
            self.attempted_revision = None
            self.status = 'waiting_next_exposure'
            self.wait_state = self.planning_start_state = None
            self.wait_feedback_stamps = {}
            self.wait_clock_confirmation = None
            self.wait_feedback_error = None
            self.deferred_revisions.clear()

    def return_started(self, *, identity, expected_ready_s):
        if not np.isfinite(expected_ready_s):
            raise ValueError('invalid predicted return time')
        with self.lock:
            if identity != self.owner:
                return
            if self.available_s != float(expected_ready_s):
                self.schedule_revision += 1
            self.available_s = float(expected_ready_s)
        self.emit('lookahead_return_window', identity=identity,
            expected_ready_monotonic_s=expected_ready_s, readiness_is_prediction=True)

    def _check_wait_state(self, state, *, previous=None):
        from .streaming import fresh_feedback
        from .execution import validate_start_state
        if state is None:
            raise PickRejected(self.wait_feedback_error or '等待位实际反馈尚未提供')
        fresh_feedback(state, self.clock(), previous)
        validate_start_state(self.config, state, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)

    @staticmethod
    def _start_changed(state, start):
        return (np.max(abs(np.asarray(state['joints'])-np.asarray(start['joints']))) > .005
                or abs(state['width_m']-start['width_m']) > .002)

    def _accept_wait_snapshot(self, state):
        # Publish a new baseline only after time, joints and jaws all pass.
        if self.wait_clock_confirmation is None:
            stamps = self.wait_feedback_stamps.copy()
            try:
                self._check_wait_state(state, previous=stamps)
            except PickRejected as error:
                if (not self.return_confirmed or
                        error.details.get('reason') != 'feedback_clock_epoch_changed'):
                    raise
                self.wait_clock_confirmation = WaitClockConfirmation()
                self.ready = self.planning_start_state = None
                self.schedule_revision += 1  # Invalidate any worker still solving.
            else:
                if not self.wait_feedback_stamps and state.get('feedback_time'):
                    epochs = {v.get('clock_epoch') for v in state['feedback_time'].values()
                              if isinstance(v, dict) and 'clock_epoch' in v}
                    if len(epochs) > 1:
                        raise PickRejected('等待首次反馈的时钟代次一致')
                self.wait_feedback_stamps = stamps
                return True

        stamps = {}
        self._check_wait_state(state, previous=stamps)
        evidence = self.wait_clock_confirmation.observe(self.wait_feedback_stamps, state, self.clock())
        if evidence is None:
            return False
        self.wait_feedback_stamps = stamps
        self.wait_clock_confirmation = None
        self.emit('lookahead_wait_clock_rebased', identity=self.owner, **evidence, robot_commands=0)
        return True

    def update_wait_state(self, state):
        """Mailbox from the cycle owner; no device access on this worker."""
        snapshot = deepcopy(state)
        with self.lock:
            if self.owner is None:
                return False
            try:
                if not self._accept_wait_snapshot(snapshot):
                    self.wait_state = None
                    self.wait_feedback_error = '正在确认校时后连续的等待位反馈'
                    return False
            except (PickRejected, KeyError, TypeError, ValueError) as error:
                if self.wait_clock_confirmation is not None:
                    self.wait_clock_confirmation = WaitClockConfirmation()
                if self.wait_state is not None:
                    self.schedule_revision += 1
                self.wait_state = None
                self.wait_feedback_error = str(error)
                return False
            restored = self.wait_state is None
            self.wait_state = snapshot
            self.wait_feedback_error = None
            start = self.ready.start_state if self.ready is not None else self.planning_start_state
            if self.return_confirmed and start is not None and self._start_changed(snapshot, start):
                old_source = 'ready_plan' if self.ready is not None else 'inflight_plan'
                self.ready = None
                self.planning_start_state = None
                self.schedule_revision += 1
                self.status = 'actual_wait_changed_replanning'
                self.emit('lookahead_wait_state_replan', identity=self.owner,
                    source=old_source, reason='measured_wait_differs_from_planned_start',
                    actual_joints_rad=snapshot['joints'], planned_joints_rad=start['joints'],
                    joint_error_rad=(np.asarray(snapshot['joints'])-np.asarray(start['joints'])).tolist(),
                    robot_commands=0)
            elif restored and self.return_confirmed:
                self.schedule_revision += 1
            return True

    def confirm_return(self, state):
        with self.lock:
            self.return_confirmed = True
            self.available_s = max(self.available_s or float('-inf'),
                self.clock()+self.config.raw['lookahead_return_margin_s'])
            self.schedule_revision += 1
            self.update_wait_state(state)

    def diagnostics(self):
        with self.lock:
            return {'enabled': True, 'stage': self.status, 'return_owner': self.owner,
                'return_confirmed': self.return_confirmed,
                'predicted_ready_monotonic_s': self.available_s,
                'target_identity': None if self.target is None else self.target.identity,
                'model_path_generated': self.ready is not None, 'rejected_targets': self.rejected,
                'planning_requires_return_schedule': False,
                'schedule_revision': self.schedule_revision,
                'attempted_schedule_revision': self.attempted_revision,
                'deferred_planning_attempts': self.deferred,
                'deferred_target_count': len(self.deferred_revisions),
                'wait_feedback_available': self.wait_state is not None,
                'wait_feedback_error': self.wait_feedback_error,
                'error': None if self.error is None else str(self.error)}

    def _reject(self, target, error, *, generated=False):
        with self.lock:
            if self.target is not target and (self.ready is None or self.ready.target is not target):
                return  # A delayed worker must not clear the next owner's cache.
            self.vision.discard_pending(target.identity)
            self.target = self.ready = None
            self.planning_start_state = None
            self.attempted_revision = None
            self.rejected += 1
            self.status = 'waiting_next_exposure'
        self.emit('lookahead_rejected', identity=target.identity, frame_sequence=target.sequence,
            capture_monotonic_s=target.capture_s, reason=str(error),
            failure_details=getattr(error, 'details', {}), model_path_generated=generated,
            robot_commands=0)

    def poll_ready(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('下一件预取失败：'+str(self.error)) from self.error
            item = self.ready if self.return_confirmed else None
            if item is not None:
                try:
                    self._check_wait_state(self.wait_state)
                except PickRejected as error:
                    self.wait_feedback_error = str(error)
                    return None
        if item is None:
            return None
        try:
            error = pending_error(self.vision, item.target, self.config, self.clock())
            if error is not None:
                raise error
            item.validate(self.config, self.vision.clock_epoch, self.clock())
        except PickRejected as error:
            self.discard_ready(item, error)
            return None
        return item

    def claim(self, item, previous_planner):
        """Called only after confirmed return; exchange exclusive planner owners."""
        with self.lock:
            if not self.return_confirmed or item is not self.ready:
                raise PickRejected('未经回位确认，不能接管预取轨迹')
            if previous_planner is item.planner:
                raise PickRejected('预取规划器与当前执行器未隔离')
            self._check_wait_state(self.wait_state)
            if self._start_changed(self.wait_state, item.start_state):
                raise PickRejected('预取轨迹起点与等待位最新反馈不一致')
            error = pending_error(self.vision, item.target, self.config, self.clock())
            if error is not None:
                raise error
            item.validate(self.config, self.vision.clock_epoch, self.clock())
            self.planner = previous_planner
            self.planner.intercept_not_before_s = None
            self.ready = self.target = None
            self.available_s = self.owner = None
            self.wait_state = self.planning_start_state = None
            return item

    def discard_ready(self, item, error):
        if getattr(error, 'details', {}).get('reason') != 'lookahead_send_deadline_missed':
            with self.lock:
                if self.ready is item:
                    self._reject(item.target, error, generated=True)
            return
        self._reschedule(item, error)

    def _reschedule(self, item, error):
        # The exposure remains unchanged. A new prediction or confirmed return
        # permits another full solve; passage of time alone must not spin on
        # speculative planning during a delayed/failed current cycle.
        with self.lock:
            if self.ready is not item or self.target is not item.target:
                return
            self.ready = None
            if self.return_confirmed:
                self.available_s = max(self.available_s or float('-inf'),
                    self.clock()+self.config.raw['lookahead_return_margin_s'])
                if self.attempted_revision == self.schedule_revision:
                    self.schedule_revision += 1
            self.status = 'old_schedule_waiting_replan'
            if self.multiple_pending and not self.return_confirmed:
                self.deferred_revisions[item.target.identity] = self.schedule_revision
                self.target = self.planning_start_state = None
                self.attempted_revision = None
        self.emit('lookahead_plan_expired_replan', identity=item.target.identity,
            frame_sequence=item.target.sequence, capture_monotonic_s=item.target.capture_s,
            failure_details=error.details, model_path_generated=True,
            return_confirmed=self.return_confirmed, robot_commands=0)

    def _select_pending(self, revision):
        if not self.multiple_pending:
            return self.vision.poll_pending()
        candidates = self.vision.pending_candidates()
        identities = {t.identity for t in candidates}
        self.deferred_revisions = {k:v for k,v in self.deferred_revisions.items() if k in identities}
        eligible = [t for t in candidates if self.deferred_revisions.get(t.identity) != revision]
        for target in sorted(eligible, key=lambda t: (latest_dispatch(t, self.config), t.capture_s)):
            if self.vision.reserve_pending(target):
                return target
        return None

    def _step(self):
        with self.lock:
            ready = self.ready
            if self.owner is None:
                return
            target, owner, available = self.target, self.owner, self.available_s
            revision, confirmed = self.schedule_revision, self.return_confirmed
            if confirmed:
                try:
                    self._check_wait_state(self.wait_state)
                except PickRejected as error:
                    self.wait_feedback_error = str(error)
                    self.status = 'waiting_fresh_actual_wait_feedback'
                    return
        if ready is not None:
            try:
                error = pending_error(self.vision, ready.target, self.config, self.clock())
                if error is not None:
                    raise error
                ready.validate(self.config, self.vision.clock_epoch, self.clock())
            except PickRejected as error:
                self.discard_ready(ready, error)
            else:
                deadline = ready.plan.reference.start_s-self.config.raw['command_advance_s']
                if not confirmed and available is not None and deadline < available:
                    self._reschedule(ready, PickRejected('回位预测晚于原发送预约，重新规划下一件', details={
                        'reason': 'lookahead_return_prediction_changed',
                        'send_deadline_monotonic_s': deadline,
                        'predicted_ready_monotonic_s': available}))
            return
        if target is None:
            with self.lock:
                if self.owner != owner or self.schedule_revision != revision:
                    return
                target = self._select_pending(revision)
            if target is None:
                return
            with self.lock:
                if self.owner != owner or self.schedule_revision != revision:
                    return
                self.target = target
                self.attempted_revision = None
            # Freshness is checked at reservation. Holding an accepted frozen
            # target never changes its timestamp; later validity uses its
            # certified absolute send deadline, not a new capture-age budget.
            admission=getattr(self.vision,'pending_admission',None)
            accepted=(admission(target,self.clock()) if admission is not None else
                0<=self.clock()-target.capture_s<=self.config.raw['maximum_observation_age_s'])
            if not accepted:
                self._reject(target, PickRejected('下一件曝光在预取接纳前已过期'))
                return
            with self.lock:
                self.status = 'frozen_ready_for_speculative_planning'
            self.emit('lookahead_frozen', identity=target.identity, frame_sequence=target.sequence,
                capture_monotonic_s=target.capture_s, velocity_base_m_s=target.velocity,
                capture_state_source='measured_capture_paired_feedback',
                capture_joint_pairing=target.state.get('capture_joint_pairing'),
                return_confirmed=False, robot_commands=0)
            if self.offer_target is not None:
                self.offer_target(target)
        error = pending_error(self.vision, target, self.config, self.clock())
        if error is not None:
            self._reject(target, error)
            return
        with self.lock:
            if self.owner != owner or self.target is not target:
                return
            revision, confirmed = self.schedule_revision, self.return_confirmed
            available = self.available_s
            if self.attempted_revision == revision:
                return
            if confirmed:
                try:
                    self._check_wait_state(self.wait_state)
                except PickRejected as error:
                    self.wait_feedback_error = str(error)
                    return
                start = deepcopy(self.wait_state)
                source = 'measured_confirmed_wait_feedback'
            else:
                start = None
                source = 'configured_wait_model_waypoint'
            self.attempted_revision = revision
        planner = self.planner
        planner.intercept_not_before_s = max(available or float('-inf'),
            self.clock()+self.config.raw['lookahead_return_margin_s'])
        if start is None:
            start = nominal_state(planner.model, self.config.standby, self.config.settings.open_width_m)
        with self.lock:
            if self.owner != owner or self.schedule_revision != revision:
                return
            self.planning_start_state = deepcopy(start)
            self.status = 'planning_from_actual_wait' if confirmed else 'planning_from_nominal_wait'
        def invalid():
            return (self.stop.is_set() or self.cancelled.is_set()
                or self.vision.clock_epoch != target.epoch or self.owner != owner
                or self.schedule_revision != revision or self.target is not target)
        submitted = self.clock()
        self.emit('lookahead_planning_started', identity=target.identity,
            frame_sequence=target.sequence, capture_monotonic_s=target.capture_s,
            return_confirmed=confirmed, return_prediction_known=available is not None,
            earliest_dispatch_monotonic_s=planner.intercept_not_before_s,
            schedule_revision=revision, start_state_source=source,
            planned_start_joints_rad=start['joints'],
            start_feedback_stamps=self._feedback_stamps(start), robot_commands=0)
        plan = None
        try:
            plan = planner.intercept(target, start, cancelled=invalid)
            item = PreparedIntercept(target, planner, plan, start, owner, self.clock(), submitted)
            item.validate(self.config, self.vision.clock_epoch, self.clock())
            if invalid():
                raise PickRejected('预取规划已取消或所属回程已改变')
        except PickRejected as error:
            if self.stop.is_set() or self.cancelled.is_set() or self.owner != owner:
                return
            expired = pending_error(self.vision, target, self.config, self.clock())
            if (expired is None
                    and (not self.return_confirmed or self.schedule_revision != revision)):
                with self.lock:
                    if self.owner != owner or self.target is not target:
                        return
                    self.deferred += 1
                    self.status = 'model_failed_waiting_schedule_update'
                    if self.multiple_pending:
                        self.deferred_revisions[target.identity] = revision
                        self.target = self.planning_start_state = None
                        self.attempted_revision = None
                self.emit('lookahead_planning_deferred', identity=target.identity,
                    frame_sequence=target.sequence, capture_monotonic_s=target.capture_s,
                    reason=str(error), failure_details=getattr(error, 'details', {}),
                    model_path_generated=plan is not None, schedule_revision=revision,
                    return_confirmed=self.return_confirmed, robot_commands=0)
            else:
                self._reject(target, expired or error, generated=plan is not None)
            return
        with self.lock:
            if invalid() or self.target is not target:
                return
            self.ready = item
            self.status = 'model_ready_waiting_confirmed_return'
        self.emit('lookahead_plan_ready', identity=target.identity, frame_sequence=target.sequence,
            capture_monotonic_s=target.capture_s, ready_monotonic_s=item.ready_s,
            submission_monotonic_s=submitted,
            return_confirmed=self.return_confirmed, return_prediction_known=available is not None,
            schedule_revision=revision,
            planned_start_joints_rad=start['joints'], start_state_source=source,
            start_feedback_stamps=self._feedback_stamps(start),
            send_deadline_monotonic_s=plan.reference.start_s-self.config.raw['command_advance_s'],
            planning_wall_s=plan.path.planning_s, model_path_generated=True, robot_commands=0)
        if self.offer_plan is not None:
            self.offer_plan(plan)

    @staticmethod
    def _feedback_stamps(state):
        from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
        return {kind:mapped_stamp(state, kind) for kind in ('joint','pose','gripper')}

    def _run(self):
        try:
            while not self.stop.is_set() and not self.cancelled.is_set():
                self._step()
                self.stop.wait(.02)
        except Exception as error:
            with self.lock:
                self.error = error
                self.status = 'error'
            self.emit('lookahead_error', error_type=type(error).__name__, reason=str(error))

    def close(self):
        self.stop.set()
        if self.started:
            self.thread.join(timeout=10.)
            if self.thread.is_alive():
                raise RuntimeError('下一件预取规划线程尚未退出')

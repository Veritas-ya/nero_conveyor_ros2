"""Single-owner measured-feedback execution through the existing NERO SDK API."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from dataclasses import replace
import time
import numpy as np
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselinePolicyConfig
from mujoco_conveyor_sorting.runtime.single_pick import PickCycle, PickRejected
from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from .command_history import CommandHistory
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD


def validate_start_state(config, state, *, require_wait=True, tolerance_rad=.005, release_joints=None):
    """Check empty jaws and measured joints against the expected start pose."""
    actual = np.asarray(state['joints'], dtype=float)
    expected = np.asarray(config.standby if release_joints is None else release_joints, dtype=float)
    if actual.shape != (7,) or not np.isfinite(actual).all():
        raise PickRejected('起始关节反馈无效', details={'reason': 'startup_joint_feedback'})
    if release_joints is not None and (expected.shape != (7,) or not np.isfinite(expected).all()
            or np.max(abs(actual-expected)) > tolerance_rad):
        raise PickRejected('起始关节偏离已确认的料框释放位置', details={'reason':'bin_release_joint_mismatch'})
    delta = actual-expected
    if require_wait and np.max(abs(delta)) > tolerance_rad:
        raise PickRejected('机械臂偏离已配置等待位', details={
            'reason': 'standby_joint_mismatch',
            'actual_joints_rad': actual.tolist(), 'standby_joints_rad': expected.tolist(),
            'joint_errors_rad': delta.tolist(),
            'actual_joints_deg': np.rad2deg(actual).tolist(),
            'standby_joints_deg': np.rad2deg(expected).tolist(),
            'joint_errors_deg': np.rad2deg(delta).tolist(),
            'max_abs_joint_error_rad': float(np.max(abs(delta))),
            'tolerance_rad': tolerance_rad, 'tolerance_deg': float(np.rad2deg(tolerance_rad)),
            'failing_joint_numbers': (np.flatnonzero(abs(delta) > tolerance_rad)+1).tolist()})
    if not np.isfinite(state['force_n']) or state['force_n'] >= config.settings.min_force_n:
        raise PickRejected('启动时夹爪可能带载', details={'reason': 'startup_gripper_force',
            'force_n': state['force_n'], 'force_threshold_n': config.settings.min_force_n})
    if not np.isfinite(state['width_m']) or abs(state['width_m']-config.settings.open_width_m) > .002:
        raise PickRejected('起点必须预先张爪；不自动盲开夹爪', details={'reason': 'startup_gripper_width',
            'width_m': state['width_m'], 'required_width_m': config.settings.open_width_m,
            'tolerance_m': .002})


class InterceptUnavailable(PickRejected):
    """Ordinary planning/admission failed before joint trajectory dispatch."""


class TwoSecondExecutor:
    """Own one target and one sender, never enable a robot or use object truth.

    No inference or global planning runs in the high-rate sending callback.
    New visual estimates cannot replace the frozen per-task reference.
    """
    def __init__(self, config, planner, device, *, epoch=lambda: None,
                 cancelled=None, clock=time.monotonic, sleep=time.sleep, event_sink=None,
                 sweep_service=None, carry_service=None):
        self.config, self.planner = config, planner
        self.sweep_service = sweep_service
        self.carry_service = carry_service
        self.carry_prediction = None
        self.epoch, self.clock, self.sleep = epoch, clock, sleep
        self.cancelled = Event() if cancelled is None else cancelled
        self.trace = ExecutionTrace(mode='hardware', clock=clock)
        self.stage = 'waiting'
        self.device = self.trace.wrap_device(device, stage=lambda: self.stage)
        self.raw_device = device
        self.owner = None
        self.version = 0
        self.motion_started = False
        self.sender = None
        self.command_lock = RLock()
        self.command_history = CommandHistory(config)
        self.rest_anchor = None
        self.pending_rest_arrival = None
        self.brake_height_guard = None
        self.height_fault_stop_sent = False
        self.feedback_stamps = {}
        self.events = []
        self.event_sink = event_sink
        self.event_logging_errors = []
        self.return_started = None
        self.prefetched_intercept = None
        self.frozen_entry = None
        self.bin_start = self.next_bin_pick = self.claim_after_release = None
        self.incoming_handoff = self.next_plans = self.pending_targets = None
        self.gates = BaselinePolicyConfig()
        self.holding = PickCycle(self.device, planner.model, config.settings,
                                 self._emit, clock=clock, sleep=sleep,
                                 heartbeat=lambda: self._active(self.owner, self.version, self.owner[0]),
                                 allow_zero_close_window=(config.raw.get('trajectory_profile') == 'smooth'
                                     and config.raw.get('intercept_strategy') == 'synchronized'
                                     and config.raw.get('closure_policy') == 'scheduled_attempt'))

    def _emit(self, name, data):
        # Producers touch memory only; the optional sink enqueues to a writer.
        row = {'event': name, 'monotonic_s': self.clock(), 'data': data}
        self.events.append(row)
        if self.event_sink is not None:
            try:
                self.event_sink(row)
            except Exception as error:
                if len(self.event_logging_errors) < 10:
                    self.event_logging_errors.append(f'{type(error).__name__}: {error}')

    def stop_for_height_fault(self, error):
        # Called by the single sender before another SDK joint command. Do not
        # wait for a background planning future to unwind before stopping.
        self.cancelled.set()
        self._emit('protected_grasp_stopped', {'stage': self.stage, 'error': str(error),
                   'details': error.details})
        self.device.stop()
        self.height_fault_stop_sent = True

    def cancel(self):
        self.cancelled.set()
        self.pending_rest_arrival = None
        if self.carry_service is not None:
            self.carry_service.discard()
        with self.command_lock:
            self.version += 1
        if self.sender is not None:
            self.sender.request_stop()

    def _active(self, identity, version, epoch):
        observed_epoch = self.epoch()
        if self.cancelled.is_set() or identity != self.owner or version != self.version:
            raise PickRejected('任务已取消或轨迹身份/版本过期')
        if observed_epoch is not None and observed_epoch != epoch:
            raise PickRejected('相机时钟已重置；本次冻结观测失效')

    def _prepare(self, future, identity, version, epoch, *, hold=None):
        while not future.done():
            self._active(identity, version, epoch)
            if hold is not None:
                hold.check()
            if hold is None and self.holding.held_width is not None:
                state = self.device.snapshot_nowait()
                if state is not None:
                    self.holding.check_holding(state)
            self.sleep(.01)
        self._active(identity, version, epoch)
        if hold is not None:
            hold.check()
        return future.result()

    def _stream(self, path, *, origin, epoch, reference=None, carrying=False, release_on_arrival=False):
        from .streaming import stream
        if (self.return_started is not None and reference is None and not carrying
                and np.max(abs(path.q1-self.config.standby)) < 1e-8):
            self.return_started(identity=self.owner, expected_ready_s=
                origin+path.duration+self.config.raw['lookahead_return_margin_s'])
        return stream(self, path, origin=origin, epoch=epoch, sender_factory=JointStreamSender,
                      reference=reference, carrying=carrying, release_on_arrival=release_on_arrival)

    def _route(self, pool, state, *, stage, goal, joints, carrying, epoch, version):
        arguments = dict(stage=stage, goal_pose=goal, goal_joints=joints,
                         payload=carrying, cancelled=self.cancelled.is_set)
        if self.config.raw.get('trajectory_profile') != 'smooth':
            return self._prepare(pool.submit(self.planner.route, state, **arguments),
                                 self.owner, version, epoch)
        from .stationary_hold import StationaryHold
        arguments['start_joints'] = self.rest_anchor.copy()
        with StationaryHold(self, JointStreamSender, epoch=epoch, carrying=carrying) as hold:
            future = pool.submit(self.planner.route, state, **arguments)
            return self._prepare(future, self.owner, version, epoch, hold=hold)

    def _taught_route(self, pool, state, segment, *, carrying, epoch, version):
        from .stationary_hold import StationaryHold
        from .taught import certify_taught
        anchor = self.rest_anchor.copy()
        with StationaryHold(self, JointStreamSender, epoch=epoch, carrying=carrying) as hold:
            future = pool.submit(certify_taught, self.planner, self.config, segment, state, anchor,
                payload=carrying, cancelled=self.cancelled.is_set, clock=self.clock)
            return self._prepare(future, self.owner, version, epoch, hold=hold)

    def start_carry_prediction(self, path, reference):
        """Called by the stream supervisor after sender start, never by sender."""
        if reference is None or not hasattr(reference, 'follow_end_s') or self.carry_prediction is not None:
            return
        began = self.clock()
        token = future = None
        if self.carry_service is not None and self.config.raw.get('carry_lookahead_enabled', True):
            try:
                if hasattr(self.carry_service, 'plan_return'):
                    token, future = self.carry_service.submit(self.planner, path, reference, carry_only=True)
                else:
                    token, future = self.carry_service.submit(self.planner, path, reference)
            except Exception as error:
                self._emit('carry_prediction_unavailable', {'error':str(error)})
            else:
                self.carry_prediction = (token, future, path.version)
                self._emit('carry_prediction_started', {'source_version':path.version,
                    'identity':path.identity, 'scene_revision':self.planner.scene_revision,
                    'predicted_brake_end_monotonic_s':reference.start_s+path.duration,
                    'predicted_start_joints_rad':path.q1.tolist(),
                    'start_source':'certified_brake_endpoint', 'payload_is_prediction':True,
                    'submission_wall_s':self.clock()-began})
        if self.next_plans is None:
            return
        try:
            self.next_plans.anticipate(self.planner, path, reference, token=token, future=future)
            for pending in self.pending_targets():
                self.next_plans.offer(path.identity, pending)
        except Exception as error:
            self._emit('next_pick_forecast_unavailable', {'error':str(error)})

    def _prepared_taught_cycle(self, pool, state, target, *, recovery=False):
        from .stationary_hold import StationaryHold
        from .prepared_cycle import prepare_cycle
        from .release import release_at_rest
        prefix = 'prepared_taught_recovery' if recovery else 'prepared_taught'
        self.stage = 'planning_'+prefix+'_cycle'
        split = hasattr(self.carry_service, 'plan_return')
        self._emit('taught_cycle_preparing', {'recovery': recovery, 'split_carry_return': split,
            'using_prediction':not recovery and self.carry_prediction is not None,
            'taught_transfer': self.config.raw['taught_segments']['transfer'].summary()})
        owner_version = self.version
        with StationaryHold(self, JointStreamSender, epoch=target.epoch, carrying=True) as hold:
            bundle = None
            if not recovery and self.carry_prediction is not None:
                from .carry_lookahead import revalidate_cycle
                token, prediction, source_version = self.carry_prediction
                began = self.clock()
                ready_before_wait = prediction.done()
                if ready_before_wait:
                    result = self._prepare(prediction, self.owner, owner_version, target.epoch, hold=hold)
                else:
                    self.carry_service.discard()
                    result = {'ok':False, 'error':'到实际减速终点仍未交付预规划，按实际载物状态重算'}
                self._emit('carry_prediction_completed', {k:v for k,v in result.items()
                    if k not in ('bundle','predicted_start_joints')})
                try:
                    if hasattr(self.carry_service,'validate') and result.get('ok'):
                        from .carry_validation import accept_validation
                        request,future = self.carry_service.validate(self.planner,state,
                            self.rest_anchor.copy(),result,token=token,source_version=source_version)
                        answer = self._prepare(future,self.owner,owner_version,target.epoch,hold=hold)
                        bundle,evidence = accept_validation(self.planner,state,self.rest_anchor,result,
                            request,answer,token=token,source_version=source_version,
                            cancelled=self.cancelled.is_set)
                    else:
                        future = pool.submit(revalidate_cycle,self.planner,state,self.rest_anchor.copy(),
                            result,token=token,source_version=source_version,cancelled=self.cancelled.is_set,
                            defer_return_validation=True)
                        bundle,evidence = self._prepare(future,self.owner,owner_version,target.epoch,hold=hold)
                except PickRejected as error:
                    self._active(self.owner, owner_version, target.epoch)
                    hold.check()
                    self._emit('carry_prediction_replanned', {'reason':str(error),
                        'details':error.details, 'elapsed_s':self.clock()-began})
                else:
                    self._emit('carry_prediction_accepted', {**evidence,
                        'ready_before_wait':ready_before_wait,
                        'wait_and_validation_wall_s':self.clock()-began})
            if bundle is None:
                future = pool.submit(prepare_cycle, self.planner, self.config, state,
                    self.rest_anchor.copy(), cancelled=self.cancelled.is_set, recovery=recovery, carry_only=split)
                bundle = self._prepare(future, self.owner, owner_version, target.epoch, hold=hold)
        self._emit('taught_carry_prepared' if split else 'taught_cycle_prepared', {'recovery': recovery,
            'carry_duration_s': bundle.carry.duration, 'return_duration_s': bundle.returning.duration if bundle.returning else None,
            'carry_pieces': bundle.carry.diagnostics['pieces'],
            'return_pieces': bundle.returning.diagnostics['pieces'] if bundle.returning else [],
            'drop_orientation_schedule':bundle.carry.diagnostics.get('drop_orientation_schedule'),
            'continuous_carry':bundle.carry.diagnostics.get('continuous_carry'),
            'carry_time_optimization':bundle.carry.diagnostics.get('post_grasp_timing'),
            'return_time_optimization': next((p['diagnostics']['post_grasp_timing']
                for p in (bundle.returning.diagnostics['pieces'] if bundle.returning else [])
                if 'post_grasp_timing' in p['diagnostics']), None),
            'internal_planning_stops': 0})
        def activate(path):
            self._active(bundle.identity, owner_version, target.epoch)
            if self.planner.scene_revision != bundle.scene_revision:
                raise PickRejected('预计算示教链的场景已改变')
            self.planner.version += 1
            return replace(path, version=self.planner.version)
        carry = activate(bundle.carry)
        pending_return = (self.carry_service.plan_return(self.planner, carry, state['width_m'], recovery=recovery)
                          if split else None)
        if pending_return is not None:
            self._emit('empty_return_planning_started', {'identity':carry.identity,
                'scene_revision':bundle.scene_revision, 'source':'nominal_release'})
        carry_origin = self.clock()+self.config.raw['command_advance_s']
        if not recovery and self.next_plans is not None:
            self.next_plans.begin(self.planner, carry, bundle.returning, carry_origin)
            if pending_return is not None:
                def update_return_estimate(job):
                    if not job.cancelled() and job.exception() is None:
                        self.next_plans.return_prepared(carry.identity, job.result().duration)
                pending_return.add_done_callback(update_return_estimate)
            for pending in self.pending_targets():
                self.next_plans.offer(target.identity, pending)
        returning = None
        abandoned = Event()
        def return_cancelled():
            return abandoned.is_set() or self.cancelled.is_set()
        def finish_split_return():
            nonlocal returning
            from .prepared_cycle import prepare_empty_return
            self._active(bundle.identity, owner_version, target.epoch)
            if self.planner.released_tree is None:
                raise PickRejected('空载回程缺少实际放料点云')
            width = self.config.settings.open_width_m
            wait_started = self.clock()
            try:
                candidate = pending_return.result()
            except PickRejected as error:
                candidate = None
                reason = str(error)
            return_wait_s = self.clock() - wait_started
            self._active(bundle.identity, owner_version, target.epoch)
            if return_cancelled() or self.planner.scene_revision != bundle.scene_revision:
                raise PickRejected('回程复核已取消或场景已改变')
            if candidate is not None:
                if (candidate.identity != carry.identity
                        or not np.allclose(candidate.q0, carry.q1, rtol=0, atol=1e-9)
                        or not np.allclose(candidate.q1, self.config.standby, rtol=0, atol=1e-9)):
                    raise PickRejected('回程候选与当前任务或端点不符')
                try:
                    self.planner.check_path(candidate, width, payload=False, cancelled=return_cancelled)
                except PickRejected as error:
                    candidate = None
                    reason = str(error)
            if candidate is None:
                self._active(bundle.identity, owner_version, target.epoch)
                if return_cancelled():
                    raise PickRejected('回程复核已取消')
                self._emit('return_replanned_after_release', {'reason': reason})
                candidate = prepare_empty_return(self.planner, self.config, carry.q1,
                    cancelled=return_cancelled, recovery=recovery)
                self.planner.check_path(candidate, width, payload=False, cancelled=return_cancelled)
            if return_cancelled():
                raise PickRejected('回程复核已取消')
            returning = activate(candidate)
            self._emit('empty_return_prepared', {'identity':carry.identity, 'duration_s':returning.duration,
                'actual_release_checked':True, 'wait_for_candidate_s':return_wait_s})
        def verify_actual_return():
            nonlocal returning
            if split:
                return pool.submit(finish_split_return)
            returning = activate(bundle.returning)
            return pool.submit(self.planner.check_path, returning, self.config.settings.open_width_m,
                payload=False, cancelled=self.cancelled.is_set)
        try:
            state = self._stream(carry, origin=carry_origin,
                epoch=target.epoch, carrying=True, release_on_arrival=True)
            self.holding.check_holding(state)
            self.stage = 'recovery_release' if recovery else 'release'
            state = release_at_rest(self, state, epoch=target.epoch, sender_factory=JointStreamSender,
                prepare_return=verify_actual_return)
        finally:
            abandoned.set()
            if pending_return is not None and not pending_return.done():
                self.carry_service.discard()
                pending_return.cancel()
        if split and not recovery and self.next_plans is not None:
            self.next_plans.return_prepared(target.identity, returning.duration)
        if not recovery and self.claim_after_release is not None:
            from .streaming import fresh_feedback
            from .frozen_queue import BinHandoff
            self._active(bundle.identity, owner_version, target.epoch)
            latest = self.device.snapshot()
            fresh_feedback(latest, self.clock(), self.feedback_stamps)
            self.planner.model.check_feedback(latest)
            validate_start_state(self.config, latest, release_joints=carry.q1,
                tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
            if self.planner.released_tree is None:
                raise PickRejected('料框交接缺少本件释放点云')
            if self.next_plans is not None:
                self.next_plans.observe_release(target.identity)
            entry = self.claim_after_release(released_owner=target.identity)
            if entry is not None:
                origin, candidate = 'bin', None
                if self.next_plans is not None:
                    origin, candidate, decision = self.next_plans.choose(target.identity, entry,
                        active=lambda: self._active(bundle.identity, owner_version, target.epoch),
                        sleep=self.sleep)
                    # A late exposure may require a bounded planning wait.
                    latest = self.device.snapshot()
                    fresh_feedback(latest, self.clock(), self.feedback_stamps)
                    self.planner.model.check_feedback(latest)
                    validate_start_state(self.config, latest, release_joints=carry.q1,
                        tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
                    origin, candidate = self.next_plans.confirm_choice(
                        target.identity, entry, origin, candidate, decision)
                    self._emit('next_pick_origin_selected', {**decision,
                        'preplan_submitted_s':candidate.submitted_s if candidate is not None else None,
                        'preplan_ready_s':candidate.ready_s if candidate is not None else None})
                release_feedback = latest
                release_q = tuple(latest['joints'])
                if origin == 'wait':
                    latest = self._stream(returning, origin=self.clock()+self.config.raw['command_advance_s'],
                        epoch=target.epoch, carrying=False)
                    validate_start_state(self.config, latest, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
                    if candidate is None and self.next_plans is not None:
                        candidate = self.next_plans.wait_candidate(target.identity, entry.target, None, after_return=True)
                self.next_bin_pick = BinHandoff(entry, target.identity, release_q,
                    self.feedback_stamps.copy(), self.planner.released_tree, origin, candidate, self.clock())
                self._emit('bin_pick_handoff' if origin == 'bin' else 'wait_pick_handoff', {'previous_identity':target.identity,
                    'next_identity':entry.target.identity, 'capture_s':entry.target.capture_s,
                    'preplan_ready_s':candidate.ready_s if candidate is not None else None,
                    'release_joints_rad':release_q, 'release_feedback':release_feedback, 'handoff_feedback':latest,
                    'observation_age_s':self.clock()-entry.target.capture_s})
                return latest
        # Opening was confirmed while the current released-cloud certificate
        # ran in the worker. No timing or path search is performed here.
        return self._stream(returning, origin=self.clock()+self.config.raw['command_advance_s'],
            epoch=target.epoch, carrying=False)

    def run(self, target):
        from .motion_gc import DeferCyclicGC
        memory = DeferCyclicGC(clock=self.clock)
        try:
            with memory:
                try:
                    result = self._run_with_recovery(target)
                finally:
                    if self.next_plans is not None:
                        self.next_plans.clear()
                    if self.bin_start is not None:
                        self.planner.bin_departure_tree = None
                    if self.carry_service is not None:
                        self.carry_service.discard()
                    self.carry_prediction = None
                    self.pending_rest_arrival = None
            from .idle_feedback import refresh_after_collection
            refresh_after_collection(self, memory.report, epoch=target.epoch)
            return result
        finally:
            if self.brake_height_guard is not None:
                self._emit('brake_height_feedback', self.brake_height_guard.feedback_report())
            self._emit('motion_memory_timing', memory.report)

    def _run_with_recovery(self, target):
        from .recovery import recover_attempt, RecoveryFailed
        try:
            report = self._run_attempt(target)
        except BaseException as error:
            self.pending_rest_arrival = None
            if self.cancelled.is_set() or not isinstance(error, PickRejected):
                self.cancel()
                if self.motion_started and not self.height_fault_stop_sent:
                    self.device.stop()
                if self.brake_height_guard is not None and self.brake_height_guard.fault is not None:
                    raise self.brake_height_guard.fault
                raise
            failure = error
        else:
            if report['status'] != 'attempt_completed_grip_unconfirmed':
                return report
            failure = PickRejected('本次抓取未确认夹持，检查后投放或回等待位',
                details={'reason': 'grip_unconfirmed'})
        # _run_attempt has joined its sender and all planner workers here.
        # A failed attempt must never set the shared user-cancellation event.
        if self.next_plans is not None:
            self.next_plans.clear()
        if self.carry_service is not None:
            self.carry_service.discard()
        self.carry_prediction = None
        try:
            return recover_attempt(self, target, failure)
        except BaseException as error:
            self.cancel()
            if self.motion_started and not self.height_fault_stop_sent:
                self.device.stop()
            if not isinstance(error, Exception):
                raise
            raise RecoveryFailed(failure, error) from error

    def _plan_standard_intercept(self, pool, target, state, prefetched, version, *, planning_budget_s=None):
        """Admit the ordinary trajectory before any joint trajectory is streamed."""
        budget = self.config.raw.get('planning_budget_s', float('inf'))
        if planning_budget_s is not None:
            budget = min(budget, planning_budget_s)
        self.stage = 'planning_intercept'
        self._emit('planning_submitted', {'identity': self.owner, 'trajectory_generation': version,
            'capture_monotonic_s': target.capture_s,
            'submission_monotonic_s': self.clock(),
            'preplanned': prefetched is not None,
            'original_submission_monotonic_s': None if prefetched is None else prefetched.submitted_s,
            'start_joints_rad': list(state['joints']),
            'start_tcp_base': self.planner.model.tcp().tolist(),
            'target_center_base_m': getattr(target, 'center', None),
            'entry_observed_position_base_m': getattr(target, 'entry_position', None),
            'velocity_base_m_s': getattr(target, 'velocity', None),
            'speed_report': getattr(target, 'speed_report', None),
            'candidate_tcp_base': [np.asarray(g).tolist() for g in target.grasps]})
        measured = self.config.raw.get('planning_reserve_mode', 'fixed') == 'measured'
        experiment = self.config.raw.get('timing_experiment')
        from .delivery_repair import enabled as repair_enabled, repair_intercept, record as record_repair
        from .return_lookahead import skip_late_return
        fixed_reservation = prefetched is not None and skip_late_return(self.config)
        use_repair = repair_enabled(self.config) and not fixed_reservation
        retry_expired = measured if experiment is None else experiment['expiry_policy'] == 'replan'
        bounded = measured or experiment is not None or planning_budget_s is not None
        delivery_limit = ((experiment['max_delivery_attempts'] if retry_expired else 1)
                          if experiment is not None else (4 if measured else 1))
        if use_repair:
            retry_expired, bounded, delivery_limit = True, True, max(4, delivery_limit)
        if fixed_reservation:
            retry_expired, delivery_limit = False, 1
        planning_started = self.clock()
        planning_deadline = planning_started+budget
        pending_repair, repair_attempts = None, 0
        prepared = False
        execution_begin = None
        for delivery_attempt in range(delivery_limit):
            if bounded and self.clock()-planning_started >= budget:
                raise InterceptUnavailable('实际发送前重规划预算耗尽')
            arguments = {'cancelled': self.cancelled.is_set}
            if use_repair or planning_budget_s is not None:
                arguments['planning_deadline'] = planning_deadline
            if experiment is not None:
                arguments['experiment_deadline'] = planning_started+budget
            try:
                plan = None
                decision = None
                if pending_repair is not None:
                    old, pending_repair = pending_repair, None
                    from .streaming import fresh_feedback
                    latest = self.device.snapshot()
                    fresh_feedback(latest, self.clock(), self.feedback_stamps)
                    validate_start_state(self.config, latest, require_wait=False)
                    if np.max(abs(np.asarray(latest['joints'])-np.asarray(state['joints']))) > .005:
                        raise PickRejected('补救前机械臂起点改变，拒绝复用')
                    state = latest
                    self._active(self.owner, version, target.epoch)
                    from .delivery_repair import rank_recovery_branch
                    decision = rank_recovery_branch(self.planner, old, state)
                    record_repair(self, 'delivery_recovery_branch_selected', decision)
                    self._active(self.owner, version, target.epoch)
                    if self.clock() >= planning_deadline:
                        raise InterceptUnavailable('选择补救分支后总预算耗尽')
                if decision is not None and decision['branch'] == 'repair':
                    repair_attempts += 1
                    repair_started = self.clock()
                    record_repair(self, 'delivery_repair_submitted', {
                        'repair_attempt':repair_attempts, 'start_state':state,
                        'original_start_s':old.reference.start_s,
                        'original_send_deadline_s':old.reference.start_s-self.config.raw['command_advance_s'],
                        'planning_deadline_s':planning_deadline}, old)
                    try:
                        future = pool.submit(repair_intercept, self.planner, old, state,
                            deadline=planning_deadline, cancelled=self.cancelled.is_set)
                        plan = self._prepare(future, self.owner, version, target.epoch)
                    except PickRejected as error:
                        record_repair(self, 'delivery_repair_rejected', {
                            'repair_attempt':repair_attempts, 'error':str(error),
                            'wall_s':self.clock()-repair_started,
                            'details':error.details})
                        self._active(self.owner, version, target.epoch)
                        if self.clock() >= planning_deadline:
                            raise
                        record_repair(self, 'delivery_repair_fallback_replan', {
                            'remaining_budget_s':planning_deadline-self.clock()})
                    else:
                        record_repair(self, 'delivery_repair_ready', {
                            'repair_attempt':repair_attempts,
                            **plan.path.diagnostics['delivery_repair'],
                            'phase_timing':plan.path.diagnostics.get('phase_timing'),
                            'bounds':plan.path.diagnostics.get('delivery_repair_bounds'),
                            'peak_acceleration_rad_s2':plan.path.diagnostics['peak_acceleration_rad_s2']}, plan)
                        offer_plan = getattr(self.planner, 'offer_intercept_plan', None)
                        if offer_plan is not None:
                            offer_plan(plan)
                adopted_next = False
                if delivery_attempt == 0 and self.incoming_handoff is not None and self.incoming_handoff.candidate is not None:
                    from .next_pick import adopt
                    handoff = self.incoming_handoff
                    check_started = self.clock()
                    send_deadline = handoff.candidate.deadline(self.config)
                    missed_preplan_deadline = False
                    try:
                        plan = adopt(self.planner, handoff.candidate, handoff,
                            target, state, cancelled=lambda: self.cancelled.is_set() or self.clock() >= planning_deadline)
                    except PickRejected as error:
                        missed_preplan_deadline = error.details.get('reason') == 'preplan_send_deadline'
                        self._active(self.owner, version, target.epoch)
                        self._emit('next_pick_preplan_rejected', {'reason':str(error), 'details':error.details,
                            'identity':target.identity, 'capture_s':target.capture_s,
                            'origin':handoff.origin, 'check_wall_s':self.clock()-check_started,
                            'send_margin_s':send_deadline-self.clock()})
                    else:
                        adopted_next = True
                        self._emit('next_pick_preplan_adopted', {'identity':target.identity, 'capture_s':target.capture_s,
                            'origin':handoff.origin, 'actual_start_joints_rad':state['joints'],
                            'check_wall_s':self.clock()-check_started, 'send_margin_s':send_deadline-self.clock()})
                        offer_plan = getattr(self.planner, 'offer_intercept_plan', None)
                        if offer_plan is not None:
                            offer_plan(plan)
                    finally:
                        if self.next_plans is not None and handoff.ready_s is not None:
                            elapsed = self.clock()-handoff.ready_s
                            if handoff.origin == 'wait':
                                self.next_plans.observe_wait_check(elapsed)
                            elif adopted_next or missed_preplan_deadline:
                                self.next_plans.observe_bin_check(elapsed, timed_out=missed_preplan_deadline)
                if plan is not None:
                    pass  # The same preparation, dispatch and complete-cycle code follows.
                elif delivery_attempt == 0 and prefetched is not None:
                    plan = prefetched.plan
                    self._emit('lookahead_plan_adopted', {'identity': target.identity,
                        'original_capture_monotonic_s': target.capture_s,
                        'prepared_monotonic_s': prefetched.ready_s,
                        'original_submission_monotonic_s': prefetched.submitted_s,
                        'actual_start_joints_rad': list(state['joints']),
                        'send_margin_s': plan.reference.start_s-
                            self.config.raw['command_advance_s']-self.clock()})
                else:
                    if use_repair and delivery_attempt > 0:
                        arguments['recovery_replan'] = True
                    future = pool.submit(self.planner.intercept, target, state, **arguments)
                    try:
                        plan = self._prepare(future, self.owner, version, target.epoch)
                    except PickRejected as error:
                        self._active(self.owner, version, target.epoch)
                        raise InterceptUnavailable(str(error), details=error.details) from error
            except PickRejected as error:
                if experiment is not None:
                    self._emit('experiment_plan_failed', {'identity': self.owner,
                        'group': experiment['group'], 'delivery_attempt': delivery_attempt+1,
                        'elapsed_since_submission_s': self.clock()-planning_started,
                        'reason': str(error), 'details': error.details})
                raise
            self._active(self.owner, version, target.epoch)
            if bounded and self.clock()-planning_started >= budget:
                raise InterceptUnavailable('实际发送前重规划预算耗尽')
            if plan.target.identity != self.owner or plan.path.version != self.planner.version:
                raise PickRejected('晚到的旧目标或旧版本规划')
            if 'phase_timing' in self.config.raw:
                from .phase_timing import verify_plan
                verify_plan(self.config, plan.path, plan.reference, for_execution=True)
            if 'execution_delay_window' in self.config.raw:
                from .execution_delay_window import verify_plan
                verify_plan(self.config, plan.path, plan.reference)
            if 'delivery_repair' in plan.path.diagnostics:
                from .delivery_repair import verify_repaired_plan
                verify_repaired_plan(self.config, plan.path, plan.reference)
            post_start = plan.path.diagnostics.get('delivery_repair',
                plan.path.diagnostics.get('timing_experiment',
                plan.path.diagnostics.get('measured_planning', {}))).get(
                'completed_monotonic_s', self.clock())
            if adopted_next or (prefetched is not None and plan is prefetched.plan):
                # Time waiting for the previous return is not dispatch overhead.
                post_start = self.clock()
            if experiment is not None:
                from .timing_experiment import plan_record
                try:
                    record = plan_record(self.config, plan, self.clock(), delivery_attempt+1)
                except Exception as error:
                    # Diagnostic collection cannot alter a checked motion.
                    record = {'group': experiment['group'], 'error': str(error),
                              'delivery_attempt': delivery_attempt+1}
                    self._emit('experiment_record_error', record)
                else:
                    self._emit('experiment_plan_ready', record)
            send_deadline = plan.reference.start_s-self.config.raw['command_advance_s']
            def missed_delivery(reason):
                nonlocal pending_repair
                if fixed_reservation:
                    details = {'reason': 'lookahead_send_deadline_missed',
                        'send_deadline_monotonic_s': send_deadline,
                        'send_margin_s': send_deadline-self.clock(), 'next_action': 'skip_target'}
                    self._emit('planning_delivery_missed', {'reason': reason, **details})
                    raise InterceptUnavailable('预取轨迹错过既定发送时刻，跳过该工件', details=details)
                if experiment is not None:
                    self._emit('experiment_delivery_missed', {'identity': self.owner,
                        'group': experiment['group'], 'reason': reason,
                        'delivery_attempt': delivery_attempt+1,
                        'post_planning_wall_s': max(0., self.clock()-post_start),
                        'send_deadline_monotonic_s': send_deadline})
                if not retry_expired:
                    raise InterceptUnavailable(reason)
                observed = max(0., self.clock()-post_start)
                previous = float(getattr(self.planner, 'measured_post_planning_s', 0.))
                if measured:
                    self.planner.measured_post_planning_s = max(previous, observed)
                if use_repair:
                    self.planner.measured_post_planning_s = max(previous, observed)
                    if repair_attempts < self.config.raw['delivery_repair']['max_attempts']:
                        pending_repair = plan
                    else:
                        record_repair(self, 'delivery_repair_limit_reached', {
                            'repair_attempts':repair_attempts})
                self._emit('planning_delivery_missed', {'reason':reason,
                    'post_planning_wall_s':observed, 'send_deadline_monotonic_s':send_deadline,
                    'next_action':'repair' if pending_repair is not None else 'replan',
                    'attempt':delivery_attempt+1})
            if self.clock() >= send_deadline:
                missed_delivery('实际发送前轨迹预约已过期')
                continue
            latest = self.device.snapshot()
            if np.max(abs(np.asarray(latest['joints'])-np.asarray(state['joints']))) > .005:
                raise PickRejected('规划期间机械臂起点改变')
            if not prepared:
                execution_begin = self.clock()
                with self.command_lock:
                    self._active(self.owner, version, target.epoch)
                    self.motion_started = True  # speed configuration is a real send
                    self.device.prepare_motion(self.config.settings.speed_percent,
                        query_limits=self.config.settings.query_controller_limits,
                        requested_velocity_limits=self.config.velocity_limits,
                        requested_acceleration_limits=self.config.acceleration_limits)
                prepared = True
            if self.clock() >= send_deadline:
                missed_delivery('厂家运动参数准备后已错过轨迹发送起点')
                continue
            prepared_state = self.device.snapshot()
            if (np.max(abs(np.asarray(prepared_state['joints'])-np.asarray(state['joints']))) > .005 or
                abs(prepared_state['width_m']-self.config.settings.open_width_m) > .002 or
                prepared_state['force_n'] >= self.config.settings.min_force_n):
                raise PickRejected('厂家运动参数准备期间机械臂或夹爪起点改变')
            if self.clock() >= send_deadline:
                missed_delivery('发送前反馈读取已错过轨迹起点')
                continue
            if measured or use_repair:
                self.planner.measured_post_planning_s = max(
                    float(getattr(self.planner, 'measured_post_planning_s', 0.)),
                    max(0., self.clock()-post_start))
            self._emit('planning_delivery', {'capture_monotonic_s': target.capture_s,
                'received_monotonic_s': target.received_s, 'device_time_s': target.device_s,
                'delivery_monotonic_s': self.clock(), 'contact_monotonic_s': plan.reference.contact_s,
                'planning_wall_s': plan.path.planning_s, 'identity': target.identity,
                'planning_components_s': plan.timings_s,
                'trajectory_diagnostics': plan.path.diagnostics,
                'selected_grasp_tcp_base': plan.selected_grasp.tolist(),
                'T_base_object_at_exposure': target.object_pose.tolist(),
                'T_object_tcp_selected': (np.linalg.inv(target.object_pose)@plan.selected_grasp).tolist(),
                'predicted_contact_tcp_base_m': plan.reference.contact_xyz.tolist(),
                'velocity_base_m_s': target.velocity.tolist(),
                'time_base': 'host_monotonic', 'clock_epoch': target.epoch})
            if self.clock() >= send_deadline:
                missed_delivery('日志交付后已错过轨迹发送起点')
                continue
            if measured or use_repair:
                self.planner.measured_post_planning_s = max(
                    self.planner.measured_post_planning_s, max(0., self.clock()-post_start))
            if ((experiment is not None or use_repair or planning_budget_s is not None) and
                    self.clock()-planning_started >= budget):
                if experiment is not None:
                    self._emit('experiment_budget_exhausted', {'identity': self.owner,
                        'group': experiment['group'], 'delivery_attempt': delivery_attempt+1,
                        'elapsed_since_submission_s': self.clock()-planning_started})
                raise InterceptUnavailable('设备准备与交付已耗尽总规划预算')
            self.trace.policy_origin = plan.reference.start_s
            break
        else:
            raise InterceptUnavailable('多次规划交付后均错过实际发送时刻')
        return plan, state, execution_begin

    def _execute_intercept(self, pool, target, state, prefetched, version):
        if self.config.raw.get('hover_descent', {}).get('enabled', False):
            from .hover_execution import execute_intercept
            return execute_intercept(self, pool, target, state)
        plan, state, begin = self._plan_standard_intercept(pool, target, state, prefetched, version)
        state = self._stream(plan.path, origin=plan.reference.start_s, epoch=target.epoch,
                             reference=plan.reference)
        return plan, state, begin

    def _run_attempt(self, target):
        self.command_history = CommandHistory(self.config)
        self.rest_anchor = None
        handoff = self.incoming_handoff or self.bin_start
        self.feedback_stamps = {} if handoff is None else handoff.feedback_stamps.copy()
        release_joints = None
        if handoff is not None:
            if (handoff.entry is not self.frozen_entry or self.frozen_entry.target is not target
                    or handoff.previous_identity == target.identity
                    or handoff.previous_identity[0] != target.epoch):
                raise PickRejected('料框交接工件身份或时钟无效')
        if self.bin_start is not None:
            self.planner.bin_departure_tree = self.bin_start.released_tree
            release_joints = self.bin_start.release_joints
        self.owner = target.identity
        self.trace.target_identity = replace(TargetIdentity.confirmed(target.capture_s, target.epoch),
            track_id=int(target.identity[2]), track_generation=int(target.identity[1]))
        self.version += 1
        version = self.version
        self._active(self.owner, version, target.epoch)
        self.stage = 'checking_start_state'
        state = self.device.snapshot()
        self.planner.model.check_feedback(state)
        validate_start_state(self.config, state, tolerance_rad=(
            TRAJECTORY_ARRIVAL_TOLERANCE_RAD if release_joints is None else .005),
            release_joints=release_joints)
        prefetched = self.prefetched_intercept
        if prefetched is not None:
            prefetched.validate(self.config, self.epoch(), self.clock(),
                allow_expired_for_repair=True)
            if prefetched.target is not target or prefetched.planner is not self.planner:
                raise PickRejected('预取轨迹不属于本次执行器')
            from .streaming import fresh_feedback
            fresh_feedback(state, self.clock(), self.feedback_stamps)
            if (np.max(abs(np.asarray(state['joints'])-prefetched.plan.path.q0))
                    > TRAJECTORY_ARRIVAL_TOLERANCE_RAD):
                from .return_lookahead import skip_late_return
                if skip_late_return(self.config):
                    raise PickRejected('预取轨迹起点与实际回位不符，跳过该工件', details={
                        'reason': 'lookahead_start_mismatch', 'next_action': 'skip_target'})
                # Recompute from measured joints using the original exposure.
                # The speculative standby waypoint is not device evidence.
                self._emit('lookahead_start_replan', {'identity': target.identity,
                    'reason': 'actual_wait_differs_from_planned_start',
                    'joint_error_rad': (np.asarray(state['joints'])-prefetched.plan.path.q0).tolist()})
                prefetched = None
        if self.frozen_entry is not None:
            if self.frozen_entry.target is not target:
                raise PickRejected('冻结队列交接的工件与本次抓取不一致')
            reason = self.frozen_entry.invalid_reason(self.config, self.epoch(), self.clock())
            if reason is not None:
                raise PickRejected('冻结工件在开始规划前已失效', details={'reason': reason})
            from .streaming import fresh_feedback
            fresh_feedback(state, self.clock(), self.feedback_stamps)
            if self.next_plans is not None and self.bin_start is None:
                self.next_plans.observe_wait(state['joints'])
        elif (self.prefetched_intercept is None and
                not 0 <= self.clock()-target.capture_s <= self.config.raw['maximum_observation_age_s']):
            raise PickRejected('目标曝光已过期')
        self._emit('bin_pick_started' if self.bin_start is not None else 'wait_pick_started', {
            'identity':target.identity, 'capture_s':target.capture_s, 'measured_start_joints_rad':state['joints']})
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='two-second-planner') as pool:
            try:
                plan, state, execution_begin = self._execute_intercept(
                    pool, target, state, prefetched, version)
                if self.config.raw.get('closure_policy', 'feedback') == 'scheduled_attempt':
                    held_width = self.holding.held_width
                    grip_at_end = (held_width is not None and state['width_m'] >= .003 and
                        abs(state['width_m']-held_width) <= .003 and
                        state['force_n'] >= self.config.settings.min_force_n)
                    if not grip_at_end:
                        self._emit('attempt_finished_grip_unconfirmed', {'width_m': state['width_m'],
                            'force_n': state['force_n'], 'return_motion_started': False})
                        self.holding.held_width = None
                        self.stage = 'attempt_finished'
                        self.owner = None
                        self.planner.identity = None
                        self.planner.version += 1
                        return {'status': 'attempt_completed_grip_unconfirmed', 'target_identity': target.identity,
                            'physical_grasp_success': None, 'physical_bin_placement_verified': False,
                            'execution_and_carry_planning_wall_s': self.clock()-execution_begin,
                            'note': '完成接近、闭合、跟随和减速；未确认夹持，停在轨迹终点'}
                if hasattr(plan.reference, 'follow_end_s'):
                    # Attachment was frozen at the actual grip-confirmation
                    # timestamp, before braking. Never extrapolate it again.
                    self.planner.reanchor_support(state)
                else:
                    self.planner.attach_measured(state, self.clock())
                self._emit('grip_confirmed_from_width_force', {'width_m': state['width_m'], 'force_n': state['force_n'],
                    'physical_lift_proven': False})
                if self.bin_start is not None:
                    self.planner.bin_departure_tree = None
                self.planner.model.check_feedback(state)
                lift = self.planner.model.tcp().copy()
                lift[:3, 3] += self.planner.support*self.config.settings.lift_m
                def do_route(state, stage, *, goal=None, joints=None, carrying=False, taught=None):
                    self.stage = 'planning_'+stage
                    connection = {'transfer_connect':'transfer', 'return_connect':'return'}.get(stage)
                    segment = self.config.raw.get('taught_segments', {}).get(connection)
                    self._emit('route_submitted', {'stage': stage, 'carrying':carrying,
                        'measured_start_joints_rad':state['joints'],
                        'goal_joints_rad':None if joints is None else np.asarray(joints).tolist(),
                        'taught_segment':None if segment is None else segment.summary()})
                    if taught is None:
                        path = self._route(pool, state, stage=stage, goal=goal, joints=joints,
                                           carrying=carrying, epoch=target.epoch, version=version)
                    else:
                        path = self._taught_route(pool, state, taught, carrying=carrying,
                                                  epoch=target.epoch, version=version)
                    self._emit('route_delivered', {'stage':stage,'planning_wall_s':path.planning_s,
                        'trajectory_duration_s':path.duration,'trajectory_version':path.version,
                        'diagnostics':path.diagnostics})
                    current = self.device.snapshot()
                    if np.max(abs(np.asarray(current['joints'])-np.asarray(state['joints']))) > .005:
                        raise PickRejected('搬运规划期间反馈起点改变')
                    return self._stream(path, origin=self.clock()+self.config.raw['command_advance_s'],
                                        epoch=target.epoch, carrying=carrying,
                                        release_on_arrival=stage in ('transfer','taught_transfer'))

                taught = (self.config.raw.get('taught_segments')
                          if self.config.raw.get('taught_playback_enabled', True) else None)
                if taught and self.config.raw.get('prepared_taught_cycle_enabled', False):
                    state = self._prepared_taught_cycle(pool, state, target)
                else:
                    state = self._legacy_transport(pool, state, target, lift, taught, do_route)
                returned = self.next_bin_pick is None or self.next_bin_pick.origin == 'wait'
                if returned:
                    validate_start_state(self.config,state,tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
                self.stage = 'waiting' if returned else 'bin_pick_handoff'
                report = {'status': 'cycle_completed_from_feedback', 'target_identity': self.owner,
                        'returned_to_wait':returned,
                        'bin': self.config.default_bin, 'physical_grasp_success': None,
                        'physical_bin_placement_verified': False,
                        'execution_and_carry_planning_wall_s': self.clock()-execution_begin,
                        'note': '宽度/受力/关节反馈只证明程序闭环，真实抬升与入框需要现场证据'}
                if self.next_bin_pick is not None:
                    report['next_target_identity'] = self.next_bin_pick.entry.target.identity
                self.owner = None
                self.planner.identity = None
                self.planner.version += 1
                return report
            except BaseException:
                if self.sender is not None:
                    self.sender.request_stop()
                    self.sender.join()
                    self.sender = None
                raise

    def _legacy_transport(self, pool, state, target, lift, taught, do_route):
        state = do_route(state, 'lift', goal=lift, carrying=True)
        if taught:
            state = do_route(state, 'transfer_connect', joints=taught['transfer'].joints[0], carrying=True)
            state = do_route(state, 'taught_transfer', carrying=True, taught=taught['transfer'])
        else:
            state = do_route(state, 'transfer', goal=self.config.bins[self.config.default_bin], carrying=True)
        self.stage = 'release'
        self.holding.check_holding(state)
        from .release import release_at_rest
        state = release_at_rest(self, state, epoch=target.epoch, sender_factory=JointStreamSender)
        if taught and self.config.raw.get('taught_return_strategy', 'recorded_then_wait') != 'direct_to_wait':
            state = do_route(state, 'return_connect', joints=taught['return'].joints[0])
            state = do_route(state, 'taught_return', taught=taught['return'])
        return do_route(state, 'return', joints=self.config.standby)

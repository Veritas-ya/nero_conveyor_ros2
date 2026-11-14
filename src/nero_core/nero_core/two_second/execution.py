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


def validate_start_state(config, state, *, require_wait=True, tolerance_rad=.005):
    """Read-only readiness with an explicit wait-position tolerance."""
    actual = np.asarray(state['joints'], dtype=float)
    expected = np.asarray(config.standby, dtype=float)
    if actual.shape != (7,) or not np.isfinite(actual).all():
        raise PickRejected('起始关节反馈无效', details={'reason': 'startup_joint_feedback'})
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
        raise PickRejected('等待位必须预先张爪；不自动盲开夹爪', details={'reason': 'startup_gripper_width',
            'width_m': state['width_m'], 'required_width_m': config.settings.open_width_m,
            'tolerance_m': .002})


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
        self.feedback_stamps = {}
        self.events = []
        self.event_sink = event_sink
        self.event_logging_errors = []
        self.return_started = None
        self.prefetched_intercept = None
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

    def cancel(self):
        self.cancelled.set()
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
        if (self.carry_service is None or reference is None
                or not hasattr(reference, 'follow_end_s') or self.carry_prediction is not None):
            return
        began = self.clock()
        try:
            token, future = self.carry_service.submit(self.planner, path, reference)
        except Exception as error:
            self._emit('carry_prediction_unavailable', {'error':str(error)})
            return
        self.carry_prediction = (token, future, path.version)
        self._emit('carry_prediction_started', {'source_version':path.version,
            'identity':path.identity, 'scene_revision':self.planner.scene_revision,
            'predicted_brake_end_monotonic_s':reference.start_s+path.duration,
            'predicted_start_joints_rad':path.q1.tolist(),
            'start_source':'certified_brake_endpoint', 'payload_is_prediction':True,
            'submission_wall_s':self.clock()-began})

    def _prepared_taught_cycle(self, pool, state, target, *, recovery=False):
        from .stationary_hold import StationaryHold
        from .prepared_cycle import prepare_cycle
        from .release import release_at_rest
        prefix = 'prepared_taught_recovery' if recovery else 'prepared_taught'
        self.stage = 'planning_'+prefix+'_cycle'
        self._emit('taught_cycle_preparing', {'recovery': recovery,
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
                future = pool.submit(revalidate_cycle, self.planner, state, self.rest_anchor.copy(),
                    result, token=token, source_version=source_version, cancelled=self.cancelled.is_set,
                    defer_return_validation=True)
                try:
                    bundle, evidence = self._prepare(future, self.owner, owner_version, target.epoch, hold=hold)
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
                    self.rest_anchor.copy(), cancelled=self.cancelled.is_set, recovery=recovery)
                bundle = self._prepare(future, self.owner, owner_version, target.epoch, hold=hold)
        self._emit('taught_cycle_prepared', {'recovery': recovery,
            'carry_duration_s': bundle.carry.duration, 'return_duration_s': bundle.returning.duration,
            'carry_pieces': bundle.carry.diagnostics['pieces'],
            'return_pieces': bundle.returning.diagnostics['pieces'],
            'drop_orientation_schedule':bundle.carry.diagnostics.get('drop_orientation_schedule'),
            'continuous_carry':bundle.carry.diagnostics.get('continuous_carry'),
            'internal_planning_stops': 0})
        def activate(path):
            self._active(bundle.identity, owner_version, target.epoch)
            if self.planner.scene_revision != bundle.scene_revision:
                raise PickRejected('预计算示教链的场景已改变')
            self.planner.version += 1
            return replace(path, version=self.planner.version)
        carry = activate(bundle.carry)
        state = self._stream(carry, origin=self.clock()+self.config.raw['command_advance_s'],
            epoch=target.epoch, carrying=True, release_on_arrival=True)
        self.holding.check_holding(state)
        self.stage = 'recovery_release' if recovery else 'release'
        returning = None
        def verify_actual_return():
            nonlocal returning
            returning = activate(bundle.returning)
            return pool.submit(self.planner.check_path, returning, self.config.settings.open_width_m,
                payload=False, cancelled=self.cancelled.is_set)
        state = release_at_rest(self, state, epoch=target.epoch, sender_factory=JointStreamSender,
            prepare_return=verify_actual_return)
        # Opening was confirmed while the current released-cloud certificate
        # ran in the worker. No timing or path search is performed here.
        return self._stream(returning, origin=self.clock()+self.config.raw['command_advance_s'],
            epoch=target.epoch, carrying=False)

    def run(self, target):
        try:
            return self._run_with_recovery(target)
        finally:
            if self.carry_service is not None:
                self.carry_service.discard()
            self.carry_prediction = None

    def _run_with_recovery(self, target):
        from .recovery import recover_attempt, RecoveryFailed
        try:
            report = self._run_attempt(target)
        except BaseException as error:
            if self.cancelled.is_set() or not isinstance(error, PickRejected):
                self.cancel()
                if self.motion_started:
                    self.device.stop()
                raise
            failure = error
        else:
            if report['status'] != 'attempt_completed_grip_unconfirmed':
                return report
            failure = PickRejected('本次抓取未确认夹持，取消并回等待位')
        # _run_attempt has joined its sender and all planner workers here.
        # A failed attempt must never set the shared user-cancellation event.
        if self.carry_service is not None:
            self.carry_service.discard()
        self.carry_prediction = None
        try:
            return recover_attempt(self, target, failure)
        except BaseException as error:
            self.cancel()
            if self.motion_started:
                self.device.stop()
            if not isinstance(error, Exception):
                raise
            raise RecoveryFailed(failure, error) from error

    def _run_attempt(self, target):
        self.command_history = CommandHistory(self.config)
        self.rest_anchor = None
        self.feedback_stamps = {}
        self.owner = target.identity
        self.trace.target_identity = replace(TargetIdentity.confirmed(target.capture_s, target.epoch),
            track_id=int(target.identity[2]), track_generation=int(target.identity[1]))
        self.version += 1
        version = self.version
        self._active(self.owner, version, target.epoch)
        self.stage = 'checking_start_state'
        state = self.device.snapshot()
        self.planner.model.check_feedback(state)
        validate_start_state(self.config, state, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
        prefetched = self.prefetched_intercept
        if prefetched is not None:
            prefetched.validate(self.config, self.epoch(), self.clock())
            if prefetched.target is not target or prefetched.planner is not self.planner:
                raise PickRejected('预取轨迹不属于本次执行器')
            from .streaming import fresh_feedback
            fresh_feedback(state, self.clock(), self.feedback_stamps)
            if np.max(abs(np.asarray(state['joints'])-prefetched.plan.path.q0)) > .005:
                # Recompute from measured joints using the original exposure.
                # The speculative standby waypoint is not device evidence.
                self._emit('lookahead_start_replan', {'identity': target.identity,
                    'reason': 'actual_wait_differs_from_planned_start',
                    'joint_error_rad': (np.asarray(state['joints'])-prefetched.plan.path.q0).tolist()})
                prefetched = None
        if (self.prefetched_intercept is None and
                not 0 <= self.clock()-target.capture_s <= self.config.raw['maximum_observation_age_s']):
            raise PickRejected('目标曝光已过期')
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='two-second-planner') as pool:
            try:
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
                retry_expired = measured if experiment is None else experiment['expiry_policy'] == 'replan'
                bounded = measured or experiment is not None
                delivery_limit = ((experiment['max_delivery_attempts'] if retry_expired else 1)
                                  if experiment is not None else (4 if measured else 1))
                planning_started = self.clock()
                prepared = False
                execution_begin = None
                for delivery_attempt in range(delivery_limit):
                    if bounded and self.clock()-planning_started >= self.config.raw['planning_budget_s']:
                        raise PickRejected('实际发送前重规划预算耗尽')
                    arguments = {'cancelled': self.cancelled.is_set}
                    if experiment is not None:
                        arguments['experiment_deadline'] = planning_started+self.config.raw['planning_budget_s']
                    try:
                        if delivery_attempt == 0 and prefetched is not None:
                            plan = prefetched.plan
                            self._emit('lookahead_plan_adopted', {'identity': target.identity,
                                'original_capture_monotonic_s': target.capture_s,
                                'prepared_monotonic_s': prefetched.ready_s,
                                'original_submission_monotonic_s': prefetched.submitted_s,
                                'actual_start_joints_rad': list(state['joints']),
                                'send_margin_s': plan.reference.start_s-
                                    self.config.raw['command_advance_s']-self.clock()})
                        else:
                            future = pool.submit(self.planner.intercept, target, state, **arguments)
                            plan = self._prepare(future, self.owner, version, target.epoch)
                    except PickRejected as error:
                        if experiment is not None:
                            self._emit('experiment_plan_failed', {'identity': self.owner,
                                'group': experiment['group'], 'delivery_attempt': delivery_attempt+1,
                                'elapsed_since_submission_s': self.clock()-planning_started,
                                'reason': str(error), 'details': error.details})
                        raise
                    self._active(self.owner, version, target.epoch)
                    if bounded and self.clock()-planning_started >= self.config.raw['planning_budget_s']:
                        raise PickRejected('实际发送前重规划预算耗尽')
                    if plan.target.identity != self.owner or plan.path.version != self.planner.version:
                        raise PickRejected('晚到的旧目标或旧版本规划')
                    if 'phase_timing' in self.config.raw:
                        from .phase_timing import verify_plan
                        verify_plan(self.config, plan.path, plan.reference, for_execution=True)
                    if 'execution_delay_window' in self.config.raw:
                        from .execution_delay_window import verify_plan
                        verify_plan(self.config, plan.path, plan.reference)
                    post_start = plan.path.diagnostics.get('timing_experiment',
                        plan.path.diagnostics.get('measured_planning', {})).get(
                        'completed_monotonic_s', self.clock())
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
                        if experiment is not None:
                            self._emit('experiment_delivery_missed', {'identity': self.owner,
                                'group': experiment['group'], 'reason': reason,
                                'delivery_attempt': delivery_attempt+1,
                                'post_planning_wall_s': max(0., self.clock()-post_start),
                                'send_deadline_monotonic_s': send_deadline})
                        if not retry_expired:
                            raise PickRejected(reason)
                        observed = max(0., self.clock()-post_start)
                        previous = float(getattr(self.planner, 'measured_post_planning_s', 0.))
                        if measured:
                            self.planner.measured_post_planning_s = max(previous, observed)
                        self._emit('planning_delivery_missed', {'reason':reason,
                            'post_planning_wall_s':observed, 'send_deadline_monotonic_s':send_deadline,
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
                    if measured:
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
                    if measured:
                        self.planner.measured_post_planning_s = max(
                            self.planner.measured_post_planning_s, max(0., self.clock()-post_start))
                    if (experiment is not None and
                            self.clock()-planning_started >= self.config.raw['planning_budget_s']):
                        self._emit('experiment_budget_exhausted', {'identity': self.owner,
                            'group': experiment['group'], 'delivery_attempt': delivery_attempt+1,
                            'elapsed_since_submission_s': self.clock()-planning_started})
                        raise PickRejected('实验设备准备与交付已耗尽总规划预算')
                    self.trace.policy_origin = plan.reference.start_s
                    break
                else:
                    raise PickRejected('多次规划交付后均错过实际发送时刻')
                state = self._stream(plan.path, origin=plan.reference.start_s, epoch=target.epoch,
                                     reference=plan.reference)
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
                validate_start_state(self.config,state,tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
                self.stage = 'waiting'
                report = {'status': 'cycle_completed_from_feedback', 'target_identity': self.owner,
                        'returned_to_wait':True,
                        'bin': self.config.default_bin, 'physical_grasp_success': None,
                        'physical_bin_placement_verified': False,
                        'execution_and_carry_planning_wall_s': self.clock()-execution_begin,
                        'note': '宽度/受力/关节反馈只证明程序闭环，真实抬升与入框需要现场证据'}
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

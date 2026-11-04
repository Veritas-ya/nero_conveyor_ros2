"""Device dispatch for the existing BaselineSortingPolicy, without a second task FSM.

The policy and RobotController own routes, timing and task transitions. This
module only clocks their ticks, commits validated commands, and stops hardware
on faults. IdealFeedbackPlant is explicitly offline and never imports a driver.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .single_pick import PickRejected, RobotModel
from .nero_pick_hardware import FEEDBACK_MAX_AGE_S


def check_startup_gripper_force(state, settings, emit, *, phase):
    """Use the separate startup threshold without changing contact detection."""
    details = {
        'phase': phase, 'force_n': state['force_n'],
        'force_signed_n': state.get('force_signed_n'), 'width_m': state['width_m'],
        'gripper_stamp': state.get('gripper_stamp'),
        'gripper_feedback': state.get('gripper_feedback'),
        'threshold_n': settings.startup_gripper_force_threshold_n,
        'rejected': state['force_n'] >= settings.startup_gripper_force_threshold_n,
    }
    emit('startup_gripper_force_check', details)
    if details['rejected']:
        raise PickRejected(
            f"夹爪启动受力检查未通过：电机力绝对值{state['force_n']:.3f}N"
            f" ≥ 阈值{settings.startup_gripper_force_threshold_n:.3f}N（可能持有物体）",
            details=details)


class FullPolicyTaskFailure(PickRejected):
    """An explicit policy/budget outcome, rather than a device fault.

    Do not classify arbitrary PickRejected exceptions this way: that class is
    also used for inconsistent measured feedback and rejected live commands.
    """

    def __init__(self, message, *, failure_code, failure_reason=None, details=None):
        self.failure_reason = message if failure_reason is None else failure_reason
        self.failure_code = failure_code
        super().__init__(message, details={**(details or {}), 'failure_code': failure_code,
                                          'failure_reason': self.failure_reason})


class IdealFeedbackPlant:
    """Deterministic test feedback, NOT dynamics simulation or physical grasp proof."""

    is_offline_replay = True

    def __init__(self, state, settings, object_width_m, *, empty_grasp=False):
        self.model = RobotModel()
        self.settings = settings
        self.q = np.asarray(state['joints'], dtype=float).copy()
        self.width = float(state['width_m'])
        self.target_width = self.width
        self.force = 0.
        self.time = 0.
        self.object_width = float(np.clip(object_width_m, .004, settings.open_width_m-.005))
        self.empty_grasp = empty_grasp
        self.stopped = False
        self.commands = 0

    def snapshot(self):
        self.model.set_state(self.q, self.width)
        flange = self.model.flange()
        return {'joints': self.q.tolist(),
                'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist(),
                'width_m': self.width, 'force_n': abs(self.force), 'force_signed_n': self.force,
                'pose_stamp': self.time, 'joint_stamp': self.time,
                'gripper_stamp': self.time, 'ctrl_mode': 1}

    def snapshot_nowait(self):
        return self.snapshot()

    def advance(self, dt):
        self.time += dt
        gap = self.target_width-self.width
        self.width += float(np.clip(gap, -.15*dt, .15*dt))
        if not self.empty_grasp and self.target_width < self.object_width and self.width <= self.object_width:
            self.width = self.object_width
            self.force = -self.settings.force_n
        else:
            self.force = 0.

    def prepare_motion(self, speed, *, query_limits=False, requested_acceleration_limits=None):
        return {'offline': True, 'speed_percent': speed, 'hardware_connected': False,
                'requested_acceleration_limits_rad_s2': (
                    None if requested_acceleration_limits is None
                    else np.asarray(requested_acceleration_limits).tolist())}

    def move_joints(self, q):
        self.q = np.asarray(q, dtype=float).copy()
        self.commands += 1

    def grip(self, width, force):
        self.target_width = float(width)

    def stop(self):
        self.stopped = True

    def hold_current_position(self, state=None):
        state = self.snapshot() if state is None else state
        self._hold_joints = np.asarray(state['joints'], dtype=float).copy()
        return self.maintain_hold_position(state=state)

    def maintain_hold_position(self, state=None):
        self.move_joints(self._hold_joints)
        return {'hold_joints_rad': self._hold_joints.tolist(),
                'method': 'offline_position_hold', 'hardware_connected': False,
                'enable_state_modified': False, 'automatic_gripper_release': False}


def hold_position_until_exit(device, pause_control, emit, *, mode, control_hz=100,
                             clock=time.monotonic, sleep=time.sleep, stop_on_failure=True,
                             task_failure=None):
    """Hold one measured position until the user exits; never resume old actions.

    Position targets use the same device path as ordinary movement. The current
    task and gripper command stop advancing; this is not a firmware emergency
    stop or a promise of instantaneous physical braking. Fault handling remains
    the same as for an active device. Pause duration has no task time budget.
    """
    offline = mode == 'offline_ideal_feedback'
    if mode not in ('hardware', 'offline_ideal_feedback') or offline != bool(
            getattr(device, 'is_offline_replay', False)):
        raise ValueError('feedback device does not match requested hold mode')
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('hold control_hz must be positive')
    period = 1./control_hz
    attempted = False
    count = 0
    started = clock()
    try:
        state = device.snapshot()
        captured = np.asarray(state['joints'], dtype=float).copy()
        if task_failure is not None:
            # This feedback was captured before sending the first hold target.
            # Keep its original stamps; it is not proof of physical braking.
            emit('failure_hold_feedback', {**state,
                'feedback_phase': 'captured_before_first_hold_command'})
        attempted = True
        report = device.hold_current_position(state=state)
        count += 1
        pause_control.mark_holding()
        emit('motion_paused', {**report, 'hold_joints_rad': captured.tolist(),
            'feedback_mode': mode, 'automatic_gripper_release': False,
            'task_will_resume': False,
            'message': '已暂停任务，保持实测关节位置；按Q或再次Ctrl+C退出，之后重新运行抓取'})
        if task_failure is not None:
            emit('task_failed_holding', {**task_failure, **report,
                'hold_joints_rad': captured.tolist(),
                'hold_feedback_joint_stamp': state.get('joint_stamp'),
                'feedback_phase': 'captured_before_first_hold_command',
                'automatic_gripper_release': False, 'task_will_resume': False,
                'message': '本轮抓取失败，正在保持失败时的实测关节位置；按Q或再次Ctrl+C结束，重新抓取需重启脚本'})
        last_status = clock()
        max_error = 0.
        while not pause_control.exit_requested:
            # Keep servicing the SDK and the fixed target while the terminal
            # thread only records user input. Never recapture a drifting pose.
            sleep(period)
            state = device.snapshot()
            device.maintain_hold_position(state=state)
            count += 1
            error = float(np.max(np.abs(np.asarray(state['joints'])-captured)))
            max_error = max(max_error, error)
            now = clock()
            if now-last_status >= 2.:
                emit('hold_feedback', {'feedback_mode': mode, 'joints_rad': state['joints'],
                    'hold_joints_rad': captured.tolist(), 'max_joint_error_rad': error,
                    'hold_command_count': count, 'holding_elapsed_s': now-started,
                    'joint_stamp': state.get('joint_stamp'),
                    'pose_stamp': state.get('pose_stamp'),
                    'gripper_stamp': state.get('gripper_stamp'),
                    'feedback_phase': 'while_repeating_fixed_hold_target'})
                last_status = now
        emit('exit_holding', {'feedback_mode': mode, 'hold_joints_rad': captured.tolist(),
            'enable_state_modified': False, 'automatic_gripper_release': False,
            'message': '结束本轮，保留最后的位置目标与使能状态；不会继续原抓取轨迹'})
        return {'status': 'paused_exit_holding', 'feedback_mode': mode,
                'hardware_motion_commands_sent': not offline,
                'hold_joints_rad': captured.tolist(), 'hold_command_count': count,
                'holding_elapsed_s': clock()-started, 'max_hold_feedback_error_rad': max_error,
                'enable_state_modified': False, 'automatic_gripper_release': False,
                'task_resumed': False, 'physical_grasp_verified': False}
    except BaseException as fault:
        if attempted and stop_on_failure:
            stop_error = None
            try:
                device.stop()
            except Exception as error:
                stop_error = error
            try:
                if stop_error is not None:
                    emit('stop_request_failed', {'error': str(stop_error)})
                emit('stopped', {'automatic_gripper_release': False, 'feedback_mode': mode,
                                 'stop_requested': True, 'during_position_hold': True})
            except Exception as reporting_error:
                fault.details = {**(getattr(fault, 'details', None) or {}),
                                 'stop_reporting_error': str(reporting_error)}
                if stop_error is not None:
                    fault.details['stop_request_error'] = str(stop_error)
        raise


def run_full_policy(context, device, emit, *, mode,
                    max_cycle_s=300., deadline=None, clock=time.monotonic,
                    sleep=time.sleep, replay_realtime=False, pause_control=None,
                    execution_trace_dir=None):
    """Run unchanged control decisions with an in-memory execution ledger."""
    from .execution_trace import ExecutionTrace
    trace = ExecutionTrace(clock=clock, mode=mode)
    trace.target_identity = getattr(context, 'target_identity', None)
    context.execution_trace = trace
    traced_device = trace.wrap_device(device,
        stage=lambda: getattr(getattr(context.policy, 'stage', None), 'value', ''))
    try:
        result = _run_full_policy(context, traced_device, emit, mode=mode,
            max_cycle_s=max_cycle_s, deadline=deadline, clock=clock,
            sleep=sleep, replay_realtime=replay_realtime, pause_control=pause_control)
        trace.task_terminal = ('FAILED' if result.get('task_failed') else
            'COMPLETED' if getattr(context,'complete',False) else 'CANCELLED')
        context._task_outcome = trace.task_terminal
        return result
    except BaseException as error:
        trace.task_terminal = 'CANCELLED' if isinstance(error, KeyboardInterrupt) else 'FAILED'
        context._task_outcome = trace.task_terminal
        raise
    finally:
        trace.task_finished_monotonic = clock()
        # _run_full_policy joins the independent sender before returning or
        # propagating a fault. No compression or filesystem work runs at 100 Hz.
        if execution_trace_dir is not None:
            context.execution_trace_artifacts = trace.write(execution_trace_dir)


def _run_full_policy(context, device, emit, *, mode,
                    max_cycle_s=300., deadline=None, clock=time.monotonic,
                    sleep=time.sleep, replay_realtime=False, pause_control=None):
    """Run the shared policy; offline feedback can never be a live fallback."""
    offline = mode == 'offline_ideal_feedback'
    if mode not in ('hardware', 'offline_ideal_feedback'):
        raise ValueError('explicit hardware/offline_ideal_feedback mode required')
    if offline != bool(getattr(device, 'is_offline_replay', False)):
        raise ValueError('feedback device does not match requested execution mode')
    if replay_realtime and not offline:
        raise ValueError('replay_realtime requires offline ideal feedback')
    accelerated = offline and not replay_realtime
    if not np.isfinite(max_cycle_s) or max_cycle_s <= 0:
        raise ValueError('max_cycle_s must be finite and positive')
    cfg = context.settings
    period = 1./cfg.control_hz
    actuation_attempted = False
    initial = device.snapshot()
    if pause_control is not None and pause_control.pause_requested:
        result = hold_position_until_exit(device, pause_control, emit, mode=mode,
            control_hz=cfg.control_hz, clock=clock, sleep=sleep)
        return {**result, 'command_count': 0, 'stages': [], 'elapsed_policy_s': 0.}
    check_startup_gripper_force(initial, cfg, emit, phase='runner_initial')
    context.geometry.check_feedback(initial)
    if deadline is not None and clock() >= deadline:
        raise PickRejected('识别和场景准备已超过本次全周期预算；未发送控制指令')
    started = clock()
    trace = getattr(context, 'execution_trace', None)
    if trace is not None:
        trace.policy_origin = started
    set_clock_origin = getattr(context, 'set_policy_clock_origin', None)
    if set_clock_origin is not None:
        set_clock_origin(started)
    tick = 0
    next_tick = started
    last_send = None
    max_gap = 0.
    gap_sum = 0.
    gap_count = 0
    max_lateness = 0.
    max_compute = 0.
    late_ticks = 0
    feedback_refreshes = 0
    stages = []
    gripper_target = None
    previous_action_time = None
    commands_sent = 0
    finalize_motion = getattr(context.policy, 'finalize_pending_motion', None)
    deferred_motion = callable(finalize_motion)
    motion_finalizations = 0
    max_decision_age = 0.
    max_motion_age = 0.
    previous_compute = 0.
    previous_post_dispatch = 0.
    previous_gap = None
    high_stream = None
    stream_command_count = 0
    stream_review_count = 0
    stream_previous_review_s = 0.
    stream_handbacks = 0

    def consume_stream(snapshot, owner):
        """Only real sends enter the policy/controller history and log."""
        nonlocal commands_sent, tick, last_send, previous_gap, gap_sum, gap_count
        nonlocal max_gap, max_lateness, stream_command_count, late_ticks
        owner.sync_history(context, snapshot)
        for item in snapshot.events:
            if item.get('event') != 'joint_stream_command':
                continue
            # Sender events are detached successful-dispatch records; callbacks
            # and evidence capture execute here, never on the device thread.
            gap = item['gap_s']
            dispatch = item['dispatch_time']
            last_send = dispatch
            previous_gap = gap
            commands_sent += 1
            stream_command_count += 1
            tick += 1
            if gap is not None:
                max_gap = max(max_gap, gap)
                gap_sum += gap
                gap_count += 1
            lateness = item['dispatch_lateness_s']
            max_lateness = max(max_lateness, lateness)
            late_ticks += int(lateness > period)
            meta = item['metadata']
            emit('joint_command', {
                'time': item['dispatch_wall_time'], 'stage': 'pregrasp',
                'joints_rad': item['joint_target'], 'shared_time_parameterized': True,
                'trajectory_retiming': meta.get('trajectory_retiming', {}),
                'policy_time_s': item['sample_time']-started,
                'dispatch_timing': {
                    'execution': 'independent_certified_high_edge_stream',
                    'dispatch_policy_time_s': dispatch-started,
                    'scheduled_dispatch_policy_time_s': item['scheduled_dispatch_time']-started,
                    'gap_s': gap, 'control_dt_s': item['sample_dt_s'],
                    'lateness_s': lateness, 'compute_s': item['step_compute_s'],
                    'feedback_receive_s': item['feedback_receive_s'],
                    'feedback_sample_to_dispatch_s': dispatch-item['feedback_read_time'],
                    'command_history': meta.get('command_history'),
                    'geometry_rechecked_on_every_send': False,
                    'geometry_review_target_period_s': owner.policy_period,
                }})

    def stop_high_stream(*, raise_fault=True):
        nonlocal high_stream, next_tick, stream_handbacks
        if high_stream is None:
            return
        owner = high_stream
        snapshot = owner.stop()
        if trace is not None:
            trace.record_handoff(owner, snapshot, clock())
        high_stream = None  # Device ownership is returned only after join.
        try:
            consume_stream(snapshot, owner)
            stream_handbacks += 1
            next_tick = clock() if last_send is None else max(clock(), last_send+period)
            emit('high_joint_stream_handoff', {
                'reason': owner.reason, 'command_count': snapshot.command_count,
                'observed_dispatch_hz': (
                    snapshot.command_count / (snapshot.last_command['dispatch_time']
                        - owner.initial_dispatch_time)
                    if snapshot.last_command is not None
                    and hasattr(owner, 'initial_dispatch_time')
                    and snapshot.last_command['dispatch_time'] > owner.initial_dispatch_time else None),
                'joined': True, 'actual_history_restored': True})
        except BaseException as reporting_error:
            if raise_fault and snapshot.exception is not None:
                raise snapshot.exception from reporting_error
            raise
        if raise_fault and snapshot.exception is not None:
            raise snapshot.exception

    def dispatch_clock(observation, scheduled, dt):
        # This is a command deadline, not a camera/feedback timestamp. The
        # shared trajectory and ETA may use it without making old data fresh.
        if deferred_motion and not accelerated:
            observation['dispatch_clock'] = {
                'actual_interval_s': previous_gap,
                'planned_interval_s': dt,
                'compute_s': previous_compute+previous_post_dispatch,
                'scheduled_dispatch_policy_time_s': scheduled-started}
        prepare_snapshot = getattr(context, 'prepare_motion_snapshot', None)
        if callable(prepare_snapshot):
            prepare_snapshot(observation)

    def pause_requested():
        return pause_control is not None and pause_control.pause_requested

    def pause_and_finish():
        nonlocal actuation_attempted
        # The capture is fresh at the moment of pausing, not the position of
        # the old trajectory sample currently waiting to be dispatched.
        actuation_attempted = True
        stop_high_stream()
        elapsed_policy = tick*period if accelerated else clock()-started
        paused = hold_position_until_exit(device, pause_control, emit, mode=mode,
            control_hz=cfg.control_hz, clock=clock, sleep=sleep, stop_on_failure=False)
        return {**paused, 'command_count': commands_sent, 'stages': stages,
                'elapsed_policy_s': elapsed_policy, 'wall_elapsed_s': clock()-started,
                'clock_mode': 'accelerated' if accelerated else 'wall_clock'}

    def stop_after_fault(*, task_failure=None):
        stream_cleanup_error = None
        try:
            stop_high_stream(raise_fault=False)
        except BaseException as cleanup_error:
            stream_cleanup_error = cleanup_error
        if actuation_attempted:
            try:
                device.stop()
            except Exception as stop_error:
                emit('stop_request_failed', {'error': str(stop_error)})
        details = {} if task_failure is None else {
            **task_failure, 'during_position_hold': True}
        emit('stopped', {'automatic_gripper_release': False, 'feedback_mode': mode,
                         'stop_requested': actuation_attempted, **details})
        if stream_cleanup_error is not None:
            raise stream_cleanup_error

    try:
        # Setting speed or querying configuration does not start motion. Do
        # not latch an electronic emergency stop on a preparation failure.
        prepare_options = {'query_limits': True} if cfg.query_controller_limits else {}
        if hasattr(context, 'max_joint_acceleration'):
            prepare_options['requested_acceleration_limits'] = context.max_joint_acceleration
        emit('controller_motion_limits', device.prepare_motion(cfg.speed_percent, **prepare_options))
        # Firmware limit queries are preparation, not a missing servo tick.
        # The overall cycle deadline still includes this time.
        next_tick = clock()
        while True:
            if pause_requested():
                return pause_and_finish()
            # Prepare for the next sending slot immediately. Sleeping a full
            # period here would add all IK/validation cost to every cycle.
            if accelerated:
                now = tick*period
            else:
                now = clock()-started
            if now > max_cycle_s or (deadline is not None and clock() >= deadline):
                raise FullPolicyTaskFailure('完整策略达到本次运行时间预算，保持夹爪并停止',
                                            failure_code='cycle_budget_exhausted')
            tick_started = clock()
            if high_stream is not None:
                stream_snapshot = high_stream.snapshot()
                try:
                    consume_stream(stream_snapshot, high_stream)
                except BaseException as reporting_error:
                    if stream_snapshot.exception is not None:
                        raise stream_snapshot.exception from reporting_error
                    raise
                if stream_snapshot.exception is not None:
                    raise stream_snapshot.exception
                if not stream_snapshot.running:
                    stop_high_stream()
                    state = stream_snapshot.last_feedback
                    if state is None:
                        state = device.snapshot()
                elif stream_snapshot.last_feedback is None:
                    sleep(period)
                    continue
                else:
                    state = stream_snapshot.last_feedback
                if state is stream_snapshot.last_feedback:
                    # A completed sender may have waited for a slow geometry
                    # review too. Preserve that cache residence in the original
                    # freshness check, without changing the packet timestamp.
                    timing = dict(state.get('feedback_timing', {}))
                    timing['oldest_packet_age_s'] = timing.get('oldest_packet_age_s', 0.) + max(
                        0., clock()-stream_snapshot.last_feedback_time)
                    state = {**state, 'feedback_timing': timing}
            else:
                state = device.snapshot()
            if pause_requested():
                return pause_and_finish()
            state_sampled_at = clock()
            feedback_receive_s = state_sampled_at-tick_started
            if not accelerated:
                # A receive retry may have waited. Predict the moving target
                # using the time of the returned sample, not before the wait.
                now = state_sampled_at-started
                if (getattr(cfg, 'planning_mode', 'legacy') == 'full_preplanned'
                        and not getattr(context.policy, '_full_pick_finished', False)):
                    # Sample the SAME absolute curve for a future sending
                    # slot, leaving the last measured cycle's computation
                    # time available before it. Never shift the certificate.
                    # An overrun still fails the final dispatch-time check.
                    reserve = max(period, previous_compute+previous_post_dispatch)
                    next_tick = max(next_tick, state_sampled_at+reserve)
                    bundle = getattr(context.policy, '_full_pick_bundle', None)
                    if bundle is not None and now < bundle.end:
                        next_tick = min(next_tick, started+bundle.end)
            if tick == 0:
                check_startup_gripper_force(state, cfg, emit, phase='first_tick')
                if np.max(np.abs(np.asarray(state['joints'])-initial['joints'])) > .003:
                    raise PickRejected('准备期间起点发生变化，请重新采集场景')
            stream_velocity = state.get('stream_joint_velocity_rad_s')
            observation = (context.observe(state, now) if stream_velocity is None else
                           context.observe(state, now, joint_velocity=stream_velocity))
            feedback_record = dict(state)
            if 'feedback_diagnostics' in observation:
                feedback_record['feedback_diagnostics'] = observation['feedback_diagnostics']
            emit('feedback', feedback_record)
            target_state_record = getattr(context, 'target_state_record', None)
            target_record = None
            if target_state_record is not None:
                target_record = target_state_record(now)
                if target_record is not None:
                    emit('target_state', target_record)
            if high_stream is not None:
                control_dt = period if previous_gap is None else previous_gap
            elif deferred_motion and not accelerated:
                control_dt = period if last_send is None else max(
                    next_tick, state_sampled_at)-last_send
            elif last_send is not None and not accelerated:
                # Approve the sample for its scheduled dispatch slot. Do the
                # work before that slot, then wait to send, so stable compute
                # time does not subtract from the intended 10 ms motion step.
                # Late compute shifts the next slot, never a catch-up burst.
                control_dt = max(max(next_tick, state_sampled_at)-last_send, 1e-6)
            else:
                control_dt = period if previous_action_time is None else max(now-previous_action_time, 1e-6)
            observation['control_dt'] = control_dt
            dispatch_clock(observation, max(next_tick, state_sampled_at), control_dt)
            decision_time = now
            decision_control_dt = control_dt
            motion_finalize_s = None
            if deferred_motion:
                observation['_defer_arm_motion'] = True
            if high_stream is not None:
                # Preserve the original overlap: a completed next-high search
                # must be consumed while this edge is moving, not after it
                # settles (late receipt would push its capture downstream).
                poll_approach = getattr(context.policy, 'poll_pregrasp_approach_result', None)
                if callable(poll_approach):
                    poll_approach(observation)
            action = context.policy.act(observation)
            if pause_requested():
                return pause_and_finish()
            if high_stream is not None and not high_stream.continues(context, observation, action):
                stop_high_stream()
                # A stage/edge handoff consumes the pending intent using fresh
                # measured feedback and the final actually-sent q/v history.
                state = device.snapshot()
                state_sampled_at = clock()
                now = state_sampled_at-started
            if (deferred_motion and not context.complete and not context.failed_reason
                    and context.policy.stage.value != 'recovering' and high_stream is None):
                # act() has advanced task bookkeeping once, but has not
                # solved/integrated an arm request or committed a jaw-close
                # transition. Refresh expired input before the unique motion
                # finalization; no policy.act/controller replay is involved.
                if not offline and not accelerated:
                    remaining = next_tick-clock()
                    if (remaining > 0. and
                            state.get('feedback_timing', {}).get('oldest_packet_age_s', 0.)
                            + next_tick-state_sampled_at > FEEDBACK_MAX_AGE_S):
                        # A forecast inherited from a slow prior computation
                        # can create a long slot wait. Put that wait BEFORE
                        # the one solve and let the existing receive refresh
                        # below obtain current feedback. Never deliberately
                        # wait past its original validity after solving.
                        sleep(remaining)
                        if pause_requested():
                            return pause_and_finish()
                if not offline:
                    while (state.get('feedback_timing', {}).get('oldest_packet_age_s', 0.)
                           + clock()-state_sampled_at > FEEDBACK_MAX_AGE_S):
                        if clock()-started > max_cycle_s or (deadline is not None and clock() >= deadline):
                            raise FullPolicyTaskFailure('完整策略达到本次运行时间预算，保持夹爪并停止',
                                                        failure_code='cycle_budget_exhausted')
                        receive_started = clock()
                        state = device.snapshot()
                        if pause_requested():
                            return pause_and_finish()
                        state_sampled_at = clock()
                        feedback_receive_s += state_sampled_at-receive_started
                        feedback_refreshes += 1
                        if tick == 0:
                            context.geometry.check_feedback(state)
                            check_startup_gripper_force(state, cfg, emit, phase='first_tick_refresh')
                            if np.max(np.abs(np.asarray(state['joints'])-initial['joints'])) > .003:
                                raise PickRejected('准备期间起点发生变化，请重新采集场景')
                if not accelerated:
                    now = clock()-started
                    # Both the profile and limiter spend the SAME scheduled
                    # send interval. If computation finishes early we wait
                    # for that slot; if it overruns we send once and account
                    # for the actual spacing afterwards. No catch-up burst.
                    next_tick = max(next_tick, clock())
                    control_dt = period if last_send is None else max(
                        period, next_tick-last_send)
                observation = context.observe(state, now)
                observation['control_dt'] = control_dt
                dispatch_clock(observation, next_tick, control_dt)
                feedback_record = dict(state)
                if 'feedback_diagnostics' in observation:
                    feedback_record['feedback_diagnostics'] = observation['feedback_diagnostics']
                emit('feedback', feedback_record)
                if target_state_record is not None:
                    target_record = target_state_record(now)
                    if target_record is not None:
                        emit('target_state', target_record)
                finalize_started = clock()
                action = finalize_motion(observation)
                motion_finalize_s = clock()-finalize_started
                motion_finalizations += 1
                if pause_requested():
                    return pause_and_finish()
            motion_reference_time = now
            events = context.policy.drain_events()
            for event in events:
                emit('policy_event', asdict(event))
            stage = context.policy.stage.value
            if not stages or stages[-1] != stage:
                stages.append(stage)
                emit('policy_stage', {'name': stage})
            if context.failed_reason or stage == 'recovering':
                reason = str(context.failed_reason or getattr(context.policy, '_full_pick_abort_reason', None)
                             or context.policy.get_info())
                raise FullPolicyTaskFailure('完整策略停止：'+reason,
                                            failure_code='policy_failed', failure_reason=reason)
            prepared_high = getattr(context.policy, 'prepared_high_state', None)
            if prepared_high is not None:
                stop_high_stream()
                emit('prepared_high_ready', {**prepared_high,
                    'message': '已到工作高位；保持当前位置，退出后重新拍照运行抓取',
                    'feedback_mode': mode, 'physical_grasp_verified': False})
                if offline:
                    held = {}
                else:
                    from .pick_pause import PauseControl
                    control = pause_control or PauseControl()
                    control.request_pause()
                    if pause_control is None:
                        control.request_exit()
                    actuation_attempted = True
                    held = hold_position_until_exit(device, control, emit, mode=mode,
                        control_hz=cfg.control_hz, clock=clock, sleep=sleep, stop_on_failure=False)
                return {**held, 'status': 'prepared_high_only', 'prepared_high': prepared_high,
                        'feedback_mode': mode, 'physical_grasp_verified': False,
                        'hardware_motion_commands_sent': not offline and actuation_attempted,
                        'command_count': commands_sent, 'stages': stages,
                        'elapsed_policy_s': now, 'wall_elapsed_s': clock()-started}
            if context.complete:
                stop_high_stream()
                return {'status': ('offline_full_policy_completed' if offline else 'cycle_completed_from_feedback'),
                        'algorithm': 'BaselineSortingPolicy + MuJoCoGlobalPlanningEngine + RobotController',
                        'feedback_mode': mode, 'hardware_motion_commands_sent': not offline and tick > 0,
                        'clock_mode': 'accelerated' if accelerated else 'wall_clock',
                        'elapsed_policy_s': now, 'wall_elapsed_s': clock()-started,
                        'stages': stages, 'command_count': tick, 'max_dispatch_gap_s': max_gap,
                        'control_timing': {'target_period_s': period,
                                           'max_dispatch_lateness_s': max_lateness,
                                           'max_tick_compute_s': max_compute,
                                           'observed_dispatch_hz': (gap_count/gap_sum if gap_sum > 0. else None),
                                           'mean_dispatch_gap_s': (gap_sum/gap_count if gap_count else None),
                                           'late_tick_count': late_ticks,
                                           'feedback_refresh_count': feedback_refreshes,
                                           'deferred_motion_finalization': deferred_motion,
                                           'motion_finalization_count': motion_finalizations,
                                           'high_stream_command_count': stream_command_count,
                                           'high_stream_geometry_review_count': stream_review_count,
                                           'high_stream_handoff_count': stream_handbacks,
                                           'max_decision_to_dispatch_s': max_decision_age,
                                           'max_motion_reference_to_dispatch_s': max_motion_age,
                                           'dispatch_timing_is_diagnostic': True},
                        'physical_grasp_verified': False}
            if high_stream is not None:
                # The policy still observes, updates ETA/reservations and starts
                # its original planner. Its pending arm solve is deferred until
                # this immutable edge is handed back. Heavy geometry is reviewed
                # here; it is explicitly NOT run by the 100 Hz sender.
                horizon = max(high_stream.policy_period, stream_previous_review_s)
                observation['geometry_validation_horizon_s'] = horizon
                prepare_snapshot = getattr(context, 'prepare_motion_snapshot', None)
                if callable(prepare_snapshot):
                    prepare_snapshot(observation)
                candidate = high_stream.geometry_target(state, now, horizon)
                if candidate is None:
                    candidate = context.robot_controller.commanded_joint_positions
                review_started = clock()
                actual_target = context.robot_controller.commanded_joint_positions
                actual_valid = context.validate_command(state, SimpleNamespace(joint_target=actual_target))
                future_valid = actual_valid and (
                    np.array_equal(actual_target, candidate)
                    or context.validate_command(state, SimpleNamespace(joint_target=candidate)))
                if not future_valid:
                    rejection = dict(getattr(context, 'motion_validation_rejection', {}))
                    stop_high_stream()
                    emit('motion_geometry_rejected', {
                        'stage': stage, 'sent': not actual_valid,
                        'joint_target_rad': (candidate if actual_valid else actual_target).tolist(),
                        'feedback': state, 'rejection': rejection,
                        'execution': 'high_stream_prospective_review'})
                    raise FullPolicyTaskFailure('高位轨迹的实时几何复核未通过',
                        failure_code='motion_geometry_rejected', details={'rejection': rejection})
                stream_review_count += 1
                stream_previous_review_s = max(high_stream.policy_period, clock()-tick_started)
                emit('high_joint_stream_review', {
                    'policy_time_s': now, 'review_compute_s': clock()-review_started,
                    'geometry_validation_horizon_s': horizon,
                    'feedback_joint_stamp': state.get('joint_stamp'),
                    'actual_sent_joint_target': context.robot_controller.commanded_joint_positions.tolist(),
                    'reviewed_future_joint_target': candidate.tolist(),
                    'full_geometry_per_send': False,
                })
                remaining = max(0., high_stream.policy_period-(clock()-tick_started))
                wait_stream = getattr(high_stream, 'wait', None)
                if callable(wait_stream):
                    wait_stream(remaining)
                else:
                    sleep(remaining)
                continue
            closure_candidate = '_dispatch_closure_intent' in action
            controller_action = action
            if closure_candidate:
                # The first-close jaw decision is committed after all slow
                # motion validation. Advance the arm limiter once now, and
                # update the jaw mirror once with its final selected action.
                controller_action = {key: value for key, value in action.items() if key != 'gripper'}
            previous_command_target = getattr(context.robot_controller, 'commanded_joint_positions', None)
            previous_command_velocity = getattr(context.robot_controller, 'commanded_joint_velocities', None)
            command = context.robot_controller.apply_action(controller_action, dt=control_dt)
            previous_action_time = now
            if action.get('arm_time_parameterized') is True and not command.time_parameterized:
                raise PickRejected('共享控制器未接受已计时轨迹样本，停止而不另行改写路径')
            width = float(context.robot_controller.gripper.target_open_fraction)*cfg.open_width_m
            if not np.isfinite(width) or not 0 <= width <= cfg.open_width_m:
                raise PickRejected('完整策略夹爪指令无效')
            grip_changed = gripper_target is None or abs(width-gripper_target) > .0001
            def validate_dispatch_state(measured):
                if not context.validate_command(measured, command):
                    # A rejected output never gets a joint_command event.
                    # Preserve it separately so offline geometry checks can
                    # inspect the actual candidate instead of the last sent q.
                    rejection = getattr(context, 'motion_validation_rejection', {})
                    emit('motion_geometry_rejected', {
                        'stage': stage, 'policy_time_s': clock()-started,
                        'sent': False, 'joint_target_rad': command.joint_target.tolist(),
                        'feedback': measured,
                        'requested_gripper_width_m': width,
                        'validation_gripper_width_m': max(float(measured['width_m']), width),
                        'rejection': rejection,
                    })
                    detail = rejection.get('message') or rejection.get('reason')
                    raise FullPolicyTaskFailure('完整策略实际发送路径未通过共享几何约束'
                                                + (('：' + detail) if detail else ''),
                                                failure_code='motion_geometry_rejected',
                                                details={'rejection': rejection})
                validate_gripper = getattr(context, 'validate_gripper_command', None)
                if grip_changed and validate_gripper is not None:
                    end_state = {**measured, 'joints': command.joint_target.tolist()}
                    start_valid = validate_gripper(measured, width)
                    same_joints = np.array_equal(np.asarray(measured['joints']), command.joint_target)
                    if not start_valid or (not same_joints and not validate_gripper(end_state, width)):
                        rejection = getattr(context, 'motion_validation_rejection', {})
                        endpoint = 'measured' if not start_valid else 'commanded'
                        opening = {
                            'endpoint': endpoint,
                            'measured_gripper_width_m': float(measured['width_m']),
                            'requested_gripper_width_m': width,
                            'measured_joints_rad': list(measured['joints']),
                            'commanded_joints_rad': command.joint_target.tolist(),
                        }
                        emit('gripper_geometry_rejected', {
                            'stage': stage, 'policy_time_s': clock()-started, 'sent': False,
                            **opening, 'rejection': rejection})
                        location = '实测机械臂姿势' if not start_valid else '待发机械臂姿势'
                        message = (f'完整策略夹爪开合路径未通过几何约束：{location}，'
                                   f'开口 {float(measured["width_m"])*1000:.1f}'
                                   f'→{width*1000:.1f} mm')
                        sampled_width = rejection.get('sampled_width_m')
                        if sampled_width is not None:
                            message += f'，检查到 {float(sampled_width)*1000:.1f} mm 时拒绝'
                        detail = rejection.get('message') or rejection.get('reason')
                        if detail:
                            message += f'；{detail}'
                        raise FullPolicyTaskFailure(message,
                                                    failure_code='gripper_geometry_rejected',
                                                    details={'rejection': rejection, **opening})
            validate_dispatch_state(state)
            closure_sweep_ready = False
            closure_preview_ready = False
            closure_preview_feedback = False
            closure_commit_feedback = False

            def dispatch_observation():
                """Read current cached feedback, never start an SDK retry wait."""
                peek = getattr(device, 'snapshot_nowait', None)
                if peek is None:
                    return None
                current = peek()
                if current is None:
                    return None
                sampled_at = clock()
                sample_time = tick*period if accelerated else sampled_at-started
                return context.observe(current, sample_time), current, sampled_at

            if closure_candidate:
                sample = dispatch_observation()
                if pause_requested():
                    return pause_and_finish()
                if sample is not None:
                    closure_preview_feedback = True
                    preview_observation, preview_state, _ = sample
                    close_status = context.policy.dispatch_closure_status(action, preview_observation)
                    if close_status['ready']:
                        closure_preview_ready = True
            prepared_at = clock()
            # For the shared deferred-motion path this is the last compute
            # boundary, not the start of another blocking SDK receive. IK and
            # validation still take wall time; log that age below and predict
            # again on the next cycle. The pre-finalization snapshot retains
            # the adapter's original feedback/fault checks. Compatibility
            # policies without the intent API retain their old dispatch path.
            slot_wait_s = 0.
            if not accelerated:
                remaining = next_tick-clock()
                if remaining > 0:
                    wait_started = clock()
                    sleep(remaining)
                    slot_wait_s = clock()-wait_started
            if pause_requested():
                return pause_and_finish()
            if not offline and not deferred_motion:
                # Reuse the adapter's existing freshness contract. A slow
                # computation is not itself a fault: refresh only if its
                # input has expired, then check the actual current edge.
                while (state.get('feedback_timing', {}).get('oldest_packet_age_s', 0.)
                       + clock()-state_sampled_at > FEEDBACK_MAX_AGE_S):
                    if clock()-started > max_cycle_s or (deadline is not None and clock() >= deadline):
                        raise FullPolicyTaskFailure('完整策略达到本次运行时间预算，保持夹爪并停止',
                                                    failure_code='cycle_budget_exhausted')
                    state = device.snapshot()
                    if pause_requested():
                        return pause_and_finish()
                    state_sampled_at = clock()
                    feedback_refreshes += 1
                    if tick == 0:
                        context.geometry.check_feedback(state)
                        check_startup_gripper_force(state, cfg, emit, phase='first_dispatch_refresh')
                        if np.max(np.abs(np.asarray(state['joints'])-initial['joints'])) > .003:
                            raise PickRejected('准备期间起点发生变化，请重新采集场景')
                    refreshed = context.observe(state, clock()-started)
                    feedback_record = dict(state)
                    if 'feedback_diagnostics' in refreshed:
                        feedback_record['feedback_diagnostics'] = refreshed['feedback_diagnostics']
                    emit('feedback', feedback_record)
                    validate_dispatch_state(state)
                    prepared_at = clock()
                    if pause_requested():
                        return pause_and_finish()
            if not accelerated and (clock()-started > max_cycle_s
                                    or (deadline is not None and clock() >= deadline)):
                raise FullPolicyTaskFailure('完整策略达到本次运行时间预算，保持夹爪并停止',
                                            failure_code='cycle_budget_exhausted')
            dispatch_timing = None
            # A failed send may have partially reached the device. From this
            # boundary onward any fault must stop without opening the jaw.
            actuation_attempted = True
            if pause_requested():
                return pause_and_finish()
            joint_target_list = command.joint_target.tolist()
            if trace is not None:
                trace.prepare_command(context.policy, policy_time_s=motion_reference_time,
                    candidate=getattr(command, 'requested_arm', action.get('arm')),
                    stage=stage)
            dispatch = clock()
            if getattr(cfg, 'planning_mode', 'legacy') == 'full_preplanned':
                failure = context.policy.full_pick_dispatch_reason(
                    observation, command.joint_target, now if accelerated else dispatch-started)
                bundle = getattr(context.policy, '_full_pick_bundle', None)
                if (failure is None and bundle is not None
                        and not getattr(context.policy, '_full_pick_finished', False)
                        and np.max(np.abs(command.joint_target-np.asarray(action['arm']))) > 1e-7):
                    failure = 'controller_retimer_changed_certified_full_reference' 
                if failure is not None:
                    context.policy._full_pick_abort(observation, failure)
                    raise FullPolicyTaskFailure('完整预规划轨迹停止：'+failure,
                        failure_code='full_pick_dispatch_invalid', failure_reason=failure)
            device.move_joints(joint_target_list)
            full_sent=getattr(context.policy,'full_pick_command_sent',None)
            if callable(full_sent):full_sent(tick*period if accelerated else dispatch-started)
            commands_sent += 1
            if not accelerated:
                lateness = max(0., dispatch-next_tick)
                compute = prepared_at-tick_started
                gap = None if last_send is None else dispatch-last_send
                previous_gap = gap
                decision_age = max(0., dispatch-started-decision_time)
                motion_age = max(0., dispatch-started-motion_reference_time)
                max_decision_age = max(max_decision_age, decision_age)
                max_motion_age = max(max_motion_age, motion_age)
                max_lateness = max(max_lateness, lateness)
                max_compute = max(max_compute, compute)
                late_ticks += int(lateness > period)
                if gap is not None:
                    max_gap = max(max_gap, gap)
                    gap_sum += gap
                    gap_count += 1
                last_send = dispatch
                # Host scheduling jitter is diagnostic, not a controller
                # fault. Policy and controller share the available control_dt;
                # never fast-forward a delayed loop with a burst of commands.
                dispatch_timing = {'lateness_s': lateness, 'gap_s': gap,
                                   'compute_s': compute, 'control_dt_s': control_dt,
                                   'scheduled_dispatch_policy_time_s': next_tick-started,
                                   'dispatch_overrun_s': max(0., dispatch-next_tick),
                                   'slot_wait_s': slot_wait_s,
                                   'previous_prepare_compute_s': previous_compute,
                                   'previous_post_dispatch_work_s': previous_post_dispatch,
                                   'compute_excluding_feedback_receive_s': max(0., compute-feedback_receive_s),
                                   'decision_control_dt_s': decision_control_dt,
                                   'decision_policy_time_s': decision_time,
                                   'motion_reference_policy_time_s': motion_reference_time,
                                   'dispatch_policy_time_s': dispatch-started,
                                   'decision_to_dispatch_s': decision_age,
                                   'decision_to_send_s': decision_age,
                                   'motion_reference_to_dispatch_s': motion_age,
                                   'motion_finalize_s': motion_finalize_s,
                                   'geometry_validation': dict(getattr(
                                       context, 'geometry_validation_stats', {})),
                                   'feedback_sample_to_dispatch_s': dispatch-state_sampled_at,
                                   'feedback_receive_s': feedback_receive_s,
                                   'feedback_oldest_packet_age_at_dispatch_s': (
                                       state.get('feedback_timing', {}).get('oldest_packet_age_s', 0.)
                                       + dispatch-state_sampled_at),
                                   'target_observed_policy_time_s': (
                                       target_record.get('observed_policy_time_s')
                                       if target_record is not None else None),
                                   'target_observation_age_at_dispatch_s': (
                                       max(0., dispatch-started-float(target_record['observed_policy_time_s']))
                                       if target_record is not None
                                       and target_record.get('observed_policy_time_s') is not None else None),
                                   'motion_finalized_once': deferred_motion,
                                   'post_motion_sdk_wait': False if deferred_motion else None,
                                   'compute_latency_is_measured_not_zero': True}
            acknowledge = getattr(context.robot_controller, 'acknowledge_dispatch', None)
            if (deferred_motion and not accelerated and callable(acknowledge)
                    and gap is not None):
                dispatch_timing['command_history'] = acknowledge(command,
                    actual_dt=gap, planned_dt=control_dt,
                    previous_target=previous_command_target,
                    previous_velocity=previous_command_velocity)
            emit('joint_command', {'stage': stage, 'joints_rad': command.joint_target.tolist(),
                                  'shared_time_parameterized': command.time_parameterized,
                                  'trajectory_retiming': dict(getattr(
                                      context.policy, '_safe_edge_retime_diagnostics', {})),
                                  'policy_time_s': now, 'dispatch_timing': dispatch_timing})
            if pause_requested():
                return pause_and_finish()
            first_close_gate_time = None
            first_close_gate_wall_time = None
            first_close_committed = False
            closed_sweep_s = None
            gate_feedback_age_s = None
            if closure_candidate:
                if closure_preview_ready:
                    sample = dispatch_observation()
                    if pause_requested():
                        return pause_and_finish()
                    if sample is not None:
                        closure_commit_feedback = True
                        closing_observation, closing_state, closing_sample_time = sample
                        status = context.policy.dispatch_closure_status(action, closing_observation)
                        if status['ready']:
                            candidate_width = .5*(float(status['gripper'])+1.)*cfg.open_width_m
                            validate_gripper = getattr(context, 'validate_gripper_command', None)
                            end_state = {**closing_state, 'joints': command.joint_target.tolist()}
                            sweep_started = clock()
                            try:
                                closure_sweep_ready = bool(validate_gripper is None or (
                                    validate_gripper(closing_state, candidate_width)
                                    and (np.array_equal(np.asarray(closing_state['joints']), command.joint_target)
                                         or validate_gripper(end_state, candidate_width))))
                            except PickRejected:
                                closure_sweep_ready = False
                            closed_sweep_s = clock()-sweep_started
                            if pause_requested():
                                return pause_and_finish()
                            if closure_sweep_ready:
                                # Qualify on the same feedback version whose
                                # closed-jaw sweep just passed. Update only
                                # prediction time after its computation; do not
                                # read another q and invalidate that geometry.
                                gate_wall_reference = clock()
                                gate_time = tick*period if accelerated else gate_wall_reference-started
                                closing_observation = context.observe(closing_state, gate_time)
                                if pause_requested():
                                    return pause_and_finish()
                                gate_feedback_age_s = (
                                    closing_state.get('feedback_timing', {}).get('oldest_packet_age_s', 0.)
                                    + clock()-closing_sample_time)
                                first_close_committed = context.policy.commit_dispatch_closure(
                                    action, closing_observation)
                        if first_close_committed:
                            first_close_gate_time = float(closing_observation['timestamp'])
                            first_close_gate_wall_time = gate_wall_reference
                            stage = context.policy.stage.value
                action.pop('_dispatch_closure_intent', None)
                context.robot_controller.gripper.command(float(action['gripper']))
                width = float(context.robot_controller.gripper.target_open_fraction)*cfg.open_width_m
                grip_changed = gripper_target is None or abs(width-gripper_target) > .0001
            if grip_changed:
                if pause_requested():
                    return pause_and_finish()
                closing_or_carrying = (stage in ('closing', 'lifting', 'transferring')
                                       or width < state['width_m']-.001)
                jaw_send_started = clock()
                device.grip(width, cfg.force_n if closing_or_carrying else 1.)
                gripper_target = width
                if closing_or_carrying:
                    close_sent=getattr(context.policy,'full_pick_closure_sent',None)
                    if callable(close_sent):close_sent(tick*period if accelerated else jaw_send_started-started)
                emit('gripper_command', {'width_m': width,
                    'first_close_committed_at_dispatch': first_close_committed,
                    'gate_policy_time_s': first_close_gate_time,
                    'jaw_send_policy_time_s': (tick*period if accelerated else jaw_send_started-started),
                    'gate_to_jaw_send_s': (None if first_close_gate_wall_time is None else
                        max(0., jaw_send_started-first_close_gate_wall_time)),
                    'gate_feedback_age_s': gate_feedback_age_s,
                    'closed_sweep_compute_s': closed_sweep_s,
                    'jaw_send_call_s': clock()-jaw_send_started})
            if first_close_committed:
                # Log after the physical send boundary, so event serialization
                # cannot add delay between final admission and the jaw request.
                for event in context.policy.drain_events():
                    emit('policy_event', asdict(event))
                if not stages or stages[-1] != stage:
                    stages.append(stage)
                    emit('policy_stage', {'name': stage})
            if closure_candidate:
                emit('closure_dispatch', {
                    'preview_feedback_available': closure_preview_feedback,
                    'preview_original_predicate_ready': closure_preview_ready,
                    'prospective_jaw_sweep_ready': closure_sweep_ready,
                    'commit_feedback_available': closure_commit_feedback,
                    'first_close_committed': first_close_committed,
                    'gate_policy_time_s': first_close_gate_time,
                    'gate_feedback_age_s': gate_feedback_age_s,
                    'closed_sweep_compute_s': closed_sweep_s,
                    'blocking_sdk_receive_performed': False})
            tick += 1
            if offline:
                device.advance(control_dt)
                # Planning is asynchronous in the shared implementation. Let it
                # finish in wall time without turning a fast replay into a
                # fictitious hundred-second physical planner timeout.
                workers = getattr(context, 'workers', getattr(context, '_workers', ()))
                while accelerated and any(worker.busy for worker in workers):
                    if pause_requested():
                        return pause_and_finish()
                    if clock()-started > max_cycle_s:
                        raise FullPolicyTaskFailure('离线规划超过墙钟时间预算',
                                                    failure_code='offline_planning_budget_exhausted')
                    sleep(.005)
            if not accelerated:
                # Forecast the next complete cycle's work, including the
                # send/logging tail that precedes its preparation. Exclude
                # the scheduled wait so a spike cannot ratchet the loop to
                # that rate forever. This is an estimate, not a real-time
                # bound; an overrun is accounted from actual send timestamps.
                previous_compute = max(0., prepared_at-tick_started)
                previous_post_dispatch = max(0., clock()-last_send)
                next_tick = last_send + (max(period, previous_compute+previous_post_dispatch)
                                         if deferred_motion else period)
                if deferred_motion and not closure_candidate:
                    from .high_joint_stream import HighJointStream, can_stream_high_edge
                    if getattr(cfg, 'planning_mode', 'legacy') != 'full_preplanned' and can_stream_high_edge(context, command):
                        high_stream = HighJointStream(context, device, origin=started,
                            last_send=last_send, clock=clock, sleep=sleep,
                            pause_requested=pause_requested)
                        if trace is not None:
                            trace.instrument_stream(high_stream)
                        high_stream.start()
                        emit('high_joint_stream_started', {
                            'send_target_hz': cfg.control_hz, 'policy_target_hz': 50.,
                            'trajectory_duration_s': high_stream.trajectory.duration,
                            'full_geometry_per_send': False,
                            'scope': 'certified_pregrasp_joint_edge_only'})
    except FullPolicyTaskFailure as failure:
        try:
            stop_high_stream()
        except BaseException:
            stop_after_fault()
            raise
        failure_record = {**failure.details, 'error': str(failure), 'task_failed': True}
        failure_event = {**failure_record, 'feedback_mode': mode,
            'automatic_gripper_release': False, 'stop_requested': False,
            'holding_loop_active': bool(actuation_attempted and pause_control is not None)}
        if not actuation_attempted:
            # No motion request has reached the device; a failed recognition or
            # first policy decision does not require an electronic stop.
            emit('task_failed', failure_event)
            raise
        try:
            # Include reporting and hold setup in the same fault boundary as
            # the hold itself. An unexpected logging/setup failure must not
            # leave the preceding motion target active without fault handling.
            emit('task_failed', failure_event)
            elapsed_policy = tick*period if accelerated else clock()-started
            control = pause_control
            if control is None:
                # Library/offline callers have no operator input loop. Replace
                # the previous target once and record that shorter lifetime.
                from .pick_pause import PauseControl
                control = PauseControl()
                control.request_exit()
            held = hold_position_until_exit(device, control, emit, mode=mode,
                control_hz=cfg.control_hz, clock=clock, sleep=sleep,
                stop_on_failure=False, task_failure=failure_record)
        except BaseException as hold_error:
            # The ordinary task outcome never suppresses a real receive/send
            # failure encountered while establishing or maintaining the hold.
            hold_error.details = {**(getattr(hold_error, 'details', None) or {}),
                                  **failure_record, 'hold_fault': str(hold_error),
                                  'hold_fault_type': type(hold_error).__name__}
            # Stop before emitting any further diagnostic. A faulty callback
            # must neither prevent the stop nor replace the original fault.
            try:
                stop_after_fault(task_failure=failure_record)
            except Exception as reporting_error:
                hold_error.details['stop_reporting_error'] = str(reporting_error)
            try:
                emit('task_failure_hold_fault', hold_error.details)
            except Exception as reporting_error:
                hold_error.details['hold_fault_reporting_error'] = str(reporting_error)
            raise hold_error from failure
        return {**held, **failure_record, 'status': 'task_failed_exit_holding',
                'command_count': commands_sent, 'stages': stages,
                'elapsed_policy_s': elapsed_policy, 'wall_elapsed_s': clock()-started,
                'clock_mode': 'accelerated' if accelerated else 'wall_clock',
                'operator_hold_loop_used': pause_control is not None,
                'holding_loop_active': False,
                'control_timing': {'target_period_s': period,
                    'high_stream_command_count': stream_command_count,
                    'high_stream_geometry_review_count': stream_review_count,
                    'high_stream_handoff_count': stream_handbacks,
                    'observed_dispatch_hz': gap_count/gap_sum if gap_sum > 0. else None,
                    'max_dispatch_gap_s': max_gap},
                'exit_origin': ('operator' if pause_control is not None
                                else 'no_pause_control_single_hold')}
    except BaseException as fault:
        try:
            stop_after_fault()
        except Exception as reporting_error:
            fault.details = {**(getattr(fault, 'details', None) or {}),
                             'stop_reporting_error': str(reporting_error)}
        raise

"""Dispatch-boundary tests; fake devices never construct or import SDK drivers."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.full_policy_runner import FullPolicyTaskFailure, run_full_policy
from mujoco_conveyor_sorting.runtime.nero_pick_hardware import FeedbackError
from mujoco_conveyor_sorting.runtime.nero_policy_context import _JawCommandMirror
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, PickSettings
from mujoco_conveyor_sorting.runtime.pick_pause import PauseControl


class Clock:
    def __init__(self):
        self.time = 10.

    def now(self):
        return self.time

    def sleep(self, duration):
        self.time += duration


class Device:
    def __init__(self, clock, *, offline=False, force=0., snapshot_failure_at=None):
        self.clock = clock
        self.is_offline_replay = offline
        self.force = force
        self.width = .1
        self.q = np.zeros(7)
        self.calls = []
        self.snapshots = 0
        self.snapshot_failure_at = snapshot_failure_at

    def snapshot(self):
        self.snapshots += 1
        if self.snapshots == self.snapshot_failure_at:
            raise FeedbackError("关节反馈持续未更新")
        return {'joints': self.q.tolist(), 'pose': [0., 0., .2, 0., 0., 0.],
                'width_m': self.width, 'force_n': self.force,
                'joint_stamp': self.clock.now(), 'pose_stamp': self.clock.now(),
                'gripper_stamp': self.clock.now(),
                'feedback_timing': {'oldest_packet_age_s': 0.}}

    def prepare_motion(self, speed):
        self.calls.append(('prepare', speed))
        return {'speed_percent': speed}

    def snapshot_nowait(self):
        # Cached read deliberately bypasses wrappers that model a blocking
        # snapshot receive retry. It never advances the fake clock.
        return Device.snapshot(self)

    def move_joints(self, q):
        self.q = np.asarray(q, dtype=float)
        self.calls.append(('move', self.q.copy()))

    def grip(self, width, force):
        self.calls.append(('grip', width, force))
        self.width = width

    def stop(self):
        self.calls.append(('stop',))

    def advance(self, dt):
        assert self.is_offline_replay
        self.clock.sleep(dt)


class Controller:
    def __init__(self, *, timed_accepted=True):
        # Exercise the actual shared policy adapter's AGX action convention.
        self.gripper = _JawCommandMirror(.1)
        self.timed_accepted = timed_accepted
        self.calls = []

    def apply_action(self, action, dt):
        self.calls.append((action, dt))
        if 'gripper' in action:
            self.gripper.command(action['gripper'])
        # The actual accepted target deliberately differs from raw policy
        # action, so accidentally bypassing RobotController fails the test.
        return SimpleNamespace(joint_target=np.full(7, .001),
                               time_parameterized=self.timed_accepted)


class Policy:
    def __init__(self, context, actions, *, recovering=False):
        self.context = context
        self.actions = actions
        self.index = 0
        self.observations = []
        self.recovering = recovering
        self.stage = SimpleNamespace(value='pregrasp')

    def act(self, observation):
        self.observations.append(observation)
        if self.recovering:
            self.stage = SimpleNamespace(value='recovering')
        if self.index == len(self.actions):
            self.context.complete = True
            return {'arm': np.zeros(7), 'gripper': 1.}
        action = dict(self.actions[self.index])
        self.index += 1
        return action

    def drain_events(self):
        return []

    def get_info(self):
        return {'stage': self.stage.value, 'reason': 'test failure'}


class Context:
    def __init__(self, actions, *, validator=lambda _state, _command: True,
                 timed_accepted=True, recovering=False):
        self.settings = PickSettings()
        self.complete = False
        self.failed_reason = None
        self.geometry = SimpleNamespace(check_feedback=lambda _state: None)
        self.robot_controller = Controller(timed_accepted=timed_accepted)
        self.policy = Policy(self, actions, recovering=recovering)
        self.validate_command = validator
        self._workers = []

    def observe(self, state, now):
        return dict(state, timestamp=now)


def action(gripper=-1., *, timed=False):
    return {'arm': np.full(7, .1), 'gripper': gripper,
            'arm_time_parameterized': timed}


def run(context, device, clock, **kwargs):
    events = []
    result = run_full_policy(
        context, device, lambda name, payload: events.append((name, payload)),
        mode='offline_ideal_feedback' if device.is_offline_replay else 'hardware',
        clock=clock.now, sleep=clock.sleep, **kwargs)
    return result, events


def test_startup_threshold_override_reaches_first_dispatch_without_changing_contact():
    clock = Clock()
    device = Device(clock, force=.3)
    context = Context([action(gripper=1.)])
    context.settings = replace(context.settings, startup_gripper_force_threshold_n=.5)
    result, events = run(context, device, clock)
    assert result['command_count'] > 0
    assert any(call[0] == 'move' for call in device.calls)
    checks = [payload for name, payload in events if name == 'startup_gripper_force_check']
    assert {'runner_initial', 'first_tick'} <= {p['phase'] for p in checks}
    assert all(p['threshold_n'] == .5 and not p['rejected'] for p in checks)
    assert context.settings.min_force_n == .2


@pytest.mark.parametrize('snapshot_index', [1, 2])
def test_startup_threshold_boundary_still_rejects_before_motion(snapshot_index):
    clock = Clock()
    device = Device(clock, force=.3)
    original_snapshot = device.snapshot
    def snapshot():
        state = original_snapshot()
        if device.snapshots >= snapshot_index:
            state['force_n'] = .5
        return state
    device.snapshot = snapshot
    context = Context([action(gripper=1.)])
    context.settings = replace(context.settings, startup_gripper_force_threshold_n=.5)
    with pytest.raises(PickRejected) as caught:
        run(context, device, clock)
    assert caught.value.details['force_n'] == .5
    assert caught.value.details['threshold_n'] == .5
    assert not any(call[0] in ('move', 'grip') for call in device.calls)


class HoldDevice(Device):
    def __init__(self, clock, control, **kwargs):
        super().__init__(clock, **kwargs)
        self.control = control
        self.holds = 0
        self.exit_after_holds = 4

    def hold_current_position(self, state=None):
        assert state is not None
        self.hold_target = np.asarray(state['joints']).copy()
        return self.maintain_hold_position(state=state)

    def maintain_hold_position(self, state=None):
        assert state is not None
        self.holds += 1
        self.calls.append(('hold', self.hold_target.copy()))
        if self.holds >= self.exit_after_holds:
            self.control.request_exit()
        return {'hold_joints_rad': self.hold_target.tolist()}


def test_pause_after_policy_calculation_discards_pending_arm_and_close_action():
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(-1.)])
    original = context.policy.act

    def request_during_calculation(observation):
        value = original(observation)
        device.q[:] = .04  # Feedback differs from the old prepared target.
        control.request_pause()
        return value

    context.policy.act = request_during_calculation
    result, events = run(context, device, clock, pause_control=control)
    assert result['status'] == 'paused_exit_holding'
    assert result['command_count'] == 0 and not result['task_resumed']
    assert [x[0] for x in device.calls] == ['prepare']+['hold']*4
    assert not context.robot_controller.calls
    np.testing.assert_array_equal(result['hold_joints_rad'], np.full(7, .04))
    assert [name for name, _ in events][-2:] == ['motion_paused', 'exit_holding']


@pytest.mark.parametrize('snapshot_number', [2, 3])
def test_pause_during_feedback_receive_precedes_old_start_and_loaded_jaw_checks(snapshot_number):
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    snapshot = device.snapshot

    def receive():
        if device.snapshots+1 == snapshot_number:
            device.q[:] = .01
            control.request_exit()
        state = snapshot()
        if control.pause_requested:
            state['force_n'] = 2.
        return state

    def slow_validation(_state, _command):
        # Force an expired-feedback refresh in the third-snapshot case.
        clock.sleep(.25)
        return True

    device.snapshot = receive
    result, _ = run(Context([action(-1.)], validator=slow_validation), device, clock,
                    pause_control=control)
    assert result['status'] == 'paused_exit_holding'
    assert [x[0] for x in device.calls] == ['prepare', 'hold']
    np.testing.assert_array_equal(result['hold_joints_rad'], np.full(7, .01))


def test_pause_at_dispatch_boundary_keeps_current_target_and_never_sends_closing():
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    device.q[-1] = np.deg2rad(97.53)

    def validator(_state, _command):
        control.request_pause()
        return True

    context = Context([action(-1.)], validator=validator)
    result, _ = run(context, device, clock, pause_control=control)
    assert result['status'] == 'paused_exit_holding'
    assert len(context.robot_controller.calls) == 1  # Prepared only, never sent.
    assert [x[0] for x in device.calls] == ['prepare']+['hold']*4
    assert device.width == .1
    assert result['hold_joints_rad'][-1] == pytest.approx(np.deg2rad(97.53))


def test_pause_after_arm_dispatch_does_not_dispatch_pending_gripper_command():
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    move = device.move_joints

    def request_after_move(q):
        move(q)
        control.request_pause()

    device.move_joints = request_after_move
    result, _ = run(Context([action(-1.)]), device, clock, pause_control=control)
    assert result['command_count'] == 1
    assert [x[0] for x in device.calls] == ['prepare', 'move']+['hold']*4
    assert device.width == .1


def test_pause_before_start_can_hold_a_loaded_gripper_without_auto_open_or_speed_change():
    clock, control = Clock(), PauseControl()
    control.request_exit()  # Q still captures and sends the hold before exit.
    device = HoldDevice(clock, control, force=2.)
    device.width = .034
    result, _ = run(Context([action(1.)]), device, clock, pause_control=control)
    assert result['status'] == 'paused_exit_holding'
    assert [x[0] for x in device.calls] == ['hold']
    assert device.width == .034 and result['command_count'] == 0


def test_pause_has_no_task_timeout_and_does_not_follow_feedback_drift():
    clock, control = Clock(), PauseControl()
    control.request_pause()
    device = HoldDevice(clock, control)
    device.exit_after_holds = 12
    snapshot = device.snapshot

    def drifting_snapshot():
        state = snapshot()
        if control.holding:
            state['joints'][2] += .008
        return state

    device.snapshot = drifting_snapshot
    result, _ = run(Context([action(-1.)]), device, clock,
                    pause_control=control, max_cycle_s=.02, deadline=clock.now()+.02)
    assert result['holding_elapsed_s'] > .1
    assert result['max_hold_feedback_error_rad'] == pytest.approx(.008)
    for call in device.calls:
        np.testing.assert_array_equal(call[1], np.zeros(7))


def test_real_feedback_fault_during_pause_still_requests_stop_without_jaw_release():
    clock, control = Clock(), PauseControl()
    control.request_pause()
    device = HoldDevice(clock, control, snapshot_failure_at=3)
    with pytest.raises(FeedbackError, match='持续未更新'):
        run(Context([action(1.)]), device, clock, pause_control=control)
    assert [x[0] for x in device.calls] == ['hold', 'stop']


def test_ctrl_c_when_already_holding_exits_without_stop():
    from mujoco_conveyor_sorting.runtime.full_policy_runner import hold_position_until_exit
    clock, control = Clock(), PauseControl()
    control.interrupt()
    device = HoldDevice(clock, control)
    events = []

    def emit(name, data):
        events.append(name)
        if name == 'motion_paused':
            assert control.holding
            control.interrupt()

    result = hold_position_until_exit(device, control, emit, mode='hardware',
                                     clock=clock.now, sleep=clock.sleep)
    assert result['status'] == 'paused_exit_holding'
    assert events == ['motion_paused', 'exit_holding']
    assert [x[0] for x in device.calls] == ['hold']


def test_dispatch_uses_shared_controller_target_and_gripper_action_convention():
    clock = Clock()
    device = Device(clock)
    context = Context([action(-1.), action(0.), action(1.)])
    result, _ = run(context, device, clock)
    assert result['status'] == 'cycle_completed_from_feedback'
    assert [item[1:] for item in device.calls if item[0] == 'grip'] == [
        (0., 2.), (.05, 1.), (.1, 1.)]
    moves = [item[1] for item in device.calls if item[0] == 'move']
    assert len(moves) == 3
    for q in moves:
        np.testing.assert_array_equal(q, np.full(7, .001))
    assert not any(item[0] == 'stop' for item in device.calls)


@pytest.mark.parametrize('raise_instead', [False, True])
def test_scene_rejection_after_controller_does_not_send_arm_or_gripper(raise_instead):
    def reject(_state, _command):
        if raise_instead:
            raise PickRejected('blocked actual command')
        return False

    clock = Clock()
    device = Device(clock)
    context = Context([action(-1.)], validator=reject)
    with pytest.raises(PickRejected):
        run(context, device, clock)
    assert [item[0] for item in device.calls] == ['prepare']
    # Updating the pure command mirror has not moved the physical jaw.
    assert context.robot_controller.gripper.target_open_fraction == 0.
    assert device.width == .1


def test_timed_controller_fallback_is_not_sent_as_a_certified_path():
    clock = Clock()
    device = Device(clock)
    context = Context([action(timed=True)], timed_accepted=False)
    with pytest.raises(PickRejected, match='未接受已计时'):
        run(context, device, clock)
    assert [item[0] for item in device.calls] == ['prepare']


@pytest.mark.parametrize('reject_endpoint', [0, 1])
def test_gripper_sweep_rejection_prevents_both_arm_and_jaw_dispatch(reject_endpoint):
    clock = Clock()
    device = Device(clock)
    context = Context([action(-1.)])
    checked = []

    def validate_gripper(state, width):
        checked.append((state['joints'], width))
        if len(checked)-1 == reject_endpoint:
            context.motion_validation_rejection = {
                'sampled_width_m': .04, 'reason': 'scene_geometry',
                'message': '路径碰到可见障碍点云：left_finger_geom',
            }
        return len(checked)-1 != reject_endpoint

    context.validate_gripper_command = validate_gripper
    with pytest.raises(PickRejected, match='夹爪开合路径') as error:
        run(context, device, clock)
    assert '100.0→0.0 mm' in str(error.value)
    assert '40.0 mm 时拒绝' in str(error.value)
    assert 'left_finger_geom' in str(error.value)
    assert error.value.details['endpoint'] == ('measured' if reject_endpoint == 0 else 'commanded')
    assert error.value.details['measured_gripper_width_m'] == .1
    assert error.value.details['requested_gripper_width_m'] == 0.
    assert len(checked) == reject_endpoint+1
    assert checked[0][0] == [0.]*7
    if reject_endpoint:
        assert checked[1][0] == [.001]*7
    assert [item[0] for item in device.calls] == ['prepare']


def test_recovery_action_cannot_automatically_open_jaws_or_return_arm():
    clock = Clock()
    device = Device(clock)
    context = Context([action(1.)], recovering=True)
    with pytest.raises(PickRejected, match='完整策略停止'):
        run(context, device, clock)
    assert not context.robot_controller.calls
    assert [item[0] for item in device.calls] == ['prepare']


def test_feedback_fault_after_closing_stops_without_release():
    clock = Clock()
    # Initial snapshot, one successful tick, then the adapter's fresh-state
    # gate fails. The next opening action must never reach the SDK boundary.
    device = Device(clock, snapshot_failure_at=3)
    context = Context([action(-1.), action(1.)])
    with pytest.raises(FeedbackError, match='持续未更新'):
        run(context, device, clock)
    assert [item[1] for item in device.calls if item[0] == 'grip'] == [0.]
    assert device.calls[-1] == ('stop',)


@pytest.mark.parametrize('during_hold', [False, True])
def test_fault_reporting_error_preserves_original_feedback_fault_after_one_stop(during_hold):
    from mujoco_conveyor_sorting.runtime.full_policy_runner import hold_position_until_exit

    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control, snapshot_failure_at=2) if during_hold else Device(
        clock, snapshot_failure_at=3)

    def emit(name, _payload):
        if name == 'stopped':
            raise OSError('failed to record stopped event')

    with pytest.raises(FeedbackError, match='持续未更新') as raised:
        if during_hold:
            hold_position_until_exit(device, control, emit, mode='hardware',
                                     clock=clock.now, sleep=clock.sleep)
        else:
            run_full_policy(Context([action(-1.), action(1.)]), device, emit,
                            mode='hardware', clock=clock.now, sleep=clock.sleep)
    assert sum(call[0] == 'stop' for call in device.calls) == 1
    assert raised.value.details['stop_reporting_error'] == 'failed to record stopped event'


def test_failed_stop_request_does_not_replace_original_fault_or_lose_stop_event():
    clock = Clock()
    device = Device(clock)
    original = PickRejected('original scene rejection')
    events = []
    validations = 0

    def reject(_state, _command):
        nonlocal validations
        validations += 1
        if validations == 1:
            return True
        raise original

    def failed_stop():
        device.calls.append(('stop',))
        raise OSError('CAN stop send failed')

    device.stop = failed_stop
    context = Context([action(-1.), action(1.)], validator=reject)
    with pytest.raises(PickRejected) as error:
        run_full_policy(context, device,
                        lambda event, payload: events.append((event, payload)),
                        mode='hardware', clock=clock.now, sleep=clock.sleep)
    assert error.value is original
    assert [name for name, _ in events][-2:] == ['stop_request_failed', 'stopped']
    assert events[-2][1]['error'] == 'CAN stop send failed'
    assert events[-1][1]['automatic_gripper_release'] is False
    assert [item[0] for item in device.calls] == ['prepare', 'move', 'grip', 'stop']
    assert events[-1][1]['stop_requested'] is True


@pytest.mark.parametrize('reason', ['loaded', 'stale'])
def test_initial_preflight_failure_makes_no_control_writes(reason):
    clock = Clock()
    device = Device(clock, force=.5 if reason == 'loaded' else 0.,
                    snapshot_failure_at=1 if reason == 'stale' else None)
    context = Context([action(1.)])

    with pytest.raises((PickRejected, FeedbackError)):
        run(context, device, clock)
    assert device.calls == []
    assert not context.robot_controller.calls


@pytest.mark.parametrize('offline,mode', [(True, 'hardware'),
                                         (False, 'offline_ideal_feedback'),
                                         (False, 'auto')])
def test_device_mode_mismatch_never_falls_back_or_touches_a_device(offline, mode):
    clock = Clock()
    device = Device(clock, offline=offline)
    context = Context([action()])
    with pytest.raises(ValueError):
        run_full_policy(context, device, lambda *_: None, mode=mode,
                        clock=clock.now, sleep=clock.sleep)
    assert device.snapshots == 0 and device.calls == []


def test_offline_completion_does_not_claim_hardware_execution_or_physical_grasp():
    clock = Clock()
    device = Device(clock, offline=True)
    result, _ = run(Context([action()]), device, clock)
    assert result['status'] == 'offline_full_policy_completed'
    assert result['feedback_mode'] == 'offline_ideal_feedback'
    assert result['hardware_motion_commands_sent'] is False
    assert result['physical_grasp_verified'] is False


def test_offline_async_planning_wait_does_not_fast_forward_policy_time():
    clock = Clock()
    device = Device(clock, offline=True)
    context = Context([action()])

    class Worker:
        @property
        def busy(self):
            return clock.now() < 10.06

    context.workers = (Worker(),)
    result, _ = run(context, device, clock)
    assert result['elapsed_policy_s'] == pytest.approx(.01)
    assert result['wall_elapsed_s'] >= .06
    assert result['command_count'] == 1


def test_explicit_policy_failure_cannot_dispatch_its_last_action():
    clock = Clock()
    device = Device(clock)
    context = Context([action(1.)])
    context.failed_reason = 'no certified route'
    with pytest.raises(PickRejected, match='no certified route'):
        run(context, device, clock)
    assert not context.robot_controller.calls
    assert [item[0] for item in device.calls] == ['prepare']


def fail_policy_after_one_command(context, reason='visual velocity unavailable'):
    original = context.policy.act

    def act(observation):
        result = original(observation)
        if context.policy.index > 1:
            context.failed_reason = reason
        return result

    context.policy.act = act


@pytest.mark.parametrize('failure', ['policy', 'recovering', 'budget'])
def test_ordinary_failure_after_motion_keeps_fixed_measured_target_and_closed_jaw(failure):
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(-1.), action(1.)])
    if failure == 'policy':
        fail_policy_after_one_command(context)
    elif failure == 'recovering':
        original = context.policy.act

        def recovering(observation):
            result = original(observation)
            if context.policy.index > 1:
                context.policy.stage.value = 'recovering'
            return result

        context.policy.act = recovering
    snapshot = device.snapshot

    def drifting_snapshot():
        state = snapshot()
        if control.holding:
            state['joints'][2] += .008
        return state

    device.snapshot = drifting_snapshot
    result, events = run(context, device, clock, pause_control=control,
                         max_cycle_s=.005 if failure == 'budget' else 300.)
    assert result['status'] == 'task_failed_exit_holding'
    assert result['task_failed'] is True and result['task_resumed'] is False
    assert result['physical_grasp_verified'] is False
    assert result['operator_hold_loop_used'] is True and result['exit_origin'] == 'operator'
    assert result['holding_loop_active'] is False
    assert result['command_count'] == 1 and result['hold_command_count'] == 4
    assert result['max_hold_feedback_error_rad'] == pytest.approx(.008)
    assert [item[0] for item in device.calls] == ['prepare', 'move', 'grip']+['hold']*4
    assert [item[1] for item in device.calls if item[0] == 'grip'] == [0.]
    for call in device.calls:
        if call[0] == 'hold':
            np.testing.assert_array_equal(call[1], np.full(7, .001))
    assert result['failure_code'] == ('cycle_budget_exhausted' if failure == 'budget' else 'policy_failed')
    if failure == 'policy':
        assert result['failure_reason'] == 'visual velocity unavailable'
    names = [name for name, _ in events]
    assert names.count('task_failed') == names.count('task_failed_holding') == 1
    assert 'stopped' not in names and names.count('gripper_command') == 1
    captured = next(data for name, data in events if name == 'failure_hold_feedback')
    assert captured['feedback_phase'] == 'captured_before_first_hold_command'
    assert captured['joint_stamp'] <= clock.now()


@pytest.mark.parametrize('offline', [False, True])
def test_task_failure_without_operator_control_sends_one_hold_and_records_lifetime(offline):
    clock, device_control = Clock(), PauseControl()
    device = HoldDevice(clock, device_control, offline=offline)
    context = Context([action(-1.), action(1.)])
    fail_policy_after_one_command(context)
    result, events = run(context, device, clock)
    assert result['task_failed'] is True and result['status'] == 'task_failed_exit_holding'
    assert result['operator_hold_loop_used'] is False
    assert result['exit_origin'] == 'no_pause_control_single_hold'
    assert result['hold_command_count'] == 1
    assert result['hardware_motion_commands_sent'] is (not offline)
    assert [item[0] for item in device.calls] == ['prepare', 'move', 'grip', 'hold']
    assert 'stopped' not in [name for name, _ in events]


def test_failed_task_hold_logs_original_feedback_stamps_while_waiting_for_operator():
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    device.exit_after_holds = 205
    context = Context([action(-1.), action(1.)])
    fail_policy_after_one_command(context)
    result, events = run(context, device, clock, pause_control=control)
    samples = [data for name, data in events if name == 'hold_feedback']
    assert result['holding_elapsed_s'] > 2.
    assert samples
    for sample in samples:
        assert sample['feedback_phase'] == 'while_repeating_fixed_hold_target'
        assert sample['joint_stamp'] == sample['pose_stamp'] == sample['gripper_stamp']
        assert sample['holding_elapsed_s'] >= 2.
        assert sample['joint_stamp'] <= clock.now()
        np.testing.assert_array_equal(sample['hold_joints_rad'], np.full(7, .001))


@pytest.mark.parametrize('rejection', ['motion', 'gripper'])
def test_bool_geometry_rejection_after_sent_command_holds_without_sending_rejected_arm_or_opening(
        rejection):
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(-1.), action(1.)])
    checks = []

    if rejection == 'motion':
        def validate_motion(_state, command):
            checks.append(command.joint_target.copy())
            context.motion_validation_rejection = {
                'reason': 'scene_geometry', 'message': '路径碰到可见障碍点云：right_finger_geom',
                'details': {'geometry': 'right_finger_geom', 'point_count': 8}}
            return len(checks) == 1

        context.validate_command = validate_motion
    else:
        def validate_gripper(state, width):
            checks.append((state['joints'], width))
            # First closing command checks both measured and commanded arm
            # endpoints. Reject the next, opening command at its first check.
            return len(checks) <= 2

        context.validate_gripper_command = validate_gripper

    result, events = run(context, device, clock, pause_control=control)
    assert result['status'] == 'task_failed_exit_holding'
    assert result['failure_code'] == ('motion_geometry_rejected' if rejection == 'motion'
                                      else 'gripper_geometry_rejected')
    assert result['command_count'] == 1
    assert len(checks) == (2 if rejection == 'motion' else 3)
    assert [call[0] for call in device.calls] == ['prepare', 'move', 'grip']+['hold']*4
    assert [call[1] for call in device.calls if call[0] == 'grip'] == [0.]
    assert device.width == 0.
    assert len(context.robot_controller.calls) == 2  # Second command was prepared only.
    assert sum(name == 'joint_command' for name, _ in events) == 1
    rejected = [payload for name, payload in events if name == 'motion_geometry_rejected']
    if rejection == 'motion':
        assert len(rejected) == 1 and rejected[0]['sent'] is False
        np.testing.assert_array_equal(rejected[0]['joint_target_rad'], checks[-1])
        assert rejected[0]['requested_gripper_width_m'] == .1
        assert rejected[0]['validation_gripper_width_m'] == .1
        assert rejected[0]['feedback']['width_m'] == 0.
        assert rejected[0]['rejection']['details']['point_count'] == 8
        assert 'right_finger_geom' in result['failure_reason']
    else:
        assert not rejected
    assert 'stopped' not in [name for name, _ in events]
    for call in device.calls:
        if call[0] == 'hold':
            np.testing.assert_array_equal(call[1], np.full(7, .001))


def test_policy_failure_before_first_command_never_requests_hold_or_emergency_stop():
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(1.)])
    context.failed_reason = 'no certified route'
    with pytest.raises(FullPolicyTaskFailure, match='no certified route') as failure:
        run(context, device, clock, pause_control=control)
    assert failure.value.failure_code == 'policy_failed'
    assert [item[0] for item in device.calls] == ['prepare']
    assert not context.robot_controller.calls


@pytest.mark.parametrize('fault', ['capture_feedback', 'maintain_feedback', 'hold_send'])
def test_real_fault_during_failed_task_hold_stops_once_and_keeps_both_causes(fault):
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control,
        snapshot_failure_at={'capture_feedback': 4, 'maintain_feedback': 5}.get(fault))
    context = Context([action(-1.), action(1.)])
    fail_policy_after_one_command(context)
    if fault == 'hold_send':
        def failed_hold(state=None):
            device.calls.append(('hold_attempt',))
            raise OSError('CAN hold send failed')

        device.hold_current_position = failed_hold
    events = []
    with pytest.raises((FeedbackError, OSError)) as error:
        run_full_policy(context, device,
            lambda name, data: events.append((name, data)), mode='hardware',
            clock=clock.now, sleep=clock.sleep, pause_control=control)
    assert sum(call[0] == 'stop' for call in device.calls) == 1
    assert [call[1] for call in device.calls if call[0] == 'grip'] == [0.]
    assert isinstance(error.value.__cause__, FullPolicyTaskFailure)
    assert error.value.details['failure_reason'] == 'visual velocity unavailable'
    assert error.value.details['hold_fault'] == str(error.value)
    stop_event = next(data for name, data in events if name == 'stopped')
    assert stop_event['during_position_hold'] is True
    assert stop_event['failure_reason'] == 'visual velocity unavailable'
    assert stop_event['automatic_gripper_release'] is False


@pytest.mark.parametrize('fail_all_followup_events', [False, True])
def test_reporting_failure_after_task_failure_still_stops_once_and_keeps_original_fault(
        fail_all_followup_events):
    clock, control = Clock(), PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(-1.), action(1.)])
    fail_policy_after_one_command(context)
    reporting_failed = False
    original_error = OSError('task failure log write failed')

    def emit(name, _data):
        nonlocal reporting_failed
        if name == 'task_failed':
            reporting_failed = True
            raise original_error
        if reporting_failed and fail_all_followup_events:
            raise OSError('followup log write failed')

    with pytest.raises(OSError) as error:
        run_full_policy(context, device, emit, mode='hardware',
            clock=clock.now, sleep=clock.sleep, pause_control=control)
    assert error.value is original_error
    assert isinstance(error.value.__cause__, FullPolicyTaskFailure)
    assert error.value.details['failure_reason'] == 'visual velocity unavailable'
    assert sum(call[0] == 'stop' for call in device.calls) == 1
    assert [call[1] for call in device.calls if call[0] == 'grip'] == [0.]
    assert not any(call[0] == 'hold' for call in device.calls)


def test_slow_first_tick_is_diagnostic_and_does_not_latch_emergency_stop():
    clock = Clock()

    def slow_validator(_state, _command):
        clock.sleep(.051)
        return True

    device = Device(clock)
    context = Context([action()], validator=slow_validator)
    result, events = run(context, device, clock)
    assert result['status'] == 'cycle_completed_from_feedback'
    assert [item[0] for item in device.calls] == ['prepare', 'move', 'grip']
    timing = result['control_timing']
    assert timing['max_dispatch_lateness_s'] == pytest.approx(.051)
    assert timing['max_tick_compute_s'] == pytest.approx(.051)
    assert timing['late_tick_count'] == 1
    first = next(data for name, data in events if name == 'joint_command')
    assert first['dispatch_timing']['gap_s'] is None
    assert first['dispatch_timing']['compute_s'] == pytest.approx(.051)
    assert not any(name == 'stopped' for name, _ in events)


def test_delayed_moving_tick_uses_actual_dt_without_catchup_burst_or_timing_stop():
    clock = Clock()
    validations = 0

    def slow_second_tick(_state, _command):
        nonlocal validations
        validations += 1
        if validations == 2:
            clock.sleep(.075)
        return True

    device = Device(clock)
    context = Context([action(), action(), action()], validator=slow_second_tick)
    result, events = run(context, device, clock)
    sends = [payload for name, payload in events if name == 'joint_command']
    assert result['max_dispatch_gap_s'] == pytest.approx(.075)
    assert sends[1]['dispatch_timing']['gap_s'] == pytest.approx(.075)
    assert sends[2]['dispatch_timing']['gap_s'] == pytest.approx(.01)
    controller_dt = [dt for _action, dt in context.robot_controller.calls]
    assert controller_dt == pytest.approx([.01, .01, .01])
    policy_dt = [obs['control_dt'] for obs in context.policy.observations[:3]]
    assert policy_dt == pytest.approx(controller_dt)
    assert not any(call[0] == 'stop' for call in device.calls)


def test_regular_computation_does_not_reduce_motion_dt_or_control_frequency():
    clock = Clock()

    def validator(_state, _command):
        clock.sleep(.004)
        return True

    device = Device(clock)
    context = Context([action(), action(), action()], validator=validator)
    result, events = run(context, device, clock)
    sends = [payload['dispatch_timing'] for name, payload in events if name == 'joint_command']
    assert [dt for _action, dt in context.robot_controller.calls] == pytest.approx([.01]*3)
    assert [item['gap_s'] for item in sends[1:]] == pytest.approx([.01]*2)
    assert [item['compute_s'] for item in sends] == pytest.approx([.004]*3)
    assert result['max_dispatch_gap_s'] == pytest.approx(.01)


@pytest.mark.parametrize('initial_age, delay', [(0., .25), (.19, .02)])
def test_expired_feedback_during_computation_is_refreshed_instead_of_timing_stop(initial_age, delay):
    clock = Clock()
    measured = []

    def validator(state, _command):
        measured.append(state['joint_stamp'])
        if len(measured) == 1:
            clock.sleep(delay)
        return True

    device = Device(clock)
    snapshot = device.snapshot

    def aged_snapshot():
        state = snapshot()
        if device.snapshots <= 2:
            state['feedback_timing']['oldest_packet_age_s'] = initial_age
        return state

    device.snapshot = aged_snapshot
    result, _ = run(Context([action()], validator=validator), device, clock)
    assert result['status'] == 'cycle_completed_from_feedback'
    assert result['control_timing']['feedback_refresh_count'] == 1
    assert measured == pytest.approx([10., 10.+delay])
    assert [call[0] for call in device.calls] == ['prepare', 'move', 'grip']


def test_real_feedback_loss_during_slow_moving_tick_still_stops():
    clock = Clock()
    validations = 0

    def validator(_state, _command):
        nonlocal validations
        validations += 1
        if validations == 2:
            clock.sleep(.25)
        return True

    device = Device(clock, snapshot_failure_at=4)
    with pytest.raises(FeedbackError, match='持续未更新'):
        run(Context([action(-1.), action(1.)], validator=validator), device, clock)
    assert [call[0] for call in device.calls] == ['prepare', 'move', 'grip', 'stop']


def test_feedback_receive_wait_uses_return_time_for_prediction_without_extra_refresh():
    clock = Clock()
    device = Device(clock)
    snapshot = device.snapshot

    def receive_with_pause():
        if device.snapshots in (1, 2):
            clock.sleep(.3)
        return snapshot()

    device.snapshot = receive_with_pause
    context = Context([action(), action()])
    result, events = run(context, device, clock)
    observations = context.policy.observations
    assert observations[0]['timestamp'] == pytest.approx(.3)
    assert observations[1]['timestamp'] == pytest.approx(.6)
    assert observations[1]['control_dt'] == pytest.approx(.3)
    assert result['control_timing']['feedback_refresh_count'] == 0
    assert [name for name, _data in events].count('joint_command') == 2
    assert not any(call[0] == 'stop' for call in device.calls)


def test_preparation_fault_records_stop_without_sending_emergency_stop():
    clock = Clock()
    device = Device(clock, snapshot_failure_at=2)
    context = Context([action()])
    events = []
    with pytest.raises(FeedbackError):
        run_full_policy(context, device, lambda name, payload: events.append((name, payload)),
                        mode='hardware', clock=clock.now, sleep=clock.sleep)
    assert [call[0] for call in device.calls] == ['prepare']
    assert events[-1] == ('stopped', {'automatic_gripper_release': False,
                                     'feedback_mode': 'hardware', 'stop_requested': False})


def test_failed_motion_send_still_stops_even_before_first_successful_command():
    clock = Clock()
    device = Device(clock)

    def failed_move(q):
        device.calls.append(('move_attempt', q))
        raise OSError('CAN send failed after partial write')

    device.move_joints = failed_move
    with pytest.raises(OSError, match='partial write'):
        run(Context([action()]), device, clock)
    assert [call[0] for call in device.calls] == ['prepare', 'move_attempt', 'stop']


def test_explicit_cycle_budget_crossed_during_preparation_prevents_first_motion():
    clock = Clock()

    def validator(_state, _command):
        clock.sleep(.051)
        return True

    device = Device(clock)
    with pytest.raises(PickRejected, match='时间预算'):
        run(Context([action()], validator=validator), device, clock, deadline=10.02)
    assert [call[0] for call in device.calls] == ['prepare']


def test_out_of_range_gripper_action_fails_before_any_physical_command():
    clock = Clock()
    device = Device(clock)
    context = Context([action(2.)])
    with pytest.raises(ValueError, match='gripper action'):
        run(context, device, clock)
    assert [item[0] for item in device.calls] == ['prepare']


def test_moving_context_uses_runner_origin_and_logs_matching_target_ticks():
    clock = Clock()
    device = Device(clock)
    context = Context([action(), action()])
    origin = []
    context.set_policy_clock_origin = origin.append
    context.target_state_record = lambda now: {'policy_time_s': now, 'absolute_time_s': origin[0]+now}
    _, events = run(context, device, clock)
    assert origin == [10.]
    target_events = [value for name, value in events if name == 'target_state']
    assert target_events
    assert all(value['absolute_time_s'] == 10.+value['policy_time_s'] for value in target_events)
    for index, (name, _) in enumerate(events):
        if name == 'target_state': assert events[index-1][0] == 'feedback'


def test_prepared_high_uses_existing_enabled_hold_without_arm_or_jaw_dispatch():
    clock = Clock(); control = PauseControl()
    device = HoldDevice(clock, control)
    context = Context([action(gripper=-1.)])
    context.policy.prepared_high_state = {'joint_positions_rad': np.zeros(7), 'timestamp': 0.}
    result, events = run(context, device, clock, pause_control=control)
    assert result['status'] == 'prepared_high_only'
    assert device.holds >= 1
    assert not any(call[0] in ('stop', 'grip') for call in device.calls)
    assert not context.robot_controller.calls
    assert any(name == 'prepared_high_ready' for name, _ in events)
    assert not result['physical_grasp_verified']

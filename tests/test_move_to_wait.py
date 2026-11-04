"""Offline CLI/device fault tests. The fake never opens CAN or a camera."""
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from nero_core.move_to_wait import main


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path/'wait.yaml'
    path.write_text(yaml.safe_dump({'joint_names': [f'joint{i}' for i in range(1, 8)],
        'standby_joint_rad': [.1]*7, 'hardware': {'can_channel': 'NO_DEVICE',
        'firmware': 'TEST_ONLY', 'joint_command_mode': 'js'},
        'pick_settings': {'speed_percent': 100, 'min_force_n': .2}}))
    return path


class Clock:
    now = 1.
    def __call__(self): return self.now
    def sleep(self, duration): self.now += duration


class Device:
    def __init__(self, clock, mode='success'):
        self.clock, self.mode = clock, mode
        self.q = np.ones(7)*(.1 if mode == 'already' else 0.)
        self.calls = []; self.sent = None
        self.robot = SimpleNamespace(set_speed_percent=lambda value: self.calls.append(('speed', value)))
    def configure_joint_command_mode(self, mode): self.calls.append(('mode', mode))
    def start(self):
        self.calls.append(('start',))
        if self.mode == 'can_missing': raise ValueError("Device 'can0' does not exist.")
    def close(self): self.calls.append(('close',))
    def stop(self): self.calls.append(('stop',))
    def move_joints(self, q):
        self.calls.append(('joint', q)); self.sent = self.clock()
        if self.mode == 'partial_send': raise RuntimeError('injected partial CAN target failure')
        if self.mode != 'timeout': self.q = np.asarray(q)
    def snapshot(self):
        now = self.clock()
        if self.mode == 'moving_start': self.q = np.ones(7)*(now-1.)
        if self.sent is not None and now-self.sent >= .04:
            if self.mode == 'interrupt': signal.raise_signal(signal.SIGINT)
            if self.mode == 'feedback_fault': raise RuntimeError('injected driver fault')
        stamp = now-.3 if self.mode == 'stale' else now
        if self.mode == 'duplicate' and self.sent is not None: stamp = self.sent
        force = 2. if self.mode == 'loaded' else .02
        state = {'joints': self.q.tolist(), 'joint_stamp': stamp, 'width_m': .1, 'force_n': force}
        if self.mode == 'epoch_reset':
            epoch = 1 if self.sent is not None and now-self.sent >= .04 else 0
            state.update(feedback_time={key: {'clock_epoch': epoch, 'mapping_quality': 'mapped'}
                for key in ('joint_12', 'joint_34', 'joint_56', 'joint_7')}, joint_stamp_monotonic=stamp)
        return state


def execute(path, tmp_path, mode='success', extra=()):
    clock = Clock(); device = Device(clock, mode); output = tmp_path/'run'
    def factory(channel, firmware, execute):
        assert (channel, firmware, execute) == ('NO_DEVICE', 'TEST_ONLY', True)
        return device
    status = main(['--config', str(path), '--output', str(output), '--execute', *extra],
                  hardware_factory=factory, clock=clock, sleep=clock.sleep)
    report = json.loads((output/'result.json').read_text())
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'return_wait_started'
    assert events[-1]['event'] == 'return_wait_finished'
    assert report['gripper_commands'] == 0 and not report['automatic_enable']
    assert ('close',) in device.calls
    return status, report, device


def test_default_only_checks_and_needs_no_camera_tcp_or_bin_config(config_path, capsys):
    def forbidden(*a): pytest.fail('check mode must not construct a hardware adapter')
    assert main(['--config', str(config_path)], hardware_factory=forbidden) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['hardware_connections'] == report['joint_target_commands'] == 0
    assert report['joint_command_api'] == 'move_j'
    assert report['target_joints_deg'][0] == pytest.approx(np.rad2deg(.1))


@pytest.mark.parametrize('speed', [None, 25, 100])
def test_one_smoothed_joint_target_and_measured_stable_arrival(config_path, tmp_path, speed):
    extra = [] if speed is None else ['--speed-percent', str(speed)]
    status, report, device = execute(config_path, tmp_path, extra=extra)
    assert status == 0 and report['status'] == 'wait_reached'
    assert ('mode', 'j') in device.calls  # cell grasp mode JS never leaks into return
    assert ('speed', 100 if speed is None else speed) in device.calls
    assert sum(c[0] == 'joint' for c in device.calls) == 1
    assert ('stop',) not in device.calls
    assert report['move_wall_s'] >= .1


def test_already_waiting_sends_neither_speed_nor_joint_command(config_path, tmp_path):
    status, report, device = execute(config_path, tmp_path, 'already')
    assert status == 0 and report['status'] == 'already_at_wait'
    assert report['speed_commands'] == report['joint_target_commands'] == 0


@pytest.mark.parametrize('mode', ['can_missing', 'loaded', 'stale', 'moving_start'])
def test_preflight_failures_do_not_move_or_send_stop(config_path, tmp_path, mode):
    status, report, device = execute(config_path, tmp_path, mode)
    assert status == 1 and report['joint_command_attempts'] == 0
    assert not any(c[0] in ('joint', 'speed', 'stop') for c in device.calls)


@pytest.mark.parametrize('mode', ['timeout', 'duplicate', 'epoch_reset', 'feedback_fault', 'partial_send', 'interrupt'])
def test_fault_or_cancel_after_target_stops_without_operating_gripper(config_path, tmp_path, mode):
    original = signal.getsignal(signal.SIGINT)
    status, report, device = execute(config_path, tmp_path, mode, extra=['--timeout-s', '.3'])
    assert status == 1 and report['joint_command_attempts'] == 1
    assert report['electronic_stop_sent'] and device.calls.count(('stop',)) == 1
    assert signal.getsignal(signal.SIGINT) == original
    if mode == 'interrupt': assert report['status'] == 'cancelled'
    if mode == 'partial_send': assert report['joint_target_commands'] == 0


@pytest.mark.parametrize('value', [[10.]*7, [float('nan')]*7, [0.]*6])
def test_bad_wait_target_is_rejected_before_hardware(config_path, value):
    data = yaml.safe_load(config_path.read_text()); data['standby_joint_rad'] = value
    config_path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(config_path)]) == 2


def test_existing_log_directory_is_not_overwritten(config_path, tmp_path):
    output = tmp_path/'run'; output.mkdir()
    assert main(['--config', str(config_path), '--execute', '--output', str(output)],
                hardware_factory=lambda *a: pytest.fail('must not connect')) == 2
    assert list(output.iterdir()) == []


@pytest.mark.parametrize('section', ['hardware', 'pick_settings'])
def test_invalid_config_mapping_does_not_connect(config_path, section):
    data = yaml.safe_load(config_path.read_text()); data[section] = None
    config_path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(config_path), '--execute'],
                hardware_factory=lambda *a: pytest.fail('must not connect')) == 2


@pytest.mark.parametrize('extra', [['--speed-percent', '0'], ['--speed-percent', '101'],
    ['--timeout-s', 'nan'], ['--timeout-s', '0']])
def test_invalid_cli_limits_do_not_connect(config_path, extra):
    assert main(['--config', str(config_path), '--execute', *extra],
                hardware_factory=lambda *a: pytest.fail('must not connect')) == 2


def test_invalid_yaml_is_reported_without_constructing_adapter(config_path):
    config_path.write_text('standby_joint_rad: [\n')
    assert main(['--config', str(config_path), '--execute'],
                hardware_factory=lambda *a: pytest.fail('must not connect')) == 2

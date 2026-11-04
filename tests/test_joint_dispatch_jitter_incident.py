"""Offline regression from the 10:52:57 J5 dispatch-timing incident."""
from types import SimpleNamespace
import numpy as np
import pytest
from nero_core.two_second.command_history import CommandHistory
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def history_at_last_command(*, timestamp, velocity, previous_interval, enforce=False):
    config = SimpleNamespace(velocity_limits=np.array([2., 2., 2., 2., 1.4, 2., 2.]),
                             acceleration_limits=np.full(7, 8.))
    if enforce:
        config.raw = {'runtime_command_acceleration_check_enabled': True}
    history = CommandHistory(config)
    history.q = np.zeros(7)
    history.dispatch = timestamp
    history.velocity = np.array([0., 0., 0., 0., velocity, 0., 0.])
    history.previous_interval = previous_interval
    return history


def test_explicit_opt_in_rejects_recorded_command_acceleration():
    interval = .007360684990999289
    previous_interval = 2*.009839318496233318-interval
    previous_velocity = -.5023608179810619
    candidate_velocity = -.6165501196232851
    target = np.zeros(7)
    target[4] = candidate_velocity*interval
    history = history_at_last_command(timestamp=1., velocity=previous_velocity,
                                      previous_interval=previous_interval, enforce=True)
    with pytest.raises(PickRejected, match='加速度') as error:
        history.validate(target, 1.+interval)
    assert error.value.details['discrete_acceleration_rad_s2'][4] == pytest.approx(-11.6054075987)
    assert abs(error.value.details['discrete_velocity_rad_s'][4]) < 1.4


def test_sample_time_difference_explains_smooth_curve_but_not_dispatch_safety():
    actual_interval = .007360684990999289
    sample_interval = .008364214998437092
    previous_sample_interval = .011569264774210751
    previous_sample_velocity = -.5348858520487396
    candidate = np.zeros(7)
    candidate[4] = -.6165501196232851*actual_interval
    history = history_at_last_command(timestamp=1., velocity=previous_sample_velocity,
                                      previous_interval=previous_sample_interval)
    # Different host/sample intervals explain the historical command estimate;
    # neither clock supplies a physical actuator acceleration measurement.
    history.validate(candidate, 1.+sample_interval)


def test_sender_defers_final_metadata_copy_and_validates_before_device_send():
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender

    class Clock:
        now = 1.
        def __call__(self): return self.now
        def sleep(self, duration): self.now += max(duration, 0.)
    clock = Clock()
    observed = {}

    class ExpensiveCopy:
        def __deepcopy__(self, memo):
            clock.sleep(.001)
            return self
    class Device:
        def snapshot(self): return {}
        def move_joints(self, q): observed['send'] = clock()
    device = Device()
    def step(*args):
        return None if 'send' in observed else (np.zeros(7), {})
    def finalize(state, stamp, dt, q, metadata):
        observed['sample'] = stamp
        clock.sleep(.0001)
        return q, {'expensive': ExpensiveCopy()}
    def validate(state, stamp, dt, q, metadata):
        observed['validation'] = stamp
        clock.sleep(.0001)
    sender = JointStreamSender(device, step, control_hz=100, clock=clock, sleep=clock.sleep,
                               finalize_send=finalize, validate_dispatch=validate)
    sender.start(); sender.join()
    assert sender.snapshot().exception is None
    assert observed['validation'] > observed['sample']
    assert observed['send'] - observed['sample'] == pytest.approx(.0002)
    assert clock()-observed['send'] >= .001  # the copy still occurs after sending


def test_check_crossing_window_does_not_send_unchecked_sample():
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    class Clock:
        now = 1.
        def __call__(self): return self.now
        def sleep(self, duration): self.now += max(duration, 0.)
    clock = Clock(); sends = []; checks = []
    class Device:
        def snapshot(self): return {}
        def move_joints(self, q): sends.append((clock(), q))
    def step(state, stamp, dt):
        return np.full(7, len(checks)+1.), {
            'prepared_from_monotonic_s': stamp, 'prepared_until_monotonic_s': stamp+.01}
    def validate(*args):
        checks.append(clock())
        if len(checks) == 1: clock.sleep(.012649)
    def after(*args): sender.request_stop()
    sender = JointStreamSender(Device(), step, after, clock=clock, sleep=clock.sleep,
                               validate_dispatch=validate)
    sender.start(); sender.join(); result = sender.snapshot()
    assert result.exception is None
    assert len(sends) == 1 and sends[0][1] == [2.]*7
    assert len([event for event in result.events if event['event'] == 'joint_stream_window_retry']) == 1


def test_dispatch_validation_failure_prevents_send():
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    class Device:
        sent = 0
        def snapshot(self): return {}
        def move_joints(self, q): self.sent += 1
    device = Device()
    def reject(*args): raise PickRejected('验证拒绝')
    sender = JointStreamSender(device, lambda *args: (np.zeros(7), {}),
                               finalize_send=lambda state, t, dt, q, m: (q, m),
                               validate_dispatch=reject)
    sender.start(); sender.join()
    assert isinstance(sender.snapshot().exception, PickRejected)
    assert device.sent == 0


@pytest.mark.parametrize('enforce', [False, True])
def test_incident_at_actual_sender_boundary_observes_by_default(enforce):
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    interval = .007360684990999289
    previous_interval = 2*.009839318496233318-interval
    history = history_at_last_command(timestamp=1., velocity=-.5023608179810619,
                                      previous_interval=previous_interval, enforce=enforce)
    candidate = np.zeros(7)
    candidate[4] = -.6165501196232851*interval
    class Device:
        sent = 0
        def snapshot(self): return {}
        def move_joints(self, q): self.sent += 1
    device = Device()
    class Clock:
        now = 1.+interval
        def __call__(self): return self.now
        def sleep(self, duration): self.now += max(duration, 0.)
    clock = Clock()
    sender = JointStreamSender(device, lambda *args: None if device.sent else (candidate, {}),
        after_send=lambda stamp, dt, q, metadata: history.record(stamp, dt, q, metadata),
        clock=clock, sleep=clock.sleep,
        finalize_send=lambda state, t, dt, q, metadata: (q, metadata),
        validate_dispatch=lambda state, t, dt, q, metadata: history.validate(q, t))
    sender.start(); sender.join()
    error = sender.snapshot().exception
    if enforce:
        assert isinstance(error, PickRejected)
        assert error.details['discrete_acceleration_rad_s2'][4] == pytest.approx(-11.6054075987)
        assert device.sent == 0
    else:
        assert error is None and device.sent == 1
        report = history.diagnostics()
        assert report['last_command_acceleration_rad_s2'][4] == pytest.approx(-11.6054075987)
        assert report['commands_exceeding_planning_acceleration_limit'] == 1
        assert report['runtime_acceleration_check_enabled'] is False
        assert report['measured_actuator_acceleration'] is False


@pytest.mark.parametrize('config_name', ['two_second.hardware.legacy_observed.yaml',
    'two_second.hardware.sim_aligned_wait_candidate.yaml'])
def test_site_disables_runtime_acceleration_but_keeps_actual_speed_limit(config_name):
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    import json
    root = Path(__file__).parents[1]
    config = CellConfig.load(root/'src/nero_bringup/config'/config_name)
    assert config.raw['runtime_command_acceleration_check_enabled'] is False
    incident = root/'runs/two_second_hardware/fixed_580_20261031_112947_516425938/summary.json'
    if not incident.exists():
        pytest.skip('local hardware incident not available')
    details = json.loads(incident.read_text())['details']
    history = CommandHistory(config)
    history.q = np.zeros(7)
    history.dispatch = details['previous_dispatch_monotonic_s']
    history.velocity = np.array(details['previous_discrete_velocity_rad_s'])
    history.previous_interval = 2*details['acceleration_interval_s']-details['actual_interval_s']
    candidate = np.array(details['discrete_velocity_rad_s'])*details['actual_interval_s']
    history.validate(candidate, details['dispatch_monotonic_s'])
    candidate[4] = (config.velocity_limits[4]+.01)*details['actual_interval_s']
    with pytest.raises(PickRejected, match='速度超限'):
        history.validate(candidate, details['dispatch_monotonic_s'])


@pytest.mark.parametrize('raw', [None, {}, {'runtime_command_acceleration_check_enabled': False}])
def test_first_send_does_not_infer_physical_acceleration_from_nominal_period(raw):
    config = SimpleNamespace(velocity_limits=np.full(7, 2.), acceleration_limits=np.full(7, 2.),
        settings=SimpleNamespace(control_hz=100))
    if raw is not None:
        config.raw = raw
    history = CommandHistory(config)
    history.validate_first(np.full(7, .01), np.zeros(7))
    with pytest.raises(PickRejected, match='首个发送点'):
        history.validate_first(np.full(7, .03), np.zeros(7))

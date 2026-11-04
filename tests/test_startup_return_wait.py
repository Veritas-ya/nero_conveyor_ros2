"""Pick-startup wiring with fake feedback only; no SDK/device connections."""
from pathlib import Path
from types import SimpleNamespace
from threading import Event
import numpy as np
import pytest

from nero_core.move_to_wait import MoveCancelled
from nero_core.two_second.startup import prepare_start_state
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


class Clock:
    now = 1.
    def __call__(self): return self.now
    def sleep(self, duration): self.now += duration


class Device:
    def __init__(self, clock, mode='success'):
        self.clock, self.mode, self.sent = clock, mode, None
        self.q = np.zeros(7)
        self.calls = []
        self.robot = SimpleNamespace(set_speed_percent=lambda v:self.calls.append(('speed',v)))
    def configure_joint_command_mode(self, mode): self.calls.append(('mode',mode))
    def move_joints(self, q):
        self.calls.append(('joint',q)); self.sent = self.clock()
        if self.mode == 'partial_send': raise RuntimeError('injected partial send')
        if self.mode != 'timeout': self.q = np.asarray(q)
    def stop(self): self.calls.append(('stop',))
    def snapshot(self):
        if self.sent is not None and self.clock()-self.sent >= .04:
            if self.mode == 'feedback_fault': raise RuntimeError('injected feedback fault')
            if self.mode == 'cancel': self.cancelled.set()
        stamp = self.clock()-.3 if self.mode == 'stale' else self.clock()
        state = dict(joints=self.q.copy(),force_n=0.,width_m=.1,joint_stamp=stamp)
        if self.mode == 'closed_after_return' and self.sent is not None: state['width_m'] = 0.
        if self.mode == 'epoch_reset':
            epoch = int(self.sent is not None and self.clock()-self.sent >= .04)
            state.update(joint_stamp_monotonic=stamp,feedback_time={key:{'clock_epoch':epoch,
                'mapping_quality':'mapped'} for key in ('joint_12','joint_34','joint_56','joint_7')})
        return state


@pytest.fixture
def startup():
    clock = Clock(); cancelled = Event(); report = {}; events = []
    cfg = SimpleNamespace(source=Path('TEST_ONLY.yaml'),source_sha256='TEST_ONLY',standby=np.ones(7)*.1,
        raw={'hardware':{'can_channel':'NO_DEVICE','firmware':'TEST_ONLY','joint_command_mode':'js'}},
        settings=SimpleNamespace(speed_percent=100,min_force_n=.2,open_width_m=.1))
    model = SimpleNamespace(command_lower=-np.ones(7),command_upper=np.ones(7))
    def run(device, initial=None):
        device.cancelled = cancelled
        return prepare_start_state(cfg,device,device.snapshot() if initial is None else initial,
            model=model,cancelled=cancelled,emit=lambda e,**d:events.append((e,d)),report=report,
            clock=clock,sleep=clock.sleep)
    return SimpleNamespace(clock=clock,cancelled=cancelled,report=report,events=events,
                           config=cfg,run=run)


def test_mismatch_uses_shared_move_then_restores_streaming_on_same_device(startup):
    device = Device(startup.clock)
    state = startup.run(device)
    np.testing.assert_allclose(state['joints'],startup.config.standby)
    assert [c[1] for c in device.calls if c[0]=='mode']==['j','js']
    assert sum(c[0]=='joint' for c in device.calls)==1
    assert ('speed',100) in device.calls and ('stop',) not in device.calls
    assert startup.report['status']=='wait_reached'
    assert startup.report['connection_reused'] and startup.report['hardware_connections']==0
    assert startup.report['gripper_commands']==0 and not startup.report['automatic_enable']
    assert startup.events[-1][0]=='startup_return_wait_passed'


def test_already_waiting_keeps_mode_speed_and_target_untouched(startup):
    device = Device(startup.clock); device.q = startup.config.standby.copy()
    startup.run(device)
    assert not device.calls
    assert startup.report['status']=='already_at_wait'


@pytest.mark.parametrize('kind,value,reason', [('force_n',.2,'startup_gripper_force'),
    ('force_n',float('nan'),'startup_gripper_force'),('width_m',0.,'startup_gripper_width'),
    ('width_m',float('nan'),'startup_gripper_width'),('joints',[float('nan')]*7,'startup_joint_feedback')])
def test_bad_gripper_or_joint_state_does_not_attempt_recovery(startup,kind,value,reason):
    device = Device(startup.clock); initial = device.snapshot(); initial[kind] = value
    with pytest.raises(PickRejected) as error: startup.run(device,initial)
    assert error.value.details['reason']==reason and not device.calls
    assert startup.report['status']=='stopped'


@pytest.mark.parametrize('mode', ['timeout','stale','epoch_reset','feedback_fault','partial_send','cancel'])
def test_failed_recovery_never_continues_as_ready(startup,mode):
    device = Device(startup.clock,mode)
    with pytest.raises((RuntimeError,TimeoutError)): startup.run(device)
    assert startup.events[-1][0]=='startup_return_wait_failed'
    assert not any(e=='startup_return_wait_passed' for e,_ in startup.events)
    assert ('mode','js') not in device.calls
    if mode == 'stale':
        assert not any(c[0] in ('joint','speed','stop') for c in device.calls)
    else:
        assert device.calls.count(('stop',))==1


def test_cancel_before_recovery_sends_nothing(startup):
    device = Device(startup.clock); startup.cancelled.set()
    with pytest.raises(MoveCancelled): startup.run(device)
    assert not device.calls and startup.report['status']=='cancelled'


def test_post_return_gripper_is_checked_again_before_readiness(startup):
    device = Device(startup.clock,'closed_after_return')
    with pytest.raises(PickRejected) as error: startup.run(device)
    assert error.value.details['reason']=='startup_gripper_width'
    assert startup.report['status']=='stopped'
    assert startup.events[-1][0]=='startup_return_wait_failed'


def test_wait_target_outside_model_bounds_never_sends(startup):
    startup.config.standby[0] = 2.
    device = Device(startup.clock)
    with pytest.raises(ValueError,match='关节范围'): startup.run(device)
    assert not device.calls

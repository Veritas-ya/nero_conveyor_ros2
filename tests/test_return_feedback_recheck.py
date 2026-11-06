"""Rejected feedback never dispatches; only new valid packets permit resuming."""
import numpy as np
import pytest

from nero_core.two_second.return_feedback import ReturnFeedbackCheck
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender, FeedbackRecheckRequired


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, amount): self.now += max(amount, .00001)


def mismatch():
    return PickRejected('当前关节FK与法兰反馈不符：6.54 mm / 0.97°', details={
        'reason': 'feedback_fk_mismatch', 'position_error_m': .00654,
        'position_tolerance_m': .005, 'orientation_tolerance_deg': 1.})


def test_same_packet_cannot_clear_error_but_new_valid_packet_can():
    clock = Clock(); checked = []
    def check(state):
        checked.append(state['pose_stamp'])
        if state.get('bad'): raise mismatch()
    gate = ReturnFeedbackCheck(check, clock, enabled=True)
    with pytest.raises(FeedbackRecheckRequired): gate.observe({'pose_stamp': 10., 'bad': True})
    clock.now += .005
    with pytest.raises(FeedbackRecheckRequired): gate.observe({'pose_stamp': 10.})
    assert checked == [10.]
    clock.now += .015
    gate.observe({'pose_stamp': 10.02})
    assert checked == [10., 10.02]
    assert gate.resumed[0]['targets_sent_from_rejected_feedback'] == 0
    assert gate.resumed[0]['original_details']['position_tolerance_m'] == .005


@pytest.mark.parametrize('valid_later', [True, False])
def test_sender_sends_no_target_until_valid_or_stops_at_original_deadline(valid_later):
    clock = Clock(); sent = []
    def check(state):
        if state['bad']: raise mismatch()
    gate = ReturnFeedbackCheck(check, clock, enabled=True)
    class Device:
        def snapshot(self):
            clock.sleep(.002)
            return {'pose_stamp': clock(), 'bad': not valid_later or clock() < 10.02}
        def move_joints(self, q): sent.append((clock(), q))
    def step(state, *args):
        gate.observe(state)
        return None if sent else (np.zeros(7), {})
    sender = JointStreamSender(Device(), step, clock=clock, sleep=clock.sleep).start()
    assert sender.join(2.)
    report = sender.snapshot()
    if valid_later:
        assert report.exception is None and len(sent) == 1
        assert sent[0][0] >= 10.02 and gate.resumed
    else:
        assert isinstance(report.exception, PickRejected)
        assert report.exception.details['position_tolerance_m'] == .005
        assert 10.03 <= clock() <= 10.04
        assert sent == []


def test_repeated_new_bad_packets_do_not_extend_the_deadline():
    clock = Clock(); gate = ReturnFeedbackCheck(lambda s: (_ for _ in ()).throw(mismatch()), clock, enabled=True)
    for delta in (0., .01, .02):
        clock.now = 10.+delta
        with pytest.raises(FeedbackRecheckRequired) as error: gate.observe({'pose_stamp': clock()})
        assert error.value.deadline_s == pytest.approx(10.03)
    clock.now = 10.031
    with pytest.raises(PickRejected): gate.observe({'pose_stamp': clock()})


def test_nonreturn_fk_failure_and_other_faults_remain_immediate():
    for enabled, error in ((False, mismatch()), (True, PickRejected('其他硬件错误'))):
        clock = Clock()
        def check(state): raise error
        gate = ReturnFeedbackCheck(check, clock, enabled=enabled)
        with pytest.raises(PickRejected) as failure: gate.observe({'pose_stamp': clock()})
        assert failure.value is error and gate.pending is None


def test_stream_integration_rechecks_return_and_keeps_all_sends_outside_bad_feedback():
    from dataclasses import replace
    from test_hardware_sim_selfcheck import moving_executor
    e, device, clock, path = moving_executor([0.])
    sent_at = []
    original_send = device.move_joints
    def send(q):
        sent_at.append(clock())
        return original_send(q)
    device.move_joints = send
    def check(state):
        if 1.4 <= clock() < 1.418: raise mismatch()
    e.planner.model.check_feedback = check
    e._stream(replace(path, stage='taught_return'), origin=1.1, epoch=0)
    assert not any(1.4 <= stamp < 1.418 for stamp in sent_at)
    assert any(row['event'] == 'return_feedback_rechecked' for row in e.events)
    np.testing.assert_allclose(e.command_history.q, .3, atol=1e-12)


def test_user_cancel_while_reading_new_feedback_sends_nothing():
    clock = Clock(); sent = []
    gate = ReturnFeedbackCheck(lambda s: (_ for _ in ()).throw(mismatch()), clock, enabled=True)
    class Device:
        def snapshot(self):
            clock.sleep(.002)
            return {'pose_stamp': clock()}
        def move_joints(self, q): sent.append(q)
    def step(state, *args):
        gate.observe(state)
        return np.zeros(7), {}
    sender = JointStreamSender(Device(), step, clock=clock, sleep=clock.sleep,
        pause_requested=lambda: clock() >= 10.01).start()
    assert sender.join(2.)
    assert not sent and sender.snapshot().paused

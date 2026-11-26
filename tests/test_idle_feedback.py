"""A completed idle pause cannot freshen cached CAN feedback or send motion."""
from types import SimpleNamespace as NS

import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.idle_feedback import refresh_after_collection


def fixture(read):
    now = [10.]
    events = []
    e = NS(sender=None, owner=(0, 1, 28), version=1, stage='returned', feedback_stamps={},
           clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0]+dt),
           device=NS(snapshot_nowait=lambda: read(now[0])),
           _active=lambda *args: None, _emit=lambda name, detail: events.append((name, detail)))
    return e, now, events


def sample(stamp):
    return dict(pose_stamp=stamp, joint_stamp=stamp, gripper_stamp=stamp)


def test_waits_for_packets_after_collection_without_commands_or_backdating():
    old = sample(9.86)
    e, now, events = fixture(lambda at: old if at < 10.04 else sample(at-.005))
    memory = dict(idle_collection_wall_s=.123582704)
    refresh_after_collection(e, memory, epoch=0)
    assert .04 <= now[0]-10. < .06
    assert old == sample(9.86)
    assert memory['post_collection_feedback']['robot_commands'] == 0
    assert events[-1][0] == 'idle_feedback_refreshed'
    assert e.feedback_stamps['pose'] >= 10.


def test_all_packet_groups_must_advance():
    def read(at):
        return dict(pose_stamp=at, joint_stamp=at, gripper_stamp=9.99)
    e, _, _ = fixture(read)
    with pytest.raises(PickRejected) as error:
        refresh_after_collection(e, dict(idle_collection_wall_s=.12), epoch=0)
    assert error.value.details['reason'] == 'post_collection_feedback_timeout'
    assert error.value.details['robot_commands'] == 0
    assert not e.feedback_stamps


def test_busy_or_missing_feedback_times_out_instead_of_moving():
    e, now, _ = fixture(lambda at: None)
    with pytest.raises(PickRejected):
        refresh_after_collection(e, dict(idle_collection_wall_s=.12), epoch=0)
    assert now[0] == pytest.approx(10.2)


@pytest.mark.parametrize('fault', ['cancel', 'device'])
def test_cancellation_and_device_fault_propagate(fault):
    def fail(*args):
        raise RuntimeError(fault)
    e, _, _ = fixture(fail if fault == 'device' else lambda at: None)
    if fault == 'cancel':
        e._active = fail
    with pytest.raises(RuntimeError, match=fault):
        refresh_after_collection(e, dict(idle_collection_wall_s=.12), epoch=0)


def test_short_collection_needs_no_device_read():
    e, _, _ = fixture(lambda at: pytest.fail('unexpected read'))
    refresh_after_collection(e, dict(idle_collection_wall_s=.001), epoch=0)
    assert e.stage == 'returned'


def test_does_not_refresh_with_a_live_sender():
    e, _, _ = fixture(lambda at: pytest.fail('unexpected read'))
    e.sender = object()
    with pytest.raises(RuntimeError, match='发送器'):
        refresh_after_collection(e, dict(idle_collection_wall_s=.12), epoch=0)

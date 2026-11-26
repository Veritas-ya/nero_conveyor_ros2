"""Distinguish absent CAN packets from decoded packets missing clock evidence."""
from threading import RLock
from types import SimpleNamespace as NS

import pytest

from mujoco_conveyor_sorting.runtime import nero_pick_hardware as hardware


def silent_adapter():
    device=hardware.Hardware.__new__(hardware.Hardware)
    device.lock=RLock()
    device.channel='fake0'
    device.execute=False
    device.feedback_wait_timeout_s=0.
    device._feedback_clock=object()  # Exercise the live mapping branch.
    device.robot=NS(_parser=NS(),has_comm_error=lambda:False,
        get_flange_pose=lambda:None,get_joint_angles=lambda:None,
        get_arm_status=lambda:None,get_driver_states=lambda index:None)
    device.gripper=NS(get_gripper_status=lambda:None)
    return device


def test_silent_live_bus_reports_absent_packets_instead_of_clock_mapping_missing():
    device=silent_adapter()
    with pytest.raises(hardware.FeedbackError,match='未收到任何机械臂关节/法兰反馈') as caught:
        device.snapshot()
    assert set(hardware.POSE_JOINT_PACKETS) <= set(caught.value.details['missing_packets'])
    assert not any('clock_mapping' in name for name in caught.value.details['missing_packets'])


def test_existing_decoded_packet_without_mapping_still_fails_closed():
    device=silent_adapter()
    device.robot._parser.joint_12=NS(timestamp=123.,msg=NS(joint_1=0.,joint_2=0.))
    with pytest.raises(hardware.FeedbackError) as caught:device.snapshot()
    assert 'joint_12:clock_mapping_missing' in caught.value.details['missing_packets']
    assert 'joint_34' in caught.value.details['missing_packets']


def test_invalid_mapping_on_existing_packet_is_not_reported_as_absent():
    device=silent_adapter()
    device.robot._parser.joint_12=NS(timestamp=123.,msg=NS(),feedback_time=NS(
        mapping_quality='ambiguous_epoch',mapped_monotonic=None))
    with pytest.raises(hardware.FeedbackError) as caught:device.snapshot()
    assert 'joint_12:clock_mapping_ambiguous_epoch' in caught.value.details['missing_packets']


@pytest.mark.parametrize('received',[False,True])
def test_passive_teaching_preserves_the_same_distinction(received):
    device=silent_adapter()
    if received:device.robot._parser.joint_12=NS(timestamp=123.,msg=NS())
    with pytest.raises(hardware.FeedbackPending) as caught:device.snapshot_teaching_joints()
    assert caught.value.details['missing_packets']==[
        'joint_12:clock_mapping_missing' if received else 'joint_12']

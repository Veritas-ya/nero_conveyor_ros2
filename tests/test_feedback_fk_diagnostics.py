"""Offline evidence capture and Euler/packet-time regression tests."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy, transform_distance
from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, RobotModel


def test_rpy_implementation_matches_fixed_xyz_including_near_gimbal_lock():
    for angles in ([.2, -.5, 2.], [-1.2, np.deg2rad(89.9), 2.4]):
        actual = pose_xyz_rpy([0, 0, 0, *angles])[:3, :3]
        np.testing.assert_allclose(actual, Rotation.from_euler('xyz', angles).as_matrix(), atol=1e-12)


def test_mixed_time_euler_packets_can_inflate_error_near_singularity():
    first = [0., 89.9, 0.]
    second = [90., 89.9, 90.]
    mixed = [first[0], second[1], second[2]]
    rotations = [Rotation.from_euler('xyz', x, degrees=True) for x in (first, second, mixed)]
    true_change = (rotations[0].inv()*rotations[1]).magnitude()*180/np.pi
    mixed_error = (rotations[0].inv()*rotations[2]).magnitude()*180/np.pi
    assert true_change < .15
    assert mixed_error > 89.


def test_fk_failure_keeps_original_inputs_and_newest_model_state():
    model = RobotModel()
    actual = np.array([.0476, .7563, -.0046, 1.7246, .0568, -.0726, .5109])
    paired = actual.copy(); paired[4] += .01
    model.set_state(paired, .1)
    flange = model.flange()
    r = Rotation.from_matrix(flange[:3, :3])*Rotation.from_euler('z', 1.24, degrees=True)
    pose = np.r_[flange[:3, 3], r.as_euler('xyz')]
    state = {'joints': actual, 'joints_at_pose_stamp': paired,
             'joints_at_pose_stamp_time': 1., 'width_m': .1, 'pose': pose,
             'pose_stamp': 1., 'pose_stamps_monotonic': [1., 1.01, 1.02],
             'feedback_timing': {'pose_joint_pairing': {'available': True},
                                 'pose_grouping': {'span_s': .001}}}
    with pytest.raises(PickRejected, match='法兰反馈不符') as info:
        model.check_feedback(state)
    details = info.value.details
    assert details['orientation_error_deg'] == pytest.approx(1.24)
    np.testing.assert_allclose(details['compared_joints_rad'], paired)
    np.testing.assert_allclose(details['feedback_flange_xyz_rpy_m_rad'], pose)
    np.testing.assert_allclose(details['fk_flange_transform'], flange)
    assert details['pose_packet_stamps_monotonic_s'] == [1., 1.01, 1.02]
    np.testing.assert_allclose(model.data.qpos[model.indices], actual)


def test_empirical_offset_pairs_only_measured_samples_and_keeps_pose_timestamp():
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    history = JointFeedbackHistory()
    history.pair(np.zeros(7), [1.]*4, 1., epoch=0)
    paired, report = history.pair(np.ones(7), [1.01]*4, 1.008,
                                  epoch=0, joint_receive_offset_s=-.002)
    np.testing.assert_allclose(paired, .6, atol=1e-12)
    assert report['pose_stamp'] == 1.008
    assert report['joint_interpolation_stamp'] == pytest.approx(1.006)
    assert report['joint_receive_offset_s'] == -.002
    assert report['extrapolated'] is False
    # The correction must never infer data outside real samples or reuse an epoch.
    assert history.pair(np.ones(7), [1.01]*4, 1.001, epoch=0,
                        joint_receive_offset_s=-.002)[0] is None
    assert history.pair(np.ones(7), [2.]*4, 2., epoch=1,
                        joint_receive_offset_s=-.002)[0] is None


@pytest.mark.parametrize('position_error_m', [0., .006])
def test_offset_fk_check_keeps_current_state_and_same_tolerances(position_error_m):
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    model = RobotModel()
    old = np.array([.0476, .7563, -.0046, 1.7246, .0568, -.0726, .5109])
    newer = old.copy(); newer[0] += .02
    h = JointFeedbackHistory()
    h.pair(old, [1.]*4, 1., epoch=0)
    paired, report = h.pair(newer, [1.01]*4, 1.008, epoch=0, joint_receive_offset_s=-.002)
    model.set_state(paired, .1); flange = model.flange()
    pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')]
    pose[0] += position_error_m
    state = {'joints': newer, 'joints_at_pose_stamp': paired,
             'joints_at_pose_stamp_time': 1.008,
             'joints_at_pose_joint_receive_time': report['joint_interpolation_stamp'],
             'pose_stamp': 1.008, 'width_m': .1, 'pose': pose,
             'feedback_timing': {'pose_joint_pairing': report}}
    if position_error_m:
        with pytest.raises(PickRejected) as failure: model.check_feedback(state)
        assert failure.value.details['position_tolerance_m'] == .005
        assert failure.value.details['orientation_tolerance_deg'] == 1.
        assert failure.value.details['joint_source'] == 'interpolated_with_receive_offset'
    else:
        model.check_feedback(state)
    np.testing.assert_allclose(model.data.qpos[model.indices], newer)


def test_trace_saves_pose_pairing_and_raw_packet_times():
    trace = ExecutionTrace(clock=lambda: 2., wall_clock=lambda: 1.)
    state = {'joints': list(range(7)), 'joint_stamp': 1., 'pose': [1., 2., 3., .1, .2, .3],
             'joints_at_pose_stamp': [x+.1 for x in range(7)],
             'pose_stamps_monotonic': [1., 1.01, 1.02],
             'feedback_timing': {'pose_joint_pairing': {'available': True},
                                 'pose_grouping': {'span_s': .001}}}
    trace.record_feedback(state, 1.9, 2.)
    arrays = trace.arrays()
    np.testing.assert_allclose(arrays['feedback_flange_pose_m_rad'][0], state['pose'])
    np.testing.assert_allclose(arrays['feedback_joints_at_pose_rad'][0], state['joints_at_pose_stamp'])
    np.testing.assert_allclose(arrays['feedback_pose_packet_monotonic_s'][0], state['pose_stamps_monotonic'])
    assert json.loads(arrays['feedback_pose_joint_pairing_json'][0])['available']
    assert json.loads(arrays['feedback_pose_grouping_json'][0])['span_s'] == .001


def test_passive_recorder_opens_read_only_and_saves_feedback(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import record_fk_receive_feedback as recorder
    events = []
    class Device:
        def __init__(self, channel, firmware, execute, offset):
            events.append(('construct', execute, offset))
        def start(self): events.append(('start',))
        def snapshot(self):
            events.append(('snapshot',))
            return {'joints': [0.]*7, 'joint_stamp': 1., 'pose': [0.]*6,
                    'joints_at_pose_stamp': [0.]*7, 'pose_stamps_monotonic': [1.]*3,
                    'feedback_timing': {'pose_joint_pairing': {'diagnostic_receive_time_offsets': [1]}}}
        def close(self): events.append(('close',))
    monkeypatch.setattr(recorder, 'Hardware', Device)
    monkeypatch.setattr(recorder.CellConfig, 'load', lambda p: SimpleNamespace(
        raw={'hardware': {'can_channel': 'fake', 'firmware': 'fake', 'joint_receive_offset_s': -.002}}))
    output = tmp_path/'feedback.npz'
    monkeypatch.setattr('sys.argv', ['record', '--config', 'fake', '--seconds', '.02',
                                    '--hz', '100', '--output', str(output)])
    recorder.main()
    assert events[0] == ('construct', False, -.002) and events[-1] == ('close',)
    data = np.load(output)
    assert len(data['feedback_q_rad']) >= 1
    assert json.loads(data['feedback_pose_joint_pairing_json'][0])['diagnostic_receive_time_offsets'] == [1]

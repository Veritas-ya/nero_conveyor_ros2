"""CAN receipt capture and no-send recovery retry; SDK is replaced with fakes."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected,RobotModel
from nero_core.two_second.recovery import _recovery_feedback,RecoveryFailed
from test_two_second_hardware import executor_fixture
from test_failed_attempt_recovery import recovery


def mismatch(stamp=1.,distance=.005010816193620153):
    return PickRejected('当前关节FK与法兰反馈不符',details={'reason':'feedback_fk_mismatch',
        'position_error_m':distance,'orientation_error_deg':.7128971408,
        'position_tolerance_m':.005,'orientation_tolerance_deg':1.,
        'joint_source':'interpolated_at_pose_receive_time','pose_joint_pairing':{'pose_stamp':stamp}})


def test_dense_receive_history_retains_samples_between_sparse_reader_polls():
    history=JointFeedbackHistory();sparse=JointFeedbackHistory()
    def q(t):return np.arange(1.,8.)*.5*t*t
    for t in np.arange(0.,.04001,.005):
        for axes in ((0,1),(2,3),(4,5),(6,)):
            history.observe_packet(axes,q(t)[list(axes)],t,epoch=0)
    sparse.pair(q(0.),[0.]*4,.027,epoch=0)
    old,_=sparse.pair(q(.04),[.04]*4,.027,epoch=0)
    current=q(.04);new,report=history.pair(current,[.04]*4,.027,epoch=0)
    assert np.linalg.norm(new-q(.027))<np.linalg.norm(old-q(.027))/20
    assert report['maximum_bracket_span_s']==pytest.approx(.005)
    assert report['received_joint_packets']==36 and report['history_source']=='can_receive_boundary'
    assert not report['extrapolated']
    np.testing.assert_array_equal(current,q(.04))


def test_packet_history_never_brackets_across_reset_or_outside_measurements():
    h=JointFeedbackHistory()
    for t in (1.,1.005):
        for axes in ((0,1),(2,3),(4,5),(6,)):h.observe_packet(axes,[t]*len(axes),t,epoch=0)
    assert h.pair([1.005]*7,[1.005]*4,1.002,epoch=0)[0] is not None
    assert h.pair([1.005]*7,[1.005]*4,1.006,epoch=0)[0] is None
    h.observe_packet((0,1),[0.,0.],2.,epoch=1)
    assert h.pair([2.]*7,[2.]*4,1.002,epoch=1)[0] is None


def test_real_adapter_guard_captures_all_four_can_joint_packet_types_without_reads(monkeypatch):
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware,JOINT_CAN_AXES
    from mujoco_conveyor_sorting.runtime.feedback_clock import FeedbackClock
    class Parser:
        def parse_packet(self,frame):
            return NS(msg=NS(**{f'joint_{i+1}':frame.stamp*(i+1) for i in JOINT_CAN_AXES[frame.arbitration_id]}))
    parser=Parser();comm=NS(_timeout=.2,send=lambda *a:pytest.fail('must never send'))
    robot=NS(_parser=parser,OPTIONS=NS(EFFECTOR=NS(AGX_GRIPPER=1)),
        init_effector=lambda _:NS(_parser=Parser()),create_comm=lambda:comm)
    monkeypatch.setitem(sys.modules,'pyAgxArm',NS(AgxArmFactory=NS(create_arm=lambda _:robot),
        create_agx_arm_config=lambda **kw:kw))
    hw=Hardware('NO_HARDWARE','fake',False)
    now=[1.]
    hw._feedback_clock=FeedbackClock(monotonic=lambda:now[0],wall=lambda:1000.+now[0])
    for t in (1.,1.005,1.01,1.015):
        now[0]=t
        for can_id in JOINT_CAN_AXES:
            robot._parser.parse_packet(NS(arbitration_id=can_id,timestamp=1000.+t,stamp=t))
    history=hw._joint_feedback_history
    assert history.received_packets==16 and hw._joint_history_capture_error is None
    actual,report=history.pair(np.arange(1.,8.)*1.015,[1.015]*4,1.0125,epoch=0)
    np.testing.assert_allclose(actual,np.arange(1.,8.)*1.0125,atol=1e-11)
    assert report['maximum_bracket_span_s']==pytest.approx(.005)


def test_recorded_five_point_zero_one_mm_mismatch_is_within_new_limit():
    model=RobotModel()
    paired=[.4670442113817257,.7498417113483694,.2070456633435994,1.7244604028096728,
        -.06206361604556415,.044557933179617655,.537102106699592]
    state={'joints':paired,'joints_at_pose_stamp':paired,'joints_at_pose_stamp_time':1.,'pose_stamp':1.,
        'width_m':.1,'pose':[-.292165,-.240073,.158195,-1.125335941808384,1.4114303127952943,-2.0488245556236233]}
    model.check_feedback(state)


def test_recovery_waits_for_new_pose_and_does_not_retry_the_identical_bad_sample():
    e,device,clock=executor_fixture();began=clock();checked=[]
    original=device.snapshot
    def read():
        state=original();state['pose_stamp']=began if clock()<began+.02 else clock()
        return state
    device.snapshot=read
    e.planner.model.check_feedback=lambda state:checked.append(state['pose_stamp'])
    state=_recovery_feedback(e,epoch=0,failure=mismatch(began))
    assert state['pose_stamp']>=began+.02 and checked==[state['pose_stamp']]
    assert device.commands==[]
    assert [row['event'] for row in e.events]==['recovery_feedback_wait','recovery_feedback_rechecked']


def test_coherent_feedback_does_not_skip_last_target_arrival_guard():
    e,device,clock=executor_fixture();began=clock()
    e.command_history.record(clock(),.01,np.zeros(7),{})
    original=device.snapshot
    def read():
        device.q[:]=.03 if clock()<began+.05 else 0.
        return original()
    device.snapshot=read
    state=_recovery_feedback(e,epoch=0)
    assert clock()>=began+.05
    np.testing.assert_array_equal(state['joints'],np.zeros(7))
    assert device.commands==[]


def test_recovery_uses_nonblocking_receive_cache_instead_of_sdk_wait():
    e,device,clock=executor_fixture()
    original=device.snapshot
    def forbidden():pytest.fail('recovery retry must not enter the SDK blocking receive wait')
    device.snapshot_nowait=lambda:original()
    device.snapshot=forbidden
    assert _recovery_feedback(e,epoch=0)['pose_stamp']==clock()
    assert device.commands==[]


@pytest.mark.parametrize('interrupt',['persistent_fk','cancel','epoch','device_fault'])
def test_recovery_retry_never_sends_or_masks_persistent_faults(interrupt):
    e,device,clock=executor_fixture();began=clock()
    def check(state):raise mismatch(state['pose_stamp'],.006)
    e.planner.model.check_feedback=check
    original_sleep=clock.sleep
    def sleep(dt):
        original_sleep(dt)
        if interrupt=='cancel':e.cancelled.set()
        if interrupt=='epoch':e.epoch=lambda:1
    e.sleep=sleep
    if interrupt=='device_fault':
        def fail():raise RuntimeError('driver fault')
        device.snapshot_nowait=fail
    expected=RuntimeError if interrupt=='device_fault' else PickRejected
    with pytest.raises(expected) as failure:_recovery_feedback(e,epoch=0)
    if interrupt=='persistent_fk':
        assert failure.value.details['reason']=='recovery_feedback_timeout'
        assert failure.value.details['last_fk_error']['position_error_m']==.006
        assert clock()-began==pytest.approx(.2)
    elif interrupt=='epoch':assert '重置' in str(failure.value)
    elif interrupt=='cancel':assert '取消' in str(failure.value)
    else:assert str(failure.value)=='driver fault'
    assert device.commands==[]


@pytest.mark.parametrize('persistent',[False,True])
def test_full_recovery_only_prepares_motion_after_consistent_feedback(recovery,persistent):
    r=recovery;began=r.clock()
    def check(state):
        if persistent or r.clock()<began+.025:raise mismatch(state['pose_stamp'],.006)
    r.e.planner.model.check_feedback=check
    if persistent:
        with pytest.raises(RecoveryFailed):r.e.run(r.target)
        assert [c[0] for c in r.device.commands]==['stop']
    else:
        result=r.e.run(r.target)
        assert result['returned_to_wait']
        assert r.device.commands[0][0]=='prepare' and r.device.commands[0][1]>=began+.025
        assert not any(c[0]=='stop' for c in r.device.commands)

"""TCP height constraints and observation-only feedback; never start a device."""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest

from nero_core.two_second.brake_height import HeightGuard, HeightViolation, axes_for
from nero_core.two_second.stationary_hold import StationaryHold
from test_failed_attempt_recovery import SyntheticSender
from test_two_second_hardware import executor_fixture
from test_synchronized_reference import make_reference


def state(z, stamp):
    return {'pose':[0.,0.,z,0.,0.,0.], 'pose_stamp':stamp}


@pytest.mark.parametrize('slope', [-.02,0.,.02])
def test_guarded_reference_preserves_contact_and_non_decreasing_two_heights(slope):
    old=replace(make_reference(velocity=np.array([.45,0.,.45*slope])),duration_s=1.1,
                prealign_s=1.1,brake_s=.6,follow_s=0.)
    ref=replace(old,brake_rise_m=.001,height_lead_s=.2,height_normal_world=(-slope,0.,1.))
    plane=axes_for([-slope,0.,1.])
    t=np.linspace(.9,1.7,401)
    pos=np.array([ref.sample(s)[0] for s in t])
    np.testing.assert_allclose(ref.sample(ref.duration_s)[0],old.contact_xyz,atol=1e-12)
    np.testing.assert_allclose(pos[:,:2],[old.sample(s)[0][:2] for s in t],atol=1e-12)
    assert np.min(np.diff(pos@plane.T,axis=0))>=-1e-12
    for bound in [.9,1.1]:
        left=ref.position_derivatives(bound-1e-8);right=ref.position_derivatives(bound+1e-8)
        for a,b in zip(left,right):np.testing.assert_allclose(a,b,atol=2e-5)
    np.testing.assert_allclose(ref.position_derivatives(1.7)[1:],0.,atol=1e-12)


def test_measured_downward_motion_is_only_recorded_and_does_not_set_command_floor():
    g=HeightGuard([0.,0.,1.],np.eye(4),lambda q:np.array([0.,0.,q[0]]))
    g.armed=True
    g.record_sent([.02]);g.observe(state(.02485,1.));g.check_command([.02])
    g.observe(state(.02267,1.142));g.check_command([.02])
    g.observe(state(.022,1.142));g.observe(state(.025,1.0))
    diag=g.feedback_report()
    assert diag['mode']=='observe_only' and not g.fault
    assert diag['maximum_base_z_drawdown_mm']==pytest.approx(2.18)
    assert diag['distinct_samples']==2 and diag['invalid_samples']==1


@pytest.mark.parametrize('q', [[0.,0.,.0199],[.01,0.,.02]])
def test_lower_command_in_either_frame_is_rejected_and_latches_one_stop(q):
    faults=[]
    g=HeightGuard([-.1,0.,1.],np.eye(4),lambda q:np.array(q),faults.append)
    g.record_sent([0.,0.,.02])
    for _ in range(2):
        with pytest.raises(HeightViolation):g.check_command(q)
    assert faults==[g.fault]


def test_protection_continues_during_stationary_hold_and_bad_anchor_stops_without_recovery():
    e,device,clock=executor_fixture()
    # Original command ended 2 mm higher than the proposed hold anchor.
    e.brake_height_guard=g=HeightGuard([0.,0.,1.],np.eye(4),lambda q:np.array([0.,0.,q[0]]),
                                        e.stop_for_height_fault)
    g.armed=True;g.record_sent([.002]);e.rest_anchor=np.zeros(7);e.motion_started=True
    def attempt(_):
        with StationaryHold(e,SyntheticSender,epoch=0,carrying=False) as hold: hold.check()
    e._run_attempt=attempt
    with pytest.raises(HeightViolation):e._run_with_recovery(None)
    assert [cmd[0] for cmd in device.commands]==['stop']
    assert e.cancelled.is_set() and e.height_fault_stop_sent
    assert any(evt['event']=='protected_grasp_stopped' for evt in e.events)


def test_lift_handoff_uses_successfully_sent_commands_and_never_feedback():
    g=HeightGuard([0.,0.,1.],np.eye(4),lambda q:np.array([0.,0.,q[0]]))
    g.record_sent([.02]);assert not g.lift_started()
    g.observe(state(.03,1.));assert not g.lift_started()
    g.record_sent([.0202]);g.observe(state(.029,1.1))
    assert g.lift_started() and not g.fault


def test_live_final_sampler_blocks_first_descending_packet_after_close(monkeypatch):
    from nero_core.two_second.trajectory import TimedPath
    from nero_core.two_second import brake_height,execution,streaming
    e,device,clock=executor_fixture()
    e.config.raw.update(brake_no_descent_rise_m=.001,closure_policy='scheduled_attempt')
    ref=replace(make_reference(),start_s=clock()+.12,duration_s=1.1,prealign_s=1.1,
                follow_s=0.,brake_s=.6,brake_rise_m=.001,height_lead_s=.2)
    e.planner.support=np.array([0.,0.,1.])
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    monkeypatch.setattr(execution,'JointStreamSender',SyntheticSender)
    monkeypatch.setattr(brake_height,'TcpFK',lambda model:lambda q:np.array([0.,0.,q[0]]))
    monkeypatch.setattr(streaming,'sweep',lambda *a:None)
    e.planner.attach_measured=lambda *a:None
    path=TimedPath(lambda t,d=0: np.r_[(-.0001 if t>1.16 and not d else 0.),np.zeros(6)],
        e.owner,e.planner.version,'smooth_intercept_follow_brake',ref.full_duration_s,0.)
    with pytest.raises(PickRejected,match='高度|认证|下移'):
        e._stream(path,origin=ref.start_s,epoch=0,reference=ref)
    assert not device.commands
    sent=[]
    original=device.move_joints
    def record(q):
        sent.append(q.copy())
        original(q)
    monkeypatch.setattr(device,'move_joints',record)
    hz=e.config.settings.control_hz
    path.diagnostics.update(reference_check={'no_descent':dict(start_s=ref.duration_s-e.config.raw['command_advance_s']-1/hz,
                            end_s=path.duration,rise_m=.001,check_hz=hz,normal=[0.,0.,1.])})
    with pytest.raises(HeightViolation):
        e._stream(path,origin=ref.start_s,epoch=0,reference=ref)
    assert any(c[0]=='grip' for c in device.commands)
    assert e.height_fault_stop_sent and device.commands[-1][0]=='stop'
    assert sent and np.array_equal(np.array(sent),np.zeros((len(sent),7)))

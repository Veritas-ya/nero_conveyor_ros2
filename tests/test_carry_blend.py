"""Continuous corner motion and release triggered by measured endpoint arrival."""
from dataclasses import replace
from concurrent.futures import Future

import numpy as np
import pytest
from scipy.interpolate import BPoly, PPoly

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.carry_blend import blend_curve, blend_checked_path
from nero_core.two_second.feedback_gates import RestArrival
from nero_core.two_second.prepared_cycle import concatenate
from nero_core.two_second.trajectory import TimedPath, extrema
from test_prepared_taught_cycle import fixture
from test_two_second_hardware import executor_fixture


def path_fixture():
    cfg,p,q=fixture();cfg.raw['taught_joint_jerk_limit_rad_s3']=500.
    states=[q(0.),q(.15),q(.4)]
    states[2][1]=.08
    parts=[]
    for start,end in zip(states[:-1],states[1:]):
        curve=PPoly.from_bernstein_basis(BPoly.from_derivatives([0.,1.],
            [[start,np.zeros(7),np.zeros(7)],[end,np.zeros(7),np.zeros(7)]]))
        parts.append(TimedPath(curve,p.identity,1,'part',1.,0.))
    return p,concatenate(parts,identity=p.identity,version=1,stage='carry')


@pytest.mark.parametrize('mode',['incoming','bisector','outgoing'])
def test_blend_preserves_waypoint_endpoints_and_qva_continuity(mode):
    p,path=path_fixture();curve=blend_curve(path.curve,[(1.,mode)],.8)
    for t in [0.,.6,1.,1.4,2.]:
        np.testing.assert_allclose(curve(t),path.curve(t),atol=1e-12)
    for t in [0.,.6,1.4,2.]:
        for d in [1,2]:np.testing.assert_allclose(curve(t,d),path.curve(t,d),atol=1e-11)
    assert np.max(abs(curve(1.,1)))>.01
    if mode=='incoming':assert abs(curve(1.,1)[1])<1e-12
    for t in curve.x[1:-1]:
        for d in [0,1,2]:np.testing.assert_allclose(curve(np.nextafter(t,-np.inf),d),curve(t,d),atol=1e-10)


def test_modified_curve_is_rechecked_and_derivative_extrema_respect_limits():
    p,path=path_fixture();p.config.acceleration_limits[:]=.1
    result=blend_checked_path(p,path,.04,[(1.,'incoming')],cancelled=lambda:False)
    assert ('check','carry',True) in p.calls
    r=result.diagnostics['corner_blend'];assert r['applied'] and r['geometry_rechecked']
    assert r['time_scale']>1.
    np.testing.assert_allclose(result.q0,path.q0,atol=1e-12)
    np.testing.assert_allclose(result.q1,path.q1,atol=1e-12)
    assert np.max(abs(np.asarray(extrema(result.curve,2))))<=.1+1e-9
    assert np.max(abs(np.asarray(extrema(result.curve,3))))<=500.+1e-9


def test_lift_patch_does_not_overwrite_subsequent_route_waypoints():
    p,path=path_fixture()
    curve=blend_curve(path.curve,[(1.,'incoming')],.2,protected_times=[1.15])
    for t in [1.15,1.2,1.4]:
        for d in [0,1,2]:np.testing.assert_allclose(curve(t,d),path.curve(t,d),atol=1e-11)


def test_new_collision_keeps_original_curve_and_records_reason():
    p,path=path_fixture()
    def reject(*args,**kwargs):raise PickRejected('bin rim')
    p.check_path=reject
    result=blend_checked_path(p,path,.04,[(1.,'incoming')],cancelled=lambda:False)
    assert result.curve is path.curve
    assert not result.diagnostics['corner_blend']['applied']
    assert len(result.diagnostics['corner_blend']['failed_attempts'])==3
    with pytest.raises(PickRejected,match='取消'):
        blend_checked_path(p,path,.04,[(1.,'incoming')],cancelled=lambda:True)


def test_release_arrival_needs_fresh_after_endpoint_in_tolerance_feedback():
    gate=RestArrival(3.,dwell_s=0.)
    q=np.zeros(7)
    assert not gate.observe(q,q,2.99)
    assert not gate.observe(q,q,2.99)
    assert not gate.observe(q+.1,q,3.)
    assert gate.observe(q,q,3.01)
    with pytest.raises(PickRejected):gate.observe(q,q,2.9)


def test_drop_stream_has_no_extra_dwell_but_other_streams_keep_it():
    finishes=[]
    for release in [False,True]:
        e,device,clock=executor_fixture()
        path=TimedPath(lambda t,d=0:np.zeros(7),e.owner,1,'drop',.2,0.)
        e._stream(path,origin=1.1,epoch=0,release_on_arrival=release)
        finishes.append(clock())
    assert finishes[0]-finishes[1]>=.09
    assert finishes[1]>=1.3


def test_release_opens_before_any_return_preparation():
    from nero_core.two_second.release import release_at_rest
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    e,dev,clock=executor_fixture()
    e.planner.validate_release_opening=lambda s:None
    e.planner.release_measured=lambda s:None
    e.holding.held_width=.04;dev.width=.04
    called=[]
    def prepare():
        assert ('grip',.1) in dev.commands
        called.append(True);future=Future();future.set_result(None);return future
    result=release_at_rest(e,dev.snapshot(),epoch=0,sender_factory=JointStreamSender,prepare_return=prepare)
    assert called and result['width_m']==.1

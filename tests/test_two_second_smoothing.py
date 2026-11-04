"""Offline reference/ordering tests; never certify a physical grasp."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.two_second_grasp.smooth_reference import SmoothDiagnosticReference
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.trajectory import sampling_times
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, PickSettings


def reference():
    return SmoothDiagnosticReference(1.,2.,2.,[.1,.2,.03],[.05,0.,0.],[0.,0.,.2],
        [1.,0.,0.,0.],Rotation.from_euler('z',.5).as_quat()[[3,0,1,2]])


def test_diagnostic_reference_has_rest_start_matched_contact_and_rest_end():
    r=reference()
    p,v,a=r.position_derivatives(0.)
    np.testing.assert_allclose(p,r.start_tcp_world)
    np.testing.assert_allclose([v,a],np.zeros((2,3)),atol=1e-12)
    p,v,a=r.position_derivatives(2.)
    np.testing.assert_allclose(p,r.contact_xyz)
    np.testing.assert_allclose(v,r.velocity_world,atol=1e-12)
    np.testing.assert_allclose(a,np.zeros(3),atol=1e-12)
    _,v,a=r.position_derivatives(r.full_duration_s)
    np.testing.assert_allclose([v,a],np.zeros((2,3)),atol=1e-12)


@pytest.mark.parametrize('boundary',[2.,2.6])
def test_diagnostic_reference_position_velocity_acceleration_are_continuous(boundary):
    r=reference()
    for before,after in zip(r.position_derivatives(boundary-1e-7),r.position_derivatives(boundary+1e-7)):
        np.testing.assert_allclose(before,after,atol=1e-6)
    a=Rotation.from_quat(r.sample(boundary-1e-7)[1][[1,2,3,0]])
    b=Rotation.from_quat(r.sample(boundary+1e-7)[1][[1,2,3,0]])
    assert (a.inv()*b).magnitude() < 1e-6


def test_diagnostic_reference_explicitly_changes_the_straight_spatial_path():
    r=reference()
    p=r.sample(1.)[0]
    chord=r.contact_xyz-r.start_tcp_world
    distance=np.linalg.norm(np.cross(p-r.start_tcp_world,chord))/np.linalg.norm(chord)
    assert distance > .001


def test_sampling_does_not_create_machine_epsilon_intervals_at_phase_boundaries():
    # arange produces 4.6000000000000005 while the phase is 4.6.
    t=sampling_times(6.1,.01,4.,4.6)
    assert min(np.diff(t)) > .009999999
    assert np.count_nonzero(t==4.6)==1
    assert t[0]==0. and t[-1]==6.1


def test_sampling_also_merges_equivalent_explicit_boundaries():
    t=sampling_times(2.,.01,1.2,1.1+.1)
    assert min(np.diff(t)) > .009999999


def fixture(monkeypatch):
    model=SimpleNamespace(check_feedback=lambda state:None,tcp=lambda:np.eye(4),
        command_lower=-np.ones(7),command_upper=np.ones(7),
        ik=SimpleNamespace(config=SimpleNamespace(position_tolerance=.0003,orientation_tolerance=.01)))
    cfg=SimpleNamespace(settings=PickSettings(speed_percent=100),direction=np.array([1.,0.,0.]),
        workspace=np.array([[-1.,1.]]*3),follow_bounds=np.array([0.,1.]),
        velocity_limits=np.ones(7)*1000,acceleration_limits=np.ones(7)*.15,
        raw={'planning_reserve_s':1.,'command_advance_s':.1,'planning_budget_s':8.,'freeze_upstream_distance_m':.5})
    planner=MeasuredPlanner(cfg,model=model,clock=lambda:1.)
    pose=np.eye(4);pose[0,3]=-.4
    target=SimpleNamespace(identity=(0,1,2),capture_s=1.,center=np.array([-.4,0,0]),
        velocity=np.array([.1,0,0]),grasps=tuple(pose.copy() for _ in range(4)))
    return planner,model,target


def test_dynamically_rejected_candidates_never_pay_for_scene_preparation(monkeypatch):
    planner,model,target=fixture(monkeypatch)
    solved=[]
    def solve(*a,**kw):
        solved.append(1)
        return SimpleNamespace(success=True,joint_positions=np.ones(7)*.1)
    model.solve_pose=solve
    def forbidden(*a,**kw): pytest.fail('failed q/v/a must not prepare or certify a scene')
    monkeypatch.setattr(planner,'install',forbidden)
    monkeypatch.setattr(planner,'check_path',forbidden)
    with pytest.raises(PickRejected) as failure:
        planner.intercept(target,{'joints':np.zeros(7)})
    assert len(failure.value.details['failures'])==2  # only short-axis symmetric candidates
    assert all(f['details']['derivative']==2 for f in failure.value.details['failures'])
    assert not failure.value.details['scene_prepared']
    assert len(solved)==2*260
    assert len(planner.diagnostics()['completed_attempts'])==2


def test_scene_is_still_required_reused_within_task_and_not_across_targets(monkeypatch):
    planner,model,target=fixture(monkeypatch)
    model.solve_pose=lambda *a,**kw:SimpleNamespace(success=True,joint_positions=np.zeros(7))
    installed=[];geometry=[]
    def install(target,state):
        installed.append(target.identity)
        planner.identity=target.identity
        planner.version+=1
    def check(path,*a,**kw):
        assert installed[-1]==path.identity
        geometry.append(path.identity)
        if len(geometry)==1: raise PickRejected('injected candidate geometry failure')
    monkeypatch.setattr(planner,'install',install)
    monkeypatch.setattr(planner,'check_path',check)
    result=planner.intercept(target,{'joints':np.zeros(7)})
    assert len(result.failures)==1 and len(installed)==1 and len(geometry)==2
    target.identity=(0,1,3)
    planner.intercept(target,{'joints':np.zeros(7)})
    assert installed==[(0,1,2),(0,1,3)]


def test_common_scene_failure_is_not_retried_for_every_grasp(monkeypatch):
    planner,model,target=fixture(monkeypatch)
    model.solve_pose=lambda *a,**kw:SimpleNamespace(success=True,joint_positions=np.zeros(7))
    calls=[]
    def fail(*a):
        calls.append(1)
        raise PickRejected('injected common scene error')
    monkeypatch.setattr(planner,'install',fail)
    with pytest.raises(PickRejected,match='injected common scene error'):
        planner.intercept(target,{'joints':np.zeros(7)})
    assert len(calls)==1

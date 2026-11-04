"""Failure evidence without real devices or changes to route acceptance."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.route_diagnostics import RouteDiagnostics
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


CONFIG=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


@pytest.mark.parametrize('bad_role', ['start','goal'])
def test_teach_connection_records_which_endpoint_failed(monkeypatch,bad_role):
    cfg=CellConfig.load(CONFIG);p=MeasuredPlanner(cfg);p.identity=(0,4,2)
    p.payload_local=np.zeros((4,3))
    p.scene_report={'scope':'measured_belt_surface_only','frame_sequence':27,'capture_monotonic_s':10.}
    q0=cfg.standby.copy();goal=q0.copy();goal[0]+=.03
    def valid(q,width,**kwargs):
        is_bad=np.array_equal(q,q0 if bad_role=='start' else goal)
        if is_bad:
            raise PickRejected('载物点云与实测场景碰撞',details={
                'kind':'payload_scene_collision','minimum_distance_m':.002,
                'payload_points_base_m':[[.1,.2,.3]],'nearest_scene_points_base_m':[[.1,.2,.302]]})
    monkeypatch.setattr(p,'valid',valid)
    with pytest.raises(PickRejected) as caught:
        p.route({'joints':q0,'width_m':.04},stage='transfer_connect',goal_joints=goal,payload=True)
    d=caught.value.details['route_diagnostics']
    assert d['outcome']=='no_certified_route' and d['goal_attempts'][0]['outcome']==bad_role+'_rejected'
    example=d['rejection_examples'][0]
    assert example['role']==bad_role and example['details']['kind']=='payload_scene_collision'
    np.testing.assert_array_equal(example['joints_rad'],q0 if bad_role=='start' else goal)
    assert d['payload'] and d['scene_frame_sequence']==27
    assert d['operation_timings_s']['path_search']>=0


def test_curve_check_failure_preserves_geometry_reason(monkeypatch):
    cfg=CellConfig.load(CONFIG);p=MeasuredPlanner(cfg);p.identity=(0,1,2)
    q0=cfg.standby.copy();goal=q0.copy();goal[0]+=.01
    monkeypatch.setattr(p,'valid',lambda *a,**kw:None)
    def fail(*a,**kw):
        raise PickRejected('曲线路径碰撞',details={'kind':'scene_collision','geometry':'link6'})
    monkeypatch.setattr(p,'check_path',fail)
    with pytest.raises(PickRejected) as caught:
        p.route({'joints':q0,'width_m':.1},stage='return_connect',goal_joints=goal)
    d=caught.value.details['route_diagnostics']
    assert d['goal_attempts'][0]['outcome']=='path_found'
    assert d['curve_failures'][0]['details']['geometry']=='link6'
    assert d['operation_timings_s']['curve_geometry']>=0


def test_ik_failures_have_bounded_evidence(monkeypatch):
    cfg=CellConfig.load(CONFIG);p=MeasuredPlanner(cfg);p.identity=(0,1,2)
    class Solver:
        def __init__(self,model):pass
        def solve(self,*args,**kwargs):
            return SimpleNamespace(success=False,position_error=.012,orientation_error=.1,message='no convergence')
    monkeypatch.setattr('nero_core.two_second.precise_ik.PrecisePoseSolver',Solver)
    with pytest.raises(PickRejected) as caught:
        p.route({'joints':cfg.standby,'width_m':.1},stage='transfer',goal_pose=np.eye(4))
    d=caught.value.details['route_diagnostics']
    assert d['outcome']=='no_ik_solution' and d['ik_attempts']==d['ik_failures']==9
    assert len(d['ik_failure_examples'])==3 and d['ik_failure_examples'][0]['position_error_m']==.012


def test_repeated_rejections_do_not_create_unbounded_examples():
    d=RouteDiagnostics(stage='test',identity=(0,1,2),payload=False,start=np.zeros(7),
        measured=np.zeros(7),width=.1,goal_joints=np.ones(7),goal_pose=None,
        scene_revision=1,scene_report=None,budget_s=1.)
    for i in range(100):
        d.checked('path',np.zeros(7),PickRejected('模拟拒绝',details={'kind':'same'}))
    report=d.snapshot(outcome='no_certified_route',elapsed_s=.1,cancelled=False,timed_out=False)
    assert report['rejection_counts']['path | same']==100 and len(report['rejection_examples'])==1


def test_payload_scene_collision_records_both_sides_of_nearest_pair():
    from scipy.spatial import cKDTree
    p=MeasuredPlanner.__new__(MeasuredPlanner)
    p.stats={'states':0,'payload_states':0};p.config=SimpleNamespace(raw={},workspace=np.array([[-1.,1.]]*3))
    scene=np.array([[0.,0.,0.]])
    tcp=np.eye(4);tcp[:3,3]=[.1,0.,.1]
    # Payload point in base frame is 2 mm above the belt observation.
    p.payload_local=np.array([[-.1,0.,-.098]])
    p.model=SimpleNamespace(check_state=lambda *a,**kw:None,tcp=lambda:tcp,
        gripper_geometries=[],meshes={},scene=scene,tree=cKDTree(scene))
    p.support=np.array([0.,0.,1.]);p.support_origin=np.zeros(3)
    p.scene_report={'scope':'measured_belt_surface_only'}
    p.capture_tcp=np.eye(4);p.target=None;p.released_points=None
    with pytest.raises(PickRejected,match='载物点云') as caught:
        p.valid(np.zeros(7),.04,payload=True)
    d=caught.value.details
    assert d['kind']=='payload_scene_collision' and d['minimum_distance_m']==pytest.approx(.002)
    np.testing.assert_allclose(d['payload_points_base_m'],[[0.,0.,.002]],atol=1e-12)
    np.testing.assert_allclose(d['nearest_scene_points_base_m'],[[0.,0.,0.]])

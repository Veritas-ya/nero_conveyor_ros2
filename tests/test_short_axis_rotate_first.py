"""Tool closing-axis, synchronized rotation lead and same-exposure display."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nero_core.two_second.vision import obb_grasps
from nero_core.two_second.config import CellConfig
from nero_core.two_second.synchronized_reference import SynchronizedIntercept, schedule_forward_join
from pathlib import Path


CONFIG=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


def test_physical_tcp_closing_axis_is_always_short_obb_axis_on_tilted_belt():
    cfg=CellConfig.load(CONFIG)
    plane=np.eye(4);plane[:3,:3]=Rotation.from_euler('xyz',[.05,-.08,.3]).as_matrix()
    plane[:3,3]=[-.35,.1,-.14]
    local=np.array([[x,y,z] for x in np.linspace(-.015,.015,9)
        for y in np.linspace(-.045,.045,11) for z in (.02,.05)])
    grasps,box=obb_grasps(local@plane[:3,:3].T+plane[:3,3],plane,cfg.canonical_tcp)
    assert len(grasps)==2
    axes=plane[:3,:3]@np.asarray(box['axes_world_columns'])
    short=int(np.argmin(box['extent_m'][:2]));long=1-short
    for g in grasps:
        assert abs(g[:3,1]@axes[:,short])==pytest.approx(1.)
        assert g[:3,1]@axes[:,long]==pytest.approx(0.,abs=1e-12)
        np.testing.assert_allclose(g[:3,2],-plane[:3,2],atol=1e-12)
    assert Rotation.from_matrix(grasps[0][:3,:3].T@grasps[1][:3,:3]).magnitude()==pytest.approx(np.pi)


def test_schedule_reserves_rotation_and_translation_as_separate_phases():
    args=dict(capture_s=0.,grasp_position=[-.55,.03,.55],velocity=[.058,0.,0.],
        standby_position=[.04,0.,.7],direction=[1,0,0],now_s=.1,planning_reserve_s=2.5,
        advance_s=.1,join_s=2.,synchronized_s=.2,follow_s=.8,follow_bounds=[-.2,.28])
    schedule=schedule_forward_join(**args,rotate_first_s=2.)
    assert schedule['translation_start_s']-schedule['join_start_s']==pytest.approx(2.)
    assert schedule['sync_start_s']-schedule['translation_start_s']==pytest.approx(2.)
    assert schedule['join_start_s']>=.1+2.5+.1-1e-9
    quat=Rotation.from_euler('z',90,degrees=True).as_quat()[[3,0,1,2]]
    ref=SynchronizedIntercept(0.,schedule['join_start_s'],4.2,np.array(args['grasp_position']),
        np.array(args['velocity']),np.array(args['standby_position']),np.array([1.,0,0,0]),quat,
        .8,1.5,4.,rotate_first_s=2.)
    for t in np.linspace(0.,2.,51):
        p,v,a=ref.position_derivatives(t)
        np.testing.assert_allclose(p,ref.start_tcp_world,atol=1e-12)
        np.testing.assert_allclose([v,a],0.,atol=1e-12)
    for t in np.linspace(2.,ref.full_duration_s,51):
        assert abs(ref.sample(t)[1]@quat)==pytest.approx(1.)
    assert ref.contact_s==pytest.approx(schedule['contact_s'])


def test_production_planner_actually_passes_rotation_lead_to_synchronized_reference(monkeypatch):
    from test_two_second_smoothing import fixture
    p,model,target=fixture(monkeypatch)
    p.config.raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',
        synchronized_duration_s=.2,prealign_orientation_s=2.,rotate_first_s=2.,
        approach_duration_candidates_s=[2.],brake_duration_s=1.5,candidate_planning_budget_s=2.)
    pose=np.eye(4);pose[:3,3]=target.center
    pose[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    target.grasps=(pose,)
    class Solver:
        def __init__(self,model):pass
        def solve(self,*a,**kw):return NS(success=True,joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver',Solver)
    monkeypatch.setattr(p,'install',lambda *a:None)
    monkeypatch.setattr(p,'check_smooth_intercept',lambda *a,**kw:{'offline_test':True})
    plan=p.intercept(target,{'joints':np.zeros(7)})
    ref=plan.reference
    assert ref.rotate_first_s==2. and ref.prealign_s==4.
    assert plan.path.diagnostics['join_s']==2.
    assert plan.path.diagnostics['grasp_axis_policy']=='obb_short_axis_only'
    np.testing.assert_allclose(ref.sample(1.)[0],ref.start_tcp_world)
    np.testing.assert_allclose(ref.sample(2.)[0],ref.start_tcp_world)
    assert abs(ref.sample(2.)[1]@ref.target_quaternion_wxyz)==pytest.approx(1.)


def test_rgb_and_cloud_grasps_use_the_selected_pose_from_the_same_exposure():
    from nero_core.two_second.camera_view import display_grasp,collision_predicted_box
    first=np.eye(4);selected=np.eye(4);selected[:3,:3]=Rotation.from_euler('z',np.pi).as_matrix()
    geometry={'source_sequence':7,'grasps':(first,selected),'obb':{
        'center_world_m':[1.,2.,3.],'axes_world_columns':np.eye(3).tolist(),'extent_m':[.03,.09,.04]}}
    plan={'metadata':{'frame_sequence':7},'selected_grasp_base':selected}
    np.testing.assert_array_equal(display_grasp(geometry,0,plan),selected)
    plan['metadata']['frame_sequence']=8
    np.testing.assert_array_equal(display_grasp(geometry,0,plan),first)
    predicted=collision_predicted_box(geometry,{'frame_sequence':7,'capture_monotonic_s':10.,
        'predicted_monotonic_s':12.,'target_velocity_base_m_s':[.1,0,0]})
    np.testing.assert_allclose(predicted['obb']['center_world_m'],[1.2,2.,3.])
    assert geometry['obb']['center_world_m']==[1.,2.,3.]


@pytest.mark.parametrize('duration',[0.,2.])
def test_console_does_not_call_concurrent_rotation_a_rotation_lead(duration):
    from nero_core.two_second.console import ConsoleProgress
    lines=[];console=ConsoleProgress(write=lines.append)
    def event(name,data):console.observe({'event':'execution_event','monotonic_s':1.,
        'data':{'record':{'event':name,'data':data}}})
    event('planning_submitted',{'start_tcp_base':np.eye(4).tolist()})
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    event('planning_delivery',{'selected_grasp_tcp_base':pose.tolist(),'planning_wall_s':.1,
        'trajectory_diagnostics':{'intercept_strategy':'synchronized','prealign_s':2.,'rotate_first_s':duration}})
    assert any(('原位旋转' if duration else '过程中计划旋转') in line for line in lines)


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_rotation_lead_duration_is_rejected_before_device_start(value,tmp_path):
    import yaml
    raw=yaml.safe_load(CONFIG.read_text())
    for key in ('camera_calibration_file','tcp_calibration_file'):
        raw[key]=str((CONFIG.parent/raw[key]).resolve())
    raw['rotate_first_s']=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='rotate_first_s'):CellConfig.load(path)


def test_adaptive_rotation_uses_existing_joint_envelope_without_delivering_a_partial_path():
    from nero_core.two_second.rotation_lead import estimate_rotation_lead
    cfg=NS(raw={'rotate_first_s':.1,'planning_knot_hz':50.},
        settings=NS(control_hz=100),velocity_limits=np.full(7,.5),acceleration_limits=np.full(7,2.))
    model=NS(command_lower=-np.ones(7),command_upper=np.ones(7))
    start=np.eye(4);target=np.eye(4);target[:3,:3]=Rotation.from_euler('z',.5).as_matrix()
    positions=[]
    class Solver:
        def solve(self,xyz,quat,seed,*,cancelled):
            positions.append(np.asarray(xyz).copy())
            q=np.zeros(7);q[0]=Rotation.from_quat(quat[[1,2,3,0]]).as_rotvec()[2]
            return NS(success=True,joint_positions=q)
    estimate=estimate_rotation_lead(cfg,model,Solver(),{'joints':np.zeros(7)},start,target,
                                    cancelled=lambda:False)
    # A 0.5 rad rest-to-rest rotation cannot fit .1 s under these limits.
    assert estimate['duration_s']>1.
    assert estimate['duration_s']>=estimate['requested_s']
    np.testing.assert_allclose(positions, np.zeros((len(positions),3)))
    assert not estimate['geometry_certified'] and estimate['full_intercept_certification_required']
    assert 'path' not in estimate


def test_adaptive_rotation_cancellation_never_runs_ik():
    from nero_core.two_second.rotation_lead import estimate_rotation_lead
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    cfg=NS(raw={'rotate_first_s':.5,'planning_knot_hz':10.})
    class Solver:
        def solve(self,*a,**kw):pytest.fail('cancelled rotation must not run IK')
    with pytest.raises(PickRejected,match='取消'):
        estimate_rotation_lead(cfg,None,Solver(),{'joints':np.zeros(7)},np.eye(4),np.eye(4),
                               cancelled=lambda:True)


def test_planner_schedules_actual_retimed_rotation_and_checks_whole_path(monkeypatch):
    from test_two_second_smoothing import fixture
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    p,model,target=fixture(monkeypatch)
    p.config.raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',
        synchronized_duration_s=0.,prealign_orientation_s=2.,rotate_first_s=.5,
        rotate_first_retime_enabled=True,approach_duration_candidates_s=[2.],
        brake_duration_s=1.5,candidate_planning_budget_s=2.)
    pose=np.eye(4);pose[:3,3]=target.center
    pose[:3,:3]=Rotation.from_euler('z',.5).as_matrix();target.grasps=(pose,)
    class Solver:
        def __init__(self,model):pass
        def solve(self,*a,**kw):return NS(success=True,joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver',Solver)
    monkeypatch.setattr('nero_core.two_second.rotation_lead.estimate_rotation_lead',lambda *a,**kw:
        {'duration_s':.9,'requested_s':.5,'geometry_certified':False})
    monkeypatch.setattr(p,'install',lambda *a:None)
    checked=[]
    monkeypatch.setattr(p,'check_smooth_intercept',lambda path,ref,**kw:checked.append(ref))
    result=p.intercept(target,{'joints':np.zeros(7)})
    ref=result.reference
    assert checked==[ref]
    assert ref.rotate_first_s==pytest.approx(.9)
    assert ref.prealign_s==pytest.approx(2.9)
    assert ref.start_s>=p.clock()+p.config.raw['planning_reserve_s']+p.config.raw['command_advance_s']-1e-9
    assert ref.contact_s>=ref.start_s+.9+2.-1e-9
    for t in np.linspace(0.,.9,11):np.testing.assert_allclose(ref.sample(t)[0],ref.start_tcp_world)
    for t in np.linspace(.9,ref.duration_s,11):
        assert abs(ref.sample(t)[1]@ref.target_quaternion_wxyz)==pytest.approx(1.)
    # A valid duration estimate must never bypass the full geometry gate.
    def reject(*a,**kw):raise PickRejected('injected complete-path geometry rejection')
    monkeypatch.setattr(p,'check_smooth_intercept',reject)
    with pytest.raises(PickRejected) as error:p.intercept(target,{'joints':np.zeros(7)})
    assert error.value.details['failures'][0]['failed_phase']=='sampled_geometry'


def test_rejected_rotation_estimate_cannot_fall_back_to_concurrent_rotation(monkeypatch):
    from test_two_second_smoothing import fixture
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    p,model,target=fixture(monkeypatch)
    p.config.raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',
        synchronized_duration_s=0.,prealign_orientation_s=2.,rotate_first_s=.5,
        rotate_first_retime_enabled=True,brake_duration_s=.5,candidate_planning_budget_s=2.)
    def reject(*a,**kw):raise PickRejected('injected rotation joint limit')
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver',lambda model:None)
    monkeypatch.setattr('nero_core.two_second.rotation_lead.estimate_rotation_lead',reject)
    monkeypatch.setattr(p,'install',lambda *a:pytest.fail('invalid rotation must not install/deliver path'))
    with pytest.raises(PickRejected) as error:p.intercept(target,{'joints':np.zeros(7)})
    assert all(f['failed_phase']=='rotation_preparation' for f in error.value.details['failures'])


@pytest.mark.parametrize('change',[{'rotate_first_retime_enabled':'true'},
    {'rotate_first_retime_enabled':True,'rotate_first_s':0.}])
def test_invalid_adaptive_rotation_configuration_rejected_before_devices(change,tmp_path):
    import yaml
    raw=yaml.safe_load(CONFIG.read_text())
    for key in ('camera_calibration_file','tcp_calibration_file'):
        raw[key]=str((CONFIG.parent/raw[key]).resolve())
    raw.update(change)
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='rotate_first_retime_enabled'):CellConfig.load(path)

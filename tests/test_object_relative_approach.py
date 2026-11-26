"""No-device checks of target-frame geometry, phase transitions and closure permissions."""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.object_approach import configure, required_clearance
from nero_core.two_second.synchronized_reference import SynchronizedIntercept
from nero_core.two_second.runtime_sweep import request_for,refresh_request,sweep

def ref(guarded):
    basis=Rotation.from_euler('xyz',[.02,-.04,.75]).as_matrix()
    v=basis@np.array([.4,0.,0.])
    cap=np.array([-.025,.03,.012])-v*.5
    return SynchronizedIntercept(1.,1.5,1.1, cap,v, np.array([.04,.06,.19]),
        [1.,0.,0.,0.],Rotation.from_euler('z',.8).as_quat()[[3,0,1,2]],0.,.6,1.1,
        brake_rise_m=.001 if guarded else 0.,height_lead_s=.11 if guarded else 0.,
        height_normal_world=basis[:,2],relative_descent_s=.32,relative_sync_s=.03,
        approach_clearance_m=.043,object_axes_world=basis)

@pytest.mark.parametrize('guarded',[False,True])
def test_world_motion_in_object_frame_and_c2_for_every_join_knot(guarded):
    r=ref(guarded);R=np.asarray(r.object_axes_world)
    def relative(t):
        return R.T@(r.sample(t)[0]-r.capture_tcp_world-r.velocity_world*(r.start_s+t-r.capture_s))
    for t in np.linspace(r.relative_alignment_s,r.relative_alignment_s+r.relative_sync_s,9):
        np.testing.assert_allclose(relative(t),[0,0,.043],atol=1e-12)
        np.testing.assert_allclose(r.position_derivatives(t)[1],r.velocity_world,atol=1e-12)
        np.testing.assert_allclose(r.sample(t)[1],r.target_quaternion_wxyz,atol=1e-12)
    for t in np.linspace(r.relative_alignment_s+r.relative_sync_s,r.duration_s,41):
        np.testing.assert_allclose(relative(t)[:2],0.,atol=1e-12)
    np.testing.assert_allclose(relative(r.duration_s),0.,atol=1e-12)
    for t in r.approach_knots[1:]:
        for left,right in zip(r.position_derivatives(t-1e-9),r.position_derivatives(t+1e-9)):
            np.testing.assert_allclose(left,right,atol=5e-6)
    np.testing.assert_allclose(r.position_derivatives(0.)[1:],0.,atol=1e-12)
    np.testing.assert_allclose(r.position_derivatives(r.full_duration_s)[1:],0.,atol=1e-12)
    if guarded:
        h=np.array([r.sample(t)[0] for t in np.linspace(r.duration_s-.06,r.full_duration_s,151)])
        assert np.min(np.diff(h[:,2]))>=-1e-10
        assert np.min(np.diff(h@R[:,2]))>=-1e-10

def test_clearance_uses_highest_frozen_point_and_open_jaw_mesh_in_grasp_pose():
    basis=np.eye(4);target=NS(object_pose=basis,points=np.array([[0.,0.,.05],[0.,0.,.033]]))
    model=NS(set_state=lambda q,w:None,tcp=lambda:basis,gripper_geometries=[0],
             mesh_vertices={0:np.array([[0.,0.,-.022],[0.,0.,.1]])},
             data=NS(geom_xmat=np.eye(3).reshape(1,9),geom_xpos=np.zeros((1,3))))
    g=np.eye(4);g[2,3]=.03
    assert required_clearance(model,target,g,[],.1,.002)==pytest.approx(.044)
    target.points[:,2]+=.01
    assert required_clearance(model,target,g,[],.1,.002)==pytest.approx(.054)

def test_relative_runtime_sweep_checks_moving_points_until_actual_close():
    reference=ref(False);seen=[]
    planner=NS(valid=lambda q,w,**kw:seen.append(kw))
    dispatch=reference.contact_s-.05;state=dict(joints=np.zeros(7),width_m=.1,
        joint_stamp=dispatch-.02,pose_stamp=dispatch-.02)
    path=NS(stage='smooth_intercept_follow_brake',position=lambda t:np.ones(7)*.025)
    args=dict(dispatch=dispatch,origin=reference.start_s,advance=.05,control_hz=100.,
        reference=reference,attached=False,smooth=True,scheduled_attempt=True,follow_end=reference.follow_end_s)
    for close in (None,dispatch-.005):
        req=request_for(planner,state,**args,close_sent_s=close)
        assert req['moving_target']
        sweep(planner,path,refresh_request(req,state,dispatch))
        assert all(x['stamp'] is not None and not x['payload'] for x in seen)
        assert [x['contact'] for x in seen]==[close is not None and x['stamp']>=close for x in seen]
        if close is not None:assert {x['contact'] for x in seen}=={False,True}
        seen.clear()

def test_preclose_collision_retries_an_alternate_and_then_a_higher_approach(monkeypatch):
    from test_coarse_first_pick import fixture
    from nero_core.two_second import object_approach
    attempts=[]
    def collision(key,r,clock):
        attempts.append((key[0],r.approach_clearance_m))
        if len(attempts)<=2:raise PickRejected('工件碰撞',details={'kind':'robot_workpiece_point_cloud',
                                                                  'contact_allowed':False,'payload':False})
    p,t,clock,*_=fixture(monkeypatch,geometry=collision)
    t.object_pose=np.eye(4);p.config.raw.update(object_relative_approach=dict(
        descent_candidates_s=[.3],sync_s=.03,forward_ramp_s=.55))
    monkeypatch.setattr(object_approach,'required_clearance',lambda *a:.04)
    plan=p.intercept(t,{'joints':np.zeros(7),'width_m':.1})
    assert attempts[:3]==[(0,.04),(1,.04),(0,.05)]
    assert plan.path.diagnostics['object_relative_approach']['clearance_m']==pytest.approx(.05)


@pytest.mark.parametrize('args,enabled',[([],True),(['--object-relative-approach'],True),
                                         (['--no-object-relative-approach'],False)])
def test_hardware_entry_selects_site_policy_and_reports_fast_thresholds(capsys,args,enabled):
    from pathlib import Path
    import json
    from nero_core.two_second.fixed_freeze_pick import main
    config=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'
    assert main(['--config',str(config),'--belt-speed-cm-s','53.5','--fast-stereo','--no-depth-separation',
                 '--check-config',*args])==0
    data=json.loads(capsys.readouterr().out)
    assert bool(data['object_relative_approach'])==enabled
    assert data['grasp_workpiece_collision_check_enabled']==enabled
    assert data['fast_min_native_coverage_fraction']==.70 and data['fast_max_center_shift_m']==.014
    assert data['commands_sent']==data['hardware_connections']==0

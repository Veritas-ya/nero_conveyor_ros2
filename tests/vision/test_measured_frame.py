"""Measured-only simulation adapter: fixture input is synthetic unless marked."""
from copy import deepcopy
from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from experiments.graspgenx_sim.measured_frame import MeasuredFrameBuilder
from mujoco_conveyor_sorting.runtime.types import TargetIdentity


def fixture():
    stamp=1.; seq=4; epoch=2
    identity=TargetIdentity('s','part','task',stamp,epoch,track_id=7,track_generation=3)
    velocity={'clock_epoch':epoch,'geometry_sequence':seq,'geometry_device_timestamp':stamp,
        'geometry_host_capture_timestamp':stamp,'speed_valid':True,'motion_source':'nero7_xy_kalman',
        'velocity_base_m_s':[.1,0.,0.], 'device_timestamp':stamp,'host_capture_timestamp':stamp,
        'sequence':seq,'track_id':7,'track_generation':3,'sample_count':8,'span_s':.4,'speed_std_m_s':.001}
    labels=np.full((4,6),-1);labels[1:3,2:4]=0
    depth=np.ones((4,6),np.float32)
    transform=np.eye(4);transform[2,3]=-1.
    frame={'timestamp':stamp,'top_depth':depth,'top_rgb':np.zeros((4,6,3),np.uint8),
        'top_instance_labels':labels,'top_camera':{'intrinsics':np.array([[100.,0.,3.],[0.,100.,2.],[0.,0.,1.]]),
        'camera_to_world_opencv':np.eye(4)},'capture_robot_feedback':{'joints':np.zeros(7),'width_m':.1,'sample_time_s':stamp},
        'nero_capture':{'sequence':seq,'clock_epoch':epoch,'device_time_s':stamp,'publication_time_s':1.1,
            'measurements':{0:{'actual_peer_track_id':7,'track_generation':3,'velocity_measurement':velocity}}}}
    region={'plane_to_base':transform,'bounds_xy_m':[[-1.,1.],[-1.,1.]]}
    return frame,identity,region


def test_target_and_scene_share_single_projection_and_cached_rays(monkeypatch):
    frame,identity,region=fixture(); builder=MeasuredFrameBuilder(); calls=[]
    original=builder.projector.project
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(builder.projector,'project',counted)
    out=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    assert len(calls)==1 and len(out.detection['object_points'])==4 and len(out.detection['scene_points'])==20
    assert out.detection['frame'].header.sample_time==1. and out.detection['frame'].header.received_time==1.1
    assert out.velocity_measurement['motion_source']=='nero7_xy_kalman'
    builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    assert builder.projector.cache_rebuilds==1
    calls.clear()
    target_only=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region,include_scene=False)
    assert not calls and 'scene_points' not in target_only.detection
    np.testing.assert_allclose(target_only.detection['object_points'],out.detection['object_points'],atol=1e-12)


@pytest.mark.parametrize('kind',['task','generation','time','speed','state'])
def test_mismatched_capture_never_becomes_planning_input(kind):
    frame,identity,region=fixture(); frame=deepcopy(frame)
    if kind=='task': frame['nero_capture']['measurements'][0]['actual_peer_track_id']=8
    if kind=='generation':frame['nero_capture']['measurements'][0]['track_generation']=4
    if kind=='time':frame['nero_capture']['measurements'][0]['velocity_measurement']['geometry_sequence']=5
    if kind=='speed':frame['nero_capture']['measurements'][0]['velocity_measurement']['speed_valid']=False
    if kind=='state':frame['capture_robot_feedback']['sample_time_s']=.9
    with pytest.raises(ValueError):
        MeasuredFrameBuilder().build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)


def test_early_geometry_preparation_never_fabricates_qualified_velocity():
    from nero_core.measured_motion import MeasuredMotionSession
    frame,identity,region=fixture()
    frame['nero_capture']['measurements'][0]['velocity_measurement']['speed_valid']=False
    item=MeasuredFrameBuilder().build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),
        belt_region=region,allow_unqualified_speed=True)
    assert item.velocity_measurement is None
    assert len(item.detection['scene_points']) == 20
    session=MeasuredMotionSession(identity)
    assert session.update(item.detection,identity=identity,timestamp=item.capture_s,sequence=item.sequence,
        clock_epoch=item.clock_epoch,source_timestamp=item.capture_s,velocity_measurement=None)
    snapshot=session.snapshot(item.capture_s)
    assert snapshot is None or not snapshot.valid


def test_saved_rendered_depth_target_matches_prior_network_input():
    root=Path('runs/vision_network_warm_20261027')
    if not root.exists():pytest.skip('optional saved rendered capture unavailable')
    f,_,region=fixture(); meta=json.loads((root/'inputs/track_0000_try_01.json').read_text()); result=json.loads((root/'result.json').read_text())
    depth=np.load(root/'inputs/track_0000_try_01_depth.npy'); mask=cv2.imread(str(root/'inputs/track_0000_try_01_trackmask.png'),cv2.IMREAD_GRAYSCALE)>0
    stamp=meta['capture_sim_s'];epoch=2;seq=188
    identity=TargetIdentity('saved','part','task',stamp,epoch,track_id=0,track_generation=1)
    v=f['nero_capture']['measurements'][0]['velocity_measurement'];v.update(track_id=0,track_generation=1,geometry_sequence=seq,sequence=seq,
        geometry_device_timestamp=stamp,geometry_host_capture_timestamp=stamp,device_timestamp=stamp,host_capture_timestamp=stamp)
    T_world=np.asarray(result['camera']['world_depth_cv']);T_base=np.linalg.inv(np.asarray(result['camera']['base_to_world']))
    frame={'timestamp':stamp,'top_depth':depth,'top_rgb':np.zeros((*depth.shape,3),np.uint8),
        'top_instance_labels':np.where(mask,0,-1),'top_camera':{'intrinsics':np.asarray(meta['K']),'camera_to_world_opencv':T_world},
        'capture_robot_feedback':{'joints':np.zeros(7),'width_m':.1,'sample_time_s':stamp},
        'nero_capture':{'sequence':seq,'clock_epoch':epoch,'device_time_s':stamp,'publication_time_s':stamp,
            'measurements':{0:{'actual_peer_track_id':0,'track_generation':1,'velocity_measurement':v}}}}
    region['plane_to_base'][2,3]=-.1
    out=MeasuredFrameBuilder().build(frame,task_id=0,identity=identity,world_to_base=T_base,belt_region=region)
    with np.load(root/'inputs/track_0000_try_01.npz') as a:expected=np.asarray(a['points'])@T_base[:3,:3].T+T_base[:3,3]
    np.testing.assert_allclose(out.detection['object_points'],expected,atol=1e-7)


def test_actual_session_receives_selected_network_pose_and_nonneutral_speed():
    from nero_core.measured_motion import MeasuredMotionSession
    from experiments.graspgenx_sim.bind_measured_network import bind_measured_network
    frame,identity,region=fixture()
    session=MeasuredMotionSession(identity)
    captured=MeasuredFrameBuilder().build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    assert session.update(captured.detection, identity=identity, timestamp=captured.capture_s,
        sequence=captured.sequence,clock_epoch=captured.clock_epoch,source_timestamp=captured.capture_s,
        velocity_measurement=captured.velocity_measurement)
    pose=np.array([.01,.02,1.,1.,0.,0.,0.]); canonical=np.eye(4)
    candidate={'tcp_pose_world_wxyz':pose,'T_world_canonical':canonical,'closing_width_m':0.,'score':.7}
    provenance={'input_valid':True,'validation_reasons':[],'track_id':0,'track_generation':'3','scene_epoch':2,
        'capture_sim_s':1.,'track_position_world_m':[0.,0.,1.],'track_velocity_world_m_s':[.1,0.,0.],
        'track_quaternion_world_wxyz':[1.,0.,0.,0.]}
    obj={'object_id':0,'track_generation':3,'position':[0.,0.,1.], 'quaternion':[1.,0.,0.,0.],
        'timestamp':1.,'measurement_timestamp':1.,'track_observation_age':0.,'visible':True}
    raw={'timestamp':1.,'scene_epoch':2,'objects':[obj],'predictions':{0:{'observed_timestamp':1.}}}
    bound,admission=bind_measured_network(session=session,captured=captured,task_id=0,provenance=provenance,
        candidate=candidate,raw_observation=raw,world_to_base=np.eye(4),clock_epoch=2,
        max_observation_age_s=1.,max_result_age_s=4.)
    assert admission['allowed']
    np.testing.assert_allclose(bound['grasp'][:3,3],pose[:3],atol=1e-8)
    snap=session.snapshot(1.2)
    np.testing.assert_allclose(snap.current_grasp[:3,3],pose[:3]+[.02,0.,0.],atol=1e-8)
    assert snap.velocity_source=='nero7_xy_kalman' and snap.observed_timestamp==1.
    assert bound['network_candidate']['closing_width_m']==0.


def test_scene_attachment_preserves_exposure_and_does_not_publish_another_motion_sample():
    from nero_core.measured_motion import MeasuredMotionSession
    frame,identity,region=fixture(); builder=MeasuredFrameBuilder(); session=MeasuredMotionSession(identity)
    small=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region,include_scene=False)
    assert session.update(small.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2,
        source_timestamp=1.,velocity_measurement=small.velocity_measurement)
    scene=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    session.attach_scene_frame(scene.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2)
    snap=session.snapshot(1.1)
    assert snap.sample_count==1 and snap.observed_timestamp==1.
    stale=deepcopy(scene.detection);stale['object_points'][0,0]+=.01
    with pytest.raises(ValueError, match='target pixels'):
        session.attach_scene_frame(stale,identity=identity,timestamp=1.,sequence=4,clock_epoch=2)
    with pytest.raises(ValueError, match='old or different'):
        session.attach_scene_frame(scene.detection,identity=identity,timestamp=1.,sequence=5,clock_epoch=2)


def test_previous_captured_scene_can_be_installed_after_newer_frame():
    from nero_core.measured_motion import MeasuredMotionSession
    frame,identity,region=fixture(); builder=MeasuredFrameBuilder(); session=MeasuredMotionSession(identity,max_coast_s=None)
    first=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    assert session.update(first.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2,
        source_timestamp=1.,velocity_measurement=first.velocity_measurement)
    grasp=np.eye(4); grasp[:3,3]=[.01,0.,1.]
    selected={**first.detection,'network_candidate':{'score':.8},'network_provenance':{},
        'network_capture_center_base_m':np.median(first.detection['object_points'],axis=0),
        'network_capture_grasp_base':grasp}
    session.bind_selected_grasp(selected,identity=identity,capture_timestamp=1.,capture_sequence=4)
    later=deepcopy(first.detection);later['grasp']=later['grasp'].copy();later['grasp'][0,3]+=.01
    later['object_points']=later['object_points']+[.01,0.,0.]
    velocity=deepcopy(first.velocity_measurement)
    velocity.update(geometry_sequence=5,sequence=5,geometry_device_timestamp=1.1,geometry_host_capture_timestamp=1.1,
        device_timestamp=1.1,host_capture_timestamp=1.1)
    assert session.update(later,identity=identity,timestamp=1.1,sequence=5,clock_epoch=2,source_timestamp=1.1,
        velocity_measurement=velocity)
    class Geometry:
        def install_scene(self,points,target,state,**kwargs):
            self.scene=points.copy(); self.target=target[::3].copy(); return {}
    geometry=Geometry()
    binding=session.install_captured_scene(geometry,first.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2)
    assert binding.capture_time==1. and session.snapshot(1.1).observed_timestamp==1.1
    np.testing.assert_allclose(binding.capture_detection['grasp'][:3,3],grasp[:3,3],atol=1e-8)
    with pytest.raises(ValueError,match='accepted target capture'):
        session.install_captured_scene(geometry,first.detection,identity=identity,timestamp=1.,sequence=99,clock_epoch=2)


def test_scene_is_filtered_while_network_result_is_pending_then_bound():
    from nero_core.measured_motion import MeasuredMotionSession
    frame,identity,region=fixture(); builder=MeasuredFrameBuilder(); session=MeasuredMotionSession(identity)
    captured=builder.build(frame,task_id=0,identity=identity,world_to_base=np.eye(4),belt_region=region)
    assert session.update(captured.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2,
        source_timestamp=1.,velocity_measurement=captured.velocity_measurement)
    class Geometry:
        def install_scene(self,points,target,state,**kwargs):
            self.scene=points.copy();self.target=target[::3].copy();return {'filtered':True}
    geometry=Geometry()
    prepared=session.prepare_captured_scene(geometry,captured.detection,identity=identity,timestamp=1.,sequence=4,clock_epoch=2)
    with pytest.raises(ValueError,match='not been bound'):
        session.bind_prepared_scene(prepared,identity=identity)
    grasp=np.eye(4);grasp[:3,3]=[.01,0.,1.]
    selected={**captured.detection,'network_candidate':{'score':.8},'network_provenance':{},
        'network_capture_center_base_m':np.median(captured.detection['object_points'],axis=0),
        'network_capture_grasp_base':grasp}
    session.bind_selected_grasp(selected,identity=identity,capture_timestamp=1.,capture_sequence=4)
    bound=session.bind_prepared_scene(prepared,identity=identity)
    np.testing.assert_allclose(bound.capture_detection['grasp'],grasp,atol=1e-8)
    geometry.scene[0,0]+=.01
    with pytest.raises(ValueError,match='geometry changed'):
        session.bind_prepared_scene(prepared,identity=identity)

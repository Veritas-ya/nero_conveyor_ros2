"""Independent camera interface: original exposures, no ID qualification."""
from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace as NS
import json
import time

import cv2
import numpy as np
import pytest

from nero_core.two_second.fixed_freeze import ExposureBuffer,NativeDepthLocator,capture_admission
from nero_core.two_second.fixed_freeze_test import FixedFreezeWorker,arguments,main
from nero_core.two_second.reference import EntryNotReady
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics,RGBDFrame,SampleHeader
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface


def data():
    config=NS(direction=np.array([1.,0.,0.]),follow_bounds=np.array([-.2,.38]),canonical_tcp=np.eye(4),
        raw=dict(freeze_position_upstream_m=.45,freeze_position_tolerance_m=.02,
            maximum_observation_age_s=1.,maximum_target_height_m=.12,
            velocity_source='operator_fixed_belt_speed',fixed_belt_speed_m_s=.435,
            environment_collision_scope='measured_belt_surface_only'))
    k=CameraIntrinsics(320,240,150.,150.,160.,120.,(0.,)*8)
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.
    depth=np.ones((240,320));depth[115:125,53:63]=.95
    frame=RGBDFrame(header=SampleHeader(10.,10.01,3,5.,0),depth_m=depth,
        rgb=np.zeros((240,320,3),np.uint8),depth_intrinsics=k,color_intrinsics=k,
        camera_to_world_cv=pose,serial_number='offline',calibration_id='fixture',source='synthetic_offline_test')
    corners=np.array([[-.9,-.5],[.8,-.5],[.8,.5],[-.9,.5]],np.float32)
    surface=BeltSurface(np.eye(4),np.array([[-.9,.8],[-.5,.5]]),cv2.convexHull(corners),
        dict(normal_base=[0.,0.,1.],inlier_rms_m=0.,tilt_from_base_xy_deg=0.))
    return config,frame,surface


def test_single_native_exposure_has_full_component_without_any_tracking_state():
    config,frame,surface=data()
    original=frame.depth_m.copy()
    locator=NativeDepthLocator(config,surface,frame)
    candidates,diag=locator.locate(frame)
    assert len(candidates)==1 and candidates[0].point_count==100
    assert len(candidates[0].seed_pixel_indices)==100
    assert diag['evaluated_depth_pixels']<frame.depth_m.size
    assert .43<=config.follow_bounds[0]-candidates[0].position@config.direction<=.45
    np.testing.assert_array_equal(frame.depth_m,original)
    capture_admission(config,10.,candidates[0].position,np.array([.435,0,0]),10.3)


def test_locator_keeps_complete_workpiece_outside_the_two_centimetre_band():
    config,frame,surface=data();candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    # The workpiece spans 5.7 cm: a trigger must not retain only its central 2cm.
    pixel_x=candidate.seed_pixel_indices%320
    span=(pixel_x.max()-pixel_x.min())*.95/150.
    assert span>.05 and candidate.point_count==100


def test_multiple_components_are_reported_instead_of_arbitrarily_picked():
    config,frame,surface=data();depth=frame.depth_m.copy();depth[115:125,73:83]=.95
    frame=replace(frame,depth_m=depth)
    candidates,diag=NativeDepthLocator(config,surface,frame).locate(frame)
    assert len(candidates)==2 and diag['candidate_count']==2


def test_unknown_native_distortion_is_not_silently_ignored():
    config,frame,surface=data()
    frame=replace(frame,depth_intrinsics=replace(frame.depth_intrinsics,distortion=(.01,0,0,0)))
    with pytest.raises(ValueError,match='零畸变'):NativeDepthLocator(config,surface,frame)


@pytest.mark.parametrize('distance,reason',[(.46,'too_far_upstream'),(.42,'past_fixed_freeze_position')])
def test_interface_refuses_to_move_or_retimestamp_an_out_of_band_cloud(distance,reason):
    config,_,_=data()
    with pytest.raises(EntryNotReady) as error:
        capture_admission(config,10.,np.array([-.2-distance,0.,0.]),np.array([.435,0.,0.]),10.3)
    assert error.value.reason==reason


@pytest.mark.parametrize('now',[9.9,11.01])
def test_stale_or_future_exposure_is_rejected(now):
    config,_,_=data()
    with pytest.raises(EntryNotReady) as error:
        capture_admission(config,10.,np.array([-.64,0,0]),np.array([.435,0,0]),now)
    assert error.value.reason=='stale_exposure'


def test_raw_buffer_is_bounded_fifo_and_reports_real_drops_and_epoch_changes():
    _,frame,_=data();buffer=ExposureBuffer(3)
    for sequence in [3,4,5,5,7]:buffer.put(replace(frame,header=replace(frame.header,sequence=sequence)))
    assert buffer.diagnostics()==dict(capacity=3,queued=3,received=4,dropped=1,capture_sequence_skips=1)
    assert [buffer.take().header.sequence for _ in range(3)]==[4,5,7]
    buffer.put(replace(frame,header=replace(frame.header,clock_epoch=1,sequence=1)))
    assert buffer.take().header.sequence==1
    buffer.close();assert buffer.take() is None


@pytest.mark.parametrize('capacity',[True,0,2,91,3.5])
def test_invalid_buffer_capacity_rejected(capacity):
    with pytest.raises(ValueError):ExposureBuffer(capacity)


def test_cloud_survives_planner_failure_and_does_not_require_a_confirmed_ID(tmp_path,monkeypatch):
    from nero_core.two_second import fixed_freeze_test as module
    config,frame,surface=data();candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    point=np.array([-.64,0.,.05]);points=np.tile(point,(100,1))
    mask=np.zeros(frame.depth_m.shape,bool);mask.ravel()[candidate.seed_pixel_indices]=True
    detection=dict(mask=mask,object_points=points,plane=surface.plane,
        belt_region=surface.region_metadata(),scene_points=np.array([[0.,0.,0.]]),scene_pixel_indices=np.array([0]))
    monkeypatch.setattr(module,'detect_target',lambda *a,**kw:detection)
    monkeypatch.setattr(module,'obb_grasps',lambda *a:(np.array([np.eye(4)]),
        dict(center_world_m=point,axes_world_columns=np.eye(3),extent_m=[.05,.05,.05])))
    monkeypatch.setattr(module.time,'monotonic',lambda:10.3)
    worker=FixedFreezeWorker(config,None,tmp_path/'new_run',plan=False)
    def fail(done,observation,gate,generation):
        assert not hasattr(observation,'confirmed')
        assert done['capture']==10. and observation.sample_count==1
        raise RuntimeError('injected planning failure')
    worker.paths=NS(_calculate=fail,generation=0)
    try:
        row,geometry,path,arrays=worker.prepare(frame,np.eye(4),surface,candidate,10.,0)
        assert row['status']=='planning_failed' and row['point_cloud_generated']
        assert geometry is not None and path is None
        assert arrays['frame_depth_m'] is frame.depth_m
        assert row['capture_monotonic_s']==10.
        assert row['multiframe_matching_used'] is False
    finally:worker.pool.shutdown();worker.archive.shutdown()


def test_geometry_inconsistent_with_trigger_is_rejected_before_planning(tmp_path,monkeypatch):
    from nero_core.two_second import fixed_freeze_test as module
    config,frame,surface=data();candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    monkeypatch.setattr(module,'detect_target',lambda *a,**kw:dict(mask=np.zeros(frame.depth_m.shape,bool)))
    worker=FixedFreezeWorker(config,None,tmp_path/'new_run',plan=False)
    try:
        row,geometry,_,_=worker.prepare(frame,np.eye(4),surface,candidate,10.,0)
        assert row['status']=='point_cloud_rejected' and geometry is None
        assert row['failure_details']['seed_overlap_ratio']==0.
    finally:worker.pool.shutdown();worker.archive.shutdown()


def test_check_config_opens_no_device_and_exposes_no_execution_flag(capsys,tmp_path):
    path='/home/ubuntu22/nero_conveyor_ros2/src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    assert main(['--config',path,'--belt-speed-cm-s','43.5','--check-config','--output',str(tmp_path/'unused')])==0
    report=json.loads(capsys.readouterr().out)
    assert report['can_connections']==report['camera_connections']==report['robot_commands_sent']==0
    assert not (tmp_path/'unused').exists()
    with pytest.raises(SystemExit):arguments(['--config',path,'--belt-speed-cm-s','43.5','--execute'])


def test_single_frame_velocity_report_does_not_claim_Kalman_or_ID_tracking():
    from nero_core.two_second.velocity_source import speed_report
    from nero_core.two_second.fixed_freeze_test import interface_velocity_policy
    config,_,_=data()
    observation=NS(sample_count=1,observation_source='single_exposure_native_depth')
    report=speed_report(config,observation,np.array([.435,0.,0.]))
    assert report['observation_sample_count']==1
    assert report['tracking_sample_count']==0
    assert report['visual_velocity_used'] is False
    assert report['internal_tracking_kalman_still_runs'] is False
    assert interface_velocity_policy(config)['internal_tracking_kalman_still_runs'] is False


@pytest.mark.parametrize('speed,override',[(43.5,False),(50.7,False),(100.,False),(100.7,True)])
def test_camera_speed_can_exceed_cell_limit_without_changing_hardware_config(tmp_path,speed,override):
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.velocity_source import with_fixed_belt_speed
    from nero_core.two_second.fixed_freeze_test import interface_velocity_policy
    path=Path('/home/ubuntu22/nero_conveyor_ros2/src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    original=path.read_bytes();cell=CellConfig.load(path)
    cell_limit=cell.raw['maximum_measured_speed_m_s']
    _,config,_=arguments(['--config',str(path),'--belt-speed-cm-s',str(speed),
        '--output',str(tmp_path/'unused'),'--check-config'])
    assert config.raw['fixed_belt_speed_m_s']==pytest.approx(speed/100.)
    assert config.raw['maximum_measured_speed_m_s']==max(cell_limit,speed/100.)
    report=interface_velocity_policy(config)
    assert report['camera_test_speed_override'] is override
    assert report['cell_speed_limit_cm_s']==cell_limit*100.
    assert path.read_bytes()==original and CellConfig.load(path).raw['maximum_measured_speed_m_s']==cell_limit
    np.testing.assert_array_equal(config.velocity_limits,cell.velocity_limits)
    np.testing.assert_array_equal(config.acceleration_limits,cell.acceleration_limits)
    with pytest.raises(ValueError,match='工作速度范围'):with_fixed_belt_speed(cell,cell_limit+.007)


@pytest.mark.parametrize('speed',['0','-1','nan','inf','-inf'])
def test_camera_speed_still_rejects_invalid_numbers(tmp_path,speed):
    path='/home/ubuntu22/nero_conveyor_ros2/src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    with pytest.raises(SystemExit):
        arguments(['--config',path,'--belt-speed-cm-s',speed,'--output',str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()


def test_raw_camera_keeps_receiving_while_geometry_is_blocked_and_shutdown_saves(tmp_path,monkeypatch):
    from nero_core.two_second import fixed_freeze_test as module
    from mujoco_conveyor_sorting.runtime.backends import orbbec
    config,frame,surface=data()
    entered,release=Event(),Event();devices=[]
    config.raw['camera']={}
    calibration=dict(serial_number='offline',calibration_id='fixture',depth_to_base_cv=frame.camera_to_world_cv)
    class Camera:
        def __init__(self,config):self.sequence=0;self.stopped=False;self.first_read=False;devices.append(self)
        def start(self):pass
        def stop(self):self.stopped=True
        def read_latest(self,**kwargs):self.first_read=True;return frame
        def get_depth_to_color_cv(self):assert self.first_read;return np.eye(4)
        def read_next(self,**kwargs):
            time.sleep(.015);self.sequence+=1
            return replace(frame,header=SampleHeader(time.monotonic()-.02,time.monotonic(),self.sequence,float(self.sequence),0))
    monkeypatch.setattr(orbbec,'OrbbecCameraBackend',Camera)
    monkeypatch.setattr(module,'apply_runtime_depth_calibration',lambda frame,*args:frame)
    monkeypatch.setattr(module,'depth_capture_timing',lambda frame:dict(depth_sample_monotonic_estimate=frame.header.sample_time))
    monkeypatch.setattr(BeltSurface,'from_frame',lambda *args,**kwargs:surface)
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *args:True)
    worker=FixedFreezeWorker(config,calibration,tmp_path/'new_run',plan=False)
    def prepare(frame,d2c,surface,candidate,capture,generation):
        entered.set();assert release.wait(3.)
        return (dict(generation=generation,frame_sequence=frame.header.sequence,capture_monotonic_s=capture,
            status='point_cloud_ready',point_cloud_generated=True,point_count=100,capture_upstream_cm=44.9,
            model_path_generated=False),dict(frame=frame),None,dict(frame_depth_m=frame.depth_m))
    monkeypatch.setattr(worker,'prepare',prepare)
    save=worker.summary
    def summary(*args,**kwargs):
        save(*args,**kwargs)
        if worker.rows:worker.stop.set()
    monkeypatch.setattr(worker,'summary',summary)
    worker.start()
    try:
        assert entered.wait(3.)
        time.sleep(.08)
        assert worker.buffer.received>=4 and worker.rows==[]
        release.set();worker.join(timeout=4.)
        assert not worker.is_alive() and devices[0].stopped
        report=json.loads((worker.output/'summary.json').read_text())
        assert report['camera_closed'] and report['can_connections']==report['robot_commands_sent']==0
        assert report['point_clouds_generated']==1
        assert (worker.output/'attempt_001.npz').exists()
    finally:
        release.set();worker.stop.set();worker.buffer.close();worker.join(timeout=4.)

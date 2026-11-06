"""Actual-feedback single-frame acquisition; all hardware is replaced in tests."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import numpy as np
import pytest

from nero_core.two_second import single_exposure_vision as module
from nero_core.two_second.config import CellConfig
from nero_core.two_second.fixed_freeze_pick import main
from nero_core.two_second.single_exposure_vision import SingleExposureVision
from nero_core.two_second.fixed_freeze import NativeDepthLocator
from nero_core.two_second.vision import obb_grasps
from test_fixed_freeze_interface import data
from test_two_second_hardware import site

CONFIG=Path('/home/ubuntu22/nero_conveyor_ros2/src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')


def fixture(monkeypatch):
    config,frame,surface=data()
    config.raw.update(intercept_strategy='synchronized',freeze_source='single_exposure_native_depth')
    config.raw.update(planning_reserve_s=0.,freeze_upstream_distance_m=.1)
    state=dict(joints=[.1]*7,width_m=.084,pose=[.2,.3,.4,.5,.6,.7],joint_stamp=10.,pose_stamp=10.)
    model=NS(measured_belt_surface=surface)
    vision=SingleExposureVision(None,{},config,model,lambda:state)
    vision.geometry_model=NS()
    vision.d2c=np.eye(4);vision.clock_epoch=0
    monkeypatch.setattr(module,'time',NS(monotonic=lambda:10.3))
    from nero_core.two_second import vision as original_vision
    monkeypatch.setattr(original_vision,'time',NS(monotonic=lambda:10.3))
    monkeypatch.setattr(module,'apply_runtime_depth_calibration',lambda frame,*a:frame)
    monkeypatch.setattr(module,'depth_capture_timing',lambda f:dict(depth_sample_monotonic_estimate=10.))
    monkeypatch.setattr(module.BeltSurface,'matches_frame',lambda *a:True)
    return vision,frame,surface,state


def geometry():
    x,y,z=np.meshgrid([-.65,-.64,-.63],[-.01,0,.01],[.02,.04,.06])
    points=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    grasps,box=obb_grasps(points,np.eye(4),np.eye(4))
    return dict(object_points=points,belt_region={'plane_to_base':np.eye(4)},obb=box),grasps


def test_hardware_target_uses_measured_feedback_and_original_exposure_without_ID(monkeypatch):
    v,frame,surface,state=fixture(monkeypatch);detection,grasps=geometry()
    def frozen(config,original,d2c,actual_surface,candidate,captured,**kwargs):
        assert original is frame and actual_surface is surface and captured==10.
        assert kwargs['robot_state'] is state and kwargs['robot_model'] is v.geometry_model
        return detection,np.array([-.64,0.,.04]),np.array([.435,0.,0.]),tuple(grasps),.98
    monkeypatch.setattr(module,'frozen_geometry',frozen)
    try:
        generation,target=v._prepare_single(frame,surface,NS(),state,10.,0)
        assert generation==0 and target.identity==(0,0,3) and target.capture_s==10.
        assert target.state is state and target.state['joints']==[.1]*7
        assert target.speed_report['internal_tracking_kalman_still_runs'] is False
        assert target.speed_report['identity_source']=='capture_epoch_generation_sequence'
        assert v.meter is None
    finally:v.close()


@pytest.mark.parametrize('kind',['missing','missing_stamp','skewed','valid'])
def test_capture_feedback_pairing_retains_existing_200ms_rule(monkeypatch,kind):
    v,_,_,state=fixture(monkeypatch)
    if kind=='missing':v.read_feedback=lambda:None
    elif kind=='missing_stamp':state.pop('pose_stamp')
    elif kind=='skewed':state['joint_stamp']=9.7
    try:
        answer=v._paired_state(10.)
        assert (answer is state)==(kind=='valid')
        if kind!='valid':assert v.progress.snapshot()['last_blocker']['reason'] in ('feedback_missing','feedback_time_missing','feedback_frame_skew')
    finally:v.close()


def test_consumption_return_and_expired_results_cannot_reuse_old_exposure(monkeypatch):
    v,frame,surface,state=fixture(monkeypatch);detection,grasps=geometry()
    monkeypatch.setattr(module,'frozen_geometry',lambda *a,**kw:(detection,np.array([-.64,0,.04]),np.array([.435,0,0]),tuple(grasps),1.))
    try:
        result=v._prepare_single(frame,surface,NS(),state,10.,0)
        future=Future();future.set_result(result);v.geometry_future=future;v._finish_geometry()
        target=v.poll();assert target is not None
        v.consume(target.identity);assert v.poll() is None
        v.resume_targets();assert v.minimum_capture_s==10.3
        v.geometry_future=future;v._finish_geometry();assert v.poll() is None
        assert v.pending==__import__('collections').deque(maxlen=3)
    finally:v.close()


def test_active_task_only_receives_camera_and_never_reextracts_or_reads_feedback(monkeypatch):
    v,frame,_,_=fixture(monkeypatch);v.active_task=(0,0,1)
    def forbidden(*a,**kw):raise AssertionError('must not perform unused target work while executing')
    v.read_feedback=forbidden
    monkeypatch.setattr(module,'apply_runtime_depth_calibration',forbidden)
    try:
        v._process_frame(frame)
        assert v.progress.snapshot()['stage']=='task_active_camera_only'
    finally:v.close()


def test_native_locator_removes_measured_robot_points_before_counting_workpieces():
    config,frame,surface=data();state={'joints':[.3]*7,'width_m':.08}
    captured=[]
    def classify(points,actual_state,**kwargs):
        assert actual_state is state
        captured.append(points.copy())
        return NS(removed_mask=np.ones(len(points),bool))
    model=NS(classify_robot_points=classify)
    locator=NativeDepthLocator(config,surface,frame)
    candidates,report=locator.locate(frame,robot_model=model,robot_state=state)
    assert not candidates and len(captured)==1
    assert report['robot_points_removed']==100
    with pytest.raises(ValueError,match='真实模型和配对反馈'):
        locator.locate(frame,robot_model=model)


def test_single_frame_locator_receives_real_state_and_only_schedules_one_snapshot(monkeypatch):
    v,frame,surface,state=fixture(monkeypatch)
    candidate=NS(position=np.array([-.64,0.,.04]))
    def locate(original,**kwargs):
        assert original is frame and kwargs['robot_state'] is state and kwargs['robot_model'] is v.model
        return [candidate],{'candidate_count':1}
    v.locator=NS(locate=locate)
    calls=[]
    monkeypatch.setattr(v.geometry_pool,'submit',lambda *args:calls.append(args) or Future())
    try:
        v._process_frame(frame)
        assert len(calls)==1 and calls[0][0]==v._prepare_single
        queued_state=calls[0][4]
        assert queued_state==state and queued_state is not state
        assert v.geometry_future is not None and v.meter is None
        assert v.diagnostics()['multiframe_matching_used'] is False
    finally:v.close()


def test_raw_epoch_change_invalidates_ready_geometry_even_while_arm_is_active(monkeypatch):
    v,frame,_,_=fixture(monkeypatch)
    v.latest=NS();v.active_task=(0,0,3)
    next_frame=replace(frame,header=replace(frame.header,clock_epoch=1,sequence=4))
    def read(**kwargs):v.stop.set();return next_frame
    v.camera=NS(read_next=read)
    try:
        v._receive()
        assert v.clock_epoch==1 and v.latest is None and v.geometry_generation==1
    finally:v.close()


def test_real_acquisition_threads_rearm_with_fresh_exposure_and_close_without_device(monkeypatch):
    import time
    config,frame,surface=data()
    config.raw.update(intercept_strategy='synchronized',planning_reserve_s=0.,freeze_upstream_distance_m=.1,
        freeze_source='single_exposure_native_depth')
    model=NS(measured_belt_surface=surface,classify_robot_points=lambda points,*a,**kw:
        NS(removed_mask=np.zeros(len(points),bool)))
    class Camera:
        sequence=0
        def read_latest(self,**kwargs):return frame
        def get_depth_to_color_cv(self):return np.eye(4)
        def read_next(self,**kwargs):
            time.sleep(.015);self.sequence+=1;now=time.monotonic()
            return replace(frame,header=replace(frame.header,sample_time=now-.01,
                received_time=now,sequence=self.sequence,source_time=self.sequence/30.))
    def state():
        now=time.monotonic()
        return dict(joints=[.17]*7,width_m=.084,pose=[.1]*6,joint_stamp=now,pose_stamp=now)
    v=SingleExposureVision(Camera(),{},config,model,state);v.geometry_model=NS()
    detection,grasps=geometry()
    def frozen(config,raw,d2c,surface,candidate,captured,**kwargs):
        assert kwargs['robot_state']['joints']==[.17]*7
        assert captured==raw.header.sample_time
        return detection,np.array([-.64,0,.04]),np.array([.435,0,0]),tuple(grasps),1.
    monkeypatch.setattr(module,'frozen_geometry',frozen)
    monkeypatch.setattr(module,'apply_runtime_depth_calibration',lambda raw,*a:raw)
    monkeypatch.setattr(module,'depth_capture_timing',lambda raw:
        dict(depth_sample_monotonic_estimate=raw.header.sample_time))
    monkeypatch.setattr(module.BeltSurface,'matches_frame',lambda *a:True)
    def ready():
        end=time.monotonic()+2.
        while time.monotonic()<end:
            target=v.poll()
            if target is not None:return target
            time.sleep(.002)
        raise AssertionError(v.diagnostics())
    v.start()
    try:
        first=ready();v.consume(first.identity);before=v.buffer.received
        time.sleep(.05)
        assert v.poll() is None and v.buffer.received>before
        v.resume_targets();minimum=v.minimum_capture_s;second=ready()
        assert second.capture_s>=minimum and second.identity!=first.identity
        assert second.state['width_m']==.084
    finally:v.close()
    assert not v.thread.is_alive() and not v.reader_thread.is_alive()


@pytest.mark.parametrize('speed',[50.7,100.])
def test_dedicated_hardware_entry_validates_without_opening_devices(monkeypatch,capsys,speed):
    from nero_core.two_second.main import main as original_main
    original=CONFIG.read_bytes()
    monkeypatch.setitem(sys.modules,'mujoco_conveyor_sorting.runtime.nero_pick_hardware',None)
    monkeypatch.setitem(sys.modules,'mujoco_conveyor_sorting.runtime.backends.orbbec',None)
    args=['--config',str(CONFIG),'--belt-speed-cm-s',str(speed),'--continuous','--check-config']
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report['freeze_source']=='single_exposure_native_depth'
    assert report['hardware_connections']==report['commands_sent']==0
    assert report['taught_segments'] and report['prepared_taught_cycle_enabled']
    assert report['velocity_policy']['internal_tracking_kalman_still_runs'] is False
    assert original_main(args)==0
    baseline=json.loads(capsys.readouterr().out)
    for key in ['joint_velocity_limits_rad_s','joint_acceleration_limits_rad_s2','prealign_orientation_s',
                'follow_duration_s','brake_duration_s','taught_segments','command_advance_s',
                'workpiece_collision_margin_m','payload_belt_clearance_m','closure_policy']:
        if key in report:assert report[key]==baseline[key]
    assert baseline['freeze_source']=='tracked' and CONFIG.read_bytes()==original


@pytest.mark.parametrize('speed',['0','-1','nan','inf','100.1'])
def test_dedicated_hardware_entry_keeps_physical_speed_validation(capsys,speed):
    assert main(['--config',str(CONFIG),'--belt-speed-cm-s',speed,'--check-config'])==2
    assert json.loads(capsys.readouterr().out)['hardware_connections']==0


def test_single_frame_entry_requires_actual_fixed_speed(capsys):
    assert main(['--config',str(CONFIG),'--check-config'])==2
    assert '真实固定带速' in json.loads(capsys.readouterr().out)['error']


def test_execute_entry_changes_only_vision_and_reuses_original_cycle_executor(site,tmp_path,monkeypatch):
    import importlib
    import yaml
    from nero_core.two_second.execution import TwoSecondExecutor
    from nero_core.two_second import cycle_loop,planning,vision
    path,raw=site
    raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',prealign_orientation_s=1.2,
        freeze_position_upstream_m=.45,freeze_position_tolerance_m=.02,maximum_measured_speed_m_s=1.)
    path.write_text(yaml.safe_dump(raw))
    calls=[]
    class Hardware:
        def __init__(self,*args):calls.append(('hardware_constructor',args))
        def configure_joint_command_mode(self,mode):calls.append(('command_mode',mode))
        def start(self):calls.append('fake_can_started')
        def snapshot(self):return dict(joints=[0.]*7,pose=[0.]*6,width_m=.1,force_n=0.)
        snapshot_nowait=snapshot
        def close(self):calls.append('fake_can_closed')
    class Camera:
        def __init__(self,*args):pass
        def start(self):calls.append('fake_camera_started')
        def close(self):calls.append('fake_camera_closed')
    class Planner:
        def __init__(self,*args):self.model=NS()
        def diagnostics(self):return {}
    class Vision:
        def __init__(self,camera,calibration,config,model,feedback):
            calls.append('single_exposure_vision_constructed')
            assert config.raw['freeze_source']=='single_exposure_native_depth'
            assert feedback()['width_m']==.1
        def start(self):return self
        def close(self):calls.append('single_exposure_vision_closed')
        def diagnostics(self):return {'freeze_source':'single_exposure_native_depth'}
    def forbidden(*args,**kwargs):raise AssertionError('old tracking vision must not be constructed')
    def cycles(config,planner,device,acquisition,**kwargs):
        assert kwargs['executor_factory'] is TwoSecondExecutor
        assert config.raw['prealign_orientation_s']==1.2
        calls.append('original_cycle_loop_received_original_executor')
        kwargs['result'].update(status='requested_feedback_cycles_completed',completed_feedback_cycles=1)
    monkeypatch.setattr(importlib.import_module('mujoco_conveyor_sorting.runtime.nero_pick_hardware'),'Hardware',Hardware)
    monkeypatch.setattr(importlib.import_module('mujoco_conveyor_sorting.runtime.backends.orbbec'),'OrbbecCameraBackend',Camera)
    monkeypatch.setattr(planning,'MeasuredPlanner',Planner)
    monkeypatch.setattr(planning,'calibrated_model',lambda *a:NS())
    monkeypatch.setattr(module,'SingleExposureVision',Vision)
    monkeypatch.setattr(vision,'LiveVision',forbidden)
    monkeypatch.setattr(cycle_loop,'run_pick_cycles',cycles)
    output=tmp_path/'fake_execute_only'
    assert main(['--config',str(path),'--belt-speed-cm-s','10','--execute','--continuous','--output',str(output)])==0
    assert 'single_exposure_vision_constructed' in calls
    assert 'original_cycle_loop_received_original_executor' in calls
    assert 'fake_can_closed' in calls and 'fake_camera_closed' in calls
    report=json.loads((output/'summary.json').read_text())
    assert report['freeze_source']=='single_exposure_native_depth'
    assert report['automatic_enable'] is False

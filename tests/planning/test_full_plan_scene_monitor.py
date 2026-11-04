"""Visible obstacle events retain the certified sample grid; no camera IO."""
from types import SimpleNamespace
from threading import Lock
from collections import deque
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.runtime.full_plan_scene_monitor import FullPlanSceneMonitor
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.types import RGBDFrame, CameraIntrinsics, SampleHeader


class Geometry:
    def check_state(self,q,width,**kwargs):
        if np.any(np.linalg.norm(self.scene-q[:3],axis=1)<.008):
            raise PickRejected('new measured obstacle')
    def tcp(self):
        return np.eye(4)


def monitor():
    obj=FullPlanSceneMonitor.__new__(FullPlanSceneMonitor)
    obj.reference=cKDTree(np.array([[0.,0.,0.]]))
    obj.geometry=Geometry()
    obj.lock=Lock();obj.status={'failed':False}
    return obj


def bundle():
    return SimpleNamespace(request=SimpleNamespace(full_grasp_width_m=.04),
        checked_samples=(('approach',1.,np.r_[.1,0.,0.,np.zeros(4)]),
                         ('descent',2.,np.r_[.2,0.,0.,np.zeros(4)])))


def test_existing_surface_noise_does_not_trigger_obstacle_replan():
    assert monitor().check_points([[.003,0.,0.]],bundle(),0.) is None
    assert monitor().check_points([[.5,0.,0.]],bundle(),0.) is None


def test_new_obstacle_on_remaining_path_triggers_abort():
    assert monitor().check_points([[.2,0.,0.]],bundle(),0.)=='new_obstacle_on_remaining_trajectory'
    # An obstacle behind the arm cannot invalidate a segment already passed.
    assert monitor().check_points([[.1,0.,0.]],bundle(),1.5) is None


def frame_monitor(monkeypatch):
    import mujoco_conveyor_sorting.runtime.full_plan_scene_monitor as module
    obj=monitor()
    frame=RGBDFrame(header=SampleHeader(sequence=1,sample_time=100.,received_time=100.),
        depth_m=np.full((2,2),.3),depth_intrinsics=CameraIntrinsics(2,2,100.,100.,.5,.5),
        camera_to_world_cv=np.eye(4),serial_number='offline',calibration_id='synthetic')
    obj.clock=lambda:100.1;obj.capture_monotonic=lambda _f:100.
    obj.calibrate_frame=lambda f:f;obj.frame_binding=obj._frame_binding(frame)
    obj.samples=deque([(100.,np.zeros(7),.1,0.,100.)]);obj.plane=obj.region=None
    obj.geometry.classify_robot_points=lambda p,*a,**k:SimpleNamespace(removed_mask=np.zeros(len(p),bool))
    obj.geometry.scene_noise_filter_config=None
    obj.context=SimpleNamespace(policy=SimpleNamespace(_full_pick_bundle=None))
    obj.status={'failed':False,'frames_checked':0}
    monkeypatch.setattr(module,'filter_scene_region',lambda p,*a:SimpleNamespace(retained_points=p))
    monkeypatch.setattr(module,'filter_scene_noise',lambda p,*a:SimpleNamespace(retained_points=p))
    return obj,frame


def test_real_frame_type_keeps_original_capture_time_and_binding(monkeypatch):
    obj,frame=frame_monitor(monkeypatch)
    obj.process_frame(frame)
    assert obj.status['validated_capture_monotonic_s']==100.
    assert obj.status['frames_checked']==1 and obj.status['reason']=='ok'
    changed=replace(frame,header=replace(frame.header,clock_epoch=1,sequence=2))
    obj.process_frame(changed)
    assert obj.status['failed'] and obj.status['failure_reason']=='scene_camera_binding_changed'


def test_depth_without_corresponding_joint_packet_is_not_a_fresh_certificate(monkeypatch):
    obj,frame=frame_monitor(monkeypatch)
    obj.samples=deque([(99.95,np.zeros(7),.1,0.,100.)])
    obj.process_frame(frame)
    assert 'validated_capture_monotonic_s' not in obj.status
    assert obj.status['frames_checked']==0

"""Actual original RGB-D/calibration/filter code plus offline concurrency tests."""
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import hashlib
import json
import time

import cv2
import numpy as np
import pytest

from nero_core.two_second.perception import make_meter
from nero_core.two_second.measured_rgbd import green_belt_points,fit_plane
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.types import RGBDFrame,CameraIntrinsics,SampleHeader
from vendor.nero7_speed_reference.nero7_perception.validation import synthetic_dataset
from vendor.nero7_speed_reference.nero7_perception.detection import detect_blocks
from vendor.nero7_speed_reference.nero7_perception.geometry import CameraGeometry
from vendor.nero7_speed_reference.nero7_perception.projection import project_cloud
from vendor.nero7_speed_reference.nero7_perception.conveyor_model import BeltModel
from vendor.nero7_speed_reference.nero7_perception.block_tracker import BlockTracker
from test_two_second_diagnostics import vision_fixture


@pytest.fixture(scope='module')
def frames():
    data,_=synthetic_dataset()
    k=data['K']
    intrinsics=CameraIntrinsics(width=320,height=240,fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2])
    return [RGBDFrame(header=SampleHeader(t,t,i,t,0),
        depth_m=np.where(np.isfinite(z),z,0.),rgb=cv2.cvtColor(c,cv2.COLOR_BGR2RGB),
        depth_intrinsics=intrinsics,color_intrinsics=intrinsics,camera_to_world_cv=transform,
        serial_number='SYNTHETIC',calibration_id='upstream-synthetic-fixture',source='synthetic_upstream_fixture')
        for i,(c,z,t,transform) in enumerate(zip(data['color_bgr'],data['depth_m'],data['stamps'],data['transforms']))]


def config(mode='rgb_depth'):
    return SimpleNamespace(direction=np.array([0.,-1.,0.]),follow_bounds=np.array([0.,.5]),raw={
        'perception_detection_mode':mode,'perception_plane_mode':'ransac3d',
        'maximum_target_height_m':.12,'maximum_measured_speed_m_s':.3})


def meter_for(frames, mode='rgb_depth'):
    surface=BeltSurface.from_frame(frames[0],np.eye(4),fit_plane=fit_plane,green_belt_points=green_belt_points)
    meter=make_meter(config(mode),surface)
    # The upstream fixture's part is at X=-0.74; published upstream validation
    # explicitly uses this same wider synthetic workspace, not the cell's ROI.
    meter.parameters['workspace_x']=[-1.,1.]
    return meter


def test_upstream_algorithms_and_calibration_are_verbatim():
    root=Path(__file__).parents[1]/'src/nero_core'
    manifest=json.loads((root/'vendor/nero7_speed_reference/source_manifest.json').read_text())
    for name in ('block_tracker.py','kalman_motion.py','detection.py','base_plane.py','validation.py'):
        entry=next(f for f in manifest['files'] if f['upstream_path'].endswith('/'+name))
        assert hashlib.sha256((root/entry['local_path']).read_bytes()).hexdigest()==entry['sha256']


def test_factory_keeps_kalman_parameters_and_vertical_height_separate_from_legacy(frames):
    fused=meter_for(frames)
    legacy=meter_for(frames,'legacy_depth')
    assert fused.parameters['block_height_max']==.12
    assert legacy.parameters['block_height_max']==.06
    for key in ('kalman_measurement_std','kalman_accel_noise_std','tracker_gate_radius',
                'tracker_lost_timeout','speed_window_s','kalman_innovation_gate'):
        assert fused.parameters[key]==legacy.parameters[key]


def test_rebuilding_adapter_in_same_epoch_never_reuses_consumed_identity(frames):
    previous=meter_for(frames)
    for frame in frames:previous.update(frame,np.eye(4))
    original=previous.latest_snapshot
    renewed=make_meter(config(),previous.surface,previous=previous)
    assert renewed._generation>original.track_generation
    assert renewed._next_id>max(t.track_id for t in original.tracks)
    assert renewed._fused.plane is None  # new geometry must be measured again


def test_actual_fused_pipeline_matches_original_full_frame_detector_and_tracker(frames):
    meter=meter_for(frames)
    p=meter.parameters
    camera=CameraGeometry(frames[0].color_intrinsics.matrix,320,240,'color_optical')
    speed_cfg={key:p['speed_'+key] for key in ('window_s','min_samples','min_span')}
    motion={key:p['kalman_'+key] for key in ('measurement_std','accel_noise_std',
        'cross_accel_noise_std','initial_velocity_std','innovation_gate','max_velocity_std',
        'observation_max_age','qualification_gap')}
    motion['max_velocity']=p['kalman_max_velocity_m_s']
    reference=BlockTracker(BeltModel([0.,-1.]),speed_cfg,max_vel=p['tracker_max_vel'],
        gate_radius=p['tracker_gate_radius'],confirm_frames=p['tracker_confirm_frames'],
        lost_frames=p['tracker_lost_frames'],lost_timeout=p['tracker_lost_timeout'],motion_cfg=motion)
    velocities=[];ids=set()
    for i,frame in enumerate(frames):
        result=meter.update(frame,np.eye(4))
        if i<14:
            assert not meter.latest_snapshot.current and not meter.latest_snapshot.tracks
            continue
        assert meter._fused.plane is not None
        aligned=project_cloud(camera.grid_rays()*frame.depth_m[...,None],np.eye(4),camera)
        detections,mask=detect_blocks(cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR),aligned,
            camera,frame.camera_to_world_cv,meter._fused.plane,p)
        np.testing.assert_array_equal(mask,meter._mask)
        tracks=reference.update(detections,frame.header.source_time)
        assert len(tracks)==len(meter._tracker.tracks)
        for a,b in zip(tracks,meter._tracker.tracks):
            assert (a.id,a.speed_valid,a.motion.accepted_count,a.motion.rejected_count)==(
                b.id,b.speed_valid,b.motion.accepted_count,b.motion.rejected_count)
            np.testing.assert_allclose(a.motion.state,b.motion.state,rtol=0,atol=1e-10)
            ids.add(b.id)
        if result['speed_valid']:velocities.append(result['signed_belt_speed_m_s'])
    assert len(velocities)>=30 and ids=={0}
    assert velocities[-1]==pytest.approx(.028,abs=.001)
    assert meter._fused.diagnostics()['frames']==15
    assert meter._fused.plane.diagnostics['frame_height_spread_m']<.004
    # The depth measurement and reported vertical height use the same plane.
    for d in meter._detections:
        assert d['centroid_base'][2]-d['belt_z']==pytest.approx(d['height'])


def test_duplicate_frames_do_not_complete_calibration_and_new_epoch_rejects_old_plane(frames):
    meter=meter_for(frames)
    for _ in range(20):meter.update(frames[0],np.eye(4))
    assert len(meter._fused.collector.stamps)==1 and meter._fused.plane is None
    changed=replace(frames[1],header=replace(frames[1].header,clock_epoch=1))
    with pytest.raises(ValueError,match='上下文变化'):
        meter.update(changed,np.eye(4))
    assert meter.latest_snapshot is None


def test_ordered_burst_delivery_preserves_original_nero7_velocity(frames):
    from mujoco_conveyor_sorting.runtime.frame_delivery import FrameDelivery
    baseline=meter_for(frames)
    buffered=meter_for(frames)
    delivery=FrameDelivery(8)
    previous=None
    # Capture continues during a three-frame consumer stall. Catch up using
    # the measured exposure timestamps, never the catch-up wall clock.
    for offset in range(0,len(frames),3):
        burst=frames[offset:offset+3]
        for frame in burst:
            baseline.update(frame,np.eye(4))
            delivery.append(frame)
        for expected in burst:
            actual=delivery.take(previous,burst[-1].header.sample_time,.25)
            assert actual is expected
            buffered.update(actual,np.eye(4))
            previous=actual.header.sequence
        for expected,actual in zip(baseline.latest_snapshot.tracks,buffered.latest_snapshot.tracks):
            assert actual.sample_count==expected.sample_count
            assert actual.velocity_base_m_s==expected.velocity_base_m_s
            assert actual.observation_source_time==expected.observation_source_time
    assert delivery.report()['ordered_delivered_frames']==len(frames)
    assert buffered.latest_snapshot.tracks[0].speed_valid


def test_device_clock_rollback_restarts_an_incomplete_calibration_window(frames):
    meter=meter_for(frames)
    for frame in frames[:5]:meter.update(frame,np.eye(4))
    assert len(meter._fused.collector.stamps)==5
    restarted=replace(frames[0],header=SampleHeader(100.,100.,6,1.,0))
    meter.update(restarted,np.eye(4))
    assert meter._fused.collector.stamps==[1.]
    assert not meter.latest_snapshot.current


def test_next_object_resets_tracking_without_recalibrating_the_same_camera(frames):
    meter=meter_for(frames)
    for frame in frames[:20]:meter.update(frame,np.eye(4))
    plane=meter._fused.plane
    assert plane is not None
    generation=meter._generation
    meter.reset()
    assert meter._fused.plane is plane
    assert meter._generation>generation


def test_upstream_roi_contains_visible_full_height_and_never_changes_native_images(frames):
    surface=BeltSurface.from_frame(frames[0],np.eye(4),fit_plane=fit_plane,green_belt_points=green_belt_points)
    cfg=config();cfg.follow_bounds=np.array([0.,.5])
    meter=make_meter(cfg,surface)
    original_rgb=frames[0].rgb.copy();original_depth=frames[0].depth_m.copy()
    transform=meter._geometry(frames[0],np.eye(4))
    roi=meter.tracking_rgb_roi
    assert roi is not None and roi.any() and not roi.all()
    flat=np.asarray(surface.hull_xy).reshape(-1,2).mean(axis=0)
    base=surface.plane_to_base[:3,:2]@flat+surface.plane_to_base[:3,3]
    assert base@cfg.direction<0
    points=base+np.array([[0.,0.,0.],[0.,0.,.06],[0.,0.,.12]])
    optical=(points-transform[:3,3])@transform[:3,:3]
    uv=np.rint(meter._camera.project(optical)).astype(int)
    assert roi[uv[:,1],uv[:,0]].all()
    np.testing.assert_array_equal(frames[0].rgb,original_rgb)
    np.testing.assert_array_equal(frames[0].depth_m,original_depth)


def test_grasp_geometry_does_not_block_camera_speed_updates(monkeypatch):
    vision=vision_fixture(monkeypatch,'eligible')
    from nero_core.two_second import vision as module
    original=module.detect_target
    entered,release=Event(),Event()
    updates=[]
    def slow(*a,**kw):
        entered.set()
        assert release.wait(2.)
        return original(*a,**kw)
    monkeypatch.setattr(module,'detect_target',slow)
    base_frame=vision.camera.read_latest()
    original_update=vision.meter.update
    def update(frame,*a):
        updates.append(frame.header.sequence)
        return original_update(frame,*a)
    vision.meter.update=update
    class Camera:
        count=0
        def get_depth_to_color_cv(self):return np.eye(4)
        def read_latest(self,**kw):
            self.count+=1
            if self.count==2:assert entered.wait(2.)
            if self.count==12:
                assert len(updates)==11 and not release.is_set()
                release.set()
                vision.geometry_future.result(timeout=2.)
                vision._finish_geometry()
                vision.stop.set()
                raise RuntimeError('offline sequence finished')
            time.sleep(.002)
            return SimpleNamespace(header=SimpleNamespace(**{
                **vars(base_frame.header),'sequence':self.count}))
    vision.camera=Camera()
    try:vision._run()
    finally:release.set();vision.close()
    assert vision.error is None and len(updates)==11
    assert vision.latest is not None
    assert vision.latest.sequence==1  # actual selected exposure, not frame 11
    assert vision.latest.velocity[0]==pytest.approx(.1)


@pytest.mark.parametrize('invalidate',['target_switch','return','clock_reset','tracker_reset','cancel'])
def test_late_geometry_cannot_publish_after_identity_or_clock_invalidation(monkeypatch,invalidate):
    vision=vision_fixture(monkeypatch,'eligible')
    frame=vision.camera.read_latest()
    peer=vision.meter.latest_snapshot;track=peer.tracks[0]
    vision.clock_epoch=0;vision.progress_snapshot=peer
    future=vision.geometry_pool.submit(vision._prepare_target,frame,np.eye(4),peer,track,
        vision.read_feedback(),frame.header.received_time-.001,vision.geometry_generation,
        vision.model.measured_belt_surface,.05)
    vision.geometry_future=future
    assert future.result(timeout=2.) is not None
    if invalidate=='target_switch':vision.consume((0,1,99))
    if invalidate=='return':vision.resume_targets()
    if invalidate=='clock_reset':vision.clock_epoch=1
    if invalidate=='tracker_reset':vision.progress_snapshot=replace(peer,track_generation=2)
    if invalidate=='cancel':vision.stop.set()
    vision._finish_geometry()
    assert vision.latest is None
    vision.close()

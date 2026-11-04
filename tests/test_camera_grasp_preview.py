"""Synthetic gate checks; these do not claim a measured camera/robot result."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nero_core.two_second.preview import freeze_candidate, CameraPreview
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7TrackSnapshot


def fixture():
    config=SimpleNamespace(direction=np.array([1.,0.,0.]),follow_bounds=np.array([-.2,.28]),
        canonical_tcp=np.eye(4),workspace=np.array([[-2,2],[-2,2],[-2,2]]),
        raw={'minimum_speed_span_s':.45,'maximum_observation_age_s':1.,
             'maximum_target_height_m':.12,
             'planning_reserve_s':1.5,'freeze_upstream_distance_m':.35,
             'approach_duration_candidates_s':[2.,3.,4.]},
        settings=SimpleNamespace(full_plan_close_window_s=.8))
    track=Nero7TrackSnapshot(1,1,(-.51,0.,.03),(-.51,0.,.03),(.1,0.,0.),True,
        10.,10.,15,.5,.001,((1e-6,0.),(0.,1e-6)),0,150.,True)
    peer=SimpleNamespace(current=True,sequence=12,clock_epoch=0,track_generation=0)
    points=np.array([[x,y,z] for x in np.linspace(-.53,-.49,5)
        for y in np.linspace(-.02,.02,5) for z in (.02,.04)])
    detection={'object_points':points,'plane':{'normal_base':[0.,0.,1.]},
               'belt_region':{'plane_to_base':np.eye(4)}}
    return config,track,peer,detection


def test_freeze_binds_observed_track_and_exposure_obb_without_robot():
    c,t,p,d=fixture();f=freeze_candidate(c,t,p,d,10.,10.1,.05)
    assert f['entry_s']==pytest.approx(13.1)
    np.testing.assert_allclose(f['track_position_base_m'],t.observed_position_base)
    assert len(f['grasps'])==2 and f['can_connections']==f['commands_sent']==0
    assert not f['trajectory_certified'] and not f['robot_self_filter_applied']
    before=f['points'].copy();d['object_points'][:]=999
    np.testing.assert_array_equal(f['points'],before)


@pytest.mark.parametrize('changes',[
    {'speed_valid':False},{'confirmed':False},{'misses':1},
    {'span_s':.44},{'velocity_base_m_s':None},
    {'velocity_base_m_s':(-.1,0.,0.)}])
def test_invalid_visual_speed_never_freezes(changes):
    c,t,p,d=fixture()
    with pytest.raises(ValueError):freeze_candidate(c,replace(t,**changes),p,d,10.,10.1,.05)


@pytest.mark.parametrize('capture,now',[(10.,11.01),(10.2,10.1)])
def test_stale_or_future_capture_never_freezes(capture,now):
    c,t,p,d=fixture()
    with pytest.raises(ValueError):freeze_candidate(c,t,p,d,capture,now,.05)


def test_mismatched_cloud_is_rejected():
    c,t,p,d=fixture();d['object_points'][:,1]+=.2
    with pytest.raises(ValueError,match='关联'):freeze_candidate(c,t,p,d,10.,10.1,.05)


def test_twenty_centimeters_keeps_window_failure_visible():
    c,t,p,d=fixture();t=replace(t,velocity_base_m_s=(.2,0.,0.))
    f=freeze_candidate(c,t,p,d,10.,10.01,.05)
    assert not any(r['window_fits'] for r in f['candidates'])


def test_fixed_speed_preview_uses_fixed_value_without_visual_velocity():
    from nero_core.two_second.preview import qualified_tracks
    from nero_core.two_second.velocity_source import FIXED
    c,t,p,d=fixture();c.raw.update(velocity_source=FIXED,fixed_belt_speed_m_s=.1)
    t=replace(t,speed_valid=False,velocity_base_m_s=None,sample_count=2,span_s=.04)
    p.tracks=(t,)
    assert qualified_tracks(c,p)==[t]
    result=freeze_candidate(c,t,p,d,10.,10.1,.05)
    assert result['velocity_source']==FIXED and result['configured_speed_m_s']==.1
    np.testing.assert_allclose(result['velocity_base_m_s'],[.1,0,0])


def test_rearm_removes_frozen_sample_and_speed_history():
    preview=CameraPreview(None,None);preview.frozen={'capture_s':1.}
    calls=[];preview.meter=SimpleNamespace(reset=lambda:calls.append('reset'))
    preview.reset();assert preview.frozen is None and calls==['reset']
    preview.close()


def test_clock_reset_invalidates_frozen_even_if_next_frame_has_no_plane(monkeypatch):
    from mujoco_conveyor_sorting.runtime import depth_calibration, frame_products
    from nero_core.two_second import measured_rgbd
    from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
    monkeypatch.setattr(depth_calibration,'apply_runtime_depth_calibration',lambda f,*_:f)
    monkeypatch.setattr(CameraPreview,'scene_cloud',lambda *_:np.zeros((5,3)))
    def reject(_):raise ValueError('test clock frame has no usable exposure time')
    monkeypatch.setattr(measured_rgbd,'depth_capture_timing',reject)
    monkeypatch.setattr(BeltSurface,'from_frame',lambda *a,**kw:reject(None))
    preview=CameraPreview(None,None);preview.epoch=0;preview.frozen={'clock_epoch':0}
    frame=SimpleNamespace(header=SimpleNamespace(clock_epoch=1))
    assert preview.process(frame,np.eye(4))['frozen'] is None
    preview.close()


def test_obb_overlay_and_tcp_use_same_base_frame_for_tilted_measured_belt():
    from scipy.spatial.transform import Rotation
    c,t,p,d=fixture()
    plane=np.eye(4);plane[:3,:3]=Rotation.from_euler('xyz',[.06,-.03,.4]).as_matrix()
    plane[:3,3]=[.15,.2,-.14]
    d['belt_region']['plane_to_base']=plane
    f=freeze_candidate(c,t,p,d,10.,10.1,.05)
    assert f['obb']['coordinate_frame']=='robot_base'
    for grasp in f['grasps']:
        np.testing.assert_allclose(f['obb']['center_world_m'],grasp[:3,3],atol=1e-10)


def test_geometry_result_is_available_before_speed_is_qualified():
    from concurrent.futures import Future
    c,t,p,d=fixture();t=replace(t,speed_valid=False,sample_count=1,span_s=0.)
    p.tracks=(t,)
    preview=CameraPreview(c,None)
    geometry={'points':d['object_points']}
    preview.future=Future();preview.future.set_result({'generation':preview.generation,
        'geometry':geometry,'peer':p,'capture':10.,'status':'ready'})
    preview.finish_geometry()
    assert preview.geometry['geometry'] is geometry
    assert preview.frozen is None and '合格测速 0 个' in preview.last_reason
    preview.close()


def test_rearm_discards_late_geometry_instead_of_freezing_old_exposure():
    from concurrent.futures import Future
    c,t,p,d=fixture();p.tracks=(t,)
    preview=CameraPreview(c,None);old=preview.generation
    preview.future=Future();preview.reset()
    preview.future.set_result({'generation':old,'geometry':{'points':d['object_points']}})
    preview.finish_geometry()
    assert preview.geometry is None and preview.frozen is None
    preview.close()


def test_track_diagnostics_explain_insufficient_frames_and_span():
    from nero_core.two_second.preview import track_labels
    c,t,p,d=fixture();p.tracks=(replace(t,sample_count=2,span_s=.1,speed_valid=False),)
    rows=track_labels(c,p,{'speed_min_samples':8,'kalman_max_velocity_std':.01})
    assert '观测2/8帧' in rows[0]['reasons']
    assert '跨度0.10/0.45秒' in rows[0]['reasons']


def test_preview_uses_original_detector_and_retained_scene_with_captured_feedback(monkeypatch):
    from nero_core.two_second import planning
    from mujoco_conveyor_sorting.runtime import single_pick
    c,t,p,d=fixture();state={'joints':np.arange(7)*.01,'width_m':.08}
    frame=SimpleNamespace(header=SimpleNamespace(sequence=12,source_time=10.))
    d.update(scene_points=np.ones((4,3)),camera_origin_base_m=np.zeros(3),
             mask=np.ones((2,2),bool),scene_pixel_indices=np.arange(4))
    retained=np.array([[.11,.22,.33]])
    calls=[]
    class Model:
        scene=retained
        def check_feedback(self,actual):assert actual is state;calls.append('feedback')
        def install_scene(self,scene,target,actual,**kwargs):
            assert scene is d['scene_points'] and target is d['object_points'] and actual is state
            assert kwargs['target_mask'] is d['mask']
            assert kwargs['scene_pixel_indices'] is d['scene_pixel_indices']
            calls.append('original_scene');return {'visible_scene_points':1}
    model=Model()
    monkeypatch.setattr(planning,'calibrated_model',lambda _:model)
    def detect(actual_frame,d2c,**kwargs):
        assert actual_frame is frame
        assert kwargs['robot_state'] is state and kwargs['robot_model'] is model
        calls.append('original_detector');return d
    monkeypatch.setattr(single_pick,'detect_target',detect)
    preview=CameraPreview(c,None)
    try:
        job=preview.geometry_job(frame,np.eye(4),object(),p,10.,preview.generation,state)
        result=preview.scene_job(job)
        assert calls==['feedback','original_detector','feedback','original_scene']
        np.testing.assert_array_equal(result['points'],retained)
        assert result['source']=='original_RobotModel.install_scene'
        assert not np.shares_memory(result['points'],model.scene)
    finally:preview.close()


def test_old_scene_filter_result_cannot_survive_rearm():
    from concurrent.futures import Future
    preview=CameraPreview(None,None);old=preview.generation
    preview.scene_future=Future();preview.reset()
    preview.scene_future.set_result({'generation':old,'points':np.ones((1,3))})
    preview.finish_scene()
    assert preview.filtered_scene is None
    preview.close()

from dataclasses import replace
from types import SimpleNamespace as NS
import json

import cv2
import numpy as np
import pytest

from legacy.test_grasp_depth import native_observation, Reconstruction
from nero_core.two_second.fast_perception import FastTargetPerception, camera_settings, validate
from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface


def config():
    return NS(raw={'maximum_target_height_m':.12,'camera':{},'perception_belt_edge_margin_m':.005},
              direction=np.array([1.,0.,0.]))


def test_reuses_old_reconstruction_retains_original_frame_scene_and_timestamp(tmp_path):
    observation=native_observation();frame=observation['frame'];cfg=config()
    backend=Reconstruction(observation)
    service=FastTargetPerception(cfg,tmp_path,backend=backend)
    out=service.refine(observation,frame,np.eye(4))
    assert out['frame'] is frame and out['scene_points'] is observation['scene_points']
    assert out['mask'] is observation['mask']
    np.testing.assert_allclose(out['object_points'][:,2],.046,atol=1e-6)
    np.testing.assert_array_equal(out['native_object_points'],observation['object_points'])
    assert out['fast_stereo']['applied'] and service.completed==1
    assert out['fast_stereo']['capture_header']['sequence']==frame.header.sequence


def test_fast_failure_is_not_silently_reported_as_success(tmp_path):
    observation=native_observation();backend=Reconstruction(observation)
    backend.error=RuntimeError('injected GPU failure')
    service=FastTargetPerception(config(),tmp_path,backend=backend)
    with pytest.raises(TargetNotReady,match='Fast工件重建未成功') as error:
        service.refine(observation,observation['frame'],np.eye(4))
    assert error.value.details['fast_stereo']['applied'] is False
    assert service.completed==0


def test_missing_ir_or_unsynchronized_capture_never_calls_network(tmp_path):
    observation=native_observation();backend=Reconstruction(observation)
    service=FastTargetPerception(config(),tmp_path,backend=backend)
    for frame in [replace(observation['frame'],left_ir=None),
                  replace(observation['frame'],timestamp_metadata={'stereo_ir':{'synchronized':False}})]:
        with pytest.raises(TargetNotReady):service.refine(observation,frame,np.eye(4))
    assert not backend.calls


def test_reconstructed_center_still_obeys_near_camera_margin(tmp_path):
    observation=native_observation();backend=Reconstruction(observation);cfg=config()
    cfg.raw['perception_belt_near_camera_margin_m']=.017
    center_y=np.median(observation['object_points'][:,1])
    high=center_y+.01
    hull=cv2.convexHull(np.array([[-.5,-.5],[.5,-.5],[.5,high],[-.5,high]],np.float32))
    surface=BeltSurface(np.eye(4),np.array([[-.5,.5],[-.5,high]]),hull,observation['plane'])
    service=FastTargetPerception(cfg,tmp_path,backend=backend)
    with pytest.raises(TargetNotReady,match='带边'):
        service.refine(observation,observation['frame'],np.eye(4),surface=surface)


def test_camera_ir_opt_in_and_validation():
    cfg=config();validate(cfg.raw)
    assert camera_settings(cfg)=={} and cfg.raw['fast_stereo_enabled'] is False
    cfg.raw['fast_stereo_enabled']=True
    assert camera_settings(cfg)=={'capture_stereo_ir':True,'enable_frame_sync':True}
    assert cfg.raw['camera']=={}
    with pytest.raises(ValueError):validate({'fast_stereo_enabled':'true'})


def test_startup_failure_closes_backend_before_devices(tmp_path):
    calls=[]
    def fail(*a):calls.append('warmup');raise RuntimeError('no CUDA')
    backend=NS(start=lambda:calls.append('start'),warmup=fail,close=lambda:calls.append('close'))
    with pytest.raises(RuntimeError,match='no CUDA'):
        FastTargetPerception(config(),tmp_path,backend=backend).start()
    assert calls==['start','warmup','close']


def test_cold_warmup_budget_is_separate_from_runtime_timeout(tmp_path):
    timeouts=[]
    backend=NS(start=lambda:None,request_timeout_s=120.,close=lambda:None)
    backend.warmup=lambda shape:timeouts.append(backend.request_timeout_s)
    FastTargetPerception(config(),tmp_path,backend=backend).start()
    assert timeouts==[120.] and backend.request_timeout_s==2.


def test_failed_worker_cannot_cold_start_during_target_processing(tmp_path):
    observation=native_observation();backend=Reconstruction(observation)
    backend.health=lambda:{'warmed':False}
    with pytest.raises(TargetNotReady,match='预热进程不可用'):
        FastTargetPerception(config(),tmp_path,backend=backend).refine(observation,observation['frame'],np.eye(4))
    assert not backend.calls


def test_bad_learned_shape_is_rejected_before_obb_with_quality_evidence(tmp_path):
    observation=native_observation();backend=Reconstruction(observation,height=.09)
    with pytest.raises(TargetNotReady,match='fast_quality_rejected') as error:
        FastTargetPerception(config(),tmp_path,backend=backend).refine(observation,observation['frame'],np.eye(4))
    quality=error.value.details['fast_stereo']['quality_check']
    assert not quality['accepted'] and quality['retained_points']==0


def test_camera_check_config_fast_opens_no_devices(tmp_path,capsys):
    from nero_core.two_second.fixed_freeze_test import main
    assert main(['--config','src/nero_bringup/config/two_second.hardware.legacy_observed.yaml',
        '--belt-speed-cm-s','28.3','--fast-stereo','--check-config','--no-plan',
        '--output',str(tmp_path/'unused')])==0
    report=json.loads(capsys.readouterr().out)
    assert report['fast_stereo_enabled']
    assert report['can_connections']==report['camera_connections']==report['robot_commands_sent']==0
    assert not (tmp_path/'unused').exists()


def test_tracked_path_uses_refined_points_and_original_capture(monkeypatch):
    from test_two_second_diagnostics import vision_fixture
    vision=vision_fixture(monkeypatch,'eligible')
    frame=vision.camera.read_latest();peer=vision.meter.latest_snapshot
    captured=frame.header.received_time-.001
    calls=[]
    def refine(detection, incoming, d2c, **kwargs):
        assert incoming is frame and kwargs['robot_state'] is vision.read_feedback()
        points=detection['object_points'].copy();points[:,2]+=.005
        calls.append(points)
        return dict(detection,object_points=points,fast_stereo={'applied':True})
    vision.fast_stereo=NS(refine=refine)
    vision.clock_epoch=0;vision.progress_snapshot=peer
    try:
        result=vision._prepare_target(frame,np.eye(4),peer,peer.tracks[0],vision.read_feedback(),
            captured,0,vision.model.measured_belt_surface,.05)
        target=result[2]
        assert len(calls)==1 and target.capture_s==pytest.approx(captured)
        np.testing.assert_array_equal(target.points,calls[0])
        assert target.center[2]==pytest.approx(.045)
        assert target.detection['obb']['center_world_m'][2]==pytest.approx(.045)
    finally:vision.close()


@pytest.mark.parametrize('now',[10.3,11.1])
def test_single_frame_refines_before_obb_but_does_not_renew_capture(now):
    from test_fixed_freeze_interface import data
    from nero_core.two_second.fixed_freeze import NativeDepthLocator,frozen_geometry
    from nero_core.two_second.reference import EntryNotReady
    cfg,frame,surface=data()
    candidate=NativeDepthLocator(cfg,surface,frame).locate(frame)[0][0]
    mask=np.zeros(frame.depth_m.shape,bool);mask.ravel()[candidate.seed_pixel_indices]=True
    points=np.tile([-.64,0.,.05],(100,1))
    detection=dict(mask=mask,object_points=points,plane=surface.plane,belt_region=surface.region_metadata())
    calls=[]
    def refine(observation,*args,**kwargs):
        calls.append('refine')
        return dict(observation,object_points=points+[0,0,.005])
    def grasp(refined,*args):
        calls.append('grasp');np.testing.assert_allclose(refined,points+[0,0,.005])
        return [np.eye(4)],{}
    def run():
        return frozen_geometry(cfg,frame,np.eye(4),surface,candidate,10.,clock=lambda:now,
            detect_fn=lambda *a,**k:detection,grasp_fn=grasp,fast_stereo=NS(refine=refine))
    if now>11.:
        with pytest.raises(EntryNotReady):run()
        assert calls==['refine']
    else:
        result=run()
        assert result[0]['frame'] is frame and calls==['refine','grasp']
        np.testing.assert_allclose(result[1],[-.64,0.,.055])


def test_archive_preserves_native_and_refined_clouds_with_ir(tmp_path):
    from test_experiment_archive import target,load
    from nero_core.two_second.experiment_archive import ExperimentArchive
    t=target();original=t.detection['frame']
    frame=replace(original,left_ir=np.ones(original.depth_m.shape,np.uint8),
        right_ir=np.full(original.depth_m.shape,2,np.uint8),stereo_baseline_m=.095,
        left_ir_intrinsics=original.depth_intrinsics,right_ir_intrinsics=original.depth_intrinsics)
    native=t.points.copy();refined=native+[0,0,.005]
    t.points=refined
    t.detection.update(frame=frame,native_object_points=native,
        target_geometry_source='estimated_fast_stereo_visible_depth',fast_stereo={'applied':True})
    archive=ExperimentArchive(tmp_path,lambda *a,**kw:None)
    archive.offer_target(t)
    assert archive.close()['complete']
    meta,arrays=load(next((tmp_path/'experiment').glob('*_target.npz')))
    assert meta['detection']['fast_stereo']['applied'] and meta['stereo_baseline_m']==.095
    np.testing.assert_array_equal(arrays['points'],refined)
    np.testing.assert_array_equal(arrays['native_object_points'],native)
    np.testing.assert_array_equal(arrays['frame_left_ir'],frame.left_ir)


@pytest.mark.parametrize('height,accepted',[(.046,True),(.09,False)])
def test_native_target_postprocessing_matches_full_grid_for_accept_and_reject(height,accepted):
    from mujoco_conveyor_sorting.runtime.grasp_depth import refine_grasp_observation
    from nero_core.two_second.fast_quality import filter_reconstruction
    obs=native_observation()
    # Nonrectangular native mask, with real color and reconstruction holes.
    rgb=obs['frame'].rgb.copy()
    rgb[395:402,702:707]=[0,150,0]
    obs['frame']=replace(obs['frame'],rgb=rgb)
    mask=obs['mask'].copy()
    mask[416:422,717:722]=False
    keep=mask[obs['mask']]
    obs.update(object_points=obs['object_points'][keep],mask=mask)
    class Holes(Reconstruction):
        def refine(self,*args,**kwargs):
            result=super().refine(*args,**kwargs)
            result.reconstructed_depth_m[409:413,715:717]=np.nan
            return result
    backend=Holes(obs,height=height)
    outputs=[refine_grasp_observation(obs,backend,quality_filter=filter_reconstruction,compact_native_geometry=flag)
             for flag in (False, True)]
    a,b=[o['grasp_depth_refinement'] for o in outputs]
    assert a['applied']==b['applied']==accepted
    assert a['reason']==b['reason']
    assert a['crop_xyxy']==b['crop_xyxy']
    assert a['timing_inputs']['geometry_region_xyxy'] is None
    assert b['timing_inputs']['geometry_region_xyxy']==[700,390,730,430]
    assert a['timing_inputs']['hybrid_valid_depth_points']==b['timing_inputs']['hybrid_valid_depth_points']
    aq,bq=[{k:v for k,v in md['quality_check'].items() if k!='wall_s'} for md in (a,b)]
    assert aq==bq
    for key in ('object_points','mask','scene_points'):
        np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
    if accepted:
        np.testing.assert_array_equal(outputs[0]['grasp_target_mask'],outputs[1]['grasp_target_mask'])
        assert outputs[1]['grasp_target_mask'].shape==mask.shape
        assert not outputs[1]['grasp_target_mask'][395:402,702:707].any()

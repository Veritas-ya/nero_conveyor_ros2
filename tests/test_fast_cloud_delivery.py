"""Exercise the Fast cloud handoff with same-capture results, without robot or camera access."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from legacy.test_grasp_depth import Reconstruction, native_observation
from mujoco_conveyor_sorting.perception import deproject_depth_to_world
from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
from nero_core.two_second.fast_perception import FastTargetPerception, validate
from nero_core.two_second.fixed_freeze_pick import main


class TestBackend(Reconstruction):
    __test__ = False
    kind = 'valid'

    def start(self): pass
    def warmup(self, shape): pass
    def close(self): pass

    def refine(self, *args, **kwargs):
        result = super().refine(*args, **kwargs)
        depth = result.reconstructed_depth_m
        if self.kind == 'missing_half':
            depth[390:430, 715:730] = np.nan
        elif self.kind == 'very_sparse':
            depth[self.observation['mask']] = np.nan
            depth[400,710] = .951
        elif self.kind == 'invalid':
            depth[self.observation['mask']] = np.inf
        elif self.kind == 'other_capture':
            result.provenance = {'capture_metadata': {'capture_header': {}}}
        elif self.kind == 'other_camera':
            result.provenance['capture_metadata']['serial_number'] = 'different-camera'
        elif self.kind == 'wrong_crop':
            result.crop_xyxy = (0, 0, 1, 1)
        elif self.kind == 'wrong_grid':
            result.reconstructed_depth_m = depth[:-1]
        elif self.kind == 'small_surface':
            depth[self.observation['mask']] = np.nan
            depth[398:401, 707:714] = .977
            depth[401, 707:710] = [0., -1., np.inf]
        return result


def service(tmp_path, observation, *, mode='all_reconstructed', saving=True):
    cfg = NS(raw=dict(maximum_target_height_m=.12, fast_cloud_delivery_mode=mode), direction=np.array([1.,0.,0.]))
    backend = TestBackend(observation)
    s = FastTargetPerception(cfg, tmp_path, backend=backend, retain_artifacts=saving)
    events = []
    s.event_sink = lambda name, **data: events.append((name,data))
    s.start()
    return s, backend, events


def test_each_nonempty_cloud_is_delivered_in_base_frame_even_when_not_graspable(tmp_path):
    obs=native_observation()
    frame=obs['frame']
    s,backend,events=service(tmp_path,obs)
    try:
        for kind in ('valid','missing_half','very_sparse','invalid','other_capture'):
            backend.kind=kind
            if kind in ('valid','missing_half'):
                d=s.refine(obs,frame,np.eye(4)); meta=d['fast_stereo']
                assert meta['applied'] and meta['quality_check']['accepted']==(kind=='valid')
                assert not meta['quality_check']['enforced_as_delivery_gate']
                assert meta['cloud_delivery']['cloud_id']==(1 if kind=='valid' else 2)
                if kind=='missing_half':
                    assert 'insufficient_native_shape_coverage' in meta['quality_check']['reasons']
                    from nero_core.two_second.vision import obb_grasps
                    grasps,_=obb_grasps(d['object_points'],np.eye(4),np.eye(4))
                    assert len(grasps)==2
            else:
                with pytest.raises(TargetNotReady):
                    s.refine(obs,frame,np.eye(4))
    finally:
        s.close()
    summary=s.cloud_delivery.summary()
    assert summary==dict(delivered=3,saved=3,archive_errors=[])
    delivered=[d for n,d in events if n=='fast_cloud_delivered']
    assert [d['cloud_id'] for d in delivered]==[1,2,3]
    full_pixels=np.flatnonzero(obs['mask'])
    index=full_pixels[full_pixels%frame.depth_m.shape[1]<715]
    for d,ix,z in zip(delivered,[full_pixels,index,np.array([400*1280+710])],[.954,.954,.951]):
        assert d['frame_sequence']==frame.header.sequence and d['clock_epoch']==frame.header.clock_epoch
        f=Path(d['file'])
        with np.load(f,allow_pickle=False) as packet:
            assert json.loads(str(packet['metadata']))['source']=='fast_depth_at_target_pixels'
            np.testing.assert_array_equal(packet['depth_pixel_indices'],ix)
            depths=np.full(frame.depth_m.shape,z,dtype=np.float32)
            truth=deproject_depth_to_world(depths,frame.depth_intrinsics.matrix,frame.camera_to_world_cv,
                                          camera_convention='opencv').reshape(-1,3)[ix]
            np.testing.assert_allclose(packet['points_base_m'],truth,rtol=0.,atol=1e-12)
    assert sum(n=='fast_cloud_quality' for n,_ in events)==3
    assert s.completed==2  # sparse cloud was delivered too, but could not supply a grasp


def test_original_policy_still_rejects_low_coverage_and_no_save_only_delivers_in_memory(tmp_path):
    obs=native_observation();s,backend,events=service(tmp_path,obs,mode='quality_gated')
    try:
        backend.kind='missing_half'
        with pytest.raises(TargetNotReady,match='fast_quality_rejected'):
            s.refine(obs,obs['frame'],np.eye(4))
        assert s.cloud_delivery is None
    finally:s.close()
    s,backend,events=service(tmp_path,obs,saving=False)
    try:
        backend.kind='very_sparse'
        with pytest.raises(TargetNotReady):s.refine(obs,obs['frame'],np.eye(4))
        meta,pts,ix=s.cloud_delivery.latest
        assert len(pts)==len(ix)==1 and not pts.flags.writeable and 'file' not in meta
        assert any(n=='fast_cloud_delivered' for n,_ in events)
    finally:s.close()
    assert not list(tmp_path.rglob('*.npz'))


def test_site_entry_activates_cloud_delivery_and_keeps_motion_checks(capsys):
    cfg=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'
    assert main(['--config',str(cfg),'--belt-speed-cm-s','53.5','--fast-stereo',
                 '--no-depth-separation','--check-config'])==0
    d=json.loads(capsys.readouterr().out)
    assert d['fast_cloud_delivery_mode']=='direct_to_planner'
    assert d['object_relative_approach'] and d['grasp_workpiece_collision_check_enabled']
    assert d['hardware_connections']==d['commands_sent']==0
    with pytest.raises(ValueError):validate({'fast_cloud_delivery_mode':'bogus'})


def test_direct_mode_uses_identical_published_points_and_pixels_for_obb(tmp_path, monkeypatch):
    from nero_core.two_second.vision import obb_grasps, record_grasp_box
    from nero_core.two_second import fast_perception
    obs = native_observation()
    s, backend, events = service(tmp_path, obs, mode='direct_to_planner', saving=False)
    monkeypatch.setattr(fast_perception, 'refine_grasp_observation',
                        lambda *a, **kw: pytest.fail('direct input ran the legacy target filters'))
    try:
        outputs = []
        for kind in ('missing_half', 'small_surface'):
            backend.kind = kind
            refined = s.refine(obs, obs['frame'], np.eye(4))
            packet, points, indices = s.cloud_delivery.latest
            assert refined['object_points'] is points and not points.flags.writeable
            assert packet == refined['fast_stereo']['cloud_delivery']
            assert refined['mask'] is obs['mask'] and refined['scene_points'] is obs['scene_points']
            assert refined['native_object_points'] is obs['object_points']
            assert refined['native_target_geometry']['width'] == obs['width']
            np.testing.assert_array_equal(np.flatnonzero(refined['grasp_target_mask']), indices)
            grasps, box = obb_grasps(points, np.eye(4), np.eye(4))
            record_grasp_box(refined, box, grasps, np.median(points,axis=0))
            assert refined['localization_points'] == len(points) == box['visible_point_count']
            assert refined['width'] == min(box['extent_m'][:2])
            np.testing.assert_array_equal(refined['grasp'], grasps[0])
            outputs.append(refined)
        assert len(outputs[1]['object_points']) == 21
        assert outputs[1]['height'] == pytest.approx(.023, abs=1e-7)
        assert len(outputs[0]['object_points']) > 21 and outputs[0]['fast_stereo']['cloud_delivery']['cloud_id']==1
        assert [ev[1]['cloud_id'] for ev in events if ev[0]=='fast_cloud_delivered'] == [1,2]
        assert len(backend.calls)==2
    finally:
        s.close()


@pytest.mark.parametrize('kind,reason,delivered', [
    ('invalid', 'no_finite_positive', 0), ('other_capture', 'capture_mismatch', 0),
    ('other_camera', 'capture_mismatch', 0), ('wrong_crop', 'crop_mismatch', 0),
    ('wrong_grid', 'grid_mismatch', 0), ('very_sparse', '至少需要 21 点', 1)])
def test_direct_mode_publishes_only_valid_exposure_and_rejects_unusable_grasp_input(tmp_path,kind,reason,delivered):
    obs=native_observation()
    s,backend,_=service(tmp_path, obs, mode='direct_to_planner', saving=False)
    try:
        backend.kind=kind
        with pytest.raises(TargetNotReady,match=reason):
            s.refine(obs, obs['frame'], np.eye(4))
        assert s.cloud_delivery.count==delivered and s.completed==0
    finally:
        s.close()


def test_interleaved_detection_keeps_its_own_cloud_identity(tmp_path,monkeypatch):
    import cv2
    from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
    obs = native_observation()
    s, backend, _ = service(tmp_path, obs, mode='direct_to_planner', saving=False)
    hull = np.array([[[-.5,-.5]],[[.5,-.5]],[[.5,.5]],[[-.5,.5]]],np.float32)
    surface = BeltSurface(np.eye(4),np.array([[-.5,.5],[-.5,.5]]),hull,obs['plane'])
    check = cv2.pointPolygonTest
    def another_component(*args):
        backend.kind = 'missing_half'
        s.refine(obs, obs['frame'], np.eye(4))
        return check(*args)
    monkeypatch.setattr(cv2,'pointPolygonTest',another_component)
    try:
        first = s.refine(obs, obs['frame'], np.eye(4), surface=surface)
        packet = first['fast_stereo']['cloud_delivery']
        assert packet['cloud_id']==1 and s.last['cloud_delivery']['cloud_id']==2
        assert packet['point_count']==len(first['object_points'])>s.last['cloud_delivery']['point_count']
    finally:
        s.close()


@pytest.mark.parametrize('mode', ['all_reconstructed','direct_to_planner'])
def test_cloud_is_already_available_if_footprint_disallows_grasp(tmp_path, mode):
    import cv2
    from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
    obs=native_observation()
    s,backend,events=service(tmp_path,obs,mode=mode)
    backend.kind='valid'
    hull=np.array([[.3,-.1],[.5,-.1],[.5,.1],[.3,.1]],np.float32)
    surface=BeltSurface(np.eye(4),np.array([[.3,.5],[-.1,.1]]),cv2.convexHull(hull),obs['plane'])
    try:
        with pytest.raises(TargetNotReady,match='带边'):
            s.refine(obs,obs['frame'],np.eye(4),surface=surface)
    finally:s.close()
    assert s.cloud_delivery.summary()['saved']==1 and s.completed==0
    meta=next(row for event,row in events if event=='fast_cloud_delivered')
    assert Path(meta['file']).is_file()


def test_write_failure_is_reported_and_does_not_discard_live_cloud(tmp_path,monkeypatch):
    obs=native_observation();s,backend,events=service(tmp_path,obs)
    backend.kind='valid'
    def fail(*args,**kwargs):raise OSError('test disk full')
    monkeypatch.setattr(np,'savez',fail)
    try:
        result=s.refine(obs,obs['frame'],np.eye(4))
        assert result['fast_stereo']['cloud_delivery']['point_count']==obs['mask'].sum()
    finally:s.close()
    summary=s.cloud_delivery.summary()
    assert summary['delivered']==1 and summary['saved']==0 and len(summary['archive_errors'])==1
    assert any(name=='fast_cloud_save_error' for name,_ in events)
    assert not list((tmp_path/'clouds').glob('*.npz'))

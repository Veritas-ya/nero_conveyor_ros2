"""Verify one-pass native component ownership and the collision cloud contract."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_fixed_freeze_interface import data
from mujoco_conveyor_sorting.runtime.belt_surface import _geometry_fingerprint
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, TargetNotReady
from nero_core.two_second import native_single_pass as mod
from nero_core.two_second.fixed_freeze import frozen_geometry
from nero_core.two_second.tracked_single_exposure import use_single_extraction


def fixture(*, low=False):
    cfg, old, surface = data()
    cfg.raw.update(native_single_geometry_mode='foreground_once', freeze_source='single_exposure_native_depth',
        freeze_position_upstream_m=None, tracked_freeze_range='detector_footprint',
        tracked_freeze_publish_before_upstream_m=.6,tracked_freeze_upstream_limit_m=.75,
        native_single_freeze_policy='upstream_exposure',native_single_min_capture_upstream_m=.45,
        native_single_min_height_m=.025,perception_detector_workspace_y_m=[-.7,1.1],
        depth_separation_enabled=True)
    cfg.settings=NS(open_width_m=.1)
    z=np.ones_like(old.depth_m);rgb=np.zeros_like(old.rgb);rgb[:,:,1]=160
    # The two measured parts share one raw depth label via a GREEN bridge.
    z[100:140,45:55]=.95
    rgb[100:110,45:55]=95;rgb[130:140,45:55]=[240,20,10]
    if low:z[130:140,45:55]=.98
    frame=replace(old,depth_m=z,rgb=rgb)
    d2c=np.eye(4)
    surface=replace(surface,geometry_fingerprint=_geometry_fingerprint(frame,d2c),
                    plane={**surface.plane,'inlier_threshold_z_m':.004})
    loc=mod.SinglePassLocator(cfg,surface,frame,d2c,capture_band_m=(.45,.75))
    return cfg,frame,surface,d2c,loc


def finish(c,f,s,d,candidate,**kwargs):
    return frozen_geometry(c,f,d,s,candidate,10.,clock=lambda:10.1,**kwargs)[0]


@pytest.mark.parametrize('tilt, bounds', [
    (.31, [[-.12, .19], [-.16, .04]]),
    (-.20, [[-.95, .01], [-.05, .90]]),
    (.44, [[2., 3.], [2., 3.]]),
])
def test_belt_scene_projection_matches_full_depth_grid(tilt, bounds):
    from scipy.spatial.transform import Rotation
    k = NS(height=91, width=133, fx=114., fy=117., cx=62.5, cy=47.)
    camera, plane = np.eye(4), np.eye(4)
    camera[:3,:3] = Rotation.from_euler('xyz', [.10, -.05, .19]).as_matrix()
    camera[:3,3] = [-.09, .08, .02]
    plane[:3,:3] = Rotation.from_euler('xyz', [.05, tilt, -.3]).as_matrix()
    plane[:3,3] = [.05, -.04, .87]
    rr, cc = np.indices((k.height, k.width))
    n = camera[:3,:3].T @ plane[:3,2]
    rate = (cc-k.cx)*n[0]/k.fx + (rr-k.cy)*n[1]/k.fy + n[2]
    origin = (camera[:3,3]-plane[:3,3]) @ plane[:3,2]
    levels = np.array([0., -.00499999, .00500001, -.006, .005000011])
    z = ((levels[(rr+cc) % len(levels)] - origin)/rate).astype(np.float32)
    z[::9, ::3] = 0
    frame = NS(depth_m=z, camera_to_world_cv=camera, depth_intrinsics=k)
    box = np.asarray(bounds)
    loc = mod.SinglePassLocator.__new__(mod.SinglePassLocator)
    loc.height_rate, loc.origin_height = rate, origin
    loc.surface = NS(plane_to_base=plane, scene_bounds_xy_m=box, bounds_xy_m=box)
    actual, ids, n_valid = loc._scene(frame)
    rows, cols = np.nonzero((z > 0) & (np.abs(z*rate+origin) <= .005+1.1e-8))
    points = mod._base_points(frame, rows, cols, z[rows,cols])
    local = (points-plane[:3,3]) @ plane[:3,:3]
    selected = ((abs(local[:,2]) <= .005+1e-8) & np.all(
        (local[:,:2] >= box[:,0]-1e-8) & (local[:,:2] <= box[:,1]+1e-8), axis=1))
    np.testing.assert_array_equal(ids, rows[selected]*k.width+cols[selected])
    np.testing.assert_array_equal(actual, points[selected])
    assert n_valid == np.count_nonzero(z > 0)
    assert not ids.flags.writeable and not actual.flags.writeable


def test_belt_scene_crossing_camera_plane_uses_whole_exposure():
    frame = NS(camera_to_world_cv=np.eye(4))
    assert mod._belt_scene_window(frame, np.eye(4), np.array([[-1.,1.],[-1.,1.]])) == np.s_[:, :]


def test_split_after_cleanup_extracts_once_and_freeze_reuses_complete_disjoint_masks(monkeypatch):
    cfg,f,s,d,loc=fixture()
    calls=[];real=mod.detect_target
    def count(*a,**kw):
        calls.append(True);return real(*a,**kw)
    monkeypatch.setattr(mod,'detect_target',count)
    cs,report=loc.locate(f)
    assert report['full_extractions']==len(calls)==1 and len(cs)==2
    def forbidden(*a,**kw): pytest.fail('a prepared candidate was re-detected')
    monkeypatch.setattr('mujoco_conveyor_sorting.runtime.single_pick.detect_target',forbidden)
    monkeypatch.setattr('mujoco_conveyor_sorting.runtime.local_target_geometry.detect_seed_local',forbidden)
    aa,bb=[finish(cfg,f,s,d,c) for c in cs]
    assert not (aa['mask']&bb['mask']).any()
    assert len(aa['object_points'])==len(bb['object_points'])==100
    assert aa['scene_points'] is bb['scene_points'] and aa['scene_pixel_indices'] is bb['scene_pixel_indices']
    assert not aa['scene_points'].flags.writeable and len(aa['scene_points'])>10000
    for c,a in zip(cs,(aa,bb)):
        np.testing.assert_array_equal(np.flatnonzero(a['mask']),c.seed_pixel_indices)
        assert a['object_points'] is c.prepared.geometry['object_points']
        assert len(a['object_points'][a['grasp_point_keep_mask']])==a['obb']['visible_point_count']


def test_touched_trigger_boundary_expands_before_the_only_extraction():
    cfg,f,s,d,loc=fixture()
    y0,y1,x0,x1=loc.crop
    loc.crop=(105,115,46,51)
    sl=np.s_[105-y0:115-y0,46-x0:51-x0]
    loc.seed_near=loc.seed_near[sl];loc.seed_far=loc.seed_far[sl]
    cs,stats=loc.locate(f)
    assert stats['expanded'] and stats['full_extractions']==1
    assert sorted(len(c.seed_pixel_indices) for c in cs)==[100,100]


def test_undersize_component_does_not_veto_a_sibling_in_the_same_rough_region():
    cfg,f,s,d,loc=fixture(low=True)
    cs,report=loc.locate(f)
    assert len(cs)==1 and len(cs[0].prepared.geometry['object_points'])==100
    assert 'target_size' in [e['reason'] for e in report['component_rejections']]


@pytest.mark.parametrize('changed', ['frame','surface','extrinsics','state','model','config'])
def test_prepared_geometry_cannot_cross_exposures_or_feedback(changed):
    cfg,f,s,d,loc=fixture()
    cs,_=loc.locate(f)
    kw={}
    if changed=='frame': f=replace(f,header=replace(f.header,sequence=f.header.sequence+1))
    if changed=='surface': s=replace(s)
    if changed=='config': cfg=NS(**vars(cfg))
    if changed=='extrinsics': d=d.copy();d[0,3]=.001
    if changed in ('state','model'):kw=dict(robot_state={},robot_model=NS())
    with pytest.raises(PickRejected):finish(cfg,f,s,d,cs[0],**kw)


def test_footprint_gate_reads_the_full_prepared_object():
    cfg,f,s,d,loc=fixture()
    cs,_=loc.locate(f)
    cfg.raw['perception_near_camera_footprint_margin_m']=.027
    # Put the camera-facing side 20 mm outside a component's upper Y edge.
    c=cs[0];upper=c.prepared.geometry['object_points'][:,1].max()+.020
    hull=s.hull_xy.copy();hull[:,:,1]=np.minimum(hull[:,:,1],upper)
    s=replace(s,hull_xy=hull);loc=mod.SinglePassLocator(cfg,s,f,d,capture_band_m=(.45,.75))
    cs,_=loc.locate(f)
    with pytest.raises(TargetNotReady) as failure:finish(cfg,f,s,d,cs[0])
    assert failure.value.details['reason']=='near_camera_footprint'


def test_fast_reuses_native_extraction_and_rgb_completion_uses_the_full_outline():
    cfg,*_=fixture()
    cfg.raw.update(fast_stereo_enabled=True,depth_separation_enabled=False)
    assert use_single_extraction(cfg)
    cfg.raw['rgb_cloud_completion_enabled']=True
    assert not use_single_extraction(cfg)
    cfg.raw.update(rgb_cloud_completion_enabled=False,freeze_source='tracked')
    assert not use_single_extraction(cfg)


def test_prepared_fast_target_uses_reconstructed_center_and_native_belt_scene():
    cfg,frame,surface,d2c,locator=fixture()
    cfg.raw.update(fast_stereo_enabled=True,depth_separation_enabled=False)
    candidate=locator.locate(frame)[0][0]
    prepared=candidate.prepared
    native=prepared.geometry['object_points']
    scene,indices,valid=prepared.scene()
    # A retained native candidate is not a Fast result. Preserve this distinction on failures, too.
    with pytest.raises(PickRejected,match='重建后端'):finish(cfg,frame,surface,d2c,candidate)
    def learned(points):
        def refine(observation,exposure,transform,**kwargs):
            assert exposure is frame and observation is prepared.geometry
            return dict(observation,object_points=points)
        return NS(refine=refine)
    refined=native[1:]+[.001,0.,0.]
    det,pos,_,_,_=frozen_geometry(cfg,frame,d2c,surface,candidate,10.,clock=lambda:10.1,
                                  fast_stereo=learned(refined))
    np.testing.assert_array_equal(pos,np.median(refined,axis=0))
    np.testing.assert_array_equal(det['object_points'],refined)
    np.testing.assert_array_equal(det['scene_points'],scene)
    np.testing.assert_array_equal(det['scene_pixel_indices'],indices)
    assert det['scene_input_count_before_prefilter']==valid-len(native)-prepared.geometry['excluded_depth_pixels']
    from nero_core.two_second.reference import EntryNotReady
    with pytest.raises(EntryNotReady):
        finish(cfg,frame,surface,d2c,candidate,fast_stereo=learned(refined+cfg.direction*.5))


def test_depth_window_uses_parent_rgb_and_reuses_the_green_mask(monkeypatch):
    from mujoco_conveyor_sorting.runtime.frame_products import frame_products, green_rgb_mask
    cfg, frame, surface, d2c, locator = fixture()
    green = green_rgb_mask(frame, d2c)
    crop = mod._depth_window(frame, d2c, (80, 142, 43, 75))
    assert crop.rgb is frame.rgb and np.shares_memory(crop.depth_m, frame.depth_m)
    assert crop.header is frame.header and green_rgb_mask(crop, d2c) is green
    assert frame_products(crop, d2c).calls.get('rgb_to_hsv', 0) == 0


def test_support_at_window_boundary_uses_full_connected_native_neighbourhood():
    from mujoco_conveyor_sorting.runtime.shape_pick import filter_target_mask
    from mujoco_conveyor_sorting.runtime.frame_products import world_points
    cfg, frame, surface, d2c, loc = fixture()
    mask = np.zeros(frame.depth_m.shape, bool)
    mask[100:110,45:55] = True
    mask[130:140,45:55] = True
    seed = np.zeros_like(mask)
    seed[100:104,45:55] = True
    points = world_points(frame,d2c)
    clean, stats = filter_target_mask(frame,d2c,mask,points,frame.camera_to_world_cv,support_seed=seed)
    expected = np.zeros_like(mask)
    expected[100:110,45:55] = True
    reference, original = filter_target_mask(frame,d2c,expected,points,frame.camera_to_world_cv)
    np.testing.assert_array_equal(clean,reference)
    assert stats['support_radius_m'] == original['support_radius_m']
    assert stats['outside_support_context'] == 100 and stats['green_rejected'] == 0

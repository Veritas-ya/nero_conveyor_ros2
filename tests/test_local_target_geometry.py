"""Real local/full detector comparisons and conservative fallback; no devices."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, _geometry_fingerprint
from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local, seed_touches_boundary
from mujoco_conveyor_sorting.runtime.single_pick import detect_target
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics, RGBDFrame, SampleHeader
from mujoco_conveyor_sorting.runtime.visible_target import project_rgb_labels
from test_two_second_hardware import site


def example():
    depth = np.ones((240,320),np.float32)
    depth[104:122,148:172] = .95
    rgb = np.full((240,320,3),[0,150,0],np.uint8)
    rgb[104:122,148:172] = [150,150,150]
    k = CameraIntrinsics(320,240,300.,300.,160.3,120.1,(0.,)*5)
    camera = np.diag([1.,-1.,-1.,1.]);camera[2,3] = 1.
    frame = RGBDFrame(SampleHeader(1.,1.01,1,1.,0),depth,k,rgb,k,camera)
    hull = np.array([[-.55,-.4],[.55,-.4],[.55,.4],[-.55,.4]],np.float32)
    surface = BeltSurface(np.eye(4),np.array([[-.55,.55],[-.4,.4]]),cv2.convexHull(hull),
        {'normal_base':[0.,0.,1.],'inlier_rms_m':0.,'tilt_from_base_xy_deg':0.,'inlier_threshold_z_m':.004},
        geometry_fingerprint=_geometry_fingerprint(frame,np.eye(4)))
    seed = np.flatnonzero(depth<1)
    return frame, dict(target_shape='simple',measured_surface=surface,
        scene_scope='measured_belt_surface_only',target_seed_pixel_indices=seed,
        maximum_target_length_m=.20, maximum_target_height_m=.12,
        fit_plane=lambda *a:pytest.fail('reuse measured plane'),
        green_belt_points=lambda *a:pytest.fail('reuse measured plane'))


def compare(frame, kwargs):
    rgb,depth = frame.rgb.copy(),frame.depth_m.copy()
    expected = detect_target(replace(frame),np.eye(4),**kwargs)
    result = detect_seed_local(replace(frame),np.eye(4),detector=detect_target,**kwargs)
    for key in ('mask','object_points','scene_points','scene_pixel_indices'):
        np.testing.assert_array_equal(result[key],expected[key])
    for key in ('grasp','center','height','width','footprint_m'):
        np.testing.assert_allclose(result[key],expected[key],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(frame.rgb,rgb)
    np.testing.assert_array_equal(frame.depth_m,depth)
    return result


def test_local_workpiece_retains_full_scene_and_native_pixel_indices():
    frame,kw = example(); result = compare(frame,kw)
    assert result['local_processing']['applied']
    assert result['local_processing']['local_depth_pixels'] < frame.depth_m.size/8
    assert len(result['scene_points']) > 50000
    np.testing.assert_array_equal(result['scene_points'],
                                  result['points'].reshape(-1,3)[result['scene_pixel_indices']])


def test_long_component_crossing_crop_falls_back_instead_of_truncating():
    frame,kw = example()
    depth,rgb = frame.depth_m.copy(),frame.rgb.copy()
    depth[104:122,114:204] = .95;rgb[104:122,114:204] = 150
    frame = replace(frame,depth_m=depth,rgb=rgb)
    kw['maximum_target_length_m'] = .30
    result = compare(frame,kw)
    assert not result['local_processing']['applied']
    assert result['local_processing']['fallback_reason']=='foreground_touches_crop_boundary'


def test_low_rgb_connected_extension_outside_crop_is_not_lost():
    frame,kw = example();rgb = frame.rgb.copy()
    # The extension lies on the belt, below the tall-component trigger.
    rgb[110:115,171:220] = 150
    depth = frame.depth_m.copy()
    # Missing native depth at the crop boundary hides the continuation from
    # a simple edge test. The global RGB/depth proof must still find it.
    depth[110:115,198:208] = 0.
    frame = replace(frame,rgb=rgb,depth_m=depth)
    result = compare(frame,kw)
    assert not result['local_processing']['applied']
    assert result['local_processing']['fallback_reason']=='rgb_outline_has_depth_outside_crop'
    assert np.count_nonzero(result['mask'][:,205:]) > 0


def test_unrelated_component_at_crop_edge_does_not_truncate_selected_workpiece():
    frame,kw = example();depth,rgb = frame.depth_m.copy(),frame.rgb.copy()
    depth[80:90,193:214] = .95;rgb[80:90,193:214] = 150
    frame = replace(frame,depth_m=depth,rgb=rgb)
    result = compare(frame,kw)
    assert result['local_processing']['applied']


def test_open_color_outline_uses_full_detector():
    frame,kw = example();frame=replace(frame,rgb=np.full(frame.rgb.shape,150,np.uint8))
    result = compare(frame,kw)
    assert result['local_processing']['fallback_reason']=='no_closed_rgb_outline'


def test_unseeded_and_general_scene_modes_keep_original_detector():
    frame,kw = example()
    for changed in ({'target_seed_pixel_indices':None},{'scene_scope':'observed_scene'}):
        result=compare(frame,{**kw,**changed})
        assert result['local_processing']['fallback_reason']=='unsupported_input_or_mode'


def test_diagonal_component_connectivity_and_empty_seed_fail_closed():
    foreground=np.zeros((12,12),bool);foreground[4:8,4:8]=True
    assert not seed_touches_boundary(foreground,np.array([5]),np.array([5]))
    foreground[np.arange(5),np.arange(5)]=True
    assert seed_touches_boundary(foreground,np.array([5]),np.array([5]))
    assert seed_touches_boundary(foreground,np.array([10]),np.array([3]))


@pytest.mark.parametrize('distortion', [(0.,)*5,(.01,-.004,.0001,-.0002,.0005)])
def test_crop_rgb_projection_preserves_absolute_pixels(distortion):
    frame,_=example()
    frame=replace(frame,color_intrinsics=replace(frame.color_intrinsics,distortion=distortion))
    y0,y1,x0,x1=80,150,120,205
    k=frame.depth_intrinsics
    cropped=replace(frame,depth_m=frame.depth_m[y0:y1,x0:x1],depth_intrinsics=replace(k,
        width=x1-x0,height=y1-y0,cx=k.cx-x0,cy=k.cy-y0))
    object.__setattr__(cropped,'_depth_pixel_origin',(y0,x0))
    object.__setattr__(cropped,'_uncropped_depth_intrinsics',k)
    labels=np.arange(frame.rgb.shape[0]*frame.rgb.shape[1],dtype=np.int32).reshape(frame.rgb.shape[:2])
    expected=project_rgb_labels(frame,np.eye(4),labels)
    actual=project_rgb_labels(cropped,np.eye(4),labels)
    np.testing.assert_array_equal(actual,expected[y0:y1,x0:x1])


def test_diagnostics_receive_full_native_masks():
    frame,kw=example();rows=[]
    result=detect_seed_local(frame,np.eye(4),detector=detect_target,diagnostic=rows.append,**kw)
    assert result['local_processing']['applied'] and rows
    assert all(r['mask'].shape==frame.depth_m.shape for r in rows if 'mask' in r)


@pytest.mark.parametrize('value',[0,1,'false',None])
def test_local_switch_requires_a_boolean(site,value):
    import yaml
    from nero_core.two_second.config import CellConfig
    path,raw=site;raw['local_target_geometry_enabled']=value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='local_target_geometry_enabled'):CellConfig.load(path)


def test_original_rejection_preserved_with_fallback_diagnostics():
    from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
    frame,kw=example();kw['target_seed_pixel_indices']=None
    original=TargetNotReady('original rejection',reason='no_workpiece',details={'count':0})
    def reject(*a,**k):raise original
    with pytest.raises(TargetNotReady) as caught:
        detect_seed_local(frame,np.eye(4),detector=reject,**kw)
    assert caught.value is original and caught.value.details['reason']=='no_workpiece'
    assert caught.value.details['local_processing']['fallback_reason']=='unsupported_input_or_mode'

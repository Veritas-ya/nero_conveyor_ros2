"""A configured early exposure must survive the old quantile search crop."""
from dataclasses import replace

import numpy as np
import pytest

from test_fixed_freeze_interface import data
from nero_core.two_second.fixed_freeze import (
    NativeDepthLocator,fixed_freeze_detection_surface,frozen_geometry)
from nero_core.two_second.measured_rgbd import fit_plane,green_belt_points
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.single_pick import detect_target,TargetNotReady


def early_part(monkeypatch,second=False):
    config,frame,surface=data()
    config.raw['freeze_position_upstream_m']=.50
    depth=np.ones_like(frame.depth_m)
    depth[115:125,45:55]=.95
    if second:depth[115:125,110:120]=.95
    frame=replace(frame,depth_m=depth)
    # The upstream part is entirely outside the 2/98% quantile rectangle,
    # but within the measured green belt hull and actual native depth view.
    surface=replace(surface,bounds_xy_m=np.array([[-.64,.8],[-.5,.5]]))
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *a:True)
    return config,frame,surface


def full(frame,surface,**kwargs):
    return detect_target(frame,np.eye(4),fit_plane=fit_plane,green_belt_points=green_belt_points,
        target_shape='simple',measured_surface=surface,**kwargs)


def test_50cm_part_outside_quantile_roi_has_real_complete_cloud_without_moving_freeze(monkeypatch):
    config,frame,surface=early_part(monkeypatch)
    original=frame.depth_m.copy();bounds=surface.bounds_xy_m.copy()
    with pytest.raises(TargetNotReady) as error:full(frame,surface)
    assert error.value.details['reason']=='no_workpiece'
    locator=NativeDepthLocator(config,surface,frame)
    candidates,report=locator.locate(frame)
    assert len(candidates)==1 and candidates[0].point_count==100
    detection,position,velocity,grasps,overlap=frozen_geometry(
        config,frame,np.eye(4),surface,candidates[0],10.,clock=lambda:10.2)
    assert .48<=config.follow_bounds[0]-position@config.direction<=.50
    assert len(detection['object_points'])==100 and overlap==1.
    assert len(grasps)>0
    np.testing.assert_array_equal(surface.bounds_xy_m,bounds)
    np.testing.assert_array_equal(detection['belt_region']['bounds_xy_m'],bounds)
    np.testing.assert_array_equal(frame.depth_m,original)
    assert report['detection_bounds_source']=='measured_native_belt_hull'


def test_same_exposure_seed_selects_early_part_and_keeps_other_part_in_scene(monkeypatch):
    config,frame,surface=early_part(monkeypatch,second=True)
    search=fixed_freeze_detection_surface(surface)
    with pytest.raises(TargetNotReady) as error:full(frame,search)
    assert error.value.details['reason']=='multiple_candidates'
    candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    result=full(frame,search,target_seed_pixel_indices=candidate.seed_pixel_indices)
    selection=result['same_exposure_seed_selection']
    assert selection['candidate_count_before_selection']==2
    assert selection['candidate_count_after_selection']==1
    assert selection['candidate_overlap_ratios']==[1.,0.]
    assert len(result['object_points'])==100
    assert np.all(result['mask'][115:125,45:55])
    other=np.zeros_like(frame.depth_m,bool);other[115:125,110:120]=True
    assert np.all(np.isin(np.flatnonzero(other),result['scene_pixel_indices']))


def test_unrelated_seed_cannot_select_nearest_or_largest_component(monkeypatch):
    _,frame,surface=early_part(monkeypatch,second=True)
    with pytest.raises(TargetNotReady) as error:
        full(frame,fixed_freeze_detection_surface(surface),target_seed_pixel_indices=np.arange(100))
    assert error.value.details['reason']=='no_workpiece'
    assert error.value.details['same_exposure_seed_selection']['candidate_overlap_ratios']==[0.,0.]


@pytest.mark.parametrize('indices',[[],[-1],[76800],[1.5],[[1,2]]])
def test_invalid_native_seed_indices_rejected_before_extraction(monkeypatch,indices):
    _,frame,surface=early_part(monkeypatch)
    with pytest.raises(ValueError,match='原生深度索引'):
        full(frame,surface,target_seed_pixel_indices=indices)


def test_private_search_copy_preserves_scene_plane_hull_and_original_surface():
    _,_,surface=data();surface.bounds_xy_m=np.array([[-.64,.75],[-.4,.4]])
    copy=fixed_freeze_detection_surface(surface)
    np.testing.assert_array_equal(copy.scene_bounds_xy_m,surface.bounds_xy_m)
    np.testing.assert_array_equal(copy.plane_to_base,surface.plane_to_base)
    np.testing.assert_array_equal(copy.hull_xy,surface.hull_xy)
    assert surface.scene_bounds_xy_m is None

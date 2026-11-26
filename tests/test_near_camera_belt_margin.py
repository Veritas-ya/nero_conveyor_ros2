from dataclasses import replace

import cv2
import numpy as np
import pytest

from test_fixed_freeze_interface import data
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, near_camera_belt_clearance
from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
from nero_core.two_second.fixed_freeze import NativeDepthLocator, frozen_geometry
from nero_core.two_second.tracked_freeze import check_near_camera_footprint


@pytest.mark.parametrize('nearest,accepted', [(.012,False),(.027,False),(.0271,True),(.99,True)])
def test_footprint_checks_edge_points_even_with_good_centre(nearest,accepted):
    cfg,frame,surface=data()
    pose=frame.camera_to_world_cv.copy()
    pose[1,3]=-1.
    frame=replace(frame,camera_to_world_cv=pose)
    cfg.raw['perception_near_camera_footprint_margin_m'] = .027
    cloud=np.array([[.10,-.5+nearest,.08],[.10,0.,.02],[.12,0.,.04]])
    detection={'object_points':cloud, 'grasp_point_keep_mask':np.array([False,True,True])}
    if accepted:
        check_near_camera_footprint(cfg,surface,frame,detection)
        assert detection['near_camera_footprint']['points_in_strip']==0
    else:
        with pytest.raises(TargetNotReady) as exc:
            check_near_camera_footprint(cfg,surface,frame,detection)
        assert exc.value.details['reason']=='near_camera_footprint'
        assert exc.value.details['points_in_strip']==1
        assert exc.value.details['minimum_clearance_m']==pytest.approx(nearest)


def test_final_geometry_rejects_cloud_with_centre_beyond_27mm(monkeypatch):
    cfg,frame,surface=data()
    cfg.raw.update(perception_belt_edge_margin_m=.025,perception_near_camera_footprint_margin_m=.027)
    # Camera faces +Y of this asymmetric hull. Component centre is 32 mm
    # inside; the component extends a further ~9.5 mm toward the edge.
    k=replace(frame.depth_intrinsics,fy=450.)
    frame=replace(frame,depth_intrinsics=k,color_intrinsics=k)
    centre_y=-(119.5-120)*.95/450.
    upper=centre_y+.032
    hull=cv2.convexHull(np.array([[-.9,-.3],[.8,-.3],[.8,upper],[-.9,upper]],np.float32))
    surface=replace(surface,hull_xy=hull,bounds_xy_m=np.array([[-.9,.8],[-.3,upper]]))
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *a:True)
    candidates,_=NativeDepthLocator(cfg,surface,frame).locate(frame)
    with pytest.raises(TargetNotReady) as exc:
        frozen_geometry(cfg,frame,np.eye(4),surface,candidates[0],10.,clock=lambda:10.2)
    assert exc.value.details['reason']=='near_camera_footprint'


@pytest.mark.parametrize('reverse', [False, True])
def test_only_camera_facing_lateral_edge_is_measured(reverse):
    _, _, surface = data()
    direction = [-1., 0., 0.] if reverse else [1., 0., 0.]
    assert near_camera_belt_clearance(surface, [0., -.49], [0., -1., 1.], direction) == pytest.approx(.01)
    assert near_camera_belt_clearance(surface, [0., .49], [0., -1., 1.], direction) == pytest.approx(.99)
    assert near_camera_belt_clearance(surface, [-.899, 0.], [0., -1., 1.], direction) == pytest.approx(.5)


@pytest.mark.parametrize('far,clearance,accepted', [
    (False, .010, False), (False, .017, False), (False, .020, True), (True, .010, True)])
def test_both_detection_stages_share_one_sided_margin(monkeypatch, far, clearance, accepted):
    config, frame, surface = data()
    config.raw.update(perception_belt_edge_margin_m=.005, perception_belt_near_camera_margin_m=.017)
    depth = np.ones_like(frame.depth_m)
    row = 200 if far else 115
    depth[row:row+10, 53:63] = .95
    intrinsics = replace(frame.depth_intrinsics, fy=450.)
    frame = replace(frame, depth_m=depth, depth_intrinsics=intrinsics, color_intrinsics=intrinsics)
    center_y = -(row+4.5-120)*.95/450.
    low, high = (center_y-clearance, -.02) if far else (-.3, center_y+clearance)
    hull = cv2.convexHull(np.array([[-.9,low],[.8,low],[.8,high],[-.9,high]], np.float32))
    surface = replace(surface, hull_xy=hull, bounds_xy_m=np.array([[-.9,.8],[low,high]]))
    monkeypatch.setattr(BeltSurface, 'matches_frame', lambda *a: True)
    # Seed from the same exposure with the old two-sided 5 mm policy.
    config.raw['perception_belt_near_camera_margin_m'] = 0.
    seed = NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    config.raw['perception_belt_near_camera_margin_m'] = .017
    candidates, report = NativeDepthLocator(config,surface,frame).locate(frame)
    assert bool(candidates) is accepted
    if accepted:
        detection, *_ = frozen_geometry(config,frame,np.eye(4),surface,seed,10.,clock=lambda:10.2)
        assert len(detection['object_points']) == 100
    else:
        assert report['size_decisions'][0]['reason'] == 'near_camera_belt_edge'
        with pytest.raises(TargetNotReady) as caught:
            frozen_geometry(config,frame,np.eye(4),surface,seed,10.,clock=lambda:10.2)
        assert caught.value.details['near_camera_edge_decisions'][0]['accepted'] is False

"""Measured search geometry: reject image background outside the belt footprint."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import yaml

from nero_core.two_second.perception import make_meter
from mujoco_conveyor_sorting.runtime.nero7_tracking_roi import upstream_rgb_roi
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics, RGBDFrame, SampleHeader
from vendor.nero7_speed_reference.nero7_perception.geometry import CameraGeometry


def geometry():
    surface = SimpleNamespace(plane_to_base=np.eye(4),
        hull_xy=np.array([[-.5,-.25],[.5,-.25],[.5,.25],[-.5,.25]],np.float32),
        bounds_xy_m=np.array([[-.5,.5],[-.25,.25]]))
    transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=1.
    camera=CameraGeometry(np.array([[100.,0.,100.],[0.,100.,60.],[0.,0.,1.]]),200,120,'color_optical')
    return surface,camera,transform


def test_plane_footprint_rejects_background_admitted_by_the_height_prism():
    surface,camera,transform=geometry()
    footprint=upstream_rgb_roi(surface,camera,transform,[1.,0.,0.],0.,0.)
    prism=upstream_rgb_roi(surface,camera,transform,[1.,0.,0.],0.,.12)
    assert footprint[60,75]  # Point on the upstream belt.
    assert not footprint[60,130]  # Downstream of the entry.
    assert prism[32,75] and not footprint[32,75]  # Above the visible belt edge.
    assert prism[60,45] and not footprint[60,45]  # Outside its upstream end.
    assert np.all(~footprint | prism)


def test_factory_uses_footprint_mask_without_changing_height_limit_or_source_images():
    surface,camera,transform=geometry()
    intrinsic=CameraIntrinsics(width=200,height=120,fx=100.,fy=100.,cx=100.,cy=60.)
    frame=RGBDFrame(header=SampleHeader(1.,1.,1,1.,0),depth_m=np.ones((120,200)),
        rgb=np.full((120,200,3),[0,150,0],np.uint8),depth_intrinsics=intrinsic,
        color_intrinsics=intrinsic,camera_to_world_cv=transform)
    cfg=SimpleNamespace(direction=np.array([1.,0.,0.]),follow_bounds=np.array([0.,.38]),raw={
        'perception_detection_mode':'rgb_depth','perception_plane_mode':'ransac3d',
        'perception_search_region':'measured_belt_footprint',
        'maximum_measured_speed_m_s':1.,'maximum_target_height_m':.12})
    rgb_before,depth_before=frame.rgb.copy(),frame.depth_m.copy()
    meter=make_meter(cfg,surface);meter._geometry(frame,np.eye(4))
    assert meter.tracking_roi_mode=='measured_belt_footprint'
    assert meter.parameters['block_height_max']==.12
    assert meter.tracking_rgb_roi[60,75] and not meter.tracking_rgb_roi[32,75]
    # The rectangular crop must not substitute for the polygon search mask.
    assert np.count_nonzero(meter.tracking_rgb_roi)<200*120//2
    np.testing.assert_array_equal(frame.rgb,rgb_before)
    np.testing.assert_array_equal(frame.depth_m,depth_before)


def test_invalid_footprint_projection_does_not_expand_search_to_the_whole_image():
    surface,camera,transform=geometry();transform[2,3]=.02
    with pytest.raises(ValueError,match='近平面'):
        upstream_rgb_roi(surface,camera,transform,[1.,0.,0.],0.,0.)


@pytest.mark.parametrize('height',[np.nan,-.01])
def test_invalid_projection_height_rejected(height):
    surface,camera,transform=geometry()
    with pytest.raises(ValueError,match='投影高度'):
        upstream_rgb_roi(surface,camera,transform,[1.,0.,0.],0.,height)


@pytest.mark.parametrize('name',[
    'two_second.hardware.legacy_observed.yaml','two_second.hardware.tracked_freeze.yaml'])
def test_field_configs_reference_the_adopted_measured_direction_and_footprint(name):
    config=Path(__file__).parents[1]/'src/nero_bringup/config'
    site=yaml.safe_load((config/name).read_text())
    provenance=json.loads((config/'calibration/belt_direction_20261109.json').read_text())
    np.testing.assert_allclose(site['belt_direction_base'],provenance['belt_direction_base'],atol=1e-12)
    assert np.linalg.norm(site['belt_direction_base'])==pytest.approx(1.)
    assert provenance['sample_count']==12
    assert site['perception_search_region']=='measured_belt_footprint'

from dataclasses import replace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from test_two_second_hardware import cloud, site
from test_fixed_freeze_interface import data
from nero_core.two_second.config import CellConfig
from nero_core.two_second.vision import obb_grasps, MeasuredTarget
from nero_core.two_second.fixed_freeze import NativeDepthLocator, frozen_geometry
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface


def test_offset_moves_only_tcp_in_base_z_on_tilted_belt():
    plane = np.eye(4)
    plane[:3,:3] = Rotation.from_euler('xyz', [.07, -.05, .6]).as_matrix()
    plane[:3,3] = [.4,-.2,.1]
    points = cloud() @ plane[:3,:3].T + plane[:3,3]
    original = points.copy()
    nominal, box = obb_grasps(points,plane,np.eye(4))
    adjusted, lowered_box = obb_grasps(points,plane,np.eye(4),tcp_z_offset_m=-.005)
    for a,b in zip(nominal,adjusted):
        expected = a.copy(); expected[2,3] -= .005
        np.testing.assert_allclose(b,expected,atol=1e-12)
    np.testing.assert_array_equal(points,original)
    for key in ('center_world_m','axes_world_columns','extent_m'):
        np.testing.assert_array_equal(lowered_box[key],box[key])
    def target(grasps,obb):
        return MeasuredTarget((0,1,2),10.,10.01,5.,0,3,np.median(points,axis=0),
            np.array([.1,0.,0.]),tuple(grasps),points,
            {'belt_region':{'plane_to_base':plane},'obb':obb},{},{'source':'nero7_xy_kalman'})
    before,after=target(nominal,box),target(adjusted,lowered_box)
    np.testing.assert_array_equal(before.object_pose,after.object_pose)
    for a,b in zip(before.grasps_object,after.grasps_object):
        np.testing.assert_allclose(b[:3,3]-a[:3,3],after.object_pose[:3,:3].T@[0.,0.,-.005],atol=1e-12)


def test_single_frame_geometry_applies_offset_once(monkeypatch):
    config,frame,surface=data()
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *a:True)
    candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    nominal=frozen_geometry(config,frame,np.eye(4),surface,candidate,10.,clock=lambda:10.2)
    config.raw['grasp_tcp_z_offset_m']=-.005
    adjusted=frozen_geometry(config,frame,np.eye(4),surface,candidate,10.,clock=lambda:10.2)
    np.testing.assert_array_equal(nominal[0]['object_points'],adjusted[0]['object_points'])
    np.testing.assert_array_equal(nominal[1],adjusted[1])
    for a,b in zip(nominal[3],adjusted[3]):
        np.testing.assert_allclose(b[:3,3]-a[:3,3],[0.,0.,-.005],atol=1e-12)


@pytest.mark.parametrize('offset',[True,float('nan'),float('inf'),-.051,.051,'-0.005'])
def test_invalid_offset_rejected_by_config_and_geometry(site,offset):
    path,raw=site
    raw['grasp_tcp_z_offset_m']=offset
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match='grasp_tcp_z_offset_m'):
        CellConfig.load(path)
    with pytest.raises(ValueError,match='grasp_tcp_z_offset_m'):
        obb_grasps(cloud(),np.eye(4),np.eye(4),tcp_z_offset_m=offset)


def test_old_config_defaults_zero_offset(site):
    path,_=site
    assert CellConfig.load(path).raw['grasp_tcp_z_offset_m']==0.


def test_sparse_tail_corrects_both_grasps_only_along_horizontal_belt():
    from nero_core.two_second.fixed_freeze import along_belt_quantile_grasps
    xyz = np.stack(np.meshgrid(np.linspace(-.020,.020,21),np.linspace(-.012,.012,13),[.025,.035]),axis=-1).reshape(-1,3)
    # A few upstream points shift the full OBB centre by -7.5 mm.
    xyz = np.vstack((xyz, [[-.035,-.012,.026],[-.035,.012,.030]]))
    plane = np.eye(4)
    plane[:3,:3] = Rotation.from_euler('zy', [.43,.08]).as_matrix()
    plane[:3,3] = [.3,-.4,.1]
    points = xyz@plane[:3,:3].T + plane[:3,3]
    original = points.copy()
    heading = np.r_[plane[:2,0]/np.linalg.norm(plane[:2,0]),0.]
    before,box = obb_grasps(points,plane,np.eye(4),tcp_z_offset_m=-.005)
    after,meta = along_belt_quantile_grasps(before,box,points,plane,heading)
    delta=heading*.0075*np.linalg.norm(plane[:2,0])
    for a,b in zip(before,after):
        np.testing.assert_allclose(b[:3,3]-a[:3,3],delta,atol=1e-12)
        np.testing.assert_array_equal(b[:3,:3],a[:3,:3])
        assert b[2,3]==a[2,3]
    for key in ('axes_world_columns','extent_m','center_world_m'):
        np.testing.assert_array_equal(meta[key],box[key])
    np.testing.assert_array_equal(points,original)
    np.testing.assert_allclose(meta['grasp_tcp_offset_base_m'],delta+[0.,0.,-.005],atol=1e-12)


@pytest.mark.parametrize('source,active',[('tracked', False),('single_exposure_native_depth', True)])
def test_quantile_center_is_scoped_to_direct_freeze(monkeypatch,source,active):
    config,frame,surface=data()
    config.raw.update(freeze_source=source,native_single_along_belt_quantile_center=True)
    monkeypatch.setattr(BeltSurface,'matches_frame',lambda *a:True)
    candidate=NativeDepthLocator(config,surface,frame).locate(frame)[0][0]
    d,*_=frozen_geometry(config,frame,np.eye(4),surface,candidate,10.,clock=lambda:10.2)
    assert ('along_belt_center' in d['obb'])==active

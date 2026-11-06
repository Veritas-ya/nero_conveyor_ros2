"""Sparse work must preserve every point, full components and their ordering."""
from collections import deque
from dataclasses import fields

import numpy as np
import pytest

from mujoco_conveyor_sorting.perception.rgbd_vision_pipeline import (
    DepthGeometryVisionPipeline, GeometryPrototype, connected_component_masks,
    deproject_depth_to_world)


def reference_components(mask, minimum):
    seen = np.zeros_like(mask)
    result = []
    for row, col in zip(*np.nonzero(mask)):
        if seen[row,col]:
            continue
        queue = deque([(row,col)]);seen[row,col]=True;component=np.zeros_like(mask)
        while queue:
            y,x=queue.popleft();component[y,x]=True
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;queue.append((ny,nx))
        if component.sum()>=minimum:
            result.append(component)
    return result


@pytest.mark.parametrize('shape', [(1,1),(1,39),(31,1),(37,53)])
@pytest.mark.parametrize('density', [0.,.02,.3,1.])
def test_compiled_labels_preserve_original_eight_connectivity_and_scan_order(shape,density):
    rng=np.random.default_rng(47)
    mask=rng.random(shape)<density
    for minimum in (1,3,8):
        expected=reference_components(mask,minimum)
        result=connected_component_masks(mask,minimum_pixels=minimum)
        assert len(result)==len(expected)
        for a,b in zip(result,expected):
            np.testing.assert_array_equal(a,b)


def test_diagonal_links_and_objects_on_image_edges_are_not_truncated():
    mask=np.eye(39,dtype=bool)
    mask[:8,-8:]=True;mask[-8:,:8]=True
    a=connected_component_masks(mask,minimum_pixels=5)
    b=reference_components(mask,5)
    assert len(a)==len(b)==3
    for first,second in zip(a,b):
        np.testing.assert_array_equal(first,second)


@pytest.mark.parametrize('convention', ['opencv','mujoco'])
def test_deprojection_buffer_reuse_is_bit_identical_to_original_arithmetic(convention):
    rng=np.random.default_rng(31)
    depth=rng.uniform(.3,2.,(43,61)).astype(np.float32);depth[::3,::7]=0.
    k=np.array([[417.3,0.,32.2],[0.,421.7,18.6],[0.,0.,1.]])
    transform=np.eye(4)
    transform[:3,:3]=np.linalg.qr(rng.normal(size=(3,3)))[0]
    transform[:3,3]=rng.normal(size=3)
    z=np.asarray(depth,dtype=np.float64);y,x=np.indices(z.shape,dtype=np.float64)
    camera=np.stack(((x-k[0,2])*z/k[0,0],(y-k[1,2])*z/k[1,1],z),axis=-1)
    if convention=='mujoco':
        camera=camera@np.diag([1.,-1.,-1.]).T
    expected=camera@transform[:3,:3].T+transform[:3,3]
    actual=deproject_depth_to_world(depth,k,transform,camera_convention=convention)
    np.testing.assert_array_equal(actual,expected)


def pipeline():
    return DepthGeometryVisionPipeline(
        [GeometryPrototype(category_id=0,dimensions=(.05,.05,.05))],
        support_plane_height=0.,minimum_height=.012,maximum_height=.12,
        minimum_mask_pixels=8,minimum_footprint=.001,maximum_footprint=.2,
        roi_x=[-.01,.01],roi_y=[-.1,.1])


@pytest.mark.parametrize('select_complete', [False,True])
def test_sparse_allowed_points_match_dense_reference_with_holes_and_nonfinite_background(select_complete):
    shape=(140,180);y,x=np.indices(shape)
    points=np.stack(((x-90)*.001,(y-70)*.001,np.full(shape,.04)),axis=-1)
    depth=np.ones(shape)
    allowed=np.zeros(shape,bool);allowed[30:60,60:120]=True;allowed[95:110,88:105]=True
    allowed[40:45,70:80]=False
    points[31,61]=np.nan;points[32,62,2]=.2;depth[33,63]=0.;depth[34,64]=np.inf
    immutable=points.copy();immutable.setflags(write=False)
    # Dense reference excludes disallowed points before its full-grid checks.
    dense_points=points.copy();dense_points[~allowed]=np.nan
    sparse,dense=pipeline(),pipeline();k=np.array([[500.,0.,90.],[0.,500.,70.],[0.,0.,1.]])
    a=sparse.extract_components(None,depth,k,np.eye(4),points_world=immutable,
        foreground_mask=allowed,roi_selects_components=select_complete)
    b=dense.extract_components(None,depth,k,np.eye(4),points_world=dense_points,
        roi_selects_components=select_complete)
    assert len(a)==len(b)==2
    for first,second in zip(a,b):
        for field in fields(first):
            np.testing.assert_array_equal(getattr(first,field.name),getattr(second,field.name))
    np.testing.assert_array_equal(sparse.last_foreground_mask,dense.last_foreground_mask)
    np.testing.assert_array_equal(immutable,points)
    assert sparse.last_component_workload['geometry_evaluated_pixels']==allowed.sum()
    assert dense.last_component_workload['geometry_evaluated_pixels']==depth.size
    if select_complete:
        assert np.count_nonzero(a[0].mask & (points[...,0]<-.01))>0


def test_more_than_two_thousand_points_are_kept_without_sampling_or_point_cap():
    y,x=np.indices((100,100));points=np.stack(((x-50)*.001,(y-50)*.001,np.full((100,100),.05)),axis=-1)
    mask=np.zeros((100,100),bool);mask[25:75,25:75]=True
    p=pipeline()
    result=p.extract_components(None,np.ones((100,100)),np.diag([500.,500.,1.]),np.eye(4),
        points_world=points,foreground_mask=mask,roi_selects_components=True)
    assert len(result)==1 and result[0].valid_depth_pixels==2500
    np.testing.assert_array_equal(result[0].mask,mask)
    assert p.last_component_workload['sampling_applied'] is False

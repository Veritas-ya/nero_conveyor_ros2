import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.graspgenx_sim.bbox_center_grasp import (center_network_tcp_on_visible_box,
    annotate_center_candidates)

ROOT=Path(__file__).resolve().parents[2]
TRANSFORM=np.asarray(json.loads((ROOT/'src/nero_core/experiments/graspgenx_sim/gripper_agx/transforms.json').read_text())['T_canonical_ee_site'])


def fixture_data():
    points=np.random.default_rng(3).uniform([-.7,-.04,.50],[-.6,.03,.56],(200,3))
    poses=np.repeat(np.eye(4)[None,:,:],4,axis=0)
    poses[:,:3,:3]=Rotation.from_euler('xyz',[[0,0,0],[.5,.4,.3],[1.,-1.,.6],[-.4,.7,1.]]).as_matrix()
    poses[:,:3,3]=[[0,.1,.2],[1,1,1],[2,2,2],[3,3,3]]
    return points,poses


def test_network_orientation_and_tcp_offset_are_preserved():
    points,poses=fixture_data();original=poses.copy()
    modified,info=center_network_tcp_on_visible_box(points,poses)
    center=(points.min(axis=0)+points.max(axis=0))/2.
    np.testing.assert_array_equal(poses,original)
    np.testing.assert_array_equal(modified[:,:3,:3],original[:,:3,:3])
    np.testing.assert_allclose((modified@TRANSFORM)[:,:3,3],np.tile(center,(len(poses),1)),atol=1e-12)
    symmetry=np.diag([-1.,-1.,1.,1.])
    np.testing.assert_allclose((modified@symmetry@TRANSFORM)[:,:3,3],np.tile(center,(len(poses),1)),atol=1e-12)
    assert info['score_semantics']=='original_network_pose_score_not_rescored_after_translation'
    assert info['visible_point_count']==len(points)


def test_invalid_geometry_fails_closed():
    points,poses=fixture_data()
    with pytest.raises(ValueError,match='21 finite'):
        center_network_tcp_on_visible_box(points[:20],poses)
    poses[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        center_network_tcp_on_visible_box(points,poses)


def test_network_result_is_translated_before_existing_selector(monkeypatch):
    from experiments.graspgenx_sim import run
    points,poses=fixture_data();scores=np.array([.9,.8,.7,.6])
    class Backend:
        def infer(self,*args):return {'poses':poses.copy(),'scores':scores.copy(),'metadata':{}}
    def selector(snapshot,modified,passed_scores,position,cfg,observed_cloud_world):
        np.testing.assert_allclose((modified@TRANSFORM)[:,:3,3],
            np.tile((points.min(0)+points.max(0))/2.,(len(poses),1)))
        np.testing.assert_array_equal(passed_scores,scores)
        np.testing.assert_array_equal(observed_cloud_world,points)
        return [{'source_index':1,'variant':'original','score':scores[1]}],[]
    monkeypatch.setattr(run,'select_candidates',selector)
    result=run.infer_and_select(Backend(),points,7,4,4,SimpleNamespace(source_timestamp=1.),np.zeros(3),{},
                                grasp_position_mode='visible_bbox_center')
    row=result['accepted_candidates'][0]
    np.testing.assert_array_equal(row['original_network_pose_world'],poses[1])
    assert row['score_semantics']=='original_network_pose_score_not_rescored_after_translation'
    assert row['grasp_position_mode']=='visible_world_axis_aligned_box_center_tcp'


def test_oriented_box_rotated_geometry_and_tcp():
    from experiments.graspgenx_sim.bbox_center_grasp import center_network_tcp_on_visible_obb
    rng=np.random.default_rng(4)
    corners=np.array([[x,y,z] for x in (-.05,.05) for y in (-.015,.015) for z in (-.02,.02)])
    cloud=np.vstack([corners,rng.uniform([-.05,-.015,-.02],[.05,.015,.02],(100,3))])
    rot=Rotation.from_euler('z',.6).as_matrix()
    cloud=cloud@rot.T + [.2,.1,.55]
    _,poses=fixture_data()
    result,info=center_network_tcp_on_visible_obb(cloud,poses)
    np.testing.assert_allclose(info['tcp_center_world_m'],[.2,.1,.55],atol=1e-10)
    np.testing.assert_allclose(sorted(info['obb']['extent_m'][:2]),[.03,.1],atol=1e-10)
    np.testing.assert_allclose((result@TRANSFORM)[:,:3,3],np.tile([.2,.1,.55],(4,1)),atol=1e-10)
    np.testing.assert_array_equal(result[:,:3,:3],poses[:,:3,:3])
    assert info['mode']=='visible_belt_aligned_obb_center_tcp'


def test_oriented_box_rejects_degenerate_xy():
    from experiments.graspgenx_sim.bbox_center_grasp import visible_belt_aligned_obb
    pts=np.column_stack([np.arange(30.),np.zeros(30),np.ones(30)])
    with pytest.raises(ValueError,match='cannot define'):
        visible_belt_aligned_obb(pts)


def test_obb_mode_reaches_existing_selector(monkeypatch):
    from experiments.graspgenx_sim import run
    from experiments.graspgenx_sim.bbox_center_grasp import visible_belt_aligned_obb
    points,poses=fixture_data();scores=np.array([.9,.8,.7,.6])
    class Backend:
        def infer(self,*args): return {'poses':poses.copy(),'scores':scores.copy(),'metadata':{}}
    def selector(snapshot,modified,passed_scores,position,cfg,observed_cloud_world):
        center=visible_belt_aligned_obb(points)['center_world_m']
        np.testing.assert_allclose((modified@TRANSFORM)[:,:3,3],np.tile(center,(4,1)))
        return [{'source_index':1,'variant':'original','score':scores[1]}],[]
    monkeypatch.setattr(run,'select_candidates',selector)
    result=run.infer_and_select(Backend(),points,7,4,4,SimpleNamespace(source_timestamp=1.),np.zeros(3),{},grasp_position_mode='visible_obb_center')
    assert result['accepted_candidates'][0]['grasp_position_mode']=='visible_belt_aligned_obb_center_tcp'

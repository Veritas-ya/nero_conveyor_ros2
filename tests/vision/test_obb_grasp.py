import numpy as np
from experiments.graspgenx_sim.obb_grasp import obb_top_down_grasp


def test_geometry_grasp_has_center_downward_approach_and_short_closing_axis():
    points=np.array([[x,y,z] for x in (-.05,.05) for y in (-.02,.02) for z in (.51,.55)]*4)
    result=obb_top_down_grasp(points)
    np.testing.assert_allclose(result['tcp'][:3,3],[0,0,.53],atol=1e-12)
    np.testing.assert_allclose(result['canonical'][:3,2],[0,0,-1],atol=1e-12)
    np.testing.assert_allclose(abs(result['canonical'][1,0]),1,atol=1e-12)
    assert np.linalg.det(result['canonical'][:3,:3]) > .999
    assert result['score'] is None and result['source']=='visible_obb_geometry_only'


def test_selector_uses_both_obb_axes_without_network_confidence(monkeypatch):
    from types import SimpleNamespace
    from experiments.graspgenx_sim.obb_grasp import select_obb_candidate
    from experiments.graspgenx_sim import candidates
    points=np.array([[x,y,z] for x in (-.05,.05) for y in (-.02,.02) for z in (.51,.55)]*4)
    def fake(snapshot, poses, scores, position, cfg, **kwargs):
        assert len(poses)==2 and np.array_equal(scores,[0.,0.])
        assert cfg['minimum_score']==0.
        assert kwargs['candidate_metadata']['candidate_sources']==['obb_top','obb_top']
        return [{'source_index':1,'original_network_pose_world':poses[1]}], []
    monkeypatch.setattr(candidates,'select_candidates',fake)
    result=select_obb_candidate(points,SimpleNamespace(source_timestamp=2.),np.zeros(3),{'minimum_score':.8})
    row=result['accepted_candidates'][0]
    assert row['obb_closing_axis']=='long' and row['source']=='visible_obb_geometry_only'
    assert 'original_network_pose_world' not in row
    assert row['score_semantics']=='ordering_placeholder_no_model_confidence'
    assert result['metadata']['network_called'] is False

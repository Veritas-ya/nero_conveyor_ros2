import numpy as np
import pytest
from experiments.graspgenx_sim.speed_comparison import compare_speed_at_capture


def test_velocity_comparison_transforms_and_pairs_exposure():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    row=compare_speed_at_capture(capture_s=1.,truth_s=1.,source_stamp_s=1.,truth_velocity_world=[.1,0,0],estimate_base=[0,-.09,0],base_to_world_rotation=R)
    assert row['visual_vx_world_m_s']==pytest.approx(.09)
    assert row['error_vx_m_s']==pytest.approx(-.01)
    assert row['truth_scope']=='evaluation_only_not_control'


def test_stale_or_misaligned_exposures_are_rejected():
    x=dict(capture_s=1.,truth_s=1.,source_stamp_s=.9,truth_velocity_world=[.1,0,0],estimate_base=[.1,0,0],base_to_world_rotation=np.eye(3))
    with pytest.raises(ValueError,match='different exposure'):compare_speed_at_capture(**x)
    x.update(source_stamp_s=1.,truth_s=.99)
    with pytest.raises(ValueError,match='not paired'):compare_speed_at_capture(**x)

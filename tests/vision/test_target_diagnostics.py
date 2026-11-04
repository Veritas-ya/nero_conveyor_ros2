import numpy as np
import pytest
from experiments.graspgenx_sim.target_diagnostics import point_world, request_target_record, result_target_record


def test_request_uses_actual_request_pose_and_policy_clock():
    transform=np.eye(4); transform[:3,3]=[1,2,3]
    row={'request_id':4,'request_source':'submitted','request_snapshot':{
        'target_poses':[[.2,.3,.4,1,0,0,0]], 'planned_target_time_s':5.}}
    out=request_target_record(row,transform,10.)
    np.testing.assert_allclose(out['target_tcp_position_world_m'],[1.2,2.3,3.4])
    assert out['contact_sim_time_s']==15.
    assert out['status']=='requested_target_not_a_certified_path'
    assert request_target_record({'request_snapshot':{}},transform,0.) is None
    with pytest.raises(ValueError):point_world([1,2],transform)


def test_failed_revised_target_is_not_mislabeled_as_contact():
    result=result_target_record({'request_id':3,'target_pose7':[1,2,3,1,0,0,0],
        'error':'full_pick_contact_outside_reachable_window'},np.eye(4))
    assert result['status']=='failed_proposal_not_contact'
    assert 'contact time' in result['meaning']

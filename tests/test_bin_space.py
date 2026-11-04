"""Footprint geometry only; physical drop success is checked in MuJoCo runs."""
import numpy as np
import pytest
from experiments.two_second_grasp.bin_space import free_center, free_centers


def test_five_varied_footprints_get_separate_points_in_same_bin():
    lower=np.array([-.105,-.73]);upper=np.array([.105,-.51]);occupied=[]
    for size in ([.055,.055],[.066,.080],[.039,.040],[.035,.035],[.052,.044]):
        size=np.asarray(size);c=free_center(lower,upper,size,occupied,[0,-.62]);a=c-size/2;b=c+size/2
        assert np.all(a>=lower-1e-12) and np.all(b<=upper+1e-12)
        for x,y in occupied:assert np.any(a>=y) or np.any(b<=x)
        occupied.append((a,b))
    assert len({tuple((a+b)/2) for a,b in occupied})==5


def test_no_space_or_oversize_is_rejected():
    with pytest.raises(ValueError):free_center([0,0],[1,1],[2,1],[],[.5,.5])
    with pytest.raises(ValueError):free_center([0,0],[1,1],[.2,.2],[([0,0],[1,1])],[.5,.5])


def test_center_is_preferred_in_an_empty_bin_and_inputs_are_unchanged():
    lower=np.array([-.1,-.1]);upper=-lower
    np.testing.assert_allclose(free_center(lower,upper,[.04,.04],[],[0,0]),[0,0])
    np.testing.assert_allclose(lower,[-.1,-.1]);np.testing.assert_allclose(upper,[.1,.1])


def test_alternative_centers_are_all_inside_and_avoid_the_existing_object():
    lower=np.array([-.1,-.1]);upper=-lower;size=np.array([.04,.04])
    occupied=[(np.array([-.02,-.02]),np.array([.02,.02]))]
    centers=free_centers(lower,upper,size,occupied,[0.,0.])
    assert len(centers)>1
    np.testing.assert_array_equal(centers[0],free_center(lower,upper,size,occupied,[0.,0.]))
    for center in centers:
        lo=center-size/2;hi=center+size/2
        assert np.all(lo>=lower-1e-12) and np.all(hi<=upper+1e-12)
        assert np.any(lo>=occupied[0][1]) or np.any(hi<=occupied[0][0])


def test_transfer_checks_another_free_pose_after_the_first_ik_failure(monkeypatch):
    from experiments.two_second_grasp import smooth_execution
    calls=[]
    def fake(env,kind,qpos,q,goal,runtime,dt,cancelled,**kw):
        calls.append(goal.tolist())
        return {'error':'loaded_route_not_found','goal_ik_count':0} if len(calls)==1 else {'path':'certified'}
    monkeypatch.setattr(smooth_execution,'plan_smooth_leg',fake)
    poses=[[0.,0.,1.,1.,0.,0.,0.],[.1,0.,1.,1.,0.,0.,0.]]
    result=smooth_execution.plan_transfer_candidates(None,None,None,poses,None,.01,lambda:False)
    assert calls==poses and result['bin_candidate_index']==1
    assert result['path']=='certified' and result['bin_candidate_attempts'][0]['error']=='loaded_route_not_found'


def test_each_tool_yaw_gets_its_own_footprint_check(monkeypatch):
    from experiments.two_second_grasp import bin_space
    calls=[]
    def select(env,runtime,retained,bin_id,goal):
        calls.append(np.asarray(goal).copy())
        return goal,{'candidate_tcp_poses_world':[list(goal)],
                     'candidate_payload_centers_xy_m':[[float(len(calls)),0.]]}
    monkeypatch.setattr(bin_space,'select_drop_pose',select)
    _,result=bin_space.select_drop_pose_candidates(None,None,[],1,[0,0,1,1,0,0,0])
    assert len(calls)==4
    assert result['candidate_yaw_offset_deg']==[0.,90.,-90.,180.]
    assert len(result['candidate_tcp_poses_world'])==4
    assert len(result['candidate_payload_centers_xy_m'])==4

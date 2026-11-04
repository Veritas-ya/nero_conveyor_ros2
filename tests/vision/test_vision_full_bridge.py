from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from nero_core.vision_full_bridge import prepare_full_observation, CandidateRejected
from experiments.continuous_intercept_sim.full_preplanned import matrix


def data():
    identity=TargetIdentity('session','object','task',2.,1,track_id=7,track_generation=3)
    provenance={'input_valid':True,'validation_reasons':[],'scene_epoch':9,'track_id':7,
       'track_generation':'3','capture_sim_s':2.,'track_position_world_m':[1.,2.,.5],
       'track_velocity_world_m_s':[.1,0.,0.], 'track_quaternion_world_wxyz':[1.,0,0,0]}
    obj=np.array([1.08,2.,.5,1.,0,0,0]); current=np.array([1.1,2.,.5,1.,0,0,0])
    pred={'observed_timestamp':2.8,'observed_pose':obj,'current_estimated_pose':current,
          'predicted_grasp_pose':np.array([1.3,2,.5,1,0,0,0]),'estimated_velocity':np.array([.1,0,0])}
    track={'object_id':7,'track_generation':3,'position':current[:3], 'quaternion':[1,0,0,0],
        'timestamp':3.,'measurement_timestamp':2.8,'track_observation_age':.2,'visible':True}
    raw={'timestamp':3.,'scene_epoch':9,'objects':[track], 'predictions':{7:pred}}
    cand={'tcp_pose_world_wxyz':[1.02,2.03,.55,0,1,0,0], 'closing_width_m':.04,'score':.8}
    return dict(provenance=provenance,candidate=cand,raw_observation=raw,result_identity=identity,
        active_identity=identity,max_observation_age_s=1.,max_result_age_s=4.)


def test_capture_binding_and_future_follow_preserve_visual_time_and_feedback():
    args=data(); raw=deepcopy(args['raw_observation']); out=prepare_full_observation(**args)
    p=out.observation['predictions'][7]
    np.testing.assert_allclose(p['current_ee_grasp_pose'][:3],[1.12,2.03,.55])
    np.testing.assert_allclose(p['observed_pose'][:3],[1.10,2.03,.55])
    np.testing.assert_allclose(p['predicted_ee_grasp_pose'][:3],[1.32,2.03,.55])
    assert p['observed_timestamp']==2.8 and out.capture_time_s==2.
    np.testing.assert_array_equal(p['object_observed_pose'],raw['predictions'][7]['observed_pose'])
    np.testing.assert_allclose(matrix(p['current_ee_grasp_pose'])@p['tcp_to_object'],matrix(raw['predictions'][7]['current_estimated_pose']))
    np.testing.assert_array_equal(args['raw_observation']['predictions'][7]['observed_pose'],raw['predictions'][7]['observed_pose'])

@pytest.mark.parametrize('change',['cancel','task','revision','track','generation','scene','age'])
def test_old_or_wrong_target_is_rejected(change):
    args=data()
    if change=='cancel':args['active_identity']=None
    if change=='task':args['active_identity']=replace(args['result_identity'],task_id='new')
    if change=='revision':args['active_identity']=replace(args['result_identity'],geometry_revision=1)
    if change=='track':args['result_identity']=replace(args['result_identity'],track_id=8);args['active_identity']=args['result_identity']
    if change=='generation':args['raw_observation']['objects'][0]['track_generation']=4
    if change=='scene':args['raw_observation']['scene_epoch']=10
    if change=='age':args['raw_observation']['objects'][0]['measurement_timestamp']=1.
    with pytest.raises(CandidateRejected):prepare_full_observation(**args)

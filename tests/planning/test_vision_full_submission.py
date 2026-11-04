"""Visual candidate -> original policy request; no hardware or physics success claim."""
from dataclasses import replace
import numpy as np
from test_policy_moving_approach import policy
from test_planning_worker_scene_validator import model
from test_full_pick_execution import setup
from nero_core.vision_full_bridge import prepare_full_observation
from mujoco_conveyor_sorting.runtime.types import TargetIdentity


def test_visual_tcp_and_observation_age_reach_original_full_request(policy, model, monkeypatch):
    _, make_observation = setup(policy,model,monkeypatch,approach_mode='continuous')
    policy._full_pick_request=None
    policy._planning_worker.requests.clear();policy._planning_worker.results.clear()
    policy._target_id=7
    target={'object_id':7,'confidence':1.,'grasp_window':{'end_time':100.}}
    policy._target=target
    monkeypatch.setattr(policy,'_find_target',lambda *_:target)
    data=make_observation(9.,age=.2)
    prior=data['predictions'].pop('part')
    # Reuse the policy fixture's current observed positions and motion while
    # binding a displaced network TCP captured 0.5 s earlier.
    stamp=prior['observed_timestamp'];velocity=data['belt_velocity']
    capture=8.3
    object_capture=prior['observed_pose'][:3]-velocity*(stamp-capture)
    identity=TargetIdentity('session','object','task',capture,0,track_id=7,track_generation=3)
    policy.env.target_identity=identity
    provenance={'input_valid':True,'validation_reasons':[],'track_id':7,'track_generation':'3',
      'scene_epoch':7,'capture_sim_s':capture,'track_position_world_m':object_capture.tolist(),
      'track_velocity_world_m_s':velocity.tolist(),'track_quaternion_world_wxyz':[1.,0.,0.,0.]}
    data['objects']=[{'object_id':7,'track_generation':3,'position':prior['current_ee_grasp_pose'][:3].copy(),
       'quaternion':[1.,0.,0.,0.],'timestamp':9.,'measurement_timestamp':stamp,
       'track_observation_age':.2,'visible':True}]
    prior['current_estimated_pose']=prior['current_ee_grasp_pose'].copy()
    prior['predicted_grasp_pose']=prior['current_estimated_pose'].copy()
    data['predictions']={7:prior}
    cand={'tcp_pose_world_wxyz':np.r_[object_capture+[.01,0.,.02],[0.,1.,0.,0.]],'closing_width_m':.04,'score':.8}
    bound=prepare_full_observation(provenance=provenance,candidate=cand,raw_observation=data,
         result_identity=identity,active_identity=identity,max_observation_age_s=1.,max_result_age_s=4.)
    selected=bound.observation['predictions'][7]
    assert policy._full_pick_act(bound.observation) is not None
    assert len(policy._planning_worker.requests)==1
    submitted=policy._planning_worker.requests[0]
    expected=selected['observed_pose'].copy()
    expected[:3]+=velocity*(submitted.planned_target_time_s-stamp)
    np.testing.assert_allclose(submitted.target_pose,expected,atol=1e-10)
    assert submitted.target_identity==identity
    assert submitted.full_observed_timestamp_s==stamp
    assert submitted.planned_target_time_s>9.

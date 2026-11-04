"""Replay one saved Nero7/GPU capture into the full-planner observation bridge.

Reconstructs current/predicted object poses from recorded visual telemetry;
never loads a device or uses object physics truth. Does not execute a plan.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from nero_core.vision_full_bridge import prepare_full_observation

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
network=json.loads((root/'inference/track_0000_try_01.json').read_text()); provenance=network['input_provenance']; ready=float(network['ready_sim_s']); track=int(network['track_id'])
with (root/'control_telemetry.jsonl').open() as stream:
    matches=[json.loads(line) for line in stream if line.strip()]
row=next(x for x in matches if abs(x['sim_s']-ready)<1e-8 and x['track_id']==track)
visual=row['visual']; stamp=visual['measurement_timestamp']; reports=json.loads((root/'nero7_reports.json').read_text()); rep=next(x for x in reports if abs(x['capture_sim_s']-stamp)<1e-8)
peer=next(x for x in rep['peer']['tracks'] if x['track_id']==track)
config=json.loads((root/'result.json').read_text());base=np.asarray(config['camera']['base_to_world']); measured=base[:3,:3]@np.asarray(peer['observed_position_base'])+base[:3,3]; velocity=np.asarray(visual['estimated_velocity']);current=np.asarray(visual['position']); future=current+velocity*max(0.,visual['estimated_grasp_time']-ready)
identity=TargetIdentity('offline-replay',str(track),'offline-task',float(provenance['capture_sim_s']),int(rep['peer']['clock_epoch']),track_id=track,track_generation=int(provenance['track_generation']))
pose=lambda xyz:np.r_[xyz,[1.,0.,0.,0.]]
prediction={'observed_timestamp':stamp,'observed_pose':pose(measured),'current_estimated_pose':pose(current),'predicted_grasp_pose':pose(future),'estimated_velocity':velocity}
live={'object_id':track,'track_generation':int(provenance['track_generation']),'position':current,'quaternion':[1.,0.,0.,0.],'timestamp':ready,'measurement_timestamp':stamp,'track_observation_age':visual['track_observation_age'],'visible':visual['visible']}
obs={'timestamp':ready,'scene_epoch':int(provenance['scene_epoch']),'objects':[live],'predictions':{track:prediction},'joint_pos':row['joint_pos'],'ee_pose':row['ee_pose']}
result=prepare_full_observation(provenance=provenance,candidate=network['accepted_candidates'][0],raw_observation=obs,result_identity=identity,active_identity=identity,max_observation_age_s=float(config['config']['tracking']['maximum_track_age_s']),max_result_age_s=config['config']['tracking'].get('maximum_result_age_s'))
p=result.observation['predictions'][track]; original=np.asarray(network['accepted_candidates'][0]['tcp_pose_world_wxyz']);capture_pos=np.asarray(provenance['track_position_world_m']);expected=original[:3]+current-capture_pos
np.testing.assert_allclose(p['current_ee_grasp_pose'][:3],expected,atol=1e-10)
print(json.dumps({'source_run':str(root),'network_input_exposure_s':result.capture_time_s,'visual_observation_s':p['observed_timestamp'],'decision_s':ready,'observation_age_s':ready-p['observed_timestamp'],'network_age_s':ready-result.capture_time_s,'measured_velocity_m_s':velocity.tolist(),'network_capture_tcp_pose':original.tolist(),'visual_observed_object_pose':p['object_observed_pose'].tolist(),'current_object_pose':p['object_current_estimated_pose'].tolist(),'current_tcp_pose':p['current_ee_grasp_pose'].tolist(),'predicted_tcp_pose':p['predicted_ee_grasp_pose'].tolist(),'actual_robot_feedback_q_rad':row['joint_pos'],'admission':result.admission,'closing_width_m':network['accepted_candidates'][0]['closing_width_m'],'note':'reconstructed future estimate uses saved velocity; no plan or device execution'},indent=2))

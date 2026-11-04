"""Replay saved capture and frozen model outputs with migrated algorithms, no device IO."""
import argparse, json, time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from replay_helpers import load_frame, green_belt_points, fit_plane
from nero_core.grasp_pipeline import prepare_and_select
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, PickSettings, PickRejected, detect_target
from mujoco_conveyor_sorting.runtime.grasp_depth import refine_grasp_observation
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, _geometry_fingerprint
from mujoco_conveyor_sorting.runtime.nero_moving_source import measured_belt_motion_region
from mujoco_conveyor_sorting.runtime.nero_graspgenx import _config

p=argparse.ArgumentParser();p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--historical-selection',action='store_true');args=p.parse_args()
r=args.recording; out=args.output; out.mkdir(parents=True,exist_ok=False)
plan=json.loads((r/'plan.json').read_text());raw=json.loads((r/'frame_000302.json').read_text());state=raw['robot_state']
_,frame,d2c=load_frame(r,'frame_000302')
pt=np.array(plan['scene']['scene_region_filter']['roi']['plane_to_base']); bounds=np.asarray(plan['scene']['scene_region_filter']['roi']['bounds_m'])[:2].copy()
t0=time.monotonic(); model=RobotModel();bounds[:,0]+=model.belt_scene_filter_config.xy_margin_m;bounds[:,1]-=model.belt_scene_filter_config.xy_margin_m
normal=pt[:3,2];plane={'normal_base':normal.tolist(),'abc':[-normal[0]/normal[2],-normal[1]/normal[2],pt[2,3]],'inlier_threshold_z_m':.004,'inlier_rms_m':.001,'tilt_from_base_xy_deg':float(np.degrees(np.arccos(normal[2])))}
green_world,green_mask,_=green_belt_points(frame,d2c);measured=green_world[green_mask];measured=measured[np.isfinite(measured).all(1)];flat=(measured-pt[:3,3])@pt[:3,:3];hull=cv2.convexHull(flat[np.abs(flat[:,2])<.005,:2].astype(np.float32))
startup=time.monotonic()-t0
model.measured_belt_surface=BeltSurface(pt.copy(),bounds.copy(),hull.copy(),dict(plane),1,_geometry_fingerprint(frame,d2c))
with np.load(r/'fast_stereo/call_0001.output.npz') as z: depth=z['depth_m'].copy()
meta=json.loads((r/'fast_stereo/call_0001.result.json').read_text());crop=tuple(meta['provenance']['crop_xyxy']);x0,y0,x1,y1=crop
class SavedFast:
 def refine(self,left,right,K,baseline,requested_crop,*,capture_metadata):
  assert tuple(requested_crop)==crop
  full=np.full(frame.depth_m.shape,np.nan,np.float32);full[y0:y1,x0:x1]=depth
  return SimpleNamespace(reconstructed_depth_m=full,crop_xyxy=crop,timings_s={},provenance=meta['provenance'],artifact_prefix='saved_fast')
with np.load(r/'graspgenx/call_0001.candidates.npz') as z:poses=z['poses'].copy();scores=z['scores'].copy()
network_meta=json.loads((r/'graspgenx/call_0001.candidates.json').read_text())
class SavedNetwork:
 config=_config(json.loads((r/'graspgenx/worker_configuration.json').read_text()))
 def infer(self,points,*args,**kw):
  return {'poses':poses.copy(),'scores':scores.copy(),'metadata':network_meta}
start=time.monotonic(); observation=detect_target(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,target_shape='simple',robot_model=model,robot_state=state)
observation.update(frame=frame,scene_state=state,depth_to_color_cv=d2c,target_identity=TargetIdentity.confirmed(raw.get('pairing',{}).get('depth_sample_monotonic_estimate'),frame.header.clock_epoch))
detect=time.monotonic()-start;t=time.monotonic();observation=refine_grasp_observation(observation,SavedFast(),robot_model=model,output_directory=None);refine=time.monotonic()-t
if not observation['grasp_depth_refinement']['applied']:raise RuntimeError('saved Fast output was not applied')
t=time.monotonic();region=measured_belt_motion_region(observation,frame,d2c,green_belt_points);region_time=time.monotonic()-t
settings=replace(PickSettings(),moving_approach_mode='continuous')
if args.historical_selection:
 from mujoco_conveyor_sorting.runtime.nero_graspgenx import PreparedGraspInference, select_measured_network_grasp
 backend=SavedNetwork(); t=time.monotonic(); prepared=PreparedGraspInference(observation,backend)
 try:
  scene_start=time.monotonic()
  scene=model.install_scene(observation['scene_points'],observation['object_points'],state,camera_origin=frame.camera_to_world_cv[:3,3],belt_region=observation.get('belt_region'),depth_frame=frame,target_mask=observation.get('mask'),scene_pixel_indices=observation['scene_pixel_indices'],gripper_nearby_scene_ignore=True)
  scene_end=time.monotonic()
  selected=select_measured_network_grasp(observation,model,state,backend,motion_region=region,prepared_inference=prepared)
  result=SimpleNamespace(timings={'scene_wall_s':scene_end-scene_start,'overlapped_delivery_wall_s':time.monotonic()-t,'inference':dict(prepared.timings)})
 finally:prepared.close()
else:
 try:
  result=prepare_and_select(observation,model,SavedNetwork(),motion_region=region,settings=settings,snapshot_provider=lambda _:None,scene_options={'gripper_nearby_scene_ignore':True})
 except PickRejected as e:
  (out/'rejection.json').write_text(json.dumps({'message':str(e),'details':e.details},ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
  raise
 selected=result.observation
delivery_wall=time.monotonic()-start
save_start=time.monotonic()
np.savez_compressed(out/'outputs.npz',target_mask=observation['grasp_target_mask'],target_points=observation['object_points'],native_target=observation['native_object_points'] if 'native_object_points' in observation else np.empty((0,3)),target_grasp=observation['grasp'],selected_grasp=selected['grasp'],scene=model.scene,voxel_indices=model.scene_voxel_result.representative_indices,voxel_coverage=model.scene_voxel_result.coverage_indices,voxel_error=model.scene_voxel_result.representative_max_errors_m)
report={'mode':'saved Fast and GraspGen-X outputs; no live inference, no velocity, no devices','startup_geometry_s':startup,'detect_s':detect,'refine_s':refine,'motion_region_s':region_time,'overlap':result.timings,'delivery_after_startup_s':delivery_wall,'numeric_output_save_s':time.monotonic()-save_start,'selection_is_provisional_without_measured_velocity':True,'historical_selection':args.historical_selection}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

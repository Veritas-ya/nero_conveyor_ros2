"""Offline comparison of full-scene and belt-only broad phases on saved RGB-D."""
import argparse,json,time,inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane,green_belt_points
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.belt_collision_scene import install_belt_surface
from nero_core.two_second.vision import obb_grasps
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.single_pick import detect_target


def original_detector():
 from mujoco_conveyor_sorting.runtime import single_pick
 source=inspect.getsource(single_pick.detect_target)
 new="""    if target_shape == 'simple' or robot_model is not None:
        # Evaluate the unchanged foreground predicates once per exposure.
        # Scalar axis comparisons avoid allocating HxWx2 temporary arrays.
        finite_flat = np.isfinite(flat).all(axis=2)
        foreground_height = ((frame.depth_m > 0) & finite_flat
                      & (flat[..., 2] >= pipeline.minimum_height)
                      & (flat[..., 2] <= pipeline.maximum_height))
        in_search_xy = ((flat[..., 0] >= lower[0]) & (flat[..., 0] <= upper[0])
                        & (flat[..., 1] >= lower[1]) & (flat[..., 1] <= upper[1]))
        foreground = foreground_height & in_search_xy"""
 old="""    if target_shape == 'simple' or robot_model is not None:
        foreground = ((frame.depth_m > 0) & np.isfinite(flat).all(axis=2)
                      & (flat[..., 2] >= pipeline.minimum_height)
                      & (flat[..., 2] <= pipeline.maximum_height)
                      & (flat[..., :2] >= lower).all(axis=2)
                      & (flat[..., :2] <= upper).all(axis=2))"""
 assert new in source
 source=source.replace(new,old,1)
 source=source.replace('            complete_foreground = foreground_height.copy()', """            complete_foreground = ((frame.depth_m > 0) & np.isfinite(flat).all(axis=2)
                & (flat[..., 2] >= pipeline.minimum_height)
                & (flat[..., 2] <= pipeline.maximum_height))""",1)
 source=source.replace('            c.mask & ~in_search_xy))', """            c.mask & (~(flat[..., :2] >= lower).all(axis=2)
                      | ~(flat[..., :2] <= upper).all(axis=2))))""",1)
 namespace=dict(vars(single_pick))
 exec(compile(source,'<previous-detection-reference>','exec'),namespace)
 return namespace['detect_target']


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--frames',nargs='+',required=True,type=Path);p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 previous_detection=original_detector()
 cfg=CellConfig.load(a.config);cal=load_runtime_depth_calibration(cfg.raw['camera_calibration_file']);model=calibrated_model(cfg);model.set_state(cfg.standby,cfg.settings.open_width_m);fl=model.flange();state={'joints':cfg.standby.tolist(),'width_m':cfg.settings.open_width_m,'pose':np.r_[fl[:3,3],Rotation.from_matrix(fl[:3,:3]).as_euler('xyz')].tolist()};results=[]
 for path in a.frames:
  frame,d2c=read(path,cal);surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
  for repeat in range(a.repeats):
   variants={}
   for mode in (['observed_scene','measured_belt_surface_only'] if repeat%2==0 else ['measured_belt_surface_only','observed_scene']):
    f,d=read(path,cal);begin=time.perf_counter();detector=previous_detection if mode=='observed_scene' else detect_target;ob=detector(f,d,fit_plane=fit_plane,green_belt_points=green_belt_points,target_shape='simple',robot_model=model,robot_state=state,measured_surface=surface,maximum_target_height_m=cfg.raw['maximum_target_height_m'],scene_scope=mode);dt=time.perf_counter()-begin
    grasps,_=obb_grasps(ob['object_points'],ob['belt_region']['plane_to_base'],cfg.canonical_tcp);ob['frame']=f;t=SimpleNamespace(detection=ob,sequence=f.header.sequence,epoch=f.header.clock_epoch,identity=(f.header.clock_epoch,1,1),capture_s=f.header.received_time)
    begin=time.perf_counter();scene_report=install_belt_surface(model,t);scene_time=time.perf_counter()-begin
    variants[mode]={'detection_s':dt,'scene_s':scene_time,'input_scene_count':len(ob['scene_points']),'mask':ob['mask'].copy(),'object':ob['object_points'].copy(),'grasps':np.asarray(grasps),'scene':model.scene.copy(),'pixels':model.scene_native_pixel_indices['belt_surface_retained'].copy(),'scene_report':scene_report}
   x,y=variants['observed_scene'],variants['measured_belt_surface_only']
   equal={k:bool(np.array_equal(x[k],y[k])) for k in ('mask','object','grasps','scene','pixels')}
   results.append({'frame':str(path),'repeat':repeat,'identical':equal,'baseline':{k:x[k] for k in ('detection_s','scene_s','input_scene_count')},'optimized':{k:y[k] for k in ('detection_s','scene_s','input_scene_count')}})
   print(path.name,repeat,equal,'detection',round(x['detection_s'],3),round(y['detection_s'],3),'scene',round(x['scene_s'],3),round(y['scene_s'],3),flush=True)
   if not all(equal.values()):raise RuntimeError('prefilter changed final result')
 stats={variant:{metric:float(np.median([r[variant][metric] for r in results])) for metric in ('detection_s','scene_s')} for variant in ('baseline','optimized')}
 report={'robot_connections':0,'commands_sent':0,'samples':len(results),'statistics_median_s':stats,'results':results,'limitation':'saved frames, assumed standby; no device or live competition; sequential wall times are not a full cycle'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(stats))
if __name__=='__main__':main()

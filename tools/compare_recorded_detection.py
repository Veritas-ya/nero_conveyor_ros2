"""Compare original/optimized depth projection on recorded frames, without devices."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from nero_core.two_second.collision_view import restore_frame
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.vision import obb_grasps
from mujoco_conveyor_sorting.runtime import visible_target
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration, load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.single_pick import detect_target


def read(path, calibration):
    with np.load(path,allow_pickle=False) as source:
        meta=json.loads(str(source['metadata']))
        frame=restore_frame(meta,{'frame_rgb':source['frame_rgb'],'frame_depth_m':source['frame_depth_m']})
    d2c=np.asarray(meta['depth_to_color_cv'])
    return apply_runtime_depth_calibration(frame,calibration,d2c),d2c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--frames',nargs='+',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=CellConfig.load(args.config)
    calibration=load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
    paths=sorted((args.run/'rgbd_frames').glob('*.npz'))
    if not paths:parser.error('no recorded frames')
    first,d2c=read(paths[0],calibration)
    surface=BeltSurface.from_frame(first,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
    model=calibrated_model(cfg);model.set_state(cfg.standby,cfg.settings.open_width_m)
    flange=model.flange();state={'joints':cfg.standby,'width_m':cfg.settings.open_width_m,
        'pose':np.r_[flange[:3,3],Rotation.from_matrix(flange[:3,:3]).as_euler('xyz')]}
    fast=visible_target._polynomial_distortion
    records=[]
    for seq in args.frames:
        path=next((p for p in paths if p.stem.endswith(f'_{seq:09d}')),None)
        if path is None:raise ValueError(f'frame {seq} not recorded')
        inputs=[]
        for variant in ('original','optimized'):
            frame,d2c=read(path,calibration)
            visible_target._polynomial_distortion=(lambda x:np.asarray(x,dtype=float).reshape(-1)) if variant=='original' else fast
            start=time.perf_counter()
            try:
                ob=detect_target(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
                    target_shape='simple',robot_model=model,robot_state=state,measured_surface=surface,
                    maximum_target_height_m=cfg.raw['maximum_target_height_m'])
                duration=time.perf_counter()-start
                grasps,_=obb_grasps(ob['object_points'],ob['belt_region']['plane_to_base'],cfg.canonical_tcp)
                inputs.append({'status':'detected','wall_s':duration,'mask':ob['mask'],
                    'points':ob['object_points'],'scene':ob['scene_points'],'grasps':np.asarray(grasps)})
            except (ValueError,RuntimeError) as error:
                inputs.append({'status':'rejected','wall_s':time.perf_counter()-start,
                    'error_type':type(error).__name__,'error':str(error)})
            finally:
                visible_target._polynomial_distortion=fast
        a,b=inputs
        row={'frame_sequence':seq,'frame_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'original_status':a['status'],'optimized_status':b['status'],
             'original_wall_s':a['wall_s'],'optimized_wall_s':b['wall_s']}
        if a['status']==b['status']=='detected':
            row.update({key+'_equal':bool(np.array_equal(a[key],b[key])) for key in ('mask','points','scene','grasps')})
            row.update(target_points=len(a['points']),scene_points=len(a['scene']))
        else:
            row.update(original_error=a.get('error'),optimized_error=b.get('error'),
                       rejection_equal=a.get('error')==b.get('error'))
        records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    report={'mode':'recorded_rgbd_offline_comparison','camera_connections':0,'can_connections':0,'commands_sent':0,
        'robot_feedback':'assumed configured standby; not measured for this replay','run':str(args.run.resolve()),
        'config_sha256':cfg.source_sha256,'frames':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

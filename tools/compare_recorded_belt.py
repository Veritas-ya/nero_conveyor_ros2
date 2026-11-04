"""Compare measured green-belt extraction on saved RGB-D, no devices."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from compare_recorded_detection import read
from nero_core.two_second import measured_rgbd as m
from nero_core.two_second.config import CellConfig
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface


def original_projection(points,frame,d2c):
    shape=points.shape[:-1]
    color=points.reshape(-1,3)@d2c[:3,:3].T+d2c[:3,3]
    uv=cv2.projectPoints(color,np.zeros(3),np.zeros(3),frame.color_intrinsics.matrix,
                         np.asarray(frame.color_intrinsics.distortion))[0].reshape(*shape,2)
    return uv,color[:,2].reshape(shape)>0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CellConfig.load(a.config)
    calibration=load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
    paths=sorted((a.run/'rgbd_frames').glob('*.npz'));fast=m.project_to_rgb;rows=[]
    for seq in a.frames:
        path=next(x for x in paths if x.stem.endswith(f'_{seq:09d}'))
        results=[]
        for variant in ('original','optimized'):
            frame,d2c=read(path,calibration);m.project_to_rgb=original_projection if variant=='original' else fast
            t=time.perf_counter()
            try:
                world,mask,green=m.green_belt_points(frame,d2c)
                wall=time.perf_counter()-t
                surface=BeltSurface.from_frame(frame,d2c,fit_plane=m.fit_plane,green_belt_points=m.green_belt_points)
                results.append((wall,world,mask,green,surface))
            finally:m.project_to_rgb=fast
        x,y=results
        row={'frame_sequence':seq,'original_wall_s':x[0],'optimized_wall_s':y[0],
             'world_equal':bool(np.array_equal(x[1],y[1])), 'green_depth_mask_equal':bool(np.array_equal(x[2],y[2])),
             'green_rgb_equal':bool(np.array_equal(x[3],y[3])),
             'plane_transform_equal':bool(np.array_equal(x[4].plane_to_base,y[4].plane_to_base)),
             'belt_bounds_equal':bool(np.array_equal(x[4].bounds_xy_m,y[4].bounds_xy_m)),
             'measured_green_points':int(np.count_nonzero(x[2]))}
        rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'mode':'recorded_belt_offline_comparison','commands_sent':0,'camera_connections':0,
                                   'can_connections':0,'frames':rows},indent=2)+'\n')


if __name__=='__main__':main()

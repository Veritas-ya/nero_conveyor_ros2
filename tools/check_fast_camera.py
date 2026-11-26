#!/usr/bin/env python3
"""Bounded camera-only Fast acceptance probe, then offline target processing.

No CAN, robot feedback or robot commands. Target mode tests visible geometry
and grasp generation, not self-filtering, moving-target deadlines or execution.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/nero_core'))
from nero_core.two_second.config import CellConfig
from nero_core.two_second.fast_perception import FastTargetPerception,camera_settings
from nero_core.two_second.collision_capture import frame_metadata
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.fixed_freeze import fixed_freeze_detection_surface
from nero_core.two_second.measured_rgbd import fit_plane,green_belt_points
from nero_core.two_second.vision import obb_grasps
from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration,apply_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.grasp_depth import _camera_geometry
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.single_pick import detect_target


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target',action='store_true',help='要求检测到工件并验证重建点云与抓姿')
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('输出目录已存在，请选择新目录')
    cfg=CellConfig.load(args.config);cfg.raw['fast_stereo_enabled']=True
    cal=load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
    args.output.mkdir(parents=True)
    report=dict(status='starting',camera_connections=0,can_connections=0,robot_commands_sent=0,
        physical_success=None,target_geometry_requested=args.target,
        limitation='Camera only; offline processing after capture. No robot self-filter feedback, motion deadline or execution validation.')
    camera=service=None
    try:
        service=FastTargetPerception(cfg,args.output/'backend').start()
        settings=camera_settings(cfg)
        settings.update(enable_color=True,require_color=True,timestamp_source='device',
            serial_number=cal['serial_number'],calibration_id=cal['calibration_id'],
            camera_to_world_cv=cal['depth_to_base_cv'])
        camera=OrbbecCameraBackend(settings);camera.start();report['camera_connections']=1
        sequence=None;report['frames']=[]
        for _ in range(12):
            frame=camera.read_latest(timeout_s=3.,after_sequence=sequence,wait_for_fresh=True)
            sequence=frame.header.sequence
            report['frames'].append(dict(sequence=sequence,stereo=frame.timestamp_metadata.get('stereo_ir')))
        d2c=camera.get_depth_to_color_cv()
        frame=apply_runtime_depth_calibration(frame,cal,d2c)
        meta=frame_metadata(frame,d2c)
        arrays=dict(frame_rgb=frame.rgb,frame_depth_m=frame.depth_m,
            metadata=np.asarray(json.dumps(json_value(meta))))
        for side in ('left_ir','right_ir'):
            value=getattr(frame,side,None)
            if value is not None:arrays['frame_'+side]=value
        np.savez(args.output/'capture.npz',**arrays)
        camera.close();camera=None
        K,baseline=_camera_geometry(frame)
        report.update(same_exposure_geometry_valid=True,frame_metadata=meta)
        if args.target:
            started=time.monotonic()
            surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
            surface=fixed_freeze_detection_surface(surface)
            report['belt_fit_s']=time.monotonic()-started
            started=time.monotonic()
            native=detect_target(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
                target_shape='simple',measured_surface=surface,
                maximum_target_height_m=cfg.raw['maximum_target_height_m'],
                maximum_target_length_m=cfg.raw.get('maximum_target_length_m',.16),
                belt_edge_margin_m=cfg.raw.get('perception_belt_edge_margin_m',.025),
                belt_near_camera_margin_m=cfg.raw.get('perception_belt_near_camera_margin_m',0.),
                belt_direction_base=cfg.direction,
                scene_scope=cfg.raw.get('environment_collision_scope','observed_scene'))
            report['native_detection_s']=time.monotonic()-started
            refined=service.refine(native,frame,d2c,surface=surface)
            report['fast_stereo']=refined['fast_stereo']
            started=time.monotonic()
            grasps,box=obb_grasps(refined['object_points'],surface.plane_to_base,cfg.canonical_tcp,
                tcp_z_offset_m=cfg.raw.get('grasp_tcp_z_offset_m',0.))
            report.update(obb_s=time.monotonic()-started,status='target_geometry_passed',
                native_points=len(native['object_points']),refined_points=len(refined['object_points']),
                grasp_count=len(grasps),obb=box,obb_coordinate_frame='measured_belt_plane',
                plane_to_base=surface.plane_to_base,grasps_coordinate_frame='robot_base')
            np.savez(args.output/'target.npz',native_object_points=native['object_points'],
                object_points=refined['object_points'],mask=native['mask'],
                grasp_target_mask=refined['grasp_target_mask'],grasps=np.asarray(grasps))
        else:
            h,w=frame.depth_m.shape;crop=(w//2-320,h//2-192,w//2+320,h//2+192)
            result=service.backend.refine(frame.left_ir,frame.right_ir,K,baseline,crop,
                capture_metadata={'capture_header':asdict(frame.header),'timestamp_metadata':frame.timestamp_metadata})
            valid=np.isfinite(result.depth_crop_m)&(result.depth_crop_m>0)
            if not valid.any():raise RuntimeError('实际红外推理无有效深度')
            report.update(status='real_stereo_inference_passed',valid_depth_pixels=int(valid.sum()),
                timings_s=result.timings_s,provenance=result.provenance)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',
            failure_details=getattr(error,'details',{}))
    finally:
        if camera:camera.close()
        if service:service.close()
        (args.output/'report.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report.get(k) for k in ('status','error','camera_connections','native_points',
            'refined_points','grasp_count')},ensure_ascii=False,indent=2))
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())

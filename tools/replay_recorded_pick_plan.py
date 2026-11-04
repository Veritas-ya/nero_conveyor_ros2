"""Replay recorded RGB-D through detection and production planner, no I/O devices."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.velocity_source import with_fixed_belt_speed, prediction_velocity, speed_report
from nero_core.two_second.vision import MeasuredTarget, obb_grasps
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.single_pick import detect_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--frame',type=int,required=True);p.add_argument('--speed',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-foreground',action='store_true',help='离线使用优化前前景及完整场景，用于对照')
    p.add_argument('--legacy-contact-scan',action='store_true',help='仅离线对照：使用优化前的完整Python接触遍历')
    p.add_argument('--trajectory-npz',type=Path,help='保存通过认证的轨迹采样，用于数值对照')
    p.add_argument('--recorded-standby-config',type=Path,help='记录帧时假定的机械臂姿态，用于自体过滤；规划起点仍使用 --config 的等待位，均非实际反馈')
    p.add_argument('--entry-observation-xyz',type=float,nargs=3,help='离线回放同一曝光的跟踪位置（m）')
    p.add_argument('--recorded-entry-position',action='store_true',
        help='使用同一曝光、同一运行记录的跟踪位置作为入区坐标；工件点云仍由回放检测')
    args=p.parse_args();cfg=with_fixed_belt_speed(CellConfig.load(args.config),args.speed)
    if args.legacy_contact_scan:
        from mujoco_conveyor_sorting.runtime import single_pick
        def legacy_first(data):
            contacts=tuple((int(c.geom1),int(c.geom2),float(c.dist)) for c in data.contact[:data.ncon])
            return next((c for c in contacts if c[2]<0),None)
        single_pick._first_native_penetrating_contact=legacy_first
    calibration=load_runtime_depth_calibration(cfg.raw['camera_calibration_file'])
    frames=sorted((args.run/'rgbd_frames').glob('*.npz'))
    path=next(x for x in frames if x.stem.endswith(f'_{args.frame:09d}'))
    first,d2c=read(frames[0],calibration)
    surface=BeltSurface.from_frame(first,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
    frame,d2c=read(path,calibration)
    timing=depth_capture_timing(frame);capture=timing['depth_sample_monotonic_estimate']
    model=calibrated_model(cfg);model.set_state(cfg.standby,cfg.settings.open_width_m)
    flange=model.flange();state={'joints':cfg.standby.tolist(),'width_m':cfg.settings.open_width_m,
        'pose':np.r_[flange[:3,3],Rotation.from_matrix(flange[:3,:3]).as_euler('xyz')].tolist()}
    detection_model, detection_state = model, state
    if args.recorded_standby_config:
        recorded=CellConfig.load(args.recorded_standby_config)
        detection_model=calibrated_model(recorded)
        detection_model.set_state(recorded.standby,recorded.settings.open_width_m)
        recorded_flange=detection_model.flange()
        detection_state={'joints':recorded.standby.tolist(),'width_m':recorded.settings.open_width_m,
            'pose':np.r_[recorded_flange[:3,3],Rotation.from_matrix(recorded_flange[:3,:3]).as_euler('xyz')].tolist()}
    report={'mode':'offline_recorded_rgbd_to_production_plan','source_frame':str(path.resolve()),
        'camera_connections':0,'can_connections':0,'commands_sent':0,'execution_performed':False,
        'robot_state':'configured_standby_and_model_flange_not_measured',
        'entry_position':'median_of_single_frame_object_cloud_not_historical_tracker',
        'clock':'actual recorded exposure plus measured receipt age and replay detection/planning wall time',
        'excluded':'tracking warmup, model/belt startup, live scheduling and SDK command setup',
        'contact_scan':'legacy_python_full_scan' if args.legacy_contact_scan else 'native_vectorized_distances',
        'configured_speed_m_s':args.speed,'capture_to_receive_s':frame.header.received_time-capture}
    report['detection_robot_state']=('assumed_recorded_standby_not_measured' if args.recorded_standby_config else report['robot_state'])
    t=time.monotonic()
    try:
        if args.baseline_foreground:
            from benchmark_belt_scene_prefilter import original_detector
            target_detector=original_detector()
        else:
            target_detector=detect_target
        ob=target_detector(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
            target_shape='simple',robot_model=detection_model,robot_state=detection_state,measured_surface=surface,
            maximum_target_height_m=cfg.raw['maximum_target_height_m'],
            scene_scope=('observed_scene' if args.baseline_foreground else
                         cfg.raw.get('environment_collision_scope', 'observed_scene')))
        grasps,box=obb_grasps(ob['object_points'],ob['belt_region']['plane_to_base'],cfg.canonical_tcp)
        detection_wall=time.monotonic()-t
        ob.update(frame=frame,depth_to_color_cv=d2c,scene_state=detection_state,obb=box)
        center=np.median(ob['object_points'],axis=0)
        entry_position=center
        if args.entry_observation_xyz is not None:
            entry_position=np.asarray(args.entry_observation_xyz,dtype=float)
            if not np.isfinite(entry_position).all():raise ValueError('回放跟踪位置无效')
            report['entry_position']='supplied_same_exposure_offline_tracker_position'
        if args.recorded_entry_position:
            prior=json.loads((args.run/'cycle_001'/'events.json').read_text())
            observed=next(row['data'] for row in prior if row['event']=='planning_submitted')
            if not np.isclose(observed['capture_monotonic_s'],capture,atol=1e-5,rtol=0):
                raise ValueError('记录的跟踪位置与回放曝光不一致')
            entry_position=np.asarray(observed['entry_observed_position_base_m'],dtype=float)
            if entry_position.shape!=(3,) or not np.isfinite(entry_position).all():
                raise ValueError('记录的跟踪位置无效')
            report['entry_position']='recorded_tracker_position_from_same_exposure'
        velocity=prediction_velocity(cfg,None,ob['plane']['normal_base'])
        target=MeasuredTarget((frame.header.clock_epoch,1,1),capture,frame.header.received_time,
            frame.header.source_time,frame.header.clock_epoch,frame.header.sequence,center,velocity,
            tuple(grasps),ob['object_points'],ob,state,speed_report(cfg,SimpleNamespace(sample_count=0),velocity),entry_position)
        submission=capture+report['capture_to_receive_s']+detection_wall
        wall_origin=time.monotonic();clock=lambda:submission+time.monotonic()-wall_origin
        planner=MeasuredPlanner(cfg,model=model,clock=clock)
        report.update(detection_and_obb_wall_s=detection_wall,age_at_submission_s=submission-capture,
            entry_coordinate_m=float(entry_position@cfg.direction),target_point_count=len(target.points))
        try:
            plan=planner.intercept(target,state)
            report.update(status='trajectory_delivered',planning_wall_s=plan.path.planning_s,
                remaining_send_lead_s=plan.reference.start_s-cfg.raw['command_advance_s']-clock(),
                timing_components_s=plan.timings_s,attempts=planner.attempt_diagnostics,
                predicted_contact_tcp_base_m=plan.reference.contact_xyz,selected_grasp_tcp_base=plan.selected_grasp)
            if args.trajectory_npz:
                times=np.linspace(0.,plan.path.duration,501)
                qva=np.asarray([plan.path.sample(t) for t in times])
                args.trajectory_npz.parent.mkdir(parents=True,exist_ok=True)
                np.savez(args.trajectory_npz,times=times,qva=qva)
        except (ValueError,RuntimeError) as error:
            report.update(status='planning_rejected',error=str(error),details=getattr(error,'details',{}),
                          planning_wall_s=clock()-submission,attempts=planner.attempt_diagnostics)
    except (ValueError,RuntimeError) as error:
        report.update(status='detection_rejected',error=str(error),details=getattr(error,'details',{}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(json_value(report),ensure_ascii=False)[:5000])


if __name__=='__main__':main()

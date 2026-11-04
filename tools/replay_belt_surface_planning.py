#!/usr/bin/env python3
"""Offline production-planner replay from a frozen real exposure; no devices.

Recomputes the measured belt plane from saved RGB-D because the original cached
plane was not archived. Uses the original frozen cloud, grasps, velocity and
entry measurement. Reconstructed kinematic feedback is not physical feedback.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.collision_view import restore_frame
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.vision import MeasuredTarget
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.belt_collision_scene import SCOPES
from nero_core.two_second.velocity_source import (with_fixed_belt_speed,is_fixed,
    prediction_velocity,speed_report,policy as velocity_policy)
from nero_core.two_second.diagnostics import json_value
from mujoco_conveyor_sorting.runtime.single_pick import frame_points, PickRejected
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface


def replay(run, config, *, candidate_index=None, start_from_standby=False):
    started=time.monotonic()
    source=run/'preview/target.npz'
    with np.load(source,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        frame=restore_frame(meta,dict(frame_rgb=data['frame_rgb'],frame_depth_m=data['frame_depth_m']))
        points=data['points'].copy(); grasps=tuple(data['grasps'].copy())
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    qualified=[e['data'] for e in events if e['event']=='qualified_target' and e['data']['identity']==meta['identity']]
    submitted=[e['data']['record']['data'] for e in events if e['event']=='execution_event'
        and e['data']['record']['event']=='planning_submitted'
        and e['data']['record']['data']['identity']==meta['identity']]
    if len(qualified)!=1 or len(submitted)!=1:
        raise ValueError('无法唯一配对历史目标与规划输入')
    original=submitted[0]; qualified=qualified[0]
    if (qualified['sequence']!=frame.header.sequence or meta['capture_s']!=original['capture_monotonic_s']
            or qualified['source_device_s']!=frame.header.source_time
            or frame.header.clock_epoch!=meta['identity'][0]):
        raise ValueError('历史曝光/目标身份不匹配')
    d2c=np.asarray(meta['depth_to_color_cv'])
    surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
    world=frame_points(frame)
    finite=np.isfinite(world).all(axis=2)&(frame.depth_m>0)
    native_ids=np.flatnonzero(finite)
    distance,nearest=cKDTree(world[finite]).query(points)
    if np.max(distance)>1e-9 or len(np.unique(nearest))!=len(points):
        raise ValueError('保存的目标点不能精确映射到本帧原生深度')
    scene_mask=finite.copy();scene_mask.flat[native_ids[nearest]]=False
    plane=surface.plane_to_base
    box=dict(meta['obb'])
    box['center_world_m']=plane[:3,:3].T@(np.asarray(box['center_world_m'])-plane[:3,3])
    box['axes_world_columns']=plane[:3,:3].T@np.asarray(box['axes_world_columns'])
    model=calibrated_model(config)
    initial=config.standby if start_from_standby else np.asarray(original['start_joints_rad'])
    model.set_state(initial,config.settings.open_width_m)
    start_tcp=model.tcp().copy()
    if not start_from_standby and not np.allclose(start_tcp,original['start_tcp_base'],rtol=0,atol=1e-8):
        raise ValueError('当前TCP模型不匹配历史规划起点')
    flange=model.flange()
    state={'joints':initial.tolist(),'width_m':config.settings.open_width_m,
        'pose':np.r_[flange[:3,3],Rotation.from_matrix(flange[:3,:3]).as_euler('xyz')].tolist()}
    velocity=(prediction_velocity(config,None,surface.normal_base) if is_fixed(config)
              else np.asarray(original['velocity_base_m_s']))
    velocity_report=(speed_report(config,SimpleNamespace(sample_count=original['speed_report'].get('sample_count')),velocity)
                     if is_fixed(config) else original['speed_report'])
    if candidate_index is not None and not 0 <= candidate_index < len(grasps):
        raise ValueError('候选索引超出保存的 OBB 候选范围')
    replay_grasps = grasps if candidate_index is None else (grasps[candidate_index],)
    target=MeasuredTarget(tuple(meta['identity']),meta['capture_s'],frame.header.received_time,
        frame.header.source_time,frame.header.clock_epoch,frame.header.sequence,
        np.asarray(original['target_center_base_m']),velocity,
        replay_grasps,points,{'scene_points':world[scene_mask],'scene_pixel_indices':np.flatnonzero(scene_mask),
            'belt_region':surface.region_metadata(),'frame':frame,'depth_to_color_cv':d2c,
            'camera_origin_base_m':frame.camera_to_world_cv[:3,3],'mask':finite&~scene_mask,'obb':box},
        state,velocity_report,np.asarray(original['entry_observed_position_base_m']))
    prepare_wall=time.monotonic()-started
    wall_origin=time.monotonic()
    # Reuse historical exposure/entry times while consuming real CPU wall time.
    clock=lambda: original['submission_monotonic_s']+time.monotonic()-wall_origin
    planner=MeasuredPlanner(config,model=model,clock=clock)
    report={'mode':'offline_production_planning_replay','source_run':str(run.resolve()),
        'source_frame_sha256':sha256(source.read_bytes()).hexdigest(),
        'source_events_sha256':sha256((run/'events.jsonl').read_bytes()).hexdigest(),
        'config_sha256':config.source_sha256,'environment_collision_scope':config.raw['environment_collision_scope'],
        'identity':meta['identity'],'frame_sequence':frame.header.sequence,
        'velocity_base_m_s':target.velocity.tolist(),'grasp_source':'original_saved_obb_candidates',
        'velocity_policy':velocity_policy(config),'counterfactual_speed_override':is_fixed(config),
        'original_logged_velocity_base_m_s':original['velocity_base_m_s'],
        'input_preparation_wall_s':prepare_wall,
        'recomputed_belt_plane':True,'original_filtered_scene_restored':False,
        'feedback_source':'configured_standby_counterfactual_model' if start_from_standby else 'saved_start_joints_with_model_reconstructed_flange',
        'start_tcp_base':start_tcp.tolist(),
        'candidate_filter_input_index_zero_based':candidate_index,
        'offline_velocity_limit_rad_s':config.raw.get('offline_velocity_limit_rad_s'),
        'hardware_connections':0,'commands_sent':0,'execution_performed':False,'physical_success':None}
    try:
        plan=planner.intercept(target,state)
        report.update(status='trajectory_delivered',planning_wall_s=clock()-original['submission_monotonic_s'],
            selected_approach_s=plan.path.diagnostics.get('approach_s',plan.reference.duration_s),
            prealign_s=getattr(plan.reference,'prealign_s',0.),trajectory_duration_s=plan.path.duration,
            remaining_send_lead_s=plan.reference.start_s-config.raw['command_advance_s']-clock(),
            timings_s=plan.timings_s,trajectory_diagnostics=plan.path.diagnostics,
            candidate_failures=plan.failures)
        selected=next(i for i,g in enumerate(grasps) if np.array_equal(g,plan.selected_grasp))
        start_rotation=Rotation.from_matrix(start_tcp[:3,:3])
        target_rotation=Rotation.from_matrix(plan.selected_grasp[:3,:3])
        orientation_samples=[]
        sample_times=np.unique(np.r_[np.linspace(0.,plan.reference.duration_s,5),
                                    getattr(plan.reference,'rotate_first_s',0.),getattr(plan.reference,'prealign_s',0.)])
        for elapsed in sample_times:
            ref_xyz,ref_quat=plan.reference.sample(elapsed)
            ref_rotation=Rotation.from_quat(ref_quat[[1,2,3,0]])
            model.set_state(plan.path.position(elapsed),config.settings.open_width_m)
            fk=model.tcp();fk_rotation=Rotation.from_matrix(fk[:3,:3])
            orientation_samples.append({'trajectory_elapsed_s':float(elapsed),
                'reference_rpy_xyz_deg':ref_rotation.as_euler('xyz',degrees=True).tolist(),
                'fk_rpy_xyz_deg':fk_rotation.as_euler('xyz',degrees=True).tolist(),
                'reference_change_from_start_deg':float(np.rad2deg((start_rotation.inv()*ref_rotation).magnitude())),
                'fk_error_from_reference_deg':float(np.rad2deg((ref_rotation.inv()*fk_rotation).magnitude())),
                'fk_translation_from_start_m':float(np.linalg.norm(fk[:3,3]-start_tcp[:3,3])),
                'fk_position_error_m':float(np.linalg.norm(fk[:3,3]-ref_xyz))})
        report['orientation_audit']={'selected_input_grasp_number':selected+1,
            'selected_target_rpy_xyz_deg':target_rotation.as_euler('xyz',degrees=True).tolist(),
            'target_change_from_start_deg':float(np.rad2deg((start_rotation.inv()*target_rotation).magnitude())),
            'samples':orientation_samples,'source':'offline_planned_reference_and_private_model_fk',
            'actual_robot_feedback':False}
        if plan.path.diagnostics.get('intercept_strategy') == 'synchronized':
            join = plan.reference.prealign_s
            contact = plan.reference.duration_s
            audit_times = np.linspace(join, contact, 101)
            sync_vel = np.asarray([plan.reference.position_derivatives(t)[1] for t in audit_times])
            join_vel = np.asarray([plan.reference.position_derivatives(t)[1]
                                   for t in np.linspace(0., join, 101)])
            report['synchronized_reference_audit'] = {
                'duration_s': contact-join,
                'maximum_velocity_error_m_s': float(np.max(np.linalg.norm(sync_vel-target.velocity,axis=1))),
                'minimum_join_velocity_along_belt_m_s': float(np.min(join_vel@config.direction)),
                'reference_only': True,
                'not_evidence_of_actual_tcp_tracking': True,
            }
    except PickRejected as error:
        report.update(status='planning_rejected',error=str(error),details=error.details,
            planning_wall_s=clock()-original['submission_monotonic_s'])
    report['planner']=planner.diagnostics()
    if planner.scene_report is not None:
        summary=json.loads((run/'summary.json').read_text())
        logged=[f['details']['points_base_m'][0] for f in summary['details'].get('failures',[])
                if (f.get('details') or {}).get('kind')=='scene_collision']
        if logged:
            distances,_=planner.model.tree.query(logged)
            report['old_trigger_to_current_environment_distance_m']=distances.tolist()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--scope',choices=SCOPES);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-belt-speed-m-s',type=float)
    p.add_argument('--start-from-standby',action='store_true',help='离线反事实：从当前配置等待位重新规划，非历史机械臂反馈')
    p.add_argument('--offline-velocity-limit-rad-s',type=float,
        help='离线反事实对照：仅覆盖规划器关节速度限制，不改变或认证现场配置')
    p.add_argument('--candidate-index',type=int,help='仅用于离线诊断：只重放该原始候选（从 0 开始），非真实筛选结果')
    args=p.parse_args();config=CellConfig.load(args.config)
    if args.offline_velocity_limit_rad_s is not None:
        value=args.offline_velocity_limit_rad_s
        if not np.isfinite(value) or value <= 0:
            p.error('离线速度上限须为正有限数')
        config=replace(config,velocity_limits=np.full(7,value),
                       raw={**config.raw,'offline_velocity_limit_rad_s':value})
    if args.scope:config=replace(config,raw={**config.raw,'environment_collision_scope':args.scope})
    if args.fixed_belt_speed_m_s is not None:config=with_fixed_belt_speed(config,args.fixed_belt_speed_m_s)
    args.output.mkdir(parents=True,exist_ok=False)
    report=replay(args.run,config,candidate_index=args.candidate_index,start_from_standby=args.start_from_standby)
    (args.output/'replay.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2,allow_nan=False))
    print(f'[离线规划] {report["status"]}，{report["planning_wall_s"]:.3f} s；设备/命令 0')
    return 0 if report['status']=='trajectory_delivered' else 1


if __name__=='__main__':raise SystemExit(main())

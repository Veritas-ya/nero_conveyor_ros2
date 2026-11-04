"""Offline: concurrent real GraspGen-X inference + captured visual-scene filtering.

Physics advances without arm commands; this is not a grasp or path certificate.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
import yaml
from experiments.graspgenx_sim.run import config_overrides, infer_and_select, warmup_model
from experiments.graspgenx_sim.inference_worker import GraspGenXBackend
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation, Nero7SensorControlEnvironment
from experiments.graspgenx_sim.sensor_control import SensorControlConfig
from experiments.graspgenx_sim.camera import cloud_for_track
from experiments.graspgenx_sim.measured_frame import MeasuredFrameBuilder
from experiments.graspgenx_sim.bind_measured_network import bind_measured_network
from experiments.graspgenx_sim.input_integrity import build_input_provenance
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, PickSettings
from mujoco_conveyor_sorting.runtime.nero_policy_context import _copy_geometry, prepare_planning_roadmap
from scipy.spatial.transform import Rotation
from nero_core.measured_motion import MeasuredMotionSession
from experiments.continuous_intercept_sim.command_advance import install_simulation_advance
from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud
from experiments.graspgenx_sim.trajectory_overlay import historical_samples, tcp_path_from_samples, draw_tcp_path
from experiments.graspgenx_sim.target_diagnostics import point_world, request_target_record, result_target_record
from experiments.graspgenx_sim.retry_candidates import (rebuild_after_rejection,
    RETRYABLE_GEOMETRY_REJECTIONS, evaluated_variants, conveyor_world_x_axis_in_base,
    remaining_window_check)
from experiments.graspgenx_sim.candidates import select_candidates
from experiments.graspgenx_sim.bbox_center_grasp import annotate_center_candidates
from experiments.graspgenx_sim.obb_grasp import select_obb_candidate
from experiments.graspgenx_sim.obb_sim_dispatch import OBBSimulationDispatch, SimulationDispatchRejected
from experiments.graspgenx_sim.obb_sim_environment import OBBCertifiedSimulation
from experiments.graspgenx_sim.simulation_envelope import align_simulation_command_envelope
from experiments.continuous_intercept_sim.metrics import contact_summary


def select_retry_batch(snapshot, inference, provenance, cfg, cloud, examined):
    start=time.perf_counter()
    accepted,rejected=select_candidates(snapshot,inference['poses'],inference['scores'],
        np.asarray(provenance['track_position_world_m']),cfg,observed_cloud_world=cloud,
        skip_variants=examined)
    if 'obb_info' in inference:
        for row in accepted:
            row['original_obb_pose_world']=row.pop('original_network_pose_world', inference['poses'][row['source_index']].copy())
            row.update(source='visible_obb_geometry_only', obb_closing_axis=('short','long')[row['source_index']], grasp_position_mode='visible_obb_geometry_only', score_semantics='ordering_placeholder_no_model_confidence', visible_box_center_world_m=inference['obb_info']['center_world_m'])
    if 'center_grasp_info' in inference:
        annotate_center_candidates(accepted,inference['original_network_poses'],inference['center_grasp_info'])
    return accepted,rejected,time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--belt-speed',type=float,default=.1,help='仿真带速 m/s，实际速度仍由视觉估计')
    parser.add_argument('--execute-sim',action='store_true',help='仅在 MuJoCo 发送通过实时检查的 full_preplanned 样本')
    parser.add_argument('--freeze-visual',action='store_true',help='仅仿真：规划提交时冻结初始观测和速度，不再更新目标预测')
    parser.add_argument('--early-vision',action='store_true',help='仿真扩大相机及视觉检测范围，在测速合格前并行准备场景和抓姿')
    parser.add_argument('--warm-roadmap',action='store_true',help='仿真开始前预热静态路线图，仅供 full_preplanned 提议并重新认证')
    parser.add_argument('--standby-at-reset',action='store_true',help='仅仿真：将初始姿态设为抓取区中心上方等待位；不表示已执行到达路径')
    parser.add_argument('--sim-condition-limit',type=float,help='仅仿真对照：雅可比条件数上限，不能用于实机')
    parser.add_argument('--sim-tcp-linear-limit',type=float,help='仅仿真对照：TCP 线速度上限 m/s')
    parser.add_argument('--sim-tcp-angular-limit',type=float,help='仅仿真对照：TCP 角速度上限 rad/s')
    parser.add_argument('--sim-disable-tcp-and-singularity-limits',action='store_true',help='仅仿真诊断：关闭 TCP 速度及雅可比条件数上限')
    parser.add_argument('--disable-scene-review',action='store_true',help='仅仿真对照：关闭交付后的在线场景复核')
    parser.add_argument('--grasp-source',choices=('network','obb'),default='network',help='obb：只从当前帧 OBB 生成抓姿，不加载网络')
    parser.add_argument('--grasp-position-mode',choices=('network','visible_bbox_center','visible_obb_center'),default='network',
        help='visible_obb_center 使用单帧可见点的带面约束 OBB 中心，网络评分不重新计算')
    parser.add_argument('--min-approach-s',type=float,help='仿真候选最短接近时间，修改后完整重新规划认证')
    parser.add_argument('--compute-budget-s',type=float,
        help='完整规划预留的策略时钟秒数；默认沿用原配置，须依据同一时钟域的实测设置')
    parser.add_argument('--candidate-wall-budget-s',type=float,
        help='单个未执行候选的可取消墙钟预算；默认读取配置，与策略时钟排期预留分开')
    parser.add_argument('--viewer',action='store_true',help='在 MuJoCo 窗口显示实时视觉点云和选中抓姿')
    parser.add_argument('--replay-certified-overlay',type=Path,help='仅显示指定历史结果中的认证轨迹，不用于控制')
    parser.add_argument('--hold-viewer',action='store_true',help='结束后保持最终画面，关闭窗口后退出')
    parser.add_argument('--capture-output',type=Path,help='可选：保存最后一帧实时点云和显示姿态')
    args=parser.parse_args()
    sim_overrides=(args.sim_condition_limit,args.sim_tcp_linear_limit,args.sim_tcp_angular_limit)
    if any(value is not None for value in sim_overrides) and not args.execute_sim:
        parser.error('仿真规划参数对照需要 --execute-sim')
    if args.sim_disable_tcp_and_singularity_limits and not args.execute_sim:
        parser.error('关闭规划上限只允许用于仿真执行诊断')
    if args.sim_disable_tcp_and_singularity_limits and any(value is not None for value in sim_overrides):
        parser.error('关闭规划上限不能与仿真上限覆盖参数同时使用')
    if any(value is not None and (not np.isfinite(value) or value <= 0.) for value in sim_overrides):
        parser.error('仿真规划限制必须为有限正数')
    if not np.isfinite(args.belt_speed) or not 0 < args.belt_speed <= 1: parser.error('--belt-speed 必须在 (0,1] m/s')
    if args.freeze_visual and not args.execute_sim: parser.error('--freeze-visual 仅支持仿真执行')
    if args.freeze_visual: args.disable_scene_review=True
    if args.disable_scene_review and not args.execute_sim: parser.error('--disable-scene-review 仅允许用于仿真执行对照')
    if args.execute_sim and args.grasp_source != 'obb': parser.error('--execute-sim 仅支持 OBB 仿真诊断')
    if args.viewer and not os.environ.get('DISPLAY'):
        parser.error('--viewer 需要图形桌面 DISPLAY；请从桌面终端运行')
    if args.hold_viewer and not args.viewer:
        parser.error('--hold-viewer 需要同时指定 --viewer')
    if args.replay_certified_overlay and not args.viewer:
        parser.error('--replay-certified-overlay 需要 --viewer')
    if args.min_approach_s is not None and (not np.isfinite(args.min_approach_s) or args.min_approach_s<=0): parser.error('--min-approach-s 必须是正的有限秒数')
    if args.compute_budget_s is not None and (not np.isfinite(args.compute_budget_s) or args.compute_budget_s<=0):
        parser.error('--compute-budget-s 必须是正的有限秒数')
    if args.candidate_wall_budget_s is not None and (not np.isfinite(args.candidate_wall_budget_s) or args.candidate_wall_budget_s<=0):
        parser.error('--candidate-wall-budget-s 必须是正的有限秒数')
    cfgpath=Path(__file__).resolve().parents[1]/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());cfg['experiment'].update(seed=1543,belt_speed_m_s=args.belt_speed)
    candidate_wall_budget=float(args.candidate_wall_budget_s if args.candidate_wall_budget_s is not None
                                else cfg['control']['candidate_planning_wall_budget_s'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path=args.output.with_suffix('.planning.jsonl')
    trace_file=trace_path.open('w',buffering=1)
    planning_runs=[]; current_run=None; progress_cursor=0; last_progress_wall=0.;last_progress_stage=None
    def trace_progress(worker, *, force=False):
        nonlocal progress_cursor,last_progress_wall,last_progress_stage
        if current_run is None or (not force and time.monotonic()-last_progress_wall<.25):return
        progress=worker.progress_snapshot(progress_cursor)
        progress_cursor=progress['sequence'];last_progress_wall=time.monotonic()
        current_run['progress']={key:value for key,value in progress.items() if key!='events'}
        trace_file.write(json.dumps({'attempt':current_run['attempt'], 'candidate_index':current_run['candidate_index'],
            'source_index':current_run['source_index'], 'sim_s':float(env.data.time), 'progress':progress},
            ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
        stage=tuple(s['name'] for s in progress['active_stages'])
        if stage != last_progress_stage or force:
            print(f"候选 {current_run['candidate_index']+1}：{progress['elapsed_wall_s']:.3f} 墙钟秒，"
                  f"{progress['status']}，阶段 {' > '.join(stage) or 'none'}",flush=True)
            last_progress_stage=stage
    result={'status':'starting','hardware_connections':0,'object_truth_used_for_input':False,
            'physics_commands_sent':0,'complete_path_certified':False,'physics_grasp_verified':False,
            'grasp_position_mode':args.grasp_position_mode, 'grasp_source':args.grasp_source, 'network_called':args.grasp_source == 'network', 'online_scene_review_enabled':not args.disable_scene_review, 'simulation_belt_speed_m_s':args.belt_speed, 'freeze_visual_requested':args.freeze_visual}
    result.update(candidate_planning_wall_budget_s=candidate_wall_budget,planning_progress_jsonl=str(trace_path),
                  planning_runs=planning_runs)
    env=ctx=viewer=None; pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='offline-vision-prepare')
    display_points=None; display_pose=None; display_stamp=None; last_view_wall=0.
    display_path=None; display_path_label=None; overlay_robot=None; path_drawn=False
    requested_contact_marker=None; current_tcp_marker=None
    pending_action=None; dispatch=None; dispatched=0; last_stage=None; first_command_q=None; max_joint_motion=0.; physical_rows=[]; visual_frozen=False; post_freeze_frame_count=0; freeze_marker=None; initial_marker=None
    wall_begin=time.perf_counter()
    try:
        backend=None
        if args.grasp_source == 'network':
            backend=GraspGenXBackend(seed=1543,max_points=cfg['graspgenx']['maximum_input_points'])
            backend.torch.set_num_threads(2)
            result['model_load_s']=backend.provenance['load_seconds']
            warmup=warmup_model(backend,1543,cfg['graspgenx']['generated_candidates'],cfg['graspgenx']['returned_candidates'])
            result['warmup_s']=warmup['wall_s']
        overrides=config_overrides(cfg)
        if args.standby_at_reset:
            overrides['robot']['robot']['initialize_at_wait_pose']=True
            overrides['scene'].setdefault('workspace',{}).update(
                waiting_ee_position=cfg['control']['simulation_standby_tcp_world_m'],
                waiting_ee_quaternion_wxyz=cfg['control']['simulation_standby_quaternion_wxyz'])
        env=OBBCertifiedSimulation(capture_path=cfgpath.parent/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1],
            config_overrides=overrides,grasp_assist=False,perception_mode='rendered_rgbd',
            grasp_event_source='physical_contacts',seed=1543,action_type='joint_position',preserve_control_profile=True,
            simulation_camera_world_xyz=cfg['camera']['early_view_world_xyz_m'] if args.early_vision else None,
            visual_workspace_x_world=cfg['tracking']['early_visual_workspace_x_world_m'] if args.early_vision else None)
        sensor=Nero7SensorControlEnvironment(env,SensorControlConfig(maximum_track_age_s=1.))
        geometry=RobotModel()
        if args.warm_roadmap:
            warm_start=time.perf_counter()
            warm_settings=PickSettings(planning_mode='full_preplanned')
            roadmap=prepare_planning_roadmap(geometry,warm_settings)
            result['roadmap_warmup_wall_s']=time.perf_counter()-warm_start
            result['roadmap_preparation']=geometry._warm_roadmap_preparation
            result['roadmap_is_route_hint_only']=True
        env.reset(seed=1543,options={'belt_speed':args.belt_speed,'initial_objects':1})
        if args.standby_at_reset:
            target=np.asarray(cfg['control']['simulation_standby_tcp_world_m'],dtype=float)
            observed=np.asarray(env._ee_pose(),dtype=float)
            error=float(np.linalg.norm(observed[:3]-target))
            desired=np.asarray(cfg['control']['simulation_standby_quaternion_wxyz'],dtype=float)
            orientation_error=float(2*np.arccos(np.clip(abs(np.dot(observed[3:],desired)),0.,1.)))
            geometry.check_state(np.asarray(env.get_joint_positions()),.10,check_scene=False)
            static_scene_safe=env.is_arm_configuration_safe(np.asarray(env.get_joint_positions()),
                minimum_tcp_height=float(env.belt_height),
                minimum_joint_margin=float(env.robot_controller.config.joint_command_margin))
            result['standby_initialization']={'mode':'mujoco_reset_ik_not_a_motion',
                'requested_tcp_world_m':target.tolist(),'actual_tcp_world_m':observed[:3].tolist(),
                'position_error_m':error,'orientation_error_rad':orientation_error,
                'robot_model_self_collision_check_passed':True,
                'simulation_static_scene_state_check_passed':bool(static_scene_safe),
                'joint_positions_rad':np.asarray(env.get_joint_positions()).tolist(),
                'connection_from_previous_pose_certified':False}
            if error > .002 or orientation_error > .012:
                raise ValueError('simulation standby IK failed requested pose tolerances')
            if not static_scene_safe:
                raise ValueError('simulation standby fails static scene or command-margin state check')
        result.update(early_vision=args.early_vision, simulation_camera_world_xyz_m=env.simulation_camera_world_xyz.tolist(),
            visual_workspace_x_world_m=None if env.visual_workspace_x_world is None else env.visual_workspace_x_world.tolist(),
            grasp_region_world_m=list(env.grasp_region))
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(env.model,env.data)
            if args.replay_certified_overlay:
                overlay_robot=RobotModel()
                display_path=tcp_path_from_samples(historical_samples(args.replay_certified_overlay),overlay_robot,env.nero_base_to_world)
                display_path_label='HISTORICAL certified path (display only)'
                result['trajectory_overlay']={'source':'historical_certified_record_display_only',
                    'path':str(args.replay_certified_overlay),'points':len(display_path)}
            print('MuJoCo 实时视觉：青色=目标深度点云；橙色=选中抓姿（方向示意）；'+('已启用仿真执行尝试。' if args.execute_sim else '无运动命令。'),flush=True)
        builder=MeasuredFrameBuilder(); session=identity=captured=provenance=None;task_id=None;network=scene=None;last=-1
        recent=deque(maxlen=8); candidate_index=0; attempt_records=[]
        delivery_retries=0; compute_budget=args.compute_budget_s
        selection_future=None; examined=set(); selection_batches=1; scene_prototype=None; batch_times=[]
        latest_frame=[None]; origin=None
        region=env.nero_surface.region_metadata();region['source']='declared_simulation_belt_geometry'
        sim_start_wall=time.perf_counter()
        while env.data.time<args.seconds:
            env.step(pending_action)
            if dispatch is not None:
                dispatched=dispatch.count;result['physics_commands_sent']=dispatched
                if getattr(env,'obb_dispatch_fault',None):
                    result.update(status='simulation_dispatch_rejected',dispatch_rejection=env.obb_dispatch_fault,
                        dispatch_guard_diagnostics=dispatch.last_guard_diagnostics);break
            if pending_action is not None and dispatched:
                if first_command_q is None: first_command_q=np.asarray(env.get_joint_positions()).copy()
                max_joint_motion=max(max_joint_motion,float(np.max(np.abs(np.asarray(env.get_joint_positions())-first_command_q))))
                result['max_joint_change_after_first_command_rad']=max_joint_motion
            pending_action=None; observation=sensor.get_observation()
            if args.execute_sim:
                runtime=next((r for r in env._objects if r.logical_id is not None),None)
                if runtime is not None:
                    forces=env._finger_object_contact_forces(runtime)
                    belt_contact=any(env.belt_geom_id in (c.geom1,c.geom2) and bool({int(c.geom1),int(c.geom2)} & set(runtime.geom_ids)) for c in env.data.contact[:env.data.ncon])
                    physical_rows.append({'sim_s':float(env.data.time),'object_pose':env.data.qpos[runtime.qpos_adr:runtime.qpos_adr+7].copy(),
                        'object_active':True,'left_normal_force_n':float(forces[0]),'right_normal_force_n':float(forces[1]),
                        'object_belt_contact':belt_contact,'gripper_command':float(env._gripper_observation()['command'])})
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence']
                items=meta['measurements']
                if identity is None:
                    qualified=[(k,v) for k,v in items.items() if v['velocity_measurement'].get('speed_valid')]
                    eligible=list(items.items()) if args.early_vision else qualified
                    if not eligible:continue
                    task_id,entry=eligible[0]
                    identity=TargetIdentity('offline-render','part','task',float(frame['timestamp']),int(meta['clock_epoch']),
                        track_id=entry['actual_peer_track_id'],track_generation=entry['track_generation'])
                    session=MeasuredMotionSession(identity,**({'max_coast_s':None,'max_prediction_horizon_s':None} if args.freeze_visual else {'max_coast_s':1.}))
                if task_id not in items:continue
                if visual_frozen:
                    # Display-only: never publish these observations to the tracker or scene monitor.
                    post_freeze_frame_count+=1
                    display_points=cloud_for_track(frame,task_id); display_stamp=float(frame['timestamp'])
                    continue
                try:
                    item=builder.build(frame,task_id=task_id,identity=identity,world_to_base=env.nero_world_to_base,
                                       belt_region=region,include_scene=network is None,
                                       allow_unqualified_speed=args.early_vision)
                    if not session.update(item.detection,identity=identity,timestamp=item.capture_s,sequence=item.sequence,
                        clock_epoch=item.clock_epoch,source_timestamp=item.capture_s,velocity_measurement=item.velocity_measurement):continue
                    recent.append(item)
                    display_points=cloud_for_track(frame,task_id)
                    display_stamp=item.capture_s
                    current=session.snapshot(float(env.data.time))
                    if item.velocity_measurement is not None and item.velocity_measurement.get('speed_valid'):
                        result.setdefault('first_qualified_speed_capture_s',item.capture_s)
                        result.setdefault('first_qualified_speed_observed_wall_s',time.perf_counter()-sim_start_wall)
                        result.setdefault('first_qualified_speed_position_world_m',
                            (env.nero_base_to_world[:3,:3] @ np.asarray(item.detection['grasp'])[:3,3] + env.nero_base_to_world[:3,3]).tolist())
                    display_pose=(env.nero_base_to_world @ current.observed_grasp
                                  if current is not None and current.valid and session._selected is not None else None)
                    latest_frame[0]=item.detection['frame']
                    if network is None:
                        _, observed=sensor._observed_state_at(task_id,item.capture_s)
                        if observed is None:continue
                        cloud=display_points
                        provenance=build_input_provenance(frame,task_id,cloud,observed,int(env._scene_epoch),float(env.data.time))
                        if not provenance['input_valid']:
                            result['last_rejection']=provenance['validation_reasons'];continue
                        captured=item
                        session.pin_network_capture(identity=identity,timestamp=item.capture_s,sequence=item.sequence)
                        snapshot=sensor.candidate_snapshot(); candidate_cfg=dict(cfg['graspgenx'])
                        # Preserve the existing IK budget and all candidate checks,
                        # but retain every candidate that passes within that budget.
                        candidate_cfg['maximum_accepted_candidates']=candidate_cfg['maximum_ik_candidates']
                        if args.grasp_source == 'obb': candidate_cfg['minimum_score']=0.
                        candidate_cfg['preferred_intercept_x']=float(env.scene_config['workspace']['preferred_grasp_x'])
                        if args.grasp_source == 'obb':
                            network=pool.submit(select_obb_candidate,cloud,snapshot,np.asarray(provenance['track_position_world_m']),candidate_cfg)
                        else:
                            network=pool.submit(infer_and_select,backend,cloud,1544,cfg['graspgenx']['generated_candidates'],
                                cfg['graspgenx']['returned_candidates'],snapshot,np.asarray(provenance['track_position_world_m']),candidate_cfg,
                                grasp_position_mode=args.grasp_position_mode)
                        scene=pool.submit(session.prepare_captured_scene,geometry,captured.detection,identity=identity,
                            timestamp=captured.capture_s,sequence=captured.sequence,clock_epoch=captured.clock_epoch)
                        result.update(network_submitted_sim_s=float(env.data.time),capture_s=item.capture_s,
                            scene_submitted_wall_s=time.perf_counter()-sim_start_wall,
                            scene_submitted_before_speed_qualified=item.velocity_measurement is None,
                            camera_points=len(item.detection['object_points']),scene_points=len(item.detection['scene_points']))
                except (ValueError,KeyError) as error:result['last_rejection']=str(error)
            if scene is not None and scene.done():
                result.setdefault('scene_ready_observed_wall_s',time.perf_counter()-sim_start_wall)
            if network is not None and network.done():
                result.setdefault('candidate_ready_observed_wall_s',time.perf_counter()-sim_start_wall)
            if selection_future is not None and selection_future.done():
                batch_accepted,batch_rejected,batch_wall=selection_future.result()
                batch_times.append(batch_wall)
                selection_future=None
                before=len(examined)
                examined.update(evaluated_variants(batch_accepted,batch_rejected))
                inference=network.result()
                inference['accepted_candidates']=batch_accepted
                inference['rejected_candidates']=batch_rejected
                candidate_index=0
                result['selection_batches']=selection_batches
                if not batch_accepted:
                    remaining=any(row.get('evaluated') is False and row.get('reason') == 'not_evaluated_budget'
                                  for row in batch_rejected)
                    latest=session.snapshot(float(env.data.time))
                    if remaining and len(examined)>before and latest is not None and latest.valid:
                        selection_batches+=1
                        selection_future=pool.submit(select_retry_batch,snapshot,inference,provenance,
                            candidate_cfg,cloud,set(examined))
                        continue
                    result['status']='retry_batch_no_accepted_candidate';break
                if len(examined)==before:
                    raise RuntimeError('candidate retry made no progress')
            if ctx is None and selection_future is None and network is not None and network.done() and scene.done():
                current=session.snapshot(float(env.data.time))
                if current is None or not current.valid or task_id not in observation.get('predictions', {}):
                    continue
                inference=network.result();prepared=scene.result();chosen=inference['accepted_candidates']
                if scene_prototype is None:
                    # Snapshot the nominal model before a context captures its
                    # start-state tolerance. Every retry gets its own isolated
                    # model; never expand a previous context's fixed limits.
                    scene_prototype=_copy_geometry(prepared.geometry)
                geometry=_copy_geometry(scene_prototype)
                prepared=replace(prepared,geometry=geometry)
                examined.update(evaluated_variants(chosen,inference['rejected_candidates']))
                result.setdefault('network_ready_sim_s',float(env.data.time))
                result.update(candidate_count=len(inference['poses']),
                    accepted_candidates=len(chosen),candidate_rejections=[{k: row.get(k) for k in ('source_index','variant','reason','approach_tilt_deg','tcp_height_above_belt_m','minimum_open_finger_clearance_m','pregrasp_ik','intercept_ik')} for row in inference['rejected_candidates']],inference_s=inference['metadata']['inference_seconds'],
                    candidate_filter_s=inference['metadata']['candidate_filter_wall_s'],scene_install_s=prepared.install_wall_s)
                if args.grasp_source == 'network': result['network_candidates']=len(inference['poses'])
                if 'center_grasp_info' in inference:
                    result['center_grasp_info']=inference['center_grasp_info']
                if 'obb_info' in inference:
                    result['obb_info']=inference['obb_info']
                if not chosen or candidate_index >= len(chosen):
                    if not chosen and any(row.get('evaluated') is False and row.get('reason') == 'not_evaluated_budget'
                                          for row in inference['rejected_candidates']):
                        selection_batches+=1
                        selection_future=pool.submit(select_retry_batch,snapshot,inference,provenance,
                            candidate_cfg,cloud,set(examined))
                        result['status']='retrying_next_candidate'
                        continue
                    result['status']='no_candidate';break
                bound,admission=bind_measured_network(session=session,captured=captured,task_id=task_id,
                    provenance=provenance,candidate=chosen[candidate_index],raw_observation=observation,
                    world_to_base=env.nero_world_to_base,clock_epoch=identity.clock_epoch,
                    max_observation_age_s=cfg['tracking']['maximum_track_age_s'],
                    max_result_age_s=cfg['tracking']['maximum_result_age_s'])
                display_pose=env.nero_base_to_world @ bound['grasp']
                scene_binding=session.bind_prepared_scene(prepared,identity=identity)
                latest=session.snapshot(float(env.data.time))
                result.update(status='measured_network_scene_bound',admission=admission,
                    velocity_source=latest.velocity_source,velocity_base_m_s=latest.velocity.tolist(),
                    latest_observation_s=latest.observed_timestamp,scene_capture_s=scene_binding.capture_time,
                    selected_network_tcp_base=bound['grasp'].tolist(),closing_width_m=chosen[candidate_index]['closing_width_m'],
                    selected_candidate_index=candidate_index,selected_source_index=chosen[candidate_index]['source_index'],
                    selected_candidate_details={key:chosen[candidate_index].get(key) for key in
                        ('score','score_semantics','grasp_position_mode','obb_closing_axis','original_obb_pose_world','original_network_pose_world',
                         'T_world_tcp','visible_box_center_world_m','variant')},
                    scene_dataflow_s=prepared.report.get('dataflow_timings_s',{}),
                    sim_wall_s=time.perf_counter()-sim_start_wall)
                # Diagnostic only: the actual full planner is run, but no
                # commands are sent to physics until the execution path exists.
                now=float(env.data.time); q=np.asarray(sensor.get_joint_positions()); jaw=sensor._jaw_state
                width=float(jaw['opening_width']);geometry.set_state(q,width);flange=geometry.flange()
                current_tcp_marker=(env.nero_base_to_world @ geometry.tcp())[:3,3]
                state={'joints':q,'pose':np.r_[flange[:3,3],Rotation.from_matrix(flange[:3,:3]).as_euler('xyz')],
                    'width_m':width,'force_n':float(np.max(np.abs(jaw['motor_effort']))),
                    'joint_stamp':now,'gripper_stamp':now,'pose_stamp':now}
                settings=PickSettings(speed_percent=100,planning_mode='full_preplanned',
                    moving_approach_mode='continuous',selection_policy='first_safe',replan_policy='event_triggered',
                    **({} if compute_budget is None else {'full_plan_compute_budget_s':compute_budget}))
                monitor={'read_frame':lambda seq: latest_frame[0] if latest_frame[0] is not None and
                    latest_frame[0].header.sequence != seq else None,
                    'calibrate_frame':lambda frame:frame,'capture_monotonic':lambda frame:frame.header.sample_time,
                    'belt_region':region,'clock':lambda:float(env.data.time)}
                direction,belt_origin=conveyor_world_x_axis_in_base(env.nero_world_to_base)
                origin=float(env.data.time)
                ctx=session.create_full_context(geometry,state,settings,identity=identity,scene_binding=scene_binding,
                    policy_clock_origin=origin,belt_direction=direction,belt_origin=belt_origin,
                    s_range=env.grasp_region,scene_monitor_options=monitor)
                if args.execute_sim:
                    result['controller_limit_comparison']={
                        'planning_command_lower_rad':ctx.robot_controller.command_joint_lower_limits.copy(),
                        'planning_command_upper_rad':ctx.robot_controller.command_joint_upper_limits.copy(),
                        'simulation_command_lower_rad':env.robot_controller.command_joint_lower_limits.copy(),
                        'simulation_command_upper_rad':env.robot_controller.command_joint_upper_limits.copy(),
                        'planning_max_velocity_rad_s':ctx.robot_controller.max_joint_velocity.copy(),
                        'simulation_max_velocity_rad_s':env.robot_controller.max_joint_velocity.copy(),
                        'planning_max_acceleration_rad_s2':ctx.robot_controller.max_joint_acceleration.copy(),
                        'simulation_max_acceleration_rad_s2':env.robot_controller.max_joint_acceleration.copy()}
                    result['effective_simulation_command_envelope']=align_simulation_command_envelope(ctx,env.robot_controller,q)
                sim_settings={}
                for name,value in (('singularity_hard_stop_condition',args.sim_condition_limit),
                                   ('cartesian_linear_speed',args.sim_tcp_linear_limit),
                                   ('cartesian_angular_speed',args.sim_tcp_angular_limit)):
                    if value is not None: sim_settings[name]=value
                if sim_settings:
                    ctx.policy.config=replace(ctx.policy.config,**sim_settings)
                    result['simulation_only_limit_overrides']=sim_settings
                if args.sim_disable_tcp_and_singularity_limits:
                    ctx.policy.config=replace(ctx.policy.config,singularity_hard_stop_condition=float('inf'),
                                              tcp_speed_limits_enabled=False)
                    result['simulation_only_disabled_limits']=['jacobian_condition_upper_bound',
                        'tcp_linear_speed_upper_bound','tcp_angular_speed_upper_bound']
                result['effective_planning_limits']={name:getattr(ctx.policy.config,name) for name in
                    ('singularity_hard_stop_condition','singularity_slowdown_condition',
                     'cartesian_linear_speed','cartesian_angular_speed','minimum_joint_limit_margin')}
                if args.sim_disable_tcp_and_singularity_limits:
                    result['effective_planning_limits']['singularity_hard_stop_condition']=None
                    result['effective_planning_limits']['tcp_speed_limits_enabled']=False
                    result['effective_planning_limits']['cartesian_speeds_role']='nominal_intercept_timing_only_not_tcp_limits'
                if args.warm_roadmap:
                    result['roadmap_at_context_creation']=ctx.planning_acceleration_report().get('warm_roadmap')
                if args.min_approach_s is not None:
                    ctx.policy.config=replace(ctx.policy.config,descent_lead_time=args.min_approach_s)
                    result['minimum_approach_s']=args.min_approach_s
                if args.freeze_visual:
                    ctx.policy.config=replace(ctx.policy.config,full_plan_observation_max_age_s=float(args.seconds)+1.,full_plan_replanning_enabled=False)
                    visual_frozen=True
                    frozen_snapshot=session.snapshot(float(env.data.time))
                    initial_marker=np.asarray(provenance['track_position_world_m'],dtype=float)
                    if abs(float(admission['live_observed_timestamp_s'])-float(frozen_snapshot.observed_timestamp))<=1e-6:
                        freeze_marker=np.asarray(admission['live_position_world_m'],dtype=float)
                    else:
                        freeze_marker=None
                    frozen_velocity_world=env.nero_base_to_world[:3,:3]@np.asarray(frozen_snapshot.velocity)
                    result.update(frozen_observed_position_world_m=None if freeze_marker is None else freeze_marker.tolist(),
                        initial_detection_position_world_m=initial_marker.tolist(), frozen_velocity_world_m_s=frozen_velocity_world.tolist(),
                        visual_frozen=True, frozen_observed_timestamp_s=frozen_snapshot.observed_timestamp,
                        frozen_velocity_base_m_s=frozen_snapshot.velocity.tolist(),
                        frozen_velocity_observed_timestamp_s=frozen_snapshot.velocity_observed_timestamp,
                        frozen_observation_age_at_plan_s=float(env.data.time)-float(frozen_snapshot.observed_timestamp),
                        frozen_mode_scope='simulator_only_stale_visual_prediction_no_online_replan',
                        original_capture_time_preserved=True)
                if args.disable_scene_review:
                    ctx.full_plan_scene_monitor.close()
                    del ctx.full_plan_scene_monitor
                    result['scene_review_disabled_reason']='explicit_user_requested_simulation_only_ablation'
                result['full_context_created_sim_s']=origin
                result['full_controller_mode']=ctx.policy.config.planning_mode
                # This is a simulation-only reference proposal, not a measured
                # hardware delay. The original full certificate remains mandatory.
                worker=ctx.policy._ensure_planning_worker()
                if worker is None:
                    raise RuntimeError('full planner worker unavailable: '+str(ctx.policy._last_error))
                worker.set_candidate_wall_budget(candidate_wall_budget)
                current_run={'attempt':len(planning_runs)+1,'candidate_index':candidate_index,
                    'source_index':chosen[candidate_index]['source_index'], 'context_created_sim_s':origin,
                    'wall_budget_s':candidate_wall_budget, 'motion_commands_before_attempt':dispatched}
                input_snapshot=session.snapshot(float(env.data.time))
                current_run['input']={'capture_s':captured.capture_s,
                    'observed_timestamp_s':input_snapshot.observed_timestamp,
                    'velocity_base_m_s':input_snapshot.velocity.tolist(),
                    'observed_grasp_world':(env.nero_base_to_world @ input_snapshot.observed_grasp).tolist(),
                    'effective_planning_limits':dict(result['effective_planning_limits'])}
                planning_runs.append(current_run)
                progress_cursor=0;last_progress_wall=0.;last_progress_stage=None
                install_simulation_advance(worker._engine, .1)
                result['simulation_reference_advance_s']=.1
            if ctx is not None:
                now=float(env.data.time);q=np.asarray(sensor.get_joint_positions());jaw=sensor._jaw_state
                width=float(jaw['opening_width']);geometry.set_state(q,width);flange=geometry.flange()
                current_tcp_marker=(env.nero_base_to_world @ geometry.tcp())[:3,3]
                state={'joints':q,'pose':np.r_[flange[:3,3],Rotation.from_matrix(flange[:3,:3]).as_euler('xyz')],
                    'width_m':width,'force_n':float(np.max(np.abs(jaw['motor_effort']))),
                    'joint_stamp':now,'gripper_stamp':now,'pose_stamp':now,
                    'feedback_time':{'joint':{'mapping_quality':'mapped'},'gripper':{'mapping_quality':'mapped'}},
                    'joint_stamp_monotonic':now,'gripper_stamp_monotonic':now,'pose_stamp_monotonic':now}
                ob=ctx.observe(state,now-origin)
                if ctx.policy.stage.value == 'waiting' and not ctx._planner_requests:
                    readiness=remaining_window_check(ob,ctx._target['object_id'],ctx.policy)
                    readiness['candidate_source_index']=chosen[candidate_index]['source_index']
                    readiness['predicted_pose']=ob['predictions'][ctx._target['object_id']].get('predicted_grasp_pose')
                    prediction=ob['predictions'][ctx._target['object_id']]
                    readiness['predicted_pose_frame']='robot_base'
                    readiness['predicted_position_world_m']=point_world(readiness['predicted_pose'],env.nero_base_to_world)
                    readiness['prediction_policy_time_s']=prediction.get('estimated_grasp_time')
                    readiness['prediction_sim_time_s']=None if prediction.get('estimated_grasp_time') is None else origin+float(prediction['estimated_grasp_time'])
                    readiness['meaning']='window prediction; not necessarily the actual planner request'
                    readiness['target_hook_feasible']=ctx.is_target_feasible(ctx._target['object_id'],ob)
                    result['selection_window_check']=readiness
                    if not readiness['available']:
                        result['status']='candidate_window_insufficient';break
                if args.execute_sim and ctx.policy._full_pick_bundle is not None:
                    ob['control_dt']=float(env.control_dt)
                    ob['dispatch_clock']={'scheduled_dispatch_policy_time_s': now-origin+float(env.control_latency),
                        'planned_interval_s':float(env.control_dt)}
                if args.execute_sim: ob['_defer_arm_motion']=True
                action=ctx.policy.act(ob)
                if args.execute_sim and action.get('_arm_motion_pending'):
                    action=ctx.policy.finalize_pending_motion(ob)
                trace_progress(worker,force=bool(ctx.policy._full_pick_abort_reason))
                stage=ctx.policy.stage.value
                if stage != last_stage:
                    print('仿真阶段:',stage,'仿真时间:',round(now,3),flush=True);last_stage=stage
                result['planning_executed']=bool(ctx._planner_requests)
                result['planner_requests']=[{k:v for k,v in row.items() if k!='request_snapshot'} for row in ctx._planner_requests]
                if ctx._planner_requests:
                    request_snapshot=ctx._planner_requests[-1].get('request_snapshot') or {}
                    limit=request_snapshot.get('singularity_hard_stop_condition')
                    result['submitted_trajectory_constraints']={
                        'singularity_hard_stop_condition':None if limit is None or not np.isfinite(limit) else float(limit),
                        'continuation_tcp_speed_limits':request_snapshot.get('continuation_tcp_speed_limits'),
                        'tcp_speed_limits_enabled':request_snapshot.get('tcp_speed_limits_enabled')}
                target_records=[request_target_record(row,env.nero_base_to_world,origin) for row in ctx._planner_requests]
                target_records=[row for row in target_records if row is not None]
                if target_records:
                    result['requested_contact_targets']=target_records
                    requested_contact_marker=np.asarray(target_records[-1]['target_tcp_position_world_m'])
                result['planner_results']=ctx._planner_results
                result['planner_result_target_diagnostics']=[record for record in
                    (result_target_record(row,env.nero_base_to_world) for row in ctx._planner_results) if record is not None]
                result['policy_full_diagnostics']=ctx.policy.get_info().get('full_preplanned',{})
                result['collision_rejection']=ctx.motion_validation_rejection
                result['planning_stage']=ctx.policy.stage.value
                result['planning_abort_reason']=ctx.policy._full_pick_abort_reason
                plan=ctx.policy._full_pick_bundle
                if plan is not None:
                    if not result.get('complete_path_certified'):
                        trace_progress(worker,force=True)
                        current_run.update(outcome='certified',planner_results=list(ctx._planner_results))
                        if viewer is not None:
                            if overlay_robot is None: overlay_robot=RobotModel()
                            display_path=tcp_path_from_samples(plan.checked_samples,overlay_robot,env.nero_base_to_world)
                            display_path_label='CURRENT certified path (lift conditional)'
                            path_drawn=False
                            result['trajectory_overlay']={'source':'current_certified_plan_display_only','points':len(display_path)}
                        attempt_records.append({'candidate_index':candidate_index,'source_index':chosen[candidate_index]['source_index'],
                            'status':'certified','planner_results':ctx._planner_results})
                        result.update(status='full_plan_received_not_executed',complete_path_certified=True,
                            plan_start_s=plan.start,contact_s=plan.contact_time,
                            plan_request_id=plan.request.request_id)
                    if not args.execute_sim: break
                    if dispatch is None:
                        dispatch=OBBSimulationDispatch(env,ctx,origin)
                        def application_guard(source,application_time):
                            sensor.get_observation()
                            actual_q=np.asarray(sensor.get_joint_positions()); actual_jaw=sensor._jaw_state
                            actual_width=float(actual_jaw['opening_width']);geometry.set_state(actual_q,actual_width);actual_flange=geometry.flange()
                            actual_state={'joints':actual_q,'pose':np.r_[actual_flange[:3,3],Rotation.from_matrix(actual_flange[:3,:3]).as_euler('xyz')],
                                'width_m':actual_width,'force_n':float(np.max(np.abs(actual_jaw['motor_effort']))),
                                'joint_stamp':application_time,'gripper_stamp':application_time,'pose_stamp':application_time,
                                'feedback_time':{'joint':{'mapping_quality':'mapped'},'gripper':{'mapping_quality':'mapped'}},
                                'joint_stamp_monotonic':application_time,'gripper_stamp_monotonic':application_time,'pose_stamp_monotonic':application_time}
                            actual_ob=ctx.observe(actual_state,application_time-origin)
                            actual_ob['control_dt']=float(env.control_dt)
                            return dispatch.prepare(source,actual_ob,actual_state,application_sim_s=application_time)
                        env.obb_apply_guard=application_guard
                        def applied_callback(sent,application_time):
                            policy_time=application_time-origin
                            ctx.policy.full_pick_command_sent(policy_time)
                            if ctx.policy.stage.value == 'closing' and float(sent['gripper']) < float(ctx.policy.config.gripper_open_action):
                                ctx.policy.full_pick_closure_sent(policy_time)
                        env.obb_applied=applied_callback
                    pending_action=action
                    if ctx.policy._full_pick_finished:
                        # No new motion command: retain the last physical target
                        # for the existing independent 0.5 s retention metric.
                        hold_until=float(env.data.time)+.5+env.control_dt
                        while float(env.data.time)<hold_until:
                            env.step(None)
                            held=next((r for r in env._objects if r.logical_id is not None),None)
                            if held is None: break
                            f=env._finger_object_contact_forces(held)
                            supported=any(env.belt_geom_id in (c.geom1,c.geom2) and bool({int(c.geom1),int(c.geom2)} & set(held.geom_ids)) for c in env.data.contact[:env.data.ncon])
                            physical_rows.append({'sim_s':float(env.data.time),'object_pose':env.data.qpos[held.qpos_adr:held.qpos_adr+7].copy(),
                                'object_active':True,'left_normal_force_n':float(f[0]),'right_normal_force_n':float(f[1]),
                                'object_belt_contact':supported,'gripper_command':float(env._gripper_observation()['command'])})
                        pending_action=None
                        result.update(status='simulation_lift_policy_completed',physical_grasp_verified=False);break
                if ctx.policy._full_pick_abort_reason:
                    reason=ctx.policy._full_pick_abort_reason
                    current_run.update(outcome=reason,planner_results=list(ctx._planner_results))
                    attempt_records.append({'candidate_index':candidate_index,'source_index':chosen[candidate_index]['source_index'],
                        'status':'rejected','reason':reason,'planner_results':ctx._planner_results})
                    # Retry only candidate-dependent geometry/IK failures. Time,
                    # feedback, cancellation, and scene failures must not be
                    # reclassified as evidence that another grasp is safe.
                    candidate_failure = reason in RETRYABLE_GEOMETRY_REJECTIONS
                    budget_failure=(reason == 'planning_candidate_budget_exceeded' and dispatched == 0
                                    and ctx.policy._full_pick_bundle is None)
                    late = not args.freeze_visual and reason == 'full_pick_delivery_missed_start' and delivery_retries == 0
                    if not candidate_failure and not late and not budget_failure:
                        result['status']='full_plan_rejected';break
                    # Tear down the old planner and its scene monitor before
                    # accepting another pose. This harness never sends commands.
                    if late:
                        starts=[e for e in ctx.policy._full_pick_events if e['event']=='full_plan_start']
                        if not starts:
                            result['status']='delivery_retry_missing_start_time';break
                        elapsed=(now-origin)-float(starts[-1]['timestamp'])
                        # Reuse actual elapsed policy-clock time; preserve the
                        # configured minimum and existing delivery lead. A new
                        # request and full certification are still required.
                        compute_budget=max(float(ctx.policy.config.full_plan_compute_budget_s),
                            elapsed+float(ctx.policy.config.trajectory_minimum_duration)+ctx.policy._execution_dt(ob))
                        retry_window=remaining_window_check(ob,ctx._target['object_id'],ctx.policy,
                            budget_override=compute_budget)
                        result['delivery_retry_window_check']=retry_window
                        if not retry_window['available']:
                            result['status']='delivery_retry_window_insufficient';break
                        delivery_retries+=1
                        result['delivery_retry_compute_budget_s']=compute_budget
                    retry_window=remaining_window_check(ob,ctx._target['object_id'],ctx.policy)
                    current_run['retry_window_check']=retry_window
                    if not retry_window['available']:
                        result['status']='retry_window_insufficient';break
                    retiring_worker=worker
                    session=rebuild_after_rejection(ctx,session,identity,captured,recent,now,reason,
                        allow_late_delivery=late,allow_budget_retry=budget_failure)
                    trace_progress(retiring_worker,force=True)
                    current_run['worker_stopped_before_retry']=not retiring_worker.is_alive
                    ctx=None
                    if retiring_worker.is_alive:
                        result['status']='retiring_worker_did_not_stop';break
                    if late:
                        # Replan the same candidate from fresh observations;
                        # never shift or dispatch the expired certificate.
                        pass
                    elif candidate_index+1 < len(chosen):
                        candidate_index+=1
                    else:
                        unexamined=any(row.get('evaluated') is False and row.get('reason')=='not_evaluated_budget'
                                       for row in inference['rejected_candidates'])
                        if not unexamined:
                            result['status']=('candidate_planning_budgets_exhausted' if budget_failure
                                              else 'evaluated_candidates_exhausted')
                            break
                        # Continue from the original network outputs with the
                        # same per-batch IK budget, preserving source indexes.
                        # The snapshot and cloud belong to the original visual
                        # exposure; admission and full scene validation still
                        # run anew before any later candidate can be used.
                        selection_batches+=1
                        selection_future=pool.submit(select_retry_batch,snapshot,inference,provenance,
                            candidate_cfg,cloud,set(examined))
                    if not late:
                        delivery_retries=0;compute_budget=args.compute_budget_s
                    result['status']='retrying_next_candidate'
                    display_pose=None
            if viewer is not None and time.monotonic()-last_view_wall >= .1:
                if not viewer.is_running():
                    result['status']='viewer_closed';break
                with viewer.lock():
                    if display_points is not None and float(env.data.time)-display_stamp <= 1.:
                        reserve=200 if display_path is not None else 12
                        draw_live_cloud(viewer.user_scn,display_points,display_pose,max_points=max(0,min(3500,viewer.user_scn.maxgeom-reserve)))
                    else:
                        viewer.user_scn.ngeom=0  # never present an expired capture as live
                    if display_path is not None:
                        draw_tcp_path(viewer.user_scn,display_path);path_drawn=True
                    if args.freeze_visual and visual_frozen:
                        import mujoco
                        for marker,colour in ((initial_marker,[.65,.2,1.,1.]),(freeze_marker,[1.,.1,.05,1.]),
                                              (requested_contact_marker,[.1,1.,.1,1.]),(current_tcp_marker,[1.,1.,1.,1.])):
                            if marker is None or viewer.user_scn.ngeom>=viewer.user_scn.maxgeom: continue
                            geom=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.014,0,0]),marker,np.eye(3).ravel(),np.array(colour,dtype=np.float32))
                            viewer.user_scn.ngeom+=1
                        p=result.get('frozen_observed_position_world_m')
                        vals=f"{result['capture_s']:.3f} s\n{result['frozen_observed_timestamp_s']:.3f} s\n{float(result['frozen_velocity_observed_timestamp_s']):.3f} s\n{100*result['frozen_velocity_world_m_s'][0]:.3f} cm/s\n"+(('%.3f, %.3f, %.3f m'%tuple(p)) if p is not None else 'not timestamp matched')
                        title='Purple: grasp-input capture\nRed: frozen position capture\nVelocity sample time\nFrozen speed world X\nRed world XYZ\nGreen: requested contact (not a path)\nWhite: actual TCP'
                        if display_path_label: title+='\n'+display_path_label+'\nPath: green approach, pink close, blue lift'
                        viewer.set_texts((None,None,title,vals))
                    elif args.freeze_visual:
                        viewer.set_texts((None,None,'Frozen-speed demonstration','Waiting for initial observation'))
                    elif display_path_label:
                        viewer.set_texts((None,None,display_path_label,'Green approach | yellow descent | pink close | blue lift'))
                viewer.sync();last_view_wall=time.monotonic()
            time.sleep(.002)
        if result['status']=='starting':result['status']='timeout_before_scene_and_network'
        elif result['status']=='retrying_next_candidate':result['status']='timeout_during_candidate_retry'
        elif result['status']=='measured_network_scene_bound':result['status']='timeout_during_full_planning'
    except Exception as error:
        result.update(status='error',error=f'{type(error).__name__}: {error}')
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        if dispatch is not None:dispatch.stop()
        if ctx is not None:ctx.close()
        if current_run is not None:
            trace_progress(worker,force=True)
        trace_file.close()
        result['post_freeze_display_only_frames']=post_freeze_frame_count
        if args.execute_sim:
            result['physical_contact_summary']=contact_summary(physical_rows)
            result['physics_grasp_verified']=result['physical_contact_summary']['physical_lift_and_hold']
        result['candidate_attempts']=attempt_records
        result['retry_selection_batch_wall_s']=batch_times
        if args.capture_output is not None and display_points is not None:
            args.capture_output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.capture_output,points_world=display_points,
                capture_sim_s=display_stamp,grasp_world=(display_pose if display_pose is not None else np.empty((0,))))
            result['last_visual_capture']=str(args.capture_output)
        result['total_wall_s']=time.perf_counter()-wall_begin
        def serial(x):
            if isinstance(x,np.ndarray):return x.tolist()
            if isinstance(x,np.generic):return x.item()
            raise TypeError(type(x).__name__)
        args.output.write_text(json.dumps(result,default=serial,ensure_ascii=False,indent=2)+'\n')
        summary={key:result.get(key) for key in ('status','error','velocity_source',
            'candidate_count','accepted_candidates','full_controller_mode',
            'simulation_reference_advance_s','planning_abort_reason','complete_path_certified',
            'physics_commands_sent','dispatch_rejection','physics_grasp_verified','total_wall_s') if key in result}
        print(json.dumps(summary,default=serial,ensure_ascii=False),flush=True)
        if viewer is not None:
            if args.hold_viewer and viewer.is_running():
                if display_path is not None and not path_drawn:
                    with viewer.lock():
                        viewer.user_scn.ngeom=0
                        draw_tcp_path(viewer.user_scn,display_path)
                        viewer.set_texts((None,None,display_path_label,'Green approach | yellow descent | pink close | blue lift'))
                print('仿真已结束；最终画面保持打开，关闭窗口退出。',flush=True)
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.1)
            viewer.close()
        if env is not None:env.close()
    return 0 if result['status']=='full_plan_received_not_executed' else 1


if __name__=='__main__':raise SystemExit(main())

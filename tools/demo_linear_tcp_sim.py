"""MuJoCo-only straight TCP motion with actual actuator/encoder feedback."""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import mujoco
import yaml
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
from experiments.graspgenx_sim.linear_tcp_demo import LinearTCPReference


def main():
    parser=argparse.ArgumentParser(description='MuJoCo 中单独验证 TCP 直线匀速，绝不连接硬件')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distance-m',type=float,default=.12)
    parser.add_argument('--speed-m-s',type=float,default=.10)
    parser.add_argument('--ramp-s',type=float,default=.30)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--hold-viewer',action='store_true')
    parser.add_argument('--playback-rate',type=float,default=1.,help='可视播放速率；0.25 为四倍慢放，仿真时钟和物理速度不变')
    parser.add_argument('--start-delay-s',type=float,default=2.,help='可视窗口打开后、仿真开始前的墙钟等待秒数')
    args=parser.parse_args()
    if args.viewer and not os.environ.get('DISPLAY'):parser.error('--viewer 需要图形桌面')
    if args.hold_viewer and not args.viewer:parser.error('--hold-viewer 需要 --viewer')
    if not np.isfinite(args.playback_rate) or not 0 < args.playback_rate <= 1.:parser.error('--playback-rate 须在 (0,1]')
    if not np.isfinite(args.start_delay_s) or not 0 <= args.start_delay_s <= 30:parser.error('--start-delay-s 须在 [0,30]')
    cfgpath=Path(__file__).resolve().parents[1]/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml'
    cfg=yaml.safe_load(cfgpath.read_text())
    wait=cfg['control']['simulation_standby_tcp_world_m']
    quat=cfg['control']['simulation_standby_quaternion_wxyz']
    overrides={'robot':{'robot':{'initialize_at_wait_pose':True}},
        'scene':{'workspace':{'waiting_ee_position':wait,'waiting_ee_quaternion_wxyz':quat},
                 'episode':{'initial_objects':0},'spawning':{'interval_seconds':1000.},
                 'conveyor':{'default_speed':0.,'speed_jitter_std':0.}}}
    env=viewer=None;rows=[];result={'mode':'mujoco_only_standalone_cartesian_line_ik_joint_position',
        'hardware_connections':0,'motion_commands_sent':0,'physical_grasp_claimed':False,
        'reference_speed_m_s':args.speed_m_s,'requested_distance_m':args.distance_m,
        'acceleration_time_s':args.ramp_s,'status':'starting'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        env=ConveyorSortingEnv(config_overrides=overrides,render_observations=False,
            perception_mode='simulator_history',grasp_assist=False,
            grasp_event_source='physical_contacts',action_type='joint_position')
        env.reset(seed=1543,options={'belt_speed':0.,'initial_objects':0})
        start_pose=env._ee_pose();start_q=np.asarray(env.get_joint_positions(),float)
        if np.linalg.norm(start_pose[:3]-np.asarray(wait))>.002:
            raise ValueError('仿真等待位逆解未达到位置容差')
        if env.robot_controller is None:raise RuntimeError('仿真关节执行器不可用')
        ref=LinearTCPReference(start_pose[:3],np.array([1.,0.,0.]),args.distance_m,args.speed_m_s,args.ramp_s)
        dt=float(env.control_dt); latency=float(env.control_latency)
        # Precompute a same-orientation IK solution for each command tick.
        # Each endpoint and intermediate joint edge is tested against the
        # current MuJoCo static scene; actual plant feedback is recorded later.
        count=int(np.ceil(ref.duration_s/dt))+1
        stamps=np.minimum(np.arange(count)*dt,ref.duration_s)
        joint_samples=[];seed=start_q.copy();previous=start_q.copy()
        command_lower=np.asarray(env.robot_controller.command_joint_lower_limits)
        command_upper=np.asarray(env.robot_controller.command_joint_upper_limits)
        plan_started=time.perf_counter()
        for stamp in stamps:
            position,_,_=ref.sample(stamp)
            ik=env.ik.solve(position,start_pose[3:],qpos_seed=seed)
            if not ik.success:
                raise RuntimeError(f'仿真 IK 未收敛，t={stamp:.3f}s，位置误差={ik.position_error:.6f}m')
            q=np.asarray(ik.joint_positions,float)
            if np.any(q<command_lower) or np.any(q>command_upper):
                raise RuntimeError(f'目标超出仿真执行器关节命令范围，t={stamp:.3f}s')
            steps=max(1,int(np.ceil(np.max(np.abs(q-previous))/.008)))
            for fraction in np.linspace(0.,1.,steps+1):
                checked=previous+(q-previous)*fraction
                if not env.is_arm_configuration_safe(checked,minimum_tcp_height=env.belt_height,
                        minimum_joint_margin=env.robot_controller.config.joint_command_margin):
                    raise RuntimeError(f'仿真静态场景或关节状态检查未通过，t={stamp:.3f}s')
            joint_samples.append(q.copy());seed=q;previous=q
        result.update(reference_start_world_m=ref.start_world_m.tolist(),
            reference_end_world_m=ref.sample(ref.duration_s)[0].tolist(),reference_duration_s=ref.duration_s,
            control_dt_s=dt,simulated_actuator_latency_s=latency,ik_sample_count=len(joint_samples),
            preparation_wall_s=time.perf_counter()-plan_started,
            state_check_scope='sampled_static_mujoco_scene_and_joint_edges_not_continuous_hardware_certificate',
            initialization_scope='mujoco_reset_ik_no_prior_motion')
        if args.viewer:
            from mujoco import viewer as mj_viewer
            viewer=mj_viewer.launch_passive(env.model,env.data)
            result['viewer_playback_rate']=args.playback_rate
            print('MuJoCo 直线运动：绿色=目标路径，蓝色=实际 TCP 路径。',flush=True)
            viewer.sync()
            if args.start_delay_s:time.sleep(args.start_delay_s)
        # Stationary settle at reset position. No connection from a prior pose
        # is implied; this entry point is a separate simulation experiment.
        for _ in range(round(.4/dt)):
            env.step({'arm':start_q,'gripper':1.})
            if viewer:
                viewer.sync();time.sleep(dt/args.playback_rate)
        origin=float(env.data.time);wall_origin=time.monotonic();last_view=0.
        jacp=np.zeros((3,env.model.nv));jacr=np.zeros((3,env.model.nv))
        samples=np.asarray(joint_samples)
        while float(env.data.time)-origin < ref.duration_s+1.0:
            elapsed=float(env.data.time)-origin
            command_t=min(ref.duration_s,max(0.,elapsed+latency))
            index=min(len(stamps)-1,int(round(command_t/dt)))
            env.step({'arm':samples[index], 'gripper':1.})
            result['motion_commands_sent']+=1
            stamp=float(env.data.time)-origin
            measured=env._ee_pose();desired,desired_velocity,phase=ref.sample(stamp)
            mujoco.mj_jacSite(env.model,env.data,jacp,jacr,env.ee_site_id)
            actual_velocity=jacp@env.data.qvel
            rotation_error=(Rotation.from_quat(start_pose[3:][[1,2,3,0]]).inv()*
                Rotation.from_quat(measured[3:][[1,2,3,0]])).magnitude()
            rows.append({'sim_elapsed_s':stamp,'phase':phase,'target_xyz_m':desired.tolist(),
                'actual_xyz_m':measured[:3].tolist(),'target_vx_m_s':float(desired_velocity[0]),
                'actual_vx_m_s':float(actual_velocity[0]),'orientation_error_rad':float(rotation_error),
                'joint_positions_rad':np.asarray(env.get_joint_positions()).tolist()})
            if getattr(env,'_severe_collision',False):
                result['status']='simulation_severe_collision';break
            if viewer and time.monotonic()-last_view>=.05:
                with viewer.lock():
                    scene=viewer.user_scn;scene.ngeom=0
                    geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,1.,.2,1.],np.float32))
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.003,ref.start_world_m,ref.sample(ref.duration_s)[0]);scene.ngeom+=1
                    for first,last in zip(rows[-100:-1],rows[-99:]):
                        if scene.ngeom>=scene.maxgeom:break
                        a=np.asarray(first['actual_xyz_m']);b=np.asarray(last['actual_xyz_m'])
                        if np.linalg.norm(a-b)<1e-8:continue
                        geom=scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.5,1.,1.],np.float32))
                        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.002,a,b);scene.ngeom+=1
                    viewer.set_texts((None,None,'TCP straight-line demo\nGreen: target | blue: measured\nPhase\nTarget speed X\nMeasured speed X',
                        f'MuJoCo, playback {args.playback_rate:.2f}x\nFixed orientation\n{phase}\n{100*desired_velocity[0]:.2f} cm/s\n{100*actual_velocity[0]:.2f} cm/s'))
                viewer.sync();last_view=time.monotonic()
            if viewer and not viewer.is_running():result['status']='viewer_closed';break
            if viewer:
                delay=wall_origin+stamp/args.playback_rate-time.monotonic()
                if delay>0:time.sleep(delay)
        if result['status']=='starting':result['status']='motion_completed'
        if rows:
            cruise=[r for r in rows if r['phase']=='cruise']
            vx=np.asarray([r['actual_vx_m_s'] for r in cruise],float)
            steady=[r for r in cruise if r['sim_elapsed_s'] >= args.ramp_s+.2]
            steady_vx=np.asarray([r['actual_vx_m_s'] for r in steady],float)
            xyz=np.asarray([r['actual_xyz_m'] for r in rows],float)
            target=np.asarray([r['target_xyz_m'] for r in rows],float)
            result.update(cruise_sample_count=len(cruise),
                actual_cruise_mean_vx_m_s=None if not len(vx) else float(vx.mean()),
                actual_cruise_std_vx_m_s=None if not len(vx) else float(vx.std()),
                actual_cruise_min_vx_m_s=None if not len(vx) else float(vx.min()),
                actual_cruise_max_vx_m_s=None if not len(vx) else float(vx.max()),
                steady_sample_count=len(steady),
                steady_start_elapsed_s=args.ramp_s+.2,
                actual_steady_mean_vx_m_s=None if not len(steady_vx) else float(steady_vx.mean()),
                actual_steady_std_vx_m_s=None if not len(steady_vx) else float(steady_vx.std()),
                maximum_cross_track_m=float(np.max(np.linalg.norm(xyz[:,1:]-ref.start_world_m[1:],axis=1))),
                maximum_same_time_position_error_m=float(np.max(np.linalg.norm(xyz-target,axis=1))),
                final_position_error_m=float(np.linalg.norm(xyz[-1]-ref.sample(ref.duration_s)[0])),
                max_orientation_error_rad=max(r['orientation_error_rad'] for r in rows),
                velocity_source='MuJoCo site translational Jacobian times simulated qvel, actual state',
                cruise_scope='acceleration and deceleration excluded; control latency and physical tracking remain')
    except Exception as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}')
    finally:
        result['rows']=rows
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        with args.output.with_suffix('.csv').open('w',newline='') as stream:
            fields=['sim_elapsed_s','phase','target_vx_m_s','actual_vx_m_s','target_xyz_m','actual_xyz_m','orientation_error_rad']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)
        if viewer:
            if args.hold_viewer and viewer.is_running():
                print('仿真画面保持打开，关闭窗口退出。',flush=True)
                while viewer.is_running():viewer.sync();time.sleep(.1)
            viewer.close()
        if env:env.close()
    return 0 if result['status']=='motion_completed' else 1


if __name__=='__main__':raise SystemExit(main())

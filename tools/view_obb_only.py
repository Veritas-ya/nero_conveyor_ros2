"""Read-only MuJoCo RGB-D visualization of a geometry-only OBB grasp."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import yaml
import mujoco.viewer
from experiments.graspgenx_sim.run import config_overrides
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation, Nero7SensorControlEnvironment
from experiments.graspgenx_sim.sensor_control import SensorControlConfig
from experiments.graspgenx_sim.camera import cloud_for_track
from experiments.graspgenx_sim.obb_grasp import obb_top_down_grasp
from experiments.graspgenx_sim.candidates import _assets
from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=20);p.add_argument('--output',type=Path,required=True);p.add_argument('--hold-viewer',action='store_true');args=p.parse_args()
    if not os.environ.get('DISPLAY'): p.error('需要图形桌面 DISPLAY')
    root=Path(__file__).resolve().parents[1]; base=root/'src/nero_core/experiments/graspgenx_sim'; cfgpath=base/'nero7_config.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());cfg['experiment'].update(seed=1543,belt_speed_m_s=.1)
    env=Nero7FixedSimulation(capture_path=base/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1], config_overrides=config_overrides(cfg),grasp_assist=False,perception_mode='rendered_rgbd',grasp_event_source='physical_contacts',seed=1543,action_type='joint_position',preserve_control_profile=True)
    sensor=Nero7SensorControlEnvironment(env,SensorControlConfig(maximum_track_age_s=1.))
    assets=base/'gripper_agx';_,vertices,_=_assets(str(assets/'transforms.json'),str(assets/'sweep_volume_params.json'))
    viewer=None;result={'source':'visible_obb_geometry_only','network_called':False,'physics_commands_sent':0,'full_path_certified':False,'physical_grasp_verified':False};last=-1; points=pose=box=None;stamp=None; drawtime=0
    try:
        env.reset(seed=1543,options={'belt_speed':.1,'initial_objects':1});viewer=mujoco.viewer.launch_passive(env.model,env.data)
        print('MuJoCo OBB: 青色=可见目标点云；绿色=OBB；橙色=仅由 OBB 生成的抓姿示意；不发送运动命令。',flush=True)
        while env.data.time < args.seconds and viewer.is_running():
            env.step(None);sensor.get_observation()
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence']; items=meta['measurements']
                qualified=[(k,v) for k,v in items.items() if v['velocity_measurement'].get('speed_valid')]
                if not qualified: continue
                task_id,entry=qualified[0];cloud=cloud_for_track(frame,task_id)
                try:g=obb_top_down_grasp(cloud)
                except ValueError:continue
                points=cloud;pose=g['tcp'];box=g['box'];stamp=float(frame['timestamp'])
                clearance=float(np.min((vertices@g['canonical'][:3,:3].T+g['canonical'][:3,3])[:,2])-env.belt_height)
                minimum=float(cfg['graspgenx']['fingertip_clearance_m']);height=float(pose[2,3]-env.belt_height)
                result.update(status='displayed',capture_sim_s=stamp,track_id=int(task_id),point_count=len(cloud),obb=box,tcp_world=pose.tolist(),canonical_world=g['canonical'].tolist(),tcp_height_above_belt_m=height,open_finger_clearance_m=clearance,required_finger_clearance_m=minimum,geometric_clearance_pass=bool(clearance>=minimum and height>=float(cfg['graspgenx']['minimum_tcp_height_above_belt_m'])), velocity_source='nero7_xy_kalman', network_score=None)
            if time.monotonic()-drawtime>=.1:
                with viewer.lock():
                    if points is not None and env.data.time-stamp<=1:draw_live_cloud(viewer.user_scn,points,pose,obb=box)
                    else:viewer.user_scn.ngeom=0
                viewer.sync();drawtime=time.monotonic()
            time.sleep(.002)
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if viewer is not None:
            if args.hold_viewer and viewer.is_running():
                print('保持最后画面；关闭窗口退出。',flush=True)
                while viewer.is_running():viewer.sync();time.sleep(.1)
            viewer.close()
        env.close()
if __name__=='__main__': main()

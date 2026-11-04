"""Display paired MuJoCo truth and RGB-D Kalman velocity; never command hardware."""
import argparse,csv,json,os,time
from pathlib import Path
import numpy as np
import yaml
import mujoco.viewer
from experiments.graspgenx_sim.run import config_overrides, serializable
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation, Nero7SensorControlEnvironment
from experiments.graspgenx_sim.sensor_control import SensorControlConfig
from experiments.graspgenx_sim.camera import cloud_for_track
from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud
from experiments.graspgenx_sim.speed_comparison import compare_speed_at_capture


class SpeedEvaluationSimulation(Nero7FixedSimulation):
    """Capture physics truth separately; never place it in sensor messages."""
    def __init__(self,*args,**kwargs):
        self.evaluation_truth={}
        super().__init__(*args,**kwargs)
    def _capture_cameras(self,*,force=False):
        if hasattr(self,'_objects'):
            snapshot={int(o.logical_id):self._snapshot_runtime(o) for o in self._objects if o.logical_id is not None}
            self.evaluation_truth[round(float(self.data.time),6)]=snapshot
        return super()._capture_cameras(force=force)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=18.);p.add_argument('--output',type=Path,required=True);p.add_argument('--viewer',action='store_true');p.add_argument('--hold-viewer',action='store_true')
    p.add_argument('--early-vision',action='store_true')
    p.add_argument('--standby-at-reset',action='store_true')
    args=p.parse_args()
    if args.viewer and not os.environ.get('DISPLAY'):p.error('viewer 需要 DISPLAY')
    if args.hold_viewer and not args.viewer:p.error('hold-viewer 需要 viewer')
    root=Path(__file__).resolve().parents[1];base=root/'src/nero_core/experiments/graspgenx_sim';cfgpath=base/'nero7_config.yaml'
    cfg=yaml.safe_load(cfgpath.read_text());cfg['experiment'].update(seed=1543,belt_speed_m_s=.1,feed_interval_s=1000.)
    overrides=config_overrides(cfg)
    if args.standby_at_reset:
        overrides['robot']['robot']['initialize_at_wait_pose']=True
        overrides['scene'].setdefault('workspace',{}).update(
            waiting_ee_position=cfg['control']['simulation_standby_tcp_world_m'],
            waiting_ee_quaternion_wxyz=cfg['control']['simulation_standby_quaternion_wxyz'])
    env=SpeedEvaluationSimulation(capture_path=base/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1],config_overrides=overrides,grasp_assist=False,perception_mode='rendered_rgbd',grasp_event_source='physical_contacts',seed=1543,action_type='joint_position',preserve_control_profile=True,
        simulation_camera_world_xyz=cfg['camera']['early_view_world_xyz_m'] if args.early_vision else None,
        visual_workspace_x_world=cfg['tracking']['early_visual_workspace_x_world_m'] if args.early_vision else None)
    sensor=Nero7SensorControlEnvironment(env,SensorControlConfig(maximum_track_age_s=1.))
    rows=[];viewer=None;last=-1;lastdraw=0.;cloud=None;capture=None; rejected={};speed_values=[]
    try:
        env.reset(seed=1543,options={'belt_speed':.1,'initial_objects':1})
        if viewer:
            raise AssertionError('viewer initialized unexpectedly')
        if args.viewer:
            viewer=mujoco.viewer.launch_passive(env.model,env.data)
            print('仿真已打开：左上显示同一曝光的真实速度、卡尔曼速度与偏差；不发送运动命令。',flush=True)
        while env.data.time<args.seconds and (viewer is None or viewer.is_running()):
            env.step(None);sensor.get_observation();speed_values.append(float(env._current_belt_speed))
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence'];items=meta['measurements'];valid=[(k,v) for k,v in items.items() if v['velocity_measurement'].get('speed_valid')]
                if len(valid)!=1:continue
                task,entry=valid[0]; m=entry['velocity_measurement'];stamp=float(frame['timestamp']);truth_stamp=round(stamp,6);truth=env.evaluation_truth.get(truth_stamp,{})
                # This association is used only for offline scoring.
                if len(truth)==1:
                    body=next(iter(truth.values()))
                else:
                    logical=env._vision_to_logical_evaluation.get(int(task))
                    body=truth.get(logical)
                    if body is None:
                        reason=f'unmatched_truth_count_{len(truth)}';rejected[reason]=rejected.get(reason,0)+1;continue
                try:
                    source=float(m.get('device_timestamp'))
                    row=compare_speed_at_capture(capture_s=stamp,truth_s=float(truth_stamp),truth_velocity_world=body['linear_velocity'],estimate_base=m['velocity_base_m_s'],base_to_world_rotation=env.nero_base_to_world[:3,:3],source_stamp_s=source)
                except (ValueError,TypeError,KeyError) as exc:
                    reason=str(exc);rejected[reason]=rejected.get(reason,0)+1;continue
                _,observed=sensor._observed_state_at(task,stamp)
                row.update(track_id=int(task),visual_sequence=int(meta['sequence']),publication_sim_s=float(env.data.time),actual_belt_speed_m_s=float(env._current_belt_speed),truth_logical_id=int(body['object_id']),
                    measured_position_world_m=None if observed is None else np.asarray(observed['position']).tolist(),
                    truth_position_world_m=np.asarray(body['position']).tolist(),
                    sample_count=m.get('sample_count'),span_s=m.get('span_s'),speed_std_m_s=m.get('speed_std_m_s'))
                rows.append(row);cloud=cloud_for_track(frame,task);capture=stamp
            if viewer and time.monotonic()-lastdraw>=.1:
                with viewer.lock():
                    if cloud is not None and env.data.time-capture<=1.:draw_live_cloud(viewer.user_scn,cloud)
                    else:viewer.user_scn.ngeom=0
                    if rows:
                        r=rows[-1];labels='Belt set\nObject true (capture)\nVision Kalman (capture)\nVision - true\nCapture sim time';values=f"10.000 cm/s\n{100*r['truth_vx_world_m_s']:.3f} cm/s\n{100*r['visual_vx_world_m_s']:.3f} cm/s\n{100*r['error_vx_m_s']:+.3f} cm/s\n{r['capture_sim_s']:.3f} s"
                    else:labels='Belt set\nObject speed';values='10.000 cm/s\nWaiting for visual estimate'
                    viewer.set_texts((None,None,labels,values))
                viewer.sync();lastdraw=time.monotonic()
            time.sleep(.002)
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        errs=np.array([r['error_vx_m_s'] for r in rows]); summary={'belt_set_m_s':.1,'actual_belt_min_m_s':min(speed_values) if speed_values else None,'actual_belt_max_m_s':max(speed_values) if speed_values else None,'sample_count':len(rows),'mean_error_m_s':float(errs.mean()) if len(errs) else None,'mean_absolute_error_m_s':float(np.abs(errs).mean()) if len(errs) else None,'max_absolute_error_m_s':float(np.abs(errs).max()) if len(errs) else None,'rejected_pairings':rejected,'source':'single_object_rendered_rgbd_nero7_xy_kalman','truth_scope':'evaluation_only_never_used_in_perception_or_control','time_pairing':'same exposure simulator state and visual capture; no interpolation','hardware_connections':0,'commands_sent':0,'rows':rows}
        summary['early_vision']=args.early_vision;summary['standby_at_reset']=args.standby_at_reset
        reports_path=args.output.with_suffix('.perception.json')
        reports_path.write_text(json.dumps(serializable(env.nero7_reports),ensure_ascii=False,indent=2)+'\n')
        summary['raw_perception_reports']=str(reports_path)
        args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        csvpath=args.output.with_suffix('.csv');fields=['capture_sim_s','truth_sim_s','visual_source_sim_s','publication_sim_s','truth_vx_world_m_s','visual_vx_world_m_s','error_vx_m_s','actual_belt_speed_m_s','track_id','visual_sequence']
        with csvpath.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False),flush=True)
        if viewer is not None:
            if args.hold_viewer and viewer.is_running():
                print('窗口保持打开，关闭后退出。',flush=True)
                while viewer.is_running():viewer.sync();time.sleep(.1)
            viewer.close()
        env.close()
if __name__=='__main__':main()

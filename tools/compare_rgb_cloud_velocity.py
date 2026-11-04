"""Same-frame simulator comparison of cloud ICP and RGB optical flow velocity.

RGB uses a colour-only orange-object mask once to initialize tracking, then
only RGB images plus a fixed calibrated belt plane. Truth is recorded solely for scoring.
"""
import argparse,csv,json,time
import cv2
from pathlib import Path
import numpy as np
import yaml
from view_sim_velocity import SpeedEvaluationSimulation
from experiments.graspgenx_sim.run import config_overrides
from experiments.graspgenx_sim.nero7_perception import Nero7SensorControlEnvironment
from experiments.graspgenx_sim.sensor_control import SensorControlConfig
from experiments.graspgenx_sim.camera import cloud_for_track
from experiments.graspgenx_sim.visual_velocity_ablation import RGBOpticalFlow,cloud_translation,rgb_orange_fixture_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=18.);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'src/nero_core/experiments/graspgenx_sim';path=base/'nero7_config.yaml';cfg=yaml.safe_load(path.read_text());cfg['experiment'].update(seed=1543,belt_speed_m_s=.1,feed_interval_s=1000.)
    env=SpeedEvaluationSimulation(capture_path=base/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1],config_overrides=config_overrides(cfg),grasp_assist=False,perception_mode='rendered_rgbd',grasp_event_source='physical_contacts',seed=1543,action_type='joint_position',preserve_control_profile=True)
    sensor=Nero7SensorControlEnvironment(env,SensorControlConfig(maximum_track_age_s=1.)); rows=[];last=-1;prev=None;rgb=RGBOpticalFlow(); belt=[]; begun=time.perf_counter()
    try:
        env.reset(seed=1543,options={'belt_speed':.1,'initial_objects':1})
        while env.data.time < args.seconds:
            env.step(None);sensor.get_observation();belt.append(float(env._current_belt_speed))
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence'];entries=[(k,v) for k,v in meta['measurements'].items() if v['velocity_measurement'].get('speed_valid')]
                if len(entries)!=1:prev=None;continue
                task,entry=entries[0]; t=float(frame['timestamp']); truth=env.evaluation_truth.get(round(t,6),{});logical=env._vision_to_logical_evaluation.get(int(task))
                body=(next(iter(truth.values())) if len(truth)==1 else truth.get(logical))
                if body is None:prev=None;continue
                cloud=cloud_for_track(frame,task);cal=frame['top_camera'];mask=(frame['top_instance_labels']==int(task));
                if not rows:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(args.output.with_name('rgb_ablation_first_frame.png')),cv2.cvtColor(frame['top_rgb'],cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(args.output.with_name('rgb_ablation_first_mask.png')),mask.astype(np.uint8)*255)
                rgb_mask=rgb_orange_fixture_mask(frame['top_rgb']) if rgb.previous_gray is None or rgb.track_id!=int(task) else None
                visual,rd=rgb.update(frame['top_rgb'],rgb_mask,t,cal['intrinsics'],cal['camera_to_world_opencv'],float(env.belt_height),int(task))
                cp,cd=None,{'reason':'first_frame'}
                dt=None if prev is None or prev[0]!=int(task) or prev[3]!=int(body['object_id']) else t-prev[1]
                truth_interval=None if dt is None or dt<=0 or dt>.25 else float((body['position'][0]-prev[4])/dt)
                if dt is not None and 0<dt<=.25:cp,cd=cloud_translation(prev[2],cloud,dt)
                prev=(int(task),t,cloud.copy(),int(body['object_id']),float(body['position'][0]))
                m=entry['velocity_measurement'];kf=env.nero_base_to_world[:3,:3]@np.asarray(m['velocity_base_m_s'],float)
                v=float(body['linear_velocity'][0]);rec={'capture_sim_s':t,'truth_vx_m_s':v,'truth_interval_avg_vx_m_s':truth_interval,'belt_m_s':float(env._current_belt_speed),'kalman_vx_m_s':float(kf[0]),'cloud_vx_m_s':None if cp is None else float(cp[0]),'rgb_vx_m_s':None if visual is None else float(visual[0]),'dt_s':dt,'track_id':int(task),'cloud_status':cd['reason'],'rgb_status':rd['reason'],'cloud_wall_s':cd.get('wall_s'),'rgb_wall_s':rd.get('wall_s'),'cloud_matches':cd.get('matches'),'rgb_features':rd.get('features'),'truth_scope':'evaluation_only'};rows.append(rec)
        def stats(key):
            vals=[r for r in rows if r[key] is not None and r['truth_interval_avg_vx_m_s'] is not None];errs=np.array([r[key]-r['truth_interval_avg_vx_m_s'] for r in vals]);times=np.array([r[key.replace('_vx_m_s','_wall_s')] for r in vals if key!='kalman_vx_m_s' and r.get(key.replace('_vx_m_s','_wall_s')) is not None]);return {'valid_samples':len(vals),'coverage':len(vals)/len(rows) if rows else 0.,'mae_m_s':float(np.mean(np.abs(errs))) if len(errs) else None,'rmse_m_s':float(np.sqrt(np.mean(errs**2))) if len(errs) else None,'mean_error_m_s':float(np.mean(errs)) if len(errs) else None,'max_abs_error_m_s':float(np.max(np.abs(errs))) if len(errs) else None,'mean_wall_s':float(np.mean(times)) if len(times) else None,'p95_wall_s':float(np.percentile(times,95)) if len(times) else None}
        common=[r for r in rows if r['truth_interval_avg_vx_m_s'] is not None and r['cloud_vx_m_s'] is not None and r['rgb_vx_m_s'] is not None]
        common_stats={key:{'mae_m_s':float(np.mean([abs(r[key]-r['truth_interval_avg_vx_m_s']) for r in common])) if common else None,'count':len(common)} for key in ('kalman_vx_m_s','cloud_vx_m_s','rgb_vx_m_s')}
        out={'description':'same rendered RGB-D frames; ICP translation versus RGB KLT flow','belt_set_m_s':.1,'belt_min_m_s':min(belt) if belt else None,'belt_max_m_s':max(belt) if belt else None,'network_called':False,'hardware_connections':0,'motion_commands':0,'rgb_method':'RGB-only orange colour fixture initialization then KLT RGB features; fixed calibrated belt plane; no per-frame depth; evaluation frames and track id gated by shared RGB-D detector','cloud_method':'translation-only ICP of successive visible target point clouds, median initialization, no speed prior','truth':'MuJoCo body world X displacement / exposure interval, evaluation only; instantaneous qvel also recorded','time_warning':'cloud/RGB are interval-average, Kalman is current filtered estimate; scores compare to same interval ground-truth displacement/time','real_time_claim':False,'total_wall_s':time.perf_counter()-begun,'total_frames':len(rows),'stats':{k:stats(k) for k in ('kalman_vx_m_s','cloud_vx_m_s','rgb_vx_m_s')},'common_frame_stats':common_stats,'rows':rows}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');csvpath=args.output.with_suffix('.csv')
        if rows:
            with csvpath.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False),flush=True)
    finally:env.close()
if __name__=='__main__':main()

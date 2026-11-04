"""Offline MuJoCo sensor-to-measured-session smoke; no real devices or commands."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml
from experiments.graspgenx_sim.run import config_overrides
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation
from experiments.graspgenx_sim.measured_frame import MeasuredFrameBuilder
from mujoco_conveyor_sorting.runtime.types import TargetIdentity
from nero_core.measured_motion import MeasuredMotionSession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=12.)
    args=parser.parse_args()
    cfgpath=Path(__file__).resolve().parents[1]/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml'
    cfg=yaml.safe_load(cfgpath.read_text())
    cfg['experiment']['seed']=1543
    cfg['experiment']['belt_speed_m_s']=.1
    env=None; started=time.monotonic(); result={'hardware_connections':0,'object_truth_used_for_input':False,
        'network_executed':False,'planning_executed':False,'status':'starting'}
    try:
        env=Nero7FixedSimulation(capture_path=cfgpath.parent/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1],
            config_overrides=config_overrides(cfg),grasp_assist=False,perception_mode='rendered_rgbd',
            grasp_event_source='physical_contacts',seed=1543,action_type='joint_position',preserve_control_profile=True)
        env.reset(seed=1543,options={'belt_speed':.1,'initial_objects':1})
        builder=MeasuredFrameBuilder(); last=-1
        while env.data.time<args.seconds:
            env.step(None)
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence']
                for task_id,entry in meta['measurements'].items():
                    if not entry['velocity_measurement'].get('speed_valid'):continue
                    ident=TargetIdentity('offline','part','task',float(frame['timestamp']),int(meta['clock_epoch']),
                        track_id=entry['actual_peer_track_id'],track_generation=entry['track_generation'])
                    region=env.nero_surface.region_metadata();region['source']='declared_simulation_belt_geometry'
                    try:
                        obs=builder.build(frame,task_id=task_id,identity=ident,world_to_base=env.nero_world_to_base,belt_region=region)
                        session=MeasuredMotionSession(ident,max_coast_s=1.)
                        accepted=session.update(obs.detection,identity=ident,timestamp=obs.capture_s,sequence=obs.sequence,
                            clock_epoch=obs.clock_epoch,source_timestamp=obs.capture_s,velocity_measurement=obs.velocity_measurement)
                        snap=session.snapshot(float(env.data.time))
                        if not accepted or not snap.valid:continue
                        result.update(status='measured_frame_accepted',sim_s=float(env.data.time),capture_s=obs.capture_s,
                            sequence=obs.sequence,clock_epoch=obs.clock_epoch,target_points=len(obs.detection['object_points']),
                            scene_points=len(obs.detection['scene_points']),velocity_base_m_s=snap.velocity.tolist(),
                            velocity_source=snap.velocity_source,observation_age_s=snap.observation_age_s,
                            target_height_m=obs.detection['height'],target_width_m=obs.detection['width'],
                            projection_cache_rebuilds=builder.projector.cache_rebuilds,
                            scene_robot_feedback=obs.detection['scene_state'])
                        break
                    except (ValueError,KeyError) as error:
                        result['last_rejection']=str(error)
                if result['status']=='measured_frame_accepted':break
            if result['status']=='measured_frame_accepted':break
            # Let the real asynchronous Nero7 process run; never fabricate a report.
            time.sleep(.002)
        if result['status']=='starting':result['status']='no_qualified_frame'
    finally:
        if env is not None:env.close()
        result['wall_s']=time.monotonic()-started
        def serial(x):
            if isinstance(x,np.ndarray):return x.tolist()
            if isinstance(x,np.generic):return x.item()
            raise TypeError(type(x).__name__)
        args.output.write_text(json.dumps(result,default=serial,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,default=serial,ensure_ascii=False),flush=True)
    return 0 if result['status']=='measured_frame_accepted' else 1


if __name__=='__main__':raise SystemExit(main())

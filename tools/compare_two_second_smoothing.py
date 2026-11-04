"""Saved-observation kinematic comparison only; no SDK or hardware imports."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.reference import Intercept, predict_entry_time
from nero_core.two_second.trajectory import certify_curve, extrema, sampling_times
from nero_core.two_second.diagnostics import json_value
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from experiments.two_second_grasp.smooth_reference import SmoothDiagnosticReference


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--durations',nargs='+',type=float,default=[2.,3.,4.,5.])
    args=p.parse_args(argv)
    if any(not np.isfinite(t) or t <= 0 for t in args.durations): p.error('durations must be positive')
    source=args.run/'cycle_001/events.json'
    event=next(e['data'] for e in json.loads(source.read_text()) if e['event']=='planning_submitted')
    cfg=CellConfig.load(args.config)
    begin=time.perf_counter();model=calibrated_model(cfg)
    startup=time.perf_counter()-begin
    initial=np.asarray(event['start_joints_rad']);model.set_state(initial,.1)
    start=model.tcp().copy()
    velocity=np.asarray(event['velocity_base_m_s'])
    entry=predict_entry_time(event['capture_monotonic_s'],
        np.asarray(event['entry_observed_position_base_m'])@cfg.direction,velocity@cfg.direction,
        cfg.follow_bounds[0],event['submission_monotonic_s'],cfg.raw['planning_reserve_s'],
        maximum_upstream_distance_m=cfg.raw['freeze_upstream_distance_m'])
    poses=sorted(map(np.array,event['candidate_tcp_base']),
        key=lambda g:Rotation.from_matrix(start[:3,:3].T@g[:3,:3]).magnitude())
    original=model.ik.config; rows=[]
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[('legacy',2.,0)]+[('smooth_diagnostic',t,i) for t in args.durations for i in range(len(poses))]
    for name,duration,index in cases:
        begun=time.perf_counter();g=poses[index]
        model.ik.config=(original if name=='legacy' else replace(original,
            position_tolerance=1e-8,orientation_tolerance=1e-7))
        arguments=(event['capture_monotonic_s'],entry,duration,g[:3,3],velocity,start[:3,3],
            Rotation.from_matrix(start[:3,:3]).as_quat()[[3,0,1,2]],
            Rotation.from_matrix(g[:3,:3]).as_quat()[[3,0,1,2]])
        ref=(Intercept(*arguments) if name=='legacy' else SmoothDiagnosticReference(*arguments))
        total=duration+.6 if name=='legacy' else ref.full_duration_s
        times=sampling_times(total,.01,duration,duration+.6)
        qs=[initial.copy()]
        row={'profile':name,'approach_s':duration,'total_s':total,'candidate':index+1,
             'geometry_certified':False,'hardware_connections':0,'commands_sent':0}
        follow_coordinates=np.array([ref.contact_xyz@cfg.direction,
            (ref.contact_xyz+velocity*.6)@cfg.direction])
        row['contact_and_follow_inside_grasp_interval']=bool(np.all(
            (follow_coordinates>=cfg.follow_bounds[0]) & (follow_coordinates<=cfg.follow_bounds[1])))
        xyz_samples=np.array([ref.sample(t)[0] for t in times])
        row['sampled_tcp_inside_workspace']=bool(np.all(
            (xyz_samples>=cfg.workspace[:,0]) & (xyz_samples<=cfg.workspace[:,1])))
        for t in times[1:]:
            xyz,quat=ref.sample(t)
            solved=model.solve_pose(xyz,quat,qs[-1])
            if not solved.success:
                row.update(status='ik_rejected',trajectory_time_s=float(t),message=solved.message)
                break
            qs.append(solved.joint_positions.copy())
        else:
            qs=np.asarray(qs)
            curve=CubicSpline(times,qs,bc_type=((1,np.zeros(7)),(1,np.zeros(7))))
            row.update(peak_velocity_rad_s=np.max(abs(np.asarray(extrema(curve,1))),axis=0).tolist(),
                peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(curve,2))),axis=0).tolist(),
                rest_to_rest_acceleration_duration_lower_bound_s=float(np.max(
                    2*np.sqrt(abs(qs[-1]-qs[0])/cfg.acceleration_limits))))
            sample=np.linspace(0,total,int(np.ceil(total*200))+1)
            position_error=orientation_error=0.
            for t in sample:
                model.set_state(curve(t),.1)
                actual=model.tcp(); xyz,quat=ref.sample(t)
                position_error=max(position_error,float(np.linalg.norm(actual[:3,3]-xyz)))
                wanted=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                orientation_error=max(orientation_error,float(Rotation.from_matrix(wanted.T@actual[:3,:3]).magnitude()))
            row.update(max_sampled_fk_position_error_m=position_error,max_sampled_fk_orientation_error_rad=orientation_error)
            try:
                certify_curve(times,qs,cfg,model.command_lower,model.command_upper,
                    identity=tuple(event['identity']),version=1,stage='offline_diagnostic')
                row['status']='joint_limits_passed'
            except PickRejected as error:
                row.update(status='joint_limits_rejected',error=str(error),details=error.details)
            np.savez_compressed(args.output/f'{name}_{duration:g}s_candidate_{index+1}.npz',times=times,joints=qs)
        row['solve_and_check_wall_s']=time.perf_counter()-begun
        rows.append(row)
        print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    report={'mode':'offline_kinematics_only','source_run':str(args.run),
        'saved_events_sha256':sha256(source.read_bytes()).hexdigest(),'config_sha256':cfg.source_sha256,
        'hardware_connections':0,'commands_sent':0,'model_startup_wall_s':startup,
        'acceleration_limits_rad_s2':cfg.acceleration_limits,'velocity_limits_rad_s':cfg.velocity_limits,
        'limitations':['no saved scene cloud: collision certification unavailable',
            'kinematic reconstruction from saved task data, not an exact original-run replay',
            'smooth diagnostic changes spatial approach and adds post-grip braking; not deployed',
            'joint-limit pass does not prove grasp, handoff, deadline or physical success'], 'cases':rows}
    (args.output/'report.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

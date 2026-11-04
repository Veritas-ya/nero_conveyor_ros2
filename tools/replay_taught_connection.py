"""Reconstruct a recorded taught connection offline; no device or commands."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree

from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.taught import TaughtSegment, model_digest
from mujoco_conveyor_sorting.perception import deproject_depth_to_world
from mujoco_conveyor_sorting.runtime.scene_region_filter import voxel_downsample
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--transfer',type=Path,required=True)
    parser.add_argument('--cycle',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scan-waypoints',action='store_true',help='诊断示教采样点；不认证相邻采样之间的路径')
    args=parser.parse_args()
    cfg=CellConfig.load(args.config)
    taught=TaughtSegment.load(args.transfer,cfg,'transfer')
    journal=[json.loads(line) for line in (args.run/'events.jsonl').open()]
    cycle=json.loads((args.run/f'cycle_{args.cycle:03d}'/'events.json').read_text())
    def event(name):return next(x['data'] for x in cycle if x['event']==name)
    submitted=event('planning_submitted')
    attachment=event('payload_estimate_attached')
    confirmed=event('grip_confirmed_from_width_force')
    trace_path=args.run/f'cycle_{args.cycle:03d}'/'execution_trace.npz'
    with np.load(trace_path,allow_pickle=False) as trace:
        stamps=trace['feedback_joint_sample_monotonic_s']
        grip_index=int(np.nanargmin(abs(stamps-attachment['joint_monotonic_s'])))
        grip_q=trace['feedback_q_rad'][grip_index].copy()
        grip_stamp=float(stamps[grip_index])
        lift_indices=np.flatnonzero(trace['sample_stage']=='lift')
        if not len(lift_indices):raise ValueError('记录中没有已发送的抬升轨迹')
        start_q=trace['dispatched_q_rad'][lift_indices[-1]].copy()
    with np.load(args.run/'preview/target.npz',allow_pickle=False) as frozen:
        points=frozen['points'].copy();depth=frozen['frame_depth_m'].copy()
        metadata=json.loads(str(frozen['metadata']))
    if tuple(metadata['identity']) != tuple(submitted['identity']):
        raise ValueError('冻结点云与回放任务身份不匹配')
    report=next(s for row in journal if (s:=row.get('data',{}).get('planner',{}).get('collision_scene'))
        and s.get('frame_sequence')==metadata['header']['sequence'] and s.get('plane_to_base'))
    planner=MeasuredPlanner(cfg);planner.identity=tuple(submitted['identity'])
    planner.plane=np.asarray(report['plane_to_base']);planner.support=planner.plane[:3,2].copy()
    planner.support_origin=planner.plane[:3,3].copy();planner.belt_bounds=np.asarray(report['bounds_xy_m'])
    planner.scene_report=report
    width=float(confirmed['width_m'])
    planner.model.set_state(grip_q,width);tcp=planner.model.tcp().copy()
    velocity=np.asarray(submitted['velocity_base_m_s']);capture=float(submitted['capture_monotonic_s'])
    planner.payload_local=(points+velocity*(grip_stamp-capture)-tcp[:3,3])@tcp[:3,:3]
    planner.payload_tree=cKDTree(planner.payload_local);planner.capture_tcp=tcp
    planner.target=SimpleNamespace(points=points,velocity=velocity,capture_s=capture,
        identity=planner.identity,sequence=metadata['header']['sequence'],epoch=metadata['header']['clock_epoch'])
    planner.target_tree=cKDTree(points)
    intrinsic=metadata['depth_intrinsics']
    matrix=np.array([[intrinsic['fx'],0,intrinsic['cx']],[0,intrinsic['fy'],intrinsic['cy']],[0,0,1]])
    xyz=deproject_depth_to_world(depth,matrix,np.asarray(metadata['depth_to_base_cv']),
                                 camera_convention='opencv').reshape(-1,3)
    valid=(depth.reshape(-1)>0)&np.isfinite(xyz).all(axis=1)
    xyz=xyz[valid]; local=(xyz-planner.support_origin)@planner.plane[:3,:3]
    inside=((local[:,:2]>=planner.belt_bounds[:,0])&(local[:,:2]<=planner.belt_bounds[:,1])).all(axis=1)
    belt=xyz[inside & (abs(local[:,2])<.005)]
    sampled=voxel_downsample(belt,voxel_size_m=planner.model.belt_scene_filter_config.voxel_size_m,
                             max_error_m=.004)
    planner.model.scene=belt[sampled.retained_indices];planner.model.tree=cKDTree(planner.model.scene)
    summary=json.loads((args.run/'summary.json').read_text())
    saved_hash=next((value for name,value in summary.get('source_hashes',{}).items()
                     if name.endswith('mujoco_conveyor_sorting/models/robot.xml')),None)
    result={'scope':'offline_reconstructed_approximation_not_original_scene_certificate',
        'hardware_connections':0,'commands_sent':0,'run':str(args.run.resolve()),'cycle':args.cycle,
        'taught_segment':taught.summary(), 'original_model_sha256':saved_hash,
        'current_model_sha256':model_digest(), 'model_matches':saved_hash==model_digest(),
        'grip_joint_stamp_difference_s':grip_stamp-float(attachment['joint_monotonic_s']),
        'width_source':'post_intercept_confirmed_width_not_saved_at_grip_timestamp',
        'start_source':'last_dispatched_lift_target_not_measured_feedback',
        'recorded_native_belt_points':report.get('native_belt_points'),
        'reconstructed_native_belt_points':len(belt),
        'recorded_collision_points':report.get('collision_points'),
        'reconstructed_collision_points':len(planner.model.scene),
        'limitation':'saved depth reprojected with saved plane; original target/rejected masks are not saved; '
                     'payload is frozen-velocity estimate, not a measured attached cloud'}
    state={'joints':start_q,'width_m':width}
    try:
        path=planner.route(state,stage='transfer_connect',goal_joints=taught.joints[0],
                           payload=True,start_joints=start_q)
        result.update(outcome='route_delivered',route_diagnostics=path.diagnostics['route_diagnostics'])
    except PickRejected as error:
        result.update(outcome='route_rejected',error=str(error),details=error.details)
    if args.scan_waypoints:
        rejected=[]
        for index,q in enumerate(taught.joints):
            try:
                planner.valid(q,width,payload=True)
            except PickRejected as error:
                rejected.append({'index':index,'recorded_elapsed_s':float(taught.times[index]),
                    'reason':str(error),'kind':error.details.get('kind')})
        bad={x['index'] for x in rejected}
        first_good=next((i for i in range(len(taught.times)) if i not in bad),None)
        last_bad=rejected[-1] if rejected else None
        result['waypoint_scan']={'checked_samples':len(taught.times),'rejected_samples':len(rejected),
            'first_rejected':rejected[0] if rejected else None,'last_rejected':last_bad,
            'first_passing':None if first_good is None else {
                'index':first_good,'recorded_elapsed_s':float(taught.times[first_good])},
            'limitation':'recorded samples only; no proof for interpolated or connection trajectory'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(json_value(result),file,ensure_ascii=False,indent=2);file.write('\n')
    print(json.dumps(json_value(result),ensure_ascii=False))


if __name__=='__main__':main()

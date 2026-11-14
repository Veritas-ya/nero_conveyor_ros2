#!/usr/bin/env python3
"""Compare taught and direct paths near recorded drop; no device access."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.taught import TaughtSegment, prepare_taught_timing
from nero_core.two_second.taught_timing import limit_jerk


def sample_path(planner,path,width,drop_xyz,*,hz,include_profile=False):
    count = int(np.ceil(path.duration*hz))+1
    times = np.linspace(0,path.duration,count)
    xyz, gripper_z, payload_z = [], [], []
    for t in times:
        planner.model.set_state(path.position(t),width)
        tcp=planner.model.tcp()
        xyz.append(tcp[:3,3].copy())
        values=[]
        for geom in planner.model.gripper_geometries:
            rotation=planner.model.data.geom_xmat[geom].reshape(3,3)
            position=planner.model.data.geom_xpos[geom]
            values.append(float(np.min((planner.model.mesh_vertices[geom]@rotation.T+position)[:,2])))
        gripper_z.append(min(values))
        payload_z.append(float(np.min((planner.payload_local@tcp[:3,:3].T+tcp[:3,3])[:,2])))
    xyz=np.asarray(xyz);gripper_z=np.asarray(gripper_z);payload_z=np.asarray(payload_z)
    distance=np.linalg.norm(xyz[:,:2]-drop_xyz[:2],axis=1)
    def region(radius):
        ids=np.flatnonzero(distance<=radius)
        if not len(ids):return None
        start=ids[0]
        return {'sample_count':len(ids),'first_time_s':float(times[start]),
            'first_tcp_xyz_m':xyz[start].tolist(),'first_radius_m':float(distance[start]),
            'minimum_tcp_z_m':float(xyz[ids,2].min()),'maximum_tcp_z_m':float(xyz[ids,2].max()),
            'minimum_gripper_z_m':float(gripper_z[ids].min()),
            'minimum_estimated_payload_z_m':float(payload_z[ids].min())}
    result = {'duration_s':path.duration,'sample_hz':hz,'radius_0p20_m':region(.2),
            'radius_0p10_m':region(.1),'end_tcp_xyz_m':xyz[-1].tolist(),
            'end_gripper_min_z_m':float(gripper_z[-1]),'end_payload_min_z_m':float(payload_z[-1])}
    if include_profile:
        result['profile']={'times_s':times.tolist(),'horizontal_radius_m':distance.tolist(),
                           'tcp_z_m':xyz[:,2].tolist(),'gripper_lowest_z_m':gripper_z.tolist(),
                           'payload_lowest_z_m':payload_z.tolist()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--hz',type=float,default=100.)
    p.add_argument('--include-profile',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); cfg=CellConfig.load(args.config)
    taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    out=[]
    for source in args.inputs:
        planner=MeasuredPlanner(cfg);planner.identity=(0,1,len(out));planner.scene_revision=len(out)+1
        with np.load(source,allow_pickle=False) as d:
            planner.model.scene=d['scene'].copy();planner.model.tree=cKDTree(planner.model.scene)
            planner.plane=d['plane'].copy();planner.belt_bounds=d['belt_bounds'].copy()
            planner.support_origin=d['support_origin'].copy();planner.support=planner.plane[:3,2].copy()
            planner.payload_local=d['payload_local'].copy();planner.payload_tree=cKDTree(planner.payload_local)
            planner.capture_tcp=d['capture_tcp'].copy()
            start=d['start_q'].copy();width=float(d['width_m'])
        planner.scene_report={'scope':'measured_belt_surface_only'}
        planner.model.set_state(taught.joints[-1],width); drop_xyz=planner.model.tcp()[:3,3].copy()
        row={'input':str(source),'drop_tcp_base_m':drop_xyz.tolist(),
             'configured_bin_tcp_base_m':cfg.bins[cfg.default_bin][:3,3].tolist()}
        state=nominal_state(planner.model,start,width)
        try:
            path=planner.route(state,stage='transfer',goal_joints=taught.joints[-1],payload=True,start_joints=start)
            path=limit_jerk(path,cfg.raw.get('taught_joint_jerk_limit_rad_s3'),cfg.settings.control_hz)
            planner.check_path(path,width,payload=True)
            taught_path,_=prepare_taught_timing(cfg,taught,planner.model,payload=True)
            row['direct']=sample_path(planner,path,width,drop_xyz,hz=args.hz,include_profile=args.include_profile)
            row['taught']=sample_path(planner,taught_path,width,drop_xyz,hz=args.hz,include_profile=args.include_profile)
            row['status']='sampled'
        except PickRejected as error:
            row.update(status='rejected',error=str(error),details=error.details)
        out.append(row)
        print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    result={'coordinate_system':'robot_base, metres; horizontal XY radius',
        'hardware_connections':0,'commands_sent':0,
        'limits':'finite sampling, modeled TCP and robot geometry, estimated payload; '
                 'historic belt scene, no bin geometry or measured rim height; '
                 'not continuous clearance certification or physical proof',
        'source_taught_sha256':taught.sha256,'samples':out}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(json_value(result),f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()

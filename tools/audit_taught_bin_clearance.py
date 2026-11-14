#!/usr/bin/env python3
"""Offline, approximate box-rim audit from four user-supplied joint snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.taught import TaughtSegment, prepare_taught_timing
from nero_core.two_second.taught_timing import limit_jerk


# Transcribed in image order. GUI XYZ is flange; the user aligned the midpoint
# of the fingertips with each corner. Model TCP and fingertip sites are nominal.
CORNER_JOINTS_DEG = np.array([
    [-74.078,-75.105,114.079,-10.683,66.121,25.438,97.559],
    [15.913,-36.754,99.709,102.309,27.839,-6.849,70.408],
    [-84.899,-63.222,125.655,26.387,52.790,2.148,95.290],
    [-15.853,-11.440,83.334,121.045,11.455,-6.977,73.320]])


def nominal_corners(model):
    left=model.model.body('gripper_link1').id;right=model.model.body('gripper_link2').id
    corners=[]
    for q in CORNER_JOINTS_DEG:
        model.set_state(np.deg2rad(q),.1)
        corners.append((model.data.xpos[left]+model.data.xpos[right]).copy()/2.)
    return np.array(corners)


def audit(planner,path,width,drop,corners,*,hz):
    hull=ConvexHull(corners[:,:2]); equations=hull.equations
    # Highest recorded corner avoids averaging away the observed height spread;
    # it is a nominal comparison plane, NOT a measured safety margin.
    level=float(np.max(corners[:,2])); times=np.linspace(0,path.duration,int(np.ceil(path.duration*hz))+1)
    count=0;first=None;max_depth=0.
    for stamp in times:
        planner.model.set_state(path.position(stamp),width);tcp=planner.model.tcp()
        if np.linalg.norm(tcp[:2,3]-drop[:2]) > .2: continue
        clouds=[]
        for geom in planner.model.gripper_geometries:
            R=planner.model.data.geom_xmat[geom].reshape(3,3);p=planner.model.data.geom_xpos[geom]
            clouds.append(planner.model.mesh_vertices[geom]@R.T+p)
        clouds.append(planner.payload_local@tcp[:3,:3].T+tcp[:3,3])
        xyz=np.concatenate(clouds); outside=(xyz[:,:2]@equations[:,:2].T+equations[:,2]).max(axis=1)>0
        below=xyz[:,2]<level;hits=np.flatnonzero(outside & below)
        if hits.size:
            count+=1;max_depth=max(max_depth,float(level-xyz[hits,2].min()))
            if first is None:
                i=hits[np.argmin(xyz[hits,2])]
                first={'time_s':float(stamp),'tcp_xyz_m':tcp[:3,3].tolist(),
                       'example_outside_below_point_m':xyz[i].tolist(),
                       'radius_m':float(np.linalg.norm(tcp[:2,3]-drop[:2]))}
    return {'sampled_hz':hz,'potential_conflict_frames':count,'first':first,
            'max_below_nominal_rim_m':max_depth}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--hz',type=float,default=100.);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=CellConfig.load(args.config);model=calibrated_model(cfg)
    corners=nominal_corners(model);taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    hull=ConvexHull(corners[:,:2]); rows=[]
    for source in args.inputs:
        planner=MeasuredPlanner(cfg);planner.identity=(0,1,len(rows));planner.scene_revision=len(rows)+1
        with np.load(source,allow_pickle=False) as d:
            planner.model.scene=d['scene'].copy();planner.model.tree=cKDTree(planner.model.scene)
            planner.plane=d['plane'].copy();planner.belt_bounds=d['belt_bounds'].copy()
            planner.support_origin=d['support_origin'].copy();planner.support=planner.plane[:3,2].copy()
            planner.payload_local=d['payload_local'].copy();planner.payload_tree=cKDTree(planner.payload_local)
            planner.capture_tcp=d['capture_tcp'].copy();start=d['start_q'].copy();width=float(d['width_m'])
        planner.scene_report={'scope':'measured_belt_surface_only'}
        planner.model.set_state(taught.joints[-1],width);drop=planner.model.tcp()[:3,3].copy()
        row={'input':str(source)}
        try:
            start_state=nominal_state(planner.model,start,width)
            direct=planner.route(start_state,stage='transfer',goal_joints=taught.joints[-1],payload=True,start_joints=start)
            direct=limit_jerk(direct,cfg.raw.get('taught_joint_jerk_limit_rad_s3'),cfg.settings.control_hz)
            tail,_=prepare_taught_timing(cfg,taught,planner.model,payload=True)
            row.update(status='sampled',direct=audit(planner,direct,width,drop,corners,hz=args.hz),
                       taught=audit(planner,tail,width,drop,corners,hz=args.hz))
        except PickRejected as e:row.update(status='rejected',error=str(e),details=e.details)
        rows.append(row);print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    result={'robot_base_fingertip_midpoints_nominal_m':corners.tolist(),
        'corners_hull_order_zero_based':hull.vertices.tolist(),
        'comparison_rim_z_m':float(max(corners[:,2])),
        'limitation':'nominal uncalibrated fingertip model, uncertain manually aligned corners, '
            'unknown rim thickness/margin; sampled vertices and estimated partial payload cloud only; '
            'not actual contact or a clearance certificate',
        'source':'user-provided four robot UI joint snapshots; fingertip midpoint alignment',
        'hardware_connections':0,'commands_sent':0,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(json_value(result),f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()

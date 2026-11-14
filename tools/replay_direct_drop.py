#!/usr/bin/env python3
"""Plan post-lift to recorded drop joints against saved scenes, without devices."""
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
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.taught_timing import limit_jerk
from nero_core.two_second.trajectory import extrema


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cfg=CellConfig.load(args.config)
    taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    drop=taught.joints[-1]; rows=[]
    for source in args.inputs:
        planner=MeasuredPlanner(cfg);planner.identity=(0,1,len(rows));planner.scene_revision=len(rows)+1
        with np.load(source,allow_pickle=False) as data:
            planner.model.scene=data['scene'].copy();planner.model.tree=cKDTree(planner.model.scene)
            planner.plane=data['plane'].copy();planner.belt_bounds=data['belt_bounds'].copy()
            planner.support_origin=data['support_origin'].copy();planner.support=planner.plane[:3,2].copy()
            planner.payload_local=data['payload_local'].copy();planner.payload_tree=cKDTree(planner.payload_local)
            planner.capture_tcp=data['capture_tcp'].copy()
            start=data['start_q'].copy();width=float(data['width_m'])
        planner.scene_report={'scope':'measured_belt_surface_only'}
        initial=nominal_state(planner.model,start,width)
        row={'input':str(source),'status':None};began=time.monotonic()
        try:
            route=planner.route(initial,stage='transfer',goal_joints=drop,payload=True,start_joints=start)
            route=limit_jerk(route,cfg.raw.get('taught_joint_jerk_limit_rad_s3'),cfg.settings.control_hz)
            planner.check_path(route,width,payload=True)
            planner.validate_release_opening(nominal_state(planner.model,route.q1,width))
            row.update(status='path_certified_offline',planning_wall_s=time.monotonic()-began,
                execution_duration_s=route.duration,
                start_error_rad=float(np.max(abs(route.q0-start))),
                end_error_rad=float(np.max(abs(route.q1-drop))),
                peak_velocity_rad_s=np.max(abs(np.asarray(extrema(route.curve,1))),axis=0),
                peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(route.curve,2))),axis=0),
                peak_jerk_rad_s3=float(np.max(abs(np.asarray(extrema(route.curve,3))))),
                diagnostics=route.diagnostics)
        except PickRejected as error:
            row.update(status='rejected',error=str(error),details=error.details,
                       planning_wall_s=time.monotonic()-began)
        rows.append(row)
        print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    report={'config':str(args.config.resolve()),'goal':'recorded transfer endpoint joints',
        'start':'saved planned post-lift anchor, not live measured feedback',
        'scope':'historical belt point cloud and estimated attached payload; no bin geometry',
        'jerk_limit_rad_s3':cfg.raw.get('taught_joint_jerk_limit_rad_s3'),
        'hardware_connections':0,'commands_sent':0,'physical_motion_tested':False,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(json_value(report),file,ensure_ascii=False,indent=2);file.write('\n')
    return 0 if all(row['status']=='path_certified_offline' for row in rows) else 1


if __name__=='__main__':raise SystemExit(main())

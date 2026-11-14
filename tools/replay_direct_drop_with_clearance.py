#!/usr/bin/env python3
"""Offline height-constrained direct drop using user-provided corner samples."""
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
from nero_core.two_second.direct_drop_guard import load_guard
from analyze_drop_neighborhood import sample_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--guard',type=Path,default=Path('src/nero_bringup/config/calibration/bin_rim_nominal_20261114.yaml'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=CellConfig.load(args.config)
    taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    guard=load_guard(args.guard,cfg,taught);cfg.raw['experimental_direct_drop_guard']=guard
    rows=[]
    for source in args.inputs:
        planner=MeasuredPlanner(cfg);planner.identity=(0,1,len(rows));planner.scene_revision=len(rows)+1
        with np.load(source,allow_pickle=False) as d:
            planner.model.scene=d['scene'].copy();planner.model.tree=cKDTree(planner.model.scene)
            planner.plane=d['plane'].copy();planner.belt_bounds=d['belt_bounds'].copy()
            planner.support_origin=d['support_origin'].copy();planner.support=planner.plane[:3,2].copy()
            planner.payload_local=d['payload_local'].copy();planner.payload_tree=cKDTree(planner.payload_local)
            planner.capture_tcp=d['capture_tcp'].copy();start=d['start_q'].copy();width=float(d['width_m'])
        planner.scene_report={'scope':'measured_belt_surface_only'}
        planner.model.set_state(taught.joints[-1],width);goal_xyz=planner.model.tcp()[:3,3].copy()
        row={'input':str(source)};start_time=time.monotonic()
        try:
            state=nominal_state(planner.model,start,width)
            path=planner.route(state,stage='transfer',goal_joints=taught.joints[-1],payload=True,start_joints=start)
            path=limit_jerk(path,cfg.raw.get('taught_joint_jerk_limit_rad_s3'),cfg.settings.control_hz)
            planner.check_path(path,width,payload=True)
            planner.validate_release_opening(nominal_state(planner.model,path.q1,width))
            plan_elapsed=time.monotonic()-start_time
            sampling_started=time.monotonic()
            samples=sample_path(planner,path,width,goal_xyz,hz=100.)
            row.update(status='passed_sampled_model_check',planning_wall_s=plan_elapsed,
                diagnostic_sampling_wall_s=time.monotonic()-sampling_started,
                execution_duration_s=path.duration, checks=dict(planner.stats), sample=samples,
                peak_velocity_rad_s=np.max(abs(np.asarray(extrema(path.curve,1))),axis=0),
                peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(path.curve,2))),axis=0),
                peak_jerk_rad_s3=float(np.max(abs(np.asarray(extrema(path.curve,3))))))
        except PickRejected as error:
            row.update(status='rejected',error=str(error),details=error.details,checks=dict(planner.stats),
                       planning_wall_s=time.monotonic()-start_time)
        rows.append(row);print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    result={'scope':'offline production planner model-guard replay, no hardware',
        'guard':guard,
        'hardware_connections':0,'commands_sent':0,'physical_motion_tested':False,
        'limitation':'unverified nominal fingertip model and manual corners; partial observed payload; '
                     'sampled check not continuous or physical clearance guarantee', 'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:json.dump(json_value(result),f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()

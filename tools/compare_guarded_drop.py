#!/usr/bin/env python3
"""Paired offline comparison: guarded direct drop versus guarded taught entry."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.direct_drop_guard import load_guard
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.taught import TaughtSegment, prepare_taught_timing, certify_taught
from nero_core.two_second.taught_timing import limit_jerk


def planner_for(cfg,source,identity):
    p=MeasuredPlanner(cfg);p.identity=identity;p.scene_revision=1
    with np.load(source,allow_pickle=False) as d:
        p.model.scene=d['scene'].copy();p.model.tree=cKDTree(p.model.scene)
        p.plane=d['plane'].copy();p.belt_bounds=d['belt_bounds'].copy()
        p.support_origin=d['support_origin'].copy();p.support=p.plane[:3,2].copy()
        p.payload_local=d['payload_local'].copy();p.payload_tree=cKDTree(p.payload_local)
        p.capture_tcp=d['capture_tcp'].copy();start=d['start_q'].copy();width=float(d['width_m'])
    p.scene_report={'scope':'measured_belt_surface_only'}
    return p,start,width


def evaluate(cfg,taught,source,kind,identity):
    planner,start,width=planner_for(cfg,source,identity)
    state=nominal_state(planner.model,start,width);began=time.monotonic()
    result={'kind':kind}
    try:
        q_goal=taught.joints[-1] if kind=='direct' else taught.joints[0]
        route=planner.route(state,stage='direct_drop' if kind=='direct' else 'transfer_connect',
            goal_joints=q_goal,payload=True,start_joints=start)
        route=limit_jerk(route,cfg.raw.get('taught_joint_jerk_limit_rad_s3'),cfg.settings.control_hz)
        planner.check_path(route,width,payload=True)
        if kind=='taught':
            suffix=certify_taught(planner,cfg,taught,nominal_state(planner.model,route.q1,width),
                route.q1,payload=True,cancelled=lambda:False)
            end=suffix.q1;duration=route.duration+suffix.duration
        else:end=route.q1;duration=route.duration
        planner.validate_release_opening(nominal_state(planner.model,end,width))
        result.update(status='passed_sampled_model_check',motion_s=duration,
                      planning_s=time.monotonic()-began,checks=planner.stats)
    except PickRejected as error:
        result.update(status='rejected',error=str(error),details=error.details,
                      planning_s=time.monotonic()-began,checks=planner.stats)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--guard',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=CellConfig.load(args.config)
    taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    guard=load_guard(args.guard,cfg,taught);cfg.raw['experimental_direct_drop_guard']=guard
    t=time.monotonic();prepare_taught_timing(cfg,taught,calibrated_model(cfg),payload=True);warm=time.monotonic()-t
    rows=[]
    for index,source in enumerate(args.inputs):
        order=('direct','taught') if index%2==0 else ('taught','direct')
        results={kind:evaluate(cfg,taught,source,kind,(0,index,0)) for kind in order}
        rows.append({'input':str(source),'order':order,**results})
        print(json.dumps(json_value(rows[-1]),ensure_ascii=False),flush=True)
    out={'scope':'offline saved scene, estimated attached payload, nominal model clearance',
         'hardware_connections':0,'commands_sent':0,'physical_motion_tested':False,
         'taught_timing_warm_s':warm,'guard':guard,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:json.dump(json_value(out),f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()

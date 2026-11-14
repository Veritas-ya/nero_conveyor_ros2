#!/usr/bin/env python3
"""Offline study of alternative taught transfer entry points; never uses devices."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.taught import TaughtSegment, prepare_taught_timing, certify_taught
from nero_core.two_second.taught_timing import limit_jerk


def suffix(segment, index):
    if index < 0 or len(segment.times)-index < 4:
        raise ValueError('切入点须保留至少四个示教采样')
    if not index:
        return segment
    times = (segment.times[index:] - segment.times[index]).copy()
    joints = segment.joints[index:].copy()
    times.setflags(write=False); joints.setflags(write=False)
    digest = sha256(f'{segment.sha256}:suffix:{index}'.encode()).hexdigest()
    return TaughtSegment('transfer', times, joints, segment.path, digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0, 5, 10, 15, 20])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = CellConfig.load(args.config)
    segment = TaughtSegment.load(cfg.raw['taught_transfer_file'], cfg, 'transfer')
    entries = [suffix(segment, index) for index in args.indices]
    model = calibrated_model(cfg)
    prewarm = []
    for index, part in zip(args.indices, entries):
        begun = time.monotonic()
        try:
            path, _ = prepare_taught_timing(cfg, part, model, payload=True)
            prewarm.append({'index':index,'duration_s':path.duration,
                            'wall_s':time.monotonic()-begun,'status':'ready'})
        except PickRejected as error:
            prewarm.append({'index':index,'status':'rejected','error':str(error),
                            'details':error.details,'wall_s':time.monotonic()-begun})
    rows = []
    for source in args.inputs:
        planner = MeasuredPlanner(cfg)
        planner.identity = (0, 1, len(rows)); planner.scene_revision = len(rows)+1
        with np.load(source, allow_pickle=False) as d:
            planner.model.scene = d['scene'].copy(); planner.model.tree = cKDTree(planner.model.scene)
            planner.plane = d['plane'].copy(); planner.belt_bounds = d['belt_bounds'].copy()
            planner.support_origin = d['support_origin'].copy(); planner.support = planner.plane[:3,2].copy()
            planner.payload_local = d['payload_local'].copy(); planner.payload_tree = cKDTree(planner.payload_local)
            planner.capture_tcp = d['capture_tcp'].copy()
            start = d['start_q'].copy(); width = float(d['width_m'])
        planner.scene_report = {'scope':'measured_belt_surface_only'}
        checks = []
        for index, part, warm in zip(args.indices, entries, prewarm):
            row = {'index':index,'source_elapsed_s':float(segment.times[index]),'status':None}
            if warm['status'] != 'ready':
                row.update(status='timing_rejected',error=warm['error']); checks.append(row); continue
            began = time.monotonic()
            try:
                state = nominal_state(planner.model, start, width)
                route = planner.route(state, stage='transfer_connect', goal_joints=part.joints[0],
                    payload=True, start_joints=start)
                route = limit_jerk(route, cfg.raw.get('taught_joint_jerk_limit_rad_s3'),
                                   cfg.settings.control_hz)
                at_entry = nominal_state(planner.model, route.q1, width)
                tail = certify_taught(planner, cfg, part, at_entry, route.q1,
                    payload=True, cancelled=lambda:False)
                if np.max(abs(tail.q1-segment.joints[-1])) > 1e-9:
                    raise PickRejected('示教末端投放位发生改变')
                row.update(status='connection_and_suffix_certified',
                    connection_duration_s=route.duration, suffix_duration_s=tail.duration,
                    total_duration_s=route.duration+tail.duration,
                    planning_wall_s=time.monotonic()-began)
            except PickRejected as error:
                row.update(status='rejected', error=str(error), details=error.details,
                           planning_wall_s=time.monotonic()-began)
            checks.append(row)
        rows.append({'input':str(source),'width_m':width,'candidates':checks})
        print(json.dumps(json_value(rows[-1]),ensure_ascii=False),flush=True)
    report = {'scope':'historical recorded scene and estimated attached payload; suffix reinterpolated',
              'config':str(args.config.resolve()),'transfer':segment.summary(), 'prewarm':prewarm,
              'hardware_connections':0,'commands_sent':0,'physical_motion_tested':False,
              'limitations':'saved start is last planned post-lift joint anchor, not current robot feedback; '
                            'not the current environment; box geometry is not installed; '
                            'same endpoint does not prove physical placement or collision-free bin entry',
              'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(json_value(report),file,ensure_ascii=False,indent=2); file.write('\n')


if __name__=='__main__': main()

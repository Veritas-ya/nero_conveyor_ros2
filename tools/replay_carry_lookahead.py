#!/usr/bin/env python3
"""Replay saved exposures/feedback through carry lookahead; never open devices."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial import cKDTree

from nero_core.two_second.config import CellConfig
from nero_core.two_second.carry_lookahead import CarryLookahead, revalidate_cycle
from nero_core.two_second.direct_drop_guard import load_guard
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.prepared_cycle import nominal_state, prepare_cycle
from nero_core.two_second.synchronized_reference import SynchronizedIntercept
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.trajectory import TimedPath
from nero_core.two_second.diagnostics import json_value


def load_case(run, config, cycle_index=1):
    records = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
    events = [e['data']['record'] for e in records if e['event']=='execution_event'
              and e['data']['cycle_index']==cycle_index]
    def event(name):
        found=next((e for e in events if e['event']==name),None)
        if found is None:
            raise ValueError(f'{run.name} cycle {cycle_index}: missing {name}')
        return found
    identity=tuple(event('planning_delivery')['data']['identity'])
    def artifact(kind):
        matches=[]
        for file in sorted((run/'experiment').glob('*_'+kind+'.npz')):
            with np.load(file,allow_pickle=False) as d:
                if tuple(json.loads(d['metadata'].item())['identity'])==identity:
                    matches.append(file)
        if len(matches)!=1:
            raise ValueError(f'{run.name} cycle {cycle_index}: expected one {kind} for {identity}, got {len(matches)}')
        return matches[0]
    with np.load(artifact('target'),allow_pickle=False) as d:
        m=json.loads(d['metadata'].item()); detection=m['detection']
        detection.update(scene_points=d['scene_points'].copy(),scene_pixel_indices=d['scene_pixel_indices'].copy())
        target=NS(identity=tuple(m['identity']), points=d['points'].copy(),
            velocity=np.asarray(m['velocity']), capture_s=m['capture_s'], epoch=m['identity'][0],
            sequence=m['identity'][2],state=m['capture_state'],detection=detection)
    with np.load(artifact('plan'),allow_pickle=False) as d:
        m=json.loads(d['metadata'].item());ref=m['reference']
        reference=SynchronizedIntercept(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in ref.items()})
        path=TimedPath(PPoly(d['curve_c'].copy(),d['curve_x'].copy()),tuple(m['identity']),
            m['version'],m['stage'],m['duration_s'],m['planning_s'],m['diagnostics'])
    p=MeasuredPlanner(config)
    p.install(target,nominal_state(p.model,path.q0,config.settings.open_width_m))
    p.version=path.version;p.scene_revision=m['scene_revision']
    return p,path,reference,event


def attach_saved_feedback(p,run,event,cycle_index=1):
    attached=event('payload_estimate_attached')['data']['attachment_monotonic_s']
    with np.load(run/f'cycle_{cycle_index:03d}/execution_trace.npz',allow_pickle=False) as d:
        stamp=d['feedback_joint_sample_monotonic_s'];i=int(np.argmin(abs(stamp-attached)))
        if abs(stamp[i]-attached)>1e-6:raise ValueError('missing exact grip-time feedback')
        q=d['feedback_q_rad'][i].copy()
        # New runs store predicted q in route diagnostics. Always recover the
        # measured brake-end joints from the trace, not that predicted field.
        end=event('stage_finished')['monotonic_s']
        valid=(d['feedback_sample_stage']=='smooth_intercept_follow_brake') & (d['feedback_read_monotonic_s']<=end)
        indices=np.flatnonzero(valid)
        if not len(indices):raise ValueError('missing brake-end feedback')
        end_q=d['feedback_q_rad'][indices[-1]].copy()
    width=event('grip_confirmed_from_width_force')['data']['width_m']
    p.model.set_state(q,width);capture=p.model.tcp().copy()
    p.payload_local=(p.target_points_at(attached)-capture[:3,3])@capture[:3,:3]
    p.payload_tree=cKDTree(p.payload_local)
    state=nominal_state(p.model,end_q,width)
    p.capture_tcp=p.model.tcp().copy()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--guard',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--cycle-index',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=CellConfig.load(args.config)
    segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    cfg.raw['taught_segments']={'transfer':segment}
    cfg.raw['experimental_direct_drop_guard']=load_guard(args.guard,cfg,segment)
    began=time.monotonic();service=CarryLookahead(cfg);warm=time.monotonic()-began;rows=[]
    try:
        for run in args.runs:
            p,path,reference,event=load_case(run,cfg,args.cycle_index)
            token,future=service.submit(p,path,reference)
            result=future.result(timeout=45.)
            actual=attach_saved_feedback(p,run,event,args.cycle_index)
            row={'run':str(run),'cycle_index':args.cycle_index,'prediction_ok':result['ok'],
                'prediction_wall_s':result['planning_wall_s'],'prediction_error':result.get('error'),
                'historical_idle_planning_s':event('taught_cycle_prepared')['monotonic_s']-event('taught_cycle_preparing')['monotonic_s'],
                'available_overlap_s':event('taught_cycle_preparing')['monotonic_s']-event('planning_delivery')['monotonic_s']}
            try:
                bundle,evidence=revalidate_cycle(p,actual,path.q1,result,token=token,
                    source_version=path.version,cancelled=lambda:False,defer_return_validation=True)
                row.update(accepted=True,carry_motion_s=bundle.carry.duration,**evidence)
            except Exception as error:
                row.update(accepted=False,error=str(error),details=getattr(error,'details',{}))
            # Current serial implementation with the same recorded attachment.
            started=time.monotonic()
            baseline=prepare_cycle(p,cfg,actual,path.q1,cancelled=lambda:False)
            row.update(serial_replay_wall_s=time.monotonic()-started,serial_carry_motion_s=baseline.carry.duration)
            rows.append(row);print(json.dumps(json_value(row),ensure_ascii=False),flush=True)
    finally:service.close()
    out={'hardware_connections':0,'commands_sent':0,'physical_success':None,
         'startup_warm_s':warm,'scope':'saved scene and actual grip/end joints; flange reconstructed from model',
         'concurrent_hardware_dispatch_tested':False,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(json_value(out),f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()

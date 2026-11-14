#!/usr/bin/env python3
"""Compare full-chain carry with former local blends on recorded real inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from replay_carry_lookahead import load_case, attach_saved_feedback
from nero_core.two_second.config import CellConfig
from nero_core.two_second.direct_drop_guard import load_guard
from nero_core.two_second.prepared_cycle import prepare_cycle
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.trajectory import extrema


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--guard',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    args.output.mkdir(parents=True,exist_ok=False)
    for run in args.runs:
        row={'run':str(run)}
        for enabled in (False,True):
            cfg=CellConfig.load(args.config)
            segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
            cfg.raw.update(taught_segments={'transfer':segment},direct_drop_orientation_first=True,
                           direct_drop_continuous_carry=enabled)
            cfg.raw['experimental_direct_drop_guard']=load_guard(args.guard,cfg,segment)
            p,path,reference,event=load_case(run,cfg)
            state=attach_saved_feedback(p,run,event);began=time.monotonic()
            label='continuous' if enabled else 'previous'
            try:
                bundle=prepare_cycle(p,cfg,state,path.q1,cancelled=lambda:False)
                curve=bundle.carry;wall=time.monotonic()-began
                times=curve.diagnostics['protected_waypoint_times_s']
                peaks={str(d):np.max(abs(np.asarray(extrema(curve.curve,d))),axis=0).tolist()
                       for d in (1,2,3)}
                np.savez(args.output/(run.name+'_'+label+'.npz'),
                    curve_c=curve.curve.c,curve_x=curve.curve.x)
                row[label]={'accepted':True,'planning_s':wall,'duration_s':curve.duration,
                    'join_times_s':times,
                    'join_max_joint_speed_rad_s':np.max(abs(curve.curve(times,1)),axis=1).tolist(),
                    'peak_joint_derivatives':peaks,
                    'continuous_carry':curve.diagnostics.get('continuous_carry'),
                    'drop_orientation_schedule':curve.diagnostics['drop_orientation_schedule'],
                    'q0':curve.q0.tolist(),'q1':curve.q1.tolist()}
            except Exception as error:
                row[label]={'accepted':False,'error':str(error),'details':getattr(error,'details',{}),
                            'planning_s':time.monotonic()-began}
        rows.append(row)
        print(json.dumps({'run':run.name,**{k:{key:v for key,v in row[k].items() if key in (
            'accepted','error','planning_s','duration_s','join_max_joint_speed_rad_s')}
            for k in ('previous','continuous')}},ensure_ascii=False),flush=True)
    report={'hardware_connections':0,'commands_sent':0,'physical_success':None,
        'scope':'same saved RGB-D scene and actual grip/brake-end feedback; model flange reconstruction',
        'results':rows}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

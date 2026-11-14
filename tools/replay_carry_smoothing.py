#!/usr/bin/env python3
"""Compare carry curves using saved scenes/actual grip feedback, no device APIs."""
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
    for run in args.runs:
        row={'run':str(run)}
        for enabled in [False,True]:
            cfg=CellConfig.load(args.config)
            segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
            cfg.raw.update(taught_segments={'transfer':segment},direct_drop_blend_enabled=enabled)
            cfg.raw['experimental_direct_drop_guard']=load_guard(args.guard,cfg,segment)
            p,path,reference,event=load_case(run,cfg)
            actual=attach_saved_feedback(p,run,event)
            began=time.monotonic()
            try:
                bundle=prepare_cycle(p,cfg,actual,path.q1,cancelled=lambda:False)
                curve=bundle.carry;piece=curve.diagnostics['pieces'][1]['diagnostics']
                dt=1e-4
                jumps=[float(np.max(abs(curve.curve(t-dt,d)-curve.curve(t+dt,d))))
                    for t in curve.curve.x[1:-1] if dt<t<curve.duration-dt for d in [0,1,2]]
                row['blended' if enabled else 'baseline']={
                    'accepted':True,'planning_s':time.monotonic()-began,
                    'duration_s':curve.duration,'q0':curve.q0.tolist(),'q1':curve.q1.tolist(),
                    'peak_acceleration_rad_s2':np.max(abs(np.asarray(extrema(curve.curve,2))),axis=0).tolist(),
                    'peak_jerk_rad_s3':float(np.max(abs(np.asarray(extrema(curve.curve,3))))),
                    'lift_join':curve.diagnostics.get('corner_blend'),
                    'route_corners':piece.get('corner_blend'),
                    'route_waypoints':piece.get('waypoints'),
                    'max_qva_change_over_0_2ms':max(jumps,default=0.)}
            except Exception as error:
                row['blended' if enabled else 'baseline']={'accepted':False,'error':str(error),
                                                          'details':getattr(error,'details',{})}
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    result={'hardware_connections':0,'commands_sent':0,'physical_success':None,
            'scope':'recorded scene and actual grip/brake-end joints; model flange reconstruction',
            'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

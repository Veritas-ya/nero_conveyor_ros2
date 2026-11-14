#!/usr/bin/env python3
"""Saved-input comparison of late vs early drop orientation; no hardware access."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

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
            cfg.raw.update(taught_segments={'transfer':segment},direct_drop_orientation_first=enabled,
                           direct_drop_continuous_carry=False)
            cfg.raw['experimental_direct_drop_guard']=load_guard(args.guard,cfg,segment)
            p,path,reference,event=load_case(run,cfg)
            actual=attach_saved_feedback(p,run,event)
            began=time.monotonic()
            try:
                bundle=prepare_cycle(p,cfg,actual,path.q1,cancelled=lambda:False)
                wall=time.monotonic()-began;curve=bundle.carry
                p.model.set_state(curve.q1,actual['width_m']);goal=p.model.tcp().copy()
                angles=[];xyz=[]
                for t in np.linspace(0.,curve.duration,501):
                    p.model.set_state(curve.position(t),actual['width_m']);tcp=p.model.tcp()
                    xyz.append(float(np.linalg.norm(tcp[:2,3]-goal[:2,3])))
                    angles.append(float(np.rad2deg(Rotation.from_matrix(tcp[:3,:3].T@goal[:3,:3]).magnitude())))
                row['early' if enabled else 'previous']={
                    'accepted':True,'planning_s':wall,'duration_s':curve.duration,
                    'max_orientation_error_in_final_10cm_deg':max(a for a,d in zip(angles,xyz) if d<=.1),
                    'peak_acceleration_rad_s2':np.max(abs(np.asarray(extrema(curve.curve,2))),axis=0).tolist(),
                    'peak_jerk_rad_s3':float(np.max(abs(np.asarray(extrema(curve.curve,3))))),
                    'q0':curve.q0.tolist(),'q1':curve.q1.tolist(),
                    'schedule':curve.diagnostics.get('drop_orientation_schedule')}
            except Exception as error:
                row['early' if enabled else 'previous']={'accepted':False,'error':str(error),
                                                          'details':getattr(error,'details',{})}
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    result={'hardware_connections':0,'commands_sent':0,'physical_success':None,
            'scope':'recorded scene and grip/end joints; model flange reconstruction','results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

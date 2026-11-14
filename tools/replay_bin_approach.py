#!/usr/bin/env python3
"""Compare virtual-guide rounding on saved RGB-D and feedback; opens no devices."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import nero_core.two_second.continuous_carry as continuous
from nero_core.two_second.config import CellConfig
from nero_core.two_second.direct_drop_guard import load_guard
from nero_core.two_second.prepared_cycle import prepare_cycle
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.trajectory import extrema
from replay_carry_lookahead import load_case, attach_saved_feedback


def tcp_profile(model, curve, width):
    """Use the model Jacobian and analytic qdot, never a command difference."""
    ts=np.linspace(0.,curve.duration,int(np.ceil(curve.duration/.005))+1)
    jp=np.zeros((3,model.model.nv));jr=jp.copy()
    dofs=model.model.jnt_dofadr[model.ids]
    xyz=[];speed=[]
    for t in ts:
        model.set_state(curve.curve(t),width)
        model.mj.mj_jacSite(model.model,model.data,jp,jr,model.tcp_id)
        xyz.append(model.tcp()[:3,3].copy())
        speed.append(float(np.linalg.norm(jp[:,dofs]@curve.curve(t,1))))
    return ts,np.asarray(xyz),np.asarray(speed)


def replay(run, cycle, config, guard, output):
    row={'run':str(run),'cycle_index':cycle}
    checked=continuous.blend_checked_path
    def previous(planner,path,width,joins,**kwargs):
        return checked(planner,path,width,
            [(t,'outgoing' if mode=='rounded' else mode) for t,mode in joins],**kwargs)
    try:
        for label in ('previous_exact_guide','rounded_guide'):
            continuous.blend_checked_path=previous if label=='previous_exact_guide' else checked
            cfg=CellConfig.load(config)
            segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
            cfg.raw.update(taught_segments={'transfer':segment},direct_drop_orientation_first=True,
                           direct_drop_continuous_carry=True)
            cfg.raw['experimental_direct_drop_guard']=load_guard(guard,cfg,segment)
            p,path,reference,event=load_case(run,cfg,cycle)
            state=attach_saved_feedback(p,run,event,cycle)
            started=time.monotonic()
            try:
                bundle=prepare_cycle(p,cfg,state,path.q1,cancelled=lambda:False)
                planning=time.monotonic()-started;c=bundle.carry
                ts,xyz,v=tcp_profile(p.model,c,state['width_m'])
                r=np.linalg.norm(xyz[:,:2]-xyz[-1,:2],axis=1)
                # Diagnostic interval/threshold only. These do not gate motion.
                mask=(r>.10)&(r<.30)
                peaks={str(d):np.max(abs(np.asarray(extrema(c.curve,d))),axis=0).tolist()
                       for d in (1,2,3)}
                knots=c.curve.x[1:-1]
                jumps=[float(np.max(abs(c.curve(knots,d)-c.curve(np.nextafter(knots,-np.inf),d))))
                       for d in (0,1,2)]
                metrics={'accepted':True,'identity':list(c.identity),'planning_s':planning,
                    'duration_s':c.duration,'minimum_tcp_speed_in_region_m_s':float(np.min(v[mask])),
                    'time_below_0_1_m_s_in_region_s':float(np.count_nonzero(mask&(v<.1))*(ts[1]-ts[0])),
                    'peak_joint_derivatives':peaks,'max_qva_knot_jump':jumps,
                    'joint_velocity_limits_rad_s':cfg.velocity_limits.tolist(),
                    'joint_acceleration_limits_rad_s2':cfg.acceleration_limits.tolist(),
                    'max_final_orientation_error_rad':c.diagnostics['drop_orientation_schedule']['max_final_orientation_error_rad'],
                    'curve_diagnostics':c.diagnostics['corner_blend'],
                    'q0':c.q0.tolist(),'q1':c.q1.tolist()}
                np.savez(output/f'{run.name}_cycle{cycle}_{label}.npz',curve_c=c.curve.c,curve_x=c.curve.x,
                         time_s=ts,tcp_base_m=xyz,tcp_speed_m_s=v,radius_m=r)
                row[label]=metrics
            except Exception as error:
                row[label]={'accepted':False,'error':str(error),'details':getattr(error,'details',{})}
    finally:
        continuous.blend_checked_path=checked
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--guard',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--cycles',type=int,nargs='+',default=[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for run in args.runs:
        for cycle in args.cycles:
            row=replay(run,cycle,args.config,args.guard,args.output);rows.append(row)
            print(json.dumps({'run':run.name,'cycle':cycle,**{k:{key:v for key,v in row[k].items()
                if key in ('accepted','error','planning_s','duration_s','minimum_tcp_speed_in_region_m_s',
                           'time_below_0_1_m_s_in_region_s')}
                for k in ('previous_exact_guide','rounded_guide')}},ensure_ascii=False),flush=True)
    report={'hardware_connections':0,'commands_sent':0,'physical_success':None,
        'scope':'same-cycle saved scene/grip/brake feedback; model flange and Jacobian',
        'diagnostic_region':'horizontal distance to drop 0.10–0.30 m; speed <0.1 m/s for reporting only',
        'results':rows}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if any(not row[k]['accepted'] for row in rows for k in ('previous_exact_guide','rounded_guide')):
        raise SystemExit(1)


if __name__=='__main__':main()

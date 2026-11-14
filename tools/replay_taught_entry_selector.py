#!/usr/bin/env python3
"""Exercise the production entry selector with saved scenes, without devices."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.taught_entry import choose_entry, prepare_entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--indices',type=int,nargs='+',default=[0,5,10,15])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cfg=CellConfig.load(args.config); cfg.raw['taught_entry_indices']=args.indices
    taught=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    started=monotonic();entries=prepare_entries(cfg,taught,calibrated_model(cfg))
    prewarm=monotonic()-started;rows=[]
    for source in args.inputs:
        planner=MeasuredPlanner(cfg);planner.identity=(0,1,len(rows));planner.scene_revision=len(rows)+1
        with np.load(source,allow_pickle=False) as d:
            planner.model.scene=d['scene'].copy();planner.model.tree=cKDTree(planner.model.scene)
            planner.plane=d['plane'].copy();planner.belt_bounds=d['belt_bounds'].copy()
            planner.support_origin=d['support_origin'].copy();planner.support=planner.plane[:3,2].copy()
            planner.payload_local=d['payload_local'].copy();planner.payload_tree=cKDTree(planner.payload_local)
            planner.capture_tcp=d['capture_tcp'].copy()
            start=d['start_q'].copy();width=float(d['width_m'])
        planner.scene_report={'scope':'measured_belt_surface_only'}
        state=nominal_state(planner.model,start,width);began=monotonic()
        connect,transfer=choose_entry(planner,cfg,state,start,entries,cancelled=lambda:False)
        np.testing.assert_allclose(transfer.q1,taught.joints[-1],atol=1e-9)
        rows.append({'input':str(source),'selection_wall_s':monotonic()-began,
            'selected_connection_s':connect.duration,'selected_suffix_s':transfer.duration,
            'selection':connect.diagnostics['taught_entry_selection']})
        print(json.dumps(json_value(rows[-1]),ensure_ascii=False),flush=True)
    result={'hardware_connections':0,'commands_sent':0,'physical_motion_tested':False,
        'limitation':'historical scene and estimated payload, no bin geometry or live feedback',
        'prewarm_s':prewarm,'results':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(json_value(result),f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()

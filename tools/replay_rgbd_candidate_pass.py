#!/usr/bin/env python3
"""Replay camera-only RGB-D sequentially through the production meter; no devices."""
import argparse,json,time
from pathlib import Path
from collections import Counter
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane,green_belt_points
from nero_core.two_second.perception import make_meter
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=sorted((a.run/'rgbd_frames').glob('*.npz'));c=CellConfig.load(a.config);cal=load_runtime_depth_calibration(c.raw['camera_calibration_file']);frame,d2c=read(paths[0],cal);surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points);meter=make_meter(c,surface);rows=[];begin=time.monotonic()
 for i,path in enumerate(paths):
  frame,d2c=read(path,cal);report=meter.update(frame,d2c);peer=meter.latest_snapshot
  rows.append({'sequence':frame.header.sequence,'reason':None if peer is None else peer.reason,'current':False if peer is None else peer.current,'tracks':[] if peer is None else [{'id':t.track_id,'confirmed':t.confirmed,'samples':t.sample_count,'position_m':t.observed_position_base} for t in peer.tracks], 'diagnostics':None if report is None else report.get('diagnostics')})
  if i%150==0:print('processed',i,flush=True)
 result={'frames':len(rows),'reasons':dict(Counter(str(r['reason']) for r in rows)),'current_frames':sum(r['current'] for r in rows),'confirmed_track_frames':sum(any(t['confirmed'] for t in r['tracks']) and r['current'] for r in rows),'track_ids':sorted({t['id'] for r in rows if r['current'] for t in r['tracks'] if t['confirmed']}),'wall_s':time.monotonic()-begin,'robot_connections':0,'robot_commands':0,'note':'Offline production RGB-D meter and geometric admission; does not run camera, planner, or actuator.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'summary':result,'frames':rows},ensure_ascii=False,default=str));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

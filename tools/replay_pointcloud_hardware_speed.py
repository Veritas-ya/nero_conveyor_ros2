"""Offline replay of archived real RGB-D frames through both speed paths."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.perception import make_meter
from nero_core.two_second.velocity_source import KALMAN, POINT_KALMAN
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=40);args=p.parse_args()
    config=CellConfig.load(args.config);cal=load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    files=sorted((args.run/'rgbd_frames').glob('*.npz'))[:args.limit]
    if not files:raise ValueError('no recorded RGB-D frames')
    frame,d2c=read(files[0],cal)
    surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
    cfgs=[replace(config,raw={**config.raw,'velocity_source':source}) for source in (KALMAN,POINT_KALMAN)]
    meters=[make_meter(cfg,surface) for cfg in cfgs]
    rows=[]
    for path in files:
        frame,d2c=read(path,cal); items=[]
        for meter in meters:
            report=meter.update(frame,d2c);peer=meter.latest_snapshot
            items.append({'status':report['status'],'reason':report['reason'],
                'detections':report['diagnostics']['candidate_count'],
                'tracks':[{'id':t.track_id,'speed_valid':t.speed_valid,
                    'observed_position_base_m':t.observed_position_base,
                    'speed_position_base_m':t.speed_position_base_m,
                    'velocity_base_m_s':t.velocity_base_m_s} for t in peer.tracks],
                'processing_ms':report['diagnostics'].get('processing_ms')})
        rows.append({'sequence':frame.header.sequence,'source_time':frame.header.source_time,
                     'legacy':items[0],'pointcloud':items[1]})
    result={'source':str(args.run.resolve()),'camera_connections':0,'can_connections':0,'commands_sent':0,
        'archive_note':'Recorded archives save approximately every 15 frames; original timestamps are retained. '
                       'Sparse frames cannot establish continuous-track speed accuracy or justify synthetic timestamps.',
        'rows':rows,'pointcloud_frames_with_observation':sum(any(t['speed_position_base_m'] is not None
            for t in row['pointcloud']['tracks']) for row in rows),
        'pointcloud_frames_with_valid_speed':sum(any(t['speed_valid'] for t in row['pointcloud']['tracks']) for row in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_value(result),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

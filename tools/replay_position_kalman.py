"""Compare equal-parameter Kalman filters using two same-frame visual points."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.two_second_grasp.visual_point_kalman import VisualPointKalman


def replay(path):
    data=json.loads(path.read_text())
    frames=[s for s in data['visual_position_samples'] if s.get('camera_diagnostics')]
    if not frames:return {'source':str(path),'error':'no_frames'}
    first=frames[0];identity=(first.get('clock_epoch',0),first['track_id'],first['track_generation'])
    frames=[s for s in frames if (s.get('clock_epoch',0),s['track_id'],s['track_generation'])==identity]
    if data.get('capture_s') is not None:frames=[s for s in frames if s['frame_time_s']<=data['capture_s']]
    truth=[s for s in data['evaluation_trace'] if s['trial_index']==1 and s['object_position_world_m'] is not None]
    stamps=np.asarray([s['sim_s'] for s in truth])
    frames=[s for s in frames if len(stamps) and stamps[0]<=s['frame_time_s']<=stamps[-1]]
    if len(frames)<8:return {'source':str(path),'error':'insufficient_frames'}
    original=VisualPointKalman(.3);corrected=VisualPointKalman(.3)
    for s in frames:
        original.update(identity,s['frame_time_s'],s['position_world_m'])
        corrected.update(identity,s['frame_time_s'],s['camera_diagnostics']['point_cloud_percentile_box_center_world_m'])
    a=original.estimate(identity);b=corrected.estimate(identity)
    if a is None or b is None:return {'source':str(path),'error':'filter_not_qualified'}
    times=np.array([s['frame_time_s'] for s in frames]); pos=np.interp(times,stamps,[s['object_position_world_m'][0] for s in truth])
    truth_speed=float(np.polyfit(times-times[0],pos,1)[0]); old=float(a['velocity_world_m_s'][0]); new=float(b['velocity_world_m_s'][0])
    return {'source':str(path),'category_id':data['random_tless_category_sequence'][0],
            'samples':len(frames),'duration_s':float(times[-1]-times[0]),
            'truth_mean_m_s':truth_speed,'legacy_kalman_m_s':old,'corrected_kalman_m_s':new,
            'legacy_error_cm_s':(old-truth_speed)*100,'corrected_error_cm_s':(new-truth_speed)*100}


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    rows=[replay(path) for path in args.inputs];good=[r for r in rows if 'error' not in r]
    result={'method':'identical Kalman parameters on same RGB-D frames; simulation truth evaluation only',
            'results':rows,'mean_absolute_error_cm_s':{
                k:float(np.mean([abs(row[k]) for row in good])) if good else None
                for k in ('legacy_error_cm_s','corrected_error_cm_s')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

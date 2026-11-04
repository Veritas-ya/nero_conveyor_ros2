"""Summarize independent recorded runs; avoid adding overlapping durations."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);args=p.parse_args()
records=[]
for folder in args.runs:
    result=json.loads((folder/'result.json').read_text())
    files=sorted((folder/'inference').glob('track_*_try_*.json'))
    files=[f for f in files if not f.name.endswith('_grasp.json')]
    jobs=[json.loads(f.read_text()) for f in files]
    events=json.loads((folder/'policy_events.json').read_text())
    reports=json.loads((folder/'nero7_reports.json').read_text())
    velocities=[float(np.linalg.norm(t['velocity_base_m_s'][:2])) for report in reports
        if 6.3<=report['capture_sim_s']<=9.2
        for t in (report.get('peer') or {}).get('tracks',[])
        if t['track_id']==0 and t['speed_valid'] and t['velocity_base_m_s'] is not None]
    records.append({'directory':str(folder), 'model_load_s':result.get('model',{}).get('load_seconds'),
        'warmup_s':(result.get('model_warmup') or {}).get('wall_s'),
        'warmup_source':(result.get('model_warmup') or {}).get('source'),
        'timing':result.get('timing'), 'seed':result.get('config',{}).get('experiment',{}).get('seed'),
        'input_pose':result.get('spawn_manifest',[{}])[0].get('pose_world_wxyz'),
        'visual_track0_speed_median_m_s_6_3_to_9_2':float(np.median(velocities)) if velocities else None,
        'visual_speed_count':len(velocities), 'jobs':[{'capture_sim_s':j.get('capture_sim_s'),
        'ready_sim_s':j.get('ready_sim_s'),'input_points_sha256':j.get('metadata',{}).get('input_points_sha256'),
        'inference_s':j.get('metadata',{}).get('inference_seconds'),
        'candidate_filter_s':j.get('metadata',{}).get('candidate_filter_wall_s'),
        'accepted':len(j.get('accepted_candidates',[])), 'admission':j.get('admission')} for j in jobs],
        'events':[{'sim_s':e.get('timestamp'),'name':e.get('name'),'reason':e.get('reason')} for e in events],
        'evaluation':result.get('evaluation')})
print(json.dumps({'note':'Different live rendered exposures and independent wall-clock runs; timings are not a controlled A/B.', 'runs':records},indent=2))

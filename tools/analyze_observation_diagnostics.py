"""Analyze recorded camera detections against independently sampled MuJoCo truth."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path):
    d = json.loads(path.read_text())
    frames = [s for s in d['visual_position_samples'] if s.get('camera_diagnostics')]
    if not frames:
        return {'source': str(path), 'error': 'no_camera_diagnostics'}
    first = frames[0]
    identity = (first.get('clock_epoch', 0), first['track_id'], first['track_generation'])
    frames = [s for s in frames if (s.get('clock_epoch', 0),s['track_id'],s['track_generation']) == identity]
    freeze = d.get('capture_s')
    if freeze is not None:
        frames = [s for s in frames if s['frame_time_s'] <= freeze]
    truth = [s for s in d['evaluation_trace'] if s['object_position_world_m'] is not None and s['trial_index'] == 1]
    stamps = np.array([s['sim_s'] for s in truth])
    frames = [s for s in frames if len(stamps) and stamps[0] <= s['frame_time_s'] <= stamps[-1]]
    if len(frames) < 8:
        return {'source': str(path), 'error': 'insufficient_time_aligned_samples'}
    t = np.array([s['frame_time_s'] for s in frames]); x = t-t[0]
    slope = lambda z: float(np.polyfit(x, np.asarray(z), 1)[0])
    diag = [s['camera_diagnostics'] for s in frames]
    uv = np.array([s['pixel_center_uv'] for s in diag]); depth = np.array([s['median_depth_m'] for s in diag])
    K = np.asarray(d['observation_diagnostic_calibration']['intrinsics'])
    transform = np.asarray(d['observation_diagnostic_calibration']['camera_to_world_opencv'])
    ray = np.column_stack(((uv[:, 0]-K[0, 2])/K[0, 0], (uv[:, 1]-K[1, 2])/K[1, 1], np.ones(len(uv))))
    ray_x = ray @ transform[0, :3]
    ref_depth = float(np.median(depth))
    observed = np.array([s['position_world_m'][0] for s in frames])
    recalculated = np.array([s['deprojected_center_world_m'][0] for s in diag])
    actual = np.interp(t, stamps, [s['object_position_world_m'][0] for s in truth])
    candidates = {}
    for key in ('deprojected_center_world_m','point_cloud_mean_world_m',
                'point_cloud_median_world_m','point_cloud_percentile_box_center_world_m'):
        candidate = np.array([s[key][0] for s in diag])
        candidates[key] = {'velocity_cm_s': 100*slope(candidate),
                           'velocity_error_cm_s': 100*(slope(candidate)-slope(actual))}
    return {'source': str(path), 'category_id': d['random_tless_category_sequence'][0],
            'samples': len(frames), 'start_s': float(t[0]), 'end_s': float(t[-1]),
            'pixel_velocity_uv_px_s': [slope(uv[:, 0]),slope(uv[:, 1])],
            'median_depth_start_m': float(depth[0]), 'median_depth_end_m': float(depth[-1]),
            'median_depth_slope_mm_s': slope(depth)*1000,
            'reference_depth_m': ref_depth,
            'fixed_depth_pixel_component_cm_s': slope(ref_depth*ray_x)*100,
            'varying_depth_component_cm_s': slope((depth-ref_depth)*ray_x)*100,
            'detector_deprojection_max_difference_mm': float(np.max(np.abs(recalculated-observed))*1000),
            'ground_truth_velocity_cm_s': slope(actual)*100, 'candidates': candidates,
            'truth_scope': 'independent 0.5s simulation object samples interpolated at capture times; evaluation only'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',required=True,type=Path); args=p.parse_args()
    result={'analysis':'same-frame observed UV, median depth, and deprojected world center',
            'items':[analyze(path) for path in args.inputs]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

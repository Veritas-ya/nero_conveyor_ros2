"""Offline, causal velocity comparison on one saved MuJoCo visual trace.

The estimator uses camera positions only. Simulator positions are used after
estimation, solely as an approximate time-aligned evaluation reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(data, window_s=0.5):
    observations = data['visual_position_samples']
    kalman = {r['frame_sequence']: r for r in data['live_visual_speed_samples']}
    truth = [r for r in data['evaluation_trace'] if r.get('object_position_world_m') is not None
             and r.get('state') in ('NEW', 'TRACKING', 'READY_TO_PLAN')]
    if len(truth) < 2:
        raise ValueError('not enough independent pre-grasp simulator positions')
    truth_t = np.array([r['sim_s'] for r in truth], dtype=float)
    truth_x = np.array([r['object_position_world_m'][0] for r in truth], dtype=float)
    if np.any(np.diff(truth_t) <= 0):
        raise ValueError('simulator position times must be strictly increasing')
    rows = []
    for i, row in enumerate(observations):
        t = float(row['frame_time_s'])
        if t - 0.25 < truth_t[0] or t + 0.25 > truth_t[-1]:
            continue
        match = kalman.get(row['frame_sequence'])
        if match is None or match['track_id'] != row['track_id']:
            continue
        samples = [r for r in observations[:i + 1]
                   if r['track_id'] == row['track_id']
                   and r['track_generation'] == row['track_generation']
                   and t - window_s <= r['frame_time_s'] <= t]
        if len(samples) < 8 or samples[-1]['frame_time_s'] - samples[0]['frame_time_s'] < 0.3:
            continue
        times = np.array([r['frame_time_s'] for r in samples], dtype=float)
        xs = np.array([r['position_world_m'][0] for r in samples], dtype=float)
        centered = times - times.mean()
        fit = float(centered @ (xs - xs.mean()) / (centered @ centered))
        # Approximate average physical object velocity over a centered 0.5 s
        # interval using separately recorded positions, never estimator input.
        reference = float((np.interp(t + .25, truth_t, truth_x)
                           - np.interp(t - .25, truth_t, truth_x)) / .5)
        rows.append({'frame_time_s': t, 'track_id': row['track_id'],
                     'kalman_m_s': match['velocity_world_m_s'][0],
                     'window_fit_m_s': fit, 'reference_m_s': reference,
                     'fit_samples': len(samples)})
    summaries = []
    for start, end in ((0.75, 1.5), (1.5, 3.0), (3.0, 5.0)):
        subset = [r for r in rows if start <= r['frame_time_s'] < end]
        if not subset:
            continue
        item = {'interval_s': [start, end], 'matched_frames': len(subset)}
        for name in ('kalman', 'window_fit'):
            err = np.array([r[name + '_m_s'] - r['reference_m_s'] for r in subset])
            item[name + '_mean_error_cm_s'] = float(100 * np.mean(err))
            item[name + '_mae_cm_s'] = float(100 * np.mean(np.abs(err)))
        summaries.append(item)
    return {'estimator': 'causal unweighted rolling linear fit of observed x and capture time',
            'window_s': window_s, 'comparison_reference': 'centered 0.5 s simulation object position difference; interpolated from 0.5 s samples',
            'simulation_truth_used_for_estimation': False,
            'limitations': 'single object/run; reference is temporally smoothed; consecutive frames are correlated',
            'intervals': summaries, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-s', type=float, default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.window_s) or args.window_s < .3:
        parser.error('--window-s must be finite and at least 0.3 s')
    result = analyze(json.loads(args.trace.read_text()), args.window_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['intervals'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

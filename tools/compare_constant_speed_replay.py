"""Replay saved visual positions and independently evaluate simulation truth.

No truth or future samples enter either velocity estimate. Ground-truth object
positions are used only after freezing to calculate an evaluation metric.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.two_second_grasp.constant_visual_speed import ConstantVisualSpeed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = []
    for index, observation in enumerate(data['early_visual_orientation']['observations'], 1):
        capture = observation.get('final_capture_s')
        if capture is None or 'final_velocity_world_m_s' not in observation:
            continue
        identity = tuple(observation['identity'])
        fit = ConstantVisualSpeed()
        samples = sorted((s for s in data['visual_position_samples']
                          if s['track_id'] == identity[1] and s['track_generation'] == identity[2]
                          and s.get('clock_epoch', identity[0]) == identity[0]
                          and s['frame_time_s'] <= capture), key=lambda s: s['frame_time_s'])
        for sample in samples:
            fit.update(identity, sample['frame_time_s'], sample.get('speed_position_world_m', sample['position_world_m']))
        estimate = fit.estimate(identity)
        attempt = data['attempts'][index-1] if index <= len(data['attempts']) else data
        contact = (attempt.get('synchronized_schedule') or {}).get('contact_s')
        truth = [s for s in data['evaluation_trace'] if s['trial_index'] == index
                 and s['object_position_world_m'] is not None]
        if estimate is None or contact is None or not truth:
            continue
        # Keep the measurement before closing/contact and interpolate the
        # stored 0.5 s evaluator samples; never use truth for the estimate.
        end = contact - .6
        stamps = np.array([s['sim_s'] for s in truth])
        x = np.array([s['object_position_world_m'][0] for s in truth])
        if not stamps[0] <= capture < end <= stamps[-1]:
            continue
        delta = float(np.interp(end, stamps, x) - np.interp(capture, stamps, x))
        dt = end-capture
        comparison = attempt.get('frozen_speed_comparison') or {}
        kalman = float((comparison.get('kalman_world_m_s') or observation['final_velocity_world_m_s'])[0])
        fitted = float(estimate['velocity_world_m_s'][0])
        rows.append({'trial_index': index, 'capture_s': capture, 'evaluation_end_s': end,
                     'sample_count': estimate['sample_count'], 'span_s': estimate['span_s'],
                     'kalman_m_s': kalman, 'constant_fit_m_s': fitted,
                     'truth_average_m_s': delta/dt,
                     'kalman_velocity_only_error_mm': (kalman*dt-delta)*1000,
                     'fit_velocity_only_error_mm': (fitted*dt-delta)*1000,
                     'fit_residual_rms_xy_mm': (estimate['residual_rms_xy_m']*1000).tolist()})
    result = {'source': str(args.input), 'truth_used_for_estimation': False,
              'truth_scope': 'offline evaluation only; 0.5 s samples linearly interpolated',
              'trial_count': len(rows), 'trials': rows,
              'mean_absolute_velocity_only_error_mm': {
                  key: float(np.mean([abs(r[key]) for r in rows])) if rows else None
                  for key in ('kalman_velocity_only_error_mm','fit_velocity_only_error_mm')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

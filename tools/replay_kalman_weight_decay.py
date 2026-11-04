"""Offline, causal replay of simulation camera positions with Kalman Q variants."""
import argparse
import json
from pathlib import Path
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.kalman_motion import KalmanMotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    values = (.002, .001, .0005, 0.)
    errors = {str(v): [] for v in values}
    rows = []
    for index, observation in enumerate(data['early_visual_orientation']['observations'], 1):
        capture = observation.get('final_capture_s')
        if capture is None or index > len(data['attempts']):
            continue
        identity = tuple(observation['identity'])
        samples = sorted((s for s in data['visual_position_samples']
                          if s['track_id'] == identity[1] and s['track_generation'] == identity[2]
                          and s.get('clock_epoch', identity[0]) == identity[0]
                          and s['frame_time_s'] <= capture), key=lambda s: s['frame_time_s'])
        attempt = data['attempts'][index-1]
        contact = (attempt.get('synchronized_schedule') or {}).get('contact_s')
        comparison = attempt.get('frozen_speed_comparison') or {}
        if not samples or contact is None or comparison.get('kalman_world_m_s') is None:
            continue
        truth = [s for s in data['evaluation_trace'] if s['trial_index'] == index
                 and s['object_position_world_m'] is not None]
        end = contact - .6
        stamps = np.array([s['sim_s'] for s in truth])
        if not len(stamps) or not stamps[0] <= capture < end <= stamps[-1]:
            continue
        x = np.array([s['object_position_world_m'][0] for s in truth])
        dt = end - capture
        displacement = float(np.interp(end, stamps, x)-np.interp(capture, stamps, x))
        row = {'trial': index, 'samples': len(samples), 'capture_s': capture,
               'evaluation_end_s': end, 'truth_average_m_s': displacement/dt,
               'logged_kalman_world_x_m_s': comparison['kalman_world_m_s'][0], 'variants': {}}
        for q in values:
            k = KalmanMotion(samples[0]['position_world_m'][:2], samples[0]['frame_time_s'],
                             accel_noise_std=q, belt_direction=[1., 0.], max_velocity=.3)
            for sample in samples[1:]:
                k.update(sample['position_world_m'][:2], sample['frame_time_s'])
            speed = float(k.state[2])
            error = (speed*dt-displacement)*1000
            errors[str(q)].append(error)
            row['variants'][str(q)] = {'velocity_m_s': speed, 'velocity_only_error_mm': error,
                                       'rejected_count': k.rejected_count}
        rows.append(row)
    baseline_difference = max((abs(r['variants']['0.002']['velocity_m_s'] -
                                   r['logged_kalman_world_x_m_s']) for r in rows), default=None)
    result = {'input': str(args.input), 'truth_used_for_estimation': False,
              'scope': 'simulation-only replay; truth used solely for offline evaluation',
              'q_parameter': 'accel_noise_std m/s^(3/2), covariance density is its square',
              'reference_alignment_max_difference_m_s': baseline_difference,
              'evaluation': '0.5s physical position samples interpolated, until 0.6s before contact; excludes initial position and controller error',
              'trials': rows, 'mean_absolute_velocity_only_error_mm': {
                  q: float(np.mean(np.abs(v))) if v else None for q, v in errors.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

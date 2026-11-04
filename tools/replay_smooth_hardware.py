#!/usr/bin/env python3
"""Reconstruct saved measured-task kinematics; never open a device or certify a scene."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.precise_ik import PrecisePoseSolver
from nero_core.two_second.reference import Intercept, predict_entry_time
from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.trajectory import joint_curve, sampling_times, extrema, certify_curve
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = CellConfig.load(args.config)
    event = next(row['data'] for row in json.loads(args.events.read_text())
                 if row['event'] == 'planning_submitted')
    begun = time.perf_counter()
    model = calibrated_model(cfg)
    startup = time.perf_counter()-begun
    initial = np.asarray(event['start_joints_rad'])
    model.set_state(initial, cfg.settings.open_width_m)
    start = model.tcp().copy()
    velocity = np.asarray(event['velocity_base_m_s'])
    entry = predict_entry_time(event['capture_monotonic_s'],
        np.asarray(event['entry_observed_position_base_m'])@cfg.direction,
        velocity@cfg.direction, cfg.follow_bounds[0], event['submission_monotonic_s'],
        cfg.raw['planning_reserve_s'], maximum_upstream_distance_m=cfg.raw['freeze_upstream_distance_m'])
    grasps = sorted(map(np.asarray, event['candidate_tcp_base']),
        key=lambda g: Rotation.from_matrix(start[:3, :3].T@g[:3, :3]).magnitude())
    solver = PrecisePoseSolver(model)
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    for profile in ('legacy', 'smooth'):
        delivered = False
        for duration in ([2.] if profile == 'legacy' else cfg.raw['approach_duration_candidates_s']):
            for index, grasp in enumerate(grasps):
                begun = time.perf_counter()
                ref_args = (event['capture_monotonic_s'], entry, duration, grasp[:3, 3], velocity,
                    start[:3, 3], Rotation.from_matrix(start[:3, :3]).as_quat()[[3, 0, 1, 2]],
                    Rotation.from_matrix(grasp[:3, :3]).as_quat()[[3, 0, 1, 2]])
                ref = (Intercept(*ref_args) if profile == 'legacy' else
                       SmoothIntercept(*ref_args, cfg.settings.full_plan_close_window_s, cfg.raw['brake_duration_s']))
                full_duration = 2.6 if profile == 'legacy' else ref.full_duration_s
                boundaries = [duration] if profile == 'legacy' else [duration, ref.follow_end_s]
                knot_hz = cfg.settings.control_hz if profile == 'legacy' else cfg.raw['planning_knot_hz']
                times = sampling_times(full_duration, 1./knot_hz, *boundaries)
                joints = [initial.copy()]
                row = {'profile': profile, 'candidate': index+1, 'approach_s': duration,
                       'geometry_certified': False, 'deadline_certified': False}
                try:
                    for stamp in times[1:]:
                        solved = ((model.solve_pose if profile == 'legacy' else solver.solve)
                                  (*ref.sample(stamp), joints[-1]))
                        if not solved.success:
                            raise PickRejected('逆解未收敛', details={'time_s': float(stamp),
                                'position_error_m': solved.position_error, 'orientation_error_rad': solved.orientation_error})
                        joints.append(solved.joint_positions.copy())
                    row['ik_wall_s'] = time.perf_counter()-begun
                    curve = joint_curve(times, joints, zero_end_acceleration=profile == 'smooth')
                    row.update(peak_velocity_rad_s=float(np.max(abs(np.asarray(extrema(curve, 1))))),
                        peak_acceleration_rad_s2=float(np.max(abs(np.asarray(extrema(curve, 2))))))
                    path = certify_curve(times, joints, cfg, model.command_lower, model.command_upper,
                        identity=tuple(event['identity']), version=1, stage='saved_task_reconstruction',
                        zero_end_acceleration=profile == 'smooth')
                    row.update(status='joint_curve_passed', full_duration_s=path.duration,
                        rest_endpoint_velocity_rad_s=[path.sample(t)[1].tolist() for t in (0., path.duration)],
                        rest_endpoint_acceleration_rad_s2=[path.sample(t)[2].tolist() for t in (0., path.duration)])
                    np.savez_compressed(args.output/f'{profile}_candidate_{index+1}_{duration:.1f}s.npz',
                                        times=times, joints=joints)
                    delivered = True
                except PickRejected as error:
                    row.update(status='rejected', error=str(error), details=error.details)
                row['total_reconstruction_wall_s'] = time.perf_counter()-begun
                rows.append(row)
                if profile == 'smooth' and delivered:
                    break
            if profile == 'smooth' and delivered:
                break
    source = Path(__file__).resolve().parents[1]/'src/nero_core/nero_core/two_second'
    report = {'mode': 'saved_measured_task_kinematic_reconstruction', 'hardware_connections': 0,
        'commands_sent': 0, 'scene_available': False, 'model_startup_wall_s': startup,
        'note': '保存事件没有原始场景：仅比较逆解与q/v/a，不能证明完整认证、在线交付或真实夹持',
        'events_sha256': sha256(args.events.read_bytes()).hexdigest(), 'config_sha256': cfg.source_sha256,
        'algorithm_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
        'effective_velocity_limits_rad_s': cfg.velocity_limits,
        'effective_acceleration_limits_rad_s2': cfg.acceleration_limits, 'attempts': rows}
    (args.output/'report.json').write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(json_value({'model_startup_wall_s': startup, 'attempts': rows}), ensure_ascii=False))


if __name__ == '__main__':
    main()

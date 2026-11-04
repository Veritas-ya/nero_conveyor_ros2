"""Coordinate and clock conversion for simulation-only planning diagnostics."""
import numpy as np


def point_world(pose_base, base_to_world):
    pose = np.asarray(pose_base, dtype=float)
    transform = np.asarray(base_to_world, dtype=float)
    if pose.shape != (7,) or transform.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(transform).all():
        raise ValueError('expected finite base pose7 and base-to-world transform')
    return (transform @ np.r_[pose[:3], 1.])[:3]


def request_target_record(row, base_to_world, clock_origin):
    snapshot = row.get('request_snapshot') or {}
    poses = snapshot.get('target_poses')
    time = snapshot.get('planned_target_time_s')
    if poses is None or len(poses) != 1 or time is None:
        return None
    point = point_world(poses[0], base_to_world)
    return {'request_id': row.get('request_id'), 'source': row.get('request_source'),
            'target_tcp_position_base_m': np.asarray(poses[0], dtype=float)[:3].tolist(),
            'target_tcp_position_world_m': point.tolist(),
            'contact_policy_time_s': float(time), 'contact_sim_time_s': float(clock_origin)+float(time),
            'time_basis': 'simulation clock = policy clock + policy origin',
            'status': 'requested_target_not_a_certified_path'}


def result_target_record(row, base_to_world):
    pose = row.get('target_pose7')
    if pose is None:
        return None
    failed = bool(row.get('error') or row.get('cancelled'))
    return {'request_id': row.get('request_id'),
            'target_tcp_position_base_m': np.asarray(pose, dtype=float)[:3].tolist(),
            'target_tcp_position_world_m': point_world(pose, base_to_world).tolist(),
            'status': 'failed_proposal_not_contact' if failed else 'result_target_requires_plan_certificate',
            'error': row.get('error'),
            'meaning': 'result pose may belong to a revised request; do not pair with submitted contact time'}

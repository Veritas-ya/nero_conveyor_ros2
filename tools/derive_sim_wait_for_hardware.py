"""Derive an offline candidate wait pose from a recorded belt, never touch hardware."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.precise_ik import PrecisePoseSolver
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--frame', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.candidate.resolve() == args.config.resolve():
        parser.error('candidate must be a separate file; it may cause automatic motion when used')
    if args.candidate.parent.resolve() != args.config.parent.resolve():
        parser.error('candidate must share the config directory for relative calibration paths')
    cfg = CellConfig.load(args.config)
    calibration_path = Path(cfg.raw['camera_calibration_file'])
    frame, d2c = read(args.frame, load_runtime_depth_calibration(calibration_path))
    belt = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    plane = belt.plane_to_base
    normal = plane[:3, 2]
    along = cfg.direction - normal * np.dot(cfg.direction, normal)
    along /= np.linalg.norm(along)
    across = np.cross(normal, along)
    # Simulation TCP x points with the belt, z points down. Rotate both into
    # the measured belt basis; no simulation-world translation is reused.
    rotation = np.column_stack((along, -across, -normal))
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    across_coordinate = float(np.mean(belt.bounds_xy_m[0]))
    midpoint = float(np.mean(cfg.raw['follow_interval_base_m']))
    height = .2
    xyz = plane[:3, 3] + plane[:3, 0]*across_coordinate + normal*height
    denominator = float(plane[:3, 1] @ cfg.direction)
    if abs(denominator) < 1e-6:
        raise ValueError('measured plane basis is inconsistent with configured belt direction')
    local_y = float((midpoint - xyz @ cfg.direction)/denominator)
    if not belt.bounds_xy_m[1, 0] <= local_y <= belt.bounds_xy_m[1, 1]:
        raise ValueError('computed wait location lies outside the observed belt bounds')
    xyz += plane[:3, 1]*local_y
    model = calibrated_model(cfg)
    solver = PrecisePoseSolver(model)
    rng = np.random.default_rng(42)
    seeds = [cfg.standby] + [rng.uniform(model.command_lower, model.command_upper) for _ in range(8)]
    solutions = [solver.solve(xyz, quaternion, seed) for seed in seeds]
    valid = [solution for solution in solutions if solution.success]
    if not valid:
        raise ValueError('no precise IK solution found for converted pose')
    # Preserve the TCP exactly and maximize the minimum distance to all
    # configured joint bounds; this does not certify a path or the real cell.
    lower, upper = model.command_lower, model.command_upper
    def residual(joints):
        model.set_state(joints, cfg.settings.open_width_m)
        actual = model.tcp()
        return np.r_[actual[:3, 3]-xyz, Rotation.from_matrix(rotation@actual[:3, :3].T).as_rotvec()]
    def margin_constraint(v):
        return np.r_[v[:7]-lower-v[7], upper-v[:7]-v[7]]
    fits = []
    for solution in valid:
        fit = minimize(lambda v: -v[7], np.r_[solution.joint_positions, 0.], method='SLSQP',
            bounds=list(zip(lower, upper))+[(0, None)],
            constraints=[{'type': 'eq', 'fun': lambda v: residual(v[:7])},
                         {'type': 'ineq', 'fun': margin_constraint}],
            options={'maxiter': 300, 'ftol': 1e-10})
        if fit.success and np.linalg.norm(residual(fit.x[:7])) < 1e-6 and np.min(margin_constraint(fit.x)) >= -1e-8:
            try:
                model.check_state(fit.x[:7], cfg.settings.open_width_m, check_scene=False)
            except (ValueError, RuntimeError):
                continue
            fits.append(fit)
    if not fits:
        raise ValueError('no self-clear, bounded solution with exact TCP found')
    fit = max(fits, key=lambda item: item.x[7])
    joints = fit.x[:7]
    model.set_state(joints, cfg.settings.open_width_m)
    actual = model.tcp().copy()
    # Sample a straight JOINT chord for diagnostic only. The manufacturer's
    # move_j path and physical obstacles are not represented by this check.
    planner = MeasuredPlanner(cfg, model=model)
    planner.plane, planner.belt_bounds = plane, belt.bounds_xy_m
    planner.support, planner.support_origin = normal, plane[:3, 3]
    planner.target = SimpleNamespace(points=np.empty((0, 3)))
    interpolation = {'samples': 401, 'status': 'passed', 'scope': 'model self collision, workspace, observed support plane; not SDK path'}
    for u in np.linspace(0., 1., interpolation['samples']):
        try:
            planner.valid(cfg.standby+(joints-cfg.standby)*u, cfg.settings.open_width_m)
        except (ValueError, RuntimeError) as error:
            interpolation.update(status='rejected', fraction=float(u), reason=str(error), details=getattr(error, 'details', {}))
            break
    text = args.config.read_text()
    old = next(line for line in text.splitlines() if line.startswith('standby_joint_rad:'))
    new = 'standby_joint_rad: [' + ', '.join(repr(float(q)) for q in joints) + ']'
    text = text.replace(old, new, 1)
    text = '# 离线候选，当前现场及实际回位轨迹未验证；抓取入口可能自动回位。\n'+text
    args.candidate.write_text(text)
    report = {'mode': 'offline_recorded_belt_wait_candidate', 'hardware_connections': 0, 'commands_sent': 0,
        'config_source': str(args.config.resolve()), 'config_sha256': digest(args.config),
        'calibration_source': str(calibration_path.resolve()), 'calibration_sha256': digest(calibration_path),
        'frame_source': str(args.frame.resolve()), 'frame_sha256': digest(args.frame),
        'candidate_config': str(args.candidate.resolve()), 'candidate_sha256': digest(args.candidate),
        'belt_plane_to_base': plane.tolist(), 'belt_bounds_xy_m': belt.bounds_xy_m.tolist(),
        'target_origin_rule': 'center of fitted belt width and configured follow interval projection',
        'target_height_above_plane_m': height, 'target_along_projection_m': midpoint,
        'target_tcp_base_m': xyz.tolist(), 'target_tcp_rotation_base': rotation.tolist(),
        'result_tcp_base': actual.tolist(), 'joint_radians': joints.tolist(),
        'joint_degrees': np.rad2deg(joints).tolist(), 'minimum_model_bound_margin_rad': float(fit.x[7]),
        'reference_wait_joints_rad': cfg.standby.tolist(), 'diagnostic_joint_interpolation': interpolation,
        'limitations': ['single historical RGB-D frame and fitted belt bounds', 'nominal TCP',
          'real current robot state and obstacles not observed', 'SDK path and physical motion not tested']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

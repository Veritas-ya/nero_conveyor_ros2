"""Empty-gripper clearance above the previous release before returning to standby."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .direct_drop_guard import lowest_gripper_point
from .precise_ik import PrecisePoseSolver
from .trajectory import retime_samples


def bin_lift_goal(planner, state):
    """Return a lift pose and minimum measured exit height, if still low in the bin."""
    guard = planner.config.raw['experimental_direct_drop_guard']
    planner.model.check_feedback(state)
    pose = planner.model.tcp().copy()
    center = np.asarray(guard['drop_tcp_xyz_base_m'])
    if np.linalg.norm(pose[:2, 3] - center[:2]) > guard['radius_m']:
        return None
    tree = planner.bin_departure_tree
    if not tree.n or not np.isfinite(tree.maxes[2]):
        raise PickRejected('空爪离框缺少上一件的有效释放点云')
    margin = planner.config.raw.get('workpiece_collision_margin_m', .004)
    required = max(float(tree.maxes[2]) + margin, guard['required_min_z_m'])
    lowest, _ = lowest_gripper_point(planner.model)
    if lowest >= required:
        return None
    pose[2, 3] += max(planner.config.settings.lift_m, required + guard['margin_m'] - lowest)
    return pose, required


def plan_bin_lift(planner, start, width, goal_z, motion, *, cancelled):
    model = planner.model
    model.set_state(start, width)
    pose = model.tcp()
    xyz = pose[:3, 3].copy()
    height = float(goal_z - xyz[2])
    if not np.isfinite(height) or height <= 0. or goal_z > planner.config.workspace[2, 1]:
        raise PickRejected('空爪竖直离框目标高度超出可用范围', details={'goal_z_m':goal_z})
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    def sample(fraction):
        position = xyz.copy()
        position[2] += height * fraction**3 * (10. + fraction * (-15. + 6. * fraction))
        return position, quat

    solver = PrecisePoseSolver(model)
    fractions = np.linspace(0., 1., max(20, int(np.ceil(height / .005))) + 1)
    joints = [start.copy()]
    for fraction in fractions[1:]:
        if cancelled():
            raise PickRejected('空爪竖直离框规划已取消或超出预算')
        position, _ = sample(fraction)
        result = solver.solve(position, quat, joints[-1], cancelled=cancelled)
        if not result.success:
            raise PickRejected('空爪竖直离框逆解未收敛', details={'lift_fraction':float(fraction),
                'position_error_m':result.position_error, 'orientation_error_rad':result.orientation_error})
        joints.append(result.joint_positions.copy())
    # Retime the rest-to-rest curve against the configured joint limits.
    path = retime_samples(.1 * fractions, joints, motion, model.command_lower, model.command_upper,
        identity=planner.identity, version=planner.version + 1, stage='recovery_bin_lift')
    reference = SimpleNamespace(duration_s=path.duration, sample=lambda t: sample(t / path.duration))
    certificate = planner.check_path(path, width, reference=reference, payload=False, cancelled=cancelled)
    return replace(path, diagnostics={**path.diagnostics, **motion.diagnostics(),
        'lift_height_m':height, 'reference_check':certificate})

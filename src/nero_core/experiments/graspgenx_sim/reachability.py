"""CPU prefilter for an unchanged visual/network grasp's vertical IK branch.

This checks nominal robot geometry only, never object pose/CAD or grasp quality.
No actuator commands, physics steps, renderer or network inference are used.
The ordinary execution planner must independently revalidate its entire route.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .candidates import CandidateEnvironmentSnapshot, _ik_info, _solve_candidate_ik


def check_vertical_reachability(
    snapshot: CandidateEnvironmentSnapshot,
    pregrasp_q,
    intercept_tcp_pose,
    *,
    maximum_z_step_m: float = .01,
    pregrasp_tcp_pose=None,
    lift_offset_world=None,
) -> dict[str, Any]:
    """Warm-start one branch: open descent, then closed vertical clearance.

    ``intercept_tcp_pose`` is world XYZ + WXYZ, already translated using the
    visual track. By default the supplied pregrasp is directly above it.
    The measured moving adapter supplies its actual earlier high endpoint
    and base-Z lift expressed in these coordinates. Only those supplied
    endpoints are sampled; no pose or quaternion is invented. Closed jaws at
    q=0 mean robot feasibility only, not a simulated or confirmed acquisition.
    Snapshot ownership must remain exclusive while its private IK is used.
    Failure reports contain the attempted q and the last fully checked q.
    """
    import mujoco

    q = np.asarray(pregrasp_q, dtype=float).copy()
    pose = np.asarray(intercept_tcp_pose, dtype=float).copy()
    step = float(maximum_z_step_m)
    if (q.shape != (7,) or pose.shape != (7,)
            or not np.isfinite(np.r_[q, pose, step]).all()
            or np.linalg.norm(pose[3:]) < 1e-9 or not 0 < step <= .01):
        raise ValueError("require finite 7-joint seed, XYZ+WXYZ and Z step in (0,.01]")
    pose[3:] /= np.linalg.norm(pose[3:])
    robot = snapshot.robot_config
    trajectory = robot.get("trajectory", {})
    controller = robot.get("controller", {})
    approach = float(trajectory.get("approach_height", .10))
    clearance = float(trajectory.get("lift_clearance_height", .68))
    margin = max(float(trajectory.get("minimum_joint_limit_margin", .025)),
                 float(controller.get("joint_command_margin", 0.)))
    reserve = float(trajectory.get("pregrasp_tracking_joint_margin_reserve", .010))
    cone = float(trajectory.get("acquisition_minimum_downward_alignment", .52))
    condition_limit = float(trajectory.get("singularity_hard_stop_condition", 45.))
    joint_jump = float(trajectory.get("max_ik_joint_step", .25))
    resolution = float(trajectory.get("planner_edge_resolution", .018))
    axis = np.asarray(trajectory.get("approach_axis_local", [0., 0., 1.]), dtype=float)
    if (not np.isfinite([approach, clearance, margin, reserve, cone,
                         condition_limit, joint_jump, resolution]).all()
            or approach <= 0 or clearance < pose[2] or min(margin, reserve) < 0
            or not -1 <= cone <= 1 or min(condition_limit, joint_jump, resolution) <= 0
            or axis.shape != (3,) or not np.isfinite(axis).all()
            or np.linalg.norm(axis) < 1e-9):
        raise ValueError("invalid robot trajectory safety configuration")
    axis = axis / np.linalg.norm(axis)
    ranges = np.asarray(snapshot.model.jnt_range[snapshot.arm_joint_ids], dtype=float)
    if ranges.shape != (7, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("all seven original joint limits must be finite and ordered")
    span = ranges[:, 1] - ranges[:, 0]
    gripper_ranges = np.asarray(snapshot.model.jnt_range[snapshot.gripper_joint_ids])
    if np.any(gripper_ranges[:, 0] > 0) or np.any(gripper_ranges[:, 1] < 0):
        raise ValueError("zero closed gripper position must lie within original limits")
    open_q = gripper_ranges[np.arange(len(gripper_ranges)),
                            np.argmax(np.abs(gripper_ranges), axis=1)]
    scratch = mujoco.MjData(snapshot.model)
    scratch.qpos[:] = snapshot.data.qpos
    scratch.qvel[:] = 0.
    jp, jr = np.zeros((3, snapshot.model.nv)), np.zeros((3, snapshot.model.nv))
    prepose = pose.copy()
    prepose[2] += approach
    if pregrasp_tcp_pose is not None:
        prepose = np.asarray(pregrasp_tcp_pose, dtype=float).copy()
        if prepose.shape != (7,) or not np.isfinite(prepose).all():
            raise ValueError("pregrasp_tcp_pose must be finite XYZ+WXYZ")
    liftpose = pose.copy()
    liftpose[2] = clearance
    if lift_offset_world is not None:
        offset = np.asarray(lift_offset_world, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("lift_offset_world must be a finite 3-vector")
        liftpose[:3] = pose[:3] + offset
    samples = []
    result = {
        "success": False, "path_fully_validated": False,
        "validation": "NOT_EXECUTED_MOTION_nominal_robot_only",
        "object_truth_used": False, "contact_quality_validated": False,
        "intercept_tcp_pose_world_wxyz": pose.tolist(),
        "pregrasp_tcp_pose_world_wxyz": prepose.tolist(),
        "lift_tcp_pose_world_wxyz": liftpose.tolist(),
        "limits": {"joint_ranges_rad": ranges.tolist(),
                   "minimum_joint_margin_rad": margin,
                   "pregrasp_minimum_joint_margin_rad": margin + reserve,
                   "minimum_downward_alignment": cone,
                   "maximum_jacobian_condition": condition_limit,
                   "maximum_ik_joint_step_rad": joint_jump,
                   "edge_maximum_joint_step_rad": .025,
                   "edge_normalized_resolution": resolution,
                   "maximum_z_step_m": step, "lift_clearance_height_m": clearance},
        "samples": samples, "failure": None,
        "final_joint_positions": None,
    }

    def state(joints, *, closed, floor, required_margin):
        # Native validation ignores object_* contacts. Object qpos, labels,
        # lifecycle, CAD and dimensions never contribute to this decision.
        scratch.qpos[snapshot.arm_qpos_adrs] = joints
        scratch.qpos[snapshot.gripper_qpos_adrs] = 0. if closed else open_q
        mujoco.mj_forward(snapshot.model, scratch)
        margins = np.minimum(joints - ranges[:, 0], ranges[:, 1] - joints)
        tcp_z = float(scratch.site_xpos[snapshot.ee_site_id, 2])
        alignment = -float((scratch.site_xmat[snapshot.ee_site_id].reshape(3, 3) @ axis)[2])
        mujoco.mj_jacSite(snapshot.model, scratch, jp, jr, snapshot.ee_site_id)
        jac = np.vstack((jp[:, snapshot.arm_dof_adrs], jr[:, snapshot.arm_dof_adrs]))
        jac[3:] *= .30
        singular = np.linalg.svd(jac, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else float("inf")
        info = {"joint_margin_rad": float(margins.min()),
                "limiting_joint_index": int(margins.argmin()), "tcp_height_m": tcp_z,
                "downward_alignment": alignment,
                "jacobian_condition": condition if np.isfinite(condition) else None}
        if margins.min() < required_margin - 1e-9:
            return "joint_margin_rejected", info
        if tcp_z < floor - 1e-9:
            return "tcp_height_rejected", info
        if alignment < cone - 1e-9:
            return "orientation_cone_rejected", info
        if condition > condition_limit:
            return "singularity_rejected", info
        if not snapshot._arm_state_data_is_safe(
            joints, scratch, approach_axis_local=axis,
            minimum_downward_alignment=cone, minimum_tcp_height=floor,
            minimum_joint_margin=required_margin,
            maximum_jacobian_condition=condition_limit,
        ):
            return "robot_scene_or_self_collision_rejected", info
        return None, info

    def fail(phase, height, reason, *, attempted_q=None, ik=None, **detail):
        result["failure"] = {"phase": phase, "height_m": float(height),
                             "reason": reason, "ik": ik, **detail}
        if attempted_q is not None:
            attempted = np.asarray(attempted_q)
            result["failure"]["joint_positions"] = attempted.tolist()
            result["failure"]["attempted_joint_margin_rad"] = float(np.min(
                np.minimum(attempted - ranges[:, 0], ranges[:, 1] - attempted)))
        return result

    start_actual = np.asarray(snapshot.ik.forward_pose(q), dtype=float)
    pos_error = float(np.linalg.norm(start_actual[:3] - prepose[:3]))
    ori_error = float(2 * np.arccos(np.clip(abs(start_actual[3:] @ prepose[3:]), 0., 1.)))
    if (pos_error > snapshot.ik.config.position_tolerance + 1e-9
            or ori_error > snapshot.ik.config.orientation_tolerance + 1e-9):
        return fail("pregrasp", prepose[2], "pregrasp_seed_pose_mismatch",
                    attempted_q=q, position_error_m=pos_error, orientation_error_rad=ori_error)
    reason, info = state(q, closed=False, floor=prepose[2] - .015,
                         required_margin=margin + reserve)
    if reason:
        return fail("pregrasp", prepose[2], reason, attempted_q=q, **info)
    result["final_joint_positions"] = q.tolist()
    samples.append({"phase": "pregrasp", "height_m": float(prepose[2]),
                    "joint_positions": q.tolist(), "state": info, "edge_samples": 0})
    for phase, start_pose, end_pose, closed in (
        ("descent", prepose, pose, False),
        ("lift", pose, liftpose, True),
    ):
        # Explicitly check the open-to-closed boundary at the grasp endpoint.
        reason, info = state(q, closed=closed, floor=start_pose[2] - .015,
                             required_margin=margin)
        if reason:
            return fail(phase, start_pose[2], reason, attempted_q=q, **info)
        count = int(math.ceil(np.linalg.norm(end_pose[:3] - start_pose[:3]) / step))
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            target = end_pose.copy()
            target[:3] = start_pose[:3] + fraction * (end_pose[:3] - start_pose[:3])
            height = float(target[2])
            try:
                solved = _solve_candidate_ik(
                    snapshot, target[:3], target[3:], qpos_seed=q,
                    allow_multiseed=False, minimum_joint_margin=margin)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                return fail(phase, height, "ik_exception",
                            error=f"{type(error).__name__}: {error}")
            ik = _ik_info(solved)
            candidate = np.asarray(solved.joint_positions, dtype=float)
            if candidate.shape != (7,) or not np.isfinite(candidate).all():
                return fail(phase, height, "invalid_ik_joint_positions", ik=ik)
            if (not solved.success
                    or solved.position_error > snapshot.ik.config.position_tolerance + 1e-9
                    or solved.orientation_error > snapshot.ik.config.orientation_tolerance + 1e-9):
                return fail(phase, height, "ik_failed", attempted_q=candidate, ik=ik)
            delta = candidate - q
            if np.max(np.abs(delta)) > joint_jump + 1e-9:
                return fail(phase, height, "ik_branch_jump_rejected", attempted_q=candidate,
                            ik=ik, maximum_joint_step_rad=float(np.max(np.abs(delta))))
            start_height = float(snapshot.ik.forward_pose(q)[2])
            floor = min(start_height, float(height)) - .015
            edge_count = max(1, int(math.ceil(np.max(np.abs(delta)) / .025)),
                             int(math.ceil(np.linalg.norm(delta / span) / resolution)))
            for edge_index in range(edge_count + 1):
                fraction = edge_index / edge_count
                reason, info = state(q + delta * fraction, closed=closed, floor=floor,
                                     required_margin=margin)
                if reason:
                    return fail(phase, height, reason, attempted_q=candidate, ik=ik,
                                edge_fraction=fraction, edge_sample_count=edge_count + 1,
                                failing_edge_joint_positions=(q + delta * fraction).tolist(), **info)
            q = candidate.copy()
            result["final_joint_positions"] = q.tolist()
            samples.append({"phase": phase, "height_m": float(height), "ik": ik,
                            "joint_positions": q.tolist(), "state": info,
                            "edge_samples": edge_count + 1, "gripper_state": "closed_zero" if closed else "fully_open"})
    result.update(success=True, path_fully_validated=True)
    return result

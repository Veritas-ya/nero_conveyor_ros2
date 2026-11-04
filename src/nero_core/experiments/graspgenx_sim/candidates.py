"""Filter original GraspGen-X SE(3) candidates for the actual AGX/NERO model.

This module never reads object poses, labels, or contact truth.  It preserves
the network grasp position/orientation, apart from an explicitly recorded
parallel-jaw 180-degree symmetry, and forecasts only belt translation for IK.
Endpoint validation is a prefilter, not a certificate of the executed path.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import json
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent


class CandidateEnvironmentSnapshot:
    """Private model/data/IK for background candidate filtering.

    Construct on the simulation owner thread between steps. Spawn/recycle
    changes model collision flags and physical properties, so the MjModel is
    also copied; sharing just a private MjData would still have a race.
    No renderer, actuator controller, object lifecycle, or hardware handle is
    retained. Physics object transforms exist in copied MjData for engine
    consistency but are never used as candidate-selection observations.
    """
    def __init__(self, env):
        import mujoco
        from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK
        self.model = copy.copy(env.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, env.data)
        self.source_timestamp = float(env.data.time)
        self.source_scene_epoch = int(getattr(env, "_scene_epoch", 0))
        self.belt_height = float(env.belt_height)
        self.scene_config = copy.deepcopy(env.scene_config)
        self.robot_config = copy.deepcopy(env.robot_config)
        self.arm_joint_names = tuple(env.arm_joint_names)
        self.ee_site_name = str(env.ee_site_name)
        self.ee_site_id = int(env.ee_site_id)
        for name in ("arm_joint_ids", "arm_qpos_adrs", "arm_dof_adrs",
                     "gripper_joint_ids", "gripper_qpos_adrs"):
            setattr(self, name, np.asarray(getattr(env, name)).copy())
        self._robot_collision_body_ids = set(env._robot_collision_body_ids)
        self.ik = DampedLeastSquaresIK(self.model, self.data, self.arm_joint_names,
                                      self.ee_site_name, copy.deepcopy(env.ik.config))
        self._path_validation_data = mujoco.MjData(self.model)
        self._path_validation_jacobian_position = np.zeros((3, self.model.nv))
        self._path_validation_jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self):
        return self.data.qpos[self.arm_qpos_adrs].copy()

    def solve_ik_result(self, *args, **kwargs):
        from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
        return ConveyorSortingEnv.solve_ik_result(self, *args, **kwargs)

    def is_arm_configuration_safe(self, *args, **kwargs):
        from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
        return ConveyorSortingEnv.is_arm_configuration_safe(self, *args, **kwargs)

    def _arm_state_data_is_safe(self, *args, **kwargs):
        from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
        return ConveyorSortingEnv._arm_state_data_is_safe(self, *args, **kwargs)


def snapshot_env_for_candidates(env):
    """Copy on the owner thread, then pass the result to one worker thread."""
    return CandidateEnvironmentSnapshot(env)


def _rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _pose_vector(transform):
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q *= -1
    return np.r_[transform[:3, 3], q]


@lru_cache(maxsize=8)
def _assets(transforms_path: str, sweep_path: str):
    transforms = json.loads(Path(transforms_path).read_text())
    sweep = json.loads(Path(sweep_path).read_text())
    canonical_to_tcp = np.asarray(transforms["T_canonical_ee_site"], dtype=float)
    canonical_to_base = np.asarray(transforms["T_canonical_gripper_base"], dtype=float)
    fingertip_depth = float(sweep["fingertip_depth"])
    project = ROOT.parents[1]
    xml = ET.parse(project / "mujoco_conveyor_sorting/models/robot.xml").getroot()
    bodies = {element.get("name"): element for element in xml.iter("body")}
    from .gripper_agx.build_assets import stl_triangles
    vertices = []
    for name in ("gripper_link1", "gripper_link2"):
        body = bodies[name]
        transform = np.eye(4)
        transform[:3, :3] = _rotation(np.fromstring(body.get("quat", "1 0 0 0"), sep=" "))
        transform[:3, 3] = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        joint = body.find("joint")
        limits = np.fromstring(joint.get("range"), sep=" ")
        open_q = float(limits[np.argmax(np.abs(limits))])
        axis = np.fromstring(joint.get("axis"), sep=" ")
        transform[:3, 3] += transform[:3, :3] @ axis * open_q
        transform = canonical_to_base @ transform
        mesh_path = Path(transforms_path).parent / "meshes" / f"{name}.stl"
        local = np.unique(stl_triangles(mesh_path).reshape(-1, 3), axis=0)
        vertices.append(local @ transform[:3, :3].T + transform[:3, 3])
    finger_vertices = np.concatenate(vertices)
    if abs(float(finger_vertices[:, 2].max()) - fingertip_depth) > 1e-6:
        raise ValueError("sweep fingertip_depth and actual AGX open-finger meshes disagree")
    return canonical_to_tcp, finger_vertices, sweep


def _ik_info(value):
    return {
        "success": bool(value.success),
        "joint_positions": np.asarray(value.joint_positions, dtype=float).tolist(),
        "position_error_m": float(value.position_error),
        "orientation_error_rad": float(value.orientation_error),
        "iterations": int(getattr(value, "iterations", 0)),
        "message": str(getattr(value, "message", "")),
    }


def _solve_candidate_ik(env, position, quaternion, *, minimum_joint_margin, **kwargs):
    """Let the measured adapter solve inside this stage's acceptance bounds.

    Historical simulation environments keep their existing IK implementation.
    The margin comes from the same stage parameters as endpoint/path validation.
    """
    solve = getattr(env, "solve_ik_result_with_margin", None)
    if callable(solve):
        return solve(position, quaternion, minimum_joint_margin=minimum_joint_margin, **kwargs)
    return env.solve_ik_result(position, quaternion, **kwargs)


def _endpoint_validator(env):
    """Check the open AGX on private physics data, even during a prior carry."""
    if not all(hasattr(env, name) for name in ("model", "data", "arm_qpos_adrs",
                                                "gripper_joint_ids", "gripper_qpos_adrs",
                                                "_arm_state_data_is_safe")):
        return env.is_arm_configuration_safe
    import mujoco
    scratch = mujoco.MjData(env.model)
    snapshot_qpos = np.asarray(env.data.qpos).copy()
    ranges = np.asarray(env.model.jnt_range[env.gripper_joint_ids])
    open_q = ranges[np.arange(len(ranges)), np.argmax(np.abs(ranges), axis=1)]

    def valid(joints, **kwargs):
        scratch.qpos[:] = snapshot_qpos
        scratch.qpos[env.arm_qpos_adrs] = joints
        scratch.qpos[env.gripper_qpos_adrs] = open_q
        scratch.qvel[:] = 0.
        mujoco.mj_forward(env.model, scratch)
        return env._arm_state_data_is_safe(
            np.asarray(joints), scratch, approach_axis_local=np.array([0., 0., 1.]),
            minimum_downward_alignment=kwargs["minimum_downward_alignment"],
            minimum_tcp_height=kwargs["minimum_tcp_height"],
            minimum_joint_margin=kwargs["minimum_joint_margin"],
            maximum_jacobian_condition=None,
        )
    return valid


def _candidate_annotations(metadata, count):
    """Keep scorer/proposal identities aligned with the returned pose array."""
    metadata = metadata or {}
    sources = metadata.get('candidate_sources', ['diffusion'] * count)
    preferences = metadata.get('candidate_task_preferences', [{'group': 2}] * count)
    if len(sources) != count or len(preferences) != count:
        raise ValueError('candidate source/preference metadata must match every returned pose')
    if any(source not in ('diffusion', 'obb_top') for source in sources):
        raise ValueError('unknown GraspGen-X candidate source')
    return [{'candidate_source': source, 'task_preference': dict(preference)}
            for source, preference in zip(sources, preferences)]


def _geometric_directions(poses, values, variants, *, canonical_to_tcp,
                          finger_vertices, belt_height, minimum_score,
                          tilt_limit, minimum_height, clearance, annotations=None):
    """Prepare the whole observed candidate pool before expensive robot IK.

    This is the same SE(3), score, support-plane and open-finger test used
    below.  Preparing a pool does not approve an endpoint or an arm path.
    The historical simulation keeps its original score-first iteration.
    """
    directions, rejected = [], []
    for index in np.argsort(-np.where(np.isfinite(values), values, -np.inf), kind="stable"):
        original = poses[index]
        base = {"source_index": int(index), "score": float(values[index]),
                **({} if annotations is None else annotations[index])}
        valid = (np.isfinite(original).all()
                 and np.allclose(original[3], [0, 0, 0, 1], atol=1e-6)
                 and np.allclose(original[:3, :3].T @ original[:3, :3], np.eye(3), atol=1e-4)
                 and abs(float(np.linalg.det(original[:3, :3])) - 1) < 1e-4)
        if not valid or not np.isfinite(values[index]):
            rejected.append({**base, "variant": "original", "reason": "invalid_network_se3_or_score"})
            continue
        for variant, symmetry_transform in variants:
            record = {**base, "variant": variant}
            if values[index] < minimum_score:
                rejected.append({**record, "reason": "below_network_score_threshold"})
                continue
            canonical = original @ symmetry_transform
            tilt = float(np.degrees(np.arccos(np.clip(-canonical[2, 2], -1., 1.))))
            record['approach_tilt_deg'] = tilt
            if tilt_limit is not None and tilt > tilt_limit + 1e-9:
                rejected.append({**record, "reason": "approach_tilt_exceeds_limit"})
                continue
            tcp = canonical @ canonical_to_tcp
            height = float(tcp[2, 3] - belt_height)
            record['tcp_height_above_belt_m'] = height
            if height < minimum_height - 1e-9:
                rejected.append({**record, "reason": "tcp_below_minimum_height"})
                continue
            lowest = float((finger_vertices @ canonical[:3, :3].T + canonical[:3, 3])[:, 2].min())
            record['minimum_open_finger_clearance_m'] = lowest - belt_height
            if lowest - belt_height < clearance - 1e-9:
                rejected.append({**record, "reason": "open_finger_intersects_belt_clearance"})
                continue
            directions.append({'source_index': int(index), 'variant': variant,
                               'score': float(values[index]), 'canonical': canonical,
                               'tcp': tcp, 'record': record,
                               **({} if annotations is None else annotations[index]),
                               'symmetry_transform': symmetry_transform})
    return directions, rejected


def select_candidates(env, network_poses_world, scores, track_position_world,
                      cfg: Mapping[str, Any], qpos_seed=None, observed_cloud_world=None,
                      candidate_metadata=None, skip_variants=None):
    """Return original network grasps after the configured robot checks.

    Accepted poses are at the capture instant. ``intercept_ik`` is evaluated
    after translating that visual track to the preferred intercept X, which
    defaults to 0.08 m in scene world coordinates. The live adapter can rank
    the full pool by kinematic hints and continue within it after an IK batch
    fails; historical environments retain their network-score order/budget.
    No candidate is changed to
    make IK succeed. Scene checks use the existing model endpoint validator;
    the original controller must still plan and verify every executed edge.
    """
    config = dict(cfg.get("graspgenx", cfg))
    direct = bool(config.get('direct_contact_selection', False))
    visual_geometry = None
    if config.get("visual_approach_validation", False):
        from experiments.graspgenx_sim.visual_clearance import load_open_fingers
        observed_cloud_world = np.asarray(observed_cloud_world, dtype=float)
        if (observed_cloud_world.ndim != 2 or observed_cloud_world.shape[1] != 3
                or not len(observed_cloud_world) or not np.isfinite(observed_cloud_world).all()):
            raise ValueError("visual approach validation requires finite captured point cloud")
        visual_geometry = load_open_fingers()
    transforms_path = Path(config.get("gripper_transforms_path", ROOT / "gripper_agx/transforms.json")).resolve()
    sweep_path = Path(config.get("gripper_sweep_path", ROOT / "gripper_agx/sweep_volume_params.json")).resolve()
    canonical_to_tcp, finger_vertices, _ = _assets(str(transforms_path), str(sweep_path))
    poses = np.asarray(network_poses_world, dtype=float)
    values = np.asarray(scores, dtype=float).reshape(-1)
    reference = np.asarray(track_position_world, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(values) != len(poses):
        raise ValueError("network poses must be [N,4,4] and scores must have N entries")
    annotations = _candidate_annotations(candidate_metadata, len(poses))
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("track_position_world must be three finite visual coordinates")
    belt_height = float(env.belt_height)
    configured_tilt = config.get("maximum_approach_tilt_deg", 25.0)
    tilt_limit = None if configured_tilt is None else float(configured_tilt)
    if tilt_limit is not None and (not np.isfinite(tilt_limit) or not 0. <= tilt_limit <= 180.):
        raise ValueError("maximum_approach_tilt_deg must be None or finite in [0,180]")
    minimum_alignment = -1. if tilt_limit is None else float(np.cos(np.radians(tilt_limit)))
    minimum_height = float(config.get("minimum_tcp_height_above_belt_m", .012))
    clearance = float(config.get("fingertip_clearance_m", .002))
    minimum_score = float(config.get("minimum_score", 0.0))
    approach_height = float(config.get("pregrasp_clearance_m", .10))
    workspace = getattr(env, "scene_config", {}).get("workspace", {})
    preferred_x = float(config.get("preferred_intercept_x", workspace.get("preferred_grasp_x", .08)))
    robot_config = getattr(env, "robot_config", {})
    command_margin = float(robot_config.get("controller", {}).get("joint_command_margin", 0.0))
    margin = max(float(config.get("minimum_joint_limit_margin", .025)), command_margin)
    pregrasp_margin = margin + float(robot_config.get("trajectory", {}).get(
        "pregrasp_tracking_joint_margin_reserve", .010))
    max_ik = int(config.get("maximum_ik_candidates", 32))
    max_accepted = int(config.get("maximum_accepted_candidates", 8))
    symmetry = bool(config.get("allow_parallel_jaw_z_pi", True))
    if min(max_ik, max_accepted) < 1:
        raise ValueError("candidate evaluation budgets must be positive")
    if qpos_seed is None:
        qpos_seed = np.asarray(env.get_joint_positions(), dtype=float)
    else:
        qpos_seed = np.asarray(qpos_seed, dtype=float).copy()
    if qpos_seed.shape != (7,) or not np.isfinite(qpos_seed).all():
        raise ValueError("qpos_seed must have seven finite NERO joint values")
    scene_endpoint_safe = _endpoint_validator(env)
    variants = [("original", np.eye(4))]
    if symmetry:
        variants.append(("parallel_jaw_z_pi", np.diag([-1., -1., 1., 1.])))
    accepted, rejected = [], []
    skip_variants = frozenset(skip_variants or ())
    ik_count = 0
    rank_pool = getattr(env, 'rank_candidate_directions', None)
    pooled = bool(config.get('reachability_ranked_selection', False)) and callable(rank_pool)
    prepared = None
    if pooled:
        prepared, geometric_rejections = _geometric_directions(
            poses, values, variants, canonical_to_tcp=canonical_to_tcp,
            finger_vertices=finger_vertices, belt_height=belt_height,
            minimum_score=minimum_score, tilt_limit=tilt_limit,
            minimum_height=minimum_height, clearance=clearance, annotations=annotations)
        rejected.extend(geometric_rejections)
        prepared = rank_pool(prepared)
        def ranked_groups(pool):
            # Resume the same pool after a first-grasp numerical shortcut
            # fails its *full* low/scene/continuation checks. Only then pay
            # for GPU hints for the remaining candidates.
            position = 0
            expanded = False
            while position < len(pool):
                row = pool[position]
                yield row['source_index'], [(row['variant'], row['symmetry_transform'], row)]
                position += 1
                end_group = position == len(pool) or pool[position]['source_index'] != row['source_index']
                expand = getattr(env, 'rank_remaining_candidate_directions', None)
                if end_group and not accepted and not expanded and callable(expand):
                    remaining = expand(pool[position:])
                    expanded = True
                    if remaining is not None:
                        pool, position = remaining, 0

        # Both allowed jaw directions remain adjacent. Rank only determines
        # work order; every accepted direction still runs all original checks.
        groups = ranked_groups(prepared)
    else:
        order = np.argsort(-np.where(np.isfinite(values), values, -np.inf), kind='stable')
        if candidate_metadata is not None:
            # Offline/non-ranked callers keep the same task preference as
            # the live pool; metadata-free historical calls remain unchanged.
            order = sorted(order, key=lambda index: (
                annotations[index]['task_preference'].get('group', 2),
                -float(values[index]) if np.isfinite(values[index]) else np.inf, int(index)))
        groups = [(index, [(variant, transform, None) for variant, transform in variants])
                  for index in order]
    for index, direction_group in groups:
        original = poses[index]
        base = {"source_index": int(index), "score": float(values[index]), **annotations[index]}
        valid_transform = (np.isfinite(original).all()
                           and np.allclose(original[3], [0, 0, 0, 1], atol=1e-6)
                           and np.allclose(original[:3, :3].T @ original[:3, :3], np.eye(3), atol=1e-4)
                           and abs(float(np.linalg.det(original[:3, :3])) - 1) < 1e-4)
        if not valid_transform or not np.isfinite(values[index]):
            rejected.append({**base, "variant": "original", "reason": "invalid_network_se3_or_score"})
            continue
        for variant, symmetry_transform, prepared_row in direction_group:
            if (int(index), variant) in skip_variants:
                continue
            record = {**base, "variant": variant}
            compare_same_grasp = (config.get('compare_parallel_jaw_variants', False)
                                  and accepted and accepted[-1]['source_index'] == int(index))
            if (len(accepted) >= max_accepted and not compare_same_grasp) or (not pooled and ik_count >= max_ik):
                rejected.append({**record, "reason": ("not_evaluated_after_selection" if pooled
                                                      else "not_evaluated_budget"), "evaluated": False})
                continue
            if values[index] < minimum_score:
                rejected.append({**record, "reason": "below_network_score_threshold"})
                continue
            canonical = original @ symmetry_transform
            approach = canonical[:3, 2]
            tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1., 1.))))
            record["approach_tilt_deg"] = tilt
            if tilt_limit is not None and tilt > tilt_limit + 1e-9:
                rejected.append({**record, "reason": "approach_tilt_exceeds_limit"})
                continue
            tcp = canonical @ canonical_to_tcp
            height = float(tcp[2, 3] - belt_height)
            record["tcp_height_above_belt_m"] = height
            if height < minimum_height - 1e-9:
                rejected.append({**record, "reason": "tcp_below_minimum_height"})
                continue
            lowest = float((finger_vertices @ canonical[:3, :3].T + canonical[:3, 3])[:, 2].min())
            record["minimum_open_finger_clearance_m"] = lowest - belt_height
            if lowest - belt_height < clearance - 1e-9:
                rejected.append({**record, "reason": "open_finger_intersects_belt_clearance"})
                continue
            if prepared_row is not None:
                record['selection_priority'] = prepared_row.get('selection_priority')
            prepare_motion = getattr(env, "prepare_candidate_motion", None)
            if callable(prepare_motion):
                config.update(prepare_motion(tcp))
                record['candidate_motion_plan'] = config.get('candidate_motion_plan')
            # In a live ranked pool, first find an actual arm branch.  Failed
            # IK must not trigger the expensive finger sweep on every sample.
            # The identical sweep still runs before accepting that branch.
            if visual_geometry is not None and not pooled:
                visual_pose = canonical.copy()
                visual_kwargs = {}
                if "intercept_translation_world" in config:
                    visual_pose[:3, 3] += config["intercept_translation_world"]
                    visual_kwargs['lift_vector_world'] = config['pregrasp_offset_world']
                if direct:
                    visual_kwargs['lift_vector_world'] = np.zeros(3)
                visual_checks = visual_geometry.vertical_descent(
                    observed_cloud_world, visual_pose, lift_m=0. if direct else approach_height,
                    step_m=float(config.get("visual_descent_sample_step_m", .01)),
                    penetration_margin_m=float(config.get("visual_penetration_margin_m", .001)), **visual_kwargs)
                record['visual_open_finger_contact' if direct else 'visual_open_finger_descent'] = visual_checks
                if visual_checks["collision_sample_count"]:
                    rejected.append({**record, "reason": "observed_points_inside_open_finger_descent"})
                    continue
            intercept = tcp.copy()
            if "intercept_translation_world" in config:
                intercept[:3, 3] += np.asarray(config["intercept_translation_world"], dtype=float)
            else:
                intercept[0, 3] += preferred_x - reference[0]
            intercept_pose = _pose_vector(intercept)
            pregrasp_pose = intercept_pose.copy()
            if not direct:
                pregrasp_pose[:3] += np.asarray(config.get(
                    "pregrasp_offset_world", [0., 0., approach_height]), dtype=float)
            ik_count += 1
            try:
                if not direct:
                    pregrasp_ik = _solve_candidate_ik(
                        env, pregrasp_pose[:3], pregrasp_pose[3:], qpos_seed=qpos_seed,
                        minimum_joint_margin=pregrasp_margin)
                    record["pregrasp_ik"] = _ik_info(pregrasp_ik)
                    if not pregrasp_ik.success:
                        rejected.append({**record, "reason": "pregrasp_ik_failed"})
                        continue
                intercept_ik = _solve_candidate_ik(
                    env, intercept_pose[:3], intercept_pose[3:],
                    qpos_seed=qpos_seed if direct else pregrasp_ik.joint_positions, minimum_joint_margin=margin)
                record["intercept_ik"] = _ik_info(intercept_ik)
                if not intercept_ik.success:
                    rejected.append({**record, "reason": "intercept_ik_failed"})
                    continue
                if visual_geometry is not None and pooled:
                    visual_pose = canonical.copy()
                    visual_kwargs = {}
                    if "intercept_translation_world" in config:
                        visual_pose[:3, 3] += config["intercept_translation_world"]
                        visual_kwargs['lift_vector_world'] = config['pregrasp_offset_world']
                    if direct:
                        visual_kwargs['lift_vector_world'] = np.zeros(3)
                    visual_checks = visual_geometry.vertical_descent(
                        observed_cloud_world, visual_pose, lift_m=0. if direct else approach_height,
                        step_m=float(config.get("visual_descent_sample_step_m", .01)),
                        penetration_margin_m=float(config.get("visual_penetration_margin_m", .001)),
                        **visual_kwargs)
                    record['visual_open_finger_contact' if direct else 'visual_open_finger_descent'] = visual_checks
                    if visual_checks["collision_sample_count"]:
                        rejected.append({**record, "reason": "observed_points_inside_open_finger_descent"})
                        continue
                safe_kwargs = {"minimum_downward_alignment": minimum_alignment,
                               "minimum_joint_margin": margin,
                               "minimum_tcp_height": belt_height + minimum_height}
                if not direct and not scene_endpoint_safe(pregrasp_ik.joint_positions,
                                           **{**safe_kwargs, "minimum_joint_margin": pregrasp_margin}):
                    rejected.append({**record, "reason": "pregrasp_scene_endpoint_unsafe",
                                     "details": getattr(env, 'last_rejection', None)})
                    continue
                if not scene_endpoint_safe(intercept_ik.joint_positions, **safe_kwargs):
                    rejected.append({**record, "reason": "intercept_scene_endpoint_unsafe",
                                     "details": getattr(env, 'last_rejection', None)})
                    continue
                if not direct and config.get("vertical_reachability_validation", False):
                    from experiments.graspgenx_sim.reachability import check_vertical_reachability
                    branch_kwargs = ({"pregrasp_tcp_pose": pregrasp_pose,
                                      "lift_offset_world": config["lift_offset_world"]}
                                     if "pregrasp_offset_world" in config else {})
                    continuation = check_vertical_reachability(
                        env, pregrasp_ik.joint_positions, intercept_pose, **branch_kwargs)
                    record["nominal_vertical_reachability"] = continuation
                    if not continuation["success"]:
                        rejected.append({**record, "reason": "nominal_descent_or_lift_branch_unavailable"})
                        continue
            except Exception as error:
                rejected.append({**record, "reason": "ik_or_scene_validation_exception",
                                 "error": f"{type(error).__name__}: {error}"})
                continue
            accepted.append({
                **record, "T_world_canonical": canonical.copy(), "T_world_tcp": tcp.copy(),
                "original_network_pose_world": original.copy(),
                "tcp_pose_world_wxyz": _pose_vector(tcp),
                "intercept_tcp_pose_world_wxyz": intercept_pose,
                "pregrasp_tcp_pose_world_wxyz": None if direct else pregrasp_pose,
                'mandatory_high_waypoint': not direct,
                "closing_width_m": 0.0,
                "source": 'graspgenx_obb_top' if record['candidate_source'] == 'obb_top' else 'graspgenx_diffusion',
                "scene_endpoint_gripper_state": "fully_open",
                "minimum_joint_margin_rad": margin,
                "pregrasp_minimum_joint_margin_rad": None if direct else pregrasp_margin,
                "path_validation": ('contact_endpoint_only_complete_continuous_curve_required' if direct else
                                    "nominal_vertical_robot_only_original_controller_must_revalidate_execution"
                                    if config.get("vertical_reachability_validation", False)
                                    else "endpoints_only_original_controller_must_validate_path"),
            })
            motion_time = getattr(env, 'candidate_joint_motion_time', None)
            if callable(motion_time):
                accepted[-1]['nominal_joint_motion_time_s'] = motion_time(
                    intercept_ik.joint_positions if direct else pregrasp_ik.joint_positions)
    if config.get('compare_parallel_jaw_variants', False):
        # Only compare already allowed, fully checked pi-symmetric versions
        # of a network grasp. Preserve score/source priority across grasps.
        accepted.sort(key=lambda row: (row.get('task_preference', {}).get('group', 2),
                                      -row['score'], row['source_index'],
                                      row.get('nominal_joint_motion_time_s', float('inf'))))
        accepted = accepted[:max_accepted]
    if pooled:
        env.candidate_selection_diagnostics.update({
            'evaluated_directions': ik_count,
            'original_batch_size': max_ik,
            'pool_batches_visited': (ik_count + max_ik - 1) // max_ik,
            'acceptance_requires_original_checks': True,
            'reused_network_candidates': True,
            'geometrically_rejected_directions': len(geometric_rejections),
            'remaining_directions_after_selection': sum(
                row['reason'] == 'not_evaluated_after_selection' for row in rejected)})
    return accepted, rejected

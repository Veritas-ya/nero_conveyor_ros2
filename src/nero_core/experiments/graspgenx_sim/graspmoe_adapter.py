"""Measured-support-frame GraspMoE adapter, without a second diffusion pass.

The geometry and merge helpers import only NumPy.  Official GraspGen-X modules
and torch are loaded only by the inference functions after the backend is ready.
Object points keep their measured metric scale and original input coordinate
frame; the support frame is used only to construct the additional top grasps.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


# The official object/scene demo defaults (demo_object_pc.py:160-163), not the
# wider depth sweep of the lower-level run_graspmoe convenience function.
TOP_Z_OFFSETS_M = (-0.02, 0.0)
TOP_YAWS_RAD = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)


def validate_support_frame(value: Any) -> np.ndarray:
    """Return T_input_support; columns are the support axes in the input frame."""
    frame = np.asarray(value, dtype=np.float64)
    if frame.shape != (4, 4) or not np.all(np.isfinite(frame)):
        raise ValueError("support_frame must be a finite 4x4 T_input_support")
    if not np.allclose(frame[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0):
        raise ValueError("support_frame must have a homogeneous last row")
    rotation = frame[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("support_frame rotation must be right-handed and orthonormal")
    return frame.copy()


def candidate_preference(
    pose: np.ndarray,
    *,
    group: int,
    support_frame: np.ndarray | None,
    obb: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe measured geometry; angles use undirected closing-axis lines."""
    result: dict[str, Any] = {
        "group": int(group),
        "side_alignment_deg": None,
        "approach_tilt_deg": None,
        "closing_span_m": None,
    }
    pose = np.asarray(pose, dtype=np.float64)
    closing = pose[:3, 0]
    if support_frame is not None:
        normal = support_frame[:3, 2]
        approach_cos = np.clip(float(-pose[:3, 2] @ normal), -1.0, 1.0)
        result["approach_tilt_deg"] = float(np.degrees(np.arccos(approach_cos)))
    if obb is not None:
        rotation = np.asarray(obb["R"], dtype=np.float64)
        half = np.asarray(obb["half_extent"], dtype=np.float64)
        components = rotation.T @ closing
        result["closing_span_m"] = float(2.0 * (np.abs(components) @ half))
        # Alignment to the nearest of the two top-face edge lines: 0 means
        # closing parallel to an edge, rather than diagonally across the face.
        projection_norm = float(np.linalg.norm(components[:2]))
        if projection_norm > 1e-12:
            best_cos = np.clip(
                float(np.max(np.abs(components[:2])) / projection_norm), 0.0, 1.0,
            )
            result["side_alignment_deg"] = float(np.degrees(np.arccos(best_cos)))
    return result


def build_top_obb_candidates(
    points: np.ndarray,
    support_frame: Any,
    *,
    gripper_depth_m: float,
    gripper_width_m: float,
    depth_offsets_m=None,
    yaw_offsets_rad=None,
    compute_obb: Callable[..., Any] | None = None,
    build_face_candidates: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    """Use official OBB fit and face sampler at four edge-aligned orientations.

    Input/output poses use the caller's measured frame.  No point is added to
    the network input and no point-cloud filtering is performed by this helper.
    The official OBB fit may use its own SOR for its geometric fit only.
    """
    frame = validate_support_frame(support_frame)
    depths=np.asarray(TOP_Z_OFFSETS_M if depth_offsets_m is None else depth_offsets_m,dtype=float)
    yaws=np.asarray(TOP_YAWS_RAD if yaw_offsets_rad is None else yaw_offsets_rad,dtype=float)
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in (depths,yaws)):
        raise ValueError('top grasp sampling requires finite nonempty one-dimensional offsets')
    if compute_obb is None or build_face_candidates is None:
        from graspgenx.samplers.graspmoe import _compute_obb, _build_face_candidates
        compute_obb = compute_obb or _compute_obb
        build_face_candidates = build_face_candidates or _build_face_candidates
    points = np.asarray(points, dtype=np.float32)
    local_points = ((points - frame[:3, 3]) @ frame[:3, :3]).astype(np.float32)
    started = time.perf_counter()
    center, half, rotation = compute_obb(local_points, mode="advanced")
    fit_s = time.perf_counter() - started
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if (center.shape != (3,) or half.shape != (3,) or rotation.shape != (3, 3)
            or not np.all(np.isfinite(np.r_[center, half, rotation.ravel()]))
            or np.any(half < 0)):
        raise ValueError("official OBB fit returned invalid geometry")
    short_axis = int(np.argmin(half[:2]))
    top_z = float(center[2] + np.abs(rotation[2]) @ half)
    started = time.perf_counter()
    local_poses = build_face_candidates(
        face_origin_world=np.array([center[0], center[1], top_z]),
        approach_dir_world=np.array([0.0, 0.0, 1.0]),
        in_plane_axis_world=rotation[:, short_axis],
        positions_local=np.array([0.0]),
        yaws=yaws,
        z_offsets_m=depths,
        gripper_depth_m=float(gripper_depth_m),
    )
    poses = frame[None] @ np.asarray(local_poses, dtype=np.float64)
    obb = {
        "center": (frame[:3, :3] @ center + frame[:3, 3]).tolist(),
        "R": (frame[:3, :3] @ rotation).tolist(),
        "half_extent": half.tolist(),
        "support_frame": frame.tolist(),
        "center_support": center.tolist(),
        "R_support": rotation.tolist(),
        "short_axis_index": short_axis,
        "depth_offsets_m": depths.tolist(),
        "yaw_offsets_rad": yaws.tolist(),
    }
    preferences = []
    kept_poses = []
    rejected_stroke = 0
    for index, pose in enumerate(poses):
        # Four yaw directions, each with both official demo depth offsets.
        yaw=float(yaws[(index//len(depths))%len(yaws)])
        group=0 if abs(np.sin(yaw))<1e-8 else 1
        preference = candidate_preference(
            pose, group=group, support_frame=frame, obb=obb,
        )
        preference.update(top_depth_offset_m=float(depths[index%len(depths)]),
            top_yaw_offset_rad=yaw)
        if preference["closing_span_m"] > float(gripper_width_m) + 1e-9:
            rejected_stroke += 1
            continue
        preferences.append(preference)
        kept_poses.append(pose)
    generation_s = time.perf_counter() - started
    return {
        "poses": np.asarray(kept_poses, dtype=np.float64).reshape(-1, 4, 4),
        "preferences": preferences,
        "obb": obb,
        "generated_candidates": len(poses),
        "stroke_rejected_candidates": rejected_stroke,
        "timing_s": {"obb_fit_s": fit_s, "obb_candidate_generation_s": generation_s},
    }


def merge_candidates(
    diffusion_poses: np.ndarray,
    diffusion_scores: np.ndarray,
    obb_poses: np.ndarray,
    obb_scores: np.ndarray,
    obb_preferences: list[dict[str, Any]],
    *,
    topk: int,
    support_frame: np.ndarray | None = None,
    obb: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict budget, stable priority order; model scores are never rewritten."""
    diffusion_poses = np.asarray(diffusion_poses, dtype=np.float64).reshape(-1, 4, 4)
    obb_poses = np.asarray(obb_poses, dtype=np.float64).reshape(-1, 4, 4)
    diffusion_scores = np.asarray(diffusion_scores, dtype=np.float64).reshape(-1)
    obb_scores = np.asarray(obb_scores, dtype=np.float64).reshape(-1)
    if (len(diffusion_poses) != len(diffusion_scores)
            or len(obb_poses) != len(obb_scores)
            or len(obb_poses) != len(obb_preferences)):
        raise ValueError("poses, model scores, and candidate preferences must align")
    if int(topk) < 1:
        raise ValueError("topk must be positive")
    poses = np.concatenate([diffusion_poses, obb_poses])
    scores = np.concatenate([diffusion_scores, obb_scores])
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores)):
        raise ValueError("candidate poses and model scores must be finite")
    sources = ["diffusion"] * len(diffusion_poses) + ["obb_top"] * len(obb_poses)
    preferences = [
        candidate_preference(pose, group=2, support_frame=support_frame, obb=obb)
        for pose in diffusion_poses
    ] + [dict(item) for item in obb_preferences]
    order = sorted(
        range(len(poses)),
        key=lambda index: (preferences[index]["group"], -scores[index], index),
    )[:int(topk)]
    return {
        "poses": poses[order],
        "scores": scores[order],
        "candidate_sources": [sources[index] for index in order],
        "candidate_task_preferences": [preferences[index] for index in order],
        "candidate_original_indices": order,
        "candidate_source_indices": [
            index if index < len(diffusion_poses) else index - len(diffusion_poses)
            for index in order
        ],
    }


def sample_diffusion_with_embedding(
    points: np.ndarray, sampler: Any, torch: Any, num_grasps: int,
) -> tuple[Any, Any, Any, Any, np.ndarray]:
    """Mirror official single-item sample, returning its discriminator feature.

    Unlike run_inference_batch this keeps every supplied point, its original
    centering, and the original gripper-input RNG calls.  The generator and
    discriminator run in the same single model.infer invocation as sample().
    """
    from graspgenx.dataset.dataset import collate

    with torch.inference_mode():
        object_pc = torch.from_numpy(np.ascontiguousarray(points)).cuda().float()
        center = object_pc.mean(axis=0)
        color = torch.zeros_like(object_pc)
        centered = object_pc - center[None]
        data = {
            "task": "pick",
            "inputs": torch.cat([centered, color[:, :3].squeeze(1)], dim=-1).float(),
            "points": centered,
        }
        data = sampler.load_gripper_input(data)
        data_batch = collate([data])
        sampler.model.grasp_generator.num_grasps_per_object = int(num_grasps)
        outputs, _, _ = sampler.model.infer(data_batch)
        poses = outputs["grasps_pred"][0]
        scores = outputs["grasp_confidence"][0][:, 0]
        mask = scores >= -1.0
        poses, scores = poses[mask], scores[mask]
        poses[:, :3, 3] += center
        poses[:, 3, 3] = 1.0
        embedding = outputs.get("object_embedding")
        if embedding is not None:
            embedding = embedding[0]
        return poses, scores, embedding, centered, center.detach().cpu().numpy()

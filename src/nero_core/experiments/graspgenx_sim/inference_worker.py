#!/usr/bin/env python3
"""Persistent adapter for the official GraspGen-X network and GraspMoE sampler.

Input: segmented object points in metres, in one caller-selected frame.
Output: T_input_gripper in the AGX canonical frame (+Z approach, +X closing).
Optional GraspMoE adds official OBB top-face samples in a measured support
frame, scored by the same network.  This module does not import robot drivers,
sample ground-truth grasps, or claim that a network score is grasp success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SOURCE_REVISION = "b9429097728cb1c430dd78b92edf17ba318aad03"
WEIGHTS_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
WEIGHT_FILES = {
    "gen/epoch_736.pth": (
        1210918342,
        "8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44",
    ),
    "dis/epoch_1056.pth": (
        483889478,
        "cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_assets(checkpoint_dir: Path, verify_hashes: bool = True) -> dict[str, Any]:
    """Fail explicitly on incomplete downloads and Git LFS pointer files."""
    verified = {}
    for relative, (expected_size, expected_hash) in WEIGHT_FILES.items():
        path = checkpoint_dir / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            actual = path.stat().st_size if path.is_file() else None
            raise FileNotFoundError(
                f"Official weight missing/incomplete: {path}; "
                f"expected {expected_size} bytes, got {actual}."
            )
        if verify_hashes and _sha256(path) != expected_hash:
            raise ValueError(f"Official checkpoint SHA256 mismatch: {path}")
        verified[relative] = {
            "bytes": expected_size,
            "sha256": expected_hash,
            "hash_verified_this_load": verify_hashes,
        }
    for relative in ("gen/config.yaml", "dis/config.yaml"):
        if not (checkpoint_dir / relative).is_file():
            raise FileNotFoundError(checkpoint_dir / relative)
    return verified


class GraspGenXBackend:
    """Load once; keep historical diffusion or add official GraspMoE top grasps.

    The official release conditions solely on the gripper sweep-volume boxes.
    Its synthetic placeholder meshes are NEVER used here for collision tests;
    collision/IK/execution belong to the caller's actual NERO/AGX simulation.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        gripper_params_path: str | Path | None = None,
        seed: int = 0,
        verify_weights: bool = True,
        max_points: int = 3500,
        planner: str = "diffusion",
    ) -> None:
        started = time.perf_counter()
        if planner not in ("diffusion", "graspmoe"):
            raise ValueError("planner must be 'diffusion' or 'graspmoe'")
        self.planner = str(planner)
        self.checkpoint_dir = Path(
            checkpoint_dir or os.environ.get("NERO_GRASPGENX_CHECKPOINT_DIR") or EXPERIMENT_ROOT / "checkpoints" / "release"
        ).resolve()
        self.gripper_params_path = Path(
            gripper_params_path
            or EXPERIMENT_ROOT / "gripper_agx" / "sweep_volume_params.json"
        ).resolve()
        self.weights = check_assets(self.checkpoint_dir, verify_weights)
        raw_params = json.loads(self.gripper_params_path.read_text())
        names = (
            "extents_open", "offset_open", "extents_mid", "offset_mid",
            "gripper_type", "fingertip_depth",
        )
        self.params = {name: raw_params[name] for name in names if name in raw_params}
        if "fingertip_depth" not in self.params:
            raise ValueError("AGX config must state its measured model fingertip_depth.")

        source = EXPERIMENT_ROOT / "vendor" / "GraspGenX"
        revision_path = source / "UPSTREAM_COMMIT.txt"
        if not revision_path.is_file() or revision_path.read_text().strip() != SOURCE_REVISION:
            raise ValueError(f"Unexpected official GraspGenX source revision at {source}")
        sys.path.insert(0, str(source))
        # Explicit local paths avoid the upstream first-import git download hook.
        os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(self.checkpoint_dir.parent)
        os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(
            EXPERIMENT_ROOT / "vendor" / "gripper_descriptions"
        )

        import torch
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Official GraspGenX backend requires CUDA; no geometry fallback is enabled."
            )
        self.torch = torch
        self.seed = int(seed)
        self.max_points = int(max_points)
        if self.max_points < 21:
            raise ValueError("max_points must be at least 21.")
        self.calls = 0
        self._lock = threading.Lock()
        self._sampler_class = GraspGenXSampler
        self._set_seed(self.seed)
        cfg = load_model_cfg(
            str(self.checkpoint_dir / "gen"),
            str(self.checkpoint_dir / "dis"),
            "epoch_736.pth", "epoch_1056.pth",
        )
        # Retain the released architecture and its 20 denoising steps.
        for side in (cfg.diffusion, cfg.discriminator):
            if str(side.object_backbone) != "ptv3vanilla":
                raise ValueError("Expected official release ptv3vanilla checkpoint.")
            if str(side.gripper_backbone) != "sweep_volume_v2":
                raise ValueError("Expected official release sweep_volume_v2 conditioning.")
        self.sampler = GraspGenXSampler.from_sweep_volume(cfg, self.params)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.provenance = {
            "algorithm": "GraspGenX",
            "planner": self.planner,
            "source_repository": "https://github.com/NVlabs/GraspGenX",
            "source_revision": SOURCE_REVISION,
            "weights_repository": "https://huggingface.co/adithyamurali/GraspGenXModel",
            "weights_revision": WEIGHTS_REVISION,
            "weights": self.weights,
            "gripper_params": self.params,
            "gripper_params_sha256": _sha256(self.gripper_params_path),
            "diffusion_steps": int(cfg.diffusion.num_diffusion_iters_eval),
            "input_units": "metres",
            "output_transform": "T_input_frame_gripper_canonical",
            "gripper_frame": "+Z approach; +X closing; origin from AGX config",
            "score_is_success_probability": False,
            "uses_ground_truth_object_pose": False,
            "uses_obb_grasp_candidates": self.planner == "graspmoe",
            "graspmoe_adapter": "measured_support_top_edge_aligned_v1"
            if self.planner == "graspmoe" else None,
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "load_seconds": self.load_seconds,
        }

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def infer(
        self,
        point_cloud_world_m: np.ndarray,
        seed: int | None = None,
        num_grasps: int = 100,
        topk: int = 100,
        *,
        support_frame: np.ndarray | None = None,
        obb_depth_offsets_m=None,
        obb_yaw_offsets_rad=None,
    ) -> dict[str, Any]:
        """Return poses/scores and provenance; preserve the caller input frame.

        Point sets above max_points are reproducibly subsampled and logged.
        The adapter never normalizes object scale, guesses an object category,
        or injects hidden surface points.  Optional support_frame is the 4x4
        T_input_support, with its +Z normal pointing away from the support.
        """
        points = np.asarray(point_cloud_world_m, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected point cloud [N,3], got {points.shape}")
        if len(points) < 21 or not np.all(np.isfinite(points)):
            raise ValueError("Need at least 21 finite object points in metres.")
        if int(num_grasps) < 1 or int(topk) < 1:
            raise ValueError("num_grasps and topk must be positive.")
        with self._lock:
            run_seed = self.seed + self.calls if seed is None else int(seed)
            self._set_seed(run_seed)
            input_point_count = len(points)
            input_points_sha256 = hashlib.sha256(
                np.ascontiguousarray(points).tobytes()
            ).hexdigest()
            if len(points) > self.max_points:
                indices = np.random.default_rng(run_seed).choice(
                    len(points), self.max_points, replace=False
                )
                points = points[indices]
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            diagnostic_start_ns = time.perf_counter_ns()
            stage_timing = {}
            obb_metadata = None
            obb_skip_reason = None
            obb_generated_count = 0
            obb_stroke_rejected_count = 0
            obb_scored_count = 0
            object_embedding_reused = False
            frame = None
            obb_result = None
            obb_scores = np.zeros(0, dtype=np.float64)
            if self.planner == "graspmoe" and support_frame is not None:
                from experiments.graspgenx_sim.graspmoe_adapter import (
                    sample_diffusion_with_embedding, validate_support_frame,
                )
                try:
                    frame = validate_support_frame(support_frame)
                except (TypeError, ValueError) as error:
                    obb_skip_reason = f"invalid_support_frame: {error}"
            elif self.planner == "graspmoe":
                obb_skip_reason = "support_frame_missing"
            if self.planner == "graspmoe" and frame is not None:
                poses_t, scores_t, embedding, centered, center = sample_diffusion_with_embedding(
                    np.ascontiguousarray(points), self.sampler, self.torch, int(num_grasps),
                )
            else:
                # Preserve the original path and its point/RNG contract by default.
                poses_t, scores_t = self._sampler_class.run_inference(
                    np.ascontiguousarray(points), self.sampler,
                    grasp_threshold=-1.0,
                    num_grasps=int(num_grasps),
                    topk_num_grasps=int(topk),
                    min_grasps=1,
                    max_tries=1,
                    remove_outliers=False,
                )
            self.torch.cuda.synchronize()
            stage_timing["diffusion_and_discriminator_s"] = time.perf_counter() - started
            if len(poses_t):
                poses = poses_t.detach().cpu().numpy().astype(np.float64)
                scores = scores_t.detach().cpu().numpy().astype(np.float64)
                if self.planner != "graspmoe" or frame is None:
                    order = np.argsort(-scores)
                    poses, scores = poses[order], scores[order]
            else:
                poses = np.zeros((0, 4, 4), dtype=np.float64)
                scores = np.zeros((0,), dtype=np.float64)
            candidate_sources = ["diffusion"] * len(poses)
            candidate_original_indices = list(range(len(poses)))
            candidate_source_indices = list(range(len(poses)))
            diffusion_candidate_count = len(poses)
            candidate_preferences = [{
                "group": 2, "side_alignment_deg": None,
                "approach_tilt_deg": None, "closing_span_m": None,
            } for _ in poses]
            if self.planner == "graspmoe" and frame is not None:
                from experiments.graspgenx_sim.graspmoe_adapter import (
                    build_top_obb_candidates, merge_candidates,
                )
                from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator
                try:
                    if embedding is None:
                        obb_skip_reason = "diffusion_discriminator_embedding_missing"
                    else:
                        obb_result = build_top_obb_candidates(
                            points, frame,
                            gripper_depth_m=float(self.sampler.gripper.depth),
                            gripper_width_m=float(self.sampler.gripper.sweep_volume[0]),
                            depth_offsets_m=obb_depth_offsets_m,
                            yaw_offsets_rad=obb_yaw_offsets_rad,
                        )
                        obb_metadata = obb_result["obb"]
                        stage_timing.update(obb_result["timing_s"])
                        obb_generated_count = obb_result["generated_candidates"]
                        obb_stroke_rejected_count = obb_result["stroke_rejected_candidates"]
                        score_started = time.perf_counter()
                        obb_scores = _score_grasps_with_discriminator(
                            obb_result["poses"], centered, center, self.sampler,
                            object_embedding=embedding,
                        )
                        self.torch.cuda.synchronize()
                        stage_timing["obb_discriminator_s"] = time.perf_counter() - score_started
                        if (len(obb_scores) != len(obb_result["poses"])
                                or not np.all(np.isfinite(obb_scores))):
                            raise ValueError("official OBB scorer returned invalid scores")
                        obb_scored_count = len(obb_scores)
                        object_embedding_reused = bool(obb_scored_count)
                        if not obb_scored_count:
                            obb_skip_reason = "obb_closing_span_exceeds_gripper_stroke"
                except Exception as error:
                    # OBB is an additional proposal source.  Its failure must
                    # not discard the original diffusion candidates.
                    obb_skip_reason = f"obb_branch_failed: {type(error).__name__}: {error}"
                    obb_result = None
                    obb_scores = np.zeros(0, dtype=np.float64)
                merge_started = time.perf_counter()
                merged = merge_candidates(
                    poses, scores,
                    obb_result["poses"] if obb_result is not None else np.zeros((0, 4, 4)),
                    obb_scores,
                    obb_result["preferences"] if obb_result is not None else [],
                    topk=int(topk), support_frame=frame, obb=obb_metadata,
                )
                poses, scores = merged["poses"], merged["scores"]
                candidate_sources = merged["candidate_sources"]
                candidate_preferences = merged["candidate_task_preferences"]
                candidate_original_indices = merged["candidate_original_indices"]
                candidate_source_indices = merged["candidate_source_indices"]
                stage_timing["candidate_merge_s"] = time.perf_counter() - merge_started
            elapsed = time.perf_counter() - started
            if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores)):
                raise RuntimeError("Official network returned non-finite candidates.")
            self.calls += 1
            metadata = {
                **self.provenance,
                "seed": run_seed,
                "input_point_count": input_point_count,
                "input_points_sha256": input_points_sha256,
                "point_count": len(points),
                "network_points_sha256": hashlib.sha256(
                    np.ascontiguousarray(points).tobytes()
                ).hexdigest(),
                "point_subsampling": "seeded uniform without replacement"
                if input_point_count > len(points) else "none",
                "requested_candidates": int(num_grasps),
                "topk": int(topk),
                "grasp_threshold": -1.0,
                "network_attempts": 1,
                "network_outlier_removal": False,
                "candidate_sources": candidate_sources,
                "candidate_task_preferences": candidate_preferences,
                "candidate_original_indices": candidate_original_indices,
                "candidate_source_indices": candidate_source_indices,
                "diffusion_candidate_count": diffusion_candidate_count,
                "obb": obb_metadata,
                "obb_sampling_source": ("official_demo_defaults" if obb_depth_offsets_m is None
                    and obb_yaw_offsets_rad is None else "explicit_local_top_sampling_scored_by_original_discriminator"),
                "obb_skip_reason": obb_skip_reason,
                "obb_generated_candidates": obb_generated_count,
                "obb_stroke_rejected_candidates": obb_stroke_rejected_count,
                "obb_scored_candidates": obb_scored_count,
                "object_embedding_reused": object_embedding_reused,
                "uses_obb_grasp_candidates": "obb_top" in candidate_sources,
                "effective_planner": "graspmoe" if "obb_top" in candidate_sources else "diffusion",
                "diagnostic_interval": {"pid": os.getpid(), "native_thread_id": threading.get_native_id(),
                    "perf_start_ns": diagnostic_start_ns, "perf_end_ns": time.perf_counter_ns(),
                    "clock": "perf_counter; original CUDA synchronization retained"},
                "stage_timing_s": stage_timing,
                "returned_candidates": len(poses),
                "inference_seconds": elapsed,
                "timing_s": elapsed,
                "peak_cuda_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            }
            return {"poses": poses, "scores": scores, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="NPY or NPZ: points/pc/point_cloud")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--gripper-params", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-grasps", type=int, default=100)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    loaded = np.load(args.input, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = [key for key in ("points", "pc", "point_cloud") if key in loaded]
        if not keys:
            raise ValueError("NPZ requires points, pc, or point_cloud [N,3] in metres.")
        points = loaded[keys[0]]
    else:
        points = loaded
    backend = GraspGenXBackend(args.checkpoint_dir, args.gripper_params, args.seed)
    result = backend.infer(points, args.seed, args.num_grasps, args.topk)
    result["metadata"]["input_file"] = str(args.input.resolve())
    result["metadata"]["input_file_sha256"] = _sha256(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, poses=result["poses"], scores=result["scores"])
    args.output.with_suffix(".json").write_text(
        json.dumps(result["metadata"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()

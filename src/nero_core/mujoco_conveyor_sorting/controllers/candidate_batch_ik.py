"""Optional batch IK hints for selecting unchanged network grasps.

The GPU solves kinematics only. Every solution is checked against the exact
current MuJoCo TCP and the caller's bounds; scene/self/object checks remain in
the existing candidate checker. Missing hints mean unknown, never unreachable.
"""
from __future__ import annotations

import atexit
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from ._math import quat_distance
from .curobo_planner import export_mujoco_chain_urdf
from .ik_solver import DampedLeastSquaresIK, IKResult


@dataclass(frozen=True)
class BatchIKHints:
    """Numerically valid pose solutions, awaiting original geometry checks."""

    solutions: tuple[tuple[IKResult, ...], ...]
    joint_distances: tuple[float, ...]
    diagnostics: dict[str, Any]

    def ordering_key(self, index: int) -> tuple[bool, float]:
        """Prefer a known solution; lack of one is only an ordering hint."""
        return not bool(self.solutions[index]), self.joint_distances[index]


def candidate_ik_seeds(start_q, lower_limits, upper_limits, extra_seeds=(), *,
                       maximum_seeds: int = 4, excluded_seeds=()) -> np.ndarray:
    """Distinct bounded starts: supplied history, joint centre, elbow variants.

    These are solver initial guesses, not new commanded joint positions.
    """
    lower, upper = np.asarray(lower_limits, float), np.asarray(upper_limits, float)
    start = np.asarray(start_q, float)
    if (start.shape != lower.shape or lower.shape != upper.shape or start.ndim != 1
            or not np.isfinite(np.r_[start, lower, upper]).all()
            or np.any(lower >= upper) or maximum_seeds < 1):
        raise ValueError("require finite ordered joint bounds and positive seed budget")
    center = (lower + upper) * .5
    candidates = [start, *extra_seeds, center]
    # Move different redundant joints in opposite directions. A second seed
    # must not be the same clipped current pose under another name.
    pattern = np.resize(np.array([0., .16, .12, -.16, -.12, .08, 0.]), start.shape)
    candidates += [center + pattern * (upper - lower), center - pattern * (upper - lower)]
    excluded = []
    for seed in excluded_seeds:
        seed = np.asarray(seed, float)
        if seed.shape != start.shape or not np.isfinite(seed).all():
            raise ValueError("excluded IK seeds must match the finite arm joint vector")
        excluded.append(np.clip(seed, lower, upper))
    unique = []
    for seed in candidates:
        seed = np.asarray(seed, float)
        if seed.shape != start.shape or not np.isfinite(seed).all():
            raise ValueError("every IK seed must match the finite arm joint vector")
        seed = np.clip(seed, lower, upper)
        if any(np.max(np.abs(seed - old)) < 1e-9 for old in excluded):
            continue
        if not any(np.max(np.abs(seed - old)) < 1e-9 for old in unique):
            unique.append(seed)
        if len(unique) >= maximum_seeds:
            break
    return np.asarray(unique).reshape(-1, len(start))


class CandidateBatchIK:
    """Own private FK/DLS data while optionally sharing one prewarmed GPU service.

    All poses use this solver's world coordinates and XYZ + WXYZ. A belt-frame
    adapter must transform poses before calling. Bounds are intersected with
    the existing numerical solver limits, including the stage's joint margin.
    """

    def __init__(self, solver: DampedLeastSquaresIK, gpu_backend=None):
        import mujoco
        self.source_solver = solver
        self.model = copy.copy(solver.model)
        self.data = mujoco.MjData(self.model)
        self.solver = DampedLeastSquaresIK(self.model, self.data, solver.joint_names,
                                          solver.ee_site_name, copy.deepcopy(solver.config))
        self.gpu_backend = gpu_backend

    def solve(self, target_poses, *, qpos_seed, lower_limits=None, upper_limits=None,
              extra_seeds=(), cpu_fallback: bool = False, maximum_solutions: int = 1,
              maximum_cpu_seeds: int = 4, excluded_seeds=(),
              cancelled: Callable[[], bool] = lambda: False
              ) -> BatchIKHints:
        import mujoco
        started = time.perf_counter()
        targets = np.asarray(target_poses, float).copy()
        dof = self.solver.dof
        start = np.asarray(qpos_seed, float)
        lower = np.array(self.source_solver.lower_limits, copy=True)
        upper = np.array(self.source_solver.upper_limits, copy=True)
        if lower_limits is not None:
            lower = np.maximum(lower, np.asarray(lower_limits, float))
        if upper_limits is not None:
            upper = np.minimum(upper, np.asarray(upper_limits, float))
        if (targets.ndim != 2 or targets.shape[1:] != (7,)
                or start.shape != (dof,) or lower.shape != (dof,) or upper.shape != (dof,)
                or not np.isfinite(np.r_[targets.ravel(), start, lower, upper]).all()
                or np.any(lower >= upper) or maximum_solutions < 1 or maximum_cpu_seeds < 1):
            raise ValueError("require finite [N,XYZ+WXYZ], arm seed and nonempty joint bounds")
        lengths = np.linalg.norm(targets[:, 3:], axis=1)
        if np.any(lengths < 1e-12):
            raise ValueError("candidate quaternions must have nonzero length")
        targets[:, 3:] /= lengths[:, None]
        self.data.qpos[:] = self.source_solver.data.qpos
        self.solver.lower_limits, self.solver.upper_limits = lower, upper
        self.solver.center_positions = .5 * (lower + upper)
        full = np.asarray(self.data.qpos).copy()
        seeds = candidate_ik_seeds(start, lower, upper, extra_seeds,
                                   maximum_seeds=maximum_cpu_seeds,
                                   excluded_seeds=excluded_seeds)
        answers: list[list[IKResult]] = [[] for _ in targets]
        info: dict[str, Any] = {"target_count": len(targets), "cpu_fallback": cpu_fallback,
            "cpu_attempts": 0, "gpu_returned_solutions": 0, "invalid_bounds": 0,
            "fk_mismatches": 0, "certification": "kinematics_only_original_geometry_checks_required",
            "missing_solution_means": "unknown_not_unreachable"}

        def accept(index, joints, iterations=0, source="batch_gpu"):
            q = np.asarray(joints, float)
            if q.shape != (dof,) or not np.isfinite(q).all():
                info["fk_mismatches"] += 1
                return
            if np.any(q < lower) or np.any(q > upper):
                info["invalid_bounds"] += 1
                return
            if any(np.max(np.abs(q - old.joint_positions)) < 1e-7 for old in answers[index]):
                return
            self.data.qpos[:] = full
            self.data.qpos[self.solver.qpos_indices] = q
            # FK is needed, contacts/dynamics are still checked by the ordinary
            # measured-scene candidate validator after this ordering stage.
            mujoco.mj_kinematics(self.model, self.data)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.solver.ee_site_id])
            position_error = float(np.linalg.norm(self.data.site_xpos[self.solver.ee_site_id]
                                                   - targets[index, :3]))
            orientation_error = quat_distance(targets[index, 3:], quat)
            if (position_error > self.solver.config.position_tolerance
                    or orientation_error > self.solver.config.orientation_tolerance):
                info["fk_mismatches"] += 1
                return
            result_full = full.copy()
            result_full[self.solver.qpos_indices] = q
            answers[index].append(IKResult(q.copy(), result_full, True, int(iterations),
                                           position_error, orientation_error, source))

        if self.gpu_backend is not None and len(targets) and not cancelled():
            try:
                proposals = self.gpu_backend.solve(targets, qpos_seed=start, seeds=seeds,
                    lower_limits=lower, upper_limits=upper, cancelled=cancelled)
                for item in proposals:
                    index = int(item["pose_index"])
                    if 0 <= index < len(targets):
                        info["gpu_returned_solutions"] += 1
                        accept(index, item["q"])
                info["gpu"] = dict(self.gpu_backend.diagnostics)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                info["gpu_error"] = f"{type(error).__name__}: {error}"
        if cpu_fallback:
            # Explicitly requested small batches only. GPU-only hint preparation
            # never fans out every network candidate into hundreds of DLS calls.
            for index, pose in enumerate(targets):
                for seed in seeds:
                    if cancelled() or len(answers[index]) >= maximum_solutions:
                        break
                    result = self.solver.solve(pose[:3], pose[3:], qpos_seed=seed,
                                                cancelled=cancelled)
                    info["cpu_attempts"] += 1
                    if result.success:
                        accept(index, result.joint_positions, result.iterations, "batch_cpu_multiseed")
        distances = []
        span = upper - lower
        for found in answers:
            found.sort(key=lambda result: float(np.linalg.norm((result.joint_positions-start)/span)))
            del found[maximum_solutions:]
            distances.append(float(np.linalg.norm((found[0].joint_positions-start)/span))
                             if found else float("inf"))
        info.update(solved_targets=sum(bool(found) for found in answers),
                    elapsed_s=time.perf_counter()-started)
        return BatchIKHints(tuple(tuple(found) for found in answers), tuple(distances), info)


class CuRoboBatchIKBackend:
    """Persistent optional IK-only process, using the existing cached cuRobo env.

    Fixed-size batches are padded and warmed once, so varying network candidate
    counts do not recapture CUDA graphs. No MotionGen/world map is duplicated.
    """

    def __init__(self, model, *, joint_names: Sequence[str], ee_site_name: str,
                 python_executable=None, num_seeds: int = 16, batch_size: int = 16,
                 maximum_solutions: int = 4):
        import mujoco
        configured = python_executable or os.environ.get("NERO_CUROBO_PYTHON")
        self.python = Path(configured).expanduser() if configured else (
            Path.home()/".cache/nero-planning/venv/bin/python")
        self.urdf = export_mujoco_chain_urdf(model, joint_names, ee_site_name)
        ids = np.asarray([model.joint(name).id for name in joint_names])
        ranges = np.asarray(model.jnt_range[ids], float)
        warm_q = ranges.mean(axis=1)
        warm_data = mujoco.MjData(model)
        warm_data.qpos[np.asarray(model.jnt_qposadr[ids], int)] = warm_q
        mujoco.mj_kinematics(model, warm_data)
        warm_quat = np.empty(4)
        site = int(model.site(ee_site_name).id)
        mujoco.mju_mat2Quat(warm_quat, warm_data.site_xmat[site])
        warm_pose = np.r_[warm_data.site_xpos[site], warm_quat]
        self._warmup_message = {"target_poses": [warm_pose.tolist()],
            "start_q": warm_q.tolist(),
            "seeds": candidate_ik_seeds(warm_q, ranges[:, 0], ranges[:, 1]).tolist(),
            "lower_limits": ranges[:, 0].tolist(), "upper_limits": ranges[:, 1].tolist()}
        self.num_seeds, self.batch_size = int(num_seeds), int(batch_size)
        self.maximum_solutions = int(maximum_solutions)
        if min(self.num_seeds, self.batch_size, self.maximum_solutions) < 1:
            raise ValueError("batch and seed budgets must be positive")
        self.maximum_solutions = min(self.maximum_solutions, self.num_seeds)
        self._lock = threading.Lock()
        self._process, self._temporary, self._stderr = None, None, None
        self._disabled_reason = None
        self.diagnostics = {"backend": "curobo_batch_multiseed_kinematics", "available": False,
                            "model_urdf_sha256": hashlib.sha256(self.urdf.encode()).hexdigest()}
        atexit.register(self.close)

    def _read(self, cancelled):
        process = self._process
        if process is None:
            return None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while process.poll() is None:
                if cancelled():
                    self.close()
                    return None
                if selector.select(.025):
                    line = process.stdout.readline()
                    if not line:
                        return None
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return None

    def warmup(self, cancelled=lambda: False):
        with self._lock:
            if self._disabled_reason:
                return False
            if self._process is not None:
                return self._process.poll() is None
            if not self.python.is_file():
                self._disabled_reason = "isolated cuRobo Python is not installed: " + str(self.python)
                self.diagnostics.update(available=False, unavailable_reason=self._disabled_reason)
                return False
            started = time.perf_counter()
            self._temporary = tempfile.TemporaryDirectory(prefix="nero-batch-ik-")
            folder = Path(self._temporary.name)
            (folder/"robot.urdf").write_text(self.urdf)
            self._stderr = (folder/"service.log").open("w")
            environment = os.environ.copy()
            toolkit = Path.home()/".cache/nero-planning/toolchain/usr/local/cuda-12.1"
            if toolkit.is_dir():
                environment.setdefault("CUDA_HOME", str(toolkit))
            environment.setdefault("MAX_JOBS", "2")
            environment.setdefault("PYTHONUNBUFFERED", "1")
            headers = Path.home()/".cache/nero-planning/python-dev/usr/include"
            if headers.is_dir():
                environment.setdefault("CPATH", str(headers)+":"+str(headers/"python3.10"))
            service = Path(__file__).with_name("_candidate_batch_ik_service.py")
            try:
                self._process = subprocess.Popen([str(self.python), str(service),
                    str(folder/"robot.urdf"), str(self.num_seeds), str(self.batch_size),
                    str(self.maximum_solutions)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._stderr, text=True, bufsize=1, env=environment)
                ready = self._read(cancelled)
                if ready is None or not ready.get("ready"):
                    raise RuntimeError((ready or {}).get("error", "batch IK process did not initialize"))
                # Exercise the actual request path too: explicit seeds, bound
                # tensor updates, padding, result copies and JSON RPC can have
                # one-time CUDA initialization beyond solve_batch capture.
                request_warmups = []
                for _ in range(2):
                    warm_started = time.perf_counter()
                    self._process.stdin.write(json.dumps(self._warmup_message)+"\n")
                    self._process.stdin.flush()
                    response = self._read(cancelled)
                    if response is None or response.get("error"):
                        raise RuntimeError((response or {}).get("error", "batch IK request warmup failed"))
                    request_warmups.append(time.perf_counter()-warm_started)
                self.diagnostics.update(available=True, runtime=ready)
                self.diagnostics["request_warmups_s"] = request_warmups
                return True
            except (OSError, RuntimeError) as error:
                self._disabled_reason = str(error)
                self.diagnostics.update(available=False, unavailable_reason=str(error))
                self.diagnostics["startup_log_tail"] = (folder/"service.log").read_text(
                    errors="replace").splitlines()[-12:]
                self.close()
                return False
            finally:
                self.diagnostics["startup_wall_s"] = time.perf_counter()-started

    def solve(self, targets, *, qpos_seed, seeds, lower_limits, upper_limits,
              cancelled=lambda: False):
        # Warmup belongs to device setup. An unwarmed or unavailable optional
        # backend supplies no hints instead of cold-loading on the first part.
        with self._lock:
            started = time.perf_counter()
            if self._process is None or self._process.poll() is not None or cancelled():
                return []
            try:
                message = {"target_poses": np.asarray(targets).tolist(),
                    "start_q": np.asarray(qpos_seed).tolist(), "seeds": np.asarray(seeds).tolist(),
                    "lower_limits": np.asarray(lower_limits).tolist(),
                    "upper_limits": np.asarray(upper_limits).tolist()}
                self._process.stdin.write(json.dumps(message, allow_nan=False)+"\n")
                self._process.stdin.flush()
                response = self._read(cancelled)
                if response is None or response.get("error"):
                    raise RuntimeError((response or {}).get("error", "batch IK process exited"))
                self.diagnostics.update(response.get("diagnostics", {}))
                return response.get("solutions", [])
            except (OSError, RuntimeError, ValueError) as error:
                self.diagnostics.update(available=False, unavailable_reason=str(error))
                self.close()
                return []
            finally:
                self.diagnostics["proposal_wall_s"] = time.perf_counter()-started

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

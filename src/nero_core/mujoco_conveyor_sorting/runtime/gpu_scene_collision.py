"""Reusable observed-point collision queries with optional CUDA broad phase.

The scene contains the original retained camera samples, in metres. It is not
an occupancy completion or an ESDF: a negative query says only that none of
these measured points intersects a queried convex body. Occluded space and
space outside the camera ROI remain unknown, exactly as in ``RobotModel``.

One batched KD-tree query excludes distant observed points. CUDA tests the
remaining body/point pairs together against the *complete* convex halfspaces.
It only removes bodies that are definitely disjoint. Possible intersections,
including a conservative floating-point boundary band, are resolved by the
original float64 cKDTree + convex halfspace calculation on the CPU. There is
no voxel quantisation, changed clearance, target-contact policy or new stop.

``auto`` keeps individual queries on the CPU: a GPU launch is not inherently
faster than one KD-tree query. CUDA is considered for batches of eight or more
poses with at least 8192 nearby point/body pairs (a backend cost choice, not a
motion acceptance condition). The optional torch/triton dependencies load only when needed;
unavailable CUDA or a kernel error falls back to the exact CPU calculation.
"""
from __future__ import annotations

import threading
import importlib.util
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree


class ObservedPointCollisionIndex:
    """Immutable native-point index, shareable across private robot mirrors.

    Geometry order is the insertion order of ``equations_by_geometry``. Query
    transforms/radii/margins/bounds follow this order; returned dictionaries
    use the original geometry IDs and native point indices. ``bounds`` are
    the existing padded-halfspace AABBs, not AABBs expanded by a guessed pad.
    """

    def __init__(self, points, equations_by_geometry: Mapping, *, backend="auto",
                 gpu_min_batch_size=8, tree=None, native_cpu_batch=True):
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("scene collision backend must be auto, cpu or cuda")
        self.points = np.array(points, dtype=np.float64, order="C", copy=True)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("scene points must be N x 3 metres")
        if not np.isfinite(self.points).all():
            raise ValueError("scene points must be finite")
        self.points.setflags(write=False)
        self.geometry_ids = tuple(equations_by_geometry)
        self.equations = tuple(np.array(equations_by_geometry[g], dtype=np.float64,
                                        order="C", copy=True)
                               for g in self.geometry_ids)
        for equations in self.equations:
            if equations.ndim != 2 or equations.shape[1] != 4 or not len(equations):
                raise ValueError("convex halfspaces must be nonempty F x 4 arrays")
            if not np.isfinite(equations).all():
                raise ValueError("convex halfspaces must be finite")
            equations.setflags(write=False)
        from .observed_convex_batch import ObservedConvexBatch
        self._native_cpu_batch = (ObservedConvexBatch(self.equations)
                                  if native_cpu_batch and self.equations else None)
        self.tree = cKDTree(self.points) if tree is None else tree
        if tree is not None and not np.array_equal(tree.data, self.points):
            raise ValueError("the supplied KD-tree must index the same scene points")
        self.backend = backend
        self.gpu_min_batch_size = max(1, int(gpu_min_batch_size))
        self._cuda = None
        self._cuda_failure = None
        self._lock = threading.Lock()
        self._preparing = None
        self._closed = False
        self._auto_cuda_selected = None
        self._auto_comparison = None
        self.last_backend = "cpu"
        self.query_count = 0
        self.cuda_query_count = 0

    @property
    def diagnostics(self):
        return {"algorithm": "native_observed_points_exact_convex_hulls",
                "requested_backend": self.backend, "last_backend": self.last_backend,
                "gpu_min_batch_size": self.gpu_min_batch_size,
                "gpu_min_nearby_pairs": 8192,
                "auto_backend_comparison": self._auto_comparison,
                "native_point_count": len(self.points),
                "query_count": self.query_count, "cuda_query_count": self.cuda_query_count,
                "native_cpu_batch": (None if self._native_cpu_batch is None
                                     else self._native_cpu_batch.diagnostics),
                "cuda_fallback_reason": self._cuda_failure,
                "coverage": "measured points only; occluded/outside-ROI space remains unknown",
                "voxel_quantisation": False, "cpu_exact_confirmation": True}

    def _arguments(self, positions, rotations, radii, margins, bounds):
        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64)
        margins = np.asarray(margins, dtype=np.float64)
        bounds = np.asarray(bounds, dtype=np.float64)
        count = len(self.geometry_ids)
        if positions.ndim != 3 or positions.shape[1:] != (count, 3):
            raise ValueError("batch positions must be B x G x 3")
        if rotations.shape != positions.shape[:2] + (3, 3):
            raise ValueError("batch rotations must be B x G x 3 x 3")
        if radii.shape != (count,) or margins.shape != (count,):
            raise ValueError("radii and margins must contain one value per geometry")
        if bounds.shape != (count, 2, 3):
            raise ValueError("bounds must be G x 2 x 3")
        if not all(np.isfinite(a).all() for a in (positions, rotations, radii, margins, bounds)):
            raise ValueError("scene query arguments must be finite")
        if (radii < 0).any() or (bounds[:, 0] > bounds[:, 1]).any():
            raise ValueError("scene query radii/bounds are invalid")
        return positions, rotations, radii, margins, bounds

    def _inside(self, points, geometry_index, position, rotation, margin, bounds):
        # Keep the arithmetic/rare exact-face fallback identical to
        # RobotModel.inside_geom, including its original input point order.
        equations = self.equations[geometry_index]
        local = (points - position) @ rotation
        lower, upper = bounds
        selected = np.flatnonzero(np.all((local >= lower) & (local <= upper), axis=1))
        result = np.zeros(len(local), dtype=bool)
        if not len(selected):
            return result
        values = local[selected] @ equations[:, :3].T + equations[:, 3]
        roundoff = 32*np.finfo(float).eps*(1 + np.sum(np.abs(local[selected]), axis=1)
                    + np.max(np.abs(equations[:, 3])) + abs(margin))
        if np.any(np.min(np.abs(values-margin), axis=1) <= roundoff):
            return np.all(local @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)
        result[selected] = np.all(values <= margin, axis=1)
        return result

    def _nearby(self, positions, radii):
        if not positions.size:
            return []
        # SciPy's vector query runs the unchanged sphere tests in C. Explicit
        # return_sorted=False preserves single-query native KD-tree ordering.
        return self.tree.query_ball_point(positions.reshape(-1, 3),
            np.tile(radii, len(positions)), return_sorted=False)

    def _cpu(self, positions, rotations, radii, margins, bounds, candidates=None, nearby=None):
        nearby = self._nearby(positions, radii) if nearby is None else nearby
        # One C call processes the original ragged KD-tree neighbours and all
        # original convex faces. This is the final narrow phase, not a coarse
        # route proposal followed by duplicate full-cloud certification.
        native = self._native_cpu_batch
        packed = (native.query(self.points, nearby, positions, rotations, margins, bounds)
                  if native is not None and len(positions) >= 4 else None)
        if packed is not None:
            point_indices, offsets, flags, uncertain = packed
            results = []
            for b in range(len(positions)):
                found = {}
                for i, g in enumerate(self.geometry_ids):
                    if candidates is not None and not candidates[b, i]:
                        continue
                    query = b*len(self.geometry_ids)+i
                    low, high = offsets[query:query+2]
                    if low == high:
                        continue
                    indices = point_indices[low:high]
                    inside = (self._inside(self.points[indices], i, positions[b, i],
                                          rotations[b, i], margins[i], bounds[i])
                              if uncertain[query] else flags[low:high] == 1)
                    if inside.any():
                        found[g] = indices[inside]
                results.append(found)
            return results
        results = []
        for b in range(len(positions)):
            found = {}
            for i, g in enumerate(self.geometry_ids):
                if candidates is not None and not candidates[b, i]:
                    continue
                indices = np.asarray(nearby[b*len(self.geometry_ids)+i], dtype=np.int64)
                if len(indices):
                    inside = self._inside(self.points[indices], i, positions[b, i],
                                          rotations[b, i], margins[i], bounds[i])
                    if inside.any():
                        found[g] = indices[inside]
            results.append(found)
        return results

    def _cuda_candidates(self, positions, rotations, radii, margins, bounds, nearby=None):
        if self._cuda is None:
            if importlib.util.find_spec("torch") is not None:
                from ._gpu_scene_collision_kernel import CudaObservedPointQueries
                self._cuda = CudaObservedPointQueries(self.points, self.equations)
            else:
                from ._gpu_scene_collision_service import RemoteCudaObservedPointQueries
                python = Path(__file__).resolve().parents[2]/"experiments/graspgenx_sim/.venv/bin/python"
                if not python.exists():
                    raise RuntimeError("optional GraspGen-X CUDA Python environment is unavailable")
                self._cuda = RemoteCudaObservedPointQueries(self.points, self.equations, python)
        return self._cuda.candidates(positions, rotations, radii, margins, bounds, nearby)

    def prepare(self):
        """Start optional upload asynchronously, without delaying CPU planning.

        Call after installing a scene, alongside network inference. Upload and
        CUDA JIT run on this thread; ``auto`` uses CPU while preparation runs.
        """
        if (self._closed or self.backend == "cpu" or self._cuda is not None
                or self._cuda_failure is not None or not len(self.points)
                or self._preparing is not None
                or (self.backend == "auto" and len(self.points) < 4096)):
            return
        # The upload thread owns this lock while importing/compiling. A second
        # batched query must keep using CPU rather than wait for that upload.
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._preparing is not None:
                return
            def prepare():
                with self._lock:
                    try:
                        # A dummy zero-radius query compiles the same N/G
                        # kernel; its result is discarded, never a certificate.
                        count = len(self.geometry_ids)
                        self._cuda_candidates(np.zeros((1, count, 3)),
                            np.broadcast_to(np.eye(3), (1, count, 3, 3)), np.zeros(count),
                            np.zeros(count), np.zeros((count, 2, 3)),
                            nearby=[[0] if len(self.points) else [] for _ in range(count)])
                    except Exception as error:
                        self._cuda_failure = f"{type(error).__name__}: {error}"
            self._preparing = threading.Thread(target=prepare, daemon=True,
                                               name="observed-scene-cuda-upload")
            self._preparing.start()
        finally:
            self._lock.release()

    def close(self):
        """Release CUDA ownership; a shared service exits after its last index.

        Synchronises with a running prepare/query. Closing twice is harmless.
        The immutable CPU index stays usable, with CUDA disabled afterwards.
        """
        self._closed = True
        thread = self._preparing
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            cuda, self._cuda = self._cuda, None
            if cuda is not None and hasattr(cuda, "close"):
                try:
                    cuda.close()
                except Exception as error:
                    self._cuda_failure = f"CUDA close {type(error).__name__}: {error}"

    def query_batch(self, positions, rotations, radii, margins, bounds):
        """Return exact native colliding indices for every pose and geometry.

        Results have one dictionary per pose. The batch is solely geometric;
        the caller still performs joint-limit, model self-collision, selected
        target contact and route/trajectory checks with their existing policy.
        """
        args = self._arguments(positions, rotations, radii, margins, bounds)
        positions, rotations, radii, margins, bounds = args
        if (self.backend == "auto" and self._cuda is None and self._cuda_failure is None
                and len(positions) >= self.gpu_min_batch_size and len(self.points) >= 4096):
            self.prepare()
        # Async upload owns this index's resources. Keep the current original
        # CPU algorithm available while importing torch/uploading the map.
        if self.backend == "auto" and self._preparing is not None and self._preparing.is_alive():
            self.query_count += len(positions)
            self.last_backend = "cpu"
            return self._cpu(*args)
        with self._lock:
            self.query_count += len(positions)
            self.last_backend = "cpu"
            candidates = None
            nearby = self._nearby(positions, radii)
            near_count = sum(len(ids) for ids in nearby)
            if not near_count:
                self.last_backend = "cpu_batched_kdtree_clear"
                return [{} for _ in range(len(positions))]
            use_cuda = (self.backend == "cuda" or
                        (self.backend == "auto" and len(positions) >= self.gpu_min_batch_size
                         and len(self.points) >= 4096 and near_count >= 8192
                         and self._auto_cuda_selected is not False))
            if use_cuda and not self._closed and self._cuda_failure is None and len(self.points) and len(self.geometry_ids):
                try:
                    if self.backend == "auto" and self._auto_cuda_selected is None:
                        start = time.perf_counter()
                        reference = self._cpu(*args, nearby=nearby)
                        cpu_s = time.perf_counter()-start
                        start = time.perf_counter()
                        self._cuda_candidates(*args, nearby=nearby)
                        warmup_s = time.perf_counter()-start
                        start = time.perf_counter()
                        candidates = self._cuda_candidates(*args, nearby=nearby)
                        result = self._cpu(*args, candidates=candidates, nearby=nearby)
                        cuda_s = time.perf_counter()-start
                        identical = all(a.keys() == b.keys() and all(
                            np.array_equal(a[g], b[g]) for g in a) for a, b in zip(reference, result))
                        self._auto_cuda_selected = bool(identical and cuda_s < cpu_s*.9)
                        self._auto_comparison = {"cpu_narrow_phase_s": cpu_s,
                            "cuda_and_exact_cpu_confirmation_s": cuda_s,
                            "cuda_warmup_query_s": warmup_s,
                            "nearby_point_body_pairs": near_count, "poses": len(positions),
                            "cuda_selected": self._auto_cuda_selected,
                            "exact_native_indices_identical": identical}
                        if not identical:
                            self._cuda_failure = "CUDA candidate result disagreed with original CPU; using CPU"
                        if self._auto_cuda_selected:
                            self.last_backend = "cuda_exact_cpu_confirmation"
                            self.cuda_query_count += len(positions)
                            return result
                        self.last_backend = "cpu_auto_comparison"
                        return reference
                    candidates = self._cuda_candidates(*args, nearby=nearby)
                    self.last_backend = "cuda_exact_cpu_confirmation"
                    self.cuda_query_count += len(positions)
                except Exception as error:
                    # This selects the original algorithm, not an empty scene
                    # or a new robot stop condition. Keep the reason observable.
                    self._cuda_failure = f"{type(error).__name__}: {error}"
            return self._cpu(*args, candidates=candidates, nearby=nearby)

    def query(self, positions, rotations, radii, margins, bounds):
        """Single-pose convenience method, same geometric order as query_batch."""
        return self.query_batch(np.asarray(positions)[None], np.asarray(rotations)[None],
                                radii, margins, bounds)[0]

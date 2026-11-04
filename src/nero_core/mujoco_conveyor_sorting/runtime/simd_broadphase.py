"""Conservative SIMD robot/self/object pruning with unchanged exact fallback.

This project kernel follows VAMP's structure-of-arrays / batch-of-states CPU
SIMD design. It does not claim to link the official VAMP or CAPT library, nor
to replace CAPT/KD-tree point queries. Original convex halfspaces and MuJoCo
remain the narrow phase. A positive clearance only proves the enclosing
AABBs are separated; zero means "run the original check", not collision.

References: https://github.com/KavrakiLab/vamp and
https://github.com/KavrakiLab/capt (RSS 2024 point-tree scope).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

_LIBRARY = None
_LOAD_ERROR = None
_LOCK = threading.Lock()
_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_INT64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
_ROUND = 64*np.finfo(np.float64).eps


def _load_native():
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None or _LOAD_ERROR is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or _LOAD_ERROR is not None:
            return _LIBRARY
        try:
            source = Path(__file__).with_name("_simd_broadphase.c")
            compiler = shutil.which(os.environ.get("CC", "cc"))
            if compiler is None:
                raise RuntimeError("C compiler unavailable; using NumPy broad phase")
            identity = (source.read_bytes()+platform.machine().encode()
                        +subprocess.check_output([compiler, "--version"]))
            digest = hashlib.sha256(identity).hexdigest()[:20]
            cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"nero-planning"/"simd"
            cache.mkdir(parents=True, exist_ok=True)
            library = cache/f"broadphase_{digest}.so"
            if not library.is_file():
                with tempfile.TemporaryDirectory(prefix="build_", dir=cache) as temporary:
                    built = Path(temporary)/"broadphase.so"
                    result = subprocess.run([compiler, "-O3", "-std=c11", "-shared", "-fPIC",
                        "-fno-fast-math", "-ffp-contract=off", str(source), "-lm", "-o", str(built)],
                        capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        raise RuntimeError(f"native broad-phase compilation failed: {result.stderr[-1000:]}")
                    # Concurrent processes may build the same content; rename
                    # publishes only a complete library and is atomic on POSIX.
                    os.replace(built, library)
            native = ctypes.CDLL(str(library))
            native.nero_simd_has_avx2.restype = ctypes.c_int
            native.nero_world_aabbs.argtypes = [_DOUBLE, _DOUBLE, _DOUBLE,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, _DOUBLE]
            native.nero_world_aabbs.restype = None
            native.nero_pair_clearances.argtypes = [_DOUBLE, _INT64,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, _DOUBLE]
            native.nero_pair_clearances.restype = None
            native.nero_object_clearances.argtypes = [_DOUBLE, _DOUBLE,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, _DOUBLE]
            native.nero_object_clearances.restype = None
            native.nero_point_clearances.argtypes = [_DOUBLE, _DOUBLE,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, _DOUBLE]
            native.nero_point_clearances.restype = None
            _LIBRARY = native
        except Exception as error:
            _LOAD_ERROR = f"{type(error).__name__}: {error}"
    return _LIBRARY


def _bounds(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (2, 3) or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} must have shape {shape or '(…, 2, 3)'}")
    if not np.isfinite(value).all() or np.any(value[..., 0, :] > value[..., 1, :]):
        raise ValueError(f"{name} must be finite ordered enclosing bounds")
    return value


def bounds_of_points(points):
    """An enclosing world box, or None for an empty native point cloud."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite N×3 world coordinates")
    if not len(points):
        return None
    return np.stack((points.min(axis=0), points.max(axis=0)))


class ConvexBroadPhase:
    """Enclose the *original padded hulls*, prune only proven separated pairs.

    ``local_bounds[G,2,3]`` must contain the convex bodies at their final
    halfspace margin, including extended offset corners. Expanding an AABB
    by a scalar margin is generally insufficient; RobotModel.mesh_bounds
    already computes the right envelope by halfspace intersection.
    """
    def __init__(self, local_bounds, *, geometry_ids=None, backend="auto"):
        local = _bounds(local_bounds, None, "local_bounds")
        if local.ndim != 3:
            raise ValueError("local_bounds must be G×2×3")
        if backend not in {"auto", "avx2", "scalar", "numpy"}:
            raise ValueError("backend must be auto, avx2, scalar or numpy")
        self.local_bounds = np.array(local, copy=True, order="C")
        self.local_bounds.setflags(write=False)
        self.geometry_ids = tuple(range(len(local)) if geometry_ids is None else geometry_ids)
        if len(self.geometry_ids) != len(local):
            raise ValueError("one geometry id is required for each local box")
        self.requested_backend = backend
        started = time.perf_counter()
        self._native = None if backend == "numpy" else _load_native()
        avx = self._native is not None and self._native.nero_simd_has_avx2()
        self.backend = ("avx2" if avx and backend != "scalar" else "scalar") if self._native else "numpy"
        self.diagnostics = {"implementation": "project_conservative_soa_aabb",
            "official_vamp_capt_linked": False, "requested_backend": backend,
            "backend": self.backend, "avx2_supported": bool(avx),
            "startup_wall_s": time.perf_counter()-started,
            "native_fallback_reason": _LOAD_ERROR, "state_boxes": 0,
            "pair_tests": 0, "pair_proven_separated": 0,
            "object_tests": 0, "object_proven_separated": 0}

    def world_aabbs(self, positions, rotations):
        """Return enclosing bounds[B,G,2,3] without doing FK again."""
        p, r = np.asarray(positions, dtype=np.float64), np.asarray(rotations, dtype=np.float64)
        g = len(self.local_bounds)
        if p.ndim != 3 or p.shape[1:] != (g, 3) or r.shape != (*p.shape[:2], 3, 3):
            raise ValueError("positions[B,G,3] and rotations[B,G,3,3] required")
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError("transforms must be finite")
        self.diagnostics["state_boxes"] += int(p.shape[0]*g)
        if self._native is not None:
            b = p.shape[0]
            # All four lanes are different states of the same fixed geometry.
            ps = np.ascontiguousarray(p.transpose(1, 2, 0))
            rs = np.ascontiguousarray(r.transpose(1, 2, 3, 0))
            output = np.empty((g, 2, 3, b), dtype=np.float64)
            self._native.nero_world_aabbs(ps, rs, self.local_bounds, b, g,
                int(self.backend == "avx2"), output)
            return np.ascontiguousarray(output.transpose(3, 0, 1, 2))
        center = self.local_bounds.mean(axis=1)
        half = .5*(self.local_bounds[:, 1]-self.local_bounds[:, 0])
        c = p+np.einsum("bgij,gj->bgi", r, center)
        e = np.einsum("bgij,gj->bgi", np.abs(r), half)
        scale = np.abs(p)+np.einsum("bgij,gj->bgi", np.abs(r), np.abs(center))
        error = _ROUND*(1+scale+e)
        return np.stack((c-e-error, c+e+error), axis=2)

    def _world_bounds(self, world_bounds):
        world = _bounds(world_bounds, None, "world_bounds")
        if world.ndim != 4 or world.shape[1] != len(self.local_bounds):
            raise ValueError("world_bounds must be B×G×2×3")
        return world

    @staticmethod
    def _numpy_clearance(first, second):
        al, au, bl, bu = first[..., 0, :], first[..., 1, :], second[..., 0, :], second[..., 1, :]
        error = _ROUND*(1+np.abs(al)+np.abs(au)+np.abs(bl)+np.abs(bu))
        gap = np.maximum(0., np.maximum(bl-au, al-bu)-error)
        distance = np.sqrt(np.sum(gap*gap, axis=-1))
        return np.maximum(0., distance-_ROUND*(1+distance))

    def pair_clearances(self, world_bounds, pairs):
        """Lower bound[B,P] on exact pair distance; 0 requires narrow phase."""
        world = self._world_bounds(world_bounds)
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0) or np.any(pairs >= world.shape[1]):
            raise ValueError("pairs must be valid P×2 local geometry indices")
        if self._native is not None:
            b, g, _, _ = world.shape
            boxes = np.ascontiguousarray(world.transpose(1, 2, 3, 0))
            output = np.empty((len(pairs), b), dtype=np.float64)
            self._native.nero_pair_clearances(boxes, np.ascontiguousarray(pairs), b,
                len(pairs), int(self.backend == "avx2"), output)
            result = np.ascontiguousarray(output.T)
        else:
            result = self._numpy_clearance(world[:, pairs[:, 0]], world[:, pairs[:, 1]])
        self.diagnostics["pair_tests"] += int(result.size)
        self.diagnostics["pair_proven_separated"] += int(np.count_nonzero(result > 0))
        return result

    def pair_candidates(self, world_bounds, pairs):
        return self.pair_clearances(world_bounds, pairs) <= 0.

    def object_clearances(self, world_bounds, object_bounds):
        """Lower bound[B,G] to an enclosing object box (single or per-state)."""
        world = self._world_bounds(world_bounds)
        b, g, _, _ = world.shape
        if object_bounds is None:
            return np.full((b, g), np.inf)
        obj = _bounds(object_bounds, None, "object_bounds")
        if obj.shape == (2, 3):
            obj = np.broadcast_to(obj, (b, 2, 3))
        if obj.shape != (b, 2, 3):
            raise ValueError("object_bounds must be 2×3 or B×2×3")
        if self._native is not None:
            boxes = np.ascontiguousarray(world.transpose(1, 2, 3, 0))
            objects = np.ascontiguousarray(obj.transpose(1, 2, 0))
            output = np.empty((g, b), dtype=np.float64)
            self._native.nero_object_clearances(boxes, objects, b, g,
                int(self.backend == "avx2"), output)
            result = np.ascontiguousarray(output.T)
        else:
            result = self._numpy_clearance(world, obj[:, None])
        self.diagnostics["object_tests"] += int(result.size)
        self.diagnostics["object_proven_separated"] += int(np.count_nonzero(result > 0))
        return result

    def object_candidates(self, world_bounds, object_bounds):
        return self.object_clearances(world_bounds, object_bounds) <= 0.

    def point_clearances(self, world_bounds, points):
        """Distance lower bounds[B,G] from boxes to native *object* points.

        Useful when the complete target AABB is too loose. This optional
        primitive does not replace the existing static scene KD-tree query.
        """
        world = self._world_bounds(world_bounds)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be finite N×3 world coordinates")
        b, g = world.shape[:2]
        if not len(points):
            return np.full((b, g), np.inf)
        if self._native is not None:
            output = np.empty((b, g), dtype=np.float64)
            self._native.nero_point_clearances(np.ascontiguousarray(world),
                np.ascontiguousarray(points.T), b, g, len(points),
                int(self.backend == "avx2"), output)
            return output
        # Keep peak memory linear in cloud size rather than B×G×N.
        result = np.empty((b, g))
        cloud_boxes = np.stack((points, points), axis=1)
        for row in range(b):
            for geom in range(g):
                result[row, geom] = np.min(self._numpy_clearance(world[row, geom], cloud_boxes))
        return result

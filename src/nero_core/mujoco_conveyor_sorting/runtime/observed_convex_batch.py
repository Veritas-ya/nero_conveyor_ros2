"""Local batched float64 narrow phase for native observed-point hull tests.

This is a project implementation inspired by batched validation in VAMP;
it does not link VAMP or substitute a sphere model. Every original convex
face and KD-tree neighbour is used. Floating-point boundary queries return
to the original NumPy calculation. Compiler failure leaves that path intact.
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
_ERROR = None
_LOCK = threading.Lock()
_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_INT64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
_BYTE = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")


def _load_native():
    global _LIBRARY, _ERROR
    if _LIBRARY is not None or _ERROR is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or _ERROR is not None:
            return _LIBRARY
        try:
            source = Path(__file__).with_name("_observed_convex_batch.c")
            compiler = shutil.which(os.environ.get("CC", "cc"))
            if compiler is None:
                raise RuntimeError("C compiler unavailable")
            identity = (source.read_bytes()+platform.machine().encode()+
                        subprocess.check_output([compiler, "--version"]))
            digest = hashlib.sha256(identity).hexdigest()[:20]
            cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"nero-planning"/"convex"
            cache.mkdir(parents=True, exist_ok=True)
            library = cache/f"observed_convex_{digest}.so"
            if not library.is_file():
                with tempfile.TemporaryDirectory(prefix="build_", dir=cache) as temporary:
                    built = Path(temporary)/"observed_convex.so"
                    result = subprocess.run([compiler, "-O3", "-std=c11", "-shared", "-fPIC",
                        "-fno-fast-math", "-ffp-contract=off", str(source), "-lm", "-o", str(built)],
                        capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        raise RuntimeError(result.stderr[-1000:])
                    os.replace(built, library)
            native = ctypes.CDLL(str(library))
            native.nero_observed_convex_batch.argtypes = [_DOUBLE, _INT64, _INT64,
                _DOUBLE, _DOUBLE, _DOUBLE, _INT64, _DOUBLE, _DOUBLE,
                ctypes.c_size_t, ctypes.c_size_t, _BYTE, _BYTE]
            native.nero_observed_convex_batch.restype = None
            _LIBRARY = native
        except Exception as error:
            _ERROR = f"{type(error).__name__}: {error}"
    return _LIBRARY


def prepare_observed_convex_batch():
    """Pay optional compilation during device/model preparation, before capture."""
    began = time.perf_counter()
    native = _load_native()
    return {'implementation': 'project_native_float64_convex_halfspace_batch',
            'available': native is not None, 'wall_s': time.perf_counter()-began,
            'fallback_reason': _ERROR}


class ObservedConvexBatch:
    """Compile once; pack the unchanged hull faces once per immutable index."""

    def __init__(self, equations):
        self.native = _load_native()
        self.face_offsets = np.r_[0, np.cumsum([len(e) for e in equations])].astype(np.int64)
        self.planes = np.ascontiguousarray(np.concatenate(equations, axis=0), dtype=np.float64)
        self.stats = dict(calls=0, state_geometry_queries=0, nearby_point_pairs=0,
                          numpy_boundary_queries=0)

    @property
    def diagnostics(self):
        return {"implementation": "project_native_float64_convex_halfspace_batch",
                "available": self.native is not None, "fallback_reason": _ERROR,
                "official_vamp_linked": False, **self.stats}

    def query(self, points, nearby, positions, rotations, margins, bounds):
        if self.native is None:
            return None
        lengths = np.fromiter((len(ids) for ids in nearby), dtype=np.int64, count=len(nearby))
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        indices = (np.concatenate([np.asarray(ids, dtype=np.int64) for ids in nearby if len(ids)])
                   if offsets[-1] else np.empty(0, dtype=np.int64))
        flags = np.empty(len(indices), dtype=np.uint8)
        uncertain = np.zeros(len(nearby), dtype=np.uint8)
        self.native.nero_observed_convex_batch(np.ascontiguousarray(points), indices, offsets,
            np.ascontiguousarray(positions), np.ascontiguousarray(rotations),
            self.planes, self.face_offsets, np.ascontiguousarray(margins),
            np.ascontiguousarray(bounds), len(nearby), len(margins), flags, uncertain)
        self.stats["calls"] += 1
        self.stats["state_geometry_queries"] += len(nearby)
        self.stats["nearby_point_pairs"] += len(indices)
        self.stats["numpy_boundary_queries"] += int(np.count_nonzero(uncertain))
        return indices, offsets, flags, uncertain

"""Official VAMP/CAPT selected-target sphere broad phase, exact fallback.

The vendored upstream headers, including their AVX2 float vector operations,
are compiled unchanged behind a small C ABI. Native double hull radii enclose
the complete original padded convex models. Float conversion and arithmetic
are rounded outward; any candidate still receives the existing double hull
check. This does not replace the static scene index or the shared robot FK.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import weakref

import numpy as np

_LIBRARY = None
_LOAD_ERROR = None
_LOCK = threading.Lock()
_DOUBLE = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_BYTE = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')


def _load_native():
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None or _LOAD_ERROR is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or _LOAD_ERROR is not None:
            return _LIBRARY
        try:
            from .simd_broadphase import _load_native as simd_native
            capability = simd_native()
            if capability is None or not capability.nero_simd_has_avx2():
                raise RuntimeError('official CAPT requires AVX2; keeping conservative AABB')
            source = Path(__file__).with_name('_official_capt.cc')
            vendor = Path(__file__).resolve().parents[2]/'vendor/vamp_capt'
            include = vendor/'include'
            compiler = shutil.which(os.environ.get('CXX', 'c++'))
            if compiler is None:
                raise RuntimeError('C++ compiler unavailable; keeping conservative AABB')
            identity = source.read_bytes()+b''.join(p.read_bytes() for p in sorted(include.rglob('*')) if p.is_file())
            identity += subprocess.check_output([compiler, '--version'])
            digest = hashlib.sha256(identity).hexdigest()[:20]
            cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'nero-planning/simd'
            cache.mkdir(parents=True, exist_ok=True)
            library = cache/f'official_capt_{digest}.so'
            if not library.is_file():
                with tempfile.TemporaryDirectory(prefix='capt_build_', dir=cache) as temporary:
                    built = Path(temporary)/'capt.so'
                    result = subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-mavx2', '-fno-fast-math', '-ffp-contract=off', '-DNDEBUG',
                        '-Wno-ignored-attributes', '-I', str(include), str(source), '-o', str(built)],
                        text=True, capture_output=True, timeout=60)
                    if result.returncode:
                        raise RuntimeError('official CAPT build failed: '+result.stderr[-1200:])
                    os.replace(built, library)
            native = ctypes.CDLL(str(library))
            native.nero_official_capt_create.argtypes = [_DOUBLE, ctypes.c_size_t, ctypes.c_double, _DOUBLE]
            native.nero_official_capt_create.restype = ctypes.c_void_p
            native.nero_official_capt_destroy.argtypes = [ctypes.c_void_p]
            native.nero_official_capt_destroy.restype = None
            native.nero_official_capt_query.argtypes = [ctypes.c_void_p, _DOUBLE, _DOUBLE,
                ctypes.c_size_t, _BYTE, ctypes.POINTER(ctypes.c_double)]
            native.nero_official_capt_query.restype = None
            _LIBRARY = native
        except Exception as error:
            _LOAD_ERROR = f'{type(error).__name__}: {error}'
    return _LIBRARY


class OfficialCAPTIndex:
    """One immutable selected-target cloud, fixed query-radius coverage."""
    def __init__(self, points, maximum_radius_m):
        self.points = np.array(points, dtype=np.float64, copy=True, order='C')
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError('CAPT target points must be finite N×3')
        self.points.flags.writeable = False
        self.maximum_radius_m = float(maximum_radius_m)
        began = time.perf_counter()
        self._native = _load_native()
        self._handle = None
        metadata = np.zeros(4)
        if self._native is not None and len(self.points):
            self._handle = self._native.nero_official_capt_create(self.points,
                len(self.points), self.maximum_radius_m, metadata)
        self.available = bool(self._handle)
        self._finalizer = (weakref.finalize(self, self._native.nero_official_capt_destroy, self._handle)
                           if self._handle else None)
        self.diagnostics = dict(backend='official_vamp_capt_avx2',
            official_vamp_capt_linked=self.available,
            vamp_commit='27cb9b66ebd8d101ce88d330f3b028398afc1cfa',
            pdqsort_commit='b1ef26a55cdb60d236a5cb199c4234c704f46726',
            available=self.available, fallback_reason=_LOAD_ERROR,
            build_wall_s=time.perf_counter()-began, target_points=len(self.points),
            point_quantization_l1_m=float(metadata[0]), point_padding_m=float(metadata[1]),
            maximum_query_radius_m=float(metadata[2]), simd_width=int(metadata[3]),
            queries=0, proven_separated=0, maximum_float_outward_expansion_m=0.)

    def query(self, centers, radii):
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        if centers.shape != (len(radii), 3) or not np.isfinite(centers).all() or not np.isfinite(radii).all():
            raise ValueError('finite centers[N,3] and radii[N] required')
        if not self.available:
            return np.ones(len(radii), dtype=bool)
        result = np.empty(len(radii), dtype=np.uint8)
        expansion = ctypes.c_double()
        self._native.nero_official_capt_query(self._handle, centers, radii, len(radii),
                                             result, ctypes.byref(expansion))
        self.diagnostics['queries'] += len(radii)
        self.diagnostics['proven_separated'] += int(np.count_nonzero(result == 0))
        self.diagnostics['maximum_float_outward_expansion_m'] = max(expansion.value,
            self.diagnostics['maximum_float_outward_expansion_m'])
        return result.astype(bool)

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self.available = False
        self._handle = None

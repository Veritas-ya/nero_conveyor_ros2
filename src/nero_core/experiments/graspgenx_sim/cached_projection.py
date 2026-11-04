"""Calibration-only cached depth projection, independent of simulator truth.

Fixed cameras can cache the world direction of every raster ray. This retains
every depth pixel and float64 output; it changes neither K nor resolution.
"""
from __future__ import annotations

import numpy as np


class CachedDepthProjector:
    def __init__(self):
        self._key = None
        self._rays_world = None
        self.cache_rebuilds = 0

    def project(self, depth, intrinsics, camera_to_world, *, camera_convention="mujoco"):
        image = np.asarray(depth, dtype=np.float64)
        matrix = np.asarray(intrinsics, dtype=np.float64)
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if image.ndim != 2 or matrix.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError("require depth H×W, K3×3 and T4×4")
        if not np.isfinite(matrix).all() or not np.isfinite(transform).all():
            raise ValueError("calibration must be finite")
        if not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-9):
            raise ValueError("camera_to_world must be homogeneous")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("focal lengths must be positive")
        if camera_convention not in ("mujoco", "opencv"):
            raise ValueError("camera_convention must be mujoco or opencv")
        key = (image.shape, matrix.tobytes(), transform.tobytes(), camera_convention)
        if key != self._key:
            height, width = image.shape
            columns = (np.arange(width, dtype=np.float64) - matrix[0, 2]) / matrix[0, 0]
            rows = (np.arange(height, dtype=np.float64) - matrix[1, 2]) / matrix[1, 1]
            rotation = transform[:3, :3].copy()
            if camera_convention == "mujoco":
                rotation[:, 1:] *= -1.
            rays = np.empty((*image.shape, 3), dtype=np.float64)
            for axis in range(3):
                rays[..., axis] = (columns[None, :] * rotation[axis, 0]
                                   + rows[:, None] * rotation[axis, 1]
                                   + rotation[axis, 2])
            self._rays_world = rays
            self._key = key
            self.cache_rebuilds += 1
        points = np.empty((*image.shape, 3), dtype=np.float64)
        np.multiply(self._rays_world, image[..., None], out=points)
        points += transform[:3, 3]
        return points

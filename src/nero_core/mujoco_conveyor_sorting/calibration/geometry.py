"""Rigid transforms: ``a_to_b`` maps column vectors from frame a into b."""

from __future__ import annotations

import numpy as np


def rigid_transform(value, name: str = "transform") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)):
        raise ValueError(f"{name} must be a rigid, right-handed homogeneous transform")
    return result


def inverse(value) -> np.ndarray:
    transform = rigid_transform(value)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def pose_xyz_rpy(value) -> np.ndarray:
    """NERO SDK pose: metres/radians, fixed XYZ, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pose = np.asarray(value, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain six finite values: x y z roll pitch yaw (m/rad)")
    roll, pitch, yaw = pose[3:]
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = pose[:3]
    return result


def transform_distance(first, second) -> tuple[float, float]:
    """Origin distance (m) and relative rotation angle (degrees)."""
    angle = np.arccos(np.clip((np.trace(first[:3, :3].T @ second[:3, :3]) - 1) / 2, -1, 1))
    return float(np.linalg.norm(first[:3, 3] - second[:3, 3])), float(np.degrees(angle))


def mean_transform(transforms) -> np.ndarray:
    result = np.eye(4)
    u, _, vt = np.linalg.svd(np.mean([item[:3, :3] for item in transforms], axis=0))
    result[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    result[:3, 3] = np.mean([item[:3, 3] for item in transforms], axis=0)
    return result

"""Small pose-math helpers shared by the controller modules.

MuJoCo stores quaternions in ``[w, x, y, z]`` order.  Keeping these helpers
local avoids adding scipy as a runtime dependency just for six-dimensional
end-effector commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import numpy as np


T = TypeVar("T")


def as_vector(
    value: Any,
    size: int,
    *,
    name: str,
    allow_scalar: bool = False,
) -> np.ndarray:
    """Return *value* as a finite float vector of the requested size."""

    array = np.asarray(value, dtype=float)
    if allow_scalar and array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    array = array.reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalized_quat(quaternion: Any, *, name: str = "quaternion") -> np.ndarray:
    """Normalize a MuJoCo-order quaternion and reject degenerate values."""

    quat = as_vector(quaternion, 4, name=name)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError(f"{name} has zero norm")
    return quat / norm


def quat_conjugate(quaternion: Any) -> np.ndarray:
    quat = normalized_quat(quaternion)
    return np.array((quat[0], -quat[1], -quat[2], -quat[3]), dtype=float)


def quat_multiply(left: Any, right: Any) -> np.ndarray:
    """Hamilton product for two ``[w, x, y, z]`` quaternions."""

    w1, x1, y1, z1 = normalized_quat(left, name="left quaternion")
    w2, x2, y2, z2 = normalized_quat(right, name="right quaternion")
    result = np.array(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=float,
    )
    return result / np.linalg.norm(result)


def rotvec_to_quat(rotation_vector: Any) -> np.ndarray:
    """Convert an axis-angle rotation vector to a quaternion."""

    rotvec = as_vector(rotation_vector, 3, name="rotation vector")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        # First-order form is more useful than returning an unrelated axis.
        quat = np.r_[1.0, 0.5 * rotvec]
        return quat / np.linalg.norm(quat)
    half_angle = 0.5 * angle
    return np.r_[np.cos(half_angle), np.sin(half_angle) * rotvec / angle]


def quat_error_vector(target: Any, current: Any) -> np.ndarray:
    """Shortest world-frame rotation vector taking ``current`` to ``target``."""

    error_quat = quat_multiply(target, quat_conjugate(current))
    # q and -q describe the same rotation.  Choosing a non-negative scalar
    # component produces the shortest rotation and prevents sign flips.
    if error_quat[0] < 0.0:
        error_quat = -error_quat
    vector_norm = float(np.linalg.norm(error_quat[1:]))
    if vector_norm < 1e-12:
        return 2.0 * error_quat[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error_quat[0], -1.0, 1.0))
    return angle * error_quat[1:] / vector_norm


def quat_distance(left: Any, right: Any) -> float:
    """Unsigned shortest angular distance between two quaternions."""

    return float(np.linalg.norm(quat_error_vector(left, right)))


def compose_world_delta(current: Any, rotation_vector: Any) -> np.ndarray:
    """Apply a world-frame axis-angle increment to a quaternion."""

    return quat_multiply(rotvec_to_quat(rotation_vector), current)


def config_from_mapping(config_type: type[T], value: T | Mapping[str, Any] | None) -> T:
    """Construct a dataclass config while ignoring unrelated YAML keys.

    Repository-level YAML files often contain a ``robot`` section with more
    fields than a controller needs.  Ignoring those unrelated fields makes the
    controller easy to wire without weakening validation of the fields it does
    consume.
    """

    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            f"config must be {config_type.__name__}, a mapping, or None; "
            f"got {type(value).__name__}"
        )
    valid = {field.name for field in fields(config_type)}
    return config_type(**{key: item for key, item in value.items() if key in valid})


def jsonable(value: Any) -> Any:
    """Convert dataclasses/numpy values to small JSON-friendly structures."""

    if is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [jsonable(item) for item in value]
    return value


def belt_plane_motion_from_xy(value, normal_base):
    """Preserve measured XY and infer Z motion along the measured plane.

    Works for either displacement (m) or velocity (m/s). This is a geometric
    motion model, not a measurement of vertical speed. Orthogonal projection
    would also change the measured XY, so solve n dot motion = 0 for Z instead.
    Without a plane, retain the caller's existing motion model.
    """
    result = np.array(value, dtype=float, copy=True)
    if normal_base is not None:
        normal = np.asarray(normal_base, dtype=float)
        if abs(float(normal[2])) <= 1e-12:
            raise ValueError('a vertical plane cannot infer Z travel from XY')
        result[2] = -float(normal[:2] @ result[:2]) / float(normal[2])
    return result

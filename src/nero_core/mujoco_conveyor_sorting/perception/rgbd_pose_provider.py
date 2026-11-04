"""RGB-D pose estimates from renderer-style instance segmentation.

This module deliberately depends only on NumPy.  In particular, it never reads
simulator state: callers must provide the segmentation image, metric depth,
camera intrinsics, and the camera-to-world transform captured with that image.

Coordinate conventions
----------------------
``camera_to_world`` maps MuJoCo camera-local coordinates into world
coordinates.  A MuJoCo camera has +X right, +Y up, and looks along -Z.  Image
deprojection and BOP poses use the OpenCV convention (+X right, +Y down, +Z
forward), so the fixed conversion between the two camera frames is
``diag(1, -1, -1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.integer]

_OPENCV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _rotation_matrix(value: object, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 9:
        array = array.reshape(3, 3)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain a finite 3x3 rotation matrix")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(array)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return array.copy()


def _homogeneous_transform(value: object, name: str) -> FloatArray:
    transform = _finite_array(value, (4, 4), name)
    _rotation_matrix(transform[:3, :3], f"{name} rotation")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    return transform


def _normalize_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _axis_yaw_near(yaw: float, reference: float | None) -> float:
    """Choose the pi-equivalent PCA axis closest to ``reference``."""

    canonical = float((yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi)
    if reference is None:
        return canonical
    reference = float(reference)
    candidates = [canonical + multiple * math.pi for multiple in range(-2, 3)]
    selected = min(
        candidates,
        key=lambda candidate: abs(_normalize_angle(candidate - reference)),
    )
    return _normalize_angle(selected)


def _yaw_quaternion(yaw: float) -> FloatArray:
    half = 0.5 * float(yaw)
    return np.array(
        [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64
    )


def _matrix_to_quaternion_wxyz(rotation: object) -> FloatArray:
    """Convert a proper rotation matrix to a canonical WXYZ quaternion."""

    matrix = _rotation_matrix(rotation, "rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            ) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(
                max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            ) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(
                max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            ) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("rotation produced a zero quaternion")
    quaternion /= norm
    # q and -q encode the same rotation.  A deterministic sign prevents
    # artificial discontinuities in recorded pose histories.
    first_nonzero = next(
        (float(value) for value in quaternion if abs(float(value)) > 1e-12),
        1.0,
    )
    if first_nonzero < 0.0:
        quaternion *= -1.0
    return quaternion


@dataclass(frozen=True, slots=True)
class SegmentationInstance:
    """Describe which segmentation labels belong to one logical object.

    MuJoCo segmentation images contain ``(object_id, object_type)`` in their
    final dimension.  Compound workpieces can therefore list multiple labels,
    one per visible geom, while retaining one logical ``object_id``.
    """

    object_id: Hashable
    category_id: int
    labels: tuple[tuple[int, int], ...]
    symmetric: bool = False
    default_yaw: float = 0.0
    support_height: float | None = None

    def __post_init__(self) -> None:
        try:
            hash(self.object_id)
        except TypeError as error:
            raise ValueError("object_id must be hashable") from error
        if self.object_id is None:
            raise ValueError("object_id cannot be None")

        raw = np.asarray(self.labels, dtype=np.int64)
        if raw.shape == (2,):
            raw = raw.reshape(1, 2)
        if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
            raise ValueError("labels must contain one or more (object_id, object_type) pairs")
        if np.any(raw < 0):
            raise ValueError("segmentation labels must be non-negative")
        normalized_labels = tuple(
            (int(label[0]), int(label[1])) for label in raw
        )
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(self, "category_id", int(self.category_id))

        yaw = float(self.default_yaw)
        if not np.isfinite(yaw):
            raise ValueError("default_yaw must be finite")
        object.__setattr__(self, "default_yaw", _normalize_angle(yaw))
        if self.support_height is not None:
            height = float(self.support_height)
            if not np.isfinite(height):
                raise ValueError("support_height must be finite or None")
            object.__setattr__(self, "support_height", height)

    @property
    def segmentation_id(self) -> tuple[int, int] | None:
        """Return the only label, or ``None`` for a compound instance."""

        return self.labels[0] if len(self.labels) == 1 else None

    def mask(self, segmentation: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Build this instance's mask from an ``(H, W, 2)`` label image."""

        labels = np.asarray(segmentation)
        if labels.ndim != 3 or labels.shape[2] != 2:
            raise ValueError(
                "segmentation must have shape (height, width, 2), "
                f"got {labels.shape}"
            )
        result = np.zeros(labels.shape[:2], dtype=bool)
        for object_id, object_type in self.labels:
            result |= (labels[..., 0] == object_id) & (
                labels[..., 1] == object_type
            )
        return result


@dataclass(frozen=True, slots=True)
class PoseDetection:
    """One upright object pose inferred solely from an RGB-D frame."""

    object_id: Hashable
    category_id: int
    position: FloatArray
    quaternion: FloatArray
    timestamp: float
    confidence: float
    mask_pixels: int
    valid_depth_pixels: int
    yaw: float
    orientation_confidence: float

    def __post_init__(self) -> None:
        position = _finite_array(self.position, (3,), "position")
        quaternion = _finite_array(self.quaternion, (4,), "quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("quaternion cannot be zero")
        quaternion /= norm
        timestamp = float(self.timestamp)
        confidence = float(self.confidence)
        orientation_confidence = float(self.orientation_confidence)
        yaw = float(self.yaw)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= orientation_confidence <= 1.0:
            raise ValueError("orientation_confidence must be in [0, 1]")
        if not np.isfinite(yaw):
            raise ValueError("yaw must be finite")
        if int(self.mask_pixels) < 0 or int(self.valid_depth_pixels) < 0:
            raise ValueError("pixel counts must be non-negative")
        if int(self.valid_depth_pixels) > int(self.mask_pixels):
            raise ValueError("valid_depth_pixels cannot exceed mask_pixels")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "mask_pixels", int(self.mask_pixels))
        object.__setattr__(
            self, "valid_depth_pixels", int(self.valid_depth_pixels)
        )
        object.__setattr__(self, "yaw", _normalize_angle(yaw))
        object.__setattr__(
            self, "orientation_confidence", orientation_confidence
        )

    @property
    def pose(self) -> FloatArray:
        """Return ``[x, y, z, qw, qx, qy, qz]`` in world coordinates."""

        return np.r_[self.position, self.quaternion]

    @property
    def position_world(self) -> FloatArray:
        return self.position.copy()

    @property
    def quaternion_wxyz(self) -> FloatArray:
        return self.quaternion.copy()


class RGBDPoseProvider:
    """Estimate support-plane poses from RGB-D instance masks.

    The estimator assumes workpieces remain upright on a horizontal support
    plane.  Metric depth is deprojected and a robust PCA-oriented extent centre
    recovers world X/Y, matching the XY-centred BOP/T-LESS model frames more
    closely than the median of an asymmetric visible surface.  Z is set to the
    configured support-plane height.  The principal axis of the world XY point
    cloud supplies yaw.  PCA's pi ambiguity is resolved against the previous
    yaw; circular or explicitly symmetric masks retain that yaw.
    """

    def __init__(
        self,
        instances: Iterable[SegmentationInstance] | None = None,
        *,
        minimum_mask_pixels: int = 8,
        full_confidence_pixels: int = 100,
        symmetry_anisotropy_threshold: float = 0.12,
        minimum_depth: float = 1e-5,
        maximum_depth: float = math.inf,
        max_pca_points: int = 20_000,
        robust_extent_quantile: float = 0.01,
    ) -> None:
        if minimum_mask_pixels < 2:
            raise ValueError("minimum_mask_pixels must be at least two")
        if full_confidence_pixels < minimum_mask_pixels:
            raise ValueError(
                "full_confidence_pixels must be >= minimum_mask_pixels"
            )
        if not 0.0 <= symmetry_anisotropy_threshold <= 1.0:
            raise ValueError("symmetry_anisotropy_threshold must be in [0, 1]")
        if minimum_depth < 0.0 or not np.isfinite(minimum_depth):
            raise ValueError("minimum_depth must be finite and non-negative")
        if maximum_depth <= minimum_depth:
            raise ValueError("maximum_depth must exceed minimum_depth")
        if max_pca_points < 2:
            raise ValueError("max_pca_points must be at least two")
        if not 0.0 <= robust_extent_quantile < 0.25:
            raise ValueError("robust_extent_quantile must be in [0, 0.25)")
        self.instances = tuple(instances or ())
        self.minimum_mask_pixels = int(minimum_mask_pixels)
        self.full_confidence_pixels = int(full_confidence_pixels)
        self.symmetry_anisotropy_threshold = float(
            symmetry_anisotropy_threshold
        )
        self.minimum_depth = float(minimum_depth)
        self.maximum_depth = float(maximum_depth)
        self.max_pca_points = int(max_pca_points)
        self.robust_extent_quantile = float(robust_extent_quantile)
        self._previous_yaws: dict[Hashable, float] = {}

    def reset(self) -> None:
        self._previous_yaws.clear()

    def estimate(
        self,
        segmentation: npt.ArrayLike,
        depth: npt.ArrayLike,
        intrinsics: npt.ArrayLike,
        camera_to_world: npt.ArrayLike,
        *,
        capture_timestamp: float,
        support_plane_height: float,
        instances: Iterable[SegmentationInstance] | None = None,
        previous_yaws: Mapping[Hashable, float] | None = None,
    ) -> tuple[PoseDetection, ...]:
        """Return detections for one capture-time RGB-D frame.

        ``camera_to_world`` must be the transform recorded with this exact
        frame, not a transform read later when the wrist camera has moved.
        """

        labels = np.asarray(segmentation)
        depth_image = np.asarray(depth, dtype=np.float64)
        if labels.ndim != 3 or labels.shape[2] != 2:
            raise ValueError(
                "segmentation must have shape (height, width, 2), "
                f"got {labels.shape}"
            )
        if depth_image.shape != labels.shape[:2]:
            raise ValueError(
                "depth must match segmentation height and width, "
                f"got {depth_image.shape} and {labels.shape[:2]}"
            )
        matrix = _finite_array(intrinsics, (3, 3), "intrinsics")
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive")
        transform = _homogeneous_transform(
            camera_to_world, "camera_to_world"
        )
        timestamp = float(capture_timestamp)
        support_height = float(support_plane_height)
        if not np.isfinite(timestamp):
            raise ValueError("capture_timestamp must be finite")
        if not np.isfinite(support_height):
            raise ValueError("support_plane_height must be finite")

        configured_instances = (
            tuple(instances) if instances is not None else self.instances
        )
        detections: list[PoseDetection] = []
        for instance in configured_instances:
            mask = instance.mask(labels)
            mask_pixels = int(np.count_nonzero(mask))
            if mask_pixels < self.minimum_mask_pixels:
                continue
            rows, columns = np.nonzero(mask)
            depths = depth_image[rows, columns]
            valid = (
                np.isfinite(depths)
                & (depths > self.minimum_depth)
                & (depths < self.maximum_depth)
            )
            valid_depth_pixels = int(np.count_nonzero(valid))
            if valid_depth_pixels < self.minimum_mask_pixels:
                continue
            rows = rows[valid].astype(np.float64)
            columns = columns[valid].astype(np.float64)
            depths = depths[valid]

            points_opencv = np.column_stack(
                (
                    (columns - cx) * depths / fx,
                    (rows - cy) * depths / fy,
                    depths,
                )
            )
            points_mujoco = points_opencv @ _OPENCV_TO_MUJOCO.T
            points_world = (
                points_mujoco @ transform[:3, :3].T + transform[:3, 3]
            )
            xy = points_world[:, :2]
            if len(xy) > self.max_pca_points:
                stride = int(math.ceil(len(xy) / self.max_pca_points))
                pca_xy = xy[::stride]
            else:
                pca_xy = xy
            centered = pca_xy - np.mean(pca_xy, axis=0)
            covariance = centered.T @ centered / max(len(centered) - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            # T-LESS CAD origins are centred in XY.  The visible-surface median
            # can be biased by holes and strongly asymmetric top geometry, so
            # recover the centre of the robust silhouette extent in PCA axes.
            # Percentiles reject isolated raster/depth boundary pixels without
            # requiring simulator pose or object-specific truth.
            projected = xy @ eigenvectors
            quantile = self.robust_extent_quantile
            lower = np.quantile(projected, quantile, axis=0)
            upper = np.quantile(projected, 1.0 - quantile, axis=0)
            centre_xy = (0.5 * (lower + upper)) @ eigenvectors.T
            position = np.array(
                [
                    float(centre_xy[0]),
                    float(centre_xy[1]),
                    support_height
                    if instance.support_height is None
                    else float(instance.support_height),
                ],
                dtype=np.float64,
            )
            largest = max(float(eigenvalues[-1]), 0.0)
            smallest = max(float(eigenvalues[0]), 0.0)
            anisotropy = (
                (largest - smallest) / (largest + smallest)
                if largest + smallest > 1e-15
                else 0.0
            )
            external_previous = (
                None
                if previous_yaws is None
                else previous_yaws.get(instance.object_id)
            )
            previous = (
                self._previous_yaws.get(instance.object_id)
                if external_previous is None
                else float(external_previous)
            )
            ambiguous = bool(
                instance.symmetric
                or anisotropy < self.symmetry_anisotropy_threshold
            )
            if ambiguous:
                yaw = (
                    instance.default_yaw
                    if previous is None
                    else _normalize_angle(previous)
                )
                orientation_confidence = 0.0
            else:
                principal = eigenvectors[:, -1]
                raw_yaw = math.atan2(float(principal[1]), float(principal[0]))
                yaw = _axis_yaw_near(raw_yaw, previous)
                orientation_confidence = float(np.clip(anisotropy, 0.0, 1.0))
            self._previous_yaws[instance.object_id] = yaw

            depth_coverage = valid_depth_pixels / max(mask_pixels, 1)
            area_score = min(1.0, mask_pixels / self.full_confidence_pixels)
            confidence = float(
                np.clip(depth_coverage * area_score, 0.0, 1.0)
            )
            detections.append(
                PoseDetection(
                    object_id=instance.object_id,
                    category_id=instance.category_id,
                    position=position,
                    quaternion=_yaw_quaternion(yaw),
                    timestamp=timestamp,
                    confidence=confidence,
                    mask_pixels=mask_pixels,
                    valid_depth_pixels=valid_depth_pixels,
                    yaw=yaw,
                    orientation_confidence=orientation_confidence,
                )
            )
        return tuple(detections)

    detect = estimate


# Readable alias for projects that prefer normal CamelCase over the RGB-D
# initialism.  Both names refer to the exact same implementation.
RgbdPoseProvider = RGBDPoseProvider


def opencv_camera_to_world(camera_to_world: npt.ArrayLike) -> FloatArray:
    """Convert a MuJoCo-local camera transform to OpenCV camera axes."""

    transform = _homogeneous_transform(camera_to_world, "camera_to_world")
    result = transform.copy()
    result[:3, :3] = transform[:3, :3] @ _OPENCV_TO_MUJOCO
    return result


def bop_transform_to_world(
    cam_R_m2c: npt.ArrayLike,
    cam_t_m2c_mm: npt.ArrayLike,
    camera_to_world: npt.ArrayLike,
) -> FloatArray:
    """Convert a BOP model-to-camera pose into a world transform.

    BOP uses OpenCV camera axes and stores ``cam_t_m2c`` in millimetres.
    ``camera_to_world`` uses MuJoCo camera-local axes and metres.
    """

    rotation_camera_object = _rotation_matrix(cam_R_m2c, "cam_R_m2c")
    translation_camera_object = np.asarray(
        cam_t_m2c_mm, dtype=np.float64
    ).reshape(-1)
    if translation_camera_object.shape != (3,) or not np.all(
        np.isfinite(translation_camera_object)
    ):
        raise ValueError("cam_t_m2c_mm must contain three finite values")
    camera_object = np.eye(4, dtype=np.float64)
    camera_object[:3, :3] = rotation_camera_object
    camera_object[:3, 3] = translation_camera_object * 1e-3
    return opencv_camera_to_world(camera_to_world) @ camera_object


def bop_pose_to_world(
    cam_R_m2c: npt.ArrayLike,
    cam_t_m2c_mm: npt.ArrayLike,
    camera_to_world: npt.ArrayLike,
) -> FloatArray:
    """Return BOP pose as world ``[x, y, z, qw, qx, qy, qz]``."""

    transform = bop_transform_to_world(
        cam_R_m2c, cam_t_m2c_mm, camera_to_world
    )
    quaternion = _matrix_to_quaternion_wxyz(transform[:3, :3])
    return np.r_[transform[:3, 3], quaternion]


__all__ = [
    "PoseDetection",
    "RGBDPoseProvider",
    "RgbdPoseProvider",
    "SegmentationInstance",
    "bop_pose_to_world",
    "bop_transform_to_world",
    "opencv_camera_to_world",
]

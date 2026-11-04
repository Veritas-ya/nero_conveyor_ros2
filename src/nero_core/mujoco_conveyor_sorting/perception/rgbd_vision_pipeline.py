"""Oracle-free RGB-D perception for upright conveyor workpieces.

The runtime API of this module accepts only camera products and calibration:
RGB, metric depth, intrinsics, a capture-time camera transform, timestamp and a
belt-speed measurement.  It deliberately has no MuJoCo dependency and no
argument through which a simulator body, geom label, category or pose can be
passed.

The CPU baseline exploits constraints that are true for this sorting cell:
parts rest upright on a known support plane and move approximately along the
conveyor X axis.  It removes the support plane in depth, extracts connected
components, classifies rotation-invariant geometry against CAD-derived
prototypes, estimates XY/yaw, and maintains independent visual track IDs.
Heavy learned 6-D backends can later implement the same ``process`` contract.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int32]

_OPENCV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _normalize_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _axis_yaw_near(yaw: float, reference: float | None) -> float:
    canonical = float((yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi)
    if reference is None:
        return canonical
    candidates = [canonical + multiple * math.pi for multiple in range(-2, 3)]
    return _normalize_angle(
        min(
            candidates,
            key=lambda candidate: abs(
                _normalize_angle(candidate - float(reference))
            ),
        )
    )


def _yaw_quaternion(yaw: float) -> FloatArray:
    half = 0.5 * float(yaw)
    return np.asarray(
        [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64
    )


@dataclass(frozen=True, slots=True)
class GeometryPrototype:
    """One class prototype obtained from CAD dimensions or synthetic RGB-D."""

    category_id: int
    dimensions: FloatArray
    symmetric: bool = False
    feature_mean: FloatArray | None = None
    feature_scale: FloatArray | None = None

    def __post_init__(self) -> None:
        dimensions = _finite_array(self.dimensions, (3,), "dimensions")
        if np.any(dimensions <= 0.0):
            raise ValueError("prototype dimensions must be positive")
        object.__setattr__(self, "category_id", int(self.category_id))
        object.__setattr__(self, "dimensions", dimensions)
        if self.feature_mean is None:
            if self.feature_scale is not None:
                raise ValueError("feature_scale requires feature_mean")
            return
        mean = np.asarray(self.feature_mean, dtype=np.float64).reshape(-1)
        if mean.shape != (9,) or not np.all(np.isfinite(mean)):
            raise ValueError("feature_mean must contain nine finite values")
        scale = np.asarray(
            np.ones(9) if self.feature_scale is None else self.feature_scale,
            dtype=np.float64,
        ).reshape(-1)
        if scale.shape != (9,) or not np.all(np.isfinite(scale)):
            raise ValueError("feature_scale must contain nine finite values")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale values must be positive")
        object.__setattr__(self, "feature_mean", mean.copy())
        object.__setattr__(self, "feature_scale", scale.copy())

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "GeometryPrototype":
        calibrated = source.get(
            "rgbd_geometry_prototype", source.get("vision_prototype", {})
        )
        if not isinstance(calibrated, Mapping):
            calibrated = {}
        return cls(
            category_id=int(source["category_id"]),
            dimensions=np.asarray(source["dimensions"], dtype=np.float64),
            symmetric=bool(source.get("symmetric", False)),
            feature_mean=(
                None
                if calibrated.get("feature_mean") is None
                else np.asarray(calibrated["feature_mean"], dtype=np.float64)
            ),
            feature_scale=(
                None
                if calibrated.get("feature_scale") is None
                else np.asarray(calibrated["feature_scale"], dtype=np.float64)
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentMeasurement:
    """One foreground component measured without semantic or track truth."""

    mask: BoolArray
    position: FloatArray
    raw_yaw: float
    orientation_confidence: float
    feature_vector: FloatArray
    mask_pixels: int
    valid_depth_pixels: int
    measurement_confidence: float


@dataclass(frozen=True, slots=True)
class VisionPoseDetection:
    """Tracked pose; category_id=-1/category_confidence=0 means unclassified."""

    track_id: int
    category_id: int
    position: FloatArray
    quaternion: FloatArray
    timestamp: float
    confidence: float
    category_confidence: float
    mask_pixels: int
    valid_depth_pixels: int
    yaw: float
    orientation_confidence: float
    visible: bool
    observation_age: float
    feature_vector: FloatArray

    @property
    def object_id(self) -> int:
        return self.track_id

    @property
    def pose(self) -> FloatArray:
        return np.r_[self.position, self.quaternion]


@dataclass(slots=True)
class _Track:
    track_id: int
    position: FloatArray
    velocity: FloatArray
    yaw: float
    category_scores: FloatArray
    feature_vector: FloatArray
    last_seen: float
    last_update: float
    confidence: float
    category_confidence: float
    mask_pixels: int
    valid_depth_pixels: int
    orientation_confidence: float
    belt_distance_at_seen: float = 0.0
    category_evidence_weight: float = 0.0


def deproject_depth_to_world(
    depth: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    camera_to_world: npt.ArrayLike,
    *,
    camera_convention: Literal["mujoco", "opencv"] = "mujoco",
) -> FloatArray:
    """Return an ``(H, W, 3)`` world point for every metric depth pixel.

    Existing simulation callers publish a MuJoCo/OpenGL camera transform and
    therefore retain the historical ``mujoco`` default.  Physical RGB-D
    cameras normally publish an OpenCV (+Z forward, +Y down) transform and
    must pass ``camera_convention="opencv"`` to avoid a duplicate axis flip.
    """

    depth_image = np.asarray(depth, dtype=np.float64)
    if depth_image.ndim != 2:
        raise ValueError("depth must have shape (height, width)")
    matrix = _finite_array(intrinsics, (3, 3), "intrinsics")
    transform = _finite_array(camera_to_world, (4, 4), "camera_to_world")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera_to_world must be homogeneous")
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    rows, columns = np.indices(depth_image.shape, dtype=np.float64)
    points_opencv = np.stack(
        (
            (columns - cx) * depth_image / fx,
            (rows - cy) * depth_image / fy,
            depth_image,
        ),
        axis=-1,
    )
    if camera_convention == "mujoco":
        points_camera = points_opencv @ _OPENCV_TO_MUJOCO.T
    elif camera_convention == "opencv":
        points_camera = points_opencv
    else:
        raise ValueError("camera_convention must be 'mujoco' or 'opencv'")
    return points_camera @ transform[:3, :3].T + transform[:3, 3]


def connected_component_masks(
    foreground: npt.ArrayLike,
    *,
    minimum_pixels: int,
) -> tuple[BoolArray, ...]:
    """Extract eight-connected masks using NumPy plus a sparse flood fill."""

    mask = np.asarray(foreground, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("foreground must have shape (height, width)")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: list[BoolArray] = []
    for row_value, column_value in zip(*np.nonzero(mask), strict=True):
        row, column = int(row_value), int(column_value)
        if visited[row, column]:
            continue
        queue: deque[tuple[int, int]] = deque([(row, column)])
        visited[row, column] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            current_row, current_column = queue.popleft()
            pixels.append((current_row, current_column))
            for row_offset, column_offset in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                next_row = current_row + row_offset
                next_column = current_column + column_offset
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        if len(pixels) < minimum_pixels:
            continue
        component = np.zeros_like(mask)
        component[tuple(np.asarray(pixels, dtype=np.int64).T)] = True
        components.append(component)
    return tuple(components)


class DepthGeometryVisionPipeline:
    """CPU RGB-D detector, optional prototype classifier and conveyor tracker.

    Disable classification when camera-view prototypes are unavailable. The
    detector and tracker then use geometric confidence alone and preserve the
    measured axis yaw without assuming a class-specific rotational symmetry.
    """

    def __init__(
        self,
        prototypes: Iterable[GeometryPrototype | Mapping[str, Any]],
        *,
        support_plane_height: float,
        roi_x: Sequence[float] = (-0.58, 0.24),
        roi_y: Sequence[float] = (-0.115, 0.115),
        minimum_height: float = 0.004,
        maximum_height: float = 0.105,
        minimum_mask_pixels: int = 24,
        maximum_footprint: float = 0.13,
        minimum_footprint: float = 0.012,
        robust_extent_quantile: float = 0.01,
        association_distance: float = 0.075,
        new_track_max_x: float = -0.20,
        trusted_measurement_max_x: float = -0.18,
        max_track_age: float = 2.40,
        coast_confidence_half_life: float = 1.00,
        category_ema: float = 0.65,
        category_evidence_mode: str = "ema",
        velocity_ema: float = 0.55,
        max_supported_lateral_speed: float = 0.001,
        category_temperature: float = 1.0,
        minimum_category_confidence: float = 0.05,
        minimum_detection_confidence: float = 0.06,
        classification_enabled: bool = True,
        depth_projector: Callable[..., FloatArray] | None = None,
    ) -> None:
        # Optional calibration-only implementation, e.g. cached rays for a
        # fixed camera. The default keeps the existing projection unchanged.
        self.depth_projector = depth_projector
        parsed = tuple(
            item
            if isinstance(item, GeometryPrototype)
            else GeometryPrototype.from_mapping(item)
            for item in prototypes
        )
        self.classification_enabled = bool(classification_enabled)
        if not parsed and self.classification_enabled:
            raise ValueError("at least one geometry prototype is required")
        categories = [item.category_id for item in parsed]
        if len(set(categories)) != len(categories):
            raise ValueError("prototype category IDs must be unique")
        self.prototypes = tuple(sorted(parsed, key=lambda item: item.category_id))
        self.category_ids = np.asarray(
            [item.category_id for item in self.prototypes], dtype=np.int64
        )
        self.support_plane_height = float(support_plane_height)
        self.roi_x = self._bounds(roi_x, "roi_x")
        self.roi_y = self._bounds(roi_y, "roi_y")
        self.minimum_height = float(minimum_height)
        self.maximum_height = float(maximum_height)
        self.minimum_mask_pixels = int(minimum_mask_pixels)
        self.maximum_footprint = float(maximum_footprint)
        self.minimum_footprint = float(minimum_footprint)
        self.robust_extent_quantile = float(robust_extent_quantile)
        self.association_distance = float(association_distance)
        self.new_track_max_x = float(new_track_max_x)
        self.trusted_measurement_max_x = float(trusted_measurement_max_x)
        self.max_track_age = float(max_track_age)
        self.coast_confidence_half_life = float(coast_confidence_half_life)
        self.category_ema = float(category_ema)
        self.category_evidence_mode = str(category_evidence_mode)
        if self.category_evidence_mode not in {"ema", "cumulative"}:
            raise ValueError("category_evidence_mode must be 'ema' or 'cumulative'")
        self.velocity_ema = float(velocity_ema)
        self.max_supported_lateral_speed = float(max_supported_lateral_speed)
        if (not np.isfinite(self.max_supported_lateral_speed)
                or self.max_supported_lateral_speed < 0.0):
            raise ValueError("max_supported_lateral_speed must be finite and non-negative")
        self.category_temperature = float(category_temperature)
        self.minimum_category_confidence = float(minimum_category_confidence)
        self.minimum_detection_confidence = float(
            minimum_detection_confidence
        )
        if not 0.0 <= self.minimum_height < self.maximum_height:
            raise ValueError("height limits are invalid")
        for name in ("new_track_max_x", "trusted_measurement_max_x"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.trusted_measurement_max_x < self.new_track_max_x:
            raise ValueError(
                "trusted_measurement_max_x cannot be upstream of "
                "new_track_max_x"
            )
        if self.minimum_mask_pixels < 2:
            raise ValueError("minimum_mask_pixels must be at least two")
        if not 0.0 <= self.robust_extent_quantile < 0.25:
            raise ValueError("robust_extent_quantile must be in [0, 0.25)")
        for name in (
            "maximum_footprint",
            "minimum_footprint",
            "association_distance",
            "max_track_age",
            "coast_confidence_half_life",
            "category_temperature",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("category_ema", "velocity_ema"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in (
            "minimum_category_confidence",
            "minimum_detection_confidence",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 0
        self.last_instance_labels: IntArray | None = None
        self.last_foreground_mask: BoolArray | None = None
        self.last_inference_seconds = 0.0
        self.reset()

    @staticmethod
    def _bounds(value: Sequence[float], name: str) -> tuple[float, float]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain two finite values")
        if array[0] >= array[1]:
            raise ValueError(f"{name} lower bound must be smaller than upper")
        return float(array[0]), float(array[1])

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 0
        self.last_instance_labels = None
        self.last_foreground_mask = None
        self.last_inference_seconds = 0.0
        self._belt_distance = 0.0
        self._belt_sample_timestamp: float | None = None
        self._belt_sample_speed = 0.0

    def extract_components(
        self,
        rgb: npt.ArrayLike | None,
        depth: npt.ArrayLike,
        intrinsics: npt.ArrayLike,
        camera_to_world: npt.ArrayLike,
        *,
        camera_convention: Literal["mujoco", "opencv"] = "mujoco",
        foreground_mask: npt.ArrayLike | None = None,
        roi_selects_components: bool = False,
        points_world: npt.ArrayLike | None = None,
    ) -> tuple[ComponentMeasurement, ...]:
        depth_image = np.asarray(depth, dtype=np.float64)
        if depth_image.ndim != 2:
            raise ValueError("depth must have shape (height, width)")
        # The current geometry baseline is intentionally depth-only.  Keep an
        # optional RGB argument in the backend contract so a learned detector
        # can be substituted later, but do not require the control loop to pay
        # for an RGB render whose pixels are never read.
        if rgb is not None:
            rgb_image = np.asarray(rgb)
            if rgb_image.shape != (*depth_image.shape, 3):
                raise ValueError("rgb must have shape (height, width, 3)")
        matrix = _finite_array(intrinsics, (3, 3), "intrinsics")
        # Read-only HxWx3 in THIS pipeline's frame (the live caller supplies
        # measured belt coordinates). Do not transform caller points again.
        if points_world is None:
            points_world = (self.depth_projector or deproject_depth_to_world)(
                depth_image, matrix, camera_to_world, camera_convention=camera_convention)
        else:
            points_world = np.asarray(points_world, dtype=np.float64)
            if points_world.shape != (*depth_image.shape, 3):
                raise ValueError('points_world must match native depth HxWx3 in pipeline frame')
        relative_height = points_world[..., 2] - self.support_plane_height
        foreground = (
            np.isfinite(depth_image)
            & (depth_image > 1e-6)
            & np.all(np.isfinite(points_world), axis=-1)
            & (relative_height >= self.minimum_height)
            & (relative_height <= self.maximum_height)
        )
        inside_roi = (
            (points_world[..., 0] >= self.roi_x[0])
            & (points_world[..., 0] <= self.roi_x[1])
            & (points_world[..., 1] >= self.roi_y[0])
            & (points_world[..., 1] <= self.roi_y[1])
        )
        if not roi_selects_components:
            foreground &= inside_roi
        if foreground_mask is not None:
            allowed = np.asarray(foreground_mask, dtype=bool)
            if allowed.shape != depth_image.shape:
                raise ValueError("foreground_mask must match native depth shape")
            foreground &= allowed
        self.last_foreground_mask = foreground.copy()
        masks = connected_component_masks(
            foreground, minimum_pixels=self.minimum_mask_pixels
        )
        measurements: list[ComponentMeasurement] = []
        for mask in masks:
            # A measured search rectangle is not an object's silhouette.
            # Select components intersecting it, then measure their complete
            # observed foreground. The default preserves historical crops.
            if roi_selects_components and not np.any(mask & inside_roi):
                continue
            points = points_world[mask]
            depths = depth_image[mask]
            if len(points) < self.minimum_mask_pixels:
                continue
            xy = points[:, :2]
            centered = xy - np.mean(xy, axis=0)
            covariance = centered.T @ centered / max(len(centered) - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            projected = xy @ eigenvectors
            quantile = self.robust_extent_quantile
            lower = np.quantile(projected, quantile, axis=0)
            upper = np.quantile(projected, 1.0 - quantile, axis=0)
            extents = np.maximum(upper - lower, 1e-9)
            centre_xy = (0.5 * (lower + upper)) @ eigenvectors.T
            footprint = np.sort(extents)[::-1]
            if (
                footprint[0] > self.maximum_footprint
                or footprint[1] < self.minimum_footprint
            ):
                continue
            largest = max(float(eigenvalues[-1]), 0.0)
            smallest = max(float(eigenvalues[0]), 0.0)
            anisotropy = (
                (largest - smallest) / (largest + smallest)
                if largest + smallest > 1e-15
                else 0.0
            )
            principal = eigenvectors[:, -1]
            raw_yaw = math.atan2(float(principal[1]), float(principal[0]))
            heights = np.clip(
                points[:, 2] - self.support_plane_height, 0.0, None
            )
            object_height = max(float(np.quantile(heights, 0.99)), 1e-6)
            # A pixel spans approximately z/f metres for this near-normal top
            # camera.  Adding one pixel to each silhouette side compensates
            # raster-centre erosion before comparing CAD bounding dimensions.
            pixel_width = float(np.median(depths)) / max(
                float(matrix[0, 0]), 1e-9
            )
            pixel_height = float(np.median(depths)) / max(
                float(matrix[1, 1]), 1e-9
            )
            corrected_footprint = footprint + (
                pixel_width + pixel_height
            )
            pixel_areas = np.square(depths) / max(
                float(matrix[0, 0] * matrix[1, 1]), 1e-12
            )
            projected_area = float(np.sum(pixel_areas))
            fill_ratio = float(
                np.clip(
                    projected_area
                    / max(
                        corrected_footprint[0] * corrected_footprint[1], 1e-12
                    ),
                    0.0,
                    1.5,
                )
            )
            height_quantiles = np.quantile(heights, [0.25, 0.50, 0.75])
            feature = np.asarray(
                [
                    corrected_footprint[0],
                    corrected_footprint[1],
                    object_height,
                    math.sqrt(max(projected_area, 0.0)),
                    fill_ratio,
                    *(height_quantiles / object_height),
                    anisotropy,
                ],
                dtype=np.float64,
            )
            area_score = min(1.0, len(points) / max(100, self.minimum_mask_pixels * 8))
            boundary_distance = min(
                float(centre_xy[0] - self.roi_x[0]),
                float(self.roi_x[1] - centre_xy[0]),
                float(centre_xy[1] - self.roi_y[0]),
                float(self.roi_y[1] - centre_xy[1]),
            )
            edge_score = float(
                np.clip(boundary_distance / max(0.5 * footprint[0], 1e-6), 0.2, 1.0)
            )
            measurements.append(
                ComponentMeasurement(
                    mask=mask.copy(),
                    position=np.asarray(
                        [
                            float(centre_xy[0]),
                            float(centre_xy[1]),
                            self.support_plane_height,
                        ],
                        dtype=np.float64,
                    ),
                    raw_yaw=_normalize_angle(raw_yaw),
                    orientation_confidence=float(np.clip(anisotropy, 0.0, 1.0)),
                    feature_vector=feature,
                    mask_pixels=int(np.count_nonzero(mask)),
                    valid_depth_pixels=len(points),
                    measurement_confidence=float(np.clip(area_score * edge_score, 0.0, 1.0)),
                )
            )
        return tuple(measurements)

    def classify(
        self, feature_vector: npt.ArrayLike
    ) -> tuple[int, float, FloatArray]:
        feature = _finite_array(feature_vector, (9,), "feature_vector")
        if not self.classification_enabled:
            return -1, 0.0, np.empty(0, dtype=np.float64)
        distances: list[float] = []
        for prototype in self.prototypes:
            if prototype.feature_mean is not None:
                residual = (feature - prototype.feature_mean) / prototype.feature_scale
                distance = float(np.sqrt(np.mean(np.square(residual))))
            else:
                expected_footprint = np.sort(prototype.dimensions[:2])[::-1]
                dimensional_scale = np.asarray([0.006, 0.006, 0.004])
                residual = (
                    feature[:3]
                    - np.r_[expected_footprint, prototype.dimensions[2]]
                ) / dimensional_scale
                distance = float(np.sqrt(np.mean(np.square(residual))))
            distances.append(distance)
        distance_array = np.asarray(distances, dtype=np.float64)
        shifted = distance_array - float(np.min(distance_array))
        logits = -0.5 * np.square(shifted / self.category_temperature)
        logits -= float(np.max(logits))
        scores = np.exp(logits)
        scores /= max(float(np.sum(scores)), 1e-12)
        best_index = int(np.argmax(scores))
        absolute_score = math.exp(
            -0.5 * (float(distance_array[best_index]) / 2.5) ** 2
        )
        confidence = float(
            np.clip(float(scores[best_index]) * absolute_score, 0.0, 1.0)
        )
        return int(self.category_ids[best_index]), confidence, scores

    def process(
        self,
        rgb: npt.ArrayLike | None,
        depth: npt.ArrayLike,
        intrinsics: npt.ArrayLike,
        camera_to_world: npt.ArrayLike,
        *,
        capture_timestamp: float,
        belt_speed: float,
        belt_position: float | None = None,
        camera_convention: Literal["mujoco", "opencv"] = "mujoco",
        robot_masked_workspace: bool = False,
    ) -> tuple[VisionPoseDetection, ...]:
        """Detect, classify and track one RGB-D capture."""

        started_at = time.perf_counter()
        timestamp = float(capture_timestamp)
        speed = float(belt_speed)
        if not np.isfinite(timestamp) or not np.isfinite(speed):
            raise ValueError("timestamp and belt_speed must be finite")
        if self._belt_sample_timestamp is not None and timestamp < self._belt_sample_timestamp:
            raise ValueError("capture timestamps must be nondecreasing; reset the tracker first")
        if belt_position is not None:
            distance = float(belt_position)
            if not np.isfinite(distance):
                raise ValueError("belt_position must be finite")
            self._belt_distance = distance
        elif self._belt_sample_timestamp is not None:
            # Standalone callers can supply sampled encoder speed. Integrate
            # consecutive samples instead of multiplying the newest speed by
            # the entire occlusion age, which creates age * dv/dt ghost motion.
            self._belt_distance += 0.5 * (self._belt_sample_speed + speed) * (
                timestamp - self._belt_sample_timestamp
            )
        self._belt_sample_timestamp, self._belt_sample_speed = timestamp, speed
        measurements = self.extract_components(
            rgb,
            depth,
            intrinsics,
            camera_to_world,
            camera_convention=camera_convention,
        )
        classified: list[
            tuple[ComponentMeasurement, int, float, FloatArray]
        ] = []
        for measurement in measurements:
            category, confidence, scores = self.classify(
                measurement.feature_vector
            )
            if (
                (self.classification_enabled and confidence < self.minimum_category_confidence)
                or measurement.measurement_confidence * (confidence if self.classification_enabled else 1.0)
                < self.minimum_detection_confidence
            ):
                continue
            classified.append((measurement, category, confidence, scores))

        # The fixed top camera owns a deliberately upstream acquisition zone.
        # Once a part enters the robot workspace, fingers and links can merge
        # with its depth component.  Treating that merged silhouette as a new
        # pose makes the tracker chase the gripper and is substantially less
        # safe than bounded conveyor-encoder propagation.  Measurements past
        # this line therefore cannot update an existing track; the track still
        # expires after ``max_track_age`` if no clean observation returns.
        trusted_indices = {
            index
            for index, (measurement, *_rest) in enumerate(classified)
            if measurement.position[0] <= self.trusted_measurement_max_x or robot_masked_workspace
        }

        unmatched_tracks = set(self._tracks)
        unmatched_measurements = set(trusted_indices)
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            elapsed = max(0.0, timestamp - track.last_seen)
            predicted = self._coasted_position(track, elapsed, speed)
            for measurement_index, (measurement, _, _, scores) in enumerate(classified):
                if robot_masked_workspace and measurement.position[0]>self.trusted_measurement_max_x:
                    ratio=measurement.feature_vector[:3]/np.maximum(track.feature_vector[:3],1e-6)
                    if np.any(ratio<.65) or np.any(ratio>1.35):continue
                distance = float(
                    np.linalg.norm(measurement.position[:2] - predicted[:2])
                )
                category_affinity = (
                    float(np.dot(track.category_scores, scores))
                    if self.classification_enabled else 1.0
                )
                cost = distance + 0.018 * (1.0 - category_affinity)
                candidate_pairs.append((cost, track_id, measurement_index))
        visible_assignment: dict[int, int] = {}
        for cost, track_id, measurement_index in sorted(candidate_pairs):
            if track_id not in unmatched_tracks or measurement_index not in unmatched_measurements:
                continue
            if cost > self.association_distance:
                continue
            unmatched_tracks.remove(track_id)
            unmatched_measurements.remove(measurement_index)
            visible_assignment[track_id] = measurement_index

        for track_id, measurement_index in visible_assignment.items():
            measurement, _, category_confidence, scores = classified[measurement_index]
            track = self._tracks[track_id]
            elapsed = max(timestamp - track.last_seen, 1e-6)
            measured_velocity = (measurement.position - track.position) / elapsed
            measured_velocity[2] = 0.0
            track.velocity = (
                self.velocity_ema * track.velocity
                + (1.0 - self.velocity_ema) * measured_velocity
            )
            track.position = measurement.position.copy()
            track.yaw = _axis_yaw_near(measurement.raw_yaw, track.yaw)
            evidence_weight = float(np.clip(
                measurement.measurement_confidence * category_confidence, 0.0, 1.0
            ))
            if self.classification_enabled and self.category_evidence_mode == "cumulative":
                if timestamp <= track.last_seen + 1e-9:
                    evidence_weight = 0.0
                # Shape class is a property of this identity, unlike position
                # and velocity. Preserve its visual evidence across views so
                # a few ambiguous projections cannot erase a long clean view.
                # Scores stay normalized for the association affinity below.
                total_weight = track.category_evidence_weight + evidence_weight
                if total_weight > 0.0:
                    track.category_scores = (
                        track.category_evidence_weight * track.category_scores
                        + evidence_weight * scores
                    ) / total_weight
                track.category_evidence_weight = total_weight
            elif self.classification_enabled:
                track.category_scores = (
                    self.category_ema * track.category_scores
                    + (1.0 - self.category_ema) * scores
                )
            track.category_scores /= max(float(np.sum(track.category_scores)), 1e-12)
            track.feature_vector = measurement.feature_vector.copy()
            track.last_seen = timestamp
            track.belt_distance_at_seen = self._belt_distance
            track.last_update = timestamp
            track.confidence = float(
                np.clip(
                    measurement.measurement_confidence
                    * (category_confidence if self.classification_enabled else 1.0),
                    0.0,
                    1.0,
                )
            )
            track.category_confidence = (
                float(np.max(track.category_scores)) if self.classification_enabled else 0.0
            )
            track.mask_pixels = measurement.mask_pixels
            track.valid_depth_pixels = measurement.valid_depth_pixels
            track.orientation_confidence = measurement.orientation_confidence

        for measurement_index in sorted(unmatched_measurements):
            measurement, _, category_confidence, scores = classified[measurement_index]
            # Parts enter from the upstream side.  Requiring new identities to
            # originate there prevents a descending gripper or a lifted part
            # from being reintroduced as a fresh workpiece track.
            if measurement.position[0] > self.new_track_max_x:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            prototype = self.prototypes[int(np.argmax(scores))] if self.classification_enabled else None
            evidence_weight = float(np.clip(
                measurement.measurement_confidence * category_confidence, 0.0, 1.0
            ))
            yaw = 0.0 if prototype is not None and prototype.symmetric else _axis_yaw_near(
                measurement.raw_yaw, None
            )
            self._tracks[track_id] = _Track(
                track_id=track_id,
                position=measurement.position.copy(),
                velocity=np.asarray([speed, 0.0, 0.0], dtype=np.float64),
                yaw=yaw,
                # A zero-weight first detection provides only a provisional
                # valid class distribution. Its stored evidence weight is
                # zero, so the first positive-weight view replaces it fully.
                category_scores=scores.copy(),
                feature_vector=measurement.feature_vector.copy(),
                last_seen=timestamp,
                last_update=timestamp,
                confidence=float(
                    np.clip(
                        measurement.measurement_confidence
                        * (category_confidence if self.classification_enabled else 1.0),
                        0.0,
                        1.0,
                    )
                ),
                category_confidence=(
                    float(np.max(scores))
                    if self.classification_enabled and self.category_evidence_mode == "cumulative"
                    else category_confidence
                ),
                mask_pixels=measurement.mask_pixels,
                valid_depth_pixels=measurement.valid_depth_pixels,
                orientation_confidence=measurement.orientation_confidence,
                belt_distance_at_seen=self._belt_distance,
                category_evidence_weight=evidence_weight,
            )
            visible_assignment[track_id] = measurement_index

        labels = np.full(np.asarray(depth).shape, -1, dtype=np.int32)
        for track_id, measurement_index in visible_assignment.items():
            labels[classified[measurement_index][0].mask] = int(track_id)
        self.last_instance_labels = labels

        expired: list[int] = []
        detections: list[VisionPoseDetection] = []
        for track_id, track in sorted(self._tracks.items()):
            age = max(0.0, timestamp - track.last_seen)
            if age > self.max_track_age:
                expired.append(track_id)
                continue
            visible = track_id in visible_assignment
            if visible:
                position = track.position.copy()
                confidence = track.confidence
            else:
                position = self._coasted_position(track, age, speed)
                confidence = track.confidence * math.exp(
                    -math.log(2.0) * age / self.coast_confidence_half_life
                )
            category_index = int(np.argmax(track.category_scores)) if self.classification_enabled else -1
            prototype = self.prototypes[category_index] if self.classification_enabled else None
            yaw = 0.0 if prototype is not None and prototype.symmetric else track.yaw
            detections.append(
                VisionPoseDetection(
                    track_id=track_id,
                    category_id=int(self.category_ids[category_index]) if self.classification_enabled else -1,
                    position=position,
                    quaternion=_yaw_quaternion(yaw),
                    timestamp=timestamp,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    category_confidence=float(
                        np.clip(track.category_confidence, 0.0, 1.0)
                    ),
                    mask_pixels=track.mask_pixels if visible else 0,
                    valid_depth_pixels=track.valid_depth_pixels if visible else 0,
                    yaw=yaw,
                    orientation_confidence=(
                        track.orientation_confidence if visible else 0.0
                    ),
                    visible=visible,
                    observation_age=age,
                    feature_vector=track.feature_vector.copy(),
                )
            )
            track.last_update = timestamp
        for track_id in expired:
            self._tracks.pop(track_id, None)
        self.last_inference_seconds = time.perf_counter() - started_at
        return tuple(detections)

    def _coasted_position(self, track: _Track, age: float, speed: float) -> FloatArray:
        """Propagate an occluded, belt-supported track without lateral drift.

        A partial depth silhouette can give the last velocity fit a large
        lateral component. Clamp it here, before emitting coasted observations:
        a downstream predictor cannot undo drift already integrated into those
        positions. Fresh measurements still retain their measured coordinates.
        """
        velocity = np.array([
            0.0,
            np.clip(track.velocity[1], -self.max_supported_lateral_speed,
                    self.max_supported_lateral_speed),
            0.0,
        ])
        position = track.position + velocity * age
        position[0] += self._belt_distance - track.belt_distance_at_seen
        return position


__all__ = [
    "ComponentMeasurement",
    "DepthGeometryVisionPipeline",
    "GeometryPrototype",
    "VisionPoseDetection",
    "connected_component_masks",
    "deproject_depth_to_world",
]

"""Fixed camera + rigidly mounted moving target: B_T_G G_T_T = B_T_C C_T_T.

OpenCV calibrateHandEye receives *base-to-gripper*, not gripper-to-base,
in this configuration. All translations are metres; cameras use OpenCV axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import logging

import cv2
import numpy as np

from .geometry import inverse, mean_transform, rigid_transform, transform_distance
from .pnp_audit import pnp_input_record, validate_pnp_input_record


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chessboard:
    columns: int = 11  # INNER corners, 12 x 9 squares on the current glass board
    rows: int = 8
    square_size_m: float = 0.02

    def __post_init__(self):
        if (not isinstance(self.columns, int) or not isinstance(self.rows, int)
                or self.columns < 3 or self.rows < 3):
            raise ValueError("chessboard columns/rows must be integer inner-corner counts >= 3")
        if not np.isfinite(self.square_size_m) or self.square_size_m <= 0:
            raise ValueError("square_size_m must be finite and positive")

    @property
    def object_points(self) -> np.ndarray:
        points = np.zeros((self.columns * self.rows, 3), dtype=np.float64)
        points[:, :2] = np.mgrid[:self.columns, :self.rows].T.reshape(-1, 2)
        return points * self.square_size_m


@dataclass(frozen=True)
class CalibrationSample:
    gripper_to_base: np.ndarray
    target_to_camera: np.ndarray
    reprojection_error_px: float = 0.0
    sample_id: str = ""

    def __post_init__(self):
        for name in ("gripper_to_base", "target_to_camera"):
            transform = rigid_transform(getattr(self, name), name)
            transform.setflags(write=False)
            object.__setattr__(self, name, transform)
        if not np.isfinite(self.reprojection_error_px) or self.reprojection_error_px < 0:
            raise ValueError("reprojection_error_px must be finite and non-negative")


def camera_parameters(intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(intrinsics["matrix"], dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)):
        raise ValueError("invalid pinhole camera matrix")
    distortion = np.asarray(intrinsics.get("distortion", []), dtype=np.float64).reshape(-1)
    if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("invalid OpenCV distortion coefficients")
    model = str(intrinsics.get("distortion_model", "opencv")).lower().split(".")[-1]
    if model not in {"none", "opencv", "brown_conrady", "brown_conrady_k6"}:
        raise ValueError(f"unsupported PnP distortion model: {model}; use calibrated OpenCV intrinsics")
    if model == "none" and np.any(distortion):
        raise ValueError("distortion model NONE must have zero coefficients")
    return matrix, distortion


def estimate_target_pose(corners, board: Chessboard, intrinsics: dict, *,
                         pnp_audit=None, image_context=None):
    matrix, distortion = camera_parameters(intrinsics)
    matrix, distortion = matrix.copy(), distortion.copy()
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if points.shape != (board.rows * board.columns, 2) or not np.isfinite(points).all():
        raise ValueError("a complete finite chessboard corner set is required")
    # Snapshot and validate here, immediately before the actual solver call.
    # The record contains these arrays, not merely the config path or SDK K/D.
    inputs = pnp_input_record(matrix, distortion, intrinsics, image_context)
    if pnp_audit is not None:
        pnp_audit.update(inputs)
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("solvePnP inputs: %s", json.dumps(inputs, allow_nan=False))
    validate_pnp_input_record(inputs)
    ok, rvec, tvec = cv2.solvePnP(board.object_points, points, matrix, distortion,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed")
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    camera_points = board.object_points @ transform[:3, :3].T + transform[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("PnP placed the board behind the camera")
    projected = cv2.projectPoints(board.object_points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1))))
    return rigid_transform(transform), rms


def detect_chessboard(image_rgb, board: Chessboard, intrinsics: dict, *, reverse=False,
                      pnp_audit=None, image_context=None):
    """Return target-to-color-camera, reprojection RMS and ordered corners.

    Plain chessboards can reverse their physical origin by 180 degrees. The
    caller displays corner 0; reverse=True lets the operator correct its order.
    """
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("chessboard image must be RGB uint8")
    if image.shape[:2] != (int(intrinsics["height"]), int(intrinsics["width"])):
        raise ValueError("image resolution differs from saved camera intrinsics")
    context = dict(image_context or {})
    actual_size = [image.shape[1], image.shape[0]]
    if context.get("image_size") is not None and list(context["image_size"]) != actual_size:
        raise ValueError("PnP context image size differs from the actual RGB array")
    context["image_size"] = actual_size
    context.setdefault("image_size_source", "rgb_array")
    camera_parameters(intrinsics)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (board.columns, board.rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise ValueError("complete chessboard not detected")
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    if reverse:
        corners = corners[::-1].copy()
    transform, error = estimate_target_pose(corners, board, intrinsics,
                                            pnp_audit=pnp_audit, image_context=context)
    return transform, error, corners.reshape(-1, 2)


def _motion_quality(samples):
    rotations = []
    translation_system = []
    for first, second in combinations(samples, 2):
        # B_T_Gj inv(B_T_Gi) are relative motions for eye-to-hand.
        relative = second.gripper_to_base @ inverse(first.gripper_to_base)
        rotations.append(cv2.Rodrigues(relative[:3, :3])[0].ravel())
        translation_system.append(relative[:3, :3] - np.eye(3))
    axes_singular = np.linalg.svd(np.asarray(rotations), compute_uv=False)
    translation_singular = np.linalg.svd(np.vstack(translation_system), compute_uv=False)
    maximum_angle = float(np.degrees(np.max(np.linalg.norm(rotations, axis=1))))
    if (maximum_angle < 10 or axes_singular[1] < 0.05 * axes_singular[0]
            or translation_singular[-1] < 0.02 * translation_singular[0]):
        raise ValueError("degenerate poses: rotate around at least two non-parallel axes, "
                         "with >10 degrees of spread; translation-only/repeated poses cannot calibrate")
    return {"max_relative_rotation_deg": maximum_angle,
            "rotation_vector_singular_values": axes_singular.tolist(),
            "translation_system_singular_values": translation_singular.tolist()}


def solve_eye_to_hand(samples, *, method="park", min_samples=8,
                      max_reprojection_px=1.5, max_translation_error_m=0.005,
                      max_rotation_error_deg=1.0) -> dict:
    samples = list(samples)
    if not isinstance(min_samples, int) or min_samples < 3:
        raise ValueError("min_samples must be an integer >= 3")
    limits = [max_reprojection_px, max_translation_error_m, max_rotation_error_deg]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("quality limits must be finite and positive")
    if len(samples) < min_samples:
        raise ValueError(f"need at least {min_samples} samples; got {len(samples)} (recommend 15-25)")
    unique = []
    for sample in samples:
        if not any(np.allclose(sample.gripper_to_base, previous.gripper_to_base, atol=1e-6, rtol=0)
                   for previous in unique):
            unique.append(sample)
    if len(unique) < min_samples:
        raise ValueError(f"need at least {min_samples} distinct robot poses; got {len(unique)}")
    methods = {"park": cv2.CALIB_HAND_EYE_PARK, "tsai": cv2.CALIB_HAND_EYE_TSAI,
               "horaud": cv2.CALIB_HAND_EYE_HORAUD}
    if method not in methods:
        raise ValueError(f"unknown hand-eye method: {method}")
    quality = _motion_quality(samples)
    base_to_gripper = [inverse(sample.gripper_to_base) for sample in samples]
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in base_to_gripper], [t[:3, 3] for t in base_to_gripper],
        [s.target_to_camera[:3, :3] for s in samples],
        [s.target_to_camera[:3, 3] for s in samples], method=methods[method],
    )
    camera_to_base = np.eye(4)
    camera_to_base[:3, :3], camera_to_base[:3, 3] = rotation, translation.ravel()
    try:
        camera_to_base = rigid_transform(camera_to_base, "solved camera_to_base")
    except ValueError as error:
        raise ValueError("invalid hand-eye solution; check pose direction/units, paired images, "
                         "and consistent chessboard origin") from error
    # A rigidly mounted target must occupy the same pose in gripper coordinates.
    targets = [inverse(s.gripper_to_base) @ camera_to_base @ s.target_to_camera for s in samples]
    target_to_gripper = mean_transform(targets)
    residuals = []
    for index, (sample, target) in enumerate(zip(samples, targets)):
        distance, angle = transform_distance(target_to_gripper, target)
        residuals.append({"sample_id": sample.sample_id or str(index),
                          "translation_error_m": distance, "rotation_error_deg": angle,
                          "reprojection_error_px": float(sample.reprojection_error_px)})
    distance = np.array([r["translation_error_m"] for r in residuals])
    angle = np.array([r["rotation_error_deg"] for r in residuals])
    reprojection = np.array([r["reprojection_error_px"] for r in residuals])
    reasons = []
    for values, limit, label in ((distance, max_translation_error_m, "target translation consistency"),
                                 (angle, max_rotation_error_deg, "target rotation consistency"),
                                 (reprojection, max_reprojection_px, "reprojection RMS")):
        if float(values.max()) > limit:
            reasons.append(f"{label}: {values.max():.6g} exceeds {limit:.6g}")
    return {
        "schema_version": 1, "calibration_type": "eye_to_hand", "method": method,
        "units": "metres", "camera_frame": "color_optical_cv",
        "sample_count": len(samples), "accepted": not reasons, "rejection_reasons": reasons,
        "camera_to_base_cv": camera_to_base.tolist(),
        "base_to_camera_cv": inverse(camera_to_base).tolist(),
        "target_to_gripper": target_to_gripper.tolist(),
        "quality": dict(quality, translation_rms_m=float(np.sqrt(np.mean(distance ** 2))),
                        translation_max_m=float(distance.max()),
                        rotation_rms_deg=float(np.sqrt(np.mean(angle ** 2))),
                        rotation_max_deg=float(angle.max()),
                        reprojection_max_px=float(reprojection.max())),
        "thresholds": dict(max_reprojection_px=max_reprojection_px,
                           max_translation_error_m=max_translation_error_m,
                           max_rotation_error_deg=max_rotation_error_deg),
        "residuals": residuals,
    }

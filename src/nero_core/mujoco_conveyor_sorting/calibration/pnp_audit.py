"""Snapshot the actual PnP inputs and compare independent calibration evidence.

The application undistortion state is separate from unknown SDK/device RGB
processing. Passing these checks establishes input consistency, not accuracy.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def calibration_file_reference(path):
    """Load a file independently of the active K/D and bind it to its bytes."""
    # Local import avoids the capture/session imports at module initialization.
    from .teach_recording import load_intrinsics

    path = Path(path).resolve()
    before = path.read_bytes()
    intrinsics, serial = load_intrinsics(path)
    if path.read_bytes() != before:
        raise ValueError("calibration file changed while reading the PnP reference")
    return {"source": "file", "path": str(path),
            "sha256": hashlib.sha256(before).hexdigest(),
            "K": intrinsics["matrix"], "D": intrinsics["distortion"],
            "image_size": [intrinsics["width"], intrinsics["height"]],
            "serial_number": serial}


def context_from_session(session, *, image_size=None,
                         image_size_source="not_recorded", application_undistorted=None):
    """Preserve unknown historical states; never infer image size from K."""
    capture = session.get("rgb_capture", {})
    geometry = capture.get("application_geometry", {})
    state_source = "caller" if application_undistorted is not None else "not_recorded"
    if application_undistorted is None and isinstance(geometry.get("undistort"), bool):
        application_undistorted = geometry["undistort"]
        state_source = "session.rgb_capture.application_geometry"
    sdk = capture.get("sdk_geometry", {})
    sdk_state = sdk.get("rgb_rectification_state", "unknown")
    return {"image_size": image_size, "image_size_source": image_size_source,
            "application_undistorted": application_undistorted,
            "application_undistortion_state_source": state_source,
            "sdk_rgb_rectification": sdk_state,
            "calibration_reference": session.get("pnp_calibration_reference"),
            "calibration_reference_error": session.get("pnp_calibration_reference_error")}


def _image_size(value, name):
    if value is None:
        return None
    try:
        if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) for v in value):
            raise ValueError
        numeric = [float(v) for v in value]
        if not np.isfinite(numeric).all() or any(v <= 0 or v != int(v) for v in numeric):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be [width, height] with positive integers") from error
    return list(map(int, numeric))


def pnp_input_record(matrix, distortion, intrinsics, image_context=None):
    """Use the exact arrays about to be passed to cv2.solvePnP."""
    context = image_context or {}
    actual_size = _image_size(context.get("image_size"), "actual image_size")
    dimensions = (None if "width" not in intrinsics or "height" not in intrinsics
                  else [intrinsics["width"], intrinsics["height"]])
    intrinsic_size = _image_size(dimensions, "intrinsics image_size")
    state = context.get("application_undistorted")
    if state is not None and not isinstance(state, bool):
        raise ValueError("application_undistorted must be true, false or null (unknown)")
    reference = context.get("calibration_reference")
    # This makes the record independent of caller-owned lists and dictionaries.
    reference = None if reference is None else json.loads(json.dumps(reference, allow_nan=False))
    reference_size = None if reference is None else _image_size(reference.get("image_size"),
                                                               "reference image_size")
    checks = {
        "image_size_matches_intrinsics": (None if actual_size is None or intrinsic_size is None
                                          else actual_size == intrinsic_size),
        "K_matches_reference": (None if reference is None or reference.get("K") is None
                                else bool(np.array_equal(matrix, np.asarray(reference["K"], dtype=float)))),
        "D_matches_reference": (None if reference is None or reference.get("D") is None
                                else bool(np.array_equal(distortion,
                                            np.asarray(reference["D"], dtype=float).reshape(-1)))),
        "reference_image_size_matches_intrinsics":
            (None if reference_size is None or intrinsic_size is None else reference_size == intrinsic_size),
        "undistorted_image_uses_zero_D": (None if state is None else
                                         (not state or not bool(np.any(distortion)))),
    }
    status = ("failed" if any(value is False for value in checks.values()) else
              "incomplete" if any(value is None for value in checks.values()) else "passed")
    return {"schema_version": 1, "solver": "cv2.solvePnP", "flags": "SOLVEPNP_ITERATIVE",
            "call_site": "calibration.eye_to_hand.estimate_target_pose",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "K": matrix.tolist(), "D": distortion.tolist(),
            "image_size": actual_size, "image_size_source": context.get("image_size_source", "not_provided"),
            "intrinsics_image_size": intrinsic_size, "application_undistorted": state,
            "application_undistortion_state_source":
                context.get("application_undistortion_state_source", "caller" if state is not None else "not_recorded"),
            "sdk_rgb_rectification": context.get("sdk_rgb_rectification", "unknown"),
            "calibration_reference": reference,
            "calibration_reference_error": context.get("calibration_reference_error"),
            "checks": checks, "status": status,
            "scope": "application PnP input consistency; not device rectification or physical accuracy"}


def validate_pnp_input_record(record):
    failed = [name for name, value in record["checks"].items() if value is False]
    if failed:
        raise ValueError("PnP input mismatch: " + ", ".join(failed))

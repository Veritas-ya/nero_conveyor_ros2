"""Tie an asynchronous grasp to its exact visual input and current identity.

Only captured depth, visual masks and visual tracker records enter this module.
No MuJoCo body ID, object pose, category truth or contact state is consulted.
Geometric residuals are diagnostics, unless the caller explicitly supplies a
validated motion residual limit. A cloud centroid is not an object center.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


_TIMESTAMP_EPSILON_S = 1e-6
_DEPROJECTION_EPSILON_M = 1e-5  # float32 storage tolerance, not a capture gate.


def select_captured_request(requested):
    """Choose the existing furthest-downstream request with its capture record.

    Each tuple is (position_x, track_id, copied_position, capture_visual_state).
    The original maximum-X selection and first-item tie behavior are preserved.
    Returning the complete tuple prevents an unrelated loop-local visual state
    from being paired with the chosen track's cloud after multiple detections.
    """
    return max(requested, key=lambda request: request[0])


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _vector(value, size):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (size,) or not np.isfinite(result).all():
        return None
    return result


def _numeric_array(value):
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.asarray(np.nan)


def _identity(record):
    if record is None:
        return None
    try:
        value = record.get("object_id", record.get("track_id"))
        identifier = int(value)
        if float(value) != identifier:
            return None
        return identifier
    except (TypeError, ValueError, OverflowError):
        return None


def _generation(record):
    if record is None:
        return None
    value = record.get("track_generation", record.get("generation"))
    return None if value is None else str(value)


def _array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_input_provenance(frame, track_id, cloud, observed, scene_epoch, request_sim_s):
    """Return a JSON-safe audit and input validity, preserving actual timestamps.

    ``observed`` must be the visual state read at ``frame['timestamp']``;
    it must not be today's predicted/published object pose. ``cloud`` is the
    full unmodified world cloud before network subsampling. Raster deprojection
    verifies that the displayed mask and network points represent this frame.
    """
    identifier = int(track_id)
    reasons = []
    capture = _finite_number(frame.get("timestamp"))
    requested = _finite_number(request_sim_s)
    record = {
        "schema_version": 1,
        "source": "captured_depth_and_visual_track_mask",
        "simulator_object_truth_used": False,
        "track_id": identifier,
        "scene_epoch": int(scene_epoch),
        "track_generation": _generation(observed),
        "capture_sim_s": capture,
        "request_sim_s": requested,
        "input_age_at_request_s": None if capture is None or requested is None else requested - capture,
        "track_position_world_m": None,
        "track_velocity_world_m_s": None,
        "track_quaternion_world_wxyz": None,
        "capture_visual_timestamp_s": None,
        "capture_visual_measurement_age_s": None,
        "last_visual_measurement_sim_s": None,
        "capture_visible": None,
        "cloud_coordinate_frame": "world",
        "depth_units": "m",
    }
    if capture is None or requested is None:
        reasons.append("nonfinite_input_timestamp")
    elif requested < capture - _TIMESTAMP_EPSILON_S:
        reasons.append("capture_timestamp_in_future")
    if frame.get("scene_epoch") is not None and int(frame["scene_epoch"]) != int(scene_epoch):
        reasons.append("capture_frame_scene_epoch_mismatch")
    if observed is None:
        reasons.append("capture_visual_track_missing")
    else:
        if _identity(observed) != identifier:
            reasons.append("capture_visual_track_id_mismatch")
        stamp = _finite_number(observed.get("timestamp"))
        age = _finite_number(observed.get("track_observation_age", observed.get("observation_age", 0.0)))
        visible = observed.get("visible")
        record.update({
            "capture_visual_timestamp_s": stamp,
            "capture_visual_measurement_age_s": age,
            "capture_visible": None if visible is None else bool(visible),
            "last_visual_measurement_sim_s": None if stamp is None or age is None else stamp - age,
            "pose_source": observed.get("pose_source"),
        })
        if stamp is None or capture is None or abs(stamp - capture) > _TIMESTAMP_EPSILON_S:
            reasons.append("capture_visual_frame_timestamp_mismatch")
        if age is None or age < 0:
            reasons.append("invalid_capture_visual_age")
        if visible is not None and not bool(visible):
            reasons.append("capture_mask_belongs_to_coasted_track")
        for key, output, size in (
            ("position", "track_position_world_m", 3),
            ("linear_velocity", "track_velocity_world_m_s", 3),
            ("quaternion", "track_quaternion_world_wxyz", 4),
        ):
            value = _vector(observed.get(key), size)
            record[output] = None if value is None else value.tolist()
            if key == "position" and value is None:
                reasons.append("nonfinite_capture_track_position")
    depth = _numeric_array(frame.get("top_depth"))
    labels = np.asarray(frame.get("top_instance_labels"))
    points = _numeric_array(cloud)
    if depth.ndim != 2 or labels.shape != depth.shape:
        reasons.append("depth_mask_raster_mismatch")
    else:
        height, width = depth.shape
        mask = labels == identifier
        rows, columns = np.nonzero(mask)
        record.update({"image_width": int(width), "image_height": int(height),
                       "mask_pixels": len(rows), "mask_sha256": _array_hash(mask),
                       "bbox_xyxy": None if not len(rows) else [int(columns.min()), int(rows.min()),
                                                                 int(columns.max()) + 1, int(rows.max()) + 1]})
        if not len(rows):
            reasons.append("track_absent_from_capture_mask")
        depths = depth[rows, columns]
        valid = np.isfinite(depths) & (depths > 0)
        record["valid_mask_depth_pixels"] = int(valid.sum())
        if not valid.all():
            reasons.append("capture_mask_contains_invalid_depth")
        calibration = frame.get("top_camera", {})
        K = _numeric_array(calibration.get("intrinsics"))
        T = _numeric_array(calibration.get("camera_to_world"))
        camera_valid = (K.shape == (3, 3) and T.shape == (4, 4)
                        and np.isfinite(K).all() and np.isfinite(T).all()
                        and K[0, 0] > 0 and K[1, 1] > 0
                        and np.allclose(K[2], [0, 0, 1])
                        and np.allclose(T[3], [0, 0, 0, 1])
                        and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
                        and np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-5))
        if not camera_valid:
            reasons.append("invalid_capture_camera_calibration")
        else:
            record["K"] = K.tolist()
            record["camera_to_world"] = T.tolist()
            record["camera_convention"] = "mujoco_x_right_y_up_minus_z_forward"
            record["preprocessing"] = calibration.get("preprocessing", "not_recorded")
            distortion = _numeric_array(calibration.get("distortion"))
            record["distortion"] = (distortion.tolist() if distortion.ndim == 1
                                      and np.isfinite(distortion).all() else None)
            if (int(calibration.get("width", width)) != width
                    or int(calibration.get("height", height)) != height):
                reasons.append("camera_resolution_differs_from_depth_raster")
            if (points.ndim == 2 and points.shape[1] == 3
                    and points.shape[0] == len(rows) and np.isfinite(points).all()
                    and valid.all() and len(rows)):
                optical = np.column_stack(((columns - K[0, 2]) * depths / K[0, 0],
                                            (rows - K[1, 2]) * depths / K[1, 1], depths))
                gl = optical * [1., -1., -1.]
                expected = gl @ T[:3, :3].T + T[:3, 3]
                error = float(np.linalg.norm(points - expected, axis=1).max())
                record["depth_mask_projection_max_error_m"] = error
                if error > _DEPROJECTION_EPSILON_M:
                    reasons.append("cloud_does_not_match_capture_depth_mask")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        reasons.append("nonfinite_or_invalid_cloud")
    else:
        record["cloud_point_count"] = len(points)
        record["cloud_storage_dtype"] = str(np.asarray(cloud).dtype)
        record["cloud_sha256"] = _array_hash(np.asarray(cloud))
        if not len(points):
            reasons.append("empty_cloud")
        else:
            record["cloud_min_world_m"] = points.min(axis=0).tolist()
            record["cloud_max_world_m"] = points.max(axis=0).tolist()
            centroid = points.mean(axis=0, dtype=float)
            record["cloud_centroid_world_m"] = centroid.tolist()
            if record["track_position_world_m"] is not None:
                record["centroid_to_track_center_m"] = float(np.linalg.norm(
                    centroid - record["track_position_world_m"]))
        if record.get("mask_pixels") is not None and len(points) != record["mask_pixels"]:
            reasons.append("cloud_point_count_differs_from_capture_mask")
    record["input_valid"] = not reasons
    record["validation_reasons"] = reasons
    return record


def _quaternion_angle_degrees(first, second):
    first, second = _vector(first, 4), _vector(second, 4)
    if first is None or second is None:
        return None
    norm_product = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm_product < 1e-12:
        return None
    dot = float(np.clip(abs(np.dot(first, second)) / norm_product, 0., 1.))
    return math.degrees(2 * math.acos(dot))


def validate_candidate_admission(provenance, live_track, scene_epoch, now_s, *,
                                 max_observation_age_s, max_result_age_s=None,
                                 max_prediction_residual_m=None, latest_allowed_x_m=None):
    """Validate input binding; log prediction/rotation changes without new rules.

    Briefly occluded/coasted live tracks remain admissible within their existing
    visual age limit. ``observed_timestamp`` should be merged from the visual
    prediction record by the caller; publication ``timestamp`` never substitutes
    silently for capture time. If only tracker age is available, its lower-bound
    meaning is recorded explicitly. Output contains only ordinary JSON values.
    """
    limits = {"max_observation_age_s": max_observation_age_s,
              "max_result_age_s": max_result_age_s,
              "max_prediction_residual_m": max_prediction_residual_m}
    for name, limit in limits.items():
        if limit is not None and (_finite_number(limit) is None or float(limit) < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_observation_age_s is None:
        raise ValueError("max_observation_age_s is required")
    if latest_allowed_x_m is not None and _finite_number(latest_allowed_x_m) is None:
        raise ValueError("latest_allowed_x_m must be finite")
    reasons = list(provenance.get("validation_reasons", []))
    if not provenance.get("input_valid", False) and not reasons:
        reasons.append("invalid_input_provenance")
    now = _finite_number(now_s)
    capture = _finite_number(provenance.get("capture_sim_s"))
    age = None if now is None or capture is None else now - capture
    record = {"allowed": False, "reasons": reasons, "ready_sim_s": now,
              "result_age_s": age, "track_id": provenance.get("track_id"),
              "scene_epoch": int(scene_epoch), "live_visible": None,
              "live_observed_timestamp_s": None, "live_publication_timestamp_s": None,
              "live_measurement_age_s": None, "live_age_basis": None,
              "capture_to_live_translation_m": None, "prediction_residual_m": None,
              "capture_to_live_rotation_deg": None}
    if int(scene_epoch) != provenance.get("scene_epoch"):
        reasons.append("scene_epoch_changed")
    if now is None or capture is None or age < -_TIMESTAMP_EPSILON_S:
        reasons.append("invalid_result_timestamp")
    elif max_result_age_s is not None and age > float(max_result_age_s):
        reasons.append("inference_result_expired")
    if live_track is None:
        reasons.append("visual_track_missing_at_admission")
    else:
        if _identity(live_track) != provenance.get("track_id"):
            reasons.append("live_visual_track_id_mismatch")
        generation = _generation(live_track)
        record["live_track_generation"] = generation
        if provenance.get("track_generation") is not None:
            if generation is None:
                reasons.append("live_visual_track_generation_unavailable")
            elif provenance["track_generation"] != generation:
                reasons.append("visual_track_generation_changed")
        record["live_visible"] = None if "visible" not in live_track else bool(live_track["visible"])
        stamp = _finite_number(live_track.get("observed_timestamp", live_track.get("capture_timestamp")))
        publication = _finite_number(live_track.get("timestamp"))
        reported_age = _finite_number(live_track.get("track_observation_age"))
        measured_stamp = _finite_number(live_track.get("measurement_timestamp"))
        record["live_observed_timestamp_s"] = stamp
        record["live_publication_timestamp_s"] = publication
        if measured_stamp is not None and now is not None:
            measurement_age = max(0., now - measured_stamp)
            record["live_age_basis"] = "last_actual_measurement_timestamp"
        elif stamp is not None and reported_age is not None and now is not None:
            measurement_age = max(0., now - stamp) + reported_age
            record["live_age_basis"] = "capture_timestamp_plus_tracker_measurement_age"
        elif reported_age is not None:
            measurement_age = reported_age
            record["live_age_basis"] = "reported_track_age_lower_bound_capture_timestamp_missing"
        else:
            measurement_age = None
            record["live_age_basis"] = "unknown"
        record["live_measurement_age_s"] = measurement_age
        if stamp is not None and now is not None and stamp > now + _TIMESTAMP_EPSILON_S:
            reasons.append("live_capture_timestamp_in_future")
        if measurement_age is None or reported_age is not None and reported_age < 0:
            reasons.append("live_visual_measurement_age_unavailable_or_invalid")
        elif measurement_age > float(max_observation_age_s):
            reasons.append("live_visual_measurement_expired")
        position = _vector(live_track.get("position"), 3)
        capture_position = _vector(provenance.get("track_position_world_m"), 3)
        velocity = _vector(provenance.get("track_velocity_world_m_s"), 3)
        if position is None:
            reasons.append("nonfinite_live_track_position")
        else:
            record["live_position_world_m"] = position.tolist()
            if latest_allowed_x_m is not None and position[0] > float(latest_allowed_x_m):
                reasons.append("visual_track_passed_grasp_workspace")
            if capture_position is not None:
                record["capture_to_live_translation_m"] = float(np.linalg.norm(position - capture_position))
                if velocity is not None and age is not None:
                    predicted = capture_position + velocity * age
                    residual = float(np.linalg.norm(position - predicted))
                    record["predicted_live_position_world_m"] = predicted.tolist()
                    record["prediction_residual_m"] = residual
                    if max_prediction_residual_m is not None and residual > float(max_prediction_residual_m):
                        reasons.append("visual_motion_prediction_residual_exceeded")
        record["capture_to_live_rotation_deg"] = _quaternion_angle_degrees(
            provenance.get("track_quaternion_world_wxyz"), live_track.get("quaternion"))
    record["allowed"] = not reasons
    return record

"""Receive-only NERO diagnostics and frame-invariant rotation comparisons.

Matching SDK FK is an internal consistency check, not a physical accuracy test.
Camera/robot angle agreement also cannot identify which sensor is inaccurate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import time

import numpy as np

from .geometry import pose_xyz_rpy, rigid_transform, transform_distance
from .teach_recording import TimedPose


COMPONENT_NAMES = ("joint_12", "joint_34", "joint_56", "joint_7",
                   "end_pose_xy", "end_pose_zrx", "end_pose_ryrz")


@dataclass
class DiagnosticPose(TimedPose):
    joints_rad: tuple
    flange_feedback_m_rad: tuple
    sdk_fk_m_rad: tuple
    fk_translation_error_m: float
    fk_rotation_error_deg: float

    def document(self):
        return dict(super().document(), component_names=list(COMPONENT_NAMES),
                    joints_rad=list(self.joints_rad),
                    flange_feedback_m_rad=list(self.flange_feedback_m_rad),
                    sdk_fk_m_rad=list(self.sdk_fk_m_rad),
                    fk_translation_error_m=self.fk_translation_error_m,
                    fk_rotation_error_deg=self.fk_rotation_error_deg)


def _finite_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(item) for item in result)


def read_diagnostic_pose(reader, sequence, offset, previous=None):
    """Atomically copy all seven CAN components; never request/transmit data.

    CAN component receipt times differ; the snapshot is not hardware-synchronous.
    The representative timestamp uses the newest of the three flange packets.
    Interpret FK errors only while joints AND flange remain stationary.
    """
    with reader._snapshot_lock:
        now_wall = time.time()
        parser = reader.robot._parser
        stamps = []
        for name in COMPONENT_NAMES:
            part = getattr(parser, name, None)
            stamp = float(part.timestamp) if part is not None else float("nan")
            if not np.isfinite(stamp) or not 0 <= now_wall - stamp <= reader.max_age_s:
                raise ValueError(f"missing/stale NERO feedback component: {name}")
            stamps.append(stamp)
        stamps = tuple(stamps)
        if max(stamps) - min(stamps) > .03:
            raise ValueError("NERO joint/flange component timestamps differ by > 30 ms")
        joint_message = reader.robot.get_joint_angles()
        flange_message = reader.robot.get_flange_pose()
        if joint_message is None or flange_message is None:
            raise ValueError("waiting for complete NERO joint/flange feedback")
        joints = _finite_vector(list(joint_message.msg), 7, "joint angles")
        flange = _finite_vector(list(flange_message.msg), 6, "flange pose")
    now_mono, now_wall = time.monotonic(), time.time()
    if not np.isfinite(offset) or abs((now_mono - now_wall) - offset) > .05:
        raise RuntimeError("系统时钟发生跳变，请结束本次记录后重新启动")
    if previous is not None:
        if len(previous) != len(COMPONENT_NAMES):
            raise ValueError("previous timestamps must contain all seven CAN components")
        if any(current <= old for current, old in zip(stamps, previous)):
            return None
    # Work on independent copies outside the parser lock; FK uses no CAN traffic.
    fk = _finite_vector(reader.robot.fk(list(joints)), 6, "SDK FK pose")
    transform, fk_transform = pose_xyz_rpy(flange), pose_xyz_rpy(fk)
    distance, angle = transform_distance(transform, fk_transform)
    return DiagnosticPose(sequence, max(stamps[-3:]) + offset, now_mono, stamps,
                          transform, joints, flange, fk, distance, angle)


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def summarize_window(history, timing):
    """Summarize diagnostic snapshots in the already bracketed stationary window."""
    start, end = float(timing["window_start_monotonic"]), float(timing["window_end_monotonic"])
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError("invalid diagnostic window")
    with history.lock:
        samples = [sample for sample in history.samples if start <= sample.time_monotonic <= end]
    summary = {"count": len(samples), "fk_translation_rms_m": None,
               "fk_translation_max_m": None, "fk_rotation_rms_deg": None,
               "fk_rotation_max_deg": None, "flange_change_max_m": None,
               "flange_change_max_deg": None, "joint_change_max_deg": None}
    if not samples:
        return summary
    distances = [float(sample.fk_translation_error_m) for sample in samples]
    angles = [float(sample.fk_rotation_error_deg) for sample in samples]
    if not np.isfinite(distances + angles).all() or min(distances + angles) < 0:
        raise ValueError("invalid FK diagnostics in pose window")
    movement = [transform_distance(samples[0].transform, sample.transform) for sample in samples]
    joints = np.asarray([sample.joints_rad for sample in samples], dtype=float)
    if joints.shape != (len(samples), 7) or not np.isfinite(joints).all():
        raise ValueError("invalid joint diagnostics in pose window")
    delta = (joints - joints[0] + np.pi) % (2 * np.pi) - np.pi
    summary.update(fk_translation_rms_m=_rms(distances), fk_translation_max_m=max(distances),
                   fk_rotation_rms_deg=_rms(angles), fk_rotation_max_deg=max(angles),
                   flange_change_max_m=max(row[0] for row in movement),
                   flange_change_max_deg=max(row[1] for row in movement),
                   joint_change_max_deg=float(np.degrees(np.max(np.abs(delta)))))
    return summary


def build_validation_report(session, angle_warning_deg=1., measured_angle=None,
                            measurement_note=None, measurement_uncertainty_deg=None):
    """Compare intrinsic rotation angles, invariant under fixed frame changes.

    For rigid mounting C_i = X^-1 G_i Y, angle(G_i^-1 G_j) equals
    angle(C_i^-1 C_j). Translation lengths generally do NOT share this invariant.
    An independent reference must be a measured full 3D principal angle, or a
    physically constrained single-axis rotation. Euler/tilt differences alone
    are not valid reference angles for general spatial motion.
    """
    angle_warning_deg = float(angle_warning_deg)
    if not np.isfinite(angle_warning_deg) or angle_warning_deg <= 0:
        raise ValueError("angle warning threshold must be finite and positive")
    samples = []
    by_id = {}
    sample_diagnostics = []
    selected_fk_distances, selected_fk_angles = [], []
    for raw in session.get("samples", []):
        sample_id = str(raw.get("sample_id", "")).strip()
        if not sample_id or sample_id in by_id:
            raise ValueError("sample IDs must be nonempty and unique")
        robot = rigid_transform(raw["gripper_to_base"], f"robot pose {sample_id}")
        camera = raw.get("target_to_camera")
        camera = None if camera is None else rigid_transform(camera, f"camera pose {sample_id}")
        sample = (sample_id, robot, camera)
        samples.append(sample)
        by_id[sample_id] = sample
        diagnostics = {"sample_id": sample_id}
        for key in ("robot_diagnostics", "stationary_window"):
            item = raw.get(key)
            if item is not None:
                if not isinstance(item, dict):
                    raise ValueError(f"{key} for {sample_id} must be an object")
                try:
                    item = json.loads(json.dumps(item, allow_nan=False))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} for {sample_id} must be finite JSON data") from exc
            diagnostics[key] = item
        robot_diagnostics = diagnostics["robot_diagnostics"]
        if robot_diagnostics is not None:
            if "joints_rad" in robot_diagnostics:
                _finite_vector(robot_diagnostics["joints_rad"], 7, "diagnostic joint angles")
            for key, values in (("fk_translation_error_m", selected_fk_distances),
                                ("fk_rotation_error_deg", selected_fk_angles)):
                if key in robot_diagnostics:
                    value = float(robot_diagnostics[key])
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {key} for {sample_id}")
                    values.append(value)
        sample_diagnostics.append(diagnostics)
    pairs = []
    for first, second in combinations(samples, 2):
        robot_angle = transform_distance(first[1], second[1])[1]
        camera_angle = None
        if first[2] is not None and second[2] is not None:
            camera_angle = transform_distance(first[2], second[2])[1]
        pairs.append({"from_id": first[0], "to_id": second[0],
                      "robot_rotation_deg": robot_angle, "camera_rotation_deg": camera_angle,
                      "angle_difference_deg": None if camera_angle is None else robot_angle - camera_angle})
    differences = [pair["angle_difference_deg"] for pair in pairs if pair["angle_difference_deg"] is not None]
    max_difference = max(map(abs, differences)) if differences else None
    summary = {"sample_count": len(samples), "pair_count": len(pairs),
               "camera_pair_count": len(differences),
               "angle_difference_rms_deg": _rms(differences),
               "angle_difference_max_abs_deg": max_difference,
               "angle_warning_deg": angle_warning_deg,
               "pair_count_above_warning": sum(abs(value) > angle_warning_deg for value in differences),
               "selected_fk_translation_rms_m": _rms(selected_fk_distances),
               "selected_fk_translation_max_m": max(selected_fk_distances) if selected_fk_distances else None,
               "selected_fk_rotation_rms_deg": _rms(selected_fk_angles),
               "selected_fk_rotation_max_deg": max(selected_fk_angles) if selected_fk_angles else None}
    reference = None
    if measured_angle is not None:
        if len(measured_angle) != 3:
            raise ValueError("measured angle requires from ID, to ID and degrees")
        first_id, second_id, angle = measured_angle
        first_id, second_id, angle = str(first_id), str(second_id), float(angle)
        if first_id == second_id or first_id not in by_id or second_id not in by_id:
            raise ValueError("reference measurement requires two distinct existing sample IDs")
        if not np.isfinite(angle) or not 0 <= angle <= 180:
            raise ValueError("reference angle must be finite and within [0, 180] degrees")
        if not isinstance(measurement_note, str) or not measurement_note.strip():
            raise ValueError("describe the independent instrument and constrained motion/reference")
        uncertainty = measurement_uncertainty_deg
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not np.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("measurement uncertainty must be finite and nonnegative")
        first, second = by_id[first_id], by_id[second_id]
        robot_angle = transform_distance(first[1], second[1])[1]
        camera_angle = (transform_distance(first[2], second[2])[1]
                        if first[2] is not None and second[2] is not None else None)
        reference = {"from_id": first_id, "to_id": second_id, "measured_angle_deg": angle,
                     "measurement_note": measurement_note.strip(),
                     "measurement_uncertainty_deg": uncertainty,
                     "robot_rotation_deg": robot_angle, "camera_rotation_deg": camera_angle,
                     "robot_error_deg": robot_angle - angle,
                     "camera_error_deg": None if camera_angle is None else camera_angle - angle}
    elif measurement_note is not None or measurement_uncertainty_deg is not None:
        raise ValueError("measurement metadata requires a measured angle")
    status = "insufficient_samples" if len(samples) < 2 else "diagnostic_only"
    if summary["pair_count_above_warning"]:
        status = "needs_review"
    if reference is not None and any(value is not None and abs(value) > angle_warning_deg
                                     for value in (reference["robot_error_deg"], reference["camera_error_deg"])):
        status = "needs_review"
    return {"schema_version": 1, "diagnostic_type": "nero_pose_validation", "status": status,
            "summary": summary, "pairs": pairs, "reference_measurement": reference,
            "sample_diagnostics": sample_diagnostics,
            "notes": [
                "Angle differences are signed robot minus camera angles; maximum is absolute.",
                "Pairs share samples and are correlated, not independent accuracy measurements.",
                "SDK FK versus controller feedback is internal consistency, not physical accuracy.",
                "Camera/robot differences do not identify the faulty sensor or establish absolute accuracy.",
                "Fixed camera, rigid target and consistent physical corner numbering are required.",
                "A reference is the 3D principal rotation angle or constrained single-axis angle; not an Euler/tilt difference.",
                "This test does not validate translation accuracy; thresholds are diagnostic, not manufacturer specifications."]}

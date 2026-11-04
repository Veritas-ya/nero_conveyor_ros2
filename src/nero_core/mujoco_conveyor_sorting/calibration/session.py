"""Portable calibration sessions, read-only NERO feedback and runtime export."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock
import time

import cv2
import numpy as np
import yaml

from .eye_to_hand import CalibrationSample, Chessboard, detect_chessboard, estimate_target_pose
from .geometry import pose_xyz_rpy, rigid_transform, transform_distance
from .pnp_audit import calibration_file_reference, context_from_session


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, document):
    """Atomic session checkpoint: keep previously collected poses on failure."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def intrinsics_document(intrinsics):
    result = asdict(intrinsics)
    result["matrix"] = intrinsics.matrix.tolist()
    return result


def sdk_calibration_reference(intrinsics):
    """Independent snapshot of the frame profile's SDK color calibration."""
    return {"source": "sdk_factory", "K": np.asarray(intrinsics["matrix"]).tolist(),
            "D": list(intrinsics.get("distortion", [])),
            "image_size": [intrinsics["width"], intrinsics["height"]]}


def rgb_capture_document(camera, frame):
    """Preserve SDK calibration alongside the K/D used by the solver.

    Pixels are the SDK-delivered color frame, not guaranteed sensor-raw pixels.
    The capture scripts only convert channel order for PNG encoding; native
    device/frame-processor correction must be recorded separately as unknown
    when it cannot be read, rather than inferred from our application pipeline.
    """
    image = np.asarray(frame.rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("capture requires a uint8 RGB image")
    intrinsic = frame.color_intrinsics
    if intrinsic is None or image.shape[:2] != (intrinsic.height, intrinsic.width):
        raise ValueError("actual RGB image size differs from SDK color intrinsics")
    getter = getattr(camera, "get_color_geometry_metadata", None)
    sdk_geometry = (getter() if callable(getter) else
                    {"status": "unavailable", "rgb_rectification_state": "unknown"})
    # Make an independent JSON-compatible snapshot rather than retaining a
    # mutable backend dictionary that can change during subsequent frames.
    sdk_geometry = json.loads(json.dumps(sdk_geometry, allow_nan=False))
    return {
        "color_intrinsics_sdk": intrinsics_document(intrinsic),
        "rgb_capture": {
            "actual_image_size": [image.shape[1], image.shape[0]],
            "pixels": "sdk_delivered_rgb",
            "saved_image_format": "PNG",
            "application_geometry": {"crop": None, "resize": None, "undistort": False},
            "sdk_geometry": sdk_geometry,
        },
    }


def load_samples(path, *, exclude=(), pnp_audits=None):
    path = Path(path)
    session = read_json(path)
    if (session.get("schema_version") != 1 or session.get("calibration_type") != "eye_to_hand"
            or session.get("units") != "metres" or session.get("camera_frame") != "color_optical_cv"):
        raise ValueError("expected schema_version=1, eye_to_hand, units=metres, camera_frame=color_optical_cv")
    board = Chessboard(**session["board"])
    # Re-open the current independent file once for this solve. A captured file
    # snapshot proves what was loaded then, not what is in that file now.
    # Historical sessions may only have stored the source filename.
    if pnp_audits is not None:
        captured_reference = session.get("pnp_calibration_reference")
        source = (captured_reference.get("path")
                  if captured_reference and captured_reference.get("source") == "file"
                  else None)
        source = source or session.get("intrinsics_source")
        if source and source != "sdk_factory":
            if captured_reference is not None:
                session["captured_pnp_calibration_reference"] = captured_reference
            session.pop("pnp_calibration_reference", None)
            reference_path = Path(source)
            if not reference_path.is_absolute():
                reference_path = path.parent / reference_path
            try:
                session["pnp_calibration_reference"] = calibration_file_reference(reference_path)
            except (OSError, ValueError, KeyError, cv2.error) as error:
                session["pnp_calibration_reference_error"] = str(error)
    samples = []
    ids = [str(item["sample_id"]) for item in session["samples"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs in session")
    if set(exclude) - set(ids):
        raise ValueError(f"excluded sample IDs do not exist: {sorted(set(exclude) - set(ids))}")
    for item in session["samples"]:
        sample_id = str(item["sample_id"])
        if sample_id in exclude:
            continue
        # Read the saved pixels even when cached corners are available: the
        # intrinsics width/height are not evidence of the image's actual size.
        bgr = None
        image_path = path.parent / item["image"] if item.get("image") else None
        if image_path is not None and image_path.exists():
            bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ValueError(f"cannot read sample image: {item['image']}")
            image_size = [bgr.shape[1], bgr.shape[0]]
            image_size_source = "saved_image_pixels"
        elif item.get("image_size") is not None:
            image_size = item["image_size"]
            image_size_source = "sample.image_size"
        else:
            image_size = session.get("rgb_capture", {}).get("actual_image_size")
            image_size_source = ("session.rgb_capture.actual_image_size"
                                 if image_size is not None else "unknown")
        context = context_from_session(session, image_size=image_size,
                                       image_size_source=image_size_source)
        audit = {} if pnp_audits is not None else None
        try:
            if "corners" in item:
                target, error = estimate_target_pose(item["corners"], board, session["color_intrinsics"],
                                                      pnp_audit=audit, image_context=context)
            elif "image" in item:
                if bgr is None:
                    raise ValueError(f"cannot read sample image: {item['image']}")
                target, error, _ = detect_chessboard(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), board,
                                                    session["color_intrinsics"],
                                                    reverse=item.get("reverse_corners", False),
                                                    pnp_audit=audit, image_context=context)
            else:
                target, error = item["target_to_camera"], item["reprojection_error_px"]
        finally:
            # A failed validation still has a useful core input snapshot. No
            # record is fabricated if detection/PnP was never reached.
            if audit:
                audit.update(sample_id=sample_id, image=item.get("image"))
                pnp_audits.append(audit)
        samples.append(CalibrationSample(item["gripper_to_base"], target, error, sample_id))
    return session, samples


def runtime_document(result, session, original, *, base_to_world, calibration_id):
    """Build a new complete runtime.yaml without changing the input settings."""
    if result.get("accepted") is not True:
        raise ValueError("failed quality checks; cannot export runtime calibration")
    if not str(calibration_id).strip():
        raise ValueError("calibration_id must not be empty")
    if not session.get("serial_number"):
        raise ValueError("runtime export requires the camera serial_number")
    world = rigid_transform(base_to_world, "base_to_world")
    color_to_base = rigid_transform(result["camera_to_base_cv"], "camera_to_base_cv")
    depth_to_color = rigid_transform(session["depth_to_color_cv"], "depth_to_color_cv")
    # Deep-copy YAML-compatible settings and bind the export to captured profiles.
    document = yaml.safe_load(yaml.safe_dump(original))
    camera = document["runtime"]["camera"]
    serial = camera.get("serial_number")
    if serial and serial != session["serial_number"]:
        raise ValueError("runtime serial_number does not match the captured camera")
    profiles = session["stream_config"]
    for prefix in ("depth", "color"):
        for suffix in ("width", "height", "fps", "format"):
            key = f"{prefix}_{suffix}"
            if profiles.get(key) is None:
                raise ValueError(f"runtime export requires captured profile field {key}")
            camera[key] = profiles[key]
        intrinsics = session[f"{prefix}_intrinsics"]
        if (profiles[f"{prefix}_width"] != intrinsics["width"]
                or profiles[f"{prefix}_height"] != intrinsics["height"]):
            raise ValueError(f"captured {prefix} profile and intrinsics disagree")
    captured_pid = session.get("product_id")
    if captured_pid is not None:
        if camera.get("expected_product_id") not in (None, captured_pid):
            raise ValueError("runtime product ID does not match the captured camera")
        camera["expected_product_id"] = captured_pid
    camera.update(serial_number=session["serial_number"], calibration_id=calibration_id,
                  camera_to_world_cv=(world @ color_to_base @ depth_to_color).tolist())
    # Exercise exactly the same config validation used when starting hardware.
    from ..runtime.backends.orbbec import OrbbecCameraConfig
    OrbbecCameraConfig.from_mapping(camera)
    return document


class NeroPoseReader:
    """Receive-only adapter for the vendored pyAgxArm feedback API.

    The SDK can return partially updated pose packets. Check all three CAN
    components before using get_flange_pose(), and copy its mutable payload.
    No enabling, mode changes, firmware requests or motion commands are sent.
    """

    def __init__(self, robot, *, max_age_s=0.2):
        self.robot = robot
        self.max_age_s = max_age_s
        self._snapshot_lock = RLock()
        self._started = False

    def _guard_parser(self):
        """Serialize the SDK's in-place decoding with pose/timestamp copies.

        ArmDriverContext's registered receive callback looks up
        ``self._parser.parse_packet`` on every frame. Install before connect,
        rather than replacing the already-registered context callback. Keep
        one guard per parser so reconnects/readers do not stack wrappers.
        """
        parser = self.robot._parser
        parse_packet = getattr(parser, "parse_packet", None)
        if not callable(parse_packet):
            return  # Static feedback adapters used by offline tests have no decoder.
        lock = getattr(parser, "_calibration_snapshot_lock", None)
        if lock is None:
            if getattr(self.robot, "_connected", False):
                raise RuntimeError("start the receive-only pose reader before connecting the robot")
            lock = RLock()

            def guarded_parse(*args, **kwargs):
                with lock:
                    return parse_packet(*args, **kwargs)

            parser.parse_packet = guarded_parse
            parser._calibration_snapshot_lock = lock
        self._snapshot_lock = lock

    def start(self):
        if self._started:
            return
        self._guard_parser()
        comm = self.robot.create_comm()

        def reject_send(*args, **kwargs):
            raise RuntimeError("calibration is receive-only; CAN transmit is forbidden")

        comm.send = reject_send
        self.robot.connect()
        self._started = True

    def read(self):
        return self.read_snapshot()[0]

    def read_snapshot(self):
        """Copy one pose and its three CAN receipt timestamps under the RX lock.

        A packet decoder updates fields *before* its timestamp. Comparing
        timestamps before/after an unlocked read cannot detect every partial
        update. The shared lock excludes that decoder throughout this copy;
        the three different CAN packets can still have different receipt times.
        """
        with self._snapshot_lock:
            now = time.time()  # SocketCAN receipt timestamps use the Unix clock.
            parser = self.robot._parser
            timestamps = []
            for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"):
                part = getattr(parser, name, None)
                if part is None or not 0 <= now - float(part.timestamp) <= self.max_age_s:
                    raise ValueError(f"missing/stale NERO pose component: {name}")
                timestamps.append(float(part.timestamp))
            sample = self.robot.get_flange_pose()
            if sample is None or not 0 <= time.time() - float(sample.timestamp) <= self.max_age_s:
                raise ValueError("no fresh NERO flange pose feedback")
            return pose_xyz_rpy(list(sample.msg)), tuple(timestamps)

    def close(self):
        self.robot.disconnect()
        self._started = False


def check_stationary(poses, *, max_translation_m=0.001, max_rotation_deg=0.3):
    if len(poses) < 2:
        raise ValueError("stationarity check requires multiple feedback readings")
    reference = poses[0]
    for pose in poses[1:]:
        distance, angle = transform_distance(reference, pose)
        if distance > max_translation_m or angle > max_rotation_deg:
            raise ValueError("robot moved during capture; hold the board still and retry")

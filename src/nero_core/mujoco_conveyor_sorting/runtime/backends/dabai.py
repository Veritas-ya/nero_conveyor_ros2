"""Native Dabai DC1 RGB-D through an isolated Orbbec SDK v1 C library.

Only exact factory RGB/depth mode pairs permit metric analysis. SDK v1's
legacy matched-profile accessor sometimes returns empty calibration on DC1;
the device's factory list contains the usable 640x400 / 1280x720 pair.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
import time
import warnings

import numpy as np

from ..types import BackendError, BackendUnavailable, CameraIntrinsics, DeviceHealth, RGBDFrame, SampleHeader
from .orbbec_v1 import LegacyOrbbecSdk, OBCameraParam
from .camera_lease import CameraLease


FORMATS = {"YUYV": 0, "MJPG": 5, "MJPEG": 5, "Y16": 8, "Y11": 11,
           "Y12": 12, "RGB": 22, "BGR": 23, "BGRA": 25, "RGBA": 31}
FORMAT_NAMES = {value: key for key, value in FORMATS.items() if key != "MJPEG"}
PROPERTY_IDS = {"OB_PROP_DEPTH_MIRROR_BOOL": 14, "OB_PROP_DEPTH_FLIP_BOOL": 15,
                "OB_PROP_COLOR_MIRROR_BOOL": 81, "OB_PROP_COLOR_FLIP_BOOL": 82,
                "OB_PROP_COLOR_ROTATE_INT": 115, "OB_PROP_DEPTH_ROTATE_INT": 118,
                "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL": 2000, "OB_PROP_COLOR_EXPOSURE_INT": 2001,
                "OB_PROP_COLOR_GAIN_INT": 2002, "OB_PROP_COLOR_SHARPNESS_INT": 2006}


@dataclass(frozen=True)
class DabaiCameraConfig:
    serial_number: str | None = None
    rgb_device: str | None = None
    legacy_sdk_dir: Path | str | None = None
    depth_width: int = 640
    depth_height: int = 400
    depth_fps: int = 30
    depth_format: str = "Y11"
    color_width: int = 1280
    color_height: int = 720
    color_fps: int = 30
    color_format: str = "MJPG"
    frame_timeout_ms: int = 500
    startup_timeout_s: float = 10.0
    max_frame_age_s: float = .5
    minimum_depth_m: float = .02
    maximum_depth_m: float = 5.0

    def __post_init__(self):
        for prefix in ("depth", "color"):
            for suffix in ("width", "height", "fps"):
                value = getattr(self, f"{prefix}_{suffix}")
                if int(value) != value or value <= 0:
                    raise ValueError("Dabai stream dimensions and fps must be positive integers")
            fmt = str(getattr(self, f"{prefix}_format")).upper()
            if fmt not in FORMATS:
                raise ValueError(f"unsupported Dabai {prefix} format {fmt}")
            object.__setattr__(self, f"{prefix}_format", fmt)
        if not 0 <= self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("invalid metric depth range")
        if self.frame_timeout_ms <= 0 or self.startup_timeout_s <= 0 or self.max_frame_age_s <= 0:
            raise ValueError("Dabai frame timeouts must be positive")


def camera_param_geometry(param: OBCameraParam, depth_size: tuple[int, int],
                          color_size: tuple[int, int]):
    """Validate an exact native factory record without resizing its geometry."""
    if param.isMirrored:
        raise BackendUnavailable("mirrored Dabai factory calibration cannot describe native optical frames")
    intrinsics = []
    for native, distortion, size in ((param.depthIntrinsic, param.depthDistortion, depth_size),
                                      (param.rgbIntrinsic, param.rgbDistortion, color_size)):
        if (native.width, native.height) != tuple(size):
            raise BackendUnavailable(f"factory calibration dimensions {(native.width, native.height)} do not match active profile {size}")
        coefficients = tuple(float(getattr(distortion, key))
                             for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
        try:
            k = CameraIntrinsics(native.width, native.height, float(native.fx), float(native.fy),
                                 float(native.cx), float(native.cy), coefficients, "brown_conrady")
        except ValueError as error:
            raise BackendUnavailable(f"invalid Dabai factory intrinsics: {error}") from error
        if not (-.5 <= k.cx <= k.width - .5 and -.5 <= k.cy <= k.height - .5):
            raise BackendUnavailable("Dabai factory principal point lies outside its native frame")
        intrinsics.append(k)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(param.transform.rot, float).reshape(3, 3)
    transform[:3, 3] = np.asarray(param.transform.trans, float) / 1000.0
    r = transform[:3, :3]
    if (not np.isfinite(transform).all() or not np.allclose(r.T @ r, np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=2e-5, rtol=0)):
        raise BackendUnavailable("invalid Dabai depth-to-color rigid transform")
    if not 0.001 <= np.linalg.norm(transform[:3, 3]) <= .3:
        raise BackendUnavailable("Dabai depth-to-color baseline is missing or implausible")
    return intrinsics[0], intrinsics[1], transform


def decode_depth(payload: bytes, width: int, height: int, format_id: int, scale_mm: float,
                 *, min_m: float = .02, max_m: float = 5.0) -> np.ndarray:
    """Decode the SDK's unpacked Y16/Y11 metric depth, preserving native pixels."""
    if format_id not in (8, 11, 12) or len(payload) != width * height * 2:
        raise BackendUnavailable("Dabai depth payload must contain unpacked 16-bit native depth")
    if not np.isfinite(scale_mm) or scale_mm <= 0:
        raise BackendUnavailable("invalid Dabai depth value scale")
    depth = np.frombuffer(payload, dtype="<u2").reshape(height, width).astype(np.float32)
    depth *= float(scale_mm) / 1000.0
    depth[(depth < min_m) | (depth > max_m)] = 0
    return depth


def decode_rgb(payload: bytes, width: int, height: int, format_id: int) -> np.ndarray:
    import cv2
    if format_id == 5:
        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (height, width, 3):
            raise BackendUnavailable("Dabai MJPEG dimensions do not match the active native RGB profile")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    channels = 4 if format_id in (25, 31) else 3
    if format_id in (22, 23, 25, 31) and len(payload) == width * height * channels:
        image = np.frombuffer(payload, np.uint8).reshape(height, width, channels)
        if format_id in (23, 25):
            image = image[..., [2, 1, 0]]
        return image[..., :3].copy()
    if format_id == 0 and len(payload) == width * height * 2:
        image = np.frombuffer(payload, np.uint8).reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)
    raise BackendUnavailable(f"unsupported Dabai RGB frame format/payload: {format_id}")


class DabaiCameraBackend:
    """Latest native RGB-D frames from exactly one DC1, without v2 imports."""

    def __init__(self, config: DabaiCameraConfig | None = None, *, sdk=None, clock=time.monotonic):
        self.config = config or DabaiCameraConfig()
        self.clock, self._sdk = clock, sdk
        self._context = self._devices = self._device = self._info = self._pipeline = self._stream_config = 0
        self._profiles = []
        self._condition = Condition()
        self._stop = Event()
        self._thread = None
        self._connected = False
        self._latest = None
        self._fault = None
        self._sequence = 0
        self._serial_number = self._device_name = self._firmware_version = ""
        self._depth_intrinsics = self._color_intrinsics = self._d2c = None
        self._calibration_source = None
        self._property_originals = {}
        self._color_profile = None
        self._calibration_record = None
        self._lease = CameraLease(label="石子尺寸分析")

    @property
    def serial_number(self):
        return self._serial_number

    @property
    def device_name(self):
        return self._device_name

    @property
    def product_id(self):
        return 0x0657

    @property
    def firmware_version(self):
        return self._firmware_version

    def _select_profile(self, sensor, prefix):
        sdk = self._sdk
        profiles = sdk.call("ob_pipeline_get_stream_profile_list", self._pipeline, sensor)
        try:
            values = [getattr(self.config, f"{prefix}_{key}") for key in ("width", "height", "format", "fps")]
            try:
                selected = sdk.call("ob_stream_profile_list_get_video_stream_profile", profiles,
                                    values[0], values[1], FORMATS[values[2]], values[3])
            except BackendError as error:
                available = []
                for i in range(sdk.call("ob_stream_profile_list_count", profiles)):
                    item = sdk.call("ob_stream_profile_list_get_profile", profiles, i)
                    try:
                        available.append(tuple(sdk.call(name, item) for name in (
                            "ob_video_stream_profile_width", "ob_video_stream_profile_height",
                            "ob_video_stream_profile_fps", "ob_stream_profile_format")))
                    finally:
                        sdk.close_handle("ob_delete_stream_profile", item)
                raise BackendUnavailable(f"Dabai {prefix} profile {values} is unavailable; supported (w,h,fps,format): {available}") from error
            self._profiles.append(selected)
            if prefix == "color":
                self._color_profile = dict(width=values[0], height=values[1], format=values[2], fps=values[3])
            return selected
        finally:
            sdk.close_handle("ob_delete_stream_profile_list", profiles)

    def _read_calibration(self):
        sdk, cfg = self._sdk, self.config
        sizes = ((cfg.depth_width, cfg.depth_height), (cfg.color_width, cfg.color_height))
        attempts = []
        try:
            param = sdk.call("ob_pipeline_get_camera_param_with_profile", self._pipeline,
                             *sizes[1], *sizes[0])
            geometry = camera_param_geometry(param, *sizes)
            self._calibration_source = "sdk_matched_native_profiles"
        except (BackendError, ValueError) as error:
            attempts.append(str(error))
            records = sdk.call("ob_device_get_calibration_camera_param_list", self._device)
            try:
                matches = []
                for index in range(sdk.call("ob_camera_param_list_count", records)):
                    candidate = sdk.call("ob_camera_param_list_get_param", records, index)
                    if ((candidate.depthIntrinsic.width, candidate.depthIntrinsic.height) == sizes[0]
                            and (candidate.rgbIntrinsic.width, candidate.rgbIntrinsic.height) == sizes[1]):
                        matches.append((index, candidate))
                if len(matches) != 1:
                    raise BackendUnavailable(f"Dabai requires one exact native factory calibration record for depth/RGB {sizes}; found {len(matches)}")
                index, param = matches[0]
                geometry = camera_param_geometry(param, *sizes)
                self._calibration_source = f"device_factory_exact_native_record_{index}"
            finally:
                sdk.close_handle("ob_delete_camera_param_list", records)
        self._depth_intrinsics, self._color_intrinsics, self._d2c = geometry
        self._calibration_record = {"source": self._calibration_source,
                                    "matched_profile_accessor_errors": attempts,
                                    "matched_native_depth_size": list(sizes[0]),
                                    "matched_native_color_size": list(sizes[1]),
                                    "depth_to_color_translation_unit": "metres",
                                    "is_mirrored": bool(param.isMirrored), "native_alignment": "disabled"}

    def start(self):
        if self._connected:
            return
        if self._thread is not None:
            raise BackendError("Dabai capture shutdown is incomplete; call stop() before starting again")
        self._lease.acquire()
        try:
            self._sdk = self._sdk or LegacyOrbbecSdk(self.config.legacy_sdk_dir)
            sdk = self._sdk
            self._context = sdk.call("ob_create_context_with_config", str(sdk.config_path).encode())
            self._devices = sdk.call("ob_query_device_list", self._context)
            indices = [i for i in range(sdk.call("ob_device_list_device_count", self._devices))
                       if sdk.call("ob_device_list_get_device_pid", self._devices, i) == self.product_id]
            if len(indices) != 1:
                raise BackendUnavailable(f"expected exactly one Dabai DC1 PID 0x0657, found {len(indices)}; no Gemini fallback")
            self._device = sdk.call("ob_device_list_get_device", self._devices, indices[0])
            self._info = sdk.call("ob_device_get_device_info", self._device)
            def text(name):
                return sdk.call(name, self._info).decode("utf-8", errors="replace")
            self._serial_number = text("ob_device_info_serial_number")
            self._device_name = text("ob_device_info_name")
            self._firmware_version = text("ob_device_info_firmware_version")
            if self.config.serial_number and self.config.serial_number != self.serial_number:
                raise BackendUnavailable(f"Dabai serial mismatch: expected {self.config.serial_number}, got {self.serial_number}")
            if self.config.rgb_device:
                path = Path(self.config.rgb_device)
                if not path.exists() or self.serial_number not in str(path):
                    raise BackendUnavailable("--rgb-device must be this DC1's stable /dev/v4l/by-id path; SDK v1 owns its paired RGB stream")
            self._pipeline = sdk.call("ob_create_pipeline_with_device", self._device)
            dp = self._select_profile(3, "depth")
            cp = self._select_profile(2, "color")
            self._read_calibration()
            # The factory record describes optical coordinates, not a mirrored
            # output. Correct supported device mirror/flip flags temporarily.
            for name in ("OB_PROP_DEPTH_MIRROR_BOOL", "OB_PROP_DEPTH_FLIP_BOOL",
                         "OB_PROP_COLOR_MIRROR_BOOL", "OB_PROP_COLOR_FLIP_BOOL"):
                state = self._read_property(name)
                if state.get("value") is True:
                    self._set_property(name, False)
            self._stream_config = sdk.call("ob_create_config")
            for profile in (dp, cp):
                sdk.call("ob_config_enable_stream", self._stream_config, profile)
            sdk.call("ob_config_set_align_mode", self._stream_config, 0)
            sdk.call("ob_config_set_frame_aggregate_output_mode", self._stream_config, 0)
            sdk.call("ob_pipeline_start_with_config", self._pipeline, self._stream_config)
            self._stop.clear()
            self._fault, self._latest, self._sequence = None, None, 0
            self._connected = True
            self._thread = Thread(target=self._capture_loop, name="dabai-native-rgbd", daemon=True)
            self._thread.start()
            self.read_latest(timeout_s=self.config.startup_timeout_s)
        except Exception as error:
            self.stop()
            if isinstance(error, (BackendError, ValueError)):
                raise
            raise BackendError(f"could not start Dabai DC1: {error}") from error

    def _capture_loop(self):
        misses = 0
        while not self._stop.is_set():
            frameset = depth = color = 0
            try:
                sdk = self._sdk
                frameset = sdk.call("ob_pipeline_wait_for_frameset", self._pipeline, self.config.frame_timeout_ms)
                if not frameset:
                    misses += 1
                    if misses > 20:
                        raise BackendUnavailable("Dabai RGB-D streams stopped delivering complete frames")
                    continue
                depth = sdk.call("ob_frameset_depth_frame", frameset)
                color = sdk.call("ob_frameset_color_frame", frameset)
                if not depth or not color:
                    continue
                received = self.clock()
                sizes = [tuple(sdk.call(name, f) for name in ("ob_video_frame_width", "ob_video_frame_height"))
                         for f in (depth, color)]
                if sizes != [(self.config.depth_width, self.config.depth_height),
                             (self.config.color_width, self.config.color_height)]:
                    raise BackendUnavailable("Dabai received frame dimensions changed; factory calibration no longer matches")
                blobs = [ct.string_at(sdk.call("ob_frame_data", f), sdk.call("ob_frame_data_size", f))
                         for f in (depth, color)]
                formats = [sdk.call("ob_frame_format", f) for f in (depth, color)]
                scale = sdk.call("ob_depth_frame_get_value_scale", depth)
                z = decode_depth(blobs[0], *sizes[0], formats[0], scale,
                                 min_m=self.config.minimum_depth_m, max_m=self.config.maximum_depth_m)
                rgb = decode_rgb(blobs[1], *sizes[1], formats[1])
                timing = {}
                for label, f in (("depth", depth), ("color", color)):
                    timing[label] = {"device_timestamp_us_raw": int(sdk.call("ob_frame_time_stamp_us", f)),
                                     "sdk_system_timestamp_us": int(sdk.call("ob_frame_system_time_stamp_us", f)),
                                     "frame_index": int(sdk.call("ob_frame_index", f))}
                self._sequence += 1
                frame = RGBDFrame(SampleHeader(received, received, self._sequence), z, self._depth_intrinsics,
                                  rgb=rgb, color_intrinsics=self._color_intrinsics,
                                  serial_number=self.serial_number, source="orbbec_dabai_dc1_sdk_v1",
                                  rgb_aligned_to_depth=False,
                                  timestamp_metadata={"selected_source": "host_frameset_receipt", "sdk_version": sdk.version,
                                      **timing, "hardware_frame_sync": False, "pairing": "SDK complete native frameset",
                                      "clock_note": "legacy color and depth device timestamps are not a proven shared clock",
                                      "depth_value_scale_mm": scale, "factory_calibration": self._calibration_record})
                misses = 0
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
            except Exception as error:
                if not self._stop.is_set():
                    with self._condition:
                        self._fault = str(error)
                        self._condition.notify_all()
                return
            finally:
                for handle in (depth, color, frameset):
                    if handle:
                        self._sdk.close_handle("ob_delete_frame", handle)

    def read_latest(self, timeout_s=0., *, after_sequence=None):
        deadline = self.clock() + timeout_s
        with self._condition:
            while True:
                if not self._connected:
                    raise BackendUnavailable("Dabai camera is not connected")
                if self._fault:
                    raise BackendUnavailable(f"Dabai capture failed: {self._fault}")
                frame = self._latest
                if frame is not None and (after_sequence is None or frame.header.sequence > after_sequence):
                    if frame.header.age(self.clock()) > self.config.max_frame_age_s:
                        raise BackendUnavailable("latest Dabai RGB-D frame is stale")
                    return frame
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise BackendUnavailable("timed out waiting for a complete Dabai RGB-D frame")
                self._condition.wait(timeout=remaining)

    def get_depth_to_color_cv(self):
        if not self._connected or self._d2c is None:
            raise BackendUnavailable("Dabai depth/color calibration requires a running stream")
        return self._d2c.copy()

    def _read_property(self, name):
        state = {"status": "unknown", "value": None, "detail": None}
        if not self._device or self._sdk is None:
            return state
        try:
            prop = PROPERTY_IDS[name]
            if not self._sdk.call("ob_device_is_property_supported", self._device, prop, 1):
                state["status"] = "unsupported"
                return state
            function = "ob_device_get_bool_property" if name.endswith("_BOOL") else "ob_device_get_int_property"
            state.update(status="read", value=self._sdk.call(function, self._device, prop))
        except BackendError as error:
            state.update(status="error", detail=str(error))
        return state

    def _set_property(self, name, value):
        state = self._read_property(name)
        prop, sdk = PROPERTY_IDS[name], self._sdk
        if state["status"] != "read" or not sdk.call("ob_device_is_property_supported", self._device, prop, 2):
            raise BackendUnavailable(f"Dabai does not support changing {name}")
        self._property_originals.setdefault(name, state["value"])
        function = "ob_device_set_bool_property" if name.endswith("_BOOL") else "ob_device_set_int_property"
        sdk.call(function, self._device, prop, value)
        actual = self._read_property(name)
        if actual["status"] != "read" or actual["value"] != value:
            raise BackendError(f"Dabai {name} did not accept requested value {value}")
        return actual["value"]

    def set_color_sharpness(self, value):
        if not self._connected:
            raise BackendUnavailable("Dabai sharpness requires a running stream")
        if isinstance(value, bool) or int(value) != value or not 0 <= value <= 100:
            raise ValueError("RGB sharpness must be an integer in 0..100")
        return self._set_property("OB_PROP_COLOR_SHARPNESS_INT", int(value))

    def get_color_geometry_metadata(self):
        return {"schema_version": 1, "actual_profile": self._color_profile,
                "intrinsics_source": self._calibration_source,
                "sdk": {"native_version": None if self._sdk is None else self._sdk.version,
                        "family": "Orbbec SDK v1 legacy OpenNI", "isolated_library": True},
                "factory_calibration": self._calibration_record,
                "device_properties": {name: self._read_property(name) for name in PROPERTY_IDS if name not in (
                    "OB_PROP_COLOR_SHARPNESS_INT", "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", "OB_PROP_COLOR_EXPOSURE_INT", "OB_PROP_COLOR_GAIN_INT")},
                "image_properties": {name: self._read_property(name) for name in (
                    "OB_PROP_COLOR_SHARPNESS_INT", "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", "OB_PROP_COLOR_EXPOSURE_INT", "OB_PROP_COLOR_GAIN_INT")},
                "rgb_rectification": {"application_undistort": False, "application_resize": False,
                                      "application_crop": False, "application_align_filter": False},
                "property_query_unix": time.time()}

    def health(self):
        return DeviceHealth(self._connected, self._latest is not None and not self._fault,
                            "DABAI_CAPTURE_FAILED" if self._fault else None, self._fault or "native RGB-D")

    def stop(self):
        self._stop.set()
        with self._condition:
            self._connected = False
            self._condition.notify_all()
        sdk = self._sdk
        if sdk is not None and self._pipeline:
            sdk.close_handle("ob_pipeline_stop", self._pipeline)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2., self.config.frame_timeout_ms / 1000 + 1.))
            if thread.is_alive():
                # The worker still owns native frame/pipeline references. Keep
                # every handle and the thread so a later stop() can retry.
                raise BackendError("Dabai capture shutdown timed out; native handles retained; retry stop()")
        if sdk is not None:
            for name, value in self._property_originals.items():
                function = "ob_device_set_bool_property" if name.endswith("_BOOL") else "ob_device_set_int_property"
                try:
                    sdk.call(function, self._device, PROPERTY_IDS[name], value)
                except BackendError as error:
                    warnings.warn(f"could not restore Dabai {name}: {error}", RuntimeWarning)
            self._property_originals.clear()
            for profile in self._profiles:
                sdk.close_handle("ob_delete_stream_profile", profile)
            self._profiles.clear()
            for name, attr in (("ob_delete_config", "_stream_config"), ("ob_delete_pipeline", "_pipeline"),
                               ("ob_delete_device_info", "_info"), ("ob_delete_device", "_device"),
                               ("ob_delete_device_list", "_devices"), ("ob_delete_context", "_context")):
                sdk.close_handle(name, getattr(self, attr))
                setattr(self, attr, 0)
        self._thread = None
        self._lease.release()

    close = stop

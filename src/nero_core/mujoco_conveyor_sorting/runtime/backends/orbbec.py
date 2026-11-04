"""Orbbec RGB-D backend using the optional official ``pyorbbecsdk`` v2 API.

The control path consumes metric depth.  RGB is optional because some Orbbec
profiles use compressed formats and the existing geometric detector is
depth-only.  Unsupported or missing frames are reported as faults; this
backend never fabricates zero images.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata as importlib_metadata
from threading import Condition, Event, Thread
import time
import warnings
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np

from ..types import (
    BackendError,
    BackendUnavailable,
    CameraIntrinsics,
    DeviceHealth,
    RGBDFrame,
    SampleHeader,
)


@dataclass(frozen=True)
class OrbbecCameraConfig:
    serial_number: str | None = None
    expected_product_id: int | None = None
    enable_color: bool = True
    require_color: bool = False
    enable_frame_sync: bool = False
    capture_stereo_ir: bool = False
    depth_width: int | None = None
    depth_height: int | None = None
    depth_fps: int | None = None
    depth_format: str | None = None
    color_width: int | None = None
    color_height: int | None = None
    color_fps: int | None = None
    color_format: str | None = None
    depth_exposure_us: int | None = None
    color_exposure_us: int | None = None
    delivery_buffer_frames: int = 8
    frame_timeout_ms: int = 200
    startup_timeout_s: float = 5.0
    max_frame_age_s: float = 0.25
    max_consecutive_timeouts: int = 10
    timestamp_source: str = "global"
    minimum_depth_m: float = 0.02
    maximum_depth_m: float = 5.0
    calibration_id: str = ""
    camera_to_world_cv: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.expected_product_id is not None:
            product_id = int(self.expected_product_id)
            if not 0 <= product_id <= 0xFFFF:
                raise ValueError("expected_product_id must be a 16-bit USB product ID")
            object.__setattr__(self, "expected_product_id", product_id)
        if self.require_color and not self.enable_color:
            raise ValueError("require_color=True requires enable_color=True")
        if self.timestamp_source not in ("global", "device"):
            raise ValueError("timestamp_source must be 'global' or 'device'")
        for prefix in ("depth", "color"):
            exposure=getattr(self,f'{prefix}_exposure_us')
            if exposure is not None and (isinstance(exposure,bool) or int(exposure)!=exposure or exposure<=0):
                raise ValueError(f'{prefix}_exposure_us must be a positive integer')
            values = tuple(
                getattr(self, f"{prefix}_{suffix}")
                for suffix in ("width", "height", "fps")
            )
            if any(item is not None for item in values) and not all(
                item is not None for item in values
            ):
                raise ValueError(
                    f"{prefix}_width, {prefix}_height and {prefix}_fps must be set together"
                )
            if any(item is not None and int(item) <= 0 for item in values):
                raise ValueError(f"{prefix} stream dimensions and fps must be positive")
        if self.frame_timeout_ms <= 0 or self.startup_timeout_s <= 0.0:
            raise ValueError("Orbbec frame timeouts must be positive")
        if (isinstance(self.delivery_buffer_frames,bool)
                or not isinstance(self.delivery_buffer_frames,int)
                or not 1<=self.delivery_buffer_frames<=32):
            raise ValueError('delivery_buffer_frames must be an integer between 1 and 32')
        if self.max_frame_age_s <= 0.0 or self.max_consecutive_timeouts < 1:
            raise ValueError("frame age and consecutive timeout limits must be positive")
        if not 0.0 <= self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("invalid Orbbec metric depth range")
        if bool(self.calibration_id) != (self.camera_to_world_cv is not None):
            raise ValueError(
                "calibration_id and camera_to_world_cv must either both be set or both be absent"
            )
        if self.camera_to_world_cv is not None:
            transform = np.asarray(self.camera_to_world_cv, dtype=float)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError("camera_to_world_cv must be a finite 4x4 transform")
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError("camera_to_world_cv must be homogeneous")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OrbbecCameraConfig":
        document = dict(value)
        backend = str(document.pop("backend", "orbbec")).lower()
        if backend != "orbbec":
            raise ValueError(f"camera backend must be 'orbbec', got {backend!r}")
        known = cls.__dataclass_fields__
        unknown = sorted(set(document) - set(known))
        if unknown:
            raise ValueError(f"unknown Orbbec camera settings: {', '.join(unknown)}")
        transform = document.get("camera_to_world_cv")
        if transform is not None:
            document["camera_to_world_cv"] = tuple(
                tuple(float(item) for item in row) for row in transform
            )
        return cls(**document)


class OrbbecCameraBackend:
    """Asynchronous capture with latest-frame and bounded ordered readers."""

    def __init__(
        self,
        config: OrbbecCameraConfig | Mapping[str, Any] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sdk: ModuleType | Any | None = None,
    ) -> None:
        if config is None:
            config = OrbbecCameraConfig()
        elif isinstance(config, Mapping):
            config = OrbbecCameraConfig.from_mapping(config)
        self.config = config
        self.clock = clock
        self._sdk = sdk
        self._pipeline: Any | None = None
        self._context: Any | None = None
        self._device: Any | None = None
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._condition = Condition()
        self._latest: RGBDFrame | None = None
        self._connected = False
        self._ready = False
        self._fault_code: str | None = None
        self._detail = "not started"
        self._serial_number = ""
        self._product_id: int | None = None
        self._device_name = ""
        self._firmware_version = ""
        self._depth_intrinsics: CameraIntrinsics | None = None
        self._color_intrinsics: CameraIntrinsics | None = None
        self._color_profile: dict[str, Any] | None = None
        self._color_intrinsics_source: str | None = None
        self._sequence = 0
        self._source_offset: float | None = None
        self._last_source_time: float | None = None
        self._last_source_clock: str | None = None
        self._clock_epoch = 0
        self._consecutive_timeouts = 0
        self._original_color_sharpness: int | None = None
        self._stereo_geometry_cache: tuple[Any, dict[str, Any], float] | None = None
        from ..frame_delivery import FrameDelivery
        self._delivery=FrameDelivery(self.config.delivery_buffer_frames)
        self._original_exposure=[]
        self._exposure_report={}

    def get_capture_diagnostics(self):
        with self._condition:
            return {**self._delivery.report(),'exposure_settings':dict(self._exposure_report),
                'serial_number':self._serial_number,'product_id':self._product_id,
                'color_profile':None if self._color_profile is None else dict(self._color_profile)}

    def _apply_exposure_settings(self):
        device,sdk=self._device,self._sdk
        for stream in ('depth','color'):
            requested=getattr(self.config,f'{stream}_exposure_us')
            if requested is None:continue
            if stream=='color' and not self.config.enable_color:
                raise BackendUnavailable('color exposure requested without color stream')
            if self._product_id!=0x0804:
                raise BackendUnavailable('Manual exposure units are verified only for Gemini 335L (0x0804)')
            scale=1 if stream=='depth' else 100
            if requested%scale:
                raise BackendUnavailable(f'{stream} exposure must be a multiple of {scale} us')
            fps=getattr(self.config,f'{stream}_fps')
            if fps and requested>1e6/fps:
                raise BackendUnavailable(f'{stream} exposure exceeds configured frame period')
            auto=getattr(sdk.OBPropertyID,f'OB_PROP_{stream.upper()}_AUTO_EXPOSURE_BOOL')
            prop=getattr(sdk.OBPropertyID,f'OB_PROP_{stream.upper()}_EXPOSURE_INT')
            permission=sdk.OBPermissionType.PERMISSION_READ_WRITE
            if not all(device.is_property_supported(p,permission) for p in (auto,prop)):
                raise BackendUnavailable(f'{stream} manual exposure properties unavailable')
            bounds=device.get_int_property_range(prop);raw=requested//scale
            if not bounds.min<=raw<=bounds.max or (raw-bounds.min)%max(1,bounds.step):
                raise BackendUnavailable(f'{stream} exposure {requested}us outside device range/step')
            self._original_exposure.append((auto,prop,device.get_bool_property(auto),device.get_int_property(prop)))
            device.set_bool_property(auto,False)
            device.set_int_property(prop,int(raw))
            actual=int(device.get_int_property(prop));ae=bool(device.get_bool_property(auto))
            if ae or actual!=raw:
                raise BackendUnavailable(f'{stream} manual exposure readback did not match request')
            self._exposure_report[stream]={'requested_us':requested,'readback_us':actual*scale,
                'property_raw':actual,'property_unit_us':scale,'auto_exposure':ae,
                'verification':'device_property_readback_frame_metadata_still_required'}

    @property
    def serial_number(self) -> str:
        return self._serial_number

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def product_id(self) -> int | None:
        return self._product_id

    @property
    def firmware_version(self) -> str:
        return self._firmware_version

    def get_color_geometry_metadata(self) -> dict[str, Any]:
        """Describe the received RGB geometry and read current device settings.

        This never starts a device, loads the SDK, changes a property or applies
        an image transform. Unsupported/unknown settings are distinct from a
        successful read of ``False``. A generic hardware rectify property does
        not establish whether the RGB stream was rectified.
        """
        with self._condition:
            profile = None if self._color_profile is None else dict(self._color_profile)
            source = self._color_intrinsics_source
            device = self._device
            sdk = self._sdk
        package_version = None
        distribution = None
        for candidate in ("pyorbbecsdk2", "pyorbbecsdk"):
            try:
                package_version = importlib_metadata.version(candidate)
                distribution = candidate
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        native_version = None
        version_error = None
        if sdk is not None and callable(getattr(sdk, "get_version", None)):
            try:
                native_version = str(sdk.get_version())
            except Exception as error:
                version_error = str(error)
        query_time = time.time()
        states: dict[str, Any] = {}
        names = (
            "OB_PROP_COLOR_MIRROR_BOOL", "OB_PROP_COLOR_FLIP_BOOL",
            "OB_PROP_COLOR_ROTATE_INT", "OB_PROP_RGB_CUSTOM_CROP_BOOL",
            "OB_PROP_D2C_PREPROCESS_BOOL", "OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL",
            "OB_PROP_HARDWARE_DISTORTION_SWITCH_BOOL",
        )
        for name in names:
            states[name] = self._read_geometry_property(device, sdk, name)
        return {
            "schema_version": 1,
            "actual_profile": profile,
            "intrinsics_source": source,
            "sdk": {
                "python_distribution": distribution,
                "python_package_version": package_version,
                "native_version": native_version,
                "native_version_error": version_error,
            },
            "device_properties": states,
            "image_properties": {
                name: self._read_geometry_property(device, sdk, name)
                for name in (
                    "OB_PROP_COLOR_SHARPNESS_INT", "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL",
                    "OB_PROP_COLOR_EXPOSURE_INT", "OB_PROP_COLOR_GAIN_INT",
                )
            },
            "property_query_unix": query_time,
            "rgb_rectification": {
                "application_undistort": False,
                "application_resize": False,
                "application_crop": False,
                "application_align_filter": False,
                "sdk_frame_processor": "unknown",
                "hardware_device": "unknown",
                "hardware_rectify_property_is_rgb_specific": False,
            },
        }

    @staticmethod
    def _read_geometry_property(device: Any, sdk: Any, name: str) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "unknown", "value": None, "detail": None}
        if device is None or sdk is None:
            result["detail"] = "no running device"
            return result
        property_id = getattr(getattr(sdk, "OBPropertyID", None), name, None)
        permission = getattr(getattr(sdk, "OBPermissionType", None), "PERMISSION_READ", None)
        supported = getattr(device, "is_property_supported", None)
        getter_name = "get_int_property" if name.endswith("_INT") else "get_bool_property"
        getter = getattr(device, getter_name, None)
        if property_id is None or permission is None or not callable(supported):
            result["detail"] = "read support check unavailable"
            return result
        try:
            if not supported(property_id, permission):
                result["status"] = "unsupported"
                return result
            if not callable(getter):
                result["detail"] = f"{getter_name} unavailable"
                return result
            value = getter(property_id)
            result.update(status="read", value=int(value) if name.endswith("_INT") else bool(value))
        except Exception as error:
            result.update(status="error", detail=str(error))
        return result

    def get_depth_to_color_cv(self) -> np.ndarray:
        """Factory native-depth to native-color optical transform, in metres.

        Only available after receiving the first frame. Missing factory data
        must never be replaced by identity: RGB PnP extrinsics belong to a
        different optical centre than the depth consumed by the runtime.
        """
        if self._pipeline is None or self._latest is None or self._color_intrinsics is None:
            raise BackendUnavailable("depth/color calibration requires a running RGB-D stream")
        try:
            extrinsic = self._pipeline.get_camera_param().transform
            result = np.eye(4)
            result[:3, :3] = np.asarray(extrinsic.rot, dtype=float).reshape(3, 3)
            result[:3, 3] = np.asarray(extrinsic.transform, dtype=float).reshape(3) / 1000.0
        except (AttributeError, TypeError, ValueError) as error:
            raise BackendUnavailable("Orbbec factory depth-to-color extrinsic unavailable") from error
        rotation = result[:3, :3]
        if (not np.isfinite(result).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)):
            raise BackendUnavailable("invalid Orbbec factory depth-to-color extrinsic")
        return result

    def set_color_sharpness(self, value: int) -> int:
        """Set supported hardware edge enhancement for this running session.

        This changes the camera's image processing, not optical focus or pixel
        sampling. The first previous value is restored by ``stop()``; snapshots
        include the current value in their RGB image properties.
        """
        if isinstance(value, bool) or int(value) != value:
            raise ValueError("RGB sharpness must be an integer")
        device, sdk = self._device, self._sdk
        if not self._connected or device is None or sdk is None:
            raise BackendUnavailable("RGB sharpness requires a running camera")
        property_id = getattr(getattr(sdk, "OBPropertyID", None), "OB_PROP_COLOR_SHARPNESS_INT", None)
        permissions = getattr(sdk, "OBPermissionType", None)
        read = getattr(permissions, "PERMISSION_READ", None)
        write = getattr(permissions, "PERMISSION_WRITE", None)
        supported = getattr(device, "is_property_supported", None)
        if (property_id is None or read is None or write is None or not callable(supported)
                or not supported(property_id, read) or not supported(property_id, write)):
            raise BackendUnavailable("camera does not support reading and writing RGB sharpness")
        limits = device.get_int_property_range(property_id)
        minimum, maximum, step = int(limits.min), int(limits.max), int(limits.step)
        value = int(value)
        if not minimum <= value <= maximum or (step > 0 and (value - minimum) % step):
            raise ValueError(f"RGB sharpness must be {minimum}..{maximum} with step {step}")
        previous = int(device.get_int_property(property_id))
        if self._original_color_sharpness is None:
            self._original_color_sharpness = previous
        device.set_int_property(property_id, value)
        actual = int(device.get_int_property(property_id))
        if actual != value:
            raise BackendError(f"RGB sharpness requested {value}, camera reported {actual}")
        return actual

    def health(self) -> DeviceHealth:
        with self._condition:
            return DeviceHealth(
                self._connected,
                self._ready,
                self._fault_code,
                self._detail,
            )

    def start(self) -> None:
        with self._condition:
            if self._connected:
                return
            self._fault_code = None
            self._detail = "starting"
            self._latest = None
            self._depth_intrinsics = None
            self._color_intrinsics = None
            self._color_profile = None
            self._color_intrinsics_source = None
            self._stereo_geometry_cache = None
            self._sequence = 0
            self._source_offset = None
            self._last_source_time = None
            self._last_source_clock = None
            self._clock_epoch = 0
            from ..frame_delivery import FrameDelivery
            self._delivery=FrameDelivery(self.config.delivery_buffer_frames)
            self._exposure_report={}
        sdk = self._load_sdk()
        try:
            context = sdk.Context()
            devices = context.query_devices()
            count = int(devices.get_count())
            if count == 0:
                raise BackendUnavailable("no Orbbec camera is connected")
            if self.config.serial_number:
                device = devices.get_device_by_serial_number(
                    self.config.serial_number
                )
            elif count == 1:
                device = devices.get_device_by_index(0)
            else:
                raise BackendUnavailable(
                    "multiple Orbbec cameras detected; configure serial_number"
                )
            info = device.get_device_info()
            serial = str(info.get_serial_number())
            if self.config.serial_number and serial != self.config.serial_number:
                raise BackendUnavailable(
                    f"Orbbec serial mismatch: expected {self.config.serial_number!r}, got {serial!r}"
                )
            get_pid = getattr(info, "get_pid", None)
            product_id = None if get_pid is None else int(get_pid())
            if self.config.expected_product_id is not None:
                if product_id is None:
                    raise BackendUnavailable(
                        "Orbbec SDK did not report a product ID; cannot verify the configured model"
                    )
                if product_id != self.config.expected_product_id:
                    raise BackendUnavailable(
                        "Orbbec product mismatch: expected "
                        f"0x{self.config.expected_product_id:04X}, got 0x{product_id:04X}"
                    )
            pipeline = sdk.Pipeline(device)
            stream_config = self._build_stream_config(sdk, pipeline)
            if self.config.enable_frame_sync or self.config.capture_stereo_ir:
                pipeline.enable_frame_sync()
            pipeline.start(stream_config)
            self._context = context
            self._device = device
            self._pipeline = pipeline
            self._serial_number = serial
            self._product_id = product_id
            self._device_name = str(info.get_name())
            firmware = getattr(info, "get_firmware_version", None)
            self._firmware_version = "" if firmware is None else str(firmware())
            self._apply_exposure_settings()
            self._stop_event.clear()
            with self._condition:
                self._connected = True
                self._ready = False
                self._detail = "waiting for first RGB-D frame"
            self._thread = Thread(
                target=self._capture_loop,
                name="orbbec-latest-frame",
                daemon=True,
            )
            self._thread.start()
            self.read_latest(timeout_s=self.config.startup_timeout_s)
        except Exception as error:
            self._mark_fault("ORBBEC_START_FAILED", str(error))
            self.stop()
            if isinstance(error, BackendError):
                raise
            raise BackendError(f"could not start Orbbec camera: {error}") from error

    def read_latest(
        self, timeout_s: float = 0.0, *, after_sequence: int | None = None,
        wait_for_fresh: bool = False,
    ) -> RGBDFrame:
        """Return a fresh frame; optionally await replacement of stale cache.

        Waiting is opt-in for startup/stationary callers. Motion monitoring
        retains the default immediate rejection of stale buffered frames.
        """
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative")
        deadline = float(self.clock()) + float(timeout_s)
        with self._condition:
            while True:
                if not self._connected:
                    detail = self._detail or "not started"
                    raise BackendUnavailable(f"Orbbec camera is not connected: {detail}")
                frame = self._latest
                is_new_enough = frame is not None and (
                    after_sequence is None
                    or frame.header.sequence > int(after_sequence)
                )
                if is_new_enough:
                    age = frame.header.age(float(self.clock()))
                    if age <= self.config.max_frame_age_s:
                        return frame
                    if not wait_for_fresh:
                        raise BackendUnavailable(
                            f"latest Orbbec frame is stale ({age:.3f}s)"
                        )
                if self._fault_code:
                    raise BackendUnavailable(
                        f"Orbbec camera fault {self._fault_code}: {self._detail}"
                    )
                remaining = deadline - float(self.clock())
                if remaining <= 0.0:
                    if is_new_enough and wait_for_fresh:
                        raise BackendUnavailable(
                            f"timed out waiting for a fresh Orbbec RGB-D frame; latest age {age:.3f}s"
                        )
                    raise BackendUnavailable("timed out waiting for an Orbbec RGB-D frame")
                self._condition.wait(timeout=remaining)

    def read_next(self,timeout_s=0.,*,after_sequence=None,wait_for_fresh=True):
        """Single perception consumer; keep short bursts without latest-cache skips."""
        if timeout_s<0:raise ValueError('timeout_s must be non-negative')
        deadline=float(self.clock())+timeout_s
        with self._condition:
            while True:
                if not self._connected:raise BackendUnavailable(f'Orbbec camera is not connected: {self._detail}')
                frame=self._delivery.take(after_sequence,float(self.clock()),self.config.max_frame_age_s)
                if frame is not None:return frame
                if self._fault_code:raise BackendUnavailable(f'Orbbec camera fault {self._fault_code}: {self._detail}')
                remaining=deadline-float(self.clock())
                if remaining<=0:raise BackendUnavailable('timed out waiting for the next fresh Orbbec frame')
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        self._stop_event.set()
        for auto,prop,ae,raw in reversed(self._original_exposure):
            try:
                self._device.set_bool_property(auto,False)
                self._device.set_int_property(prop,raw)
                self._device.set_bool_property(auto,ae)
            except Exception as error:warnings.warn(f'could not restore camera exposure: {error}',RuntimeWarning)
        self._original_exposure=[]
        if self._original_color_sharpness is not None and self._device is not None:
            try:
                property_id = self._sdk.OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT
                self._device.set_int_property(property_id, self._original_color_sharpness)
            except Exception as error:
                warnings.warn(f"could not restore RGB sharpness: {error}", RuntimeWarning)
            self._original_color_sharpness = None
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, 2.0 * self.config.frame_timeout_ms / 1000.0))
        with self._condition:
            self._connected = False
            self._ready = False
            self._detail = "stopped" if self._fault_code is None else self._detail
            self._condition.notify_all()
        self._pipeline = None
        self._device = None
        self._context = None
        self._thread = None

    close = stop

    def _load_sdk(self) -> Any:
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("pyorbbecsdk")
            except ImportError as error:
                raise BackendUnavailable(
                    "Orbbec SDK is not installed; install the optional 'orbbec' extra "
                    "and the vendor udev rules"
                ) from error
        for name in ("Context", "Pipeline", "Config", "OBSensorType"):
            if not hasattr(self._sdk, name):
                raise BackendUnavailable(
                    f"installed pyorbbecsdk does not provide the required v2 API: {name}"
                )
        return self._sdk

    def _build_stream_config(self, sdk: Any, pipeline: Any) -> Any:
        config = sdk.Config()
        depth_profile = self._select_profile(
            sdk,
            pipeline,
            sdk.OBSensorType.DEPTH_SENSOR,
            self.config.depth_width,
            self.config.depth_height,
            self.config.depth_fps,
            self.config.depth_format,
        )
        config.enable_stream(depth_profile)
        if self.config.capture_stereo_ir:
            for name in ("LEFT_IR_SENSOR", "RIGHT_IR_SENSOR"):
                sensor = getattr(sdk.OBSensorType, name, None)
                if sensor is None:
                    raise BackendUnavailable(
                        f"stereo IR capture requires SDK {name}; RGB-D-only capture remains available"
                    )
                config.enable_stream(self._select_profile(
                    sdk, pipeline, sensor, 1280, 800, 30, "Y8"
                ))
        if self.config.enable_color:
            try:
                color_profile = self._select_profile(
                    sdk,
                    pipeline,
                    sdk.OBSensorType.COLOR_SENSOR,
                    self.config.color_width,
                    self.config.color_height,
                    self.config.color_fps,
                    self.config.color_format,
                )
                config.enable_stream(color_profile)
            except Exception:
                if self.config.require_color:
                    raise
        aggregate = getattr(sdk, "OBFrameAggregateOutputMode", None)
        if (self.config.require_color or self.config.capture_stereo_ir) and aggregate is not None:
            config.set_frame_aggregate_output_mode(aggregate.FULL_FRAME_REQUIRE)
        return config

    @staticmethod
    def _select_profile(
        sdk: Any,
        pipeline: Any,
        sensor_type: Any,
        width: int | None,
        height: int | None,
        fps: int | None,
        format_name: str | None,
    ) -> Any:
        sensor = getattr(sensor_type, "name", str(sensor_type))
        requested = ("SDK default" if width is None else
                     f"{width}x{height} {str(format_name).upper()}@{fps}")
        try:
            profiles = pipeline.get_stream_profile_list(sensor_type)
        except Exception as error:
            raise BackendUnavailable(
                f"Orbbec {sensor} profile list unavailable; requested {requested}; SDK: {error}"
            ) from error
        if width is None:
            return profiles.get_default_video_stream_profile()
        if format_name is None:
            raise ValueError("a stream format is required with an explicit profile")
        try:
            stream_format = getattr(sdk.OBFormat, str(format_name).upper())
        except AttributeError as error:
            raise ValueError(f"unsupported Orbbec format {format_name!r}") from error
        # An explicitly requested profile is strict: silently selecting a
        # different mode would invalidate saved intrinsics and calibration.
        try:
            selected = profiles.get_video_stream_profile(
                int(width), int(height), stream_format, int(fps)
            )
            if selected is None:
                raise ValueError("SDK returned no matching profile")
            return selected
        except Exception as error:
            available = []
            enumeration_error = None
            try:
                for index in range(int(profiles.get_count())):
                    profile = profiles.get_stream_profile_by_index(index)
                    if hasattr(profile, "as_video_stream_profile"):
                        profile = profile.as_video_stream_profile()
                    available.append(
                        f"{profile.get_width()}x{profile.get_height()} "
                        f"{OrbbecCameraBackend._format_name(profile)}@{profile.get_fps()}"
                    )
            except Exception as diagnostic_error:
                enumeration_error = str(diagnostic_error)
            detail = ", ".join(available) if available else "none reported"
            if enumeration_error:
                detail += f"; profile enumeration failed: {enumeration_error}"
            raise BackendUnavailable(
                f"Orbbec {sensor} profile unavailable; requested {requested}; "
                f"available: {detail}; SDK: {error}. "
                "Check USB link and select an explicitly supported profile; no fallback was applied."
            ) from error

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                assert self._pipeline is not None
                frames = self._pipeline.wait_for_frames(self.config.frame_timeout_ms)
                if frames is None:
                    self._consecutive_timeouts += 1
                    if self._consecutive_timeouts >= self.config.max_consecutive_timeouts:
                        raise BackendUnavailable(
                            f"{self._consecutive_timeouts} consecutive Orbbec frame timeouts"
                        )
                    continue
                frame = self._convert_frameset(frames)
                self._consecutive_timeouts = 0
                with self._condition:
                    self._latest = frame
                    self._delivery.append(frame)
                    self._ready = True
                    self._fault_code = None
                    self._detail = (
                        f"{self._device_name} serial={self._serial_number} streaming"
                    )
                    self._condition.notify_all()
            except Exception as error:
                if self._stop_event.is_set():
                    return
                self._mark_fault("ORBBEC_CAPTURE_FAILED", str(error))
                return

    def _convert_frameset(self, frames: Any) -> RGBDFrame:
        # Bracket the wall-clock read on the same monotonic clock used by the
        # frame header. This maps SDK host-domain timestamps without including
        # an unknown camera transport delay in the clock offset. Keep the read
        # span so calibration can reject an imprecise clock reference.
        monotonic_before = float(self.clock())
        host_unix = float(time.time())
        received = float(self.clock())
        host_clock_reference = {
            "unix_s": host_unix,
            "monotonic_s": (monotonic_before + received) / 2.,
            "read_span_s": received - monotonic_before,
        }
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            raise BackendUnavailable("Orbbec frameset does not contain depth")
        color_frame = frames.get_color_frame() if self.config.enable_color else None
        if self.config.require_color and color_frame is None:
            raise BackendUnavailable("Orbbec frameset does not contain required color")
        width = int(depth_frame.get_width())
        height = int(depth_frame.get_height())
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).copy()
        if raw.size != width * height:
            raise BackendUnavailable(
                f"Orbbec depth payload has {raw.size} pixels, expected {width * height}"
            )
        # Orbbec v2 reports the scale in millimetres per raw depth unit.
        depth_m = raw.reshape(height, width).astype(np.float32)
        depth_scale_mm = float(depth_frame.get_depth_scale())
        depth_m *= depth_scale_mm / 1000.0
        invalid = (
            (depth_m < self.config.minimum_depth_m)
            | (depth_m > self.config.maximum_depth_m)
            | ~np.isfinite(depth_m)
        )
        # Separate device zeros from nonzero values discarded by the existing
        # metric range check. Diagnostics never fill depth or change limits.
        raw_zero_pixels = int(raw.size-np.count_nonzero(raw))
        discarded_nonzero_pixels = int(np.count_nonzero(invalid.reshape(-1) & (raw != 0)))
        depth_m[invalid] = 0.0
        self._ensure_intrinsics(depth_frame, color_frame, width, height)
        assert self._depth_intrinsics is not None
        rgb = None if color_frame is None else self._color_to_rgb(color_frame)
        if rgb is not None and color_frame is not None and rgb.shape != (
            int(color_frame.get_height()), int(color_frame.get_width()), 3
        ):
            raise BackendUnavailable("decoded Orbbec RGB dimensions do not match the color frame")
        if self.config.require_color and rgb is None:
            raise BackendUnavailable("required Orbbec color format could not be decoded")
        depth_timing = self._frame_timestamps(depth_frame, self._sdk,exposure_unit_us=1)
        source_time, source_clock = self._select_source_timestamp(depth_timing)
        sample_time = self._map_source_time(source_time, received, source_clock)
        timestamp_metadata = {
            "schema_version": 2, "selected_source": source_clock,
            "host_clock_reference": host_clock_reference,
            "depth": depth_timing,
            "depth_conversion": {
                "source": "sdk_uint16_depth_before_application_range_filter",
                "input_shape": [height, width],
                "depth_scale_mm_per_raw_unit": depth_scale_mm,
                "metric_range_m": [self.config.minimum_depth_m, self.config.maximum_depth_m],
                "raw_zero_pixels": raw_zero_pixels,
                "application_rejected_nonzero_pixels": discarded_nonzero_pixels,
                "retained_nonzero_pixels": int(raw.size-raw_zero_pixels-discarded_nonzero_pixels),
                "depth_filled": False,
            },
            "color": None if color_frame is None else self._frame_timestamps(color_frame, self._sdk,
                exposure_unit_us=100 if self._product_id==0x0804 else None),
        }
        stereo = self._stereo_frames(frames, depth_frame, depth_timing)
        if self.config.capture_stereo_ir:
            timestamp_metadata.update(stereo.pop("timestamp_metadata"))
        self._sequence += 1
        return RGBDFrame(
            header=SampleHeader(
                sample_time=sample_time,
                received_time=received,
                sequence=self._sequence,
                source_time=source_time,
                clock_epoch=self._clock_epoch,
            ),
            depth_m=depth_m,
            depth_intrinsics=self._depth_intrinsics,
            rgb=rgb,
            color_intrinsics=self._color_intrinsics,
            camera_to_world_cv=(
                None
                if self.config.camera_to_world_cv is None
                else np.asarray(self.config.camera_to_world_cv, dtype=float)
            ),
            serial_number=self._serial_number,
            calibration_id=self.config.calibration_id,
            source="orbbec_pyorbbecsdk_v2",
            timestamp_metadata=timestamp_metadata,
            **stereo,
        )

    def _stereo_frames(self, frames: Any, depth_frame: Any,
                       depth_timing: dict[str, Any]) -> dict[str, Any]:
        """Read optional same-capture inputs without replacing device depth.

        Missing bindings/frames/calibration are explicit metadata. They are not
        fabricated images, guessed baseline values or new capture timestamps.
        """
        if not self.config.capture_stereo_ir:
            return {}
        result: dict[str, Any] = {}
        timing: dict[str, Any] = {}
        errors: dict[str, str] = {}
        profiles = {}
        for side in ("left", "right"):
            name = f"{side}_ir"
            try:
                frame_type = getattr(self._sdk.OBFrameType, f"{side.upper()}_IR_FRAME")
                frame = frames.get_frame_by_type(frame_type)
                if frame is None:
                    raise BackendUnavailable(f"frameset does not contain {name}")
                video = frame.as_video_frame() if hasattr(frame, "as_video_frame") else frame
                width, height = int(video.get_width()), int(video.get_height())
                if self._format_name(frame) != "Y8":
                    raise BackendUnavailable(f"{name} must be native Y8, got {self._format_name(frame)}")
                image = np.frombuffer(frame.get_data(), dtype=np.uint8)
                if image.size != width * height:
                    raise BackendUnavailable(f"{name} payload does not match {width}x{height}")
                profile = frame.get_stream_profile()
                if hasattr(profile, "as_video_stream_profile"):
                    profile = profile.as_video_stream_profile()
                intrinsic = self._convert_intrinsics(profile.get_intrinsic(), profile.get_distortion())
                if (intrinsic.width, intrinsic.height) != (width, height):
                    raise BackendUnavailable(f"{name} active frame does not match factory intrinsics")
                result[name] = image.reshape(height, width).copy()
                result[f"{name}_intrinsics"] = intrinsic
                profiles[name] = profile
                timing[name] = self._frame_timestamps(frame, self._sdk,
                    exposure_unit_us=1 if self._product_id==0x0804 else None)
            except Exception as error:
                timing[name] = None
                errors[name] = str(error)
        synchronization = self._stereo_synchronization(depth_timing, timing)
        geometry: dict[str, Any] = {"status": "unavailable"}
        if len(profiles) == 2:
            try:
                # The profile signature prevents reuse after a geometry change.
                key = tuple((name, result[f"{name}_intrinsics"], self._format_name(profiles[name]),
                             int(profiles[name].get_fps())) for name in ("left_ir", "right_ir"))
                if self._stereo_geometry_cache is not None and key == self._stereo_geometry_cache[0]:
                    geometry, baseline = self._stereo_geometry_cache[1:]
                else:
                    left = profiles["left_ir"]
                    depth_profile = depth_frame.get_stream_profile()
                    to_right = self._profile_extrinsic_cv(left, profiles["right_ir"])
                    to_depth = self._profile_extrinsic_cv(left, depth_profile)
                    baseline = float(np.linalg.norm(np.asarray(to_right)[:3, 3]))
                    if not np.isfinite(baseline) or baseline <= 0.0:
                        raise BackendUnavailable("factory stereo baseline is not positive")
                    geometry = {
                        "status": "available", "source": "active_frame_stream_profiles",
                        "translation_units": "metres", "application_resize": False,
                        "application_crop": False, "application_rectification": False,
                        "left_ir_to_right_ir_cv": to_right, "left_ir_to_depth_cv": to_depth,
                        "profiles": {name: {"width": result[name].shape[1],
                            "height": result[name].shape[0], "format": self._format_name(profile),
                            "fps": int(profile.get_fps())} for name, profile in profiles.items()},
                    }
                    self._stereo_geometry_cache = (key, geometry, baseline)
                result["stereo_baseline_m"] = baseline
            except Exception as error:
                errors["calibration"] = str(error)
        result["stereo_calibration"] = geometry
        result["timestamp_metadata"] = {
            **timing, "stereo_ir": {"requested": True,
                "status": "available" if not errors else "incomplete",
                **synchronization, "read_errors": errors,
                "header_clock_unchanged": True,
                "sample_time_reference": "same device depth header; no processing-time restamp"}}
        return result

    @staticmethod
    def _profile_extrinsic_cv(source: Any, destination: Any) -> list[list[float]]:
        extrinsic = source.get_extrinsic_to(destination)
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(extrinsic.rot, dtype=float).reshape(3, 3)
        matrix[:3, 3] = np.asarray(extrinsic.transform, dtype=float).reshape(3) / 1000.0
        if not np.all(np.isfinite(matrix)):
            raise BackendUnavailable("factory stereo extrinsics are not finite")
        return matrix.tolist()

    @staticmethod
    def _stereo_synchronization(depth: dict[str, Any], ir: dict[str, Any]) -> dict[str, Any]:
        checks = {}
        observations = [depth, ir.get("left_ir"), ir.get("right_ir")]
        for name in ("frame_index", "device_time_s", "sensor_midpoint_device_time_s",
                     "exposure_us", "gain_raw"):
            values = [item.get(name) if item is not None else None for item in observations]
            checks[name] = None if any(value is None for value in values) else len(set(values)) == 1
        # Unavailable metadata is unknown, rather than evidence of synchronization.
        verdict = False if False in checks.values() else (True if all(
            value is True for value in checks.values()) else None)
        return {"synchronized": verdict, "same_depth_ir": checks}

    def _ensure_intrinsics(
        self,
        depth_frame: Any,
        color_frame: Any | None,
        width: int,
        height: int,
    ) -> None:
        if self._depth_intrinsics is None:
            self._depth_intrinsics, _ = self._frame_intrinsics(depth_frame, "depth")
        if (
            self._depth_intrinsics.width != width
            or self._depth_intrinsics.height != height
        ):
            raise BackendUnavailable(
                "Orbbec active depth frame does not match factory intrinsics"
            )
        if color_frame is None:
            return
        profile, video_profile = self._read_color_profile(color_frame)
        previous = self._color_profile
        if previous is not None:
            for name in ("width", "height", "format", "fps"):
                before, after = previous[name], profile[name]
                if before is not None and after is not None and before != after:
                    raise BackendUnavailable(f"Orbbec color profile {name} changed while streaming")
        if self._color_intrinsics is None:
            self._color_intrinsics, self._color_intrinsics_source = self._frame_intrinsics(
                color_frame, "rgb", video_profile=video_profile
            )
        if (self._color_intrinsics.width != profile["width"]
                or self._color_intrinsics.height != profile["height"]):
            raise BackendUnavailable("Orbbec active color frame does not match color intrinsics")
        self._color_profile = profile

    def _frame_intrinsics(
        self, frame: Any, prefix: str, *, video_profile: Any | None = None
    ) -> tuple[CameraIntrinsics, str]:
        try:
            profile = video_profile if video_profile is not None else frame.get_stream_profile()
            if hasattr(profile, "as_video_stream_profile"):
                profile = profile.as_video_stream_profile()
            intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
            source = "frame_stream_profile"
        except Exception:
            # Older bindings expose only the active pipeline's matched params.
            assert self._pipeline is not None
            camera = self._pipeline.get_camera_param()
            intrinsic = getattr(camera, f"{prefix}_intrinsic")
            distortion = getattr(camera, f"{prefix}_distortion")
            source = "pipeline_camera_param_fallback"
        return self._convert_intrinsics(intrinsic, distortion), source

    def _read_color_profile(self, frame: Any) -> tuple[dict[str, Any], Any | None]:
        width, height = int(frame.get_width()), int(frame.get_height())
        result = {
            "width": width, "height": height, "format": self._format_name(frame),
            "fps": None, "stream_profile_status": "unavailable", "profile_read_error": None,
        }
        profile = None
        try:
            profile = frame.get_stream_profile()
            if hasattr(profile, "as_video_stream_profile"):
                profile = profile.as_video_stream_profile()
            if profile is not None:
                result["stream_profile_status"] = "available"
        except Exception as error:
            result["profile_read_error"] = str(error)
            profile = None
        if profile is not None:
            for dimension, expected in (("width", width), ("height", height)):
                getter = getattr(profile, f"get_{dimension}", None)
                if callable(getter) and int(getter()) != expected:
                    raise BackendUnavailable(f"Orbbec color stream profile {dimension} does not match frame")
            getter = getattr(profile, "get_fps", None)
            if callable(getter):
                try:
                    result["fps"] = int(getter())
                except Exception as error:
                    result["profile_read_error"] = f"fps: {error}"
            getter = getattr(profile, "get_format", None)
            if callable(getter):
                if self._format_name(profile) != result["format"]:
                    raise BackendUnavailable("Orbbec color stream profile format does not match frame")
        return result, profile

    @staticmethod
    def _convert_intrinsics(intrinsics: Any, distortion: Any) -> CameraIntrinsics:
        coefficients = tuple(
            float(getattr(distortion, name))
            for name in ("k1", "k2", "p1", "p2", "k3")
        )
        if all(hasattr(distortion, name) for name in ("k4", "k5", "k6")):
            coefficients += tuple(float(getattr(distortion, name)) for name in ("k4", "k5", "k6"))
        model = str(getattr(distortion, "model", "orbbec_factory"))
        return CameraIntrinsics(
            int(intrinsics.width),
            int(intrinsics.height),
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
            coefficients,
            model,
        )

    @staticmethod
    def _frame_timestamps(frame: Any, sdk: Any | None = None, *, exposure_unit_us=None) -> dict[str, Any]:
        """Keep device and fitted global clocks distinct, on the actual frame.

        Global timestamps are SDK regression estimates in the host clock domain;
        their corrections are not evidence that the raw device clock reset.
        """
        result: dict[str, Any] = {}
        errors = {}
        for field, getters in (
            ("device_time_s", (("get_timestamp_us", 1e-6), ("get_timestamp", 1e-3))),
            ("global_time_s", (("get_global_timestamp_us", 1e-6),)),
            ("system_time_s", (("get_system_timestamp_us", 1e-6), ("get_system_timestamp", 1e-3))),
            ("frame_index", (("get_index", 1),)),
        ):
            result[field] = None
            for name, scale in getters:
                method = getattr(frame, name, None)
                if method is None:
                    continue
                try:
                    value = float(method()) * scale
                    if not np.isfinite(value) or value < 0:
                        raise ValueError("timestamp/index must be finite and nonnegative")
                except Exception as error:
                    errors[name] = str(error)
                    continue
                result[field] = int(value) if field == "frame_index" else value
                result[field + "_getter"] = name
                break
        if errors:
            result["read_errors"] = errors
        metadata = {}
        metadata_errors = {}
        kinds = getattr(sdk, "OBFrameMetadataType", None)
        has_metadata = getattr(frame, "has_metadata", None)
        getter = getattr(frame, "get_metadata_value", None)
        for name in ("EXPOSURE", "GAIN", "AUTO_EXPOSURE", "SENSOR_TIMESTAMP", "TIMESTAMP",
                     "ACTUAL_FRAME_RATE", "HDR_SEQUENCE_SIZE", "HDR_SEQUENCE_INDEX", "HDR_SEQUENCE_NAME"):
            kind = getattr(kinds, name, None)
            if kind is None or not callable(has_metadata) or not callable(getter):
                continue
            try:
                if has_metadata(kind):
                    metadata[name] = int(getter(kind))
            except Exception as error:
                metadata_errors[name] = str(error)
        if metadata:
            result["frame_metadata"] = metadata
        if metadata_errors:
            result["metadata_read_errors"] = metadata_errors
        result['exposure_raw']=metadata.get('EXPOSURE')
        result['exposure_unit_us']=exposure_unit_us
        result["exposure_us"] = (None if exposure_unit_us is None or metadata.get('EXPOSURE') is None
                                 else metadata['EXPOSURE']*exposure_unit_us)
        result["gain_raw"] = metadata.get("GAIN")
        midpoint_us = metadata.get("SENSOR_TIMESTAMP")
        result["sensor_midpoint_device_time_s"] = None if midpoint_us is None else midpoint_us * 1e-6
        # SENSOR_TIMESTAMP belongs to the device clock. Transfer only its
        # measured offset from SDK end time, rather than mixing device/global
        # epochs. These diagnostic fields do not change header or KF time.
        device_end = result.get("device_time_s")
        midpoint = result["sensor_midpoint_device_time_s"]
        if device_end is not None and midpoint is not None:
            delta = midpoint - device_end
            result["sensor_midpoint_offset_from_device_timestamp_s"] = delta
            global_end = result.get("global_time_s")
            result["sensor_midpoint_global_time_s"] = None if global_end is None else global_end + delta
        return result

    def _select_source_timestamp(self, timing: dict[str, Any]) -> tuple[float, str]:
        # Preserve the runtime default; teach recording explicitly selects the
        # raw device clock and must never silently fall back to fitted global time.
        choices = ("device",) if self.config.timestamp_source == "device" else ("global", "device")
        for clock in choices:
            value = timing.get(clock + "_time_s")
            if value is not None:
                return value, "depth_" + clock
        raise BackendUnavailable(
            f"Orbbec depth frame has no valid {self.config.timestamp_source} timestamp; "
            f"timestamp metadata: {timing}"
        )

    def _map_source_time(self, source_time: float, received: float,
                         source_clock: str | None = None) -> float:
        # A reconnect or device timestamp reset opens a new clock epoch.  The
        # offset is anchored again instead of publishing a non-monotonic time.
        if (
            self._source_offset is None
            or self._last_source_time is None
            or source_time < self._last_source_time - 1e-6
            or source_clock != self._last_source_clock
        ):
            if self._source_offset is not None:
                self._clock_epoch += 1
            self._source_offset = received - source_time
        self._last_source_time = source_time
        self._last_source_clock = source_clock
        return min(received, source_time + self._source_offset)

    @staticmethod
    def _format_name(frame: Any) -> str:
        value = frame.get_format()
        name = getattr(value, "name", str(value)).upper()
        return name.split(".")[-1].removeprefix("OB_FORMAT_")

    def _color_to_rgb(self, frame: Any) -> np.ndarray | None:
        width = int(frame.get_width())
        height = int(frame.get_height())
        fmt = self._format_name(frame)
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if fmt == "RGB":
            expected = width * height * 3
            return None if data.size != expected else data.reshape(height, width, 3).copy()
        if fmt == "BGR":
            expected = width * height * 3
            if data.size != expected:
                return None
            return data.reshape(height, width, 3)[..., ::-1].copy()
        if fmt in {"RGBA", "BGRA"}:
            expected = width * height * 4
            if data.size != expected:
                return None
            image = data.reshape(height, width, 4)[..., :3]
            return image.copy() if fmt == "RGBA" else image[..., ::-1].copy()
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            return None
        if fmt in {"MJPG", "MJPEG"}:
            decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return None if decoded is None else cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        conversions = {
            "YUYV": cv2.COLOR_YUV2RGB_YUY2,
            "YUY2": cv2.COLOR_YUV2RGB_YUY2,
            "UYVY": cv2.COLOR_YUV2RGB_UYVY,
        }
        if fmt in conversions and data.size == width * height * 2:
            packed = data.reshape(height, width, 2)
            return cv2.cvtColor(packed, conversions[fmt])
        return None

    def _mark_fault(self, code: str, detail: str) -> None:
        with self._condition:
            self._fault_code = code
            self._detail = detail
            self._ready = False
            self._condition.notify_all()

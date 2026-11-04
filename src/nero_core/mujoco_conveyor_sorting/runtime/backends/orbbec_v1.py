"""Small ctypes bridge to the isolated official Orbbec SDK v1 release.

Legacy OpenNI cameras such as DaBai DC1 need this SDK.  Importing this module
does not load a native library or enumerate hardware, and never imports the
SDK v2 Python extension.  ``LegacyOrbbecSdk`` loads only the selected vendor
library; camera ownership and frame lifetimes belong to its caller.
"""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

from ..types import BackendError, BackendUnavailable


class OBIntrinsic(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("fx", ct.c_float), ("fy", ct.c_float),
        ("cx", ct.c_float), ("cy", ct.c_float),
        ("width", ct.c_int16), ("height", ct.c_int16),
    ]


class OBDistortion(ct.Structure):
    _pack_ = 1
    _fields_ = [(name, ct.c_float) for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")]


class OBExtrinsic(ct.Structure):
    _pack_ = 1
    _fields_ = [("rot", ct.c_float * 9), ("trans", ct.c_float * 3)]


class OBCameraParam(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("depthIntrinsic", OBIntrinsic), ("rgbIntrinsic", OBIntrinsic),
        ("depthDistortion", OBDistortion), ("rgbDistortion", OBDistortion),
        ("transform", OBExtrinsic), ("isMirrored", ct.c_bool),
    ]


class OBIntPropertyRange(ct.Structure):
    _pack_ = 1
    _fields_ = [(name, ct.c_int32) for name in ("cur", "max", "min", "step", "def")]


class OBBoolPropertyRange(ct.Structure):
    _pack_ = 1
    _fields_ = [(name, ct.c_bool) for name in ("cur", "max", "min", "step", "def")]


# ObTypes.h explicitly packs these structs to one byte, including the trailing
# bool.  A naturally aligned CameraParam would be 156 bytes instead of 153.
if (ct.sizeof(OBIntrinsic), ct.sizeof(OBDistortion), ct.sizeof(OBExtrinsic),
        ct.sizeof(OBCameraParam), OBCameraParam.isMirrored.offset) != (20, 32, 48, 153, 152):
    raise RuntimeError("Orbbec SDK v1 ctypes calibration ABI is incompatible")


class LegacyOrbbecSdk:
    """Checked calls to SDK v1 C functions with their trailing ``ob_error**``.

    String results remain bytes, pointers remain integer handles, and returned
    calibration structures retain the SDK's field order and native units.
    Construction reads the SDK version but does not create a context or open a
    camera.  Use a separate process when another Orbbec SDK is already loaded.
    """

    def __init__(self, sdk_dir: Path | str | None = None):
        project = Path(__file__).resolve().parents[3]
        root = (project / "vendor" / "orbbec_sdk_v1" / "OrbbecSDK_v1.10.37" / "SDK"
                if sdk_dir is None else Path(sdk_dir).expanduser())
        if not (root / "lib").is_dir() and (root / "SDK" / "lib").is_dir():
            root = root / "SDK"
        self.sdk_dir = root.resolve()
        self.config_path = self.sdk_dir / "config" / "OrbbecSDKConfig_v1.0.xml"
        self.cleanup_errors: list[str] = []
        if not self.config_path.is_file():
            raise BackendUnavailable(f"Orbbec SDK v1 configuration not found: {self.config_path}")
        library = self.sdk_dir / "lib" / "libOrbbecSDK.so"
        if not library.is_file():
            raise BackendUnavailable(f"Orbbec SDK v1 library not found: {library}")
        self._reject_other_sdk(library.resolve())
        try:
            self._library = ct.CDLL(str(library), mode=ct.RTLD_LOCAL)
        except OSError as error:
            raise BackendUnavailable(f"could not load Orbbec SDK v1 {library}: {error}") from error
        self._functions: dict[str, object] = {}
        self._bind_functions()
        versions = []
        for name in ("ob_get_major_version", "ob_get_minor_version", "ob_get_patch_version"):
            function = self._bind(name, ct.c_int, (), error_argument=False)
            versions.append(int(function()))
        self.version = ".".join(str(value) for value in versions)
        if versions[0] != 1:
            raise BackendUnavailable(f"expected Orbbec SDK v1, loaded {self.version}")
        if self.has_function("ob_set_logger_severity"):
            try:
                self.call("ob_set_logger_severity", 3)  # OB_LOG_SEVERITY_ERROR; API failures still include their details.
            except BackendError:
                pass  # Logger configuration is optional and cannot block use.

    @staticmethod
    def _reject_other_sdk(target: Path) -> None:
        """Reject conflicting native SDKs before loading this library on Linux."""
        try:
            mappings = Path("/proc/self/maps").read_text()
        except OSError:
            return
        for line in mappings.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            mapped = fields[5].removesuffix(" (deleted)")
            filename = Path(mapped).name
            if "pyorbbecsdk" in filename and ".so" in filename:
                raise BackendUnavailable("Orbbec Python SDK is already loaded; run the legacy SDK in a separate process")
            if "libOrbbecSDK.so" in filename and Path(mapped).resolve() != target:
                raise BackendUnavailable(f"another Orbbec SDK is already loaded: {mapped}; use a separate process")

    def _bind(self, name, restype, arguments, *, error_argument=True, optional=False):
        try:
            function = getattr(self._library, name)
        except AttributeError as error:
            if optional:
                return None
            raise BackendUnavailable(f"Orbbec SDK v1 is missing required function {name}") from error
        function.restype = restype
        function.argtypes = list(arguments) + ([ct.POINTER(ct.c_void_p)] if error_argument else [])
        if error_argument:
            self._functions[name] = function
        return function

    def _bind_functions(self):
        ptr, integer, uint, text = ct.c_void_p, ct.c_int, ct.c_uint32, ct.c_char_p
        self._error_message = self._bind("ob_error_message", text, (ptr,), error_argument=False)
        self._error_function = self._bind("ob_error_function", text, (ptr,), error_argument=False)
        self._delete_error = self._bind("ob_delete_error", None, (ptr,), error_argument=False)
        # Names/signatures follow the headers bundled with SDK v1.10.37.
        signatures = {
            "ob_create_context": (ptr, ()),
            "ob_create_context_with_config": (ptr, (text,)),
            "ob_query_device_list": (ptr, (ptr,)),
            "ob_delete_context": (None, (ptr,)),
            "ob_device_list_device_count": (uint, (ptr,)),
            "ob_device_list_get_device_pid": (integer, (ptr, uint)),
            "ob_device_list_get_device_vid": (integer, (ptr, uint)),
            "ob_device_list_get_device_serial_number": (text, (ptr, uint)),
            "ob_device_list_get_device_uid": (text, (ptr, uint)),
            "ob_device_list_get_device": (ptr, (ptr, uint)),
            "ob_device_list_get_device_by_serial_number": (ptr, (ptr, text)),
            "ob_device_list_get_device_by_uid": (ptr, (ptr, text)),
            "ob_delete_device_list": (None, (ptr,)),
            "ob_device_get_device_info": (ptr, (ptr,)),
            "ob_device_info_name": (text, (ptr,)),
            "ob_device_info_pid": (integer, (ptr,)),
            "ob_device_info_vid": (integer, (ptr,)),
            "ob_device_info_serial_number": (text, (ptr,)),
            "ob_device_info_firmware_version": (text, (ptr,)),
            "ob_device_info_connection_type": (text, (ptr,)),
            "ob_delete_device_info": (None, (ptr,)),
            "ob_delete_device": (None, (ptr,)),
            "ob_create_pipeline_with_device": (ptr, (ptr,)),
            "ob_pipeline_start_with_config": (None, (ptr, ptr)),
            "ob_pipeline_stop": (None, (ptr,)),
            "ob_delete_pipeline": (None, (ptr,)),
            "ob_pipeline_get_stream_profile_list": (ptr, (ptr, integer)),
            "ob_pipeline_wait_for_frameset": (ptr, (ptr, uint)),
            "ob_pipeline_enable_frame_sync": (None, (ptr,)),
            "ob_pipeline_disable_frame_sync": (None, (ptr,)),
            "ob_pipeline_get_camera_param": (OBCameraParam, (ptr,)),
            "ob_pipeline_get_camera_param_with_profile": (OBCameraParam, (ptr, uint, uint, uint, uint)),
            "ob_create_config": (ptr, ()),
            "ob_config_enable_stream": (None, (ptr, ptr)),
            "ob_config_set_align_mode": (None, (ptr, integer)),
            "ob_config_set_frame_aggregate_output_mode": (None, (ptr, integer)),
            "ob_delete_config": (None, (ptr,)),
            "ob_stream_profile_list_count": (uint, (ptr,)),
            "ob_stream_profile_list_get_profile": (ptr, (ptr, integer)),
            "ob_stream_profile_list_get_video_stream_profile": (ptr, (ptr, integer, integer, integer, integer)),
            "ob_stream_profile_format": (integer, (ptr,)),
            "ob_stream_profile_type": (integer, (ptr,)),
            "ob_video_stream_profile_width": (uint, (ptr,)),
            "ob_video_stream_profile_height": (uint, (ptr,)),
            "ob_video_stream_profile_fps": (uint, (ptr,)),
            "ob_stream_profile_get_extrinsic_to": (OBExtrinsic, (ptr, ptr)),
            "ob_delete_stream_profile": (None, (ptr,)),
            "ob_delete_stream_profile_list": (None, (ptr,)),
            "ob_frameset_depth_frame": (ptr, (ptr,)),
            "ob_frameset_color_frame": (ptr, (ptr,)),
            "ob_frame_data": (ptr, (ptr,)),
            "ob_frame_data_size": (uint, (ptr,)),
            "ob_frame_format": (integer, (ptr,)),
            "ob_video_frame_width": (uint, (ptr,)),
            "ob_video_frame_height": (uint, (ptr,)),
            "ob_frame_time_stamp": (ct.c_uint64, (ptr,)),
            "ob_frame_time_stamp_us": (ct.c_uint64, (ptr,)),
            "ob_frame_system_time_stamp": (ct.c_uint64, (ptr,)),
            "ob_frame_system_time_stamp_us": (ct.c_uint64, (ptr,)),
            "ob_frame_index": (ct.c_uint64, (ptr,)),
            "ob_depth_frame_get_value_scale": (ct.c_float, (ptr,)),
            "ob_frame_get_stream_profile": (ptr, (ptr,)),
            "ob_delete_frame": (None, (ptr,)),
            "ob_device_get_calibration_camera_param_list": (ptr, (ptr,)),
            "ob_camera_param_list_count": (uint, (ptr,)),
            "ob_camera_param_list_get_param": (OBCameraParam, (ptr, uint)),
            "ob_delete_camera_param_list": (None, (ptr,)),
        }
        for name, (result, arguments) in signatures.items():
            self._bind(name, result, arguments)
        optional = {
            # These are declared in the v1.10.37 headers but not exported by
            # its Linux binary.  A matched pipeline CameraParam remains usable.
            "ob_video_stream_get_intrinsic": (OBIntrinsic, (ptr,)),
            "ob_video_stream_get_distortion": (OBDistortion, (ptr,)),
            "ob_set_logger_severity": (None, (integer,)),
            "ob_device_is_property_supported": (ct.c_bool, (ptr, integer, integer)),
            "ob_device_get_bool_property": (ct.c_bool, (ptr, integer)),
            "ob_device_set_bool_property": (None, (ptr, integer, ct.c_bool)),
            "ob_device_get_bool_property_range": (OBBoolPropertyRange, (ptr, integer)),
            "ob_device_get_int_property": (ct.c_int32, (ptr, integer)),
            "ob_device_set_int_property": (None, (ptr, integer, ct.c_int32)),
            "ob_device_get_int_property_range": (OBIntPropertyRange, (ptr, integer)),
        }
        for name, (result, arguments) in optional.items():
            self._bind(name, result, arguments, optional=True)

    def has_function(self, name: str) -> bool:
        return name in self._functions

    def call(self, name: str, *arguments):
        """Call a bound C API and release its error object before raising."""
        function = self._functions.get(name)
        if function is None:
            raise BackendUnavailable(f"Orbbec SDK v1 function is unavailable: {name}")
        error = ct.c_void_p()
        try:
            result = function(*arguments, ct.byref(error))
        except (ct.ArgumentError, TypeError, ValueError) as failure:
            raise BackendError(f"Orbbec SDK v1 {name}: invalid C API arguments: {failure}") from failure
        if error.value:
            try:
                message = self._error_message(error) or b"unknown SDK error"
                origin = self._error_function(error) or b""
                detail = message.decode("utf-8", errors="replace")
                source = origin.decode("utf-8", errors="replace")
            finally:
                self._delete_error(error)
            raise BackendError(f"Orbbec SDK v1 {name}: {detail}" + (f" ({source})" if source else ""))
        return result

    def close_handle(self, name: str, handle) -> bool:
        """Release a non-null handle without preventing further cleanup calls."""
        if not handle:
            return True
        try:
            self.call(name, handle)
        except BackendError as failure:
            self.cleanup_errors.append(str(failure))
            return False
        return True

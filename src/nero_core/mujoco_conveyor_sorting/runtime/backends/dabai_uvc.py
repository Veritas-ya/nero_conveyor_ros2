"""Independent native UVC RGB observation for the Dabai DC1, without metric K.

The 1632x1224 device output is not assumed to be a native sensor array. It
has no matching depth/color factory calibration in this project's SDK v1.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import os
from pathlib import Path
import struct
import time
import warnings

from ..types import BackendError, BackendUnavailable
from .camera_lease import CameraLease

QUERYCAP = 0x80685600
QUERYCTRL = 0xc0445624
G_CTRL = 0xc008561b
S_CTRL = 0xc008561c
USB_RESET = 0x5514
CONTROL_IDS = {
    "brightness": 0x00980900, "contrast": 0x00980901,
    "gain": 0x00980913, "sharpness": 0x0098091b,
    "exposure_auto": 0x009a0901, "exposure_absolute": 0x009a0902,
    "focus_absolute": 0x009a090a, "focus_auto": 0x009a090c,
}


class UVCDiscoveryUnavailable(BackendUnavailable):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class DabaiUVCConfig:
    width: int = 1632
    height: int = 1224
    fps: int = 30
    device: str | Path | None = None

    def __post_init__(self):
        if any(isinstance(v, bool) or int(v) != v or v <= 0
               for v in (self.width, self.height, self.fps)):
            raise ValueError("UVC dimensions and fps must be positive integers")
        if self.device is not None and not str(self.device).startswith("/dev/"):
            raise ValueError("--device must be a Dabai /dev/ path, not a video index")


def _usb_identity(node: Path, sys_root: Path = Path("/sys/class/video4linux")):
    link = sys_root / node.resolve().name / "device"
    if not link.exists():
        return None
    current = link.resolve()
    for parent in (current, *current.parents):
        if (parent / "idVendor").is_file():
            info = {key: (parent / key).read_text().strip() if (parent / key).is_file() else None
                    for key in ("idVendor", "idProduct", "serial", "product", "busnum", "devnum")}
            info["usb_path"] = str(parent)
            return info
    return None


def _is_capture_node(node: Path):
    fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
    try:
        data = bytearray(104)
        fcntl.ioctl(fd, QUERYCAP, data)
        values = struct.unpack("16s32s32s6I", data)
        flags = values[5] if values[4] & 0x80000000 else values[4]
        return bool(flags & 1)  # V4L2_CAP_VIDEO_CAPTURE, excludes metadata nodes
    finally:
        os.close(fd)


def discover_uvc_device(device=None, *, sys_root=Path("/sys/class/video4linux"),
                        dev_root=Path("/dev")):
    """Select the one actual DC1 RGB capture interface; never an arbitrary camera."""
    nodes = [Path(device)] if device is not None else [
        dev_root / item.name for item in sorted(Path(sys_root).glob("video*"))]
    matches = []
    for node in nodes:
        info = _usb_identity(node, Path(sys_root))
        if info is None or (info["idVendor"], info["idProduct"]) != ("2bc5", "0557"):
            continue
        if _is_capture_node(node):
            matches.append((node, info))
    if len(matches) != 1:
        reason = "wrong_device" if device is not None else "not_visible" if not matches else "ambiguous"
        raise UVCDiscoveryUnavailable(reason,
            f"expected one Dabai DC1 RGB UVC interface (2bc5:0557), found {len(matches)}; "
            "close the RGB-D viewer first; no webcam/Gemini fallback")
    return matches[0]


def _recover_detached_uvc():
    """Recover a kernel-bound DC1 whose legacy SDK left no V4L capture node.

    Never reset an interface still detached/claimed by an SDK. Reset only the
    exact DC1 RGB USB device, with an already bound uvcvideo kernel driver.
    """
    found = []
    for usb in Path("/sys/bus/usb/devices").glob("*"):
        if ((usb / "idVendor").is_file() and (usb / "idProduct").is_file()
                and (usb / "idVendor").read_text().strip() == "2bc5"
                and (usb / "idProduct").read_text().strip() == "0557"):
            found.append(usb)
    if len(found) != 1:
        return False
    usb = found[0]
    driver = usb / (usb.name + ":1.0") / "driver"
    if not driver.exists() or driver.resolve().name != "uvcvideo":
        return False
    path = Path("/dev/bus/usb") / f"{int((usb / 'busnum').read_text()):03d}" / f"{int((usb / 'devnum').read_text()):03d}"
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.ioctl(fd, USB_RESET, 0)
    finally:
        os.close(fd)
    return True


class DabaiUVCCamera:
    def __init__(self, config=None):
        self.config = config or DabaiUVCConfig()
        self._capture = None
        self._fd = None
        self._originals = {}
        self.device_info = {}
        self._lease = CameraLease(label="高清 RGB 窗口")

    def start(self):
        if self._capture is not None:
            return
        import cv2
        self._lease.acquire()
        try:
            try:
                node, info = discover_uvc_device(self.config.device)
            except UVCDiscoveryUnavailable as error:
                if (self.config.device is not None or error.reason != "not_visible"
                        or not _recover_detached_uvc()):
                    raise
                deadline = time.monotonic() + 3.
                while True:
                    try:
                        node, info = discover_uvc_device(self.config.device)
                        break
                    except (BackendUnavailable, OSError) as pending:
                        # A recreated video node may appear before udev adds
                        # the active user's ACL. Only retry this short race.
                        if isinstance(pending, OSError) and pending.errno != errno.EACCES:
                            raise
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.05)
            self._fd = os.open(node, os.O_RDWR | os.O_NONBLOCK)
            self.device_info = dict(info, node=str(node), purpose="RGB observation only",
                                    metric_calibration_available=False,
                                    sensor_native_sampling="unverified")
            self._capture = cv2.VideoCapture(str(node), cv2.CAP_V4L2)
            if not self._capture.isOpened():
                raise BackendUnavailable("Dabai RGB interface is busy or cannot be opened; close other camera viewers")
            self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            image = self.read_bgr()
            fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
            actual_format = bytes((fourcc >> (8*i)) & 255 for i in range(4)).rstrip(b'\x00').decode('ascii', errors='replace')
            self.device_info.update(actual_width=image.shape[1], actual_height=image.shape[0],
                                    requested_fps=self.config.fps,
                                    driver_fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
                                    requested_format="MJPG", format=actual_format or "unknown")
        except Exception:
            self.stop()
            raise

    def read_bgr(self):
        if self._capture is None:
            raise BackendUnavailable("Dabai RGB camera is not connected")
        ok, image = self._capture.read()
        if not ok or image is None:
            raise BackendUnavailable("Dabai RGB capture failed; interface may be busy or disconnected")
        expected = (self.config.height, self.config.width, 3)
        if image.shape != expected:
            raise BackendUnavailable(f"requested native RGB {expected[:2][::-1]} but device delivered {image.shape}; no application resize")
        return image

    def _query_control(self, name):
        state = dict(status="unsupported", current=None, min=None, max=None, step=None,
                     default=None, flags=None, writable=False)
        if self._fd is None:
            return dict(state, status="disconnected")
        control = CONTROL_IDS[name]
        data = bytearray(68)
        struct.pack_into("I", data, 0, control)
        try:
            fcntl.ioctl(self._fd, QUERYCTRL, data)
            values = struct.unpack("II32siiiiIII", data)
            if values[7] & 1:  # V4L2_CTRL_FLAG_DISABLED
                return state
            current = bytearray(struct.pack("Ii", control, 0))
            fcntl.ioctl(self._fd, G_CTRL, current)
            state.update(status="read", current=struct.unpack("Ii", current)[1],
                         min=values[3], max=values[4], step=values[5], default=values[6],
                         flags=values[7], writable=not bool(values[7] & (4 | 16)))
        except OSError as error:
            if error.errno != errno.EINVAL:
                state.update(status="error", detail=str(error))
        return state

    def controls_metadata(self):
        return {name: self._query_control(name) for name in CONTROL_IDS}

    def set_control(self, name, value):
        state = self._query_control(name)
        if state["status"] != "read":
            raise BackendUnavailable(f"Dabai UVC {name} is unsupported")
        if isinstance(value, bool) or int(value) != value or not state["min"] <= value <= state["max"]:
            raise ValueError(f"{name} must be within {state['min']}..{state['max']}")
        if state["flags"] & 4:  # Read-only must not trigger a switch to manual.
            raise BackendUnavailable(f"Dabai UVC {name} is currently read-only/inactive")
        before = state["current"]
        saved_originals = self._originals.copy()
        auto = None
        attempted_write = False
        try:
            if name == "exposure_absolute":
                auto = self._query_control("exposure_auto")
                if auto["status"] == "read" and auto["current"] != 1:
                    self.set_control("exposure_auto", 1)  # V4L2_EXPOSURE_MANUAL
                    state = self._query_control(name)
            if not state["writable"]:
                raise BackendUnavailable(f"Dabai UVC {name} is currently read-only/inactive")
            self._originals.setdefault(name, before)
            attempted_write = True
            fcntl.ioctl(self._fd, S_CTRL, bytearray(struct.pack("Ii", CONTROL_IDS[name], int(value))))
            actual = self._query_control(name)
            if actual["status"] != "read" or actual["current"] != value:
                raise BackendError(f"Dabai UVC {name} did not accept {value}")
            return actual["current"]
        except Exception:
            # A rejected interactive adjustment must not silently leave AE off
            # or a partially applied manual value. Keep failed rollback values
            # for stop() to retry rather than claiming they were restored.
            restored = True
            rollback = [(name, before)] if attempted_write else []
            if auto is not None and auto["status"] == "read":
                rollback.append(("exposure_auto", auto["current"]))
            for key, old_value in rollback:
                try:
                    fcntl.ioctl(self._fd, S_CTRL, bytearray(struct.pack("Ii", CONTROL_IDS[key], old_value)))
                except OSError as error:
                    restored = False
                    self._originals.setdefault(key, old_value)
                    warnings.warn(f"could not roll back UVC {key}: {error}", RuntimeWarning)
            if restored:
                self._originals = saved_originals
            raise

    def adjust_control(self, name, steps):
        state = self._query_control(name)
        if state["status"] != "read":
            raise BackendUnavailable(f"Dabai UVC {name} is unsupported")
        if isinstance(steps, bool) or int(steps) != steps:
            raise ValueError("control adjustment must be an integer number of steps")
        value = max(state["min"], min(state["max"], state["current"] + int(steps) * max(1, state["step"])))
        return self.set_control(name, value)

    def stop(self):
        if self._fd is not None:
            # Restore manual values before restoring an original auto mode.
            names = sorted(self._originals, key=lambda name: name in ("exposure_auto", "focus_auto"))
            for name in names:
                try:
                    fcntl.ioctl(self._fd, S_CTRL, bytearray(struct.pack("Ii", CONTROL_IDS[name], self._originals[name])))
                except OSError as error:
                    warnings.warn(f"could not restore UVC {name}: {error}", RuntimeWarning)
            self._originals.clear()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._lease.release()

    close = stop

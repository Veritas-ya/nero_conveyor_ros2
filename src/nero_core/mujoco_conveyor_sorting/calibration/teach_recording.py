"""Read-only pose logging and conservative stationary pairing for teach recordings.

Camera and CAN are not hardware synchronized. Moving frames are archive-only;
calibration samples require continuous stationary feedback bracketing image receipt.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np

from .eye_to_hand import camera_parameters


def check_can_interface(channel, expected_bitrate=1000000):
    """Inspect SocketCAN without changing the interface or transmitting frames."""
    try:
        result = subprocess.run(["ip", "-details", "-json", "link", "show", "dev", channel],
                                capture_output=True, text=True, timeout=3)
    except FileNotFoundError as exc:
        raise RuntimeError("缺少 ip 命令，无法检查 CAN 接口，请安装 iproute2") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"检查 CAN 接口 {channel} 超时") from exc
    if result.returncode:
        raise RuntimeError(f"无法读取 CAN 接口 {channel}，检查 USB-CAN 连接和 --can-channel；"
                           f"{result.stderr.strip()}")
    try:
        interface, = json.loads(result.stdout)
        kind = interface.get("linkinfo", {}).get("info_kind")
        details = interface.get("linkinfo", {}).get("info_data", {})
        bitrate = details.get("bittiming", {}).get("bitrate")
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError("无法解析 ip 命令返回的 CAN 状态") from exc
    if kind != "can":
        raise RuntimeError(f"{channel} 不是 CAN 接口")
    quoted = shlex.quote(channel)
    if "UP" not in interface.get("flags", []):
        raise RuntimeError(f"CAN 接口 {channel} 处于 DOWN，尚未开始录像。请在终端执行：\n"
                           f"sudo ip link set dev {quoted} type can bitrate {expected_bitrate}\n"
                           f"sudo ip link set dev {quoted} up\n然后重新运行录制脚本。")
    if details.get("state") not in ("ERROR-ACTIVE", "ERROR-WARNING"):
        raise RuntimeError(f"CAN 接口 {channel} 状态为 {details.get('state', '未知')}，"
                           "请检查 CAN 接线、供电和总线状态后重试")
    if bitrate != expected_bitrate:
        raise RuntimeError(f"CAN 接口 {channel} 波特率为 {bitrate}，需要 {expected_bitrate}；"
                           "请停止使用该接口的程序后重新配置")
    return interface


def load_intrinsics(path):
    """Accept this workspace's intrinsics.json or OpenCV FileStorage YAML."""
    path = Path(path)
    serial = None
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        width, height = data["image_size"]
        matrix, distortion = data["camera_matrix"], data["dist_coeffs"]
        camera = data.get("camera", {})
        if camera.get("sensor", "color") != "color":
            raise ValueError("必须使用彩色相机内参")
        serial = camera.get("serial")
    else:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise ValueError(f"无法打开内参文件：{path}")
            width = storage.getNode("image_width").real()
            height = storage.getNode("image_height").real()
            matrix = storage.getNode("camera_matrix").mat()
            distortion = storage.getNode("dist_coeffs").mat()
            if matrix is None or distortion is None:
                raise ValueError("内参文件缺少 camera_matrix 或 dist_coeffs")
        finally:
            storage.release()
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("内参图像尺寸必须为正整数")
    if int(width) != width or int(height) != height:
        raise ValueError("内参图像尺寸必须为整数")
    result = dict(width=int(width), height=int(height), matrix=np.asarray(matrix).tolist(),
                  distortion=np.asarray(distortion).ravel().tolist(), distortion_model="opencv")
    camera_parameters(result)
    return result, serial


@dataclass
class TimedPose:
    sequence: int
    time_monotonic: float
    read_monotonic: float
    feedback_unix: tuple
    transform: np.ndarray

    def document(self):
        return {"sequence": self.sequence, "time_monotonic": self.time_monotonic,
                "read_monotonic": self.read_monotonic,
                "component_feedback_unix": list(self.feedback_unix),
                "gripper_to_base": self.transform.tolist()}


def color_frame_timing(frame, request_monotonic=None, *, max_delay_s=.25,
                       request_guard_s=.05):
    """Map the RGB SDK global estimate to host monotonic time without reanchoring.

    SDK global time is a software clock fit, not hardware synchronization. Its
    absolute delay and its stability across the capture window must both pass;
    a depth clock anchored at first receipt cannot establish RGB freshness.
    """
    metadata = getattr(frame, "timestamp_metadata", None)
    if not isinstance(metadata, dict) or metadata.get("selected_source") != "depth_device":
        raise ValueError("RGB pairing requires raw device timestamps")
    depth = metadata.get("depth") or {}
    color = metadata.get("color") or {}
    host = metadata.get("host_clock_reference") or {}
    try:
        depth_time = float(depth["device_time_s"])
        color_time = float(color["device_time_s"])
        global_time = float(color["global_time_s"])
        system_time = float(color["system_time_s"])
        frame_index_value = float(color["frame_index"])
        source_time = float(frame.header.source_time)
        received = float(frame.header.received_time)
        host_unix = float(host["unix_s"])
        host_monotonic = float(host["monotonic_s"])
        host_read_span = float(host["read_span_s"])
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ValueError("missing RGB global/system/device timestamps or host clock reference") from exc
    if not np.isfinite([depth_time, color_time, global_time, system_time, source_time,
                        received, host_unix, host_monotonic]).all() or min(
            depth_time, color_time, global_time, system_time, source_time,
            received, host_unix, host_monotonic) <= 0:
        raise ValueError("invalid RGB global/system/device timestamps or host clock reference")
    if not np.isfinite(frame_index_value) or frame_index_value < 0 or not frame_index_value.is_integer():
        raise ValueError("invalid RGB frame index")
    if not np.isfinite(host_read_span) or not 0 <= host_read_span <= .005:
        raise ValueError("host clock reference read span exceeds 5 ms")
    if not np.isfinite([max_delay_s, request_guard_s]).all() or max_delay_s <= 0 or request_guard_s < 0:
        raise ValueError("invalid RGB freshness limits")
    if abs(depth_time - source_time) > 1e-6:
        raise ValueError("depth timestamp does not match the mapped device clock")
    if global_time > system_time:
        raise ValueError("RGB global timestamp is later than RGB system receipt timestamp")
    offset = host_monotonic - host_unix
    color_sample = global_time + offset
    system_sample = system_time + offset
    # The only allowance is host clock read/rounding error. Do not clamp a fitted
    # timestamp to receipt: that would make an old/invalid estimate appear fresh.
    if system_sample > received + .001 or color_sample > received + .001:
        raise ValueError("RGB timestamp maps into the future")
    delay = received - color_sample
    if abs(delay) > max_delay_s:
        raise ValueError("buffered RGB frame exceeds pairing delay limit")
    if request_monotonic is not None:
        if not np.isfinite(request_monotonic):
            raise ValueError("invalid RGB capture request time")
        if color_sample < request_monotonic + request_guard_s:
            raise ValueError("RGB frame predates capture request guard; waiting for a fresh frame")
    return {"color_time_source": "color_sdk_global_with_host_clock_reference",
            "color_sample_monotonic_estimate": color_sample,
            "color_device_time_s": color_time,
            "color_depth_device_delta_s": color_time - depth_time,
            "color_global_time_s": global_time,
            "color_system_time_s": system_time,
            "color_frame_index": int(frame_index_value),
            "host_wall_to_monotonic_offset_s": offset,
            "host_clock_read_span_s": host_read_span,
            "color_delivery_delay_estimate_s": delay,
            "max_color_delivery_delay_estimate_s": max_delay_s,
            "capture_request_guard_s": request_guard_s,
            "hardware_synchronized": False,
            "time_estimate_limitation": "SDK global time is a software clock fit, not hardware "
                                       "synchronization; window stability checks cannot prove exposure timing."}


class ColorTimingHistory:
    """Keep invalid RGB timing as evidence while video logging continues."""
    def __init__(self):
        self.lock = Lock()
        self.samples = deque(maxlen=4000)

    def add(self, frame):
        document = {"camera_sequence": frame.header.sequence,
                    "received_monotonic": frame.header.received_time,
                    "valid": False, "error": None, "timing": None}
        try:
            document["timing"] = color_frame_timing(frame)
            document["valid"] = True
        except ValueError as exc:
            document["error"] = str(exc)
        with self.lock:
            self.samples.append(document)
        return document

    def validate_capture(self, frame, request_monotonic, hold_seconds=1.0, after_seconds=.25):
        """Require stable RGB clocks before and after the stationary pose window."""
        timing = color_frame_timing(frame, request_monotonic)
        if not np.isfinite([hold_seconds, after_seconds]).all() or min(hold_seconds, after_seconds) < 0:
            raise ValueError("invalid RGB clock window duration")
        start = timing["color_sample_monotonic_estimate"] - hold_seconds
        end = frame.header.received_time + after_seconds
        with self.lock:
            samples = list(self.samples)
        before = [i for i, record in enumerate(samples) if record["received_monotonic"] <= start]
        after = [i for i, record in enumerate(samples) if record["received_monotonic"] >= end]
        if not before or not after:
            raise ValueError("waiting for RGB clock window")
        window = samples[before[-1]:after[0] + 1]
        if not any(record["camera_sequence"] == frame.header.sequence and
                   record["received_monotonic"] == frame.header.received_time for record in window):
            raise ValueError("waiting for RGB clock window")
        if any(not record["valid"] for record in window):
            raise ValueError("invalid RGB timing inside capture clock window")
        received = np.array([record["received_monotonic"] for record in window])
        if len(window) < 3 or np.any(np.diff(received) <= 0) or np.max(np.diff(received)) > .5:
            raise ValueError("RGB receive gap inside capture clock window")
        for field in ("color_device_time_s", "color_global_time_s", "color_frame_index"):
            values = np.array([record["timing"][field] for record in window])
            if np.any(np.diff(values) <= 0):
                raise ValueError(f"RGB {field} repeated or reversed inside capture clock window")
        global_device_offsets = np.array([
            record["timing"]["color_global_time_s"] - record["timing"]["color_device_time_s"]
            for record in window])
        host_offsets = np.array([
            record["timing"]["host_wall_to_monotonic_offset_s"] for record in window])
        fit_span, host_span = float(np.ptp(global_device_offsets)), float(np.ptp(host_offsets))
        if fit_span > .010:
            raise ValueError("RGB SDK global clock fit changed inside capture clock window")
        if host_span > .005:
            raise ValueError("host clock offset changed inside capture clock window")
        timing.update({"rgb_clock_window_start_monotonic": float(received[0]),
                       "rgb_clock_window_end_monotonic": float(received[-1]),
                       "rgb_clock_window_frame_count": len(window),
                       "rgb_clock_window_max_received_gap_s": float(np.max(np.diff(received))),
                       "rgb_clock_window_global_device_offset_span_s": fit_span,
                       "rgb_clock_window_host_offset_span_s": host_span,
                       "rgb_clock_window_max_received_gap_limit_s": .5,
                       "rgb_clock_window_global_device_offset_span_limit_s": .010,
                       "rgb_clock_window_host_offset_span_limit_s": .005})
        return timing


class PoseHistory:
    def __init__(self):
        self.lock = Lock()
        self.samples = deque(maxlen=4000)

    def add(self, sample):
        with self.lock:
            self.samples.append(sample)

    def stationary_pair(self, image_received, hold_seconds=1.0, after_seconds=.25,
                        max_gap=.1, *, image_sample_time=None,
                        max_translation_m=.001, max_rotation_deg=.3):
        """Require samples before AND after the full time window; never extrapolate."""
        image_time = image_received if image_sample_time is None else image_sample_time
        start = min(image_time, image_received) - hold_seconds
        end = max(image_time, image_received) + after_seconds
        with self.lock:
            samples = list(self.samples)
        before = [i for i, p in enumerate(samples) if p.time_monotonic <= start]
        after = [i for i, p in enumerate(samples) if p.time_monotonic >= end]
        if not before or not after:
            raise ValueError("waiting for pose window")
        window = samples[before[-1]:after[0] + 1]
        stamps = np.array([p.time_monotonic for p in window])
        if len(window) < 3 or np.any(np.diff(stamps) <= 0) or np.max(np.diff(stamps)) > max_gap:
            raise ValueError("CAN feedback gap")
        transforms = np.asarray([p.transform for p in window])
        translations = transforms[:, :3, 3]
        translation_span = float(np.max(np.linalg.norm(
            translations[:, None] - translations[None, :], axis=2)))
        rotations = transforms[:, :3, :3]
        traces = np.einsum("aij,bij->ab", rotations, rotations)
        rotation_span = float(np.degrees(np.arccos(np.clip((np.min(traces) - 1.) / 2., -1., 1.))))
        if translation_span > max_translation_m or rotation_span > max_rotation_deg:
            raise ValueError(f"robot moved during capture: span {translation_span * 1000:.3f} mm / "
                             f"{rotation_span:.3f} deg; hold still and retry")
        closest = min(window, key=lambda p: abs(p.time_monotonic - image_time))
        timing = {"method": ("stationary_window_host_receipt" if image_sample_time is None else
                             "stationary_window_color_time_estimate_and_receipt"),
                  "image_received_monotonic": image_received,
                  "image_sample_monotonic_estimate": image_time,
                  "window_start_monotonic": float(stamps[0]),
                  "window_end_monotonic": float(stamps[-1]),
                  "pose_sequence": closest.sequence,
                  "pose_time_monotonic": closest.time_monotonic,
                  "pose_delta_from_image_receipt_s": closest.time_monotonic - image_received,
                  "pose_delta_from_image_sample_estimate_s": closest.time_monotonic - image_time,
                  "max_translation_span_m": translation_span,
                  "max_rotation_span_deg": rotation_span,
                  "max_translation_span_limit_m": max_translation_m,
                  "max_rotation_span_limit_deg": max_rotation_deg,
                  "max_pose_gap_s": float(np.max(np.diff(stamps))),
                  "pose_count": len(window)}
        return closest.transform.copy(), timing


def read_timed_pose(reader, sequence, offset, previous=None):
    """Take a coherent copy of three CAN pose components; detect host clock jumps."""
    try:
        pose, last = reader.read_snapshot()
    except AttributeError as exc:
        raise ValueError("waiting for complete CAN pose feedback") from exc
    if max(last) - min(last) > .05:
        raise ValueError("CAN pose component timestamps differ by > 50 ms")
    now_mono, now_wall = time.monotonic(), time.time()
    if abs((now_mono - now_wall) - offset) > .05:
        raise RuntimeError("系统时钟发生跳变，请结束本次录制后重新启动")
    if previous is not None and any(t <= p for t, p in zip(last, previous)):
        return None  # Do not turn repeated reads of stale packets into new measurements.
    return TimedPose(sequence, max(last) + offset, now_mono, last, pose)


class PoseLogger(Thread):
    def __init__(self, reader, output, poll_hz=100, *, read_sample=None):
        super().__init__(name="teach-pose-log", daemon=True)
        self.reader, self.output, self.poll_hz = reader, Path(output), poll_hz
        self.read_sample = read_sample
        self.history = PoseHistory()
        self.stop_event = Event()
        self.ready_event = Event()
        self.error = None
        self.count = 0

    def wait_until_ready(self, timeout_s=5.5):
        """Do not start video capture until a complete, fresh pose is logged."""
        if not self.ready_event.wait(timeout_s):
            raise RuntimeError("CAN 启动检查失败：没有完整的新鲜末端位姿；"
                               "请检查接口连接以及示教模式下的 CAN 位姿推送")
        if self.error:
            raise RuntimeError(f"CAN 启动检查失败：{self.error}") from self.error

    def run(self):
        try:
            offset = time.monotonic() - time.time()
            previous, last_valid, last_warning = None, time.monotonic(), 0.
            with self.output.open("x", encoding="utf-8", buffering=1) as log:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    try:
                        sample = (self.read_sample or read_timed_pose)(self.reader, self.count, offset, previous)
                        if sample is not None:
                            log.write(json.dumps(dict(type="pose", **sample.document()), allow_nan=False) + "\n")
                            self.history.add(sample)
                            previous, last_valid = sample.feedback_unix, started
                            self.count += 1
                            self.ready_event.set()
                    except ValueError as exc:
                        if started - last_warning > .5:
                            log.write(json.dumps({"type": "invalid_pose", "time_monotonic": started,
                                                  "reason": str(exc)}) + "\n")
                            last_warning = started
                    if started - last_valid > 5:
                        raise RuntimeError("连续 5 秒没有完整的新鲜末端位姿，检查 CAN 接口状态、接线和示教模式下的位姿推送")
                    self.stop_event.wait(max(0., 1 / self.poll_hz - (time.monotonic() - started)))
        except Exception as exc:
            self.error = exc
            self.ready_event.set()

    def close(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout=3)
            if self.is_alive():
                raise RuntimeError("CAN 日志线程未正常退出")


class VideoLogger(Thread):
    """Encode each consumed backend frame once; log sequence gaps, never duplicate frames."""
    def __init__(self, camera, output, size, fps):
        super().__init__(name="teach-video-log", daemon=True)
        self.camera, self.output, self.size, self.fps = camera, Path(output), tuple(size), fps
        self.stop_event = Event()
        self.error = None
        self.count = self.skipped = 0
        self.lock = Lock()
        self.latest = None
        self.color_timing_history = ColorTimingHistory()

    def snapshot(self):
        with self.lock:
            return self.latest

    def run(self):
        writer = None
        try:
            writer = cv2.VideoWriter(str(self.output / "color.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                     self.fps, self.size)
            if not writer.isOpened():
                raise RuntimeError("无法创建 MJPG AVI 视频，请检查 OpenCV 视频编码支持")
            last_sequence, epoch, previous_document = None, None, None
            with (self.output / "frames.jsonl").open("x", encoding="utf-8", buffering=1) as log:
                while not self.stop_event.is_set():
                    frame = self.camera.read_latest(timeout_s=2., after_sequence=last_sequence)
                    if frame.rgb is None or frame.rgb.shape[:2] != self.size[::-1]:
                        raise ValueError("彩色图像尺寸改变或缺失，停止录制")
                    if epoch is not None and frame.header.clock_epoch != epoch:
                        event = {
                            "type": "camera_clock_discontinuity",
                            "previous_frame": previous_document,
                            "rejected_frame": {
                                "camera_sequence": frame.header.sequence,
                                "camera_received_monotonic": frame.header.received_time,
                                "depth_source_time_s": frame.header.source_time,
                                "depth_sample_monotonic_estimate": frame.header.sample_time,
                                "clock_epoch": frame.header.clock_epoch,
                                "timestamp_metadata": getattr(frame, "timestamp_metadata", None),
                            },
                        }
                        with (self.output / "clock_events.jsonl").open("a", encoding="utf-8") as events:
                            events.write(json.dumps(event, allow_nan=False) + "\n")
                        raise RuntimeError("相机所选时间基准回退或改变，停止录制；"
                                           "触发帧已记录到 clock_events.jsonl")
                    epoch = frame.header.clock_epoch
                    gap = 0 if last_sequence is None else frame.header.sequence - last_sequence - 1
                    if gap < 0:
                        raise RuntimeError("相机帧序号倒退")
                    color_timing = self.color_timing_history.add(frame)
                    document = {"video_frame_index": self.count, "camera_sequence": frame.header.sequence,
                                "camera_received_monotonic": frame.header.received_time,
                                "depth_sample_monotonic_estimate": frame.header.sample_time,
                                "depth_source_time_s": frame.header.source_time,
                                "clock_epoch": epoch, "backend_frames_skipped_before": gap,
                                "timestamp_metadata": getattr(frame, "timestamp_metadata", None),
                                "color_timing": color_timing,
                                "write_monotonic": time.monotonic()}
                    writer.write(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                    log.write(json.dumps(document, allow_nan=False) + "\n")
                    self.skipped += gap
                    with self.lock:
                        self.latest = (self.count, frame)
                        self.count += 1
                    last_sequence = frame.header.sequence
                    previous_document = document
        except Exception as exc:
            self.error = exc
        finally:
            if writer is not None:
                writer.release()

    def close(self):
        self.stop_event.set()
        if self.ident is not None:
            self.join(timeout=3)
            if self.is_alive():
                raise RuntimeError("视频日志线程未正常退出")

"""Persistent, isolated Fast-FoundationStereo target-depth reconstruction.

This module deliberately imports neither torch nor a hardware backend. The
original RGB-D frame remains the caller's measurement; the returned depth is a
separate reconstruction of its same-exposure IR pair, never a new observation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any

import numpy as np

from .depth_diagnostics import measured_stage, trace_context


PROJECT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / '.cache' / 'nero-stereo' / 'fast'
REPLY_PREFIX = 'FAST_STEREO_REPLY '


class FastStereoUnavailable(RuntimeError):
    """The caller can retain native depth without calling it a Fast result."""


@dataclass(frozen=True)
class FastStereoResult:
    reconstructed_depth_m: np.ndarray
    depth_crop_m: np.ndarray
    disparity_px: np.ndarray
    crop_xyxy: tuple[int, int, int, int]
    K_crop: np.ndarray
    timings_s: dict[str, float]
    provenance: dict[str, Any]
    artifact_prefix: str


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=_json_default), encoding='utf-8')


def _validate_inputs(left_ir, right_ir, K, baseline_m, crop_xyxy):
    left, right = np.asarray(left_ir), np.asarray(right_ir)
    if left.ndim != 2 or right.shape != left.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('Fast输入须为同尺寸的左右Y8红外二维图像')
    matrix = np.asarray(K, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0., 0., 1.])):
        raise ValueError('Fast输入须有有效的3×3红外相机内参')
    baseline = float(baseline_m)
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Fast双目基线须为大于0的米值')
    crop = np.asarray(crop_xyxy)
    if (crop.shape != (4,) or not np.issubdtype(crop.dtype, np.number)
            or not np.isfinite(crop).all() or not np.array_equal(crop, crop.astype(int))):
        raise ValueError('Fast裁框须为四个整数像素坐标')
    x0, y0, x1, y1 = map(int, crop)
    height, width = left.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Fast裁框超出左右红外原图范围')
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError('Fast裁框长宽至少32像素')
    cropped_K = matrix.copy()
    cropped_K[0, 2] -= x0
    cropped_K[1, 2] -= y0
    # Freeze the inputs before handing files to another process. No striding,
    # resizing, contrast stretching or completion is performed here.
    return (left[y0:y1, x0:x1].copy(), right[y0:y1, x0:x1].copy(),
            cropped_K, baseline, (x0, y0, x1, y1), left.shape)


class FastStereoProcessBackend:
    """Keep the official CUDA model in its own Python process between calls.

    Call ``start`` and ``warmup`` before target recognition. ``refine`` blocks
    only the requesting perception code; it owns no camera or control thread.
    Every request preserves its measured inputs, raw disparity and timing.
    """
    def __init__(self, output_directory, *, python_executable=None, repository=None,
                 checkpoint=None, startup_timeout_s=120., request_timeout_s=30.,
                 compiler_include_dirs=None):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.python_executable = Path(python_executable or CACHE / 'env/bin/python').absolute()
        self.repository = Path(repository or CACHE / 'repo').resolve()
        self.checkpoint = Path(checkpoint or CACHE / 'research_weights/23-36-37/model_best_bp2_serialize.pth').resolve()
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        if self.startup_timeout_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError('Fast进程等待时间须大于0秒')
        if compiler_include_dirs is None:
            # Already downloaded headers used by the validated local CUDA
            # environment. Adding include paths changes no system package.
            candidates = [Path.home() / '.cache/nero-stereo/python-dev/usr/include',
                          Path.home() / '.cache/nero-planning/python-dev/usr/include']
            compiler_include_dirs = [p / 'python3.10' for p in candidates if (p / 'python3.10/Python.h').is_file()]
            compiler_include_dirs += [p for p in candidates if (p / 'python3.10/Python.h').is_file()]
        self.compiler_include_dirs = [str(Path(p)) for p in compiler_include_dirs]
        self.process = None
        self.provenance = None
        self._lock = threading.RLock()
        self._replies = queue.Queue()
        self._reader = None
        self._log = None
        self._calls = 0
        self._completed_refinements = 0
        self._serial = 0
        self._warmed_shapes = set()

    def _read_output(self, stream, replies, log):
        try:
            for line in stream:
                log.write(line)
                log.flush()
                if line.startswith(REPLY_PREFIX):
                    try:
                        replies.put(json.loads(line[len(REPLY_PREFIX):]))
                    except (ValueError, TypeError) as error:
                        replies.put(FastStereoUnavailable(f'Fast进程返回无效JSON：{error}'))
        except (ValueError, OSError) as error:
            replies.put(FastStereoUnavailable(f'Fast进程输出读取失败：{error}'))
        finally:
            replies.put(None)

    def _receive(self, operation, request_id, timeout_s):
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty as error:
            raise FastStereoUnavailable(f'Fast进程{operation}超过{timeout_s:g}秒；日志 {self.output_directory / "worker.log"}') from error
        if reply is None:
            code = self.process.poll() if self.process is not None else None
            raise FastStereoUnavailable(f'Fast进程退出：{code}；日志 {self.output_directory / "worker.log"}')
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, dict):
            raise FastStereoUnavailable('Fast进程返回的消息不是JSON对象')
        if reply.get('operation') != operation or reply.get('request_id') != request_id:
            raise FastStereoUnavailable('Fast进程回复与本次请求不一致')
        if reply.get('error'):
            raise FastStereoUnavailable('Fast-FoundationStereo：' + str(reply['error']))
        return reply

    def _request(self, operation, **fields):
        self._serial += 1
        request = dict(operation=operation, request_id=self._serial, **fields)
        try:
            self.process.stdin.write(json.dumps(request, default=_json_default) + '\n')
            self.process.stdin.flush()
            return self._receive(operation, self._serial, self.request_timeout_s)
        except (FastStereoUnavailable, OSError, ValueError) as error:
            # After a timeout or malformed reply, do not let a late result be
            # mistaken for another capture. Close this process before fallback.
            self.close()
            if isinstance(error, FastStereoUnavailable):
                raise
            raise FastStereoUnavailable(f'Fast进程请求失败：{error}') from error

    def start(self):
        with self._lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise FastStereoUnavailable('Fast进程已退出')
                return self
            if not self.python_executable.is_file():
                raise FastStereoUnavailable(f'缺少Fast隔离Python环境：{self.python_executable}')
            configuration = {'repository': str(self.repository), 'checkpoint': str(self.checkpoint),
                             'valid_iters': 8, 'max_disp': 192, 'seed': 0,
                             'precision': 'float16', 'build_volume': 'pytorch1'}
            config_path = self.output_directory / 'worker_configuration.json'
            _write_json(config_path, configuration)
            command = [str(self.python_executable), '-u', '-m',
                       'mujoco_conveyor_sorting.runtime._fast_stereo_worker',
                       '--configuration', str(config_path)]
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(PROJECT) + os.pathsep + environment.get('PYTHONPATH', '')
            if self.compiler_include_dirs:
                environment['CPATH'] = os.pathsep.join(self.compiler_include_dirs +
                                                       ([environment['CPATH']] if environment.get('CPATH') else []))
            self._log = (self.output_directory / 'worker.log').open('a', encoding='utf-8')
            self._replies = queue.Queue()
            try:
                self.process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                                start_new_session=True)
                self._reader = threading.Thread(target=self._read_output,
                                                args=(self.process.stdout, self._replies, self._log),
                                                name='fast-stereo-output', daemon=True)
                self._reader.start()
                self.provenance = self._receive('ready', None, self.startup_timeout_s)['provenance']
                _write_json(self.output_directory / 'backend.json', {
                    'command': command, 'compiler_include_dirs': self.compiler_include_dirs,
                    'network': self.provenance, 'hardware_connections': False})
            except Exception as error:
                self.close()
                if isinstance(error, FastStereoUnavailable):
                    raise
                raise FastStereoUnavailable(f'Fast进程启动失败：{error}') from error
            except BaseException:
                self.close()
                raise
            return self

    def warmup(self, shape=(384, 640)):
        shape = tuple(map(int, shape))
        if len(shape) != 2 or min(shape) < 32:
            raise ValueError('Fast预热尺寸须为至少32像素的高宽')
        with self._lock:
            self.start()
            if shape in self._warmed_shapes:
                return dict(self.health(), already_warmed=True)
            reply = self._request('warmup', shape=list(shape))
            self._warmed_shapes.add(shape)
            _write_json(self.output_directory / f'warmup_{shape[0]}x{shape[1]}.json', reply)
            return reply

    def health(self):
        with self._lock:
            running = self.process is not None and self.process.poll() is None
            return {'ready': bool(running and self.provenance is not None),
                    'warmed': bool(running and self._warmed_shapes),
                    'warmed_shapes': [list(x) for x in sorted(self._warmed_shapes)],
                    'pid': self.process.pid if running else None,
                    'completed_refinements': self._completed_refinements}

    def refine(self, left_ir, right_ir, K, baseline_m, crop_xyxy, *, capture_metadata=None):
        spans, parent_timings = [], {}
        trace = trace_context()
        trace['spans'] = spans
        trace['capture_metadata'] = capture_metadata

        def stage(name):
            return measured_stage('fast_parent_' + name, parent_timings, events=spans)

        with stage('validate_and_copy_inputs'):
            left, right, cropped_K, baseline, crop, full_shape = _validate_inputs(
                left_ir, right_ir, K, baseline_m, crop_xyxy)
            frozen_metadata = json.loads(json.dumps(capture_metadata, default=_json_default))
            trace['capture_metadata'] = frozen_metadata
        # Measure acquisition itself separately from the work under this lock.
        with stage('lock_acquire'):
            self._lock.acquire()
        try:
            with stage('ensure_started'):
                self.start()
            started = time.monotonic()
            self._calls += 1
            stem = self.output_directory / f'call_{self._calls:04d}'
            input_path = stem.with_suffix('.input.npz')
            output_path = stem.with_suffix('.output.npz')
            with stage('input_npz_write'):
                np.savez(input_path, left_ir=left, right_ir=right,
                         K_crop=cropped_K, baseline_m=np.asarray(baseline))
            request_metadata = {'crop_xyxy': list(crop), 'full_shape': list(full_shape),
                                'capture_metadata': frozen_metadata,
                                'input': str(input_path), 'output': str(output_path)}
            with stage('request_json_write'):
                _write_json(stem.with_suffix('.request.json'), request_metadata)
            with stage('request_and_reply'):
                reply = self._request('refine', **request_metadata)
            try:
                with stage('output_npz_read_copy'):
                    with np.load(output_path, allow_pickle=False) as data:
                        depth, disparity = data['depth_m'].copy(), data['disparity_px'].copy()
                    if depth.shape != left.shape or disparity.shape != left.shape:
                        raise ValueError('重算结果尺寸与原生裁框不一致')
                # No hidden surfaces or pixels outside the request are filled.
                with stage('full_depth_allocate_copy'):
                    reconstructed = np.full(full_shape, np.nan, dtype=np.float32)
                    x0, y0, x1, y1 = crop
                    reconstructed[y0:y1, x0:x1] = depth
                timings = dict(reply['timings_s'])
                timings['parent_request_to_result_s'] = time.monotonic() - started
                provenance = dict(self.provenance, **reply.get('provenance', {}),
                                  source='fast_foundationstereo', crop_xyxy=list(crop),
                                  K_crop=cropped_K.tolist(), baseline_m=baseline,
                                  capture_metadata=frozen_metadata,
                                  parent_diagnostic_trace=trace,
                                  new_observation=False, input_resized=False,
                                  hidden_surface_completion=False,
                                  left_right_threshold_applied=False)
                with stage('result_json_write'):
                    _write_json(stem.with_suffix('.result.json'), {
                        'timings_s': dict(timings, **parent_timings), 'provenance': provenance})
                timings.update(parent_timings)
                self._completed_refinements += 1
                return FastStereoResult(reconstructed, depth, disparity, crop,
                                        cropped_K, timings, provenance, str(stem))
            except (OSError, ValueError, KeyError) as error:
                self.close()
                raise FastStereoUnavailable(f'Fast结果不可用：{error}') from error
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(json.dumps({'operation': 'close'}) + '\n')
                        process.stdin.flush()
                        process.wait(timeout=3.)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=2.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2.)
                if process.stdin is not None:
                    process.stdin.close()
                if self._reader is not None:
                    self._reader.join(timeout=2.)
                if process.stdout is not None:
                    process.stdout.close()
            self._reader = None
            self._warmed_shapes.clear()
            self.provenance = None
            if self._log is not None:
                self._log.close()
                self._log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

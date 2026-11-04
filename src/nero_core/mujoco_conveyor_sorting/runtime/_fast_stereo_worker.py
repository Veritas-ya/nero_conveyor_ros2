"""CUDA-only child for :mod:`fast_stereo`; no hardware imports or access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

REPLY_PREFIX = 'FAST_STEREO_REPLY '


def _reply(operation, request_id=None, **fields):
    print(REPLY_PREFIX + json.dumps(dict(operation=operation, request_id=request_id,
                                        **fields), ensure_ascii=False), flush=True)


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class _Model:
    def __init__(self, configuration):
        started = time.monotonic()
        repository = Path(configuration['repository']).resolve()
        checkpoint = Path(configuration['checkpoint']).resolve()
        if not (repository / 'core/foundation_stereo.py').is_file():
            raise FileNotFoundError(f'缺少Fast官方源码：{repository}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'缺少Fast官方权重：{checkpoint}')
        sys.path.insert(0, str(repository))
        import torch
        import core.foundation_stereo  # Classes required by official serialized checkpoint.
        from core.utils.utils import InputPadder

        self.torch, self.InputPadder = torch, InputPadder
        np.random.seed(configuration['seed'])
        torch.manual_seed(configuration['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.valid_iters = int(configuration['valid_iters'])
        self.max_disp = int(configuration['max_disp'])
        load_started = time.monotonic()
        self.model = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.args.valid_iters = self.valid_iters
        self.model.args.max_disp = self.max_disp
        self.model.cuda().eval()
        torch.cuda.synchronize()
        load_elapsed = time.monotonic() - load_started
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository,
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
        self.provenance = {
            'model': 'Fast-FoundationStereo', 'checkpoint': str(checkpoint),
            'checkpoint_sha256': _sha256(checkpoint), 'repository': str(repository),
            'repository_commit': commit,
            'repository_model_source_sha256': _sha256(repository / 'core/foundation_stereo.py'),
            'worker_source_sha256': _sha256(__file__), 'python_executable': sys.executable,
            'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'seed': configuration['seed'],
            'valid_iters': self.valid_iters, 'max_disp': self.max_disp,
            'precision': 'float16_autocast', 'build_volume': 'pytorch1',
            'model_load_to_cuda_s': load_elapsed,
            'worker_initialization_s': time.monotonic() - started,
            'hardware_connections': False}
        self.forward_calls = 0

    def infer(self, left_ir, right_ir):
        torch = self.torch
        started = time.monotonic()
        left_np = np.repeat(left_ir[..., None], 3, axis=2)
        right_np = np.repeat(right_ir[..., None], 3, axis=2)
        left = torch.from_numpy(left_np).cuda().float()[None].permute(0, 3, 1, 2)
        right = torch.from_numpy(right_np).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        torch.cuda.synchronize()
        prepared = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            disparity = self.model.forward(left, right, iters=self.valid_iters,
                                           test_mode=True, optimize_build_volume='pytorch1')
        torch.cuda.synchronize()
        completed = time.monotonic()
        peak_mib = torch.cuda.max_memory_allocated() / 1048576.
        disparity = padder.unpad(disparity.float()).detach().cpu().numpy().reshape(left_ir.shape)
        ended = time.monotonic()
        self.forward_calls += 1
        return disparity, {
            'prepare_upload_pad_s': prepared - started,
            'forward_cuda_synchronized_s': completed - prepared,
            'output_download_s': ended - completed,
            'input_to_numpy_output_s': ended - started,
        }, {'forward_call': self.forward_calls, 'peak_allocated_mib': peak_mib,
            'diagnostic_interval': {'pid': os.getpid(), 'native_thread_id': threading.get_native_id(),
                'monotonic_start_s': started, 'cuda_prepared_s': prepared,
                'cuda_completed_s': completed, 'numpy_output_s': ended,
                'clock': 'time.monotonic; CUDA boundaries already synchronized by original implementation'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', required=True)
    arguments = parser.parse_args()
    try:
        configuration = json.loads(Path(arguments.configuration).read_text(encoding='utf-8'))
        model = _Model(configuration)
        _reply('ready', provenance=model.provenance)
    except Exception as error:
        traceback.print_exc()
        _reply('ready', error=f'{type(error).__name__}: {error}')
        return 1
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            operation = request['operation']
            if operation == 'close':
                return 0
            if operation == 'warmup':
                height, width = map(int, request['shape'])
                _, timings, provenance = model.infer(np.zeros((height, width), dtype=np.uint8),
                                                     np.zeros((height, width), dtype=np.uint8))
                _reply(operation, request['request_id'], timings_s=timings, provenance=provenance)
            elif operation == 'refine':
                started = time.monotonic()
                with np.load(request['input'], allow_pickle=False) as data:
                    left, right = data['left_ir'], data['right_ir']
                    cropped_K, baseline = data['K_crop'], float(data['baseline_m'])
                disparity, timings, provenance = model.infer(left, right)
                depth = np.full(disparity.shape, np.nan, dtype=np.float32)
                valid = np.isfinite(disparity) & (disparity > 0)
                depth[valid] = cropped_K[0, 0] * baseline / disparity[valid]
                # Save the raw prediction; do not synthesize hidden surfaces,
                # reject by LR residual, or adjust height to a support plane.
                np.savez(request['output'], depth_m=depth, disparity_px=disparity)
                timings['worker_request_to_output_s'] = time.monotonic() - started
                _reply(operation, request['request_id'], timings_s=timings,
                       provenance=provenance)
            else:
                raise ValueError(f'未知Fast请求：{operation}')
        except Exception as error:
            traceback.print_exc()
            _reply(request.get('operation'), request.get('request_id'),
                   error=f'{type(error).__name__}: {error}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

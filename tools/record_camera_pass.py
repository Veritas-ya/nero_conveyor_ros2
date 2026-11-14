#!/usr/bin/env python3
"""Record a short ordered RGB-D pass; no robot or CAN access."""
import argparse
import json
from pathlib import Path
import time

from nero_core.two_second.config import CellConfig
from nero_core.two_second.rgbd_archive import RGBDArchive
from nero_core.two_second.diagnostics import json_value
from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=18.)
    args = p.parse_args()
    if not 0 < args.seconds <= 60:
        p.error('seconds must be greater than zero and at most 60')
    if args.output.exists():
        p.error('output directory already exists; choose a new one')
    config = CellConfig.load(args.config)
    cal = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    settings = dict(config.raw['camera'])
    settings.update(enable_color=True, require_color=True, timestamp_source='device',
        serial_number=cal['serial_number'], calibration_id=cal['calibration_id'],
        camera_to_world_cv=cal['depth_to_base_cv'])
    camera = OrbbecCameraBackend(settings)
    archive = None
    report = {'mode': 'camera_only_ordered_rgbd', 'config': str(args.config.resolve()),
        'config_sha256': config.source_sha256, 'camera_connections': 0,
        'robot_connections': 0, 'robot_commands': 0, 'requested_seconds': args.seconds,
        'status': 'starting'}
    args.output.mkdir(parents=True)
    try:
        camera.start()
        report['camera_connections'] = 1
        camera.read_latest(timeout_s=3., wait_for_fresh=True)
        d2c = camera.get_depth_to_color_cv()
        archive = RGBDArchive(args.output, rate_hz=1000.)
        sequence = None
        start = time.monotonic()
        print('CAPTURE_STARTED', flush=True)
        while time.monotonic() - start < args.seconds:
            frame = camera.read_next(timeout_s=.5, after_sequence=sequence, wait_for_fresh=True)
            sequence = frame.header.sequence
            archive.offer_frame(frame, d2c)
        report['status'] = 'completed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        if archive is not None:
            report['archive'] = archive.close()
        try:
            report['camera'] = camera.get_capture_diagnostics()
        finally:
            camera.close()
        (args.output / 'capture_report.json').write_text(
            json.dumps(json_value(report), ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(json_value(report), ensure_ascii=False), flush=True)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Display one live Orbbec RGB-D stream; does not connect to CAN or command a robot."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    path = Path(raw['camera_calibration_file'])
    calibration = load_runtime_depth_calibration(path if path.is_absolute() else args.config.parent/path)
    camera_config = dict(raw['camera'])
    camera_config.update(enable_color=True, require_color=True, timestamp_source='device',
        serial_number=calibration['serial_number'], calibration_id=calibration['calibration_id'],
        camera_to_world_cv=calibration['depth_to_base_cv'])
    camera = OrbbecCameraBackend(camera_config)
    title = 'Gemini 335L - Live RGB / Depth - Q or Esc to close'
    try:
        camera.start()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 1280, 520)
        last_sequence = None
        print('Camera preview opened; press Q or Esc to close.', flush=True)
        while True:
            frame = camera.read_latest(timeout_s=3, after_sequence=last_sequence, wait_for_fresh=True)
            last_sequence = frame.header.sequence
            rgb = cv2.resize(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), (640, 400), interpolation=cv2.INTER_AREA)
            depth = frame.depth_m
            valid = np.isfinite(depth) & (depth > 0)
            display = np.zeros(depth.shape, np.uint8)
            display[valid] = (np.clip((depth[valid]-.17)/(3.-.17), 0, 1)*255).astype(np.uint8)
            colored = cv2.applyColorMap(display, cv2.COLORMAP_TURBO)
            colored[~valid] = 0
            colored = cv2.resize(colored, (640, 400), interpolation=cv2.INTER_NEAREST)
            canvas = np.full((460, 1280, 3), 24, np.uint8)
            canvas[35:435,:640], canvas[35:435,640:] = rgb, colored
            cv2.putText(canvas, 'RGB - unmodified camera image', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
            cv2.putText(canvas, f'Native depth 0.17-3.0 m - valid {valid.mean():.1%}', (650, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 1)
            cv2.putText(canvas, 'Views are not pixel-aligned. Q / Esc: close.', (10, 455), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q'), ord('Q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

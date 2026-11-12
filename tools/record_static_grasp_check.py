#!/usr/bin/env python3
"""Record one stationary RGB-D and passive robot snapshot; never send CAN commands."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration, load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
from mujoco_conveyor_sorting.runtime.single_pick import detect_target
from nero_core.two_second.collision_capture import frame_metadata
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.measured_rgbd import depth_capture_timing, fit_plane, green_belt_points
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.vision import obb_grasps


def save_frame(output, label, frame, d2c, config, before, after):
    """Persist evidence first; detection errors cannot discard the capture."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / label
    paths = [stem.with_suffix(s) for s in ('.npz', '.json', '.png')]
    if any(p.exists() for p in paths):
        raise FileExistsError(f'{label} 已存在；请使用新的输出目录或标签')
    timing = depth_capture_timing(frame)
    pose_before = np.asarray(before['pose'], dtype=float)
    nominal_tcp = pose_xyz_rpy(pose_before) @ config.tcp
    q_before, q_after = np.asarray(before['joints']), np.asarray(after['joints'])
    report = {
        'mode': 'stationary_camera_and_passive_robot_feedback',
        'label': label, 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'configuration': str(config.source), 'configuration_sha256': config.source_sha256,
        'base_frame': config.raw['base_frame'], 'camera': frame_metadata(frame, d2c),
        'depth_capture_timing': timing, 'feedback_before': before, 'feedback_after': after,
        'nominal_tcp_base_matrix': nominal_tcp, 'nominal_tcp_source': config.raw['tcp_source'],
        'nominal_tcp_is_not_independent_finger_center_measurement': True,
        'joint_change_between_feedback_samples_deg': np.rad2deg(q_after-q_before),
        'max_abs_joint_change_deg': float(np.rad2deg(np.max(np.abs(q_after-q_before)))),
        'robot_commands_sent': 0, 'automatic_enable': False,
        'independent_finger_center_status': 'not_measured_requires_visible_markers_or_physical_measurement',
        'detection_status': 'pending',
    }
    arrays = {'frame_rgb': frame.rgb, 'frame_depth_m': frame.depth_m,
              'metadata': np.asarray(json.dumps(json_value(report['camera']), ensure_ascii=False))}
    # The first write always contains the original exposure; enrich it only on success.
    np.savez_compressed(paths[0], **arrays)
    if not cv2.imwrite(str(paths[2]), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
        raise OSError('无法保存 RGB 图片')
    try:
        model = calibrated_model(config)
        model.check_feedback(before)
        detection = detect_target(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points,
            target_shape='simple', robot_model=model, robot_state=before,
            maximum_target_height_m=config.raw['maximum_target_height_m'])
        grasps, box = obb_grasps(detection['object_points'], detection['belt_region']['plane_to_base'],
                                  config.canonical_tcp)
        arrays.update(object_points_base_m=detection['object_points'], grasps_base=np.asarray(grasps))
        # Avoid replacing the only raw copy until the enriched archive is complete.
        temp = output / (label + '.tmp.npz')
        np.savez_compressed(temp, **arrays)
        temp.replace(paths[0])
        report.update(detection_status='success', target_points=len(detection['object_points']),
            obb=box, grasp_candidates_base=grasps,
            predicted_grasp_point_base_m=np.asarray(grasps[0])[:3, 3],
            detection_note='同一静止曝光的可见点云 OBB 中心，不包含移动目标预测')
    except Exception as exc:
        report.update(detection_status='failed', detection_error_type=type(exc).__name__,
                      detection_error=str(exc), detection_error_details=getattr(exc, 'details', {}))
    paths[1].write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return report, paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pose', choices=('normal', 'rot90'), required=True)
    p.add_argument('--warmup-frames', type=int, default=12)
    args = p.parse_args(argv)
    if args.warmup_frames < 1:
        p.error('--warmup-frames 必须为正整数')
    for suffix in ('.npz', '.json', '.png'):
        if (args.output / (args.pose + suffix)).exists():
            p.error('输出文件已存在，请选择新目录')
    config = CellConfig.load(args.config)
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    cfg = dict(config.raw['camera'])
    cfg.update(enable_color=True, require_color=True, timestamp_source='device',
        serial_number=calibration['serial_number'], calibration_id=calibration['calibration_id'],
        camera_to_world_cv=calibration['depth_to_base_cv'])
    device = Hardware(config.raw['hardware']['can_channel'], config.raw['hardware']['firmware'], False,
                      config.raw['hardware']['joint_receive_offset_s'])
    camera = OrbbecCameraBackend(cfg)
    device_started = camera_started = False
    try:
        device.start(); device_started = True
        camera.start(); camera_started = True
        d2c = camera.get_depth_to_color_cv()
        seq = None
        for _ in range(args.warmup_frames):
            frame = camera.read_next(timeout_s=3., after_sequence=seq, wait_for_fresh=True)
            seq = frame.header.sequence
        before = device.snapshot()
        frame = camera.read_next(timeout_s=3., after_sequence=seq, wait_for_fresh=True)
        frame = apply_runtime_depth_calibration(frame, calibration, d2c)
        after = device.snapshot()
        report, paths = save_frame(args.output, args.pose, frame, d2c, config, before, after)
    finally:
        if camera_started: camera.close()
        if device_started: device.close()
    print(f'[采集] {args.pose} 已保存；CAN 只读，机械臂/夹爪命令 0')
    print(f'[反馈] 采集前后最大关节角变化 {report["max_abs_joint_change_deg"]:.3f}°')
    print(f'[文件] {paths[0]}  {paths[1]}  {paths[2]}')
    if report['detection_status'] == 'success':
        print('[点云] 已计算可见 OBB 抓取中心；实际指尖中心仍需独立标注或测量')
    else:
        print(f'[点云] 自动检测未成功：{report["detection_error"]}；原始 RGB-D 已保存')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

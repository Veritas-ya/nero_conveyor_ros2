#!/usr/bin/env python3
"""Inspect a running camera's shared RGB-D belt plane; no device connections."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nero_core.two_second.collision_view import restore_frame, project_base
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import (
    apply_runtime_depth_calibration, load_runtime_depth_calibration)


def inspect(path, config, calibration):
    # The producer replaces this file atomically. Retain this exact exposure.
    with np.load(path, allow_pickle=False) as packet:
        arrays = {key: packet[key].copy() for key in packet.files}
    meta = json.loads(str(arrays['metadata']))
    frame = restore_frame(meta, arrays)
    d2c = np.asarray(meta['depth_to_color_cv'])
    frame = apply_runtime_depth_calibration(frame, calibration, d2c)
    surface = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane,
                                    green_belt_points=green_belt_points)
    world, mask, green = green_belt_points(frame, d2c)
    points = world[mask]
    normal = surface.normal_base
    residual = (points-surface.plane_to_base[:3, 3]) @ normal
    inliers = np.abs(residual) < .004
    abc = np.asarray(surface.plane['abc'])
    direction = config.direction.copy()
    # Preserve configured XY heading while expressing its height on this plane.
    direction[2] = abc[:2] @ direction[:2]
    direction /= np.linalg.norm(direction)
    overlay = frame.rgb.copy()
    contours, _ = cv2.findContours(np.uint8(green), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (70, 255, 90), 2)
    flat = (points-surface.plane_to_base[:3, 3]) @ surface.plane_to_base[:3, :3]
    lo, hi = np.quantile(flat[inliers, :2], [.02, .98], axis=0)

    def draw_base_line(a, b, color, width=2):
        # Sample distortion along each projected line.
        samples = np.linspace(a, b, 45)
        uv, visible = project_base(samples, frame, d2c)
        if visible.all():
            cv2.polylines(overlay, [np.rint(uv).astype(np.int32)], False, color, width)

    def base(x, y):
        return surface.plane_to_base[:3, :3] @ np.array([x, y, 0.]) + surface.plane_to_base[:3, 3]

    for x in np.arange(np.ceil(lo[0]/.1)*.1, hi[0], .1):
        draw_base_line(base(x, lo[1]), base(x, hi[1]), (0, 225, 255), 1)
    for y in np.arange(np.ceil(lo[1]/.1)*.1, hi[1], .1):
        draw_base_line(base(lo[0], y), base(hi[0], y), (0, 225, 255), 1)

    selected = np.linspace(0, len(points)-1, min(2200, len(points)), dtype=int)
    uv, visible = project_base(points[selected], frame, d2c)
    for xy, value in zip(np.rint(uv[visible]).astype(int), residual[selected][visible]):
        color = (255, 65, 65) if abs(value) >= .004 else (90, 245, 135)
        cv2.circle(overlay, tuple(xy), 1, color, -1)

    center = base(*((lo+hi)/2))
    arrow = np.array([center, center+direction*.3])
    uv, visible = project_base(arrow, frame, d2c)
    if visible.all():
        a, b = np.rint(uv).astype(int)
        cv2.arrowedLine(overlay, tuple(a), tuple(b), (255, 65, 235), 4, tipLength=.12)

    # A static contour checks alignment only; it does not measure motion.
    contour_direction = None
    contour_angle = None
    if contours:
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        rays_xy = cv2.undistortPoints(contour.reshape(-1, 1, 2),
            frame.color_intrinsics.matrix, np.asarray(frame.color_intrinsics.distortion)).reshape(-1, 2)
        rays = np.c_[rays_xy, np.ones(len(rays_xy))]
        color_to_base = frame.camera_to_world_cv @ np.linalg.inv(d2c)
        origin = color_to_base[:3, 3]
        directions = rays @ color_to_base[:3, :3].T
        distances = ((surface.plane_to_base[:3, 3]-origin) @ normal)/(directions @ normal)
        contour_base = origin+directions*distances[:, None]
        local = (contour_base-surface.plane_to_base[:3, 3]) @ surface.plane_to_base[:3, :3]
        rect = cv2.minAreaRect(local[:, :2].astype(np.float32))
        corners = cv2.boxPoints(rect)
        edges = np.roll(corners, -1, axis=0)-corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        contour_direction = surface.plane_to_base[:3, :2] @ edge
        contour_direction /= np.linalg.norm(contour_direction)
        if contour_direction @ direction < 0: contour_direction *= -1
        contour_angle = float(np.degrees(np.arccos(np.clip(contour_direction @ direction, -1, 1))))
        uv, visible = project_base([center, center+contour_direction*.3], frame, d2c)
        if visible.all():
            a, b = np.rint(uv).astype(int)
            cv2.arrowedLine(overlay, tuple(a), tuple(b), (255, 175, 45), 3, tipLength=.12)

    reference = meta.get('timestamp_metadata', {}).get('host_clock_reference', {})
    captured_wall = reference.get('unix_s')
    age = None if captured_wall is None else time.time()-captured_wall
    report = {'mode': 'live_shared_camera_read_only', 'frame_sequence': frame.header.sequence,
        'source_file': str(path.resolve()), 'capture_wall_time': captured_wall,
        'frame_age_at_report_s': age, 'calibration_id': frame.calibration_id,
        'serial_number': frame.serial_number, 'plane': surface.plane,
        'green_native_depth_points': len(points),
        'normal_inlier_rms_mm': float(np.sqrt(np.mean(residual[inliers]**2))*1000),
        'normal_inlier_fraction': float(inliers.mean()),
        'configured_direction_on_measured_plane': direction.tolist(),
        'static_green_outline_long_axis': None if contour_direction is None else contour_direction.tolist(),
        'configured_vs_static_outline_angle_deg': contour_angle,
        'camera_connections': 0, 'robot_connections': 0, 'commands_sent': 0,
        'limitation': 'Static green outline is not motion calibration; plane fit does not independently verify camera-to-robot extrinsics.'}
    canvas = Image.new('RGB', (2560, 920), (245, 245, 245))
    canvas.paste(Image.fromarray(frame.rgb), (0, 70))
    canvas.paste(Image.fromarray(overlay), (1280, 70))
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 28)
    title = ImageFont.truetype(font_path, 34)
    draw.text((20, 12), '实时原始 RGB（共享相机，不重复连接设备）', font=title, fill=(20, 20, 20))
    draw.text((1300, 12), '实测带面平面与方向叠加', font=title, fill=(20, 20, 20))
    stamp = '?' if captured_wall is None else datetime.fromtimestamp(captured_wall, ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    lines = [f'曝光 {stamp} · 帧 {frame.header.sequence} · 绿色有效深度 {len(points)} 点 · 平面倾角 {surface.plane["tilt_from_base_xy_deg"]:.2f}° · 内点 RMS {report["normal_inlier_rms_mm"]:.2f} mm',
        '绿边：绿色带面区域  |  青色网格：拟合平面，每格 10 cm  |  绿点：距平面 <4 mm，红点：超出 4 mm',
        '紫箭头：当前配置方向  |  橙箭头：静态轮廓长轴（不是运动测速）  |  数据超过 2 秒显示过期；关闭窗口退出']
    for i, line in enumerate(lines): draw.text((20, 800+i*37), line, font=font, fill=(20, 20, 20))
    return canvas, report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = CellConfig.load(args.config)
    calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    path = args.follow_run/'preview/latest.npz'
    history = []

    def update():
        canvas, report, arrays = inspect(path, config, calibration)
        if report['frame_age_at_report_s'] is None or report['frame_age_at_report_s'] > 2:
            raise RuntimeError('Shared camera frame is stale; waiting for fresh RGB-D')
        png = args.output/'latest.png'
        canvas.save(args.output/'latest.tmp.png')
        (args.output/'latest.tmp.png').replace(png)
        (args.output/'report.tmp.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (args.output/'report.tmp.json').replace(args.output/'report.json')
        if len(history) < 12:
            history.append(report)
            np.savez_compressed(args.output/f'frame_{report["frame_sequence"]:09d}.npz', **arrays)
            (args.output/'samples.json').write_text(json.dumps(history, ensure_ascii=False, indent=2))
        print(json.dumps({k: report[k] for k in ('frame_sequence', 'frame_age_at_report_s',
            'normal_inlier_rms_mm', 'configured_vs_static_outline_angle_deg')}) , flush=True)
        return canvas

    if args.once:
        update()
        return
    from PyQt5 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    window.setWindowTitle('Gemini 335L - live belt plane check - camera shared / no robot commands')
    label = QtWidgets.QLabel()
    label.setAlignment(QtCore.Qt.AlignCenter)
    window.setCentralWidget(label)
    window.resize(1540, 600)

    def refresh():
        try:
            canvas = update()
            data = np.asarray(canvas).copy()
            image = QtGui.QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QtGui.QImage.Format_RGB888).copy()
            label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            window.statusBar().showMessage('实时检查：每两秒更新；仅读取现有相机共享文件。')
        except Exception as error:
            # Do not leave an old frame displayed as if it were live.
            label.clear()
            label.setText('等待新的有效 RGB-D：'+str(error))
            window.statusBar().showMessage(str(error))

    timer = QtCore.QTimer(window)
    timer.timeout.connect(refresh)
    timer.start(2000)
    window.show()
    QtCore.QTimer.singleShot(0, refresh)
    app.exec_()


if __name__ == '__main__':
    main()

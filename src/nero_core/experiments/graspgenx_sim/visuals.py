"""Export exact control-frame depth and project network TCPs, without simulator truth.

All full-frame PNGs keep the depth raster's native resolution. OpenCV image
arrays and colors are BGR; only the explicitly labelled detail crops are zoomed.
This module imports neither MuJoCo nor any hardware/backend module.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PNG_COMPRESSION = 1
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def _path(prefix, suffix):
    return Path(f"{prefix}_{suffix}").resolve()


def _save_png(path, bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise OSError(f"Could not encode PNG: {path}")
    # imencode/tofile also works with non-ASCII paths on OpenCV builds that do
    # not support them in imwrite/imread.
    encoded.tofile(str(path))


def _read_png(path):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise OSError(f"Could not read preview PNG: {path}")
    return result


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _transform(value, name):
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid homogeneous 4x4 transform")
    return transform


def _camera(calibration, shape):
    height, width = shape
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1])):
        raise ValueError("intrinsics must be a finite pinhole 3x3 K with positive focal lengths")
    if int(calibration.get("width", width)) != width or int(calibration.get("height", height)) != height:
        raise ValueError("Camera dimensions differ from the native depth raster; resizing is forbidden")
    transform = _transform(calibration["camera_to_world"], "camera_to_world")
    return {
        "intrinsics": intrinsics.tolist(), "camera_to_world": transform.tolist(),
        "camera_convention": "mujoco_x_right_y_up_minus_z_forward",
        "width": int(width), "height": int(height),
        "projection": "world -> inverse(camera_to_world) GL -> diag(1,-1,-1) OpenCV -> K",
    }


def project_world_points(points_world, camera):
    """Return native (u,v), optical depth and positive-depth flags for N points.

    Behind-camera and nonfinite points have NaN pixels and False flags. The
    flags concern the near plane, not image bounds; drawing clips to the raster.
    """
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    transform = _transform(camera["camera_to_world"], "camera_to_world")
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    local_gl = (homogeneous @ np.linalg.inv(transform).T)[:, :3]
    optical = local_gl @ _GL_TO_CV.T
    depth = optical[:, 2]
    forward = np.isfinite(optical).all(axis=1) & (depth > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    projected = optical[forward] @ intrinsics.T
    pixels[forward] = projected[:, :2] / projected[:, 2, None]
    return pixels, depth, forward


def _crop_image(image, metadata):
    x0, y0, x1, y1 = metadata["crop_xyxy"]
    result = image[y0:y1, x0:x1].copy()
    scale = int(metadata["crop_scale"])
    if scale > 1:
        result = cv2.resize(result, (result.shape[1] * scale, result.shape[0] * scale),
                            interpolation=cv2.INTER_NEAREST)
    return result


def save_input_visuals(output_prefix, frame, track_id):
    """Save one exact frame and return small, JSON-serializable drawing metadata.

    ``output_prefix`` is a filename stem, e.g. ``inputs/track_0001_try_01``.
    Capture this metadata with the pending inference job; its calibration and
    PNG must not be replaced by a newer frame when inference finishes.
    """
    depth = np.asarray(frame["top_depth"])
    labels = np.asarray(frame["top_instance_labels"])
    if depth.ndim != 2 or labels.shape != depth.shape:
        raise ValueError("top_depth and top_instance_labels must share a native 2D raster")
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError("top_depth must be numeric metric depth")
    height, width = depth.shape
    camera = _camera(frame["top_camera"], depth.shape)
    mask = labels == int(track_id)
    if not mask.any():
        raise ValueError(f"Track {track_id} is absent from this control frame")
    valid = np.isfinite(depth) & (depth > 0)
    target_depth = depth[mask & valid]
    if not len(target_depth):
        raise ValueError(f"Track {track_id} has no finite positive rendered depth")
    low, high = np.quantile(target_depth, [.01, .99])
    margin = max(.015, float(high - low) * .15)
    low, high = max(0., float(low - margin)), float(high + margin)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.rint(np.clip((depth[valid] - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    preview = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    preview[~mask] = np.rint(preview[~mask] * .45).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    padding = max(28, int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * .5))
    crop = [max(0, bbox[0] - padding), max(0, bbox[1] - padding),
            min(width, bbox[2] + padding), min(height, bbox[3] + padding)]
    scale = min(4, max(1, int(np.ceil(512 / max(crop[2] - crop[0], crop[3] - crop[1])))))
    timestamp = float(frame["timestamp"])
    cv2.putText(preview, f"CONTROL DEPTH | track {int(track_id)} | t={timestamp:.3f}s | native {width}x{height}",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, f"target depth color range {low:.3f}..{high:.3f} m; context dimmed",
                (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    paths = {"depth_npy": str(_path(output_prefix, "depth.npy")),
             "trackmask_png": str(_path(output_prefix, "trackmask.png")),
             "depth_preview_png": str(_path(output_prefix, "depth_preview.png")),
             "depth_preview_crop_png": str(_path(output_prefix, "depth_preview_crop.png")),
             "metadata_json": str(_path(output_prefix, "visuals.json"))}
    Path(paths["depth_npy"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["depth_npy"], depth, allow_pickle=False)
    _save_png(paths["trackmask_png"], mask.astype(np.uint8) * 255)
    _save_png(paths["depth_preview_png"], preview)
    metadata = {"source": "exact_control_frame_rendered_depth_and_visual_trackmask",
                "track_id": int(track_id), "timestamp": timestamp,
                "width": int(width), "height": int(height), "camera": camera, "paths": paths,
                "target_bbox_xyxy": bbox, "crop_xyxy": crop, "crop_scale": scale,
                "crop_coordinates": "native pixel xyxy; x1/y1 exclusive; nearest-neighbor display zoom only",
                "depth_color_range_m": [low, high], "depth_units": "m",
                "png_color_order": "OpenCV BGR", "png_compression": PNG_COMPRESSION,
                "simulator_object_truth_used": False}
    _save_png(paths["depth_preview_crop_png"], _crop_image(preview, metadata))
    _save_json(paths["metadata_json"], metadata)
    return metadata


def _draw_segment(image, pixels, positive, first, second, color, thickness=2, arrow=False):
    if not positive[first] or not positive[second]:
        return False
    ends = pixels[[first, second]]
    if not np.isfinite(ends).all() or np.max(np.abs(ends)) > 1e8:
        return False
    first_pixel, second_pixel = [tuple(np.rint(p).astype(int)) for p in ends]
    inside, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), first_pixel, second_pixel)
    if not inside:
        return False
    if arrow:
        cv2.arrowedLine(image, start, end, color, thickness, cv2.LINE_AA, tipLength=.22)
    else:
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
    return True


def draw_grasp(output_prefix, T_world_tcp, metadata, label="selected network TCP"):
    """Project the actual supplied TCP pose onto its original input depth PNG.

    TCP +Z is the AGX approach direction and TCP +Y is its jaw opening axis.
    The jaw sketch is an 80 mm orientation guide, not a network width estimate.
    The caller must convert a canonical pose with T_canonical_ee_site first.
    Return the native full-frame PNG path; crop and projection JSON sit beside it.
    """
    pose = _transform(T_world_tcp, "T_world_tcp")
    image = _read_png(metadata["paths"]["depth_preview_png"])
    if image.shape[:2] != (int(metadata["height"]), int(metadata["width"])):
        raise ValueError("Input preview is not the saved native control-frame raster")
    # Every point is defined in the supplied TCP frame, then transformed by the
    # actual network-selected pose. No object position or simulator state enters.
    local = np.array([[0., 0., 0.], [.06, 0., 0.], [0., .06, 0.], [0., 0., .06],
                      [0., 0., -.08], [0., -.04, 0.], [0., .04, 0.],
                      [0., -.04, -.04], [0., .04, -.04]])
    world = local @ pose[:3, :3].T + pose[:3, 3]
    pixels, depths, positive = project_world_points(world, metadata["camera"])
    drawn = []
    if positive[0]:
        jaw_color = (255, 255, 0)
        for first, second in ((5, 7), (7, 8), (8, 6)):
            if _draw_segment(image, pixels, positive, first, second, jaw_color):
                drawn.append(f"jaw_{first}_{second}")
        if _draw_segment(image, pixels, positive, 4, 0, (0, 255, 255), 3, arrow=True):
            drawn.append("approach_plus_z")
        for axis, color in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 80, 0))):
            if _draw_segment(image, pixels, positive, 0, axis, color, 2, arrow=True):
                drawn.append("xyz"[axis - 1])
                x, y = np.rint(pixels[axis]).astype(int)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.putText(image, "+" + "XYZ"[axis - 1], (x + 4, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        x, y = np.rint(pixels[0]).astype(int)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "TCP", (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            drawn.append("tcp_origin")
        if drawn:
            # Non-ASCII labels are preserved exactly in JSON. OpenCV's built-in
            # font only covers ASCII; avoid misleading unsupported glyphs.
            display_label = str(label).encode("ascii", errors="replace").decode("ascii")
            cv2.rectangle(image, (8, 34), (min(image.shape[1] - 1, 980), 84), (20, 20, 20), -1)
            cv2.putText(image, display_label[:110], (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "TCP XYZ: red/green/blue | approach +Z: yellow | jaws +Y: cyan (80mm guide)",
                        (12, 75), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    path = _path(output_prefix, "grasp.png")
    crop_path = _path(output_prefix, "grasp_crop.png")
    _save_png(path, image)
    _save_png(crop_path, _crop_image(image, metadata))
    names = ("tcp", "tcp_x", "tcp_y", "tcp_z", "approach_start", "jaw_minus_y", "jaw_plus_y",
             "jaw_rear_minus_y", "jaw_rear_plus_y")
    projections = {name: {"world_m": world[i].tolist(),
                          "pixel_uv": pixels[i].tolist() if positive[i] else None,
                          "optical_depth_m": float(depths[i]) if np.isfinite(depths[i]) else None,
                          "in_front_of_camera": bool(positive[i])} for i, name in enumerate(names)}
    _save_json(_path(output_prefix, "grasp.json"), {
        "source": "supplied_network_selected_T_world_tcp", "label": str(label),
        "track_id": metadata["track_id"], "capture_timestamp": metadata["timestamp"],
        "T_world_tcp": pose.tolist(), "camera": metadata["camera"],
        "input_metadata_json": metadata["paths"]["metadata_json"],
        "native_grasp_png": str(path), "crop_grasp_png": str(crop_path),
        "crop_xyxy": metadata["crop_xyxy"], "crop_scale": metadata["crop_scale"],
        "axis_length_m": .06, "approach_direction_tcp": "+Z",
        "jaw_opening_axis_tcp": "+Y (AGX canonical +X)", "jaw_guide_span_m": .08,
        "jaw_guide_is_network_predicted_width": False, "projections": projections,
        "drawn_components": drawn, "simulator_object_truth_used": False,
        "png_color_order": "OpenCV BGR", "png_compression": PNG_COMPRESSION,
    })
    return str(path)

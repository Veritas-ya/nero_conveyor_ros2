"""Safe bridge from hardware RGB-D frames to the existing vision pipeline."""

from __future__ import annotations

from typing import Any

from .types import BackendUnavailable, RGBDFrame


def process_rgbd_frame(
    pipeline: Any,
    frame: RGBDFrame,
    *,
    belt_speed_m_s: float,
) -> tuple[Any, ...]:
    """Process one physical frame with an explicit OpenCV transform contract.

    The current geometric detector is depth-only.  RGB is passed only when it
    is explicitly depth-aligned and has the same pixel grid; an unaligned color
    image is never silently combined with depth intrinsics.
    """

    if not frame.calibrated or frame.camera_to_world_cv is None:
        raise BackendUnavailable(
            "camera-to-world calibration is required before perception"
        )
    rgb = frame.rgb if frame.rgb_aligned_to_depth else None
    if rgb is not None and rgb.shape[:2] != frame.depth_m.shape:
        raise BackendUnavailable("depth-aligned RGB shape does not match depth")
    return tuple(
        pipeline.process(
            rgb,
            frame.depth_m,
            frame.depth_intrinsics.matrix,
            frame.camera_to_world_cv,
            capture_timestamp=frame.header.sample_time,
            belt_speed=float(belt_speed_m_s),
            camera_convention="opencv",
        )
    )

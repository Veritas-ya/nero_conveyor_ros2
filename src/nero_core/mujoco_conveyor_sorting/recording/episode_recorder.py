"""Portable NPZ/JSON episode recorder.

The recorder deliberately stores a simple, documented first-version format.  The
``export_*`` methods are extension points for dataset-specific conversions and do
not silently pretend to produce LeRobot/HDF5/Zarr data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class EpisodeRecorder:
    """Accumulate one episode and atomically write numeric data plus metadata."""

    NUMERIC_OBSERVATION_KEYS = (
        "top_rgb",
        "top_depth",
        "wrist_rgb",
        "joint_pos",
        "joint_vel",
        "ee_pose",
        "gripper_opening",
        "belt_speed",
        "timestamp",
        "top_depth_timestamp",
        "top_rgb_timestamp",
        "wrist_rgb_timestamp",
        "wrist_depth_timestamp",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        record_images: bool = True,
        record_video: bool = False,
        video_fps: int = 20,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_images = bool(record_images)
        self.record_video = bool(record_video)
        self.video_fps = int(video_fps)
        self._episode_index = 0
        self._active = False
        self._frames: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    def start_episode(
        self, metadata: Mapping[str, Any] | None = None, *, episode_id: int | None = None
    ) -> None:
        if self._active:
            raise RuntimeError("Finish the active episode before starting another one")
        if episode_id is not None:
            self._episode_index = int(episode_id)
        self._metadata = dict(metadata or {})
        self._frames = []
        self._active = True

    def record_step(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._active:
            raise RuntimeError("start_episode() must be called before record_step()")
        numeric: dict[str, np.ndarray] = {}
        for key in self.NUMERIC_OBSERVATION_KEYS:
            if key not in observation:
                continue
            if observation[key] is None:
                continue
            if not self.record_images and key in {"top_rgb", "top_depth", "wrist_rgb"}:
                continue
            numeric[key] = np.asarray(observation[key]).copy()
        self._frames.append(
            {
                "numeric": numeric,
                "objects": _jsonable(observation.get("objects", [])),
                "gripper_state": _jsonable(observation.get("gripper", {})),
                "task_text": str(observation.get("task_text", "")),
                "selected_target_id": _jsonable(observation.get("selected_target_id")),
                "perception": _jsonable(observation.get("perception", {})),
                "action": _jsonable(action or {}),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": _jsonable(info or {}),
            }
        )

    @staticmethod
    def _stack_or_object(values: list[np.ndarray]) -> np.ndarray:
        if not values:
            return np.empty((0,), dtype=np.float32)
        try:
            return np.stack(values)
        except ValueError:
            return np.asarray(values, dtype=object)

    def end_episode(self, metrics: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
        if not self._active:
            raise RuntimeError("No active episode")
        stem = f"episode_{self._episode_index:06d}"
        npz_path = self.output_dir / f"{stem}.npz"
        json_path = self.output_dir / f"{stem}.json"

        arrays: dict[str, np.ndarray] = {}
        keys = {
            key for frame in self._frames for key in frame.get("numeric", {}).keys()
        }
        for key in sorted(keys):
            arrays[key] = self._stack_or_object(
                [frame["numeric"][key] for frame in self._frames if key in frame["numeric"]]
            )
        arrays["reward"] = np.asarray(
            [frame["reward"] for frame in self._frames], dtype=np.float32
        )
        arrays["terminated"] = np.asarray(
            [frame["terminated"] for frame in self._frames], dtype=bool
        )
        arrays["truncated"] = np.asarray(
            [frame["truncated"] for frame in self._frames], dtype=bool
        )
        np.savez_compressed(npz_path, **arrays)

        document = {
            "format": "mujoco_conveyor_sorting.npz.v1",
            "episode_id": self._episode_index,
            "steps": len(self._frames),
            "metadata": _jsonable(self._metadata),
            "metrics": _jsonable(metrics or {}),
            "frames": [
                {key: value for key, value in frame.items() if key != "numeric"}
                for frame in self._frames
            ],
        }
        tmp_path = json_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.replace(json_path)

        if self.record_video and self.record_images:
            self._write_video(stem)
        self._active = False
        self._episode_index += 1
        return npz_path, json_path

    def _write_video(self, stem: str) -> None:
        frames = [
            frame["numeric"]["top_rgb"]
            for frame in self._frames
            if "top_rgb" in frame["numeric"]
        ]
        if not frames:
            return
        import imageio.v2 as imageio

        imageio.mimwrite(
            self.output_dir / f"{stem}.mp4",
            [np.asarray(frame, dtype=np.uint8) for frame in frames],
            fps=self.video_fps,
            quality=7,
        )

    def export_lerobot(self, destination: str | Path) -> None:
        raise NotImplementedError(
            "LeRobot export is an extension point; use the recorded NPZ/JSON v1 data"
        )

    def export_hdf5(self, destination: str | Path) -> None:
        raise NotImplementedError(
            "HDF5 export is an extension point; use the recorded NPZ/JSON v1 data"
        )

    def export_zarr(self, destination: str | Path) -> None:
        raise NotImplementedError(
            "Zarr export is an extension point; use the recorded NPZ/JSON v1 data"
        )

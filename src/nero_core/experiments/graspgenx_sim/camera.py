"""Simulation-only fixed-camera RGB-D input; no camera SDK or robot backend."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from mujoco_conveyor_sorting.envs.conveyor_sorting_env import ConveyorSortingEnv
from mujoco_conveyor_sorting.perception.rgbd_vision_pipeline import deproject_depth_to_world
from experiments.graspgenx_sim.cached_projection import CachedDepthProjector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / "recordings/pick_test/20261008_175405_103492/frame_000040.json"


class FixedGeminiSimulation(ConveyorSortingEnv):
    """Reuse NERO dynamics with a fixed camera derived from a saved capture.

    This does not read a live device. The camera pose is relative to the model
    base; the simulated conveyor layout remains the existing model layout.
    Rendering uses square pixels and a centered principal point, explicitly
    recorded separately from the physical camera principal point.
    """

    def __init__(self, *, capture_path=DEFAULT_CAPTURE, **kwargs):
        self.saved_capture_path = Path(capture_path).resolve()
        self.saved_capture = json.loads(self.saved_capture_path.read_text())
        super().__init__(**kwargs)
        self._cached_depth_projector = CachedDepthProjector()
        self.vision_pipeline.depth_projector = self._cached_depth_projector.project

    def get_observation(self, *, include_camera_images=False):
        # The control policy consumes visual tracks. Full-resolution camera
        # products remain in _camera_frames for inference and explicit export.
        return super().get_observation(include_camera_images=include_camera_images)

    def _record_truth_state(self):
        # The rendered policy reads perception_buffer, and the independent
        # placement evaluator reads current MuJoCo data and contacts directly.
        # These 500 Hz oracle copies serve only simulator_history regression
        # mode; skipping them changes neither physics nor visual captures.
        if self.perception_mode == "rendered_rgbd":
            return
        super()._record_truth_state()

    def _read_model_names(self):
        super()._read_model_names()
        mujoco.mj_forward(self.model, self.data)
        camera_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera"))
        rig_id = int(self.model.cam_bodyid[camera_id])
        parent_id = int(self.model.body_parentid[rig_id])
        base_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
        world_base = np.eye(4)
        world_base[:3, :3] = self.data.xmat[base_id].reshape(3, 3)
        world_base[:3, 3] = self.data.xpos[base_id]
        self.world_depth_cv = world_base @ np.asarray(self.saved_capture["depth_to_base_cv"])
        world_gl = self.world_depth_cv @ np.diag([1., -1., -1., 1.])
        world_parent = np.eye(4)
        world_parent[:3, :3] = self.data.xmat[parent_id].reshape(3, 3)
        world_parent[:3, 3] = self.data.xpos[parent_id]
        parent_gl = np.linalg.inv(world_parent) @ world_gl
        self.model.body_pos[rig_id] = parent_gl[:3, 3]
        mujoco.mju_mat2Quat(self.model.body_quat[rig_id], parent_gl[:3, :3].reshape(9))
        self.model.cam_pos[camera_id] = 0.
        self.model.cam_quat[camera_id] = [1., 0., 0., 0.]
        intr = self.saved_capture["depth_intrinsics"]
        self.model.cam_fovy[camera_id] = math.degrees(2 * math.atan(intr["height"] / (2 * intr["fy"])))
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(intr["width"]))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(intr["height"]))
        # MSAA resolves depth from sample locations away from the pixel center.
        # Use single-sample metric depth for an unambiguous pinhole projection.
        self.model.vis.quality.offsamples = 0
        mujoco.mj_forward(self.model, self.data)

    def _camera_calibration(self, camera_id, resolution):
        result = super()._camera_calibration(camera_id, resolution)
        # Raster pixel indices name pixel centers, not pixel-edge coordinates.
        result["intrinsics"][0, 2] -= .5
        result["intrinsics"][1, 2] -= .5
        result["distortion"] = [0.] * 5
        result["preprocessing"] = "native MuJoCo pinhole render; no crop/resize/undistort"
        return result

    def _render_camera(self, camera_name, resolution, *, depth):
        renderer = self._renderer_for(resolution)
        if renderer is None:
            raise RuntimeError("MuJoCo render failed; zero-image fallback is forbidden for this benchmark")
        option = mujoco.MjvOption()
        option.geomgroup[5] = 0
        option.sitegroup[:] = 0
        renderer.update_scene(self.data, camera=camera_name, scene_option=option)
        renderer.disable_segmentation_rendering()
        if depth:
            renderer.enable_depth_rendering()
        else:
            renderer.disable_depth_rendering()
        image = renderer.render().copy()
        renderer.disable_depth_rendering()
        return image

    def camera_record(self):
        calibration = self._camera_calibration(self.top_camera_id, self.top_resolution)
        return {
            "source": str(self.saved_capture_path),
            "camera_fixed": True,
            "world_depth_cv": calibration["camera_to_world_opencv"].tolist(),
            "resolution": list(self.top_resolution),
            "K_render": calibration["intrinsics"].tolist(),
            "D_render": [0.] * 5,
            "physical_depth_intrinsics_reference": self.saved_capture["depth_intrinsics"],
            "principal_point_note": "Simulation uses centered raster principal point; physical reference cx/cy are not silently applied to a different render.",
            "preprocessing": calibration["preprocessing"],
            "depth_projection": "float64 fixed-camera world-ray cache; full resolution; refresh on K/T/size change",
            "control_observation_images": False,
            "simulator_truth_history": (
                "disabled; current physics/contact evaluation preserved"
                if self.perception_mode == "rendered_rgbd"
                else "original simulator_history recording preserved"
            ),
            "scene_note": "Camera pose relative to base is replayed; conveyor/base/bin layout is the existing simulation, not a measured digital twin.",
        }


def cloud_for_track(frame, track_id):
    """Only metric rendered depth, calibration and visual instance masks enter."""
    calibration = frame["top_camera"]
    mask = frame["top_instance_labels"] == int(track_id)
    rows, columns = np.nonzero(mask)
    depth = np.asarray(frame["top_depth"])[rows, columns]
    K = np.asarray(calibration["intrinsics"])
    optical = np.column_stack(((columns - K[0, 2]) * depth / K[0, 0],
                               (rows - K[1, 2]) * depth / K[1, 1], depth))
    local = optical * np.array([1., -1., -1.])
    transform = np.asarray(calibration["camera_to_world"])
    cloud = local @ transform[:3, :3].T + transform[:3, 3]
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    return np.asarray(cloud, dtype=np.float32)


def projection_selfcheck(env):
    """Check rendered belt points against its analytic surface; truth is test-only."""
    depth = env._render_camera(env.top_camera_name, env.top_resolution, depth=True)
    seg = env._render_segmentation(env.top_camera_name, env.top_resolution)
    cal = env._camera_calibration(env.top_camera_id, env.top_resolution)
    points = deproject_depth_to_world(depth, cal["intrinsics"], cal["camera_to_world"])
    geom = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "conveyor_surface"))
    mask = (seg[..., 0] == geom) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    # Exclude raster edges so mixed background pixels cannot corrupt the test.
    import cv2
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    error = np.abs(points[mask, 2] - env.belt_height)
    record = {
        "test": "rendered conveyor plane deprojection",
        "pixels": int(len(error)),
        "median_error_m": float(np.median(error)) if len(error) else None,
        "p99_error_m": float(np.quantile(error, .99)) if len(error) else None,
        "segmentation_usage": "projection selfcheck only; never a policy input",
    }
    record["passed"] = bool(len(error) > 100 and record["p99_error_m"] < 1e-4)
    return record

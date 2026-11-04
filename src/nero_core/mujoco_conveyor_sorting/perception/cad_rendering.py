"""Render isolated CAD depth hypotheses in a calibrated OpenCV camera.

Only object specifications and candidate camera-relative poses are consumed.
The scene contains no conveyor, robot, simulator identities or live state.
Mesh coordinates and returned depth are in metres; object poses use the
support-plane body origin declared by :class:`ObjectSpec`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt

from ..calibration.geometry import rigid_transform
from ..tasks.object_spec import ObjectSpec


def _numbers(values: npt.ArrayLike) -> str:
    return " ".join(f"{float(value):.16g}" for value in np.asarray(values).ravel())


class CadDepthRenderer:
    """One reusable, offscreen scene for several CAD object categories.

    ``intrinsics`` describes the actual output pixels, including an off-centre
    principal point for cropped images. Distortion must already be removed.
    Mesh specs use ``bop_model_to_body_translation_m`` as their mesh offset;
    box specs place their bottom at local Z=0.
    """

    def __init__(
        self,
        specs: Iterable[ObjectSpec],
        intrinsics: npt.ArrayLike,
        width: int,
        height: int,
    ) -> None:
        parsed = tuple(specs)
        if not parsed or any(not isinstance(spec, ObjectSpec) for spec in parsed):
            raise ValueError("specs must contain at least one ObjectSpec")
        if len({spec.category_id for spec in parsed}) != len(parsed):
            raise ValueError("CAD category IDs must be unique")
        self.width, self.height = int(width), int(height)
        if self.width != width or self.height != height or min(self.width, self.height) < 1:
            raise ValueError("render width and height must be positive integers")
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-10)
            or abs(matrix[0, 1]) > 1e-10
            or abs(matrix[1, 0]) > 1e-10
        ):
            raise ValueError("intrinsics must be a finite, zero-skew pinhole 3x3 matrix")
        self.intrinsics = matrix.copy()
        self._near, self._far = 0.005, 10.0
        self._renderer = None
        self._active_category: int | None = None

        root = ET.Element("mujoco", model="isolated_cad_depth")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "option", gravity="0 0 0")
        # MuJoCo clipping distances are multiplied by stat.extent. Fix it so
        # depth validity does not depend on the number or size of loaded CADs.
        ET.SubElement(root, "statistic", extent="1", center="0 0 0")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(self.width), offheight=str(self.height))
        ET.SubElement(visual, "quality", offsamples="0")
        ET.SubElement(visual, "map", znear=str(self._near), zfar=str(self._far))
        assets = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        # World axes are the requested CV camera axes. A MuJoCo camera looks
        # along -Z with +Y upwards, so a pi rotation around X supplies CV axes.
        # principalpixel is a frustum offset with the opposite sign from cx/cy;
        # pixel indices refer to centres, hence the (resolution - 1) / 2 terms.
        ET.SubElement(
            world,
            "camera",
            name="cad_camera",
            pos="0 0 0",
            quat="0 1 0 0",
            ipd="0",
            focalpixel=_numbers([matrix[0, 0], matrix[1, 1]]),
            principalpixel=_numbers([
                (self.width - 1) / 2 - matrix[0, 2],
                (self.height - 1) / 2 - matrix[1, 2],
            ]),
            sensorsize="0.01 0.01",
            resolution=f"{self.width} {self.height}",
        )
        for spec in parsed:
            name = f"cad_{spec.category_id}"
            body = ET.SubElement(world, "body", name=name, mocap="true")
            attributes = {
                "name": f"{name}_geom", "group": "0", "contype": "0",
                "conaffinity": "0", "density": "0", "rgba": "0.7 0.7 0.7 1",
            }
            if spec.geometry_type == "mesh":
                if spec.mesh_path is None or not Path(spec.mesh_path).is_file():
                    raise FileNotFoundError(f"CAD mesh for category {spec.category_id}: {spec.mesh_path}")
                offset = np.asarray(
                    spec.geometry.get("bop_model_to_body_translation_m", [0, 0, 0]),
                    dtype=np.float64,
                )
                if offset.shape != (3,) or not np.isfinite(offset).all():
                    raise ValueError("bop_model_to_body_translation_m must be a finite XYZ vector")
                ET.SubElement(assets, "mesh", name=name, file=str(Path(spec.mesh_path).resolve()))
                attributes.update(type="mesh", mesh=name, pos=_numbers(offset))
            elif spec.geometry_type == "box":
                attributes.update(
                    type="box", size=_numbers(spec.dimensions / 2),
                    pos=_numbers([0, 0, spec.dimensions[2] / 2]),
                )
            else:
                raise ValueError(f"unsupported CAD geometry: {spec.geometry_type}")
            ET.SubElement(body, "geom", **attributes)

        # Respect an explicit EGL/OSMesa/GLFW selection made by the caller.
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self._data = mujoco.MjData(self._model)
        self._geom_ids = {
            spec.category_id: self._model.geom(f"cad_{spec.category_id}_geom").id
            for spec in parsed
        }
        self._mocap_ids = {
            spec.category_id: int(self._model.body_mocapid[self._model.body(f"cad_{spec.category_id}").id])
            for spec in parsed
        }
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[1] = 1
        self._scene_option.sitegroup[:] = 0
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._renderer.enable_depth_rendering()

    def render(
        self,
        category_id: int,
        object_to_camera_cv: npt.ArrayLike,
        scale: float = 1.0,
    ) -> npt.NDArray[np.float32]:
        """Return metric optical-axis depth, with zero for background pixels.

        Uniform scale is applied about the object's support-plane origin. A
        nominal mesh at translation ``t / scale`` projects identically to a
        scaled mesh at ``t``; multiplying the resulting depth recovers metres.
        This avoids rebuilding meshes during candidate search.
        """
        if self._renderer is None:
            raise RuntimeError("CAD depth renderer is closed")
        if category_id not in self._geom_ids:
            raise ValueError(f"unknown CAD category_id: {category_id}")
        pose = rigid_transform(object_to_camera_cv, "object_to_camera_cv")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("CAD scale must be finite and positive")
        if self._active_category != category_id:
            if self._active_category is not None:
                self._model.geom_group[self._geom_ids[self._active_category]] = 0
            self._model.geom_group[self._geom_ids[category_id]] = 1
            self._active_category = category_id
        mocap = self._mocap_ids[category_id]
        self._data.mocap_pos[mocap] = pose[:3, 3] / scale
        self._mujoco.mju_mat2Quat(self._data.mocap_quat[mocap], pose[:3, :3].ravel())
        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera="cad_camera", scene_option=self._scene_option)
        depth = self._renderer.render().copy()
        # Empty depth buffers map to the far plane (with floating-point error).
        # A small clip margin reliably excludes those values without a second
        # segmentation render for every pose hypothesis.
        valid = np.isfinite(depth) & (depth > self._near) & (depth < self._far * 0.999)
        depth[~valid] = 0
        depth *= scale
        return depth

    def close(self) -> None:
        """Release the offscreen context; repeated calls are harmless."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "CadDepthRenderer":
        if self._renderer is None:
            raise RuntimeError("CAD depth renderer is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

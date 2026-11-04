"""Offline NERO + fixed optical camera + flange mounted calibration board.

This module never imports a hardware backend.  Robot geometry and kinematics
come from the existing robot.xml, not from a separately invented DH chain.
Transforms use metres and F=base_T_flange, C=camera_T_board, X=base_T_camera,
Y=flange_T_board; the camera is expressed in OpenCV optical coordinates.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .eye_to_hand import Chessboard
from .geometry import inverse, rigid_transform


def _numbers(values) -> str:
    return " ".join(f"{float(x):.16g}" for x in values)


class NeroCalibrationRenderer:
    """Kinematic-only renderer with a single fixed camera per experiment.

    ``set_joints`` changes qpos and calls mj_forward; no physics integration,
    actuator command, transport or CAN interface is involved.  A group's
    camera remains fixed unless the caller explicitly calls set_camera.
    """

    joint_names = [f"joint{i}" for i in range(1, 8)]
    # The offline SDK audit verifies NERO get_flange_pose/fk corresponds to
    # link7, not the XML's downstream gripper_flange visual adapter.
    flange_body_name = "link7"
    base_body_name = "base_link"
    camera_name = "calibration_external_camera"
    board_body_name = "calibration_board"
    # GL: right, up, backwards. CV: right, down, forwards.
    cv_to_gl_rotation = np.diag([1.0, -1.0, -1.0])

    def __init__(self, board: Chessboard, width=1280, height=720,
                 fx=603.524651024772, fy=None):
        # Use EGL before importing mujoco. An explicit user's GL selection wins.
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mujoco = mujoco
        self.board = board
        self.width, self.height = int(width), int(height)
        self.fx, self.fy = float(fx), float(fx if fy is None else fy)
        if min(self.width, self.height, self.fx, self.fy) <= 0:
            raise ValueError("positive image dimensions and focal lengths required")
        self.package_path = Path(__file__).resolve().parents[1]
        self.robot_xml = self.package_path / "models" / "robot.xml"
        root = ET.fromstring(self.robot_xml.read_text())
        root.set("model", "nero_eye_to_hand_offline_diagnostic")
        compiler = ET.Element("compiler", angle="radian",
                              meshdir=str(self.package_path / "assets"))
        root.insert(0, compiler)
        ET.SubElement(root, "option", gravity="0 0 0")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(self.width),
                      offheight=str(self.height))
        ET.SubElement(visual, "quality", offsamples="8", shadowsize="2048")
        ET.SubElement(visual, "headlight", ambient="0.55 0.55 0.55",
                      diffuse="0.45 0.45 0.45", specular="0 0 0")
        ET.SubElement(visual, "map", znear="0.005", zfar="20")
        defaults = root.find("default")
        adapter = ET.SubElement(defaults, "default", {"class": "visual_only"})
        ET.SubElement(adapter, "geom", contype="0", conaffinity="0", density="0")
        assets = root.find("asset")
        for name, rgba in (("board_black", "0.002 0.002 0.002 1"),
                           ("board_white", "0.94 0.94 0.94 1"),
                           ("board_origin_red", "1 0 0 1")):
            ET.SubElement(assets, "material", name=name, rgba=rgba,
                          emission="1", specular="0", shininess="0")
        world = root.find("worldbody")
        ET.SubElement(world, "light", name="diagnostic_light",
                      pos="0 -1 2", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="diagnostic_floor", type="plane",
                      pos="0 0 0", size="2 2 0.01", rgba="0.30 0.32 0.35 1",
                      contype="0", conaffinity="0")
        # focalpixel permits distinct fx/fy. principalpixel=0 centres the
        # optical axis at the image rectangle centre; array coordinates below
        # subtract 0.5 because OpenGL raster samples are pixel centres.
        ET.SubElement(world, "camera", name=self.camera_name, mode="fixed",
                      pos="0 -1 1", quat="1 0 0 0", ipd="0",
                      focalpixel=_numbers([self.fx, self.fy]),
                      principalpixel="0 0", sensorsize="0.01 0.01",
                      resolution=f"{self.width} {self.height}")
        flange = world.find(f".//body[@name='{self.flange_body_name}']")
        if flange is None or world.find(f".//body[@name='{self.base_body_name}']") is None:
            raise ValueError("the existing NERO model lacks the named base/flange")
        target = ET.SubElement(flange, "body", name=self.board_body_name,
                               pos="0 0 0.20")
        self._add_board(target)
        # Only an isolated model copy is compiled; the original XML is untouched.
        self.model_xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                           for n in self.joint_names]
        if min(self._joint_ids) < 0:
            raise ValueError("NERO seven named joints are required")
        self.qpos_indices = [int(self.model.jnt_qposadr[j]) for j in self._joint_ids]
        self.joint_limits_rad = self.model.jnt_range[self._joint_ids].copy()
        self._base_id = self.model.body(self.base_body_name).id
        self._flange_id = self.model.body(self.flange_body_name).id
        self._board_id = self.model.body(self.board_body_name).id
        self._camera_id = self.model.camera(self.camera_name).id
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[3] = 0  # collision meshes remain model-only
        self._scene_option.sitegroup[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.intrinsics = {
            "width": self.width, "height": self.height,
            "matrix": [[self.fx, 0.0, (self.width - 1) / 2],
                       [0.0, self.fy, (self.height - 1) / 2], [0.0, 0.0, 1.0]],
            "distortion": [0.0] * 5, "distortion_model": "opencv",
            "source": "mujoco_focalpixel_no_distortion",
        }

    def _add_board(self, target):
        s = self.board.square_size_m
        nx, ny = self.board.columns + 1, self.board.rows + 1
        # Corners are (0..columns-1)*s, (0..rows-1)*s, z=0 exactly.
        # Checker squares extend from -s to columns*s / rows*s.
        border = 0.7 * s
        centre = [(nx - 2) * s / 2, (ny - 2) * s / 2]
        ET.SubElement(target, "geom", name="board_white_border", type="box",
                      pos=_numbers([*centre, -0.00101]),
                      size=_numbers([nx * s / 2 + border, ny * s / 2 + border, 0.001]),
                      material="board_white", contype="0", conaffinity="0", density="0")
        for row in range(ny):
            for col in range(nx):
                ET.SubElement(target, "geom", name=f"board_square_{row}_{col}",
                              type="box", pos=_numbers([(col - .5) * s, (row - .5) * s, -.0005]),
                              size=_numbers([s / 2, s / 2, .0005]),
                              material="board_black" if (row + col) % 2 == 0 else "board_white",
                              contype="0", conaffinity="0", density="0")
        # Asymmetric physical marker outside the checker region. Numbering can
        # be decided from this colour in the image, without ground-truth pose.
        self.origin_marker_board = np.array([-1.4 * s, -1.4 * s, 0.0])
        ET.SubElement(target, "geom", name="board_physical_origin_marker", type="cylinder",
                      pos=_numbers([*self.origin_marker_board[:2], -.00025]),
                      size=_numbers([.19 * s, .00025]), material="board_origin_red",
                      contype="0", conaffinity="0", density="0")

    def _body_transform(self, body_id):
        result = np.eye(4)
        result[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        result[:3, 3] = self.data.xpos[body_id]
        return result

    def _write_transform(self, transform, position, quaternion):
        transform = rigid_transform(transform)
        position[:] = transform[:3, 3]
        self.mujoco.mju_mat2Quat(quaternion, transform[:3, :3].reshape(-1))

    def set_joints(self, q7):
        q = np.asarray(q7, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("q7 must contain seven finite joint1..joint7 radians")
        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0
        self.mujoco.mj_forward(self.model, self.data)
        return inverse(self._body_transform(self._base_id)) @ self._body_transform(self._flange_id)

    def set_camera(self, X):
        X = rigid_transform(X)
        world_cv = self._body_transform(self._base_id) @ X
        world_gl = world_cv.copy()
        world_gl[:3, :3] = world_cv[:3, :3] @ self.cv_to_gl_rotation
        self._write_transform(world_gl, self.model.cam_pos[self._camera_id],
                              self.model.cam_quat[self._camera_id])
        self.mujoco.mj_forward(self.model, self.data)

    def set_board_mount(self, Y):
        self._write_transform(Y, self.model.body_pos[self._board_id],
                              self.model.body_quat[self._board_id])
        self.mujoco.mj_forward(self.model, self.data)

    def get_frame_truth(self):
        base = self._body_transform(self._base_id)
        flange = self._body_transform(self._flange_id)
        board = self._body_transform(self._board_id)
        camera_cv = np.eye(4)
        camera_cv[:3, :3] = self.data.cam_xmat[self._camera_id].reshape(3, 3) @ self.cv_to_gl_rotation
        camera_cv[:3, 3] = self.data.cam_xpos[self._camera_id]
        return {"F": inverse(base) @ flange, "Y": inverse(flange) @ board,
                "X": inverse(base) @ camera_cv, "C": inverse(camera_cv) @ board,
                "q7_rad": self.data.qpos[self.qpos_indices].copy()}

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera_name,
                                   scene_option=self._scene_option)
        return self.renderer.render().copy()

    def board_occlusion_check(self):
        """Trace visible-geometry rays near every printed inner corner.

        This is ground-truth visibility evaluation only, never a detector or
        PnP input. Four +/-10 micrometre probes avoid exact square seams. The
        geomgroup mask is identical to render(), including exclusion of the
        otherwise invisible robot collision meshes. Missing/intersection-
        ambiguous probes are explicitly unknown rather than assumed clear.
        """
        group_mask = np.asarray(self._scene_option.geomgroup, dtype=np.uint8).copy()
        camera_world = np.asarray(self.data.cam_xpos[self._camera_id]).copy()
        board_world = self._body_transform(self._board_id)
        board_geom_ids = set(np.nonzero(self.model.geom_bodyid == self._board_id)[0].tolist())
        offset = min(1e-5, self.board.square_size_m * 1e-3)
        probes = np.array([[-offset, -offset, 0], [-offset, offset, 0],
                           [offset, -offset, 0], [offset, offset, 0]])
        corner_rows = []
        totals = {"clear": 0, "occluded": 0, "unknown": 0}
        try:
            for index, corner in enumerate(self.board.object_points):
                counts = {"clear": 0, "occluded": 0, "unknown": 0}
                names, distances = [], []
                for probe in probes:
                    target = board_world[:3, :3] @ (corner + probe) + board_world[:3, 3]
                    direction = target - camera_world
                    target_distance = float(np.linalg.norm(direction))
                    if target_distance <= 1e-9:
                        status, name, distance = "unknown", None, None
                    else:
                        hit = np.array([-1], dtype=np.int32)
                        distance = float(self.mujoco.mj_ray(
                            self.model, self.data, camera_world, direction / target_distance,
                            group_mask, True, -1, hit))
                        geom_id = int(hit[0])
                        name = (self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM,
                                                     geom_id) if geom_id >= 0 else None)
                        if distance < 0 or geom_id < 0:
                            status, distance = "unknown", None
                        elif geom_id in board_geom_ids:
                            # The backing may be the first hit from the back;
                            # front/back facing is a separate printed-face test.
                            status = "clear"
                        elif distance < target_distance - 2e-5:
                            status = "occluded"
                        else:
                            status = "unknown"
                    counts[status] += 1
                    totals[status] += 1
                    names.append(name)
                    distances.append(distance)
                corner_rows.append({"corner_index": index, "probe_counts": counts,
                                    "first_hit_geom_names": names,
                                    "first_hit_distances_m": distances})
        except Exception as error:
            return {"status": "unknown", "occluded": None,
                    "reason": f"ray visibility query failed: {type(error).__name__}: {error}",
                    "method": "mj_ray with render geomgroup mask", "uses_truth_only_for_evaluation": True}
        fully = sum(row["probe_counts"]["occluded"] == len(probes) for row in corner_rows)
        partial = sum(0 < row["probe_counts"]["occluded"] < len(probes) for row in corner_rows)
        has_unknown = totals["unknown"] > 0
        occluded = True if totals["occluded"] else (None if has_unknown else False)
        return {"status": "partially_unknown" if has_unknown else "available", "occluded": occluded,
                "method": "mj_ray with render geomgroup mask; four probes per inner corner",
                "uses_truth_only_for_evaluation": True,
                "scope": "geometry occlusion at corner neighbourhoods, not texture/lighting detector success",
                "printed_face_front_back_evaluated_separately": True,
                "render_geomgroup_mask": group_mask.tolist(),
                "probe_offsets_board_m": probes.tolist(), "camera_origin_world_m": camera_world.tolist(),
                "corner_count": len(corner_rows), "probe_count": sum(totals.values()),
                "probe_status_counts": totals, "fully_occluded_corner_count": fully,
                "partially_occluded_corner_count": partial,
                "unknown_probe_count": totals["unknown"], "per_corner": corner_rows}

    def camera_projection_check(self):
        """Derive K from actual GL frustum and independently locate render pixels.

        Four coloured spheres replace the scene geoms temporarily. Their
        centroids are measured in the actual rendered array, with no use of
        detected checker corners or PnP ground truth. This exposes optical/GL
        axis flips, focal/aspect errors and half-pixel convention mistakes.
        """
        self.renderer.update_scene(self.data, camera=self.camera_name,
                                   scene_option=self._scene_option)
        scene = self.renderer.scene
        camera = scene.camera[0]
        near = float(camera.frustum_near)
        bottom, top = float(camera.frustum_bottom), float(camera.frustum_top)
        half_width = float(camera.frustum_width)
        if half_width == 0:
            half_width = (top - bottom) * self.width / self.height / 2
        derived_fx = self.width * near / (2 * half_width)
        derived_fy = self.height * near / (top - bottom)
        # centred mono camera, so the frustum centre is zero.
        derived_cx = (self.width - 1) / 2 - float(camera.frustum_center) * derived_fx / near
        derived_cy = self.height * top / (top - bottom) - .5
        derived = np.array([[derived_fx, 0, derived_cx],
                            [0, derived_fy, derived_cy], [0, 0, 1]])
        declared = np.asarray(self.intrinsics["matrix"])
        points = np.array([[-.20, -.15, .8], [0, 0, 1], [.20, .15, 1.2], [.25, -.18, 1]])
        colours = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]], dtype=np.float32)
        truth = self.get_frame_truth()
        world_cv = self._body_transform(self._base_id) @ truth["X"]
        world_points = points @ world_cv[:3, :3].T + world_cv[:3, 3]
        scene.ngeom = len(points)
        for i, (point, colour) in enumerate(zip(world_points, colours)):
            geom = scene.geoms[i]
            self.mujoco.mjv_initGeom(geom, self.mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.012, .012, .012]), point,
                                   np.eye(3).reshape(-1), colour)
            geom.emission = 1
            geom.specular = 0
        image = self.renderer.render().copy()
        predicted = points @ declared.T
        predicted = predicted[:, :2] / predicted[:, 2, None]
        measured = []
        for u, v in predicted:
            x0, y0 = max(0, int(u) - 20), max(0, int(v) - 20)
            patch = image[y0:min(self.height, int(v) + 21), x0:min(self.width, int(u) + 21)]
            rgb = patch.astype(float)
            saturation = rgb.max(axis=2) - rgb.min(axis=2)
            mask = saturation > 80
            yy, xx = np.nonzero(mask)
            if not len(xx):
                raise ValueError("projection check rendered marker was not located")
            measured.append([float(xx.mean() + x0), float(yy.mean() + y0)])
        measured = np.asarray(measured)
        errors = np.linalg.norm(measured - predicted, axis=1)
        self.projection_check_image = image
        return {"declared_K": declared.tolist(), "actual_frustum_K": derived.tolist(),
                "maximum_K_absolute_difference": float(np.max(np.abs(derived - declared))),
                "pixel_centres": "array indices; optical centre is ((width-1)/2,(height-1)/2)",
                "camera_points_m": points.tolist(), "projected_pixels": predicted.tolist(),
                "measured_render_pixels": measured.tolist(), "pixel_errors": errors.tolist(),
                "maximum_pixel_error": float(errors.max()),
                "passed": bool(np.max(np.abs(derived - declared)) < .001 and errors.max() < .25)}

    def metadata(self):
        return {"robot_xml": str(self.robot_xml), "joint_names": list(self.joint_names),
                "joint_order_source": "named robot.xml joints joint1..joint7",
                "base_body": self.base_body_name, "flange_body": self.flange_body_name,
                "flange_frame_source": "NERO SDK feedback/FK frame verified against XML link7",
                "downstream_physical_gripper_body": "gripper_flange",
                "mujoco_version": self.mujoco.__version__,
                "camera_to_opencv_rotation": self.cv_to_gl_rotation.tolist(),
                "intrinsics": copy.deepcopy(self.intrinsics),
                "image_processing": {"crop": None, "resize": None, "undistort": False},
                "board": {"columns": self.board.columns, "rows": self.board.rows,
                          "square_size_m": self.board.square_size_m,
                          "corner_origin": "first inner corner (0,0,0)",
                          "checker_plane_z_m": 0.0,
                          "checker_bounds_x_m": [-self.board.square_size_m,
                                                 self.board.columns * self.board.square_size_m],
                          "checker_bounds_y_m": [-self.board.square_size_m,
                                                 self.board.rows * self.board.square_size_m],
                          "origin_marker_board_m": self.origin_marker_board.tolist()}}

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

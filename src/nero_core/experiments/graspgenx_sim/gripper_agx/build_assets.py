#!/usr/bin/env python3
"""Build AGX GraspGen-X assets from existing NERO geometry, entirely offline.

Requires only the existing project environment: numpy, scipy and mujoco.
The upstream Xacro is archived next to this script solely to check provenance.
No camera, CAN or robot control modules are imported.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
SOURCE = PROJECT / "mujoco_conveyor_sorting/assets/robots/nero"
ROBOT_XML = PROJECT / "mujoco_conveyor_sorting/models/robot.xml"
UPSTREAM_REVISION = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"


def vector(text, default):
    return np.fromstring(text, sep=" ") if text else np.array(default, dtype=float)


def transform(element):
    result = np.eye(4)
    result[:3, 3] = vector(element.get("pos"), [0, 0, 0])
    q = vector(element.get("quat"), [1, 0, 0, 0])
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return result


def stl_triangles(path):
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if len(data) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].astype(float)


def moved(triangles, T):
    return triangles @ T[:3, :3].T + T[:3, 3]


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_obj(path, triangles):
    vertices, faces = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("# AGX geometry in GraspGen-X canonical frame, units meters\n")
        for vertex in vertices:
            stream.write("v " + " ".join(f"{x:.10g}" for x in vertex) + "\n")
        for face in faces.reshape(-1, 3) + 1:
            stream.write("f " + " ".join(str(x) for x in face) + "\n")


def sample_surface(triangles, count=10500, seed=0):
    # Real surface samples, not the zero placeholders used by upstream fallback.
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    uv = rng.uniform(size=(count, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def add_joint(robot, name, kind, parent, child, T, axis=None, limits=None, mimic=None):
    element = ET.SubElement(robot, "joint", {"name": name, "type": kind})
    ET.SubElement(element, "parent", {"link": parent})
    ET.SubElement(element, "child", {"link": child})
    ET.SubElement(element, "origin", {
        "xyz": " ".join(f"{x:.14g}" for x in T[:3, 3]),
        "rpy": " ".join(f"{x:.14g}" for x in Rotation.from_matrix(T[:3, :3]).as_euler("xyz")),
    })
    if axis is not None:
        ET.SubElement(element, "axis", {"xyz": " ".join(f"{x:.14g}" for x in axis)})
    if limits is not None:
        ET.SubElement(element, "limit", {"lower": str(limits[0]), "upper": str(limits[1]), "effort": "10", "velocity": "3"})
    if mimic is not None:
        ET.SubElement(element, "mimic", {"joint": "gripper", "multiplier": str(mimic), "offset": "0"})


def main():
    xml = ET.parse(ROBOT_XML).getroot()
    bodies = {body.get("name"): body for body in xml.iter("body")}
    archive = ET.parse(OUT / "source/nero_with_gripper_description.xacro").getroot()
    official_joints = {joint.get("name"): joint for joint in archive.findall("joint")}
    source_checks = []
    for name in ("gripper_joint1", "gripper_joint2"):
        joint = official_joints[name]
        origin = joint.find("origin")
        official_T = np.eye(4)
        official_T[:3, :3] = Rotation.from_euler("xyz", vector(origin.get("rpy"), [0, 0, 0])).as_matrix()
        official_T[:3, 3] = vector(origin.get("xyz"), [0, 0, 0])
        body_name = "gripper_link" + name[-1]
        mjcf_T = transform(bodies[body_name])
        delta = float(np.max(np.abs(official_T - mjcf_T)))
        # The archived manufacturer description predates the adopted drawing
        # and mirrored-jaw corrections. Record that difference; derive assets
        # from the active simulator and verify them against compiled MuJoCo.
        axis_matches = bool(np.allclose(vector(joint.find("axis").get("xyz"), [0, 0, 0]),
                                        vector(bodies[body_name].find("joint").get("axis"), [0, 0, 0])))
        source_checks.append({"joint": name, "max_abs_transform_difference_m_or_matrix": delta,
                              "transform_matches": bool(delta < 1e-6), "axis_matches": axis_matches,
                              "used_as_geometry_source": False})
    official_base_origin = official_joints["gripper_base_joint"].find("origin")
    assert np.allclose(vector(official_base_origin.get("xyz"), [0, 0, 0]), transform(bodies["gripper_base"])[:3, 3])
    assert np.isclose(float(official_joints["gripper"].find("limit").get("upper")), 0.1)

    # Model gripper-base +Y opens the jaws, +Z runs from palm toward tips.
    # Canonical +X = base +Y, +Y = base -X, +Z = base +Z.
    T_base_canonical = np.eye(4)
    T_base_canonical[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T_canonical_base = np.linalg.inv(T_base_canonical)
    T_flange_base = transform(bodies["gripper_base"])
    T_base_flange = np.linalg.inv(T_flange_base)
    T_base_tcp = transform(bodies["gripper_base"].find("site[@name='ee_site']"))

    names = ["gripper_flange", "gripper_base", "gripper_link1", "gripper_link2"]
    mesh_definitions = {mesh.get('name'): mesh for mesh in xml.iter('mesh')}
    # In particular, the active palm is gripper_base_drawing_20261012.stl,
    # not the older gripper_base.stl that happens to remain in the directory.
    mesh_paths = {name: PROJECT / 'mujoco_conveyor_sorting/assets' /
                  mesh_definitions[bodies[name].find('geom').get('mesh')].get('file') for name in names}
    triangles = {name: stl_triangles(mesh_paths[name]) for name in names}

    def poses(opening):
        result = {"gripper_base": T_canonical_base, "gripper_flange": T_canonical_base @ T_base_flange}
        for index in (1, 2):
            name = f"gripper_link{index}"
            body = bodies[name]
            T = transform(body)
            q = opening / 2 if index == 1 else -opening / 2
            axis = vector(body.find("joint").get("axis"), [0, 0, 0])
            T[:3, 3] += T[:3, :3] @ axis * q
            result[name] = T_canonical_base @ T
        return result

    def mesh(opening):
        Ts = poses(opening)
        return np.concatenate([moved(triangles[name], Ts[name]) for name in names])

    max_opening = 2 * float(vector(bodies["gripper_link1"].find("joint").get("range"), [0, 0])[1])
    assert np.isclose(max_opening, 0.1)
    open_poses = poses(max_opening)
    finger_vertices = np.concatenate([moved(triangles[name], open_poses[name]).reshape(-1, 3) for name in names[-2:]])
    fixed_vertices = np.concatenate([moved(triangles[name], open_poses[name]).reshape(-1, 3) for name in names[:2]])
    fingertip_z = float(finger_vertices[:, 2].max())
    moving_zmin = float(finger_vertices[:, 2].min())
    # A documented 1 mm clearance above the palm AABB gives a conservative
    # conditioning box. This box is not a replacement for exact collision.
    sweep_zmin = max(moving_zmin, float(fixed_vertices[:, 2].max()) + 0.001)
    y_min, y_max = float(finger_vertices[:, 1].min()), float(finger_vertices[:, 1].max())
    extents = [max_opening, y_max - y_min, fingertip_z - sweep_zmin]
    offset = [0.0, (y_max + y_min) / 2, (fingertip_z + sweep_zmin) / 2]
    extents_mid = [max_opening / 2, extents[1], extents[2]]
    open_mesh = mesh(max_opening)
    bbox = [open_mesh.reshape(-1, 3).min(0).tolist(), open_mesh.reshape(-1, 3).max(0).tolist()]
    config = {
        "open": {"gripper": max_opening, "gripper_joint1": max_opening / 2, "gripper_joint2": -max_opening / 2},
        "close": {"gripper": 0.0, "gripper_joint1": 0.0, "gripper_joint2": 0.0},
        "fingertip": [0.0, 0.0, fingertip_z],
        "sweep_volume": {"extents": extents, "offset": offset, "extents2": extents_mid, "offset2": offset},
        "standoff": [0.0, extents[2] / 2],
        "links": ["canonical", *names, "gripper_link"],
        "symmetric": False,
        "type": "parallel_2f",
        "bbox": bbox,
        "base_rotation": np.eye(4).tolist(),
    }
    write_json("config.json", config)
    write_json("sweep_volume_params.json", {
        "extents_open": extents, "offset_open": offset, "extents_mid": extents_mid,
        "offset_mid": offset, "gripper_type": 0, "fingertip_depth": fingertip_z,
    })
    write_json("transforms.json", {
        "notation": "T_A_B maps coordinates from B to A; all translations in meters",
        "canonical_frame": "+X=gripper_base +Y; +Y=gripper_base -X; +Z=gripper_base +Z; origin at gripper_base",
        "T_gripper_base_canonical": T_base_canonical.tolist(),
        "T_canonical_gripper_base": T_canonical_base.tolist(),
        "T_canonical_ee_site": (T_canonical_base @ T_base_tcp).tolist(),
        "T_ee_site_canonical": np.linalg.inv(T_canonical_base @ T_base_tcp).tolist(),
        "T_gripper_flange_canonical": (T_flange_base @ T_base_canonical).tolist(),
        "T_link7_canonical": (transform(bodies["gripper_flange"]) @ T_flange_base @ T_base_canonical).tolist(),
        "usage": "T_world_ee_site = T_world_canonical @ T_canonical_ee_site. Do not use canonical position directly as TCP position.",
        "real_robot_warning": "ee_site and flange naming are model frames; this is not a measured physical TCP calibration.",
    })

    urdf = ET.Element("robot", {"name": "agx_nero_stock_100mm_graspgenx"})
    ET.SubElement(urdf, "link", {"name": "canonical"})
    for name in names:
        link = ET.SubElement(urdf, "link", {"name": name})
        for kind in ("visual", "collision"):
            element = ET.SubElement(link, kind)
            ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": f"meshes/{name}.stl", "scale": "1 1 1"})
    ET.SubElement(urdf, "link", {"name": "gripper_link"})
    add_joint(urdf, "canonical_to_base", "fixed", "canonical", "gripper_base", T_canonical_base)
    add_joint(urdf, "base_to_flange", "fixed", "gripper_base", "gripper_flange", T_base_flange)
    add_joint(urdf, "gripper", "prismatic", "gripper_base", "gripper_link", np.eye(4), [0, 0, 1], [0, max_opening])
    for index in (1, 2):
        name = f"gripper_link{index}"
        joint = bodies[name].find("joint")
        add_joint(urdf, f"gripper_joint{index}", "prismatic", "gripper_base", name, transform(bodies[name]),
                  vector(joint.get("axis"), [0, 0, 0]), vector(joint.get("range"), [0, 0]), 0.5 if index == 1 else -0.5)
    ET.indent(urdf, space="  ")
    ET.ElementTree(urdf).write(OUT / "gripper.urdf", encoding="utf-8", xml_declaration=True)

    # Parse the serialized URDF independently: verify rpy serialization,
    # prismatic axes and mimic factors, rather than checking only our source FK.
    parsed_urdf = ET.parse(OUT / "gripper.urdf").getroot()
    urdf_checks = []
    for opening in (0.0, 0.05, 0.1):
        result = {"canonical": np.eye(4)}
        pending = list(parsed_urdf.findall("joint"))
        while pending:
            remaining = []
            for joint in pending:
                parent = joint.find("parent").get("link")
                if parent not in result:
                    remaining.append(joint)
                    continue
                origin = joint.find("origin")
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler("xyz", vector(origin.get("rpy"), [0, 0, 0])).as_matrix()
                T[:3, 3] = vector(origin.get("xyz"), [0, 0, 0])
                if joint.get("type") == "prismatic":
                    mimic = joint.find("mimic")
                    q = opening if mimic is None else opening * float(mimic.get("multiplier")) + float(mimic.get("offset"))
                    T[:3, 3] += T[:3, :3] @ vector(joint.find("axis").get("xyz"), [0, 0, 0]) * q
                result[joint.find("child").get("link")] = result[parent] @ T
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic generated URDF")
            pending = remaining
        for name, expected in poses(opening).items():
            error = float(np.max(np.abs(result[name] - expected)))
            assert error < 1e-10, ("URDF FK", opening, name, error)
            urdf_checks.append({"opening_m": opening, "body": name, "max_abs_transform_difference": error})
    (OUT / "meshes").mkdir(exist_ok=True)
    for name in names:
        shutil.copy2(mesh_paths[name], OUT / "meshes" / f"{name}.stl")
    shutil.copy2(SOURCE / "LICENSE", OUT / "meshes/LICENSE")
    write_obj(OUT / "vis_mesh.obj", open_mesh)
    write_obj(OUT / "coll_mesh.obj", open_mesh)
    write_json("points.json", {"open": sample_surface(open_mesh, seed=0).tolist(), "close": sample_surface(mesh(0), seed=1).tolist()})

    # Independent checks against MuJoCo's forward kinematics at three widths.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(PROJECT / "mujoco_conveyor_sorting/models/scene.xml"))
    data = mujoco.MjData(model)
    errors = []
    for opening in (0.0, 0.05, 0.1):
        for index in (1, 2):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper_joint{index}")
            data.qpos[model.jnt_qposadr[joint_id]] = opening / 2 * (1 if index == 1 else -1)
        mujoco.mj_forward(model, data)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = data.xmat[base_id].reshape(3, 3)
        T_world_base[:3, 3] = data.xpos[base_id]
        analytic = poses(opening)
        for name in names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            T_world_body = np.eye(4)
            T_world_body[:3, :3] = data.xmat[body_id].reshape(3, 3)
            T_world_body[:3, 3] = data.xpos[body_id]
            expected = T_canonical_base @ np.linalg.inv(T_world_base) @ T_world_body
            error = float(np.max(np.abs(expected - analytic[name])))
            assert error < 1e-10, (opening, name, error)
            errors.append({"opening_m": opening, "body": name, "max_abs_transform_difference": error})
        left_inner = analytic["gripper_link1"][:3, 3][0]
        right_inner = analytic["gripper_link2"][:3, 3][0]
        assert np.isclose(left_inner - right_inner, opening)
    write_json("geometry_check.json", {
        "status": "passed", "upstream_xacro_checks": source_checks, "mujoco_fk_checks": errors,
        "serialized_urdf_fk_checks": urdf_checks,
        "max_opening_m": max_opening, "fingertip_z_m": fingertip_z,
        "ee_site_z_m": float(T_base_tcp[2, 3]), "fingertip_to_ee_site_m": fingertip_z - T_base_tcp[2, 3],
        "full_moving_mesh_z_min_m": moving_zmin, "conditioning_z_min_m": sweep_zmin,
        "palm_clearance_m": 0.001, "open_geometry_bbox_m": bbox,
        "geometry_source": "active_simulator_robot_xml_and_its_referenced_meshes",
        "robot_xml_sha256": hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest(),
        "source_mesh_paths": {name: str(path.relative_to(PROJECT)) for name, path in mesh_paths.items()},
        "source_mesh_sha256": {name: hashlib.sha256(mesh_paths[name].read_bytes()).hexdigest() for name in names},
        "scope": "Offline geometry/axis/FK verification; not a grasp-success or physical TCP measurement.",
    })
    print(json.dumps({"status": "passed", "output": str(OUT), "sweep_volume": config["sweep_volume"], "fingertip_depth": fingertip_z}, indent=2))


if __name__ == "__main__":
    main()

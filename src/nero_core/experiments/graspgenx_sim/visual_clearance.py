"""CPU visual-point penetration checks against two separate open AGX fingers.

Only supplied observed points and a supplied gripper pose are queried. Object
CAD, simulator object poses and contact truth are never inputs to this module.
The gripper geometry is the existing STL/XML model at its full open limits.

Each convex hull has normalized halfspaces n_i*x+b_i <= 0. For a point x,
signed_penetration(x) = -max_i(n_i*x+b_i). It is positive inside and is the
shortest distance to the boundary for an interior point. A penetration margin
m >= 0 means flagging only signed_penetration > m, i.e. the inward-eroded
halfspaces n_i*x+b_i < -m. Negative values are NOT Euclidean exterior distance.
This never fills the jaw gap by taking the hull of the combined fingers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from experiments.graspgenx_sim.gripper_agx.build_assets import stl_triangles


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "gripper_agx"
LIMITATION = ("Partial single-view depth can expose observed surface penetration; "
              "zero observed penetration does not certify hidden-surface, palm, "
              "environment or continuous-path clearance.")


def _pose_transform(element):
    result = np.eye(4)
    result[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    q = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return result


def rigid_transform(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.], atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(transform[:3, :3]) - 1.) > 1e-5):
        raise ValueError("require a finite rigid T_world_canonical in metres")
    transform = transform.copy()
    # Network matrices originate in float32. Match the existing controller's
    # matrix -> normalized quaternion conversion rather than treating its
    # small rounding error as a physical nonrigid scale/shear.
    transform[:3, :3] = Rotation.from_matrix(transform[:3, :3]).as_matrix()
    return transform


@dataclass(frozen=True)
class FingerHull:
    name: str
    vertices_canonical: np.ndarray
    equations: np.ndarray

    @classmethod
    def from_vertices(cls, name, vertices):
        vertices = np.asarray(vertices, dtype=float)
        hull = ConvexHull(vertices)
        equations = hull.equations.copy()
        equations /= np.linalg.norm(equations[:, :3], axis=1)[:, None]
        vertices = vertices[hull.vertices].copy()
        equations.setflags(write=False)
        vertices.setflags(write=False)
        return cls(name, vertices, equations)

    def signed_penetration(self, points_canonical):
        return -(points_canonical @ self.equations[:, :3].T + self.equations[:, 3]).max(axis=1)

    def _summary_depths(self, points_canonical, penetration_margin_m):
        """Evaluate exactly the points that can affect ``inspect``'s summary.

        A subset of the original hull halfspaces gives an upper bound on each
        point's signed penetration. Keep every possible collision and every
        point that can improve an already evaluated maximum. Values omitted
        below are only noncolliding, nonmaximal points; ``signed_penetration``
        remains the full per-point API. In particular, negative maxima keep
        their original halfspace meaning rather than becoming box distances.
        """
        points = np.asarray(points_canonical, dtype=float)
        if not len(points):
            return np.empty(0)
        normals = self.equations[:, :3]
        indices = np.unique(np.r_[normals.argmax(axis=0), normals.argmin(axis=0)])
        planes = self.equations[indices]
        coarse = points @ planes[:, :3].T + planes[:, 3]
        upper = -coarse.max(axis=1)
        # Matrix kernels may round a selected plane's dot product differently
        # from the full matrix. Bound that arithmetic error outwards; this
        # broad phase never changes the final penetration comparison.
        roundoff = 16. * np.finfo(float).eps * (
            np.abs(points) @ np.abs(planes[:, :3]).T + np.abs(planes[:, 3])).max(axis=1)
        first = int(np.argmax(upper))
        verified = self.signed_penetration(points[first:first + 1])[0]
        possible = upper + roundoff >= min(float(penetration_margin_m), verified)
        possible[first] = True
        depths = np.full(len(points), -np.inf)
        depths[possible] = self.signed_penetration(points[possible])
        return depths


class OpenFingerClearance:
    def __init__(self, fingers, *, metadata=None, canonical_to_tcp=None):
        self.fingers = tuple(fingers)
        if len(self.fingers) != 2:
            raise ValueError("require exactly two independent finger hulls")
        self.metadata = metadata or {}
        self.canonical_to_tcp = canonical_to_tcp

    def penetration_depths(self, points_world, transform_world_canonical):
        points = np.asarray(points_world, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("require finite observed Nx3 world points in metres")
        transform = rigid_transform(transform_world_canonical)
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        if not len(points):
            return np.empty((2, 0))
        return np.stack([finger.signed_penetration(local) for finger in self.fingers])

    def inspect(self, points_world, transform_world_canonical, *, penetration_margin_m=0.,
                include_point_indices=False):
        if not np.isfinite(penetration_margin_m) or penetration_margin_m < 0:
            raise ValueError("penetration_margin_m must be finite and nonnegative")
        points = np.asarray(points_world, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("require finite observed Nx3 world points in metres")
        transform = rigid_transform(transform_world_canonical)
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        depths = np.stack([finger._summary_depths(local, penetration_margin_m)
                           for finger in self.fingers])
        observed = np.max(depths, axis=0)
        detected = observed > penetration_margin_m
        result = {"point_count": int(depths.shape[1]), "collision_point_count": int(detected.sum()),
                  "penetration_margin_m": float(penetration_margin_m),
                  "minimum_detected_penetration_m": float(observed[detected].min()) if detected.any() else None,
                  "maximum_detected_penetration_m": float(observed[detected].max()) if detected.any() else None,
                  "maximum_signed_penetration_m": float(observed.max()) if observed.size else None,
                  "fingers": [], "partial_depth_cannot_certify_hidden_free": True}
        for finger, values in zip(self.fingers, depths):
            inside = values > penetration_margin_m
            row = {"name": finger.name, "collision_point_count": int(inside.sum()),
                   "minimum_detected_penetration_m": float(values[inside].min()) if inside.any() else None,
                   "maximum_detected_penetration_m": float(values[inside].max()) if inside.any() else None,
                   "maximum_signed_penetration_m": float(values.max()) if values.size else None}
            if include_point_indices:
                row["point_indices"] = np.flatnonzero(inside).tolist()
            result["fingers"].append(row)
        if include_point_indices:
            result["point_indices"] = np.flatnonzero(detected).tolist()
        return result

    def vertical_descent(self, points_world, transform_world_canonical, *, lift_m=.1,
                         step_m=.01, penetration_margin_m=0., include_point_indices=False,
                         lift_vector_world=None):
        if not np.isfinite([lift_m, step_m]).all() or lift_m < 0 or step_m <= 0:
            raise ValueError("require lift_m >= 0 and step_m > 0")
        transform = rigid_transform(transform_world_canonical)
        lift_vector = np.array([0., 0., lift_m]) if lift_vector_world is None else np.asarray(lift_vector_world, dtype=float)
        if lift_vector.shape != (3,) or not np.isfinite(lift_vector).all():
            raise ValueError("lift vector must be finite XYZ")
        offsets = np.linspace(1., 0., int(np.ceil(np.linalg.norm(lift_vector) / step_m)) + 1)
        samples = []
        for offset in offsets:
            pose = transform.copy()
            pose[:3, 3] += offset * lift_vector
            samples.append({"world_z_offset_m": float(offset * lift_vector[2]), **self.inspect(
                points_world, pose, penetration_margin_m=penetration_margin_m,
                include_point_indices=include_point_indices)})
        positive = [row for row in samples if row["collision_point_count"]]
        signed = [row["maximum_signed_penetration_m"] for row in samples
                  if row["maximum_signed_penetration_m"] is not None]
        return {"direction": ("world +Z lift, descending vertically with fixed orientation"
                              if lift_vector_world is None else "specified pregrasp to capture, fixed orientation"),
                "lift_vector_world_m": lift_vector.tolist(),
                "lift_m": float(lift_m), "maximum_sample_spacing_m": float(step_m),
                "penetration_margin_m": float(penetration_margin_m), "sample_count": len(samples),
                "collision_sample_count": len(positive),
                "peak_sample_collision_point_count": max(row["collision_point_count"] for row in samples),
                "first_collision_offset_m": positive[0]["world_z_offset_m"] if positive else None,
                "minimum_detected_penetration_m": min(row["minimum_detected_penetration_m"] for row in positive) if positive else None,
                "maximum_detected_penetration_m": max(row["maximum_detected_penetration_m"] for row in positive) if positive else None,
                "maximum_signed_penetration_m": max(signed) if signed else None,
                "samples": samples, "discrete_samples_do_not_certify_continuous_free": True,
                "partial_depth_cannot_certify_hidden_free": True}


@lru_cache(maxsize=1)
def load_open_fingers():
    robot_path = ROOT / "mujoco_conveyor_sorting/models/robot.xml"
    transforms_path = ASSETS / "transforms.json"
    xml = ET.parse(robot_path).getroot()
    transforms = json.loads(transforms_path.read_text())
    canonical_base = np.asarray(transforms["T_canonical_gripper_base"])
    bodies = {body.get("name"): body for body in xml.iter("body")}
    meshes = {mesh.get("name"): mesh for mesh in xml.findall("asset/mesh")}
    fingers, details = [], []
    for name, collision_name in (("gripper_link1", "left_finger_geom"),
                                 ("gripper_link2", "right_finger_geom")):
        body = bodies[name]
        joint = body.find("joint")
        if joint.get("type") != "slide":
            raise ValueError("expected existing prismatic AGX finger joints")
        limits = np.fromstring(joint.get("range"), sep=" ")
        opened = float(limits[np.argmax(np.abs(limits))])
        local = _pose_transform(body)
        local[:3, 3] += local[:3, :3] @ np.fromstring(joint.get("axis"), sep=" ") * opened
        geom = body.find(f"geom[@name='{collision_name}']")
        mesh = meshes[geom.get("mesh")]
        if mesh.get("maxhullvert", "-1") != "-1":
            raise ValueError("truncated collision hull needs explicit compiled-graph support")
        path = ROOT / "mujoco_conveyor_sorting/assets" / mesh.get("file")
        vertices = np.unique(stl_triangles(path).reshape(-1, 3), axis=0)
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        if mesh.get("refpos") or mesh.get("refquat"):
            raise ValueError("nondefault mesh asset reference transform needs explicit handling")
        transform = canonical_base @ local @ _pose_transform(geom)
        canonical = vertices @ transform[:3, :3].T + transform[:3, 3]
        finger = FingerHull.from_vertices(collision_name, canonical)
        fingers.append(finger)
        details.append({"name": collision_name, "source_stl": str(path),
                        "source_stl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "joint_open_position_m": opened,
                        "T_canonical_raw_mesh": transform.tolist(),
                        "raw_unique_vertices": len(vertices), "hull_vertices": len(finger.vertices_canonical),
                        "hull_triangular_facets": len(finger.equations),
                        "lower_canonical_m": canonical.min(axis=0).tolist(),
                        "upper_canonical_m": canonical.max(axis=0).tolist()})
    metadata = {"geometry": "two separate full convex hulls of existing AGX finger STL meshes",
                "robot_xml_sha256": hashlib.sha256(robot_path.read_bytes()).hexdigest(),
                "transforms_sha256": hashlib.sha256(transforms_path.read_bytes()).hexdigest(),
                "fingers": details,
                "margin_definition": "flag only -max_i(unit_normal_i dot point + offset_i) > penetration_margin_m; positive margin erodes each finger hull",
                "default_margin_m": 0., "runtime_geometry_contact_margin_is_not_added": True,
                "rotation_handling": "same scipy SO(3) matrix-to-quaternion normalization as the controller",
                "limitation": LIMITATION}
    return OpenFingerClearance(fingers, metadata=metadata,
                              canonical_to_tcp=np.asarray(transforms["T_canonical_ee_site"]))


def diagnose_recording(recording, *, all_poses_track=2):
    recording = Path(recording).resolve()
    geometry = load_open_fingers()
    tracks = []
    for input_path in sorted((recording / "inputs").glob("track_*_try_01.npz")):
        name = input_path.stem
        track_id = int(name.split("_")[1])
        points = np.load(input_path)["points"].astype(float)
        archive = np.load(recording / "inference" / f"{name}.npz")
        metadata = json.loads((recording / "inference" / f"{name}.json").read_text())
        accepted = metadata.get("accepted_candidates", [])
        row = {"track_id": track_id, "input_npz": str(input_path),
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "point_count": len(points), "network_pose_count": len(archive["poses"]), "selected": None}
        if accepted:
            selected = accepted[0]
            pose = np.asarray(selected["T_world_canonical"])
            row["selected"] = {"source_index": selected["source_index"], "variant": selected["variant"],
                               "score": selected["score"],
                               "endpoint": geometry.inspect(points, pose, include_point_indices=True),
                               "vertical_descent": geometry.vertical_descent(points, pose, include_point_indices=True)}
        if track_id == all_poses_track:
            rows = []
            for index, (pose, score) in enumerate(zip(archive["poses"], archive["scores"])):
                try:
                    endpoint = geometry.inspect(points, pose, include_point_indices=True)
                    path = geometry.vertical_descent(points, pose, include_point_indices=True)
                    rows.append({"source_index": index, "score": float(score),
                                 "endpoint": endpoint, "vertical_descent": path})
                except ValueError as error:
                    rows.append({"source_index": index, "error": str(error)})
            row["all_original_network_poses"] = rows
        tracks.append(row)
    return {"purpose": "CPU recorded visual-cloud/open-finger penetration diagnostic; no candidate admission or rescoring",
            "recording": str(recording), "object_truth_or_cad_used": False, "gpu_or_hardware_used": False,
            "gripper_geometry": geometry.metadata, "all_poses_track": all_poses_track,
            "penetration_margin_m": 0., "threshold_selected_for_rejection": False,
            "world_vertical_lift_m": .1, "maximum_path_sample_step_m": .01,
            "tracks": tracks, "limitations": [LIMITATION,
                "Capture-time cloud and candidate pose are compared in the same world frame; no object-motion/roll truth is replayed.",
                "Only two open fingers are checked; palm/arm/environment and subsequent closure require their original full validation."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--all-poses-track", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = diagnose_recording(args.recording, all_poses_track=args.all_poses_track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "tracks": [
        {"track_id": row["track_id"], "selected_source_index": row["selected"]["source_index"],
         "endpoint_count": row["selected"]["endpoint"]["collision_point_count"],
         "vertical_path_colliding_samples": row["selected"]["vertical_descent"]["collision_sample_count"]}
        for row in report["tracks"] if row["selected"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

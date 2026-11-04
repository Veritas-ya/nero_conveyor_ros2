"""Optional parallel cuRobo IK proposals for the existing planning worker.

cuRobo runs in a persistent, isolated Python process. The normal runtime does
not import torch or alter its environment. Proposals are deliberately NOT
certificates: the caller must retain its MuJoCo, measured-scene, continuation,
joint-limit and trajectory checks before installing any returned route.

The exported URDF comes from the *compiled current model*, including the TCP
site transform. It does not substitute a vendor flange or another gripper.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


def export_mujoco_chain_urdf(model: Any, joint_names: Sequence[str],
                             ee_site_name: str) -> str:
    """Export hinge kinematics with joint anchors and qpos references intact."""
    import mujoco
    from scipy.spatial.transform import Rotation

    names = tuple(str(name) for name in joint_names)
    selected = {int(model.joint(name).id): name for name in names}
    site = int(model.site(ee_site_name).id)
    chain: list[int] = []
    body = int(model.site_bodyid[site])
    while body:
        chain.append(body)
        body = int(model.body_parentid[body])
    chain.reverse()
    robot = ET.Element("robot", name="nero_current_mujoco_kinematics")
    ET.SubElement(robot, "link", name="mujoco_world")

    def values(array: Any) -> str:
        return " ".join(format(float(x), ".17g") for x in array)

    def rotation(quaternion: Any) -> Any:
        q = np.asarray(quaternion, dtype=float)
        return Rotation.from_quat(q[[1, 2, 3, 0]])

    def origin(element: Any, xyz: Any, rot: Any) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rpy = rot.as_euler("xyz")
        ET.SubElement(element, "origin", xyz=values(xyz), rpy=values(rpy))

    parent = "mujoco_world"
    exported: list[str] = []
    for body in chain:
        body_name = str(model.body(body).name)
        joint_ids = list(range(int(model.body_jntadr[body]),
                               int(model.body_jntadr[body]+model.body_jntnum[body])))
        if len(joint_ids) > 1:
            raise ValueError("cuRobo chain export requires at most one joint per body")
        ET.SubElement(robot, "link", name=body_name)
        body_rot = rotation(model.body_quat[body])
        body_pos = np.asarray(model.body_pos[body], dtype=float)
        if joint_ids:
            jid = joint_ids[0]
            if jid not in selected or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("cuRobo TCP chain contains an unselected/non-hinge joint")
            name = selected[jid]
            pivot = body_name+"__joint_pivot"
            ET.SubElement(robot, "link", name=pivot)
            joint = ET.SubElement(robot, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=pivot)
            anchor = np.asarray(model.jnt_pos[jid], dtype=float)
            axis = np.asarray(model.jnt_axis[jid], dtype=float)
            reference = float(model.qpos0[int(model.jnt_qposadr[jid])])
            origin(joint, body_pos+body_rot.apply(anchor),
                   body_rot*Rotation.from_rotvec(-reference*axis))
            ET.SubElement(joint, "axis", xyz=values(axis))
            lower, upper = model.jnt_range[jid]
            ET.SubElement(joint, "limit", lower=str(float(lower)),
                          upper=str(float(upper)), velocity="2", effort="100")
            fixed = ET.SubElement(robot, "joint", name=name+"__body", type="fixed")
            ET.SubElement(fixed, "parent", link=pivot)
            ET.SubElement(fixed, "child", link=body_name)
            origin(fixed, -anchor, Rotation.identity())
            exported.append(name)
        else:
            fixed = ET.SubElement(robot, "joint", name=body_name+"__fixed", type="fixed")
            ET.SubElement(fixed, "parent", link=parent)
            ET.SubElement(fixed, "child", link=body_name)
            origin(fixed, body_pos, body_rot)
        parent = body_name
    if tuple(exported) != names:
        raise ValueError("cuRobo joint ordering differs from the current robot chain")
    ET.SubElement(robot, "link", name="nero_tcp")
    fixed = ET.SubElement(robot, "joint", name="nero_tcp__fixed", type="fixed")
    ET.SubElement(fixed, "parent", link=parent)
    ET.SubElement(fixed, "child", link="nero_tcp")
    origin(fixed, model.site_pos[site], rotation(model.site_quat[site]))
    return ET.tostring(robot, encoding="unicode")


def conservative_robot_spheres(model: Any, ee_site_name: str,
                               collision_geom_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Cover true mesh AABBs with sphere-covered boxes for GPU search only.

    Off-chain gripper links are transformed into their closest fixed chain
    ancestor, taking the envelope over their complete slide-joint travel.
    This may reject a useful GPU proposal; exact original checks decide the
    final route, and the existing planner remains available.
    """
    import mujoco
    data = mujoco.MjData(model)
    chain = set()
    body = int(model.site_bodyid[model.site(ee_site_name).id])
    while body:
        chain.add(body); body = int(model.body_parentid[body])
    geometry_ids = (range(model.ngeom) if collision_geom_ids is None else collision_geom_ids)
    spheres: dict[str, list[dict[str, Any]]] = {}
    ranks: dict[str, int] = {}
    for geom in geometry_ids:
        geom = int(geom)
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom]); anchor = body
        extra_joints = []
        while anchor and anchor not in chain:
            extra_joints.extend(range(int(model.body_jntadr[anchor]),
                int(model.body_jntadr[anchor]+model.body_jntnum[anchor])))
            anchor = int(model.body_parentid[anchor])
        if anchor not in chain:
            continue
        mesh = int(model.geom_dataid[geom])
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh < 0:
            continue
        begin, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = np.asarray(model.mesh_vert[begin:begin+count], dtype=float)
        local_sets = []
        corners = [model.jnt_range[joint] for joint in extra_joints]
        for values in itertools.product(*corners):
            data.qpos[:] = model.qpos0
            for joint, value in zip(extra_joints, values):
                data.qpos[int(model.jnt_qposadr[joint])] = value
            mujoco.mj_kinematics(model, data)
            world = vertices@data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
            local_sets.append((world-data.xpos[anchor])@data.xmat[anchor].reshape(3, 3))
        points = np.concatenate(local_sets)
        lower, upper = points.min(axis=0), points.max(axis=0)
        axis = int(np.argmax(upper-lower))
        segments = max(1, int(np.ceil((upper[axis]-lower[axis])/.04)))
        size = upper-lower; size[axis] /= segments
        radius = max(float(np.linalg.norm(size*.5)), .001)
        name = str(model.body(anchor).name)
        for index in range(segments):
            center = (lower+upper)*.5
            center[axis] = lower[axis]+(index+.5)*size[axis]
            spheres.setdefault(name, []).append({"center": center.tolist(), "radius": radius})
        rank, parent = 0, anchor
        while parent:
            rank += int(model.body_jntnum[parent])
            parent = int(model.body_parentid[parent])
        ranks[name] = rank
    names = list(spheres)
    ignore = {name: [other for other in names[index+1:] if abs(ranks[name]-ranks[other]) <= 2]
              for index, name in enumerate(names)}
    return {"collision_link_names": names, "collision_spheres": spheres,
        "self_collision_ignore": ignore,
        "statistics": {"sphere_count": sum(map(len, spheres.values())),
            "link_count": len(names), "coverage": "segmented_mesh_AABB_spheres",
            "gripper": "complete_slide_joint_travel_envelope"}}


def occupied_voxel_world(points: Any, voxel_size: float = .02,
                         maximum_cuboids: int = 4096) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cover all observed points with occupied cells; do not invent free space."""
    array = np.asarray(points, dtype=float).reshape(-1, 3)
    array = array[np.all(np.isfinite(array), axis=1)]
    size = float(voxel_size)
    cells = np.unique(np.floor(array/size).astype(np.int64), axis=0)
    while len(cells) > maximum_cuboids:
        size *= 1.25
        cells = np.unique(np.floor(array/size).astype(np.int64), axis=0)
    cubes = {"observed_"+str(index): {"dims": [size]*3,
        "pose": [*((cell.astype(float)+.5)*size).tolist(), 1., 0., 0., 0.]}
        for index, cell in enumerate(cells)}
    return {"cuboid": cubes}, {"input_points": len(array), "occupied_cells": len(cells),
        "voxel_size_m": size, "coverage": "all_finite_observed_points_covered",
        "map_type": "occupied_cuboids_not_nvblox_esdf"}


def observed_distance_field(points: Any, *, voxel_size: float = .02,
                            dims: Sequence[float] = (3.2, 3.2, 3.2)) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Conservative distance to observed occupied cells, for GPU proposals.

    This is a single-frame occupancy distance field, not nvblox depth fusion.
    Negative values mean distance from observed cells; they do not certify
    that unseen space is empty. Outside-grid points are returned for OBB cover.
    """
    from scipy.ndimage import distance_transform_edt
    started = time.perf_counter()
    array = np.asarray(points, dtype=float).reshape(-1, 3)
    array = array[np.all(np.isfinite(array), axis=1)]
    dimensions = np.asarray(dims, dtype=float)
    shape = np.rint(dimensions/voxel_size).astype(int)+1
    indices = np.floor((array+dimensions*.5)/voxel_size).astype(int)
    inside = np.all((indices >= 0) & (indices < shape), axis=1)
    occupied = np.zeros(tuple(shape), dtype=bool)
    selected = indices[inside]
    if len(selected):
        occupied[tuple(selected.T)] = True
        # Two half-diagonals cover cell-centre quantization and trilinear
        # interpolation. Native observed points are never removed.
        values = np.sqrt(3.)*voxel_size-distance_transform_edt(~occupied, sampling=voxel_size)
        field = np.asarray(np.clip(values, -1., 1.), dtype=np.float32)
    else:
        field = np.full(tuple(shape), -1., dtype=np.float32)
    return field, {"input_points": len(array), "inside_grid_points": int(inside.sum()),
        "outside_grid_points": int((~inside).sum()), "occupied_cells": int(occupied.sum()),
        "voxel_size_m": float(voxel_size), "dims_m": dimensions.tolist(),
        "grid_shape": shape.tolist(), "grid_values": int(field.size),
        "distance_field_wall_s": time.perf_counter()-started,
        "map_type": "observed_occupancy_distance_field_not_nvblox_fusion",
        "coverage": "all_native_points_retained; outside_grid_use_cuboids"}, array[~inside]


class CuRoboCandidatePathBackend:
    """Generate GPU MotionGen/IK routes, leaving exact certification to caller."""

    def __init__(self, model: Any, *, joint_names: Sequence[str], ee_site_name: str,
                 python_executable: str | Path | None = None,
                 num_seeds: int = 32, max_candidates: int = 8,
                 position_tolerance: float = 5e-4,
                 orientation_tolerance: float = 2e-3,
                 scene_points: Any = None,
                 collision_geom_ids: Sequence[int] | None = None,
                 world_representation: str = "voxel") -> None:
        import mujoco
        self.joint_names = tuple(joint_names)
        self.num_seeds = int(num_seeds)
        self.max_candidates = int(max_candidates)
        configured = python_executable or os.environ.get("NERO_CUROBO_PYTHON")
        self.python = Path(configured).expanduser() if configured else (
            Path.home()/".cache/nero-planning/venv/bin/python")
        self.urdf = export_mujoco_chain_urdf(model, joint_names, ee_site_name)
        self.model_fingerprint = hashlib.sha256(self.urdf.encode()).hexdigest()
        self._model = copy.copy(model)
        self._data = mujoco.MjData(self._model)
        self._indices = [int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names]
        self._site_id = int(model.site(ee_site_name).id)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.robot_spheres = conservative_robot_spheres(model, ee_site_name, collision_geom_ids)
        if world_representation not in ("voxel", "obb"):
            raise ValueError("cuRobo proposal world must be voxel or obb")
        self.world_representation = world_representation
        self.world, self.world_statistics = self._prepare_world(
            np.empty((0, 3)) if scene_points is None else scene_points)
        self.diagnostics: dict[str, Any] = {"backend": "curobo_motiongen_and_parallel_ik",
            "certification": "required_by_existing_worker", "available": False}
        self._process: subprocess.Popen[str] | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._stderr: Any = None
        self._lock = threading.Lock()
        self._disabled_reason: str | None = None
        atexit.register(self.close)

    def _prepare_world(self, points: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.world_representation == "obb":
            return occupied_voxel_world(points)
        array = np.asarray(points, dtype=float).reshape(-1, 3)
        return {"voxel_points": array.tolist(), "voxel_size": .02,
            "dims": [3.2, 3.2, 3.2]}, {"input_points": len(array),
                "map_type": "observed_occupancy_distance_field_not_nvblox_fusion"}

    def _start(self, cancelled: Callable[[], bool]) -> bool:
        if self._disabled_reason:
            self.diagnostics.update(available=False, unavailable_reason=self._disabled_reason)
            return False
        if self._process is not None:
            return self._process.poll() is None
        if not self.python.is_file():
            self._disabled_reason = "isolated cuRobo Python is not installed: "+str(self.python)
            self.diagnostics.update(available=False, unavailable_reason=self._disabled_reason)
            return False
        self._temporary = tempfile.TemporaryDirectory(prefix="nero-curobo-")
        folder = Path(self._temporary.name)
        (folder/"robot.urdf").write_text(self.urdf)
        (folder/"backend.json").write_text(json.dumps({"robot_spheres": self.robot_spheres,
            "world": self.world, "world_statistics": self.world_statistics}))
        self._stderr = (folder/"server.log").open("w")
        environment = os.environ.copy()
        toolkit = Path.home()/".cache/nero-planning/toolchain/usr/local/cuda-12.1"
        if toolkit.is_dir():
            environment.setdefault("CUDA_HOME", str(toolkit))
        environment.setdefault("MAX_JOBS", "2")
        environment.setdefault("PYTHONUNBUFFERED", "1")
        headers = Path.home()/".cache/nero-planning/python-dev/usr/include"
        if headers.is_dir():
            environment.setdefault("CPATH", str(headers)+":"+str(headers/"python3.10"))
        started = time.perf_counter()
        self._process = subprocess.Popen([str(self.python), str(Path(__file__).resolve()),
            "--server", str(folder/"robot.urdf"), str(self.num_seeds),
            str(self.max_candidates)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, bufsize=1, env=environment)
        ready = self._read(cancelled)
        self.diagnostics["startup_wall_s"] = time.perf_counter()-started
        if ready is None or not ready.get("ready"):
            self._disabled_reason = (ready or {}).get("error", "cuRobo process did not initialize")
            if self._temporary is not None:
                lines = (folder/"server.log").read_text(errors="replace").splitlines()
                self.diagnostics["startup_log_tail"] = lines[-12:]
            self.close()
            self.diagnostics.update(available=False, unavailable_reason=self._disabled_reason)
            return False
        self.diagnostics.update(available=True, runtime=ready)
        if ready.get("world"):
            self.world_statistics = ready["world"]
        return True

    def _read(self, cancelled: Callable[[], bool]) -> dict[str, Any] | None:
        process = self._process
        if process is None or process.stdout is None:
            return None
        # Cooperative cancellation is request ownership, not an execution
        # latency threshold. No new hardware-stop timeout is introduced.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if cancelled():
                    self.close()
                    return None
                if selector.select(.025):
                    line = process.stdout.readline()
                    if not line:
                        return None
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
                if process.poll() is not None:
                    return None

    def warmup(self, cancelled: Callable[[], bool] = lambda: False) -> bool:
        """Initialize once during device/model setup, before recognizing a part."""
        with self._lock:
            return self._start(cancelled)

    def update_scene(self, points: Any) -> dict[str, Any]:
        """Refresh the GPU proposal map without restarting CUDA/recapturing IK."""
        with self._lock:
            started = time.perf_counter()
            self.world, self.world_statistics = self._prepare_world(points)
            if self._process is not None and self._process.poll() is None:
                assert self._process.stdin is not None
                self._process.stdin.write(json.dumps({"operation": "update_world", "world": self.world})+"\n")
                self._process.stdin.flush()
                result = self._read(lambda: False)
                if result is None or result.get("error"):
                    self.diagnostics["world_update_error"] = (result or {}).get("error", "service exited")
                    self.close()
                elif result.get("world_statistics"):
                    self.world_statistics = result["world_statistics"]
            return {**self.world_statistics, "update_wall_s": time.perf_counter()-started}

    def probe_scene(self, points: Any, radius_m: float = .001) -> dict[str, Any]:
        """Read-only diagnostic of GPU collision values at supplied base points."""
        with self._lock:
            if not self._start(lambda: False):
                return {"error": "cuRobo service unavailable"}
            assert self._process is not None and self._process.stdin is not None
            array = np.asarray(points, dtype=float).reshape(-1, 3)
            self._process.stdin.write(json.dumps({"operation": "scene_query",
                "points": array.tolist(), "radius_m": float(radius_m)})+"\n")
            self._process.stdin.flush()
            return self._read(lambda: False) or {"error": "cuRobo service exited"}

    def propose(self, request: Any, cancelled: Callable[[], bool]) -> list[np.ndarray]:
        with self._lock:
            started = time.perf_counter()
            self.diagnostics = {"backend": "curobo_motiongen_and_parallel_ik",
                "certification": "required_by_existing_worker", "available": False,
                "model_urdf_sha256": self.model_fingerprint}
            try:
                if cancelled() or not self._start(cancelled):
                    return []
                message = {"target_poses": np.asarray(request.target_poses).tolist(),
                    "start_q": np.asarray(request.start_q).tolist(),
                    "seeds": np.asarray(request.goal_seeds).tolist(),
                    "lower_limits": np.asarray(request.lower_limits).tolist(),
                    "upper_limits": np.asarray(request.upper_limits).tolist()}
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(json.dumps(message, allow_nan=False)+"\n")
                self._process.stdin.flush()
                response = self._read(cancelled)
                if response is None or response.get("error"):
                    self.diagnostics["error"] = (response or {}).get("error", "cuRobo process exited/cancelled")
                    if not cancelled():
                        self._disabled_reason = self.diagnostics["error"]
                    self.close()
                    return []
                self.diagnostics.update(response.get("diagnostics", {}))
                self.diagnostics.update(world=self.world_statistics,
                    robot_spheres=self.robot_spheres["statistics"])
                paths = []
                mismatches = []
                for item in response.get("candidates", []):
                    route = np.asarray(item.get("path", [request.start_q, item.get("q")]), dtype=float)
                    if route.ndim != 2 or route.shape[1:] != (7,) or len(route) < 2:
                        continue
                    q = route[-1]
                    if q.shape != (7,) or not np.all(np.isfinite(q)):
                        continue
                    # Reject proposals outside this exact snapshot's command
                    # envelope; never clip a GPU solution into another pose.
                    if np.any(q < request.lower_limits) or np.any(q > request.upper_limits):
                        continue
                    # Use the existing model to check pose convergence too:
                    # cuRobo quaternion-distance thresholds are not the same
                    # quantity as the DLS angular-error tolerance in radians.
                    import mujoco
                    self._data.qpos[:] = request.full_qpos
                    self._data.qpos[self._indices] = q
                    mujoco.mj_kinematics(self._model, self._data)
                    pose = np.asarray(request.target_poses[int(item["pose_index"])])
                    quaternion = np.empty(4)
                    mujoco.mju_mat2Quat(quaternion, self._data.site_xmat[self._site_id])
                    position_error = float(np.linalg.norm(self._data.site_xpos[self._site_id]-pose[:3]))
                    orientation_error = 2.*float(np.arccos(np.clip(abs(quaternion@pose[3:]), 0., 1.)))
                    if position_error > self.position_tolerance or orientation_error > self.orientation_tolerance:
                        mismatches.append({"position_error_m": position_error,
                                           "orientation_error_rad": orientation_error})
                        continue
                    route[0] = request.start_q
                    paths.append(route)
                self.diagnostics["fk_mismatches"] = mismatches
                self.diagnostics["candidate_count"] = len(paths)
                return paths
            except (OSError, ValueError, RuntimeError) as error:
                self.diagnostics["error"] = str(error)
                self.close()
                return []
            finally:
                self.diagnostics["proposal_wall_s"] = time.perf_counter()-started

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def _server(urdf: str, num_seeds: int, max_candidates: int) -> None:
    """Private line-delimited service. stdout is protocol only."""
    import contextlib
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.geom.types import WorldConfig, VoxelGrid
        from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        backend_config = json.loads((Path(urdf).parent/"backend.json").read_text())
        tensor_args = TensorDeviceType()

        def memory_statistics() -> dict[str, float]:
            return {"allocated_mib": torch.cuda.memory_allocated()/1024**2,
                "reserved_mib": torch.cuda.memory_reserved()/1024**2,
                "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}

        def make_world(payload: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
            if "voxel_points" not in payload:
                return WorldConfig.from_dict(payload), None, {}
            field, statistics, outside = observed_distance_field(payload["voxel_points"],
                voxel_size=payload["voxel_size"], dims=payload["dims"])
            outside_world, outside_statistics = occupied_voxel_world(outside)
            voxel = VoxelGrid(name="observed_distance", pose=[0., 0., 0., 1., 0., 0., 0.],
                dims=payload["dims"], voxel_size=payload["voxel_size"],
                feature_tensor=tensor_args.to_device(field.reshape(-1)))
            world = WorldConfig.from_dict(outside_world)
            world.voxel = [voxel]
            statistics["outside_grid_cuboids"] = outside_statistics["occupied_cells"]
            return world, voxel, statistics

        current_world, current_voxel, current_world_statistics = make_world(backend_config["world"])
        if not current_world_statistics:
            current_world_statistics = backend_config["world_statistics"]
        robot = RobotConfig.from_basic(urdf, "mujoco_world", "nero_tcp", tensor_args)
        config = IKSolverConfig.load_from_robot_config(robot, None,
            tensor_args=tensor_args, num_seeds=num_seeds,
            position_threshold=.0002, rotation_threshold=.0015,
            self_collision_check=False, self_collision_opt=False,
            use_cuda_graph=True, high_precision=True)
        solver = IKSolver(config)
        q = solver.sample_configs(1)
        fk = solver.fk(q)
        warm = solver.solve_single(Pose(fk.ee_position, fk.ee_quaternion),
                                  retract_config=q, return_seeds=max_candidates)
        torch.cuda.synchronize()
        motion, motion_error = None, None
        geometry = backend_config["robot_spheres"]
        if geometry["collision_link_names"]:
            try:
                motion_robot = RobotConfig.from_dict({"kinematics": {
                    "urdf_path": urdf, "base_link": "mujoco_world", "ee_link": "nero_tcp",
                    **{key: geometry[key] for key in ("collision_link_names", "collision_spheres", "self_collision_ignore")},
                    "collision_sphere_buffer": 0., "self_collision_buffer": {},
                    "use_global_cumul": True}}, tensor_args)
                motion_config = MotionGenConfig.load_from_robot_config(motion_robot,
                    current_world, tensor_args=tensor_args,
                    num_ik_seeds=num_seeds, num_trajopt_seeds=4, num_graph_seeds=4,
                    trajopt_tsteps=32, interpolation_steps=2000, interpolation_dt=.02,
                    position_threshold=.0002, rotation_threshold=.001,
                    self_collision_check=True, self_collision_opt=True,
                    collision_checker_type=(CollisionCheckerType.VOXEL if current_voxel is not None
                                            else CollisionCheckerType.PRIMITIVE),
                    collision_cache={"obb": 4096, **({"voxel": {
                        "layers": 1, "dims": current_voxel.dims,
                        "voxel_size": current_voxel.voxel_size,
                        "feature_dtype": torch.float32}} if current_voxel is not None else {})},
                    use_cuda_graph=True)
                motion = MotionGen(motion_config)
                if current_voxel is not None:
                    motion.world_coll_checker.update_voxel_data(current_voxel)
                motion.warmup(enable_graph=True, warmup_js_trajopt=False)
                torch.cuda.synchronize()
            except Exception as error:
                motion, motion_error = None, type(error).__name__+": "+str(error)
        ready = {"ready": True, "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "joint_names": solver.joint_names,
            "num_seeds": num_seeds, "max_candidates": max_candidates,
            "warmup_success": bool(torch.any(warm.success).item()),
            "collision_optimization": motion is not None,
            "motiongen_available": motion is not None,
            "motiongen_error": motion_error,
            "robot_spheres": geometry["statistics"],
            "world": current_world_statistics, "cuda_memory": memory_statistics()}
    protocol.write(json.dumps(ready)+"\n"); protocol.flush()
    previous_bounds = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if message.get("operation") == "update_world":
                    current_world, current_voxel, current_world_statistics = make_world(message["world"])
                    if motion is not None:
                        motion.update_world(current_world)
                        if current_voxel is not None:
                            motion.world_coll_checker.update_voxel_data(current_voxel)
                        torch.cuda.synchronize()
                    response = {"updated": True, "motiongen_available": motion is not None,
                        "world_statistics": current_world_statistics}
                    protocol.write(json.dumps(response)+"\n"); protocol.flush()
                    continue
                if message.get("operation") == "scene_query":
                    if motion is None:
                        response = {"error": "MotionGen collision representation unavailable"}
                    else:
                        points = tensor_args.to_device(message["points"]).reshape(-1, 1, 1, 3)
                        spheres = torch.cat((points, torch.full_like(points[..., :1],
                            float(message["radius_m"]))), dim=-1)
                        query_buffer = CollisionQueryBuffer.initialize_from_shape(
                            spheres.shape, tensor_args, motion.world_coll_checker.collision_types)
                        values = motion.world_coll_checker.get_sphere_distance(spheres,
                            query_buffer, tensor_args.to_device([1.]), tensor_args.to_device([0.]))
                        torch.cuda.synchronize()
                        response = {"collision_values_m": values.detach().cpu().reshape(-1).tolist(),
                            "radius_m": message["radius_m"],
                            "positive_values_mean": "sphere overlaps occupied proposal map"}
                    protocol.write(json.dumps(response)+"\n"); protocol.flush()
                    continue
                t0 = time.perf_counter()
                # Match the existing request-private start envelope. Update
                # allocated tensors in place: captured CUDA graphs retain
                # addresses; nominal hardware/model limits are untouched.
                bounds_array = np.asarray([message["lower_limits"], message["upper_limits"]], dtype=float)
                if previous_bounds is None or not np.array_equal(bounds_array, previous_bounds):
                    bounds = tensor_args.to_device(bounds_array)
                    reachers = [solver]
                    if motion is not None:
                        reachers.extend((motion.ik_solver, motion.trajopt_solver,
                            motion.finetune_trajopt_solver, motion.js_trajopt_solver,
                            motion.finetune_js_trajopt_solver))
                    rollouts = list(solver.get_all_rollout_instances())
                    if motion is not None:
                        rollouts += motion.get_all_rollout_instances()
                    for rollout in {id(item): item for item in rollouts}.values():
                        rollout.dynamics_model.get_state_bounds().position.copy_(bounds)
                        rollout.dynamics_model.joint_limits.position.copy_(bounds)
                        for attribute in ("bound_cost", "bound_constraint"):
                            cost = getattr(rollout, attribute, None)
                            if cost is not None:
                                cost.joint_limits.position.copy_(bounds)
                    for reacher in reachers:
                        for optimizer in reacher.solver.optimizers:
                            if optimizer.rollout_fn.dynamics_model.control_space.name == "POSITION":
                                repeats = optimizer.action_lows.numel()//7
                                optimizer.action_lows.copy_(bounds[0].repeat(repeats))
                                optimizer.action_highs.copy_(bounds[1].repeat(repeats))
                                if hasattr(optimizer, "action_range"):
                                    optimizer.action_range.copy_(optimizer.action_highs-optimizer.action_lows)
                                if hasattr(optimizer, "action_step_max"):
                                    optimizer.action_step_max.copy_(optimizer.step_scale*torch.abs(optimizer.action_range))
                        sampler = getattr(reacher, "q_sample_gen", None)
                        if sampler is not None:
                            sampler.low_bounds.copy_(bounds[0]); sampler.range_b.copy_(bounds[1]-bounds[0])
                    if motion is not None:
                        motion.graph_planner.bounds.copy_(bounds)
                        motion.graph_planner.sample_gen.low_bounds.copy_(bounds[0])
                        motion.graph_planner.sample_gen.range_b.copy_(bounds[1]-bounds[0])
                        motion.graph_planner.reset_buffer()
                    previous_bounds = bounds_array.copy()
                start = np.asarray(message["start_q"], dtype=float)
                retract = tensor_args.to_device(start).view(1, -1)
                seeds = np.asarray(message["seeds"], dtype=float).reshape(-1, 7)
                seeds = np.concatenate((start[None], seeds), axis=0)[:num_seeds]
                # v0.7.8 get_seed() concatenates on dimension 1 and expects
                # [batch,seeds,dof], despite solve_single's stale docstring.
                seed_tensor = tensor_args.to_device(seeds).view(1, len(seeds), 7)
                candidates, attempts = [], []
                for index, pose in enumerate(message["target_poses"]):
                    goal = Pose(tensor_args.to_device(pose[:3]).view(1, 3),
                                tensor_args.to_device(pose[3:]).view(1, 4))
                    if motion is not None:
                        motion_start = time.perf_counter()
                        generated = motion.plan_single(
                            JointState.from_position(retract, joint_names=motion.kinematics.joint_names),
                            goal, MotionGenPlanConfig(max_attempts=2, enable_graph=True,
                                enable_graph_attempt=1, enable_finetune_trajopt=True))
                        torch.cuda.synchronize()
                        accepted = bool(torch.any(generated.success).item())
                        attempts.append({"pose_index": index, "method": "motiongen",
                            "solve_wall_s": time.perf_counter()-motion_start,
                            "success": accepted, "status": str(generated.status)})
                        if accepted:
                            route = generated.optimized_plan.position.detach().cpu().numpy().reshape(-1, 7)
                            # Propose geometric bends, not thousands of timing
                            # samples. Original worker checks every new chord
                            # and computes its own approved trajectory time.
                            keep = [0]
                            while keep[-1] < len(route)-1:
                                left = keep[-1]; right = left+1
                                for trial in range(left+2, len(route)):
                                    fraction = np.linspace(0., 1., trial-left+1)[:, None]
                                    chord = route[left]+fraction*(route[trial]-route[left])
                                    if np.max(np.abs(chord-route[left:trial+1])) > .01:
                                        break
                                    right = trial
                                keep.append(right)
                            candidates.append({"pose_index": index, "q": route[-1].tolist(),
                                "path": route[keep].tolist(), "method": "motiongen"})
                    solve_start = time.perf_counter()
                    result = solver.solve_single(goal, retract_config=retract,
                        seed_config=seed_tensor, return_seeds=max_candidates)
                    torch.cuda.synchronize()
                    valid = result.success.detach().cpu().numpy().reshape(-1)
                    solutions = result.solution.detach().cpu().numpy().reshape(-1, 7)
                    for q, accepted in zip(solutions, valid):
                        if accepted:
                            candidates.append({"pose_index": index, "q": q.tolist(), "method": "parallel_ik_direct"})
                    attempts.append({"pose_index": index,
                        "method": "parallel_ik",
                        "solve_wall_s": time.perf_counter()-solve_start,
                        "success_count": int(valid.sum())})
                span = np.maximum(np.asarray(message["upper_limits"])-np.asarray(message["lower_limits"]), 1e-6)
                candidates.sort(key=lambda item: (0 if item.get("method") == "motiongen" else 1,
                    float(np.linalg.norm((np.asarray(item["q"])-start)/span))))
                response = {"candidates": candidates, "diagnostics": {
                    "available": True, "motiongen_available": motion is not None,
                    "motiongen_error": motion_error, "solve_attempts": attempts,
                    "cuda_memory": memory_statistics(),
                    "service_wall_s": time.perf_counter()-t0}}
        except Exception as error:
            response = {"error": type(error).__name__+": "+str(error)}
        protocol.write(json.dumps(response, allow_nan=False)+"\n"); protocol.flush()


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--server":
        try:
            _server(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        except Exception as error:
            print(json.dumps({"ready": False,
                              "error": type(error).__name__+": "+str(error)}), flush=True)
            raise
    else:
        raise SystemExit("This module's executable entry is a private planning service")

"""Measured RGB-D input for the existing October 9 GraspGen-X algorithm.

The CUDA worker runs the original backend in its installed environment. The
original candidate selector and vertical branch check run in a private model
whose world is the measured belt frame; observations and the public result
remain in robot-base coordinates. No camera, CAN driver or physics step is
opened here. Live GraspMoE combines official OBB proposals and diffusion
proposals, both scored by the same GraspGen-X learned discriminator.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT / 'experiments/graspgenx_sim'
REPLY_PREFIX = 'NERO_GRASPGENX_REPLY '


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


from .artifact_writer import background_artifact


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json(value), ensure_ascii=False,
                                     indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return str(path)


_write_selection_json = background_artifact(_write_json)


def _config(config=None):
    result = yaml.safe_load((EXPERIMENT / 'config.yaml').read_text())['graspgenx']
    # The live adapter accepts the network's full orientation range. The
    # simulation's explicit tilt setting remains in its own configuration.
    result['maximum_approach_tilt_deg'] = None
    result['planner'] = 'graspmoe'
    result.update(dict(config or {}))
    return result


def _points(value):
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('网络输入须为有限 N×3 基座目标点云，单位米')
    return points


def _observed_object_points(detection):
    """Use existing native-depth segmentation and factory D2C RGB rejection."""
    if detection.get('grasp_depth_refinement', {}).get('applied'):
        # The target has already been reconstructed on the same native pixel
        # grid and passed the existing colour/density rules. Native frame and
        # scene remain untouched; reindexing native points would undo Fast.
        return _points(detection['object_points'])
    if detection.get('target_cloud_filter') is not None:
        # The real entry has already filtered native pixels before measuring
        # the component. Network, target geometry and preview share this cloud.
        return _points(detection['object_points'])
    if all(name in detection for name in ('frame', 'depth_to_color_cv', 'mask', 'points')):
        from .shape_pick import exclude_green_belt
        mask = exclude_green_belt(detection['frame'], detection['depth_to_color_cv'],
                                  np.asarray(detection['mask'], dtype=bool))
        return _points(np.asarray(detection['points'])[mask])
    return _points(detection['object_points'])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(tcp):
    """Capture-pose identity inside one immutable observed candidate pool."""
    return np.asarray(tcp, dtype=np.float64).tobytes()


def _support_frame(detection):
    """Freeze the measured support plane; its XY orientation is arbitrary."""
    value = detection.get('belt_region', {}).get('plane_to_base')
    if value is None:
        return None
    frame = np.asarray(value, dtype=float).copy()
    if (frame.shape != (4, 4) or not np.isfinite(frame).all()
            or not np.allclose(frame[3], [0., 0., 0., 1.])
            or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(frame[:3, :3]), 1., atol=1e-6)):
        raise ValueError('GraspMoE 带面坐标须为有限刚体变换')
    frame.setflags(write=False)
    return frame


def _infer_with_support(backend, points, parameters, support_frame):
    if getattr(backend, 'supports_support_frame', False) and support_frame is not None:
        return backend.infer(points, *parameters, support_frame=support_frame)
    return backend.infer(points, *parameters)


from .types import identity_dict


class PreparedGraspInference:
    """Compute only the network while the caller installs the measured scene.

    This worker owns an immutable cloud, never a robot geometry model. Resolve
    verifies the exact input and inference parameters before reusing the result;
    candidate collision and reachability checks still run in the caller.
    """
    def __init__(self, detection, backend, *, seed=0, config=None):
        cfg = _config(getattr(backend, 'config', None))
        cfg.update(dict(config or {}))
        self.target_identity = detection.get('target_identity')
        self.backend = backend
        self.points = np.ascontiguousarray(_observed_object_points(detection)).copy()
        self.points.setflags(write=False)
        self.support_frame = _support_frame(detection)
        self.parameters = (int(seed), int(cfg['generated_candidates']), int(cfg['returned_candidates']))
        self.timings = {}
        self._consumed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='graspgenx_prepare')
        self._future = self._executor.submit(self._infer)

    def _infer(self):
        started = time.monotonic()
        try:
            self.timings['inference_start_monotonic'] = started
            result = _infer_with_support(self.backend, self.points, self.parameters, self.support_frame)
            if self.target_identity is not None:
                result['metadata'] = {**result.get('metadata', {}),
                    'target_identity':identity_dict(self.target_identity)}
            return result
        finally:
            finished = time.monotonic()
            self.timings['inference_finish_monotonic'] = finished
            self.timings['inference_job_s'] = finished-started

    def resolve(self, points, backend, seed, num_grasps, topk, *, support_frame=None, target_identity=None):
        if target_identity != self.target_identity:
            raise ValueError('预先推理结果的任务/几何版本已过期')
        same_support = (not getattr(backend, 'supports_support_frame', False)
                        or (self.support_frame is None and support_frame is None)
                        or (self.support_frame is not None and support_frame is not None
                            and np.array_equal(self.support_frame, support_frame)))
        if (backend is not self.backend or self.parameters != (seed, num_grasps, topk)
                or not np.array_equal(self.points, points) or not same_support):
            raise ValueError('预先推理的输入点云、带面、网络或参数与本次选姿不一致')
        if self._consumed:
            raise ValueError('预先推理的结果已经用于一次选姿')
        self._consumed = True
        started = time.monotonic()
        try:
            return self._future.result()
        finally:
            self.timings['selection_wait_s'] = time.monotonic() - started

    def close(self):
        # Join before closing the backend: infer and backend.close share a lock.
        self._executor.shutdown(wait=True, cancel_futures=True)


class GraspGenXProcessBackend:
    """Keep the original CUDA model loaded outside the hardware Python env.

    ``start`` can run before opening devices, so model/weight loading does not
    consume the lifetime of the first moving target. Every inference retains
    the exact measured input, raw candidates, network metadata and worker log.
    """
    supports_support_frame = True

    def __init__(self, output_directory, *, python_executable=None, config=None):
        self.output_directory = Path(output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.python_executable = Path(python_executable or EXPERIMENT / '.venv/bin/python').absolute()
        self.config = _config(config)
        self.process = None
        self.provenance = None
        self._lock = threading.RLock()
        self._calls = 0
        self._log = None

    def _receive(self, operation):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f'GraspGen-X 网络进程退出：{self.process.poll()}；日志 {self.output_directory / "worker.log"}')
            self._log.write(line)
            self._log.flush()
            if line.startswith(REPLY_PREFIX):
                reply = json.loads(line[len(REPLY_PREFIX):])
                if reply.get('operation') != operation:
                    raise RuntimeError('GraspGen-X 网络进程回复与当前请求不一致')
                if reply.get('error'):
                    raise RuntimeError('GraspGen-X: ' + reply['error'])
                return reply

    def start(self):
        with self._lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError('GraspGen-X 网络进程已退出')
                return self
            if not self.python_executable.is_file():
                raise FileNotFoundError(f'缺少已有 GraspGen-X Python 环境：{self.python_executable}')
            config_path = self.output_directory / 'worker_configuration.json'
            _write_json(config_path, self.config)
            env = dict(os.environ)
            env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
            command = [str(self.python_executable), '-u', '-m',
                       'mujoco_conveyor_sorting.runtime.nero_graspgenx',
                       '--worker', '--configuration', str(config_path)]
            self._log = (self.output_directory / 'worker.log').open('a', encoding='utf-8')
            self.process = subprocess.Popen(command, cwd=PROJECT, env=env, text=True,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, bufsize=1,
                                            # Terminal Ctrl+C belongs to the parent's
                                            # pause control; close() owns this worker.
                                            start_new_session=True)
            try:
                self.provenance = self._receive('ready')['provenance']
            except BaseException:
                self.close()
                raise
            _write_json(self.output_directory / 'backend.json', {
                'command': command, 'worker_python': str(self.python_executable),
                'network': self.provenance, 'devices_opened': False})
            return self

    def infer(self, point_cloud_world_m, seed=None, num_grasps=None, topk=None, *, support_frame=None):
        points = _points(point_cloud_world_m)
        with self._lock:
            self.start()
            self._calls += 1
            stem = self.output_directory / f'call_{self._calls:04d}'
            input_path = stem.with_suffix('.input.npy')
            output_path = stem.with_suffix('.candidates.npz')
            np.save(input_path, points, allow_pickle=False)
            request = {'operation': 'infer', 'input': str(input_path), 'output': str(output_path),
                       'seed': self._calls - 1 if seed is None else int(seed),
                       'num_grasps': int(num_grasps or self.config['generated_candidates']),
                       'topk': int(topk or self.config['returned_candidates']),
                       'support_frame': _json(support_frame)}
            _write_json(stem.with_suffix('.request.json'), request)
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            reply = self._receive('infer')
            with np.load(output_path, allow_pickle=False) as result:
                poses, scores = result['poses'].copy(), result['scores'].copy()
            return {'poses': poses, 'scores': scores, 'metadata': reply['metadata'],
                    'artifact_prefix': str(stem)}

    def close(self):
        with self._lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write(json.dumps({'operation': 'close'}) + '\n')
                        process.stdin.flush()
                        process.wait(timeout=10.)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=3.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                process.stdin.close()
                process.stdout.close()
            if self._log is not None:
                self._log.close()
                self._log = None


def belt_candidate_frame(detection, motion_region):
    """Rigid coordinates for historical +X belt / +Z support-plane checks."""
    plane = np.asarray(detection['belt_region']['plane_to_base'], dtype=float)
    normal = plane[:3, 2].copy()
    normal /= np.linalg.norm(normal)
    direction = np.asarray(motion_region['belt_direction'], dtype=float)
    # Visual belt velocity belongs to the fitted support plane. Removing its
    # tiny normal component constructs coordinates, not a commanded axis.
    longitudinal = direction - normal * np.dot(direction, normal)
    longitudinal /= np.linalg.norm(longitudinal)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((longitudinal, np.cross(normal, longitudinal), normal))
    origin = np.asarray(motion_region['belt_origin'], dtype=float)
    frame[:3, 3] = origin - normal * np.dot(origin - plane[:3, 3], normal)
    return frame


class _BeltCandidateEnvironment:
    """Historical candidate hooks backed by a private observed-scene model.

    RobotModel keeps its obstacle and target point clouds in base coordinates.
    A second private kinematic model has its base rigidly moved into the belt
    coordinate system for the original vertical-reachability implementation.
    Neither model is advanced with MuJoCo physics or connected to hardware.
    """
    def __init__(self, geometry, state, belt_to_base, future_translation, *, lift_m,
                 maximum_approach_tilt_deg=None):
        import mujoco
        from .nero_policy_context import _copy_geometry
        from ..controllers.ik_solver import DampedLeastSquaresIK

        self.geometry = _copy_geometry(geometry)
        self.geometry.target = geometry.target + np.asarray(future_translation)
        self.geometry.set_state(state['joints'], state['width_m'])
        self.belt_to_base = np.asarray(belt_to_base, dtype=float).copy()
        self.base_to_belt = np.linalg.inv(self.belt_to_base)
        self.model = copy.copy(self.geometry.model)
        base = self.model.body('base_link').id
        original = np.eye(4)
        original[:3, 3] = self.model.body_pos[base]
        original[:3, :3] = Rotation.from_quat(self.model.body_quat[base][[1, 2, 3, 0]]).as_matrix()
        transformed = self.base_to_belt @ original
        self.model.body_pos[base] = transformed[:3, 3]
        self.model.body_quat[base] = Rotation.from_matrix(transformed[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.geometry.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self.arm_joint_names = tuple(geometry.names)
        self.arm_joint_ids = np.asarray(geometry.ids)
        self.arm_qpos_adrs = np.asarray(geometry.indices)
        self.arm_dof_adrs = self.model.jnt_dofadr[self.arm_joint_ids]
        self.gripper_joint_ids = np.asarray([self.model.joint(n).id for n in ('gripper_joint1', 'gripper_joint2')])
        self.gripper_qpos_adrs = self.model.jnt_qposadr[self.gripper_joint_ids]
        self.ee_site_name, self.ee_site_id = 'ee_site', self.model.site('ee_site').id
        self.ik = DampedLeastSquaresIK(self.model, self.data, self.arm_joint_names,
                                      self.ee_site_name, copy.deepcopy(geometry.ik.config))
        self.ik.lower_limits = geometry.ik.lower_limits.copy()
        self.ik.upper_limits = geometry.ik.upper_limits.copy()
        self.ik.center_positions = geometry.ik.center_positions.copy()
        self.belt_height = 0.
        self.scene_config = {'workspace': {}}
        self.lift_m = float(lift_m)
        self.robot_config = {'controller': {'joint_command_margin': geometry.joint_margin_rad},
                             'trajectory': {'approach_height': .10,
                                            'lift_clearance_height': self.lift_m,
                                            # The nominal descent/lift check otherwise
                                            # falls back to a separate .52 cone.
                                            'acquisition_minimum_downward_alignment':
                                                (-1. if maximum_approach_tilt_deg is None else
                                                 float(np.cos(np.radians(maximum_approach_tilt_deg))))}}
        self._selection_initial_joints = np.asarray(state['joints'], dtype=float).copy()
        self._candidate_hint_seeds = {}
        self._active_candidate_seed = None
        self._ranked_selection_enabled = False
        self._direct_contact_selection = False
        self._batch_gpu_backend = getattr(geometry, 'candidate_batch_ik_backend', None)

    def rank_candidate_directions(self, directions, *, _full_pool=False):
        """Order the original candidate pool using private kinematic hints.

        A GPU hint is not a collision certificate. Unknown/failed GPU goals
        stay in the pool, and the original solver and geometry checks still
        decide acceptance. No learned reachability cutoff is introduced.
        """
        from ..controllers.candidate_batch_ik import CandidateBatchIK, BatchIKHints
        started = time.monotonic()
        if not directions:
            self.candidate_selection_diagnostics = {'pool_directions': 0}
            return directions
        # The user task requests a side-aligned top grasp. Keep that task
        # preference ahead of numerical IK hints and learned-score ordering.
        directions = sorted(directions, key=lambda row: row.get('task_preference', {}).get('group', 2))
        targets = []
        for row in directions:
            capture = self.belt_to_base @ row['tcp']
            plan = self._candidate_motion_preview(capture)
            high = capture.copy()
            if plan is None:
                if not self._direct_contact_selection:
                    high[2, 3] += .10
            else:
                high[:3, 3] += (np.asarray(plan['capture_translation_base_m'])
                               + np.asarray(plan['high_offset_from_capture_base_m']))
            targets.append(np.r_[high[:3, 3],
                Rotation.from_matrix(high[:3, :3]).as_quat()[[3, 0, 1, 2]]])
        targets = np.asarray(targets)
        trajectory = self.robot_config['trajectory']
        margin = (trajectory['minimum_joint_limit_margin']
                  + (0. if self._direct_contact_selection else trajectory['pregrasp_tracking_joint_margin_reserve']))
        solver = self.geometry.ik
        ranges = self.model.jnt_range[self.arm_joint_ids]
        lower = np.maximum(solver.lower_limits, ranges[:, 0] + margin)
        upper = np.minimum(solver.upper_limits, ranges[:, 1] - margin)
        # An ordinary frame often already has a valid first network grasp.
        # Probe its allowed symmetry directions once with the existing DLS;
        # only a failed first grasp pays for the whole GPU pool. This is an
        # ordering shortcut, not a new acceptance or rejection condition.
        first_source = directions[0]['source_index']
        first = [i for i, row in enumerate(directions) if row['source_index'] == first_source]
        probe = (None if _full_pool else CandidateBatchIK(solver).solve(
            targets[first], qpos_seed=self._selection_initial_joints,
            lower_limits=lower, upper_limits=upper,
            cpu_fallback=True, maximum_cpu_seeds=1))
        if probe is not None and any(probe.solutions):
            solutions = [() for _ in directions]
            distances = [float('inf') for _ in directions]
            for local, index in enumerate(first):
                solutions[index] = probe.solutions[local]
                distances[index] = probe.joint_distances[local]
            hints = BatchIKHints(tuple(solutions), tuple(distances), {
                'source': 'first_network_grasp_original_dls',
                'first_grasp_probe': probe.diagnostics,
                'full_gpu_pool_needed': False})
        else:
            hints = CandidateBatchIK(solver, gpu_backend=self._batch_gpu_backend).solve(
                targets, qpos_seed=self._selection_initial_joints,
                lower_limits=lower, upper_limits=upper, cpu_fallback=False)
            hints.diagnostics['first_grasp_probe'] = None if probe is None else probe.diagnostics
            hints.diagnostics['full_gpu_pool_needed'] = True
        # Unknown targets retain their original quality order.  Approximating
        # reachability from one Jacobian can worsen ordinary successful frames.
        costs = np.full(len(directions), np.inf)
        for i, solutions in enumerate(hints.solutions):
            if solutions:
                seed = np.asarray(solutions[0].joint_positions).copy()
                self._candidate_hint_seeds[row_key(directions[i]['tcp'])] = seed
                costs[i] = float(hints.joint_distances[i])
        # Equal rank contributions avoid a robot-specific tuned score cutoff.
        quality_order = sorted(range(len(directions)), key=lambda i: (-directions[i]['score'], i))
        motion_order = sorted(range(len(directions)), key=lambda i: (costs[i], i))
        quality_rank = {i: rank for rank, i in enumerate(quality_order)}
        motion_rank = {i: rank for rank, i in enumerate(motion_order)}
        priorities = {i: (directions[i].get('task_preference', {}).get('group', 2),
                         *((False, quality_rank[i] + motion_rank[i], -directions[i]['score'], costs[i], i)
                         if hints.solutions[i] else
                         (True, quality_rank[i], -directions[i]['score'], costs[i], i)))
                      for i in range(len(directions))}
        groups = {}
        for i, row in enumerate(directions):
            row['selection_priority'] = {
                'task_group': row.get('task_preference', {}).get('group', 2),
                'numerical_high_hint': bool(hints.solutions[i]) and not self._direct_contact_selection,
                'numerical_contact_hint': bool(hints.solutions[i]) and self._direct_contact_selection,
                'quality_rank': quality_rank[i], 'joint_motion_rank': motion_rank[i],
                'rank_sum': (quality_rank[i] + motion_rank[i] if hints.solutions[i] else quality_rank[i]),
                'motion_cost_is_estimate': True}
            groups.setdefault(row['source_index'], []).append(i)
        ordered = []
        for indices in sorted(groups.values(), key=lambda group: min(priorities[i] for i in group)):
            ordered.extend(directions[i] for i in sorted(indices, key=lambda i: priorities[i]))
        previous = getattr(self, 'candidate_selection_diagnostics', {}) if _full_pool else {}
        self.candidate_selection_diagnostics = {
            'pool_directions': previous.get('pool_directions', len(directions)),
            'ranking_s': previous.get('ranking_s', 0.) + time.monotonic() - started,
            'ordering': 'task_group_then_numerical_hint_quality_plus_motion_rank_unknown_preserves_network_order',
            'gpu_failure_is_rejection': False, 'batch_ik': hints.diagnostics,
            'hint_target': 'capture_tcp' if self._direct_contact_selection else 'high_tcp',
            'first_grasp_full_check_failed_gpu_expansion': _full_pool,
            'initial_batch_ik': previous.get('batch_ik') if _full_pool else None}
        return ordered

    def rank_remaining_candidate_directions(self, directions):
        diagnostics = getattr(self, 'candidate_selection_diagnostics', {})
        if (self._batch_gpu_backend is not None and directions
                and diagnostics.get('batch_ik', {}).get('full_gpu_pool_needed') is False):
            return self.rank_candidate_directions(directions, _full_pool=True)
        return None


    def use_execution_joint_margins(self, state, settings):
        report = self.geometry.configure_start_joint_tolerance(
            state['joints'], settings.initial_joint_tolerance_deg)
        margin = self.geometry.joint_margin_rad if report['expanded'] else .035
        reserve = 0. if report['expanded'] else .01
        start_margin = float(np.min(np.minimum(np.asarray(state['joints']) - self.geometry.limits[:, 0],
                                               self.geometry.limits[:, 1] - np.asarray(state['joints']))))
        reserve = min(reserve, max(0., start_margin - .003 - margin))
        self.model.jnt_range[:] = self.geometry.model.jnt_range
        self.robot_config['trajectory'].update(minimum_joint_limit_margin=margin,
                                               pregrasp_tracking_joint_margin_reserve=reserve)
        return margin

    def get_joint_positions(self):
        return self.data.qpos[self.arm_qpos_adrs].copy()

    def solve_ik_result_with_margin(self, position, quaternion, qpos_seed, *,
                                   minimum_joint_margin, allow_multiseed=False):
        """Search within the bounds that the caller will actually accept.

        High poses include the existing tracking reserve; capture, descent and
        lift use only their existing base margin. Intersect with the original
        target IK limits so an expanded measured start cannot expand new goals.
        Only this privately owned solver is changed, for this one call.
        """
        solver = self.geometry.ik
        original = solver.lower_limits, solver.upper_limits, solver.center_positions
        ranges = self.model.jnt_range[self.arm_joint_ids]
        try:
            solver.lower_limits = np.maximum(original[0], ranges[:, 0] + minimum_joint_margin)
            solver.upper_limits = np.minimum(original[1], ranges[:, 1] - minimum_joint_margin)
            solver.center_positions = .5 * (solver.lower_limits + solver.upper_limits)
            return self.solve_ik_result(position, quaternion, qpos_seed,
                                       allow_multiseed=allow_multiseed)
        finally:
            solver.lower_limits, solver.upper_limits, solver.center_positions = original

    def solve_ik_result(self, position, quaternion, qpos_seed, *, allow_multiseed=False):
        pose = np.eye(4)
        pose[:3, 3] = position
        pose[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        base_pose = self.belt_to_base @ pose
        q = Rotation.from_matrix(base_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        # The original checker reads this only after its pregrasp/intercept
        # solves, when the most recent requested height is the grasp height.
        self.robot_config['trajectory']['lift_clearance_height'] = float(position[2]) + self.lift_m
        initial_high = (self._ranked_selection_enabled
                        and np.array_equal(np.asarray(qpos_seed), self._selection_initial_joints))
        seed = (self._active_candidate_seed if initial_high and self._active_candidate_seed is not None
                else qpos_seed)
        result = self.geometry.ik.solve(base_pose[:3, 3], q, qpos_seed=seed)
        # GPU already explores multiple seeds for all directions. Without GPU
        # hints, visit the remaining original candidates before spending four
        # CPU solves on each early failure. A known branch that fails its
        # updated pose may still try alternative seeds on the CPU.
        if not result.success and (allow_multiseed or (initial_high and self._active_candidate_seed is not None)):
            from ..controllers.candidate_batch_ik import CandidateBatchIK
            hints = CandidateBatchIK(self.geometry.ik).solve(
                np.r_[base_pose[:3, 3], q][None], qpos_seed=np.asarray(qpos_seed),
                lower_limits=self.geometry.ik.lower_limits,
                upper_limits=self.geometry.ik.upper_limits,
                extra_seeds=(() if self._active_candidate_seed is None else (self._active_candidate_seed,)),
                excluded_seeds=(seed,), cpu_fallback=True)
            if hints.solutions[0]:
                return hints.solutions[0][0]
        return result

    def _arm_state_data_is_safe(self, joints, data, *, approach_axis_local=(0., 0., 1.),
                                minimum_downward_alignment=-1., minimum_tcp_height=-np.inf,
                                minimum_joint_margin=0., maximum_jacobian_condition=None):
        self.last_rejection = None
        joints = np.asarray(joints, dtype=float)
        ranges = self.model.jnt_range[self.arm_joint_ids]
        if np.min(np.minimum(joints - ranges[:, 0], ranges[:, 1] - joints)) < minimum_joint_margin - 1e-9:
            self.last_rejection = {'reason': 'joint_margin', 'required_margin_rad': minimum_joint_margin}
            return False
        axis = np.asarray(approach_axis_local, dtype=float)
        axis /= np.linalg.norm(axis)
        position = data.site_xpos[self.ee_site_id]
        rotation = data.site_xmat[self.ee_site_id].reshape(3, 3)
        if position[2] < minimum_tcp_height - 1e-9 or -float((rotation @ axis)[2]) < minimum_downward_alignment - 1e-9:
            self.last_rejection = {'reason': 'height_or_alignment'}
            return False
        if maximum_jacobian_condition is not None:
            jacobian = self.ik.compute_jacobian(joints)
            jacobian[3:] *= .3
            singular = np.linalg.svd(jacobian, compute_uv=False)
            if singular[-1] <= 1e-12 or singular[0] / singular[-1] > maximum_jacobian_condition:
                return False
        width = float(abs(data.qpos[self.gripper_qpos_adrs[0]]) + abs(data.qpos[self.gripper_qpos_adrs[1]]))
        try:
            self.geometry.check_state(joints, width, allow_finger_contact=True)
        except RuntimeError as error:
            self.last_rejection = {'reason': str(error), 'details': getattr(error, 'details', None)}
            return False
        return True

    def is_arm_configuration_safe(self, joints, **kwargs):
        import mujoco
        self.data.qpos[self.arm_qpos_adrs] = joints
        self.data.qpos[self.gripper_qpos_adrs] = [.05, -.05]
        mujoco.mj_forward(self.model, self.data)
        return self._arm_state_data_is_safe(joints, self.data, **kwargs)


def measured_candidate_motion_plan(detection, geometry, state, region, snapshot, now, settings,
                                   *, candidate_grasp=None, minimum_lead_s=0., current_tcp_position=None):
    """Build the initial capture clock from measured motion, never the belt midpoint.

    The existing two-leg geometric ETA is only an initial estimate. The full
    global planner replaces it with joint-path timing before issuing motion.
    """
    from ..controllers.baseline_sorting_policy import BaselinePolicyConfig
    from ..controllers.dynamic_intercept import intercept_lead, pregrasp_high_pose
    from ..controllers._math import belt_plane_motion_from_xy

    cfg = BaselinePolicyConfig()
    direct = getattr(settings, 'moving_approach_mode', 'high') == 'continuous'
    if snapshot is None or not snapshot.valid or snapshot.velocity_source == 'unavailable':
        return None
    velocity = getattr(snapshot, 'prediction_velocity', None)
    if velocity is None:
        velocity = np.asarray(snapshot.velocity, dtype=float)
        if snapshot.velocity_source == 'nero7_xy_kalman':
            velocity = belt_plane_motion_from_xy(velocity, detection['plane']['normal_base'])
    velocity = np.asarray(velocity, dtype=float)
    current = np.asarray(snapshot.current_grasp)[:3, 3]
    displacement = current - np.asarray(detection['grasp'])[:3, 3]
    if candidate_grasp is not None:
        current = np.asarray(candidate_grasp)[:3, 3] + displacement
    direction = np.asarray(region['belt_direction'])
    coordinate = float((current - np.asarray(region['belt_origin'])) @ direction)
    speed = float(velocity @ direction)
    horizon = (300. if abs(speed) < 1e-6 else
               max(1e-6, max((float(s) - coordinate) / speed for s in region['s_range'])
                   - cfg.intercept_deadline_margin))
    # Read FK from the same measured joints used by candidate IK, not from a
    # flange pose mistaken for the TCP. Restore the caller's private state.
    if current_tcp_position is None:
        qpos = geometry.data.qpos.copy()
        try:
            geometry.set_state(state['joints'], state['width_m'])
            ee = geometry.tcp()[:3, 3].copy()
        finally:
            geometry.data.qpos[:] = qpos
            geometry.mj.mj_forward(geometry.model, geometry.data)
    else:
        ee = np.asarray(current_tcp_position, dtype=float)
    lead = intercept_lead(current, velocity, ee, grasp_height=0.,
        approach_clearance=0. if direct else cfg.approach_height,
        cartesian_speed=min(.25, settings.joint_velocity_limit * .15),
        minimum_motion_time=max(cfg.prediction_lead_time, (.18 if direct else cfg.descent_lead_time) + 1./settings.control_hz,
                                minimum_lead_s),
        latency=0., maximum_horizon=horizon)
    translation = displacement + velocity * lead
    high_offset = (np.zeros(3) if direct else pregrasp_high_pose(np.zeros(7), velocity,
        approach_height=cfg.approach_height, descent_lead_time=cfg.descent_lead_time)[:3])
    return {'source': 'measured_visual_motion_and_shared_geometric_intercept_eta',
            'clock': 'host_monotonic', 'decision_host_time_s': float(now),
            'observed_host_time_s': float(snapshot.observed_timestamp),
            'planned_capture_host_time_s': float(now + lead), 'initial_eta_s': lead,
            'prediction_velocity_base_m_s': velocity,
            'capture_translation_base_m': translation,
            'high_offset_from_capture_base_m': high_offset,
            'continuation_offset_from_capture_base_m': np.zeros(3) if direct else velocity * .04,
            'approach_height_m': 0. if direct else cfg.approach_height,
            'descent_lead_time_s': 0. if direct else cfg.descent_lead_time,
            'approach_mode': 'continuous' if direct else 'high',
            'mandatory_high_waypoint': not direct,
            'timing_is_initial_estimate': True}


def select_measured_network_grasp(detection, geometry, state, backend, *,
                                  motion_region, seed=0, config=None, lift_m=.05,
                                  prepared_inference=None, snapshot_provider=None, settings=None):
    """Run the original NN + candidate checks, return capture-time base TCP.

    With live settings, each candidate uses the latest measured track and a
    capture clock estimated from the original Cartesian/joint trajectories.
    Candidate high/low geometry and the execution policy share that clock,
    base-Z clearance and belt advance. Historical callers without settings
    retain their configured preferred intercept.
    The complete shared planner must still certify the actual timed route.
    """
    started = phase_started = time.monotonic()
    timings = {}

    def finish_phase(name):
        nonlocal phase_started
        now = time.monotonic()
        timings[name] = now - phase_started
        phase_started = now

    from experiments.graspgenx_sim.candidates import select_candidates
    from .single_pick import PickRejected

    # Use the configuration already resolved for this inference backend. A
    # caller may override individual selector settings explicitly, but the
    # default must not silently reread config.yaml and discard the backend's
    # experiment settings (including maximum_approach_tilt_deg).
    cfg = _config(getattr(backend, 'config', None))
    cfg.update(dict(config or {}))
    direct = settings is not None and getattr(settings, 'moving_approach_mode', 'high') == 'continuous'
    cfg['direct_contact_selection'] = direct
    finish_phase('configuration_s')
    points = _observed_object_points(detection)
    finish_phase('input_cloud_preparation_s')
    infer_args = (seed, int(cfg['generated_candidates']), int(cfg['returned_candidates']))
    support_frame = _support_frame(detection)
    inference_started = time.monotonic()
    result = (_infer_with_support(backend, points, infer_args, support_frame) if prepared_inference is None else
              prepared_inference.resolve(points, backend, *infer_args, support_frame=support_frame,
                  target_identity=detection.get('target_identity')))
    inference_finished = time.monotonic()
    if detection.get('target_identity') is not None:
        result['metadata'] = {**result.get('metadata',{}),
            'target_identity':identity_dict(detection['target_identity']),
            'inference_start_monotonic':(inference_started if prepared_inference is None else
                prepared_inference.timings.get('inference_start_monotonic')),
            'inference_finish_monotonic':(inference_finished if prepared_inference is None else
                prepared_inference.timings.get('inference_finish_monotonic'))}
    finish_phase('backend_infer_call_s')
    belt_to_base = belt_candidate_frame(detection, motion_region)
    base_to_belt = np.linalg.inv(belt_to_base)
    center = np.median(points.astype(float), axis=0)
    reference = base_to_belt[:3, :3] @ center + base_to_belt[:3, 3]
    motion_plan = None
    if snapshot_provider is not None and settings is not None:
        now = time.monotonic()
        motion_plan = measured_candidate_motion_plan(detection, geometry, state, motion_region,
            snapshot_provider(now), now, settings)
    preferred_s = float(np.mean(motion_region['s_range']))
    preferred_position = np.asarray(motion_region['belt_origin']) + preferred_s * np.asarray(motion_region['belt_direction'])
    preferred_x = float((base_to_belt[:3, :3] @ preferred_position + base_to_belt[:3, 3])[0])
    cfg['preferred_intercept_x'] = preferred_x
    future_translation = belt_to_base[:3, 0] * (preferred_x - reference[0])
    if settings is not None:
        cfg['compare_parallel_jaw_variants'] = True
        # A missing first speed does not become a fixed belt-speed guess.
        # Its capture-only check is explicitly provisional; normal policy
        # observation and the full planner still need real visual motion.
        future_translation = (np.zeros(3) if motion_plan is None else
                              np.asarray(motion_plan['capture_translation_base_m']))
        high_offset = ((np.zeros(3) if direct else np.array([0., 0., .10])) if motion_plan is None else
                       np.asarray(motion_plan['high_offset_from_capture_base_m']))
        continuation_offset = (np.zeros(3) if motion_plan is None else
                               np.asarray(motion_plan['continuation_offset_from_capture_base_m']))
        cfg['intercept_translation_world'] = base_to_belt[:3, :3] @ (future_translation + continuation_offset)
        cfg['pregrasp_offset_world'] = base_to_belt[:3, :3] @ (high_offset - continuation_offset)
        cfg['lift_offset_world'] = base_to_belt[:3, :3] @ np.array([0., 0., lift_m])
        preferred_s = float((center + future_translation - np.asarray(motion_region['belt_origin']))
                            @ np.asarray(motion_region['belt_direction']))
    finish_phase('belt_coordinate_setup_s')
    environment = _BeltCandidateEnvironment(geometry, state, belt_to_base, future_translation,
        lift_m=lift_m, maximum_approach_tilt_deg=cfg['maximum_approach_tilt_deg'])
    if settings is not None:
        cfg['minimum_joint_limit_margin'] = environment.use_execution_joint_margins(state, settings)
        cfg['reachability_ranked_selection'] = bool(cfg.get('reachability_ranked_selection', True))
        environment._ranked_selection_enabled = cfg['reachability_ranked_selection']
        environment._direct_contact_selection = direct
        selection_tcp_position = environment.geometry.ik.forward_pose(state['joints'])[:3]

        def candidate_motion_preview(candidate_capture):
            now = time.monotonic()
            snapshot = None if snapshot_provider is None else snapshot_provider(now)
            return measured_candidate_motion_plan(detection, geometry, state, motion_region,
                snapshot, now, settings, candidate_grasp=candidate_capture,
                current_tcp_position=selection_tcp_position)

        environment._candidate_motion_preview = candidate_motion_preview
        def prepare_candidate_motion(tcp_in_belt):
            nonlocal motion_plan, future_translation, preferred_s
            now = time.monotonic()
            snapshot = None if snapshot_provider is None else snapshot_provider(now)
            candidate_capture = belt_to_base @ np.asarray(tcp_in_belt)
            environment._active_candidate_seed = environment._candidate_hint_seeds.get(row_key(tcp_in_belt))
            motion_plan = measured_candidate_motion_plan(detection,
                geometry, state, motion_region, snapshot, now, settings,
                candidate_grasp=candidate_capture, current_tcp_position=selection_tcp_position)
            if motion_plan is not None:
                # As in the global worker's forecast, use a numerical IK
                # result only to estimate joint travel time, never as motion
                # approval. The selector below re-solves/checks the retimed
                # high point, descent, lift, margins and scene from scratch.
                from ..controllers.joint_trajectory import QuinticJointTrajectory
                high = candidate_capture.copy()
                high[:3, 3] += (np.asarray(motion_plan['capture_translation_base_m'])
                               + np.asarray(motion_plan['high_offset_from_capture_base_m']))
                quaternion = Rotation.from_matrix(high[:3, :3]).as_quat()[[3, 0, 1, 2]]
                estimate = environment.geometry.ik.solve(high[:3, 3], quaternion,
                    qpos_seed=(environment._active_candidate_seed
                               if environment._active_candidate_seed is not None else np.asarray(state['joints'])))
                if estimate.success:
                    # The numerical estimate is a seed, never a certificate.
                    if direct:
                        environment._active_candidate_seed = np.asarray(estimate.joint_positions).copy()
                    duration = QuinticJointTrajectory(state['joints'], estimate.joint_positions,
                        np.full(7, settings.joint_velocity_limit),
                        settings.full_policy_joint_acceleration_limits, minimum_duration=.18).duration
                    needed = duration + motion_plan['descent_lead_time_s'] + 1./settings.control_hz
                    now = time.monotonic()
                    snapshot = snapshot_provider(now)
                    motion_plan = measured_candidate_motion_plan(detection,
                        geometry, state, motion_region, snapshot, now, settings,
                        candidate_grasp=candidate_capture, minimum_lead_s=needed,
                        current_tcp_position=selection_tcp_position)
                    if motion_plan is not None:
                        motion_plan['nominal_joint_motion_time_s'] = duration
                        motion_plan['joint_eta_is_full_path_certificate'] = False
            future_translation = (np.zeros(3) if motion_plan is None else
                                  np.asarray(motion_plan['capture_translation_base_m']))
            high_offset = ((np.zeros(3) if direct else np.array([0., 0., .10])) if motion_plan is None else
                           np.asarray(motion_plan['high_offset_from_capture_base_m']))
            continuation = (np.zeros(3) if motion_plan is None else
                            np.asarray(motion_plan['continuation_offset_from_capture_base_m']))
            environment.geometry.target = np.asarray(geometry.target) + future_translation
            preferred_s = float((center + future_translation - np.asarray(motion_region['belt_origin']))
                                @ np.asarray(motion_region['belt_direction']))
            values = {'intercept_translation_world': base_to_belt[:3, :3] @ (future_translation + continuation),
                      'pregrasp_offset_world': base_to_belt[:3, :3] @ (high_offset - continuation),
                      'candidate_motion_plan': _json(motion_plan)}
            cfg.update(values)
            return values
        environment.prepare_candidate_motion = prepare_candidate_motion
        def candidate_joint_motion_time(joints):
            from ..controllers.joint_trajectory import QuinticJointTrajectory
            return QuinticJointTrajectory(state['joints'], joints,
                np.full(7, settings.joint_velocity_limit), settings.full_policy_joint_acceleration_limits,
                minimum_duration=.18).duration
        environment.candidate_joint_motion_time = candidate_joint_motion_time
    else:
        cfg['reachability_ranked_selection'] = False
    finish_phase('private_environment_setup_s')
    poses = np.einsum('ij,njk->nik', base_to_belt, np.asarray(result['poses']))
    # The selected target is a permitted finger-contact object. Retain the
    # original visual finger-descent query for measured surrounding obstacles;
    # non-finger target checks remain in the RobotModel endpoint/edge hooks.
    scene_points = np.asarray(geometry.scene, dtype=float)
    observed = scene_points @ base_to_belt[:3, :3].T + base_to_belt[:3, 3]
    finish_phase('scene_coordinate_transform_s')
    accepted, rejected = select_candidates(environment, poses, result['scores'], reference, cfg,
                                            qpos_seed=np.asarray(state['joints']),
                                            observed_cloud_world=observed,
                                            candidate_metadata=result['metadata'])
    if settings is not None and accepted:
        motion_plan = accepted[0].get('candidate_motion_plan')
        if motion_plan is not None:
            future_translation = np.asarray(motion_plan['capture_translation_base_m'])
            preferred_s = float((center + future_translation - np.asarray(motion_region['belt_origin']))
                                @ np.asarray(motion_region['belt_direction']))
            cfg.update(candidate_motion_plan=motion_plan,
                intercept_translation_world=base_to_belt[:3, :3] @ (future_translation +
                    np.asarray(motion_plan['continuation_offset_from_capture_base_m'])),
                pregrasp_offset_world=base_to_belt[:3, :3] @ (
                    np.asarray(motion_plan['high_offset_from_capture_base_m']) -
                    np.asarray(motion_plan['continuation_offset_from_capture_base_m'])))
    finish_phase('original_candidate_checks_s')
    provenance = {
        'target_identity':identity_dict(detection.get('target_identity')),
        'network': result['metadata'], 'seed': int(seed),
        'input_frame': 'robot_base', 'input_units': 'metres',
        'input_source': ('fast_stereo_estimated_visible_object_points'
            if detection.get('grasp_depth_refinement', {}).get('applied') else
            'measured_RGBD_segmented_object_points'),
        'grasp_depth_refinement': _json(detection.get('grasp_depth_refinement')),
        'scene_geometry_source': 'native_RGBD_measured_scene',
        'velocity_geometry_source': 'native_RGBD_nero7_xy_kalman',
        'segmentation': 'existing native depth component; RGB green belt rejection through factory D2C when frame is available',
        'target_cloud_filter': _json(detection.get('target_cloud_filter')),
        'input_points_sha256': hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
        'input_point_count': len(points), 'measured_capture_center_base_m': center,
        'candidate_frame': 'measured_belt_longitudinal_x_support_normal_z',
        'T_base_candidate_frame': belt_to_base,
        'preferred_intercept_s_m': preferred_s,
        'candidate_prefilter_translation_base_m': future_translation,
        'candidate_prefilter_lift_m': float(lift_m),
        'candidate_motion_plan': _json(motion_plan),
        'candidate_motion_check': ('capture_only_motion_unavailable' if motion_plan is None else
                                   'shared_timed_high_and_descent_continuation') if settings is not None
                                  else 'historical_preferred_intercept',
        'candidate_position_output': 'original_capture_time_no_intercept_translation',
        'candidate_selection': 'original_experiments.graspgenx_sim.candidates.select_candidates',
        'vertical_reachability': 'original_experiments.graspgenx_sim.reachability.check_vertical_reachability',
        'selected_target_finger_contact_allowed': True,
        'visual_finger_query_source': 'filtered measured scene excluding selected target',
        'CAD_or_geometry_grasp_used': bool(accepted and accepted[0].get('candidate_source') == 'obb_top'),
        'candidate_generation': ('official_obb_top_and_diffusion_same_learned_discriminator'
            if result['metadata'].get('planner') == 'graspmoe' else 'diffusion'),
        'hidden_surface_added': False,
        'hardware_commands_sent': False,
        'source_files_sha256': {str(p.relative_to(PROJECT)): _sha256(p) for p in [
            Path(__file__), EXPERIMENT / 'inference_worker.py', EXPERIMENT / 'candidates.py',
            EXPERIMENT / 'graspmoe_adapter.py',
            EXPERIMENT / 'reachability.py', EXPERIMENT / 'visual_clearance.py',
            EXPERIMENT / 'config.yaml', EXPERIMENT / 'gripper_agx/transforms.json',
            EXPERIMENT / 'gripper_agx/sweep_volume_params.json',
            PROJECT / 'mujoco_conveyor_sorting/controllers/candidate_batch_ik.py',
            PROJECT / 'mujoco_conveyor_sorting/controllers/_candidate_batch_ik_service.py',
            PROJECT / 'mujoco_conveyor_sorting/models/robot.xml']},
        'parameters': cfg, 'accepted_count': len(accepted), 'rejected_count': len(rejected),
        'rejected_candidates': rejected,
        'candidate_pool': getattr(environment, 'candidate_selection_diagnostics', None),
    }
    finish_phase('provenance_s')
    provenance['selection_timings_s'] = timings
    provenance['prepared_inference'] = ({'enabled': False} if prepared_inference is None else
        {'enabled': True, **prepared_inference.timings,
         'scope': 'Network job overlaps scene preparation; candidate checks start after scene installation.'})
    provenance['selection_timing_scope'] = (
        'Host monotonic elapsed time; backend call includes IPC and any unprepared model startup; '
        'with prepared_inference the backend call measures remaining wait only; '
        'network-only inference time is in network.inference_seconds. Artifact writes excluded.')
    artifact_prefix = result.get('artifact_prefix')
    if not accepted:
        timings['selection_total_s'] = time.monotonic() - started
        if artifact_prefix:
            _write_selection_json(Path(artifact_prefix).with_suffix('.selection.json'),
                        {'provenance': provenance, 'accepted': accepted, 'rejected': rejected})
        raise PickRejected('GraspGen-X 已推理，但本帧没有通过原候选检查的抓取；查看 network selection 记录',
                           details=_json(provenance))
    candidate = dict(accepted[0])
    for key in ('T_world_canonical', 'T_world_tcp', 'original_network_pose_world'):
        candidate[key] = belt_to_base @ np.asarray(candidate[key])
    for key in ('tcp_pose_world_wxyz', 'intercept_tcp_pose_world_wxyz', 'pregrasp_tcp_pose_world_wxyz'):
        if candidate[key] is None:
            continue
        value = np.asarray(candidate[key])
        matrix = np.eye(4)
        matrix[:3, 3] = value[:3]
        matrix[:3, :3] = Rotation.from_quat(value[3:][[1, 2, 3, 0]]).as_matrix()
        matrix = belt_to_base @ matrix
        candidate[key] = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    candidate['coordinate_frame'] = 'robot_base'
    # Width remains a measurement along the NN closing axis. Its geometry is
    # diagnostic, not a replacement position/orientation for the network grasp.
    projected = points @ np.asarray(candidate['T_world_canonical'])[:3, 0]
    width = float(np.diff(np.quantile(projected, [.01, .99]))[0])
    if detection.get('target_identity') is not None:
        candidate['target_identity'] = identity_dict(detection['target_identity'])
    provenance['selected_candidate'] = candidate
    finish_phase('output_conversion_s')
    timings['selection_total_s'] = time.monotonic() - started
    if artifact_prefix:
        detail_path = Path(artifact_prefix).with_suffix('.selection.json')
        detail_provenance = {key: value for key, value in provenance.items() if key != 'rejected_candidates'}
        detail_provenance['rejected_candidates_key'] = 'rejected'
        _write_selection_json(detail_path, {'provenance': detail_provenance, 'accepted': accepted,
                                  'rejected': rejected})
        compact = {key: value for key, value in provenance.items()
                   if key not in ('rejected_candidates', 'candidate_pool')}
        compact['rejected_candidates_file'] = str(detail_path)
        compact['candidate_pool_file'] = str(detail_path)
        _write_selection_json(Path(artifact_prefix).with_suffix('.selected_base.json'),
                    {'candidate': candidate, 'provenance': compact})
        provenance = compact
    return {**detection, 'grasp': candidate['T_world_tcp'].copy(), 'width': width,
            'target_geometry_source': (candidate.get('source', 'graspgenx_diffusion') + '_fast_stereo_estimated_points'
                if detection.get('grasp_depth_refinement', {}).get('applied') else
                candidate.get('source', 'graspgenx_diffusion') + '_measured_points'),
            'network_candidate': candidate, 'network_provenance': _json(provenance),
            'network_capture_center_base_m': center.copy(),
            'network_capture_grasp_base': candidate['T_world_tcp'].copy()}


def translate_bound_network_grasp(initial_network_detection, new_depth_detection):
    """Track the selected network offset using new measured object points.

    A later depth detector supplies segmentation and measured translation only;
    it does not replace the selected network quaternion or TCP/contact offset.
    Image exposure/capture timestamps remain the caller's camera records.
    """
    points = _observed_object_points(new_depth_detection)
    center = np.median(points.astype(float), axis=0)
    initial = np.asarray(initial_network_detection['network_capture_center_base_m'])
    displacement = center - initial
    grasp = np.asarray(initial_network_detection['network_capture_grasp_base']).copy()
    grasp[:3, 3] += displacement
    return {**new_depth_detection, 'grasp': grasp,
            'width': initial_network_detection['width'],
            'target_geometry_source': 'bound_graspgenx_pose_with_measured_visual_translation',
            'network_candidate': initial_network_detection['network_candidate'],
            'network_provenance': initial_network_detection['network_provenance'],
            'network_capture_center_base_m': initial.copy(),
            'network_capture_grasp_base': np.asarray(initial_network_detection['network_capture_grasp_base']).copy(),
            'network_tracking_translation_base_m': displacement,
            'network_tracking_translation_source': 'new_measured_object_cloud_median_minus_capture_median'}


def _worker(configuration):
    from experiments.graspgenx_sim.inference_worker import GraspGenXBackend
    cfg = json.loads(Path(configuration).read_text())

    def reply(operation, **value):
        print(REPLY_PREFIX + json.dumps({'operation': operation, **_json(value)},
                                       ensure_ascii=False, allow_nan=False), flush=True)

    try:
        backend = GraspGenXBackend(max_points=int(cfg['maximum_input_points']), planner=cfg['planner'])
    except Exception as error:
        reply('ready', error=f'{type(error).__name__}: {error}')
        return
    reply('ready', provenance=backend.provenance)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('operation') == 'close':
            return
        try:
            result = backend.infer(np.load(request['input'], allow_pickle=False),
                                   request['seed'], request['num_grasps'], request['topk'],
                                   support_frame=request.get('support_frame'))
            output = Path(request['output'])
            np.savez_compressed(output, poses=result['poses'], scores=result['scores'])
            result['metadata']['input_file'] = request['input']
            result['metadata']['input_file_sha256'] = _sha256(request['input'])
            _write_json(output.with_suffix('.json'), result['metadata'])
            reply('infer', metadata=result['metadata'])
        except Exception as error:
            reply('infer', error=f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--configuration', type=Path, required=True)
    args = parser.parse_args()
    if not args.worker:
        parser.error('该模块入口仅用于独立网络进程')
    _worker(args.configuration)

"""Read-only recorded joint demonstrations and certified task-specific replay."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .motion_profile import select_motion_profile
from .trajectory import retime_samples

JOINT_NAMES = [f'joint{i}' for i in range(1, 8)]


def model_digest():
    root = Path(__file__).resolve().parents[2]
    return sha256((root / 'mujoco_conveyor_sorting/models/robot.xml').read_bytes()).hexdigest()


def provenance(config):
    return {'joint_names': JOINT_NAMES, 'unit': 'rad', 'base_frame': config.raw['base_frame'],
            'tcp_sha256': sha256(Path(config.raw['tcp_calibration_file']).read_bytes()).hexdigest(),
            'robot_model_sha256': model_digest(), 'bin_id': config.default_bin,
            'can_channel': config.raw['hardware']['can_channel'],
            'firmware': config.raw['hardware']['firmware']}


@dataclass(frozen=True)
class TaughtSegment:
    kind: str
    times: np.ndarray
    joints: np.ndarray
    path: str
    sha256: str

    @classmethod
    def load(cls, path, config, expected_kind):
        path = Path(path).expanduser().resolve(strict=True)
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError(f'{path}: 示教文件必须是 JSON 对象')
        if raw.get('schema') != 'nero_taught_joint_path_v1' or raw.get('segment') != expected_kind:
            raise ValueError(f'{path}: 示教文件格式或段类型不匹配')
        for key, value in provenance(config).items():
            if raw.get('provenance', {}).get(key) != value:
                raise ValueError(f'{path}: 示教记录与当前配置不匹配: {key}')
        times = np.asarray(raw.get('elapsed_s'), dtype=float)
        joints = np.asarray(raw.get('joints_rad'), dtype=float)
        if (times.ndim != 1 or len(times) < 4 or joints.shape != (len(times), 7)
                or not np.isfinite(times).all() or not np.isfinite(joints).all()
                or abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0)):
            raise ValueError(f'{path}: 示教需至少四个时间递增的七轴样本')
        times.setflags(write=False)
        joints.setflags(write=False)
        return cls(expected_kind, times, joints, str(path), sha256(raw_bytes).hexdigest())

    def summary(self):
        return {'segment': self.kind, 'path': self.path, 'sha256': self.sha256,
                'samples': len(self.times), 'recorded_duration_s': float(self.times[-1])}


def prepare_taught_timing(config, segment, model, *, payload, cancelled=lambda: False):
    """Cache only scene-independent timing. Geometry always needs the current task."""
    motion = select_motion_profile(config, payload=payload)
    jerk = config.raw.get('taught_joint_jerk_limit_rad_s3')
    maximum_duration = None
    if segment.kind == 'transfer':
        jerk = config.raw.get('taught_transfer_jerk_limit_rad_s3', jerk)
        maximum_duration = config.raw.get('taught_transfer_max_duration_s')
    key = (segment.sha256, payload, tuple(motion.velocity_limits), tuple(motion.acceleration_limits),
           tuple(model.command_lower), tuple(model.command_upper), config.settings.control_hz,
           config.raw.get('taught_transfer_timing_policy'), jerk, maximum_duration)
    cache = config.raw.setdefault('_taught_timing_cache', {})
    if key in cache:
        if cancelled():
            raise PickRejected('示教轨迹规划已取消')
        return cache[key], True
    arguments = dict(identity=None, version=0, stage='taught_'+segment.kind)
    if segment.kind == 'transfer' and config.raw.get('taught_transfer_timing_policy') == 'path_preserving_variable_speed':
        from .taught_timing import retime_taught_path
        result = retime_taught_path(segment.times, segment.joints, motion,
            model.command_lower, model.command_upper, **arguments, cancelled=cancelled, jerk_limit=jerk)
    else:
        result = retime_samples(segment.times, segment.joints, motion,
            model.command_lower, model.command_upper, **arguments)
        if jerk is not None:
            from .taught_timing import limit_jerk
            result = limit_jerk(result, jerk, config.settings.control_hz)
    if cancelled():
        raise PickRejected('示教轨迹规划已取消')
    if maximum_duration is not None:
        if result.duration > maximum_duration + 1e-9:
            raise PickRejected('示教投放段无法在时长上限内满足运动约束', details={
                'stage': result.stage, 'duration_s': result.duration,
                'maximum_duration_s': maximum_duration, 'joint_jerk_limit_rad_s3': jerk})
        result = replace(result, diagnostics={**result.diagnostics,
            'maximum_duration_s': maximum_duration})
    cache[key] = result
    return result, False


def certify_taught(planner, config, segment, state, start_joints, *, payload, cancelled, clock=time.monotonic):
    """Retime the measured samples, then certify the emitted curve in the current scene."""
    begun = clock()
    q0 = np.asarray(start_joints, float)
    if (np.max(abs(q0 - segment.joints[0])) > 1e-8 or
            np.max(abs(q0 - np.asarray(state['joints']))) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD):
        raise PickRejected('实际反馈或连接终点与示教起点不一致')
    if payload and planner.payload_local is None:
        raise PickRejected('示教搬运缺少当前工件的载物估计')
    if cancelled():
        raise PickRejected('示教轨迹规划已取消')
    def aborted():
        return cancelled() or clock()-begun > config.raw['planning_budget_s']
    motion = select_motion_profile(config, payload=payload)
    planner.version += 1
    template, cached = prepare_taught_timing(config, segment, planner.model, payload=payload, cancelled=aborted)
    result = replace(template, identity=planner.identity, version=planner.version)
    planner.check_path(result, float(state['width_m']), payload=payload, cancelled=aborted)
    if aborted():
        raise PickRejected('示教轨迹认证已取消或超过规划预算')
    return replace(result, planning_s=clock()-begun,
        diagnostics={**result.diagnostics, **motion.diagnostics(), 'taught_segment': segment.summary(),
                     'timing_precomputed': cached,
                     'timing_policy': result.diagnostics.get('timing_policy',
                         'recorded_relative_time_uniformly_retimed_rest_to_rest'),
                     'scene_certificate': 'task_frozen_scene_and_current_payload_estimate'})

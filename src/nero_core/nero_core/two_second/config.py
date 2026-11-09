"""Explicit physical-cell configuration, with no simulation geometry defaults."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import yaml
from mujoco_conveyor_sorting.calibration.geometry import rigid_transform
from mujoco_conveyor_sorting.runtime.single_pick import PickSettings
from .reference import DEFAULT_FREEZE_UPSTREAM_DISTANCE_M
from .motion_profile import loaded_acceleration
from .feedback_gates import GRIP_CONFIRMATION_DWELL_S


def vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} 必须为 {size} 个有限数值')
    return array.copy()


@dataclass(frozen=True)
class CellConfig:
    raw: dict
    settings: PickSettings
    source: Path
    source_sha256: str
    tcp: np.ndarray
    canonical_tcp: np.ndarray
    direction: np.ndarray
    follow_bounds: np.ndarray
    workspace: np.ndarray
    standby: np.ndarray
    bins: dict
    default_bin: str
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        raw = yaml.safe_load(content)
        if not isinstance(raw, dict) or raw.get('strategy') != 'two_second':
            raise ValueError('strategy 必须显式为 two_second')
        required = ('base_frame', 'camera_calibration_file', 'tcp_calibration_file',
                    'joint_names', 'belt_direction_base', 'follow_interval_base_m',
                    'workspace_base_m', 'standby_joint_rad', 'bins_base',
                    'unknown_category_bin', 'camera', 'hardware')
        required += ('maximum_target_height_m',)
        missing = [name for name in required if raw.get(name) is None]
        if missing:
            raise ValueError('现场必填配置缺失: ' + ', '.join(missing))
        height = float(raw['maximum_target_height_m'])
        if not np.isfinite(height) or height <= .012:
            raise ValueError('maximum_target_height_m 必须高于原检测下限12mm')
        raw['maximum_target_height_m'] = height
        mode=raw.setdefault('perception_detection_mode','rgb_depth')
        if mode not in ('legacy_depth','depth_plane','rgb','rgb_depth'):
            raise ValueError('perception_detection_mode 不受原Nero7检测器支持')
        if not isinstance(raw.setdefault('isolate_boundary_depth_gaps', False), bool):
            raise ValueError('isolate_boundary_depth_gaps 必须为布尔值')
        if raw.setdefault('perception_plane_mode','ransac3d') not in ('fixed','ransac3d'):
            raise ValueError('perception_plane_mode 必须为 fixed 或 ransac3d')
        if raw.setdefault('perception_search_region','height_prism') not in (
                'height_prism','measured_belt_footprint'):
            raise ValueError('perception_search_region 必须为 height_prism 或 measured_belt_footprint')
        from .belt_collision_scene import OBSERVED_SCENE, SCOPES
        scope = raw.setdefault('environment_collision_scope', OBSERVED_SCENE)
        if scope not in SCOPES:
            raise ValueError('environment_collision_scope 必须为 observed_scene 或 measured_belt_surface_only')
        margin = float(raw.setdefault('workpiece_collision_margin_m', .004))
        # The robot collision broad phase is built for at most 8 mm.
        if not np.isfinite(margin) or not 0 <= margin <= .008:
            raise ValueError('workpiece_collision_margin_m 必须在 0 到 0.008 m 之间')
        raw['workpiece_collision_margin_m'] = margin
        payload_belt_margin = float(raw.setdefault('payload_belt_clearance_m', .008))
        if not np.isfinite(payload_belt_margin) or payload_belt_margin < 0:
            raise ValueError('payload_belt_clearance_m 必须是非负有限数值')
        raw['payload_belt_clearance_m'] = payload_belt_margin
        if not isinstance(raw.setdefault('taught_playback_enabled', True), bool):
            raise ValueError('taught_playback_enabled 必须为布尔值')
        if raw.setdefault('obb_grasp_orientation_reference', 'current_tcp') not in (
                'current_tcp', 'taught_transfer_start'):
            raise ValueError('obb_grasp_orientation_reference 必须为 current_tcp 或 taught_transfer_start')
        if (raw['obb_grasp_orientation_reference'] == 'taught_transfer_start'
                and not raw['taught_playback_enabled']):
            raise ValueError('使用示教起点排序 OBB 时必须启用示教轨迹')
        if raw.setdefault('taught_transfer_timing_policy', 'recorded_uniform') not in (
                'recorded_uniform', 'path_preserving_variable_speed'):
            raise ValueError('taught_transfer_timing_policy 不受支持')
        if not isinstance(raw.setdefault('prepared_taught_cycle_enabled', False), bool):
            raise ValueError('prepared_taught_cycle_enabled 必须为布尔值')
        for key in ('taught_joint_jerk_limit_rad_s3', 'taught_transfer_jerk_limit_rad_s3',
                    'taught_transfer_max_duration_s'):
            if raw.get(key) is not None:
                value = float(raw[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} 必须为正有限数')
                raw[key] = value
        taught_files = ('taught_transfer_file', 'taught_return_file')
        configured_taught = [raw.get(key) is not None for key in taught_files]
        if any(configured_taught) and not all(configured_taught):
            raise ValueError('示教默认文件须同时配置 taught_transfer_file 和 taught_return_file')
        if all(configured_taught):
            for key in taught_files:
                if not isinstance(raw[key], str) or not raw[key].strip():
                    raise ValueError(f'{key} 必须为非空路径')
                taught_path = Path(raw[key]).expanduser()
                if not taught_path.is_absolute():
                    taught_path = path.parent/taught_path
                # The selected pair is opened by the entry point. Explicit
                # CLI overrides and disabled playback need not open defaults.
                raw[key] = str(taught_path.resolve())
        if not isinstance(raw.setdefault('grasp_workpiece_collision_check_enabled', True), bool):
            raise ValueError('grasp_workpiece_collision_check_enabled 必须为布尔值')
        if not isinstance(raw.setdefault('recovery_allow_workpiece_contact', False), bool):
            raise ValueError('recovery_allow_workpiece_contact 必须为布尔值')
        if raw['joint_names'] != [f'joint{i}' for i in range(1, 8)]:
            raise ValueError('joint_names 必须按当前厂家接口 joint1..joint7 排列')
        for key in ('camera_calibration_file', 'tcp_calibration_file'):
            resolved = Path(raw[key]).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            raw[key] = str(resolved.resolve(strict=True))
        tcp_content = Path(raw['tcp_calibration_file']).read_bytes()
        tcp_file = yaml.safe_load(tcp_content)
        if not isinstance(tcp_file, dict) or not tcp_file.get('calibration_id'):
            raise ValueError('TCP 文件必须声明来源及 calibration_id')
        tcp_source = tcp_file.get('source')
        if tcp_source == 'model_nominal':
            # The operator may explicitly confirm a particular model, without
            # misrepresenting it as an independently measured calibration.
            digest = hashlib.sha256(tcp_content).hexdigest()
            if raw.get('confirmed_nominal_tcp_sha256') != digest:
                raise ValueError('模型名义 TCP 需要确认文件 SHA-256；与确认值不一致')
        elif tcp_source != 'physical_measurement':
            raise ValueError('TCP 来源必须为 physical_measurement 或已确认的 model_nominal')
        raw['tcp_source'] = tcp_source
        raw['tcp_calibration_id'] = tcp_file['calibration_id']
        tcp = rigid_transform(tcp_file['T_flange_tcp'], 'T_flange_tcp')
        canonical = rigid_transform(tcp_file['T_canonical_tcp'], 'T_canonical_tcp')
        direction = vector(raw['belt_direction_base'], 3, 'belt_direction_base')
        if np.linalg.norm(direction) < 1e-8:
            raise ValueError('传送带方向不能为零')
        direction /= np.linalg.norm(direction)
        follow = vector(raw['follow_interval_base_m'], 2, 'follow_interval_base_m')
        workspace = np.asarray(raw['workspace_base_m'], dtype=float)
        if follow[0] >= follow[1] or workspace.shape != (3, 2) or not np.isfinite(workspace).all() or np.any(workspace[:, 0] >= workspace[:, 1]):
            raise ValueError('抓取区间或基座工作区上下界无效')
        bins = {str(key): rigid_transform(value, 'bin pose') for key, value in raw['bins_base'].items()}
        default_bin = str(raw['unknown_category_bin'])
        if default_bin not in bins:
            raise ValueError('未知类别料框必须存在于 bins_base')
        settings = PickSettings(**raw.get('pick_settings', {}))
        settings.validate()
        geometry_hz = float(raw.get('geometry_check_hz', 2.*settings.control_hz))
        if not np.isfinite(geometry_hz) or geometry_hz <= 0:
            raise ValueError('geometry_check_hz 必须为正有限数')
        raw['geometry_check_hz'] = geometry_hz
        if not isinstance(raw.setdefault('runtime_command_acceleration_check_enabled', False), bool):
            raise ValueError('runtime_command_acceleration_check_enabled 必须为布尔值')
        # Deleting a site override restores the existing strategy profile.
        # An explicit null/NaN/infinity is never an unbounded hardware profile.
        velocity_key = 'joint_velocity_limits_rad_s'
        acceleration_key = 'joint_acceleration_limits_rad_s2'
        velocity = vector(raw.get(velocity_key, [settings.joint_velocity_limit]*7), 7, 'velocity limits')
        acceleration = vector(raw.get(acceleration_key, [settings.joint_acceleration_limit]*7), 7, 'acceleration limits')
        raw['motion_limit_sources'] = {
            'velocity': 'site_override' if velocity_key in raw else 'PickSettings',
            'acceleration': 'site_override' if acceleration_key in raw else 'PickSettings'}
        velocity_cap = np.full(7, settings.joint_velocity_limit)
        velocity_cap_key = 'manufacturer_joint_velocity_limits_rad_s'
        if velocity_cap_key in raw:
            manufacturer_velocity = vector(raw[velocity_cap_key], 7, velocity_cap_key)
            velocity_source = raw.get('manufacturer_velocity_limit_source')
            if (np.any(manufacturer_velocity <= 0) or not isinstance(velocity_source, str)
                    or not velocity_source.strip()):
                raise ValueError('厂家速度上限须为正数并提供来源说明')
            # This percentage is a host-side planning policy, not a claim
            # about how the device firmware scales its physical limit.
            velocity_cap = manufacturer_velocity * settings.speed_percent / 100.
            raw['motion_limit_sources']['velocity_cap'] = velocity_source
        # The old PickCycle's software default is not a manufacturer limit.
        # A cell may supply its documented capability without changing the
        # legacy controller or writing a new limit to device firmware.
        capability_key = 'manufacturer_joint_acceleration_limits_rad_s2'
        acceleration_cap = np.full(7, settings.joint_acceleration_limit)
        if capability_key in raw:
            acceleration_cap = vector(raw[capability_key], 7, capability_key)
            source = raw.get('manufacturer_acceleration_limit_source')
            if np.any(acceleration_cap <= 0) or not isinstance(source, str) or not source.strip():
                raise ValueError('厂家加速度上限须为正数并提供来源说明')
            raw['motion_limit_sources']['acceleration_cap'] = source
        if np.any(velocity <= 0) or np.any(acceleration <= 0):
            raise ValueError('关节速度和加速度须为正数')
        if np.any(velocity > velocity_cap + 1e-10) or np.any(acceleration > acceleration_cap + 1e-10):
            raise ValueError('现场上位机限速不能超出所选速度配置及有来源的加速度上限')
        _, loaded_source = loaded_acceleration(raw, acceleration)
        raw['motion_limit_sources']['loaded_acceleration'] = loaded_source
        if raw.setdefault('planning_reserve_mode', 'fixed') not in ('fixed', 'measured'):
            raise ValueError('planning_reserve_mode 须为 fixed 或 measured')
        for key, default_value in (('command_advance_s', .1), ('minimum_speed_span_s', .45),
                                   ('planning_reserve_s', 1.5), ('maximum_observation_age_s', 1.),
                                   ('freeze_upstream_distance_m', DEFAULT_FREEZE_UPSTREAM_DISTANCE_M),
                                   ('maximum_measured_speed_m_s', .30), ('planning_budget_s', 8.)):
            value = float(raw.get(key, default_value))
            allow_zero = (key == 'command_advance_s' or
                          (key == 'planning_reserve_s' and raw['planning_reserve_mode'] == 'measured'))
            if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f'{key} 无效')
            raw[key] = value
        if raw['minimum_speed_span_s'] > .6:
            raise ValueError('测速资格窗口沿用0.6秒，minimum_speed_span_s不能超过该值')
        from .velocity_source import validate_velocity_configuration
        validate_velocity_configuration(raw)
        follow_window = float(settings.full_plan_close_window_s)
        if not np.isfinite(follow_window) or follow_window <= 0:
            raise ValueError('闭合跟随段须为正有限秒数')
        profile = raw.setdefault('trajectory_profile', 'legacy')
        if profile not in ('legacy', 'smooth'):
            raise ValueError('trajectory_profile 必须为 legacy 或 smooth')
        if raw.setdefault('intercept_strategy', 'classic') not in ('classic', 'synchronized'):
            raise ValueError('intercept_strategy 必须为 classic 或 synchronized')
        if raw['intercept_strategy'] == 'synchronized' and profile != 'smooth':
            raise ValueError('synchronized 抓取策略需要 smooth 轨迹')
        position = raw.setdefault('freeze_position_upstream_m', None)
        tolerance = float(raw.setdefault('freeze_position_tolerance_m', .02))
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('freeze_position_tolerance_m 必须为正有限数')
        raw['freeze_position_tolerance_m'] = tolerance
        if position is not None:
            position = float(position)
            if not np.isfinite(position) or position <= tolerance:
                raise ValueError('freeze_position_upstream_m 必须为大于取帧容差的正有限数')
            if raw['intercept_strategy'] != 'synchronized':
                raise ValueError('固定位置冻结需要 synchronized 策略')
            raw['freeze_position_upstream_m'] = position
        if raw['planning_reserve_mode'] == 'measured' and (profile != 'smooth' or raw['intercept_strategy'] != 'synchronized'):
            raise ValueError('measured 规划时序需要 smooth synchronized 策略')
        synchronized_duration = float(raw.get('synchronized_duration_s', .2))
        if not np.isfinite(synchronized_duration) or synchronized_duration < 0:
            raise ValueError('synchronized_duration_s 须为非负有限秒数')
        raw['synchronized_duration_s'] = synchronized_duration
        adaptive = raw.setdefault('synchronized_adaptive_freeze', False)
        if not isinstance(adaptive, bool):
            raise ValueError('synchronized_adaptive_freeze 必须为布尔值')
        if adaptive:
            if raw['intercept_strategy'] != 'synchronized':
                raise ValueError('自适应冻结需要 synchronized 策略')
            maximum_freeze = float(raw.get('synchronized_max_freeze_upstream_distance_m', .35))
            if not np.isfinite(maximum_freeze) or maximum_freeze < raw['freeze_upstream_distance_m']:
                raise ValueError('自适应冻结的最大上游距离必须不小于基础距离')
            raw['synchronized_max_freeze_upstream_distance_m'] = maximum_freeze
        if raw.setdefault('closure_policy', 'feedback') not in ('feedback', 'scheduled_attempt'):
            raise ValueError('closure_policy 必须为 feedback 或 scheduled_attempt')
        if profile == 'smooth':
            prealign = float(raw.get('prealign_orientation_s', 0.))
            if not np.isfinite(prealign) or prealign < 0:
                raise ValueError('prealign_orientation_s 须为非负有限秒数')
            if raw['intercept_strategy'] == 'synchronized' and prealign <= 0:
                raise ValueError('synchronized 策略需要正的连续接入时间')
            raw['prealign_orientation_s'] = prealign
            rotation=float(raw.get('rotate_first_s',0.))
            if not np.isfinite(rotation) or rotation<0 or (rotation>0 and raw['intercept_strategy']!='synchronized'):
                raise ValueError('rotate_first_s 须为非负有限秒数，独立预转仅用于 synchronized 策略')
            raw['rotate_first_s']=rotation
            retime_rotation=raw.setdefault('rotate_first_retime_enabled',False)
            if not isinstance(retime_rotation,bool) or (retime_rotation and (
                    raw['intercept_strategy']!='synchronized' or rotation<=0)):
                raise ValueError('rotate_first_retime_enabled 需要布尔值及正的同步预转时长')
            if 'synchronized_join_candidates_s' in raw:
                joins=np.asarray(raw['synchronized_join_candidates_s'],float)
                if (raw['intercept_strategy']!='synchronized' or joins.ndim!=1 or not len(joins)
                        or not np.isfinite(joins).all() or np.any(joins<=0)
                        or len(np.unique(joins))!=len(joins)):
                    raise ValueError('同步接入候选必须是互不重复的正有限时长')
                raw['synchronized_join_candidates_s']=joins.tolist()
            knot_hz = float(raw.get('planning_knot_hz', min(50., settings.control_hz)))
            if not np.isfinite(knot_hz) or not 0 < knot_hz <= settings.control_hz:
                raise ValueError('planning_knot_hz 必须为不高于控制频率的正有限数')
            raw['planning_knot_hz'] = knot_hz
            fallback_hz = float(raw.get('planning_knot_hz_fallback', knot_hz))
            if not np.isfinite(fallback_hz) or not knot_hz <= fallback_hz <= settings.control_hz:
                raise ValueError('planning_knot_hz_fallback 必须不低于初始频率且不高于控制频率')
            raw['planning_knot_hz_fallback'] = fallback_hz
            durations = np.asarray(raw.get('approach_duration_candidates_s', [2., 3., 4., 5.5, 6.5]), float)
            if durations.ndim != 1 or not len(durations) or not np.isfinite(durations).all() or np.any(durations <= 0) or np.any(np.diff(durations) <= 0):
                raise ValueError('平滑接近候选时长须为递增正有限秒数')
            raw['approach_duration_candidates_s'] = durations.tolist()
            for key, default in (('brake_duration_s', 1.5), ('candidate_planning_budget_s', 1.5)):
                value = float(raw.get(key, default))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(key+' 须为正有限秒数')
                raw[key] = value
        if follow_window < GRIP_CONFIRMATION_DWELL_S:
            # Only scheduled_attempt continues grip confirmation into braking.
            # Feedback-gated closure requires the full dwell during follow.
            if (profile != 'smooth' or raw['intercept_strategy'] != 'synchronized'
                    or raw['closure_policy'] != 'scheduled_attempt'):
                raise ValueError(f'不足{GRIP_CONFIRMATION_DWELL_S:g}秒的跟随仅支持平滑同步定时闭爪，确认在减速期间继续')
            # Equality is allowed: fresh grip feedback is still checked while
            # holding the certified endpoint for arrival. This admission check
            # never replaces the measured dwell or authorizes a lift.
            if follow_window + raw['brake_duration_s'] < GRIP_CONFIRMATION_DWELL_S-1e-9:
                raise ValueError(f'跟随与减速总时长不得短于{GRIP_CONFIRMATION_DWELL_S:g}秒夹持确认时间')
        if raw['hardware'].get('joint_command_mode') not in ('j', 'js'):
            raise ValueError('必须明确选择厂家 j 或 js 接口')
        if not raw['hardware'].get('can_channel') or not raw['hardware'].get('firmware'):
            raise ValueError('必须配置 CAN 通道和厂家固件版本')
        offset = float(raw['hardware'].get('joint_receive_offset_s', 0.))
        if not np.isfinite(offset):
            raise ValueError('joint_receive_offset_s 必须为有限秒数')
        raw['hardware']['joint_receive_offset_s'] = offset
        return cls(raw, settings, path, hashlib.sha256(content).hexdigest(), tcp,
                   canonical, direction, follow, workspace,
                   vector(raw['standby_joint_rad'], 7, 'standby joints'), bins,
                   default_bin, velocity, acceleration)

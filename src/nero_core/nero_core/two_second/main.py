"""Physical integration entry. Default only validates files, opening no devices."""
import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import json
import importlib.util
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import time

from .config import CellConfig
from .diagnostics import json_value, RunJournal, ProgressMonitor
from .console import ConsoleProgress, short
from .motion_profile import select_motion_profile
from .startup import prepare_start_state, return_wait_policy, MoveCancelled
from .velocity_source import is_fixed, with_fixed_belt_speed, policy as velocity_policy
from .feedback_gates import GRIP_CONFIRMATION_DWELL_S
from ..pick_origin_policy import POLICY_BY_GROUP, WAIT_ONLY


def provenance(config):
    files = [config.source, Path(config.raw['tcp_calibration_file']),
             Path(config.raw['camera_calibration_file'])]
    if 'execution_delay_window' in config.raw:
        files.append(Path(config.raw['execution_delay_window']['profile_file']))
    if config.raw.get('experimental_direct_drop_guard'):
        files.append(Path(config.raw['experimental_direct_drop_guard']['source_file']))
    files += sorted(Path(__file__).parent.glob('*.py'))
    files.append(Path(__file__).parent.parent/'move_to_wait.py')
    files.append(Path(__file__).parent.parent/'pick_origin_policy.py')
    root=Path(__file__).resolve().parents[2]
    files += [root/'mujoco_conveyor_sorting/runtime'/name for name in (
        'nero_pick_hardware.py','single_pick.py','joint_stream_sender.py','execution_trace.py','archive_work.py','frame_products.py','scene_region_filter.py',
        'nero7_belt_velocity.py','nero7_occlusion_tracker.py','nero7_fused_detection.py','nero7_tracking_roi.py','nero7_boundary_depth.py','nero7_height_split.py','workpiece_admission.py',
        'pointcloud_speed_tracker.py','frame_delivery.py','backends/orbbec.py',
        'belt_surface.py','local_target_geometry.py','shape_pick.py','depth_separation.py',
        'fast_stereo.py','_fast_stereo_worker.py','grasp_depth.py')]
    files += [root/'mujoco_conveyor_sorting/models/robot.xml',
              root/'mujoco_conveyor_sorting/configs/robot.yaml',
              root/'mujoco_conveyor_sorting/perception/rgbd_vision_pipeline.py']
    vendor=root/'vendor/nero7_speed_reference'
    files.append(vendor/'source_manifest.json')
    files += [vendor/'nero7_perception'/name for name in (
        'block_tracker.py','kalman_motion.py','detection.py','base_plane.py','geometry.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def preview_environment():
    """Make checkout imports available to the separate read-only viewer."""
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    inherited = env.get('PYTHONPATH')
    env['PYTHONPATH'] = source_root + (os.pathsep + inherited if inherited else '')
    return env


def main(argv=None, *, freeze_source='tracked', candidate_retry_policy=None,
         phase_timing_hardware_test=False, hover_descent_entry=False, executor_class=None):
    with ExitStack() as lifetime:
        return _main(argv, lifetime, freeze_source=freeze_source, candidate_retry_policy=candidate_retry_policy,
                     phase_timing_hardware_test=phase_timing_hardware_test, hover_descent_entry=hover_descent_entry,
                     executor_class=executor_class)


def _main(argv,lifetime,*,freeze_source='tracked',candidate_retry_policy=None,
         phase_timing_hardware_test=False, hover_descent_entry=False, executor_class=None):
    if freeze_source not in ('tracked','single_exposure_native_depth'):
        raise ValueError('unknown freeze acquisition source')
    if candidate_retry_policy not in (None,'per_grasp','coarse_first'):
        raise ValueError('unknown candidate retry policy')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/two_second_hardware'))
    parser.add_argument('--execute', action='store_true', help='连接实际相机/CAN并执行；不会使能设备')
    parser.add_argument('--check-config', action='store_true', help='只检查标定和配置，不导入厂家SDK')
    parser.add_argument('--fast-stereo', action='store_true', help='同曝光左右红外Fast工件点云重建；启动时先检查并预热GPU')
    parser.add_argument('--object-relative-approach', action=argparse.BooleanOptionalAction, default=None,
                        help='Fast 单帧抓取时使用现场配置的高位同步和相对下降轨迹')
    parser.add_argument('--rgb-cloud-completion', action=argparse.BooleanOptionalAction, default=None,
                        help='按可见 RGB 轮廓扩展工件原生深度点云，默认关闭')
    parser.add_argument('--depth-separation', action=argparse.BooleanOptionalAction, default=None,
                        help='OBB前启用原生深度分离；保留完整碰撞点云')
    cycles=parser.add_mutually_exclusive_group()
    cycles.add_argument('--count', type=int, default=1, help='完成指定数量的反馈周期后退出')
    cycles.add_argument('--continuous', action='store_true', help='连续抓取，Ctrl+C退出')
    parser.add_argument('--next-pick-group', choices=tuple(POLICY_BY_GROUP),
        help='连续抓取起点策略：A 只用等待位，B 放料位优先，C 按发送期限优先等待位')
    parser.add_argument('--nominal-feed-interval-s', type=float, help='实验标称投件间隔（秒），供记录使用')
    belt_speed = parser.add_mutually_exclusive_group()
    belt_speed.add_argument('--belt-speed-cm-s', type=float,
        help='本次运行的固定预测带速，单位cm/s；真机执行必须明确指定或由配置提供')
    belt_speed.add_argument('--fixed-belt-speed-m-s', type=float,
        help='本次运行的固定预测带速，单位m/s；真机执行必须明确指定或由配置提供')
    parser.add_argument('--preview', action='store_true', help='共用当前相机来源，打开只读画面及碰撞预测窗口')
    parser.add_argument('--web-preview', action='store_true', help='共享原子预览文件供本地网页读取，不启动额外相机或Qt窗口')
    parser.add_argument('--record-rgbd', action='store_true', help='异步记录同源RGB-D，供事后重建场景点云')
    parser.add_argument('--native-observation-shadow', action='store_true',
                        help='固定带速：旁路记录原生工件几何和反馈配对，供离线对照')
    parser.add_argument('--no-save', action='store_true',
        help='关闭本次抓取的全部场次归档；实时预览使用临时共享内存，覆盖 --record-rgbd 和补救模式的自动录制')
    parser.add_argument('--enable-delivery-repair', action='store_true',
        help='交付超时自动修正复用旧轨迹；开启工件碰撞检查和RGB-D记录，保留现场阶段时长')
    parser.add_argument('--enable-hover-fallback', action='store_true',
        help='实验：普通规划无法交付时尝试跟随下降补救；实际带速不超过28.4 cm/s')
    parser.add_argument('--taught-transfer', type=Path, help='覆盖配置中的载物投放示教文件，须与 --taught-return 成对提供')
    parser.add_argument('--taught-return', type=Path, help='覆盖配置中的空爪返回示教文件，须与 --taught-transfer 成对提供')
    parser.add_argument('--experimental-direct-drop-guard', type=Path,
        help='实验：按给定名义料框高度文件规划直接投放；不会证明实际物理间隙')
    args = parser.parse_args(argv)
    preview_enabled = args.preview or args.web_preview
    if args.check_config and args.execute:
        parser.error('--check-config 与 --execute 互斥')
    if args.count < 1:
        parser.error('--count 必须大于零')
    interval = args.nominal_feed_interval_s
    if interval is not None and (not math.isfinite(interval) or interval <= 0 or not args.next_pick_group):
        parser.error('--nominal-feed-interval-s 需要正有限秒数和明确的 --next-pick-group')
    if (args.taught_transfer is None) != (args.taught_return is None):
        parser.error('使用示教时必须同时指定 --taught-transfer 和 --taught-return')
    # All configuration checks happen before importing a camera or CAN SDK.
    hover_fallback_config = None
    try:
        config = CellConfig.load(args.config)
        native_band = (freeze_source == 'single_exposure_native_depth'
            and config.raw.get('tracked_freeze_range') == 'detector_footprint'
            and config.raw.get('tracked_freeze_publish_before_upstream_m') is not None)
        if native_band and args.depth_separation is None:
            config.raw['depth_separation_enabled'] = True
        if args.fast_stereo:
            config.raw['fast_stereo_enabled'] = True
        if args.rgb_cloud_completion is not None:
            config.raw['rgb_cloud_completion_enabled'] = args.rgb_cloud_completion
        relative_site = native_band and config.raw['fast_stereo_enabled'] and config.raw.get('fast_object_relative_approach') is not None
        if args.object_relative_approach is True and not relative_site:
            raise ValueError('相对接近需选用带阶段配置的 Fast 单帧冻结入口')
        if args.object_relative_approach is False or not native_band or not config.raw['fast_stereo_enabled']:
            config.raw.pop('object_relative_approach', None)
        if relative_site and args.object_relative_approach is not False:
            from .object_approach import configure as configure_object_approach
            configure_object_approach(config)
        if args.depth_separation is not None:
            config.raw['depth_separation_enabled'] = args.depth_separation
        from .depth_separation import validate as validate_depth_separation
        validate_depth_separation(config.raw)
        candidate_retry_policy = candidate_retry_policy or config.raw['candidate_retry_policy']
        from .delivery_repair import DEFAULTS as REPAIR_DEFAULTS, validate as validate_repair, enabled as repair_enabled
        if args.enable_delivery_repair:
            config.raw['delivery_repair'] = {**REPAIR_DEFAULTS,
                **config.raw.get('delivery_repair', {}), 'enabled': True}
            config.raw['grasp_workpiece_collision_check_enabled'] = True
            validate_repair(config.raw)
        if repair_enabled(config):
            args.record_rgbd = True
        if args.execute:
            from .phase_timing import authorize_hardware_test
            authorize_hardware_test(config, enabled=phase_timing_hardware_test)
        if candidate_retry_policy == 'coarse_first' and (
                config.raw['trajectory_profile'] != 'smooth' or
                config.raw['intercept_strategy'] != 'synchronized'):
            raise ValueError('粗算优先入口需要平滑同步抓取策略')
        # Explicit experiment entries override the configured production policy.
        config.raw['candidate_retry_policy'] = candidate_retry_policy
        speed_m_s = (args.belt_speed_cm_s/100. if args.belt_speed_cm_s is not None
                     else args.fixed_belt_speed_m_s)
        if speed_m_s is not None:
            config = with_fixed_belt_speed(config, speed_m_s)
        if args.native_observation_shadow:
            from nero_core.native_observation.geometry import Parameters
            Parameters(**config.raw.get('native_observation', {}))
            if not is_fixed(config) or args.no_save:
                raise ValueError('原生观测对照需要固定带速并开启场次归档')
        from .hover_config import authorize as authorize_hover
        authorize_hover(config, entry=hover_descent_entry, speed=speed_m_s)
        if args.execute and not is_fixed(config):
            raise ValueError('真机抓取不使用视觉测速；请指定实际固定带速 --belt-speed-cm-s 或 --fixed-belt-speed-m-s')
        if freeze_source=='single_exposure_native_depth':
            if not is_fixed(config):
                raise ValueError('单帧冻结入口必须填写真实固定带速 --belt-speed-cm-s')
            if (config.raw['intercept_strategy']!='synchronized'
                    or (not native_band and config.raw['freeze_position_upstream_m'] is None)):
                raise ValueError('单帧冻结入口需要配置现有同步策略和固定冻结位置')
            config.raw['freeze_source']=freeze_source
        taught_source = ('cli' if args.taught_transfer is not None else
                         'configuration' if config.raw.get('taught_transfer_file') else None)
        taught_transfer = args.taught_transfer if taught_source == 'cli' else config.raw.get('taught_transfer_file')
        taught_return = args.taught_return if taught_source == 'cli' else config.raw.get('taught_return_file')
        taught_requested = taught_source is not None
        taught_ignored = taught_requested and not config.raw['taught_playback_enabled']
        if taught_requested and not taught_ignored:
            from .taught import TaughtSegment
            if config.raw['trajectory_profile'] != 'smooth':
                raise ValueError('示教轨迹需要平滑轨迹模式')
            config.raw['taught_segments'] = {
                'transfer': TaughtSegment.load(taught_transfer, config, 'transfer')}
            if config.raw['taught_return_strategy'] == 'recorded_then_wait':
                config.raw['taught_segments']['return'] = TaughtSegment.load(taught_return, config, 'return')
            if config.raw['taught_entry_search_enabled']:
                from .taught_entry import suffix
                for index in config.raw['taught_entry_indices']:
                    suffix(config.raw['taught_segments']['transfer'], index)
        elif config.raw['taught_entry_search_enabled']:
            raise ValueError('动态示教切入需要已配置的投放示教文件')
        if (config.raw['obb_grasp_orientation_reference'] == 'taught_transfer_start'
                and not config.raw.get('taught_segments', {}).get('transfer')):
            raise ValueError('示教起点抓姿排序需要已加载的搬运示教轨迹')
        if config.raw['carry_lookahead_enabled'] and not config.raw.get('taught_segments', {}).get('transfer'):
            raise ValueError('搬运预规划需要已加载投放目标')
        if config.raw['carry_lookahead_enabled'] and config.raw['trajectory_profile'] != 'smooth':
            raise ValueError('搬运预规划需要平滑抓取轨迹')
        if args.experimental_direct_drop_guard is not None:
            if not config.raw.get('taught_segments', {}).get('transfer') or not config.raw['prepared_taught_cycle_enabled']:
                raise ValueError('实验直接投放需要投放示教终点及预计算搬运')
            if config.raw['taught_entry_search_enabled']:
                raise ValueError('实验直接投放不能同时启用多示教切入点选择')
            from .direct_drop_guard import load_guard
            config.raw['experimental_direct_drop_guard'] = load_guard(
                args.experimental_direct_drop_guard, config, config.raw['taught_segments']['transfer'])
        if args.enable_hover_fallback:
            if executor_class is not None or hover_descent_entry:
                raise ValueError('原生自动补救不能叠加独立分段测试入口')
            from .native_hover_fallback import configure
            from .hover_fallback_test import HoverFallbackExecutor
            config, hover_fallback_config = configure(config)
            executor_class = partial(HoverFallbackExecutor, standard_config=config,
                fallback_config=hover_fallback_config)
            args.record_rgbd = True
        multiple_picks = args.continuous or args.count > 1
        buffered_freeze = (multiple_picks and hover_fallback_config is not None
            and config.raw['hover_fallback']['freeze_during_pick_enabled'])
        handoff_enabled = bool(buffered_freeze and config.raw.get('experimental_direct_drop_guard')
            and config.raw.get('prepared_taught_cycle_enabled') and config.raw.get('taught_playback_enabled'))
        if args.next_pick_group and not handoff_enabled:
            raise ValueError('起点对照需要连续或多轮抓取、搬运期间冻结、示教预计算和直接放料保护')
        group = (args.next_pick_group or 'C') if handoff_enabled else None
        policy = POLICY_BY_GROUP.get(group)
        direct_bin_pick = handoff_enabled and policy != WAIT_ONLY
        config.raw['next_pick_policy'] = policy
        pick_protocol = dict(next_pick_group=group, next_pick_policy=policy, direct_bin_pick=direct_bin_pick,
                             nominal_feed_interval_s=interval)
        from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
        calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    except (ValueError, OSError, KeyError, TypeError) as error:
        if args.execute:
            print(f'[停止] 配置检查未通过：{short(error)}', flush=True)
        else:
            print(json.dumps({'status': 'configuration_incomplete', 'error': str(error),
                              'hardware_connections': 0, 'commands_sent': 0}, ensure_ascii=False))
        return 2
    overlap = (multiple_picks and config.raw['continuous_lookahead_enabled']
        and config.raw['trajectory_profile'] == 'smooth'
        and config.raw.get('intercept_strategy') == 'synchronized')
    split_carry_return = bool(config.raw.get('prepared_taught_cycle_enabled')
        and config.raw.get('taught_playback_enabled', True) and config.raw.get('taught_segments'))
    along_belt_quantile_center = (freeze_source == 'single_exposure_native_depth'
        and config.raw['native_single_along_belt_quantile_center'])
    save_data = not args.no_save
    from .tracked_single_exposure import use_single_extraction
    geometry_pipeline = ('foreground_single_extraction' if native_band and use_single_extraction(config)
                         else 'seed_then_detector' if freeze_source=='single_exposure_native_depth'
                         else 'tracked')
    args.record_rgbd = save_data and args.record_rgbd
    if not args.execute:
        print(json.dumps({'status': 'configuration_valid', 'strategy': 'two_second',
            'native_observation_shadow': args.native_observation_shadow,
            'fast_stereo_enabled':config.raw.get('fast_stereo_enabled',False),
            'depth_separation_enabled':config.raw['depth_separation_enabled'],
            'object_relative_approach':config.raw.get('object_relative_approach'),
            'fast_cloud_delivery_mode':config.raw['fast_cloud_delivery_mode'],
            'fast_min_native_coverage_fraction':config.raw['fast_min_native_coverage_fraction'],
            'fast_max_center_shift_m':config.raw['fast_max_center_shift_m'],
            'rgb_cloud_completion_enabled':config.raw['rgb_cloud_completion_enabled'],
            'phase_timing': config.raw.get('phase_timing'),
            'delivery_repair': config.raw.get('delivery_repair', {'enabled':False}),
            'hover_fallback': config.raw.get('hover_fallback', {'enabled':False}),
            'freeze_source':freeze_source, 'native_geometry_pipeline':geometry_pipeline,
            'save_run_data':save_data, 'record_rgbd':args.record_rgbd,
            'preview_enabled':preview_enabled,
            'candidate_retry_policy':candidate_retry_policy,
            'planning_time_estimator':config.raw['planning_time_estimator'],
            'continuous':args.continuous, 'requested_count':None if args.continuous else args.count,
            'mode': 'check_only', 'hardware_connections': 0, 'commands_sent': 0,
            'hardware_speed_ready': is_fixed(config),
            'tcp_source': config.raw['tcp_source'],
            'taught_segments': {k:v.summary() for k,v in config.raw.get('taught_segments', {}).items()},
            'taught_playback_enabled': config.raw['taught_playback_enabled'],
            'obb_grasp_orientation_reference': config.raw['obb_grasp_orientation_reference'],
            'prepared_taught_cycle_enabled': config.raw['prepared_taught_cycle_enabled'],
            'split_carry_return_enabled': split_carry_return,
            'along_belt_quantile_center_enabled': along_belt_quantile_center,
            'carry_lookahead_enabled': config.raw['carry_lookahead_enabled'],
            'direct_drop_orientation_first': config.raw['direct_drop_orientation_first'],
            'direct_drop_continuous_carry': config.raw['direct_drop_continuous_carry'],
            'taught_return_strategy': config.raw['taught_return_strategy'],
            'taught_entry_search_enabled': config.raw['taught_entry_search_enabled'],
            'taught_entry_indices': config.raw['taught_entry_indices'],
            'experimental_direct_drop_guard': config.raw.get('experimental_direct_drop_guard'),
            'continuous_lookahead_enabled': config.raw['continuous_lookahead_enabled'],
            'lookahead_late_return_policy': config.raw['lookahead_late_return_policy'],
            'freeze_during_pick': overlap or buffered_freeze,
            **pick_protocol,
            'lookahead_return_margin_s': config.raw['lookahead_return_margin_s'],
            'taught_joint_jerk_limit_rad_s3': config.raw.get('taught_joint_jerk_limit_rad_s3'),
            'taught_transfer_jerk_limit_rad_s3': config.raw.get('taught_transfer_jerk_limit_rad_s3'),
            'taught_transfer_max_duration_s': config.raw.get('taught_transfer_max_duration_s'),
            'taught_segments_source': taught_source,
            'taught_arguments_ignored': taught_ignored,
            'velocity_policy': velocity_policy(config),
            'perception_detection_mode':config.raw['perception_detection_mode'],
            'perception_search_region':config.raw['perception_search_region'],
            'perception_detector_workspace_y_m':config.raw.get('perception_detector_workspace_y_m'),
            'perception_near_camera_footprint_margin_m':config.raw['perception_near_camera_footprint_margin_m'],
            'isolate_boundary_depth_gaps':config.raw['isolate_boundary_depth_gaps'],
            'split_mixed_belt_candidates':config.raw['split_mixed_belt_candidates'],
            'perception_plane_mode':config.raw['perception_plane_mode'],
            'tcp_calibration_id': config.raw['tcp_calibration_id'],
            'default_bin': config.default_bin,
            'drop_tcp_base': config.bins[config.default_bin].tolist(),
            'motion_limit_sources': config.raw['motion_limit_sources'],
            'trajectory_profile': config.raw['trajectory_profile'],
            'intercept_strategy': config.raw['intercept_strategy'],
            'synchronized_duration_s': config.raw['synchronized_duration_s'],
            'closure_policy': config.raw['closure_policy'],
            'grip_confirmation_dwell_s': config.raw['grip_confirmation_dwell_s'],
            'environment_collision_scope': config.raw['environment_collision_scope'],
            'workpiece_collision_margin_m': config.raw['workpiece_collision_margin_m'],
            'payload_belt_clearance_m': config.raw['payload_belt_clearance_m'],
            'grasp_workpiece_collision_check_enabled': config.raw['grasp_workpiece_collision_check_enabled'],
            'recovery_allow_workpiece_contact': config.raw['recovery_allow_workpiece_contact'],
            'return_wait_precision_enabled': config.raw['return_wait_precision_enabled'],
            'planning_knot_hz': config.raw.get('planning_knot_hz'),
            'planning_knot_hz_fallback': config.raw.get('planning_knot_hz_fallback'),
            'geometry_check_hz': config.raw['geometry_check_hz'],
            'prealign_orientation_s': config.raw.get('prealign_orientation_s', 0.),
            'brake_no_descent_rise_m': config.raw.get('brake_no_descent_rise_m', 0.),
            'rotate_first_s':config.raw.get('rotate_first_s',0.),
            'rotate_first_retime_enabled':config.raw.get('rotate_first_retime_enabled',False),
            'planning_reserve_mode': config.raw['planning_reserve_mode'],
            'planning_reserve_s': config.raw['planning_reserve_s'],
            'freeze_upstream_distance_m': config.raw['freeze_upstream_distance_m'],
            'tracked_freeze_range': config.raw['tracked_freeze_range'],
            'tracked_freeze_upstream_limit_m': config.raw.get('tracked_freeze_upstream_limit_m'),
            'tracked_freeze_publish_before_upstream_m': config.raw.get('tracked_freeze_publish_before_upstream_m'),
            'native_single_freeze_policy': config.raw['native_single_freeze_policy'] if native_band else None,
            'native_single_min_capture_upstream_m': config.raw.get('native_single_min_capture_upstream_m') if native_band else None,
            'tracked_local_geometry': config.raw['tracked_local_geometry'],
            'freeze_position_upstream_m': config.raw['freeze_position_upstream_m'],
            'freeze_position_tolerance_m': config.raw['freeze_position_tolerance_m'],
            'maximum_measured_speed_m_s':config.raw['maximum_measured_speed_m_s'],
            'synchronized_adaptive_freeze': config.raw['synchronized_adaptive_freeze'],
            'synchronized_max_freeze_upstream_distance_m': config.raw.get('synchronized_max_freeze_upstream_distance_m'),
            'approach_duration_candidates_s': config.raw.get('approach_duration_candidates_s'),
            'follow_duration_s': config.settings.full_plan_close_window_s,
            'brake_duration_s': config.raw.get('brake_duration_s'),
            'joint_velocity_limits_rad_s': config.velocity_limits.tolist(),
            'joint_acceleration_limits_rad_s2': config.acceleration_limits.tolist(),
            'joint_position_margins_rad': config.raw['joint_position_margins_rad'],
            'joint_receive_offset_s': config.raw['hardware']['joint_receive_offset_s'],
            'runtime_command_acceleration_check_enabled': config.raw['runtime_command_acceleration_check_enabled'],
            'runtime_command_velocity_check_enabled': config.raw['runtime_command_velocity_check_enabled'],
            'runtime_sweep_mode': config.raw['runtime_sweep_mode'],
            'runtime_sweep_hz': config.raw['runtime_sweep_hz'],
            'runtime_sweep_max_age_s': config.raw['runtime_sweep_max_age_s'],
            'dispatch_hz': config.settings.control_hz,
            'loaded_motion_profile': select_motion_profile(config, payload=True).diagnostics(),
            'startup_return_wait_policy': return_wait_policy(config),
            'source_hashes': provenance(config),
            'hardware_dependencies_available': {name: importlib.util.find_spec(name) is not None
                for name in ('can','pyorbbecsdk')}}, ensure_ascii=False))
        return 0
    if save_data and args.output.exists():
        print('[停止] 输出目录已存在，请使用新的 --output 目录', flush=True)
        return 2
    # Create a readable journal before model/SDK/device startup can fail.
    try:
        if save_data:
            args.output.mkdir(parents=True, exist_ok=False)
        else:
            from ..live_session import open_live_run
            args.output = open_live_run(lifetime, web_preview=args.web_preview)
            print('[记录] 本次抓取不保存场次数据', flush=True)
        console = ConsoleProgress(direction=config.direction)
        observer = console.observe
        if not save_data and preview_enabled:
            from ..control_panel.runtime_state import LivePanelStatus
            live_status = LivePanelStatus(args.output)
            def observer(row):
                console.observe(row)
                live_status.observe(row)
        journal = RunJournal(args.output if save_data else None, observer=observer)
    except (OSError, ValueError) as error:
        print(f'[停止] 无法准备运行目录：{short(error)}', flush=True)
        return 2
    cancelled = Event()
    begin = time.monotonic()
    reports, executors = [], []
    result = {'mode': 'hardware', 'strategy': 'two_second', 'freeze_source':freeze_source,
              'native_geometry_pipeline':geometry_pipeline,'source_hashes': provenance(config),
              'save_run_data':save_data,
              'phase_timing': config.raw.get('phase_timing'),
              'delivery_repair': config.raw.get('delivery_repair', {'enabled':False}),
              'phase_timing_hardware_test': phase_timing_hardware_test,
              'candidate_retry_policy':candidate_retry_policy,
              'planning_time_estimator':config.raw['planning_time_estimator'],
              'joint_receive_offset_s': config.raw['hardware']['joint_receive_offset_s'],
              'taught_segments': {k:v.summary() for k,v in config.raw.get('taught_segments', {}).items()},
              'taught_playback_enabled': config.raw['taught_playback_enabled'],
              'obb_grasp_orientation_reference': config.raw['obb_grasp_orientation_reference'],
              'prepared_taught_cycle_enabled': config.raw['prepared_taught_cycle_enabled'],
              'split_carry_return_enabled': split_carry_return,
              'along_belt_quantile_center_enabled': along_belt_quantile_center,
              'carry_lookahead_enabled': config.raw['carry_lookahead_enabled'],
              'direct_drop_orientation_first': config.raw['direct_drop_orientation_first'],
              'direct_drop_continuous_carry': config.raw['direct_drop_continuous_carry'],
              'taught_return_strategy': config.raw['taught_return_strategy'],
              'taught_entry_search_enabled': config.raw['taught_entry_search_enabled'],
              'taught_entry_indices': config.raw['taught_entry_indices'],
              'experimental_direct_drop_guard': config.raw.get('experimental_direct_drop_guard'),
              'continuous_lookahead_enabled': config.raw['continuous_lookahead_enabled'],
              'lookahead_late_return_policy': config.raw['lookahead_late_return_policy'],
              'freeze_during_pick': overlap or buffered_freeze,
              **pick_protocol,
              'taught_joint_jerk_limit_rad_s3': config.raw.get('taught_joint_jerk_limit_rad_s3'),
              'taught_transfer_jerk_limit_rad_s3': config.raw.get('taught_transfer_jerk_limit_rad_s3'),
              'taught_transfer_max_duration_s': config.raw.get('taught_transfer_max_duration_s'),
              'taught_segments_source': taught_source,
              'taught_arguments_ignored': taught_ignored,
              'prealign_orientation_s': config.raw.get('prealign_orientation_s', 0.),
              'brake_no_descent_rise_m': config.raw.get('brake_no_descent_rise_m', 0.),
              'rotate_first_s':config.raw.get('rotate_first_s',0.),
              'rotate_first_retime_enabled':config.raw.get('rotate_first_retime_enabled',False),
              'planning_reserve_mode':config.raw['planning_reserve_mode'],
              'tcp_source': config.raw['tcp_source'], 'tcp_calibration_id': config.raw['tcp_calibration_id'],
              'grasp_source': 'single_frame_visible_obb', 'network_called': False,
              'fast_stereo_enabled':config.raw.get('fast_stereo_enabled',False),
              'depth_separation_enabled':config.raw['depth_separation_enabled'],
              'object_relative_approach':config.raw.get('object_relative_approach'),
              'fast_cloud_delivery_mode':config.raw['fast_cloud_delivery_mode'],
              'fast_min_native_coverage_fraction':config.raw['fast_min_native_coverage_fraction'],
              'fast_max_center_shift_m':config.raw['fast_max_center_shift_m'],
              'rgb_cloud_completion_enabled':config.raw['rgb_cloud_completion_enabled'],
              'velocity_source': config.raw['velocity_source'],
              'velocity_policy': velocity_policy(config),
              'perception_detection_mode':config.raw['perception_detection_mode'],
              'perception_search_region':config.raw['perception_search_region'],
              'perception_detector_workspace_y_m':config.raw.get('perception_detector_workspace_y_m'),
              'perception_near_camera_footprint_margin_m':config.raw['perception_near_camera_footprint_margin_m'],
              'isolate_boundary_depth_gaps':config.raw['isolate_boundary_depth_gaps'],
              'split_mixed_belt_candidates':config.raw['split_mixed_belt_candidates'],
              'perception_plane_mode':config.raw['perception_plane_mode'],
              'environment_collision_scope': config.raw['environment_collision_scope'],
              'payload_belt_clearance_m': config.raw['payload_belt_clearance_m'],
              'grasp_workpiece_collision_check_enabled': config.raw['grasp_workpiece_collision_check_enabled'],
              'recovery_allow_workpiece_contact': config.raw['recovery_allow_workpiece_contact'],
              'return_wait_precision_enabled': config.raw['return_wait_precision_enabled'],
              'closure_policy': config.raw['closure_policy'],
              'grip_confirmation_dwell_s': config.raw['grip_confirmation_dwell_s'],
              'intercept_strategy': config.raw['intercept_strategy'],
              'synchronized_duration_s': config.raw['synchronized_duration_s'],
              'freeze_upstream_distance_m': config.raw['freeze_upstream_distance_m'],
              'tracked_freeze_range': config.raw['tracked_freeze_range'],
              'tracked_freeze_upstream_limit_m': config.raw.get('tracked_freeze_upstream_limit_m'),
              'tracked_freeze_publish_before_upstream_m': config.raw.get('tracked_freeze_publish_before_upstream_m'),
              'native_single_freeze_policy': config.raw['native_single_freeze_policy'] if native_band else None,
              'native_single_min_capture_upstream_m': config.raw.get('native_single_min_capture_upstream_m') if native_band else None,
              'tracked_local_geometry': config.raw['tracked_local_geometry'],
              'freeze_position_upstream_m': config.raw['freeze_position_upstream_m'],
              'freeze_position_tolerance_m': config.raw['freeze_position_tolerance_m'],
              'maximum_measured_speed_m_s':config.raw['maximum_measured_speed_m_s'],
              'synchronized_adaptive_freeze': config.raw['synchronized_adaptive_freeze'],
              'synchronized_max_freeze_upstream_distance_m': config.raw.get('synchronized_max_freeze_upstream_distance_m'),
              'completed_feedback_cycles': 0, 'physical_success_count': None,
              'automatic_enable': False, 'continuous':args.continuous,
              'requested_count':None if args.continuous else args.count, 'status': 'starting',
              'archived_attempt_count':0,'log_save_errors':[],
              'cycle_report_scope':'last_20_attempts' if args.continuous or not save_data else 'all_attempts',
              'received_signal': None}
    result.update(startup_return_wait_policy=return_wait_policy(config),
                  startup_return_wait={'status': 'not_checked'})
    result['hover_fallback'] = config.raw.get('hover_fallback', {'enabled':False})
    result.update(runtime_command_acceleration_check_enabled=config.raw['runtime_command_acceleration_check_enabled'],
                  runtime_command_velocity_check_enabled=config.raw['runtime_command_velocity_check_enabled'],
        runtime_sweep_mode=config.raw['runtime_sweep_mode'],
        runtime_sweep_hz=config.raw['runtime_sweep_hz'],
        runtime_sweep_max_age_s=config.raw['runtime_sweep_max_age_s'],
        trajectory_profile=config.raw['trajectory_profile'],
        approach_duration_candidates_s=config.raw.get('approach_duration_candidates_s'),
        follow_duration_s=config.settings.full_plan_close_window_s,
        brake_duration_s=config.raw.get('brake_duration_s'),
        motion_limit_sources=config.raw['motion_limit_sources'],
        joint_velocity_limits_rad_s=config.velocity_limits.tolist(),
        joint_acceleration_limits_rad_s2=config.acceleration_limits.tolist(),
        joint_position_margins_rad=config.raw['joint_position_margins_rad'],
        loaded_motion_profile=select_motion_profile(config, payload=True).diagnostics())
    runtime = {'phase': 'starting'}
    vision = executor = planner = visuals = viewer = recorder = lookahead = runtime_sweeps = carry_plans = None
    hover_plans = next_plans = fast_stereo = None
    experiment_archive = None
    attempt_archive = None

    def set_phase(phase, **context):
        nonlocal runtime
        runtime = {'phase': phase, **context}
        journal.emit('phase_started', **runtime)

    def progress_snapshot():
        active_executor = executors[-1] if executors else executor
        return {**runtime, 'execution_stage': None if active_executor is None else active_executor.stage,
            'startup_return_wait': result['startup_return_wait'],
            'planner': None if planner is None else getattr(active_executor, 'planner', planner).diagnostics(),
            'lookahead': None if lookahead is None else lookahead.diagnostics(),
            'vision': None if vision is None else vision.diagnostics()}

    def request_cancel(number, _):
        # A signal can interrupt a producer holding a log lock: do not log here.
        result['received_signal'] = signal.Signals(number).name
        cancelled.set()

    def save_attempt(active,index,report=None,*,background=True):
        from .attempt_archive import CompletedAttempt, write_attempt
        record = getattr(active,'completed_archive_record',None)
        if record is None:
            record = CompletedAttempt.capture(active,index,report)
            active.completed_archive_record = record
        if attempt_archive is not None and background:
            active.trace_archive_ticket = attempt_archive.submit(record)
            return
        directory = write_attempt(record,args.output)
        active.trace_saved=True
        result['archived_attempt_count']+=1
        journal.emit('cycle_trace_saved',cycle_index=index,directory=str(directory),
            command_diagnostics=active.command_history.diagnostics(),
            event_logging_errors=active.event_logging_errors)

    old_signals = {s: signal.signal(s, request_cancel) for s in (signal.SIGINT, signal.SIGTERM)}
    journal.emit('run_started', config=str(config.source), output=str(args.output) if save_data else None,
        brake_no_descent_rise_m=config.raw['brake_no_descent_rise_m'],
        brake_height_feedback_mode='observe_only' if config.raw['brake_no_descent_rise_m'] else None,
        native_observation_shadow=args.native_observation_shadow,
        object_relative_approach=config.raw.get('object_relative_approach'),
        fast_cloud_delivery_mode=config.raw['fast_cloud_delivery_mode'],
        fast_min_native_coverage_fraction=config.raw['fast_min_native_coverage_fraction'],
        fast_max_center_shift_m=config.raw['fast_max_center_shift_m'],
        follow_duration_s=config.settings.full_plan_close_window_s,
        panel_parameters={key: config.raw.get(key) for key in (
            'lookahead_late_return_policy', 'brake_duration_s', 'command_advance_s',
            'synchronized_join_candidates_s', 'synchronized_duration_s', 'geometry_check_hz',
            'depth_separation_enabled', 'rgb_cloud_completion_enabled', 'parallel_belt_scene_enabled', 'tracked_frozen_queue_capacity')},
        **pick_protocol,
        freeze_source=freeze_source, native_geometry_pipeline=geometry_pipeline,
        source_hashes=result['source_hashes'], requested_count=result['requested_count'],
        continuous=args.continuous,
        delivery_repair=result['delivery_repair'],
        environment_collision_scope=config.raw['environment_collision_scope'],
        payload_belt_clearance_m=config.raw['payload_belt_clearance_m'],
        grasp_workpiece_collision_check_enabled=config.raw['grasp_workpiece_collision_check_enabled'],
        isolate_boundary_depth_gaps=config.raw['isolate_boundary_depth_gaps'],
        recovery_allow_workpiece_contact=config.raw['recovery_allow_workpiece_contact'],
        closure_policy=config.raw['closure_policy'],
        grip_confirmation_dwell_s=config.raw['grip_confirmation_dwell_s'],
        taught_segments=result['taught_segments'],
        taught_playback_enabled=result['taught_playback_enabled'],
        prepared_taught_cycle_enabled=result['prepared_taught_cycle_enabled'],
        split_carry_return_enabled=split_carry_return,
        along_belt_quantile_center_enabled=along_belt_quantile_center,
        carry_lookahead_enabled=result['carry_lookahead_enabled'],
        direct_drop_orientation_first=result['direct_drop_orientation_first'],
        direct_drop_continuous_carry=result['direct_drop_continuous_carry'],
        taught_return_strategy=result['taught_return_strategy'],
        taught_entry_search_enabled=result['taught_entry_search_enabled'],
        taught_entry_indices=result['taught_entry_indices'],
        experimental_direct_drop_guard=result['experimental_direct_drop_guard'],
        taught_joint_jerk_limit_rad_s3=result['taught_joint_jerk_limit_rad_s3'],
        taught_transfer_jerk_limit_rad_s3=result['taught_transfer_jerk_limit_rad_s3'],
        taught_transfer_max_duration_s=result['taught_transfer_max_duration_s'],
        taught_segments_source=result['taught_segments_source'],
        taught_arguments_ignored=result['taught_arguments_ignored'],
        velocity_policy=velocity_policy(config),
        freeze_position_upstream_m=config.raw['freeze_position_upstream_m'],
        freeze_position_tolerance_m=config.raw['freeze_position_tolerance_m'],
        clock_domains={'monotonic_s': 'host_monotonic', 'wall_time_utc': 'host_wall_display_only',
            'source_device_s': 'camera_device_clock_no_direct_subtraction_from_host_time'},
        joint_velocity_limits_rad_s=config.velocity_limits,
        joint_acceleration_limits_rad_s2=config.acceleration_limits,
        runtime_command_acceleration_check_enabled=config.raw['runtime_command_acceleration_check_enabled'],
        runtime_command_velocity_check_enabled=config.raw['runtime_command_velocity_check_enabled'])
    monitor = ProgressMonitor(journal, progress_snapshot).start()
    try:
        if args.continuous and save_data:
            from .attempt_archive import AttemptArchive
            attempt_archive = AttemptArchive(args.output,journal.emit)
        # This path is selected only by a human's explicit --execute command.
        set_phase('loading_dependencies')
        from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
        from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
        from .planning import calibrated_model, MeasuredPlanner
        if freeze_source=='single_exposure_native_depth' and not native_band:
            from .single_exposure_vision import SingleExposureVision as LiveVision
        else:
            from .vision import LiveVision
        from .execution import TwoSecondExecutor
        from .visual_archive import VisualArchive
        from .experiment_archive import ExperimentArchive
        if save_data:
            experiment_archive = ExperimentArchive(args.output, journal.emit)
            experiment_archive.save_configuration(config)
            journal.emit('experiment_recording_enabled', directory=str(experiment_archive.directory),
                frozen_targets=True, planning_inputs=True, successful_and_failed_plans=True,
                physical_success_inferred=False)
        if hover_fallback_config is not None:
            # Include both effective policies; config.source still refers to
            # the user's unchanged native site YAML.
            fallback_record = dict(policy=result['hover_fallback'], raw={k:v
                for k,v in hover_fallback_config.raw.items() if k != 'taught_segments'})
            if save_data:
                (args.output/'hover_fallback.json').write_text(
                    json.dumps(json_value(fallback_record), ensure_ascii=False, indent=2)+'\n')
            journal.emit('native_hover_fallback_enabled', **result['hover_fallback'],
                freeze_during_pick=buffered_freeze, **pick_protocol)
        if save_data or preview_enabled:
            visuals = VisualArchive(args.output, journal.emit, live=preview_enabled,direction=config.direction,
                path_model_factory=(lambda: calibrated_model(config)) if preview_enabled else None,
                live_only=not save_data)
        if args.record_rgbd:
            from .rgbd_archive import RGBDArchive
            recorder = RGBDArchive(args.output)
        if args.preview:
            # The viewer follows atomic files only; it never opens SDK/CAN.
            with (args.output/'preview/viewer.log' if save_data else Path(os.devnull)).open('w') as viewer_log:
                viewer = subprocess.Popen([sys.executable, '-m', 'nero_core.two_second.camera_view',
                    '--config', str(config.source), '--follow-run', str(args.output.resolve()),
                    '--output', str((args.output/'preview/viewer').resolve())],
                    stdout=viewer_log, stderr=subprocess.STDOUT, start_new_session=True,
                    env=preview_environment())
            result['preview_process_pid'] = viewer.pid
            journal.emit('preview_requested',viewer_pid=viewer.pid)
        # Private mirrors are prepared once, before either device is opened.
        set_phase('model_startup')
        planner = MeasuredPlanner(config)
        if hover_fallback_config is not None or config.raw.get('hover_descent', {}).get('enabled', False):
            from .hover_planning import HoverService
            hover_plans = HoverService(hover_fallback_config or config)
            planner.hover_service = hover_plans
            journal.emit('hover_planning_worker_ready', **hover_plans.worker)
        spare_planner = MeasuredPlanner(config) if overlap else None
        if experiment_archive is not None:
            experiment_archive.bind_planner(planner)
            if spare_planner is not None:
                experiment_archive.bind_planner(spare_planner)
        result['continuous_lookahead_active'] = bool(overlap)
        if config.raw.get('taught_segments'):
            from .taught import prepare_taught_timing
            timings = {}
            for kind, segment in config.raw['taught_segments'].items():
                prepared, _ = prepare_taught_timing(config, segment, planner.model,
                    payload=kind == 'transfer', cancelled=cancelled.is_set)
                timings[kind] = {'duration_s': prepared.duration, **prepared.diagnostics}
            if config.raw['taught_entry_search_enabled']:
                from .taught_entry import prepare_entries
                config.raw['_prepared_taught_entries'] = prepare_entries(
                    config, config.raw['taught_segments']['transfer'], planner.model,
                    cancelled=cancelled.is_set)
                timings['entry_candidates'] = [item[0] for item in config.raw['_prepared_taught_entries']]
                timings['entry_candidate_rejections'] = config.raw['_taught_entry_prewarm_rejections']
            journal.emit('taught_timing_prepared', segments=timings, hardware_connections=0)
        if preview_enabled:
            def add_visualization(candidate):
                original_intercept = candidate.intercept
                def offer_plan(plan):
                    try:
                        visuals.offer_plan(plan, width_m=config.settings.open_width_m)
                    except Exception as error:
                        journal.emit('visualization_error', error_type=type(error).__name__, error=str(error))
                def intercept_with_visualization(*call_args, **call_kwargs):
                    plan = original_intercept(*call_args, **call_kwargs)
                    offer_plan(plan)
                    return plan
                candidate.intercept = intercept_with_visualization
                candidate.offer_intercept_plan = offer_plan
            add_visualization(planner)
            if spare_planner is not None:
                add_visualization(spare_planner)
        if visuals is not None:
            planner.collision_sink = visuals.submit
            if spare_planner is not None:
                spare_planner.collision_sink = visuals.submit
        detector_model = calibrated_model(config)
        if config.raw['runtime_sweep_mode'] == 'process_periodic':
            from .runtime_sweep import SweepService
            # Spawn/warm the geometry worker before opening either device.
            runtime_sweeps = SweepService(config)
            result['runtime_sweep_worker'] = runtime_sweeps.worker
            journal.emit('runtime_sweep_ready', **runtime_sweeps.worker,
                check_hz=config.raw['runtime_sweep_hz'], dispatch_hz=config.settings.control_hz)
        if config.raw['carry_lookahead_enabled'] or split_carry_return:
            from .carry_lookahead import CarryLookahead
            carry_plans = CarryLookahead(config)
            result['carry_lookahead_worker'] = carry_plans.worker
            journal.emit('carry_lookahead_ready', **carry_plans.worker)
        if handoff_enabled:
            from .next_pick import NextPickPlans
            next_plans = NextPickPlans(config, emit=journal.emit)
        result['model_startup_wall_s'] = time.monotonic()-begin
        with ExitStack() as resources:
            if config.raw.get('fast_stereo_enabled', False):
                from .fast_perception import FastTargetPerception
                set_phase('fast_stereo_startup')
                fast_stereo = FastTargetPerception(config, args.output/'fast_stereo', retain_artifacts=save_data)
                fast_stereo.event_sink = journal.emit
                resources.callback(fast_stereo.close)
                fast_stereo.start()  # Complete GPU warm-up before any CAN connection.
                result['network_called'] = True
                result['network_model'] = 'Fast-FoundationStereo'
                result['grasp_source'] = 'single_frame_visible_obb_from_fast_stereo'
                journal.emit('fast_stereo_ready', **fast_stereo.backend.health())
            set_phase('can_startup', can_channel=config.raw['hardware']['can_channel'])
            hw = Hardware(config.raw['hardware']['can_channel'], config.raw['hardware']['firmware'], True,
                          config.raw['hardware']['joint_receive_offset_s'])
            resources.callback(hw.close)
            hw.configure_joint_command_mode(config.raw['hardware']['joint_command_mode'])
            hw.start()
            initial_state = hw.snapshot()  # original fault/enable/driver checks; never auto-enable
            set_phase('startup_readiness')
            journal.emit('startup_feedback', actual_joints_rad=initial_state['joints'],
                standby_joints_rad=config.standby, width_m=initial_state['width_m'],
                force_n=initial_state['force_n'])
            prepare_start_state(config, hw, initial_state, model=planner.model,
                cancelled=cancelled, emit=journal.emit, report=result['startup_return_wait'])
            result['startup_return_wait_wall_s'] = result['startup_return_wait']['total_wall_s']
            if cancelled.is_set():
                raise MoveCancelled('启动准备后已取消，不再开启相机或抓取')
            journal.emit('startup_readiness_passed')
            set_phase('camera_startup')
            from .fast_perception import camera_settings
            camera_config = camera_settings(config)
            camera_config.update(enable_color=True, require_color=True, timestamp_source='device',
                serial_number=calibration['serial_number'], calibration_id=calibration['calibration_id'],
                camera_to_world_cv=calibration['depth_to_base_cv'])
            camera = OrbbecCameraBackend(camera_config)
            resources.callback(camera.close)
            camera.start()  # exactly one SDK capture owner
            vision = LiveVision(camera, calibration, config, detector_model, hw.snapshot_nowait)
            vision.fast_stereo = fast_stereo
            vision.event_sink=journal.emit
            if experiment_archive is not None:
                vision.geometry_rejection_sink=experiment_archive.offer_geometry_rejection
                if freeze_source=='single_exposure_native_depth' and not native_band:
                    experiment_archive.bind_single_exposure(vision)
            if overlap or buffered_freeze or getattr(vision, 'single_frame_backup', None) is not None:
                vision.capture_state_reader = hw.snapshot_for_capture
                vision.capture_states_reader = getattr(hw, 'snapshot_for_captures', None)
            if buffered_freeze:
                vision.enable_lookahead()
                if next_plans is not None:
                    vision.next_pick_sink = next_plans.offer
            shadow = None
            if args.native_observation_shadow:
                from nero_core.native_observation.shadow import NativeShadow
                shadow = NativeShadow(config, calibration, args.output, hw.snapshot_for_capture,
                                      calibrated_model(config))
                shadow.start()
                resources.callback(shadow.close)
            if preview_enabled or recorder is not None or shadow is not None:
                def offer_frame(frame, d2c):
                    if shadow is not None:
                        shadow.offer(frame, d2c)
                    if preview_enabled:
                        visuals.offer_frame(frame, d2c)
                    if recorder is not None:
                        recorder.offer_frame(frame, d2c)
                vision.frame_sink = offer_frame
            if preview_enabled:
                vision.speed_sink = visuals.offer_speed
            resources.callback(vision.close)
            vision.start()
            if overlap:
                from .return_lookahead import ReturnLookahead
                def offer_experiment_target(target):
                    if experiment_archive is not None:
                        experiment_archive.offer_target(target)
                    if preview_enabled or args.record_rgbd:
                        visuals.offer_target(target)
                lookahead = ReturnLookahead(config, vision, spare_planner, cancelled=cancelled,
                    emit=journal.emit, offer_target=offer_experiment_target)
                resources.callback(lookahead.close)
                lookahead.start()
            result['device_startup_wall_s'] = time.monotonic()-begin-result['model_startup_wall_s']
            from .cycle_loop import run_pick_cycles
            def cycle_phase(phase, **context):
                nonlocal executor
                executor = executors[-1] if executors else None
                if attempt_archive is not None:
                    attempt_archive.check_health()
                set_phase(phase, **context)
            run_pick_cycles(config, planner, hw, vision, count=result['requested_count'], cancelled=cancelled,
                result=result, reports=reports, executors=executors, set_phase=cycle_phase,
                journal=journal, executor_factory=partial(executor_class or TwoSecondExecutor,
                    sweep_service=runtime_sweeps, carry_service=carry_plans),
                finish_attempt=save_attempt if args.continuous and save_data else None, save_data=save_data,
                offer_target=visuals.offer_target if preview_enabled or args.record_rgbd else None,
                lookahead=lookahead, buffered_freeze=buffered_freeze, direct_bin_pick=direct_bin_pick,
                next_plans=next_plans)
            executor = executors[-1] if executors else None
    except BaseException as error:
        executor = executors[-1] if executors else executor
        result.update(status='cancelled' if isinstance(error, MoveCancelled) or result['received_signal'] else 'stopped',
                      error=str(error), error_type=type(error).__name__,
                      details=getattr(error, 'details', {}))
        journal.emit('run_failed', **runtime, execution_stage=None if executor is None else executor.stage,
            error=str(error), error_type=type(error).__name__, details=getattr(error, 'details', {}))
        cancelled.set()
    finally:
        if next_plans is not None:
            next_plans.close()
        if hover_plans is not None:
            hover_plans.close()
        if carry_plans is not None:
            carry_plans.close()
        if runtime_sweeps is not None:
            runtime_sweeps.close()
        monitor.close()
        if getattr(fast_stereo,'cloud_delivery',None) is not None:
            result['fast_cloud_delivery'] = dict(fast_stereo.cloud_delivery.summary(),
                reconstruction_requests=fast_stereo.requests, grasp_geometry_ready=fast_stereo.completed)
            result['log_save_errors'].extend(result['fast_cloud_delivery']['archive_errors'])
        for number, handler in old_signals.items():
            signal.signal(number, handler)
        result['last_runtime'] = progress_snapshot()
        result['vision_diagnostics'] = result['last_runtime']['vision']
        if recorder is not None:
            result['rgbd_recording'] = recorder.close()
        if experiment_archive is not None:
            result['experiment_recording'] = experiment_archive.close()
            if not result['experiment_recording']['complete']:
                result['log_save_errors'].extend(result['experiment_recording']['errors'])
                journal.emit('experiment_recording_incomplete', **result['experiment_recording'])
        if visuals is not None:
            visual_close_begin = time.monotonic()
            result['visualization'] = visuals.close()
            result['visualization_shutdown_wall_s'] = time.monotonic()-visual_close_begin
        if planner is not None:
            result['collision_diagnostic_errors'] = list(getattr(planner,'collision_diagnostic_errors',[]))
        if viewer is not None:
            if args.continuous and viewer.poll() is None:
                result['preview_closed_by_parent']=True
                try:
                    viewer.terminate()
                    try:viewer.wait(timeout=8.)
                    except subprocess.TimeoutExpired:
                        viewer.kill()
                        viewer.wait(timeout=2.)
                except Exception as error:
                    result['preview_cleanup_error']=str(error)
            result['preview_process_exit_code'] = viewer.poll()
        # Save traces after all senders, camera reader and CAN ownership ended.
        save_begin = time.monotonic()
        if attempt_archive is not None:
            archive_result = attempt_archive.close()
            result['attempt_archive'] = archive_result
            result['archived_attempt_count'] += archive_result['saved']
            result['log_save_errors'].extend(archive_result['errors'])
            result['log_save_errors'].extend(archive_result['notification_errors'])
        for i, executor in enumerate(executors if save_data else ()):
            index=getattr(executor,'cycle_index',i+1)
            if getattr(executor,'trace_archive_ticket',None) is not None:continue
            if getattr(executor,'trace_saved',False):continue
            try:
                save_attempt(executor,index,background=False)
            except Exception as error:
                result['log_save_errors'].append(f'cycle_{index:03d}: {type(error).__name__}: {error}')
        journal.emit('run_finished', status=result['status'], received_signal=result['received_signal'],
            completed_feedback_cycles=result['completed_feedback_cycles'], **result['last_runtime'])
        result['journal'] = journal.close()
        result.update(cycles=reports,
                      runtime_command_diagnostics=[x.command_history.diagnostics() for x in executors],
                      total_wall_before_log_save_s=save_begin-begin,
                      log_save_wall_s=time.monotonic()-save_begin,
                      execution_event_logging_errors=[e for x in executors for e in x.event_logging_errors])
        if save_data:
            temporary = args.output/'summary.json.tmp'
            try:
                temporary.write_text(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
                temporary.replace(args.output/'summary.json')
            except OSError as error:
                result['log_save_errors'].append(f'summary: {type(error).__name__}: {error}')
        console.finish(result, args.output if save_data else None)
    return 0 if result['status'] in ('requested_feedback_cycles_completed', 'attempt_completed_grip_unconfirmed',
                                   'hover_test_attempt_completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Physical integration entry. Default only validates files, opening no devices."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import importlib.util
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
from .velocity_source import with_fixed_belt_speed, policy as velocity_policy


def provenance(config):
    files = [config.source, Path(config.raw['tcp_calibration_file']),
             Path(config.raw['camera_calibration_file'])]
    files += sorted(Path(__file__).parent.glob('*.py'))
    files.append(Path(__file__).parent.parent/'move_to_wait.py')
    root=Path(__file__).resolve().parents[2]
    files += [root/'mujoco_conveyor_sorting/runtime'/name for name in (
        'nero_pick_hardware.py','single_pick.py','frame_products.py','scene_region_filter.py',
        'nero7_belt_velocity.py','nero7_fused_detection.py','nero7_tracking_roi.py','nero7_boundary_depth.py','workpiece_admission.py',
        'pointcloud_speed_tracker.py')]
    files += [root/'mujoco_conveyor_sorting/models/robot.xml',
              root/'mujoco_conveyor_sorting/configs/robot.yaml',
              root/'mujoco_conveyor_sorting/perception/rgbd_vision_pipeline.py']
    vendor=root/'vendor/nero7_speed_reference'
    files.append(vendor/'source_manifest.json')
    files += [vendor/'nero7_perception'/name for name in (
        'block_tracker.py','kalman_motion.py','detection.py','base_plane.py','geometry.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def main(argv=None,*,freeze_source='tracked'):
    if freeze_source not in ('tracked','single_exposure_native_depth'):
        raise ValueError('unknown freeze acquisition source')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/two_second_hardware'))
    parser.add_argument('--execute', action='store_true', help='连接实际相机/CAN并执行；不会使能设备')
    parser.add_argument('--check-config', action='store_true', help='只检查标定和配置，不导入厂家SDK')
    cycles=parser.add_mutually_exclusive_group()
    cycles.add_argument('--count', type=int, default=1, help='完成指定数量的反馈周期后退出')
    cycles.add_argument('--continuous', action='store_true', help='连续抓取；每轮回等待位，Ctrl+C退出')
    belt_speed = parser.add_mutually_exclusive_group()
    belt_speed.add_argument('--belt-speed-cm-s', type=float,
        help='本次运行的固定预测带速，单位cm/s，例如4.53；省略时沿用配置')
    belt_speed.add_argument('--fixed-belt-speed-m-s', type=float,
        help='本次运行覆盖配置使用用户设定带速m/s；省略时使用配置指定的速度来源')
    parser.add_argument('--preview', action='store_true', help='共用当前相机来源，打开只读画面及碰撞预测窗口')
    parser.add_argument('--record-rgbd', action='store_true', help='异步记录同源RGB-D，供事后重建场景点云')
    parser.add_argument('--taught-transfer', type=Path, help='覆盖配置中的载物投放示教文件，须与 --taught-return 成对提供')
    parser.add_argument('--taught-return', type=Path, help='覆盖配置中的空爪返回示教文件，须与 --taught-transfer 成对提供')
    args = parser.parse_args(argv)
    if args.check_config and args.execute:
        parser.error('--check-config 与 --execute 互斥')
    if args.count < 1:
        parser.error('--count 必须大于零')
    if (args.taught_transfer is None) != (args.taught_return is None):
        parser.error('使用示教时必须同时指定 --taught-transfer 和 --taught-return')
    # All configuration checks happen before importing a camera or CAN SDK.
    try:
        config = CellConfig.load(args.config)
        speed_m_s = (args.belt_speed_cm_s/100. if args.belt_speed_cm_s is not None
                     else args.fixed_belt_speed_m_s)
        if speed_m_s is not None:
            config = with_fixed_belt_speed(config, speed_m_s)
        if freeze_source=='single_exposure_native_depth':
            from .velocity_source import is_fixed
            if not is_fixed(config):
                raise ValueError('单帧冻结入口必须填写真实固定带速 --belt-speed-cm-s')
            if (config.raw['intercept_strategy']!='synchronized'
                    or config.raw['freeze_position_upstream_m'] is None):
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
                'transfer': TaughtSegment.load(taught_transfer, config, 'transfer'),
                'return': TaughtSegment.load(taught_return, config, 'return')}
        if (config.raw['obb_grasp_orientation_reference'] == 'taught_transfer_start'
                and not config.raw.get('taught_segments', {}).get('transfer')):
            raise ValueError('示教起点抓姿排序需要已加载的搬运示教轨迹')
        from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
        calibration = load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    except (ValueError, OSError, KeyError, TypeError) as error:
        if args.execute:
            print(f'[停止] 配置检查未通过：{short(error)}', flush=True)
        else:
            print(json.dumps({'status': 'configuration_incomplete', 'error': str(error),
                              'hardware_connections': 0, 'commands_sent': 0}, ensure_ascii=False))
        return 2
    if not args.execute:
        print(json.dumps({'status': 'configuration_valid', 'strategy': 'two_second',
            'freeze_source':freeze_source,
            'continuous':args.continuous, 'requested_count':None if args.continuous else args.count,
            'mode': 'check_only', 'hardware_connections': 0, 'commands_sent': 0,
            'tcp_source': config.raw['tcp_source'],
            'taught_segments': {k:v.summary() for k,v in config.raw.get('taught_segments', {}).items()},
            'taught_playback_enabled': config.raw['taught_playback_enabled'],
            'obb_grasp_orientation_reference': config.raw['obb_grasp_orientation_reference'],
            'prepared_taught_cycle_enabled': config.raw['prepared_taught_cycle_enabled'],
            'taught_joint_jerk_limit_rad_s3': config.raw.get('taught_joint_jerk_limit_rad_s3'),
            'taught_transfer_jerk_limit_rad_s3': config.raw.get('taught_transfer_jerk_limit_rad_s3'),
            'taught_transfer_max_duration_s': config.raw.get('taught_transfer_max_duration_s'),
            'taught_segments_source': taught_source,
            'taught_arguments_ignored': taught_ignored,
            'velocity_policy': velocity_policy(config),
            'perception_detection_mode':config.raw['perception_detection_mode'],
            'isolate_boundary_depth_gaps':config.raw['isolate_boundary_depth_gaps'],
            'perception_plane_mode':config.raw['perception_plane_mode'],
            'tcp_calibration_id': config.raw['tcp_calibration_id'],
            'default_bin': config.default_bin,
            'drop_tcp_base': config.bins[config.default_bin].tolist(),
            'motion_limit_sources': config.raw['motion_limit_sources'],
            'trajectory_profile': config.raw['trajectory_profile'],
            'intercept_strategy': config.raw['intercept_strategy'],
            'synchronized_duration_s': config.raw['synchronized_duration_s'],
            'closure_policy': config.raw['closure_policy'],
            'environment_collision_scope': config.raw['environment_collision_scope'],
            'workpiece_collision_margin_m': config.raw['workpiece_collision_margin_m'],
            'payload_belt_clearance_m': config.raw['payload_belt_clearance_m'],
            'grasp_workpiece_collision_check_enabled': config.raw['grasp_workpiece_collision_check_enabled'],
            'recovery_allow_workpiece_contact': config.raw['recovery_allow_workpiece_contact'],
            'planning_knot_hz': config.raw.get('planning_knot_hz'),
            'planning_knot_hz_fallback': config.raw.get('planning_knot_hz_fallback'),
            'geometry_check_hz': config.raw['geometry_check_hz'],
            'prealign_orientation_s': config.raw.get('prealign_orientation_s', 0.),
            'rotate_first_s':config.raw.get('rotate_first_s',0.),
            'rotate_first_retime_enabled':config.raw.get('rotate_first_retime_enabled',False),
            'planning_reserve_mode': config.raw['planning_reserve_mode'],
            'planning_reserve_s': config.raw['planning_reserve_s'],
            'freeze_upstream_distance_m': config.raw['freeze_upstream_distance_m'],
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
            'joint_receive_offset_s': config.raw['hardware']['joint_receive_offset_s'],
            'runtime_command_acceleration_check_enabled': config.raw['runtime_command_acceleration_check_enabled'],
            'loaded_motion_profile': select_motion_profile(config, payload=True).diagnostics(),
            'startup_return_wait_policy': return_wait_policy(config),
            'source_hashes': provenance(config),
            'hardware_dependencies_available': {name: importlib.util.find_spec(name) is not None
                for name in ('can','pyorbbecsdk')}}, ensure_ascii=False))
        return 0
    if args.output.exists():
        print('[停止] 输出目录已存在，请使用新的 --output 目录', flush=True)
        return 2
    # Create a readable journal before model/SDK/device startup can fail.
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        console = ConsoleProgress(direction=config.direction)
        journal = RunJournal(args.output, observer=console.observe)
    except OSError as error:
        print(f'[停止] 无法创建运行日志：{short(error)}', flush=True)
        return 2
    cancelled = Event()
    begin = time.monotonic()
    reports, executors = [], []
    result = {'mode': 'hardware', 'strategy': 'two_second', 'freeze_source':freeze_source,'source_hashes': provenance(config),
              'joint_receive_offset_s': config.raw['hardware']['joint_receive_offset_s'],
              'taught_segments': {k:v.summary() for k,v in config.raw.get('taught_segments', {}).items()},
              'taught_playback_enabled': config.raw['taught_playback_enabled'],
              'obb_grasp_orientation_reference': config.raw['obb_grasp_orientation_reference'],
              'prepared_taught_cycle_enabled': config.raw['prepared_taught_cycle_enabled'],
              'taught_joint_jerk_limit_rad_s3': config.raw.get('taught_joint_jerk_limit_rad_s3'),
              'taught_transfer_jerk_limit_rad_s3': config.raw.get('taught_transfer_jerk_limit_rad_s3'),
              'taught_transfer_max_duration_s': config.raw.get('taught_transfer_max_duration_s'),
              'taught_segments_source': taught_source,
              'taught_arguments_ignored': taught_ignored,
              'prealign_orientation_s': config.raw.get('prealign_orientation_s', 0.),
              'rotate_first_s':config.raw.get('rotate_first_s',0.),
              'rotate_first_retime_enabled':config.raw.get('rotate_first_retime_enabled',False),
              'planning_reserve_mode':config.raw['planning_reserve_mode'],
              'tcp_source': config.raw['tcp_source'], 'tcp_calibration_id': config.raw['tcp_calibration_id'],
              'grasp_source': 'single_frame_visible_obb', 'network_called': False,
              'velocity_source': config.raw['velocity_source'],
              'velocity_policy': velocity_policy(config),
              'perception_detection_mode':config.raw['perception_detection_mode'],
              'isolate_boundary_depth_gaps':config.raw['isolate_boundary_depth_gaps'],
              'perception_plane_mode':config.raw['perception_plane_mode'],
              'environment_collision_scope': config.raw['environment_collision_scope'],
              'payload_belt_clearance_m': config.raw['payload_belt_clearance_m'],
              'grasp_workpiece_collision_check_enabled': config.raw['grasp_workpiece_collision_check_enabled'],
              'recovery_allow_workpiece_contact': config.raw['recovery_allow_workpiece_contact'],
              'closure_policy': config.raw['closure_policy'],
              'intercept_strategy': config.raw['intercept_strategy'],
              'synchronized_duration_s': config.raw['synchronized_duration_s'],
              'freeze_upstream_distance_m': config.raw['freeze_upstream_distance_m'],
              'freeze_position_upstream_m': config.raw['freeze_position_upstream_m'],
              'freeze_position_tolerance_m': config.raw['freeze_position_tolerance_m'],
              'maximum_measured_speed_m_s':config.raw['maximum_measured_speed_m_s'],
              'synchronized_adaptive_freeze': config.raw['synchronized_adaptive_freeze'],
              'synchronized_max_freeze_upstream_distance_m': config.raw.get('synchronized_max_freeze_upstream_distance_m'),
              'completed_feedback_cycles': 0, 'physical_success_count': None,
              'automatic_enable': False, 'continuous':args.continuous,
              'requested_count':None if args.continuous else args.count, 'status': 'starting',
              'archived_attempt_count':0,'log_save_errors':[],
              'cycle_report_scope':'last_20_attempts' if args.continuous else 'all_attempts',
              'received_signal': None}
    result.update(startup_return_wait_policy=return_wait_policy(config),
                  startup_return_wait={'status': 'not_checked'})
    result.update(runtime_command_acceleration_check_enabled=config.raw['runtime_command_acceleration_check_enabled'],
        trajectory_profile=config.raw['trajectory_profile'],
        approach_duration_candidates_s=config.raw.get('approach_duration_candidates_s'),
        follow_duration_s=config.settings.full_plan_close_window_s,
        brake_duration_s=config.raw.get('brake_duration_s'),
        motion_limit_sources=config.raw['motion_limit_sources'],
        joint_velocity_limits_rad_s=config.velocity_limits.tolist(),
        joint_acceleration_limits_rad_s2=config.acceleration_limits.tolist(),
        loaded_motion_profile=select_motion_profile(config, payload=True).diagnostics())
    runtime = {'phase': 'starting'}
    vision = executor = planner = visuals = viewer = recorder = None

    def set_phase(phase, **context):
        nonlocal runtime
        runtime = {'phase': phase, **context}
        journal.emit('phase_started', **runtime)

    def progress_snapshot():
        active_executor = executors[-1] if executors else executor
        return {**runtime, 'execution_stage': None if active_executor is None else active_executor.stage,
            'startup_return_wait': result['startup_return_wait'],
            'planner': None if planner is None else planner.diagnostics(),
            'vision': None if vision is None else vision.diagnostics()}

    def request_cancel(number, _):
        # A signal can interrupt a producer holding a log lock: do not log here.
        result['received_signal'] = signal.Signals(number).name
        cancelled.set()

    def save_attempt(active,index,report=None):
        directory=args.output/f'cycle_{index:03d}'
        active.trace.write(directory)
        (directory/'events.json').write_text(json.dumps(json_value(active.events),ensure_ascii=False,indent=2))
        if report is not None:
            (directory/'report.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2))
        active.trace_saved=True
        result['archived_attempt_count']+=1
        journal.emit('cycle_trace_saved',cycle_index=index,directory=str(directory),
            command_diagnostics=active.command_history.diagnostics(),
            event_logging_errors=active.event_logging_errors)

    old_signals = {s: signal.signal(s, request_cancel) for s in (signal.SIGINT, signal.SIGTERM)}
    journal.emit('run_started', config=str(config.source), output=str(args.output),
        freeze_source=freeze_source,
        source_hashes=result['source_hashes'], requested_count=result['requested_count'],
        continuous=args.continuous,
        environment_collision_scope=config.raw['environment_collision_scope'],
        payload_belt_clearance_m=config.raw['payload_belt_clearance_m'],
        grasp_workpiece_collision_check_enabled=config.raw['grasp_workpiece_collision_check_enabled'],
        isolate_boundary_depth_gaps=config.raw['isolate_boundary_depth_gaps'],
        recovery_allow_workpiece_contact=config.raw['recovery_allow_workpiece_contact'],
        closure_policy=config.raw['closure_policy'],
        taught_segments=result['taught_segments'],
        taught_playback_enabled=result['taught_playback_enabled'],
        prepared_taught_cycle_enabled=result['prepared_taught_cycle_enabled'],
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
        runtime_command_acceleration_check_enabled=config.raw['runtime_command_acceleration_check_enabled'])
    monitor = ProgressMonitor(journal, progress_snapshot).start()
    try:
        # This path is selected only by a human's explicit --execute command.
        set_phase('loading_dependencies')
        from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
        from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
        from .planning import calibrated_model, MeasuredPlanner
        if freeze_source=='single_exposure_native_depth':
            from .single_exposure_vision import SingleExposureVision as LiveVision
        else:
            from .vision import LiveVision
        from .execution import TwoSecondExecutor
        from .visual_archive import VisualArchive
        visuals = VisualArchive(args.output, journal.emit, live=args.preview,direction=config.direction,
            path_model_factory=(lambda: calibrated_model(config)) if args.preview else None)
        if args.record_rgbd:
            from .rgbd_archive import RGBDArchive
            recorder = RGBDArchive(args.output)
        if args.preview:
            # The viewer follows atomic files only; it never opens SDK/CAN.
            with (args.output/'preview/viewer.log').open('w') as viewer_log:
                viewer = subprocess.Popen([sys.executable, '-m', 'nero_core.two_second.camera_view',
                    '--config', str(config.source), '--follow-run', str(args.output.resolve()),
                    '--output', str((args.output/'preview/viewer').resolve())],
                    stdout=viewer_log, stderr=subprocess.STDOUT, start_new_session=True)
            result['preview_process_pid'] = viewer.pid
            journal.emit('preview_requested',viewer_pid=viewer.pid)
        # Private mirrors are prepared once, before either device is opened.
        set_phase('model_startup')
        planner = MeasuredPlanner(config)
        if config.raw.get('taught_segments'):
            from .taught import prepare_taught_timing
            timings = {}
            for kind, segment in config.raw['taught_segments'].items():
                prepared, _ = prepare_taught_timing(config, segment, planner.model,
                    payload=kind == 'transfer', cancelled=cancelled.is_set)
                timings[kind] = {'duration_s': prepared.duration, **prepared.diagnostics}
            journal.emit('taught_timing_prepared', segments=timings, hardware_connections=0)
        if args.preview:
            original_intercept = planner.intercept
            def intercept_with_visualization(*call_args, **call_kwargs):
                plan = original_intercept(*call_args, **call_kwargs)
                try:
                    visuals.offer_plan(plan, width_m=config.settings.open_width_m)
                except Exception as error:
                    journal.emit('visualization_error', error_type=type(error).__name__, error=str(error))
                return plan
            planner.intercept = intercept_with_visualization
        planner.collision_sink = visuals.submit
        detector_model = calibrated_model(config)
        result['model_startup_wall_s'] = time.monotonic()-begin
        with ExitStack() as resources:
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
            camera_config = dict(config.raw['camera'])
            camera_config.update(enable_color=True, require_color=True, timestamp_source='device',
                serial_number=calibration['serial_number'], calibration_id=calibration['calibration_id'],
                camera_to_world_cv=calibration['depth_to_base_cv'])
            camera = OrbbecCameraBackend(camera_config)
            resources.callback(camera.close)
            camera.start()  # exactly one SDK capture owner
            vision = LiveVision(camera, calibration, config, detector_model, hw.snapshot_nowait)
            if args.preview or recorder is not None:
                def offer_frame(frame, d2c):
                    if args.preview:
                        visuals.offer_frame(frame, d2c)
                    if recorder is not None:
                        recorder.offer_frame(frame, d2c)
                vision.frame_sink = offer_frame
            if args.preview:
                vision.speed_sink = visuals.offer_speed
            resources.callback(vision.close)
            vision.start()
            result['device_startup_wall_s'] = time.monotonic()-begin-result['model_startup_wall_s']
            from .cycle_loop import run_pick_cycles
            def cycle_phase(phase, **context):
                nonlocal executor
                executor = executors[-1] if executors else None
                set_phase(phase, **context)
            run_pick_cycles(config, planner, hw, vision, count=result['requested_count'], cancelled=cancelled,
                result=result, reports=reports, executors=executors, set_phase=cycle_phase,
                journal=journal, executor_factory=TwoSecondExecutor,
                finish_attempt=save_attempt if args.continuous else None,
                offer_target=visuals.offer_target if args.preview or args.record_rgbd else None)
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
        monitor.close()
        for number, handler in old_signals.items():
            signal.signal(number, handler)
        result['last_runtime'] = progress_snapshot()
        result['vision_diagnostics'] = result['last_runtime']['vision']
        if recorder is not None:
            result['rgbd_recording'] = recorder.close()
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
        for i, executor in enumerate(executors):
            index=getattr(executor,'cycle_index',i+1)
            if getattr(executor,'trace_saved',False):continue
            try:
                save_attempt(executor,index)
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
        temporary = args.output/'summary.json.tmp'
        try:
            temporary.write_text(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
            temporary.replace(args.output/'summary.json')
        except OSError as error:
            result['log_save_errors'].append(f'summary: {type(error).__name__}: {error}')
        console.finish(result, args.output)
    return 0 if result['status'] in ('requested_feedback_cycles_completed', 'attempt_completed_grip_unconfirmed') else 1


if __name__ == '__main__':
    raise SystemExit(main())

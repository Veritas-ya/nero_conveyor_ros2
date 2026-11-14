"""Independent fixed-follow phase allocation experiment; default connects no devices."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import yaml

from .config import CellConfig
from .diagnostics import json_value
from .phase_timing import KEY, verify_plan
from .phase_timing_experiment import digest, source_hashes, write_new


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE = ROOT/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


def prepare_test(base, output, *, follow_s=.1, join_times=(1.1,),
                 brake_times=(.4, .6, .9, 1.2), mode='robust', stop_interval=None,
                 maximum_target_length_m=.20, timing_history_root=None, timing_history_run=None,
                 local_target_geometry_enabled=None):
    """Write a private snapshot. Site settings and motion limits are never edited."""
    base, output = Path(base).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError('测试目录已存在，请换一个 --output：'+str(output))
    cfg = CellConfig.load(base)
    if KEY in cfg.raw or 'execution_delay_window' in cfg.raw:
        raise ValueError('请使用原始现场配置；不可叠加已有阶段分配/延迟窗口配置')
    raw = yaml.safe_load(base.read_text())
    for key in ('camera_calibration_file', 'tcp_calibration_file', 'taught_transfer_file', 'taught_return_file'):
        if raw.get(key):
            raw[key] = str((base.parent/Path(raw[key]).expanduser()).resolve(strict=True))
    raw.pop('_phase_timing_hardware_test_authorized', None)
    raw['maximum_target_length_m'] = maximum_target_length_m
    if local_target_geometry_enabled is not None:
        raw['local_target_geometry_enabled'] = local_target_geometry_enabled
    raw.setdefault('pick_settings', {})['full_plan_close_window_s'] = follow_s
    raw[KEY] = {
        'mode': mode, 'join_candidates_s': list(join_times),
        'follow_candidates_s': [follow_s], 'brake_candidates_s': list(brake_times),
        'closure_window_policy': 'fixed_follow_experiment',
        'gripper_close_time_s': .1, 'gripper_time_margin_s': .04,
        'residual_time_bound_s': .015, 'position_margin_m': .003,
        'stop_interval_base_m': (cfg.follow_bounds.tolist() if stop_interval is None else list(stop_interval)),
        'max_tcp_acceleration_m_s2': 4.,
        'calibration_status': 'experimental_hardware',
        'calibration_note': 'Explicit fixed-follow hardware experiment, not measured calibration. '
            'Follow duration is operator selected, NOT a proven jaw closure bound. '
            '100ms grip confirmation may continue during braking under scheduled_attempt. '
            '15ms residual timing, 3mm position and 4m/s2 TCP bounds are design assumptions; '
            'not measured uncertainty or no-slip guarantees. Gripper timing fields are '
            'retained as assumptions, not used as a fixed-follow feasibility certificate.'}
    from .planning_timing_history import startup_estimate
    initial, initial_source = startup_estimate(
        ROOT/'runs/phase_timing_test' if timing_history_root is None else timing_history_root,
        base_sha256=digest(base), phase_timing=raw[KEY],
        maximum_target_length_m=maximum_target_length_m, run=timing_history_run)
    configured = cfg.raw.get('measured_planning_initial_s', 0.)
    if configured > initial:
        initial, initial_source = configured, {'policy':'base_configuration', 'path':str(base)}
    raw['measured_planning_initial_s'] = initial
    raw['measured_planning_initial_source'] = initial_source
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp)/'test'; stage.mkdir()
        path = stage/'effective.yaml'
        path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
        CellConfig.load(path)
        write_new(stage/'experiment.json', {
            'schema': 'nero_phase_timing_hardware_test_v1', 'base': str(base),
            'base_sha256': digest(base), 'config_sha256': digest(path),
            'phase_timing': raw[KEY], 'algorithm': 'discrete_candidate_search_not_global_optimization',
            'maximum_target_length_m':maximum_target_length_m,
            'local_target_geometry_enabled':raw.get('local_target_geometry_enabled', True),
            'host':platform.node(), 'measured_planning_initial_s':initial,
            'measured_planning_initial_source':initial_source,
            'source_sha256': source_hashes(), 'physical_parameters_validated': False,
            'site_configuration_modified': False})
        stage.rename(output)
    return output/'effective.yaml'


def plan_once(config, target, state, *, clock=time.monotonic, cancelled=lambda: False):
    """Python planning API. Caller supplies timestamp-aligned target and arm state.

    Returns the existing certified InterceptPlan; never connects or sends commands.
    Keep model setup outside the replay clock; online time remains wall time.
    """
    from .planning import MeasuredPlanner, calibrated_model
    from .taught import TaughtSegment
    cfg = deepcopy(config)
    if KEY not in cfg.raw:
        raise ValueError('plan_once requires a phase_timing configuration')
    if cfg.raw.get('taught_playback_enabled', True):
        cfg.raw['taught_segments'] = {
            kind: TaughtSegment.load(cfg.raw['taught_'+kind+'_file'], cfg, kind)
            for kind in ('transfer', 'return')}
    planner = MeasuredPlanner(cfg, model=calibrated_model(cfg), clock=clock)
    plan = planner.intercept(deepcopy(target), deepcopy(state), cancelled=cancelled)
    verify_plan(cfg, plan.path, plan.reference)
    return plan


def replay_one(config_path, run, output):
    """Exercise the same full planner on one frozen exposure, without hardware."""
    from .paper_replay import restore_input
    from .planning import MeasuredPlanner, calibrated_model
    from .taught import TaughtSegment
    cfg = CellConfig.load(config_path)
    target, state, info = restore_input(Path(run), cfg)
    if cfg.raw.get('taught_playback_enabled', True):
        cfg.raw['taught_segments'] = {
            kind: TaughtSegment.load(cfg.raw['taught_'+kind+'_file'], cfg, kind)
            for kind in ('transfer', 'return')}
    model = calibrated_model(cfg)
    started = time.monotonic()
    clock = lambda: info['submission_s'] + time.monotonic()-started
    planner = MeasuredPlanner(cfg, model=model, clock=clock)
    report = {'input': info, 'commands_sent': 0, 'hardware_connections': 0,
              'physical_success_inferred': False, 'config_sha256': digest(config_path),
              'limitations': ['One frozen exposure; open jaw and model-derived flange assumed.',
                             'Current calibration/config, not historical hardware execution.',
                             'No physical grip, friction or controller dynamics validation.']}
    try:
        plan = planner.intercept(target, state)
        verify_plan(cfg, plan.path, plan.reference)
        report.update(status='plan_ready', selected_times_s={
            'join': plan.reference.prealign_s-plan.reference.rotate_first_s,
            'follow': plan.reference.follow_s, 'brake': plan.reference.brake_s},
            diagnostics=plan.path.diagnostics,
            remaining_send_lead_s=plan.reference.start_s-cfg.raw['command_advance_s']-clock())
    except (RuntimeError, ValueError) as error:
        report.update(status='planning_rejected', error=str(error),
                      details=getattr(error, 'details', {}))
    report.update(planning_wall_s=time.monotonic()-started, attempts=planner.attempt_diagnostics)
    write_new(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='独立真机阶段时长实验：固定跟随，搜索接近和减速；默认只检查。')
    parser.add_argument('--config', type=Path, default=DEFAULT_BASE, help='原始现场配置，仅复制不修改')
    parser.add_argument('--output', type=Path, help='新的测试目录；默认 runs/phase_timing_test/时间戳')
    parser.add_argument('--timing-history-run', type=Path,
                        help='指定兼容测时日志；默认自动选最近配置匹配的本地独立实验日志')
    parser.add_argument('--full-frame-geometry', action='store_true',
                        help='仅在本次私有配置关闭工件局部识别，用于原完整帧流程对照')
    parser.add_argument('--follow-s', type=float, default=.1, help='固定跟随秒数（默认0.1，尚非实测闭合保证）')
    parser.add_argument('--max-target-length-cm', type=float, default=20.,
                        help='识别的工件长边上限cm，默认20；夹爪开口和碰撞检查仍独立生效')
    parser.add_argument('--join-times', type=float, nargs='+', default=[1.1],
                        help='接近时间候选；默认固定1.1秒')
    parser.add_argument('--brake-times', type=float, nargs='+', default=[.4, .6, .9, 1.2])
    parser.add_argument('--mode', choices=['nominal', 'robust'], default='robust')
    parser.add_argument('--stop-interval', type=float, nargs=2, help='沿带方向的停止区间m，默认沿用抓取区间')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true', help='明确连接真实相机/CAN并执行实验')
    action.add_argument('--check-config', action='store_true', help='只验证配置；默认行为')
    action.add_argument('--replay-run', type=Path, help='已有抓取日志目录；只做完整离线规划')
    parser.add_argument('--belt-speed-cm-s', type=float, help='真机单帧入口使用的实际带速，例如50.8')
    cycles = parser.add_mutually_exclusive_group()
    cycles.add_argument('--count', type=int, default=1)
    cycles.add_argument('--continuous', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--record-rgbd', action='store_true')
    parser.add_argument('--freeze-source', choices=['single_exposure_native_depth', 'tracked'],
                        default='single_exposure_native_depth')
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error('--count 必须大于零')
    if args.belt_speed_cm_s is not None and (not np.isfinite(args.belt_speed_cm_s) or args.belt_speed_cm_s <= 0):
        parser.error('--belt-speed-cm-s 必须为正有限数')
    if not args.replay_run and args.freeze_source == 'single_exposure_native_depth' and args.belt_speed_cm_s is None:
        parser.error('单帧入口检查/执行必须填写 --belt-speed-cm-s；例如50.8表示0.508m/s')
    if args.replay_run and args.belt_speed_cm_s is not None:
        parser.error('回放使用日志中的速度，不接受 --belt-speed-cm-s 覆盖')
    output = args.output or ROOT/'runs/phase_timing_test'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    try:
        config = prepare_test(args.config, output, follow_s=args.follow_s,
            join_times=args.join_times, brake_times=args.brake_times, mode=args.mode,
            stop_interval=args.stop_interval, maximum_target_length_m=args.max_target_length_cm/100.,
            timing_history_run=args.timing_history_run,
            local_target_geometry_enabled=False if args.full_frame_geometry else None)
        output = config.parent
        write_new(output/'invocation.json', {'arguments': vars(args), 'effective_config': str(config),
            'execution_requested': args.execute, 'actual_hardware_execution_verified': False})
        if args.replay_run:
            report = replay_one(config, args.replay_run, output/'replay.json')
            print(json.dumps(json_value({k: report[k] for k in
                ('status', 'selected_times_s', 'error', 'planning_wall_s') if k in report}), ensure_ascii=False))
            return 0 if report['status'] == 'plan_ready' else 1
        from .main import main as hardware_main
        forwarded = ['--config', str(config), '--output', str(output/'hardware'),
                     '--execute' if args.execute else '--check-config']
        if args.belt_speed_cm_s is not None:
            forwarded += ['--belt-speed-cm-s', str(args.belt_speed_cm_s)]
        forwarded += ['--continuous'] if args.continuous else ['--count', str(args.count)]
        if args.preview: forwarded.append('--preview')
        if args.record_rgbd: forwarded.append('--record-rgbd')
        print(f'[阶段时长实验] 配置：{config}', flush=True)
        timing = CellConfig.load(config).raw
        print(f'[工件识别] {"候选局部处理，边界不确定时回退" if timing["local_target_geometry_enabled"] else "原完整帧处理"}', flush=True)
        print(f'[规划测时] 首轮估计 {timing["measured_planning_initial_s"]:.3f}s；'
              f'来源 {timing["measured_planning_initial_source"].get("run", timing["measured_planning_initial_source"]["policy"])}', flush=True)
        print(f'[工件尺寸] 识别长边上限 {args.max_target_length_cm:g} cm；完整点云仍用于抓姿和碰撞检查', flush=True)
        print(f'[阶段时长实验] 固定跟随 {args.follow_s:g}s；接近 {args.join_times}；减速 {args.brake_times}', flush=True)
        print('[阶段时长实验] 跟随时间未经闭合实测验证；夹持确认允许跨越减速段。', flush=True)
        code = hardware_main(forwarded, freeze_source=args.freeze_source,
                             phase_timing_hardware_test=True)
        write_new(output/'entry_result.json', {'return_code': code, 'execution_requested': args.execute,
            'hardware_log_directory': str(output/'hardware')})
        return code
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'test_entry_rejected', 'error': str(error)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

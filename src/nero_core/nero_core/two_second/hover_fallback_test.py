"""Ordinary interception, then feedback-gated hover descent on rejection."""
import argparse
from copy import deepcopy
from datetime import datetime
from functools import partial
import json
from pathlib import Path

import numpy as np
import yaml

from .command_history import CommandHistory
from .config import CellConfig
from .execution import TwoSecondExecutor, InterceptUnavailable, validate_start_state
from .hover_descent_test import prepare_config as prepare_hover_config
from .phase_timing_experiment import digest, source_hashes, write_new
from .phase_timing_test import DEFAULT_BASE, ROOT
from .velocity_source import with_fixed_belt_speed
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


class HoverFallbackExecutor(TwoSecondExecutor):
    def __init__(self, config, planner, device, *, standard_config, force_fallback=False,
                 fallback_config=None, **kwargs):
        super().__init__(config, planner, device, **kwargs)
        self.hover_config = config if fallback_config is None else deepcopy(fallback_config)
        if not self.hover_config.raw.get('_hover_descent_authorized'):
            raise ValueError('自动补救需要独立入口授权')
        self.force_fallback = force_fallback
        self.standard_config = deepcopy(standard_config)
        # Hardware startup loads the taught routes and fixes the velocity.
        for key in ('taught_segments', '_prepared_taught_entries',
                    '_taught_entry_prewarm_rejections', 'freeze_source'):
            if key in config.raw:
                self.standard_config.raw[key] = config.raw[key]
                self.hover_config.raw[key] = config.raw[key]
        self.standard_config = with_fixed_belt_speed(
            self.standard_config, config.raw['fixed_belt_speed_m_s'])
        # The previous attempt may have selected hover. Recovery can run even
        # before this new owner submits its first ordinary plan.
        self.planner.config = config

    def _select_config(self, config):
        # Both snapshots share calibration/geometry/limits. Only timing differs.
        # This may only run before a trajectory or gripper command is sent.
        if self.command_history.command_count or self.holding.sent or self.sender is not None:
            raise PickRejected('已开始轨迹或闭爪，禁止切换补救模式')
        self.config = self.planner.config = config
        self.holding.cfg = config.settings
        self.command_history = CommandHistory(config)

    def _execute_intercept(self, pool, target, state, prefetched, version):
        self._active(self.owner, version, target.epoch)
        self._select_config(self.standard_config)
        raw = self.standard_config.raw
        budget = min(raw['planning_budget_s'], raw['hover_fallback_normal_budget_s'])
        try:
            if self.force_fallback:
                self._emit('hover_fallback_test_injected', dict(identity=target.identity,
                    capture_s=target.capture_s, normal_planning_skipped=True,
                    natural_normal_failure_verified=False))
                raise InterceptUnavailable('测试选项主动触发补救，未判定普通抓取不可行',
                    details={'reason': 'test_forced_normal_rejection', 'test_injected': True})
            self._emit('hover_fallback_normal_started', dict(identity=target.identity,
                capture_s=target.capture_s, planning_budget_s=budget))
            # A single deadline spans every ordinary planning retry. The
            # site's carry/return/recovery budgets remain unchanged.
            plan, state, begin = self._plan_standard_intercept(
                pool, target, state, prefetched, version, planning_budget_s=budget)
        except InterceptUnavailable as error:
            # Do not catch stream, device, feedback, version, or grip failures.
            self._active(self.owner, version, target.epoch)
            from .streaming import fresh_feedback
            latest = self.device.snapshot()
            fresh_feedback(latest, self.clock(), self.feedback_stamps)
            self.planner.model.check_feedback(latest)
            if self.bin_start is None:
                validate_start_state(self.config, latest)
            else:
                from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
                validate_start_state(self.config, latest, release_joints=self.bin_start.release_joints,
                    tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
                reason = self.frozen_entry.invalid_reason(self.config, self.epoch(), self.clock())
                if reason is not None:
                    raise PickRejected('料框补救工件已失效', details={'reason':reason})
            if np.max(abs(np.asarray(latest['joints'])-np.asarray(state['joints']))) > .005:
                raise PickRejected('普通规划后起点改变，禁止启动跟随补救') from error
            if self.carry_service is not None:
                self.carry_service.discard()
                self.carry_service = None  # Native carry prediction belongs to the ordinary timing.
            self._select_config(self.hover_config)
            self._emit('hover_fallback_triggered', dict(identity=target.identity,
                capture_s=target.capture_s, observation_age_s=self.clock()-target.capture_s,
                normal_error=str(error), normal_details=error.details,
                trigger_source='test_injection' if self.force_fallback else 'normal_planning',
                start_state=latest, target_reobserved=False))
            from .hover_execution import execute_intercept
            # This plans from NOW using the original exposure, checks remaining
            # workspace/time, then uses the existing sync gate and stop branch.
            return execute_intercept(self, pool, target, latest)
        self._emit('hover_fallback_normal_selected', dict(identity=target.identity))
        state = self._stream(plan.path, origin=plan.reference.start_s,
            epoch=target.epoch, reference=plan.reference)
        return plan, state, begin


def prepare_config(base, output, *, normal_budget_s=.6, force_fallback=False, continuous=False):
    if not np.isfinite(normal_budget_s) or normal_budget_s <= 0:
        raise ValueError('普通规划预算必须为正有限秒数')
    base = Path(base).resolve(strict=True)
    original = CellConfig.load(base)
    if original.raw.get('delivery_repair', {}).get('enabled'):
        raise ValueError('自动跟随补救不能叠加 delivery_repair，请使用原始现场配置')
    hover_path = prepare_hover_config(base, output, enabled=True, grasp=True)
    hover_raw = yaml.safe_load(hover_path.read_text())
    standard_raw = yaml.safe_load(base.read_text())
    # Keep ordinary intercept/closure/braking timing from the actual base.
    for key in ('camera_calibration_file', 'tcp_calibration_file',
                'taught_transfer_file', 'taught_return_file'):
        if key in hover_raw:
            standard_raw[key] = hover_raw[key]
    for key in ('continuous_lookahead_enabled', 'carry_lookahead_enabled',
                'grasp_workpiece_collision_check_enabled', 'runtime_sweep_mode',
                'maximum_target_length_m'):
        standard_raw[key] = hover_raw[key]
    standard_raw['hover_fallback_normal_budget_s'] = min(normal_budget_s, original.raw['planning_budget_s'])
    standard_path = hover_path.parent/'normal.yaml'
    standard_path.write_text(yaml.safe_dump(standard_raw, allow_unicode=True, sort_keys=False))
    CellConfig.load(standard_path)
    write_new(hover_path.parent/'fallback.json', dict(
        schema='nero_hover_fallback_test_v1', base=str(base), base_sha256=digest(base),
        normal_config=str(standard_path), normal_sha256=digest(standard_path),
        hover_config=str(hover_path), hover_sha256=digest(hover_path),
        normal_planning_budget_s=standard_raw['hover_fallback_normal_budget_s'],
        force_fallback=force_fallback,
        source_sha256=source_hashes(), single_exposure=True, continuous=continuous,
        maximum_attempts=None if continuous else 1,
        physical_success=None, site_configuration_modified=False))
    return hover_path, standard_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--belt-speed-cm-s', type=float, required=True,
        help='实际带速，0 < v <= 20 cm/s；不控制传送带电机')
    parser.add_argument('--normal-planning-budget-s', type=float, default=.6,
        help='普通规划预算，默认0.6秒；预算耗尽也会尝试补救')
    parser.add_argument('--force-fallback', action='store_true',
        help='测试专用：跳过普通规划并触发补救；仍检查补救可行性及实际同步反馈')
    parser.add_argument('--continuous', action='store_true',
        help='连续抓取；每次确认回等待位后接收下一件，Ctrl+C退出')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true', help='连接真机；默认一次尝试，--continuous 持续运行')
    action.add_argument('--check-config', action='store_true', help='只校验，不连接设备（默认）')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(argv)
    try:
        speed = args.belt_speed_cm_s
        if not np.isfinite(speed) or not 0 < speed <= 20:
            raise ValueError('当前跟随下降实现只支持大于0且不超过20 cm/s')
        output = args.output or ROOT/'runs/hover_fallback_test'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        hover, normal = prepare_config(args.config, output,
            normal_budget_s=args.normal_planning_budget_s, force_fallback=args.force_fallback,
            continuous=args.continuous)
        standard_config = with_fixed_belt_speed(CellConfig.load(normal), speed/100.)
        with_fixed_belt_speed(CellConfig.load(hover), speed/100.)
        write_new(hover.parent/'invocation.json', dict(arguments=vars(args),
            execution_requested=args.execute, hardware_execution_verified=False))
        if not args.execute:
            print(json.dumps(dict(status='configuration_valid', belt_speed_cm_s=speed,
                force_fallback=args.force_fallback, continuous=args.continuous,
                normal_config=str(normal), hover_config=str(hover),
                hardware_connections=0, commands_sent=0), ensure_ascii=False))
            return 0
        from .main import main as hardware_main
        forwarded = ['--config', str(hover), '--output', str(hover.parent/'hardware'),
            '--execute', '--belt-speed-cm-s', str(speed), '--record-rgbd']
        forwarded.extend(['--continuous'] if args.continuous else ['--count', '1'])
        if args.preview:
            forwarded.append('--preview')
        return hardware_main(forwarded, freeze_source='single_exposure_native_depth',
            phase_timing_hardware_test=True, hover_descent_entry=True,
            executor_class=partial(HoverFallbackExecutor, standard_config=standard_config,
                force_fallback=args.force_fallback))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(status='hover_fallback_entry_rejected', error=str(error),
            hardware_execution_verified=False), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

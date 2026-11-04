#!/usr/bin/env python3
"""Physical GraspGen-X benchmark with the deployed Nero7 vision chain.

NN and candidate checks run asynchronously while normal MuJoCo steps continue.
No fixed inference delay or pause-to-wait-for-inference is installed. Wall-clock
pacing is the default; measured wall/simulation lag is part of every result.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.graspgenx_sim import run
from experiments.graspgenx_sim.continuous import ContinuousGraspAdapter
from experiments.graspgenx_sim.nero7_policy import Nero7NetworkSeedPolicy
from nero_core.sim_common import make_policy
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation, Nero7SensorControlEnvironment


def measurement_rates(stamps, episode_timing):
    """Separate capture-clock frequency from full-episode wall throughput."""
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.
    native_rate = (len(stamps)-1)/span if span > 0. else None
    wall = episode_timing.get('wall_seconds')
    valid_wall = isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0.
    return {
        'processed_rate_hz': native_rate,
        'processed_rate_clock': 'simulation_capture_time',
        'processed_reports_per_wall_episode_s': len(stamps)/wall if valid_wall else None,
        'wall_rate_interval': 'full timed episode, including periods without processed reports',
        'episode_wall_seconds': wall,
        'episode_sim_seconds': episode_timing.get('sim_seconds'),
    }


def _adapter(env, *, adapter_class=ContinuousGraspAdapter, **kwargs):
    original = make_policy(env, compensation=True)
    config = replace(original.config, control_timestep=.01,
        cartesian_linear_speed=.25, tcp_speed_limits_enabled=True,
        trajectory_velocity_limits=(2.,)*7, trajectory_acceleration_limits=(2.,)*7,
        tracking_velocity_feedforward_time=.04, closing_tracking_lead_time=.04,
        closure_contact_lead_time=.04, descent_lead_time=.8,
        closure_horizontal_tolerance=.008, closure_vertical_tolerance=.004,
        closure_orientation_tolerance=.025)
    original.close()
    parameters = {name: kwargs[name] for name in (
        'enable_speculative_prefetch', 'acquisition_translation_step_m',
        'acquisition_command_integrator', 'visual_acquisition_tracking',
        'minimum_motor_hold_effort', 'visual_servo_bias_compensation') if name in kwargs}
    policy = Nero7NetworkSeedPolicy(config, env=env, **parameters)
    return adapter_class(env, policy=policy, **kwargs)


def _correct_result_metadata(output, env, timing):
    """Write factual declarations after the generic runner's legacy report.

    This changes result/report descriptions only, never physics, observations,
    controller inputs, success metrics or any runtime admission condition.
    """
    path = output/'result.json'
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding='utf-8'))
    camera = env.camera_record()
    result['camera'] = camera
    old = 'Fixed camera matches the saved relative pose; conveyor layout remains simulated.'
    actual = ('Declared synthetic near-top-down camera at world[-0.35,0,1.25]m; '
              'render intrinsics reference a saved capture, while camera pose and '
              'conveyor/bin layout are simulated rather than a field digital twin.')
    result['limitations'] = [actual if line == old else line for line in result.get('limitations', [])]
    if actual not in result['limitations']:
        result['limitations'].append(actual)
    result['nero7_perception'] = {
        'source': 'same deployed Nero7 detector/BlockTracker/XY Kalman parameters',
        'belt_encoder_used': False, 'initial_velocity_prior': None,
        'prediction_uncertainty_status': 'not_estimated',
        'prediction_uncertainty_numeric_zero': 'legacy numeric compatibility placeholder; not measured precision',
        'speculative_prefetch_enabled': bool(result['config'].get('control', {}).get('speculative_prefetch', False)),
        'kalman_covariance': 'internal filter state covariance, not independent measured speed accuracy',
        'task_identity': 'existing Nero7GraspTracking._select_track callable with original unique geometry/velocity gate',
        'speed_qualification': 'actual Nero7VelocitySource.query callable with recent_measured_speed original capture timestamp',
        'actual_timing': timing}
    run.save_json(path, result)
    run.write_report(output, result)
    report_path = output/'report.txt'
    report = report_path.read_text(encoding='utf-8')
    report = report.replace('相机：保存数据的固定外置相机相对基座姿态；当前传送带/箱体布局属于已有仿真。',
        '相机：明确声明的近俯视模拟相机，世界位置[-0.35,0,1.25]米；内参参考保存帧，外参及传送带/箱体布局属于模拟条件。')
    report = report.replace('控制：视觉深度组件跟踪 + 带速补偿 + 既有NERO运动控制；GraspGen-X提供抓取姿态。',
        '控制：同一Nero7深度检测、BlockTracker与XY卡尔曼测速 + 共享VisualAcquisition与原规划控制；GraspGen-X提供抓姿。')
    report = report.replace('正式控制：延迟视觉buffer、关节与夹爪编码器、执行器力和带速编码器；不读取对象真值位姿、接触对象ID或物理生命周期。',
        '正式控制：带原捕获时刻的Nero7视觉位置/合格速度、关节与夹爪编码器、执行器力；无带速编码器或固定速度先验。对象真值及接触对象ID只用于独立评分。')
    report += ('预测不确定度：尚未评估；输出中的数值零仅为兼容字段，不表示零误差。卡尔曼协方差不是独立参考测得的测速精度。\n'
               '速度资格：直接复用真机source.query；最近合格速度保留其原始观测时间，不由新位置刷新年龄。\n')
    report_path.write_text(report, encoding='utf-8')


def main(argv=None, *, adapter_type=None):
    # The current project uses only full_preplanned. This historical entry's
    # policy and actuator transport are not the full-planning controller.
    # Refuse before opening a renderer, loading weights or advancing physics.
    from experiments.graspgenx_sim.full_preplanned_entry import main as current_entry
    return current_entry(argv)


def _historical_main_not_a_current_entry(argv=None, *, adapter_type=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if '--config' not in args:
        args = ['--config', str(Path(__file__).with_name('nero7_config.yaml')), *args]
    if '--output' not in args:
        args += ['--output', str(Path.cwd()/'runs'/datetime.now().strftime('nero7_%Y%m%d_%H%M%S_%f'))]
    print('注意：视觉网络物理诊断使用历史 Nero7 仿真策略；未接入当前 full_preplanned/P1-A。', flush=True)
    output = Path(args[args.index('--output')+1]).resolve()
    source_paths = [Path(__file__), Path(__file__).with_name('nero7_perception.py'),
                    Path(__file__).with_name('nero7_config.yaml'), Path(__file__).with_name('nero7_policy.py'),
                    Path(__file__).with_name('nero7_identity.py'), Path(run.__file__),
                    ROOT/'vendor/nero7_speed_reference/upstream_perception.yaml',
                    ROOT/'mujoco_conveyor_sorting/runtime/nero7_belt_velocity.py',
                    ROOT/'mujoco_conveyor_sorting/runtime/nero7_velocity_source.py',
                    ROOT/'mujoco_conveyor_sorting/runtime/nero7_grasp_tracking.py']
    source_before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    environments = []
    def create_environment(**kwargs):
        env = Nero7FixedSimulation(**kwargs)
        environments.append(env)
        return env
    try:
        return run.main(args, environment_type=create_environment,
                        control_environment_type=Nero7SensorControlEnvironment,
                        adapter_type=adapter_type or _adapter)
    finally:
        if environments:
            env = environments[0]
            output.mkdir(parents=True, exist_ok=True)
            run.save_json(output/'nero7_reports.json', env.nero7_reports)
            run.save_json(output/'nero7_capture_timings.json', env.nero7_capture_timings)
            run.save_json(output/'nero7_planning_diagnostics.json', env.nero7_planning_diagnostics)
            stamps = [r['capture_sim_s'] for r in env.nero7_reports]
            result_path = output/'result.json'
            episode_timing = (json.loads(result_path.read_text()).get('timing', {})
                              if result_path.is_file() else {})
            timing = {'captured_frames': env._nero_captured, 'processed_frames': len(stamps),
                'skipped_pending_frames': env._nero_skipped, 'capture_rate_upper_bound_hz': 30.,
                **measurement_rates(stamps, episode_timing),
                'hardware_access': False, 'fixed_inference_delay_s': None,
                'physics_pauses_for_inference': False, 'initial_velocity_prior': None,
                'maximum_coast_s': 1., 'control_hz': 1/env.control_dt,
                'shared_acquisition_method_identity': Nero7NetworkSeedPolicy.shared_method_identity(),
                'source_before_sha256': source_before,
                'source_after_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                'source_unchanged': all(hashlib.sha256(p.read_bytes()).hexdigest()==source_before[str(p.relative_to(ROOT))] for p in source_paths)}
            run.save_json(output/'nero7_timing.json', timing)
            _correct_result_metadata(output, env, timing)


if __name__ == '__main__':
    raise SystemExit(main())

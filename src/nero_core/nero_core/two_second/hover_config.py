"""Explicit admission profiles for experimental staged interception."""
import numpy as np

KEY = 'hover_descent'
NATIVE_MAXIMUM_SPEED_M_S = .284
DEFAULTS = dict(enabled=False, grasp=False, ramp_times_s=[1., 1.2, 1.4],
    hover_heights_m=[.04, .03, .02], descent_times_s=[.35, .45, .6],
    alignment_fractions=[.75, 1.],
    sync_hold_s=.30, send_guard_s=.03, preparation_reserve_s=1.2,
    maximum_speed_m_s=.20, admission_profile='low_speed_test',
    feedback_max_age_s=.06, observation_span_s=.04,
    horizontal_tolerance_m=.005, height_tolerance_m=.005,
    orientation_tolerance_rad=.05236, speed_tolerance_m_s=.02,
    joint_tracking_tolerance_rad=.03)


def validate(raw):
    value = raw.get(KEY)
    if value is None:
        return None  # Do not change legacy configuration semantics.
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('hover_descent 字段无效')
    spec = {**DEFAULTS, **value}
    for key in ('enabled', 'grasp'):
        if not isinstance(spec[key], bool):
            raise ValueError('hover_descent '+key+' 必须为布尔值')
    lists = ('ramp_times_s', 'hover_heights_m', 'descent_times_s', 'alignment_fractions')
    for key in lists:
        values = spec[key]
        if (not isinstance(values, list) or not 1 <= len(values) <= 8 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0 for x in values)):
            raise ValueError('hover_descent '+key+' 必须为正有限候选列表')
    if any(x > 1. for x in spec['alignment_fractions']):
        raise ValueError('姿态对齐必须在速度同步之前完成')
    profile = spec['admission_profile']
    if profile not in ('low_speed_test', 'native_fallback_experiment'):
        raise ValueError('未知分段接续准入配置')
    speed_limit = .20 if profile == 'low_speed_test' else min(
        NATIVE_MAXIMUM_SPEED_M_S, raw.get('maximum_measured_speed_m_s', .20))
    for key in set(DEFAULTS)-{'enabled','grasp','admission_profile',*lists}:
        x = spec[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0:
            raise ValueError('hover_descent '+key+' 必须为正有限数')
    if spec['maximum_speed_m_s'] > speed_limit or spec['feedback_max_age_s'] > .06:
        raise ValueError(f'分段接续当前配置限速{speed_limit*100:g}cm/s、反馈年龄不超过60ms')
    if (spec['horizontal_tolerance_m'] > .005 or spec['height_tolerance_m'] > .005
            or spec['speed_tolerance_m_s'] > .02 or spec['joint_tracking_tolerance_rad'] > .03
            or spec['orientation_tolerance_rad'] > .05236 or spec['send_guard_s'] < .02):
        raise ValueError('不可放宽分段接续门限或发送余量')
    if spec['sync_hold_s'] <= (raw['command_advance_s']+spec['send_guard_s']+
            spec['feedback_max_age_s']+spec['observation_span_s']):
        raise ValueError('同步跟随时间不足以覆盖提前发送和反馈确认')
    if spec['enabled']:
        if raw.get('trajectory_profile') != 'smooth' or 'phase_timing' not in raw:
            raise ValueError('分段接续需要平滑轨迹及阶段约束')
        if raw.get('closure_policy') != 'scheduled_attempt' or raw.get('hardware',{}).get('joint_command_mode') != 'js':
            raise ValueError('分段接续首版需要 scheduled_attempt 闭爪策略和 JS 连续下发')
        if raw.get('continuous_lookahead_enabled') or raw.get('carry_lookahead_enabled'):
            raise ValueError('独立分段接续暂不支持其他预规划并行入口')
        if not raw.get('grasp_workpiece_collision_check_enabled', True):
            raise ValueError('分段接续必须开启运动工件碰撞检查')
        if raw.get('runtime_sweep_mode', 'inline') != 'inline':
            raise ValueError('分段接续首版需要同步检查实际反馈到当前分支的几何')
        if 'execution_delay_window' in raw or 'timing_experiment' in raw:
            raise ValueError('分段接续不可叠加其他轨迹时序实验')
    raw[KEY] = spec
    return spec


def enabled(config):
    return config.raw.get(KEY, {}).get('enabled', False)


def authorize(config, *, entry=False, speed=None, native_entry=False):
    if not enabled(config):
        return
    spec = config.raw[KEY]
    native = spec['admission_profile'] == 'native_fallback_experiment'
    if not entry or (native and not native_entry):
        raise ValueError('新模式须通过独立入口或显式启用原生抓取补救入口')
    if speed is None or not np.isfinite(speed) or not 0 < speed <= spec['maximum_speed_m_s']:
        raise ValueError(f"请填写实际低速/实验带速，范围0至{spec['maximum_speed_m_s']*100:g}cm/s")
    config.raw['_hover_descent_authorized'] = True

"""Native tracked-target fallback configuration; ordinary timing stays intact."""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from .hover_config import DEFAULTS, NATIVE_MAXIMUM_SPEED_M_S, validate, authorize
from .phase_timing import validate_configuration, authorize_hardware_test


def configure(config):
    """Return normal and staged snapshots after all native CLI overrides load.

    28.4 cm/s is an experimental admission ceiling, not measured grasp proof.
    Joint limits, collision checking and synchronization gates remain mandatory.
    """
    raw = deepcopy(config.raw)
    for key in ('phase_timing', 'execution_delay_window', 'timing_experiment'):
        if key in raw:
            raise ValueError('原生跟随补救不能叠加 '+key)
    if raw.get('hover_descent', {}).get('enabled') or raw.get('delivery_repair', {}).get('enabled'):
        raise ValueError('原生跟随补救不能叠加其他补救入口')
    speed = raw.get('fixed_belt_speed_m_s')
    if (raw.get('velocity_source') != 'operator_fixed_belt_speed' or speed is None
            or not np.isfinite(speed) or not 0 < speed <= min(
                NATIVE_MAXIMUM_SPEED_M_S, raw['maximum_measured_speed_m_s'])):
        raise ValueError('原生跟随补救实验要求实际固定带速大于0且不超过28.4 cm/s')
    if raw['intercept_strategy'] != 'synchronized' or raw['trajectory_profile'] != 'smooth':
        raise ValueError('原生跟随补救需要平滑同步抓取策略')
    # The owner must see ordinary rejections to try hover. Keep next-target
    # acquisition available independently of the speculative trajectory worker.
    previous_lookahead = raw['continuous_lookahead_enabled']
    raw['continuous_lookahead_enabled'] = False
    raw['hover_fallback_normal_budget_s'] = raw['planning_budget_s']
    raw['hover_fallback'] = dict(enabled=True, entry='two_second_pick',
        normal_planning_budget_s=raw['planning_budget_s'],
        maximum_speed_m_s=NATIVE_MAXIMUM_SPEED_M_S, configured_speed_m_s=speed,
        physical_parameters_validated=False,
        previous_continuous_lookahead_enabled=previous_lookahead,
        freeze_during_pick_enabled=previous_lookahead,
        continuous_lookahead_enabled=False, normal_timing_preserved=True,
        target_source='original_native_frozen_target', force_fallback=False)
    normal = replace(config, raw=raw)
    hover_raw = deepcopy(raw)
    hover_raw.update(carry_lookahead_enabled=False,
        grasp_workpiece_collision_check_enabled=True, runtime_sweep_mode='inline')
    # Retain the cell's braking duration. Only the staged close/follow window
    # changes to the existing 0.1 s experiment, with measured grip confirmation.
    follow_s, brake_s = .1, raw['brake_duration_s']
    hover_raw['phase_timing'] = dict(mode='robust',
        join_candidates_s=list(DEFAULTS['descent_times_s']), follow_candidates_s=[follow_s],
        brake_candidates_s=[brake_s], closure_window_policy='fixed_follow_experiment',
        gripper_close_time_s=.1, gripper_time_margin_s=.04, residual_time_bound_s=.015,
        position_margin_m=.003, stop_interval_base_m=config.follow_bounds.tolist(),
        max_tcp_acceleration_m_s2=4., calibration_status='experimental_hardware',
        calibration_note='Native hover fallback experiment. Timing and uncertainty margins '
            'are design assumptions; 28.4 cm/s admission is not physical grasp validation.')
    hover_raw['hover_descent'] = {**deepcopy(DEFAULTS), 'enabled': True, 'grasp': True,
        'admission_profile': 'native_fallback_experiment',
        'maximum_speed_m_s': min(NATIVE_MAXIMUM_SPEED_M_S, raw['maximum_measured_speed_m_s'])}
    hover_raw.setdefault('pick_settings', {})['full_plan_close_window_s'] = follow_s
    hover = replace(config, raw=hover_raw,
        settings=replace(config.settings, full_plan_close_window_s=follow_s))
    validate_configuration(hover.raw, hover.workspace, hover.direction)
    validate(hover.raw)
    authorize_hardware_test(hover, enabled=True)
    authorize(hover, entry=True, native_entry=True, speed=speed)
    return normal, hover

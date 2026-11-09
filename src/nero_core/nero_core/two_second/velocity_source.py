"""Explicit prediction velocity source; operator speed is never a measurement."""
from dataclasses import replace
import numpy as np
from mujoco_conveyor_sorting.controllers._math import belt_plane_motion_from_xy

KALMAN = 'nero7_xy_kalman'
POINT_KALMAN = 'pointcloud_xy_kalman'
FIXED = 'operator_fixed_belt_speed'


def validate_velocity_configuration(raw):
    source = raw.setdefault('velocity_source', KALMAN)
    if source not in (KALMAN, POINT_KALMAN, FIXED):
        raise ValueError('velocity_source 必须为 nero7_xy_kalman、pointcloud_xy_kalman 或 operator_fixed_belt_speed')
    if source == FIXED:
        value = raw.get('fixed_belt_speed_m_s')
        if isinstance(value, bool) or value is None:
            raise ValueError('固定带速须明确提供正有限 fixed_belt_speed_m_s，单位m/s')
        speed = float(value)
        if not np.isfinite(speed) or not 0 < speed <= raw['maximum_measured_speed_m_s']:
            raise ValueError('固定带速须为正有限m/s，且不超过已配置传送带工作速度范围')
        raw['fixed_belt_speed_m_s'] = speed
    elif 'fixed_belt_speed_m_s' in raw:
        raise ValueError('提供固定带速时必须显式选择 operator_fixed_belt_speed')


def with_fixed_belt_speed(config, speed):
    raw = {**config.raw, 'velocity_source':FIXED, 'fixed_belt_speed_m_s':speed,
           'velocity_override_origin':'cli_one_pick'}
    validate_velocity_configuration(raw)
    return replace(config, raw=raw)


def is_fixed(config):
    return config.raw.get('velocity_source', KALMAN) == FIXED


def policy(config):
    fixed = is_fixed(config)
    report = {'source':FIXED if fixed else config.raw.get('velocity_source',KALMAN),
        'fixed_belt_speed_m_s':config.raw.get('fixed_belt_speed_m_s') if fixed else None,
        'fixed_belt_speed_cm_s':config.raw['fixed_belt_speed_m_s']*100. if fixed else None,
        'origin':config.raw.get('velocity_override_origin','configuration' if fixed else 'camera'),
        'visual_velocity_used':not fixed,
        'tracking_source':'nero7_position_identity_tracking',
        'association_prediction_source':FIXED if fixed else KALMAN,
        'id_confirmation_source':'distinct_real_rgbd_observations',
        'internal_tracking_kalman_still_runs':True,
        'speed_semantics':'surface_speed_along_configured_belt_direction' if fixed else 'measured_XY_velocity'}
    if config.raw.get('freeze_source')=='single_exposure_native_depth':
        report.update(tracking_source='none_single_exposure_native_depth',
            association_prediction_source='none_single_exposure_native_depth',
            id_confirmation_source='single_real_native_depth_exposure',
            internal_tracking_kalman_still_runs=False,
            identity_source='capture_epoch_generation_sequence')
    return report


def prediction_velocity(config, measured_velocity, normal):
    if not is_fixed(config):
        return belt_plane_motion_from_xy(measured_velocity, normal)
    direction = belt_plane_motion_from_xy(config.direction, normal)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('固定带速缺少有效的带面运动方向')
    # Keep calibrated XY heading and measured-plane height, with the supplied
    # speed interpreted as metres travelled along the actual belt surface.
    return direction * (config.raw['fixed_belt_speed_m_s']/norm)


def speed_report(config, track, velocity):
    if is_fixed(config):
        report = {'source':FIXED,'configured_speed_m_s':config.raw['fixed_belt_speed_m_s'],
            'prediction_velocity_base_m_s':np.asarray(velocity).tolist(),
            'measured_velocity_base_m_s':None,'visual_velocity_used':False,
            'sample_count':None,'span_s':None,'speed_std_m_s':None,
            'prediction_model':'operator_fixed_surface_speed_with_measured_belt_plane',
            'tracking_source':'nero7_position_identity_tracking',
            'association_prediction_source':FIXED,
            'tracking_sample_count':track.sample_count,
            'internal_tracking_kalman_still_runs':True}
        if getattr(track, 'observation_source', None) == 'single_exposure_native_depth':
            report.update(tracking_source='none_single_exposure_native_depth',
                association_prediction_source='none_single_exposure_native_depth',
                tracking_sample_count=0, internal_tracking_kalman_still_runs=False,
                observation_sample_count=1, identity_source=getattr(track,'identity_source','manually_armed_exposure_event'))
        return report
    return {'source':config.raw.get('velocity_source',KALMAN),'sample_count':track.sample_count,
        'measured_velocity_base_m_s':list(track.velocity_base_m_s),
        'legacy_velocity_base_m_s':getattr(track,'legacy_velocity_base_m_s',None),
        'speed_position_base_m':getattr(track,'speed_position_base_m',None),
        'prediction_model':'constant_XY_velocity_with_measured_belt_plane_height',
        'span_s':track.span_s,'speed_std_m_s':track.speed_std_m_s,
        'velocity_covariance_xy':track.velocity_covariance_xy}

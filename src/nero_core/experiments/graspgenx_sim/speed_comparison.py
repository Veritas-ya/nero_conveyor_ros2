"""Evaluation-only, timestamp-paired simulator and visual belt speed records."""
import numpy as np


def compare_speed_at_capture(*, capture_s, truth_s, truth_velocity_world, estimate_base, base_to_world_rotation, source_stamp_s):
    """Compare a qualified visual velocity to separate same-exposure truth.

    The caller must establish unambiguous object identity. Never feeds truth
    into the detector, tracker, prediction, or controller.
    """
    if not np.isfinite([capture_s,truth_s,source_stamp_s]).all():
        raise ValueError('invalid time')
    if abs(float(capture_s)-float(source_stamp_s)) > 1e-6:
        raise ValueError('visual velocity belongs to a different exposure')
    if abs(float(capture_s)-float(truth_s)) > .00200001:
        raise ValueError('simulator truth is not paired to the exposure')
    v=np.asarray(truth_velocity_world,dtype=float); est=np.asarray(estimate_base,dtype=float); R=np.asarray(base_to_world_rotation,dtype=float)
    if v.shape!=(3,) or est.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(np.r_[v,est,R.ravel()]).all():
        raise ValueError('invalid velocity or coordinate transform')
    e=R@est
    return {'capture_sim_s':float(capture_s),'truth_sim_s':float(truth_s),
            'visual_source_sim_s':float(source_stamp_s),'truth_vx_world_m_s':float(v[0]),
            'visual_vx_world_m_s':float(e[0]), 'error_vx_m_s':float(e[0]-v[0]),
            'truth_velocity_world_m_s':v.tolist(),'visual_velocity_world_m_s':e.tolist(),
            'speed_source':'nero7_xy_kalman','truth_scope':'evaluation_only_not_control'}

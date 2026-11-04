"""Align simulation planning with the actuator envelope without widening it."""
from dataclasses import replace
import numpy as np


def align_simulation_command_envelope(context, actuator, start_q):
    planning = context.robot_controller
    lower = np.maximum(np.asarray(planning.command_joint_lower_limits, float),
                       np.asarray(actuator.command_joint_lower_limits, float))
    upper = np.minimum(np.asarray(planning.command_joint_upper_limits, float),
                       np.asarray(actuator.command_joint_upper_limits, float))
    q = np.asarray(start_q, float)
    if (lower.shape != (7,) or upper.shape != (7,) or q.shape != (7,)
            or not np.isfinite(np.r_[lower, upper, q]).all() or np.any(lower >= upper)
            or np.any(q < lower) or np.any(q > upper)):
        raise ValueError('current simulation feedback is outside the shared command envelope')
    # Planning's hard limits must represent the same robot; a margin alone
    # cannot model arbitrary per-axis differences in hard-stop geometry.
    if not (np.allclose(planning.joint_lower_limits, actuator.joint_lower_limits, atol=1e-9)
            and np.allclose(planning.joint_upper_limits, actuator.joint_upper_limits, atol=1e-9)):
        raise ValueError('simulation and planning hard joint limits differ')
    margin = max(float(context.policy.config.minimum_joint_limit_margin),
                 float(actuator.config.joint_command_margin))
    planning.command_joint_lower_limits = lower.copy()
    planning.command_joint_upper_limits = upper.copy()
    context.policy.config = replace(context.policy.config, minimum_joint_limit_margin=margin)
    return {'effective_lower_rad':lower.tolist(), 'effective_upper_rad':upper.tolist(),
            'effective_planning_margin_rad':margin,
            'source':'intersection_of_simulation_actuator_and_planning_command_limits'}

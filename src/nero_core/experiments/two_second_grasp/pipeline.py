"""Simulation-only snapshots for speculative post-grasp planning."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def close_ramp_action(elapsed, duration):
    """Smooth simulation jaw command from fully open to fully closed."""
    if not np.isfinite([elapsed,duration]).all() or duration < 0:
        raise ValueError('invalid close ramp time')
    if duration == 0:
        return -1.
    u=float(np.clip(elapsed/duration,0.,1.))
    return 1.-2.*(10*u**3-15*u**4+6*u**5)


def feed_delay(spawn_coordinate, freeze_coordinate, speed, clearance_s):
    """Estimate when simulation should feed the next object during release."""
    values=np.asarray([spawn_coordinate,freeze_coordinate,speed,clearance_s],float)
    if not np.isfinite(values).all() or speed<=0 or clearance_s<=0:
        raise ValueError('invalid simulation feed timing')
    travel=max(0.,(freeze_coordinate-spawn_coordinate)/speed)
    return max(0.,clearance_s-travel),travel


def predicted_loaded_snapshot(env, qpos, runtime, endpoint_q):
    """Move an observed TCP/object attachment to a predicted arm endpoint.

    The input is copied; the live simulation is never mutated. This is a
    planning proposal only and must be checked again against actual feedback.
    """
    qpos = np.asarray(qpos, float).copy()
    data = mujoco.MjData(env.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(env.model, data)
    tcp = data.site_xpos[env.ee_site_id].copy()
    rotation = data.site_xmat[env.ee_site_id].reshape(3, 3).copy()
    obj = qpos[runtime.qpos_adr:runtime.qpos_adr+7].copy()
    offset = rotation.T @ (obj[:3]-tcp)
    relative = rotation.T @ Rotation.from_quat(obj[3:][[1, 2, 3, 0]]).as_matrix()
    qpos[env.arm_qpos_adrs] = endpoint_q
    data.qpos[:] = qpos
    mujoco.mj_forward(env.model, data)
    end_rotation = data.site_xmat[env.ee_site_id].reshape(3, 3).copy()
    end_tcp = data.site_xpos[env.ee_site_id].copy()
    end_object_rotation = end_rotation @ relative
    qpos[runtime.qpos_adr:runtime.qpos_adr+7] = np.r_[end_tcp + end_rotation @ offset,
        Rotation.from_matrix(end_object_rotation).as_quat()[[3, 0, 1, 2]]]
    return qpos, np.r_[end_tcp, Rotation.from_matrix(end_rotation).as_quat()[[3, 0, 1, 2]]]


def matching_drop_candidate(goal, fresh_goals, *, position_tolerance_m=1e-6,
                            orientation_tolerance_rad=1e-6):
    """Match a speculative bin pose to freshly generated free-space poses."""
    goal = np.asarray(goal, float)
    if goal.shape != (7,) or not np.isfinite(goal).all():
        return None
    goal_rotation = Rotation.from_quat(goal[3:][[1, 2, 3, 0]])
    for index, candidate in enumerate(fresh_goals):
        candidate = np.asarray(candidate, float)
        if candidate.shape != (7,) or not np.isfinite(candidate).all():
            continue
        angle = (goal_rotation.inv() * Rotation.from_quat(candidate[3:][[1, 2, 3, 0]])).magnitude()
        if np.linalg.norm(goal[:3]-candidate[:3]) <= position_tolerance_m and angle <= orientation_tolerance_rad:
            return index
    return None

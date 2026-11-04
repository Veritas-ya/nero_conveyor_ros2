"""Estimate a fixed-TCP rotation time; never deliver or send a motion here."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .trajectory import retime_samples


def estimate_rotation_lead(config, model, solver, state, start, grasp, *, cancelled):
    """Use the simulator's rest-to-rest retiming rule with physical-cell IK.

    This auxiliary curve supplies a duration only. The complete intercept is
    then rebuilt for that duration and must pass all normal geometry, TCP,
    joint and deadline gates before any part, including rotation, can execute.
    """
    requested = config.raw['rotate_first_s']
    count = max(4, int(np.ceil(requested*config.raw['planning_knot_hz']))+1)
    times = np.linspace(0., requested, count)
    orientations = Slerp([0., 1.], Rotation.from_matrix(np.array([start[:3,:3], grasp[:3,:3]])))
    joints = [np.asarray(state['joints']).copy()]
    for t in times[1:]:
        if cancelled():
            raise PickRejected('预转测时已取消或超出规划预算')
        u = t/requested
        phase = 10*u**3-15*u**4+6*u**5
        quat = orientations([phase]).as_quat()[0][[3,0,1,2]]
        solved = solver.solve(start[:3,3], quat, joints[-1], cancelled=cancelled)
        if not solved.success:
            raise PickRejected('等待位预转逆解未收敛', details={'time_s':float(t),
                'position_error_m':solved.position_error, 'orientation_error_rad':solved.orientation_error})
        joints.append(solved.joint_positions.copy())
    if cancelled():
        raise PickRejected('预转测时已取消或超出规划预算')
    estimate = retime_samples(times, joints, config, model.command_lower, model.command_upper,
        identity=(0, 'rotation_timing_estimate'), version=0, stage='rotation_timing_estimate')
    return {'requested_s':float(requested), 'duration_s':float(estimate.duration),
            'time_scale':estimate.diagnostics['time_scale'], 'ik_solve_count':len(times)-1,
            'geometry_certified':False, 'full_intercept_certification_required':True}

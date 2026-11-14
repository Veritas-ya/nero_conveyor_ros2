"""Finish drop orientation before the final Cartesian approach to the bin."""
from dataclasses import replace
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .motion_profile import select_motion_profile
from .precise_ik import PrecisePoseSolver
from .trajectory import retime_samples, geometry_check_times
from .taught_timing import limit_jerk


def orientation_first_drop(planner, config, state, start, goal, *, cancelled):
    """Backward IK preserves the taught final joint branch, not just its pose.

    The 20 cm approach distance comes from the configured bin region. The
    approach uses the final orientation, then the final segment translates
    at that orientation. All geometry is certified with the current payload.
    """
    from .prepared_cycle import nominal_state, concatenate
    from .carry_blend import blend_checked_path
    began = time.monotonic()
    def abort():
        return cancelled() or time.monotonic()-began > config.raw['planning_budget_s']
    width = float(state['width_m'])
    planner.model.set_state(start,width)
    origin = planner.model.tcp().copy()
    planner.model.set_state(goal,width)
    destination = planner.model.tcp().copy()
    direction = origin[:2,3]-destination[:2,3]
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise PickRejected('提前调整投放姿态缺少水平接近方向')
    radius = config.raw['experimental_direct_drop_guard']['radius_m']
    pre = destination[:3,3].copy()
    pre[:2] += direction/length*min(radius,length)
    pre[2] = max(pre[2],origin[2,3])
    quat = Rotation.from_matrix(destination[:3,:3]).as_quat()[[3,0,1,2]]
    solver = PrecisePoseSolver(planner.model)
    # Spatially short terminal segment. Solve backwards from the exact taught
    # endpoint; there is no late joint-branch correction at the drop point.
    count = max(20,int(np.ceil(np.linalg.norm(pre-destination[:3,3])/.01)))
    u = np.linspace(0.,1.,count+1)
    progress = u**3*(10.+u*(-15.+6.*u))
    positions = pre+(destination[:3,3]-pre)*progress[:,None]
    joints = np.empty((len(u),7));joints[-1] = goal
    for i in range(len(u)-2,-1,-1):
        if abort():
            raise PickRejected('提前调整投放姿态已取消或超出预算')
        solved = solver.solve(positions[i],quat,joints[i+1],cancelled=abort)
        if not solved.success:
            raise PickRejected('固定姿态投放末段逆解未收敛',details={'sample':i,
                'position_error_m':solved.position_error,'orientation_error_rad':solved.orientation_error})
        joints[i] = solved.joint_positions
    planner.version += 1
    terminal = retime_samples(u,joints,select_motion_profile(config,payload=True),
        planner.model.command_lower,planner.model.command_upper,identity=planner.identity,
        version=planner.version,stage='drop_fixed_orientation')
    terminal = limit_jerk(terminal,config.raw.get('taught_joint_jerk_limit_rad_s3'),config.settings.control_hz)
    def sample(t):
        x = np.clip(t/terminal.duration,0.,1.)
        s = x**3*(10.+x*(-15.+6.*x))
        return pre+(destination[:3,3]-pre)*s,quat
    # Includes reference errors as well as current payload / rim geometry.
    reference = SimpleNamespace(sample=sample,duration_s=terminal.duration)
    certificate = planner.check_path(terminal,width,payload=True,reference=reference,cancelled=abort)
    terminal = replace(terminal,diagnostics={**terminal.diagnostics,
        'fixed_drop_orientation':True,'reference_check':certificate,
        'orientation_completed_at_tcp_base_m':pre.tolist()})
    connecting = planner.route(state,stage='direct_drop',goal_joints=joints[0],payload=True,
        start_joints=start,cancelled=abort)
    connecting = limit_jerk(connecting,config.raw.get('taught_joint_jerk_limit_rad_s3'),config.settings.control_hz)
    combined = concatenate((connecting,terminal),identity=planner.identity,
        version=planner.version,stage='direct_drop')
    if (config.raw.get('direct_drop_blend_enabled',True) and
            not config.raw.get('direct_drop_continuous_carry',False)):
        blended = blend_checked_path(planner,combined,width,[(connecting.duration,'outgoing')],cancelled=abort)
        # The local join can round the start of the constant-pose leg, but
        # orientation must already be settled throughout the final half of
        # the configured bin radius, not merely at its endpoint.
        angle = check_final_orientation(planner,blended,width,destination,radius/2.,cancelled=abort)
        combined = blended
    else:
        angle = check_final_orientation(planner,combined,width,destination,radius/2.,cancelled=abort)
    return replace(combined,planning_s=time.monotonic()-began,diagnostics={**combined.diagnostics,
        'drop_orientation_schedule':{'applied':True,'pre_drop_tcp_base_m':pre.tolist(),
            'terminal_start_s':connecting.duration*combined.duration/(connecting.duration+terminal.duration),
            'time_base':'direct_drop_start',
            'final_fixed_orientation_radius_m':radius/2.,'max_final_orientation_error_rad':angle,
            'terminal_ik_count':count,'taught_endpoint_joints_preserved':True}})


def check_final_orientation(planner,path,width,destination,radius,*,cancelled):
    maximum = 0.
    for t in geometry_check_times(path.duration,planner.config.raw.get('geometry_check_hz',25.)):
        if cancelled():
            raise PickRejected('投放姿态复核已取消或超出预算')
        planner.model.set_state(path.position(t),width)
        tcp = planner.model.tcp()
        if np.linalg.norm(tcp[:2,3]-destination[:2,3]) <= radius:
            angle = Rotation.from_matrix(tcp[:3,:3].T@destination[:3,:3]).magnitude()
            maximum = max(maximum,float(angle))
            if angle > planner.model.ik.config.orientation_tolerance:
                raise PickRejected('进入投放末段后姿态仍未完成',details={'time_s':float(t),
                    'orientation_error_rad':float(angle),'final_region_radius_m':radius})
    return maximum

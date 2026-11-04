"""Simulation-only early visual orientation planning, never a hardware sender."""
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

from experiments.graspgenx_sim.obb_grasp import obb_top_down_grasp
from experiments.two_second_grasp.placement import plan_leg
from experiments.two_second_grasp.smooth_execution import check_curve_geometry
from nero_core.two_second.motion_profile import select_motion_profile
from nero_core.two_second.trajectory import retime_samples
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from experiments.two_second_grasp.early_observation_quality import nearly_square


def candidate_orientations(cloud, start_quat):
    """Prefer the visible short OBB axis, then the closest symmetric grasp."""
    current=Rotation.from_quat(np.asarray(start_quat)[[1,2,3,0]])
    candidates=[];square=None
    for axis in ('short','long'):
        grasp=obb_top_down_grasp(cloud,closing_axis=axis)
        if square is None:square=nearly_square(grasp['box'])
        for flip in (0.,np.pi):
            rot=Rotation.from_matrix(grasp['tcp'][:3,:3]) * Rotation.from_euler('z',flip)
            candidates.append({'axis':axis,'flip_rad':flip,
                'quaternion_wxyz':rot.as_quat()[[3,0,1,2]],
                'rotation_angle_rad':float((current.inv()*rot).magnitude())})
    if square:
        # A nearly square visible OBB cannot reliably identify a unique axis.
        # Try keeping the current tool direction; it is still physically tested.
        reference=Rotation.from_quat(candidates[0]['quaternion_wxyz'][[1,2,3,0]])
        if np.allclose(current.apply([0.,0.,1.]),reference.apply([0.,0.,1.]),atol=1e-6):
            candidates.append({'axis':'hold_current','flip_rad':0.,
                'quaternion_wxyz':np.asarray(start_quat,float).copy(),'rotation_angle_rad':0.})
        return sorted(candidates,key=lambda c:(c['rotation_angle_rad'],c['axis']!='short'))
    return sorted(candidates,key=lambda c:(c['axis']!='short',c['rotation_angle_rad']))


def final_orientation_candidates(cloud, early_quat, early_square):
    """Order new-frame candidates by proximity to the chosen early direction.

    For two near-square visible silhouettes, try keeping the earlier direction
    as an extra candidate at the new OBB center. A square silhouette does not
    prove physical symmetry; every candidate still requires full planning and
    the simulation's normal physical grasp verification.
    """
    q=np.asarray(early_quat,float);early=Rotation.from_quat(q[[1,2,3,0]])
    result=[];default=obb_top_down_grasp(cloud,closing_axis='short')
    if early_square and nearly_square(default['box']):
        pose=default['tcp'].copy();pose[:3,:3]=early.as_matrix()
        result.append({'pose':pose,'source':'stable_early_direction','rotation_difference_rad':0.})
    for axis in ('short','long'):
        pose=obb_top_down_grasp(cloud,closing_axis=axis)['tcp']
        for flip in (0.,np.pi):
            candidate=pose.copy();candidate[:3,:3]=pose[:3,:3]@Rotation.from_euler('z',flip).as_matrix()
            difference=(early.inv()*Rotation.from_matrix(candidate[:3,:3])).magnitude()
            result.append({'pose':candidate,'source':axis,'flip_rad':flip,'rotation_difference_rad':float(difference)})
    return sorted(result,key=lambda row:row['rotation_difference_rad'])


def plan_early_orientation(env, qpos, q, start_pose, cloud, dt, requested_s,
                           conditions, cancelled, *, wall_budget_s=4.):
    """Try early OBB orientations, retime and check an actual joint curve.

    The Cartesian center is held at the observed start. Time parameterization
    may lengthen the requested rotation to honor the existing joint envelope.
    """
    started=time.perf_counter();attempts=[]
    qpos=np.asarray(qpos,float).copy();qpos[env.arm_qpos_adrs]=q
    stop=lambda:cancelled() or time.perf_counter()-started>=wall_budget_s
    for candidate in candidate_orientations(cloud,start_pose[3:]):
        if stop():break
        goal=np.r_[start_pose[:3],candidate['quaternion_wxyz']]
        proposed=plan_leg(env,qpos,q,start_pose,goal,requested_s,max(dt,.05),stop,smooth=True)
        row={'axis':candidate['axis'],'flip_rad':candidate['flip_rad'],
             'rotation_angle_rad':candidate['rotation_angle_rad'],'error':proposed.get('error')}
        if 'error' in proposed:
            attempts.append(row);continue
        try:
            limits=select_motion_profile(conditions.config,payload=False)
            path=retime_samples(proposed['times'],proposed['samples'],limits,
                env.robot_controller.command_joint_lower_limits,env.robot_controller.command_joint_upper_limits,
                identity=(0,'early_orientation'),version=1,stage='early_orientation')
            data=mujoco.MjData(env.model)
            interpolation=Slerp([0.,1.],Rotation.from_quat(np.array([start_pose[3:],goal[3:]])[:,[1,2,3,0]]))
            max_pos=max_angle=0.
            for stamp in np.linspace(0.,path.duration,max(2,int(np.ceil(path.duration*50))+1)):
                if stop():break
                data.qpos[:]=qpos;data.qpos[env.arm_qpos_adrs]=path.sample(stamp)[0]
                mujoco.mj_forward(env.model,data)
                u=stamp/path.duration;u=10*u**3-15*u**4+6*u**5
                desired=interpolation([u])[0]
                actual=Rotation.from_matrix(data.site_xmat[env.ee_site_id].reshape(3,3))
                max_pos=max(max_pos,float(np.linalg.norm(data.site_xpos[env.ee_site_id]-start_pose[:3])))
                max_angle=max(max_angle,float((desired.inv()*actual).magnitude()))
            if stop() or max_pos>conditions.mirror.ik.config.position_tolerance or max_angle>conditions.mirror.ik.config.orientation_tolerance:
                row.update(error='early_rotation_tcp_reference_mismatch',max_position_error_m=max_pos,max_orientation_error_rad=max_angle)
                attempts.append(row);continue
            geometry=check_curve_geometry(env,path,qpos,stop)
            if 'error' in geometry:
                row['error']=geometry['error'];attempts.append(row);continue
        except PickRejected as error:
            row.update(error=str(error),details=error.details);attempts.append(row);continue
        row['error']=None;attempts.append(row)
        return {'path':path,'duration_s':float(path.duration),
                'requested_duration_s':float(requested_s),'candidate':candidate,
                'geometry':geometry,'attempts':attempts,'planning_wall_s':time.perf_counter()-started}
    return {'error':'early_orientation_no_certified_path','attempts':attempts,
            'planning_wall_s':time.perf_counter()-started,'budget_expired':stop()}

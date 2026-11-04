"""Smooth-reference planning and feedback gates for MuJoCo only.

No SDK imports, live-state writes, pose teleportation or grasp assistance.
The emitted spline is the spline checked for q/v/a and sampled geometry.
"""
from dataclasses import replace
from types import SimpleNamespace
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nero_core.two_second.trajectory import certify_curve, extrema, sampling_times, retime_samples, geometry_check_times
from nero_core.two_second.motion_profile import select_motion_profile
from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselinePolicyConfig
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def _rejected_state_details(env, positions, data):
    """Explain the existing state gate without changing its decision or order."""
    ranges = env.model.jnt_range[env.arm_joint_ids]
    margins = np.minimum(positions-ranges[:, 0], ranges[:, 1]-positions)
    required = float(env.robot_controller.config.joint_command_margin)
    if np.min(margins) < required-1e-9:
        index = int(np.argmin(margins))
        return {'reason': 'joint_margin', 'joint_number': index+1,
                'margin_rad': float(margins[index]), 'required_rad': required}
    height = float(data.site_xpos[env.ee_site_id, 2])
    if height < float(env.belt_height)-1e-9:
        return {'reason': 'tcp_below_belt', 'tcp_world_z_m': height,
                'belt_world_z_m': float(env.belt_height)}
    axis = data.site_xmat[env.ee_site_id].reshape(3, 3) @ np.array([0., 0., 1.])
    if -float(axis[2]) < -1e-9:
        return {'reason': 'approach_axis_not_downward', 'world_axis_z': float(axis[2])}
    for contact in data.contact[:data.ncon]:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(env.model.geom_bodyid[g1]), int(env.model.geom_bodyid[g2])
        robot1, robot2 = b1 in env._robot_collision_body_ids, b2 in env._robot_collision_body_ids
        if not (robot1 or robot2):
            continue
        other = b2 if robot1 else b1
        other_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, other) or ''
        if other_name.startswith('object_'):
            continue
        if float(contact.dist) <= 1e-6:
            return {'reason': 'contact', 'geom1': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                    'geom2': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                    'other_body': other_name, 'distance_m': float(contact.dist)}
    return {'reason': 'unclassified_state_rejection'}


def plan_smooth(env, ref, q, qpos, dt, width, cancelled, *, pose_solver=None, moving_target=None, knot_hz_override=None):
    started = time.perf_counter()
    data = mujoco.MjData(env.model)
    data.qpos[:] = qpos
    # Tight numerical convergence prevents discretization noise from becoming
    # fictitious acceleration. This does not relax any geometric/physical gate.
    config = replace(env.ik.config, position_tolerance=1e-7,
                     orientation_tolerance=1e-6, joint_centering_gain=0., max_iterations=8)
    ik = DampedLeastSquaresIK(env.model, data, [f'joint{i}' for i in range(1, 8)],
                             'ee_site', config)
    lower = np.asarray(env.robot_controller.command_joint_lower_limits)
    upper = np.asarray(env.robot_controller.command_joint_upper_limits)
    ik.lower_limits, ik.upper_limits = lower.copy(), upper.copy()
    knot_dt = dt if pose_solver is None else 1./(knot_hz_override or pose_solver.config.raw['planning_knot_hz'])
    times = sampling_times(ref.full_duration_s, knot_dt, getattr(ref,'rotate_first_s',0.), ref.prealign_s, ref.duration_s,
                           ref.duration_s+ref.follow_s)
    joints = [q.copy()]
    for stamp in times[1:]:
        if cancelled():
            return {'error': 'planning_cancelled'}
        xyz, quat = ref.sample(stamp)
        seed = joints[-1]
        if pose_solver is not None:
            solved = pose_solver.solve_world(xyz, quat, seed, cancelled=cancelled)
            if not solved.success:
                return {'error': 'deployed_ik_failed', 'time_s': float(stamp),
                    'position_error_m': solved.position_error, 'rotation_error_rad': solved.orientation_error}
            joints.append(solved.joint_positions.copy())
            continue
        solved = ik.solve(xyz, quat, qpos_seed=seed, cancelled=cancelled)
        if solved.success:
            joints.append(solved.joint_positions.copy())
            continue
        target = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        def residual(value):
            data.qpos[:] = qpos
            data.qpos[env.arm_qpos_adrs] = value
            mujoco.mj_kinematics(env.model, data)
            actual = data.site_xmat[env.ee_site_id].reshape(3, 3)
            return 1000*np.r_[data.site_xpos[env.ee_site_id]-xyz,
                         config.orientation_weight*Rotation.from_matrix(target@actual.T).as_rotvec()]
        fit = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                            method='dogbox', max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-10)
        error = residual(fit.x)/1000
        if (np.linalg.norm(error[:3]) > config.position_tolerance or
                np.linalg.norm(error[3:])/config.orientation_weight > config.orientation_tolerance):
            return {'error': 'smooth_ik_failed', 'time_s': float(stamp),
                    'position_error_m': float(np.linalg.norm(error[:3])),
                    'planning_wall_s': time.perf_counter()-started}
        joints.append(fit.x.copy())
    limits = SimpleNamespace(velocity_limits=env.robot_controller.max_joint_velocity,
                             acceleration_limits=env.robot_controller.max_joint_acceleration)
    try:
        path = certify_curve(times, joints, limits, lower, upper,
                             identity=(0, ref.capture_s), version=1, stage='smooth_sim_intercept',
                             zero_end_acceleration=True)
    except PickRejected as failure:
        return {'error': 'smooth_joint_curve_rejected', 'details': failure.details,
                'planning_wall_s': time.perf_counter()-started}
    ik_wall = time.perf_counter()-started
    previous = q.copy()
    previous_time = 0.
    checks = 0
    max_position_error = 0.
    max_orientation_error = 0.
    geometry_hz = 2./dt if pose_solver is None else pose_solver.config.raw.get('geometry_check_hz',2./dt)
    for stamp in geometry_check_times(path.duration,geometry_hz,getattr(ref,'rotate_first_s',0.),ref.prealign_s,ref.duration_s,ref.follow_end_s):
        if cancelled():
            return {'error': 'planning_cancelled'}
        desired = path.sample(stamp)[0]
        count = max(1, int(np.ceil(np.max(abs(desired-previous))/.008)))
        # The previous endpoint was checked on the previous iteration of this
        # immutable snapshot. Still check t=0 once and every intermediate edge.
        for u in np.linspace(0., 1., count+1)[1:]:
            check = previous+(desired-previous)*u
            data.qpos[:] = qpos
            data.qpos[env.arm_qpos_adrs] = check
            mujoco.mj_forward(env.model, data)
            checks += 1
            if not env._arm_state_data_is_safe(check, data, approach_axis_local=np.array([0., 0., 1.]),
                    minimum_downward_alignment=0., minimum_tcp_height=env.belt_height,
                    minimum_joint_margin=env.robot_controller.config.joint_command_margin,
                    maximum_jacobian_condition=None):
                return {'error': 'smooth_robot_state_rejected', 'details': _rejected_state_details(env, check, data), 'time_s': float(stamp),
                        'planning_wall_s': time.perf_counter()-started}
            elapsed=float(previous_time+(stamp-previous_time)*u)
            # Production switches to attached-payload checks at follow end;
            # this simulator keeps its separately verified physical brake.
            if moving_target is not None and elapsed < ref.follow_end_s:
                hit=moving_target.check(check,ref,elapsed)
                if hit is not None:
                    return {'error':'predicted_moving_target_cloud_collision','details':hit,
                            'time_s':elapsed,'planning_wall_s':time.perf_counter()-started}
        previous = desired
        previous_time = float(stamp)
        if pose_solver is not None:
            # Match the hardware's dense FK-versus-reference gate, in the
            # private model's base frame. This is not device feedback.
            model=pose_solver.mirror
            if moving_target is None or stamp >= ref.follow_end_s:
                model.set_state(desired,width)
            actual=model.tcp();xyz,quat=ref.sample(stamp)
            world_to_base=env.nero_world_to_base
            expected_xyz=world_to_base[:3,:3]@xyz+world_to_base[:3,3]
            expected_rotation=world_to_base[:3,:3]@Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            angle=Rotation.from_matrix(expected_rotation.T@actual[:3,:3]).magnitude()
            position_error=float(np.linalg.norm(actual[:3,3]-expected_xyz))
            max_position_error=max(max_position_error,position_error)
            max_orientation_error=max(max_orientation_error,float(angle))
            if (position_error>model.ik.config.position_tolerance or
                    angle>model.ik.config.orientation_tolerance):
                return {'error':'sampled_fk_reference_mismatch','time_s':float(stamp),
                        'position_error_m':float(np.linalg.norm(actual[:3,3]-expected_xyz)),
                        'orientation_error_rad':float(angle),'planning_wall_s':time.perf_counter()-started}
    return {'samples': np.asarray(joints), 'stamps': times, 'path': path,
            'ik_solve_count': len(times)-1, 'ik_knot_hz': 1./knot_dt,
            'reference_check': None if pose_solver is None else {
                'configured_frequency_hz':geometry_hz,
                'max_position_error_m':max_position_error,'max_orientation_error_rad':max_orientation_error,
                'position_tolerance_m':pose_solver.mirror.ik.config.position_tolerance,
                'orientation_tolerance_rad':pose_solver.mirror.ik.config.orientation_tolerance},
            'planning_wall_s': time.perf_counter()-started, 'ik_and_qva_wall_s': ik_wall,
            'geometry_wall_s': time.perf_counter()-started-ik_wall, 'geometry_states': checks,
            'geometry_scope': ('sampled_robot_static_and_moving_visual_cloud; payload_brake_checked_after_physical_grip'
                if moving_target is not None else 'sampled_robot_static; payload_brake_checked_after_physical_grip'),
            'max_joint_speed_rad_s': float(np.max(abs(np.asarray(extrema(path.curve, 1))))),
            'peak_joint_acceleration_rad_s2': np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0)}


def closure_gate(ref, elapsed, pose, previous_pose, previous_s, now, *, command_elapsed=None):
    """Same measured-time closure policy as hardware; no contact truth enters."""
    from nero_core.two_second.feedback_gates import closure_feedback
    def transform(value):
        if value is None:
            return None
        tcp = np.eye(4)
        tcp[:3, 3] = value[:3]
        tcp[:3, :3] = Rotation.from_quat(value[3:][[1, 2, 3, 0]]).as_matrix()
        return tcp
    return closure_feedback(ref, transform(pose), now, transform(previous_pose), previous_s,
        command_elapsed=elapsed if command_elapsed is None else command_elapsed)


def check_loaded_brake(env, path, ref, runtime, qpos, cancelled):
    """Check actual grasp attachment on the future brake, on private MjData.

    Initial conveyor support is allowed during horizontal braking, as during
    the existing lift's first 2 mm. No other payload/environment contact is
    allowed. This is sampled geometry, not a guarantee against grasp slip.
    """
    return check_curve_geometry(env, path, qpos, cancelled, runtime=runtime,
        start_s=ref.duration_s+ref.follow_s, allow_initial_belt_support=True)


def check_curve_geometry(env, path, qpos, cancelled, *, runtime=None, start_s=0.,
                         allow_initial_belt_support=False):
    begun = time.perf_counter()
    data = mujoco.MjData(env.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(env.model, data)
    tcp = data.site_xpos[env.ee_site_id].copy()
    rot = data.site_xmat[env.ee_site_id].reshape(3, 3).copy()
    if runtime is not None:
        obj = qpos[runtime.qpos_adr:runtime.qpos_adr+7].copy()
        offset = rot.T@(obj[:3]-tcp)
        relative = rot.T@Rotation.from_quat(obj[3:][[1, 2, 3, 0]]).as_matrix()
    objects = set() if runtime is None else set(map(int, runtime.geom_ids))
    fingers = set(map(int, env.finger_geom_ids))
    others = {int(g) for o in env._objects if o.logical_id is not None and o is not runtime for g in o.geom_ids}
    belt = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'conveyor_surface')
    support = allow_initial_belt_support and any((int(c.geom1) in objects and int(c.geom2) == belt or
                   int(c.geom2) in objects and int(c.geom1) == belt) and c.dist <= 1e-6
                  for c in data.contact[:data.ncon])
    count = 0
    knots = path.curve.x
    stamps = np.unique(np.r_[start_s, knots[knots >= start_s],
                            ((knots[:-1]+knots[1:])/2)[(knots[:-1]+knots[1:])/2 >= start_s]])
    # Include the measured-feedback connection to the future brake contact.
    previous = np.asarray(qpos[env.arm_qpos_adrs]).copy()
    for stamp in stamps:
        if cancelled():
            return {'error': 'brake_check_cancelled'}
        desired = path.sample(stamp)[0]
        steps = max(1, int(np.ceil(np.max(abs(desired-previous))/.008)))
        for u in np.linspace(0., 1., steps+1):
            q = previous+(desired-previous)*u
            data.qpos[:] = qpos
            data.qpos[env.arm_qpos_adrs] = q
            mujoco.mj_forward(env.model, data)
            if runtime is not None:
                rot = data.site_xmat[env.ee_site_id].reshape(3, 3).copy()
                xyz = data.site_xpos[env.ee_site_id]+rot@offset
                data.qpos[runtime.qpos_adr:runtime.qpos_adr+7] = np.r_[xyz,
                    Rotation.from_matrix(rot@relative).as_quat()[[3, 0, 1, 2]]]
                mujoco.mj_forward(env.model, data)
            count += 1
            if not env._arm_state_data_is_safe(q, data, approach_axis_local=np.array([0., 0., 1.]),
                    minimum_downward_alignment=0., minimum_tcp_height=env.belt_height,
                    minimum_joint_margin=env.robot_controller.config.joint_command_margin,
                    maximum_jacobian_condition=None):
                return {'error': 'smooth_leg_robot_static_collision', 'time_s': float(stamp)}
            for c in data.contact[:data.ncon]:
                a, b = int(c.geom1), int(c.geom2)
                if c.dist > 1e-6:
                    continue
                if (a in others) != (b in others):
                    other = b if a in others else a
                    if int(env.model.geom_bodyid[other]) in env._robot_collision_body_ids:
                        return {'error': 'smooth_leg_robot_other_object_collision', 'time_s': float(stamp)}
                if (a in objects) == (b in objects):
                    continue
                other = b if a in objects else a
                if other in fingers:
                    continue
                minimum_support_rise = -.002 if start_s > 0 else -.00001
                if other == belt and support and minimum_support_rise <= xyz[2]-obj[2] <= .002:
                    continue
                return {'error': 'smooth_payload_collision', 'other_geom': env.model.geom(other).name,
                        'time_s': float(stamp), 'wall_s': time.perf_counter()-begun}
        previous = desired
    return {'states': count, 'wall_s': time.perf_counter()-begun,
            'attachment_source': 'actual_mujoco_grasp_feedback_snapshot',
            'initial_belt_support_allowed': support}


def plan_smooth_leg(env, kind, qpos, q, goal, runtime, dt, cancelled, *, pose_solver=None, acceleration_override=None):
    """Retime the existing lift/RRT route; certify and execute one spline."""
    from experiments.two_second_grasp.placement import plan_leg, plan_joint_route, plan_return
    begun = time.perf_counter()
    qpos = qpos.copy()
    qpos[env.arm_qpos_adrs] = q
    if kind == 'lift':
        data = mujoco.MjData(env.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(env.model, data)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[env.ee_site_id])
        pose = np.r_[data.site_xpos[env.ee_site_id], quat]
        planned = plan_leg(env, qpos, q, pose, goal, 2., dt, cancelled, runtime=runtime, smooth=True)
    elif kind == 'transfer':
        planned = plan_joint_route(env, qpos, q, goal, runtime, dt, cancelled, merge_collinear=True,
                                   pose_solver=pose_solver)
    elif kind == 'return':
        planned = plan_return(env, qpos, q, goal, dt, cancelled, merge_collinear=True)
    else:
        raise ValueError('unknown smooth leg')
    if 'error' in planned:
        return planned
    limits = SimpleNamespace(velocity_limits=env.robot_controller.max_joint_velocity,
                             acceleration_limits=env.robot_controller.max_joint_acceleration,
                             settings=SimpleNamespace(control_hz=1./dt))
    profile_diagnostics = {}
    if pose_solver is not None:
        limits = select_motion_profile(pose_solver.config, payload=kind in ('lift', 'transfer'))
        if acceleration_override is not None:
            requested=float(acceleration_override)
            if kind != 'return' or not np.isfinite(requested) or requested <= 0:
                raise ValueError('only a positive simulation return acceleration override is supported')
            limits=replace(limits,acceleration_limits=np.minimum(limits.acceleration_limits,requested),
                           source='simulation_return_acceleration_experiment')
        profile_diagnostics = limits.diagnostics()
    times, samples = planned['times'], planned['samples']
    path = retime_samples(times, samples, limits,
        env.robot_controller.command_joint_lower_limits, env.robot_controller.command_joint_upper_limits,
        identity=(0, kind), version=1, stage=kind)
    scale = path.diagnostics['time_scale']
    geometry = check_curve_geometry(env, path, qpos, cancelled, runtime=runtime,
                                    allow_initial_belt_support=kind == 'lift')
    if 'error' in geometry:
        return geometry
    planned.update(path=path, times=times*scale, duration_s=path.duration,
        time_scale=scale, emitted_curve_geometry=geometry,
        **profile_diagnostics,
        peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0),
        total_planning_wall_s=time.perf_counter()-begun)
    return planned


def plan_transfer_candidates(env, qpos, q, goals, runtime, dt, cancelled, *, pose_solver=None,
                             maximum_candidates=32, maximum_wall_s=25.):
    """Try other already-free bin poses when a candidate has no certified route."""
    begun=time.perf_counter();attempts=[]
    for index,goal in enumerate(goals[:maximum_candidates]):
        if cancelled() or time.perf_counter()-begun >= maximum_wall_s:
            break
        def stop():return cancelled() or time.perf_counter()-begun >= maximum_wall_s
        result=plan_smooth_leg(env,'transfer',qpos,q,np.asarray(goal,float),runtime,dt,stop,pose_solver=pose_solver)
        attempts.append({'candidate_index':index,'goal_pose':list(goal),'error':result.get('error'),
                         'goal_ik_count':result.get('goal_ik_count'),'timed_out':result.get('timed_out')})
        if 'error' not in result and not stop():
            result.update(bin_candidate_index=index,bin_candidate_attempts=attempts,
                          all_bin_candidates_wall_s=time.perf_counter()-begun)
            return result
    return {'error':'all_free_bin_routes_failed','bin_candidate_attempts':attempts,
            'all_bin_candidates_wall_s':time.perf_counter()-begun,
            'cancelled_or_deadline':bool(cancelled() or time.perf_counter()-begun >= maximum_wall_s)}

"""MuJoCo-only sampled lift and transfer planning for the independent controller."""
import time
from dataclasses import replace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares
from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK


from nero_core.two_second.trajectory import merge_collinear_waypoints


def plan_leg(env, start_qpos, start_q, start_pose, goal_pose, duration, dt, cancelled, *, runtime=None, smooth=False):
    """Plan from an actual feedback snapshot; never mutate live MuJoCo data."""
    started=time.perf_counter(); data=mujoco.MjData(env.model);data.qpos[:]=start_qpos
    ik_config=(replace(env.ik.config,position_tolerance=1e-7,orientation_tolerance=1e-6,
                       joint_centering_gain=0.,max_iterations=8) if smooth else env.ik.config)
    ik=DampedLeastSquaresIK(env.model,data,[f'joint{i}' for i in range(1,8)],'ee_site',ik_config)
    ik.lower_limits=np.asarray(env.robot_controller.command_joint_lower_limits).copy()
    ik.upper_limits=np.asarray(env.robot_controller.command_joint_upper_limits).copy()
    start=np.asarray(start_pose,float);goal=np.asarray(goal_pose,float)
    if start.shape!=(7,) or goal.shape!=(7,) or not np.isfinite(np.r_[start,goal,duration,dt]).all() or duration<=0 or dt<=0:
        raise ValueError('invalid placement leg')
    n=int(np.ceil(duration/dt));times=np.linspace(0,duration,n+1)
    slerp=Slerp([0.,1.],Rotation.from_quat(np.array([start[3:],goal[3:]])[:,[1,2,3,0]]))
    seed=np.asarray(start_q,float).copy();samples=[]
    # The simulation-only held body is moved rigidly with the TCP for every
    # checked state. Its initial belt support is permitted only while rising
    # from the original contact; all other non-finger contacts are rejected.
    if runtime is not None:
        mujoco.mj_forward(env.model,data)
        tcp0=data.site_xpos[env.ee_site_id].copy();rot0=data.site_xmat[env.ee_site_id].reshape(3,3).copy()
        object0=np.asarray(start_qpos[runtime.qpos_adr:runtime.qpos_adr+7]).copy()
        relative_pos=rot0.T@(object0[:3]-tcp0)
        relative_rot=rot0.T@Rotation.from_quat(object0[3:][[1,2,3,0]]).as_matrix()
        object_geoms=set(map(int,runtime.geom_ids));finger_geoms=set(map(int,env.finger_geom_ids))
        belt=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'conveyor_surface')
        support_at_start=any(((int(c.geom1) in object_geoms and int(c.geom2)==belt) or (int(c.geom2) in object_geoms and int(c.geom1)==belt)) and c.dist<=1e-6 for c in data.contact[:data.ncon])
        other_geoms={int(g) for obj in env._objects if obj.logical_id is not None and obj is not runtime for g in obj.geom_ids}
    for stamp in times:
        if cancelled():return {'error':'cancelled'}
        u=stamp/duration;u=(10*u**3-15*u**4+6*u**5) if smooth else u*u*(3.-2.*u)
        xyz=start[:3]*(1.-u)+goal[:3]*u;quat=slerp([u]).as_quat()[0][[3,0,1,2]]
        solved=ik.solve(xyz,quat,qpos_seed=seed,cancelled=cancelled)
        if solved.success:
            q=np.asarray(solved.joint_positions).copy()
        else:
            # Same pose and command bounds, with the project's bounded IK
            # fallback; never silently widen limits or skip a waypoint.
            target_rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            def residual(q):
                data.qpos[:]=start_qpos;data.qpos[env.arm_qpos_adrs]=q;mujoco.mj_kinematics(env.model,data)
                actual_rotation=data.site_xmat[env.ee_site_id].reshape(3,3)
                return (1000. if smooth else 1.)*np.r_[data.site_xpos[env.ee_site_id]-xyz, ik.config.orientation_weight*Rotation.from_matrix(target_rotation@actual_rotation.T).as_rotvec()]
            fit=least_squares(residual,np.clip(seed,ik.lower_limits,ik.upper_limits),bounds=(ik.lower_limits,ik.upper_limits),method='dogbox' if smooth else 'trf',max_nfev=150,ftol=1e-12 if smooth else 1e-10,xtol=1e-12 if smooth else 1e-10,gtol=1e-10)
            err=residual(fit.x)/(1000. if smooth else 1.);pos_err=float(np.linalg.norm(err[:3]));rot_err=float(np.linalg.norm(err[3:])/ik.config.orientation_weight)
            if pos_err>ik.config.position_tolerance or rot_err>ik.config.orientation_tolerance:
                return {'error':'ik_failed','time_s':float(stamp),'position_error_m':pos_err,'orientation_error_rad':rot_err,'wall_s':time.perf_counter()-started}
            q=fit.x.copy()
        steps=max(1,int(np.ceil(np.max(abs(q-seed))/.008)))
        for fraction in np.linspace(0,1,steps+1):
            check=seed+(q-seed)*fraction;data.qpos[:]=start_qpos;data.qpos[env.arm_qpos_adrs]=check;mujoco.mj_forward(env.model,data)
            if runtime is not None:
                tcp=data.site_xpos[env.ee_site_id].copy();rot=data.site_xmat[env.ee_site_id].reshape(3,3).copy()
                object_pos=tcp+rot@relative_pos
                data.qpos[runtime.qpos_adr:runtime.qpos_adr+7]=np.r_[object_pos,Rotation.from_matrix(rot@relative_rot).as_quat()[[3,0,1,2]]]
                mujoco.mj_forward(env.model,data)
                for contact in data.contact[:data.ncon]:
                    a,b=int(contact.geom1),int(contact.geom2)
                    if contact.dist>1e-6:continue
                    if (a in object_geoms)!=(b in object_geoms):
                        other=b if a in object_geoms else a
                        if other in finger_geoms:continue
                        if other==belt and support_at_start and -1e-5<=object_pos[2]-object0[2]<=.002:continue
                        return {'error':'lift_loaded_collision','other_geom':env.model.geom(other).name,'time_s':float(stamp),'wall_s':time.perf_counter()-started}
                    if (a in other_geoms)!=(b in other_geoms):
                        other=b if a in other_geoms else a
                        if int(env.model.geom_bodyid[other]) in env._robot_collision_body_ids:
                            return {'error':'lift_robot_object_collision','time_s':float(stamp),'wall_s':time.perf_counter()-started}
            if not env._arm_state_data_is_safe(check,data,approach_axis_local=np.array([0.,0.,1.]),minimum_downward_alignment=0.,minimum_tcp_height=env.belt_height,minimum_joint_margin=env.robot_controller.config.joint_command_margin,maximum_jacobian_condition=None):
                return {'error':'robot_static_collision_or_joint_limit','time_s':float(stamp),'wall_s':time.perf_counter()-started}
        samples.append(q);seed=q
    samples=np.asarray(samples);v=np.abs(np.diff(samples,axis=0)/(duration/n));limits=np.asarray(env.robot_controller.max_joint_velocity)
    if np.any(v>limits+1e-6):return {'error':'joint_velocity_exceeded','wall_s':time.perf_counter()-started}
    return {'samples':samples,'times':times,'duration_s':duration,'wall_s':time.perf_counter()-started,'goal_pose':goal.tolist(),'max_joint_speed_rad_s':float(v.max()),'loaded_collision_check':runtime is not None}


def evaluate_placement(env, runtime, bin_id):
    """Independent physical containment/support check; simulator state only."""
    from experiments.graspgenx_sim.physical_containment import CollisionGeometryBounds, catalog_bounds, containment_checks
    box=next(b for b in env._bins if int(b['id'])==int(bin_id))
    bin_geoms=tuple(i for i in range(env.model.ngeom) if (env.model.geom(i).name or '').startswith(f'bin_{bin_id}_'))
    spec=next(s for s in env.object_specs if int(s.category_id)==int(runtime.category_id))
    bounds=getattr(env,'_placement_bounds_cache',None)
    if bounds is None or bounds.model is not env.model:
        bounds=CollisionGeometryBounds(env.model);env._placement_bounds_cache=bounds
    pos=env.data.xpos[runtime.body_id].copy();rot=env.data.xmat[runtime.body_id].reshape(3,3)
    obj=bounds.object_bounds(env.data,runtime.geom_ids,catalog_bounds(pos,rot,spec.dimensions));interior=bounds.bin_interior(env.data,box,bin_geoms)
    checks=containment_checks(obj,interior);pairs=env._contact_pairs()
    # A second part can rest on a first part already supported by the bin.
    # Only contained, slow, finger-free parts may bridge the support graph.
    from .support_graph import support_chain
    eligible={runtime.logical_id:runtime.geom_ids}
    for other in env._objects:
        if other is runtime or other.logical_id is None:
            continue
        other_spec=next(s for s in env.object_specs if int(s.category_id)==int(other.category_id))
        other_pos=env.data.xpos[other.body_id].copy();other_rot=env.data.xmat[other.body_id].reshape(3,3)
        other_bounds=bounds.object_bounds(env.data,other.geom_ids,catalog_bounds(other_pos,other_rot,other_spec.dimensions))
        contained=all(containment_checks(other_bounds,interior).values())
        touching=any((min(f,g),max(f,g)) in pairs for f in env.finger_geom_ids for g in other.geom_ids)
        velocity=env.data.qvel[other.qvel_adr:other.qvel_adr+6]
        if contained and not touching and np.linalg.norm(velocity[:3])<=.08 and np.linalg.norm(velocity[3:])<=1.:
            eligible[other.logical_id]=other.geom_ids
    support_pairs=set();force=np.zeros(6)
    for contact_index,contact in enumerate(env.data.contact[:env.data.ncon]):
        # MuJoCo's compliant contact margin can carry positive load with a
        # sub-mm geometric gap. Use actual solved normal force, not dist<=0.
        mujoco.mj_contactForce(env.model,env.data,contact_index,force)
        if force[0]>0.:
            support_pairs.add((int(contact.geom1),int(contact.geom2)))
    chain=support_chain(runtime.logical_id,bin_geoms,eligible,support_pairs)
    checks.update(supported_by_bin=chain is not None,
        support_chain=chain, support_source='direct_bin_contact' if chain is not None and len(chain)==2 else
            'contact_chain_to_bin' if chain is not None else 'unsupported',
        touching_finger=any((min(f,g),max(f,g)) in pairs for f in env.finger_geom_ids for g in runtime.geom_ids),
        linear_speed_m_s=float(np.linalg.norm(env.data.qvel[runtime.qvel_adr:runtime.qvel_adr+3])),
        angular_speed_rad_s=float(np.linalg.norm(env.data.qvel[runtime.qvel_adr+3:runtime.qvel_adr+6])),
        object_bounds=obj.diagnostics(),bin_interior=interior.diagnostics())
    return checks


def plan_joint_route(env, start_qpos, start_q, goal_pose, runtime, dt, cancelled, *, max_wall_s=8., seed=41, merge_collinear=False, pose_solver=None):
    """Search a sampled joint-space route with the simulated held object attached.

    The attachment is a snapshot of MuJoCo truth, for this simulation only.
    It is not a hardware pose estimate and cannot guarantee the real part will
    remain rigidly grasped. The controller must monitor actual grip feedback.
    """
    from mujoco_conveyor_sorting.controllers.safe_path_planner import ConstrainedRRTConnect, SafePathConfig
    begun=time.perf_counter();data=mujoco.MjData(env.model);data.qpos[:]=start_qpos;mujoco.mj_forward(env.model,data)
    lower=np.asarray(env.robot_controller.command_joint_lower_limits).copy();upper=np.asarray(env.robot_controller.command_joint_upper_limits).copy()
    ik=DampedLeastSquaresIK(env.model,data,[f'joint{i}' for i in range(1,8)],'ee_site',env.ik.config);ik.lower_limits=lower;ik.upper_limits=upper
    held=runtime is not None
    if held:
        object_indices=set(map(int,runtime.geom_ids));finger_indices=set(map(int,env.finger_geom_ids))
        tcp_pos=data.site_xpos[env.ee_site_id].copy();tcp_R=data.site_xmat[env.ee_site_id].reshape(3,3).copy()
        obj_pose=np.asarray(start_qpos[runtime.qpos_adr:runtime.qpos_adr+7]).copy()
        obj_R=Rotation.from_quat(obj_pose[3:][[1,2,3,0]]).as_matrix()
        relative_pos=tcp_R.T@(obj_pose[:3]-tcp_pos);relative_R=tcp_R.T@obj_R
    checks={'states':0,'robot_rejected':0,'loaded_rejected':0,'other_object_robot_rejected':0,'examples':[]}
    other_objects={int(g) for o in env._objects if o.logical_id is not None and o is not runtime for g in o.geom_ids}
    def abort():return cancelled() or time.perf_counter()-begun>=max_wall_s
    def valid(q):
        if abort():return False
        data.qpos[:]=start_qpos;data.qpos[env.arm_qpos_adrs]=q;mujoco.mj_forward(env.model,data)
        if held:
            tcp=data.site_xpos[env.ee_site_id].copy();rot=data.site_xmat[env.ee_site_id].reshape(3,3).copy()
            oq=Rotation.from_matrix(rot@relative_R).as_quat()[[3,0,1,2]]
            data.qpos[runtime.qpos_adr:runtime.qpos_adr+7]=np.r_[tcp+rot@relative_pos,oq]
            mujoco.mj_forward(env.model,data)
        checks['states']+=1
        if not env._arm_state_data_is_safe(q,data,approach_axis_local=np.array([0.,0.,1.]),minimum_downward_alignment=0.,minimum_tcp_height=env.belt_height,minimum_joint_margin=env.robot_controller.config.joint_command_margin,maximum_jacobian_condition=None):
            checks['robot_rejected']+=1;return False
        for contact in data.contact[:data.ncon]:
            a,b=int(contact.geom1),int(contact.geom2)
            if (a in other_objects)!=(b in other_objects):
                other=b if a in other_objects else a
                if int(env.model.geom_bodyid[other]) in env._robot_collision_body_ids and float(contact.dist)<=1e-6:
                    checks['other_object_robot_rejected']+=1;return False
        if held:
            for contact in data.contact[:data.ncon]:
                a,b=int(contact.geom1),int(contact.geom2)
                if (a in object_indices)==(b in object_indices):continue
                other=b if a in object_indices else a
                if other in finger_indices:continue
                if float(contact.dist)<=1e-6:
                    checks['loaded_rejected']+=1
                    if len(checks['examples'])<5:checks['examples'].append({'other_geom':env.model.geom(other).name,'distance_m':float(contact.dist)})
                    return False
        return True
    def edge(a,b):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.008)))
        return all(valid(a+(b-a)*(i/n)) for i in range(n+1))
    planner=ConstrainedRRTConnect(lower,upper,valid,config=SafePathConfig(normalized_step_size=.12,execution_joint_step=.08,max_iterations=1500,shortcut_attempts=40),seed=seed,cancelled=abort,segment_valid=edge)
    if not valid(start_q):return {'error':'initial_loaded_state_collision','checks':checks,'wall_s':time.perf_counter()-begun}
    goal=np.asarray(goal_pose,float).copy();rng=np.random.default_rng(seed);goals=[]
    # Change wrist yaw while preserving downwards tool orientation. The target
    # x/y is adjusted by the rigid part offset, so the OBJECT, not just TCP,
    # is over the configured bin. No category inference is introduced.
    base_rotation=Rotation.from_quat(goal[3:][[1,2,3,0]])
    for angle in ((0.,np.pi/2,-np.pi/2,np.pi) if pose_solver is None else (0.,)):
        goal_rotation=Rotation.from_euler('z',angle)*base_rotation
        p=goal[:3].copy()
        if held and pose_solver is None:p[:2]-=(goal_rotation.as_matrix()@relative_pos)[:2]
        qgoal=goal_rotation.as_quat()[[3,0,1,2]]
        seeds=[np.asarray(start_q)] + [rng.uniform(lower,upper) for _ in range(8)]
        for qseed in seeds:
            if abort():break
            data.qpos[:]=start_qpos
            solved=(ik.solve(p,qgoal,qpos_seed=qseed,cancelled=abort) if pose_solver is None else
                    pose_solver.solve_world(p,qgoal,qseed,cancelled=abort))
            if not solved.success:continue
            q=np.asarray(solved.joint_positions).copy()
            if any(np.max(np.abs(q-old))<.02 for old in goals):continue
            if not valid(q):continue
            goals.append(q)
            path=planner.plan(start_q,q)
            if path is None:continue
            original_waypoints=len(path)
            if merge_collinear:path=merge_collinear_waypoints(path)
            # Each edge is a smoothstep on exactly the checked joint chord.
            # Consecutive velocities are zero; per-joint velocity/acceleration
            # bounds determine duration. Recheck every emitted sample edge.
            velocity=np.asarray(env.robot_controller.max_joint_velocity);accel=np.asarray(env.robot_controller.max_joint_acceleration)
            samples=[np.asarray(path[0]).copy()];times=[0.]
            for a,b in zip(path,path[1:]):
                delta=np.abs(b-a);seconds=max(float(np.max(1.5*delta/velocity)),float(np.max(np.sqrt(6*delta/accel))),2*dt)
                n=int(np.ceil(seconds/dt));seconds=n*dt
                for i in range(1,n+1):
                    u=i/n;v=u*u*(3-2*u);samples.append(a+(b-a)*v);times.append(times[-1]+dt)
            if abort():return {'error':'route_search_cancelled','checks':checks,'wall_s':time.perf_counter()-begun}
            return {'samples':np.asarray(samples),'times':np.asarray(times),'duration_s':float(times[-1]),'wall_s':time.perf_counter()-begun,'goal_pose':np.r_[p,qgoal].tolist(),'waypoints':len(path),'original_waypoints':original_waypoints,'checks':checks,'method':'rrt_connect_sampled_robot_and_attached_object','attachment_source':'mujoco_state_simulation_only'}
    return {'error':'loaded_route_not_found','goal_ik_count':len(goals),'checks':checks,'wall_s':time.perf_counter()-begun,'timed_out':abort()}


def plan_return(env, start_qpos, start_q, target_q, dt, cancelled, *, max_wall_s=8., seed=72, merge_collinear=False):
    """Search an unloaded path back to the actual episode waiting joints."""
    from mujoco_conveyor_sorting.controllers.safe_path_planner import ConstrainedRRTConnect, SafePathConfig
    begun=time.perf_counter();data=mujoco.MjData(env.model)
    lower=np.asarray(env.robot_controller.command_joint_lower_limits).copy();upper=np.asarray(env.robot_controller.command_joint_upper_limits).copy()
    object_geoms={int(g) for obj in env._objects if obj.logical_id is not None for g in obj.geom_ids}
    checks={'states':0,'rejected':0}
    def abort():return cancelled() or time.perf_counter()-begun>=max_wall_s
    def valid(q):
        if abort():return False
        data.qpos[:]=start_qpos;data.qpos[env.arm_qpos_adrs]=q;mujoco.mj_forward(env.model,data);checks['states']+=1
        if not env._arm_state_data_is_safe(q,data,approach_axis_local=np.array([0.,0.,1.]),minimum_downward_alignment=0.,minimum_tcp_height=env.belt_height,minimum_joint_margin=env.robot_controller.config.joint_command_margin,maximum_jacobian_condition=None):checks['rejected']+=1;return False
        for contact in data.contact[:data.ncon]:
            a,b=int(contact.geom1),int(contact.geom2)
            if a in object_geoms or b in object_geoms:
                other=b if a in object_geoms else a
                if int(env.model.geom_bodyid[other]) in env._robot_collision_body_ids and float(contact.dist)<=1e-6:checks['rejected']+=1;return False
        return True
    def edge(a,b):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.008)))
        return all(valid(a+(b-a)*i/n) for i in range(n+1))
    planner=ConstrainedRRTConnect(lower,upper,valid,config=SafePathConfig(normalized_step_size=.12,execution_joint_step=.08,max_iterations=1500,shortcut_attempts=40),seed=seed,cancelled=abort,segment_valid=edge)
    path=planner.plan(np.asarray(start_q),np.asarray(target_q))
    if path is None:return {'error':'return_route_not_found','checks':checks,'wall_s':time.perf_counter()-begun,'timed_out':abort()}
    original_waypoints=len(path)
    if merge_collinear:path=merge_collinear_waypoints(path)
    vel=np.asarray(env.robot_controller.max_joint_velocity);acc=np.asarray(env.robot_controller.max_joint_acceleration)
    samples=[np.asarray(path[0]).copy()];times=[0.]
    for a,b in zip(path,path[1:]):
        d=np.abs(b-a);seconds=max(float(np.max(1.5*d/vel)),float(np.max(np.sqrt(6*d/acc))),2*dt);n=int(np.ceil(seconds/dt))
        for i in range(1,n+1):
            u=i/n;samples.append(a+(b-a)*(u*u*(3-2*u)));times.append(times[-1]+dt)
    return {'samples':np.asarray(samples),'times':np.asarray(times),'duration_s':float(times[-1]),'wall_s':time.perf_counter()-begun,'waypoints':len(path),'original_waypoints':original_waypoints,'checks':checks,'method':'rrt_connect_sampled_unloaded_return'}

"""Simulation-only free bin footprint selection; no change to interception."""
import numpy as np


def free_centers(lower, upper, size, occupied, preferred):
    """Return free footprint centers sorted by clearance and preference."""
    lower,upper,size,preferred=(np.asarray(x,float) for x in (lower,upper,size,preferred))
    rectangles=[(np.asarray(a,float),np.asarray(b,float)) for a,b in occupied]
    if (any(x.shape!=(2,) or not np.isfinite(x).all() for x in (lower,upper,size,preferred))
            or np.any(size<=0) or np.any(upper-lower<size)):
        raise ValueError('payload footprint cannot fit the bin')
    half=size/2;lo=lower+half;hi=upper-half
    axes=[]
    for axis in range(2):
        values=[lo[axis],hi[axis],np.clip(preferred[axis],lo[axis],hi[axis])]
        for a,b in rectangles:values.extend([a[axis]-half[axis],b[axis]+half[axis]])
        values=np.unique(np.clip(values,lo[axis],hi[axis]))
        axes.append(np.unique(np.r_[values,(values[:-1]+values[1:])/2]))
    candidates=[]
    for x in axes[0]:
        for y in axes[1]:
            center=np.array([x,y]);a=center-half;b=center+half
            margin=float(min(np.min(a-lower),np.min(upper-b)))
            blocked=False
            for c,d in rectangles:
                gaps=np.maximum(c-b,a-d)
                if np.all(gaps<0):blocked=True;break
                margin=min(margin,float(np.linalg.norm(np.maximum(gaps,0))))
            if not blocked:candidates.append((margin,-float(np.linalg.norm(center-preferred)),center))
    if not candidates:raise ValueError('no free footprint in the configured bin')
    candidates.sort(key=lambda row:row[:2],reverse=True)
    return [item[2] for item in candidates]


def free_center(lower, upper, size, occupied, preferred):
    """Return the maximum-clearance available footprint center."""
    return free_centers(lower,upper,size,occupied,preferred)[0]


def select_drop_pose(env, runtime, retained_ids, bin_id, nominal_goal):
    """Read actual held-payload geometry on private data, then plan normally.

    This deliberately uses simulator geometry for simulation bin occupancy.
    It is not a measured-camera bin-space implementation or a grasp shortcut.
    """
    import mujoco
    from scipy.spatial.transform import Rotation
    from experiments.graspgenx_sim.physical_containment import CollisionGeometryBounds,catalog_bounds
    goal=np.asarray(nominal_goal,float).copy()
    data=mujoco.MjData(env.model);data.qpos[:]=env.data.qpos;mujoco.mj_forward(env.model,data)
    bounds=CollisionGeometryBounds(env.model)
    specs={int(s.category_id):s for s in env.object_specs}
    def object_box(obj, snapshot):
        position=snapshot.xpos[obj.body_id];rotation=snapshot.xmat[obj.body_id].reshape(3,3)
        return bounds.object_bounds(snapshot,obj.geom_ids,
            catalog_bounds(position,rotation,specs[int(obj.category_id)].dimensions))
    box=next(b for b in env._bins if int(b['id'])==int(bin_id))
    bin_geoms=tuple(i for i in range(env.model.ngeom) if (env.model.geom(i).name or '').startswith(f'bin_{bin_id}_'))
    interior=bounds.bin_interior(data,box,bin_geoms)
    occupied=[]
    for obj in env._objects:
        if obj is runtime or obj.logical_id not in retained_ids:continue
        other=object_box(obj,data);occupied.append((other.lower[:2],other.upper[:2]))
    tcp=data.site_xpos[env.ee_site_id].copy();tcp_r=data.site_xmat[env.ee_site_id].reshape(3,3).copy()
    rel_p=tcp_r.T@(data.xpos[runtime.body_id]-tcp)
    rel_r=tcp_r.T@data.xmat[runtime.body_id].reshape(3,3)
    goal_r=Rotation.from_quat(goal[3:][[1,2,3,0]]).as_matrix()
    projected_p=goal[:3]+goal_r@rel_p;projected_r=goal_r@rel_r
    data.qpos[runtime.qpos_adr:runtime.qpos_adr+7]=np.r_[projected_p,
        Rotation.from_matrix(projected_r).as_quat()[[3,0,1,2]]]
    mujoco.mj_forward(env.model,data)
    projected=object_box(runtime,data);size=projected.upper-projected.lower
    if size[2]>interior.upper[2]-interior.lower[2]:raise ValueError('payload taller than bin interior')
    old_center=(projected.lower[:2]+projected.upper[:2])/2
    centers=free_centers(interior.lower[:2],interior.upper[:2],size[:2],occupied,old_center)
    center=centers[0]
    candidate_poses=[]
    for candidate in centers:
        pose=goal.copy();pose[:2]+=candidate-old_center;candidate_poses.append(pose.tolist())
    goal[:2]+=center-old_center
    return goal,{'source':'mujoco_payload_and_placed_geometry_simulation_only',
        'nominal_tcp_pose_world':np.asarray(nominal_goal).tolist(),
        'selected_tcp_pose_world':goal.tolist(),'payload_footprint_size_m':size[:2].tolist(),
        'selected_payload_center_xy_m':center.tolist(),'occupied_footprints':len(occupied),
        'bin_interior':interior.diagnostics(),'live_object_state_modified':False,
        'candidate_tcp_poses_world':candidate_poses,
        'candidate_payload_centers_xy_m':[c.tolist() for c in centers],
        'route_and_payload_collision_check_required':True}


def select_drop_pose_candidates(env, runtime, retained_ids, bin_id, nominal_goal):
    """Generate separately checked footprints for multiple upright tool yaws."""
    from scipy.spatial.transform import Rotation
    nominal=np.asarray(nominal_goal,float)
    base=Rotation.from_quat(nominal[3:][[1,2,3,0]])
    combined=None;failures=[]
    for angle in (0.,np.pi/2,-np.pi/2,np.pi):
        rotated=nominal.copy()
        rotated[3:]=(Rotation.from_euler('z',angle)*base).as_quat()[[3,0,1,2]]
        try:
            _,group=select_drop_pose(env,runtime,retained_ids,bin_id,rotated)
        except ValueError as error:
            failures.append({'yaw_offset_deg':float(np.rad2deg(angle)),'reason':str(error)})
            continue
        count=len(group['candidate_tcp_poses_world'])
        group['candidate_yaw_offset_deg']=[float(np.rad2deg(angle))]*count
        if combined is None:
            combined=group
        else:
            for key in ('candidate_tcp_poses_world','candidate_payload_centers_xy_m','candidate_yaw_offset_deg'):
                combined[key].extend(group[key])
    if combined is None:
        raise ValueError('no free footprint in any tested tool yaw: '+str(failures))
    combined['yaw_group_failures']=failures
    combined['orientation_scope']='four explicit yaw offsets, each with separate occupied-footprint check'
    return np.asarray(combined['candidate_tcp_poses_world'][0],float),combined

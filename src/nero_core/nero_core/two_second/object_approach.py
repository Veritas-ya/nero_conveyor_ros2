"""Continuous grasp approach in the translating frozen object's local frame."""
import numpy as np
from scipy.interpolate import BPoly


def configure(config):
    """Activate this site's relative reference for the selected single-frame Fast entry."""
    spec = config.raw.get('fast_object_relative_approach')
    if spec is None:
        return
    if not isinstance(spec,dict) or config.raw['intercept_strategy'] != 'synchronized':
        raise ValueError('相对接近需要同步截获策略与阶段配置')
    def positive(value):
        return (not isinstance(value,bool) and isinstance(value,(int,float))
                and np.isfinite(value) and value>0.)
    for key in ('join_candidates_s','descent_candidates_s'):
        if not isinstance(spec.get(key),list) or not spec[key] or not all(positive(v) for v in spec[key]):
            raise ValueError('相对接近阶段时长须为有限正数：'+key)
    if not all(positive(spec.get(k)) for k in ('sync_s','forward_ramp_s')):
        raise ValueError('高位同步时间和沿带加速时间须为正')
    lead = (config.raw['command_advance_s']+1./config.settings.control_hz+.5/config.raw['planning_knot_hz']
            if config.raw.get('brake_no_descent_rise_m') else 0.)
    if min(spec['join_candidates_s']) <= max(spec['descent_candidates_s'])+spec['sync_s']+lead:
        raise ValueError('接近时间不足以完成高位同步、下降及闭爪高度保护')
    config.raw['object_relative_approach'] = dict(spec)
    config.raw['synchronized_join_candidates_s'] = list(spec['join_candidates_s'])
    config.raw['grasp_workpiece_collision_check_enabled'] = True


def wrist_time(angle, config):
    """Estimate only: all seven actual joint polynomials still require certification."""
    return max(1.875*angle/min(config.velocity_limits[-3:]),
               np.sqrt(10./np.sqrt(3.)*angle/min(config.acceleration_limits[-3:])))


def order_joins(durations, approach, angle, config, height_lead, *, earliest, target, grasp):
    estimate = wrist_time(angle,config) + approach['descent_candidates_s'][0] + approach['sync_s'] + height_lead
    velocity = target.velocity
    speed = velocity @ config.direction
    remaining = (config.follow_bounds[1]-grasp@config.direction)/speed
    for axis,rate in enumerate(velocity):
        if abs(rate)>1e-8:
            bound=config.workspace[axis,int(rate>0)]
            remaining=min(remaining,(bound-grasp[axis])/rate-config.raw['brake_duration_s']/2)
    available=remaining-(earliest-target.capture_s)-config.raw.get('synchronized_duration_s',0.)-config.settings.full_plan_close_window_s
    # A longer alignment can help the wrist but may run out of downstream
    # workspace. This estimate only orders candidates, never authorizes motion.
    return sorted(durations,key=lambda t:(t>available,abs(t-estimate)+max(0.,estimate-t)))


def lowest_mesh_projection(model, direction, origin):
    """Lowest gripper vertex relative to origin along a base-frame unit direction."""
    return min(float(np.min(model.mesh_vertices[g] @
        (model.data.geom_xmat[g].reshape(3, 3).T @ direction))
        + (model.data.geom_xpos[g]-origin) @ direction)
        for g in model.gripper_geometries)


def required_clearance(model, target, grasp, start, width, margin):
    """Height of the highest visible target point above the lowest open gripper vertex at alignment."""
    basis = np.asarray(target.object_pose)[:3, :3]
    normal = basis[:, 2]
    # All gripper meshes have a fixed transform to TCP at this opening.
    # Use the measured start joints (not a pose left by a previous candidate).
    model.set_state(start, width)
    tcp = model.tcp()
    local_normal = tcp[:3, :3] @ grasp[:3, :3].T @ normal
    lowest = lowest_mesh_projection(model, local_normal, tcp[:3, 3])
    top = float(np.max((target.points-grasp[:3, 3]) @ normal))
    return max(0., top-lowest) + margin


def join_curve(ref, tool_velocity):
    """Construct the Cartesian Bernstein curve from jets in the object's coordinates.

    The moving origin is the grasp point fixed in the OBB; using another
    origin on the same rigid object only adds a constant to its local offsets.
    """
    R = np.asarray(ref.object_axes_world, float)
    if (R.shape != (3,3) or not np.isfinite(R).all()
            or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0) or np.linalg.det(R)<0.):
        raise ValueError('无效的工件姿态')
    t0 = ref.rotate_first_s
    end = ref.prealign_s
    level = min(end, ref.duration_s-ref.height_lead_s)
    down = level-ref.relative_descent_s
    align = down-ref.relative_sync_s
    if not (np.isfinite(ref.approach_clearance_m) and t0 < align < down < level <= end and ref.approach_clearance_m > 0):
        raise ValueError('相对接近时段不足')
    v = ref.velocity_world
    zero = np.zeros(3)
    def origin(t):
        return ref.capture_tcp_world + v*(ref.start_s+t-ref.capture_s)
    knots, values = [], []
    def waypoint(t, offset, rate, acceleration=zero):
        knots.append(t-t0)
        values.append([origin(t)+R@offset, v+R@rate, R@acceleration])
    waypoint(t0, R.T@(ref.start_tcp_world-origin(t0)), -R.T@v)
    high = np.array([0., 0., ref.approach_clearance_m])
    waypoint(align, high, zero)
    waypoint(down, high, zero)
    # Meet the existing close/brake height guard at its original transition:
    # p(contact)=moving grasp, v(contact)=tool_velocity and a(contact)=0.
    drift = R.T@(tool_velocity-v)
    waypoint(level, drift*(level-ref.duration_s), drift)
    if level < end:
        waypoint(end, drift*(end-ref.duration_s), drift)
    curve = BPoly.from_derivatives(knots, values)
    # A high wrist rotation need not force a long trip downstream. Where
    # the rendezvous is nearby, start the along-belt ramp later while the
    # normal/cross-belt motion and wrist alignment continue uninterrupted.
    axis = v - R[:,2]*(R[:,2]@v)
    speed = np.linalg.norm(axis)
    if speed <= 1e-9:
        return curve
    axis /= speed
    aligned = align-t0
    distance = (curve(aligned)-ref.start_tcp_world)@axis
    ramp = 2*distance/speed
    if not 1e-6 < ramp < aligned-1e-6:
        return curve
    onset = aligned-ramp
    begin = float(ref.start_tcp_world@axis)
    along = BPoly.from_derivatives([onset, aligned], [[begin,0.,0.],[begin+distance,speed,0.]])
    times = sorted([*knots, onset])
    jets = []
    for t in times:
        jet = np.stack([curve(t,d) for d in range(3)])
        if t < aligned:
            desired = np.array([begin,0.,0.]) if t <= onset else np.array([along(t,d) for d in range(3)])
            jet += (desired-jet@axis)[:,None]*axis
        jets.append(jet)
    return BPoly.from_derivatives(times, jets)

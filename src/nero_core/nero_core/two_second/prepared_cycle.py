"""Prepare current-payload taught transport and empty return before moving."""
from dataclasses import dataclass, replace
import time
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .trajectory import TimedPath
from .taught import certify_taught
from .taught_timing import limit_jerk


@dataclass(frozen=True)
class PreparedCycle:
    identity: tuple
    scene_revision: int
    carry: TimedPath
    returning: TimedPath


def concatenate(paths, *, identity, version, stage):
    paths = tuple(p for p in paths if p.duration > 1e-10)
    if not paths:
        raise PickRejected('预计算链没有有效轨迹')
    degree = max(len(p.curve.c) for p in paths)
    coefficients, knots, pieces, protected = [], [0.], [], []
    offset = 0.
    for i, path in enumerate(paths):
        if i:
            protected.append(offset)
            for derivative in (0, 1, 2):
                if np.max(abs(paths[i-1].curve(paths[i-1].duration, derivative)-path.curve(0., derivative))) > 1e-7:
                    raise PickRejected('预计算链的q/v/a连接不连续')
        c = np.pad(path.curve.c, ((degree-len(path.curve.c), 0), (0, 0), (0, 0)))
        coefficients.append(c)
        knots.extend((path.curve.x[1:]+offset).tolist())
        pieces.append({'stage': path.stage, 'start_s': offset, 'end_s': offset+path.duration,
                       'diagnostics': path.diagnostics})
        protected.extend(offset+t for t in path.diagnostics.get('route_waypoint_times_s', []))
        protected.extend(offset+t for t in path.diagnostics.get('protected_waypoint_times_s', []))
        offset += path.duration
    curve = PPoly(np.concatenate(coefficients, axis=1), knots, extrapolate=False)
    return TimedPath(curve, identity, version, stage, offset,
        sum(p.planning_s for p in paths), {'prepared_chain': True, 'continuous_qva': True,
        'pieces': pieces, 'protected_waypoint_times_s':sorted(set(protected)),
        'internal_arrival_dwell_s': 0., 'internal_online_planning': False})


def nominal_state(model, joints, width, force=0.):
    """A planning waypoint, explicitly not an invented device observation."""
    model.set_state(joints, width)
    flange = model.flange().copy()
    return {'joints': np.asarray(joints).tolist(), 'width_m': float(width), 'force_n': float(force),
            'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}


def direct_return(planner, config, start, opening, *, cancelled):
    """Reuse a route candidate, never its task-specific scene certificate."""
    model = planner.model
    start = np.asarray(start, float)
    key = (tuple(start), tuple(config.standby), float(opening),
           tuple(config.velocity_limits), tuple(config.acceleration_limits),
           tuple(model.command_lower), tuple(model.command_upper),
           config.settings.control_hz, config.raw.get('taught_joint_jerk_limit_rad_s3'),
           config.raw.get('trajectory_profile'), config.raw.get('geometry_check_hz'))
    cached = getattr(planner, '_direct_return_template', None)
    if cached is not None and cached[0] == key:
        started = time.monotonic()
        template = cached[1]
        try:
            if cancelled():
                raise PickRejected('直接返回轨迹检查已取消')
            # Identity, released object and collision scene belong to this
            # cycle. Only the curve coefficients survive between cycles.
            candidate = replace(template, identity=planner.identity, version=planner.version+1)
            planner.check_path(candidate, opening, payload=False, cancelled=cancelled)
            if cancelled():
                raise PickRejected('直接返回轨迹检查已取消')
            planner.version += 1
            diagnostics = {k: v for k, v in candidate.diagnostics.items() if k != 'route_diagnostics'}
            return replace(candidate, planning_s=time.monotonic()-started,
                diagnostics={**diagnostics, 'route_template_reused': True,
                             'scene_certificate': 'rechecked_for_current_release_and_scene',
                             'scene_revision': planner.scene_revision,
                             'target_identity': planner.identity})
        except PickRejected:
            if cancelled():
                raise
            # A changed scene may require a different certified route.
            planner._direct_return_template = None
    state = nominal_state(model, start, opening)
    path = planner.route(state, stage='return', goal_joints=config.standby,
        payload=False, start_joints=start, cancelled=cancelled)
    path = limit_jerk(path, config.raw.get('taught_joint_jerk_limit_rad_s3'),
                      config.settings.control_hz)
    path = replace(path, diagnostics={**path.diagnostics, 'route_template_reused': False})
    planner._direct_return_template = (key, path)
    return path


def prepare_cycle(planner, config, state, start_joints, *, cancelled, recovery=False):
    segments = config.raw['taught_segments']
    if planner.payload_local is None:
        raise PickRejected('预计算搬运缺少当前工件载物估计')
    saved = {key: getattr(planner, key, None) for key in (
        'payload_local', 'payload_tree', 'capture_tcp', 'released_points', 'released_tree')}
    prefix = 'prepared_taught_recovery' if recovery else 'prepared_taught'
    actual_joints, width = np.array(state['joints']), float(state['width_m'])
    def route(*args, **kwargs):
        path = planner.route(*args, **kwargs)
        return limit_jerk(path, config.raw.get('taught_joint_jerk_limit_rad_s3'),
                          config.settings.control_hz)
    try:
        planner.model.check_feedback(state)
        lift = planner.model.tcp().copy()
        lift[:3, 3] += planner.support*config.settings.lift_m
        first = route(state, stage='lift', goal_pose=lift, payload=True,
            start_joints=start_joints, cancelled=cancelled)
        after_lift = nominal_state(planner.model, first.q1, width)
        if config.raw.get('experimental_direct_drop_guard'):
            if config.raw.get('direct_drop_orientation_first',False):
                from .drop_orientation import orientation_first_drop
                transfer = orientation_first_drop(planner,config,after_lift,first.q1,
                    segments['transfer'].joints[-1],cancelled=cancelled)
            else:
                transfer = route(after_lift, stage='direct_drop',
                    goal_joints=segments['transfer'].joints[-1], payload=True,
                    start_joints=first.q1, cancelled=cancelled)
            carry = concatenate((first, transfer), identity=planner.identity,
                version=planner.version, stage=prefix+'_carry')
            if (config.raw.get('direct_drop_continuous_carry', False) and
                    config.raw.get('direct_drop_orientation_first',False)):
                from .continuous_carry import compile_carry
                planner.model.set_state(transfer.q1,width)
                carry = compile_carry(planner,carry,width,lift_end_s=first.duration,
                    terminal_start_s=first.duration+
                        transfer.diagnostics['drop_orientation_schedule']['terminal_start_s'],
                    destination=planner.model.tcp().copy(),
                    final_radius_m=config.raw['experimental_direct_drop_guard']['radius_m']/2.,
                    cancelled=cancelled)
            elif config.raw.get('direct_drop_blend_enabled', True):
                from .carry_blend import blend_checked_path
                carry = blend_checked_path(planner,carry,width,[(first.duration,'incoming')],
                                           cancelled=cancelled)
            if config.raw.get('direct_drop_orientation_first',False):
                from .drop_orientation import check_final_orientation
                planner.model.set_state(transfer.q1,width)
                drop = planner.model.tcp().copy()
                angle = check_final_orientation(planner,carry,width,drop,
                    config.raw['experimental_direct_drop_guard']['radius_m']/2.,cancelled=cancelled)
                carry = replace(carry,diagnostics={**carry.diagnostics,
                    'drop_orientation_schedule':{**transfer.diagnostics['drop_orientation_schedule'],
                        'terminal_start_s':(first.duration+
                            transfer.diagnostics['drop_orientation_schedule']['terminal_start_s'])*
                            carry.duration/(first.duration+transfer.duration),
                        'time_base':'carry_start',
                        'max_final_orientation_error_rad':angle}})
        elif config.raw.get('taught_entry_search_enabled', False):
            from .taught_entry import choose_entry, prepare_entries
            entries = config.raw.get('_prepared_taught_entries')
            if entries is None:
                entries = prepare_entries(config, segments['transfer'], planner.model, cancelled=cancelled)
            connecting, transfer = choose_entry(planner, config, after_lift, first.q1,
                                                 entries, cancelled=cancelled)
        else:
            connecting = route(after_lift, stage='transfer_connect',
                goal_joints=segments['transfer'].joints[0], payload=True, start_joints=first.q1, cancelled=cancelled)
            at_transfer = nominal_state(planner.model, connecting.q1, width)
            transfer = certify_taught(planner, config, segments['transfer'], at_transfer,
                connecting.q1, payload=True, cancelled=cancelled)
        if not config.raw.get('experimental_direct_drop_guard'):
            carry = concatenate((first, connecting, transfer), identity=planner.identity,
                version=planner.version, stage=prefix+'_carry')
        # Certify the nominal released cloud now. The final actual release
        # estimate is checked again while the gripper opening is confirmed.
        at_drop = nominal_state(planner.model, transfer.q1, width)
        planner.validate_release_opening(at_drop)
        planner.release_measured(at_drop)
        opening = config.settings.open_width_m
        start = transfer.q1
        return_parts = []
        if config.raw.get('taught_return_strategy', 'recorded_then_wait') == 'direct_to_wait':
            return_parts.append(direct_return(planner, config, start, opening, cancelled=cancelled))
        else:
            returning = segments['return']
            if np.max(abs(start-returning.joints[0])) > 1e-9:
                return_parts.append(route(nominal_state(planner.model, start, opening),
                    stage='return_connect', goal_joints=returning.joints[0], payload=False,
                    start_joints=start, cancelled=cancelled))
                start = return_parts[-1].q1
            return_parts.append(certify_taught(planner, config, returning,
                nominal_state(planner.model, start, opening), start, payload=False, cancelled=cancelled))
            start = return_parts[-1].q1
            if np.max(abs(start-config.standby)) > 1e-9:
                return_parts.append(route(nominal_state(planner.model, start, opening),
                    stage='return', goal_joints=config.standby, payload=False,
                    start_joints=start, cancelled=cancelled))
        result_return = concatenate(return_parts, identity=planner.identity,
            version=planner.version, stage=prefix+'_return')
        if cancelled():
            raise PickRejected('预计算示教链已取消')
        return PreparedCycle(planner.identity, planner.scene_revision, carry, result_return)
    finally:
        for key, value in saved.items():
            setattr(planner, key, value)
        planner.model.set_state(actual_joints, width)

"""Prepare current-payload taught transport and empty return before moving."""
from dataclasses import dataclass
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
    coefficients, knots, pieces = [], [0.], []
    offset = 0.
    for i, path in enumerate(paths):
        if i:
            for derivative in (0, 1, 2):
                if np.max(abs(paths[i-1].curve(paths[i-1].duration, derivative)-path.curve(0., derivative))) > 1e-7:
                    raise PickRejected('预计算链的q/v/a连接不连续')
        c = np.pad(path.curve.c, ((degree-len(path.curve.c), 0), (0, 0), (0, 0)))
        coefficients.append(c)
        knots.extend((path.curve.x[1:]+offset).tolist())
        pieces.append({'stage': path.stage, 'start_s': offset, 'end_s': offset+path.duration,
                       'diagnostics': path.diagnostics})
        offset += path.duration
    curve = PPoly(np.concatenate(coefficients, axis=1), knots, extrapolate=False)
    return TimedPath(curve, identity, version, stage, offset,
        sum(p.planning_s for p in paths), {'prepared_chain': True, 'continuous_qva': True,
        'pieces': pieces, 'internal_arrival_dwell_s': 0., 'internal_online_planning': False})


def nominal_state(model, joints, width, force=0.):
    """A planning waypoint, explicitly not an invented device observation."""
    model.set_state(joints, width)
    flange = model.flange().copy()
    return {'joints': np.asarray(joints).tolist(), 'width_m': float(width), 'force_n': float(force),
            'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}


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
        connecting = route(after_lift, stage='transfer_connect',
            goal_joints=segments['transfer'].joints[0], payload=True, start_joints=first.q1, cancelled=cancelled)
        at_transfer = nominal_state(planner.model, connecting.q1, width)
        transfer = certify_taught(planner, config, segments['transfer'], at_transfer,
            connecting.q1, payload=True, cancelled=cancelled)
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

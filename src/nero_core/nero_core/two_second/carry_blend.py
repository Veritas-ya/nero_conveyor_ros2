"""Locally blend stopped route corners, retaining exact waypoints and endpoints."""
from dataclasses import replace
from math import factorial
import time

import numpy as np
from scipy.interpolate import BPoly, PPoly

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .motion_profile import select_motion_profile
from .trajectory import extrema


def blend_curve(curve, joins, gain, *, protected_times=()):
    """Quintic patches match q/v/a on both sides and pass through the waypoint.

    The lift join uses the incoming tangent, so reaching the lift waypoint
    does not already require sideways motion. Every modified curve still
    needs a fresh full geometry certificate.
    """
    centers = [j[0] for j in joins]
    bounds = np.unique([curve.x[0], *centers, *protected_times, curve.x[-1]])
    patches = []
    for t, mode in joins:
        left = t-.4*(t-bounds[bounds < t-1e-10][-1])
        right = t+.4*(bounds[bounds > t+1e-10][0]-t)
        q = curve(t)
        v = ((q-curve(left))/(t-left) if mode == 'incoming' else
             (curve(right)-q)/(right-t) if mode == 'outgoing' else
             (curve(right)-curve(left))/(right-left))*gain
        values = [[curve(left,d) for d in (0,1,2)], [q,v,np.zeros(7)],
                  [curve(right,d) for d in (0,1,2)]]
        patch = PPoly.from_bernstein_basis(BPoly.from_derivatives([left,t,right],values))
        patches.append((left,right,patch))
    knots = np.unique(np.concatenate([curve.x, *[p.x for _,_,p in patches]]))
    degree = max(curve.c.shape[0]-1, 5)
    blocks = []
    for a,b in zip(knots[:-1],knots[1:]):
        source = next((p for left,right,p in patches if left <= (a+b)/2 <= right),curve)
        blocks.append(np.array([source(a,d)/factorial(d) for d in range(degree,-1,-1)]))
    return PPoly(np.stack(blocks,axis=1),knots,extrapolate=False)


def blend_checked_path(planner, path, width, joins, *, cancelled,
                       gains=(.2,.1,.05), validate=None, required=False):
    """Certify the whole curve; optionally require a successful moving join.

    Legacy local patches may keep the original checked route. Whole-chain
    transport uses required=True so a stopped fallback is never mislabeled.
    """
    if not joins:
        return path
    started = time.monotonic()
    motion = select_motion_profile(planner.config,payload=True)
    protected = list(path.diagnostics.get('route_waypoint_times_s', []))
    protected.extend(path.diagnostics.get('protected_waypoint_times_s', []))
    for piece in path.diagnostics.get('pieces', []):
        protected.extend(piece['start_s']+t for t in piece['diagnostics'].get('route_waypoint_times_s', []))
    failures = []
    for gain in gains:
        if cancelled():
            raise PickRejected('搬运平滑已取消')
        try:
            curve = blend_curve(path.curve,joins,gain,protected_times=protected)
            low,high = extrema(curve,0)
            if np.any(low < planner.model.command_lower-1e-9) or np.any(high > planner.model.command_upper+1e-9):
                raise PickRejected('搬运平滑超出关节位置范围')
            v = np.max(abs(np.asarray(extrema(curve,1)))/motion.velocity_limits)
            a = np.sqrt(np.max(abs(np.asarray(extrema(curve,2)))/motion.acceleration_limits))
            jerk = float(np.max(abs(np.asarray(extrema(curve,3)))))
            limit = planner.config.raw.get('taught_joint_jerk_limit_rad_s3')
            scale = max(1.,float(v),float(a),0. if limit is None else np.cbrt(jerk/limit))
            if scale > 1.:
                scale *= 1.02
            hz = motion.settings.control_hz
            scale = np.ceil(path.duration*scale*hz-1e-9)/hz/path.duration
            c = curve.c.copy()
            for i in range(len(c)):
                c[i] /= scale**(len(c)-1-i)
            curve = PPoly(c,curve.x*scale,extrapolate=False)
            report = {'applied':True,'gain':gain,'time_scale':float(scale),
                'original_duration_s':path.duration,'exact_waypoints_retained':True,
                'joins':[{'time_s':t*scale,'mode':mode,
                          'max_joint_speed_rad_s':float(np.max(abs(curve(t*scale,1))))}
                         for t,mode in joins],
                'peak_joint_velocity_rad_s':np.max(abs(np.asarray(extrema(curve,1))),axis=0).tolist(),
                'peak_joint_acceleration_rad_s2':np.max(abs(np.asarray(extrema(curve,2))),axis=0).tolist(),
                'peak_joint_jerk_rad_s3':jerk/scale**3}
            diagnostics = {**path.diagnostics,'corner_blend':report}
            for key in ('route_waypoint_times_s','protected_waypoint_times_s'):
                if key in diagnostics:
                    diagnostics[key] = [t*scale for t in diagnostics[key]]
            if 'pieces' in diagnostics:
                diagnostics['pieces'] = [{**p,'start_s':p['start_s']*scale,'end_s':p['end_s']*scale}
                                         for p in diagnostics['pieces']]
            candidate = replace(path,curve=curve,duration=path.duration*scale,diagnostics=diagnostics)
            planner.check_path(candidate,width,payload=True,cancelled=cancelled)
            if validate is not None:
                validate(candidate)
            report.update(geometry_rechecked=True,wall_s=time.monotonic()-started,failed_attempts=failures)
            return candidate
        except PickRejected as error:
            if cancelled():
                raise
            failures.append({'gain':gain,'reason':str(error),'details':error.details})
    if required:
        raise PickRejected('整段投放轨迹未找到通过认证的连续连接',details={'failed_attempts':failures})
    return replace(path,diagnostics={**path.diagnostics,'corner_blend':{
        'applied':False,'failed_attempts':failures,'wall_s':time.monotonic()-started}})

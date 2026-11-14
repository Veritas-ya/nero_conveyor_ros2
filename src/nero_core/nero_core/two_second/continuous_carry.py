"""Compile lift, transfer and bin approach into one continuous timed curve."""
from dataclasses import replace
import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .carry_blend import blend_checked_path
from .drop_orientation import check_final_orientation


def compile_carry(planner, path, width, *, lift_end_s, terminal_start_s,
                  destination, final_radius_m, cancelled):
    """Keep exact route vertices; blend all joins before one global retiming.

    Internal waypoints are geometry constraints, not rest/arrival commands.
    The lift uses its incoming tangent and the bin approach its outgoing
    tangent. A larger initial blend than the former local patches reduces
    the almost-stopped motion at these joins; full q/v/a/jerk and geometry
    certification still determines its usable duration.
    """
    joins = [(float(t), 'bisector') for t in
             path.diagnostics.get('protected_waypoint_times_s', [])]
    joins = [(t, 'incoming' if abs(t-lift_end_s) < 1e-9 else
                 'outgoing' if abs(t-terminal_start_s) < 1e-9 else mode)
             for t, mode in joins]
    def validate(candidate):
        for join in candidate.diagnostics['corner_blend']['joins']:
            # Numerical zero, not a new hardware motion threshold. Do not
            # label a failed/reversing join as non-stop continuous transport.
            if join['max_joint_speed_rad_s'] <= 1e-9:
                raise PickRejected('投放内部连接仍为全轴静止点',details=join)
        check_final_orientation(planner,candidate,width,destination,final_radius_m,
                                cancelled=cancelled)
    result = blend_checked_path(planner,path,width,joins,cancelled=cancelled,
        gains=(.5,.35,.2,.1,.05),validate=validate,required=True)
    report = result.diagnostics['corner_blend']
    scale = result.duration/path.duration
    return replace(result,diagnostics={**result.diagnostics,
        'continuous_carry':{
            'applied':True,'time_base':'carry_start',
            'scope':'brake_endpoint_to_drop_after_grip_confirmation',
            'single_stream':True,'internal_arrival_checks':0,
            'internal_planned_stops':0,'internal_planning':False,
            'lift_end_s':float(lift_end_s*scale),
            'terminal_start_s':float(terminal_start_s*scale),
            'duration_s':result.duration,'joins':report['joins'],
            'start_and_drop_joints_preserved':True,
            'joint_derivative_checks':'complete_polynomial_extrema',
            'geometry_rechecked':True,
            'rest_qva_endpoints':all(np.max(abs(result.curve(t,d))) < 1e-7
                for t in (0.,result.duration) for d in (1,2))}})

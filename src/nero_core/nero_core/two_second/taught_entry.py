"""Rank and certify multiple possible entries into a recorded transfer path."""
from dataclasses import replace
from hashlib import sha256
import time
import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .taught import TaughtSegment, prepare_taught_timing, certify_taught
from .taught_timing import limit_jerk
from .trajectory import extrema


def suffix(segment, index):
    if index < 0 or len(segment.times)-index < 4:
        raise ValueError('示教切入点必须保留至少四个采样')
    if index == 0:
        return segment
    times = (segment.times[index:]-segment.times[index]).copy()
    joints = segment.joints[index:].copy()
    times.setflags(write=False); joints.setflags(write=False)
    digest = sha256(f'{segment.sha256}:suffix:{index}'.encode()).hexdigest()
    return TaughtSegment('transfer', times, joints, segment.path, digest)


def prepare_entries(config, segment, model, *, cancelled=lambda:False):
    """Precompute only timing; current scene and payload are checked later."""
    entries = []
    rejected = []
    for index in config.raw.get('taught_entry_indices', [0]):
        if cancelled():
            raise PickRejected('示教切入点时序准备已取消')
        part = suffix(segment, index)
        try:
            timing, _ = prepare_taught_timing(config, part, model, payload=True, cancelled=cancelled)
        except PickRejected as error:
            if cancelled() or index == 0:
                raise
            rejected.append({'index':index,'error':str(error),'details':error.details})
            continue
        entries.append((index, part, timing.duration))
    config.raw['_taught_entry_prewarm_rejections'] = rejected
    return tuple(entries)


def choose_entry(planner, config, state, start_joints, entries, *, cancelled):
    """Select among completely checked paths using distance, time and smoothness.

    The score is the sum of ordinal ranks among the successful candidates;
    it does not assert a physical optimum or invent conversion weights.
    """
    from .prepared_cycle import nominal_state
    start = np.asarray(start_joints, float)
    width = float(state['width_m'])
    planner.model.set_state(start, width)
    start_tcp = planner.model.tcp().copy()
    rows = []
    for index, segment, _ in entries:
        if cancelled():
            raise PickRejected('示教切入点选择已取消')
        planner.model.set_state(segment.joints[0], width)
        distance = float(np.linalg.norm(planner.model.tcp()[:3,3]-start_tcp[:3,3]))
        joint_distance = float(np.linalg.norm(segment.joints[0]-start))
        begun = time.monotonic()
        try:
            anchor = nominal_state(planner.model, start, width)
            connection = planner.route(anchor, stage='transfer_connect', goal_joints=segment.joints[0],
                payload=True, start_joints=start, cancelled=cancelled)
            connection = limit_jerk(connection, config.raw.get('taught_joint_jerk_limit_rad_s3'),
                                    config.settings.control_hz)
            at_entry = nominal_state(planner.model, connection.q1, width)
            transfer = certify_taught(planner, config, segment, at_entry, connection.q1,
                                      payload=True, cancelled=cancelled)
            jerk_connect = float(np.max(abs(np.asarray(extrema(connection.curve,3)))))
            jerk_transfer = float(np.max(abs(np.asarray(extrema(transfer.curve,3)))))
            row = {'index':index,'tcp_distance_m':distance,'joint_distance_rad':joint_distance,
                   'connection_s':connection.duration,'suffix_s':transfer.duration,
                   'total_s':connection.duration+transfer.duration,
                   'peak_jerk_rad_s3':max(jerk_connect,jerk_transfer),
                   'planning_wall_s':time.monotonic()-begun,'status':'certified',
                   '_connection':connection,'_transfer':transfer}
        except PickRejected as error:
            if cancelled():
                raise
            row = {'index':index,'tcp_distance_m':distance,'joint_distance_rad':joint_distance,
                   'status':'rejected','error':str(error),'details':error.details,
                   'planning_wall_s':time.monotonic()-begun}
        rows.append(row)
    successful = [row for row in rows if row['status']=='certified']
    if not successful:
        raise PickRejected('所有示教切入点均无法交付', details={'candidates':rows})
    # Ranking avoids adding metres, seconds, and rad/s^3 with arbitrary units.
    for row in successful:
        row['rank_score'] = sum(1+sum(other[key] < row[key]-1e-9 for other in successful)
                                for key in ('tcp_distance_m','total_s','peak_jerk_rad_s3'))
    selected = min(successful, key=lambda x:(x['rank_score'],x['total_s'],x['index']))
    evidence = [{k:v for k,v in row.items() if not k.startswith('_')} for row in rows]
    details = {'selected_index':selected['index'], 'candidate_evidence':evidence,
               'criteria':'equal ordinal ranks: TCP distance, combined duration, peak joint jerk',
               'connection_rest_to_rest':True,'not_continuous_nonzero_speed_blending':True}
    connection = replace(selected['_connection'], diagnostics={**selected['_connection'].diagnostics,
                         'taught_entry_selection':details})
    return connection, selected['_transfer']

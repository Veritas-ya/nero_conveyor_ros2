"""In-memory observation of trajectory, dispatch and feedback clocks.

This module never chooses a command or an arrival condition.  Device calls,
callback results and exceptions pass through unchanged.  File output happens
only after the sender has returned ownership, outside the control loop.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np
from .feedback_clock import mapped_stamp
from .types import identity_dict


def _vector(value):
    try:
        result = np.asarray(value, dtype=float).reshape(7)
        return result.copy() if np.isfinite(result).all() else np.full(7, np.nan)
    except (TypeError, ValueError):
        return np.full(7, np.nan)


def _fixed_vector(value, size):
    try:
        result = np.asarray(value, dtype=float).reshape(size)
        return result.copy() if np.isfinite(result).all() else np.full(size, np.nan)
    except (TypeError, ValueError):
        return np.full(size, np.nan)


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class ExecutionTrace:
    """Detached numeric records; collection failure cannot reject motion."""

    def __init__(self, *, clock=time.monotonic, wall_clock=time.time,
                 mode='hardware', policy_origin=None):
        self.clock, self.wall_clock, self.mode = clock, wall_clock, mode
        self.policy_origin = policy_origin
        self.command_rows, self.feedback_rows, self.handoffs = [], [], []
        self.edges = {}
        self.errors = []
        self.target_identity = None
        self.gripper_rows = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._last_feedback = {}
        self.timestamp_source = 'unknown_adapter_timestamp'
        self.joint_command_mode = 'unknown'
        self.stamp_clock_domain = 'unknown'
        self.host_wall_to_monotonic_offset_s = self.clock()-self.wall_clock()

    def _attempt(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            # Diagnostics must not turn a working trajectory into a stop.
            with self._lock:
                if len(self.errors) < 10:
                    self.errors.append(f'{type(error).__name__}: {error}')
            return None

    def wrap_device(self, device, *, stage=lambda: ''):
        module = type(device).__module__
        if module.endswith('.nero_pick_hardware') and type(device).__name__ == 'Hardware':
            self.timestamp_source = 'SDK joint_stamp=min(host CAN receive timestamps of joint packets)'
            self.stamp_clock_domain = 'host_wall'
            self.joint_command_mode = getattr(device, '_joint_command_mode', 'j')
        elif getattr(device, 'is_offline_replay', False):
            self.timestamp_source = 'offline ideal feedback synthetic device clock'
            self.stamp_clock_domain = 'offline_synthetic'
        return _TracedDevice(device, self, stage)

    def record_feedback(self, state, read_started, read_finished, *, stage=''):
        return self._attempt(self._record_feedback, state, read_started, read_finished, stage)

    def _record_feedback(self, state, read_started, read_finished, stage):
        if state is None:
            return
        stamp = _number(state.get('joint_stamp'))
        q = _vector(state.get('joints'))
        # The offset maps a host receipt timestamp, not encoder exposure time.
        read_wall = self.wall_clock()
        offset = self.clock()-read_wall
        stamp_mono = _number(mapped_stamp(state)) if 'feedback_time' in state else np.nan
        stamps = list(state.get('joint_stamps') or [])
        packet_stamps = np.asarray((stamps+[np.nan]*4)[:4], dtype=float)
        axis_stamps = packet_stamps[[0, 0, 1, 1, 2, 2, 3]]
        mapped = list(state.get('joint_stamps_monotonic') or [])
        axis_monotonic = np.asarray((mapped+[np.nan]*4)[:4], dtype=float)[[0,0,1,1,2,2,3]]
        row = {'feedback_q_rad': q, 'feedback_joint_stamp_s': stamp,
               'feedback_flange_pose_m_rad': _fixed_vector(state.get('pose'), 6),
               'feedback_joints_at_pose_rad': _vector(state.get('joints_at_pose_stamp')),
               'feedback_pose_packet_monotonic_s': _fixed_vector(state.get('pose_stamps_monotonic'), 3),
               'feedback_pose_joint_pairing': (state.get('feedback_timing') or {}).get('pose_joint_pairing'),
               'feedback_pose_grouping': (state.get('feedback_timing') or {}).get('pose_grouping'),
               'feedback_joint_sample_monotonic_s': stamp_mono,
               'feedback_read_started_monotonic_s': read_started,
               'feedback_read_monotonic_s': read_finished,
               'feedback_read_wall_s': read_wall,
               'feedback_host_wall_to_monotonic_offset_s': offset,
               'feedback_joint_axis_sample_monotonic_s': axis_monotonic,
               'sample_stage': str(stage), 'thread_id': threading.get_ident(),
               'stream_id': getattr(self._local, 'stream_id', ''),
               'feedback_joint_packet_stamps_s': packet_stamps,
               'feedback_time': state.get('feedback_time', {'mapping_quality':'missing'}),
               'encoder_sample_time': None, 'target_identity':identity_dict(self.target_identity)}
        with self._lock:
            self.feedback_rows.append(row)
            self._last_feedback[threading.get_ident()] = len(self.feedback_rows)-1

    def prepare_command(self, policy, *, policy_time_s, candidate, stage,
                        stream_id=None, trajectory=None, edge_started=None):
        return self._attempt(self._prepare_command, policy, policy_time_s,
                             candidate, stage, stream_id, trajectory, edge_started)

    def update_final_sample(self, policy_time_s, q, velocity, acceleration, sample_monotonic_s):
        """Update the already prepared row without rebuilding trace metadata."""
        return self._attempt(self._update_final_sample, policy_time_s, q, velocity, acceleration, sample_monotonic_s)

    def _update_final_sample(self, policy_time_s, q, velocity, acceleration, sample_monotonic_s):
        row = getattr(self._local, 'pending', None)
        if row is None:
            return
        phase = float(policy_time_s) - float(self.edges[row['stream_id']]['edge_started_policy_s'])
        duration = row['nominal_duration_s']
        row.update(policy_sample_time_s=float(policy_time_s), raw_phase_time_s=phase,
                   final_sample_monotonic_s=float(sample_monotonic_s),
                   raw_phase_progress=np.clip(phase/duration, 0., 1.) if duration > 0 else np.nan,
                   raw_trajectory_q_rad=np.asarray(q).copy(),
                   raw_trajectory_qd_rad_s=np.asarray(velocity).copy(),
                   raw_trajectory_qdd_rad_s2=np.asarray(acceleration).copy(),
                   policy_candidate_q_rad=np.asarray(q).copy())

    def _prepare_command(self, policy, policy_time_s, candidate, stage,
                         stream_id, trajectory, edge_started):
        self.target_identity = getattr(getattr(policy, 'env', None), 'target_identity', self.target_identity)
        trajectory = (getattr(policy, '_safe_edge_trajectory', None)
                      if trajectory is None else trajectory)
        edge_started = (getattr(policy, '_safe_edge_started_at', None)
                        if edge_started is None else edge_started)
        if trajectory is None:
            trajectory = getattr(policy, '_joint_trajectory', None)
            edge_started = getattr(policy, '_trajectory_started_at', None)
        if trajectory is None:
            intercept = getattr(policy, '_continuous_plan', None)
            if intercept is not None:
                segment = intercept.segment_at(policy_time_s)
                trajectory = segment.result.continuation_trajectory
                edge_started = segment.start
        full = getattr(policy, '_full_pick_bundle', None)
        full_reference = full if not getattr(policy, '_full_pick_finished', False) else None
        if full is not None and not getattr(policy, '_full_pick_finished', False):
            policy_time_s = getattr(policy, '_full_pick_sample_time_s', policy_time_s)
            full = full.segment_at(policy_time_s)
            name = full.phase(policy_time_s)
            trajectory, edge_started = {'approach': (full.approach, full.start),
                'descent': (full.descent, full.high_time), 'closing': (full.closing, full.contact_time),
                'lift': (full.lift, full.lift_time)}[name]
        key = str(stream_id) if stream_id is not None else (
            f'edge-{id(trajectory)}' if trajectory is not None else 'local-unparameterized')
        phase = (_number(policy_time_s)-_number(edge_started))
        raw = (np.full(7, np.nan),)*3
        duration = _number(getattr(trajectory, 'duration', None))
        if trajectory is not None and np.isfinite(phase) and callable(getattr(trajectory, 'sample_full', None)):
            raw = trajectory.sample_full(phase)
        request = getattr(policy, '_full_pick_request', None)
        if full is not None:request = full.request
        if request is None:
            intercept = getattr(policy, '_continuous_plan', None)
            if intercept is not None:request = intercept.segment_at(policy_time_s).request
        record = {'target_identity':identity_dict(self.target_identity),
                  'request_id':getattr(request, 'request_id', None),
                  'trajectory_version':None if request is None else [request.generation,request.request_id],
                  'planned_contact_time':getattr(request, 'planned_target_time_s', None),
                  'command_prepare_monotonic':self.clock(), 'sample_stage': str(stage), 'stream_id': key,
                  'policy_sample_time_s': _number(policy_time_s),
                  'raw_phase_time_s': phase, 'nominal_duration_s': duration,
                  'raw_phase_progress': (np.clip(phase/duration, 0., 1.)
                                         if duration > 0 and np.isfinite(phase) else np.nan),
                  'raw_trajectory_q_rad': _vector(raw[0]),
                  'raw_trajectory_qd_rad_s': _vector(raw[1]),
                  'raw_trajectory_qdd_rad_s2': _vector(raw[2]),
                  'policy_candidate_q_rad': _vector(candidate)}
        response = dict(getattr(policy, '_safe_edge_reference_diagnostics', {}) or {})
        record['model_reference_q_rad'] = _vector(response.get('candidate_q_rad'))
        record['model_reference_qd_rad_s'] = _vector(response.get('candidate_qd_rad_s'))
        record['model_reference_qdd_rad_s2'] = _vector(response.get('candidate_qdd_rad_s2'))
        record['applied_reference_q_rad'] = _vector(response.get('applied_reference_q_rad'))
        record['joint_response_diagnostics'] = response
        if trajectory is not None:
            config = getattr(policy, 'config', None)
            edge = {'q0_rad': _vector(getattr(trajectory, 'q0', None)),
                    'q1_rad': _vector(getattr(trajectory, 'q1', None)),
                    'nominal_duration_s': duration, 'edge_started_policy_s': _number(edge_started),
                    'position_tolerance_rad': _number(getattr(config, 'planner_corner_settle_tolerance', None)),
                    'velocity_tolerance_rad_s': _number(getattr(config, 'planner_waypoint_velocity_tolerance', None))}
            with self._lock:
                stored = self.edges.setdefault(key, edge)
                # The first stationary handoff can precede a known start time.
                # Fill only that unknown value when the same edge starts.
                if (not np.isfinite(stored['edge_started_policy_s'])
                        and np.isfinite(edge['edge_started_policy_s'])):
                    stored['edge_started_policy_s'] = edge['edge_started_policy_s']
        self._local.pending = record
        self._local.stream_id = key
        with self._lock:
            index = self._last_feedback.get(threading.get_ident())
            if index is not None:
                self.feedback_rows[index]['stream_id'] = key
                # Retain an immutable reference, without evaluating seven
                # packet-time samples in the real-time control loop.
                if full_reference is not None:
                    self.feedback_rows[index]['_full_reference'] = full_reference
                    self.feedback_rows[index]['_policy_origin'] = self.policy_origin

    def instrument_stream(self, owner):
        """Observe the existing callback; never solve or alter its result."""
        self._attempt(self._instrument_stream, owner)
        return owner

    def _instrument_stream(self, owner):
        original = owner.sender._step
        key = f'edge-{owner.key[1]}' if getattr(owner, 'key', None) else f'stream-{id(owner)}'

        def step(state, sample_time, dt):
            self._local.stream_id = key
            result = original(state, sample_time, dt)
            if result is not None:
                command = getattr(owner, '_command', None)
                candidate = getattr(command, 'requested_arm', result[0])
                self.prepare_command(owner.numeric,
                    policy_time_s=sample_time-owner.origin, candidate=candidate,
                    stage='pregrasp', stream_id=key, trajectory=owner.trajectory,
                    edge_started=owner.edge_started)
            return result

        owner.sender._step = step
        return owner

    def record_dispatch(self, q, begin, end, *, succeeded):
        return self._attempt(self._record_dispatch, q, begin, end, succeeded)

    def _record_dispatch(self, q, begin, end, succeeded):
        pending = getattr(self._local, 'pending', {})
        self._local.pending = {}
        row = dict(pending)
        row.update(dispatched_q_rad=_vector(q), dispatch_start_monotonic_s=begin,
                   dispatch_end_monotonic_s=end, dispatch_succeeded=bool(succeeded),
                   thread_id=threading.get_ident())
        with self._lock:
            row['feedback_row_index'] = self._last_feedback.get(threading.get_ident(), -1)
            self.command_rows.append(row)

    def record_handoff(self, owner, snapshot, when):
        return self._attempt(self._record_handoff, owner, snapshot, when)

    def _record_handoff(self, owner, snapshot, when):
        state = snapshot.last_feedback or {}
        goal = _vector(owner.trajectory.q1)
        q = _vector(state.get('joints'))
        reason = getattr(owner, 'reason', None) or {}
        self.handoffs.append({'stream_id': f'edge-{owner.key[1]}',
            'handoff_monotonic_s': when, 'feedback_joint_stamp_s': state.get('joint_stamp'),
            'feedback_q_rad': q, 'goal_q_rad': goal,
            'handoff_goal_error_rad': q-goal,
            'feedback_velocity_rad_s': _vector(reason.get('feedback_velocity_rad_s')),
            'handoff_velocity_error_rad_s': _vector(reason.get('feedback_velocity_rad_s')),
            'position_tolerance_rad': _number(owner.numeric.config.planner_corner_settle_tolerance),
            'velocity_tolerance_rad_s': _number(owner.numeric.config.planner_waypoint_velocity_tolerance),
            'reason': reason.get('reason'), 'diagnostics': dict(reason)})

    def arrays(self):
        with self._lock:
            commands, feedback = list(self.command_rows), list(self.feedback_rows)
        vectors = ('raw_trajectory_q_rad', 'raw_trajectory_qd_rad_s',
                   'raw_trajectory_qdd_rad_s2', 'model_reference_q_rad',
                   'model_reference_qd_rad_s', 'model_reference_qdd_rad_s2',
                   'applied_reference_q_rad', 'policy_candidate_q_rad', 'dispatched_q_rad')
        fields = ('policy_sample_time_s', 'raw_phase_time_s', 'nominal_duration_s',
                  'final_sample_monotonic_s',
                  'raw_phase_progress', 'dispatch_start_monotonic_s', 'dispatch_end_monotonic_s',
                  'feedback_row_index')
        result = {key: np.asarray([_vector(row.get(key)) for row in commands]).reshape(-1, 7)
                  for key in vectors}
        result.update({key: np.asarray([_number(row.get(key)) for row in commands]) for key in fields})
        result['final_sample_to_sdk_call_s'] = (result['dispatch_start_monotonic_s'] -
                                                result['final_sample_monotonic_s'])
        for key in ('sample_stage', 'stream_id'):
            result[key] = np.asarray([str(row.get(key, '')) for row in commands], dtype=str)
        result['command_context_json'] = np.asarray([json.dumps(_json({k:row.get(k) for k in
            ('target_identity','request_id','trajectory_version','planned_contact_time','command_prepare_monotonic')}),ensure_ascii=False) for row in commands],dtype=str)
        result['dispatch_succeeded'] = np.asarray([row['dispatch_succeeded'] for row in commands], dtype=bool)
        result['joint_response_diagnostics_json'] = np.asarray([
            json.dumps(_json(row.get('joint_response_diagnostics', {})), ensure_ascii=False)
            for row in commands], dtype=str)
        result['feedback_q_rad'] = np.asarray([row['feedback_q_rad'] for row in feedback]).reshape(-1, 7)
        for key, size in (('feedback_flange_pose_m_rad', 6), ('feedback_joints_at_pose_rad', 7),
                          ('feedback_pose_packet_monotonic_s', 3)):
            result[key] = np.asarray([row[key] for row in feedback]).reshape(-1, size)
        result['feedback_pose_joint_pairing_json'] = np.asarray([
            json.dumps(_json(row.get('feedback_pose_joint_pairing')), ensure_ascii=False)
            for row in feedback], dtype=str)
        result['feedback_pose_grouping_json'] = np.asarray([
            json.dumps(_json(row.get('feedback_pose_grouping')), ensure_ascii=False)
            for row in feedback], dtype=str)
        for key in ('feedback_joint_stamp_s', 'feedback_joint_sample_monotonic_s',
                    'feedback_read_started_monotonic_s', 'feedback_read_monotonic_s', 'feedback_read_wall_s'):
            result[key] = np.asarray([row[key] for row in feedback])
        result['feedback_sample_stage'] = np.asarray([row['sample_stage'] for row in feedback], dtype=str)
        result['feedback_stream_id'] = np.asarray([row['stream_id'] for row in feedback], dtype=str)
        result['feedback_joint_axis_sample_monotonic_s'] = np.asarray([
            row['feedback_joint_axis_sample_monotonic_s'] for row in feedback]).reshape(-1, 7)
        result['feedback_host_wall_to_monotonic_offset_s'] = np.asarray([
            row['feedback_host_wall_to_monotonic_offset_s'] for row in feedback])
        result['feedback_joint_packet_stamps_s'] = np.asarray(
            [row['feedback_joint_packet_stamps_s'] for row in feedback]).reshape(-1, 4)
        reference = np.full((len(feedback),7),np.nan)
        for i,row in enumerate(feedback):
            full,origin=row.get('_full_reference'),row.get('_policy_origin')
            if full is None or origin is None or not np.isfinite(row['feedback_joint_sample_monotonic_s']):
                continue
            # This diagnostic runs on export, after control. CAN receipt time
            # is the available alignment estimate, not encoder acquisition.
            for j,stamp in enumerate(row['feedback_joint_axis_sample_monotonic_s']):
                t=stamp-origin
                if np.isfinite(t) and full.execution_start<=t<=full.end:
                    reference[i,j]=full.sample_command_full(t)[0][j]
        result['feedback_reference_q_rad']=reference
        result['feedback_packet_aligned_error_rad']=result['feedback_q_rad']-reference
        return result

    def summary(self):
        a = self.arrays()
        good = a['dispatch_succeeded']
        differences = np.abs(a['raw_trajectory_q_rad']-a['policy_candidate_q_rad'])
        changed = np.isfinite(differences).all(axis=1) & (np.max(differences, axis=1, initial=0.) > 1e-9)
        controller_difference = np.abs(a['policy_candidate_q_rad']-a['dispatched_q_rad'])
        controller_changed = (np.isfinite(controller_difference).all(axis=1)
            & (np.max(controller_difference, axis=1, initial=0.) > 1e-9))
        return _json({'schema_version': 1, 'mode': self.mode,
            'joint_command_mode': self.joint_command_mode,
            'clock_domain': 'host_monotonic', 'stamp_clock_domain': self.stamp_clock_domain,
            'feedback_timestamp_source': self.timestamp_source,
            'target_identity':identity_dict(self.target_identity),
            'task_finished_monotonic':getattr(self,'task_finished_monotonic',None),
            'task_terminal':getattr(self,'task_terminal',None),
            'gripper_commands':_json(self.gripper_rows),
            'feedback_time_records': [_json({
                'target_identity':r.get('target_identity'),'feedback_time':r.get('feedback_time')}) for r in self.feedback_rows],
            'clock_mapping': 'Immutable decode-boundary mapping; missing legacy mapping remains unknown. Paired joint packets map to axes [12,34,56,7].',
            'feedback_time_limitation': 'Mapped CAN receipt time is not encoder acquisition time; joint_stamp is oldest constituent packet.',
            'host_wall_to_monotonic_offset_s': self.host_wall_to_monotonic_offset_s,
            'policy_origin_monotonic_s': self.policy_origin,
            'command_count': len(good), 'successful_dispatch_count': int(np.count_nonzero(good)),
            'feedback_read_count': len(a['feedback_q_rad']),
            'feedback_tracking': self.feedback_tracking(a),
            'raw_vs_policy_changed_count': int(np.count_nonzero(changed)),
            'policy_vs_dispatched_changed_count': int(np.count_nonzero(controller_changed)),
            'SDK_call_duration_s': {
                'median': float(np.median((a['dispatch_end_monotonic_s']-a['dispatch_start_monotonic_s'])[good]))
                    if np.any(good) else None,
                'maximum': float(np.max((a['dispatch_end_monotonic_s']-a['dispatch_start_monotonic_s'])[good]))
                    if np.any(good) else None},
            'edges': self.edges, 'edge_timing': self.edge_timing(a),
            'handoffs': self.handoffs, 'diagnostic_errors': list(self.errors),
            'collection': 'memory only during execution; compressed arrays and JSON written after sender join',
            'identification_limitation': 'Position commands and CAN receipt feedback do not directly identify internal firmware trajectory or motor torque.'})

    @staticmethod
    def feedback_tracking(arrays):
        error=np.abs(arrays['feedback_packet_aligned_error_rad'])
        return {'alignment':'per-axis mapped CAN receipt time; encoder sample time unknown',
            'reference':'certified full command reference, including preserved prefixes; not retimer output',
            'valid_samples_per_axis':[int(np.isfinite(error[:,j]).sum()) for j in range(7)],
            'p50_abs_rad':[float(np.nanmedian(error[:,j])) if np.isfinite(error[:,j]).any() else None for j in range(7)],
            'p95_abs_rad':[float(np.nanpercentile(error[:,j],95)) if np.isfinite(error[:,j]).any() else None for j in range(7)]}

    def edge_timing(self, arrays=None):
        """Locate existing goal tolerances diagnostically, without changing them."""
        a = self.arrays() if arrays is None else arrays
        result = []
        command_times = a['dispatch_start_monotonic_s']
        for key, edge in self.edges.items():
            selected = (a['stream_id'] == key) & a['dispatch_succeeded']
            indexes = np.flatnonzero(selected)
            if not len(indexes):
                continue
            tolerance = edge['position_tolerance_rad']
            goal = edge['q1_rad']
            def first_time(field):
                matching = indexes[np.max(np.abs(a[field][indexes]-goal), axis=1) <= tolerance]
                return command_times[matching[0]] if len(matching) else None
            first_sent_goal = first_time('dispatched_q_rad')
            first_candidate_goal = first_time('policy_candidate_q_rad')
            nominal_end = (self.policy_origin+edge['edge_started_policy_s']+edge['nominal_duration_s']
                           if self.policy_origin is not None else np.nan)
            handoffs = [h for h in self.handoffs if h['stream_id'] == key]
            handoff = handoffs[-1] if handoffs else None
            handoff_time = handoff['handoff_monotonic_s'] if handoff else None
            result.append({'stream_id': key, 'sample_stage': a['sample_stage'][indexes[0]],
                'first_dispatch_monotonic_s': command_times[indexes[0]],
                'nominal_end_monotonic_s': nominal_end,
                'nominal_duration_s': edge['nominal_duration_s'],
                'candidate_goal_first_dispatch_monotonic_s': first_candidate_goal,
                'dispatched_goal_first_monotonic_s': first_sent_goal,
                'handoff_monotonic_s': handoff_time,
                'nominal_end_to_goal_dispatch_s': (first_sent_goal-nominal_end
                    if first_sent_goal is not None else None),
                'goal_dispatch_to_feedback_handoff_s': (handoff_time-first_sent_goal
                    if handoff_time is not None and first_sent_goal is not None else None),
                'arrival_definition': 'Original position tolerance locates command goal; handoff retains original position AND feedback velocity predicate.',
                'nominal_clock_limitation': ('Accelerated offline policy time is not host wall time.'
                    if self.mode == 'offline_ideal_feedback' else None)})
        return result

    def write(self, directory):
        """One post-run write; recording errors never replace task outcomes."""
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/'execution_trace.npz', **self.arrays())
            (directory/'execution_trace_summary.json').write_text(
                json.dumps(self.summary(), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            return {'npz': str(directory/'execution_trace.npz'),
                    'summary': str(directory/'execution_trace_summary.json')}
        except Exception as error:
            return {'write_error': f'{type(error).__name__}: {error}'}


class _TracedDevice:
    def __init__(self, device, trace, stage):
        self._device, self._trace, self._stage = device, trace, stage

    def __getattr__(self, name):
        if name == 'snapshot_nowait':
            original = getattr(self._device, name)
            def snapshot_nowait(*args, **kwargs):
                begin = self._trace.clock()
                result = original(*args, **kwargs)
                end = self._trace.clock()
                self._trace.record_feedback(result, begin, end,
                    stage=self._trace._attempt(self._stage) or '')
                return result
            return snapshot_nowait
        return getattr(self._device, name)

    def snapshot(self, *args, **kwargs):
        begin = self._trace.clock()
        result = self._device.snapshot(*args, **kwargs)
        end = self._trace.clock()
        self._trace.record_feedback(result, begin, end,
                                    stage=self._trace._attempt(self._stage) or '')
        return result

    def grip(self, width, force):
        begin = self._trace.clock()
        succeeded = False
        try:
            result = self._device.grip(width, force)
            succeeded = True
            return result
        finally:
            self._trace.gripper_rows.append({'sdk_start_monotonic':begin,
                'sdk_finish_monotonic':self._trace.clock(), 'width_m':width,'force_n':force,
                'succeeded':succeeded,'target_identity':identity_dict(self._trace.target_identity),
                'physical_contact_time':None})

    def move_joints(self, q):
        begin = self._trace.clock()
        succeeded = False
        try:
            result = self._device.move_joints(q)
            succeeded = True
            return result
        finally:
            end = self._trace.clock()
            self._trace.record_dispatch(q, begin, end, succeeded=succeeded)

"""Sample certified paths in one sender; follow, confirm grip and brake in place."""
from types import SimpleNamespace
import numpy as np
from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.joint_stream_sender import DispatchWindowExpired
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.types import identity_dict
from .feedback_gates import (closure_feedback, GripConfirmation, RestArrival,
    TRAJECTORY_ARRIVAL_TOLERANCE_RAD, ARRIVAL_DWELL_S)
from .motion_profile import select_motion_profile


class CheckedWindowExpired(PickRejected, DispatchWindowExpired):
    """The sender may retry, but only after a new full geometry check."""


def fresh_feedback(state, now, previous=None):
    packet_keys = {'pose': ('end_pose_xy','end_pose_zrx','end_pose_ryrz'),
                   'joint': ('joint_12','joint_34','joint_56','joint_7'), 'gripper': ('gripper',)}
    for kind in ('pose', 'joint', 'gripper'):
        stamp = mapped_stamp(state, kind)
        if stamp is None or not 0 <= now-stamp <= .2:
            raise PickRejected('实际发送时反馈无效或过期')
        if previous is not None:
            if kind in previous and stamp < previous[kind]:
                raise PickRejected('设备反馈时钟倒退', details={'kind': kind, 'previous': previous[kind], 'current': stamp})
            evidence = state.get('feedback_time', {})
            epoch = tuple((evidence.get(key) or {}).get('clock_epoch') for key in packet_keys[kind])
            epoch_key = kind+'_clock_epoch'
            if epoch_key in previous and previous[epoch_key] != epoch:
                raise PickRejected('设备反馈时钟代次重置', details={
                    'kind':kind,'previous_epoch':previous[epoch_key],'current_epoch':epoch})
            previous[kind], previous[epoch_key] = stamp, epoch


def stream(executor, path, *, origin, epoch, sender_factory, reference=None, carrying=False):
    e = executor
    identity, version = e.owner, e.version
    if path.identity != identity or path.version != e.planner.version:
        raise PickRejected('规划目标或版本不属于当前任务')
    e.stage = path.stage
    e._emit('stage_started', {'stage': path.stage,
        'prealign_s': path.diagnostics.get('prealign_s', 0.),
        'rotate_first_s':path.diagnostics.get('rotate_first_s',
            0. if path.diagnostics.get('intercept_strategy')=='synchronized' else path.diagnostics.get('prealign_s',0.))})
    motion = select_motion_profile(e.config, payload=carrying)
    request = SimpleNamespace(generation=path.version,
        request_id=f'{e.trace.target_identity.task_id if e.trace.target_identity else identity}:{path.stage}',
        planned_target_time_s=None if reference is None else reference.contact_s)
    context = SimpleNamespace(env=SimpleNamespace(target_identity=e.trace.target_identity), _full_pick_request=request)
    close = {'sent': False, 'since': None, 'stamp': None, 'held': None}
    previous = {'tcp': None, 'stamp': None, 'endpoint_sent': False}
    advance = e.config.raw['command_advance_s']
    smooth = reference is not None and hasattr(reference, 'follow_end_s')
    scheduled_attempt = reference is not None and e.config.raw.get('closure_policy', 'feedback') == 'scheduled_attempt'
    follow_end = (reference.follow_end_s if smooth else path.duration)
    confirmation = None
    arrival = RestArrival(origin+path.duration, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
    from .return_feedback import ReturnFeedbackCheck, RETURN_STAGES
    feedback_check = ReturnFeedbackCheck(e.planner.model.check_feedback, e.clock,
        enabled=not carrying and reference is None and path.stage in RETURN_STAGES)

    def step(state, dispatch, dt):
        nonlocal confirmation
        e._active(identity, version, epoch)
        fresh_feedback(state, e.clock(), e.feedback_stamps)
        feedback_check.observe(state)
        if carrying or (close['held'] is not None and not scheduled_attempt):
            e.holding.check_holding(state)
        elapsed = dispatch+advance-origin
        if elapsed < 0:
            raise PickRejected('发送线程提前于本次轨迹预约')
        if elapsed > path.duration+e.config.settings.move_timeout_s:
            raise PickRejected('轨迹结束后反馈未到位', details={'stage': path.stage,
                'max_joint_error_rad': float(np.max(abs(np.asarray(state['joints'])-path.q1))),
                'tolerance_rad': arrival.tolerance_rad, 'required_stable_s': ARRIVAL_DWELL_S})
        q, velocity, acceleration = path.sample(elapsed)
        if e.command_history.q is None:
            e.command_history.validate_first(q, path.q0, limits=motion)
        tcp = pose_xyz_rpy(state['pose'])@e.config.tcp
        stamp = mapped_stamp(state, 'pose')
        if reference is not None:
            if not close['sent'] and abs(state['width_m']-e.config.settings.open_width_m) > .002:
                raise PickRejected('接近期间夹爪开口改变')
            eligible, gate = closure_feedback(reference, tcp, stamp, previous['tcp'], previous['stamp'],
                command_elapsed=elapsed, support=e.planner.support, gates=e.gates, follow_end_s=follow_end)
            should_close = elapsed >= reference.duration_s-1e-9 if scheduled_attempt else eligible
            if not close['sent'] and should_close:
                with e.command_lock:
                    e._active(identity, version, epoch)
                    e.motion_started = True
                    close_started = e.clock()
                    e.device.grip(0., e.config.settings.force_n)
                    close_completed = e.clock()
                # Feedback copied before this actual command cannot start the
                # physical 0.2 s confirmation dwell, even if it arrives late.
                close.update(sent=True, stamp=close_completed)
                confirmation = GripConfirmation(e.config.settings.open_width_m,
                    e.config.settings.min_force_n, close['stamp'])
                e._emit('close_during_follow', {'reference_time_s': elapsed,
                    'close_command_started_monotonic_s':close_started,
                    'close_command_completed_monotonic_s':close_completed,
                    'gripper_feedback_before_command_monotonic_s':mapped_stamp(state,'gripper'),
                    'feedback_pose_monotonic_s': stamp, 'closure_policy': 'scheduled_attempt' if scheduled_attempt else 'feedback',
                    'feedback_would_allow_close': bool(eligible), **gate})
            grip_stamp = mapped_stamp(state, 'gripper')
            if close['held'] is None and confirmation is not None and confirmation.observe(
                    state['width_m'], state['force_n'], grip_stamp):
                close['held'] = dict(state)
                e.holding.held_width = state['width_m']
                if smooth:
                    joint_stamp = mapped_stamp(state, 'joint')
                    e.planner.attach_measured(state, joint_stamp)
                    e._emit('payload_estimate_attached', {'pose_monotonic_s': stamp,
                        'joint_monotonic_s': joint_stamp, 'attachment_monotonic_s': joint_stamp,
                        'gripper_monotonic_s': grip_stamp, 'reference_time_s': elapsed,
                        'source': 'frozen_visual_prediction_and_latest_joint_fk', 'physical_proof': False})
            if previous['stamp'] is None or stamp > previous['stamp']:
                previous.update(tcp=tcp.copy(), stamp=stamp)
            # Process this feedback before testing the brake boundary: a fresh
            # sample completing the 0.2 s dwell at that boundary is valid.
            if elapsed >= follow_end and close['held'] is None and not scheduled_attempt:
                raise PickRejected('跟随窗口结束但夹持未确认', details={'close_sent': close['sent']})
        attached = carrying or (close['held'] is not None and smooth)
        actual = np.asarray(state['joints'])
        until = dispatch+1./e.config.settings.control_hz
        if reference is not None and close['held'] is None and elapsed < follow_end:
            until = min(until, origin-advance+follow_end)
        # Runtime checks protect each 100 Hz send slot, independent of the
        # configurable offline trajectory geometry sampling frequency.
        for sample_time in np.linspace(dispatch, until, 3):
            target_q = path.sample(sample_time+advance-origin)[0]
            n = max(1, int(np.ceil(np.max(abs(target_q-actual))/.008)))
            for u in np.linspace(0., 1., n+1):
                e.planner.valid(actual+(target_q-actual)*u, state['width_m'],
                    stamp=(stamp+(sample_time+advance-stamp)*u
                           if reference is not None and not attached and not scheduled_attempt else None),
                    contact=reference is not None and elapsed >= reference.duration_s,
                    payload=attached, allow_belt_support=smooth and attached)
        e.command_history.validate(q, dispatch, limits=motion)
        if elapsed >= path.duration and previous['endpoint_sent'] and arrival.observe(
                actual, path.q1, mapped_stamp(state, 'joint')):
            return None
        e.trace.prepare_command(context, policy_time_s=elapsed, candidate=q,
            stage=path.stage, stream_id=request.request_id, trajectory=path, edge_started=0.)
        return q, {'task_identity': identity_dict(e.trace.target_identity), 'trajectory_version': path.version,
            'time_base': 'host_monotonic', 'reference_time_s': elapsed,
            'prepared_from_monotonic_s': dispatch, 'prepared_until_monotonic_s': until,
            'phase': ('brake' if smooth and elapsed >= follow_end else
                      'follow' if reference is not None and elapsed >= reference.duration_s else path.stage),
            'raw_velocity_rad_s': velocity.tolist(), 'raw_acceleration_rad_s2': acceleration.tolist(),
            **motion.diagnostics()}

    def finalize(state, dispatch, dt, prepared_q, metadata):
        e._active(identity, version, epoch)
        if path.version != e.planner.version:
            raise PickRejected('实际发送前规划版本已过期')
        fresh_feedback(state, e.clock())
        if not metadata['prepared_from_monotonic_s']-1e-9 <= dispatch <= metadata['prepared_until_monotonic_s']+1e-9:
            raise CheckedWindowExpired('实际发送已超出本次几何检查时窗', details={'dispatch_monotonic_s': dispatch,
                'prepared_until_monotonic_s': metadata['prepared_until_monotonic_s']})
        elapsed = dispatch+advance-origin
        if reference is not None and elapsed >= follow_end and close['held'] is None and not scheduled_attempt:
            raise PickRejected('实际发送进入减速前夹持仍未确认')
        q, velocity, acceleration = path.sample(elapsed)
        if e.command_history.q is None:
            e.command_history.validate_first(q, path.q0, limits=motion)
        e.command_history.validate(q, dispatch, limits=motion)
        metadata.update(reference_time_s=elapsed, final_sample_monotonic_s=dispatch,
            phase=('brake' if smooth and elapsed >= follow_end else
                   'follow' if reference is not None and elapsed >= reference.duration_s else path.stage),
            raw_velocity_rad_s=velocity.tolist(), raw_acceleration_rad_s2=acceleration.tolist())
        e.trace.update_final_sample(elapsed, q, velocity, acceleration, dispatch)
        return q, metadata

    def validate_dispatch(state, dispatch, dt, q, metadata):
        e._active(identity, version, epoch)
        if path.version != e.planner.version:
            raise PickRejected('实际发送前规划版本已过期')
        fresh_feedback(state, dispatch)
        if not metadata['prepared_from_monotonic_s']-1e-9 <= dispatch <= metadata['prepared_until_monotonic_s']+1e-9:
            raise CheckedWindowExpired('实际发送已超出本次几何检查时窗', details={'dispatch_monotonic_s': dispatch,
                'prepared_until_monotonic_s': metadata['prepared_until_monotonic_s']})
        if reference is not None and dispatch+advance-origin >= follow_end and close['held'] is None and not scheduled_attempt:
            raise PickRejected('实际发送进入减速前夹持仍未确认')
        e.command_history.validate(q, dispatch, limits=motion)

    def after_send(dispatch, actual_dt, q, metadata):
        e.motion_started = True
        e.command_history.record(dispatch, actual_dt, q, metadata, limits=motion)
        previous['endpoint_sent'] = metadata['reference_time_s'] >= path.duration

    e.sender = sender_factory(e.device, step, after_send,
        control_hz=e.config.settings.control_hz, clock=e.clock, sleep=e.sleep,
        pause_requested=e.cancelled.is_set, initial_dispatch_time=origin-advance,
        finalize_send=finalize, validate_dispatch=validate_dispatch).start()
    try:
        while not e.sender.wait_stopped(.05):
            e._active(identity, version, epoch)
        e.sender.join()
        report = e.sender.snapshot()
        for item in feedback_check.resumed:
            e._emit('return_feedback_rechecked', {'stage': path.stage, **item})
        for event in getattr(report, 'events', ()):
            if event.get('event') == 'joint_stream_window_retry':
                e._emit('dispatch_window_retry', {'stage': path.stage,
                    'attempt': event['attempt'], 'monotonic_s': event['time'],
                    **event['details']})
        if report.exception is not None:
            if isinstance(report.exception, DispatchWindowExpired) and not isinstance(report.exception, PickRejected):
                raise CheckedWindowExpired(str(report.exception),
                    details=getattr(report.exception, 'details', {})) from report.exception
            raise report.exception
        e._active(identity, version, epoch)
        if not previous['endpoint_sent']:
            raise PickRejected('发送器退出但未完整交付轨迹终点')
        e.rest_anchor = path.q1.copy()
        e._emit('stage_finished', {'stage': path.stage, 'commands': report.command_count,
            'trajectory_duration_s': path.duration, 'diagnostics': path.diagnostics})
        if reference is not None:
            e.holding.held_width = None if close['held'] is None else close['held']['width_m']
            e._emit('intercept_attempt_finished', {'close_sent': close['sent'],
                'grip_confirmed': close['held'] is not None,
                'closure_policy': 'scheduled_attempt' if scheduled_attempt else 'feedback'})
        return report.last_feedback
    finally:
        e.sender.request_stop()
        e.sender.join()
        e.sender = None

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
from .wait_precision import enabled_for, finish_wait, JS_ENDPOINT_HOLD_S, MAX_CORRECTION_RAD
from .runtime_sweep import (SweepSettings, SweepService, PeriodicSweep, scene_snapshot,
                            request_for, sweep)


class CheckedWindowExpired(PickRejected, DispatchWindowExpired):
    """The sender may retry, but only after a new full geometry check."""


def fresh_feedback(state, now, previous=None):
    packet_keys = {'pose': ('end_pose_xy','end_pose_zrx','end_pose_ryrz'),
                   'joint': ('joint_12','joint_34','joint_56','joint_7'), 'gripper': ('gripper',)}
    updates = {}
    for kind in ('pose', 'joint', 'gripper'):
        stamp = mapped_stamp(state, kind)
        if stamp is None or not 0 <= now-stamp <= .2:
            raise PickRejected('实际发送时反馈无效或过期', details={
                'kind': kind, 'feedback_monotonic_s': stamp,
                'checked_monotonic_s': now,
                'feedback_age_s': None if stamp is None else now-stamp,
                'maximum_age_s': .2})
        if previous is not None:
            if kind in previous and stamp < previous[kind]:
                raise PickRejected('设备反馈时钟倒退', details={'kind': kind, 'previous': previous[kind], 'current': stamp})
            evidence = state.get('feedback_time', {})
            epoch = tuple((evidence.get(key) or {}).get('clock_epoch') for key in packet_keys[kind])
            epoch_key = kind+'_clock_epoch'
            if epoch_key in previous and previous[epoch_key] != epoch:
                raise PickRejected('设备反馈时钟代次重置', details={
                    'reason':'feedback_clock_epoch_changed',
                    'kind':kind,'previous_epoch':previous[epoch_key],'current_epoch':epoch})
            updates[kind], updates[epoch_key] = stamp, epoch
    if previous is not None:
        # Commit only a fully checked snapshot. Recovery needs the last good
        # receive-boundary references, never a partially accepted new epoch.
        updates['clock_mapping_evidence'] = {
            key: dict(value) for key, value in state.get('feedback_time', {}).items()
            if isinstance(value, dict)}
        previous.update(updates)


def stream(executor, path, *, origin, epoch, sender_factory, reference=None, carrying=False,
           release_on_arrival=False):
    e = executor
    identity, version = e.owner, e.version
    if path.identity != identity or path.version != e.planner.version:
        raise PickRejected('规划目标或版本不属于当前任务')
    if reference is not None and 'phase_timing' in e.config.raw:
        from .phase_timing import verify_plan
        if not np.isfinite(origin) or abs(origin-reference.start_s) > 1e-9:
            raise PickRejected('阶段时长与发送轨迹时钟起点不一致')
        report = verify_plan(e.config, path, reference, for_execution=True)
        e._emit('phase_timing_checked', report)
    if reference is not None and 'execution_delay_window' in e.config.raw:
        from .execution_delay_window import verify_plan
        if not np.isfinite(origin) or abs(origin-reference.start_s) > 1e-9:
            raise PickRejected('执行时延窗口与发送轨迹时钟起点不一致')
        report = verify_plan(e.config, path, reference)
        e._emit('execution_delay_window_checked', report)
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
    arrival = RestArrival(origin+path.duration, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD,
        dwell_s=0. if release_on_arrival else ARRIVAL_DWELL_S)
    from .return_feedback import ReturnFeedbackCheck, RETURN_STAGES, TRANSPORT_STAGES
    feedback_check = ReturnFeedbackCheck(e.planner.model.check_feedback, e.clock,
        enabled=reference is None and ((not carrying and path.stage in RETURN_STAGES) or
                                      (carrying and path.stage in TRANSPORT_STAGES)))
    precision = enabled_for(e, path, reference=reference, carrying=carrying)
    sweep_settings = SweepSettings.from_raw(e.config.raw, e.config.settings.control_hz)
    monitor = None

    def geometry_request(state, dispatch, attached):
        return request_for(e.planner, state, dispatch=dispatch, origin=origin, advance=advance,
            control_hz=e.config.settings.control_hz, reference=reference, attached=attached,
            smooth=smooth, scheduled_attempt=scheduled_attempt, follow_end=follow_end,
            grip_confirmed=close['held'] is not None)

    def geometry_guard(now):
        if monitor is not None:
            if e.planner.scene_revision != monitor.snapshot['scene_revision']:
                raise PickRejected('发送期间碰撞场景版本已改变')
            return monitor.guard(now)

    def joint_bounds(q):
        if monitor is not None:
            model = e.planner.model
            if np.any(q < model.command_lower-1e-10) or np.any(q > model.command_upper+1e-10):
                raise PickRejected('插值发送目标超出关节规划范围')

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
                'tolerance_rad': arrival.tolerance_rad, 'required_stable_s': arrival.dwell_s})
        q, velocity, acceleration = path.sample(elapsed)
        joint_bounds(q)
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
                # physical confirmation dwell, even if it arrives late.
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
            # sample completing the confirmation dwell at that boundary is valid.
            if elapsed >= follow_end and close['held'] is None and not scheduled_attempt:
                raise PickRejected('跟随窗口结束但夹持未确认', details={'close_sent': close['sent']})
        attached = carrying or (close['held'] is not None and smooth)
        actual = np.asarray(state['joints'])
        request_state = geometry_request(state, dispatch, attached)
        if monitor is None:
            # Compatibility mode: the original full sweep in each send slot.
            sweep(e.planner, path, request_state)
            until = request_state['until_s']
        else:
            # No process submission, serialization, waiting, or geometry work
            # on this thread. Continue the already checked continuous curve.
            until = geometry_guard(dispatch)
            monitor.offer(request_state)
        e.command_history.validate(q, dispatch, limits=motion)
        if elapsed >= path.duration and previous['endpoint_sent'] and arrival.observe(
                actual, path.q1, mapped_stamp(state, 'joint')):
            if release_on_arrival:
                e._emit('drop_arrival_confirmed', {
                    'joint_feedback_monotonic_s': mapped_stamp(state, 'joint'),
                    'scheduled_endpoint_monotonic_s': origin+path.duration,
                    'max_joint_error_rad':float(np.max(abs(actual-path.q1))),
                    'arrival_dwell_s':arrival.dwell_s})
            return None
        if (precision and previous['endpoint_sent'] and elapsed >= path.duration+JS_ENDPOINT_HOLD_S
                and np.max(abs(actual-path.q1)) <= MAX_CORRECTION_RAD):
            # Stop and join this JS sender before testing stillness or issuing
            # the single J precision target. This is not an arrival success.
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
        if monitor is not None:
            metadata['prepared_until_monotonic_s'] = geometry_guard(dispatch)
        if not metadata['prepared_from_monotonic_s']-1e-9 <= dispatch <= metadata['prepared_until_monotonic_s']+1e-9:
            raise CheckedWindowExpired('实际发送已超出本次几何检查时窗', details={'dispatch_monotonic_s': dispatch,
                'prepared_until_monotonic_s': metadata['prepared_until_monotonic_s']})
        elapsed = dispatch+advance-origin
        if reference is not None and elapsed >= follow_end and close['held'] is None and not scheduled_attempt:
            raise PickRejected('实际发送进入减速前夹持仍未确认')
        q, velocity, acceleration = path.sample(elapsed)
        joint_bounds(q)
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
        if monitor is not None:
            metadata['prepared_until_monotonic_s'] = geometry_guard(dispatch)
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

    own_service = False
    sender_options = {}
    cache_counts = {'missing_sender_reads': 0, 'monitor_feedback_used': 0}
    try:
        if sweep_settings.mode == 'process_periodic':
            service = e.sweep_service
            if service is None:
                service = SweepService(e.config)
                own_service = True
            # This observer has its own clock high-water marks; it must not
            # race the sender's acquisition order or depend on its first tick.
            monitor_feedback_stamps = {}
            # Single supervisor writer; publish whole validated snapshots.
            # The sender reads this mailbox without waiting for SDK/parser locks.
            monitor_feedback = [None]
            def read_monitor_feedback():
                state = e.device.snapshot_nowait()
                if state is not None:
                    fresh_feedback(state, e.clock(), monitor_feedback_stamps)
                    monitor_feedback[0] = state
                return state
            monitor = PeriodicSweep(service, scene_snapshot(e.planner, path), sweep_settings, clock=e.clock,
                collision_sink=e.planner.collision_sink,
                collision_frame=e.planner.target.detection.get('frame'),
                feedback_reader=read_monitor_feedback)
            initial_state = e.device.snapshot()
            fresh_feedback(initial_state, e.clock(), e.feedback_stamps)
            monitor_feedback_stamps.update(e.feedback_stamps)
            monitor.prime(geometry_request(initial_state, max(e.clock(), origin-advance), carrying))
            e._active(identity, version, epoch)
            latest_feedback = [initial_state]
            def read_available_feedback():
                state = e.device.snapshot_nowait()
                if state is not None:
                    latest_feedback[0] = state
                else:
                    cache_counts['missing_sender_reads'] += 1
                    candidate = monitor_feedback[0]
                    # A delayed supervisor read must not replace a newer
                    # sender sample. Compare every packet group, not host read
                    # completion time; all original packet stamps stay intact.
                    if candidate is not None and all(
                            mapped_stamp(candidate, kind) >= mapped_stamp(latest_feedback[0], kind)
                            for kind in ('pose', 'joint', 'gripper')):
                        latest_feedback[0] = candidate
                        cache_counts['monitor_feedback_used'] += 1
                # Keep the packet stamps. Missing reads do not become new
                # feedback, satisfy grip dwell, or extend its 200 ms lifetime.
                return latest_feedback[0]
            sender_options['feedback_reader'] = read_available_feedback
            sender_options['max_prepare_lookahead_s'] = 1./e.config.settings.control_hz
        e.sender = sender_factory(e.device, step, after_send,
            control_hz=e.config.settings.control_hz, clock=e.clock, sleep=e.sleep,
            pause_requested=e.cancelled.is_set, initial_dispatch_time=origin-advance,
            finalize_send=finalize, validate_dispatch=validate_dispatch, **sender_options).start()
        if reference is not None and getattr(e, 'carry_service', None) is not None:
            e.start_carry_prediction(path, reference)
        while not e.sender.wait_stopped(.002 if monitor is not None else .05):
            e._active(identity, version, epoch)
            if monitor is not None:
                monitor.pump()
        e.sender.join()
        report = e.sender.snapshot()
        commands = [item for item in getattr(report, 'events', ())
            if item.get('event') == 'joint_stream_command']
        if commands:
            timings = {}
            for key in ('gap_s', 'step_compute_s', 'feedback_receive_s', 'dispatch_lateness_s'):
                values = [item[key] for item in commands if item.get(key) is not None]
                if values:
                    timings[key] = dict(zip(('p50_s', 'p95_s', 'max_s'),
                        map(float, np.quantile(values, [.5, .95, 1.]))))
            e._emit('dispatch_timing', {'stage': path.stage,
                'command_count': report.command_count, 'timings': timings})
        for item in feedback_check.resumed:
            e._emit('transport_feedback_rechecked' if carrying else 'return_feedback_rechecked',
                    {'stage': path.stage, **item})
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
    finally:
        if e.sender is not None:
            e.sender.request_stop()
            e.sender.join()
            e.sender = None
        if monitor is not None:
            try:
                monitor.close()
            finally:
                e._emit('runtime_sweep_timing', {'stage': path.stage, **monitor.diagnostics(),
                    'sender_feedback_cache': cache_counts})
                if own_service:
                    service.close()
    state = finish_wait(e, path, epoch=epoch) if precision else report.last_feedback
    e.rest_anchor = path.q1.copy()
    e._emit('stage_finished', {'stage': path.stage, 'commands': report.command_count,
        'trajectory_duration_s': path.duration, 'diagnostics': path.diagnostics})
    if reference is not None:
        e.holding.held_width = None if close['held'] is None else close['held']['width_m']
        e._emit('intercept_attempt_finished', {'close_sent': close['sent'],
            'grip_confirmed': close['held'] is not None,
            'closure_policy': 'scheduled_attempt' if scheduled_attempt else 'feedback'})
    return state

"""Keep the last certified rest command while a worker prepares a route.

The worker owns the planner's mutable kinematic model. This sender deliberately
uses only feedback, the immutable rest anchor and the execution command ledger.
"""
from types import SimpleNamespace
from threading import Event
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .streaming import fresh_feedback
from .trajectory import TimedPath
from .motion_profile import select_motion_profile
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp


class StationaryHold:
    def __init__(self, executor, sender_factory, *, epoch, carrying, payload_motion=None):
        self.e, self.factory = executor, sender_factory
        self.epoch, self.carrying = epoch, carrying
        self.payload_motion = carrying if payload_motion is None else payload_motion

    def __enter__(self):
        e = self.e
        identity, version = e.owner, e.version
        anchor = np.asarray(e.rest_anchor).copy()
        guard = getattr(e, 'brake_height_guard', None)
        pending = getattr(e,'pending_rest_arrival',None)
        self.arrival_ready = Event()
        if pending is None:
            self.arrival_ready.set()
        motion = select_motion_profile(e.config, payload=self.payload_motion)
        path = TimedPath(lambda t, d=0: anchor.copy() if d == 0 else np.zeros(7),
                         identity, e.planner.version, 'stationary_hold', 0., 0.)
        context = SimpleNamespace(env=SimpleNamespace(target_identity=e.trace.target_identity),
            _full_pick_request=SimpleNamespace(generation=path.version, request_id=f'{identity}:hold:{e.stage}',
                                               planned_target_time_s=None))

        def step(state, dispatch, dt):
            e._active(identity, version, self.epoch)
            fresh_feedback(state, e.clock(), e.feedback_stamps)
            if guard is not None and guard.armed:
                guard.observe(state, 'stationary_hold')
                guard.check_command(anchor)
            if np.max(abs(np.asarray(state['joints'])-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('静止连接期间实际关节偏离已认证终点')
            if self.carrying:
                e.holding.check_holding(state)
            if pending is not None and not self.arrival_ready.is_set():
                if e.clock() > pending.not_before+e.config.settings.move_timeout_s:
                    raise PickRejected('搬运复核期间减速终点到位确认超时')
                if pending.observe(state['joints'],anchor,mapped_stamp(state,'joint')):
                    self.arrival_ready.set()
                    e._emit('carry_arrival_overlap_confirmed', {
                        'joint_feedback_monotonic_s':mapped_stamp(state,'joint'),
                        'first_in_tolerance_monotonic_s':pending.since,
                        'required_stable_s':pending.dwell_s,
                        'actual_stable_s':mapped_stamp(state,'joint')-pending.since})
            e.command_history.validate(anchor, dispatch, limits=motion)
            e.trace.prepare_command(context, policy_time_s=0., candidate=anchor,
                stage='stationary_hold', stream_id=context._full_pick_request.request_id,
                trajectory=path, edge_started=0.)
            return anchor, {'phase': 'stationary_hold', 'trajectory_version': path.version,
                            'time_base': 'host_monotonic', **motion.diagnostics()}

        def finalize(state, dispatch, dt, prepared_q, metadata):
            e._active(identity, version, self.epoch)
            fresh_feedback(state, e.clock())
            if np.max(abs(np.asarray(state['joints'])-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('静止连接期间实际关节偏离已认证终点')
            if self.carrying:
                e.holding.check_holding(state)
            e.command_history.validate(anchor, dispatch, limits=motion)
            return anchor.copy(), metadata

        def after(dispatch, dt, q, metadata):
            if guard is not None and guard.armed:
                guard.record_sent(q)
            e.command_history.record(dispatch, dt, q, metadata, limits=motion)

        def validate_dispatch(state, dispatch, dt, q, metadata):
            e._active(identity, version, self.epoch)
            fresh_feedback(state, dispatch)
            if guard is not None and guard.armed:
                guard.observe(state, 'stationary_hold')
                guard.check_command(q)
            if np.max(abs(np.asarray(state['joints'])-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('静止连接期间实际关节偏离已认证终点')
            if self.carrying:
                e.holding.check_holding(state)
            e.command_history.validate(q, dispatch, limits=motion)

        self.sender = self.factory(e.device, step, after,
            control_hz=e.config.settings.control_hz, clock=e.clock, sleep=e.sleep,
            pause_requested=e.cancelled.is_set, finalize_send=finalize,
            validate_dispatch=validate_dispatch).start()
        e.sender = self.sender
        return self

    def check(self):
        report = self.sender.snapshot()
        if report.exception is not None:
            raise report.exception
        if not report.running:
            raise PickRejected('静止连接发送器意外退出')

    def __exit__(self, kind, error, traceback):
        try:
            if error is None:
                # A fast validation must not bypass the original arrival
                # dwell. A slow validation already overlaps all of it.
                while not self.arrival_ready.is_set():
                    self.check()
                    self.e.sleep(.002)
                self.check()
        finally:
            self.sender.request_stop()
            self.sender.join()
            self.e.sender = None
            self.e.pending_rest_arrival = None
        if error is None and self.sender.snapshot().exception is not None:
            raise self.sender.snapshot().exception

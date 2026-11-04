"""Keep the last certified rest command while a worker prepares a route.

The worker owns the planner's mutable kinematic model. This sender deliberately
uses only feedback, the immutable rest anchor and the execution command ledger.
"""
from types import SimpleNamespace
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .streaming import fresh_feedback
from .trajectory import TimedPath
from .motion_profile import select_motion_profile
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD


class StationaryHold:
    def __init__(self, executor, sender_factory, *, epoch, carrying, payload_motion=None):
        self.e, self.factory = executor, sender_factory
        self.epoch, self.carrying = epoch, carrying
        self.payload_motion = carrying if payload_motion is None else payload_motion

    def __enter__(self):
        e = self.e
        identity, version = e.owner, e.version
        anchor = np.asarray(e.rest_anchor).copy()
        motion = select_motion_profile(e.config, payload=self.payload_motion)
        path = TimedPath(lambda t, d=0: anchor.copy() if d == 0 else np.zeros(7),
                         identity, e.planner.version, 'stationary_hold', 0., 0.)
        context = SimpleNamespace(env=SimpleNamespace(target_identity=e.trace.target_identity),
            _full_pick_request=SimpleNamespace(generation=path.version, request_id=f'{identity}:hold:{e.stage}',
                                               planned_target_time_s=None))

        def step(state, dispatch, dt):
            e._active(identity, version, self.epoch)
            fresh_feedback(state, e.clock(), e.feedback_stamps)
            if np.max(abs(np.asarray(state['joints'])-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('静止连接期间实际关节偏离已认证终点')
            if self.carrying:
                e.holding.check_holding(state)
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
            e.command_history.record(dispatch, dt, q, metadata, limits=motion)

        def validate_dispatch(state, dispatch, dt, q, metadata):
            e._active(identity, version, self.epoch)
            fresh_feedback(state, dispatch)
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
        self.sender.request_stop()
        self.sender.join()
        self.e.sender = None
        if error is None:
            fault = self.sender.snapshot().exception
            if fault is not None:
                raise fault

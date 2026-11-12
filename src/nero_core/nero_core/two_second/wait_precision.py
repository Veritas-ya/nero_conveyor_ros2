"""Bounded, empty-arm J positioning after the JS return sender has joined.

The destination remains the last certified wait command. J uses manufacturer
internal timing: the sampled joint connections below check geometry, not that
controller's continuous trajectory or acceleration. No control gain is changed.
"""
from collections import deque
from types import SimpleNamespace
import numpy as np

from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.joint_stream_sender import FeedbackRecheckRequired
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import JOINT_ARRIVAL_TOLERANCE_RAD, RestArrival
from .motion_profile import select_motion_profile
from .return_feedback import RETURN_STAGES, ReturnFeedbackCheck


JS_ENDPOINT_HOLD_S = .5
MAX_CORRECTION_RAD = float(np.deg2rad(3.))
STILL_WINDOW_S = .15
STILL_RANGE_RAD = float(np.deg2rad(.05))
MAX_SAMPLE_GAP_S = .05
PRECISION_TIMEOUT_S = 2.


def enabled_for(e, path, *, reference, carrying):
    return bool(e.config.raw.get('return_wait_precision_enabled', False)
        and e.config.raw.get('hardware', {}).get('joint_command_mode') == 'js'
        and reference is None and not carrying and path.stage in RETURN_STAGES
        and np.max(abs(path.q1-e.config.standby)) <= 1e-8)


class JointStillness:
    """Fresh position samples spanning a window; not an acceleration estimate."""
    def __init__(self):
        self.samples = deque()

    def observe(self, q, stamp):
        if self.samples:
            previous = self.samples[-1][0]
            if stamp < previous:
                raise PickRejected('等待位精定位反馈时钟倒退')
            if stamp == previous:
                return False
            if stamp-previous > MAX_SAMPLE_GAP_S:
                self.samples.clear()
        self.samples.append((stamp, np.asarray(q).copy()))
        while len(self.samples) > 1 and self.samples[1][0] <= stamp-STILL_WINDOW_S:
            self.samples.popleft()
        return (len(self.samples) >= 3
            and stamp-self.samples[0][0] >= STILL_WINDOW_S-1e-9
            and np.max(np.ptp([item[1] for item in self.samples], axis=0)) <= STILL_RANGE_RAD)


def finish_wait(e, path, *, epoch):
    """Send at most one unchanged J target, only from a checked still state."""
    from .streaming import fresh_feedback

    if not enabled_for(e, path, reference=None, carrying=False):
        raise PickRejected('当前路径未启用等待位精定位')
    if e.sender is not None:
        raise PickRejected('返回发送器尚未退出，禁止切换等待位精定位')
    identity, version = e.owner, e.version
    goal = np.asarray(path.q1).copy()
    if (e.command_history.q is None or
            np.max(abs(e.command_history.q-goal)) > 1e-8):
        raise PickRejected('等待位精定位目标不是最后已发送的返回终点')
    began = e.clock()
    deadline = began+min(PRECISION_TIMEOUT_S, e.config.settings.move_timeout_s)
    e.stage = 'return_wait_precision'
    e._emit('return_wait_precision_started', {
        'source_stage': path.stage, 'max_correction_deg': float(np.rad2deg(MAX_CORRECTION_RAD)),
        'arrival_tolerance_rad': JOINT_ARRIVAL_TOLERANCE_RAD,
        'stable_window_s': STILL_WINDOW_S, 'stable_range_rad': STILL_RANGE_RAD,
        'timeout_s': deadline-began, 'trajectory_source': 'manufacturer_internal_smoothing',
        'controller_trajectory_certified': False})
    last_error = None
    feedback_check = ReturnFeedbackCheck(e.planner.model.check_feedback, e.clock, enabled=True)

    def check_active():
        e._active(identity, version, epoch)
        if e.planner.version != path.version:
            raise PickRejected('等待位精定位路径版本已过期')
        if e.clock() >= deadline:
            raise PickRejected('等待位精定位未在时限内稳定到位', details={
                'reason': 'return_wait_precision_timeout', 'waited_s': e.clock()-began,
                'last_target_error_rad': last_error,
                'tolerance_rad': JOINT_ARRIVAL_TOLERANCE_RAD})

    def read():
        nonlocal last_error
        check_active()
        state = e.device.snapshot_nowait()
        if state is None:
            return None
        fresh_feedback(state, e.clock(), e.feedback_stamps)
        try:
            feedback_check.observe(state)
        except FeedbackRecheckRequired:
            return None
        while feedback_check.resumed:
            e._emit('return_feedback_rechecked', {'stage': e.stage, **feedback_check.resumed.pop(0)})
        q = np.asarray(state['joints'])
        if q.shape != (7,) or not np.isfinite(q).all():
            raise PickRejected('等待位精定位关节反馈无效')
        width, force = float(state['width_m']), float(state['force_n'])
        if (not np.isfinite([width, force]).all() or force >= e.config.settings.min_force_n
                or not (abs(width-e.config.settings.open_width_m) <= .002 or 0 <= width <= .003)):
            raise PickRejected('等待位精定位无法确认空载，禁止继续定位')
        last_error = float(np.max(abs(q-goal)))
        if last_error > MAX_CORRECTION_RAD+1e-9:
            raise PickRejected('等待位误差超过3度精定位范围', details={
                'max_joint_error_rad': last_error, 'maximum_rad': MAX_CORRECTION_RAD})
        e.planner.valid(q, width, payload=False)
        return state

    def check_connection(q0, q1, width):
        steps = max(1, int(np.ceil(np.max(abs(q1-q0))/.004)))
        for u in np.linspace(0., 1., steps+1):
            check_active()
            e.planner.valid(q0+(q1-q0)*u, width, payload=False)

    def settled():
        still = JointStillness()
        while True:
            state = read()
            if state is not None and still.observe(state['joints'], mapped_stamp(state, 'joint')):
                return state
            e.sleep(.005)

    state = settled()
    start = np.asarray(state['joints']).copy()
    if last_error <= JOINT_ARRIVAL_TOLERANCE_RAD:
        e._emit('return_wait_precision_finished', {'joint_commands': 0,
            'max_joint_error_rad': last_error, 'waited_s': e.clock()-began})
        e.stage = path.stage
        return state
    check_connection(start, goal, state['width_m'])
    # The geometry check can take time. Require another measured sample before
    # mode selection, and reject a moved or no-longer-fresh starting state.
    current = read()
    while current is None:
        e.sleep(.005)
        current = read()
    if np.max(abs(np.asarray(current['joints'])-start)) > STILL_RANGE_RAD:
        raise PickRejected('等待位精定位检查期间实际起点改变，未切换J模式')
    start = np.asarray(current['joints']).copy()
    check_connection(start, goal, current['width_m'])
    motion = select_motion_profile(e.config, payload=False)
    context = SimpleNamespace(env=SimpleNamespace(target_identity=e.trace.target_identity),
        _full_pick_request=SimpleNamespace(generation=path.version,
            request_id=f'{identity}:return_wait_precision:move_j', planned_target_time_s=None),
        _safe_edge_reference_diagnostics={'joint_command_api': 'move_j',
            'trajectory_source': 'manufacturer_internal_smoothing',
            'controller_trajectory_certified': False})
    try:
        with e.command_lock:
            check_active()
            fresh_feedback(current, e.clock())
            e.device.configure_joint_command_mode('j')
            dispatch = e.clock()
            e.command_history.validate(goal, dispatch, limits=motion)
            e.trace.prepare_command(context, policy_time_s=0., candidate=goal,
                stage=e.stage, stream_id=context._full_pick_request.request_id, edge_started=0.)
            e._active(identity, version, epoch)
            e.device.move_joints(goal)
            e.motion_started = True
            e.command_history.record(dispatch, dispatch-e.command_history.dispatch, goal,
                {'phase': 'return_wait_precision', 'joint_command_api': 'move_j'}, limits=motion)
        e._emit('return_wait_precision_target_sent', {'joint_command_api': 'move_j',
            'target_joints_rad': goal.tolist(), 'measured_start_joints_rad': start.tolist(),
            'max_correction_rad': float(np.max(abs(goal-start))), 'joint_commands': 1})
        arrival = RestArrival(dispatch, tolerance_rad=JOINT_ARRIVAL_TOLERANCE_RAD)
        still = JointStillness()
        previous = start
        lower = np.minimum(start, goal)-STILL_RANGE_RAD
        upper = np.maximum(start, goal)+STILL_RANGE_RAD
        while True:
            state = read()
            if state is not None:
                actual = np.asarray(state['joints'])
                if np.any(actual < lower) or np.any(actual > upper):
                    raise PickRejected('等待位J精定位反馈超出小角度修正范围')
                check_connection(previous, actual, state['width_m'])
                previous = actual.copy()
                stamp = mapped_stamp(state, 'joint')
                reached = arrival.observe(actual, goal, stamp)
                stationary = still.observe(actual, stamp)
                if reached and stationary:
                    e._emit('return_wait_precision_finished', {'joint_commands': 1,
                        'max_joint_error_rad': last_error, 'waited_s': e.clock()-began})
                    e.stage = path.stage
                    return state
            e.sleep(.005)
    finally:
        # Local API selection only. The next checked trajectory sends its own
        # JS mode; no compensating target or gripper command is issued here.
        e.device.configure_joint_command_mode(e.config.raw['hardware']['joint_command_mode'])

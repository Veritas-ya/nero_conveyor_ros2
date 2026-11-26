"""Resolve an interrupted close or reconfirm a payload before recovery transport."""
import numpy as np

from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import GRIP_CONFIRMATION_DWELL_S, TRAJECTORY_ARRIVAL_TOLERANCE_RAD


MAX_SAMPLE_GAP_S = .05
GRIP_FAILURES = frozenset(('grip_unconfirmed', 'grip_width_out_of_range', 'grip_force_lost'))


class ClosingGrip:
    """Keep contact-onset feedback for this one close command and frozen target."""
    def __init__(self, target, closed_at, settings):
        self.target = target
        self.closed_at = self.last_stamp = closed_at
        self.settings = settings
        self.loaded = False
        self.contact = None

    def observe(self, state):
        grip = mapped_stamp(state, 'gripper')
        if grip is None or grip <= self.last_stamp:
            return
        prior, self.last_stamp = self.last_stamp, grip
        loaded = (.003 < state['width_m'] < self.settings.open_width_m-.004
                  and state['force_n'] >= self.settings.min_force_n)
        if not loaded:
            self.contact = None
        elif not self.loaded:
            joint = mapped_stamp(state, 'joint')
            # Localize the first observed contact, never a late stationary
            # sample after an unobserved interval on a moving belt.
            if (grip-prior <= MAX_SAMPLE_GAP_S and joint is not None and joint > self.closed_at
                    and abs(joint-grip) <= MAX_SAMPLE_GAP_S):
                self.contact = dict(state)
        self.loaded = loaded

    def mark_collision(self, error):
        kind = error.details.get('kind')
        if (kind in ('scene_collision', 'robot_workpiece_point_cloud', 'belt_plane_penetration',
                     'payload_scene_collision', 'bin_departure_released_workpiece')
                or str(error).startswith('模型自碰撞：')):
            error.closing_grip = self


def reconfirm_payload(e, state, failure, *, epoch):
    """Observe at rest. Bind missing geometry only to feedback at contact onset."""
    from .streaming import fresh_feedback
    closing = getattr(failure, 'closing_grip', None)
    if closing is not None and closing.target is not e.planner.target:
        raise PickRejected('闭爪复核与当前冻结工件不一致')
    if closing is None and ((getattr(failure, 'details', None) or {}).get('reason') not in GRIP_FAILURES
            or e.planner.payload_local is None):
        return state
    previous_width = e.holding.held_width
    anchor = np.asarray(e.command_history.q if e.command_history.q is not None
                        else state['joints'], float)
    began = e.clock()
    deadline = began + .5
    window, last_grip, last_joint = [], None, None
    width = force = None
    outcome = None
    stamps = dict(e.feedback_stamps)
    e.stage = 'recovery_grip_checking'
    e._emit('recovery_grip_wait', {'timeout_s': .5,
        'required_stable_s': GRIP_CONFIRMATION_DWELL_S,
        'previous_width_m': previous_width, 'motion_commands_sent': 0,
        'trigger': 'post_close_collision' if closing else 'grip_failure'})
    while True:
        e._active(e.owner, e.version, epoch)
        if e.clock() >= deadline:
            raise PickRejected('恢复时夹持仍未稳定；保持夹爪，等待人工处理', details={
                'reason': 'recovery_grip_unstable', 'waited_s': e.clock()-began,
                'last_width_m': width, 'last_force_n': force})
        if state is not None:
            fresh_feedback(state, e.clock(), stamps)
            e.planner.model.check_feedback(state)
            width, force = float(state['width_m']), float(state['force_n'])
            if (not np.isfinite([width, force]).all() or not 0. <= width <= .11 or force < 0.):
                raise PickRejected('恢复时夹爪反馈无效')
            q = np.asarray(state['joints'], float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise PickRejected('恢复时关节反馈无效')
            if np.max(abs(q-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                raise PickRejected('夹持复核期间实际关节距离最后指令超过1度')
            if closing is not None:
                closing.observe(state)
            elif width <= .003:
                return state  # Closed end-stop force is not evidence of a grasp.
            grip, joint = mapped_stamp(state, 'gripper'), mapped_stamp(state, 'joint')
            if (grip > began and joint > began and
                    (last_grip is None or grip > last_grip) and
                    (last_joint is None or joint > last_joint)):
                stamp = min(grip, joint)
                last_grip, last_joint = grip, joint
                candidate = .003 < width < e.config.settings.open_width_m-.004
                candidate = candidate and force >= e.config.settings.min_force_n
                observed = 'loaded' if candidate else ('empty' if closing and width <= .003 else None)
                if observed != outcome:
                    window.clear()
                outcome = observed
                if outcome is None:
                    window.clear()
                else:
                    if window and stamp-window[-1][0] > MAX_SAMPLE_GAP_S:
                        window.clear()
                    window.append((stamp, width, q.copy()))
                    while len(window) > 1 and stamp-window[1][0] >= GRIP_CONFIRMATION_DWELL_S:
                        window.pop(0)
                    if (len(window) >= 3 and stamp-window[0][0] >= GRIP_CONFIRMATION_DWELL_S-1e-9
                            and np.ptp([x[1] for x in window]) <= .001
                            and np.max(np.ptp([x[2] for x in window], axis=0)) <= .001):
                        if outcome == 'empty':
                            e._emit('recovery_grip_empty', {'width_m': width, 'force_n': force,
                                'stable_feedback_s': stamp-window[0][0], 'motion_commands_sent': 0})
                            return state
                        source = 'retained_original_grip_time_estimate'
                        if e.planner.payload_local is None:
                            contact = closing.contact
                            if contact is None:
                                raise PickRejected('夹持已稳定，但缺少接触时刻的配对反馈；保持夹爪',
                                    details={'reason': 'missing_grip_contact_feedback',
                                             'width_m': width, 'force_n': force})
                            e.planner.attach_measured(contact, mapped_stamp(contact, 'joint'))
                            source = 'post_close_contact_onset_feedback'
                        # Check the measured width against the retained payload
                        # and scene before accepting a new retention baseline.
                        e.planner.valid(q, width, payload=True)
                        e._active(e.owner, e.version, epoch)
                        fresh_feedback(state, e.clock(), stamps)
                        e.holding.held_width = width
                        e.holding.force_lost_since = None
                        e.holding.check_holding(state)
                        e._emit('recovery_grip_confirmed', {'previous_width_m': previous_width,
                            'width_m': width, 'force_n': force,
                            'stable_feedback_s': stamp-window[0][0], 'waited_s': e.clock()-began,
                            'payload_source': source,
                            'contact_joint_stamp_s': (mapped_stamp(closing.contact, 'joint')
                                if source == 'post_close_contact_onset_feedback' else None),
                            'physical_grasp_proven': False, 'motion_commands_sent': 0})
                        return state
        e.sleep(min(.01, max(0., deadline-e.clock())))
        state = e.device.snapshot_nowait()

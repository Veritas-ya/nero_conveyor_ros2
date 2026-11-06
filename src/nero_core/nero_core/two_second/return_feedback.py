"""Bounded fresh-packet recheck of FK mismatches during empty return only."""
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.joint_stream_sender import FeedbackRecheckRequired
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


RETURN_STAGES = frozenset(('return', 'return_connect', 'taught_return', 'recovery_return',
                          'prepared_taught_return', 'prepared_taught_recovery_return'))


class ReturnFeedbackCheck:
    def __init__(self, check, clock, *, enabled, timeout_s=.03):
        self.check, self.clock, self.enabled = check, clock, enabled
        self.timeout_s = timeout_s
        self.pending = None
        self.resumed = []

    def observe(self, state):
        now, stamp = self.clock(), mapped_stamp(state, 'pose')
        if self.pending is not None:
            if now >= self.pending['deadline']:
                raise self.pending['error']
            if stamp is None or stamp <= self.pending['blocked_stamp']:
                self._retry()
        try:
            self.check(state)
        except PickRejected as error:
            if (not self.enabled or stamp is None or
                    (getattr(error, 'details', None) or {}).get('reason') != 'feedback_fk_mismatch'):
                raise
            if self.pending is None:
                self.pending = {'began': now, 'deadline': now+self.timeout_s, 'error': error,
                                'original_stamp': stamp, 'blocked_stamp': stamp, 'checks': 1}
            else:
                self.pending.update(blocked_stamp=stamp, checks=self.pending['checks']+1)
            self._retry()
        else:
            if self.pending is not None:
                self.resumed.append({'waited_s': now-self.pending['began'],
                    'original_pose_stamp': self.pending['original_stamp'], 'new_pose_stamp': stamp,
                    'fk_checks': self.pending['checks']+1,
                    'original_error': str(self.pending['error']),
                    'original_details': self.pending['error'].details,
                    'targets_sent_from_rejected_feedback': 0})
                self.pending = None

    def _retry(self):
        pending = self.pending
        raise FeedbackRecheckRequired(pending['error'], pending['deadline'], details={
            'requires_new_pose_after': pending['blocked_stamp'], 'deadline_s': pending['deadline'],
            'original_error': str(pending['error']), 'targets_sent_from_rejected_feedback': 0})

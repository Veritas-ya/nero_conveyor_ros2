"""Cache a complete near-time flange packet group at the CAN receive boundary.

Receipt-time proximity is an engineering grouping criterion, not proof of a
common controller sampling instant. Caller serializes capture/select with the
same lock used for SDK parser updates.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math

POSE_PACKET_NAMES = ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz')
POSE_CAN_IDS = frozenset((0x2A2, 0x2A3, 0x2A4))
MAX_POSE_PACKET_SPAN_S = .005  # recorded coherent groups typically < 1.2 ms


@dataclass(frozen=True)
class FlangeGroup:
    pose: tuple
    parts: tuple
    epoch: int
    span_s: float


class CoherentFlangeCache:
    def __init__(self, *, max_age_s=.2, max_span_s=MAX_POSE_PACKET_SPAN_S):
        self.max_age_s, self.max_span_s = max_age_s, max_span_s
        self.epoch, self.group = None, None

    def observe_epoch(self, epoch):
        if self.epoch != epoch:
            self.group = None
            self.epoch = epoch

    def capture(self, parser, get_pose, now):
        parts = [getattr(parser, name, None) for name in POSE_PACKET_NAMES]
        times = [getattr(p, 'feedback_time', None) for p in parts]
        if any(t is None or t.mapping_quality != 'mapped' or t.mapped_monotonic is None for t in times):
            return False
        if any(t.clock_epoch != self.epoch for t in times):
            return False
        stamps = [float(t.mapped_monotonic) for t in times]
        if (not all(math.isfinite(t) and 0 <= now-t <= self.max_age_s for t in stamps)
                or max(stamps)-min(stamps) > self.max_span_s):
            return False
        pose_message = get_pose()
        if pose_message is None:
            return False
        pose = tuple(float(x) for x in pose_message.msg)
        if len(pose) != 6 or not all(math.isfinite(x) for x in pose):
            return False
        # The parser mutates MessageAbstract in place; detach timestamps and
        # immutable FeedbackTime objects, plus the driver's converted pose.
        detached = tuple(SimpleNamespace(timestamp=float(p.timestamp), msg=None, feedback_time=t)
                         for p, t in zip(parts, times))
        self.group = FlangeGroup(pose, detached, self.epoch, max(stamps)-min(stamps))
        return True

    def select(self, now):
        group = self.group
        if group is None or group.epoch != self.epoch:
            return None
        stamps = [p.feedback_time.mapped_monotonic for p in group.parts]
        if not all(math.isfinite(t) and 0 <= now-t <= self.max_age_s for t in stamps):
            return None
        return group

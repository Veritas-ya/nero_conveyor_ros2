"""Immutable CAN receive-boundary clock evidence, never encoder sample time.

Unix timestamps cannot identify their epoch across a backwards wall-clock step.
Ambiguous packets are explicitly unusable; callback receipt never freshens them.
"""
from dataclasses import asdict, dataclass
import math
import time


@dataclass(frozen=True)
class FeedbackTime:
    raw_unix_timestamp: float | None
    host_monotonic_received: float
    mapped_monotonic: float | None
    clock_epoch: int
    mapping_quality: str
    reference_wall: float
    reference_monotonic: float
    reference_uncertainty_s: float
    clock_domain: str = 'host_monotonic_from_can_unix'
    encoder_sample_time: None = None

    def to_dict(self):
        return asdict(self)


class FeedbackClock:
    """One mapper shared by robot and jaw parsers under the adapter lock.

    References are sampled at decode entry. Historical epochs are retained so
    queued packets cannot be interpreted only with the newest Unix offset.
    A 5 ms reference discontinuity creates an epoch, not a new feedback limit.
    """
    def __init__(self, *, monotonic=None, wall=None):
        self.monotonic = monotonic or time.monotonic
        self.wall = wall or time.time
        self.references = []  # (wall, mono, uncertainty, end_mono, epoch_start)
        self.last = {}

    def receive(self, stream, raw, *, callback_entered=None):
        before = self.monotonic(); wall = self.wall(); after = self.monotonic()
        mono = (before+after)/2.; uncertainty = (after-before)/2.
        received = after if callback_entered is None else float(callback_entered)
        if not all(math.isfinite(v) for v in (wall, before, after)) or not 0 <= uncertainty <= .005:
            return FeedbackTime(float(raw), received, None, max(0,len(self.references)-1),
                                'invalid_reference',wall,mono,uncertainty)
        if not self.references:
            self.references.append([wall, mono, uncertainty, math.inf, -math.inf])
        elif abs((wall-mono)-(self.references[-1][0]-self.references[-1][1])) > .005:
            self.references[-1][3] = mono
            self.references.append([wall, mono, uncertainty, math.inf, mono])
        else:
            # Use THIS receipt's bracket for a new packet, including gradual
            # clock slew. Already cached FeedbackTime objects never change.
            self.references[-1][:3] = [wall, mono, uncertainty]
        epoch = len(self.references)-1
        previous = self.last.get(stream)
        raw = float(raw)
        if previous is not None and raw == previous.raw_unix_timestamp:
            return previous  # repeated read/delivery never refreshes its time
        quality = 'invalid_reference' if uncertainty < 0 or uncertainty > .005 else 'invalid_raw'
        mapped = None
        if math.isfinite(raw) and raw > 0 and quality != 'invalid_reference':
            candidates = []
            for i, (w, m, u, end, start) in enumerate(self.references):
                value = raw+(m-w)
                # The first epoch can contain packets queued before construction.
                epsilon = u+2*math.ulp(w)+1e-9
                if start-epsilon <= value <= min(end, received)+epsilon:
                    candidates.append((i, min(value, received)))
            if len(candidates) == 1:
                epoch, mapped = candidates[0]; quality = 'mapped'
                if (previous is not None and previous.mapped_monotonic is not None
                        and mapped <= previous.mapped_monotonic):
                    mapped = None; quality = 'out_of_order'
            else:
                quality = 'ambiguous_epoch' if candidates else 'future_or_unknown_epoch'
        w, m, u, _, _ = self.references[epoch]
        result = FeedbackTime(raw if math.isfinite(raw) else None, received, mapped,
                              epoch, quality, w, m, u)
        if mapped is not None:
            self.last[stream] = result
        return result


def mapped_stamp(state, kind='joint', *, legacy=None):
    """Use immutable mapped stamps when provided; explicit legacy clock otherwise.

    This helper does not map raw Unix time. Old saved/fake states without the new
    schema keep their original synthetic clock. A new schema with missing or
    invalid evidence never falls back to raw time.
    """
    if 'feedback_time' in state:
        evidence = state['feedback_time']
        if not evidence:
            return None
        keys = {'joint':('joint_12','joint_34','joint_56','joint_7'),
                'pose':('end_pose_xy','end_pose_zrx','end_pose_ryrz'),
                'gripper':('gripper',)}.get(kind,())
        if any(key in evidence and (evidence[key] is None
                or evidence[key].get('mapping_quality') != 'mapped') for key in keys):
            return None
        values = state.get(kind+'_stamps_monotonic')
        if values and all(v is not None and math.isfinite(v) for v in values):
            return min(values)
        value = state.get(kind+'_stamp_monotonic')
        return float(value) if value is not None and math.isfinite(value) else None
    return state.get(kind+'_stamp', legacy)

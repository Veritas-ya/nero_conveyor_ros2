"""Bracketed interpolation of measured CAN joints at flange feedback time.

No prediction/extrapolation and no replacement of the current measured joints.
Each joint uses its own constituent packet clock; histories never cross epochs.
"""
from collections import deque
import numpy as np


class JointFeedbackHistory:
    def __init__(self, max_age_s=.2):
        self.history = [deque(maxlen=256) for _ in range(7)]
        self.retention_s = 2*max_age_s
        self.epoch = None
        self.received_packets=0

    def _begin(self,epoch,axes,stamps):
        backwards=any(self.history[i] and stamp<self.history[i][-1][0] for i,stamp in zip(axes,stamps))
        if epoch!=self.epoch or backwards:
            for h in self.history:h.clear()
        self.epoch=epoch

    def _append(self,axis,stamp,value):
        h=self.history[axis];sample=(float(stamp),float(value))
        if h and stamp==h[-1][0]:h[-1]=sample
        else:h.append(sample)
        while len(h)>2 and h[1][0]<stamp-self.retention_s:h.popleft()

    def observe_packet(self,axes,values,stamp,*,epoch=None):
        """Capture each decoded CAN packet, independently of snapshot reads."""
        if (not axes or len(axes)!=len(values) or any(i not in range(7) for i in axes)
                or not np.isfinite([stamp,*values]).all()):
            raise ValueError('invalid measured joint packet')
        self._begin(epoch,axes,[stamp]*len(axes))
        for axis,value in zip(axes,values):self._append(axis,stamp,value)
        self.received_packets+=1

    def pair(self, joints, packet_stamps, pose_stamp, *, epoch=None, joint_receive_offset_s=0.):
        joints = np.asarray(joints, dtype=float)
        packet_stamps = np.asarray(packet_stamps, dtype=float)
        if joints.shape != (7,) or packet_stamps.shape != (4,) or not np.isfinite(
                np.r_[joints, packet_stamps, pose_stamp, joint_receive_offset_s]).all():
            raise ValueError('关节时间对齐需要七轴及四个真实反馈包时间')
        stamps = packet_stamps[[0, 0, 1, 1, 2, 2, 3]]
        self._begin(epoch,range(7),stamps)
        for axis,(stamp,value) in enumerate(zip(stamps,joints)):
            self._append(axis,stamp,value)
        target_stamp = float(pose_stamp)+float(joint_receive_offset_s)
        result, report = self.at(target_stamp)
        # pose_stamp remains the unmodified flange receipt time: recovery
        # uses it to wait for a genuinely newer flange packet group.
        report.update(pose_stamp=float(pose_stamp),
                      joint_interpolation_stamp=target_stamp,
                      joint_receive_offset_s=float(joint_receive_offset_s),
                      offset_basis='empirical_host_receive_time_not_device_sample_time')
        return result, report

    def at(self, pose_stamp):
        """Interpolate a past receive time without mutating packet history."""
        if not np.isfinite(pose_stamp):
            raise ValueError('invalid interpolation time')
        result, spans, brackets = [], [], []
        for h in self.history:
            if not h or not h[0][0] <= pose_stamp <= h[-1][0]:
                return None, {'available': False, 'reason': 'no_measured_bracket', 'extrapolated': False}
            for index, (stamp, value) in enumerate(h):
                if stamp == pose_stamp:
                    result.append(value); spans.append(0.);brackets.append([stamp,stamp])
                    break
                if stamp > pose_stamp:
                    before, q0 = h[index-1]
                    weight = (pose_stamp-before)/(stamp-before)
                    result.append(q0+(value-q0)*weight); spans.append(stamp-before);brackets.append([before,stamp])
                    break
        return result, {'available': True, 'method': 'measured_joint_packet_linear_interpolation',
            'pose_stamp': float(pose_stamp), 'maximum_bracket_span_s': max(spans), 'extrapolated': False,
            'bracket_bounds_s':brackets,'received_joint_packets':self.received_packets,
            'history_source':'can_receive_boundary' if self.received_packets else 'snapshot_reads'}

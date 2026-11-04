#!/usr/bin/env python3
"""Offline sender replay on recorded host timestamps; never opens CAN or cameras."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
from nero_core.two_second.command_history import CommandHistory


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, duration): self.now += max(0., duration)


class Device:
    def __init__(self, clock): self.clock, self.sent = clock, []
    def snapshot(self): return {}
    def move_joints(self, q): self.sent.append((self.clock(), np.asarray(q)))


def replay(path, output, *, sample_to_send_s=0.0001):
    if not np.isfinite(sample_to_send_s) or sample_to_send_s < 0:
        raise ValueError('sample_to_send_s must be finite and nonnegative')
    trace = np.load(path/'cycle_001'/'execution_trace.npz', allow_pickle=False)
    summary = json.loads((path/'summary.json').read_text())
    phase = trace['raw_phase_time_s']
    q = trace['raw_trajectory_q_rad']
    qd = trace['raw_trajectory_qd_rad_s']
    sent = trace['dispatch_start_monotonic_s']
    # The last rejected point is not recorded: its rejection timestamp is
    # injected as a synthetic next send. No claim that it reached the robot.
    rejected = float(summary['details']['dispatch_monotonic_s'])
    if not (len(sent) >= 3 and np.all(np.diff(sent) > 0) and
            np.all(np.diff(phase) > 0) and rejected > sent[-1]):
        raise ValueError('recorded clocks/trajectory are not strictly ordered')
    send_times = np.r_[sent, rejected]
    # Reference origin is not independently stored here. The minimum recorded
    # send-minus-sample offset is an estimate; the replay is illustrative.
    estimated_origin = float(np.min(sent-phase))
    interpolator = CubicHermiteSpline(phase, q, qd, axis=0, extrapolate=True)
    injected = send_times-send_times[0]+1.
    clock, targets = Clock(), []
    device = Device(clock)
    limits = SimpleNamespace(velocity_limits=np.array([2., 2., 2., 2., 1.4, 2., 2.]),
        acceleration_limits=np.full(7, 8.))
    history = CommandHistory(limits)

    class InjectedSender(JointStreamSender):
        def _wait(self, deadline):
            # Inject observed send intervals directly, rather than emulate
            # the OS scheduler or assert that future dispatches will match.
            index = len(device.sent)
            if index >= len(injected): return False
            clock.sleep(injected[index]-sample_to_send_s-clock())
            return True

    def step(state, motion_time, dt): return np.zeros(7), {}
    def finalize(state, stamp, dt, prepared, metadata):
        sample_phase = (stamp-1.)+send_times[0]-estimated_origin
        sampled = interpolator(sample_phase)
        metadata['sample_monotonic_s'] = stamp
        clock.sleep(sample_to_send_s*.8)
        return sampled, metadata
    def validate(state, stamp, dt, candidate, metadata):
        history.validate(candidate, stamp)
        clock.sleep(sample_to_send_s*.2)
    def after(stamp, dt, candidate, metadata):
        history.record(stamp, dt, candidate, metadata)
        targets.append({'sample_s': metadata['sample_monotonic_s'], 'send_s': stamp})
    sender = InjectedSender(device, step, after, control_hz=100., clock=clock,
        sleep=clock.sleep, finalize_send=finalize, validate_dispatch=validate)
    sender.start(); sender.join()
    result = sender.snapshot()
    if result.exception is not None: raise result.exception
    times = np.array([x[0] for x in device.sent]); values = np.array([x[1] for x in device.sent])
    delta = np.diff(times); velocities = np.diff(values, axis=0)/delta[:, None]
    accelerations = np.diff(velocities, axis=0)/((delta[:-1]+delta[1:])/2)[:, None]
    report = {'source': str(path), 'clock': 'synthetic_host_monotonic',
        'method': 'recorded dispatch intervals plus synthetic rejected timestamp; cubic Hermite reconstruction',
        'recorded_accepted_commands': len(sent), 'synthetic_candidate_count': 1,
        'replayed_commands': len(device.sent), 'simulated_sample_to_send_s': sample_to_send_s,
        'maximum_injected_interval_error_s': float(np.max(abs(np.diff(times)-np.diff(send_times)))),
        'max_abs_velocity_rad_s': np.max(abs(velocities), axis=0).tolist(),
        'max_abs_acceleration_rad_s2': np.max(abs(accelerations), axis=0).tolist(),
        'last_j5_acceleration_rad_s2': float(accelerations[-1, 4]),
        'recorded_rejection_j5_acceleration_rad_s2': summary['details']['discrete_acceleration_rad_s2'][4],
        'limits_unchanged': True, 'hardware_or_mujoco_used': False,
        'limitations': ['rejected target reconstructed rather than recorded',
            'trajectory interpolated and extrapolated from saved q/qd',
            'constant synthetic sample-to-send latency; no OS or CAN timing proof',
            'host timestamp is not controller receipt or encoder sampling time']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sample-to-send-s', type=float, default=.0001)
    args = parser.parse_args()
    print(json.dumps(replay(args.run, args.output, sample_to_send_s=args.sample_to_send_s), indent=2, ensure_ascii=False))

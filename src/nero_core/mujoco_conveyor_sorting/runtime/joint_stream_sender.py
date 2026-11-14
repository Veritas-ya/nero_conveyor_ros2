"""Clock one device sender independently of trajectory preparation.

This module owns no grasp policy, geometry certificate or emergency-stop
decision. ``step`` samples a trajectory already accepted by its caller. The
caller takes device ownership back only after ``join`` and handles a recorded
exception using its existing fault/position-hold path.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

import numpy as np


class DispatchWindowExpired(RuntimeError):
    """A prepared target needs a fresh, complete check before sending."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {} if details is None else details


class FeedbackRecheckRequired(RuntimeError):
    """Read fresh feedback without dispatching a target, with a fixed deadline."""
    def __init__(self, original_error, deadline_s, *, details=None):
        super().__init__(str(original_error))
        if not np.isfinite(deadline_s):
            raise ValueError('feedback recheck deadline must be finite')
        self.original_error, self.deadline_s = original_error, float(deadline_s)
        self.details = {} if details is None else details


@dataclass(frozen=True)
class JointStreamSnapshot:
    """Detached copies of the sender ledger; device stamps are not rewritten."""

    running: bool
    stop_requested: bool
    paused: bool
    exception: BaseException | None
    command_count: int
    last_feedback: dict | None
    last_feedback_time: float | None
    last_command: dict | None
    events: tuple[dict, ...]


class JointStreamSender:
    """A single device-owning thread with an absolute dispatch cadence.

    ``step(state, sample_time, dt)`` returns ``(joint_target, metadata)`` or
    None to hand device ownership back normally without sending another point.
    ``sample_time`` is the predicted dispatch time used to sample motion;
    ``dt`` is that time minus the last actual dispatch (one period initially).
    Feedback retains its own original stamps and the separate real read time.
    It is a sampling callback, not a place to do blocking route preparation.

    ``after_send(dispatch_time, actual_dt, target, metadata)`` runs once after
    each successful send, on this same thread. It can update a private command
    history and optionally return diagnostic metadata. ``actual_dt`` is one
    configured period for the first send; the ledger's first ``gap_s`` is None.
    Callbacks must not invoke the production event logger or share mutable
    policy/controller objects with the producer.

    Requesting stop cancels pending samples. It waits for an already-started
    device send to finish; after it returns no new send can begin. ``join``
    additionally waits for an in-flight feedback read or callback to return.
    Neither operation sends a hold, opens the jaw, or disables the device.
    """

    def __init__(self, device, step: Callable, after_send: Callable | None = None,
                 *, control_hz=100., clock=time.monotonic, sleep=time.sleep,
                 pause_requested: Callable | None = None,
                 initial_dispatch_time: float | None = None,
                 finalize_send: Callable | None = None,
                 validate_dispatch: Callable | None = None,
                 feedback_reader: Callable | None = None,
                 max_prepare_lookahead_s: float | None = None):
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be finite and positive')
        self._device = device
        # A caller may use original-stamped cached feedback to avoid blocking
        # on packet retries. Its normal freshness/fault guards still apply.
        self._feedback_reader = device.snapshot if feedback_reader is None else feedback_reader
        self._step = step
        self._after_send = after_send
        # Optional, fast resampling of an already checked time window. Existing
        # users retain their original step-only behavior. Never plan here.
        self._finalize_send = finalize_send
        self._validate_dispatch = validate_dispatch
        self._period = 1. / float(control_hz)
        if (max_prepare_lookahead_s is not None and
                (not np.isfinite(max_prepare_lookahead_s) or max_prepare_lookahead_s < 0)):
            raise ValueError('prepare lookahead must be a finite nonnegative time')
        self._max_prepare_lookahead = max_prepare_lookahead_s
        self._clock = clock
        self._sleep = sleep
        self._pause_requested = pause_requested
        if initial_dispatch_time is not None and not np.isfinite(initial_dispatch_time):
            raise ValueError('initial_dispatch_time must be finite')
        self._initial_dispatch_time = initial_dispatch_time
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._finished.set()  # There is no active sender before start().
        self._lock = threading.RLock()
        # A stop can cancel a slow step/read without waiting for that work.
        # Only the physical send shares this short lock with request_stop.
        self._send_lock = threading.RLock()
        self._thread = None
        self._running = False
        self._paused = False
        self._exception = None
        self._command_count = 0
        self._last_feedback = None
        self._last_feedback_time = None
        self._last_command = None
        self._events = deque()

    def start(self):
        with self._lock:
            if self._thread is not None:
                raise RuntimeError('joint sender can only be started once')
            self._running = True
            self._finished.clear()
            self._thread = threading.Thread(target=self._run,
                                            name='joint-stream-sender', daemon=False)
            try:
                self._thread.start()
            except BaseException:
                self._running = False
                self._finished.set()
                raise
        return self

    def request_stop(self):
        # Latch before waiting: a sender finishing its in-flight call must
        # not win the lock again and start another command ahead of us.
        self._stop.set()
        with self._send_lock:
            pass

    def join(self, timeout=None):
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def wait_stopped(self, timeout=None):
        """Wake when the sender exits, including normal handoff and faults.

        Before ``start`` this returns True immediately. A stop request alone
        does not complete the wait: an in-flight read/callback must finish.
        Call ``join`` before taking device ownership back; this event only
        lets a producer wake promptly instead of waiting its next policy tick.
        """
        return self._finished.wait(timeout)

    def snapshot(self, *, drain_events=True):
        with self._lock:
            events = tuple(deepcopy(tuple(self._events)))
            if drain_events:
                self._events.clear()
            return JointStreamSnapshot(
                running=self._running, stop_requested=self._stop.is_set(),
                paused=self._paused, exception=self._exception,
                command_count=self._command_count,
                last_feedback=deepcopy(self._last_feedback),
                last_feedback_time=self._last_feedback_time,
                last_command=deepcopy(self._last_command), events=events)

    def _cancelled(self):
        if self._stop.is_set():
            return True
        if self._pause_requested is not None and self._pause_requested():
            with self._lock:
                self._paused = True
            self._stop.set()
            return True
        return False

    def _wait(self, deadline):
        while not self._cancelled():
            remaining = deadline - self._clock()
            if remaining <= 0:
                return True
            if self._sleep is time.sleep:
                self._stop.wait(remaining)
            else:
                self._sleep(remaining)
        return False

    def _run(self):
        last_send = self._initial_dispatch_time
        previous_receive = 0.
        previous_step = 0.
        window_retries = 0
        try:
            origin = self._clock() if last_send is None else last_send
            slot = 0 if last_send is None else max(
                1, math.ceil((self._clock() - origin) / self._period - 1e-9))
            def lead(duration):
                return duration if self._max_prepare_lookahead is None else min(duration, self._max_prepare_lookahead)
            while self._wait(origin + slot * self._period - lead(previous_receive + previous_step)):
                scheduled = origin + slot * self._period
                if self._cancelled():
                    break
                receive_started = self._clock()
                # Retain the device adapter's existing retry/fault contract.
                # A missing sample is never replaced by an invented state.
                state = self._feedback_reader()
                sampled = self._clock()
                with self._lock:
                    self._last_feedback = deepcopy(state)
                    self._last_feedback_time = sampled
                if self._cancelled():
                    break
                # Prepare before the next slot. Spending only the interval
                # available at the beginning of a solve, then acknowledging
                # at its end, would lose the solve's duration on every tick.
                # The predicted send clock is not a new feedback timestamp.
                # A previous outlier must not manufacture another long wait
                # now that interpolation itself is cheap and independent.
                motion_time = max(scheduled, sampled + lead(previous_step))
                dt = self._period if last_send is None else motion_time - last_send
                try:
                    result = self._step(state, motion_time, dt)
                except FeedbackRecheckRequired as error:
                    now = self._clock()
                    if now >= error.deadline_s:
                        raise error.original_error
                    with self._lock:
                        self._events.append({'event': 'joint_stream_feedback_recheck',
                            'time': now, 'details': deepcopy(error.details)})
                    # No target is sent from this rejected snapshot. The next
                    # iteration reads and fully validates a new sample, then
                    # samples the trajectory at its current dispatch time.
                    previous_receive = max(0., sampled-receive_started)
                    previous_step = 0.
                    if not self._wait(min(error.deadline_s, now+.001)):
                        break
                    continue
                if result is None:
                    break
                target, metadata = result
                q = np.asarray(target, dtype=float).copy()
                if q.ndim != 1 or not q.size or not np.isfinite(q).all():
                    raise ValueError('joint stream target must be a finite joint vector')
                metadata = deepcopy(dict(metadata or {}))
                prepared = self._clock()
                # A single solve supplies this slot. If it overruns, send
                # once and acknowledge the actual interval; never replay it.
                if not self._wait(motion_time):
                    break
                try:
                    with self._send_lock:
                        if self._cancelled():
                            break
                        dispatched = self._clock()
                        dispatched_wall = time.time()
                        gap = None if last_send is None else dispatched - last_send
                        actual_dt = self._period if gap is None else gap
                        if self._finalize_send is not None:
                            prepared_shape = q.shape
                            target, metadata = self._finalize_send(state, dispatched, actual_dt, q, metadata)
                            q = np.asarray(target, dtype=float).copy()
                            if q.shape != prepared_shape or not np.isfinite(q).all():
                                raise ValueError('final joint stream target must be finite')
                        if self._cancelled():
                            break
                        joint_target = q.tolist()
                        dispatched = self._clock()
                        gap = None if last_send is None else dispatched-last_send
                        actual_dt = self._period if gap is None else gap
                        if self._validate_dispatch is not None:
                            self._validate_dispatch(state, dispatched, actual_dt, q, metadata)
                        if self._cancelled():
                            break
                        # Validate the window again at the send-call boundary;
                        # the callback itself can take nonzero time.
                        dispatched = self._clock()
                        dispatched_wall = time.time()
                        gap = None if last_send is None else dispatched-last_send
                        actual_dt = self._period if gap is None else gap
                        window_start = metadata.get('prepared_from_monotonic_s')
                        window_end = metadata.get('prepared_until_monotonic_s')
                        if (window_start is not None and window_end is not None and
                                not window_start-1e-9 <= dispatched <= window_end+1e-9):
                            raise DispatchWindowExpired('实际发送已超出本次几何检查时窗', details={
                                'dispatch_monotonic_s': dispatched,
                                'prepared_until_monotonic_s': window_end})
                        if self._cancelled():
                            break
                        self._device.move_joints(joint_target)
                except DispatchWindowExpired as error:
                    window_retries += 1
                    with self._lock:
                        self._events.append({'event': 'joint_stream_window_retry',
                            'attempt': window_retries, 'time': self._clock(),
                            'details': deepcopy(getattr(error, 'details', {}))})
                    if window_retries >= 3:
                        raise
                    # Obtain a new device snapshot and perform the entire
                    # geometry/identity check again. Never send the expired q.
                    previous_receive = max(0., sampled-receive_started)
                    previous_step = max(0., prepared-sampled)
                    continue
                window_retries = 0
                last_send = dispatched
                command = {
                    'joint_target': q.tolist(), 'dispatch_time': dispatched,
                    'dispatch_wall_time': dispatched_wall,
                    'gap_s': gap, 'actual_dt_s': actual_dt,
                    'sample_time': motion_time, 'sample_dt_s': dt,
                    'feedback_read_time': sampled,
                    'feedback_receive_s': sampled - receive_started,
                    'scheduled_dispatch_time': scheduled,
                    'dispatch_lateness_s': max(0., dispatched - scheduled),
                    'dispatch_overrun_s': max(0., dispatched - motion_time),
                    'prepared_time': prepared,
                    'step_compute_s': prepared - sampled,
                    'slot_wait_s': max(0., dispatched - prepared),
                    'previous_feedback_receive_s': previous_receive,
                    'previous_step_compute_s': previous_step,
                    'metadata': metadata,
                }
                # Count successful sends even when acknowledging one fails.
                with self._lock:
                    self._command_count += 1
                    self._last_command = deepcopy(command)
                try:
                    if self._after_send is not None:
                        acknowledged = self._after_send(
                            dispatched, actual_dt, q.copy(), deepcopy(metadata))
                        if acknowledged is not None:
                            command['metadata'].update(deepcopy(dict(acknowledged)))
                finally:
                    with self._lock:
                        self._last_command = deepcopy(command)
                        self._events.append({'event': 'joint_stream_command', **deepcopy(command)})
                previous_receive = max(0., sampled - receive_started)
                previous_step = max(0., prepared - sampled)
                # Stay on the absolute cadence and skip expired slots. Small
                # wake-up jitter can shorten an actual interval; requiring a
                # full period after every late wake-up would skip the next
                # grid point and turn a nominal 100 Hz loop into 50 Hz.
                next_slot = math.floor((self._clock() - origin) / self._period + 1e-9) + 1
                slot = max(slot + 1, next_slot)
        except BaseException as error:
            with self._lock:
                self._exception = error  # retain type, identity and details
                self._events.append({'event': 'joint_stream_fault',
                                     'error': str(error),
                                     'error_type': type(error).__name__,
                                     'time': self._clock()})
            self._stop.set()
        finally:
            with self._lock:
                self._running = False
            self._finished.set()

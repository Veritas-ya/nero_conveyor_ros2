"""Keep original exposures eligible while their measured joint brackets arrive."""
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field
from math import ceil
from threading import Event, Lock
from typing import Callable


@dataclass(eq=False)
class WaitingExposure:
    captured: float
    epoch: int
    generation: int
    deadline: float
    calculate: Callable[[dict, dict], object]
    identity: tuple | None
    track_gate: tuple | None
    done: Future = field(default_factory=Future)
    first_miss: float | None = None


class CaptureFeedbackWorker:
    """Use the geometry lane's own thread. A missing pair does not block newer ready exposures."""

    def __init__(self, vision, pool, progress, clock):
        self.vision, self.pool, self.progress, self.clock = vision, pool, progress, clock
        raw = vision.config.raw
        fps = raw.get('camera', {}).get('depth_fps', 30)
        self.capacity = max(1, ceil(fps * raw['maximum_observation_age_s']) + 1)
        self.lock = Lock()
        self.wakeup = Event()
        self.pending = []
        self.running = self.closed = False
        self.active = None
        self.counts = Counter()

    def submit(self, captured, epoch, generation, deadline, calculate, *, identity=None, track_gate=None):
        with self.lock:
            if self.closed or self.active is not None or self.clock() >= deadline:
                return None
            if len(self.pending) >= self.capacity:
                self.counts['full'] += 1
                return None
            item = WaitingExposure(captured, epoch, generation, deadline, calculate, identity, track_gate)
            self.pending.append(item)
            if not self.running:
                self.running = True
                self.pool.submit(self._run)
            self.wakeup.set()
        return item.done

    def active_match(self):
        with self.lock:
            req = self.active
            if req is not None and req.track_gate is not None:
                return (req.captured, *req.track_gate, req.done)
        return None

    def diagnostics(self):
        with self.lock:
            return dict(pending=len(self.pending), capacity=self.capacity, **self.counts)

    def close(self, error=None):
        with self.lock:
            self.closed = True
            pending, self.pending = self.pending, []
            self.wakeup.set()
        for req in pending:
            if not req.done.set_running_or_notify_cancel():
                continue
            if error is None:
                req.done.set_result(None)
            else:
                req.done.set_exception(error)

    def _invalid(self, req, now):
        if req.done.cancelled():
            return 'cancelled'
        if now >= req.deadline:
            return 'expired'
        v = self.vision
        with v.lock:
            if (v.stop.is_set() or v.error is not None or req.epoch != v.clock_epoch or req.generation != v.geometry_generation
                    or req.captured < v.minimum_capture_s or v.frozen_queue.full
                    or (v.active_task is not None and not v.lookahead_enabled)):
                return 'invalidated'
            if req.identity is not None and (req.identity in v.consumed or v.progress_snapshot is None
                    or v.progress_snapshot.track_generation != req.identity[1]
                    or any(e.target.identity == req.identity for e in v.frozen_queue.entries)):
                return 'invalidated'
        return None

    def _discard(self, req, reason):
        with self.lock:
            if req not in self.pending:
                return
            self.pending.remove(req)
            self.counts[reason] += 1
        if req.done.set_running_or_notify_cancel():
            req.done.set_result(None)

    def _run(self):
        active = None
        try:
            while True:
                with self.lock:
                    if not self.pending:
                        self.running = False
                        return
                    self.wakeup.clear()
                    requests = sorted(self.pending, key=lambda req: req.deadline)
                now = self.clock()
                live = []
                for req in requests:
                    reason = self._invalid(req, now)
                    if reason is None:
                        live.append(req)
                    else:
                        self._discard(req, reason)
                if not live:
                    continue
                pair, armed = self.vision._first_paired_capture(live, self.progress, self.wakeup)
                unpaired = live if pair is None else live[:live.index(pair[0])]
                if unpaired:
                    with self.lock:
                        new = [req for req in unpaired if req.first_miss is None and req in self.pending]
                        for req in new:
                            req.first_miss = now
                        self.counts['retained'] += len(new)
                    for req in new:
                        self.progress.block('waiting_feedback', 'retained_capture_waiting_for_feedback',
                                            capture_monotonic_s=req.captured, deadline_monotonic_s=req.deadline)
                if pair is None:
                    # CAN receive, another exposure, or shutdown wakes the lane.
                    # A busy cache or a replay reader without notifications is retried too.
                    remaining = max(0., live[0].deadline-self.clock())
                    self.wakeup.wait(remaining if armed else min(.005, remaining))
                    continue
                req, state = pair
                ready = self.clock()
                reason = self._invalid(req, ready)
                if reason is not None:
                    self._discard(req, reason)
                    continue
                with self.lock:
                    if req not in self.pending:
                        continue
                    self.pending.remove(req)
                    if not req.done.set_running_or_notify_cancel():
                        continue
                    self.active = req
                    if req.first_miss is not None:
                        self.counts['paired_after_wait'] += 1
                active = req
                timing = dict(capture_feedback_ready_monotonic_s=ready, capture_feedback_retried=req.first_miss is not None,
                              capture_feedback_wait_s=0. if req.first_miss is None else ready-req.first_miss,
                              capture_processing_started_monotonic_s=self.clock())
                result = req.calculate(state, timing)
                # Publish/complete this original before another job can use the lane.
                req.done.set_result(result)
                active = None
                with self.lock:
                    self.active = None
        except BaseException as error:
            with self.vision.lock:
                self.vision.error = error
            with self.lock:
                self.active = None
            if active is not None and not active.done.done():
                active.done.set_exception(error)
            self.close(error)

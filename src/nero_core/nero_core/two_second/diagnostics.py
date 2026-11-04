"""Small diagnostic snapshots and asynchronous logs; never authorize motion."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class VisionProgress:
    """Count every rejected observation, even between persisted snapshots.

    Stage residence is wall time on the vision worker. Operation timings are
    separate measurements and must not be added to that residence or to the
    concurrent planning/execution time.
    """
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = Lock()
        self.started = self.since = clock()
        self.stage = 'waiting_for_rgbd_and_feedback'
        self.residence = Counter()
        self.blockers = Counter()
        self.frames = self.ready = 0
        self.context = {}
        self.last_blocker = None
        self.last_blocker_by_reason = {}
        self.timings = {}
        self.finished = None

    def enter(self, stage):
        now = self.clock()
        with self.lock:
            if self.stage != stage:
                self.residence[self.stage] += max(0., now-self.since)
                self.stage, self.since = stage, now

    def frame(self, **context):
        with self.lock:
            self.frames += 1
            self.context = dict(context)

    def annotate_frame(self, **context):
        with self.lock:
            self.context = {**self.context, **context}

    def finish(self):
        with self.lock:
            self.finished = self.clock()

    def block(self, stage, reason, **details):
        self.enter(stage)
        with self.lock:
            self.blockers[reason] += 1
            self.last_blocker = {'stage': stage, 'reason': reason,
                'monotonic_s': self.clock(), 'frame': dict(self.context), 'details': details}
            self.last_blocker_by_reason[reason] = self.last_blocker

    def mark_ready(self):
        self.enter('target_ready')
        with self.lock:
            self.ready += 1

    @contextmanager
    def measure(self, operation):
        begin = self.clock()
        try:
            yield
        finally:
            elapsed = max(0., self.clock()-begin)
            with self.lock:
                item = self.timings.setdefault(operation,
                    {'count': 0, 'total_s': 0., 'last_s': 0., 'max_s': 0.})
                item.update(count=item['count']+1, total_s=item['total_s']+elapsed,
                    last_s=elapsed, max_s=max(item['max_s'], elapsed))

    def snapshot(self):
        with self.lock:
            now = self.clock() if self.finished is None else self.finished
            residence = dict(self.residence)
            residence[self.stage] = residence.get(self.stage, 0.)+max(0., now-self.since)
            return {'snapshot_monotonic_s': now, 'stage': self.stage,
                'stage_elapsed_s': max(0., now-self.since),
                'vision_wall_s': max(0., now-self.started), 'stage_wall_s': residence,
                'frames_received': self.frames, 'target_ready_evaluations': self.ready,
                'blocker_counts': dict(self.blockers), 'last_blocker': self.last_blocker,
                'last_blocker_by_reason': dict(self.last_blocker_by_reason),
                'frame': dict(self.context),
                'operation_timings_s': {k: {**v, 'mean_s': v['total_s']/v['count']}
                    for k, v in self.timings.items()}}


class RunJournal:
    """Bounded nonblocking producers; only the writer touches events.jsonl.

    Each line is flushed, so completed records can be read during a run. This
    is not an fsync/power-loss guarantee. Overflow and I/O failures are exposed
    in summary instead of silently turning diagnostics into a motion failure.
    """
    def __init__(self, directory, *, clock=time.monotonic, capacity=256, observer=None):
        self.clock = clock
        self.observer = observer
        self.path = Path(directory)/'events.jsonl'
        self.file = self.path.open('x', encoding='utf-8')
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.closing = Event()
        self.enqueued = self.written = self.dropped = 0
        self.write_wall_s = 0.
        self.errors = []
        self.observer_errors = []
        self.worker = Thread(target=self._write, name='two-second-log-writer', daemon=True)
        self.worker.start()

    def emit(self, event, **data):
        with self.lock:
            if self.closing.is_set():
                self.dropped += 1
                return False
            row = {'event': event, 'monotonic_s': self.clock(),
                'wall_time_utc': datetime.now(timezone.utc).isoformat(), 'data': data}
            try:
                self.queue.put_nowait(row)
            except Full:
                self.dropped += 1
                return False
            self.enqueued += 1
        return True

    def _write(self):
        try:
            while not self.closing.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=.05)
                except Empty:
                    continue
                begin = time.monotonic()
                try:
                    self.file.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False)+'\n')
                    self.file.flush()
                    with self.lock:
                        self.written += 1
                except Exception as error:
                    with self.lock:
                        if len(self.errors) < 10:
                            self.errors.append(f'{type(error).__name__}: {error}')
                finally:
                    with self.lock:
                        self.write_wall_s += time.monotonic()-begin
                    if self.observer is not None:
                        try:
                            self.observer(row)
                        except Exception as error:
                            with self.lock:
                                if len(self.observer_errors) < 10:
                                    self.observer_errors.append(f'{type(error).__name__}: {error}')
                    self.queue.task_done()
        finally:
            try:
                self.file.close()
            except Exception as error:
                with self.lock:
                    if len(self.errors) < 10:
                        self.errors.append(f'close: {type(error).__name__}: {error}')

    def close(self):
        with self.lock:
            self.closing.set()
        self.worker.join(timeout=5.)
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            return {'path': str(self.path), 'enqueued': self.enqueued,
                'written': self.written, 'dropped': self.dropped, 'errors': list(self.errors),
                'writer_alive': self.worker.is_alive(), 'writer_wall_s': self.write_wall_s,
                'observer_errors': list(self.observer_errors)}


class ProgressMonitor:
    """Persist low-rate progress while main waits for a target or a planner."""
    def __init__(self, journal, snapshot, *, interval_s=1.):
        self.journal, self.snapshot, self.interval_s = journal, snapshot, interval_s
        self.stop = Event()
        self.thread = Thread(target=self._run, name='two-second-progress', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(self.interval_s):
            try:
                self.journal.emit('runtime_progress', **self.snapshot())
            except Exception as error:
                self.journal.emit('diagnostic_error', error_type=type(error).__name__, error=str(error))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)

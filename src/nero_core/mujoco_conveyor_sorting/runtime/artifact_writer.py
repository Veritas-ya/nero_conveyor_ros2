"""Bounded diagnostic writes; computation/decisions never depend on a job.

Library calls stay synchronous without an explicit session. Sessions share a
writer with their worker threads and drain on every normal/exceptional exit.
Queue pressure waits (and records it), never silently drops evidence.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import copy
import functools
import json
from pathlib import Path
from queue import Queue
from threading import Condition, Lock, Thread
import time
import warnings
import numpy as np

_writer = None
_writer_lock = Lock()


def _snapshot(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        result = value.copy()
        memo[id(value)] = result
        return result
    if is_dataclass(value):
        result = copy.copy(value)
        memo[id(value)] = result
        # Copy only declared data, not locks / lazily attached frame caches.
        if hasattr(result, '_perception_products'):
            object.__setattr__(result, '_perception_products', None)
        for field in fields(value):
            object.__setattr__(result, field.name, _snapshot(getattr(value, field.name), memo))
        return result
    if isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, _snapshot(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, (list, tuple)):
        result = [_snapshot(item, memo) for item in value]
        return tuple(result) if isinstance(value, tuple) else result
    return copy.deepcopy(value, memo)


def _bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return sum(_bytes(getattr(value, f.name), seen) for f in fields(value))
    return 0


def _record_output(job, output, args, kwargs):
    """Save helpers may report a caught IO error instead of raising it."""
    if isinstance(output, (str, dict)):
        job['output'] = output
    found = []
    def errors(value):
        if isinstance(value, dict) and value.get('artifact_error'):
            found.append(str(value['artifact_error']))
        elif hasattr(value, 'statistics'):
            errors(value.statistics)
    for value in (output, *args, *kwargs.values()):
        errors(value)
    job['status'] = 'failed' if found else 'completed'
    if found:
        job['error'] = '; '.join(dict.fromkeys(found))


class ArtifactWriter:
    def __init__(self, directory=None, *, max_jobs=4, max_bytes=64*1024*1024):
        self.directory = None if directory is None else Path(directory)
        self.max_jobs, self.max_bytes = max_jobs, max_bytes
        self.condition = Condition()
        self.queue = Queue()
        self.pending_jobs = self.pending_bytes = 0
        self.peak_jobs = self.peak_bytes = 0
        self.records = []
        self.closed = False
        self.worker = Thread(target=self._run, name='perception-artifact-writer', daemon=False)
        self.worker.start()

    def submit(self, function, *args, **kwargs):
        started = time.perf_counter()
        size = _bytes((args, kwargs))
        if size > self.max_bytes:
            # A single oversized payload stays with its producer, after prior
            # jobs drain. No oversized snapshot or silent evidence loss.
            self.queue.join()
            job = {'job_id': len(self.records), 'function': function.__qualname__,
                   'status': 'writing', 'synchronous_oversize': True, 'input_bytes': size,
                   'capacity_wait_s': time.perf_counter()-started}
            self.records.append(job)
            begin, cpu = time.perf_counter(), time.thread_time()
            try:
                output = function(*args, **kwargs)
                _record_output(job, output, args, kwargs)
            except Exception as error:
                job.update(status='failed', error=f'{type(error).__name__}: {error}')
            job['write_wall_s'] = time.perf_counter()-begin
            job['write_thread_cpu_s'] = time.thread_time()-cpu
            return job
        with self.condition:
            if self.closed:
                raise RuntimeError('diagnostic writer is closed')
            while self.pending_jobs >= self.max_jobs or (self.pending_jobs and
                    self.pending_bytes+size > self.max_bytes):
                self.condition.wait()
            self.pending_jobs += 1
            self.pending_bytes += size
            self.peak_jobs = max(self.peak_jobs, self.pending_jobs)
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
            job = {'job_id': len(self.records), 'function': function.__qualname__,
                   'status': 'snapshot', 'input_bytes': size,
                   'capacity_wait_s': time.perf_counter()-started}
            self.records.append(job)
        snapshot = time.perf_counter()
        try:
            # Copy before returning to the producer. Frame/points can be reused
            # or modified immediately; the background job owns stable data.
            frozen_args, frozen_kwargs = _snapshot((args, kwargs))
        except BaseException:
            with self.condition:
                self.pending_jobs -= 1
                self.pending_bytes -= size
                job['status'] = 'snapshot_failed'
                self.condition.notify_all()
            raise
        job['snapshot_s'] = time.perf_counter()-snapshot
        job['submitted_perf_s'] = time.perf_counter()
        job['status'] = 'queued'
        self.queue.put((function, frozen_args, frozen_kwargs, job, size))
        return job

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            function, args, kwargs, job, size = item
            start, cpu = time.perf_counter(), time.thread_time()
            job.update(status='writing', queue_wait_s=start-job['submitted_perf_s'])
            try:
                output = function(*args, **kwargs)
                _record_output(job, output, args, kwargs)
            except Exception as error:
                job.update(status='failed', error=f'{type(error).__name__}: {error}')
            finally:
                job.update(write_wall_s=time.perf_counter()-start,
                           write_thread_cpu_s=time.thread_time()-cpu)
                with self.condition:
                    self.pending_jobs -= 1
                    self.pending_bytes -= size
                    self.condition.notify_all()
                self.queue.task_done()

    def close(self):
        with self.condition:
            if self.closed: return
            self.closed = True
        start = time.perf_counter()
        with self.condition:
            while self.pending_jobs:
                self.condition.wait()
        self.queue.put(None)
        self.worker.join()
        self.flush_s = time.perf_counter()-start
        if self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                (self.directory/'artifact_writes.json').write_text(json.dumps({
                    'max_jobs': self.max_jobs, 'max_bytes': self.max_bytes,
                    'peak_jobs': self.peak_jobs, 'peak_bytes': self.peak_bytes,
                    'exit_flush_s': self.flush_s, 'jobs': self.records}, ensure_ascii=False, indent=2)+'\n')
            except Exception as error:
                self.report_error = f'{type(error).__name__}: {error}'
                warnings.warn('diagnostic writer report failed: '+self.report_error,
                              RuntimeWarning, stacklevel=2)


def current_writer():
    return _writer


@contextmanager
def artifact_session(directory=None, **limits):
    global _writer
    with _writer_lock:
        previous = _writer
        writer = ArtifactWriter(directory, **limits)
        _writer = writer
    try:
        yield writer
    finally:
        _writer = previous
        writer.close()


def background_artifact(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        writer = current_writer()
        if writer is None:
            return function(*args, **kwargs)
        try:
            return writer.submit(function, *args, **kwargs)
        except Exception as error:
            # Preserve diagnostic failures without changing task decisions.
            writer.records.append({'function': function.__qualname__, 'status': 'failed',
                                   'error': f'{type(error).__name__}: {error}'})
    return wrapped

"""Bounded archival of completed attempts, outside the next-pick handoff."""
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore, Lock, Thread
import time

from .diagnostics import json_value


@dataclass
class CompletedAttempt:
    index: int
    trace: object
    events: object
    report: object
    command_diagnostics: dict
    logging_errors: list

    @classmethod
    def capture(cls, active, index, report=None):
        if getattr(active, 'sender', None) is not None:
            raise RuntimeError('发送器尚未退出，不能归档本轮记录')
        # run() has joined its producers. Retain their completed records, not
        # the executor/device/planner, and do no array/JSON conversion here.
        return cls(index, active.trace, tuple(active.events),
            None if report is None else dict(report),
            active.command_history.diagnostics(), list(active.event_logging_errors))


def write_attempt(record, output):
    directory = Path(output)/f'cycle_{record.index:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    result = record.trace.write(directory)
    if isinstance(result, dict) and result.get('write_error'):
        raise OSError(result['write_error'])
    for name, value in [('events', record.events), ('report', record.report)]:
        if value is None:
            continue
        temporary = directory/(name+'.json.tmp')
        temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2))
        temporary.replace(directory/(name+'.json'))
    return directory


@dataclass
class ArchiveTicket:
    saved: bool = False
    error: str | None = None


class AttemptArchive:
    """One writer; capacity includes the active write and queued attempts."""
    def __init__(self, output, emit, *, capacity=4, writer=write_attempt):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError('归档容量必须为正整数')
        self.output, self.emit, self.writer = Path(output), emit, writer
        self.capacity = capacity
        self.slots = BoundedSemaphore(capacity)
        self.queue = Queue()
        self.lock = Lock()
        self.submitted = self.saved = self.pending = self.peak_pending = 0
        self.errors = []
        self.notification_errors = []
        self.failed = []
        self.write_wall_s = 0.
        self.maximum_submit_wall_s = 0.
        self.closed = False
        self.thread = Thread(target=self._run, name='completed-attempt-writer', daemon=True)
        self.thread.start()

    def check_health(self):
        with self.lock:
            if self.errors:
                raise RuntimeError('后台归档失败，回位后停止接收下一件：'+self.errors[0])

    def _emit(self, event, **data):
        try:
            self.emit(event, **data)
        except Exception as error:
            with self.lock:
                if len(self.notification_errors) < 10:
                    self.notification_errors.append(f'{type(error).__name__}: {error}')

    def submit(self, record):
        begin = time.monotonic()
        with self.lock:
            if self.closed:
                raise RuntimeError('归档线程已关闭')
            if self.errors:
                raise RuntimeError('后台归档失败，回位后停止接收下一件：'+self.errors[0])
            if not self.slots.acquire(blocking=False):
                raise RuntimeError('后台归档队列已满，回位后停止接收下一件并保存剩余记录')
            ticket = ArchiveTicket()
            self.pending += 1
            self.submitted += 1
            self.peak_pending = max(self.peak_pending, self.pending)
            self.queue.put_nowait((record, ticket))
            self.maximum_submit_wall_s = max(self.maximum_submit_wall_s, time.monotonic()-begin)
        return ticket

    def _save(self, record, ticket):
        begin = time.monotonic()
        try:
            directory = self.writer(record, self.output)
        except Exception as error:
            ticket.error = f'cycle_{record.index:03d}: {type(error).__name__}: {error}'
            with self.lock:
                self.errors.append(ticket.error)
                self.failed.append((record, ticket))
            self._emit('cycle_trace_save_failed', cycle_index=record.index, error=ticket.error)
        else:
            ticket.saved = True
            with self.lock:
                self.saved += 1
            self._emit('cycle_trace_saved', cycle_index=record.index, directory=str(directory),
                command_diagnostics=record.command_diagnostics,
                event_logging_errors=record.logging_errors, archival_mode='background',
                archive_wall_s=time.monotonic()-begin)
        finally:
            with self.lock:
                self.write_wall_s += time.monotonic()-begin

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            record, ticket = item
            try:
                self._save(record, ticket)
            finally:
                with self.lock:
                    self.pending -= 1
                self.slots.release()

    def close(self):
        # Called after all robot/camera owners have exited. Only shutdown may
        # wait for disk; no handoff, sender or recovery waits here.
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put_nowait(None)
        self.thread.join()
        return self.diagnostics()

    def diagnostics(self):
        with self.lock:
            return dict(mode='bounded_background_thread', capacity=self.capacity,
                submitted=self.submitted, saved=self.saved, pending=self.pending,
                peak_pending=self.peak_pending, errors=list(self.errors),
                notification_errors=list(self.notification_errors),
                write_wall_s=self.write_wall_s, maximum_submit_wall_s=self.maximum_submit_wall_s,
                writer_alive=self.thread.is_alive())

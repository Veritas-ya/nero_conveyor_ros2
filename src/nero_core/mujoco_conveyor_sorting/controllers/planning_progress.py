"""Thread-safe live planning stages; wall and CPU times use separate clocks."""
from collections import Counter, deque
from contextlib import contextmanager
from functools import wraps
from threading import RLock
import time


class PlanningProgress:
    def __init__(self):
        self._lock = RLock()
        self._start = None
        self._stack = []
        self._totals = {}
        self._events = deque(maxlen=512)
        self._sequence = 0
        self._status = 'idle'
        self._rejects = Counter()
        self._ik = Counter()
        self._budget = None
        self._budget_hit = None
        self._finish = None
        self._last_active = []
        self._request = {}

    def _event(self, kind, **fields):
        self._sequence += 1
        self._events.append({'sequence': self._sequence, 'event': kind,
            'elapsed_wall_s': time.monotonic()-self._start, **fields})

    def begin(self, request, budget):
        with self._lock:
            self._start = time.monotonic()
            self._finish = None
            self._budget = budget
            self._budget_hit = None
            self._stack = []
            self._totals = {}
            self._rejects.clear(); self._ik.clear(); self._events.clear()
            self._last_active = []
            self._status = 'running'
            self._request = {name: getattr(request, name, None) for name in
                ('request_id', 'generation', 'target_id', 'scene_epoch', 'stage')}
            self._event('request_started', wall_budget_s=budget)

    def budget_exceeded(self):
        with self._lock:
            if self._budget_hit is None:
                self._budget_hit = time.monotonic()
                self._last_active = [frame['name'] for frame in self._stack]
                self._event('budget_exceeded', active_stages=self._last_active.copy())
                self._status = 'cancelling_budget'

    def finish(self, status):
        with self._lock:
            self._finish = time.monotonic()
            self._status = status
            self._event('request_finished', status=status)

    def rejection(self, reason):
        with self._lock:
            self._rejects[reason] += 1

    def ik(self, success, iterations):
        with self._lock:
            self._ik['attempts'] += 1
            self._ik['successes' if success else 'failures'] += 1
            self._ik['iterations'] += int(iterations or 0)

    @contextmanager
    def stage(self, name, context, *, events=True):
        with self._lock:
            if self._start is None:
                self._start = time.monotonic()
            frame = dict(name=name, wall=time.monotonic(), cpu=time.thread_time(),
                         child_wall=0., child_cpu=0.)
            self._stack.append(frame)
            if events:
                self._event('stage_started', name=name, context=context)
        try:
            yield
        finally:
            wall = time.monotonic()-frame['wall']
            cpu = time.thread_time()-frame['cpu']
            with self._lock:
                self._stack.pop()
                total = self._totals.setdefault(name, dict(calls=0, inclusive_wall_s=0.,
                    exclusive_wall_s=0., inclusive_thread_cpu_s=0., exclusive_thread_cpu_s=0.))
                total['calls'] += 1
                total['inclusive_wall_s'] += wall
                total['exclusive_wall_s'] += max(0., wall-frame['child_wall'])
                total['inclusive_thread_cpu_s'] += cpu
                total['exclusive_thread_cpu_s'] += max(0., cpu-frame['child_cpu'])
                if self._stack:
                    self._stack[-1]['child_wall'] += wall
                    self._stack[-1]['child_cpu'] += cpu
                if events:
                    self._event('stage_finished', name=name, wall_s=wall, thread_cpu_s=cpu)

    def snapshot(self, after_sequence=0):
        with self._lock:
            now = self._finish or time.monotonic()
            active = [dict(name=f['name'], elapsed_wall_s=now-f['wall']) for f in self._stack]
            totals = {k: dict(v) for k, v in self._totals.items()}
            # Inclusive and exclusive are alternatives, never additive columns.
            for i, frame in enumerate(self._stack):
                elapsed = now-frame['wall']
                child_active = now-self._stack[i+1]['wall'] if i+1 < len(self._stack) else 0.
                total = totals.setdefault(frame['name'], dict(calls=0, inclusive_wall_s=0.,
                    exclusive_wall_s=0., inclusive_thread_cpu_s=0., exclusive_thread_cpu_s=0.))
                total['inclusive_wall_s'] += elapsed
                total['exclusive_wall_s'] += max(0., elapsed-frame['child_wall']-child_active)
            return {**self._request, 'status':self._status, 'wall_budget_s':self._budget,
                'elapsed_wall_s':0. if self._start is None else now-self._start,
                'active_stages':active, 'stages_at_budget_exceeded':self._last_active.copy(),
                'budget_exceeded':self._budget_hit is not None,
                'cancellation_ack_wall_s':None if self._budget_hit is None or self._finish is None
                    else self._finish-self._budget_hit,
                'stage_totals':totals, 'rejection_counts':dict(self._rejects), 'ik_counts':dict(self._ik),
                'sequence':self._sequence,
                'events':[dict(e) for e in self._events if e['sequence'] > after_sequence],
                'clock':'host_monotonic', 'cpu_clock':'planning_thread_only',
                'timing_rule':'sum exclusive stages only; inclusive stages overlap; active CPU not yet sampled'}


def planning_stage(name, phase_arg=None, *, events=True):
    def decorate(function):
        @wraps(function)
        def measured(self, *args, **kwargs):
            progress = getattr(self, '_planning_progress', None)
            if progress is None:
                return function(self, *args, **kwargs)
            label = name if phase_arg is None else name+':'+str(args[phase_arg] if len(args)>phase_arg else kwargs.get('phase', 'unknown'))
            context = dict(getattr(self, '_diagnostic_context', {}))
            with progress.stage(label, context, events=events):
                return function(self, *args, **kwargs)
        return measured
    return decorate

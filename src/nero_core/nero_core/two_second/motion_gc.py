"""Defer CPython's cyclic collector until all motion owners have exited.

Reference counting still frees ordinary objects. This changes neither the
trajectory nor device checks. gc.disable is process-wide, so overlapping
scopes share ownership and restore the caller's original GC setting.
"""
import gc
from threading import RLock, get_ident
import time


_lock = RLock()
_owners = 0
_restore_enabled = False


class DeferCyclicGC:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.collections = []
        self.started = {}
        self.report = {}

    def _collection(self, phase, info):
        key = get_ident()
        if phase == 'start':
            self.started[key] = (self.clock(), info['generation'])
        elif key in self.started:
            stamp, generation = self.started.pop(key)
            self.collections.append({'start_monotonic_s': stamp,
                'end_monotonic_s': self.clock(), 'generation': generation,
                'thread_id': key})

    def __enter__(self):
        global _owners, _restore_enabled
        with _lock:
            if _owners == 0:
                _restore_enabled = gc.isenabled()
                gc.disable()
            _owners += 1
            self.begin = self.clock()
            gc.callbacks.append(self._collection)
        return self

    def __exit__(self, kind, error, traceback):
        global _owners
        with _lock:
            gc.callbacks.remove(self._collection)
            _owners -= 1
            finished = self.clock()
            collected, idle_wall = None, 0.
            try:
                if _owners == 0 and _restore_enabled:
                    # The caller has joined its senders before leaving this
                    # scope, including failed/cancelled recovery attempts.
                    began = self.clock()
                    collected = gc.collect()
                    idle_wall = self.clock()-began
            finally:
                if _owners == 0 and _restore_enabled:
                    gc.enable()
                self.report = {'automatic_cyclic_gc_deferred': True,
                    'scope_started_monotonic_s': self.begin,
                    'scope_finished_monotonic_s': finished,
                    'collections_during_scope': self.collections,
                    'idle_collection_wall_s': idle_wall,
                    'idle_collected_objects': collected,
                    'other_motion_scopes': _owners,
                    'automatic_gc_restored': _owners == 0 and gc.isenabled()}

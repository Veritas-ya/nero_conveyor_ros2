"""Read-only interval diagnostics; never configure pools, devices or control.

Nested spans inherit a caller-owned list through ContextVar. Missing platform
resource counters remain unavailable. Wall-minus-thread-CPU is not a cause.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
import time

_active_events = ContextVar('depth_diagnostic_events', default=None)


def resources():
    if os.environ.get('NERO_DEPTH_RESOURCE_DIAGNOSTICS') != '1':
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_THREAD)
        return dict(minor_faults=value.ru_minflt, major_faults=value.ru_majflt,
                    voluntary_switches=value.ru_nvcsw, involuntary_switches=value.ru_nivcsw)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def trace_context():
    # Separate samples, not an assertion of equal epochs or exposure time.
    configuration = None
    if os.environ.get('NERO_DEPTH_RESOURCE_DIAGNOSTICS') == '1':
        configuration = runtime_configuration()
    return dict(runtime_configuration=configuration, pid=os.getpid(), native_thread_id=threading.get_native_id(),
                thread_name=threading.current_thread().name,
                clock_anchor=dict(perf_counter_ns=time.perf_counter_ns(),
                                  monotonic_ns=time.monotonic_ns(), unix_time_ns=time.time_ns()),
                perf_clock=time.get_clock_info('perf_counter').implementation,
                monotonic_clock=time.get_clock_info('monotonic').implementation,
                resource_counters_requested=os.environ.get('NERO_DEPTH_RESOURCE_DIAGNOSTICS') == '1',
                span_semantics='siblings disjoint; nested spans inclusive; never sum parent and child')


@contextmanager
def nested_events(events):
    token = _active_events.set(events)
    try:
        yield
    finally:
        _active_events.reset(token)


@contextmanager
def measured_stage(name, timings, *, events=None):
    events = _active_events.get() if events is None else events
    before = resources() if events is not None else None
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    try:
        yield
    finally:
        cpu_end, wall_end = time.thread_time_ns(), time.perf_counter_ns()
        for suffix, value in (('_s', (wall_end-wall)*1e-9),
                              ('_thread_cpu_s', (cpu_end-cpu)*1e-9)):
            key = name+suffix
            timings[key] = (timings.get(key) or 0.)+value
        if events is not None:
            after = resources()
            events.append(dict(name=name, perf_start_ns=wall, perf_end_ns=wall_end,
                thread_cpu_start_ns=cpu, thread_cpu_end_ns=cpu_end,
                resource_delta=({k: after[k]-before[k] for k in before}
                                if before is not None and after is not None else None)))


def runtime_configuration():
    result = dict(environment={k: os.environ.get(k) for k in
        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_WAIT_POLICY')})
    try:
        import cv2
        result.update(opencv_version=cv2.__version__, opencv_threads=cv2.getNumThreads())
    except Exception as error:
        result['opencv_threads'] = None
        result['opencv_diagnostic_error'] = type(error).__name__
    try:
        from threadpoolctl import threadpool_info
        result['threadpools'] = threadpool_info()
    except Exception as error:
        result['threadpools'] = None
        result['threadpool_diagnostic_error'] = type(error).__name__
    return result

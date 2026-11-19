"""No hardware; cyclic collection cannot run automatically inside motion."""
import gc
import json
import time
from threading import Event,Thread

import pytest

from nero_core.two_second.motion_gc import DeferCyclicGC
from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace


@pytest.mark.parametrize('failure',[False,True])
def test_restore_original_gc_state_and_collect_after_scope(failure):
    was_enabled=gc.isenabled();gc.enable()
    window=DeferCyclicGC()
    try:
        try:
            with window:
                assert not gc.isenabled()
                cycles=[]
                for _ in range(2000):
                    item=[];item.append(item);cycles.append(item)
                del cycles,item
                if failure:raise RuntimeError('cancel/failure')
        except RuntimeError:
            assert failure
        assert gc.isenabled()
        assert not window.report['collections_during_scope']
        assert window.report['idle_collected_objects']>=2000
        assert window.report['idle_collection_wall_s']>=0.
    finally:
        if not was_enabled:gc.disable()


def test_respect_previously_disabled_gc():
    was_enabled=gc.isenabled();gc.disable()
    try:
        with DeferCyclicGC() as window:assert not gc.isenabled()
        assert not gc.isenabled() and window.report['idle_collected_objects'] is None
    finally:
        if was_enabled:gc.enable()


def test_overlapping_threads_restore_only_after_last_owner():
    entered,release=Event(),Event()
    was_enabled=gc.isenabled();gc.enable()
    def secondary():
        with DeferCyclicGC():
            entered.set();assert release.wait(2.)
    worker=Thread(target=secondary)
    try:
        with DeferCyclicGC() as first:
            worker.start();assert entered.wait(2.)
        assert not gc.isenabled()
        assert first.report['other_motion_scopes']==1
        release.set();worker.join(timeout=2.)
        assert not worker.is_alive() and gc.isenabled()
    finally:
        release.set();worker.join(timeout=2.)
        if not was_enabled:gc.disable()


def test_explicit_external_collection_is_diagnosed_not_hidden():
    with DeferCyclicGC() as window:gc.collect()
    events=window.report['collections_during_scope']
    assert len(events)==1 and events[0]['generation']==2
    assert events[0]['end_monotonic_s']>=events[0]['start_monotonic_s']


def test_sender_scalar_timings_detached_and_saved_after_run(tmp_path):
    trace=ExecutionTrace()
    event={'event':'joint_stream_command','dispatch_time':7.,'gap_s':.06,
        'feedback_receive_s':.001,'step_compute_s':.002,'metadata':{'large':'omitted'}}
    trace.record_sender_events([event,{'event':'joint_stream_fault'}],stage='carry')
    event['gap_s']=999.
    report=trace.write(tmp_path)
    assert 'write_error' not in report
    saved=json.loads((tmp_path/'execution_dispatch_timing.jsonl').read_text())
    assert saved['gap_s']==.06 and saved['stage']=='carry' and 'metadata' not in saved


def test_executor_restores_gc_after_cancelled_run(monkeypatch):
    from test_two_second_hardware import executor_fixture
    e,dev,_=executor_fixture()
    def cancelled(_):
        assert not gc.isenabled()
        e.cancel()
        raise RuntimeError('cancelled')
    monkeypatch.setattr(e,'_run_with_recovery',cancelled)
    original=gc.isenabled()
    with pytest.raises(RuntimeError,match='cancelled'):e.run(None)
    assert gc.isenabled()==original and e.sender is None and not dev.commands
    assert e.events[-1]['event']=='motion_memory_timing'

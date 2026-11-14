"""Completed-run persistence never waits in the next-target handoff."""
import json
from threading import Event, Thread
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nero_core.two_second.attempt_archive import AttemptArchive, CompletedAttempt, write_attempt


def record(index):
    class Trace:
        def write(self, directory):
            (directory/'trace.json').write_text(json.dumps({'cycle':index}))
    return CompletedAttempt(index, Trace(), [{'cycle':index}], {'cycle':index}, {}, [])


@pytest.mark.parametrize('deadline_margin_s',[.195,.319])
def test_handoff_keeps_real_deadline_while_archive_is_blocked(tmp_path,deadline_margin_s):
    writing, next_started = Event(), Event()
    def slow_write(item, output):
        writing.set()
        assert next_started.wait(timeout=2.)
        return write_attempt(item,output)
    archive=AttemptArchive(tmp_path,lambda *a,**k:None,writer=slow_write)
    begin=time.monotonic()
    try:
        ticket=archive.submit(record(1))
        assert writing.wait(timeout=deadline_margin_s)
        # Stand in for claiming the certified next path at its actual deadline.
        assert time.monotonic()-begin<deadline_margin_s
        assert not ticket.saved
        next_started.set()
    finally:
        next_started.set()
        report=archive.close()
    assert report['saved']==1 and report['pending']==0 and not report['writer_alive']
    assert json.loads((tmp_path/'cycle_001/report.json').read_text())=={'cycle':1}


def test_queue_full_is_bounded_nonblocking_and_does_not_discard_current_attempt(tmp_path):
    writing, release = Event(), Event()
    def slow_write(item, output):
        writing.set()
        assert release.wait(timeout=2.)
        return write_attempt(item,output)
    archive=AttemptArchive(tmp_path,lambda *a,**k:None,capacity=2,writer=slow_write)
    current=record(3)
    try:
        archive.submit(record(1));assert writing.wait(timeout=1.)
        archive.submit(record(2))
        begin=time.monotonic()
        with pytest.raises(RuntimeError,match='队列已满'):archive.submit(current)
        assert time.monotonic()-begin<.1
        assert archive.diagnostics()['pending']==2
    finally:
        release.set(); result=archive.close()
    # The rejected submission remains owned by the caller for shutdown saving.
    write_attempt(current,tmp_path)
    assert result['saved']==2 and result['peak_pending']==2
    assert [json.loads((tmp_path/f'cycle_{i:03d}/trace.json').read_text())['cycle']
            for i in (1,2,3)]==[1,2,3]


def test_write_error_is_explicit_and_blocks_new_admissions(tmp_path):
    def broken_write(item, output):raise OSError('disk full')
    archive=AttemptArchive(tmp_path,lambda *a,**k:None,writer=broken_write)
    ticket=archive.submit(record(1))
    result=archive.close()
    assert not ticket.saved and 'disk full' in ticket.error
    assert result['saved']==0 and result['errors'] and not result['writer_alive']
    with pytest.raises(RuntimeError,match='后台归档失败'):archive.check_health()


def test_trace_write_error_return_cannot_be_counted_as_saved(tmp_path):
    item=record(1)
    item.trace=NS(write=lambda directory:{'write_error':'partial npz failure'})
    with pytest.raises(OSError,match='partial npz failure'):write_attempt(item,tmp_path)
    assert not (tmp_path/'cycle_001/report.json').exists()


def test_logger_exception_does_not_kill_writer_or_drop_following_records(tmp_path):
    def broken_emit(*a,**k):raise RuntimeError('observer failed')
    archive=AttemptArchive(tmp_path,broken_emit)
    archive.submit(record(1));archive.submit(record(2))
    result=archive.close()
    assert result['saved']==2 and result['notification_errors']
    assert result['pending']==0 and not result['writer_alive']


def test_shutdown_waits_for_pending_records_without_holding_caller_before_shutdown(tmp_path):
    writing,release,closed=Event(),Event(),Event()
    def slow_write(item, output):
        writing.set();assert release.wait(timeout=2.)
        return write_attempt(item,output)
    archive=AttemptArchive(tmp_path,lambda *a,**k:None,writer=slow_write)
    archive.submit(record(1));assert writing.wait(timeout=1.)
    def shutdown():archive.close();closed.set()
    thread=Thread(target=shutdown);thread.start()
    try:assert not closed.wait(.03)
    finally:release.set();thread.join(timeout=2.)
    assert closed.is_set() and archive.diagnostics()['saved']==1


def test_completed_record_has_no_executor_or_device_access_and_owns_list_snapshot():
    class Device:
        def __getattr__(self,name):pytest.fail('archival accessed a device')
    active=NS(trace=object(),events=[{'old':True}],event_logging_errors=[],
        command_history=NS(diagnostics=lambda:{}),device=Device(),planner=Device(),sender=None)
    item=CompletedAttempt.capture(active,1,{'returned_to_wait':True})
    active.events.append({'next':True})
    assert len(item.events)==1 and not hasattr(item,'device') and not hasattr(item,'planner')
    active.sender=object()
    with pytest.raises(RuntimeError,match='发送器尚未退出'):CompletedAttempt.capture(active,2)


def test_trace_export_evaluates_arrays_once(tmp_path,monkeypatch):
    from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
    trace=ExecutionTrace()
    calls=[];original=trace.arrays
    def arrays():calls.append(1);return original()
    monkeypatch.setattr(trace,'arrays',arrays)
    result=trace.write(tmp_path)
    assert 'write_error' not in result and len(calls)==1
    with np.load(tmp_path/'execution_trace.npz') as data:
        assert data['dispatched_q_rad'].shape==(0,7)
    assert json.loads((tmp_path/'execution_trace_summary.json').read_text())['command_count']==0

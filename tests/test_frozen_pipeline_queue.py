"""Multi-piece admission, capture-age preservation and independent cancellation."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from nero_core.two_second.frozen_queue import FrozenTargetQueue
from nero_core.two_second.reference import EntryNotReady
from test_two_second_diagnostics import vision_fixture


def target(n,stamp=10.,x=-.5):
    return NS(identity=(0,1,n),capture_s=stamp,epoch=0,sequence=n,
        entry_position=np.array([x,0,0]),velocity=np.array([.1,0,0]),
        points=np.array([[x-.01,0,0],[x+.01,.02,.03]]),detection={})


def test_queue_pins_head_bounds_memory_and_keeps_original_clock():
    q=FrozenTargetQueue(2);a,b,c=[target(i) for i in (1,2,3)]
    assert q.offer(a,10.2,1.) and q.offer(b,10.3,1.)
    assert q.head is a and not q.offer(c,10.4,1.)
    assert not q.offer(a,10.4,1.)
    assert q.admitted(a,12.,1.) and a.capture_s==10.
    assert not q.admitted(target(1),12.,1.)
    assert not q.admitted(a,9.,1.)
    assert q.remove(a.identity,'consumed') is a and q.head is b
    assert not q.offer(c,12.,1.)
    assert q.snapshot()['counts']==dict(accepted=2,full=1,duplicate_identity=1,consumed=1,stale_at_publication=1)


def multi_vision(monkeypatch):
    v=vision_fixture(monkeypatch,'ready');v.frozen_queue=FrozenTargetQueue(4)
    v.config.raw.update(tracked_local_geometry=True,tracked_frozen_queue_capacity=4)
    v.clock_epoch=0;v.enable_lookahead();v.progress_snapshot=v.meter.latest_snapshot
    return v


def test_consume_and_cancel_do_not_invalidate_another_inflight_piece(monkeypatch):
    v=multi_vision(monkeypatch);now=time.monotonic()
    a,b,c=[target(i,now-.1) for i in (1,2,3)]
    events=[];v.event_sink=lambda name,**kw:events.append((name,kw))
    try:
        for t in (a,b):assert v.frozen_queue.offer(t,now,1.)
        generation=v.geometry_generation
        future=Future();v.geometry_future=future
        v.consume(a.identity);v.discard_pending(b.identity)
        assert v.geometry_generation==generation and v.active_task==a.identity
        future.set_result((generation,1,c));v._finish_geometry()
        assert v.poll_pending() is c and v.poll() is None
        assert events[-1][0]=='target_frozen' and events[-1][1]['identity']==c.identity
        assert events[-1][1]['capture_monotonic_s']==c.capture_s
        v.resume_targets();assert v.poll() is c
        assert v.pending_admission(c,now+1.5)
        assert not v.pending_admission(c,now+10.)
        assert v.repeats_recent_target(c.entry_position+c.velocity*.2,c.capture_s+.2)
    finally:v.close()


@pytest.mark.parametrize('invalidate',['clock','cancel','generation','old_tracker'])
def test_late_geometry_never_resurrects_invalidated_identity(monkeypatch,invalidate):
    v=multi_vision(monkeypatch);now=time.monotonic();a=target(1,now-.1)
    f=Future();v.geometry_future=f;f.set_result((0,1,a))
    if invalidate=='clock':v.clock_epoch=1
    if invalidate=='cancel':v.discard_pending(a.identity)
    if invalidate=='generation':v.geometry_generation+=1
    if invalidate=='old_tracker':v.progress_snapshot=replace(v.progress_snapshot,track_generation=2)
    try:
        v._finish_geometry();assert v.latest is None
    finally:v.close()


def test_expired_head_is_removed_without_removing_next_piece(monkeypatch):
    v=multi_vision(monkeypatch);now=time.monotonic()
    a=target(1,now-.1,x=.29);b=target(2,now-.1,x=-.5)
    try:
        for t in (a,b):v.frozen_queue.offer(t,now,1.)
        assert v.poll_pending() is b and a.identity in v.consumed
        v.clock_epoch=1
        assert v.poll_pending() is None
    finally:v.close()


def test_multiple_tracks_use_own_seed_and_failed_track_cannot_starve_other(monkeypatch):
    v=multi_vision(monkeypatch);frame=v.camera.read_latest()
    a=v.meter.latest_snapshot.tracks[0]
    b=replace(a,track_id=2,observed_position_base=(-.51,0,.04))
    peer=replace(v.meter.latest_snapshot,tracks=(a,b))
    masks={1:np.ones((2,2),bool),2:np.zeros((2,2),bool)}
    v.meter.geometry_seed_rgb=lambda f,t:masks[t.track_id]
    try:
        selected,seed=v._select_geometry_track(frame,peer,[a,b],time.monotonic())
        assert selected is a and seed is masks[1]
        v.geometry_attempts[(0,1,1)]=7
        selected,seed=v._select_geometry_track(frame,peer,[a,b],time.monotonic())
        assert selected is b and seed is masks[2]
        # A piece outside the admission window does not block the next one.
        original=v.entry_time
        def gate(capture,pos,velocity,now):
            if pos[0]==a.observed_position_base[0]:
                raise EntryNotReady('outside window','already_at_entry')
            return original(capture,pos,velocity,now)
        v.geometry_attempts.clear();v.entry_time=gate
        assert v._select_geometry_track(frame,peer,[a,b],time.monotonic())[0] is b
        v.meter.geometry_seed_rgb=lambda f,t:None
        assert v._select_geometry_track(frame,peer,[a,b],time.monotonic()) is None
    finally:v.close()


def test_return_lookahead_plans_queued_original_exposure_after_one_second(monkeypatch):
    from test_return_lookahead import fixture, wait_state, target as next_target
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    import nero_core.two_second.vision as module
    worker,clock,item,_,planner,events=fixture()
    v=multi_vision(monkeypatch);v.config=worker.config;worker.vision=v
    monkeypatch.setattr(module,'time',NS(monotonic=clock))
    second=next_target(3)
    try:
        assert v.frozen_queue.offer(item,10.2,1.)
        assert v.frozen_queue.offer(second,10.3,1.)
        clock.now=11.5;worker._step()
        assert worker.ready.target is item and item.capture_s==10.
        worker.confirm_return(wait_state(clock));worker._step()
        assert worker.poll_ready().target is item
        worker.discard_ready(worker.ready,PickRejected('cancel only the first queued target'))
        worker._step()
        assert worker.ready.target is second and second.capture_s==10.
        assert item.identity in v.consumed and second.identity not in v.consumed
        assert [call[0] for call in planner.calls][-1] is second
    finally:v.close()


def test_rejected_geometry_events_are_complete_and_large_arrays_are_not_copied_every_retry(monkeypatch):
    from mujoco_conveyor_sorting.runtime.single_pick import TargetNotReady
    v=multi_vision(monkeypatch);frame=v.camera.read_latest();saved=[];events=[]
    v.geometry_rejection_sink=lambda *args:saved.append(args)
    v.event_sink=lambda name,**kw:events.append((name,kw))
    v.geometry_progress.frame(sequence=7,capture_monotonic_s=12.)
    failure=TargetNotReady('no component',reason='no_workpiece',details={
        'candidate_count':0,'candidate_filter_stages':{'components_after_size':0}})
    try:
        args=(failure,frame,np.eye(4),{},(0,1,1),v.model.measured_belt_surface,None)
        v._record_geometry_failure(*args);v._record_geometry_failure(*args)
        assert len(saved)==1 and len(events)==2
        assert events[0][1]['details']['candidate_filter_stages']['components_after_size']==0
        assert saved[0][0] is frame
    finally:v.close()

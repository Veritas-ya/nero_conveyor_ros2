"""Late capture feedback on the production geometry lanes. No devices or robot commands."""
from threading import Event
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from nero_core.two_second import tracked_single_exposure as native
from nero_core.two_second.fixed_freeze import DepthCandidate
from test_tracked_freeze_deadline import cell, measurement


def historical(stamp):
    return dict(joints=[0.]*7, width_m=.08, capture_state_role='self_filter_only_not_motion_feedback',
                capture_joint_pairing=dict(available=True, extrapolated=False, joint_interpolation_stamp=stamp,
                    bracket_bounds_s=[[stamp-.003,stamp+.003]]*7))


@pytest.mark.parametrize('has_match',[True, False])
def test_same_original_exposure_is_published_once_without_another_camera_frame(cell,monkeypatch,has_match):
    v,events=cell
    captured=time.monotonic()-.01
    item=measurement(v,captured)
    frame=NS(header=NS(clock_epoch=0,sequence=item.sequence,received_time=captured+.005,source_time=83.))
    seed=np.ones((2,2),bool)
    track=NS(track_id=1,confirmed=True,misses=0,velocity_base_m_s=None,speed_valid=False,observed_position_base=item.center)
    peer=NS(current=True,clock_epoch=0,track_generation=1,tracks=(track,))
    v.meter=NS(parameters={'tracker_gate_radius':.03})
    monkeypatch.setattr(v,'_select_geometry_track',lambda *a:(track,seed))
    calculated=[]
    def tracked(original, d2c, original_peer, original_track, st, stamp, generation, *a, **kw):
        assert original is frame and original_peer is peer and original_track is track
        assert stamp==captured==st['capture_joint_pairing']['joint_interpolation_stamp'] and kw['seed_rgb_mask'] is seed
        calculated.append('tracked')
        item.detection['freeze_delivery']={}
        return generation,peer.track_generation,item
    monkeypatch.setattr(v,'_prepare_target',tracked)
    class Locator:
        def __init__(self,*a,**kw): pass
        def locate(self,original,*,robot_state,**kw):
            assert original is frame and robot_state['capture_joint_pairing']['joint_interpolation_stamp']==captured
            return [DepthCandidate(item.center,8,np.flatnonzero(item.detection['mask']))],{}
    monkeypatch.setattr(native,'NativeDepthLocator',Locator)
    def geometry(cfg, original, d2c, surface, candidate, stamp, **kw):
        assert original is frame and stamp==captured==kw['robot_state']['capture_joint_pairing']['joint_interpolation_stamp']
        calculated.append('native')
        return {**item.detection,'object_points':item.points},item.center,item.velocity,item.grasps,1.
    monkeypatch.setattr(native,'frozen_geometry',geometry)
    ready=Event()
    asked={lane:Event() for lane in ('native','tracked')}
    def snapshot(captures,*,wakeup):
        lane='native' if wakeup is v.single_frame_backup.feedback_worker.wakeup else 'tracked'
        assert captures==(captured,)
        asked[lane].set()
        return ((captured,historical(captured)),False) if ready.is_set() else (None,True)
    v.capture_states_reader=snapshot
    decision=v.single_frame_backup.offer(frame,np.eye(4),v.model.measured_belt_surface,captured)
    match=v._schedule_geometry(frame,np.eye(4),peer,captured) if has_match else None
    decision.set_result(match)
    assert asked['native'].wait(.2)
    if has_match:assert asked['tracked'].wait(.2)
    assert not calculated and v.latest is None
    ready.set()
    v.single_frame_backup.feedback_worker.wakeup.set()
    v.tracked_feedback_worker.wakeup.set()
    v.single_frame_backup.future.result(timeout=1.)
    if match is not None:match[-1].result(timeout=1.)
    assert v.latest is not None and v.latest.capture_s==captured and v.latest.sequence==frame.header.sequence
    published=[d for name,d in events if name=='target_frozen']
    assert len(published)==1 and published[0]['published_upstream_distance_m']>=.60
    assert published[0]['freeze_source']==('tracked' if has_match else 'single_exposure_native_depth_fallback')
    assert published[0]['freeze_delivery']['capture_feedback_retried']


def test_older_missing_feedback_cannot_block_newer_ready_capture(cell):
    v,_=cell
    worker=v.single_frame_backup.feedback_worker
    first=time.monotonic()
    newer=first+.001
    asked=Event(); available={newer}
    def reader(captured):
        asked.set()
        return historical(captured) if captured in available else None
    v.capture_state_reader=reader
    def stamp(state,timing):return state['capture_joint_pairing']['joint_interpolation_stamp']
    old=worker.submit(first,0,0,first+1.,stamp)
    assert asked.wait(.2)
    other=worker.submit(newer,0,0,newer+1.,stamp)
    assert other.result(timeout=.3)==newer and not old.done()
    available.add(first)
    assert old.result(timeout=.3)==first


def test_backup_sees_the_actual_tracked_job_when_pairs_arrive_out_of_order(cell):
    v,_=cell
    worker=v.tracked_feedback_worker
    captured=time.monotonic(); late=Event();begin=Event();finish=Event(); queried=Event()
    def read(stamp):
        queried.set()
        return historical(stamp) if stamp>captured or late.is_set() else None
    def calculate(state,timing):
        begin.set()
        assert finish.wait(.5)
    v.capture_state_reader=read
    gate=(np.zeros(3),.03)
    old=worker.submit(captured,0,0,captured+1.,lambda *a:None,identity=(0,1,1),track_gate=gate)
    assert queried.wait(.3)
    new=worker.submit(captured+.001,0,0,captured+1.,calculate,identity=(0,1,2),track_gate=gate)
    try:
        assert begin.wait(.3)
        assert worker.active_match()[-1] is new and not old.done()
    finally:finish.set()
    new.result(timeout=.3);late.set();worker.wakeup.set();old.result(timeout=.3)


@pytest.mark.parametrize('reason',['deadline','epoch','generation','stop','full','consumed'])
def test_invalidated_original_never_starts_geometry(cell,reason):
    v,_=cell
    w=v.tracked_feedback_worker
    now=[10.];w.clock=lambda:now[0]
    queried=Event();ready=Event();calculated=[]
    def reader(t):
        queried.set()
        # A newer frame's apparently valid encoder state cannot satisfy this frame.
        return historical(t if ready.is_set() else t+.02)
    v.capture_state_reader=reader
    job=w.submit(10.,0,0,10.3,lambda *a:calculated.append(a),identity=(0,1,1))
    assert queried.wait(.2)
    with v.lock:
        if reason=='deadline':now[0]=10.31
        if reason=='epoch':v.clock_epoch=1
        if reason=='generation':v.geometry_generation+=1
        if reason=='stop':v.stop.set()
        if reason=='full':
            for i in range(v.frozen_queue.capacity):
                t=measurement(v,10.,index=i)
                assert v.frozen_queue.offer(t,10.,1.)
        if reason=='consumed':v.consumed.add((0,1,1))
    ready.set();w.wakeup.set()
    assert job.result(timeout=.3) is None and not calculated


def test_retained_count_is_bounded_and_shutdown_resolves_every_waiter(cell):
    v,_=cell
    w=v.single_frame_backup.feedback_worker;w.capacity=2
    stamp=time.monotonic();q=Event()
    def missing(_):
        q.set()
        return None
    v.capture_state_reader=missing
    task=lambda state,timing:pytest.fail('geometry ran without paired feedback')
    a=w.submit(stamp,0,0,stamp+1.,task)
    assert q.wait(.2)
    b=w.submit(stamp+.001,0,0,stamp+1.,task)
    assert b is not None and w.submit(stamp+.002,0,0,stamp+1.,task) is None
    assert w.diagnostics()['pending']==2
    w.close()
    assert a.result(timeout=.3) is None and b.result(timeout=.3) is None
    assert w.diagnostics()['pending']==0


def test_armed_cache_wakes_on_update_and_propagates_controller_fault(cell):
    v,_=cell
    w=v.single_frame_backup.feedback_worker
    first,again,fault=Event(),Event(),Event()
    def reader(captures,*,wakeup):
        if first.is_set():again.set()
        first.set()
        if fault.is_set():raise RuntimeError('controller feedback fault')
        return None,True
    v.capture_states_reader=reader
    captured=time.monotonic()
    job=w.submit(captured,0,0,captured+1.,lambda *a:pytest.fail('fault cannot authorize geometry'))
    assert first.wait(.2) and not again.wait(.025)
    fault.set();w.wakeup.set()
    with pytest.raises(RuntimeError,match='controller feedback fault'):
        job.result(timeout=.3)
    with pytest.raises(RuntimeError,match='controller feedback fault'):
        v.poll()

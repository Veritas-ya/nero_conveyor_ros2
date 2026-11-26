"""Arrival dwell overlaps computation, never vanishes before loaded motion."""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.stationary_hold import StationaryHold
from nero_core.two_second.trajectory import TimedPath
from nero_core.two_second.feedback_gates import RestArrival,ARRIVAL_DWELL_S
from test_two_second_hardware import executor_fixture


class Clock:
    def __call__(self):return time.monotonic()
    def sleep(self,t):time.sleep(t)


def run_case(overlap,work_s,empty=False):
    e,dev,clock=executor_fixture(Clock(),empty=empty)
    e.config.raw.update(prepared_taught_cycle_enabled=True,taught_segments={'transfer':'test'},
                        closure_policy='scheduled_attempt')
    e.planner.payload_local=np.zeros((1,3));e.planner.capture_tcp=np.eye(4)
    e.planner.attach_measured=lambda *args:None
    e.carry_service=NS(discard=lambda:None) if overlap else None
    e.start_carry_prediction=lambda *args:None
    origin=clock()+.12
    ref=SmoothIntercept(origin,origin,.2,np.zeros(3),np.zeros(3),np.zeros(3),
        [1,0,0,0],[1,0,0,0],follow_s=.2,brake_s=.2)
    path=TimedPath(lambda t,d=0:np.zeros(7),e.owner,1,'smooth_intercept_follow_brake',ref.full_duration_s,0.)
    state=e._stream(path,origin=origin,epoch=0,reference=ref)
    if not empty:
        with StationaryHold(e,JointStreamSender,epoch=0,carrying=True):
            time.sleep(work_s)
    return e,state,clock()-(origin+path.duration)


@pytest.mark.parametrize('work_s',[.03,.15])
def test_original_dwell_overlaps_fast_or_slow_validation(work_s):
    old,_,serial=run_case(False,work_s)
    e,state,parallel=run_case(True,work_s)
    names=[x['event'] for x in e.events]
    assert 'carry_arrival_overlap_started' in names
    confirmed=next(x['data'] for x in e.events if x['event']=='carry_arrival_overlap_confirmed')
    assert confirmed['actual_stable_s']>=ARRIVAL_DWELL_S-1e-9
    assert e.pending_rest_arrival is None
    assert parallel>=max(work_s,ARRIVAL_DWELL_S)
    assert serial-parallel>min(work_s,ARRIVAL_DWELL_S)-.04
    assert state['width_m']==.04


def test_no_held_object_cannot_start_early_carry_handoff():
    e,state,elapsed=run_case(True,0.,empty=True)
    assert state['width_m']==0. and e.pending_rest_arrival is None
    assert 'carry_arrival_overlap_started' not in [x['event'] for x in e.events]
    assert elapsed>=ARRIVAL_DWELL_S


@pytest.mark.parametrize('failure',['cancel','clock','grip','drift'])
def test_pending_arrival_aborts_and_clears_on_loss_of_execution_conditions(failure):
    e,dev,clock=executor_fixture(Clock())
    e.rest_anchor=np.zeros(7);e.holding.held_width=.04;dev.width=.04;dev.force=2.
    e.pending_rest_arrival=RestArrival(clock(),since=clock())
    with pytest.raises(PickRejected):
        with StationaryHold(e,JointStreamSender,epoch=0,carrying=True):
            if failure=='cancel':e.cancel()
            if failure=='clock':e.epoch=lambda:1
            if failure=='grip':dev.force=0.
            if failure=='drift':
                read=dev.snapshot
                def drifting():
                    state=read();state['joints'][0]=.1
                    return state
                dev.snapshot=drifting
            # The existing force-loss debounce is 250 ms, not immediate.
            time.sleep(.3 if failure=='grip' else .03)
    assert e.pending_rest_arrival is None and e.sender is None

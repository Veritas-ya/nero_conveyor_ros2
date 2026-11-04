import pytest
from mujoco_conveyor_sorting.runtime.feedback_clock import FeedbackClock

class Clock:
    mono=100.
    wall=1000.
    def advance(self,dt,jump=0.):
        self.mono+=dt;self.wall+=dt+jump
    def mapper(self):
        return FeedbackClock(monotonic=lambda:self.mono,wall=lambda:self.wall)

@pytest.mark.parametrize('jump',[.5,-.5])
def test_cached_packet_keeps_frozen_time_after_wall_jump(jump):
    c=Clock();m=c.mapper();first=m.receive('j1',1000.)
    c.advance(.01,jump)
    same=m.receive('j1',1000.)
    assert same is first and same.mapped_monotonic==100.
    assert c.mono-same.mapped_monotonic==pytest.approx(.01)

def test_normal_queue_delay_duplicate_and_out_of_order():
    c=Clock();m=c.mapper();first=m.receive('j1',999.95)
    assert first.mapped_monotonic==pytest.approx(99.95)
    c.advance(.01)
    assert m.receive('j1',999.95) is first
    assert m.receive('j1',999.94).mapping_quality=='out_of_order'
    assert m.receive('j2',999.9).mapped_monotonic==pytest.approx(99.9)
    assert m.receive('j1',1000.01).mapped_monotonic==pytest.approx(100.01)

@pytest.mark.parametrize('jump',[.5,-.5])
def test_jump_new_and_queued_packets_never_freshen_ambiguous_data(jump):
    c=Clock();m=c.mapper();m.receive('j1',1000.)
    c.advance(.01,jump)
    queued=m.receive('j2',999.9)
    if queued.mapped_monotonic is not None:
        assert queued.mapped_monotonic==pytest.approx(99.9)
    new=m.receive('j3',c.wall)
    if jump>0:assert new.mapped_monotonic==pytest.approx(c.mono)
    else:assert new.mapped_monotonic is None and new.mapping_quality=='ambiguous_epoch'
    c.advance(.51)
    assert m.receive('j3',c.wall).mapped_monotonic==pytest.approx(c.mono)

@pytest.mark.parametrize('raw',[float('nan'),float('inf'),0.,1001.])
def test_invalid_or_future_not_mapped(raw):
    assert Clock().mapper().receive('j1',raw).mapped_monotonic is None

@pytest.mark.parametrize('jump',[0.,.5,-.5])
def test_live_snapshot_ages_immutable_packets_with_monotonic_clock(monkeypatch,jump):
    from test_nero_pick_hardware_snapshot import asynchronous_adapter
    from mujoco_conveyor_sorting.runtime import nero_pick_hardware as module
    c=Clock();adapter=asynchronous_adapter(monkeypatch);mapper=c.mapper()
    adapter._feedback_clock=mapper
    monkeypatch.setattr(module.time,'monotonic',lambda:c.mono)
    monkeypatch.setattr(module.time,'time',lambda:c.wall)
    for i,name in enumerate(module.POSE_JOINT_PACKETS):
        part=getattr(adapter.robot._parser,name);part.timestamp=(999.99+i*.001 if i<3 else 1000.-i*.001)
        part.feedback_time=mapper.receive(name,part.timestamp)
    # This __new__ test double must also provide the live adapter's complete
    # flange group and measured joint bracket; no real receive thread exists.
    from mujoco_conveyor_sorting.runtime.coherent_flange import CoherentFlangeCache
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    adapter._pose_group_cache=CoherentFlangeCache();adapter._pose_group_cache.observe_epoch(0)
    assert adapter._pose_group_cache.capture(adapter.robot._parser,adapter.robot.get_flange_pose,c.mono)
    adapter._joint_feedback_history=JointFeedbackHistory()
    adapter._joint_feedback_history.pair([0.]*7,[99.985]*4,99.99,epoch=0)
    for owner,name in [(adapter.robot,'get_arm_status'),(adapter.robot,'get_driver_states'),
                       (adapter.gripper,'get_gripper_status')]:
        original=getattr(owner,name)
        def read(*args,_original=original):
            msg=_original(*args);msg.timestamp=1000.
            msg.feedback_time=mapper.receive((_original,args),1000.)
            return msg
        # Seed the real packet BEFORE the wall jump (subsequent getters are cached).
        cache={}
        def cached(*args,_read=read,_cache=cache):
            if args not in _cache:_cache[args]=_read(*args)
            return _cache[args]
        setattr(owner,name,cached)
    adapter.snapshot()
    c.advance(.01,jump)
    state=adapter.snapshot(wait_timeout_s=0.)
    assert state['joint_stamp']==pytest.approx(999.994) # raw retained
    assert state['joint_stamp_monotonic']==pytest.approx(99.994)
    assert state['feedback_timing']['oldest_packet_age_s']==pytest.approx(.02)
    c.advance(.201)
    with pytest.raises(module.FeedbackError):adapter.snapshot(wait_timeout_s=0.)


def test_monitor_and_trace_do_not_remap_old_packet_at_read(monkeypatch):
    from types import SimpleNamespace
    from collections import deque
    from threading import Lock
    import numpy as np
    from mujoco_conveyor_sorting.runtime.full_plan_scene_monitor import FullPlanSceneMonitor
    from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
    state={'joints':np.zeros(7),'width_m':.1,'joint_stamp':1000.,
        'feedback_time':{'joint_12':{'mapping_quality':'mapped','clock_epoch':0}},
        'joint_stamps_monotonic':[100.]*4}
    monitor=FullPlanSceneMonitor.__new__(FullPlanSceneMonitor)
    monitor.clock=lambda:100.01;monitor.lock=Lock();monitor.samples=deque()
    monitor.publish_state(state,.01)
    assert monitor.samples[-1][0]==100.
    trace=ExecutionTrace(clock=lambda:100.01,wall_clock=lambda:1000.51)
    trace.stamp_clock_domain='host_wall'
    trace.record_feedback(state,100.01,100.01)
    assert trace.arrays()['feedback_joint_sample_monotonic_s'][0]==100.
    np.testing.assert_equal(trace.arrays()['feedback_joint_axis_sample_monotonic_s'][0],100.)


def test_new_schema_missing_mapping_never_falls_back_to_raw():
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    assert mapped_stamp({'feedback_time':{},'joint_stamp':1000.}) is None
    assert mapped_stamp({'joint_stamp':1.})==1. # saved/fake legacy domain untouched


def test_invalid_quality_cannot_be_hidden_by_a_numeric_mapped_stamp():
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import fresh,FeedbackPending
    from dataclasses import replace
    from types import SimpleNamespace
    timing=Clock().mapper().receive('j1',1000.)
    invalid=replace(timing,mapping_quality='ambiguous_epoch')
    with pytest.raises(FeedbackPending):fresh(SimpleNamespace(msg='unsafe',timestamp=1000.),'j1',100.01,timing=invalid)
    assert mapped_stamp({'feedback_time':{'joint_12':invalid.to_dict()},'joint_stamp':1000.,
        'joint_stamp_monotonic':100.}) is None


def test_reference_scheduling_delay_is_not_validated_mapping():
    values=iter([100.,100.02])
    m=FeedbackClock(monotonic=lambda:next(values),wall=lambda:1000.)
    result=m.receive('j1',1000.)
    assert result.mapping_quality=='invalid_reference' and result.mapped_monotonic is None


def test_callback_entry_is_distinct_from_later_decode_reference():
    m=Clock().mapper()
    packet=m.receive('j1',999.98,callback_entered=99.99)
    assert packet.host_monotonic_received==99.99
    assert packet.reference_monotonic==100.
    assert packet.mapped_monotonic==pytest.approx(99.98)
    assert m.receive('j2',1000.,callback_entered=99.99).mapped_monotonic is None


def test_gradual_wall_slew_uses_each_new_receipt_reference_not_initial_offset():
    c=Clock();m=c.mapper();first=m.receive('j1',1000.)
    c.advance(.1,jump=.004)
    new=m.receive('j2',c.wall-.203)
    assert new.mapping_quality=='mapped'
    assert new.reference_monotonic==c.mono
    assert c.mono-new.mapped_monotonic==pytest.approx(.203)
    assert first.mapped_monotonic==100. and first.reference_wall==1000.
    assert new.clock_epoch==first.clock_epoch # gradual reference change, no fake reset
    from types import SimpleNamespace
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import fresh,FeedbackPending
    with pytest.raises(FeedbackPending):
        fresh(SimpleNamespace(msg='old'), 'j2', c.mono, timing=new)

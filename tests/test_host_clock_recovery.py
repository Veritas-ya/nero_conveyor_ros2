"""Host-step recovery handshake; synthetic feedback only, no SDK or device."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_two_second_hardware import executor_fixture
from test_failed_attempt_recovery import recovery
from nero_core.two_second.clock_recovery import PACKETS, TIMEOUT_S
from nero_core.two_second.recovery import _recovery_feedback, RecoveryFailed
from nero_core.two_second.streaming import fresh_feedback
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def timed_state(original, now, *, epoch=1, step=.418513, stamp=None):
    state = original()
    stamp = now-.001 if stamp is None else stamp
    offset = 1000.+step
    state['feedback_time'] = {key: {
        'raw_unix_timestamp':stamp+offset,'host_monotonic_received':now,
        'mapped_monotonic':stamp,'clock_epoch':epoch,'mapping_quality':'mapped',
        'reference_wall':now+offset,'reference_monotonic':now,
        'reference_uncertainty_s':0.,'clock_domain':'host_monotonic_from_can_unix',
        'encoder_sample_time':None} for key in PACKETS}
    for kind in ('pose', 'joint', 'gripper'):
        state[kind+'_stamp_monotonic'] = stamp
    return state


def seed(e, device, clock, *, step=.418513):
    original = device.snapshot
    old = timed_state(original, clock(), epoch=0, step=0.)
    fresh_feedback(old, clock(), e.feedback_stamps)
    if e.command_history.q is None:
        e.command_history.record(clock()-.01, .01, device.q, {})
    clock.sleep(.01)
    read = lambda:timed_state(original, clock(), step=step)
    with pytest.raises(PickRejected, match='代次重置') as failure:
        fresh_feedback(read(), clock(), e.feedback_stamps)
    return original, read, failure.value


@pytest.fixture
def clock_recovery():
    e, device, clock = executor_fixture()
    original, read, failure = seed(e, device, clock)
    device.snapshot_nowait = read
    return SimpleNamespace(e=e, device=device, clock=clock, original=original,
                           read=read, failure=failure, began=clock())


@pytest.mark.parametrize('step', [.418513, -.418513])
def test_rebases_only_after_new_stable_feedback_without_commands(step):
    e, device, clock = executor_fixture()
    _, read, failure = seed(e, device, clock, step=step)
    device.snapshot_nowait = read
    began = clock()
    state = _recovery_feedback(e, epoch=0, failure=failure)
    assert clock()-began >= .1-1e-9
    assert device.commands == []
    assert e.feedback_stamps['pose_clock_epoch'] == (1, 1, 1)
    assert e.feedback_stamps['gripper_clock_epoch'] == (1,)
    assert [x['event'] for x in e.events] == ['recovery_clock_wait','recovery_clock_rebased']
    event = e.events[-1]['data']
    assert event['host_wall_step_s'] == pytest.approx(step)
    assert event['stable_samples'] >= 3 and event['stable_span_s'] >= .1-1e-9
    assert event['recovery_targets_sent'] == 0
    # The same adjustment no longer rejects the subsequent recovery read.
    assert _recovery_feedback(e, epoch=0) is not None
    assert state['feedback_time']['joint_12']['clock_epoch'] == 1


def test_actual_mapper_forward_and_unambiguous_backward_step():
    from mujoco_conveyor_sorting.runtime.feedback_clock import FeedbackClock
    for jump in (.418513, -.418513):
        e, device, clock = executor_fixture()
        offset = [1000.]
        mapper = FeedbackClock(monotonic=clock, wall=lambda:offset[0]+clock())
        original = device.snapshot
        def read():
            state = original()
            state['feedback_time'] = {k:mapper.receive(k, offset[0]+clock()-.001).to_dict() for k in PACKETS}
            for kind, keys in (('pose',PACKETS[:3]),('joint',PACKETS[3:7]),('gripper',PACKETS[7:])):
                state[kind+'_stamp_monotonic'] = min(state['feedback_time'][k]['mapped_monotonic'] for k in keys)
            return state
        fresh_feedback(read(), clock(), e.feedback_stamps)
        e.command_history.record(clock()-.01,.01,device.q,{})
        clock.sleep(.01);offset[0] += jump
        first_after_jump = mapper.receive('probe',offset[0]+clock())
        if jump < 0:
            # Ambiguous backward-step packets remain unusable. Only start
            # recovery once the mapper itself can prove a unique epoch.
            assert first_after_jump.mapping_quality == 'ambiguous_epoch'
            clock.sleep(.5)
        else:
            clock.sleep(.01)
        with pytest.raises(PickRejected,match='代次重置') as failure:
            fresh_feedback(read(),clock(),e.feedback_stamps)
        device.snapshot_nowait = read
        _recovery_feedback(e,epoch=0,failure=failure.value)
        assert e.events[-1]['event'] == 'recovery_clock_rebased'
        assert device.commands == []


def test_in_flight_epoch_change_still_cancels_and_baseline_update_is_atomic():
    e, device, clock = executor_fixture()
    original, read, _ = seed(e, device, clock)
    before = deepcopy(e.feedback_stamps)
    state = read()
    for key in PACKETS[:3]:
        state['feedback_time'][key] = timed_state(original,clock(),epoch=0,step=0.)['feedback_time'][key]
    with pytest.raises(PickRejected,match='代次重置'):
        fresh_feedback(state,clock(),e.feedback_stamps)
    assert e.feedback_stamps == before


def test_waits_for_all_streams_to_cross_epoch(clock_recovery):
    r = clock_recovery
    old_jaw = deepcopy(r.e.feedback_stamps['clock_mapping_evidence']['gripper'])
    def read():
        state = r.read()
        if r.clock()-r.began < .03:
            state['feedback_time']['gripper'] = old_jaw
            state['gripper_stamp_monotonic'] = old_jaw['mapped_monotonic']
        return state
    r.device.snapshot_nowait = read
    _recovery_feedback(r.e,epoch=0,failure=r.failure)
    assert r.clock()-r.began >= .13-1e-9
    assert r.device.commands == []


@pytest.mark.parametrize('fault', [
    'stale','future','backwards','missing','unknown_domain','ambiguous','bad_mapping',
    'inconsistent_step','second_step','duplicate','far_target','moving','persistent_fk',
    'other_fk_fault','cancel','camera_epoch','device_fault','sender_alive','no_anchor',
    'missing_old_evidence','stale_evidence','inconsistent_state_stamp','stuck_one_packet'])
def test_bad_feedback_never_rebases_or_sends(clock_recovery, fault):
    r = clock_recovery
    before = deepcopy(r.e.feedback_stamps)
    frozen_jaw = deepcopy(r.read()['feedback_time']['gripper'])
    def read():
        elapsed = r.clock()-r.began
        state = r.read()
        if fault == 'stale':return timed_state(r.original,r.clock(),stamp=r.clock()-.3)
        if fault == 'future':return timed_state(r.original,r.clock(),stamp=r.clock()+.01)
        if fault == 'backwards' and elapsed > .03:
            return timed_state(r.original,r.clock(),stamp=r.began-.005)
        if fault == 'duplicate':return timed_state(r.original,r.began)
        if fault == 'missing':del state['feedback_time']['gripper']
        if fault == 'unknown_domain':state['feedback_time']['joint_12']['clock_domain'] = 'device_encoder'
        if fault == 'ambiguous':state['feedback_time']['joint_12']['mapping_quality'] = 'ambiguous_epoch'
        if fault == 'bad_mapping':state['feedback_time']['joint_12']['raw_unix_timestamp'] += .03
        if fault == 'inconsistent_step':
            state['feedback_time']['joint_12']['raw_unix_timestamp'] += .03
            state['feedback_time']['joint_12']['reference_wall'] += .03
        if fault == 'second_step' and elapsed > .03:return timed_state(r.original,r.clock(),epoch=2,step=.8)
        if fault == 'far_target':state['joints'][0] += .03
        if fault == 'moving':state['joints'][0] += .005*np.sin(2*np.pi*elapsed/.1)
        if fault == 'stale_evidence':
            state['feedback_time']['joint_12'] = timed_state(r.original,r.clock()-.3)['feedback_time']['joint_12']
        if fault == 'inconsistent_state_stamp':state['joint_stamp_monotonic'] += .002
        if fault == 'stuck_one_packet':
            state['feedback_time']['gripper'] = frozen_jaw
            state['gripper_stamp_monotonic'] = frozen_jaw['mapped_monotonic']
        if fault == 'device_fault':raise RuntimeError('driver fault')
        return state
    r.device.snapshot_nowait = read
    if fault in ('persistent_fk','other_fk_fault'):
        def check(state):
            raise PickRejected('FK mismatch' if fault=='persistent_fk' else 'self collision',
                details={'reason':'feedback_fk_mismatch' if fault=='persistent_fk' else 'self_collision'})
        r.e.planner.model.check_feedback = check
    original_sleep = r.e.sleep
    def sleep(dt):
        original_sleep(dt)
        if fault == 'cancel':r.e.cancelled.set()
        if fault == 'camera_epoch':r.e.epoch = lambda:1
    r.e.sleep = sleep
    if fault == 'sender_alive':r.e.sender = object()
    if fault == 'no_anchor':r.e.command_history.q = None
    if fault == 'missing_old_evidence':r.e.feedback_stamps.pop('clock_mapping_evidence')
    before = deepcopy(r.e.feedback_stamps)
    with pytest.raises(RuntimeError if fault=='device_fault' else PickRejected):
        _recovery_feedback(r.e,epoch=0,failure=r.failure)
    assert r.e.feedback_stamps == before
    assert not any(x['event']=='recovery_clock_rebased' for x in r.e.events)
    assert r.device.commands == []
    assert r.clock()-r.began <= TIMEOUT_S+.01


def test_fk_and_anchor_must_stay_good_for_a_full_new_window(clock_recovery):
    r = clock_recovery
    def check(state):
        if r.clock()-r.began < .04:
            raise PickRejected('FK mismatch',details={'reason':'feedback_fk_mismatch'})
    r.e.planner.model.check_feedback = check
    _recovery_feedback(r.e,epoch=0,failure=r.failure)
    assert r.clock()-r.began >= .14-1e-9
    assert r.device.commands == []


def test_slow_fk_cannot_commit_after_recovery_deadline(clock_recovery):
    r = clock_recovery
    before = deepcopy(r.e.feedback_stamps)
    r.e.planner.model.check_feedback = lambda state:r.clock.sleep(.19)
    with pytest.raises(PickRejected):
        _recovery_feedback(r.e,epoch=0,failure=r.failure)
    assert r.e.feedback_stamps == before
    assert not any(x['event']=='recovery_clock_rebased' for x in r.e.events)
    assert r.device.commands == []


def test_receive_gap_restarts_the_stable_window(clock_recovery):
    r = clock_recovery
    def read():
        elapsed = r.clock()-r.began
        return None if .025 <= elapsed < .3 else r.read()
    r.device.snapshot_nowait = read
    _recovery_feedback(r.e,epoch=0,failure=r.failure)
    assert r.clock()-r.began >= .4-1e-9
    assert r.device.commands == []


@pytest.mark.parametrize('bad', [False, True])
@pytest.mark.parametrize('held', [False, True])
def test_full_recovery_after_step_retains_routes_and_fault_stop(recovery, bad, held):
    r = recovery
    if held:
        r.device.width = .04;r.device.force = 2.
        r.e.holding.held_width = .04;r.e.planner.payload_local = np.zeros((2,3))
    original, read, failure = seed(r.e,r.device,r.clock)
    if bad:
        def faulty():
            state = read();state['feedback_time']['gripper']['clock_domain'] = 'unknown'
            return state
        r.device.snapshot = faulty
    else:
        r.device.snapshot = read
    def fail(_):raise failure
    r.e._run_attempt = fail
    if bad:
        with pytest.raises(RecoveryFailed):r.e.run(r.target)
        assert [c[0] for c in r.device.commands] == ['stop']
    else:
        result = r.e.run(r.target)
        assert result['returned_to_wait']
        expected = ['recovery_lift','recovery_transfer','recovery_return'] if held else ['recovery_return']
        assert [x[1] for x in r.calls if x[0]=='route'] == expected
        rebased = next(row for row in r.e.events if row['event']=='recovery_clock_rebased')
        assert r.device.commands[0][0] == 'prepare'
        assert r.device.commands[0][1] >= rebased['monotonic_s']
        assert not any(c[0]=='stop' for c in r.device.commands)

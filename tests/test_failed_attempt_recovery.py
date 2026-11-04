"""Failure injection against recovery and rearming; all devices and scenes are synthetic."""
from concurrent.futures import Future
from types import SimpleNamespace
from threading import Event
import time
import numpy as np
import pytest
from scipy.interpolate import BPoly

from test_two_second_hardware import executor_fixture
from nero_core.two_second.trajectory import TimedPath
from nero_core.two_second.recovery import RecoveryFailed
from nero_core.two_second.cycle_loop import run_pick_cycles
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


class ImmediatePool:
    def __init__(self, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def submit(self, fn, *a, **kw):
        future = Future()
        try: future.set_result(fn(*a, **kw))
        except BaseException as error: future.set_exception(error)
        return future


class SyntheticSender:
    """Deterministic sender double driving real streaming/hold callbacks."""
    def __init__(self, device, step, after, **kw):
        self.device, self.step, self.after, self.kw = device, step, after, kw
        self.report = SimpleNamespace(running=True, exception=None, command_count=0, last_feedback=None)
    def start(self):
        clock = self.device.clock
        streaming = 'initial_dispatch_time' in self.kw
        if streaming:
            clock.sleep(max(0., self.kw['initial_dispatch_time']-clock()))
        try:
            for _ in range(3000 if streaming else 1):
                clock.sleep(.01)
                state = self.device.snapshot(); self.report.last_feedback = state
                item = self.step(state, clock(), .01)
                if item is None:
                    self.report.running = False
                    break
                q, metadata = item
                if self.kw.get('finalize_send'):
                    q, metadata = self.kw['finalize_send'](state, clock(), .01, q, metadata)
                if self.kw.get('validate_dispatch'):
                    self.kw['validate_dispatch'](state, clock(), .01, q, metadata)
                self.device.move_joints(q)
                self.after(clock(), .01, q, metadata)
                self.report.command_count += 1
            else:
                if streaming: raise AssertionError('synthetic stream did not reach endpoint')
        except BaseException as error:
            self.report.exception = error; self.report.running = False
        return self
    def snapshot(self): return self.report
    def wait_stopped(self, *a): return not self.report.running
    def request_stop(self): self.report.running = False
    def join(self): return True


@pytest.fixture
def recovery(monkeypatch):
    import nero_core.two_second.execution as execution
    import nero_core.two_second.recovery as module
    monkeypatch.setattr(execution, 'JointStreamSender', SyntheticSender)
    monkeypatch.setattr(module, 'ThreadPoolExecutor', ImmediatePool)
    e, device, clock = executor_fixture()
    e.config.raw['trajectory_profile'] = 'smooth'
    device.q[:] = .05
    e.motion_started = True
    e.command_history.record(clock()-.01, .01, device.q, {})
    target = SimpleNamespace(identity=e.owner, epoch=0)
    p = e.planner
    p.target = target; p.identity = target.identity; p.payload_local = p.payload_tree = None
    calls = []
    def install(*a): pytest.fail('the installed frozen scene must be reused')
    p.install = install
    p.valid = lambda *a, **kw: calls.append(('geometry', kw.get('payload', False)))
    p.check_path = lambda path, width, **kw: calls.append(('timed_check', path.stage, kw))
    p.reanchor_support = lambda *a: calls.append(('reanchor',))
    p.validate_release_opening = lambda *a: calls.append(('release_check',))
    def release(*a):
        calls.append(('release',)); p.payload_local = p.payload_tree = None
    p.release_measured = release
    def route(state, *, stage, goal_pose, goal_joints, payload, start_joints, **kw):
        calls.append(('route', stage, payload, goal_pose))
        p.version += 1
        q0 = np.asarray(start_joints)
        q1 = np.asarray(goal_joints) if goal_joints is not None else q0+.01
        zero = np.zeros(7)
        curve = BPoly.from_derivatives([0., .5], [[q0, zero, zero], [q1, zero, zero]])
        return TimedPath(curve, e.owner, p.version, stage, .5, 0.)
    p.route = route
    failure = PickRejected('实际发送已超出本次几何检查时窗')
    def fail(_): raise failure
    e._run_attempt = fail
    return SimpleNamespace(e=e, device=device, clock=clock, target=target, calls=calls, failure=failure)


def test_empty_failure_recovers_without_emergency_stop_and_preserves_cancel_event(recovery):
    r = recovery
    result = r.e.run(r.target)
    assert result['status'] == 'attempt_cancelled_returned_to_wait'
    assert result['returned_to_wait'] and not result['carried_to_bin']
    assert not r.e.cancelled.is_set() and r.e.sender is None
    assert not any(x[0] == 'stop' for x in r.device.commands)
    assert [x[1] for x in r.calls if x[0] == 'route'] == ['recovery_return']
    timed = [x for x in r.calls if x[0] == 'timed_check']
    assert len(timed) == 1 and timed[0][2]['origin'] > 0. and timed[0][2]['payload'] is False
    np.testing.assert_allclose(r.device.q, r.e.config.standby)
    assert r.device.width == .1 and r.device.force == 0.
    assert r.e.owner is None and r.e.stage == 'waiting'


def test_held_workpiece_goes_to_configured_bin_before_release_and_return(recovery):
    r = recovery; r.device.width = .04; r.device.force = 2.
    r.e.holding.held_width = .04; r.e.planner.payload_local = np.zeros((2, 3))
    result = r.e.run(r.target)
    routes = [x for x in r.calls if x[0] == 'route']
    assert [x[1] for x in routes] == ['recovery_lift', 'recovery_transfer', 'recovery_return']
    assert [x[2] for x in routes] == [True, True, False]
    np.testing.assert_array_equal(routes[1][3], r.e.config.bins[r.e.config.default_bin])
    assert r.calls.index(('release',)) > r.calls.index(routes[1])
    assert r.calls.index(('release',)) < r.calls.index(routes[2])
    assert result['carried_to_bin'] and not r.e.cancelled.is_set()
    assert not any(x[0] == 'stop' for x in r.device.commands)


@pytest.mark.parametrize('held', [False, True])
def test_workpiece_collision_during_retreat_returns_without_estop(recovery, held):
    from nero_core.two_second.workpiece_contact import recovery_contact_allowed
    r = recovery
    r.e.config.raw['recovery_allow_workpiece_contact'] = True
    if held:
        r.device.width = .04; r.device.force = 2.
        r.e.holding.held_width = .04
        r.e.planner.payload_local = np.zeros((2, 3))
    checks = []
    def check(*args, **kw):
        checks.append(recovery_contact_allowed(r.e.planner))
        if not checks[-1]:
            raise PickRejected('机器人与实测工件点云碰撞', details={'kind':'robot_workpiece_point_cloud'})
    r.e.planner.valid = check
    def timed_check(*args, **kw): check()
    r.e.planner.check_path = timed_check
    # Covers an initial collision and a miss that encounters the part only
    # during recovery, as in the recorded closed-jaw recovery failure.
    if held:
        def fail(_): raise PickRejected('机器人与实测工件点云碰撞',
            details={'kind':'robot_workpiece_point_cloud'})
        r.e._run_attempt = fail
    result = r.e.run(r.target)
    assert result['returned_to_wait'] and result['carried_to_bin'] == held
    assert result['recovery_allow_workpiece_contact'] is True
    assert checks and all(checks)
    assert not recovery_contact_allowed(r.e.planner)
    assert not r.e.cancelled.is_set() and r.e.sender is None
    assert not any(command[0] == 'stop' for command in r.device.commands)
    np.testing.assert_allclose(r.device.q, r.e.config.standby)


def test_non_workpiece_failure_during_contact_permitted_recovery_still_stops(recovery):
    from nero_core.two_second.workpiece_contact import recovery_contact_allowed
    r = recovery
    r.e.config.raw['recovery_allow_workpiece_contact'] = True
    def reject(*args, **kw): raise PickRejected('模型自碰撞')
    r.e.planner.valid = reject
    with pytest.raises(RecoveryFailed, match='模型自碰撞'):
        r.e.run(r.target)
    assert not recovery_contact_allowed(r.e.planner)
    assert sum(command[0] == 'stop' for command in r.device.commands) == 1


def test_empty_grasp_completion_also_returns_instead_of_ending_run(recovery):
    r = recovery; r.device.width = r.device.force = 0.
    r.e._run_attempt = lambda _: {'status':'attempt_completed_grip_unconfirmed'}
    assert r.e.run(r.target)['returned_to_wait']
    assert r.device.width == .1


def test_recovery_with_real_sender_thread_joins_before_returning_to_wait(recovery, monkeypatch):
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    from nero_core.two_second.command_history import CommandHistory
    class RealClock:
        def __call__(self): return time.monotonic()
        def sleep(self, seconds): time.sleep(seconds)
    r = recovery; clock = RealClock()
    r.e.clock = r.e.holding.clock = r.device.clock = clock
    r.e.sleep = r.e.holding.sleep = clock.sleep
    r.e.config.raw['planning_reserve_s'] = .5
    r.e.command_history = CommandHistory(r.e.config)
    r.e.command_history.record(clock()-.01, .01, r.device.q, {})
    monkeypatch.setattr('nero_core.two_second.execution.JointStreamSender', JointStreamSender)
    report = r.e.run(r.target)
    assert report['returned_to_wait'] and r.e.sender is None
    sent = len(r.device.commands)
    time.sleep(.03)
    assert len(r.device.commands) == sent
    assert not any(x[0] == 'stop' for x in r.device.commands)


def test_recovery_observer_accepts_delayed_snapshot_when_sender_saw_newer_one(recovery, monkeypatch):
    import nero_core.two_second.recovery as module
    from nero_core.two_second.streaming import fresh_feedback as actual_check
    r = recovery
    interleavings=[]
    def interleaved(state, now, previous=None):
        if r.e.stage == 'recovery_settling' and previous is not None:
            # Simulate the sender checking a newer acquisition while this
            # observer is about to inspect its already captured snapshot.
            r.e.feedback_stamps['joint'] = state['joint_stamp']+.001
            interleavings.append(previous is not r.e.feedback_stamps)
        return actual_check(state, now, previous)
    monkeypatch.setattr(module, 'fresh_feedback', interleaved)
    result=r.e.run(r.target)
    assert result['returned_to_wait'] and interleavings and all(interleavings)


def test_recovery_observer_still_rejects_its_own_clock_reversal(recovery, monkeypatch):
    import nero_core.two_second.recovery as module
    from nero_core.two_second.streaming import fresh_feedback as actual_check
    r=recovery; observed=[]
    def reverse(state, now, previous=None):
        if r.e.stage == 'recovery_settling' and previous is not None:
            observed.append(1)
            if len(observed)==3:
                state=dict(state);state['joint_stamp']-=.02
        return actual_check(state, now, previous)
    monkeypatch.setattr(module,'fresh_feedback',reverse)
    with pytest.raises(RecoveryFailed,match='反馈时钟倒退'):
        r.e.run(r.target)
    assert sum(x[0]=='stop' for x in r.device.commands)==1


@pytest.mark.parametrize('fault', ['stale_feedback', 'invalid_geometry', 'route_failure',
    'uncertain_payload', 'position_drift', 'unstable_feedback', 'timed_collision', 'missed_reservation'])
def test_failed_recovery_stops_once_and_never_opens_a_possible_payload(recovery, fault):
    r = recovery
    if fault == 'stale_feedback':
        original = r.device.snapshot
        def snapshot():
            state = original(); state['joint_stamp'] -= .3
            return state
        r.device.snapshot = snapshot
    elif fault == 'invalid_geometry':
        def reject(*a, **kw): raise PickRejected('恢复路径碰撞')
        r.e.planner.valid = reject
    elif fault == 'route_failure':
        def reject(*a, **kw): raise PickRejected('无恢复路径')
        r.e.planner.route = reject
    elif fault == 'uncertain_payload':
        r.device.width = .04; r.device.force = 2.  # no confirmed attachment estimate
    elif fault == 'position_drift':
        r.device.q[0] += np.deg2rad(1.1)
    elif fault == 'timed_collision':
        def reject(*a, **kw): raise PickRejected('恢复轨迹会碰到移动目标')
        r.e.planner.check_path = reject
    elif fault == 'missed_reservation':
        r.e.planner.check_path = lambda *a, **kw: r.clock.sleep(2.)
    else:
        original = r.device.snapshot
        def snapshot():
            state = original()
            state['joints'][0] += .004 if int(r.clock()*100)%2 else -.004
            return state
        r.device.snapshot = snapshot
    with pytest.raises(RecoveryFailed) as failure:
        r.e.run(r.target)
    assert failure.value.details['original_error'] == str(r.failure)
    assert sum(x[0] == 'stop' for x in r.device.commands) == 1
    assert not any(x[0] == 'grip' for x in r.device.commands)
    assert r.e.cancelled.is_set() and r.e.sender is None


def test_user_cancel_never_starts_automatic_recovery(recovery):
    r = recovery; r.e.cancelled.set()
    with pytest.raises(PickRejected): r.e.run(r.target)
    assert not any(x['event'] == 'recovery_started' for x in r.e.events)
    assert not any(x[0] in ('arm', 'grip') for x in r.device.commands)


def test_cancel_during_recovery_prevents_routes_and_next_target(recovery):
    r = recovery
    r.e.event_sink = lambda row: r.e.cancelled.set() if row['event'] == 'recovery_started' else None
    with pytest.raises(RecoveryFailed): r.e.run(r.target)
    assert not any(x[0] == 'route' for x in r.calls)
    assert sum(x[0] == 'stop' for x in r.device.commands) == 1


def test_recovered_attempt_does_not_count_as_success_and_next_target_is_fresh():
    from test_two_second_hardware import Clock
    clock = Clock(); cancelled = Event(); consumed = []; events = []; instances = []
    class Vision:
        clock_epoch = 0
        rearmed = 0
        def consume(self, identity): consumed.append(identity)
        def resume_targets(self): self.rearmed += 1; clock.sleep(.1)
        def poll(self):
            return SimpleNamespace(identity=(0, self.rearmed+1, 1), epoch=0, capture_s=clock(),
                received_s=clock(), device_s=1., sequence=self.rearmed+1, speed_report={}, grasps=(np.eye(4),))
    class Executor:
        def __init__(self, *a, **kw): instances.append(self)
        def run(self, target):
            return ({'status':'attempt_cancelled_returned_to_wait', 'returned_to_wait':True}
                if len(instances) == 1 else {'status':'cycle_completed_from_feedback','returned_to_wait':True})
    result = {'completed_feedback_cycles':0}; reports = []; executors = []; vision = Vision()
    journal = SimpleNamespace(emit=lambda name, **data: events.append((name, data)))
    run_pick_cycles(SimpleNamespace(raw={'maximum_observation_age_s':1.}), None, None, vision,
        count=1, cancelled=cancelled, result=result, reports=reports, executors=executors,
        set_phase=lambda *a, **kw:None, journal=journal, executor_factory=Executor, clock=clock)
    assert result['completed_feedback_cycles'] == 1 and result['recovered_attempts'] == 1
    assert result['attempt_count'] == 2 and vision.rearmed == 1
    assert consumed == [(0,1,1), (0,2,1)] and len(reports) == 2
    assert [n for n, _ in events].count('attempt_recovered') == 1
    assert result['status'] == 'requested_feedback_cycles_completed'


@pytest.mark.parametrize('cancelled_after_recovery', [False, True])
def test_loop_never_rearms_after_failed_recovery_or_user_cancel(cancelled_after_recovery):
    cancelled = Event(); calls = []
    target = SimpleNamespace(identity=(0,1,1), capture_s=1., received_s=1., device_s=1.,
        sequence=1, speed_report={}, grasps=(np.eye(4),))
    vision = SimpleNamespace(clock_epoch=0, poll=lambda:target,
        consume=lambda identity:calls.append('consume'), resume_targets=lambda:calls.append('rearm'))
    class Executor:
        def __init__(self, *a, **kw): pass
        def run(self, target):
            if cancelled_after_recovery:
                cancelled.set()
                return {'status':'attempt_cancelled_returned_to_wait', 'returned_to_wait':True}
            raise RecoveryFailed(PickRejected('发送超时'), PickRejected('回位失败'))
    result = {'completed_feedback_cycles':0}
    def run():
        run_pick_cycles(SimpleNamespace(raw={'maximum_observation_age_s':1.}), None, None, vision,
            count=1, cancelled=cancelled, result=result, reports=[], executors=[],
            set_phase=lambda *a, **kw:None, journal=SimpleNamespace(emit=lambda *a, **kw:None),
            executor_factory=Executor, clock=lambda:1.)
    if cancelled_after_recovery:
        run()
        assert result['status'] == 'cancelled'
    else:
        with pytest.raises(RecoveryFailed): run()
    assert calls == ['consume'] and result['completed_feedback_cycles'] == 0

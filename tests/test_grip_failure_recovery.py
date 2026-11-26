"""Grip-failure recovery: synthetic devices only, including recorded widths."""
import numpy as np
import pytest

from test_failed_attempt_recovery import recovery
from nero_core.two_second.recovery import RecoveryFailed
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def fail_grip(r, reason='grip_width_out_of_range'):
    r.failure = PickRejected('夹持异常', details={'reason': reason})
    def attempt(_):
        raise r.failure
    r.e._run_attempt = attempt


def payload(r, *, width=.028895, previous=.03263, force=1.402):
    r.device.width, r.device.force = width, force
    r.e.holding.held_width = previous
    r.e.planner.payload_local = np.zeros((2, 3))


def test_recorded_tightening_reconfirms_then_drops_before_return_without_estop(recovery):
    r = recovery
    payload(r)
    fail_grip(r)
    begun = r.clock()
    original = r.device.prepare_motion
    def prepare(*args, **kwargs):
        assert r.clock()-begun >= .1-1e-9
        assert r.e.holding.held_width == pytest.approx(.028895)
        return original(*args, **kwargs)
    r.device.prepare_motion = prepare
    estimate = r.e.planner.payload_local
    original_check = r.e.planner.valid
    def geometry(*args, **kwargs):
        if not r.device.commands:
            assert r.e.planner.payload_local is estimate
            assert kwargs['payload'] is True
        return original_check(*args, **kwargs)
    r.e.planner.valid = geometry
    report = r.e.run(r.target)
    routes = [x for x in r.calls if x[0] == 'route']
    assert [x[1] for x in routes] == ['recovery_lift', 'recovery_transfer', 'recovery_return']
    assert [x[2] for x in routes] == [True, True, False]
    assert r.calls.index(('release',)) > r.calls.index(routes[1])
    assert report['carried_to_bin'] and report['returned_to_wait']
    assert not r.e.cancelled.is_set()
    assert not any(x[0] == 'stop' for x in r.device.commands)
    confirmed = [x['data'] for x in r.e.events if x['event'] == 'recovery_grip_confirmed']
    assert len(confirmed) == 1 and confirmed[0]['motion_commands_sent'] == 0
    assert confirmed[0]['previous_width_m'] == .03263


def test_scheduled_unconfirmed_result_keeps_existing_payload_and_goes_to_drop(recovery):
    r = recovery
    payload(r, width=.027848, force=1.877)
    r.e.holding.held_width = None  # The old scheduled-attempt branch cleared this.
    r.e._run_attempt = lambda _: {'status': 'attempt_completed_grip_unconfirmed'}
    report = r.e.run(r.target)
    assert report['carried_to_bin'] and report['returned_to_wait']
    assert report['failure_details']['reason'] == 'grip_unconfirmed'
    assert not any(x[0] == 'stop' for x in r.device.commands)


def test_transient_force_loss_must_recover_for_a_new_confirmation_dwell(recovery):
    r = recovery
    payload(r, previous=.04, width=.04, force=0.)
    fail_grip(r, 'grip_force_lost')
    begun, original = r.clock(), r.device.snapshot
    def snapshot():
        r.device.force = 2. if r.clock()-begun >= .12 and r.device.width < .096 else 0.
        return original()
    r.device.snapshot = snapshot
    r.e.holding.force_lost_since = begun-.3
    report = r.e.run(r.target)
    assert report['carried_to_bin']
    confirmed = next(x['data'] for x in r.e.events if x['event'] == 'recovery_grip_confirmed')
    assert confirmed['waited_s'] >= .22-1e-9
    assert r.e.holding.force_lost_since is None


@pytest.mark.parametrize('has_old_payload', [False, True])
def test_closed_empty_jaws_with_endstop_force_open_then_return_without_estop(recovery, has_old_payload):
    r = recovery
    if has_old_payload:
        payload(r)
    r.device.width, r.device.force = 0., 2.
    r.e._run_attempt = lambda _: {'status': 'attempt_completed_grip_unconfirmed'}
    route = r.e.planner.route
    def checked_route(*args, **kwargs):
        assert r.device.width == .1  # End-stop force removed before empty motion.
        assert r.device.force == 0.
        return route(*args, **kwargs)
    r.e.planner.route = checked_route
    report = r.e.run(r.target)
    assert report['returned_to_wait'] and not report['carried_to_bin']
    assert [x[1] for x in r.calls if x[0] == 'route'] == ['recovery_return']
    assert not any(x[0] == 'stop' for x in r.device.commands)


@pytest.mark.parametrize('fault', ['no_payload', 'oscillating_width', 'low_force',
    'repeated_gripper_stamp', 'geometry', 'joint_drift', 'cancel', 'driver'])
def test_uncertain_or_faulted_regrip_cannot_authorize_transport_or_release(recovery, fault):
    r = recovery
    payload(r)
    fail_grip(r)
    begun, original = r.clock(), r.device.snapshot
    if fault == 'no_payload':
        r.e.planner.payload_local = None
    elif fault == 'geometry':
        def collision(*args, **kwargs):
            raise PickRejected('模型自碰撞')
        r.e.planner.valid = collision
    else:
        def snapshot():
            state = original()
            if fault == 'oscillating_width':
                state['width_m'] += .004 if int(round(r.clock()*100)) % 2 else 0.
            elif fault == 'low_force':
                state['force_n'] = 0.
            elif fault == 'repeated_gripper_stamp':
                state['gripper_stamp'] = begun
            elif r.clock()-begun > .03:
                if fault == 'joint_drift':
                    state['joints'][0] += np.deg2rad(1.1)
                elif fault == 'cancel':
                    r.e.cancelled.set()
                elif fault == 'driver':
                    raise PickRejected('设备驱动故障')
            return state
        r.device.snapshot = snapshot
    with pytest.raises(RecoveryFailed):
        r.e.run(r.target)
    assert sum(x[0] == 'stop' for x in r.device.commands) == 1
    assert not any(x[0] in ('arm', 'grip') for x in r.device.commands)
    assert not any(x[0] == 'route' for x in r.calls)


def test_a_second_width_change_after_confirmation_is_not_rebased_again(recovery):
    r = recovery
    payload(r)
    fail_grip(r)
    original = r.device.snapshot
    def snapshot():
        state = original()
        if any(x['event'] == 'recovery_grip_confirmed' for x in r.e.events):
            state['width_m'] += .004
        return state
    r.device.snapshot = snapshot
    with pytest.raises(RecoveryFailed, match='夹持开口异常'):
        r.e.run(r.target)
    assert sum(x['event'] == 'recovery_grip_confirmed' for x in r.e.events) == 1
    assert not any(x[0] == 'grip' for x in r.device.commands)


def test_other_failure_does_not_reset_the_original_retention_baseline(recovery):
    r = recovery
    payload(r)
    with pytest.raises(RecoveryFailed, match='夹持开口异常'):
        r.e.run(r.target)
    assert not any(x['event'] == 'recovery_grip_confirmed' for x in r.e.events)
    assert not any(x[0] == 'grip' for x in r.device.commands)


@pytest.mark.parametrize('result', ['loaded', 'empty', 'ambiguous', 'persistent_collision'])
def test_collision_after_real_close_rechecks_then_selects_route(recovery, monkeypatch, result):
    from nero_core.two_second import streaming
    from nero_core.two_second.reference import Intercept
    from nero_core.two_second.trajectory import TimedPath

    r = recovery
    r.device.empty = result == 'empty'
    r.e.config.raw['closure_policy'] = 'scheduled_attempt'
    q = r.device.q.copy()
    origin = r.clock() + .2
    path = TimedPath(lambda t, d=0: q if d == 0 else np.zeros(7),
                     r.target.identity, r.e.planner.version, 'intercept_follow', .6, 0.)
    ref = Intercept(origin, origin, .2, np.zeros(3), np.zeros(3), np.zeros(3),
                    [1,0,0,0], [1,0,0,0])
    def sweep(planner, moving_path, request):
        if moving_path is path and request['width_m'] < .096:
            raise PickRejected('路径碰到可见障碍点云：left_finger_geom',
                               details={'kind': 'scene_collision', 'point_count': 1})
    monkeypatch.setattr(streaming, 'sweep', sweep)
    attach_times = []
    def attach(contact, stamp):
        assert stamp == contact['joint_stamp']
        assert r.clock()-stamp >= .1-1e-9
        attach_times.append(stamp)
        r.e.planner.payload_local = np.zeros((2, 3))
    r.e.planner.attach_measured = attach
    if result == 'ambiguous':
        original = r.device.grip
        def low_force(width, force):
            original(width, force)
            r.device.force = 0.
        r.device.grip = low_force
    if result == 'persistent_collision':
        def reject(*args, **kwargs):
            raise PickRejected('左夹指仍与带面碰撞')
        r.e.planner.valid = reject
    r.e._run_attempt = lambda _: r.e._stream(path, origin=origin, epoch=0, reference=ref)
    if result in ('ambiguous', 'persistent_collision'):
        with pytest.raises(RecoveryFailed):
            r.e.run(r.target)
        assert not any(c[0] == 'route' for c in r.calls)
        assert [c for c in r.device.commands if c[0] == 'grip'] == [('grip', 0.)]
    else:
        report = r.e.run(r.target)
        assert report['returned_to_wait'] and report['carried_to_bin'] == (result == 'loaded')
        routes = [c[1] for c in r.calls if c[0] == 'route']
        if result == 'loaded':
            assert len(attach_times) == 1
            assert routes == ['recovery_lift', 'recovery_transfer', 'recovery_return']
            assert ('release',) in r.calls
        else:
            assert not attach_times and routes == ['recovery_return']
            assert any(x['event'] == 'recovery_grip_empty' for x in r.e.events)
    assert any(x['event'] == 'recovery_grip_wait' and
               x['data']['trigger'] == 'post_close_collision' for x in r.e.events)


def test_late_first_loaded_feedback_cannot_locate_payload(recovery):
    from nero_core.two_second.grip_recovery import ClosingGrip
    r = recovery
    closing = ClosingGrip(r.target, r.clock(), r.e.config.settings)
    r.clock.sleep(.2)
    r.device.width, r.device.force = .04, 1.
    error = PickRejected('带面碰撞', details={'kind': 'scene_collision'})
    closing.mark_collision(error)
    def fail(_):
        raise error
    r.e._run_attempt = fail
    with pytest.raises(RecoveryFailed) as failure:
        r.e.run(r.target)
    assert failure.value.details['recovery_details']['reason'] == 'missing_grip_contact_feedback'
    assert not any(c[0] in ('grip', 'arm') for c in r.device.commands)


def test_stale_sweep_without_geometric_hit_does_not_arm_grip_recheck(recovery):
    from nero_core.two_second.grip_recovery import ClosingGrip
    closing = ClosingGrip(recovery.target, recovery.clock(), recovery.e.config.settings)
    error = PickRejected('后台碰撞监测结果过期', details={'reason': 'runtime_sweep_expired'})
    closing.mark_collision(error)
    assert not hasattr(error, 'closing_grip')

"""Offline mode handoff and fault injection; these do not prove real tracking."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

from test_two_second_hardware import executor_fixture, site
from test_failed_attempt_recovery import SyntheticSender
from nero_core.two_second.config import CellConfig
from nero_core.two_second.trajectory import TimedPath
from nero_core.two_second.wait_precision import (
    finish_wait, enabled_for, JointStillness, MAX_CORRECTION_RAD,
    JS_ENDPOINT_HOLD_S, JOINT_ARRIVAL_TOLERANCE_RAD, STILL_WINDOW_S, STILL_RANGE_RAD)
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


@pytest.fixture
def precision():
    e, device, clock = executor_fixture()
    e.config.raw.update(return_wait_precision_enabled=True,
        hardware={'joint_command_mode': 'js'})
    device.mode = 'js'
    device.q[5] = np.deg2rad(1.25)
    def configure(mode):
        assert e.sender is None, 'a live JS sender must never overlap J positioning'
        device.commands.append(('mode', mode))
        device.mode = mode
    device.configure_joint_command_mode = configure
    def send(q):
        device.commands.append(('target', device.mode, clock(), np.asarray(q).copy()))
        device.q = np.asarray(q).copy()
        if device.mode == 'js':
            device.q[5] += np.deg2rad(1.25)
    device.move_joints = send
    e.command_history.record(clock()-.01, .01, np.zeros(7), {})
    path = TimedPath(lambda t, d=0: np.zeros(7), e.owner, e.planner.version,
        'prepared_taught_return', .3, 0.)
    return SimpleNamespace(e=e, device=device, clock=clock, path=path)


def targets(r):
    return [call for call in r.device.commands if call[0] == 'target']


def test_precision_recovers_recorded_sized_error_with_one_unchanged_j_target(precision):
    r = precision
    state = finish_wait(r.e, r.path, epoch=0)
    assert len(targets(r)) == 1 and targets(r)[0][1] == 'j'
    np.testing.assert_array_equal(targets(r)[0][3], r.e.config.standby)
    np.testing.assert_array_equal(state['joints'], r.e.config.standby)
    assert r.device.mode == 'js' and r.clock() >= 1.+.3-1e-9
    events = [x['event'] for x in r.e.events]
    assert events == ['return_wait_precision_started', 'return_wait_precision_target_sent',
                      'return_wait_precision_finished']
    row = r.e.trace.command_rows[-1]
    assert row['joint_response_diagnostics']['joint_command_api'] == 'move_j'
    assert row['joint_response_diagnostics']['controller_trajectory_certified'] is False
    assert not any(call[0] == 'grip' for call in r.device.commands)


def test_existing_accurate_wait_does_not_switch_or_send(precision):
    r = precision; r.device.q[:] = 0.
    state = finish_wait(r.e, r.path, epoch=0)
    assert not r.device.commands
    assert r.e.events[-1]['data']['joint_commands'] == 0
    np.testing.assert_array_equal(state['joints'], np.zeros(7))


def test_gradual_manufacturer_motion_is_observed_until_stable_without_resending(precision):
    r = precision; original = r.device.snapshot; native = {}
    def send(q):
        r.device.commands.append(('target', r.device.mode, r.clock(), np.asarray(q).copy()))
        native.update(start=r.device.q.copy(), goal=np.asarray(q).copy(), began=r.clock())
    def snapshot():
        if native:
            progress = min(1., (r.clock()-native['began'])/.4)
            r.device.q = native['start']+(native['goal']-native['start'])*progress
        return original()
    r.device.move_joints = send
    r.device.snapshot_nowait = snapshot
    state = finish_wait(r.e, r.path, epoch=0)
    assert len(targets(r)) == 1 and r.device.mode == 'js'
    assert r.clock() >= native['began']+.4+.1-1e-9
    tail = [row['feedback_q_rad'] for row in r.e.trace.feedback_rows
        if row['feedback_read_monotonic_s'] >= r.clock()-STILL_WINDOW_S-1e-9]
    assert np.max(np.ptp(tail, axis=0)) <= STILL_RANGE_RAD+1e-9
    np.testing.assert_array_equal(state['joints'], r.e.config.standby)


def test_real_stream_joins_js_then_aligns_instead_of_waiting_twelve_seconds(precision, monkeypatch):
    r = precision
    monkeypatch.setattr('nero_core.two_second.execution.JointStreamSender', SyntheticSender)
    state = r.e._stream(r.path, origin=r.clock()+.1, epoch=0)
    js = [x for x in targets(r) if x[1] == 'js']
    j = [x for x in targets(r) if x[1] == 'j']
    assert js and len(j) == 1 and js[-1][2] < j[0][2]
    assert r.e.sender is None and r.device.mode == 'js'
    assert r.clock() < 1.+r.path.duration+JS_ENDPOINT_HOLD_S+1.
    assert max(abs(np.asarray(state['joints']))) <= JOINT_ARRIVAL_TOLERANCE_RAD
    assert r.e.events[-1]['event'] == 'stage_finished'
    assert not any(x[0] in ('stop', 'grip') for x in r.device.commands)


@pytest.mark.parametrize('kind', ['disabled', 'carrying', 'reference', 'nonreturn', 'nonwait', 'j_mode'])
def test_precision_is_limited_to_enabled_empty_js_return_to_configured_wait(precision, kind):
    r = precision; path = r.path; carrying = False; reference = None
    if kind == 'disabled': r.e.config.raw['return_wait_precision_enabled'] = False
    if kind == 'carrying': carrying = True
    if kind == 'reference': reference = object()
    if kind == 'nonreturn': path = replace(path, stage='transfer')
    if kind == 'nonwait': r.e.config.standby[0] = .1
    if kind == 'j_mode': r.e.config.raw['hardware']['joint_command_mode'] = 'j'
    assert not enabled_for(r.e, path, reference=reference, carrying=carrying)
    assert not r.device.commands


@pytest.mark.parametrize('kind', ['large_error', 'loaded', 'unknown_jaw', 'stale',
                                  'fk', 'geometry', 'cancelled', 'version', 'sender', 'wrong_anchor'])
def test_preflight_faults_never_switch_or_send_a_precision_target(precision, kind):
    r = precision
    if kind == 'large_error': r.device.q[5] = MAX_CORRECTION_RAD+.001
    if kind == 'loaded': r.device.force = r.e.config.settings.min_force_n
    if kind == 'unknown_jaw': r.device.width = .04
    if kind == 'stale':
        original = r.device.snapshot
        def stale():
            state = original(); state['joint_stamp'] -= .3
            return state
        r.device.snapshot_nowait = stale
    if kind == 'fk':
        r.e.planner.model.check_feedback = lambda s: (_ for _ in ()).throw(PickRejected('FK不符'))
    if kind == 'geometry':
        def valid(q, *a, **kw):
            if abs(q[5]) < .005: raise PickRejected('修正连接碰撞')
        r.e.planner.valid = valid
    if kind == 'cancelled': r.e.cancelled.set()
    if kind == 'version': r.e.planner.version += 1
    if kind == 'sender': r.e.sender = object()
    if kind == 'wrong_anchor': r.e.command_history.q[0] = .1
    with pytest.raises(PickRejected): finish_wait(r.e, r.path, epoch=0)
    assert not targets(r)
    assert not any(x[0] == 'mode' for x in r.device.commands)


def test_moving_feedback_cannot_trigger_j_switch(precision):
    r = precision; original = r.device.snapshot; count = [0]
    def moving():
        count[0] += 1
        r.device.q[5] = np.deg2rad(1.25+(.1 if count[0] % 2 else -.1))
        return original()
    r.device.snapshot_nowait = moving
    with pytest.raises(PickRejected, match='未在时限内'): finish_wait(r.e, r.path, epoch=0)
    assert not targets(r) and not any(x[0] == 'mode' for x in r.device.commands)


def test_transient_fk_mismatch_needs_new_valid_feedback_before_precision(precision):
    r = precision
    def check(state):
        if r.clock() < 1.02:
            raise PickRejected('FK不符', details={'reason': 'feedback_fk_mismatch'})
    r.e.planner.model.check_feedback = check
    finish_wait(r.e, r.path, epoch=0)
    assert len(targets(r)) == 1 and targets(r)[0][2] >= 1.17-1e-9
    assert any(x['event'] == 'return_feedback_rechecked' for x in r.e.events)


def test_persistent_fk_mismatch_keeps_original_short_recheck_deadline(precision):
    r = precision
    def check(state):
        raise PickRejected('FK不符', details={'reason': 'feedback_fk_mismatch'})
    r.e.planner.model.check_feedback = check
    with pytest.raises(PickRejected, match='FK不符'): finish_wait(r.e, r.path, epoch=0)
    assert r.clock() <= 1.04 and not targets(r)


@pytest.mark.parametrize('fault', ['no_convergence', 'cancel_after_send', 'leaves_envelope', 'sdk_send'])
def test_failed_j_correction_never_reports_success_and_restores_local_api(precision, fault):
    r = precision; original = r.device.move_joints
    def send(q):
        original(q)
        if fault == 'no_convergence': r.device.q[5] = np.deg2rad(1.25)
        if fault == 'cancel_after_send': r.e.cancelled.set()
        if fault == 'leaves_envelope': r.device.q[0] = np.deg2rad(.1)
        if fault == 'sdk_send': raise RuntimeError('synthetic SDK send failure')
    r.device.move_joints = send
    with pytest.raises((PickRejected, RuntimeError)): finish_wait(r.e, r.path, epoch=0)
    assert len(targets(r)) == 1 and r.device.mode == 'js'
    assert not any(x['event'] == 'return_wait_precision_finished' for x in r.e.events)


def test_repeated_packets_or_a_gap_cannot_prove_stillness():
    gate = JointStillness(); q = np.zeros(7)
    for _ in range(50): assert not gate.observe(q, 1.)
    assert not gate.observe(q, 1.1)
    for t in (1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24):
        assert not gate.observe(q, t)
    assert gate.observe(q, 1.26)


@pytest.mark.parametrize('value', [True, False, 'true', 1, None])
def test_configuration_requires_boolean_precision_policy(site, value):
    path, data = site; data['return_wait_precision_enabled'] = value
    path.write_text(yaml.safe_dump(data))
    if isinstance(value, bool):
        assert CellConfig.load(path).raw['return_wait_precision_enabled'] is value
    else:
        with pytest.raises(ValueError, match='return_wait_precision_enabled'):
            CellConfig.load(path)

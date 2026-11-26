"""Fixed next-object reservations: synthetic clocks/planners, no device access."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_return_lookahead import fixture, wait_state, target, Planner
from test_two_second_hardware import site


def fixed_fixture():
    worker, clock, item, vision, planner, events = fixture()
    worker.config.raw['lookahead_late_return_policy'] = 'skip_target'

    def intercept(item, state, *, cancelled):
        assert planner.intercept_not_before_s is None  # No previous return estimate.
        assert not cancelled()
        planner.calls.append((item, state))
        planner.identity = item.identity
        planner.version += 1
        planner.scene_revision += 1
        return NS(target=item, scene_revision=planner.scene_revision,
            reference=NS(capture_s=item.capture_s, start_s=clock()+.4),
            path=NS(identity=item.identity, version=planner.version,
                    q0=np.asarray(state['joints']), planning_s=.05))

    planner.intercept = intercept
    return worker, clock, item, vision, planner, events


def deadline(worker):
    return worker.ready.plan.reference.start_s-worker.config.raw['command_advance_s']


def test_late_return_prediction_does_not_move_or_cancel_reservation():
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker.return_started(identity=worker.owner, expected_ready_s=clock()+30.)
    worker._step()
    old, dispatch, revision = worker.ready, deadline(worker), worker.schedule_revision
    for predicted in (clock()+40., clock()+5., clock()+50.):
        worker.return_started(identity=worker.owner, expected_ready_s=predicted)
        worker._step()
    assert worker.ready is old and deadline(worker) == dispatch
    assert len(planner.calls) == 1 and worker.schedule_revision == revision
    assert worker.poll_ready() is None and not vision.discarded
    # Actual early return wins over the pessimistic prediction.
    worker.confirm_return(wait_state(clock, np.deg2rad(.9)))
    assert worker.claim(worker.poll_ready(), Planner()) is old
    assert not any(n.endswith('_replan') for n, _ in events)


@pytest.mark.parametrize('callback', ['prediction', 'confirmation'])
def test_return_update_during_solve_does_not_cancel_inflight_result(callback):
    worker, clock, item, vision, planner, events = fixed_fixture()
    original = planner.intercept
    def intercept(*args, **kwargs):
        if callback == 'prediction':
            worker.return_started(identity=worker.owner, expected_ready_s=clock()+20.)
        else:
            worker.confirm_return(wait_state(clock, np.deg2rad(.8)))
        assert not kwargs['cancelled']()
        return original(*args, **kwargs)
    planner.intercept = intercept
    worker._step()
    assert worker.ready.target is item and len(planner.calls) == 1 and not vision.discarded


@pytest.mark.parametrize('expire_via', ['worker', 'poll', 'late_return'])
def test_deadline_skips_busy_target_once_and_plans_next_without_return_time(expire_via):
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker._step()
    clock.now = deadline(worker)  # Equality is already too late to prepare sending.
    if expire_via == 'worker': worker._step()
    elif expire_via == 'poll': assert worker.poll_ready() is None
    else: worker.confirm_return(wait_state(clock))
    assert worker.ready is None and worker.target is None
    assert vision.discarded == [item.identity]
    row = next(d for n, d in events if n == 'lookahead_rejected')
    assert row['failure_details']['next_action'] == 'skip_target'
    assert row['failure_details']['return_confirmed'] is False
    for _ in range(3): worker._step()
    assert len(planner.calls) == 1 and vision.discarded == [item.identity]
    next_item = target(3)
    next_item.capture_s = clock()
    vision.latest = next_item
    worker._step()
    assert worker.ready.target is next_item and len(planner.calls) == 2


@pytest.mark.parametrize('fresh', [True, False])
def test_actual_feedback_must_be_fresh_before_original_deadline(fresh):
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker._step()
    worker.confirm_return(wait_state(clock))
    old = worker.ready
    clock.now = deadline(worker)-.001
    if fresh:
        worker.update_wait_state(wait_state(clock))
        assert worker.claim(worker.poll_ready(), Planner()) is old
    else:
        assert worker.poll_ready() is None
        clock.now += .001
        worker._step()
        assert worker.ready is None and vision.discarded == [item.identity]
    assert len(planner.calls) == 1


def test_expired_handoff_cannot_use_opted_in_repair_or_wait_for_restored_feedback():
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker.config.raw['delivery_repair'] = {'enabled': True}
    worker._step()
    old = worker.ready
    worker.confirm_return(wait_state(clock))
    bad = wait_state(clock)
    del bad['joint_stamp']
    assert not worker.update_wait_state(bad)
    clock.now = deadline(worker)
    with pytest.raises(PickRejected):
        old.validate(worker.config, 0, clock(), allow_expired_for_repair=True)
    worker._step()
    worker.update_wait_state(wait_state(clock))
    worker._step()
    assert worker.ready is None and len(planner.calls) == 1
    assert vision.discarded == [item.identity]


def test_claim_deadline_race_discards_instead_of_replanning():
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker._step()
    worker.confirm_return(wait_state(clock))
    ready = worker.poll_ready()
    clock.now = deadline(worker)
    worker.update_wait_state(wait_state(clock))
    with pytest.raises(PickRejected) as error:
        worker.claim(ready, Planner())
    worker.discard_ready(ready, error.value)
    worker._step()
    assert vision.discarded == [item.identity] and len(planner.calls) == 1


def test_changed_actual_start_skips_rather_than_assigning_new_start_time():
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker.confirm_return(wait_state(clock, -.01))
    worker._step()
    worker.update_wait_state(wait_state(clock, .01))
    worker._step()
    assert worker.ready is None and len(planner.calls) == 1
    assert vision.discarded == [item.identity]
    assert events[-1][1]['failure_details']['reason'] == 'lookahead_start_mismatch'


def test_failed_speculation_is_not_retried_when_return_time_becomes_known():
    worker, clock, item, vision, planner, events = fixed_fixture()
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise PickRejected('IK branch unavailable', details={'stage': 'join'})
    planner.intercept = fail
    worker._step()
    worker.return_started(identity=worker.owner, expected_ready_s=clock()+10.)
    worker.confirm_return(wait_state(clock))
    worker._step()
    assert len(calls) == 1 and vision.discarded == [item.identity]
    assert not any(n == 'lookahead_planning_deferred' for n, _ in events)


def test_clock_rebase_discards_old_plan_and_does_not_replan_same_exposure():
    from test_lookahead_clock_rebase import sample
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker.confirm_return(sample(clock, epoch=0, step=0.))
    worker._step()
    clock.now += .01
    assert not worker.update_wait_state(sample(clock))
    assert worker.ready is None and vision.discarded == [item.identity]
    for _ in range(3):
        clock.now += .04
        worker.update_wait_state(sample(clock))
        worker._step()
    assert worker.wait_state is not None and len(planner.calls) == 1


def test_cycle_loop_skips_busy_second_object_then_executes_third():
    from nero_core.two_second.cycle_loop import run_pick_cycles
    worker, clock, second, vision, planner, events = fixed_fixture()
    first, third = target(1), target(3)
    worker.owner = None
    vision.latest, vision.active = first, None
    selected = []
    class Executor:
        def __init__(self, config, planner, device, **kwargs): pass
        def run(self, item):
            selected.append(item.identity)
            if item is first:
                vision.latest = second
                worker._step()
                clock.now = deadline(worker)
                worker._step()
                third.capture_s = clock()
                vision.latest = third
                worker._step()
            else:
                assert item is third and self.prefetched_intercept.target is third
            return {'status': 'cycle_completed_from_feedback', 'returned_to_wait': True}
    result = {'completed_feedback_cycles': 0}
    run_pick_cycles(worker.config, Planner(), NS(snapshot=lambda: wait_state(clock)), vision,
        count=2, cancelled=worker.cancelled, result=result, reports=[], executors=[],
        set_phase=lambda *a, **kw: None, journal=NS(emit=lambda *a, **kw: None),
        executor_factory=Executor, clock=clock, lookahead=worker)
    assert result['completed_feedback_cycles'] == 2
    assert selected == [first.identity, third.identity]
    assert vision.discarded == [second.identity]
    assert len(planner.calls) == 2


@pytest.mark.parametrize('change', ['epoch', 'scene', 'version', 'identity', 'target'])
def test_invalidated_results_still_cannot_be_claimed(change):
    worker, clock, item, vision, planner, events = fixed_fixture()
    worker._step()
    worker.confirm_return(wait_state(clock))
    if change == 'epoch': vision.clock_epoch += 1
    elif change == 'scene': planner.scene_revision += 1
    elif change == 'version': planner.version += 1
    elif change == 'identity': planner.identity = (0, 9, 9)
    elif change == 'target': vision.latest = target(9)
    assert worker.poll_ready() is None
    assert vision.discarded == [item.identity] and len(planner.calls) == 1


@pytest.mark.parametrize('policy', ['skip_target', 'replan', 'invalid', True])
def test_policy_config_validation(site, policy):
    from nero_core.two_second.config import CellConfig
    path, raw = site
    raw['lookahead_late_return_policy'] = policy
    path.write_text(yaml.safe_dump(raw))
    if policy in ('skip_target', 'replan'):
        assert CellConfig.load(path).raw['lookahead_late_return_policy'] == policy
    else:
        with pytest.raises(ValueError, match='lookahead_late_return_policy'):
            CellConfig.load(path)


@pytest.mark.parametrize('expiry_stage', ['handoff', 'device_prepare'])
def test_executor_never_repairs_or_streams_expired_prefetch(monkeypatch, expiry_stage):
    from test_delivery_repair import full_cycle
    executor, item, events, stages, calls, repairs = full_cycle(monkeypatch, prefetched=True)
    executor.config.raw['lookahead_late_return_policy'] = 'skip_target'
    if expiry_stage == 'device_prepare':
        old = executor.prefetched_intercept
        plan = replace(old.plan, reference=replace(old.plan.reference, start_s=executor.clock()+.15))
        executor.prefetched_intercept = replace(old, plan=plan)
    with pytest.raises(PickRejected) as error:
        executor._run_attempt(item)
    assert error.value.details['reason'] == 'lookahead_send_deadline_missed'
    assert not calls and not repairs and not stages


def test_return_wait_time_is_not_learned_as_dispatch_preparation_cost(monkeypatch):
    from test_delivery_repair import full_cycle
    executor, item, events, stages, calls, repairs = full_cycle(monkeypatch, prefetched=True)
    executor.config.raw.update(lookahead_late_return_policy='skip_target',
                               planning_reserve_mode='measured')
    old = executor.prefetched_intercept
    path = replace(old.plan.path, diagnostics={'measured_planning': {
        'completed_monotonic_s': executor.clock()-.2}})
    plan = replace(old.plan, path=path,
                   reference=replace(old.plan.reference, start_s=executor.clock()+.3))
    executor.prefetched_intercept = replace(old, plan=plan)
    def stream(path, **kwargs):
        stages.append(path.stage)
        if path.stage == 'intercept_follow':
            executor.raw_device.width = .04
            executor.raw_device.force = 2.
            executor.holding.held_width = .04
        return executor.device.snapshot()
    monkeypatch.setattr(executor, '_stream', stream)
    result = executor._run_attempt(item)
    assert result['status'] == 'cycle_completed_from_feedback'
    assert stages == ['intercept_follow', 'lift', 'transfer', 'return']
    assert executor.planner.measured_post_planning_s == pytest.approx(.06)
    assert not calls and not repairs

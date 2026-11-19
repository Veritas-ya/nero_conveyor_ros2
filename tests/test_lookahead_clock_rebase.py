"""Read-only return lookahead across a host wall-clock step."""
from copy import deepcopy

import pytest

from test_host_clock_recovery import timed_state
from test_return_lookahead import Planner, fixture, target, wait_state


def sample(clock, *, epoch=1, step=-.418, offset=0.):
    return timed_state(lambda: wait_state(clock, offset), clock(), epoch=epoch, step=step)


def waiting_with_path(clock_step=-.418):
    worker, clock, item, vision, planner, events = fixture()
    worker.confirm_return(sample(clock, epoch=0, step=0.))
    worker._step()
    original = worker.poll_ready()
    assert original.target is item
    clock.now += .01
    assert not worker.update_wait_state(sample(clock, step=clock_step))
    assert worker.ready is None and worker.poll_ready() is None
    assert worker.wait_feedback_stamps['pose_clock_epoch'] == (0, 0, 0)
    return worker, clock, item, vision, planner, events


@pytest.mark.parametrize('step', [-.418, .418])
def test_replan_original_exposure_on_confirmed_new_epoch(step):
    worker, clock, item, vision, planner, events = waiting_with_path(step)
    first = sample(clock, step=step)
    for _ in range(3):
        clock.now += .01
        assert not worker.update_wait_state(first)
        worker._step()
    assert len(planner.calls) == 1
    for i in range(2):
        clock.now += .04
        assert worker.update_wait_state(sample(clock, step=step)) == (i == 1)
    worker._step()
    prepared = worker.poll_ready()
    assert prepared.target is item and prepared.plan.reference.capture_s == 10.
    assert worker.claim(prepared, Planner()) is prepared
    assert len(planner.calls) == 2 and not vision.discarded
    evidence, = [data for name, data in events if name == 'lookahead_wait_clock_rebased']
    assert evidence['host_wall_step_s'] == pytest.approx(step)
    assert evidence['stable_samples'] == 3 and evidence['robot_commands'] == 0


@pytest.mark.parametrize('fault', ['outside_wait', 'closed', 'stale', 'ambiguous',
                                    'mixed', 'missing', 'wrong_offset', 'no_host_step', 'unstable', 'loaded'])
def test_invalid_feedback_cannot_rebase(fault):
    worker, clock, _, _, planner, events = waiting_with_path()
    for i in range(7):
        clock.now += .03
        state = sample(clock, offset=.006 if fault == 'unstable' and i % 2 else 0.)
        packet = state['feedback_time']['joint_12']
        if fault == 'outside_wait':
            state['joints'][0] = .02
        elif fault == 'closed':
            state['width_m'] = .04
        elif fault == 'loaded':
            state['force_n'] = 2.
        elif fault == 'no_host_step':
            state = sample(clock, step=0.)
        elif fault == 'stale':
            state['pose_stamp_monotonic'] -= .3
        elif fault == 'ambiguous':
            packet['mapping_quality'] = 'ambiguous_epoch'
        elif fault == 'mixed':
            packet['clock_epoch'] = 0
        elif fault == 'missing':
            del state['feedback_time']['joint_12']
        elif fault == 'wrong_offset':
            packet['raw_unix_timestamp'] += .2
            packet['reference_wall'] += .2
        assert not worker.update_wait_state(state)
        worker._step()
    assert len(planner.calls) == 1 and worker.poll_ready() is None
    assert worker.wait_feedback_stamps['joint_clock_epoch'] == (0,)*4
    assert not any(name == 'lookahead_wait_clock_rebased' for name, _ in events)
    for _ in range(6):
        clock.now += .03
        accepted = worker.update_wait_state(sample(clock))
    assert accepted and worker.wait_feedback_stamps['joint_clock_epoch'] == (1,)*4


def test_stationary_cached_gripper_cannot_extend_window_and_expired_target_is_released():
    worker, clock, item, vision, planner, events = waiting_with_path()
    first = sample(clock)
    for _ in range(5):
        clock.now += .03
        partial = sample(clock)
        partial['feedback_time']['gripper'] = deepcopy(first['feedback_time']['gripper'])
        partial['gripper_stamp_monotonic'] = first['gripper_stamp_monotonic']
        assert not worker.update_wait_state(partial)
    clock.now += 6.  # Original exposure now predicts a position past the downstream edge.
    for i in range(3):
        clock.now += .06
        assert worker.update_wait_state(sample(clock)) == (i == 2)
    worker._step()
    assert vision.discarded == [item.identity] and len(planner.calls) == 1
    assert events[-1][1]['failure_details']['reason'] == 'lookahead_target_past_region'
    later = target(3)
    later.capture_s = clock()
    vision.latest = later
    worker._step()
    assert worker.poll_ready().target is later


def test_clock_change_during_planning_invalidates_inflight_result():
    worker, clock, item, _, planner, _ = fixture()
    worker.confirm_return(sample(clock, epoch=0, step=0.))
    original = planner.intercept
    def step_after_solve(*args, **kwargs):
        plan = original(*args, **kwargs)
        clock.now += .01
        assert not worker.update_wait_state(sample(clock))
        assert kwargs['cancelled']()
        return plan
    planner.intercept = step_after_solve
    worker._step()
    assert worker.poll_ready() is None and worker.target is item
    planner.intercept = original
    for _ in range(3):
        clock.now += .06
        worker.update_wait_state(sample(clock))
    worker._step()
    assert worker.poll_ready().target is item


def test_clock_mapper_ambiguous_backward_step_then_replans_from_actual_wait():
    from mujoco_conveyor_sorting.runtime.feedback_clock import FeedbackClock
    from nero_core.two_second.clock_recovery import PACKETS

    worker, clock, _, _, planner, _ = fixture()
    offset = 1000.
    mapper = FeedbackClock(monotonic=clock, wall=lambda: clock()+offset)

    def read():
        state = wait_state(clock)
        raw = clock() + offset - .001
        packets = {k: mapper.receive(k, raw).to_dict() for k in PACKETS}
        state['feedback_time'] = packets
        for kind, key in [('pose', PACKETS[0]), ('joint', PACKETS[3]), ('gripper', PACKETS[7])]:
            state[kind+'_stamp_monotonic'] = packets[key]['mapped_monotonic']
        return state

    worker.confirm_return(read())
    worker._step()
    clock.now += .01
    offset -= .418
    mapper.receive('probe', clock()+offset)
    clock.now += .01  # Both epochs now cover the packet's raw Unix timestamp.
    ambiguous = read()
    assert ambiguous['feedback_time']['joint_12']['mapping_quality'] == 'ambiguous_epoch'
    assert not worker.update_wait_state(ambiguous)
    assert worker.poll_ready() is None
    clock.now += .45
    for i in range(3):
        clock.now += .05
        assert worker.update_wait_state(read()) == (i == 2)
    worker._step()
    assert len(planner.calls) == 2
    assert worker.poll_ready().start_state['feedback_time']['joint_12']['clock_epoch'] == 1


def test_in_region_but_insufficient_time_for_approach_after_rebase():
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    from nero_core.two_second.synchronized_reference import schedule_forward_join

    worker, clock, item, vision, planner, _ = waiting_with_path()
    clock.now = 15.  # -.8 m + .2 m/s * 5 s = +.2 m, still inside the region.
    for _ in range(3):
        clock.now += .06
        worker.update_wait_state(sample(clock))
    assert worker.wait_feedback_error is None

    def plan(target, state, *, cancelled):
        assert target is item and not cancelled()
        try:
            schedule_forward_join(capture_s=target.capture_s, grasp_position=target.entry_position,
                velocity=target.velocity, standby_position=[-.4, 0., .7],
                direction=worker.config.direction, now_s=clock(), planning_reserve_s=.2,
                advance_s=.09, join_s=1.1, synchronized_s=0., follow_s=0.,
                follow_bounds=worker.config.follow_bounds, earliest_dispatch_s=planner.intercept_not_before_s)
        except ValueError as error:
            raise PickRejected(str(error)) from error
        pytest.fail('A 1.1 s approach cannot reach this part before the downstream limit')
    planner.intercept = plan
    worker._step()
    assert vision.discarded == [item.identity] and worker.poll_ready() is None

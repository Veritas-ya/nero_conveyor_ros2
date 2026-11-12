"""Offline timing ablations and unchanged legacy behaviour; no device access."""
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nero_core.two_second.config import CellConfig
from nero_core.two_second.timing_experiment import plan_experiment, validate_experiment, plan_record
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_two_second_hardware import Clock, cloud, executor_fixture, site
from test_two_second_smoothing import fixture as planner_fixture


def experiment(group):
    return {'group': group, 'expiry_policy': 'reject' if group == 'A' else 'replan',
            'max_planning_rounds': 4, 'max_delivery_attempts': 4}


def raw(group):
    return {'trajectory_profile': 'smooth', 'intercept_strategy': 'synchronized',
        'planning_reserve_mode': 'measured' if group == 'C' else 'fixed',
        'planning_reserve_s': .08, 'planning_budget_s': 1., 'freeze_position_upstream_m': .5,
        'timing_experiment': experiment(group)}


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_experiment_configuration_is_explicit_and_validated(site, group):
    path, data = site
    baseline = CellConfig.load(path)
    assert 'timing_experiment' not in baseline.raw
    data.update(raw(group), prealign_orientation_s=1.2)
    path.write_text(yaml.safe_dump(data))
    assert CellConfig.load(path).raw['timing_experiment'] == experiment(group)


@pytest.mark.parametrize('value', [None, False, {}, {'group': 'A'},
    {'group': 'A', 'expiry_policy': 'replan'},
    {'group': 'C', 'expiry_policy': 'replan', 'max_planning_rounds': True},
    {'group': 'C', 'expiry_policy': 'replan', 'max_delivery_attempts': 5},
    {'group': 'C', 'expiry_policy': 'replan', 'unknown': 1}])
def test_invalid_experiment_does_not_silently_select_a_policy(value):
    data = raw('C'); data['timing_experiment'] = value
    with pytest.raises(ValueError):
        validate_experiment(data)


def test_experiment_requires_common_explicit_freeze_position():
    data = raw('C'); data['freeze_position_upstream_m'] = None
    with pytest.raises(ValueError, match='冻结'):
        validate_experiment(data)


@dataclass(frozen=True)
class Path:
    planning_s: float = 0.
    diagnostics: dict = None


@dataclass(frozen=True)
class Plan:
    path: Path
    timings_s: dict


@pytest.mark.parametrize('group,expected_calls', [('A', 1), ('B', 2), ('C', 2)])
def test_reservation_estimation_is_independent_of_expiry_recovery(group, expected_calls):
    clock = Clock()
    planner = SimpleNamespace(clock=clock, config=SimpleNamespace(raw=raw(group)),
        measured_planning_lead_s=.01, measured_post_planning_s=.02)
    leads, deadlines = [], []
    def planning(*a, **kw):
        leads.append(kw['reservation_s']); deadlines.append(kw['global_deadline'])
        clock.sleep(.1)
        if len(leads) == 1:
            raise PickRejected('late', details={'failures': [{'details': {
                'reason': 'measured_deadline_missed', 'measured_candidate_wall_s': .1}}]})
        return Plan(Path(diagnostics={}), {})
    if group == 'A':
        with pytest.raises(PickRejected):
            plan_experiment(planner, None, None, cancelled=lambda: False, planning_fn=planning)
    else:
        result = plan_experiment(planner, None, None, cancelled=lambda: False, planning_fn=planning)
        assert result.path.planning_s == pytest.approx(.2)
        assert result.path.diagnostics['timing_experiment']['rounds'][0]['result'] == 'missed'
    assert len(leads) == expected_calls
    assert len(set(deadlines)) == 1
    if group in ('A', 'B'):
        assert leads == [.08]*expected_calls
        assert planner.measured_planning_lead_s == .01
    else:
        assert leads[0] == pytest.approx(.03)
        assert leads[1] > .12


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_non_timing_rejection_is_not_retried(group):
    clock = Clock(); calls = []
    planner = SimpleNamespace(clock=clock, config=SimpleNamespace(raw=raw(group)))
    def planning(*a, **kw):
        calls.append(1)
        raise PickRejected('IK failure', details={'failures': [{'reason': 'IK'}]})
    with pytest.raises(PickRejected, match='IK'):
        plan_experiment(planner, None, None, cancelled=lambda: False, planning_fn=planning)
    assert len(calls) == 1


def test_shared_outer_deadline_and_cancellation_are_not_reset_by_retry():
    clock = Clock(); planner = SimpleNamespace(clock=clock, config=SimpleNamespace(raw=raw('C')))
    def slow(*a, **kw):
        assert kw['global_deadline'] == pytest.approx(1.05)
        clock.sleep(.06)
        return Plan(Path(diagnostics={}), {})
    with pytest.raises(PickRejected, match='总预算'):
        plan_experiment(planner, None, None, cancelled=lambda: False, planning_fn=slow, deadline=1.05)
    def forbidden(*a, **kw):
        pytest.fail('cancelled work cannot begin planning')
    with pytest.raises(PickRejected):
        plan_experiment(planner, None, None, cancelled=lambda: True, planning_fn=forbidden)


class ImmediatePool:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try: future.set_result(fn(*args, **kwargs))
        except BaseException as error: future.set_exception(error)
        return future


def delivery_fixture(monkeypatch, group, *, budget=1., prepare_s=.06, first_lead=.15):
    from nero_core.two_second.planning import InterceptPlan
    from nero_core.two_second.synchronized_reference import SynchronizedIntercept
    from nero_core.two_second.trajectory import TimedPath
    from nero_core.two_second.vision import MeasuredTarget, obb_grasps
    e, device, clock = executor_fixture()
    e.config.raw.update(raw(group), command_advance_s=.1, planning_budget_s=budget,
                        closure_policy='scheduled_attempt')
    e.config.direction = np.array([1., 0., 0.]); e.config.follow_bounds = np.array([-.2, 1.])
    points = cloud(); grasps, obb = obb_grasps(points, np.eye(4), np.eye(4))
    target = MeasuredTarget((0, 1, 2), clock(), clock(), 100., 0, 1, np.zeros(3),
        np.array([.1, 0., 0.]), tuple(grasps), points,
        {'belt_region': {'plane_to_base': np.eye(4)}, 'obb': obb}, {}, {'source': 'nero7_xy_kalman'})
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor', ImmediatePool)
    calls = []
    def intercept(target, state, **kwargs):
        calls.append(kwargs['experiment_deadline'])
        ref = SynchronizedIntercept(target.capture_s, clock()+(first_lead if len(calls) == 1 else .3),
            1.2, np.zeros(3), target.velocity, np.zeros(3), [1, 0, 0, 0], [1, 0, 0, 0], .3, .5, 1.2)
        timing = {**experiment(group), 'rounds': [], 'completed_monotonic_s': clock()}
        path = TimedPath(lambda t, d=0: np.zeros(7), target.identity, 1,
                        'smooth_intercept_follow_brake', 2., 0., {'timing_experiment': timing})
        return InterceptPlan(path, ref, target, 1, target.grasps[0], ())
    e.planner.intercept = intercept
    device.prepare_motion = lambda *a, **kw: clock.sleep(prepare_s)
    streams = []
    def stream(*a, **kw):
        streams.append(1); device.width = 0.; device.force = 0.
        return device.snapshot()
    monkeypatch.setattr(e, '_stream', stream)
    return e, device, clock, target, calls, streams


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_delivery_recovery_independent_of_reservation_mode(monkeypatch, group):
    e, device, clock, target, calls, streams = delivery_fixture(monkeypatch, group)
    if group == 'A':
        with pytest.raises(PickRejected, match='准备后'):
            e._run_attempt(target)
        assert not streams and len(calls) == 1
    else:
        assert e._run_attempt(target)['status'] == 'attempt_completed_grip_unconfirmed'
        assert len(calls) == 2 and len(streams) == 1 and len(set(calls)) == 1
    assert any(r['event'] == 'experiment_delivery_missed' for r in e.events)
    assert any(r['event'] == 'experiment_plan_ready' for r in e.events)
    if group == 'C':
        assert e.planner.measured_post_planning_s == pytest.approx(.06)
    else:
        assert not hasattr(e.planner, 'measured_post_planning_s')
    assert not device.commands


def test_budget_exhaustion_after_preparation_never_streams(monkeypatch):
    e, device, clock, target, calls, streams = delivery_fixture(monkeypatch, 'B', budget=.05)
    with pytest.raises(PickRejected, match='预算耗尽'):
        e._run_attempt(target)
    assert len(calls) == 1 and not streams and not device.commands


def test_total_budget_covers_preparation_even_when_send_deadline_is_still_future(monkeypatch):
    e, device, clock, target, calls, streams = delivery_fixture(
        monkeypatch, 'B', budget=.05, prepare_s=.06, first_lead=.4)
    with pytest.raises(PickRejected, match='总规划预算'):
        e._run_attempt(target)
    assert not streams and not device.commands
    assert any(r['event'] == 'experiment_budget_exhausted' for r in e.events)


def test_experimental_diagnostics_cannot_turn_valid_motion_into_a_failure(monkeypatch):
    e, device, clock, target, calls, streams = delivery_fixture(monkeypatch, 'C', prepare_s=0.)
    def broken(*a, **kw): raise RuntimeError('diagnostic-only failure')
    monkeypatch.setattr('nero_core.two_second.timing_experiment.plan_record', broken)
    assert e._run_attempt(target)['status'] == 'attempt_completed_grip_unconfirmed'
    assert len(streams) == 1
    assert any(r['event'] == 'experiment_record_error' for r in e.events)


@pytest.mark.parametrize('group', ['A', 'B', 'C'])
def test_production_smooth_planner_rechecks_new_target_time_and_geometry(monkeypatch, group):
    planner, model, target = planner_fixture(monkeypatch)
    clock = Clock(); planner.clock = clock
    planner.config.raw.update(raw(group), planning_reserve_s=.001,
        synchronized_duration_s=0., prealign_orientation_s=1.2,
        rotate_first_s=0., brake_duration_s=.5, candidate_planning_budget_s=.5,
        planning_knot_hz=10., planning_knot_hz_fallback=10.)
    planner.config.follow_bounds = np.array([-.2, 2.])
    planner.config.workspace = np.array([[-3., 3.]]*3)
    from dataclasses import replace
    planner.config.settings = replace(planner.config.settings, full_plan_close_window_s=.2)
    # The immutable exposure is within the same fixed freeze band for all groups.
    target.center = np.array([-.7, 0., 0.]); target.velocity = np.array([1., 0., 0.])
    poses = [np.eye(4), np.eye(4)]
    for pose in poses: pose[0, 3] = -.7
    target.grasps = tuple(poses)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw):
            clock.sleep(.002)
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    def install(*a):
        clock.sleep(.01); planner.version += 1; planner.scene_revision += 1
    monkeypatch.setattr(planner, 'install', install)
    references = []
    def geometry(path, ref, **kw):
        references.append(ref); clock.sleep(.03)
        return {'offline_only': True}
    monkeypatch.setattr(planner, 'check_smooth_intercept', geometry)
    if group == 'C':
        result = planner.intercept(target, {'joints': np.zeros(7)}, experiment_deadline=2.)
        assert len(references) >= 2
        assert result.reference.contact_s > references[0].contact_s
        assert result.reference.start_s-.1 > clock()
        assert result.target is target
        record = plan_record(planner.config, result, clock(), 1)
        assert record['upstream_margin_m'] >= 0 and record['downstream_margin_m'] >= 0
    else:
        with pytest.raises(PickRejected):
            planner.intercept(target, {'joints': np.zeros(7)}, experiment_deadline=2.)
        assert len(references) == (1 if group == 'A' else 4)

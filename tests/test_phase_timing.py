"""Offline analytic, planner and execution-boundary checks for phase allocation."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, PickSettings
from nero_core.two_second.phase_timing import (KEY, assess_reference, candidate_options,
    make_reference, polynomial_range, require_reference, validate_configuration, verify_plan)
from test_two_second_hardware import site
from test_two_second_smoothing import fixture as planner_fixture


def specification(mode='robust'):
    return {'mode': mode, 'join_candidates_s': [.6, .9, 1.2],
        'follow_candidates_s': [.3, .4, .5], 'brake_candidates_s': [.4, .6, .9],
        'gripper_close_time_s': .08, 'gripper_time_margin_s': .02,
        'residual_time_bound_s': .01, 'position_margin_m': .002,
        'stop_interval_base_m': [-.2, .8], 'max_tcp_acceleration_m_s2': 4.,
        'calibration_status': 'assumed_offline', 'calibration_note': 'synthetic test only'}


def fixture(mode='robust'):
    cfg = SimpleNamespace(raw={KEY: specification(mode), 'trajectory_profile': 'smooth',
        'intercept_strategy': 'synchronized', 'closure_policy': 'scheduled_attempt',
        'command_advance_s': .09, 'synchronized_duration_s': 0., 'rotate_first_s': 0.},
        workspace=np.array([[-1., 1.]]*3), follow_bounds=np.array([-.2, .6]),
        direction=np.array([1., 0., 0.]), settings=PickSettings(control_hz=100),
        velocity_limits=np.ones(7), acceleration_limits=np.ones(7))
    grasp = np.eye(4); grasp[0, 3] = -.5
    start = np.eye(4); start[2, 3] = .1
    target = SimpleNamespace(capture_s=0., velocity=np.array([.4, 0., 0.]), grasps=(grasp,))
    return cfg, target, start, grasp


def test_braking_distance_and_peak_acceleration_match_dense_curve():
    cfg, target, start, grasp = fixture()
    for brake in [.2, .6, 1.]:
        ref = make_reference(cfg, target, start, grasp, .9, .4, brake, now_s=.1, reservation_s=.2)
        report = assess_reference(cfg, ref)
        samples = [ref.position_derivatives(t) for t in np.linspace(ref.follow_end_s, ref.full_duration_s, 1001)]
        assert np.linalg.norm(samples[-1][0]-samples[0][0]) == pytest.approx(.4*brake/2)
        assert max(np.linalg.norm(s[2]) for s in samples) == pytest.approx(report['brake_peak_acceleration_m_s2'])
        np.testing.assert_allclose(samples[-1][1:], 0., atol=1e-12)


def test_exact_join_extrema_cover_dense_samples_and_acceleration():
    cfg, target, start, grasp = fixture()
    start[1, 3] = .3
    ref = make_reference(cfg, target, start, grasp, .9, .4, .6, now_s=.1, reservation_s=.2)
    for derivative in (0, 2):
        bounds = polynomial_range(ref._join, derivative)
        samples = ref._join(np.linspace(0, .9, 1001), derivative)
        assert np.all(samples >= bounds[:, 0]-1e-10)
        assert np.all(samples <= bounds[:, 1]+1e-10)


def test_preclosure_synchronization_and_advance_do_not_count_as_closing_time():
    cfg, target, start, grasp = fixture('nominal')
    cfg.raw['synchronized_duration_s'] = .5
    cfg.follow_bounds[1] = .9
    ref = make_reference(cfg, target, start, grasp, .6, .1, .4, now_s=0., reservation_s=0.)
    report = assess_reference(cfg, ref)
    assert 'closure_time_s' in report['reasons']
    assert report['required_closure_window_s'] == pytest.approx(.19)
    cfg.raw['command_advance_s'] = .2
    assert assess_reference(cfg, ref)['required_closure_window_s'] == pytest.approx(.19)


def test_residual_margin_applied_once_and_long_braking_is_not_always_feasible():
    cfg, target, start, grasp = fixture()
    ref = make_reference(cfg, target, start, grasp, .9, .4, .6, now_s=.1, reservation_s=.2)
    robust = assess_reference(cfg, ref)
    cfg.raw[KEY]['mode'] = 'nominal'
    nominal = assess_reference(cfg, ref)
    assert nominal['margins']['follow_interval_m']-robust['margins']['follow_interval_m'] == pytest.approx(.006)
    assert robust['required_closure_window_s']-nominal['required_closure_window_s'] == pytest.approx(.03)
    cfg.raw[KEY]['stop_interval_base_m'][1] = float(ref.position_derivatives(ref.follow_end_s)[0][0])+.1
    assert 'stop_interval_m' in assess_reference(cfg, ref)['reasons']


def test_selector_adapts_to_speed_and_late_arrival_without_mutating_inputs():
    cfg, target, start, grasp = fixture()
    original = deepcopy(cfg.raw)
    choices = []
    for speed, age in [(.2, .1), (.4, .1), (.5, .5)]:
        target.velocity[0] = speed
        options, audit = candidate_options(cfg, target, start, [grasp], now_s=age, reservation_s=.1)
        assert options and audit['evaluated_pairs'] == 27
        assert options[0]['allowed'] and not audit['global_optimum_claimed']
        choices.append(tuple(options[0][k] for k in ('join_s', 'follow_s', 'brake_s')))
    assert len(set(choices)) > 1
    assert cfg.raw == original


def test_empty_window_and_cancellation_return_no_plan():
    cfg, target, start, grasp = fixture()
    options, audit = candidate_options(cfg, target, start, [grasp], now_s=5., reservation_s=.1)
    assert not options and audit['rejection_counts']['schedule'] == 27
    with pytest.raises(PickRejected, match='预算'):
        candidate_options(cfg, target, start, [grasp], now_s=.1, reservation_s=.1, cancelled=lambda: True)


def test_search_tries_distinct_join_families_before_repeating_expensive_failures():
    cfg, target, start, grasp = fixture()
    options, _ = candidate_options(cfg, target, start, [grasp, grasp.copy()], now_s=.1, reservation_s=.1)
    families = {(o['join_s'], o['grasp_index']) for o in options}
    assert len({(o['join_s'], o['grasp_index']) for o in options[:len(families)]}) == len(families)


def test_more_grip_uncertainty_can_invalidate_a_nominal_candidate():
    cfg, target, start, grasp = fixture('nominal')
    ref = make_reference(cfg, target, start, grasp, .9, .2, .6, now_s=.1, reservation_s=.1)
    assert assess_reference(cfg, ref)['allowed']
    cfg.raw[KEY]['mode'] = 'robust'
    assert 'closure_time_s' in assess_reference(cfg, ref)['reasons']


def test_later_scheduling_is_rechecked_instead_of_reusing_old_margins():
    cfg, target, start, grasp = fixture()
    ref = make_reference(cfg, target, start, grasp, .9, .4, .6, now_s=.1, reservation_s=.1)
    require_reference(cfg, ref)
    # The same durations become invalid after enough compute delay.
    later = replace(ref, start_s=ref.start_s+1.0)
    with pytest.raises(PickRejected): require_reference(cfg, later)


@pytest.mark.parametrize('mutation', ['negative', 'nan', 'boolean', 'huge_grid', 'unknown', 'policy', 'stop', 'delay'])
def test_invalid_configuration_fails_closed(mutation):
    cfg, _, _, _ = fixture()
    spec = cfg.raw[KEY]
    if mutation == 'negative': spec['join_candidates_s'] = [-1.]
    elif mutation == 'nan': spec['gripper_time_margin_s'] = float('nan')
    elif mutation == 'boolean': spec['gripper_close_time_s'] = True
    elif mutation == 'huge_grid': spec['join_candidates_s'] = list(range(1, 50))
    elif mutation == 'unknown': spec['typo'] = 1
    elif mutation == 'policy': cfg.raw['closure_policy'] = 'feedback'
    elif mutation == 'stop': spec['stop_interval_base_m'] = [-2, 2]
    else: cfg.raw['execution_delay_window'] = {}
    with pytest.raises(ValueError): validate_configuration(cfg.raw, cfg.workspace, cfg.direction)


@pytest.mark.parametrize('mutation', ['start', 'follow', 'workspace', 'limit', 'spec', 'missing'])
def test_stale_certification_cannot_reach_execution(mutation):
    cfg, target, start, grasp = fixture()
    ref = make_reference(cfg, target, start, grasp, .9, .4, .6, now_s=.1, reservation_s=.2)
    path = SimpleNamespace(diagnostics={KEY: require_reference(cfg, ref)})
    verify_plan(cfg, path, ref)
    if mutation == 'start': ref = replace(ref, start_s=ref.start_s+.01)
    elif mutation == 'follow': ref = replace(ref, follow_s=.5)
    elif mutation == 'workspace': cfg.workspace[0, 1] -= .01
    elif mutation == 'limit': cfg.acceleration_limits[0] += .1
    elif mutation == 'spec': cfg.raw[KEY]['gripper_time_margin_s'] += .001
    else: path.diagnostics.clear()
    with pytest.raises(PickRejected): verify_plan(cfg, path, ref)


def test_assumed_parameters_are_planning_only_and_block_sender():
    from nero_core.two_second.streaming import stream
    cfg, target, start, grasp = fixture()
    ref = make_reference(cfg, target, start, grasp, .9, .4, .6, now_s=.1, reservation_s=.2)
    path = SimpleNamespace(identity=(0, 1, 2), version=1, diagnostics={KEY: require_reference(cfg, ref)})
    executor = SimpleNamespace(config=cfg, owner=path.identity, planner=SimpleNamespace(version=1), version=1)
    with pytest.raises(PickRejected, match='离线假设'):
        stream(executor, path, origin=ref.start_s, epoch=0,
               sender_factory=lambda *a, **kw: pytest.fail('offline parameters reached sender'), reference=ref)


def test_configuration_is_opt_in_and_does_not_rewrite_fixed_timings(site):
    from nero_core.two_second.config import CellConfig
    path, raw = site
    assert KEY not in CellConfig.load(path).raw
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
               closure_policy='scheduled_attempt', prealign_orientation_s=1.1, brake_duration_s=.9)
    raw[KEY] = specification()
    path.write_text(yaml.safe_dump(raw))
    cfg = CellConfig.load(path)
    assert cfg.raw['prealign_orientation_s'] == 1.1 and cfg.raw['brake_duration_s'] == .9
    np.testing.assert_array_equal(cfg.standby, raw['standby_joint_rad'])
    np.testing.assert_array_equal(cfg.follow_bounds, raw['follow_interval_base_m'])


def test_real_planner_uses_selected_times_and_retains_full_geometry_check(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    cfg, _, _, _ = fixture()
    planner.config.raw.update(cfg.raw, planning_reserve_s=.5, prealign_orientation_s=1.1,
        brake_duration_s=.9, synchronized_join_candidates_s=[1.1], candidate_planning_budget_s=3.,
        planning_knot_hz=10., planning_knot_hz_fallback=10.)
    planner.config.follow_bounds = cfg.follow_bounds
    planner.config.settings = replace(planner.config.settings, full_plan_close_window_s=0.)
    target.velocity = np.array([.4, 0., 0.]); target.center[0] = -.5
    for grasp in target.grasps: grasp[0, 3] = -.5
    calls = []
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs):
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *args: None)
    def geometry(path, ref, **kwargs):
        calls.append(ref)
        if len(calls) == 1: raise PickRejected('injected collision')
        return {'synthetic_geometry_only': True}
    monkeypatch.setattr(planner, 'check_smooth_intercept', geometry)
    original = deepcopy(planner.config.raw)
    plan = planner.intercept(target, {'joints': np.zeros(7)})
    assert len(calls) >= 2 and plan.reference.follow_s > 0.
    assert plan.reference.follow_s in original[KEY]['follow_candidates_s']
    assert plan.path.diagnostics[KEY]['allowed']
    verify_plan(planner.config, plan.path, plan.reference)
    assert planner.config.raw == original
    assert plan.path.diagnostics['reference_check']['synthetic_geometry_only']


def test_offline_config_copies_preserve_site_and_reject_tampering(site, tmp_path, monkeypatch):
    import builtins
    from nero_core.two_second.phase_timing_experiment import prepare, load_groups
    path, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        closure_policy='scheduled_attempt', prealign_orientation_s=1.1, brake_duration_s=.9)
    path.write_text(yaml.safe_dump(raw))
    before = path.read_bytes()
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if 'pyAgxArm' in name or name.endswith('nero_pick_hardware') or name.endswith('backends.orbbec'):
            pytest.fail('offline tool imported device backend')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    directory = tmp_path/'copies'
    prepare(path, specification(), directory, fixed_times=[1.1, .4, .9])
    groups, manifest = load_groups(directory)
    assert path.read_bytes() == before and not manifest['physical_parameters_validated']
    for cfg in groups.values():
        np.testing.assert_array_equal(cfg.standby, raw['standby_joint_rad'])
        np.testing.assert_array_equal(cfg.follow_bounds, raw['follow_interval_base_m'])
        np.testing.assert_array_equal(cfg.workspace, raw['workspace_base_m'])
    with pytest.raises(FileExistsError): prepare(path, specification(), directory)
    p = directory/'robust.yaml'; p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='changed'): load_groups(directory)


def test_assumed_feature_does_not_change_legacy_reference_when_absent(monkeypatch):
    from nero_core.two_second.smooth_planning import plan_smooth_intercept
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        closure_policy='scheduled_attempt', planning_reserve_s=.5, prealign_orientation_s=1.1,
        synchronized_duration_s=0., brake_duration_s=.9, synchronized_join_candidates_s=[1.1],
        candidate_planning_budget_s=3., planning_knot_hz=10., planning_knot_hz_fallback=10.)
    planner.config.settings = replace(planner.config.settings, full_plan_close_window_s=0.)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs): return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *args: None)
    monkeypatch.setattr(planner, 'check_smooth_intercept', lambda *a, **kw: {})
    result = plan_smooth_intercept(planner, target, {'joints': np.zeros(7)})
    assert result.reference.follow_s == 0. and result.reference.brake_s == .9
    assert result.reference.prealign_s == 1.1 and KEY not in result.path.diagnostics

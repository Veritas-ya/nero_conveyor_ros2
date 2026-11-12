"""Conditional delay-window math and opt-in integration; no physical-success claims."""
import builtins
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nero_core.two_second.execution_delay_window import (KEY, PROFILE_SCHEMA, DelayWindowError,
    assess_reference, envelope, planning_bounds, require_reference, sha256, validate_configuration, verify_plan)
from nero_core.two_second.execution_delay_experiment import audit_recorded, build_profile, config_copy
from nero_core.two_second.synchronized_reference import SynchronizedIntercept, schedule_forward_join
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_two_second_hardware import site
from test_two_second_smoothing import fixture as planner_fixture


def profile_value():
    return {'schema': PROFILE_SCHEMA, 'measurement': 'equivalent_host_send_to_joint_feedback_lag',
        'coverage': 'observed_range_plus_declared_margin_not_hardware_bound',
        'phase': 'near_contact', 'condition_id': 'synthetic_test_only',
        'samples': 14, 'delay_bounds_s': [.085, .095]}


def configuration(tmp_path, *, mode='experimental', bounds=(-.2, .38), advance=.09):
    path = tmp_path/'profile.json'; path.write_text(json.dumps(profile_value()))
    cfg = SimpleNamespace(raw={'trajectory_profile': 'smooth', 'intercept_strategy': 'synchronized',
        'command_advance_s': advance, KEY: {'mode': mode, 'profile_file': str(path),
            'profile_sha256': sha256(path), 'condition_id': 'synthetic_test_only',
            'maximum_relative_error_m': .005}}, direction=np.array([1., 0., 0.]),
        follow_bounds=np.array(bounds), workspace=np.array([[-2., 2.]]*3))
    validate_configuration(cfg.raw, tmp_path/'config.yaml')
    return cfg


def reference(contact=.1, follow=.1):
    return SynchronizedIntercept(capture_s=0., start_s=1., duration_s=1., prealign_s=1.,
        capture_tcp_world=np.array([contact-.8, 0., 0.]), velocity_world=np.array([.4, 0., 0.]),
        start_tcp_world=np.array([-1., 0., .2]), start_quaternion_wxyz=[1., 0., 0., 0.],
        target_quaternion_wxyz=[1., 0., 0., 0.], follow_s=follow, brake_s=.5)


def test_advance_is_subtracted_once_and_only_residual_erodes_the_window(tmp_path):
    cfg = configuration(tmp_path)
    report = envelope(cfg, [.4, 0., 0.])
    assert report['residual_delay_bounds_s'] == pytest.approx([-.005, .005])
    assert report['effective_follow_bounds_m'] == pytest.approx([-.198, .378])
    assert report['relative_position_error_bound_m'] == pytest.approx(.002)
    assert report['allowed'] and not report['physical_delay_guaranteed']
    assert not report['gripper_closure_latency_modelled']


@pytest.mark.parametrize('delay_bounds', [[.04, .06], [.085, .095], [.12, .14]])
def test_interval_extrema_cover_every_intermediate_delay_in_the_declared_model(tmp_path, delay_bounds):
    cfg = configuration(tmp_path)
    cfg.raw['_execution_delay_profile']['delay_bounds_s'] = delay_bounds
    cfg.raw[KEY]['maximum_relative_error_m'] = 1.
    low, high = planning_bounds(cfg, [.4, 0., 0.])
    for coordinate in np.linspace(low, high, 21):
        for lag in np.linspace(*delay_bounds, 51):
            actual_target = coordinate+.4*(lag-.09)
            assert cfg.follow_bounds[0]-1e-12 <= actual_target <= cfg.follow_bounds[1]+1e-12


def test_nominal_boundary_pass_can_fail_residual_window_and_zero_follow_is_supported(tmp_path):
    cfg = configuration(tmp_path)
    ref = reference(contact=.379, follow=0.)
    assert ref.contact_xyz[0] < cfg.follow_bounds[1]
    with pytest.raises(PickRejected, match='残余执行时延'):
        require_reference(cfg, ref)
    report = assess_reference(cfg, ref)
    assert report['envelope_maximum_coordinate_m'] == pytest.approx(.381)
    assert report['downstream_margin_m'] < 0


def test_planner_selects_a_new_join_with_upstream_residual_room(tmp_path):
    cfg = configuration(tmp_path)
    args = dict(capture_s=0., grasp_position=[-.6, 0., 0.], velocity=[.4, 0., 0.],
        standby_position=[-1., 0., 0.], direction=[1., 0., 0.], now_s=0.,
        planning_reserve_s=0., advance_s=.09, join_s=.2, synchronized_s=0., follow_s=.1)
    original = schedule_forward_join(**args, follow_bounds=cfg.follow_bounds)
    robust = schedule_forward_join(**args, follow_bounds=planning_bounds(cfg, [.4, 0., 0.]))
    assert original['sync_start_coordinate_m'] == pytest.approx(-.2)
    assert robust['sync_start_coordinate_m'] == pytest.approx(-.198)
    assert robust['contact_s']-original['contact_s'] == pytest.approx(.005)


def test_shadow_reports_rejection_but_keeps_nominal_bounds_and_reference(tmp_path):
    cfg = configuration(tmp_path, mode='shadow')
    ref = reference(contact=.379, follow=0.)
    np.testing.assert_array_equal(planning_bounds(cfg, ref.velocity_world), cfg.follow_bounds)
    report = require_reference(cfg, ref)
    assert not report['allowed']
    assert verify_plan(cfg, SimpleNamespace(diagnostics={KEY: report}), ref) == report


def test_error_bound_and_empty_window_fail_before_any_geometry_or_motion(tmp_path):
    cfg = configuration(tmp_path)
    with pytest.raises(DelayWindowError) as error:
        planning_bounds(cfg, [2., 0., 0.])
    assert 'relative_position_error_exceeds_declared_tolerance' in error.value.details[KEY]['reasons']
    cfg.follow_bounds = np.array([0., .001])
    with pytest.raises(DelayWindowError) as error:
        planning_bounds(cfg, [.4, 0., 0.])
    assert 'empty_finite_grasp_interval' in error.value.details[KEY]['reasons']


@pytest.mark.parametrize('change', ['advance', 'reference', 'profile', 'workspace', 'missing'])
def test_stale_record_cannot_authorize_a_different_reference_or_configuration(tmp_path, change):
    cfg = configuration(tmp_path); ref = reference()
    path = SimpleNamespace(diagnostics={KEY: require_reference(cfg, ref)})
    verify_plan(cfg, path, ref)
    if change == 'advance': cfg.raw['command_advance_s'] = .091
    elif change == 'reference': ref = replace(ref, start_s=1.01)
    elif change == 'profile': cfg.raw['_execution_delay_profile']['delay_bounds_s'] = [.086, .094]
    elif change == 'workspace': cfg.workspace[0, 0] = -1.9
    else: path.diagnostics = {}
    with pytest.raises(PickRejected): verify_plan(cfg, path, ref)


def test_profile_pin_schema_and_conditions_are_explicit_and_original_config_is_untouched(site, tmp_path):
    from nero_core.two_second.config import CellConfig
    base, raw = site
    assert KEY not in CellConfig.load(base).raw
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.1,
        command_advance_s=.09)
    base.write_text(yaml.safe_dump(raw)); original = base.read_bytes()
    cfg = configuration(tmp_path)
    output = tmp_path/'copies/shadow.yaml'
    config_copy(base, cfg.raw[KEY]['profile_file'], output, mode='shadow', maximum_relative_error_m=.005)
    loaded = CellConfig.load(output)
    assert loaded.raw['command_advance_s'] == .09 and base.read_bytes() == original
    with pytest.raises(FileExistsError):
        config_copy(base, cfg.raw[KEY]['profile_file'], output, mode='shadow', maximum_relative_error_m=.005)
    p = tmp_path/'profile.json'; p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError, match='hash'):
        CellConfig.load(output)


@pytest.mark.parametrize('mutation', ['condition', 'mode', 'unknown_field', 'error_bool', 'schema', 'range'])
def test_bad_profile_or_configuration_cannot_silently_enable_a_window(tmp_path, mutation):
    cfg = configuration(tmp_path)
    if mutation == 'condition': cfg.raw[KEY]['condition_id'] = 'different_host_or_condition'
    elif mutation == 'mode': cfg.raw[KEY]['mode'] = 'automatic_hardware_guarantee'
    elif mutation == 'unknown_field': cfg.raw[KEY]['extra'] = 1
    elif mutation == 'error_bool': cfg.raw[KEY]['maximum_relative_error_m'] = True
    else:
        path = tmp_path/'profile.json'; data = profile_value()
        if mutation == 'schema': data['coverage'] = 'guaranteed_99_percent'
        else: data['delay_bounds_s'] = [.1, .05]
        path.write_text(json.dumps(data)); cfg.raw[KEY]['profile_sha256'] = sha256(path)
    with pytest.raises(ValueError): validate_configuration(cfg.raw, tmp_path/'config.yaml')


def test_recorded_audit_matches_submission_identity_and_retains_unplanned_attempts(site, tmp_path):
    base, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.1,
        command_advance_s=.09)
    base.write_text(yaml.safe_dump(raw))
    cfg = configuration(tmp_path)
    copied = tmp_path/'shadow.yaml'
    config_copy(base, cfg.raw[KEY]['profile_file'], copied, mode='shadow', maximum_relative_error_m=.005)
    run = tmp_path/'run'; run.mkdir(); (run/'summary.json').write_text('{}')
    cycle = run/'cycle_001'; cycle.mkdir()
    start, grasp = np.eye(4), np.eye(4); grasp[0, 3] = -.7
    delivery = {'identity': [0, 1, 2], 'capture_monotonic_s': 0.,
        'selected_grasp_tcp_base': grasp.tolist(), 'velocity_base_m_s': [.4, 0., 0.],
        'trajectory_diagnostics': {'synchronized_schedule': {'join_start_s': 1.},
            'prealign_s': 1., 'synchronized_s': 0., 'follow_s': .1, 'brake_s': .5, 'rotate_first_s': 0.}}
    events = [{'event': 'planning_submitted', 'data': {'identity': [0, 1, 2],
        'capture_monotonic_s': 0., 'start_tcp_base': start.tolist()}},
        {'event': 'planning_delivery', 'data': delivery}]
    (cycle/'events.json').write_text(json.dumps(events))
    other = run/'cycle_002'; other.mkdir(); (other/'events.json').write_text('[]')
    result = audit_recorded(copied, run, tmp_path/'audit.json')
    assert [r['status'] for r in result['rows']] == ['conditional_window_pass', 'no_planning_delivery']
    assert not result['grasp_success_inferred']
    events[0]['data']['identity'] = [0, 1, 3]
    (cycle/'events.json').write_text(json.dumps(events))
    result = audit_recorded(copied, run, tmp_path/'mismatched.json')
    assert result['rows'][0]['status'] == 'input_error'


def test_production_smooth_planner_attaches_checked_record_and_refuses_bad_delay_before_ik(monkeypatch, tmp_path):
    planner, model, target = planner_fixture(monkeypatch)
    delay_cfg = configuration(tmp_path)
    planner.config.raw.update(delay_cfg.raw, planning_reserve_mode='fixed', planning_reserve_s=.5,
        synchronized_duration_s=0., prealign_orientation_s=1.1, rotate_first_s=0., brake_duration_s=.5,
        synchronized_join_candidates_s=[1.1], candidate_planning_budget_s=3.,
        planning_knot_hz=10., planning_knot_hz_fallback=10.)
    planner.config.follow_bounds = delay_cfg.follow_bounds
    planner.config.settings = replace(planner.config.settings, full_plan_close_window_s=.1)
    target.velocity = np.array([.4, 0., 0.]); target.center = np.array([-.7, 0., 0.])
    target.grasps = tuple(np.eye(4) for _ in range(2))
    for pose in target.grasps: pose[0, 3] = -.7
    calls = []
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs):
            calls.append(1); return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *args: None)
    monkeypatch.setattr(planner, 'check_smooth_intercept', lambda *args, **kwargs: {'synthetic_only': True})
    plan = planner.intercept(target, {'joints': np.zeros(7)})
    assert calls and plan.path.diagnostics[KEY]['allowed']
    verify_plan(planner.config, plan.path, plan.reference)
    calls.clear()
    planner.config.raw['_execution_delay_profile']['delay_bounds_s'] = [.2, .3]
    with pytest.raises(PickRejected) as error: planner.intercept(target, {'joints': np.zeros(7)})
    assert not calls
    assert error.value.details['failures'][0]['details']['reason'] == 'execution_delay_window_rejected'


def test_stream_refuses_missing_record_before_sender_or_device_use(tmp_path):
    from nero_core.two_second.streaming import stream
    cfg = configuration(tmp_path)
    e = SimpleNamespace(config=cfg, owner=(0, 1, 2), version=1, planner=SimpleNamespace(version=1))
    path = SimpleNamespace(identity=e.owner, version=1, diagnostics={})
    def forbidden(*args, **kwargs): pytest.fail('unbound plan reached a sender')
    with pytest.raises(PickRejected, match='记录缺失'):
        stream(e, path, origin=1., epoch=0, sender_factory=forbidden, reference=reference())
    with pytest.raises(PickRejected, match='时钟起点'):
        stream(e, path, origin=1.01, epoch=0, sender_factory=forbidden, reference=reference())


def test_profile_builder_uses_all_cycles_checks_source_hashes_and_never_imports_devices(tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if 'pyAgxArm' in name or name.endswith('nero_pick_hardware') or name.endswith('backends.orbbec'):
            pytest.fail('offline profile builder imported a device backend')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    run = tmp_path/'run'; run.mkdir(); (run/'summary.json').write_text('{}')
    rows = []
    for i, delay in enumerate([.085, .09, .095], 1):
        cycle = run/f'cycle_{i:03d}'; cycle.mkdir(); (cycle/'events.json').write_text('[]')
        rows.append({'cycle': cycle.name, 'fits': {'near_contact': {'delay_s': delay,
            'rms_error_rad': .003, 'constant_axis_bias_fitted': False, 'observations_per_axis': [20]*7}},
            'files_sha256': {'events.json': sha256(cycle/'events.json')}})
    source = tmp_path/'analysis.json'
    data = {'run': str(run), 'run_summary_sha256': sha256(run/'summary.json'),
        'synthetic_known_delay_tests_passed': True, 'rows': rows, 'attempts': 3,
        'command_advance_s': .05, 'config_sha256': 'synthetic'}
    source.write_text(json.dumps(data))
    profile = build_profile(source, tmp_path/'out.json', condition_id='fixture', margin_s=.005, maximum_fit_rms_rad=.01)
    assert profile['samples'] == 3 and profile['delay_bounds_s'] == pytest.approx([.08, .1])
    data['rows'][-1]['fits']['near_contact']['rms_error_rad'] = .02
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='silently discard'):
        build_profile(source, tmp_path/'bad.json', condition_id='fixture', margin_s=.005, maximum_fit_rms_rad=.01)
    assert not (tmp_path/'bad.json').exists()


@pytest.mark.parametrize('delay', [.05, .085, .12])
def test_identification_recovers_known_actual_send_to_feedback_lag(delay):
    from nero_core.two_second.execution_delay_identification import fit_delay
    t = np.arange(0., 2.001, .002)
    q = np.column_stack([np.sin((1.+j*.15)*t+j*.15) for j in range(7)])
    fq = np.column_stack([np.interp(t-delay, t, q[:, j]) for j in range(7)])
    result = fit_delay(t, q, np.tile(t[:, None], (1, 7)), fq, start_s=.3, end_s=1.7)
    assert result['delay_s'] == pytest.approx(delay, abs=1e-5)
    assert result['rms_error_rad'] < result['rms_error_without_delay_rad']*.001


def test_static_or_out_of_order_motion_cannot_produce_a_plausible_delay_number():
    from nero_core.two_second.execution_delay_identification import fit_delay
    t = np.arange(0., 2.001, .002); q = np.zeros((len(t), 7))
    with pytest.raises(ValueError, match='excitation'):
        fit_delay(t, q, np.tile(t[:, None], (1, 7)), q, start_s=.3, end_s=1.7)
    with pytest.raises(ValueError, match='unordered'):
        fit_delay(t[::-1], q, np.tile(t[:, None], (1, 7)), q, start_s=.3, end_s=1.7)

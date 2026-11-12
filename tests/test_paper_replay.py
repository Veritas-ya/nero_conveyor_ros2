"""Paired replay integrity, missing inputs, clock origin and device isolation."""
import builtins
import csv
from dataclasses import dataclass
import json
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nero_core.two_second import paper_replay as replay
from nero_core.two_second.paper_experiment import prepare_session
from test_two_second_hardware import site


def event_fixture(tmp_path):
    cycle = tmp_path/'run/cycle_001'; cycle.mkdir(parents=True)
    data = {'identity': [2, 3, 4], 'capture_monotonic_s': 100., 'submission_monotonic_s': 100.3}
    row = {'event': 'planning_submitted', 'monotonic_s': 100.31, 'data': data}
    (cycle/'events.json').write_text(json.dumps([row]))
    return cycle, row, {'identity': [2, 3, 4], 'capture_s': 100.}


def test_matching_requires_exact_exposure_identity_and_unique_event(tmp_path):
    cycle, row, metadata = event_fixture(tmp_path)
    path, data = replay.matched_event(cycle.parent, metadata)
    assert path == cycle/'events.json' and data['submission_monotonic_s'] == 100.3
    with pytest.raises(ValueError, match='exposure/submission'):
        replay.matched_event(cycle.parent, {**metadata, 'capture_s': 99.})
    with pytest.raises(ValueError, match='exactly one'):
        replay.matched_event(cycle.parent, {**metadata, 'identity': [1, 3, 4]})
    (cycle/'events.json').write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match='exactly one'):
        replay.matched_event(cycle.parent, metadata)


def target_fixture():
    return SimpleNamespace(identity=(0, 1, 1), capture_s=100., received_s=100.1, device_s=5.,
        speed_report={'source': 'operator_fixed_belt_speed', 'configured_speed_m_s': .4},
        center=np.zeros(3), entry_position=np.zeros(3), velocity=np.array([.4, 0., 0.]),
        points=np.ones((21, 3)), grasps=[np.eye(4)], object_pose=np.eye(4),
        detection={'scene_points': np.ones((5, 3)), 'belt_region': {
            'plane_to_base': np.eye(4), 'bounds_xy_m': np.array([[-1., 1.]]*2)}})


def test_clone_separates_numeric_input_and_omits_noncopyable_frame_caches():
    @dataclass(frozen=True)
    class Frame:
        depth_m: np.ndarray
    target = target_fixture()
    frame = Frame(np.ones((2, 2)))
    object.__setattr__(frame, '_perception_products', SimpleNamespace(lock=RLock()))
    target.detection['frame'] = frame
    cloned = replay.clone_input(target)
    assert cloned.detection['frame'] is not frame
    assert not hasattr(cloned.detection['frame'], '_perception_products')
    cloned.points[0, 0] = 12.
    assert target.points[0, 0] == 1.


def test_replay_uses_same_historical_age_preserves_failures_and_detects_input_mutation():
    config = SimpleNamespace(raw={})
    target, state = target_fixture(), {'joints': [0.]*7, 'width_m': .1, 'pose': [0.]*6}
    times = iter([10., 10.2, 10.4])
    def factory(cfg, clock):
        assert cfg.raw['fixed_belt_speed_m_s'] == .4
        def intercept(t, s):
            assert clock() == pytest.approx(100.5)  # 0.3 s historical + 0.2 s replay
            raise ValueError('intentional offline rejection')
        return SimpleNamespace(intercept=intercept, attempt_diagnostics=[{'rejected': True}])
    result = replay.replay_one(config, target, state, 100.3, planner_factory=factory, now=lambda: next(times))
    assert result['status'] == 'planning_rejected'
    assert result['age_at_submission_s'] == pytest.approx(.3)
    assert result['planning_wall_s'] == pytest.approx(.4)
    assert result['physical_success'] is None and result['input_unchanged']
    assert config.raw == {}  # Recorded-speed override remains local to replay.
    def mutating_factory(cfg, clock):
        def intercept(t, s):
            t.points[0, 0] += 1.
            raise RuntimeError('bad planner')
        return SimpleNamespace(intercept=intercept, attempt_diagnostics=[])
    with pytest.raises(RuntimeError, match='mutated'):
        replay.replay_one(config, target, state, 100.3, planner_factory=mutating_factory, now=lambda: 10.)


def test_configuration_checks_equal_settings_and_no_sdk(site, tmp_path, monkeypatch):
    base, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.2)
    base.write_text(yaml.safe_dump(raw))
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if 'pyAgxArm' in name or name.endswith('nero_pick_hardware') or name.endswith('backends.orbbec'):
            pytest.fail('offline replay imported device backend')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    session = tmp_path/'session'
    prepare_session(base, session, fixed_reserve_s=.3, freeze_position_m=.45,
        repeats=1, objects=['W01'], reservation_note='fixture', history_rule='cold')
    configs, info = replay.load_configs(session)
    assert set(configs) == {'A', 'B', 'C'} and info['current_source_hashes']
    path = session/'configs/B.yaml'
    changed = yaml.safe_load(path.read_text()); changed['prealign_orientation_s'] = 1.3
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match='changed after preparation'):
        replay.load_configs(session)
    manifest = json.loads((session/'manifest.json').read_text())
    manifest['groups']['B']['config_sha256'] = replay.digest(path)
    (session/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='non-ablation'):
        replay.load_configs(session)


def test_manifest_precedes_results_missing_inputs_stay_in_all_groups_no_ledger_write(tmp_path, monkeypatch):
    target, state = target_fixture(), {'joints': [0.]*7}
    output = tmp_path/'out'
    monkeypatch.setattr(replay, 'load_configs', lambda _: ({g: object() for g in 'ABC'}, {}))
    def restore(run, config):
        manifest = json.loads((output/'manifest.json').read_text())
        assert manifest['status'] == 'running' and len(manifest['selected_runs']) == 2
        if run.name == 'missing':
            raise FileNotFoundError('no target archive')
        return target, state, {'submission_s': 100.3, 'input_sha256': replay.input_fingerprint(target, state)}
    monkeypatch.setattr(replay, 'restore_input', restore)
    calls = []
    def one(config, t, s, stamp):
        calls.append(replay.input_fingerprint(t, s))
        return {'status': 'planning_rejected', 'input_sha256': calls[-1], 'physical_success': None}
    monkeypatch.setattr(replay, 'replay_one', one)
    summary = replay.run_replay(tmp_path/'unused', [tmp_path/'good', tmp_path/'missing'], output, repeats=2)
    assert summary['rows'] == 12 and summary['physical_success_rate'] is None
    assert len(set(calls)) == 1 and len(calls) == 6
    assert all(s == {'planning_rejected': 2, 'input_rejected': 2} for s in summary['groups'].values())
    with (output/'results.csv').open(encoding='utf-8-sig') as stream:
        assert len(list(csv.DictReader(stream))) == 12
    assert json.loads((output/'manifest.json').read_text())['status'] == 'completed'
    assert not list(output.glob('*.sqlite3'))
    with pytest.raises(FileExistsError):
        replay.run_replay(tmp_path/'unused', [tmp_path/'good'], output)


def test_group_tool_error_is_retained_separately_and_other_groups_continue(tmp_path, monkeypatch):
    target, state = target_fixture(), {'joints': [0.]*7}
    monkeypatch.setattr(replay, 'load_configs', lambda _: ({g: g for g in 'ABC'}, {}))
    monkeypatch.setattr(replay, 'restore_input', lambda *args: (target, state,
        {'submission_s': 100.3, 'input_sha256': replay.input_fingerprint(target, state)}))
    def one(config, t, s, stamp):
        if config == 'B':
            raise TypeError('tool initialization failed, not a robot-planner verdict')
        return {'status': 'planning_rejected', 'input_sha256': replay.input_fingerprint(t, s)}
    monkeypatch.setattr(replay, 'replay_one', one)
    summary = replay.run_replay(tmp_path/'unused', [tmp_path/'run'], tmp_path/'out')
    assert summary['groups']['B'] == {'replay_error': 1}
    assert summary['groups']['A'] == summary['groups']['C'] == {'planning_rejected': 1}
    assert len(list((tmp_path/'out').glob('I001_R001_*.json'))) == 3

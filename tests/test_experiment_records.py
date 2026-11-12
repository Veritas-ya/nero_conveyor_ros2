"""Research ledger integrity and realistic archived-event import, without SDKs."""
import builtins
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import yaml

from nero_core.two_second.experiment_records import Ledger, export, summarize
from nero_core.two_second.paper_experiment import prepare_session, main
from test_two_second_hardware import site


@pytest.fixture
def session(site, tmp_path):
    base, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.2)
    base.write_text(yaml.safe_dump(raw))
    original = base.read_bytes()
    directory = tmp_path/'session'
    manifest = prepare_session(base, directory, fixed_reserve_s=.3, freeze_position_m=.45,
        repeats=1, objects=['W01'], reservation_note='synthetic pilot fixture', history_rule='cold blocks')
    assert base.read_bytes() == original
    ledger = Ledger(directory)
    yield ledger
    ledger.close()


def group_trial(ledger, group):
    return next(r['trial_id'] for r in ledger.rows() if r['group_name'] == group)


def begin(ledger, trial, tmp_path):
    video = tmp_path/'video.mp4'; video.touch()
    ledger.begin(trial, video_file=video, video_offset_s=0.)


def fake_run(ledger, tmp_path, group, *, velocity_override=False):
    run = tmp_path/('run_'+group); run.mkdir()
    expected = ledger.manifest['groups'][group]
    effective = {'source_hashes': expected['source_hashes'], 'velocity_policy': dict(expected['velocity_policy']),
        'freeze_position_upstream_m': expected['freeze_position_upstream_m'],
        'freeze_position_tolerance_m': expected['freeze_position_tolerance_m'],
        'taught_segments': {}, 'freeze_source': 'tracked'}
    if velocity_override:
        effective['velocity_policy'] = {'source': 'operator_fixed_belt_speed', 'fixed_belt_speed_m_s': .4}
    (run/'summary.json').write_text(json.dumps(effective))
    directory = run/'cycle_001'; directory.mkdir()
    events = [
        {'event': 'planning_submitted', 'monotonic_s': 30., 'data': {
            'identity': [0, 1, 2], 'capture_monotonic_s': 29.9}},
        {'event': 'experiment_plan_ready', 'monotonic_s': 31., 'data': {'identity': [0, 1, 2],
            'group': group, 'delivery_attempt': 1, 'send_deadline_monotonic_s': 31.3,
            'contact_coordinate_m': .1, 'upstream_margin_m': .1, 'downstream_margin_m': .05,
            'timing': {'completed_monotonic_s': 31., 'rounds': [{'wall_s': 1., 'result': 'delivered'}]}}},
        {'event': 'planning_delivery', 'monotonic_s': 31.15,
            'data': {'delivery_monotonic_s': 31.15}},
        {'event': 'stage_finished', 'monotonic_s': 35., 'data': {'stage': 'taught_return'}}]
    (directory/'events.json').write_text(json.dumps(events))
    (directory/'report.json').write_text(json.dumps({'status': 'cycle_completed_from_feedback',
        'returned_to_wait': True, 'physical_grasp_success': None, 'physical_bin_placement_verified': False}))
    np.savez(directory/'execution_trace.npz', sample_stage=np.array(['recovery',
        'smooth_intercept_follow_brake', 'smooth_intercept_follow_brake']),
        dispatch_succeeded=np.array([True, False, True]),
        dispatch_start_monotonic_s=np.array([30.1, 31.1, 31.25]))
    return run


def test_prepare_is_deterministic_does_not_overwrite_or_import_devices(site, tmp_path, monkeypatch):
    base, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.2)
    base.write_text(yaml.safe_dump(raw))
    original = base.read_bytes()
    original_import = builtins.__import__
    def guarded(name, *a, **kw):
        if 'pyAgxArm' in name or name.endswith('nero_pick_hardware') or name.endswith('backends.orbbec'):
            pytest.fail('offline experiment tool imported a device SDK')
        return original_import(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    args = dict(fixed_reserve_s=.3, freeze_position_m=.45, repeats=5,
                reservation_note='test fixture', history_rule='reset per block')
    for name in ('s1', 's2'):
        prepare_session(base, tmp_path/name, **args)
    assert base.read_bytes() == original
    assert (tmp_path/'s1/plan.csv').read_bytes() == (tmp_path/'s2/plan.csv').read_bytes()
    with (tmp_path/'s1/plan.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 45
    for pair in {r['pair_id'] for r in rows}:
        assert {r['group'] for r in rows if r['pair_id'] == pair} == {'A', 'B', 'C'}
    configs = [yaml.safe_load((tmp_path/f's1/configs/{g}.yaml').read_text()) for g in 'ABC']
    allowed = {'planning_reserve_mode', 'planning_reserve_s', 'timing_experiment'}
    assert all({k: v for k, v in c.items() if k not in allowed} ==
               {k: v for k, v in configs[0].items() if k not in allowed} for c in configs)
    with pytest.raises(ValueError, match='overwrite'):
        prepare_session(base, tmp_path/'s1', **args)


def test_program_completion_never_sets_physical_success(session, tmp_path):
    trial = group_trial(session, 'C'); begin(session, trial, tmp_path)
    run = fake_run(session, tmp_path, 'C')
    session.bind(trial, run, 1)
    row = session._row(trial)
    assert row['status'] == 'pending'
    assert row['physical_grasp'] is row['physical_place'] is row['physical_return'] is None
    program = json.loads(row['program_json'])
    assert program['first_intercept_dispatch_monotonic_s'] == 31.25
    assert program['deadline_margin_at_first_dispatch_s'] == pytest.approx(.05)
    assert program['post_planning_to_delivery_s'] == pytest.approx(.15)
    assert program['planning_wall_s_all_rounds'] == 1.
    with pytest.raises(ValueError, match='AND'):
        session.finish(trial, status='success', grasp=1, place=1, evidence_note='incomplete')
    session.finish(trial, status='success', grasp=1, place=1, returned=1,
        evidence_note='synthetic video fixture', measured_speed_m_s=.4)
    stats = export(session, tmp_path/'export')
    assert stats['groups']['pilot:C']['success_rate'] == 1.
    assert stats['groups']['pilot:A']['success_rate'] is None
    assert stats['groups']['pilot:C']['planning_time_s']['median'] == 1.
    assert (tmp_path/'export/program_evidence.json').is_file()
    with (tmp_path/'export/trials.csv').open(encoding='utf-8-sig') as stream:
        row = next(r for r in csv.DictReader(stream) if r['trial_id'] == trial)
    assert float(row['planning_wall_s_all_rounds']) == 1.


def test_duplicate_binding_and_changed_effective_cli_config_are_rejected(session, tmp_path):
    trial = group_trial(session, 'A'); begin(session, trial, tmp_path)
    run = fake_run(session, tmp_path, 'A', velocity_override=True)
    with pytest.raises(ValueError, match='velocity'):
        session.bind(trial, run, 1)
    assert session._row(trial)['run_path'] is None
    meta = json.loads((run/'summary.json').read_text())
    meta['velocity_policy'] = session.manifest['groups']['A']['velocity_policy']
    (run/'summary.json').write_text(json.dumps(meta))
    session.bind(trial, run, 1)
    with pytest.raises(ValueError, match='one program'):
        session.bind(trial, run, 1)
    other = group_trial(session, 'B'); begin(session, other, tmp_path)
    with pytest.raises(ValueError, match='provenance'):
        session.bind(other, run, 1)


def test_missing_detection_is_counted_without_program_logs_and_corrections_are_audited(session, tmp_path):
    trial = group_trial(session, 'A'); begin(session, trial, tmp_path)
    session.finish(trial, status='failure', reason='missed_detection', evidence_note='video at zero')
    assert session._row(trial)['physical_place'] is None
    assert summarize(session.rows())['groups']['pilot:A']['completed'] == 1
    with pytest.raises(ValueError, match='correction'):
        session.finish(trial, status='success', grasp=1, place=1, returned=1, evidence_note='new observation')
    session.finish(trial, status='external_interruption', reason='operator interruption',
        evidence_note='reviewed video', correction_reason='first classification was incorrect')
    stats = summarize(session.rows())['groups']['pilot:A']
    assert stats['completed'] == 0 and stats['external_interruptions'] == 1
    audit = list(session.db.execute('SELECT action FROM audit WHERE trial_id=? ORDER BY seq', (trial,)))
    assert [r[0] for r in audit] == ['begin', 'finish', 'correct']


def test_unresolved_pair_is_not_imputed_as_failure_and_phase_is_not_pooled(session, tmp_path):
    for group, status in [('A', 'failure'), ('B', 'success')]:
        trial = group_trial(session, group); begin(session, trial, tmp_path)
        session.finish(trial, status=status, grasp=1 if group == 'B' else 0,
            place=1 if group == 'B' else None, returned=1 if group == 'B' else None,
            reason='empty_grasp' if group == 'A' else '', evidence_note='video fixture')
    summary = summarize(session.rows())
    assert summary['paired_comparisons']['pilot:B-A']['success_rate_difference'] == 1.
    assert summary['paired_comparisons']['pilot:C-B']['complete_pairs'] == 0
    assert summary['groups']['pilot:C']['completed'] == 0
    rows = session.rows(); formal = dict(rows[0], phase='formal', pair_id='other')
    summary = summarize(rows+[formal])
    assert 'formal:'+formal['group_name'] in summary['groups']
    assert summary['paired_comparisons']['formal:B-A']['complete_pairs'] == 0


def test_invalid_annotation_and_repeated_begin_do_not_change_history(session, tmp_path):
    trial = group_trial(session, 'A'); begin(session, trial, tmp_path)
    with pytest.raises(ValueError, match='already started'):
        begin(session, trial, tmp_path)
    for speed in [0., -1., float('nan'), float('inf'), True]:
        with pytest.raises(ValueError):
            session.finish(trial, status='success', grasp=1, place=1, returned=1,
                evidence_note='fixture', measured_speed_m_s=speed)
    assert session._row(trial)['status'] == 'pending'


def test_cli_returns_errors_without_devices(tmp_path, capsys):
    assert main(['status', '--session', str(tmp_path/'missing')]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['hardware_connections'] == report['commands_sent'] == 0


def test_main_experiment_can_generate_only_c_and_record_physical_cycle_time(site, tmp_path):
    base, data = site
    data.update(trajectory_profile='smooth', intercept_strategy='synchronized', prealign_orientation_s=1.2)
    base.write_text(yaml.safe_dump(data))
    folder = tmp_path/'only_c'
    manifest = prepare_session(base, folder, fixed_reserve_s=.3, freeze_position_m=.45,
        repeats=2, objects=['W01', 'W02'], groups=['C'], phase='formal',
        reservation_note='not used by C-only results', history_rule='documented cold block')
    assert manifest['planned_trials'] == 4
    ledger = Ledger(folder)
    try:
        trial = ledger.rows()[0]['trial_id']; begin(ledger, trial, tmp_path)
        ledger.finish(trial, status='success', grasp=1, place=1, returned=1,
            evidence_note='synthetic fixture', observed_cycle_time_s=5.3)
        assert ledger._row(trial)['observed_cycle_time_s'] == 5.3
        with pytest.raises(ValueError, match='separate'):
            ledger.finish(trial, status='failure', reason='test', evidence_note='fixture',
                observed_cycle_time_s=4., correction_reason='test')
    finally:
        ledger.close()


def test_zero_command_failure_can_be_bound_without_inventing_a_dispatch(session, tmp_path):
    trial = group_trial(session, 'A'); begin(session, trial, tmp_path)
    run = fake_run(session, tmp_path, 'A')
    cycle = run/'cycle_001'
    events = json.loads((cycle/'events.json').read_text())[:1]
    events.append({'event': 'experiment_plan_failed', 'monotonic_s': 31., 'data': {
        'group': 'A', 'details': {'timing_experiment': {'rounds': [
            {'wall_s': 1., 'result': 'missed'}]}}}})
    (cycle/'events.json').write_text(json.dumps(events))
    np.savez(cycle/'execution_trace.npz', sample_stage=np.array([], dtype=str),
        dispatch_succeeded=np.array([], dtype=bool), dispatch_start_monotonic_s=np.array([]))
    session.bind(trial, run, 1)
    program = json.loads(session._row(trial)['program_json'])
    assert program['first_intercept_dispatch_monotonic_s'] is None
    assert program['planning_wall_s_all_rounds'] == 1.
    assert session._row(trial)['status'] == 'pending'

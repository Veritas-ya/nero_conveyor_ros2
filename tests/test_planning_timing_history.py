"""Startup measurement provenance and compatibility, with no devices."""
import json
import platform

import pytest
import yaml

from nero_core.two_second.config import CellConfig
from nero_core.two_second.phase_timing_experiment import digest
from nero_core.two_second.phase_timing_test import prepare_test
from nero_core.two_second.planning_timing_history import startup_estimate
from test_two_second_hardware import site
from test_phase_timing_test_entry import base_site


def historical(base, root):
    config = prepare_test(base, root/'20261113', timing_history_root=root)
    directory = config.parent/'hardware/cycle_001'
    directory.mkdir(parents=True)
    (directory/'events.json').write_text(json.dumps([{'event':'attempt_cancelled', 'data':{
        'details':{'measured_rounds':[
            {'result':'missed', 'wall_s':.23}, {'result':'rejected', 'wall_s':.009}]}}}]))
    return config.parent


def test_new_process_first_round_seed_includes_failed_historical_work(site, tmp_path):
    base = base_site(site); history = tmp_path/'history'
    previous = historical(base, history)
    path = prepare_test(base, tmp_path/'next', timing_history_root=history)
    cfg = CellConfig.load(path)
    assert cfg.raw['measured_planning_initial_s'] == .23
    source = cfg.raw['measured_planning_initial_source']
    assert source['run'] == str(previous) and source['sample_count'] == 2
    assert source['sources'][0]['sha256'] == digest(previous/'hardware/cycle_001/events.json')
    assert source['guaranteed_time_bound'] is False


@pytest.mark.parametrize('change', ['base', 'phase', 'host', 'length', 'tampered', 'malformed'])
def test_incompatible_history_is_skipped_or_explicitly_rejected(site, tmp_path, change):
    base = base_site(site); history = tmp_path/'history'
    previous = historical(base, history)
    path = previous/'experiment.json'; manifest = json.loads(path.read_text())
    kwargs = dict(base_sha256=digest(base), phase_timing=manifest['phase_timing'], maximum_target_length_m=.2)
    if change == 'base': manifest['base_sha256'] = 'different'
    if change == 'phase': manifest['phase_timing'] = {}
    if change == 'host': manifest['host'] = platform.node()+'-other'
    if change == 'length': manifest['maximum_target_length_m'] = .16
    if change == 'tampered': (previous/'effective.yaml').write_text('modified')
    path.write_text('{' if change == 'malformed' else json.dumps(manifest))
    assert startup_estimate(history, **kwargs)[0] == 0.
    with pytest.raises(ValueError): startup_estimate(history, run=previous, **kwargs)


def test_cached_round_is_converted_back_to_cold_estimate(site, tmp_path):
    base = base_site(site); history = tmp_path/'history'
    previous = historical(base, history)
    (previous/'hardware/cycle_001/events.json').write_text(json.dumps([{'data':{
        'measured_planning':{'rounds':[{'result':'delivered', 'wall_s':.16,
                                      'cold_equivalent_wall_s':.24}]}}}]))
    path = prepare_test(base, tmp_path/'next', timing_history_root=history)
    assert CellConfig.load(path).raw['measured_planning_initial_s'] == .24


def test_cancelled_measurement_excluded_and_duplicate_diagnostics_counted_once(site, tmp_path):
    base = base_site(site); history = tmp_path/'history'
    previous = historical(base, history)
    details = {'measured_planning':{'rounds':[
        {'result':'rejected', 'wall_s':.8, 'estimate_eligible':False},
        {'result':'delivered', 'wall_s':.21, 'estimate_eligible':True}]}}
    (previous/'hardware/cycle_001/events.json').write_text(json.dumps([
        {'data':{'components':details, 'trajectory_diagnostics':details}}]))
    path = prepare_test(base, tmp_path/'next', timing_history_root=history)
    raw = CellConfig.load(path).raw
    assert raw['measured_planning_initial_s'] == .21
    assert raw['measured_planning_initial_source']['sample_count'] == 1


@pytest.mark.parametrize('value', [-1., True, '0.2', float('nan'), float('inf')])
def test_invalid_initial_estimate_rejected(site, value):
    path, raw = site
    raw['measured_planning_initial_s'] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='measured_planning_initial_s'): CellConfig.load(path)

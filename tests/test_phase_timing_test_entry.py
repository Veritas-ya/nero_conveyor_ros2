"""Dedicated live entry wiring with fake hardware; never connect devices."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.phase_timing import (KEY, assess_reference, authorize_hardware_test,
    make_reference, require_reference, validate_configuration, verify_plan)
from nero_core.two_second.phase_timing_test import main, prepare_test
from test_phase_timing import fixture
from test_two_second_hardware import site


def base_site(site):
    path, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
               closure_policy='scheduled_attempt', prealign_orientation_s=1.1,
               brake_duration_s=.9, taught_playback_enabled=False)
    path.write_text(yaml.safe_dump(raw))
    return path


def experimental():
    cfg, target, start, grasp = fixture()
    cfg.raw[KEY].update(closure_window_policy='fixed_follow_experiment',
                       follow_candidates_s=[.1], calibration_status='experimental_hardware')
    ref = make_reference(cfg, target, start, grasp, .9, .1, .6, now_s=.1, reservation_s=.2)
    return cfg, ref


def test_fixed_follow_does_not_claim_physical_closure_or_require_confirmation_dwell():
    cfg, ref = experimental()
    validate_configuration(cfg.raw, cfg.workspace, cfg.direction)
    report = assess_reference(cfg, ref)
    assert report['allowed']
    assert report['required_closure_window_s'] == .1
    assert report['reference_margin_score'] > 0  # fixed-time equality does not flatten ranking
    assert report['confirmation_required_before_brake'] is False
    assert report['physical_closure_before_brake_guaranteed'] is False
    cfg.raw[KEY].pop('closure_window_policy')
    assert 'closure_time_s' in assess_reference(cfg, ref)['reasons']


def test_experimental_execution_requires_dedicated_runtime_authorization():
    cfg, ref = experimental()
    path = SimpleNamespace(diagnostics={KEY: require_reference(cfg, ref)})
    with pytest.raises(PickRejected): verify_plan(cfg, path, ref, for_execution=True)
    cfg.raw['_phase_timing_hardware_test_authorized'] = True  # a saved YAML flag is insufficient
    with pytest.raises(ValueError): authorize_hardware_test(cfg)
    assert '_phase_timing_hardware_test_authorized' not in cfg.raw
    authorize_hardware_test(cfg, enabled=True)
    verify_plan(cfg, path, ref, for_execution=True)
    cfg.raw[KEY]['calibration_status'] = 'assumed_offline'
    with pytest.raises(ValueError): authorize_hardware_test(cfg, enabled=True)


@pytest.mark.parametrize('change', [
    {'closure_window_policy':'typo'}, {'follow_candidates_s':[.1,.2]},
    {'closure_window_policy':'confirm_before_brake'}])
def test_experimental_policy_validation(change):
    cfg, _ = experimental(); cfg.raw[KEY].update(change)
    with pytest.raises(ValueError): validate_configuration(cfg.raw, cfg.workspace, cfg.direction)


def test_private_snapshot_preserves_site_and_limits_and_refuses_overwrite(site, tmp_path):
    base = base_site(site); before = base.read_bytes()
    path = prepare_test(base, tmp_path/'experiment')
    cfg = CellConfig.load(path)
    assert base.read_bytes() == before
    original = yaml.safe_load(before); copied = yaml.safe_load(path.read_text())
    for key in ('joint_velocity_limits_rad_s', 'joint_acceleration_limits_rad_s2',
                'workspace_base_m', 'follow_interval_base_m', 'standby_joint_rad'):
        assert original[key] == copied[key]
    assert cfg.settings.full_plan_close_window_s == .1
    assert cfg.raw[KEY]['calibration_status'] == 'experimental_hardware'
    assert '_phase_timing_hardware_test_authorized' not in copied
    with pytest.raises(FileExistsError): prepare_test(base, path.parent)


@pytest.mark.parametrize('follow', [0., -.1, float('nan'), float('inf')])
def test_invalid_follow_does_not_publish_config(site, tmp_path, follow):
    base = base_site(site)
    with pytest.raises(ValueError): prepare_test(base, tmp_path/'invalid', follow_s=follow)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('execute', [False, True])
def test_cli_forwards_execution_only_when_explicit(site, tmp_path, monkeypatch, execute):
    import nero_core.two_second.main as hardware
    base = base_site(site); calls = []
    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        cfg = CellConfig.load(argv[argv.index('--config')+1])
        if '--execute' in argv: authorize_hardware_test(cfg, enabled=kwargs['phase_timing_hardware_test'])
        return 0
    monkeypatch.setattr(hardware, 'main', fake)
    args = ['--config', str(base), '--output', str(tmp_path/'cli'), '--belt-speed-cm-s', '50.8']
    if execute: args += ['--execute', '--count', '2']
    assert main(args) == 0
    argv, kwargs = calls[0]
    assert ('--execute' in argv) is execute
    assert ('--check-config' in argv) is not execute
    assert kwargs['freeze_source'] == 'single_exposure_native_depth'
    assert kwargs['phase_timing_hardware_test'] is True
    assert argv[argv.index('--belt-speed-cm-s')+1] == '50.8'
    assert json.loads((tmp_path/'cli/invocation.json').read_text())['execution_requested'] is execute


def test_original_hardware_entry_rejects_experimental_config_before_sdk(site, tmp_path, capsys):
    from nero_core.two_second.main import main as hardware_main
    path = prepare_test(base_site(site), tmp_path/'experiment')
    assert hardware_main(['--config', str(path), '--execute', '--output', str(tmp_path/'no_devices')]) == 2
    assert '独立真机测试入口' in capsys.readouterr().out
    assert not (tmp_path/'no_devices').exists()


def test_missing_speed_and_mutually_exclusive_actions_stop_before_creating_output(tmp_path):
    with pytest.raises(SystemExit): main(['--execute', '--output', str(tmp_path/'test')])
    with pytest.raises(SystemExit): main(['--execute', '--replay-run', 'unused'])
    assert not (tmp_path/'test').exists()


def test_full_frame_comparison_switch_changes_only_private_geometry_policy(site,tmp_path,monkeypatch):
    import nero_core.two_second.main as hardware
    base=base_site(site);before=base.read_bytes();seen=[]
    def fake(argv,**kwargs):
        cfg=CellConfig.load(argv[argv.index('--config')+1]);seen.append(cfg.raw['local_target_geometry_enabled'])
        return 0
    monkeypatch.setattr(hardware,'main',fake)
    assert main(['--config',str(base),'--output',str(tmp_path/'full'),
                 '--belt-speed-cm-s','53.5','--full-frame-geometry'])==0
    assert seen==[False] and base.read_bytes()==before

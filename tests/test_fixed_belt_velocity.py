"""User-provided speed wiring; synthetic frames/feedback, no physical success."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest

from nero_core.two_second.config import CellConfig
from nero_core.two_second.velocity_source import (
    FIXED,KALMAN,POINT_KALMAN,with_fixed_belt_speed,prediction_velocity,policy,speed_report)
from nero_core.two_second.reference import Intercept

CONFIG=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


@pytest.mark.parametrize('speed',[0.,-.01,float('nan'),float('inf'),5.95,True])
def test_invalid_speed_never_becomes_an_implicit_hardware_override(speed):
    with pytest.raises(ValueError):with_fixed_belt_speed(CellConfig.load(CONFIG),speed)


def test_current_site_config_uses_pointcloud_visual_speed():
    cfg=CellConfig.load(CONFIG)
    assert policy(cfg)['source']==POINT_KALMAN
    assert policy(cfg)['fixed_belt_speed_m_s'] is None
    assert policy(cfg)['visual_velocity_used'] is True


def test_override_is_explicit_detached_and_preserves_visual_policy():
    site=CellConfig.load(CONFIG)
    raw={k:v for k,v in site.raw.items() if k!='fixed_belt_speed_m_s'}
    raw['velocity_source']=KALMAN
    cfg=replace(site,raw=raw);fixed=with_fixed_belt_speed(cfg,.0595)
    assert cfg.raw['velocity_source']==KALMAN and 'fixed_belt_speed_m_s' not in cfg.raw
    assert policy(fixed)['source']==FIXED and not policy(fixed)['visual_velocity_used']
    assert fixed.source_sha256==cfg.source_sha256
    assert fixed.raw['environment_collision_scope']==cfg.raw['environment_collision_scope']


@pytest.mark.parametrize('normal',[[0,0,1],[.03,-.04,np.sqrt(1-.03**2-.04**2)]])
def test_fixed_speed_has_actual_surface_magnitude_and_cannot_use_visual_vector(normal):
    cfg=with_fixed_belt_speed(CellConfig.load(CONFIG),.0595)
    result=prediction_velocity(cfg,None,np.array(normal))
    other=prediction_velocity(cfg,np.array([123.,-456.,999.]),np.array(normal))
    np.testing.assert_array_equal(result,other)
    assert np.linalg.norm(result)==pytest.approx(.0595)
    assert result@normal==pytest.approx(0,abs=1e-12)
    assert result@cfg.direction>0


def test_prediction_still_includes_full_observation_age():
    cfg=with_fixed_belt_speed(CellConfig.load(CONFIG),.0595)
    velocity=prediction_velocity(cfg,None,np.array([0,0,1]))
    ref=Intercept(10.,11.,2.,np.zeros(3),velocity,np.array([0,0,.2]),[1,0,0,0],[1,0,0,0])
    assert np.linalg.norm(ref.contact_xyz)==pytest.approx(.0595*3.)


def test_fixed_worker_does_not_require_measured_speed_or_its_warmup(monkeypatch):
    from test_two_second_diagnostics import vision_fixture
    vision=vision_fixture(monkeypatch,'ready')
    vision.config.raw.update(velocity_source=FIXED,fixed_belt_speed_m_s=.0595)
    peer=vision.meter.latest_snapshot
    track=replace(peer.tracks[0],velocity_base_m_s=None,speed_valid=False,sample_count=2,span_s=.033)
    vision.meter.latest_snapshot=replace(peer,tracks=(track,))
    vision._run()
    assert vision.error is None
    target=vision.poll();assert target is not None
    assert np.linalg.norm(target.velocity)==pytest.approx(.0595)
    assert target.speed_report['source']==FIXED
    assert target.speed_report['measured_velocity_base_m_s'] is None
    assert target.speed_report['sample_count'] is None
    assert target.speed_report['tracking_sample_count']==2


@pytest.mark.parametrize('case',['feedback_missing','feedback_frame_skew','meter_not_current',
    'multiple_qualified_tracks','observation_stale_before_detection','observation_stale_after_detection'])
def test_fixed_speed_cannot_hide_other_live_observation_failures(monkeypatch,case):
    from test_two_second_diagnostics import vision_fixture
    vision=vision_fixture(monkeypatch,case)
    vision.config.raw.update(velocity_source=FIXED,fixed_belt_speed_m_s=.0595)
    vision._run()
    assert vision.poll() is None and vision.error is None
    diagnostics=vision.diagnostics()
    assert diagnostics['blocker_counts'] or diagnostics['grasp_geometry']['blocker_counts']


def test_fixed_target_report_cannot_claim_camera_measurement(monkeypatch):
    from test_two_second_diagnostics import vision_fixture
    vision=vision_fixture(monkeypatch,'ready')
    vision.config.raw.update(velocity_source=FIXED,fixed_belt_speed_m_s=.0595)
    vision._run();target=vision.poll()
    bad={**target.speed_report,'measured_velocity_base_m_s':target.velocity.tolist()}
    with pytest.raises(ValueError,match='不能标为视觉实测'):replace(target,speed_report=bad)
    bad={**target.speed_report,'configured_speed_m_s':.06}
    with pytest.raises(ValueError):replace(target,speed_report=bad)


def test_check_only_cli_validates_override_without_device_imports(monkeypatch,capsys):
    import builtins
    from nero_core.two_second.main import main
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if 'nero_pick_hardware' in name or 'backends.orbbec' in name:
            raise AssertionError('check-only must not import devices')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert main(['--config',str(CONFIG),'--check-config'])==0
    site_data=json.loads(capsys.readouterr().out)
    assert site_data['velocity_policy']['source']==POINT_KALMAN
    assert site_data['velocity_policy']['fixed_belt_speed_m_s'] is None
    assert site_data['hardware_connections']==site_data['commands_sent']==0
    assert main(['--config',str(CONFIG),'--fixed-belt-speed-m-s','0.0595','--check-config'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['velocity_policy']['source']==FIXED
    assert data['velocity_policy']['fixed_belt_speed_m_s']==.0595
    assert data['hardware_connections']==data['commands_sent']==0
    assert main(['--config',str(CONFIG),'--fixed-belt-speed-m-s','0.0595','--count','2'])==0


@pytest.mark.parametrize('speed', ['4.53', '12', '30'])
def test_cm_per_second_cli_converts_units_without_modifying_config_or_connecting_devices(speed, capsys):
    from nero_core.two_second.main import main
    before = CONFIG.read_bytes()
    assert main(['--config', str(CONFIG), '--belt-speed-cm-s', speed, '--check-config']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['velocity_policy']['fixed_belt_speed_m_s'] == pytest.approx(float(speed)/100.)
    assert report['velocity_policy']['fixed_belt_speed_cm_s'] == pytest.approx(float(speed))
    assert not report['velocity_policy']['visual_velocity_used']
    assert report['hardware_connections'] == report['commands_sent'] == 0
    assert CONFIG.read_bytes() == before


@pytest.mark.parametrize('speed', ['0', '-4.53', 'nan', 'inf', '30.01'])
def test_invalid_cm_per_second_is_rejected_before_hardware_start(speed, capsys, monkeypatch):
    import builtins
    from nero_core.two_second.main import main
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if 'nero_pick_hardware' in name or 'backends.orbbec' in name:
            pytest.fail('invalid speed must be rejected before device imports')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert main(['--config', str(CONFIG), '--belt-speed-cm-s='+speed, '--execute']) == 2


def test_speed_units_cannot_be_supplied_simultaneously():
    from nero_core.two_second.main import main
    with pytest.raises(SystemExit) as failure:
        main(['--config', str(CONFIG), '--belt-speed-cm-s', '4.53', '--fixed-belt-speed-m-s', '.0453'])
    assert failure.value.code == 2


def test_terminal_and_preview_do_not_call_operator_speed_kalman_measurement():
    from nero_core.two_second.console import ConsoleProgress
    from nero_core.two_second.speed_overlay import speed_lines,frozen_speed_snapshot
    from test_speed_overlay import target_fixture
    cfg=with_fixed_belt_speed(CellConfig.load(CONFIG),.0595)
    target,frame=target_fixture();target.velocity=np.array([0,-.0595,0])
    target.speed_report=speed_report(cfg,SimpleNamespace(sample_count=2),target.velocity)
    lines=[];console=ConsoleProgress(write=lines.append)
    console.observe({'event':'run_started','monotonic_s':1.,'data':{'velocity_policy':policy(cfg)}})
    console.observe({'event':'qualified_target','monotonic_s':2.,'data':{'speed_report':target.speed_report}})
    assert any('用户设定 5.95 cm/s' in line for line in lines)
    assert not any('卡尔曼' in line for line in lines)
    frozen=frozen_speed_snapshot(target,np.array([0,-1,0]),2.)
    text=speed_lines(None,frozen,now=2.,maximum_age_s=1.)[1][0]
    assert '用户设定带速' in text and '5.95' in text

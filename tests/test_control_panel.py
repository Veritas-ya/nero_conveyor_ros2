"""Operator intent, device ownership and HTTP boundaries, using fake devices."""
from io import StringIO
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from nero_core.control_panel.controller import CanInterface, PanelController, PanelError
from nero_core.control_panel.device_worker import enable_arm
from nero_core.control_panel.server import PanelApplication, PanelHTTPServer

ROOT = Path(__file__).parents[1]


class FakeCan:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = 0
    def status(self):
        self.calls += 1
        return {'ready': self.ready, 'present': self.ready, 'up': self.ready, 'message': 'fake CAN'}


class FakeProcess:
    pid = 987654
    def __init__(self):
        self.stdout = StringIO('[测试] fake child\n')
        self.code = None
    def poll(self): return self.code
    def wait(self, timeout=None): return self.code


@pytest.fixture
def controller(tmp_path):
    calls = []
    process = FakeProcess()
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process
    result = PanelController(tmp_path, can=FakeCan(), popen=popen, owner_busy=lambda:False)
    result._presets = PanelController(ROOT).presets()
    result.calls, result.fake_process = calls, process
    return result


def options(**changes):
    return {'preset': 'single', 'belt_speed_cm_s': 43.5, 'continuous': True, **changes}


@pytest.mark.parametrize('speed', [None, True, -1., 0., 100.01, float('nan'), float('inf'), '43.5; touch /tmp/no'])
def test_invalid_speed_cannot_start_a_child(controller, speed):
    with pytest.raises(PanelError): controller.start('pick', options(belt_speed_cm_s=speed))
    assert not controller.calls


def test_no_device_access_on_construction_or_preset_read(controller):
    assert controller.presets()[0]['freeze_upstream_cm'] == 70.
    assert controller.presets()[0]['freeze_tolerance_cm'] == 5.
    assert controller.can.calls == 0 and not controller.calls


def test_pick_reuses_original_entry_and_taught_paths_without_shell(controller):
    controller.start('pick', options(preset='coarse'))
    command, kwargs = controller.calls[0]
    assert 'nero_core.two_second.fixed_freeze_coarse_first_pick' in command
    assert '--execute' in command and '--web-preview' in command and '--preview' not in command
    assert '--taught-transfer' in command and '--taught-return' in command
    assert '--continuous' in command
    assert kwargs['start_new_session'] and not kwargs.get('shell', False)
    assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
    assert controller.snapshot()['job']['elapsed_s'] >= 0


def test_single_pick_count_and_camera_preview_have_distinct_permissions(controller):
    selected = controller.validate(options(continuous=False))
    command = controller.command('pick', selected, Path('/tmp/example-output'))
    assert command[command.index('--count')+1] == '1' and '--continuous' not in command
    controller.can.ready = False
    controller.start('camera', options())
    command = controller.calls[0][0]
    assert 'nero_core.control_panel.camera_worker' in command
    assert '--execute' not in command and controller.can.calls == 0


@pytest.mark.parametrize('action', ['can', 'enable', 'open_gripper', 'camera', 'pick'])
def test_active_owner_blocks_all_other_start_actions(controller, action):
    controller.start('camera', options())
    with pytest.raises(PanelError, match='已有任务'): controller.start(action, options())
    assert len(controller.calls) == 1


def test_stop_requests_original_handler_and_waits_for_process_exit(controller, monkeypatch):
    signals = []
    monkeypatch.setattr('nero_core.control_panel.controller.os.killpg', lambda *a: signals.append(a))
    controller.start('pick', options())
    controller.stop()
    assert signals[0][0] == controller.fake_process.pid
    assert controller.job['active'] and controller.job['stopping']
    with pytest.raises(PanelError): controller.start('camera', options())
    controller.fake_process.code = 0
    controller.refresh()
    assert not controller.job['active'] and controller.process is None


def test_rearm_is_not_a_robot_command(controller, monkeypatch):
    calls = []
    monkeypatch.setattr('nero_core.control_panel.controller.os.kill', lambda *a: calls.append(a))
    controller.start('pick', options())
    with pytest.raises(PanelError): controller.rearm()
    assert not calls


def test_no_implicit_switch_from_preview_to_motion(controller):
    controller.start('camera', options())
    with pytest.raises(PanelError): controller.start('pick', options())
    assert '--execute' not in controller.calls[0][0]


@pytest.mark.parametrize('action', ['can', 'enable', 'open_gripper', 'camera', 'pick'])
def test_existing_external_motion_owner_blocks_device_actions(controller, action):
    controller.owner_busy = lambda:True
    with pytest.raises(PanelError, match='另一个程序'):controller.start(action, options())
    assert not controller.calls and controller.job is None


def test_observe_and_end_view_never_signal_the_external_process(controller, monkeypatch):
    run = controller.root/'runs/external'
    (run/'preview').mkdir(parents=True)
    (run/'preview/latest.npz').touch()
    (run/'events.jsonl').write_text(json.dumps({'event':'runtime_progress','data':{
        'vision':{'velocity_policy':{'fixed_belt_speed_cm_s':22.7}}}})+'\n')
    controller.owner_busy = lambda:True
    controller.observe(run)
    assert controller.job['kind']=='observe' and controller.job['options']['belt_speed_cm_s']==22.7
    monkeypatch.setattr('nero_core.control_panel.controller.os.killpg',lambda *_:pytest.fail('must not signal external owner'))
    controller.stop()
    assert not controller.job['active'] and not controller.calls


def test_demo_never_checks_can_or_launches_device_processes(tmp_path):
    def forbidden(*a, **kw): pytest.fail('demo must not access devices')
    panel = PanelController(tmp_path, demo=True, can=SimpleNamespace(status=forbidden), popen=forbidden)
    panel._presets = PanelController(ROOT).presets()
    for kind in ('can', 'enable', 'open_gripper', 'camera'):
        panel.start(kind, options())
        panel.refresh()
    panel.stop()
    panel.start('pick', options())
    panel.stop()


def test_manual_gripper_open_uses_its_own_device_worker(controller):
    controller.start('open_gripper', options())
    command=controller.calls[0][0]
    assert 'nero_core.control_panel.gripper_worker' in command
    assert '--execute' not in command and '--taught-transfer' not in command
    assert controller.job['kind']=='open_gripper'


def test_manual_gripper_open_requires_ready_can(controller):
    controller.can.ready=False
    with pytest.raises(PanelError):controller.start('open_gripper', options())
    assert not controller.calls


def test_refresh_restores_operator_speed_without_starting_a_device(controller):
    controller.start('camera', options(belt_speed_cm_s=50.8))
    restored=PanelController(controller.root,can=FakeCan(),popen=lambda *a,**k:pytest.fail('must not resume'))
    restored._presets=controller.presets()
    assert restored.operator_options()==options(belt_speed_cm_s=50.8)
    assert restored.job is None and restored.can.calls==0


def test_can_start_never_restarts_an_already_ready_bus(monkeypatch):
    from mujoco_conveyor_sorting.runtime import motion_lease
    class Lease:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(motion_lease, 'MotionLease', Lease)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{'flags':['UP'], 'linkinfo':
            {'info_kind':'can', 'info_data': {'state':'ERROR-ACTIVE', 'bittiming': {'bitrate':1000000}}}}]))
    assert CanInterface(run).start()['ready']
    assert len(calls) == 1 and 'set' not in calls[0]


def test_can_permission_failure_is_reported_without_claiming_ready(monkeypatch):
    from mujoco_conveyor_sorting.runtime import motion_lease
    class Lease:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(motion_lease, 'MotionLease', Lease)
    def run(command, **kwargs):
        if '-j' in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'flags':[],
                'linkinfo': {'info_kind':'can','info_data':{}}}]))
        return SimpleNamespace(returncode=1, stderr='sudo: a password is required', stdout='')
    with pytest.raises(PanelError, match='sudo -v'): CanInterface(run).start()


def test_enable_sends_no_motion_gripper_or_zeroing_targets():
    log = []
    class Device:
        def __init__(self, *args):
            self.robot = SimpleNamespace(enable=lambda: log.append('enable') or True)
        def start(self): log.append('start')
        def snapshot(self, **kw):
            return {'joints':[0.]*7,'width_m':.1,'force_n':0.}
        def close(self): log.append('close')
    cfg = SimpleNamespace(raw={'hardware': {'can_channel':'fake0','firmware':'v121','joint_receive_offset_s':0.}})
    result = enable_arm(cfg, Event(), factory=Device)
    assert log == ['start', 'enable', 'close']
    assert result['enabled'] and result['motion_commands'] == result['gripper_commands'] == 0


def test_enable_does_not_mask_real_feedback_failure():
    closed = []
    class Device:
        robot = SimpleNamespace(enable=lambda: True)
        def __init__(self, *a): pass
        def start(self): pass
        def snapshot(self, **kw): raise RuntimeError('driver fault')
        def close(self): closed.append(True)
    cfg = SimpleNamespace(raw={'hardware': {'can_channel':'fake0','firmware':'v121','joint_receive_offset_s':0.}})
    with pytest.raises(RuntimeError, match='driver fault'): enable_arm(cfg, Event(), factory=Device)
    assert closed == [True]


@pytest.mark.parametrize('module,config', [
    ('main','two_second.hardware.tracked_freeze.yaml'),
    ('fixed_freeze_pick','two_second.hardware.legacy_observed.yaml'),
    ('fixed_freeze_coarse_first_pick','two_second.hardware.legacy_observed.yaml')])
def test_web_preview_config_check_keeps_all_three_entries_device_free(module,config,monkeypatch,capsys):
    from importlib import import_module
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    def forbidden(*a,**kw):pytest.fail('configuration check must not construct hardware or a viewer')
    monkeypatch.setattr(Hardware,'__init__',forbidden)
    monkeypatch.setattr('subprocess.Popen',forbidden)
    code=import_module('nero_core.two_second.'+module).main([
        '--config',str(ROOT/'src/nero_bringup/config'/config),'--belt-speed-cm-s','22.7',
        '--web-preview','--check-config'])
    result=json.loads(capsys.readouterr().out)
    assert code==0 and result['status']=='configuration_valid'
    assert result['hardware_connections']==result['commands_sent']==0


@pytest.fixture
def http_panel(controller):
    controller.demo = True
    app = PanelApplication(controller)
    server = PanelHTTPServer(('127.0.0.1', 0), app)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = 'http://127.0.0.1:'+str(server.server_address[1])
    yield base, app
    server.shutdown(); server.server_close(); thread.join()


def post(base, app, data, **headers):
    request = Request(base+'/api/action', json.dumps(data).encode(),
        {'Content-Type':'application/json', 'X-Control-Token':app.token, **headers})
    return urlopen(request)


def test_http_bootstrap_scene_and_static_assets_are_read_only(http_panel, controller):
    base, app = http_panel
    for path in ('/', '/app.css', '/app.js', '/api/bootstrap', '/api/state', '/api/scene'):
        with urlopen(base+path) as response: assert response.status == 200
    assert controller.job is None and not controller.calls


@pytest.mark.parametrize('headers', [
    {'X-Control-Token':'wrong'}, {'Origin':'http://evil.example'},
    {'Host':'evil.example'}, {'Content-Type':'application/x-www-form-urlencoded'},
])
def test_http_rejects_cross_origin_or_untrusted_actions(http_panel, controller, headers):
    base, app = http_panel
    with pytest.raises(HTTPError) as error: post(base, app, {'action':'pick','options':options()}, **headers)
    assert error.value.code in (403,415)
    assert controller.job is None and not controller.calls


def test_http_only_accepts_allowlisted_intent_not_shell_commands(http_panel, controller):
    base, app = http_panel
    for data in ({'action':'sh -c true','options':options()},
                 {'action':'camera','options':options(preset='../../evil')},
                 {'action':'camera','options':{**options(), 'command':'true'}}):
        with pytest.raises(HTTPError) as error: post(base, app, data)
        assert error.value.code == 409
    assert not controller.calls


def test_http_scene_etag_and_frozen_image_version(http_panel):
    base, app = http_panel
    with urlopen(base+'/api/scene') as response: etag=response.headers['ETag']
    with pytest.raises(HTTPError) as unchanged:
        urlopen(Request(base+'/api/scene', headers={'If-None-Match':etag}))
    assert unchanged.value.code == 304
    with urlopen(base+'/api/image/frozen.jpg?v=demo') as response:
        assert response.headers['Content-Type'] == 'image/jpeg'
    with pytest.raises(HTTPError) as wrong: urlopen(base+'/api/image/frozen.jpg?v=old')
    assert wrong.value.code == 409

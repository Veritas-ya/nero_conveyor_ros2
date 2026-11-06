"""No hardware: provenance, feedback-only capture, replay and task sequencing."""
from concurrent.futures import Future
import json
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

from nero_core.two_second.config import CellConfig
from nero_core.two_second.taught import TaughtSegment, certify_taught, provenance
from nero_core.two_second.teach_record import collect
from nero_core.two_second.trajectory import extrema, TimedPath
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_two_second_hardware import site, executor_fixture, cloud


def make_segment(tmp_path, cfg, kind='transfer'):
    joints = np.zeros((5, 7))
    joints[:, 0] = np.array([0., .01, .025, .04, .05])
    path = tmp_path / (kind+'.json')
    path.write_text(json.dumps({'schema':'nero_taught_joint_path_v1', 'segment':kind,
        'provenance':provenance(cfg), 'elapsed_s':[0., .2, .4, .6, .8], 'joints_rad':joints.tolist()}))
    return path


def test_recorded_segment_checks_provenance_and_time(site, tmp_path):
    path, _ = site; cfg = CellConfig.load(path)
    taught_path = make_segment(tmp_path, cfg)
    segment = TaughtSegment.load(taught_path, cfg, 'transfer')
    assert segment.joints.shape == (5, 7) and not segment.joints.flags.writeable
    with pytest.raises(ValueError, match='段类型'):
        TaughtSegment.load(taught_path, cfg, 'return')
    raw = json.loads(taught_path.read_text()); raw['provenance']['tcp_sha256'] = 'other'
    taught_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='tcp_sha256'):
        TaughtSegment.load(taught_path, cfg, 'transfer')
    raw['provenance'] = provenance(cfg); raw['elapsed_s'][2] = raw['elapsed_s'][1]
    taught_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='时间递增'):
        TaughtSegment.load(taught_path, cfg, 'transfer')


def test_configured_taught_pair_is_loaded_without_cli_flags(site, tmp_path, capsys):
    from nero_core.two_second.main import main
    path, data = site
    cfg = CellConfig.load(path)
    transfer = make_segment(tmp_path, cfg, 'transfer')
    returning = make_segment(tmp_path, cfg, 'return')
    data.update(trajectory_profile='smooth', taught_transfer_file=transfer.name,
                taught_return_file=returning.name)
    path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(path), '--check-config']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['taught_segments_source'] == 'configuration'
    assert result['taught_segments']['transfer']['path'] == str(transfer)
    assert result['taught_segments']['return']['path'] == str(returning)
    assert not result['taught_arguments_ignored']
    assert result['hardware_connections'] == result['commands_sent'] == 0


def test_cli_pair_overrides_missing_configured_defaults(site, tmp_path, capsys):
    from nero_core.two_second.main import main
    path, data = site
    cfg = CellConfig.load(path)
    transfer = make_segment(tmp_path, cfg, 'transfer')
    returning = make_segment(tmp_path, cfg, 'return')
    data.update(trajectory_profile='smooth', taught_transfer_file='missing_transfer.json',
                taught_return_file='missing_return.json')
    path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(path), '--check-config',
                 '--taught-transfer', str(transfer), '--taught-return', str(returning)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['taught_segments_source'] == 'cli'
    assert result['taught_segments']['transfer']['path'] == str(transfer)


@pytest.mark.parametrize('enabled', [True, False])
def test_missing_default_taught_files_do_not_silently_fall_back(site, capsys, enabled):
    from nero_core.two_second.main import main
    path, data = site
    data.update(trajectory_profile='smooth', taught_playback_enabled=enabled,
                taught_transfer_file='missing_transfer.json', taught_return_file='missing_return.json')
    path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(path), '--check-config']) == (2 if enabled else 0)
    result = json.loads(capsys.readouterr().out)
    assert result['hardware_connections'] == result['commands_sent'] == 0
    if enabled:
        assert result['status'] == 'configuration_incomplete'
    else:
        assert result['taught_segments'] == {} and result['taught_arguments_ignored']


@pytest.mark.parametrize('key', ['taught_transfer_file', 'taught_return_file'])
def test_default_taught_files_must_be_a_pair(site, key):
    path, data = site
    data[key] = 'one_segment.json'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='须同时配置'):
        CellConfig.load(path)


def test_certify_retimes_and_checks_actual_emitted_curve(site, tmp_path):
    path, _ = site; cfg = CellConfig.load(path)
    segment = TaughtSegment.load(make_segment(tmp_path, cfg), cfg, 'transfer')
    checks = []
    planner = SimpleNamespace(identity=(0, 1, 2), version=4, payload_local=np.zeros((3,3)),
        model=SimpleNamespace(command_lower=-np.ones(7), command_upper=np.ones(7)),
        check_path=lambda curve, width, **kw: checks.append((curve, width, kw)))
    state = {'joints': segment.joints[0].copy(), 'width_m':.04}
    curve = certify_taught(planner, cfg, segment, state, segment.joints[0],
                           payload=True, cancelled=lambda:False)
    assert checks[0][0] is curve or checks[0][0].curve is curve.curve
    assert checks[0][2]['payload'] is True
    assert curve.identity == planner.identity and curve.version == 5
    np.testing.assert_allclose(curve.q0, segment.joints[0], atol=1e-12)
    np.testing.assert_allclose(curve.q1, segment.joints[-1], atol=1e-12)
    for derivative in (1, 2):
        np.testing.assert_allclose(curve.curve([0., curve.duration], derivative), 0., atol=1e-9)
    assert np.max(np.abs(np.asarray(extrema(curve.curve, 1)))) <= .05+1e-8
    with pytest.raises(PickRejected, match='示教起点'):
        certify_taught(planner,cfg,segment,state,np.ones(7)*.01,payload=True,cancelled=lambda:False)
    with pytest.raises(PickRejected, match='取消'):
        certify_taught(planner,cfg,segment,state,segment.joints[0],payload=True,cancelled=lambda:True)


def test_real_robot_model_checks_synthetic_unloaded_taught_curve(tmp_path):
    from pathlib import Path
    from nero_core.two_second.planning import MeasuredPlanner
    path=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    cfg=CellConfig.load(path)
    planner=MeasuredPlanner(cfg); planner.identity=(0,1,2)
    planner.plane=np.eye(4);planner.plane[2,3]=-1.
    planner.support=np.array([0.,0.,1.]);planner.support_origin=planner.plane[:3,3].copy()
    planner.belt_bounds=np.array([[-5.,5.],[-5.,5.]])
    joints=np.tile(cfg.standby,(5,1));joints[:,0] += np.linspace(0,.02,5)
    segment=TaughtSegment('return',np.linspace(0.,1.,5),joints,'synthetic', 'test-only')
    state={'joints':joints[0],'width_m':cfg.settings.open_width_m}
    curve=certify_taught(planner,cfg,segment,state,joints[0],payload=False,cancelled=lambda:False)
    assert planner.stats['states'] > 5
    np.testing.assert_allclose(curve.q1,joints[-1],atol=1e-12)


def test_feedback_recording_only_reads_and_rejects_clock_reset():
    class Clock:
        def __init__(self): self.value=10.
        def __call__(self): return self.value
        def sleep(self, value): self.value += max(value,.001)
    clock=Clock()
    class ReadOnly:
        def snapshot_teaching_joints(self):
            return {'joints':[clock.value/100]*7, 'joint_stamp':clock.value,
                    'clock_epoch':0, 'joint_packet_span_s':.001}
    result=collect(ReadOnly(), .3, 20, clock=clock, sleep=clock.sleep)
    assert len(result['elapsed_s']) >= 4 and result['elapsed_s'][0] == 0.
    assert result['capture']['max_joint_packet_span_s'] == .001
    class Reset(ReadOnly):
        def snapshot_teaching_joints(self):
            row=super().snapshot_teaching_joints(); row['clock_epoch']=int(clock.value > 10.1); return row
    clock.value=10.
    with pytest.raises(ValueError, match='重置'):
        collect(Reset(), .3, 20, clock=clock, sleep=clock.sleep)


def test_passive_teaching_feedback_works_without_normal_execution_status():
    from threading import RLock
    import time
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    now=time.monotonic()
    class Timing:
        mapped_monotonic=now-.001
        mapping_quality='mapped'
        clock_epoch=4
        def to_dict(self):return {'mapping_quality':'mapped'}
    parser=SimpleNamespace(**{k:SimpleNamespace(feedback_time=Timing(),msg=None)
        for k in ('joint_12','joint_34','joint_56','joint_7')})
    robot=SimpleNamespace(_parser=parser,has_comm_error=lambda:False,
        get_joint_angles=lambda:SimpleNamespace(msg=[0.]*7))
    device=Hardware.__new__(Hardware);device.execute=False;device.robot=robot
    device.lock=RLock();device.feedback_wait_timeout_s=.1
    state=device.snapshot_teaching_joints()
    assert state['clock_epoch']==4 and mapped_stamp(state,'joint')==pytest.approx(now-.001)
    device.execute=True
    with pytest.raises(RuntimeError,match='只读'):
        device.snapshot_teaching_joints()


@pytest.mark.parametrize('enabled', [True, False])
def test_taught_cycle_connects_release_and_returns_to_configured_standby(monkeypatch, enabled):
    from nero_core.two_second.execution import TwoSecondExecutor
    from nero_core.two_second.planning import InterceptPlan
    from nero_core.two_second.reference import Intercept
    from nero_core.two_second.vision import MeasuredTarget, obb_grasps
    e, device, clock = executor_fixture()
    points=cloud(); poses,box=obb_grasps(points,np.eye(4),np.eye(4))
    target=MeasuredTarget((0,1,2),clock(),clock(),100.,0,1,np.zeros(3),np.zeros(3),tuple(poses),points,
        {'belt_region':{'plane_to_base':np.eye(4)},'obb':box},{},{'source':'nero7_xy_kalman'})
    e.config.raw['taught_playback_enabled'] = enabled
    e.config.raw['taught_segments']={'transfer':SimpleNamespace(kind='transfer',joints=np.zeros((4,7)),summary=lambda:{'segment':'transfer'}),
                                    'return':SimpleNamespace(kind='return',joints=np.zeros((4,7)),summary=lambda:{'segment':'return'})}
    class ImmediatePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,*a,**kw):
            f=Future()
            try:f.set_result(fn(*a,**kw))
            except BaseException as error:f.set_exception(error)
            return f
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor',ImmediatePool)
    def path(stage):
        e.planner.version += 1
        return TimedPath(lambda t,d=0:np.zeros(7),target.identity,e.planner.version,stage,1.,0.)
    def intercept(target,state,**kwargs):
        ref=Intercept(clock(),clock()+2.,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
        return InterceptPlan(TimedPath(lambda t,d=0:np.zeros(7),target.identity,1,'intercept_follow',2.6,0.),ref,target,1,target.grasps[0],())
    e.planner.intercept=intercept; e.planner.attach_measured=lambda *a:None
    e.planner.validate_release_opening=lambda *a:None; e.planner.release_measured=lambda *a:None
    e.planner.route=lambda state,stage,**kw:path(stage)
    monkeypatch.setattr(e,'_taught_route',lambda pool,state,segment,**kw:path('taught_'+segment.kind))
    stages=[]
    def stream(p,**kw):
        stages.append((p.stage,kw.get('carrying',False)))
        if p.stage=='intercept_follow':
            device.width=.04;device.force=2.;e.holding.held_width=.04
        return device.snapshot()
    monkeypatch.setattr(e,'_stream',stream)
    result=e.run(target)
    expected = ([('intercept_follow',False),('lift',True),('transfer_connect',True),
        ('taught_transfer',True),('return_connect',False),('taught_return',False),('return',False)]
        if enabled else [('intercept_follow',False),('lift',True),('transfer',True),('return',False)])
    assert stages == expected
    assert ('grip',.1) in device.commands and result['returned_to_wait']


def test_disabled_playback_ignores_old_cli_paths_without_loading(site, capsys):
    import yaml
    from nero_core.two_second.main import main
    path, data = site
    data['taught_playback_enabled'] = False
    path.write_text(yaml.safe_dump(data))
    assert main(['--config', str(path), '--check-config',
                 '--taught-transfer', '/nonexistent/transfer.json',
                 '--taught-return', '/nonexistent/return.json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['taught_arguments_ignored'] is True
    assert result['taught_segments'] == {}
    assert result['commands_sent'] == result['hardware_connections'] == 0

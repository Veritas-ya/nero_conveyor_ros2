"""Offline integration tests. Fakes below prove wiring, never physical grasps."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from threading import Event
import json
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.reference import Intercept
from nero_core.two_second.trajectory import certify_curve, extrema, TimedPath
from nero_core.two_second.vision import obb_grasps, MeasuredTarget
from nero_core.two_second.execution import TwoSecondExecutor
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected, PickSettings


@pytest.fixture
def site(tmp_path):
    identity = np.eye(4).tolist()
    (tmp_path/'tcp.yaml').write_text(yaml.safe_dump({'source': 'physical_measurement',
        'calibration_id': 'TEST_ONLY_NOT_FIELD_CALIBRATION', 'T_flange_tcp': identity,
        'T_canonical_tcp': identity}))
    (tmp_path/'camera.yaml').write_text(yaml.safe_dump({'runtime': {'camera': {
        'serial_number': 'TEST_ONLY', 'calibration_id': 'TEST_ONLY', 'camera_to_world_cv': identity,
        'depth_width': 640, 'depth_height': 480, 'color_width': 640, 'color_height': 480}}}))
    data = {'strategy': 'two_second', 'base_frame': 'test_base', 'camera_calibration_file': 'camera.yaml',
        'tcp_calibration_file': 'tcp.yaml', 'joint_names': [f'joint{i}' for i in range(1,8)],
        'belt_direction_base': [1.,0.,0.], 'follow_interval_base_m': [-.2,.8],
        'workspace_base_m': [[-2,2],[-2,2],[-2,2]], 'standby_joint_rad': [0.]*7,
        'bins_base': {'test_bin': identity}, 'unknown_category_bin': 'test_bin',
        'joint_velocity_limits_rad_s': [.05]*7, 'joint_acceleration_limits_rad_s2': [.1]*7,
        'maximum_target_height_m': .08, 'camera': {},
        'hardware': {'can_channel': 'NO_DEVICE', 'firmware': 'TEST', 'joint_command_mode': 'j'}}
    path = tmp_path/'site.yaml'
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_config_requires_actual_site_values_and_resolves_paths(site):
    path, _ = site
    cfg = CellConfig.load(path)
    assert Path(cfg.raw['tcp_calibration_file']).is_absolute()
    assert cfg.raw['command_advance_s'] == .1
    assert cfg.default_bin == 'test_bin'
    assert cfg.source_sha256
    template = Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.template.yaml'
    with pytest.raises(ValueError, match='必填'):
        CellConfig.load(template)


@pytest.mark.parametrize('value', [False, True, 'true', 'false', 1, None])
def test_workpiece_recovery_policy_requires_explicit_boolean(site, value):
    path, data = site
    assert CellConfig.load(path).raw['recovery_allow_workpiece_contact'] is False
    data['recovery_allow_workpiece_contact'] = value
    path.write_text(yaml.safe_dump(data))
    if isinstance(value, bool):
        assert CellConfig.load(path).raw['recovery_allow_workpiece_contact'] is value
    else:
        with pytest.raises(ValueError, match='recovery_allow_workpiece_contact'):
            CellConfig.load(path)


@pytest.mark.parametrize('speed', [3, 100])
@pytest.mark.parametrize('omitted', [
    ('joint_velocity_limits_rad_s',), ('joint_acceleration_limits_rad_s2',),
    ('joint_velocity_limits_rad_s', 'joint_acceleration_limits_rad_s2')])
def test_removed_site_limits_inherit_the_selected_original_profile(site, speed, omitted):
    path, data = site
    data['pick_settings'] = {'speed_percent': speed}
    for name in omitted:
        del data[name]
    path.write_text(yaml.safe_dump(data))
    cfg = CellConfig.load(path)
    for name, value, default, kind in (
        ('joint_velocity_limits_rad_s', cfg.velocity_limits, cfg.settings.joint_velocity_limit, 'velocity'),
        ('joint_acceleration_limits_rad_s2', cfg.acceleration_limits, cfg.settings.joint_acceleration_limit, 'acceleration')):
        np.testing.assert_allclose(value, [default]*7 if name in omitted else data[name])
        assert cfg.raw['motion_limit_sources'][kind] == ('PickSettings' if name in omitted else 'site_override')


@pytest.mark.parametrize('field', ['joint_velocity_limits_rad_s', 'joint_acceleration_limits_rad_s2'])
@pytest.mark.parametrize('value', [None, [float('inf')]*7, [float('nan')]*7])
def test_null_and_nonfinite_limits_never_disable_hardware_checks(site, field, value):
    path, data = site
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CellConfig.load(path)


def test_removed_acceleration_override_does_not_certify_an_unbounded_motion(site):
    path, data = site
    data['pick_settings'] = {'speed_percent': 100}
    del data['joint_velocity_limits_rad_s']
    del data['joint_acceleration_limits_rad_s2']
    path.write_text(yaml.safe_dump(data))
    cfg = CellConfig.load(path)
    np.testing.assert_allclose(cfg.acceleration_limits, 2.)
    with pytest.raises(PickRejected) as failure:
        certify_curve([0., .1], [[0.]*7, [.1]*7], cfg, -np.ones(7), np.ones(7),
                      identity=(0,1,2), version=1, stage='test')
    assert failure.value.details['derivative'] == 2


def test_config_accepts_sourced_per_joint_cap_and_rejects_excess(site):
    path, data = site
    data['pick_settings'] = {'speed_percent': 100}
    caps = [np.pi]*3+[1.25*np.pi]*4
    data['manufacturer_joint_velocity_limits_rad_s'] = caps
    data['manufacturer_velocity_limit_source'] = 'user_provided_test_spec'
    data['joint_velocity_limits_rad_s'] = caps
    path.write_text(yaml.safe_dump(data))
    cfg = CellConfig.load(path)
    np.testing.assert_allclose(cfg.velocity_limits, caps)
    assert cfg.raw['motion_limit_sources']['velocity_cap'] == 'user_provided_test_spec'
    data['joint_velocity_limits_rad_s'] = [*caps[:4], caps[4]+.01, *caps[5:]]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='上位机限速'):
        CellConfig.load(path)
    data['joint_velocity_limits_rad_s'] = caps
    del data['manufacturer_velocity_limit_source']
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='来源'):
        CellConfig.load(path)


def test_runtime_acceleration_policy_requires_explicit_boolean(site):
    path, data = site
    data['runtime_command_acceleration_check_enabled'] = 'false'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='布尔值'):
        CellConfig.load(path)


def test_missing_runtime_acceleration_option_defaults_to_observation_only(site):
    path, data = site
    data.pop('runtime_command_acceleration_check_enabled', None)
    path.write_text(yaml.safe_dump(data))
    cfg = CellConfig.load(path)
    assert cfg.raw['runtime_command_acceleration_check_enabled'] is False


def test_synchronized_strategy_is_explicit_and_requires_smooth(site):
    path, data = site
    data['intercept_strategy'] = 'synchronized'
    data['trajectory_profile'] = 'legacy'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='需要 smooth'):
        CellConfig.load(path)
    data['trajectory_profile'] = 'smooth'
    data['prealign_orientation_s'] = 2.
    path.write_text(yaml.safe_dump(data))
    assert CellConfig.load(path).raw['intercept_strategy'] == 'synchronized'
    data['intercept_strategy'] = 'unknown'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='intercept_strategy'):
        CellConfig.load(path)


def test_manufacturer_cap_percent_is_host_policy(site):
    path, data = site
    data['pick_settings'] = {'speed_percent': 50}
    data['manufacturer_joint_velocity_limits_rad_s'] = [4.]*7
    data['manufacturer_velocity_limit_source'] = 'test'
    data['joint_velocity_limits_rad_s'] = [2.]*7
    path.write_text(yaml.safe_dump(data))
    np.testing.assert_allclose(CellConfig.load(path).velocity_limits, 2.)
    data['joint_velocity_limits_rad_s'][4] = 2.01
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='上位机限速'):
        CellConfig.load(path)


def test_optional_firmware_query_compares_each_requested_axis():
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware, FeedbackError
    device = Hardware.__new__(Hardware)
    device.execute = True
    calls = []
    device.robot = SimpleNamespace(
        get_joint_angle_vel_limits=lambda j, **kw: SimpleNamespace(msg=SimpleNamespace(max_joint_spd=3.0 if j == 1 else 4.0)),
        get_joint_acc_limits=lambda j, **kw: SimpleNamespace(msg=SimpleNamespace(max_joint_acc=8.0)),
        set_speed_percent=lambda speed: calls.append(speed))
    requested = np.array([np.pi]*3+[1.25*np.pi]*4)
    with pytest.raises(FeedbackError, match='关节1固件速度'):
        device.prepare_motion(100, query_limits=True, requested_velocity_limits=requested,
                              requested_acceleration_limits=[8.]*7)
    assert calls == []
    device.robot.get_joint_angle_vel_limits = lambda j, **kw: SimpleNamespace(msg=SimpleNamespace(max_joint_spd=4.0))
    report = device.prepare_motion(100, query_limits=True, requested_velocity_limits=requested,
                                   requested_acceleration_limits=[8.]*7)
    assert report['firmware_limits_verified']
    np.testing.assert_allclose(report['requested_joint_velocity_limits_rad_s'], requested)
    assert calls == [100]


@pytest.mark.parametrize('field,value', [('standby_joint_rad',[0]*6), ('joint_velocity_limits_rad_s',[.2]*7),
    ('joint_acceleration_limits_rad_s2',[-1]*7), ('unknown_category_bin','missing'),
    ('joint_names',[f'joint{i}' for i in range(7,0,-1)]), ('planning_budget_s',float('nan')),
    ('maximum_target_height_m',.01)])
def test_config_rejects_unsafe_or_ambiguous_values(site, field, value):
    path, data = site
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CellConfig.load(path)


def test_default_entry_only_checks_config_without_importing_hardware(site, monkeypatch, capsys):
    import builtins
    from nero_core.two_second.main import main
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.endswith('nero_pick_hardware') or 'pyAgxArm' in name or name.endswith('backends.orbbec'):
            raise AssertionError('configuration check imported a device')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert main(['--config', str(site[0])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['commands_sent'] == result['hardware_connections'] == 0
    assert result['startup_return_wait_policy']['enabled']
    assert result['startup_return_wait_policy']['joint_command_api']=='move_j'
    assert any(Path(name).name=='move_to_wait.py' for name in result['source_hashes'])


def test_physical_tcp_is_applied_to_kinematic_site(site):
    from nero_core.two_second.planning import calibrated_model
    cfg = CellConfig.load(site[0])
    tcp = np.eye(4)
    tcp[:3, 3] = [.01, -.02, .15]
    tcp[:3, :3] = Rotation.from_euler('x', .1).as_matrix()
    model = calibrated_model(replace(cfg, tcp=tcp))
    np.testing.assert_allclose(model.tcp(), model.flange()@tcp, atol=1e-8)


def cloud():
    x, y, z = np.meshgrid(np.linspace(-.02,.02,7), np.linspace(-.01,.01,5), [.02,.04])
    return np.column_stack((x.ravel(),y.ravel(),z.ravel()))


def test_obb_uses_same_algorithm_and_tcp_rotation_as_simulation():
    from experiments.graspgenx_sim.obb_grasp import obb_top_down_grasp
    path = Path(__file__).parents[1]/'src/nero_core/experiments/graspgenx_sim/gripper_agx/transforms.json'
    transform = np.array(json.loads(path.read_text())['T_canonical_ee_site'])
    poses, _ = obb_grasps(cloud(), np.eye(4), transform)
    np.testing.assert_allclose(poses[0], obb_top_down_grasp(cloud())['tcp'], atol=1e-10)


def test_obb_is_equivariant_to_measured_belt_transform():
    plane = np.eye(4)
    plane[:3,:3] = Rotation.from_euler('xyz', [.04,.02,.7]).as_matrix()
    plane[:3,3] = [.4,-.2,.5]
    poses, _ = obb_grasps(cloud(), np.eye(4), np.eye(4))
    moved, _ = obb_grasps(cloud()@plane[:3,:3].T+plane[:3,3], plane, np.eye(4))
    # Rectangle axes can swap for floating-point equal-area solutions. All
    # parallel-jaw equivalent poses must still use the same physical centre.
    for candidate in moved:
        assert min(np.max(abs(candidate-plane@p)) for p in poses) < 1e-8


def test_shared_reference_keeps_exposure_age_and_two_second_duration():
    from experiments.two_second_grasp.controller import Intercept as SimIntercept
    assert SimIntercept is Intercept
    ref = Intercept(10., 11., 2., np.zeros(3), [.2,0,0], [0,0,.2], [1,0,0,0], [1,0,0,0])
    np.testing.assert_allclose(ref.contact_xyz, [.6,0,0])


def test_entire_polynomial_extrema_not_only_waypoints_are_checked():
    from scipy.interpolate import CubicSpline
    times = [0,1,2,3]
    qs = np.zeros((4,7)); qs[:,0] = [0,1,1,0]
    curve = CubicSpline(times, qs)
    assert extrema(curve,0)[1][0] > 1.
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*10, acceleration_limits=np.ones(7)*10)
    with pytest.raises(PickRejected, match='曲线'):
        certify_curve(times, qs, cfg, -np.ones(7), np.ones(7), identity=(0,1,2), version=1, stage='test', stationary_ends=False)


def test_curve_supplies_continuous_q_v_a():
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*10, acceleration_limits=np.ones(7)*10)
    path = certify_curve([0,1,2], np.array([[0]*7,[.1]*7,[.2]*7]), cfg,
        -np.ones(7), np.ones(7), identity=(0,1,2), version=1, stage='test')
    for a,b in zip(path.sample(1-1e-8), path.sample(1+1e-8)):
        np.testing.assert_allclose(a,b,atol=1e-7)


class Clock:
    def __init__(self): self.now = 1.
    def __call__(self): return self.now
    def sleep(self, dt): self.now += dt


class IdealDevice:
    """Explicit ideal feedback test double; no physical model or SDK."""
    def __init__(self, clock, empty=False):
        self.clock=clock; self.q=np.zeros(7); self.width=.1; self.force=0.
        self.commands=[]; self.empty=empty
    def snapshot(self):
        return {'joints':self.q.tolist(),'pose':[0,0,0,0,0,0], 'width_m':self.width,'force_n':self.force,
            'pose_stamp':self.clock(),'joint_stamp':self.clock(),'gripper_stamp':self.clock()}
    def snapshot_nowait(self): return self.snapshot()
    def move_joints(self,q): self.q=np.array(q); self.commands.append(('arm',self.clock()))
    def grip(self,width,force):
        self.commands.append(('grip',width)); self.width = (0. if self.empty else .04) if width==0 else width
        self.force = 0. if self.empty or width else force
    def prepare_motion(self,*a,**kw): self.commands.append(('prepare',self.clock()))
    def stop(self): self.commands.append(('stop',self.clock()))


def executor_fixture(clock=None, empty=False):
    clock=Clock() if clock is None else clock
    settings=PickSettings(speed_percent=100)
    config=SimpleNamespace(settings=settings, raw={'command_advance_s':.1,'maximum_observation_age_s':.25},
        tcp=np.eye(4),standby=np.zeros(7), default_bin='b', bins={'b':np.eye(4)},
        velocity_limits=np.ones(7)*10, acceleration_limits=np.ones(7)*100)
    device=IdealDevice(clock, empty)
    model=SimpleNamespace(check_feedback=lambda state:None, tcp=lambda:np.eye(4))
    planner=SimpleNamespace(model=model, version=1, support=np.array([0,0,1.]),
        valid=lambda *a,**kw:None, target=SimpleNamespace(velocity=np.zeros(3)))
    executor=TwoSecondExecutor(config,planner,device,clock=clock,sleep=clock.sleep)
    executor.owner=(0,1,2);executor.version=1
    return executor,device,clock


def test_old_identity_cancel_and_clock_reset_are_rejected_before_sends():
    executor, device, clock = executor_fixture()
    with pytest.raises(PickRejected, match='过期'):
        executor._active((0,1,9),1,0)
    executor.epoch=lambda:1
    with pytest.raises(PickRejected, match='重置'):
        executor._active((0,1,2),1,0)
    executor.cancel()
    with pytest.raises(PickRejected, match='取消'):
        executor._active((0,1,2),1,0)
    assert not device.commands


def test_wrong_path_identity_and_version_never_start_sender():
    executor,device,_ = executor_fixture()
    for identity, version in [((0,1,9),1),((0,1,2),0)]:
        path=SimpleNamespace(identity=identity,version=version)
        with pytest.raises(PickRejected,match='版本'):
            executor._stream(path,origin=1.,epoch=0)
    assert not device.commands


def test_cancelled_late_worker_result_cannot_be_accepted():
    from concurrent.futures import Future
    executor,device,_ = executor_fixture()
    future=Future(); future.set_result('late plan')
    executor.cancelled.set()
    with pytest.raises(PickRejected):
        executor._prepare(future,(0,1,2),1,0)
    assert not device.commands


@pytest.mark.parametrize('empty', [False,True])
def test_follow_continues_during_close_and_empty_jaw_never_lifts(monkeypatch, empty):
    """Drive the production sampling callback on an ideal synthetic clock."""
    executor,device,clock=executor_fixture(empty=empty)
    class Sender:
        def __init__(self, dev, step, after, **kwargs):
            self.dev,self.step,self.after=dev,step,after
            self.report=SimpleNamespace(exception=None,command_count=0,last_feedback=None)
        def start(self):
            try:
                for _ in range(300):
                    clock.sleep(.01)
                    state=self.dev.snapshot()
                    self.report.last_feedback=state
                    item=self.step(state,clock(),.01)
                    if item is None: break
                    q,meta=item;self.dev.move_joints(q);self.after(clock(),.01,q,meta)
                    self.report.command_count+=1
            except BaseException as error:self.report.exception=error
            return self
        def wait_stopped(self,*a):return True
        def join(self):return True
        def request_stop(self):pass
        def snapshot(self):return self.report
    monkeypatch.setattr('nero_core.two_second.execution.JointStreamSender',Sender)
    curve=lambda t,d=0:np.zeros(7)
    path=TimedPath(curve,(0,1,2),1,'intercept_follow',2.6,0.)
    reference=Intercept(1.,1.,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
    if empty:
        with pytest.raises(PickRejected,match='夹持未确认'):
            executor._stream(path,origin=1.,epoch=0,reference=reference)
    else:
        state=executor._stream(path,origin=1.,epoch=0,reference=reference)
        assert state['width_m']==.04
    close_index=next(i for i,c in enumerate(device.commands) if c[0]=='grip')
    assert sum(c[0]=='arm' for c in device.commands[close_index+1:]) > 35
    assert not any(c[0]=='grip' and c[1]==.1 for c in device.commands)
    assert executor.trace.command_rows
    assert np.isfinite(executor.trace.command_rows[0]['raw_trajectory_qd_rad_s']).all()
    assert np.isfinite(executor.trace.command_rows[0]['raw_trajectory_qdd_rad_s2']).all()


def test_payload_uses_visual_prediction_and_actual_tcp_not_simulator():
    from nero_core.two_second.planning import MeasuredPlanner
    cfg=SimpleNamespace()
    tcp=np.eye(4);tcp[:3,3]=[.5,0,.2]
    model=SimpleNamespace(check_feedback=lambda s:None,tcp=lambda:tcp)
    planner=MeasuredPlanner(cfg,model=model)
    planner.target=SimpleNamespace(points=cloud(),velocity=np.array([.1,0,0]),capture_s=1.)
    planner.attach_measured({},2.)
    np.testing.assert_allclose(planner.payload_local+tcp[:3,3],cloud()+[.1,0,0])


def test_cancel_reaches_actual_ik_before_work_is_started():
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel
    model = RobotModel()
    with pytest.raises(PickRejected,match='逆解已取消'):
        model.solve_pose([0,0,0],[1,0,0,0],np.zeros(7),cancelled=lambda:True)


def test_measured_target_copies_input_and_keeps_relative_grasp():
    points=cloud()
    poses,box=obb_grasps(points,np.eye(4),np.eye(4))
    target=MeasuredTarget((0,1,2),1.,1.1,100.,0,8,np.zeros(3),np.array([.1,0,0]),tuple(poses),points,
        {'belt_region':{'plane_to_base':np.eye(4)},'obb':box},{},{'source':'nero7_xy_kalman'})
    points[:]=8.
    assert not target.points.flags.writeable
    np.testing.assert_allclose(target.object_pose@target.grasps_object[0],target.grasps[0])
    assert np.max(target.points)<1.


@pytest.mark.parametrize('broken_log_sink', [False, True])
@pytest.mark.parametrize('unconfirmed', [False, True])
def test_full_cycle_wiring_returns_to_wait_and_never_claims_physical_success(monkeypatch, broken_log_sink, unconfirmed):
    """Ideal feedback proves stage order, not routes, contact or physical lifts."""
    from concurrent.futures import Future
    from nero_core.two_second.planning import InterceptPlan
    executor,device,clock=executor_fixture()
    if unconfirmed:
        executor.config.raw['closure_policy']='scheduled_attempt'
    if broken_log_sink:
        def broken_sink(row): raise OSError('injected log sink failure')
        executor.event_sink = broken_sink
    stages=[]
    identity=(0,1,2)
    points=cloud();poses,box=obb_grasps(points,np.eye(4),np.eye(4))
    target=MeasuredTarget(identity,clock(),clock(),100.,0,1,np.zeros(3),np.zeros(3),tuple(poses),points,
        {'belt_region':{'plane_to_base':np.eye(4)},'obb':box},{},{'source':'nero7_xy_kalman'})
    class ImmediatePool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,*a,**kw):
            future=Future()
            try:future.set_result(fn(*a,**kw))
            except BaseException as error:future.set_exception(error)
            return future
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor',ImmediatePool)
    def intercept(target,state,**kw):
        reference=Intercept(clock(),clock()+2.,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
        path=TimedPath(lambda t,d=0:np.zeros(7),identity,1,'intercept_follow',2.6,0.)
        return InterceptPlan(path,reference,target,1,target.grasps[0],())
    def route(state,*,stage,**kw):
        executor.planner.version+=1
        return TimedPath(lambda t,d=0:np.zeros(7),identity,executor.planner.version,stage,1.,0.)
    executor.planner.intercept=intercept
    executor.planner.route=route
    executor.planner.attach_measured=lambda *a:None
    executor.planner.validate_release_opening=lambda *a:None
    executor.planner.release_measured=lambda *a:None
    def stream(path,**kw):
        stages.append(path.stage)
        if path.stage=='intercept_follow':
            device.width=0. if unconfirmed else .04;device.force=0. if unconfirmed else 2.
            executor.holding.held_width=None if unconfirmed else .04
        return device.snapshot()
    monkeypatch.setattr(executor,'_stream',stream)
    # This fixture tests the attempt stages; automatic recovery has its own
    # sender/scene fixtures in test_failed_attempt_recovery.py.
    result=executor._run_attempt(target) if unconfirmed else executor.run(target)
    assert stages==(['intercept_follow'] if unconfirmed else ['intercept_follow','lift','transfer','return'])
    assert result['status']==('attempt_completed_grip_unconfirmed' if unconfirmed else 'cycle_completed_from_feedback')
    assert result['physical_grasp_success'] is None
    assert not result['physical_bin_placement_verified']
    assert executor.owner is None and executor.stage==('attempt_finished' if unconfirmed else 'waiting')
    assert (('grip',.1) in device.commands) != unconfirmed
    assert bool(executor.event_logging_errors) == broken_log_sink


def test_second_obb_candidate_is_attempted_after_first_ik_failure(monkeypatch):
    """Candidate scheduling only; IK and geometry below are explicit doubles."""
    from nero_core.two_second.planning import MeasuredPlanner
    count = [0]
    def solve(*args,**kwargs):
        count[0] += 1
        return SimpleNamespace(success=count[0] != 1, joint_positions=np.zeros(7),
            position_error=.001, orientation_error=.02, iterations=150,
            message='bounded IK diagnostic test')
    model = SimpleNamespace(tcp=lambda:np.eye(4),solve_pose=solve,check_feedback=lambda s:None,
        command_lower=-np.ones(7),command_upper=np.ones(7),
        ik=SimpleNamespace(config=SimpleNamespace(position_tolerance=.0003,orientation_tolerance=.01)))
    cfg = SimpleNamespace(settings=PickSettings(speed_percent=100),direction=np.array([1.,0,0]),
        workspace=np.array([[-1.,1.]]*3),
        follow_bounds=np.array([0.,1.]),velocity_limits=np.ones(7),acceleration_limits=np.ones(7),
        raw={'planning_reserve_s':1.,'command_advance_s':.1,'planning_budget_s':8.,'freeze_upstream_distance_m':.5})
    planner = MeasuredPlanner(cfg,model=model,clock=lambda:1.)
    monkeypatch.setattr(planner,'install',lambda *a:None)
    checked=[]
    monkeypatch.setattr(planner,'check_path',lambda path,*a,**kw:checked.append(path))
    grasp=np.eye(4);grasp[0,3]=-.4
    target=SimpleNamespace(identity=(0,1,2),capture_s=1.,center=np.array([-.4,0,0]),
        velocity=np.array([.1,0,0]),grasps=(grasp,grasp.copy()))
    result=planner.intercept(target,{'joints':np.zeros(7)})
    assert len(result.failures)==1 and '逆解' in result.failures[0]['reason']
    details = result.failures[0]['details']
    assert details['candidate_index'] == 0 and details['position_error'] == .001
    assert details['orientation_error'] == .02 and details['iterations'] == 150
    assert len(details['target_tcp_base_m']) == 3 and len(details['seed_joints_rad']) == 7
    assert details['returned_joint_margin_rad'] == [1.]*7
    assert count[0]>100 and len(checked)==1


def test_legacy_site_camera_and_provenance_are_self_contained():
    from hashlib import sha256
    from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
    from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraConfig
    root = Path(__file__).parents[1]
    config = root/'src/nero_bringup/config'
    site = yaml.safe_load((config/'two_second.hardware.legacy_observed.yaml').read_text())
    assert site['unknown_category_bin'] == 'common_bin'
    assert set(site['bins_base']) == {'common_bin'}
    assert site['maximum_target_height_m'] == .12
    calibration_path = config/site['camera_calibration_file']
    calibration = load_runtime_depth_calibration(calibration_path)
    camera = OrbbecCameraConfig.from_mapping(site['camera'])
    reconstruction_path = config/'calibration/eye_to_hand_20261017_reconstructed.json'
    result = json.loads(reconstruction_path.read_text())
    manifest = json.loads((root/'docs/旧项目现场配置来源.json').read_text())
    assert sha256(calibration_path.read_bytes()).hexdigest() == manifest['copied_runtime_sha256']
    assert sha256(reconstruction_path.read_bytes()).hexdigest() == manifest['copied_reconstructed_result_sha256']
    assert result['reconstruction']['is_reconstructed'] is True
    assert result['reconstruction']['original_file_restored_byte_for_byte'] is False
    assert result['accepted'] is True
    np.testing.assert_allclose(calibration['depth_to_base_cv'], result['depth_to_base_cv'], atol=1e-12)
    assert camera.serial_number == calibration['serial_number'] == result['serial_number']
    assert camera.calibration_id == calibration['calibration_id'] == result['calibration_id']
    assert [camera.depth_width, camera.depth_height] == calibration['capture_profiles']['depth']
    assert [camera.color_width, camera.color_height] == calibration['capture_profiles']['color']
    simulation = yaml.safe_load((root/'src/nero_core/mujoco_conveyor_sorting/configs/scene.yaml').read_text())
    base = np.asarray(simulation['workspace']['robot_base_position'])
    bounds = np.asarray([simulation['workspace']['safety_region'][axis] for axis in ('x', 'y', 'z')])
    np.testing.assert_allclose(site['workspace_base_m'], bounds-base[:, None], atol=1e-12)
    # Both ranges were explicitly extended to 38 cm; coordinates are checked
    # separately because the simulation and physical cell use different frames.
    np.testing.assert_allclose(site['follow_interval_base_m'], [-.20, .38], atol=1e-12)
    np.testing.assert_allclose(np.asarray(simulation['conveyor']['grasp_region'])-base[0],
                               [-.20, .38], atol=1e-12)
    expected_joints_deg = [2.728, 43.334, -0.263, 98.814, 3.253, -4.163, 29.269]
    # The current cell retains the original 50mm height increase and moves
    # the wait TCP upstream by 150mm (the additional 50mm is dated 2026-11-06).
    # Verify the Cartesian change, rather than fixing the old joint angles.
    from nero_core.two_second.planning import calibrated_model
    cfg=CellConfig.load(config/'two_second.hardware.legacy_observed.yaml')
    model=calibrated_model(cfg);raised=model.tcp().copy()
    model.set_state(np.deg2rad(expected_joints_deg),cfg.settings.open_width_m)
    expected=model.tcp().copy();expected[2,3]+=.05
    expected[:3,3] -= .15*cfg.direction
    np.testing.assert_allclose(raised,expected,atol=1e-8)
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel
    model = RobotModel()
    model.set_state(site['standby_joint_rad'], .1)
    flange = model.flange()
    np.testing.assert_allclose(flange[:3, 3], np.array([-377.009, -40.174, 202.380])/1000-.15*cfg.direction, atol=.00005)
    reported_rotation = Rotation.from_euler('xyz', [-59.456, 79.925, -147.315], degrees=True)
    assert (reported_rotation.inv() * Rotation.from_matrix(flange[:3, :3])).magnitude() < np.deg2rad(.01)
    cfg = CellConfig.load(config/'two_second.hardware.legacy_observed.yaml')
    provided_flange = np.eye(4)
    provided_flange[:3, 3] = np.array([-261.230, -273.775, 151.859])/1000
    provided_flange[:3, :3] = Rotation.from_euler('xyz', [-67.689, 80.611, -113.175], degrees=True).as_matrix()
    np.testing.assert_allclose(cfg.bins['common_bin'], provided_flange @ cfg.tcp, atol=1e-12)
    model.set_state(np.deg2rad([42.973, 43.327, -0.257, 98.791, 0.715, -4.168, 29.269]), .1)
    np.testing.assert_allclose(model.flange()[:3, 3], provided_flange[:3, 3], atol=.00005)
    assert Rotation.from_matrix(provided_flange[:3, :3].T @ model.flange()[:3, :3]).magnitude() < np.deg2rad(.01)
    np.testing.assert_allclose(cfg.bins['common_bin'], model.tcp(), atol=1e-4)


def test_legacy_nominal_tcp_requires_confirmation_bound_to_file(site):
    from hashlib import sha256
    from nero_core.two_second.planning import calibrated_model
    path, data = site
    nominal = Path(__file__).parents[1]/'src/nero_bringup/config/calibration/tcp_legacy_model_nominal.yaml'
    data['tcp_calibration_file'] = str(nominal)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='确认文件 SHA-256'):
        CellConfig.load(path)
    data['confirmed_nominal_tcp_sha256'] = '0'*64
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='确认文件 SHA-256'):
        CellConfig.load(path)
    data['confirmed_nominal_tcp_sha256'] = sha256(nominal.read_bytes()).hexdigest()
    path.write_text(yaml.safe_dump(data))
    cfg = CellConfig.load(path)
    assert cfg.raw['tcp_source'] == 'model_nominal'
    source = yaml.safe_load(nominal.read_text())
    np.testing.assert_allclose(cfg.tcp, source['T_flange_tcp'], atol=1e-12)
    np.testing.assert_allclose(cfg.canonical_tcp, source['T_canonical_tcp'], atol=1e-12)
    model = calibrated_model(cfg)
    np.testing.assert_allclose(model.tcp(), model.flange() @ cfg.tcp, atol=1e-8)


def test_predicted_contact_outside_workspace_is_rejected_before_ik(monkeypatch):
    """Replay the geometry of the observed long-horizon failure without hardware."""
    from nero_core.two_second.planning import MeasuredPlanner
    start = np.eye(4)
    start[:3, 3] = [-.40301017948256734, -.03278254853360396, -.01659617635638104]
    def forbidden(*args, **kwargs):
        pytest.fail('workspace rejection must happen before IK')
    model = SimpleNamespace(tcp=lambda: start, solve_pose=forbidden,check_feedback=lambda s:None)
    cfg = SimpleNamespace(settings=PickSettings(), direction=np.array([.07842607911448404,-.9969199316468343,0.]),
        follow_bounds=np.array([-.2,.28]), workspace=np.array([[-.62,.62],[-.48,.54],[-.6,1.]]),
        raw={'planning_reserve_s':1.5,'command_advance_s':.1,'planning_budget_s':8.,'freeze_upstream_distance_m':.5})
    planner = MeasuredPlanner(cfg, model=model, clock=lambda:.886)
    monkeypatch.setattr(planner, 'install', lambda *a: None)
    grasp = np.eye(4)
    grasp[:3, 3] = [-.43856789405312535,.6336322832966848,-.12277400392071434]
    target = SimpleNamespace(identity=(0,1,43),capture_s=0.,
        center=np.array([-.43888053132593263,.6264546170988521,-.12010143426441372]),
        velocity=np.array([-.02007274646756201,-.05022625705508764,.0022919135911264265]), grasps=(grasp,))
    with pytest.raises(PickRejected) as caught:
        planner.intercept(target, {'joints':np.zeros(7)})
    failure = caught.value.details['failures'][0]
    assert failure['details']['kind'] == 'predicted_contact_outside_workspace'
    assert failure['details']['contact_tcp_base_m'][0] == pytest.approx(-.66866780484)
    assert failure['details']['prediction_horizon_s'] == pytest.approx(11.463299811)
    # With the shared simulation freeze window this same exposure remains
    # in tracking, before even preparing a scene or consuming a task.
    from nero_core.two_second.reference import EntryNotReady
    cfg.raw['freeze_upstream_distance_m'] = .35
    monkeypatch.setattr(planner, 'install', forbidden)
    with pytest.raises(EntryNotReady) as pending:
        planner.intercept(target, {'joints':np.zeros(7)})
    assert pending.value.reason == 'too_far_upstream'


def test_upstream_camera_keeps_tracking_without_freezing_distant_object():
    from nero_core.two_second.vision import LiveVision
    from nero_core.two_second.reference import EntryNotReady
    cfg=SimpleNamespace(direction=np.array([1.,0,0]),follow_bounds=np.array([-.2,.28]),
        raw={'planning_reserve_s':1.5,'freeze_upstream_distance_m':.35})
    vision=LiveVision(None,None,cfg,None,None)
    with pytest.raises(EntryNotReady):
        vision.entry_time(1.,[-.66,0,0],[.05,0,0],1.1)
    assert vision.latest is None and not vision.consumed
    assert vision.diagnostics()['entry_gate']['reason']=='too_far_upstream'
    # A later actual observation of the same unconsumed track is eligible.
    assert vision.entry_time(4.,[-.50,0,0],[.05,0,0],4.1)==pytest.approx(10.)
    assert not vision.consumed
    assert vision.diagnostics()['entry_gate']['reason']=='eligible'


def test_faster_belt_admits_frozen_track_earlier_without_shortening_deadlines():
    from nero_core.two_second.vision import LiveVision
    from nero_core.two_second.reference import EntryNotReady
    raw={'planning_reserve_s':2.5,'command_advance_s':.1,'freeze_upstream_distance_m':.1,
         'synchronized_adaptive_freeze':True,'synchronized_max_freeze_upstream_distance_m':.35,
         'intercept_strategy':'synchronized','maximum_observation_age_s':1.,
         'prealign_orientation_s':2.,'synchronized_duration_s':.2}
    cfg=SimpleNamespace(direction=np.array([1.,0,0]),follow_bounds=np.array([-.2,.28]),
        settings=SimpleNamespace(full_plan_close_window_s=.8),raw=raw)
    vision=LiveVision(None,None,cfg,None,None)
    with pytest.raises(EntryNotReady, match='too far upstream'):
        vision.entry_time(1.,[-.46,0,0],[.111,0,0],1.1)
    vision.entry_time(1.,[-.44,0,0],[.111,0,0],1.1)
    window=vision.diagnostics()['entry_gate']['freeze_window']
    assert window['distance_m'] == pytest.approx(.2526)
    assert raw['planning_reserve_s'] == 2.5 and raw['command_advance_s'] == .1


@pytest.mark.parametrize('value', [False, True, 'true', 1, None])
def test_grasp_workpiece_policy_is_explicit_boolean(site, value):
    path, data = site
    assert CellConfig.load(path).raw['grasp_workpiece_collision_check_enabled'] is True
    data['grasp_workpiece_collision_check_enabled'] = value
    path.write_text(yaml.safe_dump(data))
    if isinstance(value, bool):
        assert CellConfig.load(path).raw['grasp_workpiece_collision_check_enabled'] is value
    else:
        with pytest.raises(ValueError, match='grasp_workpiece_collision_check_enabled'):
            CellConfig.load(path)

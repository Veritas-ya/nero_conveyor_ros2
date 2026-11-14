"""Independent geometry worker and real sender on explicitly ideal feedback."""
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.interpolate import BPoly
from scipy.spatial import cKDTree

from nero_core.two_second.config import CellConfig
from nero_core.two_second.execution import TwoSecondExecutor
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.runtime_sweep import (SweepService, SweepSettings, PeriodicSweep,
    scene_snapshot, request_for, refresh_request, sweep, _job)
from nero_core.two_second.trajectory import TimedPath
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


ROOT = Path(__file__).parents[1]


def delay_then_job(snapshot, request, seconds):
    """Runs in the real geometry process, never in the sender or test thread."""
    time.sleep(seconds)
    return _job(snapshot, request)


def delayed_test_helper_ready():
    """Warm this test-only module before measuring the injected sleep."""
    return True


class DelayedService:
    def __init__(self, service, seconds):
        self.base, self.seconds = service, seconds
        self.cancel, self.worker = service.cancel, service.worker
        self.submissions = []
        # ProcessPool imports a submitted callable's module on first use.
        # Importing pytest is not part of the deliberate 100 ms job delay.
        assert service.pool.submit(delayed_test_helper_ready).result(timeout=5.)

    def submit(self, snapshot, request):
        self.submissions.append(time.monotonic())
        if len(self.submissions) == 2:
            return self.base.pool.submit(delay_then_job, snapshot, request, self.seconds)
        return self.base.submit(snapshot, request)


@pytest.fixture(scope='module')
def worker():
    config = CellConfig.load(ROOT/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    service = SweepService(config)
    yield service
    service.close()


def fixture():
    config = CellConfig.load(ROOT/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    planner = MeasuredPlanner(config)
    planner.identity, planner.version, planner.scene_revision = (0,1,2), 1, 1
    # Explicit synthetic empty scene, support far below this short motion.
    planner.plane = np.eye(4); planner.plane[2,3] = -1.
    planner.support, planner.support_origin = planner.plane[:3,2], planner.plane[:3,3]
    planner.belt_bounds = np.array([[-2.,2.],[-2.,2.]])
    planner.scene_report = {'scope':'observed_scene'}
    planner.target = NS(points=np.array([[2.,2.,2.]]), velocity=np.array([.15,0.,0.]),
        capture_s=time.monotonic(), identity=planner.identity, detection={},state={},epoch=0,sequence=1)
    planner.target_tree = cKDTree(planner.target.points)
    end = config.standby.copy(); end[0] += .03
    curve = BPoly.from_derivatives([0.,.4], [[config.standby,np.zeros(7),np.zeros(7)],
                                           [end,np.zeros(7),np.zeros(7)]])
    return config, planner, TimedPath(curve, planner.identity, 1, 'test_process_sweep', .4, 0.)


class IdealDevice:
    """No SDK/model truth: follows sent q immediately; timestamps stay original."""
    def __init__(self, q):
        self.q = q.copy()
        self.commands = []
        self.pending_until = 0.
        self.blocking_reads = 0

    def snapshot(self):
        self.blocking_reads += 1
        return self._state()

    def _state(self):
        now = time.monotonic()
        return dict(joints=self.q.tolist(),pose=[0.]*6,width_m=.1,force_n=0.,
            joint_stamp=now, pose_stamp=now,gripper_stamp=now)

    def snapshot_nowait(self):
        return None if time.monotonic() < self.pending_until else self._state()

    def move_joints(self, q):
        self.q = np.array(q)
        self.commands.append((time.monotonic(),self.q.copy()))


def executor(service):
    config, planner, path = fixture()
    # Synthetic device has no flange kinematics. This check is tested with
    # measured data elsewhere; leave worker's real collision checks intact.
    planner.model.check_feedback = lambda state: None
    dev = IdealDevice(config.standby)
    e = TwoSecondExecutor(config,planner,dev,clock=time.monotonic,sleep=time.sleep,sweep_service=service)
    e.owner,e.version = planner.identity,1
    return e,dev,path


@pytest.mark.parametrize('key,value', [('runtime_sweep_mode','disabled'),
    ('runtime_sweep_hz',0),('runtime_sweep_hz',101),('runtime_sweep_hz',True),
    ('runtime_sweep_hz',float('nan')),('runtime_sweep_max_age_s',.201),
    ('runtime_sweep_max_age_s',.05),('runtime_sweep_max_age_s','0.2')])
def test_invalid_policy_rejected_without_devices(key,value):
    with pytest.raises(ValueError): SweepSettings.from_raw({key:value})


def test_200ms_feedback_lifetime_does_not_allow_arbitrary_slow_monitor():
    with pytest.raises(ValueError): SweepSettings.from_raw({'runtime_sweep_hz':5.})


@pytest.mark.parametrize('obstacle,payload',[(False,False),(True,False),(False,True)])
def test_process_geometry_matches_parent_geometry(worker, obstacle, payload):
    config,p,path=fixture()
    p.model.set_state(config.standby,.1)
    if obstacle:
        geom = next(iter(p.model.meshes))
        point = p.model.mesh_vertices[geom].mean(axis=0)@p.model.data.geom_xmat[geom].reshape(3,3).T+p.model.data.geom_xpos[geom]
        p.model.scene=np.array([point]);p.model.tree=cKDTree(p.model.scene)
    if payload:
        p.payload_local=np.array([[0.,0.,0.]])
        p.payload_tree=cKDTree(p.payload_local)
        p.capture_tcp=p.model.tcp().copy()
    snapshot=scene_snapshot(p,path)
    dev=IdealDevice(config.standby);state=dev.snapshot(); now=time.monotonic()
    request=request_for(p,state,dispatch=now,origin=now+.09,advance=.09,control_hz=100.,
        reference=None,attached=payload,smooth=False,scheduled_attempt=False,follow_end=.4)
    request['token']=snapshot['token']
    worker.cancel.clear()
    result=worker.submit(snapshot,request).result(timeout=5.)
    assert result['pid'] != __import__('os').getpid()
    try:
        sweep(p,path,request)
    except PickRejected as error:
        assert not result['ok'] and result['error']==str(error)
        assert result['details']==error.details
    else:
        assert result['ok'], result
    assert result['ok'] != obstacle


def test_direct_drop_height_guard_reaches_independent_geometry_process():
    config,p,path=fixture()
    p.model.set_state(config.standby,.1)
    tcp=p.model.tcp().copy()
    config.raw['experimental_direct_drop_guard']={
        'drop_tcp_xyz_base_m':tcp[:3,3].tolist(), 'radius_m':2.,
        'required_min_z_m':10., 'source_sha256':'test'}
    p.payload_local=np.array([[0.,0.,0.]])
    p.payload_tree=cKDTree(p.payload_local)
    p.capture_tcp=tcp
    snapshot=scene_snapshot(p,path)
    state=IdealDevice(config.standby).snapshot();now=time.monotonic()
    request=request_for(p,state,dispatch=now,origin=now+.09,advance=.09,control_hz=100.,
        reference=None,attached=True,smooth=False,scheduled_attempt=False,follow_end=.4)
    request['token']=snapshot['token']
    service=SweepService(config)
    try:
        service.cancel.clear()
        result=service.submit(snapshot,request).result(timeout=5.)
    finally:
        service.close()
    assert not result['ok']
    assert '低于名义框沿' in result['error']


def test_sender_interpolates_while_real_geometry_process_is_delayed(worker):
    service=DelayedService(worker,.10)
    e,dev,path=executor(service)
    try:
        state=e._stream(path,origin=time.monotonic()+.09,epoch=0)
    except PickRejected as error:
        pytest.fail(json.dumps({'error':str(error),'details':error.details,
            'submissions':service.submissions,'sends':[t for t,q in dev.commands],
            'monitor':[event['data'] for event in e.events if event['event']=='runtime_sweep_timing']}))
    began=service.submissions[1]
    during=[(t,q) for t,q in dev.commands if began+.005<=t<=began+.095]
    # Explicit progress, not repeating one endpoint to fake the frequency.
    assert len(during)>=5
    assert during[-1][1][0]>during[0][1][0]
    timing=next(v['data'] for v in e.events if v['event']=='runtime_sweep_timing')
    assert timing['worker_pid']!=__import__('os').getpid()
    assert timing['maximum_job_wall_s']<.1  # injected wait is outside _job
    assert timing['submitted'] < len(dev.commands)/2
    assert dev.blocking_reads==1  # No blocking packet retry on the sender.
    np.testing.assert_allclose(state['joints'],path.q1)


def test_expired_monitor_cancels_instead_of_extrapolating_forever(worker):
    service=DelayedService(worker,.30)
    e,dev,path=executor(service)
    with pytest.raises(PickRejected,match='后台碰撞监测结果过期'):
        e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert len(dev.commands)>5
    assert e.sender is None
    count=len(dev.commands);time.sleep(.03)
    assert len(dev.commands)==count


def test_sender_continues_with_original_stamped_feedback_for_short_parser_gap(worker):
    e,dev,path=executor(worker)
    # Prime is a normal initial snapshot; subsequent nowait reads can miss.
    dev.pending_until=time.monotonic()+.07
    state=e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert len(dev.commands)>20 and dev.blocking_reads==1
    np.testing.assert_allclose(state['joints'],path.q1)


def test_stale_feedback_still_stops_sender_despite_interpolable_curve(worker):
    e,dev,path=executor(worker)
    dev.pending_until=time.monotonic()+1.
    with pytest.raises(PickRejected,match='反馈无效或过期|碰撞监测结果过期'):
        e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert e.sender is None and 5<len(dev.commands)<30


def test_scene_version_change_does_not_reuse_cached_geometry(worker):
    e,dev,path=executor(worker)
    original=dev.move_joints
    def send(q):
        original(q)
        if len(dev.commands)==3:e.planner.scene_revision+=1
    dev.move_joints=send
    with pytest.raises(PickRejected,match='场景版本已改变'):
        e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert len(dev.commands)==3


def test_cancellation_cannot_keep_sending_from_cached_curve(worker):
    e,dev,path=executor(worker)
    original=dev.move_joints
    def send(q):
        original(q)
        if len(dev.commands)==3:e.cancelled.set()
    dev.move_joints=send
    with pytest.raises(PickRejected,match='取消'):
        e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert len(dev.commands)==3


def test_observed_collision_rejects_before_first_robot_command(worker):
    e,dev,path=executor(worker)
    model=e.planner.model
    model.set_state(dev.q,.1)
    geom=next(iter(model.meshes))
    point=model.mesh_vertices[geom].mean(axis=0)@model.data.geom_xmat[geom].reshape(3,3).T+model.data.geom_xpos[geom]
    model.scene=np.array([point]);model.tree=cKDTree(model.scene)
    with pytest.raises(PickRejected,match='路径碰到可见障碍点云'):
        e._stream(path,origin=time.monotonic()+.09,epoch=0)
    assert not dev.commands and e.sender is None


def test_foreign_or_closed_worker_result_cannot_refresh_monitor():
    cancel=NS(set=lambda:None)
    monitor=PeriodicSweep(NS(cancel=cancel,worker={'pid':-1}),{'token':'active'},SweepSettings(),clock=lambda:1.)
    result=dict(token='active',ok=True,source_s=.95,started_s=.9,completed_s=1.)
    monitor._accept(result)
    monitor._accept({**result,'token':'old','source_s':1.})
    assert monitor.result['source_s']==.95
    monitor.close()
    monitor._accept({**result,'source_s':1.})
    with pytest.raises(PickRejected,match='关闭'):monitor.guard(1.)


def test_one_slow_tick_does_not_create_another_120ms_forecast_wait():
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    class Clock:
        now=1.
        def __call__(self):return self.now
        def sleep(self,seconds):self.now+=max(0.,seconds)
    clock=Clock();sends=[];calls=[]
    class Device:
        def snapshot(self):return {}
        def move_joints(self,q):sends.append(clock())
    def step(*args):
        calls.append(clock())
        if len(calls)==1:clock.sleep(.12)
        return np.zeros(7),{}
    def after(*args):
        if len(sends)==2:sender.request_stop()
    sender=JointStreamSender(Device(),step,after,clock=clock,sleep=clock.sleep,
        max_prepare_lookahead_s=.01).start()
    sender.join()
    assert sender.snapshot().exception is None
    assert sends[1]-sends[0]==pytest.approx(.01)


@pytest.mark.parametrize('recovery',[False,True])
def test_workpiece_collision_evidence_and_recovery_policy_cross_process(recovery):
    config,p,path=fixture();p.model.set_state(config.standby,.1)
    # The site's operator policy disables the moving-workpiece guard. Test
    # enabled geometry explicitly, without changing that production policy.
    config.raw['grasp_workpiece_collision_check_enabled']=True
    geom=next(iter(p.model.meshes))
    point=p.model.mesh_vertices[geom].mean(axis=0)@p.model.data.geom_xmat[geom].reshape(3,3).T+p.model.data.geom_xpos[geom]
    p.target.points=np.array([point]);p.target.velocity=np.zeros(3)
    p.target_tree=cKDTree(p.target.points)
    if recovery:p._recovery_workpiece_contact_target=p.target
    state=IdealDevice(config.standby).snapshot();now=time.monotonic()
    request=request_for(p,state,dispatch=now,origin=now+.09,advance=.09,control_hz=100,
        reference=NS(duration_s=1.),attached=False,smooth=False,scheduled_attempt=False,follow_end=1.)
    snapshot=scene_snapshot(p,path);request['token']=snapshot['token']
    worker=SweepService(config)
    try:
        worker.cancel.clear()
        result=worker.submit(snapshot,request).result(timeout=5.)
    finally:worker.close()
    assert result['ok']==recovery
    if not recovery:
        assert result['error']=='机器人与实测工件点云碰撞'
        assert result['details']['inside_nominal_hull_count']==1
        assert result['collision_packets'][0]['metadata']['target_identity']==[0,1,2]
        assert 'diagnostic_error' not in result['details']


@pytest.mark.parametrize('reservation_s',[.41,3.22])
def test_monitor_reads_fresh_feedback_before_first_reserved_send(worker,reservation_s):
    """Reproduce the real failure: no sender step runs during this wait."""
    e,dev,path=executor(worker)
    not_before=time.monotonic()+reservation_s
    state=e._stream(path,origin=not_before+.09,epoch=0)
    assert len(dev.commands)>20
    assert dev.commands[0][0]>=not_before
    np.testing.assert_allclose(state['joints'],path.q1)
    timing=next(v['data'] for v in e.events if v['event']=='runtime_sweep_timing')
    assert timing['independent_feedback_refreshes']>=int(reservation_s*10)
    assert timing['maximum_observed_result_age_s']<.2
    assert e.sender is None


@pytest.mark.parametrize('reservation_s,stage',[
    (3.44,'smooth_intercept_follow_brake'),(.41,'recovery_return')])
def test_sender_uses_monitor_feedback_when_first_reserved_reads_miss(worker,reservation_s,stage):
    """Real sender/process, but synthetic feedback and no hardware connection."""
    from threading import get_ident
    owner=get_ident()
    e,dev,path=executor(worker)
    path=replace(path,stage=stage)
    first_read=[]
    misses=[]
    sender_reads=[]
    def read():
        if get_ident()!=owner:
            now=time.monotonic()
            sender_reads.append(now)
            if not first_read:first_read.append(now)
            # Also miss alternate later reads, when the last sender sample
            # can be newer than the supervisor's 20 Hz snapshot.
            if now-first_read[0]<.07 or len(sender_reads)%2==0:
                misses.append(now)
                return None
        return dev._state()
    dev.snapshot_nowait=read
    not_before=time.monotonic()+reservation_s
    state=e._stream(path,origin=not_before+.09,epoch=0)
    assert misses and len(dev.commands)>20 and dev.blocking_reads==1
    assert dev.commands[0][0]>=not_before
    np.testing.assert_allclose(state['joints'],path.q1)
    timing=next(v['data'] for v in e.events if v['event']=='runtime_sweep_timing')
    assert timing['sender_feedback_cache']['missing_sender_reads']==len(misses)
    assert timing['sender_feedback_cache']['monitor_feedback_used']>0
    assert e.sender is None


def test_shared_feedback_cache_cannot_hide_outage_before_first_send(worker):
    from threading import get_ident
    owner=get_ident()
    e,dev,path=executor(worker)
    began=time.monotonic()
    def read():
        if get_ident()!=owner or time.monotonic()-began>.12:
            return None
        return dev._state()
    dev.snapshot_nowait=read
    with pytest.raises(PickRejected,match='反馈无效或过期|碰撞监测结果过期'):
        e._stream(path,origin=began+.41+.09,epoch=0)
    assert not dev.commands and e.sender is None


def test_shared_feedback_cache_does_not_mask_device_exception(worker):
    from threading import get_ident
    owner=get_ident()
    e,dev,path=executor(worker)
    def read():
        if get_ident()!=owner:
            raise RuntimeError('synthetic device fault')
        return dev._state()
    dev.snapshot_nowait=read
    with pytest.raises(RuntimeError,match='synthetic device fault'):
        e._stream(path,origin=time.monotonic()+.41+.09,epoch=0)
    assert not dev.commands and e.sender is None


def test_fresh_feedback_reports_expired_packet_group_and_original_age():
    from nero_core.two_second.streaming import fresh_feedback
    state=dict(pose_stamp=1.,joint_stamp=1.,gripper_stamp=.7)
    with pytest.raises(PickRejected) as caught:
        fresh_feedback(state,1.05)
    assert caught.value.details['kind']=='gripper'
    assert caught.value.details['feedback_age_s']==pytest.approx(.35)
    assert caught.value.details['feedback_monotonic_s']==.7
    assert caught.value.details['maximum_age_s']==.2


def test_periodic_supervisor_never_renews_one_unchanged_feedback_packet():
    from concurrent.futures import Future
    class Service:
        worker={'pid':-1}
        cancel=NS(set=lambda:None,clear=lambda:None)
        def __init__(self):self.requests=[]
        def submit(self,snapshot,request):
            self.requests.append(request)
            result=Future();result.set_result(dict(token='task',ok=True,source_s=request['source_s'],
                started_s=1.,completed_s=1.))
            return result
    now=[1.]
    service=Service();state=dict(joints=[0.]*7,width_m=.1,pose_stamp=1.,joint_stamp=1.)
    template=dict(origin_s=2.,advance_s=.09,control_period_s=.01,contact_dispatch_s=None,
        unconfirmed_follow_end_dispatch_s=None,source_s=1.,payload=False)
    monitor=PeriodicSweep(service,{'token':'task'},SweepSettings(),clock=lambda:now[0],
        feedback_reader=lambda:state)
    monitor.prime(template)
    for t in [1.05,1.11,1.17,1.23]:
        now[0]=t;monitor.pump()
    assert len(service.requests)==1 and monitor.duplicate_requests_skipped>=3
    assert monitor.feedback_refreshes==0
    with pytest.raises(PickRejected,match='结果过期'):monitor.guard(now[0])
    monitor.close()


def test_supervisor_preserves_attachment_published_during_its_feedback_read():
    from concurrent.futures import Future
    class Service:
        worker={'pid':-1}
        cancel=NS(set=lambda:None)
        def submit(self,snapshot,request):
            self.request=request
            return Future()
    now=[1.1];service=Service()
    initial=dict(token='task',origin_s=1.,advance_s=.09,control_period_s=.01,contact_dispatch_s=1.,
        unconfirmed_follow_end_dispatch_s=1.2,source_s=1.,payload=False)
    payload=np.ones((3,3))
    def read():
        monitor.offer({**initial,'source_s':1.01,'payload':True,'payload_local':payload,
            'unconfirmed_follow_end_dispatch_s':None})
        return dict(joints=[.01]*7,width_m=.04,pose_stamp=1.09,joint_stamp=1.08)
    monitor=PeriodicSweep(service,{'token':'task'},SweepSettings(),clock=lambda:now[0],feedback_reader=read)
    monitor.offer(initial);monitor.pump()
    assert service.request['payload'] and service.request['payload_local'] is payload
    assert service.request['source_s']==1.08 and service.request['contact']
    assert monitor.latest_request['source_s']==1.01  # supervisor never overwrites sender's publication


def test_refresh_uses_original_packet_times_and_new_contact_boundary():
    state=dict(joints=[.1]*7,width_m=.04,pose_stamp=1.02,joint_stamp=1.01)
    template=dict(control_period_s=.01,contact_dispatch_s=1.1,
        unconfirmed_follow_end_dispatch_s=1.105,payload=True,payload_local=np.ones((2,3)))
    request=refresh_request(template,state,1.1)
    assert request['source_s']==1.01 and request['pose_s']==1.02
    assert request['contact'] and request['until_s']==1.105
    assert request['payload_local'] is template['payload_local']

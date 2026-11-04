"""Fault-injected real execution callbacks; no camera, CAN or physical robot."""
from types import SimpleNamespace
import numpy as np
import pytest
from test_two_second_hardware import executor_fixture
from nero_core.two_second.feedback_gates import (OpenConfirmation, RestArrival,
    TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.trajectory import TimedPath
from nero_core.two_second.stationary_hold import StationaryHold
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def test_close_confirmation_starts_after_actual_command_completed(monkeypatch):
    import nero_core.two_second.streaming as module
    e,device,clock=executor_fixture()
    e.config.raw['trajectory_profile']='smooth';starts=[];sends=[]
    original=device.grip
    def grip(width,force):
        original(width,force);clock.sleep(.02);sends.append(clock())
    device.grip=grip
    original_confirmation=module.GripConfirmation
    def confirmation(opening,force,requested):
        starts.append(requested);return original_confirmation(opening,force,requested)
    monkeypatch.setattr(module,'GripConfirmation',confirmation)
    e.planner.attach_measured=lambda *a:None
    class Sender:
        def __init__(self,dev,step,after,**kwargs):
            self.dev,self.step,self.after=dev,step,after
            self.report=SimpleNamespace(exception=None,command_count=0,last_feedback=None)
        def start(self):
            try:
                for _ in range(700):
                    clock.sleep(.01);state=self.dev.snapshot();self.report.last_feedback=state
                    item=self.step(state,clock(),.01)
                    if item is None:break
                    q,meta=item;self.dev.move_joints(q);self.after(clock(),.01,q,meta)
                    self.report.command_count+=1
            except BaseException as error:self.report.exception=error
            return self
        def wait_stopped(self,*a):return True
        def join(self):return True
        def request_stop(self):pass
        def snapshot(self):return self.report
    ref=SmoothIntercept(1.,1.,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
    path=TimedPath(lambda t,d=0:np.zeros(7),e.owner,1,'selfcheck',ref.full_duration_s,0.)
    module.stream(e,path,origin=1.,epoch=0,sender_factory=Sender,reference=ref)
    assert len(starts)==len(sends)==1 and starts[0]>=sends[0]


class CapturedSender:
    def __init__(self,dev,step,after,**kwargs):
        self.step,self.after,self.kwargs=step,after,kwargs
        self.report=SimpleNamespace(running=True,exception=None)
    def start(self):return self
    def snapshot(self):return self.report
    def request_stop(self):self.report.running=False
    def join(self):return True


def loaded_executor():
    e,device,clock=executor_fixture();e.rest_anchor=np.zeros(7)
    e.config.acceleration_limits[:]=8.
    e.config.raw.update(trajectory_profile='smooth',loaded_joint_acceleration_limits_rad_s2=[2.]*7,
        loaded_acceleration_limit_source='offline_test_only')
    e.holding.held_width=.04;device.width=.04;device.force=2.
    return e,device,clock


@pytest.mark.parametrize('error', [.00577091557535192, .009413863428370028,
    .007049144729764367, np.deg2rad(1.)])
def test_recorded_endpoint_errors_finish_after_fresh_stable_feedback(error):
    e, device, clock = executor_fixture()
    original_snapshot = device.snapshot
    def snapshot():
        state = original_snapshot()
        state['joints'][4] -= error
        return state
    device.snapshot = snapshot
    path = TimedPath(lambda t, d=0: np.zeros(7), e.owner, 1, 'endpoint_regression', .2, 0.)
    state = e._stream(path, origin=1.1, epoch=0)
    assert clock() >= 1.4  # physical end at 1.3, plus 0.1 seconds of fresh feedback
    assert abs(state['joints'][4]) == pytest.approx(error)
    np.testing.assert_array_equal(e.rest_anchor, np.zeros(7))


def test_endpoint_above_one_degree_times_out_with_matching_diagnostic():
    e, device, clock = executor_fixture()
    original_snapshot = device.snapshot
    def snapshot():
        state = original_snapshot()
        state['joints'][6] += np.deg2rad(1.01)
        return state
    device.snapshot = snapshot
    path = TimedPath(lambda t, d=0: np.zeros(7), e.owner, 1, 'endpoint_regression', .2, 0.)
    with pytest.raises(PickRejected, match='轨迹结束后反馈未到位') as failure:
        e._stream(path, origin=1.1, epoch=0)
    assert failure.value.details['tolerance_rad'] == pytest.approx(np.deg2rad(1.))
    assert failure.value.details['required_stable_s'] == .1


def test_one_degree_arrival_still_resets_dwell_when_one_joint_leaves_tolerance():
    gate = RestArrival(3., tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD)
    q = np.zeros(7); actual = np.full(7, np.deg2rad(.9))
    assert not gate.observe(actual, q, 3.)
    outside = actual.copy(); outside[6] = np.deg2rad(1.01)
    assert not gate.observe(outside, q, 3.09)
    assert not gate.observe(actual, q, 3.1)
    assert not gate.observe(actual, q, 3.19)
    assert gate.observe(actual, q, 3.2)


@pytest.mark.parametrize('error,allowed', [(np.deg2rad(1.), True), (np.deg2rad(1.01), False)])
def test_endpoint_tolerance_survives_all_stationary_handoff_checks(error, allowed):
    e, device, clock = loaded_executor()
    state = device.snapshot(); state['joints'][6] = error
    with StationaryHold(e, CapturedSender, epoch=0, carrying=True) as hold:
        callbacks = [
            lambda: hold.sender.step(state, clock(), .01),
            lambda: hold.sender.kwargs['finalize_send'](state, clock(), .01, e.rest_anchor, {}),
            lambda: hold.sender.kwargs['validate_dispatch'](state, clock(), .01, e.rest_anchor, {}),
        ]
        for callback in callbacks:
            if allowed:
                callback()
            else:
                with pytest.raises(PickRejected, match='偏离已认证终点'):
                    callback()


def test_loaded_stationary_hold_uses_loaded_limits_and_checks_actual_send():
    e,device,clock=loaded_executor()
    e.command_history.check_command_acceleration = True  # explicit enforcement test only
    e.command_history.q=np.full(7,-.001);e.command_history.dispatch=clock()-.01
    e.command_history.velocity=np.full(7,.1);e.command_history.previous_interval=.01
    with StationaryHold(e,CapturedSender,epoch=0,carrying=True) as hold:
        q,metadata=hold.sender.step(device.snapshot(),clock(),.01)
        assert metadata['motion_profile']=='loaded'
        assert metadata['joint_acceleration_limits_rad_s2']==[2.]*7
        # Preparation passes; an actual delayed send would require a=3.333,
        # greater than the loaded2 request but less than capability8.
        state=device.snapshot();clock.sleep(.01)
        with pytest.raises(PickRejected,match='加速度'):
            hold.sender.kwargs['finalize_send'](state,clock(),.02,q,metadata)
    assert not device.commands


def test_opening_hold_retains_loaded_profile_without_rejecting_intended_release():
    e,device,clock=loaded_executor();device.width=.1;device.force=0.
    with StationaryHold(e,CapturedSender,epoch=0,carrying=False,payload_motion=True) as hold:
        _,metadata=hold.sender.step(device.snapshot(),clock(),.01)
        assert metadata['motion_profile']=='loaded'
        assert metadata['joint_acceleration_limits_rad_s2']==[2.]*7


@pytest.mark.parametrize('gate', ['opening','arrival'])
def test_nonfinite_feedback_cannot_pass_confirmation(gate):
    if gate=='opening':
        item=OpenConfirmation(.1)
        def observe(t):return item.observe(float('nan'),t)
    else:
        item=RestArrival(0.)
        def observe(t):return item.observe(np.full(7,float('nan')),np.zeros(7),t)
    with pytest.raises(PickRejected):
        observe(1.);observe(1.5)


def test_mapped_feedback_epoch_reset_is_rejected_even_if_time_increases():
    from nero_core.two_second.streaming import fresh_feedback
    e,device,clock=executor_fixture();previous={}
    def state(epoch):
        value=device.snapshot()
        value.update(pose_stamp_monotonic=clock(),joint_stamp_monotonic=clock(),gripper_stamp_monotonic=clock(),
            feedback_time={key:{'clock_epoch':epoch,'mapping_quality':'mapped'} for key in (
                'joint_12','joint_34','joint_56','joint_7','end_pose_xy','end_pose_zrx','end_pose_ryrz','gripper')})
        return value
    fresh_feedback(state(0),clock(),previous);clock.sleep(.01)
    with pytest.raises(PickRejected,match='代次重置'):fresh_feedback(state(1),clock(),previous)


def test_opening_dwell_ignores_feedback_older_than_actual_open_command():
    gate=OpenConfirmation(.1,not_before=10.)
    for stamp in [9.8,9.96,10.]:assert not gate.observe(.1,stamp)
    assert not gate.observe(.1,10.01)
    assert not gate.observe(.1,10.15)
    assert gate.observe(.1,10.16)


class WallClock:
    def __call__(self):
        import time
        return time.monotonic()
    def sleep(self,dt):
        import time
        time.sleep(dt)


@pytest.mark.parametrize('failure', [None,'cancel','sender_fault','open_timeout'])
def test_release_keeps_one_sender_and_cleans_it_up_on_success_or_failure(failure):
    from nero_core.two_second.release import release_at_rest
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    e,device,clock=executor_fixture(clock=WallClock())
    e.config.raw['trajectory_profile']='smooth';e.stage='release';e.rest_anchor=np.zeros(7)
    e.planner.validate_release_opening=lambda s:None;e.planner.release_measured=lambda s:None
    e.holding.held_width=.04;device.width=.04;device.force=2.
    grip=device.grip
    opened=[False]
    def open_jaw(width,force):
        if failure!='open_timeout':grip(width,force)
        opened[0]=True
        if failure=='cancel':e.cancelled.set()
    device.grip=open_jaw
    snapshot=device.snapshot
    def sample():
        if opened[0] and failure=='sender_fault':raise RuntimeError('injected post-open feedback fault')
        return snapshot()
    device.snapshot=sample
    if failure=='open_timeout':
        original_wait=e.holding.wait
        def wait(*args,**kwargs):return original_wait(*args,timeout=.06,**kwargs)
        e.holding.wait=wait
    if failure is None:
        state=release_at_rest(e,device.snapshot(),epoch=0,sender_factory=JointStreamSender)
        assert state['width_m']==.1 and e.holding.held_width is None
        assert sum(c[0]=='arm' for c in device.commands)>=5
    else:
        with pytest.raises((RuntimeError,PickRejected)):
            release_at_rest(e,device.snapshot(),epoch=0,sender_factory=JointStreamSender)
        assert e.holding.held_width==.04
    assert e.sender is None
    count=len(device.commands);clock.sleep(.025);assert len(device.commands)==count

"""Return/perception overlap with isolated models and fake devices only."""
from dataclasses import replace
from threading import Event
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest
import yaml

from test_two_second_hardware import site

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.return_lookahead import ReturnLookahead
from nero_core.two_second.synchronized_reference import schedule_forward_join
from nero_core.two_second.cycle_loop import run_pick_cycles


class Clock:
    now = 10.2
    def __call__(self): return self.now


def wait_state(clock, offset=0.):
    return dict(joints=[offset]*7, width_m=.1, force_n=0., pose=[0.]*6,
        joint_stamp=clock(), pose_stamp=clock(), gripper_stamp=clock())


def target(index=2):
    return NS(identity=(0, index, index), epoch=0, sequence=index, capture_s=10.,
        received_s=10.01, device_s=1., velocity=np.array([.2, 0., 0.]),
        entry_position=np.array([-.8, 0., .05]),
        points=np.array([[-.81, -.01, .05], [-.79, .01, .06]]),
        state={'joints':[.8]*7, 'width_m':.06}, grasps=(np.eye(4),), speed_report={})


class Vision:
    clock_epoch = 0
    def __init__(self, item): self.latest=item; self.active=(0,1,1); self.discarded=[]
    def enable_lookahead(self): self.enabled=True
    def poll_pending(self): return self.latest
    def poll(self): return self.latest if self.active is None else None
    def consume(self, identity): self.active=identity; self.latest=None
    def resume_targets(self): self.active=None
    def discard_pending(self, identity):
        self.discarded.append(identity)
        if self.latest is not None and self.latest.identity==identity:self.latest=None


class Model:
    def set_state(self, joints, width): self.q=list(joints)
    def flange(self): return np.eye(4)


class Planner:
    def __init__(self):
        self.model=Model(); self.version=0; self.scene_revision=0; self.identity=None
        self.calls=[]; self.intercept_not_before_s=None
    def intercept(self, item, state, *, cancelled):
        if cancelled():raise PickRejected('fake planning cancelled')
        assert item.state['joints']==[.8]*7
        self.calls.append((item,state)); self.identity=item.identity
        self.version+=1; self.scene_revision+=1
        return NS(target=item, scene_revision=self.scene_revision,
            reference=NS(capture_s=item.capture_s,start_s=self.intercept_not_before_s+.2),
            path=NS(identity=item.identity,version=self.version,q0=np.asarray(state['joints']),planning_s=.05))


def fixture():
    clock=Clock();item=target();vision=Vision(item);planner=Planner();events=[]
    config=NS(standby=np.zeros(7),direction=np.array([1.,0.,0.]),follow_bounds=np.array([-.2,.38]),
        settings=NS(open_width_m=.1,min_force_n=1.),raw={
        'maximum_observation_age_s':1.,'command_advance_s':.09,'lookahead_return_margin_s':.2})
    worker=ReturnLookahead(config,vision,planner,cancelled=Event(),clock=clock,
        emit=lambda name,**data:events.append((name,data)))
    worker.begin_cycle((0,1,1))
    return worker,clock,item,vision,planner,events


@pytest.mark.parametrize('field,value',[
    ('continuous_lookahead_enabled','true'),('continuous_lookahead_enabled',1),
    ('lookahead_return_margin_s',True),('lookahead_return_margin_s',.09),
    ('lookahead_return_margin_s',float('nan')),('lookahead_return_margin_s','0.2')])
def test_defaults_enable_lookahead_and_invalid_options_fail_without_devices(site,field,value):
    from nero_core.two_second.config import CellConfig
    path,raw=site
    config=CellConfig.load(path)
    assert config.raw['continuous_lookahead_enabled'] is True
    assert config.raw['lookahead_return_margin_s']==.2
    raw[field]=value;path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match=field):CellConfig.load(path)


def test_freezes_while_returning_and_plans_before_confirmed_return_without_touching_active_model():
    worker,clock,item,vision,planner,events=fixture()
    active=Planner();active.identity=(0,1,1);active.version=17
    worker._step()
    assert worker.target is item and worker.ready.target is item and len(planner.calls)==1
    assert planner.calls[0][1]['joints'] == [0.]*7
    assert 'joint_stamp' not in planner.calls[0][1]
    assert not worker.return_confirmed and worker.available_s is None
    started=next(d for n,d in events if n=='lookahead_planning_started')
    assert started['return_prediction_known'] is False and started['return_confirmed'] is False
    worker.return_started(identity=(0,1,1),expected_ready_s=12.)
    worker._step();worker._step()
    assert worker.ready.target is item and len(planner.calls)==2
    assert worker.poll_ready() is None  # The arm is still returning.
    assert active.identity==(0,1,1) and active.version==17
    clock.now=12.01;worker.confirm_return(wait_state(worker.clock))
    prepared=worker.poll_ready()
    assert prepared.target.capture_s==10.  # Older than 1 s, accepted fresh originally.
    assert prepared.target.state is item.state
    assert worker.claim(prepared,active) is prepared and worker.planner is active
    assert [e[0] for e in events][:3]==['lookahead_frozen','lookahead_planning_started','lookahead_plan_ready']


def test_confirmed_wait_uses_latest_measured_state_not_nominal_or_capture_state():
    worker,clock,item,vision,planner,events=fixture()
    worker.confirm_return(wait_state(clock,.0068))
    clock.now += .01
    latest = wait_state(clock,.0069)
    worker.update_wait_state(latest)
    latest['joints'][0] = .8  # producer mutation must not change mailbox
    worker._step()
    assert planner.calls[0][1]['joints'] == [.0069]*7
    assert planner.calls[0][1]['joint_stamp'] == clock()
    assert item.state['joints'] == [.8]*7
    np.testing.assert_allclose(worker.ready.plan.path.q0, [.0069]*7)
    assert worker.poll_ready() is worker.ready
    data = next(d for name,d in events if name=='lookahead_plan_ready')
    assert data['start_state_source'] == 'measured_confirmed_wait_feedback'
    assert data['start_feedback_stamps']['joint'] == clock()


def test_nominal_ready_path_replans_in_background_after_actual_return_offset():
    worker,clock,item,vision,planner,events=fixture()
    worker._step();old=worker.ready
    worker.confirm_return(wait_state(clock,.00685))
    assert worker.ready is None and worker.target is item and not vision.discarded
    worker._step()
    assert len(planner.calls)==2 and worker.ready is not old
    np.testing.assert_allclose(worker.ready.plan.path.q0, [.00685]*7)
    assert worker.ready.target.capture_s == 10.
    assert any(name=='lookahead_wait_state_replan' for name,_ in events)


@pytest.mark.parametrize('change', ['stale','future','missing','closed','loaded','outside_wait'])
def test_confirmed_wait_invalid_feedback_never_falls_back_to_nominal(change):
    worker,clock,item,vision,planner,_=fixture()
    state = wait_state(clock,.0068)
    if change=='stale': state['joint_stamp']-=.21
    if change=='future': state['pose_stamp']+=.01
    if change=='missing': del state['gripper_stamp']
    if change=='closed': state['width_m']=.05
    if change=='loaded': state['force_n']=2.
    if change=='outside_wait': state['joints']=[.02]*7
    worker.confirm_return(state)
    worker._step()
    assert not planner.calls and worker.poll_ready() is None
    assert worker.wait_feedback_error
    worker.update_wait_state(wait_state(clock,.0068))
    worker._step()
    assert len(planner.calls)==1
    assert planner.calls[0][1]['joints']==[.0068]*7


def test_snapshot_expiration_blocks_ready_claim_until_fresh_update():
    worker,clock,item,vision,planner,_=fixture()
    worker.confirm_return(wait_state(clock,.0068));worker._step();ready=worker.ready
    clock.now+=.21
    assert worker.poll_ready() is None
    with pytest.raises(PickRejected,match='反馈无效或过期'):
        worker.claim(ready,Planner())
    worker.update_wait_state(wait_state(clock,.0068))
    assert worker.poll_ready() is ready and len(planner.calls)==1


def test_new_wait_motion_cancels_inflight_plan_and_retains_exposure():
    worker,clock,item,vision,planner,_=fixture()
    worker.confirm_return(wait_state(clock,.0068))
    original=planner.intercept
    def changing(*args,**kwargs):
        worker.update_wait_state(wait_state(clock,0.))
        assert kwargs['cancelled']()
        raise PickRejected('old actual start cancelled')
    planner.intercept=changing
    worker._step()
    assert worker.ready is None and worker.target is item and not vision.discarded
    planner.intercept=original;worker._step()
    assert worker.ready.start_state['joints']==[0.]*7


def test_unchanged_fresh_snapshots_do_not_restart_planning_and_clock_reversal_blocks():
    worker,clock,item,vision,planner,_=fixture()
    worker.confirm_return(wait_state(clock,.0068));worker._step()
    revision=worker.schedule_revision
    for _ in range(4):
        clock.now+=.01
        worker.update_wait_state(wait_state(clock,.0069));worker._step()
    assert len(planner.calls)==1 and worker.schedule_revision==revision
    old=wait_state(clock,.0069);old['joint_stamp']-=.02
    assert not worker.update_wait_state(old)
    assert worker.poll_ready() is None and '倒退' in worker.wait_feedback_error


def test_no_claim_before_actual_return_even_with_a_complete_model_path():
    worker,_,_,_,planner,_=fixture()
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    with pytest.raises(PickRejected,match='回位确认'):worker.claim(worker.ready,Planner())
    assert len(planner.calls)==1


def test_speculative_failure_keeps_original_exposure_and_retries_only_on_schedule_updates():
    worker,clock,item,vision,planner,events=fixture();calls=[]
    def fail(*args,**kwargs):
        calls.append(worker.schedule_revision)
        raise PickRejected('IK branch unavailable', details={'stage':'join','joint':6})
    planner.intercept=fail
    for _ in range(10):worker._step()
    assert len(calls)==1 and worker.target is item and not vision.discarded
    assert events[-1][0]=='lookahead_planning_deferred'
    assert events[-1][1]['failure_details']=={'stage':'join','joint':6}
    worker.return_started(identity=(0,1,1),expected_ready_s=12.)
    for _ in range(10):worker._step()
    assert len(calls)==2 and not vision.discarded
    clock.now=12.01;worker.confirm_return(wait_state(worker.clock));worker._step()
    assert len(calls)==3 and vision.discarded==[item.identity]
    assert events[-1][0]=='lookahead_rejected' and item.capture_s==10.


def test_schedule_update_cancels_inflight_speculation_without_discarding_the_target():
    worker,_,item,vision,planner,events=fixture();original=planner.intercept;calls=[]
    def plan(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:
            worker.return_started(identity=(0,1,1),expected_ready_s=12.)
            assert kwargs['cancelled']()
            raise PickRejected('old schedule cancelled')
        return original(*args,**kwargs)
    planner.intercept=plan
    worker._step()
    assert worker.ready is None and worker.target is item and not vision.discarded
    worker._step()
    assert len(calls)==2 and worker.ready.target is item


def test_late_failed_claim_cannot_discard_target_already_queued_for_rescheduling():
    worker,_,item,vision,_,_=fixture();worker._step();old=worker.ready
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    assert worker.ready is None
    worker.discard_ready(old,PickRejected('cache changed before claim'))
    assert worker.target is item and not vision.discarded
    worker._step()
    assert worker.ready.target is item


def test_actual_cycle_can_plan_next_target_before_any_return_path_or_return_confirmation():
    worker,clock,item,vision,spare,events=fixture()
    first=target(1);vision.latest=first;vision.active=None
    active=Planner();active.identity=(0,8,8);active.version=17
    planned=Event();original=spare.intercept
    def plan(*args,**kwargs):
        result=original(*args,**kwargs)
        planned.set()
        return result
    spare.intercept=plan
    class Executor:
        def __init__(self,config,planner,device,**kwargs):self.planner=planner
        def run(self,selected):
            if selected is first:
                # Simulate an arm still carrying; no return announcement exists.
                vision.latest=item
                assert planned.wait(2.), 'planning was incorrectly gated on return'
                assert worker.available_s is None and not worker.return_confirmed
                assert worker.poll_ready() is None
                assert active.identity==(0,8,8) and active.version==17
            else:
                assert selected is item and self.planner is spare
                assert self.prefetched_intercept.target is item
            return {'status':'cycle_completed_from_feedback','returned_to_wait':True}
    result={'completed_feedback_cycles':0};executors=[]
    worker.owner=None  # Production starts idle, without the unit fixture's old cycle.
    worker.start()
    try:
        run_pick_cycles(worker.config,active,NS(snapshot=lambda:wait_state(clock)),vision,count=2,cancelled=worker.cancelled,
            result=result,reports=[],executors=executors,set_phase=lambda *a,**kw:None,
            journal=NS(emit=lambda *a,**kw:None),executor_factory=Executor,
            clock=clock,lookahead=worker)
    finally:
        worker.close()
    assert result['completed_feedback_cycles']==2
    starts=[d for name,d in events if name=='lookahead_planning_started']
    assert starts and starts[0]['return_prediction_known'] is False
    assert starts[0]['return_confirmed'] is False


@pytest.mark.parametrize('change',['epoch','scene','version','identity','exposure'])
def test_stale_cache_is_skipped_without_dispatch_and_preserves_raw_failure_details(change):
    worker,clock,item,vision,planner,events=fixture()
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    if change=='epoch':vision.clock_epoch=1
    elif change=='scene':planner.scene_revision+=1
    elif change=='version':planner.version+=1
    elif change=='identity':planner.identity=(0,9,9)
    elif change=='exposure':vision.latest=target(3)
    worker.confirm_return(wait_state(worker.clock))
    assert worker.poll_ready() is None
    assert events[-1][0]=='lookahead_rejected' and events[-1][1]['model_path_generated']
    assert vision.discarded==[item.identity]


def test_old_exposure_is_rejected_at_first_reservation_not_silently_retimestamped():
    worker,clock,item,vision,planner,events=fixture();clock.now=12.
    worker._step()
    assert not planner.calls and vision.discarded==[item.identity]
    assert events[-1][1]['model_path_generated'] is False and item.capture_s==10.


def test_pending_target_that_passes_region_is_released_without_waiting_for_return_schedule():
    worker,clock,item,vision,planner,events=fixture();worker._step()
    clock.now=17.;worker._step();worker._step()
    assert len(planner.calls)==1 and vision.discarded==[item.identity]
    assert events[-1][1]['failure_details']['reason']=='lookahead_target_past_region'


def test_expired_schedule_retains_exposure_and_recomputes_after_actual_return():
    worker,clock,item,vision,planner,events=fixture()
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    clock.now=12.2;worker._step()
    assert worker.ready is None and not worker.return_confirmed
    assert not vision.discarded and vision.latest is item
    assert events[-1][1]['model_path_generated']
    assert events[-1][0]=='lookahead_plan_expired_replan'
    # No repeated speculative planning while a delayed return is unconfirmed.
    for _ in range(4):worker._step()
    assert len(planner.calls)==1
    clock.now=13.;worker.confirm_return(wait_state(worker.clock));worker._step()
    assert len(planner.calls)==2 and worker.ready.target is item
    assert worker.ready.target.capture_s==10. and worker.poll_ready() is worker.ready
    assert worker.ready.plan.reference.start_s>13.2


def test_trace_archiving_can_expire_a_plan_without_discarding_reachable_piece():
    worker,clock,item,vision,planner,events=fixture()
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    clock.now=12.01;worker.confirm_return(wait_state(worker.clock))
    clock.now=12.9  # Synchronous old-cycle trace persistence.
    worker.update_wait_state(wait_state(clock))
    assert worker.poll_ready() is None and not vision.discarded
    worker._step()
    assert len(planner.calls)==2 and worker.poll_ready().target is item
    assert item.capture_s==10. and item.velocity[0]==.2


def test_retained_expired_target_is_skipped_once_physically_past_region():
    worker,clock,item,vision,_,events=fixture()
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker._step()
    clock.now=12.2;worker._step()
    clock.now=17.;worker._step()
    assert vision.discarded==[item.identity]
    assert events[-1][1]['failure_details']['reason']=='lookahead_target_past_region'
    new=target(3);new.capture_s=12.1;vision.latest=new
    new.capture_s=16.9
    worker.confirm_return(wait_state(worker.clock))
    worker._step()
    assert worker.ready.target is new


def test_failed_model_keeps_failure_details_and_allows_a_new_target():
    worker,_,item,vision,planner,events=fixture()
    def fail(*a,**kw):raise PickRejected('IK failed',details={'joint':6,'stage':'approach'})
    planner.intercept=fail
    worker.confirm_return(wait_state(worker.clock));worker._step()
    assert events[-1][1]['failure_details']=={'joint':6,'stage':'approach'}
    assert not events[-1][1]['model_path_generated']
    vision.latest=target(3);worker._step()
    assert len(vision.discarded)==2


def test_delayed_old_worker_cannot_discard_a_new_owners_pending_target():
    worker,_,old,vision,_,_=fixture()
    new=target(3);worker.begin_cycle((0,2,2));worker.target=new;vision.latest=new
    worker._reject(old,PickRejected('old result'),generated=True)
    assert worker.target is new and vision.latest is new and not vision.discarded


def test_thread_closes_cancels_planning_and_never_owns_a_device():
    worker,_,_,vision,planner,_=fixture();entered=Event()
    def plan(item,state,*,cancelled):
        entered.set()
        while not cancelled():time.sleep(.001)
        raise PickRejected('cancelled')
    planner.intercept=plan
    worker.return_started(identity=(0,1,1),expected_ready_s=12.);worker.start()
    assert entered.wait(2.)
    worker.close()
    assert not worker.thread.is_alive() and vision.enabled


def test_schedule_uses_future_return_as_dispatch_bound_and_recomputes_contact_position():
    args=dict(capture_s=0.,grasp_position=[-.8,0.,.5],velocity=[.2,0.,0.],
        standby_position=[-.4,0.,.7],direction=[1,0,0],now_s=.1,planning_reserve_s=.2,
        advance_s=.09,join_s=1.1,synchronized_s=0.,follow_s=0.,follow_bounds=[-.2,.38])
    ordinary=schedule_forward_join(**args)
    future=schedule_forward_join(**args,earliest_dispatch_s=3.)
    assert future['join_start_s']-.09==pytest.approx(3.)
    assert future['contact_coordinate_m']>ordinary['contact_coordinate_m']
    with pytest.raises(ValueError):schedule_forward_join(**args,earliest_dispatch_s=6.)


def test_real_cycle_loop_adopts_next_plan_only_after_original_executor_confirms_return():
    worker,clock,item,vision,spare,events=fixture()
    first=target(1);vision.latest=first;vision.active=None
    cancelled=Event();config=worker.config;reports=[];executors=[];active=Planner()
    class Executor:
        def __init__(self,config,planner,device,**kw):self.planner=planner
        def run(self,selected):
            if selected is first:
                vision.latest=item
                worker._step()
                self.return_started(identity=first.identity,expected_ready_s=12.)
                worker._step();worker._step()
                assert worker.poll_ready() is None
                clock.now=12.01
            else:
                assert selected is item and self.planner is spare
                assert self.prefetched_intercept.target is item
            return {'status':'cycle_completed_from_feedback','returned_to_wait':True}
    result={'completed_feedback_cycles':0}
    run_pick_cycles(config,active,NS(snapshot=lambda:wait_state(clock)),vision,count=2,cancelled=cancelled,result=result,
        reports=reports,executors=executors,set_phase=lambda *a,**kw:None,
        journal=NS(emit=lambda *a,**kw:None),executor_factory=Executor,clock=clock,lookahead=worker)
    assert result['completed_feedback_cycles']==2 and len(executors)==2
    assert executors[0].planner is active and executors[1].planner is spare


def test_single_exposure_can_finish_during_active_task_and_survives_return(monkeypatch):
    from test_single_exposure_hardware import fixture as camera_fixture,geometry
    from nero_core.two_second import single_exposure_vision as module
    from concurrent.futures import Future
    vision,frame,surface,state=camera_fixture(monkeypatch);detection,grasps=geometry()
    vision.enable_lookahead();vision.active_task=(0,9,9)
    monkeypatch.setattr(module,'frozen_geometry',lambda *a,**kw:
        (detection,np.array([-.64,0,.04]),np.array([.435,0,0]),tuple(grasps),1.))
    try:
        result=vision._prepare_single(frame,surface,NS(),state,10.,0)
        f=Future();f.set_result(result);vision.geometry_future=f;vision._finish_geometry()
        assert vision.poll() is None and vision.poll_pending().state is state
        assert vision.diagnostics()['target_processing_paused'] is False
        pending=vision.poll_pending();vision.resume_targets()
        assert vision.poll() is pending and vision.minimum_capture_s==float('-inf')
        vision.consume(pending.identity)
        assert vision.repeats_recent_target(pending.entry_position+pending.velocity*.03,10.03)
        assert not vision.repeats_recent_target(pending.entry_position+np.array([-.15,0,0]),10.03)
    finally:vision.close()


def test_capture_filter_interpolates_actual_CAN_history_without_movement_or_extrapolation():
    from threading import RLock
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    hardware=Hardware.__new__(Hardware);hardware.lock=RLock()
    history=hardware._joint_feedback_history=JointFeedbackHistory()
    history.observe_packet(tuple(range(7)),[0.]*7,10.,epoch=2)
    history.observe_packet(tuple(range(7)),[.2]*7,10.02,epoch=2)
    actual={'joints':[.2]*7,'width_m':.1}
    hardware.snapshot_nowait=lambda:actual
    result=hardware.snapshot_for_capture(10.01)
    np.testing.assert_allclose(result['joints'],[.1]*7)
    assert actual['joints']==[.2]*7 and not result['capture_joint_pairing']['extrapolated']
    assert hardware.snapshot_for_capture(9.9) is None
    assert hardware.snapshot_for_capture(10.03) is None


def test_skipped_piece_is_not_frozen_again_as_a_new_single_frame_identity(monkeypatch):
    from test_single_exposure_hardware import fixture as camera_fixture
    vision,_,_,_=camera_fixture(monkeypatch)
    skipped=target();vision.enable_lookahead();vision.latest=skipped
    try:
        vision.discard_pending(skipped.identity)
        assert vision.latest is None
        assert vision.repeats_recent_target(skipped.entry_position+skipped.velocity*.03,10.03)
    finally:vision.close()


def test_final_empty_return_announces_future_availability_without_changing_sender_owner(monkeypatch):
    from test_two_second_hardware import executor_fixture
    from nero_core.two_second import streaming
    executor,device,clock=executor_fixture();announcements=[]
    executor.config.raw['lookahead_return_margin_s']=.2
    executor.return_started=lambda **data:announcements.append(data)
    monkeypatch.setattr(streaming,'stream',lambda *a,**kw:'original sender')
    path=NS(q1=np.zeros(7),duration=1.5)
    assert executor._stream(path,origin=10.,epoch=0)=='original sender'
    assert announcements==[{'identity':executor.owner,'expected_ready_s':11.7}]
    executor._stream(path,origin=10.,epoch=0,carrying=True)
    executor._stream(path,origin=10.,epoch=0,reference=NS())
    assert len(announcements)==1 and not device.commands

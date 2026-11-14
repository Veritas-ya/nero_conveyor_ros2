"""Speculative geometry is never treated as an actual attachment certificate."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.carry_lookahead import revalidate_cycle, predict_cycle
from nero_core.two_second.prepared_cycle import prepare_cycle, nominal_state
from nero_core.two_second.runtime_sweep import scene_snapshot
from test_prepared_taught_cycle import fixture
from test_two_second_hardware import executor_fixture


def cpu_work(seconds):
    """Synthetic CPU load in the real warmed planner process, no devices."""
    import os
    import time
    start=time.monotonic();value=1
    while time.monotonic()-start<seconds:
        value=(value*17+31)%997
    return start,time.monotonic(),os.getpid()


def test_private_planner_cpu_load_does_not_block_real_sender():
    import os
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.taught import TaughtSegment
    from nero_core.two_second.carry_lookahead import CarryLookahead
    from nero_core.two_second.runtime_sweep import SweepService
    from test_runtime_sweep_process import executor
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    cfg.raw['taught_segments']={'transfer':TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')}
    carry=CarryLookahead(cfg);geometry=SweepService(cfg)
    try:
        carry.pool.submit(cpu_work,0.).result(timeout=5.)  # warm test module import
        e,dev,path=executor(geometry)
        future=carry.pool.submit(cpu_work,.6)
        e._stream(path,origin=e.clock()+.09,epoch=0)
        start,end,pid=future.result(timeout=5.)
        assert pid!=os.getpid()
        during=[(t,q) for t,q in dev.commands if start<=t<=end]
        assert len(during)>=10
        assert during[-1][1][0]>during[0][1][0]
    finally:
        carry.close();geometry.close()


def candidate():
    cfg,p,q=fixture();p.config=cfg
    state=nominal_state(p.model,q(0.),.04)
    bundle=prepare_cycle(p,cfg,state,q(0.),cancelled=lambda:False)
    result=dict(ok=True,token='current',identity=p.identity,source_version=p.version,
                scene_revision=p.scene_revision,bundle=bundle)
    p.calls=[]
    return p,state,bundle,result


def validate(p,state,bundle,result,**kw):
    return revalidate_cycle(p,state,bundle.carry.q0,result,token='current',
        source_version=p.version,cancelled=kw.get('cancelled',lambda:False))


def test_actual_payload_and_width_rechecked_without_repeating_search():
    p,state,bundle,result=candidate();actual=np.full((2,3),.007)
    p.payload_local=actual;state['width_m']=.06
    seen=[]
    def check(path,width,*,payload,cancelled):
        seen.append((path,width,payload,p.payload_local))
    p.check_path=check
    accepted,report=validate(p,state,bundle,result)
    assert accepted is bundle
    assert seen[0][1:3]==(.06,True) and seen[0][3] is actual
    assert seen[1][2] is False and seen[1][3] is None
    assert p.payload_local is actual and p.released_points is None
    assert report['path_search_reused'] and 'lift' not in p.calls


def test_actual_return_can_be_deferred_to_release_without_skipping_carry_check():
    p,state,bundle,result=candidate();original=p.payload_local
    accepted,evidence=revalidate_cycle(p,state,bundle.carry.q0,result,token='current',
        source_version=p.version,cancelled=lambda:False,defer_return_validation=True)
    assert accepted is bundle
    assert ('check',bundle.carry.stage,True) in p.calls
    assert ('check',bundle.returning.stage,False) not in p.calls
    assert 'opening' in p.calls
    assert 'release_estimate' not in p.calls
    assert p.payload_local is original
    assert evidence['return_validation']=='deferred_until_actual_release'


@pytest.mark.parametrize('key,value', [('token','old'),('identity',(2,3,4)),
    ('source_version',-1),('scene_revision',-1),('ok',False)])
def test_late_result_wrong_task_scene_version_or_token_cannot_be_used(key,value):
    p,state,bundle,result=candidate();result[key]=value
    with pytest.raises(PickRejected,match='失效'):
        validate(p,state,bundle,result)
    assert not p.calls


@pytest.mark.parametrize('reason',['cancel','anchor','feedback','missing_payload','new_collision'])
def test_invalid_actual_state_or_cancel_never_returns_candidate(reason):
    p,state,bundle,result=candidate();original=p.payload_local
    if reason=='anchor':result['bundle']=replace(bundle,carry=replace(bundle.carry,curve=lambda t,d=0:np.ones(7)))
    if reason=='feedback':state['joints'][0]+=.1
    if reason=='missing_payload':p.payload_local=None
    if reason=='new_collision':
        def reject(*a,**kw):raise PickRejected('new collision')
        p.check_path=reject
    with pytest.raises(PickRejected):
        validate(p,state,bundle,result,cancelled=lambda:reason=='cancel')
    if reason=='new_collision':assert p.payload_local is original


def test_predicts_attachment_then_carries_to_brake_endpoint(monkeypatch):
    from test_runtime_sweep_process import fixture as scene_fixture
    cfg,p,path=scene_fixture();snapshot=scene_snapshot(p,path)
    reference=NS(start_s=1.,follow_end_s=.1)
    def prepare(private,config,state,start_joints,*,cancelled):
        np.testing.assert_allclose(start_joints,path.q1)
        np.testing.assert_allclose(state['joints'],path.q1)
        # Transport begins at brake end, not at the earlier contact joint state.
        assert not np.allclose(path.q1,path.position(.1))
        private.model.set_state(path.position(.1),cfg.settings.open_width_m)
        tcp=private.model.tcp()
        world=private.payload_local@tcp[:3,:3].T+tcp[:3,3]
        np.testing.assert_allclose(world,private.target_points_at(1.1))
        return 'predicted'
    monkeypatch.setattr('nero_core.two_second.carry_lookahead.prepare_cycle',prepare)
    assert predict_cycle(p,snapshot,reference,cancelled=lambda:False)=='predicted'


def test_nonrest_grab_endpoint_is_not_spliced_to_stationary_carry():
    from scipy.interpolate import BPoly
    from test_runtime_sweep_process import fixture as scene_fixture
    cfg,p,path=scene_fixture();zero=np.zeros(7)
    curve=BPoly.from_derivatives([0.,path.duration],
        [[path.q0,zero,zero],[path.q1,np.ones(7),zero]])
    path=replace(path,curve=curve)
    with pytest.raises(PickRejected,match='零速度'):
        predict_cycle(p,scene_snapshot(p,path),NS(start_s=1.,follow_end_s=.1),cancelled=lambda:False)


def test_owner_cancellation_cancels_private_work_and_rejects_ready_prediction():
    e,dev,clock=executor_fixture();discard=[]
    e.carry_service=NS(discard=lambda:discard.append(True))
    e.cancel()
    future=Future();future.set_result({'ok':True})
    with pytest.raises(PickRejected):e._prepare(future,e.owner,e.version,0)
    assert discard and not dev.commands


def test_clock_reset_rejects_completed_speculation_before_handoff():
    e,dev,clock=executor_fixture();e.epoch=lambda:1
    future=Future();future.set_result({'ok':True})
    with pytest.raises(PickRejected,match='重置'):e._prepare(future,e.owner,e.version,0)
    assert not dev.commands


def test_prediction_submission_runs_on_supervisor_before_close():
    import time
    from threading import get_ident
    from scipy.interpolate import BPoly
    from nero_core.two_second.trajectory import TimedPath
    from nero_core.two_second.reference import Intercept
    class Clock:
        def __call__(self):return time.monotonic()
        def sleep(self,t):time.sleep(t)
    e,dev,clock=executor_fixture(Clock())
    e.carry_service=object();calls=[];owner=get_ident()
    def submitted(path,reference):
        assert get_ident()==owner and e.sender is not None
        assert not any(c[0]=='grip' for c in dev.commands)
        calls.append(True)
    e.start_carry_prediction=submitted
    zero=np.zeros(7)
    curve=BPoly.from_derivatives([0.,.6],[[zero,zero,zero],[zero,zero,zero]])
    path=TimedPath(curve,e.owner,e.planner.version,'intercept_follow',.6,0.)
    origin=clock()+.15
    ref=Intercept(origin,origin,.2,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
    e._stream(path,origin=origin,epoch=0,reference=ref)
    assert calls and any(c[0]=='grip' for c in dev.commands)


@pytest.mark.parametrize('valid',[True,False,None])
def test_executor_reuses_prediction_or_replans_with_actual_payload(monkeypatch,valid):
    from test_failed_attempt_recovery import ImmediatePool
    p,state,bundle,result=candidate()
    e,dev,clock=executor_fixture()
    e.planner=p;e.config=p.config;e.owner=p.identity;e.rest_anchor=bundle.carry.q0
    e.config.raw['command_advance_s']=.1
    future=Future()
    if valid is not None:future.set_result(result if valid else {**result,'ok':False})
    discarded=[];e.carry_service=NS(discard=lambda:discarded.append(True))
    e.carry_prediction=('current',future,p.version)
    class Hold:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def check(self):pass
    monkeypatch.setattr('nero_core.two_second.stationary_hold.StationaryHold',Hold)
    e.holding.check_holding=lambda state:None
    def stream(path,**kw):return nominal_state(p.model,path.q1,state['width_m'])
    e._stream=stream
    def release(executor,st,*,prepare_return,**kw):
        executor.planner.release_measured(st)
        prepare_return().result()
        return {**st,'width_m':.1,'force_n':0.}
    monkeypatch.setattr('nero_core.two_second.release.release_at_rest',release)
    e._prepared_taught_cycle(ImmediatePool(),state,NS(epoch=0))
    names=[x['event'] for x in e.events]
    if valid:
        assert 'carry_prediction_accepted' in names and 'lift' not in p.calls
    else:
        assert 'carry_prediction_replanned' in names and 'lift' in p.calls
    if valid is None:assert discarded

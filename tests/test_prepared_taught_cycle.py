"""Prepare everything before transport; never cache a payload certificate."""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest

from nero_core.two_second.prepared_cycle import prepare_cycle, concatenate, nominal_state, PreparedCycle, direct_return
from nero_core.two_second.taught import TaughtSegment, prepare_taught_timing, certify_taught
from nero_core.two_second.trajectory import retime_samples, extrema
from nero_core.two_second.taught_timing import limit_jerk
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_failed_attempt_recovery import recovery


def fixture():
    cfg = NS(velocity_limits=np.ones(7)*20, acceleration_limits=np.ones(7)*20,
        settings=NS(control_hz=100., open_width_m=.1, lift_m=.05), standby=np.zeros(7),
        raw={'planning_budget_s':8., 'taught_transfer_timing_policy':'recorded_uniform'})
    def q(x):
        value=np.zeros(7);value[0]=x;return value
    def segment(kind, start, end):
        return TaughtSegment(kind,np.linspace(0.,1.,6),np.linspace(q(start),q(end),6),kind,kind)
    cfg.raw['taught_segments']={'transfer':segment('transfer',.2,.3),'return':segment('return',.3,0.)}
    class Model:
        command_lower=-np.ones(7)*2;command_upper=np.ones(7)*2
        def set_state(self,q,width):self.q=np.asarray(q).copy()
        def check_feedback(self,state):self.set_state(state['joints'],state['width_m'])
        def tcp(self):
            t=np.eye(4);t[:3,3]=self.q[:3];return t
        def flange(self):return self.tcp()
    model=Model();model.set_state(q(0.),.04)
    class Planner:
        def __init__(self):
            self.config=cfg
            self.model=model;self.identity=(0,1,2);self.version=0;self.scene_revision=7
            self.payload_local=np.ones((3,3));self.payload_tree=object();self.capture_tcp=np.eye(4)
            self.released_points=None;self.released_tree=None;self.support=np.array([0.,0.,1.])
            self.calls=[]
        def route(self,state,*,stage,goal_pose=None,goal_joints=None,payload=False,start_joints=None,cancelled):
            self.calls.append(stage);self.version+=1
            end=q(.1) if goal_joints is None else goal_joints
            return retime_samples(np.linspace(0.,1.,6),np.linspace(start_joints,end,6),cfg,
                model.command_lower,model.command_upper,identity=self.identity,version=self.version,stage=stage)
        def check_path(self,path,width,**kw):
            self.calls.append(('check',path.stage,kw.get('payload')))
        def validate_release_opening(self,state):self.calls.append('opening')
        def release_measured(self,state):
            self.calls.append('release_estimate');self.payload_local=None;self.released_points=np.ones((3,3))
    return cfg,Planner(),q


def test_prepared_chain_checks_transport_and_return_and_restores_live_payload():
    cfg,p,q=fixture();original=p.payload_local
    bundle=prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert p.payload_local is original and p.released_points is None
    assert bundle.identity==p.identity and bundle.scene_revision==7
    assert bundle.carry.diagnostics['internal_online_planning'] is False
    assert bundle.carry.diagnostics['internal_arrival_dwell_s']==0.
    assert [x['stage'] for x in bundle.carry.diagnostics['pieces']]==['lift','transfer_connect','taught_transfer']
    # Recorded drop==return start and return end==standby: no artificial holds.
    assert [x['stage'] for x in bundle.returning.diagnostics['pieces']]==['taught_return']
    assert ('check','taught_transfer',True) in p.calls
    assert ('check','taught_return',False) in p.calls
    np.testing.assert_allclose(bundle.carry.q1,cfg.raw['taught_segments']['transfer'].joints[-1],atol=1e-12)
    np.testing.assert_allclose(bundle.returning.q1,cfg.standby,atol=1e-12)


def test_experimental_direct_drop_uses_one_loaded_route_to_taught_endpoint():
    cfg,p,q=fixture()
    cfg.raw.update(experimental_direct_drop_guard={'experiment':True},taught_return_strategy='direct_to_wait')
    bundle=prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert [x['stage'] for x in bundle.carry.diagnostics['pieces']]==['lift','direct_drop']
    assert 'direct_drop' in p.calls and 'transfer_connect' not in p.calls
    assert ('check','taught_transfer',True) not in p.calls
    np.testing.assert_allclose(bundle.carry.q1,cfg.raw['taught_segments']['transfer'].joints[-1],atol=1e-12)
    np.testing.assert_allclose(bundle.returning.q1,cfg.standby,atol=1e-12)


def test_direct_drop_failure_cannot_fall_back_to_taught_path():
    cfg,p,q=fixture();original=p.payload_local
    cfg.raw.update(experimental_direct_drop_guard={'experiment':True},taught_return_strategy='direct_to_wait')
    route=p.route
    def reject(*args, **kwargs):
        if kwargs['stage']=='direct_drop':
            raise PickRejected('直接投放不可行')
        return route(*args, **kwargs)
    p.route=reject
    with pytest.raises(PickRejected,match='直接投放不可行'):
        prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert 'transfer_connect' not in p.calls
    assert p.payload_local is original


def test_direct_return_plans_one_path_to_current_standby_without_old_demonstration():
    cfg,p,q=fixture()
    cfg.standby=q(-.2)
    cfg.raw['taught_return_strategy']='direct_to_wait'
    # The direct policy cannot depend on an obsolete demonstration or its end.
    del cfg.raw['taught_segments']['return']
    bundle=prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert [x['stage'] for x in bundle.returning.diagnostics['pieces']]==['return']
    assert p.calls.count('return')==1 and 'return_connect' not in p.calls
    assert ('check','taught_return',False) not in p.calls
    np.testing.assert_allclose(bundle.returning.q0,cfg.raw['taught_segments']['transfer'].joints[-1],atol=1e-12)
    np.testing.assert_allclose(bundle.returning.q1,cfg.standby,atol=1e-12)


def test_direct_return_failure_is_not_replaced_with_old_route():
    cfg,p,q=fixture();cfg.raw['taught_return_strategy']='direct_to_wait'
    original=p.route
    def reject_return(*args, **kwargs):
        if kwargs['stage']=='return':
            raise PickRejected('直接回位不可行')
        return original(*args,**kwargs)
    p.route=reject_return
    with pytest.raises(PickRejected,match='直接回位不可行'):
        prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert ('check','taught_return',False) not in p.calls


def test_adaptive_taught_entry_certifies_candidates_and_keeps_drop():
    cfg,p,q=fixture();cfg.raw.update(taught_entry_search_enabled=True,taught_entry_indices=[0,1,2],
                                      taught_return_strategy='direct_to_wait')
    bundle=prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    pieces=bundle.carry.diagnostics['pieces']
    selection=pieces[1]['diagnostics']['taught_entry_selection']
    assert len(selection['candidate_evidence'])==3
    assert all(x['status']=='certified' for x in selection['candidate_evidence'])
    chosen=next(x for x in selection['candidate_evidence'] if x['index']==selection['selected_index'])
    assert chosen['rank_score']==min(x['rank_score'] for x in selection['candidate_evidence'])
    assert selection['connection_rest_to_rest'] and selection['not_continuous_nonzero_speed_blending']
    np.testing.assert_allclose(bundle.carry.q1,cfg.raw['taught_segments']['transfer'].joints[-1],atol=1e-12)
    assert p.calls.count(('check','taught_transfer',True))==3


def test_adaptive_taught_entry_skips_failed_candidate_but_propagates_cancellation():
    from nero_core.two_second.taught_entry import prepare_entries, choose_entry
    cfg,p,q=fixture();cfg.raw.update(taught_entry_indices=[0,1,2])
    entries=prepare_entries(cfg,cfg.raw['taught_segments']['transfer'],p.model)
    initial=nominal_state(p.model,q(0.),.04);original=p.route
    def rejected(*args,**kw):
        if np.allclose(kw['goal_joints'],entries[1][1].joints[0]):
            raise PickRejected('candidate blocked')
        return original(*args,**kw)
    p.route=rejected
    connection,transfer=choose_entry(p,cfg,initial,q(0.),entries,cancelled=lambda:False)
    report=connection.diagnostics['taught_entry_selection']
    assert next(x for x in report['candidate_evidence'] if x['index']==1)['status']=='rejected'
    assert report['selected_index'] in (0,2)
    with pytest.raises(PickRejected,match='取消'):
        choose_entry(p,cfg,initial,q(0.),entries,cancelled=lambda:True)


def test_adaptive_entry_prewarm_skips_bad_alternative_but_not_baseline(monkeypatch):
    from nero_core.two_second import taught_entry
    cfg,p,q=fixture();cfg.raw['taught_entry_indices']=[0,1,2]
    original=taught_entry.prepare_taught_timing
    def one_bad(config,segment,model,**kwargs):
        if segment.joints.shape[0]==5:
            raise PickRejected('alternative too slow')
        return original(config,segment,model,**kwargs)
    monkeypatch.setattr(taught_entry,'prepare_taught_timing',one_bad)
    entries=taught_entry.prepare_entries(cfg,cfg.raw['taught_segments']['transfer'],p.model)
    assert [entry[0] for entry in entries]==[0,2]
    assert cfg.raw['_taught_entry_prewarm_rejections'][0]['index']==1
    monkeypatch.setattr(taught_entry,'prepare_taught_timing',lambda *a,**kw: (_ for _ in ()).throw(PickRejected('bad baseline')))
    with pytest.raises(PickRejected,match='baseline'):
        taught_entry.prepare_entries(cfg,cfg.raw['taught_segments']['transfer'],p.model)


def test_direct_return_reuses_only_curve_and_rechecks_each_new_scene():
    cfg,p,q=fixture(); cfg.raw['taught_return_strategy']='direct_to_wait'
    start=q(.3)
    first=direct_return(p,cfg,start,.1,cancelled=lambda:False)
    assert not first.diagnostics['route_template_reused']
    p.scene_revision += 1; p.identity=(0,2,3)
    second=direct_return(p,cfg,start,.1,cancelled=lambda:False)
    assert second.diagnostics['route_template_reused']
    assert second.identity==p.identity
    assert ('check','return',False) in p.calls
    assert p.calls.count('return')==1
    np.testing.assert_allclose(second.q1,cfg.standby,atol=1e-12)
    cfg.standby=q(-.1)
    third=direct_return(p,cfg,start,.1,cancelled=lambda:False)
    assert not third.diagnostics['route_template_reused']
    assert p.calls.count('return')==2
    np.testing.assert_allclose(third.q1,cfg.standby,atol=1e-12)


def test_cached_direct_return_replans_if_new_scene_rejects_candidate():
    cfg,p,q=fixture(); start=q(.3)
    direct_return(p,cfg,start,.1,cancelled=lambda:False)
    def changed(*a,**kw): raise PickRejected('场景已改变')
    p.check_path=changed
    new=direct_return(p,cfg,start,.1,cancelled=lambda:False)
    assert not new.diagnostics['route_template_reused']
    assert p.calls.count('return')==2


def test_cached_direct_return_cancellation_never_falls_back():
    cfg,p,q=fixture(); start=q(.3)
    direct_return(p,cfg,start,.1,cancelled=lambda:False)
    with pytest.raises(PickRejected,match='取消'):
        direct_return(p,cfg,start,.1,cancelled=lambda:True)
    assert p.calls.count('return')==1


def test_timing_cache_reuses_curve_but_checks_new_payload_every_time(monkeypatch):
    cfg,p,q=fixture();segment=cfg.raw['taught_segments']['transfer']
    template,cached=prepare_taught_timing(cfg,segment,p.model,payload=True)
    assert not cached
    def forbidden(*a,**kw):pytest.fail('timing must be precomputed')
    monkeypatch.setattr('nero_core.two_second.taught.retime_samples',forbidden)
    state=nominal_state(p.model,segment.joints[0],.04)
    for owner in ((0,1,3),(0,2,4)):
        p.identity=owner
        curve=certify_taught(p,cfg,segment,state,segment.joints[0],payload=True,cancelled=lambda:False)
        assert curve.identity==owner and curve.diagnostics['timing_precomputed']
    assert p.calls.count(('check','taught_transfer',True))==2
    assert template.identity is None and template.version==0


def test_transfer_only_jerk_override_invalidates_timing_cache():
    cfg,p,q=fixture();segments=cfg.raw['taught_segments']
    cfg.raw['taught_joint_jerk_limit_rad_s3']=500.
    original,_=prepare_taught_timing(cfg,segments['transfer'],p.model,payload=True)
    cfg.raw['taught_transfer_jerk_limit_rad_s3']=10000.
    fast,cached=prepare_taught_timing(cfg,segments['transfer'],p.model,payload=True)
    returning,_=prepare_taught_timing(cfg,segments['return'],p.model,payload=False)
    assert not cached and fast is not original
    assert fast.diagnostics['joint_jerk_limit_rad_s3']==10000.
    assert returning.diagnostics['joint_jerk_limit_rad_s3']==500.
    assert prepare_taught_timing(cfg,segments['transfer'],p.model,payload=True)==(fast,True)


def test_transfer_duration_cap_cannot_reuse_an_overlong_cached_path():
    cfg,p,q=fixture();segment=cfg.raw['taught_segments']['transfer']
    original,_=prepare_taught_timing(cfg,segment,p.model,payload=True)
    cfg.raw['taught_transfer_max_duration_s']=original.duration/2
    for _ in range(2):
        with pytest.raises(PickRejected,match='时长上限') as rejected:
            prepare_taught_timing(cfg,segment,p.model,payload=True)
        assert rejected.value.details['duration_s']==original.duration
        assert len(cfg.raw['_taught_timing_cache'])==1
    # The transfer duration cap must not constrain the independently taught return.
    returning,_=prepare_taught_timing(cfg,cfg.raw['taught_segments']['return'],p.model,payload=False)
    assert returning.duration>cfg.raw['taught_transfer_max_duration_s']
    cfg.raw['taught_transfer_max_duration_s']=original.duration
    accepted,cached=prepare_taught_timing(cfg,segment,p.model,payload=True)
    assert not cached and accepted.diagnostics['maximum_duration_s']==original.duration


@pytest.mark.parametrize('requested', [
    [8.,8.,8.,8.,10.,8.,8.],
    [8.,8.,8.,8.,10.,10.,8.],
])
def test_relaxing_real_transfer_joint_limits_keeps_previous_feasible_timing(requested):
    """Increasing J5/J6 limits must retain the existing feasible 1.48 s path."""
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.planning import calibrated_model
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    model=calibrated_model(cfg)
    segment=TaughtSegment.load(cfg.raw['taught_transfer_file'],cfg,'transfer')
    requested=list(requested)
    cfg.raw['loaded_joint_acceleration_limits_rad_s2']=[8.]*7
    original,_=prepare_taught_timing(cfg,segment,model,payload=True)
    cfg.raw['loaded_joint_acceleration_limits_rad_s2']=requested
    updated,cached=prepare_taught_timing(cfg,segment,model,payload=True)
    assert not cached
    assert updated.duration<=original.duration<=cfg.raw['taught_transfer_max_duration_s']
    np.testing.assert_allclose(updated.curve.x,original.curve.x,atol=1e-12)
    np.testing.assert_allclose(updated.curve.c,original.curve.c,atol=1e-10)
    assert updated.diagnostics['timing_limit_search']['selected']=='balanced_lower_acceleration'
    assert cfg.raw['loaded_joint_acceleration_limits_rad_s2']==requested
    for derivative,limits in ((1,cfg.velocity_limits),(2,np.asarray(requested))):
        assert np.all(abs(np.asarray(extrema(updated.curve,derivative)))<=limits+1e-8)
    assert np.max(abs(np.asarray(extrema(updated.curve,3))))<=10000.+1e-8
    assert prepare_taught_timing(cfg,segment,model,payload=True)==(updated,True)


def test_prepare_failure_restores_payload_and_never_returns_partial_chain():
    cfg,p,q=fixture();original=p.payload_local
    def rejected(*a,**kw):raise PickRejected('返回碰撞')
    p.check_path=rejected
    with pytest.raises(PickRejected):
        prepare_cycle(p,cfg,nominal_state(p.model,q(0.),.04),q(0.),cancelled=lambda:False)
    assert p.payload_local is original and p.released_points is None


def test_concatenation_rejects_a_discontinuous_join():
    cfg,p,q=fixture()
    args=dict(identity=p.identity,version=1,stage='part')
    a=retime_samples(np.linspace(0.,1.,6),np.linspace(q(0.),q(.1),6),cfg,-np.ones(7),np.ones(7),**args)
    b=retime_samples(np.linspace(0.,1.,6),np.linspace(q(.2),q(.3),6),cfg,-np.ones(7),np.ones(7),**args)
    with pytest.raises(PickRejected,match='不连续'):
        concatenate([a,b],**args)


def test_jerk_bound_preserves_geometry_and_limits_the_full_polynomial():
    cfg,p,q=fixture()
    path=retime_samples(np.linspace(0.,1.,6),np.linspace(q(0.),q(.8),6),cfg,-np.ones(7),np.ones(7),
        identity=p.identity,version=1,stage='taught_transfer')
    result=limit_jerk(path,20.,100.)
    assert np.max(abs(np.asarray(extrema(result.curve,3))))<=20.+1e-8
    t=np.linspace(0.,path.duration,100)
    np.testing.assert_allclose(result.curve(t*result.duration/path.duration),path.curve(t),atol=1e-11)


def test_executor_uses_prepared_paths_and_checks_actual_release_before_return(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from test_two_second_hardware import executor_fixture
    from nero_core.two_second import prepared_cycle,stationary_hold,release
    e,device,clock=executor_fixture();cfg,p,q=fixture();order=[]
    segment=cfg.raw['taught_segments']['transfer'];e.config.raw['taught_segments']={'transfer':segment}
    e.planner.scene_revision=7;e.rest_anchor=q(0.);e.holding.held_width=.04
    device.width=.04;device.force=2.
    carry=retime_samples(np.linspace(0.,1.,6),np.linspace(q(0.),q(.3),6),cfg,-np.ones(7),np.ones(7),
        identity=e.owner,version=1,stage='prepared_taught_carry')
    returning=replace(carry,stage='prepared_taught_return')
    bundle=PreparedCycle(e.owner,7,replace(carry,diagnostics={'pieces':[]}),replace(returning,diagnostics={'pieces':[]}))
    monkeypatch.setattr(prepared_cycle,'prepare_cycle',lambda *a,**kw:order.append('prepare_all') or bundle)
    class Hold:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def check(self):pass
    monkeypatch.setattr(stationary_hold,'StationaryHold',Hold);monkeypatch.setattr(release,'StationaryHold',Hold)
    e.planner.validate_release_opening=lambda s:order.append('opening_checked')
    e.planner.release_measured=lambda s:order.append('actual_release_estimate')
    e.planner.check_path=lambda *a,**kw:order.append('check_actual_return')
    def stream(path,**kw):
        order.append(path.stage);device.q=path.q1.copy();e.rest_anchor=path.q1.copy();return device.snapshot()
    monkeypatch.setattr(e,'_stream',stream)
    with ThreadPoolExecutor(max_workers=1) as pool:
        e._prepared_taught_cycle(pool,device.snapshot(),NS(epoch=0))
    assert order.index('prepare_all')<order.index('prepared_taught_carry')
    assert order.index('actual_release_estimate')<order.index('check_actual_return')<order.index('prepared_taught_return')
    assert order.count('prepared_taught_carry')==1
    assert [s for s in order if s.startswith('prepared_taught_')]==[
        'prepared_taught_carry','prepared_taught_return']


def test_held_recovery_uses_taught_chain_instead_of_other_bin_route(recovery, monkeypatch):
    r=recovery;r.device.width=.04;r.device.force=2.
    r.e.holding.held_width=.04;r.e.planner.payload_local=np.zeros((2,3))
    r.e.config.raw.update(prepared_taught_cycle_enabled=True,taught_playback_enabled=True,
        taught_segments={'transfer':object(),'return':object()})
    selected=[]
    def taught(pool,state,target,*,recovery):
        selected.append(recovery)
        r.device.width=.1;r.device.force=0.;r.device.q=r.e.config.standby.copy()
        r.e.rest_anchor=r.device.q.copy()
        r.e.holding.held_width=None;r.e.planner.payload_local=None
        return r.device.snapshot()
    monkeypatch.setattr(r.e,'_prepared_taught_cycle',taught)
    result=r.e.run(r.target)
    assert selected==[True] and result['carried_to_bin']
    assert not [x for x in r.calls if x[0]=='route']
    assert any(row['event']=='recovery_started' and row['data']['payload_policy']=='taught_drop_then_wait' for row in r.e.events)

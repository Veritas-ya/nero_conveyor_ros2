"""Opt-in repair: actual polynomial/space gates, fake IK/devices, no hardware."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second import delivery_repair as repair
from nero_core.two_second.phase_timing import make_reference, require_reference, verify_plan
from nero_core.two_second.planning import InterceptPlan
from nero_core.two_second.trajectory import certify_curve, sampling_times, TimedPath
from test_phase_timing import fixture as phase_fixture
from test_two_second_hardware import site, Clock, executor_fixture, cloud


def core(monkeypatch):
    cfg, target, start, grasp = phase_fixture()
    cfg.raw.update(delivery_repair={**repair.DEFAULTS, 'enabled':True},
        planning_knot_hz=20., planning_knot_hz_fallback=30.)
    cfg.velocity_limits[:]=2.; cfg.acceleration_limits[:]=10.
    target.identity=(0,1,2)
    clock=Clock(); clock.now=.4
    state=dict(joints=np.zeros(7), width_m=cfg.settings.open_width_m, force_n=0.)
    class Model:
        command_lower=-np.ones(7)*2; command_upper=np.ones(7)*2
        def check_feedback(self, state): self.q=np.array(state['joints'])
        def tcp(self):
            tcp=start.copy(); tcp[:3,3]+=self.q[:3]; return tcp
    model=Model(); model.check_feedback(state)
    ref=make_reference(cfg,target,start,grasp,.9,.4,.6,now_s=.1,reservation_s=.2)
    times=sampling_times(ref.full_duration_s,.05,ref.prealign_s,ref.follow_end_s)
    def joints(xyz): return np.r_[np.asarray(xyz)-start[:3,3],np.zeros(4)]
    path=certify_curve(times,[joints(ref.sample(t)[0]) for t in times],cfg,
        model.command_lower,model.command_upper,identity=target.identity,version=1,
        stage='smooth_intercept_follow_brake',zero_end_acceleration=True)
    path=replace(path,diagnostics={'phase_timing':require_reference(cfg,ref)})
    old=InterceptPlan(path,ref,target,1,grasp,())
    clock.now=ref.start_s-cfg.raw['command_advance_s']+.001
    calls=[]
    class Solver:
        def __init__(self, model): pass
        def solve(self, xyz, quat, seed, *, cancelled):
            assert not cancelled()
            calls.append((xyz.copy(),seed.copy()))
            return NS(success=True,joint_positions=joints(xyz))
    monkeypatch.setattr(repair,'PrecisePoseSolver',Solver)
    planner=NS(config=cfg,model=model,clock=clock,target=target,identity=target.identity,
        version=1,scene_revision=1,check_smooth_intercept=lambda *a,**kw:{'toy_geometry':True})
    return planner,old,state,clock,calls


def test_repair_reuses_scene_and_branch_but_certifies_new_curve_from_measured_start(monkeypatch):
    p,old,state,clock,calls=core(monkeypatch)
    coefficients=old.path.curve.c.copy()
    state['joints'][0]=.001
    new=repair.repair_intercept(p,old,state,deadline=clock()+1.)
    verify_plan(p.config,new.path,new.reference)
    assert new.target is old.target and new.scene_revision==old.scene_revision
    assert new.path.version==2 and old.path.version==1
    np.testing.assert_array_equal(old.path.curve.c,coefficients)
    np.testing.assert_allclose(new.path.q0,state['joints'],atol=1e-12)
    shift=new.reference.start_s-old.reference.start_s
    np.testing.assert_allclose(new.reference.contact_xyz-old.reference.contact_xyz,
        old.target.velocity*shift)
    assert calls and new.path.diagnostics['delivery_repair']['send_margin_s'] >= .02
    assert new.timings_s['scene_prepare']==0. and new.timings_s['candidate_selection']==0.
    for t in (0.,new.path.duration):
        np.testing.assert_allclose(new.path.curve(t,1),0.,atol=1e-10)
        np.testing.assert_allclose(new.path.curve(t,2),0.,atol=1e-9)


@pytest.mark.parametrize('mutation',['target','identity','version','scene','capture','speed','moved','loaded','closed','cancel'])
def test_invalidated_old_result_never_reaches_ik(monkeypatch,mutation):
    p,old,state,clock,calls=core(monkeypatch)
    if mutation=='target':p.target=deepcopy(old.target)
    if mutation=='identity':p.identity=(0,1,99)
    if mutation=='version':p.version+=1
    if mutation=='scene':p.scene_revision+=1
    if mutation=='capture':old.target.capture_s+=.001
    if mutation=='speed':old.target.velocity=old.target.velocity+.01
    if mutation=='moved':state['joints'][0]=.006
    if mutation=='loaded':state['force_n']=100.
    if mutation=='closed':state['width_m']=0.
    with pytest.raises(PickRejected):
        repair.repair_intercept(p,old,state,deadline=clock()+1.,cancelled=lambda:mutation=='cancel')
    assert not calls


def test_spatial_rejection_precedes_ik(monkeypatch):
    p,old,state,clock,calls=core(monkeypatch)
    with pytest.raises(PickRejected):
        repair.repair_intercept(p,old,state,deadline=4.,new_start_s=3.)
    assert not calls and p.version==1


def test_joint_acceleration_is_not_relaxed(monkeypatch):
    p,old,state,clock,calls=core(monkeypatch)
    # Keep the old binding valid under the test's smaller limit, then demand a
    # real acceleration check on the shifted spline.
    p.config.acceleration_limits[:]=.01
    old=replace(old,path=replace(old.path,diagnostics={'phase_timing':require_reference(p.config,old.reference)}))
    with pytest.raises(PickRejected,match='约束') as error:
        repair.repair_intercept(p,old,state,deadline=clock()+1.)
    assert calls and all(f['failed_phase']=='polynomial_limits' for f in error.value.details['failures'])


@pytest.mark.parametrize('phase',['ik','geometry'])
def test_computation_overrun_never_returns_late_curve_and_updates_estimate(monkeypatch,phase):
    p,old,state,clock,calls=core(monkeypatch)
    if phase=='ik':
        class SlowSolver:
            def __init__(self,model):pass
            def solve(self,*args,cancelled):
                clock.sleep(.15)
                assert cancelled()
                raise PickRejected('injected cancelled IK')
        monkeypatch.setattr(repair,'PrecisePoseSolver',SlowSolver)
    else:
        def slow(*args,cancelled):clock.sleep(.15);assert cancelled();return {}
        p.check_smooth_intercept=slow
    with pytest.raises(PickRejected,match='过期'):
        repair.repair_intercept(p,old,state,deadline=clock()+1.)
    assert p.repair_compute_s >= .15-1e-12


@pytest.mark.parametrize('spec',[{'enabled':'true'},{'max_attempts':3},{'send_guard_s':.001},
    {'initial_compute_s':float('nan')},{'unknown':1}])
def test_bad_repair_configuration_rejected(spec):
    with pytest.raises(ValueError):repair.validate({'delivery_repair':spec})


@pytest.mark.parametrize('change',[
    {'hardware':{'joint_command_mode':'j'}},{'hover_descent':{'enabled':True}},
    {'timing_experiment':{}},{'execution_delay_window':{}},
    {'grasp_workpiece_collision_check_enabled':False}])
def test_incompatible_modes_rejected(change):
    raw=dict(trajectory_profile='smooth',intercept_strategy='synchronized',phase_timing={},
        hardware={'joint_command_mode':'js'},delivery_repair={'enabled':True})
    raw.update(change)
    with pytest.raises(ValueError):repair.validate(raw)


def test_flag_private_config_recording_and_complete_cycle_arguments(site,tmp_path,monkeypatch):
    from nero_core.two_second.phase_timing_test import main
    from nero_core.two_second.config import CellConfig
    path,raw=site
    raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',
        closure_policy='scheduled_attempt',taught_playback_enabled=False,
        prealign_orientation_s=1.1,brake_duration_s=.9)
    raw['hardware']['joint_command_mode']='js'
    path.write_text(yaml.safe_dump(raw)); before=path.read_bytes()
    assert not repair.enabled(CellConfig.load(path))
    seen=[]
    def hardware(argv,**kwargs):
        cfg=CellConfig.load(argv[argv.index('--config')+1]);seen.append(argv)
        assert repair.enabled(cfg) and cfg.raw['grasp_workpiece_collision_check_enabled']
        assert '--record-rgbd' in argv and '--continuous' in argv and '--execute' in argv
        for field in ('workspace_base_m','standby_joint_rad','joint_acceleration_limits_rad_s2'):
            assert cfg.raw[field]==raw[field]
        return 0
    monkeypatch.setattr('nero_core.two_second.main.main',hardware)
    assert main(['--config',str(path),'--output',str(tmp_path/'run'),'--belt-speed-cm-s','10',
        '--enable-delivery-repair','--execute','--continuous'])==0
    assert seen and path.read_bytes()==before


class ImmediatePool:
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def submit(self,fn,*args,**kwargs):
        f=Future()
        try:f.set_result(fn(*args,**kwargs))
        except BaseException as e:f.set_exception(e)
        return f


def full_cycle(monkeypatch,*,prefetched=False,failure=None):
    from nero_core.two_second.reference import Intercept
    from nero_core.two_second.vision import MeasuredTarget,obb_grasps
    e,device,clock=executor_fixture();events=[];stages=[];calls=[];repairs=[]
    e.event_sink=events.append
    e.planner.config=e.config
    e.config.raw['obb_grasp_orientation_reference']='current_tcp'
    e.config.raw.update(delivery_repair={**repair.DEFAULTS,'enabled':True},planning_budget_s=1.)
    points=cloud();poses,box=obb_grasps(points,np.eye(4),np.eye(4));identity=(0,1,2)
    target=MeasuredTarget(identity,clock(),clock(),100.,0,1,np.zeros(3),np.zeros(3),tuple(poses),points,
        {'belt_region':{'plane_to_base':np.eye(4)},'obb':box},{},{'source':'nero7_xy_kalman'})
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor',ImmediatePool)
    def plan(lead):
        ref=Intercept(target.capture_s,clock()+lead,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
        return InterceptPlan(TimedPath(lambda t,d=0:np.zeros(7),identity,e.planner.version,
            'intercept_follow',2.6,0.),ref,target,1,target.grasps[1 if failure=='rank_switch' else 0],())
    def intercept(target,state,**kw):
        assert kw.get('recovery_replan', False) == (bool(repairs) or (failure=='rank_switch' and bool(calls)))
        calls.append(kw['planning_deadline']);e.planner.version+=1
        return plan(.15 if len(calls)==1 else .3)
    e.planner.intercept=intercept
    e.planner.identity=identity;e.planner.scene_revision=1;e.epoch=lambda:0
    if prefetched:
        from nero_core.two_second.return_lookahead import PreparedIntercept
        old=plan(.15)
        e.prefetched_intercept=PreparedIntercept(target,e.planner,old,device.snapshot(),(0,0,0),clock())
        clock.sleep(.2)
    def prepare(*a,**kw):device.commands.append(('prepare',clock()));clock.sleep(.06)
    device.prepare_motion=prepare
    def repaired(p,old,state,**kw):
        repairs.append(kw['deadline']);clock.sleep(.03)
        if failure=='reject':raise PickRejected('injected space rejection')
        if failure=='cancel':e.cancelled.set();raise PickRejected('cancelled')
        if failure=='budget':clock.sleep(1.);raise PickRejected('budget exhausted')
        p.version+=1
        new=plan(.05 if failure=='late' else .3)
        return replace(new,path=replace(new.path,diagnostics={'delivery_repair':{
            'completed_monotonic_s':clock(),'wall_s':.03},'phase_timing':{},'peak_acceleration_rad_s2':[0.]*7}))
    monkeypatch.setattr(repair,'repair_intercept',repaired)
    # This executor fixture uses a synthetic Intercept rather than a physical
    # synchronized curve. The real certificate is exercised by core() tests.
    monkeypatch.setattr(repair, 'verify_repaired_plan', lambda *args: {})
    def route(state,*,stage,**kw):
        e.planner.version+=1
        return TimedPath(lambda t,d=0:np.zeros(7),identity,e.planner.version,stage,1.,0.)
    e.planner.route=route
    e.planner.attach_measured=lambda *a:None
    e.planner.validate_release_opening=lambda *a:None
    e.planner.release_measured=lambda *a:None
    def stream(path,**kw):
        stages.append(path.stage)
        if path.stage=='intercept_follow':
            assert kw['origin']-e.config.raw['command_advance_s']>clock()
            assert bool(repairs) or failure=='rank_switch'
            device.width=.04;device.force=2.;e.holding.held_width=.04
        return device.snapshot()
    monkeypatch.setattr(e,'_stream',stream)
    return e,target,events,stages,calls,repairs


@pytest.mark.parametrize('prefetched',[False,True])
@pytest.mark.parametrize('failure',[None,'reject'])
def test_expired_delivery_repairs_or_replans_then_completes_grasp_carry_release_return(monkeypatch,prefetched,failure):
    e,target,events,stages,calls,repairs=full_cycle(monkeypatch,prefetched=prefetched,failure=failure)
    # A fallback from a prefetched plan should be immediately feasible.
    if prefetched and failure=='reject':
        original=e.planner.intercept
        def fallback(*a,**kw):
            plan=original(*a,**kw)
            return replace(plan,reference=replace(plan.reference,start_s=e.clock()+.3))
        e.planner.intercept=fallback
    result=e._run_attempt(target)
    assert stages==['intercept_follow','lift','transfer','return']
    assert result['status']=='cycle_completed_from_feedback' and result['physical_grasp_success'] is None
    assert len(calls)==(0 if prefetched else 1)+(failure=='reject')
    assert len(repairs)==1 and len(set(calls+repairs))==1
    names=[row['event'] for row in events]
    assert 'delivery_repair_submitted' in names
    assert ('delivery_repair_fallback_replan' if failure else 'delivery_repair_ready') in names


def test_ranked_recovery_replans_without_first_attempting_old_pose(monkeypatch):
    e,target,events,stages,calls,repairs=full_cycle(monkeypatch,failure='rank_switch')
    result=e._run_attempt(target)
    assert result['status']=='cycle_completed_from_feedback'
    assert not repairs and len(calls)==2 and len(set(calls))==1
    assert any(row['event']=='delivery_recovery_branch_selected' and
               row['data']['branch']=='replan' for row in events)


@pytest.mark.parametrize('failure',['cancel','budget'])
def test_cancel_or_exhausted_budget_after_failed_repair_never_replans_or_streams(monkeypatch,failure):
    e,target,events,stages,calls,repairs=full_cycle(monkeypatch,failure=failure)
    with pytest.raises(PickRejected):e._run_attempt(target)
    assert not stages and len(calls)==len(repairs)==1


def test_repeated_late_repairs_are_bounded_then_use_one_remaining_budget(monkeypatch):
    e,target,events,stages,calls,repairs=full_cycle(monkeypatch,failure='late')
    result=e._run_attempt(target)
    assert result['status']=='cycle_completed_from_feedback'
    assert len(repairs)==2 and len(calls)==2 and len(set(calls+repairs))==1
    assert stages.count('intercept_follow')==1
    assert any(row['event']=='delivery_repair_limit_reached' for row in events)


@pytest.mark.parametrize('change',['stale','moved'])
def test_changed_feedback_between_expiry_and_repair_never_reaches_solver_or_stream(monkeypatch,change):
    e,target,events,stages,calls,repairs=full_cycle(monkeypatch)
    snapshot=e.device.snapshot
    def changed():
        state=snapshot()
        if any(command[0]=='prepare' for command in e.device.commands):
            if change=='stale':state['joint_stamp']-=.5
            else:state['joints'][0]+=.01
        return state
    e.device.snapshot=changed
    with pytest.raises(PickRejected):e._run_attempt(target)
    assert not repairs and not stages


def test_continuous_expired_prefetch_can_only_be_claimed_for_opted_in_repair():
    from test_return_lookahead import fixture,wait_state,Planner
    worker,clock,target,vision,planner,events=fixture()
    worker.config.raw['delivery_repair']={**repair.DEFAULTS,'enabled':True}
    worker._step();item=worker.ready
    clock.now=item.plan.reference.start_s+1.
    worker.confirm_return(wait_state(clock));worker._step()
    assert worker.ready is item and len(planner.calls)==1
    with pytest.raises(PickRejected):item.validate(worker.config,0,clock())
    assert worker.claim(worker.poll_ready(),Planner()) is item
    # Opt-in cannot override scene or target validity.
    planner.scene_revision+=1
    with pytest.raises(PickRejected):item.validate(worker.config,0,clock(),allow_expired_for_repair=True)


def test_archive_contains_old_and_repaired_coefficients(monkeypatch,tmp_path):
    from nero_core.two_second.experiment_archive import ExperimentArchive
    p,old,state,clock,calls=core(monkeypatch)
    archive=ExperimentArchive(tmp_path,lambda *a,**kw:None);p.experiment_archive=archive
    e=NS(planner=p,owner=old.target.identity,_emit=lambda *a:None)
    repair.record(e,'delivery_repair_submitted',{'start_state':state},old)
    new=repair.repair_intercept(p,old,state,deadline=clock()+1.)
    repair.record(e,'delivery_repair_ready',{},new)
    archive.close()
    files=list((tmp_path/'experiment').rglob('*.npz'))
    assert len(files)==2
    for file in files:
        with np.load(file,allow_pickle=False) as data:
            assert {'curve_c','curve_x','selected_grasp'} <= set(data.files)


def test_full_replan_cannot_restart_total_budget():
    from nero_core.two_second.adaptive_planning import plan_with_measured_timing
    clock=Clock();p=NS(clock=clock,config=NS(raw={'planning_budget_s':8.}))
    calls=[]
    def late(*args,**kwargs):
        calls.append(kwargs['global_deadline']);clock.sleep(.11)
        raise PickRejected('late',details={'failures':[{'details':{'reason':'measured_deadline_missed'}}]})
    with pytest.raises(PickRejected):
        plan_with_measured_timing(p,None,None,cancelled=lambda:False,planning_fn=late,global_deadline=1.1)
    assert calls==[1.1]


def fixed_core(monkeypatch):
    p, old, state, clock, calls = core(monkeypatch)
    p.config.raw.pop('phase_timing')
    p.config.raw.update(prealign_orientation_s=.9, synchronized_join_candidates_s=[.9],
                        brake_duration_s=.6)
    p.config.settings = replace(p.config.settings, full_plan_close_window_s=.4)
    return p, old, state, clock, calls


def test_fixed_timing_repair_keeps_schedule_parameters_and_rechecks_certificate(monkeypatch):
    p, old, state, clock, calls = fixed_core(monkeypatch)
    new = repair.repair_intercept(p, old, state, deadline=clock()+1.)
    assert 'phase_timing' not in p.config.raw and 'phase_timing' not in new.path.diagnostics
    assert calls and new.reference.start_s > old.reference.start_s
    assert (new.reference.prealign_s, new.reference.follow_s, new.reference.brake_s) == (.9, .4, .6)
    report = repair.verify_repaired_plan(p.config, new.path, new.reference)
    assert min(report['margins'].values()) >= 0
    with pytest.raises(PickRejected, match='认证'):
        repair.verify_repaired_plan(p.config, new.path, replace(new.reference, start_s=new.reference.start_s+.001))


@pytest.mark.parametrize('change', ['stop', 'workspace', 'duration'])
def test_fixed_repair_outside_bounds_or_changed_durations_rejected_before_ik(monkeypatch, change):
    p, old, state, clock, calls = fixed_core(monkeypatch)
    if change == 'stop':
        p.config.follow_bounds[1] = .45
    elif change == 'workspace':
        p.config.workspace[2, 1] = .05
    else:
        p.config.raw['brake_duration_s'] = .5
    with pytest.raises(PickRejected):
        repair.repair_intercept(p, old, state, deadline=clock()+1.)
    assert not calls


def test_regular_single_frame_cli_enables_repair_without_phase_experiment(site, capsys):
    import json
    from nero_core.two_second.fixed_freeze_pick import main
    path, raw = site
    raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        closure_policy='scheduled_attempt', prealign_orientation_s=1.1, brake_duration_s=.5,
        freeze_position_upstream_m=.7, maximum_measured_speed_m_s=1.,
        grasp_workpiece_collision_check_enabled=False)
    raw['hardware']['joint_command_mode'] = 'js'
    raw['pick_settings'] = {'full_plan_close_window_s': 0.}
    path.write_text(yaml.safe_dump(raw)); original = path.read_bytes()
    args = ['--config', str(path), '--check-config', '--belt-speed-cm-s', '53.4']
    assert main(args) == 0
    before = json.loads(capsys.readouterr().out)
    assert main(args+['--enable-delivery-repair']) == 0
    after = json.loads(capsys.readouterr().out)
    assert not before['delivery_repair']['enabled'] and after['delivery_repair']['enabled']
    assert after['phase_timing'] is None and not after.get('phase_timing_hardware_test', False)
    assert after['grasp_workpiece_collision_check_enabled']
    for field in ('prealign_orientation_s', 'follow_duration_s', 'brake_duration_s',
                  'joint_velocity_limits_rad_s', 'joint_acceleration_limits_rad_s2'):
        assert before[field] == after[field]
    assert after['hardware_connections'] == after['commands_sent'] == 0
    assert path.read_bytes() == original

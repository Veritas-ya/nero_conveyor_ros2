"""Candidate order/deadline tests and device-free independent-entry validation."""
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import builtins
import json
import numpy as np
import pytest

from test_two_second_smoothing import fixture as planner_fixture
from test_two_second_hardware import Clock, site
from nero_core.two_second import smooth_planning
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def fixture(monkeypatch, *, policy='coarse_first', reject=(), geometry=None, advance=None):
    planner, model, target = planner_fixture(monkeypatch)
    clock=Clock();planner.clock=clock
    cfg=planner.config
    cfg.settings=replace(cfg.settings,full_plan_close_window_s=0.)
    cfg.raw.update(trajectory_profile='smooth',intercept_strategy='synchronized',
        synchronized_duration_s=0.,prealign_orientation_s=1.1,
        synchronized_join_candidates_s=[1.1],brake_duration_s=.9,
        candidate_planning_budget_s=3.,planning_knot_hz=10.,planning_knot_hz_fallback=50.)
    if policy is not None:cfg.raw['candidate_retry_policy']=policy
    tried=[];installed=[];checked=[]
    class Solver:
        def __init__(self,model):pass
        def solve(self,*a,**kw):
            if kw['cancelled']():raise PickRejected('cancelled test IK')
            return SimpleNamespace(success=True,joint_positions=np.zeros(7))
    monkeypatch.setattr(smooth_planning,'PrecisePoseSolver',Solver)
    original=smooth_planning.certify_curve
    def certify(times,*args,**kwargs):
        index=planner.progress['candidate_index']
        hz=int(round(1./np.median(np.diff(times))))
        key=(index,hz);tried.append(key)
        if advance:advance(key,clock)
        if key in reject:
            raise PickRejected('整条曲线超出关节位置/速度/加速度约束',details={'derivative':2})
        return original(times,*args,**kwargs)
    monkeypatch.setattr(smooth_planning,'certify_curve',certify)
    def install(*args):
        installed.append(target.identity);planner.version+=1
    monkeypatch.setattr(planner,'install',install)
    def check(path,ref,**kwargs):
        checked.append(tried[-1])
        if geometry:geometry(tried[-1],ref,clock)
        return {'samples':52,'max_position_error_m':0.}
    monkeypatch.setattr(planner,'check_smooth_intercept',check)
    return planner,target,clock,tried,installed,checked


def run(planner,target,**kwargs):
    return planner.intercept(target,{'joints':np.zeros(7),'width_m':.1},**kwargs)


def test_other_coarse_grasp_precedes_dense_retry_and_keeps_scene_checks(monkeypatch):
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(0,10)})
    result=run(p,t)
    assert tried==[(0,10),(1,10)]
    assert installed==[t.identity] and checked==[(1,10)]
    assert result.path.diagnostics['candidate_retry_policy']=='coarse_first'
    assert result.path.diagnostics['selected_input_grasp_index']==1
    assert result.failures[0]['candidate_retry_policy']=='coarse_first'


@pytest.mark.parametrize('policy',[None,'per_grasp'])
def test_legacy_policy_still_densifies_first_grasp_before_next(monkeypatch,policy):
    p,t,c,tried,installed,checked=fixture(monkeypatch,policy=policy,reject={(0,10)})
    result=run(p,t)
    assert tried==[(0,10),(0,50)] and checked==[(0,50)]
    assert result.path.diagnostics['candidate_retry_policy']=='per_grasp'


def test_first_valid_coarse_candidate_returns_without_extra_work(monkeypatch):
    p,t,c,tried,installed,checked=fixture(monkeypatch)
    run(p,t)
    assert tried==[(0,10)] and checked==[(0,10)]


def test_all_coarse_candidates_precede_dense_and_each_gets_fresh_schedule(monkeypatch):
    p,t,c,tried,installed,checked=fixture(monkeypatch,
        reject={(0,10),(1,10),(0,50)},advance=lambda key,clock:clock.sleep(.01))
    calls=[];original=smooth_planning.schedule_forward_join
    def schedule(**kw):
        calls.append(kw['now_s']);return original(**kw)
    monkeypatch.setattr(smooth_planning,'schedule_forward_join',schedule)
    result=run(p,t)
    assert tried==[(0,10),(1,10),(0,50),(1,50)]
    assert calls==pytest.approx([1.,1.01,1.02,1.03])
    assert result.path.diagnostics['ik_knot_hz']==50.
    assert installed==[t.identity] and checked==[(1,50)]


def test_actual_collision_is_not_enqueued_for_density_retry(monkeypatch):
    def collision(key,*args):
        if key==(0,10):raise PickRejected('机器人与实测工件点云碰撞')
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(1,10)},geometry=collision)
    run(p,t)
    assert tried==[(0,10),(1,10),(1,50)]
    assert installed==[t.identity] and checked==[(0,10),(1,50)]


def test_reference_interpolation_error_can_defer_density_retry(monkeypatch):
    def deviation(key,*args):
        if key==(0,10):raise PickRejected('关节曲线偏离指定 TCP 参考/方向')
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(1,10)},geometry=deviation)
    run(p,t)
    assert tried==[(0,10),(1,10),(0,50)]
    assert len(installed)==1 and checked==[(0,10),(0,50)]


def test_workspace_reject_is_not_retried_at_dense_rate(monkeypatch):
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(1,10)})
    # A has the same along-belt coordinate as B but is outside the Y workspace.
    t.grasps[0][1,3]=2.
    result=run(p,t)
    assert tried==[(1,10),(1,50)]
    assert result.failures[0]['failed_phase']=='bounds'


def test_deferred_dense_retry_does_not_restart_original_private_budget(monkeypatch):
    def advance(key,clock):
        if key==(0,10):clock.sleep(.02)
        elif key==(1,10):clock.sleep(.09)
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(0,10),(1,10)},advance=advance)
    p.config.raw['candidate_planning_budget_s']=.1
    result=run(p,t)
    assert tried==[(0,10),(1,10),(1,50)]
    skipped=[f for f in result.failures if f['failed_phase']=='deferred_retry_budget']
    assert len(skipped)==1 and skipped[0]['candidate_index']==0
    assert skipped[0]['stop_reason']=='candidate_budget'


def test_shared_scene_preparation_does_not_consume_deferred_private_budget(monkeypatch):
    def deviation(key,*args):
        if key==(1,10):raise PickRejected('关节曲线偏离指定 TCP 参考/方向')
    def advance(key,clock):
        if key==(0,10):clock.sleep(.02)
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(0,10)},
        geometry=deviation,advance=advance)
    p.config.raw['candidate_planning_budget_s']=.1
    install=p.install
    def prepare(*args):
        c.sleep(.09);install(*args)
    monkeypatch.setattr(p,'install',prepare)
    result=run(p,t)
    assert tried==[(0,10),(1,10),(0,50)]
    assert checked==[(1,10),(0,50)] and len(installed)==1
    assert not any(f['failed_phase']=='deferred_retry_budget' for f in result.failures)


def test_cancel_after_coarse_rejection_prevents_next_candidate(monkeypatch):
    stopped=Event()
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(0,10)},
        advance=lambda key,clock:stopped.set())
    with pytest.raises(PickRejected) as failed:run(p,t,cancelled=stopped.is_set)
    assert tried==[(0,10)] and not installed and not checked
    assert failed.value.details['stop_reason']=='cancelled'


def test_global_budget_is_not_reset_between_candidates(monkeypatch):
    p,t,c,tried,installed,checked=fixture(monkeypatch,reject={(0,10)},
        advance=lambda key,clock:clock.sleep(.06))
    p.config.raw['planning_budget_s']=.05
    with pytest.raises(PickRejected) as failed:run(p,t)
    assert tried==[(0,10)] and not installed and not checked
    assert failed.value.details['stop_reason']=='global_planning_budget'


def test_measured_send_deadline_miss_never_returns_expired_path(monkeypatch):
    def late(key,ref,clock):
        clock.sleep(ref.start_s-p.config.raw['command_advance_s']-clock()+.001)
    p,t,c,tried,installed,checked=fixture(monkeypatch,geometry=late)
    p.config.raw['candidate_planning_budget_s']=10.
    with pytest.raises(PickRejected) as failed:
        smooth_planning.plan_smooth_intercept(p,t,{'joints':np.zeros(7),'width_m':.1},measure_late=True)
    assert tried==[(0,10)] and checked==[(0,10)]
    assert failed.value.details['failures'][0]['details']['reason']=='measured_deadline_missed'


def test_independent_entry_selects_policy_without_changing_legacy_entry(monkeypatch):
    from nero_core.two_second import fixed_freeze_coarse_first_pick as new
    from nero_core.two_second import fixed_freeze_pick as old
    calls=[]
    def shared(argv,**kwargs):calls.append((argv,kwargs));return 17
    monkeypatch.setattr(new,'hardware_main',shared)
    monkeypatch.setattr(old,'hardware_main',shared)
    args=['--config','example.yaml','--belt-speed-cm-s','54']
    assert new.main(args)==old.main(args)==17
    assert calls==[(args,{'freeze_source':'single_exposure_native_depth','candidate_retry_policy':'coarse_first'}),
                   (args,{'freeze_source':'single_exposure_native_depth'})]


def test_independent_check_config_keeps_taught_paths_and_never_imports_devices(monkeypatch,capsys,tmp_path):
    from nero_core.two_second.fixed_freeze_coarse_first_pick import main
    path=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    before=path.read_bytes();original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.endswith('nero_pick_hardware') or 'pyAgxArm' in name or name=='pyorbbecsdk':
            pytest.fail('check-config must not import device adapters')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    output=tmp_path/'unused'
    assert main(['--config',str(path),'--belt-speed-cm-s','54','--check-config','--output',str(output)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['candidate_retry_policy']=='coarse_first'
    assert result['freeze_source']=='single_exposure_native_depth'
    assert result['hardware_connections']==result['commands_sent']==0
    assert set(result['taught_segments'])=={'transfer','return'}
    assert result['prealign_orientation_s']==1.1 and result['brake_duration_s']==.9
    assert not output.exists() and path.read_bytes()==before


def test_independent_entry_rejects_other_strategy_before_execution(site,capsys):
    from nero_core.two_second.fixed_freeze_coarse_first_pick import main
    path,data=site
    assert main(['--config',str(path),'--belt-speed-cm-s','54','--check-config'])==2
    result=json.loads(capsys.readouterr().out)
    assert '平滑同步' in result['error']
    assert result['hardware_connections']==result['commands_sent']==0

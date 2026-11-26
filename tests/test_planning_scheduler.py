"""Recent timing, multi-target fair scheduling, and truthful expiry reasons."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.adaptive_planning import plan_with_measured_timing
from nero_core.two_second.planning_time import PlanningTimeHistory
from nero_core.two_second.lookahead_selection import pending_error
from test_two_second_hardware import Clock
from test_return_lookahead import fixture, target, wait_state
from test_frozen_pipeline_queue import multi_vision


@dataclass
class Path:
    diagnostics: dict
    planning_s: float = 0.


@dataclass
class Plan:
    path: object
    timings_s: dict


def test_rolling_completed_measurements_recover_after_slow_sample():
    h = PlanningTimeHistory(1.29, size=8, quantile=.8)
    h.observe(1.28, outcome='missed', completed=True, cancelled=False)
    for value in [.31,.32,.33,.30,.34,.32,.31,.33]:
        h.observe(value, outcome='delivered', completed=True, cancelled=False)
    assert h.estimate_s < .35
    assert h.samples.maxlen == len(h.samples) == 8
    assert h.initial_s == 1.29  # A bootstrap is not a permanent floor.


def test_fast_schedule_reject_and_cancel_do_not_train_full_planning_cost():
    clock = Clock()
    p = NS(clock=clock, config=NS(raw={'planning_budget_s':8.,
        'planning_time_estimator':'recent_complete', 'measured_planning_initial_s':.3}))
    def failed(*args, **kw):
        clock.sleep(.005)
        raise PickRejected('no space', details={'failures':[{'failed_phase':'synchronized_schedule'}]})
    for _ in range(9):
        with pytest.raises(PickRejected):
            plan_with_measured_timing(p,None,None,cancelled=lambda:False,planning_fn=failed)
    assert p.measured_planning_lead_s == .3
    assert p.planning_time_history.diagnostics()['sample_count'] == 0
    def success(*args, **kw):
        clock.sleep(.2)
        return Plan(Path({}), {'selected_candidate_ik':.15,'sampled_geometry':.05})
    result = plan_with_measured_timing(p,None,None,cancelled=lambda:False,planning_fn=success)
    assert p.measured_planning_lead_s == pytest.approx(.2)
    assert result.path.diagnostics['measured_planning']['rounds'][0]['estimate_eligible']
    assert p.planning_time_history.diagnostics()['excluded'] == {'rejected':9}


def test_same_target_miss_uses_measured_cost_even_if_recent_percentile_is_low():
    clock = Clock(); p = NS(clock=clock,config=NS(raw={'planning_budget_s':8.,
        'planning_time_estimator':'recent_complete'}))
    h = p.planning_time_history = PlanningTimeHistory(0.)
    for _ in range(8):h.observe(.2,outcome='delivered',completed=True,cancelled=False)
    leads=[]
    def solve(*a, **kw):
        leads.append(kw['reservation_s'])
        if len(leads)==1:
            clock.sleep(.8)
            raise PickRejected('late',details={'failures':[{'details':{'reason':'measured_deadline_missed'}}]})
        clock.sleep(.7)
        return Plan(Path({}),{})
    plan_with_measured_timing(p,None,None,cancelled=lambda:False,planning_fn=solve)
    assert leads == pytest.approx([.2,.8])


def test_remaining_window_is_shared_by_both_coarse_candidates(monkeypatch):
    from test_coarse_first_pick import fixture as coarse_fixture
    from nero_core.two_second import smooth_planning
    p,t,clock,_,_,_=coarse_fixture(monkeypatch)
    p.config.follow_bounds=np.array([0.,.2])
    p.config.raw.update(planning_reserve_mode='measured', measured_planning_initial_s=.1)
    t.center=np.zeros(3)
    for pose in t.grasps:pose[0,3]=0.
    attempted=[]
    class SlowSolver:
        def __init__(self,model):pass
        def solve(self,*args,**kw):
            attempted.append(p.progress['candidate_index'])
            if kw['cancelled']():raise PickRejected('cancelled test IK')
            clock.sleep(.03)
            return NS(success=True,joint_positions=np.zeros(7))
    monkeypatch.setattr(smooth_planning,'PrecisePoseSolver',SlowSolver)
    with pytest.raises(PickRejected) as caught:
        p.intercept(t,{'joints':np.zeros(7),'width_m':.1})
    assert set(attempted)=={0,1}
    failures=caught.value.details['failures']
    timed=[f for f in failures if 'candidate_deadline_monotonic_s' in f]
    assert timed[0]['candidate_deadline_monotonic_s']==pytest.approx(1.4)
    assert timed[1]['candidate_deadline_monotonic_s']==pytest.approx(1.8)
    assert timed[-1]['stop_reason']=='target_window'
    assert clock() < 1.84  # One cooperative solver step may straddle the deadline.


def queued_worker(monkeypatch):
    import nero_core.two_second.vision as module
    worker,clock,item,_,planner,events=fixture()
    v=multi_vision(monkeypatch);v.config=worker.config;worker.vision=v
    worker.multiple_pending=True;worker.config.raw['tracked_frozen_queue_capacity']=4
    monkeypatch.setattr(module,'time',NS(monotonic=clock))
    a,b=item,target(3)
    for t,x in ((a,-.3),(b,-.5)):
        t.grasps[0][0,3]=x
        t.entry_position[0]=x
        assert v.frozen_queue.offer(t,clock(),1.)
    return worker,clock,a,b,v,planner,events


def test_failed_prefetch_allows_second_piece_and_retries_first_on_schedule_change(monkeypatch):
    worker,clock,a,b,v,p,events=queued_worker(monkeypatch)
    original=p.intercept;attempts=[]
    def solve(t,*args,**kw):
        attempts.append(t.identity)
        if t is a:raise PickRejected('first IK failed',details={'failed_phase':'ik'})
        return original(t,*args,**kw)
    p.intercept=solve
    try:
        worker._step();assert worker.target is None
        worker._step();assert worker.ready.target is b and v.poll_pending() is b
        for _ in range(5):worker._step()
        assert attempts == [a.identity,b.identity]
        assert a in v.pending_candidates() and a.identity not in v.consumed
        worker.confirm_return(wait_state(clock))
        assert worker.poll_ready().target is b
        worker.discard_ready(worker.ready,PickRejected('cancel B only'))
        worker._step()
        assert attempts[-1]==a.identity
        assert a.capture_s==b.capture_s==10.
    finally:v.close()


def test_all_failed_targets_do_not_spin_until_return_revision_changes(monkeypatch):
    worker,clock,a,b,v,p,events=queued_worker(monkeypatch);attempts=[]
    def fail(t,*args,**kw):
        attempts.append(t.identity);raise PickRejected('IK')
    p.intercept=fail
    try:
        for _ in range(10):worker._step()
        assert attempts==[a.identity,b.identity]
        assert not v.consumed
        worker.return_started(identity=worker.owner,expected_ready_s=10.8)
        for _ in range(10):worker._step()
        assert attempts==[a.identity,b.identity,a.identity,b.identity]
    finally:v.close()


def test_expired_ready_exposure_does_not_pin_queue_during_unknown_return(monkeypatch):
    worker,clock,a,b,v,p,events=queued_worker(monkeypatch)
    try:
        worker._step();assert worker.ready.target is a
        clock.now=worker.ready.plan.reference.start_s+.01
        worker._step();assert worker.ready is None and worker.target is None
        worker._step();assert worker.ready.target is b
        assert a in v.pending_candidates() and a.identity not in v.consumed
    finally:v.close()


@pytest.mark.parametrize('reason',['past','clock','cancel','replace'])
def test_expiry_reason_survives_queue_pruning(monkeypatch,reason):
    worker,clock,a,b,v,p,events=queued_worker(monkeypatch)
    expired=[];v.event_sink=lambda name,**kw:expired.append((name,kw))
    try:
        if reason=='past':
            clock.now=15.;v.poll_pending()
        elif reason=='clock':v.clock_epoch=1;v.poll_pending()
        elif reason=='cancel':v.discard_pending(a.identity)
        else:v.frozen_queue.remove(a.identity,'test_replace')
        error=pending_error(v,a,worker.config,clock())
        expected={'past':'lookahead_target_past_region','clock':'lookahead_clock_invalidated',
                  'cancel':'lookahead_target_cancelled','replace':'lookahead_exposure_replaced'}
        assert error.details['reason']==expected[reason]
        if reason in ('past','clock'):
            mine=[e for name,e in expired if name=='frozen_target_expired' and e['identity']==a.identity]
            assert len(mine)==1 and mine[0]['capture_monotonic_s']==a.capture_s
            assert mine[0]['reason']==('past_follow_region' if reason=='past' else 'clock_invalidated')
    finally:v.close()


def test_tracked_entry_uses_configured_scheduler_without_device_connections(capsys):
    from pathlib import Path
    import json
    from nero_core.two_second.main import main
    cfg=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'
    assert main(['--config',str(cfg),'--check-config','--fixed-belt-speed-m-s','.283'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['candidate_retry_policy']=='coarse_first'
    assert result['planning_time_estimator']=='recent_complete'
    assert result['hardware_connections']==result['commands_sent']==0


def test_idle_selection_uses_last_dispatch_not_insertion_order(monkeypatch):
    worker,clock,a,b,v,p,events=queued_worker(monkeypatch)
    try:
        b.entry_position[0]=-.21;b.grasps[0][0,3]=-.21
        worker._step()
        assert worker.ready.target is b and v.poll_pending() is b
        assert v.frozen_queue.snapshot()['counts']['planning_promotions']==1
    finally:v.close()

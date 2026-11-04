"""Actual-timing scheduling: late plans are recomputed, never shifted/executed."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from nero_core.two_second.adaptive_planning import plan_with_measured_timing
from nero_core.two_second.synchronized_reference import freeze_window_for_speed
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_two_second_hardware import Clock, executor_fixture, cloud
from test_two_second_smoothing import fixture


def test_production_planner_recomputes_target_after_measured_miss(monkeypatch):
    planner, model, target = fixture(monkeypatch)
    clock = Clock(); planner.clock = clock
    planner.config.follow_bounds = np.array([-.2, 1.])
    planner.config.raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        planning_reserve_mode='measured', planning_reserve_s=0., synchronized_duration_s=0.,
        prealign_orientation_s=2., rotate_first_s=0., brake_duration_s=.5,
        candidate_planning_budget_s=1., planning_knot_hz=10., planning_knot_hz_fallback=10.)
    target.center=np.zeros(3)
    poses=[]
    for _ in range(2):
        pose=np.eye(4);poses.append(pose)
    target.grasps=tuple(poses)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw):
            clock.sleep(.002)
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver',Solver)
    installed=[]
    def install(*a):
        clock.sleep(.01);planner.version+=1;planner.scene_revision+=1;installed.append(clock())
    monkeypatch.setattr(planner,'install',install)
    references=[]
    def geometry(path, ref, **kw):
        references.append(ref);clock.sleep(.03);return {'offline_test':True}
    monkeypatch.setattr(planner,'check_smooth_intercept',geometry)
    result=planner.intercept(target,{'joints':np.zeros(7)})
    rounds=result.path.diagnostics['measured_planning']['rounds']
    assert rounds[0]['result']=='missed' and rounds[-1]['result']=='delivered'
    assert result.reference.start_s-planner.config.raw['command_advance_s'] > clock()
    assert len(references)>=2 and references[-1].contact_s > references[0].contact_s
    assert result.reference is references[-1]
    assert rounds[1]['reservation_s'] > 0.
    assert result.path.planning_s == pytest.approx(clock()-1.)


def test_non_timing_failure_never_invents_retry_or_sends():
    clock=Clock(); planner=SimpleNamespace(clock=clock,config=SimpleNamespace(raw={'planning_budget_s':8.}))
    calls=[]
    def rejected(*args,**kwargs):
        calls.append(1);clock.sleep(.2)
        raise PickRejected('逆解未收敛',details={'failures':[{'reason':'ik'}]})
    with pytest.raises(PickRejected,match='逆解'):
        plan_with_measured_timing(planner,None,None,cancelled=lambda:False,planning_fn=rejected)
    assert len(calls)==1


def test_measured_mode_freezes_at_existing_upstream_limit_not_2_5_seconds():
    cfg=SimpleNamespace(raw={'freeze_upstream_distance_m':.1,'synchronized_adaptive_freeze':True,
        'synchronized_max_freeze_upstream_distance_m':.35,'planning_reserve_mode':'measured'},
        follow_bounds=np.array([-.2,.28]),settings=SimpleNamespace(full_plan_close_window_s=.3))
    result=freeze_window_for_speed(cfg,.16)
    assert result['distance_m']==.35 and result['policy'].startswith('earliest_configured')


@pytest.mark.parametrize('measured_s', [0., .46])
def test_recorded_16cm_target_fits_time_schedule_without_claiming_full_path(measured_s):
    from pathlib import Path
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.synchronized_reference import schedule_forward_join
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    result=schedule_forward_join(capture_s=16972.82088780403,
        grasp_position=[-.37628971397367683,.5642714108346515,-.11726616219762855],
        velocity=[-.0005722041257250607,-.1598652246700016,.006540835088000046],
        standby_position=[-.40302374899926946,-.03280077810872306,.03340051273551789],
        direction=cfg.direction,now_s=16973.323785748,planning_reserve_s=measured_s,
        advance_s=.1,join_s=2.,synchronized_s=0.,follow_s=.3,follow_bounds=cfg.follow_bounds)
    assert result['follow_end_coordinate_m']==pytest.approx(.242069,abs=1e-5)
    assert result['follow_end_coordinate_m'] <= cfg.follow_bounds[1]


def test_send_preparation_miss_replans_with_measured_delay(monkeypatch):
    from concurrent.futures import Future
    from nero_core.two_second.planning import InterceptPlan
    from nero_core.two_second.reference import Intercept
    from nero_core.two_second.trajectory import TimedPath
    from nero_core.two_second.vision import MeasuredTarget, obb_grasps
    e,device,clock=executor_fixture()
    e.config.raw.update(planning_reserve_mode='measured', planning_budget_s=8.,
                        closure_policy='scheduled_attempt')
    pts=cloud();poses,box=obb_grasps(pts,np.eye(4),np.eye(4))
    target=MeasuredTarget((0,1,2),clock(),clock(),100.,0,1,np.zeros(3),np.zeros(3),tuple(poses),pts,
        {'belt_region':{'plane_to_base':np.eye(4)},'obb':box},{},{'source':'nero7_xy_kalman'})
    class ImmediatePool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,*a,**kw):
            f=Future()
            try:f.set_result(fn(*a,**kw))
            except BaseException as error:f.set_exception(error)
            return f
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor',ImmediatePool)
    calls=[]
    def intercept(target,state,**kwargs):
        calls.append(clock())
        start=clock()+(.15 if len(calls)==1 else .3)
        ref=Intercept(clock(),start,2.,np.zeros(3),np.zeros(3),np.zeros(3),[1,0,0,0],[1,0,0,0])
        path=TimedPath(lambda t,d=0:np.zeros(7),target.identity,1,'intercept_follow',2.6,0.,
                       {'measured_planning':{'completed_monotonic_s':clock()}})
        return InterceptPlan(path,ref,target,1,target.grasps[0],())
    e.planner.intercept=intercept
    device.prepare_motion=lambda *a,**kw:clock.sleep(.06)
    def stream(*args,**kwargs):
        device.width=0.;device.force=0.
        return device.snapshot()
    monkeypatch.setattr(e,'_stream',stream)
    result=e._run_attempt(target)
    assert len(calls)==2 and result['status']=='attempt_completed_grip_unconfirmed'
    assert e.planner.measured_post_planning_s == pytest.approx(.06)
    assert any(row['event']=='planning_delivery_missed' for row in e.events)


def test_expanded_measured_freeze_window_admits_recorded_early_track():
    from nero_core.two_second.synchronized_reference import admit_synchronized_observation
    from nero_core.two_second.reference import EntryNotReady
    coordinate = -.6189359741893515  # observed, qualified track in RGB-D frame 31
    with pytest.raises(EntryNotReady):
        admit_synchronized_observation(10., coordinate, .3846, [-.2,.38], 10.03, .35)
    assert np.isfinite(admit_synchronized_observation(10., coordinate, .3846, [-.2,.38], 10.03, .45))
    with pytest.raises(EntryNotReady):
        admit_synchronized_observation(10., -.651, .3846, [-.2,.38], 10.03, .45)

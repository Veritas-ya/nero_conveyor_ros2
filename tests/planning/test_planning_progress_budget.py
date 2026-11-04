"""Real IK/RRT worker cancellation and snapshot evidence, no devices."""
from dataclasses import replace
import threading
import time
import numpy as np
import pytest
from test_planning_worker_scene_validator import model, make_engine, make_request
from mujoco_conveyor_sorting.controllers.planning_worker import AsyncPlanningWorker
from mujoco_conveyor_sorting.controllers.planning_progress import PlanningProgress


def wait_result(worker, seconds=3.):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=worker.poll()
        if value is not None:return value
        time.sleep(.002)
    pytest.fail('worker did not acknowledge cancellation')


def test_worker_times_out_real_search_then_accepts_next_request(model):
    entered=threading.Event()
    def expensive_scene(request,q):
        entered.set()
        if request.request_id==1:time.sleep(.012)
        return True
    worker=AsyncPlanningWorker(make_engine(model,expensive_scene))
    try:
        worker.set_candidate_wall_budget(.025)
        first=make_request()
        assert worker.submit(first) and entered.wait(1.)
        running=worker.progress_snapshot()
        assert running['status']=='running' and running['active_stages']
        result=wait_result(worker)
        assert result.error=='planning_candidate_budget_exceeded'
        assert result.cancelled and not result.success and result.path is None
        assert result.full_pick_plan is None and not result.path_fully_validated
        progress=result.diagnostics['live_progress']
        assert progress['budget_exceeded'] and progress['cancellation_ack_wall_s'] is not None
        assert progress['stage_totals'] and progress['stages_at_budget_exceeded']
        worker.set_candidate_wall_budget(2.)
        assert worker.submit(replace(first,request_id=2))
        second=wait_result(worker)
        assert second.request_id==2 and second.success
        assert not second.diagnostics['live_progress']['budget_exceeded']
        assert second.diagnostics['live_progress']['request_id']==2
    finally:
        assert worker.close()


def test_external_cancel_is_not_reclassified_as_budget(model):
    entered=threading.Event()
    def scene(request,q):
        entered.set(); time.sleep(.01); return True
    worker=AsyncPlanningWorker(make_engine(model,scene))
    try:
        worker.set_candidate_wall_budget(2.)
        assert worker.submit(make_request()) and entered.wait(1.)
        worker.cancel_pending()
        result=wait_result(worker)
        assert result.error!='planning_candidate_budget_exceeded'
        assert not result.diagnostics['live_progress']['budget_exceeded']
    finally:
        assert worker.close()


def test_progress_nested_times_do_not_double_count_and_snapshot_is_detached():
    progress=PlanningProgress();progress.begin(make_request(),1.)
    with progress.stage('parent',{}):
        time.sleep(.002)
        with progress.stage('child',{}):time.sleep(.002)
        snapshot=progress.snapshot()
        snapshot['stage_totals']['child']['exclusive_wall_s']=100.
    progress.finish('failed')
    snapshot=progress.snapshot()
    totals=snapshot['stage_totals']
    assert totals['child']['exclusive_wall_s']<1.
    assert totals['parent']['inclusive_wall_s']==pytest.approx(
        totals['parent']['exclusive_wall_s']+totals['child']['inclusive_wall_s'],abs=.002)
    assert sum(t['exclusive_wall_s'] for t in totals.values())<=snapshot['elapsed_wall_s']+.002


def test_late_success_cannot_survive_budget_even_without_engine_polling(model):
    engine=make_engine(model);request=make_request()
    certificate=engine.plan(request,lambda:False)
    assert certificate.success
    def ignores_cancellation(request,cancelled):
        time.sleep(.025)
        return certificate
    engine.plan=ignores_cancellation
    worker=AsyncPlanningWorker(engine)
    try:
        worker.set_candidate_wall_budget(.005)
        assert worker.submit(request)
        result=wait_result(worker)
        assert result.error=='planning_candidate_budget_exceeded'
        assert result.path is None and not result.path_fully_validated and not result.success
    finally:
        assert worker.close()

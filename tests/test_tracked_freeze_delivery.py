"""Upstream admission and bounded asynchronous geometry publication; no devices."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from nero_core.two_second.tracked_freeze import detector_freeze_window
from nero_core.two_second.reference import EntryNotReady
from nero_core.two_second.synchronized_reference import admit_synchronized_observation
from nero_core.two_second.synchronized_reference import freeze_window_for_speed
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_two_second_diagnostics import vision_fixture


def footprint(axis=(0., -1., 0.), y=(-.7, 1.1)):
    cfg=SimpleNamespace(direction=np.array(axis),follow_bounds=np.array([-.2,.38]),raw={
        'perception_detector_workspace_y_m':list(y),'maximum_target_height_m':.12})
    surface=SimpleNamespace(hull_xy=np.array([[-.5,-.8],[-.2,-.8],[-.2,1.2],[-.5,1.2]]),
        plane_to_base=np.eye(4))
    return cfg,surface


def test_freeze_covers_detected_upstream_and_respects_y_clip():
    cfg,surface=footprint(); before=cfg.follow_bounds.copy()
    window=detector_freeze_window(cfg,surface)
    assert window['distance_m']==pytest.approx(.9)
    assert np.isfinite(admit_synchronized_observation(10.,-.8139097455,.535,
        cfg.follow_bounds,10.06,window['distance_m']))
    with pytest.raises(EntryNotReady,match='too far upstream'):
        admit_synchronized_observation(10.,-1.11,.535,cfg.follow_bounds,10.06,window['distance_m'])
    np.testing.assert_array_equal(before,cfg.follow_bounds)


def test_tracked_limit_applies_at_original_exposure_and_respects_shorter_footprint():
    cfg, surface = footprint()
    cfg.raw.update(tracked_freeze_range='detector_footprint', tracked_freeze_upstream_limit_m=.75)
    region = {'hull_xy_m': surface.hull_xy, 'plane_to_base': surface.plane_to_base}
    for geometry in ({'surface': surface}, {'belt_region': region}):
        window = freeze_window_for_speed(cfg, .148, **geometry)
        assert window['distance_m'] == .75 and window['capped']
        assert window['upstream_coordinate_m'] == pytest.approx(-.95)
        # The part has crossed the line by processing time, but exposure was too early.
        with pytest.raises(EntryNotReady, match='too far upstream'):
            admit_synchronized_observation(10., -.951, .148, cfg.follow_bounds, 10.1, .75)
        assert np.isfinite(admit_synchronized_observation(10., -.95, .148, cfg.follow_bounds, 10.1, .75))
    cfg.raw['perception_detector_workspace_y_m'] = [-.7, .8]
    window = detector_freeze_window(cfg, surface)
    assert window['distance_m'] == pytest.approx(.6) and not window['capped']


def test_freeze_projects_belt_axis_instead_of_using_y_as_distance():
    cfg,surface=footprint(axis=(1.,0.,0.))
    assert detector_freeze_window(cfg,surface)['distance_m']==pytest.approx(.3)
    cfg.direction=np.array([0.,1.,0.])
    assert detector_freeze_window(cfg,surface)['distance_m']==pytest.approx(.5)
    cfg.raw['perception_detector_workspace_y_m']=[2.,3.]
    with pytest.raises(EntryNotReady):detector_freeze_window(cfg,surface)


@pytest.mark.parametrize('invalidate',[None,'consume','cancel','epoch','generation','stale'])
def test_completed_job_publishes_without_next_camera_frame(monkeypatch,invalidate):
    vision=vision_fixture(monkeypatch,'eligible')
    frame=vision.camera.read_latest(); peer=vision.meter.latest_snapshot
    vision.clock_epoch=0; vision.progress_snapshot=peer
    result=vision._prepare_target(frame,np.eye(4),peer,peer.tracks[0],vision.read_feedback(),
        frame.header.received_time-.001,0,vision.model.measured_belt_surface,.05)
    assert result is not None
    job=Future();vision.geometry_future=job;job.add_done_callback(vision._geometry_completed)
    if invalidate=='consume':vision.consume(result[2].identity)
    if invalidate=='cancel':vision.stop.set()
    if invalidate=='epoch':vision.clock_epoch=1
    if invalidate=='generation':vision.progress_snapshot=replace(peer,track_generation=99)
    if invalidate=='stale':result=(*result[:2],replace(result[2],capture_s=time.monotonic()-2.))
    with ThreadPoolExecutor(2) as pool:
        pool.submit(job.set_result,result).result(timeout=2.)
        # A second polling drain cannot publish a completed job twice.
        vision._finish_geometry()
    assert (vision.latest is not None)==(invalidate is None)
    if invalidate is None:
        assert vision.latest.sequence==frame.header.sequence
        assert vision.latest.detection['freeze_delivery']['geometry_done_to_publish_s']<.1
    vision.close()


def test_completion_exception_reaches_consumer(monkeypatch):
    vision=vision_fixture(monkeypatch,'eligible')
    job=Future();vision.geometry_future=job;job.add_done_callback(vision._geometry_completed)
    job.set_exception(RuntimeError('offline worker failed'))
    with pytest.raises(RuntimeError,match='offline worker failed'):vision.poll()
    vision.close()


def test_live_and_frozen_footprint_share_policy_even_above_legacy_cap():
    cfg, surface = footprint()
    cfg.raw.update(tracked_freeze_range='detector_footprint',
        synchronized_adaptive_freeze=True, synchronized_max_freeze_upstream_distance_m=.6)
    measured = BeltSurface(surface.plane_to_base, np.array([[-.5,-.2],[-.8,1.2]]),
        surface.hull_xy, {})
    region = measured.region_metadata()
    live = freeze_window_for_speed(cfg, .283, surface=measured)
    frozen = freeze_window_for_speed(cfg, .283, belt_region=region)
    assert frozen['distance_m'] == live['distance_m'] == pytest.approx(.9)
    # Later calibration/surface updates must not change the frozen planner input.
    measured.hull_xy[:] = 0.
    assert freeze_window_for_speed(cfg,.283,belt_region=region)['distance_m']==pytest.approx(.9)
    region.pop('hull_xy_m')
    legacy = freeze_window_for_speed(cfg,.283,belt_region=region)
    assert legacy['distance_m']==pytest.approx(.9)
    assert legacy['footprint_source']=='legacy_frozen_native_bounds'
    cfg.raw.update(freeze_position_upstream_m=.4, freeze_position_tolerance_m=.02)
    assert freeze_window_for_speed(cfg,.283)['policy']=='fixed_capture_position'


def upstream_planner(monkeypatch):
    from test_two_second_smoothing import fixture
    planner, model, target = fixture(monkeypatch)
    planner.config.follow_bounds = np.array([-.2,1.])
    planner.config.raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        planning_reserve_mode='measured',planning_reserve_s=0., synchronized_duration_s=.2,
        prealign_orientation_s=2.,rotate_first_s=0.,brake_duration_s=.5,
        candidate_planning_budget_s=1.,planning_knot_hz=10.,planning_knot_hz_fallback=10.,
        synchronized_adaptive_freeze=True,synchronized_max_freeze_upstream_distance_m=.6,
        tracked_freeze_range='detector_footprint',
        perception_detector_workspace_y_m=[-.7,1.1],maximum_target_height_m=.12)
    target.center=np.array([-.95,0.,0.])
    target.grasps=tuple(np.eye(4) for _ in range(2))
    for pose in target.grasps:pose[0,3]=-.95
    surface=BeltSurface(np.eye(4),np.array([[-1.2,.4],[-.1,.1]]),
        np.array([[-1.2,-.1],[.4,-.1],[.4,.1],[-1.2,.1]]),{})
    target.detection={'belt_region':surface.region_metadata()}
    class Solver:
        def __init__(self, model):pass
        def solve(self,*a,**kw):
            return SimpleNamespace(success=True,joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver',Solver)
    def install(*a):planner.version+=1;planner.scene_revision+=1
    monkeypatch.setattr(planner,'install',install)
    monkeypatch.setattr(planner,'check_smooth_intercept',lambda *a,**kw:{'offline_test':True})
    return planner, target


def test_real_planner_admits_upstream_frozen_input_through_to_delivery(monkeypatch):
    planner,target=upstream_planner(monkeypatch)
    with pytest.raises(EntryNotReady,match='too far upstream'):
        admit_synchronized_observation(target.capture_s,target.center[0],.1,
            planner.config.follow_bounds,planner.clock(),.6)
    plan=planner.intercept(target,{'joints':np.zeros(7)})
    assert plan.path.diagnostics['freeze_window']['distance_m']==pytest.approx(1.)
    assert plan.path.diagnostics['ik_solve_count']>0
    assert plan.reference.start_s-planner.config.raw['command_advance_s']>planner.clock()
    assert plan.reference.capture_s==target.capture_s


@pytest.mark.parametrize('rejection',['too_far_upstream','past_follow_region','detection_footprint_missing'])
def test_expected_admission_failure_does_not_kill_next_piece_worker(monkeypatch,rejection):
    from test_return_lookahead import fixture, wait_state, target as next_target
    planner,invalid=upstream_planner(monkeypatch)
    if rejection=='too_far_upstream':invalid.center=np.array([-2.,0.,0.])
    if rejection=='past_follow_region':invalid.center=np.array([2.,0.,0.])
    if rejection=='detection_footprint_missing':invalid.detection={}
    with pytest.raises(PickRejected) as caught:
        planner.intercept(invalid,{'joints':np.zeros(7)})
    assert caught.value.details['reason']==rejection
    worker,clock,item,vision,fake,events=fixture()
    original=fake.intercept
    def fail(*a,**kw):
        return planner.intercept(invalid,{'joints':np.zeros(7)})
    fake.intercept=fail
    worker.confirm_return(wait_state(clock))
    worker._step()
    assert worker.error is None and worker.poll_ready() is None
    assert vision.discarded==[item.identity]
    assert events[-1][1]['failure_details']['reason']==rejection
    fake.intercept=original
    vision.latest=next_target(3)
    worker._step()
    assert worker.poll_ready().target is vision.latest

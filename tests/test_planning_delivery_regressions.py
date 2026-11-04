"""Offline regressions for the 2026-10-30 measured planning timeout; no devices."""
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.trajectory import TimedPath, geometry_check_times
from nero_core.two_second.config import CellConfig
from test_two_second_hardware import Clock, site
from test_two_second_smoothing import fixture as planner_fixture
from test_two_second_diagnostics import vision_fixture


@pytest.mark.parametrize('requested,valid', [(8., True), (8.01, False), (0., False)])
def test_reported_manufacturer_acceleration_cap_replaces_legacy_software_ceiling(site, requested, valid):
    path, data = site
    data['pick_settings'] = {'speed_percent': 100}
    data['manufacturer_joint_acceleration_limits_rad_s2'] = [8.]*7
    data['manufacturer_acceleration_limit_source'] = 'user_reported_manufacturer_spec_20261030'
    data['joint_acceleration_limits_rad_s2'] = [requested]*7
    path.write_text(yaml.safe_dump(data))
    if not valid:
        with pytest.raises(ValueError):
            CellConfig.load(path)
        return
    config = CellConfig.load(path)
    np.testing.assert_allclose(config.acceleration_limits, 8.)
    assert config.settings.joint_acceleration_limit == 2.  # legacy behavior remains available
    assert config.raw['motion_limit_sources']['acceleration_cap'].startswith('user_reported_')


@pytest.mark.parametrize('capability,source', [(None, 'test'), ([float('inf')]*7, 'test'),
                                             ([-1.]*7, 'test'), ([8.]*7, '')])
def test_invalid_or_unsourced_manufacturer_limit_is_rejected(site, capability, source):
    path, data = site
    data.update(manufacturer_joint_acceleration_limits_rad_s2=capability,
                manufacturer_acceleration_limit_source=source)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CellConfig.load(path)


def test_geometry_keeps_every_dense_endpoint_and_extra_joint_subdivision(monkeypatch):
    planner, _, _ = planner_fixture(monkeypatch)
    planner.config.settings = SimpleNamespace(control_hz=100)
    path = TimedPath(lambda t, d=0: np.ones(7)*(5.*t if d == 0 else 5. if d == 1 else 0.),
                     (0, 1, 2), 1, 'coverage', .01, 0.)
    checked = []
    planner.valid = lambda q, width, **kw: checked.append((q.copy(), kw['stamp']))
    planner.check_path(path, .1, origin=10.)
    assert len(checked) == 9  # 0, four subdivisions to .005, four to .010
    np.testing.assert_allclose([q[0] for q, _ in checked], np.linspace(0., .05, 9))
    np.testing.assert_allclose([stamp for _, stamp in checked], np.linspace(10., 10.01, 9))
    assert planner.progress['sample_index'] == 2
    assert planner.progress['sample_count'] == 3


def test_configured_50hz_preserves_phase_boundaries_and_motion_subdivisions(monkeypatch):
    times=geometry_check_times(5.8,50.,1.5,3.5,4.3)
    assert len(times)==291 and max(np.diff(times)) <= .0200000001
    assert all(np.any(np.isclose(times,t,atol=1e-12)) for t in (0.,1.5,3.5,4.3,5.8))
    assert .013 in geometry_check_times(.04,50.,.013)
    planner, _, _=planner_fixture(monkeypatch)
    planner.config.settings=SimpleNamespace(control_hz=100)
    planner.config.raw['geometry_check_hz']=50.
    path=TimedPath(lambda t,d=0:np.ones(7)*(5.*t if d==0 else 5. if d==1 else 0.),
                   (0,1,2),1,'coverage',.02,0.)
    checked=[]
    planner.valid=lambda q,width,**kw:checked.append((q.copy(),kw['stamp']))
    planner.check_path(path,.1,origin=10.)
    assert len(checked)>2  # .008-rad spatial subdivisions survive the coarser time grid.
    assert checked[0][1]==10. and checked[-1][1]==pytest.approx(10.02)


def test_coarser_geometry_grid_does_not_claim_continuous_collision_coverage(monkeypatch):
    planner, _, _=planner_fixture(monkeypatch)
    planner.config.settings=SimpleNamespace(control_hz=100)
    path=TimedPath(lambda t,d=0:np.full(7,t if d==0 else 1. if d==1 else 0.),
                   (0,1,2),1,'sampled_only',.02,0.)
    def narrow_collision(q,width,**kwargs):
        if .00499 < q[0] < .00501:
            raise PickRejected('synthetic_narrow_collision')
    planner.valid=narrow_collision
    planner.config.raw['geometry_check_hz']=50.
    planner.check_path(path,.1,origin=10.)
    planner.config.raw['geometry_check_hz']=200.
    with pytest.raises(PickRejected,match='synthetic_narrow_collision'):
        planner.check_path(path,.1,origin=10.)


@pytest.mark.parametrize('rate',[0.,-1.,float('inf')])
def test_invalid_geometry_rate_is_rejected(site,rate):
    path,data=site
    data['geometry_check_hz']=rate
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='geometry_check_hz'):
        CellConfig.load(path)


def test_same_joint_pose_at_new_time_still_checks_contact_and_payload_boundary(monkeypatch):
    planner, _, _ = planner_fixture(monkeypatch)
    planner.config.settings = SimpleNamespace(control_hz=100)
    path = TimedPath(lambda t, d=0: np.zeros(7), (0, 1, 2), 1, 'phases', .01, 0.)
    reference = SimpleNamespace(follow_end_s=.01, sample=lambda t: (np.zeros(3), np.array([1., 0., 0., 0.])))
    checked = []
    planner.valid = lambda q, width, **kw: checked.append(kw)
    planner.check_path(path, .1, origin=10., contact_time=10.005, reference=reference)
    assert len(checked) == 3
    assert [p['contact'] for p in checked] == [False, True, True]
    assert [p['payload'] for p in checked] == [False, False, True]
    assert checked[-1]['stamp'] is None and checked[-1]['allow_belt_support']
    # Another exposure/scene must perform all checks again, even at identical q.
    planner.check_path(path, .1, origin=11., reference=reference)
    assert len(checked) == 6


def test_geometry_cancellation_is_checked_inside_extra_subdivisions(monkeypatch):
    planner, _, _ = planner_fixture(monkeypatch)
    planner.config.settings = SimpleNamespace(control_hz=100)
    path = TimedPath(lambda t, d=0: np.ones(7)*(5.*t if d == 0 else 5. if d == 1 else 0.),
                     (0, 1, 2), 1, 'cancel', .01, 0.)
    checked = []
    planner.valid = lambda *a, **kw: checked.append(1)
    with pytest.raises(PickRejected, match='取消'):
        planner.check_path(path, .1, cancelled=lambda: len(checked) >= 2)
    assert len(checked) == 2


def budget_fixture(monkeypatch):
    planner, _, target = planner_fixture(monkeypatch)
    clock = Clock()
    planner.clock = clock
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=.1)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw):
            clock.sleep(.0001)
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *a: None)
    return planner, target, clock


def test_failed_candidate_records_its_partial_phase_cost_and_next_grasp_runs(monkeypatch):
    planner, target, clock = budget_fixture(monkeypatch)
    calls = []
    def geometry(*a, cancelled):
        calls.append(1)
        if len(calls) == 1:
            clock.sleep(.09)
        if cancelled():
            raise PickRejected('injected geometry timeout')
    planner.check_smooth_intercept = geometry
    result = planner.intercept(target, {'joints': np.zeros(7)})
    failed = result.failures[0]
    assert failed['failed_phase'] == 'sampled_geometry'
    assert failed['stop_reason'] == 'candidate_budget'
    assert failed['phase_timings_s']['sampled_geometry'] == pytest.approx(.09)
    assert failed['phase_timings_s']['ik'] > .02
    assert len(calls) == 2


def test_send_deadline_is_distinguished_from_private_candidate_budget(monkeypatch):
    planner, target, clock = budget_fixture(monkeypatch)
    planner.config.raw['candidate_planning_budget_s'] = 10.
    def geometry(*a, cancelled):
        clock.sleep(4.)
        assert cancelled()
        raise PickRejected('injected deadline')
    planner.check_smooth_intercept = geometry
    with pytest.raises(PickRejected) as failure:
        planner.intercept(target, {'joints': np.zeros(7)})
    detail = failure.value.details
    assert detail['stop_reason'] == 'send_deadline'
    assert detail['available_planning_s'] == pytest.approx(3.9)
    assert len(detail['failures']) == 1
    assert detail['failures'][0]['failed_phase'] == 'sampled_geometry'


def test_active_frozen_task_keeps_camera_epoch_live_without_unused_perception(monkeypatch):
    vision = vision_fixture(monkeypatch, 'eligible')
    vision.consume((0, 1, 1))
    read_frame = vision.camera.read_latest
    def reset_clock(**kwargs):
        frame = read_frame(**kwargs)
        frame.header.clock_epoch = 3
        return frame
    vision.camera.read_latest = reset_clock
    def unused(*a, **kw):
        pytest.fail('frozen task must not compute another velocity or grasp')
    vision.meter.update = unused
    monkeypatch.setattr('nero_core.two_second.vision.apply_runtime_depth_calibration', unused)
    vision._run()
    assert vision.error is None and vision.poll() is None
    assert vision.clock_epoch == 3  # the executor can cancel its epoch-0 observation
    report = vision.diagnostics()
    assert report['frames_received'] == 1 and report['target_processing_paused']
    assert 'task_active_camera_only' in report['stage_wall_s']


def test_resume_rejects_pre_return_exposure_and_keeps_old_target_consumed(monkeypatch):
    vision = vision_fixture(monkeypatch, 'eligible')
    vision.consume((0, 1, 1))
    vision.resume_targets()
    vision._run()
    assert vision.error is None and vision.latest is None
    assert vision.consumed == {(0, 1, 1)}
    assert vision.reset_tracking
    assert 'waiting_new_exposure' in vision.diagnostics()['stage_wall_s']


def test_resume_resets_kalman_history_on_the_camera_worker(monkeypatch):
    vision = vision_fixture(monkeypatch, 'eligible')
    vision.consume((0, 1, 1))
    vision.resume_targets()
    vision.minimum_capture_s = float('-inf')  # fixture exposes one explicitly fresh frame
    reset = []
    vision.meter.reset = lambda: reset.append(1)
    vision._run()
    assert vision.error is None
    assert reset == [1] and not vision.reset_tracking
    assert vision.latest is None  # the consumed identity is still unavailable


def test_frame_already_in_detection_cannot_publish_after_return_handoff(monkeypatch):
    vision = vision_fixture(monkeypatch, 'eligible')
    from nero_core.two_second import vision as module
    original = module.obb_grasps
    def handoff(*args, **kwargs):
        result = original(*args, **kwargs)
        vision.consume((0, 1, 99))  # a different foreground task won the handoff
        vision.resume_targets()
        return result
    monkeypatch.setattr(module, 'obb_grasps', handoff)
    vision._run()
    assert vision.error is None and vision.latest is None

"""Production smooth hardware chain with explicitly ideal devices; no physical success claim."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import time
import numpy as np
import pytest
import yaml
from test_two_second_hardware import executor_fixture, Clock, site, cloud
from test_two_second_smoothing import fixture as planner_fixture
from nero_core.two_second.config import CellConfig
from nero_core.two_second.feedback_gates import GRIP_CONFIRMATION_DWELL_S
from nero_core.two_second.command_history import CommandHistory
from nero_core.two_second.smooth_reference import SmoothIntercept
from nero_core.two_second.trajectory import (TimedPath, joint_curve, extrema,
    certify_curve, retime_samples, merge_collinear_waypoints)
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


@pytest.mark.parametrize('field,value', [
    ('trajectory_profile', 'unchecked'), ('approach_duration_candidates_s', [3., 2.]),
    ('approach_duration_candidates_s', []), ('brake_duration_s', 0.),
    ('candidate_planning_budget_s', float('inf')), ('closure_policy', 'invalid')])
def test_smooth_config_rejects_invalid_scheduling(site, field, value):
    path, data = site
    data['trajectory_profile'] = 'smooth'
    data[field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CellConfig.load(path)


@pytest.mark.parametrize('policy,profile,strategy,brake,allowed', [
    ('scheduled_attempt', 'smooth', 'synchronized', .8, True),
    ('scheduled_attempt', 'smooth', 'synchronized', .199, False),
    ('feedback', 'smooth', 'synchronized', .8, False),
    ('scheduled_attempt', 'smooth', 'classic', .8, False),
    ('scheduled_attempt', 'legacy', 'classic', .8, False)])
def test_zero_follow_requires_scheduled_sync_and_retains_grip_dwell(
        site, policy, profile, strategy, brake, allowed):
    from mujoco_conveyor_sorting.runtime.single_pick import PickSettings
    with pytest.raises(ValueError, match='closure window'):
        PickSettings(full_plan_close_window_s=0.).validate()
    path, data = site
    data.update(trajectory_profile=profile, intercept_strategy=strategy,
                closure_policy=policy, prealign_orientation_s=1.1, brake_duration_s=brake,
                pick_settings={'full_plan_close_window_s': 0.})
    path.write_text(yaml.safe_dump(data))
    if allowed:
        cfg = CellConfig.load(path)
        assert cfg.settings.full_plan_close_window_s == 0.
        assert cfg.raw['brake_duration_s'] >= GRIP_CONFIRMATION_DWELL_S
    else:
        with pytest.raises(ValueError):
            CellConfig.load(path)


def test_quintic_curve_extrema_and_rest_qva_endpoints():
    t = np.linspace(0., 1., 41)
    q = np.zeros((len(t), 7)); q[:, 0] = .1*(10*t**3-15*t**4+6*t**5)
    curve = joint_curve(t, q, zero_end_acceleration=True)
    for d in (1, 2):
        np.testing.assert_allclose(curve([0., 1.], d), 0., atol=1e-10)
    assert np.max(abs(np.asarray(extrema(curve, 2)))) == pytest.approx(.577350269, rel=1e-7)
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*.05, acceleration_limits=np.ones(7)*.1,
                          settings=SimpleNamespace(control_hz=100))
    path = retime_samples(t, q, cfg, -np.ones(7), np.ones(7), identity=(0, 1), version=1, stage='transfer')
    assert path.duration > 3.7
    assert path.diagnostics['rest_qva_endpoints']
    assert np.max(abs(np.asarray(extrema(path.curve, 1)))) < .05
    assert np.max(abs(np.asarray(extrema(path.curve, 2)))) < .1
    np.testing.assert_allclose(path.curve(t*path.duration), q, atol=1e-12)
    with pytest.raises(PickRejected) as error:
        certify_curve(t, q, cfg, -np.ones(7), np.ones(7), identity=(0, 1), version=1,
                      stage='transfer', zero_end_acceleration=True)
    assert error.value.details['derivative'] == 1


def test_site_per_joint_speed_cap_rejects_only_when_exceeded():
    from pathlib import Path
    cfg = CellConfig.load(Path(__file__).parents[1]/
                          'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    np.testing.assert_allclose(cfg.velocity_limits, [np.pi]*3+[1.25*np.pi]*4)
    assert cfg.raw['prealign_orientation_s'] > 0
    times = np.linspace(0., 1., 41)
    def samples(amount):
        q = np.zeros((len(times), 7))
        q[:, 4] = amount*times
        return q
    options = dict(identity=(0, 1, 2), version=1, stage='offline_joint5',
                   stationary_ends=False)
    path = certify_curve(times, samples(3.9), cfg, -np.ones(7)*10, np.ones(7)*10, **options)
    assert np.max(abs(np.asarray(extrema(path.curve, 1))[:, 4])) < cfg.velocity_limits[4]
    with pytest.raises(PickRejected) as error:
        certify_curve(times, samples(4.0), cfg, -np.ones(7)*10, np.ones(7)*10, **options)
    assert error.value.details['derivative'] == 1
    assert error.value.details['exceeded_joint_numbers'] == [5]


def test_collinear_merge_keeps_corners_and_reversals():
    q = np.eye(7)
    path = [np.zeros(7), .5*q[0], q[0], .5*q[0], .5*q[0]+q[1]]
    result = merge_collinear_waypoints(path)
    np.testing.assert_allclose(result, [path[i] for i in (0, 2, 3, 4)])


def test_dispatch_history_survives_sender_handoffs():
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*2.,
        raw={'runtime_command_acceleration_check_enabled': True})
    history = CommandHistory(cfg)
    history.record(1., .01, np.zeros(7), {})
    history.record(1.01, .01, np.zeros(7), {})
    # A new path cannot hide a 1 rad/s velocity jump by resetting its sender.
    with pytest.raises(PickRejected, match='加速度'):
        history.validate(np.ones(7)*.01, 1.02)


def test_late_first_command_cannot_skip_the_rest_connection():
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*2.,
                          settings=SimpleNamespace(control_hz=100),
                          raw={'runtime_command_acceleration_check_enabled': True})
    history = CommandHistory(cfg)
    with pytest.raises(PickRejected, match='首个发送点'):
        history.validate_first(np.ones(7)*.01, np.zeros(7))
    assert history.dispatch is None  # no fabricated send timestamp


@pytest.mark.parametrize('acceleration_limit', [2., 8.])
def test_native_rrt_route_uses_retiming_and_rest_handoff_in_explicit_empty_scene(acceleration_limit):
    """Real model/RRT/curve checking, synthetic empty scene; no field obstacle claim."""
    from pathlib import Path
    from nero_core.two_second.planning import MeasuredPlanner
    cfg = CellConfig.load(Path(__file__).parents[1]/
                          'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    cfg = replace(cfg, acceleration_limits=np.ones(7)*acceleration_limit)
    planner = MeasuredPlanner(cfg)
    planner.identity = (0, 1, 2)
    # Explicit test fixture: supporting plane below the whole robot, no scene.
    planner.plane = np.eye(4); planner.plane[2, 3] = -1.
    planner.support = np.array([0., 0., 1.])
    planner.support_origin = planner.plane[:3, 3].copy()
    planner.belt_bounds = np.array([[-5., 5.], [-5., 5.]])
    goal = cfg.standby.copy(); goal[0] += .02
    # Recorded hardware residual must pass the handoff without shifting the command anchor.
    actual = cfg.standby.copy(); actual[4] += .009413863428370028
    outside = cfg.standby.copy(); outside[4] += np.deg2rad(1.01)
    with pytest.raises(PickRejected, match='指令起点与实际反馈偏差过大'):
        planner.route({'joints': outside, 'width_m': .1},
            stage='synthetic_empty_scene_route', goal_joints=goal, start_joints=cfg.standby)
    path = planner.route({'joints': actual, 'width_m': .1},
        stage='synthetic_empty_scene_route', goal_joints=goal, start_joints=cfg.standby)
    assert path.diagnostics['rest_qva_endpoints']
    np.testing.assert_allclose(path.q0, cfg.standby, atol=1e-12)
    np.testing.assert_allclose(path.q1, goal, atol=1e-12)
    for derivative in (1, 2):
        np.testing.assert_allclose(path.curve([0., path.duration], derivative), 0., atol=1e-9)
    assert np.max(abs(np.asarray(extrema(path.curve, 2)))) < acceleration_limit


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('scheduled', [False, True])
@pytest.mark.parametrize('synchronized', [False, True])
@pytest.mark.parametrize('short_follow,join_s,brake_s', [
    (None, 2., 1.5), (.3, 2., .5), (.1, 2., .5), (.1, 1.2, .3), (0., 1.1, .8)])
def test_one_stream_confirms_grip_before_braking_and_sends_rest_endpoint(monkeypatch, empty, scheduled, synchronized, short_follow, join_s, brake_s):
    if short_follow is not None and not (scheduled and synchronized):
        pytest.skip('short follow is allowed only for synchronized scheduled closure')
    e, device, clock = executor_fixture(empty=empty)
    e.config.raw['trajectory_profile'] = 'smooth'
    if scheduled:
        e.config.raw['closure_policy'] = 'scheduled_attempt'
        monkeypatch.setattr('nero_core.two_second.streaming.closure_feedback', lambda *a, **k: (False, {}))
    if short_follow == 0.:
        from nero_core.two_second.execution import TwoSecondExecutor
        e.config.raw['intercept_strategy'] = 'synchronized'
        e.config.settings = replace(e.config.settings, full_plan_close_window_s=0.)
        e = TwoSecondExecutor(e.config, e.planner, device, clock=clock, sleep=clock.sleep)
        e.owner = (0, 1, 2)
        e.version = 1
    phases, attachments, runtime_stamps = [], [], []
    e.planner.valid = lambda *a, **kw: runtime_stamps.append(kw.get('stamp'))
    e.planner.attach_measured = lambda state, stamp: attachments.append((clock(), stamp))
    class Sender:
        def __init__(self, dev, step, after, **kw):
            self.dev, self.step, self.after = dev, step, after
            self.report = SimpleNamespace(exception=None, command_count=0, last_feedback=None)
        def start(self):
            try:
                for _ in range(700):
                    clock.sleep(.01)
                    self.report.last_feedback = self.dev.snapshot()
                    item = self.step(self.report.last_feedback, clock(), .01)
                    if item is None:
                        break
                    q, meta = item
                    phases.append((clock(), meta['phase']))
                    self.dev.move_joints(q)
                    self.after(clock(), .01, q, meta)
                    self.report.command_count += 1
            except BaseException as error:
                self.report.exception = error
            return self
        def wait_stopped(self, *a): return True
        def join(self): return True
        def request_stop(self): pass
        def snapshot(self): return self.report
    monkeypatch.setattr('nero_core.two_second.execution.JointStreamSender', Sender)
    if synchronized:
        from nero_core.two_second.synchronized_reference import SynchronizedIntercept
        ref = SynchronizedIntercept(1., 1., join_s if short_follow == 0. else 2.2,
                                     np.zeros(3), np.zeros(3), np.zeros(3),
                                     [1, 0, 0, 0], [1, 0, 0, 0], prealign_s=join_s,
                                     follow_s=.8 if short_follow is None else short_follow,
                                     brake_s=brake_s)
    else:
        ref = SmoothIntercept(1., 1., 2., np.zeros(3), np.zeros(3), np.zeros(3),
                              [1, 0, 0, 0], [1, 0, 0, 0])
    path = TimedPath(lambda t, d=0: np.zeros(7), e.owner, 1, 'smooth_intercept_follow_brake',
                     ref.full_duration_s, 0.)
    if empty and not scheduled:
        with pytest.raises(PickRejected, match='夹持未确认'):
            e._stream(path, origin=1., epoch=0, reference=ref)
        assert not attachments and not any(p == 'brake' for _, p in phases)
    else:
        result = e._stream(path, origin=1., epoch=0, reference=ref)
        assert result['width_m'] == (0. if empty else .04)
        assert len(attachments) == (0 if empty else 1)
        if attachments:
            brake_time = next(t for t, p in phases if p == 'brake')
            assert (attachments[0][0] > brake_time) if short_follow is not None and short_follow < GRIP_CONFIRMATION_DWELL_S else (attachments[0][0] < brake_time)
        assert phases[-1][0]+.1-1. >= path.duration
        np.testing.assert_allclose(e.rest_anchor, path.q1)
        assert e.command_history.dispatch == phases[-1][0]
        if scheduled:
            event=next(event for event in e.events if event['event']=='intercept_attempt_finished')
            assert event['data']['grip_confirmed'] is (not empty)
            event=next(event for event in e.events if event['event']=='close_during_follow')
            assert not event['data']['feedback_would_allow_close']
            assert event['data']['reference_time_s'] >= ref.duration_s-1e-9
            assert any(phase == 'brake' for _, phase in phases)
            if short_follow == 0.:
                assert not any(phase == 'follow' for _, phase in phases)
                assert len([event for event in e.events if event['event']=='close_during_follow']) == 1
            if empty:
                assert runtime_stamps and all(stamp is None for stamp in runtime_stamps)


def test_smooth_candidate_budget_allows_second_grasp_and_shares_scene(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    clock = Clock(); planner.clock = clock
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=.1)
    calls = []
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, cancelled, **kw):
            calls.append(1)
            if len(calls) == 1:
                clock.sleep(.11)
            if cancelled():
                raise PickRejected('injected candidate budget cancellation')
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    installs, checks = [], []
    def install(target, state):
        planner.version += 1
        installs.append(target.identity)
    monkeypatch.setattr(planner, 'install', install)
    monkeypatch.setattr(planner, 'check_smooth_intercept', lambda *a, **kw: checks.append(a))
    result = planner.intercept(target, {'joints': np.zeros(7)})
    assert len(result.failures) == 1 and result.failures[0]['candidate_budget_exhausted']
    assert planner.attempt_diagnostics[-1]['candidate_index'] == 1
    assert len(installs) == len(checks) == 1
    assert result.path.diagnostics['profile'] == 'smooth'


def test_future_brake_certificate_restores_temporary_attachment_on_failure(monkeypatch):
    planner, _, target = planner_fixture(monkeypatch)
    planner.target = SimpleNamespace(points=np.zeros((5, 3)), velocity=np.array([.1, 0, 0]), capture_s=1.)
    old = np.ones((5, 3)); planner.payload_local = old
    marker = object(); planner.payload_tree = marker
    ref = SmoothIntercept(1., 2., 2., np.zeros(3), [.1, 0, 0], np.zeros(3), [1, 0, 0, 0], [1, 0, 0, 0])
    def failed(*a, **kw):
        assert planner.payload_local is not old
        raise PickRejected('injected brake collision')
    monkeypatch.setattr(planner, 'check_path', failed)
    with pytest.raises(PickRejected):
        planner.check_smooth_intercept(None, ref)
    assert planner.payload_local is old and planner.payload_tree is marker


def test_sparse_reference_mismatch_retries_same_candidate_at_dense_rate(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=1.5,
        planning_knot_hz=10., planning_knot_hz_fallback=50.)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs):
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    installs, counts = [], []
    def install(*args):
        installs.append(1)
        planner.version += 1
    monkeypatch.setattr(planner, 'install', install)
    def check(path, reference, **kwargs):
        counts.append(len(path.curve.x))
        if len(counts) == 1:
            raise PickRejected('关节曲线偏离指定 TCP 参考/方向')
        return {'samples': 1041, 'max_position_error_m': .00001}
    monkeypatch.setattr(planner, 'check_smooth_intercept', check)
    result = planner.intercept(target, {'joints': np.zeros(7)})
    assert len(installs) == 1 and len(counts) == 2 and counts[1] > counts[0]*4
    assert result.failures[0]['candidate_index'] == 0
    assert result.failures[0]['ik_knot_hz'] == 10.
    assert result.path.diagnostics['ik_knot_hz'] == 50.
    assert result.path.diagnostics['reference_check']['samples'] == 1041


def test_sparse_collision_tries_next_grasp_without_unnecessary_dense_retry(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=1.5,
        planning_knot_hz=10., planning_knot_hz_fallback=50.)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *args, **kwargs):
            return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *args: None)
    candidates=[]
    def check(path, reference, **kwargs):
        candidates.append(planner.candidate_context['candidate_index'])
        if len(candidates) == 1:
            raise PickRejected('机器人与实测工件点云碰撞')
    monkeypatch.setattr(planner, 'check_smooth_intercept', check)
    result=planner.intercept(target, {'joints': np.zeros(7)})
    assert candidates == [0, 1]
    assert result.path.diagnostics['ik_knot_hz'] == 10.


def test_geometry_budget_also_leaves_time_for_second_candidate(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    clock = Clock(); planner.clock = clock
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2.],
        brake_duration_s=1.5, candidate_planning_budget_s=1.5)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw): return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *a: None)
    calls = []
    def geometry(*a, cancelled):
        calls.append(1)
        if len(calls) == 1:
            clock.sleep(1.6)
        if cancelled():
            raise PickRejected('injected geometry budget cancellation')
    monkeypatch.setattr(planner, 'check_smooth_intercept', geometry)
    result = planner.intercept(target, {'joints': np.zeros(7)})
    assert len(calls) == 2 and len(result.failures) == 1
    assert result.failures[0]['candidate_budget_exhausted']


def test_longer_approach_recomputes_contact_instead_of_stretching_old_endpoint(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.raw.update(trajectory_profile='smooth', approach_duration_candidates_s=[2., 3.],
        brake_duration_s=1.5, candidate_planning_budget_s=1.5)
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw): return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    installs, checked = [], []
    monkeypatch.setattr(planner, 'install', lambda *a: installs.append(a))
    def check(path, reference, **kw):
        checked.append(reference)
        if reference.duration_s == 2.:
            raise PickRejected('injected short duration geometry rejection')
    monkeypatch.setattr(planner, 'check_smooth_intercept', check)
    result = planner.intercept(target, {'joints': np.zeros(7)})
    # Only the first two symmetric short-axis OBB candidates are attempted.
    assert len(installs) == 1 and len(result.failures) == 2
    assert result.reference.duration_s == 3.
    np.testing.assert_allclose(result.reference.contact_xyz-checked[0].contact_xyz, target.velocity)
    assert result.reference.contact_s-checked[0].contact_s == 1.


def test_synchronized_planner_retries_with_feasible_join_duration(monkeypatch):
    planner, model, target = planner_fixture(monkeypatch)
    planner.config.follow_bounds = np.array([-.2, .28])
    planner.config.settings = replace(planner.config.settings, full_plan_close_window_s=.4)
    planner.config.raw.update(trajectory_profile='smooth', intercept_strategy='synchronized',
        prealign_orientation_s=2., synchronized_duration_s=.2,
        synchronized_join_candidates_s=[2., 1.5], brake_duration_s=1.5,
        candidate_planning_budget_s=1.5)
    target.velocity = np.array([.2, 0., 0.])
    class Solver:
        def __init__(self, model): pass
        def solve(self, *a, **kw): return SimpleNamespace(success=True, joint_positions=np.zeros(7))
    monkeypatch.setattr('nero_core.two_second.smooth_planning.PrecisePoseSolver', Solver)
    monkeypatch.setattr(planner, 'install', lambda *a: None)
    monkeypatch.setattr(planner, 'check_smooth_intercept', lambda *a, **kw: None)
    result = planner.intercept(target, {'joints': np.zeros(7)})
    assert result.reference.prealign_s == 1.5
    assert len(result.failures) == 2  # short-axis candidates only
    assert all(f['failed_phase'] == 'synchronized_schedule' for f in result.failures)
    assert result.reference.contact_xyz[0] + .2*.4 <= .28


def test_route_worker_holds_command_without_sharing_kinematic_model():
    class RealClock:
        def __call__(self): return time.monotonic()
        def sleep(self, dt): time.sleep(dt)
    e, device, clock = executor_fixture(clock=RealClock())
    e.config.raw['trajectory_profile'] = 'smooth'
    e.rest_anchor = np.zeros(7)
    e.command_history.record(clock()-.02, .01, e.rest_anchor, {})
    def forbidden(*a): pytest.fail('hold sender must not share mutable planner model')
    e.planner.model.check_feedback = forbidden
    def route(state, *, start_joints, **kwargs):
        np.testing.assert_allclose(start_joints, e.rest_anchor)
        time.sleep(.055)
        e.planner.version += 1
        return TimedPath(lambda t, d=0: np.zeros(7), e.owner, e.planner.version, 'lift', 1., .055)
    e.planner.route = route
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = e._route(pool, device.snapshot(), stage='lift', goal=np.eye(4), joints=None,
                          carrying=False, epoch=0, version=e.version)
    assert result.stage == 'lift'
    assert len(device.commands) >= 3
    count = len(device.commands)
    time.sleep(.025)
    assert len(device.commands) == count and e.sender is None


def test_complete_smooth_cycle_uses_attachment_once_and_returns_to_wait(monkeypatch):
    from concurrent.futures import Future
    from nero_core.two_second.planning import InterceptPlan
    from nero_core.two_second.vision import MeasuredTarget
    from nero_core.two_second.vision import obb_grasps
    e, device, clock = executor_fixture()
    e.config.raw['trajectory_profile'] = 'smooth'
    calls, anchors, attachments, hold_stages = [], [], [], []
    class ImmediatePool:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def submit(self, fn, *a, **kw):
            future = Future()
            try: future.set_result(fn(*a, **kw))
            except BaseException as error: future.set_exception(error)
            return future
    class Sender:
        def __init__(self, dev, step, after, **kw):
            self.dev, self.step, self.after, self.kw = dev, step, after, kw
            self.report = SimpleNamespace(exception=None, running=True, command_count=0, last_feedback=None)
        def start(self):
            # A static hold has no trajectory clock; sample it once and keep
            # ownership until request_stop. Other paths run to their endpoint.
            hold = 'initial_dispatch_time' not in self.kw
            if hold: hold_stages.append(e.stage)
            clock.now = max(clock(), self.kw.get('initial_dispatch_time', clock()))
            try:
                for _ in range(1 if hold else 700):
                    clock.sleep(.01)
                    self.report.last_feedback = self.dev.snapshot()
                    item = self.step(self.report.last_feedback, clock(), .01)
                    if item is None:
                        self.report.running = False
                        break
                    q, meta = item
                    self.dev.move_joints(q)
                    self.after(clock(), .01, q, meta)
                    self.report.command_count += 1
            except BaseException as error:
                self.report.exception, self.report.running = error, False
            return self
        def snapshot(self): return self.report
        def wait_stopped(self, *a): return not self.report.running
        def join(self): return True
        def request_stop(self): self.report.running = False
    monkeypatch.setattr('nero_core.two_second.execution.ThreadPoolExecutor', ImmediatePool)
    monkeypatch.setattr('nero_core.two_second.execution.JointStreamSender', Sender)
    points = cloud()
    poses, box = obb_grasps(points, np.eye(4), np.eye(4))
    target = MeasuredTarget(e.owner, clock(), clock(), 100., 0, 1, np.zeros(3), np.zeros(3),
        tuple(poses), points, {'belt_region': {'plane_to_base': np.eye(4)}, 'obb': box},
        {}, {'source': 'nero7_xy_kalman'})
    def intercept(target, state, **kw):
        ref = SmoothIntercept(clock(), clock()+2., 2., np.zeros(3), np.zeros(3), np.zeros(3),
                              [1, 0, 0, 0], [1, 0, 0, 0])
        path = TimedPath(lambda t, d=0: np.zeros(7), e.owner, 1, 'smooth_intercept_follow_brake',
                         ref.full_duration_s, 0.)
        return InterceptPlan(path, ref, target, 1, np.eye(4), ())
    def route(state, *, stage, start_joints, **kw):
        calls.append(stage); anchors.append(start_joints.copy())
        e.planner.version += 1
        return TimedPath(lambda t, d=0: np.zeros(7), e.owner, e.planner.version, stage, .1, 0.)
    e.planner.intercept, e.planner.route = intercept, route
    e.planner.attach_measured = lambda state, stamp: attachments.append(stamp)
    e.planner.reanchor_support = lambda state: calls.append('reanchor')
    e.planner.validate_release_opening = lambda *a: None
    e.planner.release_measured = lambda *a: None
    report = e.run(target)
    assert calls == ['reanchor', 'lift', 'transfer', 'return']
    assert len(attachments) == 1
    np.testing.assert_allclose(anchors, np.zeros((3, 7)))
    assert e.owner is None and e.stage == 'waiting' and e.sender is None
    assert 'release' in hold_stages
    assert report['physical_grasp_success'] is None
    assert report['physical_bin_placement_verified'] is False


@pytest.mark.parametrize('failure', ['cancel', 'clock_reset', 'grip_lost'])
def test_stationary_hold_failure_ends_sender_before_next_route(failure):
    from nero_core.two_second.stationary_hold import StationaryHold
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    class RealClock:
        def __call__(self): return time.monotonic()
        def sleep(self, dt): time.sleep(dt)
    e, device, clock = executor_fixture(clock=RealClock())
    e.rest_anchor = np.zeros(7)
    device.width = .04; device.force = 2.; e.holding.held_width = .04
    with pytest.raises(PickRejected):
        with StationaryHold(e, JointStreamSender, epoch=0, carrying=True) as hold:
            time.sleep(.025)
            if failure == 'cancel': e.cancel()
            elif failure == 'clock_reset': e.epoch = lambda: 1
            else: device.width = 0.
            time.sleep(.025)
            hold.check()
    count = len(device.commands)
    time.sleep(.02)
    assert e.sender is None and len(device.commands) == count

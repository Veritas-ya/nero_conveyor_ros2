"""Failure regressions found by comparing deployed settings with physical simulation."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from test_two_second_hardware import site
from nero_core.two_second.config import CellConfig
from nero_core.two_second.feedback_gates import closure_feedback, GripConfirmation, RestArrival, OpenConfirmation
from nero_core.two_second.command_history import CommandHistory
from nero_core.two_second.smooth_reference import SmoothIntercept
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


@pytest.mark.parametrize('brake,budget', [(1.5, 1.5), (2., .3)])
def test_real_smooth_loader_preserves_bin_identity(site, brake, budget):
    path, data = site
    data.update(trajectory_profile='smooth', brake_duration_s=brake, candidate_planning_budget_s=budget)
    path.write_text(yaml.safe_dump(data))
    config = CellConfig.load(path)
    assert config.default_bin == data['unknown_category_bin'] == 'test_bin'
    np.testing.assert_allclose(config.bins[config.default_bin], np.eye(4))


def test_deployed_configuration_resolves_actual_destination_and_common_profile():
    config = CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    assert config.default_bin == 'common_bin'
    assert config.bins[config.default_bin].shape == (4, 4)
    assert config.settings.control_hz == 100
    assert config.raw['planning_knot_hz'] == 10
    assert config.raw['planning_knot_hz_fallback'] == 50
    assert config.raw['geometry_check_hz'] == 25


def test_closure_uses_actual_pose_time_not_advanced_arm_sample():
    ref = SmoothIntercept(0., 1., 2., np.zeros(3), np.zeros(3), np.zeros(3),
                          [1, 0, 0, 0], [1, 0, 0, 0])
    pose = np.eye(4)
    eligible, detail = closure_feedback(ref, pose, 2.95, pose, 2.94, command_elapsed=2.05)
    assert detail['drift_ok']
    assert not eligible  # the future arm command crossed contact; feedback did not
    eligible, _ = closure_feedback(ref, pose, 3., pose, 2.99, command_elapsed=2.1)
    assert eligible


def test_final_feedback_can_finish_dwell_at_brake_boundary():
    grip = GripConfirmation(.1, .2, 1.)
    assert not grip.observe(.04, 2., 1.1)
    assert not grip.observe(.04, 2., 1.199)
    assert grip.observe(.04, 2., 1.2)


@pytest.mark.parametrize('width,force', [(0., 2.), (.04, .1)])
def test_bad_gripper_sample_restarts_continuous_confirmation(width, force):
    grip = GripConfirmation(.1, .2, 1.)
    assert not grip.observe(.04, 2., 1.1)
    assert not grip.observe(width, force, 1.15)
    assert not grip.observe(.04, 2., 1.16)
    assert not grip.observe(.04, 2., 1.259)
    assert grip.observe(.04, 2., 1.26)


def test_repeated_gripper_feedback_never_accumulates_dwell():
    grip = GripConfirmation(.1, .2, 1.)
    for _ in range(100):
        assert not grip.observe(.04, 2., 1.1)
    with pytest.raises(PickRejected, match='时钟倒退'):
        grip.observe(.04, 2., 1.09)


def test_rest_arrival_requires_physical_end_time_and_new_stable_samples():
    gate = RestArrival(3.)
    q = np.zeros(7)
    assert not gate.observe(q, q, 2.95)
    assert not gate.observe(q, q, 3.)
    for _ in range(100):
        assert not gate.observe(q, q, 3.)
    assert not gate.observe(q, q, 3.05)
    assert gate.observe(q, q, 3.1)


def test_bad_joint_sample_resets_arrival_dwell():
    gate = RestArrival(3.); q = np.zeros(7)
    assert not gate.observe(q, q, 3.)
    assert not gate.observe(q+.006, q, 3.09)
    assert not gate.observe(q, q, 3.1)
    assert not gate.observe(q, q, 3.19)
    assert gate.observe(q, q, 3.2)


def test_open_confirmation_needs_new_width_feedback_for_original_dwell():
    gate = OpenConfirmation(.1)
    assert not gate.observe(.1, 1.)
    assert not gate.observe(.1, 1.)
    assert not gate.observe(.097, 1.1)
    assert not gate.observe(.099, 1.2)
    assert gate.observe(.099, 1.35)


def test_stacked_parts_use_a_physical_contact_chain_to_the_same_bin():
    from experiments.two_second_grasp.support_graph import support_chain
    # Two geoms belong to the same rigid lower part; no imaginary direct
    # contact between the upper part and the bin is needed.
    chain = support_chain(2, [100], {1: [10, 11], 2: [20]}, [(20, 10), (11, 100)])
    assert chain == [2, 1, 'container']
    assert support_chain(1, [100], {1: [10]}, [(10, 100)]) == [1, 'container']


def test_floating_contact_cycles_and_ineligible_support_parts_are_rejected():
    from experiments.two_second_grasp.support_graph import support_chain
    assert support_chain(2, [100], {1: [10], 2: [20]}, [(10, 20)]) is None
    assert support_chain(2, [100], {2: [20]}, [(20, 10), (10, 100)]) is None


def test_irregular_send_intervals_do_not_inflate_a_valid_acceleration():
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*2.,
        raw={'runtime_command_acceleration_check_enabled': True})
    history = CommandHistory(cfg)
    for elapsed in (0., .02, .025, .04, .05, .07):
        # Exact a=1.9 curve. Old /dt calculation reported 4.75 at t=.025.
        history.record(1.+elapsed, .01, np.ones(7)*(.5*1.9*elapsed**2), {})
    assert history.previous_interval == pytest.approx(.02)


def test_explicit_opt_in_still_rejects_command_acceleration_violation():
    cfg = SimpleNamespace(velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*2.,
        raw={'runtime_command_acceleration_check_enabled': True})
    history = CommandHistory(cfg)
    history.record(1., .01, np.zeros(7), {})
    history.record(1.01, .01, np.ones(7)*(.5*3.*.01**2), {})
    with pytest.raises(PickRejected, match='加速度'):
        history.record(1.02, .01, np.ones(7)*(.5*3.*.02**2), {})


def test_common_conditions_never_claim_a_calibrated_physical_twin():
    from experiments.two_second_grasp.hardware_conditions import HardwareConditions
    conditions = HardwareConditions(Path(__file__).parents[1]/
        'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    overrides = {'robot': {'robot': {}}, 'scene': {}}
    conditions.apply(overrides)
    assert overrides['scene']['simulation']['control_dt'] == .01
    assert overrides['robot']['robot']['controller']['joint_command_margin'] == .02
    env = SimpleNamespace(control_dt=.01, robot_controller=SimpleNamespace(
        max_joint_velocity=np.ones(7)*2., max_joint_acceleration=np.ones(7)*2.,
        command_joint_lower_limits=conditions.mirror.command_lower,
        command_joint_upper_limits=conditions.mirror.command_upper))
    report = conditions.report(env, False)
    assert report['arrival_tolerance_rad'] == pytest.approx(np.deg2rad(1.))
    assert report['arrival_dwell_s'] == .1
    assert not report['same_physical_cell']
    assert not report['hardware_gravity_compensation_verified']
    assert 'standby_joints_and_bin_pose' in report['unmatched']


def test_hardware_conditions_use_one_degree_for_simulated_arrival():
    from experiments.two_second_grasp.hardware_conditions import HardwareConditions
    conditions = HardwareConditions.__new__(HardwareConditions)
    conditions.arrivals = {}
    conditions.settings = SimpleNamespace(move_timeout_s=12.)
    actual = np.zeros(7); actual[4] = .009413863428370028
    conditions.env = SimpleNamespace(get_joint_positions=lambda: actual)
    assert not conditions.arrival('lift', np.zeros(7), 3., 3.)
    assert conditions.arrival('lift', np.zeros(7), 3., 3.1)
    actual[4] = np.deg2rad(1.01)
    with pytest.raises(PickRejected, match='到位条件下超时') as failure:
        conditions.arrival('transfer', np.zeros(7), 3., 16.)
    assert failure.value.details['tolerance_rad'] == pytest.approx(np.deg2rad(1.))


def moving_executor(costs):
    from scipy.interpolate import BPoly
    from test_two_second_hardware import executor_fixture
    from nero_core.two_second.trajectory import TimedPath
    e, device, clock = executor_fixture()
    e.config.velocity_limits[:] = 2.; e.config.acceleration_limits[:] = 2.
    curve = BPoly.from_derivatives([0., 2.], [[np.zeros(7), np.zeros(7), np.zeros(7)],
                                            [np.ones(7)*.3, np.zeros(7), np.zeros(7)]])
    count = [0]
    def feedback(state):
        clock.sleep(costs[count[0] % len(costs)]); count[0] += 1
    e.planner.model.check_feedback = feedback
    return e, device, clock, TimedPath(curve, e.owner, 1, 'variable_geometry_cost', 2., 0.)


def test_actual_dispatch_resampling_survives_variable_guard_cost_without_relaxing_limits():
    e, device, clock, path = moving_executor([.001, .005])
    # Before final resampling this exact example stopped at a late-sampled 2.118 rad/s²
    # command jump, despite the continuous curve peaking below 0.434 rad/s².
    e._stream(path, origin=1.1, epoch=0)
    assert len(device.commands) >= 200
    np.testing.assert_allclose(e.command_history.q, .3, atol=1e-12)
    np.testing.assert_allclose(e.command_history.velocity, 0., atol=1e-12)
    assert e.sender is None


def test_overrun_is_rechecked_before_the_next_send():
    e, device, clock, path = moving_executor([.025])
    e._stream(path, origin=1.1, epoch=0)
    assert device.commands and e.sender is None
    np.testing.assert_allclose(e.command_history.q, .3, atol=1e-12)


def test_expired_dispatch_is_reprepared_and_persistent_overrun_does_not_send():
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    from nero_core.two_second.streaming import CheckedWindowExpired
    from test_two_second_hardware import Clock, IdealDevice
    for persistent in (False, True):
        clock = Clock(); device = IdealDevice(clock); checks = []
        def step(state, stamp, dt):
            checks.append(stamp)
            # One nominal 10 ms window plus the observed 2.649 ms overrun;
            # the original total preparation cost was not recorded here.
            if persistent or len(checks) == 1:
                clock.sleep(.012649)
            return np.zeros(7), {'from': stamp, 'until': stamp+.01}
        def finalize(state, stamp, dt, q, meta):
            if not meta['from'] <= stamp <= meta['until']:
                raise CheckedWindowExpired('expired', details={'prepared_until': meta['until']})
            return q, meta
        def after(*args): sender.request_stop()
        sender = JointStreamSender(device, step, after, clock=clock, sleep=clock.sleep,
                                   finalize_send=finalize)
        # Repeated overruns must defeat any adaptive cost prediction.
        if persistent:
            def step(state, stamp, dt):
                checks.append(stamp); clock.sleep(.012649 * len(checks))
                return np.zeros(7), {'from': stamp, 'until': stamp+.01}
            sender._step = step
        sender.start(); sender.join(); result = sender.snapshot()
        retries = [e for e in result.events if e['event'] == 'joint_stream_window_retry']
        if persistent:
            assert isinstance(result.exception, CheckedWindowExpired)
            assert len(retries) == 3 and not device.commands
        else:
            assert result.exception is None and len(device.commands) == 1
            assert len(retries) == 1 and len(checks) == 2


def test_cancel_in_final_sampler_never_sends():
    from test_two_second_hardware import Clock, IdealDevice
    from mujoco_conveyor_sorting.runtime.joint_stream_sender import JointStreamSender
    clock = Clock(); device = IdealDevice(clock)
    holder = {}
    def finish(state, dispatch, dt, q, metadata):
        holder['sender'].request_stop()
        return q, metadata
    sender = JointStreamSender(device, lambda *a: (np.zeros(7), {}), finalize_send=finish,
                               clock=clock, sleep=clock.sleep)
    holder['sender'] = sender
    sender.start(); sender.join()
    assert not device.commands and sender.snapshot().command_count == 0


def test_can_joint_packets_are_interpolated_at_pose_time_without_extrapolation():
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    history = JointFeedbackHistory()
    packet_times = np.array([1., 1.001, 1.002, 1.003])
    axis_times = packet_times[[0, 0, 1, 1, 2, 2, 3]]
    velocity = np.arange(1., 8.)/10
    first = velocity*axis_times
    paired, _ = history.pair(first, packet_times, 1.005)
    assert paired is None
    latest = velocity*(axis_times+.01)
    paired, report = history.pair(latest, packet_times+.01, 1.005)
    np.testing.assert_allclose(paired, velocity*1.005, atol=1e-14)
    np.testing.assert_allclose(latest, velocity*(axis_times+.01))  # current input stays current
    assert report['available'] and not report['extrapolated']
    assert history.pair(latest, packet_times+.01, .99)[0] is None


def test_diagnostic_offset_uses_measured_history_and_never_changes_pairing():
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    history = JointFeedbackHistory()
    history.pair(np.zeros(7), [1.]*4, 1.)
    paired, _ = history.pair(np.ones(7), [1.01]*4, 1.008)
    diagnostic, evidence = history.at(1.006)
    np.testing.assert_allclose(paired, .8, atol=1e-12)
    np.testing.assert_allclose(diagnostic, .6, atol=1e-12)
    assert evidence['extrapolated'] is False
    assert history.at(1.02)[0] is None
    np.testing.assert_allclose(history.at(1.008)[0], paired, atol=1e-12)


def test_can_pairing_never_reuses_an_old_epoch_or_backward_clock():
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    history = JointFeedbackHistory()
    history.pair(np.zeros(7), [1.]*4, 1., epoch=0)
    assert history.pair(np.ones(7), [1.01]*4, 1.005, epoch=1)[0] is None
    assert history.pair(np.ones(7), [.5]*4, .9, epoch=1)[0] is None


@pytest.mark.parametrize('calibration_error_m', [0., .011])
def test_time_aligned_fk_keeps_latest_joints_and_does_not_mask_true_errors(calibration_error_m):
    from scipy.spatial.transform import Rotation
    from nero_core.two_second.planning import calibrated_model
    from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
    cfg = CellConfig.load(Path(__file__).parents[1]/
                          'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    model = calibrated_model(cfg)
    old = cfg.standby.copy(); latest = old.copy(); latest[0] += .04
    model.set_state(old, .1); flange = model.flange()
    pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')]
    state = {'joints': latest.tolist(), 'pose': pose.tolist(), 'width_m': .1,
             'pose_stamp': 1., 'joint_stamp': 1.02}
    with pytest.raises(PickRejected, match='不符') as error:
        model.check_feedback(state)  # 20 ms skew, despite all packets being fresh
    assert error.value.details['joint_source']=='latest_joint_feedback'
    assert error.value.details['orientation_tolerance_deg']==2.
    history = JointFeedbackHistory()
    history.pair(old, [1.]*4, 1.)
    paired, _ = history.pair(latest, [1.02]*4, 1.)
    state.update(joints_at_pose_stamp=paired, joints_at_pose_stamp_time=1.)
    state['pose'][0] += calibration_error_m
    if calibration_error_m:
        with pytest.raises(PickRejected, match='不符') as error:
            model.check_feedback(state)
        assert error.value.details['joint_source']=='interpolated_at_pose_receive_time'
        assert error.value.details['position_error_m']>error.value.details['position_tolerance_m']
    else:
        model.check_feedback(state)
    np.testing.assert_allclose(model.data.qpos[model.indices], latest, atol=1e-12)


def test_payload_prediction_uses_latest_joint_fk_timestamp():
    from nero_core.two_second.planning import MeasuredPlanner
    points = np.zeros((25, 3)); tcp = np.eye(4); tcp[0, 3] = .5
    model = SimpleNamespace(check_feedback=lambda state: None, tcp=lambda: tcp)
    planner = MeasuredPlanner(SimpleNamespace(), model=model)
    planner.target = SimpleNamespace(points=points, velocity=np.array([.1, 0., 0.]), capture_s=1.)
    planner.attach_measured({'joint_stamp': 2., 'pose_stamp': 1.98}, 1.98)
    np.testing.assert_allclose(planner.payload_local+tcp[:3, 3], points+[.1, 0., 0.])

"""Offline response plants; these tests are not Nero hardware evidence."""
import json

import numpy as np

from mujoco_conveyor_sorting.controllers.joint_response import (
    JointAxisResponse, JointResponseModel, fit_joint_response, predict_position_response,
)
from mujoco_conveyor_sorting.controllers.joint_trajectory import QuinticJointTrajectory


def _plant(command_time, command, observation_time, delay, tau):
    """Independent fine Euler plant used as a test oracle, not the fitted code."""
    step = .0002
    time = np.arange(command_time[0], observation_time[-1] + step, step)
    command_index = np.clip(np.searchsorted(command_time, time - delay,
                                          side="right") - 1, 0, len(command) - 1)
    state = np.empty_like(time)
    state[0] = command[0]
    for k in range(1, len(time)):
        state[k] = state[k - 1] + step / tau * (command[command_index[k - 1]] - state[k - 1])
    return np.interp(observation_time, time, state)


def _axis(delay=.06, tau=.12):
    return JointAxisResponse(delay, tau, 0., 0., .1, 500, 200, 1., 4., False)


def test_identifies_distinct_axes_and_keeps_later_holdout_separate():
    rng = np.random.default_rng(123)
    ct = np.arange(0., 10., .01)
    command = np.column_stack([.25 * np.sin((1.4 + i) * ct) +
                               .08 * np.sin((5.1 + i * .7) * ct) for i in range(3)])
    ft = np.arange(.01, 9.9, .014) + rng.uniform(-.001, .001, len(np.arange(.01, 9.9, .014)))
    delays, taus = [.04, .09, .02], [.075, .15, .035]
    feedback = np.column_stack([_plant(ct, command[:, i], ft, delays[i], taus[i])
                                for i in range(3)])
    feedback += rng.normal(0., 3e-5, feedback.shape)
    model, report = fit_joint_response(ct, command, ft, feedback,
        maximum_delay_s=.2, maximum_time_constant_s=.4,
        metadata={"runtime_eligible": True, "evidence": "synthetic_plant_only"})
    for i, axis in enumerate(model.axes):
        assert axis.status == "identified"
        assert abs(axis.delay_s - delays[i]) < .003
        assert abs(axis.time_constant_s - taus[i]) < .003
        assert axis.holdout_rmse_rad < .0004
        assert axis.holdout_rmse_rad < axis.unmodeled_holdout_rmse_rad / 50
        assert axis.holdout_samples > 100
    assert report["delay_is_encoder_internal"] is False
    assert JointResponseModel.from_dict(model.to_dict()) == model


def test_duplicate_packet_reads_are_not_added_as_measurements_and_constant_axis_unknown():
    ct = np.arange(0., 2., .01)
    q = np.full((len(ct), 2), .4)
    ft = np.repeat(ct, 2)
    measured = np.repeat(q, 2, axis=0)
    model, report = fit_joint_response(ct, q, ft, measured, maximum_delay_s=.1)
    assert model.axes == (None, None)
    assert report["axes"][0]["status"] == "unobservable_constant_command"
    assert report["axes"][0]["feedback_duplicate_timestamps_removed"] == len(ct)
    trajectory = QuinticJointTrajectory(np.zeros(7), np.ones(7), 2., 6.)
    missing = JointResponseModel((None,) * 7, {"runtime_eligible": True})
    q_original = trajectory.sample_full(.4)[0]
    q_preview, info = missing.preview_on_edge(trajectory, .4)
    np.testing.assert_array_equal(q_preview, q_original)
    assert not info["applied"]


def test_common_response_projection_improves_offline_tracking_and_arrival():
    delta = np.array([.4, -.6, .2, .3, -.5, .7, -.25])
    trajectory = QuinticJointTrajectory(np.zeros(7), delta, 2., 8., minimum_duration=1.8)
    model = JointResponseModel((_axis(),) * 7, {"runtime_eligible": True})
    time = np.arange(0., 4.01, .01)
    nominal = np.array([trajectory.sample_full(t - .5)[0] for t in time])
    preview = np.array([model.preview_on_edge(trajectory, t - .5)[0]
                        if t >= .5 else nominal[i] for i, t in enumerate(time)])
    baseline = _plant(time, nominal[:, 5], time, .06, .12)
    compensated = _plant(time, preview[:, 5], time, .06, .12)
    desired = nominal[:, 5]
    original_rmse = np.sqrt(np.mean((baseline - desired) ** 2))
    preview_rmse = np.sqrt(np.mean((compensated - desired) ** 2))
    original_arrival = time[np.flatnonzero(abs(baseline - delta[5]) < .005)[0]]
    preview_arrival = time[np.flatnonzero(abs(compensated - delta[5]) < .005)[0]]
    assert preview_rmse < .3 * original_rmse
    assert preview_arrival < original_arrival - .15
    # Projection remains on the exact certified chord, including endpoint hold.
    np.testing.assert_allclose(preview / delta, (preview[:, 5] / delta[5])[:, None] * np.ones((1, 7)),
                               atol=1e-14)
    assert np.min(preview / delta) >= 0.
    assert np.max(preview / delta) <= 1.


def test_asynchronous_axes_are_projected_with_visible_uncompensated_residual():
    trajectory = QuinticJointTrajectory(np.zeros(7), np.arange(1., 8.) / 10., 2., 8.)
    model = JointResponseModel(tuple(_axis(.01 * i, .03 + .015 * i) for i in range(7)),
                               {"runtime_eligible": True})
    candidate, info = model.preview_on_edge(trajectory, trajectory.duration * .5)
    inverse, _ = model.preview_command(trajectory.sample_full, trajectory.duration * .5)
    assert info["applied"]
    assert np.linalg.norm(info["projection_residual_rad"]) > .01
    assert not np.allclose(candidate, inverse)
    np.testing.assert_allclose(candidate / trajectory.q1,
                               np.full(7, candidate[0] / trajectory.q1[0]))
    weights = trajectory.q1 ** 2
    expected = weights @ np.array([a.delay_s + a.time_constant_s for a in model.axes]) / weights.sum()
    advance, _ = model.edge_preview_time_s(trajectory.q0, trajectory.q1)
    assert abs(advance - expected) < 1e-14


def test_proxy_and_unknown_metadata_always_keep_the_original_command():
    trajectory = QuinticJointTrajectory(np.zeros(7), np.ones(7), 2., 8.)
    for metadata in ({}, {"runtime_eligible": False, "timestamp_source": "old_log_proxy"}):
        model = JointResponseModel((_axis(),) * 7, metadata)
        original = trajectory.sample_full(.3)[0]
        command, info = model.preview_on_edge(trajectory, .3)
        np.testing.assert_array_equal(command, original)
        assert not info["applied"]
        assert model.edge_preview_time_s(trajectory.q0, trajectory.q1)[0] == 0.


def test_exact_irregular_zoh_matches_independent_fine_plant():
    ct = np.array([0., .31, .78, 1.1, 1.7])
    command = np.array([.2, .4, -.1, .3, .2])
    ft = np.array([0., .25, .32, .7, 1., 1.4, 1.9, 2.])
    exact = predict_position_response(ct, command, ft, command[0], .06, .12)
    np.testing.assert_allclose(exact, _plant(ct, command, ft, .06, .12), atol=.0005)


def test_trace_loader_uses_actual_dispatch_and_per_axis_receive_times(tmp_path):
    from nero_core.fit_joint_response import read_trace
    dispatch = np.array([1., 1.01, 1.02])
    packet_times = np.arange(21.).reshape(3, 7) * .001 + 1.
    np.savez(tmp_path / "execution_trace.npz",
             dispatch_end_monotonic_s=dispatch, dispatched_q_rad=np.ones((3, 7)),
             dispatch_succeeded=[True, False, True],
             feedback_joint_axis_sample_monotonic_s=packet_times,
             feedback_joint_sample_monotonic_s=np.zeros(3), feedback_q_rad=np.ones((3, 7)),
             sample_stage=np.array(["pregrasp"] * 3),
             feedback_sample_stage=np.array(["pregrasp"] * 3))
    (tmp_path / "execution_trace_summary.json").write_text(json.dumps({
        "clock_domain": "host_monotonic", "stamp_clock_domain": "host_wall",
        "feedback_timestamp_source": "SDK joint_stamps host CAN receive times"}))
    samples, evidence = read_trace(tmp_path)
    np.testing.assert_array_equal(samples["command_times_s"], dispatch[[0, 2]])
    np.testing.assert_array_equal(samples["feedback_times_s"], packet_times)
    assert evidence["runtime_eligible"]
    assert not evidence["encoder_internal_sample_time_available"]


def test_old_event_times_only_generate_sufficiency_report_by_default(tmp_path):
    from argparse import Namespace
    from nero_core.fit_joint_response import run
    events = [{"event": "joint_command", "time": 1., "stage": "pregrasp", "joints_rad": [0.] * 7},
              {"event": "joint_command", "time": 2., "stage": "pregrasp", "joints_rad": [1.] * 7},
              {"event": "feedback", "time": 1.2, "joint_stamp": 1.1, "joints": [.1] * 7}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    output = tmp_path / "fit"
    report = run(Namespace(recording=tmp_path, output=output, stage="pregrasp",
                           allow_proxy_times=False, diagnostic_only=False))
    assert report["status"] == "insufficient_precise_timestamps_report_only"
    assert not report["model_written"]
    assert not (output / "joint_response_model.json").exists()

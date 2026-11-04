"""Offline certificates for scene constraints added to the shared planner."""

from dataclasses import replace
import json
import threading

import mujoco
import numpy as np
import pytest

from mujoco_conveyor_sorting.controllers.planning_worker import (
    AsyncPlanningWorker,
    MuJoCoGlobalPlanningEngine,
    PlanningRequest,
)
from mujoco_conveyor_sorting.controllers.safe_path_planner import SafePathConfig


@pytest.fixture
def model():
    # A full-rank, seven-coordinate test manipulator. Cartesian slides make
    # the scene-obstacle locations exact; all IK, edge, continuation and RRT
    # code below is the production implementation, without hardware adapters.
    axes = (("slide", "1 0 0"), ("slide", "0 1 0"),
            ("slide", "0 0 1"), ("hinge", "1 0 0"),
            ("hinge", "0 1 0"), ("hinge", "0 0 1"),
            ("hinge", "0 0 1"))
    bodies = "".join(
        f'<body name="link{i}"><inertial pos="0 0 0" mass="1" '
        f'diaginertia="1 1 1"/><joint name="joint{i}" type="{kind}" '
        f'axis="{axis}" limited="true" range="-2 2"/>'
        for i, (kind, axis) in enumerate(axes, 1)
    )
    xml = ('<mujoco><compiler angle="radian"/><worldbody>' + bodies
           + '<site name="ee_site"/>' + '</body>' * 7
           + '</worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def make_engine(model, validator=None):
    return MuJoCoGlobalPlanningEngine(
        model, joint_names=[f"joint{i}" for i in range(1, 8)],
        ee_site_name="ee_site", ik_config={"max_iterations": 160},
        robot_collision_body_ids=range(1, model.nbody),
        state_validator=validator,
    )


def make_request():
    start = np.array([0., 0., .3, 0., 0., 0., 0.])
    goal = start.copy()
    goal[0] = .2
    return PlanningRequest(
        request_id=1, generation=0, stage="pregrasp", target_id="part",
        scene_epoch=3, target_poses=(np.array([.2, 0., .3, 1., 0., 0., 0.]),),
        start_q=start, full_qpos=start, goal_seeds=(goal,),
        lower_limits=np.full(7, -2.), upper_limits=np.full(7, 2.),
        minimum_tcp_height=0., minimum_joint_margin=.02,
        minimum_downward_alignment=-1., approach_axis_local=np.array([0., 0., 1.]),
        singularity_hard_stop_condition=45.,
        planner_config=SafePathConfig(normalized_edge_resolution=.005,
                                      execution_joint_step=.02,
                                      max_iterations=12, shortcut_attempts=0),
        planner_seed=9, trajectory_velocity_limits=np.ones(7),
        trajectory_acceleration_limits=np.ones(7),
        trajectory_minimum_duration=.01, tcp_speed_limits_enabled=False,
        rrt_iteration_slice=12, rrt_restart_count=1,
    )


def test_default_and_permissive_extension_return_identical_certificates(model):
    request = make_request()
    original = make_engine(model).plan(request, lambda: False)
    extended = make_engine(model, lambda _request, _q: True).plan(request, lambda: False)
    assert original.success and extended.success
    assert original.path_fully_validated and extended.path_fully_validated
    np.testing.assert_array_equal(original.path, extended.path)
    np.testing.assert_array_equal(original.edge_durations, extended.edge_durations)
    assert original.cost == extended.cost


def test_extension_rejects_ik_goal_before_a_route_can_be_certified(model):
    request = make_request()
    checked = []

    def target_obstacle(current_request, q):
        checked.append((current_request, q.copy()))
        return q[0] < .15

    result = make_engine(model, target_obstacle).plan(request, lambda: False)
    assert not result.success and not result.path_fully_validated
    assert checked and checked[0][0] is request
    assert any(q[0] > .15 for _, q in checked)
    assert result.diagnostics["rejection_counts"] == {"external_state_validator": 1}
    sample = result.diagnostics["rejection_samples"][0]
    assert sample["phase"] == "goal_search" and sample["check"] == "goal"
    assert result.diagnostics["ik_attempts"][0]["success"]


def test_valid_endpoints_do_not_hide_an_obstacle_inside_an_edge(model):
    request = make_request()
    engine = make_engine(model, lambda _request, q: not .08 < q[0] < .12)
    planner = engine._planner(request, seed=1, cancelled=lambda: False)
    goal = request.goal_seeds[0]
    assert planner.state_valid(request.start_q)
    assert planner.state_valid(goal)
    assert not planner.edge_valid(request.start_q, goal)
    assert planner.direct_path(request.start_q, goal) is None
    # An infinite separating slab cannot be bypassed by another RRT branch.
    result = engine.plan(request, lambda: False)
    assert not result.success and not result.path_fully_validated


def test_pregrasp_continuation_checks_scene_between_high_and_low_poses(model):
    high_request = make_request()
    request = replace(
        high_request,
        continuation_target_poses=(np.array([.2, 0., .1, 1., 0., 0., 0.]),),
        continuation_minimum_tcp_height=.05,
        continuation_cartesian_step=.02,
    )
    obstacle = lambda _request, q: not .17 < q[2] < .23
    engine = make_engine(model, obstacle)
    high_q = request.goal_seeds[0]
    low_q = high_q.copy()
    low_q[2] = .1
    assert engine._state_valid(request, high_q, lambda: False)
    assert engine._state_valid(request, low_q, lambda: False)
    assert make_engine(model).plan(request, lambda: False).success
    result = engine.plan(request, lambda: False)
    assert not result.success and not result.continuation_fully_validated
    samples = result.diagnostics["rejection_samples"]
    failed = next(item for item in samples if item["reason"] == "external_state_validator")
    assert failed["check"] == "continuation"
    assert failed["continuation_step"] <= failed["continuation_steps"]
    assert result.diagnostics["rejection_counts"]["continuation_state_rejected"] == 1


def test_extension_cannot_override_native_constraints_or_mutate_candidate(model):
    request = make_request()
    checked = []

    def permissive(_request, q):
        checked.append(q)
        assert not q.flags.writeable
        assert not np.shares_memory(q, request.start_q)
        return True

    engine = make_engine(model, permissive)
    assert not engine._state_valid(
        replace(request, minimum_tcp_height=.4), request.start_q, lambda: False)
    out_of_bounds = request.start_q.copy()
    out_of_bounds[0] = 2.
    assert not engine._state_valid(request, out_of_bounds, lambda: False)
    assert not checked
    assert engine._state_valid(request, request.start_q, lambda: False)
    assert len(checked) == 1


def test_validator_failure_is_reported_by_worker_instead_of_ignored(model):
    request = make_request()

    def broken(_request, _q):
        raise RuntimeError("scene snapshot unavailable")

    engine = make_engine(model, broken)
    with pytest.raises(RuntimeError, match="scene snapshot unavailable"):
        engine.plan(request, lambda: False)
    worker = AsyncPlanningWorker(engine)
    try:
        assert worker.submit(request)
        with worker._condition:
            assert worker._condition.wait_for(lambda: bool(worker._results), timeout=2.)
        result = worker.poll()
        assert result is not None and not result.success
        assert "scene snapshot unavailable" in result.error
        assert not result.path_fully_validated
    finally:
        assert worker.close()


def test_cancellation_during_scene_validation_does_not_publish_a_route(model):
    request = make_request()
    cancel = threading.Event()

    def slow_scene(_request, _q):
        cancel.set()
        return True

    result = make_engine(model, slow_scene).plan(request, cancel.is_set)
    assert result.cancelled
    assert not result.success and not result.path_fully_validated


def test_extension_must_be_callable(model):
    with pytest.raises(TypeError, match="state_validator must be callable"):
        make_engine(model, False)


@pytest.mark.parametrize("overrides, reason, measured_key, limit_key", [
    # The target is outside the accepted window, so bounded IK now fails
    # before producing the previously converged but unusable joint solution.
    ({"minimum_joint_margin": 1.8}, "ik_not_converged", "position_error_m", "iterations"),
    ({"minimum_tcp_height": .4}, "minimum_tcp_height", "tcp_height_m", "minimum_tcp_height_m"),
    ({"minimum_downward_alignment": 0.}, "downward_alignment", "alignment", "minimum_alignment"),
    ({"singularity_hard_stop_condition": 1.}, "jacobian_condition", "condition", "maximum_condition"),
])
def test_diagnostics_report_native_rejection_and_actual_values(
    model, overrides, reason, measured_key, limit_key,
):
    request = replace(make_request(), **overrides)
    result = make_engine(model).plan(request, lambda: False)
    assert not result.success
    assert result.diagnostics["rejection_counts"] == {reason: 1}
    sample = result.diagnostics["rejection_samples"][0]
    assert measured_key in sample and limit_key in sample
    assert sample["phase"] == "goal_search"
    json.dumps(result.diagnostics, allow_nan=False)


def test_failed_ik_keeps_its_original_residual_without_extra_solves(model, monkeypatch):
    request = replace(make_request(), target_poses=(np.array([5., 0., .3, 1., 0., 0., 0.]),))
    engine = make_engine(model)
    original = engine.ik.solve
    solves = []

    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        solves.append(result)
        return result

    monkeypatch.setattr(engine.ik, "solve", solve)
    result = engine.plan(request, lambda: False)
    assert not result.success and len(solves) == 1
    attempt = result.diagnostics["ik_attempts"][0]
    assert not attempt["success"]
    assert attempt["position_error_m"] == solves[0].position_error
    assert attempt["orientation_error_rad"] == solves[0].orientation_error
    assert attempt["iterations"] == solves[0].iterations
    assert result.diagnostics["rejection_counts"] == {"ik_not_converged": 1}


def test_external_validator_evidence_is_copied_and_request_local(model):
    def obstacle(_request, _q):
        obstacle.last_rejection = {"reason": "scene_collision", "geom": "palm"}
        return False

    engine = make_engine(model, obstacle)
    failed = engine.plan(make_request(), lambda: False)
    sample = failed.diagnostics["rejection_samples"][0]
    assert sample["validator_details"]["geom"] == "palm"
    obstacle.last_rejection["geom"] = "changed"
    engine._state_validator = lambda _request, _q: True
    passed = engine.plan(make_request(), lambda: False)
    assert passed.success and passed.diagnostics["rejection_counts"] == {}
    assert sample["validator_details"]["geom"] == "palm"


def test_forecast_rejection_is_distinguished_from_final_goal_search(model):
    request = replace(make_request(), intercept_velocity_m_s=np.array([.1, 0., 0.]),
                      intercept_motion_start_time_s=0., planned_target_time_s=1.)
    result = make_engine(model, lambda _request, _q: False).plan(request, lambda: False)
    assert not result.success
    samples = result.diagnostics["rejection_samples"]
    assert [item["phase"] for item in samples] == ["intercept_forecast", "goal_search"]


def test_exact_request_cache_preserves_all_checked_states_and_is_not_cross_request(model):
    request = make_request()
    before, after = [], []
    baseline = make_engine(model, lambda _r, q: before.append(q.tobytes()) or True)
    baseline._state_valid = baseline._check_state_valid
    expected = baseline.plan(request, lambda: False)
    engine = make_engine(model, lambda _r, q: after.append(q.tobytes()) or True)
    actual = engine.plan(request, lambda: False)
    assert expected.success and actual.success
    assert set(before) == set(after)
    assert len(after) < len(before)
    assert len(after) == len(set(after))
    np.testing.assert_array_equal(actual.path, expected.path)
    assert actual.cost == expected.cost
    assert actual.diagnostics['state_validation']['exact_reused'] > 0
    engine._state_validator = lambda _r, q: q[0] < .15
    assert not engine.plan(replace(request, scene_epoch=4), lambda: False).success


def test_exact_request_cache_distinguishes_height_and_honors_cancellation(model):
    request = make_request()
    engine = make_engine(model)
    engine._request_valid_states = set()
    engine._validation_cache_request = None
    engine._request_diagnostics = {'rejection_counts': {}, 'rejection_samples': []}
    assert engine._state_valid(request, request.start_q, lambda: False)
    assert not engine._state_valid(request, request.start_q, lambda: False, minimum_tcp_height=.4)
    assert not engine._state_valid(request, request.start_q, lambda: True)


def _timed_batch_request():
    return replace(make_request(),
        continuation_target_poses=(np.array([.2, 0., .1, 1., 0., 0., 0.]),),
        continuation_minimum_tcp_height=.05, continuation_cartesian_step=.02,
        timed_continuation_duration_s=.8,
        planner_config=replace(make_request().planner_config, normalized_edge_resolution=.001))


def test_curve_batch_preserves_original_states_and_complete_curve(model):
    checked, batches = [[], []], []
    engines, results = [], []
    for variant in (0, 1):
        def valid(_request, q, index=variant):
            checked[index].append(np.asarray(q).tobytes())
            return True
        def batch(_request, states, index=variant):
            assert not states.flags.writeable
            if engine._diagnostic_context.get('check') == 'timed_continuation':
                assert len(states) <= 32
            batches.append(np.array(states))
            checked[index].extend(q.tobytes() for q in states)
            return True
        valid.batch = batch
        engine = make_engine(model, valid)
        engine.batch_curve_validation = bool(variant)
        result = engine.plan(_timed_batch_request(), lambda: False)
        assert result.success and result.continuation_fully_validated
        assert result.continuation_trajectory is not None
        engines.append(engine); results.append(result)
    original, batched = results
    assert batches
    assert set(checked[0]) == set(checked[1])
    assert original.diagnostics['state_validation']['computed'] == batched.diagnostics['state_validation']['computed']
    assert batched.diagnostics['final_curve_batch_validation']['original_grid_states'] > 32
    np.testing.assert_array_equal(original.path, batched.path)
    np.testing.assert_array_equal(original.continuation_path, batched.continuation_path)
    for t in np.linspace(0., original.continuation_trajectory.duration, 101):
        for before, after in zip(original.continuation_trajectory.sample_full(t),
                                 batched.continuation_trajectory.sample_full(t)):
            np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('batched', [False, True])
def test_curve_batch_keeps_interior_curve_rejection(model, batched):
    engine = None
    def safe(q):
        # Isolate the actual timed-curve pass. The geometric IK polyline and
        # all route checks remain normal; both variants use this same oracle.
        return (engine._diagnostic_context.get('check') != 'timed_continuation'
                or not .17 < q[2] < .23)
    def valid(_request, q):
        return safe(q)
    valid.batch = lambda _request, states: all(safe(q) for q in states)
    engine = make_engine(model, valid)
    engine.batch_curve_validation = batched
    result = engine.plan(_timed_batch_request(), lambda: False)
    assert not result.success and not result.continuation_fully_validated
    assert result.continuation_trajectory is None
    assert result.diagnostics['rejection_counts']['timed_continuation_curve_rejected'] >= 1


def test_cancellation_inside_final_curve_batch_cannot_publish_curve(model):
    cancelled = threading.Event()
    def valid(_request, _q):
        return True
    def batch(_request, _states):
        cancelled.set()
        return True
    valid.batch = batch
    engine = make_engine(model, valid)
    result = engine.plan(_timed_batch_request(), cancelled.is_set)
    assert result.cancelled and not result.success
    assert not result.path_fully_validated
    assert result.continuation_trajectory is None

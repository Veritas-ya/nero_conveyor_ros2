"""Real IK/RRT regressions for redundant high approaches, without devices."""
from dataclasses import replace
import threading

import numpy as np
import pytest
import mujoco_conveyor_sorting.controllers.planning_worker as planning_module

from test_planning_worker_scene_validator import model, make_engine, make_request
from mujoco_conveyor_sorting.controllers.safe_path_planner import SafePathConfig


def redundant_request():
    original = make_request()
    near = original.goal_seeds[0]
    far = near.copy()
    far[5:] = [1.5, -1.5]  # Same physical TCP orientation, different arm posture.
    low = original.target_pose.copy()
    low[2] = .2
    return replace(original, goal_seeds=(far, near),
        continuation_target_poses=(low,), continuation_minimum_tcp_height=.05,
        planner_config=SafePathConfig(normalized_edge_resolution=.005,
            execution_joint_step=.05, max_iterations=300, shortcut_attempts=20))


def obstacle(_request, q):
    # The close branch needs a detour. A large wrist turn happens to provide
    # a direct edge through this synthetic scene, reproducing the old choice.
    return not (.07 < q[0] < .13 and q[1] < .06 and abs(q[5]) < .5)


def test_near_branch_gets_a_real_detour_before_the_far_direct_branch(model):
    request = redundant_request()
    old = make_engine(model, obstacle).plan(request, lambda: False)
    assert old.success
    assert np.max(np.abs(old.path[-1][5:])) > 1.
    engine = make_engine(model, obstacle)
    engine.rank_goal_routes = True
    result = engine.plan(request, lambda: False)
    assert result.success and result.path_fully_validated
    assert result.continuation_fully_validated
    np.testing.assert_allclose(result.path[-1], request.goal_seeds[1], atol=1e-9)
    selection = result.diagnostics['goal_route_selection'][0]
    assert selection['selected']['route_source'] == 'rrt'
    # Check interior states too; valid endpoints alone cannot pass this test.
    for a, b in zip(result.path, result.path[1:]):
        for t in np.linspace(0., 1., 101):
            assert obstacle(request, a+(b-a)*t)


def test_ranked_search_uses_a_near_direct_solution_even_if_far_seed_was_first(model):
    request = redundant_request()
    engine = make_engine(model)
    engine.rank_goal_routes = True
    result = engine.plan(request, lambda: False)
    assert result.success and result.continuation_fully_validated
    np.testing.assert_allclose(result.path[-1], request.goal_seeds[1], atol=1e-9)
    assert result.diagnostics['goal_route_selection'][0]['selected']['route_source'] == 'direct'


def test_forecast_direct_fast_return_keeps_complete_route_and_descent_certificates(model):
    request = redundant_request()
    request = replace(request, goal_seeds=tuple(reversed(request.goal_seeds)),
        planned_target_time_s=2.5, intercept_velocity_m_s=np.array([.02, 0., 0.]),
        intercept_motion_start_time_s=0., intercept_descent_duration_s=.8,
        timed_continuation_duration_s=.8)
    engine = make_engine(model)
    engine.rank_goal_routes = True
    result = engine.plan(request, lambda: False)
    assert result.success and result.path_fully_validated
    assert result.continuation_fully_validated and result.continuation_trajectory is not None
    assert result.diagnostics['goal_route_selection'][0]['selected']['route_source'] == 'forecast_direct'
    np.testing.assert_array_equal(result.path[-1], result.continuation_trajectory.q0)


def test_cancel_during_ranking_cannot_publish_a_preference_or_fallback(model):
    stop = threading.Event()
    engine = None
    def validator(_request, _q):
        if engine._diagnostic_context.get('phase') == 'rank_goal_routes':
            stop.set()
        return True
    engine = make_engine(model, validator)
    engine.rank_goal_routes = True
    result = engine.plan(redundant_request(), stop.is_set)
    assert result.cancelled and not result.success
    assert not result.path_fully_validated


def test_failed_preference_keeps_original_seed_search_available(model, monkeypatch):
    engine = make_engine(model)
    engine.rank_goal_routes = True
    # A preference can spend its opportunity budget without a complete route.
    # The original full search still gets the unchanged request and all seeds.
    seen = []
    def unfinished(request, _cancelled, _started, **_kwargs):
        seen.append(request)
        return None
    monkeypatch.setattr(engine, '_ranked_goal_route', unfinished)
    request = redundant_request()
    result = engine.plan(request, lambda: False)
    assert seen == [request] and result.success
    np.testing.assert_array_equal(result.path[-1], request.goal_seeds[0])


def test_no_motion_saving_budget_still_allows_original_rrt_to_complete(model):
    request = redundant_request()
    near = request.goal_seeds[1]
    other = near.copy()
    other[5:] = [.2, -.2]
    request = replace(request, goal_seeds=(near, other), rrt_iteration_slice=300)
    def wall(_request, q):
        return not (.07 < q[0] < .13 and q[1] < .06)
    engine = make_engine(model, wall)
    engine.rank_goal_routes = True
    result = engine.plan(request, lambda: False)
    assert result.success and result.path_fully_validated
    search = result.diagnostics['goal_route_selection'][0]
    assert search['selected'] is None
    assert all(trial['extra_search_budget_s'] == 0. for trial in search['trials'])
    assert result.continuation_fully_validated


def test_intercept_forecast_keeps_selected_route_duration_floor(model):
    request = replace(make_request(), planned_target_time_s=.5,
        intercept_velocity_m_s=np.array([.03, 0., 0.]),
        intercept_motion_start_time_s=2., intercept_descent_duration_s=.8)
    engine = make_engine(model)
    engine._request_diagnostics = {'ik_attempt_count': 0, 'ik_attempts': [],
                                   'rejection_counts': {}, 'rejection_samples': []}
    engine._diagnostic_context = {}
    forecast = engine._intercept_request(request, lambda: False, minimum_motion_s=2.5)
    assert forecast.planned_target_time_s >= 2.+2.5+.8
    shift = .03*(forecast.planned_target_time_s-request.planned_target_time_s)
    assert forecast.target_pose[0] == pytest.approx(request.target_pose[0]+shift)


def test_live_appointment_includes_existing_feedback_start_connector(model):
    request = replace(make_request(), planned_target_time_s=.5,
        intercept_velocity_m_s=np.array([.03, 0., 0.]),
        intercept_motion_start_time_s=2., intercept_descent_duration_s=.8,
        trajectory_minimum_duration=.18)
    engine = make_engine(model)
    engine.rank_goal_routes = True
    engine._request_diagnostics = {'ik_attempt_count': 0, 'ik_attempts': [],
                                   'rejection_counts': {}, 'rejection_samples': []}
    engine._diagnostic_context = {}
    forecast = engine._intercept_request(request, lambda: False, minimum_motion_s=2.5)
    assert forecast.planned_target_time_s >= 2.+2.5+.18+.8
    fixed = replace(request, fixed_high_continuation=True,
        continuation_target_poses=(request.target_pose,), continuation_minimum_tcp_height=0.,
        intercept_velocity_m_s=None)
    # A fixed-high descent starts from its given high state; no extra high
    # approach connector is invented for that separate job.
    assert engine._approach_time_forecast(fixed, 0.) == 0.


@pytest.mark.parametrize('overrun', [False, True])
def test_delivery_rebooking_reuses_selected_branch_and_actual_eta(model, monkeypatch, overrun):
    request = replace(redundant_request(), planned_target_time_s=.4,
        intercept_velocity_m_s=np.array([.02, 0., 0.]),
        intercept_motion_start_time_s=0., intercept_request_time_s=0.,
        intercept_descent_duration_s=.8, intercept_account_for_planning_time=True,
        timed_continuation_duration_s=.8)
    engine = make_engine(model)
    engine.rank_goal_routes = True
    native_forecast = engine._intercept_request
    native_plan = engine._plan
    plans, forecasts = [], []
    clock = [0.]
    if overrun:
        monkeypatch.setattr(planning_module.time, 'perf_counter', lambda: clock[0])
    def forecast(value, cancelled, **kwargs):
        forecasts.append((value, kwargs))
        if not kwargs:
            # Reproduce an initial estimate that did not yet know the final
            # route/branch. All route and revised forecast calculations below
            # still use the production implementation.
            return value
        return native_forecast(value, cancelled, **kwargs)
    def plan(value, cancelled):
        result = native_plan(value, cancelled)
        plans.append(result)
        if overrun:
            clock[0] += .3 if len(plans) == 2 else .1
        return result
    monkeypatch.setattr(engine, '_intercept_request', forecast)
    monkeypatch.setattr(engine, '_plan', plan)
    result = engine.plan(request, lambda: False)
    assert result.success and 2 <= len(plans) <= 3 and len(forecasts) == len(plans)
    value, kwargs = forecasts[1]
    np.testing.assert_array_equal(value.goal_seeds[0], plans[0].path[-1])
    assert kwargs['minimum_motion_s'] == plans[0].cost[0]
    assert result.diagnostics['intercept_delivery']['recertified']
    if overrun:
        assert len(plans) == 3
        delivery = result.diagnostics['intercept_delivery']
        assert delivery['delivered_with_approach_eta']
        assert delivery['refinements'][1]['compute_forecast_s'] == pytest.approx(.5)

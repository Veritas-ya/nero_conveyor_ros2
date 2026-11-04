"""Certified moving-target approach prefixes hide planning latency."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselineSortingPolicy, PolicyStage
from mujoco_conveyor_sorting.controllers.planning_worker import PlanningResult, MuJoCoGlobalPlanningEngine
from mujoco_conveyor_sorting.controllers.joint_trajectory import QuinticJointTrajectory
from mujoco_conveyor_sorting.controllers._math import quat_distance


class Worker:
    def __init__(self):
        self.requests = []
        self.results = []
        self.cancel_count = 0

    def submit(self, request):
        self.requests.append(request)
        return True

    def poll(self):
        return self.results.pop(0) if self.results else None

    def cancel_pending(self):
        self.cancel_count += 1

    def close(self):
        return True


@pytest.fixture
def policy(monkeypatch):
    env = SimpleNamespace(get_future_open_planning_snapshot=lambda q: {
        'arm_qpos': q.copy(), 'full_qpos': q.copy(), 'scene_epoch': 7})
    instance = BaselineSortingPolicy({'action_mode': 'joint_position',
        'dynamic_intercept_tracking': True, 'control_timestep': .01,
        'planner_waypoint_velocity_tolerance': .03}, env=env)
    instance._stage = PolicyStage.PREGRASP
    instance._target_id = 'part'
    instance._target = {'planned_grasp_time': 4.}
    instance._observed_scene_epoch = 7
    instance._safe_joint_path = [np.zeros(7), np.full(7, .2)]
    instance._safe_joint_path_index = 1
    instance._safe_path_target_pose = np.array([-.35, 0., .1, 1., 0., 0., 0.])
    instance._safe_path_stage = PolicyStage.PREGRASP
    instance._safe_path_generation = instance._planning_generation
    instance._safe_path_target_id = 'part'
    instance._safe_path_scene_epoch = 7
    instance._planning_worker = Worker()
    monkeypatch.setattr(instance, '_joint_safety_limits', lambda: (np.full(7, -3.), np.full(7, 3.)))
    monkeypatch.setattr(instance, '_planning_snapshot', lambda q: (q.copy(), q.copy(), 7))
    monkeypatch.setattr(instance, '_pregrasp_goal_seeds', lambda q, *_args, **_kwargs: (q.copy(),))
    yield instance
    instance.close()


def observation(q=None):
    return {'joint_pos': np.zeros(7) if q is None else np.asarray(q), 'joint_vel': np.zeros(7),
            'ee_pose': np.array([-.35, 0., .1, 1., 0., 0., 0.]), 'timestamp': 2.,
            'predictions': {'part': {'planned_grasp_time': 6.}}, 'scene_epoch': 7}


def future_pose():
    return np.array([-.35, .3, .1, 1., 0., 0., 0.])


@pytest.mark.parametrize('alignments', [(0.8, 0.3, 0., -0.5, -1.), (-1.,)])
def test_disabled_orientation_cone_has_no_directional_entry_dwell(policy, alignments):
    policy.config = replace(policy.config, acquisition_minimum_downward_alignment=-1.)
    # The unchanged default two-sample/hysteresis settings must not reintroduce
    # an orientation preference when the acquisition cone is fully disabled.
    assert policy.config.acquisition_entry_stable_steps == 2
    for alignment in alignments:
        assert policy._acquisition_entry_ready(alignment)


def test_enabled_acquisition_cone_keeps_original_threshold_hysteresis_and_dwell(policy):
    assert policy.config.acquisition_minimum_downward_alignment == .52
    assert policy.config.acquisition_entry_alignment_hysteresis == .003
    assert not policy._acquisition_entry_ready(.522)  # Below the entry threshold.
    assert not policy._acquisition_entry_ready(.524)  # First qualifying sample.
    assert policy._acquisition_entry_ready(.525)
    assert policy._acquisition_entry_ready(.521)  # Existing latch above the hard limit.
    assert not policy._acquisition_entry_ready(.519)  # Hard-limit violation clears it.
    assert not policy._acquisition_entry_ready(.526)
    assert not policy._acquisition_entry_ready(.525)  # Decrease restarts the dwell.
    assert not policy._acquisition_entry_ready(.527)
    assert policy._acquisition_entry_ready(.528)


def result(request):
    return PlanningResult(request_id=request.request_id, generation=request.generation,
        stage=request.stage, target_id=request.target_id, scene_epoch=request.scene_epoch,
        target_pose=request.target_pose, start_q=request.start_q,
        path=(request.start_q, np.full(7, .3)), cost=(1., .1, 2), edge_durations=(1.,),
        path_fully_validated=True, continuation_fully_validated=True)


def test_future_request_uses_certified_endpoint_and_updated_prediction_clock(policy):
    original_path = [q.copy() for q in policy._safe_joint_path]
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    anchor = original_path[-1]
    np.testing.assert_allclose(request.start_q, anchor)
    np.testing.assert_array_equal(request.full_qpos, request.start_q)
    np.testing.assert_array_equal(request.goal_seeds[0], request.start_q)
    assert request.planned_target_time_s == 6.  # Not the stale selected-target copy's 4 s.
    np.testing.assert_array_equal(policy._safe_joint_path, original_path)


def test_pending_or_completed_future_plan_does_not_stop_certified_prefix(policy, monkeypatch):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    worker = policy._planning_worker
    worker.results.append(result(worker.requests[-1]))
    executed = []
    monkeypatch.setattr(policy, '_execute_safe_joint_path', lambda q, _obs: executed.append(q.copy()) or q+.01)
    monkeypatch.setattr(policy, '_install_planning_result', lambda *_: pytest.fail('future anchor is not reached'))
    command = policy._arm_along_pregrasp_approach(future_pose(), observation())
    np.testing.assert_allclose(command, np.full(7, .01))
    assert len(executed) == 1
    assert policy._pregrasp_approach_result is not None


def test_independent_future_result_poll_adopts_visual_reservation_without_advancing_motion(policy, monkeypatch):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    new_pose = future_pose(); new_pose[1] += .02
    effective = replace(request, target_poses=(new_pose,), planned_target_time_s=6.5)
    completed = replace(result(effective), effective_request=effective)
    policy._planning_worker.results.append(completed)
    policy._grasp_pose = future_pose()
    policy._pregrasp_capture_high_active = True
    refinements = []
    def refine(target_id, duration, margin, **kwargs):
        refinements.append((target_id, duration, kwargs))
        return {'accepted': True, 'estimated_grasp_time': kwargs['requested_time'],
                'predicted_grasp_pose': kwargs['requested_pose']}
    policy.env.refine_selected_grasp_plan = refine
    original_path = [q.copy() for q in policy._safe_joint_path]
    original_index = policy._safe_joint_path_index
    original_started = policy._safe_edge_started_at
    monkeypatch.setattr(policy, '_execute_safe_joint_path',
                        lambda *_: pytest.fail('poll must not advance the joint trajectory'))
    monkeypatch.setattr(policy, '_install_planning_result',
                        lambda *_: pytest.fail('poll must not install a route before actual arrival'))
    monkeypatch.setattr(policy, '_arm_for_pose',
                        lambda *_: pytest.fail('poll must not solve arm motion'))

    assert policy.poll_pregrasp_approach_result(observation()) == 'checked'
    assert policy._pregrasp_approach_result is completed
    assert policy._pregrasp_approach_request is effective
    assert policy._planning_request is None
    assert policy._pregrasp_capture_time_s == 6.5
    np.testing.assert_array_equal(policy._pregrasp_approach_pose, new_pose)
    np.testing.assert_array_equal(policy._safe_joint_path, original_path)
    assert policy._safe_joint_path_index == original_index
    assert policy._safe_edge_started_at == original_started
    assert policy.stage is PolicyStage.PREGRASP
    assert len(refinements) == 1
    assert refinements[0][1] == 0.  # Exact worker reservation, no new motion ETA.
    assert refinements[0][2]['requested_time'] == 6.5
    # A second slow-side observation retains the completed result for the
    # ordinary executor to install later, without repeated adoption.
    assert policy.poll_pregrasp_approach_result(observation()) == 'checked'
    assert policy._pregrasp_approach_result is completed
    assert len(refinements) == 1


def test_independent_future_poll_does_not_submit_a_new_request_or_move_when_inactive(policy, monkeypatch):
    monkeypatch.setattr(policy, '_ensure_planning_worker',
                        lambda: pytest.fail('no pending future request'))
    assert policy.poll_pregrasp_approach_result(observation()) == 'inactive'
    assert policy._planning_worker.requests == []


def test_independent_cancelled_future_poll_keeps_current_certified_prefix(policy, monkeypatch):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    policy._planning_worker.results.append(replace(result(request), cancelled=True))
    original_path = [q.copy() for q in policy._safe_joint_path]
    monkeypatch.setattr(policy, '_execute_safe_joint_path',
                        lambda *_: pytest.fail('poll must not send a prefix sample'))
    assert policy.poll_pregrasp_approach_result(observation()) == 'prefix_only'
    assert policy._pregrasp_approach_request is None
    assert policy._pregrasp_approach_result is None
    np.testing.assert_array_equal(policy._safe_joint_path, original_path)


def test_independent_future_poll_retries_cancelled_request_while_prefix_keeps_playing(policy, monkeypatch):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    policy._planning_worker.results.append(replace(result(request), cancelled=True))
    path = policy._safe_joint_path
    generation = policy._planning_generation
    trajectory = policy._safe_edge_trajectory
    monkeypatch.setattr(policy, '_execute_safe_joint_path',
                        lambda *_: pytest.fail('async retry must not advance the active edge'))
    assert policy.poll_pregrasp_approach_result(observation()) == 'prefix_only'
    assert policy.poll_pregrasp_approach_result(observation()) == 'submitted'
    assert len(policy._planning_worker.requests) == 2
    assert policy._pregrasp_approach_request.request_id != request.request_id
    np.testing.assert_array_equal(policy._pregrasp_approach_request.start_q, path[-1])
    assert policy._safe_joint_path is path
    assert policy._planning_generation == generation
    assert policy._safe_edge_trajectory is trajectory


def test_independent_future_poll_keeps_original_retry_when_worker_temporarily_busy(policy, monkeypatch):
    policy._pregrasp_approach_pose = future_pose()
    worker = policy._planning_worker
    original_submit = worker.submit
    monkeypatch.setattr(worker, 'submit', lambda _request: False)
    assert policy.poll_pregrasp_approach_result(observation()) == 'pending_submission'
    assert policy._pregrasp_approach_request is None
    np.testing.assert_array_equal(policy._pregrasp_approach_pose, future_pose())
    monkeypatch.setattr(worker, 'submit', original_submit)
    assert policy.poll_pregrasp_approach_result(observation()) == 'submitted'
    assert policy._pregrasp_approach_request is worker.requests[-1]


def test_independent_future_poll_preserves_scene_invalidation(policy):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    policy._observed_scene_epoch = 8
    assert policy.poll_pregrasp_approach_result(observation()) == 'invalid'
    assert policy._safe_joint_path == []
    assert policy._pregrasp_approach_request is None
    assert policy._planning_worker.cancel_count == 1


def test_future_leg_is_admitted_only_at_measured_settled_anchor(policy, monkeypatch):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    policy._planning_worker.results.append(result(request))
    admitted = []
    def install(value, immutable, target, obs):
        admitted.append((value, immutable, obs['joint_pos'].copy()))
        policy._safe_joint_path = [q.copy() for q in value.path]
        policy._safe_path_target_pose = target.copy()
        return True
    monkeypatch.setattr(policy, '_install_planning_result', install)
    monkeypatch.setattr(policy, '_refine_pregrasp_intercept', lambda *_args, **_kwargs: (True, False))
    monkeypatch.setattr(policy, '_execute_safe_joint_path', lambda q, _obs: q+.01)
    obs = observation(request.start_q); obs['joint_vel'][:] = .1
    policy._arm_along_pregrasp_approach(future_pose(), obs)
    assert admitted == []
    obs['joint_vel'][:] = 0.
    policy._arm_along_pregrasp_approach(future_pose(), obs)
    assert len(admitted) == 1
    np.testing.assert_array_equal(admitted[0][2], request.start_q)
    assert policy._pregrasp_approach_pose is None


def test_scene_change_cancels_future_leg_and_its_prefix(policy):
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    worker = policy._planning_worker
    policy._observed_scene_epoch = 8
    command = policy._arm_along_pregrasp_approach(future_pose(), observation())
    np.testing.assert_array_equal(command, np.zeros(7))
    assert policy._safe_joint_path == []
    assert policy._pregrasp_approach_request is None
    assert worker.cancel_count == 1


def test_offline_continuation_resolution_does_not_depend_on_control_hz(policy):
    request = policy._make_planning_request(future_pose(), observation())
    assert request.planned_target_time_s == 6.
    assert request.continuation_cartesian_step == policy.config.max_translation_step
    assert request.continuation_cartesian_step > policy.config.cartesian_linear_speed*.01
    assert request.planner_config.normalized_edge_resolution == policy.config.planner_edge_resolution
    assert request.continuation_max_joint_step == policy.config.max_ik_joint_step


def test_future_clock_reserves_anchor_and_search_before_following_route(policy):
    policy._safe_path_eta_s = 1.8
    policy._pregrasp_last_planning_time_s = .9
    policy._pregrasp_capture_time_s = 8.9
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    assert request.intercept_motion_start_time_s == pytest.approx(3.8)
    assert request.planned_target_time_s == 8.9
    assert request.intercept_descent_duration_s == policy.config.descent_lead_time+policy.config.control_timestep


def test_retry_at_completed_prefix_has_no_remaining_travel_eta(policy):
    policy._safe_path_eta_s = 1.8
    anchor = policy._safe_joint_path[-1].copy()
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation(anchor))
    request = policy._planning_worker.requests[-1]
    np.testing.assert_array_equal(request.start_q, anchor)
    assert request.intercept_motion_start_time_s == 2.


def test_worker_forecasts_three_dimensional_clock_before_complete_certification(policy, monkeypatch):
    request = policy._make_planning_request(future_pose(), observation())
    request = replace(request, planned_target_time_s=.45,
        intercept_motion_start_time_s=0., intercept_request_time_s=0., intercept_descent_duration_s=.8,
        intercept_velocity_m_s=np.array([.06, -.08, .01]))
    engine = object.__new__(MuJoCoGlobalPlanningEngine)
    # A known, cheap branch lets this test isolate the clock/geometry contract.
    engine.state_evaluator = None
    engine.adaptive_segment_validation = False
    engine.rank_goal_routes = False  # Match the constructor default in this clock-only double.
    monkeypatch.setattr(engine, '_solve_goal_seed', lambda *_: np.full(7, .2))
    monkeypatch.setattr(engine, '_path_profile', lambda *_args, **_kw: ((1., .1, 2), (1.,)))
    certified = []
    def certify(effective, _cancelled):
        certified.append(effective)
        return result(effective)
    monkeypatch.setattr(engine, '_plan', certify)
    completed = engine.plan(request, lambda: False)
    effective = completed.effective_request
    assert certified == [effective]  # One complete final route, no estimate commands.
    assert effective.planned_target_time_s >= 1.8
    shift = request.intercept_velocity_m_s*(effective.planned_target_time_s-.45)
    np.testing.assert_allclose(effective.target_pose[:3], request.target_pose[:3]+shift)
    np.testing.assert_allclose(effective.continuation_target_poses[0][:3],
                               request.continuation_target_poses[0][:3]+shift)
    np.testing.assert_array_equal(effective.start_q, request.start_q)
    np.testing.assert_array_equal(effective.lower_limits, request.lower_limits)
    assert completed.path_fully_validated and completed.continuation_fully_validated


def test_effective_worker_clock_is_adopted_with_its_goal_before_install(policy):
    original = policy._make_planning_request(future_pose(), observation())
    effective_pose = future_pose(); effective_pose[1] += .1
    effective = replace(original, target_poses=(effective_pose,), planned_target_time_s=7.)
    completed = replace(result(effective), effective_request=effective)
    admissions = []
    policy.env.refine_selected_grasp_plan = lambda *_args, **kw: admissions.append(kw) or {'accepted': True}
    goal = future_pose()
    admitted = policy._adopt_pregrasp_effective_request(completed, original, goal, observation())
    assert admitted is effective
    assert admissions[0]['requested_time'] == 7.
    capture = effective_pose.copy(); capture[2] -= policy.config.approach_height
    capture[:3] += effective.intercept_velocity_m_s * policy.config.descent_lead_time
    np.testing.assert_array_equal(admissions[0]['requested_pose'], capture)
    np.testing.assert_array_equal(goal, effective_pose)
    assert policy._pregrasp_capture_time_s == 7.


def test_actual_worker_receipt_time_retimes_high_low_and_capture_before_execution(policy):
    policy._safe_path_eta_s = .5
    policy._pregrasp_capture_time_s = 2.5
    policy._grasp_pose = future_pose()
    policy._grasp_pose[2] -= policy.config.approach_height
    reserved = []
    def refine(_target, duration, _margin):
        timestamp = 2.+duration
        pose = policy._grasp_pose.copy()
        pose[1] -= .1*(timestamp-2.5)
        reserved.append(timestamp)
        return {'accepted': True, 'estimated_grasp_time': timestamp,
                'predicted_grasp_pose': pose}
    policy.env.refine_selected_grasp_plan = refine
    goal = future_pose()
    accepted, changed = policy._refine_pregrasp_intercept(goal, observation())
    assert accepted and changed
    assert reserved == [pytest.approx(2.+.5+policy.config.descent_lead_time
                                      +policy.config.control_timestep)]
    assert policy._pregrasp_capture_time_s == reserved[0]
    np.testing.assert_allclose(goal, policy._pregrasp_high_pose(policy._grasp_pose, observation()))
    assert goal[1] < future_pose()[1]
    assert policy._grasp_pose[1] < future_pose()[1]  # Low capture moves with its time.


def test_pending_future_route_holds_actual_settled_high_endpoint(policy, monkeypatch):
    """No result yet: waiting uses the original certified endpoint only."""
    assert policy._submit_pregrasp_approach_refresh(future_pose(), observation())
    request = policy._planning_worker.requests[-1]
    obs = observation(request.start_q)
    monkeypatch.setattr(policy, '_solve_pose', lambda *_a, **_k: pytest.fail('no low Cartesian request while planning'))
    command = policy._arm_along_pregrasp_approach(future_pose(), obs)
    np.testing.assert_array_equal(command, obs['joint_pos'])
    assert policy.stage is PolicyStage.PREGRASP
    assert not policy.drain_events()


def test_descent_clock_uses_current_reservation_instead_of_older_publication(policy):
    policy._pregrasp_capture_time_s = 8.
    target = {'object_id': 'part', 'estimated_grasp_time': 4., 'planned_grasp_time': 4.}
    obs = observation()
    obs['predictions']['part'].update(estimated_grasp_time=4., planned_grasp_time=4.)
    assert not policy._descent_time_ready(target, obs, 7.199)
    assert policy._descent_time_ready(target, obs, 7.2)


def test_descent_clock_without_internal_reservation_uses_planned_time(policy):
    policy._pregrasp_capture_time_s = None
    target = {'object_id': 'part', 'estimated_grasp_time': 4., 'planned_grasp_time': 8.}
    obs = observation()
    obs['predictions']['part'].update(estimated_grasp_time=4., planned_grasp_time=8.)
    assert not policy._descent_time_ready(target, obs, 7.199)
    assert policy._descent_time_ready(target, obs, 7.2)


def test_moving_high_and_low_waypoints_use_different_prediction_times(policy):
    velocity = np.array([.06, -.08, .01])
    obs = observation(); obs['belt_velocity'] = velocity
    policy.env.scene_config = {'workspace': {'dynamic_intercept': {'belt_direction': velocity.tolist()}}}
    policy._belt_velocity = lambda _obs: velocity
    capture = future_pose(); capture[2] -= policy.config.approach_height
    high = policy._pregrasp_high_pose(capture, obs)
    expected = capture.copy(); expected[:3] -= velocity * policy.config.descent_lead_time
    expected[2] += policy.config.approach_height
    np.testing.assert_allclose(high, expected)
    low, _height = policy._pregrasp_continuation_poses((high,), obs)
    np.testing.assert_allclose(low[0][:3], capture[:3] + velocity * policy.config.tracking_velocity_feedforward_time)


def test_tracking_time_alignment_uses_current_visual_pose_before_descent(policy):
    velocity = np.array([.06, -.08, .01])
    policy._belt_velocity = lambda _obs: velocity
    policy.env.scene_config = {'workspace': {'dynamic_intercept': {'tracking_horizon_s': .35, 'servo_lookahead_s': .04}}}
    current = future_pose(); current[1] -= .1
    predicted = current.copy(); predicted[:3] += velocity * .8
    obs = observation(); obs['predictions']['part'].update(current_estimated_pose=current,
        predicted_grasp_pose=predicted, estimated_grasp_time=2.8)
    tcp = predicted.copy(); tcp[2] += .007
    command = policy._tracking_command_target(tcp, obs, target={'object_id':'part'}, synchronize_to_intercept=True)
    expected = current.copy(); expected[2] += .007; expected[:3] += velocity * .04
    np.testing.assert_allclose(command, expected)


def test_failed_moving_goal_advances_capture_and_high_at_their_own_times(policy):
    velocity = np.array([.06, -.08, .01])
    policy._belt_velocity = lambda _obs: velocity
    policy.env.scene_config = {'workspace': {'dynamic_intercept': {
        'belt_direction': velocity.tolist(), 's_range': [-2., 2.], 'candidate_step_m': .04}}}
    policy._target.update(estimated_grasp_time=3., estimated_velocity=velocity)
    policy._pregrasp_capture_time_s = 6.
    policy._grasp_pose = future_pose(); policy._grasp_pose[2] -= policy.config.approach_height
    old_capture = policy._grasp_pose.copy()
    requests = []
    def refine(_id, _duration, _margin, **kw):
        requests.append(kw)
        return {'accepted': True, 'estimated_grasp_time': kw['requested_time'],
                'predicted_grasp_pose': kw['requested_pose'].copy()}
    policy.env.refine_selected_grasp_plan = refine
    high = policy._pregrasp_high_pose(old_capture, observation())
    assert policy._advance_dynamic_intercept(high, observation())
    expected = old_capture.copy(); expected[:3] += velocity * (.04 / np.linalg.norm(velocity))
    np.testing.assert_allclose(requests[0]['requested_pose'], expected)
    assert requests[0]['requested_time'] == pytest.approx(6.+.04/np.linalg.norm(velocity))
    np.testing.assert_allclose(high, policy._pregrasp_high_pose(expected, observation()))


def test_coasted_tracking_before_capture_keeps_same_time_visual_reference(policy):
    velocity = np.array([.06, -.08, .01])
    policy._belt_velocity = lambda _obs: velocity
    policy._stage = PolicyStage.TRACKING_DESCENT
    policy.env.scene_config = {'workspace': {'dynamic_intercept': {
        'belt_direction': velocity.tolist(), 'servo_lookahead_s': .04}}}
    current = future_pose(); current[2] -= policy.config.approach_height
    predicted = current.copy(); predicted[:3] += velocity * .8
    obs = observation()
    obs['predictions']['part'].update(current_estimated_pose=current,
        predicted_grasp_pose=predicted, planned_grasp_pose=predicted,
        estimated_grasp_time=2.8, planned_grasp_time=2.8,
        visible=False, track_observation_age=.1)
    target = {'object_id': 'part', 'visible': False, 'track_observation_age': .1}
    command = policy._tracking_command_target(predicted, obs, target=target,
        synchronize_to_intercept=True)
    expected = current.copy(); expected[:3] += velocity * .04
    np.testing.assert_allclose(command, expected)
    assert np.linalg.norm(predicted[:2]-command[:2]) > policy.config.coasted_tracking_xy_limit


@pytest.mark.parametrize('stage', [PolicyStage.TRACKING_DESCENT,
                                   PolicyStage.CLOSING, PolicyStage.RELEASING])
@pytest.mark.parametrize('tcp_limits_enabled', [True, False])
def test_moving_servo_does_not_accumulate_unexecuted_motion_with_lag(
        stage, tcp_limits_enabled):
    """A held/slower plant must not inherit the pending IK request's position."""
    env = SimpleNamespace(scene_config={}, get_release_pose=lambda _: None)
    instance = BaselineSortingPolicy({'dynamic_intercept_tracking': True,
        'tcp_speed_limits_enabled': tcp_limits_enabled, 'control_timestep': .01,
        'cartesian_linear_speed': .25}, env=env)
    try:
        instance._stage = stage
        measured = np.array([-.35, -.32, -.04, 1., 0., 0., 0.])
        target = measured.copy(); target[2] = -.065
        # An earlier IK succeeded, but joint limiting left this pose unexecuted.
        outstanding = measured.copy(); outstanding[:3] += [.015, -.035, -.04]
        instance._cartesian_command_pose = outstanding
        first = measured.copy()
        held_commands = []
        for tick in range(16):
            target[1] -= .1 * .01
            before = measured.copy()
            command = instance._cartesian_servo_pose(target,
                {'ee_pose': before, 'control_dt': .01})
            instance._cartesian_command_pose = command.copy()
            # Preserve the normal speed at 100 Hz despite actuator lag.
            assert np.linalg.norm(command[:3]-before[:3]) <= .0025 + 1e-12
            assert before[2] >= command[2] >= target[2]
            assert target[1] <= command[1] <= before[1]
            if tick < 3:
                held_commands.append(command.copy())
                np.testing.assert_array_equal(measured, first)
            else:
                measured[:3] += .2 * (command[:3]-measured[:3])
        assert max(abs(c[2]-first[2]) for c in held_commands) <= .0025
        assert first[2] > measured[2] > target[2]
        assert measured[1] < first[1]
    finally:
        instance.close()


def test_static_transit_preserves_cartesian_command_integration():
    instance = BaselineSortingPolicy({'dynamic_intercept_tracking': True})
    try:
        instance._stage = PolicyStage.TRANSFERRING
        measured = np.array([-.35, -.32, .02, 1., 0., 0., 0.])
        pending = measured.copy(); pending[0] += .02
        target = pending.copy(); target[0] += .01
        instance._cartesian_command_pose = pending
        command = instance._cartesian_servo_pose(target,
            {'ee_pose': measured, 'control_dt': .01})
        assert pending[0] < command[0] <= target[0]
        assert command[0]-measured[0] > .02
    finally:
        instance.close()


def pending_request(policy, *, velocity=None):
    """An initial high approach with no previously certified transit."""
    policy._safe_joint_path = []
    policy._safe_path_target_pose = None
    policy._grasp_pose = future_pose()
    policy._grasp_pose[2] -= policy.config.approach_height
    obs = observation()
    obs['belt_velocity'] = np.array([.0032060563423931044, -.07531041795643954,
        .001987612514741158]) if velocity is None else np.asarray(velocity)
    high = policy._pregrasp_high_pose(policy._grasp_pose, obs)
    request = policy._make_planning_request(high, obs)
    policy._planning_worker.submit(request)
    policy._planning_request = request
    return request, high, obs


def test_saved_velocity_revision_finishes_one_immutable_pending_request(policy):
    request, high, obs = pending_request(policy)
    obs['belt_velocity'] = np.array([.0007362065976551905, -.08347790009455082,
        .002237496039202137])
    updated = policy._pregrasp_high_pose(policy._grasp_pose, obs)
    assert np.linalg.norm(updated[:3]-high[:3]) == pytest.approx(.006829131201609563)
    assert not policy._request_matches_goal(request, updated)
    for _ in range(10):
        assert policy._request_async_global_path(updated, obs) == 'pending'
    assert policy._planning_request is request
    assert policy._planning_worker.cancel_count == 0
    assert len(policy._planning_worker.requests) == 1
    assert policy._planning_generation == request.generation
    assert len([event for event in policy.drain_events()
                if event.name == 'planning_pending_goal_update']) == 1
    assert policy.config.trajectory_replan_position_tolerance == .004


@pytest.mark.parametrize('change', ['static', 'stage', 'target', 'scene', 'generation', 'orientation'])
def test_pending_immutable_reservation_keeps_original_invalidations(policy, change):
    request, high, obs = pending_request(policy)
    high[1] += .007
    if change == 'static':
        policy.config = replace(policy.config, dynamic_intercept_tracking=False)
    elif change == 'stage':
        policy._stage = PolicyStage.TRANSFERRING
    elif change == 'target':
        policy._target_id = 'another-part'
    elif change == 'scene':
        policy._observed_scene_epoch += 1
    elif change == 'generation':
        policy._planning_generation += 1
    else:
        high[3:] = np.array([0., 0., 0., 1.])
    assert policy._request_async_global_path(high, obs) == 'pending'
    assert policy._planning_worker.cancel_count == 1
    assert policy._planning_request.request_id != request.request_id


@pytest.mark.parametrize('change', ['request_id', 'generation', 'stage', 'target_id', 'scene_epoch', 'start_q'])
def test_stale_effective_result_has_no_reservation_side_effect(policy, change):
    request, high, obs = pending_request(policy)
    values = {'request_id': request.request_id+100, 'generation': request.generation+1,
        'stage': 'returning', 'target_id': 'another-part', 'scene_epoch': 90,
        'start_q': request.start_q+.01}
    stale = replace(request, **{change: values[change]})
    effective = replace(stale, planned_target_time_s=8.)
    policy._planning_worker.results.append(replace(result(effective), effective_request=effective))
    policy.env.refine_selected_grasp_plan = lambda *_a, **_k: pytest.fail('stale result changed reservation')
    original_capture = policy._grasp_pose.copy()
    assert policy._request_async_global_path(high, obs) == 'pending'
    assert policy._planning_request is request
    np.testing.assert_array_equal(policy._grasp_pose, original_capture)
    assert policy._planning_stale_results == 1


@pytest.mark.parametrize('cancelled', [False, True])
def test_failed_or_cancelled_forecast_does_not_commit_capture(policy, monkeypatch, cancelled):
    request, high, obs = pending_request(policy)
    shifted = high.copy(); shifted[1] += .2
    effective = replace(request, target_poses=(shifted,), planned_target_time_s=8.)
    failure = replace(result(effective), path=None, cost=None, effective_request=effective,
        cancelled=cancelled, error='cancelled' if cancelled else 'global planner found no safe IK goal')
    policy._planning_worker.results.append(failure)
    policy.env.refine_selected_grasp_plan = lambda *_a, **_k: pytest.fail('failure committed forecast')
    original_capture = policy._grasp_pose.copy()
    capture_time = policy._pregrasp_capture_time_s
    monkeypatch.setattr(policy, '_activate_capture_high_fallback', lambda *_: False)
    monkeypatch.setattr(policy, '_advance_dynamic_intercept', lambda *_: False)
    status = policy._request_async_global_path(high, obs)
    assert status == ('pending' if cancelled else 'failed')
    np.testing.assert_array_equal(policy._grasp_pose, original_capture)
    assert policy._pregrasp_capture_time_s == capture_time
    assert policy._planning_request is not request


def test_current_completed_result_admission_refusal_releases_pending_request(policy):
    request, high, obs = pending_request(policy)
    effective = replace(request, planned_target_time_s=8.)
    policy._planning_worker.results.append(replace(result(effective), effective_request=effective))
    admissions = []
    def reject(*_a, **kwargs):
        admissions.append(kwargs)
        return {'accepted': False, 'reason': 'insufficient_intercept_time'}
    policy.env.refine_selected_grasp_plan = reject
    assert policy._request_async_global_path(high, obs) == 'pending'
    assert len(admissions) == 1
    assert policy._planning_request.request_id != request.request_id
    assert len(policy._planning_worker.requests) == 2
    assert policy._planning_worker.cancel_count == 0


def test_delivery_refines_live_capture_before_executing_certified_prefix(policy, monkeypatch):
    request, high, obs = pending_request(policy)
    effective = replace(request, planned_target_time_s=8.)
    policy._planning_worker.results.append(replace(result(effective), effective_request=effective))
    admissions = []
    def refine(_target, _duration, _margin, **kwargs):
        admissions.append(kwargs)
        if kwargs:
            return {'accepted': True}
        live_capture = policy._grasp_pose.copy(); live_capture[1] += .02
        return {'accepted': True, 'estimated_grasp_time': 8., 'predicted_grasp_pose': live_capture}
    policy.env.refine_selected_grasp_plan = refine
    def install(value, _request, pose, _obs):
        policy._safe_joint_path = [q.copy() for q in value.path]
        policy._safe_joint_path_index = 1
        policy._safe_path_target_pose = pose.copy()
        policy._safe_path_eta_s = 1.
        return True
    monkeypatch.setattr(policy, '_install_planning_result', install)
    new_high = []
    def continuation(pose, _obs):
        new_high.append(pose.copy())
        policy._pregrasp_approach_pose = pose.copy()
        return True
    monkeypatch.setattr(policy, '_submit_pregrasp_approach_refresh', continuation)
    executed = []
    monkeypatch.setattr(policy, '_execute_safe_joint_path', lambda q, _obs: executed.append(q.copy()) or q+.01)
    high[1] += .007  # The revised live high is not the worker's certificate.
    command = policy._arm_along_global_path(high, obs)
    assert len(admissions) == 2 and admissions[0] and not admissions[1]
    assert len(new_high) == len(executed) == 1
    assert np.linalg.norm(new_high[0][:3]-effective.target_pose[:3]) > .004
    np.testing.assert_array_equal(policy._safe_path_target_pose, effective.target_pose)
    np.testing.assert_allclose(command, obs['joint_pos']+.01)
    assert policy.stage is PolicyStage.PREGRASP


@pytest.mark.parametrize('missing', ['path', 'continuation'])
def test_immutable_delivery_does_not_skip_complete_certificate(policy, missing):
    request, high, obs = pending_request(policy)
    completed = replace(result(request), **{
        'path_fully_validated' if missing == 'path' else 'continuation_fully_validated': False})
    policy._planning_worker.results.append(completed)
    high[1] += .007
    assert policy._request_async_global_path(high, obs) == 'pending'
    assert policy._safe_joint_path == []
    assert policy._planning_request.request_id != request.request_id


def test_failed_future_forecast_keeps_original_intent_until_measured_anchor(policy, monkeypatch):
    high = future_pose()
    assert policy._submit_pregrasp_approach_refresh(high, observation())
    request = policy._pregrasp_approach_request
    shifted = high.copy(); shifted[1] += .1
    effective = replace(request, target_poses=(shifted,), planned_target_time_s=8.)
    failure = replace(result(effective), path=None, cost=None,
        effective_request=effective, error='global planner found no safe IK goal')
    policy._planning_worker.results.append(failure)
    policy.env.refine_selected_grasp_plan = lambda *_a, **_kw: pytest.fail('failed forecast is not a reservation')
    original_prefix = [q.copy() for q in policy._safe_joint_path]
    executed = []
    monkeypatch.setattr(policy, '_execute_safe_joint_path', lambda q, _obs: executed.append(q.copy()) or q+.01)
    for _ in range(2):
        command = policy._arm_along_pregrasp_approach(high, observation())
        np.testing.assert_allclose(command, np.full(7, .01))
    assert policy._planning_worker.cancel_count == 0
    assert policy._pregrasp_approach_request is request
    assert policy._pregrasp_approach_result is failure
    np.testing.assert_array_equal(policy._pregrasp_approach_pose, high)
    np.testing.assert_array_equal(policy._safe_joint_path, original_prefix)
    assert len(executed) == 2


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('stage', [PolicyStage.PREGRASP, PolicyStage.TRACKING_DESCENT,
    PolicyStage.CLOSING, PolicyStage.LIFTING, PolicyStage.TRANSFERRING,
    PolicyStage.RELEASING, PolicyStage.RETURNING])
def test_joint_only_high_timing_is_scoped_to_dynamic_pregrasp(policy, dynamic, stage):
    # Shared policy defaults preserve the original angular timing everywhere.
    assert policy.config.dynamic_pregrasp_global_tcp_speed_limits_enabled
    assert policy._global_path_tcp_speed_limits_enabled(stage)
    policy.config = replace(policy.config, dynamic_intercept_tracking=dynamic,
        dynamic_pregrasp_global_tcp_speed_limits_enabled=False)
    policy._stage = stage
    assert policy._global_path_tcp_speed_limits_enabled() == (
        not dynamic or stage != PolicyStage.PREGRASP)
    assert policy.config.tcp_speed_limits_enabled is True
    assert policy.config.singularity_hard_stop_condition == 45.


def test_joint_only_high_timing_uses_identical_worker_policy_and_rebased_profile(policy):
    policy.config = replace(policy.config,
        dynamic_pregrasp_global_tcp_speed_limits_enabled=False)
    calls = []
    def jacobian(q):
        calls.append(q.copy())
        return np.vstack([np.zeros((3, 7)), np.full((3, 7), 3.)])
    policy.env.ik = SimpleNamespace(compute_jacobian=jacobian)
    request = policy._make_planning_request(future_pose(), observation())
    assert request.tcp_speed_limits_enabled is False
    engine = object.__new__(MuJoCoGlobalPlanningEngine)
    engine.ik = policy.env.ik
    start = np.zeros(7); end = np.full(7, .2)
    expected = QuinticJointTrajectory(start, end,
        request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
        minimum_duration=request.trajectory_minimum_duration).duration
    worker_cost, worker_durations = engine._path_profile(request, [start, end])
    policy_cost, policy_durations = policy._joint_path_execution_profile([start, end])
    assert worker_cost[0] == pytest.approx(expected)
    assert policy_cost[0] == pytest.approx(expected)
    assert worker_durations == pytest.approx(policy_durations)
    assert not calls  # No hidden angular lower bound in either ETA.
    actual = start+.001
    rebased_expected = QuinticJointTrajectory(actual, end,
        request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
        minimum_duration=expected).duration
    # A future PREGRASP route may be admitted while the policy still returns.
    policy._stage = PolicyStage.RETURNING
    rebased = policy._realtime_rebased_edge_duration(start, actual, end,
        expected, stage=request.stage)
    assert rebased == pytest.approx(rebased_expected)
    assert policy._edge_execution_duration(start, end, stage=request.stage) == pytest.approx(expected)
    assert policy._realtime_rebased_edge_duration(start, actual, end, expected) > rebased
    # Local joint trajectories retain the overall TCP cap, including PREGRASP
    # when a caller did not request the certified global-route timing path.
    policy._stage = PolicyStage.PREGRASP
    assert policy._edge_minimum_duration(start, end) > expected
    assert calls


@pytest.mark.parametrize('phase', ['pregrasp', 'carry', 'retreat'])
def test_future_route_timing_uses_requested_phase_not_current_payload_stage(policy, monkeypatch, phase):
    policy.config = replace(policy.config,
        dynamic_pregrasp_global_tcp_speed_limits_enabled=False)
    policy._stage = PolicyStage.TRANSFERRING
    monkeypatch.setattr(policy, '_late_prefetch_pregrasp_target', lambda *_: None)
    monkeypatch.setattr(policy, '_tracking_branch_goal_seeds', lambda q, *_a, **_k: (q.copy(),))
    monkeypatch.setattr(policy, '_global_goal_seeds', lambda q, *_a, **_k: (q.copy(),))
    request = policy._future_prefetch_request(target_id='next', target_pose=future_pose(),
        start_q=np.zeros(7), full_qpos=np.zeros(7), scene_epoch=7,
        target={'object_id': 'next'}, phase=phase, observation=observation())
    assert request is not None
    assert request.tcp_speed_limits_enabled == (phase != 'pregrasp')
    assert policy._global_path_tcp_speed_limits_enabled() is True


def test_return_lookahead_high_timing_uses_future_unloaded_stage(policy, monkeypatch):
    policy.config = replace(policy.config,
        dynamic_pregrasp_global_tcp_speed_limits_enabled=False)
    policy._stage = PolicyStage.RETURNING
    monkeypatch.setattr(policy, '_global_goal_seeds', lambda q, *_a, **_k: (q.copy(),))
    request = policy._make_return_lookahead_request({'object_id': 'next'},
        future_pose(), observation())
    assert request.stage == 'pregrasp'
    assert request.tcp_speed_limits_enabled is False
    assert policy._global_path_tcp_speed_limits_enabled() is True


@pytest.mark.parametrize('stage', [PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING,
    PolicyStage.LIFTING, PolicyStage.TRANSFERRING, PolicyStage.RELEASING])
def test_joint_only_high_timing_does_not_relax_local_tcp_commands(policy, stage):
    policy.config = replace(policy.config,
        dynamic_pregrasp_global_tcp_speed_limits_enabled=False,
        cartesian_linear_speed=.25, cartesian_angular_speed=.8)
    policy._stage = stage
    current = np.array([0., 0., .2, 1., 0., 0., 0.])
    target = np.array([.1, .1, .1, np.cos(.2), 0., np.sin(.2), 0.])
    command = policy._cartesian_servo_pose(target, {'ee_pose': current, 'control_dt': .01})
    assert np.linalg.norm(command[:3]-current[:3]) <= .0025+1e-12
    assert quat_distance(command[3:], current[3:]) <= .008+1e-12


def test_legacy_overall_tcp_disable_still_selects_original_unlimited_profile(policy):
    policy.config = replace(policy.config, tcp_speed_limits_enabled=False)
    for stage in PolicyStage:
        assert policy._global_path_tcp_speed_limits_enabled(stage) is False


def test_wall_clock_delivery_recertifies_shifted_pose_and_includes_full_cost(policy, monkeypatch):
    from mujoco_conveyor_sorting.controllers import planning_worker as module
    request = policy._make_planning_request(future_pose(), observation())
    request = replace(request, planned_target_time_s=.45,
        intercept_motion_start_time_s=0., intercept_request_time_s=0., intercept_descent_duration_s=.8,
        intercept_velocity_m_s=np.array([.06, -.08, .01]),
        intercept_account_for_planning_time=True, intercept_motion_time_scale=1.5)
    engine = object.__new__(MuJoCoGlobalPlanningEngine)
    clock = [0.]
    engine.state_evaluator = None
    engine.adaptive_segment_validation = False
    engine.rank_goal_routes = False
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(engine, '_solve_goal_seed', lambda *_: np.full(7, .2))
    monkeypatch.setattr(engine, '_path_profile', lambda *_a, **_kw: ((1., .1, 2), (1.,)))
    certified = []
    def certify(effective, _cancelled):
        certified.append(effective)
        clock[0] += .6
        return replace(result(effective), cost=(1., .1, 2), edge_durations=(1.,))
    monkeypatch.setattr(engine, '_plan', certify)
    completed = engine.plan(request, lambda: False)
    assert len(certified) == 2
    effective = completed.effective_request
    assert effective is certified[-1]
    assert effective.planned_target_time_s == pytest.approx(1.2+1.5+.8)
    assert completed.planning_time_s == pytest.approx(1.2)
    assert completed.diagnostics['intercept_delivery']['recertified']
    shift = request.intercept_velocity_m_s*(effective.planned_target_time_s-.45)
    np.testing.assert_allclose(effective.target_pose[:3], request.target_pose[:3]+shift)
    np.testing.assert_allclose(completed.target_pose, effective.target_pose)


def test_failed_delivery_revision_keeps_original_certificate(policy, monkeypatch):
    from mujoco_conveyor_sorting.controllers import planning_worker as module
    request = policy._make_planning_request(future_pose(), observation())
    request = replace(request, planned_target_time_s=.45,
        intercept_motion_start_time_s=0., intercept_request_time_s=0., intercept_descent_duration_s=.8,
        intercept_velocity_m_s=np.array([.06, -.08, .01]), intercept_account_for_planning_time=True)
    engine = object.__new__(MuJoCoGlobalPlanningEngine)
    clock = [0.]; certified = []
    engine.state_evaluator = None
    engine.adaptive_segment_validation = False
    engine.rank_goal_routes = False
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(engine, '_solve_goal_seed', lambda *_: np.full(7, .2))
    monkeypatch.setattr(engine, '_path_profile', lambda *_a, **_kw: ((1., .1, 2), (1.,)))
    def certify(effective, _cancelled):
        certified.append(effective); clock[0] += .6
        value = replace(result(effective), cost=(1., .1, 2), edge_durations=(1.,))
        return value if len(certified) == 1 else replace(value, path=(), error='unreachable')
    monkeypatch.setattr(engine, '_plan', certify)
    completed = engine.plan(request, lambda: False)
    assert completed.success
    assert completed.effective_request is certified[0]
    assert completed.diagnostics['intercept_delivery']['revision_failed']


def test_cancelling_delivery_revision_cannot_publish_original_route(policy, monkeypatch):
    from mujoco_conveyor_sorting.controllers import planning_worker as module
    request = replace(policy._make_planning_request(future_pose(), observation()),
        planned_target_time_s=.45, intercept_motion_start_time_s=0.,
        intercept_request_time_s=0., intercept_descent_duration_s=.8,
        intercept_velocity_m_s=np.array([.06, -.08, .01]), intercept_account_for_planning_time=True)
    engine = object.__new__(MuJoCoGlobalPlanningEngine)
    clock = [0.]; count = [0]; cancelled = [False]
    engine.state_evaluator = None
    engine.adaptive_segment_validation = False
    engine.rank_goal_routes = False
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(engine, '_solve_goal_seed', lambda *_: np.full(7, .2))
    monkeypatch.setattr(engine, '_path_profile', lambda *_a, **_kw: ((1., .1, 2), (1.,)))
    def certify(effective, _cancelled):
        count[0] += 1; clock[0] += .6
        if count[0] == 2: cancelled[0] = True
        return replace(result(effective), cost=(1., .1, 2), edge_durations=(1.,))
    monkeypatch.setattr(engine, '_plan', certify)
    completed = engine.plan(request, lambda: cancelled[0])
    assert completed.cancelled and not completed.success
    assert completed.path is None

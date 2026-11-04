from __future__ import annotations

import numpy as np
import pytest

from mujoco_conveyor_sorting.perception import MotionPredictor


def test_constant_velocity_prediction() -> None:
    predictor = MotionPredictor(
        model="constant_velocity",
        max_prediction_horizon=2.0,
    )
    history = [
        {"timestamp": 0.0, "position": [0.0, 0.0, 0.2]},
        {"timestamp": 0.1, "position": [0.02, 0.0, 0.2]},
        {"timestamp": 0.2, "position": [0.04, 0.0, 0.2]},
    ]
    result = predictor.predict(
        history,
        current_time=0.3,
        system_latency=0.1,
        robot_arrival_time=0.5,
        belt_velocity=0.2,
    )
    assert result.observed_timestamp == 0.2
    assert result.current_estimated_position[0] == pytest.approx(0.06, abs=1e-6)
    assert result.predicted_position[0] == pytest.approx(0.16, abs=1e-6)


def test_compensation_changes_grasp_target(env) -> None:
    action = {"arm": env.get_joint_positions(), "gripper": 1.0}
    observation = env.get_observation()
    for _ in range(10):
        observation, *_ = env.step(action)
    object_id = observation["objects"][0]["object_id"]
    env.enable_motion_compensation = True
    compensated = env.get_predicted_grasp_pose(object_id)
    env.enable_motion_compensation = False
    uncompensated = env.get_predicted_grasp_pose(object_id)
    assert compensated is not None and uncompensated is not None
    assert compensated[0] > uncompensated[0] + 0.02


def test_fast_belt_arrival_is_measured_to_fixed_intercept_station(env) -> None:
    """An early upstream detection must not inflate the robot route ETA."""

    env.belt_speed_jitter_enabled = False
    observation = env.reset(
        seed=7,
        options={"belt_speed": 0.30, "initial_objects": 1},
    )
    action = {"arm": env.get_joint_positions(), "gripper": 1.0}
    for _ in range(40):
        observation, *_ = env.step(action)

    object_id = observation["objects"][0]["object_id"]
    prediction = observation["predictions"][object_id]
    preferred_x = env.scene_config["workspace"]["preferred_grasp_x"]

    assert prediction["prediction_ready"]
    assert prediction["grasp_window"]["feasible"]
    assert prediction["grasp_window"]["reason"] == "ok"
    assert prediction["predicted_grasp_pose"][0] == pytest.approx(
        preferred_x, abs=0.01
    )
    assert (
        prediction["estimated_grasp_time"]
        <= prediction["grasp_window"]["end_time"]
    )

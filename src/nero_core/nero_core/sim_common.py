"""Shared CLI helpers; kept outside the environment's control loop."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mujoco_conveyor_sorting import ConveyorSortingEnv
from mujoco_conveyor_sorting.controllers import BaselineSortingPolicy


def make_policy(env: ConveyorSortingEnv, compensation: bool) -> BaselineSortingPolicy:
    quaternion = tuple(
        env.robot_config.get("end_effector", {}).get(
            "nominal_down_quaternion_wxyz", [0.0, 1.0, 0.0, 0.0]
        )
    )
    trajectory = env.robot_config.get("trajectory", {})
    return BaselineSortingPolicy(
        {
            "action_mode": env.action_type,
            "enable_motion_compensation": compensation,
            "dynamic_intercept_tracking": bool(env.scene_config.get("workspace", {}).get("dynamic_intercept", {}).get("enabled", False)),
            "grasp_quaternion": quaternion,
            # The physical baseline must react to empty grasps and drops; the
            # environment confirms a grasp only after bilateral finger contact.
            "require_grasp_confirmation": True,
            "position_tolerance": 0.022,
            "enable_time_parameterization": bool(
                trajectory.get("enabled", True)
            ),
            "trajectory_minimum_duration": float(
                trajectory.get("minimum_duration", 0.18)
            ),
            "track_belt_during_close": bool(
                trajectory.get("track_belt_during_close", True)
            ),
            "close_lead_time": float(
                trajectory.get("close_lead_time", 0.18)
            ),
            "closure_contact_lead_time": float(
                trajectory.get("closure_contact_lead_time", 0.0)
            ),
            "confirmed_grasp_settle_time": float(
                trajectory.get("confirmed_grasp_settle_time", 0.25)
            ),
            "contact_hold_squeeze_fraction": float(
                trajectory.get("contact_hold_squeeze_fraction", 0.08)
            ),
            "contact_hold_max_relaxation_fraction": float(
                trajectory.get(
                    "contact_hold_max_relaxation_fraction", 1.0
                )
            ),
            "minimum_grasp_normal_force": float(
                trajectory.get("minimum_grasp_normal_force", 0.0)
            ),
            "grasp_force_preload_rate": float(
                trajectory.get("grasp_force_preload_rate", 1.0)
            ),
            "grasp_force_settle_time": float(
                trajectory.get("grasp_force_settle_time", 0.06)
            ),
            "carrying_velocity_scale": float(
                trajectory.get("carrying_velocity_scale", 0.65)
            ),
            "carrying_acceleration_scale": float(
                trajectory.get("carrying_acceleration_scale", 0.50)
            ),
            "gentle_carry_categories": tuple(
                int(item)
                for item in trajectory.get(
                    "gentle_carry_categories", [3]
                )
            ),
            "trajectory_velocity_limits": tuple(
                env.robot_config.get("velocity_limits", [1.4] * 7)
            ),
            "trajectory_acceleration_limits": tuple(
                env.robot_config.get("acceleration_limits", [3.0] * 7)
            ),
            "trajectory_replan_position_tolerance": float(
                trajectory.get("pose_replan_position_tolerance", 0.004)
            ),
            "trajectory_replan_orientation_tolerance": float(
                trajectory.get("pose_replan_orientation_tolerance", 0.025)
            ),
            "safe_transfer_height": float(
                trajectory.get("safe_transfer_height", 0.43)
            ),
            "lift_clearance_height": float(
                trajectory.get("lift_clearance_height", 0.0)
            ),
            "release_retreat_distance": float(
                trajectory.get("release_retreat_distance", 0.055)
            ),
            "prefetch_handoff_retreat_distance": float(
                trajectory.get(
                    "prefetch_handoff_retreat_distance", 0.020
                )
            ),
            "cartesian_linear_speed": float(
                trajectory.get(
                    "cartesian_linear_speed",
                    env.robot_config.get("controller", {}).get(
                        "nominal_cartesian_speed", 0.28
                    ),
                )
            ),
            "cartesian_angular_speed": float(
                trajectory.get("cartesian_angular_speed", 1.20)
            ),
            "maximum_tcp_angular_speed": float(
                trajectory.get("maximum_tcp_angular_speed", 1.85)
            ),
            "tcp_speed_limits_enabled": bool(
                trajectory.get("tcp_speed_limits_enabled", True)
            ),
            "descent_lead_time": float(
                trajectory.get("descent_lead_time", 0.80)
            ),
            "pregrasp_blend_horizontal_tolerance": float(
                trajectory.get(
                    "pregrasp_blend_horizontal_tolerance", 0.035
                )
            ),
            "pregrasp_blend_vertical_clearance": float(
                trajectory.get("pregrasp_blend_vertical_clearance", 0.060)
            ),
            "pregrasp_blend_orientation_tolerance": float(
                trajectory.get(
                    "pregrasp_blend_orientation_tolerance", 0.20
                )
            ),
            "tracking_velocity_feedforward_time": float(
                trajectory.get("tracking_velocity_feedforward_time", 0.16)
            ),
            "tracking_synchronization_horizon": float(
                trajectory.get("tracking_synchronization_horizon", 0.35)
            ),
            "closing_tracking_lead_time": float(
                trajectory.get("closing_tracking_lead_time", 0.04)
            ),
            "max_ik_joint_step": float(
                trajectory.get("max_ik_joint_step", 0.25)
            ),
            "max_verified_ik_transition": float(
                trajectory.get("max_verified_ik_transition", 0.35)
            ),
            "approach_axis_local": tuple(
                env.robot_config.get("end_effector", {}).get(
                    "approach_axis_local", [0.0, 0.0, 1.0]
                )
            ),
            "minimum_downward_alignment": float(
                trajectory.get("minimum_downward_alignment", 0.50)
            ),
            "acquisition_minimum_downward_alignment": float(
                trajectory.get(
                    "acquisition_minimum_downward_alignment", 0.52
                )
            ),
            "acquisition_entry_alignment_hysteresis": float(
                trajectory.get(
                    "acquisition_entry_alignment_hysteresis", 0.003
                )
            ),
            "acquisition_entry_stable_steps": int(
                trajectory.get("acquisition_entry_stable_steps", 2)
            ),
            "minimum_joint_limit_margin": float(
                trajectory.get("minimum_joint_limit_margin", 0.025)
            ),
            "adaptive_grasp_yaw_fallback_step": float(
                trajectory.get("adaptive_grasp_yaw_fallback_step", 0.09)
            ),
            "maximum_grasp_axis_misalignment": float(
                trajectory.get("maximum_grasp_axis_misalignment", 0.18)
            ),
            "minimum_grasp_orientation_confidence": float(
                trajectory.get(
                    "minimum_grasp_orientation_confidence", 0.12
                )
            ),
            "pregrasp_tracking_joint_margin_reserve": float(
                trajectory.get(
                    "pregrasp_tracking_joint_margin_reserve", 0.010
                )
            ),
            "planning_retry_attempts": int(
                trajectory.get("planning_retry_attempts", 3)
            ),
            "return_lookahead_planner_max_iterations": int(
                trajectory.get(
                    "return_lookahead_planner_max_iterations", 100
                )
            ),
            "return_lookahead_retry_attempts": int(
                trajectory.get("return_lookahead_retry_attempts", 8)
            ),
            "return_lookahead_planning_budget": float(
                trajectory.get("return_lookahead_planning_budget", 0.40)
            ),
            "singularity_slowdown_condition": float(
                trajectory.get("singularity_slowdown_condition", 25.0)
            ),
            "singularity_hard_stop_condition": float(
                trajectory.get("singularity_hard_stop_condition", 45.0)
            ),
            "enable_global_transfer_planner": bool(
                trajectory.get("global_transfer_planner", True)
            ),
            "planner_max_iterations": int(
                trajectory.get("planner_max_iterations", 2400)
            ),
            "planner_step_size": float(
                trajectory.get("planner_step_size", 0.10)
            ),
            "planner_edge_resolution": float(
                trajectory.get("planner_edge_resolution", 0.018)
            ),
            "planner_execution_joint_step": float(
                trajectory.get("planner_execution_joint_step", 0.16)
            ),
            "planner_waypoint_tolerance": float(
                trajectory.get("planner_waypoint_tolerance", 0.075)
            ),
            "planner_waypoint_velocity_tolerance": float(
                trajectory.get("planner_waypoint_velocity_tolerance", 0.12)
            ),
            "planner_corner_settle_tolerance": float(
                trajectory.get("planner_corner_settle_tolerance", 0.018)
            ),
            "global_path_tracking_error_limit": float(
                trajectory.get("global_path_tracking_error_limit", 0.20)
            ),
            "acquisition_path_tracking_error_limit": float(
                trajectory.get("acquisition_path_tracking_error_limit", 0.12)
            ),
            "carry_path_tracking_error_limit": float(
                trajectory.get("carry_path_tracking_error_limit", 0.10)
            ),
            "carry_path_cross_track_enter_limit": float(
                trajectory.get("carry_path_cross_track_enter_limit", 0.015)
            ),
            "carry_path_cross_track_exit_limit": float(
                trajectory.get("carry_path_cross_track_exit_limit", 0.010)
            ),
            "carry_path_velocity_guard_horizon": float(
                trajectory.get("carry_path_velocity_guard_horizon", 0.04)
            ),
            "carry_path_stalled_replan_steps": int(
                trajectory.get("carry_path_stalled_replan_steps", 5)
            ),
            "planner_shortcut_attempts": int(
                trajectory.get("planner_shortcut_attempts", 120)
            ),
            "planner_minimum_downward_alignment": float(
                trajectory.get(
                    "planner_minimum_downward_alignment", 0.45
                )
            ),
            "intercept_deadline_margin": float(
                trajectory.get("intercept_deadline_margin", 0.25)
            ),
            "prefetch_intercept_clock_guard": float(
                trajectory.get("prefetch_intercept_clock_guard", 0.08)
            ),
            "prefetch_planning_execution_reserve": float(
                trajectory.get("prefetch_planning_execution_reserve", 3.0)
            ),
            "prefetch_planning_wall_realtime_ratio": float(
                trajectory.get("prefetch_planning_wall_realtime_ratio", 1.05)
            ),
            "max_pregrasp_timing_refinements": int(
                trajectory.get("max_pregrasp_timing_refinements", 1)
            ),
            "attempt_timeout": 28.0,
            "pregrasp_timeout": 6.0,
            "descent_timeout": 8.0,
            "lift_timeout": 5.0,
            "transfer_timeout": 12.0,
            "return_timeout": 8.0,
            "recovery_timeout": 6.0,
        },
        env=env,
    )


def run_episode(
    env: ConveyorSortingEnv,
    policy: BaselineSortingPolicy,
    *,
    seed: int,
    recorder: Any | None = None,
    realtime: bool = False,
    max_steps: int | None = None,
    pace_control_loop: bool = True,
    observation_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    observation = env.reset(seed=seed)
    policy.reset(seed=seed)
    policy.set_motion_compensation(env.enable_motion_compensation)
    if observation_callback is not None:
        observation_callback(observation)
    if recorder is not None:
        recorder.start_episode(
            {
                "seed": seed,
                "motion_compensation": env.enable_motion_compensation,
                "grasp_assist": env.grasp_assist,
                "action_type": env.action_type,
            }
        )
    step_index = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if max_steps is not None and step_index >= max_steps:
            truncated = True
            break
        process_viewer_commands = getattr(
            env, "process_pending_viewer_commands", None
        )
        if callable(process_viewer_commands):
            command_result = process_viewer_commands()
            if bool(command_result.get("reset", False)):
                reset_observation = command_result.get("observation")
                observation = (
                    reset_observation
                    if isinstance(reset_observation, dict)
                    else env.get_observation()
                )
                policy.reset()
                policy.set_motion_compensation(
                    env.enable_motion_compensation
                )
        if env.paused and env.render_mode == "human":
            if observation_callback is not None:
                observation_callback(observation)
            env.render()
            time.sleep(0.02)
            continue
        wall_start = time.monotonic()
        previous_timestamp = observation.get("timestamp")
        action = policy.act(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        if observation_callback is not None:
            observation_callback(observation)
        try:
            reset_during_step = bool(
                float(observation.get("timestamp")) + 1e-12
                < float(previous_timestamp)
            )
        except (TypeError, ValueError):
            reset_during_step = False
        if reset_during_step:
            # The human-viewer R key resets MuJoCo from inside env.render(),
            # after this tick's policy command has already been produced.
            # Synchronize the state machine before its next command so an
            # environment-cancelled request cannot remain pending forever.
            policy.reset()
            policy.set_motion_compensation(env.enable_motion_compensation)
        if recorder is not None:
            # Image collection is an explicit display-stream consumer.  RGB
            # and wrist renders remain outside policy.act and keep their own
            # timestamps instead of extending the top-depth control capture.
            if bool(getattr(recorder, "record_images", False)):
                observation = env.capture_display_observation(
                    include_wrist=True,
                    force=False,
                )
            recorder.record_step(
                observation, action, reward, terminated, truncated, info
            )
        step_index += 1
        # Background planning runs concurrently with physics.  Advancing the
        # simulator many times faster than its configured 50 Hz clock makes a
        # request consume a machine-dependent amount of *simulation* time and
        # breaks deadline determinism.  Pace every real async-policy episode;
        # ``realtime`` additionally controls whether a viewer is present.
        should_pace = bool(
            pace_control_loop
            and getattr(policy, "_planning_worker", None) is not None
        ) or bool(realtime and env.render_mode == "human")
        if should_pace:
            remaining = env.control_dt - (time.monotonic() - wall_start)
            if remaining > 0:
                time.sleep(remaining)
    metrics = env.get_metrics()
    metrics["steps"] = step_index
    metrics["terminated"] = terminated
    metrics["truncated"] = truncated
    if recorder is not None:
        recorder.end_episode(metrics)
    return metrics


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    totals = {
        key: sum(int(item.get(key, 0)) for item in results)
        for key in (
            "total_objects",
            "valid_sorting_objects",
            "grasp_attempts",
            "successful_grasps",
            "correctly_sorted",
            "empty_grasps",
            "dropped",
            "missed",
            "skipped",
            "wrong_bin",
            "collisions",
            "failed",
        )
    }
    valid = totals["valid_sorting_objects"]
    attempts = totals["grasp_attempts"]
    errors = [
        float(value)
        for result in results
        for value in result.get("prediction_errors", [])
    ]
    totals.update(
        {
            "episodes": len(results),
            "grasp_success_rate": (
                totals["successful_grasps"] / attempts if attempts else 0.0
            ),
            "attempt_grasp_success_rate": (
                totals["successful_grasps"] / attempts if attempts else 0.0
            ),
            "sorting_success_rate": (
                totals["correctly_sorted"] / valid if valid else 0.0
            ),
            "end_to_end_sorting_success_rate": (
                totals["correctly_sorted"] / valid if valid else 0.0
            ),
            "average_prediction_error": (
                float(np.mean(errors)) if errors else 0.0
            ),
            "max_prediction_error": float(np.max(errors)) if errors else 0.0,
        }
    )
    return totals


def print_metrics(metrics: dict[str, Any]) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def output_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

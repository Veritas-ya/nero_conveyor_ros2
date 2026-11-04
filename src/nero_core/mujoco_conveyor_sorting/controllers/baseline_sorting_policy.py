"""Deterministic multi-stage policy for continuous conveyor sorting.

The policy is deliberately an action generator, not a physics shortcut.  It
selects and tracks a target, produces arm/gripper commands, and reports stage
events.  It never moves a workpiece or creates a weld.  Optional environment
hooks are discovered by duck typing so this module does not import the
environment, perception, task, or recording packages.
"""

from __future__ import annotations

import inspect
import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._math import (
    as_vector,
    compose_world_delta,
    config_from_mapping,
    normalized_quat,
    quat_distance,
    quat_error_vector,
    quat_multiply,
)
from .joint_trajectory import (
    QuinticJointTrajectory,
    sampled_tcp_angular_minimum_duration,
)
from .planning_worker import PlanningRequest, PlanningResult
from .continuous_joint_trajectory import build_certified_edge_trajectory
from .robot_controller import discrete_braking_displacement
from .safe_path_planner import ConstrainedRRTConnect, SafePathConfig
from .timed_descent import TimedDescentMixin
from .full_pick_execution import FullPickExecutionMixin, validate_full_pick_settings
from .continuous_intercept import ContinuousInterceptMixin


class PolicyStage(str, Enum):
    WAITING = "waiting"
    PREGRASP = "pregrasp"
    TRACKING_DESCENT = "tracking_descent"
    CLOSING = "closing"
    LIFTING = "lifting"
    TRANSFERRING = "transferring"
    RELEASING = "releasing"
    RETURNING = "returning"
    RECOVERING = "recovering"

    # Readable aliases used by some downstream state visualizers.
    IDLE = "waiting"
    TRACKING = "tracking_descent"
    CLOSE = "closing"
    LIFT = "lifting"
    PLACE = "transferring"
    RELEASE = "releasing"
    RETURN = "returning"
    RECOVERY = "recovering"


TERMINAL_OBJECT_STATES = {
    "SORTED",
    "SKIPPED",
    "MISSED",
    "FAILED",
}
SELECTABLE_OBJECT_STATES = {
    "",
    "NEW",
    "TRACKING",
    "READY_TO_PLAN",
}


@dataclass(frozen=True)
class BaselinePolicyConfig:
    planning_mode: str = 'legacy'
    selection_policy: str = 'default'
    replan_policy: str = 'legacy'
    full_plan_close_window_s: float = .6
    full_plan_replanning_enabled: bool = True
    full_plan_contact_handoff_enabled: bool = False
    full_plan_observation_max_age_s: float = 2.0
    full_plan_compute_budget_s: float = 2.0
    """Motion waypoints, limits, timing, and deterministic selection settings."""

    action_mode: str = "ee_delta_pose"
    enable_motion_compensation: bool = True
    control_timestep: float = 0.02
    grasp_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    place_quaternion: tuple[float, float, float, float] = ()
    wait_pose: tuple[float, ...] = ()
    approach_height: float = 0.10
    # Optional commissioning task: execute the original certified approach,
    # then hand actual settled high feedback to the runner's position hold.
    prepare_high_only: bool = False
    # Explicit experimental alternative to the certified high/stop/descent route.
    # Shared and historical callers retain the original high approach.
    moving_approach_mode: str = "high"
    grasp_clearance: float = 0.012
    lift_height: float = 0.14
    # Keep the loaded bin approach just above the shallow joint-space dip of
    # the calibrated corridor route.  The extra 15 mm above the legacy drop
    # height is loaded feedback/braking headroom;
    # the worker and live executor still enforce the unchanged 0.408 m hard
    # TCP floor, collision, cone, singularity and command-envelope checks.
    place_clearance: float = 0.095
    max_translation_step: float = 0.022
    max_rotation_step: float = 0.10
    tracking_target_max_step: float = 0.012
    # During wrist/arm occlusion the RGB-D tracker may keep propagating a
    # plausible-looking ghost.  Corrections in the final acquisition stages
    # are therefore bounded around the already admitted intercept.  This does
    # not freeze ordinary coasting; it only caps accumulated XY drift.
    coasted_tracking_xy_limit: float = 0.035
    position_tolerance: float = 0.012
    orientation_tolerance: float = 0.15
    # A single 3-D radius allowed the jaw to start closing while it was still
    # too high above thin workpieces.  Keep the legacy field for callers that
    # explicitly configure it, but use separate planar/vertical capture
    # tolerances for the default NERO policy.
    closure_position_tolerance: float = 0.035
    closure_horizontal_tolerance: float = 0.022
    closure_vertical_tolerance: float = 0.012
    closure_orientation_tolerance: float = 0.25
    target_lost_timeout: float = 0.25
    close_settle_time: float = 0.25
    confirmed_grasp_settle_time: float = 0.02
    max_close_time: float = 0.75
    close_lead_time: float = 0.18
    # Zero retains clock-based acquisition for adapters without a current
    # visual position/velocity. Positive values predict physical jaw contact
    # against measured TCP instead of a stale, long-horizon intercept clock.
    closure_contact_lead_time: float = 0.0
    descent_lead_time: float = 0.80
    # Blend from the high approach corridor into the intercept instead of
    # braking to zero at an artificial pregrasp point.  The horizontal and
    # orientation gates keep the open fingers inside the belt access gap.
    pregrasp_blend_horizontal_tolerance: float = 0.035
    pregrasp_blend_vertical_clearance: float = 0.060
    pregrasp_blend_orientation_tolerance: float = 0.20
    # Short fixed lead compensates command/plant lag while tracking the belt.
    # It must not grow with elapsed close time because the perceived target
    # pose is already refreshed every control tick.
    tracking_velocity_feedforward_time: float = 0.16
    tracking_synchronization_horizon: float = 0.35
    closing_tracking_lead_time: float = 0.04
    track_belt_during_close: bool = False
    dynamic_intercept_tracking: bool = False
    pre_release_settle_time: float = 0.20
    release_settle_time: float = 0.25
    attempt_timeout: float = 8.0
    pregrasp_timeout: float = 2.5
    descent_timeout: float = 1.8
    lift_timeout: float = 1.8
    transfer_timeout: float = 3.0
    return_timeout: float = 3.0
    recovery_timeout: float = 3.0
    prediction_lead_time: float = 0.45
    minimum_prediction_confidence: float = 0.0
    require_grasp_confirmation: bool = False
    # A real motor-force/width proxy is distinct from two physical pad sensors.
    grasp_confirmation_source: str = "bilateral_contact"
    gripper_open_action: float = 1.0
    gripper_close_action: float = -1.0
    adaptive_grip_width: bool = True
    contact_hold_squeeze_fraction: float = 0.08
    contact_hold_max_relaxation_fraction: float = 1.0
    # Optional pad-force preload before the first lift command. Zero keeps
    # the historical contact-only adapter contract; no force sensor required.
    minimum_grasp_normal_force: float = 0.0
    grasp_force_preload_rate: float = 1.0
    grasp_force_settle_time: float = 0.06
    # The mesh-pad distance contains roughly 40 mm of mounting offset relative
    # to the ObjectSpec's outer part width.
    grip_width_margin: float = 0.040
    category_bin_size: int = 5
    bin_poses: Mapping[int | str, Sequence[float]] = field(default_factory=dict)
    enable_time_parameterization: bool = True
    joint_trajectory_backend: str = 'quintic'
    trajectory_jerk_limits: tuple[float, ...] | None = None
    trajectory_minimum_duration: float = 0.18
    trajectory_velocity_limits: tuple[float, ...] = ()
    trajectory_acceleration_limits: tuple[float, ...] = ()
    carrying_velocity_scale: float = 0.65
    carrying_acceleration_scale: float = 0.50
    gentle_carry_categories: tuple[int, ...] = (3,)
    trajectory_replan_position_tolerance: float = 0.004
    trajectory_replan_orientation_tolerance: float = 0.025
    safe_transfer_height: float = 0.43
    # Zero disables the separate vertical clearance leg for legacy adapters.
    # A payload must leave the conveyor rails before the wrist changes yaw.
    lift_clearance_height: float = 0.0
    # A locally blocked clearance is replanned from the measured grasp branch.
    # This changes the route, never the original height/limits/timeout.
    lift_clearance_global_replan_enabled: bool = True
    # If the upstream descent-start high pose has no IK, the same captured
    # grasp may be approached from directly above its reserved capture point.
    pregrasp_capture_high_fallback_enabled: bool = True
    pregrasp_capture_high_vertical_handoff_enabled: bool = True
    # Keep the already configured measured-load preload active as load shifts
    # from conveyor support into the jaws.  No new force target is introduced.
    maintain_grasp_preload_during_lift: bool = False
    # Optional experiment: avoid retracting to the old capture XY while the
    # accepted joint command is still braking out of conveyor following.
    lift_stopping_reference_enabled: bool = False
    # Correct loaded position-servo static error only after the ordinary
    # clearance request has reached its nominal height. This changes the
    # command, never the nominal clearance pose or feedback acceptance test.
    lift_feedback_correction_enabled: bool = True
    lift_feedback_integral_gain_s_inv: float = 1.0
    lift_feedback_max_correction_m: float = 0.010
    release_retreat_distance: float = 0.055
    # The ordinary return still withdraws the full distance above.  The
    # isolated A→B handoff may use this shorter, worker-certified endpoint
    # when the full retreat would put a redundant joint inside the actuator
    # command envelope.  It is used only after the gripper is open and both
    # fingers have been verified physically clear of the released object.
    prefetch_handoff_retreat_distance: float = 0.020
    # Cartesian servo limits.  Static point-to-point moves and moving-target
    # tracking both advance through small task-space increments; this keeps the
    # IK seed on one branch and prevents a joint-space chord from flipping the
    # tool between otherwise valid endpoint poses.
    cartesian_linear_speed: float = 0.28
    # Disable only the local linear-speed * dt clipping. Keep the original
    # Cartesian step, angular profile, joint dynamics and geometry checks.
    # cartesian_linear_speed may still supply a nominal motion-time estimate.
    local_tcp_linear_speed_limit_enabled: bool = True
    cartesian_angular_speed: float = 1.20
    maximum_tcp_angular_speed: float = 1.85
    # Simulation experiments can rely on joint velocity/acceleration limits
    # without an additional Cartesian speed cap. Defaults remain unchanged.
    tcp_speed_limits_enabled: bool = True
    # Separate the post-generation TCP rejection of timed descent curves
    # from local Cartesian tracking and loaded motion. Joint dynamics and
    # full-curve geometry validation still apply when this is disabled.
    timed_descent_tcp_speed_limits_enabled: bool = True
    # PREGRASP is the unloaded high approach, before any descent/closure.
    # Keep shared defaults unchanged; a measured-scene adapter may explicitly
    # use only the original joint limits for this global route's timing.
    dynamic_pregrasp_global_tcp_speed_limits_enabled: bool = True
    max_ik_joint_step: float = 0.25
    # A calibration-seeded redundancy change may be slightly larger than an
    # ordinary servo update, but is accepted only after its complete joint
    # chord passes dense collision/orientation/limit validation.  Keeping this
    # separate preserves the strict branch-jump guard for normal online IK.
    max_verified_ik_transition: float = 0.35
    # The gripper's local approach axis must stay in the world-downward cone.
    approach_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    minimum_downward_alignment: float = 0.50
    # Acquisition is deliberately tighter than bin transfer.  Some valid bin
    # paths need the wider transit cone, while an open gripper above the narrow
    # conveyor must retain dynamic margin from a horizontal or upward tool.
    acquisition_minimum_downward_alignment: float = 0.52
    acquisition_entry_alignment_hysteresis: float = 0.003
    acquisition_entry_stable_steps: int = 2
    minimum_joint_limit_margin: float = 0.025
    adaptive_grasp_yaw_fallback_step: float = 0.09
    # Real workpieces do not arrive at one calibrated world yaw.  For
    # object_short_axis grasps, follow the visual object's short axis exactly
    # first and permit only this much jaw-axis error while looking for a safer
    # IK branch.  Larger errors are not a valid substitute for a grasp.
    maximum_grasp_axis_misalignment: float = 0.18
    minimum_grasp_orientation_confidence: float = 0.12
    pregrasp_tracking_joint_margin_reserve: float = 0.010
    # A single numerical IK/RRT miss must not discard a moving workpiece.
    # Retry with new deterministic samples while holding the last safe command;
    # the bound still turns a genuinely unreachable goal into recovery.
    planning_retry_attempts: int = 3
    # Speculative PREGRASP uses short deterministic RRT slices while the arm
    # continues its return route.  A hard search is cancelled/restarted from
    # the newer measured q instead of consuming the entire remaining belt
    # window from an obsolete return state.
    return_lookahead_planner_max_iterations: int = 100
    return_lookahead_retry_attempts: int = 8
    return_lookahead_planning_budget: float = 0.40
    singularity_slowdown_condition: float = 25.0
    singularity_hard_stop_condition: float = 45.0
    enable_global_transfer_planner: bool = True
    planner_max_iterations: int = 2400
    planner_step_size: float = 0.10
    planner_edge_resolution: float = 0.018
    planner_execution_joint_step: float = 0.16
    planner_waypoint_tolerance: float = 0.075
    planner_waypoint_velocity_tolerance: float = 0.12
    planner_corner_settle_tolerance: float = 0.018
    # Hardware can retain an equilibrium error at both transit corners and
    # the high endpoint. Optional PREGRASP overrides keep static/loaded and
    # historical simulation settings independent. Every measured-to-command
    # connector still needs the original geometry validation.
    pregrasp_endpoint_settle_tolerance: float | None = None
    pregrasp_corner_settle_tolerance: float | None = None
    # Keep the requested set-point close enough to measured feedback that
    # unequal joint-servo lag cannot pull the physical arm outside a densely
    # certified RRT edge.  Transit can retain a wider torque-producing lead;
    # acquisition and the first loaded carry leg use tighter followers near
    # their orientation/height envelopes.  Every resulting measured-to-
    # command chord is still checked live with the unchanged full constraints.
    global_path_tracking_error_limit: float = 0.20
    acquisition_path_tracking_error_limit: float = 0.12
    carry_path_tracking_error_limit: float = 0.10
    # Loaded C execution additionally watches transverse deviation and a
    # conservative reaction-plus-stopping projection of measured joint
    # velocity.  The hysteresis starts edge-following recovery before inertia
    # can cross the immutable route's height/orientation envelope without
    # slowing ordinary transit.
    # Cross-track is an early diagnostic, not a physical safety relaxation:
    # every measured state, predicted velocity sweep and command chord still
    # receives the full MuJoCo height/cone/collision/margin validation.  Match
    # the entry threshold to the certified-corner settle tolerance: accepting
    # an 18 mrad connector and then immediately calling 15 mrad cross-track a
    # topology failure is an inconsistent contract.  Four mrad of hysteresis
    # remains above the measured loaded equilibrium.
    carry_path_cross_track_enter_limit: float = 0.018
    carry_path_cross_track_exit_limit: float = 0.014
    carry_path_velocity_guard_horizon: float = 0.04
    carry_path_tcp_height_guard_reserve: float = 0.004
    # A loaded arm can settle a few milliradians off a certified corner under
    # gravity even though every live constraint remains safe.  Do not pulse
    # forward through that mismatch: after this many settled, future-safe
    # recovery ticks, replan asynchronously from latest measured q under the
    # exact same carry envelope.
    carry_path_stalled_replan_steps: int = 5
    planner_shortcut_attempts: int = 120
    planner_minimum_downward_alignment: float = 0.45
    # A planned intercept must leave enough belt travel after the nominal
    # grasp timestamp for pose settling and the first physical jaw contact.
    # This is a deadline reserve, not permission to cross the x=0.28 m
    # planning boundary.
    intercept_deadline_margin: float = 0.25
    # B is built from a delayed 15 Hz RGB-D window estimate.  Aim one bounded
    # frame upstream of the hard deadline reserve so a normal refreshed track
    # cannot invalidate an otherwise identical immutable request by a few
    # tens of milliseconds.  This is extra scheduling reserve, never a
    # relaxation of ``intercept_deadline_margin``.
    prefetch_intercept_clock_guard: float = 0.08
    # B may improve a fully certified direct route in the background, but its
    # watchdog must reserve enough immutable clock for executing that route.
    # The wall/simulation ratio is the strict harness upper pacing envelope;
    # A and C are not clock-bound and therefore retain no wall budget.
    prefetch_planning_execution_reserve: float = 3.0
    # A refreshed B still receives a wholly new moving-pose/continuation/path
    # certificate.  A matching completed certificate may only tighten the
    # worker watchdog's execution reserve to its measured ETA plus this
    # margin; initial and identity-mismatched requests retain the fixed value.
    prefetch_refresh_execution_margin: float = 0.15
    prefetch_planning_wall_realtime_ratio: float = 1.05
    # One fixed-point update is enough for the local constant-velocity belt
    # model.  The resulting path is validated once more without moving the
    # target, so timing remains fail-closed without spending another control
    # interval on a redundant replan.
    max_pregrasp_timing_refinements: int = 1
    pregrasp_fixed_high_enabled: bool = False
    timed_descent_enabled: bool = False

    def __post_init__(self) -> None:
        validate_full_pick_settings(self)
        if self.moving_approach_mode not in {"high", "continuous"}:
            raise ValueError("moving_approach_mode must be high or continuous")
        if self.moving_approach_mode == "continuous" and self.prepare_high_only:
            raise ValueError("continuous interception cannot prepare a fixed high pose")
        if self.moving_approach_mode == "continuous" and (
                not self.dynamic_intercept_tracking or self.action_mode != "joint_position"
                or not self.enable_time_parameterization):
            raise ValueError("continuous interception needs dynamic joint-position trajectories")
        if self.joint_trajectory_backend not in ('quintic', 'ruckig'):
            raise ValueError('unknown joint trajectory backend')
        if self.trajectory_jerk_limits is not None:
            values = tuple(float(value) for value in self.trajectory_jerk_limits)
            if len(values) != 7 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError('jerk limits must contain seven positive finite values')
            object.__setattr__(self, 'trajectory_jerk_limits', values)
        if self.grasp_confirmation_source not in {"bilateral_contact", "width_and_force"}:
            raise ValueError("unknown grasp_confirmation_source")
        if not np.isfinite(self.safe_transfer_height):
            raise ValueError("safe_transfer_height must be finite in the selected frame")
        if not np.isfinite(self.lift_clearance_height) or self.lift_clearance_height < 0.0:
            raise ValueError("lift_clearance_height must be finite and non-negative")
        for name in ("lift_feedback_integral_gain_s_inv", "lift_feedback_max_correction_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("minimum_grasp_normal_force", "grasp_force_settle_time"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.grasp_force_preload_rate) or self.grasp_force_preload_rate <= 0.0:
            raise ValueError("grasp_force_preload_rate must be finite and positive")
        if (
            not np.isfinite(self.closure_contact_lead_time)
            or self.closure_contact_lead_time < 0.0
        ):
            raise ValueError(
                "closure_contact_lead_time must be finite and non-negative"
            )
        if self.action_mode not in {"joint_position", "ee_delta_pose"}:
            raise ValueError(
                "action_mode must be 'joint_position' or 'ee_delta_pose'"
            )
        positive = {
            "control_timestep": self.control_timestep,
            "approach_height": self.approach_height,
            "lift_height": self.lift_height,
            "max_translation_step": self.max_translation_step,
            "max_rotation_step": self.max_rotation_step,
            "tracking_target_max_step": self.tracking_target_max_step,
            "coasted_tracking_xy_limit": self.coasted_tracking_xy_limit,
            "position_tolerance": self.position_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "closure_position_tolerance": self.closure_position_tolerance,
            "closure_horizontal_tolerance": (
                self.closure_horizontal_tolerance
            ),
            "closure_vertical_tolerance": self.closure_vertical_tolerance,
            "closure_orientation_tolerance": self.closure_orientation_tolerance,
            "target_lost_timeout": self.target_lost_timeout,
            "close_settle_time": self.close_settle_time,
            "confirmed_grasp_settle_time": self.confirmed_grasp_settle_time,
            "max_close_time": self.max_close_time,
            "descent_lead_time": self.descent_lead_time,
            "pregrasp_blend_horizontal_tolerance": (
                self.pregrasp_blend_horizontal_tolerance
            ),
            "pregrasp_blend_vertical_clearance": (
                self.pregrasp_blend_vertical_clearance
            ),
            "pregrasp_blend_orientation_tolerance": (
                self.pregrasp_blend_orientation_tolerance
            ),
            "tracking_synchronization_horizon": (
                self.tracking_synchronization_horizon
            ),
            "release_settle_time": self.release_settle_time,
            "attempt_timeout": self.attempt_timeout,
            "pregrasp_timeout": self.pregrasp_timeout,
            "descent_timeout": self.descent_timeout,
            "lift_timeout": self.lift_timeout,
            "transfer_timeout": self.transfer_timeout,
            "return_timeout": self.return_timeout,
            "recovery_timeout": self.recovery_timeout,
            "trajectory_replan_position_tolerance": (
                self.trajectory_replan_position_tolerance
            ),
            "trajectory_replan_orientation_tolerance": (
                self.trajectory_replan_orientation_tolerance
            ),
            "cartesian_linear_speed": self.cartesian_linear_speed,
            "cartesian_angular_speed": self.cartesian_angular_speed,
            "maximum_tcp_angular_speed": self.maximum_tcp_angular_speed,
            "max_ik_joint_step": self.max_ik_joint_step,
            "max_verified_ik_transition": self.max_verified_ik_transition,
            "adaptive_grasp_yaw_fallback_step": (
                self.adaptive_grasp_yaw_fallback_step
            ),
            "maximum_grasp_axis_misalignment": (
                self.maximum_grasp_axis_misalignment
            ),
            "singularity_slowdown_condition": (
                self.singularity_slowdown_condition
            ),
            "singularity_hard_stop_condition": (
                self.singularity_hard_stop_condition
            ),
            "planner_step_size": self.planner_step_size,
            "planner_edge_resolution": self.planner_edge_resolution,
            "planner_execution_joint_step": (
                self.planner_execution_joint_step
            ),
            "planner_waypoint_tolerance": self.planner_waypoint_tolerance,
            "planner_waypoint_velocity_tolerance": (
                self.planner_waypoint_velocity_tolerance
            ),
            "planner_corner_settle_tolerance": (
                self.planner_corner_settle_tolerance
            ),
            "intercept_deadline_margin": self.intercept_deadline_margin,
            "prefetch_intercept_clock_guard": (
                self.prefetch_intercept_clock_guard
            ),
            "prefetch_planning_execution_reserve": (
                self.prefetch_planning_execution_reserve
            ),
            "prefetch_refresh_execution_margin": (
                self.prefetch_refresh_execution_margin
            ),
            "prefetch_planning_wall_realtime_ratio": (
                self.prefetch_planning_wall_realtime_ratio
            ),
            "return_lookahead_planning_budget": (
                self.return_lookahead_planning_budget
            ),
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(
                "baseline timing/motion values must be positive: "
                + ", ".join(invalid)
            )
        if self.max_pregrasp_timing_refinements < 1:
            raise ValueError(
                "max_pregrasp_timing_refinements must be at least one"
            )
        if (
            self.grasp_clearance < 0.0
            or self.place_clearance < 0.0
            or self.pre_release_settle_time < 0.0
            or self.grip_width_margin < 0.0
            or self.contact_hold_squeeze_fraction < 0.0
            or self.contact_hold_max_relaxation_fraction < 0.0
            or self.release_retreat_distance < 0.0
            or self.prefetch_handoff_retreat_distance < 0.0
            or self.minimum_joint_limit_margin < 0.0
            or self.pregrasp_tracking_joint_margin_reserve < 0.0
            or self.acquisition_entry_alignment_hysteresis < 0.0
        ):
            raise ValueError(
                "grasp/place/grip clearances and release retreat "
                "cannot be negative"
            )
        if (
            self.prefetch_handoff_retreat_distance
            > self.release_retreat_distance + 1e-12
        ):
            raise ValueError(
                "prefetch_handoff_retreat_distance cannot exceed "
                "release_retreat_distance"
            )
        if self.contact_hold_squeeze_fraction > 1.0:
            raise ValueError("contact_hold_squeeze_fraction cannot exceed one")
        if self.contact_hold_max_relaxation_fraction > 1.0:
            raise ValueError(
                "contact_hold_max_relaxation_fraction cannot exceed one"
            )
        if self.acquisition_entry_stable_steps < 1:
            raise ValueError("acquisition_entry_stable_steps must be positive")
        if self.planning_retry_attempts < 1:
            raise ValueError("planning_retry_attempts must be positive")
        if self.carry_path_stalled_replan_steps < 1:
            raise ValueError(
                "carry_path_stalled_replan_steps must be positive"
            )
        if self.return_lookahead_planner_max_iterations < 1:
            raise ValueError(
                "return_lookahead_planner_max_iterations must be positive"
            )
        if self.return_lookahead_retry_attempts < 1:
            raise ValueError(
                "return_lookahead_retry_attempts must be positive"
            )
        approach_axis = np.asarray(self.approach_axis_local, dtype=float)
        if (
            approach_axis.shape != (3,)
            or not np.all(np.isfinite(approach_axis))
            or np.linalg.norm(approach_axis) < 1e-9
        ):
            raise ValueError("approach_axis_local must be a non-zero 3-vector")
        if not -1.0 <= self.minimum_downward_alignment <= 1.0:
            raise ValueError("minimum_downward_alignment must be in [-1, 1]")
        if not -1.0 <= self.acquisition_minimum_downward_alignment <= 1.0:
            raise ValueError(
                "acquisition_minimum_downward_alignment must be in [-1, 1]"
            )
        if not -1.0 <= self.planner_minimum_downward_alignment <= 1.0:
            raise ValueError(
                "planner_minimum_downward_alignment must be in [-1, 1]"
            )
        if (
            self.singularity_hard_stop_condition
            <= self.singularity_slowdown_condition
        ):
            raise ValueError(
                "singularity_hard_stop_condition must exceed the slowdown value"
            )
        if self.planner_max_iterations < 1 or self.planner_shortcut_attempts < 0:
            raise ValueError("global planner iteration counts are invalid")
        tracking_limits = {
            "global_path_tracking_error_limit": (
                self.global_path_tracking_error_limit
            ),
            "acquisition_path_tracking_error_limit": (
                self.acquisition_path_tracking_error_limit
            ),
            "carry_path_tracking_error_limit": (
                self.carry_path_tracking_error_limit
            ),
            "carry_path_cross_track_enter_limit": (
                self.carry_path_cross_track_enter_limit
            ),
            "carry_path_cross_track_exit_limit": (
                self.carry_path_cross_track_exit_limit
            ),
            "carry_path_velocity_guard_horizon": (
                self.carry_path_velocity_guard_horizon
            ),
        }
        if self.pregrasp_endpoint_settle_tolerance is not None:
            tracking_limits["pregrasp_endpoint_settle_tolerance"] = (
                self.pregrasp_endpoint_settle_tolerance)
        if self.pregrasp_corner_settle_tolerance is not None:
            tracking_limits["pregrasp_corner_settle_tolerance"] = (
                self.pregrasp_corner_settle_tolerance)
        invalid_tracking_limits = [
            name
            for name, value in tracking_limits.items()
            if not np.isfinite(value) or value <= 0.0
        ]
        if invalid_tracking_limits:
            raise ValueError(
                "global path tracking limits must be finite and positive: "
                + ", ".join(invalid_tracking_limits)
            )
        if (
            not np.isfinite(self.carry_path_tcp_height_guard_reserve)
            or self.carry_path_tcp_height_guard_reserve < 0.0
        ):
            raise ValueError(
                "carry_path_tcp_height_guard_reserve must be finite and "
                "non-negative"
            )
        if (
            self.carry_path_cross_track_exit_limit
            >= self.carry_path_cross_track_enter_limit
        ):
            raise ValueError(
                "carry_path_cross_track_exit_limit must be below the enter "
                "limit"
            )
        if (
            self.carry_path_cross_track_enter_limit
            > self.planner_corner_settle_tolerance
        ):
            raise ValueError(
                "carry_path_cross_track_enter_limit cannot exceed the "
                "certified corner settle tolerance"
            )
        if self.max_close_time < self.close_settle_time:
            raise ValueError("max_close_time cannot be below close_settle_time")
        if not 0.0 <= self.confirmed_grasp_settle_time <= self.max_close_time:
            raise ValueError(
                "confirmed_grasp_settle_time must be within the close horizon"
            )
        if (
            self.close_lead_time < 0.0
            or self.tracking_velocity_feedforward_time < 0.0
            or self.closing_tracking_lead_time < 0.0
        ):
            raise ValueError("tracking lead times cannot be negative")
        if not 0.0 <= self.minimum_prediction_confidence <= 1.0:
            raise ValueError("minimum_prediction_confidence must be in [0, 1]")
        if not 0.0 <= self.minimum_grasp_orientation_confidence <= 1.0:
            raise ValueError(
                "minimum_grasp_orientation_confidence must be in [0, 1]"
            )
        if self.maximum_grasp_axis_misalignment > 0.5 * math.pi:
            raise ValueError(
                "maximum_grasp_axis_misalignment cannot exceed pi/2"
            )
        if self.category_bin_size < 1:
            raise ValueError("category_bin_size must be at least one")
        if self.trajectory_minimum_duration < 0.0:
            raise ValueError("trajectory_minimum_duration cannot be negative")
        if not 0.0 < self.carrying_velocity_scale <= 1.0:
            raise ValueError("carrying_velocity_scale must be in (0, 1]")
        if not 0.0 < self.carrying_acceleration_scale <= 1.0:
            raise ValueError("carrying_acceleration_scale must be in (0, 1]")
        if any(int(category) < 0 for category in self.gentle_carry_categories):
            raise ValueError("gentle_carry_categories cannot contain negatives")
        for name, limits in (
            ("trajectory_velocity_limits", self.trajectory_velocity_limits),
            (
                "trajectory_acceleration_limits",
                self.trajectory_acceleration_limits,
            ),
        ):
            if limits:
                values = np.asarray(limits, dtype=float).reshape(-1)
                if values.size not in {1, 7}:
                    raise ValueError(f"{name} must contain one or seven values")
                if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                    raise ValueError(f"{name} must contain positive finite values")


@dataclass(frozen=True)
class PolicyEvent:
    """A timestamped transition/event for task metrics and debug overlays."""

    name: str
    timestamp: float
    stage: str
    target_id: str | int | None
    reason: str = ""
    target_identity: Any = None


def _normalize_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source: Mapping[str, Any] = config
    for section_name in ("baseline_policy", "policy", "baseline"):
        if isinstance(source.get(section_name), Mapping):
            source = source[section_name]  # type: ignore[assignment]
            break
    result = dict(source)
    aliases = {
        "motion_compensation": "enable_motion_compensation",
        "grasp_orientation": "grasp_quaternion",
        "home_pose": "wait_pose",
        "pregrasp_height": "approach_height",
        "close_duration": "close_settle_time",
        "release_duration": "release_settle_time",
        "max_step": "max_translation_step",
        "dt": "control_timestep",
    }
    for old, new in aliases.items():
        if old in result and new not in result:
            result[new] = result[old]
    # Preserve the old tuning knob for external callers while allowing the
    # default policy to use the tighter anisotropic capture envelope.
    if "closure_position_tolerance" in source:
        result.setdefault(
            "closure_horizontal_tolerance",
            source["closure_position_tolerance"],
        )
        result.setdefault(
            "closure_vertical_tolerance",
            source["closure_position_tolerance"],
        )
    return result


def _value(item: Any, names: str | Iterable[str], default: Any = None) -> Any:
    """Read the first matching mapping key or attribute."""

    if isinstance(names, str):
        names = (names,)
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _object_id(item: Any) -> Any:
    return _value(item, ("object_id", "id", "target_id", "name"))


_MISSING = object()


def _prediction_validity(prediction: Any) -> bool | None:
    """Read an explicit predictor validity flag without inferring from confidence."""

    if prediction is None:
        return None
    validity = _value(prediction, ("valid", "is_valid"), _MISSING)
    if validity is _MISSING:
        return None
    if callable(validity):
        try:
            validity = validity()
        except TypeError:
            return None
    if isinstance(validity, (bool, np.bool_)):
        return bool(validity)
    return None


def _status_name(item: Any) -> str:
    raw = _value(item, ("state", "status"), "")
    raw = getattr(raw, "value", raw)
    return str(raw).upper()


def _as_pose(value: Any, default_quaternion: np.ndarray) -> np.ndarray | None:
    """Parse common pose mappings and 3/7-vectors into a pose vector."""

    if value is None:
        return None
    if isinstance(value, Mapping) or any(
        hasattr(value, name)
        for name in (
            "position",
            "pose",
            "predicted_position",
            "predicted_grasp_pose",
        )
    ):
        position = _value(
            value,
            (
                "position",
                "center",
                "pos",
                "xyz",
                "translation",
                "predicted_grasp_pose",
                "predicted_position",
            ),
        )
        quaternion = _value(
            value, ("quaternion", "quat", "orientation", "rotation")
        )
        if position is None:
            nested = _value(value, ("pose", "ee_pose"))
            if nested is not None and nested is not value:
                return _as_pose(nested, default_quaternion)
            return None
        position = as_vector(position, 3, name="pose position")
        quat = (
            default_quaternion
            if quaternion is None
            else normalized_quat(quaternion)
        )
        return np.r_[position, quat]
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(array)):
        return None
    if array.size == 3:
        return np.r_[array, default_quaternion]
    if array.size == 7:
        return np.r_[array[:3], normalized_quat(array[3:])]
    return None


def _yaw_from_quaternion(quaternion: Any) -> float:
    """Return world-Z yaw from a MuJoCo-order ``[w, x, y, z]`` quaternion."""

    w, x, y, z = normalized_quat(quaternion)
    return float(
        math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _jaw_axis_yaw(quaternion: Any) -> float:
    """Return the planar heading of the gripper's local +Y jaw axis."""

    w, x, y, z = normalized_quat(quaternion)
    axis_x = 2.0 * (x * y - w * z)
    axis_y = 1.0 - 2.0 * (x * x + z * z)
    return float(math.atan2(axis_y, axis_x))


def _wrap_undirected_axis_angle(angle: float) -> float:
    """Wrap an axis alignment error to ``[-pi/2, pi/2)``.

    A parallel-jaw closing axis is undirected: rotating it by pi exchanges the
    two fingers without changing the grasp geometry.
    """

    return float((angle + 0.5 * math.pi) % math.pi - 0.5 * math.pi)


def _hook(
    owner: Any,
    names: Iterable[str],
    argument_variants: Sequence[tuple[Any, ...]],
) -> Any:
    """Call the first compatible optional hook without hard dependencies."""

    if owner is None:
        return None
    for name in names:
        callback = getattr(owner, name, None)
        if not callable(callback):
            continue
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            signature = None
        for arguments in argument_variants:
            if signature is not None:
                try:
                    signature.bind(*arguments)
                except TypeError:
                    continue
            return callback(*arguments)
    return None


class BaselineSortingPolicy(FullPickExecutionMixin, ContinuousInterceptMixin, TimedDescentMixin):
    """Deterministic tracking/grasp/lift/place state machine.

    Primary API:

    ``reset()`` clears state, and ``act(observation, env=None)`` returns exactly
    ``{"arm": np.ndarray, "gripper": float}``.  Diagnostics are available via
    :meth:`get_info` and :meth:`drain_events`.
    """

    def __init__(
        self,
        config: BaselinePolicyConfig | Mapping[str, Any] | None = None,
        *,
        env: Any | None = None,
        predictor: Any | None = None,
        grasp_window: Any | None = None,
        planning_worker: Any | None = None,
        prefetch_worker: Any | None = None,
    ) -> None:
        if isinstance(config, Mapping):
            config = _normalize_config(config)
        self.config = config_from_mapping(BaselinePolicyConfig, config)
        self._lift_feedback_correction_enabled = bool(self.config.lift_feedback_correction_enabled)
        self.env = env
        self.predictor = predictor
        self.grasp_window = grasp_window
        self._planning_worker = planning_worker
        self._planning_worker_injected = planning_worker is not None
        self._planning_worker_factory_attempted = planning_worker is not None
        self._planning_request: PlanningRequest | None = None
        self._planning_generation = 0
        self._planning_next_request_id = 1
        self._planning_stale_results = 0
        self._planning_pending_goal_update: dict[str, Any] = {}
        # Speculative post-release planning has its own queue, result channel
        # and generation.  Ordinary stage transitions may cancel the primary
        # route, but can never cancel work whose only purpose is a future
        # retreat/PREGRASP hand-off.
        self._prefetch_worker = prefetch_worker
        self._prefetch_worker_injected = prefetch_worker is not None
        self._prefetch_worker_factory_attempted = prefetch_worker is not None
        self._prefetch_generation = 0
        self._prefetch_next_request_id = 1_000_000
        self._prefetch_stale_results = 0
        self._observed_scene_epoch: int | None = None
        self._grasp_quaternion = normalized_quat(
            self.config.grasp_quaternion, name="grasp_quaternion"
        )
        self._force_place_quaternion = bool(self.config.place_quaternion)
        self._place_quaternion = (
            self._grasp_quaternion.copy()
            if not self._force_place_quaternion
            else normalized_quat(
                self.config.place_quaternion, name="place_quaternion"
            )
        )
        self._approach_axis_local = np.asarray(
            self.config.approach_axis_local, dtype=float
        )
        self._approach_axis_local /= np.linalg.norm(self._approach_axis_local)
        self.reset()
        # Model cloning is a one-time initialization cost.  Do it while the
        # policy is constructed, never on the first 50 Hz control tick.
        self._ensure_planning_worker()
        self._ensure_prefetch_worker()

    @property
    def state(self) -> PolicyStage:
        return self._stage

    @property
    def stage(self) -> PolicyStage:
        return self._stage

    @property
    def target_id(self) -> Any:
        return self._target_id

    @property
    def motion_compensation_enabled(self) -> bool:
        return self.config.enable_motion_compensation

    def reset(self, seed: int | None = None) -> None:
        """Reset deterministic state; ``seed`` is accepted for gym symmetry."""

        del seed
        self._cancel_async_plan()
        self._cancel_prefetch_plan()
        self._observed_scene_epoch = None
        self._stage = PolicyStage.WAITING
        self._target_id: Any = None
        self._target: Any = None
        self._stage_started_at: float | None = None
        self._attempt_started_at: float | None = None
        self._last_seen_target_at: float | None = None
        self._internal_time = 0.0
        self._last_observation_time: float | None = None
        self._wait_pose: np.ndarray | None = (
            None
            if not self.config.wait_pose
            else _as_pose(self.config.wait_pose, self._grasp_quaternion)
        )
        if self.config.wait_pose and self._wait_pose is None:
            raise ValueError("wait_pose must have 3 or 7 values")
        self._grasp_pose: np.ndarray | None = None
        self._contact_tracking_xy: np.ndarray | None = None
        self._contact_tracking_reference_xy: np.ndarray | None = None
        self._selected_grasp_quaternion: np.ndarray | None = None
        self._pregrasp_ready = False
        self.prepared_high_state = None
        self._pregrasp_capture_high_active = False
        self._pregrasp_capture_high_diagnostics: dict[str, Any] = {}
        self._lift_pose: np.ndarray | None = None
        self._lift_clearance_pose: np.ndarray | None = None
        self._lift_clearance_complete = False
        self._lift_clearance_global_replan_active = False
        self._lift_clearance_global_replan_attempted = False
        self._lift_clearance_local_rejection: str | None = None
        self._lift_clearance_global_replan_diagnostics: dict[str, Any] = {}
        self._lift_stopping_reference_diagnostics: dict[str, Any] = {}
        self._lift_feedback_correction_diagnostics: dict[str, Any] = {}
        self._reset_lift_feedback_correction("policy_reset")
        self._dynamic_lift_repositioned = False
        self._place_pose: np.ndarray | None = None
        self._joint_trajectory: QuinticJointTrajectory | None = None
        self._trajectory_started_at: float | None = None
        self._trajectory_target_pose: np.ndarray | None = None
        self._trajectory_stage: PolicyStage | None = None
        self._cartesian_command_pose: np.ndarray | None = None
        self._arm_command_is_time_parameterized = False
        self._pending_arm_motion: dict[str, Any] | None = None
        self._deferred_closure_context: dict[str, Any] | None = None
        self._safe_joint_path: list[np.ndarray] = []
        self._safe_joint_path_index = 0
        self._safe_path_target_pose: np.ndarray | None = None
        self._safe_path_stage: PolicyStage | None = None
        self._safe_path_planner: ConstrainedRRTConnect | None = None
        self._safe_path_eta_s: float | None = None
        self._safe_path_minimum_tcp_height: float | None = None
        self._safe_path_minimum_joint_margin: float | None = None
        self._safe_path_minimum_downward_alignment: float | None = None
        self._safe_path_generation: int | None = None
        self._safe_path_target_id: Any = None
        self._safe_path_scene_epoch: int | None = None
        self._safe_path_edge_durations: list[float] = []
        self._safe_path_worker_validated = False
        self._safe_path_continuation_validated = False
        # A carry-prefetch route is the immutable parent of the speculative
        # place -> retreat -> next-pregrasp legs.  If live execution rejects
        # that parent route, all of those descendants must be discarded too;
        # otherwise TRANSFERRING can later compare a primary fallback branch
        # against the stale C endpoint forever.
        self._safe_path_from_carry_prefetch = False
        self._carry_path_recovery_active = False
        self._carry_path_recovery_predictive = False
        self._carry_path_stalled_steps = 0
        self._carry_path_predictive_warning_edge_index: int | None = None
        self._carry_path_replan_constraints: (
            tuple[float, float, float] | None
        ) = None
        self._carry_path_replan_brake_q: np.ndarray | None = None
        # A continuation-certified PREGRASP endpoint is also a useful
        # redundancy-branch token when the same moving target forces a fresh
        # endpoint plan.  It is only an IK seed: every replacement endpoint,
        # continuation probe and global path is certified from scratch.
        self._pregrasp_branch_q: np.ndarray | None = None
        self._pregrasp_branch_target_id: Any = None
        self._pregrasp_branch_scene_epoch: int | None = None
        # Search hints survive a same-target endpoint revision, but never act
        # as collision certificates. Only unfinished interior corners are
        # suggested; the obsolete grasp endpoint is an IK seed above, not a
        # compulsory detour on the replacement route.
        self._pregrasp_previous_route: dict[str, Any] | None = None
        self._trajectory_reuse_diagnostics: dict[str, Any] = {}
        self._safe_edge_trajectory: QuinticJointTrajectory | None = None
        self._safe_edge_started_at: float | None = None
        self._safe_edge_goal_index: int | None = None
        # A speculative result may arrive while feedback is a few milliradians
        # away from the immutable worker start even though the actuator is
        # already holding that exact start target.  Such a short live-checked
        # connector is a feedback-settling edge, not a new time law.
        self._safe_path_feedback_connector_goal_index: int | None = None
        self._safe_edge_passthrough_eligible = False
        self._safe_edge_dynamic_retime_active = False
        self._safe_edge_handoff_pending = False
        self._safe_edge_retime_diagnostics: dict[str, Any] = {}
        self._pregrasp_timing_refinements = 0
        # A moving interception can change after the first executable ETA is
        # known. Keep that certified high approach as useful motion while the
        # same worker plans the next leg from its immutable joint endpoint.
        self._pregrasp_approach_pose: np.ndarray | None = None
        self._pregrasp_approach_request: PlanningRequest | None = None
        self._pregrasp_approach_result: PlanningResult | None = None
        self._pregrasp_last_planning_time_s = 0.0
        self._pregrasp_progress_sample = None
        self._pregrasp_motion_time_scale = 1.0
        self._pregrasp_capture_time_s: float | None = None
        self._pregrasp_execution_timing: dict[str, Any] = {}
        self._fixed_high_request: PlanningRequest | None = None
        self._fixed_high_validated_low: np.ndarray | None = None
        self._fixed_high_validated_capture_time: float | None = None
        self._fixed_high_rejected = False
        self._fixed_high_ready = False
        self._reset_full_pick()
        self._reset_timed_descent()
        self._reset_continuous_intercept()
        self._motion_plan_failed = False
        self._motion_plan_failure_count = 0
        self._terminal_motion_plan_reason: str | None = None
        self._last_ik_solution: np.ndarray | None = None
        self._dynamic_vertical_retry = False
        self._transfer_waypoints: list[np.ndarray] = []
        self._transfer_waypoint_index = 0
        self._return_waypoints: list[np.ndarray] = []
        self._return_waypoint_index = 0
        self._events: list[PolicyEvent] = []
        self._last_reason = "reset"
        self._last_action: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._last_jacobian_condition: float | None = None
        self._minimum_observed_joint_margin: float | None = None
        self._minimum_observed_downward_alignment: float | None = None
        self._grasp_gripper_action = self.config.gripper_close_action
        self._secured_hold_pose: np.ndarray | None = None
        self._grasp_confirmed_at: float | None = None
        self._grasp_force_stable_since: float | None = None
        self._grasp_force_last_update_at: float | None = None
        self._grasp_force_status = "disabled"
        self._closure_intercept_at: float | None = None
        self._bilateral_contact_stable_since: float | None = None
        self._previous_return_alignment: float | None = None
        self._return_alignment_stable_steps = 0
        self._return_acquisition_latched = False
        self._return_candidate_id: Any = None
        self._return_candidate_stable_steps = 0
        # RETURNING look-ahead is deliberately separate from the installed
        # return route.  The worker may search for the next visual target while
        # the already validated return path keeps executing; no result is
        # allowed to reserve a target or replace commands until the live
        # acquisition gate and first edge have both been revalidated.
        self._return_lookahead_request: PlanningRequest | None = None
        self._return_lookahead_result: PlanningResult | None = None
        self._return_lookahead_category: Any = None
        self._return_lookahead_pose: np.ndarray | None = None
        self._return_lookahead_scene_epoch: int | None = None
        self._return_lookahead_observed_at: float | None = None
        self._return_lookahead_failures = 0
        self._prefetch_phase: str | None = None
        # The same isolated worker is a serial three-leg pipeline: C plans the
        # deterministic high-corridor-to-bin carry while the current target is
        # still approaching; A plans the post-release retreat; B plans the
        # next visual interception.  Keeping C on this channel avoids primary
        # stage-transition cancellation without adding a third native worker.
        self._carry_prefetch_request: PlanningRequest | None = None
        self._carry_prefetch_completed_request: PlanningRequest | None = None
        self._carry_prefetch_result: PlanningResult | None = None
        self._carry_prefetch_target_id: Any = None
        self._carry_prefetch_bin_index: int | None = None
        self._carry_prefetch_place_pose: np.ndarray | None = None
        self._carry_prefetch_anchor_q: np.ndarray | None = None
        self._carry_prefetch_scene_epoch: int | None = None
        self._prefetch_retreat_request: PlanningRequest | None = None
        self._prefetch_completed_retreat_request: PlanningRequest | None = None
        self._prefetch_retreat_result: PlanningResult | None = None
        self._prefetch_retreat_failed = False
        self._prefetch_retreat_not_before: float | None = None
        self._prefetch_release_terminal_confirmed = False
        self._prefetch_request: PlanningRequest | None = None
        self._prefetch_completed_request: PlanningRequest | None = None
        self._prefetch_result: PlanningResult | None = None
        self._prefetch_anchor_q: np.ndarray | None = None
        self._prefetch_place_q: np.ndarray | None = None
        self._prefetch_retreat_pose: np.ndarray | None = None
        # Immutable provenance for A's certified source bin.  B topology hints
        # are keyed by this previous-bin branch and the next part category; the
        # current target cannot be used after C is consumed because it already
        # describes the next destination, not the retreat branch B starts on.
        self._prefetch_source_bin_index: int | None = None
        self._prefetch_category: Any = None
        self._prefetch_pose: np.ndarray | None = None
        self._prefetch_scene_epoch: int | None = None
        self._prefetch_observed_at: float | None = None
        self._prefetch_candidate_id: Any = None
        self._prefetch_candidate_stable_steps = 0
        # Once B has observed a concrete next target, a missing 15 Hz visual
        # sample is a dropout rather than an unborn fixed-cadence part.  Keep
        # that identity across a short grace interval so a single dropped
        # frame cannot cancel the isolated worker or strand its certified A
        # anchor.  The bound is cleared on any explicit version discard.
        self._prefetch_bound_candidate_id: Any = None
        self._prefetch_candidate_missing_since: float | None = None
        # A non-cancelled B failure is terminal for this previous-target/A
        # anchor version.  Holding that anchor after its bounded worker search
        # produced no certificate only suppresses the ordinary moving-return
        # lookahead until the part expires.  Keep the failed next-target id
        # outside `_discard_prefetch` long enough to release all A/B ownership;
        # `_clear_target` starts a fresh service version.
        self._prefetch_terminal_b_target_id: Any = None
        self._prefetch_failures = 0
        self._prefetch_release_complete = False
        self._prefetch_retreat_installed = False
        # A hard intercept rejection cannot improve while both the arm and
        # the physical deadline remain unchanged.  Remember that measured
        # state so the same EDF candidate cannot be released/reselected every
        # control tick.  It becomes eligible again after a meaningful arm
        # displacement; other visible candidates remain selectable meanwhile.
        self._intercept_rejected_target_id: Any = None
        self._intercept_rejection_q: np.ndarray | None = None

    def close(self) -> None:
        """Cooperatively stop the optional background planner."""

        worker = self._planning_worker
        prefetch_worker = self._prefetch_worker
        self._planning_request = None
        self._return_lookahead_request = None
        self._return_lookahead_result = None
        self._planning_generation += 1
        # Closing is also a terminal version boundary for every speculative
        # C -> A -> B request.  Stopping the worker alone would leave a
        # completed carry result (and its copied MuJoCo snapshot) reachable
        # from the policy after shutdown, which is both stale state and an
        # avoidable native-memory lifetime extension.
        self._cancel_prefetch_plan()
        for owned_worker in dict.fromkeys((worker, prefetch_worker)):
            if owned_worker is None:
                continue
            owned_worker.cancel_pending()
            closed = owned_worker.close()
            alive = getattr(owned_worker, "is_alive", False)
            if callable(alive):
                alive = alive()
            if closed is False or alive is True:
                raise RuntimeError(
                    "background planning worker did not stop within its close budget"
                )

    def _ensure_planning_worker(self) -> Any | None:
        """Lazily create the real isolated worker outside module import time."""

        if self._planning_worker is not None:
            return self._planning_worker
        if self._planning_worker_factory_attempted:
            # Worker construction clones native MuJoCo state.  A failed
            # factory is sticky for this policy instance; retrying it at 50 Hz
            # would move a potentially expensive clone/error loop straight
            # back onto the viewer thread.
            return None
        factory = getattr(self.env, "create_global_planning_worker", None)
        if not callable(factory):
            return None
        self._planning_worker_factory_attempted = True
        try:
            self._planning_worker = factory()
        except Exception as error:
            self._last_error = f"background planner unavailable: {error}"
            return None
        return self._planning_worker

    def _ensure_prefetch_worker(self) -> Any | None:
        """Create the independent speculative worker before control starts."""

        if self._prefetch_worker is not None:
            return self._prefetch_worker
        if self._prefetch_worker_factory_attempted:
            return None
        factory = getattr(self.env, "create_prefetch_planning_worker", None)
        if not callable(factory):
            return None
        self._prefetch_worker_factory_attempted = True
        try:
            self._prefetch_worker = factory()
        except Exception as error:
            self._last_error = f"prefetch planner unavailable: {error}"
            return None
        return self._prefetch_worker

    def _planning_certificate_contract_required(self) -> bool:
        """Whether the primary worker is on the production safety boundary."""

        return bool(
            not self._planning_worker_injected
            or callable(
                getattr(self.env, "create_global_planning_worker", None)
            )
        )

    def _prefetch_certificate_contract_required(self) -> bool:
        """Whether speculative results require production certificates."""

        return bool(
            not self._prefetch_worker_injected
            or callable(
                getattr(self.env, "create_prefetch_planning_worker", None)
            )
        )

    def _cancel_async_plan(self) -> None:
        """Invalidate both queued and running work by monotonically version."""

        self._planning_generation = getattr(self, "_planning_generation", 0) + 1
        self._planning_request = None
        worker = getattr(self, "_planning_worker", None)
        if worker is not None:
            worker.cancel_pending()

    def _cancel_prefetch_plan(self) -> None:
        """Invalidate only speculative work, never the primary route."""

        self._prefetch_generation = getattr(self, "_prefetch_generation", 0) + 1
        self._carry_prefetch_request = None
        self._carry_prefetch_completed_request = None
        self._carry_prefetch_result = None
        self._carry_prefetch_target_id = None
        self._carry_prefetch_bin_index = None
        self._carry_prefetch_place_pose = None
        self._carry_prefetch_anchor_q = None
        self._carry_prefetch_scene_epoch = None
        self._prefetch_retreat_request = None
        self._prefetch_completed_retreat_request = None
        self._prefetch_retreat_result = None
        self._prefetch_retreat_failed = False
        self._prefetch_retreat_not_before = None
        self._prefetch_release_terminal_confirmed = False
        self._prefetch_source_bin_index = None
        self._prefetch_request = None
        self._prefetch_completed_request = None
        self._prefetch_result = None
        worker = getattr(self, "_prefetch_worker", None)
        if worker is not None:
            worker.cancel_pending()

    def set_motion_compensation(self, enabled: bool) -> None:
        """Runtime switch used by compensation A/B evaluation scripts."""

        # The dataclass is frozen; keep the runtime override separate.
        self._motion_compensation_override = bool(enabled)

    def _motion_compensation(self) -> bool:
        return getattr(
            self,
            "_motion_compensation_override",
            self.config.enable_motion_compensation,
        )

    def _execution_dt(self, observation: Mapping[str, Any] | None = None) -> float:
        """Share the actual command interval with an external controller.

        Simulation and older adapters retain their configured fixed step.
        Hardware supplies the same elapsed interval to act() and apply_action;
        mixing nominal and measured periods breaks discrete acceleration bounds.
        """
        raw = self.config.control_timestep if observation is None else observation.get(
            "control_dt", self.config.control_timestep
        )
        dt = float(raw)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        return dt

    def _timestamp(self, observation: Mapping[str, Any]) -> float:
        raw = _value(observation, ("timestamp", "sim_time", "time"))
        if raw is None and self.env is not None:
            raw = _value(getattr(self.env, "data", None), "time")
        if raw is None:
            self._internal_time += self.config.control_timestep
            return self._internal_time
        now = float(raw)
        if not np.isfinite(now):
            self._internal_time += self.config.control_timestep
            return self._internal_time
        # A reset can legitimately move simulation time backwards.
        self._internal_time = now
        return now

    def _emit(self, name: str, now: float, reason: str = "") -> None:
        event = PolicyEvent(
            target_identity=getattr(self.env, 'target_identity', None),
            name=name,
            timestamp=now,
            stage=self._stage.value,
            target_id=self._target_id,
            reason=reason,
        )
        self._events.append(event)
        _hook(
            self.env,
            ("on_policy_event", "handle_policy_event"),
            ((event,), (name, self._target_id, reason), (name,)),
        )

    def _transition(
        self,
        stage: PolicyStage,
        now: float,
        *,
        reason: str,
        event: str | None = None,
    ) -> None:
        if stage != self._stage:
            preserve_descent = (self._stage == PolicyStage.PREGRASP
                and stage == PolicyStage.TRACKING_DESCENT and self._descent_plan is not None)
            self._invalidate_motion_plan(preserve_timed_descent=preserve_descent)
            self._pending_arm_motion = None
            self._deferred_closure_context = None
            self._pregrasp_capture_high_active = False
            self._pregrasp_capture_high_diagnostics["active"] = False
        self._stage = stage
        self._stage_started_at = now
        self._last_reason = reason
        if event is not None:
            self._emit(event, now, reason)

    def _invalidate_motion_plan(self, *, preserve_timed_descent=False) -> None:
        """Discard a cached IK endpoint/trajectory after a stage or goal change."""

        self._joint_trajectory = None
        self._trajectory_started_at = None
        self._trajectory_target_pose = None
        self._trajectory_stage = None
        self._cartesian_command_pose = None
        self._lift_clearance_local_rejection = None
        self._reset_lift_feedback_correction("motion_plan_invalidated")
        self._safe_joint_path = []
        self._safe_joint_path_index = 0
        self._safe_path_target_pose = None
        self._safe_path_stage = None
        self._safe_path_planner = None
        self._safe_path_eta_s = None
        self._pregrasp_execution_timing = {}
        self._fixed_high_request = None
        self._fixed_high_validated_low = None
        self._fixed_high_validated_capture_time = None
        self._fixed_high_rejected = False
        self._fixed_high_ready = False
        self._safe_path_minimum_tcp_height = None
        self._safe_path_minimum_joint_margin = None
        self._safe_path_minimum_downward_alignment = None
        self._safe_path_generation = None
        self._safe_path_target_id = None
        self._safe_path_scene_epoch = None
        self._safe_path_edge_durations = []
        self._safe_path_worker_validated = False
        self._safe_path_continuation_validated = False
        self._safe_path_from_carry_prefetch = False
        self._carry_path_recovery_active = False
        self._carry_path_recovery_predictive = False
        self._carry_path_stalled_steps = 0
        self._carry_path_predictive_warning_edge_index = None
        self._carry_path_replan_constraints = None
        self._carry_path_replan_brake_q = None
        self._safe_edge_trajectory = None
        self._safe_edge_started_at = None
        self._safe_edge_goal_index = None
        self._safe_path_feedback_connector_goal_index = None
        self._safe_edge_passthrough_eligible = False
        self._safe_edge_dynamic_retime_active = False
        self._safe_edge_handoff_pending = False
        self._safe_edge_retime_diagnostics = {}
        self._pregrasp_approach_pose = None
        self._pregrasp_approach_request = None
        self._pregrasp_approach_result = None
        self._pregrasp_last_planning_time_s = 0.0
        self._pregrasp_progress_sample = None
        self._pregrasp_motion_time_scale = 1.0
        self._motion_plan_failed = False
        self._motion_plan_failure_count = 0
        if not preserve_timed_descent:
            self._descent_plan = None
            self._descent_update_request = self._descent_replacement = None
            self._descent_rebook_pending = False
            self._descent_high_hold_q = None
            self._descent_feedback_request_id = None
            self._cancel_async_plan()
            self._discard_continuous_intercept()
        self._return_lookahead_request = None
        self._return_lookahead_result = None
        self._return_lookahead_pose = None
        self._return_lookahead_scene_epoch = None
        self._return_lookahead_observed_at = None

    def _invalidate_unsafe_safe_path(self) -> None:
        """Fail closed and invalidate speculative descendants of carry C."""

        carry_parent_failed = self._safe_path_from_carry_prefetch
        self._invalidate_motion_plan()
        if carry_parent_failed:
            # A and B were certified from C's final joint state.  Once live
            # feedback rejects C, neither their start state nor the cached
            # place endpoint describes the primary fallback route.
            self._discard_prefetch(cancel_worker=True)

    def _clear_terminal_motion_plan_failure(self) -> None:
        """Clear an admission failure only when starting a different target."""

        self._terminal_motion_plan_reason = None

    def _record_motion_plan_failure(self) -> None:
        """Remember one bounded retry without emitting an unsafe command."""

        self._motion_plan_failed = True
        self._motion_plan_failure_count += 1

    def _record_motion_plan_success(self) -> None:
        self._motion_plan_failed = False
        self._motion_plan_failure_count = 0

    def _prepare_motion_plan_retry(self, now: float) -> bool:
        """Clear a transient failure so the current stage replans this tick."""

        if not self._motion_plan_failed:
            return False
        if self._motion_plan_failure_count >= self.config.planning_retry_attempts:
            return False
        detail = self._last_error or "motion planner returned no path"
        self._motion_plan_failed = False
        self._emit(
            "motion_replan",
            now,
            f"attempt={self._motion_plan_failure_count + 1}: {detail}",
        )
        return True

    def _elapsed(self, now: float) -> float:
        if self._stage_started_at is None:
            return 0.0
        return max(0.0, now - self._stage_started_at)

    def _objects(self, observation: Mapping[str, Any]) -> list[Any]:
        objects = _value(observation, ("objects", "targets"), ())
        if isinstance(objects, Mapping):
            normalized: list[Any] = []
            for key, item in objects.items():
                if isinstance(item, Mapping) and _object_id(item) is None:
                    copied = dict(item)
                    copied["object_id"] = key
                    normalized.append(copied)
                else:
                    normalized.append(item)
            return normalized
        if objects is None:
            return []
        return list(objects)

    def _find_target(
        self, observation: Mapping[str, Any], target_id: Any
    ) -> Any | None:
        for item in self._objects(observation):
            if _object_id(item) == target_id or str(_object_id(item)) == str(target_id):
                return item
        found = _hook(
            self.env,
            ("get_object_state", "get_target", "get_object"),
            ((target_id,), (target_id, observation)),
        )
        return found

    def _window_valid(
        self, target: Any, observation: Mapping[str, Any]
    ) -> bool:
        """Whether an attempt is feasible, even if closure time is still ahead."""

        status = _status_name(target)
        if status in TERMINAL_OBJECT_STATES:
            return False
        target_prediction = _value(
            target, ("prediction", "motion_prediction", "prediction_result")
        )
        observation_prediction = self._prediction_from_mapping(
            observation, _object_id(target)
        )
        if (
            _prediction_validity(target) is False
            or _prediction_validity(target_prediction) is False
            or _prediction_validity(observation_prediction) is False
        ):
            return False
        confidence = _value(
            target, ("prediction_confidence", "confidence"), 1.0
        )
        if confidence is not None:
            try:
                if float(confidence) < self.config.minimum_prediction_confidence:
                    return False
            except (TypeError, ValueError):
                return False
        # An anisotropic real part needs a measured in-plane orientation to
        # define its short-axis grasp.  A fixed nominal wrist quaternion is
        # not a safe fallback for a low-confidence or malformed object yaw.
        # Exact end-effector poses supplied by a 6-D grasp backend remain
        # authoritative and bypass this 2.5-D PCA confidence gate.
        mode = str(_value(target, "grasp_yaw_mode", "fixed")).lower()
        exact_ee_pose = _value(
            target,
            (
                "predicted_ee_grasp_pose",
                "ee_grasp_pose",
                "gripper_grasp_pose",
            ),
            None,
        )
        if mode == "object_short_axis" and exact_ee_pose is None:
            orientation_confidence = _value(
                target, ("orientation_confidence",), None
            )
            if orientation_confidence is None:
                return False
            try:
                if (
                    not np.isfinite(float(orientation_confidence))
                    or float(orientation_confidence)
                    < self.config.minimum_grasp_orientation_confidence
                ):
                    return False
            except (TypeError, ValueError):
                return False
        explicit = _value(
            target,
            (
                "window_valid",
                "reachable",
                "feasible",
            ),
        )
        if explicit is False:
            return False
        nested_window = _value(target, ("grasp_window", "window"))
        if nested_window is not None:
            feasible = _value(nested_window, ("feasible", "is_graspable"))
            if callable(feasible):
                try:
                    feasible = feasible()
                except TypeError:
                    feasible = None
            if feasible is False:
                return False

        hook_result = _hook(
            self.env,
            (
                "is_target_feasible",
                "is_grasp_window_feasible",
                "target_window_feasible",
            ),
            (
                (_object_id(target), observation),
                (target, observation),
                (_object_id(target),),
                (target,),
            ),
        )
        if hook_result is False:
            return False
        evaluator_result = _hook(
            self.grasp_window,
            ("is_feasible", "evaluate"),
            ((target, observation), (target,)),
        )
        if isinstance(evaluator_result, Mapping):
            evaluator_result = _value(
                evaluator_result, ("is_graspable", "valid", "in_window")
            )
        return evaluator_result is not False

    def _grasp_abort_reason(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> str | None:
        """Return a hard-abort reason before the object has been secured."""

        status = _status_name(target)
        if status == "GRASPED":
            return None
        if status in TERMINAL_OBJECT_STATES:
            recorded = _value(
                target,
                ("failure_reason", "reason"),
                None,
            )
            return str(recorded or "target_terminal")
        if (
            self._stage
            in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
            and self._contact_tracking_xy is not None
        ):
            # A latched pad contact is stronger local evidence than a coasted
            # RGB-D estimate.  Keep the bounded vertical acquisition alive if
            # the ghost leaves its visual workspace; terminal lifecycle and
            # severe-collision checks above/at the caller still fail closed.
            return None
        # Once both the pose and timing gates have started physical closure,
        # give the fingers their configured confirmation horizon.  The hard
        # geometric belt boundary and terminal object states still abort.
        if self._stage == PolicyStage.CLOSING:
            return None
        if self._window_valid(target, observation):
            return None
        prediction = _value(
            target, ("prediction", "motion_prediction", "prediction_result")
        )
        observation_prediction = self._prediction_from_mapping(
            observation, _object_id(target)
        )
        # Match _window_valid's first rejected condition. A spatial window
        # may still be feasible when measured velocity has expired; its "ok"
        # must not replace the actual motion-prediction failure.
        for source in (target, prediction, observation_prediction):
            if _prediction_validity(source) is False:
                reason = _value(source, "reason", None)
                return str(reason if reason and reason != "ok"
                           else "motion_prediction_unavailable")
        reason = _value(prediction, "reason", None)
        nested_window = _value(target, ("grasp_window", "window"))
        window_reason = _value(nested_window, "reason", None)
        if window_reason and window_reason != "ok":
            return str(window_reason)
        return str(reason if reason and reason != "ok" else "grasp_window_closed")

    def _window_current(
        self, target: Any, observation: Mapping[str, Any]
    ) -> bool | None:
        """Return closure readiness, or None when no window signal is supplied."""

        current = _value(
            target,
            (
                "in_grasp_window",
                "within_grasp_window",
                "current_in_window",
                "is_graspable",
            ),
        )
        if callable(current):
            try:
                current = current()
            except TypeError:
                current = None
        nested = _value(target, ("grasp_window", "window"))
        if nested is not None:
            nested_current = _value(
                nested,
                ("current_in_window", "in_grasp_window", "is_graspable"),
            )
            if callable(nested_current):
                try:
                    nested_current = nested_current()
                except TypeError:
                    nested_current = None
            if nested_current is not None:
                current = nested_current
        hook_result = _hook(
            self.env,
            (
                "is_target_graspable",
                "is_in_grasp_window",
                "target_in_grasp_window",
            ),
            (
                (_object_id(target), observation),
                (target, observation),
                (_object_id(target),),
                (target,),
            ),
        )
        if hook_result is not None:
            current = hook_result
        return None if current is None else bool(current)

    def _closure_time_ready(
        self,
        target: Any,
        observation: Mapping[str, Any],
        now: float,
    ) -> bool:
        prediction = self._prediction_from_mapping(
            observation, _object_id(target)
        )
        planned_time = _value(
            prediction,
            ("estimated_grasp_time", "grasp_time"),
            _value(target, ("estimated_grasp_time", "grasp_time")),
        )
        if self._stage == PolicyStage.TRACKING_DESCENT and self._descent_plan is not None:
            # Use the capture clock of the curve actually being executed,
            # including accepted future-state splices/rebookings.
            planned_time = self._descent_plan.end
        if planned_time is None:
            return True
        lead_time = _value(
            target,
            ("close_lead_time", "gripper_close_lead_time"),
            self.config.close_lead_time,
        )
        try:
            lead = max(0.0, float(lead_time))
            return now >= float(planned_time) - lead
        except (TypeError, ValueError):
            return True

    def _intercept_time_reached(
        self,
        target: Any,
        observation: Mapping[str, Any],
        now: float,
    ) -> bool:
        """Whether the admitted workpiece centre has reached its intercept.

        Jaw motion deliberately starts ``close_lead_time`` early.  Contact on
        a long distal edge during that lead interval is not yet a centred,
        transportable grasp, so it must not freeze tracking or start lifting.
        """

        if self._closure_intercept_at is not None:
            return now >= self._closure_intercept_at - 1e-9
        prediction = self._prediction_from_mapping(
            observation, _object_id(target)
        )
        planned_time = _value(
            prediction,
            ("estimated_grasp_time", "grasp_time"),
            _value(target, ("estimated_grasp_time", "grasp_time")),
        )
        if planned_time is None:
            return True
        try:
            value = float(planned_time)
        except (TypeError, ValueError):
            return True
        return bool(np.isfinite(value) and now >= value - 1e-9)

    def _visual_closure_intercept(
        self, target: Any, observation: Mapping[str, Any], now: float
    ) -> tuple[bool, float] | None:
        """Predict jaw contact using only the current visual track and TCP.

        Route admission can precede contact by several seconds. Belt spin-up
        and slip then move the corrected intercept away from its original
        clock/pose, especially when final occlusion bounds the TCP correction.
        Closure must refer to the actual jaw location, not that obsolete clock.
        """
        lead = self.config.closure_contact_lead_time
        if lead <= 0.0:
            return None
        try:
            position = np.asarray(_value(target, "position"), dtype=float)
            velocity = np.asarray(
                _value(target, ("estimated_velocity", "linear_velocity")),
                dtype=float,
            )
            timestamp = float(_value(target, "timestamp"))
            height = float(_value(target, "grasp_height"))
            horizontal_tolerance = float(
                _value(target, "closure_horizontal_tolerance",
                       self.config.closure_horizontal_tolerance)
            )
        except (TypeError, ValueError):
            return None
        if position.shape != (3,) or velocity.shape != (3,):
            return None
        if (
            not np.all(np.isfinite(np.r_[
                position, velocity, timestamp, height, horizontal_tolerance
            ]))
            or abs(now - timestamp) > 2.0 * self.config.control_timestep
        ):
            return None
        speed_squared = float(velocity[:2] @ velocity[:2])
        if speed_squared < 1e-6 or self._grasp_pose is None:
            return None
        current = self._ee_pose(observation)
        contact_pose = self._grasp_pose.copy()
        contact_velocity = velocity.copy()
        dynamic_relative = False
        if self.config.dynamic_intercept_tracking:
            solver = _value(self.env, ("ik", "ik_solver"))
            jacobian = _hook(solver, ("compute_jacobian",), ((self._joint_positions(observation),),))
            joint_velocity = np.asarray(_value(observation, ("joint_vel", "joint_velocity", "joint_velocities"), []), dtype=float)
            if jacobian is not None and np.shape(jacobian) == (6, 7) and joint_velocity.shape == (7,) and np.all(np.isfinite(joint_velocity)):
                contact_velocity -= np.asarray(jacobian)[:3] @ joint_velocity
                dynamic_relative = True
        contact_pose[:3] = position + contact_velocity * lead
        contact_pose[2] = position[2] + height
        # A narrow longitudinal band prevents a long part's leading edge
        # being mistaken for a centred grasp; retain tighter category gates.
        tolerance = min(0.010, horizontal_tolerance)
        ready = bool(
            np.linalg.norm(contact_pose[:2] - current[:2]) <= tolerance
            and self._closure_pose_reached(contact_pose, observation, target)
            and self._window_valid(target, observation)
        )
        crossing = max(
            0.0,
            float((current[:2] - position[:2]) @ velocity[:2]) / speed_squared,
        )
        return ready, now + (lead if dynamic_relative else crossing)

    def _descent_time_ready(
        self,
        target: Any,
        observation: Mapping[str, Any],
        now: float,
    ) -> bool:
        """Start descent late enough that open fingers do not sweep the belt."""

        if self._descent_plan is not None and self._timed_descent_enabled(observation):
            return now + 1e-9 >= self._descent_plan.start
        prediction = self._prediction_from_mapping(
            observation, _object_id(target)
        )
        planned_time = self._pregrasp_capture_time_s
        if planned_time is None:
            planned_time = _value(
                prediction,
                ("planned_grasp_time", "estimated_grasp_time", "grasp_time"),
                _value(target, ("planned_grasp_time", "estimated_grasp_time", "grasp_time")),
            )
        if planned_time is None:
            return True
        try:
            return now + 1e-9 >= float(planned_time) - self.config.descent_lead_time
        except (TypeError, ValueError):
            return True

    def _belt_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the base-frame origin and positive travel direction.

        ``workspace.dynamic_intercept.belt_direction`` and ``belt_origin``
        define s = dot(position - origin, direction), in metres. Legacy
        configuration keeps the original world +X axis and zero origin.
        These coordinates never rotate robot poses or mechanical constraints.
        """

        dynamic = _value(self.env, "scene_config", {}).get("workspace", {}).get(
            "dynamic_intercept", {}
        )
        direction = as_vector(
            dynamic.get("belt_direction", (1.0, 0.0, 0.0)), 3,
            name="belt_direction",
        )
        origin = as_vector(
            dynamic.get("belt_origin", (0.0, 0.0, 0.0)), 3,
            name="belt_origin",
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("belt_direction must have nonzero length")
        return origin, direction / norm

    def _belt_coordinate(self, position: Sequence[float]) -> float:
        origin, direction = self._belt_frame()
        return float((as_vector(position, 3, name="belt position") - origin) @ direction)

    def _belt_bounds(self, *names: str, default: Sequence[float]) -> np.ndarray:
        """Read signed longitudinal bounds, retaining legacy x_range keys."""

        dynamic = _value(self.env, "scene_config", {}).get("workspace", {}).get(
            "dynamic_intercept", {}
        )
        raw = next((dynamic[name] for name in names if name in dynamic), default)
        bounds = as_vector(raw, 2, name="belt longitudinal range")
        if bounds[0] > bounds[1]:
            raise ValueError("belt longitudinal range must be ordered")
        return bounds

    def _retimed_grasp_position(
        self, previous: np.ndarray, body_position: np.ndarray,
    ) -> np.ndarray:
        """Refresh body XY while retaining tool height above an inclined belt.

        Retiming hooks supply body poses, so copying their absolute Z would
        discard the grasp-height offset. Infer belt travel from its horizontal
        displacement and add only that travel's vertical component. A level
        +X belt retains the historical unchanged grasp Z.
        """

        result = np.asarray(previous, dtype=float).copy()
        delta_xy = np.asarray(body_position, dtype=float)[:2] - result[:2]
        _, direction = self._belt_frame()
        horizontal_squared = float(direction[:2] @ direction[:2])
        if horizontal_squared > 1e-12:
            result[2] += float(delta_xy @ direction[:2]) / horizontal_squared * direction[2]
        result[:2] = np.asarray(body_position, dtype=float)[:2]
        return result

    def _belt_velocity(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a finite XYZ belt velocity for Cartesian feed-forward."""

        raw = _value(
            observation,
            ("belt_velocity", "conveyor_velocity", "belt_speed"),
            0.0,
        )
        try:
            velocity = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.zeros(3, dtype=float)
        if velocity.size == 1:
            velocity = self._belt_frame()[1] * velocity[0]
        if velocity.size < 3 or not np.all(np.isfinite(velocity[:3])):
            return np.zeros(3, dtype=float)
        return velocity[:3].copy()

    def _local_closure_reference(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Predict the visual centre at the locally admitted jaw crossing.

        Visual closure may supersede an old route's intercept clock.  Both
        the refreshed grasp pose and its tracking lead must then use this
        same local clock.  ``position`` is already the current visual
        estimate, so image latency must not be integrated a second time.
        """

        if (
            self._stage != PolicyStage.CLOSING
            or self._closure_intercept_at is None
        ):
            return None
        try:
            position = np.asarray(_value(target, "position"), dtype=float)
            velocity = np.asarray(
                _value(target, ("estimated_velocity", "linear_velocity")),
                dtype=float,
            )
            timestamp = float(_value(target, "timestamp"))
            now = self._timestamp_without_advancing(observation)
            crossing = float(self._closure_intercept_at)
        except (TypeError, ValueError):
            return None
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or not np.all(np.isfinite(np.r_[
                position, velocity, timestamp, now, crossing
            ]))
            or abs(now - timestamp) > 2.0 * self.config.control_timestep
        ):
            return None
        reference = position + velocity * max(0.0, crossing - now)
        return reference, velocity, crossing

    def _tracking_command_target(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        lead_time: float | None = None,
        target: Any | None = None,
        synchronize_to_intercept: bool = False,
    ) -> np.ndarray:
        """Generate a belt-synchronous Cartesian target.

        A predicted intercept is a pose at a future timestamp, not a static
        point where open fingers should wait in the workpiece's path.  During
        the final approach, move that pose backwards/forwards along the belt
        by ``now - planned_time`` so the tool first matches conveyor velocity
        and crosses the original intercept at the planned instant.
        """

        horizon = (
            self.config.tracking_velocity_feedforward_time
            if lead_time is None
            else max(0.0, float(lead_time))
        )
        if self.config.dynamic_intercept_tracking and (lead_time is None or lead_time > 0):
            dynamic_cfg = getattr(self.env, "scene_config", {}).get("workspace", {}).get("dynamic_intercept", {})
            horizon = float(dynamic_cfg.get("servo_lookahead_s", horizon))
        local_closure = (
            self._local_closure_reference(target, observation)
            if synchronize_to_intercept and target is not None
            else None
        )
        if synchronize_to_intercept and target is not None:
            prediction = self._prediction_from_mapping(
                observation, _object_id(target)
            )
            if self.config.dynamic_intercept_tracking and local_closure is None:
                current_pose = _as_pose(_value(prediction, "current_estimated_pose"), self._grasp_quaternion)
                predicted_pose = _as_pose(_value(prediction, "predicted_grasp_pose"), self._grasp_quaternion)
                if current_pose is not None and predicted_pose is not None:
                    command = target_pose.copy()
                    # Preserve the selected TCP height/orientation offset,
                    # but evaluate belt position at this command's time.
                    command[:3] += current_pose[:3] - predicted_pose[:3]
                    command[:3] += self._belt_velocity(observation) * horizon
                    return self._guard_final_tracking_xy(command, target, observation,
                        reference_offset_xy=self._belt_velocity(observation)[:2] * horizon)
            planned_time = _value(
                prediction,
                ("estimated_grasp_time", "grasp_time"),
                _value(target, ("estimated_grasp_time", "grasp_time")),
            )
            if local_closure is not None:
                planned_time = local_closure[2]
            try:
                time_error = self._timestamp(observation) - float(planned_time)
            except (TypeError, ValueError):
                time_error = 0.0
            if np.isfinite(time_error):
                # Legacy execution keeps its vertical intercept corridor.
                # Dynamic execution time-aligns the moving target instead;
                # low-margin branches below retain the certified corridor,
                # and every other small servo chord is still live-checked.
                horizon += float(
                    np.clip(
                        time_error,
                        -(
                            float(getattr(self.env, "scene_config", {}).get("workspace", {}).get("dynamic_intercept", {}).get("tracking_horizon_s", self.config.tracking_synchronization_horizon))
                            if self.config.dynamic_intercept_tracking
                            else self.config.tracking_synchronization_horizon
                        ),
                        0.0,
                    )
                )
                if not self.config.dynamic_intercept_tracking:
                    horizon = max(0.0, horizon)
        command = target_pose.copy()
        if (
            self.config.dynamic_intercept_tracking
            and self._pregrasp_branch_q is not None
            and self._stage == PolicyStage.TRACKING_DESCENT
            and self._joint_limit_margin(self._pregrasp_branch_q)
            < 2.0 * self._pregrasp_planning_joint_margin()
        ):
            # A low-margin high waypoint was certified with a vertical
            # continuation. Do not invalidate it by first steering upstream
            # into the inner reach boundary. Keep this dynamically selected
            # intercept until the moving target reaches it; positive belt
            # tracking remains available thereafter and every chord is checked.
            horizon = max(0.0, horizon)
        if local_closure is not None:
            # Match the visual velocity used to retime the grasp centre;
            # belt speed can differ during spin-up or object slip.  Preserve
            # the separately calibrated grasp height and orientation.
            command[:3] += local_closure[1] * horizon
        else:
            command[:3] += self._belt_velocity(observation) * horizon
        if self.config.dynamic_intercept_tracking:
            # Follow the current visual trajectory within the certified access
            # corridor. IK, collision, contact and joint-envelope guards remain.
            bounds = self._belt_bounds(
                "tracking_s_range", "tracking_x_range", "s_range", "x_range",
                default=(0.02, 0.18),
            )
            coordinate = self._belt_coordinate(command[:3])
            command[:3] += (
                float(np.clip(coordinate, *bounds)) - coordinate
            ) * self._belt_frame()[1]
            # A coasted visual residual is bounded around the prediction at
            # the SAME time as this servo command. Bounding a current target
            # around a future intercept silently turned dynamic following
            # back into waiting at that intercept (and blocked incoming parts
            # with a longitudinally oriented open finger).
            return self._guard_final_tracking_xy(
                command, target, observation,
                reference_offset_xy=self._belt_velocity(observation)[:2] * horizon,
            )
        return self._guard_final_tracking_xy(command, target, observation)

    def _planned_tracking_reference(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray | None, float | None, bool, float]:
        """Return policy-visible plan/track evidence for final acquisition."""

        target_id = _object_id(target)
        prediction = self._prediction_from_mapping(observation, target_id)
        planned = _value(
            prediction,
            ("planned_grasp_pose", "admitted_grasp_pose"),
            _value(target, ("planned_grasp_pose", "admitted_grasp_pose")),
        )
        planned_pose = _as_pose(planned, self._grasp_quaternion)
        planned_time_raw = _value(
            prediction,
            ("planned_grasp_time", "estimated_grasp_time", "grasp_time"),
            _value(
                target,
                ("planned_grasp_time", "estimated_grasp_time", "grasp_time"),
            ),
        )
        try:
            planned_time = float(planned_time_raw)
            if not np.isfinite(planned_time):
                planned_time = None
        except (TypeError, ValueError):
            planned_time = None
        visible = bool(
            _value(
                target,
                "visible",
                _value(prediction, "visible", True),
            )
        )
        age_raw = _value(
            target,
            "track_observation_age",
            _value(prediction, "track_observation_age", 0.0),
        )
        try:
            track_age = max(0.0, float(age_raw))
        except (TypeError, ValueError):
            track_age = 0.0
        return planned_pose, planned_time, visible, track_age

    def _external_finger_contact(
        self, observation: Mapping[str, Any]
    ) -> bool:
        """Read target-agnostic fingertip feedback (real pad-sensor analogue)."""

        gripper = _value(observation, "gripper")
        if gripper is None:
            return False
        explicit = _value(
            gripper,
            (
                "any_external_finger_contact",
                "any_finger_contact",
                "finger_contact",
            ),
            None,
        )
        if explicit is not None:
            return bool(explicit)
        left = _value(gripper, "left_finger_contact", False)
        right = _value(gripper, "right_finger_contact", False)
        return bool(left) or bool(right)

    @staticmethod
    def _bilateral_finger_contact(observation: Mapping[str, Any]) -> bool:
        """Whether both pads currently report contact with external geometry.

        A single leading pad can brush a moving workpiece several centimetres
        before the TCP reaches its admitted intercept.  Treating that brush as
        a completed capture freezes belt tracking and lets the part escape the
        other pad.  Bilateral contact is the first target-agnostic signal that
        the part is actually between the jaws; the environment's stronger
        eight-substep confirmation remains responsible for declaring GRASPED.
        """

        gripper = _value(observation, "gripper")
        if gripper is None:
            return False
        explicit = _value(
            gripper,
            ("both_fingers_contact", "bilateral_contact"),
            None,
        )
        if explicit is not None:
            return bool(explicit)
        left = _value(gripper, "left_finger_contact", False)
        right = _value(gripper, "right_finger_contact", False)
        return bool(left) and bool(right)

    @staticmethod
    def _bounded_planar_correction(
        candidate_xy: np.ndarray,
        reference_xy: np.ndarray,
        limit: float,
    ) -> np.ndarray:
        correction = np.asarray(candidate_xy, dtype=float) - np.asarray(
            reference_xy, dtype=float
        )
        distance = float(np.linalg.norm(correction))
        if distance <= limit or distance <= 1e-12:
            return np.asarray(candidate_xy, dtype=float).copy()
        return np.asarray(reference_xy, dtype=float) + correction * (
            float(limit) / distance
        )

    def _guard_final_tracking_xy(
        self,
        pose: np.ndarray,
        target: Any | None,
        observation: Mapping[str, Any],
        *,
        reference_offset_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Bound a coasted ghost and freeze tangential motion at first touch.

        The admitted intercept is the only positional reference.  A fresh
        visual mask remains unconstrained.  While a mask is occluded, limited
        coasting is retained inside a finite XY ball around that plan.  A
        target-agnostic finger contact then latches the *measured TCP* XY so
        the controller can descend/close without shearing sideways.  The latch
        survives transient contact loss throughout ``CLOSING``.
        """

        guarded = np.asarray(pose, dtype=float).copy()
        if (
            target is None
            or self._stage
            not in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
        ):
            return guarded

        planned_pose, planned_time, visible, track_age = (
            self._planned_tracking_reference(target, observation)
        )
        if planned_pose is None or planned_time is None:
            # Fail closed with respect to this feature: without the policy's
            # admitted plan, do not invent a positional anchor.
            return guarded

        reference_xy = planned_pose[:2].copy()
        if self.config.dynamic_intercept_tracking:
            # A future capture pose must be moved back to the current command
            # time during descent, including when visual measurements coast.
            reference_xy += self._belt_velocity(observation)[:2] * (
                self._timestamp_without_advancing(observation) - planned_time)
            if reference_offset_xy is not None:
                reference_xy += np.asarray(reference_offset_xy, dtype=float)
        # A leading-pad brush does not prove that the workpiece is inside the
        # capture envelope.  Freeze only after both fingers make contact; the
        # environment's stronger force/debounce gate remains responsible for
        # declaring the grasp confirmed.
        # A dynamic capture keeps following through the existing force and
        # debounce confirmation dwell. Binary pad contact can be only a brush;
        # latching XY here would stop the jaws while the belt is still moving.
        # The confirmed CLOSING -> LIFTING transition freezes the measured TCP.
        # Retain the original contact latch for the legacy fixed interception.
        contact = (
            not self.config.dynamic_intercept_tracking
            and self._bilateral_finger_contact(observation)
        )
        contact_time = (
            self._closure_intercept_at
            if self._closure_intercept_at is not None else planned_time
        )
        intercept_ready = bool(
            contact_time is None
            or self._timestamp_without_advancing(observation) >= contact_time - 1e-9
        )
        if self._contact_tracking_xy is None and contact and intercept_ready:
            measured = self._ee_pose(observation)
            # Ignore a high-altitude rail/bin brush.  Freezing is allowed only
            # once both pads are in the ordinary closure-height band; before
            # then the controller must keep matching belt motion.
            vertical_band = self.config.closure_vertical_tolerance
            if abs(float(measured[2] - guarded[2])) <= vertical_band:
                self._contact_tracking_xy = measured[:2].copy()
                self._contact_tracking_reference_xy = reference_xy.copy()

        if self._contact_tracking_xy is not None:
            guarded[:2] = self._contact_tracking_xy
            return guarded

        # ``visible=False`` is explicit tracker coasting.  Track age is read
        # as evidence (and rejects malformed negative ages), while plan time
        # scopes the reference to this admitted interception.  Do not freeze:
        # permit bounded belt/slip correction until physical contact arrives.
        if not visible and track_age > 0.0:
            correction_limit = self.config.coasted_tracking_xy_limit
            if self._stage == PolicyStage.CLOSING:
                elapsed = max(
                    0.0,
                    self._timestamp_without_advancing(observation)
                    - float(self._stage_started_at or 0.0),
                )
                belt_speed = float(
                    np.linalg.norm(self._belt_velocity(observation)[:2])
                )
                if np.isfinite(belt_speed):
                    correction_limit += belt_speed * min(
                        elapsed, self.config.max_close_time
                    )
            guarded[:2] = self._bounded_planar_correction(
                guarded[:2],
                reference_xy,
                correction_limit,
            )
        return guarded

    def _pregrasp_blend_ready(
        self,
        grasp_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        """Whether the TCP is safely inside the vertical intercept corridor."""

        current = self._ee_pose(observation)
        return bool(
            np.linalg.norm(current[:2] - grasp_pose[:2])
            <= self.config.pregrasp_blend_horizontal_tolerance
            and current[2] - grasp_pose[2]
            >= self.config.pregrasp_blend_vertical_clearance
            and quat_distance(current[3:], grasp_pose[3:])
            <= self.config.pregrasp_blend_orientation_tolerance
            and self._orientation_is_safe(current[3:])
        )

    def _candidate_key(self, target: Any) -> tuple[Any, ...]:
        """Stable EDF ordering: earliest window exit, then deterministic ID."""

        infinity = float("inf")
        deadline = _value(
            target,
            (
                "grasp_window_end",
                "window_end",
                "time_to_exit",
                "deadline",
            ),
            infinity,
        )
        try:
            deadline_value = float(deadline)
        except (TypeError, ValueError):
            deadline_value = infinity
        position = _value(
            target,
            (
                "predicted_position",
                "current_ground_truth_position",
                "position",
            ),
            (infinity, infinity, infinity),
        )
        try:
            conveyor_coordinate = self._belt_coordinate(position)
        except (TypeError, ValueError, IndexError):
            conveyor_coordinate = infinity
        return (
            deadline_value,
            conveyor_coordinate,
            str(_object_id(target)),
        )

    def _remember_intercept_rejection(
        self,
        target_id: Any,
        observation: Mapping[str, Any],
    ) -> None:
        """Suppress a deadline-infeasible target until the arm state changes.

        Belt deadlines are monotonic: immediately reserving the same part from
        the same measured joints cannot turn a rejected route into a feasible
        one.  The suppression is deliberately tied to arm displacement rather
        than wall-clock time, so recovery to a materially better staging state
        permits one new admission check without creating a 50 Hz retry storm.
        """

        self._intercept_rejected_target_id = target_id
        try:
            self._intercept_rejection_q = self._joint_positions(observation)
        except (KeyError, TypeError, ValueError):
            self._intercept_rejection_q = None

    def _intercept_retry_suppressed(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> bool:
        rejected_id = self._intercept_rejected_target_id
        if rejected_id is None or not self._same_target_id(
            _object_id(target), rejected_id
        ):
            return False
        reference = self._intercept_rejection_q
        if reference is None:
            return True
        try:
            current = self._joint_positions(observation)
        except (KeyError, TypeError, ValueError):
            return True
        if np.max(np.abs(current - reference)) <= (
            self.config.planner_corner_settle_tolerance
        ):
            return True
        self._intercept_rejected_target_id = None
        self._intercept_rejection_q = None
        return False

    def _selection_observation(
        self, observation: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Hide a same-state deadline rejection from atomic target hooks."""

        objects = self._objects(observation)
        filtered = [
            target
            for target in objects
            if not self._intercept_retry_suppressed(target, observation)
        ]
        if len(filtered) == len(objects):
            return observation
        selected_observation = dict(observation)
        selected_observation["objects"] = filtered
        return selected_observation

    def _select_target(self, observation: Mapping[str, Any]) -> Any | None:
        selection_observation = self._selection_observation(observation)
        # A zero-argument hook cannot observe the suppression set.  Skip that
        # legacy overload while a same-state rejection is active; local EDF
        # still selects every unsuppressed candidate from the observation.
        hook_variants = (
            ((selection_observation,),)
            if self._intercept_rejected_target_id is not None
            else ((selection_observation,), ())
        )
        selected = _hook(
            self.env,
            ("select_target", "select_sorting_target"),
            hook_variants,
        )
        if selected is not None:
            if not isinstance(selected, (Mapping,)) and not hasattr(
                selected, "__dict__"
            ):
                selected_object = self._find_target(observation, selected)
                if selected_object is not None:
                    selected = selected_object
                else:
                    selected = None
            if selected is not None:
                # Environment target hooks may retain the just-released object
                # until it physically settles into a bin.  Do not let that
                # GRASPED object re-enter the grasp pipeline during the
                # opportunistic RETURNING -> WAITING hand-off.  PLANNING is
                # allowed because the environment hook may atomically reserve
                # a fresh candidate before returning it.
                status = _status_name(selected)
                hook_selectable = (
                    status in SELECTABLE_OBJECT_STATES
                    or status == "PLANNING"
                )
                if (
                    hook_selectable
                    and not self._intercept_retry_suppressed(
                        selected, observation
                    )
                    and self._window_valid(selected, observation)
                ):
                    return selected
                if self._intercept_retry_suppressed(selected, observation):
                    _hook(
                        self.env,
                        (
                            "release_selected_target",
                            "release_target_reservation",
                        ),
                        ((_object_id(selected),),),
                    )

        candidates = []
        for target in self._objects(observation):
            status = _status_name(target)
            if status not in SELECTABLE_OBJECT_STATES:
                continue
            if self._intercept_retry_suppressed(target, observation):
                continue
            if _object_id(target) is None or not self._window_valid(target, observation):
                continue
            candidates.append(target)
        return min(candidates, key=self._candidate_key) if candidates else None

    def _peek_target(self, observation: Mapping[str, Any]) -> Any | None:
        """Return the next visible candidate without reserving environment state."""

        candidates = []
        for target in self._objects(observation):
            if _status_name(target) not in SELECTABLE_OBJECT_STATES:
                continue
            if self._intercept_retry_suppressed(target, observation):
                continue
            if _object_id(target) is None or not self._window_valid(
                target, observation
            ):
                continue
            candidates.append(target)
        return min(candidates, key=self._candidate_key) if candidates else None

    def _prediction_from_mapping(
        self, observation: Mapping[str, Any], target_id: Any
    ) -> Any:
        predictions = _value(
            observation,
            ("predictions", "target_predictions", "motion_predictions"),
        )
        if not isinstance(predictions, Mapping):
            direct = _value(
                observation,
                ("predicted_grasp_pose", "target_prediction"),
            )
            selected_id = _value(
                observation, ("selected_target_id", "target_id")
            )
            return direct if selected_id in {None, target_id, str(target_id)} else None
        if target_id in predictions:
            return predictions[target_id]
        return predictions.get(str(target_id))

    def _history_for_target(
        self, target: Any, observation: Mapping[str, Any]
    ) -> Any:
        history = _value(
            target,
            (
                "history",
                "position_history",
                "state_history",
                "observations",
            ),
        )
        if history is not None:
            return history
        histories = _value(
            observation,
            ("object_history", "target_history", "history"),
        )
        if isinstance(histories, Mapping):
            target_id = _object_id(target)
            return histories.get(target_id, histories.get(str(target_id)))
        return None

    def _predict_with_standard_api(
        self, target: Any, observation: Mapping[str, Any]
    ) -> Any:
        """Adapter for perception.MotionPredictor without importing it."""

        callback = getattr(self.predictor, "predict", None)
        history = self._history_for_target(target, observation)
        if not callable(callback) or history is None:
            return None
        now = _value(observation, ("timestamp", "sim_time", "time"), self._internal_time)
        latency = _value(
            observation,
            ("system_latency", "total_latency", "latency"),
            0.0,
        )
        if isinstance(latency, Mapping):
            latency = _value(latency, ("total", "total_latency"), 0.0)
        belt_velocity = self._belt_velocity(observation)
        try:
            return callback(
                history,
                current_time=float(now),
                system_latency=float(latency),
                robot_arrival_time=self.config.prediction_lead_time,
                belt_velocity=belt_velocity,
                enable_motion_compensation=self._motion_compensation(),
                object_id=_object_id(target),
            )
        except TypeError:
            # A custom predictor may expose a smaller signature; the generic
            # duck-typed calls below still get a chance.
            return None

    def _pose_source(
        self, target: Any, observation: Mapping[str, Any]
    ) -> tuple[np.ndarray | None, bool]:
        """Return target/object pose and whether it is already an exact EE pose."""

        target_id = _object_id(target)
        if self._motion_compensation():
            observation_prediction = self._prediction_from_mapping(
                observation, target_id
            )
            if _prediction_validity(observation_prediction) is False:
                self._last_error = str(
                    _value(
                        observation_prediction,
                        "reason",
                        "invalid_prediction",
                    )
                )
                return None, False
            direct_names = (
                "predicted_grasp_pose",
                "future_grasp_pose",
                "predicted_pose",
                "predicted_position",
            )
            prediction = _hook(
                self.env,
                ("get_predicted_grasp_pose", "predict_grasp_pose", "predict_target"),
                (
                    (target_id, observation),
                    (target, observation),
                    (target_id,),
                    (target,),
                ),
            )
            if prediction is None:
                prediction = self._predict_with_standard_api(target, observation)
            if prediction is None and self._history_for_target(
                target, observation
            ) is None:
                prediction = _hook(
                    self.predictor,
                    ("predict_grasp_pose", "predict", "get_prediction"),
                    ((target, observation), (target,), (target_id, observation)),
                )
            if prediction is None:
                prediction = self._prediction_from_mapping(observation, target_id)
            if prediction is None:
                prediction = _value(target, direct_names)
            if _prediction_validity(prediction) is False:
                self._last_error = str(
                    _value(prediction, "reason", "invalid_prediction")
                )
                return None, False
            predicted_value = _value(
                prediction,
                (
                    "predicted_grasp_pose",
                    "predicted_position",
                    "future_position",
                ),
                prediction,
            )
            parsed = _as_pose(predicted_value, self._grasp_quaternion)
            if parsed is not None:
                return parsed, False

        observed = _value(
            target,
            (
                "observed_pose",
                "observed_position",
                "historical_pose",
                "pose",
                "position",
            ),
        )
        parsed = _as_pose(observed, self._grasp_quaternion)
        if parsed is None:
            return None, False

        if self._motion_compensation():
            velocity = _value(
                target, ("estimated_velocity", "velocity", "linear_velocity")
            )
            if velocity is not None:
                try:
                    velocity_values = np.asarray(velocity, dtype=float).reshape(-1)
                    parsed[:3] += (
                        as_vector(
                            self._belt_frame()[1] * velocity_values[0]
                            if velocity_values.size == 1 else velocity_values,
                            3, name="target velocity",
                        )
                        * self.config.prediction_lead_time
                    )
                except ValueError:
                    pass
        return parsed, False

    def _grasp_target_pose(
        self,
        target: Any,
        observation: Mapping[str, Any],
        *,
        remember_orientation: bool = True,
    ) -> np.ndarray | None:
        pose, exact_ee_pose = self._pose_source(target, observation)
        if pose is None:
            return None
        local_closure = self._local_closure_reference(target, observation)
        exact = _value(
            target,
            ("predicted_ee_grasp_pose", "ee_grasp_pose", "gripper_grasp_pose"),
        )
        exact_pose = _as_pose(exact, self._grasp_quaternion)
        if exact_pose is not None:
            if local_closure is not None:
                exact_pose[:3] = self._retimed_grasp_position(exact_pose[:3], local_closure[0])
            return exact_pose
        if not exact_ee_pose:
            height = _value(
                target,
                ("grasp_height", "recommended_grasp_height"),
                self.config.grasp_clearance,
            )
            try:
                pose[2] += float(height)
            except (TypeError, ValueError):
                pose[2] += self.config.grasp_clearance
        explicit_orientation = _value(
            target,
            ("grasp_quaternion", "grasp_orientation"),
            None,
        )
        if explicit_orientation is None:
            selected_quaternion = self._selected_grasp_quaternion
            if selected_quaternion is None or not remember_orientation:
                selected_quaternion = self._adaptive_grasp_quaternion(target)
                if remember_orientation:
                    self._selected_grasp_quaternion = selected_quaternion.copy()
            pose[3:] = selected_quaternion
        else:
            pose[3:] = normalized_quat(explicit_orientation)
        if local_closure is not None:
            pose[:3] = self._retimed_grasp_position(pose[:3], local_closure[0])
        return pose

    def _adaptive_grasp_yaw_calibration(
        self, target: Any
    ) -> tuple[float, float, float] | None:
        """Return ``(preferred, lower, upper)`` calibrated world-yaw deltas."""

        mode = str(_value(target, "grasp_yaw_mode", "fixed")).lower()
        if mode not in {"short_axis", "object_short_axis"}:
            return None

        observed = _value(
            target,
            ("observed_pose", "historical_pose", "pose", "quaternion"),
            None,
        )
        try:
            raw = np.asarray(observed, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if raw.size == 7:
            object_quaternion = raw[3:]
        elif raw.size == 4:
            object_quaternion = raw
        else:
            return None
        try:
            object_yaw = _yaw_from_quaternion(object_quaternion)
        except ValueError:
            return None

        # Bundled anisotropic parts use local X as their long axis, so their
        # local Y / yaw+pi/2 direction is the desired parallel-jaw axis.
        desired_axis_yaw = object_yaw + 0.5 * math.pi
        nominal_axis_yaw = _jaw_axis_yaw(self._grasp_quaternion)
        delta = _wrap_undirected_axis_angle(
            desired_axis_yaw - nominal_axis_yaw
        )
        if mode == "object_short_axis":
            # The preferred orientation is derived from this workpiece pose,
            # not from a category-wide world-yaw calibration.  Candidate
            # fallbacks may move toward the nominal wrist orientation only
            # within the declared jaw-axis error; otherwise the planner must
            # reject this workpiece instead of executing a geometrically wrong
            # but kinematically convenient grasp.
            allowed_error = self.config.maximum_grasp_axis_misalignment
            fallback = math.copysign(
                max(0.0, abs(delta) - allowed_error), delta
            )
            lower, upper = sorted((float(delta), float(fallback)))
            return float(delta), lower, upper

        bounds = _value(target, "grasp_yaw_delta_range", None)
        try:
            lower, upper = (float(item) for item in bounds)
        except (TypeError, ValueError):
            return None
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower > upper
        ):
            return None
        delta = float(np.clip(delta, lower, upper))
        return delta, lower, upper

    def _grasp_quaternion_for_yaw_delta(self, delta: float) -> np.ndarray:
        world_yaw = np.array(
            [
                math.cos(0.5 * delta),
                0.0,
                0.0,
                math.sin(0.5 * delta),
            ]
        )
        return normalized_quat(
            quat_multiply(world_yaw, self._grasp_quaternion)
        )

    def _adaptive_grasp_quaternion_candidates(
        self, target: Any
    ) -> list[np.ndarray]:
        """Prefer short-axis alignment, then safely step toward nominal yaw.

        A long anisotropic part still receives its requested jaw alignment
        first.  If that exact wrist yaw has no safe IK branch, callers may try
        the following calibrated candidates.  Every candidate must still pass
        the full IK, joint-margin, orientation, collision and RRT edge checks.
        """

        calibration = self._adaptive_grasp_yaw_calibration(target)
        if calibration is None:
            return [self._grasp_quaternion.copy()]
        preferred, lower, upper = calibration
        deltas = [preferred]
        cursor = preferred
        step = self.config.adaptive_grasp_yaw_fallback_step
        # Move toward the member of the calibrated interval closest to nominal
        # zero yaw.  The old one-sided decrement was correct for [0,+] ranges
        # but drove a negative interval such as cat4 [-.20,-.10] farther away
        # from nominal and toward the wrist limit.
        fallback = float(np.clip(0.0, lower, upper))
        while abs(cursor - fallback) > 1e-9:
            direction = math.copysign(1.0, fallback - cursor)
            cursor = cursor + direction * min(step, abs(fallback - cursor))
            if abs(cursor - deltas[-1]) <= 1e-9:
                break
            deltas.append(cursor)
        return [
            self._grasp_quaternion_for_yaw_delta(delta)
            for delta in deltas
        ]

    def _prefetch_grasp_quaternion_candidates(
        self, target: Any
    ) -> list[np.ndarray]:
        """Prefer equivalent grasps aligned with the following carry corridor.

        A continuous-rotational part has no physical yaw preference.  The
        worker returns the first feasible orientation, so prioritizing the
        preceding bin's retreat yaw saves only the empty return motion while
        moving its wrist unwind into the slower, loaded lift and transfer.
        Try the canonical belt/carry orientation first and retain the
        retreat-nearest equivalent as a fallback.  Every candidate still
        requires the complete continuation, IK, cone, collision, singularity,
        margin, edge and deadline certificates.  Anisotropic short-axis parts
        retain their calibrated ordering unchanged.
        """

        candidates = self._adaptive_grasp_quaternion_candidates(target)
        symmetry = _value(target, ("symmetry",), {})
        kind = str(_value(symmetry, ("kind",), "none")).lower()
        anchor_pose = self._prefetch_retreat_pose
        if kind != "continuous_rotational" or anchor_pose is None:
            return candidates
        try:
            yaw_delta = _jaw_axis_yaw(anchor_pose[3:]) - _jaw_axis_yaw(
                self._grasp_quaternion
            )
            nearest = self._grasp_quaternion_for_yaw_delta(yaw_delta)
        except (TypeError, ValueError):
            return candidates
        ordered: list[np.ndarray] = []
        for quaternion in (*candidates, nearest):
            if not any(
                quat_distance(quaternion, existing) <= 1e-9
                for existing in ordered
            ):
                ordered.append(quaternion.copy())
        return ordered

    def _adaptive_grasp_quaternion(self, target: Any) -> np.ndarray:
        """Return the preferred calibrated short-axis grasp orientation."""

        return self._adaptive_grasp_quaternion_candidates(target)[0].copy()

    def _bin_index(self, target: Any) -> int:
        explicit = _value(
            target, ("bin_id", "bin_index", "target_bin", "sorting_bin")
        )
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass
        category = _value(target, ("category_id", "class_id"), 0)
        try:
            return int(category) // self.config.category_bin_size
        except (TypeError, ValueError):
            return 0

    def _bin_pose(
        self, target: Any, observation: Mapping[str, Any]
    ) -> np.ndarray | None:
        bin_index = self._bin_index(target)
        pose = _hook(
            self.env,
            ("get_bin_drop_pose", "get_bin_pose", "bin_drop_pose"),
            ((bin_index,), (bin_index, observation), (target,)),
        )
        if pose is None:
            bins = _value(observation, ("bins", "bin_poses"))
            if isinstance(bins, Mapping):
                pose = bins.get(bin_index, bins.get(str(bin_index)))
            elif isinstance(bins, Sequence) and not isinstance(bins, (str, bytes)):
                if 0 <= bin_index < len(bins):
                    pose = bins[bin_index]
        if pose is None:
            pose = self.config.bin_poses.get(
                bin_index, self.config.bin_poses.get(str(bin_index))
            )
        parsed = _as_pose(pose, self._place_quaternion)
        if parsed is None:
            return None
        parsed[2] += self.config.place_clearance
        # A calibrated bin pose may need a different wrist branch for each
        # side of the robot.  Preserve that orientation unless the caller
        # explicitly requested one global place quaternion.
        if self._force_place_quaternion:
            parsed[3:] = self._place_quaternion
        return parsed

    def _ee_pose(self, observation: Mapping[str, Any]) -> np.ndarray:
        pose = _as_pose(
            _value(observation, ("ee_pose", "end_effector_pose")),
            self._grasp_quaternion,
        )
        if pose is None:
            position = _value(observation, ("ee_position", "end_effector_position"))
            quaternion = _value(
                observation,
                ("ee_quat", "ee_quaternion", "end_effector_quaternion"),
                self._grasp_quaternion,
            )
            if position is not None:
                pose = np.r_[
                    as_vector(position, 3, name="ee_position"),
                    normalized_quat(quaternion),
                ]
        if pose is None:
            state = _hook(
                self.env,
                ("get_robot_state", "get_state"),
                ((),),
            )
            if isinstance(state, Mapping):
                pose = _as_pose(
                    _value(state, ("ee_pose", "end_effector_pose")),
                    self._grasp_quaternion,
                )
        if pose is None:
            raise KeyError("observation does not provide an end-effector pose")
        return pose

    def _joint_positions(self, observation: Mapping[str, Any]) -> np.ndarray:
        positions = _value(observation, ("joint_pos", "joint_positions", "qpos"))
        if positions is None:
            positions = _hook(
                self.env,
                ("get_joint_positions",),
                ((),),
            )
        if positions is None:
            controller = _value(self.env, ("robot_controller", "controller"))
            positions = _hook(controller, ("get_joint_positions",), ((),))
        if positions is None:
            raise KeyError(
                "joint_position policy mode needs observation['joint_pos']"
            )
        positions = np.asarray(positions, dtype=float).reshape(-1)
        if positions.size < 7:
            raise ValueError("joint_pos must contain at least seven arm positions")
        return positions[:7].copy()

    @staticmethod
    def _joint_command_history(
        observation: Mapping[str, Any], fallback_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the controller's accepted target position and velocity.

        Older adapters/tests may not expose command history; for those callers
        the current edge start is the conservative zero-velocity default.  A
        production observation that exposes malformed history never receives
        that compatibility fallback and therefore cannot start passthrough.
        """

        target_raw = _value(
            observation,
            ("joint_target", "joint_command_target"),
            None,
        )
        velocity_raw = _value(
            observation,
            ("joint_target_velocity", "joint_command_velocity"),
            None,
        )
        if target_raw is None and velocity_raw is None:
            return np.asarray(fallback_q, dtype=float).reshape(7).copy(), np.zeros(7)
        try:
            target = np.asarray(target_raw, dtype=float).reshape(-1)[:7]
            velocity = np.asarray(velocity_raw, dtype=float).reshape(-1)[:7]
        except (TypeError, ValueError):
            return np.full(7, np.nan), np.full(7, np.nan)
        if (
            target.shape != (7,)
            or velocity.shape != (7,)
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(velocity))
        ):
            return np.full(7, np.nan), np.full(7, np.nan)
        return target.copy(), velocity.copy()

    def _downward_alignment(self, quaternion: Any) -> float:
        """Return the tool approach-axis dot product with world down."""

        w, x, y, z = normalized_quat(quaternion)
        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )
        world_axis = rotation @ self._approach_axis_local
        return float(-world_axis[2])

    def _orientation_is_safe(self, quaternion: Any) -> bool:
        return bool(
            self._downward_alignment(quaternion)
            >= self._local_downward_alignment_limit() - 1e-9
        )

    def _local_downward_alignment_limit(self) -> float:
        if self._stage in {
            PolicyStage.PREGRASP,
            PolicyStage.TRACKING_DESCENT,
            PolicyStage.CLOSING,
            PolicyStage.LIFTING,
        }:
            return self.config.acquisition_minimum_downward_alignment
        return self.config.minimum_downward_alignment

    def _planner_downward_alignment_limit(self) -> float:
        # PREGRASP and the first carrying move always stay inside the strict
        # acquisition cone.  The latter prevents a freshly secured part from
        # being swung through the wider transfer cone before it reaches the
        # calibrated overhead corridor.
        # RETURNING may need the wider, still-downward transit cone to reach
        # the calibrated wait pose; opportunistic pre-emption has its own
        # measured, hysteretic `_acquisition_entry_ready` gate, so a low
        # return posture can never be relabelled as PREGRASP.
        if self._stage in {PolicyStage.PREGRASP, PolicyStage.LIFTING}:
            return self.config.acquisition_minimum_downward_alignment
        return self.config.planner_minimum_downward_alignment

    def _acquisition_entry_ready(self, alignment: float) -> bool:
        """Enter conservatively, then hold a strict-cone Schmitt latch.

        A disabled acquisition cone has no directional entry preference;
        path geometry and actual dispatch validation still apply independently.
        Two non-descending samples above the hysteretic entry threshold are
        still required before any speculative route may commit.  Once that
        condition has been earned, tiny measured decreases must not starve a
        completed background result while the tool remains inside the *hard*
        acquisition cone.  Falling below that original safety threshold clears
        the latch immediately; first-edge and every command edge retain their
        independent 0.52 alignment validation.
        """

        hard_threshold = self.config.acquisition_minimum_downward_alignment
        if hard_threshold <= -1.0:
            return True
        threshold = (
            hard_threshold
            + self.config.acquisition_entry_alignment_hysteresis
        )
        if self._return_acquisition_latched:
            if alignment >= hard_threshold - 1e-9:
                self._previous_return_alignment = float(alignment)
                return True
            self._return_acquisition_latched = False
            self._return_alignment_stable_steps = 0
        non_descending = (
            self._previous_return_alignment is None
            or alignment >= self._previous_return_alignment - 1e-5
        )
        if alignment >= threshold and non_descending:
            self._return_alignment_stable_steps += 1
        else:
            self._return_alignment_stable_steps = 0
        self._previous_return_alignment = float(alignment)
        ready = (
            self._return_alignment_stable_steps
            >= self.config.acquisition_entry_stable_steps
        )
        self._return_acquisition_latched = bool(ready)
        return ready

    def _joint_safety_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        controller = _value(self.env, ("robot_controller", "controller"))
        lower = _value(controller, "joint_lower_limits")
        upper = _value(controller, "joint_upper_limits")
        if lower is None or upper is None:
            return None
        lower_values = np.asarray(lower, dtype=float).reshape(-1)
        upper_values = np.asarray(upper, dtype=float).reshape(-1)
        if lower_values.size < 7 or upper_values.size < 7:
            return None
        return lower_values[:7], upper_values[:7]

    def _joint_limit_margin(self, positions: np.ndarray) -> float | None:
        limits = self._joint_safety_limits()
        if limits is None:
            return None
        lower, upper = limits
        return float(np.min(np.minimum(positions - lower, upper - positions)))

    def _jacobian_condition(self, positions: np.ndarray) -> float | None:
        solver = _value(self.env, ("ik", "ik_solver"))
        jacobian = _hook(solver, ("compute_jacobian",), ((positions,),))
        if jacobian is None:
            return None
        values = np.asarray(jacobian, dtype=float)
        if values.shape != (6, 7) or not np.all(np.isfinite(values)):
            return None
        # Match the lower orientation weight used by the DLS solver so the
        # diagnostic describes the task actually being controlled.
        scaled = values.copy()
        scaled[3:] *= 0.30
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        if singular_values[-1] <= 1e-12:
            return float("inf")
        return float(singular_values[0] / singular_values[-1])

    def _local_edge_is_safe(
        self,
        start: np.ndarray,
        end: np.ndarray,
        target_pose: np.ndarray,
    ) -> bool:
        """Densely validate one short Cartesian-servo joint edge."""

        validator = getattr(self.env, "is_arm_configuration_safe", None)
        if not callable(validator):
            return True
        solver = _value(self.env, ("ik", "ik_solver"))
        start_pose = _hook(solver, ("forward_pose",), ((start,),))
        start_height = (
            float(np.asarray(start_pose, dtype=float)[2])
            if start_pose is not None
            and np.asarray(start_pose, dtype=float).size >= 3
            else float(target_pose[2])
        )
        minimum_height = min(start_height, float(target_pose[2])) - 0.015
        samples = max(
            1,
            int(math.ceil(np.max(np.abs(end - start)) / 0.025)),
        )
        for index in range(1, samples + 1):
            state = start + (end - start) * (index / samples)
            if not validator(
                state,
                approach_axis_local=self._approach_axis_local,
                minimum_downward_alignment=(
                    self._local_downward_alignment_limit()
                ),
                minimum_tcp_height=minimum_height,
                minimum_joint_margin=self.config.minimum_joint_limit_margin,
            ):
                return False
        return True

    def _safe_calibrated_ik_alternative(
        self,
        target_pose: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray | None:
        """Try known collision-free NERO redundancy anchors lazily.

        The first local DLS result is normally the smoothest branch.  Near the
        conveyor it can nevertheless drift into an elbow-down redundancy that
        puts link3 on the belt.  Re-seeding from measured, collision-checked
        calibration postures supplies another solution without relaxing any
        pose, cone, collision, singularity, or joint-limit constraint.
        """

        # A real ConveyorSortingEnv owns an isolated background planner.  Its
        # 50 Hz control thread must never fall back to calibration-seeded IK:
        # two DLS seeds here can each consume hundreds of MuJoCo forwards and
        # were another source of visible viewer stalls.  Lightweight external
        # environments without a worker retain the legacy compatibility path.
        if (
            self._planning_worker is not None
            or callable(
                getattr(self.env, "create_global_planning_worker", None)
            )
        ):
            return None

        solver = _value(self.env, ("ik", "ik_solver"))
        solve = getattr(solver, "solve", None)
        if not callable(solve):
            return None
        robot_config = getattr(self.env, "robot_config", {})
        calibrations = (
            robot_config.get("calibration_poses", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if not isinstance(calibrations, Mapping):
            return None
        if self._stage in {
            PolicyStage.TRACKING_DESCENT,
            PolicyStage.CLOSING,
        }:
            names = ("belt_intercept_grasp", "belt_intercept_pregrasp")
        elif self._stage == PolicyStage.LIFTING:
            names = ("belt_intercept_pregrasp", "waiting_top_down")
        else:
            names = ("belt_intercept_pregrasp", "belt_intercept_grasp")

        current_condition = self._jacobian_condition(current)
        for name in names:
            calibration = calibrations.get(name)
            seed_value = (
                calibration.get("joint_positions")
                if isinstance(calibration, Mapping)
                else None
            )
            if seed_value is None:
                continue
            seed = np.asarray(seed_value, dtype=float).reshape(-1)
            if seed.shape != (7,) or not np.all(np.isfinite(seed)):
                continue
            try:
                result = solve(
                    target_pose[:3],
                    target_pose[3:],
                    qpos_seed=seed,
                )
            except (TypeError, ValueError):
                continue
            if result is None or _value(result, "success", False) is not True:
                continue
            candidate = np.asarray(
                _value(result, ("joint_positions", "qpos"), result),
                dtype=float,
            ).reshape(-1)
            if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                continue
            if (
                np.max(np.abs(candidate - current))
                > self.config.max_verified_ik_transition
            ):
                continue
            margin = self._joint_limit_margin(candidate)
            if (
                margin is not None
                and margin < self.config.minimum_joint_limit_margin
            ):
                continue
            condition = self._jacobian_condition(candidate)
            if (
                condition is not None
                and condition > self.config.singularity_hard_stop_condition
                and (current_condition is None or condition >= current_condition)
            ):
                continue
            if self._local_edge_is_safe(current, candidate, target_pose):
                return candidate.copy()
        return None

    def _calibrated_ik_seed_for_retry(
        self, retry_index: int
    ) -> np.ndarray | None:
        """Return one deterministic redundancy seed for this control tick.

        The old local recovery tried both seeds synchronously after an unsafe
        warm solution.  Production now asks for at most one seed per tick and
        lets the ordinary bounded retry state advance to the next seed.
        """

        if retry_index < 0:
            return None
        if self._stage in {
            PolicyStage.TRACKING_DESCENT,
            PolicyStage.CLOSING,
        }:
            names = ("belt_intercept_grasp", "belt_intercept_pregrasp")
        elif self._stage == PolicyStage.LIFTING:
            names = ("belt_intercept_pregrasp", "waiting_top_down")
        else:
            names = ("belt_intercept_pregrasp", "belt_intercept_grasp")
        if retry_index >= len(names):
            return None
        robot_config = getattr(self.env, "robot_config", {})
        calibrations = (
            robot_config.get("calibration_poses", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        calibration = (
            calibrations.get(names[retry_index])
            if isinstance(calibrations, Mapping)
            else None
        )
        value = (
            calibration.get("joint_positions")
            if isinstance(calibration, Mapping)
            else None
        )
        if value is None:
            return None
        seed = np.asarray(value, dtype=float).reshape(-1)
        if seed.shape != (7,) or not np.all(np.isfinite(seed)):
            return None
        return seed.copy()

    def _cartesian_servo_reference(
        self,
        observation: Mapping[str, Any],
        *,
        dynamic_acquisition: bool,
    ) -> np.ndarray:
        """Select the Cartesian origin without changing the speed profile.

        Hardware acquisition adapters can retain a bounded command-space
        integrator. The ordinary baseline keeps its measured-state origin
        for moving acquisition and release.
        """

        return (
            self._ee_pose(observation)
            if dynamic_acquisition or self._cartesian_command_pose is None
            else self._cartesian_command_pose.copy()
        )

    def _local_translation_step_limit(self, observation: Mapping[str, Any]) -> float:
        """Shared displacement budget for local Cartesian command proposals."""
        limit = self.config.max_translation_step
        if self.config.local_tcp_linear_speed_limit_enabled:
            limit = min(limit, self.config.cartesian_linear_speed * self._execution_dt(observation))
        if (self._full_pick_contact_handed_off and self._stage==PolicyStage.LIFTING
                and self._full_pick_handoff_lift_speed is not None):
            limit=min(limit,self._full_pick_handoff_lift_speed*self._execution_dt(observation))
        return float(limit)

    def _cartesian_servo_pose(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Advance toward the target using the configured TCP speed profile."""

        dynamic_acquisition = bool(
            self.config.dynamic_intercept_tracking
            and (self._stage in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
                 or (self._stage == PolicyStage.RELEASING
                     and callable(getattr(self.env, 'get_release_pose', None))))
        )
        # The joint controller can limit a successful IK request before the
        # plant executes it. During conveyor tracking, integrating from that
        # unexecuted Cartesian request accumulates error and can drive the TCP
        # below the workpiece while the intended pose remains above it. Start
        # each moving acquisition/release increment from measured feedback,
        # matching the measured-state DLS seed used by _arm_for_pose.
        current = self._cartesian_servo_reference(
            observation, dynamic_acquisition=dynamic_acquisition,
        )
        translation = target_pose[:3] - current[:3]
        angular_handoff=self._contact_handoff_global_angular_speed()
        if (not self.config.tcp_speed_limits_enabled and not self._clearing_payload()
                and not dynamic_acquisition and angular_handoff is None):
            # IK branch/edge validation and actuator joint-rate limits still
            # apply. Retry scaling is numerical backtracking, not a TCP cap.
            rotation = quat_error_vector(target_pose[3:], current[3:])
            return np.r_[
                current[:3] + translation * float(scale),
                compose_world_delta(current[3:], rotation * float(scale)),
            ]
        translation_limit = self._local_translation_step_limit(observation) * float(scale)
        if self._full_pick_contact_handed_off and self._stage==PolicyStage.LIFTING:
            # Preserve the previous accepted command as integration origin.
            # Spend half the remaining measured-feedback envelope on progress;
            # do not advance another full step while the plant is catching up.
            lead=float(np.linalg.norm(current[:3]-self._ee_pose(observation)[:3]))
            headroom=max(0.,self.config.max_translation_step-lead)
            translation_limit=min(translation_limit,.5*headroom)
            self._full_pick_metrics.update(handoff_servo_existing_lead_m=lead,
                handoff_servo_headroom_m=headroom,handoff_servo_step_limit_m=translation_limit)
        distance = float(np.linalg.norm(translation))
        if distance > translation_limit:
            translation *= translation_limit / distance

        rotation = quat_error_vector(target_pose[3:], current[3:])
        rotation_limit = min(
            self.config.max_rotation_step,
            min(self.config.cartesian_angular_speed,
                angular_handoff if angular_handoff is not None else self.config.cartesian_angular_speed)
                * self._execution_dt(observation),
        ) * float(scale)
        angle = float(np.linalg.norm(rotation))
        if angle > rotation_limit:
            rotation *= rotation_limit / angle

        return np.r_[
            current[:3] + translation,
            compose_world_delta(current[3:], rotation),
        ]

    def _solve_pose(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        qpos_seed: np.ndarray | None = None,
        enforce_safety: bool = False,
        allow_verified_transition: bool = False,
    ) -> np.ndarray | None:
        current = self._joint_positions(observation)
        reference = current if qpos_seed is None else np.asarray(qpos_seed, dtype=float)
        solved = None
        production_realtime = callable(
            getattr(self.env, "create_global_planning_worker", None)
        )
        rich_solver = getattr(self.env, "solve_ik_result", None)
        if production_realtime and not callable(rich_solver):
            self._last_error = "real-time single-seed IK solver unavailable"
            return None
        if callable(rich_solver):
            supports_realtime_flag = True
            try:
                signature = inspect.signature(rich_solver)
                signature.bind(
                    target_pose[:3], target_pose[3:], reference,
                    allow_multiseed=False,
                )
            except TypeError:
                supports_realtime_flag = False
            except (ValueError, AttributeError):
                # Some extension callables do not expose a Python signature;
                # optimistically use the new contract once, without retrying
                # if their implementation itself raises TypeError.
                supports_realtime_flag = True
            if production_realtime and not supports_realtime_flag:
                # An old rich-solver signature provides no way to prove that
                # it will avoid its internal multi-seed fallback.  Production
                # therefore holds until the explicit bounded contract is
                # available; only lightweight compatibility environments may
                # call that legacy interface.
                self._last_error = (
                    "real-time IK solver lacks single-seed contract"
                )
                return None
            try:
                if supports_realtime_flag:
                    solved = rich_solver(
                        target_pose[:3],
                        target_pose[3:],
                        reference,
                        allow_multiseed=False,
                    )
                else:
                    # Compatibility with solvers predating the explicit
                    # real-time keyword.  Signature binding, rather than
                    # catching a callback's own TypeError, selects this path.
                    solved = rich_solver(
                        target_pose[:3],
                        target_pose[3:],
                        reference,
                    )
            except ValueError:
                solved = None
            if solved is not None and _value(solved, "success", True) is False:
                position_error = _value(solved, "position_error", float("nan"))
                orientation_error = _value(
                    solved, "orientation_error", float("nan")
                )
                # Grasp acquisition keeps the strict solver tolerances.  High,
                # collision-free transfer/return waypoints may use a bounded
                # near solution that is still far inside the physical policy
                # tolerances; this avoids retrying a 3.7 mm residual forever.
                near_safe_waypoint = (
                    self._stage
                    in {
                        PolicyStage.LIFTING,
                        PolicyStage.TRANSFERRING,
                        PolicyStage.RELEASING,
                        PolicyStage.RETURNING,
                        PolicyStage.RECOVERING,
                    }
                    and float(position_error) <= 0.005
                    and float(orientation_error) <= 0.040
                )
                if not near_safe_waypoint:
                    self._last_error = (
                        "IK did not converge "
                        f"(position_error={float(position_error):.4f}, "
                        f"orientation_error={float(orientation_error):.4f})"
                    )
                    return None
            if solved is None and callable(
                getattr(self.env, "create_global_planning_worker", None)
            ):
                # A production rich solver that declines or fails one bounded
                # warm attempt must hold.  Falling through to env.solve_ik
                # would silently re-enable its default multi-seed search on
                # the 50 Hz/viewer thread.
                self._last_error = "real-time warm-start IK unavailable"
                return None
        if solved is None:
            solved = _hook(
                self.env,
                ("solve_ik", "inverse_kinematics"),
                (
                    (target_pose[:3], target_pose[3:], reference),
                    (target_pose[:3], target_pose[3:]),
                    (target_pose,),
                ),
            )
        if solved is None:
            controller = _value(self.env, ("robot_controller", "controller"))
            solver = _value(controller, ("ik_solver", "ik"))
            if solver is None:
                solver = _value(self.env, ("ik", "ik_solver"))
            solved = _hook(
                solver,
                ("solve", "solve_joint_positions"),
                (
                    (target_pose[:3], target_pose[3:]),
                    (target_pose,),
                ),
            )
        if solved is None:
            return None
        solved = _value(solved, ("joint_positions", "qpos"), solved)
        vector = np.asarray(solved, dtype=float).reshape(-1)
        if vector.size < 7 or not np.all(np.isfinite(vector[:7])):
            return None
        candidate = vector[:7].copy()
        if enforce_safety:
            joint_step = float(np.max(np.abs(candidate - current)))
            if joint_step > self.config.max_ik_joint_step:
                # A warm-started DLS solve can change redundancy branches near
                # the conveyor even though a nearby calibrated branch remains
                # reachable.  Do not command that raw jump.  Instead, try the
                # bounded calibration-seeded solution whose *complete* joint
                # chord is checked by ``_safe_calibrated_ik_alternative`` for
                # collisions, the downward-orientation cone, TCP clearance and
                # the soft joint-limit margin.  This is deliberately stricter
                # than merely raising ``max_ik_joint_step``.
                verified_transition = bool(
                    allow_verified_transition
                    and joint_step
                    <= self.config.max_verified_ik_transition + 1e-9
                )
                alternate = (
                    None
                    if verified_transition
                    else self._safe_calibrated_ik_alternative(
                        target_pose, current
                    )
                )
                if not verified_transition and alternate is None:
                    self._last_error = (
                        "IK branch jump rejected "
                        f"(max_joint_step={joint_step:.4f} rad)"
                    )
                    return None
                if alternate is not None:
                    candidate = alternate
            margin = self._joint_limit_margin(candidate)
            if margin is not None:
                self._minimum_observed_joint_margin = (
                    margin
                    if self._minimum_observed_joint_margin is None
                    else min(self._minimum_observed_joint_margin, margin)
                )
                if margin < self.config.minimum_joint_limit_margin:
                    self._last_error = (
                        "IK soft joint limit rejected "
                        f"(margin={margin:.4f} rad)"
                    )
                    return None
            condition = self._jacobian_condition(candidate)
            self._last_jacobian_condition = condition
            if condition is not None:
                current_condition = self._jacobian_condition(current)
                worsening = (
                    current_condition is None or condition >= current_condition
                )
                if (
                    condition > self.config.singularity_hard_stop_condition
                    and worsening
                ):
                    self._last_error = (
                        "IK singularity hard stop "
                        f"(condition={condition:.2f})"
                    )
                    return None
            if not self._local_edge_is_safe(current, candidate, target_pose):
                alternate = self._safe_calibrated_ik_alternative(
                    target_pose, current
                )
                if alternate is None:
                    self._last_error = (
                        "local Cartesian edge rejected by collision/"
                        "orientation safety validation"
                    )
                    return None
                candidate = alternate
        self._last_ik_solution = candidate
        self._last_error = None
        return self._last_ik_solution.copy()

    def _trajectory_limits(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = self.config.trajectory_velocity_limits
        acceleration = self.config.trajectory_acceleration_limits
        if not velocity:
            velocity = _value(
                self.env,
                ("max_joint_velocity", "joint_velocity_limits"),
                (1.4,) * 7,
            )
        if not acceleration:
            acceleration = _value(
                self.env,
                ("max_joint_acceleration", "joint_acceleration_limits"),
                (3.0,) * 7,
            )
        velocity_limits = np.broadcast_to(
            np.asarray(velocity, dtype=float), (7,)
        ).copy()
        acceleration_limits = np.broadcast_to(
            np.asarray(acceleration, dtype=float), (7,)
        ).copy()
        if self._gentle_carry_required():
            velocity_limits *= self.config.carrying_velocity_scale
            acceleration_limits *= self.config.carrying_acceleration_scale
        return velocity_limits, acceleration_limits

    def _nominal_trajectory_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return unscaled joint limits for a future non-carrying route.

        B is constructed while the previous object can still be in
        TRANSFERRING.  Calling ``_trajectory_limits`` there accidentally
        inherited that previous object's gentle carry scale and inflated the
        next open-gripper PREGRASP ETA, even though execution happens after
        release.  Resolve the configured/base limits without consulting the
        current policy stage; the eventual B executor applies these same
        nominal acquisition limits.
        """

        velocity = self.config.trajectory_velocity_limits
        acceleration = self.config.trajectory_acceleration_limits
        if not velocity:
            velocity = _value(
                self.env,
                ("max_joint_velocity", "joint_velocity_limits"),
                (1.4,) * 7,
            )
        if not acceleration:
            acceleration = _value(
                self.env,
                ("max_joint_acceleration", "joint_acceleration_limits"),
                (3.0,) * 7,
            )
        return (
            np.broadcast_to(np.asarray(velocity, dtype=float), (7,)).copy(),
            np.broadcast_to(
                np.asarray(acceleration, dtype=float), (7,)
            ).copy(),
        )

    def _carry_stopping_guard_horizon(
        self,
        joint_velocity: np.ndarray,
        observation: Mapping[str, Any] | None = None,
    ) -> float | None:
        """Return a conservative loaded-carry reaction-plus-stop horizon.

        A fixed two-tick velocity preview can still notice a narrow TCP-height
        crossing only after measured feedback has crossed it.  The executor
        already declares the acceleration envelope used to time this loaded
        route, so extend the preview by the slowest joint stopping time after
        the larger of the configured reaction floor and the complete control
        period plus communication/robot-response latency.  Extrapolating
        every joint at its current speed for that whole interval is
        deliberately more conservative than the corresponding deceleration
        ramp.  The resulting dense live edge is still checked with the
        original height/cone/margin/collision limits.
        """

        velocity = np.asarray(joint_velocity, dtype=float).reshape(-1)[:7]
        if velocity.shape != (7,) or not np.all(np.isfinite(velocity)):
            return None
        try:
            _velocity_limits, acceleration_limits = self._trajectory_limits()
            acceleration = np.asarray(
                acceleration_limits, dtype=float
            ).reshape(-1)[:7]
        except (TypeError, ValueError):
            return None
        if (
            acceleration.shape != (7,)
            or not np.all(np.isfinite(acceleration))
            or np.any(acceleration <= 0.0)
        ):
            return None
        stopping_time = float(np.max(np.abs(velocity) / acceleration))
        latency_sum = 0.0
        latency = _value(
            {} if observation is None else observation,
            "system_latency",
            {},
        )
        if isinstance(latency, Mapping):
            for name in ("control", "robot_response"):
                try:
                    component = float(_value(latency, name, 0.0))
                except (TypeError, ValueError):
                    component = 0.0
                if np.isfinite(component) and component > 0.0:
                    latency_sum += component
        reaction_time = max(
            float(self.config.carry_path_velocity_guard_horizon),
            self._execution_dt(observation) + latency_sum,
        )
        return reaction_time + stopping_time

    def _global_path_tcp_speed_limits_enabled(
        self, stage: PolicyStage | str | None = None
    ) -> bool:
        route_stage = self._stage if stage is None else stage
        pregrasp = route_stage in (PolicyStage.PREGRASP, PolicyStage.PREGRASP.value)
        return bool(self.config.tcp_speed_limits_enabled and not (
            self.config.dynamic_intercept_tracking and pregrasp
            and not self.config.dynamic_pregrasp_global_tcp_speed_limits_enabled
        ))

    def _edge_minimum_duration(
        self, start: np.ndarray, end: np.ndarray, *,
        global_path: bool = False, stage: PolicyStage | str | None = None,
    ) -> float:
        if np.max(np.abs(np.asarray(end) - np.asarray(start))) <= 1e-12:
            return 0.0
        minimum = self.config.trajectory_minimum_duration
        enabled = (self._global_path_tcp_speed_limits_enabled(stage)
                   if global_path else self.config.tcp_speed_limits_enabled)
        linear_limit=self._contact_handoff_global_linear_speed(stage) if global_path else None
        angular_limit=self._contact_handoff_global_angular_speed(stage) if global_path else None
        if angular_limit is not None:enabled=True
        if not enabled and linear_limit is None:
            return minimum
        solver = _value(self.env, ("ik", "ik_solver"))
        jacobian_at = getattr(solver, "compute_jacobian", None)
        if not callable(jacobian_at):
            return minimum
        try:
            angular_duration = (sampled_tcp_angular_minimum_duration(
                start,
                end,
                lambda positions: np.asarray(
                    jacobian_at(positions), dtype=float
                )[3:],
                min(self.config.maximum_tcp_angular_speed,
                    angular_limit if angular_limit is not None else self.config.maximum_tcp_angular_speed),
            ) if enabled else 0.)
            linear_duration=(sampled_tcp_angular_minimum_duration(start,end,
                lambda q:np.asarray(jacobian_at(q),dtype=float)[:3],linear_limit)
                if linear_limit is not None else 0.)
        except (TypeError, ValueError, RuntimeError):
            return minimum
        return max(minimum, angular_duration,linear_duration)

    def _contact_handoff_global_linear_speed(self,stage=None):
        selected=self._stage if stage is None else stage
        if (self._full_pick_contact_handed_off and selected in (
                PolicyStage.LIFTING,'lifting',PolicyStage.TRANSFERRING,'transferring',
                PolicyStage.RELEASING,'releasing',PolicyStage.RETURNING,'returning')
                and self._full_pick_handoff_lift_speed is not None):
            return self._full_pick_handoff_lift_speed
        return None

    def _contact_handoff_global_angular_speed(self,stage=None):
        selected=self._stage if stage is None else stage
        if (self._full_pick_contact_handed_off and selected in (
                PolicyStage.LIFTING,'lifting',PolicyStage.TRANSFERRING,'transferring',
                PolicyStage.RELEASING,'releasing',PolicyStage.RETURNING,'returning')):
            return self._full_pick_handoff_angular_speed
        return None

    def _gentle_carry_required(self) -> bool:
        if self._stage not in {
            PolicyStage.LIFTING,
            PolicyStage.TRANSFERRING,
            PolicyStage.RELEASING,
        }:
            return False
        category = _value(self._target, ("category_id", "class_id"), -1)
        try:
            category_id = int(category)
        except (TypeError, ValueError):
            return False
        return category_id in {
            int(item) for item in self.config.gentle_carry_categories
        }

    def _trajectory_goal_changed(self, target_pose: np.ndarray) -> bool:
        previous = self._trajectory_target_pose
        if previous is None or self._trajectory_stage != self._stage:
            return True
        return bool(
            np.linalg.norm(target_pose[:3] - previous[:3])
            > self.config.trajectory_replan_position_tolerance
            or quat_distance(target_pose[3:], previous[3:])
            > self.config.trajectory_replan_orientation_tolerance
        )

    def _start_joint_trajectory(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        solved = self._solve_pose(target_pose, observation)
        if solved is None:
            self._trajectory_target_pose = target_pose.copy()
            self._trajectory_stage = self._stage
            self._record_motion_plan_failure()
            return False
        current = self._joint_positions(observation)
        velocity_limits, acceleration_limits = self._trajectory_limits()
        self._joint_trajectory = QuinticJointTrajectory(
            current,
            solved,
            velocity_limits,
            acceleration_limits,
            minimum_duration=self._edge_minimum_duration(current, solved),
        )
        self._trajectory_started_at = self._timestamp_without_advancing(observation)
        self._trajectory_target_pose = target_pose.copy()
        self._trajectory_stage = self._stage
        self._record_motion_plan_success()
        return True

    def _timestamp_without_advancing(
        self, observation: Mapping[str, Any]
    ) -> float:
        raw = _value(observation, ("timestamp", "sim_time", "time"))
        if raw is None and self.env is not None:
            raw = _value(getattr(self.env, "data", None), "time")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = self._internal_time
        return value if np.isfinite(value) else self._internal_time

    def _pose_reached(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> bool:
        current = self._ee_pose(observation)
        return (
            np.linalg.norm(target_pose[:3] - current[:3])
            <= self.config.position_tolerance
            and quat_distance(target_pose[3:], current[3:])
            <= self.config.orientation_tolerance
        )

    def _pregrasp_pose_reached(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> bool:
        """Use the existing planning-position reuse range at the high handoff."""
        tolerance = self.config.position_tolerance
        if (self.config.dynamic_intercept_tracking
                and self.config.pregrasp_endpoint_settle_tolerance is not None):
            tolerance = max(tolerance, self.config.trajectory_replan_position_tolerance)
        current = self._ee_pose(observation)
        return bool(
            np.linalg.norm(target_pose[:3] - current[:3]) <= tolerance
            and quat_distance(target_pose[3:], current[3:]) <= self.config.orientation_tolerance
        )

    def _closure_pose_reached(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        target: Any | None = None,
    ) -> bool:
        """Use the physical jaw capture envelope at the moving intercept."""

        current = self._ee_pose(observation)
        position_error = target_pose[:3] - current[:3]
        horizontal_tolerance = _value(
            target,
            "closure_horizontal_tolerance",
            self.config.closure_horizontal_tolerance,
        )
        vertical_tolerance = _value(
            target,
            "closure_vertical_tolerance",
            self.config.closure_vertical_tolerance,
        )
        orientation_tolerance = _value(
            target,
            "closure_orientation_tolerance",
            self.config.closure_orientation_tolerance,
        )
        try:
            horizontal_tolerance = max(0.0, float(horizontal_tolerance))
            vertical_tolerance = max(0.0, float(vertical_tolerance))
            orientation_tolerance = max(0.0, float(orientation_tolerance))
        except (TypeError, ValueError):
            horizontal_tolerance = self.config.closure_horizontal_tolerance
            vertical_tolerance = self.config.closure_vertical_tolerance
            orientation_tolerance = self.config.closure_orientation_tolerance
        return (
            np.linalg.norm(position_error[:2])
            <= horizontal_tolerance
            and abs(float(position_error[2]))
            <= vertical_tolerance
            and quat_distance(target_pose[3:], current[3:])
            <= orientation_tolerance
        )

    def _global_path_state_valid(
        self,
        positions: np.ndarray,
        *,
        minimum_tcp_height: float,
        minimum_joint_margin: float | None = None,
        minimum_downward_alignment: float | None = None,
    ) -> bool:
        required_margin = (
            self.config.minimum_joint_limit_margin
            if minimum_joint_margin is None
            else max(0.0, float(minimum_joint_margin))
        )
        margin = self._joint_limit_margin(positions)
        if (
            margin is not None
            and margin < required_margin - 1e-9
        ):
            return False
        condition = self._jacobian_condition(positions)
        if (
            condition is not None
            and condition > self.config.singularity_hard_stop_condition
        ):
            return False
        validator = getattr(self.env, "is_arm_configuration_safe", None)
        alignment_limit = (
            self._planner_downward_alignment_limit()
            if minimum_downward_alignment is None
            else float(minimum_downward_alignment)
        )
        if callable(validator):
            return bool(
                validator(
                    positions,
                    approach_axis_local=self._approach_axis_local,
                    minimum_downward_alignment=alignment_limit,
                    minimum_tcp_height=minimum_tcp_height,
                    minimum_joint_margin=required_margin,
                )
            )
        solver = _value(self.env, ("ik", "ik_solver"))
        pose = _hook(solver, ("forward_pose",), ((positions,),))
        if pose is None:
            return True
        pose = np.asarray(pose, dtype=float)
        return bool(
            pose.shape == (7,)
            and pose[2] >= minimum_tcp_height - 1e-9
            and self._downward_alignment(pose[3:])
            >= alignment_limit - 1e-9
        )

    def _global_goal_candidates(
        self,
        target_pose: np.ndarray,
        current: np.ndarray,
        *,
        minimum_tcp_height: float,
        minimum_joint_margin: float | None = None,
    ) -> list[np.ndarray]:
        required_margin = (
            self.config.minimum_joint_limit_margin
            if minimum_joint_margin is None
            else max(0.0, float(minimum_joint_margin))
        )
        solver = _value(self.env, ("ik", "ik_solver"))
        if solver is None:
            return []
        seeds = [current.copy()]
        robot_config = getattr(self.env, "robot_config", {})
        if isinstance(robot_config, Mapping):
            for key in ("safe_joint_positions", "home_joint_positions"):
                value = robot_config.get(key)
                if value is not None:
                    seeds.append(np.asarray(value, dtype=float))
            calibrations = robot_config.get("calibration_poses", {})
            if isinstance(calibrations, Mapping):
                for calibration in calibrations.values():
                    if isinstance(calibration, Mapping) and (
                        "joint_positions" in calibration
                    ):
                        seeds.append(
                            np.asarray(
                                calibration["joint_positions"], dtype=float
                            )
                        )

        limits = self._joint_safety_limits()
        if limits is not None:
            lower, upper = limits
            margin = required_margin
            rng_seed = (
                7919
                + 101 * self._transfer_waypoint_index
                + 1009 * self._motion_plan_failure_count
            )
            try:
                rng_seed += 37 * int(self._target_id or 0)
            except (TypeError, ValueError):
                pass
            rng = np.random.default_rng(rng_seed)
            for _ in range(10):
                seeds.append(rng.uniform(lower + margin, upper - margin))

        candidates: list[np.ndarray] = []
        for seed in seeds:
            seed = np.asarray(seed, dtype=float).reshape(-1)
            if seed.shape != (7,) or not np.all(np.isfinite(seed)):
                continue
            solve = getattr(solver, "solve", None)
            if not callable(solve):
                continue
            try:
                result = solve(
                    target_pose[:3],
                    target_pose[3:],
                    qpos_seed=seed,
                )
            except (TypeError, ValueError):
                continue
            if result is None or _value(result, "success", False) is not True:
                continue
            candidate = np.asarray(
                _value(result, ("joint_positions", "qpos"), result),
                dtype=float,
            ).reshape(-1)
            if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                continue
            if not self._global_path_state_valid(
                candidate,
                minimum_tcp_height=minimum_tcp_height,
                minimum_joint_margin=required_margin,
            ):
                continue
            if any(np.max(np.abs(candidate - item)) < 1e-4 for item in candidates):
                continue
            candidates.append(candidate.copy())

        limits = self._joint_safety_limits()
        if limits is None:
            scale = np.ones(7, dtype=float)
        else:
            scale = np.maximum(limits[1] - limits[0], 1e-6)
        candidates.sort(
            key=lambda item: (
                float(np.linalg.norm((item - current) / scale)),
                -float(self._joint_limit_margin(item) or 0.0),
            )
        )
        return candidates

    def _global_goal_seeds(
        self,
        current: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
        *,
        target_id: Any = _MISSING,
    ) -> tuple[np.ndarray, ...]:
        """Build the deterministic IK seed order without solving on this tick."""

        seeds = [np.asarray(current, dtype=float).copy()]
        robot_config = getattr(self.env, "robot_config", {})
        if isinstance(robot_config, Mapping):
            for key in ("safe_joint_positions", "home_joint_positions"):
                value = robot_config.get(key)
                if value is not None:
                    seeds.append(np.asarray(value, dtype=float).reshape(-1))
            calibrations = robot_config.get("calibration_poses", {})
            if isinstance(calibrations, Mapping):
                for calibration in calibrations.values():
                    if isinstance(calibration, Mapping) and (
                        "joint_positions" in calibration
                    ):
                        seeds.append(
                            np.asarray(
                                calibration["joint_positions"], dtype=float
                            ).reshape(-1)
                        )
        rng_seed = (
            7919
            + 101 * self._transfer_waypoint_index
            + 1009 * self._motion_plan_failure_count
        )
        try:
            seed_target_id = (
                self._target_id if target_id is _MISSING else target_id
            )
            rng_seed += 37 * int(seed_target_id or 0)
        except (TypeError, ValueError):
            pass
        rng = np.random.default_rng(rng_seed)
        random_lower = lower + margin
        random_upper = upper - margin
        if np.all(random_lower < random_upper):
            seeds.extend(
                rng.uniform(random_lower, random_upper) for _ in range(10)
            )
        return tuple(
            seed.copy()
            for seed in seeds
            if seed.shape == (7,) and np.all(np.isfinite(seed))
        )

    def _tracking_branch_goal_seeds(
        self,
        current: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
        *,
        target_id: Any,
    ) -> tuple[np.ndarray, ...]:
        """Prefer the calibrated belt/corridor redundancy branch.

        A future PREGRASP request starts at the bin-side retreat anchor.  Using
        that anchor as the first DLS seed can produce a perfectly safe endpoint
        on a redundancy branch with no local descent continuation.  Likewise,
        LIFTING targets the calibrated wait pose: reaching the same Cartesian
        pose on an arbitrary branch made the following identical bin transfer
        take 5--11 s depending on the previous workpiece.  Try the immutable
        top-down calibration seeds first while retaining the complete fallback
        set.  Only candidate order changes; every IK endpoint and complete
        path still passes the same margin, singularity, cone and collision
        checks.
        """

        preferred: list[np.ndarray] = []
        robot_config = getattr(self.env, "robot_config", {})
        calibrations = (
            robot_config.get("calibration_poses", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if isinstance(calibrations, Mapping):
            for name in (
                "waiting_top_down",
                "belt_intercept_pregrasp",
                "belt_intercept_grasp",
            ):
                calibration = calibrations.get(name)
                value = (
                    calibration.get("joint_positions")
                    if isinstance(calibration, Mapping)
                    else None
                )
                if value is None:
                    continue
                seed = np.asarray(value, dtype=float).reshape(-1)
                if (
                    seed.shape == (7,)
                    and np.all(np.isfinite(seed))
                    and np.all(seed >= lower + margin - 1e-9)
                    and np.all(seed <= upper - margin + 1e-9)
                ):
                    preferred.append(seed.copy())

        ordered: list[np.ndarray] = []
        for seed in (
            *preferred,
            *self._global_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=target_id,
            ),
        ):
            if not any(
                np.max(np.abs(seed - existing)) < 1e-9
                for existing in ordered
            ):
                ordered.append(np.asarray(seed, dtype=float).copy())
        return tuple(ordered)

    def _pregrasp_goal_seeds(
        self,
        current: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
        *,
        target_id: Any,
        scene_epoch: int,
    ) -> tuple[np.ndarray, ...]:
        """Keep a certified redundancy branch across moving-endpoint replans."""

        ordered: list[np.ndarray] = []
        token = self._pregrasp_branch_q
        if (
            token is not None
            and self._same_target_id(
                self._pregrasp_branch_target_id, target_id
            )
            and self._pregrasp_branch_scene_epoch == int(scene_epoch)
        ):
            seed = np.asarray(token, dtype=float).reshape(-1)
            if (
                seed.shape == (7,)
                and np.all(np.isfinite(seed))
                and np.all(seed >= lower + margin - 1e-9)
                and np.all(seed <= upper - margin + 1e-9)
            ):
                ordered.append(seed.copy())

        for seed in self._tracking_branch_goal_seeds(
            current,
            lower,
            upper,
            margin,
            target_id=target_id,
        ):
            if not any(
                np.max(np.abs(seed - existing)) < 1e-9
                for existing in ordered
            ):
                ordered.append(np.asarray(seed, dtype=float).copy())
        return tuple(ordered)

    def _corridor_goal_seeds(
        self,
        current: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
        *,
        target_id: Any,
    ) -> tuple[np.ndarray, ...]:
        """Prefer the canonical wait redundancy branch while lifting.

        LIFTING and TRANSFERRING meet at the same calibrated wait Cartesian
        pose.  Solving that pose from the just-grasped joint state can land on
        a history-dependent 7-DOF branch whose following wait-to-bin route is
        either very long or infeasible.  Seed the single canonical
        ``waiting_top_down`` branch first, then retain the complete current,
        safe, home, calibration and deterministic-random fallback set.  This
        changes only candidate order; every endpoint and edge still receives
        the ordinary collision/cone/singularity/soft-limit certificate.
        """

        preferred: list[np.ndarray] = []
        robot_config = getattr(self.env, "robot_config", {})
        calibrations = (
            robot_config.get("calibration_poses", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if isinstance(calibrations, Mapping):
            for name in (
                "transfer_corridor_top_down",
                "waiting_top_down",
            ):
                calibration = calibrations.get(name)
                value = (
                    calibration.get("joint_positions")
                    if isinstance(calibration, Mapping)
                    else None
                )
                if value is None:
                    continue
                seed = np.asarray(value, dtype=float).reshape(-1)
                if (
                    seed.shape == (7,)
                    and np.all(np.isfinite(seed))
                    and np.all(seed >= lower + margin - 1e-9)
                    and np.all(seed <= upper - margin + 1e-9)
                ):
                    preferred.append(seed.copy())

        ordered: list[np.ndarray] = []
        for seed in (
            *preferred,
            *self._global_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=target_id,
            ),
        ):
            if not any(
                np.max(np.abs(seed - existing)) < 1e-9
                for existing in ordered
            ):
                ordered.append(np.asarray(seed, dtype=float).copy())
        return tuple(ordered)

    def _return_goal_seeds(
        self,
        current: np.ndarray,
        target_pose: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        margin: float,
        *,
        target_id: Any,
    ) -> tuple[np.ndarray, ...]:
        """Preserve an installed future-retreat redundancy branch.

        Prefetch leg B is certified from the immutable joint endpoint of leg
        A.  If live feedback invalidates one of A's short execution chords,
        the ordinary RETURNING recovery planner must still solve the same
        Cartesian retreat pose on that endpoint's redundancy branch.  Seeding
        the recovery from the measured bin-side state first allowed a 55 mm
        retreat to wrap joint 1 by more than 2.4 rad; the resulting endpoint
        could never match B's start state.  The cached anchor is only a seed:
        IK, the complete path, and every live command retain all normal
        collision, cone, singularity and soft-limit validation.
        """

        ordered: list[np.ndarray] = []
        anchor = self._prefetch_anchor_q
        retreat_pose = self._prefetch_retreat_pose
        use_anchor = bool(
            self._prefetch_retreat_installed
            and anchor is not None
            and retreat_pose is not None
            and np.linalg.norm(target_pose[:3] - retreat_pose[:3])
            <= self.config.trajectory_replan_position_tolerance + 1e-9
            and quat_distance(target_pose[3:], retreat_pose[3:])
            <= self.config.trajectory_replan_orientation_tolerance + 1e-9
        )
        if use_anchor:
            seed = np.asarray(anchor, dtype=float).reshape(-1)
            if (
                seed.shape == (7,)
                and np.all(np.isfinite(seed))
                and np.all(seed >= lower + margin - 1e-9)
                and np.all(seed <= upper - margin + 1e-9)
            ):
                ordered.append(seed.copy())

        for seed in self._global_goal_seeds(
            current,
            lower,
            upper,
            margin,
            target_id=target_id,
        ):
            if not any(
                np.max(np.abs(seed - existing)) < 1e-9
                for existing in ordered
            ):
                ordered.append(np.asarray(seed, dtype=float).copy())
        return tuple(ordered)

    def _async_target_poses(
        self,
        target_pose: np.ndarray,
        *,
        stage: PolicyStage | None = None,
        target: Any = _MISSING,
    ) -> tuple[np.ndarray, ...]:
        effective_stage = self._stage if stage is None else stage
        effective_target = self._target if target is _MISSING else target
        if (self._full_pick_contact_handed_off and effective_stage==PolicyStage.LIFTING
                and self._lift_clearance_global_replan_active and self._clearing_payload()):
            # A blocked vertical lift can use the existing downstream corridor
            # as a proposal. Keep height/orientation and certify the complete
            # payload route; no unconditional horizontal motion is sent here.
            cfg=_value(self.env,'scene_config',{}).get('workspace',{}).get('dynamic_intercept',{})
            offset=float(cfg.get('lift_downstream_offset_m',.08))
            direction=self._belt_frame()[1]
            velocity=np.asarray(self._full_pick_bundle.request.continuation_velocity_m_s)
            sign=-1. if float(velocity@direction)<0. else 1.
            bounds=self._belt_bounds('tracking_s_range','tracking_x_range',default=(-.15,.25))
            coordinate=self._belt_coordinate(target_pose[:3])
            destination=float(np.clip(coordinate+sign*offset,*bounds))
            if sign*(destination-coordinate)>1e-6:
                alternative=target_pose.copy();alternative[:3]+=(destination-coordinate)*direction
                # Saved loaded-request comparison: the near-limit vertical
                # goal can select a long redundant branch. Try the existing
                # downstream corridor first; both still need full certificates.
                return (alternative,target_pose.copy())
        if (
            self.config.dynamic_intercept_tracking
            and effective_stage == PolicyStage.PREGRASP
            and target_pose[1] > 0.06
            and str(_value(effective_target, "grasp_yaw_mode", "fixed")) in {"short_axis", "object_short_axis"}
            and _value(effective_target, ("grasp_quaternion", "grasp_orientation")) is None
        ):
            # The old central-lane yaw calibration can put an open jaw into
            # the far rail. Offer the observed short-axis grasp and its
            # equivalent pi-rotated wrist before that legacy option. No
            # object orientation is changed and no long-axis grasp is invented.
            shape_target = {
                "grasp_yaw_mode": "object_short_axis",
                "observed_pose": _value(effective_target, ("observed_pose", "historical_pose", "pose", "quaternion")),
            }
            calibration = self._adaptive_grasp_yaw_calibration(shape_target)
            if calibration is not None:
                delta = calibration[0]
                positive = delta % math.pi
                return (
                    np.r_[target_pose[:3], self._grasp_quaternion_for_yaw_delta(positive)],
                    np.r_[target_pose[:3], self._grasp_quaternion_for_yaw_delta(positive - math.pi)],
                    target_pose.copy(),
                )
        if (
            self.config.dynamic_intercept_tracking
            and effective_stage == PolicyStage.PREGRASP
            and str(_value(_value(effective_target, "symmetry", {}), "kind", "none"))
            == "continuous_rotational"
            and _value(effective_target, ("grasp_quaternion", "grasp_orientation")) is None
        ):
            # Axisymmetric parts admit equivalent jaw yaw.  At the far rail
            # the open fingers need to run along the belt during descent.
            # These are candidates only: the worker certifies the entire
            # open-jaw descent, not just IK at the high waypoint.
            nominal = self._grasp_quaternion.copy()
            diagonal = self._grasp_quaternion_for_yaw_delta(1.2)
            along_belt = self._grasp_quaternion_for_yaw_delta(math.pi / 2)
            opposite = self._grasp_quaternion_for_yaw_delta(-math.pi / 2)
            preferred = [diagonal, along_belt, nominal, opposite] if target_pose[1] > 0.06 else [nominal]
            if self._selected_grasp_quaternion is not None:
                preferred.insert(0, self._selected_grasp_quaternion)
            poses = []
            for quaternion in preferred:
                if not any(quat_distance(quaternion, p[3:]) < 1e-9 for p in poses):
                    poses.append(np.r_[target_pose[:3], quaternion])
            return tuple(poses)
        continuous_rotational = bool(
            effective_stage == PolicyStage.PREGRASP
            and str(
                _value(
                    _value(effective_target, ("symmetry",), {}),
                    ("kind",),
                    "none",
                )
            ).lower()
            == "continuous_rotational"
            and self._prefetch_retreat_pose is not None
        )
        adaptive = bool(
            effective_stage == PolicyStage.PREGRASP
            and (
                self._adaptive_grasp_yaw_calibration(effective_target) is not None
                or continuous_rotational
            )
            and _value(
                effective_target,
                ("grasp_quaternion", "grasp_orientation"),
                None,
            )
            is None
        )
        if not adaptive:
            return (target_pose.copy(),)
        quaternions = self._prefetch_grasp_quaternion_candidates(effective_target)
        if continuous_rotational:
            return tuple(
                np.r_[target_pose[:3], quaternion]
                for quaternion in quaternions
            )
        current_quaternion = (
            target_pose[3:]
            if self._selected_grasp_quaternion is None
            else self._selected_grasp_quaternion
        )
        similarities = [
            abs(float(np.dot(current_quaternion, quaternion)))
            for quaternion in quaternions
        ]
        start_index = int(np.argmax(similarities))
        poses: list[np.ndarray] = []
        for quaternion in quaternions[start_index:]:
            pose = target_pose.copy()
            pose[3:] = quaternion
            poses.append(pose)
        return tuple(poses) if poses else (target_pose.copy(),)

    def _pregrasp_high_pose(
        self, capture_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Predict the high waypoint at descent start, before capture."""
        from .dynamic_intercept import pregrasp_high_pose
        return pregrasp_high_pose(capture_pose, self._belt_velocity(observation),
            approach_height=self.config.approach_height,
            descent_lead_time=self.config.descent_lead_time,
            capture_high=(not self.config.dynamic_intercept_tracking
                          or self._pregrasp_capture_high_active))

    def _pregrasp_continuation_poses(
        self,
        target_poses: tuple[np.ndarray, ...],
        observation: Mapping[str, Any],
    ) -> tuple[tuple[np.ndarray, ...], float]:
        """Build worker-only capture probes for each high PREGRASP pose.

        The final local controller descends while matching the belt.  Include
        its normal short velocity feed-forward in the probe so a returned
        redundancy branch is certified beyond merely reaching the high
        waypoint.  These poses never become commands or use simulator truth.
        """

        # The global endpoint belongs to descent START, whereas the low
        # continuation belongs to capture. They are different places on a
        # moving belt, even when tool orientation stays fixed.
        belt_advance = self._belt_velocity(observation) * (
            max(0.0, self.config.tracking_velocity_feedforward_time)
            + (self.config.descent_lead_time
               if self.config.dynamic_intercept_tracking
               and not self._pregrasp_capture_high_active else 0.0)
        )
        continuations: list[np.ndarray] = []
        for target_pose in target_poses:
            pose = np.asarray(target_pose, dtype=float).copy()
            pose[2] -= self.config.approach_height
            pose[:3] += belt_advance
            continuations.append(pose)
        minimum_height = min(float(pose[2]) for pose in continuations) - 0.015
        return tuple(continuations), minimum_height

    def _planning_snapshot(
        self, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        snapshot = _hook(
            self.env,
            ("get_global_planning_snapshot",),
            ((),),
        )
        if isinstance(snapshot, Mapping):
            arm = np.asarray(
                snapshot.get("arm_qpos", current), dtype=float
            ).reshape(-1)
            full = np.asarray(
                snapshot.get("full_qpos", arm), dtype=float
            ).reshape(-1)
            try:
                scene_epoch = int(snapshot.get("scene_epoch", 0))
            except (TypeError, ValueError):
                scene_epoch = 0
            if arm.shape == (7,) and full.size >= 7:
                return arm.copy(), full.copy(), scene_epoch
        data = getattr(self.env, "data", None)
        qpos = getattr(data, "qpos", None)
        if qpos is not None:
            full = np.asarray(qpos, dtype=float).reshape(-1)
            if full.size >= 7 and np.all(np.isfinite(full)):
                return current.copy(), full.copy(), int(
                    self._observed_scene_epoch or 0
                )
        return current.copy(), current.copy(), int(
            self._observed_scene_epoch or 0
        )

    def _make_planning_request(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        minimum_joint_margin: float | None = None,
        minimum_tcp_height: float | None = None,
        minimum_downward_alignment: float | None = None,
    ) -> PlanningRequest | None:
        limits = self._joint_safety_limits()
        if limits is None:
            self._last_error = "global planner needs finite joint limits"
            return None
        lower, upper = limits
        margin = (
            self.config.minimum_joint_limit_margin
            if minimum_joint_margin is None
            else max(0.0, float(minimum_joint_margin))
        )
        current_observed = self._joint_positions(observation)
        current, full_qpos, scene_epoch = self._planning_snapshot(
            current_observed
        )
        current_pose = self._ee_pose(observation)
        minimum_height = (
            min(float(current_pose[2]), float(target_pose[2])) - 0.012
            if minimum_tcp_height is None
            else float(minimum_tcp_height)
        )
        alignment_limit = (
            self._planner_downward_alignment_limit()
            if minimum_downward_alignment is None
            else float(minimum_downward_alignment)
        )
        planner_config = SafePathConfig(
            normalized_step_size=self.config.planner_step_size,
            normalized_edge_resolution=self.config.planner_edge_resolution,
            execution_joint_step=self.config.planner_execution_joint_step,
            max_iterations=self.config.planner_max_iterations,
            shortcut_attempts=self.config.planner_shortcut_attempts,
        )
        planner_seed = (
            1543
            + 17 * self._transfer_waypoint_index
            + 1009 * self._motion_plan_failure_count
        )
        try:
            planner_seed += 53 * int(self._target_id or 0)
        except (TypeError, ValueError):
            pass
        velocity, acceleration = self._trajectory_limits()
        target_poses = self._async_target_poses(target_pose)
        planned_target_time_s = None
        if self._stage == PolicyStage.PREGRASP and self.config.dynamic_intercept_tracking:
            prediction = self._prediction_from_mapping(observation, self._target_id)
            if prediction is not None:
                planned_target_time_s = prediction.get("planned_grasp_time")
            if planned_target_time_s is None and isinstance(self._target, Mapping):
                planned_target_time_s = self._target.get("planned_grasp_time")
            if self._pregrasp_capture_time_s is not None:
                planned_target_time_s = self._pregrasp_capture_time_s
        continuation_poses: tuple[np.ndarray, ...] | None = None
        continuation_minimum_height: float | None = None
        if self._stage == PolicyStage.PREGRASP:
            (
                continuation_poses,
                continuation_minimum_height,
            ) = self._pregrasp_continuation_poses(
                target_poses, observation
            )
        goal_seeds = (
            self._pregrasp_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=self._target_id,
                scene_epoch=scene_epoch,
            )
            if self._stage == PolicyStage.PREGRASP
            else self._corridor_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=self._target_id,
            )
            if self._stage == PolicyStage.LIFTING
            else self._return_goal_seeds(
                current,
                target_pose,
                lower,
                upper,
                margin,
                target_id=self._target_id,
            )
            if self._stage == PolicyStage.RETURNING
            else self._global_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=self._target_id,
            )
        )
        request = PlanningRequest(
            target_identity=getattr(self.env, 'target_identity', None),
            selection_policy=self.config.selection_policy,
            request_id=self._planning_next_request_id,
            generation=self._planning_generation,
            stage=self._stage.value,
            target_id=self._target_id,
            scene_epoch=scene_epoch,
            target_poses=target_poses,
            start_q=current,
            full_qpos=full_qpos,
            goal_seeds=goal_seeds,
            lower_limits=lower,
            upper_limits=upper,
            minimum_tcp_height=minimum_height,
            minimum_joint_margin=margin,
            minimum_downward_alignment=(
                alignment_limit
            ),
            approach_axis_local=self._approach_axis_local,
            singularity_hard_stop_condition=(
                self.config.singularity_hard_stop_condition
            ),
            planner_config=planner_config,
            planner_seed=planner_seed,
            trajectory_velocity_limits=velocity,
            trajectory_acceleration_limits=acceleration,
            joint_trajectory_backend=self._global_joint_trajectory_backend(self._stage),
            trajectory_jerk_limits=self.config.trajectory_jerk_limits,
            trajectory_minimum_duration=(
                self.config.trajectory_minimum_duration
            ),
            maximum_tcp_angular_speed=(
                self._contact_handoff_global_angular_speed(self._stage)
                or self.config.maximum_tcp_angular_speed
            ),
            maximum_tcp_linear_speed=self._contact_handoff_global_linear_speed(self._stage),
            tcp_speed_limits_enabled=(self._global_path_tcp_speed_limits_enabled(self._stage)
                or self._contact_handoff_global_angular_speed(self._stage) is not None),
            planned_target_time_s=(
                None if planned_target_time_s is None else float(planned_target_time_s)
            ),
            intercept_velocity_m_s=(self._belt_velocity(observation)
                if planned_target_time_s is not None else None),
            intercept_motion_start_time_s=(self._timestamp_without_advancing(observation)
                if planned_target_time_s is not None else None),
            intercept_request_time_s=(self._timestamp_without_advancing(observation)
                if planned_target_time_s is not None else None),
            intercept_descent_duration_s=(self._new_descent_duration()
                if self._timed_descent_enabled(observation) else self.config.descent_lead_time
                )+self._execution_dt(observation),
            intercept_account_for_planning_time=isinstance(observation.get("dispatch_clock"), Mapping),
            intercept_motion_time_scale=self._pregrasp_motion_time_scale,
            intercept_position_tolerance_m=self.config.trajectory_replan_position_tolerance,
            continuation_target_poses=continuation_poses,
            continuation_minimum_tcp_height=(
                continuation_minimum_height
            ),
            continuation_max_joint_step=self.config.max_ik_joint_step,
            # This is an offline Cartesian branch probe. Its joint chords
            # receive the complete dense geometry checks, independent of the
            # actuator dispatch period; 100 Hz must not imply 40 IK solves for
            # every ten-centimetre prospective descent.
            continuation_cartesian_step=self.config.max_translation_step,
            timed_continuation_duration_s=(self._new_descent_duration()
                if continuation_poses is not None and self._timed_descent_enabled(observation) else None),
            continuation_velocity_m_s=tuple(self._belt_velocity(observation)),
            continuation_tcp_speed_limits=((self.config.cartesian_linear_speed,
                self.config.cartesian_angular_speed)
                if (self.config.tcp_speed_limits_enabled
                    and self.config.timed_descent_tcp_speed_limits_enabled) else None),
            # A strict loaded recovery starts at latest measured q because the
            # old C topology proved untrackable with the live payload.  Do not
            # steer it back through that same fixed bin gateway.  Ordinary
            # speculative C still receives (and fully revalidates) its route
            # hint in ``_future_prefetch_request``.
            route_waypoint_hints=self._previous_pregrasp_route_hints(
                current, target_pose, scene_epoch=scene_epoch,
            ),
            route_hints_after_direct=(self._stage == PolicyStage.PREGRASP
                                      and self.config.dynamic_intercept_tracking),
        )
        self._planning_next_request_id += 1
        return request

    def _previous_pregrasp_route_hints(
        self, current: np.ndarray, target_pose: np.ndarray, *, scene_epoch: int,
    ) -> tuple[np.ndarray, ...]:
        """Warm-start a new moving-endpoint search with remaining old corners.

        A waypoint is only a geometric search suggestion. The planning worker
        still solves the new grasp/high endpoint and certifies all states,
        edges and its descent against the new immutable request. Projection
        onto the old polyline drops corners already passed by measured q.
        Neither the old start nor its old endpoint is imposed on a new route.
        """
        previous = self._pregrasp_previous_route
        if (not self.config.dynamic_intercept_tracking
                or self._stage != PolicyStage.PREGRASP or previous is None
                or previous["scene_epoch"] != int(scene_epoch)
                or not self._same_target_id(previous["target_id"], self._target_id)
                or quat_distance(previous["target_pose"][3:], target_pose[3:])
                > self.config.trajectory_replan_orientation_tolerance):
            return ()
        path = previous["path"]
        if len(path) <= 2:
            return ()
        q = np.asarray(current, dtype=float)
        nearest_edge, nearest_progress, nearest_distance = 0, 0.0, float("inf")
        for edge_index, (first, second) in enumerate(zip(path[:-1], path[1:])):
            delta = second-first
            norm_squared = float(delta @ delta)
            progress = (0.0 if norm_squared <= 1e-24 else
                        float(np.clip((q-first) @ delta/norm_squared, 0.0, 1.0)))
            distance = float(np.linalg.norm(q-(first+progress*delta)))
            # At a shared corner prefer the outgoing edge.
            if distance <= nearest_distance+1e-12:
                nearest_edge, nearest_progress, nearest_distance = edge_index, progress, distance
        first_unpassed = nearest_edge+1+(nearest_progress >= 1.0-1e-12)
        hints = tuple(state.copy() for state in path[first_unpassed:-1]
                      if np.max(np.abs(state-q)) > 1e-9)
        self._trajectory_reuse_diagnostics.update(
            route_hint_count=len(hints), source_scene_epoch=int(scene_epoch),
            measured_to_previous_route_rad=nearest_distance,
            previous_endpoint_used_as_waypoint=False,
        )
        return hints

    def _make_return_lookahead_request(
        self,
        candidate: Any,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> PlanningRequest | None:
        """Snapshot a strict PREGRASP search without reserving ``candidate``.

        This is intentionally not routed through the active stage fields:
        while the request runs the policy remains RETURNING and continues its
        already installed collision-checked path.  Only policy-visible RGB-D
        candidate data and the normal isolated planning snapshot are copied.
        """

        limits = self._joint_safety_limits()
        target_id = _object_id(candidate)
        if limits is None or target_id is None:
            return None
        lower, upper = limits
        margin = self._pregrasp_planning_joint_margin()
        current_observed = self._joint_positions(observation)
        current, full_qpos, scene_epoch = self._planning_snapshot(
            current_observed
        )
        current_pose = self._ee_pose(observation)
        minimum_height = min(
            float(current_pose[2]), float(target_pose[2])
        ) - 0.012
        planner_config = SafePathConfig(
            normalized_step_size=self.config.planner_step_size,
            normalized_edge_resolution=self.config.planner_edge_resolution,
            execution_joint_step=self.config.planner_execution_joint_step,
            max_iterations=min(
                self.config.planner_max_iterations,
                self.config.return_lookahead_planner_max_iterations,
            ),
            shortcut_attempts=min(
                self.config.planner_shortcut_attempts, 30
            ),
        )
        planner_seed = 4219 + 1009 * self._return_lookahead_failures
        try:
            planner_seed += 53 * int(target_id)
        except (TypeError, ValueError):
            pass
        velocity, acceleration = self._trajectory_limits()
        target_poses = self._async_target_poses(
            target_pose,
            stage=PolicyStage.PREGRASP,
            target=candidate,
        )
        (
            continuation_poses,
            continuation_minimum_height,
        ) = self._pregrasp_continuation_poses(
            target_poses, observation
        )
        request = PlanningRequest(
            target_identity=getattr(self.env, 'target_identity', None),
            selection_policy=self.config.selection_policy,
            request_id=self._planning_next_request_id,
            generation=self._planning_generation,
            stage=PolicyStage.PREGRASP.value,
            target_id=target_id,
            scene_epoch=scene_epoch,
            target_poses=target_poses,
            start_q=current,
            full_qpos=full_qpos,
            goal_seeds=self._global_goal_seeds(
                current,
                lower,
                upper,
                margin,
                target_id=target_id,
            ),
            lower_limits=lower,
            upper_limits=upper,
            minimum_tcp_height=minimum_height,
            minimum_joint_margin=margin,
            minimum_downward_alignment=(
                self.config.acquisition_minimum_downward_alignment
            ),
            approach_axis_local=self._approach_axis_local,
            singularity_hard_stop_condition=(
                self.config.singularity_hard_stop_condition
            ),
            planner_config=planner_config,
            planner_seed=planner_seed,
            trajectory_velocity_limits=velocity,
            trajectory_acceleration_limits=acceleration,
            joint_trajectory_backend=self._global_joint_trajectory_backend(PolicyStage.PREGRASP),
            trajectory_jerk_limits=self.config.trajectory_jerk_limits,
            trajectory_minimum_duration=(
                self.config.trajectory_minimum_duration
            ),
            maximum_tcp_angular_speed=(
                self.config.maximum_tcp_angular_speed
            ),
            tcp_speed_limits_enabled=self._global_path_tcp_speed_limits_enabled(PolicyStage.PREGRASP),
            wall_time_budget_s=(
                self.config.return_lookahead_planning_budget
            ),
            continuation_target_poses=continuation_poses,
            continuation_minimum_tcp_height=(
                continuation_minimum_height
            ),
            continuation_max_joint_step=self.config.max_ik_joint_step,
            continuation_cartesian_step=self.config.max_translation_step,
        )
        self._planning_next_request_id += 1
        return request

    def _future_open_planning_snapshot(
        self, arm_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Return future arm/full qpos with open-jaw collision geometry."""

        arm = np.asarray(arm_q, dtype=float).reshape(-1)
        if arm.shape != (7,) or not np.all(np.isfinite(arm)):
            return None
        snapshot = _hook(
            self.env,
            ("get_future_open_planning_snapshot",),
            ((arm,),),
        )
        if not isinstance(snapshot, Mapping):
            # Production environments exposing a prefetch factory must also
            # supply an open-jaw future snapshot.  A test double without that
            # contract may use arm-only state for protocol tests.
            if callable(
                getattr(self.env, "create_prefetch_planning_worker", None)
            ):
                return None
            return arm.copy(), arm.copy(), int(self._observed_scene_epoch or 0)
        future_arm = np.asarray(snapshot.get("arm_qpos", arm), dtype=float).reshape(-1)
        full = np.asarray(snapshot.get("full_qpos", future_arm), dtype=float).reshape(-1)
        try:
            scene_epoch = int(snapshot.get("scene_epoch", 0))
        except (TypeError, ValueError):
            return None
        if (
            future_arm.shape != (7,)
            or full.size < 7
            or not np.all(np.isfinite(future_arm))
            or not np.all(np.isfinite(full))
        ):
            return None
        return future_arm.copy(), full.copy(), scene_epoch

    def _carry_corridor_calibration(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the configured immutable carry anchor joint/TCP pair."""

        robot_config = getattr(self.env, "robot_config", {})
        calibrations = (
            robot_config.get("calibration_poses", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        calibration = (
            calibrations.get("transfer_corridor_top_down")
            if isinstance(calibrations, Mapping)
            else None
        )
        if not isinstance(calibration, Mapping):
            return None
        try:
            q = np.asarray(
                calibration["joint_positions"], dtype=float
            ).reshape(-1)
            position = np.asarray(
                calibration["expected_ee_position"], dtype=float
            ).reshape(-1)
            quaternion = normalized_quat(
                calibration["expected_ee_quaternion_wxyz"]
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            q.shape != (7,)
            or position.shape != (3,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(position))
        ):
            return None
        return q.copy(), np.r_[position, quaternion]

    def _carry_route_waypoint_hints(
        self, target: Any
    ) -> tuple[np.ndarray, ...]:
        """Return untrusted, bin-scoped joint waypoints for carry search.

        These values only select a useful topology for a repeated fixed-bin
        query.  They never become a path certificate here: the isolated
        worker must revalidate every hinted state and dense edge against the
        request's copied full-qpos/gripper geometry and all current safety
        constraints before it can publish a route.
        """

        robot_config = getattr(self.env, "robot_config", {})
        configured = (
            robot_config.get("transfer_route_joint_hints", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if not isinstance(configured, Mapping):
            return ()
        bin_index = self._bin_index(target)
        raw = configured.get(bin_index, configured.get(str(bin_index)))
        if raw is None:
            return ()
        try:
            waypoints = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return ()
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1:] != (7,)
            or not np.all(np.isfinite(waypoints))
        ):
            return ()
        return tuple(row.copy() for row in waypoints)

    def _pregrasp_route_waypoint_hints(
        self, target: Any
    ) -> tuple[np.ndarray, ...]:
        """Return untrusted B topology hints for one exact A/category pair.

        The seven-DoF route from a bin-side retreat anchor to the moving belt
        can have narrow, repeatable topology.  A configured waypoint is only a
        search hint: the isolated worker still solves the current visual pose
        and revalidates every state, dense edge, compressed chord, continuation
        and trajectory profile against this request's immutable snapshot.  A
        missing provenance/category or malformed exact key fails closed to the
        ordinary direct/RRT search; hints are never borrowed from a nearby bin
        or object class.
        """

        source_bin = self._prefetch_source_bin_index
        raw_category = _value(target, ("category_id", "class_id"), None)
        try:
            source_key = int(source_bin)  # type: ignore[arg-type]
            category_key = int(raw_category)
        except (TypeError, ValueError):
            return ()
        robot_config = getattr(self.env, "robot_config", {})
        configured = (
            robot_config.get("pregrasp_route_joint_hints", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if not isinstance(configured, Mapping):
            return ()
        by_category = configured.get(
            source_key, configured.get(str(source_key))
        )
        if not isinstance(by_category, Mapping):
            return ()
        raw = by_category.get(
            category_key, by_category.get(str(category_key))
        )
        if raw is None:
            return ()
        try:
            waypoints = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return ()
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1:] != (7,)
            or not np.all(np.isfinite(waypoints))
        ):
            return ()
        return tuple(row.copy() for row in waypoints)

    def _prefetch_source_has_pregrasp_hints(self) -> bool:
        """Whether A's exact source bin has configured fast B topologies."""

        source_bin = self._prefetch_source_bin_index
        if source_bin is None:
            source_bin = self._carry_prefetch_bin_index
        if source_bin is None and self._target is not None:
            raw_bin = _value(self._target, ("bin_id", "target_bin"), None)
            raw_category = _value(
                self._target, ("category_id", "class_id"), None
            )
            try:
                source_bin = (
                    int(raw_bin)
                    if raw_bin is not None
                    else int(raw_category) // self.config.category_bin_size
                )
            except (TypeError, ValueError):
                return False
        try:
            source_key = int(source_bin)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        robot_config = getattr(self.env, "robot_config", {})
        configured = (
            robot_config.get("pregrasp_route_joint_hints", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        if not isinstance(configured, Mapping):
            return False
        by_category = configured.get(
            source_key, configured.get(str(source_key))
        )
        return bool(isinstance(by_category, Mapping) and by_category)

    def _route_hint_target_poses(
        self,
        target_poses: tuple[np.ndarray, ...],
        target: Any,
        route_waypoint_hints: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        """Prioritize the canonical member of a hinted symmetric query.

        A continuous-rotational part has no physical tool-yaw preference.  In
        the ordinary search the canonical belt quaternion is preferred to
        avoid unwinding while carrying the next part.  A route hint also
        describes one exact joint-space topology.  Keep that hint useful when
        receiving externally ordered candidates by prioritizing the same
        canonical quaternion; the isolated
        worker still solves the current visual endpoint and certifies the
        continuation, every hinted state and every dense edge.  Other target
        types and unhinted requests retain their existing candidate order.
        """

        if not route_waypoint_hints or len(target_poses) <= 1:
            return target_poses
        symmetry = _value(target, ("symmetry",), {})
        if (
            str(_value(symmetry, ("kind",), "none")).lower()
            != "continuous_rotational"
        ):
            return target_poses
        similarities = [
            abs(float(np.dot(pose[3:], self._grasp_quaternion)))
            for pose in target_poses
        ]
        canonical_index = int(np.argmax(similarities))
        if canonical_index == 0:
            return target_poses
        return (
            target_poses[canonical_index],
            *(
                pose
                for index, pose in enumerate(target_poses)
                if index != canonical_index
            ),
        )

    def _future_carry_planning_snapshot(
        self,
        arm_q: np.ndarray,
        gripper_action: float,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        snapshot = _hook(
            self.env,
            ("get_future_carry_planning_snapshot",),
            ((arm_q, gripper_action),),
        )
        if not isinstance(snapshot, Mapping):
            if callable(
                getattr(self.env, "create_prefetch_planning_worker", None)
            ):
                return None
            arm = np.asarray(arm_q, dtype=float).reshape(-1)
            return arm.copy(), arm.copy(), int(self._observed_scene_epoch or 0)
        arm = np.asarray(snapshot.get("arm_qpos", arm_q), dtype=float).reshape(-1)
        full = np.asarray(snapshot.get("full_qpos", arm), dtype=float).reshape(-1)
        try:
            scene_epoch = int(snapshot.get("scene_epoch", 0))
        except (TypeError, ValueError):
            return None
        if (
            arm.shape != (7,)
            or full.size < 7
            or not np.all(np.isfinite(arm))
            or not np.all(np.isfinite(full))
        ):
            return None
        return arm.copy(), full.copy(), scene_epoch

    def _carry_trajectory_limits(
        self, target: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity, acceleration = self._trajectory_limits()
        if self._stage in {
            PolicyStage.LIFTING,
            PolicyStage.TRANSFERRING,
            PolicyStage.RELEASING,
        }:
            return velocity, acceleration
        category = _value(target, ("category_id", "class_id"), -1)
        try:
            gentle = int(category) in {
                int(item) for item in self.config.gentle_carry_categories
            }
        except (TypeError, ValueError):
            gentle = False
        if gentle:
            velocity = velocity * self.config.carrying_velocity_scale
            acceleration = (
                acceleration * self.config.carrying_acceleration_scale
            )
        return velocity, acceleration

    def _late_prefetch_pregrasp_target(
        self,
        target: Any,
        target_pose: np.ndarray,
        observation: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, float] | None:
        """Project B to the latest configured deadline-safe intercept.

        The nominal RGB-D pregrasp is normally centred on the preferred work
        station.  A speculative route may take several seconds, so certifying
        that early endpoint and only then applying route-aware retiming makes
        the delivered path stale by tens of millimetres.  B instead plans to
        ``window_end - intercept_deadline_margin`` up front.  The arm may
        arrive early and wait in the high pregrasp corridor; the normal
        descent clock remains unchanged.  This uses only policy-visible track
        velocity/time and never simulator object state.
        """

        window = _value(target, "grasp_window", {})
        window_end_raw = _value(
            target,
            ("grasp_window_end", "window_end", "deadline"),
            _value(window, ("end_time", "end", "deadline"), None),
        )
        nominal_time_raw = _value(
            target, ("estimated_grasp_time", "grasp_time"), None
        )
        try:
            window_end = float(window_end_raw)
            nominal_time = float(nominal_time_raw)
        except (TypeError, ValueError):
            return None
        planned_time = (
            window_end
            - self.config.intercept_deadline_margin
            - self.config.prefetch_intercept_clock_guard
        )
        if (
            not np.isfinite(planned_time)
            or not np.isfinite(nominal_time)
        ):
            return None
        if observation is not None and planned_time <= (
            self._timestamp_without_advancing(observation) + 1e-9
        ):
            return None

        prediction = (
            None
            if observation is None
            else self._prediction_from_mapping(
                observation, _object_id(target)
            )
        )
        raw_velocity = _value(
            target,
            ("estimated_velocity", "linear_velocity", "velocity"),
            _value(
                prediction,
                ("estimated_velocity", "linear_velocity", "velocity"),
                self._belt_velocity({} if observation is None else observation),
            ),
        )
        try:
            velocity = np.asarray(raw_velocity, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if velocity.size == 1:
            planar_velocity = self._belt_frame()[1][:2] * velocity[0]
        elif velocity.size >= 2:
            planar_velocity = velocity[:2]
        else:
            return None
        if not np.all(np.isfinite(planar_velocity)):
            return None

        projected = np.asarray(target_pose, dtype=float).copy()
        # A refreshed deadline can move just before the predictor's preferred
        # intercept.  Linear back-projection over this short visual horizon is
        # the same constant-velocity model used downstream; refusing negative
        # dt here silently downgraded retries to an unclocked legacy request.
        projected[:3] = self._retimed_grasp_position(
            projected[:3], np.r_[projected[:2] + planar_velocity * (planned_time - nominal_time), projected[2]],
        )
        if projected.shape != (7,) or not np.all(np.isfinite(projected)):
            return None
        return projected, float(planned_time)

    def _future_prefetch_request(
        self,
        *,
        target_id: Any,
        target_pose: np.ndarray,
        start_q: np.ndarray,
        full_qpos: np.ndarray,
        scene_epoch: int,
        target: Any | None,
        phase: str,
        observation: Mapping[str, Any] | None = None,
        execution_reserve_s: float | None = None,
    ) -> PlanningRequest | None:
        """Build one full-budget request for the independent A/B pipeline."""

        limits = self._joint_safety_limits()
        if limits is None:
            return None
        lower, upper = limits
        margin = (
            self._commandable_joint_margin()
            if phase in {"carry", "retreat"}
            else self._pregrasp_planning_joint_margin()
        )
        start = np.asarray(start_q, dtype=float).reshape(-1)
        pose = np.asarray(target_pose, dtype=float).reshape(-1)
        if start.shape != (7,) or pose.shape != (7,):
            return None
        planned_target_time_s: float | None = None
        if (
            phase == "pregrasp"
            and target is not None
            and observation is not None
        ):
            late_target = self._late_prefetch_pregrasp_target(
                target, pose, observation
            )
            if late_target is not None:
                pose, planned_target_time_s = late_target
        if phase == "pregrasp":
            # Prefer the additional tracking reserve, but never admit a B
            # start below the actuator's actual command envelope.  Planning
            # against only the generic 25 mrad path margin can certify an
            # endpoint that the controller (55 mrad command guard in the real
            # robot config) will deterministically clip and therefore never
            # settle at.
            available_margin = float(
                np.min(np.minimum(start - lower, upper - start))
            )
            if available_margin < margin - 1e-9:
                margin = self._commandable_joint_margin()
        # Both future starts are endpoints of already certified Cartesian
        # requests.  Reuse their immutable pose metadata rather than running a
        # MuJoCo forward pass in the 50 Hz callback merely to recover z.
        carry_calibration = self._carry_corridor_calibration()
        known_start_pose = (
            carry_calibration[1]
            if phase == "carry" and carry_calibration is not None
            else self._place_pose
            if phase == "retreat"
            else self._prefetch_retreat_pose
        )
        start_height = (
            float(known_start_pose[2])
            if known_start_pose is not None
            else float(pose[2])
        )
        minimum_height = min(start_height, float(pose[2])) - 0.012
        planner_config = SafePathConfig(
            normalized_step_size=self.config.planner_step_size,
            normalized_edge_resolution=self.config.planner_edge_resolution,
            execution_joint_step=self.config.planner_execution_joint_step,
            max_iterations=self.config.planner_max_iterations,
            shortcut_attempts=self.config.planner_shortcut_attempts,
        )
        seed_offset = (
            900_019
            if phase == "carry"
            else 0
            if phase == "retreat"
            else 500_009
        )
        seed = 12_289 + seed_offset + 1009 * self._prefetch_failures
        seed_identity = (
            self._bin_index(target)
            if phase == "carry" and target is not None
            else target_id
        )
        try:
            seed += 53 * int(seed_identity or 0)
        except (TypeError, ValueError):
            pass
        velocity, acceleration = (
            self._carry_trajectory_limits(target)
            if phase == "carry"
            else self._nominal_trajectory_limits()
            if phase == "pregrasp"
            else self._trajectory_limits()
        )
        route_waypoint_hints = (
            self._carry_route_waypoint_hints(target)
            if phase == "carry" and target is not None
            else self._pregrasp_route_waypoint_hints(target)
            if phase == "pregrasp" and target is not None
            else ()
        )
        if phase == "retreat":
            target_poses_list = [pose.copy()]
            full_distance = float(self.config.release_retreat_distance)
            handoff_distance = float(
                self.config.prefetch_handoff_retreat_distance
            )
            if handoff_distance < full_distance - 1e-12:
                handoff = pose.copy()
                handoff[1] -= full_distance - handoff_distance
                if self._prefetch_source_has_pregrasp_hints():
                    # The exact source-bin pipeline has a freshly validated B
                    # gateway and is deadline-bound by A's measured arrival.
                    # Prefer its configured 20 mm open-jaw handoff; execution
                    # is still withheld until the just-released object is
                    # visibly terminal and both fingers are physically clear.
                    target_poses_list = [handoff, pose.copy()]
                else:
                    target_poses_list.append(handoff)
            target_poses = tuple(target_poses_list)
        elif phase == "carry":
            target_poses = (pose.copy(),)
        else:
            target_poses = self._async_target_poses(
                pose, stage=PolicyStage.PREGRASP, target=target
            )
            target_poses = self._route_hint_target_poses(
                target_poses,
                target,
                route_waypoint_hints,
            )
        continuation_poses: tuple[np.ndarray, ...] | None = None
        continuation_minimum_height: float | None = None
        wall_time_budget_s: float | None = None
        if phase == "pregrasp":
            (
                continuation_poses,
                continuation_minimum_height,
            ) = self._pregrasp_continuation_poses(
                target_poses,
                {} if observation is None else observation,
            )
            if planned_target_time_s is not None and observation is not None:
                execution_reserve = (
                    float(self.config.prefetch_planning_execution_reserve)
                    if execution_reserve_s is None
                    else float(execution_reserve_s)
                )
                if (
                    not np.isfinite(execution_reserve)
                    or execution_reserve <= 0.0
                ):
                    return None
                latency_sum = 0.0
                latency = _value(observation, "system_latency", {})
                if isinstance(latency, Mapping):
                    for name in ("planning", "control", "robot_response"):
                        try:
                            component = float(_value(latency, name, 0.0))
                        except (TypeError, ValueError):
                            component = 0.0
                        if np.isfinite(component) and component > 0.0:
                            latency_sum += component
                search_clock = (
                    planned_target_time_s
                    - self._timestamp_without_advancing(observation)
                    - self.config.descent_lead_time
                    - latency_sum
                    - self.config.control_timestep
                    - execution_reserve
                )
                wall_time_budget_s = (
                    search_clock
                    / self.config.prefetch_planning_wall_realtime_ratio
                )
                if (
                    not np.isfinite(wall_time_budget_s)
                    or wall_time_budget_s <= 0.0
                ):
                    return None
        request = PlanningRequest(
            target_identity=getattr(self.env, 'target_identity', None),
            selection_policy=self.config.selection_policy,
            request_id=self._prefetch_next_request_id,
            generation=self._prefetch_generation,
            stage=(
                PolicyStage.TRANSFERRING.value
                if phase == "carry"
                else PolicyStage.RETURNING.value
                if phase == "retreat"
                else PolicyStage.PREGRASP.value
            ),
            target_id=target_id,
            scene_epoch=int(scene_epoch),
            target_poses=target_poses,
            start_q=start,
            full_qpos=np.asarray(full_qpos, dtype=float),
            goal_seeds=(
                self._tracking_branch_goal_seeds(
                    start,
                    lower,
                    upper,
                    margin,
                    target_id=target_id,
                )
                if phase == "pregrasp"
                else self._global_goal_seeds(
                    start, lower, upper, margin, target_id=target_id
                )
            ),
            lower_limits=lower,
            upper_limits=upper,
            minimum_tcp_height=minimum_height,
            minimum_joint_margin=margin,
            minimum_downward_alignment=(
                self.config.acquisition_minimum_downward_alignment
            ),
            approach_axis_local=self._approach_axis_local,
            singularity_hard_stop_condition=(
                self.config.singularity_hard_stop_condition
            ),
            planner_config=planner_config,
            planner_seed=seed,
            trajectory_velocity_limits=velocity,
            trajectory_acceleration_limits=acceleration,
            joint_trajectory_backend=self._global_joint_trajectory_backend(
                PolicyStage.TRANSFERRING if phase == 'carry'
                else PolicyStage.RETURNING if phase == 'retreat' else PolicyStage.PREGRASP),
            trajectory_jerk_limits=self.config.trajectory_jerk_limits,
            trajectory_minimum_duration=self.config.trajectory_minimum_duration,
            maximum_tcp_angular_speed=self.config.maximum_tcp_angular_speed,
            tcp_speed_limits_enabled=self._global_path_tcp_speed_limits_enabled(
                PolicyStage.TRANSFERRING if phase == "carry"
                else PolicyStage.RETURNING if phase == "retreat"
                else PolicyStage.PREGRASP),
            # B is tied to an immutable physical intercept clock.  Its worker
            # watchdog reserves a conservative execution horizon and returns
            # any already certified direct incumbent when search time expires.
            # A/C are not clock-bound and remain unbudgeted.
            wall_time_budget_s=wall_time_budget_s,
            planned_target_time_s=planned_target_time_s,
            optimize_direct_path_eta=(phase == "pregrasp"),
            # B is independent speculative work.  If its best direct branch
            # is slow, give the two most promising IK goals one uninterrupted
            # full-budget tree each; short sliced restarts repeatedly rebuild
            # before this corridor's useful branch can connect.  Other phases
            # retain the latency-first defaults.
            full_rrt_candidates_after_direct=(
                2 if phase == "pregrasp" else 0
            ),
            full_rrt_minimum_direct_eta_s=(
                (
                    self.config.prefetch_planning_execution_reserve
                    if execution_reserve_s is None
                    else float(execution_reserve_s)
                )
                if phase == "pregrasp"
                else None
            ),
            route_waypoint_hints=route_waypoint_hints,
            continuation_target_poses=continuation_poses,
            continuation_minimum_tcp_height=(
                continuation_minimum_height
            ),
            continuation_max_joint_step=self.config.max_ik_joint_step,
            continuation_cartesian_step=(
                min(0.020, self.config.max_translation_step)
                if phase == "retreat"
                else self.config.max_translation_step
            ),
            cartesian_local_only=(phase == "retreat"),
            maximum_goal_joint_delta=(
                self.config.max_verified_ik_transition
                if phase == "retreat"
                else None
            ),
        )
        self._prefetch_next_request_id += 1
        return request

    @staticmethod
    def _same_target_id(left: Any, right: Any) -> bool:
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return left is right

    def _request_identity_is_current(self, request):
        identity = getattr(request, 'target_identity', None)
        current = getattr(self.env, 'target_identity', None)
        return identity is None if current is None else current.same_task(identity)

    def _request_has_current_owner(self, request: PlanningRequest) -> bool:
        return bool(
            self._request_identity_is_current(request)
            and
            request.generation == self._planning_generation
            and request.stage == self._stage.value
            and self._same_target_id(request.target_id, self._target_id)
            and request.scene_epoch == int(self._observed_scene_epoch or 0)
        )

    def _request_matches_goal(
        self, request: PlanningRequest, target_pose: np.ndarray
    ) -> bool:
        return bool(
            self._request_has_current_owner(request)
            and any(
                np.linalg.norm(pose[:3] - target_pose[:3])
                <= self.config.trajectory_replan_position_tolerance
                and quat_distance(pose[3:], target_pose[3:])
                <= self.config.trajectory_replan_orientation_tolerance
                for pose in (request.target_poses if self.config.dynamic_intercept_tracking else (request.target_pose,))
            )
        )

    def _pending_pregrasp_request_can_finish(
        self, request: PlanningRequest, target_pose: np.ndarray
    ) -> bool:
        """Keep one visual reservation immutable until its worker completes.

        A revised belt velocity changes the high waypoint even while the
        reserved low capture stays fixed. It must not destroy the only route
        search. Delivery still refines the live capture before executing a
        certified high prefix, and any new endpoint needs its own certificate.
        """
        return bool(
            self.config.dynamic_intercept_tracking
            and self._stage == PolicyStage.PREGRASP
            and self._request_has_current_owner(request)
            and request.planned_target_time_s is not None
            and request.intercept_velocity_m_s is not None
            and any(
                quat_distance(pose[3:], target_pose[3:])
                <= self.config.trajectory_replan_orientation_tolerance
                for pose in request.target_poses
            )
        )

    def _result_matches_request(
        self, result: PlanningResult, request: PlanningRequest
    ) -> bool:
        if not (
            result.target_identity == request.target_identity
            and self._request_identity_is_current(request)
            and result.request_id == request.request_id
            and result.generation == request.generation
            and result.stage == request.stage
            and self._same_target_id(result.target_id, request.target_id)
            and result.scene_epoch == request.scene_epoch
        ):
            return False
        return any(
            np.linalg.norm(result.target_pose[:3] - pose[:3]) <= 1e-9
            and quat_distance(result.target_pose[3:], pose[3:]) <= 1e-9
            for pose in request.target_poses
        )

    def _completed_planning_request(
        self, result: PlanningResult, request: PlanningRequest
    ) -> PlanningRequest | None:
        """Validate result ownership/forecast without changing reservations."""
        if not (
            result.target_identity == request.target_identity
            and self._request_identity_is_current(request)
            and result.request_id == request.request_id
            and result.generation == request.generation
            and result.stage == request.stage
            and self._same_target_id(result.target_id, request.target_id)
            and result.scene_epoch == request.scene_epoch
            and np.array_equal(result.start_q, request.start_q)
        ):
            return None
        effective = result.effective_request
        if effective is None:
            return request if self._result_matches_request(result, request) else None
        if not (
            request.stage == PolicyStage.PREGRASP.value
            and effective.target_identity == request.target_identity
            and effective.request_id == request.request_id
            and effective.generation == request.generation
            and effective.stage == request.stage
            and effective.scene_epoch == request.scene_epoch
            and self._same_target_id(effective.target_id, request.target_id)
            and np.array_equal(effective.start_q, request.start_q)
            and self._result_matches_request(result, effective)
        ):
            return None
        return effective

    @staticmethod
    def _lookahead_category(target: Any) -> Any:
        return _value(target, ("category_id", "class_id", "category", "label"))

    def _discard_return_lookahead(
        self,
        *,
        cancel_worker: bool,
        reset_candidate: bool = False,
    ) -> None:
        """Invalidate speculative work without touching the active return path."""

        request = getattr(self, "_return_lookahead_request", None)
        self._return_lookahead_request = None
        self._return_lookahead_result = None
        self._return_lookahead_category = None
        self._return_lookahead_pose = None
        self._return_lookahead_scene_epoch = None
        self._return_lookahead_observed_at = None
        if cancel_worker and request is not None:
            worker = getattr(self, "_planning_worker", None)
            if worker is not None:
                worker.cancel_pending()
        if reset_candidate:
            self._return_candidate_id = None
            self._return_candidate_stable_steps = 0
            self._return_lookahead_failures = 0

    def _return_pregrasp_pose(
        self,
        candidate: Any,
        observation: Mapping[str, Any],
    ) -> np.ndarray | None:
        grasp = self._grasp_target_pose(
            candidate,
            observation,
            remember_orientation=False,
        )
        if grasp is None:
            return None
        pregrasp = grasp.copy()
        pregrasp[2] += self.config.approach_height
        return pregrasp

    def _stable_return_candidate(
        self, observation: Mapping[str, Any]
    ) -> tuple[Any | None, np.ndarray | None]:
        """Track one deterministic visual candidate across control ticks."""

        candidate = self._peek_target(observation)
        if candidate is None:
            self._discard_return_lookahead(
                cancel_worker=True, reset_candidate=True
            )
            return None, None
        target_id = _object_id(candidate)
        category = self._lookahead_category(candidate)
        same_candidate = bool(
            self._same_target_id(target_id, self._return_candidate_id)
            and category == self._return_lookahead_category
        )
        if not same_candidate:
            self._discard_return_lookahead(
                cancel_worker=True, reset_candidate=False
            )
            self._return_candidate_id = target_id
            self._return_candidate_stable_steps = 1
            self._return_lookahead_category = category
        else:
            self._return_candidate_stable_steps += 1
        pose = self._return_pregrasp_pose(candidate, observation)
        if pose is None:
            self._discard_return_lookahead(
                cancel_worker=True, reset_candidate=True
            )
            return None, None
        return candidate, pose

    def _return_lookahead_pose_is_compatible(
        self,
        request: PlanningRequest,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        """Accept only bounded, belt-consistent drift while a search runs.

        An unreserved RGB-D prediction advances at belt speed on every frame.
        Comparing it to the worker snapshot with the ordinary 4 mm static-goal
        tolerance cancelled any search lasting more than one 50 Hz tick.  This
        gate models only the expected longitudinal motion; lateral/vertical
        jumps, orientation changes, reverse motion, and drift outside the
        already configured high-pregrasp correction corridor remain stale.

        Compatibility never authorizes execution of the old endpoint.  The
        commit path below invalidates the delivered route whenever the latest
        pose differs by the normal static-goal tolerance, then replans the now
        reserved and therefore stable intercept under all original limits.
        """

        reference = self._return_lookahead_pose
        if reference is None:
            reference = request.target_pose
        if (
            quat_distance(candidate_pose[3:], reference[3:])
            > self.config.trajectory_replan_orientation_tolerance
        ):
            return False
        delta = np.asarray(candidate_pose[:3] - reference[:3], dtype=float)
        static_tolerance = self.config.trajectory_replan_position_tolerance
        if np.linalg.norm(delta) <= static_tolerance:
            return True
        if abs(float(delta[2])) > static_tolerance:
            return False

        raw_velocity = _value(
            candidate,
            ("estimated_velocity", "linear_velocity", "velocity"),
            self._belt_velocity(observation),
        )
        try:
            velocity_values = np.asarray(
                raw_velocity, dtype=float
            ).reshape(-1)
        except (TypeError, ValueError):
            return False
        if velocity_values.size == 1:
            planar_velocity = self._belt_frame()[1][:2] * velocity_values[0]
        elif velocity_values.size >= 2:
            planar_velocity = velocity_values[:2]
        else:
            return False
        speed = float(np.linalg.norm(planar_velocity))
        if (
            planar_velocity.shape != (2,)
            or not np.all(np.isfinite(planar_velocity))
            or speed <= 1e-5
        ):
            return False
        submitted_at = self._return_lookahead_observed_at
        if submitted_at is None:
            return False
        elapsed = max(
            0.0,
            self._timestamp_without_advancing(observation) - submitted_at,
        )
        direction = planar_velocity / speed
        longitudinal = float(np.dot(delta[:2], direction))
        lateral = float(
            np.linalg.norm(delta[:2] - longitudinal * direction)
        )
        prediction = self._prediction_from_mapping(
            observation, _object_id(candidate)
        )
        uncertainty_raw = _value(
            candidate, "prediction_uncertainty",
            _value(prediction, "prediction_uncertainty", 0.0),
        )
        try:
            uncertainty = np.asarray(uncertainty_raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            uncertainty = np.zeros(0, dtype=float)
        lateral_uncertainty = (
            float(abs(uncertainty[1]))
            if uncertainty.size >= 2 and np.all(np.isfinite(uncertainty))
            else 0.0
        )
        # This tolerance controls only whether an unreserved background search
        # may keep running.  It never admits the old endpoint: any displacement
        # beyond the ordinary 4 mm tolerance causes full re-planning after the
        # target is atomically reserved.  Bound the RGB-D noise allowance so a
        # real lane jump is still stale.
        lateral_tolerance = float(
            np.clip(
                3.0 * lateral_uncertainty,
                static_tolerance,
                0.009,
            )
        )
        longitudinal_uncertainty = (
            float(abs(uncertainty[0]))
            if uncertainty.size >= 1 and np.all(np.isfinite(uncertainty))
            else 0.0
        )
        # A 15 Hz perception update can arrive as one 50 Hz control-tick jump.
        # Keep that bounded sampling/uncertainty allowance in the speculative
        # compatibility model.  Longitudinal distance itself is already
        # bounded by belt speed * elapsed time; an additional 35 mm radial cap
        # cancelled every valid search after about 170 ms.  The mandatory
        # frozen-endpoint replan still prevents the old endpoint executing.
        refresh_allowance = float(
            np.clip(
                3.0 * longitudinal_uncertainty + speed / 15.0,
                static_tolerance,
                0.018,
            )
        )
        maximum_advance = speed * elapsed + refresh_allowance
        return bool(
            longitudinal >= -static_tolerance
            and longitudinal <= maximum_advance + 1e-9
            and lateral <= lateral_tolerance
        )

    def _rebase_return_lookahead_result(
        self,
        result: PlanningResult,
        request: PlanningRequest,
        observation: Mapping[str, Any],
    ) -> PlanningResult | None:
        """Rebase timing/path structure onto the latest measured return state.

        Geometry is deliberately validated only at final installation, after
        target reservation, using the newest state.  Revalidating the same
        long first edge here and twice again during admission caused its own
        control-tick spike without strengthening the final proof.
        """

        if not result.success or result.path is None or result.cost is None:
            return None
        result_states = (
            np.asarray(result.path, dtype=float)
            if (
                result.path_fully_validated
                and self._planning_certificate_contract_required()
            )
            else self._corner_path(result.path)
        )
        path = [np.asarray(item, dtype=float).copy() for item in result_states]
        durations = result.edge_durations
        if (
            not path
            or durations is None
            or len(durations) != max(0, len(path) - 1)
            or any(
                not np.isfinite(duration) or duration < 0.0
                for duration in durations
            )
        ):
            return None
        current = self._joint_positions(observation)
        edge_durations = [float(item) for item in durations]
        start_error = float(np.max(np.abs(current - result.start_q)))
        if start_error > 1e-9:
            # Do not turn a certified ``expected_start -> first_corner`` edge
            # into the different chord ``measured -> first_corner``.  Even a
            # small feedback error can make that diagonal graze a narrow
            # joint-margin/orientation boundary while the short connector and
            # the original worker edge are each safe.  Prepend the measured
            # connector, validate it once against live MuJoCo at commit, then
            # execute the immutable certified polyline exactly as returned.
            path = [current.copy(), result.start_q.copy(), *path[1:]]
            already_settled_at_start = bool(
                start_error <= self.config.planner_corner_settle_tolerance
                and self._joint_velocity_is_settled(
                    observation,
                    self.config.planner_waypoint_velocity_tolerance,
                )
            )
            connector_duration = (
                float(self.config.control_timestep)
                if already_settled_at_start
                else self._realtime_rebased_edge_duration(
                    result.start_q,
                    current,
                    result.start_q,
                    0.0,
                    stage=request.stage,
                )
            )
            edge_durations = [connector_duration, *edge_durations]
        else:
            path[0] = current.copy()
        span = np.maximum(
            request.upper_limits - request.lower_limits, 1e-6
        )
        connector_length = (
            float(np.linalg.norm((current - result.start_q) / span))
            if start_error > 1e-9
            else 0.0
        )
        return PlanningResult(
            request_id=result.request_id,
            generation=result.generation,
            stage=result.stage,
            target_id=result.target_id,
            scene_epoch=result.scene_epoch,
            target_pose=result.target_pose,
            start_q=current,
            path=tuple(path),
            cost=(
                float(sum(edge_durations)),
                float(result.cost[1]) + connector_length,
                len(path),
            ),
            edge_durations=tuple(edge_durations),
            path_fully_validated=result.path_fully_validated,
            continuation_fully_validated=(
                result.continuation_fully_validated
            ),
            planning_time_s=result.planning_time_s,
            planner=result.planner,
        )

    def _lookahead_path_is_current(
        self,
        result: PlanningResult,
        request: PlanningRequest,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        """Fail closed on stale track, start state, first edge, or deadline."""

        if (
            not self._same_target_id(_object_id(candidate), request.target_id)
            or self._lookahead_category(candidate)
            != self._return_lookahead_category
            or not self._window_valid(candidate, observation)
            or request.scene_epoch != int(self._observed_scene_epoch or 0)
            or not self._return_lookahead_pose_is_compatible(
                request, candidate, candidate_pose, observation
            )
            or not result.success
            or result.path is None
        ):
            return False
        worker_durations = result.edge_durations
        has_worker_profile = bool(
            worker_durations is not None
            and len(worker_durations)
            == max(0, len(result.path) - 1)
            and all(
                np.isfinite(duration) and duration >= 0.0
                for duration in worker_durations
            )
        )
        if not has_worker_profile:
            # Metadata is constant-time to validate and must precede the live
            # first-edge scan.  A malformed/old worker result is stale; doing
            # hundreds of MuJoCo samples before discovering that fact would
            # violate the same 50 Hz latency contract as synchronous RRT.
            return False
        if (
            self._planning_certificate_contract_required()
            and not result.path_fully_validated
        ):
            return False
        if (
            request.stage == PolicyStage.PREGRASP.value
            and self._planning_certificate_contract_required()
            and (
                request.continuation_target_poses is None
                or not result.continuation_fully_validated
            )
        ):
            return False
        result_states = (
            np.asarray(result.path, dtype=float)
            if (
                result.path_fully_validated
                and self._planning_certificate_contract_required()
            )
            else self._corner_path(result.path)
        )
        path = [np.asarray(item, dtype=float).copy() for item in result_states]
        if not path:
            return False
        current = self._joint_positions(observation)
        if np.max(np.abs(current - result.start_q)) > 1e-9:
            return False
        path[0] = current.copy()
        first = path[1] if len(path) > 1 else path[0]
        # This is the sole look-ahead first-edge check.  The same observation
        # is passed directly into the atomic commit below, so installation can
        # consume this proof instead of scanning the edge a third time.
        if not self._global_edge_is_safe(
            current,
            first,
            minimum_tcp_height=request.minimum_tcp_height,
            minimum_joint_margin=request.minimum_joint_margin,
            minimum_downward_alignment=request.minimum_downward_alignment,
        ):
            return False
        window = _value(candidate, "grasp_window", {})
        deadline = _value(
            candidate,
            ("grasp_window_end", "window_end", "deadline"),
            _value(window, ("end_time", "end", "deadline"), None),
        )
        planned_time = _value(
            candidate, ("estimated_grasp_time", "grasp_time"), None
        )
        try:
            deadline = float(deadline)
            planned_time = float(planned_time)
        except (TypeError, ValueError):
            return False
        now = self._timestamp_without_advancing(observation)
        if len(worker_durations) == max(0, len(path) - 1):
            edge_durations = [float(item) for item in worker_durations]
            eta = float(sum(edge_durations))
        else:
            return False
        required_time = eta + self.config.descent_lead_time
        latency = _value(observation, "system_latency", {})
        if isinstance(latency, Mapping):
            for name in ("planning", "control", "robot_response"):
                try:
                    component = float(_value(latency, name, 0.0))
                except (TypeError, ValueError):
                    component = 0.0
                if np.isfinite(component) and component > 0.0:
                    required_time += component
        required_time += self.config.control_timestep
        earliest = now + required_time
        return bool(
            np.isfinite(deadline)
            and np.isfinite(planned_time)
            # ``planned_time`` is the visual module's nominal Cartesian
            # arrival estimate.  A collision-checked RRT route can be slower
            # while the target still has ample physical window remaining.
            # Rejecting merely because that nominal clock is earlier caused
            # RETURNING to discard feasible look-ahead paths until the part
            # reached the downstream boundary.  Admission is against the hard
            # visual deadline; after the atomic reservation below, the normal
            # route-aware refinement moves time *and pose* together and either
            # replans the changed endpoint or fails closed.
            and earliest
            <= deadline - self.config.intercept_deadline_margin + 1e-9
        )

    def _commit_return_lookahead(
        self,
        result: PlanningResult,
        request: PlanningRequest,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
        now: float,
    ) -> bool:
        """Atomically reserve and install a fully revalidated look-ahead path."""

        selected = self._select_target(observation)
        if (
            selected is None
            or not self._same_target_id(_object_id(selected), request.target_id)
            or self._lookahead_category(selected)
            != self._return_lookahead_category
        ):
            if selected is not None:
                _hook(
                    self.env,
                    ("release_selected_target", "release_target_reservation"),
                    ((_object_id(selected),),),
                )
            return False

        # Preserve local copies: clearing the completed attempt intentionally
        # invalidates worker state and the installed return route.
        latest_candidate_pose = candidate_pose.copy()
        route_endpoint_pose = result.target_pose.copy()
        committed_pose = latest_candidate_pose.copy()
        committed_pose[3:] = result.target_pose[3:]
        self._clear_target()
        self._target = selected
        self._target_id = _object_id(selected)
        self._grasp_gripper_action = self._grip_action(selected)
        self._attempt_started_at = now
        self._last_seen_target_at = now
        self._grasp_pose = committed_pose.copy()
        self._grasp_pose[2] -= self.config.approach_height
        self._selected_grasp_quaternion = result.target_pose[3:].copy()
        self._transition(
            PolicyStage.PREGRASP,
            now,
            reason="return_lookahead_ready",
            event="target_selected",
        )
        if not self._install_planning_result(
            result,
            request,
            committed_pose,
            observation,
            first_edge_already_validated=True,
        ):
            # This should only be reachable if live state changed inside a
            # duck-typed hook.  No path command has been emitted; release the
            # reservation and recover to staging.
            _hook(
                self.env,
                ("release_selected_target", "release_target_reservation"),
                ((self._target_id,),),
            )
            self._clear_target()
            self._transition(
                PolicyStage.RECOVERING,
                now,
                reason="lookahead_first_edge_stale",
            )
            return False
        # The request ran under RETURNING's generation but was built with the
        # strict PREGRASP constraints.  Bind the installed cache to the new
        # active generation after the stage transition.
        self._safe_path_generation = self._planning_generation
        self._safe_path_target_id = self._target_id
        self._safe_path_stage = PolicyStage.PREGRASP

        accepted, pose_changed = self._refine_pregrasp_intercept(
            committed_pose, observation
        )
        moving_endpoint_changed = bool(
            np.linalg.norm(
                latest_candidate_pose[:3] - route_endpoint_pose[:3]
            )
            > self.config.trajectory_replan_position_tolerance
            or quat_distance(
                latest_candidate_pose[3:], route_endpoint_pose[3:]
            )
            > self.config.trajectory_replan_orientation_tolerance
        )
        if not accepted:
            self._terminal_motion_plan_reason = (
                self._last_error or "insufficient_intercept_time"
            )
            self._invalidate_motion_plan()
            self._record_motion_plan_failure()
        elif pose_changed or moving_endpoint_changed:
            # The worker legitimately planned while the unreserved visual
            # target moved.  Reserving it freezes the admitted intercept, but
            # the old endpoint is never executed: hold and request a complete
            # fresh IK/RRT validation for the now-stable pose next tick.
            self._invalidate_motion_plan()
        return True

    def _update_return_lookahead(
        self,
        observation: Mapping[str, Any],
        *,
        acquisition_ready: bool,
        now: float,
    ) -> bool:
        """Advance speculative PREGRASP planning; never emit an arm command."""

        if self.config.dynamic_intercept_tracking:
            return False

        candidate, candidate_pose = self._stable_return_candidate(observation)
        if candidate is None or candidate_pose is None:
            return False
        minimum_stable = max(2, self.config.acquisition_entry_stable_steps)
        worker = self._ensure_planning_worker()
        if worker is None:
            return False

        request = self._return_lookahead_request
        if request is not None and (
            not self._same_target_id(request.target_id, _object_id(candidate))
            or request.scene_epoch != int(self._observed_scene_epoch or 0)
            or not self._return_lookahead_pose_is_compatible(
                request, candidate, candidate_pose, observation
            )
        ):
            self._discard_return_lookahead(cancel_worker=True)
            request = None

        # The shared result channel belongs to the ordinary return request
        # while one is active.  When look-ahead owns it, cache its matching
        # completion even if the measured acquisition latch briefly drops;
        # reservation and command remain gated below.
        if request is not None and self._planning_request is None:
            for _ in range(2):
                result = worker.poll()
                if result is None:
                    break
                if not self._result_matches_request(result, request):
                    self._planning_stale_results += 1
                    continue
                if result.cancelled or not result.success:
                    self._return_lookahead_failures += 1
                    self._last_error = (
                        result.error or "return lookahead cancelled"
                    )
                    self._discard_return_lookahead(cancel_worker=False)
                    return False
                self._return_lookahead_result = result
                break

        cached_result = self._return_lookahead_result
        request = self._return_lookahead_request
        if cached_result is not None and request is not None:
            if not acquisition_ready or not self._safe_joint_path:
                return False
            rebased_result = self._rebase_return_lookahead_result(
                cached_result, request, observation
            )
            if rebased_result is None or not self._lookahead_path_is_current(
                rebased_result,
                request,
                candidate,
                candidate_pose,
                observation,
            ):
                self._planning_stale_results += 1
                self._return_lookahead_failures += 1
                self._discard_return_lookahead(cancel_worker=False)
                return False
            self._return_lookahead_result = None
            self._return_lookahead_request = None
            return self._commit_return_lookahead(
                rebased_result,
                request,
                candidate,
                candidate_pose,
                observation,
                now,
            )

        if self._return_lookahead_request is not None:
            return False
        if (
            not acquisition_ready
            or self._return_candidate_stable_steps < minimum_stable
            or self._planning_request is not None
            or not self._safe_joint_path
        ):
            # Candidate tracking/polling is allowed before the gate;
            # submitting, reserving, installing and commanding are not.
            return False
        if (
            self._return_lookahead_failures
            >= self.config.return_lookahead_retry_attempts
        ):
            return False
        request = self._make_return_lookahead_request(
            candidate, candidate_pose, observation
        )
        if request is None or not worker.submit(request):
            return False
        self._return_lookahead_request = request
        self._return_lookahead_result = None
        self._return_lookahead_category = self._lookahead_category(candidate)
        self._return_lookahead_pose = candidate_pose.copy()
        self._return_lookahead_scene_epoch = request.scene_epoch
        self._return_lookahead_observed_at = (
            self._timestamp_without_advancing(observation)
        )
        return False

    def _discard_prefetch(
        self,
        *,
        cancel_worker: bool,
        preserve_retreat: bool = False,
    ) -> None:
        """Clear speculative ownership without touching the primary command."""

        if cancel_worker:
            self._prefetch_generation += 1
            worker = getattr(self, "_prefetch_worker", None)
            if worker is not None:
                worker.cancel_pending()
        self._prefetch_request = None
        self._prefetch_completed_request = None
        self._prefetch_result = None
        self._prefetch_category = None
        self._prefetch_pose = None
        self._prefetch_scene_epoch = None
        self._prefetch_observed_at = None
        self._prefetch_candidate_id = None
        self._prefetch_candidate_stable_steps = 0
        self._prefetch_bound_candidate_id = None
        self._prefetch_candidate_missing_since = None
        self._carry_prefetch_request = None
        self._carry_prefetch_completed_request = None
        self._carry_prefetch_result = None
        self._carry_prefetch_target_id = None
        self._carry_prefetch_bin_index = None
        self._carry_prefetch_place_pose = None
        self._carry_prefetch_anchor_q = None
        self._carry_prefetch_scene_epoch = None
        if preserve_retreat:
            return
        self._prefetch_phase = None
        self._prefetch_retreat_request = None
        self._prefetch_completed_retreat_request = None
        self._prefetch_retreat_result = None
        self._prefetch_retreat_failed = False
        self._prefetch_retreat_not_before = None
        self._prefetch_release_terminal_confirmed = False
        self._prefetch_anchor_q = None
        self._prefetch_place_q = None
        self._prefetch_retreat_pose = None
        self._prefetch_source_bin_index = None
        self._prefetch_release_complete = False
        self._prefetch_retreat_installed = False
        self._prefetch_failures = 0

    @staticmethod
    def _joint_velocity_is_settled(
        observation: Mapping[str, Any], tolerance: float
    ) -> bool:
        raw = _value(
            observation,
            ("joint_vel", "joint_velocity", "arm_joint_velocities"),
            None,
        )
        try:
            velocity = np.asarray(raw, dtype=float).reshape(-1)[:7]
        except (TypeError, ValueError):
            return False
        return bool(
            velocity.shape == (7,)
            and np.all(np.isfinite(velocity))
            and np.max(np.abs(velocity)) <= max(0.0, float(tolerance))
        )

    @staticmethod
    def _release_is_physically_clear(observation: Mapping[str, Any]) -> bool:
        """Require actual open jaws and no residual part/finger contact."""

        gripper = _value(observation, "gripper", None)
        if not isinstance(gripper, Mapping):
            return False
        open_flag = _value(gripper, "is_open", None)
        fraction = _value(gripper, "open_fraction", None)
        target_fraction = _value(gripper, "target_open_fraction", fraction)
        try:
            fraction_value = float(fraction)
            target_value = float(target_fraction)
        except (TypeError, ValueError):
            fraction_value = target_value = float("nan")
        opened = bool(open_flag) or bool(
            np.isfinite(fraction_value)
            and np.isfinite(target_value)
            and fraction_value >= 0.985
            and target_value >= 0.985
        )
        residual = bool(
            _value(gripper, "any_external_finger_contact", False)
            or _value(gripper, "both_fingers_contact", False)
            or _value(gripper, "is_holding", False)
        )
        return bool(opened and not residual)

    def _latch_released_target_terminal(
        self, observation: Mapping[str, Any]
    ) -> None:
        """Remember policy-visible proof that the placed object is sorted."""

        if self._prefetch_release_terminal_confirmed or self._target_id is None:
            return
        target = self._find_target(observation, self._target_id)
        self._prefetch_release_terminal_confirmed = bool(
            target is not None and _status_name(target) == "SORTED"
        )

    @staticmethod
    def _retreat_result_uses_short_handoff(
        result: PlanningResult, request: PlanningRequest
    ) -> bool:
        """Whether A selected a shorter endpoint than its full +Y retreat."""

        poses = tuple(request.target_poses)
        if len(poses) <= 1:
            return False
        full_retreat_y = max(float(np.asarray(pose)[1]) for pose in poses)
        return bool(float(result.target_pose[1]) < full_retreat_y - 1e-9)

    def _capture_prefetch_place_anchor(
        self, observation: Mapping[str, Any]
    ) -> None:
        """Snapshot certified final-place q and submit future retreat leg A."""

        worker = self._ensure_prefetch_worker()
        if (
            worker is None
            or self._prefetch_retreat_result is not None
            or self._prefetch_retreat_request is not None
            or self._prefetch_retreat_failed
            or self._place_pose is None
            or not self._safe_joint_path
            or self._safe_path_stage != PolicyStage.TRANSFERRING
            or self._transfer_waypoint_index
            != max(0, len(self._transfer_waypoints) - 1)
            or self._safe_path_target_pose is None
            or np.linalg.norm(
                self._safe_path_target_pose[:3] - self._place_pose[:3]
            ) > self.config.trajectory_replan_position_tolerance
            or quat_distance(
                self._safe_path_target_pose[3:], self._place_pose[3:]
            ) > self.config.trajectory_replan_orientation_tolerance
        ):
            return
        if (
            self._planning_certificate_contract_required()
            and not self._safe_path_worker_validated
        ):
            return
        place_q = np.asarray(self._safe_joint_path[-1], dtype=float).copy()
        snapshot = self._future_open_planning_snapshot(place_q)
        if snapshot is None:
            return
        future_arm, full_qpos, scene_epoch = snapshot
        retreat = self._place_pose.copy()
        retreat[1] += self.config.release_retreat_distance
        request = self._future_prefetch_request(
            target_id=self._target_id,
            target_pose=retreat,
            start_q=future_arm,
            full_qpos=full_qpos,
            scene_epoch=scene_epoch,
            target=None,
            phase="retreat",
        )
        if request is None or not worker.submit(request):
            return
        self._prefetch_place_q = place_q
        self._prefetch_retreat_pose = retreat
        try:
            self._prefetch_source_bin_index = int(
                self._bin_index(self._target)
            )
        except (TypeError, ValueError):
            self._prefetch_source_bin_index = None
        self._prefetch_retreat_request = request
        self._prefetch_retreat_failed = False
        self._prefetch_phase = "retreat"

    def _submit_carry_prefetch(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> None:
        """Plan canonical corridor-to-bin leg C before the grasp completes."""

        worker = self._ensure_prefetch_worker()
        target_id = _object_id(target)
        if (
            worker is None
            or target_id is None
            or self._carry_prefetch_request is not None
            or self._carry_prefetch_result is not None
            or self._prefetch_retreat_request is not None
            or self._prefetch_retreat_result is not None
            or self._prefetch_request is not None
            or self._prefetch_result is not None
        ):
            return
        calibration = self._carry_corridor_calibration()
        place_pose = self._bin_pose(target, observation)
        if calibration is None or place_pose is None:
            return
        anchor_q, _anchor_pose = calibration
        grip_action = self._grip_action(target)
        snapshot = self._future_carry_planning_snapshot(
            anchor_q, grip_action
        )
        if snapshot is None:
            return
        future_arm, full_qpos, scene_epoch = snapshot
        request = self._future_prefetch_request(
            target_id=target_id,
            target_pose=place_pose,
            start_q=future_arm,
            full_qpos=full_qpos,
            scene_epoch=scene_epoch,
            target=target,
            phase="carry",
            observation=observation,
        )
        if request is None or not worker.submit(request):
            return
        self._place_pose = place_pose.copy()
        self._carry_prefetch_request = request
        self._carry_prefetch_completed_request = None
        self._carry_prefetch_result = None
        self._carry_prefetch_target_id = target_id
        self._carry_prefetch_bin_index = self._bin_index(target)
        self._carry_prefetch_place_pose = place_pose.copy()
        self._carry_prefetch_anchor_q = anchor_q.copy()
        self._carry_prefetch_scene_epoch = scene_epoch
        self._prefetch_phase = "carry"

    def _submit_prefetch_retreat_from_carry(self) -> None:
        """Use C's certified place endpoint to start A on the same worker."""

        if self.config.dynamic_intercept_tracking:
            # Dynamic collection completes the safe return instead of holding
            # an immutable A/B anchor while a state-dependent target changes.
            return

        result = self._carry_prefetch_result
        request = self._carry_prefetch_completed_request
        worker = self._prefetch_worker
        place_pose = self._carry_prefetch_place_pose
        if (
            result is None
            or request is None
            or worker is None
            or place_pose is None
            or result.path is None
            or self._prefetch_retreat_request is not None
            or self._prefetch_retreat_result is not None
            or self._prefetch_retreat_failed
        ):
            return
        place_q = np.asarray(result.path[-1], dtype=float).copy()
        snapshot = self._future_open_planning_snapshot(place_q)
        if snapshot is None:
            return
        future_arm, full_qpos, scene_epoch = snapshot
        if scene_epoch != request.scene_epoch:
            return
        retreat = place_pose.copy()
        retreat[1] += self.config.release_retreat_distance
        retreat_request = self._future_prefetch_request(
            target_id=request.target_id,
            target_pose=retreat,
            start_q=future_arm,
            full_qpos=full_qpos,
            scene_epoch=scene_epoch,
            target=None,
            phase="retreat",
        )
        if retreat_request is None or not worker.submit(retreat_request):
            return
        self._prefetch_place_q = place_q
        self._prefetch_retreat_pose = retreat
        self._prefetch_source_bin_index = self._carry_prefetch_bin_index
        self._prefetch_retreat_request = retreat_request
        self._prefetch_retreat_failed = False
        self._prefetch_phase = "retreat"

    def _update_carry_prefetch(
        self,
        target: Any,
        observation: Mapping[str, Any],
    ) -> None:
        """Advance C then A without reserving or commanding speculative data."""

        if self._ensure_prefetch_worker() is None:
            return
        self._submit_carry_prefetch(target, observation)
        self._poll_prefetch()
        if self._carry_prefetch_result is not None:
            self._submit_prefetch_retreat_from_carry()

    def _install_carry_prefetch(
        self, observation: Mapping[str, Any]
    ) -> bool:
        """Install C only from the measured, settled canonical anchor."""

        result = self._carry_prefetch_result
        request = self._carry_prefetch_completed_request
        place_pose = self._carry_prefetch_place_pose
        anchor = self._carry_prefetch_anchor_q
        if (
            result is None
            or request is None
            or place_pose is None
            or anchor is None
            or self._stage != PolicyStage.TRANSFERRING
            or not self._same_target_id(
                self._target_id, self._carry_prefetch_target_id
            )
            or self._carry_prefetch_bin_index
            != self._bin_index(self._target)
            or request.scene_epoch != int(self._observed_scene_epoch or 0)
            or self._carry_prefetch_scene_epoch != request.scene_epoch
            or self._place_pose is None
            or np.linalg.norm(self._place_pose[:3] - place_pose[:3])
            > self.config.trajectory_replan_position_tolerance
            or quat_distance(self._place_pose[3:], place_pose[3:])
            > self.config.trajectory_replan_orientation_tolerance
            or not self._prefetch_result_is_complete(result, request)
        ):
            return False
        current = self._joint_positions(observation)
        if (
            np.max(np.abs(current - anchor))
            > self.config.planner_corner_settle_tolerance
            or not self._joint_velocity_is_settled(
                observation, self.config.planner_waypoint_velocity_tolerance
            )
        ):
            return False
        rebased = self._rebase_return_lookahead_result(
            result, request, observation
        )
        if rebased is None or rebased.path is None:
            return False
        first = rebased.path[1] if len(rebased.path) > 1 else rebased.path[0]
        if not self._global_edge_is_safe(
            current,
            first,
            minimum_tcp_height=request.minimum_tcp_height,
            minimum_joint_margin=request.minimum_joint_margin,
            minimum_downward_alignment=(
                self.config.acquisition_minimum_downward_alignment
            ),
        ):
            return False
        self._invalidate_motion_plan()
        target_pose = place_pose.copy()
        if not self._install_planning_result(
            rebased,
            request,
            target_pose,
            observation,
            first_edge_already_validated=True,
        ):
            return False
        self._safe_path_generation = self._planning_generation
        self._safe_path_target_id = self._target_id
        self._safe_path_stage = PolicyStage.TRANSFERRING
        self._safe_path_scene_epoch = int(self._observed_scene_epoch or 0)
        self._safe_path_from_carry_prefetch = True
        self._carry_path_recovery_active = False
        self._carry_path_recovery_predictive = False
        self._carry_prefetch_request = None
        self._carry_prefetch_completed_request = None
        self._carry_prefetch_result = None
        self._carry_prefetch_target_id = None
        self._carry_prefetch_bin_index = None
        self._carry_prefetch_place_pose = None
        self._carry_prefetch_anchor_q = None
        self._carry_prefetch_scene_epoch = None
        return True

    def _prefetch_pose_is_compatible(
        self,
        request: PlanningRequest,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        """Bound moving RGB-D drift; never authorize the old endpoint itself."""

        reference = self._prefetch_pose
        if reference is None:
            reference = request.target_pose
        if request.planned_target_time_s is not None:
            # A late B is not executed merely because its speculative visual
            # pose remains nearby.  Atomic commit first binds the exact
            # planned clock in the environment and requires the freshly
            # refined object/body XY to match result.target_pose within the
            # strict 4 mm endpoint tolerance.  Rechecking raw 15 Hz yaw or
            # lateral innovations here is therefore redundant and harmful:
            # short-axis yaw aliases repeatedly discarded a fully certified
            # route before that authoritative reservation gate ran.  Retain
            # only the immutable request/result until the live gate can decide.
            candidate_array = np.asarray(candidate_pose, dtype=float).reshape(-1)
            return bool(
                candidate_array.shape == (7,)
                and np.all(np.isfinite(candidate_array))
                and abs(float(candidate_array[2] - reference[2]))
                <= self.config.trajectory_replan_position_tolerance + 1e-9
            )
        # The moving visual pose may jitter by more than the 25 mrad static
        # replanning threshold between 15 Hz depth frames (short-axis objects
        # in particular).  Keeping a speculative search does not authorize
        # that old quaternion: commit still freezes the latest pose, preserves
        # only the worker-certified route orientation, and invalidates the
        # path whenever their difference exceeds the strict 25 mrad execution
        # threshold.  Bound search retention by the already validated
        # high-pregrasp orientation envelope instead of restarting every
        # camera frame.
        if quat_distance(candidate_pose[3:], reference[3:]) > max(
            self.config.trajectory_replan_orientation_tolerance,
            self.config.pregrasp_blend_orientation_tolerance,
        ):
            return False
        delta = np.asarray(candidate_pose[:3] - reference[:3], dtype=float)
        tolerance = self.config.trajectory_replan_position_tolerance
        if np.linalg.norm(delta) <= tolerance:
            return True
        if abs(float(delta[2])) > tolerance:
            return False
        raw_velocity = _value(
            candidate,
            ("estimated_velocity", "linear_velocity", "velocity"),
            self._belt_velocity(observation),
        )
        try:
            velocity = np.asarray(raw_velocity, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False
        planar = (
            self._belt_frame()[1][:2] * velocity[0]
            if velocity.size == 1
            else velocity[:2]
        )
        speed = float(np.linalg.norm(planar))
        if planar.shape != (2,) or not np.all(np.isfinite(planar)) or speed <= 1e-5:
            return False
        if self._prefetch_observed_at is None:
            return False
        elapsed = max(
            0.0,
            self._timestamp_without_advancing(observation)
            - self._prefetch_observed_at,
        )
        direction = planar / speed
        longitudinal = float(np.dot(delta[:2], direction))
        lateral = float(
            np.linalg.norm(delta[:2] - longitudinal * direction)
        )
        return bool(
            longitudinal >= -tolerance
            and longitudinal <= speed * elapsed + 0.018 + 1e-9
            and lateral <= max(tolerance, 0.009) + 1e-9
        )

    def _track_prefetch_candidate(
        self, observation: Mapping[str, Any]
    ) -> tuple[Any | None, np.ndarray | None]:
        candidate = self._peek_target(observation)
        if candidate is None:
            # With no previously bound B target this is the normal gap before
            # the next fixed-cadence spawn; A continues to its anchor and may
            # wait there.  Once a target was bound, tolerate a few missing
            # RGB-D samples, but never retain ownership indefinitely after it
            # disappeared or left the physical window.
            if self._prefetch_bound_candidate_id is None:
                self._prefetch_candidate_id = None
                self._prefetch_candidate_stable_steps = 0
                self._prefetch_candidate_missing_since = None
                return None, None
            now = self._timestamp_without_advancing(observation)
            if self._prefetch_candidate_missing_since is None:
                self._prefetch_candidate_missing_since = now
            if (
                now - self._prefetch_candidate_missing_since
                > self.config.target_lost_timeout + 1e-9
            ):
                self._discard_prefetch(cancel_worker=True)
            return None, None
        target_id = _object_id(candidate)
        category = self._lookahead_category(candidate)
        same = bool(
            (
                self._same_target_id(target_id, self._prefetch_candidate_id)
                or self._same_target_id(
                    target_id, self._prefetch_bound_candidate_id
                )
            )
            and category == self._prefetch_category
        )
        if not same:
            if self._prefetch_request is not None or self._prefetch_result is not None:
                self._discard_prefetch(cancel_worker=True, preserve_retreat=True)
            self._prefetch_candidate_id = target_id
            self._prefetch_candidate_stable_steps = 1
            self._prefetch_category = category
            self._prefetch_bound_candidate_id = target_id
        else:
            self._prefetch_candidate_stable_steps += 1
        self._prefetch_bound_candidate_id = target_id
        self._prefetch_candidate_missing_since = None
        pose = self._return_pregrasp_pose(candidate, observation)
        if pose is None:
            return None, None
        return candidate, pose

    def _prefetch_result_is_complete(
        self, result: PlanningResult, request: PlanningRequest
    ) -> bool:
        if (
            not self._result_matches_request(result, request)
            or not result.success
            or result.path is None
            or result.cost is None
            or result.edge_durations is None
            or len(result.edge_durations) != max(0, len(result.path) - 1)
        ):
            return False
        if request.cartesian_local_only:
            maximum_delta = request.maximum_goal_joint_delta
            if maximum_delta is None or result.path is None:
                return False
            states = np.asarray(result.path, dtype=float)
            if (
                states.ndim != 2
                or states.shape[1:] != (7,)
                or np.max(np.abs(states[-1] - request.start_q))
                > maximum_delta + 1e-9
                or (
                    len(states) > 1
                    and np.max(np.abs(np.diff(states, axis=0)))
                    > request.continuation_max_joint_step + 1e-9
                )
            ):
                return False
        if (
            self._prefetch_certificate_contract_required()
            and not result.path_fully_validated
        ):
            return False
        if (
            request.stage == PolicyStage.PREGRASP.value
            and self._prefetch_certificate_contract_required()
            and (
                request.continuation_target_poses is None
                or not result.continuation_fully_validated
            )
        ):
            return False
        return True

    def _prefetch_refresh_execution_reserve(
        self,
        candidate: Any,
        request: PlanningRequest | None,
        result: PlanningResult | None,
    ) -> float | None:
        """Return a matching certified B ETA plus refresh-only reserve.

        A visual clock refresh always creates a new immutable request and a
        new full path/continuation certificate.  The completed result may only
        tighten that new worker's search watchdog: it is an execution-time
        witness for the same target, scene, and A anchor, never a certificate
        for the refreshed pose.
        """

        anchor = self._prefetch_anchor_q
        if (
            request is None
            or result is None
            or anchor is None
            or not self._prefetch_result_is_complete(result, request)
            or not self._same_target_id(
                request.target_id, _object_id(candidate)
            )
            or request.scene_epoch != int(self._observed_scene_epoch or 0)
            or request.start_q.shape != anchor.shape
            or np.max(np.abs(request.start_q - anchor))
            > self.config.planner_waypoint_tolerance + 1e-9
            or self._lookahead_category(candidate) != self._prefetch_category
            or result.cost is None
        ):
            return None
        try:
            eta = float(result.cost[0])
        except (TypeError, ValueError, IndexError):
            return None
        reserve = eta + float(self.config.prefetch_refresh_execution_margin)
        if not np.isfinite(reserve) or reserve <= 0.0:
            return None
        return reserve

    def _poll_prefetch(self) -> None:
        worker = self._prefetch_worker
        if worker is None:
            return
        for _ in range(2):
            result = worker.poll()
            if result is None:
                break
            carry_request = self._carry_prefetch_request
            retreat_request = self._prefetch_retreat_request
            request = self._prefetch_request
            if carry_request is not None and self._result_matches_request(
                result, carry_request
            ):
                self._carry_prefetch_request = None
                if self._prefetch_result_is_complete(result, carry_request):
                    self._carry_prefetch_completed_request = carry_request
                    self._carry_prefetch_result = result
                else:
                    self._prefetch_failures += 1
                    self._prefetch_phase = None
                continue
            if retreat_request is not None and self._result_matches_request(
                result, retreat_request
            ):
                self._prefetch_retreat_request = None
                if self._prefetch_result_is_complete(result, retreat_request):
                    self._prefetch_completed_retreat_request = retreat_request
                    self._prefetch_retreat_result = result
                    self._prefetch_retreat_failed = False
                    assert result.path is not None
                    # A first tries the configured full withdrawal, then a
                    # shorter open-jaw handoff only when the full endpoint has
                    # no commandable local branch.  Bind all later execution
                    # and B-version checks to the pose that was actually
                    # certified, never to the rejected primary endpoint.
                    self._prefetch_retreat_pose = np.asarray(
                        result.target_pose, dtype=float
                    ).copy()
                    self._prefetch_anchor_q = np.asarray(result.path[-1], dtype=float).copy()
                    self._prefetch_phase = "pregrasp"
                else:
                    self._prefetch_failures += 1
                    # A is a deterministic local Cartesian corridor from an
                    # immutable C endpoint.  A non-cancelled failure cannot
                    # improve while neither endpoint nor scene changes, so do
                    # not resubmit it at 50 Hz.  The normal return path remains
                    # available and reset/new-target cleanup clears the latch.
                    if not result.cancelled:
                        self._prefetch_retreat_failed = True
                    self._prefetch_phase = None
                continue
            if request is not None and self._result_matches_request(result, request):
                self._prefetch_request = None
                if self._prefetch_result_is_complete(result, request):
                    self._prefetch_completed_request = request
                    self._prefetch_result = result
                else:
                    self._prefetch_failures += 1
                    if not result.cancelled:
                        self._prefetch_terminal_b_target_id = request.target_id
                continue
            self._prefetch_stale_results += 1

    def _install_prefetch_retreat(
        self, observation: Mapping[str, Any]
    ) -> bool:
        result = self._prefetch_retreat_result
        request = self._prefetch_retreat_request
        # Completed A clears the active request, so reconstruct its immutable
        # identity from the result fields and cached endpoint constraints.
        if result is None:
            return False
        if request is None:
            request = getattr(self, "_prefetch_completed_retreat_request", None)
        if request is None:
            return False
        if not self._prefetch_release_complete or not self._release_is_physically_clear(
            observation
        ):
            return False
        if (
            self._retreat_result_uses_short_handoff(result, request)
            and not self._prefetch_release_terminal_confirmed
        ):
            # Twenty millimetres is a fast, fully certified open-jaw handoff,
            # not a substitute for the ordinary 55 mm placement withdrawal.
            # It may own return only after the policy-visible lifecycle says
            # the released object is sorted and live finger sensing is clear.
            return False
        current = self._joint_positions(observation)
        if np.max(np.abs(current - result.start_q)) > (
            self.config.planner_waypoint_tolerance
        ):
            return False
        rebased = self._rebase_return_lookahead_result(result, request, observation)
        if rebased is None or rebased.path is None:
            return False
        first = rebased.path[1] if len(rebased.path) > 1 else rebased.path[0]
        if not self._global_edge_is_safe(
            current,
            first,
            minimum_tcp_height=request.minimum_tcp_height,
            minimum_joint_margin=request.minimum_joint_margin,
            minimum_downward_alignment=(
                self.config.acquisition_minimum_downward_alignment
            ),
        ):
            return False
        self._invalidate_motion_plan()
        if not self._install_planning_result(
            rebased,
            request,
            self._prefetch_retreat_pose,
            observation,
            first_edge_already_validated=True,
        ):
            return False
        self._safe_path_generation = self._planning_generation
        self._safe_path_target_id = self._target_id
        self._safe_path_stage = PolicyStage.RETURNING
        self._prefetch_retreat_installed = True
        eta = self._safe_path_eta_s
        now = self._timestamp_without_advancing(observation)
        self._prefetch_retreat_not_before = (
            now + float(eta)
            if eta is not None and np.isfinite(eta) and eta >= 0.0
            else None
        )
        return True

    def _prefetch_anchor_remaining_time(
        self, observation: Mapping[str, Any]
    ) -> float:
        """Conservative remaining A time before B may start at its anchor."""

        anchor = self._prefetch_anchor_q
        if anchor is not None:
            current = self._joint_positions(observation)
            if (
                np.max(np.abs(current - anchor))
                <= self.config.planner_corner_settle_tolerance
                and self._joint_velocity_is_settled(
                    observation,
                    self.config.planner_waypoint_velocity_tolerance,
                )
            ):
                return 0.0
        now = self._timestamp_without_advancing(observation)
        not_before = self._prefetch_retreat_not_before
        if not_before is not None and np.isfinite(not_before):
            return max(
                self.config.control_timestep,
                float(not_before) - now,
            )
        result = self._prefetch_retreat_result
        if (
            not self._prefetch_retreat_installed
            and result is not None
            and result.cost is not None
            and np.isfinite(float(result.cost[0]))
        ):
            return max(0.0, float(result.cost[0]))
        # A completed B is meaningful only from its immutable retreat anchor.
        # Without either measured settlement or a frozen A execution clock,
        # its earliest start is unknown and must fail closed.
        return float("inf")

    def _prefetch_deadline_valid(
        self,
        result: PlanningResult,
        candidate: Any,
        observation: Mapping[str, Any],
        request: PlanningRequest | None = None,
        *,
        enforce_request_clock: bool = True,
    ) -> bool:
        window = _value(candidate, "grasp_window", {})
        deadline = _value(
            candidate,
            ("grasp_window_end", "window_end", "deadline"),
            _value(window, ("end_time", "end", "deadline"), None),
        )
        try:
            deadline_value = float(deadline)
        except (TypeError, ValueError):
            return False
        eta = float(result.cost[0]) if result.cost is not None else float("inf")
        latency = _value(observation, "system_latency", {})
        latency_sum = 0.0
        if isinstance(latency, Mapping):
            for name in ("planning", "control", "robot_response"):
                try:
                    component = float(_value(latency, name, 0.0))
                except (TypeError, ValueError):
                    component = 0.0
                if np.isfinite(component) and component > 0.0:
                    latency_sum += component
        required = (
            self._prefetch_anchor_remaining_time(observation)
            +
            eta
            + self.config.descent_lead_time
            + latency_sum
            + self.config.control_timestep
        )
        hard_intercept = (
            deadline_value - self.config.intercept_deadline_margin
        )
        planned_intercept = (
            None if request is None else request.planned_target_time_s
        )
        if enforce_request_clock and planned_intercept is not None:
            if (
                not np.isfinite(planned_intercept)
                or planned_intercept > hard_intercept + 1e-9
            ):
                return False
            hard_intercept = min(hard_intercept, planned_intercept)
        return bool(
            np.isfinite(deadline_value)
            and np.isfinite(required)
            and self._timestamp_without_advancing(observation) + required
            <= hard_intercept + 1e-9
        )

    def _prefetch_candidate_has_minimum_slack(
        self, candidate: Any, observation: Mapping[str, Any]
    ) -> bool:
        """Bound an anchor hold by the remaining physical grasp window."""

        window = _value(candidate, "grasp_window", {})
        deadline = _value(
            candidate,
            ("grasp_window_end", "window_end", "deadline"),
            _value(window, ("end_time", "end", "deadline"), None),
        )
        try:
            deadline_value = float(deadline)
        except (TypeError, ValueError):
            return False
        minimum_remaining = (
            self.config.descent_lead_time
            + self.config.intercept_deadline_margin
            + self.config.control_timestep
        )
        return bool(
            np.isfinite(deadline_value)
            and self._timestamp_without_advancing(observation) + minimum_remaining
            <= deadline_value + 1e-9
        )

    def _prefetch_should_hold_anchor(
        self, observation: Mapping[str, Any]
    ) -> bool:
        """Hold only at measured retreat q while a still-useful B is pending."""

        anchor = self._prefetch_anchor_q
        candidate = self._peek_target(observation)
        if (
            anchor is None
            or not self._prefetch_retreat_installed
            or (
                self._prefetch_request is None
                and self._prefetch_result is None
            )
            or candidate is None
            or not self._prefetch_candidate_has_minimum_slack(
                candidate, observation
            )
        ):
            return False
        if (
            self._prefetch_result is not None
            and not self._prefetch_deadline_valid(
                self._prefetch_result,
                candidate,
                observation,
                (
                    self._prefetch_request
                    if self._prefetch_request is not None
                    else self._prefetch_completed_request
                ),
            )
        ):
            return False
        current = self._joint_positions(observation)
        return bool(
            np.max(np.abs(current - anchor))
            <= self.config.planner_corner_settle_tolerance
        )

    def _submit_prefetch_pregrasp(
        self,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        execution_reserve_s: float | None = None,
    ) -> None:
        worker = self._ensure_prefetch_worker()
        anchor = self._prefetch_anchor_q
        if (
            worker is None
            or anchor is None
            or self._prefetch_request is not None
            or self._prefetch_result is not None
            or self._prefetch_candidate_stable_steps
            < max(2, self.config.acquisition_entry_stable_steps)
        ):
            return
        snapshot = self._future_open_planning_snapshot(anchor)
        if snapshot is None:
            return
        future_arm, full_qpos, scene_epoch = snapshot
        request = self._future_prefetch_request(
            target_id=_object_id(candidate),
            target_pose=candidate_pose,
            start_q=future_arm,
            full_qpos=full_qpos,
            scene_epoch=scene_epoch,
            target=candidate,
            phase="pregrasp",
            observation=observation,
            execution_reserve_s=execution_reserve_s,
        )
        if request is None or not worker.submit(request):
            return
        self._prefetch_request = request
        self._prefetch_completed_request = None
        self._prefetch_result = None
        self._prefetch_category = self._lookahead_category(candidate)
        self._prefetch_bound_candidate_id = _object_id(candidate)
        self._prefetch_pose = candidate_pose.copy()
        self._prefetch_scene_epoch = scene_epoch
        self._prefetch_observed_at = self._timestamp_without_advancing(observation)
        self._prefetch_phase = "pregrasp"

    def _admit_prefetch_target_plan(
        self,
        selected: Any,
        result: PlanningResult,
        request: PlanningRequest,
        observation: Mapping[str, Any],
        now: float,
    ) -> np.ndarray | None:
        """Atomically bind the live visual reservation to B's pose clock."""

        planned_time = request.planned_target_time_s
        if planned_time is None:
            # Legacy/injected workers retain their old reservation behavior.
            return np.asarray(result.target_pose, dtype=float).copy()
        if not np.isfinite(planned_time) or planned_time < now - 1e-9:
            return None
        refiner = getattr(self.env, "refine_selected_grasp_plan", None)
        if not callable(refiner):
            refiner = getattr(self.env, "retime_selected_grasp_plan", None)
        if not callable(refiner):
            return None
        arguments = (
            _object_id(selected),
            max(0.0, planned_time - now),
            self.config.intercept_deadline_margin,
        )
        # This is an atomic reservation of B's already certified body clock,
        # not another "earliest feasible" retime.  The bundled environment's
        # fail-closed hook supports an explicit requested clock/pose; legacy
        # injected fixtures retain the positional contract used by unit tests.
        supports_exact_plan = False
        try:
            signature = inspect.signature(refiner)
            signature.bind(
                *arguments,
                requested_time=planned_time,
                requested_pose=np.asarray(result.target_pose, dtype=float),
            )
            supports_exact_plan = True
        except (TypeError, ValueError):
            supports_exact_plan = False
        refined = (
            refiner(
                *arguments,
                requested_time=planned_time,
                requested_pose=np.asarray(result.target_pose, dtype=float),
            )
            if supports_exact_plan
            else refiner(*arguments)
        )
        if not isinstance(refined, Mapping) or not bool(
            refined.get("accepted", False)
        ):
            return None
        try:
            admitted_time = float(refined["estimated_grasp_time"])
            admitted_pose = np.asarray(
                refined["predicted_grasp_pose"], dtype=float
            ).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            admitted_pose.shape != (7,)
            or not np.all(np.isfinite(admitted_pose))
            or not np.isfinite(admitted_time)
            or abs(admitted_time - planned_time) > 1e-6
            # The hook returns the object/body pose; B's target is the high
            # end-effector pregrasp.  Their belt-plane coordinates must be the
            # same, while z intentionally differs by object grasp height and
            # approach clearance.
            or np.linalg.norm(admitted_pose[:2] - result.target_pose[:2])
            > self.config.trajectory_replan_position_tolerance + 1e-9
        ):
            return None
        return admitted_pose.copy()

    def _commit_prefetch(
        self,
        candidate: Any,
        candidate_pose: np.ndarray,
        observation: Mapping[str, Any],
        now: float,
        *,
        acquisition_ready: bool,
    ) -> bool:
        """Atomically reserve/install B after every future-anchor proof holds."""

        result = self._prefetch_result
        request = self._prefetch_request
        # Polling clears the active pointer; retain the immutable completed B
        # request separately so no later candidate can inherit its result.
        if request is None:
            request = getattr(self, "_prefetch_completed_request", None)
        anchor = self._prefetch_anchor_q
        if (
            result is None
            or request is None
            or anchor is None
            or not self._prefetch_release_complete
            or not self._prefetch_retreat_installed
            or not acquisition_ready
            or not self._release_is_physically_clear(observation)
            or request.scene_epoch != int(self._observed_scene_epoch or 0)
            or not self._same_target_id(_object_id(candidate), request.target_id)
            or self._lookahead_category(candidate) != self._prefetch_category
            or not self._window_valid(candidate, observation)
            or not self._prefetch_pose_is_compatible(
                request, candidate, candidate_pose, observation
            )
            or not self._prefetch_result_is_complete(result, request)
            or not self._prefetch_deadline_valid(
                result, candidate, observation, request
            )
        ):
            return False
        current = self._joint_positions(observation)
        if (
            np.max(np.abs(current - anchor))
            > self.config.planner_corner_settle_tolerance
            or not self._joint_velocity_is_settled(
                observation, self.config.planner_waypoint_velocity_tolerance
            )
        ):
            return False
        rebased = self._rebase_return_lookahead_result(result, request, observation)
        if rebased is None or rebased.path is None:
            return False
        first = rebased.path[1] if len(rebased.path) > 1 else rebased.path[0]
        if not self._global_edge_is_safe(
            current,
            first,
            minimum_tcp_height=request.minimum_tcp_height,
            minimum_joint_margin=request.minimum_joint_margin,
            minimum_downward_alignment=(
                self.config.acquisition_minimum_downward_alignment
            ),
        ):
            return False

        selected = self._select_target(observation)
        if (
            selected is None
            or not self._same_target_id(_object_id(selected), request.target_id)
            or self._lookahead_category(selected) != self._prefetch_category
        ):
            if selected is not None:
                _hook(
                    self.env,
                    ("release_selected_target", "release_target_reservation"),
                    ((_object_id(selected),),),
                )
            return False

        admitted_pose = self._admit_prefetch_target_plan(
            selected, result, request, observation, now
        )
        if admitted_pose is None:
            _hook(
                self.env,
                ("release_selected_target", "release_target_reservation"),
                ((_object_id(selected),),),
            )
            return False

        latest_pose = candidate_pose.copy()
        route_pose = result.target_pose.copy()
        committed_pose = (
            route_pose.copy()
            if request.planned_target_time_s is not None
            else latest_pose.copy()
        )
        committed_pose[3:] = route_pose[3:]
        self._clear_target()
        self._target = selected
        self._target_id = _object_id(selected)
        self._grasp_gripper_action = self._grip_action(selected)
        self._attempt_started_at = now
        self._last_seen_target_at = now
        self._grasp_pose = committed_pose.copy()
        self._grasp_pose[2] -= self.config.approach_height
        self._selected_grasp_quaternion = route_pose[3:].copy()
        self._transition(
            PolicyStage.PREGRASP,
            now,
            reason="prefetch_retreat_ready",
            event="target_selected",
        )
        if not self._install_planning_result(
            rebased,
            request,
            committed_pose,
            observation,
            first_edge_already_validated=True,
        ):
            _hook(
                self.env,
                ("release_selected_target", "release_target_reservation"),
                ((self._target_id,),),
            )
            self._clear_target()
            self._transition(
                PolicyStage.RECOVERING,
                now,
                reason="prefetch_first_edge_stale",
            )
            self._discard_prefetch(cancel_worker=False)
            return False
        self._safe_path_generation = self._planning_generation
        self._safe_path_target_id = self._target_id
        self._safe_path_stage = PolicyStage.PREGRASP

        accepted, pose_changed = self._refine_pregrasp_intercept(
            committed_pose, observation
        )
        # ``result.target_pose`` may intentionally be a later adaptive-yaw
        # alternative from this request.  Its complete high path and low
        # continuation were certified together, so comparing it to the
        # candidate's preferred quaternion would discard the very fallback
        # the worker proved.  The policy adopts that route quaternion above;
        # only an unplanned Cartesian endpoint displacement is stale.
        endpoint_changed = bool(
            (
                request.planned_target_time_s is None
                and np.linalg.norm(latest_pose[:3] - route_pose[:3])
                > self.config.trajectory_replan_position_tolerance
            )
            or (
                request.planned_target_time_s is not None
                and np.linalg.norm(admitted_pose[:2] - route_pose[:2])
                > self.config.trajectory_replan_position_tolerance
            )
        )
        if not accepted:
            self._terminal_motion_plan_reason = (
                self._last_error or "insufficient_intercept_time"
            )
            self._invalidate_motion_plan()
            self._record_motion_plan_failure()
        elif pose_changed or endpoint_changed:
            # Reservation freezes the newest visual endpoint, but the route
            # certified for the older moving pose is never executed.
            self._invalidate_motion_plan()
        self._discard_prefetch(cancel_worker=False)
        return True

    def _update_prefetch(
        self,
        observation: Mapping[str, Any],
        now: float,
        *,
        acquisition_ready: bool = False,
    ) -> bool:
        """Advance independent A/B work; return True only on atomic B commit."""

        if self._ensure_prefetch_worker() is None:
            return False
        self._poll_prefetch()
        if self._prefetch_terminal_b_target_id is not None:
            # No certified B path was found within its immutable physical
            # budget.  The same anchor and an even later clock cannot improve
            # that result.  Release A/B immediately so the primary RETURNING
            # route and ordinary lookahead can move to a different start state;
            # never wait until only the descent-only minimum slack remains.
            self._discard_prefetch(cancel_worker=False)
            return False
        if self._stage == PolicyStage.TRANSFERRING:
            had_carry = bool(
                self._carry_prefetch_request is not None
                or self._carry_prefetch_result is not None
            )
            carry_installed = (
                self._install_carry_prefetch(observation)
                if self._carry_prefetch_result is not None
                else False
            )
            if had_carry and not carry_installed:
                # C had the entire grasp/descent interval to finish.  At the
                # transfer boundary a pending, stale or unsafe C must not own
                # the only speculative worker or delay the ordinary primary
                # fallback.  Any A derived from an uncommitted C endpoint is
                # invalidated with it.
                self._discard_prefetch(cancel_worker=True)
            if not self.config.dynamic_intercept_tracking:
                self._capture_prefetch_place_anchor(observation)
        if self.config.dynamic_intercept_tracking:
            # Preserve certified carry prefetch C above. Ordinary RETURNING
            # uses the original guarded path; A/B must not own an idle anchor.
            return False
        candidate, candidate_pose = self._track_prefetch_candidate(observation)

        missing_within_grace = bool(
            candidate is None
            and self._prefetch_bound_candidate_id is not None
            and self._prefetch_candidate_missing_since is not None
            and self._timestamp_without_advancing(observation)
            - self._prefetch_candidate_missing_since
            <= self.config.target_lost_timeout + 1e-9
        )

        request = self._prefetch_request
        completed_request = bool(
            request is None and self._prefetch_result is not None
        )
        if completed_request:
            # Polling deliberately clears the active pointer.  Continue to
            # version-check the immutable completed request while its result
            # waits for the arm to reach A's anchor; otherwise a moving RGB-D
            # endpoint can drift stale and permanently block a newer B.
            request = self._prefetch_completed_request
        if (
            request is not None
            and not missing_within_grace
            and candidate is not None
            and candidate_pose is not None
            and (
                not self._same_target_id(
                    request.target_id, _object_id(candidate)
                )
                or request.scene_epoch
                != int(self._observed_scene_epoch or 0)
            )
        ):
            # An A anchor is reusable only while B still names the same
            # physical target and scene.  Treat an identity/version mismatch
            # as lineage loss, not ordinary moving-pose drift: otherwise the
            # cleared B fields leave a bare anchor owning RETURNING with no
            # request that can ever commit.
            self._discard_prefetch(cancel_worker=True)
            return False
        if request is not None and (
            (
                (candidate is None or candidate_pose is None)
                and not missing_within_grace
            )
            or (
                not missing_within_grace
                and (
                    candidate is None
                    or candidate_pose is None
                    or not self._same_target_id(
                        request.target_id, _object_id(candidate)
                    )
                    or request.scene_epoch
                    != int(self._observed_scene_epoch or 0)
                    # A visual update may move by 4--12 mm at one 15 Hz
                    # camera tick.  Let a running immutable B finish, then
                    # version-check it once; no stale path is executed.
                    or (
                        completed_request
                        and not self._prefetch_pose_is_compatible(
                            request,
                            candidate,
                            candidate_pose,
                            observation,
                        )
                    )
                )
            )
        ):
            self._discard_prefetch(cancel_worker=True, preserve_retreat=True)
            request = None

        if candidate is not None and candidate_pose is not None:
            self._submit_prefetch_pregrasp(candidate, candidate_pose, observation)

        if (
            self._stage == PolicyStage.RETURNING
            and self._prefetch_release_complete
            and not self._prefetch_retreat_installed
        ):
            self._install_prefetch_retreat(observation)

        if (
            self._stage == PolicyStage.RETURNING
            and candidate is not None
            and candidate_pose is not None
            and self._prefetch_result is not None
        ):
            return self._commit_prefetch(
                candidate,
                candidate_pose,
                observation,
                now,
                acquisition_ready=acquisition_ready,
            )
        return False

    def _install_planning_result(
        self,
        result: PlanningResult,
        request: PlanningRequest,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        first_edge_already_validated: bool = False,
    ) -> bool:
        """Commit a live-safe route, retaining an identically certified edge.

        A changed target never permits a nonzero-speed splice onto a different
        edge. If a new complete certificate includes the exact edge already
        being executed, preserve that edge's original polynomial and clock;
        rebuilding it would unnecessarily restart the commanded velocity.
        """

        previous_trajectory = self._safe_edge_trajectory
        previous_started_at = self._safe_edge_started_at
        previous_edge_may_continue = bool(
            self.config.dynamic_intercept_tracking
            and self._stage == PolicyStage.PREGRASP
            and self._safe_path_stage == self._stage
            and self._safe_path_generation == request.generation
            and self._same_target_id(self._safe_path_target_id, request.target_id)
            and self._safe_path_scene_epoch == request.scene_epoch
            and self._safe_path_worker_validated
            and previous_trajectory is not None
            and previous_started_at is not None
            and not self._safe_edge_handoff_pending
            and self._safe_path_feedback_connector_goal_index is None
        )
        previous_passthrough = self._safe_edge_passthrough_eligible
        previous_retime_active = self._safe_edge_dynamic_retime_active

        if not result.success or result.path is None or result.cost is None:
            self._last_error = result.error or "background planner returned no path"
            return False
        path = [np.asarray(item, dtype=float).copy() for item in result.path]
        if not path:
            self._last_error = "background planner returned an empty path"
            return False
        current = self._joint_positions(observation)
        if (
            np.max(np.abs(current - result.start_q))
            > self.config.planner_waypoint_tolerance
        ):
            self._last_error = "stale global path start state"
            return False
        if not (
            result.path_fully_validated
            and self._planning_certificate_contract_required()
        ):
            path = [state.copy() for state in self._corner_path(path)]
        feedback_connector_goal_index = (
            1
            if (
                len(path) > 1
                and np.max(np.abs(path[1] - request.start_q)) <= 1e-9
                and np.max(np.abs(path[0] - request.start_q)) > 1e-9
            )
            else None
        )
        worker_path_start = path[0].copy()
        # Rebase the geometric route on the latest measured state accepted by
        # the stale-start tolerance.  Deadline admission must use the route we
        # will actually execute, not the worker's older snapshot.
        path[0] = current.copy()
        index = 1 if len(path) > 1 else 0
        worker_durations = result.edge_durations
        has_worker_profile = (
            worker_durations is not None
            and len(worker_durations) == max(0, len(path) - 1)
            and all(
                np.isfinite(duration) and duration >= 0.0
                for duration in worker_durations
            )
        )
        if (
            not has_worker_profile
            and self._planning_certificate_contract_required()
        ):
            # Production results must be complete.  Rebuilding a missing
            # timing profile here performs 33 Jacobian evaluations per edge
            # and violates the control-thread latency contract.  Fail closed
            # and let the versioned worker retry instead.
            self._last_error = "background path is missing its timing profile"
            return False
        if (
            has_worker_profile
            and self._planning_certificate_contract_required()
            and not result.path_fully_validated
        ):
            self._last_error = "background path is missing its safety certificate"
            return False
        if (
            request.stage == PolicyStage.PREGRASP.value
            and self._planning_certificate_contract_required()
            and (
                request.continuation_target_poses is None
                or not result.continuation_fully_validated
            )
        ):
            self._last_error = (
                "background PREGRASP path is missing its continuation contract"
            )
            return False

        # Static measured-scene adapters can preserve the worker's complete
        # immutable certificate. A live model/scene adapter must explicitly
        # prove that non-arm geometry and constraints still match; a bare
        # path_fully_validated flag is not enough. Keep the exact original
        # edge when feedback drifts, adding only a short certified connector.
        reuse_hook = getattr(self.env, "can_reuse_planning_certificate", None)
        reused_certificate = bool(
            self._planning_certificate_contract_required()
            and result.path_fully_validated
            and has_worker_profile
            and callable(reuse_hook)
            and self._result_matches_request(result, request)
            and self._request_matches_goal(request, target_pose)
            and np.array_equal(result.start_q, request.start_q)
            and np.array_equal(worker_path_start, request.start_q)
            and reuse_hook(request, result, observation) is True
        )
        if reused_certificate:
            path[0] = worker_path_start.copy()
            if not np.array_equal(current, worker_path_start):
                if not self._global_edge_is_safe(
                    current, worker_path_start,
                    minimum_tcp_height=request.minimum_tcp_height,
                    minimum_joint_margin=request.minimum_joint_margin,
                    minimum_downward_alignment=request.minimum_downward_alignment,
                ):
                    self._last_error = "delivered path start connector is unsafe"
                    return False
                connector_duration = self._edge_execution_duration(
                    current, worker_path_start, stage=request.stage,
                )
                path.insert(0, current.copy())
                worker_durations = (connector_duration, *worker_durations)
                index = 1
                feedback_connector_goal_index = 1
            first_edge_already_validated = True

        first_command = path[index]
        if (
            not first_edge_already_validated
            and not self._global_edge_is_safe(
                current,
                first_command,
                minimum_tcp_height=request.minimum_tcp_height,
                minimum_joint_margin=request.minimum_joint_margin,
                minimum_downward_alignment=request.minimum_downward_alignment,
            )
        ):
            self._last_error = "delivered global path first edge is no longer safe"
            return False

        if has_worker_profile:
            # The worker already sampled every unchanged edge's angular
            # Jacobian and built its stop-to-stop timing profile.  Only the
            # first edge has been rebased onto latest measured q, so only that
            # edge may be recomputed on the control thread.
            edge_durations = [float(item) for item in worker_durations]
            if edge_durations:
                rebased_delta = path[0] - result.start_q
                if reused_certificate or np.max(np.abs(rebased_delta)) <= 1e-9:
                    # Exact current-start delivery or an explicitly inserted
                    # connector preserves the worker's original timing.
                    # Exact current-start delivery is the overwhelmingly
                    # common case.  The worker timing is still exact, while
                    # the live first edge above has already been revalidated.
                    pass
                else:
                    edge_durations[0] = self._realtime_rebased_edge_duration(
                        result.start_q,
                        path[0],
                        path[1],
                        edge_durations[0],
                        stage=request.stage,
                    )
            rebased_cost = (
                float(sum(edge_durations)),
                float(result.cost[1]),
                len(path),
            )
            worker_validated = bool(
                result.path_fully_validated
                and self._planning_certificate_contract_required()
            )
        else:
            # Backward-compatible path for simple test/third-party workers.
            # The production worker always supplies its immutable profile.
            rebased_cost, edge_durations = (
                self._joint_path_execution_profile(path)
            )
            worker_validated = False
        continued_edge_index = None
        if previous_edge_may_continue and reused_certificate:
            assert previous_trajectory is not None
            for candidate_index in range(1, len(path)):
                expected = self._build_global_edge_trajectory(
                    path[candidate_index-1], path[candidate_index],
                    *self._trajectory_limits(),
                    minimum_duration=edge_durations[candidate_index-1], stage=request.stage)
                same_time_law = (
                    getattr(previous_trajectory, 'backend', 'quintic')
                        == getattr(expected, 'backend', 'quintic')
                    and np.array_equal(getattr(previous_trajectory, 'jerk_limits', None),
                                       getattr(expected, 'jerk_limits', None)))
                if (np.array_equal(path[candidate_index-1], previous_trajectory.q0)
                        and np.array_equal(path[candidate_index], previous_trajectory.q1)
                        and abs(edge_durations[candidate_index-1]-previous_trajectory.duration) <= 1e-12
                        and np.array_equal(previous_trajectory.velocity_limits, self._trajectory_limits()[0])
                        and np.array_equal(previous_trajectory.acceleration_limits, self._trajectory_limits()[1])
                        and same_time_law):
                    continued_edge_index = candidate_index
                    break
        self._safe_joint_path = path
        self._safe_joint_path_index = index
        target_pose[:] = result.target_pose
        self._safe_path_target_pose = result.target_pose.copy()
        self._safe_path_stage = self._stage
        self._safe_path_planner = None
        self._safe_path_eta_s = float(rebased_cost[0])
        self._safe_path_edge_durations = edge_durations
        self._safe_path_worker_validated = worker_validated
        self._safe_path_continuation_validated = bool(
            result.continuation_fully_validated
            and request.continuation_target_poses is not None
        )
        # Generic installs are primary/look-ahead routes.  The carry caller
        # marks C provenance only after this atomic install succeeds.
        self._safe_path_from_carry_prefetch = False
        self._safe_path_minimum_tcp_height = request.minimum_tcp_height
        self._safe_path_minimum_joint_margin = request.minimum_joint_margin
        self._safe_path_minimum_downward_alignment = (
            request.minimum_downward_alignment
        )
        self._safe_path_generation = request.generation
        self._safe_path_target_id = request.target_id
        self._safe_path_scene_epoch = request.scene_epoch
        self._safe_edge_trajectory = None
        self._safe_edge_started_at = None
        self._safe_edge_goal_index = None
        self._safe_path_feedback_connector_goal_index = (
            feedback_connector_goal_index
        )
        self._safe_edge_passthrough_eligible = False
        self._safe_edge_dynamic_retime_active = False
        self._safe_edge_handoff_pending = False
        if continued_edge_index is not None:
            # No time-parameterizer or controller history is reset: the next
            # sample comes from the same curve at the same absolute clock.
            # The executor still checks latest feedback -> next command.
            self._safe_joint_path_index = continued_edge_index
            self._safe_edge_goal_index = continued_edge_index
            self._safe_edge_trajectory = previous_trajectory
            self._safe_edge_started_at = previous_started_at
            self._safe_edge_passthrough_eligible = previous_passthrough
            self._safe_edge_dynamic_retime_active = previous_retime_active
            self._safe_path_feedback_connector_goal_index = None
        self._trajectory_reuse_diagnostics.update(
            active_edge_clock_preserved=continued_edge_index is not None,
            active_edge_index=continued_edge_index,
            reason=("identical_certified_edge_and_profile" if continued_edge_index is not None
                    else "replacement_edge_uses_original_settled_handoff"),
        )
        if self._stage == PolicyStage.PREGRASP:
            self._pregrasp_last_planning_time_s = max(0.0, float(result.planning_time_s))
            self._fixed_high_request = None
            self._fixed_high_rejected = False
            self._fixed_high_ready = False
            self._fixed_high_validated_low = (
                request.continuation_target_poses[0].copy()
                if result.continuation_fully_validated and request.continuation_target_poses
                and len(request.target_poses) == 1 else None)
            self._fixed_high_validated_capture_time = (
                request.planned_target_time_s if self._fixed_high_validated_low is not None else None)
            self._accept_initial_descent(result, request, observation)
            self._selected_grasp_quaternion = result.target_pose[3:].copy()
            if self._grasp_pose is not None:
                self._grasp_pose[3:] = result.target_pose[3:]
            if self._safe_path_continuation_validated:
                self._pregrasp_branch_q = path[-1].copy()
                self._pregrasp_branch_target_id = request.target_id
                self._pregrasp_branch_scene_epoch = request.scene_epoch
                self._pregrasp_previous_route = {
                    "target_id": request.target_id,
                    "scene_epoch": request.scene_epoch,
                    "target_pose": result.target_pose.copy(),
                    "path": tuple(state.copy() for state in path),
                }
        self._last_error = None
        return True

    def _adopt_pregrasp_effective_request(
        self, result: PlanningResult, request: PlanningRequest,
        target_pose: np.ndarray, observation: Mapping[str, Any],
    ) -> PlanningRequest | None:
        """Admit a worker's forecast pose and visual clock as one reservation."""
        effective = self._completed_planning_request(result, request)
        if effective is None:
            return None
        if result.effective_request is None:
            return effective
        refiner = getattr(self.env, "refine_selected_grasp_plan", None)
        if not callable(refiner):
            return None
        capture_pose = result.target_pose.copy()
        capture_pose[2] -= self.config.approach_height
        if not self._pregrasp_capture_high_active:
            capture_pose[:3] += np.asarray(effective.intercept_velocity_m_s) * self.config.descent_lead_time
        admitted = refiner(self._target_id, 0.0, self.config.intercept_deadline_margin,
            requested_time=effective.planned_target_time_s,
            requested_pose=capture_pose)
        if not isinstance(admitted, Mapping) or not admitted.get("accepted", False):
            self._last_error = str(_value(admitted, "reason", "insufficient_intercept_time"))
            return None
        self._pregrasp_capture_time_s = float(effective.planned_target_time_s)
        target_pose[:] = result.target_pose
        if self._grasp_pose is not None:
            self._grasp_pose = capture_pose.copy()
        return effective

    def _request_async_global_path(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        minimum_joint_margin: float | None = None,
        minimum_tcp_height: float | None = None,
        minimum_downward_alignment: float | None = None,
    ) -> str:
        """Poll/submit without waiting: ``ready``, ``pending`` or ``failed``."""

        worker = self._ensure_planning_worker()
        if worker is None:
            return "unavailable"
        request = self._planning_request
        if request is not None and not self._request_matches_goal(
            request, target_pose
        ):
            if self._pending_pregrasp_request_can_finish(request, target_pose):
                # Record the first revision per request, not every servo tick.
                if self._planning_pending_goal_update.get("request_id") != request.request_id:
                    self._planning_pending_goal_update = {
                        "request_id": request.request_id,
                        "generation": request.generation,
                        "timestamp": self._timestamp_without_advancing(observation),
                        "pending_high_pose": request.target_pose.copy(),
                        "updated_high_pose": target_pose.copy(),
                        "displacement_m": float(np.linalg.norm(
                            request.target_pose[:3] - target_pose[:3])),
                        "decision": "finish_immutable_request_then_refine_live_capture",
                    }
                    self._emit("planning_pending_goal_update",
                        self._timestamp_without_advancing(observation),
                        f"request {request.request_id}: finish immutable forecast")
            else:
                self._cancel_async_plan()
                request = None

        # Consume at most two completions per 20 ms tick.  Version checks make
        # it safe to leave stale results for the next tick, while this bound
        # makes polling cost independent of prior cancellation churn.
        for _ in range(2):
            result = worker.poll()
            if result is None:
                break
            request = self._planning_request
            if (
                request is None
                or not self._request_has_current_owner(request)
                or self._completed_planning_request(result, request) is None
            ):
                self._planning_stale_results += 1
                continue
            # This request has delivered its only completion. Even a rejected
            # receipt must release it, otherwise no worker can wake pending.
            self._planning_request = None
            if result.cancelled:
                self._planning_stale_results += 1
                request = None
                continue
            if not result.success:
                if self._stage == PolicyStage.PREGRASP:
                    self._pregrasp_last_planning_time_s = max(0., float(result.planning_time_s))
                if (
                    result.error == "global planner found no safe IK goal"
                    and self._activate_capture_high_fallback(target_pose, observation)
                ):
                    # Keep the same capture reservation and orientation, but
                    # retry its high approach from a different Cartesian start.
                    # Its new descent continuation is certified from scratch.
                    request = None
                    break
                if (
                    result.error in {"global planner found no safe IK goal",
                                     "constrained RRT-Connect could not find a safe path"}
                    and self._advance_dynamic_intercept(target_pose, observation)
                ):
                    # The moving target can enter another reachable part of
                    # the workspace even while the arm holds the same q.
                    # Reserve a later visual clock/pose together, then ask the
                    # same strict planner to certify it. Never execute a
                    # failed goal or suppress the part merely because the
                    # earlier geometric estimate had no safe IK branch.
                    request = None
                    break
                self._last_error = result.error or "background planner returned no path"
                return "failed"
            # Only a successful current result may change the reservation.
            # In particular, a cancelled/failed forecast is not an accepted
            # new capture position merely because its IK estimate moved it.
            request = self._adopt_pregrasp_effective_request(
                result, request, target_pose, observation)
            if request is None:
                self._planning_stale_results += 1
                continue
            # A completed immutable request owns this certified endpoint.
            # The caller refines against live vision and ETA before emitting
            # motion, using the existing certified-prefix/continuation path.
            target_pose[:] = result.target_pose
            if self._install_planning_result(
                result, request, target_pose, observation
            ):
                return "ready"
            # The robot or scene moved while planning.  This is a stale plan,
            # not an IK failure: hold and immediately request from fresh q.
            self._planning_stale_results += 1
            self._planning_generation += 1
            request = None

        if self._planning_request is not None:
            return "pending"
        request = self._make_planning_request(
            target_pose,
            observation,
            minimum_joint_margin=minimum_joint_margin,
            minimum_tcp_height=minimum_tcp_height,
            minimum_downward_alignment=minimum_downward_alignment,
        )
        if request is None:
            return "failed"
        if not worker.submit(request):
            # A cooperatively-cancelled request may still be unwinding.  Never
            # queue repeatedly or block waiting for it.
            return "pending"
        self._planning_request = request
        return "pending"

    def _activate_capture_high_fallback(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> bool:
        """Retry an unreachable upstream high pose over the same capture."""

        if not (
            self.config.pregrasp_capture_high_fallback_enabled
            and self.config.dynamic_intercept_tracking
            and self._stage == PolicyStage.PREGRASP
            and not self._pregrasp_capture_high_active
            and self._grasp_pose is not None
        ):
            return False
        before = target_pose.copy()
        now = self._timestamp_without_advancing(observation)
        if (self._pregrasp_capture_time_s is not None
                and self._pregrasp_capture_time_s <= now + self.config.descent_lead_time):
            # A failed computation consumed real time. The alternate high
            # point must belong to a fresh visual capture, not a past booking.
            if not self._advance_dynamic_intercept(target_pose, observation):
                return False
        alternate = self._grasp_pose.copy()
        alternate[2] += self.config.approach_height
        if np.linalg.norm(alternate[:3] - before[:3]) <= 1e-9:
            return False
        self._pregrasp_capture_high_active = True
        target_pose[:] = alternate
        self._pregrasp_capture_high_diagnostics = {
            "active": True,
            "trigger": "global planner found no safe IK goal",
            "timestamp_s": self._timestamp_without_advancing(observation),
            "rejected_high_pose": before,
            "alternate_high_pose": alternate.copy(),
            "reserved_capture_pose": self._grasp_pose.copy(),
            "reserved_capture_time_s": self._pregrasp_capture_time_s,
            "object_ground_truth_used": False,
        }
        return True

    def _advance_dynamic_intercept(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> bool:
        """Try the next future intercept after a fully checked IK rejection.

        Only policy-visible visual state and estimated belt velocity are used. Each
        option carries its own absolute time; it is never a spatial offset
        applied while keeping the old capture clock.
        """
        if not self.config.dynamic_intercept_tracking or self._stage != PolicyStage.PREGRASP:
            return False
        refiner = getattr(self.env, "refine_selected_grasp_plan", None)
        if not callable(refiner) or self._grasp_pose is None:
            return False
        cfg = _value(getattr(self.env, "scene_config", {}), "workspace", {})
        cfg = _value(cfg, "dynamic_intercept", {})
        bounds = self._belt_bounds("s_range", "x_range", default=(.02, .18))
        step = float(_value(cfg, "candidate_step_m", .04))
        prediction = self._prediction_from_mapping(observation, self._target_id)
        velocity = np.asarray(_value(prediction, "estimated_velocity",
            _value(self._target, "estimated_velocity", self._belt_velocity(observation))), dtype=float)
        now = self._timestamp_without_advancing(observation)
        try:
            old_time = float(self._pregrasp_capture_time_s
                if self._pregrasp_capture_time_s is not None
                else _value(self._target, "estimated_grasp_time"))
        except (TypeError, ValueError):
            return False
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)) or step <= 0:
            return False
        speed = float(velocity @ self._belt_frame()[1])
        if abs(speed) <= 1e-5:
            return False
        # Bounds describe the capture corridor, not its raised PREGRASP
        # waypoint. The latter's vertical offset projects onto an inclined
        # belt and must not consume available longitudinal travel.
        reference_pose = self._grasp_pose.copy()
        reference_time = old_time
        earliest = now + self.config.descent_lead_time + self._execution_dt(observation)
        if old_time < earliest:
            current = _value(prediction, "current_estimated_pose")
            if current is not None:
                reference_pose[:3] = np.asarray(current, dtype=float)[:3]
            else:
                reference_pose[:3] += velocity * (now - old_time)
            # This is a prediction from the real observation, never a fresh
            # observation timestamp. Include the last measured search cost.
            reference_time = earliest + self._pregrasp_last_planning_time_s
            reference_pose[:3] += velocity * (reference_time - now)
        coordinate = self._belt_coordinate(reference_pose[:3])
        remaining = float(bounds[1] - coordinate if speed > 0. else coordinate - bounds[0])
        if remaining <= 1e-4:
            return False
        extra_time = min(step, remaining) / abs(speed)
        proposed_time = reference_time + extra_time
        proposed_pose = reference_pose.copy()
        proposed_pose[:3] += velocity * extra_time
        result = refiner(
            self._target_id, max(0., proposed_time - now),
            self.config.intercept_deadline_margin,
            requested_time=proposed_time, requested_pose=proposed_pose,
        )
        if not isinstance(result, Mapping) or not result.get("accepted", False):
            self._emit("intercept_retry_rejected", now,
                str(_value(result, "reason", "invalid_intercept_reply")))
            return False
        self._pregrasp_capture_time_s = float(result.get("estimated_grasp_time", proposed_time))
        refined = np.asarray(result["predicted_grasp_pose"], dtype=float)
        self._grasp_pose[:3] = self._retimed_grasp_position(self._grasp_pose[:3], refined[:3])
        target_pose[:] = self._pregrasp_high_pose(self._grasp_pose, observation)
        self._pregrasp_timing_refinements = 0
        self._emit("intercept_retry_updated", now,
            f"capture_time_s={old_time:.6f}->{self._pregrasp_capture_time_s:.6f}")
        return True

    def _global_edge_is_safe(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        minimum_tcp_height: float,
        minimum_joint_margin: float,
        minimum_downward_alignment: float | None = None,
    ) -> bool:
        """Revalidate a delivered edge exclusively against current live state."""

        # The production MuJoCo environment evaluates every constraint from
        # one forwarded state and can reuse its already-forwarded live data
        # for the measured start.  Besides avoiding duplicate ``mj_forward``
        # calls for Jacobian and collision checks, keeping these checks in one
        # validator prevents diagnostics from accidentally referring to a
        # different sampled state.  Lightweight environments retain the
        # explicit per-state fallback below.
        edge_validator = getattr(self.env, "is_arm_edge_safe", None)
        if callable(edge_validator):
            try:
                return bool(
                    edge_validator(
                        start,
                        end,
                        normalized_resolution=(
                            self.config.planner_edge_resolution
                        ),
                        approach_axis_local=self._approach_axis_local,
                        minimum_downward_alignment=(
                            self._planner_downward_alignment_limit()
                            if minimum_downward_alignment is None
                            else float(minimum_downward_alignment)
                        ),
                        minimum_tcp_height=minimum_tcp_height,
                        minimum_joint_margin=minimum_joint_margin,
                        maximum_jacobian_condition=(
                            self.config.singularity_hard_stop_condition
                        ),
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                # Duck-typed third-party environments may expose a similarly
                # named hook with an older signature.  They still receive the
                # complete legacy validation path; production errors fail
                # closed in the environment and return ``False``.
                if callable(
                    getattr(self.env, "create_global_planning_worker", None)
                ):
                    return False

        limits = self._joint_safety_limits()
        if limits is None:
            return False
        lower, upper = limits
        span = np.maximum(upper - lower, 1e-6)
        normalized_distance = float(np.linalg.norm((end - start) / span))
        samples = max(
            1,
            int(
                math.ceil(
                    normalized_distance / self.config.planner_edge_resolution
                )
            ),
        )
        for index in range(samples + 1):
            state = start + (end - start) * (index / samples)
            if not self._global_path_state_valid(
                state,
                minimum_tcp_height=minimum_tcp_height,
                minimum_joint_margin=minimum_joint_margin,
                minimum_downward_alignment=minimum_downward_alignment,
            ):
                return False
        return True

    def _plan_global_joint_path(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        minimum_joint_margin: float | None = None,
    ) -> bool:
        limits = self._joint_safety_limits()
        if limits is None:
            self._last_error = "global planner needs finite joint limits"
            return False
        current = self._joint_positions(observation)
        current_pose = self._ee_pose(observation)
        minimum_height = min(
            float(current_pose[2]),
            float(target_pose[2]),
        ) - 0.012
        candidates = self._global_goal_candidates(
            target_pose,
            current,
            minimum_tcp_height=minimum_height,
            minimum_joint_margin=minimum_joint_margin,
        )
        if not candidates:
            self._last_error = "global planner found no safe IK goal"
            return False

        lower, upper = limits
        margin = (
            self.config.minimum_joint_limit_margin
            if minimum_joint_margin is None
            else max(0.0, float(minimum_joint_margin))
        )
        planner_config = SafePathConfig(
            normalized_step_size=self.config.planner_step_size,
            normalized_edge_resolution=self.config.planner_edge_resolution,
            execution_joint_step=self.config.planner_execution_joint_step,
            max_iterations=self.config.planner_max_iterations,
            shortcut_attempts=self.config.planner_shortcut_attempts,
        )
        seed = (
            1543
            + 17 * self._transfer_waypoint_index
            + 1009 * self._motion_plan_failure_count
        )
        try:
            seed += 53 * int(self._target_id or 0)
        except (TypeError, ValueError):
            pass
        best: tuple[
            tuple[float, float, int],
            list[np.ndarray],
            ConstrainedRRTConnect,
            list[float],
        ] | None = None
        for candidate_index, candidate in enumerate(candidates[:5]):
            planner = ConstrainedRRTConnect(
                lower + margin,
                upper - margin,
                lambda state: self._global_path_state_valid(
                    state,
                    minimum_tcp_height=minimum_height,
                    minimum_joint_margin=margin,
                ),
                config=planner_config,
                seed=seed + candidate_index,
            )
            path = planner.plan(current, candidate)
            if path is None:
                continue
            cost, edge_durations = self._joint_path_execution_profile(path)
            if best is None or cost < best[0]:
                best = (cost, path, planner, edge_durations)
        if best is not None:
            cost, path, planner, edge_durations = best
            self._safe_joint_path = [
                state.copy() for state in self._corner_path(path)
            ]
            self._safe_joint_path_index = (
                1 if len(self._safe_joint_path) > 1 else 0
            )
            self._safe_path_target_pose = target_pose.copy()
            self._safe_path_stage = self._stage
            self._safe_path_planner = planner
            self._safe_path_eta_s = float(cost[0])
            self._safe_path_minimum_tcp_height = minimum_height
            self._safe_path_minimum_joint_margin = margin
            self._safe_path_minimum_downward_alignment = (
                self._planner_downward_alignment_limit()
            )
            self._safe_path_generation = self._planning_generation
            self._safe_path_target_id = self._target_id
            self._safe_path_scene_epoch = int(
                self._observed_scene_epoch or 0
            )
            self._safe_path_edge_durations = edge_durations
            self._last_error = None
            return True
        self._last_error = "constrained RRT-Connect could not find a safe path"
        return False

    def _joint_path_execution_cost(
        self, path: Sequence[np.ndarray]
    ) -> tuple[float, float, int]:
        """Rank safe paths by an acceleration/velocity-limited ETA estimate."""

        return self._joint_path_execution_profile(path)[0]

    def _edge_execution_duration(
        self, start: np.ndarray, end: np.ndarray, *,
        stage: PolicyStage | str | None = None,
    ) -> float:
        """Return the exact duration used by one stop-to-stop executor edge."""

        velocity, acceleration = self._trajectory_limits()
        return float(
            self._build_global_edge_trajectory(
                start,
                end,
                velocity,
                acceleration,
                minimum_duration=self._edge_minimum_duration(
                    start, end, global_path=True, stage=stage),
                stage=stage,
            ).duration
        )

    def _global_joint_trajectory_backend(self, stage=None):
        selected_stage = self._safe_path_stage if stage is None else stage
        if selected_stage is None:
            selected_stage = self.stage
        # The existing angular-speed time certificate is specifically derived
        # for quintic smoothstep. Keep it unchanged outside unloaded dynamic
        # high motion, rather than reusing that proof for a new time law.
        eligible = (str(getattr(selected_stage, 'value', selected_stage)) == 'pregrasp'
                    and not self._global_path_tcp_speed_limits_enabled(selected_stage))
        return self.config.joint_trajectory_backend if eligible else 'quintic'

    def _build_global_edge_trajectory(self, start, goal, velocity, acceleration,
                                      *, minimum_duration=0., stage=None):
        if (self._contact_handoff_global_linear_speed(stage) is not None
                or self._contact_handoff_global_angular_speed(stage) is not None):
            minimum_duration=max(minimum_duration,self._edge_minimum_duration(
                start,goal,global_path=True,stage=stage))
        return build_certified_edge_trajectory(start, goal, velocity, acceleration,
            minimum_duration=minimum_duration,
            backend=self._global_joint_trajectory_backend(stage),
            jerk_limits=self.config.trajectory_jerk_limits)

    def _sample_safe_edge_reference(self, trajectory, elapsed, *, record=True):
        """Propose a reference on the certified chord before normal limits.

        A measured response model changes actuator targets, never measured
        feedback, the physical goal or the geometric collision certificate.
        The original retimer/controller and live measured-to-command checks
        remain responsible for accepting the proposal.
        """
        nominal, _ = trajectory.sample(elapsed)
        backend = getattr(trajectory, 'backend', 'quintic')
        diagnostic = {'applied': False, 'status': 'original_reference',
                      'trajectory_backend': backend,
                      'time_law_changed': backend != 'quintic',
                      'candidate_q_rad': np.asarray(nominal).tolist(),
                      'applied_reference_q_rad': np.asarray(nominal).tolist()}
        model = getattr(self.env, 'joint_response_model', None)
        eligible = (self.stage == PolicyStage.PREGRASP
                    and self.config.dynamic_intercept_tracking
                    and not self._global_path_tcp_speed_limits_enabled(PolicyStage.PREGRASP))
        if model is not None and eligible:
            try:
                proposed, details = model.preview_on_edge(trajectory, elapsed)
                diagnostic.update(details)
                diagnostic['candidate_q_rad'] = np.asarray(proposed).tolist()
                if details.get('applied', False):
                    nominal = np.asarray(proposed, dtype=float)
                    diagnostic['time_law_changed'] = True
                diagnostic['applied_reference_q_rad'] = np.asarray(nominal).tolist()
            except (TypeError, ValueError, AttributeError) as error:
                diagnostic.update(status='original_reference_model_unavailable', reason=str(error))
        if record:
            self._safe_edge_reference_diagnostics = diagnostic
        return np.asarray(nominal, dtype=float)

    def _realtime_rebased_edge_duration(
        self,
        expected_start: np.ndarray,
        actual_start: np.ndarray,
        end: np.ndarray,
        cached_duration: float,
        *, stage: PolicyStage | str | None = None,
    ) -> float:
        """Bound a small live start rebase without sampling 33 Jacobians.

        The background worker already time-parameterized the immutable edge.
        At delivery, or after settling near a later RRT corner, measured joint
        feedback can differ slightly from that certified start.  Repeating the
        full sampled-Jacobian calculation on the viewer thread caused 100--285
        ms spikes.  For a serial revolute arm every rotational Jacobian column
        is a unit joint axis, so the extra angular-speed coefficient caused by
        a start perturbation is conservatively bounded by its joint-space L1
        norm.  The ordinary quintic constructor below independently enforces
        every joint velocity and acceleration limit for the actual edge.
        """

        expected = np.asarray(expected_start, dtype=float).reshape(7)
        actual = np.asarray(actual_start, dtype=float).reshape(7)
        target = np.asarray(end, dtype=float).reshape(7)
        cached = float(cached_duration)
        if not np.isfinite(cached) or cached < 0.0:
            raise ValueError("cached edge duration must be finite and non-negative")
        start_error_l1 = float(np.sum(np.abs(actual - expected)))
        certified_delta_l1 = float(np.sum(np.abs(target - expected)))
        # Along a serial revolute chain, joint-axis i can rotate only with its
        # upstream joints.  Therefore changing the interpolation start by e
        # perturbs Jw(q)d by at most ||e||_1 * (1 + ||d||_1): the first term
        # covers the changed joint rates and the second covers rotated axes.
        angular_reserve = (
            1.05
            * (15.0 / 8.0)
            * start_error_l1
            * (1.0 + certified_delta_l1)
            / self.config.maximum_tcp_angular_speed
            if self._global_path_tcp_speed_limits_enabled(stage)
            else 0.0
        )
        velocity, acceleration = self._trajectory_limits()
        minimum = max(
            cached + angular_reserve,
            0.0
            if np.max(np.abs(target - actual)) <= 1e-12
            else self.config.trajectory_minimum_duration,
        )
        return float(
            self._build_global_edge_trajectory(
                actual,
                target,
                velocity,
                acceleration,
                minimum_duration=minimum,
                stage=stage,
            ).duration
        )

    def _joint_path_execution_profile(
        self, path: Sequence[np.ndarray]
    ) -> tuple[tuple[float, float, int], list[float]]:
        """Return path ranking cost plus reusable per-corner durations."""

        states = self._corner_path(path)
        if states.ndim != 2 or states.shape[1:] != (7,) or len(states) < 1:
            return (
                (float("inf"), float("inf"), 2**31 - 1),
                [],
            )
        if len(states) == 1:
            return ((0.0, 0.0, 1), [])
        deltas = np.diff(states, axis=0)
        # RRT waypoints form a polyline.  Treating all same-joint travel as one
        # trapezoid silently assumes that velocity can pass through every
        # corner, including reversals.  The executor has no corner-blending
        # contract, so use a finite-acceleration stop-to-stop quintic for each
        # edge and sum the durations.  This is conservative and executable;
        # a future forward/backward time-parameterizer may safely reduce it.
        execution_eta = 0.0
        edge_durations: list[float] = []
        for start, end in zip(states, states[1:]):
            duration = self._edge_execution_duration(start, end)
            execution_eta += duration
            edge_durations.append(duration)
        limits = self._joint_safety_limits()
        span = (
            np.ones(7, dtype=float)
            if limits is None
            else np.maximum(limits[1] - limits[0], 1e-6)
        )
        normalized_length = float(
            np.sum(np.linalg.norm(deltas / span, axis=1))
        )
        # ETA is the primary objective.  Length and waypoint count provide
        # deterministic tie-breaks and prefer straighter, lower-curvature
        # commands when two paths have the same idealized execution time.
        return (
            (
                float(execution_eta),
                normalized_length,
                len(states),
            ),
            edge_durations,
        )

    @staticmethod
    def _corner_path(path: Sequence[np.ndarray]) -> np.ndarray:
        """Remove execution samples that lie on the same joint-space chord.

        ``ConstrainedRRTConnect`` densifies an already validated shortcut path
        to cap the per-command joint step.  Those collinear samples are not
        physical stops, so charging ``trajectory_minimum_duration`` for every
        one would make ETA depend on the sampling resolution.  Keep endpoints,
        reversals and genuine corners; the original planner continues to
        validate and execute every dense waypoint.
        """

        states = np.asarray(path, dtype=float)
        if states.ndim != 2 or states.shape[1:] != (7,) or len(states) <= 2:
            return states
        kept = [states[0]]
        for index in range(1, len(states) - 1):
            incoming = states[index] - kept[-1]
            outgoing = states[index + 1] - states[index]
            in_norm = float(np.linalg.norm(incoming))
            out_norm = float(np.linalg.norm(outgoing))
            if in_norm <= 1e-12:
                continue
            if out_norm <= 1e-12:
                continue
            cosine = float(np.dot(incoming, outgoing) / (in_norm * out_norm))
            scale = max(in_norm, out_norm, 1.0)
            collinear = bool(
                cosine >= 1.0 - 1e-8
                and np.linalg.norm(
                    incoming / in_norm - outgoing / out_norm
                )
                <= 1e-7 * scale
            )
            if not collinear:
                kept.append(states[index])
        kept.append(states[-1])
        return np.asarray(kept, dtype=float)

    @staticmethod
    def _quintic_phase_for_progress(progress: float) -> float:
        """Invert the worker's normalized quintic without using elapsed time."""
        if progress <= 0.0:
            return 0.0
        if progress >= 1.0:
            return 1.0
        lower, upper = 0.0, 1.0
        for _ in range(36):
            middle = 0.5 * (lower + upper)
            value = 10.0 * middle**3 - 15.0 * middle**4 + 6.0 * middle**5
            if value < progress:
                lower = middle
            else:
                upper = middle
        return lower

    def _safe_path_settle_tolerance(self, goal_index: int | None) -> float:
        """Shared feedback admission for sender, path execution and ETA."""
        if (self.config.dynamic_intercept_tracking
                and self._stage == PolicyStage.PREGRASP
                and self._safe_path_stage == PolicyStage.PREGRASP
                and self._safe_joint_path
                and goal_index is not None
                and 0 <= goal_index < len(self._safe_joint_path)):
            tolerance = (self.config.pregrasp_endpoint_settle_tolerance
                         if goal_index == len(self._safe_joint_path) - 1
                         else self.config.pregrasp_corner_settle_tolerance)
            if tolerance is not None:
                return float(tolerance)
        return float(self.config.planner_corner_settle_tolerance)

    def _safe_path_remaining_motion_time(
        self, observation: Mapping[str, Any]
    ) -> float | None:
        """Estimate remaining traversal from command and measured progress.

        A delayed loop must not subtract wall time from a route it has not
        sent, nor treat an advanced command as an already reached position.
        Use the less advanced of the accepted target and measured joints on
        the same certified edge. Later edges retain their worker durations.
        With a dispatch clock, successive observations scale nominal time by
        measured phase advancement. This includes execution retiming and
        feedback lag, not an identified motor transfer function.
        """
        path = self._safe_joint_path
        try:
            measured = self._joint_positions(observation)
        except (KeyError, TypeError, ValueError):
            measured = None
        if (path and measured is not None and np.all(np.isfinite(measured))
                and self._safe_joint_path_index >= len(path) - 1
                and np.max(np.abs(measured-path[-1])) <= self._safe_path_settle_tolerance(len(path)-1)
                and self._joint_velocity_is_settled(
                    observation, self.config.planner_waypoint_velocity_tolerance)):
            return 0.0
        total = self._safe_path_eta_s
        if total is None or not np.isfinite(total):
            return None
        durations = self._safe_path_edge_durations
        index = self._safe_joint_path_index
        if (not path or not 0 < index < len(path)
                or len(durations) != len(path) - 1
                or not all(np.isfinite(value) and value >= 0.0 for value in durations)
                or self._safe_edge_trajectory is None
                or self._safe_edge_goal_index != index):
            return float(total * (self._pregrasp_motion_time_scale
                if isinstance(observation.get("dispatch_clock"), Mapping) else 1.0))
        raw = _value(observation, ("joint_target", "joint_command_target"), None)
        if raw is None:
            return float(total)
        try:
            accepted = np.asarray(raw, dtype=float).reshape(7)
        except (TypeError, ValueError):
            return float(total)
        if not np.all(np.isfinite(accepted)):
            return float(total)
        if (index < len(path) - 1 and measured is not None
                and np.max(np.abs(accepted-path[index])) <= 1e-8
                and np.max(np.abs(measured-path[index]))
                    <= self._safe_path_settle_tolerance(index)
                and self._joint_velocity_is_settled(
                    observation, self.config.planner_waypoint_velocity_tolerance)):
            # The sent edge has ended and feedback permits corner handoff.
            # Do not turn its stationary servo residual into an ever-growing
            # delay or move the intercept while the next edge is being installed.
            scale = (self._pregrasp_motion_time_scale
                     if isinstance(observation.get("dispatch_clock"), Mapping) else 1.0)
            return float(sum(durations[index:]) * scale)
        trajectory = self._safe_edge_trajectory
        start = np.asarray(trajectory.q0, dtype=float)
        delta = np.asarray(path[index], dtype=float) - start
        moving = np.abs(delta) > 1e-15
        # Exact certified commands give one identical scalar in every moving
        # joint.  If the ordinary limiter has profiled joints independently,
        # use the least advanced joint, not a projection that hides its lag.
        progress = (float(np.min((accepted[moving] - start[moving]) / delta[moving]))
                    if np.any(moving) else 1.0)
        if measured is not None and np.all(np.isfinite(measured)) and np.any(moving):
            progress = min(progress, float(np.min(
                (measured[moving]-start[moving])/delta[moving])))
        fraction = float(np.clip(progress, 0.0, 1.0))
        if getattr(trajectory, 'backend', 'quintic') == 'quintic':
            phase = self._quintic_phase_for_progress(fraction)
        else:
            # Invert the installed time law, not the old quintic assumption.
            lower, upper = 0., float(trajectory.duration)
            norm = float(delta @ delta)
            for _ in range(36):
                middle = .5 * (lower + upper)
                q, _ = trajectory.sample(middle)
                scalar = float((q-start) @ delta / norm) if norm > 1e-18 else 1.
                if scalar < fraction:
                    lower = middle
                else:
                    upper = middle
            phase = lower / trajectory.duration if trajectory.duration > 0 else 1.
        remaining = (1.0 - phase) * trajectory.duration + sum(durations[index:])
        if not isinstance(observation.get("dispatch_clock"), Mapping):
            return float(remaining)
        # Observe the slower of SENT target and measured joints, in nominal
        # trajectory seconds. Sampling twice at the same clock cannot create
        # a speed estimate. Reset the sample at every new certified edge.
        now = self._timestamp_without_advancing(observation)
        key = (id(path), id(trajectory), index)
        sample = self._pregrasp_progress_sample
        progress_s = phase * trajectory.duration
        if sample is None or sample[0] != key or now < sample[1]:
            self._pregrasp_progress_sample = (key, now, progress_s)
        else:
            elapsed, advanced = now - sample[1], progress_s - sample[2]
            if elapsed > 0.0 and advanced > 1e-9:
                self._pregrasp_motion_time_scale = max(1.0, elapsed / advanced)
            elif elapsed > 0.0:
                # A stationary measurement supplies no finite speed. Preserve
                # the previous estimate and include the observed lost time;
                # never manufacture a motion speed or a hard-stop threshold.
                return float(remaining * self._pregrasp_motion_time_scale + elapsed)
        return float(remaining * self._pregrasp_motion_time_scale)

    def _pregrasp_required_motion_time(
        self, observation: Mapping[str, Any]
    ) -> float | None:
        """Return a deadline-safe duration from now to nominal interception."""

        remaining = self._safe_path_remaining_motion_time(observation)
        if remaining is None:
            return None
        latency = _value(observation, "system_latency", {})
        if isinstance(latency, Mapping):
            planning = _value(latency, "planning", 0.0)
            control = _value(latency, "control", 0.0)
            response = _value(latency, "robot_response", 0.0)
        else:
            planning = control = response = 0.0
        components = [planning, control, response]
        parsed: list[float] = []
        for component in components:
            try:
                value = float(component)
            except (TypeError, ValueError):
                value = 0.0
            parsed.append(value if np.isfinite(value) and value > 0.0 else 0.0)
        return float(
            remaining
            + self.config.descent_lead_time
            + sum(parsed)
            + self._execution_dt(observation)
        )

    def _fixed_high_enabled(self, observation: Mapping[str, Any]) -> bool:
        return bool(self.config.pregrasp_fixed_high_enabled
            and self.config.dynamic_intercept_tracking and not self.config.prepare_high_only
            and isinstance(observation.get("dispatch_clock"), Mapping)
            and callable(getattr(self.env, "preview_fixed_high_grasp_plan", None)))

    def _refresh_fixed_high_intercept(self, observation: Mapping[str, Any]) -> bool:
        """Keep the high joint branch; asynchronously recertify changed descent.

        True means this anchor still owns the approach. False uses the original
        high replanner after an actual continuation failure, never a new stop.
        Every new low corridor uses the original worker's warm IK/dense edges.
        """
        if self._refresh_timed_descent(observation):
            return True
        if (not self._fixed_high_enabled(observation) or self._fixed_high_rejected
                or self._pregrasp_approach_pose is not None or not self._safe_joint_path):
            return False
        now = self._timestamp_without_advancing(observation)
        anchor = self._safe_joint_path[-1]
        if self._descent_high_hold_q is not None:
            anchor = self._descent_high_hold_q
        settled = (self._safe_joint_path_index >= len(self._safe_joint_path)-1
            and np.max(np.abs(self._joint_positions(observation)-anchor)) <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)
            and self._joint_velocity_is_settled(observation, self.config.planner_waypoint_velocity_tolerance))
        remaining = 0. if settled else self._descent_high_remaining(observation)
        high = self._safe_path_target_pose
        timed = self._timed_descent_enabled(observation)
        if (timed and self._descent_rebook_pending and settled
                and self._descent_high_hold_q is None and self._fixed_high_request is None):
            # The high route is over. Capture one measured start/hold, not
            # the old nominal waypoint or a moving sequence of feedbacks.
            # Fixed-high validation requires its target pose to match FK(q).
            solver = getattr(self.env, 'ik', None)
            if callable(getattr(solver, 'forward_pose', None)):
                self._descent_high_hold_q = self._joint_positions(observation).copy()
                anchor = self._descent_high_hold_q
                self._safe_edge_trajectory = None
                self._safe_edge_passthrough_eligible = False
        if self._descent_high_hold_q is not None:
            high = np.asarray(self.env.ik.forward_pose(anchor), dtype=float)
        duration = self._new_descent_duration() if timed else self.config.descent_lead_time
        proposal = self.env.preview_fixed_high_grasp_plan(
            self._target_id, high, remaining, duration,
            capture_high=self._pregrasp_capture_high_active)
        checked_time = self._fixed_high_validated_capture_time
        # A certificate is for a low pose AND its capture clock. Preserve that
        # clock on later ticks, not just on the tick which received the result.
        # While travelling, retain the full descent estimate. Once settled,
        # permit the same late-but-still-future capture accepted on delivery;
        # moving it to now+lead each tick would endlessly postpone descent.
        required = remaining + (0. if settled else duration)
        if checked_time is not None and now + required <= checked_time:
            checked_proposal = self.env.preview_fixed_high_grasp_plan(
                self._target_id, high, remaining, duration,
                capture_high=self._pregrasp_capture_high_active, capture_time=checked_time)
            if checked_proposal.get('accepted', False):
                proposal = checked_proposal
        self._fixed_high_ready = False
        diagnostics = {"timestamp": now, "mode": "fixed_high", "accepted": False,
            "actual_high_joint_state_ready": bool(settled), "remaining_high_motion_s": remaining,
            "high_pose": high.copy(), "reason": proposal.get("reason", "unavailable")}
        self._pregrasp_execution_timing = diagnostics
        if not proposal.get("accepted", False):
            return True  # Keep measuring; original visual/window failure handling remains.
        capture = np.asarray(proposal['predicted_grasp_pose'], dtype=float).copy()
        capture[3:] = self._grasp_pose[3:]
        low = capture.copy()
        low[:3] += self._belt_velocity(observation)*self.config.tracking_velocity_feedforward_time
        worker = self._ensure_planning_worker()
        request = self._fixed_high_request
        if request is not None:
            result = worker.poll() if worker is not None else None
            if result is None:
                diagnostics.update(reason="fixed_high_descent_check_pending", request_id=request.request_id)
                return True
            completed_request = self._completed_planning_request(result, request)
            if completed_request is None:
                self._planning_stale_results += 1
                return True
            request = completed_request
            if self._timed_descent_enabled(observation):
                self._record_descent_computation(result, request)
            self._fixed_high_request = None
            self._planning_request = None
            if result.cancelled:
                diagnostics['reason'] = "fixed_high_descent_check_cancelled"
                return True
            if not (result.success and result.path_fully_validated and result.continuation_fully_validated):
                self._fixed_high_rejected = True
                self._descent_rebook_pending = False
                self._descent_high_hold_q = None
                self._descent_feedback_request_id = None
                self._emit("fixed_high_descent_rejected", now, result.error or "continuation_not_validated")
                # A real low-corridor rejection may require another high IK
                # branch even when the corrected high is only millimetres
                # away. Explicitly queue the original full planner in that
                # case; the ordinary 4 mm cache test must not leave us stuck.
                admitted = self.env.refine_selected_grasp_plan(self._target_id, 0.,
                    self.config.intercept_deadline_margin, requested_time=proposal['estimated_grasp_time'],
                    requested_pose=capture)
                if isinstance(admitted, Mapping) and admitted.get('accepted', False):
                    self._grasp_pose = capture
                    self._pregrasp_capture_time_s = float(proposal['estimated_grasp_time'])
                    self._submit_pregrasp_approach_refresh(self._pregrasp_high_pose(capture, observation), observation)
                    diagnostics['reason'] = "fixed_high_descent_rejected_replanning_high"
                    return True
                return False
            self._fixed_high_validated_low = request.continuation_target_poses[0].copy()
            self._fixed_high_validated_capture_time = request.planned_target_time_s
            self._accept_initial_descent(result, request, observation)
            self._emit("fixed_high_descent_validated", now, f"request {request.request_id}")
            if self._descent_plan is not None and self._refresh_timed_descent(observation):
                return True
            # Compare new measurements at the SAME capture timestamp as the
            # completed check. Replacing it with now+0.8 on receipt would
            # mistake the workpiece's motion during computation for an error
            # and endlessly recheck a late but still reachable descent.
            checked_proposal = self.env.preview_fixed_high_grasp_plan(
                self._target_id, high, remaining, self.config.descent_lead_time,
                capture_high=self._pregrasp_capture_high_active,
                capture_time=request.planned_target_time_s)
            if checked_proposal.get('accepted', False) and not self._timed_descent_enabled(observation):
                proposal = checked_proposal
                capture = np.asarray(proposal['predicted_grasp_pose'], dtype=float).copy()
                capture[3:] = self._grasp_pose[3:]
                low = capture.copy()
                low[:3] += self._belt_velocity(observation)*self.config.tracking_velocity_feedforward_time
        checked = self._fixed_high_validated_low
        correction = float('inf') if checked is None else float(np.linalg.norm(low[:3]-checked[:3]))
        diagnostics.update(low_correction_m=correction,
            estimated_grasp_time_s=float(proposal['estimated_grasp_time']))
        if (not self._timed_descent_enabled(observation)
                and checked is not None and correction <= self.config.trajectory_replan_position_tolerance
                and quat_distance(low[3:], checked[3:]) <= self.config.trajectory_replan_orientation_tolerance):
            admitted = self.env.refine_selected_grasp_plan(self._target_id, 0.,
                self.config.intercept_deadline_margin, requested_time=proposal['estimated_grasp_time'],
                requested_pose=capture)
            if isinstance(admitted, Mapping) and admitted.get('accepted', False):
                self._grasp_pose = capture
                self._pregrasp_capture_time_s = float(proposal['estimated_grasp_time'])
                self._fixed_high_ready = True
                diagnostics.update(accepted=True, reason="fixed_high_waiting_for_descent_time",
                    descent_start_lateness_s=max(0., now-(self._pregrasp_capture_time_s-self.config.descent_lead_time)))
            return True
        if timed:
            # This is a NEW check. An existing accepted curve retains its
            # original clock in _refresh_timed_descent; do not rebook per tick.
            proposal = self._preview_new_descent(observation, high, remaining)
            diagnostics.update(descent_compute_forecast_s=self._descent_compute_s,
                remaining_high_motion_s=remaining, reason=proposal.get('reason', 'unavailable'))
            if not proposal.get('accepted', False):
                return True
            capture = np.asarray(proposal['predicted_grasp_pose'], dtype=float).copy()
            capture[3:] = self._grasp_pose[3:]
            low = capture.copy()
            low[:3] += self._belt_velocity(observation)*self.config.tracking_velocity_feedforward_time
            duration = self._new_descent_duration()
            diagnostics.update(estimated_grasp_time_s=float(proposal['estimated_grasp_time']),
                               planned_descent_start_s=float(proposal['estimated_grasp_time'])-duration)
        snapshot = self._future_open_planning_snapshot(anchor)
        if worker is None or snapshot is None:
            diagnostics['reason'] = "fixed_high_descent_check_queued"
            return True
        request = self._make_planning_request(high, observation,
            minimum_joint_margin=self._pregrasp_planning_joint_margin(),
            minimum_tcp_height=self._safe_path_minimum_tcp_height)
        if request is None:
            return True
        arm, full, epoch = snapshot
        request = replace(request, start_q=arm, full_qpos=full, scene_epoch=epoch,
            target_poses=(high.copy(),), goal_seeds=(arm.copy(),),
            continuation_target_poses=(low,), continuation_minimum_tcp_height=float(low[2])-.015,
            fixed_high_continuation=True, intercept_velocity_m_s=None,
            timed_continuation_duration_s=duration if timed else None,
            continuation_compute_forecast_s=self._descent_compute_s if timed else 0.,
            planned_target_time_s=float(proposal['estimated_grasp_time']),
            intercept_motion_start_time_s=now+remaining,
            intercept_descent_duration_s=duration+self._execution_dt(observation),
            intercept_account_for_planning_time=timed)
        if worker.submit(request):
            self._fixed_high_request = self._planning_request = request
            if self._descent_high_hold_q is not None:
                self._descent_feedback_request_id = request.request_id
            self._emit("fixed_high_descent_check_started", now, f"request {request.request_id}")
            diagnostics.update(reason="fixed_high_descent_check_pending", request_id=request.request_id)
        return True

    def _refresh_executing_pregrasp_intercept(
        self, observation: Mapping[str, Any]
    ) -> None:
        """Keep a delayed capture attached to a certified high approach.

        The existing route may remain a useful high transit prefix. A moved
        capture gets a matching high waypoint through the original background
        planner; the prefix alone can never authorize its descent. One pending
        continuation owns its immutable reservation until it is delivered.
        """
        if (self.config.prepare_high_only
                or not isinstance(observation.get("dispatch_clock"), Mapping)
                or not self.config.dynamic_intercept_tracking
                or self._stage != PolicyStage.PREGRASP
                or self._target_id is None or self._grasp_pose is None
                or not self._safe_joint_path
                or self._safe_path_stage != PolicyStage.PREGRASP
                or self._safe_path_generation != self._planning_generation
                or not self._same_target_id(self._safe_path_target_id, self._target_id)
                or self._safe_path_scene_epoch != int(self._observed_scene_epoch or 0)):
            return
        if self._pregrasp_approach_pose is not None:
            self._pregrasp_execution_timing.update(reason="high_approach_replan_pending")
            return
        if self._refresh_fixed_high_intercept(observation):
            return
        remaining = self._safe_path_remaining_motion_time(observation)
        required = self._pregrasp_required_motion_time(observation)
        if remaining is None or required is None:
            return
        now = self._timestamp_without_advancing(observation)
        current = self._joint_positions(observation)
        high_ready = bool(
            self._safe_joint_path_index >= len(self._safe_joint_path)-1
            and np.max(np.abs(current-self._safe_joint_path[-1])) <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)
            and self._joint_velocity_is_settled(
                observation, self.config.planner_waypoint_velocity_tolerance))
        if high_ready:
            # Planning and high-route traversal are already over. Adding a
            # new dispatch/planning delay on every settled tick would keep
            # pushing capture away and prevent the original descent clock.
            required = self.config.descent_lead_time
        diagnostics = {"timestamp": now, "remaining_command_and_feedback_motion_s": remaining,
            "required_capture_motion_s": required, "control_dt_s": self._execution_dt(observation),
            "estimate_source": "measured_phase_rate" if self._pregrasp_progress_sample is not None
                else "slower_command_and_feedback_quintic_phase",
            "observed_motion_time_scale": self._pregrasp_motion_time_scale,
            "actual_high_joint_state_ready": high_ready,
            "motor_settling_time_measured": False}
        self._pregrasp_execution_timing = diagnostics
        reserved = self._pregrasp_capture_time_s
        expected_high = self._pregrasp_high_pose(self._grasp_pose, observation)
        high_displacement = (float("inf") if self._safe_path_target_pose is None else float(
            np.linalg.norm(expected_high[:3]-self._safe_path_target_pose[:3])))
        if (high_displacement <= self.config.trajectory_replan_position_tolerance
                and reserved is not None and np.isfinite(reserved)
                and now + required <= reserved + 1e-9):
            diagnostics.update(reason="existing_capture_time_still_reachable",
                               estimated_grasp_time_s=float(reserved))
            return
        refiner = getattr(self.env, "refine_selected_grasp_plan", None)
        if not callable(refiner):
            diagnostics["reason"] = "environment_refiner_unavailable"
            return
        result = refiner(self._target_id, required, self.config.intercept_deadline_margin)
        if not isinstance(result, Mapping) or not bool(result.get("accepted", False)):
            diagnostics["reason"] = (str(result.get("reason", "refresh_not_accepted"))
                                     if isinstance(result, Mapping) else "refresh_unavailable")
            diagnostics["accepted"] = False
            return
        try:
            capture_time = float(result["estimated_grasp_time"])
            refined = np.asarray(result["predicted_grasp_pose"], dtype=float).reshape(7)
        except (KeyError, TypeError, ValueError):
            diagnostics["reason"] = "refresh_result_unavailable"
            return
        if not np.isfinite(capture_time) or not np.all(np.isfinite(refined)):
            diagnostics["reason"] = "refresh_result_unavailable"
            return
        self._pregrasp_capture_time_s = capture_time
        self._grasp_pose[:3] = self._retimed_grasp_position(self._grasp_pose[:3], refined[:3])
        expected_high = self._pregrasp_high_pose(self._grasp_pose, observation)
        high_displacement = (float("inf") if self._safe_path_target_pose is None else float(
            np.linalg.norm(expected_high[:3]-self._safe_path_target_pose[:3])))
        diagnostics.update(accepted=True, reason="ok", estimated_grasp_time_s=capture_time,
            predicted_grasp_pose=self._grasp_pose.copy(), required_high_pose=expected_high.copy(),
            certified_high_displacement_m=high_displacement)
        if high_displacement > self.config.trajectory_replan_position_tolerance:
            # Preserve this goal even if the worker is still unwinding a
            # previous request. Its old certified route remains only a
            # transit prefix until the new high leg has been installed.
            self._pregrasp_approach_pose = expected_high.copy()
            submitted = self._submit_pregrasp_approach_refresh(expected_high, observation)
            diagnostics["reason"] = ("high_approach_replan_pending" if submitted
                                     else "high_approach_replan_queued")

    def _refine_pregrasp_intercept(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
        *,
        from_approach_prefix: bool = False,
    ) -> tuple[bool, bool]:
        """Retime a reserved visual target from the executable safe-path ETA.

        Returns ``(accepted, pose_changed)``.  The environment owns the visual
        track and therefore performs the actual belt extrapolation; the policy
        supplies only its robot-motion duration and never reads object truth.
        """

        if (
            self.config.prepare_high_only
            or self._stage != PolicyStage.PREGRASP
            or self._target_id is None
        ):
            return True, False
        if self._refresh_fixed_high_intercept(observation):
            target_pose[:] = self._safe_path_target_pose
            return True, False
        required_time = self._pregrasp_required_motion_time(observation)
        if required_time is None:
            return True, False
        if (
            not from_approach_prefix
            and self._pregrasp_timing_refinements
            >= self.config.max_pregrasp_timing_refinements
        ):
            validation = _hook(
                self.env,
                ("validate_selected_grasp_plan",),
                (
                    (
                        self._target_id,
                        required_time,
                        self.config.intercept_deadline_margin,
                        self.config.pregrasp_blend_horizontal_tolerance,
                    ),
                    (
                        self._target_id,
                        required_time,
                        self.config.intercept_deadline_margin,
                    ),
                    (self._target_id, required_time),
                ),
            )
            if validation is None:
                # External environments without the optional validation hook
                # retain the legacy behavior.  Our MuJoCo environment always
                # exposes it, so the final recursively replanned path is
                # checked against the same visual clock as both refinements.
                return True, False
            # Validation may admit a small live correction and publish its
            # updated capture. Apply that same reservation below; returning
            # early would leave the policy's high/low pair behind the env.
            result = validation
        else:
            result = _hook(
                self.env,
                (
                    "refine_selected_grasp_plan",
                    "retime_selected_grasp_plan",
                ),
                (
                    (
                        self._target_id,
                        required_time,
                        self.config.intercept_deadline_margin,
                    ),
                    (self._target_id, required_time),
                ),
            )
            self._pregrasp_timing_refinements += 1
        if result is None:
            return True, False
        if isinstance(result, Mapping):
            accepted = bool(result.get("accepted", True))
            reason = str(result.get("reason", "insufficient_intercept_time"))
            refined = result.get("predicted_grasp_pose")
            if accepted and result.get("estimated_grasp_time") is not None:
                self._pregrasp_capture_time_s = float(result["estimated_grasp_time"])
        else:
            accepted = bool(result)
            reason = "insufficient_intercept_time"
            refined = None
        if not accepted:
            self._last_error = reason
            return False, False
        try:
            refined_pose = np.asarray(refined, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return True, False
        if refined_pose.shape != (7,) or not np.all(np.isfinite(refined_pose)):
            return True, False

        # The hook returns the predicted object/body pose.  Preserve the
        # already selected collision-checked wrist orientation and grasp
        # height, changing only the belt-plane interception coordinates.
        if self._grasp_pose is None:
            return True, False
        new_grasp_pose = self._grasp_pose.copy()
        new_grasp_pose[:3] = self._retimed_grasp_position(new_grasp_pose[:3], refined_pose[:3])
        displacement = float(
            np.linalg.norm(new_grasp_pose[:3] - self._grasp_pose[:3])
        )
        self._grasp_pose = new_grasp_pose
        target_pose[:] = self._pregrasp_high_pose(new_grasp_pose, observation)
        changed = displacement > self.config.trajectory_replan_position_tolerance
        if self._safe_path_target_pose is not None:
            changed = bool(np.linalg.norm(target_pose[:3]-self._safe_path_target_pose[:3])
                > self.config.trajectory_replan_position_tolerance)
        return True, changed

    def _commandable_joint_margin(self) -> float:
        """Return the planner margin required for an executable endpoint."""

        controller = _value(self.env, ("robot_controller", "controller"))
        controller_config = _value(controller, "config")
        command_margin = _value(controller_config, "joint_command_margin", 0.0)
        if not command_margin:
            robot_config = getattr(self.env, "robot_config", {})
            if isinstance(robot_config, Mapping):
                configured_controller = robot_config.get("controller", {})
                if isinstance(configured_controller, Mapping):
                    command_margin = configured_controller.get(
                        "joint_command_margin", 0.0
                    )
        try:
            command_margin = float(command_margin)
        except (TypeError, ValueError):
            command_margin = 0.0
        if not np.isfinite(command_margin):
            command_margin = 0.0
        return max(
            self.config.minimum_joint_limit_margin,
            command_margin,
        )

    def _pregrasp_planning_joint_margin(self) -> float:
        """Reserve tracking headroom beyond the actuator command envelope."""

        return max(
            self.config.minimum_joint_limit_margin,
            self._commandable_joint_margin()
            + self.config.pregrasp_tracking_joint_margin_reserve,
        )

    def _plan_adaptive_pregrasp_path(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> bool:
        """Search yaw toward nominal until a fully safe path is available."""

        target = self._target
        if target is None or _value(
            target, ("grasp_quaternion", "grasp_orientation"), None
        ) is not None:
            return False
        candidates = self._adaptive_grasp_quaternion_candidates(target)
        if self._adaptive_grasp_yaw_calibration(target) is None:
            return False

        current_quaternion = (
            target_pose[3:]
            if self._selected_grasp_quaternion is None
            else self._selected_grasp_quaternion
        )
        similarities = [
            abs(float(np.dot(current_quaternion, candidate)))
            for candidate in candidates
        ]
        start_index = int(np.argmax(similarities))
        required_margin = self._pregrasp_planning_joint_margin()
        for quaternion in candidates[start_index:]:
            candidate_pose = target_pose.copy()
            candidate_pose[3:] = quaternion
            if not self._plan_global_joint_path(
                candidate_pose,
                observation,
                minimum_joint_margin=required_margin,
            ):
                continue
            target_pose[:] = candidate_pose
            self._selected_grasp_quaternion = quaternion.copy()
            if self._grasp_pose is not None:
                self._grasp_pose[3:] = quaternion
            self._last_error = None
            return True
        return False

    def _submit_pregrasp_approach_refresh(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> bool:
        """Plan the moving endpoint from a certified high route's future q.

        This is the same future-open snapshot used by return prefetch. The
        robot can execute the existing route while this immutable request is
        searched; no speculative waypoint is sent before reaching its anchor.
        """
        if not self._safe_joint_path:
            return False
        # Keeping this intent is essential if submit is temporarily busy:
        # the old endpoint then remains a transit anchor, never a descent
        # certificate for the already updated capture reservation.
        self._pregrasp_approach_pose = target_pose.copy()
        worker = self._ensure_planning_worker()
        if worker is None:
            return False
        # Its endpoint was already predicted by the worker's cheap IK/ETA
        # forecast. Keep this useful certified transit while the future leg
        # is searched, and include its arrival clock in that forecast.
        anchor = self._safe_joint_path[-1].copy()
        prefix_duration = float(self._safe_path_remaining_motion_time(observation) or 0.0)
        current = self._joint_positions(observation)
        if (self._safe_joint_path_index >= len(self._safe_joint_path)-1
                and np.max(np.abs(current-anchor)) <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)):
            prefix_duration = 0.0
        snapshot = self._future_open_planning_snapshot(anchor)
        if snapshot is None:
            return False
        future_arm, full_qpos, scene_epoch = snapshot
        request = self._make_planning_request(
            target_pose, observation,
            minimum_joint_margin=self._pregrasp_planning_joint_margin(),
            minimum_tcp_height=min(float(self._safe_path_target_pose[2]), float(target_pose[2]))-.012,
        )
        if request is None:
            return False
        prediction = self._prediction_from_mapping(observation, self._target_id)
        planned_time = _value(
            prediction, ("planned_grasp_time", "estimated_grasp_time"),
            _value(self._target, ("planned_grasp_time", "estimated_grasp_time"), None),
        )
        if self._pregrasp_capture_time_s is not None:
            planned_time = self._pregrasp_capture_time_s
        request = replace(
            request, start_q=future_arm, full_qpos=full_qpos, scene_epoch=scene_epoch,
            route_waypoint_hints=self._previous_pregrasp_route_hints(
                future_arm, target_pose, scene_epoch=scene_epoch,
            ),
            goal_seeds=self._pregrasp_goal_seeds(
                future_arm, request.lower_limits, request.upper_limits,
                request.minimum_joint_margin, target_id=self._target_id, scene_epoch=scene_epoch,
            ),
            planned_target_time_s=None if planned_time is None else float(planned_time),
            intercept_motion_start_time_s=(
                self._timestamp_without_advancing(observation)
                + (prefix_duration if self._fixed_high_enabled(observation)
                   else max(prefix_duration, self._pregrasp_last_planning_time_s))
                if planned_time is not None else None),
        )
        if not worker.submit(request):
            return False
        self._pregrasp_approach_pose = target_pose.copy()
        self._pregrasp_approach_request = request
        self._pregrasp_approach_result = None
        self._planning_request = request
        return True

    def poll_pregrasp_approach_result(
        self, observation: Mapping[str, Any], *, target_pose: np.ndarray | None = None
    ) -> str:
        """Receive a future high route while its certified prefix is moving.

        Receipt and visual reservation adoption belong to the policy thread,
        independently of the joint interpolator. A pending intent without a
        request retries the original asynchronous submission. This does not
        solve IK, install a route, advance a trajectory, or produce a command.
        The original arm executor alone admits a saved result at its measured
        settled anchor. Return values preserve that executor's early exits.
        """
        request = self._pregrasp_approach_request
        if self._stage != PolicyStage.PREGRASP or not self._safe_joint_path:
            return "inactive"
        if target_pose is None:
            if self._pregrasp_approach_pose is None:
                return "inactive"
            target_pose = self._pregrasp_approach_pose.copy()
        if request is None:
            submitted = self._submit_pregrasp_approach_refresh(target_pose, observation)
            return "submitted" if submitted else "pending_submission"
        worker = self._ensure_planning_worker()
        if worker is None or not self._request_matches_goal(request, target_pose):
            self._invalidate_motion_plan()
            return "invalid"
        for _ in range(2):
            result = worker.poll()
            if result is None:
                break
            effective = self._completed_planning_request(result, request)
            if effective is None:
                self._planning_stale_results += 1
                continue
            if result.cancelled:
                self._planning_stale_results += 1
                self._pregrasp_approach_request = None
                self._pregrasp_approach_result = None
                self._planning_request = None
                return "prefix_only"
            if result.success:
                effective = self._adopt_pregrasp_effective_request(
                    result, request, target_pose, observation)
            if effective is None:
                self._planning_stale_results += 1
                if (result.request_id == request.request_id
                        and result.generation == request.generation
                        and result.scene_epoch == request.scene_epoch
                        and self._same_target_id(result.target_id, request.target_id)):
                    # This search is complete, but its forecast could not be
                    # admitted at the live receipt time. Queue a fresh search
                    # of the same high intent instead of waiting forever on
                    # a worker that has already returned its only result.
                    self._pregrasp_approach_request = None
                    self._pregrasp_approach_result = None
                    self._planning_request = None
                    return "prefix_only"
                continue
            # A failed forecast carries diagnostic coordinates only. Keep
            # the original high intent while finishing the certified prefix;
            # otherwise the next tick would mismatch an unadmitted endpoint
            # and throw away that prefix before its normal failure handling.
            if result.success:
                request = effective
            self._pregrasp_approach_request = request
            self._pregrasp_approach_pose = target_pose.copy()
            self._pregrasp_approach_result = result
            self._planning_request = None
        return "checked"

    def _arm_along_pregrasp_approach(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Execute a certified prefix and admit the future leg at measured q."""
        current = self._joint_positions(observation)
        request = self._pregrasp_approach_request
        if request is None:
            self._submit_pregrasp_approach_refresh(target_pose, observation)
            return self._execute_safe_joint_path(current, observation)
        received = self.poll_pregrasp_approach_result(observation, target_pose=target_pose)
        if received == "invalid":
            return current
        if received != "checked":
            return self._execute_safe_joint_path(current, observation)
        request = self._pregrasp_approach_request
        anchor = self._safe_joint_path[-1]
        at_anchor = bool(
            self._safe_joint_path_index >= len(self._safe_joint_path)-1
            and np.max(np.abs(current-anchor)) <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)
            and self._joint_velocity_is_settled(observation, self.config.planner_waypoint_velocity_tolerance)
        )
        if not at_anchor:
            # Prefix endpoints cannot trigger descent: the active capture
            # pose differs from this high route's old target. Live short-edge
            # checks still use the currently observed moving workpiece.
            return self._execute_safe_joint_path(current, observation)
        result = self._pregrasp_approach_result
        if result is None:
            return current
        if not result.success:
            self._pregrasp_approach_result = None
            self._pregrasp_approach_request = None
            if (result.error == "global planner found no safe IK goal"
                    and self._advance_dynamic_intercept(target_pose, observation)):
                self._submit_pregrasp_approach_refresh(target_pose, observation)
                return current
            self._last_error = result.error or "background approach planner returned no path"
            self._pregrasp_approach_pose = None
            self._record_motion_plan_failure()
            return current
        if not self._install_planning_result(result, request, target_pose, observation):
            self._invalidate_motion_plan()
            return current
        self._pregrasp_approach_pose = None
        self._pregrasp_approach_request = None
        self._pregrasp_approach_result = None
        accepted, pose_changed = self._refine_pregrasp_intercept(
            target_pose, observation, from_approach_prefix=True,
        )
        if not accepted:
            self._terminal_motion_plan_reason = self._last_error or "insufficient_intercept_time"
            self._invalidate_motion_plan()
            self._record_motion_plan_failure()
            return current
        if pose_changed:
            self._submit_pregrasp_approach_refresh(target_pose, observation)
        return self._execute_safe_joint_path(current, observation)

    def _arm_along_global_path(
        self,
        target_pose: np.ndarray,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        current = self._joint_positions(observation)
        if (self._stage == PolicyStage.PREGRASP and self.config.dynamic_intercept_tracking
                and self._pregrasp_approach_pose is not None and self._safe_joint_path):
            return self._arm_along_pregrasp_approach(target_pose, observation)
        target_changed = bool(
            self._safe_joint_path
            and (
                self._safe_path_target_pose is None
                or self._safe_path_stage != self._stage
                or self._safe_path_generation != self._planning_generation
                or (
                    self._safe_path_scene_epoch is not None
                    and self._safe_path_scene_epoch
                    != int(self._observed_scene_epoch or 0)
                )
                or not self._same_target_id(
                    self._safe_path_target_id, self._target_id
                )
                or np.linalg.norm(
                    target_pose[:3] - self._safe_path_target_pose[:3]
                )
                > self.config.trajectory_replan_position_tolerance
                or quat_distance(
                    target_pose[3:], self._safe_path_target_pose[3:]
                )
                > self.config.trajectory_replan_orientation_tolerance
            )
        )
        if target_changed:
            self._invalidate_motion_plan()

        if not self._safe_joint_path or target_changed:
            carry_replan = (
                self._carry_path_replan_constraints
                if self._stage == PolicyStage.TRANSFERRING
                else None
            )
            margin = (
                carry_replan[1]
                if carry_replan is not None
                else self._pregrasp_planning_joint_margin()
                if self._stage == PolicyStage.PREGRASP
                else self.config.minimum_joint_limit_margin
            )
            status = self._request_async_global_path(
                target_pose,
                observation,
                minimum_joint_margin=margin,
                minimum_tcp_height=(
                    carry_replan[0] if carry_replan is not None else None
                ),
                minimum_downward_alignment=(
                    carry_replan[2] if carry_replan is not None else None
                ),
            )
            if carry_replan is not None and status != "ready":
                joint_velocity = _value(
                    observation,
                    (
                        "joint_vel",
                        "joint_velocity",
                        "arm_joint_velocities",
                    ),
                    np.full(7, np.nan, dtype=float),
                )
                brake_safe = self._carry_replan_hold_is_safe(
                    current,
                    np.asarray(joint_velocity, dtype=float),
                )
                if status in {"failed", "unavailable"}:
                    self._last_error = self._last_error or (
                        "strict loaded carry replanner unavailable"
                    )
                    self._record_motion_plan_failure()
                if brake_safe:
                    assert self._carry_path_replan_brake_q is not None
                    return self._carry_path_replan_brake_q.copy()
                self._last_error = (
                    "loaded carry brake became unsafe while replanning"
                )
                if status == "pending":
                    self._record_motion_plan_failure()
                return current
            if status == "unavailable":
                factory = getattr(
                    self.env, "create_global_planning_worker", None
                )
                if callable(factory):
                    # A real environment whose isolated worker failed to start
                    # must hold/fail closed.  Falling through to synchronous
                    # multi-seed IK/RRT would recreate the original 3.5 s
                    # viewer freeze precisely on an error path.
                    self._last_error = (
                        self._last_error or "background planner unavailable"
                    )
                    self._record_motion_plan_failure()
                    return current
                # Duck-typed lightweight environments without an async
                # contract retain the synchronous compatibility planner.
                adaptive_pregrasp = bool(
                    self._stage == PolicyStage.PREGRASP
                    and self._adaptive_grasp_yaw_calibration(self._target)
                    is not None
                )
                planned = (
                    self._plan_adaptive_pregrasp_path(target_pose, observation)
                    if adaptive_pregrasp
                    else self._plan_global_joint_path(
                        target_pose,
                        observation,
                        minimum_joint_margin=margin,
                    )
                )
                status = "ready" if planned else "failed"
            if status == "pending":
                # Holding measured q is an explicit safe command while search
                # proceeds; simulation, cameras and viewer continue at 50 Hz.
                return current
            if status == "failed":
                self._record_motion_plan_failure()
                return current

            if carry_replan is not None:
                # The replacement path was generated from latest measured q
                # under the exact immutable C envelope.  Keep the loaded-path
                # follower/provenance, but descendants of the old endpoint
                # remain discarded until the new final q is captured.
                self._safe_path_from_carry_prefetch = True
                self._carry_path_replan_constraints = None
                self._carry_path_replan_brake_q = None
                self._carry_path_recovery_active = False
                self._carry_path_recovery_predictive = False
                self._carry_path_stalled_steps = 0

            if self._stage == PolicyStage.PREGRASP:
                accepted, pose_changed = self._refine_pregrasp_intercept(
                    target_pose, observation
                )
                if not accepted:
                    self._terminal_motion_plan_reason = (
                        self._last_error or "insufficient_intercept_time"
                    )
                    # The delivered route is physically safe but no longer
                    # temporally admissible.  Remove it before recording the
                    # terminal failure so no subsequent call can emit one of
                    # its waypoints.
                    self._invalidate_motion_plan()
                    self._record_motion_plan_failure()
                    return current
                if pose_changed:
                    if self.config.dynamic_intercept_tracking:
                        self._submit_pregrasp_approach_refresh(target_pose, observation)
                        return self._execute_safe_joint_path(current, observation)
                    # The first path was a safe timing preview.  Its earlier,
                    # deadline-aware intercept is a different Cartesian goal,
                    # so it must receive the same complete IK/RRT validation
                    # before any waypoint is commanded.
                    self._invalidate_motion_plan()
                    return current

        return self._execute_safe_joint_path(current, observation)

    def _safe_path_edge_is_safe(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Validate an edge with the exact constraints used for this path."""

        minimum_height = self._safe_path_minimum_tcp_height
        minimum_margin = self._safe_path_minimum_joint_margin
        minimum_alignment = self._safe_path_minimum_downward_alignment
        if minimum_height is not None and minimum_margin is not None:
            return self._global_edge_is_safe(
                start,
                end,
                minimum_tcp_height=minimum_height,
                minimum_joint_margin=minimum_margin,
                minimum_downward_alignment=minimum_alignment,
            )
        if self._safe_path_planner is not None:
            return bool(self._safe_path_planner.edge_valid(start, end))
        # Compatibility for lightweight duck-typed test environments.  The
        # production MuJoCo worker always attaches explicit constraints.
        return True

    def _carry_replan_hold_is_safe(
        self,
        current: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> bool:
        """Revalidate the strict loaded brake while its replacement plans."""

        constraints = self._carry_path_replan_constraints
        brake = self._carry_path_replan_brake_q
        if constraints is None or brake is None:
            return False
        velocity = np.asarray(joint_velocity, dtype=float).reshape(-1)[:7]
        if velocity.shape != (7,) or not np.all(np.isfinite(velocity)):
            return False
        minimum_height, minimum_margin, minimum_alignment = constraints
        # An unsafe inertial future is precisely why recovery is braking.  It
        # must prevent a new forward route from being installed, but cannot
        # justify replacing the already validated brake with an unvalidated
        # measured hold.  Recheck the actual emitted brake under the strict C
        # envelope every pending tick; if even that fails, fail closed.
        return bool(
            self._global_edge_is_safe(
                current,
                brake,
                minimum_tcp_height=minimum_height,
                minimum_joint_margin=minimum_margin,
                minimum_downward_alignment=minimum_alignment,
            )
        )

    def _safe_live_command_edge_is_safe(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Validate the next short servo chord at its actual command density.

        A global corner is densely revalidated before execution.  During that
        edge, the safety-critical live question is only whether the measured
        state can reach this tick's nearby command.  Using the much finer RRT
        resolution for that short chord adds redundant intermediate states
        (sometimes tens of them after feedback lag) and causes latency spikes;
        the local servo validator's 0.025-rad density is already the contract
        used by Cartesian tracking.  All state constraints remain identical.
        """

        minimum_height = self._safe_path_minimum_tcp_height
        minimum_margin = self._safe_path_minimum_joint_margin
        minimum_alignment = self._safe_path_minimum_downward_alignment
        edge_validator = getattr(self.env, "is_arm_edge_safe", None)
        if (
            minimum_height is not None
            and minimum_margin is not None
            and callable(edge_validator)
        ):
            try:
                return bool(
                    edge_validator(
                        start,
                        end,
                        normalized_resolution=(
                            self.config.planner_edge_resolution
                        ),
                        maximum_joint_step=0.025,
                        approach_axis_local=self._approach_axis_local,
                        minimum_downward_alignment=(
                            self._planner_downward_alignment_limit()
                            if minimum_alignment is None
                            else minimum_alignment
                        ),
                        minimum_tcp_height=minimum_height,
                        minimum_joint_margin=minimum_margin,
                        maximum_jacobian_condition=(
                            self.config.singularity_hard_stop_condition
                        ),
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                if callable(
                    getattr(self.env, "create_global_planning_worker", None)
                ):
                    return False
        return self._safe_path_edge_is_safe(start, end)

    def _safe_carry_predictive_edge_status(
        self, start: np.ndarray, end: np.ndarray
    ) -> tuple[bool, bool]:
        """Return C's hard preview status and its upstream-height warning.

        The immutable worker route and every command chord keep their exact
        hard height/cone/margin/collision certificate.  A loaded arm can sag
        several millimetres away from that joint edge, however, and a preview
        checked only against the hard floor first becomes false on the same
        tick as measured feedback crosses it.  Recheck the predicted endpoint
        with a small *higher* TCP floor so recovery can command an otherwise
        hard-valid brake while the measured start is still admissible.

        The ordinary hard edge check remains mandatory and includes the
        Jacobian/collision samples.  The endpoint warning adds no execution
        permission; missing production support fails closed.
        """

        hard_safe = self._safe_live_command_edge_is_safe(start, end)
        if not hard_safe:
            return False, False
        reserve = float(self.config.carry_path_tcp_height_guard_reserve)
        if reserve <= 0.0:
            return True, True
        minimum_height = self._safe_path_minimum_tcp_height
        minimum_margin = self._safe_path_minimum_joint_margin
        minimum_alignment = self._safe_path_minimum_downward_alignment
        state_validator = getattr(self.env, "is_arm_configuration_safe", None)
        if (
            minimum_height is not None
            and minimum_margin is not None
            and callable(state_validator)
        ):
            try:
                warning_safe = bool(
                    state_validator(
                        end,
                        approach_axis_local=self._approach_axis_local,
                        minimum_downward_alignment=(
                            self._planner_downward_alignment_limit()
                            if minimum_alignment is None
                            else minimum_alignment
                        ),
                        minimum_tcp_height=minimum_height + reserve,
                        minimum_joint_margin=minimum_margin,
                    )
                )
                return True, warning_safe
            except (TypeError, ValueError, RuntimeError):
                if callable(
                    getattr(self.env, "create_global_planning_worker", None)
                ):
                    return True, False
        # Lightweight test/compatibility environments without the production
        # state hook retain the already completed hard-edge result.
        return True, True

    def _safe_path_tracking_error_limit(self) -> float:
        """Return the feedback lead bound for the installed path contract."""

        if self._safe_path_from_carry_prefetch:
            return float(self.config.carry_path_tracking_error_limit)
        if self._safe_path_stage == PolicyStage.PREGRASP:
            return float(self.config.acquisition_path_tracking_error_limit)
        return float(self.config.global_path_tracking_error_limit)

    @staticmethod
    def _bounded_certified_edge_command(
        current: np.ndarray,
        requested: np.ndarray,
        certified_start: np.ndarray,
        certified_goal: np.ndarray,
        maximum_error: float,
    ) -> np.ndarray | None:
        """Pick the furthest requested point on an edge within feedback lead.

        The scalar interval is solved joint-by-joint, rather than estimating
        progress from the edge's largest joint.  Consequently the returned
        command is both exactly on the certified segment and within the
        infinity-norm feedback envelope even when the measured plant has a
        transverse tracking error.  ``None`` means no non-retreating point on
        that segment can satisfy the envelope, so the caller must fail closed.
        """

        current_q = np.asarray(current, dtype=float).reshape(7)
        requested_q = np.asarray(requested, dtype=float).reshape(7)
        start_q = np.asarray(certified_start, dtype=float).reshape(7)
        goal_q = np.asarray(certified_goal, dtype=float).reshape(7)
        limit = float(maximum_error)
        if not (
            np.all(np.isfinite(current_q))
            and np.all(np.isfinite(requested_q))
            and np.all(np.isfinite(start_q))
            and np.all(np.isfinite(goal_q))
            and np.isfinite(limit)
            and limit > 0.0
        ):
            return None

        delta = goal_q - start_q
        squared_length = float(np.dot(delta, delta))
        if squared_length <= 1e-18:
            if np.max(np.abs(goal_q - current_q)) <= limit + 1e-12:
                return goal_q.copy()
            return None

        # Preserve the worker's exact stop-to-stop time law whenever its raw
        # sample is already inside the measured-feedback envelope.  The old
        # interval solver always clamped progress to at least the measured
        # projection.  Consequently even a 0.1 mrad plant lead rewrote an
        # otherwise valid quintic sample, broke the controller's discrete
        # acceleration history, and sent the remainder of the certified edge
        # through a second online profile.  A small retreat of the actuator
        # target along the same certified edge is safe: the measured-to-command
        # chord is still checked below by the complete live validator.
        requested_scalar = float(
            np.dot(requested_q - start_q, delta) / squared_length
        )
        requested_on_edge = start_q + np.clip(
            requested_scalar, 0.0, 1.0
        ) * delta
        if (
            -1e-10 <= requested_scalar <= 1.0 + 1e-10
            and np.max(np.abs(requested_on_edge - requested_q)) <= 1e-10
            and np.max(np.abs(requested_q - current_q)) <= limit + 1e-12
        ):
            return requested_q.copy()

        measured_progress = float(
            np.clip(
                np.dot(current_q - start_q, delta) / squared_length,
                0.0,
                1.0,
            )
        )
        requested_progress = float(
            np.clip(
                np.dot(requested_q - start_q, delta) / squared_length,
                0.0,
                1.0,
            )
        )
        lower = measured_progress
        upper = max(measured_progress, requested_progress)
        for joint_delta, start_value, measured_value in zip(
            delta, start_q, current_q, strict=True
        ):
            if abs(joint_delta) <= 1e-15:
                if abs(start_value - measured_value) > limit + 1e-12:
                    return None
                continue
            first = (measured_value - limit - start_value) / joint_delta
            second = (measured_value + limit - start_value) / joint_delta
            lower = max(lower, min(first, second), 0.0)
            upper = min(upper, max(first, second), 1.0)
            if lower > upper + 1e-12:
                return None
        progress = float(np.clip(upper, 0.0, 1.0))
        command = start_q + progress * delta
        if np.max(np.abs(command - current_q)) > limit + 1e-10:
            return None
        return command

    def _retimed_certified_edge_command(
        self,
        current: np.ndarray,
        requested: np.ndarray,
        certified_start: np.ndarray,
        certified_goal: np.ndarray,
        maximum_error: float,
        observation: Mapping[str, Any],
        trajectory_duration: float,
    ) -> np.ndarray | None:
        """Retard one certified-edge sample without starting a second profile.

        A wall-clock quintic can outrun torque-limited feedback even though its
        command history is dynamically valid.  Treating the feedback-bounded
        point as a new static goal invokes the controller's ordinary braking
        profile on every tick and can almost double a long edge's execution
        time.  Instead, advance as far as possible along the *same certified
        edge* while preserving the controller's accepted target velocity and
        acceleration history and the ability to stop at this corner.

        This helper grants no geometric permission: the caller still runs the
        complete measured-to-command MuJoCo edge validator.  The controller
        independently repeats its position, velocity, acceleration and soft-
        limit stopping checks before accepting the sample verbatim.  Any
        malformed/off-edge history or empty feasible interval returns ``None``
        and retains the existing fail-closed online limiter.
        """

        diagnostics: dict[str, Any] = {"accepted": False}
        self._safe_edge_retime_diagnostics = diagnostics

        def reject(reason: str, **details):
            diagnostics.update(reason=reason, **details)
            return None

        try:
            current_q = np.asarray(current, dtype=float).reshape(7)
            requested_q = np.asarray(requested, dtype=float).reshape(7)
            start_q = np.asarray(certified_start, dtype=float).reshape(7)
            goal_q = np.asarray(certified_goal, dtype=float).reshape(7)
            limit = float(maximum_error)
            dt = self._execution_dt(observation)
            duration = float(trajectory_duration)
        except (TypeError, ValueError):
            return reject("invalid_input_shape")
        if not (
            np.all(np.isfinite(current_q))
            and np.all(np.isfinite(requested_q))
            and np.all(np.isfinite(start_q))
            and np.all(np.isfinite(goal_q))
            and np.isfinite(limit)
            and limit > 0.0
            and np.isfinite(dt)
            and dt > 0.0
            and np.isfinite(duration)
            and duration > 0.0
        ):
            return reject("invalid_input_value")

        delta = goal_q - start_q
        squared_length = float(np.dot(delta, delta))
        if squared_length <= 1e-18:
            return reject("zero_length_edge")
        if (
            _value(
                observation,
                ("joint_target", "joint_command_target"),
                None,
            )
            is None
            or _value(
                observation,
                (
                    "joint_target_velocity",
                    "joint_command_velocity",
                ),
                None,
            )
            is None
        ):
            return reject("missing_command_history")
        target_q, target_velocity = self._joint_command_history(
            observation, start_q
        )
        if not (
            np.all(np.isfinite(target_q))
            and np.all(np.isfinite(target_velocity))
        ):
            return reject("invalid_command_history")

        def edge_progress(value: np.ndarray) -> tuple[float, float]:
            scalar = float(np.dot(value - start_q, delta) / squared_length)
            projection = start_q + np.clip(scalar, 0.0, 1.0) * delta
            return scalar, float(np.max(np.abs(value - projection)))

        target_progress, target_cross_track = edge_progress(target_q)
        requested_progress, requested_cross_track = edge_progress(requested_q)
        current_progress, _ = edge_progress(current_q)
        history_tolerance = 1e-8
        target_progress_speed = float(
            np.dot(target_velocity, delta) / squared_length
        )
        target_velocity_cross_track = float(
            np.max(
                np.abs(
                    target_velocity - target_progress_speed * delta
                )
            )
        )
        if (
            target_cross_track > history_tolerance
            or requested_cross_track > history_tolerance
            or target_velocity_cross_track > history_tolerance
            or target_progress_speed < -history_tolerance
            or target_progress < -history_tolerance
            or target_progress > 1.0 + history_tolerance
            or requested_progress < -history_tolerance
            or requested_progress > 1.0 + history_tolerance
        ):
            return reject("history_not_on_certified_edge",
                position_residual_rad=target_cross_track,
                velocity_residual_rad_s=target_velocity_cross_track)
        target_progress = float(np.clip(target_progress, 0.0, 1.0))
        requested_progress = float(np.clip(requested_progress, 0.0, 1.0))
        current_progress = float(np.clip(current_progress, 0.0, 1.0))

        # Retiming is a delayed traversal of the worker's original normalized
        # quintic, never a faster catch-up in joint space.  Invert the monotone
        # smoothstep at the controller's accepted progress and permit at most
        # one original phase tick.  This preserves the worker's sampled TCP
        # angular-speed certificate and prevents a greedy catch-up sample from
        # becoming dynamically infeasible when the raw quintic enters its
        # deceleration half on the following tick.
        # The returned phase is the conservative side of the inverse; using the
        # midpoint can exceed the accepted progress by a few ulps and thereby
        # overstate the inherited angular certificate.
        target_phase = self._quintic_phase_for_progress(target_progress)
        next_phase = min(1.0, target_phase + dt / duration)
        phase_progress_limit = (
            10.0 * next_phase**3
            - 15.0 * next_phase**4
            + 6.0 * next_phase**5
        )
        # During deceleration the previous step's AVERAGE speed exceeds the
        # derivative at its endpoint. A shorter next interval may then demand
        # more discrete deceleration than allowed, despite a valid quintic.
        # Delay this step until its average speed fits the derivative at the
        # accepted endpoint. This only reduces the original phase advance;
        # it never accelerates past the worker's TCP/angular-speed certificate.
        if (next_phase > 0.5 and
                (phase_progress_limit - target_progress) / dt
                > 30.0 * next_phase**2 * (1.0-next_phase)**2 / duration):
            low, high = target_phase, next_phase
            for _ in range(36):
                phase = 0.5 * (low + high)
                progress = phase**3 * (10.0 + phase * (-15.0 + 6.0*phase))
                speed = 30.0 * phase**2 * (1.0-phase)**2 / duration
                if (progress-target_progress)/dt <= speed:
                    low = phase
                else:
                    high = phase
            phase_progress_limit = low**3 * (10.0 + low * (-15.0 + 6.0*low))

        # A local Ruckig profile or measured-response reference has its own
        # time law. Applying the *old quintic* phase cap here would silently
        # erase that reference a second time. This branch is only available
        # for dynamic, unloaded high motion, whose worker deliberately has
        # no quintic-derived TCP angular-time certificate. The original
        # per-axis velocity/acceleration, feedback lead, braking distance and
        # corner limits below still determine the command that can be sent.
        reference = getattr(self, '_safe_edge_reference_diagnostics', {})
        changed_time_law = bool(reference.get('time_law_changed', False)
            and self.stage == PolicyStage.PREGRASP
            and not self._global_path_tcp_speed_limits_enabled(PolicyStage.PREGRASP))
        if changed_time_law:
            trajectory = getattr(self, '_safe_edge_trajectory', None)
            model = getattr(self.env, 'joint_response_model', None)
            use_response = bool(reference.get('applied', False) and model is not None)
            compiled_reference = None
            if use_response and callable(getattr(model, 'compile_edge_reference', None)):
                reference_key = (id(trajectory), id(model))
                cached = getattr(self, '_compiled_edge_response', None)
                if cached is None or cached[0] != reference_key:
                    cached = (reference_key, model.compile_edge_reference(trajectory))
                    self._compiled_edge_response = cached
                compiled_reference = cached[1]

            def state_at(reference_time):
                if use_response:
                    if compiled_reference is not None:
                        return compiled_reference.scalar_state(reference_time)
                    position, details = model.preview_on_edge(trajectory, reference_time)
                    velocity = np.asarray(details['candidate_velocity_rad_s'])
                else:
                    position, velocity, _ = trajectory.sample_full(reference_time)
                scalar = float(delta @ (position-start_q) / squared_length)
                speed = float(delta @ velocity / squared_length)
                return scalar, speed

            if trajectory is not None:
                # Invert THIS reference at the last sent position. Advance
                # at most one profile interval, even if feedback clipping has
                # put it behind the wall clock. Greedy catch-up can enter a
                # deceleration point too fast to satisfy the next interval.
                lower, upper = 0., float(trajectory.duration)
                if not use_response and callable(getattr(trajectory, 'time_at_progress', None)):
                    phase_time = trajectory.time_at_progress(target_progress)
                else:
                    for _ in range(36):
                        middle = .5 * (lower + upper)
                        if state_at(middle)[0] < target_progress:
                            lower = middle
                        else:
                            upper = middle
                    phase_time = lower
                next_time = min(trajectory.duration, phase_time + dt)
                next_progress, next_speed = state_at(next_time)
                if (next_progress-target_progress)/dt > next_speed + 1e-12:
                    lower, upper = phase_time, next_time
                    for _ in range(36):
                        middle = .5 * (lower + upper)
                        progress, speed = state_at(middle)
                        if (progress-target_progress)/dt <= speed:
                            lower = middle
                        else:
                            upper = middle
                    next_progress = state_at(lower)[0]
                phase_progress_limit = min(requested_progress, max(target_progress, next_progress))
        diagnostics['reference_time_law'] = (
            reference.get('trajectory_backend', 'quintic')
            if changed_time_law else 'quintic')
        diagnostics['response_reference_applied'] = bool(
            changed_time_law and reference.get('applied', False))

        # Never retreat the accepted actuator target and never advance beyond
        # the worker's current wall-clock sample.  Measured feedback may coast
        # slightly ahead of the accepted command along the same certified
        # edge; requiring the next dynamically reachable target to overtake it
        # in one tick can make an otherwise valid acceleration interval empty.
        # A behind-feedback target is safe because the full measured-to-target
        # chord is live-validated below, while actuator-target progress remains
        # monotonic.
        minimum_progress = target_progress
        maximum_progress = min(requested_progress, phase_progress_limit)
        if minimum_progress > maximum_progress + 1e-10:
            return reject("nominal_phase_behind_command")

        velocity_limits, acceleration_limits = self._trajectory_limits()
        if not (
            np.all(np.isfinite(velocity_limits))
            and np.all(velocity_limits > 0.0)
            and np.all(np.isfinite(acceleration_limits))
            and np.all(acceleration_limits > 0.0)
        ):
            return reject("invalid_motion_limits")

        # Intersect the nominal measured-feedback infinity-norm tube with the
        # scalar certified segment.  Do not borrow a one-tick acceleration
        # reserve here: without a proven lower bound on plant motion that
        # reserve can become a permanent tracking-limit expansion.
        for joint_delta, start_value, measured_value in zip(
            delta, start_q, current_q, strict=True
        ):
            if abs(joint_delta) <= 1e-15:
                if abs(start_value - measured_value) > limit + 1e-12:
                    return reject("stationary_joint_outside_feedback_envelope")
                continue
            lower_position = measured_value - limit
            upper_position = measured_value + limit
            first = (lower_position - start_value) / joint_delta
            second = (upper_position - start_value) / joint_delta
            minimum_progress = max(minimum_progress, min(first, second), 0.0)
            maximum_progress = min(maximum_progress, max(first, second), 1.0)
            if minimum_progress > maximum_progress + 1e-12:
                return reject("empty_feedback_position_interval")

        # Solve a one-dimensional interval for progress speed u=ds/dt.  Each
        # joint independently restricts u by its velocity, acceleration and
        # exact discrete stopping distance to this immutable corner.
        minimum_progress_speed = max(
            0.0, (minimum_progress - target_progress) / dt
        )
        maximum_progress_speed = max(
            0.0, (maximum_progress - target_progress) / dt
        )
        # Stopping only at the distant route corner is insufficient: a target
        # can reach the feedback-lead boundary while still moving too fast to
        # brake there, making the NEXT tick's speed/position interval empty.
        # Reserve continuous braking at both the route corner and the CURRENT
        # feedback boundary, so shorter future intervals have room. Assume no
        # future plant motion; do not expand the envelope or invent feedback.
        corner_remaining = np.abs(goal_q - target_q)
        feedback_remaining = np.maximum(
            0.0, limit - np.sign(delta) * (target_q - current_q))
        remaining = corner_remaining
        scaled_distance = remaining / (
            acceleration_limits * dt * dt
        )
        intervals = np.maximum(
            1.0,
            np.ceil(
                0.5
                * (np.sqrt(1.0 + 8.0 * scaled_distance) - 1.0)
            ),
        )
        stopping_speed = np.where(
            remaining > 0.0,
            (
                remaining / dt
                + 0.5
                * acceleration_limits
                * dt
                * intervals
                * (intervals - 1.0)
            )
            / intervals,
            0.0,
        )
        # The next host interval is not guaranteed to equal this one. Reserve
        # continuous braking distance AFTER this command step at the frozen
        # boundary/corner: v*dt + v²/(2*a) <= distance. Unlike an equal-step
        # discrete reserve, this remains feasible if the next cycle is shorter.
        braking_remaining = np.minimum(corner_remaining, feedback_remaining)
        feedback_stop_speed = (2.0 * acceleration_limits * braking_remaining) / (
            np.sqrt((acceleration_limits * dt)**2
                    + 2.0 * acceleration_limits * braking_remaining)
            + acceleration_limits * dt)
        stopping_speed = np.minimum(stopping_speed, feedback_stop_speed)
        for joint_index, (joint_delta, previous_velocity, velocity_limit, acceleration_limit, stop_limit) in enumerate(zip(
            delta,
            target_velocity,
            velocity_limits,
            acceleration_limits,
            stopping_speed,
            strict=True,
        )):
            if abs(joint_delta) <= 1e-15:
                if abs(previous_velocity) > acceleration_limit * dt + 1e-9:
                    return reject("stationary_joint_cannot_brake", joint_index=joint_index)
                continue
            maximum_progress_speed = min(
                maximum_progress_speed,
                velocity_limit / abs(joint_delta),
                stop_limit / abs(joint_delta),
            )
            first = (
                previous_velocity - acceleration_limit * dt
            ) / joint_delta
            second = (
                previous_velocity + acceleration_limit * dt
            ) / joint_delta
            minimum_progress_speed = max(
                minimum_progress_speed, min(first, second)
            )
            maximum_progress_speed = min(
                maximum_progress_speed, max(first, second)
            )
            if (
                minimum_progress_speed
                > maximum_progress_speed + 1e-12
            ):
                return reject("empty_velocity_acceleration_interval", joint_index=joint_index,
                    minimum_progress_speed_s_inv=minimum_progress_speed,
                    maximum_progress_speed_s_inv=maximum_progress_speed,
                    feedback_braking_distance_rad=float(feedback_remaining[joint_index]),
                    corner_braking_distance_rad=float(corner_remaining[joint_index]),
                    phase_progress_limit=float(phase_progress_limit),
                    previous_progress=float(target_progress),
                    previous_command_velocity_rad_s=float(previous_velocity), dt_s=dt)

        progress_speed = max(0.0, maximum_progress_speed)
        if progress_speed < minimum_progress_speed - 1e-12:
            return reject("empty_progress_speed_interval")
        progress = float(
            np.clip(
                target_progress + dt * progress_speed,
                minimum_progress,
                maximum_progress,
            )
        )
        command = start_q + progress * delta
        if (
            np.max(np.abs(command - current_q)) > limit + 1e-10
            or progress > requested_progress + 1e-10
            or progress + 1e-10 < target_progress
        ):
            return reject("output_outside_original_envelope")
        diagnostics.update(accepted=True, reason="ok", dt_s=dt,
            progress=progress, progress_speed_s_inv=progress_speed,
            requested_progress=requested_progress,
            reference_clipped_by_original_limits=bool(progress < requested_progress - 1e-10),
            feedback_braking_distance_rad=feedback_remaining.tolist(),
            feedback_stop_speed_rad_s=feedback_stop_speed.tolist())
        return command

    def _execute_safe_joint_path(
        self,
        current: np.ndarray,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Time-parameterize and live-validate one RRT corner at a time.

        RRT produces a collision-checked polyline, not actuator commands.  A
        synchronized quintic makes the executed velocity/acceleration match
        the ETA used for deadline admission.  Before each corner and before
        every short servo command, the chord from measured q is revalidated;
        any drift fails closed and requests a fresh versioned plan next tick.
        """

        self._arm_command_is_time_parameterized = False
        if not self._safe_joint_path:
            return current.copy()
        last_index = len(self._safe_joint_path) - 1
        self._safe_joint_path_index = int(
            np.clip(self._safe_joint_path_index, 0, last_index)
        )
        goal_index = self._safe_joint_path_index
        goal = self._safe_joint_path[goal_index]
        now = self._timestamp_without_advancing(observation)

        if (
            self._safe_edge_trajectory is None
            or self._safe_edge_goal_index != goal_index
        ):
            expected_start = (
                self._safe_joint_path[goal_index - 1]
                if goal_index > 0
                else current
            )
            # The worker has already densely validated every immutable corner
            # edge, and delivery revalidates a rebased first edge.  Repeating
            # an entire later RRT edge here performed hundreds of mj_forward+
            # SVD calls in one 50 Hz tick (up to 285 ms on long routes).
            # Settled-corner starts therefore reuse that immutable proof.
            # If feedback is not exactly on the certified corner, validate the
            # drift connector first; every sampled command below is still
            # checked from the latest measured q with the full collision,
            # cone, singularity, height and soft-limit constraints.
            settled_validated_corner = bool(
                self._safe_path_worker_validated
                and
                goal_index > 0
                and np.max(np.abs(current - expected_start))
                <= self._safe_path_settle_tolerance(goal_index - 1)
            )
            start_drift = float(np.max(np.abs(current - expected_start)))
            edge_is_safe = True
            if settled_validated_corner:
                if start_drift > 1e-9:
                    edge_is_safe = self._safe_live_command_edge_is_safe(
                        current, expected_start
                    )
            else:
                # Legacy/synchronous planners do not carry the worker's
                # immutable full-edge certificate, so preserve their original
                # fail-closed whole-edge validation.
                edge_is_safe = self._safe_path_edge_is_safe(current, goal)
            if not edge_is_safe:
                self._last_error = (
                    "measured state left the validated global edge"
                )
                self._invalidate_unsafe_safe_path()
                return current.copy()
            velocity, acceleration = self._trajectory_limits()
            cached_duration = (
                self._safe_path_edge_durations[goal_index - 1]
                if (
                    goal_index > 0
                    and goal_index - 1
                    < len(self._safe_path_edge_durations)
                    and np.isfinite(
                        self._safe_path_edge_durations[goal_index - 1]
                    )
                )
                else None
            )
            if cached_duration is not None:
                edge_minimum_duration = (
                    float(cached_duration)
                    if settled_validated_corner
                    else self._realtime_rebased_edge_duration(
                        expected_start,
                        current,
                        goal,
                        float(cached_duration),
                        stage=self._safe_path_stage,
                    )
                )
            else:
                edge_minimum_duration = self._edge_minimum_duration(
                    current, goal, global_path=True, stage=self._safe_path_stage,
                )
            trajectory_start = (
                expected_start
                if settled_validated_corner
                else current
            )
            self._safe_edge_trajectory = self._build_global_edge_trajectory(
                trajectory_start,
                goal,
                velocity,
                acceleration,
                minimum_duration=max(
                    edge_minimum_duration,
                    0.0
                    if np.max(np.abs(goal - current)) <= 1e-9
                    else self.config.trajectory_minimum_duration
                ),
            )
            if goal_index > 0:
                duration_index = goal_index - 1
                while len(self._safe_path_edge_durations) <= duration_index:
                    self._safe_path_edge_durations.append(float("nan"))
                self._safe_path_edge_durations[duration_index] = float(
                    self._safe_edge_trajectory.duration
                )
            # Do not advance wall-clock phase until the controller's accepted
            # target history is exactly at this stop-to-stop start.  Stage
            # changes can leave the previous local/global command a few
            # milliradians away with nonzero target velocity; starting now
            # would make every later quintic sample fail the acceleration
            # continuity check and fall back to a second online profile.
            self._safe_edge_started_at = None
            self._safe_edge_goal_index = goal_index
            self._safe_edge_passthrough_eligible = True
            self._safe_edge_dynamic_retime_active = False
            self._safe_edge_handoff_pending = True

        trajectory = self._safe_edge_trajectory
        assert trajectory is not None
        feedback_connector = bool(
            goal_index == self._safe_path_feedback_connector_goal_index
        )
        if self._safe_edge_handoff_pending:
            handoff_q = goal if feedback_connector else trajectory.q0
            command_target, command_velocity = self._joint_command_history(
                observation, handoff_q
            )
            handoff_position_tolerance = 1e-9
            handoff_velocity_tolerance = 1e-3
            history_aligned = bool(
                np.all(np.isfinite(command_target))
                and np.all(np.isfinite(command_velocity))
                and np.max(np.abs(command_target - handoff_q))
                <= handoff_position_tolerance
                and np.max(np.abs(command_velocity))
                <= handoff_velocity_tolerance
            )
            handoff_safe = bool(
                history_aligned
                and not feedback_connector
            ) or self._safe_live_command_edge_is_safe(current, handoff_q)
            if not handoff_safe:
                self._last_error = (
                    "certified edge command-history handoff became unsafe"
                )
                self._invalidate_unsafe_safe_path()
                return current.copy()
            if feedback_connector:
                # This edge exists only because feedback was not bit-exactly
                # at the worker start while the actuator already held it.  A
                # constant exact start target closes that feedback error; an
                # artificial measured->start quintic would instead break the
                # already-correct controller history.
                if (
                    np.max(np.abs(current - goal))
                    <= self._safe_path_settle_tolerance(goal_index)
                    and self._joint_velocity_is_settled(
                        observation,
                        self.config.planner_waypoint_velocity_tolerance,
                    )
                    and goal_index < last_index
                ):
                    self._safe_joint_path_index += 1
                    self._safe_edge_trajectory = None
                    self._safe_edge_started_at = None
                    self._safe_edge_goal_index = None
                    self._safe_edge_handoff_pending = False
                self._record_motion_plan_success()
                # This connector can require the ordinary limiter while
                # braking prior command history. It is not an already timed
                # sample until that history is stationary at the anchor.
                self._arm_command_is_time_parameterized = history_aligned
                return goal.copy()
            if not history_aligned:
                # The ordinary limiter joins the prior target sequence to the
                # new certified start. Phase remains frozen; this is an
                # untimed connector, not a passthrough-certified sample.
                self._record_motion_plan_success()
                self._arm_command_is_time_parameterized = False
                return handoff_q.copy()
            # Treat sub-milliradian/s history as controller quantization noise,
            # but normalize it explicitly before the first moving sample.  A
            # trajectory that starts against even that tiny residual can fail
            # the discrete acceleration gate two ticks later and poison the
            # entire edge.  One exact zero-speed q0 sample closes the history;
            # phase remains frozen until the following observation confirms it.
            if np.max(np.abs(command_velocity)) > 1e-9:
                self._record_motion_plan_success()
                self._arm_command_is_time_parameterized = True
                return handoff_q.copy()
            self._safe_edge_handoff_pending = False
            self._safe_edge_started_at = now

        started_at = (
            now
            if self._safe_edge_started_at is None
            else self._safe_edge_started_at
        )
        sample_time = now
        dispatch_clock = observation.get("dispatch_clock", {})
        if isinstance(dispatch_clock, Mapping):
            try:
                scheduled = float(dispatch_clock.get("scheduled_dispatch_policy_time_s", now))
            except (TypeError, ValueError):
                scheduled = now
            if np.isfinite(scheduled):
                sample_time = max(now, scheduled)
        # Only the trajectory request looks ahead to its reserved dispatch
        # slot.  Observation timestamps and visual ages stay at real capture/
        # decision time, and the retimer/controller share that slot's dt.
        elapsed = max(0.0, sample_time - started_at)
        raw_command = self._sample_safe_edge_reference(trajectory, elapsed)
        tracking_limit = self._safe_path_tracking_error_limit()
        certified_start = (
            self._safe_joint_path[goal_index - 1]
            if goal_index > 0
            else trajectory.q0
        )

        # The wall-clock quintic can run ahead of torque-limited feedback.
        # Project every command onto the immutable certified edge *before* the
        # first live validation.  Doing this even inside the lead envelope
        # continuously corrects transverse plant drift instead of waiting for
        # a hard collision/height/orientation constraint to reject it.
        command = self._bounded_certified_edge_command(
            current,
            raw_command,
            certified_start,
            goal,
            tracking_limit,
        )
        command_is_safe = False
        projected = command is not None
        command_follows_time_law = bool(
            command is not None
            and np.allclose(command, raw_command, atol=1e-12, rtol=0.0)
        )
        command_is_dynamically_retimed = False
        # Once an edge deviates from its original wall-clock phase, every
        # subsequent sample must continue through the history-aware retimer;
        # a later raw sample can re-enter the feedback tube while still asking
        # for an impossible velocity/acceleration jump from the retimed target.
        dynamic_retime_active = bool(
            getattr(self, "_safe_edge_dynamic_retime_active", False)
        )
        explicit_command_history = bool(
            _value(
                observation,
                ("joint_target", "joint_command_target"),
                None,
            )
            is not None
            and _value(
                observation,
                (
                    "joint_target_velocity",
                    "joint_command_velocity",
                ),
                None,
            )
            is not None
        )
        if command is not None and self._safe_edge_passthrough_eligible:
            retimed = None
            if explicit_command_history:
                retimed = self._retimed_certified_edge_command(
                    current,
                    raw_command,
                    certified_start,
                    goal,
                    tracking_limit,
                    observation,
                    trajectory.duration,
                )
            if retimed is not None:
                command = retimed
                command_is_dynamically_retimed = bool(
                    dynamic_retime_active
                    or not np.allclose(
                        retimed, raw_command, atol=1e-12, rtol=0.0
                    )
                )
                if command_is_dynamically_retimed:
                    self._safe_edge_dynamic_retime_active = True
            else:
                # Passthrough is a proof, not a hint.  Missing, malformed or
                # off-edge controller history and an empty dynamic interval
                # all mean that the controller cannot reproduce this sample's
                # certified command history.  Disable the marker even when the
                # raw wall-clock point happens to remain inside the feedback
                # tube; otherwise the controller can reject it and synthesize
                # an unvalidated per-joint fallback away from the immutable
                # edge.  Eligibility remains sticky-false below until the next
                # independently certified corner.
                command_follows_time_law = False

        # A static command chord cannot see inertia that will cross a hard
        # boundary during the following delayed physics ticks.  Carry C is the
        # only loaded path with a narrow height envelope, so give it a bounded
        # velocity look-ahead and a hysteretic certified-edge recovery.  No
        # threshold is relaxed: both the look-ahead and recovery chord use the
        # existing complete live MuJoCo validator.
        if command is not None and self._safe_path_from_carry_prefetch:
            delta = goal - certified_start
            squared_length = float(np.dot(delta, delta))
            joint_velocity = _value(
                observation,
                ("joint_vel", "joint_velocity", "arm_joint_velocities"),
                np.zeros(7, dtype=float),
            )
            try:
                joint_velocity = np.asarray(
                    joint_velocity, dtype=float
                ).reshape(-1)[:7]
            except (TypeError, ValueError):
                joint_velocity = np.full(7, np.nan, dtype=float)
            velocity_valid = bool(
                joint_velocity.shape == (7,)
                and np.all(np.isfinite(joint_velocity))
            )
            if squared_length > 1e-18 and velocity_valid:
                progress = float(
                    np.clip(
                        np.dot(current - certified_start, delta)
                        / squared_length,
                        0.0,
                        1.0,
                    )
                )
                projected_current = certified_start + progress * delta
                cross_track = float(
                    np.max(np.abs(current - projected_current))
                )
                horizon = self._carry_stopping_guard_horizon(
                    joint_velocity, observation
                )
                if horizon is None:
                    # Missing/non-finite braking data cannot justify advancing
                    # a loaded arm near the immutable C envelope.
                    self._carry_path_recovery_active = True
                    self._carry_path_recovery_predictive = True
                    self._carry_path_stalled_steps = 0
                    hard_future_safe = False
                    warning_future_safe = False
                else:
                    future = current + horizon * joint_velocity
                    (
                        hard_future_safe,
                        warning_future_safe,
                    ) = self._safe_carry_predictive_edge_status(
                        current,
                        future,
                    )
                warning_armed = bool(
                    self._carry_path_predictive_warning_edge_index
                    != goal_index
                )
                reserve_warning = bool(
                    warning_armed
                    and hard_future_safe
                    and not warning_future_safe
                )
                if reserve_warning:
                    # Arm this upstream warning at most once per immutable
                    # certified edge.  Recovery below still has to emit and
                    # validate a real hard-envelope brake before advancing.
                    self._carry_path_predictive_warning_edge_index = goal_index
                if self._carry_path_recovery_active and horizon is not None:
                    if (
                        cross_track
                        > self.config.carry_path_cross_track_enter_limit
                    ):
                        self._carry_path_recovery_predictive = False
                    velocity_settled = bool(
                        np.max(np.abs(joint_velocity))
                        <= self.config.planner_waypoint_velocity_tolerance
                    )
                    if velocity_settled:
                        if (
                            hard_future_safe
                            and cross_track
                            < (
                                self.config.carry_path_cross_track_enter_limit
                                if self._carry_path_recovery_predictive
                                else self.config.carry_path_cross_track_exit_limit
                            )
                        ):
                            # The higher floor is an early-warning reserve, not
                            # a second route constraint.  Once its hard-valid
                            # brake has settled on this immutable edge, consume
                            # the warning for that edge and resume.  The full
                            # dynamic stopping preview against the real .408 m
                            # floor still runs on every subsequent tick.
                            self._carry_path_recovery_active = False
                            self._carry_path_recovery_predictive = False
                            self._carry_path_stalled_steps = 0
                        else:
                            # Only a persistent transverse equilibrium proves
                            # that the certified joint edge is untrackable.
                            # A reserve-only height warning is intentionally
                            # consumed after one settled, hard-valid brake; it
                            # must not rebuild the same .408-certified topology
                            # in a loop.
                            if (
                                not self._carry_path_recovery_predictive
                                and
                                cross_track
                                >= self.config.carry_path_cross_track_exit_limit
                            ):
                                self._carry_path_stalled_steps += 1
                            else:
                                self._carry_path_stalled_steps = 0
                    else:
                        self._carry_path_stalled_steps = 0
                elif (
                    cross_track
                    > self.config.carry_path_cross_track_enter_limit
                    or not hard_future_safe
                    or reserve_warning
                ):
                    self._carry_path_recovery_active = True
                    self._carry_path_recovery_predictive = bool(
                        cross_track
                        <= self.config.carry_path_cross_track_enter_limit
                        and (not hard_future_safe or reserve_warning)
                    )
                    self._carry_path_stalled_steps = 0
                else:
                    self._carry_path_stalled_steps = 0

                if self._carry_path_recovery_active and horizon is not None:
                    progress_velocity = float(
                        np.dot(joint_velocity, delta) / squared_length
                    )
                    brake_progress = float(
                        np.clip(
                            progress
                            - max(progress_velocity, 0.0) * horizon,
                            0.0,
                            1.0,
                        )
                    )
                    command = certified_start + brake_progress * delta
                    command_follows_time_law = False
                    command_is_safe = self._safe_live_command_edge_is_safe(
                        current, command
                    )
                    if (
                        command_is_safe
                        and self._carry_path_stalled_steps
                        >= self.config.carry_path_stalled_replan_steps
                        and self._safe_path_minimum_tcp_height is not None
                        and self._safe_path_minimum_joint_margin is not None
                        and self._safe_path_minimum_downward_alignment
                        is not None
                    ):
                        constraints = (
                            float(self._safe_path_minimum_tcp_height),
                            float(self._safe_path_minimum_joint_margin),
                            float(
                                self._safe_path_minimum_downward_alignment
                            ),
                        )
                        # The current/brake chord is safe, but the immutable
                        # edge is not feedback-trackable under this loaded
                        # equilibrium.  Atomically discard C and its A/B
                        # descendants, then let the ordinary isolated worker
                        # certify a replacement from latest measured q.  This
                        # tick still emits the already validated brake.
                        self._last_error = (
                            "carry path stalled off its certified edge; "
                            "replanning from measured state"
                        )
                        self._invalidate_unsafe_safe_path()
                        self._carry_path_replan_constraints = constraints
                        self._carry_path_replan_brake_q = command.copy()
                        self._carry_path_recovery_predictive = False
                        return command.copy()
                elif horizon is not None:
                    command_is_safe = self._safe_live_command_edge_is_safe(
                        current, command
                    )
            else:
                self._carry_path_recovery_active = True
                self._carry_path_recovery_predictive = True
                self._carry_path_stalled_steps = 0
        elif command is not None:
            self._carry_path_stalled_steps = 0
            command_is_safe = self._safe_live_command_edge_is_safe(
                current, command
            )

        if not command_is_safe and not (
            self._safe_path_from_carry_prefetch
            and self._carry_path_recovery_active
        ):
            # A half-lead retry can recover a chord that approaches a hard
            # height/orientation envelope too aggressively.  It is recomputed
            # on the immutable edge and receives the same full live check;
            # the raw command is never substituted here.
            retry = self._bounded_certified_edge_command(
                current,
                raw_command,
                certified_start,
                goal,
                0.5 * tracking_limit,
            )
            if retry is not None and (
                not projected
                or not np.allclose(retry, command, atol=1e-12, rtol=0.0)
            ):
                command = retry
                command_follows_time_law = False
                command_is_safe = self._safe_live_command_edge_is_safe(
                    current, command
                )
        if not command_is_safe:
            self._last_error = "live global path command edge became unsafe"
            self._invalidate_unsafe_safe_path()
            return current.copy()

        if elapsed >= trajectory.duration - 1e-12:
            # The command has reached the stop-to-stop endpoint.  Wait for the
            # measured arm to enter its existing tracking tolerance before
            # validating the next corner from that measured state.
            joint_velocity = _value(
                observation,
                ("joint_vel", "joint_velocity", "arm_joint_velocities"),
                np.zeros(7, dtype=float),
            )
            try:
                joint_velocity = np.asarray(
                    joint_velocity, dtype=float
                ).reshape(-1)[:7]
            except (TypeError, ValueError):
                joint_velocity = np.full(7, np.inf, dtype=float)
            velocity_settled = bool(
                joint_velocity.shape == (7,)
                and np.all(np.isfinite(joint_velocity))
                and np.max(np.abs(joint_velocity))
                <= self.config.planner_waypoint_velocity_tolerance
            )
            if (
                np.max(np.abs(current - goal))
                <= self._safe_path_settle_tolerance(goal_index)
                and velocity_settled
                and goal_index < last_index
            ):
                self._safe_joint_path_index += 1
                self._safe_edge_trajectory = None
                self._safe_edge_started_at = None
                self._safe_edge_goal_index = None
        self._record_motion_plan_success()
        # This exact sample came from the installed stop-to-stop trajectory,
        # was projected back onto its immutable certified edge, and passed the
        # newest live MuJoCo chord check.  The controller may therefore skip
        # only its duplicate target profiling after independently checking the
        # sample's discrete position/velocity/acceleration envelope.
        if not (
            command_follows_time_law or command_is_dynamically_retimed
        ):
            # Once feedback projection, a retry, or a carry brake changes one
            # sample, the remaining wall-clock quintic is no longer continuous
            # with the controller's accepted target history.  Keep the rest of
            # this edge on the ordinary fail-closed limiter; the next certified
            # corner starts a fresh exact stop-to-stop sequence.
            self._safe_edge_passthrough_eligible = False
        self._arm_command_is_time_parameterized = bool(
            self._safe_edge_passthrough_eligible or feedback_connector
        )
        return command.copy()

    def _reset_lift_feedback_correction(self, reason: str) -> None:
        """Discard integral state while retaining the last command evidence."""

        self._lift_feedback_correction_m = 0.0
        self._lift_feedback_goal_key = None
        self._lift_feedback_last_timestamp = None
        self._lift_feedback_pending_correction_m = None
        self._lift_feedback_correction_diagnostics.update(
            active=False, integral_correction_m=0.0, last_reset_reason=reason,
        )

    def _lift_feedback_command_target(
        self, nominal_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Add a bounded Z integral to a request, leaving its goal immutable.

        The previous IK-accepted Cartesian request must already have reached
        the nominal height. Consequently the original long upward move is
        unchanged and cannot wind up the integral. Every corrected request
        still uses the ordinary Cartesian profile, IK and full edge checks.
        State is committed separately only after those checks accept it.
        """

        enabled = bool(self._lift_feedback_correction_enabled)
        active = bool(enabled and self._clearing_payload())
        if not active:
            self._reset_lift_feedback_correction("disabled" if not enabled else "clearance_inactive")
            self._lift_feedback_correction_diagnostics.update(enabled=enabled)
            return nominal_pose

        now = self._timestamp_without_advancing(observation)
        key = tuple(np.asarray(nominal_pose, dtype=float))
        goal_changed = self._lift_feedback_goal_key is not None and key != self._lift_feedback_goal_key
        time_reversed = self._lift_feedback_last_timestamp is not None and now < self._lift_feedback_last_timestamp
        if goal_changed or time_reversed:
            self._reset_lift_feedback_correction("goal_changed" if goal_changed else "time_reversed")
        self._lift_feedback_goal_key = key
        duplicate = self._lift_feedback_last_timestamp == now
        self._lift_feedback_last_timestamp = now

        previous = self._cartesian_command_pose
        nominal_z = float(nominal_pose[2])
        measured_z = float(self._ee_pose(observation)[2])
        eligible = bool(
            not goal_changed and not time_reversed
            and previous is not None
            and float(previous[2]) >= nominal_z - 1e-9
        )
        dt = self._execution_dt(observation)
        candidate = self._lift_feedback_correction_m
        if eligible and not duplicate:
            candidate = float(np.clip(
                candidate + self.config.lift_feedback_integral_gain_s_inv * dt * (nominal_z - measured_z),
                0.0, self.config.lift_feedback_max_correction_m,
            ))
        elif not eligible:
            # In particular, a new route must reach its own nominal request.
            candidate = 0.0
            self._lift_feedback_correction_m = 0.0
        self._lift_feedback_pending_correction_m = candidate
        command_pose = nominal_pose.copy()
        command_pose[2] += candidate
        self._lift_feedback_correction_diagnostics.update(
            enabled=enabled, active=True, eligible=eligible, duplicate_timestamp=duplicate,
            accepted=False, timestamp_s=now, control_dt_s=dt,
            nominal_goal_pose=nominal_pose.copy(), nominal_goal_z_m=nominal_z,
            feedback_z_m=measured_z, feedback_height_error_m=nominal_z - measured_z,
            previous_accepted_request_z_m=None if previous is None else float(previous[2]),
            requested_z_m=float(command_pose[2]), candidate_correction_m=candidate,
            integral_correction_m=self._lift_feedback_correction_m,
            integral_gain_s_inv=self.config.lift_feedback_integral_gain_s_inv,
            maximum_correction_m=self.config.lift_feedback_max_correction_m,
            object_ground_truth_used=False,
        )
        return command_pose

    def _commit_lift_feedback_correction(self, accepted_pose: np.ndarray) -> None:
        """Back-calculate the integral from the accepted profiled request."""

        candidate = self._lift_feedback_pending_correction_m
        if candidate is None:
            return
        evidence = self._lift_feedback_correction_diagnostics
        admitted = max(0.0, float(accepted_pose[2]) - evidence["nominal_goal_z_m"])
        self._lift_feedback_correction_m = min(candidate, admitted)
        self._lift_feedback_pending_correction_m = None
        evidence.update(
            accepted=True, accepted_request_z_m=float(accepted_pose[2]),
            integral_correction_m=self._lift_feedback_correction_m,
            rate_saturation_backcalculated=bool(admitted + 1e-12 < candidate),
        )

    def _start_capture_high_handoff(self, pregrasp, target, observation) -> None:
        """Continue the original capture-column route from measured XY.

        Visual acquisition may select its existing following allocator when
        the reserved capture has moved downstream of this high endpoint.
        The caller passes the endpoint because the transition clears its path.
        """
        self._dynamic_vertical_retry = True

    def _arm_for_pose(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        full = self._arm_on_full_pick(observation)
        if full is not None:
            return full
        continuous = self._arm_on_continuous_intercept(observation)
        if continuous is not None:
            return continuous
        timed = self._arm_on_timed_descent(observation)
        if timed is not None:
            return timed
        high_hold = self._arm_on_descent_high_hold(observation)
        if high_hold is not None:
            return high_hold
        if not self._clearing_payload():
            self._reset_lift_feedback_correction("clearance_inactive")
        if self.config.action_mode == "joint_position":
            if not self.config.enable_time_parameterization:
                solved = self._solve_pose(target_pose, observation)
                if solved is not None:
                    return solved
                self._last_error = (
                    self._last_error
                    or "joint_position mode requires a converged IK solution"
                )
                return self._joint_positions(observation)

            clearance_replan = self._arm_for_blocked_clearance(
                target_pose, observation
            )
            if clearance_replan is not None:
                return clearance_replan

            if (
                self.config.enable_global_transfer_planner
                and not self._clearing_payload()
                and self._stage
                in {
                    PolicyStage.LIFTING,
                    PolicyStage.TRANSFERRING,
                    PolicyStage.RETURNING,
                    PolicyStage.RECOVERING,
                }
            ):
                return self._arm_along_global_path(target_pose, observation)

            if (
                self.config.enable_global_transfer_planner
                and self._stage == PolicyStage.PREGRASP
                and not self._pregrasp_ready
            ):
                return self._arm_along_global_path(target_pose, observation)

            current_pose = self._ee_pose(observation)
            vertical_retry = bool(
                self.config.dynamic_intercept_tracking
                and self._stage == PolicyStage.TRACKING_DESCENT
                and self._dynamic_vertical_retry
                and current_pose[2] > target_pose[2] + 0.015
            )
            self._dynamic_vertical_retry = vertical_retry
            if vertical_retry:
                # An upstream correction can leave the validated descent
                # corridor near the inner reach boundary. After a rejected
                # chord, descend from measured XY until near capture height.
                # Each bounded step passes exactly the same IK/edge guards;
                # no second expensive solve or branch jump is used.
                target_pose = target_pose.copy()
                target_pose[:2] = current_pose[:2]
            # The global recovery preserves the captured XY and orientation.
            # Keep the old downstream-only retry for explicitly disabled A/B.
            if not (
                self.config.lift_clearance_global_replan_enabled
                and self.config.enable_global_transfer_planner
            ):
                target_pose = self._dynamic_lift_retry_pose(target_pose, observation)
            target_pose = self._lift_feedback_command_target(target_pose, observation)
            current_alignment = self._downward_alignment(current_pose[3:])
            target_alignment = self._downward_alignment(target_pose[3:])
            self._minimum_observed_downward_alignment = min(
                current_alignment,
                target_alignment,
                (
                    self._minimum_observed_downward_alignment
                    if self._minimum_observed_downward_alignment is not None
                    else float("inf")
                ),
            )
            if not self._orientation_is_safe(target_pose[3:]):
                self._last_error = (
                    "unsafe tool orientation rejected "
                    f"(downward_alignment={target_alignment:.3f})"
                )
                self._record_motion_plan_failure()
                return self._joint_positions(observation)

            # Lightweight MoveIt-Servo-style slowdown near singularities.  It
            # is deliberately applied to the Cartesian command before IK, not
            # to a precomputed endpoint joint vector.
            current = self._joint_positions(observation)
            reference = (
                current
                if self._cartesian_command_pose is None
                or self._last_ik_solution is None
                else self._last_ik_solution.copy()
            )
            condition = self._jacobian_condition(current)
            self._last_jacobian_condition = condition
            scale = 1.0
            if (
                condition is not None
                and condition > self.config.singularity_slowdown_condition
            ):
                span = (
                    self.config.singularity_hard_stop_condition
                    - self.config.singularity_slowdown_condition
                )
                scale = float(
                    np.clip(
                        (
                            self.config.singularity_hard_stop_condition
                            - condition
                        )
                        / span,
                        0.12,
                        1.0,
                    )
                )

            # If a full increment has poor local conditioning, retry smaller
            # increments.  Every solve is warm-started from the measured state,
            # and large IK branch jumps or soft-limit violations are rejected.
            realtime_mode = bool(
                self._planning_worker is not None
                or callable(
                    getattr(self.env, "create_global_planning_worker", None)
                )
            )
            if realtime_mode:
                # Preserve the old robust full/half/quarter Cartesian retry,
                # but spread it over three 20 ms ticks.  This restores small-
                # chord reachability without ever running multiple 260-step DLS
                # solves in one viewer/control callback.
                previous_error = self._last_error or ""
                acquisition_servo = self._stage in {
                    PolicyStage.TRACKING_DESCENT,
                    PolicyStage.CLOSING,
                } or self._clearing_payload() or (
                    self._stage == PolicyStage.RELEASING
                    and callable(getattr(self.env, 'get_release_pose', None))
                )
                # Once the global planner has delivered a continuous
                # PREGRASP branch, final acquisition is a local servo problem.
                # Never jump back to a calibration seed beside a moving part:
                # after one rejected chord that produced a repeatable
                # 0.25--0.29 rad branch-jump loop and let the belt escape.
                # Re-anchor both Cartesian integration and DLS at the latest
                # measured state, then spread full/half/quarter steps over
                # successive ticks.  Transit stages may still use one bounded
                # calibration seed per tick for redundancy recovery.
                needs_redundancy_retry = bool(
                    not acquisition_servo
                    and self._motion_plan_failure_count > 0
                    and any(
                            marker in previous_error
                            for marker in (
                                "branch jump",
                                "local Cartesian edge",
                                "soft joint limit",
                            )
                    )
                )
                calibration_seed = (
                    self._calibrated_ik_seed_for_retry(
                        self._motion_plan_failure_count - 1
                    )
                    if needs_redundancy_retry
                    else None
                )
                retry_factor = (
                    1.0
                    if calibration_seed is not None
                    else (1.0, 0.5, 0.25)[
                        min(self._motion_plan_failure_count, 2)
                    ]
                )
                retry_scales = (scale * retry_factor,)
                servo_seed = (
                    current
                    if acquisition_servo
                    else (
                        calibration_seed
                        if calibration_seed is not None
                        else reference
                    )
                )
            else:
                calibration_seed = None
                servo_seed = reference
                retry_scales = (scale, 0.5 * scale, 0.25 * scale)
            if self._clearing_payload():
                # Attribute the next failure to these lift attempts only.
                # Some optional IK adapters return None without setting an
                # error; that must not inherit an old CLOSING soft-limit text.
                self._lift_clearance_local_rejection = None
                self._last_error = None
            for retry_scale in retry_scales:
                servo_pose = self._cartesian_servo_pose(
                    target_pose,
                    observation,
                    scale=retry_scale,
                )
                servo_alignment = self._downward_alignment(servo_pose[3:])
                # Dynamic tracking error can put the measured tool just
                # outside the preferred cone.  Permit only a monotonic move
                # back toward safety; never deadlock by requiring a single IK
                # increment to cross the boundary instantaneously.
                alignment_limit = self._local_downward_alignment_limit()
                minimum_servo_alignment = (
                    alignment_limit
                    if current_alignment >= alignment_limit
                    else max(0.0, current_alignment - 1e-6)
                )
                if servo_alignment < minimum_servo_alignment:
                    continue
                solved = self._solve_pose(
                    servo_pose,
                    observation,
                    qpos_seed=servo_seed,
                    enforce_safety=True,
                    allow_verified_transition=(
                        calibration_seed is not None
                    ),
                )
                if solved is not None:
                    self._cartesian_command_pose = servo_pose.copy()
                    self._lift_clearance_local_rejection = None
                    self._commit_lift_feedback_correction(servo_pose)
                    self._record_motion_plan_success()
                    return solved

            # A newly arriving part can pre-empt the return trajectory while
            # the seven-axis arm is on a different redundancy branch.  The
            # local servo must not accept a large jump; use the same constrained
            # global planner for the fixed pregrasp waypoint instead.
            if (
                self.config.enable_global_transfer_planner
                and self._stage == PolicyStage.PREGRASP
            ):
                return self._arm_along_global_path(target_pose, observation)

            if realtime_mode and acquisition_servo:
                # The previous command-space integrator may be ahead of a
                # plant that just held on an unsafe edge.  The next reduced
                # retry must start from measured TCP/q, not continue advancing
                # an unexecuted Cartesian command.
                self._cartesian_command_pose = None
                self._last_ik_solution = None
                self._dynamic_vertical_retry = bool(
                    self.config.dynamic_intercept_tracking
                    and self._stage == PolicyStage.TRACKING_DESCENT
                    and current_pose[2] > target_pose[2] + 0.015
                )
            self._record_motion_plan_failure()
            self._last_error = self._last_error or "safe Cartesian IK failed"
            if self._clearing_payload() and any(
                marker in self._last_error
                for marker in ("soft joint limit", "branch jump")
            ):
                self._lift_clearance_local_rejection = self._last_error
            return current

        current = self._ee_pose(observation)
        translation = target_pose[:3] - current[:3]
        distance = float(np.linalg.norm(translation))
        if (
            self.config.tcp_speed_limits_enabled
            and distance > self.config.max_translation_step
        ):
            translation *= self.config.max_translation_step / distance
        rotation = quat_error_vector(target_pose[3:], current[3:])
        angle = float(np.linalg.norm(rotation))
        if (
            self.config.tcp_speed_limits_enabled
            and angle > self.config.max_rotation_step
        ):
            rotation *= self.config.max_rotation_step / angle
        return np.r_[translation, rotation]

    def _hold_arm(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self.config.action_mode == "ee_delta_pose":
            return np.zeros(6, dtype=float)
        return self._joint_positions(observation)

    def _descent_closure_qualification(
        self, target: Any, observation: Mapping[str, Any], now: float,
        capture_pose: np.ndarray,
    ) -> tuple[bool, Any]:
        """Evaluate the original closure predicate without a task transition."""

        if self._full_pick_enabled() and not self._full_pick_contact_handed_off:
            bundle = self._full_pick_bundle
            if (bundle is None or self._full_pick_abort_reason is not None
                    or now < bundle.contact_time-self.config.close_lead_time
                    or now >= bundle.lift_time):
                return False, None
        if self._continuous_intercept_enabled() and not self._continuous_armed:
            # The new route must have delivered a complete certificate and an
            # accepted arm sample before its first close can be considered.
            return False, None
        closure_ready = self._window_current(target, observation)
        closure_time_ready = self._closure_time_ready(
            target, observation, now
        )
        predictive_closure = bool(
            _value(
                target,
                "allow_predictive_closure_outside_window",
                False,
            )
        )
        closure_pose_ready = self._closure_pose_reached(
            capture_pose, observation, target
        )
        closure_arm_settled = bool(
            not bool(_value(target, "closure_requires_arm_settle", False))
            or self._joint_velocity_is_settled(
                observation,
                self.config.planner_waypoint_velocity_tolerance,
            )
        )
        physical_intercept_contact = bool(
            closure_pose_ready
            and closure_arm_settled
            and self._external_finger_contact(observation)
            and closure_time_ready
        )
        visual_intercept = self._visual_closure_intercept(
            target, observation, now
        )
        if visual_intercept is not None:
            closure_pose_ready = visual_intercept[0]
            # Visual alignment can replace an obsolete untimed intercept, but
            # cannot finish an active, certified descent half a second early.
            # Keep its EXISTING close_lead_time; keep following while waiting.
            closure_time_ready = bool(visual_intercept[0] and (
                self._descent_plan is None or closure_time_ready))
            closure_ready = visual_intercept[0]
            physical_intercept_contact = False
        ready = bool(
            physical_intercept_contact
            or (
                closure_pose_ready
                and closure_arm_settled
                and (
                    closure_ready is not False
                    or (
                        predictive_closure
                        and closure_time_ready
                        and self._window_valid(target, observation)
                    )
                )
                and closure_time_ready
            )
        )
        return ready, visual_intercept

    def _admit_descent_closure(
        self,
        target: Any,
        observation: Mapping[str, Any],
        now: float,
        capture_pose: np.ndarray,
        tracking_pose: np.ndarray,
        tracking_reference_before_update: np.ndarray | None,
    ) -> np.ndarray:
        """Admit the original non-deferred simulation close and motion reference."""

        ready, visual_intercept = self._descent_closure_qualification(
            target, observation, now, capture_pose
        )
        if ready:
            # A pad touch can advance closure only after the ordinary pose
            # gate has been met.  It may bypass a stale visual window/time,
            # but a high or early side brush cannot trigger closure.  XY is
            # intentionally not frozen until bilateral pad contact, so the
            # still-moving part remains centred during the closing stroke.
            self._transition(
                PolicyStage.CLOSING,
                now,
                reason="grasp_pose_reached",
                event="gripper_close_started",
            )
            if visual_intercept is not None:
                self._closure_intercept_at = visual_intercept[1]
                # The first jaw command must already use the newly
                # admitted local crossing, just like later CLOSING ticks.
                # Retiming only the feed-forward clock would combine it
                # with the old route's future pose and count travel twice.
                closing_reference = self._grasp_target_pose(
                    target, observation
                )
                if closing_reference is not None:
                    reference_origin = (
                        self._grasp_pose
                        if tracking_reference_before_update is None
                        else tracking_reference_before_update
                    )
                    correction = (
                        closing_reference[:3] - reference_origin[:3]
                    )
                    distance = float(np.linalg.norm(correction))
                    if distance > self.config.tracking_target_max_step:
                        closing_reference = closing_reference.copy()
                        closing_reference[:3] = (
                            reference_origin[:3]
                            + correction
                            * self.config.tracking_target_max_step
                            / distance
                        )
                    self._grasp_pose = self._guard_final_tracking_xy(
                        closing_reference, target, observation
                    )
                    tracking_pose = self._tracking_command_target(
                        self._grasp_pose,
                        observation,
                        lead_time=self.config.closing_tracking_lead_time,
                        target=target,
                        synchronize_to_intercept=True,
                    )
        return tracking_pose

    def _action(
        self,
        observation: Mapping[str, Any],
        *,
        target_pose: np.ndarray | None,
        gripper: float,
    ) -> dict[str, Any]:
        profile = "carrying" if self._gentle_carry_required() else "nominal"
        if bool(_value(observation, "_defer_arm_motion", False)):
            context = self._deferred_closure_context or {}
            self._deferred_closure_context = None
            self._pending_arm_motion = {
                "target_pose": None if target_pose is None else target_pose.copy(),
                "gripper": float(gripper),
                "profile": profile,
                "stage": self._stage,
                "target_id": self._target_id,
                "reference_target": deepcopy(
                    self._find_target(observation, self._target_id)
                    if self._target_id is not None else None
                ),
                **context,
            }
            # The runner must finalize before sending.  No IK, trajectory,
            # integral, actuator target or motion profile is advanced here.
            return {
                "arm": self._hold_arm(observation),
                "gripper": float(gripper),
                "_arm_motion_pending": True,
            }
        _hook(self.env, ("set_motion_profile",), ((profile,),))
        self._arm_command_is_time_parameterized = False
        arm = (
            self._hold_arm(observation)
            if target_pose is None
            else self._arm_for_pose(target_pose, observation)
        )
        arm = self._guard_contact_handoff_command(observation, arm)
        result = {"arm": arm, "gripper": float(gripper)}
        if self._arm_command_is_time_parameterized:
            result["arm_time_parameterized"] = True
        self._last_action = result
        return result

    def _refresh_pending_motion_target(
        self,
        target_pose: np.ndarray | None,
        observation: Mapping[str, Any],
        intent: dict[str, Any],
    ) -> np.ndarray | None:
        """Static targets keep their original intent; moving mixins override."""

        return target_pose

    def finalize_pending_motion(
        self, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Consume one policy intent using the last received feedback.

        This deliberately does not call ``act`` again.  Stage timers, contact
        confirmation and target selection therefore advance once per cycle;
        First-close admission is left for the sending boundary after IK and
        geometry validation. This motion is still an open-jaw descent sample.
        """

        intent = self._pending_arm_motion
        if intent is None:
            raise RuntimeError("no deferred arm motion to finalize")
        self._pending_arm_motion = None
        target_pose = self._refresh_pending_motion_target(
            intent["target_pose"], observation, intent
        )
        gripper = float(intent["gripper"])
        _hook(self.env, ("set_motion_profile",), ((intent["profile"],),))
        self._arm_command_is_time_parameterized = False
        arm = (
            self._hold_arm(observation) if target_pose is None
            else self._arm_for_pose(target_pose, observation)
        )
        arm = self._guard_contact_handoff_command(observation, arm)
        result = {"arm": arm, "gripper": gripper}
        if (self._stage == PolicyStage.TRACKING_DESCENT
                and "capture_pose" in intent and target_pose is not None):
            intent["close_gripper"] = self._grasp_gripper_action
            result["_dispatch_closure_intent"] = intent
        if self._arm_command_is_time_parameterized:
            result["arm_time_parameterized"] = True
        self._last_action = result
        return result

    def dispatch_closure_status(
        self, action: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Read closure eligibility at sending time; no IK/filter/task step."""

        pending = action.get("_dispatch_closure_intent")
        if not isinstance(pending, Mapping) or self._stage != PolicyStage.TRACKING_DESCENT:
            return {"ready": False}
        intent = dict(pending)
        self._refresh_pending_motion_target(intent["target_pose"], observation, intent)
        target = intent.get("refreshed_target")
        if target is None:
            target = self._find_target(observation, intent["target_id"])
        if target is None:
            return {"ready": False}
        ready, intercept = self._descent_closure_qualification(
            target, observation, self._timestamp_without_advancing(observation),
            intent["capture_pose"],
        )
        return {"ready": ready, "visual_intercept": intercept,
                "capture_pose": intent["capture_pose"], "target": target,
                "gripper": float(intent["close_gripper"])}

    def commit_dispatch_closure(
        self, action: dict[str, Any], observation: Mapping[str, Any]
    ) -> bool:
        """Commit one first-close decision immediately before the jaw send.

        The arm sample has already passed IK, rate and geometry checks as an
        open-jaw following step. Keep that accepted Cartesian origin; the next
        CLOSING tick computes its ordinary fresh closing reference.
        """

        status = self.dispatch_closure_status(action, observation)
        action.pop("_dispatch_closure_intent", None)
        if not status["ready"]:
            return False
        accepted = None if self._cartesian_command_pose is None else self._cartesian_command_pose.copy()
        solved = None if self._last_ik_solution is None else self._last_ik_solution.copy()
        now = self._timestamp_without_advancing(observation)
        self._transition(PolicyStage.CLOSING, now,
                         reason="grasp_pose_reached", event="gripper_close_started")
        self._cartesian_command_pose = accepted
        self._last_ik_solution = solved
        self._grasp_pose = np.asarray(status["capture_pose"], dtype=float).copy()
        if status["visual_intercept"] is not None:
            self._closure_intercept_at = status["visual_intercept"][1]
        action["gripper"] = status["gripper"]
        self._last_action = action
        return True

    def _build_transfer_waypoints(
        self, observation: Mapping[str, Any]
    ) -> None:
        assert self._place_pose is not None
        current = self._ee_pose(observation)
        safe_height = max(
            self.config.safe_transfer_height,
            float(current[2]),
            float(self._place_pose[2]),
        )
        # Do not demand a vertical lift at the grasp x/y while preserving the
        # workpiece yaw.  Near the centre of NERO's belt reach that pose drives
        # joint 7 (and, for some yaws, joint 6) onto its hard stop even though
        # the calibrated high waiting corridor is directly reachable.  Route
        # through that known-safe corridor instead; every delivered edge still
        # receives the same dense collision, cone, singularity and soft-limit
        # validation before execution.
        staging = self._wait_target(observation)
        staging[2] = safe_height
        above_bin = self._place_pose.copy()
        above_bin[2] = safe_height
        candidates = [
            staging,
            above_bin,
            self._place_pose.copy(),
        ]
        # Do not hand the asynchronous planner a zero-length duplicate leg.
        # In the default three bins ``above_bin`` and the final release pose
        # are exactly the same (both at the configured release height); planning
        # both forced a needless stop, generation bump and worker cancellation.
        # Keep a real
        # above-bin-to-place descent whenever either position or orientation
        # differs.
        waypoints: list[np.ndarray] = []
        for candidate in candidates:
            if waypoints and (
                np.linalg.norm(candidate[:3] - waypoints[-1][:3]) <= 1e-9
                and quat_distance(candidate[3:], waypoints[-1][3:]) <= 1e-9
            ):
                continue
            waypoints.append(candidate.copy())
        # LIFTING may already have ended inside the calibrated corridor.  The
        # previous implementation still submitted a route to a first waypoint
        # within the ordinary pose tolerance, then cancelled it one tick later.
        # Skip only the leading waypoint under the exact same measured-pose
        # predicate used by the state machine; all later geometric legs remain.
        if len(waypoints) > 1 and self._pose_reached(
            waypoints[0], observation
        ):
            waypoints.pop(0)
        self._transfer_waypoints = waypoints
        self._transfer_waypoint_index = 0

    def _transfer_target(self) -> np.ndarray:
        if not self._transfer_waypoints:
            assert self._place_pose is not None
            return self._place_pose.copy()
        index = min(
            self._transfer_waypoint_index,
            len(self._transfer_waypoints) - 1,
        )
        return self._transfer_waypoints[index].copy()

    def _build_return_waypoints(
        self, observation: Mapping[str, Any]
    ) -> None:
        self._previous_return_alignment = None
        self._return_alignment_stable_steps = 0
        self._return_acquisition_latched = False
        current = self._ee_pose(observation)
        wait = self._wait_target(observation)
        safe_height = max(
            self.config.safe_transfer_height,
            float(current[2]),
            float(wait[2]),
        )
        # Open over the bin, then withdraw horizontally toward the robot before
        # lifting.  A direct vertical retreat can carry an asymmetric part on
        # the outside of the NERO finger mesh even after bilateral contact is
        # gone.  The short +Y withdrawal slides the finger out while the part
        # remains over the bin under gravity.
        retreat = current.copy()
        retreat[1] += self.config.release_retreat_distance
        above_wait = wait.copy()
        above_wait[2] = safe_height
        above_wait[3:] = self._grasp_quaternion
        # Keep the explicit high transit target: it gives the return planner a
        # collision-clear route over the conveyor rail.  Upstream perception
        # lead, rather than removing this safety waypoint, supplies the 10 s
        # cadence margin.
        self._return_waypoints = (
            [retreat, above_wait, wait]
            if self.config.release_retreat_distance > 0.0
            else [above_wait, wait]
        )
        self._return_waypoint_index = 0

    def _return_target(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not self._return_waypoints:
            self._build_return_waypoints(observation)
        index = min(
            self._return_waypoint_index,
            len(self._return_waypoints) - 1,
        )
        return self._return_waypoints[index].copy()

    def _wait_target(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._wait_pose is None:
            pose = _hook(
                self.env,
                ("get_wait_pose", "get_home_pose"),
                ((observation,), ()),
            )
            self._wait_pose = _as_pose(pose, self._grasp_quaternion)
        if self._wait_pose is None:
            self._wait_pose = self._ee_pose(observation)
        return self._wait_pose.copy()

    def _arm_for_blocked_clearance(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray | None:
        """Use the existing global planner after a local lift branch blocks.

        A captured 7-DOF configuration need not share the candidate's earlier
        vertical IK branch.  Replan only after that branch was actually
        rejected; ordinary clearances retain their original local servo.
        """

        if not (
            self.config.lift_clearance_global_replan_enabled
            and self.config.enable_global_transfer_planner
            and self._clearing_payload()
        ):
            self._lift_clearance_global_replan_active = False
            return None
        if not self._lift_clearance_global_replan_active:
            if self._lift_clearance_global_replan_attempted or not any(
                marker in (self._lift_clearance_local_rejection or "")
                for marker in ("soft joint limit", "branch jump")
            ):
                return None
            reason = self._lift_clearance_local_rejection
            self._invalidate_motion_plan()
            self._lift_clearance_global_replan_attempted = True
            self._lift_clearance_global_replan_active = True
            self._lift_clearance_global_replan_diagnostics = {
                "trigger": reason,
                "start_timestamp_s": self._timestamp_without_advancing(observation),
                "start_joint_positions": self._joint_positions(observation).copy(),
                "start_tcp_pose": self._ee_pose(observation).copy(),
                "nominal_goal_pose": target_pose.copy(),
                "object_ground_truth_used": False,
                "active": True,
                "completed": False,
            }

        command = self._arm_along_global_path(target_pose, observation)
        if (self._full_pick_contact_handed_off and self._safe_joint_path
                and self._safe_path_target_pose is not None):
            # Commit only the endpoint of an installed, certified route.
            self._lift_pose=self._safe_path_target_pose.copy()
            self._lift_clearance_pose=self._lift_pose.copy()
            self._lift_clearance_global_replan_diagnostics['certified_goal_pose']=self._lift_pose.copy()
        endpoint = self._safe_joint_path[-1] if self._safe_joint_path else None
        reached = bool(
            endpoint is not None
            and self._safe_joint_path_index >= len(self._safe_joint_path) - 1
            and np.max(np.abs(self._joint_positions(observation) - endpoint))
            <= self.config.planner_corner_settle_tolerance
            and self._joint_velocity_is_settled(
                observation, self.config.planner_waypoint_velocity_tolerance
            )
        )
        evidence = self._lift_clearance_global_replan_diagnostics
        evidence["path_installed"] = bool(self._safe_joint_path)
        evidence["path_corner_count"] = len(self._safe_joint_path)
        if reached:
            # The endpoint already certified the nominal Cartesian request.
            # Return to the bounded feedback servo if payload sag still needs
            # its small terminal correction, without restarting the timeout.
            accepted_endpoint = endpoint.copy()
            self._invalidate_motion_plan()
            self._cartesian_command_pose = target_pose.copy()
            self._last_ik_solution = accepted_endpoint
            self._lift_clearance_global_replan_active = False
            evidence.update(
                active=False,
                completed=True,
                completed_timestamp_s=self._timestamp_without_advancing(observation),
                completed_feedback_pose=self._ee_pose(observation).copy(),
            )
        return command

    def _dynamic_lift_retry_pose(
        self, target_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Escape a rejected vertical lift along the belt, after first rising.

        Keep transverse position and yaw while moving downstream. Every
        incremental chord still passes local IK, collision and joint guards;
        this changes the route, never the acceptance envelopes. One bounded
        attempt avoids oscillating between Cartesian targets on each retry.
        """
        if (
            not self.config.dynamic_intercept_tracking
            or not self._clearing_payload()
            or self._dynamic_lift_repositioned
            or self._secured_hold_pose is None
            or not any(s in (self._last_error or "") for s in ("soft joint limit", "branch jump"))
        ):
            return target_pose
        current = self._ee_pose(observation)
        cfg = _value(self.env, "scene_config", {}).get("workspace", {}).get("dynamic_intercept", {})
        vertical = float(cfg.get("lift_vertical_distance_m", .06))
        offset = float(cfg.get("lift_downstream_offset_m", .08))
        if current[2] < self._secured_hold_pose[2] + vertical:
            return target_pose
        bounds = self._belt_bounds("tracking_s_range", "tracking_x_range", default=(-.15, .25))
        direction = self._belt_frame()[1]
        travel_sign = -1. if float(self._belt_velocity(observation) @ direction) < 0. else 1.
        current_s = self._belt_coordinate(current[:3])
        target_s = self._belt_coordinate(target_pose[:3])
        coordinate = (min(bounds[1], max(current_s, target_s) + offset)
                      if travel_sign > 0.
                      else max(bounds[0], min(current_s, target_s) - offset))
        if travel_sign * (coordinate - target_s) <= 1e-6:
            return target_pose
        self._lift_clearance_pose = target_pose.copy()
        self._lift_clearance_pose[:3] += (coordinate - target_s) * direction
        self._lift_pose = self._lift_clearance_pose.copy()
        self._dynamic_lift_repositioned = True
        return self._lift_clearance_pose.copy()

    def _lift_stopping_reference(
        self, measured_pose: np.ndarray, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Choose lift XY from the actuator's accepted command braking model.

        Keep the measured capture pose separately, including its put-back
        reference.  The stopping FK only chooses an alternate lift target;
        every actual command still receives the ordinary IK/path/rate guards.
        """

        evidence: dict[str, Any] = {
            "enabled": self.config.lift_stopping_reference_enabled,
            "applied": False,
            "measured_capture_pose": measured_pose.copy(),
            "object_ground_truth_used": False,
            "is_exact_physical_stopping_prediction": False,
        }
        self._lift_stopping_reference_diagnostics = evidence
        if not self.config.lift_stopping_reference_enabled:
            evidence["reason"] = "disabled"
            return measured_pose
        if (
            _value(observation, ("joint_target", "joint_command_target"), None) is None
            or _value(observation, ("joint_target_velocity", "joint_command_velocity"), None) is None
        ):
            evidence["reason"] = "accepted_controller_history_unavailable"
            return measured_pose
        target, velocity = self._joint_command_history(
            observation, self._joint_positions(observation)
        )
        controller = _value(self.env, ("robot_controller", "controller"))
        raw_acceleration = _value(controller, "max_joint_acceleration", None)
        controller_config = _value(controller, "config", None)
        raw_period = _value(controller_config, "control_timestep", None)
        try:
            acceleration = np.broadcast_to(
                np.asarray(raw_acceleration, dtype=float), (7,)
            ).copy()
            period = float(raw_period)
            dt = self._execution_dt(observation)
            valid = bool(
                np.all(np.isfinite(target)) and np.all(np.isfinite(velocity))
                and np.all(np.isfinite(acceleration)) and np.all(acceleration > 0.0)
                and np.isfinite(period) and period > 0.0
                and abs(period - dt) <= 1e-9
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            evidence["reason"] = "controller_braking_model_unavailable_or_period_changed"
            return measured_pose
        # A carrying profile can change the controller's acceleration limits
        # immediately after capture.  Until that future profile is observable,
        # do not pretend the closing profile predicts its stopping distance.
        category = _value(self._target, ("category_id", "class_id"), -1)
        try:
            gentle = int(category) in self.config.gentle_carry_categories
        except (TypeError, ValueError):
            gentle = False
        if gentle and self.config.carrying_acceleration_scale != 1.0:
            evidence["reason"] = "acceleration_changes_on_lift"
            return measured_pose
        stopping_q = target + discrete_braking_displacement(velocity, acceleration, dt)
        solver = _value(self.env, ("ik", "ik_solver"))
        stopping_pose = _hook(solver, ("forward_pose",), ((stopping_q,),))
        try:
            stopping_pose = np.asarray(stopping_pose, dtype=float).reshape(7)
        except (TypeError, ValueError):
            stopping_pose = np.full(7, np.nan)
        if not np.all(np.isfinite(stopping_pose)):
            evidence["reason"] = "stopping_fk_unavailable"
            return measured_pose
        lifted = measured_pose.copy()
        lifted[:2] = stopping_pose[:2]
        evidence.update(
            applied=True, reason="accepted_controller_discrete_brake",
            accepted_joint_target=target, accepted_joint_velocity=velocity,
            acceleration_limits_rad_s2=acceleration, control_period_s=dt,
            stopping_joint_target=stopping_q, stopping_fk_pose=stopping_pose,
            lift_reference_pose=lifted.copy(),
        )
        return lifted

    def _lift_corridor_target(
        self, observation: Mapping[str, Any]
    ) -> np.ndarray:
        """Move a secured part directly into the calibrated safe corridor."""

        if self.config.lift_clearance_height > 0.0 and not self._lift_clearance_complete:
            if self._lift_clearance_pose is None:
                self._lift_clearance_pose = self._lift_stopping_reference(
                    self._ee_pose(observation), observation
                )
                self._lift_clearance_pose[2] = max(
                    self._lift_clearance_pose[2], self.config.lift_clearance_height
                )
            return self._lift_clearance_pose.copy()

        configured = _hook(self.env, ("get_lift_pose",), ((observation,), ()))
        configured_pose = _as_pose(configured, self._grasp_quaternion)
        if configured_pose is not None:
            return configured_pose

        # A speculative C request is rooted at one immutable calibrated
        # joint/TCP pair.  When that lineage exists, LIFT must converge to the
        # paired TCP pose rather than merely entering Cartesian tolerance of a
        # raised bin waypoint: C installation separately requires measured q
        # to settle at the exact anchor.  C itself then performs the certified
        # upward/sideways move to the configured release height.
        if self._carry_prefetch_anchor_q is not None:
            calibration = self._carry_corridor_calibration()
            if calibration is not None:
                return calibration[1].copy()

        current = self._ee_pose(observation)
        target = self._wait_target(observation)
        # The wait calibration supplies the collision-aware XY and tool
        # orientation.  Never command a downward move if contact happened
        # above that calibration, and never carry below the transfer height.
        target[2] = max(
            float(current[2]),
            float(target[2]),
            float(self.config.safe_transfer_height),
            (
                float(self._place_pose[2])
                if self._place_pose is not None
                else -float("inf")
            ),
        )
        return target

    def _clearing_payload(self) -> bool:
        return bool(
            self._stage == PolicyStage.LIFTING
            and self._lift_clearance_pose is not None
            and not self._lift_clearance_complete
        )

    def _severe_collision(self, observation: Mapping[str, Any]) -> bool:
        direct = _value(
            observation,
            ("severe_collision", "unsafe_collision", "collision_abort"),
            False,
        )
        if bool(direct):
            return True
        result = _hook(
            self.env,
            ("has_severe_collision", "collision_abort_required"),
            ((), (observation,)),
        )
        return bool(result)

    def _grasp_feedback(
        self, target: Any, observation: Mapping[str, Any]
    ) -> bool | None:
        if self.config.grasp_confirmation_source == "width_and_force":
            gripper = _value(observation, "gripper", None)
            return bool(
                _value(gripper, "confirmation_source", None) == "width_and_force"
                and _value(gripper, "width_force_holding", None) is True
            )
        target_status = _status_name(target)
        if target_status == "GRASPED":
            return True
        result = _hook(
            self.env,
            ("is_object_grasped", "is_target_grasped"),
            (
                (self._target_id,),
                (self._target_id, observation),
                (target,),
            ),
        )
        if result is not None:
            return bool(result)
        gripper = _value(observation, "gripper")
        if isinstance(gripper, Mapping) or gripper is not None:
            held = _value(
                gripper,
                ("is_holding", "holding", "object_grasped", "grasped"),
            )
            if held is not None:
                return bool(held)
            closed = _value(gripper, ("is_closed", "closed"))
            bilateral = _value(
                gripper,
                ("both_fingers_contact", "bilateral_contact"),
            )
            if bilateral is not None:
                return bool(bilateral) and not bool(closed)
            if closed is True:
                return False
        return None

    @staticmethod
    def _gripper_command_settled(
        observation: Mapping[str, Any], *, tolerance: float = 0.015
    ) -> bool:
        """Whether the measured jaw has reached its adaptive position target."""

        gripper = _value(observation, "gripper")
        measured = _value(gripper, "open_fraction", None)
        target = _value(gripper, "target_open_fraction", None)
        try:
            measured_value = float(measured)
            target_value = float(target)
        except (TypeError, ValueError):
            # External grippers without target telemetry retain the legacy
            # settle-time behavior; the bundled NERO controller always
            # publishes both values.
            return True
        return bool(
            np.isfinite(measured_value)
            and np.isfinite(target_value)
            and abs(measured_value - target_value) <= max(0.0, tolerance)
        )

    def _clear_target(self) -> None:
        # Target-scoped certificates cannot survive into the next workpiece.
        # This clears bookkeeping only; controller dispatch history is intact.
        self._reset_full_pick()
        self._pending_arm_motion = None
        self._deferred_closure_context = None
        self._pregrasp_capture_time_s = None
        self._dynamic_vertical_retry = False
        self._dynamic_lift_repositioned = False
        self._target_id = None
        self._target = None
        self._attempt_started_at = None
        self._last_seen_target_at = None
        self._grasp_pose = None
        self._contact_tracking_xy = None
        self._contact_tracking_reference_xy = None
        self._selected_grasp_quaternion = None
        self._pregrasp_branch_q = None
        self._pregrasp_branch_target_id = None
        self._pregrasp_branch_scene_epoch = None
        self._pregrasp_previous_route = None
        self._trajectory_reuse_diagnostics = {}
        self._pregrasp_ready = False
        self._pregrasp_capture_high_active = False
        self._pregrasp_capture_high_diagnostics = {}
        self._lift_pose = None
        self._lift_clearance_pose = None
        self._lift_clearance_complete = False
        self._lift_clearance_global_replan_active = False
        self._lift_clearance_global_replan_attempted = False
        self._lift_clearance_global_replan_diagnostics = {}
        self._lift_stopping_reference_diagnostics = {}
        self._reset_lift_feedback_correction("target_cleared")
        self._place_pose = None
        self._transfer_waypoints = []
        self._transfer_waypoint_index = 0
        self._return_waypoints = []
        self._return_waypoint_index = 0
        self._last_ik_solution = None
        self._grasp_gripper_action = self.config.gripper_close_action
        self._secured_hold_pose = None
        self._grasp_confirmed_at = None
        self._grasp_force_stable_since = None
        self._grasp_force_last_update_at = None
        self._grasp_force_status = "disabled"
        self._closure_intercept_at = None
        self._bilateral_contact_stable_since = None
        self._previous_return_alignment = None
        self._return_alignment_stable_steps = 0
        self._return_acquisition_latched = False
        self._return_candidate_id = None
        self._return_candidate_stable_steps = 0
        self._return_lookahead_request = None
        self._return_lookahead_result = None
        self._return_lookahead_category = None
        self._return_lookahead_pose = None
        self._return_lookahead_scene_epoch = None
        self._return_lookahead_observed_at = None
        self._return_lookahead_failures = 0
        self._prefetch_terminal_b_target_id = None
        self._pregrasp_timing_refinements = 0
        self._invalidate_motion_plan()
        self._clear_terminal_motion_plan_failure()

    def _grip_action(self, target: Any) -> float:
        """Convert ObjectSpec grip width into the normalized gripper action."""

        if not self.config.adaptive_grip_width:
            return self.config.gripper_close_action
        configured_width = _value(
            target,
            ("recommended_grip_width", "recommended_grasp_width"),
            None,
        )
        if configured_width is None:
            return self.config.gripper_close_action
        robot_config = getattr(self.env, "robot_config", {})
        gripper_config = (
            robot_config.get("gripper", {})
            if isinstance(robot_config, Mapping)
            else {}
        )
        try:
            open_width = float(gripper_config.get("open_width", 0.10))
            closed_width = float(gripper_config.get("closed_width", 0.0))
            margin = float(
                _value(
                    target,
                    ("grip_width_margin", "grasp_width_margin"),
                    self.config.grip_width_margin,
                )
            )
            desired_width = float(configured_width) - margin
        except (TypeError, ValueError):
            return self.config.gripper_close_action
        if not (
            np.isfinite(open_width)
            and np.isfinite(closed_width)
            and np.isfinite(desired_width)
            and open_width > closed_width
        ):
            return self.config.gripper_close_action
        open_fraction = np.clip(
            (desired_width - closed_width) / (open_width - closed_width),
            0.0,
            1.0,
        )
        return float(
            self.config.gripper_close_action
            + open_fraction
            * (
                self.config.gripper_open_action
                - self.config.gripper_close_action
            )
        )

    def _capture_contact_hold_action(self, target: Any) -> None:
        """Convert first bilateral contact into a modest position preload.

        The finger position at first contact can be substantially wider than
        the commanded target because the position actuator is already loading
        the object.  Continuing all the way to the original position target
        can eject a narrow workpiece, while commanding the measured position
        would remove all normal force.  Keep a calibrated fraction of preload
        below the measured contact opening instead.
        """

        if self.config.grasp_confirmation_source == "width_and_force":
            # The AGX driver regulates commanded motor force; a guessed pad
            # preload/position relaxation would change that hardware contract.
            return
        gripper = _value(self.env, "gripper_controller")
        state = _hook(gripper, ("get_state", "read_state"), ((),))
        fraction = _value(state, "open_fraction", None)
        try:
            measured = float(fraction)
        except (TypeError, ValueError):
            return
        if not np.isfinite(measured):
            return
        squeeze = _value(
            target,
            ("contact_hold_squeeze_fraction", "grip_preload_fraction"),
            self.config.contact_hold_squeeze_fraction,
        )
        try:
            squeeze_fraction = float(squeeze)
        except (TypeError, ValueError):
            squeeze_fraction = self.config.contact_hold_squeeze_fraction
        if not np.isfinite(squeeze_fraction):
            squeeze_fraction = self.config.contact_hold_squeeze_fraction
        squeeze_fraction = float(np.clip(squeeze_fraction, 0.0, 1.0))
        relaxation = _value(
            target,
            "contact_hold_max_relaxation_fraction",
            self.config.contact_hold_max_relaxation_fraction,
        )
        try:
            relaxation_fraction = float(relaxation)
        except (TypeError, ValueError):
            relaxation_fraction = (
                self.config.contact_hold_max_relaxation_fraction
            )
        if not np.isfinite(relaxation_fraction):
            relaxation_fraction = (
                self.config.contact_hold_max_relaxation_fraction
            )
        relaxation_fraction = float(np.clip(relaxation_fraction, 0.0, 1.0))
        action_span = (
            self.config.gripper_open_action
            - self.config.gripper_close_action
        )
        commanded_fraction = float(
            np.clip(
                (
                    self._grasp_gripper_action
                    - self.config.gripper_close_action
                )
                / action_span,
                0.0,
                1.0,
            )
        )
        hold_fraction = float(
            np.clip(
                min(
                    measured - squeeze_fraction,
                    commanded_fraction + relaxation_fraction,
                ),
                0.0,
                1.0,
            )
        )
        self._grasp_gripper_action = float(
            self.config.gripper_close_action
            + hold_fraction
            * (
                self.config.gripper_open_action
                - self.config.gripper_close_action
            )
        )

    def _update_grasp_force_preload(
        self, observation: Mapping[str, Any], now: float, feedback: bool | None
    ) -> bool:
        """Tighten an acquired grasp until bilateral force survives a dwell.

        The position command can only move toward closed. A threshold dip
        resets confirmation and adds preload, creating margin against contact
        noise without weakening the required continuous above-threshold dwell.
        Missing force telemetry preserves legacy adapters; malformed telemetry
        never confirms a lift. The existing max-close watchdog bounds failures.
        """

        if self.config.grasp_confirmation_source == "width_and_force":
            self._grasp_force_status = "motor_width_proxy"
            return feedback is True
        threshold = self.config.minimum_grasp_normal_force
        gripper = _value(observation, "gripper", None)
        raw_force = _value(gripper, "minimum_normal_force", None)
        if threshold <= 0.0 or raw_force is None:
            self._grasp_force_stable_since = None
            self._grasp_force_last_update_at = None
            self._grasp_force_status = (
                "disabled" if threshold <= 0.0 else "sensor_unavailable"
            )
            return True
        try:
            force = float(raw_force)
        except (TypeError, ValueError):
            force = float("nan")
        previous = self._grasp_force_last_update_at
        dt = (
            self.config.control_timestep
            if previous is None
            else min(self.config.control_timestep, max(0.0, now - previous))
        )
        if previous is not None and (
            now < previous
            or now - previous > 1.5 * self.config.control_timestep + 1e-9
        ):
            # A telemetry gap is not evidence of continuous sufficient force.
            self._grasp_force_stable_since = None
        self._grasp_force_last_update_at = now
        acquired = self._grasp_confirmed_at is not None
        force_valid = bool(np.isfinite(force) and force >= 0.0)
        bilateral = self._bilateral_finger_contact(observation)
        above_threshold = bool(
            acquired and feedback is True and bilateral and force_valid
            and force + 1e-9 >= threshold
        )
        if not above_threshold:
            self._grasp_force_stable_since = None
            self._grasp_force_status = "preloading" if acquired else "waiting_for_grasp"
            if acquired and force_valid and force < threshold:
                span = self.config.gripper_open_action - self.config.gripper_close_action
                fraction = (self._grasp_gripper_action - self.config.gripper_close_action) / span
                fraction = float(np.clip(fraction - self.config.grasp_force_preload_rate * dt, 0.0, 1.0))
                self._grasp_gripper_action = self.config.gripper_close_action + fraction * span
            return False
        if self._grasp_force_stable_since is None:
            self._grasp_force_stable_since = now
        ready = now - self._grasp_force_stable_since + 1e-9 >= self.config.grasp_force_settle_time
        self._grasp_force_status = "ready" if ready else "settling"
        return ready

    def _begin_recovery(self, now: float, reason: str) -> None:
        # A carry/retreat result is scoped to the target and grasp outcome that
        # created its copied gripper geometry.  A failed attempt must not leave
        # that speculative version available to a recycled logical id.
        self._discard_prefetch(cancel_worker=True)
        self._transition(
            PolicyStage.RECOVERING,
            now,
            reason=reason,
            event="attempt_failed",
        )

    def _timeout_reason(
        self,
        now: float,
        observation: Mapping[str, Any] | None = None,
    ) -> str | None:
        if (
            self._attempt_started_at is not None
            and now - self._attempt_started_at > self.config.attempt_timeout
        ):
            return "attempt_timeout"
        stage_limits = {
            PolicyStage.PREGRASP: self.config.pregrasp_timeout,
            PolicyStage.TRACKING_DESCENT: self.config.descent_timeout,
            PolicyStage.LIFTING: self.config.lift_timeout,
            PolicyStage.TRANSFERRING: self.config.transfer_timeout,
            PolicyStage.RETURNING: self.config.return_timeout,
            PolicyStage.RECOVERING: self.config.recovery_timeout,
        }
        limit = stage_limits.get(self._stage)
        if limit is not None and self._gentle_carry_required():
            # These budgets describe nominal motion. A carrying profile
            # stretches the same certified geometric path: velocity scales
            # linearly with inverse duration and acceleration quadratically.
            # Preserve a finite watchdog, but do not abort a sound grasp just
            # because the deliberately slower lift outlives its nominal ETA.
            time_scale = min(
                self.config.carrying_velocity_scale,
                math.sqrt(self.config.carrying_acceleration_scale),
            )
            limit /= max(time_scale, 1e-6)
        if (
            self._stage == PolicyStage.PREGRASP
            and observation is not None
            and self._target_id is not None
            and self._stage_started_at is not None
        ):
            prediction = self._prediction_from_mapping(
                observation, self._target_id
            )
            planned_time = _value(
                prediction,
                ("estimated_grasp_time", "grasp_time"),
                _value(
                    self._target,
                    ("estimated_grasp_time", "grasp_time"),
                    None,
                ),
            )
            try:
                planned_wait = (
                    float(planned_time)
                    - float(self._stage_started_at)
                    + self.config.control_timestep
                )
            except (TypeError, ValueError):
                planned_wait = 0.0
            if np.isfinite(planned_wait):
                # The extended upstream camera can reserve a slow-belt part
                # more than six seconds before interception.  Reaching the
                # verified high waypoint and waiting there is not a motion
                # deadlock.  Allow that frozen visual clock to govern the
                # PREGRASP wait while the independent grasp-window lifecycle
                # still fails closed at its hard downstream boundary.
                limit = max(float(limit or 0.0), planned_wait)
        if limit is not None and self._elapsed(now) > limit:
            return f"{self._stage.value}_timeout"
        return None

    def act(
        self,
        observation: Mapping[str, Any],
        env: Any | None = None,
    ) -> dict[str, Any]:
        """Advance one policy tick and return a VLA-compatible action dict."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        self._execution_dt(observation)  # Reject a malformed external clock before state changes.
        if env is not None:
            self.env = env
        full_owner = (self._full_pick_bundle.request if self._full_pick_bundle is not None
                      else self._full_pick_request)
        if (self._full_pick_enabled() and full_owner is not None
                and int(observation.get('scene_epoch', self._observed_scene_epoch or 0))
                    != full_owner.scene_epoch):
            return self._full_pick_abort(observation, 'full_pick_scene_changed')
        if self._full_pick_enabled() and self._full_pick_abort_reason is not None:
            return self._full_pick_abort(observation, self._full_pick_abort_reason)
        raw_epoch = _value(observation, "scene_epoch", None)
        if raw_epoch is not None:
            try:
                scene_epoch = int(raw_epoch)
            except (TypeError, ValueError):
                scene_epoch = None
            if scene_epoch is not None:
                previous_epoch = self._observed_scene_epoch
                if (
                    previous_epoch is not None
                    and scene_epoch != previous_epoch
                ):
                    # Reset can be requested by the viewer callback between
                    # control ticks.  Drop every installed/queued version and
                    # restart the state machine before consuming the new scene.
                    self.reset()
                self._observed_scene_epoch = scene_epoch
        now = self._timestamp(observation)
        if self._stage_started_at is None:
            self._stage_started_at = now
        if self._wait_pose is None:
            self._wait_target(observation)

        observed_target = (
            None
            if self._target_id is None
            else self._find_target(observation, self._target_id)
        )
        target = observed_target
        if observed_target is not None:
            if (
                self._stage == PolicyStage.PREGRASP
                and self._target is not None
                and self._same_target_id(
                    _object_id(observed_target), _object_id(self._target)
                )
                and self._lookahead_category(observed_target)
                != self._lookahead_category(self._target)
            ):
                # An upstream partial silhouette can be corrected while the
                # arm is still approaching.  Category owns the jaw width,
                # grasp height/yaw and bin: replace all of those contracts
                # atomically before consuming another old route command.
                # Keep the reservation and overall watchdog; this is neither
                # a second grasp attempt nor a failed physical acquisition.
                # Controller target/velocity history remains in observation
                # and is consumed normally when replanning from measured q.
                attempt_started_at = self._attempt_started_at
                self._discard_prefetch(cancel_worker=True)
                self._clear_target()
                self._target_id = _object_id(observed_target)
                self._attempt_started_at = attempt_started_at
                self._grasp_gripper_action = self._grip_action(observed_target)
                self._transition(
                    PolicyStage.PREGRASP,
                    now,
                    reason="visual_category_revised",
                    event="category_revised",
                )
            self._target = observed_target
            self._last_seen_target_at = now
        elif (
            self._contact_tracking_xy is not None
            and self._stage
            in {PolicyStage.TRACKING_DESCENT, PolicyStage.CLOSING}
        ):
            # Once physical acquisition has begun, a tracker tombstone cannot
            # make the controller abandon the contacted part and command an
            # open-jaw return.  Continue from the last policy-visible target;
            # no pose truth is queried or synthesized here.
            target = self._target
        elif (
            self._target_id is not None
            and self._stage
            in {
                PolicyStage.PREGRASP,
                PolicyStage.TRACKING_DESCENT,
                PolicyStage.CLOSING,
            }
            and self._last_seen_target_at is not None
            and now - self._last_seen_target_at > self.config.target_lost_timeout
        ):
            self._begin_recovery(now, "target_lost")

        if self._severe_collision(observation) and self._stage not in {
            PolicyStage.WAITING,
            PolicyStage.RECOVERING,
        }:
            self._begin_recovery(now, "severe_collision")
        if self._motion_plan_failed and self._stage not in {
            PolicyStage.WAITING,
            PolicyStage.RECOVERING,
        }:
            if self._terminal_motion_plan_reason == "insufficient_intercept_time":
                # This is an admission failure before any path command was
                # emitted, not a failed workpiece.  Release the reservation
                # and return to staging.  Remember the measured rejection
                # state so EDF cannot reserve this same monotonically expiring
                # target at 50 Hz; it is eligible once arm motion can actually
                # change the route duration, and later candidates remain free.
                reason = self._terminal_motion_plan_reason
                self._remember_intercept_rejection(
                    self._target_id, observation
                )
                _hook(
                    self.env,
                    ("release_selected_target", "release_target_reservation"),
                    ((self._target_id,),),
                )
                self._terminal_motion_plan_reason = None
                self._motion_plan_failed = False
                self._discard_prefetch(cancel_worker=True)
                self._clear_target()
                self._transition(
                    PolicyStage.RECOVERING,
                    now,
                    reason=reason,
                )
            elif self._terminal_motion_plan_reason is not None:
                self._begin_recovery(
                    now, self._terminal_motion_plan_reason
                )
            elif not self._prepare_motion_plan_retry(now):
                if self._stage == PolicyStage.PREGRASP:
                    # Exhausting a route from the current redundancy branch is
                    # an admission failure, not a physical failure of an
                    # untouched workpiece.  Release it, move back toward the
                    # calibrated staging posture, and allow one later retry
                    # only after measured arm motion changes the route.  This
                    # avoids both falsely counting FAILED and a 50 Hz reselect
                    # storm from the identical q state.
                    rejected_id = self._target_id
                    self._remember_intercept_rejection(
                        rejected_id, observation
                    )
                    _hook(
                        self.env,
                        (
                            "release_selected_target",
                            "release_target_reservation",
                        ),
                        ((rejected_id,),),
                    )
                    self._motion_plan_failed = False
                    self._terminal_motion_plan_reason = None
                    self._discard_prefetch(cancel_worker=True)
                    self._clear_target()
                    self._transition(
                        PolicyStage.RECOVERING,
                        now,
                        reason="pregrasp_route_unavailable",
                    )
                else:
                    self._begin_recovery(now, "ik_unreachable")

        if (
            target is not None
            and self._stage
            in {
                PolicyStage.PREGRASP,
                PolicyStage.TRACKING_DESCENT,
                PolicyStage.CLOSING,
                PolicyStage.LIFTING,
            }
        ):
            # C is pure speculation over immutable task metadata and copied
            # robot state.  It never reserves, transitions or emits commands;
            # doing it throughout acquisition hides the fixed bin-route search
            # behind useful grasp work.
            self._update_carry_prefetch(target, observation)

        timeout = self._timeout_reason(now, observation)
        if timeout is not None:
            if self._stage == PolicyStage.RECOVERING:
                self._emit("recovery_timeout", now, timeout)
                self._clear_target()
                self._transition(PolicyStage.WAITING, now, reason=timeout)
            elif self._stage == PolicyStage.RETURNING:
                self._emit("return_timeout", now, timeout)
                self._discard_prefetch(cancel_worker=True)
                self._clear_target()
                self._transition(PolicyStage.WAITING, now, reason=timeout)
            elif self._stage != PolicyStage.WAITING:
                self._begin_recovery(now, timeout)

        # Transitions deliberately fall through where the new stage has an
        # immediately useful command; one action is still returned per tick.
        if self._stage == PolicyStage.WAITING:
            candidate = self._select_target(observation)
            if candidate is None:
                return self._action(
                    observation,
                    target_pose=self._wait_target(observation),
                    gripper=self.config.gripper_open_action,
                )
            self._target = candidate
            self._target_id = _object_id(candidate)
            self._grasp_gripper_action = self._grip_action(candidate)
            self._attempt_started_at = now
            self._last_seen_target_at = now
            self._transition(
                PolicyStage.PREGRASP,
                now,
                reason="target_selected",
                event="target_selected",
            )
            target = candidate

        full_action = self._full_pick_act(observation)
        if full_action is not None:
            return full_action
        tracking_reference_before_update = None
        if self._stage in {
            PolicyStage.PREGRASP,
            PolicyStage.TRACKING_DESCENT,
            PolicyStage.CLOSING,
        }:
            if target is None:
                return self._action(
                    observation,
                    target_pose=self._wait_target(observation),
                    gripper=self.config.gripper_open_action,
                )
            abort_reason = self._grasp_abort_reason(target, observation)
            if abort_reason is not None:
                self._begin_recovery(now, abort_reason)
            else:
                if (
                    self._stage == PolicyStage.TRACKING_DESCENT
                    and self._grasp_pose is not None
                ):
                    tracking_reference_before_update = self._grasp_pose.copy()
                updated_grasp_pose = (
                    self._grasp_pose
                    if (
                        self._stage == PolicyStage.PREGRASP
                        and self._grasp_pose is not None
                    )
                    else self._grasp_target_pose(target, observation)
                )
                if updated_grasp_pose is None:
                    self._begin_recovery(now, "prediction_unavailable")
                else:
                    if (
                        self._grasp_pose is not None
                        and self._stage
                        in {
                            PolicyStage.TRACKING_DESCENT,
                            PolicyStage.CLOSING,
                        }
                    ):
                        correction = (
                            updated_grasp_pose[:3] - self._grasp_pose[:3]
                        )
                        correction_norm = float(np.linalg.norm(correction))
                        if correction_norm > self.config.tracking_target_max_step:
                            updated_grasp_pose = updated_grasp_pose.copy()
                            updated_grasp_pose[:3] = (
                                self._grasp_pose[:3]
                                + correction
                                * (
                                    self.config.tracking_target_max_step
                                    / correction_norm
                                )
                            )
                    updated_grasp_pose = self._guard_final_tracking_xy(
                        updated_grasp_pose,
                        target,
                        observation,
                    )
                    self._grasp_pose = updated_grasp_pose.copy()

        if self._stage == PolicyStage.PREGRASP:
            assert self._grasp_pose is not None
            if self._continuous_intercept_enabled():
                # The complete approach owns the route from the present arm
                # state to the future contact. There is no high waypoint gate.
                self._transition(PolicyStage.TRACKING_DESCENT, now,
                    reason="continuous_intercept", event="continuous_intercept_started")
                self._continuous_hold_q = self._joint_positions(observation).copy()
                self._pregrasp_ready = True
                return self._action(observation, target_pose=self._grasp_pose,
                                    gripper=self.config.gripper_open_action)
            self._refresh_executing_pregrasp_intercept(observation)
            pregrasp = self._pregrasp_high_pose(self._grasp_pose, observation)
            if self._pregrasp_approach_pose is not None:
                pregrasp = self._pregrasp_approach_pose.copy()
            elif (self.config.dynamic_intercept_tracking
                    and self._safe_path_stage == PolicyStage.PREGRASP
                    and self._safe_path_target_pose is not None
                    and self._safe_path_generation == self._planning_generation):
                pregrasp = self._safe_path_target_pose.copy()
            measured_descent_ready = self._timed_descent_feedback_high_ready(observation)
            certified_handoff_ready = measured_descent_ready or bool(
                self._safe_joint_path
                and (not self._fixed_high_enabled(observation) or self._fixed_high_ready)
                and self._pregrasp_approach_pose is None
                and self._pregrasp_execution_timing.get("accepted", True)
                and self._safe_path_stage == PolicyStage.PREGRASP
                and (
                    self._safe_path_continuation_validated
                    or not self._planning_certificate_contract_required()
                )
                and self._safe_path_generation
                == self._planning_generation
                and self._same_target_id(
                    self._safe_path_target_id, self._target_id
                )
                and self._safe_path_scene_epoch
                == int(self._observed_scene_epoch or 0)
                and self._safe_path_target_pose is not None
                and np.linalg.norm(
                    pregrasp[:3] - self._safe_path_target_pose[:3]
                )
                <= self.config.trajectory_replan_position_tolerance
                and quat_distance(
                    pregrasp[3:], self._safe_path_target_pose[3:]
                )
                <= self.config.trajectory_replan_orientation_tolerance
                and self._safe_joint_path_index
                >= len(self._safe_joint_path) - 1
                and np.max(
                    np.abs(
                        self._joint_positions(observation)
                        - self._safe_joint_path[-1]
                    )
                )
                <= self._safe_path_settle_tolerance(len(self._safe_joint_path)-1)
                and self._joint_velocity_is_settled(
                    observation,
                    self.config.planner_waypoint_velocity_tolerance,
                )
            )
            handoff_pose_ready = measured_descent_ready or bool(
                self._pregrasp_pose_reached(pregrasp, observation)
                or self._pregrasp_blend_ready(
                    self._tracking_command_target(self._grasp_pose, observation,
                        lead_time=0.0, target=target, synchronize_to_intercept=True)
                    if self.config.dynamic_intercept_tracking else self._grasp_pose,
                    observation,
                )
            )
            if self.config.prepare_high_only and certified_handoff_ready and handoff_pose_ready:
                self.prepared_high_state = {
                    "timestamp": float(now),
                    "joint_positions_rad": self._joint_positions(observation).copy(),
                    "certified_high_joints_rad": self._safe_joint_path[-1].copy(),
                    "high_pose7": pregrasp.copy(),
                    "scene_epoch": int(self._observed_scene_epoch or 0),
                    "target_id": self._target_id,
                }
                # The runner consumes this completion before any arm request
                # is sent. Keep the policy in PREGRASP and never close the jaw.
                return self._action(observation, target_pose=pregrasp,
                                    gripper=self.config.gripper_open_action)
            if (
                certified_handoff_ready
                and handoff_pose_ready
                and self._descent_time_ready(target, observation, now)
            ):
                # Do not leave the worker-certified endpoint while waiting for
                # the intercept clock.  The continuation certificate starts at
                # ``path[-1]``; switching to the moving Cartesian pregrasp
                # servo early would invalidate that proof before descent.
                capture_high_handoff = bool(
                    self._pregrasp_capture_high_active
                    and self.config.pregrasp_capture_high_vertical_handoff_enabled
                )
                self._pregrasp_ready = True
                self._transition(
                    PolicyStage.TRACKING_DESCENT,
                    now,
                    reason="pregrasp_reached",
                    event="descent_started",
                )
                if capture_high_handoff:
                    self._start_capture_high_handoff(pregrasp, target, observation)
            return self._action(
                observation,
                target_pose=(
                    self._tracking_command_target(
                        self._grasp_pose,
                        observation,
                        target=target,
                        synchronize_to_intercept=True,
                    )
                    if self._stage == PolicyStage.TRACKING_DESCENT
                    else pregrasp
                ),
                gripper=self.config.gripper_open_action,
            )

        if self._stage == PolicyStage.TRACKING_DESCENT:
            assert self._grasp_pose is not None
            self._refresh_timed_descent(observation)
            tracking_pose = self._tracking_command_target(
                self._grasp_pose,
                observation,
                target=target,
                synchronize_to_intercept=True,
            )
            capture_pose = self._tracking_command_target(
                self._grasp_pose,
                observation,
                lead_time=0.0,
                target=target,
                synchronize_to_intercept=True,
            )
            if bool(_value(observation, "_defer_arm_motion", False)):
                # Final admission uses the same predicate, after receive waits
                # and at the feedback snapshot used for the actual command.
                self._deferred_closure_context = {
                    "capture_pose": capture_pose.copy(),
                    "tracking_reference_before_update": (
                        None if tracking_reference_before_update is None
                        else tracking_reference_before_update.copy()
                    ),
                }
            else:
                tracking_pose = self._admit_descent_closure(
                    target, observation, now, capture_pose, tracking_pose,
                    tracking_reference_before_update,
                )
            return self._action(
                observation,
                target_pose=tracking_pose,
                gripper=(
                    self._grasp_gripper_action
                    if self._stage == PolicyStage.CLOSING
                    else self.config.gripper_open_action
                ),
            )

        if self._stage == PolicyStage.CLOSING:
            assert self._grasp_pose is not None
            feedback = self._grasp_feedback(target, observation)
            elapsed = self._elapsed(now)
            close_settle_time = _value(
                target,
                "close_settle_time",
                self.config.close_settle_time,
            )
            max_close_time = _value(
                target,
                "max_close_time",
                self.config.max_close_time,
            )
            try:
                close_settle_time = max(0.0, float(close_settle_time))
            except (TypeError, ValueError):
                close_settle_time = self.config.close_settle_time
            try:
                max_close_time = max(
                    close_settle_time, float(max_close_time)
                )
            except (TypeError, ValueError):
                max_close_time = self.config.max_close_time
            closing_pose = self._grasp_pose.copy()
            if (
                self.config.track_belt_during_close
                or self.config.closing_tracking_lead_time > 0.0
            ):
                # `_grasp_pose` is refreshed from perception every tick, so
                # adding elapsed close time here would count belt travel twice
                # and drive the gripper ahead of the workpiece.
                closing_pose = self._tracking_command_target(
                    closing_pose,
                    observation,
                    lead_time=self.config.closing_tracking_lead_time,
                    target=target,
                    synchronize_to_intercept=True,
                )
            if (
                feedback is True
                and self._grasp_confirmed_at is None
                and self._intercept_time_reached(target, observation, now)
            ):
                # The environment reports GRASPED only after stable bilateral
                # force.  Record that event and apply the calibrated preload,
                # but keep following the belt during the short confirmation
                # dwell.  Freezing immediately at the first debounced sample
                # can reverse the arm while a wide part is only just between
                # the pads, throwing it out before lifting starts.
                self._grasp_confirmed_at = now
                previous_grasp_action = self._grasp_gripper_action
                self._capture_contact_hold_action(target)
                if (
                    self.config.minimum_grasp_normal_force > 0.0
                    and _value(_value(observation, "gripper", None), "minimum_normal_force", None) is not None
                ):
                    # Force-controlled capture must not loosen the adaptive
                    # command once contact has already begun loading the pads.
                    span = self.config.gripper_open_action - self.config.gripper_close_action
                    self._grasp_gripper_action = self.config.gripper_close_action + span * min(
                        (previous_grasp_action - self.config.gripper_close_action) / span,
                        (self._grasp_gripper_action - self.config.gripper_close_action) / span,
                    )
            gripper_observation = _value(observation, "gripper", None)
            contact_settle_time = _value(
                target,
                "confirmed_grasp_settle_time",
                self.config.confirmed_grasp_settle_time,
            )
            try:
                contact_settle_time = max(0.0, float(contact_settle_time))
            except (TypeError, ValueError):
                contact_settle_time = self.config.confirmed_grasp_settle_time
            if gripper_observation is None:
                bilateral_stable = True
            else:
                bilateral_now = (
                    feedback is True
                    if self.config.grasp_confirmation_source == "width_and_force"
                    else self._bilateral_finger_contact(observation)
                )
                if bilateral_now:
                    if self._bilateral_contact_stable_since is None:
                        self._bilateral_contact_stable_since = now
                else:
                    self._bilateral_contact_stable_since = None
                bilateral_stable = bool(
                    self._bilateral_contact_stable_since is not None
                    and now - self._bilateral_contact_stable_since
                    >= contact_settle_time
                )
            force_ready = self._update_grasp_force_preload(observation, now, feedback)
            confirmed = (
                feedback is True
                and self._grasp_confirmed_at is not None
                and now - self._grasp_confirmed_at
                >= contact_settle_time
                and bilateral_stable
                and force_ready
            )
            unobserved_but_settled = (
                feedback is None
                and not self.config.require_grasp_confirmation
                and elapsed >= close_settle_time
                and force_ready
            )
            if confirmed or unobserved_but_settled:
                if confirmed:
                    # The contact survived the dwell.  Only now freeze the
                    # measured TCP and cancel the closing servo before issuing
                    # the first lift command.
                    self._secured_hold_pose = self._ee_pose(observation)
                    self._invalidate_motion_plan()
                self._lift_pose = self._lift_corridor_target(observation)
                self._transition(
                    PolicyStage.LIFTING,
                    now,
                    reason=(
                        "grasp_confirmed" if confirmed else "close_settled"
                    ),
                    event="lift_started",
                )
            elif (
                elapsed >= close_settle_time
                and feedback is False
                and self._gripper_command_settled(observation)
                and not self._external_finger_contact(observation)
            ):
                # MuJoCo's force/debounce lifecycle becomes visible one
                # control tick after the first physical pad contact.  A jaw
                # that reaches its position target on that same tick is not an
                # empty grasp; keep closing until confirmation or the existing
                # bounded max-close timeout.
                self._begin_recovery(now, "empty_grasp")
            elif elapsed >= max_close_time:
                self._begin_recovery(now, "grasp_not_confirmed")
            return self._action(
                observation,
                target_pose=(
                    self._lift_pose
                    if self._stage == PolicyStage.LIFTING
                    else (
                        self._secured_hold_pose
                        if self._secured_hold_pose is not None
                        else closing_pose
                    )
                ),
                gripper=self._grasp_gripper_action,
            )

        if self._stage == PolicyStage.LIFTING:
            if self._lift_pose is None:
                self._lift_pose = self._lift_corridor_target(observation)
            feedback = self._grasp_feedback(target, observation)
            if self.config.maintain_grasp_preload_during_lift and feedback is not False:
                # The original closing preload reads this adapter's genuine
                # measured-load channel.  Its readiness result is diagnostic
                # here; it never adds a new lift admission gate or restores a
                # grasp whose feedback has already reported object loss.
                self._update_grasp_force_preload(observation, now, feedback)
            if self._clearing_payload() and feedback is not False:
                measured = self._ee_pose(observation)
                cleared = bool(
                    measured[2] >= self._lift_pose[2] - 0.005
                    and np.linalg.norm(measured[:2] - self._lift_pose[:2]) <= 0.010
                    and quat_distance(measured[3:], self._lift_pose[3:]) <= 0.10
                    and self._joint_velocity_is_settled(
                        observation, self.config.planner_waypoint_velocity_tolerance
                    )
                )
                if cleared:
                    self._lift_clearance_complete = True
                    if self._full_pick_contact_handed_off:
                        # The measured clearance has finished. The following
                        # move to the transfer corridor owns a separate phase
                        # budget; do not charge it the earlier clearance time.
                        self._stage_started_at=now
                        self._emit('lift_clearance_complete',now,'measured_clearance_reached_transfer_corridor_started')
                    self._invalidate_motion_plan()
                    self._lift_pose = self._lift_corridor_target(observation)
                return self._action(
                    observation, target_pose=self._lift_pose,
                    gripper=self._grasp_gripper_action,
                )
            if feedback is False and self.config.require_grasp_confirmation:
                self._begin_recovery(now, "object_dropped")
            elif self._pose_reached(self._lift_pose, observation) and (
                self._carry_prefetch_anchor_q is None
                or (
                    np.max(
                        np.abs(
                            self._joint_positions(observation)
                            - self._carry_prefetch_anchor_q
                        )
                    )
                    <= self.config.planner_corner_settle_tolerance
                    and self._joint_velocity_is_settled(
                        observation,
                        self.config.planner_waypoint_velocity_tolerance,
                    )
                )
            ):
                self._place_pose = self._bin_pose(
                    self._target if target is None else target, observation
                )
                if self._place_pose is None:
                    self._begin_recovery(now, "bin_pose_unavailable")
                else:
                    self._build_transfer_waypoints(observation)
                    self._transition(
                        PolicyStage.TRANSFERRING,
                        now,
                        reason="lift_reached",
                        event="transfer_started",
                    )
            return self._action(
                observation,
                target_pose=(
                    self._wait_target(observation)
                    if self._stage == PolicyStage.RECOVERING
                    else (
                        self._transfer_target()
                        if self._stage == PolicyStage.TRANSFERRING
                        else self._lift_pose
                    )
                ),
                gripper=self._grasp_gripper_action,
            )

        if self._stage == PolicyStage.TRANSFERRING:
            self._update_prefetch(
                observation, now, acquisition_ready=False
            )
            assert self._place_pose is not None
            transfer_target = self._transfer_target()
            feedback = self._grasp_feedback(target, observation)
            if feedback is False and self.config.require_grasp_confirmation:
                self._begin_recovery(now, "object_dropped")
            elif self._pose_reached(transfer_target, observation):
                if self._transfer_waypoint_index + 1 < len(
                    self._transfer_waypoints
                ):
                    self._transfer_waypoint_index += 1
                    self._invalidate_motion_plan()
                    transfer_target = self._transfer_target()
                else:
                    # Cartesian pose alone does not identify a 7-DoF route
                    # endpoint.  Releasing as soon as FK entered the loose
                    # pose tolerance allowed a redundant joint branch still
                    # 0.38 rad from C's certified final q to open the gripper;
                    # A and B were then planned from an anchor the plant never
                    # reached.  Finish and settle the installed joint route
                    # before the final release transition.
                    endpoint = (
                        self._prefetch_place_q
                        if self._safe_path_from_carry_prefetch
                        else None
                    )
                    if (
                        endpoint is None
                        and self._safe_joint_path
                        and self._safe_path_stage
                        == PolicyStage.TRANSFERRING
                    ):
                        endpoint = self._safe_joint_path[-1]
                    endpoint_settled = bool(
                        self._carry_path_replan_constraints is None
                        and (
                            endpoint is None
                            or (
                                np.max(
                                    np.abs(
                                        self._joint_positions(observation)
                                        - np.asarray(endpoint, dtype=float)
                                    )
                                )
                                <= self.config.planner_corner_settle_tolerance
                                and self._joint_velocity_is_settled(
                                    observation,
                                    self.config.planner_waypoint_velocity_tolerance,
                                )
                            )
                        )
                    )
                    if not endpoint_settled:
                        return self._action(
                            observation,
                            target_pose=transfer_target,
                            gripper=self._grasp_gripper_action,
                        )
                    self._transition(
                        PolicyStage.RELEASING,
                        now,
                        reason="bin_pose_reached",
                        event="release_started",
                    )
            return self._action(
                observation,
                target_pose=(
                    self._wait_target(observation)
                    if self._stage == PolicyStage.RECOVERING
                    else transfer_target
                ),
                gripper=self._grasp_gripper_action,
            )

        if self._stage == PolicyStage.RELEASING:
            self._latch_released_target_terminal(observation)
            self._update_prefetch(
                observation, now, acquisition_ready=False
            )
            release_elapsed = self._elapsed(now)
            release_pose = self._place_pose
            release_provider = getattr(self.env, 'get_release_pose', None)
            if callable(release_provider):
                supplied = _as_pose(release_provider(observation), self._place_quaternion)
                if supplied is not None:
                    release_pose = supplied
            release_feedback_ready = True
            if self.config.grasp_confirmation_source == "width_and_force":
                jaw = _value(observation, "gripper", None)
                fraction = _value(jaw, "open_fraction", None)
                try:
                    release_feedback_ready = bool(
                        np.isfinite(float(fraction)) and float(fraction) >= .96
                    )
                except (TypeError, ValueError):
                    release_feedback_ready = False
            # A moving-part adapter can complete the already requested upward
            # retreat while the tool still follows the belt.  Leaving release
            # at low height would abandon that motion and let the part run
            # into an open finger during the global return route.
            release_clear_provider = getattr(self.env, 'is_release_clear', None)
            if callable(release_clear_provider):
                release_feedback_ready = bool(
                    release_feedback_ready and release_clear_provider(observation)
                )
            if release_feedback_ready and release_elapsed >= (
                self.config.pre_release_settle_time
                + self.config.release_settle_time
            ):
                self._emit("release_complete", now, "release_settled")
                self._prefetch_release_complete = True
                self._build_return_waypoints(observation)
                self._transition(
                    PolicyStage.RETURNING,
                    now,
                    reason="release_settled",
                )
            return self._action(
                observation,
                target_pose=(
                    self._return_target(observation)
                    if self._stage == PolicyStage.RETURNING
                    else release_pose
                ),
                gripper=(
                    self._grasp_gripper_action
                    if release_elapsed < self.config.pre_release_settle_time
                    else self.config.gripper_open_action
                ),
            )

        if self._stage == PolicyStage.RETURNING:
            # Physics classifies a released part during env.step(), so SORTED
            # can become visible immediately after the policy transitioned out
            # of RELEASING.  Continue latching the immutable lifecycle fact in
            # RETURNING; otherwise the one safe short-A admission opportunity
            # is missed even though the object is already terminal in its bin.
            self._latch_released_target_terminal(observation)
            # A new RGB-D target may be visible before the previous return is
            # physically complete.  Once the *measured* tool has entered the
            # strict acquisition cone for two non-descending samples, search
            # a versioned PREGRASP route in the background.  Pending work does
            # not replace the installed return command and does not reserve
            # the part.  A matching result may pre-empt return only after its
            # latest measured first edge and visual deadline are revalidated.
            return_alignment = self._downward_alignment(
                self._ee_pose(observation)[3:]
            )
            acquisition_ready = self._acquisition_entry_ready(
                return_alignment
            )
            if self._update_prefetch(
                observation,
                now,
                acquisition_ready=acquisition_ready,
            ):
                return self.act(observation, self.env)
            # A target can become visible while the return trajectory is still
            # just outside the strict acquisition cone.  Waiting to finish the
            # old return edge before even asking the arm to stage costs roughly
            # one belt second and can close an otherwise feasible 10 s window.
            # Do not reserve or plan the object yet: head for the known wait
            # waypoint.  Never cancel an installed or pending return route to
            # do this; that cancellation used to throw away the result queued
            # on the release tick and submit the same route again.  Selection
            # and strict PREGRASP planning still require the two-sample gate.
            if self._peek_target(observation) is not None:
                wait = self._wait_target(observation)
                if (
                    self._planning_request is None
                    and not self._safe_joint_path
                    and (
                        not self._return_waypoints
                        or self._return_waypoint_index
                        != len(self._return_waypoints) - 1
                    )
                ):
                    self._return_waypoints = [wait.copy()]
                    self._return_waypoint_index = 0
                    self._invalidate_motion_plan()
            else:
                wait = self._wait_target(observation)
            independent_prefetch_active = bool(
                self._prefetch_anchor_q is not None
                or self._prefetch_retreat_request is not None
                or self._prefetch_request is not None
                or self._prefetch_result is not None
            )
            if not independent_prefetch_active:
                if self._update_return_lookahead(
                    observation,
                    acquisition_ready=acquisition_ready,
                    now=now,
                ):
                    return self.act(observation, self.env)
            prefetch_candidate = (
                self._peek_target(observation)
                if independent_prefetch_active
                else None
            )
            prefetch_result_deadline_invalid = bool(
                self._prefetch_result is not None
                and prefetch_candidate is not None
                and not self._prefetch_deadline_valid(
                    self._prefetch_result,
                    prefetch_candidate,
                    observation,
                    (
                        self._prefetch_request
                        if self._prefetch_request is not None
                        else self._prefetch_completed_request
                    ),
                )
            )
            prefetch_candidate_expired = bool(
                independent_prefetch_active
                and prefetch_candidate is not None
                and not self._prefetch_candidate_has_minimum_slack(
                    prefetch_candidate,
                    observation,
                )
            )
            if prefetch_candidate_expired:
                # Losing the candidate (or its remaining physical window) is
                # the only reason to abandon B while A is still travelling to
                # the certified retreat anchor.  Merely not being at that
                # anchor yet is expected and must not cancel the isolated
                # worker.
                self._discard_prefetch(cancel_worker=True)
                independent_prefetch_active = False
            elif prefetch_result_deadline_invalid:
                completed_request = (
                    self._prefetch_request
                    if self._prefetch_request is not None
                    else self._prefetch_completed_request
                )
                refresh_execution_reserve = (
                    self._prefetch_refresh_execution_reserve(
                        prefetch_candidate,
                        completed_request,
                        self._prefetch_result,
                    )
                )
                window = _value(prefetch_candidate, "grasp_window", {})
                raw_deadline = _value(
                    prefetch_candidate,
                    ("grasp_window_end", "window_end", "deadline"),
                    _value(window, ("end_time", "end", "deadline"), None),
                )
                try:
                    hard_intercept = (
                        float(raw_deadline)
                        - self.config.intercept_deadline_margin
                    )
                except (TypeError, ValueError):
                    hard_intercept = float("-inf")
                execution_with_latest_clock = bool(
                    completed_request is not None
                    and self._prefetch_result is not None
                    and self._prefetch_deadline_valid(
                        self._prefetch_result,
                        prefetch_candidate,
                        observation,
                        completed_request,
                        enforce_request_clock=False,
                    )
                )
                # Include A's frozen remaining ETA in the physical decision.
                # A B-only route can look admissible while the arm is still
                # 0.1--0.2 rad from its retreat anchor; retaining ownership in
                # that state starves ordinary RETURNING until the part expires.
                # A changed visual clock may keep A only if a *new* immutable
                # pose/clock request is submitted in this same tick.  The old
                # result itself is never retimed or executed.
                self._prefetch_failures += 1
                can_refresh = bool(
                    execution_with_latest_clock
                    and self._prefetch_failures
                    < self.config.planning_retry_attempts
                    and np.isfinite(hard_intercept)
                )
                if not can_refresh:
                    self._discard_prefetch(cancel_worker=True)
                    independent_prefetch_active = False
                else:
                    refresh_pose = self._return_pregrasp_pose(
                        prefetch_candidate, observation
                    )
                    self._discard_prefetch(
                        cancel_worker=True,
                        preserve_retreat=True,
                    )
                    if refresh_pose is not None:
                        self._prefetch_candidate_id = _object_id(
                            prefetch_candidate
                        )
                        self._prefetch_bound_candidate_id = _object_id(
                            prefetch_candidate
                        )
                        self._prefetch_candidate_stable_steps = max(
                            2,
                            self.config.acquisition_entry_stable_steps,
                        )
                        self._prefetch_category = self._lookahead_category(
                            prefetch_candidate
                        )
                        self._submit_prefetch_pregrasp(
                            prefetch_candidate,
                            refresh_pose,
                            observation,
                            execution_reserve_s=(
                                refresh_execution_reserve
                            ),
                        )
                    if self._prefetch_request is None:
                        self._discard_prefetch(cancel_worker=False)
                        independent_prefetch_active = False
                    else:
                        independent_prefetch_active = True
            elif (
                self._prefetch_retreat_installed
                and self._prefetch_anchor_q is not None
                and self._prefetch_retreat_pose is not None
            ):
                # The certified retreat leg owns RETURNING once installed.
                # A commonly finishes before the next 10 s-spaced object is
                # born; absence of a candidate at that moment is expected,
                # not target loss.  Continue A to its measured endpoint, then
                # hold that strict-cone/open-jaw anchor while the independent
                # worker awaits or computes B.  Falling through to the normal
                # return waypoints would invalidate A and make B's immutable
                # start stale before it could ever be committed.
                anchor_reached = bool(
                    np.max(
                        np.abs(
                            self._joint_positions(observation)
                            - self._prefetch_anchor_q
                        )
                    )
                    <= self.config.planner_corner_settle_tolerance
                    and self._joint_velocity_is_settled(
                        observation,
                        self.config.planner_waypoint_velocity_tolerance,
                    )
                )
                if not self._release_is_physically_clear(observation):
                    # A short handoff is admitted only after SORTED plus a
                    # clear open jaw.  Reappearing external contact while it
                    # executes invalidates that dynamic-clearance assumption;
                    # discard the speculative lineage and let the ordinary
                    # 55 mm RETURNING route withdraw from latest measured q.
                    self._discard_prefetch(cancel_worker=True)
                    independent_prefetch_active = False
                else:
                    return self._action(
                        observation,
                        target_pose=(
                            None
                            if anchor_reached
                            else self._prefetch_retreat_pose
                        ),
                        gripper=self.config.gripper_open_action,
                    )
            staging_ready = self._pose_reached(wait, observation)
            next_target = (
                self._select_target(observation)
                if (
                    not independent_prefetch_active
                    and
                    staging_ready
                    and acquisition_ready
                )
                else None
            )
            if next_target is not None:
                self._clear_target()
                self._target = next_target
                self._target_id = _object_id(next_target)
                self._grasp_gripper_action = self._grip_action(next_target)
                self._attempt_started_at = now
                self._last_seen_target_at = now
                self._transition(
                    PolicyStage.PREGRASP,
                    now,
                    reason="next_target_available",
                    event="target_selected",
                )
                return self.act(observation, self.env)
            return_target = self._return_target(observation)
            if self._pose_reached(return_target, observation):
                if self._return_waypoint_index + 1 < len(
                    self._return_waypoints
                ):
                    self._return_waypoint_index += 1
                    self._invalidate_motion_plan()
                    return_target = self._return_target(observation)
                else:
                    self._emit("attempt_complete", now, "wait_pose_reached")
                    if independent_prefetch_active:
                        self._discard_prefetch(cancel_worker=True)
                    self._clear_target()
                    self._transition(
                        PolicyStage.WAITING, now, reason="wait_pose_reached"
                    )
            return self._action(
                observation,
                target_pose=(
                    self._wait_target(observation)
                    if self._stage == PolicyStage.WAITING
                    else return_target
                ),
                gripper=self.config.gripper_open_action,
            )

        # RECOVERING: open the jaws and take the same acceleration-limited path
        # back to the configured wait pose.  No workpiece state is changed.
        wait_pose = self._wait_target(observation)
        if self._pose_reached(wait_pose, observation):
            self._emit("recovery_complete", now, self._last_reason)
            self._clear_target()
            self._transition(PolicyStage.WAITING, now, reason="recovered")
        return self._action(
            observation,
            target_pose=wait_pose,
            gripper=self.config.gripper_open_action,
        )

    # Common policy API aliases.
    compute_action = act
    __call__ = act

    def drain_events(self) -> list[PolicyEvent]:
        events = self._events
        self._events = []
        return events

    def get_info(self) -> dict[str, Any]:
        return {
            "stage": self._stage.value,
            "target_id": self._target_id,
            "motion_compensation": self._motion_compensation(),
            "moving_approach_mode": self.config.moving_approach_mode,
            "full_preplanned": self._full_pick_info(),
            "continuous_intercept": self._continuous_intercept_info(),
            "tcp_speed_limits_enabled": self.config.tcp_speed_limits_enabled,
            "local_tcp_linear_speed_limit_enabled": self.config.local_tcp_linear_speed_limit_enabled,
            "timed_descent_tcp_speed_limits_enabled": (
                self.config.timed_descent_tcp_speed_limits_enabled),
            "dynamic_pregrasp_global_tcp_speed_limits_enabled": (
                self.config.dynamic_pregrasp_global_tcp_speed_limits_enabled),
            "current_global_path_tcp_speed_limits_enabled": (
                self._global_path_tcp_speed_limits_enabled()),
            "closure_intercept_time": self._closure_intercept_at,
            "safe_path_eta_s": self._safe_path_eta_s,
            "pregrasp_execution_timing": dict(self._pregrasp_execution_timing),
            "planning_pending_goal_update": dict(self._planning_pending_goal_update),
            "trajectory_retiming": dict(self._safe_edge_retime_diagnostics),
            "joint_response": dict(getattr(self, '_safe_edge_reference_diagnostics', {})),
            "trajectory_reuse": dict(self._trajectory_reuse_diagnostics),
            "pregrasp_timing_refinements": (
                self._pregrasp_timing_refinements
            ),
            "pregrasp_capture_high_fallback": dict(
                self._pregrasp_capture_high_diagnostics
            ),
            "stage_started_at": self._stage_started_at,
            "attempt_started_at": self._attempt_started_at,
            "last_reason": self._last_reason,
            "last_error": self._last_error,
            "grasp_force_status": self._grasp_force_status,
            "grasp_force_stable_since": self._grasp_force_stable_since,
            "jacobian_condition": self._last_jacobian_condition,
            "minimum_joint_limit_margin": self._minimum_observed_joint_margin,
            "minimum_downward_alignment": (
                self._minimum_observed_downward_alignment
            ),
            "grasp_pose": (
                None if self._grasp_pose is None else self._grasp_pose.copy()
            ),
            "contact_tracking_xy": (
                None
                if self._contact_tracking_xy is None
                else self._contact_tracking_xy.copy()
            ),
            "contact_tracking_reference_xy": (
                None
                if self._contact_tracking_reference_xy is None
                else self._contact_tracking_reference_xy.copy()
            ),
            "lift_pose": (
                None if self._lift_pose is None else self._lift_pose.copy()
            ),
            "lift_feedback_correction": dict(self._lift_feedback_correction_diagnostics),
            "lift_clearance_global_replan": dict(
                self._lift_clearance_global_replan_diagnostics
            ),
            "lift_stopping_reference": dict(self._lift_stopping_reference_diagnostics),
            "place_pose": (
                None if self._place_pose is None else self._place_pose.copy()
            ),
            "pending_events": len(self._events),
        }


# Compatibility aliases with explicit names in debug/config code.
BaselinePolicy = BaselineSortingPolicy
SortingPolicyState = PolicyStage

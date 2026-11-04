"""Non-blocking deterministic global planning on an isolated MuJoCo copy.

The viewer/control thread must never run multi-seed IK or constrained RRT.
This module owns one daemon worker and, for the real MuJoCo environment, a
private model/data/IK stack.  Requests are immutable snapshots.  Results are
versioned and must still be revalidated against live measured state by the
policy before their first command is used.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import mujoco
import numpy as np

from .ik_solver import DampedLeastSquaresIK, IKResult
from .joint_trajectory import (
    QuinticJointTrajectory,
    sampled_tcp_angular_minimum_duration,
)
from .continuous_joint_trajectory import build_certified_edge_trajectory, BoundaryQuinticJointTrajectory
from .full_pick_planning import FullPickPlanningMixin
from .safe_path_planner import ConstrainedRRTConnect, SafePathConfig
from .planning_progress import PlanningProgress, planning_stage


class _PlanningCancelled(RuntimeError):
    """Internal cooperative unwind from post-search validation/timing."""


CancelCheck = Callable[[], bool]


def _array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(shape).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _arrays(values: Sequence[Any], shape: tuple[int, ...], *, name: str) -> tuple[np.ndarray, ...]:
    return tuple(_array(value, shape, name=name) for value in values)


@dataclass(frozen=True)
class PlanningRequest:
    """Complete, immutable input to one background global-plan search."""

    request_id: int
    generation: int
    stage: str
    target_id: Any
    scene_epoch: int
    target_poses: tuple[np.ndarray, ...]
    start_q: np.ndarray
    full_qpos: np.ndarray
    goal_seeds: tuple[np.ndarray, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    minimum_tcp_height: float
    minimum_joint_margin: float
    minimum_downward_alignment: float
    approach_axis_local: np.ndarray
    singularity_hard_stop_condition: float
    planner_config: SafePathConfig
    planner_seed: int
    trajectory_velocity_limits: np.ndarray
    trajectory_acceleration_limits: np.ndarray
    trajectory_minimum_duration: float
    target_identity: Any = None
    selection_policy: str = 'default'
    planning_mode: str = 'legacy'
    replan_policy: str = 'legacy'
    full_high_clearance_m: float = .10
    full_approach_mode: str = 'high'
    full_close_window_s: float = .60
    full_lift_m: float = .03
    full_grasp_width_m: float = .04
    full_plan_compute_budget_s: float = 2.0
    # Policy-clock bounds for the visual target's certified contact time.
    full_contact_earliest_s: float | None = None
    full_contact_latest_s: float | None = None
    # Immutable calibrated-model snapshot. Absent/ineligible is ordinary motion.
    full_parent_version: tuple[int, int] | None = None
    full_observed_timestamp_s: float | None = None
    full_observation_clock_epoch: int | None = None
    full_observation_track_generation: int | None = None
    full_response_model_json: str | None = None
    full_contact_handoff_enabled: bool = False
    full_orientation_preparation_s: float = 0.
    full_orientation_preparation_tracking_entry: bool = False
    full_contact_acceleration_m_s2: float = 0.
    full_finger_entry_height_m: float = 0.
    full_finger_entry_lead_s: float = .8
    full_response_projection_fraction: float = .25
    full_approach_response_enabled: bool = True
    full_close_lead_s: float = .18
    full_command_lower_limits: tuple[float, ...] | None = None
    full_command_upper_limits: tuple[float, ...] | None = None
    full_control_period_s: float = .01
    full_dispatch_lead_s: float = 0.
    full_feedback_translation_limit_m: float = .022
    full_feedback_rotation_limit_rad: float = .1
    acquisition_joint_limit_rad: float = .20
    full_closure_horizontal_tolerance_m: float = .022
    full_closure_vertical_tolerance_m: float = .012
    full_closure_orientation_tolerance_rad: float = .25
    full_sample_time_s: float | None = None
    full_sample_start_time_s: float | None = None
    full_phase: str = 'approach'
    joint_trajectory_backend: str = 'quintic'
    trajectory_jerk_limits: tuple[float, ...] | None = None
    max_goal_candidates: int = 5
    rrt_iteration_slice: int = 100
    rrt_restart_count: int = 24
    maximum_tcp_angular_speed: float = 1.85
    maximum_tcp_linear_speed: float | None = None
    tcp_speed_limits_enabled: bool = True
    wall_time_budget_s: float | None = None
    # Absolute visual intercept timestamp associated with ``target_poses``.
    # Independent moving-object prefetch uses this to certify a route to the
    # same late-but-deadline-safe pose that will be admitted atomically on the
    # live environment.  Ordinary/static requests leave it unset.
    planned_target_time_s: float | None = None
    # Only wall-clock execution pays for asynchronous planning while the belt
    # keeps moving. Fixed-step simulation keeps its original clock behavior.
    intercept_account_for_planning_time: bool = False
    intercept_motion_time_scale: float = 1.0
    # Speculative next-object work starts several seconds before admission and
    # may compare several fully certified direct IK branches by execution ETA.
    # Active lift/transfer requests keep the latency-minimizing first-safe
    # behavior so candidate comparison cannot consume their stage deadline.
    optimize_direct_path_eta: bool = False
    # Speculative requests may spend a fixed number of round-robin RRT restart
    # rounds trying to beat an already certified direct route.  Zero preserves
    # the latency-first behavior for every ordinary active-stage request.
    rrt_restarts_after_direct: int = 0
    # A speculative request may instead give the first few prioritized IK
    # branches one persistent, full-iteration RRT search apiece.  This avoids
    # repeatedly rebuilding a 100-iteration tree that can never reach a route
    # found by one uninterrupted search.  Zero preserves the ordinary sliced
    # RRT behavior, including for active LIFT/TRANSFER requests.
    full_rrt_candidates_after_direct: int = 0
    # Do not spend a full persistent-tree budget trying to improve an already
    # short speculative route.  ``None`` preserves the unconditional behavior
    # for explicit diagnostic/test requests; production B sets a finite ETA
    # threshold so only a genuinely slow direct redundancy branch triggers
    # the extra background search.
    full_rrt_minimum_direct_eta_s: float | None = None
    # Optional caller-provided joint-space waypoints for a known safe route
    # topology.  They are suggestions, never certificates: the isolated
    # worker must validate every state and edge under this exact request before
    # it may return a hinted route.  Empty preserves the normal direct/RRT
    # search with no additional planner work.
    route_waypoint_hints: tuple[np.ndarray, ...] = ()
    # Active moving-target replans first test the cheap current->new-goal
    # chord, then reuse the old interior corners before starting RRT. Existing
    # configured/precomputed routes keep their original hint-first ordering.
    route_hints_after_direct: bool = False
    # PREGRASP requests may carry one corresponding low capture pose per
    # target pose.  A high waypoint is useful only if the same redundancy
    # branch admits a continuous, collision-checked Cartesian descent.  The
    # worker proves that continuation off the control thread before returning
    # the global route.  Other stages leave this empty.
    continuation_target_poses: tuple[np.ndarray, ...] | None = None
    continuation_minimum_tcp_height: float | None = None
    continuation_cartesian_step: float = 0.020
    continuation_max_joint_step: float = 0.25
    # Short post-release motions may require a topology-preserving Cartesian
    # corridor instead of a free 7-DOF RRT.  The worker derives the source TCP
    # pose from ``full_qpos`` + ``start_q``, warm-starts every Cartesian sample,
    # and returns failure rather than falling back to another IK branch.
    cartesian_local_only: bool = False
    # Recheck descent from the exact certified high joint branch, without
    # searching another high IK solution or moving that spatial anchor.
    fixed_high_continuation: bool = False
    # Opt-in direct interception: target_poses are contact poses at the fixed
    # planned_target_time_s; no pregrasp/high endpoint is inserted.
    continuous_intercept: bool = False
    # Preserve contact-vs-high target semantics when the direct-interception
    # branch delegates geometry to the existing ordinary route planner.
    target_is_contact: bool = False
    # Optional executable continuation. Absolute clock is supplied for an
    # in-flight splice; initial high-to-low certificates use the returned
    # duration counted backwards from planned_target_time_s.
    timed_continuation_duration_s: float | None = None
    continuation_start_time_s: float | None = None
    continuation_start_velocity: tuple[float, ...] = (0.,) * 7
    continuation_start_acceleration: tuple[float, ...] = (0.,) * 7
    continuation_velocity_m_s: tuple[float, ...] = (0.,) * 3
    continuation_tcp_speed_limits: tuple[float, float] | None = None
    # Measured previous descent job, for forecasting only (never a timeout).
    continuation_compute_forecast_s: float = 0.0
    maximum_goal_joint_delta: float | None = None
    # Moving PREGRASP forecasts are solved with the shared IK and trajectory
    # duration calculation before one complete geometric route is certified.
    intercept_velocity_m_s: np.ndarray | None = None
    intercept_motion_start_time_s: float | None = None
    intercept_request_time_s: float | None = None
    intercept_descent_duration_s: float = 0.0
    intercept_position_tolerance_m: float = 0.004

    def __post_init__(self) -> None:
        if self.full_parent_version is not None:
            version=tuple(int(v) for v in self.full_parent_version)
            if len(version)!=2 or min(version)<0:
                raise ValueError('invalid full parent trajectory version')
            object.__setattr__(self,'full_parent_version',version)
        if self.full_observed_timestamp_s is not None and not np.isfinite(self.full_observed_timestamp_s):
            raise ValueError('invalid full observation timestamp')
        if self.full_command_lower_limits is None or self.full_command_upper_limits is None:
            object.__setattr__(self,'full_command_lower_limits',None)
            object.__setattr__(self,'full_command_upper_limits',None)
        else:
            lower=np.asarray(self.full_command_lower_limits,dtype=float)
            upper=np.asarray(self.full_command_upper_limits,dtype=float)
            if lower.shape!=(7,) or upper.shape!=(7,) or not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>=upper):
                raise ValueError('invalid full command position limit snapshot')
            object.__setattr__(self,'full_command_lower_limits',tuple(float(v) for v in lower))
            object.__setattr__(self,'full_command_upper_limits',tuple(float(v) for v in upper))
        if self.selection_policy not in {'default', 'first_safe'}:
            raise ValueError('unknown selection_policy')
        if self.planning_mode not in {'legacy', 'full_preplanned'}:
            raise ValueError('unknown planning_mode')
        if self.replan_policy not in {'legacy', 'event_triggered'}:
            raise ValueError('unknown replan_policy')
        if not np.isfinite(self.full_orientation_preparation_s) or self.full_orientation_preparation_s<0.:
            raise ValueError('orientation preparation must be finite and nonnegative')
        if self.full_orientation_preparation_s and not (self.full_contact_handoff_enabled and self.full_approach_mode=='continuous'):
            raise ValueError('orientation preparation requires continuous contact prefix')
        if self.full_orientation_preparation_tracking_entry and not self.full_orientation_preparation_s:
            raise ValueError('tracking entry requires explicit orientation preparation')
        if not np.isfinite(self.full_contact_acceleration_m_s2) or self.full_contact_acceleration_m_s2<0.:
            raise ValueError('contact acceleration must be finite and nonnegative')
        if (not np.isfinite(self.full_finger_entry_height_m) or self.full_finger_entry_height_m<0.
                or not np.isfinite(self.full_finger_entry_lead_s) or self.full_finger_entry_lead_s<=0.):
            raise ValueError('finger entry shape requires finite nonnegative height and positive lead')
        if self.full_finger_entry_height_m and not (self.full_contact_handoff_enabled and self.full_approach_mode=='continuous'):
            raise ValueError('finger entry shape requires continuous contact prefix')
        if not np.isfinite(self.full_response_projection_fraction) or not 0.<=self.full_response_projection_fraction<=1.:
            raise ValueError('response projection fraction must be in [0,1]')
        if self.full_contact_acceleration_m_s2 and not (self.full_contact_handoff_enabled and self.full_approach_mode=='continuous'):
            raise ValueError('contact acceleration requires continuous contact prefix')
        if self.full_approach_mode not in {'high', 'continuous'}:
            raise ValueError('unknown full_approach_mode')
        if self.planning_mode == 'full_preplanned':
            for name in ('full_contact_earliest_s', 'full_contact_latest_s'):
                value=getattr(self,name)
                if value is not None and not np.isfinite(value):
                    raise ValueError('full contact window bounds must be finite')
            if (self.full_contact_earliest_s is not None and self.full_contact_latest_s is not None
                    and self.full_contact_earliest_s > self.full_contact_latest_s):
                raise ValueError('full contact window is inverted')
            if self.selection_policy != 'first_safe' or self.replan_policy != 'event_triggered':
                raise ValueError('full_preplanned needs first_safe and event_triggered')
            prefix_only=self.full_contact_handoff_enabled and self.full_approach_mode=='continuous'
            values = np.array([self.full_close_window_s,
                               self.full_lift_m,
                               *([] if prefix_only else [self.full_grasp_width_m]),
                               self.full_plan_compute_budget_s,
                               *([self.full_high_clearance_m] if self.full_approach_mode == 'high' else [])])
            if not np.isfinite(values).all() or np.any(values <= 0.):
                raise ValueError('full pick dimensions and closure window must be positive')
            if prefix_only and (not np.isfinite(self.full_grasp_width_m) or self.full_grasp_width_m<0.):
                raise ValueError('contact-only closure width must be finite and nonnegative')
            if not self.continuous_intercept:
                raise ValueError('full_preplanned needs absolute continuous intercept clocks')
            response_values = (self.full_control_period_s, self.full_feedback_translation_limit_m,
                self.full_feedback_rotation_limit_rad, self.acquisition_joint_limit_rad)
            closure_bounds = (self.full_closure_horizontal_tolerance_m,
                self.full_closure_vertical_tolerance_m,self.full_closure_orientation_tolerance_rad)
            if (not np.isfinite(response_values).all() or min(response_values) <= 0.
                    or not np.isfinite(closure_bounds).all() or min(closure_bounds) < 0.
                    or not np.isfinite(self.full_dispatch_lead_s) or self.full_dispatch_lead_s < 0.):
                raise ValueError('invalid full pick response forecast periods or inherited bounds')
        if self.selection_policy == 'first_safe':
            object.__setattr__(self, 'optimize_direct_path_eta', False)
            object.__setattr__(self, 'rrt_restarts_after_direct', 0)
            object.__setattr__(self, 'full_rrt_candidates_after_direct', 0)
        if not np.isfinite(self.continuation_compute_forecast_s) or self.continuation_compute_forecast_s < 0.:
            raise ValueError('continuation compute forecast must be finite and nonnegative')
        if self.timed_continuation_duration_s is not None:
            duration = float(self.timed_continuation_duration_s)
            if not np.isfinite(duration) or duration <= 0 or (
                    self.continuation_target_poses is None and not self.continuous_intercept):
                raise ValueError("timed continuation needs positive duration and a low target")
            object.__setattr__(self, 'timed_continuation_duration_s', duration)
        if self.continuation_start_time_s is not None and (
                not np.isfinite(self.continuation_start_time_s)
                or (self.timed_continuation_duration_s is None and not self.continuous_intercept)):
            raise ValueError("continuation start clock needs a timed continuation")
        if self.continuous_intercept:
            if (self.continuation_start_time_s is None
                    or self.planned_target_time_s is None
                    or not np.isfinite(self.planned_target_time_s)
                    or self.planned_target_time_s <= self.continuation_start_time_s):
                raise ValueError('continuous intercept needs ordered absolute start and capture clocks')
            if self.fixed_high_continuation or self.continuation_target_poses is not None:
                raise ValueError('continuous intercept uses contact target_poses without a fixed high')
            if self.timed_continuation_duration_s is not None and not np.isclose(
                    self.timed_continuation_duration_s,
                    self.planned_target_time_s-self.continuation_start_time_s,
                    rtol=0., atol=1e-9):
                raise ValueError('continuous intercept duration must match its absolute clocks')
        for name in ('continuation_start_velocity', 'continuation_start_acceleration'):
            object.__setattr__(self, name, tuple(_array(getattr(self, name), (7,), name=name)))
        object.__setattr__(self, 'continuation_velocity_m_s',
            tuple(_array(self.continuation_velocity_m_s, (3,), name='continuation velocity')))
        if self.continuation_tcp_speed_limits is not None:
            limits = _array(self.continuation_tcp_speed_limits, (2,), name='continuation TCP limits')
            if np.any(limits <= 0):
                raise ValueError('continuation TCP limits must be positive')
            object.__setattr__(self, 'continuation_tcp_speed_limits', tuple(limits))
        if self.joint_trajectory_backend not in ('quintic', 'ruckig'):
            raise ValueError('unknown joint trajectory backend')
        if self.trajectory_jerk_limits is not None:
            jerks = tuple(float(value) for value in self.trajectory_jerk_limits)
            if len(jerks) != 7 or not np.isfinite(jerks).all() or min(jerks) <= 0:
                raise ValueError('jerk limits must contain seven positive finite values')
            object.__setattr__(self, 'trajectory_jerk_limits', jerks)
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "scene_epoch", int(self.scene_epoch))
        if self.fixed_high_continuation and (
                self.stage not in {"pregrasp", "tracking_descent"} or self.continuation_target_poses is None
                or len(self.target_poses) != 1 or self.intercept_velocity_m_s is not None):
            raise ValueError("fixed high validation needs one pregrasp continuation and no moving-high forecast")
        scale = float(self.intercept_motion_time_scale)
        if not np.isfinite(scale) or scale < 1.0:
            raise ValueError("intercept motion time scale must be finite and at least one")
        object.__setattr__(self, "intercept_motion_time_scale", scale)
        if self.intercept_velocity_m_s is not None:
            object.__setattr__(self, "intercept_velocity_m_s", _array(
                self.intercept_velocity_m_s, (3,), name="intercept_velocity_m_s"))
            if self.intercept_motion_start_time_s is None or self.planned_target_time_s is None:
                raise ValueError("moving intercept needs motion start and target clocks")
        object.__setattr__(
            self,
            "target_poses",
            _arrays(self.target_poses, (7,), name="target pose"),
        )
        if not self.target_poses:
            raise ValueError("target_poses cannot be empty")
        object.__setattr__(self, "start_q", _array(self.start_q, (7,), name="start_q"))
        full = np.asarray(self.full_qpos, dtype=float).reshape(-1).copy()
        if full.size < 7 or not np.all(np.isfinite(full)):
            raise ValueError("full_qpos must be a finite full-model vector")
        full.setflags(write=False)
        object.__setattr__(self, "full_qpos", full)
        object.__setattr__(
            self,
            "goal_seeds",
            _arrays(self.goal_seeds, (7,), name="goal seed"),
        )
        object.__setattr__(
            self, "lower_limits", _array(self.lower_limits, (7,), name="lower_limits")
        )
        object.__setattr__(
            self, "upper_limits", _array(self.upper_limits, (7,), name="upper_limits")
        )
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("planning joint limits are invalid")
        axis = _array(self.approach_axis_local, (3,), name="approach_axis_local")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("approach_axis_local must be non-zero")
        axis = np.asarray(axis / norm, dtype=float)
        axis.setflags(write=False)
        object.__setattr__(self, "approach_axis_local", axis)
        object.__setattr__(
            self,
            "trajectory_velocity_limits",
            _array(self.trajectory_velocity_limits, (7,), name="velocity limits"),
        )
        object.__setattr__(
            self,
            "trajectory_acceleration_limits",
            _array(
                self.trajectory_acceleration_limits,
                (7,),
                name="acceleration limits",
            ),
        )
        if np.any(self.trajectory_velocity_limits <= 0.0) or np.any(
            self.trajectory_acceleration_limits <= 0.0
        ):
            raise ValueError("trajectory limits must be positive")
        if self.max_goal_candidates < 1:
            raise ValueError("max_goal_candidates must be positive")
        if self.rrt_iteration_slice < 1 or self.rrt_restart_count < 1:
            raise ValueError("RRT slice/restart counts must be positive")
        if (
            not np.isfinite(self.maximum_tcp_angular_speed)
            or self.maximum_tcp_angular_speed <= 0.0
        ):
            raise ValueError("maximum_tcp_angular_speed must be positive")
        if self.maximum_tcp_linear_speed is not None and (
                not np.isfinite(self.maximum_tcp_linear_speed) or self.maximum_tcp_linear_speed<=0.):
            raise ValueError('maximum_tcp_linear_speed must be positive when supplied')
        if self.wall_time_budget_s is not None:
            budget = float(self.wall_time_budget_s)
            if not np.isfinite(budget) or budget <= 0.0:
                raise ValueError("wall_time_budget_s must be positive")
            object.__setattr__(self, "wall_time_budget_s", budget)
        if self.planned_target_time_s is not None:
            planned_time = float(self.planned_target_time_s)
            if not np.isfinite(planned_time):
                raise ValueError("planned_target_time_s must be finite")
            object.__setattr__(
                self, "planned_target_time_s", planned_time
            )
        if self.continuation_target_poses is not None:
            continuation = _arrays(
                self.continuation_target_poses,
                (7,),
                name="continuation target pose",
            )
            if len(continuation) != len(self.target_poses):
                raise ValueError(
                    "continuation_target_poses must match target_poses"
                )
            object.__setattr__(
                self, "continuation_target_poses", continuation
            )
            height = float(self.continuation_minimum_tcp_height)
            if not np.isfinite(height):
                raise ValueError(
                    "continuation_minimum_tcp_height must be finite"
                )
            object.__setattr__(
                self, "continuation_minimum_tcp_height", height
            )
        elif self.continuation_minimum_tcp_height is not None:
            raise ValueError(
                "continuation_minimum_tcp_height needs continuation poses"
            )
        for name in (
            "continuation_cartesian_step",
            "continuation_max_joint_step",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "cartesian_local_only", bool(self.cartesian_local_only)
        )
        object.__setattr__(
            self,
            "optimize_direct_path_eta",
            bool(self.optimize_direct_path_eta),
        )
        rrt_restarts = float(self.rrt_restarts_after_direct)
        if (
            not np.isfinite(rrt_restarts)
            or rrt_restarts < 0.0
            or not rrt_restarts.is_integer()
        ):
            raise ValueError("rrt_restarts_after_direct must be non-negative")
        object.__setattr__(
            self, "rrt_restarts_after_direct", int(rrt_restarts)
        )
        full_rrt_candidates = float(
            self.full_rrt_candidates_after_direct
        )
        if (
            not np.isfinite(full_rrt_candidates)
            or full_rrt_candidates < 0.0
            or not full_rrt_candidates.is_integer()
        ):
            raise ValueError(
                "full_rrt_candidates_after_direct must be non-negative"
            )
        object.__setattr__(
            self,
            "full_rrt_candidates_after_direct",
            int(full_rrt_candidates),
        )
        if self.full_rrt_minimum_direct_eta_s is not None:
            threshold = float(self.full_rrt_minimum_direct_eta_s)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    "full_rrt_minimum_direct_eta_s must be non-negative"
                )
            object.__setattr__(
                self, "full_rrt_minimum_direct_eta_s", threshold
            )
        object.__setattr__(
            self,
            "route_waypoint_hints",
            _arrays(
                self.route_waypoint_hints,
                (7,),
                name="route waypoint hint",
            ),
        )
        if self.maximum_goal_joint_delta is not None:
            delta = float(self.maximum_goal_joint_delta)
            if not np.isfinite(delta) or delta <= 0.0:
                raise ValueError("maximum_goal_joint_delta must be positive")
            object.__setattr__(self, "maximum_goal_joint_delta", delta)
        if self.cartesian_local_only and self.maximum_goal_joint_delta is None:
            raise ValueError(
                "cartesian_local_only needs maximum_goal_joint_delta"
            )

    @property
    def target_pose(self) -> np.ndarray:
        """Primary requested pose, retained for simple workers/tests."""

        return self.target_poses[0]


@dataclass(frozen=True)
class PlanningResult:
    """Versioned output from a worker; paths are not live-safe until committed."""

    request_id: int
    generation: int
    stage: str
    target_id: Any
    scene_epoch: int
    target_pose: np.ndarray
    start_q: np.ndarray
    path: tuple[np.ndarray, ...] | None = None
    cost: tuple[float, float, int] | None = None
    edge_durations: tuple[float, ...] | None = None
    # True only for a route whose every immutable corner edge was densely
    # checked by MuJoCoGlobalPlanningEngine under this request's constraints.
    # The live policy may then avoid repeating a whole far edge in one 50 Hz
    # tick; delivery's rebased first edge and every emitted short command are
    # still revalidated against latest measured state.
    path_fully_validated: bool = False
    continuation_fully_validated: bool = False
    continuation_path: tuple[np.ndarray, ...] | None = None
    full_pick_plan: Any | None = None
    continuation_trajectory: Any | None = None
    continuation_target_pose: np.ndarray | None = None
    target_identity: Any = None
    error: str | None = None
    cancelled: bool = False
    planning_time_s: float = 0.0
    planner: Any | None = None
    effective_request: PlanningRequest | None = None
    # Request-local evidence from the original checks.  No diagnostic check
    # is re-run and these values never participate in admission decisions.
    diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "scene_epoch", int(self.scene_epoch))
        object.__setattr__(
            self, "target_pose", _array(self.target_pose, (7,), name="target_pose")
        )
        object.__setattr__(self, "start_q", _array(self.start_q, (7,), name="start_q"))
        if self.path is not None:
            object.__setattr__(
                self, "path", _arrays(self.path, (7,), name="path waypoint")
            )
        if self.continuation_path is not None:
            object.__setattr__(self, 'continuation_path',
                _arrays(self.continuation_path, (7,), name='continuation waypoint'))
        if self.continuation_target_pose is not None:
            object.__setattr__(self, 'continuation_target_pose',
                _array(self.continuation_target_pose, (7,), name='continuation target'))
        if self.cost is not None:
            if len(self.cost) != 3:
                raise ValueError("path cost must contain eta, length, and count")
            eta = float(self.cost[0])
            length = float(self.cost[1])
            count_value = float(self.cost[2])
            if (
                not np.isfinite(eta)
                or eta < 0.0
                or not np.isfinite(length)
                or length < 0.0
                or not np.isfinite(count_value)
                or count_value < 1.0
                or not count_value.is_integer()
            ):
                raise ValueError(
                    "path cost must be finite, non-negative, and have a positive count"
                )
            object.__setattr__(self, "cost", (eta, length, int(count_value)))
        if self.edge_durations is not None:
            durations = tuple(float(item) for item in self.edge_durations)
            if any(not np.isfinite(item) or item < 0.0 for item in durations):
                raise ValueError("edge durations must be finite and non-negative")
            object.__setattr__(self, "edge_durations", durations)
        object.__setattr__(
            self, "path_fully_validated", bool(self.path_fully_validated)
        )
        object.__setattr__(
            self,
            "continuation_fully_validated",
            bool(self.continuation_fully_validated),
        )

    @property
    def success(self) -> bool:
        return bool(not self.cancelled and self.error is None and self.path)


class PlanningEngine(Protocol):
    def plan(self, request: PlanningRequest, cancelled: CancelCheck) -> PlanningResult:
        ...


class MuJoCoGlobalPlanningEngine(FullPickPlanningMixin):
    """Multi-seed IK + constrained RRT using no live environment state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        joint_names: Sequence[str],
        ee_site_name: str,
        ik_config: Mapping[str, Any] | Any | None,
        robot_collision_body_ids: Sequence[int],
        state_validator: Callable[[PlanningRequest, np.ndarray], bool] | None = None,
        candidate_path_backend: Any | None = None,
        shared_state_evaluation: bool = True,
        minimal_state_evaluation: bool = True,
        adaptive_segment_validation: bool = True,
        warm_roadmap_backend: Any | None = None,
        batch_curve_validation: bool = True,
        rank_goal_routes: bool = False,
    ) -> None:
        # A real-scene adapter may add measured obstacle checks without
        # replacing the shared IK/RRT/certificate pipeline.  It must own its
        # immutable scene snapshot and private scratch state: never capture
        # the live environment's mutable model/data.  Exceptions deliberately
        # propagate to the worker's fail-closed error result.
        if state_validator is not None and not callable(state_validator):
            raise TypeError("state_validator must be callable")
        self._state_validator = state_validator
        self._candidate_path_backend = candidate_path_backend
        self._warm_roadmap_backend = warm_roadmap_backend
        self._static_native_regions = getattr(warm_roadmap_backend, 'static_native_regions', None)
        self._static_native_binding_request = None
        self._static_native_binding_valid = False
        self.adaptive_segment_validation = bool(adaptive_segment_validation)
        self.batch_curve_validation = bool(batch_curve_validation)
        self.rank_goal_routes = bool(rank_goal_routes)
        # MjModel copy constructs independent mutable native storage.  Every
        # MjData and scratch array below belongs exclusively to this worker.
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.ik = DampedLeastSquaresIK(
            self.model,
            self.data,
            tuple(joint_names),
            str(ee_site_name),
            ik_config,
        )
        self.arm_qpos_adrs = self.ik.qpos_indices.copy()
        self.ee_site_id = int(self.ik.ee_site_id)
        self.validation_data = mujoco.MjData(self.model)
        self._validation_jacobian_position = np.zeros(
            (3, self.model.nv), dtype=float
        )
        self._validation_jacobian_rotation = np.zeros(
            (3, self.model.nv), dtype=float
        )
        self.robot_collision_body_ids = {
            int(identifier) for identifier in robot_collision_body_ids
        }
        self.state_evaluator = None
        if shared_state_evaluation:
            from .state_evaluator import SharedStateEvaluator
            self.state_evaluator = SharedStateEvaluator(self.model, self.validation_data,
                                                       minimal=minimal_state_evaluation)
            bind = getattr(state_validator, 'bind_state_evaluator', None)
            if callable(bind):
                bind(self.state_evaluator)
            if self._static_native_regions is not None:
                # A proof query is separate from the raw contacts cache: a
                # consumer that actually requests contacts still gets them.
                self.state_evaluator.native_contacts_prevalidated = self._shared_native_contacts_prevalidated

    @staticmethod
    def _diagnostic_value(value: Any) -> Any:
        """Convert the small evidence samples to strict JSON-compatible data."""

        if isinstance(value, np.ndarray):
            return MuJoCoGlobalPlanningEngine._diagnostic_value(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): MuJoCoGlobalPlanningEngine._diagnostic_value(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MuJoCoGlobalPlanningEngine._diagnostic_value(item) for item in value]
        return value

    def _record_rejection(self, reason: str, **details: Any) -> None:
        progress = getattr(self, '_planning_progress', None)
        if progress is not None:
            progress.rejection(reason)
        diagnostics = getattr(self, "_request_diagnostics", None)
        if diagnostics is None:
            return
        counts = diagnostics["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1
        sample = {"reason": reason, **getattr(self, "_diagnostic_context", {}),
                  **details}
        self._last_rejection = self._diagnostic_value(sample)
        # RRT can reject thousands of states. Keep counts for all of them,
        # but bound evidence storage and serialization cost per request.
        samples = diagnostics["rejection_samples"]
        if len(samples) < 32:
            samples.append(self._last_rejection)

    def _record_ik(self, result: Any, target_pose: np.ndarray, seed: np.ndarray) -> None:
        progress = getattr(self, '_planning_progress', None)
        if progress is not None:
            progress.ik(result.success, getattr(result, 'iterations', 0))
        diagnostics = getattr(self, "_request_diagnostics", None)
        if diagnostics is None:
            return
        attempts = diagnostics["ik_attempts"]
        diagnostics["ik_attempt_count"] += 1
        if len(attempts) < 64:
            attempts.append(self._diagnostic_value({
                **getattr(self, "_diagnostic_context", {}),
                "target_pose7": target_pose, "seed_joints_rad": seed,
                "success": bool(result.success),
                "iterations": getattr(result, "iterations", None),
                "position_error_m": getattr(result, "position_error", None),
                "orientation_error_rad": getattr(result, "orientation_error", None),
                "message": getattr(result, "message", None),
                "joint_positions_rad": getattr(result, "joint_positions", None),
            }))

    @staticmethod
    def _corner_path(path: Sequence[np.ndarray]) -> np.ndarray:
        states = np.asarray(path, dtype=float)
        if states.ndim != 2 or states.shape[1:] != (7,) or len(states) <= 2:
            return states
        kept = [states[0]]
        for index in range(1, len(states) - 1):
            incoming = states[index] - kept[-1]
            outgoing = states[index + 1] - states[index]
            in_norm = float(np.linalg.norm(incoming))
            out_norm = float(np.linalg.norm(outgoing))
            if in_norm <= 1e-12 or out_norm <= 1e-12:
                continue
            cosine = float(np.dot(incoming, outgoing) / (in_norm * out_norm))
            if not (
                cosine >= 1.0 - 1e-8
                and np.linalg.norm(incoming / in_norm - outgoing / out_norm)
                <= 1e-7 * max(in_norm, out_norm, 1.0)
            ):
                kept.append(states[index])
        kept.append(states[-1])
        return np.asarray(kept, dtype=float)

    def _edge_minimum_duration(
        self,
        request: PlanningRequest,
        start: np.ndarray,
        end: np.ndarray,
        cancelled: CancelCheck | None = None,
    ) -> float:
        if np.max(np.abs(np.asarray(end) - np.asarray(start))) <= 1e-12:
            return 0.0
        if not request.tcp_speed_limits_enabled and request.maximum_tcp_linear_speed is None:
            return request.trajectory_minimum_duration
        def checked_jacobian(positions: np.ndarray) -> np.ndarray:
            if cancelled is not None and cancelled():
                raise _PlanningCancelled
            return self.ik.compute_jacobian(positions)

        angular_duration = (sampled_tcp_angular_minimum_duration(
            start,
            end,
            checked_jacobian,
            request.maximum_tcp_angular_speed,
        ) if request.tcp_speed_limits_enabled else 0.)
        linear_duration=(sampled_tcp_angular_minimum_duration(start,end,
            lambda q:checked_jacobian(q)[:3],request.maximum_tcp_linear_speed)
            if request.maximum_tcp_linear_speed is not None else 0.)
        return max(request.trajectory_minimum_duration, angular_duration,linear_duration)

    def _path_profile(
        self,
        request: PlanningRequest,
        path: Sequence[np.ndarray],
        *,
        path_is_certified_corner_path: bool = False,
        cancelled: CancelCheck | None = None,
    ) -> tuple[tuple[float, float, int], tuple[float, ...]]:
        states = (
            np.asarray(path, dtype=float)
            if path_is_certified_corner_path
            else self._corner_path(path)
        )
        if states.ndim != 2 or states.shape[1:] != (7,) or len(states) < 1:
            return (
                (float("inf"), float("inf"), 2**31 - 1),
                (),
            )
        if len(states) == 1:
            return ((0.0, 0.0, 1), ())
        eta = 0.0
        edge_durations: list[float] = []
        for start, end in zip(states, states[1:]):
            if cancelled is not None and cancelled():
                raise _PlanningCancelled
            trajectory = build_certified_edge_trajectory(
                start,
                end,
                request.trajectory_velocity_limits,
                request.trajectory_acceleration_limits,
                minimum_duration=self._edge_minimum_duration(
                    request, start, end, cancelled
                ),
                backend=(request.joint_trajectory_backend
                         if request.stage == 'pregrasp' and not request.tcp_speed_limits_enabled
                         else 'quintic'),
                jerk_limits=request.trajectory_jerk_limits,
            )
            eta += trajectory.duration
            edge_durations.append(float(trajectory.duration))
        span = np.maximum(request.upper_limits - request.lower_limits, 1e-6)
        length = float(
            np.sum(np.linalg.norm(np.diff(states, axis=0) / span, axis=1))
        )
        return (
            (float(eta), length, len(states)),
            tuple(edge_durations),
        )

    def _certified_corner_path(
        self,
        planner: ConstrainedRRTConnect,
        path: Sequence[np.ndarray],
        cancelled: CancelCheck,
    ) -> np.ndarray | None:
        """Compress dense samples only after validating every new chord.

        The planner certifies the dense path it returns.  Near-collinear
        compression changes those edges, however slightly, so its longer
        chords need an explicit worker-side certificate before the result can
        truthfully set ``path_fully_validated``.  If a numerical corner fails,
        retain the original already-certified dense route instead of weakening
        any collision, cone, singularity, height or soft-limit constraint.
        """

        dense = np.asarray(path, dtype=float)
        corners = self._corner_path(dense)
        for start, end in zip(corners, corners[1:]):
            if cancelled():
                return None
            if not planner.edge_valid(start, end):
                if cancelled():
                    return None
                return dense
        return corners

    def _hinted_path(
        self,
        request: PlanningRequest,
        planner: ConstrainedRRTConnect,
        candidate: np.ndarray,
        cancelled: CancelCheck,
        *,
        shorten: bool = False,
    ) -> tuple[
        np.ndarray,
        tuple[float, float, int],
        tuple[float, ...],
    ] | None:
        """Validate, densify, compress, and time one suggested route.

        Hints deliberately bypass no checks.  Validate their original long
        segments first, then ask the request-scoped planner to create and
        certify execution-resolution samples.  Compression can introduce new
        chords, so ``_certified_corner_path`` revalidates those as well (and
        retains the already-certified dense path if a shortcut is unsafe).
        Any failure simply declines the suggestion and lets normal direct/RRT
        planning continue.
        """

        if not request.route_waypoint_hints or cancelled():
            return None
        waypoints = (
            np.asarray(request.start_q, dtype=float),
            *request.route_waypoint_hints,
            np.asarray(candidate, dtype=float),
        )
        for state in waypoints:
            if cancelled() or not planner.state_valid(state):
                return None
        for start, end in zip(waypoints, waypoints[1:]):
            if cancelled() or not planner.edge_valid(start, end):
                return None
        dense = planner.densify(waypoints)
        if dense is None or cancelled():
            return None
        corner_path = self._certified_corner_path(
            planner, dense, cancelled
        )
        if corner_path is None:
            return None
        try:
            cost, edge_durations = self._path_profile(
                request,
                corner_path,
                path_is_certified_corner_path=True,
                cancelled=cancelled,
            )
            if shorten and len(corner_path) > 2:
                # A sparse startup PRM supplies topology, not a well-shaped
                # motion. Shortcut its DENSE samples with the original RRT
                # checker: shortening just its 3 vertices can only retry the
                # already-blocked start->goal chord, leaving a huge detour.
                started = time.perf_counter()
                original_cost = cost
                shorter = planner._shortcut(dense)
                if cancelled():
                    raise _PlanningCancelled
                shorter_corners = self._certified_corner_path(
                    planner, shorter, cancelled)
                selected = False
                shorter_cost = None
                if shorter_corners is not None:
                    shorter_cost, shorter_durations = self._path_profile(
                        request, shorter_corners, path_is_certified_corner_path=True,
                        cancelled=cancelled)
                    # Shorter joint distance can introduce extra stops. Keep
                    # the original ETA objective and its existing tie breaks.
                    if shorter_cost < cost:
                        corner_path = shorter_corners
                        cost, edge_durations = shorter_cost, shorter_durations
                        selected = True
                self._request_diagnostics.setdefault("warm_roadmap", {}).setdefault(
                    "route_shortening", []).append({
                        "method": "existing_rrt_shortcut_on_dense_prm_route",
                        "original_cost": original_cost, "shortened_cost": shorter_cost,
                        "selected": selected,
                        "attempts_budget": planner.config.shortcut_attempts,
                        "wall_s": time.perf_counter()-started,
                    })
        except _PlanningCancelled:
            return None
        return corner_path, cost, edge_durations

    def _path_cost(
        self, request: PlanningRequest, path: Sequence[np.ndarray]
    ) -> tuple[float, float, int]:
        """Compatibility wrapper for tests and external diagnostic callers."""

        return self._path_profile(request, path)[0]

    def _state_valid(
        self, request: PlanningRequest, positions: np.ndarray,
        cancelled: CancelCheck, *, minimum_tcp_height: float | None = None,
    ) -> bool:
        """Reuse only identical successful states inside one immutable request.

        Direct-path sampling, densification and chord certification frequently
        visit exactly the same state. A new request (including a changed capture
        clock), a different height constraint, or cancellation cannot reuse it.
        """
        cache = getattr(self, "_request_valid_states", None)
        if cache is None:
            return self._check_state_valid(request, positions, cancelled,
                                           minimum_tcp_height=minimum_tcp_height)
        if cancelled():
            self._record_rejection("cancelled")
            return False
        if self._validation_cache_request is not request:
            cache.clear()
            self._validation_cache_request = request
        q = np.asarray(positions, dtype=float).reshape(-1)
        height = request.minimum_tcp_height if minimum_tcp_height is None else float(minimum_tcp_height)
        key = (height, q.tobytes())
        counts = self._request_diagnostics.setdefault("state_validation", {"computed": 0, "exact_reused": 0})
        if key in cache:
            counts["exact_reused"] += 1
            return True
        counts["computed"] += 1
        valid = self._check_state_valid(request, q, cancelled, minimum_tcp_height=minimum_tcp_height)
        if valid:
            cache.add(key)
        return valid

    def _shared_native_contacts_prevalidated(self, full_qpos):
        bank = self._static_native_regions
        request = self._static_native_binding_request
        if request is None or not bank.matches_qpos(full_qpos):
            return False
        q = np.asarray(full_qpos)[self.arm_qpos_adrs]
        return self._static_native_region_covers(request, q, q)

    def _static_native_region_covers(self, request, lower, upper):
        """Skip only native contacts inside an independently preproved box."""
        bank = getattr(self, '_static_native_regions', None)
        if bank is None:
            return False
        from .static_native_regions import _supported
        metrics = self._request_diagnostics.setdefault('static_native_regions', {
            'queries': 0, 'hits': 0, 'query_wall_s': 0., 'binding_wall_s': 0.,
            'bypassed_uncovered_or_incompatible': 0,
            'scope': 'native_contacts_only_measured_scene_and_target_still_checked',
            'preprocessing': dict(bank.report)})
        if self._static_native_binding_request is not request:
            self._static_native_binding_request = request
            binding_started = time.perf_counter()
            # A preheated graph samples the whole joint domain.  If neither
            # this request's start nor its first checked candidate is covered,
            # keep the original checker for the request instead of repeatedly
            # searching irrelevant boxes or hashing a model it cannot reuse.
            # This only forgoes an optional acceleration, never a constraint.
            covered = (bank.contains(request.start_q) or bank.covers_box(lower, upper))
            self._static_native_binding_valid = (
                covered and bank.matches_qpos(request.full_qpos)
                and bank.compatible(self.model, self.ik.joint_ids,
                                    self.robot_collision_body_ids))
            metrics['request_has_preheated_coverage'] = covered
            metrics['binding_wall_s'] += time.perf_counter()-binding_started
        if not self._static_native_binding_valid:
            metrics['bypassed_uncovered_or_incompatible'] += 1
            return False
        began = time.perf_counter()
        metrics['queries'] += 1
        valid = (self._static_native_binding_valid and _supported(self.model)
                 and bank.covers_box(lower, upper))
        metrics['hits'] += int(valid)
        metrics['query_wall_s'] += time.perf_counter()-began
        return valid

    @planning_stage('state_constraints_and_collision', events=False)
    def _check_state_valid(
        self,
        request: PlanningRequest,
        positions: np.ndarray,
        cancelled: CancelCheck,
        *,
        minimum_tcp_height: float | None = None,
        check_external: bool = True,
    ) -> bool:
        if cancelled():
            self._record_rejection("cancelled")
            return False
        q = np.asarray(positions, dtype=float).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            self._record_rejection("invalid_joint_vector", joint_positions_rad=q)
            return False
        margin = np.minimum(
            q - request.lower_limits,
            request.upper_limits - q,
        )
        if np.min(margin) < request.minimum_joint_margin - 1e-9:
            index = int(np.argmin(margin))
            self._record_rejection("joint_margin", joint_index=index,
                joint_position_rad=q[index], lower_limit_rad=request.lower_limits[index],
                upper_limit_rad=request.upper_limits[index], margin_rad=margin[index],
                required_margin_rad=request.minimum_joint_margin)
            return False
        scratch = self.validation_data
        evaluator = self.state_evaluator
        if evaluator is None:
            scratch.qpos[:] = request.full_qpos
            scratch.qpos[self.arm_qpos_adrs] = q
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(self.model, scratch)
        else:
            full = np.array(request.full_qpos, copy=True)
            full[self.arm_qpos_adrs] = q
            evaluator.prepare(full)
        # One forward pass supplies both kinematic safety and collision state.
        # Calling DLSIK.compute_jacobian here used to perform a second forward
        # at every RRT sample, which dominated hard-route planning latency.
        if evaluator is None:
            mujoco.mj_jacSite(
            self.model,
            scratch,
            self._validation_jacobian_position,
            self._validation_jacobian_rotation,
            self.ee_site_id,
            )
        dofs = self.ik.dof_indices
        jacobian = np.vstack(
            (
                self._validation_jacobian_position[:, dofs],
                self._validation_jacobian_rotation[:, dofs],
            )
        ) if evaluator is None else evaluator.jacobian(self.ee_site_id, dofs)
        if evaluator is None:
            scaled = jacobian.copy()
            scaled[3:] *= 0.30
            singular_values = np.linalg.svd(scaled, compute_uv=False)
        else:
            singular_values = evaluator.singular_values(self.ee_site_id, dofs)
        condition = (
            float("inf")
            if singular_values[-1] <= 1e-12
            else float(singular_values[0] / singular_values[-1])
        )
        if condition > request.singularity_hard_stop_condition:
            self._record_rejection("jacobian_condition", condition=condition,
                maximum_condition=request.singularity_hard_stop_condition)
            return False
        height_limit = (
            request.minimum_tcp_height
            if minimum_tcp_height is None
            else float(minimum_tcp_height)
        )
        if float(scratch.site_xpos[self.ee_site_id, 2]) < (
            height_limit - 1e-9
        ):
            self._record_rejection("minimum_tcp_height",
                tcp_height_m=float(scratch.site_xpos[self.ee_site_id, 2]),
                minimum_tcp_height_m=height_limit)
            return False
        world_axis = (
            scratch.site_xmat[self.ee_site_id].reshape(3, 3)
            @ request.approach_axis_local
        )
        if -float(world_axis[2]) < request.minimum_downward_alignment - 1e-9:
            self._record_rejection("downward_alignment", alignment=-float(world_axis[2]),
                minimum_alignment=request.minimum_downward_alignment)
            return False
        native_prevalidated = self._static_native_region_covers(request, q, q)
        contacts = (() if native_prevalidated else
                    (tuple((int(c.geom1), int(c.geom2), float(c.dist))
                          for c in scratch.contact[:scratch.ncon])
                    if evaluator is None else evaluator.contacts()))
        for geom1, geom2, contact_distance in contacts:
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            robot1 = body1 in self.robot_collision_body_ids
            robot2 = body2 in self.robot_collision_body_ids
            if not (robot1 or robot2):
                continue
            other_body = body2 if robot1 else body1
            other_name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, other_body
                )
                or ""
            )
            if other_name.startswith("object_"):
                continue
            if contact_distance <= 1e-6:
                self._record_rejection("model_contact", distance_m=contact_distance,
                    geom1=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                    geom2=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                    body1=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1),
                    body2=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2))
                return False
        if cancelled():
            self._record_rejection("cancelled")
            return False
        validator = self._state_validator
        if validator is not None and check_external:
            # Do not let an extension mutate a candidate after the built-in
            # bounds, kinematics and collision checks have certified it.
            candidate = q.copy()
            candidate.setflags(write=False)
            if not bool(validator(request, candidate)):
                self._record_rejection("external_state_validator",
                    validator_details=getattr(validator, "last_rejection", None))
                return False
        return not cancelled()

    @planning_stage('batch_scene_validation')
    def _states_valid(self, request, states, cancelled, *, minimum_tcp_height=None):
        """Batch only the measured-scene extension, retaining native checks."""
        validator = self._state_validator
        batch = getattr(validator, 'batch', None)
        if not callable(batch):
            return all(self._state_valid(request, q, cancelled,
                       minimum_tcp_height=minimum_tcp_height) for q in states)
        if cancelled():
            return False
        cache = getattr(self, '_request_valid_states', None)
        if cache is not None and self._validation_cache_request is not request:
            cache.clear()
            self._validation_cache_request = request
        height = request.minimum_tcp_height if minimum_tcp_height is None else float(minimum_tcp_height)
        missing, keys, seen = [], [], set()
        counts = getattr(self, '_request_diagnostics', {}).setdefault('state_validation',
                                                      {'computed': 0, 'exact_reused': 0})
        for q in states:
            q = np.asarray(q, dtype=float)
            key = (height, q.tobytes())
            if (cache is not None and key in cache) or key in seen:
                counts['exact_reused'] += 1
                continue
            counts['computed'] += 1
            if not self._check_state_valid(request, q, cancelled,
                    minimum_tcp_height=minimum_tcp_height, check_external=False):
                return False
            missing.append(q.copy()); keys.append(key); seen.add(key)
        if missing:
            positions = np.array(missing)
            positions.setflags(write=False)
            if cancelled() or not bool(batch(request, positions)):
                self._record_rejection('external_state_validator',
                    validator_details=getattr(validator, 'last_rejection', None))
                return False
            if cancelled():
                return False
            if cache is not None:
                cache.update(keys)
        return True

    @planning_stage('inverse_kinematics')
    def _solve_request_ik(
        self,
        request: PlanningRequest,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        qpos_seed: np.ndarray,
        cancelled: CancelCheck,
    ) -> IKResult:
        """Search within the same joint window used to accept this request.

        The worker owns this solver. Intersect with its existing bounds so an
        expanded measured starting range cannot enlarge normal target IK.
        Restore the arrays even on cancellation or failure: the next request
        may use a different margin, and limits must not shrink cumulatively.
        """
        solver = self.ik
        original = (solver.lower_limits, solver.upper_limits, solver.center_positions)
        lower = np.maximum(original[0], request.lower_limits + request.minimum_joint_margin)
        upper = np.minimum(original[1], request.upper_limits - request.minimum_joint_margin)
        if np.any(lower > upper):
            raise ValueError("planning request has no joint range shared with target IK")
        try:
            solver.lower_limits, solver.upper_limits = lower, upper
            solver.center_positions = solver._joint_centers()
            return solver.solve(
                position, quaternion, qpos_seed=qpos_seed, cancelled=cancelled,
            )
        finally:
            solver.lower_limits, solver.upper_limits, solver.center_positions = original

    def _goal_candidates(
        self,
        request: PlanningRequest,
        target_pose: np.ndarray,
        cancelled: CancelCheck,
    ) -> list[np.ndarray]:
        self.data.qpos[:] = request.full_qpos
        self.data.qvel[:] = 0.0
        candidates: list[np.ndarray] = []
        for seed in request.goal_seeds:
            if cancelled():
                return []
            result = self._solve_request_ik(
                request,
                target_pose[:3],
                target_pose[3:],
                qpos_seed=seed,
                cancelled=cancelled,
            )
            if not result.success:
                continue
            candidate = np.asarray(result.joint_positions, dtype=float)
            if not self._state_valid(request, candidate, cancelled):
                continue
            if any(
                np.max(np.abs(candidate - item)) < 1e-4
                for item in candidates
            ):
                continue
            candidates.append(candidate.copy())
        span = np.maximum(request.upper_limits - request.lower_limits, 1e-6)
        candidates.sort(
            key=lambda item: (
                float(np.linalg.norm((item - request.start_q) / span)),
                -float(
                    np.min(
                        np.minimum(
                            item - request.lower_limits,
                            request.upper_limits - item,
                        )
                    )
                ),
            )
        )
        return candidates

    def _solve_goal_seed(
        self,
        request: PlanningRequest,
        target_pose: np.ndarray,
        seed: np.ndarray,
        cancelled: CancelCheck,
    ) -> np.ndarray | None:
        if cancelled():
            return None
        diagnostics = getattr(self, '_request_diagnostics', {})
        diagnostics['ik_seeds_tried'] = diagnostics.get('ik_seeds_tried', 0)+1
        continuous_report = getattr(self, '_request_diagnostics', {}).get('continuous_intercept')
        rechecking_contact = (continuous_report is not None
            and request.target_is_contact and not request.continuous_intercept)
        recheck_started = time.perf_counter() if rechecking_contact else None
        result = self._solve_request_ik(
            request,
            target_pose[:3],
            target_pose[3:],
            qpos_seed=seed,
            cancelled=cancelled,
        )
        if rechecking_contact:
            continuous_report['route_goal_recheck_count'] += 1
            continuous_report['route_goal_recheck_s'] += time.perf_counter()-recheck_started
        self._record_ik(result, target_pose, seed)
        if not result.success:
            self._record_rejection("ik_not_converged",
                position_error_m=getattr(result, "position_error", None),
                orientation_error_rad=getattr(result, "orientation_error", None),
                iterations=getattr(result, "iterations", None),
                message=getattr(result, "message", None))
            return None
        candidate = np.asarray(result.joint_positions, dtype=float)
        return (
            candidate.copy()
            if self._state_valid(request, candidate, cancelled)
            else None
        )

    def _continuation_is_valid(self, request, pose_index, start_pose, start_q, cancelled):
        started = time.perf_counter()
        try:
            return self._check_continuation(request, pose_index, start_pose, start_q, cancelled)
        finally:
            if request.continuation_target_poses is not None:
                elapsed = time.perf_counter() - started
                timing = getattr(self, '_request_diagnostics', {}).setdefault(
                    'continuation_timing', {'checks': 0, 'total_s': 0., 'max_check_s': 0.})
                timing['checks'] += 1
                timing['total_s'] += elapsed
                timing['max_check_s'] = max(timing['max_check_s'], elapsed)

    @staticmethod
    def _continuation_curve(request, start_q, end_q):
        return BoundaryQuinticJointTrajectory(
            start_q, end_q, request.trajectory_velocity_limits,
            request.trajectory_acceleration_limits,
            initial_velocity=request.continuation_start_velocity,
            initial_acceleration=request.continuation_start_acceleration,
            jerk_limits=request.trajectory_jerk_limits,
            minimum_duration=request.timed_continuation_duration_s,
            allow_path_change=True)

    @staticmethod
    def _curve_validation_intervals(request, trajectory):
        lower_v, upper_v = trajectory.derivative_bounds(1)
        peak_v = np.maximum(np.abs(lower_v), np.abs(upper_v))
        normalized_rate = np.linalg.norm(peak_v / (
            request.upper_limits-request.lower_limits-2*request.minimum_joint_margin))
        return max(1, int(math.ceil(trajectory.duration * max(
            normalized_rate / request.planner_config.normalized_edge_resolution,
            float(np.max(peak_v)) / request.planner_config.execution_joint_step))))

    def _check_continuation(
        self,
        request: PlanningRequest,
        pose_index: int,
        start_pose: np.ndarray,
        start_q: np.ndarray,
        cancelled: CancelCheck,
    ) -> bool:
        """Prove a PREGRASP branch can follow the Cartesian capture corridor.

        Reaching a high pregrasp pose does not imply that a redundant 7-DOF
        solution can descend beside the narrow belt.  For each candidate,
        warm-start a bounded sequence of Cartesian IK samples down to the
        corresponding capture pose and certify every joint chord with the
        same collision/cone/singularity/soft-limit checks.  This work belongs
        to the isolated worker; the 50 Hz control thread only consumes the
        chosen branch.
        """

        continuations = request.continuation_target_poses
        if continuations is None:
            return True
        if pose_index < 0 or pose_index >= len(continuations):
            self._record_rejection("continuation_missing_target", pose_index=pose_index)
            return False
        target = np.asarray(continuations[pose_index], dtype=float)
        source = np.asarray(start_pose, dtype=float)
        position_distance = float(np.linalg.norm(target[:3] - source[:3]))
        source_quaternion = source[3:] / max(
            float(np.linalg.norm(source[3:])), 1e-12
        )
        target_quaternion = target[3:] / max(
            float(np.linalg.norm(target[3:])), 1e-12
        )
        if float(np.dot(source_quaternion, target_quaternion)) < 0.0:
            target_quaternion = -target_quaternion
        quaternion_dot = float(
            np.clip(
                np.dot(source_quaternion, target_quaternion), -1.0, 1.0
            )
        )
        orientation_distance = 2.0 * math.acos(
            min(1.0, abs(quaternion_dot))
        )
        steps = max(
            1,
            int(
                math.ceil(
                    max(
                        position_distance
                        / request.continuation_cartesian_step,
                        orientation_distance / 0.10,
                    )
                )
            ),
        )
        height_limit = request.continuation_minimum_tcp_height
        if height_limit is None:
            self._record_rejection("continuation_missing_height")
            return False
        state_valid = lambda state: self._state_valid(
            request,
            state,
            cancelled,
            minimum_tcp_height=height_limit,
        )
        planner = ConstrainedRRTConnect(
            request.lower_limits + request.minimum_joint_margin,
            request.upper_limits - request.minimum_joint_margin,
            state_valid,
            config=request.planner_config,
            seed=request.planner_seed + 900_001 + pose_index,
            cancelled=cancelled,
            states_valid=(lambda states: self._states_valid(
                request, states, cancelled, minimum_tcp_height=height_limit))
                if callable(getattr(self._state_validator, "batch", None)) else None,
            segment_valid=self._segment_validator(request, cancelled,
                                                   minimum_tcp_height=height_limit),
        )
        previous_q = np.asarray(start_q, dtype=float).copy()
        continuation_path = [previous_q.copy()]
        for index in range(1, steps + 1):
            self._diagnostic_context = {
                **getattr(self, "_diagnostic_context", {}),
                "check": "continuation", "continuation_step": index,
                "continuation_steps": steps,
            }
            if cancelled():
                self._record_rejection("cancelled")
                return False
            fraction = index / steps
            quaternion = (
                (1.0 - fraction) * source_quaternion
                + fraction * target_quaternion
            )
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm <= 1e-12:
                self._record_rejection("continuation_invalid_quaternion")
                return False
            quaternion /= quaternion_norm
            position = source[:3] + fraction * (target[:3] - source[:3])
            result = self._solve_request_ik(
                request,
                position,
                quaternion,
                qpos_seed=previous_q,
                cancelled=cancelled,
            )
            self._record_ik(result, np.r_[position, quaternion], previous_q)
            if not result.success:
                self._record_rejection("continuation_ik_not_converged",
                    position_error_m=getattr(result, "position_error", None),
                    orientation_error_rad=getattr(result, "orientation_error", None),
                    iterations=getattr(result, "iterations", None),
                    message=getattr(result, "message", None))
                return False
            candidate = np.asarray(result.joint_positions, dtype=float)
            if candidate.shape != (7,) or not np.all(np.isfinite(candidate)):
                self._record_rejection("continuation_invalid_joint_vector")
                return False
            joint_step = float(np.max(np.abs(candidate - previous_q)))
            if joint_step > request.continuation_max_joint_step + 1e-9:
                self._record_rejection("continuation_joint_step", joint_step_rad=joint_step,
                    maximum_joint_step_rad=request.continuation_max_joint_step)
                return False
            if not state_valid(candidate):
                self._record_rejection("continuation_state_rejected",
                    state_rejection=getattr(self, "_last_rejection", None))
                return False
            if not planner.edge_valid(previous_q, candidate):
                self._record_rejection("continuation_edge_rejected",
                    state_rejection=getattr(self, "_last_rejection", None))
                return False
            previous_q = candidate.copy()
            continuation_path.append(previous_q.copy())
        trajectory = None
        if request.timed_continuation_duration_s is not None:
            try:
                trajectory = self._continuation_curve(request, start_q, previous_q)
            except ValueError as error:
                self._record_rejection("timed_continuation_dynamics", message=str(error))
                return False
            # A future splice has an absolute end clock: stretching its curve
            # would change the interception pose and cannot be done silently.
            if (request.continuation_start_time_s is not None
                    and trajectory.duration > request.timed_continuation_duration_s + 1e-8):
                self._record_rejection("timed_continuation_deadline",
                    required_duration_s=trajectory.duration,
                    available_duration_s=request.timed_continuation_duration_s)
                return False
            # Certify the NEW curve, not just the Cartesian IK polyline above.
            # Time density bounds each joint's possible displacement by the
            # original execution sample spacing, including curved excursions.
            intervals = self._curve_validation_intervals(request, trajectory)
            previous = np.asarray(start_q)
            times = np.linspace(0., trajectory.duration, intervals + 1)
            batched = (self.batch_curve_validation
                       and callable(getattr(self._state_validator, 'batch', None)))
            batch_size = 32 if batched else 1
            for offset in range(0, len(times), batch_size):
                samples = []
                for t in times[offset:offset+batch_size]:
                    if cancelled():
                        self._record_rejection('cancelled')
                        return False
                    q, velocity, _ = trajectory.sample_full(t)
                    samples.append((t, q, velocity))
                if batched:
                    self._diagnostic_context = {**getattr(self, '_diagnostic_context', {}),
                        'check': 'timed_continuation', 'trajectory_time_s': None,
                        'trajectory_batch_time_range_s': [float(samples[0][0]), float(samples[-1][0])]}
                    counts = getattr(self, '_request_diagnostics', {}).setdefault('final_curve_batch_validation',
                        {'batches': 0, 'original_grid_states': 0, 'maximum_batch_size': 32})
                    counts['batches'] += 1
                    counts['original_grid_states'] += len(samples)
                    if not self._states_valid(request, [q for _, q, _ in samples], cancelled,
                            minimum_tcp_height=height_limit):
                        self._record_rejection('timed_continuation_curve_rejected')
                        return False
                for t, q, velocity in samples:
                    self._diagnostic_context = {**getattr(self, "_diagnostic_context", {}),
                        "check": "timed_continuation", "trajectory_time_s": float(t)}
                    if cancelled() or not state_valid(q) or not planner.edge_valid(previous, q):
                        self._record_rejection("timed_continuation_curve_rejected")
                        return False
                    if request.continuation_tcp_speed_limits is not None:
                        self.data.qpos[self.arm_qpos_adrs] = q
                        mujoco.mj_forward(self.model, self.data)
                        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros_like(jacp)
                        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
                        angular = float(np.linalg.norm(jacr[:, self.ik.dof_indices] @ velocity))
                        linear = float(np.linalg.norm(jacp[:, self.ik.dof_indices] @ velocity))
                        if (angular > request.continuation_tcp_speed_limits[1] + 1e-9
                                or linear > request.continuation_tcp_speed_limits[0] + 1e-9):
                            self._record_rejection("timed_continuation_tcp_speed",
                                angular_speed_rad_s=angular, linear_speed_m_s=linear)
                            return False
                    previous = q
        # Request identity is part of the key: no geometry result crosses a
        # moving-target snapshot, a scene change, or a delivery recertification.
        cache = getattr(self, '_certified_continuations', None)
        if cache is not None:
            cache[(id(request), np.asarray(start_q).tobytes())] = (
                tuple(continuation_path), trajectory, target.copy())
        return True

    @planning_stage('segment_planner_setup', events=False)
    def _planner(
        self,
        request: PlanningRequest,
        *,
        seed: int,
        cancelled: CancelCheck,
    ) -> ConstrainedRRTConnect:
        return ConstrainedRRTConnect(
            request.lower_limits + request.minimum_joint_margin,
            request.upper_limits - request.minimum_joint_margin,
            lambda state: self._state_valid(request, state, cancelled),
            config=request.planner_config,
            seed=seed,
            cancelled=cancelled,
            states_valid=(lambda states: self._states_valid(request, states, cancelled))
                if callable(getattr(self._state_validator, 'batch', None)) else None,
            segment_valid=self._segment_validator(request, cancelled),
        )

    @planning_stage('segment_certificate_setup', events=False)
    def _segment_validator(self, request, cancelled, *, minimum_tcp_height=None):
        """Optional proof accelerator; unsupported inputs retain the dense oracle."""
        if not self.adaptive_segment_validation:
            return None
        from .planning_segment_validation import PlanningSegmentValidator
        if not PlanningSegmentValidator.available(self):
            return None
        cache = getattr(self, '_segment_validators', None)
        if cache is None:
            cache = self._segment_validators = {}
        height = request.minimum_tcp_height if minimum_tcp_height is None else float(minimum_tcp_height)
        key = (id(request), height)
        if key not in cache:
            try:
                adapter = PlanningSegmentValidator(self, request, cancelled,
                    minimum_tcp_height=minimum_tcp_height)
                cache[key] = adapter if adapter.supported else None
            except Exception as error:
                cache[key] = None
                getattr(self, '_request_diagnostics', {}).setdefault(
                    'adaptive_segment_fallbacks', []).append(
                        {'reason': type(error).__name__, 'message': str(error)})
        adapter = cache[key]
        return None if adapter is None else adapter.validate

    @planning_stage('cartesian_branch_probe')
    def _local_cartesian_path(
        self,
        request: PlanningRequest,
        target_pose: np.ndarray,
        cancelled: CancelCheck,
    ) -> tuple[np.ndarray, tuple[float, float, int], tuple[float, ...]] | None:
        """Certify a short warm-IK Cartesian corridor without an RRT fallback."""

        if cancelled():
            return None
        self.data.qpos[:] = request.full_qpos
        self.data.qpos[self.arm_qpos_adrs] = request.start_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        source = np.empty(7, dtype=float)
        source[:3] = self.data.site_xpos[self.ee_site_id]
        mujoco.mju_mat2Quat(
            source[3:], self.data.site_xmat[self.ee_site_id]
        )
        target = np.asarray(target_pose, dtype=float)
        source_quaternion = source[3:] / max(
            float(np.linalg.norm(source[3:])), 1e-12
        )
        target_quaternion = target[3:] / max(
            float(np.linalg.norm(target[3:])), 1e-12
        )
        if float(np.dot(source_quaternion, target_quaternion)) < 0.0:
            target_quaternion = -target_quaternion
        orientation_distance = 2.0 * math.acos(
            min(
                1.0,
                abs(
                    float(
                        np.clip(
                            np.dot(source_quaternion, target_quaternion),
                            -1.0,
                            1.0,
                        )
                    )
                ),
            )
        )
        position_distance = float(np.linalg.norm(target[:3] - source[:3]))
        steps = max(
            1,
            int(
                math.ceil(
                    max(
                        position_distance
                        / request.continuation_cartesian_step,
                        orientation_distance / 0.10,
                    )
                )
            ),
        )
        planner = self._planner(
            request,
            seed=request.planner_seed + 700_001,
            cancelled=cancelled,
        )
        previous_q = np.asarray(request.start_q, dtype=float).copy()
        path = [previous_q.copy()]
        maximum_delta = float(request.maximum_goal_joint_delta or 0.0)
        for index in range(1, steps + 1):
            if cancelled():
                return None
            fraction = index / steps
            quaternion = (
                (1.0 - fraction) * source_quaternion
                + fraction * target_quaternion
            )
            norm = float(np.linalg.norm(quaternion))
            if norm <= 1e-12:
                return None
            quaternion /= norm
            position = source[:3] + fraction * (target[:3] - source[:3])
            result = self._solve_request_ik(
                request,
                position,
                quaternion,
                qpos_seed=previous_q,
                cancelled=cancelled,
            )
            if not result.success:
                return None
            candidate = np.asarray(result.joint_positions, dtype=float)
            if (
                candidate.shape != (7,)
                or not np.all(np.isfinite(candidate))
                or np.max(np.abs(candidate - previous_q))
                > request.continuation_max_joint_step + 1e-9
                or np.max(np.abs(candidate - request.start_q))
                > maximum_delta + 1e-9
                or not self._state_valid(request, candidate, cancelled)
                or not planner.edge_valid(previous_q, candidate)
            ):
                return None
            path.append(candidate.copy())
            previous_q = candidate.copy()
        corner_path = self._certified_corner_path(
            planner, path, cancelled
        )
        if corner_path is None:
            return None
        try:
            cost, durations = self._path_profile(
                request,
                corner_path,
                path_is_certified_corner_path=True,
                cancelled=cancelled,
            )
        except _PlanningCancelled:
            return None
        return corner_path, cost, durations

    def _success_result(
        self,
        request: PlanningRequest,
        target_pose: np.ndarray,
        path: Sequence[np.ndarray],
        cost: tuple[float, float, int],
        edge_durations: tuple[float, ...],
        started: float,
    ) -> PlanningResult:
        corner_path = np.asarray(path, dtype=float)
        source = getattr(self, '_diagnostic_context', {}).get('check', 'direct')
        reports = self._request_diagnostics.get('goal_route_selection', [])
        if reports and reports[-1].get('selected'):
            source = reports[-1]['selected']['route_source']
        self._request_diagnostics['route_source'] = source
        continuation = getattr(self, '_certified_continuations', {}).get(
            (id(request), corner_path[-1].tobytes()))
        return PlanningResult(
            target_identity=request.target_identity, request_id=request.request_id,
            generation=request.generation,
            stage=request.stage,
            target_id=request.target_id,
            scene_epoch=request.scene_epoch,
            target_pose=target_pose,
            start_q=request.start_q,
            path=tuple(corner_path),
            cost=cost,
            edge_durations=edge_durations,
            path_fully_validated=True,
            continuation_fully_validated=(
                request.continuation_target_poses is not None
            ),
            continuation_path=None if continuation is None else continuation[0],
            continuation_trajectory=None if continuation is None else continuation[1],
            continuation_target_pose=None if continuation is None else continuation[2],
            planning_time_s=time.perf_counter() - started,
        )

    def _first_safe_appointment_valid(self, request, goal, cost):
        if (request.selection_policy != 'first_safe' or request.intercept_velocity_m_s is None):
            return True
        saved = self._certified_continuations.get((id(request), np.asarray(goal).tobytes()))
        descent = request.intercept_descent_duration_s
        if saved is not None and saved[1] is not None:
            descent = max(descent, saved[1].duration + max(0., descent
                - float(request.timed_continuation_duration_s or 0.)))
        required = (request.intercept_motion_start_time_s
            + self._approach_time_forecast(request, cost[0]) + descent)
        valid = required <= request.planned_target_time_s+1e-9
        if not valid:
            self._record_rejection('first_safe_contact_deadline',
                required_capture_time_s=required, planned_target_time_s=request.planned_target_time_s)
        return valid

    def _approach_time_forecast(self, request, motion_s):
        # The measured-scene policy preserves the certified start exactly.
        # Changed feedback adds a short certified connector on receipt, whose
        # existing minimum duration was absent from the old appointment ETA.
        connector = (request.trajectory_minimum_duration
            if self.rank_goal_routes and not request.fixed_high_continuation else 0.)
        return (float(motion_s)+connector)*request.intercept_motion_time_scale

    def _intercept_request(
        self, request: PlanningRequest, cancelled: CancelCheck, *, minimum_motion_s: float = 0.,
    ) -> PlanningRequest:
        """Forecast IK/trajectory arrival before paying for full certification.

        The estimate selects a pose and an IK seed, never a motion command or
        a path certificate. Every returned motion still passes ``_plan``.
        """
        velocity = request.intercept_velocity_m_s
        if velocity is None or request.stage != "pregrasp":
            return request
        current = request
        seeds = request.goal_seeds
        started = time.perf_counter()
        for iteration in range(request.max_goal_candidates):
            if cancelled():
                break
            candidate = None
            for seed_index, seed in enumerate(seeds):
                self._diagnostic_context = {"phase": "intercept_forecast",
                    "forecast_iteration": iteration, "seed_index": seed_index,
                    "planned_target_time_s": current.planned_target_time_s}
                candidate = self._solve_goal_seed(current, current.target_pose, seed, cancelled)
                if candidate is not None:
                    break
            if candidate is None:
                break
            cost, _durations = self._path_profile(
                current, [current.start_q, *current.route_waypoint_hints, candidate],
                cancelled=cancelled)
            capture_time = max(float(request.planned_target_time_s),
                float(request.intercept_motion_start_time_s)
                + self._approach_time_forecast(current, max(minimum_motion_s, cost[0]))
                + request.intercept_descent_duration_s + time.perf_counter()-started)
            shift = velocity*(capture_time-float(current.planned_target_time_s))
            targets = tuple(np.r_[pose[:3]+shift, pose[3:]] for pose in current.target_poses)
            continuations = None if current.continuation_target_poses is None else tuple(
                np.r_[pose[:3]+shift, pose[3:]] for pose in current.continuation_target_poses)
            current = replace(current, target_poses=targets,
                continuation_target_poses=continuations,
                planned_target_time_s=capture_time,
                goal_seeds=((request.start_q, candidate, *request.goal_seeds)
                    if request.selection_policy == 'first_safe' else (candidate, *request.goal_seeds)))
            seeds = current.goal_seeds
            if np.linalg.norm(shift) <= request.intercept_position_tolerance_m:
                break
        return current

    def _high_revision_work_ratio(self, request, result, cancelled):
        """Estimate changed descent sample work before full recertification.

        Uses the same endpoint IK, curve generator and original validation
        interval formula as the fixed-high forecast. This predicts cost only;
        the actual revised high endpoint/curve still receives every check.
        """
        if result.continuation_trajectory is None or request.continuation_target_poses is None:
            return 1.
        pose_index = min(range(len(request.target_poses)),
            key=lambda i: np.linalg.norm(request.target_poses[i]-result.target_pose))
        low = request.continuation_target_poses[pose_index]
        self._diagnostic_context = {'phase': 'high_delivery_work_forecast'}
        estimate = self._solve_request_ik(request, low[:3], low[3:],
            qpos_seed=result.continuation_path[-1], cancelled=cancelled)
        self._record_ik(estimate, low, result.continuation_path[-1])
        if not estimate.success or cancelled():
            return 1.
        try:
            curve = self._continuation_curve(request, request.goal_seeds[0],
                                              estimate.joint_positions)
        except ValueError:
            return 1.
        original = self._curve_validation_intervals(request, result.continuation_trajectory)
        proposed = self._curve_validation_intervals(request, curve)
        return max(1., proposed/original)

    def _fixed_descent_delivery_request(self, request, result, certification_s, started, cancelled):
        """Forecast the cost of a longer low route before its full validation.

        Moving the low endpoint changes joint travel and the number of curve
        samples. A measured old calculation time alone underestimates that
        work. Warm endpoint IK and the existing curve generator estimate both
        duration and sample workload; _plan still certifies the entire result.
        """
        curve = result.continuation_trajectory
        if curve is None:
            return request
        intervals = self._curve_validation_intervals(request, curve)
        overhead = max(0., request.intercept_descent_duration_s-request.timed_continuation_duration_s)
        current = request
        seed = result.continuation_path[-1]
        computation = certification_s
        duration = curve.duration
        forecasts = []
        # Preserve the observed forecast error when workload changes. Using
        # only the old per-sample average repeats a just-observed under-run.
        overrun = (max(0., certification_s-request.continuation_compute_forecast_s)
                   if request.continuation_compute_forecast_s > 0. else 0.)
        for _ in range(request.max_goal_candidates):
            if cancelled():
                break
            delivery = request.intercept_request_time_s + time.perf_counter()-started + computation
            capture = max(float(current.planned_target_time_s),
                max(float(request.intercept_motion_start_time_s), delivery) + duration + overhead)
            shift = np.asarray(request.continuation_velocity_m_s) * (capture-request.planned_target_time_s)
            current = replace(request,
                continuation_target_poses=tuple(np.r_[pose[:3]+shift, pose[3:]]
                    for pose in request.continuation_target_poses),
                continuation_minimum_tcp_height=request.continuation_minimum_tcp_height+float(shift[2]),
                planned_target_time_s=capture, intercept_descent_duration_s=duration+overhead)
            low = current.continuation_target_poses[0]
            self._diagnostic_context = {'phase': 'fixed_descent_delivery_forecast'}
            ik = self._solve_request_ik(current, low[:3], low[3:], qpos_seed=seed, cancelled=cancelled)
            self._record_ik(ik, low, seed)
            if not ik.success:
                break  # Ordinary complete planner owns branch feasibility.
            seed = np.asarray(ik.joint_positions)
            try:
                estimate = self._continuation_curve(current, current.start_q, seed)
            except ValueError:
                break
            next_intervals = self._curve_validation_intervals(current, estimate)
            computation = certification_s * max(1., next_intervals/intervals) + overrun
            duration = max(curve.duration, estimate.duration)
            forecasts.append({'capture_s': capture, 'duration_s': duration,
                'validation_intervals': next_intervals, 'compute_forecast_s': computation,
                'observed_compute_overrun_s': overrun})
            needed = max(float(request.intercept_motion_start_time_s),
                request.intercept_request_time_s+time.perf_counter()-started+computation) + duration + overhead
            if capture >= needed:
                break
        self._request_diagnostics['intercept_delivery']['fixed_high_forecasts'] = forecasts
        return current

    def _continuous_delivery_request(self, request, result, certification_s, started, cancelled):
        """Forecast ONE new static-start appointment before full recertification.

        Reuse the measured full-job cost and curve validation workload, as in
        fixed-high delivery forecasting. Cheap IK/curve probes are forecasts
        only; no moved target receives a certificate without another _plan.
        """
        from .continuous_intercept_trajectory import ContinuousInterceptTrajectory, InterceptTimingError

        curve = result.continuation_trajectory
        duration = float(curve.duration)
        available = float(request.planned_target_time_s-request.continuation_start_time_s)
        dispatch_allowance = max(0., request.intercept_descent_duration_s-available)
        prior_forecast = float(request.continuation_compute_forecast_s)
        # Zero means no measured prior job. The initial configured fallback
        # must not turn the first long observation into a second full margin.
        overrun = max(0., certification_s-prior_forecast) if prior_forecast > 0. else 0.
        report = self._request_diagnostics['continuous_intercept_delivery']
        original_samples = max(1, int(report['first_curve_samples']))
        computation = certification_s+overrun
        seed = np.asarray(result.path[-1])
        current = request
        velocity = np.asarray(request.continuation_velocity_m_s)
        start_z = float(self.ik.forward_pose(request.start_q)[2])
        original_floor = min(start_z, *(float(pose[2]) for pose in request.target_poses))
        forecasts = []
        for index in range(request.max_goal_candidates):
            if cancelled():
                break
            future_start = max(float(request.continuation_start_time_s),
                request.intercept_request_time_s+time.perf_counter()-started
                +computation+dispatch_allowance)
            capture = future_start+duration
            shift = velocity*(capture-request.planned_target_time_s)
            targets = tuple(np.r_[pose[:3]+shift, pose[3:]] for pose in request.target_poses)
            floor = min(start_z, *(float(pose[2]) for pose in targets))
            current = replace(request, target_poses=targets,
                minimum_tcp_height=request.minimum_tcp_height+floor-original_floor,
                continuation_start_time_s=future_start, planned_target_time_s=capture,
                timed_continuation_duration_s=duration,
                intercept_descent_duration_s=duration+dispatch_allowance,
                goal_seeds=(seed, *request.goal_seeds),
                route_waypoint_hints=tuple(result.path[1:-1]))
            target = np.r_[result.target_pose[:3]+shift, result.target_pose[3:]]
            self._diagnostic_context = {'phase': 'continuous_delivery_forecast', 'iteration': index}
            ik = self._solve_request_ik(current, target[:3], target[3:],
                                      qpos_seed=seed, cancelled=cancelled)
            self._record_ik(ik, target, seed)
            forecast = {'start_s': future_start, 'capture_s': capture,
                'duration_s': duration, 'compute_forecast_s': computation,
                'observed_compute_overrun_s': overrun,
                'dispatch_allowance_s': dispatch_allowance, 'ik_success': bool(ik.success)}
            forecasts.append(forecast)
            if not ik.success:
                break
            seed = np.asarray(ik.joint_positions)
            final_velocity = np.linalg.lstsq(self.ik.compute_jacobian(seed),
                np.r_[velocity, np.zeros(3)], rcond=None)[0]
            try:
                estimate = ContinuousInterceptTrajectory((*result.path[:-1], seed),
                    request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                    duration=duration, initial_velocity=request.continuation_start_velocity,
                    initial_acceleration=request.continuation_start_acceleration,
                    final_velocity=final_velocity, jerk_limits=request.trajectory_jerk_limits)
            except InterceptTimingError as error:
                duration = max(duration, error.suggested_duration_s)
                forecast['next_duration_s'] = duration
                continue
            except ValueError as error:
                forecast['curve_forecast_error'] = str(error)
                break
            intervals = self._curve_validation_intervals(current, estimate)
            sample_count = len(np.unique(np.r_[np.linspace(0., duration, intervals+1),
                                              np.cumsum(estimate.edge_durations)]))
            computation = certification_s*max(1., sample_count/original_samples)+overrun
            forecast.update(validation_samples=sample_count,
                next_compute_forecast_s=computation)
            current = replace(current, goal_seeds=(seed, *request.goal_seeds))
            if future_start >= (request.intercept_request_time_s+time.perf_counter()-started
                                +computation+dispatch_allowance):
                break
        report.update(forecasts=forecasts, original_curve_samples=original_samples,
            observed_compute_overrun_s=overrun, dispatch_allowance_s=dispatch_allowance)
        return current

    @planning_stage('whole_request')
    def plan(
        self, request: PlanningRequest, cancelled: CancelCheck,
    ) -> PlanningResult:
        self._request_valid_states = set()
        self._validation_cache_request = None
        self._certified_continuations = {}
        try:
            return self._plan_request(request, cancelled)
        finally:
            self._request_valid_states = None
            self._validation_cache_request = None
            self._certified_continuations = {}
            self._full_response_candidate_cache = None

    def _plan_request(
        self, request: PlanningRequest, cancelled: CancelCheck,
    ) -> PlanningResult:
        started = time.perf_counter()
        self._request_started = started
        if request.selection_policy == 'first_safe':
            seeds = (request.start_q, *(q for q in request.goal_seeds
                if not np.array_equal(q, request.start_q)))
            request = replace(request, goal_seeds=seeds)
        thread_started = time.thread_time()
        self._segment_validators = {}
        evaluator_before = (dict(self.state_evaluator.metrics)
                            if self.state_evaluator is not None else None)
        self._request_diagnostics = {"schema_version": 1, "rejection_counts": {},
            "rejection_samples": [], "ik_attempt_count": 0, "ik_attempts": [],
            "selection_policy": request.selection_policy, "ik_seeds_tried": 0,
            "first_safe_route_wall_s": None, "extra_comparison_wall_s": 0.,
            "route_source": None}
        self._diagnostic_context = {"phase": "intercept_forecast"}
        self._last_rejection = None
        if hasattr(self, "data"):
            self.data.qpos[:] = request.full_qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        effective = request if request.continuous_intercept else self._intercept_request(request, cancelled)
        self._diagnostic_context = {"phase": "goal_search"}
        certification_started = time.perf_counter()
        result = (self._plan_full_pick_delivery(effective, cancelled, started)
                  if request.planning_mode == 'full_preplanned' else self._plan(effective, cancelled))
        if request.planning_mode == 'full_preplanned':
            effective = result.effective_request or effective
        certification_s = time.perf_counter() - certification_started
        static_continuous = (request.planning_mode == 'legacy' and request.continuous_intercept
            and request.intercept_account_for_planning_time
            and request.intercept_request_time_s is not None
            and not np.any(np.asarray(request.continuation_start_velocity) != 0.)
            and not np.any(np.asarray(request.continuation_start_acceleration) != 0.))
        if static_continuous and result.success and not cancelled():
            first_report = copy.deepcopy(self._request_diagnostics['continuous_intercept'])
            first_delivery = request.intercept_request_time_s+time.perf_counter()-started
            duration = result.continuation_trajectory.duration
            allowance = max(0., request.intercept_descent_duration_s-duration)
            delivery = self._request_diagnostics['continuous_intercept_delivery'] = {
                'mode': 'static_start_direct_intercept', 'first_certification_s': certification_s,
                'first_delivery_policy_time_s': first_delivery,
                'first_start_s': request.continuation_start_time_s,
                'first_capture_s': request.planned_target_time_s,
                'first_curve_samples': first_report['curve_samples'],
                'first_result': first_report, 'recertified': False}
            if first_delivery+allowance > request.continuation_start_time_s+1e-9:
                revision = self._continuous_delivery_request(request, result,
                    certification_s, started, cancelled)
                if not cancelled():
                    self._request_diagnostics.pop('continuous_intercept', None)
                    self._diagnostic_context = {'phase': 'continuous_delivery_recertification'}
                    revised_result = self._plan(revision, cancelled)
                    delivery['revision_result'] = copy.deepcopy(
                        self._request_diagnostics.get('continuous_intercept', {}))
                    if revised_result.success and not cancelled():
                        effective, result = revision, revised_result
                        delivery['recertified'] = True
                        self._request_diagnostics['continuous_intercept']['appointment_shifted'] = True
                    else:
                        # Retain the original exact certificate; policy will
                        # still reject its missed start. Never label a failed
                        # future endpoint as geometrically certified.
                        delivery['revision_failed'] = True
                        self._request_diagnostics['continuous_intercept'] = first_report
            final_delivery = request.intercept_request_time_s+time.perf_counter()-started
            delivery.update(final_start_s=effective.continuation_start_time_s,
                final_capture_s=effective.planned_target_time_s,
                final_delivery_policy_time_s=final_delivery,
                time_until_start_s=effective.continuation_start_time_s-final_delivery,
                delivered_before_start=final_delivery <= effective.continuation_start_time_s)
            if cancelled():
                result = replace(result, path=None, cancelled=True, error='planning cancelled')
        fixed_delivery = (request.fixed_high_continuation
            and request.stage == 'pregrasp'
            and request.timed_continuation_duration_s is not None
            and request.continuation_start_time_s is None
            and request.planned_target_time_s is not None
            and request.intercept_motion_start_time_s is not None)
        if (not request.continuous_intercept and request.intercept_account_for_planning_time
                and request.intercept_request_time_s is not None
                and (request.intercept_velocity_m_s is not None or fixed_delivery)
                and result.success and not cancelled()):
            # The cheap IK forecast precedes complete path/continuation
            # certification. Use that measured cost to reserve the delivery of
            # ONE revised, fully certified plan. Never shift a certified pose
            # after validation, and never loop indefinitely chasing the belt.
            delivered_at = request.intercept_request_time_s + time.perf_counter() - started
            descent_duration = request.intercept_descent_duration_s
            if result.continuation_trajectory is not None:
                descent_duration = max(descent_duration,
                    result.continuation_trajectory.duration
                    + max(0., request.intercept_descent_duration_s
                          - float(request.timed_continuation_duration_s or 0.)))
            needed = (max(float(request.intercept_motion_start_time_s), delivered_at)
                      + self._approach_time_forecast(effective, result.cost[0])
                      + descent_duration)
            self._request_diagnostics["intercept_delivery"] = {
                "mode": "fixed_high_descent" if fixed_delivery else "moving_high",
                "first_certification_s": certification_s,
                "first_delivery_policy_time_s": delivered_at,
                "first_capture_time_s": effective.planned_target_time_s,
                "required_capture_time_s": needed,
                "recertified": False,
            }
            previous_compute_s = certification_s
            refinements = []
            # One normal rebooking plus one catch-up correction for a measured
            # compute/branch overrun. Every changed appointment is recertified;
            # keep the completed route if a future proposal fails.
            attempts = 2 if self.rank_goal_routes and not fixed_delivery else 1
            for refinement in range(attempts):
                delivered_at = request.intercept_request_time_s+time.perf_counter()-started
                needed = (max(float(request.intercept_motion_start_time_s), delivered_at)
                    + self._approach_time_forecast(effective, result.cost[0]) + descent_duration)
                if needed <= float(effective.planned_target_time_s)+1e-9 or cancelled():
                    break
                compute_forecast = max(certification_s, previous_compute_s)
                if refinement:
                    compute_forecast += max(0., previous_compute_s-certification_s)
                motion_start = max(float(request.intercept_motion_start_time_s),
                                   delivered_at + compute_forecast)
                work_ratio = 1.
                if fixed_delivery:
                    # The high branch is stationary. Only the future low pose
                    # moves with the observed belt vector; certify it afresh.
                    # Explicit in-flight splices never enter this branch.
                    revision = self._fixed_descent_delivery_request(
                        effective, result, certification_s, started, cancelled)
                else:
                    revision = replace(effective,
                        intercept_descent_duration_s=descent_duration,
                        intercept_motion_start_time_s=motion_start,
                        # Reuse the selected elbow branch and actual corners
                        # when estimating the next appointment. A short IK
                        # chord is not the duration of a certified detour.
                        goal_seeds=(result.path[-1], *effective.goal_seeds),
                        route_waypoint_hints=tuple(result.path[1:-1]))
                    revision = self._intercept_request(revision, cancelled,
                        minimum_motion_s=result.cost[0])
                    if self.rank_goal_routes:
                        work_ratio = self._high_revision_work_ratio(revision, result, cancelled)
                        if work_ratio > 1.:
                            compute_forecast *= work_ratio
                            revision = replace(revision, intercept_motion_start_time_s=max(
                                float(request.intercept_motion_start_time_s),
                                request.intercept_request_time_s+time.perf_counter()-started
                                + compute_forecast))
                            revision = self._intercept_request(revision, cancelled,
                                minimum_motion_s=result.cost[0])
                self._diagnostic_context = {"phase": "delivery_recertification"}
                revision_started = time.perf_counter()
                revised_result = self._plan(revision, cancelled)
                previous_compute_s = time.perf_counter()-revision_started
                refinements.append({'compute_forecast_s': compute_forecast,
                    'workload_ratio': work_ratio,
                    'compute_actual_s': previous_compute_s, 'required_capture_time_s': needed,
                    'proposed_capture_time_s': revision.planned_target_time_s,
                    'success': revised_result.success})
                if revised_result.success and not cancelled():
                    effective, result = revision, revised_result
                    self._request_diagnostics["intercept_delivery"]["recertified"] = True
                else:
                    # Keep the original certified route as a possible prefix;
                    # existing receipt-time admission handles a missed booking.
                    self._request_diagnostics["intercept_delivery"]["revision_failed"] = True
                    break
            self._request_diagnostics['intercept_delivery']['refinements'] = refinements
            if cancelled():
                result = replace(result, path=None, cancelled=True, error="planning cancelled")
            if result.continuation_trajectory is not None:
                final_delivery = request.intercept_request_time_s+time.perf_counter()-started
                final_needed = (max(float(request.intercept_motion_start_time_s), final_delivery)
                    + self._approach_time_forecast(effective, result.cost[0]) + descent_duration)
                self._request_diagnostics['intercept_delivery'].update(
                    final_capture_time_s=effective.planned_target_time_s,
                    final_descent_start_s=(effective.planned_target_time_s
                                           - result.continuation_trajectory.duration),
                    final_delivery_policy_time_s=final_delivery,
                    final_required_capture_time_s=final_needed,
                    feedback_connector_forecast_s=(self._approach_time_forecast(effective, 0.)),
                    delivered_with_approach_eta=(final_needed <= effective.planned_target_time_s))
        if request.selection_policy == 'first_safe' and request.planning_mode == 'legacy':
            delivery = self._request_diagnostics.get('intercept_delivery', {})
            if result.success and delivery.get('delivered_with_approach_eta') is False:
                result = replace(result, path=None, path_fully_validated=False,
                    continuation_fully_validated=False, error='certified route missed moving contact time')
        if result.success:
            self._request_diagnostics['first_safe_route_wall_s'] = (
                self._request_diagnostics['first_safe_route_wall_s'] or time.perf_counter()-started)
        else:
            self._request_diagnostics['first_safe_route_wall_s'] = None
            self._request_diagnostics['route_source'] = None
        diagnostics = copy.deepcopy({**self._request_diagnostics,
                'shared_state_evaluation': ({
                    'geometry_facts_cache_may_persist_between_requests': True,
                    **{key: value-evaluator_before[key]
                       for key, value in self.state_evaluator.metrics.items()}}
                    if evaluator_before is not None else {'enabled': False}),
                'adaptive_segment_validation': {
                    'enabled': self.adaptive_segment_validation,
                    'scopes': [adapter.diagnostics() for adapter in self._segment_validators.values()
                               if adapter is not None]}})
        thread_cpu_s = time.thread_time()-thread_started
        planning_time_s = time.perf_counter()-started
        # Two thread-clock reads per whole request, never per pose. This
        # difference also includes scheduling, GIL and blocking time; it is
        # not a measurement of hardware waiting or isolated CPU contention.
        diagnostics['request_compute_timing'] = {
            'wall_s': planning_time_s,
            'thread_cpu_s': thread_cpu_s,
            'wall_minus_thread_cpu_s': planning_time_s-thread_cpu_s,
            'scope': 'whole_request_including_forecast_and_delivery_recertification',
            'thread_cpu_scope': 'planning_worker_thread_only_excludes_other_threads',
            'difference_meaning': 'includes_scheduling_GIL_and_blocking_not_isolated_hardware_wait_or_CPU_contention',
        }
        return replace(result,
            # Static first-safe jobs only prepend an IK seed. Their original
            # targets, scene and timing remain the submitted contract; do not
            # label that seed ordering as a pregrasp forecast revision.
            effective_request=effective if (effective is not request or
                (request.selection_policy == 'first_safe' and request.stage == 'pregrasp')) else None,
            diagnostics=diagnostics, planning_time_s=planning_time_s)

    def _candidate_backend_route(self, request, cancelled, started):
        """Treat accelerated proposals as hints, never as route certificates."""
        backend = self._candidate_path_backend
        if backend is None:
            return None
        report = {'selected': False, 'proposed': 0, 'certified': 0}
        self._request_diagnostics['candidate_path_backend'] = report
        for pose_index, target_pose in enumerate(request.target_poses):
            if cancelled():
                return None
            continuations = (None if request.continuation_target_poses is None else
                             (request.continuation_target_poses[pose_index],))
            proposal_request = replace(request, target_poses=(target_pose,),
                                       continuation_target_poses=continuations)
            try:
                paths = backend.propose(proposal_request, cancelled)
            except Exception as error:
                # An optional accelerator cannot turn a normal CPU task into
                # a stop. Genuine geometry and cancellation remain unchanged.
                report['fallback_reason'] = str(error)
                paths = ()
            evidence = getattr(backend, 'diagnostics', {})
            if callable(evidence):
                evidence = evidence()
            report['backend'] = self._diagnostic_value(evidence)
            for proposal_index, path in enumerate(paths):
                if cancelled():
                    return None
                report['proposed'] += 1
                try:
                    path = np.asarray(path, dtype=float)
                    if (path.ndim != 2 or path.shape[1] != 7 or not len(path)
                            or not np.isfinite(path).all()):
                        continue
                    self._diagnostic_context = {'phase': 'accelerated_candidate',
                        'pose_index': pose_index, 'seed_index': proposal_index,
                        'check': 'goal', 'planned_target_time_s': request.planned_target_time_s}
                    goal = self._solve_goal_seed(request, target_pose, path[-1], cancelled)
                    if goal is None or not self._continuation_is_valid(
                            request, pose_index, target_pose, goal, cancelled):
                        continue
                    hints = tuple(q.copy() for q in path[:-1]
                                  if not np.array_equal(q, request.start_q))
                    planner = self._planner(request, seed=request.planner_seed,
                                            cancelled=cancelled)
                    # A GPU curve can contain many optimizer knots. If its
                    # endpoint already has a valid direct connection, keep
                    # that shorter route rather than stop at every GPU knot.
                    direct = planner.direct_path(request.start_q, goal)
                    if direct is not None:
                        corners = (None if direct is None else
                                   self._certified_corner_path(planner, direct, cancelled))
                        if corners is None:
                            route = None
                        else:
                            cost, durations = self._path_profile(request, corners,
                                path_is_certified_corner_path=True, cancelled=cancelled)
                            route = (corners, cost, durations)
                        route_source = 'gpu_goal_with_certified_direct_connection'
                    elif hints:
                        hinted_request = replace(request, route_waypoint_hints=hints)
                        planner = self._planner(hinted_request, seed=request.planner_seed,
                                                cancelled=cancelled)
                        route = self._hinted_path(hinted_request, planner, goal, cancelled)
                        route_source = 'gpu_motiongen_waypoints'
                    else:
                        route = None
                    if route is None:
                        continue
                    corner_path, cost, durations = route
                    if cancelled():
                        return None
                    report.update(selected=True, certified=report['certified']+1,
                                  selected_pose_index=pose_index, route_source=route_source)
                    return self._success_result(request, target_pose, corner_path,
                                                cost, durations, started)
                except _PlanningCancelled:
                    return None
            if cancelled():
                return None
        report.setdefault('fallback_reason', 'no_accelerated_candidate_passed_original_checks')
        return None

    def _warm_roadmap_route(self, request, candidate, cancelled):
        """Try preheated first-object topology, then retain original RRT.

        This is called only after all cheap direct branches fail. Its cached
        static edges are suggestions: original current request state, scene,
        target, dense/adaptive edge, compression and timing checks all remain.
        """
        backend = self._warm_roadmap_backend
        if backend is None or cancelled():
            return None
        report = self._request_diagnostics.setdefault('warm_roadmap', {
            'queried': 0, 'proposed': 0, 'certified': 0, 'selected': False,
            'source': 'startup_static_graph_without_previous_workpiece_trajectory',
        })
        began = time.perf_counter()
        try:
            from .warm_roadmap import runtime_geometry_signature
            signature = backend.signature
            jaw = tuple(getattr(backend, '_gripper_qpos_indices', ()))
            if jaw:
                signature = replace(signature,
                                    gripper_snapshot=tuple(request.full_qpos[list(jaw)]))
            runtime_key = runtime_geometry_signature(signature,
                gripper_qpos=signature.gripper_snapshot,
                scene_key=request.scene_epoch, object_key=str(request.target_id),
                request_constraints_key={'generation': request.generation,
                    'request_id': request.request_id, 'stage': request.stage,
                    'minimum_height': request.minimum_tcp_height,
                    'minimum_margin': request.minimum_joint_margin})
            report['queried'] += 1
            routes = backend.propose(request.start_q, candidate, signature=signature,
                                     cancelled=cancelled, runtime_geometry_key=runtime_key)
            report['backend_diagnostics'] = dict(backend.diagnostics)
            for path in routes:
                if cancelled():
                    return None
                path = np.asarray(path, dtype=float)
                if (path.ndim != 2 or path.shape[1] != 7 or len(path) < 2
                        or not np.isfinite(path).all()
                        or not np.array_equal(path[0], request.start_q)
                        or not np.array_equal(path[-1], candidate)):
                    continue
                hints = tuple(q.copy() for q in path[1:-1])
                if not hints:
                    continue
                report['proposed'] += 1
                hinted_request = replace(request, route_waypoint_hints=hints)
                # The validator closes over the original request: only hint
                # metadata differs, so already-computed geometric facts remain
                # valid and the frozen moving collision snapshot is unchanged.
                planner = self._planner(request, seed=request.planner_seed, cancelled=cancelled)
                route = self._hinted_path(hinted_request, planner, candidate, cancelled,
                                          shorten=True)
                if route is not None and not cancelled():
                    report.update(selected=True, certified=report['certified'] + 1)
                    return route
            report.setdefault('fallback_reason', 'no_hint_passed_current_request_validation')
            return None
        except _PlanningCancelled:
            return None
        except Exception as error:
            # An optional preprocessed graph cannot disable the established
            # direct/RRT path or add a new task rejection condition.
            report['fallback_reason'] = f'{type(error).__name__}: {error}'
            return None
        finally:
            report['query_and_certification_wall_s'] = report.get('query_and_certification_wall_s', 0.) + time.perf_counter() - began

    def _plan_continuous_intercept(self, request, cancelled, started):
        """Certify current/future q/v/a directly to a moving contact pose.

        The ordinary planner supplies each complete IK/global route. A new
        polynomial, including its nonzero contact velocity, is then certified
        separately against this same immutable scene and target snapshot.
        """
        from .continuous_intercept_trajectory import (
            ContinuousInterceptTrajectory, InterceptTimingError)

        duration = float(request.planned_target_time_s-request.continuation_start_time_s)
        report = self._request_diagnostics.setdefault('continuous_intercept', {
            'start_time_s': request.continuation_start_time_s,
            'capture_time_s': request.planned_target_time_s,
            'available_duration_s': duration,
            'contact_velocity_m_s': list(request.continuation_velocity_m_s),
            'candidate_attempts': 0, 'seed_attempts': 0, 'duplicate_goals': 0,
            'candidates_per_pose': [], 'route_goal_recheck_count': 0,
            'route_goal_recheck_s': 0., 'certified': False,
            'appointment_shifted': False,
        })
        best_error = 'no safe continuous interception route'
        for pose_index, target_pose in enumerate(request.target_poses):
            candidates = []
            report['candidates_per_pose'].append(0)
            for seed_index, seed in enumerate(request.goal_seeds):
                if cancelled() or len(candidates) >= request.max_goal_candidates:
                    break
                report['seed_attempts'] += 1
                self._diagnostic_context = {'phase': 'continuous_intercept',
                    'pose_index': pose_index, 'seed_index': seed_index}
                candidate = self._solve_goal_seed(request, target_pose, seed, cancelled)
                if candidate is None:
                    continue
                if any(np.max(np.abs(candidate-item)) < 1e-4 for item in candidates):
                    report['duplicate_goals'] += 1
                    continue
                candidates.append(candidate)
                report['candidate_attempts'] += 1
                report['candidates_per_pose'][-1] += 1
                # Use all established IK, direct-edge, optional backend,
                # roadmap and RRT checks without invoking the high forecast.
                # The budget counts distinct valid goals, exactly as ordinary
                # _plan does; failed seeds must not hide a later valid branch.
                # The converged goal is rechecked by that existing route path,
                # with the extra IK cost recorded separately above.
                route_request = replace(request, continuous_intercept=False, target_is_contact=True,
                    target_poses=(target_pose,), goal_seeds=(candidate,),
                    continuation_start_time_s=None, timed_continuation_duration_s=None,
                    intercept_velocity_m_s=None, intercept_account_for_planning_time=False,
                    optimize_direct_path_eta=False)
                route = self._plan(route_request, cancelled)
                if not route.success or cancelled():
                    best_error = route.error or best_error
                    continue
                goal = np.asarray(route.path[-1])
                jacobian = self.ik.compute_jacobian(goal)
                twist = np.r_[request.continuation_velocity_m_s, np.zeros(3)]
                final_velocity = np.linalg.lstsq(jacobian, twist, rcond=None)[0]
                residual = float(np.linalg.norm(jacobian @ final_velocity-twist))
                if not np.isfinite(final_velocity).all() or residual > 1e-6+1e-4*np.linalg.norm(twist):
                    self._record_rejection('continuous_intercept_contact_velocity',
                        twist_residual=residual)
                    best_error = 'contact velocity is unavailable at the selected IK goal'
                    continue
                path = route.path if len(route.path) > 1 else (request.start_q, goal)
                try:
                    trajectory = ContinuousInterceptTrajectory(path,
                        request.trajectory_velocity_limits, request.trajectory_acceleration_limits,
                        duration=duration, initial_velocity=request.continuation_start_velocity,
                        initial_acceleration=request.continuation_start_acceleration,
                        final_velocity=final_velocity, jerk_limits=request.trajectory_jerk_limits)
                except InterceptTimingError as error:
                    report['suggested_duration_s'] = min(
                        report.get('suggested_duration_s', float('inf')),
                        error.suggested_duration_s)
                    self._record_rejection('continuous_intercept_deadline',
                        available_duration_s=duration, suggested_duration_s=error.suggested_duration_s)
                    best_error = str(error)
                    continue
                except ValueError as error:
                    self._record_rejection('continuous_intercept_dynamics', message=str(error))
                    best_error = str(error)
                    continue
                planner = self._planner(request, seed=request.planner_seed, cancelled=cancelled)
                checked = []
                previous = request.start_q
                intervals = self._curve_validation_intervals(request, trajectory)
                # Include exact corner clocks as well as derivative-bounded
                # uniform samples. Every chord retains the original full
                # state/scene/cone/singularity/edge certification.
                times = np.unique(np.r_[np.linspace(0., duration, intervals+1),
                                        np.cumsum(trajectory.edge_durations)])
                passed = True
                for timestamp in times:
                    self._diagnostic_context = {'phase': 'continuous_intercept',
                        'check': 'full_curve', 'trajectory_time_s': float(timestamp)}
                    q, velocity, _ = trajectory.sample_full(timestamp)
                    if (cancelled() or not self._state_valid(request, q, cancelled)
                            or not planner.edge_valid(previous, q)):
                        self._record_rejection('continuous_intercept_curve_rejected')
                        passed = False
                        break
                    limits = request.continuation_tcp_speed_limits
                    if limits is not None or request.tcp_speed_limits_enabled:
                        tcp_velocity = self.ik.compute_jacobian(q) @ velocity
                        linear, angular = (float(np.linalg.norm(tcp_velocity[:3])),
                                           float(np.linalg.norm(tcp_velocity[3:])))
                        if ((limits is not None and linear > limits[0]+1e-9)
                                or angular > (limits[1] if limits is not None
                                              else request.maximum_tcp_angular_speed)+1e-9):
                            self._record_rejection('continuous_intercept_tcp_speed',
                                linear_speed_m_s=linear, angular_speed_rad_s=angular)
                            passed = False
                            break
                    checked.append(q.copy())
                    previous = q
                if not passed or cancelled():
                    best_error = 'continuous interception curve did not pass complete checks'
                    continue
                report.update(certified=True, actual_duration_s=trajectory.duration,
                    terminal_joint_velocity=final_velocity.tolist(),
                    terminal_twist_residual=residual, curve_samples=len(checked),
                    route_waypoint_count=len(path), trajectory_backend=trajectory.backend)
                return replace(route, path=tuple(path),
                    cost=(duration, route.cost[1], len(path)),
                    edge_durations=trajectory.edge_durations,
                    continuation_fully_validated=True, continuation_path=tuple(checked),
                    continuation_trajectory=trajectory, continuation_target_pose=target_pose,
                    planning_time_s=time.perf_counter()-started)
            if cancelled():
                break
        return PlanningResult(target_identity=request.target_identity, request_id=request.request_id, generation=request.generation,
            stage=request.stage, target_id=request.target_id, scene_epoch=request.scene_epoch,
            target_pose=request.target_pose, start_q=request.start_q,
            cancelled=cancelled(), error=best_error, planning_time_s=time.perf_counter()-started)

    def _ranked_goal_route(self, request, cancelled, started, *, explicit_cancelled=None):
        """Rank IK endpoints cheaply, then reuse the complete route planner.

        Only the live high approach opts in. Near-identical IK endpoints are
        grouped for this preference search; a failed preference still falls
        back to the original complete search with every original seed.
        Extra RRT/selection time is paid from estimated motion-time savings.
        Expiry returns the best completed certificate, never a task stop.
        """
        report = {'candidate_count': 0, 'similar_ik_trials_skipped': 0,
                  'trials': [], 'selected': None}
        self._request_diagnostics.setdefault('goal_route_selection', []).append(report)
        begin = time.perf_counter()
        def continuation_valid(pose_index, target, q, stop):
            if stop():
                return False
            saved = self._certified_continuations.get((id(request), q.tobytes()))
            if saved is not None and np.array_equal(
                    saved[2], request.continuation_target_poses[pose_index]):
                return True
            return self._continuation_is_valid(request, pose_index, target, q, stop)
        # The intercept forecast already proposes a specific branch/time.
        # If its full direct route meets that appointment, there is no reason
        # to pay for more IK restarts. A failed direct edge or slower branch
        # falls through to the ranked detour search below.
        first_q = None
        if request.intercept_velocity_m_s is not None and request.goal_seeds:
            target = request.target_pose
            self._diagnostic_context = {'phase': 'rank_goal_routes', 'check': 'goal',
                'pose_index': 0, 'seed_index': 0,
                'planned_target_time_s': request.planned_target_time_s}
            first_q = self._solve_goal_seed(request, target, request.goal_seeds[0], cancelled)
            if first_q is not None:
                estimate, _ = self._path_profile(request, (request.start_q, first_q))
                window = (request.planned_target_time_s-request.intercept_motion_start_time_s
                          - request.intercept_descent_duration_s)
                if (self._approach_time_forecast(request, estimate[0]) <= window+1e-9
                        and continuation_valid(0, target, first_q, cancelled)):
                    planner = self._planner(request, seed=request.planner_seed, cancelled=cancelled)
                    path = planner.direct_path(request.start_q, first_q)
                    if path is not None:
                        corners = self._certified_corner_path(planner, path, cancelled)
                        if corners is not None:
                            cost, durations = self._path_profile(request, corners,
                                path_is_certified_corner_path=True)
                            if not cancelled():
                                selected = {'pose_index': 0, 'seed_index': 0,
                                    'route_source': 'forecast_direct', 'cost': cost}
                                report.update(candidate_count=1, selected=selected,
                                              wall_s=time.perf_counter()-begin)
                                return self._success_result(request, target, corners,
                                                            cost, durations, started)
        ranked = []
        for pose_index, target in enumerate(request.target_poses):
            group = []
            for seed_index, seed in enumerate(request.goal_seeds):
                if cancelled():
                    break
                self._diagnostic_context = {'phase': 'rank_goal_routes',
                    'check': 'goal', 'pose_index': pose_index, 'seed_index': seed_index,
                    'planned_target_time_s': request.planned_target_time_s}
                q = (first_q if pose_index == 0 and seed_index == 0 and first_q is not None
                     else self._solve_goal_seed(request, target, seed, cancelled))
                if q is None:
                    continue
                # Search redundancy, not a new pose/command tolerance. All
                # representatives still receive the original full checks;
                # untested variants remain available in the ordinary fallback.
                if any(np.max(np.abs(q-item[2])) <= .01 for item in group):
                    report['similar_ik_trials_skipped'] += 1
                    continue
                try:
                    estimate, _ = self._path_profile(request, (request.start_q, q),
                                                     cancelled=cancelled)
                except _PlanningCancelled:
                    break
                group.append((estimate, seed_index, q, pose_index, target))
                if len(group) >= request.max_goal_candidates:
                    break
            ranked.extend(group)
        ranked.sort(key=lambda item: item[0])
        report.update(candidate_count=len(ranked), cheap_ranking_s=time.perf_counter()-begin)
        best = None
        improvement_deadline = None
        first_completed_at = None
        for rank, (estimate, seed_index, q, pose_index, target) in enumerate(ranked):
            if cancelled():
                break
            if best is not None and (estimate[0] >= best[0][0]
                    or time.perf_counter() >= improvement_deadline):
                break
            def selection_cancelled():
                return cancelled() or (improvement_deadline is not None
                    and time.perf_counter() >= improvement_deadline)
            trial = {'pose_index': pose_index, 'seed_index': seed_index,
                     'direct_eta_estimate_s': estimate[0], 'route_source': None}
            report['trials'].append(trial)
            self._diagnostic_context = {'phase': 'rank_goal_routes', 'check': 'continuation',
                'pose_index': pose_index, 'seed_index': seed_index,
                'planned_target_time_s': request.planned_target_time_s}
            if not continuation_valid(pose_index, target, q, selection_cancelled):
                trial['continuation_valid'] = False
                continue
            trial['continuation_valid'] = True
            self._diagnostic_context['check'] = 'route'
            planner = self._planner(request, seed=request.planner_seed+pose_index*100_003+rank,
                                    cancelled=selection_cancelled)
            path = planner.direct_path(request.start_q, q)
            source = 'direct'
            option = None
            if path is None and not selection_cancelled():
                option = self._hinted_path(request, planner, q, selection_cancelled)
                source = 'previous_route' if option is not None else 'rrt'
                if option is None:
                    # Estimate the benefit over a remaining endpoint. This is
                    # a search heuristic, not a geometric or optimality proof.
                    reference = (best[0][0] if best is not None else
                                 max((item[0][0] for item in ranked[rank+1:]),
                                     default=estimate[0]))
                    allowance = max(0., reference-estimate[0])
                    trial['extra_search_budget_s'] = allowance
                    if allowance > 0.:
                        until = time.perf_counter()+allowance
                        if improvement_deadline is not None:
                            until = min(until, improvement_deadline)
                        def rrt_cancelled():
                            return cancelled() or time.perf_counter() >= until
                        planner = self._planner(request,
                            seed=request.planner_seed+pose_index*100_003+rank,
                            cancelled=rrt_cancelled)
                        path = planner.plan(request.start_q, q, check_direct=False)
                        if path is not None:
                            corners = self._certified_corner_path(planner, path, rrt_cancelled)
                            if corners is not None:
                                try:
                                    cost, durations = self._path_profile(request, corners,
                                        path_is_certified_corner_path=True, cancelled=rrt_cancelled)
                                    option = (corners, cost, durations)
                                except _PlanningCancelled:
                                    pass
                    path = None
            if path is not None:
                corners = self._certified_corner_path(planner, path, selection_cancelled)
                if corners is not None:
                    try:
                        cost, durations = self._path_profile(request, corners,
                            path_is_certified_corner_path=True, cancelled=selection_cancelled)
                        option = (corners, cost, durations)
                    except _PlanningCancelled:
                        pass
            if option is None:
                trial['route_source'] = 'no_completed_route'
                continue
            corners, cost, durations = option
            trial.update(route_source=source, cost=cost,
                         maximum_joint_change_rad=float(np.max(np.abs(q-request.start_q))))
            if first_completed_at is None:
                first_completed_at = time.perf_counter()
            if best is None or cost < best[0]:
                best = (cost, durations, corners, target, trial)
                if improvement_deadline is None:
                    next_eta = min((item[0][0] for item in ranked[rank+1:]), default=cost[0])
                    allowance = max(0., cost[0]-next_eta)
                    improvement_deadline = time.perf_counter()+allowance
                    report['extra_comparison_budget_s'] = allowance
        report['wall_s'] = time.perf_counter()-begin
        if first_completed_at is not None:
            self._request_diagnostics['extra_comparison_wall_s'] += time.perf_counter()-first_completed_at
        # An ordinary request budget may stop extra selection while keeping a
        # completed route. Explicit user/worker cancellation discards it.
        if best is None or (explicit_cancelled or cancelled)():
            return None
        cost, durations, corners, target, trial = best
        report['selected'] = trial.copy()
        return self._success_result(request, target, corners, cost, durations, started)

    @planning_stage('joint_route_search')
    def _plan(
        self, request: PlanningRequest, cancelled: CancelCheck
    ) -> PlanningResult:
        started = time.perf_counter()
        # IK owns a second private MjData from the validation scratch.  Restore
        # the immutable request snapshot here so gripper/open-jaw and all
        # non-arm coordinates match the route certificate; arm values are
        # overwritten by each supplied DLS seed.
        if hasattr(self, "data") and hasattr(self, "model"):
            self.data.qpos[:] = request.full_qpos
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        def should_cancel() -> bool:
            return bool(
                cancelled()
                or (
                    request.wall_time_budget_s is not None
                    and time.perf_counter() - started
                    >= request.wall_time_budget_s
                )
            )

        if request.continuous_intercept:
            return self._plan_continuous_intercept(request, should_cancel, started)

        if request.fixed_high_continuation:
            # The same state, branch-step and dense-edge checks as ordinary
            # PREGRASP continuation. No RRT and no speculative high movement.
            self.data.qpos[self.arm_qpos_adrs] = request.start_q
            mujoco.mj_forward(self.model, self.data)
            position = self.data.site_xpos[self.ee_site_id].copy()
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self.ee_site_id])
            high = request.target_pose
            pose_matches = (np.linalg.norm(position-high[:3]) <= self.ik.config.position_tolerance
                and 2.*math.acos(min(1., abs(float(quaternion @ high[3:]))))
                    <= self.ik.config.orientation_tolerance)
            if (pose_matches and self._state_valid(request, request.start_q, should_cancel)
                    and self._continuation_is_valid(request, 0, high, request.start_q, should_cancel)
                    and not should_cancel()):
                return self._success_result(request, high, (request.start_q,),
                    (0., 0., 1), (), started)
            return PlanningResult(target_identity=request.target_identity, request_id=request.request_id, generation=request.generation,
                stage=request.stage, target_id=request.target_id, scene_epoch=request.scene_epoch,
                target_pose=high, start_q=request.start_q, cancelled=cancelled(),
                error="fixed high descent is unavailable" if pose_matches else "fixed high branch pose mismatch",
                planning_time_s=time.perf_counter()-started)

        if request.cartesian_local_only:
            # Try the caller-ordered local endpoints without ever falling
            # back to free 7-DOF RRT.  The post-release pipeline puts the full
            # withdrawal first and may provide a shorter, contact-free
            # handoff endpoint second when the full pose has no branch inside
            # the actuator command envelope.  Every option starts from the
            # same immutable snapshot and receives the complete local path
            # certificate before it can be returned.
            for target_pose in request.target_poses:
                if should_cancel():
                    break
                local = self._local_cartesian_path(
                    request, target_pose, should_cancel
                )
                if local is None:
                    continue
                path, cost, durations = local
                return self._success_result(
                    request,
                    target_pose,
                    path,
                    cost,
                    durations,
                    started,
                )
            return PlanningResult(
                target_identity=request.target_identity, request_id=request.request_id,
                generation=request.generation,
                stage=request.stage,
                target_id=request.target_id,
                scene_epoch=request.scene_epoch,
                target_pose=request.target_pose,
                start_q=request.start_q,
                error=(
                    "cancelled"
                    if cancelled()
                    else "local Cartesian corridor is unavailable"
                ),
                cancelled=cancelled(),
                planning_time_s=time.perf_counter() - started,
            )

        accelerated = (None if request.selection_policy == 'first_safe'
            else self._candidate_backend_route(request, should_cancel, started))
        if accelerated is not None:
            return accelerated
        if (request.selection_policy != 'first_safe' and self.rank_goal_routes and request.stage == 'pregrasp'
                and request.continuation_target_poses is not None):
            ranked = self._ranked_goal_route(request, should_cancel, started,
                                             explicit_cancelled=cancelled)
            if ranked is not None:
                return ranked
        error = "global planner found no safe IK goal"
        for pose_index, target_pose in enumerate(request.target_poses):
            if should_cancel():
                break
            # The caller-prioritized seed is deliberately first, but the first
            # certified direct branch is not necessarily the quickest one to
            # execute.  Compare a request-bounded number of direct branches by
            # their trajectory ETA before selecting one.  Every compared route
            # still receives the continuation, dense-edge, compressed-chord,
            # and timing certificates entirely on this worker thread.  If no
            # compared seed has a direct edge, retain the complete multi-seed
            # RRT fallback below.
            candidates: list[np.ndarray] = []
            direct_options: list[
                tuple[
                    tuple[float, float, int],
                    tuple[float, ...],
                    np.ndarray,
                ]
            ] = []
            direct_candidate_checks = 0
            for seed_index, goal_seed in enumerate(request.goal_seeds):
                if should_cancel():
                    break
                context = {"phase": "goal_search", "pose_index": pose_index,
                    "seed_index": seed_index, "check": "goal",
                    "planned_target_time_s": request.planned_target_time_s}
                self._diagnostic_context = context
                candidate = self._solve_goal_seed(
                    request,
                    target_pose,
                    goal_seed,
                    should_cancel,
                )
                if candidate is None:
                    continue
                if any(
                    np.max(np.abs(candidate - item)) < 1e-4
                    for item in candidates
                ):
                    continue
                if not self._continuation_is_valid(
                    request,
                    pose_index,
                    target_pose,
                    candidate,
                    should_cancel,
                ):
                    continue
                self._diagnostic_context = {**context, "check": "route"}
                candidates.append(candidate)
                direct_candidate_checks += 1
                planner = self._planner(
                    request,
                    seed=(
                        request.planner_seed
                        + pose_index * request.max_goal_candidates
                        + seed_index
                    ),
                    cancelled=should_cancel,
                )
                hinted = None if request.route_hints_after_direct else self._hinted_path(
                    request,
                    planner,
                    candidate,
                    should_cancel,
                )
                if hinted is not None:
                    # A wall-time deadline may expire immediately after a
                    # complete certificate; as with direct options, retain
                    # that finished work.  Explicit cancellation invalidates
                    # the entire speculative request and must publish nothing.
                    if cancelled():
                        break
                    corner_path, cost, edge_durations = hinted
                    if not self._first_safe_appointment_valid(request, candidate, cost):
                        continue
                    self._diagnostic_context['check'] = 'previous_route'
                    return self._success_result(
                        request,
                        target_pose,
                        corner_path,
                        cost,
                        edge_durations,
                        started,
                    )
                if should_cancel():
                    break
                self._diagnostic_context['check'] = 'direct'
                direct = planner.direct_path(request.start_q, candidate)
                if direct is not None:
                    corner_path = self._certified_corner_path(
                        planner, direct, should_cancel
                    )
                    if corner_path is None:
                        break
                    try:
                        cost, edge_durations = self._path_profile(
                            request,
                            corner_path,
                            path_is_certified_corner_path=True,
                            cancelled=should_cancel,
                        )
                    except _PlanningCancelled:
                        break
                    if not self._first_safe_appointment_valid(request, candidate, cost):
                        continue
                    direct_options.append(
                        (cost, edge_durations, corner_path)
                    )
                    if not request.optimize_direct_path_eta:
                        if cancelled():
                            break
                        return self._success_result(
                            request,
                            target_pose,
                            corner_path,
                            cost,
                            edge_durations,
                            started,
                        )
                if request.route_hints_after_direct and direct is None:
                    hinted = self._hinted_path(request, planner, candidate, should_cancel)
                    if hinted is not None and not cancelled():
                        corner_path, cost, edge_durations = hinted
                        if not self._first_safe_appointment_valid(request, candidate, cost):
                            continue
                        self._diagnostic_context['check'] = 'previous_route'
                        return self._success_result(request, target_pose, corner_path,
                            cost, edge_durations, started)
                if direct_candidate_checks >= request.max_goal_candidates:
                    break
            # Explicit cancellation invalidates even an already compared
            # speculative option.  A wall-time budget, on the other hand, may
            # stop further comparison while retaining the best fully certified
            # route completed before the deadline.
            if cancelled():
                break
            best_direct = (
                min(
                    direct_options,
                    key=lambda option: option[0],
                )
                if direct_options
                else None
            )
            if (
                best_direct is not None
                and request.rrt_restarts_after_direct == 0
                and request.full_rrt_candidates_after_direct == 0
            ):
                if cancelled():
                    break
                cost, edge_durations, corner_path = best_direct
                return self._success_result(
                    request,
                    target_pose,
                    corner_path,
                    cost,
                    edge_durations,
                    started,
                )
            if should_cancel():
                if cancelled():
                    break
                if best_direct is not None:
                    cost, edge_durations, corner_path = best_direct
                    return self._success_result(
                        request,
                        target_pose,
                        corner_path,
                        cost,
                        edge_durations,
                        started,
                    )
                break
            if not candidates:
                continue
            if best_direct is None:
                for candidate in candidates:
                    self._diagnostic_context = {'phase': 'goal_search', 'pose_index': pose_index,
                                                'check': 'warm_roadmap_route'}
                    warm_route = self._warm_roadmap_route(request, candidate, should_cancel)
                    if warm_route is not None and not cancelled():
                        corner_path, cost, edge_durations = warm_route
                        if not self._first_safe_appointment_valid(request, candidate, cost):
                            continue
                        return self._success_result(request, target_pose, corner_path,
                                                    cost, edge_durations, started)
                    if should_cancel():
                        break
                if should_cancel():
                    break
            error = "constrained RRT-Connect could not find a safe path"
            # Interleave short deterministic searches instead of letting one
            # unlucky candidate exhaust all 2400 iterations.  A valid branch
            # often appears in another candidate/restart within 0.1--0.4 s;
            # round-robin slices bound latency without reducing the aggregate
            # search budget or any safety validation density.
            persistent_after_direct = bool(
                best_direct is not None
                and request.full_rrt_candidates_after_direct > 0
                and (
                    request.full_rrt_minimum_direct_eta_s is None
                    or best_direct[0][0]
                    > request.full_rrt_minimum_direct_eta_s + 1e-12
                )
            )
            if persistent_after_direct:
                # PREGRASP supplies calibration/branch-token seeds in a
                # deliberate continuation-first order.  Re-sorting those
                # certified endpoints by distance from the remote retreat
                # anchor made the full-budget trees explore an unrelated
                # redundancy branch first and spend the entire intercept
                # window there.  Preserve discovery/caller priority for this
                # strictly bounded speculative search.
                # One uninterrupted deterministic tree per candidate.  The
                # candidate bound, cooperative cancellation callback, and
                # optional wall-clock deadline keep this speculative search
                # finite without weakening any path certificate.
                slice_iterations = request.planner_config.max_iterations
                restart_count = 1
                rrt_candidates = candidates[
                    : request.full_rrt_candidates_after_direct
                ]
            else:
                span = np.maximum(
                    request.upper_limits - request.lower_limits, 1e-6
                )
                candidates.sort(
                    key=lambda item: (
                        float(
                            np.linalg.norm(
                                (item - request.start_q) / span
                            )
                        ),
                        -float(
                            np.min(
                                np.minimum(
                                    item - request.lower_limits,
                                    request.upper_limits - item,
                                )
                            )
                        ),
                    )
                )
                slice_iterations = min(
                    request.rrt_iteration_slice,
                    request.planner_config.max_iterations,
                )
                restart_count = min(
                    request.rrt_restart_count,
                    max(
                        1,
                        int(
                            math.ceil(
                                request.planner_config.max_iterations
                                / slice_iterations
                            )
                        ),
                    ),
                )
                if best_direct is not None:
                    restart_count = min(
                        restart_count,
                        request.rrt_restarts_after_direct,
                    )
                rrt_candidates = candidates[: request.max_goal_candidates]
            slice_config = SafePathConfig(
                normalized_step_size=(
                    request.planner_config.normalized_step_size
                ),
                normalized_edge_resolution=(
                    request.planner_config.normalized_edge_resolution
                ),
                execution_joint_step=(
                    request.planner_config.execution_joint_step
                ),
                max_iterations=slice_iterations,
                goal_sample_probability=(
                    request.planner_config.goal_sample_probability
                ),
                shortcut_attempts=(
                    request.planner_config.shortcut_attempts
                ),
            )
            for restart_index in range(restart_count):
                for candidate_index, candidate in enumerate(
                    rrt_candidates
                ):
                    if should_cancel():
                        break
                    self._diagnostic_context = {"phase": "goal_search",
                        "pose_index": pose_index, "check": "rrt_route",
                        "candidate_index": candidate_index, "restart_index": restart_index}
                    planner = ConstrainedRRTConnect(
                        request.lower_limits + request.minimum_joint_margin,
                        request.upper_limits - request.minimum_joint_margin,
                        lambda state: self._state_valid(
                            request, state, should_cancel
                        ),
                        config=slice_config,
                        seed=(
                            request.planner_seed
                            + pose_index * 100_003
                            + candidate_index
                            + restart_index * 101
                        ),
                        cancelled=should_cancel,
                    )
                    # Every goal candidate's direct chord was already checked
                    # exactly once above.  Repeating that dense validation on
                    # each restart consumed a large and timing-variable share
                    # of the planning budget without changing feasibility.
                    path = planner.plan(
                        request.start_q,
                        candidate,
                        check_direct=False,
                    )
                    if path is None:
                        continue
                    corner_path = self._certified_corner_path(
                        planner, path, should_cancel
                    )
                    if corner_path is None:
                        break
                    try:
                        cost, edge_durations = self._path_profile(
                            request,
                            corner_path,
                            path_is_certified_corner_path=True,
                            cancelled=should_cancel,
                        )
                    except _PlanningCancelled:
                        break
                    if cancelled():
                        break
                    if not self._first_safe_appointment_valid(request, candidate, cost):
                        continue
                    rrt_option = (cost, edge_durations, corner_path)
                    if best_direct is not None:
                        if rrt_option[0] >= best_direct[0]:
                            # A certified but slower tree is not a reason to
                            # stop the bounded improvement search.  In the
                            # persistent full-RRT mode, later configured goal
                            # candidates often represent deliberately ordered
                            # redundancy branches; one of them may still beat
                            # the direct fallback.  Keep that already complete
                            # direct certificate and try the remaining bounded
                            # candidates/restarts.  If none improves it, the
                            # common fallback below publishes the direct route.
                            if persistent_after_direct:
                                continue
                            selected = best_direct
                        else:
                            selected = rrt_option
                    else:
                        selected = rrt_option
                    cost, edge_durations, corner_path = selected
                    return self._success_result(
                        request,
                        target_pose,
                        corner_path,
                        cost,
                        edge_durations,
                        started,
                    )
            if cancelled():
                break
            if best_direct is not None:
                cost, edge_durations, corner_path = best_direct
                return self._success_result(
                    request,
                    target_pose,
                    corner_path,
                    cost,
                    edge_durations,
                    started,
                )
        return PlanningResult(
            target_identity=request.target_identity, request_id=request.request_id,
            generation=request.generation,
            stage=request.stage,
            target_id=request.target_id,
            scene_epoch=request.scene_epoch,
            target_pose=request.target_pose,
            start_q=request.start_q,
            error=(
                "planning budget exhausted"
                if (
                    not cancelled()
                    and request.wall_time_budget_s is not None
                    and time.perf_counter() - started
                    >= request.wall_time_budget_s
                )
                else ("cancelled" if cancelled() else error)
            ),
            cancelled=cancelled(),
            planning_time_s=time.perf_counter() - started,
        )


class AsyncPlanningWorker:
    """Single-capacity, cooperative-cancellation background planner."""

    def __init__(
        self,
        engine: PlanningEngine,
        *,
        name: str = "nero-global-planner",
        join_timeout_s: float = 2.0,
    ) -> None:
        self._engine = engine
        self._condition = threading.Condition()
        self._pending: PlanningRequest | None = None
        self._active: PlanningRequest | None = None
        self._active_cancel: threading.Event | None = None
        self._results: deque[PlanningResult] = deque()
        self._closed = False
        self._candidate_wall_budget_s = None
        self._progress = PlanningProgress()
        self._engine._planning_progress = self._progress
        self._join_timeout_s = max(0.0, float(join_timeout_s))
        self._thread = threading.Thread(
            target=self._run,
            name=str(name),
            daemon=True,
        )
        self._thread.start()

    def set_candidate_wall_budget(self, seconds: float | None) -> None:
        """Cooperative whole-request deadline; no effect on trajectory timestamps."""
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0.):
            raise ValueError('candidate wall budget must be finite and positive')
        with self._condition:
            if self._active is not None or self._pending is not None:
                raise RuntimeError('configure candidate budget before submitting work')
            self._candidate_wall_budget_s = None if seconds is None else float(seconds)

    def progress_snapshot(self, after_sequence=0):
        return self._progress.snapshot(after_sequence)

    def submit(self, request: PlanningRequest) -> bool:
        with self._condition:
            if self._closed or self._pending is not None:
                return False
            self._pending = request
            self._condition.notify()
            return True

    def poll(self) -> PlanningResult | None:
        with self._condition:
            return self._results.popleft() if self._results else None

    def cancel_pending(self) -> None:
        with self._condition:
            self._pending = None
            self._results.clear()
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._condition.notify_all()

    def close(self) -> bool:
        with self._condition:
            if self._closed:
                return not self._thread.is_alive()
            self._closed = True
            self._pending = None
            self._results.clear()
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._join_timeout_s)
        return not self._thread.is_alive()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._active is not None or self._pending is not None

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                cancel_event = threading.Event()
                self._active = request
                self._active_cancel = cancel_event
            assert request is not None
            planning_started = time.monotonic()
            budget = self._candidate_wall_budget_s
            deadline = None if budget is None else planning_started+budget
            self._progress.begin(request, budget)
            budget_hit = False
            def cancelled():
                nonlocal budget_hit
                if cancel_event.is_set():
                    return True
                if deadline is not None and time.monotonic() >= deadline:
                    budget_hit = True
                    self._progress.budget_exceeded()
                    return True
                return False
            try:
                result = self._engine.plan(request, cancelled)
            except Exception as error:  # fail closed; surfaced to policy diagnostics
                result = PlanningResult(
                    target_identity=request.target_identity, request_id=request.request_id,
                    generation=request.generation,
                    stage=request.stage,
                    target_id=request.target_id,
                    scene_epoch=request.scene_epoch,
                    target_pose=request.target_pose,
                    start_q=request.start_q,
                    error=f"planning worker error: {error}",
                    cancelled=cancelled(),
                    diagnostics=copy.deepcopy(getattr(self._engine, "_request_diagnostics", None)),
                )
            if isinstance(result, PlanningResult):
                # A result completed beyond the deadline cannot become a late
                # success, even if the final native operation did not poll.
                cancelled()
                if budget_hit or cancel_event.is_set():
                    result = replace(result, target_pose=request.target_pose,
                        start_q=request.start_q, effective_request=None,
                        path=None, cost=None, edge_durations=None,
                        continuation_path=None, continuation_trajectory=None,
                        continuation_target_pose=None, full_pick_plan=None,
                        path_fully_validated=False, continuation_fully_validated=False,
                        cancelled=True, error=('planning_candidate_budget_exceeded'
                                               if budget_hit else 'planning_cancelled'))
                self._progress.finish('budget_exceeded' if budget_hit else
                    ('cancelled' if cancel_event.is_set() else ('succeeded' if result.success else 'failed')))
                result = replace(result, diagnostics={**(result.diagnostics or {}),
                    'worker_timing':{'planning_start_monotonic':planning_started,
                                     'planning_finish_monotonic':time.monotonic()},
                    'live_progress':self.progress_snapshot()})
            with self._condition:
                self._active = None
                self._active_cancel = None
                if not self._closed:
                    self._results.append(result)
                    self._condition.notify_all()


__all__ = [
    "AsyncPlanningWorker",
    "MuJoCoGlobalPlanningEngine",
    "PlanningEngine",
    "PlanningRequest",
    "PlanningResult",
]

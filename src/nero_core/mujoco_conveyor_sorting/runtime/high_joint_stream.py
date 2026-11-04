"""Stream one already certified PREGRASP edge; the shared policy still decides.

Only numerical interpolation/retiming and the controller's discrete envelope
run on the sender. Full measured-to-command geometry remains on the policy
thread at its explicitly recorded rate. This is not a 100 Hz geometry checker.
No local IK trajectory or gripper permission is exported to this module.
"""
from copy import copy, deepcopy
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

from .feedback_clock import mapped_stamp

import numpy as np

from .joint_stream_sender import JointStreamSender


def edge_key(context):
    policy = context.policy
    trajectory = getattr(policy, '_safe_edge_trajectory', None)
    if trajectory is None:
        return None
    return (id(policy._safe_joint_path), id(trajectory), policy._safe_edge_goal_index,
            context.scene_epoch, policy._target_id, policy.stage.value,
            tuple(getattr(policy, name, None) for name in (
                '_planning_generation', '_safe_path_generation', '_safe_path_scene_epoch',
                '_safe_path_target_id', '_safe_path_stage', '_safe_path_worker_validated',
                '_safe_path_minimum_tcp_height', '_safe_path_minimum_joint_margin',
                '_safe_path_minimum_downward_alignment')),
            policy.config.planner_edge_resolution,
            policy.config.singularity_hard_stop_condition,
            policy._safe_path_tracking_error_limit(),
            policy._safe_path_settle_tolerance(policy._safe_edge_goal_index),
            float(context.robot_controller.gripper.target_open_fraction),
            context.robot_controller.command_joint_lower_limits.tobytes(),
            context.robot_controller.command_joint_upper_limits.tobytes(),
            context.robot_controller.max_joint_velocity.tobytes(),
            context.robot_controller.max_joint_acceleration.tobytes())


def can_stream_high_edge(context, command):
    """Only the shared worker's active unloaded edge, after exact handoff."""
    policy = context.policy
    return bool(
        getattr(context.settings, 'high_joint_stream_enabled', False)
        and callable(getattr(context.robot_controller, 'adopt_joint_dispatch_history', None))
        and policy.stage.value == 'pregrasp'
        and getattr(policy.config, 'dynamic_intercept_tracking', False)
        and getattr(policy, '_safe_path_worker_validated', False)
        and policy._safe_path_generation == policy._planning_generation
        and policy._safe_path_target_id == policy._target_id
        and policy._safe_path_scene_epoch == context.scene_epoch
        and policy._safe_path_stage == policy.stage
        and getattr(policy, '_safe_edge_trajectory', None) is not None
        and policy._safe_edge_started_at is not None
        and not policy._safe_edge_handoff_pending
        and not policy._safe_path_from_carry_prefetch
        and policy._safe_edge_goal_index != policy._safe_path_feedback_connector_goal_index
        and policy._safe_edge_passthrough_eligible
        and command.time_parameterized
        and np.max(np.abs(policy._safe_edge_trajectory.q1
                          - command.joint_target)) > 1e-8
    )


class HighJointStream:
    """Exclusive sender plus a detached numeric copy of the original retimer."""

    def __init__(self, context, device, *, origin, last_send, clock, sleep,
                 pause_requested):
        self.key = edge_key(context)
        self.origin = origin
        self.initial_dispatch_time = last_send
        self.period = 1. / context.settings.control_hz
        self.policy_period = max(self.period, 1. / 50.)
        self.width = context.robot_controller.gripper.target_open_fraction * context.settings.open_width_m
        self.trajectory = deepcopy(context.policy._safe_edge_trajectory)
        self.edge_started = context.policy._safe_edge_started_at
        self.arrival_tolerance = context.policy._safe_path_settle_tolerance(
            context.policy._safe_edge_goal_index)
        self.limit = context.policy._safe_path_tracking_error_limit()
        self.reason = None
        # The clone only calls pure numeric helpers; it owns no planning worker,
        # FSM transitions, geometry/model or mutable environment.
        self.numeric = copy(context.policy)
        self.numeric.config = deepcopy(context.policy.config)
        velocity, acceleration = context.policy._trajectory_limits()
        self.numeric.env = SimpleNamespace(max_joint_velocity=velocity.copy(),
                                          max_joint_acceleration=acceleration.copy(),
                                          joint_response_model=getattr(context, 'joint_response_model', None))
        self.controller = copy(context.robot_controller)
        self.controller.data = copy(context.robot_controller.data)
        self.controller.gripper = None
        self.controller.adopt_joint_dispatch_history(
            context.robot_controller.commanded_joint_positions,
            context.robot_controller.commanded_joint_velocities)
        self._command = None
        self._previous_q = None
        self._previous_v = None
        self._dt = None
        self._feedback_lock = Lock()
        self._feedback_stamp = getattr(context, '_last_joint_stamp', None)
        self._feedback_position = deepcopy(getattr(context, '_last_joint_position', None))
        self._feedback_velocity = np.asarray(getattr(context, '_joint_velocity', np.zeros(7))).copy()
        self.offline = bool(getattr(device, 'is_offline_replay', False))
        self.device = device
        self.sender = JointStreamSender(device, self._step, self._after_send,
            control_hz=context.settings.control_hz, clock=clock, sleep=sleep,
            pause_requested=pause_requested, initial_dispatch_time=last_send)

    def start(self):
        self.sender.start()
        return self

    def _step(self, state, sample_time, dt):
        measured = np.asarray(state['joints'], dtype=float)
        stamp = mapped_stamp(state)
        if stamp is None and 'feedback_time' in state:
            self.reason = {'reason': 'feedback_clock_mapping_invalid'}
            return None
        with self._feedback_lock:
            if stamp is not None and (self._feedback_stamp is None or stamp > self._feedback_stamp):
                if self._feedback_stamp is not None and self._feedback_position is not None:
                    self._feedback_velocity = (measured-self._feedback_position)/(stamp-self._feedback_stamp)
                self._feedback_position = measured.copy()
                self._feedback_stamp = stamp
            measured_velocity = self._feedback_velocity.copy()
        # The original arrival predicate belongs to the feedback cadence,
        # not the slower geometry/strategy cadence. Hand back immediately and
        # wake the policy thread, using real packet stamps for velocity.
        if (sample_time-self.origin-self.edge_started >= self.trajectory.duration
                and np.max(np.abs(measured-self.trajectory.q1))
                    <= self.arrival_tolerance
                and np.max(np.abs(measured_velocity))
                    <= self.numeric.config.planner_waypoint_velocity_tolerance):
            self.reason = {'reason': 'original_feedback_position_and_velocity_ready',
                           'joint_stamp': stamp,
                           'joint_error_rad': (measured-self.trajectory.q1).tolist(),
                           'joint_arrival_tolerance_rad': self.arrival_tolerance,
                           'feedback_velocity_rad_s': measured_velocity.tolist()}
            return None
        self._previous_q = self.controller.commanded_joint_positions
        self._previous_v = self.controller.commanded_joint_velocities
        self._dt = dt
        raw = self.numeric._sample_safe_edge_reference(
            self.trajectory, sample_time-self.origin-self.edge_started)
        observation = {'control_dt': dt, 'joint_target': self._previous_q,
                       'joint_target_velocity': self._previous_v}
        q = self.numeric._retimed_certified_edge_command(
            measured, raw, self.trajectory.q0, self.trajectory.q1,
            self.limit, observation, self.trajectory.duration)
        if q is None:
            self.reason = dict(self.numeric._safe_edge_retime_diagnostics)
            return None  # The original synchronous path handles this case.
        self._command = self.controller.set_time_parameterized_joint_target(q, dt=dt)
        if not self._command.time_parameterized:
            self.reason = {'reason': 'controller_declined_stream_sample'}
            return None  # Never send the controller's off-edge fallback.
        return self._command.joint_target, {
            'trajectory_retiming': dict(self.numeric._safe_edge_retime_diagnostics),
            'shared_time_parameterized': True,
            'joint_response': dict(self.numeric._safe_edge_reference_diagnostics),
        }

    def _after_send(self, dispatched, actual_dt, q, metadata):
        history = self.controller.acknowledge_dispatch(self._command,
            actual_dt=actual_dt, planned_dt=self._dt,
            previous_target=self._previous_q, previous_velocity=self._previous_v)
        if self.offline:
            # Ideal feedback only; the production device has no advance().
            self.device.advance(actual_dt)
        return {'joint_velocity': self.controller.commanded_joint_velocities.tolist(),
                'command_history': history}

    def snapshot(self):
        return self._with_feedback_velocity(self.sender.snapshot())

    def _with_feedback_velocity(self, snapshot):
        state = snapshot.last_feedback
        if state is not None:
            with self._feedback_lock:
                if mapped_stamp(state) is not None and mapped_stamp(state) == self._feedback_stamp:
                    state = {**state, 'stream_joint_velocity_rad_s': self._feedback_velocity.tolist(),
                             'stream_velocity_clock': 'real_joint_packet_stamp'}
            snapshot = replace(snapshot, last_feedback=state)
        return snapshot

    def wait(self, timeout):
        self.sender.wait_stopped(timeout)

    def stop(self):
        self.sender.request_stop()
        self.sender.join()
        return self._with_feedback_velocity(self.sender.snapshot())

    def sync_history(self, context, snapshot):
        command = snapshot.last_command
        if command is not None and 'joint_velocity' in command['metadata']:
            context.robot_controller.adopt_joint_dispatch_history(
                command['joint_target'], command['metadata']['joint_velocity'])
            context.policy._safe_edge_dynamic_retime_active = True
            context.policy._safe_edge_retime_diagnostics = dict(
                command['metadata']['trajectory_retiming'])

    def continues(self, context, observation, action):
        if self.key != edge_key(context):
            return False
        requested_width = .5 * (float(action.get('gripper', 1.)) + 1.) * context.settings.open_width_m
        if abs(requested_width-self.width) > 1e-9:
            return False
        # Use the same endpoint/corner rule as the policy and feedback sender.
        return not (
            np.max(np.abs(np.asarray(observation['joint_pos'])-self.trajectory.q1))
                <= self.arrival_tolerance
            and np.max(np.abs(observation['joint_vel']))
                <= context.policy.config.planner_waypoint_velocity_tolerance
        )

    def geometry_target(self, state, policy_time, horizon):
        """Review ahead along the same certified edge, with the original lead."""
        raw = self.numeric._sample_safe_edge_reference(
            self.trajectory, policy_time+horizon-self.edge_started, record=False)
        q = self.numeric._bounded_certified_edge_command(
            np.asarray(state['joints']), raw, self.trajectory.q0,
            self.trajectory.q1, self.limit)
        # A missing interval is handled by the original retimer on its next
        # sample; still review the most recently sent point on the slow side.
        return q

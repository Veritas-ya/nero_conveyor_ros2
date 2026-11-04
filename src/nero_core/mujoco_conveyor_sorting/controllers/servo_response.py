"""Bounded scalar response lag inferred from robot command/encoder FK only.

This is a feed-forward observer, not a safety certificate or a measurement of
object motion. Vertical command error contributes zero for a planar belt;
future visual intercept poses must never be used as the command input.
"""
from __future__ import annotations

import numpy as np


class ScalarServoResponseObserver:
    def __init__(self, *, initial_horizon_s=.16, maximum_horizon_s=.35, filter_time_s=.2):
        values = np.asarray([initial_horizon_s, maximum_horizon_s, filter_time_s], dtype=float)
        if (not np.isfinite(values).all() or maximum_horizon_s <= 0. or filter_time_s <= 0.
                or not 0. <= initial_horizon_s <= maximum_horizon_s):
            raise ValueError("require finite positive response bounds and a bounded initial horizon")
        self.initial = float(initial_horizon_s)
        self.maximum = float(maximum_horizon_s)
        self.filter_time = float(filter_time_s)
        self.reset()

    def reset(self):
        self.key = None
        self.last_time = None
        self.filtered = self.initial

    def update(self, *, timestamp, commanded_position, feedback_position,
               track_velocity, acquisition_key, fallback_horizon_s):
        fallback = float(fallback_horizon_s)
        evidence = {"estimator": "applied_joint_command_fk_minus_encoder_fk_projected_on_visual_velocity",
                    "object_ground_truth_used": False, "filter_time_s": self.filter_time,
                    "maximum_horizon_s": self.maximum, "fallback_horizon_s": fallback,
                    "raw_horizon_s": None, "bounded_raw_horizon_s": None,
                    "filtered_horizon_s": self.filtered}
        try:
            now = float(timestamp)
            command = np.asarray(commanded_position, dtype=float)
            measured = np.asarray(feedback_position, dtype=float)
            velocity = np.asarray(track_velocity, dtype=float)
            valid = (command.shape == measured.shape == velocity.shape == (3,)
                     and np.isfinite(np.r_[now, command, measured, velocity]).all())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            evidence.update(status="robot_fk_telemetry_unavailable", effective_horizon_s=fallback)
            return fallback, evidence
        if self.key != acquisition_key or (self.last_time is not None and now < self.last_time):
            self.reset()
            self.key = acquisition_key
        speed_squared = float(velocity @ velocity)
        if speed_squared < 1e-6:
            evidence.update(status="stopped_target_fallback", effective_horizon_s=fallback)
            return fallback, evidence
        error = command - measured
        raw = float(error @ velocity / speed_squared)
        evidence.update(commanded_position_world_m=command.copy(), feedback_position_world_m=measured.copy(),
                        track_velocity_world_m_s=velocity.copy(), command_tracking_error_world_m=error,
                        raw_horizon_s=raw)
        if raw < 0.:
            # An upstream catch-up command is not a positive response delay.
            # Do not wind the observer forward on a changing route direction.
            evidence.update(status="negative_projection_fallback", effective_horizon_s=fallback)
            self.last_time = now
            return fallback, evidence
        dt = 0. if self.last_time is None else max(0., now - self.last_time)
        if dt > self.maximum:
            self.filtered = self.initial
            dt = 0.
            evidence["telemetry_gap_reset"] = True
        bounded = float(np.clip(raw, 0., self.maximum))
        alpha = min(1., dt / self.filter_time)
        self.filtered += alpha * (bounded - self.filtered)
        self.last_time = now
        evidence.update(status="valid_robot_response_projection", sample_dt_s=dt, filter_alpha=alpha,
                        bounded_raw_horizon_s=bounded, filtered_horizon_s=self.filtered,
                        effective_horizon_s=self.filtered)
        return self.filtered, evidence


class VisualFollowingLeadAdapter(ScalarServoResponseObserver):
    """Bounded integral correction of actual visual following bias.

    ``residual_seconds`` is spatial tracking error divided by speed, NOT
    packet delay or actuator latency. A gain/backtracking-limited servo can
    have residual error even when its applied-command latency is small.
    """
    def update(self, *, timestamp, contact_position, feedback_position,
               track_velocity, acquisition_key, fallback_horizon_s, near_capture_height):
        fallback = float(fallback_horizon_s)
        evidence = {"estimator": "bounded_visual_following_bias_integral",
                    "object_ground_truth_used": False, "is_packet_latency_measurement": False,
                    "filter_time_s": self.filter_time, "maximum_horizon_s": self.maximum,
                    "fallback_horizon_s": fallback, "residual_seconds": None,
                    "near_capture_height": bool(near_capture_height)}
        try:
            now = float(timestamp)
            target = np.asarray(contact_position, dtype=float)
            measured = np.asarray(feedback_position, dtype=float)
            velocity = np.asarray(track_velocity, dtype=float)
            valid = (target.shape == measured.shape == velocity.shape == (3,)
                     and np.isfinite(np.r_[now, target, measured, velocity]).all())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            evidence.update(status="visual_or_encoder_telemetry_unavailable", effective_horizon_s=fallback)
            return fallback, evidence
        if self.key != acquisition_key or (self.last_time is not None and now < self.last_time):
            self.reset()
            self.key = acquisition_key
        speed_squared = float(velocity @ velocity)
        if speed_squared < 1e-6:
            self.last_time = now
            evidence.update(status="stopped_target_fallback", effective_horizon_s=fallback)
            return fallback, evidence
        dt = 0. if self.last_time is None else max(0., now - self.last_time)
        if dt > self.maximum:
            self.filtered = self.initial
            dt = 0.
            evidence["telemetry_gap_reset"] = True
        self.last_time = now
        if not near_capture_height:
            evidence.update(status="vertical_approach_no_integration", effective_horizon_s=fallback,
                            filtered_horizon_s=self.filtered)
            return fallback, evidence
        residual = float((target - measured) @ velocity / speed_squared)
        previous = self.filtered
        requested = previous + residual
        bounded = float(np.clip(requested, 0., self.maximum))
        alpha = min(1., dt / self.filter_time)
        self.filtered += alpha * (bounded - self.filtered)
        evidence.update(status="valid_near_height_visual_following_bias", residual_seconds=residual,
                        residual_world_m=target - measured, previous_horizon_s=previous,
                        raw_effective_horizon_s=requested, bounded_effective_horizon_s=bounded,
                        sample_dt_s=dt, filter_alpha=alpha, filtered_horizon_s=self.filtered,
                        effective_horizon_s=self.filtered)
        return self.filtered, evidence

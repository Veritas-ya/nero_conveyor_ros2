"""Offline identification of the externally visible position servo response.

The model is ``tau * q_dot + q = q_command(t - delay)`` with unit DC gain.
Its delay includes every delay between the recorded host dispatch and CAN
receive timestamps; it is not an encoder-internal latency measurement.  The
helpers only propose references.  Existing motion and geometry validation must
still be applied by the caller, including for the optional per-axis inverse.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from bisect import bisect_right
from math import comb, isfinite
from typing import Any, Callable

import numpy as np


def _clipped_scalar(position: float, velocity: float) -> tuple[float, float]:
    if position <= 0.:
        return 0., 0.
    if position >= 1.:
        return 1., 0.
    return position, velocity


class _PolynomialEdgeResponse:
    """Precompiled delay intervals; evaluation allocates no NumPy arrays."""
    backend = "compiled_quintic_response"
    applied = True

    def __init__(self, duration: float, delays: tuple[float, ...],
                 taus: tuple[float, ...], weights: tuple[float, ...]):
        self.duration = duration
        self.inverse_duration = 1. / duration
        self.breakpoints = tuple(sorted(set([-delay for delay in delays] +
                                            [duration - delay for delay in delays])))
        origins, coefficients = [], []
        # Coefficients in local normalized time z=(t-origin)/duration prevent
        # cancellation from large absolute times. Each active interval has
        # length at most one normalized trajectory duration.
        for index in range(len(self.breakpoints) + 1):
            origin = self.breakpoints[0] if index == 0 else self.breakpoints[index - 1]
            probe = (self.breakpoints[0] - duration if index == 0 else
                     self.breakpoints[-1] + duration if index == len(self.breakpoints) else
                     (origin + self.breakpoints[index]) * .5)
            coefficients_here = [0.] * 6
            for delay, tau, weight in zip(delays, taus, weights):
                u_probe = (probe + delay) / duration
                if u_probe <= 0.:
                    continue
                if u_probe >= 1.:
                    coefficients_here[0] += weight
                    continue
                offset = (origin + delay) / duration
                ratio = tau / duration
                # s(u)+tau/T*s'(u), s=10u^3-15u^4+6u^5.
                powers = (0., 0., 30. * ratio, 10. - 60. * ratio,
                          -15. + 30. * ratio, 6.)
                for degree, value in enumerate(powers):
                    for output_degree in range(degree + 1):
                        coefficients_here[output_degree] += (
                            weight * value * comb(degree, output_degree) *
                            offset ** (degree - output_degree))
            origins.append(origin)
            coefficients.append(tuple(coefficients_here))
        self.origins = tuple(origins)
        self.coefficients = tuple(coefficients)

    def scalar_state(self, time: float) -> tuple[float, float]:
        timestamp = float(time)
        if not isfinite(timestamp):
            raise ValueError("sample time must be finite")
        index = bisect_right(self.breakpoints, timestamp)
        if index == 0:
            return 0., 0.
        if index == len(self.breakpoints):
            return 1., 0.
        z = (timestamp - self.origins[index]) * self.inverse_duration
        c0, c1, c2, c3, c4, c5 = self.coefficients[index]
        position = c0 + z * (c1 + z * (c2 + z * (c3 + z * (c4 + z * c5))))
        velocity = (c1 + z * (2. * c2 + z * (3. * c3 + z * (4. * c4 + z * 5. * c5)))) * self.inverse_duration
        return _clipped_scalar(position, velocity)


class _NativeScalarEdgeResponse:
    """Native Ruckig scalar calls grouped by identical fitted delays."""
    backend = "compiled_native_scalar_response"
    applied = True

    def __init__(self, trajectory: Any, delays: tuple[float, ...],
                 taus: tuple[float, ...], weights: tuple[float, ...]):
        grouped: dict[float, tuple[float, float]] = {}
        for delay, tau, weight in zip(delays, taus, weights):
            first, second = grouped.get(delay, (0., 0.))
            grouped[delay] = first + weight, second + weight * tau
        self.groups = tuple((delay, first, second) for delay, (first, second) in grouped.items())
        self.duration = float(trajectory.duration)
        self.native = trajectory._trajectory

    def scalar_state(self, time: float) -> tuple[float, float]:
        timestamp = float(time)
        if not isfinite(timestamp):
            raise ValueError("sample time must be finite")
        position = velocity = 0.
        for delay, first, second in self.groups:
            sample_time = min(self.duration, max(0., timestamp + delay))
            p, v, a = self.native.at_time(sample_time)
            position += first * p[0] + second * v[0]
            velocity += first * v[0] + second * a[0]
        return _clipped_scalar(position, velocity)


class _GenericEdgeResponse:
    """Compatibility with other trajectory types, with no execution stop."""
    backend = "original_response_helper"

    def __init__(self, model: Any, trajectory: Any, delta: np.ndarray, applied: bool):
        self.model, self.trajectory, self.delta = model, trajectory, delta.copy()
        self.length_squared = float(delta @ delta)
        self.applied = applied

    def scalar_state(self, time: float) -> tuple[float, float]:
        if self.length_squared == 0.:
            return 1., 0.
        position, details = self.model.preview_on_edge(self.trajectory, time)
        progress = float(self.delta @ (position - self.trajectory.q0) / self.length_squared)
        velocity = details.get("candidate_velocity_rad_s")
        if velocity is None:
            velocity = self.trajectory.sample_full(time)[1]
        speed = float(self.delta @ np.asarray(velocity) / self.length_squared)
        return _clipped_scalar(progress, speed)


@dataclass(frozen=True)
class JointAxisResponse:
    delay_s: float
    time_constant_s: float
    train_rmse_rad: float
    holdout_rmse_rad: float
    unmodeled_holdout_rmse_rad: float
    train_samples: int
    holdout_samples: int
    command_span_rad: float
    local_parameter_condition: float | None
    fit_at_search_boundary: bool
    status: str = "identified"


@dataclass(frozen=True)
class JointResponseModel:
    axes: tuple[JointAxisResponse | None, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def compile_edge_reference(self, trajectory: Any) -> Any:
        """Compile the same projected inverse once for numerical retiming.

        ``scalar_state(t)`` returns progress and progress speed for the exact
        candidate generated by :meth:`preview_on_edge`. Quintic references use
        piecewise polynomials; scalar Ruckig references call its native scalar
        time law once per distinct delay. Unknown trajectory types retain the
        original helper. Nothing in this method sends or certifies motion.
        """
        from .joint_trajectory import QuinticJointTrajectory

        start, end = np.asarray(trajectory.q0, float), np.asarray(trajectory.q1, float)
        if start.shape != end.shape or start.shape != (len(self.axes),):
            raise ValueError("edge vectors must match response model joint count")
        delta = end - start
        moving = np.flatnonzero(delta != 0.)
        eligible = (self.metadata.get("runtime_eligible", False) is True and len(moving) > 0
                    and all(self.axes[i] is not None and self.axes[i].status == "identified"
                            for i in moving))
        if not eligible or float(trajectory.duration) == 0.:
            return _GenericEdgeResponse(self, trajectory, delta, False)
        weights_array = delta[moving] ** 2 / float(delta @ delta)
        weights = tuple(float(value) for value in weights_array)
        delays = tuple(float(self.axes[i].delay_s) for i in moving)
        taus = tuple(float(self.axes[i].time_constant_s) for i in moving)
        if isinstance(trajectory, QuinticJointTrajectory):
            return _PolynomialEdgeResponse(float(trajectory.duration), delays, taus, weights)
        if (getattr(trajectory, "backend", None) == "ruckig_local"
                and getattr(trajectory, "_scalar", False)
                and callable(getattr(getattr(trajectory, "_trajectory", None), "at_time", None))):
            return _NativeScalarEdgeResponse(trajectory, delays, taus, weights)
        return _GenericEdgeResponse(self, trajectory, delta, True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JointResponseModel":
        if not isinstance(value, dict):
            raise ValueError('joint response model must be a JSON object')
        if value.get("schema_version", 1) != 1:
            raise ValueError("unsupported joint response model schema")
        axes = tuple(None if axis is None else JointAxisResponse(**axis)
                     for axis in value["axes"])
        for axis in axes:
            if axis is not None and (not np.isfinite([axis.delay_s, axis.time_constant_s]).all()
                                     or axis.delay_s < 0. or axis.time_constant_s <= 0.):
                raise ValueError("response delay must be nonnegative and time constant positive")
        return cls(axes, dict(value.get("metadata", {})))

    def edge_preview_time_s(self, q0: Any, q1: Any) -> tuple[float, dict[str, Any]]:
        """Project delay + first-order lag onto one certified straight edge.

        A single time advance keeps the commanded point on that edge.  It is
        the least-squares projection weighted by squared joint displacement,
        not a claim that distinct joint dynamics can be inverted exactly.  A
        missing/ineligible model leaves the original phase unchanged.
        """
        start, end = np.asarray(q0, float), np.asarray(q1, float)
        if start.shape != end.shape or start.shape != (len(self.axes),):
            raise ValueError("edge vectors must match response model joint count")
        if not np.isfinite(np.r_[start, end]).all():
            raise ValueError("edge joints must be finite")
        delta = end - start
        weights = delta * delta
        moving = np.flatnonzero(weights > 0.)
        diagnostic = {"method": "common_edge_phase_projection",
                      "preserves_original_straight_edge": True,
                      "requires_existing_motion_limits": True,
                      "is_physical_validation": False,
                      "preview_time_s": 0.}
        if self.metadata.get("runtime_eligible", False) is not True:
            return 0., dict(diagnostic, status="diagnostic_model_original_phase")
        if not len(moving):
            return 0., dict(diagnostic, status="stationary_edge")
        if any(self.axes[i] is None or self.axes[i].status != "identified" for i in moving):
            return 0., dict(diagnostic, status="unidentified_moving_joint_original_phase")
        lags = np.asarray([self.axes[i].delay_s + self.axes[i].time_constant_s
                           for i in moving])
        advance = float(weights[moving] @ lags / weights[moving].sum())
        residual = (lags - advance) * delta[moving]
        return advance, dict(diagnostic, status="candidate_requires_motion_validation",
                             preview_time_s=advance,
                             moving_joint_indices=moving.tolist(),
                             per_joint_low_frequency_lag_s=lags.tolist(),
                             lag_projection_residual_rad_seconds=residual.tolist())

    def preview_on_edge(self, trajectory: Any, elapsed: float
                        ) -> tuple[np.ndarray, dict[str, Any]]:
        """Project the per-axis inverse onto the already certified joint chord.

        This uses actual fitted delay/tau and future derivatives, rather than
        merely adding a constant time shift.  Projection preserves the chord;
        the caller's normal velocity, acceleration, braking/lead and endpoint
        validation still controls how much of this candidate can be sent.
        """
        start, end = np.asarray(trajectory.q0, float), np.asarray(trajectory.q1, float)
        nominal = np.asarray(trajectory.sample_full(float(elapsed))[0], float)
        if start.shape != end.shape or start.shape != (len(self.axes),):
            raise ValueError("edge vectors must match response model joint count")
        delta = end - start
        length_squared = float(delta @ delta)
        diagnostic = {"method": "per_joint_inverse_projected_to_certified_chord",
                      "preserves_original_straight_edge": True,
                      "requires_existing_motion_limits": True,
                      "is_physical_validation": False,
                      "nominal_q_rad": nominal.tolist()}
        moving = np.flatnonzero(delta != 0.)
        if self.metadata.get("runtime_eligible", False) is not True:
            return nominal.copy(), dict(diagnostic, status="diagnostic_model_original_command",
                                        applied=False)
        if not len(moving):
            return nominal.copy(), dict(diagnostic, status="stationary_edge", applied=False)
        if any(self.axes[i] is None or self.axes[i].status != "identified" for i in moving):
            return nominal.copy(), dict(diagnostic,
                status="unidentified_moving_joint_original_command", applied=False)
        inverse, inverse_info = self.preview_command(trajectory.sample_full, float(elapsed))
        raw_progress = float(delta @ (inverse - start) / length_squared)
        progress = float(np.clip(raw_progress, 0., 1.))
        nominal_progress = float(delta @ (nominal - start) / length_squared)
        candidate = start + progress * delta
        # Derivative of the proposed inverse, before the original execution
        # limits: d/dt[q_d(t+d)+tau*qdot_d(t+d)]. This lets the retimer retain
        # this reference's own deceleration law rather than a quintic cap.
        inverse_velocity = np.zeros_like(start)
        for i in moving:
            axis = self.axes[i]
            _, velocity, acceleration = trajectory.sample_full(float(elapsed) + axis.delay_s)
            inverse_velocity[i] = velocity[i] + axis.time_constant_s * acceleration[i]
        scalar_velocity = float(delta @ inverse_velocity / length_squared)
        candidate_velocity = delta * scalar_velocity if 0. < raw_progress < 1. else np.zeros_like(start)
        _, lag_info = self.edge_preview_time_s(start, end)
        return candidate, dict(diagnostic, status="candidate_requires_motion_validation", applied=True,
            inverse_candidate_q_rad=inverse.tolist(), inverse_details=inverse_info,
            raw_progress=raw_progress, progress=progress, nominal_progress=nominal_progress,
            requested_progress_advance=progress - nominal_progress,
            endpoint_clipped=raw_progress != progress,
            projection_residual_rad=(inverse - candidate).tolist(),
            candidate_correction_rad=(candidate - nominal).tolist(),
            candidate_velocity_rad_s=candidate_velocity.tolist(),
            common_low_frequency_lag_s=lag_info["preview_time_s"])

    def preview_command(self, sample_full: Callable[[float], Any], phase_time_s: float
                        ) -> tuple[np.ndarray, dict[str, Any]]:
        """Per-axis inverse for offline experiments or full geometric recheck.

        For a known future reference the continuous first-order inverse is
        ``q_d(t+delay) + tau*qdot_d(t+delay)``.  Different per-joint delays can
        leave a certified path; this API explicitly does not certify it.
        """
        nominal = np.asarray(sample_full(float(phase_time_s))[0], float).copy()
        if nominal.shape != (len(self.axes),):
            raise ValueError("trajectory sample must match response model joint count")
        info = {"method": "per_joint_preview_inverse", "is_physical_validation": False,
                "requires_full_geometry_validation": True,
                "preserves_original_straight_edge": False,
                "nominal_q_rad": nominal.tolist(), "applied_joint_indices": []}
        if self.metadata.get("runtime_eligible", False) is not True:
            return nominal, dict(info, status="diagnostic_model_original_command")
        candidate = nominal.copy()
        for i, axis in enumerate(self.axes):
            if axis is None or axis.status != "identified":
                continue
            position, velocity, *_ = sample_full(float(phase_time_s) + axis.delay_s)
            candidate[i] = float(position[i]) + axis.time_constant_s * float(velocity[i])
            info["applied_joint_indices"].append(i)
        return candidate, dict(info, status="candidate_requires_motion_and_geometry_validation",
                               correction_rad=(candidate - nominal).tolist())


def predict_position_response(command_times_s: Any, commands_rad: Any,
                              feedback_times_s: Any, initial_position_rad: float,
                              delay_s: float, time_constant_s: float) -> np.ndarray:
    """Exact ZOH response at irregular observations, initialized at first sample.

    Initial state is the first observed position, not an assumed rest state.
    The last command at/before a time is held.  The caller must supply command
    history covering the requested delay window; fitting enforces that range.
    """
    commands_t = np.asarray(command_times_s, float)
    commands = np.asarray(commands_rad, float)
    observations_t = np.asarray(feedback_times_s, float)
    if (commands_t.ndim != 1 or commands.shape != commands_t.shape or not len(commands_t)
            or observations_t.ndim != 1 or not len(observations_t)
            or np.any(np.diff(commands_t) <= 0.) or np.any(np.diff(observations_t) <= 0.)
            or not np.isfinite(np.r_[commands_t, commands, observations_t,
                                      initial_position_rad, delay_s, time_constant_s]).all()
            or delay_s < 0. or time_constant_s <= 0.):
        raise ValueError("require sorted finite samples, nonnegative delay and positive tau")
    shifted = commands_t + float(delay_s)
    # Reduce large epoch values before accumulating integration intervals.
    origin = observations_t[0]
    shifted = shifted - origin
    samples = observations_t - origin
    result = np.empty_like(samples)
    result[0] = state = float(initial_position_rad)
    command_index = max(0, int(np.searchsorted(shifted, 0., side="right") - 1))
    previous = 0.
    for k in range(1, len(samples)):
        now = samples[k]
        while command_index + 1 < len(shifted) and shifted[command_index + 1] <= now:
            change = max(previous, float(shifted[command_index + 1]))
            state = commands[command_index] + (state - commands[command_index]) * np.exp(
                -(change - previous) / time_constant_s)
            previous = change
            command_index += 1
        state = commands[command_index] + (state - commands[command_index]) * np.exp(
            -(now - previous) / time_constant_s)
        previous = now
        result[k] = state
    return result


def _samples(times: Any, values: Any) -> tuple[np.ndarray, np.ndarray, int]:
    t, y = np.asarray(times, float), np.asarray(values, float)
    if t.ndim != 1 or y.ndim != 2 or len(t) != len(y):
        raise ValueError("times must be N and joint samples must be N by joints")
    finite = np.isfinite(t) & np.isfinite(y).all(axis=1)
    t, y = t[finite], y[finite]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    # Repeated readbacks of the same CAN packet are one observation.  Keep its
    # first recorded value; never create artificial samples from stale reads.
    keep = np.r_[True, np.diff(t) > 0.] if len(t) else np.zeros(0, bool)
    duplicates = int(len(t) - keep.sum())
    return t[keep], y[keep], duplicates


def fit_joint_response(command_times_s: Any, commands_rad: Any,
                       feedback_times_s: Any, feedback_rad: Any, *,
                       maximum_delay_s: float = .5, maximum_time_constant_s: float = 2.,
                       train_fraction: float = .7, metadata: dict[str, Any] | None = None
                       ) -> tuple[JointResponseModel, dict[str, Any]]:
    """Fit each axis and evaluate a later, untouched contiguous holdout window.

    Bounds describe the offline identification search, not motion stop gates.
    Constant inputs and locally unidentifiable delay/tau pairs are reported,
    rather than published as calibrated coefficients.  A poor held-out fit is
    visible in the report and is not silently replaced by another model.
    """
    from scipy.optimize import minimize

    if (not 0. < train_fraction < 1. or not np.isfinite(
            [maximum_delay_s, maximum_time_constant_s]).all()
            or maximum_delay_s <= 0. or maximum_time_constant_s <= 0.):
        raise ValueError("require positive search ranges and train fraction between zero and one")
    ct, command, command_duplicates = _samples(command_times_s, commands_rad)
    ft = np.asarray(feedback_times_s, float)
    feedback = np.asarray(feedback_rad, float)
    if feedback.ndim != 2 or command.shape[1:] != feedback.shape[1:]:
        raise ValueError("command and feedback joint counts must match")
    if ft.ndim == 1:
        ft = np.broadcast_to(ft[:, None], feedback.shape)
    if ft.shape != feedback.shape:
        raise ValueError("feedback times must be N or N by joints")
    count = command.shape[1]
    report: dict[str, Any] = {"model": "unit_gain_first_order_plus_host_visible_delay",
                             "delay_is_encoder_internal": False,
                             "holdout": "later_contiguous_window_not_used_for_fitting",
                             "command_duplicate_timestamps_removed": command_duplicates,
                             "axes": []}
    axes: list[JointAxisResponse | None] = []
    for axis_index in range(count):
        yt, observed, duplicates = _samples(ft[:, axis_index], feedback[:, axis_index, None])
        y = observed[:, 0]
        valid = ((yt >= ct[0] + maximum_delay_s) & (yt <= ct[-1])) if len(ct) else np.zeros(len(yt), bool)
        yt, y = yt[valid], y[valid]
        split = int(len(y) * train_fraction)
        item: dict[str, Any] = {"joint_index": axis_index,
                                "feedback_duplicate_timestamps_removed": duplicates,
                                "retained_samples": len(y), "train_samples": split,
                                "holdout_samples": len(y) - split}
        if len(ct) < 2 or split < 4 or len(y) - split < 2:
            axes.append(None)
            report["axes"].append(dict(item, status="insufficient_timestamped_samples"))
            continue
        origin = yt[0]
        at, ot = ct - origin, yt - origin
        u = command[:, axis_index]
        # Only input motion within this axis's training history is informative.
        train_u = u[(at >= -maximum_delay_s) & (at <= ot[split - 1])]
        span = float(np.ptp(train_u)) if len(train_u) else 0.
        item["command_span_rad"] = span
        if span <= np.finfo(float).eps * max(1., float(np.max(np.abs(train_u)))):
            axes.append(None)
            report["axes"].append(dict(item, status="unobservable_constant_command"))
            continue
        # An observed start position anchors all free-run predictions, including
        # holdout.  No later measurements are injected into this predictor.
        def prediction(params, upto=None):
            return predict_position_response(at, u, ot[:upto], y[0], *params)

        def objective(params):
            residual = prediction(params, split)[1:] - y[1:split]
            return float(np.mean(residual * residual))

        tau_floor = max(1e-6, min(maximum_time_constant_s * 1e-4,
                                  float(np.median(np.diff(ot))) * .01))
        bounds = [(0., maximum_delay_s), (tau_floor, maximum_time_constant_s)]
        delay_grid = np.linspace(0., maximum_delay_s, 9)
        tau_grid = np.geomspace(tau_floor, maximum_time_constant_s, 10)
        seeds = sorted((objective((d, tau)), (d, tau)) for d in delay_grid for tau in tau_grid)[:3]
        fits = [minimize(objective, seed, method="Powell", bounds=bounds,
                         options={"xtol": 1e-7, "ftol": 1e-10, "maxiter": 80})
                for _, seed in seeds]
        fit = min(fits, key=lambda value: value.fun)
        parameters = np.asarray(fit.x, float)
        predicted = prediction(parameters)
        unmodeled = u[np.clip(np.searchsorted(at, ot, side="right") - 1, 0, len(u) - 1)]
        # Local identifiability of the two fitted parameters.  This reports the
        # degeneracy of pure ramp data rather than pretending delay and tau are
        # independently known.  The condition is diagnostic, never a run gate.
        sensitivity = []
        for j in range(2):
            h = max(1e-6, parameters[j] * 1e-3)
            plus, minus = parameters.copy(), parameters.copy()
            plus[j] += h
            minus[j] = max(bounds[j][0], parameters[j] - h)
            sensitivity.append((prediction(plus, split) - prediction(minus, split)) /
                               (plus[j] - minus[j]))
        singular = np.linalg.svd(np.asarray(sensitivity).T[1:], compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0. else None
        boundary = bool(any(np.isclose(parameters[j], boundary_value, rtol=1e-4, atol=1e-6)
                            for j in range(2) for boundary_value in bounds[j]))
        result = JointAxisResponse(
            delay_s=float(parameters[0]), time_constant_s=float(parameters[1]),
            train_rmse_rad=float(np.sqrt(np.mean((predicted[1:split] - y[1:split]) ** 2))),
            holdout_rmse_rad=float(np.sqrt(np.mean((predicted[split:] - y[split:]) ** 2))),
            unmodeled_holdout_rmse_rad=float(np.sqrt(np.mean((unmodeled[split:] - y[split:]) ** 2))),
            train_samples=split, holdout_samples=len(y) - split,
            command_span_rad=span, local_parameter_condition=condition,
            fit_at_search_boundary=boundary,
            status="identified" if singular[-1] > np.finfo(float).eps * singular[0] else "unidentifiable_parameters")
        axes.append(result)
        report["axes"].append(dict(item, **asdict(result)))
    model_metadata = dict(metadata or {})
    model_metadata.update(maximum_delay_search_s=maximum_delay_s,
                          maximum_time_constant_search_s=maximum_time_constant_s,
                          training_fraction=train_fraction,
                          feedback_timestamp_limitation="host CAN receive time, not encoder internal sample time")
    return JointResponseModel(tuple(axes), model_metadata), report

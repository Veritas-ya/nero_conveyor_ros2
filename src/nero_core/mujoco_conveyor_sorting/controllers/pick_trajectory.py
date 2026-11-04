"""A smooth, conservatively timed joint curve through an entire pick path.

Clamped cubic interpolation preserves the waypoints and continuity of joint
position, velocity, and acceleration. A quintic time warp makes both ends
stationary without stopping at every intermediate waypoint. Limits are bounded
over the complete time-scaled curve using outward-rounded Bernstein polynomial
bounds, rather than only at sampled times.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline


_ARM_DOF = 7
_JointArray = npt.NDArray[np.float64]
_Interval = tuple[_JointArray, _JointArray]


def _point(value: Any) -> _Interval:
    values = np.asarray(value, dtype=float)
    return values, values


def _add(a: _Interval, b: _Interval) -> _Interval:
    return np.nextafter(a[0] + b[0], -np.inf), np.nextafter(a[1] + b[1], np.inf)


def _multiply(a: _Interval, b: _Interval) -> _Interval:
    products = np.array([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]])
    return (
        np.nextafter(np.min(products, axis=0), -np.inf),
        np.nextafter(np.max(products, axis=0), np.inf),
    )


def _poly_add(a: _Interval, b: _Interval) -> _Interval:
    shape = (max(len(a[0]), len(b[0])), max(a[0].shape[1], b[0].shape[1]))
    alo, ahi, blo, bhi = (np.zeros(shape) for _ in range(4))
    alo[:len(a[0])], ahi[:len(a[0])] = a
    blo[:len(b[0])], bhi[:len(b[0])] = b
    return _add((alo, ahi), (blo, bhi))


def _poly_multiply(a: _Interval, b: _Interval) -> _Interval:
    shape = (len(a[0]) + len(b[0]) - 1, max(a[0].shape[1], b[0].shape[1]))
    lower, upper = np.zeros(shape), np.zeros(shape)
    for i in range(len(a[0])):
        for j in range(len(b[0])):
            product = _multiply((a[0][i], a[1][i]), (b[0][j], b[1][j]))
            lower[i+j], upper[i+j] = _add((lower[i+j], upper[i+j]), product)
    return lower, upper


def _local_polynomial(coefficients: list[float], left: float, right: float) -> _Interval:
    """Enclose power coefficients after substituting u=left+(right-left)*w."""

    width = right - left
    affine = (
        np.array([[left], [np.nextafter(width, -np.inf)]]),
        np.array([[left], [np.nextafter(width, np.inf)]]),
    )
    polynomial = _point(np.array([[coefficients[-1]]]))
    for coefficient in coefficients[-2::-1]:
        polynomial = _poly_add(
            _poly_multiply(polynomial, affine), _point(np.array([[coefficient]]))
        )
    return polynomial


def _warp_at(u: float) -> tuple[float, float]:
    """Outward-rounded value of the monotone quintic warp at a scalar u."""

    value = _point(np.asarray(u))
    factor = _add(_point(10.0), _multiply(value, _add(_point(-15.0), _multiply(_point(6.0), value))))
    result = _multiply(_multiply(_multiply(value, value), value), factor)
    return float(result[0]), float(result[1])


def _inverse_warp_bracket(s: float) -> tuple[float, float]:
    """Enclose the inverse with monotone bisection, including rounding error."""

    if s == 0.0 or s == 1.0:
        return s, s
    lower, upper = 0.0, 1.0
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if middle == lower or middle == upper:
            break
        lo, hi = _warp_at(middle)
        if hi < s:
            lower = middle
        elif lo > s:
            upper = middle
        else:
            # A root exactly at the midpoint (for example s=0.5) must not
            # leave the original wide bracket. Probe each half separately.
            left, right = (lower + middle) / 2.0, (middle + upper) / 2.0
            changed = False
            if _warp_at(left)[1] < s:
                lower, changed = left, True
            if _warp_at(right)[0] > s:
                upper, changed = right, True
            if not changed:
                break
    return lower, upper


def _power_to_bernstein(polynomial: _Interval) -> _Interval:
    """Enclose Bernstein coefficients on w in [0, 1]."""

    degree = len(polynomial[0]) - 1
    lower, upper = np.zeros_like(polynomial[0]), np.zeros_like(polynomial[1])
    for k in range(degree + 1):
        for i in range(k + 1):
            ratio = math.comb(k, i) / math.comb(degree, i)
            weight = (np.nextafter(ratio, -np.inf), np.nextafter(ratio, np.inf))
            term = _multiply((polynomial[0][i], polynomial[1][i]), weight)
            lower[k], upper[k] = _add((lower[k], upper[k]), term)
    return lower, upper


def _split_bernstein(coefficients: _Interval) -> tuple[_Interval, _Interval]:
    """Outward-rounded de Casteljau subdivision at the interval midpoint."""

    low, high = coefficients
    left_low, left_high = np.empty_like(low), np.empty_like(high)
    right_low, right_high = np.empty_like(low), np.empty_like(high)
    left_low[0], left_high[0] = low[0], high[0]
    right_low[-1], right_high[-1] = low[-1], high[-1]
    for level in range(1, len(low)):
        low, high = _multiply(_add((low[:-1], high[:-1]), (low[1:], high[1:])), _point(0.5))
        left_low[level], left_high[level] = low[0], high[0]
        right_low[-level-1], right_high[-level-1] = low[-1], high[-1]
    return (left_low, left_high), (right_low, right_high)


def _bernstein_absolute_bounds(pieces: list[_Interval]) -> _JointArray:
    """Bound all values, refining convex hulls rather than sampling for safety.

    Endpoints provide lower estimates used only to stop refinement. Every
    returned upper bound comes from a Bernstein convex hull with outward
    rounding. The depth cap can reduce tightness but cannot reduce coverage.
    """

    observed = np.zeros(_ARM_DOF)
    maximum = np.zeros(_ARM_DOF)
    pending = [(piece, 0) for piece in pieces]
    while pending:
        (lower, upper), depth = pending.pop()
        endpoints_low, endpoints_high = lower[[0, -1]], upper[[0, -1]]
        endpoint_minimum = np.maximum(0.0, np.maximum(endpoints_low, -endpoints_high))
        observed = np.maximum(observed, np.max(endpoint_minimum, axis=0))
        bound = np.max(np.maximum(-lower, upper), axis=0)
        if depth >= 14 or np.all(bound <= observed * 1.0001 + 1e-12):
            maximum = np.maximum(maximum, bound)
        else:
            left, right = _split_bernstein((lower, upper))
            pending.extend(((left, depth + 1), (right, depth + 1)))
    return maximum


def _finite_scalar(value: Any, name: str) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _positive_limits(value: Any, name: str) -> _JointArray:
    try:
        limits = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive scalar or seven-element vector") from error
    if limits.ndim == 0:
        limits = np.full(_ARM_DOF, float(limits))
    if limits.shape != (_ARM_DOF,) or not np.all(np.isfinite(limits)):
        raise ValueError(f"{name} must be a finite scalar or seven-element vector")
    if np.any(limits <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return limits.copy()


class SmoothPickTrajectory:
    """Interpolate an ``N x 7`` joint path with one continuous motion.

    ``velocity_limit`` and ``acceleration_limit`` are finite, positive scalars
    or seven-element vectors, in rad/s and rad/s² respectively. Consecutive
    identical nodes are removed; a single remaining node defines a stationary
    trajectory. ``minimum_duration`` is a finite, non-negative number of seconds.

    Spline parameter distances use the largest joint displacement between each
    pair of nodes. The complete path receives a single quintic time warp, so
    intermediate nodes do not impose a stop. Cubic interpolation can curve away
    from the straight segments; use :meth:`verification_samples` to check this
    actual curve against joint and collision constraints before executing it.
    """

    def __init__(
        self,
        path: Any,
        velocity_limit: Any,
        acceleration_limit: Any,
        minimum_duration: float = 0.1,
    ) -> None:
        try:
            nodes = np.asarray(path, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("path must be a finite, nonempty N x 7 array") from error
        if nodes.ndim != 2 or nodes.shape[1] != _ARM_DOF or nodes.shape[0] == 0:
            raise ValueError("path must be a finite, nonempty N x 7 array")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("path must contain only finite joint positions")
        self._velocity_limit = _positive_limits(velocity_limit, "velocity_limit")
        self._acceleration_limit = _positive_limits(
            acceleration_limit, "acceleration_limit"
        )
        minimum = _finite_scalar(minimum_duration, "minimum_duration")
        if minimum < 0.0:
            raise ValueError("minimum_duration must be non-negative")

        keep = np.r_[True, np.any(nodes[1:] != nodes[:-1], axis=1)]
        self._path = nodes[keep].copy()
        self._spline: CubicSpline | None = None
        self._knots = np.array([0.0])
        self._max_qprime = np.zeros(_ARM_DOF)
        self._max_qsecond = np.zeros(_ARM_DOF)
        self._max_normalized_velocity = np.zeros(_ARM_DOF)
        self._max_normalized_acceleration = np.zeros(_ARM_DOF)
        self._duration = minimum
        if self._path.shape[0] == 1:
            return

        distances = np.max(np.abs(np.diff(self._path, axis=0)), axis=1)
        cumulative = np.r_[0.0, np.cumsum(distances)]
        if not np.all(np.isfinite(cumulative)) or cumulative[-1] <= 0.0:
            raise ValueError("path distances exceed the supported numeric range")
        self._knots = cumulative / cumulative[-1]
        if np.any(np.diff(self._knots) <= 0.0):
            raise ValueError("distinct path nodes must have representable parameter spacing")
        self._spline = CubicSpline(
            self._knots,
            self._path,
            axis=0,
            bc_type=((1, np.zeros(_ARM_DOF)), (1, np.zeros(_ARM_DOF))),
        )
        self._max_qprime, self._max_qsecond = self._derivative_bounds()
        self._max_normalized_velocity, self._max_normalized_acceleration = (
            self._timed_derivative_bounds()
        )

        velocity_duration = np.nextafter(
            self._max_normalized_velocity / self._velocity_limit, np.inf
        )
        acceleration_duration = np.nextafter(
            np.sqrt(np.nextafter(
                self._max_normalized_acceleration / self._acceleration_limit, np.inf
            )), np.inf
        )
        self._duration = max(
            minimum,
            float(np.max(velocity_duration)),
            float(np.max(acceleration_duration)),
        )
        if not np.isfinite(self._duration) or self._duration <= 0.0:
            raise ValueError("trajectory duration exceeds the supported numeric range")

    def _derivative_bounds(self) -> tuple[_JointArray, _JointArray]:
        """Find absolute extrema of the quadratic and linear derivatives."""

        assert self._spline is not None
        cubic, quadratic, linear, _ = self._spline.c
        widths = np.diff(self._knots)[:, None]
        prime_left = linear
        prime_right = (3.0 * cubic * widths + 2.0 * quadratic) * widths + linear
        maximum_prime = np.maximum(np.abs(prime_left), np.abs(prime_right))

        # The quadratic first derivative may have one interior extremum.
        vertex = np.zeros_like(cubic)
        np.divide(-quadratic, 3.0 * cubic, out=vertex, where=cubic != 0.0)
        inside = (cubic != 0.0) & (vertex > 0.0) & (vertex < widths)
        interior = (3.0 * cubic * vertex + 2.0 * quadratic) * vertex + linear
        maximum_prime = np.maximum(maximum_prime, np.where(inside, np.abs(interior), 0.0))

        second_left = 2.0 * quadratic
        second_right = 6.0 * cubic * widths + second_left
        maximum_second = np.maximum(np.abs(second_left), np.abs(second_right))
        return np.max(maximum_prime, axis=0), np.max(maximum_second, axis=0)

    def _timed_derivative_bounds(self) -> tuple[_JointArray, _JointArray]:
        """Enclose q_u and q_uu on the actual composed piecewise polynomial.

        On each cubic segment, q(f(u)) is a polynomial of degree 15. Its
        normalized velocity and acceleration have degrees 14 and 13. Inverse
        warp brackets enclose each segment's full time interval; interval
        coefficient arithmetic and Bernstein subdivision enclose every value.
        This retains correlation between path curvature and the time warp,
        unlike multiplying their unrelated global maxima.
        """

        assert self._spline is not None
        brackets = [_inverse_warp_bracket(float(s)) for s in self._knots]
        velocity_pieces, acceleration_pieces = [], []
        for interval in range(len(self._knots) - 1):
            left, right = brackets[interval][0], brackets[interval + 1][1]
            warp = _local_polynomial([0.0, 0.0, 0.0, 10.0, -15.0, 6.0], left, right)
            shift = _poly_add(warp, _point(np.array([[-self._knots[interval]]])))
            warp_prime = _local_polynomial([0.0, 0.0, 30.0, -60.0, 30.0], left, right)
            warp_second = _local_polynomial([0.0, 60.0, -180.0, 120.0], left, right)
            cubic, quadratic, linear = self._spline.c[:3, interval]
            prime = _poly_add(
                _multiply(_poly_multiply(shift, shift), _multiply(_point(cubic[None]), _point(3.0))),
                _poly_add(
                    _multiply(shift, _multiply(_point(quadratic[None]), _point(2.0))),
                    _point(linear[None]),
                ),
            )
            second = _poly_add(
                _multiply(shift, _multiply(_point(cubic[None]), _point(6.0))),
                _multiply(_point(quadratic[None]), _point(2.0)),
            )
            velocity = _poly_multiply(prime, warp_prime)
            acceleration = _poly_add(
                _poly_multiply(second, _poly_multiply(warp_prime, warp_prime)),
                _poly_multiply(prime, warp_second),
            )
            velocity_pieces.append(_power_to_bernstein(velocity))
            acceleration_pieces.append(_power_to_bernstein(acceleration))
        return (
            _bernstein_absolute_bounds(velocity_pieces),
            _bernstein_absolute_bounds(acceleration_pieces),
        )

    @property
    def duration(self) -> float:
        """Conservative synchronized duration, in seconds."""

        return self._duration

    def sample(self, t: float) -> _JointArray:
        """Return seven joint positions, clamping time to the motion interval."""

        return self.sample_full(t)[0]

    def sample_full(self, t: float) -> tuple[_JointArray, _JointArray, _JointArray]:
        """Return position, velocity, and acceleration using the chain rule."""

        time = _finite_scalar(t, "sample time")
        zeros = np.zeros(_ARM_DOF)
        if self._spline is None or time <= 0.0:
            return self._path[0].copy(), zeros.copy(), zeros
        if time >= self._duration:
            return self._path[-1].copy(), zeros.copy(), zeros

        u = time / self._duration
        s = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        sdot = 30.0 * u**2 * (1.0 - u) ** 2 / self._duration
        sddot = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u) / self._duration**2
        prime = self._spline(s, 1)
        position = self._spline(s)
        velocity = prime * sdot
        acceleration = self._spline(s, 2) * sdot**2 + prime * sddot
        return position, velocity, acceleration

    def _position_extrema_parameters(self) -> _JointArray:
        """Find every isolated real, interior root of every joint derivative."""

        assert self._spline is not None
        parameters = []
        for interval, width in enumerate(np.diff(self._knots)):
            for joint in range(_ARM_DOF):
                a, b, c = self._spline.c[:3, interval, joint]
                # Scale before forming the discriminant to avoid unnecessary
                # overflow; scaling does not change a polynomial's roots.
                scale = max(abs(a), abs(b), abs(c))
                if scale == 0.0:
                    continue  # A constant joint has no isolated extrema.
                a, b, c = 3.0 * (a / scale), 2.0 * (b / scale), c / scale
                if a == 0.0:
                    roots = () if b == 0.0 else (-c / b,)
                else:
                    discriminant = b * b - 4.0 * a * c
                    if discriminant < 0.0:
                        continue
                    # This form avoids cancellation for the smaller root.
                    numerator = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
                    roots = (-b / (2.0 * a),) if numerator == 0.0 else (
                        numerator / a, c / numerator
                    )
                for root in roots:
                    if 0.0 < root < width:
                        parameter = self._knots[interval] + root
                        if self._knots[interval] < parameter < self._knots[interval + 1]:
                            parameters.append(parameter)
        return np.unique(parameters)

    def verification_samples(self, max_joint_step: float = 0.004) -> _JointArray:
        """Return the ordered spline curve with bounded joint change per interval.

        A uniform parameter grid has spacing at most ``max_joint_step`` divided
        by the analytic maximum absolute joint derivative. Including all spline
        knots and joint position extrema only shortens those intervals, so every
        joint moves at most the requested step anywhere within each interval,
        including between samples. Exact extrema also expose joint-limit
        excursions that could otherwise fall between uniform samples. The result
        includes the original retained waypoints and both endpoints.
        """

        step = _finite_scalar(max_joint_step, "max_joint_step")
        if step <= 0.0:
            raise ValueError("max_joint_step must be strictly positive")
        if self._spline is None:
            return self._path.copy()
        interval_count = max(1, math.ceil(float(np.max(self._max_qprime)) / step))
        parameters = np.union1d(np.linspace(0.0, 1.0, interval_count + 1), self._knots)
        parameters = np.union1d(parameters, self._position_extrema_parameters())
        samples = np.asarray(self._spline(parameters))
        # Return the input values exactly at knots, even at the last endpoint.
        samples[np.searchsorted(parameters, self._knots)] = self._path
        return samples


__all__ = ["SmoothPickTrajectory"]

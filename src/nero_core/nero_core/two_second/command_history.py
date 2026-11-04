"""One actual-dispatch history across trajectory and stationary-hold senders."""
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


class CommandHistory:
    def __init__(self, config):
        self.config = config
        # Host dispatch differences describe commands, not measured actuator
        # acceleration. Observe by default; enforcement requires explicit opt-in.
        # Planning curve limits remain independent.
        self.check_command_acceleration = getattr(config, 'raw', {}).get(
            'runtime_command_acceleration_check_enabled', False)
        self.q = self.dispatch = None
        self.velocity = np.zeros(7)
        self.previous_interval = None
        self.command_count = self.acceleration_limit_exceedances = 0
        self.last_command_acceleration = None
        self.peak_abs_command_acceleration = np.zeros(7)

    def velocity_at(self, q, dispatch):
        if self.q is None:
            return np.zeros(7), None
        dt = dispatch-self.dispatch
        if dt <= 0:
            raise PickRejected('实际发送时钟未递增')
        return (np.asarray(q)-self.q)/dt, dt

    def validate(self, q, dispatch, *, limits=None):
        limits = self.config if limits is None else limits
        velocity, dt = self.velocity_at(q, dispatch)
        if dt is not None:
            # Finite-difference velocities belong to interval midpoints. Using
            # only the newest dt falsely rejects a bounded curve on uneven ticks.
            acceleration_dt = dt if self.previous_interval is None else .5*(dt+self.previous_interval)
            acceleration = (velocity-self.velocity)/acceleration_dt
            bad_velocity = np.any(abs(velocity) > limits.velocity_limits+1e-8)
            bad_acceleration = (self.check_command_acceleration and
                                np.any(abs(acceleration) > limits.acceleration_limits+1e-8))
            if not bad_velocity and not bad_acceleration:
                return
            details = {'dispatch_monotonic_s': dispatch, 'previous_dispatch_monotonic_s': self.dispatch,
                'actual_interval_s': dt, 'discrete_velocity_rad_s': velocity.tolist(),
                'previous_discrete_velocity_rad_s': self.velocity.tolist(),
                'acceleration_interval_s': acceleration_dt,
                'discrete_acceleration_rad_s2': acceleration.tolist(),
                'velocity_limits_rad_s': limits.velocity_limits.tolist(),
                'acceleration_limits_rad_s2': limits.acceleration_limits.tolist()}
            if bad_velocity:
                raise PickRejected('实际发送间隔下的关节速度超限', details=details)
            if bad_acceleration:
                raise PickRejected('实际发送间隔下的关节加速度超限', details=details)

    def validate_first(self, q, rest_anchor, *, limits=None):
        """A delayed first send cannot skip into the middle of a moving path."""
        limits = self.config if limits is None else limits
        dt = 1./self.config.settings.control_hz
        velocity = (np.asarray(q)-rest_anchor)/dt
        if (np.any(abs(velocity) > limits.velocity_limits+1e-8) or
                (self.check_command_acceleration and
                 np.any(abs(velocity)/dt > limits.acceleration_limits+1e-8))):
            raise PickRejected('首个发送点无法从静止端点连续接入')

    def record(self, dispatch, actual_dt, q, metadata, *, limits=None):
        # Check the real clock too: preparation can overrun its predicted slot.
        self.validate(q, dispatch, limits=limits)
        velocity, interval = self.velocity_at(q, dispatch)
        if interval is not None:
            acceleration_dt = interval if self.previous_interval is None else .5*(interval+self.previous_interval)
            acceleration = (velocity-self.velocity)/acceleration_dt
            self.last_command_acceleration = acceleration.copy()
            self.peak_abs_command_acceleration = np.maximum(self.peak_abs_command_acceleration, abs(acceleration))
            limits = self.config if limits is None else limits
            self.acceleration_limit_exceedances += int(np.any(abs(acceleration) > limits.acceleration_limits+1e-8))
        self.velocity, self.previous_interval = velocity, interval
        self.q, self.dispatch = np.asarray(q).copy(), dispatch
        self.command_count += 1

    def diagnostics(self):
        return {'source': 'host_dispatch_command_finite_difference',
            'measured_actuator_acceleration': False,
            'runtime_acceleration_check_enabled': self.check_command_acceleration,
            'commands_recorded': self.command_count,
            'commands_exceeding_planning_acceleration_limit': self.acceleration_limit_exceedances,
            'last_command_acceleration_rad_s2': None if self.last_command_acceleration is None else self.last_command_acceleration.tolist(),
            'peak_abs_command_acceleration_rad_s2': self.peak_abs_command_acceleration.tolist()}

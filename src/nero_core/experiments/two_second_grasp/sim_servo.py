"""Optional physical inner-loop model, not a calibrated factory controller.

Adds integral torque to MuJoCo position actuators to model a servo that removes
gravity/friction steady-state error. Never changes qpos, qvel, grasp state,
trajectory references or object attachment. Combined actuator torque is bounded
by the existing model's actuator force range. Its parameters must be reported.
"""
import mujoco
import numpy as np


class IntegralPositionServo:
    def __init__(self, env, integral_time_s):
        if not np.isfinite(integral_time_s) or integral_time_s <= 0:
            raise ValueError('simulation integral time must be positive')
        self.env, self.integral_time_s = env, float(integral_time_s)
        self.actuators = np.array([env.model.actuator(f'joint{i}').id for i in range(1, 8)])
        self.kp = env.model.actuator_gainprm[self.actuators, 0].copy()
        self.ki = self.kp/self.integral_time_s
        self.limits = env.model.actuator_forcerange[self.actuators].copy()
        self.integral = np.zeros(7)
        self.peak_torque = np.zeros(7)
        self.last_time = None
        self.saturated_steps = 0
        self.previous_callback = mujoco.get_mjcb_control()
        self.bound_callback = self.control
        mujoco.set_mjcb_control(self.bound_callback)

    def control(self, model, data):
        if self.previous_callback is not None:
            self.previous_callback(model, data)
        # Private IK/collision MjData must never integrate this physical servo.
        if data.qpos.ctypes.data != self.env.data.qpos.ctypes.data:
            return
        now = float(data.time)
        if self.last_time is None or now < self.last_time:
            self.integral[:] = 0.
        else:
            dt = min(now-self.last_time, float(model.opt.timestep))
            error = data.ctrl[self.actuators]-data.qpos[self.env.arm_qpos_adrs]
            self.integral = np.clip(self.integral+self.ki*error*dt, self.limits[:, 0], self.limits[:, 1])
        self.last_time = now
        bias = model.actuator_biasprm[self.actuators]
        pd = (self.kp*data.ctrl[self.actuators]+bias[:, 0]+
              bias[:, 1]*data.actuator_length[self.actuators]+
              bias[:, 2]*data.actuator_velocity[self.actuators])
        pd = np.clip(pd, self.limits[:, 0], self.limits[:, 1])
        combined = np.clip(pd+self.integral, self.limits[:, 0], self.limits[:, 1])
        correction = combined-pd
        self.saturated_steps += int(np.any(abs(correction-self.integral) > 1e-9))
        data.qfrc_applied[self.env.arm_dof_adrs] = correction
        self.peak_torque = np.maximum(self.peak_torque, abs(correction))

    def report(self):
        return {'model': 'force_limited_position_PD_plus_integral',
            'integral_time_s': self.integral_time_s, 'integral_gains_Nm_per_rad_s': self.ki.tolist(),
            'peak_added_torque_Nm': self.peak_torque.tolist(), 'saturated_physics_steps': self.saturated_steps,
            'hardware_identified': False, 'ideal_executor': False}

    def close(self):
        mujoco.set_mjcb_control(self.previous_callback)

"""Explicit common software conditions; physical calibration is never copied into a fake cell."""
import numpy as np
from scipy.spatial.transform import Rotation
from nero_core.two_second.config import CellConfig
from nero_core.two_second.feedback_gates import (RestArrival, GripConfirmation, OpenConfirmation,
    TRAJECTORY_ARRIVAL_TOLERANCE_RAD, ARRIVAL_DWELL_S, GRIP_CONFIRMATION_DWELL_S)
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.precise_ik import PrecisePoseSolver
from nero_core.two_second.motion_profile import select_motion_profile
from mujoco_conveyor_sorting.runtime.single_pick import PickCycle, PickRejected


class HardwareConditions:
    def __init__(self, path):
        self.config = CellConfig.load(path)
        self.settings = self.config.settings
        self.confirmation = None
        self.arrivals = {}
        self.retention = None
        self.opening = OpenConfirmation(self.settings.open_width_m)
        self.mirror = calibrated_model(self.config)
        self.solver = PrecisePoseSolver(self.mirror)

    def apply(self, overrides):
        c = self.config
        overrides['scene'].setdefault('simulation', {})['control_dt'] = 1./c.settings.control_hz
        robot = overrides['robot']['robot']
        robot.update(control_timestep=1./c.settings.control_hz,
            max_joint_velocity=c.velocity_limits.tolist(), velocity_limits=c.velocity_limits.tolist(),
            max_joint_acceleration=c.acceleration_limits.tolist(), acceleration_limits=c.acceleration_limits.tolist())
        robot.setdefault('controller', {})['joint_command_margin'] = self.mirror.joint_margin_rad
        overrides.setdefault('latency', {}).setdefault('latency', {})['randomization'] = {'enabled': False}

    def bind(self, env, *, gravity_compensation=False):
        self.env = env
        self.retention = PickCycle(None, None, self.settings, lambda *a: None,
                                   clock=lambda: float(env.data.time))
        # A force-limited physical actuator, never a weld or ideal pose setter.
        self.gripper_actuator = int(env.model.actuator('gripper').id)
        env.model.actuator_forcelimited[self.gripper_actuator] = 1
        env.model.actuator_forcerange[self.gripper_actuator] = [-self.settings.force_n, self.settings.force_n]
        if gravity_compensation:
            base = int(env.model.body('base_link').id)
            for body in range(env.model.nbody):
                parent = body
                while parent > 0 and parent != base:
                    parent = int(env.model.body_parentid[parent])
                if parent == base:
                    env.model.body_gravcomp[body] = 1.
        # Exactly the deployed private kinematic model and high-precision IK.

    def solve_world(self, xyz, quat, seed, *, cancelled):
        transform = self.env.nero_world_to_base
        position = transform[:3, :3]@xyz+transform[:3, 3]
        orientation = Rotation.from_matrix(transform[:3, :3]@
            Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()).as_quat()[[3, 0, 1, 2]]
        return self.solver.solve(position, orientation, seed, cancelled=cancelled)

    def begin_grip(self, now):
        self.confirmation = GripConfirmation(self.settings.open_width_m, self.settings.min_force_n, now)

    def grip_feedback(self, jaw, now):
        effort = np.asarray(jaw['motor_effort']).reshape(-1)
        state = {'width_m': float(jaw['opening_width']), 'force_n': float(abs(effort[0]))}
        if self.confirmation is None:
            return False
        if self.retention.held_width is not None:
            self.retention.check_holding(state)
            return True
        confirmed = self.confirmation.observe(state['width_m'], state['force_n'], now)
        if confirmed:
            self.retention.held_width = state['width_m']
        return confirmed

    def arrival(self, key, target, end_s, now, *, tolerance_rad=TRAJECTORY_ARRIVAL_TOLERANCE_RAD):
        gate = self.arrivals.setdefault(key, RestArrival(end_s,
            tolerance_rad=tolerance_rad))
        if gate.tolerance_rad != tolerance_rad:
            raise ValueError('arrival tolerance cannot change while a gate is active')
        actual = np.asarray(self.env.get_joint_positions())
        result = gate.observe(actual, np.asarray(target), now)
        if not result and now > end_s+self.settings.move_timeout_s:
            raise PickRejected('仿真在真机到位条件下超时', details={'phase': str(key),
                'max_joint_error_rad': float(np.max(abs(actual-target))),
                'tolerance_rad': gate.tolerance_rad, 'required_dwell_s': ARRIVAL_DWELL_S,
                'actual_joints_rad': actual.tolist(), 'target_joints_rad': np.asarray(target).tolist()})
        return result

    def report(self, env, gravity_compensation):
        cfg = self.config
        return {'config_source': str(cfg.source), 'config_sha256': cfg.source_sha256,
            'default_bin': cfg.default_bin, 'software_settings': {key: getattr(self.settings, key) for key in (
                'lift_m', 'force_n', 'open_width_m', 'min_force_n', 'move_timeout_s',
                'control_hz', 'speed_percent', 'full_plan_close_window_s')},
            'control_hz': 1./env.control_dt, 'velocity_limits_rad_s': env.robot_controller.max_joint_velocity,
            'acceleration_limits_rad_s2': env.robot_controller.max_joint_acceleration,
            'loaded_motion_profile': select_motion_profile(cfg, payload=True).diagnostics(),
            'arrival_tolerance_rad': TRAJECTORY_ARRIVAL_TOLERANCE_RAD, 'arrival_dwell_s': ARRIVAL_DWELL_S,
            'planning_knot_hz': cfg.raw['planning_knot_hz'],
            'planning_knot_hz_fallback': cfg.raw.get('planning_knot_hz_fallback',cfg.raw['planning_knot_hz']),
            'geometry_check_hz': cfg.raw['geometry_check_hz'],
            'moving_target_collision':'same 4mm visible-cloud predicate as hardware during approach/follow',
            'joint_margin_rad': self.mirror.joint_margin_rad,
            'joint_lower_rad': np.asarray(env.robot_controller.command_joint_lower_limits).tolist(),
            'joint_upper_rad': np.asarray(env.robot_controller.command_joint_upper_limits).tolist(),
            'closure_policy': cfg.raw.get('closure_policy', 'feedback'),
            'closure_feedback_role': ('diagnostic_only' if cfg.raw.get('closure_policy', 'feedback') == 'scheduled_attempt' else 'gate'),
            'grip_policy': f'shared_width_force_{GRIP_CONFIRMATION_DWELL_S:g}s_then_original_retention',
            'grip_confirmation_dwell_s': GRIP_CONFIRMATION_DWELL_S,
            'opening_policy': 'shared_width_2mm_new_feedback_0.15s',
            'gripper_force_cap_N': self.settings.force_n,
            'simulated_force_source': 'absolute_common_actuator_joint_force_N_not_factory_sensor_calibration',
            'simulated_gravity_compensation': bool(gravity_compensation),
            'hardware_gravity_compensation_verified': False,
            'same_physical_cell': False,
            'unmatched': ['synthetic_camera_and_belt_calibration', 'standby_joints_and_bin_pose',
                'full_mesh_payload_collision_vs_frozen_visible_cloud',
                'MuJoCo_position_actuator_vs_factory_inner_control_loop',
                'host_simulation_clock_vs_monotonic_CAN_dispatch', 'real_camera_noise_and_hardware_delay'],
            'physical_truth_use': 'independent_physical_evaluation_and_full_mesh_payload_collision_only'}

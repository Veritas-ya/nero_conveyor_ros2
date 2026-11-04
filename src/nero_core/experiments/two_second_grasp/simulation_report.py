"""Compact summaries of measured MuJoCo runs, never grasp control inputs."""
import numpy as np
from scipy.spatial.transform import Rotation


def summarize_retention(rows):
    """Read-only physical diagnostics; never used to authorize grasp control."""
    report = {}
    for phase in ('lift', 'transfer'):
        samples = [row for row in rows if row.get('phase') == phase and 'gripper_width_m' in row]
        if not samples:
            continue
        relative = np.array([row['object_in_tcp_xyz_m'] for row in samples])
        rotations = Rotation.from_rotvec([row['object_in_tcp_rotation_vector_rad'] for row in samples])
        width_errors = [abs(row['gripper_width_m']-row['confirmed_width_m']) for row in samples
                        if row['confirmed_width_m'] is not None]
        report[phase] = {'samples': len(samples),
            'maximum_width_change_m': max(width_errors, default=None),
            'maximum_relative_translation_change_m': float(np.max(np.linalg.norm(relative-relative[0], axis=1))),
            'maximum_relative_rotation_change_rad': float(np.max((rotations[0].inv()*rotations).magnitude())),
            'single_or_no_finger_contact_samples': sum(row['left_force_n'] <= 0 or row['right_force_n'] <= 0 for row in samples),
            'nonfinger_contact_samples': sum(bool(row.get('nonfinger_object_contacts')) for row in samples),
            'object_pose_source': 'mujoco_independent_diagnostic_only'}
    return report


def summarize(result):
    summary = {key: result.get(key) for key in (
        'status', 'profile', 'requested_count', 'completed_count', 'random_tless_category_sequence', 'speed_source',
        'duration_s', 'follow_s', 'brake_s', 'simulation_belt_set_m_s', 'velocity_world_m_s',
        'simulation_velocity_limits_rad_s', 'simulation_acceleration_limits_rad_s2',
        'planning_wall_s', 'ik_and_qva_wall_s', 'geometry_wall_s', 'delivery_sim_s',
        'scheduled_start_s', 'peak_joint_acceleration_rad_s2', 'physical_grasp_verified',
        'selected_join_s', 'synchronized_schedule', 'synchronized_schedule_rejections',
        'lift_object_rise_m', 'total_wall_s', 'final_sim_s', 'environment_construction_wall_s',
        'first_step_and_sensor_wall_s', 'hardware_connections', 'network_called', 'grasp_assist')}
    summary['loaded_motion_profile'] = result.get('condition_audit', {}).get('loaded_motion_profile')
    summary['simulation_belt_speed_range_m_s'] = result.get('simulation_belt_speed_range_m_s')
    summary['simulation_spawn_y_range_m'] = result.get('simulation_spawn_y_range_m')
    summary['bin_placement_mode'] = result.get('bin_placement_mode','fixed')
    summary['scenario_trials'] = result.get('scenario_trials')
    summary['trial_results'] = [{key:a.get(key) for key in (
        'index','category_id','spawn_record','velocity_world_m_s','frozen_speed_comparison','observed_center_world_m',
        'predicted_contact_tcp_world_m','lift_object_rise_m','return_verified','return_joint_error_rad','bin_space_selection',
        'selected_join_s','synchronized_schedule','synchronized_schedule_rejections')}
        | {'placement_stable':a.get('placement_verification',{}).get('stable',False)}
        for a in result.get('attempts',[])]
    summary['retention_diagnostics'] = result.get('retention_diagnostics', {})
    summary['placement_stable'] = result.get('placement_verification', {}).get('stable', False)
    summary['returned_to_wait'] = bool(result.get('status') == 'all_placements_and_returns_completed' and result.get('attempts') and
                                       all(a.get('return_verified', False) for a in result['attempts']))
    summary['return_joint_error_rad'] = [a.get('return_joint_error_rad') for a in result.get('attempts', [])]
    summary['contact_tracking_error_m'] = result.get('contact_feedback', {}).get('error_m')
    summary['stages'] = {}
    for kind in ('lift', 'transfer', 'return'):
        leg = result.get(f'planning_{kind}_result', {})
        summary['stages'][kind] = {key: leg.get(key) for key in (
            'duration_s', 'total_planning_wall_s', 'time_scale', 'peak_acceleration_rad_s2',
            'original_waypoints', 'waypoints', 'error', 'motion_profile', 'joint_acceleration_limits_rad_s2')}
    rows = result.get('command_rows', [])
    summary['recorded_control_steps'] = len(rows)
    summary['controller_clipped_steps'] = sum(r['clipped'] for r in rows)
    if len(rows) > 1:
        stamps = np.array([r['sim_s'] for r in rows])
        dt = np.diff(stamps)
        if np.all(dt > 0):
            velocities = np.array([r['applied_qd_rad_s'] for r in rows])
            summary['peak_discrete_applied_command_acceleration_rad_s2'] = np.max(
                abs(np.diff(velocities, axis=0)/dt[:, None]), axis=0).tolist()
        qacc_after_viewer_forward = np.max(
            abs(np.array([r['actual_qdd_rad_s2'] for r in rows])), axis=0).tolist()
        summary['peak_post_forward_qacc_rad_s2'] = qacc_after_viewer_forward
        # Retain historical readers, but the old name is not a substep dv/dt.
        summary['peak_actual_physics_joint_acceleration_rad_s2'] = qacc_after_viewer_forward
    trace=result.get('physics_substep_trace')
    if trace:
        peaks=trace['per_joint_absolute_peaks']
        summary['peak_physics_substep_velocity_difference_rad_s2']=[
            None if row is None else abs(row['value_rad_s2']) for row in peaks['velocity_difference']]
        summary['physics_substep_timestep_s']=trace['physics_timestep_s']
    summary['timing_notes'] = (
        'wall_s is host elapsed time; sim_s is MuJoCo simulation time; '
        'IK/qva and geometry are components of intercept planning_wall_s, do not add twice; '
        'control rows are post-step snapshots, not exact device dispatch timestamps; '
        'the historical peak_actual_physics_joint_acceleration_rad_s2 field is post-forward qacc, '
        'not the measured substep velocity difference; '
        'physical actuator response and contact impulses can exceed command acceleration; these are different measurements')
    return summary

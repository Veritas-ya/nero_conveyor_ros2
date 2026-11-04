"""Offline MuJoCo simulation. The output directory must not exist."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('output', description='New output directory; must not already exist'),
        DeclareLaunchArgument('seconds', default_value='55.0'),
        DeclareLaunchArgument('belt_speed', default_value='0.04'),
        DeclareLaunchArgument('seed', default_value='1543'),
        DeclareLaunchArgument('category_id', default_value='0'),
        DeclareLaunchArgument('spawn_x', default_value='-1.28'),
        DeclareLaunchArgument('minimum_approach_s', default_value='3.0'),
        DeclareLaunchArgument('compute_budget_s', default_value='3.0'),
        DeclareLaunchArgument('command_advance_s', default_value='0.1',
            description='Simulation-only reference advance; requires a newly certified command curve'),
    ]
    simulator = ExecuteProcess(
        cmd=['ros2', 'run', 'nero_core', 'physics_sim', '--mode', 'continuous',
             '--planning-mode', 'full_preplanned', '--no-video', '--through-bin',
             '--seconds', LaunchConfiguration('seconds'),
             '--belt-speed', LaunchConfiguration('belt_speed'),
             '--seed', LaunchConfiguration('seed'),
             '--category-id', LaunchConfiguration('category_id'),
             '--spawn-x', LaunchConfiguration('spawn_x'),
             '--minimum-approach-s', LaunchConfiguration('minimum_approach_s'),
             '--full-plan-compute-budget-s', LaunchConfiguration('compute_budget_s'),
             '--simulation-command-advance-s', LaunchConfiguration('command_advance_s'),
             '--output', LaunchConfiguration('output')],
        output='screen', additional_env={'MUJOCO_GL': 'egl', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    return LaunchDescription([*arguments, simulator])

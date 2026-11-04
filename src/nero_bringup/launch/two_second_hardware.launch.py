"""Defaults to file validation; execute must be explicitly requested."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def start(context):
    execute = LaunchConfiguration('execute').perform(context).lower()
    if execute not in ('true', 'false'):
        raise ValueError('execute must be true or false')
    arguments = ['--config', LaunchConfiguration('config').perform(context),
                 '--output', LaunchConfiguration('output').perform(context)]
    continuous=LaunchConfiguration('continuous').perform(context).lower()
    if continuous not in ('true','false'):
        raise ValueError('continuous must be true or false')
    arguments.extend(['--continuous'] if continuous=='true' else
        ['--count', LaunchConfiguration('count').perform(context)])
    arguments.append('--execute' if execute == 'true' else '--check-config')
    speed = LaunchConfiguration('belt_speed_cm_s').perform(context).strip()
    if speed:
        arguments.extend(['--belt-speed-cm-s', speed])
    preview = LaunchConfiguration('preview').perform(context).lower()
    if preview not in ('true','false'):
        raise ValueError('preview must be true or false')
    if preview == 'true':
        arguments.append('--preview')
    return [ExecuteProcess(cmd=['ros2', 'run', 'nero_core', 'two_second_pick', *arguments],
                 output='screen', additional_env={'OPENBLAS_NUM_THREADS': '1'})]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', description='现场标定配置文件，必须显式提供'),
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('count', default_value='1'),
        DeclareLaunchArgument('continuous', default_value='false',description='每轮回位后继续，Ctrl+C退出'),
        DeclareLaunchArgument('belt_speed_cm_s', default_value='', description='固定预测带速，cm/s；留空沿用配置'),
        DeclareLaunchArgument('preview', default_value='false'),
        DeclareLaunchArgument('output', default_value='runs/two_second_hardware'),
        OpaqueFunction(function=start)])

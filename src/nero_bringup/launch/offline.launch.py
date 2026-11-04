from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([Node(package='nero_bringup', executable='offline_demo', output='screen')])

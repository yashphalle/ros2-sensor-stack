from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Serial port for GPS device'
    )

    return LaunchDescription([
        port_arg,
        Node(
            package='gps_driver',
            executable='driver',    #setup.py
            arguments=[LaunchConfiguration('port')],
        ),
    ])
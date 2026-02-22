from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/pts/3',
        description='Serial port for GPS device'
    )

    return LaunchDescription([
        port_arg,
        Node(
            package='rtk_driver',
            executable='rtk_driver',   #setup.py entry point
            arguments=[LaunchConfiguration('port')],
        ),
    ])
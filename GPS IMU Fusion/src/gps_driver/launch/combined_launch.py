from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    gps_port_arg = DeclareLaunchArgument(
        'gps_port',
        default_value='/dev/ttyUSB0',
        description='Serial port for GPS device'
    )

    imu_port_arg = DeclareLaunchArgument(
        'imu_port',
        default_value='/dev/ttyUSB1',
        description='Serial port for IMU device'
    )

    pkg1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gps_driver'), 'launch', 'gps_launch.py')
        ),
        launch_arguments={'port': LaunchConfiguration('gps_port')}.items()
    )

    pkg2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('imu_driver'), 'launch', 'imu_launch.py')
        ),
        launch_arguments={'port': LaunchConfiguration('imu_port')}.items()
    )

    return LaunchDescription([gps_port_arg, imu_port_arg, pkg1_launch, pkg2_launch])

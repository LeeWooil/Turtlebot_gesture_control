from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    turtlebot3_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'launch/turtlebot3_world.launch.py'
            )
        ])
    )

    hand_gesture_publisher = Node(
        package='teleop_simulation_pkg',
        executable='hand_tracking_node2',
        output='screen'
    )

    teleop_turtlebot3 = Node(
        package='teleop_simulation_pkg',
        executable='turtlebot3_teleop_key',
        output='screen'
    )

    return LaunchDescription([
        turtlebot3_world,
        hand_gesture_publisher,
        teleop_turtlebot3
    ])


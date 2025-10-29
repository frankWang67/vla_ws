import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    # package_src_dir = get_package_share_directory("my_franka_description")
    package_src_dir = "src/my_franka_description"
    urdf_file = os.path.join(package_src_dir, "urdf", "panda_gripper_node.xacro")
    panda_mesh_dir = "package://my_franka_description/urdf/panda/meshes/"
    rviz_config_file = os.path.join(package_src_dir, "rviz", "vis.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_description",
                default_value=urdf_file,
                description="Path to URDF file",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(
                                [
                                    "xacro ",
                                    str(urdf_file),
                                    f" panda_mesh_dir:={panda_mesh_dir}",
                                ]
                            ),
                            value_type=str,
                        )
                    }
                ],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config_file],
            ),
        ]
    )

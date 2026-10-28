"""Run a softsim scenario under ROS 2.

    ros2 launch softsim_bringup softsim.launch.py                         # drop test, RViz
    ros2 launch softsim_bringup softsim.launch.py config:=drop_test/cases/w100.yaml viewer:=true
    ros2 launch softsim_bringup softsim.launch.py headless:=true output_dir:=runs/ros
    ros2 launch softsim_bringup softsim.launch.py overrides:="drop.height=0.2;drop.tilt=[5 deg, 0]"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    arg = LaunchConfiguration
    args = [
        DeclareLaunchArgument("scenario", default_value="drop_test", description="scenario name"),
        DeclareLaunchArgument("config", default_value="",
                              description="scenario YAML (absolute or relative to share/softsim_bringup/scenarios)"),
        DeclareLaunchArgument("overrides", default_value="", description="';'-separated key=value overrides"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("viewer", default_value="false", description="MuJoCo passive viewer"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="batch mode: no GUI, run once, save results, exit"),
        DeclareLaunchArgument("output_dir", default_value="", description="save results here when a run ends"),
        DeclareLaunchArgument("on_finish", default_value="hold", description="hold | loop | stop (GUI mode)"),
        DeclareLaunchArgument("publish_every", default_value="1"),
    ]
    headless = arg("headless")
    gui = PythonExpression(["'", headless, "' != 'true'"])

    def sim_node(gui_mode: bool) -> Node:
        return Node(
            package="softsim_ros",
            executable="softsim_node",
            name="softsim",
            output="screen",
            condition=UnlessCondition(headless) if gui_mode else IfCondition(headless),
            parameters=[{
                "scenario": arg("scenario"),
                "config": arg("config"),
                "overrides": arg("overrides"),
                "viewer": PythonExpression(["'", arg("viewer"), "' == 'true'"]) if gui_mode else False,
                "on_finish": arg("on_finish") if gui_mode else "stop",
                "output_dir": arg("output_dir") if gui_mode else PythonExpression(
                    ["'", arg("output_dir"), "' or 'runs/ros'"]),
                "publish_every": arg("publish_every"),
            }],
        )

    gui_node, batch_node = sim_node(True), sim_node(False)
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("softsim_bringup"), "rviz", "softsim.rviz"])],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(PythonExpression(["'", arg("rviz"), "' == 'true' and ", gui])),
    )
    # batch mode: shut the launch down once the simulation node exits
    stop_after_batch = RegisterEventHandler(OnProcessExit(target_action=batch_node,
                                                          on_exit=[EmitEvent(event=Shutdown())]))
    return LaunchDescription([*args, gui_node, batch_node, rviz, stop_after_batch])

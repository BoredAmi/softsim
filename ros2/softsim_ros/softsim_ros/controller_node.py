"""Closed-loop controller node: wraps :class:`softsim.control.controller.Controller` (plain Python, no ROS inside).

    ros2 run softsim_ros controller_node --ros-args -p surrogate:=/path/surrogate.json \
        -p outputs:="[bend_angle]" -p period:=0.02 -p ki:=6.0 -p use_sim_time:=true

Subscribes to the simulation node's ``/softsim/metrics/<output>`` (std_msgs/Float64) and to the targets
``~/target/<output>`` (Float64, SI), and publishes ``/softsim/actuators/<name>/command`` once per ``period`` (a timer
on ROS time: run it with ``use_sim_time:=true``, the simulation is slower than real time). It does nothing until a
measurement and a target have arrived. ``~/reset`` (std_srvs/Trigger) clears the integrator.

Parameters: ``surrogate`` (file), ``outputs`` (names to track; default all), ``period`` [s], ``ki`` [1/s],
``mode`` (integral | ff | mpc), ``damping``, ``max_correction``, ``sim_namespace`` ("/softsim").
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from softsim.control.controller import Controller
from softsim.errors import ConfigError


class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("softsim_controller")
        p = self.declare_parameter
        p("surrogate", "")
        p("outputs", rclpy.Parameter.Type.STRING_ARRAY)
        p("period", 0.02)
        p("ki", 4.0)
        p("mode", "integral")
        p("damping", 0.05)
        p("max_correction", 0.5)
        p("sim_namespace", "/softsim")
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        outputs = list(g("outputs")) if self.get_parameter("outputs").type_ != rclpy.Parameter.Type.NOT_SET else None
        try:
            self.ctl = Controller.from_json(g("surrogate"), outputs=outputs, dt=float(g("period")), ki=float(g("ki")),
                                            mode=g("mode"), damping=float(g("damping")),
                                            max_correction=float(g("max_correction")))
        except ConfigError as exc:
            raise SystemExit(f"controller_node: {exc}") from None
        ns = g("sim_namespace").rstrip("/")
        self.y: dict[str, float] = {}
        self.r: dict[str, float] = {}
        for o in self.ctl.outputs:
            self.create_subscription(Float64, f"{ns}/metrics/{o}", lambda m, o=o: self.y.__setitem__(o, m.data), 10)
            self.create_subscription(Float64, f"~/target/{o}", lambda m, o=o: self.r.__setitem__(o, m.data), 10)
        self.pubs = {n: self.create_publisher(Float64, f"{ns}/actuators/{n}/command", 10) for n in self.ctl.inputs}
        self.create_service(Trigger, "~/reset", self._reset)
        self.create_timer(float(g("period")), self._tick)

    def _reset(self, _req, res):
        self.ctl.reset()
        res.success = True
        return res

    def _tick(self) -> None:
        if len(self.y) < len(self.ctl.outputs) or len(self.r) < len(self.ctl.outputs):
            return
        try:
            u = self.ctl.step(self.y, self.r)
        except ConfigError as exc:
            self.get_logger().error(str(exc))
            return
        for n, v in zip(self.ctl.inputs, np.asarray(u)):
            self.pubs[n].publish(Float64(data=float(v)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

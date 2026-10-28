"""Generic softsim simulation node.

    ros2 run softsim_ros softsim_node --ros-args -p scenario:=drop_test
    ros2 run softsim_ros softsim_node --ros-args -p config:=drop_test/cases/w050.yaml \
        -p overrides:="drop.height=0.2;specimen.mesh.size=0.005"

Parameters (all optional):
    scenario        name; default config is <scenarios_dir>/<scenario>/scenario.yaml
    config          scenario YAML; absolute, or relative to scenarios_dir
    scenarios_dir   default: share/softsim_bringup/scenarios
    overrides       ';'-separated dotted overrides (softsim --set syntax)
    backend         override the config's backend
    frame_id        world frame name ("world")
    publish_every   publish every N output samples (1)
    publish_soft_state  also publish full SoftBodyState messages (false, heavy)
    metric_topics   one std_msgs/Float64 topic per metric for rqt_plot (true)
    autostart       start stepping immediately (true)
    on_finish       hold | loop | stop (stop shuts the node down, for batch runs)
    output_dir      if set, save results (config, timeseries, summary) when a run finishes
                    (hold/stop); in loop mode only on ~/save
    viewer          open the MuJoCo passive viewer (false)

The node publishes simulated time on /clock; it does not itself use sim time
(its stepping timer runs on the steady clock). Published stamps stay monotonic
across resets/loops (an offset accumulates), so TF and RViz never see time jump
back; the ``t`` metric is the scenario's own time.
"""

from __future__ import annotations

import datetime as _dt
import time
import traceback
from pathlib import Path

import rclpy
import yaml
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock as ClockMsg
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray

from softsim.runner import Simulation
from softsim_interfaces.msg import ContactWrenchArray, ScenarioMetrics, SimulationStatus, SoftBodyState
from softsim_interfaces.srv import RunAction, SetMaterial
from softsim_ros import conversions as conv

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _default_scenarios_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return str(Path(get_package_share_directory("softsim_bringup")) / "scenarios")
    except Exception:  # bringup not installed: configs must be absolute paths
        return ""


def _metric_topic(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class SoftSimNode(Node):
    """Runs one softsim Simulation and exposes it over ROS 2."""

    def __init__(self) -> None:
        super().__init__("softsim")
        p = self.declare_parameter
        p("scenario", "drop_test")
        p("config", "")
        p("scenarios_dir", _default_scenarios_dir())
        p("overrides", "")
        p("backend", "")
        p("frame_id", "world")
        p("publish_every", 1)
        p("publish_soft_state", False)
        p("metric_topics", True)
        p("autostart", True)
        p("on_finish", "hold")
        p("output_dir", "")
        p("viewer", False)
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.frame = g("frame_id")
        self.publish_every = max(1, int(g("publish_every")))
        self.publish_soft_state = bool(g("publish_soft_state"))
        self.metric_topics = bool(g("metric_topics"))
        self.on_finish = g("on_finish")
        if self.on_finish not in ("hold", "loop", "stop"):
            raise ValueError(f"on_finish must be hold, loop or stop, got {self.on_finish!r}")
        self.output_dir = g("output_dir")
        self.use_viewer = bool(g("viewer"))

        config = self._resolve_config(g("scenario"), g("config"), g("scenarios_dir"))
        overrides = [o.strip() for o in g("overrides").split(";") if o.strip()]
        self.get_logger().info(f"loading {config} overrides={overrides}")
        self.sim = Simulation(config, overrides, backend=g("backend") or None)

        self.clock_pub = self.create_publisher(ClockMsg, "/clock", 10)
        self.tf = TransformBroadcaster(self)
        self.contacts_pub = self.create_publisher(ContactWrenchArray, "~/contacts", 10)
        self.ground_pub = self.create_publisher(WrenchStamped, "~/ground_reaction", 10)
        self.metrics_pub = self.create_publisher(ScenarioMetrics, "~/metrics", 10)
        self.status_pub = self.create_publisher(SimulationStatus, "~/status", 10)
        self.rigid_markers_pub = self.create_publisher(MarkerArray, "~/rigid_markers", LATCHED)
        self.soft_markers_pub = self.create_publisher(MarkerArray, "~/soft_markers", 10)
        self.pose_pubs, self.twist_pubs, self.cloud_pubs, self.soft_state_pubs, self.metric_pubs = {}, {}, {}, {}, {}

        self.create_service(Trigger, "~/reset", self._srv_reset)
        self.create_service(SetBool, "~/set_paused", self._srv_paused)
        self.create_service(Trigger, "~/save", self._srv_save)
        self.create_service(SetMaterial, "~/set_material", self._srv_material)
        self.create_service(RunAction, "~/action", self._srv_action)

        self.viewer = None
        self.time_offset = 0.0  # added to sim time in stamps; grows on every reset
        self.last_stamp = -1.0
        self.paused = not bool(g("autostart"))
        self.finished = False
        self._on_rebuild()

        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.0005, self._tick, clock=steady)
        self.create_timer(0.5, self._publish_status, clock=steady)

    # -- setup -----------------------------------------------------------------

    def _resolve_config(self, scenario: str, config: str, scenarios_dir: str) -> str:
        if config:
            path = Path(config).expanduser()
            if not path.is_absolute() and scenarios_dir and not path.exists():
                path = Path(scenarios_dir) / config
        else:
            if not scenarios_dir:
                raise ValueError("no 'config' given and softsim_bringup (default scenarios) is not installed")
            path = Path(scenarios_dir) / scenario / "scenario.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"scenario config not found: {path}")
        return str(path)

    def _on_rebuild(self) -> None:
        """(Re)create per-body publishers and static data after a build."""
        scene = self.sim.scene
        for body in scene.rigid_bodies:
            if body.name not in self.pose_pubs:
                self.pose_pubs[body.name] = self.create_publisher(PoseStamped, f"~/rigid/{body.name}/pose", 10)
                self.twist_pubs[body.name] = self.create_publisher(TwistStamped, f"~/rigid/{body.name}/twist", 10)
        for body in scene.soft_bodies:
            if body.name not in self.cloud_pubs:
                self.cloud_pubs[body.name] = self.create_publisher(PointCloud2, f"~/soft/{body.name}/points", 10)
                if self.publish_soft_state:
                    self.soft_state_pubs[body.name] = self.create_publisher(
                        SoftBodyState, f"~/soft/{body.name}/state", 10)
        self.surface = conv.surface(scene)
        if self.last_stamp >= 0:
            self.time_offset = self.last_stamp + self.sim.config.simulation.timestep - self.sim.time
        self.rigid_markers_pub.publish(conv.rigid_markers(scene.rigid_bodies))
        self.rows, self.wall_start, self.sample = [], time.perf_counter(), 0
        self.rate_window = (time.perf_counter(), self.sim.time)
        self.realtime_factor = 0.0
        self.finished = False
        self._open_viewer()
        self._publish(self.sim.state())

    def _open_viewer(self) -> None:
        if not self.use_viewer:
            return
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        backend = self.sim.backend
        if not hasattr(backend, "model"):
            self.get_logger().warning(f"viewer requested but backend {backend.name} has none")
            return
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(backend.model, backend.data)

    # -- stepping ----------------------------------------------------------------

    def _tick(self) -> None:
        if self.paused or self.finished:
            return
        try:
            self.sim.step()
        except Exception as exc:  # divergence etc.: stop stepping, keep services alive
            self.get_logger().error(f"simulation stopped: {exc}")
            self.finished = True
            return
        state = self.sim.state()
        self.rows.append(self.sim.scenario.metrics(state))
        self.sample += 1
        if self.sample % self.publish_every == 0:
            self._publish(state, metrics=self.rows[-1])
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = None
        if self.sim.time >= self.sim.config.simulation.duration - 1e-12:
            self._finish()

    def _finish(self) -> None:
        self.finished = True
        wall = time.perf_counter() - self.wall_start
        self.get_logger().info(f"run finished: t={self.sim.time:.4f} s in {wall:.1f} s wall")
        if self.output_dir and self.on_finish != "loop":  # in loop mode, save on request (~/save)
            self._save(Path(self.output_dir))
        if self.on_finish == "loop":
            self.sim.reset()
            self._on_rebuild()
        elif self.on_finish == "stop":
            raise SystemExit(0)

    def _save(self, directory: Path) -> Path:
        name = self.sim.config.name or self.sim.scenario.name
        out = directory / f"{name}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
        result = self.sim.make_result(self.rows, time.perf_counter() - self.wall_start)
        result.save(out, self.sim.config.output.format)
        self.get_logger().info(f"saved results to {out}")
        return out

    # -- publishing ------------------------------------------------------------

    def _publish(self, state, metrics: dict | None = None) -> None:
        t, f = state.time + self.time_offset, self.frame
        self.last_stamp = t
        self.clock_pub.publish(ClockMsg(clock=conv.stamp(t)))
        self.tf.sendTransform([conv.transform(n, s, f, t) for n, s in state.rigid.items()])
        for name, s in state.rigid.items():
            self.pose_pubs[name].publish(conv.pose_stamped(s, f, t))
            self.twist_pubs[name].publish(conv.twist_stamped(s, f, t))
        soft_markers = MarkerArray()
        for i, (name, s) in enumerate(state.soft.items()):
            self.cloud_pubs[name].publish(conv.point_cloud(s.vertices, f, t))
            body = self.sim.scene.soft(name)
            soft_markers.markers.append(conv.soft_mesh_marker(name, i, s.vertices, self.surface[name], body.rgba, f, t))
            if name in self.soft_state_pubs:
                self.soft_state_pubs[name].publish(conv.soft_body_state(name, s.vertices, s.velocities, f, t))
        self.soft_markers_pub.publish(soft_markers)
        self.contacts_pub.publish(conv.contact_array(state.contacts, f, t))
        self.ground_pub.publish(conv.ground_reaction(state.contacts, f, t))
        metrics = metrics if metrics is not None else self.sim.scenario.metrics(state)
        self.metrics_pub.publish(conv.scenario_metrics(self.sim.scenario.name, metrics, f, t))
        if self.metric_topics:
            for key, value in metrics.items():
                if key not in self.metric_pubs:
                    self.metric_pubs[key] = self.create_publisher(Float64, f"~/metrics/{_metric_topic(key)}", 10)
                self.metric_pubs[key].publish(Float64(data=float(value)))

    def _publish_status(self) -> None:
        now = time.perf_counter()
        w0, s0 = self.rate_window
        if now > w0:
            self.realtime_factor = (self.sim.time - s0) / (now - w0)
        self.rate_window = (now, self.sim.time)
        cfg = self.sim.config
        msg = SimulationStatus(header=conv.header(self.frame, self.sim.time + self.time_offset), scenario=self.sim.scenario.name,
                               config_name=cfg.name or "", backend=self.sim.backend.name,
                               sim_time=self.sim.time, duration=cfg.simulation.duration,
                               realtime_factor=self.realtime_factor, paused=self.paused, finished=self.finished,
                               warnings=[str(x) for x in self.sim.findings if x.level != "info"])
        self.status_pub.publish(msg)

    # -- services ----------------------------------------------------------------

    def _guard(self, response, fn, ok_message: str):
        try:
            fn()
            response.success, response.message = True, ok_message
        except Exception as exc:
            self.get_logger().error(traceback.format_exc(limit=2))
            response.success, response.message = False, f"{type(exc).__name__}: {exc}"
        return response

    def _srv_reset(self, request, response):
        def run():
            self.sim.reset()
            self._on_rebuild()
        return self._guard(response, run, "reset")

    def _srv_paused(self, request, response):
        self.paused = bool(request.data)
        response.success, response.message = True, "paused" if self.paused else "running"
        return response

    def _srv_save(self, request, response):
        out = []
        return self._guard(response, lambda: out.append(self._save(Path(self.output_dir or "runs"))),
                           "saved")

    def _srv_material(self, request, response):
        def run():
            ref = yaml.safe_load(request.material) if request.material.strip().startswith("{") else request.material
            self.sim.set_material(request.body, ref)
            self._on_rebuild()
        return self._guard(response, run, f"material of {request.body!r} set; simulation reset")

    def _srv_action(self, request, response):
        def run():
            kwargs = yaml.safe_load(request.arguments) if request.arguments.strip() else {}
            if not isinstance(kwargs, dict):
                raise ValueError("arguments must be a YAML mapping")
            self.sim.action(request.action, **kwargs)
            self._on_rebuild()
        return self._guard(response, run, f"action {request.action!r} applied; simulation reset")

    def destroy_node(self):
        if self.viewer is not None:
            self.viewer.close()
        self.sim.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SoftSimNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

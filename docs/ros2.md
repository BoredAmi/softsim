# ROS 2 integration (Humble)

Three packages under `ros2/` wrap the core library. The physics stays in
`softsim`, so anything the CLI can run, the node can run.

| Package | Type | Contents |
| --- | --- | --- |
| `softsim_interfaces` | ament_cmake | `SoftBodyState`, `ContactWrench(Array)`, `ScenarioMetrics`, `SimulationStatus`; `SetMaterial`, `RunAction` services |
| `softsim_ros` | ament_python | `softsim_node`: generic node that runs any scenario |
| `softsim_bringup` | ament_cmake | `softsim.launch.py`, RViz config, and the repository's `scenarios/` installed to `share/softsim_bringup/scenarios` |

## Build

`softsim` and MuJoCo live in the project venv (see the README), so build and run
with the venv active. The node's shebang is `/usr/bin/env python3`, so it uses
whichever Python is active.

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --base-paths ros2        # from the repository root
source install/setup.bash
```

Always run with the same venv active: the generated message modules are
compiled against the venv's NumPy (1.26).

## Run

```bash
ros2 launch softsim_bringup softsim.launch.py                                   # drop test (20 g) + RViz
ros2 launch softsim_bringup softsim.launch.py config:=drop_test/cases/w100.yaml viewer:=true
ros2 launch softsim_bringup softsim.launch.py on_finish:=loop                   # repeat the scenario forever
ros2 launch softsim_bringup softsim.launch.py overrides:="drop.height=0.2;drop.tilt=[5 deg, 0]"
ros2 launch softsim_bringup softsim.launch.py headless:=true output_dir:=runs/ros   # batch: run, save, exit
```

| Launch argument | Default | Meaning |
| --- | --- | --- |
| `scenario` | `drop_test` | default config is `<scenario>/scenario.yaml` |
| `config` | | scenario YAML, absolute or relative to the installed `scenarios/` |
| `overrides` | | `;`-separated `key=value` (same syntax as `softsim --set`) |
| `rviz` | `true` | start RViz with `rviz/softsim.rviz` |
| `viewer` | `false` | MuJoCo passive viewer |
| `headless` | `false` | no GUI; run once, save to `output_dir` (default `runs/ros`), shut down |
| `output_dir` | | save results when a run finishes |
| `on_finish` | `hold` | `hold`, `loop` or `stop` |
| `publish_every` | `1` | publish every N output samples |

The node (`softsim`) also takes `backend`, `frame_id`, `publish_soft_state`,
`metric_topics`, `autostart` and `scenarios_dir` as parameters; see the
docstring in `softsim_ros/node.py`.

## Topics

All of these are under the node namespace (`/softsim/...`), except `/clock` and `/tf`.

| Topic | Type | Notes |
| --- | --- | --- |
| `/clock` | rosgraph_msgs/Clock | simulated time, monotonic across resets |
| `/tf` | world → each rigid body frame | |
| `rigid/<name>/pose`, `rigid/<name>/twist` | PoseStamped, TwistStamped | world frame |
| `rigid_markers` | MarkerArray (latched) | primitive parts, frame-locked to TF |
| `soft/<name>/points` | PointCloud2 | soft-body vertices, world frame |
| `soft_markers` | MarkerArray | soft-body surface meshes (TRIANGLE_LIST) |
| `soft/<name>/state` | SoftBodyState | only with `publish_soft_state:=true` |
| `contacts` | ContactWrenchArray | per body pair: force by `body_a` on `body_b`, centre of pressure |
| `ground_reaction` | WrenchStamped | total ground wrench, torque about the world origin |
| `metrics` | ScenarioMetrics | all scenario metrics; for the drop test: indentation, compression, bulge, forces, ... |
| `metrics/<name>` | std_msgs/Float64 | one topic per metric (rqt_plot / PlotJuggler) |
| `status` | SimulationStatus | 2 Hz: sim time, real-time factor, paused/finished, warnings |

## Services

| Service | Type | Example |
| --- | --- | --- |
| `reset` | std_srvs/Trigger | |
| `set_paused` | std_srvs/SetBool | `{data: true}` |
| `save` | std_srvs/Trigger | write the run so far to `output_dir` (or `runs/`) |
| `set_material` | softsim_interfaces/SetMaterial | `{body: specimen, material: '{preset: ecoflex_00_10, elastic: {youngs_modulus: 40 kPa}}'}` |
| `action` | softsim_interfaces/RunAction | `{action: drop, arguments: '{preset: czopek2026_w050, height: 0.2}'}` |

`set_material` and `action` rebuild the model and reset the scenario. Invalid
requests, including physics the backend cannot simulate, return
`success: false` with the reason, and the node keeps running.

```bash
ros2 service call /softsim/action softsim_interfaces/srv/RunAction \
  "{action: drop, arguments: '{mass: 0.1, height: 0.3}'}"
ros2 topic echo /softsim/metrics/indentation
```

## Timing

The simulation runs slower than real time (about 0.005–0.02× for the drop test,
depending on mesh). The node steps as fast as it can and publishes simulated
time on `/clock`, so RViz (`use_sim_time: true`) shows it in slow motion.
Consumers should use `use_sim_time`. The node's own timers use the steady
clock, so it never waits on the clock it publishes.

## Tests

```bash
python -m pytest ros2/softsim_ros/test -p no:launch_testing -p no:launch_ros   # ROS sourced, venv active
```

# Roadmap

## M1: core + MuJoCo drop test (done)

Config and units, materials and presets, soft/rigid geometry, tetrahedral
meshing, backend-neutral scene, capability system, MuJoCo flex backend, SOFA
stub, physics checks, drop-test scenario with all five dataset weights, runner,
sweeps, CLI, tests (including an analytic uniaxial check, convergence and
no-penetration checks), and docs.

## M2: ROS 2 layer, Humble (done; see [ros2.md](ros2.md))

- `ros2/softsim_interfaces` (ament_cmake): `SoftBodyState`, `ContactWrench(Stamped)`,
  `ScenarioMetrics`, and services `Reset`, `SetMaterial` (preset name or YAML),
  `RunAction` (action name + YAML kwargs; drop = mass/height/preset).
- `ros2/softsim_ros` (ament_python): a generic `softsim_node` that loads any
  scenario through `Simulation` (`scenario:=… config:=… overrides:=[…]`) and
  publishes `/clock` (simulated time; the simulation is ~0.005× real time),
  TF and pose/twist for rigid bodies, `PointCloud2` and `Marker` mesh for soft
  bodies, metrics, contact wrenches and ground reaction.
- `ros2/softsim_bringup`: launch with viewer on/off, headless batch mode, RViz config.

## M3: tracking + calibration

- Dataset registry (`datasets/*.yaml`: DOI, files, checksums, frame rate,
  overlay layout) and a loader that reads TIFFs straight from the zips.
- Tracking: green-key background, per-weight appearance (brass vs black),
  translucent cylinder, base flash masked. Outputs weight top-edge/knob
  trajectory, cylinder top and bulge width; px→mm from the 40 mm diameter;
  time from the frame index (2999.3 fps), checked against the overlay.
- Calibration: joint fit of (E, beta[, contact timeconst, friction]) across all
  five weights with scipy; coarse-to-fine mesh; per-case RMSE; fitted preset
  saved with mesh and backend provenance. Plots: sim vs experiment overlays,
  peak compression vs mass, rebound, settling, lift-off.

## M4: SOFA backend

Same scene through SofaPython3; Neo-Hookean vs Yeoh/Ogden on the same data, to
quantify what MuJoCo's model family misses at large strain.

## Later

More scenarios (quasi-static compression, indentation, bending, soft
actuators/fingers, grippers), MRE coupling, mesh import (CAD → gmsh), and a GPU
backend if the scale demands it.

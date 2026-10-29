# Roadmap

## M1: core + MuJoCo drop test (done)

Config and units, materials and presets, soft/rigid geometry, tetrahedral
meshing, backend-neutral scene, capability system, MuJoCo flex backend, SOFA
stub (replaced by the real backend in M4), physics checks, drop-test scenario with all five dataset weights, runner,
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

## M3: tracking + calibration (done; see [experiments.md](experiments.md))

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

## M4: SOFA backend (done; see [backends/sofa.md](backends/sofa.md))

- SOFA v26.06 backend through SofaPython3, found via `$SOFA_ROOT`: linear,
  StVK, classic and stable Neo-Hookean, Mooney-Rivlin, one-term Ogden,
  Rayleigh/Kelvin-Voigt damping, rigid bodies, frictional Lagrange contact,
  bonded bases. Material mappings checked against analytic uniaxial
  compression; SOFA tests skip without SOFA.
- `stable_neo_hookean` split from `neo_hookean`: MuJoCo's flex model is the
  stable variant and now says so; MuJoCo refuses classic Neo-Hookean.
- Calibration hardening: blow-up guard in the SOFA backend (it segfaults
  instead of producing NaN), crash-tolerant and resumable calibrations,
  ground-penetration metric.
- Results ([experiments.md](experiments.md#backends-and-material-models-m4)):
  Ogden fits 20/50 g within 3–11% but no material law fits 500 g; it prefers
  less large-strain stiffening than Neo-Hookean. Compressibility (ν = 0.45)
  explains part of the 500 g gap; rate dependence and the table are the
  remaining suspects.

## Next

- **Run the bulk-modulus refit** (`calibrations/ecoflex_00_10_drop_sofa_ogden_bulk.yaml`,
  about 3–4 h; then again at a 3.5 mm mesh to separate locking). Needs a
  faster machine than the development laptop.
- Prony-series viscoelasticity to test rate dependence. The SOFA hyperelastic
  force field softsim uses has no viscoelastic terms, so this needs another
  SOFA component, a custom force field or another backend.
- A compliant support (table) model, constrained by the tracked table motion.
- Yeoh and multi-term Ogden in SOFA (custom force field).
- The SOFA frictionless-plate contact artefact (7% stiff) and the rim-flaring
  ground penetration.

## Later

More scenarios (quasi-static compression, indentation, bending, soft
actuators/fingers, grippers), MRE coupling, mesh import (CAD → gmsh), and a GPU
backend if the scale demands it.

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

## Direction

softsim aims to be the soft-robotics package where every model is **validated
against real data**, independent of the physics engine. It grows in three
stages: validated sim-to-real (M1–M6), then design and control of real soft
robots (M6–M10), then GPU learning (M11). Every robot family is planned
(pneumatic, cable, gripper, magnetic). Every feature ships with a capability
flag, a refusal where unsupported, an analytic test and a published-data
validation.

## Drop test after the first contact (done, between M4 and M5)

See [experiments.md](experiments.md#after-the-first-contact-tack-and-the-hopping-specimen).

- **Recordings re-read:** the specimen hops and drags the cloth up; it is not
  the table moving (M3 was wrong). The first 100 ms are now usable data.
- **Contact adhesion** (`material.adhesion`, `ground.adhesion`): a cohesive
  law, backend-neutral (`softsim/contact/adhesion.py`), on MuJoCo and SOFA,
  with an explicit-stiffness check, verified against analytic equilibria on
  both backends.
- **Spring-mounted rigid bodies** (`mount: slider_z`, MuJoCo) and the drop
  test's compliant `support`.
- **Calibration over any observable and window** (weight-relative position,
  specimen lift), not only the first contact.
- **Fixes** in the SOFA backend: light rigid bodies (CG → direct solve) and
  pinned bodies on the ground (NaN).
- **Reference FEM** (`softsim.fem`, [reference_fem.md](reference_fem.md)):
  axisymmetric P2/P0, locking-free, JAX tangents, Prony viscoelasticity,
  punch and ground contact, verified against analytic solutions. Its 50 g
  drop disagrees with MuJoCo by 40% (open).

## M5: foundations for robots

- **Named regions** on soft bodies (cavity surfaces, clamp faces, cable
  channels), resolved at meshing time. Fixed boundaries become region-based.
- **Geometry:** mesh and CAD import (STL/OBJ/STEP through gmsh, .msh/.vtk), with
  physical groups mapped to regions. Parametric bodies with internal cavities
  (PneuNet finger, cable-finger body). Multi-material bodies
  (strain-limiting layer).
- **Actuation and control API:** `pressure`, `cable`, region force and
  prescribed displacement actuators. Backend `set_controls()` and
  `actuator_state()`. Control schedules in YAML and Python callbacks. ROS 2
  command and state topics.
- **Backend mappings:**
  - SOFA: SoftRobots `SurfacePressureConstraint` and `CableConstraint` (both
    ship in the v26.06 binary and instantiate; a first cable probe was not yet
    physical, which is the first thing to fix).
  - MuJoCo: pressure as nodal forces on vertex bodies, cables as spatial
    tendons.
- **Damping debt:** Rayleigh α on SOFA. (The compliant support and
  post-contact observables are done; see above.)
- **Analytic checks:** pressurised thick-walled cylinder and sphere (Lamé), and
  a cable-bent cantilever (Euler–Bernoulli), on both backends.

## M6: pneumatic finger

PneuNet scenario (pressure → bending angle, tip force, cavity volume), public
datasets registered with DOI and licence, and a **generic experiment API** so
calibration and plots work for any scenario. Yeoh in SOFA (custom force field)
or refused.

## M7: cable finger, soft arm and rods

Cable-finger scenario and a multi-segment cable arm. A rod (Cosserat) backend
for slender bodies: SOFA BeamAdapter, PyElastica or our own. Rod vs FEM
cross-check.

## M8: soft gripper and grasping

N-finger gripper on a rigid palm. Grasp metrics (success, contact patch,
forces, slip, pull-out). Contact quality work, including the M4 SOFA
artefacts, finger self-collision and a friction benchmark. A grasp benchmark
grid.

## M9: magnetic soft robots (MRE)

Hard-magnetic elastomer model (per-element magnetisation in an external
field). Magnetic cantilever check against analytic and published data, and a
magnetic crawler/gripper.

## M10: design and control toolkit

Quasi-static inverse control (optimisation; SoftRobots.Inverse fast path on
SOFA), reduced-order and Koopman surrogates for real-time ROS 2 control,
design optimisation over parametric geometry (`softsim optimize`), and
sim-to-real identification from short recorded motions.

## M11: GPU backend and learning (GPU machine)

Evaluate MuJoCo Warp/MJX, NVIDIA Warp FEM and Genesis for correctness, speed
and differentiability, and implement the best behind the same interface with a
batched API. Gymnasium environments per robot, with domain randomisation over
the calibrated error bars. Run the deferred long calibrations (bulk-modulus
refit, finer meshes).

## M12: benchmarks, docs, release (ongoing)

`softsim bench` writes `docs/benchmarks.md` (every validated scenario × every
backend: error vs data, wall time, mesh). Also: a docs site with tutorials, a
pip release, CI (with a SOFA-skipping job and a nightly SOFA job) and
contribution guides for robots, presets and datasets.

## Open items

- **The 500 g first contact** stops 2–3× harder and faster than any model
  allows, and the tracking is confirmed independently. The cause is unknown;
  it is kept out of fits until understood.
- **Rate-dependent tack:** the cohesive law is elastic. The 50 g weight's
  peel-and-flip loses more energy than an elastic bond would.
- **Spring mounts on SOFA** need per-contact force reporting (SOFA reports
  contacts by momentum balance, which can't separate support–ground from
  support–specimen).
- The bulk-modulus refit (`calibrations/ecoflex_00_10_drop_sofa_ogden_bulk.yaml`)
  was stopped after 12 evaluations without improvement; it targets 500 g,
  whose data is now in question. Its evaluations are kept and it resumes if
  re-run.
- Prony viscoelasticity to test rate dependence.
- The SOFA frictionless-plate artefact (7% stiff) and rim-flare ground
  penetration (M8).

## Later

Material-characterisation scenarios (quasi-static compression, indentation,
bending), more robot morphologies (crawlers, swimmers, continuum arms), and
fluid–structure coupling.

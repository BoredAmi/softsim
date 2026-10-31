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
stages: validated sim-to-real (M1–M7, with soft materials first in M5), then design and
control of real soft robots (M7–M11), then GPU learning (M12). Every robot family is planned
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

## M5: soft-material fidelity (in progress)

Goal: a physical material card that runs with the same physics on every
backend, with stated error. Every value can be changed from YAML, CLI, Python
or ROS. The cards rest on public material-test data. Acceptance test: all five
drop weights. See [materials.md](materials.md).

**Done:**

- **Material cards:**
  - `kind: physical | effective`, `validity`, `uncertainty`, `references`;
  - any two isotropic constants as input; `poisson_ratio` for the bulk-modulus
    laws;
  - new laws Gent and Arruda–Boyce;
  - new units (1/s, J/m², Pa·s, K/°C, deg/s).
- **Parameterisation:**
  - `softsim params` lists every changeable value with its unit and range;
  - `softsim material show`;
  - `load_material(overrides=...)`, `Material.with_overrides`, `sample()` from
    the uncertainty;
  - `Simulation.set_material(body, None, overrides)` and the same overrides on
    the ROS `set_material` service.
- **Homogeneous tests** (`softsim.materials.homogeneous`): uniaxial,
  equibiaxial, planar, simple shear, volumetric and confined. Also relaxation,
  constant-rate loading and DMA with Prony. All checked against closed forms,
  plus `softsim material plot`.
- **Public data and fitting:**
  - the Marechal et al. 2021 tension data (ODbL; fetched, not redistributed);
  - `softsim fit-material` over 7 laws, with AIC/BIC, uncertainties and a
    stability check;
  - `softsim material project` for backends that lack a law; the backends'
    refusal messages point to it;
  - a generated library of 18 physical cards ([material_library.md](material_library.md)).
- **Prony viscoelasticity on MuJoCo and SOFA:** the long-term law is native,
  and the Maxwell overstress is applied as explicit vertex forces with the
  reference's update (exact against the homogeneous reference). It comes with
  an explicit-stability check.
- **Drop validation over all five weights** (`tools/validate_drop_bounces.py`).
  20 and 50 g are reproduced. 100 and 200 g are tilted edge impacts. 500 g is
  stiffer and lossier than any model.
- **In-plane tilt at contact** from the weights' side edges
  (`softsim.experiments.video.tilt`; reliable for the dark weights). It is
  applied in every simulation of a recording, and it brings 500 g's specimen
  hop to within 10%.
- **3D locking-free reference** (`softsim.fem.solid3d`, 10-node tetrahedra,
  P2/P0, 6-DOF rigid bodies, sticking, Coulomb and adhesive contact, Prony
  inside the implicit solve, PARDISO) and the **`fem` backend**. It is checked
  against homogeneous solutions, and against the axisymmetric solver within
  4% on a coarse mesh.
- **A physical Ecoflex 00-10 card** (Neo-Hookean 143 kPa + Prony 0.69 at 2 ms)
  reproduces the 20 and 50 g first contacts on the 3D reference within 7% and
  10%, with each recording's offset and spin.
- **Transfer:** reference runs saved as tracks; MuJoCo calibrated against
  them (3–10% error bar), SOFA likewise (3–7%,
  `calibrations/transfer_ecoflex_00_10_sofa.yaml`).
- **Second dataset:** Roels et al. 2025 (CC-BY-4.0), tension, compression and
  3 h relaxation for 7 silicones. Their cards are fitted to tension and
  compression together, with Prony from the relaxation, and are now the
  preferred ones (25 cards in the library). `fit-material` handles several
  specimens and test kinds.
- **Ogden in 3D** through matrix functions (slow; project to Yeoh or Gent for
  long runs).
- **Joint fits:** calibration files take `material_tests` terms (datasets or
  inline datasheet points) next to simulated cases. Prepared:
  `calibrations/ecoflex_00_10_drop_fem_joint.yaml`, the drop plus the
  datasheet's 100% modulus. The drop-only card relaxes to about half the
  datasheet's quasi-static stiffness.
- **Sensitivity and uncertainty:** `softsim sensitivity` (Morris screening)
  and `softsim sweep --sample N` from a card's uncertainty.
- **Contact fidelity,** measured against the reference with elastic cards
  (`tools/contact_restitution.py`):
  - MuJoCo's flex elasticity is damped like backward Euler (ζ ≈ ω·dt/2);
  - its contact settings don't change impact losses;
  - without friction it is within 3.5% at meshes of 3.5 mm or finer, while
    soft-constraint friction slip costs about twice as much as everything
    else; `contact.cone` / `contact.impratio` mitigate part of it
    ([backends/mujoco.md](backends/mujoco.md#energy-and-contact-fidelity));
  - SOFA loses less with sticking friction, but much more frictionless.
- **500 g:**
  - out-of-plane tilt ruled out (a 20° sweep closes 17% of a 2× gap);
  - a model-free check shows no passive specimen can explain it: the
    recording needs 12× the 50 g force at the same indentation and rate.
- **3D solver ~1.8× faster** (fixed sparsity pattern, PARDISO analysis once,
  closed-form tangents); `tools/bench_fem.py` for the GPU machine.

**Next:**

- **Long 3D fits on the faster machine** ([new_machine.md](new_machine.md)):
  the drop-only and the joint physical card, then new reference tracks and
  transfers.
- **Whole bounce in 3D** (tack + support) for 20 and 50 g.
- **SOFA contact fidelity:** frictionless impacts lose 58% of the energy
  (e = 0.65 against 0.87, [backends/sofa.md](backends/sofa.md#known-limits)),
  plus the M4 rim-flare penetration.
- **Rate data** below 1 s (DMA or impact tests) for silicones. Today's
  relaxation data start at 1 s, while impacts relax within milliseconds.
- **500 g:** with the material excluded, check the setup and the measurement
  (a second camera view, or the original authors' notes).

## M6: foundations for robots (in progress)

Done so far ([actuation.md](actuation.md)):
- **Named surface regions** on every soft shape, resolved on the mesh; `fixed`
  regions pin vertices.
- **Hollow shapes:** `hollow_sphere` and `closed_tube` (balloon actuator), gmsh.
- **Pressure actuators** on all three backends: exact `-p V` follower load in
  the reference; explicit follower forces on MuJoCo and SOFA. Schedules in YAML,
  `set_control` from Python, ROS 2 command and state topics.
- **`inflation` scenario** (pressure–volume curve and extent).
- **Check:** the reference matches the exact finite-strain thick-sphere
  solution within 0.1% up to 62% hoop strain, and it finds the limit pressure.
- **Backend error bar:** at the same card, MuJoCo and SOFA agree with each other
  but inflate 27% less than the reference at 3 mm (13% at 2 mm): linear-tet
  locking at ν = 0.45.
- **`softsim transfer`** (M5.6): fits a fast backend's effective card against
  the reference and checks it on unseen load cases. For MuJoCo at 3 mm, E 30 →
  23.7 kPa takes the inflation error from 20% to 0.5%, and it holds at 1.5 and
  4.5 kPa (1.2%, 2.2%). SOFA needs the same correction (23.6 kPa).
- **Bending actuator** (off-axis cavity, `scenarios/inflation/bending.yaml`) with
  tip and bend-angle outputs. MuJoCo at 2 mm bends 37% less than the reference.
  After transfer, the error is 4% at the fitted pressure and 8% at half of it.

Still to do:
- **Geometry:** mesh and CAD import (STL/OBJ/STEP through gmsh, .msh/.vtk), with
  physical groups mapped to regions. Parametric bodies with internal cavities
  (PneuNet finger, cable-finger body). Multi-material bodies
  (strain-limiting layer).
- **More actuators:** `cable`, region force and prescribed displacement, with
  Python control callbacks.
- **Backend mappings:**
  - SOFA: SoftRobots `CableConstraint` (ships in the v26.06 binary; a first
    cable probe was not yet physical, which is the first thing to fix);
    `SurfacePressureConstraint` as an implicit alternative to the explicit
    pressure forces.
  - MuJoCo: cables as spatial tendons.
- **Damping debt:** Rayleigh α on SOFA. (The compliant support and
  post-contact observables are done; see above.)
- **Analytic checks:** the pressurised thick-walled cylinder (Lamé) and a
  cable-bent cantilever (Euler–Bernoulli), on every backend.

## M7: pneumatic finger

PneuNet scenario (pressure → bending angle, tip force, cavity volume), public
datasets registered with DOI and licence, and a **generic experiment API** so
calibration and plots work for any scenario. Yeoh in SOFA (custom force field)
or refused.

## M8: cable finger, soft arm and rods

Cable-finger scenario and a multi-segment cable arm. A rod (Cosserat) backend
for slender bodies: SOFA BeamAdapter, PyElastica or our own. Rod vs FEM
cross-check.

## M9: soft gripper and grasping

N-finger gripper on a rigid palm. Grasp metrics (success, contact patch,
forces, slip, pull-out). Contact quality work, including the M4 SOFA
artefacts, finger self-collision and a friction benchmark. A grasp benchmark
grid.

## M10: magnetic soft robots (MRE)

Hard-magnetic elastomer model (per-element magnetisation in an external
field). Magnetic cantilever check against analytic and published data, and a
magnetic crawler/gripper.

## M11: design and control toolkit

Quasi-static inverse control (optimisation; SoftRobots.Inverse fast path on
SOFA), reduced-order and Koopman surrogates for real-time ROS 2 control,
design optimisation over parametric geometry (`softsim optimize`), and
sim-to-real identification from short recorded motions.

## M12: GPU backend and learning (GPU machine)

Evaluate MuJoCo Warp/MJX, NVIDIA Warp FEM and Genesis for correctness, speed
and differentiability, and implement the best behind the same interface with a
batched API. Gymnasium environments per robot, with domain randomisation over
the calibrated error bars. Run the deferred long calibrations (bulk-modulus
refit, finer meshes).

## M13: benchmarks, docs, release (ongoing)

`softsim bench` writes `docs/benchmarks.md` (every validated scenario × every
backend: error vs data, wall time, mesh). Also: a docs site with tutorials, a
pip release, CI (with a SOFA-skipping job and a nightly SOFA job) and
contribution guides for robots, presets and datasets.

## Open items

- **The 500 g first contact** stops 2–3× harder and faster than any model
  allows, and the tracking is confirmed independently. It lands about 11°
  tilted in the image plane, with an unknown out-of-plane tilt. The cause is
  unknown, and it is kept out of fits until understood (M5).
- **Rate-dependent tack:** the cohesive law is elastic. The 50 g weight's
  peel-and-flip loses more energy than an elastic bond would.
- **Spring mounts on SOFA** need per-contact force reporting (SOFA reports
  contacts by momentum balance, which can't separate support–ground from
  support–specimen).
- The bulk-modulus refit (`calibrations/ecoflex_00_10_drop_sofa_ogden_bulk.yaml`)
  was stopped after 12 evaluations without improvement; it targets 500 g,
  whose data is now in question. Its evaluations are kept and it resumes if
  re-run.
- Rate dependence: Prony now runs on every backend (M5). What is missing is
  rate data for silicones, and a test of whether any Prony spectrum explains
  the heavy impacts.
- The SOFA frictionless-plate artefact (7% stiff) and rim-flare ground
  penetration (M9).

## Later

Material-characterisation scenarios (quasi-static compression, indentation,
bending), more robot morphologies (crawlers, swimmers, continuum arms), and
fluid–structure coupling.

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
  - (since withdrawn: the two weights do not load the specimen alike, docs/experiments.md) a model-free check shows no passive specimen can explain it: the
    recording needs 12× the 50 g force at the same indentation and rate.
- **3D solver ~1.8× faster** (fixed sparsity pattern, PARDISO analysis once,
  closed-form tangents); `tools/bench_fem.py` for the GPU machine.

**Next:**

- **Long 3D fits on the faster machine** ([new_machine.md](new_machine.md)):
  the drop-only and the joint physical card, then new reference tracks and
  transfers.
- **Whole bounce in 3D** (tack + support) for 20 and 50 g.
- **SOFA contact fidelity:** the "58% loss" frictionless was a friction mismatch (SOFA uses one coefficient for every
  pair); with every pair frictionless it loses 3.4% more than the reference and peaks 14% shallow (2026-11-11,
  [backends/sofa.md](backends/sofa.md#known-limits)). Left: the M4 rim-flare penetration, and the frictionless plate.
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
- **Cable actuators** on all three backends: tendons through via points embedded
  in the elements, potential `T L` (implicit in the reference, explicit forces
  with a stability check on MuJoCo/SOFA), YAML/Python/ROS control.
- **Check:** the cable-bent cantilever. The reference matches Euler–Bernoulli
  within 0.3% up to 0.27 rad, with the Neo-Hookean tangent under the cable's
  pre-compression.
- **Multi-material bodies** (`body.zones`: elements with their own material, e.g.
  a strain-limiting layer): any law in the reference, per-zone force fields on
  SOFA, per-element material data on MuJoCo. Check: a two-material
  cable beam against composite Euler–Bernoulli (reference within 1.4% at small
  load).
- **PneuNet finger** (`pneunet` shape: chambers, channel, conforming
  strain-limiting layer). MuJoCo and SOFA agree to 0.1% and bend 24% less than
  the reference at 8 kPa. A two-card transfer (body and layer) brings MuJoCo
  to 2.9%, holding at 11 kPa (3.2%) and on a 4-chamber finger (4.3%).
- **Bodies from files** (`mesh_file`: STEP/IGES/BREP, STL/OBJ, .msh/VTK/Medit,
  .geo through gmsh). Physical groups become regions and zones, closed inner
  shells become `cavity`, and boxes name faces. Several CAD solids are bonded
  into one body. `softsim export --cad` writes the parametric shapes as STEP.
  The PneuNet read back from its own STEP is the same body (volumes exact) and
  bends within 0.5% on MuJoCo.
- **Zones with their own density** on all backends. Check: a cantilever with a
  heavy outer half sags as Euler–Bernoulli says (reference within 0.15%).
- **Stroke-controlled cables** (`control: stroke`: a servo reeling in a tendon
  of given stiffness, slack when short; implicit in the reference, explicit
  with a stability check on MuJoCo/SOFA). Check: the reference within 0.8% of
  Euler–Bernoulli with the beam's bending and axial compliance in series with
  the tendon.
- **Region forces** (`type: force`: a dead load spread over a surface region by
  area, consistent on the reference's quadratic faces). Check: a tip-loaded
  cantilever converges to beam theory.
- **Moving mount** (`body.mount_path`: the fixed regions follow a minimum-jerk
  translation), simulated in the mount's frame on every backend. Check: the
  reference agrees with a world-frame run with prescribed base motion to 0.03%
  (second order in the step).
- **Prescribed region motion** (`body.prescribed`: any surface region follows a
  minimum-jerk path per axis, so a roller or a grip; the reaction force is an
  output on every backend). Check: a roller tension test. The reference
  gives the exact homogeneous stress (1e-7 at 10%, 1e-5 at 30% stretch), and
  force control with the reported reaction stretches the bar by the prescribed
  amount on all three backends; a fast pull checks the reaction against the
  body's momentum.
- **Rotating mount** (`body.mount_rotation`: the fixed regions turn about an axis,
  minimum-jerk waypoints, composing with the translation). MuJoCo and SOFA
  simulate in the mount's frame with rotated gravity and the Euler, centrifugal
  and Coriolis forces; the reference runs in the world with the base nodes
  driven along the rotation (trapezoidal rule). Checks: the terms against
  numerical differentiation of a free particle, the applied forces against
  `m a`, the rotating rod's centrifugal stretch (reference +2.8%, MuJoCo -1.6%,
  SOFA -2.3%), the Euler lag at the start, and a spinning round beam that must
  move as a still one (reference 6.4%; the linear-tet backends 17-21%, from the
  mesh's anisotropy and implicit Euler, with each broken force 48-355%).
- **Cable anchor spread over a face** (`anchor: <region>`: the end force becomes the
  linear traction with the same resultant and moment, on all backends). Check:
  the reference is within 0.5% of Euler-Bernoulli at the tip (a point anchor
  2.6% low), and under stroke control within 0.7% with no stiff cap (a point
  anchor -19%).
- **Zones with their own damping** (reference only: a Prony series, or none, per
  zone; MuJoCo and SOFA have one damping per body and refuse). Check: the plate
  force of a body with a quarter-section zone of another series is the
  area-weighted sum of each series' exact history.
- **Cable finger** (`cable_finger` shape: segments cut by slots, a tendon through
  the web, `path: geometry`). Check: exact mesh volume, one segment is
  Euler-Bernoulli (-0.9%), four segments bend 2.54 times as far, mesh
  converged to 0.5%; a plane-section beam model (1.55 times) is a lower bound.
- **Analytic checks:** the closed-end thick tube (Lame, free extension) on the
  reference converges at second order (radial -10% to -0.5% from 4 to 1.3 mm;
  axial strain +0.3% at 1.8 mm); SOFA's Rayleigh alpha against a free fall with
  linear drag (the implicit-Euler recurrence to 1e-12).
- **Closed-loop control from Python:** `Simulation.run(controller=...)`
  (a PI loop holding a cable beam's tip angle is a test).

Still to do:
- ~~**More actuators:** a constant-rate rotating mount~~ done 2026-11-11: waypoints `[time, angle, rate]`
  ([actuation.md](actuation.md#rotating-mount)).
- **Backend mappings (optional, for speed or stability):** tried and not shipped.
  SoftRobots' `CableConstraint` (plugin present in the SOFA v26.06 install; load
  it with `RequiredPlugin SoftRobots`, via points as a `BarycentricMapping`
  child, `pullPoint` a little behind the first point) in `valueType="force"`
  reproduces the explicit cable exactly: the cable cantilever's tip angle at
  0.7 N (a step, 2 mm, 1 ms) agreed to 1e-14 at steps 50, 100, 200 and 500
  (0.2837, 0.1523, 0.2471, 0.2861 rad). The solver's λ is an impulse, so the
  value must be force × dt (a force of 0.01 N without the factor gave 6 m/s).
  A force-type constraint is still an applied tension at the step's start,
  so it has no stability advantage over `ConstantForceField`; the one that
  would is the displacement type (a rigid tendon, `stiffness → ∞`), which is a
  different model from the finite-stiffness tendon here, and SOFA-only. Not
  pursued: it needs a rigid-tendon option in the config first.
  `SurfacePressureConstraint` was not tried. MuJoCo spatial tendons attach to
  sites on bodies, not to barycentric points inside flex elements, so the
  cable's via points would have to be vertices.

## M7: pneumatic finger (in progress; blocked on a data decision)

PneuNet scenario (pressure → bending angle, tip force, cavity volume), public
datasets registered with DOI and licence, and a **generic experiment API** so
calibration and plots work for any scenario. Yeoh in SOFA (custom force field)
or refused.

Done:
- **PneuNet scenario**, backends compared and MuJoCo transferred (above).
- **Generic experiment API:** `softsim fit` (any observable column of any
  scenario against a registered measured table, curve or series;
  `docs/experiments.md`). Checked by a twin experiment: it recovers a hidden
  modulus to 0.4% (clean) and 3.5% (1% noise); four code mutants are caught.
- **Yeoh in SOFA: refused**, with the `softsim material project` hint.
- **Finer meshes on the GPU:** the element kernels run in chunks (a 100k-element
  launch asked the 12 GB card for 4.8 GB and failed); the reference PneuNet now
  runs to 0.7 mm (152k tetrahedra, 25 min for four loads) and converges to
  within about 0.4% per step; the 1.5 mm reference is 3.6% under-bent
  (`docs/actuation.md`).

Open: real data. No public pressure–angle dataset with a geometry and
material I can read was usable; the options and my recommendation are in
[decisions.md](decisions.md), entry 3.

## M8: cable finger, soft arm and rods

Done on the reference: the cable-finger shape and scenario (`cable_finger`,
`docs/actuation.md`) and a two-segment cable arm with stiff disks
(`cable_arm.yaml`, within 2% of piecewise Euler–Bernoulli). Not done: a rod
(Cosserat) backend for slender bodies (SOFA BeamAdapter, PyElastica or our
own): chosen by the recommendation, SOFA BeamAdapter ([decisions.md](decisions.md), entry 4).
**Done: `--backend rod`** ([backends/rod.md](backends/rod.md)): a slender clamped body of any scenario
is reduced to beam frames (section stiffnesses from the tets with material zones exact; the mesh follows the frames),
cables and region forces carried by virtual work. Verified on the tip load (0.8641 mm vs E–B 0.864),
cable cantilever, 64° large rotation (arc of the compressed beam), layered section and the S of the
two-segment arm (linear E–B within 0.7%; the 3D reference is 8% lower from the compression stiffening
`1 + 2N/EA`, which a materially linear rod cannot have); 16 tests, 6 mutants caught, a figure.
Open: alpha damping is inert on BeamAdapter's mass (refused), the explicit cable step limit, and
rods that are not columns along z (curved or branched), PneuNet-like bodies being out of scope.

## M9: soft gripper and grasping

N-finger gripper on a rigid palm. Grasp metrics (success, contact patch,
forces, slip, pull-out). Contact quality work, including the M4 SOFA
artefacts, finger self-collision (done for MuJoCo and SOFA, one geometry) and a friction benchmark. A grasp benchmark
grid.

Started 2026-11-06: two-finger gripper on the reference, a fast finger fitted to it, a local live page and a grasp map
([gripper.md](gripper.md)). Two finite-element grasps with Coulomb friction (10 g held, 30 g dropped) agree with the fast
finger and play in the page's Truth tab. The finite-element finger's tip force is about 5.5 times below the measurement
with its tip sliding on the gauge, and about twice above with its tip stuck to it: the unrecorded contact explains the
gap, so the force cannot test the model (2026-11-11, [actuation.md](actuation.md#tip-force-a-test-it-fails)).

2026-11-11: **friction benchmark** (`scenarios/actuation/sliding_block.yaml`, `benchmarks/sliding_block.yaml` and
`sticking_block.yaml`, `tests/test_friction.py`). A silicone block pushed across the ground against exact Coulomb
sliding: all three backends within 0.8-0.94%. Below the limit, MuJoCo creeps at 0.13 mm/s (soft constraints) and the
reference at a rate set by its time step (its friction smoothing is per step; [decisions.md](decisions.md), entry 6).

2026-11-11: **self-collision** (`body.self_collision`, `tests/test_self_collision.py`). A finger pulled until its slot
closes: with self-collision on, the slot's walls stay apart (MuJoCo +0.19 mm, SOFA +0.10 mm); off, they pass through by
1.3 mm. fem and rod refuse it. Verified on one geometry only ([actuation.md](actuation.md#a-body-that-touches-itself)).

## M10: magnetic soft robots (MRE)

Soft-magnetic elastomer first, because that is what isotropic MRE fingers are: carbonyl-iron particles with no
remanence, magnetised by the applied field. The energy is linear-magnetic with axes that follow the deformation, in a
given field (uniform plus a point dipole); the field is not solved for. Then a permanent magnet's field, an MRE finger
against a measured force, and a magnetic gripper. A hard-magnetic model (fixed magnetisation) follows if a design needs it.

Started 2026-11-08: `softsim/fem/magnetic.py` on the reference, checked against a turned body's energy, the pull up a
dipole's gradient and a cantilever's closed form up to buckling (`tests/test_magnetic.py`).

2026-11-09 ([magnetic.md](magnetic.md)):

- **Config.** `magneto_active` cards take a susceptibility or an iron fraction, with block demagnetising factors.
  A scene-level `magnetic_field:` holds a uniform field and ring or disc magnets, which can move along paths and can be
  solid.
- **Field.** The magnets' fields are exact, checked against current loops to 1e-7.
- **Checks.** A bar built from the config follows its closed form to 2.2%. A mutation sweep tries 32 slips.
- **Against Bernat et al.'s gripper** (arXiv:2407.13477), digitised from the paper's vector figures:
  - The rolling torque is flat, as in their ANSYS. The point where it falls to zero suggests a strip about 2 mm thick.
  - With the susceptibility from the recipe alone, the torque is 58-66% of theirs.
  - The 0.7 N finger force is not checked yet: the paper does not give the contact point.
- **Solid magnets.** A magnet can also be a rigid cylinder for contact. Each magnet has a chamfer (`edge`, 0.2 mm) that
  softens its field's singular rims and rounds its contact surface: the singular rims had drawn a strip into the magnet's
  corner.
- **Fixed: H inside a magnet.** It is now `B / mu0 - M`; taking `B / mu0` made a false well under the rim.
- **Finger force.**
  - Energy balance for a free end peeled along the rim's tangent gives 0.24-0.28 N on our torque and 0.44 N on
    ANSYS's, against 0.66-0.83 N measured.
  - A clamped strip pulled that way cannot peel. Finite elements show the force rising past 1.9 N.
  - So the measured force depends on the gauge's unrecorded contact point.
- **Not yet simulated:**
  - the 15 mm strip, whose overhanging flaps snap onto the magnet's poles;
  - the strip closing from a gap: it hits the magnet at metres per second.
- **Next:**
  - ~~saturation~~ done 2026-11-11: a tanh law from the card's `saturation_magnetization` (0 keeps the linear law),
    checked against the closed forms ([magnetic.md](magnetic.md));
  - the finger against a gauge, once the set-up is known;
  - ~~a hard-magnetic model~~ done 2026-11-11: remanent magnetisation `-mu0 (F M_r) . H` (Zhao et al. 2019),
    checked against closed forms ([magnetic.md](magnetic.md)).
  - ~~magnetic zones~~ done 2026-11-11: zones carry their own `magneto_active` card (hard, soft or none), the energy
    assembled per element; two zones of opposite remanence cancel and a non-magnetic one halves the energy, to 1e-12;
    a zone with induced magnetisation needs explicit demagnetising factors (`block` is refused);
  - ~~a hard-magnetic gripper on the reference~~ done 2026-11-11: a U-shaped body with two oppositely magnetised
    arms closes in a uniform field (`scenarios/magnetic/hard_magnetic_gripper.yaml`); the arm's bending is the
    cantilever's under a distributed couple to 1.4% at 1 mm (0.5% at 0.75 mm), the tip 17-18% more because the palm
    rotates at the root. Not yet: gripping an object (a GPU-machine job, [magnetic.md](magnetic.md)).

## M11: design and control toolkit

Quasi-static inverse control (optimisation; SoftRobots.Inverse fast path on
SOFA; the optimisation part is done: `softsim inverse`, see [control.md](control.md)), reduced-order and Koopman surrogates for real-time ROS 2 control,
design optimisation over parametric geometry (`softsim optimize`), and
sim-to-real identification from short recorded motions.

- **Design optimisation (done):** `softsim optimize <job.yaml>` ([design.md](design.md)). Nelder-Mead over
  dotted design parameters (units, bounds, log, integer, tied keys), objective maximise/minimise/target on any
  summary metric (or a ratio), constraints as penalties with the best chosen among feasible evaluations and
  infeasibility reported, resumable evaluation log, parallel initial simplex. Checked on the cable cantilever
  against a known maximum (the bound) and a bisection reference; the MuJoCo-locking bias of the optimum and
  the `fem` check are documented. Open in M11: inverse control, surrogates, sim-to-real identification.

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

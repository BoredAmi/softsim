# Changelog

## 0.1.0 (unreleased)

The first release: a validated core, three robot families and the reference solver they are checked against.
See [docs/roadmap.md](docs/roadmap.md) for the details of each milestone.

- **Core (M1):** YAML configs with units and strict schemas, `--set` overrides, material presets, soft and rigid
  geometry, tetrahedral meshing, a backend-neutral scene, capability flags with refusals, pre-run physics
  checks, sweeps, CLI.
- **Backends:** MuJoCo flex (M1), SOFA v26.06 (M4), `fem`, the locking-free 3D reference (quadratic
  tetrahedra, P2/P0; M5), and `rod`, slender bodies as SOFA BeamAdapter beams (M8).
- **Self-collision (M9):** `body.self_collision` for the actuation scenario (MuJoCo flex, SOFA); fem and rod refuse.
- **ROS 2 Humble layer (M2):** a generic node, interfaces, launch and RViz.
- **Experiments (M3, M5):** drop-test video tracking (including tilt), calibration over any observable,
  `softsim fit` against any measured table, leave-one-out validation.
- **Materials (M5):** physical and effective cards with validity, uncertainty and references; Gent and
  Arruda-Boyce; homogeneous tests; `fit-material` on public data (Marechal 2021, Roels 2025); 25 fitted silicone
  cards; Prony viscoelasticity on every backend; `softsim transfer` with error bars; sensitivity and sampling.
- **Actuation (M6):** pressure, cable (tension or stroke), and region-force actuators; material zones with their
  own density and damping; bodies from STEP/STL/OBJ/msh; moving and rotating mounts; prescribed region
  motion; closed-loop control from Python.
- **Robots (M7-M10):** PneuNet and the Wang et al. ribbed finger (reference within 0.8 deg RMS of the
  measurements), cable finger and arm, a two-finger gripper with finite-element grasps and a live page,
  soft-magnetic elastomers (MRE) on the reference.
- **Friction:** a sliding-block benchmark against exact Coulomb sliding (all backends within 0.8-0.94%) and its sticking
  case; the reference's friction smoothing is now a slip speed (`contact.slip_speed`, default 1e-5 m/s), so its stick creep
  no longer depends on the time step (docs/decisions.md, entry 6). This changes finite-element grasps; `null` restores
  the old behaviour.
- **SOFA contact:** its "58% energy loss" in frictionless impacts was a friction mismatch (one coefficient for every
  pair); compared like for like it loses 3.4% more than the reference. Runs now warn (`single-friction`) when a SOFA
  scene's coefficients differ.
- **Mounts:** a rotating mount can turn at a constant rate (waypoints `[time, angle, rate]`).
- **Magnetic:** saturation of the induced magnetisation (a tanh law from the card's `saturation_magnetization`) and
  hard-magnetic particles (`remanent_magnetization`, Zhao et al. 2019), both on the reference;
  magnetic zones (each element with its own card) and a hard-magnetic U gripper whose oppositely magnetised arms close
  in a uniform field.
- **Design and control (M11, first pieces):** `softsim optimize` (design optimisation over parametric geometry, a
  resumable evaluation log, constraints with an honest infeasible flag) and `softsim inverse` (quasi-static inverse
  control: the actuator values that reach target outputs), both checked on MuJoCo; values are the backend's own.
- **Runs:** a run that fails partway saves the samples it computed (`summary.yaml` → `metadata.incomplete`).
- **Release (M13):** `softsim bench` and the benchmark table, CI, a docs site with tutorials, a test that every command
  in the README and tutorials parses and its config validates, `softsim --version`.

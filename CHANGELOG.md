# Changelog

## 0.1.0 (2026-11-13)

The first release: a validated core, three robot families and the reference solver they are checked against.
See [docs/roadmap.md](docs/roadmap.md) for the details of each milestone.

- **Core (M1):** YAML configs with units and strict schemas, `--set` overrides, material presets, soft and rigid
  geometry, tetrahedral meshing, a backend-neutral scene, capability flags with refusals, pre-run physics
  checks, sweeps, CLI.
- **SOFA inverse (`--method sofa-qp`):** angle targets (`bend_angle`, `chord_angle`) with any actuator, alone or with position targets (Broyden iteration on the effector goals, `angle_direction`); a pressure and a cable together (`scenarios/inflation/bending_cable.yaml`); stroke-controlled cables (solved as tensions, stroke read back, stroke bounds iterated). Round trips recover the actuation to 0.5% or better.
- **SOFA cable (fix):** tension-controlled cables are springs with per-step rest lengths, so their tangent stiffness is implicit; the two-cable arm no longer swings in a +-0.5 mm limit cycle on SOFA and the repeated inverse solve converges.
- **Backends:** MuJoCo flex (M1), SOFA v26.06 (M4), `fem`, the locking-free 3D reference (quadratic
  tetrahedra, P2/P0; M5), and `rod`, slender bodies as SOFA BeamAdapter beams (M8).
- **Self-collision (M9):** `body.self_collision` for the actuation scenario (MuJoCo flex, SOFA); fem and rod refuse.
- **ROS 2 Humble layer (M2):** a generic node, interfaces, launch and RViz.
- **Experiments (M3, M5):** drop-test video tracking (including tilt), calibration over any observable,
  `softsim fit` against any measured table, leave-one-out validation.
- **Materials (M5):** physical and effective cards with validity, uncertainty and references; Gent and
  Arruda-Boyce; homogeneous tests; `fit-material` on public data (Marechal 2021, Roels 2025); 25 fitted silicone
  cards; Prony viscoelasticity on every backend; `softsim transfer` with error bars; sensitivity and sampling.
- **Gymnasium environments (M12 groundwork):** `softsim.envs` (extra `rl`), `softsim/CableBeamReach-v0` and
  `softsim/CableArmReach-v0`, YAML-defined reach tasks with domain randomisation over the cards' uncertainty; `softsim/PneuNetBend-v0`
  (pressure-driven bend angle), `softsim/ReducedGripperLift-v0` (grasp and lift on the reduced gripper, randomised
  object mass and friction) and `tools/train_baseline.py` (numpy CEM baseline vs random actions).
- **Actuation (M6):** pressure, cable (tension or stroke), and region-force actuators; material zones with their
  own density and damping; bodies from STEP/STL/OBJ/msh; moving and rotating mounts; prescribed region
  motion; closed-loop control from Python.
- **Robots (M7-M10):** PneuNet and the Wang et al. ribbed finger (reference within 0.8 deg RMS of the
  measurements), cable finger and arm, a two-finger gripper with finite-element grasps and a live page,
  soft-magnetic elastomers (MRE) on the reference.
- **Benchmarks, full tier:** `docs/benchmarks.md` now has the full tier (hard-magnetic cantilever and the Wang finger: the reference is 0.76 deg RMS from the measurement, MuJoCo and SOFA 18-19 deg, listed as known failures). The 10 g finite-element grasp was re-run with the default friction on the 2 mm mesh (unchanged) and on the validated 1.5 mm finger with teeth (grip while holding 13% higher); the 30 g one stops converging at t = 1.04 s.
- **Drop comparisons:** calibrations made before the measured landing tilt was applied (28-29 Oct) are re-simulated
  without it (`tilt: false` in the calibration file); with the tilt the SOFA 500 g fits blew up and were missing from
  the figures. The SOFA Ogden fit's 500 g NRMSE is 79% without the rim artefact (it was reported as 58%).
- **SOFA contact:** the alarm distance follows the fastest vertex's travel per step (rim flare of the 500 g drop: a vertex
  went 2.83 mm through the ground, now 0.006 mm; the restitution table is unchanged, +12% CPU time in that run;
  `SOFTSIM_SOFA_ADAPTIVE_ALARM=0` restores the fixed one). The "frictionless plate 7% stiff" artefact was friction:
  SOFA's single coefficient (0.1 gives about -7%, 0.8 -33%), at 0 the plate agrees with no contact to 0.6%.
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
  `softsim inverse --method sofa-qp` is the SOFA fast path with SoftRobots.Inverse (`QPInverseProblemSolver`,
  `CableActuator`, `PositionEffector`): same tensions as the repeated solve to 1%, 7-11 times faster on the cantilever,
  and it solves the two-cable arm where the repeated solve on SOFA does not settle. Pressure actuators too
  (`SurfacePressureActuator` on the cavity, `jobs/inverse_bending_sofa.yaml`): 4 kPa recovered to 0.03% on the bending
  tube (7 times faster than the repeated solve), 5 kPa to 0.05% on the PneuNet.
- **Surrogates (M11):** `softsim surrogate` fits a polynomial or RBF model of the equilibrium map from seeded samples
  (resumable log, held-out validation, JSON evaluable from numpy) and `softsim inverse --method surrogate` solves on it
  in milliseconds with an optional verifying simulation; optional DMD-with-control dynamic model. MuJoCo arm: the known
  tensions found to 0.01%, ~1500 times faster per solve than the repeated solve.
- **Closed-loop control (M11):** `softsim.control.controller.Controller` (surrogate feedforward, damped-least-squares
  integral feedback with anti-windup, optional MPC-lite on the DMD model) and a ROS 2 `controller_node`; with a
  stiffness 20 % off, feedback removes the 17 to 33 % (cantilever) and 1 to 3 mm (arm) feedforward error on MuJoCo.
- **Runs:** a run that fails partway saves the samples it computed (`summary.yaml` → `metadata.incomplete`).
- **Release (M13):** `softsim bench` and the benchmark table, CI, a docs site with tutorials, a test that every command
  in the README and tutorials parses and its config validates, `softsim --version`.

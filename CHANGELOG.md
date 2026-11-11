# Changelog

## 0.1.0 (unreleased)

The first release: a validated core, three robot families and the reference solver they are checked against.
See [docs/roadmap.md](docs/roadmap.md) for the details of each milestone.

- **Core (M1):** YAML configs with units and strict schemas, `--set` overrides, material presets, soft and rigid
  geometry, tetrahedral meshing, a backend-neutral scene, capability flags with refusals, pre-run physics
  checks, sweeps, CLI.
- **Backends:** MuJoCo flex (M1), SOFA v26.06 (M4), `fem`, the locking-free 3D reference (quadratic
  tetrahedra, P2/P0; M5), and `rod`, slender bodies as SOFA BeamAdapter beams (M8).
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
- **Release (M13):** `softsim bench` and the benchmark table, CI, a docs site.

# MuJoCo backend: physics notes and limits

Verified against **MuJoCo 3.17.0**, the pinned range (`mujoco>=3.17,<3.18`).
The flex API changes every release, so read the changelog before bumping.

## What is used

| softsim | MuJoCo |
| --- | --- |
| Soft body | `flexcomp type="direct" dim="3"` with our tetrahedral mesh; one 3-dof body per vertex |
| `stvk` | default flex elasticity: Saint Venant–Kirchhoff, linear tetrahedra |
| `neo_hookean` | Stable Neo-Hookean, `MjsFlex.elastic3d = 1` (experimental since 3.15, **spec-only**, not in MJCF) |
| `rayleigh.beta`, `kelvin_voigt` | flex `damping` (stiffness-proportional; Kelvin–Voigt maps to beta = eta / E) |
| `base: bonded` | `<pin>` on the bottom-face vertices |
| Contact | soft constraints `solref = (timeconst, dampratio)`, `solimp`; friction combines by max |
| Integrator | `discrete` (new in 3.13, Sept 2026, experimental): implicit in position, so flex stiffness has no explicit stability limit |

**Not supported, and refused:** small-strain linear elasticity, Mooney-Rivlin,
Yeoh, Ogden, Prony viscoelasticity, mass-proportional damping, magneto-active
coupling, negative Poisson's ratio, any integrator other than `discrete`, and
the PGS solver.

## Measured behaviour (drop test, this repo)

- **Material mapping is correct.** Under a frictionless 3% uniaxial
  compression, the result matches δ = m·g·H/(E·A) within 0.7–1.8% for both
  `neo_hookean` and `stvk` (`tests/test_drop_test.py::test_static_uniaxial_compression`).
- **The timestep is not the accuracy bottleneck.** With the contact model held
  fixed, peak indentation at dt = 0.1 ms and 0.05 ms differs by 0.2–0.7%. Use
  dt ≤ 0.1 ms for impacts; 0.2 ms is within about 3%.
- **Contact compliance matters as much as the material.** Peak indentation for
  20 g changes by about 10% between `timeconst` 0.2 ms and 0.4 ms. Penetration
  scales with roughly v·timeconst. Keep `timeconst` fixed in configs (scenario
  default 0.2 ms) and treat it as a model parameter; `auto` (= 2·dt) couples it
  to the timestep.
- **Results are not mesh-converged.** Linear tetrahedra at ν = 0.45 lock: peak
  indentation grows 10–15% per refinement step (5 → 3.5 → 2.5 mm). A calibrated
  modulus is therefore an effective value for one mesh resolution; fitted
  presets must record their mesh in `provenance`. Pushing ν toward Ecoflex's
  real ~0.49 makes this worse.
- **StVK collapses in compression.** With 200 g and 500 g, StVK lets the weight
  crush through the full 17 mm. Neo-Hookean stays physical (8 and 13 mm in the
  Step 0 probe). Neo-Hookean is the default.
- **Neo-Hookean still underestimates silicone stiffening at large strain.**
  Expect the 200–500 g cases to fit worse than 20–50 g; Yeoh or Ogden in a FEM
  backend (SOFA, M4) is the proper fix.
- **Damping is a single Rayleigh beta**, a crude proxy for viscoelasticity: the
  loss factor grows linearly with frequency. Beta and E trade off strongly over
  a millisecond impact. Fit both across several impact energies at once, and
  read the leftover misfit as a model limitation.
- **Peak contact force is not converged.** It is a first-contact spike on tiny
  vertex masses and changes with dt (70–140 N for 20 g). Use indentation,
  timing, rebound and ground force for calibration.
- **Specimen lift-off in the simulation.** In the free-base configuration the
  specimen's elastic recoil can launch it off the rigid table after the weight
  separates (1 mm at 20 g up to 16 mm at 500 g with the uncalibrated preset).
  It is reported as `specimen_lift_off`. *Correction:* I earlier said the 500 g
  recording confirms this. M3 tracking showed that the table under the specimen
  in the recordings is not rigid and moves by several mm from about 20 ms after
  impact, so the apparent lift in the video is mostly support motion. See
  [experiments.md](../experiments.md).
- **Solver warnings.** At large dt on finer meshes, the inner M+K solve can run
  out of CG iterations (`mjWARN_INERTIA`). softsim reports this as a
  `backend-warnings` finding; raise `simulation.iterations` or lower dt.
- **Divergence.** MuJoCo resets the state itself on NaN; softsim detects this
  through the bad-qacc/qpos/qvel warning counters and raises `StabilityError`.

## Performance

762-vertex cylinder (3.5 mm mesh), dt = 0.1 ms, one CPU core: about 0.005×
real time, so a 250 ms drop takes about 50 s. The 5 mm mesh is about 3× faster.
Sweeps run in parallel processes (`--jobs`). No GPU is used or needed.

# MuJoCo backend: physics notes and limits

Verified against **MuJoCo 3.17.0**, the pinned range (`mujoco>=3.17,<3.18`).
The flex API changes every release, so read the changelog before bumping.

## What is used

| softsim | MuJoCo |
| --- | --- |
| Soft body | `flexcomp type="direct" dim="3"` with our tetrahedral mesh; one 3-dof body per vertex |
| `stvk` | default flex elasticity: Saint Venant–Kirchhoff, linear tetrahedra |
| `stable_neo_hookean` | Stable Neo-Hookean (Smith et al. 2018), `MjsFlex.elastic3d = 1` (experimental since 3.15, **spec-only**, not in MJCF). Classic `neo_hookean` (ln J energy) is not available and is refused. |
| `rayleigh.beta`, `kelvin_voigt` | flex `damping` (stiffness-proportional; Kelvin–Voigt maps to beta = eta / E) |
| `prony` (on `stable_neo_hookean`) | the flex carries the long-term law (shear modulus × (1 − Σg), bulk unchanged); the Maxwell-branch overstress of every tetrahedron (`softsim/materials/viscoelastic.py`, needs the `fem` extra) is applied as explicit vertex forces (`xfrc_applied`) before every step |
| `base: bonded` | `<pin>` on the bottom-face vertices |
| Contact | soft constraints `solref = (timeconst, dampratio)`, `solimp`; friction combines by max |
| `adhesion` (material, ground) | external forces (`xfrc_applied`) on surface vertices and rigid bodies, recomputed before every step from analytic signed distances (`softsim/contact/adhesion.py`) |
| `mount: slider_z` | `slide` joint along z with `stiffness`, `damping`, `springref = 0` (rest = initial pose) |
| Integrator | `discrete` (new in 3.13, Sept 2026, experimental): implicit in position, so flex stiffness has no explicit stability limit |

**Not supported, and refused:** small-strain linear elasticity, Mooney-Rivlin,
Yeoh, Ogden, Gent, Arruda–Boyce, mass-proportional damping, magneto-active
coupling, negative Poisson's ratio, any integrator other than `discrete`, and
the PGS solver.

## Measured behaviour (drop test, this repo)

- **Material mapping is correct.** Under a frictionless 3% uniaxial
  compression, the result matches δ = m·g·H/(E·A) within 0.7–1.8% for both
  `stable_neo_hookean` and `stvk` (`tests/test_drop_test.py::test_static_uniaxial_compression`).
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
  crush through the full 17 mm. Stable Neo-Hookean stays physical (8 and 13 mm in the
  Step 0 probe) and is the default. *Naming:* this was called `neo_hookean`
  until M4; it is the stable variant, a different energy at large strain from
  classic Neo-Hookean, so it now has its own model name.
- **The 500 g misfit is not a MuJoCo limitation.** Expect the 200–500 g
  cases to fit worse than 20–50 g. M4 tested this on SOFA: an Ogden fit
  doesn't reproduce 500 g either, and actually prefers *less* large-strain
  stiffening than Neo-Hookean. Compressibility (ν = 0.45) and missing rate
  dependence matter more; see [experiments.md](../experiments.md#backends-and-material-models-m4).
- **Native damping is a single Rayleigh beta**, a crude proxy for
  viscoelasticity: the loss factor grows linearly with frequency. Beta and E
  trade off strongly over a millisecond impact. Prony cards avoid this.
- **Prony viscoelasticity is explicit.** Its overstress lags one step, so its
  stiffness Σg·μ0 must be resolved by the time step (`softsim check` reports
  ω·dt). This is no issue at drop-test steps: the limit is about 0.25 ms on a
  3 mm mesh. The forces match the homogeneous reference exactly under uniform
  deformation (`tests/test_viscoelastic.py`). A 50 g drop with g = 0.6 at
  τ = 2 ms peaks between the instantaneous and the long-term material, at
  5.38 mm on MuJoCo and 5.39 mm on SOFA.
- **Friction changes the impact a lot.** 50 g drop, no material damping, 5 mm
  mesh: peak 4.89, 5.72 and 6.80 mm at sliding friction 1, 0.3 and 0. The
  bulging specimen slides against the weight and the table, and that matters
  as much as the modulus.
- **An undamped impact loses energy that a converged solver keeps.** See
  [Energy and contact fidelity](#energy-and-contact-fidelity) below. Fitted
  damping (β) is therefore an effective value on top of MuJoCo's own losses.
- **Mesh stiffness against a converged reference:** for a static flat punch
  (small strain) the 5 mm mesh is 20–60% stiffer than the locking-free
  reference ([reference_fem.md](../reference_fem.md)).
- **Peak contact force is not converged.** It is a first-contact spike on tiny
  vertex masses and changes with dt (70–140 N for 20 g). Use indentation,
  timing, rebound and ground force for calibration.
- **Specimen lift-off.** In the free-base configuration the specimen's
  elastic recoil can launch it off the rigid table after the weight separates
  (1 mm at 20 g up to 16 mm at 500 g with the uncalibrated preset). It is
  reported as `specimen_lift_off`. The real specimens do hop, by 8–20 mm, but
  for a different reason: the tacky specimen is pulled up by the weight stuck
  to it, and drags the cloth under it along. Reproducing that needs adhesion
  and the compliant support (below). *Correction history:* M3 read the
  specimen's hop as table motion; that was wrong. See
  [experiments.md](../experiments.md#after-the-first-contact-tack-and-the-hopping-specimen).
- **Solver warnings.** At large dt on finer meshes, the inner M+K solve can run
  out of CG iterations (`mjWARN_INERTIA`). softsim reports this as a
  `backend-warnings` finding; raise `simulation.iterations` or lower dt.
- **Divergence.** MuJoCo resets the state itself on NaN; softsim detects this
  through the bad-qacc/qpos/qvel warning counters and raises `StabilityError`.

## Energy and contact fidelity

How much of an impact's energy MuJoCo loses that it shouldn't, measured with
purely elastic cards (no damping of any kind) against the locking-free 3D
reference (`tools/contact_restitution.py`, 2026-10-31).

**1. Flex elasticity is numerically damped like backward Euler.** A free axial
vibration of the bonded specimen (no contact, no damping, 6.25 ms period)
loses energy per cycle as predicted by a damping ratio **ζ ≈ ω·dt/2**:

| Timestep | 100 µs | 50 µs | 25 µs | 12.5 µs |
| --- | --- | --- | --- | --- |
| Energy lost per cycle | 52% | 30% | 15% | 6.5% |
| 1 − exp(−4πζ), ζ = ω·dt/2 | 47% | 27% | 15% | 7.6% |

For the mode you care about (angular frequency ω), pick dt so that ω·dt/2 is
well below the material damping you want to model.

**2. Impacts: friction slip costs most.** A centred 50 g weight hits the drop
specimen at 2.7 m/s. Restitution e = rebound speed / impact speed:

| Contact | Reference (fem) | MuJoCo, 5 mm mesh | MuJoCo, 3.5 / 2.5 mm |
| --- | --- | --- | --- |
| Frictionless | 0.869 | 0.791 | 0.839 / 0.838 |
| Sticking, μ = 1 | 0.905 | 0.71–0.75 | – |

- **Contact settings don't matter.** e stays at 0.71–0.75 over damping ratio
  1 to 0.05, time constant 0.1–0.2 ms and timestep 50 to 6 µs.
- **Without friction MuJoCo is within 3.5%** at meshes of 3.5 mm or finer.
  At 5 mm, each surface vertex's impact on the weight is plastic, and that
  costs about 6% more.
- **With friction it loses about twice as much.** The reference sticks,
  which does no work. MuJoCo's soft friction constraint slips while the
  specimen bulges under the weight, and the slip dissipates.
- **The best available mitigation:** `contact.cone: elliptic` with
  `contact.impratio: 10` (10–100 are equivalent). This raises e from 0.748 to
  0.78 at 5 mm. MuJoCo's no-slip solver would remove the slip, but it is not
  supported with flex under the discrete integrator. The defaults stay
  pyramidal and 1, which is what every fit in this repository used.

**3. Locking:** at ν = 0.45 the 5 mm mesh's peak compression is 24% below the
reference's (4.9 against 6.5 mm with sticking).

These losses are why MuJoCo's fitted cards are `effective`. Transfer a
physical card ([materials.md](../materials.md#from-a-physical-card-to-a-backend-transfer))
rather than using its values directly.

## Adhesion (tack)

- **Model:** an exponential cohesive law in the normal direction,
  t(g) = σ (g/δ) e^(1 − g/δ), acting on the surface area that faces the
  counterpart. A bond forms on contact and breaks for good beyond 6δ. Work of
  separation is e·σ·δ per area. See `softsim.materials.Adhesion`.
- **Explicit.** The forces are computed from the state at the start of each
  step, so the adhesive stiffness e·σ/δ per vertex area must be resolvable:
  `softsim check` reports `adhesion-stiff` (ω·dt > 0.3) and refuses
  `adhesion-unstable` (ω·dt > 1). Lower the timestep or raise `range`.
- **No damping of its own.** A body held only by adhesion oscillates on the
  bond undamped (Rayleigh β damps strain, not rigid motion). The energy still
  stored in a bond when it breaks is the only loss. Rate-dependent tack is not
  modelled.
- **Verified** (`tests/test_adhesion.py`): a stiff block held to the ground,
  and a disc held to a block, under slowly reversed gravity settle at the
  analytic opening of the traction law within 2%. Above the bond capacity
  the block tears off. The reported contact force equals the load.
- **Mesh dependence.** Tractions are lumped on surface vertices, so the bonded
  area is resolved at the mesh size (a 12 mm disc on a 5 mm mesh bonds about
  10% too much area). More importantly, *peeling* is only mesh-converged when
  the cohesive zone, l ≈ E*·(e·σ·δ)/σ², spans several elements. For the
  drop-test fit (σ ≈ 14 kPa, δ = 0.3 mm, E* ≈ 73 kPa) that is about 4 mm,
  less than the 5 mm mesh. On 5 mm the bond acts all-or-nothing; on 3.5 mm
  it peels and lets the 20 g weight go. Fitted tack parameters are therefore
  effective values for the mesh they were fitted on, like the fitted moduli.

## Performance

762-vertex cylinder (3.5 mm mesh), dt = 0.1 ms, one CPU core: about 0.005×
real time, so a 250 ms drop takes about 50 s. The 5 mm mesh is about 3× faster.
Sweeps run in parallel processes (`--jobs`). No GPU is used or needed.

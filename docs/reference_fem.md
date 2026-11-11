# Reference FEM (locking-free, axisymmetric and 3D)

`softsim.fem` holds two small finite-element solvers that give
**mesh-converged** answers for the general backends (MuJoCo flex, SOFA) to be
checked against. Both of those use linear tetrahedra, which lock for nearly
incompressible materials, so their fitted moduli are "effective" values tied
to a mesh.

- `softsim.fem.axisym`: axisymmetric (cylinders, centred punches, uniaxial
  tests). Fast.
- `softsim.fem.solid3d`: 3D, any geometry, rigid bodies with 6 DOF. It is
  also available as the backend **`fem`** (`softsim run ... --backend fem`).
  It is slow, about 1 s per step on a 5 mm mesh on a laptop (see
  [Performance](#performance)), but exact where the others are not.

```bash
pip install -e ".[fem]"     # JAX (CPU, float64) for exact tangents
python -m pytest tests/test_fem_axisym.py
```

```python
from softsim.fem.drop import run_drop, FemSettings
ts, info = run_drop("scenarios/drop_test/cases/w050.yaml", ["simulation.duration=0.012"],
                    FemSettings(cells_r=16, cells_z=10))
```

## Design

- **Elements:** 6-node quadratic triangles in the r–z half-section with one
  dilatation per element (mean dilatation, P2/P0). No volumetric locking, up
  to incompressibility.
- **Materials:** isochoric/volumetric split, JAX-traceable laws: Neo-Hookean
  (also used for `stable_neo_hookean`), Mooney-Rivlin, Yeoh, Ogden. Prony
  viscoelasticity acts on the isochoric part (Abaqus convention). Volumetric
  energy κ/4 (J² − 1 − 2 ln J).
- **Solver:** each step minimises an incremental potential (elastic + Prony
  history + inertia + gravity + contact). Newton uses exact JAX tangents, an
  energy line search, a projected-Hessian fallback where the tangent is
  indefinite, and automatic step halving. The energy is infinite for any
  folded quadrature point (r–z fold or crossing the axis), so inverted
  states are never accepted.
- **Dynamics:** Newmark with γ = ½ + a, β = (1 + a)²/4. The drop uses a = 0.25:
  with penalty contact, smaller values gained energy when contact released.
  This costs about 5% of the energy per impact.
- **Contact:** a rigid flat punch (one vertical degree of freedom, the falling
  weight) and the ground, by penalty on boundary nodes with their exact annular
  tributary areas. Tangentially it is frictionless, sticking, or Coulomb
  (an incremental friction potential with the lagged normal pressure, C¹
  stick zone), with the same cohesive adhesion law as
  `softsim.materials.Adhesion`. Both the punch and the ground can touch the
  top, bottom *and* side surfaces: under full-face loads the side bulges
  and rolls down onto the table. An earlier version let it pass through,
  which made a 500 g drop far too soft. Contact is node-based because,
  with quadratic edges in axisymmetry, the node on the axis gets exactly zero
  load from a uniform pressure, and quadrature-point contact let the punch
  pierce it.

## 3D solver and the `fem` backend

The formulation is the axisymmetric one, in 3D:

- **Elements:** 10-node quadratic tetrahedra, built from any linear tetrahedral
  mesh by adding edge midpoints. There is one dilatation per element (P2/P0),
  so no locking.
- **Integration:** an 8-point rule for stiffness and 27 points for mass.
- **Tangents:** assembled from the 3×3 material tangents at the quadrature
  points. For invariant laws (Neo-Hookean, Mooney–Rivlin, Yeoh, Gent,
  Arruda–Boyce) they are in closed form and JAX only differentiates the scalar
  ψ(Ī₁, Ī₂). Ogden uses JAX derivatives of the full 3×3 energy.
- **Rigid bodies:**
  - `free` mounts have 6 DOF: the centre-of-mass displacement and one
    incremental rotation vector per step. The gyroscopic term is neglected,
    which is exact for bodies spinning about their axis of symmetry.
  - `slider_z` mounts are springs with dampers. Bodies can also be `fixed` or
    `prescribed`.
  - Shapes are unions of cylinders and boxes with rounded edges, as exact
    signed distances.
- **Contact:** node-based penalty with lumped areas. It is frictionless,
  sticking or Coulomb, with cohesive adhesion.
- **Viscoelasticity:** Prony runs inside the implicit solve. Besides dynamics
  there is a quasi-static viscoelastic mode for relaxation tests.
- **Linear solves:** MKL PARDISO when `pip install -e ".[fast]"` is
  installed, about 35× faster than SuperLU.

**The `fem` backend** maps a scene onto this solver:

- one soft body;
- every hyperelastic law and Prony. Ogden runs through matrix functions
  (tr C̄^(α/2) = tr exp((α/2) log C̄)), which are smooth where principal
  stretches coincide. Its tangents cost about 2 ms per quadrature point,
  roughly 100× the invariant laws, so project Ogden cards to Yeoh or Gent for
  long dynamic runs;
- free and spring-mounted rigid bodies, and the ground;
- friction: coefficients combine by maximum. 0 is frictionless, ≥ 1 is
  sticking (identical to μ = 1 statically, see below, and much faster for
  Newton), and anything in between is Coulomb. Coulomb is smoothed (IPC's mollifier) below a slip speed
  `contact.slip_speed` (default 1e-5 m/s), so a contact held below its limit creeps at a fraction of it, whatever the
  step (0.005 mm/s at 0.71 of the limit, `benchmarks/sticking_block.yaml`). Before 2026-11-11 the smoothing was 1 um
  per step (`slip_speed: null`), whose creep grows as the step shrinks (0.23 mm/s at 2 ms; [decisions.md](decisions.md),
  entry 6);
- adhesion from the material and the ground;
- magneto-active (MRE) material in the scene's magnetic field, with solid magnets as rigid cylinders. See
  [magnetic.md](magnetic.md).

Rayleigh and Kelvin–Voigt damping are refused: they are numerical damping,
so give the card a Prony series or no damping.

| Check (`tests/test_fem_3d.py`) | Result |
| --- | --- |
| Quadrature, shape functions, mesh volume and surface | exact |
| Closed-form tangents vs JAX `hessian`, every invariant law, with Prony history | within 1e-9 |
| 30% frictionless compression, Yeoh, between ground and a prescribed plate | force within 0.1%, lateral stretch within 1e-4 of the homogeneous solution |
| Quasi-static Prony ramp-and-hold | plate force within 0.2% of the homogeneous history |
| Rigid free flight and spin | exact |
| Centred 50 g drop (E = 100 kPa, ν = 0.4995, sticking) vs the axisymmetric solver | 5 mm mesh: 5.53 mm at 4.04 ms; axisymmetric (converged): 5.75 mm at 4.14 ms |

## Performance

`python tools/bench_fem.py` times one Newton iteration of the drop-test
specimen part by part (element tangents, energy, assembly, linear solve) and a
short 50 g drop. Run it on each machine to compare.

- **Tangents:** closed form for invariant laws, about 2× faster than
  differentiating the 3×3 energy twice. `jnp.linalg.det` was replaced by
  cofactor formulas, because JAX differentiates it through an LU
  factorisation.
- **Assembly:** the Hessian's sparsity pattern (free DOFs, upper triangle) is
  fixed when the solver is built, from every element, contact candidate and
  rigid block. Each iteration only scatters values into it.
- **Linear solve:** PARDISO's symbolic analysis (reordering) runs once per
  solver; each iteration only refactorises.
- **Compilation:** compiled kernels are kept in `~/.cache/softsim/jax`, so
  repeated runs on one mesh compile faster (9 → 6 s). Set
  `SOFTSIM_JAX_CACHE` to another directory, or to `0` to turn it off.

Together these took a short 8 mm drop from 75 s to 43 s on the laptop, with
results matching the earlier code to the Newton tolerance.

Profile of a 5 mm, 50 g drop on the laptop (8 cores, shared with two other
runs). A step takes about 4 Newton iterations and about 1.1 s.

| Part of a Newton iteration | Time |
| --- | --- |
| Element tangents and their contraction into 30×30 matrices (JAX, mostly one core) | ~110 ms (~45%) |
| PARDISO refactorisation (MKL, multi-threaded) | ~75 ms (~30%) |
| Scatter into the fixed pattern, line-search energies, contact, Python | the rest |
| Once per run: JIT compilation of the kernels | ~10 s |

`Solid3DSolver(..., quadrature=4)` uses a 4-point stiffness rule, the one in
Abaqus C3D10, instead of 8 points. In the drop it moved the peak by 0.5%. The
default stays 8.

**On a GPU:** `pip install -U "jax[cuda12]"` and run the benchmark again.
The element kernels run on the GPU unchanged. The linear solve stays on the
CPU (PARDISO). Expect the kernel share to shrink a lot and the solve to
dominate. `JAX_PLATFORMS=cpu` forces the CPU for comparison.

## Verified (`tests/test_fem_axisym.py`)

| Check | Result |
| --- | --- |
| Volume of revolution, tributary areas | exact (to 1e-12) |
| 30% frictionless compression, Neo-Hookean, κ/μ = 10³ and 10⁵ | force within 0.03% / 3·10⁻⁶ of incompressible theory; radius from volume conservation |
| 20% compression, Mooney-Rivlin, 3-term Yeoh, Ogden (α = 3) | within 0.2% of each law's exact uniaxial stress |
| Bonded disc, ν = 0.49999, 3 meshes | converges (2.7% then 1.05% change); E_c/E = 1.73, close to Gent's 1 + 2S² = 1.69. No locking |
| Ramp-and-hold relaxation, Prony g = 0.5, τ = 0.1 s | within 3% of the hereditary integral (the residue is real ringing) |
| Static punch with Coulomb friction | μ = 0 equals frictionless exactly; indentation falls monotonically with μ; μ = 10 equals stick |

**Friction matters as much as the modulus.** At 20 N (E = 100 kPa,
ν = 0.49) the static indentation is 3.91 mm frictionless, 3.08 mm at μ = 0.3,
2.56 mm at μ = 1 and 2.56 mm sticking. For a tacky silicone (μ ≈ 1),
sticking contact is an excellent approximation.

## Static punch: MuJoCo vs reference

27 mm flat punch on the 40 × 17 mm cylinder, E = 54.8 kPa, ν = 0.45, sticking
contact, static loads (MuJoCo: heavily damped, settled):

| Load | Reference (P2/P0, 16 × 10) | MuJoCo, 5 mm mesh | MuJoCo, 3.5 mm mesh |
| --- | --- | --- | --- |
| 0.5 N | 0.150 mm | ~0.065 mm | |
| 1 N | 0.291 mm | ~0.19 mm | |
| 2 N | 0.567 mm | ~0.445 mm | |
| 5 N | 1.327 mm | 1.156 mm | 1.358 mm |
| 10 N | 2.504 mm | 2.269 mm | 2.629 mm |

The small-load MuJoCo values are corrected for a ~0.01 mm contact-skin offset.
The reference is linear at small loads (about 3.5 kN/m, which agrees with a
flat punch on a finite elastic layer) and stiffens at large ones. **At large
loads the two solvers agree within 2–5% once MuJoCo's mesh is 3.5 mm.** The
5 mm mesh is 10–13% stiffer (locking), and more at small loads, where
MuJoCo's soft contact also enters. Static elasticity and contact geometry
are cross-validated.

## Drop test: where the solvers differ

For the 50 g drop with the M3 material (E = 54.8 kPa, ν = 0.45, no damping),
the reference peaks at **7.7 mm** (mesh-converged: 7.94 → 7.69 mm from 12 × 8
to 24 × 16). MuJoCo peaks at **4.9 mm** on a 5 mm mesh and about **5.5 mm**
converged (5 → 3.5 → 2.5 mm: 4.94, 5.33, 5.46). Since static elasticity agrees
(above), the difference comes from the impact dynamics:

- **MuJoCo depends strongly on friction:** 4.89, 5.72 and 6.80 mm at
  μ = 1, 0.3 and 0. The reference sticks perfectly, which should make it the
  stiffest.
- **MuJoCo's undamped impact loses energy.** Frictionless and without
  material damping, the weight's and specimen's kinetic energy plus gravity
  after separation is 77% of the input. With the specimen's (unreported)
  vibrational strain energy, roughly 10–25% is lost in the contact
  mechanics. This is independent of the timestep (0.1 to 0.01 ms) and of the
  contact damping ratio. With friction it loses more: soft constraints let
  the bulging specimen slip. The reference loses about 5%, to its numerical
  damping.
- **The material laws differ at large strain.** MuJoCo uses Stable
  Neo-Hookean and the reference uses split Neo-Hookean. They are equal at
  small strain (the static table above), but the drop reaches 30–45%
  compression under the punch.

So MuJoCo's drop-test fit (E = 55 kPa, β = 0.31 ms) is an effective
description that includes MuJoCo's own friction and contact losses, not the
silicone's modulus and damping. Next steps: Stable Neo-Hookean in the reference (an
F-bar form of a non-split energy, for a like-for-like law comparison) and
fitting the material on the reference (below).

## What modulus does the real silicone have?

Since MuJoCo's drop-test fit includes MuJoCo's own contact losses, the
reference solver asks the material question directly. Setup:

- Neo-Hookean, ν = 0.4995;
- sticking contact, no damping;
- the measured impact speed;
- first-contact comparison as in M3.

| E | 50 g peak / time (exp 4.31 mm / 3.7 ms) | 20 g peak / time (exp 3.74 mm / 3.3 ms) |
| --- | --- | --- |
| 60 kPa | 6.34 mm / 4.9 ms | 5.15 mm / 3.5 ms |
| 100 kPa | 4.88 mm / 3.6 ms (NRMSE 13%) | 3.94 mm / 3.0 ms |
| 150 kPa | 3.87 mm / 2.9 ms | 3.13 mm / 2.4 ms |

(20 g lands 6 mm off centre and is run centred, so it is approximate.)

- **E ≈ 100–130 kPa** matches the peaks. That is in line with the Ecoflex
  00-10 datasheet: its 100% modulus of 55 kPa means μ ≈ 31 kPa for a
  Neo-Hookean solid, so E ≈ 94 kPa, plus some stiffening at impact rates.
  MuJoCo's 55 kPa is an effective value.
- **No purely elastic E matches both peak and time to peak.** The stiffness
  that gives the right depth peaks too early, the signature of material
  damping. `tools/fit_reference_drop.py` fits E together with one Prony term.

**E + one Prony term (τ = 2 ms), fitted jointly on 50 and 20 g** (12 × 8
mesh, Nelder–Mead, 30 evaluations, `runs/reference_fit/w050_w020/`):

| | E (instantaneous) | g | 50 g NRMSE (peak) | 20 g NRMSE (peak) |
| --- | --- | --- | --- | --- |
| Joint best | 143 kPa | 0.69 | 20% (4.81 mm @ 3.7 ms) | 15% (3.62 mm @ 2.9 ms) |
| Best for 50 g alone | 141 kPa | 0.50 | 7.5% | 34% |
| Best for 20 g alone | 125 kPa | 0.73 | 35% | 4.8% |

- The joint fit gives an instantaneous modulus of about 145 kPa relaxing by
  ~70% within a few ms, so a long-time modulus of about 45 kPa. Read it as an
  effective impact description, not material constants: a single fixed τ
  cannot represent a broad relaxation spectrum.
- **Each case fits well on its own, but not together.** The two cases want
  different (E, g), and the joint error (18% mean) is no better than the
  elastic E = 100 kPa on 50 g (13%). Either one Prony term at τ = 2 ms is too
  simple (a free τ or two terms is the next step), or the 20 g case is
  biased, since it runs centred although the real weight landed 6 mm off
  centre and spinning. A 3D reference or an off-centre axisymmetric
  approximation would separate the two.
- **Caveat: the comparison rule.** As everywhere on the experiment side,
  "contact" is when the weight's centre of mass departs from free fall by
  0.1 mm (the tracker's noise limit), and the indentation counts from there.
  If the force builds slowly after the first touch, that point comes after
  some travel, in the recordings and in every simulation alike. The
  comparison is consistent, but it depends on how fast the force builds, and
  that differs between solvers: MuJoCo's soft contacts load abruptly, the
  reference's smooth contact gradually. For 500 g on the reference the
  detection fires several millimetres after the first touch, so its
  first-contact metric is not comparable. 500 g is left out of the
  reference fit.

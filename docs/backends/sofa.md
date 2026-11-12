# SOFA backend: setup, physics notes and limits

Verified against the **SOFA v26.06.00** Linux binary release built for Python
3.10, with its bundled SofaPython3. It runs on CPU; no GPU is needed.

## Install

```bash
# download and unzip the release, e.g. into ~/.local/opt/sofa
curl -LO https://github.com/sofa-framework/sofa/releases/download/v26.06.00/SOFA_v26.06.00_Linux_Python3.10.zip
mkdir -p ~/.local/opt/sofa && unzip SOFA_v26.06.00_Linux_Python3.10.zip -d ~/.local/opt/sofa
export SOFA_ROOT=~/.local/opt/sofa/SOFA_v26.06.00_Linux        # add to your shell profile
softsim run scenarios/drop_test/cases/w020.yaml --backend sofa --set simulation.timestep=5e-5
```

softsim finds SofaPython3 through `$SOFA_ROOT`. Without it, the SOFA tests
skip and `softsim check` still reports what SOFA would support.

## Scene

| softsim | SOFA |
| --- | --- |
| Soft body | `TetrahedronSetTopologyContainer`, `MechanicalObject`, lumped `MeshMatrixMass`, `EulerImplicitSolver`, `SparseLDLSolver` |
| `linear` | `TetrahedronFEMForceField method="small"` |
| `stvk`, `neo_hookean`, `stable_neo_hookean` | `TetrahedronHyperelasticityFEMForceField`, ParameterSet [μ, λ] |
| `mooney_rivlin` | same force field, [C10, C01, bulk] |
| `ogden` (one term) | same force field, [μ·α, α, bulk] (SOFA writes the term as μ₁/α²) |
| Tension-controlled cable | springs between the via points (`StiffSpringForceField` on a `BarycentricMapping` child) whose rest lengths are reset every step to `length - T / k`: the force is the tension at the current positions and the cable's tangent stiffness `T (I - u u^T) / l` is in the implicit matrix and the Rayleigh damping (`k` is 1% of `T` per via-point spacing, so the axial stiffness is negligible). An explicit follower force (a `ConstantForceField`, as before; identical numbers to the force-type `CableConstraint`) swung the two-cable arm in a limit cycle, 0.6-3.9 mm peak to peak at every step from 10 ms to 0.25 ms, because the elastic law's geometric stiffness is implicit and the cable stiffness that balances it was not ([control.md](../control.md#sofa-fast-path-softrobotsinverse-method-sofa-qp)). Stroke-controlled and anchored cables, pressure and region forces are still `ConstantForceField`s |
| Rayleigh α, β; Kelvin-Voigt | `EulerImplicitSolver rayleighMass/rayleighStiffness` (Kelvin-Voigt: β = η/E). α is checked against a free-falling body with linear drag (`v' = g − αv`): the velocity matches the implicit-Euler recurrence `v ← (v + g dt)/(1 + α dt)` to 1e-12 and the continuous solution to 1% at every step (first order in dt), 0.1% at the end (`tests/test_sofa_backend.py`). One damping per body: the solver owns it, and a force field's own `rayleighStiffness` has no effect on the elastic force fields (measured), so zones with their own damping are refused |
| `prony` | the force field carries the long-term law; the Maxwell-branch overstress (`softsim/materials/viscoelastic.py`, the same code as on MuJoCo) goes through a `ConstantForceField` updated before every step. SOFA's own `MJEDTetrahedralForceField` (SofaMJEDFEM) is not used: it relaxes the whole stress including the volumetric part, and its laws use non-isochoric invariants (its Mooney-Rivlin has a stress at rest) |
| `base: bonded` | `FixedProjectiveConstraint` |
| Rigid body | `Rigid3d` DOF at the centre of mass, `UniformMass` with inertia, triangulated parts (`RigidMapping`) |
| Contact | `FreeMotionAnimationLoop`, `BlockGaussSeidelConstraintSolver`, `MinProximityIntersection`, frictional Lagrange contact |

**Refused:** Yeoh, Gent, Arruda–Boyce, multi-term Ogden, torsional and
rolling friction, self-collision, magneto-active coupling. A card with one of
these laws can be projected onto one SOFA has (`softsim material project`).

## Verified

- **Material mappings.** All of `linear`, `neo_hookean`, `stable_neo_hookean`,
  `mooney_rivlin` and `ogden` are checked against analytic 1% uniaxial
  compression with forces applied directly to nodes (no contact), within 4%
  (`tests/test_sofa_backend.py`). Mooney-Rivlin's energy function in the
  SOFA sources has an integer division (`pow(J, -2/3)` evaluates to J⁰), but
  its forces were checked: zero stress at rest and the standard stiffness
  μ = 2(C10 + C01).
- **Timestep.** For the 20 g drop, dt = 0.1 ms and 0.05 ms agree within 2%.
  **The 500 g full-face impact diverges at 0.1 ms** (a rim vertex blows up at
  about 6 ms), and is stable at 0.05 ms. It can still diverge at 0.05 ms with
  stiff, heavily damped parameter sets (E = 100 kPa, β = 4 ms).
- **Blow-up guard.** A diverging FEM state floods the constraint solver with
  contacts, and SOFA segfaults while allocating the compliance matrix before
  any NaN shows up. The backend therefore stops stepping once any vertex or
  rigid body moves faster than `SOFTSIM_SOFA_MAX_SPEED` (default 100 m/s) and
  reports the run as diverged (`StabilityError`), like a NaN.

## Known limits

- **Speed.** About 0.001× real time for the drop test on a 5 mm mesh: a 22 ms
  impact takes 30 s, or 90 s at dt = 0.05 ms. MuJoCo is about 10× faster.
- **Line-line collision is off** for soft bodies. It's about 3× faster, and the
  drop test result didn't change.
- **Frictionless plate test (explained: friction).** The old "7% stiff, too much spreading" comes from a plate
  that was not frictionless: SOFA applies the scene's largest coefficient to every pair, so a 0.8 on any body, the
  ground or the plate sticks the plate to the specimen's top face and the ground. A weighted plate on a cylinder
  (R 20 mm, H 17 mm, stable Neo-Hookean 50 kPa, ν 0.45, 2% load, rim measured as the mean radius), contact against the
  same load as nodal forces on the same mesh (`tests/test_sofa_backend.py`, 8 mm mesh):

  | Friction on the ground and plate | Compression, contact / no contact | Rim spreading, contact / no contact |
  | --- | --- | --- |
  | 0 | 1.108 / 1.101 (+0.6%) | 1.012% / 1.011% (rim radius change) |
  | 0.01 | 0.996 of the no-contact one | 0.988 |
  | 0.05 | 0.976 | 0.959 |
  | 0.2 | 0.898 | 0.838 |
  | 0.8 | 0.670 (-33%) | 0.48% (half) |

  (Rows 0.01 to 0.2 are contact over no contact; the 0.8 row is at 8 mm for both, with 6 mm giving 0.765 and ground
  only, plate only and both identical.) So the 7% corresponds to a coefficient of about 0.1, and friction on either
  of the pair acts. At 0 the defaults already agree to 0.6% (0.4% at 6 mm), so the other suspects need no fix: the
  contact distance (0.1 mm per surface, 1.2% of 17 mm at most, and below the 0.6% seen), solver tolerance and
  iterations (defaults, converged), mesh coarseness (6 and 8 mm agree), and the analytic reference's boundary
  (free-slip faces on both sides; the no-contact FEM sits 1% above the analytic rim spreading of 0.91% at either mesh,
  the discretisation and not the contact). Keep one friction coefficient per scene on SOFA.
- **Frictionless impacts: compare with every pair frictionless.** With purely elastic cards, the centred 50 g
  drop at 2.7 m/s against the locking-free 3D reference (`tools/contact_restitution.py`; restitution e = rebound
  speed / impact speed):

  | Contact | Reference | SOFA, 5 mm | MuJoCo, 5 mm |
  | --- | --- | --- | --- |
  | Sticking, μ = 1 | 0.905 | 0.794 | 0.73–0.75 |
  | Every pair frictionless | 0.823, peak 7.53 mm | **0.795** (−3.4%), peak 6.45 mm (−14%) | – |
  | Specimen and ground frictionless, the weight at 0.5 | 0.869 | 0.650, peak 5.1 mm (not a like-for-like case) | 0.791 |

  The last row was long read as "SOFA loses 58% of the energy frictionless". It doesn't: SOFA applies the scene's
  largest friction coefficient to every pair, so the weight's 0.5 also acted between specimen and ground, where the
  reference and MuJoCo (friction per pair) had none. With every pair frictionless SOFA loses 3.4% more than the
  reference, and its peak is 14% shallow, the linear tetrahedra's stiffness at 5 mm. Halving the step (0.784) or
  tightening the constraint tolerance to 1e-9 does not change that. Keep one friction coefficient per scene on SOFA.
  With sticking friction SOFA loses less than MuJoCo. That is why its transfer of the physical card needs less extra
  relaxation (g 0.77 against 0.94, [experiments.md](../experiments.md)).
- **Rim flare through the ground (fixed: the alarm distance follows the per-step travel).** In the 500 g drop
  with the fitted Ogden material the specimen's top rim rolls under the specimen, and then snaps back onto the
  ground at about 12 m/s: 0.6 mm per 50 µs step, while the intersection's alarm distance was 0.4 mm. A point
  farther than the alarm distance from the ground at the start of a step gets no contact, ends it below the
  surface, and `MinProximityIntersection`'s normal then points away from the far side and keeps it there: one
  vertex (a surface vertex that was on the top face at rest) went 2.8 mm through at 8.3 ms (8.5 mm by 22 ms), with no
  inverted element. Measured on the 5 mm mesh, 8.8 ms (the artefact does not appear at 5.5 mm and coarser, and does not
  in a head-on drop of a block at 6 m/s per 0.5 ms steps):

  | Run | Deepest vertex through the ground | Peak compression |
  | --- | --- | --- |
  | Before (alarm 0.4 mm) | 2.83 mm | 13.13 mm |
  | Alarm 1.5 mm | 0.005 mm | 13.16 mm |
  | Time step 25 µs (0.3 mm per step) | 0.002 mm | 13.17 mm |
  | Constraint solver 1000 iterations, tolerance 1e-10 | 8.13 mm (worse) | 13.19 mm |
  | Every pair frictionless | 0.002 mm (no rim snap: the trajectory differs, peak 16.08 mm) | 16.08 mm |
  | **Alarm = max(0.4 mm, contact distance + fastest vertex speed × dt)** | **0.006 mm** | 13.14 mm |

  The solver's limits are not the cause (above), nor is the ground or the collision models: the ground is a 30 m
  triangle the rim cannot leave (the specimen reaches 42 mm), and every surface vertex carries a point and a triangle
  model. The
  backend now sets the alarm distance at each step from the fastest vertex speed
  (`SofaBackend._alarm`; `SOFTSIM_SOFA_ADAPTIVE_ALARM=0` restores the fixed distance, `SOFTSIM_SOFA_ALARM_FACTOR`
  scales the travel, default 1). It costs 12% more CPU time in this run (a factor 2, which fixes it as well, 60%).
  Below 6 m/s at 50 µs the nominal 0.4 mm applies and nothing changes. `ground_penetration` in calibration histories
  still records any left.
- **One friction coefficient** (the largest in the scene) applies to all pairs; `softsim check` and every run warn
  (`single-friction`) when the scene's coefficients differ.
  MuJoCo's contact compliance settings (`contact.timeconst`, `dampratio`,
  `solimp`) don't apply here: SOFA contacts are hard constraints.
  `simulation.integrator` and `solver` are MuJoCo settings and are ignored.
- **Contact forces** come from momentum balance per body: net force only, at
  the centre of mass, with no centre of pressure or torque.
- **Ground.** The ground is one large triangle. A shared triangle edge under
  the specimen let the vertex sitting exactly on it slip through (0.5 mm
  penetration) before this was fixed.
- **Fitted parameters are backend-specific.** With the same E, β and mesh,
  SOFA and MuJoCo differ by about 30% in peak indentation for the 20 g drop,
  and the stable-NH drop-test fits end at E = 105 kPa (SOFA) vs 55 kPa
  (MuJoCo). The SOFA fits and their comparison are in
  [experiments.md](../experiments.md#backends-and-material-models-m4); preset
  `ecoflex_00_10_drop_fit_sofa_ogden` is the SOFA one-term Ogden fit.
  The differences are hard vs compliant contact and integrator damping
  (MuJoCo's contact alone adds about 0.5 mm of penetration). Calibrate per
  backend.

## Rods (BeamAdapter)

`softsim.backends.sofa.rod` is the bare clamped cantilever on BeamAdapter that the rod backend grew
from (`F L^3 / (3 E I)` to 0.2%, `tests/test_rod_sofa.py`). The scene backend is `--backend rod`:
see [rod.md](rod.md).

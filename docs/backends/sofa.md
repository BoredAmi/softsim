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
- **Frictionless plate test.** Static compression by a frictionless plate
  through contact comes out 7% stiffer than analytic, with too much lateral
  spreading, while the same FEM without contact matches. This looks like a
  contact artifact in that configuration and is not resolved.
- **Frictionless impacts lose too much energy.** With purely elastic cards,
  the centred 50 g drop at 2.7 m/s against the locking-free 3D reference
  (`tools/contact_restitution.py`; restitution e = rebound speed / impact
  speed):

  | Contact | Reference | SOFA, 5 mm | MuJoCo, 5 mm |
  | --- | --- | --- | --- |
  | Sticking, μ = 1 | 0.905 | 0.794 | 0.73–0.75 |
  | Frictionless | 0.869 | **0.650** (peak 5.1 mm vs 7.5, contact 11.9 vs 10.5 ms) | 0.791 |

  With sticking friction SOFA loses less than MuJoCo. That is why its
  transfer of the physical card needs less extra relaxation (g 0.77 against
  0.94, [experiments.md](../experiments.md)). Frictionless, it loses 58% of the
  energy and is much too stiff, like the frictionless plate test below.
- **Contact fails at extreme rim flaring.** In the 500 g drop with the fitted
  Ogden material, the specimen's rim spreads to 45 mm radius (from 20 mm) and
  rim vertices then pass up to 8.5 mm through the ground, without any
  inverted element. Calibration histories record `ground_penetration` per
  case to catch this.
- **One friction coefficient** (the largest in the scene) applies to all pairs.
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

`softsim.backends.sofa.rod` builds a chain of rigid frames joined by Euler–Bernoulli
beams (`AdaptiveBeamForceFieldAndMass`). It is not yet a scene backend: only the
clamped cantilever with a tip load (`RodCantileverConfig`). Check against
`F L^3 / (3 E I)` (0.2% at 10 elements; independent of the element count; the
long, large-deflection case is geometrically nonlinear, so compare it only at
small loads). Reproduce: `SOFA_ROOT=... pytest tests/test_rod_sofa.py`.

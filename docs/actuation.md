# Actuation (M6, in progress)

Pressure is the first actuator. Cables, region forces and prescribed
displacements follow the same pattern (see [roadmap.md](roadmap.md), M6).

## Regions

Every soft shape names its surface regions: sets of boundary triangles with
outward normals, found on the mesh from the shape's own geometry
(`softsim/geometry/soft.py`).

| Shape (`geometry.type`) | Regions |
| --- | --- |
| `cylinder`, `box` | `base`, `top`, `side` |
| `hollow_sphere` (`outer_diameter`, `inner_diameter`) | `cavity`, `outer` |
| `closed_tube` (`outer_diameter`, `inner_diameter`, `height`, `cap`) | `cavity`, `base`, `top`, `side` |

The hollow shapes are meshed with gmsh (`pip install softsim[mesh]`). Regions
fix vertices (`body.fixed: [base]` pins every vertex of the base) and carry
loads.

## Pressure actuators

```yaml
scenario: inflation
body:
  geometry: {type: closed_tube, outer_diameter: 20 mm, inner_diameter: 12 mm, height: 50 mm, cap: 4 mm}
  material: ecoflex_00_30
  fixed: [base]
actuators:
  - name: cavity
    region: cavity
    schedule: [[0 ms, 0 kPa], [100 ms, 3 kPa], [200 ms, 3 kPa]]   # linear in between, held after
```

A pressure actuator loads its region with gauge pressure `p`: the traction
`-p n` on the deforming surface (a follower load). Its value comes from the
schedule, or from a controller:

```python
sim = Simulation("scenarios/inflation/scenario.yaml")
sim.backend.set_control("cavity", 5e3)     # Pa, from the next step on
sim.backend.release_control("cavity")      # back to the schedule
sim.state().controls                       # {"cavity": ...}, the values in effect
```

From the CLI, any value can be changed as usual:
`--set actuators[0].schedule="[[0, 0], [0.1, 4000]]"`.

On ROS 2, `~/actuators/<name>/command` (std_msgs/Float64, Pa) sets the
pressure, NaN releases it, and `~/actuators/<name>/state` publishes the value in
effect ([ros2.md](ros2.md)).

The `inflation` scenario writes each actuator's pressure and cavity volume
(`<name>_pressure`, `<name>_volume`) and the body's extent at every sample, so a
run gives the actuator's pressure–volume curve directly.

### How each backend applies it

- **fem (reference):** the potential `-p V`, with `V` the volume the region
  encloses, integrated exactly on the quadratic faces. It is the consistent
  follower load with a symmetric tangent, so Newton stays quadratic. The
  pressure is the value at the end of the step (implicit). The region must be
  closed; an open region is refused, because `-p V` is then not the follower
  load.
- **MuJoCo, SOFA:** nodal forces `-p A n / 3` on the linear surface triangles,
  recomputed from the current positions before every step (explicit, through
  `xfrc_applied` and a `ConstantForceField`, like adhesion).

## Verification: thick hollow sphere

The exact solution for an incompressible Neo-Hookean sphere (radii `a < b`,
shear modulus `mu`) under internal pressure, at inner hoop stretch `la`, is
(`softsim.analytic.hollow_sphere_pressure`):

```text
p = 2 mu [(1/lb + 1/(4 lb^4)) - (1/la + 1/(4 la^4))],   lb^3 = 1 + (a/b)^3 (la^3 - 1)
```

It has a maximum, the limit pressure: 0.816 mu at `la` = 1.83 for `b = 2a`.
Above it, no inflated static state exists and a real balloon snaps through.

**Reference (one octant, symmetry planes, nu = 0.4995, 4 mm mesh):** the
inner stretch matches the exact solution within **0.1%** up to `la` = 1.62
(p = 0.8 mu), and the quasi-static solve fails above the limit pressure, as it
must (`tests/test_actuation.py`). The symmetry planes pass through the centre,
where `x . n = 0`, so `-p V` stays exact on the open octant.

**Backends at the same card** (`python tools/compare_inflation.py`; 40/20 mm
sphere, Neo-Hookean E = 30 kPa, nu = 0.45, 3 kPa held):

| Model | Cavity V/V0 | Volume change vs reference |
| --- | --- | --- |
| **fem reference** (quadratic, 3 mm) | 1.3336 | – |
| MuJoCo, 3 mm | 1.2426 | **−27%** |
| MuJoCo, 2 mm | 1.2890 | **−13%** |
| SOFA, 3 mm | 1.2427 | **−27%** |
| Linear Lamé, ν = 0.45 | 1.2748 | −18% |
| Exact incompressible solution, μ = 10.3 kPa | 1.3354 | +0.5% |

- **The reference agrees with the exact finite-strain solution within 0.5%.**
  That solution is incompressible and the card has ν = 0.45, so this is a
  consistency check, not an exact comparison. Linear Lamé misses 18% of the
  volume change at this 33% inflation: at actuator pressures, finite strain
  matters.
- **MuJoCo and SOFA agree with each other to 0.01%, and both are 27% too
  stiff at 3 mm.** Two independent implementations agreeing shows the pressure
  load is right. The gap is volumetric locking of linear tetrahedra at
  ν = 0.45, as in the drop test ([reference_fem.md](reference_fem.md)).
- **Locking shrinks with the mesh** (−27% → −13% from 3 to 2 mm, about h^1.8),
  but slowly and at a high cost: 2 mm is 3.4× the elements. For a pneumatic
  actuator, either use a fine mesh, or fit an effective modulus for the fast
  backend at its mesh against the reference (as the drop transfers do,
  `calibrations/transfer_*.yaml`; for inflation this is still to be wired up).
  On a coarse linear-tet mesh, a physical card makes the actuator too stiff.

## Transfer: making a fast backend inflate like the reference

`softsim transfer` fits a fast backend's effective card at its mesh, so that
it reproduces the reference on chosen outputs. It then checks the fit on load
cases it never saw (`softsim/transfer.py`;
`calibrations/transfer_inflation_mujoco.yaml`, 9 evaluations, 11 min):

| Backend, 3 mm | E | Fitted: 3 kPa | Check: 1.5 kPa | Check: 4.5 kPa |
| --- | --- | --- | --- | --- |
| MuJoCo: physical card | 30 kPa | 20% | 19% | 22% |
| **MuJoCo: transferred** | **23.7 kPa** | **0.5%** | **1.2%** | **2.2%** |
| SOFA, 3 mm: physical card | 30 kPa | 20% | 19% | 22% |
| **SOFA, 3 mm: transferred** (`transfer_inflation_sofa.yaml`) | **23.6 kPa** | **0.9%** | **1.4%** | **2.4%** |

The error is the NRMSE of the cavity-volume history against the reference.

- **One effective modulus is enough over a 3× range of pressure:** fitted at
  33% inflation, it holds at 15% and 52% inflation within 2.2%.
- **The softening is 21%,** close to the locking measured above (27% too
  little volume change). MuJoCo and SOFA need the same correction (23.7 and
  23.6 kPa), as expected for the same linear tetrahedra on the same mesh.
- **The result is an `effective` card** (`kind: effective`, with backend,
  mesh, element and check errors in its provenance). Use it only with that
  backend and mesh.

The identity test (`tests/test_actuation.py`) transfers MuJoCo to itself and
recovers the generating modulus within 1%.

## A bending actuator

`scenarios/inflation/bending.yaml` is a single-material bending actuator:
- an Ecoflex 00-30 tube, 16 mm wide and 40 mm tall, bonded at its base;
- an 8 mm cavity moved 2 mm off-axis (`cavity_offset`), so one wall is 2 mm
  thick and the other 6 mm.

The thin wall stretches more, so the tube curls towards the thick side. The
scenario reports the top face's tilt (`bend_angle`) and position (`tip_x`,
`tip_y`, `tip_z`).

| At 8 kPa, 2 mm mesh, same card | Bend angle | Tip shift | Cavity V/V0 |
| --- | --- | --- | --- |
| fem reference | 3.45° | −1.28 mm | 1.407 |
| MuJoCo | 2.16° (−37%) | −0.77 mm | 1.258 |

- **The bending is strongly nonlinear:** the reference bends 1.0° at 4 kPa and
  3.4° at 8 kPa.
- **Thin walls make locking worse:** MuJoCo is 37% too stiff even at 2 mm,
  because the 2 mm wall is a single element thick.
- **This shape needs `simulation.iterations: 300` on MuJoCo.** With the
  default of 100, the stiff flex's M+K solve stops at a residual of about 1e-7,
  and MuJoCo warns. The result doesn't change, but the config sets 300 so the
  run is clean.

**Transfer** (`calibrations/transfer_bending_mujoco.yaml`; fitted on the bend
angle and cavity volume from 0 to 8 kPa):

| MuJoCo, 2 mm | E | Fitted: 0–8 kPa | Check: 0–4 kPa |
| --- | --- | --- | --- |
| Physical card | 114 kPa | 23% | 18% |
| **Transferred** | **86.5 kPa** | **4.1%** (bend 3.1%, volume 5.1%) | **8.4%** |

The transfer helps, but less than on the sphere. One effective modulus can't
fully match a thin wall that locks differently as it stretches, so the error
at half pressure is twice the fitted one. For bending actuators on linear
tetrahedra, plan for an error bar of about 10% after transfer. Without the
transfer, it is about 20%.

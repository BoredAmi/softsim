# Actuation (M6, in progress)

Three actuators so far: pressure on a surface region, cables routed through
the body (by tension or by stroke), and forces on a surface region. The fixed
regions can also follow a prescribed translation (a moving mount). See
[roadmap.md](roadmap.md), M6. The scenario is `actuation` (`inflation`
is the same scenario under its first name).

## Regions

Every soft shape names its surface regions: sets of boundary triangles with
outward normals, found on the mesh from the shape's own geometry
(`softsim/geometry/soft.py`).

| Shape (`geometry.type`) | Regions |
| --- | --- |
| `cylinder`, `box` | `base`, `top`, `side` |
| `hollow_sphere` (`outer_diameter`, `inner_diameter`) | `cavity`, `outer` |
| `closed_tube` (`outer_diameter`, `inner_diameter`, `height`, `cap`) | `cavity`, `base`, `top`, `side` |
| `pneunet` (see [below](#a-pneunet-finger)) | `cavity`, `base`, `tip`, `bottom`, `top`, `side` |
| `mesh_file` (STEP, STL, OBJ, `.msh`, VTK; [adding_a_body.md](adding_a_body.md#soft-body-from-a-file-no-code)) | the file's named groups, `surface`, `cavity` (found by itself), boxes you name |

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

## Cable actuators

```yaml
scenario: actuation
body:
  geometry: {type: box, size: [10 mm, 10 mm, 60 mm]}
  fixed: [base]
  ...
actuators:
  - name: cable
    type: cable
    path: [[3 mm, 0, 0], [3 mm, 0, 55 mm]]     # body frame: from the exit to the motor, to the anchor
    schedule: [[0 ms, 0 N], [400 ms, 0.7 N]]    # tension; or `tension: 0.5 N`
    # spacing: 2 mm                             # via points at most this far apart (default: mesh size)
```

A cable is a tendon of tension `T` through via points along `path`. The via
points are embedded in the body's elements, so they act as frictionless guides
that move with the material. The motor sits on the body at the first point,
so the cable is internal: its net force and moment on the body are zero.
`set_control("cable", 0.3)` and ROS `~/actuators/cable/command` set the tension
in N. A cable can only pull, so a negative tension is refused.

The load is the potential `T L(x)`, where `L` is the cable's length through the
via points:
- **fem (reference):** in the implicit solve. Via points use the elements'
  quadratic shape functions. The tangent is `T w w^T (I - u u^T) / l` per
  segment, symmetric and PSD, so Newton stays robust.
- **MuJoCo, SOFA:** the same forces `-T dL/dx` on the linear elements'
  vertices, recomputed before every step (`softsim.actuation.cable_forces`).
  They are explicit, so `softsim check` compares the cable's string stiffness
  `T (1/l1 + 1/l2)` with the vertex masses at the timestep. It warns or refuses,
  as for adhesion: tight via-point spacing at high tension needs a smaller
  timestep, or a larger `spacing`.

The scenario reports `<name>_tension` and `<name>_length` (the current length
through the via points), and in the summary the cable's stroke (how far it was
pulled in).

### Verification: cable-bent cantilever

`scenarios/actuation/cable_cantilever.yaml`:
- a 10 × 10 × 60 mm beam, Neo-Hookean E = 500 kPa, ν = 0.45, clamped at its
  base, no gravity;
- a cable 3 mm off-axis, anchored at L = 55 mm.

Below the anchor the cable puts a pure moment `T d` on the beam. Euler–Bernoulli
then gives a circular arc whose tip angle is exact at any rotation:

```text
theta = T d L / (E_t I),   E_t = E (1 + 2 T / (E A))
```

`E_t` is the Neo-Hookean tangent under the axial pre-compression `T / (E A)`
that the cable also applies: 2.8% stiffer at 0.7 N.

**The reference matches it within 0.3%** at 0.1 and 0.7 N (0.04 and 0.27 rad,
quasi-static, 2.5 mm; `tests/test_cable.py`). Two things the check taught:
- **Without `E_t`, the error grows with the load** (−2.4% → −5.2% over the same
  range). It halves on a beam twice as long at the same tip angle, so it scales
  with strain, not rotation: material nonlinearity, as the formula says.
- **Anchor the cable inside the body, not on the face where you measure.** A
  point anchor on the top face dents it, and the face's tilt then reads 4–6%
  low. Anchored 5 mm below, the top face rotates rigidly with the beam.


![Cable-bent cantilever: tip angle against tension, and the deformed beam against the
Euler-Bernoulli arc](media/cable_cantilever.png)

*Left: tip angle over the whole ramp. Right: the side faces at 0.7 N, the
reference and MuJoCo (2 mm) against the theoretical arc (`python
tools/plot_cable.py`).*

**Backends at the same card** (`python tools/compare_cable.py`; tip angle,
error against Euler–Bernoulli with `E_t`):

| Model | 0.35 N | 0.7 N | vs reference |
| --- | --- | --- | --- |
| Euler–Bernoulli | 7.83° | 15.45° | |
| **fem reference**, 2 mm, dynamic | 7.78° (−0.7%) | 15.34° (−0.7%) | – |
| MuJoCo, 2 mm | 5.46° (−30%) | 10.79° (−30%) | −30% |
| SOFA, 2 mm | 5.45° (−30%) | 10.79° (−30%) | −30% |
| MuJoCo, 1 mm (via points 3 mm apart) | 6.96° (−11%) | 13.74° (−11%) | −10% |

- **The reference is right in dynamics too:** a 400 ms ramp against the beam's
  10 Hz mode is quasi-static, and the dynamic run agrees with the quasi-static
  solve.
- **MuJoCo and SOFA agree with each other to 0.1% and bend 30% too little at
  2 mm.** Linear tetrahedra lock in bending: 5 elements across a 10 mm beam,
  ν = 0.45. The error is the same at both tensions, so one effective modulus
  fixes it (`softsim transfer`, as for pressure). At 1 mm it is 10%.
- **On MuJoCo and SOFA, choose the via-point spacing from the check, not the
  mesh.** On the 1 mm mesh at 1 mm spacing (ω·dt ≈ 1.4), MuJoCo tracks the ramp
  up to 0.6 N, then rings between 14° and 18° instead of settling. Halving the
  timestep only takes ω·dt to 1.15, because the implicit elastic stiffness
  already dominates the step. Spacing the via points 3 mm apart (ω·dt ≈ 0.8)
  runs clean. The cable path is straight, so it follows the beam just as well.

### Stroke control: a servo reeling the cable in

```yaml
actuators:
  - name: cable
    type: cable
    control: stroke                              # the value is a length, not a tension
    stiffness: 300 N/m                           # the tendon's (or servo's) axial stiffness
    path: [[3 mm, 0, 0], [3 mm, 0, 58 mm]]
    schedule: [[0 ms, 0 mm], [400 ms, 3 mm]]     # stroke; or `stroke: 2 mm`
```

A stroke-controlled cable is a tendon of stiffness `k` and free length
`L0 - s`: `L0` is its length at rest, `s` the stroke the motor has reeled in.
It pulls with `T = k max(L - L0 + s, 0)` and is slack when shorter. With a stiff
tendon the stroke sets the cable's shortening (1e5 N/m: 1 mm of stroke shortens
it by 0.996 mm), and the tension follows from the body. `set_control` and ROS
take the stroke in m. A negative stroke pays the cable out.
- **fem (reference):** `k/2 (L - L0 + s)^2` while taut, in the implicit solve.
  The tangent adds `k dL dL^T`, which couples the whole cable, so the stiff
  limit stays robust.
- **MuJoCo, SOFA:** `T` from the current length before every step, then the
  usual cable forces. The check adds the tendon's axial term
  `k sum |dL/dx|^2 / m` to the string stiffness. At 0.1 ms on the 2 mm beam,
  300 N/m runs clean and 3000 N/m is refused (ω·dt = 1.45). A stiff tendon
  needs the reference, or a compliant one standing in for the servo.

The scenario reports `<name>_command` (the stroke) and `<name>_tension`, which
follows from the current length.

**Anchor a stroke-controlled cable in something stiff.** A cable ending at a
point inside the soft body pulls that point into the material. A point load is
singular, so how far it moves depends on the mesh, and under stroke control
that local give counts as cable shortening: 10% less tension and tip angle
here. `scenarios/actuation/cable_cantilever_stroke.yaml` anchors the cable
3 mm into a 5 mm tip cap (a 50 MPa zone), as a real finger anchors its tendon
in a rigid tip.

**Verification:** Euler–Bernoulli puts the beam's compliance along the cable in
series with the tendon (`softsim.analytic.cable_beam_stroke`):
`s = T/k + L (a^2 T / (E_t I) + T / (E A))`. The axial term is as large as the
bending term on this beam.

| Model, 2 mm | 0.3 mm: tension | 0.3 mm: tip angle | 3 mm: tension | 3 mm: tip angle |
| --- | --- | --- | --- | --- |
| Euler–Bernoulli | 0.0534 N | 1.209° | 0.536 N | 11.91° |
| **fem reference** (2.5 mm, quasi-static) | 0.0537 N (+0.5%) | 1.204° (−0.4%) | 0.540 N (+0.8%) | 11.90° (−0.1%) |
| fem, point anchor, no cap | 0.0480 N (−10%) | 1.084° (−10%) | 0.483 N (−10%) | 10.71° (−10%) |
| MuJoCo | | | 0.584 N (+9%) | 8.79° (−26%) |
| SOFA (1 ms) | | | 0.584 N (+9%) | 8.79° (−26%) |

MuJoCo and SOFA agree to 0.01%. Their beam is too stiff (the locking of the
tension-controlled table), so the same stroke takes more tension and bends it
less. The stroke run's equilibrium is the tension run's at the tension it
found (`tests/test_cable.py`).

## Region forces

```yaml
actuators:
  - name: tip_force
    type: force
    region: top                 # any surface region
    direction: [1, 0, 0]        # world frame, normalised
    schedule: [[0 ms, 0 mN], [400 ms, 5 mN]]     # total force [N]; or `force: 5 mN`
```

A total force along a fixed direction, spread over a surface region by area:
a dead load, such as a payload or a probe pressing on a fingertip. A negative
force pushes the other way.
- **fem (reference):** the consistent load of a uniform traction on the
  quadratic faces. It is all on the mid-edge nodes, a third of each face's
  area each.
- **MuJoCo, SOFA:** a third of each triangle's share to each of its vertices.

It doesn't depend on positions, so it needs no stability check. Outputs:
`<name>_force` and `<name>_displacement`, the region's area-weighted centre
moving along the force. Force against displacement is the body's stiffness
there.

**Verification:** `scenarios/actuation/tip_load_cantilever.yaml`, the beam
above with 5 mN on its top face. Euler–Bernoulli gives 0.864 mm. Timoshenko
adds 2.4% of shear (0.884 mm,
`softsim.analytic.cantilever_tip_deflection`). The reference converges to
0.870, 0.867, then 0.864 mm at 2.5, 2 and 1.25 mm (`tests/test_region_force.py`),
between the two. On this short beam (L/h = 6), the fully clamped root (no
warping, no Poisson contraction) offsets the shear. On MuJoCo and SOFA, the
vertex forces sum to the commanded force exactly. Both deflect 0.602 mm at
2 mm, 30% stiff: the locking of the cable tables, now under a known force.

## Moving mount

```yaml
body:
  fixed: [base]
  mount_path: [[0 ms, [0, 0, 0]], [200 ms, [20 mm, 0, 0]], [400 ms, [20 mm, 0, 10 mm]]]
```

The fixed regions follow a prescribed translation, such as a finger on a robot
arm. Waypoints are displacements from rest (the first is zero). Between
waypoints the mount moves point to point with minimum jerk: at rest at every
waypoint, continuous acceleration, no overshoot. After the last one it holds.

Every backend simulates in the mount's frame. For a translation that is exact:
the pinned vertices stay put, and every element feels the inertial force
`-rho a(t)`, so gravity becomes `g - a(t)` (`Backend.mount_gravity`, set before
every step). Positions and velocities are reported in the world, with the
mount's added back (`Backend._to_world`). A rotating mount would also need
centrifugal and Coriolis terms, and the ground and rigid bodies would have to
move with the frame, so those are refused.

**Verification:** the reference run two ways on
`scenarios/actuation/moving_mount.yaml` (the beam, its base sliding 20 mm in
200 ms):
- in the mount's frame;
- in the world, with the base nodes driven along the path
  (`Solid3DSolver.fixed_motion`, Newmark-consistent).

With the trapezoidal rule the tip trajectories agree to 0.10% of the tip's lag
at 2 ms and 0.026% at 1 ms. That is second order: the two formulations converge
to one solution (`tests/test_moving_mount.py`). With the default numerically
damped Newmark (first order) the gap is 2.2% at 2 ms and halves with the step.
On MuJoCo and SOFA the base vertices follow the path exactly, and the tip lags
while the mount accelerates.

| Model, 2 mm | Tip's peak lag | Swing after the mount stops |
| --- | --- | --- |
| **fem reference** (2 ms) | 2.15 mm at 64 ms | 0.51 mm |
| MuJoCo | 1.47 mm (−32%) at 56 ms | 0.94 mm |
| SOFA (1 ms) | 1.40 mm (−35%) at 56 ms | 0.75 mm |

The locked beams of MuJoCo and SOFA are stiffer, so they lag about a third
less and ring at a higher frequency. The swing left after the stop depends on
where in its cycle the beam is when the mount stops, so it differs more.
SOFA's implicit Euler also damps it.

**Closed-loop control from Python:** `Simulation.run(controller=f)` calls
`f(sim)` before every output interval. `f` reads `sim.time`, `sim.metrics()`
and `sim.state()`, and returns the actuator values to hold, `{name: value}`;
NaN hands an actuator back to its schedule (`Simulation.apply_controls`). In
`tests/test_cable.py`, a PI loop on the cable tension holds the lightly damped
beam at a 0.1 rad tip angle on MuJoCo, within 3%.

## Multi-material bodies

```yaml
body:
  geometry: {type: box, size: [10 mm, 10 mm, 60 mm]}
  material: {density: 1070 kg/m^3, elastic: {model: stable_neo_hookean, youngs_modulus: 500 kPa, poisson_ratio: 0.45}}
  zones:
    - name: layer                     # e.g. a strain-limiting layer
      material: {density: 1070 kg/m^3, elastic: {model: stable_neo_hookean, youngs_modulus: 5 MPa, poisson_ratio: 0.45}}
      box: [[-5 mm, -5 mm, 0], [-3 mm, 5 mm, 60 mm]]   # elements whose centroid is inside (body frame)
```

A zone gives some elements their own material. Make the mesh resolve the
zone: a zone smaller than the elements catches only the elements whose centroid
falls inside it. A zone selects elements by `box` or by a shape's volume
region (`region: layer` on a `pneunet`, a named 3D group of a mesh file).

A zone may have its own density: every element carries its material's
density, and the vertex masses are lumped from them (a `body.mass` override
scales all densities alike). All three backends use these masses (MuJoCo per
vertex body, SOFA a per-element `massDensity`, the reference in its consistent
mass matrix). Under gravity, a cantilever whose outer half is four times as
dense sags 3.56 times as much as a uniform one. That is Euler–Bernoulli's ratio
(3.562). The reference is within 0.15% of the theory's tip deflection
(`tests/test_zones.py`). A zone must still have the body's damping, because the
viscous history is that of one material.

| Backend | Zones |
| --- | --- |
| fem (reference) | any law per zone; Prony applies to each zone's own stress |
| SOFA | one force field per material, each in a child node on that material's elements |
| MuJoCo | the body's elastic model (Stable Neo-Hookean or StVK, one per flex), any modulus and Poisson's ratio. MuJoCo precomputes 24 values per element: the material constants times the element's geometry, plus geometry alone. A zone's rows are copied from a model compiled with the zone's material, so they match it exactly (`tests/test_zones.py`). Scaling the rows is wrong, because it also scales the geometry-only value. |

**Verification: a two-material cable beam**
(`scenarios/actuation/cable_cantilever_layered.yaml`,
`python tools/compare_cable.py --layered`). This is the cable cantilever with
a 2 mm layer, ten times stiffer, on the face away from the cable. In composite
Euler–Bernoulli (`softsim.analytic.cable_beam_tip_angle`):
- the neutral axis moves to x = −2.57 mm;
- EI grows 2.3×;
- the cable's moment arm grows to 5.57 mm.

The tip angle at 0.7 N falls from 15.4° to 12.7°.

| Model | 0.35 N | 0.7 N | Layered ÷ homogeneous, 0.7 N |
| --- | --- | --- | --- |
| Composite Euler–Bernoulli | 6.36° | 12.66° | 0.820 |
| **fem reference**, 2 mm | 6.28° (−1.4%) | 12.35° (−2.4%) | 0.805 |
| SOFA, 2 mm | 4.18° (−34%) | 8.35° (−34%) | 0.774 |
| MuJoCo, 2 mm | 4.20° (−34%) | 8.35° (−34%) | 0.774 |
| MuJoCo StVK, 2 mm | 3.48° (−45%) | 7.07° (−44%) | 0.745 |
| MuJoCo StVK, 1 mm | 5.16° (−19%) | 10.51° (−17%) | 0.772 |

- **The reference matches composite beam theory** within 0.55% at 0.1 N
  (`tests/test_zones.py`). Its gap at 0.7 N is Neo-Hookean nonlinearity, not a
  zone error: the stiff layer works in compression and the soft part in
  tension, so the two no longer cancel as in one material. The gap halves on a
  beam twice as long at the same angle.
- **MuJoCo and SOFA agree on the layered beam to 0.1%** (8.35° both), with
  two independent zone implementations.
- **A zone with the body's own material reproduces the plain body exactly**
  (to 1e-9 on MuJoCo, 1e-6 on SOFA, and on the reference). SOFA needs a child
  node per material: a force field's `topology` link is ignored, and two force
  fields in one node both act on every element. The identity test caught that.
- **On MuJoCo and SOFA the layer stiffens the beam more than theory says**
  (ratio 0.75–0.77, against 0.82). With linear tetrahedra, a stiff layer one
  element thick locks even more than the soft body does. At 1 mm the MuJoCo
  ratio moves from 0.745 to 0.772, towards theory, so this is discretisation.
  Resolve a stiff layer with two or more elements, or transfer the card
  (`softsim transfer`).
- **MuJoCo's StVK flex locks more than its Stable Neo-Hookean flex** in bending
  (−39% against −30% at 2 mm), so prefer Stable Neo-Hookean.

## A PneuNet finger

`geometry: {type: pneunet, ...}` is a parametric PneuNet bending actuator:
- a block along +x, its base face at x = 0;
- `chambers` closed chambers in its upper part, joined by a channel just above
  a solid base slab, so they form one cavity;
- an optional strain-limiting `layer` at the bottom of the slab. The mesh
  conforms to it, and a zone takes it by name (`region: layer`).

Regions: `cavity`, `base`, `tip`, `bottom`, `top`, `side`. The tip angle is
the `tip` face's rotation from its rest orientation.

`scenarios/actuation/pneunet.yaml`:
- six 6 mm chambers in Ecoflex 00-30 (datasheet card), 54 × 16 × 16 mm;
- 2 mm walls, a 5 mm slab whose bottom 1.5 mm is a 1 MPa silicone layer;
- base bonded, no gravity, a smooth ramp to 8 kPa
  (`python tools/compare_pneunet.py`; 1.5 mm mesh, 17.9k tetrahedra).

| Model | 4 kPa: tip angle | 4 kPa: V/V0 | 8 kPa: tip angle | 8 kPa: V/V0 | Wall time |
| --- | --- | --- | --- | --- | --- |
| **fem reference** | 5.29° | 1.232 | 13.51° | 1.562 | 7 min |
| MuJoCo | 4.29° (−19%) | 1.169 (−27%) | 10.25° (−24%) | 1.404 (−28%) | 6.4 min |
| SOFA | 4.22° (−20%) | 1.168 (−28%) | 10.25° (−24%) | 1.404 (−28%) | 17 min |

(Volume errors are relative to the reference's volume change.)

- **MuJoCo and SOFA agree with each other** to 0.1% at 8 kPa, and both bend
  24% less than the reference, as on the beams. Locking is the same story with
  2 mm walls only one or two elements thick.
- **`softsim transfer` fixes it, with both moduli fitted together**
  (`calibrations/transfer_pneunet_mujoco.yaml`, 13 MuJoCo runs, 1.8 h). It
  fits the body and the layer on the 0–8 kPa bend-angle and cavity-volume
  histories. MuJoCo at 1.5 mm needs body E 114 → 91.0 kPa and layer
  E 1 → 1.76 MPa. The error (NRMSE against the reference) falls from 17.7% to
  2.9% (bend angle 4.6%, volume 1.2%). It holds on cases the fit never saw:

  | Check | Raw card | Transferred cards |
  | --- | --- | --- |
  | 11 kPa (larger strain) | 21.5% | 3.2% |
  | 4 chambers (a shorter finger) | 17.1% | 4.3% |

  The fit softens the body, where the thin walls lock, and stiffens the layer:
  its bending resistance is the other half of the curl. The transfer writes
  one effective card per fitted material
  (`ecoflex_00_30_mujoco_transfer.yaml`, `ecoflex_00_30_layer_mujoco_transfer.yaml`).
  They are valid for MuJoCo, linear tetrahedra and 1.5 mm.
- **Above about 15 kPa the walls balloon.** On MuJoCo the volume runs away:
  V/V0 2.85 and 39° at 16 kPa, then the run diverges. These thin Ecoflex
  chambers approach their limit pressure, like the sphere. The reference,
  which is softer, gets there sooner, so this scenario stops at 8 kPa.
- **Not yet compared with a real finger.** I found no open dataset of
  bending angle against pressure with full geometry and material. Measured
  data is M7 (PneuNet datasets with DOI and licence). Until then these numbers
  check the backends against each other and the reference, not against
  reality.

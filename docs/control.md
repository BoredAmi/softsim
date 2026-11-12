# Inverse control (M11, first piece)

Given a scenario with actuators and a target for one or more of its outputs, find the actuator values that
reach it at equilibrium. Code: `softsim/control/inverse.py`; tests: `tests/test_inverse.py`.

```bash
softsim inverse jobs/inverse_cable_cantilever.yaml     # tension for a 2 degree tip tilt
softsim inverse jobs/inverse_cable_arm.yaml            # two cables, tip x and z
```

```python
from softsim.control.inverse import solve_inverse

res = solve_inverse("scenarios/actuation/cable_cantilever.yaml", {"bend_angle": "2 deg"},
                    ["cable"], {"cable": (0, "0.7 N")}, overrides=["body.mesh.size=4mm"])
res.values, res.status, res.residual, res.simulations
```

A job file names the `config`, `overrides`, `backend`, `targets` (any output column of the scenario, such as
`tip_x`, `tip_z`, `bend_angle`, `chord_angle`; values with units), `actuators` (name: `[lower, upper]`, finite) and
optionally `tolerance`, `max_evaluations`, `initial`, `scales` and the settling options below. The result is printed
as JSON: `values`, the `outputs` reached, `errors`, `residual`, `status`, `at_bounds`, `simulations`, `steps`,
`wall_time`.

## How it solves

1. **The equilibrium map.** The actuators are held at constant values (`Simulation.apply_controls`), the scene runs
   until every target output has stopped changing, and the outputs are read. One `Simulation` is built and reused:
   each evaluation continues from the previous equilibrium (`warm_start`; `reset()` is called only at the start, or
   for every evaluation with `warm_start: false`). For an elastic body the equilibrium does not depend on the path,
   so this only saves time. Actuators that are not unknowns are held at their value at t = 0.
2. **Settling.** An elastic body rings forever, so the solve gives the body (and each zone without damping of its
   own) stiffness-proportional damping: `settle_damping`, Rayleigh beta, default 10 ms. A damping force vanishes at
   rest, so the equilibrium is where it was. MuJoCo has no mass-proportional damping, so the damping time scale has to
   suit the slowest mode: the cantilever (first mode ~20 Hz at 4 mm) settles with 10 ms; the two-segment arm (first
   mode ~1 Hz) needs 150 ms. The state counts as settled when every target output moves less than `settle_tol`
   (default 0.001) of its scale over the last `settle_window` (default 0.1 s). The window has to be longer than the
   slowest relaxation: with a 20 ms window the arm was declared settled while it had only just started to move and
   the finite-difference Jacobian was wrong (found while building this, see the arm job). Hitting `max_settle` is
   counted in `unsettled`.
3. **The solve.** Scaled residuals `(output - target) / scale` (scale: 1 mm for lengths, 0.01 rad for angles, 1 mN for
   forces, 100 Pa for pressures; override with `scales`) are minimised by `scipy.optimize.least_squares` (trust
   region, bounds) over the actuator values scaled to [0, 1] of their bounds, with a forward-difference Jacobian
   (step 2% of the range): one extra equilibrium per actuator per iteration. It stops when the largest scaled
   residual is below `tolerance` (default 0.05) or after `max_evaluations` equilibria.
4. **The outcome.** `status` is `converged`; `bounded_best` (an actuator sits at a bound and the target is not
   reached: the bounded best is returned, `at_bounds` names it, no exception); `max_evaluations`; or `stalled`.

Targets that do not determine the actuators are refused (fewer targets than actuators).

## Measured (MuJoCo, 4 mm cantilever mesh; the machine was shared, times are indicative)

| Case | Result | Simulations | Wall time |
| --- | --- | --- | --- |
| Cantilever, target = angle at a known 0.40 N, start 0.05 N | 0.3998 N (-0.06%), residual 0.003 | 11 | 28 s |
| same, start 0.65 N | 0.4002 N (+0.04%), residual 0.002 | 7 | 19 s |
| Job: 2 degrees (0.0349 rad) | 0.2716 N, angle error 8.7e-5 rad | 5 | 13 s |
| Unreachable: 0.5 rad with at most 0.3 N | 0.3 N (the bound), angle 0.0386 rad, `bounded_best` | 10 | 24 s |
| Arm (6 mm mesh, 0.5 ms), known 0.45 / 0.15 N, tip x and z | 0.4479 / 0.1487 N, errors 22 um in x and 7.5 um in z | 13 | 44 s |

**Linearity.** On the cantilever MuJoCo's own forward map gives 0.01296, 0.02583, 0.05130 rad at 0.1, 0.2, 0.4 N:
the slope varies by 1.1% over the range, as Euler-Bernoulli says for small angles (`theta = T d L / (E I)`), and
the inverse of twice the 0.1 N angle is 0.2 N within 4% (tested). MuJoCo's angle is 0.33 times the EB value at 4 mm
(the locking documented in [actuation.md](actuation.md)).

**Conditioning.** Two cables with the tip position as targets is an ill-posed problem unless the targets carry both
degrees of freedom. On the arm, tip x and the tilt `bend_angle` fail: `bend_angle` is an unsigned angle, folded at
zero, so the map has a kink. Tip x and tip z work, but z moves ~1.5 mm over the whole range of the cables (it carries
the "S against C" shape in second order), so it is judged to 0.2 mm (`scales`); at a loose tolerance (0.3) the
solution reached 0.435 / 0.144 N for the true 0.45 / 0.15 N, at 0.1 it reached 0.448 / 0.149 N. The tolerance bounds
the residual, not the actuator error.

## Limits

- **Quasi-static only.** Each answer is a held value that reaches the target after settling; the path (and the time
  it takes, which depends on the damping used to settle) is not planned. Transient control is the next M11 piece.
- **The values are the backend's.** The MuJoCo flex locks (about 30% too stiff in bending at 2 mm, a factor 3 at the
  4 mm used here for speed), so tensions found on it are not physical ones. For physical values use a card
  transferred to the backend (`softsim transfer`, [actuation.md](actuation.md)) and a mesh that has been checked
  against the reference, or solve on the reference backend, which is much slower.
- **One equilibrium per value.** A structure that snaps or buckles can have several equilibria; the one found
  depends on the start, and the finite-difference Jacobian is wrong across a snap.
- **History-dependent bodies** (contact friction, viscoelastic relaxation beyond `max_settle`) have no single
  equilibrium map; `warm_start: false` removes the dependence on the previous value only.
- **Cost.** Each iteration costs `1 + n_actuators` equilibria, each a settling run of 0.1-1 s of simulated time.
  Slow bodies (the arm) cost minutes. The SOFA fast path below removes the repeated solves; reduced models are not implemented.
- Not done: constraints on outputs, bounds on rates, other backends than MuJoCo and SOFA in tests (`backend=` is passed
  through, untested here for the reference).

## SOFA fast path: SoftRobots.Inverse (`--method sofa-qp`)

```bash
softsim inverse jobs/inverse_cable_cantilever_sofa.yaml                  # `method: sofa-qp` in the job
softsim inverse jobs/inverse_cable_cantilever.yaml --method sofa-qp      # or on the command line
softsim inverse jobs/inverse_cable_arm_sofa.yaml                         # two cables, tip x and z, one QP run
softsim inverse jobs/inverse_bending_sofa.yaml                           # a pressure (SurfacePressureActuator)
softsim inverse jobs/inverse_bending_angle_sofa.yaml                     # a tip tilt with a pressure
softsim inverse jobs/inverse_bending_cable_sofa.yaml                     # a pressure and a cable together
softsim inverse jobs/inverse_cable_stroke_sofa.yaml                      # a stroke-controlled cable
```

```python
from softsim.control.sofa_inverse import solve_inverse_sofa

res = solve_inverse_sofa("scenarios/actuation/cable_cantilever.yaml", {"tip_x": "2 mm"}, ["cable"],
                         {"cable": (0, "0.7 N")}, overrides=["body.mesh.size=4mm"])
```

Code: `softsim/control/sofa_inverse.py`; tests: `tests/test_sofa_inverse.py` (skipped without SOFA). The result is an
`InverseResult` like `solve_inverse`'s (`simulations` is the number of settled QP runs).

**What it is.** SOFA's own inverse solver does the inversion inside the constraint solve. The scene is the scenario's
body (same mesh, material, fixed vertices) with `QPInverseProblemSolver` replacing the Gauss-Seidel solver, each cable a
`CableActuator` (the via points a `BarycentricMapping` child of the body, `pullPoint` 1 mm behind the first via point)
with `minForce` / `maxForce` from the job's bounds, and the target a `PositionEffector` on a barycentric point at the
tip face's rest centroid. At each step the QP finds the cable forces that bring the effector to its goal at the end of
the step, within the bounds; the run is stepped until forces and outputs have stopped moving (`settle_window`,
`settle_tol` as above), and the tensions are read.

- **Units.** `CableActuator`'s force is the constraint impulse: the bounds are given to it as `force * dt` and the
  reported tension is `lambda / dt` (without the factor the bound is 1/dt times too loose: the `0.7 N` limit becomes
  7000 N and the tip flies off; a mutation of this makes the tests fail).
- **Targets.** `tip_x`, `tip_y`, `tip_z` are the effector's goal (axes chosen with `useDirections`); there must be at
  least as many axes as cables. The effector is a barycentric point and the scenario's output is the face's area-weighted
  centroid, so the goal is corrected by the remaining error after each settled run (up to `max_corrections`; the
  cantilever needed none, the error of the point against the centroid being 2 um). `bend_angle` / `chord_angle`
  are not a position: the effector goal is moved along an axis no position target uses (`angle_direction`, `+x` by
  default; `-x` is tried when `+x` leaves every actuator at its lower bound, as for a tube that curls to -x) and the
  goals of all the targets are found by a Broyden iteration on the angle / position read from the vertices (a secant
  for one angle: 2 QP runs for the 2 degree job). This works with any actuator (a pressure, a cable, both) and
  with angle and position targets together; each angle needs a free axis (at most three targets in all). It needs no
  second effector point (a pair of points on the tip face fixes the translation as well as the tilt, and a `Rigid3d`
  effector does not exist in this build: `PositionEffector`, `CableEffector`, `VolumeEffector` and
  `BarycentricCenterEffector` are the registered ones), so the angle is the scenario's own metric, exactly.
  Other outputs are refused.
- **Step.** SOFA is implicit, so the step is 2 ms (`timestep`), not the scenario's 0.1 ms; the body gets the same
  stiffness-proportional damping as the repeated solve (`settle_damping`, default 10 ms) so that it settles.
- **Pressure actuators.** A pneumatic actuator (`SurfacePressureActuator`) is built on the actuator's `cavity` region
  (a child node with the cavity triangles, mapped to the body); bounds are pressures (`cavity: [0 kPa, 10 kPa]`) and the
  targets are positions (`tip_x`, `tip_y`, `tip_z`; mind the direction a robot bends: the bending tube curls to -x, the
  PneuNet to -z). Measured: the solver's lambda is an impulse, `pressure * dt`, like the cables (the bounds go in as
  `pressure * dt`, the pressure is `lambda / dt`: a run clamped at 2 kPa gives the tip the ordinary backend's explicit
  follower pressure gives at 2 kPa, -0.1230656 mm, to 7 digits). Two traps, both found by measurement: the cavity
  triangles need the opposite winding to softsim's (else the volume shrinks and the pressure clamps at 0), and in SI units the
  QP's matrices are tiny (the pressure's entry is ~4e-12), so qpOASES returns lambda = 0 for every reachable goal and
  bangs between 0 and the bound for an unreachable one. The effector `weight` scales the objective: 1e6 (`qp_weight`) is
  the default with pressures, 1e5 and 1e7 give the same pressure. Angle targets work with a pressure (below).
- **Mixed actuators.** Pressures and cables can be solved together (one `QPInverseProblemSolver`, each actuator with
  its own impulse bounds), with `qp_weight` 1e6 as soon as one is a pressure (it does not hurt the cable's scaling).
  `scenarios/inflation/bending_cable.yaml` is the bending tube with a cable in its thin +x wall: the pressure curls
  the tip to -x and lengthens the tube, the cable pulls it to +x and shortens it, so tip x and z together fix both.
- **Stroke-controlled cables.** A cable with `control: stroke` (tension `k (L - L0 + s)`) is solved as a tension and the
  stroke read back as `s = T / k - (L - L0)` from the final vertices; the bounds are strokes (`cable: [0 mm, 3 mm]`)
  and the QP's tension bounds `k (s_bound + L - L0)` are re-set after each run at the length reached (the length
  depends on the tension, so the bound is found by a secant: 3 runs when a bound is active; a bound that is
  not active needs no iteration).
- **Outcome.** Same statuses as above. A goal beyond the bounds leaves the cable at its bound (`bounded_best`,
  `at_bounds`): no second solve is made once a force is saturated.

**Measured** (4 mm cantilever mesh, 6 mm arm, 2 / 5 ms steps; shared CPU, times indicative; wall time includes building
the scene):

| Case | QP (SOFA) | Repeated solve, SOFA backend | Speed |
| --- | --- | --- | --- |
| Cantilever, 2 degrees | 0.2699 N (angle error 1.4e-4 rad), 2 runs, 90 steps, 1.2 s | 0.2716 N (error 8.6e-5 rad), 5 equilibria, 693 steps, 9.1 s | 7.5x |
| Cantilever, tip x = 2 mm | 0.5575 N (tip 1.9977 mm), 1 run, 60 steps, 0.9 s | 0.5509 N (tip 1.9767 mm), 6 equilibria, 868 steps, 10.0 s | 11x |
| Same, 8 mm with at most 0.7 N | 0.7 N, tip 2.49 mm, `bounded_best` | - | - |
| Arm, tip x 3.994 mm and z 118.515 mm | 0.4497 / 0.1500 N (true 0.45 / 0.15 N), errors 0.02 um / 0.3 um, 1 run, 2.8 s | 0.4464 / 0.1488 N (0.8% low, `converged`, tip errors 20 um and 12 nm), 13 equilibria, 2689 steps, 47 s; before the cable fix it was `stalled`, 0.245 / 0.234 N, 28 equilibria unsettled, 336 s | 17x |
| Bending tube (2 mm mesh, 1279 points), tip x = -0.28681 mm (what 4 kPa gives) | 3999.0 Pa (true 4000, 0.03%; tip error 0.09 um), 3 runs, 110 steps of 2 ms, 32 s | 4000.3 Pa, 7 equilibria, 824 steps, 225 s | 7x |
| PneuNet (1.5 mm mesh, 4936 points), tip z = 5.3077 mm (what 5 kPa gives, 10 ms steps) | 4997.7 Pa (true 5000, 0.05%; tip error 1.5 um), 2 runs, 40 steps, 96 s | not run (the explicit backend needs minutes per equilibrium at this size) | - |
| Bending tube, tip x = -1 mm with at most 2 kPa | 2000 Pa, tip -0.1231 mm, `bounded_best` | - | - |
| Bending tube, tilt 0.013947 rad (what 4 kPa gives) | 4000.02 Pa (0.0005%; angle error 9e-8 rad), 4 runs (one on the wrong side, +x), 38 s; 3 runs, 25 s with `angle_direction: -x` | not run (225 s per position target) | - |
| Tube with a cable in the thin wall, tip x 2.9624 mm and z 39.4169 mm (4 kPa and 0.2 N) | 3983 Pa (0.4%) and 0.19992 N (0.04%), tip errors 0.1 and 0.3 um, 6 runs, 210 steps, 37 s | not run (two actuators, minutes per equilibrium) | - |
| Same, bend_angle 0.138112 rad and tip z 39.4169 mm | 4004.5 Pa (0.11%) and 0.20003 N (0.01%), 5 runs, 220 steps, 42 s | - | - |
| Same, cable limited to 0.1 N | cable 0.1 N, `bounded_best` | - | - |
| Stroke cable (300 N/m), tip x 1.14667 mm (what a 1.5 mm stroke gives, 4 mm mesh) | stroke 1.4997 mm (0.02%), 1 run, 0.5 s; for the angle target 1.4969 mm (0.2%), 2 runs | - | - |
| Same, stroke limited to 1 mm | stroke 1.0002 mm, tip 0.7693 mm (ordinary backend at a 1 mm stroke: 0.7682 mm), 3 runs, `bounded_best` | - | - |

The tensions agree to 0.6% (angle target; the two stop at residuals of 0.014 and 0.009 of a 0.01 rad scale, i.e. 0.4%
in angle) and 1.2% (position target; the repeated solve stopped 23 um short of the target, and scaling its tension by
the ratio of the tips gives 0.5517 N against the QP's 0.5575 N: 1%). The QP's tension, applied as the ordinary
backend's explicit cable force, puts the tip within 0.05 mm of the 2 mm target (tested). The arm's targets are the tip
MuJoCo reaches at 0.45 / 0.15 N on the same mesh, so the 0.07% agreement is across backends.

The pressure targets are the tips the same model reaches at a held pressure: the bending tube's from the ordinary SOFA
backend (explicit follower pressure, 2 ms steps, 59 s), the PneuNet's from a QP run clamped at 5 kPa (the explicit
backend is too slow at 1.5 mm; the clamped run was checked against it on the tube). The tube's tip moves only 0.3 mm at
4 kPa (stiff walls), so its scale is 0.01 mm, not the default 1 mm, which a run that never pressurised would satisfy.

**Why the arm stalled on the repeated solve (found and fixed).** The ordinary SOFA backend used to apply the cable as an
explicit follower force, and on the 6 mm arm that did not settle: the tip's x swung by 0.6-1.9 mm peak to peak
indefinitely (period 0.39 s, 2.6 Hz, along the mesh diagonal), at every step from 10 ms to 0.25 ms, while MuJoCo
settled. Measured on the 6 mm mesh at 0.45 / 0.15 N, Rayleigh beta 0.15 s:

| Run | Tip x peak to peak, last seconds |
| --- | --- |
| explicit cable, dt 10 / 5 / 1 / 0.25 ms | 0.59 / 0.85 / 1.9 / 3.9 mm (grows as dt shrinks; still growing at 0.25 ms) |
| explicit, no damping, 5 ms | 1.8 mm |
| explicit, beta 0.5 / 1.5 s, 5 ms | regrows to 0.67 mm / decays (0.012 mm at 3.5 s) |
| cable A alone, B alone, both at half tension (10 ms) | settle |
| both cables at twice the tension (10 ms) | 1.7 mm |
| released at 2 s (no cable) | decays in 1 s |
| force-type `CableConstraint` (SoftRobots) instead of the force field | identical to the explicit run to 4 digits |
| linear (`small`) or corotational (`large`, `svd`) elastic law, same load | settles (tip x 1.80, 4.39, 4.74 mm) |
| cable as springs in the implicit system (below), 5 / 2 / 1 ms | settles; tip x 4.0025 mm, z 118.514 mm at every step |

So it is not the step size (it gets worse as the step shrinks, and extrapolating the positions to the end of the step
for the force changes little), not the routing points (the cable's Jacobian is symmetric and positive semi-definite,
checked against finite differences), and not the constraint form of the cable (identical). It needs both cables (the S
shape) and the hyperelastic force field. The reading that fits all of it, and that the fix below supports (an inference, the hyperelastic tangent was not computed directly), is: the cable is a stabilising lateral stiffness `T (I - u u^T) / l` per segment, and under
the arm's compression (0.6 N against a column whose Euler load is about 0.03 N) it balances the elastic force field's
negative geometric stiffness. `TetrahedronHyperelasticityFEMForceField` has, we infer, that geometric part in its tangent, so the
implicit matrix (and with it the Rayleigh `beta K` damping, and implicit Euler's own numerical damping) is nearly
singular or indefinite in the bending mode, with the cable's compensating stiffness missing from it. The slowest
modes are then undamped or anti-damped whatever the step: a limit cycle. The corotational field has no geometric
stiffness in its tangent and the small-strain one no geometric effect at all, so they settle (but are wrong: 4.39 and
1.80 mm against 3.994 mm).

The backend now puts a tension-controlled cable into the implicit system: springs between the via points (mapped on the
mesh) whose rest lengths are reset every step to `length - T / k`, which gives the force `T` at the current positions
and the tangent `T (I - u u^T) / l` as stiffness, with a negligible axial `k` (1% of `T` per via-point spacing). The
arm now settles within 1 s at 5 ms to 1 ms steps (tip x 4.0025 mm, z 118.514 mm; the QP path's tip is 3.994 mm and
118.515 mm, MuJoCo's 3.994 mm: 8.5 um apart in x, and the tension is the same), and the repeated solve on SOFA now converges (table above). Stroke-controlled and anchored
cables are still explicit follower forces.

**Limits.**

- **Same model as the repeated solve, same locking.** SOFA's linear tetrahedra lock in bending like MuJoCo's flex: the
  4 mm cantilever is too stiff, and the tensions are those of the mesh (see [actuation.md](actuation.md)). The QP solves
  the model faster; it does not make the model more accurate.
- **Quasi-static.** The forces are the held values at equilibrium; the path and the time to get there depend on the
  step and the damping. The QP is a per-step solution, not a trajectory plan.
- **The effector.** A position (up to three axes) of one point. Angles are reached through that point's goal (the
  iteration above, a few extra QP runs), not as an orientation constraint of the QP; at most three targets
  (positions and angles) in all, an angle's direction must be a free axis, and the angle direction is a single sign for
  all angles. A tilt of an unreachable direction (e.g. `+x` for a tube that only curls to -x) is detected only
  when the actuators sit at their lower bound. Output constraints are not built. Fewer targets than actuators is
  refused (the repeated solve refuses it too).
- **Scope.** Cables (tension- or stroke-controlled, no anchors) and pressures, alone or together, on one soft body with a
  `tip` / `top` region; force actuators, rigid bodies, contact, viscoelasticity, prescribed motion and moving mounts are
  refused or not built. A stroke cable's tendon is the scenario's (`stiffness`) and the stroke read back assumes
  the ordinary backend's model, tension = `k (L - L0 + s)` (an explicit follower force there; the QP has the
  tension as a constraint force).
- **One equilibrium.** As above: the QP follows one branch from the rest state; a snapping structure may have others.
- **Bounds on the forces** (tensions, pressures) and, for stroke cables, on the stroke through the iteration above;
  `maxDispVariation` / `CableActuator`'s own displacement limits are not used (they would model a rigid tendon,
  not the series spring).

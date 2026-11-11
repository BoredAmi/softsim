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
  (one target, one cable) are not a position: the effector goal's x is moved by a secant iteration on the angle read
  from the vertices (2 QP runs for the 2 degree job). Other outputs are refused.
- **Step.** SOFA is implicit, so the step is 2 ms (`timestep`), not the scenario's 0.1 ms; the body gets the same
  stiffness-proportional damping as the repeated solve (`settle_damping`, default 10 ms) so that it settles.
- **Outcome.** Same statuses as above. A goal beyond the bounds leaves the cable at its bound (`bounded_best`,
  `at_bounds`): no second solve is made once a force is saturated.

**Measured** (4 mm cantilever mesh, 6 mm arm, 2 / 5 ms steps; shared CPU, times indicative; wall time includes building
the scene):

| Case | QP (SOFA) | Repeated solve, SOFA backend | Speed |
| --- | --- | --- | --- |
| Cantilever, 2 degrees | 0.2699 N (angle error 1.4e-4 rad), 2 runs, 90 steps, 1.2 s | 0.2716 N (error 8.6e-5 rad), 5 equilibria, 693 steps, 9.1 s | 7.5x |
| Cantilever, tip x = 2 mm | 0.5575 N (tip 1.9977 mm), 1 run, 60 steps, 0.9 s | 0.5509 N (tip 1.9767 mm), 6 equilibria, 868 steps, 10.0 s | 11x |
| Same, 8 mm with at most 0.7 N | 0.7 N, tip 2.49 mm, `bounded_best` | - | - |
| Arm, tip x 3.994 mm and z 118.515 mm | 0.4497 / 0.1500 N (true 0.45 / 0.15 N), errors 0.02 um / 0.3 um, 1 run, 2.8 s | `stalled`, 0.245 / 0.234 N, 28 equilibria unsettled, 336 s | 120x |

The tensions agree to 0.6% (angle target; the two stop at residuals of 0.014 and 0.009 of a 0.01 rad scale, i.e. 0.4%
in angle) and 1.2% (position target; the repeated solve stopped 23 um short of the target, and scaling its tension by
the ratio of the tips gives 0.5517 N against the QP's 0.5575 N: 1%). The QP's tension, applied as the ordinary
backend's explicit cable force, puts the tip within 0.05 mm of the 2 mm target (tested). The arm's targets are the tip
MuJoCo reaches at 0.45 / 0.15 N on the same mesh, so the 0.07% agreement is across backends.

**Why the arm stalls on the repeated solve.** The ordinary SOFA backend applies the cable as an explicit follower force.
On the 6 mm arm that force does not settle: the tip's x oscillates by about +-0.5 mm indefinitely at 1, 2 and 5 ms steps
(not investigated further; MuJoCo with its damping settles). The constraint-based cable has no such limit cycle, which
is a reason to use the QP path for slender cable robots on SOFA besides speed.

**Limits.**

- **Same model as the repeated solve, same locking.** SOFA's linear tetrahedra lock in bending like MuJoCo's flex: the
  4 mm cantilever is too stiff, and the tensions are those of the mesh (see [actuation.md](actuation.md)). The QP solves
  the model faster; it does not make the model more accurate.
- **Quasi-static.** The forces are the held values at equilibrium; the path and the time to get there depend on the
  step and the damping. The QP is a per-step solution, not a trajectory plan.
- **The effector.** A position (up to three axes) of one point. An angle needs the secant wrapper (one cable, one
  target); several angle or mixed targets, tip orientation (a `Rigid3d` effector) and output constraints are not
  built. Fewer axes than cables is refused (the repeated solve refuses it too).
- **Scope.** Tension-controlled cables on one soft body with a `tip` / `top` region; stroke-controlled cables, pressure
  and force actuators, rigid bodies, contact, viscoelasticity, prescribed motion and moving mounts are refused or not
  built. `SurfacePressureActuator` was not tried.
- **One equilibrium.** As above: the QP follows one branch from the rest state; a snapping structure may have others.
- **Bounds only on the forces.** `maxDispVariation` / displacement limits of `CableActuator` are not exposed.

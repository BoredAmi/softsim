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
  Slow bodies (the arm) cost minutes. SOFA's SoftRobots.Inverse fast path and reduced models are not implemented.
- Not done: constraints on outputs, bounds on rates, other backends than MuJoCo in tests (`backend=` is passed through,
  untested here for SOFA and the reference).

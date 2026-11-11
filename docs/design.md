# Design optimisation

`softsim optimize <job.yaml>` tunes design parameters of a scenario (geometry, a cable offset, a material
value, any dotted config key) to maximise, minimise or hit a number from the run's summary, optionally under
limits on other summary numbers. It reuses the calibration engine's patterns: a YAML job, bounds, a
resumable evaluation log, spawned worker processes, a report. Code: `softsim/design/optimize.py`.

## A job

```yaml
scenario: ../scenarios/actuation/cable_cantilever.yaml    # relative to the job file
backend: mujoco                                            # optional
overrides: [body.mesh.size=5mm]                            # applied before the design values
parameters:
  "actuators[0].path[0][0]":                               # dotted key, list indices allowed (also nested)
    initial: 3 mm
    bounds: [0.5 mm, 4.5 mm]
    tie: ["actuators[0].path[1][0]"]                       # more keys that take the same value
objective: {metric: final_bend_angle, target: 0.06 rad}   # or goal: maximize | minimize; `per: <metric>` divides
constraints:                                               # optional; a metric or a design key, min and/or max
  - {metric: cable_stroke, max: 4 mm}
max_evaluations: 30
output: ../runs/design/cantilever_target_angle
```

- **Parameters**: `bounds` and `initial` are numbers (SI) or unit strings; mixing dimensions is an error.
  `log: true` searches in log space, `integer: true` rounds (e.g. `segments`), `tie` sets further keys to the
  same value (the cantilever's cable runs from `path[0]` to `path[1]`, both at offset `d`). A key that does not
  exist in the scenario is an error: a typo must not silently optimise nothing.
- **Objective**: any key of the scenario's summary (`softsim run` prints them: `final_bend_angle`,
  `final_tip_x`, `cable_stroke`, `peak_max_radius`, ...). `goal: maximize|minimize`, or `target: v`
  (minimise |metric - v|). `per: other_metric` optimises the ratio (e.g. angle per unit tension).
- **Constraints**: bounds on summary metrics (or on design keys).
- Examples: `jobs/cantilever_target_angle.yaml` (match a tip angle with the cable offset) and
  `jobs/cable_finger_slots.yaml` (slot depth and width of the cable finger, under a stroke limit).

Outputs in `output`: `report.yaml` (best design, objective, constraint values, whether the best is feasible,
which parameters sit on a bound, evaluations simulated/failed, wall time), `history.parquet` (every
evaluation in order, with every summary metric), `best_overrides.txt` (the best design as `key=value`
lines for `softsim run --set`; a nested list index is written as the whole inner list) and `evaluations.jsonl`.

## Method

Nelder-Mead (scipy, derivative-free, bounded) in the unit cube of the parameters (log or linear scale),
started at `initial` with a simplex step of 25 % of each range (`simplex_step`). The initial simplex is
evaluated first, in parallel with `--jobs`; later evaluations are sequential (Nelder-Mead is). No new
dependency.

**Constraints are exterior penalties**, not a constrained solver: cost = f / |f0| + 100 x sum of the relative
violations (how far a metric lies beyond its bound, divided by the bound), with f0 the objective at the
first evaluation. The penalty is steep but finite, so the simplex can cross an infeasible region. The
reported best is chosen **among feasible evaluations only**; the penalised cost only steers the search. If no
evaluation is feasible, the report has `feasible: false`, a `warning`, the violated constraints with their
violation, and the least-violating design; the CLI prints the warning and exits with status 2. Infeasibility
is never hidden. A failed simulation (crash, unstable, non-finite objective) costs 1000 and is counted in
`failed`.

**Log and resume**: every evaluation is appended to `evaluations.jsonl` with its full summary. Re-running the
job replays them (Nelder-Mead is deterministic), so a stopped job continues, and raising `max_evaluations`
extends a finished one. Entries are keyed by the scenario (its file and the resolved config, `extends`
included), softsim's own code, the overrides (and backend) and the parameter keys, so a code change starts
afresh; objective, constraints and bounds are not in the key, since they only re-weigh stored summaries.

## Checked against known optima

`tests/test_optimize.py` (MuJoCo, cable cantilever at 6 mm mesh and 0.3 ms steps, about 2 s per simulation;
the angle grows monotonically with the cable offset d because the cable puts a moment T d on the beam):

- maximising the tip angle for d in [1, 4] mm ends at the upper bound (4 mm, `at_bounds` lists it);
- matching a 0.05 rad target gives the d that nine bisection steps on the same forward map give, to 20 um
  (bracket 6 um); the objective reaches the target to 0.2 mrad;
- a stopped job (5 evaluations) continues to 8 without re-simulating logged points (the number of
  simulations is counted); an identical rerun simulates nothing; a changed override does not reuse the log;
- an unreachable constraint (cable stroke >= 1 m) returns `feasible: false` with the violation; a cap on the
  angle itself gives a feasible best just below the cap;
- bad job files (unknown keys, bad or mixed-unit bounds, initial outside bounds, a typo in a design key,
  missing goal, empty constraint, unknown metric) raise `ConfigError` with the reason.

Mutation checks: a log that is never read fails the resume test; flipping the maximise sign fails the
upper-bound test.

## Measured examples

**Cantilever, target tip angle** (`jobs/cantilever_target_angle.yaml`; MuJoCo, 5 mm mesh, target 0.06 rad):
cable offset d = 2.127 mm found in 30 evaluations (26 simulations, 109 s on one core, about 4 s each); the
objective reaches 0.060001 rad. The angle at this d on other models (same scenario, measured):

| model | tip angle at d = 2.127 mm |
|---|---|
| MuJoCo, 5 mm mesh (the one optimised) | 0.0600 rad |
| MuJoCo, 2 mm mesh (scenario default) | 0.134 rad |
| `fem` reference, 5 mm mesh | 0.198 rad |
| Euler-Bernoulli, T d L / (E I) | 0.196 rad |

The optimum on the coarse MuJoCo mesh is 3.3 times off the reference: that d bends the real beam to 0.2 rad,
not 0.06. This is the locking bias of the limits section below, measured; re-running the job with
`backend: fem` (and a coarser time step) gives the physical d.

**Cable finger, slot geometry** (`jobs/cable_finger_slots.yaml`; MuJoCo, 3 mm mesh, `--jobs 2`):
maximise the final bend angle over slot depth in [1, 4] mm and slot width in [1, 5] mm, subject to a cable
stroke of at most 4 mm. Best: depth 4.0 mm (the upper bound: deeper slots bend more, and 4.5 mm would cut the
tendon's web), width 4.875 mm, bend angle 0.278 rad against 0.160 rad at the starting design (3 mm, 3 mm);
the stroke there is 1.84 mm, so the constraint is inactive. 30 evaluations (21 distinct designs, the rest
revisits served from the log), 437 s for the last run (earlier evaluations were replayed from a first,
stopped run). An earlier version of this job allowed depths up to 6 mm: the scenario refuses a slot deeper than
the tendon offset, those evaluations were reported as `FAILED` with the reason, and the bounds were fixed.
The width optimum is not sharp: the bend angle jitters by about 0.01 rad between neighbouring widths
(4.0 mm: 0.260, 4.5 mm: 0.271, 4.625 mm: 0.258, 4.75 mm: 0.276, 5.0 mm: 0.270) because each design is
re-meshed, so the optimiser sees mesh noise of about 4 % and the width is determined only to about 0.3 mm.
The winner has not been checked on the `fem` reference (a 3D finger run there takes much longer).

## Limits

- **The backend's error bar applies to the optimum.** The optimiser exploits the simulator, including its
  errors. MuJoCo's linear tetrahedra lock in bending: the coarse cantilever above bends 0.084 rad where
  Euler-Bernoulli gives 0.28 rad, so a design tuned to a target angle on MuJoCo is a design for the locked
  beam, not for the real one (the table above: 3.3 times off). **Check the winner on the
  `fem` reference** (`softsim run <scenario> --backend fem --set ...` with the `best_overrides.txt` lines) and
  re-optimise, or calibrate the backend's effective material first (`softsim transfer`).
- **Local, deterministic search.** Nelder-Mead finds the optimum nearest the start; use several starts for
  multimodal problems. It stalls on the plateaus of integer parameters (they are rounded), so scan those by
  hand. Dimensions beyond ~6 parameters need many more evaluations than the default 40.
- **Penalty, not a constrained solver**: with a loose `max_evaluations` the best feasible point can lie
  slightly inside the true constrained optimum.
- **Geometry must stay valid**: a design the mesher or backend refuses is a failed evaluation (cost 1000);
  keep the bounds inside the shape's valid range (e.g. the slots must not exceed the body).
- **Mesh noise**: a design change re-meshes the body, so the objective carries discretisation jitter (about
  4 % on the finger); the optimum is not resolved finer than that. Use a mesh fine enough that the jitter is
  below the differences you care about.
- Each evaluation is a full simulation; there is no surrogate and no gradient.

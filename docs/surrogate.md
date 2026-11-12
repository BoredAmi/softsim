# Reduced-order surrogates for real-time control (M11)

A repeated equilibrium solve ([control.md](control.md)) takes tens of seconds: every iteration settles a simulation.
A controller running at 50 Hz cannot wait for that. `softsim surrogate` samples the equilibrium map once, fits a smooth
model of it, and `softsim inverse --method surrogate` then answers in milliseconds. Code:
`softsim/control/surrogate.py`; tests: `tests/test_surrogate.py`.

```bash
softsim surrogate jobs/surrogate_cable_cantilever.yaml      # sample, fit, validate -> runs/surrogate/.../surrogate.json
softsim inverse jobs/inverse_cable_cantilever_surrogate.yaml --method surrogate
softsim surrogate jobs/surrogate_cable_arm.yaml             # two cables, tip x and z
softsim inverse jobs/inverse_cable_arm_surrogate.yaml --method surrogate
```

## Method

1. **Design of experiments.** `inputs` (name: `[lower, upper]`, with units) are sampled on a seeded design: the corners
   of the box (up to 4 inputs; so the bounds are interpolated, not extrapolated) plus a Latin hypercube
   (`design: lhs`), or a regular `grid`. `holdout: m` adds m more points from another seeded hypercube that the fit
   never sees.
2. **Samples.** Each point is brought to equilibrium by the same `Equilibrium` object `softsim inverse` uses (constant
   actuators, stiffness-proportional settling damping, settled when the outputs stop moving; `settle_*` keys and `scales`
   are those of the inverse job) and the chosen `outputs` (any column of the scenario: `tip_x`, `tip_z`, `bend_angle`,
   ...) are recorded. Every sample is appended to `samples.jsonl` in the output folder, keyed by the resolved scenario
   (`extends` included), overrides, backend, settling options, softsim's code hash and the sample's inputs: a stopped
   job resumes, a rerun simulates nothing, a changed scenario or code does not reuse stale samples.
3. **Fit.** `model: poly` (total-degree polynomial in the inputs scaled to [-1, 1], least squares; `degree: auto`
   picks the degree with the lowest k-fold error, at most 0.7 terms per sample) or `model: rbf` (cubic radial basis
   with a linear tail). numpy only.
4. **Validation (held out, never the fit residual).** `cv`: k-fold cross-validation over the fitting samples
   (`folds`); `holdout`: error of the model fitted without the extra points. RMS and max, in SI units and relative to
   the output's range over the samples. The saved model is then refitted on all points.
5. **The file.** `surrogate.json` holds `inputs` (name, SI unit, bounds), `outputs` (name, unit, range), the model
   (exponents and coefficients, or centres and weights), `validation`, `scenario_hash`, `code_hash` and the sample key.
   It needs nothing but numpy; for a polynomial, with `z = 2 (x - lower) / (upper - lower) - 1`:

```python
import json, numpy as np
s = json.load(open("surrogate.json")); m = s["model"]
lo = np.array([i["lower"] for i in s["inputs"]]); hi = np.array([i["upper"] for i in s["inputs"]])
def predict(x):                                           # x in SI, in the order of s["inputs"]
    z = 2 * (np.asarray(x) - lo) / (hi - lo) - 1
    return np.array(m["coefficients"]) @ np.prod(z ** np.array(m["exponents"]), axis=1)
```

(`softsim.control.surrogate.Surrogate.load(path).predict(x)` does the same for polynomial and RBF models and refuses
inputs outside the sampled box unless `extrapolate=True`.)

## The inverse on the surrogate

`softsim inverse job.yaml --method surrogate --surrogate file.json` (or a `surrogate:` key in the job) solves the same
scaled least-squares problem as the repeated solve on the surrogate, with `scipy.optimize.least_squares` from the box
centre and 7 seeded random starts (`starts`). The job's `actuators` bounds tighten the surrogate's box; inputs not
solved for go in `fixed`. `verify: true` runs one forward equilibrium (the job's `config`, `overrides`, settling
options; refused if the scenario hash differs from the surrogate's) at the solution and reports the **true** error and
how far the surrogate was off. The status is `converged`, `bounded_best` or `stalled` as for the repeated solve.

## Measured (MuJoCo; the machine was shared, times are indicative)

Cable cantilever, 4 mm mesh, tension 0 to 0.7 N, outputs `bend_angle`, `tip_x`, `tip_z`: 8 samples plus 4 held out,
cubic polynomial, 56 s of simulation.

| Held-out error | RMS | Max | Relative to range |
| --- | --- | --- | --- |
| `bend_angle` | 1.8e-7 rad | 2.3e-7 rad | 3e-6 |
| `tip_x` | 7e-9 m | 8e-9 m | 4e-6 |

Two-cable arm (`jobs/inverse_cable_arm.yaml` settings), 0 to 0.5 N per cable, 20 samples plus 6 held out, cubic
polynomial (10 terms), 225 s of simulation:

| Held-out error | RMS | Max | Relative to range |
| --- | --- | --- | --- |
| `tip_x` | 21 um | 36 um | 0.12% |
| `tip_z` | 1.1 um | 1.7 um | 0.05% |

Inverse via the surrogate against the repeated solve and the truth:

| Case | Tension | Error at the verifying simulation | Solve time |
| --- | --- | --- | --- |
| Cantilever, 2 degrees: surrogate | 0.2709 N | 3.2e-7 rad | 26 ms |
| same, repeated solve (5 simulations) | 0.2716 N | 8.7e-5 rad | 22 s |
| Arm, targets from 0.45 / 0.15 N: surrogate | 0.4500 / 0.1500 N | 0.53 um in x, 0.18 um in z | 38 ms |
| same, repeated solve (13 simulations) | 0.4479 / 0.1487 N | 22 um in x, 7.5 um in z | 57 s |

The surrogate finds the known arm tensions to 0.01%, closer than the repeated solve, whose answer is limited by its
own stopping tolerance. The speed-up of a solve is ~850x (cantilever) and ~1500x (arm); one evaluation of the model is
~40 us in Python. It is paid for with the samples: 56 s (cantilever) and 225 s (arm), the cost of 3 and 4 repeated
solves, so a surrogate pays off from the fourth target on.

**Linear dynamic model (optional, `dynamics:`).** DMD with control, `s_{k+1} = A s_k + B (u_k - u0)`, with `s` the
outputs minus their rest value and one earlier sample stacked (`delay: 1`, which gives it a velocity), fitted by least
squares on constant-input step responses from rest (3 training steps, inputs from a seeded hypercube) and checked by
open-loop rollout on 2 held-out steps. Cantilever, 0.3 s: held-out rollout RMS 0.37% of the travel for `bend_angle`
(2.7e-4 rad), 0.5% for `tip_x`, 1.1% for `tip_z`. The state is the outputs, not the full state; it models the
damped step response the settling damping produces, not the free vibration.

## Limits

- **A surrogate inherits the backend.** It reproduces MuJoCo's equilibrium map, locking included (the 4 mm mesh is
  about 3 times too stiff in bending, [actuation.md](actuation.md)): tensions found are the backend's, not physical
  ones. Fit it on a transferred card and a checked mesh, or on the reference backend, for physical values.
- **Inside the box only.** Predictions outside the sampled bounds are refused; the inverse is bounded by it.
- **Smooth maps.** A polynomial cannot follow a kink (the unsigned `bend_angle` of the arm folds at zero) or a snap;
  the validation error shows it, so read it. Compare `degree` and `model: rbf` by the held-out error.
- **Curse of dimension.** The number of terms grows as `C(d + p, p)`; beyond about 4 actuators use `rbf` or more samples.
- **Equilibria only.** The static surrogate has no time; the DMD model is linear, fitted on few steps, and its error
  grows with the nonlinearity and the distance from the training steps.
- **Reuse.** The sample log is keyed to the code hash, so any change to softsim makes the next run resimulate.

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

## Closed-loop control on the surrogate

`softsim/control/controller.py` (tests: `tests/test_controller.py`) turns the file into a controller you call once per
control period, with no ROS and no simulator in it:

```python
from softsim.control.controller import Controller

ctl = Controller.from_json("surrogate.json", dt=0.02, ki=6.0, outputs=["bend_angle"])   # 50 Hz
ctl.reset()
u = ctl.step(measured={"bend_angle": 0.0}, target={"bend_angle": 0.05})    # SI, in the order of ctl.inputs
```

* **Feedforward.** The actuation that puts the target at equilibrium on the surrogate: bounded least squares,
  warm-started from the last solution (a multi-start solve only if that is poor); about 0.1 ms per step.
* **Feedback.** An integral correction in actuator space, `v += ki dt (J^T J + lambda^2 I)^-1 J^T e` in scaled
  coordinates (`J` the surrogate's Jacobian at the applied actuation, `e = target - measured`; damped least squares, so
  two cables that move the tip alike do not blow up). Anti-windup by clamping: feedforward + `v` stays inside the
  bounds (the surrogate's box, or tighter `bounds=`), and `v` inside `max_correction` of the range, so an unreachable
  target winds nothing up (tested: after 400 saturated steps the controller is back on a reachable target in the same
  number of steps as from rest). `mode="ff"` switches the feedback off.
* **MPC-lite (`mode="mpc"`, needs `dynamics:` in the surrogate job).** An offset-free step on the DMD model: the state is
  built from the last two measured outputs, the one-step prediction error is held over the horizon as a disturbance,
  and a bounded linear least-squares problem over `horizon` periods (tracking error plus `move_weight` times the
  moves) gives the first actuation. `dt` must equal the model's sample period. Tested on a linear plant 15 % off the
  model (converges to 1e-5 of the target, within bounds); **not** evaluated on MuJoCo, because the stored
  models sample at the physics step (1 ms), not at a control rate (set `dynamics.sample_every` to the control period to
  try it), and a 3-step DMD fit is coarse.

**Measured on MuJoCo** (`tools/eval_controller.py cantilever|arm`, shared machine). The surrogates are the two
above (`jobs/surrogate_cable_cantilever.yaml`, `jobs/surrogate_cable_arm.yaml`, nominal body); the plant is the same
scenario with the Gymnasium environments' settings and a deliberate mismatch. Cantilever: 4 mm mesh, 1 ms step,
**50 Hz**, bend-angle steps 0 to 0.03, to 0.06, to 0.04 rad, 2 s each, and a ramp 0.02 to 0.06 rad over 4 s. Arm:
6 mm mesh, 0.5 ms step, **20 Hz**, tip steps of 4 to 8 mm, 3 s each, and a 5 s ramp between two of them. Settling time:
the last time the error is outside 5 % of the step; steady-state error: mean of the last 20 % of the hold (the norm
over the outputs); ramp error: mean tracking error over the second half of the ramp.

| Plant | | Settling (s), 3 steps | Steady-state error | Ramp error |
| --- | --- | --- | --- | --- |
| cantilever, nominal | feedforward | 0.04 / 0.02 / 0.02 | < 1e-7 rad | 2.0e-5 rad |
| | feedback | 0.16 / 0.18 / 0.18 | < 1e-7 rad | 1.4e-4 rad |
| cantilever, E +20 % | feedforward | never | 5.0 / 9.9 / 6.6 mrad (17 to 33 %) | 9.1 mrad |
| | feedback | 0.02 / 0.02 / 0.02 | 1e-7 rad | 8.7e-5 rad |
| cantilever, E -20 % | feedforward | never | 7.4 / 14.7 / 9.9 mrad | 13.6 mrad |
| | feedback | 0.28 / 0.28 / 0.28 | < 1e-7 rad | 3.5e-4 rad |
| cantilever, 6 mm mesh | feedforward | never | 2.2 / 4.2 / 2.8 mrad | 3.9 mrad |
| | feedback | 0.06 / 0.06 / 0.06 | < 1e-7 rad | 5.2e-5 rad |
| arm, nominal | feedforward | 0.40 / 0.45 / 0.55 | 0.010 / 0.003 / 0.007 mm | 0.49 mm |
| | feedback | 0.45 / 1.65 / 1.70 | 0.001 / 0.026 / 0.047 mm | 0.21 mm |
| arm, E +20 % | feedforward | never | 0.66 / 0.70 / 1.82 mm | 1.9 mm |
| | feedback | never (0.63 mm) / 1.10 / 1.20 | 0.63 / 0.014 / 0.015 mm | 0.13 mm |
| arm, E -20 % | feedforward | never | 0.98 / 1.03 / 2.70 mm | 1.6 mm |
| | feedback | 2.05 / 2.20 / 1.70 | 0.10 / 0.20 / 0.084 mm | 0.37 mm |

So a 20 % stiffness error (or a coarser mesh) leaves feedforward 17 to 33 % off on the cantilever and 1 to 3 mm off
on the arm, and the integral feedback brings both to the noise floor of the settled simulation (cantilever, 1e-7 rad)
or to 0.01 to 0.2 mm (arm, whose 1 Hz first mode and the strong settling damping make it slow: with `ki = 3` per
second the correction has a time constant of ~0.3 s and a hold of 3 s ends before the E -20 % plant has converged).
The first step of the E +20 % arm shows the same effect (0.63 mm left). The controller costs 0.4 to 0.6 ms per step
(feedforward alone 0.15 ms), under 3 % of a 20 ms period. Note what feedback costs when the model is right: on the
nominal plants it only adds lag (settling 0.16 s against 0.04 s, 1.7 s against 0.5 s), because the integrator reacts
to the dynamic lag of the body, which the static map does not know; use a lower `ki` or `mode="ff"` when the surrogate
is trusted. The gain `ki` must stay well under the plant's bandwidth (`ki dt` below ~0.5); raise it only after checking
for overshoot.

**Limits of the controller.** It is only as good as the surrogate's Jacobian (a wrong gain changes the speed, not the
final value, but a wrong sign diverges); it tracks quasi-static targets (a ramp is followed with a small lag, a fast
reference is not, since the model has no time except in the DMD mode); the surrogate's box is a hard bound; it does not
estimate sensor bias or noise (filter `measured` first); nothing here handles contact or a payload that changes the
map. The loop above is simulated with the true state read exactly each period.

**ROS 2.** `ros2/softsim_ros/softsim_ros/controller_node.py` (`ros2 run softsim_ros controller_node --ros-args -p
surrogate:=... -p use_sim_time:=true`, [ros2.md](ros2.md)) wraps the controller: it reads
`/softsim/metrics/<output>` and `~/target/<output>`, and publishes `/softsim/actuators/<name>/command` on a timer.
It is a thin wrapper, but it has **not been run** (no ROS in the development environment): a test only parses it.

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

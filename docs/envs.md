# Gymnasium environments (M12, CPU groundwork)

A reach task on any actuation scene as a [Gymnasium](https://gymnasium.farama.org) environment: the action is the
actuator values, the observation a list of scenario metrics, the reward a smooth distance to a target. Code:
`softsim/envs/`; tests: `tests/test_envs.py`. The extra is optional: `pip install "softsim[rl]"` (gymnasium >= 0.29, scipy).
Without it `import softsim.envs` works and `softsim.envs.make_env` raises an ImportError that says what to install.

```python
import gymnasium as gym
import softsim.envs

softsim.envs.register_envs()              # registers the ids below (Gymnasium 0.x also does it on `import gymnasium`; 1.x does not)

env = gym.make("softsim/CableBeamReach-v0", randomize=True)
obs, info = env.reset(seed=0)             # redraws the material and the target
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())   # tension in N
```

| id | config | actions | task |
|---|---|---|---|
| `softsim/CableBeamReach-v0` | `scenarios/envs/cable_beam_reach.yaml` | cable tension, 0-0.7 N | tip angle (`bend_angle`) to a target drawn from 0.02-0.07 rad, tolerance 0.004 rad |
| `softsim/CableArmReach-v0` | `scenarios/envs/cable_arm_reach.yaml` | cables A and B, 0-0.5 N each | tip (x, z) to (3.994 mm, 118.515 mm), tolerance 0.4 / 0.3 mm |
| `softsim/PneuNetBend-v0` | `scenarios/envs/pneunet_bend.yaml` | chamber pressure, 0-25 kPa | tip bend angle to a target drawn from 0.1-0.3 rad, tolerance 0.02 rad |
| `softsim/ReducedGripperLift-v0` | `softsim/envs/gripper.py` (no YAML) | finger pressure 0-40 kPa and palm height 0-0.06 m | grasp a box and lift it 0.04 m, hold it to 4 s |

Your own: write a YAML next to a scenario (the two files above are the templates) and call
`softsim.envs.make_env("my_env.yaml", reward_fn=None)`. The config keys are in the docstring of
`softsim.envs.env.SoftsimEnv`.

## Pneumatic finger: `PneuNetBend-v0`

The PneuNet of [actuation](actuation.md) with thicker walls (3 mm walls and layer, 6 mm base, 20 mm high) so that a
3 mm mesh is allowed; the validated 1.5 mm finger takes about 5 s per 20 ms step, which is useless for control. Physics
step 5 ms (30 solver iterations), 50 ms per action, 20 actions per episode, Rayleigh damping. This is a coarse stand-in
with MuJoCo's values, not the validated finger: settled it bends 0.09 rad at 10 kPa and 0.4 rad at 25 kPa, and it balloons
above about 30 kPa, so the pressure bound is 25 kPa. The finger creeps towards equilibrium over ~0.3 s, so the observation
is `[bend_angle, tip_x, tip_z, d(bend_angle)/dt, pressure, target]`: the rate is the velocity the policy needs
(`observation_rate:` in the YAML, any metric). Randomisation: Ecoflex modulus +-20 %.

## Reduced gripper: `ReducedGripperLift-v0`

The reduced gripper (`softsim.reduced`, finger tables `softsim/reduced/wang_finger.json`) is its own MuJoCo model of link
chains, so it is a separate small class, `softsim.envs.gripper.ReducedGripperLiftEnv`, with the same conventions. It
is the fast env: about 150 steps per second of a 50 ms action, 8x real time.

* **Action** `[pressure Pa, palm height m]`, both ramped linearly from the previous value over the control period.
* **Observation** (7): the held pressure and lift command, the palm's actual rise, the object's rise above the table, its
  vertical velocity and horizontal offset (x, y) from the palm axis.
* **Reward** per step: `min(object_rise, 0.04) / 0.04 - 0.05 * pressure / 40 kPa`; 80 steps (4 s) then truncation.
  `info["held"]` / `info["success"]` is `softsim.scenarios.grasp.is_held` on the rise history, evaluated at the end
  (and every fourth step).
* **Randomisation**, on by default (`randomize=False` fixes the means): object mass 8-20 g and friction 0.7-1.2, uniform,
  from the `reset(seed=)` generator; `reset(options={"mass": ..., "friction": ...})` fixes them. The object is a box
  (`shape="cylinder"` also works). The fitted finger holds this range at 30 kPa or more and slips below 20 kPa; a 50 g
  box is never held by it, so the range stops at 20 g.
* The rise can exceed the palm's: curling fingers push the object up. The reward only sees the object.

## Definitions

* **Action**: a `Box` over the listed actuators, in order, in SI (N for a cable tension or a force, m for a stroke,
  Pa for a pressure; bounds are written with units in the YAML). Values are clipped to the box and applied with
  `Simulation.apply_controls`, then held for `control_period` (e.g. 20 ms = 20 physics steps of 1 ms) while the
  simulation runs. After a reset every actuator is at its lower bound.
* **Observation**: `float32` vector: the metrics listed under `observation` (any scenario metric: `tip_x`, `tip_z`,
  `bend_angle`, `cable_length`, ...; SI, radians), then optionally the held actuator values (`include_actuators`) and
  the target (`include_target`). Nothing is normalised: scale it in a wrapper if the learner needs it.
* **Reward**: `-||(metric - target) / scale||` over the task metrics (default scales 1 mm for lengths, 0.01 rad for
  angles, set in the YAML), plus `success_bonus` (default 1) on the step that reaches the target, `failure_penalty`
  (default -10) on a numerical failure. `reward_fn(env, metrics, target)` replaces it.
* **Termination**: every task metric within its tolerance of the target (`info["success"]`), or a backend
  exception (divergence, solver failure): then `info["failure"]` holds the message and the last observation is
  returned. **Truncation**: after `max_steps` actions.
* **Info**: all scenario `metrics`, simulation `time`, `target`, `error`, and the drawn `material` values.

## Domain randomisation

`reset(seed=s)` seeds everything random. With randomisation on (`randomize=True` in `gym.make`, or no
`enabled: false` in the config) each reset rebuilds the body with a redrawn material and redraws the target:

* `randomize.card_uncertainty: true`: parameters drawn from the card's `uncertainty` with
  `softsim.materials.sampling` (the calibrated error bars; needs a card that states them);
* `randomize.ranges`: `{"elastic.youngs_modulus": {rel: 0.2}}` (uniform, +-20 %), `{abs: ...}` or `[low, high]`,
  applied on top for what a card does not cover (the example envs use this: their inline materials have no bars).

The drawn values are in `info["material"]`. A reset with `options={"target": [...]}` fixes the target and
`options={"material": False}` skips the redraw.

## Vectorised use

`gymnasium.vector.SyncVectorEnv([lambda: gym.make(id) for _ in range(n)])` works (tested: batched spaces, per-env seeds,
autoreset). It steps the environments one after the other in one process. The batched GPU version (many scenes in one
call on MuJoCo Warp/MJX or similar) is M12 proper and needs the GPU machine.

## Speed and limits

Measured on this laptop (MuJoCo, one process, 2 threads), per `step` including the physics:

| env | physics per step | steps/s | `reset` with randomisation |
|---|---|---|---|
| CableBeamReach (4 mm mesh, dt 1 ms) | 20 ms | about 21 | 0.04 s (the model is rebuilt) |
| CableArmReach (6 mm mesh, dt 0.5 ms) | 50 ms | about 3.3 | 0.08 s |
| PneuNetBend (3 mm mesh, dt 5 ms) | 50 ms | about 7 (0.14 s) | 1 s (the model is rebuilt) |
| ReducedGripperLift (1 ms, 50 substeps) | 50 ms | about 150 | 3 ms (rebuilt every reset) |

That is about 0.4x real time for the beam, 0.16x for the arm, 0.35x for the PneuNet and 8x for the reduced gripper:
enough for a controller, tests and small experiments (the gripper even for a few thousand episodes), not for
sample-hungry deep RL on the FE-based ones.

* CPU MuJoCo only. Its values are MuJoCo's, not physical, unless a transferred card is used: on the coarse 4 mm mesh
  linear tetrahedra lock and the beam reaches 0.089 rad at 0.7 N where Euler-Bernoulli gives 0.28 rad
  ([actuation](actuation.md)). A policy trained here learns this model, not the real beam.
* The environments add Rayleigh damping (equilibrium unchanged) so the beam settles in ~0.1 s and the arm in ~1 s; the
  arm's response lags the action by that much, so it needs a 2 s episode.
* Only the first soft body's material is redrawn (`randomize.body` picks another); zones keep their cards.
* The default arm target is one reachable point; randomised targets need a reachable set (see
  [inverse control](control.md)).
* The PneuNet env is a thick-walled, coarse variant; the gripper env learns the reduced model (finger tables from the
  reference, no chambers, no local deformation around the object).
* A controller is a callable on the observation: a proportional-integral one that adds `5 N/rad x error` to the
  tension reaches the beam's 0.05 rad target in 2 steps (`tests/test_envs.py`).

## Training baseline

`tools/train_baseline.py` is a numpy-only cross-entropy method over a linear policy
(`action = low + (high - low) * sigmoid(W z + b)`, `z` the standardised observation), scored by mean return over a few
seeded episodes per candidate and compared with uniformly random actions on 20 held-out seeds. It needs no torch;
for deep RL use stable-baselines3 or similar on the same ids (optional, not a dependency of softsim).

```
python tools/train_baseline.py                                       # CableBeamReach-v0
python tools/train_baseline.py --env softsim/ReducedGripperLift-v0 --generations 8 --pop 16 --episodes 3
python tools/train_baseline.py --env softsim/PneuNetBend-v0 --generations 4 --pop 8 --episodes 2 --eval-episodes 8
```

Measured on this laptop (2 workers, randomisation on, held-out seeds, mean return +- std over episodes):

| env | random actions | CEM linear policy | training |
|---|---|---|---|
| CableBeamReach (20 episodes) | -13.8 +- 23.2 | -0.87 +- 1.0 | 288 episodes, 70 s |
| ReducedGripperLift (20 episodes) | -2.4 +- 0.5, held 0 % | 44.0 +- 17.8, held 90 % | 384 episodes, 223 s |
| PneuNetBend (8 episodes) | -20.8 +- 15.1 | -3.0 +- 3.1 | 64 episodes, 197 s |

Random actions are a weak reference, and the beam's returns are small because the episode ends at the first step within
tolerance. The point is that the envs are learnable and what number to beat, not that a linear policy is good.

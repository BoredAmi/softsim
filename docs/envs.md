# Gymnasium environments (M12, CPU groundwork)

A reach task on any actuation scene as a [Gymnasium](https://gymnasium.farama.org) environment: the action is the
actuator values, the observation a list of scenario metrics, the reward a smooth distance to a target. Code:
`softsim/envs/`; tests: `tests/test_envs.py`. The extra is optional: `pip install "softsim[rl]"` (gymnasium >= 0.29).
Without it `import softsim.envs` works and `softsim.envs.make_env` raises an ImportError that says what to install.

```python
import gymnasium as gym
import softsim.envs                       # registers the ids below (also done by the `gymnasium.envs` entry point)

env = gym.make("softsim/CableBeamReach-v0", randomize=True)
obs, info = env.reset(seed=0)             # redraws the material and the target
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())   # tension in N
```

| id | config | actions | task |
|---|---|---|---|
| `softsim/CableBeamReach-v0` | `scenarios/envs/cable_beam_reach.yaml` | cable tension, 0-0.7 N | tip angle (`bend_angle`) to a target drawn from 0.02-0.07 rad, tolerance 0.004 rad |
| `softsim/CableArmReach-v0` | `scenarios/envs/cable_arm_reach.yaml` | cables A and B, 0-0.5 N each | tip (x, z) to (3.994 mm, 118.515 mm), tolerance 0.4 / 0.3 mm |

Your own: write a YAML next to a scenario (the two files above are the templates) and call
`softsim.envs.make_env("my_env.yaml", reward_fn=None)`. The config keys are in the docstring of
`softsim.envs.env.SoftsimEnv`.

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

That is about 0.4x real time for the beam and 0.16x for the arm: enough for a controller, tests and small
experiments, not for sample-hungry deep RL.

* CPU MuJoCo only. Its values are MuJoCo's, not physical, unless a transferred card is used: on the coarse 4 mm mesh
  linear tetrahedra lock and the beam reaches 0.089 rad at 0.7 N where Euler-Bernoulli gives 0.28 rad
  ([actuation](actuation.md)). A policy trained here learns this model, not the real beam.
* The environments add Rayleigh damping (equilibrium unchanged) so the beam settles in ~0.1 s and the arm in ~1 s; the
  arm's response lags the action by that much, so it needs a 2 s episode.
* Only the first soft body's material is redrawn (`randomize.body` picks another); zones keep their cards.
* The default arm target is one reachable point; randomised targets need a reachable set (see
  [inverse control](control.md)).
* A controller is a callable on the observation: a proportional-integral one that adds `5 N/rad x error` to the
  tension reaches the beam's 0.05 rad target in 2 steps (`tests/test_envs.py`).

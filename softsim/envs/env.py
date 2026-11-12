"""The generic Gymnasium environment: a soft robot scene, actuator values as actions, metrics as observations.

One environment step applies the action (actuator values in SI units: N, m or Pa), holds it for ``control_period``
(several physics steps) and reads the scenario metrics. Everything physical is the simulation's: the environment only
wires actions, observations, a reward and the reset/randomisation around :class:`softsim.runner.Simulation`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from softsim.envs import ENV_IDS, GRIPPER_ID, require_gymnasium

gym = require_gymnasium()
from gymnasium import spaces  # noqa: E402

from softsim.config.loader import get_dotted, load_yaml  # noqa: E402
from softsim.config.units import parse_quantity  # noqa: E402
from softsim.control.inverse import actuator_dimension, output_dimension  # noqa: E402
from softsim.errors import ConfigError  # noqa: E402
from softsim.materials.sampling import sample  # noqa: E402
from softsim.runner import Simulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCALES = {"length": 1e-3, "angle": 1e-2, "force": 1e-3, "pressure": 100.0}


def _quantity(value: Any, dimension: str) -> float:
    return float(parse_quantity(value, dimension))


class SoftsimEnv(gym.Env):
    """A reach task on a softsim scenario.

    Config keys (YAML mapping; ``scenario`` is a path relative to the config file or a scenario mapping):

    * ``scenario``, ``overrides`` (dotted ``key=value``), ``backend``: the simulation (coarse meshes keep a step fast).
    * ``actuators``: ``{name: [low, high]}`` in SI with units (``0 N``); the action space is this box, in this order.
    * ``control_period``: time each action is held, a multiple of the physics ``timestep`` (default one output interval).
    * ``observation``: scenario metric names (``tip_x``, ``bend_angle``, ``cable_length``, ...) in SI;
      ``include_actuators`` appends the held actuator values, ``include_target`` the target; ``observation_rate``
      (metric names) appends their rate of change over the last action, ``(new - old) / control_period``, a velocity.
    * ``task``: ``metrics`` (the controlled outputs), ``target`` (fixed) or ``target_range`` (``[[low, high], ...]``
      drawn at every reset), ``tolerance``, ``scale`` (default 1 mm / 0.01 rad per metric), ``success_bonus``.
      The reward is ``-|(metric - target) / scale|`` (Euclidean over the metrics), smooth away from the target,
      plus ``success_bonus`` on the step that reaches it; ``failure_penalty`` on a numerical failure.
    * ``max_steps``: truncation. ``randomize``: ``card_uncertainty`` (redraw the body's material from the card's
      ``uncertainty``) and ``ranges`` (``{"elastic.youngs_modulus": {"rel": 0.2}}``, ``{"abs": ...}`` or
      ``[low, high]``, drawn uniformly) applied on top, each reset; ``body`` names the soft body (default: the first). ``enabled: false`` keeps the rules
      but switches them off; the constructor argument ``randomize`` overrides it.

    The episode terminates when every controlled metric is within ``tolerance`` of its target (``info["success"]``)
    or when the backend fails (``info["failure"]`` holds the message); it is truncated after ``max_steps``.
    ``reward_fn(env, metrics, target) -> float`` replaces the built-in reward.

    ``reset(options={"target": [...]})`` fixes the target; ``options={"material": False}`` skips the redraw.
    Seeds: ``reset(seed=s)`` seeds the target and material draws (nothing else is random).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: str | Path | Mapping[str, Any], reward_fn: Callable | None = None,
                 render_mode: str | None = None, randomize: bool | None = None):
        super().__init__()
        if isinstance(config, (str, Path)):
            base = Path(config).resolve().parent
            cfg = dict(load_yaml(config))
        else:
            base, cfg = Path.cwd(), dict(config)
        scenario = cfg.get("scenario")
        if scenario is None:
            raise ConfigError("env config has no 'scenario'")
        self.sim = Simulation(scenario if isinstance(scenario, Mapping) else base / scenario,
                              overrides=cfg.get("overrides", ()), backend=cfg.get("backend"), base_dir=base)
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        names = [a.name for a in self.sim.scene.actuators]
        spec = cfg.get("actuators") or {}
        if not spec:
            raise ConfigError(f"env config needs 'actuators: {{name: [low, high]}}'; scenario actuators: {names}")
        self.actuators = list(spec)
        for n in self.actuators:
            if n not in names:
                raise ConfigError(f"no actuator {n!r}; scenario actuators: {names}")
        low, high = [], []
        for n in self.actuators:
            dim = actuator_dimension(self.sim, n)
            lo, hi = (_quantity(v, dim) for v in spec[n])
            if not lo < hi:
                raise ConfigError(f"actuator {n!r}: need low < high, got {spec[n]}")
            low.append(lo)
            high.append(hi)
        self._low, self._high = np.array(low), np.array(high)
        self.action_space = spaces.Box(self._low.astype(np.float32), self._high.astype(np.float32), dtype=np.float32)

        dt = self.sim.config.simulation.timestep
        period = cfg.get("control_period")
        self.substeps = (self.sim.config.simulation.output_every if period is None
                         else max(int(round(_quantity(period, "time") / dt)), 1))
        self.max_steps = int(cfg.get("max_steps", 50))

        known = self.sim.metrics()
        self.obs_metrics = list(cfg.get("observation", []))
        for m in self.obs_metrics:
            if m not in known:
                raise ConfigError(f"unknown metric {m!r}; this scenario reports {sorted(known)}")
        self.rate_metrics = list(cfg.get("observation_rate", []))
        for m in self.rate_metrics:
            if m not in known:
                raise ConfigError(f"unknown observation_rate metric {m!r}; this scenario reports {sorted(known)}")
        self.control_period = self.substeps * dt
        self.include_actuators = bool(cfg.get("include_actuators", False))
        self.include_target = bool(cfg.get("include_target", True))

        task = cfg.get("task") or {}
        self.task_metrics = list(task.get("metrics", []))
        for m in self.task_metrics:
            if m not in known:
                raise ConfigError(f"unknown task metric {m!r}; this scenario reports {sorted(known)}")
        k = len(self.task_metrics)
        self._range = None
        self._dims = [output_dimension(m) for m in self.task_metrics]
        if k:
            if ("target" in task) == ("target_range" in task):
                raise ConfigError("task needs exactly one of 'target' and 'target_range'")
            if "target" in task:
                self._target0 = np.array([_quantity(v, d) for v, d in zip(task["target"], self._dims)])
            else:
                self._range = np.array([[_quantity(v, d) for v in r] for r, d in zip(task["target_range"], self._dims)])
                self._target0 = self._range.mean(axis=1)
            tol = task.get("tolerance", [])
            self.tolerance = np.array([_quantity(v, d) for v, d in zip(tol, self._dims)]) if tol else np.zeros(k)
            sc = task.get("scale")
            self.scale = (np.array([_quantity(v, d) for v, d in zip(sc, self._dims)]) if sc
                          else np.array([_DEFAULT_SCALES[d] for d in self._dims]))
            if len(self._target0) != k or len(self.tolerance) != k or len(self.scale) != k:
                raise ConfigError("task target/tolerance/scale need one entry per metric")
        self.success_bonus = float(task.get("success_bonus", 1.0))
        self.failure_penalty = float(task.get("failure_penalty", -10.0))
        self.target = self._target0.copy() if k else np.zeros(0)

        n_obs = len(self.obs_metrics) + len(self.rate_metrics) + (len(self.actuators) if self.include_actuators else 0) \
            + (k if self.include_target else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)

        rnd = cfg.get("randomize") or {}
        has_rule = bool(rnd.get("card_uncertainty") or rnd.get("ranges"))
        self.randomize = has_rule and (rnd.get("enabled", True) if randomize is None else randomize)
        self._rnd = rnd
        bodies = [b.name for b in self.sim.scene.soft_bodies]
        self._body = rnd.get("body", bodies[0])
        if self._body not in bodies:
            raise ConfigError(f"no soft body {self._body!r}; soft bodies: {bodies}")
        self._nominal = next(b.material for b in self.sim.scene.soft_bodies if b.name == self._body)
        if rnd.get("card_uncertainty") and not self._nominal.uncertainty:
            raise ConfigError(f"randomize.card_uncertainty: material {self._nominal.name!r} has no uncertainty")
        self.material_draw: dict[str, float] = {}
        self._action = self._low.copy()
        self._steps = 0
        self._obs = np.zeros(n_obs, dtype=np.float32)
        self._metrics: dict[str, float] = known
        self._previous: dict[str, float] = known
        self.last_step_time = 0.0

    # -- helpers -------------------------------------------------------------

    def _draw_material(self) -> dict[str, float]:
        """Material variant: the card's uncertainty first, then the configured ranges. Returns the drawn values."""
        rng = self.np_random
        draws: dict[str, float] = {}
        if self._rnd.get("card_uncertainty"):
            seed = int(rng.integers(2**31 - 1))
            drawn = sample(self._nominal, 1, seed)[0].provenance["sample"]
            draws.update({k: v for k, v in drawn.items() if k not in ("index", "seed")})
        data = self._nominal.model_dump(mode="json", exclude_none=True)
        for key, rule in (self._rnd.get("ranges") or {}).items():
            if isinstance(rule, Mapping):
                nominal = draws.get(key, float(get_dotted(data, key)))
                half = rule["rel"] * abs(nominal) if "rel" in rule else float(rule["abs"])
                draws[key] = float(nominal + rng.uniform(-half, half))
            else:
                draws[key] = float(rng.uniform(float(rule[0]), float(rule[1])))
        return draws

    def _observe(self) -> np.ndarray:
        parts = [self._metrics[k] for k in self.obs_metrics]
        parts += [(self._metrics[k] - self._previous[k]) / self.control_period for k in self.rate_metrics]
        if self.include_actuators:
            parts += list(self._action)
        if self.include_target:
            parts += list(self.target)
        return np.asarray(parts, dtype=np.float32)

    def _error(self) -> np.ndarray:
        return np.array([self._metrics[m] for m in self.task_metrics]) - self.target

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"metrics": dict(self._metrics), "time": self.sim.time, "target": self.target.copy(),
                                "material": dict(self.material_draw)}
        if self.task_metrics:
            info["error"] = self._error()
        info.update(extra)
        return info

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self.randomize and options.get("material", True):
            self.material_draw = self._draw_material()
            self.sim.set_material(self._body, self._nominal.model_dump(mode="json"),
                                  [f"{k}={v!r}" for k, v in self.material_draw.items()])
        elif self.material_draw:  # a previous reset drew a variant: back to the card as given
            self.material_draw = {}
            self.sim.set_material(self._body, self._nominal.model_dump(mode="json"))
        self.sim.reset()
        if "target" in options:
            self.target = np.array([_quantity(v, d) for v, d in zip(options["target"], self._dims)])
        elif self._range is not None:
            self.target = self.np_random.uniform(self._range[:, 0], self._range[:, 1])
        else:
            self.target = self._target0.copy()
        self._action = self._low.copy()
        self.sim.apply_controls(dict(zip(self.actuators, self._action)))
        self._steps = 0
        self._metrics = self._previous = self.sim.metrics()
        self._obs = self._observe()
        return self._obs.copy(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), self._low, self._high)
        if action.shape != self._low.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be {len(self._low)} finite values for {self.actuators}, got {action}")
        self._action = action
        self._steps += 1
        t0 = time.perf_counter()
        try:
            self.sim.apply_controls(dict(zip(self.actuators, action)))
            self.sim.step(self.substeps)
            self._previous, self._metrics = self._metrics, self.sim.metrics()
        except Exception as exc:  # backend failure (divergence, solver error): end the episode, keep the last obs
            self.last_step_time = time.perf_counter() - t0
            return self._obs.copy(), self.failure_penalty, True, False, self._info(failure=f"{type(exc).__name__}: {exc}")
        self.last_step_time = time.perf_counter() - t0
        self._obs = self._observe()
        success, reward = False, 0.0
        if self.task_metrics:
            err = self._error()
            success = bool(np.all(np.abs(err) <= self.tolerance))
            reward = -float(np.linalg.norm(err / self.scale)) + (self.success_bonus if success else 0.0)
        if self.reward_fn is not None:
            reward = float(self.reward_fn(self, self._metrics, self.target))
        truncated = self._steps >= self.max_steps and not success
        return self._obs.copy(), reward, success, truncated, self._info(success=success)

    def close(self) -> None:
        self.sim.close()


def register_envs() -> None:
    """Register the example environments with Gymnasium (idempotent): see :data:`softsim.envs.ENV_IDS`."""
    for env_id, fname in ENV_IDS.items():
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point="softsim.envs.env:SoftsimEnv",
                         kwargs={"config": str(ROOT / "scenarios" / "envs" / fname)})
    if GRIPPER_ID not in gym.registry:
        gym.register(id=GRIPPER_ID, entry_point="softsim.envs.gripper:ReducedGripperLiftEnv")


def make_env(config: str | Path | Mapping[str, Any], **kwargs: Any) -> SoftsimEnv:
    """Environment from a YAML file or mapping; ``kwargs`` go to :class:`SoftsimEnv` (e.g. ``reward_fn``)."""
    return SoftsimEnv(config, **kwargs)


register_envs()

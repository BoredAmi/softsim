"""Gymnasium environment on the reduced gripper (:mod:`softsim.reduced.mjcf`): grasp an object and lift it.

Not a :class:`~softsim.envs.env.SoftsimEnv` (that wraps a :class:`~softsim.runner.Simulation` scene): the reduced
gripper is its own MuJoCo model of link chains, tens of times faster than real time, so it has its own small class
with the same conventions (SI actions and observations, ``reset(seed=)``, ``options``, ``info``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from softsim.envs import require_gymnasium

gym = require_gymnasium()
from gymnasium import spaces  # noqa: E402

from softsim.reduced import ReducedFinger, ReducedGripper  # noqa: E402
from softsim.reduced.mjcf import OBJECTS  # noqa: E402
from softsim.scenarios.grasp import is_held  # noqa: E402

FINGER_JSON = Path(__file__).resolve().parents[1] / "reduced" / "wang_finger.json"


class ReducedGripperLiftEnv(gym.Env):
    """Grasp and lift an object with the reduced two-finger gripper (``softsim/ReducedGripperLift-v0``).

    * **Action** (2, float32): finger pressure [Pa] in ``[0, max_pressure]`` and the palm height command [m] in
      ``[0, max_lift]``. Both are ramped linearly from their previous value to the new one over the control period.
    * **Observation** (7, float32): pressure and lift command, the palm's actual rise [m], the object's rise above the
      table [m] and its vertical velocity [m/s], and its horizontal offset (x, y) from the palm axis [m].
    * **Reward** per step: ``min(object_rise, lift_goal) / lift_goal - pressure_cost * pressure / max_pressure``: the
      fraction of the lift goal the object has reached, minus a small cost for squeezing. ``info["held"]`` is
      :func:`softsim.scenarios.grasp.is_held` on the rise history, ``info["success"]`` that at the last step.
    * **Episode**: ``max_steps`` actions of ``control_period`` (default 80 x 50 ms = 4 s, the length of the scripted
      grasp: ramp, settle, lift, hold), then truncation. It never terminates early.
    * **Randomisation** (on by default; ``randomize=False`` fixes mass and friction at the means): object ``mass``
      [kg] and ``friction`` uniform over the ranges, drawn from ``reset(seed=)``'s generator, the shape is fixed.
      With the default fitted finger a 50 g box is never held; the default range (8-20 g, mu 0.7-1.2) holds at
      moderate pressure and slips at low pressure, so the policy has something to find.
    * ``reset(options={"mass": ..., "friction": ...})`` fixes the draw.
    """

    metadata = {"render_modes": []}

    def __init__(self, finger: str | Path | ReducedFinger = FINGER_JSON, shape: str = "box", fingers: int = 2,
                 max_pressure: float = 40e3, max_lift: float = 0.06, lift_goal: float = 0.04,
                 control_period: float = 0.05, max_steps: int = 80, pressure_cost: float = 0.05,
                 mass_range: tuple[float, float] = (0.008, 0.020), friction_range: tuple[float, float] = (0.7, 1.2),
                 randomize: bool = True, render_mode: str | None = None):
        super().__init__()
        if shape not in OBJECTS:
            raise ValueError(f"unknown object shape {shape!r}; choose from {sorted(OBJECTS)}")
        self.finger = finger if isinstance(finger, ReducedFinger) else ReducedFinger.from_json(finger)
        self.shape, self.fingers = shape, fingers
        self.max_pressure, self.max_lift, self.lift_goal = float(max_pressure), float(max_lift), float(lift_goal)
        self.max_steps, self.pressure_cost = int(max_steps), float(pressure_cost)
        self.mass_range, self.friction_range, self.randomize = tuple(mass_range), tuple(friction_range), randomize
        self.render_mode = render_mode
        self.action_space = spaces.Box(np.zeros(2, np.float32), np.array([self.max_pressure, self.max_lift], np.float32),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        self.gripper: ReducedGripper | None = None
        self.object_params: dict[str, float] = {}
        self._period = float(control_period)
        self._steps = 0
        self._cmd = np.zeros(2)
        self._rise = []
        self._z0 = 0.0
        self._obs = np.zeros(7, np.float32)

    def _build(self, mass: float, friction: float) -> None:
        self.gripper = ReducedGripper(self.finger, fingers=self.fingers, shape=self.shape, size=OBJECTS[self.shape],
                                      mass=mass, friction=friction)
        mujoco.mj_forward(self.gripper.model, self.gripper.data)  # body positions, so the starting height is read
        self.substeps = max(int(round(self._period / self.gripper.model.opt.timestep)), 1)
        self.dt = self.gripper.model.opt.timestep

    def _observe(self) -> np.ndarray:
        g = self.gripper
        vz = float(g.data.qvel[g.model.jnt_dofadr[g.model.body_jntadr[g._object]] + 2])
        pos = g.data.xpos[g._object]
        return np.array([*self._cmd, g.palm_rise, g.object_height - self._z0, vz, pos[0], pos[1]], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        mass = float(np.mean(self.mass_range))
        mu = float(np.mean(self.friction_range))
        if self.randomize:
            mass = float(self.np_random.uniform(*self.mass_range))
            mu = float(self.np_random.uniform(*self.friction_range))
        mass, mu = float(options.get("mass", mass)), float(options.get("friction", mu))
        self.object_params = {"mass": mass, "friction": mu}
        self._build(mass, mu)
        self._steps, self._cmd, self._rise = 0, np.zeros(2), []
        self._z0 = self.gripper.object_height
        self._obs = self._observe()
        return self._obs.copy(), self._info()

    def _info(self, **extra: Any) -> dict[str, Any]:
        return {"object": dict(self.object_params), "time": self._steps * self._period,
                "rise": float(self._rise[-1]) if self._rise else 0.0, **extra}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), self.action_space.low, self.action_space.high)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f"action must be 2 finite values (pressure [Pa], lift [m]), got {action}")
        g, old = self.gripper, self._cmd
        for i in range(1, self.substeps + 1):
            p, lift = old + (a - old) * i / self.substeps
            g.step(float(p), float(lift))
        self._cmd = a
        self._steps += 1
        self._obs = self._observe()
        rise = g.object_height - self._z0
        self._rise.append(rise)
        reward = min(rise, self.lift_goal) / self.lift_goal - self.pressure_cost * a[0] / self.max_pressure
        truncated = self._steps >= self.max_steps
        info = self._info()
        if truncated or self._steps % 4 == 0:
            t = np.arange(1, len(self._rise) + 1) * self._period
            info["held"] = info["success"] = bool(is_held(t, np.array(self._rise)))
        return self._obs.copy(), float(reward), False, truncated, info

    def close(self) -> None:
        self.gripper = None

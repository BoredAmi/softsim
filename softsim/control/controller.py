"""Closed-loop control on a surrogate, in plain Python (no ROS, no simulator), for a fixed control rate.

``Controller.from_json("surrogate.json", ki=4.0)`` loads a surrogate written by ``softsim surrogate``
(``docs/surrogate.md``) and turns it into a controller; ``step(measured, target)`` is called once per control period
``dt`` and returns the actuation (SI units, in the order of ``Controller.inputs``).

Two layers:

* **Feedforward.** The actuation that puts the *target* at equilibrium on the surrogate: bounded least squares on the
  fitted map, warm-started from the last solution (milliseconds; a cold multi-start solve only when the warm one is
  poor). It is exact if the surrogate is the plant, and leaves a steady-state error when it is not.
* **Feedback.** An integral correction ``v`` in actuator space, added to the feedforward. Each period
  ``v += ki dt (J^T J + lambda^2 I)^-1 J^T e`` (damped least squares in scaled coordinates, ``J`` the surrogate's
  Jacobian at the applied actuation, ``e = target - measured``), so every output error is pushed down along the
  directions the actuators can move it, and singular directions (two cables that move the tip the same way) do not
  blow up. Anti-windup by clamping: ``feedforward + v`` is kept inside the actuator bounds, and ``v`` itself inside
  ``max_correction`` of the range, so a target that cannot be reached does not wind the integrator up.

Optional (``mode="mpc"``, needs the ``dynamics`` model in the JSON): an offset-free model-predictive step. The state
of the DMD model ``s_{k+1} = A s_k + B (u_k - u0)`` is built from the last measured outputs (one earlier sample is
stacked when the model has a delay), the one-step prediction error is held constant over the horizon as an output
disturbance, and the actuation sequence that minimises the tracking error plus ``move_weight`` times the move size
over ``horizon`` periods, within the bounds, is found by bounded linear least squares; its first element is applied.
``dt`` must equal the model's sample period.

Limits: the controller is only as good as the Jacobian of the surrogate (the sign and size of the gain; a Jacobian off
by a factor 2 changes the speed, not the final value); the integral action is slower than the plant's slowest mode by
design (``ki dt`` per period, keep it well below 1); the surrogate's box is a hard limit; nothing here estimates a
measurement bias or noise (filter ``measured`` before the call).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from softsim.control.inverse import _SCALES, output_dimension
from softsim.control.surrogate import Surrogate, solve_on_surrogate
from softsim.errors import ConfigError


class Controller:
    """Feedforward from a surrogate's inverse plus integral (or MPC-lite) feedback; see the module docstring.

    Args:
        surrogate: A :class:`~softsim.control.surrogate.Surrogate`.
        outputs: The outputs to track (default: all of the surrogate's). At least as many as the controlled inputs.
        controlled: Names of the inputs the controller moves (default: all); the others stay at ``fixed``.
        fixed: Values of the uncontrolled inputs (default: the lower bound).
        dt: Control period [s].
        ki: Integral gain [1/s]: the fraction of the remaining (model-predicted) correction applied per second.
            ``ki * dt`` is the fraction per period; keep it below ~0.5 and below the plant's bandwidth.
        damping: Damped-least-squares ``lambda`` in scaled coordinates (0.05: singular values below it are ignored).
        max_correction: Largest integral correction, as a fraction of each input's range.
        scales: Output scale for the error weighting (default: softsim's by dimension, 1 mm and 0.01 rad).
        bounds: Tighter actuator bounds ``{name: (lower, upper)}`` than the surrogate's box.
        mode: ``"integral"`` (default), ``"ff"`` (feedforward only) or ``"mpc"``.
        horizon, move_weight: MPC horizon [periods] and weight on the actuation move (scaled units).
    """

    def __init__(self, surrogate: Surrogate, outputs: Sequence[str] | None = None, controlled: Sequence[str] | None = None,
                 fixed: Mapping[str, float] | None = None, dt: float = 0.02, ki: float = 4.0, damping: float = 0.05,
                 max_correction: float = 0.5, scales: Mapping[str, float] | None = None,
                 bounds: Mapping[str, tuple[float, float]] | None = None, mode: str = "integral", horizon: int = 10,
                 move_weight: float = 0.01):
        self.sur = surrogate
        self.inputs = list(surrogate.inputs)
        self.outputs = list(outputs) if outputs is not None else list(surrogate.outputs)
        for o in self.outputs:
            if o not in surrogate.outputs:
                raise ConfigError(f"surrogate has no output {o!r}; it has {surrogate.outputs}")
        self.controlled = list(controlled) if controlled is not None else list(self.inputs)
        for n in self.controlled:
            if n not in self.inputs:
                raise ConfigError(f"surrogate has no input {n!r}; it has {self.inputs}")
        if len(self.outputs) < len(self.controlled):
            raise ConfigError(f"{len(self.controlled)} controlled inputs but {len(self.outputs)} outputs: "
                              "the actuation would not be unique")
        if dt <= 0 or ki < 0 or damping < 0 or not 0 < max_correction <= 1:
            raise ConfigError("need dt > 0, ki >= 0, damping >= 0 and 0 < max_correction <= 1")
        if mode not in ("integral", "ff", "mpc"):
            raise ConfigError(f"mode must be integral, ff or mpc, not {mode!r}")
        self.dt, self.ki, self.damping, self.mode = float(dt), float(ki), float(damping), mode
        self.max_correction, self.horizon, self.move_weight = float(max_correction), int(horizon), float(move_weight)
        self._ci = [self.inputs.index(n) for n in self.controlled]
        self._oi = [surrogate.outputs.index(o) for o in self.outputs]
        self.lower, self.upper = surrogate.lower.copy(), surrogate.upper.copy()
        for n, (a, b) in (bounds or {}).items():
            if n not in self.inputs:
                raise ConfigError(f"bounds: surrogate has no input {n!r}; it has {self.inputs}")
            k = self.inputs.index(n)
            self.lower[k], self.upper[k] = max(self.lower[k], a), min(self.upper[k], b)
            if self.upper[k] <= self.lower[k]:
                raise ConfigError(f"bounds of {n!r} do not overlap the surrogate's box")
        self._base = self.lower.copy()
        for n, v in (fixed or {}).items():
            if n not in self.inputs:
                raise ConfigError(f"fixed: surrogate has no input {n!r}; it has {self.inputs}")
            self._base[self.inputs.index(n)] = v
        self.scale = np.array([(scales or {}).get(o, _SCALES[output_dimension(o)]) for o in self.outputs])
        self._range = (self.upper - self.lower)[self._ci]
        self.dyn = surrogate.data.get("dynamics")
        if mode == "mpc":
            self._init_mpc()
        self.reset()

    # ------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "Controller":
        """Load ``surrogate.json`` (a file written by ``softsim surrogate``); ``kwargs`` are the constructor's."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"surrogate file not found: {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigError(f"{path} is not a surrogate file")
        try:
            return cls(Surrogate(data), **kwargs)
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{path} is not a valid surrogate file (missing or malformed {exc})") from None

    # ----------------------------------------------------------------------------------------------- state
    def reset(self) -> None:
        """Forget the integral state and the history (call at the start of every run)."""
        self.v = np.zeros(len(self._ci))                  # integral correction, actuator units
        self._ff = None
        self._ff_target = None
        self.u = self._base.copy()
        self.u[self._ci] = np.clip(self.u[self._ci], self.lower[self._ci], self.upper[self._ci])
        self._prev_ym = None
        self._s_pred = None
        self._u_prev = None
        self.info: dict[str, Any] = {}

    # -------------------------------------------------------------------------------------------- helpers
    def _vec(self, x: Mapping[str, float] | Sequence[float] | np.ndarray, what: str) -> np.ndarray:
        if isinstance(x, Mapping):
            missing = [o for o in self.outputs if o not in x]
            if missing:
                raise ConfigError(f"{what} lacks outputs {missing}; the controller tracks {self.outputs}")
            x = [x[o] for o in self.outputs]
        a = np.asarray(x, dtype=float).reshape(-1)
        if a.shape != (len(self.outputs),):
            raise ConfigError(f"{what} needs {len(self.outputs)} values for {self.outputs}, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ConfigError(f"{what} is not finite: {a}")
        return a

    def _full(self, uc: np.ndarray) -> np.ndarray:
        u = self._base.copy()
        u[self._ci] = uc
        return u

    def _predict(self, uc: np.ndarray) -> np.ndarray:
        return self.sur.predict(self._full(uc), extrapolate=True)[0][self._oi]

    def jacobian(self, uc: np.ndarray) -> np.ndarray:
        """d outputs / d controlled inputs of the surrogate at ``uc`` (central differences, shape (outputs, inputs))."""
        h = 1e-4 * self._range
        lo, hi = self.lower[self._ci], self.upper[self._ci]
        cols = []
        for k in range(len(uc)):
            a, b = uc.copy(), uc.copy()
            a[k], b[k] = min(uc[k] + h[k], hi[k] + h[k]), max(uc[k] - h[k], lo[k] - h[k])
            cols.append((self._predict(a) - self._predict(b)) / (a[k] - b[k]))
        return np.array(cols).T

    def feedforward(self, target: np.ndarray) -> np.ndarray:
        """Controlled-input values that put ``target`` at equilibrium on the surrogate (bounded least squares)."""
        lo, hi = self.lower[self._ci], self.upper[self._ci]
        sc = self.scale
        x0 = self._ff if self._ff is not None else 0.5 * (lo + hi)
        r = least_squares(lambda v: (self._predict(v) - target) / sc, np.clip(x0, lo, hi), bounds=(lo, hi),
                          x_scale=hi - lo, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        worst = np.abs(r.fun).max()
        if (self._ff is None and worst > 0.05) or worst > 0.5:
            # a poor local solution: use the multi-start solver (cold start, or a target far from the last one)
            fixed = {n: float(self._base[k]) for k, n in enumerate(self.inputs) if n not in self.controlled}
            s = solve_on_surrogate(self.sur, dict(zip(self.outputs, target)), dict(zip(self.outputs, sc)),
                                   {n: (self.lower[k], self.upper[k]) for k, n in enumerate(self.inputs)},
                                   fixed, starts=8)
            cand = np.array([s["values"][n] for n in self.controlled])
            if s["residual"] < np.abs(r.fun).max():
                return np.clip(cand, lo, hi)
        return r.x

    # ---------------------------------------------------------------------------------------------- the step
    def step(self, measured: Mapping[str, float] | Sequence[float] | np.ndarray,
             target: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        """One control period: measured and target outputs (SI) -> actuation (SI, in the order of ``inputs``).

        ``self.info`` holds the last step's ``error`` (target - measured), feedforward, correction and saturation.
        """
        y, r = self._vec(measured, "measured"), self._vec(target, "target")
        lo, hi = self.lower[self._ci], self.upper[self._ci]
        if self._ff is None or self._ff_target is None or not np.array_equal(r, self._ff_target):
            self._ff, self._ff_target = self.feedforward(r), r.copy()
        e = r - y
        if self.mode == "mpc":
            uc = self._mpc(y, r)
            self.v = uc - self._ff
        else:
            if self.mode == "integral" and self.ki > 0:
                uc = np.clip(self._ff + self.v, lo, hi)
                jt = self.jacobian(uc) / self.scale[:, None] * self._range[None, :]       # scaled coordinates
                du = np.linalg.solve(jt.T @ jt + self.damping ** 2 * np.eye(jt.shape[1]), jt.T @ (e / self.scale))
                self.v = self.v + self.ki * self.dt * du * self._range
                # anti-windup: the correction cannot take the actuation outside the bounds or past max_correction
                lim = self.max_correction * self._range
                self.v = np.clip(self.v, np.maximum(lo - self._ff, -lim), np.minimum(hi - self._ff, lim))
            uc = np.clip(self._ff + self.v, lo, hi)
        self.u = self._full(uc)
        self.info = {"error": e, "feedforward": self._ff.copy(), "correction": self.v.copy(),
                     "saturated": [n for k, n in enumerate(self.controlled)
                                   if uc[k] <= lo[k] + 1e-9 or uc[k] >= hi[k] - 1e-9]}
        return self.u.copy()

    # -------------------------------------------------------------------------------------------------- MPC
    def _init_mpc(self) -> None:
        d = self.dyn
        if not d or d.get("type") != "dmd_control":
            raise ConfigError("mode='mpc' needs the 'dynamics' model (DMD) in the surrogate file: add `dynamics:` "
                              "to the surrogate job")
        try:
            self._A, self._B = np.array(d["A"], dtype=float), np.array(d["B"], dtype=float)
            self._x0, self._u0 = np.array(d["x0"], dtype=float), np.array(d["u0"], dtype=float)
            self._delay, self._mdt = int(d["delay"]), float(d["dt"])
            names_in, names_out = list(d["input_names"]), list(d["output_names"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"malformed dynamics model (missing or invalid {exc})") from None
        if names_in != self.inputs:
            raise ConfigError(f"dynamics inputs {names_in} differ from the surrogate's {self.inputs}")
        if sorted(names_out) != sorted(self.outputs):
            raise ConfigError(f"mode='mpc' tracks the model's outputs {names_out}, not {self.outputs}")
        if abs(self._mdt - self.dt) > 1e-6 * max(self._mdt, 1.0):
            raise ConfigError(f"MPC needs dt = the model's sample period {self._mdt:g} s, got {self.dt:g} s")
        if self.horizon < 1:
            raise ConfigError("horizon must be >= 1")
        p, n, H, m = len(names_out), self._A.shape[0], self.horizon, len(self._ci)
        if self._B.shape != (n, len(self.inputs)) or n != p * (self._delay + 1):
            raise ConfigError("malformed dynamics model: A and B do not match the inputs and outputs")
        self._perm = [self.outputs.index(o) for o in names_out]          # model output j = tracked output perm[j]
        self._p = p
        Bc = self._B[:, self._ci]
        unc = [k for k in range(len(self.inputs)) if k not in self._ci]
        # constant drift: the uncontrolled inputs held at `fixed`, and the model's offset u0 of the controlled ones
        drift = (self._B[:, unc] @ (self._base[unc] - self._u0[unc]) if unc else 0.0) - Bc @ self._u0[self._ci]
        powers = [np.eye(n)]
        for _ in range(H):
            powers.append(powers[-1] @ self._A)
        F = np.zeros((H * p, n))
        G = np.zeros((H * p, H * m))
        fd = np.zeros(H * p)
        acc = np.zeros(n)
        for k in range(H):
            F[k * p:(k + 1) * p] = powers[k + 1][:p]
            acc = self._A @ acc + drift
            fd[k * p:(k + 1) * p] = acc[:p]
            for j in range(k + 1):
                G[k * p:(k + 1) * p, j * m:(j + 1) * m] = (powers[k - j] @ Bc)[:p]
        self._F, self._G, self._fd = F, G, fd

    def _mpc(self, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        p, H, m = self._p, self.horizon, len(self._ci)
        ym, rm = y[self._perm], r[self._perm]                    # in the model's output order
        sc = self.scale[self._perm]
        prev = ym if self._prev_ym is None else self._prev_ym
        s = np.concatenate([ym - self._x0] + [prev - self._x0] * self._delay)
        d = (ym - self._x0 - self._s_pred[:p]) if self._s_pred is not None else np.zeros(p)   # output disturbance
        lo = self.lower[self._ci]
        # predicted outputs = x0 + d + F s + fd + G (lo + range z); minimise the scaled distance to the target
        w = np.tile(1.0 / sc, H)
        const = np.tile(self._x0 + d, H) + self._F @ s + self._fd + self._G @ np.tile(lo, H)
        a_track = self._G * np.tile(self._range, H)[None, :] * w[:, None]
        b_track = (np.tile(rm, H) - const) * w
        z_prev = (self.u[self._ci] - lo) / self._range if self._u_prev is None else (self._u_prev - lo) / self._range
        D = np.eye(H * m) - np.eye(H * m, k=-m)                  # moves dz_k = z_k - z_{k-1}
        rhs = np.zeros(H * m)
        rhs[:m] = z_prev
        sw = np.sqrt(self.move_weight)
        sol = lsq_linear(np.vstack([a_track, sw * D]), np.concatenate([b_track, sw * rhs]),
                         bounds=(np.zeros(H * m), np.ones(H * m)))
        uc = lo + sol.x[:m] * self._range
        self._prev_ym, self._u_prev = ym, uc.copy()
        self._s_pred = self._A @ s + self._B @ (self._full(uc) - self._u0)
        return uc

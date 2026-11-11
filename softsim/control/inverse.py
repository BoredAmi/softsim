"""Quasi-static inverse control: the actuator values that put chosen outputs at target values at equilibrium.

    values = solve_inverse("scenarios/actuation/cable_cantilever.yaml", {"bend_angle": 0.03},
                           ["cable"], {"cable": (0.0, "0.7 N")})

The forward map is the *equilibrium map* ``u -> outputs``: the actuators are held at constant values ``u``, the scene
runs until every target output has stopped changing (:class:`Equilibrium`), and the outputs are read. The inverse is
a bounded nonlinear least-squares problem on that map, solved with ``scipy.optimize.least_squares`` (trust region,
bounds) and a forward-difference Jacobian we supply ourselves (``n`` extra equilibria per iteration, ``n`` = number of
actuators).

How a state is brought to rest: the elastic energy is conservative, so an undamped body rings forever. The solve
therefore adds stiffness-proportional damping (``settle_damping``, Rayleigh beta; default 10 ms) to the body and its
zones that have none of their own. A damping force vanishes at rest, so it changes how fast the equilibrium is
reached and not where it is. Settling stops when the range of every target output over the last ``settle_window``
seconds (default 0.1 s, longer than the relaxation time of what is being settled) is below ``settle_tol`` of its scale (or after ``max_settle`` seconds, which is flagged). By default
one :class:`~softsim.runner.Simulation` is built and reused: every evaluation starts from the previous equilibrium
(``warm_start``), which for an elastic body gives the same answer as starting from rest, only sooner.

Residuals are scaled: ``(output - target) / scale`` with the scale of the output's kind (1 mm for lengths, 0.01 rad
for angles, 1 mN for forces, 100 Pa for pressures; override with ``scales``). ``tolerance`` bounds the largest scaled
residual. Checked against: the forward map itself (round trip), and the Euler-Bernoulli linearity of the cable
cantilever (``tests/test_inverse.py``).

Limits: quasi-static (the path to the state is not planned), one equilibrium per value (a snapping or buckling
structure may have several: the solution found depends on the start), no contact history, and the values are those of
the backend's own physics (MuJoCo's flex locks, see ``docs/actuation.md``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import least_squares

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.runner import Simulation

DEFAULT_DAMPING = {"model": "rayleigh", "beta": 0.01}   # stiffness-proportional [s]


def output_dimension(name: str) -> str:
    """Physical dimension of a scenario output column, from its name (for units and default scales)."""
    if "angle" in name:
        return "angle"
    if name.endswith(("_force", "_tension")):
        return "force"
    if name.endswith("_pressure"):
        return "pressure"
    return "length"


_SCALES = {"length": 1e-3, "angle": 1e-2, "force": 1e-3, "pressure": 100.0}


def actuator_dimension(sim: Simulation, name: str) -> str:
    """Dimension of an actuator's value: tension/force, a stroke (length) or a pressure."""
    for a in sim.scene.actuators:
        if a.name == name:
            if hasattr(a, "region") and not hasattr(a, "direction") and not hasattr(a, "via"):
                return "pressure"
            return "length" if getattr(a, "stiffness", None) is not None else "force"
    raise ConfigError(f"no actuator {name!r}; actuators: {[a.name for a in sim.scene.actuators]}")


class _Done(Exception):
    """Raised inside the residual function to end the search (converged or out of evaluations)."""


class Equilibrium:
    """The equilibrium map of a simulation: ``values -> outputs`` with constant actuators.

    Args:
        sim: The simulation to reuse.
        outputs: Output columns to read (keys of ``sim.metrics()``).
        scales: Scale of each output, used for the settling criterion.
        settle_tol: Settled when every output's range over the window is below ``settle_tol * scale``.
        settle_window: Length of the window [s]. It must be longer than the body's slowest relaxation time: a
            window shorter than that sees a body that has only just started to move as settled.
        max_settle: Longest settling run [s]; hitting it is counted in ``unsettled``.
        warm_start: Start each evaluation from the previous state instead of resetting.
    """

    def __init__(self, sim: Simulation, outputs: list[str], scales: Mapping[str, float], settle_tol: float = 1e-3,
                 settle_window: float = 0.1, max_settle: float = 1.0, warm_start: bool = True,
                 fixed: Mapping[str, float] | None = None):
        self.sim, self.outputs = sim, outputs
        self.scales = np.array([scales[o] for o in outputs])
        interval = sim.config.simulation.output_every * sim.config.simulation.timestep
        self.settle_tol, self.window, self.warm_start = settle_tol, max(int(round(settle_window / interval)), 2), warm_start
        self.max_samples = max(int(round(max_settle / interval)), self.window + 1)
        # actuators we do not solve for are held at their value at t = 0, not left to follow their schedule
        self.fixed = {a.name: a.value(0.0) for a in sim.scene.actuators}
        self.fixed.update(fixed or {})
        self.evaluations = 0
        self.steps = 0
        self.unsettled = 0
        self.sim.reset()

    def __call__(self, values: Mapping[str, float]) -> np.ndarray:
        sim = self.sim
        if not self.warm_start:
            sim.reset()
        sim.apply_controls({**self.fixed, **values})
        history: list[np.ndarray] = []
        for i in range(self.max_samples):
            sim.step()
            self.steps += 1
            m = sim.metrics()
            history.append(np.array([m[o] for o in self.outputs]))
            if len(history) >= self.window:
                recent = np.array(history[-self.window:])
                if np.all(np.ptp(recent, axis=0) <= self.settle_tol * self.scales):
                    break
        else:
            self.unsettled += 1
        self.evaluations += 1
        return history[-1]


@dataclass
class InverseResult:
    """Outcome of :func:`solve_inverse`.

    Attributes:
        values: Solved actuator values (SI: N, m or Pa).
        outputs: The outputs reached at ``values``.
        targets: The targets.
        errors: ``outputs - targets`` per output.
        residual: Largest scaled residual (1 = one scale unit, see module docstring).
        status: ``converged`` (residual within tolerance), ``bounded_best`` (some actuator sits at a bound and the
            target is not reached: probably unreachable), ``max_evaluations`` or ``stalled``.
        at_bounds: Actuators at a bound in the solution.
        simulations: Equilibria computed (each a settling run).
        steps: Output intervals simulated in all.
        unsettled: Equilibria that hit ``max_settle`` without settling.
        wall_time: Seconds.
        history: Every equilibrium computed, in order: ``{"values": ..., "outputs": ...}``.
    """

    values: dict[str, float]
    outputs: dict[str, float]
    targets: dict[str, float]
    errors: dict[str, float]
    residual: float
    status: str
    at_bounds: list[str]
    simulations: int
    steps: int
    unsettled: int
    wall_time: float
    history: list[dict[str, Any]] = field(default_factory=list)
    converged: bool = field(init=False)

    def __post_init__(self) -> None:
        self.converged = self.status == "converged"

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("status", "converged", "values", "outputs", "targets", "errors", "residual",
                                              "at_bounds", "simulations", "steps", "unsettled", "wall_time")}


def _with_damping(raw: dict[str, Any], damping: Mapping[str, Any]) -> dict[str, Any]:
    """Give the body and every zone that has no damping of its own ``damping`` (equilibrium is unchanged)."""
    raw = apply_overrides(raw, [])
    body = raw.get("body")
    if not isinstance(body, dict):
        return raw
    holders = [body] + [z for z in body.get("zones", []) if isinstance(z, dict)]
    for h in holders:
        mat = h.get("material")
        if mat is None:
            continue
        if isinstance(mat, str):
            mat = {"preset": mat}
        if mat.get("damping") is None:
            mat = {**mat, "damping": dict(damping)}
        h["material"] = mat
    return raw


def solve_inverse(config: str | Path | Mapping[str, Any] | Simulation, targets: Mapping[str, Any],
                  actuators: Sequence[str], bounds: Mapping[str, tuple[Any, Any]], backend: str = "mujoco",
                  overrides: list[str] = (), initial: Mapping[str, Any] | None = None, tolerance: float = 0.05,
                  max_evaluations: int = 40, scales: Mapping[str, float] | None = None,
                  settle_damping: Mapping[str, Any] | None = DEFAULT_DAMPING, settle_tol: float = 1e-3,
                  settle_window: float | str = 0.1, max_settle: float | str = 1.0, warm_start: bool = True,
                  jacobian_step: float = 0.02) -> InverseResult:
    """Find the constant actuator values that bring ``targets`` to equilibrium.

    Args:
        config: Scenario YAML path, mapping or a built :class:`Simulation` (then reused as is, and not closed;
            ``overrides``, ``backend`` and ``settle_damping`` do not apply).
        targets: ``{output column: value}``, values in SI or ``"<n> <unit>"`` strings (``"20 mm"``, ``"10 deg"``).
        actuators: Names of the actuators to solve for; others are held at their value at t = 0.
        bounds: ``{actuator name: (lower, upper)}``, finite, in SI or with units (a tension: ``(0, "1 N")``).
        backend: Backend name (the equilibrium map is MuJoCo's by default: cheap, but see its locking).
        overrides: Dotted ``key=value`` config overrides.
        initial: Start values (default: the middle of the bounds).
        tolerance: Stop when the largest scaled residual is below this (scale: 1 mm, 0.01 rad, 1 mN, 100 Pa).
        max_evaluations: Cap on equilibria (each is a settling run).
        scales: Per-output scale overrides (SI).
        settle_damping: Damping card added where there is none (``None``: none; the body must then settle by itself).
        settle_tol, settle_window, max_settle, warm_start: See :class:`Equilibrium`.
        jacobian_step: Forward-difference step as a fraction of each actuator's range.

    Returns:
        :class:`InverseResult`. An unreachable target is not an error: the best bounded point is returned with
        ``status == "bounded_best"``.
    """
    t0 = time.perf_counter()
    max_settle, settle_window = parse_quantity(max_settle, "time"), parse_quantity(settle_window, "time")
    own = not isinstance(config, Simulation)
    if own:
        if isinstance(config, (str, Path)):
            base, raw = Path(config).resolve().parent, load_yaml(config)
        else:
            base, raw = Path.cwd(), dict(config)
        raw = apply_overrides(raw, list(overrides))
        if settle_damping:
            raw = _with_damping(raw, settle_damping)
        sim = Simulation(raw, backend=backend, base_dir=base)
    else:
        sim = config
    try:
        return _solve(sim, targets, actuators, bounds, initial, tolerance, max_evaluations, scales, settle_tol, settle_window,
                      max_settle, warm_start, jacobian_step, t0)
    finally:
        if own:
            sim.close()


def _solve(sim, targets, actuators, bounds, initial, tolerance, max_evaluations, scales, settle_tol, settle_window, max_settle,
           warm_start, jacobian_step, t0) -> InverseResult:
    names = list(actuators)
    missing = [n for n in names if n not in bounds]
    if missing:
        raise ConfigError(f"no bounds for actuators {missing}")
    outs = list(targets)
    if not names or not outs:
        raise ConfigError("need at least one actuator and one target")
    if len(outs) < len(names):
        raise ConfigError(f"{len(names)} actuators but only {len(outs)} targets: the solution would not be unique")
    probe = sim.metrics()
    for o in outs:
        if o not in probe:
            raise ConfigError(f"unknown output {o!r}; available: {sorted(probe)}")
    target = np.array([parse_quantity(targets[o], output_dimension(o)) for o in outs])
    lo, hi = np.empty(len(names)), np.empty(len(names))
    for i, n in enumerate(names):
        dim = actuator_dimension(sim, n)
        lo[i], hi[i] = (parse_quantity(b, dim) for b in bounds[n])
        if not (np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i]):
            raise ConfigError(f"actuator {n!r}: bounds must be finite with upper > lower, got {bounds[n]!r}")
    sc = {o: parse_quantity((scales or {}).get(o, _SCALES[output_dimension(o)]), output_dimension(o)) for o in outs}
    scale = np.array([sc[o] for o in outs])
    start = np.array([parse_quantity((initial or {}).get(n, 0.5 * (lo[i] + hi[i])), actuator_dimension(sim, n))
                      for i, n in enumerate(names)])
    eq = Equilibrium(sim, outs, sc, settle_tol, settle_window, max_settle, warm_start)
    span = hi - lo

    def values(u: np.ndarray) -> dict[str, float]:
        return {n: float(lo[i] + span[i] * u[i]) for i, n in enumerate(names)}

    best: dict[str, Any] = {"res": np.inf}
    history: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}

    def residual(u: np.ndarray) -> np.ndarray:
        if "u" in cache and np.array_equal(cache["u"], u):
            return cache["r"]
        out = eq(values(u))
        history.append({"values": values(u), "outputs": dict(zip(outs, map(float, out)))})
        r = (out - target) / scale
        cache.update(u=u.copy(), r=r, out=out)
        worst = float(np.abs(r).max())
        if worst < best["res"]:
            best.update(res=worst, u=u.copy(), out=out)
        if worst <= tolerance or eq.evaluations >= max_evaluations:
            raise _Done
        return r

    def jac(u: np.ndarray) -> np.ndarray:
        r0 = residual(u)
        cols = []
        for i in range(len(u)):
            h = jacobian_step if u[i] + jacobian_step <= 1.0 else -jacobian_step
            v = u.copy()
            v[i] += h
            cols.append((residual(v) - r0) / h)
        cache.pop("u", None)  # the last evaluation was a perturbed point
        return np.stack(cols, axis=1)

    u0 = np.clip((start - lo) / span, 0.0, 1.0)
    try:
        least_squares(residual, u0, jac=jac, bounds=(0.0, 1.0), method="trf", x_scale=1.0, ftol=1e-10, xtol=1e-10,
                      gtol=1e-10, max_nfev=max_evaluations)
    except _Done:
        pass
    u = best["u"]
    at_bounds = [n for i, n in enumerate(names) if u[i] <= 1e-3 or u[i] >= 1 - 1e-3]
    if best["res"] <= tolerance:
        status = "converged"
    elif at_bounds:
        status = "bounded_best"
    elif eq.evaluations >= max_evaluations:
        status = "max_evaluations"
    else:
        status = "stalled"
    out = best["out"]
    return InverseResult(values(u), dict(zip(outs, map(float, out))), dict(zip(outs, map(float, target))),
                         dict(zip(outs, map(float, out - target))), best["res"], status, at_bounds, eq.evaluations,
                         eq.steps, eq.unsettled, time.perf_counter() - t0, history)


METHODS = ("repeated", "sofa-qp")


def run_job(path: str | Path, method: str | None = None) -> dict[str, Any]:
    """Run a job file (see ``jobs/inverse_cable_cantilever.yaml``) and return the result as a dict.

    ``method`` (or the job's ``method:`` key; default ``repeated``) selects the repeated equilibrium solve of this
    module or ``sofa-qp``, SOFA's QP inverse solver (:mod:`softsim.control.sofa_inverse`)."""
    path = Path(path).resolve()
    job = yaml.safe_load(path.read_text()) or {}
    for key in ("config", "targets", "actuators"):
        if key not in job:
            raise ConfigError(f"{path}: missing {key!r}")
    method = method or job.pop("method", "repeated")
    job.pop("method", None)
    if method not in METHODS:
        raise ConfigError(f"{path}: unknown method {method!r}; choose from {list(METHODS)}")
    if method == "sofa-qp":
        from softsim.control.sofa_inverse import run_job_sofa

        return run_job_sofa(path, job)
    known = {"config", "overrides", "backend", "targets", "actuators", "initial", "tolerance", "max_evaluations", "scales",
             "settle_damping", "settle_tol", "settle_window", "max_settle", "warm_start", "jacobian_step"}
    extra = set(job) - known
    if extra:
        raise ConfigError(f"{path}: unknown keys {sorted(extra)}")
    kwargs = {k: v for k, v in job.items() if k not in ("config", "targets", "actuators")}
    bounds = {n: tuple(b) for n, b in job["actuators"].items()}
    result = solve_inverse(path.parent / job["config"], job["targets"], list(bounds), bounds, **kwargs)
    return result.to_dict()

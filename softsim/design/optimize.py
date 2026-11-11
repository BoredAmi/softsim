"""Design optimisation: tune geometry (or any config value) to maximise, minimise or hit a summary metric.

A job file (``softsim optimize jobs/cable_finger_stroke.yaml``)::

    scenario: ../scenarios/actuation/cable_finger.yaml   # relative to the job file
    backend: mujoco                                       # optional, overrides the scenario's
    overrides: [body.mesh.size=3mm, simulation.duration=0.3]
    parameters:                                           # the design variables, dotted keys
      body.geometry.slot_depth: {initial: 3 mm, bounds: [1 mm, 6 mm]}              # linear by default
      body.geometry.slot_width: {initial: 3 mm, bounds: [1 mm, 5 mm], log: true}
      body.geometry.segments:   {initial: 4, bounds: [2, 6], integer: true}        # rounded before use
      body.geometry.tendon_offset:                                                 # one value, several keys
        {initial: 4.5 mm, bounds: [2 mm, 6 mm], tie: ["actuators[0].path[0][0]"]}
    objective: {metric: final_bend_angle, goal: maximize}   # or minimize, or `target: 0.5` (match a value)
    #          per: cable_final_tension                      # optional: optimise metric / per
    constraints:                                            # summary metrics (or design keys) with bounds
      - {metric: cable_stroke, max: 8 mm}
    max_evaluations: 40
    output: ../runs/design/cable_finger

Units: bounds, initial values and constraint bounds are numbers (SI) or unit strings (``3 mm``, ``0.7 N``);
the unit carries its own dimension, and an optional ``dimension:`` (parameters, constraints) makes a
mismatching unit an error. Metrics are the scenario's summary (``softsim run`` prints them); any of them
can serve, and a constraint may also name a design key.

Method: Nelder-Mead (scipy) in the unit cube of the parameters (log or linear, bounds enforced), started
from ``initial``; the initial simplex is evaluated first (in parallel with ``--jobs``). Constraints are
**exterior penalties**: cost = f/|f0| + 100 * sum(violation), where f0 is the objective at the first
evaluation and a violation is how far a metric lies beyond its bound, relative to the bound. The penalty
is steep but finite, so the search can cross an infeasible region; the reported best is chosen among
**feasible** evaluations only. If no evaluation is feasible the report says so (``feasible: false``, the
violations) and the least-violating point is returned: infeasibility is never hidden (and the CLI exits
with status 2). A failed simulation (crash, unstable) costs 1000 and is counted in the report. Integer
parameters are rounded after the optimiser's step, so the cost is piecewise constant in them and
Nelder-Mead can stall on a plateau; give integers few values or scan them by hand. Nelder-Mead is local:
it finds the optimum nearest the start, not a global one.

Every evaluation is appended to ``<output>/evaluations.jsonl`` with its full summary; re-running the job
replays them instead of simulating (Nelder-Mead is deterministic), so a stopped job resumes where it
stopped and raising ``max_evaluations`` continues it. Entries are keyed by the scenario (its file and the
config it resolves to, ``extends`` included), softsim's own code (:func:`softsim.bench.code_hash`), the
overrides (incl. backend) and the parameter keys; the objective, constraints and bounds are not part of the
key (they only re-weigh stored summaries). The result is only as good as the backend: MuJoCo's linear tetrahedra
lock, which biases designs; check the winner on the ``fem`` reference (docs/design.md).

Outputs in ``output``: ``report.yaml`` (best design, objective, constraint values, feasibility),
``history.parquet`` (every evaluation in order, with all summary metrics), ``best_overrides.txt`` (the best
design as ``key=value`` lines for ``softsim run --set``) and ``evaluations.jsonl``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.units import _QUANTITY, _UNITS, parse_quantity
from softsim.errors import ConfigError
from softsim.io import save_table

FAILED_COST = 1000.0
PENALTY = 100.0
_PARAM_FIELDS = {"initial", "bounds", "log", "integer", "dimension", "tie"}
_JOB_FIELDS = {"scenario", "backend", "overrides", "parameters", "objective", "constraints", "max_evaluations",
               "output", "simplex_step"}


@dataclass(frozen=True)
class DesignParameter:
    key: str
    initial: float
    lower: float
    upper: float
    log: bool = False
    integer: bool = False
    tie: tuple[str, ...] = ()

    def _span(self) -> tuple[float, float]:
        return (math.log(self.lower), math.log(self.upper)) if self.log else (self.lower, self.upper)

    def from_unit(self, u: float) -> float:
        """Design value (SI) at position ``u`` in [0, 1] of the bounds (log or linear)."""
        a, b = self._span()
        value = a + min(max(float(u), 0.0), 1.0) * (b - a)
        value = min(max(math.exp(value) if self.log else value, self.lower), self.upper)
        return float(round(value)) if self.integer else value

    def to_unit(self, value: float) -> float:
        a, b = self._span()
        return ((math.log(value) if self.log else value) - a) / (b - a)


@dataclass(frozen=True)
class Constraint:
    metric: str
    min: float | None = None
    max: float | None = None

    def violation(self, value: float) -> float:
        """Amount by which ``value`` lies outside the bounds, relative to the bound (0 when satisfied)."""
        if not math.isfinite(value):
            return 1e3
        v = 0.0
        if self.min is not None:
            v = max(v, (self.min - value) / max(abs(self.min), 1e-12))
        if self.max is not None:
            v = max(v, (value - self.max) / max(abs(self.max), 1e-12))
        return v


@dataclass(frozen=True)
class Objective:
    metric: str
    goal: str = "maximize"            # maximize | minimize | target
    target: float | None = None
    per: str | None = None

    def value(self, summary: dict[str, float]) -> float:
        v = float(summary[self.metric])
        if self.per is not None:
            d = float(summary[self.per])
            v = v / d if d != 0 else float("nan")
        return v

    def signed(self, value: float) -> float:
        """The quantity to minimise."""
        if self.goal == "maximize":
            return -value
        if self.goal == "minimize":
            return value
        return abs(value - float(self.target))


def _quantity(value: Any, dimension: str | None, where: str) -> float:
    """SI float of a number or unit string; the unit's own dimension must match ``dimension`` if given."""
    if isinstance(value, str):
        m = _QUANTITY.match(value)
        if m and m.group(2) in _UNITS:
            own = _UNITS[m.group(2)][1]
            if dimension and own != dimension:
                raise ConfigError(f"{where}: {value!r} is a {own}, expected a {dimension}")
            dimension = own
    try:
        return parse_quantity(value, dimension or "length")
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}") from None


def _common_dimension(values: list[Any], where: str) -> str | None:
    """The dimension shared by the unit strings among ``values`` (None if there are none)."""
    dims = set()
    for v in values:
        m = _QUANTITY.match(v) if isinstance(v, str) else None
        if m and m.group(2) in _UNITS:
            dims.add(_UNITS[m.group(2)][1])
    if len(dims) > 1:
        raise ConfigError(f"{where}: bounds and initial mix units of different dimensions ({', '.join(sorted(dims))})")
    return next(iter(dims), None)


def parse_parameters(spec: Any) -> list[DesignParameter]:
    if not isinstance(spec, dict) or not spec:
        raise ConfigError("job needs a non-empty 'parameters' mapping of dotted keys")
    out = []
    for key, p in spec.items():
        if not isinstance(p, dict):
            raise ConfigError(f"parameter {key}: expected a mapping with initial and bounds")
        unknown = set(p) - _PARAM_FIELDS
        if unknown:
            raise ConfigError(f"parameter {key}: unknown fields {sorted(unknown)}; known: {sorted(_PARAM_FIELDS)}")
        if "bounds" not in p or "initial" not in p:
            raise ConfigError(f"parameter {key}: 'initial' and 'bounds' are required")
        if not isinstance(p["bounds"], (list, tuple)) or len(p["bounds"]) != 2:
            raise ConfigError(f"parameter {key}: bounds must be [lower, upper]")
        dim = p.get("dimension") or _common_dimension([*p["bounds"], p["initial"]], f"parameter {key}")
        lo, hi = (_quantity(b, dim, f"parameter {key} bounds") for b in p["bounds"])
        init = _quantity(p["initial"], dim, f"parameter {key} initial")
        if not lo < hi:
            raise ConfigError(f"parameter {key}: lower bound {lo} must be below upper bound {hi}")
        if not lo <= init <= hi:
            raise ConfigError(f"parameter {key}: initial {init} outside bounds [{lo}, {hi}]")
        log = bool(p.get("log", False))
        if log and lo <= 0:
            raise ConfigError(f"parameter {key}: log scale needs positive bounds")
        tie = p.get("tie", [])
        if not isinstance(tie, list):
            raise ConfigError(f"parameter {key}: tie must be a list of dotted keys")
        out.append(DesignParameter(key, init, lo, hi, log, bool(p.get("integer", False)), tuple(tie)))
    return out


def parse_objective(spec: Any) -> Objective:
    if not isinstance(spec, dict) or "metric" not in spec:
        raise ConfigError("job needs an 'objective' with a 'metric'")
    unknown = set(spec) - {"metric", "goal", "target", "per"}
    if unknown:
        raise ConfigError(f"objective: unknown fields {sorted(unknown)}")
    if "target" in spec:
        if spec.get("goal", "target") != "target":
            raise ConfigError("objective: 'target' goes with goal: target (or no goal)")
        return Objective(spec["metric"], "target", _quantity(spec["target"], None, "objective target"),
                         spec.get("per"))
    goal = spec.get("goal")
    if goal not in ("maximize", "minimize"):
        raise ConfigError(f"objective: goal must be maximize or minimize (or give a target), got {goal!r}")
    return Objective(spec["metric"], goal, None, spec.get("per"))


def parse_constraints(spec: Any) -> list[Constraint]:
    if spec is not None and not isinstance(spec, list):
        raise ConfigError("constraints must be a list")
    out = []
    for i, c in enumerate(spec or []):
        if not isinstance(c, dict) or "metric" not in c:
            raise ConfigError(f"constraints[{i}]: expected a mapping with a 'metric'")
        unknown = set(c) - {"metric", "min", "max", "dimension"}
        if unknown:
            raise ConfigError(f"constraints[{i}]: unknown fields {sorted(unknown)}")
        if "min" not in c and "max" not in c:
            raise ConfigError(f"constraints[{i}] ({c['metric']}): give min and/or max")
        lo = _quantity(c["min"], c.get("dimension"), f"constraints[{i}].min") if "min" in c else None
        hi = _quantity(c["max"], c.get("dimension"), f"constraints[{i}].max") if "max" in c else None
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(f"constraints[{i}] ({c['metric']}): min {lo} above max {hi}")
        out.append(Constraint(c["metric"], lo, hi))
    return out


_STEP = re.compile(r"^([^\[\]]+)((?:\[\d+\])*)$")


def _steps(key: str) -> list[str | int]:
    """``"actuators[0].path[1][0]"`` -> ``["actuators", 0, "path", 1, 0]`` (nested list indices allowed)."""
    out: list[str | int] = []
    for part in key.split("."):
        m = _STEP.match(part)
        if not m:
            raise ConfigError(f"bad parameter key {key!r}")
        out.append(m.group(1))
        out.extend(int(i) for i in re.findall(r"\[(\d+)\]", m.group(2)))
    return out


def get_path(config: Any, key: str) -> Any:
    node = config
    for step in _steps(key):
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            raise KeyError(key) from None
    return node


def set_path(config: Any, key: str, value: Any) -> None:
    """Set an existing entry of ``config`` in place (the key must exist: a typo must not optimise nothing)."""
    steps = _steps(key)
    get_path(config, key)
    node = config
    for step in steps[:-1]:
        node = node[step]
    node[steps[-1]] = value


def _run_summary(task: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Simulate one design (worker process) and return its summary, or ``{"error": ...}``."""
    from softsim.runner import Simulation

    raw, base_dir = task
    try:
        sim = Simulation(raw, base_dir=base_dir)
        try:
            summary = sim.run().summary
        finally:
            sim.close()
        return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, np.floating))}
    except Exception as exc:  # noqa: BLE001 - an unstable or invalid design costs the maximum
        return {"error": f"{type(exc).__name__}: {exc}"}


def _process_pool(workers: int):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Optimization:
    """Runs a design-optimisation job file; see the module docstring.

    Checked against: a cable cantilever whose tip angle grows with the cable offset (the maximum must sit
    at the upper bound) and whose target angle has a bisection reference on the same forward map
    (``tests/test_optimize.py``).
    """

    path: Path
    jobs: int = 1
    history: list[dict[str, Any]] = field(default_factory=list)
    n_simulated: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path).resolve()
        if not self.path.is_file():
            raise ConfigError(f"job file not found: {self.path}")
        spec = load_yaml(self.path)
        if not isinstance(spec, dict):
            raise ConfigError("job file must be a mapping")
        unknown = set(spec) - _JOB_FIELDS
        if unknown:
            raise ConfigError(f"unknown job keys: {sorted(unknown)}; known: {sorted(_JOB_FIELDS)}")
        if "scenario" not in spec:
            raise ConfigError("job needs a 'scenario' config file")
        scenario = (self.path.parent / spec["scenario"]).resolve()
        if not scenario.is_file():
            raise ConfigError(f"scenario file not found: {scenario}")
        self.params = parse_parameters(spec.get("parameters"))
        self.objective = parse_objective(spec.get("objective"))
        self.constraints = parse_constraints(spec.get("constraints"))
        try:
            self.max_evaluations = int(spec.get("max_evaluations", 40))
            self.simplex_step = float(spec.get("simplex_step", 0.25))
        except (TypeError, ValueError):
            raise ConfigError("max_evaluations must be an integer and simplex_step a number") from None
        if self.max_evaluations < 1:
            raise ConfigError("max_evaluations must be at least 1")
        if not 0 < self.simplex_step <= 1:
            raise ConfigError("simplex_step must be in (0, 1] (fraction of each parameter's range)")
        overrides = list(spec.get("overrides", []))
        if spec.get("backend"):
            overrides.append(f"backend={spec['backend']}")
        self.base_dir = str(scenario.parent)
        self._raw = apply_overrides(load_yaml(scenario), overrides)
        self.output = (self.path.parent / spec.get("output", "design_out")).resolve()
        for p in self.params:  # a typo would otherwise silently optimise nothing
            for k in (p.key, *p.tie):
                try:
                    get_path(self._raw, k)
                except KeyError:
                    raise ConfigError(f"parameter key {k!r} does not exist in the scenario (after overrides)") from None
        from softsim.bench import code_hash

        self._key_hash = _hash({"scenario": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                                "resolved": load_yaml(scenario),  # a changed `extends` parent counts too
                                "code": code_hash(),  # and so does softsim's own code
                                "overrides": overrides,
                                "parameters": {p.key: [p.integer, list(p.tie)] for p in self.params}})
        self._log = self.output / "evaluations.jsonl"
        self._done = self._load_log()
        self._pool = None
        self._fresh: set[str] = set()   # simulated in this run, not yet reported
        self._f0: float | None = None

    @staticmethod
    def _key(values: dict[str, float]) -> str:
        return json.dumps({k: float(f"{v:.12g}") for k, v in values.items()}, sort_keys=True)

    def _load_log(self) -> dict[str, dict[str, Any]]:
        done = {}
        if self._log.is_file():
            for line in self._log.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("spec") == self._key_hash:
                    done[self._key(entry["values"])] = entry["summary"]
        return done

    def _config(self, values: dict[str, float]) -> dict[str, Any]:
        raw = copy.deepcopy(self._raw)
        for p in self.params:
            for key in (p.key, *p.tie):
                set_path(raw, key, values[p.key])
        return raw

    def overrides_for(self, values: dict[str, float]) -> list[str]:
        """``key=value`` lines (SI) applying a design with ``softsim run --set``, tied keys included. A key
        with a nested list index (``path[1][0]``) is written as the whole inner list (``path[1]=[...]``),
        which the config loader can set."""
        raw = self._config(values)
        out: list[str] = []
        for p in self.params:
            for key in (p.key, *p.tie):
                if re.search(r"\]\[\d+\]$", key):
                    key = key[:key.rindex("[")]
                    line = f"{key}={json.dumps(get_path(raw, key))}"
                else:
                    line = f"{key}={float(values[p.key]):.10g}"
                if line not in out:
                    out.append(line)
        return out

    def _simulate(self, tasks: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
        from concurrent.futures.process import BrokenProcessPool

        if self._pool is None:
            return [_run_summary(t) for t in tasks]
        try:
            return list(self._pool.map(_run_summary, tasks))
        except BrokenProcessPool:  # a native crash killed a worker: one fresh process per design
            self._pool.shutdown(cancel_futures=True)
            self._pool = _process_pool(self.jobs)
            out = []
            for t in tasks:
                with _process_pool(1) as single:
                    try:
                        out.append(single.submit(_run_summary, t).result())
                    except BrokenProcessPool:
                        out.append({"error": "worker process crashed (native error in the backend)"})
            return out

    def _ensure(self, designs: list[dict[str, float]]) -> None:
        """Simulate the designs not yet in the log (in parallel when ``jobs`` > 1) and log them."""
        todo = {}
        for v in designs:
            todo.setdefault(self._key(v), v)
        todo = {k: v for k, v in todo.items() if k not in self._done}
        if not todo:
            return
        results = self._simulate([(self._config(v), self.base_dir) for v in todo.values()])
        self.output.mkdir(parents=True, exist_ok=True)
        with self._log.open("a") as f:
            for (k, v), summary in zip(todo.items(), results):
                self.n_simulated += 1
                self._fresh.add(k)
                self._done[k] = summary
                f.write(json.dumps({"spec": self._key_hash, "values": v, "summary": summary}, default=float) + "\n")

    def _score(self, values: dict[str, float], summary: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"evaluation": len(self.history), **values}
        if "error" in summary:
            row.update(error=summary["error"], objective=float("nan"), violation=float("inf"), feasible=False,
                       cost=FAILED_COST)
            return row
        try:
            obj = self.objective.value(summary)
            lookup = {**values, **summary}
            cons = {c.metric: float(lookup[c.metric]) for c in self.constraints}
        except KeyError as exc:
            raise ConfigError(f"metric {exc.args[0]!r} is not in the scenario's summary; available: "
                              f"{', '.join(sorted(summary))}") from None
        viol = sum(c.violation(cons[c.metric]) for c in self.constraints)
        if not math.isfinite(obj):
            row.update(error="non-finite objective", objective=obj, violation=viol, feasible=False, cost=FAILED_COST)
            return row
        signed = self.objective.signed(obj)
        if self._f0 is None:
            self._f0 = max(abs(signed), abs(obj), 1e-12)
        row.update(summary)
        row.update(objective=obj, violation=viol, feasible=viol <= 0.0, cost=signed / self._f0 + PENALTY * viol)
        row.update({f"constraint.{k}": v for k, v in cons.items()})
        return row

    def evaluate(self, values: dict[str, float]) -> float:
        self._ensure([values])
        replayed = self._key(values) not in self._fresh
        self._fresh.discard(self._key(values))
        row = self._score(values, self._done[self._key(values)])
        self.history.append(row)
        flag = ("" if row["feasible"] else f"  FAILED ({row['error'][:90]})" if "error" in row
                else "  INFEASIBLE")
        print(f"  eval {row['evaluation']:3d}: objective {row['objective']:.5g}  cost {row['cost']:.4f}"
              f"{' (cached)' if replayed else ''}{flag}  " +
              "  ".join(f"{k.split('.')[-1]}={v:.5g}" for k, v in values.items()), flush=True)
        return row["cost"]

    def run(self) -> dict[str, Any]:
        from scipy.optimize import minimize

        start = time.perf_counter()
        self._pool = _process_pool(self.jobs) if self.jobs > 1 else None
        try:
            x0 = np.array([p.to_unit(p.initial) for p in self.params])
            steps = np.where(x0 + self.simplex_step <= 1.0, self.simplex_step, -self.simplex_step)
            simplex = np.vstack([x0] + [x0 + steps[i] * e for i, e in enumerate(np.eye(len(x0)))])

            def design(u: np.ndarray) -> dict[str, float]:
                return {p.key: p.from_unit(x) for p, x in zip(self.params, u)}

            self._ensure([design(u) for u in simplex][: self.max_evaluations])  # the simplex, in parallel
            minimize(lambda u: self.evaluate(design(u)), x0, method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(x0),
                     options={"maxfev": self.max_evaluations, "initial_simplex": simplex, "xatol": 1e-3,
                              "fatol": 1e-6})
        finally:
            if self._pool:
                self._pool.shutdown()
                self._pool = None
        return self._report(time.perf_counter() - start)

    def _report(self, wall: float) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        save_table(pd.DataFrame(self.history), self.output / "history")
        ok = [r for r in self.history if "error" not in r]
        if not ok:
            raise ConfigError("every evaluation failed; first error: " + str(self.history[0].get("error")))
        feasible = [r for r in ok if r["feasible"]]
        if feasible:
            best = min(feasible, key=lambda r: self.objective.signed(r["objective"]))
        else:
            best = min(ok, key=lambda r: (r["violation"], r["cost"]))
        values = {p.key: float(best[p.key]) for p in self.params}
        report: dict[str, Any] = {
            "best": values,
            "objective": {"metric": self.objective.metric, "goal": self.objective.goal, "value": best["objective"]},
            "feasible": bool(best["feasible"]), "evaluations": len(self.history), "simulated": self.n_simulated,
            "failed": sum("error" in r for r in self.history), "wall_time_s": wall,
            "constraints": {c.metric: {"value": best[f"constraint.{c.metric}"], "min": c.min, "max": c.max,
                                       "violation": c.violation(best[f"constraint.{c.metric}"])}
                            for c in self.constraints},
            "at_bounds": [p.key for p in self.params
                          if min(abs(values[p.key] - p.lower), abs(values[p.key] - p.upper))
                          <= 1e-3 * (p.upper - p.lower)],
            "overrides": self.overrides_for(values),
        }
        if self.objective.goal == "target":
            report["objective"]["target"] = self.objective.target
        if not best["feasible"]:
            report["warning"] = ("NO FEASIBLE DESIGN FOUND: the constraints could not all be met within the "
                                 "bounds and evaluations; 'best' is the least-violating design")
            print("warning: " + report["warning"], flush=True)
        (self.output / "report.yaml").write_text(
            yaml.safe_dump(json.loads(json.dumps(report, default=float)), sort_keys=False))
        (self.output / "best_overrides.txt").write_text("\n".join(report["overrides"]) + "\n")
        return report


def best_overrides(report: dict[str, Any]) -> list[str]:
    """Dotted ``key=value`` overrides (SI) that apply a report's best design (``softsim run --set``)."""
    return [f"{k}={float(v):.10g}" for k, v in report["best"].items()]

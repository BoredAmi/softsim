"""Fit material parameters to tracked experiments, all cases at once.

A calibration file::

    dataset: ../datasets/czopek2026_ecoflex_drop.yaml
    tracks: ../data/processed/czopek2026_ecoflex_drop
    cases: {w020: 1.0, w050: 1.0, w500: 1.0}          # case -> weight in the cost
    overrides: [specimen.mesh.size=0.005, simulation.duration=0.02]
    parameters:
      specimen.material.elastic.youngs_modulus: {initial: 50 kPa, bounds: [5 kPa, 1 MPa], log: true}
      specimen.material.damping.beta: {initial: 2 ms, bounds: [0.01 ms, 30 ms], log: true}
    grid: 4                     # optional coarse grid (points per parameter) before the local search
    max_evaluations: 60
    output: ../calibrations/ecoflex_00_10_drop

Each case is simulated with its measured impact (speed, offset, rotation rate;
see ``ExperimentCase.sim_overrides``) and compared on the first contact phase
(``compare.py``). Cost = weighted mean of RMSE / experimental peak. The fitted
values are only meaningful for the mesh size, backend and model they were fitted
with; the saved preset records that in ``provenance``.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from softsim.config.loader import apply_overrides, load_yaml, set_dotted
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.experiments.compare import ExperimentCase, compare_case, simulated_indentation
from softsim.experiments.datasets import load_dataset
from softsim.io import save_table

_DIMENSIONS = {"youngs_modulus": "pressure", "beta": "time", "timeconst": "time", "c10": "pressure",
               "c01": "pressure", "bulk_modulus": "pressure", "density": "density"}


@dataclass(frozen=True)
class Parameter:
    key: str
    initial: float
    lower: float
    upper: float
    log: bool = True

    def to_unit(self, value: float) -> float:
        return math.log(value) if self.log else value

    def from_unit(self, u: float) -> float:
        return float(np.clip(math.exp(u) if self.log else u, self.lower, self.upper))


def _parse_parameters(spec: dict[str, Any]) -> list[Parameter]:
    params = []
    for key, p in spec.items():
        dim = _DIMENSIONS.get(key.split(".")[-1])
        conv = (lambda v: parse_quantity(v, dim)) if dim else float
        lo, hi = (conv(b) for b in p["bounds"])
        init = conv(p["initial"])
        if not lo <= init <= hi:
            raise ConfigError(f"{key}: initial {init} outside bounds [{lo}, {hi}]")
        params.append(Parameter(key, init, lo, hi, bool(p.get("log", True))))
    return params


def _evaluate_case(task: tuple[dict[str, Any], str, str, str]) -> dict[str, float]:
    """Run one case (worker process) and return its comparison metrics."""
    from softsim.runner import Simulation

    raw, base_dir, tracks, case = task
    exp = ExperimentCase.load(tracks, case)
    raw = apply_overrides(raw, exp.sim_overrides())
    try:
        sim = Simulation(raw, base_dir=base_dir)
        try:
            ts = sim.run().timeseries
        finally:
            sim.close()
        metrics = compare_case(simulated_indentation(ts), exp)
    except Exception as exc:  # unstable / invalid parameter sets cost the maximum
        metrics = {"rmse": float("nan"), "nrmse": float("nan"), "error": f"{type(exc).__name__}: {exc}"}
    return metrics


class Calibration:
    """Runs a calibration file; see module docstring."""

    def __init__(self, path: str | Path, jobs: int = 1):
        self.path = Path(path).resolve()
        spec = load_yaml(self.path)
        base = self.path.parent
        self.dataset = load_dataset(base / spec["dataset"])
        self.tracks = str((base / spec["tracks"]).resolve())
        self.cases: dict[str, float] = {k: float(v) for k, v in spec["cases"].items()}
        self.params = _parse_parameters(spec["parameters"])
        self.overrides = list(spec.get("overrides", []))
        self.grid = int(spec.get("grid", 0))
        self.max_evaluations = int(spec.get("max_evaluations", 60))
        self.output = (base / spec.get("output", "calibration_out")).resolve()
        self.jobs = jobs
        self.history: list[dict[str, Any]] = []
        self._raw = {}
        for case in self.cases:
            cfg_path = self.dataset.config_path(case)
            self._raw[case] = (apply_overrides(load_yaml(cfg_path), self.overrides), str(cfg_path.parent))

    def _configs(self, values: dict[str, float]) -> list[tuple[dict[str, Any], str, str, str]]:
        tasks = []
        for case, (raw, base_dir) in self._raw.items():
            r = raw
            for key, value in values.items():
                r = set_dotted(r, key, value)
            tasks.append((r, base_dir, self.tracks, case))
        return tasks

    def evaluate(self, values: dict[str, float], pool: ProcessPoolExecutor | None) -> float:
        tasks = self._configs(values)
        results = list(pool.map(_evaluate_case, tasks)) if pool else [_evaluate_case(t) for t in tasks]
        weights = np.array([self.cases[t[3]] for t in tasks])
        nrmse = np.array([r.get("nrmse", np.nan) for r in results])
        cost = float(np.sum(weights * np.where(np.isfinite(nrmse), nrmse, 10.0)) / weights.sum())
        row = {"evaluation": len(self.history), "cost": cost, **values}
        for t, r in zip(tasks, results):
            row.update({f"{t[3]}.{k}": v for k, v in r.items()})
        self.history.append(row)
        print(f"  eval {row['evaluation']:3d}: cost {cost:.4f}  " +
              "  ".join(f"{k.split('.')[-1]}={v:.4g}" for k, v in values.items()), flush=True)
        return cost

    def run(self) -> dict[str, Any]:
        from scipy.optimize import minimize

        start = time.perf_counter()
        pool = ProcessPoolExecutor(min(self.jobs, len(self.cases))) if self.jobs > 1 else None
        try:
            def objective(u: np.ndarray) -> float:
                return self.evaluate({p.key: p.from_unit(x) for p, x in zip(self.params, u)}, pool)

            x0 = np.array([p.to_unit(p.initial) for p in self.params])
            if self.grid > 1:  # coarse grid in (log) space to pick the starting point
                axes = [np.linspace(p.to_unit(p.lower), p.to_unit(p.upper), self.grid + 2)[1:-1] for p in self.params]
                costs = {pt: objective(np.array(pt)) for pt in itertools.product(*axes)}
                x0 = np.array(min(costs, key=costs.get))
            remaining = max(self.max_evaluations - len(self.history), 1)
            simplex_step = 0.3  # ~35 % in log space
            init_simplex = np.vstack([x0] + [x0 + simplex_step * e for e in np.eye(len(x0))])
            minimize(objective, x0, method="Nelder-Mead",
                     options={"maxfev": remaining, "initial_simplex": init_simplex, "xatol": 0.01, "fatol": 1e-4})
        finally:
            if pool:
                pool.shutdown()
        best = min(self.history, key=lambda r: r["cost"])
        return self._report(best, time.perf_counter() - start)

    def _report(self, best: dict[str, Any], wall: float) -> dict[str, Any]:
        from softsim.materials import load_material, save_material

        self.output.mkdir(parents=True, exist_ok=True)
        history = pd.DataFrame(self.history)
        save_table(history, self.output / "history")
        values = {p.key: best[p.key] for p in self.params}

        # fitted material preset with provenance
        first_case = next(iter(self.cases))
        raw = self._raw[first_case][0]
        for k, v in values.items():
            raw = set_dotted(raw, k, v)
        mat_ref = raw["specimen"]["material"]
        material = load_material(mat_ref, base_dir=Path(self._raw[first_case][1]))
        mesh = raw["specimen"].get("mesh", {})
        provenance = {
            "fitted_by": "softsim calibrate",
            "calibration_file": _relative(self.path),
            "dataset_doi": self.dataset.doi,
            "cases": self.cases,
            "backend": raw.get("backend", "mujoco"),
            "mesh_size": mesh.get("size"),
            "timestep": raw.get("simulation", {}).get("timestep"),
            "contact_timeconst": raw.get("contact", {}).get("timeconst"),
            "cost": best["cost"],
            "per_case_nrmse": {c: best.get(f"{c}.nrmse") for c in self.cases},
            "note": "Effective parameters: valid only with this backend, element type and mesh size.",
        }
        fitted = material.model_copy(update={
            "name": f"{material.name}_fit_{self.dataset.name}",
            "source": f"Fitted to {self.dataset.title} (doi:{self.dataset.doi}) by softsim calibrate.",
            "provenance": provenance,
        })
        save_material(fitted, self.output / f"{fitted.name}.yaml")
        report = {"best": {k: float(v) for k, v in values.items()}, "cost": best["cost"],
                  "evaluations": len(self.history), "wall_time_s": wall,
                  "per_case": {c: {k.split(".", 1)[1]: best[k] for k in best if k.startswith(c + ".")}
                               for c in self.cases},
                  "material_file": str(self.output / f"{fitted.name}.yaml")}
        (self.output / "report.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(report, default=float)),
                                                                sort_keys=False))
        return report


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def best_overrides(report: dict[str, Any]) -> list[str]:
    """Dotted overrides that apply a calibration report's best values."""
    return [f"{k}={v}" for k, v in report["best"].items()]

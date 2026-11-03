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
(``compare.py``). A case may instead list observables and windows::

    cases:
      w020: {weight: 1.0, observables: {indentation_rel: [0, 100 ms], specimen_lift: [0, 100 ms]}}
      w050: {observables: {indentation_rel: first_contact}}

Material-test data can join the same cost (a joint fit of lab tests and the
drop), one term per entry, evaluated on the card's homogeneous-test curves
(``softsim.materials.homogeneous``; milliseconds, no simulation)::

    material_tests:
      - {dataset: ../datasets/roels2025_elastomers.yaml, material: ecoflex_00_30, kind: tension,
         strain: [0, 1.0], weight: 0.5}
      - {dataset: ../datasets/roels2025_elastomers.yaml, material: ecoflex_00_30, kind: compression,
         strain: [-0.25, 0]}
      - {name: datasheet, points: {strain: [1.0], stress: [55 kPa]}, weight: 0.5}   # e.g. "100% modulus"
    material_key: specimen.material      # where the fitted card sits in the scenario (default)

A term's NRMSE is the mean over the specimens of RMSE / the specimen's largest
stress. ``kind: relaxation`` compares the normalised hold instead (stress over
its value at the recorded hold's start): the card is ramped at ``rate``
(default 0.4/s) to each recording's hold strain, and the recorded hold starts
``hold_offset`` (default 1.3 s) after the ramp, as in Roels et al. 2025. With a Prony card and a test rate (dataset ``strain_rate``, or ``rate:``
in the entry) the curve is loaded at that rate, otherwise fully relaxed.

Per-case cost = mean over its observables of RMSE / largest experimental value
in the window; total cost = weighted mean over cases and material-test terms. The fitted
values are only meaningful for the mesh size, backend and model they were fitted
with; the saved preset records that in ``provenance``.

Every evaluation is appended to ``<output>/evaluations.jsonl``. Re-running the
calibration replays them from there instead of simulating again (the optimizer
is deterministic), so an interrupted run resumes where it stopped. Entries are
keyed by what determines a simulation's result (dataset, tracks, cases,
overrides, parameter names, and the values), so changing ``max_evaluations``,
``grid``, bounds or initial values keeps them; changing the rest starts over.
The key does not cover softsim's own code: after changing simulation code,
delete the log, or old results are replayed as if they were current. It does cover the **contents** of
the tracks: each entry records a digest of the cases' track files, and entries made against other tracks
(e.g. reference tracks regenerated from a new card, same path) are ignored with a note. Entries from
before the digest existed carry none and are still trusted. A simulation that crashes its worker
process (e.g. a native segfault) costs the same as an unstable one. Workers are spawned, not forked
(``_process_pool``), so they work on a GPU after the parent has used JAX.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from softsim.config.loader import apply_overrides, get_dotted, load_yaml, set_dotted
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.experiments.compare import OBSERVABLES, ExperimentCase, compare_observable, simulated_indentation
from softsim.experiments.datasets import load_dataset
from softsim.io import save_table

_DIMENSIONS = {"youngs_modulus": "pressure", "beta": "time", "timeconst": "time", "c10": "pressure",
               "c01": "pressure", "bulk_modulus": "pressure", "density": "density", "mu": "pressure",
               "strength": "pressure", "range": "length", "stiffness": "stiffness", "damping": "viscous_damping",
               "mass": "mass", "tau": "time", "c20": "pressure", "c30": "pressure", "shear_modulus": "pressure",
               "viscosity": "viscosity"}
FIRST_CONTACT = "first_contact"


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
        dim = _DIMENSIONS.get(key.split(".")[-1].split("[")[0])
        conv = (lambda v: parse_quantity(v, dim)) if dim else float
        lo, hi = (conv(b) for b in p["bounds"])
        init = conv(p["initial"])
        if not lo <= init <= hi:
            raise ConfigError(f"{key}: initial {init} outside bounds [{lo}, {hi}]")
        params.append(Parameter(key, init, lo, hi, bool(p.get("log", True))))
    return params


Window = tuple[float, float] | None  # None: first contact phase


def _parse_cases(spec: dict[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, Window]]]:
    """Case weights and, per case, the observables to compare with their windows."""
    weights, observables = {}, {}
    for case, value in spec.items():
        if not isinstance(value, dict):
            weights[case], observables[case] = float(value), {"indentation_rel": None}
            continue
        weights[case] = float(value.get("weight", 1.0))
        obs = {}
        for name, window in (value.get("observables") or {"indentation_rel": FIRST_CONTACT}).items():
            if name not in OBSERVABLES:
                raise ConfigError(f"case {case}: unknown observable {name!r}; known: {', '.join(OBSERVABLES)}")
            if window == FIRST_CONTACT:
                if name != "indentation_rel":
                    raise ConfigError(f"case {case}: '{FIRST_CONTACT}' is only defined for indentation_rel")
                obs[name] = None
            else:
                t0, t1 = (parse_quantity(w, "time") for w in window)
                if not t0 < t1:
                    raise ConfigError(f"case {case}: empty window {window} for {name}")
                obs[name] = (t0, t1)
        observables[case] = obs
    return weights, observables


@dataclass(frozen=True)
class MaterialTestTerm:
    """One ``material_tests`` entry: specimens' curves of one test kind, compared with the card's."""

    name: str
    weight: float
    curves: tuple
    material_key: str = "specimen.material"
    kind: str = "curve"                   # "curve" (stress vs stretch) or "relaxation" (normalised hold)
    ramp_rate: float = 0.4                # relaxation: loading rate before the hold [1/s]
    hold_offset: float = 1.3              # relaxation: recorded hold starts this long after the ramp [s]

    def _relaxation_errors(self, card, c) -> np.ndarray:
        """Normalised relaxation of the card along the recording's history: ramp to the hold strain at
        ``ramp_rate``, hold; both curves divided by their value at the recorded hold's start."""
        from softsim.materials.homogeneous import history

        hold = float(c.meta["hold_strain"])
        ramp = hold / self.ramp_rate
        t0 = ramp + self.hold_offset
        t = np.unique(np.concatenate([np.linspace(0.0, ramp, 41), ramp + np.geomspace(1e-4, self.hold_offset, 40),
                                      t0 + np.asarray(c.time, float)]))
        x = 1.0 - hold * np.minimum(t / ramp, 1.0)  # compression
        s = history(card, "uniaxial", t, x).stress.to_numpy()
        model = np.interp(t0 + np.asarray(c.time, float), t, s)
        return model / model[0] - c.stress / c.stress[0]

    def evaluate(self, raw: dict[str, Any], base_dir: str) -> dict[str, float]:
        from softsim.materials import load_material
        from softsim.materials.fit import _model_stress

        try:
            card = load_material(get_dotted(raw, self.material_key), base_dir=Path(base_dir))
            if self.kind == "relaxation":
                errors = [self._relaxation_errors(card, c) for c in self.curves]
            else:
                errors = [(_model_stress(card, c) - c.stress) / float(np.abs(c.stress).max()) for c in self.curves]
            nrmse = [float(np.sqrt(np.mean(e**2))) for e in errors]
            return {"nrmse": float(np.mean(nrmse)), "max_error": float(max(np.abs(e).max() for e in errors)),
                    "specimens": len(self.curves)}
        except Exception as exc:  # noqa: BLE001 - an invalid card costs the maximum
            return {"nrmse": float("nan"), "error": f"{type(exc).__name__}: {exc}"}


def _parse_material_tests(spec: list[dict[str, Any]] | None, base: Path, material_key: str) -> list[MaterialTestTerm]:
    from softsim.experiments.material_tests import load_material_tests
    from softsim.materials.fit import _resample

    terms = []
    for i, t in enumerate(spec or []):
        if "points" in t:  # inline points, e.g. a datasheet's 100% modulus
            terms.append(_points_term(i, t, material_key))
            continue
        missing = {"dataset", "material"} - set(t)
        if missing:
            raise ConfigError(f"material_tests[{i}]: missing {', '.join(sorted(missing))}")
        ds = load_material_tests(base / t["dataset"])
        kind = t.get("kind", "tension")
        if kind not in ("tension", "compression", "relaxation"):
            raise ConfigError(f"material_tests[{i}]: kind must be tension, compression or relaxation, got {kind!r}")
        if kind == "relaxation":
            curves = tuple(ds.curves(t["material"], "relaxation"))
            terms.append(MaterialTestTerm(t.get("name", f"{t['material']}.relaxation"), float(t.get("weight", 1.0)),
                                          curves, material_key, "relaxation",
                                          parse_quantity(t.get("rate", 0.4), "rate"),
                                          parse_quantity(t.get("hold_offset", 1.3), "time")))
            continue
        entry = ds.entry(t["material"])
        curves = [ds.curve(t["material"])] if entry.member is not None else ds.curves(t["material"], kind)
        lo, hi = (float(v) for v in t.get("strain", [-1.0, 10.0]))
        rate = parse_quantity(t["rate"], "rate") if "rate" in t else None
        kept = []
        for c in curves:
            c = _resample(c.within(lo, hi))
            if len(c.stretch) >= 3:
                kept.append(c if rate is None else type(c)(c.stretch, c.stress, c.time, c.material, c.mode, rate,
                                                           dict(c.meta)))
        if not kept:
            raise ConfigError(f"material_tests[{i}]: no {kind} data of {t['material']} in strain [{lo}, {hi}]")
        terms.append(MaterialTestTerm(t.get("name", f"{t['material']}.{kind}"), float(t.get("weight", 1.0)),
                                      tuple(kept), material_key))
    return terms


def _points_term(i: int, t: dict[str, Any], material_key: str) -> MaterialTestTerm:
    """``{points: {strain: [1.0], stress: [55 kPa]}, mode: uniaxial, name: datasheet}``: nominal stress at
    engineering strains (shear strain for simple shear), fully relaxed unless ``rate`` is given."""
    from softsim.experiments.material_tests import TestCurve

    pts = t["points"]
    strain = np.atleast_1d(np.asarray(pts.get("strain", []), float))
    stress = np.array([parse_quantity(v, "pressure") for v in np.atleast_1d(pts.get("stress", []))])
    if len(strain) == 0 or len(strain) != len(stress):
        raise ConfigError(f"material_tests[{i}]: points need equally many strain and stress values")
    mode = t.get("mode", "uniaxial")
    x = strain if mode == "simple_shear" else 1.0 + strain
    rate = parse_quantity(t["rate"], "rate") if "rate" in t else None
    order = np.argsort(x)
    return MaterialTestTerm(t.get("name", f"points{i}"), float(t.get("weight", 1.0)),
                            (TestCurve(x[order], stress[order], None, "", mode, rate),), material_key)


def _tracks_digest(tracks: str | Path, cases) -> str:
    """Digest of the track files (``<case>.parquet`` and ``.yaml``) of ``cases``: what a log entry was fitted against."""
    h = hashlib.sha256()
    for case in sorted(cases):
        for suffix in (".parquet", ".yaml"):
            path = Path(tracks) / f"{case}{suffix}"
            h.update(f"{case}{suffix}".encode() + (path.read_bytes() if path.is_file() else b""))
    return h.hexdigest()[:16]


def _process_pool(workers: int) -> ProcessPoolExecutor:
    """Worker processes for the case simulations, started with ``spawn``.

    A forked worker inherits the parent's JAX state. Material-test terms run in the
    parent (``softsim.materials.homogeneous``, JAX), and once that has initialised
    CUDA, a forked child cannot use it: every ``fem`` case crashed its worker on a
    GPU machine. A spawned worker starts a fresh interpreter and its own context.
    """
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def _evaluate_case(task: tuple) -> dict[str, float]:
    """Run one case (worker process) and return its comparison metrics.

    ``nrmse`` is the mean over the case's observables; per-observable metrics are
    prefixed with the observable name (the first-contact indentation metrics are
    also reported unprefixed).
    """
    from softsim.runner import Simulation

    raw, base_dir, tracks, case = task[:4]
    observables = task[4] if len(task) > 4 else {"indentation_rel": None}
    exp = ExperimentCase.load(tracks, case)
    raw = apply_overrides(raw, exp.sim_overrides())
    try:
        sim = Simulation(raw, base_dir=base_dir)
        try:
            ts = sim.run().timeseries
        finally:
            sim.close()
        curves = simulated_indentation(ts)
        metrics: dict[str, Any] = {}
        parts = []
        for name, window in observables.items():
            m = compare_observable(curves, exp, name, window)
            parts.append(m["nrmse"])
            metrics.update({f"{name}.{k}": v for k, v in m.items()})
            if name == "indentation_rel" and window is None:
                metrics.update(m)
        if "indentation_rel" not in observables or observables["indentation_rel"] is not None:
            metrics.update(compare_observable(curves, exp, "indentation_rel", None))  # for reports and plots
        metrics["nrmse"] = float(np.mean(parts))
        if "specimen_min_z" in ts:  # contact artefacts (soft body through the ground) show up here
            metrics["ground_penetration"] = float(max(0.0, -ts["specimen_min_z"].min()))
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
        self.cases, self.observables = _parse_cases(spec["cases"])
        self.params = _parse_parameters(spec["parameters"])
        self.overrides = list(spec.get("overrides", []))
        self.material_tests = _parse_material_tests(spec.get("material_tests"), base,
                                                    spec.get("material_key", "specimen.material"))
        self.grid = int(spec.get("grid", 0))
        self.max_evaluations = int(spec.get("max_evaluations", 60))
        self.output = (base / spec.get("output", "calibration_out")).resolve()
        self.jobs = jobs
        self._workers = min(jobs, len(self.cases))
        self.history: list[dict[str, Any]] = []
        self._pool: ProcessPoolExecutor | None = None
        self._spec_hash = _hash({k: spec.get(k) for k in ("dataset", "tracks", "cases", "overrides")}
                                | {"parameters": sorted(spec["parameters"])})
        self._legacy_hash = _hash(spec)  # logs written before the key was narrowed
        self._log = self.output / "evaluations.jsonl"
        self._tracks_digest = _tracks_digest(self.tracks, self.cases)
        self._stale = 0  # logged evaluations ignored: made against other tracks
        self._done = self._load_log()
        self._raw = {}
        for case in self.cases:
            cfg_path = self.dataset.config_path(case)
            self._raw[case] = (apply_overrides(load_yaml(cfg_path), self.overrides), str(cfg_path.parent))

    @property
    def _term_names(self) -> list[str]:
        return [t.name for t in self.material_tests]

    def _configs(self, values: dict[str, float]) -> list[tuple]:
        tasks = []
        for case, (raw, base_dir) in self._raw.items():
            r = raw
            for key, value in values.items():
                r = set_dotted(r, key, value)
            tasks.append((r, base_dir, self.tracks, case, self.observables[case]))
        return tasks

    @staticmethod
    def _key(values: dict[str, float]) -> str:
        return json.dumps({k: float(f"{v:.12g}") for k, v in values.items()}, sort_keys=True)

    def _load_log(self) -> dict[str, list[dict[str, float]]]:
        """Results of earlier runs with the same simulations, keyed by parameter values."""
        done = {}
        if self._log.is_file():
            for line in self._log.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("spec") in (self._spec_hash, self._legacy_hash):
                    if entry.get("tracks") not in (None, self._tracks_digest):
                        self._stale += 1
                        continue
                    done[self._key(entry["values"])] = entry["results"]
        return done

    def _simulate(self, tasks: list[tuple[dict[str, Any], str, str, str]]) -> list[dict[str, Any]]:
        if self._pool is None:
            return [_evaluate_case(t) for t in tasks]
        try:
            return list(self._pool.map(_evaluate_case, tasks))
        except BrokenProcessPool:
            # a native crash killed a worker: find out which case(s), one fresh process each
            self._pool.shutdown(cancel_futures=True)
            self._pool = _process_pool(self._workers)
            results = []
            for t in tasks:
                with _process_pool(1) as single:
                    try:
                        results.append(single.submit(_evaluate_case, t).result())
                    except BrokenProcessPool:
                        results.append({"rmse": float("nan"), "nrmse": float("nan"),
                                        "error": "worker process crashed (native error in the backend)"})
            return results

    def evaluate(self, values: dict[str, float]) -> float:
        tasks = self._configs(values)
        key = self._key(values)
        replayed = key in self._done
        results = self._done[key] if replayed else self._simulate(tasks)
        if not replayed:
            self.output.mkdir(parents=True, exist_ok=True)
            with self._log.open("a") as f:
                f.write(json.dumps({"spec": self._spec_hash, "tracks": self._tracks_digest, "values": values,
                                "results": results}, default=float) + "\n")
        names = [t[3] for t in tasks]
        weights = [self.cases[n] for n in names]
        if self.material_tests:
            raw, base_dir = tasks[0][0], tasks[0][1]
            for term in self.material_tests:
                names.append(term.name)
                weights.append(term.weight)
                results = [*results, term.evaluate(raw, base_dir)]
        weights = np.array(weights)
        nrmse = np.array([r.get("nrmse", np.nan) for r in results])
        cost = float(np.sum(weights * np.where(np.isfinite(nrmse), nrmse, 10.0)) / weights.sum())
        row = {"evaluation": len(self.history), "cost": cost, **values}
        for name, r in zip(names, results):
            row.update({f"{name}.{k}": v for k, v in r.items()})
        self.history.append(row)
        print(f"  eval {row['evaluation']:3d}: cost {cost:.4f}{' (replayed)' if replayed else ''}  " +
              "  ".join(f"{k.split('.')[-1]}={v:.4g}" for k, v in values.items()), flush=True)
        return cost

    def run(self) -> dict[str, Any]:
        from scipy.optimize import minimize

        start = time.perf_counter()
        if self._stale:
            print(f"  note: {self._stale} logged evaluations were fitted against other tracks and are ignored", flush=True)
        self._pool = _process_pool(self._workers) if self._workers > 1 else None
        try:
            def objective(u: np.ndarray) -> float:
                return self.evaluate({p.key: p.from_unit(x) for p, x in zip(self.params, u)})

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
            if self._pool:
                self._pool.shutdown()
                self._pool = None
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
            "mesh_size": mesh.get("size") or "scenario default",
            "element": "linear tetrahedra",
            "timestep": raw.get("simulation", {}).get("timestep"),
            "contact_timeconst": raw.get("contact", {}).get("timeconst"),
            "cost": best["cost"],
            "per_case_nrmse": {c: best.get(f"{c}.nrmse") for c in [*self.cases, *self._term_names]},
            "note": "Effective parameters: valid only with this backend, element type and mesh size.",
        }
        fitted = material.model_copy(update={
            "kind": "effective",
            "name": f"{material.name}_fit_{self.dataset.name}",
            "source": f"Fitted to {self.dataset.title} (doi:{self.dataset.doi}) by softsim calibrate.",
            "provenance": provenance,
        })
        save_material(fitted, self.output / f"{fitted.name}.yaml")
        report = {"best": {k: float(v) for k, v in values.items()}, "cost": best["cost"],
                  "evaluations": len(self.history), "wall_time_s": wall,
                  "per_case": {c: {k[len(c) + 1:]: best[k] for k in best if k.startswith(c + ".")}
                               for c in [*self.cases, *self._term_names]},
                  "material_file": str(self.output / f"{fitted.name}.yaml")}
        (self.output / "report.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(report, default=float)),
                                                                sort_keys=False))
        return report


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def best_overrides(report: dict[str, Any]) -> list[str]:
    """Dotted overrides that apply a calibration report's best values."""
    return [f"{k}={v}" for k, v in report["best"].items()]

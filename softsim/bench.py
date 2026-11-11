"""Benchmarks: every validated case on every backend, against its truth, in one table.

A benchmark entry (``benchmarks/*.yaml``) names a scenario config, the quantity to read from a run, the truth
it is checked against and, per backend, the overrides and the tolerance that backend is held to::

    name: cable_cantilever
    description: Cable-bent cantilever, tip angle at 0.7 N.
    config: ../scenarios/actuation/cable_cantilever.yaml
    tier: quick                       # quick: minutes on a laptop; full: long runs (GPU box, overnight)
    quantity: tip angle [rad]
    metric: {summary: final_bend_angle}
    truth:
      analytic: {function: cable_beam_tip_angle, args: {tension: 0.7, ...}}   # softsim.analytic
      source: Euler-Bernoulli with the Neo-Hookean tangent
    runs:
      - {backend: fem, overrides: [simulation.timestep=2ms], tolerance: 0.02}
      - {backend: mujoco, tolerance: 0.35, note: linear-tet locking at 2 mm}

Metrics:
    * ``{summary: key}``: a scalar of the run's summary; the error is relative, ``sim / truth - 1``, or with
      ``absolute: true`` the difference ``sim - truth`` in the quantity's units (for a truth of zero).
    * ``{curve: {x: column, y: column, x_scale: s, y_scale: s}}``: a timeseries column read at the truth's
      x values along the rising part of ``x`` (a quasi-static ramp); the error is the RMS difference in
      the truth's units (``y_scale`` converts, e.g. rad to deg).

Truths: ``value`` (a number), ``analytic`` (a function of :mod:`softsim.analytic`; a tuple result is indexed
by ``index``) or ``dataset`` (``file``, ``key``, ``x`` and ``y`` columns of a table in a dataset YAML).

A tolerance is the error bar softsim states for that backend on that case (e.g. MuJoCo's linear tetrahedra
lock by 20-30% on bending at 2 mm, docs/actuation.md), so "pass" means "within the documented error bar",
not "exact". Results are cached by a hash of the entry run and the resolved config, under ``runs/bench/``,
so the report can be rebuilt without re-running. A backend that is not installed is reported as not run.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from softsim.errors import ConfigError

TIERS = ("quick", "full")


@dataclass
class BenchRun:
    backend: str
    tolerance: float
    overrides: list[str] = field(default_factory=list)
    note: str = ""
    known_failure: bool = False  # a documented gap: shown as a failure, not counted as a regression


@dataclass
class BenchEntry:
    name: str
    description: str
    config: Path
    tier: str
    quantity: str
    metric: dict[str, Any]
    truth: dict[str, Any]
    runs: list[BenchRun]
    source_file: Path | None = None

    @property
    def kind(self) -> str:
        return "curve" if "curve" in self.metric else "scalar"


def load_entry(path: str | Path) -> BenchEntry:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    try:
        tier = raw.get("tier", "quick")
        if tier not in TIERS:
            raise ConfigError(f"{path}: tier must be one of {TIERS}, not {tier!r}")
        metric = raw["metric"]
        if not (("summary" in metric) ^ ("curve" in metric)):
            raise ConfigError(f"{path}: metric needs exactly one of 'summary' or 'curve'")
        truth = raw["truth"]
        if sum(k in truth for k in ("value", "analytic", "dataset")) != 1:
            raise ConfigError(f"{path}: truth needs exactly one of 'value', 'analytic' or 'dataset'")
        runs = [BenchRun(backend=r["backend"], tolerance=float(r["tolerance"]),
                         overrides=list(r.get("overrides", [])), note=r.get("note", ""),
                         known_failure=bool(r.get("known_failure", False))) for r in raw["runs"]]
        return BenchEntry(name=raw["name"], description=raw.get("description", ""),
                          config=(path.parent / raw["config"]).resolve(), tier=tier,
                          quantity=raw.get("quantity", ""), metric=metric, truth=truth, runs=runs, source_file=path)
    except KeyError as exc:
        raise ConfigError(f"{path}: missing {exc.args[0]!r}") from None


def load_entries(directory: str | Path) -> list[BenchEntry]:
    files = sorted(Path(directory).glob("*.yaml"))
    if not files:
        raise ConfigError(f"no benchmark entries (*.yaml) in {directory}")
    return [load_entry(f) for f in files]


def truth_values(entry: BenchEntry) -> tuple[np.ndarray | None, np.ndarray]:
    """(x, y) of the truth: x is None for a scalar."""
    t = entry.truth
    if "value" in t:
        return None, np.atleast_1d(float(t["value"]))
    if "analytic" in t:
        from softsim import analytic

        spec = t["analytic"]
        module, _, name = spec["function"].rpartition(":")
        if module:
            import importlib

            analytic = importlib.import_module(module)
        fn = getattr(analytic, name, None)
        if fn is None:
            raise ConfigError(f"{entry.name}: softsim.analytic has no {spec['function']!r}")
        out = fn(**spec.get("args", {}))
        if isinstance(out, tuple):
            out = out[int(spec.get("index", 0))]
        return None, np.atleast_1d(float(out))
    spec = t["dataset"]
    base = entry.source_file.parent if entry.source_file else Path.cwd()
    table = np.asarray(yaml.safe_load((base / spec["file"]).read_text())[spec["key"]], float)
    return table[:, int(spec.get("x", 0))], table[:, int(spec.get("y", 1))]


def measure(entry: BenchEntry, summary: dict[str, float], timeseries) -> np.ndarray:
    """The run's value(s) of the entry's quantity, in the truth's units."""
    if entry.kind == "scalar":
        key = entry.metric["summary"]
        if key not in summary:
            raise ConfigError(f"{entry.name}: summary has no {key!r}; has {sorted(summary)}")
        return np.atleast_1d(float(summary[key]) * float(entry.metric.get("scale", 1.0)))
    c = entry.metric["curve"]
    x = timeseries[c["x"]].to_numpy() * float(c.get("x_scale", 1.0))
    y = timeseries[c["y"]].to_numpy() * float(c.get("y_scale", 1.0))
    keep = np.concatenate([[True], np.diff(x) > 1e-12])  # the rising part, once
    tx, _ = truth_values(entry)
    return np.interp(tx, x[keep], y[keep])


def error(entry: BenchEntry, value: np.ndarray) -> float:
    """Relative error (scalar) or RMS difference (curve) against the truth."""
    _, truth = truth_values(entry)
    if entry.kind == "scalar":
        if entry.metric.get("absolute"):
            return float(value[0] - truth[0])
        return float(value[0] / truth[0] - 1.0)
    return float(np.sqrt(np.mean((value - truth) ** 2)))


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "") if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _key(entry: BenchEntry, run: BenchRun) -> str:
    from softsim.config.loader import load_yaml

    resolved = load_yaml(entry.config)  # resolves `extends`, so a change in a parent config invalidates too
    blob = json.dumps({"config": resolved, "backend": run.backend, "overrides": run.overrides,
                       "metric": entry.metric}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _unavailable(exc: Exception) -> bool:
    text = str(exc)
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or any(
        s in text for s in ("not installed", "SOFA_ROOT", "No module named", "needs the"))


def _judge(rec: dict[str, Any], run: BenchRun) -> dict[str, Any]:
    """The verdict against the run's current tolerance (a cached result is not re-run when only that changes)."""
    if "error" in rec and rec.get("status") in ("pass", "fail"):
        rec = {**rec, "tolerance": run.tolerance, "known_failure": run.known_failure, "note": run.note,
               "status": "pass" if abs(rec["error"]) <= run.tolerance else "fail"}
    return rec


def run_one(entry: BenchEntry, run: BenchRun, cache_dir: Path, force: bool = False) -> dict[str, Any]:
    """Run (or read from the cache) one backend of one entry; returns its result record."""
    path = cache_dir / entry.name / f"{run.backend}.json"
    key = _key(entry, run)
    if not force and path.exists():
        cached = json.loads(path.read_text())
        if cached.get("key") == key and cached.get("status") in ("pass", "fail"):
            return _judge(cached, run)
    record: dict[str, Any] = {"name": entry.name, "backend": run.backend, "key": key, "tolerance": run.tolerance,
                              "note": run.note, "known_failure": run.known_failure}
    start = time.perf_counter()
    try:
        from softsim.runner import Simulation

        sim = Simulation(entry.config, ["output.save_model=false", *run.overrides], backend=run.backend)
        try:
            result = sim.run()
            body = sim.scene.soft_bodies[0] if sim.scene.soft_bodies else None
            settings = getattr(getattr(sim.scenario.config, "simulation", None), "timestep", None)
            mesh_size = None
            if body is not None:
                cfg = sim.raw_config
                for top in ("body", "specimen"):
                    if isinstance(cfg.get(top), dict) and isinstance(cfg[top].get("mesh"), dict):
                        mesh_size = cfg[top]["mesh"].get("size")
            value = measure(entry, result.summary, result.timeseries)
        finally:
            sim.close()
    except Exception as exc:  # noqa: BLE001 - a benchmark reports every failure, it never aborts the table
        record.update(status="not run" if _unavailable(exc) else "error", reason=f"{type(exc).__name__}: {exc}"[:300],
                      wall_time_s=time.perf_counter() - start)
        if record["status"] == "error":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1))
        return record
    err = error(entry, value)
    record.update(status="pass" if abs(err) <= run.tolerance else "fail", value=value.tolist(), error=err,
                  wall_time_s=result.metadata.get("wall_time_s", time.perf_counter() - start),
                  mesh_size=None if mesh_size is None else str(mesh_size),
                  timestep=None if settings is None else float(settings),
                  n_tets=int(body.mesh.n_tets) if body is not None else None,
                  date=_dt.date.today().isoformat(), commit=_git_commit(entry.config.parent))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return record


def cached(entry: BenchEntry, run: BenchRun, cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / entry.name / f"{run.backend}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("key") != _key(entry, run):
        return {**rec, "status": "stale"}
    return _judge(rec, run)


def _fmt(x: float) -> str:
    if x == 0 or 1e-3 <= abs(x) < 1e4:
        return f"{x:.4g}"
    return f"{x:.3e}"


def _cell_value(entry: BenchEntry, rec: dict[str, Any]) -> tuple[str, str]:
    if entry.kind == "scalar":
        if entry.metric.get("absolute"):
            return _fmt(rec["value"][0]), f"{rec['error']:+.3g}"
        return _fmt(rec["value"][0]), f"{100 * rec['error']:+.1f}%"
    return "curve", f"{rec['error']:.2f} RMS"


def _tol(entry: BenchEntry, run: BenchRun) -> str:
    if entry.kind == "scalar":
        return f"±{run.tolerance:.3g}" if entry.metric.get("absolute") else f"±{100 * run.tolerance:.2g}%"
    return f"≤{run.tolerance:g} RMS"


def report(entries: list[BenchEntry], cache_dir: Path) -> str:
    """Markdown page of every entry x backend from the cache."""
    status_mark = {"pass": "✅ pass", "fail": "❌ fail", "error": "⚠️ error", "not run": "– not run",
                   "stale": "… stale"}
    lines = [
        "# Benchmarks",
        "",
        "Every validated case on every backend against its truth (an analytic solution or a measurement),",
        "generated by `softsim bench`. The tolerance is the error bar softsim states for that backend on that",
        "case, so *pass* means *within the documented error bar*, not *exact*: MuJoCo and SOFA use linear",
        "tetrahedra, which lock in bending ([actuation.md](actuation.md)); the `fem` reference is locking-free.",
        "",
        "Regenerate: `softsim bench --tier quick` (minutes) or `--tier full` (hours; GPU machine), then",
        "`softsim bench --report-only`. A row marked *stale* has a cached result from an older config.",
        "",
    ]
    for tier in TIERS:
        tier_entries = [e for e in entries if e.tier == tier]
        if not tier_entries:
            continue
        lines += [f"## {'Quick' if tier == 'quick' else 'Full'} tier", "",
                  "| Case | Quantity | Truth | Backend | Value | Error | Tolerance | Status | Mesh | Wall [s] | Run |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in tier_entries:
            tx, ty = truth_values(e)
            truth = _fmt(ty[0]) if tx is None else f"{len(ty)} points"
            source = e.truth.get("source", "")
            for run in e.runs:
                rec = cached(e, run, cache_dir)
                if rec is None or rec.get("status") in ("not run",) or "value" not in rec:
                    st = (rec or {}).get("status", "not run")
                    lines.append(f"| {e.name} | {e.quantity} | {truth} | {run.backend} | | | {_tol(e, run)} | "
                                 f"{status_mark.get(st, st)} | | | |")
                    continue
                value, err = _cell_value(e, rec)
                mesh = rec.get("mesh_size") or ""
                if rec.get("n_tets"):
                    mesh = f"{mesh} ({rec['n_tets']} tets)".strip()
                run_on = f"{rec.get('date', '')} {rec.get('commit', '')}".strip()
                status = status_mark.get(rec["status"], rec["status"])
                if rec["status"] == "fail" and run.known_failure:
                    status += " (known)"
                lines.append(f"| {e.name} | {e.quantity} | {truth} | {run.backend} | {value} | {err} | "
                             f"{_tol(e, run)} | {status} | {mesh} | "
                             f"{rec.get('wall_time_s', 0):.0f} | {run_on} |")
            if source:
                lines.append(f"| | *truth: {source}* | | | | | | | | | |")
        lines.append("")
    lines += ["## Cases", ""]
    for e in entries:
        notes = "; ".join(f"{r.backend}: {r.note}" for r in e.runs if r.note)
        lines.append(f"- **{e.name}**: {e.description}" + (f" Notes: {notes}." if notes else ""))
    lines.append("")
    return "\n".join(lines)


def run_bench(entries: list[BenchEntry], cache_dir: Path, tier: str = "quick", only: list[str] | None = None,
              backends: list[str] | None = None, force: bool = False, echo=print) -> list[dict[str, Any]]:
    """Run the selected entries (``tier`` 'quick' runs quick ones, 'full' runs all)."""
    out = []
    for e in entries:
        if only and e.name not in only:
            continue
        if not only and tier == "quick" and e.tier != "quick":
            continue
        for run in e.runs:
            if backends and run.backend not in backends:
                continue
            rec = run_one(e, run, cache_dir, force=force)
            shown = ""
            if "error" in rec:
                if e.kind == "curve" or e.metric.get("absolute"):
                    shown = f"{rec['error']:+.3g}"
                else:
                    shown = f"{100 * rec['error']:+.2f}%"
            echo(f"{e.name:28s} {run.backend:7s} {rec['status']:8s} {shown:>12s}  "
                 f"{rec.get('wall_time_s', 0):6.0f} s  {rec.get('reason', '')}")
            out.append(rec)
    return out

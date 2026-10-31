"""Parameter sweeps.

A sweep file::

    base: ../scenarios/drop_test/cases/w020.yaml   # relative to the sweep file
    overrides: ["simulation.duration=0.03"]        # applied to every run
    grid:                                          # full Cartesian product
      specimen.material.elastic.youngs_modulus: [30000, 50000, 70000]
      specimen.mesh.size: [0.005, 0.0035]
    zip:                                           # varied together (equal lengths)
      simulation.timestep: [0.0002, 0.0001]
      contact.timeconst: [0.0004, 0.0002]
    sample:                                        # material variants drawn from its card's `uncertainty`
      material: specimen.material                  #   (crossed with the grid; or `softsim sweep --sample N`)
      n: 32
      seed: 0
    save_timeseries: true

Each run's summary becomes one row of ``results.parquet`` (parameters as
columns, sampled values as ``sample.<card key>`` columns, failed runs keep
their error message).
"""

from __future__ import annotations

import copy
import itertools
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from softsim.config.loader import load_yaml, set_dotted
from softsim.errors import ConfigError
from softsim.io import save_table


def expand(grid: Mapping[str, list] | None = None, zipped: Mapping[str, list] | None = None) -> list[dict[str, Any]]:
    """All parameter combinations: product over ``grid`` x the zipped block."""
    grid = dict(grid or {})
    zipped = dict(zipped or {})
    lengths = {len(v) for v in zipped.values()}
    if len(lengths) > 1:
        raise ConfigError(f"zip lists must have equal lengths, got {sorted(lengths)}")
    zip_rows = [dict(zip(zipped, values)) for values in zip(*zipped.values())] if zipped else [{}]
    grid_rows = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] if grid else [{}]
    return [{**g, **z} for g in grid_rows for z in zip_rows]


def _run_one(args: tuple) -> dict[str, Any]:
    from softsim.runner import Simulation

    raw, base_dir, params, index, out_dir, fmt, hidden, record = args
    raw = copy.deepcopy(raw)
    row: dict[str, Any] = {"run": index, **params, **record}
    try:
        for key, value in {**params, **hidden}.items():
            raw = set_dotted(raw, key, value)
        sim = Simulation(raw, base_dir=base_dir)
        try:
            result = sim.run()
        finally:
            sim.close()
        row.update(result.summary)
        row["wall_time_s"] = result.metadata["wall_time_s"]
        row["warnings"] = sum(f.level == "warning" for f in result.findings)
        if out_dir is not None:
            result.save(Path(out_dir) / f"run_{index:04d}", fmt)
        row["error"] = ""
    except Exception as exc:  # keep the sweep going; record why this point failed
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def material_samples(raw: Mapping[str, Any], base_dir: Path, key: str, n: int,
                     seed: int | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(columns, hidden overrides) for ``n`` draws of the material at config ``key`` from its uncertainty."""
    from softsim.config.loader import get_dotted
    from softsim.materials import load_material, sample

    extra = [base_dir / p for p in raw.get("material_paths", [])]
    try:
        ref = get_dotted(raw, key)
    except KeyError:
        raise ConfigError(f"sample: no material at {key!r} in the base config") from None
    card = load_material(ref, extra=extra, base_dir=base_dir)
    out = []
    for m in sample(card, n, seed):
        draws = {f"sample.{k}": v for k, v in m.provenance["sample"].items() if k not in ("index", "seed")}
        out.append((draws, {key: m.model_dump(mode="json", exclude_none=True)}))
    return out


def run_sweep(sweep_file: str | Path, out_dir: str | Path, jobs: int = 1,
              extra_overrides: Iterable[str] = (), sample_n: int | None = None,
              sample_seed: int | None = None) -> pd.DataFrame:
    """Run a sweep file; writes ``results.parquet`` (+ per-run folders) to ``out_dir``.

    ``sample_n`` (or a ``sample:`` block) crosses the grid with material variants drawn
    from the card's ``uncertainty``.
    """
    from softsim.config.loader import apply_overrides

    sweep_file = Path(sweep_file).resolve()
    spec = load_yaml(sweep_file)
    unknown = set(spec) - {"base", "overrides", "grid", "zip", "sample", "save_timeseries", "format"}
    if unknown:
        raise ConfigError(f"unknown sweep keys: {sorted(unknown)}")
    if "base" not in spec:
        raise ConfigError("sweep file needs a 'base' scenario config")
    base_path = (sweep_file.parent / spec["base"]).resolve()
    raw = apply_overrides(load_yaml(base_path), [*spec.get("overrides", []), *extra_overrides])
    points = [(p, {}, {}) for p in expand(spec.get("grid"), spec.get("zip"))]  # (applied, hidden, recorded)
    sample_spec = dict(spec.get("sample") or {})
    if sample_n is not None:
        sample_spec["n"] = sample_n
    if sample_seed is not None:
        sample_spec["seed"] = sample_seed
    if sample_spec:
        draws = material_samples(raw, base_path.parent, sample_spec.get("material", "specimen.material"),
                                 int(sample_spec.get("n", 16)), sample_spec.get("seed"))
        points = [(p, hidden, cols) for p, _, _ in points for cols, hidden in draws]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.yaml").write_text(yaml.safe_dump({**spec, "base": str(base_path)}, sort_keys=False))
    fmt = spec.get("format", "parquet")
    runs_dir = str(out_dir / "runs") if spec.get("save_timeseries", False) else None
    tasks = [(raw, str(base_path.parent), p, i, runs_dir, fmt, hidden, rec)
             for i, (p, hidden, rec) in enumerate(points)]
    if jobs > 1:
        with mp.get_context("spawn").Pool(jobs) as pool:
            rows = pool.map(_run_one, tasks)
    else:
        rows = [_run_one(t) for t in tasks]
    df = pd.DataFrame(rows)
    save_table(df, out_dir / "results", fmt)
    return df

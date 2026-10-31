"""Which parameters matter? Morris elementary-effects screening for any scenario and metric.

    softsim sensitivity scenarios/drop_test/cases/w050.yaml \\
        --param "specimen.material.elastic.youngs_modulus=30 kPa:150 kPa" \\
        --param "specimen.material.damping.beta=0.1 ms:3 ms" --param "drop.height=0.2:0.4" \\
        --metric peak_indentation --metric max_contact_force --trajectories 10 --jobs 8

Each of ``r`` trajectories starts at a random point of a ``levels``-level grid
over the ranges and moves one parameter at a time by Delta = levels / (2 (levels - 1))
of its range: r (k + 1) runs for k parameters. Per parameter and metric it
reports mu* (mean |elementary effect|, the overall influence) and sigma (spread:
non-linearity or interaction), both as a fraction of the metric's mean
magnitude per full parameter range. Ranges spanning more than a decade are
sampled in log space.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError


@dataclass(frozen=True)
class Range:
    key: str
    low: float
    high: float

    @property
    def log(self) -> bool:
        return self.low > 0 and self.high / self.low > 10.0

    def value(self, u: float) -> float:
        if self.log:
            return float(math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low))))
        return float(self.low + u * (self.high - self.low))


def parse_range(text: str, config: str | Path, overrides: Iterable[str] = ()) -> Range:
    """``"key=low:high"`` with units allowed (taken from the key's field, see ``softsim params``)."""
    from softsim.config.params import config_params

    if "=" not in text or ":" not in text.split("=", 1)[1]:
        raise ConfigError(f"--param must look like key=low:high, got {text!r}")
    key, bounds = text.split("=", 1)
    params = {p.key: p for p in config_params(str(config), list(overrides))}
    if key not in params:
        raise ConfigError(f"unknown parameter {key!r}; see `softsim params {config}`")
    dim = params[key].unit.dimension if params[key].unit else None
    lo, hi = (parse_quantity(b.strip(), dim) if dim else float(b) for b in bounds.split(":", 1))
    if not lo < hi:
        raise ConfigError(f"{key}: low must be below high")
    return Range(key, lo, hi)


def morris_design(k: int, trajectories: int, levels: int = 4, seed: int | None = 0) -> np.ndarray:
    """(trajectories, k + 1, k) unit-cube points; consecutive points differ in one coordinate by +-Delta."""
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    out = np.zeros((trajectories, k + 1, k))
    for t in range(trajectories):
        x = rng.choice(grid[grid <= 1.0 - delta + 1e-12], size=k)
        out[t, 0] = x
        for step, i in enumerate(rng.permutation(k), 1):
            x = x.copy()
            x[i] = x[i] + delta if x[i] + delta <= 1.0 + 1e-12 else x[i] - delta
            out[t, step] = x
    return out


def morris_indices(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """mu* and sigma per parameter from outputs ``y`` (trajectories, k + 1); NaN runs are skipped."""
    k = design.shape[2]
    effects = [[] for _ in range(k)]
    for pts, ys in zip(design, y):
        for j in range(k):
            diff = pts[j + 1] - pts[j]
            i = int(np.argmax(np.abs(diff)))
            ee = (ys[j + 1] - ys[j]) / diff[i]
            if np.isfinite(ee):
                effects[i].append(ee)
    mu_star = np.array([np.mean(np.abs(e)) if e else np.nan for e in effects])
    sigma = np.array([np.std(e) if len(e) > 1 else np.nan for e in effects])
    return mu_star, sigma


def run_sensitivity(config: str | Path, ranges: list[Range], metrics: list[str], trajectories: int = 8,
                    levels: int = 4, seed: int | None = 0, jobs: int = 1, overrides: Iterable[str] = ()) -> pd.DataFrame:
    """Run the Morris design; one row per parameter and metric with ``mu_star`` and ``sigma`` (relative)."""
    from softsim.sweep import _run_one

    config = Path(config).resolve()
    raw = apply_overrides(load_yaml(config), list(overrides))
    design = morris_design(len(ranges), trajectories, levels, seed)
    points = design.reshape(-1, len(ranges))
    tasks = [(raw, str(config.parent), {r.key: r.value(u) for r, u in zip(ranges, pt)}, i, None, "parquet", {}, {})
             for i, pt in enumerate(points)]
    if jobs > 1:
        with mp.get_context("spawn").Pool(jobs) as pool:
            rows = pool.map(_run_one, tasks)
    else:
        rows = [_run_one(t) for t in tasks]
    runs = pd.DataFrame(rows)
    out = []
    for metric in metrics:
        if metric not in runs:
            raise ConfigError(f"metric {metric!r} not in the run summaries; have {sorted(c for c in runs if c not in ('run', 'error'))}")
        y = pd.to_numeric(runs[metric], errors="coerce").to_numpy().reshape(trajectories, len(ranges) + 1)
        scale = float(np.nanmean(np.abs(y))) or 1.0
        mu_star, sigma = morris_indices(design, y)
        for r, m, s in zip(ranges, mu_star, sigma):
            out.append({"metric": metric, "parameter": r.key, "low": r.low, "high": r.high, "log": r.log,
                        "mu_star": m / scale, "sigma": s / scale})
    df = pd.DataFrame(out).sort_values(["metric", "mu_star"], ascending=[True, False])
    df.attrs["runs"] = runs
    return df

"""Plots for experiment/simulation comparison (matplotlib, Agg backend)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from softsim.errors import StabilityError
from softsim.experiments.compare import ExperimentCase, compare_case, simulated_indentation


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def simulate_cases(dataset, tracks: str | Path, cases: list[str], overrides: list[str],
                   duration: float = 0.03) -> dict[str, pd.DataFrame]:
    """Run each case with its measured impact; returns simulated indentation curves.

    Cases whose simulation diverges are left out (with a warning on stderr), so
    one unstable comparison run doesn't lose the plots of the others.
    """
    from softsim.runner import run_config

    out = {}
    for case in cases:
        exp = ExperimentCase.load(tracks, case)
        ov = [*overrides, *exp.sim_overrides(), f"simulation.duration={duration}", "output.save_model=false"]
        try:
            ts = run_config(dataset.config_path(case), ov).timeseries
        except StabilityError as exc:
            print(f"warning: {case} not plotted: {exc}", file=sys.stderr)
            continue
        out[case] = simulated_indentation(ts)
    return out


def overlay(tracks: str | Path, sims: Mapping[str, Mapping[str, pd.DataFrame]], path: str | Path,
            fitted: list[str] | None = None, title: str = "") -> Path:
    """Indentation (relative to specimen base) vs time per case; ``sims`` maps label -> case -> curve."""
    plt = _plt()
    cases = sorted({c for curves in sims.values() for c in curves})
    fig, axes = plt.subplots(1, len(cases), figsize=(3.4 * len(cases), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    styles = ["-", "--", ":", "-."]
    for ax, case in zip(axes, cases):
        exp = ExperimentCase.load(tracks, case)
        t, y = exp.curve()
        sel = (t > -0.002) & (t < 0.025)
        ax.plot(t[sel] * 1e3, y[sel] * 1e3, "k.", ms=2.5, label="experiment")
        w0, w1 = exp.window()
        ax.axvspan(w0 * 1e3, w1 * 1e3, color="0.9", zorder=0)
        for i, ((label, curves), style) in enumerate(zip(sims.items(), styles)):
            if case in curves:
                c = curves[case]
                sel = (c.t > -0.002) & (c.t < 0.025)
                ax.plot(c.t[sel] * 1e3, c.indentation_rel[sel] * 1e3, style, color=f"C{i}", lw=1.5, label=label)
        tag = "" if fitted is None else (" (fitted)" if case in fitted else " (not fitted)")
        ax.set_title(f"{case}{tag}", fontsize=10)
        ax.set_xlabel("time since contact [ms]")
        ax.grid(alpha=0.3)
        ax.set_ylim(-3, 12)
    axes[0].set_ylabel("indentation rel. to base [mm]")
    axes[0].legend(fontsize=7, loc="upper right")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def peak_summary(tracks: str | Path, sims: Mapping[str, Mapping[str, pd.DataFrame]], masses: Mapping[str, float],
                 path: str | Path) -> tuple[Path, pd.DataFrame]:
    """Peak indentation and time to peak vs weight mass, experiment vs simulations."""
    plt = _plt()
    rows = []
    for case, mass in masses.items():
        exp = ExperimentCase.load(tracks, case)
        row = {"case": case, "mass_g": mass * 1e3}
        for label, curves in sims.items():
            if case in curves:
                m = compare_case(curves[case], exp)
                row.update({"peak_exp": m.get("peak_exp"), "t_peak_exp": m.get("t_peak_exp"),
                            f"peak_{label}": m.get("peak_sim"), f"t_peak_{label}": m.get("t_peak_sim"),
                            f"nrmse_{label}": m.get("nrmse")})
        rows.append(row)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, unit in [(axes[0], "peak", "peak indentation [mm]"), (axes[1], "t_peak", "time to peak [ms]")]:
        ax.plot(df.mass_g, df[f"{key}_exp"] * 1e3, "ko-", label="experiment")
        for i, label in enumerate(sims):
            if f"{key}_{label}" in df:
                ax.plot(df.mass_g, df[f"{key}_{label}"] * 1e3, "s--", color=f"C{i}", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("weight mass [g]")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path, df


def calibration_plots(report: Mapping[str, Any], calibration_file: str | Path, out_dir: str | Path,
                      all_cases: bool = True) -> list[Path]:
    """Overlay and summary plots for a finished calibration (before vs after)."""
    from softsim.experiments.calibrate import Calibration, best_overrides

    cal = Calibration(calibration_file)
    cases = list(cal.dataset.cases) if all_cases else list(cal.cases)
    base = simulate_cases(cal.dataset, cal.tracks, cases, cal.overrides)
    fit = simulate_cases(cal.dataset, cal.tracks, cases, [*cal.overrides, *best_overrides(report)])
    best = ", ".join(f"{k.split('.')[-1]}={v:.3g}" for k, v in report["best"].items())
    sims = {"preset (uncalibrated)": base, f"fitted ({best})": fit}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = overlay(cal.tracks, sims, out_dir / "overlay.png", fitted=list(cal.cases),
                 title=f"{cal.dataset.title}: simulation vs experiment (grey = compared window)")
    masses = {c: cal.dataset.case(c).weight_mass for c in cases}
    p2, df = peak_summary(cal.tracks, sims, masses, out_dir / "peaks.png")
    df.to_csv(out_dir / "peaks.csv", index=False)
    return [p1, p2]

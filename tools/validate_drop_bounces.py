"""Validate every weight's bounce against the recordings: docs/media/bounce_validation.png.

    python tools/validate_drop_bounces.py            # ~10 min on 8 cores
    python tools/validate_drop_bounces.py --replot   # only plot runs/bounce_validation/*.parquet
    python tools/validate_drop_bounces.py --force    # re-simulate everything (saved runs are reused otherwise)

Runs all five drop weights with each recording's measured impact (speed, offset,
spin, in-plane tilt) and compares, over the whole first 200 ms, the first contact,
the weight's rebound and the specimen's hop. Models: on MuJoCo the first-contact
fit on a rigid table (M3) and the same material plus fitted tack and compliant
support; the MuJoCo and SOFA transfers of the physical card (rigid table, no tack;
SOFA needs SOFA_ROOT); and the 3D reference with the physical card, whose runs
cover only the first contact of 20 and 50 g (runs/reference_tracks). Writes the
figure, runs/bounce_validation/metrics.csv and a markdown table on stdout.

Which numbers mean something per case (see docs/experiments.md):
    w020  everything (the weight stays seated and hops with the specimen);
    w050  weight only until ~15 ms (then it flips and its centroid is not its COM),
          specimen hop throughout;
    w100, w200  edge impacts: the weight rolls off the rim; compared qualitatively;
    w500  first contact unexplained by any model; shown for completeness.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from softsim.experiments.calibrate import Calibration, best_overrides
from softsim.experiments.compare import ExperimentCase, compare_observable, simulated_indentation

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "calibrations" / "ecoflex_00_10_drop_adhesion_support.yaml"
OUT = ROOT / "runs" / "bounce_validation"
CASES = ["w020", "w050", "w100", "w200", "w500"]
DURATION = 0.2
RIGID = ["specimen.material=ecoflex_00_10_drop_fit", "specimen.mesh.size=0.005", "simulation.timestep=0.00005"]
# where the tracked weight centroid stands for the weight's centre of mass [s]
WEIGHT_VALID = {"w050": 0.015}
NOTES = {"w050": "weight flips after ~15 ms", "w100": "edge impact, rolls off", "w200": "edge impact",
         "w500": "first contact unexplained"}
LIFT_ZERO = 5e-4  # "landed" once the specimen is back within 0.5 mm


TRANSFERS = {"mujoco transfer": ROOT / "calibrations" / "transfer_ecoflex_00_10_mujoco.yaml",
             "sofa transfer": ROOT / "calibrations" / "transfer_ecoflex_00_10_sofa.yaml"}
REFERENCE_TRACKS = ROOT / "runs" / "reference_tracks" / "ecoflex_00_10_physical"  # 3D reference, first contact


def _models() -> dict[str, list[str]]:
    cal = Calibration(FIT)
    report = yaml.safe_load((cal.output / "report.yaml").read_text())
    models = {"rigid": RIGID, "tack+support": [*cal.overrides, *best_overrides(report)]}
    for name, path in TRANSFERS.items():
        t = Calibration(path)
        if (t.output / "report.yaml").exists():
            models[name] = [*t.overrides, *best_overrides(yaml.safe_load((t.output / "report.yaml").read_text()))]
    return models


LABELS = {"rigid": "MuJoCo: first-contact fit, rigid table, no tack",
          "tack+support": "MuJoCo: + tack + compliant support (fitted)",
          "mujoco transfer": "MuJoCo: transfer of the physical card",
          "sofa transfer": "SOFA: transfer of the physical card",
          "fem physical": "3D reference: physical card (first contact only)"}


def _run(task: tuple[str, str, list[str]]) -> tuple[str, str, pd.DataFrame | None, str]:
    from softsim.runner import run_config

    model, case, overrides = task
    cal = Calibration(FIT)
    exp = ExperimentCase.load(cal.tracks, case)
    ov = [*overrides, *exp.sim_overrides(), f"simulation.duration={DURATION}", "simulation.output_rate=5000",
          "output.save_model=false"]
    try:
        ts = run_config(cal.dataset.config_path(case), ov).timeseries
    except Exception as exc:  # noqa: BLE001 - report and keep the other runs
        return model, case, None, f"{type(exc).__name__}: {exc}"
    sim = simulated_indentation(ts)
    sim["weight_rise"] = sim.specimen_lift - sim.indentation_rel
    return model, case, sim, ""


def _first_contact(d: pd.DataFrame) -> dict[str, float]:
    d = d[(d.t >= 0) & (d.t <= 0.02)].dropna(subset=["indentation_rel"])
    k = d.indentation_rel.idxmax()
    back = d[(d.t > d.t[k]) & (d.indentation_rel <= 0)]
    return {"peak": d.indentation_rel[k], "t_peak": d.t[k], "t_sep": back.t.iloc[0] if len(back) else np.nan}


def _hop(d: pd.DataFrame, t_max: float) -> dict[str, float]:
    d = d[(d.t >= 0) & (d.t <= t_max)].dropna(subset=["specimen_lift"])
    k = d.specimen_lift.idxmax()
    down = d[(d.t > d.t[k]) & (d.specimen_lift <= LIFT_ZERO)]
    return {"hop": d.specimen_lift[k], "t_hop": d.t[k], "t_land": down.t.iloc[0] if len(down) else np.nan}


def _weight(d: pd.DataFrame, t_max: float) -> dict[str, float]:
    """Weight after the first contact: lowest position relative to the base (negative = above
    its height at contact relative to the specimen) and highest rise above its contact height."""
    d = d[(d.t >= 0.002) & (d.t <= t_max)].dropna(subset=["indentation_rel", "specimen_lift"])
    rise = d.specimen_lift - d.indentation_rel
    return {"rel_min": d.indentation_rel.min(), "rise": rise.max(), "t_rise": d.t[rise.idxmax()]}


def metrics(case: str, sim: pd.DataFrame | None, exp: ExperimentCase) -> dict[str, float]:
    d = exp.data.copy()
    t_w = min(WEIGHT_VALID.get(case, 1.0), float(d.dropna(subset=["indentation_rel"]).t.max()), 0.1)
    t_l = min(float(d.dropna(subset=["specimen_lift"]).t.max()), DURATION)
    row = {"case": case, "t_weight_valid": t_w, "t_lift_valid": t_l}
    for side, src in [("exp", d), ("sim", sim)]:
        if src is None:
            continue
        for k, v in {**_first_contact(src), **_hop(src, t_l), **_weight(src, t_w)}.items():
            row[f"{k}_{side}"] = float(v)
    if sim is not None:
        row["nrmse_first"] = compare_observable(sim, exp, "indentation_rel")["nrmse"]
        row["nrmse_weight"] = compare_observable(sim, exp, "indentation_rel", (0.0, t_w))["nrmse"]
        row["nrmse_lift"] = compare_observable(sim, exp, "specimen_lift", (0.0, min(t_l, 0.1)))["nrmse"]
    return row


def figure(tracks: str, sims: dict[str, dict[str, pd.DataFrame]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("indentation_rel", "weight rel. to specimen base [mm]\n(+ = into the specimen)", 0.1),
            ("weight_rise", "weight height above contact [mm]", 0.1),
            ("specimen_lift", "specimen hop [mm]", DURATION)]
    fig, axes = plt.subplots(len(rows), len(CASES), figsize=(3.3 * len(CASES), 2.6 * len(rows)), squeeze=False)
    styles = {"rigid": ":", "tack+support": "-", "mujoco transfer": "--", "sofa transfer": "-.", "fem physical": "-"}
    for j, case in enumerate(CASES):
        exp = ExperimentCase.load(tracks, case)
        d = exp.data.copy()
        d["weight_rise"] = d.specimen_lift - d.indentation_rel
        for i, (col, label, t_max) in enumerate(rows):
            ax = axes[i, j]
            e = d.dropna(subset=[col])
            e = e[(e.t > -0.003) & (e.t < t_max)]
            ax.plot(e.t * 1e3, e[col] * 1e3, "k.", ms=1.8, label="experiment")
            t_valid = WEIGHT_VALID.get(case)
            if col != "specimen_lift" and t_valid:
                ax.axvspan(t_valid * 1e3, t_max * 1e3, color="0.92", zorder=0)
            for k, (model, curves) in enumerate(sims.items()):
                if case in curves:
                    c = curves[case]
                    c = c[(c.t > -0.003) & (c.t < t_max)]
                    ax.plot(c.t * 1e3, c[col] * 1e3, styles.get(model, "--"), color=f"C{k}", lw=1.3,
                            label=LABELS.get(model, model))
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(case + (f"\n{NOTES[case]}" if case in NOTES else ""), fontsize=9)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            if i == len(rows) - 1 or col == "weight_rise":
                ax.set_xlabel("time since contact [ms]", fontsize=8)
    lim = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(max(lim[0], -20), lim[1])
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Drop tests, every weight: first contact, weight rebound and specimen hop vs experiment "
                 "(grey: weight centroid not its centre of mass)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _mm(v: float) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v * 1e3:.1f}"


def table(df: pd.DataFrame) -> str:
    head = ("| Case | Model | First peak mm (t ms) | Separation ms | Weight lowest rel. base mm | "
            "Weight max rise mm | Specimen hop mm (t ms) | Landed ms | NRMSE first / weight / hop |")
    lines = [head, "| " + " | ".join(["---"] * 9) + " |"]
    for _, r in df.iterrows():
        def pair(k, t):
            return f"{_mm(r.get(k))} ({_mm(r.get(t))})"
        nr = " / ".join("–" if not np.isfinite(r.get(k, np.nan)) else f"{100 * r[k]:.0f}%"
                        for k in ("nrmse_first", "nrmse_weight", "nrmse_lift"))
        if r.model == "rigid":
            lines.append(f"| {r.case} | experiment | {pair('peak_exp', 't_peak_exp')} | {_mm(r.get('t_sep_exp'))} | "
                         f"{_mm(r.get('rel_min_exp'))} | {_mm(r.get('rise_exp'))} | {pair('hop_exp', 't_hop_exp')} | "
                         f"{_mm(r.get('t_land_exp'))} | |")
        lines.append(f"| | {r.model} | {pair('peak_sim', 't_peak_sim')} | {_mm(r.get('t_sep_sim'))} | "
                     f"{_mm(r.get('rel_min_sim'))} | {_mm(r.get('rise_sim'))} | {pair('hop_sim', 't_hop_sim')} | "
                     f"{_mm(r.get('t_land_sim'))} | {nr} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replot", action="store_true", help="only plot the saved simulations")
    ap.add_argument("--force", action="store_true", help="re-simulate runs that are already saved")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cal = Calibration(FIT)
    models = _models()
    sims: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in models}
    for m in models:  # saved runs are reused; --force re-simulates
        for case in CASES:
            p = OUT / f"{m}_{case}.parquet"
            if p.exists() and not args.force:
                sims[m][case] = pd.read_parquet(p)
    if not args.replot:
        tasks = [(m, case, ov) for m, ov in models.items() for case in CASES if case not in sims[m]]
        with ProcessPoolExecutor(max(1, min(args.jobs, len(tasks)))) as pool:
            for model, case, sim, err in pool.map(_run, tasks):
                if sim is None:
                    print(f"warning: {model} {case} failed: {err}", file=sys.stderr)
                    continue
                sim.to_parquet(OUT / f"{model}_{case}.parquet")
                sims[model][case] = sim
    for case in CASES:  # the 3D reference runs exist for the first contact of 20 and 50 g
        if (REFERENCE_TRACKS / f"{case}.parquet").exists():
            ref = ExperimentCase.load(REFERENCE_TRACKS, case).data.copy()
            ref["weight_rise"] = ref.specimen_lift - ref.indentation_rel
            sims.setdefault("fem physical", {})[case] = ref
    models = {**models, **({"fem physical": []} if "fem physical" in sims else {})}
    rows = []
    for m in models:
        for case in CASES:
            exp = ExperimentCase.load(cal.tracks, case)
            rows.append({"model": m, **metrics(case, sims[m].get(case), exp)})
    df = pd.DataFrame(rows).sort_values(["case", "model"], key=lambda s: s.map(
        {m: i for i, m in enumerate(models)}).fillna(s) if s.name == "model" else s)
    df.to_csv(OUT / "metrics.csv", index=False)
    print(figure(cal.tracks, sims, ROOT / "docs" / "media" / "bounce_validation.png"))
    print(table(df))


if __name__ == "__main__":
    main()

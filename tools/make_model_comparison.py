"""Compare finished calibrations on the same cases: docs/media/model_comparison_*.png.

    SOFA_ROOT=... python tools/make_model_comparison.py

Re-simulates each calibration's best fit (its own overrides: backend, mesh,
timestep) for the fitted cases, then draws one overlay and one peak summary.
Needs the reports in runs/calibration/ (``softsim calibrate ...``) and the
tracking results. About 15 min, dominated by the SOFA 500 g runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from softsim.experiments.calibrate import Calibration, best_overrides
from softsim.experiments.plots import overlay, peak_summary, simulate_cases

ROOT = Path(__file__).resolve().parents[1]
CALIBRATIONS = {  # label -> calibration file (report in its output directory)
    "MuJoCo, stable NH": "calibrations/ecoflex_00_10_drop.yaml",
    "SOFA, stable NH": "calibrations/ecoflex_00_10_drop_sofa_stable_nh.yaml",
    "SOFA, Ogden": "calibrations/ecoflex_00_10_drop_sofa_ogden.yaml",
}


def _simulate(path: str):
    cal = Calibration(ROOT / path)
    report = yaml.safe_load((cal.output / "report.yaml").read_text())
    return simulate_cases(cal.dataset, cal.tracks, list(cal.cases), [*cal.overrides, *best_overrides(report)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    args = ap.parse_args()
    with ProcessPoolExecutor(len(CALIBRATIONS)) as pool:
        sims = dict(zip(CALIBRATIONS, pool.map(_simulate, CALIBRATIONS.values())))
    cal = Calibration(ROOT / next(iter(CALIBRATIONS.values())))
    args.out.mkdir(parents=True, exist_ok=True)
    print(overlay(cal.tracks, sims, args.out / "model_comparison_overlay.png",
                  title="Best fit per backend/model vs experiment (grey = compared window)"))
    masses = {c: cal.dataset.case(c).weight_mass for c in cal.cases}
    path, df = peak_summary(cal.tracks, sims, masses, args.out / "model_comparison_peaks.png")
    print(path)
    print(df.to_string())


if __name__ == "__main__":
    main()

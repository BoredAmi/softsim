"""Whole drop trajectories (0-100 ms) vs experiment: docs/media/after_contact_*.png.

    python tools/make_trajectory_figure.py

Compares the first-contact fit on a rigid table (M3) with the same material plus
fitted tack and compliant support (calibrations/ecoflex_00_10_drop_adhesion_support.yaml;
its report must exist). Needs the tracking results. About 2 min on 6 cores.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from softsim.experiments.calibrate import Calibration, best_overrides
from softsim.experiments.plots import simulate_cases, trajectories

ROOT = Path(__file__).resolve().parents[1]
FIT = "calibrations/ecoflex_00_10_drop_adhesion_support.yaml"
CASES = ["w020", "w050", "w500"]
BASE = ["specimen.material=ecoflex_00_10_drop_fit", "specimen.mesh.size=0.005", "simulation.timestep=0.00005",
        "simulation.output_rate=5000"]
NOTES = {"w050": "weight flips from ~15 ms: its centroid is not its COM",
         "w500": "first contact not reproduced (see M4)"}


def _variants() -> dict[str, list[str]]:
    cal = Calibration(ROOT / FIT)
    report = yaml.safe_load((cal.output / "report.yaml").read_text())
    return {
        "first-contact fit, rigid table, no tack": BASE,
        "+ tack + compliant support (fitted)": [*cal.overrides, *best_overrides(report)],
    }


def _simulate(overrides: list[str]):
    cal = Calibration(ROOT / FIT)
    return simulate_cases(cal.dataset, cal.tracks, CASES, overrides, duration=0.105)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    args = ap.parse_args()
    variants = _variants()
    with ProcessPoolExecutor(len(variants)) as pool:
        sims = dict(zip(variants, pool.map(_simulate, variants.values())))
    cal = Calibration(ROOT / FIT)
    args.out.mkdir(parents=True, exist_ok=True)
    path = trajectories(cal.tracks, sims, args.out / "after_contact_trajectories.png", notes=NOTES,
                        title="Drop tests, whole trajectory after contact: simulation vs experiment")
    print(path)


if __name__ == "__main__":
    main()

"""Can an unobserved out-of-plane tilt explain the 500 g first contact? (docs/experiments.md)

    python tools/sweep_out_of_plane_tilt.py            # ~10 min on 4 cores
    python tools/sweep_out_of_plane_tilt.py --case w200 --backend mujoco

The side camera measures the in-plane tilt (about the camera axis) but not the
tilt towards or away from the camera. This runs a case with its measured impact
(speed, offset, spin, in-plane tilt) on MuJoCo, for out-of-plane tilts 0-20 deg,
and compares the first
contact (peak, time of peak) with the recording, using the experiment's
contact rule. Writes runs/out_of_plane_tilt/<case>_<model>.csv and prints a table.

Materials (``--model``):
    effective  the first-contact fit ecoflex_00_10_drop_fit (the bounce validation's rigid-table model)
    transfer   the MuJoCo transfer of the physical card (Prony g 0.94): inverts elements under 500 g
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from softsim.experiments.calibrate import Calibration
from softsim.experiments.compare import ExperimentCase, simulated_indentation

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "calibrations" / "transfer_ecoflex_00_10_mujoco.yaml"
FIT = ROOT / "calibrations" / "ecoflex_00_10_drop_adhesion_support.yaml"  # dataset + tracks of the recordings
OUT = ROOT / "runs" / "out_of_plane_tilt"
TILTS_DEG = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20]


EFFECTIVE = ["specimen.material=ecoflex_00_10_drop_fit", "specimen.mesh.size=0.005", "simulation.timestep=0.00005",
             "output.save_model=false"]


def _material(model: str = "effective") -> list[str]:
    import yaml

    if model == "effective":
        return EFFECTIVE
    cal = Calibration(TRANSFER)
    best = yaml.safe_load((cal.output / "report.yaml").read_text())["best"]
    return [*cal.overrides, *(f"{k}={v}" for k, v in best.items())]


def _first_contact(d: pd.DataFrame) -> tuple[float, float]:
    d = d[(d.t >= 0) & (d.t <= 0.02)].dropna(subset=["indentation_rel"])
    k = d.indentation_rel.idxmax()
    return float(d.indentation_rel[k]), float(d.t[k])


def _run(task) -> dict:
    from softsim.runner import run_config

    case, tilt_deg, material = task
    cal = Calibration(FIT)
    exp = ExperimentCase.load(cal.tracks, case)
    ov = [o for o in exp.sim_overrides() if not o.startswith("drop.tilt")]
    tilt = exp.meta.get("tilt_at_contact", {})
    tilt_y = math.radians(tilt["tilt_deg"]) if tilt.get("reliable") else 0.0
    ov += [f"drop.tilt=[{math.radians(tilt_deg)}, {tilt_y}]", "simulation.duration=0.02"]
    ts = run_config(cal.dataset.config_path(case), [*material, *ov]).timeseries
    peak, t_peak = _first_contact(simulated_indentation(ts))
    return {"case": case, "tilt_x_deg": tilt_deg, "tilt_y_deg": math.degrees(tilt_y), "peak_mm": 1e3 * peak,
            "t_peak_ms": 1e3 * t_peak}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="w500")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model", choices=["effective", "transfer"], default="effective")
    args = ap.parse_args()
    material = _material(args.model)
    exp = ExperimentCase.load(Calibration(FIT).tracks, args.case)
    peak_exp, t_exp = _first_contact(exp.data)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(_run, [(args.case, t, material) for t in TILTS_DEG]))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{args.case}_{args.model}.csv", index=False)
    print(f"{args.case} measured: first contact {1e3 * peak_exp:.2f} mm at {1e3 * t_exp:.2f} ms")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

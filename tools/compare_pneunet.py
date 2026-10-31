"""PneuNet finger on every backend against the reference (docs/actuation.md).

    python tools/compare_pneunet.py            # runs what is missing (MuJoCo ~8 min, SOFA, fem: long)
    python tools/compare_pneunet.py --force    # rerun everything

Runs scenarios/actuation/pneunet.yaml (six chambers in Ecoflex 00-30, a 1 MPa
strain-limiting layer, bonded base, 0-8 kPa) per backend, and compares the tip
rotation and the cavity volume at half pressure (4 kPa, mid-ramp at 200 ms; the
ramp is quasi-static) and held at 8 kPa (mean over 450-500 ms) with the 3D
reference. Writes runs/pneunet/report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_cable import run_case  # noqa: E402

CONFIG = ROOT / "scenarios" / "actuation" / "pneunet.yaml"
OUT = ROOT / "runs" / "pneunet"
RUNS = {  # name -> (backend, overrides)
    "fem": ("fem", ["simulation.timestep=2ms"]),
    "mujoco": ("mujoco", []),
    "sofa": ("sofa", ["simulation.timestep=1ms"]),
}
LOADS = {"4 kPa": (0.199, 0.201), "8 kPa": (0.45, 0.5)}  # sample window [s]


def _run(name: str, force: bool) -> Path | None:
    return run_case(CONFIG, OUT / name, *RUNS[name], force)


def _values(out: Path) -> dict[str, dict[str, float]]:
    ts = pd.read_parquet(out / "timeseries.parquet")
    v0 = ts["cavity_volume"].iloc[0]
    res = {}
    for k, (lo, hi) in LOADS.items():
        w = ts[(ts.t >= lo - 1e-9) & (ts.t <= hi + 1e-9)]
        res[k] = {"bend_deg": float(np.degrees(w["bend_angle"].mean())), "volume_ratio": float(w["cavity_volume"].mean() / v0),
                  "tip_z_mm": float(1e3 * (w["tip_z"].mean() - ts["tip_z"].iloc[0]))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    values = {}
    for name in RUNS:
        out = _run(name, args.force)
        if out is not None:
            values[name] = _values(out)
    ref = values.get("fem")
    report = {"values": values, "relative_to_fem": {
        n: {k: {q: (x / ref[k][q] - 1 if q == "bend_deg" else (x - 1) / (ref[k][q] - 1) - 1)
                for q, x in d.items() if q != "tip_z_mm"} for k, d in v.items()} for n, v in values.items()} if ref else {}}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("Tip rotation [deg] and cavity V/V0; in brackets: relative to the reference (angle; volume change)")
    print("| Model | " + " | ".join(f"{k}: angle | {k}: V/V0" for k in LOADS) + " |")
    print("| --- |" + " --- | --- |" * len(LOADS))
    for name, v in values.items():
        cells = []
        for k in LOADS:
            r = report["relative_to_fem"].get(name, {}).get(k, {}) if name != "fem" else {}
            cells.append(f"{v[k]['bend_deg']:.2f}" + (f" ({100 * r['bend_deg']:+.0f}%)" if r else ""))
            cells.append(f"{v[k]['volume_ratio']:.3f}" + (f" ({100 * r['volume_ratio']:+.0f}%)" if r else ""))
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

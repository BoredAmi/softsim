"""Cast finger of Wang et al. (Actuators 15(9), 488, 2026) against its measured and published-FE chord angles.

    softsim run scenarios/actuation/ribbed_finger.yaml --backend fem --out runs/wang/fem_1p5mm \\
        --set body.mesh.size=1.5mm --set simulation.timestep=10ms
    python tools/compare_wang_finger.py runs/wang/fem_1p5mm [runs/wang/other ...]      # table (+ figure with --plot)

Each run's chord angle is read at the paper's pressures (5-40 kPa) from the smooth quasi-static ramp (the
pressure ramp is slow against the finger's dynamics, so the held values follow). The table and the figure
compare with the paper's measurements (Table 5) and its Abaqus chord angles (datasets/wang2026_finger.yaml,
CC BY 4.0). Writes docs/media/wang_finger.png with --plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "wang2026_finger.yaml"


def timeseries(run: Path) -> pd.DataFrame:
    """A run's timeseries: a ``softsim run`` output directory, or a CSV a script wrote."""
    return pd.read_csv(run) if run.suffix == ".csv" else pd.read_parquet(run / "timeseries.parquet")


def chord_at(run: Path, pressures_kpa: np.ndarray) -> np.ndarray:
    """Chord angle [deg] of a run at the given cavity pressures (the ramp is monotone)."""
    ts = timeseries(run)
    p = ts["cavity_pressure"].to_numpy() / 1e3
    keep = np.concatenate([[True], np.diff(p) > 1e-9])  # the rising part, once
    return np.degrees(np.interp(pressures_kpa, p[keep], ts["chord_angle"].to_numpy()[keep]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    table = np.array(yaml.safe_load(DATA.read_text())["table"], dtype=float)
    p, measured, paper_fe = table[:, 0], table[:, 1], table[:, 2]
    sims = {r.name: chord_at(r, p) for r in args.runs}
    print("chord angle [deg]; error against the measurement in brackets")
    print("| p [kPa] | measured | paper FE | " + " | ".join(sims) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in sims) + " |")
    for i, pk in enumerate(p):
        cells = [f"{v[i]:.2f} ({v[i] - measured[i]:+.2f})" for v in sims.values()]
        print(f"| {pk:.0f} | {measured[i]:.2f} | {paper_fe[i]:.2f} ({paper_fe[i] - measured[i]:+.2f}) | " + " | ".join(cells) + " |")
    rms = lambda v, ref: float(np.sqrt(np.mean((v - ref) ** 2)))  # noqa: E731
    print(f"\nRMS error against the measurement [deg]: paper FE {rms(paper_fe, measured):.2f}, "
          + ", ".join(f"{k} {rms(v, measured):.2f}" for k, v in sims.items()))
    print("RMS difference from the paper's FE [deg]: " + ", ".join(f"{k} {rms(v, paper_fe):.2f}" for k, v in sims.items()))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        ax.plot(p, measured, "ko", label="measured (Wang et al.)")
        ax.plot(p, paper_fe, "k--", label="their Abaqus FE")
        for k, v in sims.items():
            ax.plot(p, v, "-s", ms=3, label=k)
        ax.set_xlabel("pressure [kPa]")
        ax.set_ylabel("chord angle [deg]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = ROOT / "docs" / "media" / "wang_finger.png"
        fig.savefig(out)
        print("wrote", out)


if __name__ == "__main__":
    main()

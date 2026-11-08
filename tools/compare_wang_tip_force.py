"""Blocked tip force of the cast finger against Wang et al.'s measurement (Table 7), which no fit here uses.

    softsim run scenarios/actuation/ribbed_finger.yaml --backend fem --out runs/wang/blocked_1p5mm \\
        --set body.mesh.size=1.5mm --set simulation.timestep=10ms --set simulation.duration=900ms \\
        --set "body.prescribed=[{region: tip, path: [[0 ms, [null, null, 0 mm]], [900 ms, [null, null, 0 mm]]]}]"
    python tools/compare_wang_tip_force.py runs/wang/blocked_1p5mm [more runs ...]

The tip face is held in z only, a roller: the finger can't bend down, its tip still slides sideways and the face
still turns, which is the closest this model comes to a tip pressing on a gauge whose contact point the paper doesn't
give. The force is the face's reaction ``tip_reaction_z`` [N], read at the paper's pressures on the rising ramp. A run
is a ``softsim run`` directory or a CSV of the same timeseries (datasets/wang2026_finger.yaml, CC BY 4.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from compare_wang_finger import DATA, timeseries


def force_at(run: Path, pressures_kpa: np.ndarray) -> np.ndarray:
    """Blocked tip force [N] of a run at the given cavity pressures, from the rising part of the ramp."""
    ts = timeseries(run)
    p = ts["cavity_pressure"].to_numpy() / 1e3
    keep = np.concatenate([[True], np.diff(p) > 1e-9])
    if p.max() < pressures_kpa.max():
        raise SystemExit(f"{run} reaches {p.max():.1f} kPa, short of {pressures_kpa.max():.0f}")
    return np.abs(np.interp(pressures_kpa, p[keep], ts["tip_reaction_z"].to_numpy()[keep]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    table = np.array(yaml.safe_load(DATA.read_text())["tip_force"], dtype=float)
    p, measured = table[:, 0], table[:, 1]
    sims = {r.stem if r.suffix else r.name: force_at(r, p) for r in args.runs}
    print("blocked tip force [N]; the share of the measurement in brackets")
    print("| p [kPa] | measured | " + " | ".join(sims) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in sims) + " |")
    for i, pk in enumerate(p):
        print(f"| {pk:.0f} | {measured[i]:.2f} | " + " | ".join(f"{v[i]:.3f} ({v[i] / measured[i]:.2f})" for v in sims.values()) + " |")


if __name__ == "__main__":
    main()

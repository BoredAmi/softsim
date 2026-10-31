"""Hollow-sphere inflation on every backend against the reference and closed forms (docs/actuation.md).

    python tools/compare_inflation.py            # runs what is missing (~1 min MuJoCo, ~15 min fem, SOFA)
    python tools/compare_inflation.py --force    # rerun everything

Runs scenarios/inflation/hollow_sphere.yaml (3 kPa in a 40/20 mm sphere of a
30 kPa, nu = 0.45 Neo-Hookean card) per backend and mesh, and compares the cavity
volume, held at 3 kPa (mean over 200-250 ms), with the 3D reference at the same
card. Context columns: linear Lame with the card's nu, and the exact
incompressible finite-strain solution at the card's shear modulus. Writes
runs/inflation/report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scenarios" / "inflation" / "hollow_sphere.yaml"
OUT = ROOT / "runs" / "inflation"
RUNS = {  # name -> (backend, overrides)
    "fem": ("fem", ["simulation.timestep=2ms"]),
    "mujoco": ("mujoco", []),
    "mujoco_2mm": ("mujoco", ["body.mesh.size=2mm"]),
    "sofa": ("sofa", ["simulation.timestep=1ms"]),
}
HOLD = (0.2, 0.25)
E, NU, A, B, P = 30e3, 0.45, 0.01, 0.02, 3e3


def _run(name: str, force: bool) -> Path | None:
    backend, overrides = RUNS[name]
    out = OUT / f"sphere_{name}"
    if (out / "timeseries.parquet").exists() and not force:
        return out
    cmd = [str(ROOT / ".venv" / "bin" / "softsim"), "run", str(CONFIG), "--backend", backend, "--out", str(out),
           "--quiet", *sum((["--set", o] for o in overrides), [])]
    env = {**os.environ}
    env.setdefault("SOFA_ROOT", str(Path.home() / ".local/opt/sofa/SOFA_v26.06.00_Linux"))
    print("running", name, file=sys.stderr)
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        print(f"{name} failed:\n{done.stderr[-2000:]}", file=sys.stderr)
        return None
    return out


def _held_ratio(out: Path) -> float:
    ts = pd.read_parquet(out / "timeseries.parquet")
    hold = ts[(ts.t >= HOLD[0]) & (ts.t <= HOLD[1])]
    return float(hold["cavity_volume"].mean() / ts["cavity_volume"].iloc[0])


def main() -> None:
    from softsim.analytic import hollow_sphere_linear, hollow_sphere_pressure

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ratios = {}
    for name in RUNS:
        out = _run(name, args.force)
        if out is not None:
            ratios[name] = _held_ratio(out)
    mu = E / (2 * (1 + NU))
    la = np.linspace(1.0, 1.5, 50001)
    exact = float(la[np.argmin(np.abs(hollow_sphere_pressure(mu, A, B, la) - P))] ** 3)
    linear = float((1 + hollow_sphere_linear(E, NU, A, B, P, A) / A) ** 3)
    ref = ratios.get("fem")
    report = {"volume_ratio": ratios, "linear_lame": linear, "incompressible_exact": exact,
              "relative_to_fem": {k: (v - 1) / (ref - 1) - 1 for k, v in ratios.items()} if ref else {}}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(f"Cavity volume V/V0 at {P / 1e3:.0f} kPa (held); error = volume change relative to the reference")
    print("| Model | V/V0 | Volume change vs reference |")
    print("| --- | --- | --- |")
    for name, v in ratios.items():
        rel = f"{100 * report['relative_to_fem'][name]:+.1f}%" if ref else "-"
        print(f"| {name} | {v:.4f} | {rel} |")
    print(f"| linear Lame, nu = {NU} | {linear:.4f} | "
          f"{100 * ((linear - 1) / (ref - 1) - 1):+.1f}% |" if ref else "")
    print(f"| exact, incompressible, mu = {mu / 1e3:.2f} kPa | {exact:.4f} | "
          f"{100 * ((exact - 1) / (ref - 1) - 1):+.1f}% |" if ref else "")


if __name__ == "__main__":
    main()

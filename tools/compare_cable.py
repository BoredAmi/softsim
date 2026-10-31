"""Cable-bent cantilever on every backend against Euler-Bernoulli and the reference (docs/actuation.md).

    python tools/compare_cable.py            # runs what is missing (~1 min MuJoCo 2 mm, fem ~10 min, SOFA)
    python tools/compare_cable.py --force    # rerun everything

Runs scenarios/actuation/cable_cantilever.yaml (10 x 10 x 60 mm beam, E = 500
kPa, cable 3 mm off-axis, anchored at 55 mm) per backend and mesh. It compares
the top face's tilt at half tension (0.35 N, mid-ramp at 200 ms; the ramp is
quasi-static) and held at full tension (0.7 N, mean over 450-500 ms) with the
pure-moment solution theta = T d L / (E_t I), and with the 3D reference.
E_t = E (1 + 2 T / (E A)) is the Neo-Hookean tangent under the cable's axial
pre-compression T / (E A). Writes runs/cable/report.json.
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
CONFIG = ROOT / "scenarios" / "actuation" / "cable_cantilever.yaml"
OUT = ROOT / "runs" / "cable"
RUNS = {  # name -> (backend, overrides)
    "fem": ("fem", ["simulation.timestep=2ms"]),
    "mujoco": ("mujoco", []),
    "mujoco_1mm": ("mujoco", ["body.mesh.size=1mm", "actuators[0].spacing=3mm"]),  # stable explicit cable
    "sofa": ("sofa", ["simulation.timestep=1ms"]),
}
E, B, L, D = 500e3, 0.01, 0.055, 0.003
LOADS = {"0.35 N": (0.35, (0.199, 0.201)), "0.7 N": (0.7, (0.45, 0.5))}  # tension, sample window [s]


def _run(name: str, force: bool) -> Path | None:
    backend, overrides = RUNS[name]
    out = OUT / name
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


def _angles(out: Path) -> dict[str, float]:
    ts = pd.read_parquet(out / "timeseries.parquet")
    return {k: float(ts[(ts.t >= lo - 1e-9) & (ts.t <= hi + 1e-9)]["bend_angle"].mean())
            for k, (_, (lo, hi)) in LOADS.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    angles = {}
    for name in RUNS:
        out = _run(name, args.force)
        if out is not None:
            angles[name] = _angles(out)
    beam = {k: t * D * L / (E * (1 + 2 * t / (E * B * B)) * B**4 / 12) for k, (t, _) in LOADS.items()}
    ref = angles.get("fem")
    report = {"bend_angle": angles, "euler_bernoulli": beam,
              "relative_to_beam": {n: {k: v / beam[k] - 1 for k, v in a.items()} for n, a in angles.items()},
              "relative_to_fem": {n: {k: v / ref[k] - 1 for k, v in a.items()} for n, a in angles.items()} if ref
              else {}}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print("Tip angle [deg]; in brackets: relative to Euler-Bernoulli, then to the reference")
    print("| Model | " + " | ".join(LOADS) + " |")
    print("| --- |" + " --- |" * len(LOADS))
    print("| Euler-Bernoulli | " + " | ".join(f"{np.degrees(beam[k]):.2f}" for k in LOADS) + " |")
    for name, a in angles.items():
        cells = []
        for k in LOADS:
            rel = f"{100 * report['relative_to_beam'][name][k]:+.1f}%"
            if ref and name != "fem":
                rel += f", {100 * report['relative_to_fem'][name][k]:+.1f}%"
            cells.append(f"{np.degrees(a[k]):.2f} ({rel})")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

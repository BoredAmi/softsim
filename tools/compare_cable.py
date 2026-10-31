"""Cable-bent cantilever on every backend against Euler-Bernoulli and the reference (docs/actuation.md).

    python tools/compare_cable.py            # runs what is missing (~1 min MuJoCo 2 mm, fem ~10 min, SOFA)
    python tools/compare_cable.py --force    # rerun everything
    python tools/compare_cable.py --layered  # the two-material beam (cable_cantilever_layered.yaml)

Runs scenarios/actuation/cable_cantilever.yaml (10 x 10 x 60 mm beam, E = 500
kPa, cable 3 mm off-axis, anchored at 55 mm) per backend and mesh. It compares
the top face's tilt at half tension (0.35 N, mid-ramp at 200 ms; the ramp is
quasi-static) and held at full tension (0.7 N, mean over 450-500 ms) with the
pure-moment solution theta = T d L / (E_t I), and with the 3D reference.
E_t = E (1 + 2 T / (E A)) is the Neo-Hookean tangent under the cable's axial
pre-compression T / (E A). With --layered: the beam with a 2 mm, 5 MPa layer
on its -x face, against composite Euler-Bernoulli (softsim.analytic). MuJoCo
runs with its Stable Neo-Hookean and its StVK flex; the layered / homogeneous
ratio on the same backend and mesh checks the zone against the composite-beam
ratio. Writes runs/cable[_layered]/report.json.
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
STVK = ["body.material.elastic.model=stvk"]
RUNS = {  # name -> (backend, overrides)
    "fem": ("fem", ["simulation.timestep=2ms"]),
    "mujoco": ("mujoco", []),
    "mujoco_1mm": ("mujoco", ["body.mesh.size=1mm", "actuators[0].spacing=3mm"]),  # stable explicit cable
    "sofa": ("sofa", ["simulation.timestep=1ms"]),
    "mujoco_stvk": ("mujoco", STVK),
}
LAYERED_RUNS = {
    "fem": ("fem", ["simulation.timestep=2ms"]),
    "sofa": ("sofa", ["simulation.timestep=1ms"]),
    "mujoco": ("mujoco", []),
    "mujoco_stvk": ("mujoco", [*STVK, "body.zones[0].material.elastic.model=stvk"]),
}
E, B, L, D = 500e3, 0.01, 0.055, 0.003
LAYERS = {False: ([E], [-B / 2, B / 2]), True: ([10 * E, E], [-B / 2, -0.003, B / 2])}  # moduli, x edges
LOADS = {"0.35 N": (0.35, (0.199, 0.201)), "0.7 N": (0.7, (0.45, 0.5))}  # tension, sample window [s]


def _run(name: str, force: bool, layered: bool = False) -> Path | None:
    backend, overrides = (LAYERED_RUNS if layered else RUNS)[name]
    out = (OUT.with_name("cable_layered") if layered else OUT) / name
    config = CONFIG.with_name("cable_cantilever_layered.yaml") if layered else CONFIG
    return run_case(config, out, backend, overrides, force)


def run_case(config: Path, out: Path, backend: str, overrides: list[str], force: bool) -> Path | None:
    """``softsim run`` of ``config`` on ``backend`` into ``out`` unless it is there already; None if it failed."""
    if (out / "timeseries.parquet").exists() and not force:
        return out
    cmd = [str(ROOT / ".venv" / "bin" / "softsim"), "run", str(config), "--backend", backend, "--out", str(out),
           "--quiet", *sum((["--set", o] for o in overrides), [])]
    env = {**os.environ}
    env.setdefault("SOFA_ROOT", str(Path.home() / ".local/opt/sofa/SOFA_v26.06.00_Linux"))
    print("running", out.name, file=sys.stderr)
    done = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        print(f"{out.name} failed:\n{done.stderr[-2000:]}", file=sys.stderr)
        return None
    return out


def _angles(out: Path) -> dict[str, float]:
    ts = pd.read_parquet(out / "timeseries.parquet")
    return {k: float(ts[(ts.t >= lo - 1e-9) & (ts.t <= hi + 1e-9)]["bend_angle"].mean())
            for k, (_, (lo, hi)) in LOADS.items()}


def main() -> None:
    from softsim.analytic import cable_beam_tip_angle

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--layered", action="store_true", help="the beam with a stiff layer")
    args = ap.parse_args()
    out_dir = OUT.with_name("cable_layered") if args.layered else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = {}
    for name in (LAYERED_RUNS if args.layered else RUNS):
        out = _run(name, args.force, args.layered)
        if out is not None:
            angles[name] = _angles(out)
    beam = {k: cable_beam_tip_angle(t, D, L, *LAYERS[args.layered], B) for k, (t, _) in LOADS.items()}
    ref = angles.get("fem")
    report = {"bend_angle": angles, "euler_bernoulli": beam,
              "relative_to_beam": {n: {k: v / beam[k] - 1 for k, v in a.items()} for n, a in angles.items()},
              "relative_to_fem": {n: {k: v / ref[k] - 1 for k, v in a.items()} for n, a in angles.items()} if ref
              else {}}
    if args.layered:
        report["layered_over_homogeneous"] = {"beam": {
            k: beam[k] / cable_beam_tip_angle(t, D, L, *LAYERS[False], B) for k, (t, _) in LOADS.items()}}
        for name in angles:
            if (OUT / name / "timeseries.parquet").exists():
                h = _angles(OUT / name)
                report["layered_over_homogeneous"][name] = {k: angles[name][k] / h[k] for k in LOADS}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
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
    for name, r in report.get("layered_over_homogeneous", {}).items():
        print(f"layered / homogeneous, {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()

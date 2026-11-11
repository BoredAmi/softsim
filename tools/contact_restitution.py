"""How much impact energy does each backend's contact lose? Elastic drop, backends against the 3D reference.

    python tools/contact_restitution.py              # ~15 min: reference + MuJoCo grid on 4 cores
    python tools/contact_restitution.py --replot     # reuse runs/contact_restitution/*.parquet

A centred 50 g weight hits the drop-test specimen at 2.7 m/s. The card is purely
elastic (stable Neo-Hookean, no damping of any kind) with sticking contact, so
an exact solver returns almost all of the energy: the restitution
e = rebound speed / impact speed is close to 1, minus the elastic waves left
ringing in the specimen. The locking-free 3D reference (backend ``fem``,
Newmark numerical damping a = 0.25) sets the target. MuJoCo then runs over its
contact settings (``contact.dampratio``, ``contact.timeconst``, timestep), and
each run is compared on restitution, peak compression and time of the peak.
Further runs isolate the losses: frictionless (both solvers), finer MuJoCo
meshes, MuJoCo's elliptic friction cone, and SOFA (needs ``SOFA_ROOT``). Writes
runs/contact_restitution/summary.csv (docs/backends/mujoco.md, "Energy and
contact fidelity").
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "contact_restitution"
CASE = ROOT / "scenarios" / "drop_test" / "cases" / "w050.yaml"
CARD = ("{name: elastic_probe, density: 1040, elastic: {model: stable_neo_hookean, youngs_modulus: 100 kPa, "
        "poisson_ratio: 0.45}, friction: {sliding: 1.0}}")
COMMON = [f"specimen.material={CARD}", "specimen.mesh.size=0.005", "ground.friction.sliding=1.0", "drop.gap=0.0005",
          "drop.initial_speed=2.7", "drop.offset=[0.0, 0.0]", "simulation.duration=0.015", "simulation.output_rate=10000",
          "output.save_model=false"]
REFERENCE = ["backend=fem", "simulation.timestep=0.00005"]
FRICTIONLESS = ["specimen.material.friction.sliding=0.0", "ground.friction.sliding=0.0"]  # the weight keeps its 0.5
# every pair frictionless: SOFA applies the scene's largest coefficient to all pairs, so with the weight at 0.5 its
# specimen-ground contact had 0.5 too, while the reference and MuJoCo (per pair) had 0 there
ALL_FRICTIONLESS = [*FRICTIONLESS, "weight.friction.sliding=0.0"]
FINE = ["simulation.timestep=1.25e-5", "contact.timeconst=0.1 ms"]
EXTRA = {
    "reference_frictionless": [*REFERENCE, *FRICTIONLESS],
    "mujoco_frictionless_5mm": ["backend=mujoco", *FINE, *FRICTIONLESS],
    "mujoco_frictionless_3.5mm": ["backend=mujoco", *FINE, *FRICTIONLESS, "specimen.mesh.size=0.0035"],
    "mujoco_frictionless_2.5mm": ["backend=mujoco", *FINE, *FRICTIONLESS, "specimen.mesh.size=0.0025"],
    "sofa": ["backend=sofa", "simulation.timestep=5e-5"],
    "sofa_frictionless": ["backend=sofa", "simulation.timestep=5e-5", *FRICTIONLESS],
    "reference_all_frictionless": [*REFERENCE, *ALL_FRICTIONLESS],
    "sofa_all_frictionless": ["backend=sofa", "simulation.timestep=5e-5", *ALL_FRICTIONLESS],
    "mujoco_elliptic": ["backend=mujoco", "simulation.timestep=2.5e-5", "contact.timeconst=0.1 ms", "contact.cone=elliptic"],
    "mujoco_elliptic_impratio10": ["backend=mujoco", "simulation.timestep=2.5e-5", "contact.timeconst=0.1 ms",
                                   "contact.cone=elliptic", "contact.impratio=10"],
}
GRID = {"contact.dampratio": [1.0, 0.5, 0.2, 0.05], "contact.timeconst": ["0.2 ms", "0.1 ms"],
        "simulation.timestep": [5e-5, 2.5e-5]}


def _label(ov: list[str]) -> str:
    return ", ".join(o.split(".", 1)[1] for o in ov) or "reference"


def _run(task: tuple[str, list[str]]) -> tuple[str, str]:
    from softsim.runner import run_config

    name, ov = task
    path = OUT / f"{name}.parquet"
    if not path.exists():
        try:
            run_config(CASE, [*COMMON, *ov]).timeseries.to_parquet(path)
        except Exception as exc:  # noqa: BLE001 - report and keep the other runs
            return name, f"{type(exc).__name__}: {exc}"
    return name, ""


def _metrics(ts: pd.DataFrame) -> dict[str, float]:
    v = ts["weight_vz"].to_numpy()
    c = ts["compression"].to_numpy()
    t = ts["t"].to_numpy()
    touching = np.nonzero(ts["contact_force_z"].to_numpy() > 0)[0]
    if not len(touching):
        return {"restitution": np.nan, "peak_mm": np.nan, "t_peak_ms": np.nan}
    v_in = -v[max(touching[0] - 1, 0)]
    after = touching[-1] + 1 if touching[-1] + 1 < len(v) else len(v) - 1
    k = int(np.argmax(c))
    return {"restitution": float(v[after] / v_in), "peak_mm": 1e3 * float(c[k]),
            "t_peak_ms": 1e3 * float(t[k] - t[touching[0]]), "contact_ms": 1e3 * float(t[after] - t[touching[0]])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--replot", action="store_true", help="only summarise existing runs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(GRID)
    tasks = [("reference", REFERENCE)]
    for i, values in enumerate(itertools.product(*GRID.values())):
        tasks.append((f"mujoco_{i:02d}", ["backend=mujoco", *(f"{k}={v}" for k, v in zip(keys, values))]))
    tasks += list(EXTRA.items())
    if not args.replot:
        with ProcessPoolExecutor(args.jobs) as pool:
            for name, err in pool.map(_run, tasks):
                if err:
                    print(f"{name}: {err}")
    rows = []
    for name, ov in tasks:
        path = OUT / f"{name}.parquet"
        if path.exists():
            rows.append({"run": name, "settings": "fem reference" if name == "reference" else _label(ov[1:]),
                         **_metrics(pd.read_parquet(path))})
    df = pd.DataFrame(rows)
    ref = df[df.run == "reference"].iloc[0] if (df.run == "reference").any() else None
    if ref is not None:
        for col in ("restitution", "peak_mm", "t_peak_ms"):
            df[f"{col}_vs_ref"] = df[col] / ref[col] - 1.0
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

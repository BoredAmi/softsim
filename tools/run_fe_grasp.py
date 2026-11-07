"""One finite-element grasp with its surface recorded, for the page's Truth tab and the fast model's held-out check.

    python tools/run_fe_grasp.py --mass 10 --out runs/grasp/box_10g.npz

Runs scenarios/grasp/box.yaml with vertex recording. Writes t, x (F, V, 3), tris, rest, object (F, 3), grip (F,),
palm (F,), held. Hours on the RTX 3060: run in the background, from a frozen copy of the package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softsim.runner import Simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "grasp" / "box.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, default=10.0, help="g")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    over = ["output.save_model=false", f"object.mass={a.mass}g", *a.overrides] + ([f"object.friction={a.friction}"] if a.friction else [])
    sim = Simulation(SCENARIO, over, backend="fem")
    result = sim.run(record_vertices=True)
    ts = result.timeseries
    body = sim.scene.soft_bodies[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, t=ts["t"].to_numpy(), x=result.vertices[body.name], tris=body.mesh.surface_triangles(),
                        rest=body.mesh.points, object=ts[["object_x", "object_y", "object_z"]].to_numpy(),
                        grip=ts["grip_force"].to_numpy(), palm=ts["palm_lift"].to_numpy(), held=result.summary["held"])
    print(a.out, result.summary, flush=True)


if __name__ == "__main__":
    main()

"""Stiffness probes of the ribbed finger on the reference: inflate, push, and record the shape, all quasi-statically.

    python tools/probe_finger.py --pressures 5 15 20 25 35 --forces 0.1 0.2 --out runs/reduced/probes.json

At each pressure [kPa] the static solver (no inertia, so every state is settled) steps the cavity up, then applies a
force along +z, against the curl, first on the ``tip`` face and then, from the same inflated state, spread over the
pads (``bottom``). Each state is written as its layer stations (:func:`softsim.reduced.finger.layer_points`), so the
fast finger (softsim/reduced) can be compared along its whole length. The tip load sets the stiffness; the spread load
and the pressure left out of the fit test it. Same scenario, mesh and card as the validation
(scenarios/actuation/ribbed_finger.yaml); results are appended as each pressure ends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from softsim.backends.fem.backend import FemBackend
from softsim.config.loader import load_yaml
from softsim.fem.solid3d import Solid3DSolver
from softsim.reduced.finger import ReducedFingerConfig, layer_points
from softsim.runner import Simulation

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "actuation" / "ribbed_finger.yaml"
PRESSURE_STEP = 5000.0  # Pa per static step


def build(mesh: str) -> tuple[Solid3DSolver, np.ndarray, float]:
    raw = load_yaml(SCENARIO)
    raw["actuators"] += [{"name": name, "type": "force", "region": region, "direction": [0, 0, 1]}
                         for name, region in (("tip_load", "tip"), ("pad_load", "bottom"))]
    sim = Simulation(raw, ["simulation.timestep=10ms", f"body.mesh.size={mesh}"], backend="fem", base_dir=SCENARIO.parent)
    scene = sim.scenario.build_scene()
    fem = FemBackend()
    fem.build(scene)
    body = scene.soft_bodies[0]
    return Solid3DSolver(fem._problem, dynamic=False), body.mesh.points, float(body.mesh.points[:, 0].max())


def stations(s: Solid3DSolver, rest: np.ndarray, length: float) -> list:
    n = len(rest)
    x = s.p.mesh.nodes[:n] + s.y[:3 * s.p.mesh.nodes.shape[0]].reshape(-1, 3)[:n]
    return layer_points(rest, x[None], length, ReducedFingerConfig())[0].tolist()


def settle(s: Solid3DSolver) -> None:
    if not s.advance(1.0):
        raise RuntimeError(f"no static solution at pressure {s.pressures}, forces {s.forces}")


def probe(s: Solid3DSolver, rest: np.ndarray, length: float, pressure: float, forces: list[float]) -> dict:
    """The states at ``pressure`` [Pa]: {"free": stations, "tip": [...], "pads": [...]}, one entry per force."""
    start = float(s.pressures[0])  # pressures ascend, each carrying on from the last inflated state
    for p in np.append(np.arange(start + PRESSURE_STEP, pressure, PRESSURE_STEP), pressure):
        s.pressures[:] = p
        settle(s)
    inflated = s.y.copy()
    row = dict(pressure=pressure, forces=forces, free=stations(s, rest, length), tip=[], pads=[])
    for k, key in enumerate(("tip", "pads")):
        s.y[:] = inflated
        for f in forces:
            s.forces[:] = 0.0
            s.forces[k] = f
            settle(s)
            row[key].append(stations(s, rest, length))
    s.forces[:] = 0.0
    s.y[:] = inflated
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pressures", type=float, nargs="+", required=True, help="kPa, increasing")
    ap.add_argument("--forces", type=float, nargs="+", default=[0.1, 0.2], help="N, increasing")
    ap.add_argument("--mesh", default="1.5mm")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(a.out.read_text()) if a.out.exists() else []
    s, rest, length = build(a.mesh)
    for p in a.pressures:
        rows.append(probe(s, rest, length, p * 1e3, a.forces))
        a.out.write_text(json.dumps(rows))
        print(p, "kPa free tip", rows[-1]["free"][-1], "loaded", rows[-1]["tip"][-1][-1], rows[-1]["pads"][-1][-1], flush=True)


if __name__ == "__main__":
    main()

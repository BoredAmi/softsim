"""Hold every grasp the map calls held for much longer, and show what the slow sink is.

    python tools/check_long_hold.py runs/reduced/grasp_map.json --finger softsim/reduced/wang_finger.json --out runs/reduced/long_hold.json

The map stops 2 s after the lift. This runs each held cell on the same schedule to ``END`` and asks
:func:`softsim.scenarios.grasp.is_held` again at each of ``TIMES``. Then one box and one cylinder run with MuJoCo's
``impratio`` raised: a sink that shrinks in proportion is the solver's soft friction creeping, not the grasp slipping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS
from softsim.scenarios.grasp import is_held

SETTLE, LIFT, RAMP = 1500, 0.04, 600  # steps, m, steps: run_grasp's schedule
TIMES = (3.5, 10.0, 20.0)  # s after the pressure starts; 3.5 s is where the map stops
END = TIMES[-1]
IMPRATIOS = (10.0, 100.0)  # the model's value, and ten times it


def rises(finger: Path, shape: str, mu: float, mass: float, pressure: float, end: float = END, impratio: float | None = None) -> tuple[np.ndarray, float]:
    """Height above the settled grasp [m] after each step of the lift and hold, and the timestep [s]."""
    g = ReducedGripper(ReducedFinger.from_json(finger), fingers=2, shape=shape, size=OBJECTS[shape], mass=mass, friction=mu)
    if impratio:
        g.model.opt.impratio = impratio
    for i in range(SETTLE):
        g.step(pressure * min(i / RAMP, 1))
    z0, dt = g.object_height, g.model.opt.timestep
    out = np.zeros(round(end / dt) - SETTLE)
    for i in range(len(out)):
        g.step(pressure, LIFT * min(i / 500, 1))
        out[i] = g.object_height - z0
    return out, dt


def cell(finger: Path, c: tuple) -> dict:
    r, dt = rises(finger, *c)
    n = [round(t / dt) - SETTLE for t in TIMES]
    return dict(shape=c[0], mu=c[1], mass=c[2], pressure=c[3], rise=[float(r[k - 1]) for k in n],
                held=[bool(is_held(np.arange(k) * dt, r[:k])) for k in n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    m = json.loads(a.map.read_text())
    cells = [(s, float(mu), m["masses"][i], m["pressures"][j]) for s, by_mu in m["grid"].items() for mu, rows in by_mu.items()
             for i, row in enumerate(rows) for j, held in enumerate(row) if held]
    with Pool(a.workers) as pool:
        out = pool.map(partial(cell, a.finger), cells, chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out))
    for k, t in enumerate(TIMES):
        by = Counter(c["shape"] for c in out if c["held"][k])
        print(f"held at {t:4.1f} s: {sum(by.values()):3d} of {len(out)}  {dict(sorted(by.items()))}")
    for impratio in IMPRATIOS:
        r, dt = rises(a.finger, "box", 0.8, 0.01, 30e3, end=11.5, impratio=impratio)
        i, j = round(3.5 / dt) - SETTLE, round(6 / dt) - SETTLE
        print(f"10 g box, 30 kPa, impratio {impratio:5.0f}: sinks {(r[i] - r[j]) * 1e3 / 2.5:.2f} mm/s from 3.5 to 6 s")


if __name__ == "__main__":
    main()

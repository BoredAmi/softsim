"""Which objects the gripper holds: fast-model grasps swept over object, mass, friction and pressure.

    python tools/grasp_map.py --finger softsim/reduced/wang_finger.json --out runs/reduced/grasp_map.json

Every cell is one :func:`softsim.reduced.mjcf.run_grasp`, two fingers; the page's Grasp map tab and the README figure
read the JSON. Each entry of ``grid[shape][friction]`` is a (masses x pressures) table of 1 for held, 0 for dropped.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

from softsim.reduced import ReducedFinger
from softsim.reduced.mjcf import OBJECTS, run_grasp

MASSES = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]  # kg
PRESSURES = [10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3]  # Pa
FRICTIONS = [0.4, 0.8, 1.2]


def cell(finger: Path, case: tuple) -> tuple:
    shape, mu, m, p = case
    r = run_grasp(ReducedFinger.from_json(finger), p, m, friction=mu, shape=shape)
    return case, r["held"], r["grip"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cases = list(product(OBJECTS, FRICTIONS, MASSES, PRESSURES))
    with ProcessPoolExecutor(a.jobs) as pool:
        results = list(pool.map(partial(cell, a.finger), cases, chunksize=8))
    grid = {s: {str(mu): np.zeros((len(MASSES), len(PRESSURES)), int).tolist() for mu in FRICTIONS} for s in OBJECTS}
    grip = {s: {str(mu): np.zeros((len(MASSES), len(PRESSURES))).tolist() for mu in FRICTIONS} for s in OBJECTS}
    for (shape, mu, m, p), held, force in results:
        i, j = MASSES.index(m), PRESSURES.index(p)
        grid[shape][str(mu)][i][j] = int(held)
        grip[shape][str(mu)][i][j] = round(force, 4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(masses=MASSES, pressures=PRESSURES, frictions=FRICTIONS, grid=grid, grip=grip)))
    print(f"{len(cases)} grasps, {sum(r[1] for r in results)} held -> {a.out}")


if __name__ == "__main__":
    main()

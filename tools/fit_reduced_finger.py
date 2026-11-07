"""Fit the fast finger: tables from the reference's free-bend ramp, stiffness from its static probes.

    python tools/fit_reduced_finger.py --ramp runs/media/wang/fem.npz --probes runs/reduced/probes_static.json \
        --hold-out 20 --out softsim/reduced/wang_finger.json

The stiffness comes from the tip loads at every pressure but ``--hold-out``. Printed checks, none of them used in the
fit: the free shape against the static solution, the held-out pressure, and the load spread over the pads. Each line
gives the reference's tip displacement, the chain's, and the error [mm].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from softsim.reduced import ReducedFinger, extract_tables
from softsim.reduced.finger import fit_stiffness, probe_error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp", type=Path, required=True)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--hold-out", type=float, default=20.0, help="kPa")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = np.load(a.ramp)
    end = int(np.argmax(d["value"])) + 1  # the ramp up; the hold after it adds nothing
    length = float(d["rest"][:, 0].max())
    p, angles, lengths = extract_tables(d["rest"], d["x"][:end], d["value"][:end], length)
    probes = json.loads(a.probes.read_text())
    fit = [r for r in probes if r["pressure"] != a.hold_out * 1e3]
    f = ReducedFinger(p, angles, lengths, np.ones(len(p)))
    f = ReducedFinger(p, angles, lengths, fit_stiffness(f, fit))
    for r in probes:
        kpa = r["pressure"] / 1e3
        free = f.centreline(r["pressure"])[-1] - np.asarray(r["free"])[-1] + (0.0, 0.002)  # the layer sits 2 mm up
        print(f"{kpa:4.0f} kPa{' (held out)' if r not in fit else '':11s} k {np.interp(r['pressure'], f.pressures, f.stiffness):.4f} "
              f"N m/rad; free tip off the static shape by {np.linalg.norm(free) * 1e3:.1f} mm")
        for key in ("tip", "pads"):
            err = probe_error(f, r, key=key)
            for force, e, s in zip(r["forces"], err, r[key]):
                moved = np.linalg.norm(np.asarray(s)[-1] - np.asarray(r["free"])[-1])
                print(f"    {force:.2f} N on {key:4s}: reference moved {moved * 1e3:5.1f} mm, error {np.linalg.norm(e) * 1e3:5.1f} mm "
                      f"({np.linalg.norm(e) / moved:.0%})")
    f.to_json(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

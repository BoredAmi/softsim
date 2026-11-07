"""Check the drop page's solver against the 3D reference and the recordings.

    python tools/check_drop_solver.py

tools/drop_solver.js (linear tetrahedra, explicit) runs the reference's physical Ecoflex card unchanged: nothing is
fitted, so both comparisons are checks. Each drop runs with the recording's measured impact speed, offset and spin.
Reference tracks: runs/reference_tracks (tools/make_reference_tracks.py). Needs Node.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from softsim.geometry import MeshSettings
from softsim.geometry.soft import SoftCylinder

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("run_drop_solver.js")
TRACKS = ROOT / "data" / "processed" / "czopek2026_ecoflex_drop"
REFERENCE = ROOT / "runs" / "reference_tracks" / "ecoflex_00_10_physical"
HEIGHT, DIAMETER, SPECIMEN_MASS = 0.017, 0.04, 19.49e-3  # m, m, kg
MESH_SIZE = 0.0035  # m
PHYSICAL = dict(E=143.3e3, nu=0.4995, g=0.691, tau=2e-3)  # the reference's card
WEIGHTS = {"w020": (0.020, 0.01125, 0.0056), "w050": (0.050, 0.0135, 0.0099)}  # kg, radius m, height m
WINDOW = 0.010  # s after contact


def mesh(size: float = MESH_SIZE) -> dict:
    m = SoftCylinder(diameter=DIAMETER, height=HEIGHT).mesh(MeshSettings(size=size, method="structured"))
    return dict(points=m.points.ravel().tolist(), tets=m.tets.ravel().tolist())


def run_js(job: dict) -> dict:
    return json.loads(subprocess.run(["node", str(RUNNER)], input=json.dumps(job), capture_output=True, text=True, check=True).stdout)


def drop(card: dict, case: str, cells: dict, duration: float = WINDOW + 1e-3) -> pd.DataFrame:
    """One recorded drop: time from contact [s], indentation [m]."""
    info = yaml.safe_load((TRACKS / f"{case}.yaml").read_text())
    mass, radius, height = WEIGHTS[case]
    gap = 1e-4
    obj = dict(shape="disc", mass=mass, radius=radius, height=height, position=[info["offset_at_contact_mm"] * 1e-3, 0, HEIGHT + height / 2 + gap],
               velocity=[0, 0, -info["impact_speed"]], spin=np.radians(info["rotation_rate_at_contact_deg_s"]))
    material = dict(card, mass=SPECIMEN_MASS)
    r = run_js(dict(mesh=cells, material=material, object=obj, friction=dict(object=5, table=5), duration=duration, sample=1e-4))
    if r["failed"]:
        raise RuntimeError(f"{case}: the solver failed")
    ind = HEIGHT - (np.array(r["z"]) - height / 2)
    t = np.array(r["t"]) - gap / info["impact_speed"]
    out = pd.DataFrame(dict(t=t, indentation=ind))
    out.attrs.update(wall=r["wall"], dt=r["dt"])
    return out


def track(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)[["t", "indentation_rel"]].dropna()
    return d[(d.t >= 0) & (d.t <= WINDOW)]


def rms(sim: pd.DataFrame, ref: pd.DataFrame) -> float:
    """RMS gap over the first contact (while the reference indents), as a share of the reference's peak."""
    peak = ref.indentation_rel.max()
    last = ref.t[(ref.indentation_rel > 0) & (ref.t > ref.t[ref.indentation_rel.idxmax()])].max()
    sel = ref[ref.t <= last]
    return float(np.sqrt(np.mean((np.interp(sel.t, sim.t, sim.indentation) - sel.indentation_rel) ** 2)) / peak)


def table(card: dict, cells: dict) -> None:
    print("case | measured peak | reference peak | page solver peak | gap to reference | gap to measured (first contact)")
    for c in WEIGHTS:
        sim, ref, exp = drop(card, c, cells), track(REFERENCE / f"{c}.parquet"), track(TRACKS / f"{c}.parquet")
        pk = lambda d, col: (d[col].max() * 1e3, d.t[d[col].idxmax()] * 1e3)
        print(f"{c} | {pk(exp, 'indentation_rel')[0]:.2f} mm at {pk(exp, 'indentation_rel')[1]:.2f} ms | {pk(ref, 'indentation_rel')[0]:.2f} mm at {pk(ref, 'indentation_rel')[1]:.2f} ms"
              f" | {pk(sim, 'indentation')[0]:.2f} mm at {pk(sim, 'indentation')[1]:.2f} ms | {rms(sim, ref) * 100:.1f}% | {rms(sim, exp) * 100:.1f}%"
              f"  ({sim.attrs['wall'] / (WINDOW + 1e-3):.0f}x slower than real time)")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    table(PHYSICAL, mesh())


if __name__ == "__main__":
    main()

"""Write the drop page: one self-contained HTML file, solver included.

    python tools/make_drop_page.py --out docs/viewer/drop.html

Embeds three.js, tools/drop_solver.js, the cylinder's mesh, the physical Ecoflex card, library materials and the film
tracks of the 20 and 50 g drops. It opens by double click, with no server. tools/check_drop_solver.py checks the solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from check_drop_solver import DIAMETER, HEIGHT, MESH_SIZE, PHYSICAL, SPECIMEN_MASS, TRACKS
from make_gripper_page import vendor
from softsim.geometry import MeshSettings
from softsim.geometry.soft import SoftCylinder
from softsim.materials.library import load_material

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("drop_page.html")
SOLVER = Path(__file__).with_name("drop_solver.js")
PRESETS = ROOT / "softsim" / "geometry" / "presets" / "rigid_bodies.yaml"
TRACK_END = 0.06  # s after contact
BALL = dict(shape="sphere", label="Steel ball", mass=0.0642, radius=0.0125)
LIBRARY = (("Ecoflex 00-10, slow test", "ecoflex_00_10_marechal2021"), ("Ecoflex 00-30", "ecoflex_00_30_marechal2021"),
           ("Ecoflex 00-50", "ecoflex_00_50_marechal2021"), ("Dragon Skin 20", "dragon_skin_20_marechal2021"),
           ("Dragon Skin 30", "dragon_skin_30_marechal2021"), ("Posilicone A20 (the finger)", "posilicone_a20_wang2026"))
OBJECT_MATERIALS = {"Steel": 7850, "Brass": 8500, "Aluminium": 2700, "PLA": 1240, "Oak": 700}  # kg/m^3, handbook values
CAVEAT = ("Live solver: explicit linear tetrahedra with the reference's physical Ecoflex card, nothing fitted. On the two "
          "filmed drops its peak is within 12%. The film's 500 g case is known not to be a passive impact. Every other "
          "setting is unchecked. Library materials are Neo-Hookean at their initial shear modulus from slow tests, with no "
          "relaxation: under impact a real silicone is stiffer, about three times for Ecoflex 00-10.")


def things() -> dict:
    presets = yaml.safe_load(PRESETS.read_text())
    out = {}
    for key, grams in (("w020", 20), ("w050", 50), ("w100", 100), ("w200", 200), ("w500", 500)):
        s = presets[f"czopek2026_{key}"]["shape"]
        mm = lambda v: float(str(v).split()[0]) * 1e-3
        out[key] = dict(shape="disc", label=f"{grams} g", mass=grams * 1e-3, radius=mm(s["diameter"]) / 2, height=mm(s["height"]),
                        knobRadius=mm(s["knob_diameter"]) / 2, knobHeight=mm(s["knob_height"]))
    return out | dict(ball=BALL, none=dict(shape="none", label="Cylinder"))


def materials() -> list[dict]:
    """Library cards as the page uses them: initial shear modulus [Pa] and density [kg/m^3]."""
    cards = [(label, load_material(name)) for label, name in LIBRARY]
    return [dict(label=label, G=m.elastic.initial_shear_modulus(), density=m.density) for label, m in cards]


def recording(case: str) -> dict:
    info = yaml.safe_load((TRACKS / f"{case}.yaml").read_text())
    d = pd.read_parquet(TRACKS / f"{case}.parquet")[["t", "indentation_rel"]].dropna()
    d = d[(d.t >= 0) & (d.t <= TRACK_END)]
    return dict(speed=info["impact_speed"], offset=info["offset_at_contact_mm"] * 1e-3, spin=float(np.radians(info["rotation_rate_at_contact_deg_s"])),
                track=[[round(t * 1e3, 3), round(i * 1e3, 3)] for t, i in zip(d.t, d.indentation_rel)])


def page() -> str:
    m = SoftCylinder(diameter=DIAMETER, height=HEIGHT).mesh(MeshSettings(size=MESH_SIZE, method="structured"))
    top = int(np.argmin(np.hypot(m.points[:, 0], m.points[:, 1]) - 1e3 * (m.points[:, 2] - HEIGHT)))
    data = dict(mesh=dict(points=np.round(m.points, 7).ravel().tolist(), tets=m.tets.ravel().tolist()), tris=m.surface_triangles().ravel().tolist(),
                height=HEIGHT, specimenMass=SPECIMEN_MASS, card=PHYSICAL, topNode=top, things=things(), caveat=CAVEAT,
                materials=materials(), objectMaterials=OBJECT_MATERIALS, diameter=DIAMETER,
                recordings={c: recording(c) for c in ("w020", "w050")})
    html = TEMPLATE.read_text()
    for token, value in (("/*@THREE@*/", vendor()["three.min.js"].decode()), ("/*@SOLVER@*/", SOLVER.read_text()), ("/*@DATA@*/null", json.dumps(data))):
        html = html.replace(token, value.replace("</script>", "<\\/script>"))
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "viewer" / "drop.html")
    a = ap.parse_args()
    a.out.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{page()}</html>')
    print(f"{a.out}: {a.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

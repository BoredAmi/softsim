"""Ribbed cast finger (Wang et al., Actuators 15(9), 488): the shape rebuilt from the paper's Abaqus mesh."""

import math
from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import enclosed_volume, open_edges
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.geometry import MeshSettings, SoftRibbedFinger
from softsim.materials import load_material
from softsim.scenarios.inflation import InflationConfig, InflationScenario

pytest.importorskip("gmsh")

FINGER = Path(__file__).parents[1] / "scenarios" / "actuation" / "ribbed_finger.yaml"
# Read off the supplement's finger_A20_C3D10H_3mm_1x_*.inp: the corner-node tetrahedra integrate to this
# solid volume, and the cavity surface (3562 faces) encloses this volume [mm^3].
PAPER_SOLID_MM3, PAPER_CAVITY_MM3 = 19568.0, 12400.0
PAPER_BOX_MM = (103.0, 24.0, 16.0)  # length, width, height of the same mesh


def _volume(mesh) -> float:
    p, t = mesh.points, mesh.tets
    return float(np.abs(np.einsum("ij,ij->i", np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]),
                                  p[t[:, 3]] - p[t[:, 0]])).sum() / 6)


def test_dimensions_are_the_papers():
    shape = SoftRibbedFinger()
    assert shape.length * 1e3 == pytest.approx(PAPER_BOX_MM[0]) and shape.width * 1e3 == pytest.approx(24)
    assert shape.top_height * 1e3 == pytest.approx(PAPER_BOX_MM[2])
    # the paper's active chamber-array length a = 89 mm is the cavity's length: the cast less both end walls
    assert (shape.body_length - 2 * shape.wall) * 1e3 == pytest.approx(89)
    assert shape.n_teeth == 19  # 3 mm teeth every 5 mm along the 93 mm cast, as in the mesh
    assert shape.volume() * 1e9 == pytest.approx(PAPER_SOLID_MM3, abs=1e-6)


def test_mesh_has_the_papers_solid_and_cavity_volumes():
    """Teeth included, at 1.5 mm: the meshed solid is the paper mesh's to the last mm^3 (the boxes are exact)
    and the cavity surface is closed and encloses the paper's 12,400 mm^3."""
    shape = SoftRibbedFinger()
    mesh = shape.mesh(MeshSettings(size=0.0015))
    assert _volume(mesh) * 1e9 == pytest.approx(PAPER_SOLID_MM3, rel=1e-6)
    cavity = shape.regions(mesh)["cavity"]
    assert len(open_edges(cavity)) == 0
    assert enclosed_volume(mesh.points, cavity) * 1e9 == pytest.approx(PAPER_CAVITY_MM3, rel=1e-6)


def test_toothless_variant_removes_exactly_the_teeth():
    teeth = 19 * 3e-3 * 24e-3 * 1e-3
    plain = SoftRibbedFinger(tooth_height=0.0)
    assert plain.n_teeth == 0
    assert SoftRibbedFinger().volume() - plain.volume() == pytest.approx(teeth, rel=1e-9)
    mesh = plain.mesh(MeshSettings(size=0.002))
    assert _volume(mesh) == pytest.approx(plain.volume(), rel=1e-6)


def test_card_is_the_papers_yeoh_fit():
    e = load_material("posilicone_a20_wang2026").elastic
    assert e.model == "yeoh" and e.c10 == pytest.approx(44163.0) and e.c20 == pytest.approx(530.71)
    assert e.c30 == pytest.approx(16.251)
    assert e.initial_shear_modulus() == pytest.approx(88326.0)  # mu0 = 2 C10
    assert e.bulk_modulus == pytest.approx(2 / 0.226434e-6, rel=1e-4)  # K = 2 / D1
    nu = (3 * e.bulk_modulus - 2 * e.initial_shear_modulus()) / (2 * (3 * e.bulk_modulus + e.initial_shear_modulus()))
    assert nu == pytest.approx(0.495, abs=1e-3)  # the paper's nu0


def _scenario(overrides=()):
    raw = apply_overrides(load_yaml(FINGER), ["body.mesh.size=2mm", "body.geometry.tooth_height=0mm", *overrides])
    return InflationScenario(validate(InflationConfig, raw), FINGER.parent)


def test_chord_angle_is_the_root_to_tip_line_not_the_tip_rotation():
    """Rigid rotation by a about y: the chord and the tip face turn by a. Rotating only the tip face about
    its own centre turns the tip face (bend_angle) but leaves the chord, so the two metrics differ."""
    from softsim.backends.base import SimState, SoftState

    sc = _scenario()
    body = sc.build_scene().soft_bodies[0]
    rest = body.mesh.points

    def metrics(x):
        return sc.metrics(SimState(0.0, soft={body.name: SoftState(x, np.zeros_like(x))}, controls={"cavity": 0.0}))

    a = 0.3
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    m = metrics(rest @ rot.T)
    assert m["chord_angle"] == pytest.approx(a, rel=1e-6) and m["bend_angle"] == pytest.approx(a, rel=1e-6)
    tip = np.unique(body.region("tip"))
    centre = rest[tip].mean(axis=0)
    x = rest.copy()
    x[tip] = (rest[tip] - centre) @ rot.T + centre
    m = metrics(x)
    assert m["bend_angle"] == pytest.approx(a, rel=0.02) and m["chord_angle"] < 0.02 * a

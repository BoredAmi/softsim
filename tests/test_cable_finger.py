"""Parametric cable finger (notched block with a tendon web): validation, exact geometry, the tendon path
taken from the shape, and its bending against Euler-Bernoulli (plain) and the slots' extra compliance."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from softsim.actuation import embed
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftCableFinger
from softsim.scenarios.inflation import InflationConfig, InflationScenario

FINGER = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_finger.yaml"
DIMS = dict(length=0.06, width=0.012, depth=0.01, segments=4, slot_width=0.003, slot_depth=0.003,
            tendon_offset=0.0045)


def _finger(overrides=()):
    raw = apply_overrides(load_yaml(FINGER), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), FINGER.parent)


def test_validation():
    SoftCableFinger(**DIMS)
    bad = {
        "slots missing": {"slot_width": None, "slot_depth": None},
        "slot as long as a segment": {"slot_width": 0.015},
        "tendon in a slot": {"tendon_offset": 0.003},
        "tendon off the +x side": {"tendon_offset": 0.006},
    }
    for what, change in bad.items():
        with pytest.raises(ValidationError):
            SoftCableFinger(**{**DIMS, **change})
    plain = SoftCableFinger(length=0.06, width=0.012, depth=0.01, tendon_offset=0.0045)  # no slots needed
    assert plain.slots() == [] and plain.volume() == pytest.approx(0.06 * 0.012 * 0.01)


def test_mesh_is_the_exact_geometry():
    """The tetrahedra fill exactly the block minus the slots (volume to rounding), the slots sit at the
    segment boundaries, and every region is where it should be."""
    g = SoftCableFinger(**DIMS)
    assert np.array(g.slots()) == pytest.approx(np.array([(0.0135, 0.0165), (0.0285, 0.0315), (0.0435, 0.0465)]))
    mesh = g.mesh(MeshSettings(size=0.0015, method="gmsh"))
    x = mesh.points[mesh.tets]
    volume = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])).sum() / 6
    assert volume == pytest.approx(g.volume(), rel=1e-9)
    regions = g.regions(mesh)
    p = mesh.points
    for name, test in {"base": lambda q: q[:, 2] < 1e-9, "top": lambda q: np.abs(q[:, 2] - 0.06) < 1e-9,
                       "side": lambda q: np.abs(np.abs(q[:, 1]) - 0.005) < 1e-9}.items():
        assert np.all(test(p[np.unique(regions[name])])), name
    cable_side = p[np.unique(regions["cable_side"])]
    assert np.all((np.abs(cable_side[:, 0] - 0.006) < 1e-9) | (np.abs(cable_side[:, 0] - 0.003) < 1e-9)
                  | (cable_side[:, 2] > 0.0135 - 1e-9))  # +x face, slot floors, flanks
    assert (np.abs(cable_side[:, 0] - 0.003) < 1e-9).any()  # the slot floors are part of it
    # the tendon runs through solid (the web) along its whole length
    via = np.array([[g.tendon_x, 0.0, z] for z in np.linspace(0, 0.06, 41)])
    embed(mesh.points, mesh.tets, via)


def test_tendon_path_from_geometry():
    scene = _finger(["body.mesh.size=2.5mm"]).build_scene()
    (cable,) = scene.actuators
    via = np.array(cable.via)
    assert cable.anchor == "top" and via[0] == pytest.approx([0.0015, 0, 0]) and via[-1] == pytest.approx([0.0015, 0, 0.06])
    assert np.all(via[:, 0] == pytest.approx(0.0015)) and np.all(np.diff(via[:, 2]) > 0)
    with pytest.raises(ConfigError, match="geometry"):
        _finger(["body.geometry={type: box, size: [10 mm, 10 mm, 60 mm]}", "actuators[0].anchor=null"]).build_scene()


def _tip_rotation(overrides):
    """Quasi-static reference at 0.2 N: the tip face's rotation [rad] and its x displacement [m]."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    backend = FemBackend()
    backend.build(_finger(overrides).build_scene())
    s = Solid3DSolver(backend._problem, dynamic=False)
    s.tensions[:] = [0.2]
    assert s.advance(1.0)
    x = s.p.mesh.nodes + s.y[:3 * s.n].reshape(-1, 3)
    top = np.abs(s.p.mesh.nodes[:, 2] - 0.06) < 1e-9
    return np.arctan(-np.polyfit(x[top, 0], x[top, 2], 1)[0]), x[top, 0].mean() - s.p.mesh.nodes[top, 0].mean()


def test_plain_finger_is_euler_bernoulli_and_slots_add_compliance():
    """One segment (a plain 12 x 10 x 60 mm beam, E = 114 kPa, tendon 1.5 mm off the axis, spread over the tip) is
    Euler-Bernoulli's theta = T d L / (E_t I) within 2% (measured -0.9% at 2.5 mm, -1.4% at 1.5 mm). The
    three slots make the finger 2.5 times as compliant (2.54 at 2.5 mm, 2.53 at 1.5 mm). A beam model with plane
    sections through each slot predicts only 1.55: the slots are short for their depth, and the material next
    to them takes part, so the 1-D estimate is a lower bound here, not a prediction."""
    from softsim.analytic import cable_beam_tip_angle

    plain, _ = _tip_rotation(["body.mesh.size=2.5mm", "body.geometry.segments=1"])
    notched, tip_x = _tip_rotation(["body.mesh.size=2.5mm"])
    theory = cable_beam_tip_angle(0.2, 0.0015, 0.06, [114e3], [-0.006, 0.006], 0.01)
    assert plain == pytest.approx(theory, rel=0.02)
    assert notched / plain == pytest.approx(2.54, abs=0.15)
    assert notched / plain > 1.55 and tip_x > 0.005  # curls towards the cable side
    # a deeper slot is more compliant, a narrower web being weaker
    deeper, _ = _tip_rotation(["body.mesh.size=2.5mm", "body.geometry.slot_depth=3.5mm"])
    assert deeper > notched


@pytest.mark.slow
def test_finger_bending_is_mesh_converged():
    """Tip rotation of the notched finger at 1.5 mm is within 1% of the 2.5 mm mesh's (0.2664 vs 0.2677)."""
    coarse, _ = _tip_rotation(["body.mesh.size=2.5mm"])
    fine, _ = _tip_rotation(["body.mesh.size=1.5mm"])
    assert fine == pytest.approx(coarse, rel=0.01)


def test_finger_bends_towards_the_tendon_on_mujoco():
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco import MujocoBackend

    sc = _finger(["body.mesh.size=2.5mm"])
    backend = MujocoBackend()
    backend.build(sc.build_scene())
    backend.set_control("cable", 0.1)
    backend.step(1500)
    s = sc.metrics(backend.state())
    assert s["tip_x"] > 0.001 and s["bend_angle"] > 0.02 and s["cable_length"] < 0.06


def test_cad_export_reads_back_as_the_same_body(tmp_path):
    """The finger written as CAD (metres, body frame) and meshed from the file has the parametric body's
    volume and bounding box (the slots are in the CAD)."""
    from softsim.geometry import SoftMeshFile

    g = SoftCableFinger(**DIMS)
    g.export(tmp_path / "finger.brep")
    mesh = SoftMeshFile(path=str(tmp_path / "finger.brep")).mesh(MeshSettings(size=0.002, method="gmsh"))
    x = mesh.points[mesh.tets]
    assert np.abs(np.linalg.det(x[:, 1:] - x[:, :1])).sum() / 6 == pytest.approx(g.volume(), rel=1e-6)
    np.testing.assert_allclose(mesh.points.max(axis=0) - mesh.points.min(axis=0), [0.012, 0.01, 0.06], atol=1e-9)

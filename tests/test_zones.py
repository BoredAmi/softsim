"""Material zones (multi-material bodies): validation, the reference against composite Euler-Bernoulli,
per-zone materials on MuJoCo and SOFA."""

from pathlib import Path

import numpy as np
import pytest

from softsim.analytic import cable_beam_tip_angle, composite_section
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError, UnsupportedFeatureError
from softsim.scenarios.inflation import InflationConfig, InflationScenario

LAYERED = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_cantilever_layered.yaml"
STVK = ["body.material.elastic.model=stvk", "body.zones[0].material.elastic.model=stvk"]


def _beam(overrides=()):
    raw = apply_overrides(load_yaml(LAYERED), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), LAYERED.parent)


def test_composite_section():
    x_na, ei, ea = composite_section([1.0], [-0.5, 0.5], 1.0)
    assert x_na == 0 and ei == pytest.approx(1 / 12) and ea == 1.0
    x_na, ei, _ = composite_section([10.0, 1.0], [-5, -3, 5], 10)  # mm units: the docs' layered beam
    assert x_na == pytest.approx(-2.5714, abs=1e-4) and ei == pytest.approx(1922.0, rel=1e-3)
    # a zone with the body's modulus is the homogeneous beam
    assert cable_beam_tip_angle(0.7, 0.003, 0.055, [5e5, 5e5], [-0.005, -0.003, 0.005], 0.01) == pytest.approx(
        cable_beam_tip_angle(0.7, 0.003, 0.055, [5e5], [-0.005, 0.005], 0.01))


def test_zone_validation():
    body = _beam().build_scene().soft_bodies[0]
    (zone,) = body.zones
    assert len(zone.elements) == len(body.mesh.tets) // 5  # one of five 2 mm columns across x
    groups = body.material_groups()
    assert sum(len(e) for _, e in groups) == len(body.mesh.tets) and groups[1][0].elastic.youngs_modulus == 5e6
    with pytest.raises(ConfigError, match="density"):
        _beam(["body.zones[0].material.density=2000 kg/m^3"]).build_scene()
    with pytest.raises(ConfigError, match="no elements"):
        _beam(["body.zones[0].box=[[20 mm, 0, 0], [30 mm, 1 mm, 1 mm]]"]).build_scene()


def test_reference_matches_composite_beam():
    """Quasi-static reference, 2 mm: within 1% of composite Euler-Bernoulli at small load; at 0.7 N
    2.4% stiffer (Neo-Hookean nonlinearity: the stiff layer in compression, the soft part in tension;
    the gap halves on a beam twice as long at the same angle)."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    sc = _beam()
    backend = FemBackend()
    backend.build(sc.build_scene())
    s = Solid3DSolver(backend._problem, dynamic=False)
    backend.solver = s
    for tension, rel in ((0.1, 0.01), (0.7, 0.03)):
        s.tensions[:] = [tension]
        assert s.advance(1.0)
        theory = cable_beam_tip_angle(tension, 0.003, 0.055, [5e6, 5e5], [-0.005, -0.003, 0.005], 0.01)
        assert sc.metrics(backend.state())["bend_angle"] == pytest.approx(theory, rel=rel)


def _final_angle(backend_cls, overrides, steps):
    sc = _beam(["body.mesh.size=2.5mm", "body.zones[0].box=[[-5 mm, -5 mm, 0], [-2.5 mm, 5 mm, 60 mm]]",
                *overrides])
    backend = backend_cls()
    backend.build(sc.build_scene())
    backend.set_control("cable", 0.2)
    backend.step(steps)
    return sc.metrics(backend.state())


def test_mujoco_zones():
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco import MujocoBackend

    with pytest.raises(UnsupportedFeatureError, match="needs the body's elastic model"):
        MujocoBackend.check(_beam(["body.zones[0].material.elastic.model=stvk"]).build_scene())
    for law in ([], STVK):
        plain = _final_angle(MujocoBackend, [*law, "body.zones[0].material.elastic.youngs_modulus=500 kPa"], 300)
        no_zone = _final_angle(MujocoBackend, [*law, "body.zones=[]"], 300)
        np.testing.assert_allclose(plain["bend_angle"], no_zone["bend_angle"], rtol=1e-9)  # identity zone
        assert _final_angle(MujocoBackend, law, 300)["bend_angle"] < 0.9 * no_zone["bend_angle"]
    # the zone's rows are those of a model compiled with the zone's material, Poisson's ratio included
    sc = _beam(["body.mesh.size=2.5mm", "body.zones[0].material.elastic.poisson_ratio=0.3"])
    zoned, full = MujocoBackend(), MujocoBackend()
    zoned.build(sc.build_scene())
    full.build(_beam(["body.mesh.size=2.5mm", "body.zones=[]", "body.material.elastic.youngs_modulus=5 MPa",
                      "body.material.elastic.poisson_ratio=0.3"]).build_scene())
    elements = sc.build_scene().soft_bodies[0].zones[0].elements
    np.testing.assert_array_equal(zoned._stiffness_rows(zoned.model, "beam")[elements],
                                  full._stiffness_rows(full.model, "beam")[elements])


def test_sofa_zones():
    from softsim.backends.sofa import SofaBackend
    from softsim.backends.sofa.backend import import_sofa
    from softsim.errors import SoftSimError

    try:
        import_sofa()
    except SoftSimError as exc:
        pytest.skip(str(exc))

    ov = ["simulation.timestep=1ms"]
    plain = _final_angle(SofaBackend, [*ov, "body.zones[0].material.elastic.youngs_modulus=500 kPa"], 30)
    no_zone = _final_angle(SofaBackend, [*ov, "body.zones=[]"], 30)
    np.testing.assert_allclose(plain["bend_angle"], no_zone["bend_angle"], rtol=1e-6)
    assert _final_angle(SofaBackend, ov, 30)["bend_angle"] < 0.9 * no_zone["bend_angle"]

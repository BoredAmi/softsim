"""Cable actuation: embedding, the follower load, the reference against Euler-Bernoulli, backends."""

from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import CableActuator, cable_forces, cable_length, embed, resample
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftBox
from softsim.scenarios.inflation import InflationConfig, InflationScenario

BEAM = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_cantilever.yaml"


def _beam(overrides=()):
    raw = apply_overrides(load_yaml(BEAM), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), BEAM.parent)


def test_cable_forces_are_the_length_gradient():
    mesh = SoftBox(size=(0.01, 0.01, 0.03)).mesh(MeshSettings(size=0.0025))
    via = resample(np.array([[0.003, 0.0, 0.0], [0.003, 0.0, 0.02], [-0.002, 0.002, 0.029]]), 0.004)
    assert len(via) == 9 and np.allclose(via[[0, 5, 8]], [[0.003, 0, 0], [0.003, 0, 0.02], [-0.002, 0.002, 0.029]])
    elems, bary = embed(mesh.points, mesh.tets, via)
    nodes = mesh.tets[elems]
    np.testing.assert_allclose(np.einsum("kj,kji->ki", bary, mesh.points[nodes]), via, atol=1e-12)
    x = mesh.points + np.random.default_rng(1).normal(size=mesh.points.shape) * 3e-4  # deformed
    f = cable_forces(x, nodes, bary, 0.5)
    dx = np.random.default_rng(2).normal(size=x.shape) * 1e-8
    dl = cable_length(x + dx, nodes, bary) - cable_length(x - dx, nodes, bary)
    assert float(np.sum(f * 2 * dx)) == pytest.approx(-0.5 * dl, rel=1e-5)
    # internal: no net force or moment on the body
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(x, f).sum(axis=0), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="outside the body"):
        embed(mesh.points, mesh.tets, np.array([[0.006, 0.0, 0.01]]))


def test_scene_and_control_validation():
    with pytest.raises(ConfigError, match="outside the body"):
        _beam(["actuators[0].path=[[3 mm, 0, 0], [3 mm, 0, 70 mm]]"]).build_scene()
    with pytest.raises(Exception, match="greater than or equal to 0"):
        _beam(["actuators[0].tension=-1"])
    scene = _beam().build_scene()
    (cable,) = scene.actuators
    assert isinstance(cable, CableActuator) and len(cable.via) == 29  # 55 mm at the 2 mm mesh size
    assert cable.value(0.2) == pytest.approx(0.35) and cable.value(1.0) == 0.7


def test_mujoco_bends_towards_the_cable():
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco import MujocoBackend

    sc = _beam(["body.mesh.size=2.5mm"])
    backend = MujocoBackend()
    backend.build(sc.build_scene())
    backend.set_control("cable", 0.2)
    backend.step(1500)
    s = sc.metrics(backend.state())
    assert s["cable_tension"] == 0.2 and s["tip_x"] > 0 and s["bend_angle"] > 0.02
    assert s["cable_length"] < 0.06
    with pytest.raises(ValueError, match="can only pull"):
        backend.set_control("cable", -0.1)


def test_reference_matches_euler_bernoulli():
    """Quasi-static reference, 2.5 mm quadratic tets: a cable at d = 3 mm anchored at L = 55 mm puts a
    pure moment T d on the clamped beam below the anchor, theta = T d L / (E_t I) (Euler-Bernoulli,
    exact at any rotation). The cable also compresses the beam by T / (E A), which stiffens
    Neo-Hookean: E_t = E (1 + 2 T / (E A)), 3% at 0.7 N. With it, the reference is within 0.3%."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    backend = FemBackend()
    backend.build(_beam(["body.mesh.size=2.5mm"]).build_scene())
    s = Solid3DSolver(backend._problem, dynamic=False)
    e, b, length, d = 500e3, 0.01, 0.055, 0.003
    for tension in (0.1, 0.7):
        s.tensions[:] = [tension]
        assert s.advance(1.0)
        x = s.p.mesh.nodes + s.y[:3 * s.n].reshape(-1, 3)
        top = np.abs(s.p.mesh.nodes[:, 2] - 0.06) < 1e-9
        slope = np.polyfit(x[top, 0], x[top, 2], 1)[0]  # the top face stays plane
        theta = np.arctan(-slope)
        e_t = e * (1 + 2 * tension / (e * b * b))
        assert theta == pytest.approx(tension * d * length / (e_t * b**4 / 12), rel=0.01)


def test_explicit_cable_stability_check():
    from softsim.checks.stability import check_scene

    def codes(overrides, implicit=False):
        scene = _beam(overrides).build_scene()
        return {f.code for f in check_scene(scene, implicit_actuation=implicit)} & {"cable-stiff", "cable-unstable"}

    assert codes([]) == set()
    assert codes(["actuators[0].spacing=0.25mm"]) == {"cable-unstable"}
    assert codes(["actuators[0].spacing=0.25mm"], implicit=True) == set()  # the reference: in the implicit solve

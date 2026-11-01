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


STROKE = BEAM.parent / "cable_cantilever_stroke.yaml"


def _stroke_beam(overrides=()):
    raw = apply_overrides(load_yaml(STROKE), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), STROKE.parent)


def test_stroke_control_config_and_check():
    from softsim.checks.stability import check_scene

    (cable,) = _stroke_beam().build_scene().actuators
    assert cable.stroke_controlled and cable.stiffness == 300 and cable.value(0.2) == pytest.approx(0.0015)
    assert cable.tension_at(0.050, 0.002, 0.055) == 0.0  # 3 mm shorter than its free length: slack
    assert cable.tension_at(0.0545, 0.002, 0.055) == pytest.approx(300 * 0.0015)
    with pytest.raises(Exception, match="needs the tendon's stiffness"):
        _beam(["actuators[0].control=stroke", "actuators[0].schedule=[]"])
    with pytest.raises(Exception, match="is a force, expected a length"):
        _stroke_beam(["actuators[0].schedule=[[0, 0.5 N]]"])

    def codes(overrides, implicit=False):
        scene = _stroke_beam(overrides).build_scene()
        return {f.code for f in check_scene(scene, implicit_actuation=implicit)} & {"cable-stiff", "cable-unstable"}

    assert codes([]) == set()
    assert codes(["actuators[0].stiffness=3000 N/m"]) == {"cable-unstable"}  # the taut tendon's axial term
    assert codes(["actuators[0].stiffness=3000 N/m"], implicit=True) == set()


def test_reference_stroke_matches_euler_bernoulli():
    """Quasi-static reference, stroke-controlled, anchor in a stiff tip cap: the tension and tip angle
    follow the beam's compliance along the cable (bending and axial) in series with the tendon
    (measured: within 0.8%; a point anchor in the soft body instead gives way locally, -10%); a stiff
    tendon (1e5 N/m, implicit only) sets the shortening; and the stroke run's equilibrium is the
    tension run's at the tension it found."""
    pytest.importorskip("jax")
    from softsim.analytic import cable_beam_stroke
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    def solver(overrides):
        backend = FemBackend()
        backend.build(_stroke_beam(["body.mesh.size=2.5mm", *overrides]).build_scene())
        return Solid3DSolver(backend._problem, dynamic=False)

    def angle(s):
        x = s.p.mesh.nodes + s.y[:3 * s.n].reshape(-1, 3)
        top = np.abs(s.p.mesh.nodes[:, 2] - 0.06) < 1e-9
        return np.arctan(-np.polyfit(x[top, 0], x[top, 2], 1)[0])

    s = solver([])
    for stroke in (0.0003, 0.003):
        s.free_lengths[:] = s.rest_lengths - stroke
        assert s.advance(1.0)
        tension, theta = cable_beam_stroke(stroke, 300.0, 0.003, 0.055, [5e5], [-0.005, 0.005], 0.01)
        assert s.cable_tension(0) == pytest.approx(tension, rel=0.01)
        assert angle(s) == pytest.approx(theta, rel=0.01)
    # the same equilibrium under the tension it found
    t = solver(["actuators[0].control=tension", "actuators[0].schedule=[]"])
    t.tensions[:] = [s.cable_tension(0)]
    assert t.advance(1.0)
    np.testing.assert_allclose(t.y, s.y, atol=1e-9)
    # a stiff tendon: displacement control, the cable shortens by the stroke
    stiff = solver(["actuators[0].stiffness=1e5 N/m"])
    stiff.free_lengths[:] = stiff.rest_lengths - 0.001
    assert stiff.advance(1.0)
    assert stiff.rest_lengths[0] - stiff.cable_length(0) == pytest.approx(0.001, rel=0.01)


def test_mujoco_stroke_control():
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco import MujocoBackend

    sc = _stroke_beam(["body.mesh.size=2.5mm"])
    backend = MujocoBackend()
    backend.build(sc.build_scene())
    rest = sc.metrics(backend.state())["cable_length"]
    backend.set_control("cable", 0.002)
    backend.step(3000)
    s = sc.metrics(backend.state())
    assert s["cable_command"] == 0.002 and s["bend_angle"] > 0.05
    assert s["cable_tension"] == pytest.approx(300 * (s["cable_length"] - rest + 0.002), rel=1e-9)
    backend.set_control("cable", -0.001)  # paid out: slack, the (undamped) beam swings back through straight
    angles = []
    for _ in range(40):
        backend.step(50)
        s = sc.metrics(backend.state())
        assert s["cable_tension"] == pytest.approx(300 * max(s["cable_length"] - rest - 0.001, 0.0), abs=1e-12)
        angles.append(s["bend_angle"])
    assert min(angles) < 0.01


def test_closed_loop_controller_holds_a_tip_angle():
    """Simulation.run(controller=...): a PI loop on the tension holds the (lightly damped) beam at 0.1 rad
    (MuJoCo)."""
    pytest.importorskip("mujoco")
    from softsim.runner import Simulation

    sim = Simulation(BEAM, ["body.mesh.size=2.5mm", "simulation.duration=600ms", "output.save_model=false",
                            "body.material.damping={model: rayleigh, alpha: 0, beta: 2 ms}"])  # ~10% on mode 1
    state = {"integral": 0.0}

    def controller(s):
        error = 0.1 - s.metrics()["bend_angle"]
        state["integral"] += error * 1e-3
        return {"cable": max(0.0, 0.5 * error + 40.0 * state["integral"])}

    ts = sim.run(controller=controller).timeseries
    late = ts[ts.t > 0.45]
    assert late.bend_angle.mean() == pytest.approx(0.1, rel=0.03) and late.bend_angle.std() < 0.005
    # the tension it found is the one that bends this (locked) beam by 0.1 rad: above Euler-Bernoulli's
    assert late.cable_tension.mean() > 0.1 * (500e3 * 0.01**4 / 12) / (0.003 * 0.055)
    sim.apply_controls({"cable": float("nan")})  # back to the schedule
    assert sim.backend.control("cable", 0.2) == pytest.approx(0.35)

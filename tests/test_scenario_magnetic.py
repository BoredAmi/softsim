"""A magneto-active body and an applied field from a scenario config (softsim.magnetism, MagnetoActive), through the FE
backend: the material card's susceptibilities and axes, the field's schedule and magnets, the closed form of a bar
bending towards buckling, and the backends that must refuse it."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from softsim.backends import get_backend
from softsim.backends.fem.backend import FemBackend
from softsim.config.loader import load_yaml
from softsim.errors import ConfigError, UnsupportedFeatureError
from softsim.fem.magnetic import MU0, prism_demagnetizing_factor
from softsim.fem.solid3d import Solid3DSolver
from softsim.magnetism import MagneticFieldConfig, field_parameters, ring
from softsim.runner import Simulation
from softsim.config.schema import validate

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "magnetic" / "mre_strip_buckling.yaml"
LENGTH, SIDE, E_MOD = 0.06, 0.004, 2e5


def build(overrides=(), raw=None):
    sim = Simulation(raw or load_yaml(SCENARIO), list(overrides), base_dir=SCENARIO.parent)
    scene = sim.scenario.build_scene()
    fem = FemBackend()
    fem.build(scene)
    return scene, fem


def test_the_card_gives_each_axis_its_blocks_susceptibility():
    """chi / (1 + N chi) along the bar's length (z) and across it (x, y), N of the 4 x 4 x 60 mm block; the axes are the
    card's, in the world."""
    _, fem = build()
    m = fem._problem.magnetic
    n_along, n_across = prism_demagnetizing_factor(SIDE, SIDE, LENGTH), prism_demagnetizing_factor(LENGTH, SIDE, SIDE)
    assert m.chi_along == pytest.approx(1 / (1 + n_along), rel=1e-12)
    assert m.chi_normal == pytest.approx(1 / (1 + n_across), rel=1e-12)
    assert m.chi_across == pytest.approx(1 / (1 + n_across), rel=1e-12)
    np.testing.assert_allclose(m.along, (0, 0, 1))
    np.testing.assert_allclose(m.normal, (1, 0, 0))
    assert m.saturation == 0.0  # the card sets none: linear


def test_the_cards_saturation_reaches_the_reference():
    _, fem = build(["body.material.elastic.saturation_magnetization=4e5"])
    assert fem._problem.magnetic.saturation == 4e5


def test_a_bar_from_the_config_bends_by_the_closed_form():
    """The scenario's end-of-ramp field, solved statically on the backend's own problem: the tip moves
    e (sin(lL) / l + tan(lL) (1 - cos(lL)) / l - L) along the field, l^2 = mu0 A (chi_a - chi_n) H^2 / E I (as in
    tests/test_magnetic.py, there built by hand). The config's field is 70% of buckling, tilted 0.2 deg."""
    scene, fem = build(["body.mesh.size=1mm"])  # 2 mm: 11% over the closed form, 1 mm: 2.2%
    m = fem._problem.magnetic
    q = scene.magnetic_field.parameters(2.0)
    h, tilt = np.hypot(q[0], q[2]), np.arctan2(q[2], q[0])
    assert tilt == pytest.approx(np.radians(0.2), rel=1e-3)
    k = MU0 * SIDE * SIDE * (m.chi_along - m.chi_normal) * h * h
    l = np.sqrt(k / (E_MOD * SIDE ** 4 / 12))
    assert l * LENGTH == pytest.approx(0.7 * np.pi / 2, rel=1e-3)  # the config is 70% of buckling, as its comment says
    want = tilt * (np.sin(l * LENGTH) / l + np.tan(l * LENGTH) * (1 - np.cos(l * LENGTH)) / l - LENGTH)
    s = Solid3DSolver(fem._problem, dynamic=False)
    s.magnetic_field = q
    assert s.advance(1.0)
    nodes = s.p.mesh.nodes
    tip = np.flatnonzero(np.isclose(nodes[:, 2], nodes[:, 2].max()))
    assert s.y[:3 * s.n].reshape(-1, 3)[tip, 0].mean() == pytest.approx(want, rel=0.05)


def test_each_step_applies_the_field_at_its_end():
    """Like every other load in the backend: after a step to t, the solver holds the field of t."""
    scene, fem = build(["simulation.timestep=50ms"])
    for _ in range(3):
        fem.step()
        np.testing.assert_allclose(fem.solver.magnetic_field, scene.magnetic_field.parameters(fem.time), rtol=1e-12)
    assert fem.solver.magnetic_field[0] == pytest.approx(15504 * 0.15 / 2.0, rel=1e-9)  # linear: 0.15 s of the 2 s ramp


def test_a_magnet_moves_along_its_path():
    """A ring magnet's parameters: the disc and its hole reversed, centred on its path (minimum jerk, as a moving mount)."""
    cfg = validate(MagneticFieldConfig, {"magnets": [{
        "name": "ring", "remanence": "1.23 T", "diameter": "20 mm", "hole": "4.2 mm", "thickness": "5 mm",
        "position": ["0 mm", "0 mm", "-12 mm"], "axis": [0, 2, 0], "path": [["0 s", [0, 0, 0]], ["0.2 s", [0, 0, "12 mm"]]]}]})
    f = cfg.build()
    for t, z in ((0.0, -0.012), (0.1, -0.006), (0.3, 0.0)):
        want = field_parameters(cylinders=ring((0, 0, z), (0, 1, 0), 1.23, 0.01, 0.0021, 0.005, 2e-4))
        np.testing.assert_allclose(f.parameters(t), want, rtol=1e-12, atol=1e-15)
    assert f.parameters(0.0)[15] == pytest.approx(1.23 / MU0)  # the outer disc's magnetisation [A/m]
    assert f.parameters(0.0)[25] == pytest.approx(-1.23 / MU0)  # the hole's


@pytest.mark.parametrize("bad, words", [
    ({"magnets": [{"remanence": "1 T", "diameter": "10 mm", "hole": "10 mm", "thickness": "2 mm"}]}, "hole"),
    ({"magnets": [{"remanence": "1 T", "diameter": "10 mm", "thickness": "2 mm"}] * 2}, "unique"),
    ({"uniform": [["1 s", [0, 0, 1]], ["0.5 s", [0, 0, 2]]]}, "increasing"),
    ({"magnets": [{"remanence": "1 T", "diameter": "10 mm", "thickness": "2 mm", "axis": [0, 0, 0]}]}, "non-zero"),
    ({"uniform": ["20 kPa", 0, 0]}, "is a pressure, expected a field_strength"),
])
def test_a_wrong_field_config_is_an_error(bad, words):
    with pytest.raises(ConfigError, match=words):
        validate(MagneticFieldConfig, bad)


def test_the_material_needs_one_source_of_susceptibility():
    raw = load_yaml(SCENARIO)
    raw["body"]["material"]["elastic"]["particle_volume_fraction"] = 0.12
    with pytest.raises(ConfigError, match="exactly one"):
        Simulation(raw, [], base_dir=SCENARIO.parent)


@pytest.mark.parametrize("backend", ["mujoco", "sofa"])
def test_backends_without_magnetism_refuse(backend):
    """The field and the material both need the coupling; a backend without it refuses instead of running a passive bar."""
    scene, _ = build()
    with pytest.raises(UnsupportedFeatureError, match="magneto-active"):
        get_backend(backend).check(scene)


def test_the_reference_refuses_a_field_on_a_moving_mount():
    raw = load_yaml(SCENARIO)
    raw["body"]["mount_path"] = [["0 s", [0, 0, 0]], ["1 s", ["1 mm", 0, 0]]]
    with pytest.raises(UnsupportedFeatureError, match="moving or rotating mount"):
        Simulation(raw, [], base_dir=SCENARIO.parent)  # refused as the run is set up


def test_a_turned_body_in_a_turned_field_has_the_same_energy():
    """The material's axes are the body's: turn the bar (pose) and the field by the same rotation and its magnetic energy,
    in any deformation turned the same way, is unchanged; the problem's axes are the card's turned."""
    from dataclasses import replace

    from softsim.fem.solid3d import quat_to_matrix
    from softsim.scene import Pose

    scene, fem = build()
    pose = Pose.from_tilt((0.0, 0.0, 0.0), tilt_x=0.4, tilt_y=-1.1)
    rot = quat_to_matrix(pose.quaternion)
    turned_scene = replace(scene, soft_bodies=(replace(scene.soft_bodies[0], pose=pose),))
    turned = FemBackend()
    turned.build(turned_scene)
    m = turned._problem.magnetic
    np.testing.assert_allclose(m.along, rot @ (0, 0, 1), atol=1e-12)
    np.testing.assert_allclose(m.normal, rot @ (1, 0, 0), atol=1e-12)
    q = scene.magnetic_field.parameters(2.0)
    rng = np.random.default_rng(3)
    u = 1e-3 * rng.standard_normal((fem.solver.n, 3))  # a deformation of the bar, and the same one turned
    s, st = Solid3DSolver(fem._problem, dynamic=False), Solid3DSolver(turned._problem, dynamic=False)
    s.magnetic_field = q
    st.magnetic_field = np.concatenate([rot @ q[:3], q[3:]])
    pad = np.zeros(s.ndof - 3 * s.n)
    e, et = s.magnetic_energy(np.concatenate([u.ravel(), pad])), st.magnetic_energy(np.concatenate([(u @ rot.T).ravel(), pad]))
    assert e != 0 and et == pytest.approx(e, rel=1e-10)


def test_a_solid_magnet_is_a_cylinder_the_body_touches():
    """``solid: true`` adds the magnet's outer cylinder as a rigid body of the reference: its axis the magnet's (a cylinder
    part's axis is its body's z), centred on the magnet, fixed or moving along the magnet's path; the strip's friction."""
    from softsim.fem.solid3d import quat_to_matrix

    raw = load_yaml(SCENARIO)
    magnet = {"name": "ring", "remanence": "1.23 T", "diameter": "20 mm", "hole": "4.2 mm", "thickness": "5 mm",
              "position": ["14 mm", 0, "40 mm"], "axis": [0.3, 1, 0], "solid": True}
    raw["magnetic_field"] = {"magnets": [magnet, {**magnet, "name": "moving", "position": ["-14 mm", 0, "40 mm"],
                                                  "path": [["0 s", [0, 0, 0]], ["0.1 s", [0, 0, "5 mm"]]]}]}
    scene, fem = build(raw=raw)
    fixed, moving = fem._problem.rigid
    axis = np.array([0.3, 1, 0]) / np.hypot(0.3, 1)
    for body, at in ((fixed, (0.014, 0, 0.04)), (moving, (-0.014, 0, 0.04))):
        np.testing.assert_allclose(quat_to_matrix(body.quaternion) @ (0, 0, 1), axis, atol=1e-12)
        np.testing.assert_allclose(body.position, at, atol=1e-15)
        assert body.parts[0].kind == "cylinder" and body.parts[0].size[:2] == pytest.approx((0.01, 0.0025))
    assert fixed.mount == "fixed" and moving.mount == "prescribed"
    # its edges rounded by the magnet's chamfer (0.2 mm), not the 0.5 mm of other rigid bodies: that let a strip into the
    # magnet's corner, where its field is strongest (docs/magnetic.md)
    assert fixed.edge_radius == pytest.approx(2e-4) and moving.edge_radius == pytest.approx(2e-4)
    np.testing.assert_allclose(moving.motion(0.1), (0, 0, 0.005), atol=1e-15)
    assert fem._names[-2:] == ["ring", "moving"]


@pytest.mark.slow
def test_a_strip_pulled_onto_a_solid_magnet_stays_outside_it():
    """A short strip released 0.5 mm from a weak solid ring magnet's rim (0.05 T: a full-strength one accelerates it at
    ~1e4 m/s^2 and it lands at metres per second, an impact this test is not about) is pulled onto the rim and lies on
    it, not in it: within the contact's penetration (pressure / stiffness) of the cylinder, with a contact force."""
    raw = load_yaml(SCENARIO)
    raw["body"]["geometry"]["size"] = ["2 mm", "6 mm", "20 mm"]
    raw["body"]["material"]["elastic"]["matrix"]["youngs_modulus"] = "800 kPa"
    raw["magnetic_field"] = {"magnets": [{"name": "ring", "remanence": "0.05 T", "diameter": "20 mm", "hole": "4.2 mm",
                                          "thickness": "5 mm", "position": ["11.5 mm", 0, "16 mm"], "axis": [0, 1, 0],
                                          "solid": True}]}
    scene, fem = build(["simulation.timestep=2ms", "body.mesh.size=1mm"], raw=raw)
    centre, radius = np.array([0.0115, 0, 0.016]), 0.01
    for _ in range(20):  # it lands after about 8 ms
        fem.step()
    state = fem.state()
    x = state.soft["strip"].vertices
    rho = np.hypot(x[:, 0] - centre[0], x[:, 2] - centre[2])
    beside = np.abs(x[:, 1]) <= 0.0025 - 5e-4  # clear of the contact's rounded edges (0.5 mm)
    assert rho[beside].min() > radius - 2e-6  # not in it (the penalty lets a node in by microns)
    assert rho[beside].min() < radius + 2e-5  # on it: a flat strip touches the rim along a line, its nearest nodes up to
    #                                           half a node spacing off it, sqrt(R^2 + 0.5 mm^2) - R = 12.5 um
    assert any(c.body_b == "ring" and np.linalg.norm(c.force) > 0 for c in state.contacts)


def test_prony_limits_only_the_backends_that_add_it_explicitly():
    """MuJoCo and SOFA add the Maxwell overstress as forces each step, so its stiffness bounds the step; the reference has it
    in the energy Newton minimises (no bound). A stiff branch on the 1 mm strip at 0.25 ms: an error for the first only."""
    from softsim.backends.mujoco.backend import MujocoBackend
    from softsim.checks.stability import check_scene

    raw = load_yaml(SCENARIO)
    raw["body"]["material"]["damping"] = {"model": "prony", "g": [0.3], "tau": ["2 ms"]}
    raw["body"]["mesh"]["size"] = "1 mm"
    raw["simulation"]["timestep"] = "0.25 ms"
    scene = Simulation(raw, [], base_dir=SCENARIO.parent).scenario.build_scene()
    codes = {cls.name: {f.code for f in check_scene(scene, implicit_prony=cls.implicit_prony)}
             for cls in (FemBackend, MujocoBackend)}
    assert "prony-unstable" in codes["mujoco"] and "prony-unstable" not in codes["fem"]


def test_the_cards_remanence_reaches_the_reference_in_the_world_frame():
    """The card's remanence is in the body frame; the bar's body frame is the world's here, and a purely hard-magnetic
    card has susceptibility 0."""
    _, fem = build(["body.material.elastic.remanent_magnetization=[0, 0, 9e4]", "body.material.elastic.susceptibility=0"])
    m = fem._problem.magnetic
    np.testing.assert_allclose(m.remanence, (0, 0, 9e4))
    assert m.chi_along == m.chi_normal == 0.0


GRIPPER = SCENARIO.with_name("hard_magnetic_gripper.yaml")


def gripper(overrides=()):
    sim = Simulation(load_yaml(GRIPPER), list(overrides), base_dir=GRIPPER.parent)
    fem = FemBackend()
    fem.build(sim.scenario.build_scene())
    return fem


def test_each_zones_card_reaches_the_reference():
    """The gripper's arms are zones with remanence +z and -z (body frame, which is the world's), the palm has none: the
    problem holds one magnet per arm over exactly that zone's elements, and none for the rest."""
    fem = gripper(["body.mesh.size=2mm"])
    p = fem._problem
    assert p.magnetic is None and len(p.magnetic_zones) == 2
    centroids = p.mesh.nodes[p.mesh.tets[:, :4]].mean(axis=1)
    for (elements, magnet), side in zip(sorted(p.magnetic_zones, key=lambda g: centroids[g[0], 0].mean()), (-1, 1)):
        np.testing.assert_allclose(magnet.remanence, (0, 0, -side * 1e5))  # left (x < 0): +z, right: -z
        assert side * centroids[elements, 0].min() > 0.009 and centroids[elements, 2].min() > 0.0035
        assert len(elements) == int(np.sum((side * centroids[:, 0] > 0.009) & (centroids[:, 2] > 0.004)))
    assert fem.solver.magnetic_field is not None and fem._problem.is_magnetic


def test_the_arms_of_a_hard_magnetic_gripper_close_by_the_cantilever_closed_form():
    """Opposite arms in one uniform field along x, 100 A/m: each bends towards the other like a cantilever under the
    distributed couple c = mu0 M_r H A (c L^3 / 3EI = 1.005 mm at the arm's 40 mm). Checked as the arm's own bending:
    the cubic of the arm's centre line has u''' = c / EI (1.034 at a 1.5 mm mesh, 1.005 at 0.75 mm), whatever the palm
    does under it; and the tip, which also rides the palm's rotation, is the cantilever's plus that root slope times
    the arm. The field reversed opens them by the same amount."""
    fem = gripper(["body.mesh.size=1.5mm"])
    x = fem._problem.mesh.nodes
    e_mod, inertia, area, h = 2e5, 4e-3 * 4e-3 ** 3 / 12, 16e-6, 100.0
    couple = MU0 * 1e5 * h * area
    top = np.isclose(x[:, 2], x[:, 2].max())
    for sign in (1, -1):
        s = Solid3DSolver(fem._problem, dynamic=False)
        s.magnetic_field[:3] = (sign * h, 0, 0)
        assert s.advance(1.0)
        u = s.y[:3 * s.n].reshape(-1, 3)
        for side in (1, -1):  # side = +1: the left arm, which closes towards +x
            arm = (-side * x[:, 0] > 0.009) & (x[:, 2] > 0.010)
            fit = np.polyfit(x[arm, 2], side * u[arm, 0], 3)
            assert 6 * fit[0] == pytest.approx(-sign * couple / (e_mod * inertia), rel=0.05)
            tip = side * u[top & (-side * x[:, 0] > 0), 0].mean()
            cantilever = couple * 0.04 ** 3 / (3 * e_mod * inertia)
            assert tip == pytest.approx(sign * cantilever + np.polyval(np.polyder(fit), 0.004) * 0.04, rel=0.05)
            assert sign * tip > 0.9 * cantilever  # closes (sign 1) or opens (-1)


def test_a_soft_magnetic_zone_needs_its_demagnetising_factors():
    """A zone's field is shaped by its neighbours, not by its own bounding block: 'block' on a zone with induced
    magnetisation is refused (explicit factors, or a purely hard zone, are not)."""
    sim_raw = load_yaml(GRIPPER)
    for key, value in (("susceptibility", 1.0), ("demagnetization", "block")):
        sim_raw["body"]["zones"][0]["material"]["elastic"][key] = value
    with pytest.raises(UnsupportedFeatureError, match="demagnetization"):
        Simulation(sim_raw, [], base_dir=GRIPPER.parent)
    sim_raw["body"]["zones"][0]["material"]["elastic"]["demagnetization"] = [0.2, 0.4, 0.4]
    Simulation(sim_raw, [], base_dir=GRIPPER.parent)  # explicit factors: allowed

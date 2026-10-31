"""Pressure actuation: regions, follower loads, the reference against the exact thick-sphere solution."""

import math
from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import PressureActuator, enclosed_volume, open_edges, pressure_forces
from softsim.analytic import hollow_sphere_limit, hollow_sphere_pressure, hollow_sphere_linear
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError, UnsupportedFeatureError
from softsim.geometry import MeshSettings, SoftClosedTube, SoftCylinder, SoftHollowSphere
from softsim.scenarios.inflation import InflationConfig, InflationScenario

pytest.importorskip("gmsh")

SPHERE = Path(__file__).parents[1] / "scenarios" / "inflation" / "hollow_sphere.yaml"


def _sphere_scene(overrides=()):
    raw = apply_overrides(load_yaml(SPHERE), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), SPHERE.parent)


def test_regions_and_closed_surface_loads():
    shape = SoftHollowSphere(outer_diameter=0.04, inner_diameter=0.02)
    mesh = shape.mesh(MeshSettings(size=0.005, method="gmsh"))
    regions = shape.regions(mesh)
    cavity = regions["cavity"]
    assert len(open_edges(cavity)) == 0 and len(open_edges(regions["outer"])) == 0
    assert len(cavity) + len(regions["outer"]) == len(mesh.surface_triangles())
    # faceted, so slightly smaller than the sphere
    assert enclosed_volume(mesh.points, cavity) == pytest.approx(math.pi / 6 * 0.02**3, rel=0.08)
    f = pressure_forces(mesh.points, cavity, 1e3)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)  # closed: no net force
    c = np.array([0, 0, 0.02])
    assert np.all(np.einsum("ij,ij->i", f, mesh.points - c)[np.unique(cavity)] > 0)  # pushes outward
    # virtual work: dV/dx . f = p |dV/dx|^2 > 0 and -f is the gradient of -p V
    x = mesh.points.copy()
    dx = np.random.default_rng(0).normal(size=x.shape) * 1e-7
    dv = enclosed_volume(x + dx, cavity) - enclosed_volume(x - dx, cavity)
    assert float(np.sum(f * 2 * dx)) == pytest.approx(1e3 * dv, rel=1e-6)

    cyl = SoftCylinder(diameter=0.02, height=0.01)
    m = cyl.mesh(MeshSettings(size=0.004))
    r = cyl.regions(m)
    assert len(open_edges(r["top"])) > 0
    tube = SoftClosedTube(outer_diameter=0.02, inner_diameter=0.012, height=0.05, cap=0.004)
    tm = tube.mesh(MeshSettings(size=0.004, method="gmsh"))
    assert len(open_edges(tube.regions(tm)["cavity"])) == 0


def test_schedule():
    a = PressureActuator("p", "b", "cavity", schedule=((0.0, 0.0), (0.1, 2e3), (0.2, 2e3)))
    assert a.value(0.05) == pytest.approx(1e3) and a.value(1.0) == 2e3
    assert PressureActuator("p", "b", "cavity", pressure=5.0).value(3.0) == 5.0


def test_analytic_small_strain_limit():
    # finite-strain incompressible solution -> linear Lame (nu = 1/2) at small pressure
    mu, a, b = 1e4, 0.01, 0.02
    u = hollow_sphere_linear(3 * mu, 0.5, a, b, 0.1, a)  # strain 3e-6: nonlinear terms ~1e-5
    assert hollow_sphere_pressure(mu, a, b, 1 + u / a) == pytest.approx(0.1, rel=1e-4)
    p_max, la = hollow_sphere_limit(mu, a, b)
    assert p_max / mu == pytest.approx(0.8164, abs=1e-3) and la == pytest.approx(1.83, abs=0.01)


def test_scene_validation():
    with pytest.raises(ConfigError, match="no region 'lid'"):
        _sphere_scene(["actuators[0].region=lid"]).build_scene()
    with pytest.raises(ConfigError, match="no region 'base'"):
        _sphere_scene(["body.fixed=[base]"]).build_scene()


def test_reference_refuses_open_regions():
    from softsim.backends.fem import FemBackend

    raw = apply_overrides(load_yaml(SPHERE), ["body.geometry={type: cylinder, diameter: 20 mm, height: 10 mm}",
                                              "body.mesh.method=structured", "body.mesh.size=4 mm",
                                              "actuators[0].region=top"])
    scene = InflationScenario(validate(InflationConfig, raw), SPHERE.parent).build_scene()
    with pytest.raises(UnsupportedFeatureError, match="is open"):
        FemBackend.check(scene)


def test_mujoco_inflates_and_obeys_controls():
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco import MujocoBackend

    sc = _sphere_scene(["body.mesh.size=6 mm"])
    scene = sc.build_scene()
    backend = MujocoBackend()
    backend.build(scene)
    v0 = sc.metrics(backend.state())["cavity_volume"]
    backend.step(300)
    s = sc.metrics(backend.state())
    assert s["cavity_pressure"] == pytest.approx(600.0, rel=1e-6)  # 30 ms into the 3 kPa / 150 ms ramp
    assert s["cavity_volume"] > v0
    backend.set_control("cavity", 0.0)
    assert backend.control("cavity") == 0.0 and backend.state().controls == {"cavity": 0.0}
    backend.release_control("cavity")
    assert backend.control("cavity") == pytest.approx(600.0, rel=1e-6)
    com = backend.state().soft["sphere"].vertices.mean(axis=0)
    np.testing.assert_allclose(com, [0, 0, 0.02], atol=2e-4)  # closed cavity: no net force, no drift


def test_reference_matches_exact_thick_sphere():
    """Octant of a 20/40 mm sphere with symmetry planes (x . n = 0 there, so -p V is exact on the open
    cap), nearly incompressible Neo-Hookean: the inner stretch matches the exact solution within 0.3%
    up to 0.6 mu, and no static state exists above the limit pressure (0.82 mu)."""
    pytest.importorskip("jax")
    from softsim.fem.materials import NeoHookeanIso
    from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, Solid3DSolver
    from softsim.meshing import gmsh_hollow_sphere

    mu, a, b = 1e4, 0.01, 0.02
    lin = gmsh_hollow_sphere(a, b, 0.004, octant=True)
    cav = lin.surface_where(lambda p: np.linalg.norm(p, axis=1) < a + 1e-8)
    outer = lin.surface_where(lambda p: np.linalg.norm(p, axis=1) > b - 1e-8)
    m = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    fixed = np.concatenate([3 * np.nonzero(np.abs(m.nodes[:, k]) < 1e-9)[0] + k for k in range(3)])
    s = Solid3DSolver(Problem3D(m, NeoHookeanIso(mu), 1000 * mu, 1000.0, gravity=(0, 0, 0), fixed_dofs=fixed,
                                pressure_faces=(m.faces_of(cav), m.faces_of(outer))), dynamic=False)
    va, vb = s.enclosed_volume(0), -s.enclosed_volume(1)
    # flat faces: the cavity holds ~94% of the sphere at 4 mm
    assert va == pytest.approx(math.pi / 6 * a**3, rel=0.08) and vb == pytest.approx(math.pi / 6 * b**3, rel=0.02)
    for p in (0.2 * mu, 0.4 * mu, 0.6 * mu):
        s.pressures[:] = [p, 0.0]
        assert s.advance(1.0)
        la = (s.enclosed_volume(0) / va) ** (1 / 3)
        # effective radii from the faceted volumes
        assert hollow_sphere_pressure(mu, va ** (1 / 3), vb ** (1 / 3), la) == pytest.approx(p, rel=3e-3)
    assert la > 1.25
    s.pressures[:] = [0.9 * mu, 0.0]
    assert not s.solve_step(1.0, max_iter=25)


def test_transfer_recovers_the_identity(tmp_path):
    """Transfer MuJoCo against MuJoCo run with E = 30 kPa, starting from 20 kPa: recovers 30 kPa."""
    pytest.importorskip("mujoco")
    import yaml

    from softsim.transfer import Transfer

    spec = {"config": str(SPHERE), "backend": "mujoco",
            "overrides": ["body.mesh.size=7mm", "simulation.duration=0.04", "simulation.timestep=0.0002",
                          "actuators[0].schedule=[[0, 0], [0.03, 2000], [0.04, 2000]]", "output.save_model=false"],
            "reference": {"backend": "mujoco", "overrides": ["body.material.elastic.youngs_modulus=30 kPa"]},
            "observables": {"cavity_volume": 1.0},
            "parameters": {"body.material.elastic.youngs_modulus": {"initial": "20 kPa",
                                                                    "bounds": ["10 kPa", "60 kPa"]}},
            "checks": {"other_pressure": ["actuators[0].schedule=[[0, 0], [0.03, 1000], [0.04, 1000]]"]},
            "max_evaluations": 12, "output": str(tmp_path / "out")}
    path = tmp_path / "transfer.yaml"
    path.write_text(yaml.safe_dump(spec))
    report = Transfer(path).run()
    assert report["best"]["body.material.elastic.youngs_modulus"] == pytest.approx(30e3, rel=0.01)
    assert report["cost"] < 0.01 < report["raw_card_cost"]
    assert report["checks"]["other_pressure"]["transferred"] < 0.01
    card = yaml.safe_load(Path(report["material_file"]).read_text())
    assert card["kind"] == "effective" and card["provenance"]["backend"] == "mujoco"
    # resumable: a second run reuses every logged evaluation
    assert len(Transfer(path).history) == report["evaluations"]


def test_bending_outputs_follow_the_top_face():
    from softsim.backends.base import SimState, SoftState

    path = SPHERE.parent / "bending.yaml"
    sc = InflationScenario(validate(InflationConfig, load_yaml(path)), path.parent)
    scene = sc.build_scene()
    body = scene.soft_bodies[0]
    assert len(open_edges(body.region("cavity"))) == 0 and body.pinned
    # thin wall on +x: the cavity's vertices sit off-axis
    assert body.mesh.points[np.unique(body.region("cavity")), 0].mean() == pytest.approx(0.002, abs=2e-4)
    a = 0.3  # rigid rotation about y: the top face tilts by a, its centre moves to (h sin a, h cos a)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    x = body.mesh.points @ rot.T
    m = sc.metrics(SimState(0.0, soft={body.name: SoftState(x, np.zeros_like(x))}, controls={"cavity": 0.0}))
    assert m["bend_angle"] == pytest.approx(a, rel=1e-6)
    assert m["tip_x"] == pytest.approx(0.04 * math.sin(a), rel=1e-3) and m["tip_z"] == pytest.approx(
        0.04 * math.cos(a), rel=1e-3)

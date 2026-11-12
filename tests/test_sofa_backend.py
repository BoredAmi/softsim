"""SOFA backend: capabilities and mappings (always), physics checks (only if SOFA is installed)."""

import math

import numpy as np
import pytest

from softsim.backends.sofa import SofaBackend, force_field_args, lame, rigid_mass_properties
from softsim.config.loader import apply_overrides
from softsim.config.schema import validate
from softsim.errors import SoftSimError, UnsupportedFeatureError
from softsim.geometry.rigid import Part
from softsim.materials.models import Material
from softsim.meshing import structured_cylinder
from softsim.scenarios.drop_test import DropTestConfig, DropTestScenario
from softsim.scene import RigidBodySpec, SoftBodySpec


def _scene(raw, base_dir, overrides=()):
    cfg = validate(DropTestConfig, apply_overrides(raw, overrides))
    return DropTestScenario(cfg, base_dir).build_scene()


@pytest.mark.parametrize("elastic,ok", [
    ({"model": "mooney_rivlin", "c10": 1e4, "c01": 1e3, "bulk_modulus": 1e6}, True),
    ({"model": "ogden", "mu": [2e4], "alpha": [2.0], "bulk_modulus": 1e6}, True),
    ({"model": "ogden", "mu": [2e4, 1e3], "alpha": [2.0, -2.0], "bulk_modulus": 1e6}, False),
    ({"model": "yeoh", "c10": 1e4, "bulk_modulus": 1e6}, False),
])
def test_capabilities(drop_config, drop_base_dir, elastic, ok):
    raw = apply_overrides(drop_config, [])
    raw["specimen"]["material"] = {"density": 1000, "elastic": elastic}
    scene = _scene(raw, drop_base_dir)
    if ok:
        SofaBackend.check(scene)
    else:
        with pytest.raises(UnsupportedFeatureError):
            SofaBackend.check(scene)


def test_torsional_friction_rejected(drop_config, drop_base_dir):
    scene = _scene(drop_config, drop_base_dir, ["weight.friction.torsional=0.01"])
    assert any("torsional" in m for m in SofaBackend.unsupported(scene))


def test_material_mapping():
    mu, lam = lame(50e3, 0.45)
    assert mu == pytest.approx(50e3 / 2.9) and lam == pytest.approx(50e3 * 0.45 / (1.45 * 0.1))
    mesh = structured_cylinder(0.02, 0.017, 0.008)

    def args(elastic):
        body = SoftBodySpec("s", mesh, validate(Material, {"density": 1000, "elastic": elastic}))
        return force_field_args(body)

    assert args({"model": "neo_hookean", "youngs_modulus": 50e3, "poisson_ratio": 0.45}) == (
        "TetrahedronHyperelasticityFEMForceField", {"materialName": "NeoHookean", "ParameterSet": [mu, lam]})
    assert args({"model": "linear", "youngs_modulus": 5e4, "poisson_ratio": 0.3})[1]["method"] == "small"
    # softsim Ogden term mu/alpha (shear 0.5 mu alpha) -> SOFA mu1/alpha^2 (shear mu1/2): mu1 = mu*alpha
    assert args({"model": "ogden", "mu": [1e4], "alpha": [3.0], "bulk_modulus": 1e6})[1]["ParameterSet"] == [3e4, 3.0, 1e6]


def test_rigid_mass_properties():
    r, h, m = 0.01, 0.02, 0.1
    body = RigidBodySpec("w", (Part("cylinder", (r, h / 2, 0), (0, 0, h / 2)),), m)
    com, inertia = rigid_mass_properties(body)
    assert com == pytest.approx([0, 0, h / 2])
    assert np.diag(inertia) == pytest.approx([m * (3 * r**2 + h**2) / 12] * 2 + [m * r**2 / 2])


# --- physics, only with SOFA --------------------------------------------------------------

def _sofa():
    try:
        from softsim.backends.sofa.backend import import_sofa

        return import_sofa()
    except SoftSimError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("elastic,expected", [
    ({"model": "linear", "youngs_modulus": 50e3, "poisson_ratio": 0.45}, 1.0),
    ({"model": "neo_hookean", "youngs_modulus": 50e3, "poisson_ratio": 0.45}, 1.0),
    ({"model": "stable_neo_hookean", "youngs_modulus": 50e3, "poisson_ratio": 0.45}, 1.0),
    # Mooney-Rivlin / Ogden built with the same small-strain E and nu
    ({"model": "mooney_rivlin", "c10": 50e3 / 2.9 / 4, "c01": 50e3 / 2.9 / 4, "bulk_modulus": 50e3 / 0.3}, 1.0),
    ({"model": "ogden", "mu": [50e3 / 2.9 / 4], "alpha": [8.0], "bulk_modulus": 50e3 / 0.3}, 1.0),
])
def test_uniaxial_fem_without_contact(elastic, expected):
    """1 % frictionless uniaxial compression by nodal forces (no contact): validates the material mapping."""
    Sofa = _sofa()
    from softsim.backends.sofa.backend import PLUGINS

    H, R, strain = 0.017, 0.02, 0.01
    material = validate(Material, {"density": 1000, "elastic": elastic})
    e0 = material.elastic.initial_youngs_modulus()
    mesh = structured_cylinder(R, H, 0.004)
    pts = mesh.points
    top = np.nonzero(np.isclose(pts[:, 2], H))[0]
    bot = np.nonzero(np.isclose(pts[:, 2], 0))[0]
    tris = mesh.surface_triangles()
    fz = np.zeros(len(pts))
    for t in tris[np.all(np.isin(tris, top), axis=1)]:
        fz[t] += 0.5 * np.linalg.norm(np.cross(pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]])) / 3
    fz *= strain * e0 * math.pi * R**2 / fz.sum()
    ff, args = force_field_args(SoftBodySpec("s", mesh, material))
    root = Sofa.Core.Node("root")
    root.addObject("RequiredPlugin", pluginName=PLUGINS + ["Sofa.Component.MechanicalLoad"])
    root.gravity = [0, 0, 0]
    root.addObject("DefaultAnimationLoop")
    n = root.addChild("s")
    n.addObject("StaticSolver", newton_iterations=20)
    n.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    n.addObject("TetrahedronSetTopologyContainer", position=pts.tolist(), tetrahedra=mesh.tets.tolist())
    dofs = n.addObject("MechanicalObject", position=pts.tolist())
    n.addObject(ff, **args)
    n.addObject("PartialFixedProjectiveConstraint", indices=bot.tolist(), fixedDirections=[0, 0, 1])
    centre = int(bot[np.argmin(np.linalg.norm(pts[bot, :2], axis=1))])
    n.addObject("FixedProjectiveConstraint", indices=[centre])
    rim = int(bot[np.argmax(pts[bot, 0])])
    n.addObject("PartialFixedProjectiveConstraint", name="no_spin", indices=[rim], fixedDirections=[0, 1, 1])
    n.addObject("ConstantForceField", indices=top.tolist(), forces=[[0, 0, -f] for f in fz[top]])
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, 0.01)
    p = np.asarray(dofs.position.value)
    delta = H - (p[top, 2].mean() - p[bot, 2].mean())
    Sofa.Simulation.unload(root)
    assert delta / (strain * H) == pytest.approx(expected, rel=0.04)


@pytest.mark.slow
def test_short_drop_on_sofa(drop_config, drop_base_dir):
    _sofa()
    from softsim.runner import run_config

    r = run_config(apply_overrides(drop_config, ["simulation.duration=0.006", "simulation.timestep=0.0001",
                                                "output.save_model=true"]),
                   backend="sofa", base_dir=drop_base_dir)
    s = r.summary
    assert np.isfinite(r.timeseries.drop(columns=["weight_com_x"]).to_numpy()).all()
    assert s["contact"] == 1.0 and 0.5e-3 < s["peak_indentation"] < 8e-3
    assert r.timeseries.specimen_min_z.min() > -0.5e-3  # no sinking through the ground
    assert "TetrahedronHyperelasticityFEMForceField" in r.model_text


def test_blow_up_guard():
    ok = {"s": np.zeros((4, 3)), "w": np.array([0, 0, -2.7, 50.0, 0, 0])}  # 50 rad/s spin is fine
    assert not SofaBackend._blown_up(ok)
    assert SofaBackend._blown_up({**ok, "s": np.full((4, 3), 150.0)})
    assert SofaBackend._blown_up({**ok, "s": np.array([[0, 0, np.nan]])})


def test_mass_proportional_damping_matches_free_fall_with_drag():
    """Rayleigh alpha is a linear drag ``-alpha m v``: a body in free fall (nothing fixed, no ground, internal
    forces zero) obeys v' = g - alpha v, so v -> g / alpha. SOFA's implicit Euler gives exactly
    v_{k+1} = (v_k + g dt) / (1 + alpha dt) (matched to 1e-12); the continuous solution g/alpha (1 - e^{-alpha t})
    is within 1% at every step (first order in dt) and 0.1% at the end. Without damping the same body falls at g t. Mutants: alpha ignored,
    alpha applied as the stiffness coefficient."""
    from pathlib import Path

    from softsim.config.loader import load_yaml
    from softsim.scenarios.inflation import InflationConfig, InflationScenario

    _sofa()
    path = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_cantilever.yaml"
    g, dt, steps = 9.81, 1e-3, 200

    def fall(alpha):
        overrides = ["body.mesh.size=4mm", "body.fixed=[]", "actuators=[]", f"simulation.timestep={dt}",
                     f"simulation.gravity=[0,0,{-g}]",
                     "body.material.damping={model: rayleigh, alpha: %g, beta: 0.0}" % alpha]
        scene = InflationScenario(validate(InflationConfig, apply_overrides(load_yaml(path), overrides)),
                                  path.parent).build_scene()
        backend = SofaBackend()
        backend.build(scene)
        v = []
        for _ in range(steps):
            backend.step(1)
            v.append(backend.state().soft["beam"].velocities.mean(axis=0)[2])
        backend.close()
        return np.array(v)

    t = dt * np.arange(1, steps + 1)
    np.testing.assert_allclose(fall(0.0), -g * t, rtol=1e-12)
    alpha = 20.0
    v = np.array(fall(alpha))
    exact = np.zeros(steps)
    x = 0.0
    for k in range(steps):
        x = (x - g * dt) / (1 + alpha * dt)
        exact[k] = x
    np.testing.assert_allclose(v, exact, rtol=1e-12)
    continuous = -g / alpha * (1 - np.exp(-alpha * t))
    np.testing.assert_allclose(v, continuous, rtol=1.5e-2)
    assert v[-1] == pytest.approx(continuous[-1], rel=1e-3)


def _plate_compression(mu, h=0.008, strain=0.02, steps=500, dt=5e-4):
    """A weighted plate on a cylinder (R 20 mm, H 17 mm), the ground and the plate with friction ``mu``: the
    compression it settles to, and the same body's compression under the same load without contact (nodal forces
    on the top face, static, with its own weight), each as a fraction of ``strain``; plus the mean rim radius
    change of the contact run."""
    from softsim.backends import get_backend
    from softsim.materials.homogeneous import curve
    from softsim.scene import Friction, GroundSettings, Pose, Scene, SimulationSettings

    Sofa = _sofa()
    from softsim.backends.sofa.backend import PLUGINS

    R, H = 0.02, 0.017
    material = validate(Material, {"density": 1000, "friction": {"sliding": mu},
                                   "elastic": {"model": "stable_neo_hookean", "youngs_modulus": 50e3, "poisson_ratio": 0.45},
                                   "damping": {"model": "rayleigh", "beta": 2e-2}})
    mesh = structured_cylinder(R, H, h)
    pts = mesh.points
    sigma = float(-curve(material, "uniaxial", [1 - strain]).stress.iloc[0])
    top, bot = np.isclose(pts[:, 2], H), np.isclose(pts[:, 2], 0)
    rim = np.linalg.norm(pts[:, :2], axis=1) > R - 1e-6
    plate = RigidBodySpec("plate", (Part("box", (0.05, 0.05, 0.003), (0, 0, 0.003)),), sigma * math.pi * R**2 / 9.81,
                          pose=Pose((0, 0, H + 2e-4)), friction=Friction(sliding=mu))
    scene = Scene((SoftBodySpec("s", mesh, material),), (plate,), ground=GroundSettings(friction=Friction(sliding=mu)),
                  simulation=SimulationSettings(timestep=dt, duration=steps * dt, output_rate=1000))
    backend = get_backend("sofa")()
    backend.build(scene)
    backend.step(steps)
    v = backend.state().soft["s"].vertices
    backend.close()
    contact = (H - (v[top, 2].mean() - v[bot, 2].mean())) / H / strain
    spread = float(np.linalg.norm(v[rim, :2] - v[:, :2].mean(0), axis=1).mean() / R - 1)

    # the same load without contact
    tris = mesh.surface_triangles()
    fz = np.zeros(len(pts))
    ti = np.nonzero(top)[0]
    for t in tris[np.all(np.isin(tris, ti), axis=1)]:
        fz[t] += 0.5 * np.linalg.norm(np.cross(pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]])) / 3
    fz *= sigma * math.pi * R**2 / fz.sum()
    ff, args = force_field_args(SoftBodySpec("s", mesh, material))
    root = Sofa.Core.Node("root")
    root.addObject("RequiredPlugin", pluginName=PLUGINS + ["Sofa.Component.MechanicalLoad"])
    root.gravity = [0, 0, -9.81]
    root.addObject("DefaultAnimationLoop")
    n = root.addChild("s")
    n.addObject("StaticSolver", newton_iterations=30)
    n.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
    n.addObject("TetrahedronSetTopologyContainer", position=pts.tolist(), tetrahedra=mesh.tets.tolist())
    dofs = n.addObject("MechanicalObject", position=pts.tolist())
    n.addObject("MeshMatrixMass", massDensity=1000.0, lumping=True)
    n.addObject(ff, **args)
    bi = np.nonzero(bot)[0]
    n.addObject("PartialFixedProjectiveConstraint", indices=bi.tolist(), fixedDirections=[0, 0, 1])
    n.addObject("FixedProjectiveConstraint", indices=[int(bi[np.argmin(np.linalg.norm(pts[bi, :2], axis=1))])])
    n.addObject("PartialFixedProjectiveConstraint", indices=[int(bi[np.argmax(pts[bi, 0])])], fixedDirections=[0, 1, 1])
    n.addObject("ConstantForceField", indices=ti.tolist(), forces=[[0, 0, -f] for f in fz[ti]])
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, 0.01)
    p = np.asarray(dofs.position.value)
    free = (H - (p[top, 2].mean() - p[bot, 2].mean())) / H / strain
    free_spread = float(np.linalg.norm(p[rim, :2], axis=1).mean() / R - 1)
    Sofa.Simulation.unload(root)
    return contact, free, spread, free_spread


def test_frictionless_plate_through_contact_matches_no_contact():
    """Static compression by a frictionless plate through contact (the 'frictionless plate 7% stiff' limit): the
    compression and the lateral spreading of the rim match the same load applied as nodal forces (measured at
    8 mm: 0.6% and 0.1%). Nothing is wrong with the contact; see ``test_any_friction_makes_the_plate_sticky``."""
    contact, free, spread, free_spread = _plate_compression(0.0)
    assert contact == pytest.approx(free, rel=0.015)
    assert spread == pytest.approx(free_spread, rel=0.03)


@pytest.mark.slow
def test_any_friction_makes_the_plate_sticky():
    """SOFA applies the scene's largest friction coefficient to every pair: with 0.8 on the ground only, the plate
    contact sticks and the body compresses 33% less and spreads half as much (measured at 8 mm: 0.738 against
    1.101, rim 0.48% against 1.01%). That was the 'stiff plate'."""
    contact, free, spread, free_spread = _plate_compression(0.8)
    assert contact < 0.85 * free and spread < 0.7 * free_spread


def test_alarm_distance_follows_the_fastest_vertex():
    backend = SofaBackend()
    backend._alarm0, backend._contact_distance, backend._alarm_factor = 4e-4, 1e-4, 1.0
    backend._prev = {"s": np.zeros((3, 3)), "w": np.array([0, 0, -2.7, 50.0, 0, 0])}  # angular rates do not count
    assert backend._alarm(5e-5) == 4e-4
    backend._prev["s"][1] = [0.0, 3.0, -12.0]  # 12 m/s * 50 us = 0.6 mm per step
    assert backend._alarm(5e-5) == pytest.approx(1e-4 + 12.0 * 5e-5)


@pytest.mark.slow
def test_rim_flare_does_not_go_through_the_ground():
    """The 500 g drop with the fitted Ogden material (5 mm mesh, 8.8 ms): the top rim, rolled under the specimen,
    falls back onto the ground at 12 m/s, 0.6 mm per 50 us step against a 0.4 mm alarm distance, and a vertex
    passed 2.8 mm through it (8.5 mm by 22 ms). The alarm distance now follows the per-step travel."""
    _sofa()
    from pathlib import Path

    from softsim.runner import run_config

    case = Path(__file__).parents[1] / "scenarios" / "drop_test" / "cases" / "w500.yaml"
    r = run_config(case, ["backend=sofa", "specimen.material=ecoflex_00_10_drop_fit_sofa_ogden",
                          "specimen.mesh.size=0.005", "simulation.duration=0.0088", "simulation.timestep=5e-5",
                          "simulation.output_rate=10000", "output.save_model=false"])
    ts = r.timeseries
    assert ts.specimen_min_z.min() > -0.1e-3  # 2.83 mm before
    assert ts.compression.max() == pytest.approx(13.13e-3, rel=0.02)

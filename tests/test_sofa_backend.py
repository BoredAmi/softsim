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

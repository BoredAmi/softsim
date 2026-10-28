"""MuJoCo model generation and conventions."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from softsim.backends.mujoco import MujocoBackend, scene_to_mjcf  # noqa: E402
from softsim.config.loader import apply_overrides  # noqa: E402
from softsim.config.schema import validate  # noqa: E402
from softsim.scenarios.drop_test import DropTestConfig, DropTestScenario  # noqa: E402


def _built(raw, base_dir, overrides=()):
    cfg = validate(DropTestConfig, apply_overrides(raw, overrides))
    scene = DropTestScenario(cfg, base_dir).build_scene()
    backend = MujocoBackend()
    backend.build(scene)
    return scene, backend


def test_mjcf_generation(drop_config, drop_base_dir):
    cfg = validate(DropTestConfig, drop_config)
    xml = scene_to_mjcf(DropTestScenario(cfg, drop_base_dir).build_scene())
    assert 'integrator="discrete"' in xml
    assert '<flexcomp name="specimen" type="direct" dim="3"' in xml
    assert 'young="91000"' in xml and 'poisson="0.45"' in xml and 'damping="0.002"' in xml


def test_vertices_masses_and_material_flags(drop_config, drop_base_dir):
    scene, backend = _built(drop_config, drop_base_dir)
    body = scene.soft_bodies[0]
    m = backend.model
    state = backend.state()
    np.testing.assert_allclose(state.soft["specimen"].vertices, body.mesh.points, atol=1e-9)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "specimen")
    vert_bodies = m.flex_vertbodyid[m.flex_vertadr[fid]:m.flex_vertadr[fid] + m.flex_vertnum[fid]]
    np.testing.assert_allclose(m.body_mass[vert_bodies], body.mesh.lumped_masses(body.effective_density))
    assert m.body_mass[vert_bodies].sum() == pytest.approx(0.01949)
    assert backend.spec.flexes[0].elastic3d == 1  # Neo-Hookean
    assert "elastic3d=1" in backend.export_model()
    assert m.body("weight").subtreemass == pytest.approx(0.020) if hasattr(m.body("weight"), "subtreemass") \
        else m.body_subtreemass[m.body("weight").id] == pytest.approx(0.020)


def test_stvk_flag(drop_config, drop_base_dir):
    _, backend = _built(drop_config, drop_base_dir, ["specimen.material.preset=ecoflex_00_10",
                                                     "specimen.material.elastic.model=stvk"])
    assert backend.spec.flexes[0].elastic3d == 0


def test_bonded_base_pins_bottom(drop_config, drop_base_dir):
    scene, backend = _built(drop_config, drop_base_dir, ["specimen.base=bonded"])
    pinned = scene.soft_bodies[0].pinned
    bottom = np.nonzero(scene.soft_bodies[0].mesh.points[:, 2] < 1e-9)[0]
    assert sorted(pinned) == sorted(bottom.tolist())
    backend.step(50)
    v = backend.state().soft["specimen"].vertices
    np.testing.assert_allclose(v[bottom], scene.soft_bodies[0].mesh.points[bottom], atol=1e-9)


def test_initial_conditions(drop_config, drop_base_dir):
    scene, backend = _built(drop_config, drop_base_dir, ["drop.tilt=[10 deg, 0]"])
    w = backend.state().rigid["weight"]
    assert w.linear_velocity[2] == pytest.approx(-np.sqrt(2 * 9.81 * (0.38 - 0.0005)), rel=1e-6)
    # tilted weight: lowest rim point is exactly `gap` above the specimen top
    r = 0.01125
    assert w.position[2] - r * np.sin(np.deg2rad(10)) == pytest.approx(0.017 + 0.0005, abs=1e-9)


def test_contact_force_sign(drop_config, drop_base_dir):
    """Weight resting on the specimen: contact force on the weight balances gravity; ground carries both."""
    scene, backend = _built(drop_config, drop_base_dir, ["drop.initial_speed=0", "drop.gap=0.0001",
                                                         "specimen.material.preset=ecoflex_00_10",
                                                         "specimen.material.damping.beta=0.01"])
    backend.step(1500)  # 0.3 s at dt = 0.2 ms
    s = backend.state()
    on_weight = s.contact_between("specimen", "weight").force
    assert on_weight[2] == pytest.approx(0.020 * 9.81, rel=0.03)
    ground = s.contact_between("ground", "specimen").force
    assert ground[2] == pytest.approx((0.020 + 0.01949) * 9.81, rel=0.03)
    assert s.net_contact_force("weight")[2] == pytest.approx(on_weight[2])

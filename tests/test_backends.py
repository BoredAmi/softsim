"""Capability checks: unsupported combinations must fail loudly, listing every gap."""

import pytest

from softsim.backends import available_backends, get_backend
from softsim.config.loader import apply_overrides
from softsim.config.schema import validate
from softsim.errors import ConfigError, SoftSimError, UnsupportedFeatureError
from softsim.scenarios.drop_test import DropTestConfig, DropTestScenario


def _scene(raw, base_dir, overrides=()):
    cfg = validate(DropTestConfig, apply_overrides(raw, overrides))
    return DropTestScenario(cfg, base_dir).build_scene()


def test_registry():
    assert {"mujoco", "sofa"} <= set(available_backends())
    with pytest.raises(ConfigError):
        get_backend("bullet")


@pytest.mark.parametrize("material,expected", [
    ({"density": 1000, "elastic": {"model": "linear", "youngs_modulus": 5e4, "poisson_ratio": 0.4}},
     ["small-strain linear elasticity"]),
    ({"density": 1000, "elastic": {"model": "mooney_rivlin", "c10": 1e4, "c01": 1e3, "bulk_modulus": 1e6},
      "damping": {"model": "prony", "g": [0.3], "tau": [0.01]}},
     ["Mooney-Rivlin hyperelasticity", "Prony-series viscoelasticity"]),
    ({"density": 1000, "elastic": {"model": "neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.4},
      "damping": {"model": "rayleigh", "alpha": 1.0}},
     ["mass-proportional (Rayleigh alpha) damping"]),
    ({"density": 2000, "elastic": {"model": "magneto_active", "particle_volume_fraction": 0.2,
                                    "matrix": {"model": "neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.4}}},
     ["magneto-active (MRE) coupling"]),
])
def test_mujoco_rejects(drop_config, drop_base_dir, material, expected):
    raw = {**drop_config, "specimen": {**drop_config["specimen"], "material": material}}
    scene = _scene(raw, drop_base_dir)
    backend = get_backend("mujoco")
    with pytest.raises(UnsupportedFeatureError) as info:
        backend.check(scene)
    assert info.value.missing == sorted(expected) or set(expected) <= set(info.value.missing)


def test_mujoco_rejects_settings(drop_config, drop_base_dir):
    scene = _scene(drop_config, drop_base_dir, ["simulation.integrator=implicitfast", "simulation.solver=PGS"])
    missing = get_backend("mujoco").unsupported(scene)
    assert any("discrete" in m for m in missing) and any("PGS" in m for m in missing)


def test_mujoco_accepts_default(drop_config, drop_base_dir):
    get_backend("mujoco").check(_scene(drop_config, drop_base_dir))


def test_sofa_stub(drop_config, drop_base_dir):
    scene = _scene(drop_config, drop_base_dir)
    sofa = get_backend("sofa")
    sofa.check(scene)  # SOFA will support this scene...
    with pytest.raises(SoftSimError, match="not implemented"):
        sofa().build(scene)  # ...but is not implemented yet

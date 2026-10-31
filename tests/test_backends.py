"""Capability checks: unsupported combinations must fail loudly, listing every gap."""

import pytest

from softsim.backends import available_backends, get_backend
from softsim.config.loader import apply_overrides
from softsim.config.schema import validate
from softsim.errors import ConfigError, UnsupportedFeatureError
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
     ["Mooney-Rivlin hyperelasticity"]),  # Prony itself is supported (explicit overstress)
    ({"density": 1000, "elastic": {"model": "stable_neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.4},
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


def test_mujoco_has_no_classic_neo_hookean(drop_config, drop_base_dir):
    """MuJoCo's flex Neo-Hookean is the stable variant; the classic ln(J) model must not run silently."""
    raw = {**drop_config, "specimen": {**drop_config["specimen"], "material": {
        "density": 1000, "elastic": {"model": "neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.4}}}}
    missing = get_backend("mujoco").unsupported(_scene(raw, drop_base_dir))
    assert missing[0] == "Neo-Hookean hyperelasticity"
    assert "softsim material project <card> --to neo_hookean" in missing[1]  # how to convert the card


def test_mujoco_friction_cone_and_impratio_reach_the_model(drop_config, drop_base_dir):
    pytest.importorskip("mujoco")
    from softsim.backends.mujoco.builder import scene_to_mjcf

    default = scene_to_mjcf(_scene(drop_config, drop_base_dir))
    assert 'cone="pyramidal"' in default and 'impratio="1"' in default
    mjcf = scene_to_mjcf(_scene(drop_config, drop_base_dir, ["contact.cone=elliptic", "contact.impratio=10"]))
    assert 'cone="elliptic"' in mjcf and 'impratio="10"' in mjcf
    with pytest.raises(ConfigError):
        _scene(drop_config, drop_base_dir, ["contact.impratio=0.5"])


def test_locking_warning_only_for_backends_that_lock(drop_config, drop_base_dir):
    from softsim.checks.stability import check_scene

    raw = {**drop_config, "specimen": {**drop_config["specimen"], "material": {
        "density": 1000, "elastic": {"model": "neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.4995}}}}
    scene = _scene(raw, drop_base_dir)
    codes = {f.code for f in check_scene(scene)}
    assert "volumetric-locking" in codes
    assert get_backend("fem").locking_free and not get_backend("mujoco").locking_free
    assert "volumetric-locking" not in {f.code for f in check_scene(scene, locking_free=True)}

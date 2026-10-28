import pytest

from softsim.capabilities import Capability
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.materials import (
    Material,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    PronySeries,
    list_presets,
    load_material,
    save_material,
)


def test_presets_load():
    names = list_presets()
    for required in ("ecoflex_00_10", "ecoflex_00_30", "dragon_skin_10", "generic_gel"):
        assert required in names
        m = load_material(required)
        assert m.density > 0 and m.source


def test_preset_with_overrides():
    m = load_material({"preset": "ecoflex_00_10", "elastic": {"youngs_modulus": "40 kPa"}})
    assert m.elastic.youngs_modulus == 40e3
    assert m.elastic.poisson_ratio == 0.45  # inherited


def test_unknown_preset():
    with pytest.raises(ConfigError, match="unknown material preset"):
        load_material("unobtainium")


def test_env_search_path(tmp_path, monkeypatch):
    (tmp_path / "lab_gel.yaml").write_text(
        "density: 1000\nelastic: {model: stvk, youngs_modulus: 5 kPa, poisson_ratio: 0.3}\n")
    monkeypatch.setenv("SOFTSIM_MATERIAL_PATH", str(tmp_path))
    assert "lab_gel" in list_presets()
    assert load_material("lab_gel").elastic.model == "stvk"


@pytest.mark.parametrize("bad", [
    {"model": "neo_hookean", "youngs_modulus": 1e4, "poisson_ratio": 0.5},
    {"model": "neo_hookean", "youngs_modulus": -1, "poisson_ratio": 0.3},
    {"model": "ogden", "mu": [1e3], "alpha": [2.0, 3.0], "bulk_modulus": 1e6},
    {"model": "unknown"},
])
def test_invalid_elastic(bad):
    with pytest.raises(ConfigError):
        validate(Material, {"density": 1000, "elastic": bad})


def test_small_strain_equivalents():
    nh = NeoHookean(youngs_modulus=60e3, poisson_ratio=0.45)
    assert nh.shear_modulus == pytest.approx(60e3 / 2.9)
    # Mooney-Rivlin with mu0 = 2 (C10 + C01) and a bulk modulus equivalent to nu = 0.45
    mu = 20e3
    k = 2 * mu * 1.45 / (3 * (1 - 0.9))
    mr = MooneyRivlin(c10=mu / 4, c01=mu / 4, bulk_modulus=k)
    assert mr.initial_poisson_ratio() == pytest.approx(0.45)
    assert mr.initial_youngs_modulus() == pytest.approx(2 * mu * 1.45)
    og = Ogden(mu=[mu], alpha=[2.0], bulk_modulus=k)
    assert og.initial_youngs_modulus() == pytest.approx(2 * mu * 1.45)


def test_prony_validation():
    with pytest.raises(ValueError):
        PronySeries(g=[0.6, 0.5], tau=[0.01, 0.1])


def test_requirements():
    m = load_material("ecoflex_00_10")
    assert Capability.NEO_HOOKEAN in m.requires
    assert Capability.DAMPING_STIFFNESS_PROPORTIONAL in m.requires
    mag = validate(Material, {"density": 2000, "elastic": {
        "model": "magneto_active", "particle_volume_fraction": 0.3,
        "matrix": {"model": "neo_hookean", "youngs_modulus": 5e4, "poisson_ratio": 0.45}}})
    assert Capability.MAGNETO_ACTIVE in mag.requires


def test_save_roundtrip(tmp_path):
    m = load_material({"preset": "ecoflex_00_10", "name": "fitted", "provenance": {"mesh_size": 0.0035}})
    path = save_material(m, tmp_path / "fitted.yaml")
    again = load_material(str(path))
    assert again == m


def test_switching_model_replaces_section():
    m = load_material({"preset": "ecoflex_00_10",
                       "elastic": {"model": "mooney_rivlin", "c10": 1e4, "c01": 1e3, "bulk_modulus": 1e6}})
    assert m.elastic.model == "mooney_rivlin"
    assert m.damping.model == "rayleigh"  # untouched sections still inherited
    # different model with the same parameters keeps them
    m = load_material({"preset": "ecoflex_00_10", "elastic": {"model": "stvk"}})
    assert m.elastic.model == "stvk" and m.elastic.youngs_modulus == 91e3
    # same model: fields merge as before
    m = load_material({"preset": "ecoflex_00_10", "elastic": {"model": "neo_hookean", "poisson_ratio": 0.4}})
    assert m.elastic.youngs_modulus == 91e3 and m.elastic.poisson_ratio == 0.4

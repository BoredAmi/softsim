"""Material cards: alternative inputs, new laws, card metadata, overrides, sampling, parameter listing."""

import math

import numpy as np
import pytest

from softsim.cli import main
from softsim.config.params import config_params, find_param, iter_params
from softsim.config.schema import validate
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.materials import ArrudaBoyce, Gent, Material, load_material, sample
from softsim.materials.models import arruda_boyce_shear_factor, isotropic_e_nu
from softsim.materials.summary import material_summary

E, NU = 60e3, 0.45
MU, K = E / (2 * (1 + NU)), E / (3 * (1 - 2 * NU))


@pytest.mark.parametrize("given", [
    {"youngs_modulus": E, "poisson_ratio": NU},
    {"shear_modulus": MU, "poisson_ratio": NU},
    {"bulk_modulus": K, "poisson_ratio": NU},
    {"youngs_modulus": E, "shear_modulus": MU},
    {"youngs_modulus": E, "bulk_modulus": K},
    {"shear_modulus": MU, "bulk_modulus": K},
])
def test_any_two_isotropic_constants(given):
    e, nu = isotropic_e_nu(given)
    assert e == pytest.approx(E) and nu == pytest.approx(NU)
    m = validate(Material, {"density": 1000, "elastic": {"model": "neo_hookean", **given}})
    assert m.elastic.youngs_modulus == pytest.approx(E)
    assert m.elastic.initial_shear_modulus() == pytest.approx(MU)
    assert m.elastic.initial_bulk_modulus() == pytest.approx(K)


def test_isotropic_constants_with_units_and_overdetermined():
    m = validate(Material, {"density": 1000, "elastic": {"model": "stvk", "shear_modulus": "20 kPa",
                                                         "bulk_modulus": "1 MPa"}})
    assert m.elastic.initial_shear_modulus() == pytest.approx(20e3)
    with pytest.raises(ConfigError, match="exactly two"):
        validate(Material, {"density": 1000, "elastic": {"model": "neo_hookean", "youngs_modulus": E,
                                                         "poisson_ratio": NU, "shear_modulus": MU}})


@pytest.mark.parametrize("elastic", [
    {"model": "mooney_rivlin", "c10": "8 kPa", "c01": "2 kPa"},
    {"model": "yeoh", "c10": "10 kPa", "c20": "1 kPa"},
    {"model": "ogden", "mu": ["20 kPa", "1 kPa"], "alpha": [1.3, 5.0]},
    {"model": "gent", "mu": "25 kPa", "jm": 30},
    {"model": "arruda_boyce", "mu": "25 kPa", "lambda_m": 3.5},
])
def test_poisson_ratio_instead_of_bulk_modulus(elastic):
    m = validate(Material, {"density": 1000, "elastic": {**elastic, "poisson_ratio": 0.49}})
    assert m.elastic.initial_poisson_ratio() == pytest.approx(0.49)
    with pytest.raises(ConfigError, match="not both"):
        validate(Material, {"density": 1000, "elastic": {**elastic, "poisson_ratio": 0.49, "bulk_modulus": 1e6}})


def test_gent_and_arruda_boyce():
    g = Gent(mu=30e3, jm=20, bulk_modulus=1e6)
    assert g.initial_shear_modulus() == 30e3 and g.requires
    a = ArrudaBoyce(mu=30e3, lambda_m=4.0, bulk_modulus=1e6)
    assert a.initial_shear_modulus() == pytest.approx(30e3 * (1 + 3 / 80 + 99 / 175 / 256 + 513 / 875 / 4096
                                                               + 42039 / 67375 / 65536))
    assert arruda_boyce_shear_factor(1e4) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ArrudaBoyce(mu=30e3, lambda_m=0.9, bulk_modulus=1e6)


def test_effective_card_needs_backend_and_mesh():
    with pytest.raises(ConfigError, match="backend and mesh_size"):
        validate(Material, {"density": 1000, "kind": "effective",
                            "elastic": {"model": "stvk", "youngs_modulus": E, "poisson_ratio": NU}})
    for name in ("ecoflex_00_10_drop_fit", "ecoflex_00_10_drop_fit_tacky", "ecoflex_00_10_drop_fit_sofa_ogden"):
        assert load_material(name).kind == "effective"
    assert load_material("ecoflex_00_10").kind == "physical"


def test_validity_and_references():
    m = load_material({"preset": "ecoflex_00_10", "validity": {"strain": [-0.5, 3.0], "strain_rate": ["0 1/s", "300 1/s"],
                                                               "temperature": "23 degC"},
                       "references": [{"doi": "10.1089/soro.2019.0115", "note": "uniaxial tension"}]})
    assert m.validity.temperature == pytest.approx(296.15) and m.validity.strain_rate == (0.0, 300.0)
    with pytest.raises(ConfigError, match="low, high"):
        load_material({"preset": "ecoflex_00_10", "validity": {"strain": [1.0, -0.5]}})


def test_uncertainty_keys_checked_and_units_parsed():
    m = load_material({"preset": "ecoflex_00_10", "uncertainty": {
        "elastic.youngs_modulus": {"abs": "5 kPa"}, "damping.beta": {"rel": 0.5, "dist": "lognormal"}}})
    assert m.uncertainty["elastic.youngs_modulus"].abs == 5e3
    with pytest.raises(ConfigError, match="not a parameter"):
        load_material({"preset": "ecoflex_00_10", "uncertainty": {"elastic.c10": {"rel": 0.1}}})
    with pytest.raises(ConfigError, match="not numeric"):
        load_material({"preset": "ecoflex_00_10", "uncertainty": {"elastic.model": {"rel": 0.1}}})
    with pytest.raises(ConfigError, match="has no"):
        load_material({"preset": "ecoflex_00_10", "uncertainty": {"elastic.poisson_ratio": {"abs": "1 kPa"}}})
    with pytest.raises(ConfigError, match="exactly one"):
        load_material({"preset": "ecoflex_00_10", "uncertainty": {"density": {"rel": 0.1, "abs": 1.0}}})


def test_overrides_switch_model_and_index_lists():
    base = load_material("ecoflex_00_10")
    y = base.with_overrides(["elastic.model=yeoh", "elastic.c10=12 kPa"])
    assert y.elastic.model == "yeoh" and y.elastic.initial_poisson_ratio() == pytest.approx(0.45)  # nu carried over
    assert base.elastic.model == "stable_neo_hookean"  # the original is unchanged
    p = base.with_overrides(["damping.model=prony", "damping.g=[0.5, 0.2]", "damping.tau=[1 ms, 20 ms]",
                             "damping.tau[1]=30 ms"])
    assert p.damping.tau == [1e-3, 30e-3]
    o = load_material("ecoflex_00_10_drop_fit_sofa_ogden", overrides=["elastic.mu[0]=30 kPa"])
    assert o.elastic.mu[0] == 30e3
    with pytest.raises(ConfigError):
        base.with_overrides(["elastic.poisson_ratio=0.5"])


def test_sample_statistics_redraws_and_reproducibility():
    m = load_material({"preset": "ecoflex_00_10", "uncertainty": {
        "elastic.youngs_modulus": {"rel": 0.1},
        "elastic.poisson_ratio": {"abs": 0.05, "dist": "uniform"},
        "damping.beta": {"rel": 0.5, "dist": "lognormal"}}})
    s = sample(m, 400, seed=3)
    e = np.array([x.elastic.youngs_modulus for x in s])
    nu = np.array([x.elastic.poisson_ratio for x in s])
    beta = np.array([x.damping.beta for x in s])
    assert e.mean() == pytest.approx(91e3, rel=0.02) and e.std() / 91e3 == pytest.approx(0.1, rel=0.15)
    assert nu.max() < 0.5 and nu.min() >= 0.4  # invalid draws (nu >= 0.5) were redrawn
    assert np.median(beta) == pytest.approx(2e-3, rel=0.1) and np.log(beta).std() == pytest.approx(0.5, rel=0.15)
    assert s[7].provenance["sample"]["index"] == 7
    again = sample(m, 3, seed=3)
    assert [x.elastic.youngs_modulus for x in again] == [x.elastic.youngs_modulus for x in s[:3]]
    with pytest.raises(ConfigError, match="no uncertainty"):
        sample(load_material("ecoflex_00_10"), 2)


def test_iter_params_units_bounds_choices_and_lists():
    m = load_material({"preset": "ecoflex_00_10_drop_fit_sofa_ogden", "damping": {"model": "prony", "g": [0.3],
                                                                                 "tau": ["2 ms"]}})
    params = {p.key: p for p in iter_params(m)}
    assert params["elastic.mu[0]"].unit.symbol == "Pa"
    assert params["damping.tau[0]"].unit.symbol == "s" and params["damping.tau[0]"].bounds == "> 0"
    assert params["damping.g[0]"].bounds == "> 0, < 1"
    assert "arruda_boyce" in params["elastic.model"].choices
    assert params["density"].describe() == "1040 kg/m^3"
    assert find_param(m, "elastic.alpha[0]").value == m.elastic.alpha[0]
    with pytest.raises(KeyError):
        find_param(m, "elastic.nope")


def test_config_params_resolve_materials():
    params = {p.key: p for p in config_params("scenarios/drop_test/cases/w050.yaml",
                                              ["specimen.material.elastic.youngs_modulus=40 kPa"])}
    assert params["specimen.material.elastic.youngs_modulus"].value == 40e3
    assert params["drop.angular_velocity[1]"].unit.symbol == "rad/s"
    assert params["support.stiffness"].unit.symbol == "N/m"


def test_units_added():
    assert parse_quantity("23 degC", "temperature") == pytest.approx(296.15)
    assert parse_quantity("300 1/s", "rate") == 300.0
    assert parse_quantity("11.6 J/m^2", "energy_per_area") == 11.6
    assert parse_quantity("2 kPa*s", "viscosity") == 2000.0
    assert parse_quantity("90 deg/s", "angular_velocity") == pytest.approx(math.pi / 2)


def test_summary_and_cli(capsys):
    s = material_summary(load_material("ecoflex_00_10"))
    assert s["shear_wave_speed"] == pytest.approx(math.sqrt(91e3 / 2.9 / 1040))
    assert s["dt_explicit"][0.005] == pytest.approx(0.005 / s["p_wave_speed"])
    assert main(["material", "show", "ecoflex_00_10", "--set", "elastic.model=gent", "--set", "elastic.mu=30 kPa",
                 "--set", "elastic.jm=20"]) == 0
    assert "gent" in capsys.readouterr().out
    assert main(["params", "scenarios/drop_test/cases/w050.yaml", "--grep", "elastic"]) == 0
    assert "specimen.material.elastic.youngs_modulus" in capsys.readouterr().out

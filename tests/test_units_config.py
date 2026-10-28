import pytest

from softsim.config.loader import apply_overrides, deep_merge, load_yaml
from softsim.config.schema import validate
from softsim.config.units import parse_quantity
from softsim.errors import ConfigError
from softsim.scenarios.drop_test import DropTestConfig

from conftest import DROP_DIR


@pytest.mark.parametrize("text,dim,expected", [
    ("17 mm", "length", 0.017),
    ("38cm", "length", 0.38),
    ("19.49 g", "mass", 0.01949),
    ("55 kPa", "pressure", 55e3),
    ("8 psi", "pressure", 8 * 6894.757293168),
    ("0.1 ms", "time", 1e-4),
    ("1040 kg/m^3", "density", 1040.0),
    ("1.04 g/cm^3", "density", 1040.0),
    ("10 deg", "angle", 0.17453292519943295),
    ("3 kHz", "frequency", 3000.0),
    (2.5, "length", 2.5),
    ("1e-3", "length", 1e-3),
])
def test_parse_quantity(text, dim, expected):
    assert parse_quantity(text, dim) == pytest.approx(expected)


@pytest.mark.parametrize("text,dim", [("5 g", "length"), ("abc", "length"), ("3 furlongs", "length"), (True, "mass")])
def test_parse_quantity_rejects(text, dim):
    with pytest.raises(ValueError):
        parse_quantity(text, dim)


def test_deep_merge_and_overrides():
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    merged = deep_merge(base, {"a": {"c": 3}, "d": [9]})
    assert merged == {"a": {"b": 1, "c": 3}, "d": [9]}
    assert base["a"]["c"] == 2  # input untouched
    out = apply_overrides(merged, ["a.b=0.5", "x.y=[1, 2]", "s=hello"])
    assert out["a"]["b"] == 0.5 and out["x"] == {"y": [1, 2]} and out["s"] == "hello"
    with pytest.raises(ConfigError):
        apply_overrides({}, ["novalue"])


def test_extends_and_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb: {c: 2, d: 3}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nb: {d: 4}\n")
    assert load_yaml(tmp_path / "child.yaml") == {"a": 1, "b": {"c": 2, "d": 4}}
    (tmp_path / "x.yaml").write_text("extends: y.yaml\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\n")
    with pytest.raises(ConfigError, match="cyclic"):
        load_yaml(tmp_path / "x.yaml")


@pytest.mark.parametrize("case", sorted((DROP_DIR / "cases").glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_cases_validate(case):
    cfg = validate(DropTestConfig, load_yaml(case))
    assert cfg.specimen.geometry.height == pytest.approx(0.017)
    assert cfg.simulation.integrator == "discrete"


def test_unknown_keys_rejected():
    raw = load_yaml(DROP_DIR / "cases" / "w020.yaml")
    with pytest.raises(ConfigError, match="youngs_modulos"):
        validate(DropTestConfig, apply_overrides(raw, ["specimen.youngs_modulos=1"]))
    with pytest.raises(ConfigError, match="timestep"):
        validate(DropTestConfig, apply_overrides(raw, ["simulation.timestep=-1"]))

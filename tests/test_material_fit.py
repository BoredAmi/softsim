"""Material-test datasets and card fitting (needs the fem extra and scipy)."""

import hashlib
import zipfile

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

from softsim.errors import ConfigError  # noqa: E402
from softsim.experiments.material_tests import TestCurve, load_material_tests, parse_srmd_csv  # noqa: E402
from softsim.materials import load_material  # noqa: E402
from softsim.materials.fit import fit, project, stability_limits, with_fit_uncertainty  # noqa: E402
from softsim.materials.homogeneous import curve  # noqa: E402

BASE = load_material({"name": "fit_base", "density": 1000,
                      "elastic": {"model": "neo_hookean", "youngs_modulus": 1e4, "poisson_ratio": 0.49}})
STRETCH = np.linspace(1.02, 3.0, 80)


def _synthetic(elastic, noise=0.0, seed=0, stretch=STRETCH):
    card = load_material({"density": 1000, "elastic": {**elastic, "poisson_ratio": 0.49}})
    s = curve(card, "uniaxial", stretch).stress.to_numpy()
    rng = np.random.default_rng(seed)
    return TestCurve(stretch, s + noise * s.max() * rng.standard_normal(len(s)))


@pytest.mark.parametrize("law, elastic, keys", [
    ("yeoh", {"model": "yeoh", "c10": 12e3, "c20": 300.0, "c30": 5.0}, ["elastic.c10", "elastic.c20", "elastic.c30"]),
    ("gent", {"model": "gent", "mu": 25e3, "jm": 40.0}, ["elastic.mu", "elastic.jm"]),
    ("arruda_boyce", {"model": "arruda_boyce", "mu": 25e3, "lambda_m": 3.5}, ["elastic.mu", "elastic.lambda_m"]),
    ("ogden1", {"model": "ogden", "mu": [20e3], "alpha": [2.6]}, ["elastic.mu[0]", "elastic.alpha[0]"]),
    ("mooney_rivlin", {"model": "mooney_rivlin", "c10": 10e3, "c01": 3e3}, ["elastic.c10", "elastic.c01"]),
])
def test_fit_recovers_parameters(law, elastic, keys):
    r = fit([_synthetic(elastic)], law, BASE)
    truth = load_material({"density": 1000, "elastic": {**elastic, "poisson_ratio": 0.49}})
    for key in keys:
        name, _, idx = key.split(".")[1].partition("[")
        expected = getattr(truth.elastic, name)
        expected = expected[int(idx[:-1])] if idx else expected
        assert r.values[key] == pytest.approx(expected, rel=1e-3, abs=1e-3 * abs(truth.elastic.initial_shear_modulus()))
    assert r.nrmse < 1e-4 and r.stable
    assert r.material.elastic.initial_poisson_ratio() == pytest.approx(0.49)


def test_neo_hookean_fit_stores_youngs_modulus_and_uncertainty_matches_scatter():
    truth_e = 60e3
    fits = [fit([_synthetic({"model": "neo_hookean", "youngs_modulus": truth_e}, noise=0.01, seed=s)],
                "neo_hookean", BASE) for s in range(25)]
    e = np.array([f.values["elastic.youngs_modulus"] for f in fits])
    sigma = np.median([f.sigma["elastic.youngs_modulus"] for f in fits])
    assert e.mean() == pytest.approx(truth_e, rel=2e-3)
    assert e.std() == pytest.approx(sigma, rel=0.5)  # predicted spread matches the actual one
    card = with_fit_uncertainty(fits[0])
    assert card.uncertainty["elastic.youngs_modulus"].rel == pytest.approx(fits[0].sigma["elastic.youngs_modulus"] / e[0])


def test_stability_detection():
    unstable = load_material({"density": 1000, "elastic": {"model": "mooney_rivlin", "c10": 10e3, "c01": -4e3,
                                                           "poisson_ratio": 0.49}})
    assert stability_limits(unstable, 3.0)  # negative c01: loses stability somewhere in [0.4, 3]
    gent = load_material({"density": 1000, "elastic": {"model": "gent", "mu": 10e3, "jm": 30.0, "poisson_ratio": 0.49}})
    assert stability_limits(gent, 3.0) == {}  # stiffening towards the chain limit is not an instability


def test_projection():
    yeoh = load_material({"name": "y", "density": 1000, "damping": {"model": "prony", "g": [0.3], "tau": [0.01]},
                          "elastic": {"model": "yeoh", "c10": 12e3, "c20": 300.0, "poisson_ratio": 0.49}})
    same = project(yeoh, "yeoh")
    assert same.nrmse < 1e-4
    nh = project(yeoh, "neo_hookean", (-0.3, 0.5))
    assert nh.material.elastic.model == "stable_neo_hookean" and nh.material.damping == yeoh.damping
    assert 0 < nh.nrmse < 0.05 and nh.material.provenance["projected_from"] == "y"
    assert nh.material.name == "y_as_neo_hookean"


CSV = """PARAMETER;INFO;URL;;
Specimen Thickness (mm);3;;;
Material;TestGel;;;
;;;;
Time (s);True Strain;True Stress (MPa);Engineering Strain;Engineering Stress (MPa)
""" + "\n".join(f"{0.1 * i:.1f};0;0;{0.05 * i:.4f};{(0.002 + 0.03 * 0.05 * i if i < 60 else 0.003) :.6f}"
                for i in range(1, 70))


def test_parse_srmd_csv_offset_and_rupture():
    c = parse_srmd_csv(CSV)
    assert c.meta["Material"] == "TestGel"
    assert c.stress[0] == pytest.approx(0.03e6 * c.stretch[0] - 0.03e6, abs=1.0)  # 2 kPa offset removed
    assert len(c.stretch) == 59  # ends at the maximum (rupture after point 59)
    assert np.all(np.diff(c.stretch) > 0)


def test_material_test_dataset(tmp_path):
    raw = tmp_path / "data" / "raw" / "tg"
    raw.mkdir(parents=True)
    archive = raw / "tg.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("db/TestGel.csv", CSV)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    (tmp_path / "datasets").mkdir()
    spec = f"""kind: material_tests
name: tg
title: Test gels
authors: [Nobody]
doi: 10.0/x
license: CC0
url: https://example.org
local_dir: data/raw/tg
format: srmd_csv
source: {{url: https://example.org/tg.zip, file: tg.zip, md5: {md5}, member_prefix: db/}}
conditions: {{mode: uniaxial, strain_rate: 0.5 1/s}}
materials:
  test_gel: {{member: TestGel.csv, product: Test Gel, density: 1.1 g/cm^3}}
"""
    (tmp_path / "datasets" / "tg.yaml").write_text(spec)
    ds = load_material_tests(tmp_path / "datasets" / "tg.yaml")
    assert ds.fetch() == archive.parent
    c = ds.curve("test_gel")
    assert c.material == "test_gel" and c.mode == "uniaxial" and c.rate == pytest.approx(0.5, rel=1e-6)
    with pytest.raises(ConfigError, match="no material"):
        ds.curve("nope")
    (tmp_path / "datasets" / "tg.yaml").write_text(spec.replace(md5, "0" * 32))
    with pytest.raises(ConfigError, match="md5"):
        load_material_tests(tmp_path / "datasets" / "tg.yaml").fetch()


def test_parse_xy_csv_tension_offset_and_compression_order():
    from softsim.experiments.material_tests import parse_xy_csv

    strain = np.linspace(0, 100, 201)                                    # %
    stress = 0.1 * strain / 100 + 0.012                                  # MPa, 12 kPa preload
    stress[(strain > 0) & (strain < 1)] += 0.05                          # gripping transient
    text = "Strain (%),Stress (MPa)\n" + "\n".join(f"{e},{s}" for e, s in zip(strain, stress))
    t = parse_xy_csv(text, "tension")
    assert t.stretch[0] >= 1.05 and np.allclose(t.stress, 1e5 * (t.stretch - 1), atol=1.0)
    c = parse_xy_csv("Strain (%),Stress (MPa)\n" + "\n".join(f"{e / 4},{0.05 * e / 100}" for e in strain), "compression")
    assert np.all(np.diff(c.stretch) > 0) and c.stretch.min() == pytest.approx(0.75) and np.all(c.stress <= 0)


def test_parse_relaxation_skips_the_lagging_overshoot_and_fit_recovers_prony():
    from softsim.experiments.material_tests import parse_relaxation_csv
    from softsim.materials.fit import fit_relaxation

    t = np.r_[np.linspace(0, 1.0, 1001), np.linspace(1.001, 2000, 20000)]
    strain = np.where(t < 1.0, 25 * t, 25.0)
    hold = t >= 1.3                                                    # stress peaks 0.3 s late
    s_hold = 0.05 * (1 - 0.03 * (1 - np.exp(-(t - 1.3) / 10.0)) - 0.02 * (1 - np.exp(-(t - 1.3) / 300.0)))
    stress = np.where(hold, s_hold, 0.06 * np.minimum(t / 1.3, 1.0))   # overshoot peak just before the hold
    text = "Force (N),Strain (%),Stress (MPa),Time (s)\n" + "\n".join(
        f"0,{e},{s},{tt}" for e, s, tt in zip(strain, stress, t))
    r = parse_relaxation_csv(text)
    assert r.time[0] == 0.0 and r.meta["hold_strain"] == pytest.approx(0.25) and np.all(r.stress < 0)
    g, tau, rms = fit_relaxation([r], taus=[1.0, 10.0, 100.0, 300.0, 1000.0])
    assert sum(g) == pytest.approx(0.05, abs=0.01) and rms < 1e-3  # 3% + 2% (minus the first second)
    assert dict(zip(tau, g)).get(10.0, 0) == pytest.approx(0.03, abs=0.006)


def test_fit_material_multi_test_dataset(tmp_path):
    """Tension + compression specimens of a synthetic Yeoh solid: recovered jointly, compression only when asked."""
    from softsim.materials.fit_cli import fit_material

    truth = {"model": "yeoh", "c10": 15e3, "c20": 400.0, "c30": 0.0}
    tens = _synthetic(truth, stretch=np.linspace(1.0, 2.5, 120))
    comp = _synthetic(truth, stretch=np.linspace(0.7, 1.0, 60))

    def csv(strain, stress):  # Strain (%), Stress (MPa), both positive
        return "Strain (%),Stress (MPa)\n" + "\n".join(f"{100 * e:.5f},{s / 1e6:.9f}" for e, s in zip(strain, stress))

    raw = tmp_path / "data" / "raw" / "mt"
    raw.mkdir(parents=True)
    md5 = {}
    for kind, text in [("tension", [csv(tens.stretch - 1, tens.stress * f) for f in (0.98, 1.02)]),
                       ("compression", [csv(1 - comp.stretch[::-1], -comp.stress[::-1])])]:
        with zipfile.ZipFile(raw / f"{kind}.zip", "w") as z:
            for i, t in enumerate(text):
                z.writestr(f"{kind}/s{i}.csv", t)
        md5[kind] = hashlib.md5((raw / f"{kind}.zip").read_bytes()).hexdigest()
    (tmp_path / "datasets").mkdir()
    spec = f"""kind: material_tests
name: mt
title: Multi-test gel
authors: [Nobody]
doi: 10.0/y
license: CC0
url: https://example.org
local_dir: data/raw/mt
format: xy_csv
sources:
  tension: {{url: https://example.org/t.zip, file: tension.zip, md5: {md5['tension']}, member_prefix: tension/}}
  compression: {{url: https://example.org/c.zip, file: compression.zip, md5: {md5['compression']}, member_prefix: compression/}}
materials:
  gel:
    tests:
      - {{source: tension, kind: tension, members: [s0.csv, s1.csv]}}
      - {{source: compression, kind: compression, members: [s0.csv]}}
"""
    (tmp_path / "datasets" / "mt.yaml").write_text(spec)
    both = fit_material(tmp_path / "datasets" / "mt.yaml", "gel", ["yeoh"], (-0.25, 1.4), out=tmp_path / "both")
    (fit,) = both["fits"]
    assert fit["values"]["elastic.c10"] == pytest.approx(15e3, rel=0.03) and fit["stable"]
    card = load_material(fit["file"])
    assert card.validity.strain == pytest.approx([-0.25, 1.4])
    assert "2 specimen" not in card.provenance["uncertainty_note"] and "3 specimen" in card.provenance["uncertainty_note"]
    tension_only = fit_material(tmp_path / "datasets" / "mt.yaml", "gel", ["yeoh"], (0.0, 1.4), out=tmp_path / "t")
    assert "2 specimen" in load_material(tension_only["fits"][0]["file"]).provenance["uncertainty_note"]
    assert (tmp_path / "both" / "fit.png").exists()

"""Homogeneous material tests against closed forms (needs the fem extra)."""

import numpy as np
import pytest

pytest.importorskip("jax")

from softsim.materials import load_material  # noqa: E402
from softsim.materials.homogeneous import constant_rate, curve, dma, dma_small_strain, relaxation  # noqa: E402
from softsim.materials.models import ARRUDA_BOYCE_C  # noqa: E402

MU = 30e3
KAPPA = MU * 1e5  # nearly incompressible: closed forms are the incompressible ones
LAM = np.array([0.5, 0.8, 1.2, 1.5, 2.0, 3.0])
I1 = LAM**2 + 2.0 / LAM  # uniaxial, incompressible


def _card(elastic, damping=None):
    data = {"density": 1000, "elastic": {**elastic, "bulk_modulus": KAPPA}}
    if damping:
        data["damping"] = damping
    return load_material(data)


NH = {"model": "neo_hookean", "shear_modulus": MU}


def _nh():
    return load_material({"density": 1000, "elastic": {"model": "neo_hookean", "shear_modulus": MU,
                                                       "bulk_modulus": KAPPA}})


@pytest.mark.parametrize("mode, exact", [
    ("uniaxial", lambda l: MU * (l - l**-2.0)),
    ("equibiaxial", lambda l: MU * (l - l**-5.0)),
    ("planar", lambda l: MU * (l - l**-3.0)),
])
def test_neo_hookean_modes(mode, exact):
    d = curve(_nh(), mode, LAM)
    np.testing.assert_allclose(d.stress, exact(LAM), rtol=2e-4)


def test_neo_hookean_shear_volumetric_lateral():
    m = _nh()
    g = np.array([0.1, 1.0, 2.0])
    np.testing.assert_allclose(curve(m, "simple_shear", g).stress, MU * g, rtol=1e-9)
    np.testing.assert_allclose(curve(m, "uniaxial", LAM).lateral, LAM**-0.5, rtol=1e-4)
    j = np.array([0.98, 1.02])
    np.testing.assert_allclose(curve(m, "volumetric", j).stress, 0.5 * KAPPA * (j - 1 / j), rtol=1e-9)


@pytest.mark.parametrize("elastic, dw_di1", [
    ({"model": "mooney_rivlin", "c10": 12e3, "c01": 3e3}, None),
    ({"model": "yeoh", "c10": 12e3, "c20": 800.0, "c30": 20.0},
     lambda i1: 12e3 + 2 * 800.0 * (i1 - 3) + 3 * 20.0 * (i1 - 3) ** 2),
    ({"model": "gent", "mu": MU, "jm": 20.0}, lambda i1: 0.5 * MU * 20.0 / (20.0 - i1 + 3)),
    ({"model": "arruda_boyce", "mu": MU, "lambda_m": 3.0},
     lambda i1: MU * sum(k * c * i1 ** (k - 1) / 3.0 ** (2 * k - 2) for k, c in enumerate(ARRUDA_BOYCE_C, 1))),
])
def test_invariant_laws_uniaxial(elastic, dw_di1):
    d = curve(_card(elastic), "uniaxial", LAM)
    if elastic["model"] == "mooney_rivlin":
        exact = 2 * (LAM - LAM**-2.0) * (12e3 + 3e3 / LAM)
    else:
        exact = 2 * (LAM - LAM**-2.0) * dw_di1(I1)
    np.testing.assert_allclose(d.stress, exact, rtol=2e-4)


def test_ogden_uniaxial():
    mu, al = (20e3, 1e3), (1.3, 5.0)
    d = curve(_card({"model": "ogden", "mu": list(mu), "alpha": list(al)}), "uniaxial", LAM)
    exact = sum(m * (LAM ** (a - 1) - LAM ** (-a / 2 - 1)) for m, a in zip(mu, al))
    np.testing.assert_allclose(d.stress, exact, rtol=2e-4)


@pytest.mark.parametrize("elastic", [
    {"model": "neo_hookean", "youngs_modulus": 60e3, "poisson_ratio": 0.45},
    {"model": "mooney_rivlin", "c10": 8e3, "c01": 2e3, "poisson_ratio": 0.45},
    {"model": "ogden", "mu": [20e3, 1e3], "alpha": [1.3, 5.0], "poisson_ratio": 0.45},
    {"model": "gent", "mu": 20e3, "jm": 10.0, "poisson_ratio": 0.45},
    {"model": "arruda_boyce", "mu": 20e3, "lambda_m": 3.0, "poisson_ratio": 0.45},
])
def test_small_strain_moduli(elastic):
    m = load_material({"density": 1000, "elastic": elastic})
    eps = 1e-5
    e = curve(m, "uniaxial", [1 - eps, 1 + eps]).stress.to_numpy()
    assert (e[1] - e[0]) / (2 * eps) == pytest.approx(m.elastic.initial_youngs_modulus(), rel=1e-6)
    g = curve(m, "simple_shear", [eps]).stress.iloc[0] / eps
    assert g == pytest.approx(m.elastic.initial_shear_modulus(), rel=1e-6)
    lat = curve(m, "uniaxial", [1 + eps]).lateral.iloc[0]
    assert (1 - lat) / eps == pytest.approx(m.elastic.initial_poisson_ratio(), rel=1e-4)


PRONY = {"model": "prony", "g": [0.5, 0.2], "tau": [0.01, 1.0]}


def test_prony_relaxation_and_limits():
    m = _card(NH, PRONY)
    strain = 0.01
    r = relaxation(m, strain, ramp=1e-5, hold=30.0)
    inst = curve(m, "uniaxial", [1 + strain], "instantaneous").stress.iloc[0]
    hold = r[r.t > 2e-5]
    t = hold.t.to_numpy() - 1e-5
    expected = 1 - 0.5 * (1 - np.exp(-t / 0.01)) - 0.2 * (1 - np.exp(-t / 1.0))
    np.testing.assert_allclose(hold.stress / inst, expected, atol=4e-3)
    relaxed = curve(m, "uniaxial", [1 + strain], "relaxed").stress.iloc[0]
    assert r.stress.iloc[-1] == pytest.approx(relaxed, rel=2e-3)
    assert relaxed == pytest.approx(0.3 * inst, rel=1e-4)  # only the (tiny) volumetric part doesn't relax


def test_prony_rate_dependence():
    m = _card(NH, PRONY)
    slow = constant_rate(m, "uniaxial", 1.5, rate=1e-3).stress.iloc[-1]
    fast = constant_rate(m, "uniaxial", 1.5, rate=1e5).stress.iloc[-1]  # 5 us << tau
    assert slow == pytest.approx(curve(m, "uniaxial", [1.5]).stress.iloc[0], rel=0.01)
    assert fast == pytest.approx(curve(m, "uniaxial", [1.5], "instantaneous").stress.iloc[0], rel=0.01)


def test_dma_matches_closed_form():
    m = _card(NH, PRONY)
    f = [0.05, 1.0, 16.0, 200.0]
    num, exact = dma(m, f), dma_small_strain(m, f)
    np.testing.assert_allclose(num.storage, exact.storage, rtol=5e-3)
    np.testing.assert_allclose(num.loss, exact.loss, rtol=1e-2, atol=1e-3 * exact.storage.max())
    shear = dma_small_strain(m, f, "simple_shear")
    np.testing.assert_allclose(exact.storage / shear.storage, 3.0, rtol=1e-4)  # incompressible: E = 3 G


def test_numerical_damping_ignored_and_unsupported_laws_refused():
    from softsim.errors import UnsupportedFeatureError

    tacky = load_material("ecoflex_00_10_drop_fit")  # Rayleigh beta: not a material response
    d = curve(tacky, "uniaxial", [1.2], "instantaneous")
    assert d.stress.iloc[0] == pytest.approx(curve(tacky, "uniaxial", [1.2], "relaxed").stress.iloc[0])
    with pytest.raises(UnsupportedFeatureError):
        curve(load_material("ecoflex_00_10", overrides=["elastic.model=stvk"]), "uniaxial", [1.1])

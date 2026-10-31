"""Prony viscoelasticity on the tetrahedral backends (explicit Maxwell overstress)."""

import numpy as np
import pytest

pytest.importorskip("jax")

from softsim.config.schema import validate  # noqa: E402
from softsim.errors import ConfigError  # noqa: E402
from softsim.materials import Material, load_material  # noqa: E402
from softsim.meshing import structured_box  # noqa: E402
from softsim.materials.viscoelastic import PronyOverstress, long_term_material  # noqa: E402

PRONY = {"model": "prony", "g": [0.5, 0.2], "tau": [0.005, 0.2]}


@pytest.mark.parametrize("elastic", [
    {"model": "neo_hookean", "youngs_modulus": 60e3, "poisson_ratio": 0.49},
    {"model": "stable_neo_hookean", "youngs_modulus": 60e3, "poisson_ratio": 0.45},
    {"model": "mooney_rivlin", "c10": 8e3, "c01": 2e3, "bulk_modulus": 1e6},
    {"model": "yeoh", "c10": 8e3, "c20": 300.0, "bulk_modulus": 1e6},
    {"model": "ogden", "mu": [10e3], "alpha": [2.5], "bulk_modulus": 1e6},
    {"model": "gent", "mu": 20e3, "jm": 30.0, "bulk_modulus": 1e6},
])
def test_long_term_law_relaxes_shear_only(elastic):
    m = validate(Material, {"density": 1000, "elastic": elastic, "damping": PRONY})
    lt = long_term_material(m)
    assert lt.damping is None and lt.elastic.model == m.elastic.model
    assert lt.elastic.initial_shear_modulus() == pytest.approx(0.3 * m.elastic.initial_shear_modulus())
    assert lt.elastic.initial_bulk_modulus() == pytest.approx(m.elastic.initial_bulk_modulus())


def test_prony_needs_a_split_law():
    with pytest.raises(ConfigError, match="isochoric"):
        validate(Material, {"density": 1000, "elastic": {"model": "stvk", "youngs_modulus": 1e4, "poisson_ratio": 0.4},
                            "damping": PRONY})


def test_overstress_forces_match_the_homogeneous_reference():
    """Under a uniform deformation the face forces of a meshed cube equal the overstress
    P_total - P_long-term of the homogeneous reference (same scheme), step by step."""
    from softsim.fem import jax_setup
    from softsim.materials.homogeneous import _potential, history

    jax, jnp = jax_setup()
    m = load_material({"density": 1000, "damping": PRONY,
                       "elastic": {"model": "yeoh", "c10": 10e3, "c20": 500.0, "poisson_ratio": 0.49}})
    t = np.r_[np.linspace(0, 0.004, 9), 0.004 + np.geomspace(1e-4, 0.5, 30)]
    x = 1.0 + 0.4 * np.minimum(t / 0.004, 1.0)
    h = history(m, "uniaxial", t, x)
    mesh = structured_box((0.01, 0.01, 0.01), 0.005)  # x in [-5, 5] mm
    ov = PronyOverstress(m, mesh.points, mesh.tets)
    face = np.isclose(mesh.points[:, 0], 0.005)
    area = 0.01 * 0.01

    from softsim.fem.materials import from_material
    law, kappa, _ = from_material(m)
    p_long = jax.jit(jax.grad(lambda f: _potential(law, kappa, f, 0.3, jnp.zeros((3, 3)))))
    for n in range(1, len(t)):
        f_grad = np.diag([x[n], h.lateral[n], h.lateral[n]])
        forces = ov.forces(mesh.points @ f_grad.T, t[n] - t[n - 1])
        p_over = h.stress[n] - float(p_long(jnp.asarray(f_grad))[0, 0])
        assert -forces[face, 0].sum() / area == pytest.approx(p_over, rel=1e-6, abs=1e-6 * h.stress.max())
        assert np.abs(forces.sum(axis=0)).max() < 1e-9 * abs(h.stress.max()) * area  # internal: no net force
    assert 0 < p_over < 0.03 * h.stress.max()  # after 0.5 s only the 0.2 s term is left (~e^-2.5 of its share)


def _drop_peak(backend, damping, extra=()):
    from softsim.runner import Simulation

    ov = [f"backend={backend}", "simulation.duration=0.01", "specimen.mesh.size=0.005",
          "simulation.timestep=0.00003" if backend == "mujoco" else "simulation.timestep=0.00005",
          f"specimen.material.damping={damping}", *extra]
    sim = Simulation("scenarios/drop_test/cases/w050.yaml", ov)
    try:
        return sim.run().summary["peak_indentation"]
    finally:
        sim.close()


def _sofa_available():
    try:
        from softsim.backends.sofa.backend import import_sofa

        import_sofa()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("backend", ["mujoco", pytest.param("sofa", marks=pytest.mark.skipif(
    not _sofa_available(), reason="SOFA not available"))])
def test_prony_drop_between_instantaneous_and_long_term(backend):
    prony = "{model: prony, g: [0.6], tau: [2 ms]}"
    inst = _drop_peak(backend, "null")
    long_term = _drop_peak(backend, "null", ["specimen.material.elastic.youngs_modulus=37 kPa"])
    visco = _drop_peak(backend, prony)
    assert inst < visco < long_term
    assert visco == pytest.approx(5.38e-3, rel=0.05)  # MuJoCo 5.38 mm, SOFA 5.39 mm when written


def test_prony_from_fractional_kelvin_voigt():
    """The Prony branches follow mu0 (1 + (i w tau)^n), normalised at the test rate, inside the band."""
    import math

    from softsim.materials.homogeneous import dma_small_strain
    from softsim.materials.viscoelastic import prony_from_fractional, with_fast_relaxation

    n, tau = 0.2, 2.9e-3
    a, taus, rms = prony_from_fractional(n, tau, band=(1.0, 2000.0))
    assert rms < 0.02 and min(taus) >= 1.0 / (2 * math.pi * 2000.0) * 0.999
    base = load_material({"density": 1070, "elastic": {"model": "yeoh", "c10": 27e3, "c20": -900.0, "c30": 150.0,
                                                       "poisson_ratio": 0.49},
                          "damping": {"model": "prony", "g": [0.02], "tau": [100.0]}})
    card = with_fast_relaxation(base, a, taus)
    frac = lambda f: 1 + (1j * 2 * math.pi * f * tau) ** n  # noqa: E731
    f = np.array([0.06, 3.0, 30.0, 300.0, 1500.0])
    d = dma_small_strain(card, f, mode="simple_shear")
    d0 = dma_small_strain(base, f, mode="simple_shear")
    np.testing.assert_allclose(d.storage / d.storage.iloc[0], [(frac(x) / frac(0.06)).real for x in f], rtol=0.03)
    assert d.storage.iloc[0] == pytest.approx(d0.storage.iloc[0], rel=0.01)  # the test-rate response is kept
    # long-term modulus and slow relaxation unchanged
    lt = lambda m: m.elastic.initial_shear_modulus() * (1 - sum(m.damping.g))  # noqa: E731
    assert lt(card) == pytest.approx(lt(base), rel=1e-9)
    assert card.elastic.c10 == pytest.approx(base.elastic.c10 * (1 + sum(a)))

"""Reference axisymmetric FEM: analytic checks (needs JAX; skipped otherwise)."""

import math

import numpy as np
import pytest

pytest.importorskip("jax")

from softsim.fem.axisym import AxisymSolver, Problem, _tributary_areas, cylinder_mesh  # noqa: E402
from softsim.fem.materials import MooneyRivlinIso, NeoHookeanIso, OgdenIso, Prony, YeohIso, from_material  # noqa: E402
from softsim.config.schema import validate  # noqa: E402
from softsim.materials import Material  # noqa: E402

MU, R, H = 20e3, 0.02, 0.017


def test_mesh_volume_and_tributary_areas():
    mesh = cylinder_mesh(R, H, 6, 5, grading=1.7)
    assert mesh.volume() == pytest.approx(math.pi * R**2 * H, rel=1e-12)
    top = _tributary_areas(mesh.nodes, mesh.edges["top"])
    assert sum(top.values()) == pytest.approx(math.pi * R**2, rel=1e-12)
    axis = min(top, key=lambda i: mesh.nodes[i, 0])
    assert top[axis] == pytest.approx(math.pi * (R / 6 / 2 / 2) ** 2)  # disc out to the first half-segment's middle
    side = _tributary_areas(mesh.nodes, mesh.edges["side"])
    assert sum(side.values()) == pytest.approx(2 * math.pi * R * H, rel=1e-12)


def _compress(law, kappa, strain, base="slide", top_bonded=False, cells=(4, 3), prony=None, dynamic=False):
    prob = Problem(cylinder_mesh(R, H, *cells), law, kappa, 1000.0, prony, gravity=0.0, base=base,
                   top_displacement=lambda t: -strain * H * min(t, 1.0), top_bonded=top_bonded)
    s = AxisymSolver(prob, dynamic=dynamic)
    assert all(s.advance(0.1) for _ in range(10))
    return s


@pytest.mark.parametrize("bulk_ratio", [1e3, 1e5])
def test_frictionless_uniaxial_compression_is_exact(bulk_ratio):
    """Homogeneous 30% compression: incompressible Neo-Hookean P = mu (lambda - lambda^-2)."""
    s = _compress(NeoHookeanIso(MU), bulk_ratio * MU, 0.3)
    lam = 0.7
    assert s.top_reaction() == pytest.approx(MU * (lam - lam**-2) * -math.pi * R**2, rel=2e-3)
    assert s.current_nodes()[:, 0].max() == pytest.approx(R / math.sqrt(lam), rel=2e-3)  # volume conserved


LAM = 0.8  # 20% compression


@pytest.mark.parametrize("law,nominal_stress", [
    (MooneyRivlinIso(MU / 4, MU / 4), 2 * (LAM - LAM**-2) * (MU / 4 + MU / 4 / LAM)),
    (YeohIso(MU / 2, MU / 10, MU / 50), 2 * (LAM - LAM**-2) * sum(
        k * c * (LAM**2 + 2 / LAM - 3) ** (k - 1) for k, c in ((1, MU / 2), (2, MU / 10), (3, MU / 50)))),
    (OgdenIso((MU / 1.5,), (3.0,)), MU / 1.5 * (LAM**2 - LAM**-2.5)),
], ids=["mooney_rivlin", "yeoh", "ogden"])
def test_uniaxial_compression_of_each_law(law, nominal_stress):
    """20% frictionless compression vs each law's exact incompressible uniaxial stress."""
    assert law.shear_modulus == pytest.approx(MU)
    s = _compress(law, 1e4 * MU, 1 - LAM)
    assert s.top_reaction() / (math.pi * R**2) == pytest.approx(-nominal_stress, rel=2e-3)


def test_bonded_compression_does_not_lock():
    """Bonded disc, nu -> 0.5: converges with the mesh to about Gent's E_c = E (1 + 2 S^2), S = R / 2H."""
    moduli = []
    for cells in ((4, 3), (8, 6), (16, 12)):
        s = _compress(NeoHookeanIso(MU), 1e5 * MU, 0.01, base="bonded", top_bonded=True, cells=cells)
        moduli.append(s.top_reaction() / (math.pi * R**2 * 0.01) / (3 * MU))
    changes = np.abs(np.diff(moduli)) / moduli[1:]
    assert changes[1] < changes[0] / 2  # converging (a locking element would stiffen without bound instead)
    assert moduli[-1] == pytest.approx(1 + 2 * (R / (2 * H)) ** 2, rel=0.05)


def test_prony_relaxation_matches_linear_viscoelasticity():
    """Ramp (20 ms) to 1% compression and hold: sigma(t) from the hereditary integral of G(t)."""
    g, tau, t_r, eps = 0.5, 0.1, 0.02, 0.01
    prob = Problem(cylinder_mesh(R, H, 4, 3), NeoHookeanIso(MU), 1e3 * MU, 1000.0, Prony((g,), (tau,)), gravity=0.0,
                   base="slide", top_displacement=lambda t: -eps * H * min(t / t_r, 1.0))
    s = AxisymSolver(prob, dynamic=True, numerical_damping=0.1)
    rate = eps / t_r

    def analytic(t):
        return 3 * rate * MU * ((1 - g) * t_r + g * tau * (math.exp(-(t - t_r) / tau) - math.exp(-t / tau)))

    for t in (0.05, 0.15, 0.4):
        while s.t < t - 1e-12:
            assert s.advance(min(1e-3, t - s.t))
        assert s.top_reaction() / (math.pi * R**2) == pytest.approx(analytic(t), rel=0.03)


def test_material_mapping():
    nh = validate(Material, {"density": 1000, "elastic": {"model": "stable_neo_hookean", "youngs_modulus": 60e3,
                                                         "poisson_ratio": 0.45},
                             "damping": {"model": "prony", "g": [0.3], "tau": [0.01]}})
    law, kappa, prony = from_material(nh)
    assert law.shear_modulus == pytest.approx(60e3 / 2.9) and kappa == pytest.approx(60e3 / (3 * 0.1))
    assert prony.g == (0.3,) and prony.tau == (0.01,)


@pytest.mark.slow
def test_reference_drop_smoke():
    """Short coarse 50 g drop: contact, deceleration, no energy gain; asymmetric impacts are refused."""
    from softsim.errors import ConfigError
    from softsim.fem.drop import FemSettings, run_drop

    case = "scenarios/drop_test/cases/w050.yaml"
    over = ["drop.initial_speed=2.47", "drop.gap=0.0005", "simulation.duration=0.004", "simulation.output_rate=2000",
            "specimen.material={density: 1040, elastic: {model: neo_hookean, youngs_modulus: 100 kPa, poisson_ratio: 0.49}}"]
    ts, info = run_drop(case, over, FemSettings(cells_r=8, cells_z=5))
    assert ts.contact_force_z.max() > 1.0                      # the punch hit the specimen
    assert 0 > ts.weight_vz.iloc[-1] > -2.47                    # and is being decelerated
    assert ts.specimen_min_z.min() > -2e-4                      # no sinking through the ground
    with pytest.raises(ConfigError, match="centred"):
        run_drop(case, [*over, "drop.offset=[0.006, 0]"], FemSettings(cells_r=8, cells_z=5))


@pytest.mark.slow
def test_coulomb_friction_lies_between_frictionless_and_stick():
    """Static punch, 10 N: mu = 0 is frictionless, indentation falls with mu, large mu approaches stick."""
    import dataclasses
    from pathlib import Path

    from softsim.config.loader import apply_overrides, load_yaml
    from softsim.fem.drop import FemSettings, build_problem

    raw = apply_overrides(load_yaml("scenarios/drop_test/cases/w050.yaml"), [
        "specimen.material={density: 1040, elastic: {model: neo_hookean, youngs_modulus: 100 kPa, poisson_ratio: 0.49}}",
        "weight.body={preset: czopek2026_w050, mass: 1.0}", "drop.gap=0.00001", "drop.initial_speed=0"])
    depth = {}
    for friction in ("frictionless", 0.0, 0.3, 10.0, "stick"):
        prob, _ = build_problem(raw, Path("scenarios/drop_test/cases"), FemSettings(cells_r=6, cells_z=4, friction=friction))
        prob.punch = dataclasses.replace(prob.punch, z0=H - 1e-6)  # static: start in contact
        prob.gravity = 0.0
        s = AxisymSolver(prob, dynamic=False)
        while s.p.gravity < 10.0 - 1e-9:  # 10 N on the 1 kg punch, in 1 N steps
            s.p.gravity = min(10.0, s.p.gravity + 1.0)
            assert s.advance(1.0)
        depth[friction] = H - s.zp
    assert depth[0.0] == pytest.approx(depth["frictionless"], rel=1e-9)
    assert depth["frictionless"] > depth[0.3] > depth[10.0]
    assert depth[10.0] == pytest.approx(depth["stick"], rel=0.01)

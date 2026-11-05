"""``body.plane_strain``: a wide strip held in its width direction is stiffer in bending by 1 / (1 - nu^2)."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import UnsupportedFeatureError
from softsim.scenarios.inflation import InflationConfig, InflationScenario

BEAM = Path(__file__).parents[1] / "scenarios" / "actuation" / "tip_load_cantilever.yaml"
NU = 0.45  # the card's Poisson ratio


def _scene(overrides=()):
    sc = InflationScenario(validate(InflationConfig, apply_overrides(load_yaml(BEAM), list(overrides))), BEAM.parent)
    return sc.build_scene()


def _tip(overrides=()):
    """Static tip deflection [m] of the 5 mN load on the 10 x 10 x 60 mm beam (reference, 2 mm mesh)."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    scene = _scene(["body.mesh.size=2mm", *overrides])
    b = FemBackend()
    b.build(scene)
    s = Solid3DSolver(b._problem, dynamic=False)
    s.forces[:] = 5e-3
    assert s.advance(1.0)
    tip = np.abs(s.p.mesh.nodes[:, 2] - 0.06) < 1e-9
    return float(s.y[:3 * s.n].reshape(-1, 3)[tip, 0].mean())


def test_plane_strain_stiffens_the_beam_by_one_minus_nu_squared():
    """Euler-Bernoulli with E' = E / (1 - nu^2) (width y held): the tip deflection falls by 1 - nu^2 = 0.7975.
    The 1:6 section is not a thin plate, so the match is to a few percent (measured 0.8106 at 2 mm, +1.6%)."""
    free, held = _tip(), _tip(["body.plane_strain=y"])
    assert held / free == pytest.approx(1 - NU**2, rel=0.04)
    assert free == pytest.approx(0.864e-3, rel=0.02)  # the unconstrained beam still hits Euler-Bernoulli


def test_plane_strain_lowers_to_a_zero_prescribed_motion():
    """MuJoCo and SOFA have no plane-strain option: the scene is lowered to displacement control of every vertex
    that is not pinned, holding the axis only (the others move freely)."""
    scene = _scene(["body.plane_strain=y"])
    body = scene.soft_bodies[0]
    assert body.plane_strain == 1
    low = scene.lowered().soft_bodies[0]
    (hold,) = low.prescribed
    assert low.plane_strain is None and hold.axes == (False, True, False)
    assert set(hold.vertices.tolist()) | set(low.pinned) == set(range(body.mesh.n_points))
    assert not set(hold.vertices.tolist()) & set(low.pinned)
    assert hold.path.position(0.3).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.slow  # SOFA: 12 min, MuJoCo: 2 min
@pytest.mark.parametrize("backend", ["mujoco", "sofa"])
def test_plane_strain_on_the_fast_backends_stiffens_like_the_reference(backend):
    """Same ratio test through the lowered scene: the tip deflection with the width held over the free one is
    1 - nu^2 within 6% (linear tets at 2 mm, quasi-static ramp)."""
    if backend == "sofa":
        from softsim.backends.registry import get_backend

        try:
            get_backend("sofa").check(_scene())
        except Exception:
            pytest.skip("SOFA not available")
    from softsim.runner import Simulation

    def tip(extra):
        sim = Simulation(BEAM, ["body.mesh.size=2mm", "output.save_model=false",
                                *extra], backend=backend)
        return abs(sim.run().timeseries["tip_force_displacement"].iloc[-1])

    ratio = tip(["body.plane_strain=y"]) / tip([])
    assert ratio == pytest.approx(1 - NU**2, rel=0.06)

"""Region forces: the load's distribution, the reference against Timoshenko, every backend's total."""

from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import ForceActuator, region_force_weights
from softsim.analytic import cantilever_tip_deflection
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.scenarios.inflation import InflationConfig, InflationScenario

BEAM = Path(__file__).parents[1] / "scenarios" / "actuation" / "tip_load_cantilever.yaml"


def _beam(overrides=()):
    raw = apply_overrides(load_yaml(BEAM), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), BEAM.parent)


def test_force_config_and_distribution():
    scene = _beam().build_scene()
    (force,) = scene.actuators
    assert isinstance(force, ForceActuator) and force.direction == (1.0, 0.0, 0.0)
    assert force.value(0.2) == pytest.approx(0.0025) and force.value(1.0) == pytest.approx(0.005)
    body = scene.soft_bodies[0]
    w = region_force_weights(body.mesh.points, body.region("top"))
    assert w.sum() == pytest.approx(1.0) and np.all(body.mesh.points[w > 0, 2] == pytest.approx(0.06))
    # uniform traction: the load's centre is the face's centre
    np.testing.assert_allclose(w @ body.mesh.points, [0, 0, 0.06], atol=1e-12)
    d = _beam(["actuators[0].direction=[0, 3, 4]"]).build_scene().actuators[0].direction
    np.testing.assert_allclose(d, [0, 0.6, 0.8])
    with pytest.raises(ConfigError, match="no region 'lid'"):
        _beam(["actuators[0].region=lid"]).build_scene()
    with pytest.raises(Exception, match="non-zero"):
        _beam(["actuators[0].direction=[0, 0, 0]"])


def test_reference_matches_timoshenko():
    """Quasi-static reference, 2.5 mm quadratic tets, 5 mN on the top face: the face's displacement
    along the force lies between Euler-Bernoulli bending and Timoshenko (bending + shear, 2.4% more):
    the fully clamped root (no warping, no Poisson contraction) offsets the shear on this short beam
    (L/h = 6). Measured: 2.5 / 2 / 1.25 mm -> +0.71 / +0.36 / +0.00% of Euler-Bernoulli. It doubles
    with the force (linear range)."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    sc = _beam(["body.mesh.size=2.5mm"])
    backend = FemBackend()
    backend.build(sc.build_scene())
    s = Solid3DSolver(backend._problem, dynamic=False)
    backend.solver = s
    out = []
    for force in (0.0025, 0.005):
        s.forces[:] = [force]
        assert s.advance(1.0)
        out.append(sc.metrics(backend.state())["tip_force_displacement"])
    bending = 0.005 * 0.06**3 / (3 * 5e5 * 0.01**4 / 12)
    assert 0.995 * bending < out[1] < cantilever_tip_deflection(0.005, 0.06, 5e5, 0.45, 0.01, 0.01)
    assert out[1] == pytest.approx(bending, rel=0.01)
    assert out[1] / out[0] == pytest.approx(2.0, rel=0.01)


@pytest.mark.parametrize("name", ["mujoco", "sofa"])
def test_explicit_backends_apply_the_total_force(name):
    """MuJoCo and SOFA: the vertex forces sum to the commanded force, along its direction."""
    from softsim.backends import get_backend
    from softsim.errors import SoftSimError

    try:
        backend = get_backend(name)()
        backend.build(_beam(["body.mesh.size=2.5mm", "actuators[0].direction=[1, 1, 0]"]).build_scene())
    except (ImportError, SoftSimError) as exc:
        pytest.skip(str(exc))
    backend.set_control("tip_force", 0.004)
    backend.step(1)
    if name == "mujoco":
        backend._apply_external()
        total = backend.data.xfrc_applied[:, :3].sum(axis=0)
    else:
        field, _ = backend._soft["beam"]["actuation"]
        total = np.asarray(field.forces.value).sum(axis=0)
    np.testing.assert_allclose(total, 0.004 * np.array([1, 1, 0]) / np.sqrt(2), atol=1e-12)
    backend.set_control("tip_force", -0.001)  # pushes the other way
    backend.step(300)
    assert backend.control("tip_force") == -0.001

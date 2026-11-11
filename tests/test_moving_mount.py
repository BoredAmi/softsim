"""Moving mount: the path, the mount's-frame formulation against a world-frame reference run with
prescribed base motion, and the explicit backends."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.scenarios.inflation import InflationConfig, InflationScenario
from softsim.scene import MountMotion

MOUNT = Path(__file__).parents[1] / "scenarios" / "actuation" / "moving_mount.yaml"


def _beam(overrides=()):
    raw = apply_overrides(load_yaml(MOUNT), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), MOUNT.parent)


def test_path():
    m = MountMotion((0.0, 0.2, 0.4), ((0, 0, 0), (0.02, 0, 0), (0.02, 0, 0.01)))
    for t in (0.0, 0.2, 0.4, 0.6):  # at rest at every waypoint, held outside
        assert np.all(m.velocity(t) == 0) and np.all(m.acceleration(t) == 0)
    np.testing.assert_allclose(m.position(0.1), [0.01, 0, 0])
    np.testing.assert_allclose(m.position(0.5), [0.02, 0, 0.01])
    h, t = 1e-6, 0.13
    np.testing.assert_allclose((m.position(t + h) - m.position(t - h)) / (2 * h), m.velocity(t), atol=1e-7)
    np.testing.assert_allclose((m.velocity(t + h) - m.velocity(t - h)) / (2 * h), m.acceleration(t), atol=1e-5)
    assert np.all(np.diff([m.position(t)[0] for t in np.linspace(0, 0.2, 41)]) >= 0)  # no overshoot
    with pytest.raises(ValueError, match="rest position"):
        MountMotion((0.0, 0.1), ((0.01, 0, 0), (0.02, 0, 0)))
    with pytest.raises(ConfigError, match="no ground"):
        _beam(["ground.enabled=true"]).build_scene()
    with pytest.raises(ConfigError, match="fix a region"):
        _beam(["body.fixed=[]"]).build_scene()


@pytest.mark.slow
def test_mount_frame_matches_world_frame_reference():
    """The reference two ways, dynamic, 2 ms: in the mount's frame (gravity g - a(t), what every backend
    does) and in the world with the base nodes driven along the path (Newmark-consistent prescribed
    motion). With the trapezoidal rule (second order) the tip trajectories agree to 0.10% of the tip's
    lag at 2 ms and 0.026% at 1 ms: the two formulations converge to one solution. (With the default
    numerically damped Newmark, first order, the gap is 2.2% at 2 ms and halves with dt.)"""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    ov = ["body.mesh.size=3mm", "simulation.timestep=2ms", "simulation.duration=300ms"]
    framed = _beam(ov)
    a = FemBackend()
    a.build(framed.build_scene())
    a.solver = Solid3DSolver(a._problem, numerical_damping=0.0)
    world = _beam([*ov, "body.mount_path=[]"])
    b = FemBackend()
    b.build(world.build_scene())
    b.solver = Solid3DSolver(b._problem, numerical_damping=0.0)
    mount = framed.build_scene().mount
    n_fixed = int(b.solver.fixed.sum())
    b.solver.fixed_motion = lambda t: np.tile(mount.position(t), n_fixed // 3)
    lag_a, lag_b = [], []
    for k in range(150):
        a.step(1)
        b.step(1)
        p = mount.position(a.time)[0]
        lag_a.append(framed.metrics(a.state())["tip_x"] - p)
        lag_b.append(world.metrics(b.state())["tip_x"] - p)
        if k == 0:  # the base moves with the mount in both
            base = framed.build_scene().soft_bodies[0].pinned
            np.testing.assert_allclose(a.state().soft["beam"].vertices[list(base)],
                                       b.state().soft["beam"].vertices[list(base)], atol=1e-12)
    lag_a, lag_b = np.array(lag_a), np.array(lag_b)
    scale = np.abs(lag_b).max()
    assert scale > 1e-3  # the tip lags by millimetres while the mount accelerates
    assert np.abs(lag_a - lag_b).max() < 0.003 * scale
    assert lag_b[np.argmax(np.abs(lag_b[:50]))] < 0  # lags behind first (the mount accelerates to +x)


@pytest.mark.parametrize("name", ["mujoco", "sofa"])
def test_explicit_backends_follow_the_mount(name):
    from softsim.backends import get_backend
    from softsim.errors import SoftSimError

    sc = _beam(["body.mesh.size=3mm", "simulation.timestep=1ms" if name == "sofa" else "simulation.timestep=0.1ms"])
    try:
        backend = get_backend(name)()
        backend.build(sc.build_scene())
    except (ImportError, SoftSimError) as exc:
        pytest.skip(str(exc))
    scene = sc.build_scene()
    rest = scene.soft_bodies[0].mesh.points
    pinned = list(scene.soft_bodies[0].pinned)
    steps = int(round(0.04 / scene.simulation.timestep))
    backend.step(steps)  # the mount accelerating: the base is on the path, the tip lags behind
    s = backend.state()
    np.testing.assert_allclose(s.soft["beam"].vertices[pinned], rest[pinned] + scene.mount.position(s.time), atol=1e-9)
    np.testing.assert_allclose(s.soft["beam"].velocities[pinned],
                               np.tile(scene.mount.velocity(s.time), (len(pinned), 1)), atol=1e-9)
    assert sc.metrics(s)["tip_x"] < scene.mount.position(s.time)[0] - 1e-4

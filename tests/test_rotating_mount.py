"""Rotating mount: the mount frame's kinematics against numerical differentiation of an arbitrary world
motion, then (per backend) a cantilever turned under gravity, a bar's centrifugal stretch, the Foucault
plane (Coriolis) and the inertia at the start of a spin-up (Euler)."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.scenarios.inflation import InflationConfig, InflationScenario
from softsim.scene import MountMotion, MountSpin, mount_frame

SPIN = MountSpin(axis=tuple(np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])),
                 center=(0.01, -0.02, 0.03), times=(0.0, 0.2, 0.5), angles=(0.0, 1.2, 0.4))
SLIDE = MountMotion((0.0, 0.3, 0.6), ((0, 0, 0), (0.02, -0.01, 0.015), (0.0, 0.0, 0.0)))


def _world_path(t):
    """An arbitrary smooth world trajectory of a free particle (so a_world is known analytically)."""
    x0, v0, g = np.array([0.03, 0.01, 0.08]), np.array([0.1, -0.2, 0.05]), np.array([0.0, 0.0, -9.81])
    return x0 + v0 * t + 0.5 * g * t * t, v0 + g * t, g


@pytest.mark.parametrize("slide", [None, SLIDE], ids=["rotation", "rotation+translation"])
def test_fictitious_accelerations_against_numerical_differentiation(slide):
    """A free particle moves on a known world trajectory; its mount-frame trajectory is
    x_m = R^T (x_w - c - p) + c. Differentiating that numerically twice must give the acceleration the
    frame says it has: R^T (a_w - p'') - alpha x r - omega x (omega x r) - 2 omega x v, i.e. gravity
    (``MountFrame.gravity``) plus ``MountFrame.fictitious``: every sign and factor of Euler,
    centrifugal and Coriolis, at once."""
    def mount_position(t):
        f = mount_frame(slide, SPIN, t)
        return f.rotation.T @ (_world_path(t)[0] - f.center - f.offset) + f.center

    for t in (0.07, 0.18, 0.31, 0.44):  # inside the quintic pieces, away from the waypoints
        h = 1e-4
        x = mount_position(t)
        v = (mount_position(t + h) - mount_position(t - h)) / (2 * h)
        a = (mount_position(t + h) - 2 * x + mount_position(t - h)) / h**2
        f = mount_frame(slide, SPIN, t)
        predicted = f.gravity(_world_path(t)[2]) + f.fictitious(x[None], v[None])[0]
        np.testing.assert_allclose(a, predicted, rtol=2e-5, atol=2e-4 * np.abs(predicted).max())
        xw, vw = f.to_world(x[None], v[None])  # and back to the world
        np.testing.assert_allclose(xw[0], _world_path(t)[0], atol=1e-12)
        np.testing.assert_allclose(vw[0], _world_path(t)[1], rtol=1e-5)  # v is a central difference: O(h^2)


def test_spin_validation():
    with pytest.raises(ValueError, match="unit vector"):
        MountSpin((0, 0, 2), (0, 0, 0), (0.0, 1.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="rest position"):
        MountSpin((0, 0, 1), (0, 0, 0), (0.0, 1.0), (0.5, 1.0))
    np.testing.assert_allclose(SPIN.rotation(0.0), np.eye(3), atol=1e-15)
    assert SPIN.angle(0.2) == pytest.approx(1.2) and SPIN.rate(0.2) == pytest.approx(0.0, abs=1e-12)
    r = SPIN.rotation(0.1)  # a proper rotation about the axis
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(r @ np.asarray(SPIN.axis), SPIN.axis, atol=1e-14)


SPIN_SCENARIO = Path(__file__).parents[1] / "scenarios" / "actuation" / "spinning_cantilever.yaml"
STEPS = {"mujoco": [], "sofa": ["simulation.timestep=0.5ms"], "fem": ["simulation.timestep=5ms"]}


def _spin_scenario(overrides=()):
    raw = apply_overrides(load_yaml(SPIN_SCENARIO), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), SPIN_SCENARIO.parent)


def _backend(name, scene):
    from softsim.backends import get_backend
    from softsim.errors import SoftSimError

    if name == "fem":
        pytest.importorskip("jax")
    try:
        b = get_backend(name)()
        b.build(scene)
    except (ImportError, SoftSimError) as exc:
        pytest.skip(str(exc))
    return b


def _mount_frame_state(b, state):
    """Mount-frame positions and velocities derived from a *reported* (world) state: the inverse of the
    frame, independent of how a backend finds them internally."""
    f = b.scene.frame(state.time)
    s = state.soft["beam"]
    r = (s.vertices - f.offset - f.center) @ f.rotation          # R^T (x - c - p)
    v = (s.velocities - f.velocity) @ f.rotation - np.cross(f.omega, r)
    return r + f.center, v


@pytest.mark.parametrize("name", ["mujoco", "sofa"])
def test_applied_fictitious_forces_are_mass_times_the_frames_acceleration(name):
    """The mount-frame backends add ``m * MountFrame.fictitious(x, v)`` at every vertex: the force each
    applied during a step equals it, computed from the state reported before the step, for a tilted
    axis through an off-centre point (so a wrong centre, offset, lift or velocity frame would show). The
    reference has no such force (it runs in the world)."""
    spin = "body.mount_rotation={axis: [1, 1, 0], center: [3 mm, -2 mm, 10 mm], path: [[0 ms, 0 rad], [500 ms, 6 rad]]}"
    scene = _spin_scenario([spin, "body.mesh={size: 5 mm, method: gmsh}", *STEPS[name]]).build_scene()
    b = _backend(name, scene)
    dt = scene.simulation.timestep
    b.step(int(round(0.15 / dt)))   # t = 150 ms: spinning at 14 rad/s, still speeding up, the beam is swinging
    before = b.state()
    x, v = _mount_frame_state(b, before)
    mass = scene.soft_bodies[0].vertex_masses()
    b.step(1)
    if name == "mujoco":
        t = before.time + 0.5 * dt  # the step's middle, like gravity
        free = np.array([i for i in range(len(mass)) if b._flex["beam"][3][i, 0] >= 0])
        applied = b.data.xfrc_applied[b._flex["beam"][2][free], :3]
    else:
        t = before.time + dt        # the step's end, like gravity (implicit Euler)
        free = np.arange(len(mass))
        applied = np.asarray(b._soft["beam"]["fictitious"].forces.value)[free]
    frame = scene.frame(t)
    expected = (mass[:, None] * frame.fictitious(x, v))[free]
    r = x - frame.center  # all three terms are in play, so a missing or wrong one is not hidden in the tolerance
    for term in (np.cross(frame.alpha, r), np.cross(frame.omega, np.cross(frame.omega, r)), 2 * np.cross(frame.omega, v)):
        assert np.abs(term[free]).max() > 0.05 * np.abs(frame.fictitious(x, v)[free]).max()
    np.testing.assert_allclose(applied, expected, rtol=2e-4, atol=1e-6 * np.abs(expected).max())


def _early_tip_lag(name, euler=True):
    """Spin-up about an axis across the beam from rest, no gravity: the fraction of how far a rigidly
    turning tip would have moved (``L theta``, at the first sample where that is 1 mm) that the tip has moved
    in the world. The elastic signal from the clamp has not reached the tip yet (shear wave 12 m/s, 60 mm:
    5 ms; the sample is at 3.3 ms), so only the beam's inertia holds the tip back: the reference shows 0.0% and
    in the mount's frame it is the Euler force ``-alpha x r`` alone."""
    from softsim.runner import run_config
    from softsim.scene import MountFrame

    dt = {"mujoco": "0.1ms", "sofa": "0.25ms", "fem": "0.25ms"}[name]
    spin = MountSpin((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.02), (0.0, 0.5))
    overrides = [f"backend={name}", f"simulation.timestep={dt}", "output.save_model=false", "simulation.duration=0.006",
                 "simulation.output_rate=10 kHz", "simulation.gravity=[0, 0, 0]",
                 "body.mount_rotation={axis: [0, 1, 0], center: [0, 0, 0], path: [[0 ms, 0 rad], [20 ms, 0.5 rad]]}"]
    original = MountFrame.fictitious
    if not euler:
        MountFrame.fictitious = lambda self, x, v: original(self, x, v) + np.cross(self.alpha, x - self.center)
    try:
        ts = run_config(SPIN_SCENARIO, overrides).timeseries
    except Exception as exc:  # noqa: BLE001 - a backend that is not installed
        if name != "fem" and ("SOFA" in str(exc) or "Sofa" in str(exc)):
            pytest.skip(str(exc))
        raise
    finally:
        MountFrame.fictitious = original
    lever = 0.06 * np.array([spin.angle(t) for t in ts.t])
    k = int(np.argmin(np.abs(lever - 1e-3)))
    assert ts.t[k] < 4e-3  # before the elastic signal reaches the tip
    return abs(float(ts.tip_x[k])) / lever[k]


@pytest.mark.parametrize("name, limit", [("mujoco", 0.10), ("sofa", 0.35), ("fem", 0.02)])
def test_spin_up_leaves_the_tip_behind(name, limit):
    """The Euler force: measured 4.2% (MuJoCo), 21.7% (SOFA, cause not investigated), 0.0% (the reference,
    in the world). Without it the tip follows the clamp rigidly: 100%."""
    if name == "fem":
        pytest.importorskip("jax")
    lag = _early_tip_lag(name)
    assert lag < limit, lag
    if name != "fem":  # and the check has teeth: with the force dropped the tip is not left behind
        assert _early_tip_lag(name, euler=False) > 0.9


def _spin_vs_plain(name, broken=None):
    """The round beam of ``spinning_cantilever.yaml`` under horizontal gravity, with and without its
    mount spinning (two turns in 0.5 s, peak 45 rad/s): the largest distance between the two tip
    trajectories in the world, as a fraction of the sag. ``broken`` replaces ``MountFrame.fictitious``."""
    from softsim.backends.fem import FemBackend
    from softsim.runner import run_config
    from softsim.scene import MountFrame

    dt = {"mujoco": "0.1ms", "sofa": "1ms", "fem": "2.5ms"}[name]
    base = [f"backend={name}", f"simulation.timestep={dt}", "output.save_model=false"]
    original, new_solver = MountFrame.fictitious, FemBackend._new_solver
    # both runs integrate alike: the reference's spinning runs use the trapezoidal rule (undamped), and the
    # default Newmark of the still run would decay its oscillation (150% apart)
    FemBackend._new_solver = lambda self: self._solver_cls(self._problem, numerical_damping=0.0)
    try:
        plain = run_config(SPIN_SCENARIO, [*base, "body.mount_rotation=null"]).timeseries
        if broken:
            MountFrame.fictitious = broken
        spin = run_config(SPIN_SCENARIO, base).timeseries
    except Exception as exc:  # noqa: BLE001 - a backend that is not installed
        if name != "fem" and ("SOFA" in str(exc) or "Sofa" in str(exc)):
            pytest.skip(str(exc))
        raise
    finally:
        MountFrame.fictitious, FemBackend._new_solver = original, new_solver
    n = min(len(plain), len(spin))
    d = np.hypot(spin.tip_x.to_numpy()[:n] - plain.tip_x.to_numpy()[:n], spin.tip_y.to_numpy()[:n] - plain.tip_y.to_numpy()[:n])
    return d.max() / plain.tip_x.max()


@pytest.mark.slow
@pytest.mark.parametrize("name, limit", [("fem", 0.12), ("sofa", 0.30), ("mujoco", 0.30)])
def test_a_spinning_round_beam_moves_as_a_still_one(name, limit):
    """A round beam bends the same in every direction, so in the world its tip moves exactly as without the
    spinning clamp: in the mount's frame that needs the rotated gravity, the centrifugal force and the
    Coriolis force to be right whatever the beam's stiffness. The reference (the world frame, no fictitious
    forces, trapezoidal rule at 2.5 ms: 0.11 rad per step) differs by 6.4% of the sag, its bending frequency
    identical to 4 digits; at 5 ms (0.23 rad per step) its undamped whirl grows (118%, 6 mm lateral), see
    the ``spin-step`` check. MuJoCo and SOFA differ by 17-21% at 3 mm elements, and the forces are not the
    cause (they are checked exactly in ``test_applied_fictitious_forces...``): (1) the 3 mm mesh is
    anisotropic, its sag differing 4.2% between gravity directions (0.3% at 1.5 mm), which a spinning beam
    near 0.7 of its bending frequency amplifies; (2) implicit Euler damps a mode by an amount that depends on
    its frequency in the integrated frame, and the whirl components have other frequencies in the mount's
    frame than the still beam's: SOFA at 2 mm gives 9.6, 7.3 and 6.3% for 1, 0.5 and 0.25 ms. Here each term is
    shown to matter: without the Coriolis force the difference is 131% (SOFA), with the wrong sign 356%,
    without the centrifugal force 48%, with it reversed 70%."""
    if name == "fem":
        pytest.importorskip("jax")
    assert _spin_vs_plain(name) < limit
    if name == "sofa":
        from softsim.scene import MountFrame
        original = MountFrame.fictitious
        assert _spin_vs_plain(name, lambda self, x, v: original(self, x, v) + 2.0 * np.cross(self.omega, v)) > 0.9   # no Coriolis
        assert _spin_vs_plain(name, lambda self, x, v: original(self, x, v) - 2.0 * np.cross(self.omega, np.cross(self.omega, x - self.center))) > 0.4  # centrifugal reversed


def _centrifugal_stretch(name, damped_newmark=False, broken=None, dt=None):
    """Mid-way through a slow spin-up about an axis across the round beam (60 rad/s, no angular acceleration,
    no gravity): the tip's stretch along the beam against the rotating rod's ``rho omega^2 L^3 / (3 E)``
    (relative difference), and its lateral deflection [m], both in the mount's frame."""
    from softsim.backends.fem import FemBackend
    from softsim.runner import run_config
    from softsim.scene import MountFrame

    dt = dt or {"mujoco": "0.1ms", "sofa": "1ms", "fem": "2.5ms"}[name]
    spin = MountSpin((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0), (0.0, 32.0))
    overrides = [f"backend={name}", f"simulation.timestep={dt}", "output.save_model=false", "simulation.gravity=[0, 0, 0]",
                 "simulation.duration=0.5", "body.mount_rotation={axis: [0, 1, 0], center: [0, 0, 0], path: [[0 ms, 0 rad], [1000 ms, 32 rad]]}"]
    originals = MountFrame.fictitious, FemBackend._new_solver
    if broken:
        MountFrame.fictitious = broken
    if damped_newmark:
        FemBackend._new_solver = lambda self: self._solver_cls(self._problem, numerical_damping=0.25)
    try:
        ts = run_config(SPIN_SCENARIO, overrides).timeseries
    except Exception as exc:  # noqa: BLE001 - a backend that is not installed
        if name != "fem" and ("SOFA" in str(exc) or "Sofa" in str(exc)):
            pytest.skip(str(exc))
        raise
    finally:
        MountFrame.fictitious, FemBackend._new_solver = originals
    last = ts.iloc[-1]
    assert last.t == pytest.approx(0.5) and spin.acceleration(0.5) == pytest.approx(0.0, abs=1e-9)
    x = spin.rotation(0.5).T @ np.array([last.tip_x, last.tip_y, last.tip_z])
    expected = 1070.0 * spin.rate(0.5) ** 2 * 0.06**3 / (3 * 500e3)
    return (x[2] - 0.06) / expected - 1.0, float(np.hypot(x[0], x[1]))


@pytest.mark.parametrize("name, lateral", [pytest.param("fem", 3e-4, marks=pytest.mark.slow), ("sofa", 1e-4), ("mujoco", 1e-4)])
def test_centrifugal_stretch_of_a_spinning_beam(name, lateral):
    """The rotating rod's stretch rho omega^2 L^3 / (3 E) = 0.555 mm at 60 rad/s. Measured: the reference +3.35%
    at 5 ms, +2.83% at 2.5 ms (second order; +2.7% extrapolated: the 3D rod against the 1D formula), MuJoCo -1.6%
    and SOFA -2.3% (linear tetrahedra). The beam must also stay straight (no angular acceleration, so no
    lateral load): 0.12 mm on the reference at 2.5 ms (0.39 at 5), 0.01 mm on MuJoCo and SOFA. The reference's
    default Newmark (numerical damping 0.25) dissipates the rotation: -32% and 4.6 mm lateral at 5 ms."""
    if name == "fem":
        pytest.importorskip("jax")
    error, bend = _centrifugal_stretch(name)
    assert abs(error) < 0.045, error
    assert bend < lateral, bend
    if name == "fem":  # why the trapezoidal rule: the default scheme is far off
        damped, _ = _centrifugal_stretch(name, damped_newmark=True, dt="5ms")
        assert abs(damped) > 0.2, damped
    if name == "sofa":  # and without the centrifugal force there is no stretch at all
        from softsim.scene import MountFrame
        original = MountFrame.fictitious
        none, _ = _centrifugal_stretch(name, broken=lambda self, x, v: original(self, x, v) + np.cross(self.omega, np.cross(self.omega, x - self.center)))
        assert none < -0.9, none


def test_rotating_mount_config_and_refusals():
    """The config's checks, and what a rotating mount refuses on every backend: a region force (its direction
    is a world direction and would have to turn with the mount's frame) and a prescribed region."""
    from softsim.backends import get_backend
    from softsim.errors import ConfigError

    good = "body.mount_rotation={axis: [0, 0, 5], center: [0, 0, 0], path: [[0 ms, 0 rad], [100 ms, 1 rad]]}"
    scene = _spin_scenario([good]).build_scene()
    assert scene.spin.axis == (0.0, 0.0, 1.0)  # normalised
    for overrides, message in [
            (["body.mount_rotation={axis: [0, 0, 0], path: [[0 ms, 0 rad], [100 ms, 1 rad]]}"], "axis must be non-zero"),
            (["body.mount_rotation={axis: [0, 0, 1], path: [[0 ms, 0.3 rad], [100 ms, 1 rad]]}"], "rest position"),
            (["body.mount_rotation={axis: [0, 0, 1], path: [[0 ms, 0 rad]]}"], "two or more waypoints"),
            ([good, "body.fixed=[]"], "fix a region")]:
        with pytest.raises(ConfigError, match=message):
            _spin_scenario(overrides).build_scene()
    forced = _spin_scenario([good, "actuators=[{name: push, type: force, region: top, direction: [1, 0, 0], force: 1 mN}]"]).build_scene()
    held = _spin_scenario([good, "body.prescribed=[{region: top, path: [[0 ms, [null, null, 0]], [100 ms, [null, null, 1 mm]]]}]"]).build_scene()
    for name in ("mujoco", "sofa", "fem"):
        try:
            backend = get_backend(name)
        except Exception:  # noqa: BLE001 - not installed
            continue
        assert any("region force with a rotating mount" in m for m in backend.unsupported(forced)), name
        assert any("prescribed region 'top' with a rotating mount" in m for m in backend.unsupported(held)), name


def test_spin_checks_warn_about_step_and_centrifugal_strain():
    """The checks warn where measurement says to: 0.23 rad per step (the reference's whirl grows, 6 mm lateral
    on the 60 mm beam) but not 0.11 (0.5 mm), and a centrifugal strain of 51% (450 rad/s about an axis across
    this beam) but not the 0.9% of the 60 rad/s test, nor a spin about the beam's own axis."""
    from softsim.checks.stability import check_scene

    def codes(overrides):
        return {f.code for f in check_scene(_spin_scenario(overrides).build_scene(), locking_free=True)}

    assert "spin-step" in codes(["simulation.timestep=5ms"])      # 45 rad/s x 5 ms = 0.23
    assert "spin-step" not in codes(["simulation.timestep=2.5ms"])  # 0.11
    across = "body.mount_rotation={axis: [0, 1, 0], path: [[0 ms, 0 rad], [%s, %s rad]]}"
    assert "spin-centrifugal" in codes([across % ("50 ms", 12)])       # 450 rad/s: 51%
    assert "spin-centrifugal" not in codes([across % ("1000 ms", 32)])  # 60 rad/s: 0.9%
    assert "spin-centrifugal" not in codes([])                          # about its own axis, 45 rad/s

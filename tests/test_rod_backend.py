"""Rod backend (BeamAdapter) on the actuation scenarios, against the analytic beam solutions; needs SOFA.

The rod is geometrically nonlinear and materially linear, so it is compared with the *linear-material*
Euler-Bernoulli solutions. The 3D reference differs from those by the compression stiffening
E_t = E (1 + 2N/EA) (tests/test_cable_arm.py); that factor is checked here too."""

from pathlib import Path

import numpy as np
import pytest

from softsim.backends.rod import RodBackend
from softsim.backends.rod.reduce import reduce_body
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import SoftSimError, UnsupportedFeatureError
from softsim.scenarios.inflation import InflationConfig, InflationScenario

DIR = Path(__file__).parents[1] / "scenarios" / "actuation"
E, W = 500e3, 0.01  # the 10 x 10 mm cantilevers


def _scene(name, overrides=()):
    path = DIR / f"{name}.yaml"
    return InflationScenario(validate(InflationConfig, apply_overrides(load_yaml(path), list(overrides))), DIR).build_scene()


def _run(name, overrides=(), until=None):
    """Backend after running scenario ``name`` to ``until`` [s] (default: its duration)."""
    scene = _scene(name, ["simulation.timestep=2ms", *overrides])
    backend = RodBackend()
    try:
        backend.build(scene)
    except SoftSimError as exc:
        if "SofaPython3" in str(exc):
            pytest.skip(str(exc))
        raise
    backend.step(int(round((until or scene.simulation.duration) / scene.simulation.timestep)))
    return backend


def _tilt(backend, z):
    """Rotation [rad] of the cross-section at height ``z`` (fit of x against z over its vertices)."""
    x = backend.state().soft[backend.scene.soft_bodies[0].name].vertices
    sel = np.abs(backend.model.rest_points[:, 2] - z) < 1e-9
    return float(np.arctan(-np.polyfit(x[sel, 0], x[sel, 2], 1)[0]))


def _tip_x(backend):
    x = backend.state().soft[backend.scene.soft_bodies[0].name].vertices
    return float(x[np.abs(backend.model.rest_points[:, 2] - 0.06) < 1e-9, 0].mean())


def test_tip_load_matches_euler_bernoulli():
    b = _run("tip_load_cantilever")
    assert _tip_x(b) == pytest.approx(5e-3 * 0.06**3 / (3 * E * W**4 / 12), rel=0.005)  # 0.864 mm


def test_cable_cantilever_curls_to_the_circular_arc_at_large_rotation():
    """A pure moment T d bends the beam to theta = T d L / (E I) at any rotation (0.28 rad here, 16 deg): a rod
    linear in the rotation would also pass at this angle, the tip position below does not (it needs the arc)."""
    b = _run("cable_cantilever")
    theta = 0.7 * 0.003 * 0.055 / (E * W**4 / 12)
    assert _tilt(b, 0.06) == pytest.approx(theta, rel=0.01)
    # the tip sits on the arc: x = L (1 - cos(theta)) / theta below the anchor, then straight up to 60 mm
    arc = 0.055 * (1 - np.cos(theta)) / theta + 0.005 * np.sin(theta)
    assert _tip_x(b) == pytest.approx(arc, rel=0.02)


def test_large_rotation_is_geometrically_nonlinear():
    """Four times the tension: the tilt reaches T d L / EI = 1.11 rad (64 deg) and the tip follows the arc, which
    pulls it 16 mm down from 60 mm (a beam linear in the rotation would keep it there). The cable's 2.8 N
    also compresses the beam, so the arc is that of the length L / (1 + N / EA) (EA = 50 N)."""
    b = _run("cable_cantilever", ["actuators[0].schedule=[[0, 0], [0.4, 2.8], [0.5, 2.8]]"])
    theta = 2.8 * 0.003 * 0.055 / (E * W**4 / 12)
    tilt = _tilt(b, 0.06)
    assert tilt == pytest.approx(theta, rel=0.02)
    length = 0.055 / (1 + 2.8 / (E * W * W))
    x = b.state().soft["beam"].vertices
    top = np.abs(b.model.rest_points[:, 2] - 0.06) < 1e-9
    assert x[top, 0].mean() == pytest.approx(length * (1 - np.cos(tilt)) / tilt + 0.005 * np.sin(tilt), rel=0.01)
    assert x[top, 2].mean() == pytest.approx(length * np.sin(tilt) / tilt + 0.005 * np.cos(tilt), rel=0.01)
    assert x[top, 2].mean() < 0.05


def test_layered_section_uses_the_composite_stiffness_and_moment_arm():
    b = _run("cable_cantilever_layered")
    rod = reduce_body(b.scene.soft_bodies[0])
    theta = 0.7 * (0.003 - rod.centre[0, 0]) * 0.055 / rod.ei_x[0]
    assert _tilt(b, 0.06) == pytest.approx(theta, rel=0.01)
    assert rod.centre[0, 0] < -2e-3 and rod.ei_x[0] > 2 * E * W**4 / 12  # neutral axis moved, EI more than doubled


def test_stroke_controlled_cable_balances_its_tendon():
    b = _run("cable_cantilever_stroke")
    state = b.state()
    tension = state.controls  # the stroke is the control; the tendon tension follows from the length
    assert tension["cable"] > 0
    # the tilt is that of the moment the tendon's tension makes: T d L / EI (the stiff cap adds 5 mm of rigid beam)
    assert _tilt(b, 0.06) > 0.15


def test_two_segment_arm_follows_piecewise_linear_euler_bernoulli():
    """The S of the cable arm: 0.2315 rad after the first segment, then back to 0.1219 rad (linear E); the
    3D reference is lower by the compression stiffening 1 + 2N/EA (0.2119 / 0.1074 rad)."""
    D = "{model: rayleigh, beta: 0.3}"
    b = _run("cable_arm", ["simulation.timestep=1ms", f"body.material.damping={D}", f"body.zones[0].material.damping={D}",
                           f"body.zones[1].material.damping={D}", "simulation.duration=2"], until=1.5)
    e, bw = 114e3, 0.012
    inertia, area = bw**4 / 12, bw * bw
    th1 = 0.3 * 0.004 * 0.038 / (e * inertia)
    th2 = th1 - 0.15 * 0.004 * 0.036 / (e * inertia)
    assert _tilt(b, 0.038) == pytest.approx(th1, rel=0.015) and _tilt(b, 0.042) == pytest.approx(th1, rel=0.015)
    assert _tilt(b, 0.078) == pytest.approx(th2, rel=0.015) and _tilt(b, 0.12) == pytest.approx(th2, rel=0.015)
    # the reference's compression stiffening explains the difference to it (its measured tilts: 0.2119, 0.1074)
    stiffen = lambda n: 1 + 2 * n / (e * area)  # noqa: E731
    assert th1 / stiffen(0.6) == pytest.approx(0.2119, rel=0.02)


def test_mass_damping_and_contact_are_refused():
    with pytest.raises(UnsupportedFeatureError, match="mass-proportional"):
        RodBackend().build(_scene("cable_cantilever", ["body.material.damping={model: rayleigh, alpha: 5}"]))
    with pytest.raises(UnsupportedFeatureError, match="ground plane"):
        RodBackend.check(_scene("cable_cantilever", ["ground.enabled=true"]))
    with pytest.raises(UnsupportedFeatureError, match="Prony|viscoelastic"):
        RodBackend.check(_scene("cable_cantilever", ["body.material.damping={model: prony, g: [0.5], tau: [0.01]}"]))


def test_state_velocities_follow_the_frames():
    b = _run("tip_load_cantilever", until=0.2)
    s = b.state().soft["beam"]
    assert s.velocities.shape == s.vertices.shape and np.isfinite(s.velocities).all()
    assert s.velocities[:, 0].max() > 0  # accelerating along the force
    base = b.model.rest_points[:, 2] < 1e-9
    assert np.abs(s.velocities[base]).max() < 1e-12  # the clamped end does not move

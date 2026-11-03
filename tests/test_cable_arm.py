"""Two-segment cable arm: stiff disks, two cables ending in different disks, against piecewise Euler-Bernoulli."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.scenarios.inflation import InflationConfig, InflationScenario

ARM = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_arm.yaml"
E, B = 114e3, 0.012  # Ecoflex 00-30 card, section width
TA, TB, ARM_X = 0.45, 0.15, 0.004


def _arm(overrides=()):
    return InflationScenario(validate(InflationConfig, apply_overrides(load_yaml(ARM), list(overrides))), ARM.parent)


def _tilts(overrides=(), levels=(0.038, 0.042, 0.078, 0.082, 0.12)):
    """Quasi-static reference at (TA, TB): rotation [rad] of the cross-sections at ``levels`` and the lateral
    displacement of the tip."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.fem.solid3d import Solid3DSolver

    backend = FemBackend()
    backend.build(_arm(["body.mesh.size=2mm", *overrides]).build_scene())
    s = Solid3DSolver(backend._problem, dynamic=False)
    s.tensions[:] = [TA, TB]
    assert s.advance(1.0)
    x = s.p.mesh.nodes + s.y[:3 * s.n].reshape(-1, 3)
    out = []
    for z in levels:
        sel = np.abs(s.p.mesh.nodes[:, 2] - z) < 1e-9
        out.append(np.arctan(-np.polyfit(x[sel, 0], x[sel, 2], 1)[0]))
    tip = np.abs(s.p.mesh.nodes[:, 2] - 0.12) < 1e-9
    return np.array(out), float(x[tip, 0].mean() - s.p.mesh.nodes[tip, 0].mean())


def test_arm_config():
    scene = _arm(["body.mesh.size=2mm"]).build_scene()
    body = scene.soft_bodies[0]
    assert [z.name for z in body.zones] == ["disk1", "disk2"] and [a.name for a in scene.actuators] == ["cable_a", "cable_b"]
    za = [v[2] for v in scene.actuators[0].via]
    zb = [v[2] for v in scene.actuators[1].via]
    assert max(za) == pytest.approx(0.04) and max(zb) == pytest.approx(0.08)
    assert {v[0] for v in scene.actuators[0].via} == {0.004} and {v[0] for v in scene.actuators[1].via} == {-0.004}


def test_reference_matches_piecewise_euler_bernoulli():
    """Lower soft segment (0-38 mm) carries both cables' moments, (T_A - T_B) d, with the axial force T_A + T_B;
    the upper (42-78 mm) only B's, -T_B d, with T_B; disks rigid, the tail above the second disk unloaded.
    theta = sum M L / (E_t I), E_t = E (1 + 2 N / (E A)). Measured: -1.8% after the first segment, -0.6% at the tip.
    The S: the arm turns back above the first disk (the tilt falls from 0.212 to 0.107 rad)."""
    tilts, tip_x = _tilts()
    inertia, area = B**4 / 12, B * B
    et = lambda n: E * (1 + 2 * n / (E * area))  # noqa: E731
    th1 = (TA - TB) * ARM_X * 0.038 / (et(TA + TB) * inertia)
    th2 = th1 - TB * ARM_X * 0.036 / (et(TB) * inertia)
    assert tilts[0] == pytest.approx(th1, rel=0.03) and tilts[1] == pytest.approx(th1, rel=0.03)  # disk 1 turns rigidly
    assert tilts[2] == pytest.approx(th2, rel=0.03) and tilts[3] == pytest.approx(th2, rel=0.03)
    assert tilts[4] == pytest.approx(tilts[3], abs=2e-3)  # the unloaded tail stays straight
    assert th1 > 1.5 * th2 > 0 and tip_x > 0.01  # S-shape, tip displaced towards cable A's side


def test_second_cable_in_the_first_disk_removes_the_return():
    """Control for the S: cable B ending in the same disk as A leaves nothing to bend the upper segment, so
    everything above the disk is straight and the tip tilt equals the disk's (no return)."""
    tilts, _ = _tilts(["actuators[1].path=[[-4 mm, 0 mm, 0 mm], [-4 mm, 0 mm, 40 mm]]"])
    assert tilts[4] == pytest.approx(tilts[1], abs=2e-3) and tilts[1] > 0.2

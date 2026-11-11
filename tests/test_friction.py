"""Friction benchmark: a soft block pushed across the ground against Coulomb's law (scenarios/actuation/
sliding_block.yaml). The exact travel against closed forms; MuJoCo slides as Coulomb says and, below the limit,
sticks up to its soft-constraint creep."""

from pathlib import Path

import numpy as np
import pytest

from softsim.analytic import sliding_block_travel

BLOCK = Path(__file__).parents[1] / "scenarios" / "actuation" / "sliding_block.yaml"
MASS, MU, G = 1070 * 0.04 * 0.04 * 0.01, 0.5, 9.81


def test_exact_travel_against_closed_forms():
    # a constant force over the limit from t = 0: x = a t^2 / 2
    force = 0.12
    a = (force - MU * MASS * G) / MASS
    assert sliding_block_travel([0, 1], [force, force], MASS, MU, 0.3) == pytest.approx(0.5 * a * 0.09, rel=1e-6)
    # under the limit it never moves
    assert sliding_block_travel([0, 1], [0.05, 0.05], MASS, MU, 0.3) == 0.0
    # a linear ramp F = k t from zero: moves from t0 = mu m g / k, x = k (t - t0)^3 / (6 m)
    k = 2.0
    t0 = MU * MASS * G / k
    assert sliding_block_travel([0, 1], [0, k], MASS, MU, 0.3) == pytest.approx(k * (0.3 - t0) ** 3 / (6 * MASS),
                                                                                  rel=1e-6)
    with pytest.raises(ValueError, match="must not decrease"):
        sliding_block_travel([0, 1], [1, 0], MASS, MU, 0.3)


def _travel(overrides=()):
    from softsim.runner import Simulation

    sim = Simulation(BLOCK, ["output.save_model=false", *overrides], backend="mujoco")
    try:
        return sim.run().timeseries
    finally:
        sim.close()


def test_mujoco_slides_as_coulomb_says():
    pytest.importorskip("mujoco")
    ts = _travel()
    exact = [sliding_block_travel([0, 0.05, 0.3], [0, 0.12, 0.12], MASS, MU, t) for t in (0.1, 0.2, 0.3)]
    sim = np.interp([0.1, 0.2, 0.3], ts.t, ts.push_displacement)
    np.testing.assert_allclose(sim, exact, rtol=0.05)
    assert sim[-1] == pytest.approx(exact[-1], rel=0.02)


def test_mujoco_sticks_below_the_limit_up_to_its_creep():
    """0.71 of the limit: Coulomb says it sticks. MuJoCo's soft friction creeps at a steady rate (about 0.13 mm/s here),
    far below the 6 mm a frictionless block would travel by 300 ms."""
    pytest.importorskip("mujoco")
    ts = _travel(["actuators[0].schedule=[[0 ms, 0 N], [50 ms, 0.06 N], [300 ms, 0.06 N]]"])
    x = ts.push_displacement.to_numpy()
    assert x[-1] < 5e-5
    late = ts[ts.t > 0.1]
    rate = np.polyfit(late.t, late.push_displacement, 1)[0]
    assert 0 < rate < 3e-4  # creeps, slowly, and steadily: no acceleration
    frictionless = sliding_block_travel([0, 0.05, 0.3], [0, 0.06, 0.06], MASS, 0.0, 0.3)
    assert x[-1] < 0.01 * frictionless


def _fem_creep(timestep: str, slip_speed: float | None) -> float:
    """Creep rate [m/s] of the reference's stuck block at 0.71 of the friction limit (coarse, fast)."""
    from softsim.runner import Simulation

    overrides = ["output.save_model=false", "body.mesh.size=10mm", f"simulation.timestep={timestep}",
                 "simulation.duration=200ms",
                 "actuators[0].schedule=[[0 ms, 0 N], [50 ms, 0.06 N], [200 ms, 0.06 N]]"]
    if slip_speed is not None:
        overrides.append(f"contact.slip_speed={slip_speed}")
    sim = Simulation(BLOCK, overrides, backend="fem")
    try:
        ts = sim.run().timeseries
    finally:
        sim.close()
    late = ts[ts.t > 0.08]
    return float(np.polyfit(late.t, late.push_displacement, 1)[0])


@pytest.mark.slow
def test_reference_stick_creep_is_set_by_slip_speed_not_by_the_step():
    """docs/decisions.md, entry 6. The reference smooths Coulomb friction with IPC's mollifier: below a slip ``eps`` the
    force is mu N (2 y - y^2), y = slip / eps, so a contact held at a fraction r of its limit slips y = 1 - sqrt(1 - r)
    of eps per step. With contact.slip_speed, eps = slip_speed x dt and the creep is y x slip_speed at any step;
    without it, eps = 1 um and the creep is y x 1 um / dt. (Uniform r here; the push tilts the normal load, so the
    nodes' r and the measured creep spread around it.)"""
    pytest.importorskip("jax")
    r = 0.06 / (MU * MASS * G)  # 0.71 of the limit
    y = 1 - np.sqrt(1 - r)
    fixed = [_fem_creep(dt, 1e-5) for dt in ("2.5ms", "5ms")]
    assert fixed[0] == pytest.approx(fixed[1], rel=0.1)  # independent of the step
    assert fixed[0] == pytest.approx(y * 1e-5, rel=0.3)
    per_step = [_fem_creep(dt, None) for dt in ("2.5ms", "5ms")]
    assert per_step[0] / per_step[1] == pytest.approx(2.0, rel=0.1)  # 1 / dt
    assert per_step[0] == pytest.approx(y * 1e-6 / 2.5e-3, rel=0.3)

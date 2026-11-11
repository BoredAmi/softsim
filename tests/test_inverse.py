"""Quasi-static inverse control (softsim.control.inverse), MuJoCo, coarse meshes."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from softsim.config.loader import apply_overrides, load_yaml  # noqa: E402
from softsim.control.inverse import Equilibrium, run_job, solve_inverse  # noqa: E402
from softsim.errors import ConfigError  # noqa: E402

ROOT = Path(__file__).parents[1]
CANTILEVER = ROOT / "scenarios" / "actuation" / "cable_cantilever.yaml"
ARM = ROOT / "scenarios" / "actuation" / "cable_arm.yaml"
COARSE = ["body.mesh.size=4mm"]
ARM_FAST = ["body.mesh.size=6mm", "simulation.timestep=0.5ms", "simulation.iterations=100"]
ANGLE = "bend_angle"


def _forward(tensions):
    """MuJoCo's own equilibrium bend angle for each tension, on the 4 mm cantilever (one simulation reused)."""
    from softsim.control.inverse import DEFAULT_DAMPING, _with_damping
    from softsim.runner import Simulation

    raw = _with_damping(apply_overrides(load_yaml(CANTILEVER), COARSE), DEFAULT_DAMPING)
    sim = Simulation(raw, base_dir=CANTILEVER.parent)
    try:
        eq = Equilibrium(sim, [ANGLE], {ANGLE: 1e-2})
        return [float(eq({"cable": T})[0]) for T in tensions]
    finally:
        sim.close()


@pytest.mark.slow
def test_round_trip_recovers_the_tension():
    """Simulate at a known tension, take the angle as the target, solve from another start: recovers the tension
    to within what the 0.05 tolerance (5e-4 rad) allows, ~1.5% of the 0.04 rad signal."""
    true = 0.4
    target = _forward([true])[0]
    res = solve_inverse(CANTILEVER, {ANGLE: target}, ["cable"], {"cable": (0.0, 0.7)}, overrides=COARSE,
                        initial={"cable": 0.05})
    assert res.status == "converged" and res.converged
    assert res.values["cable"] == pytest.approx(true, rel=0.03)
    assert abs(res.errors[ANGLE]) <= 0.05 * 1e-2 + 1e-9
    assert res.unsettled == 0 and res.simulations <= 12


@pytest.mark.slow
def test_tip_angle_is_linear_in_tension_and_the_inverse_follows():
    """Euler-Bernoulli: theta = T d L / (E I), linear in T at small angles. MuJoCo's flex locks at 4 mm (it bends far
    less than EB) so the slope is compared with MuJoCo's own forward map, not with EB: theta/T is constant within 4%
    over 0.1-0.4 N, and the inverse of twice the 0.1 N angle is 0.2 N."""
    angles = _forward([0.1, 0.2, 0.4])
    slope = np.array(angles) / np.array([0.1, 0.2, 0.4])
    assert slope.max() / slope.min() < 1.04
    eb = 0.1 * 3e-3 * 55e-3 / (500e3 * 10e-3**4 / 12)  # theta at 0.1 N
    assert 0.1 < angles[0] / eb < 0.7  # locked, as documented (docs/actuation.md)
    res = solve_inverse(CANTILEVER, {ANGLE: 2 * angles[0]}, ["cable"], {"cable": (0.0, 0.7)}, overrides=COARSE,
                        initial={"cable": 0.5})
    assert res.converged and res.values["cable"] == pytest.approx(0.2, rel=0.04)


@pytest.mark.slow
def test_unreachable_target_returns_the_bounded_best():
    """A target far beyond what 0.3 N can bend: no exception, the upper bound, flagged."""
    res = solve_inverse(CANTILEVER, {ANGLE: 0.5}, ["cable"], {"cable": (0.0, 0.3)}, overrides=COARSE)
    assert res.status == "bounded_best" and not res.converged
    assert res.at_bounds == ["cable"]
    assert res.values["cable"] == pytest.approx(0.3, abs=1e-3)
    assert res.errors[ANGLE] < 0 and res.residual > 1


def test_input_errors():
    with pytest.raises(ConfigError, match="unknown output"):
        solve_inverse(CANTILEVER, {"nope": 1.0}, ["cable"], {"cable": (0, 1)}, overrides=COARSE)
    with pytest.raises(ConfigError, match="no actuator"):
        solve_inverse(CANTILEVER, {ANGLE: 0.1}, ["motor"], {"motor": (0, 1)}, overrides=COARSE)
    with pytest.raises(ConfigError, match="not be unique"):
        solve_inverse(ARM, {"tip_x": 0.0}, ["cable_a", "cable_b"], {"cable_a": (0, 1), "cable_b": (0, 1)},
                      overrides=COARSE)


@pytest.mark.slow
def test_two_cables_two_targets_round_trip():
    """The two-segment arm (6 mm mesh, 0.5 ms; ~1 s of simulated settling per equilibrium): known tensions
    (0.45, 0.15) N give a tip x and z; solving for them from the middle of the bounds recovers both tensions within
    0.03 N (of a 0.5 N range). tip_z moves ~1.5 mm over the whole range, so it is judged to 0.2 mm and the problem is
    ill-conditioned: the residual, not the tensions, is what the tolerance bounds."""
    from softsim.control.inverse import _with_damping
    from softsim.runner import Simulation

    scales = {"tip_x": 1e-3, "tip_z": 2e-4}
    raw = _with_damping(apply_overrides(load_yaml(ARM), ARM_FAST), {"model": "rayleigh", "beta": 0.15})
    sim = Simulation(raw, base_dir=ARM.parent)
    try:
        out = Equilibrium(sim, ["tip_x", "tip_z"], scales, settle_tol=0.005, max_settle=3.0)({"cable_a": 0.45, "cable_b": 0.15})
        res = solve_inverse(sim, {"tip_x": out[0], "tip_z": out[1]}, ["cable_a", "cable_b"],
                            {"cable_a": (0.0, 0.5), "cable_b": (0.0, 0.5)}, tolerance=0.1, scales=scales, settle_tol=0.005,
                            max_settle=3.0)
    finally:
        sim.close()
    assert res.converged and res.unsettled == 0
    assert res.values["cable_a"] == pytest.approx(0.45, abs=0.03) and res.values["cable_b"] == pytest.approx(0.15, abs=0.03)


def test_job_file(tmp_path):
    res = run_job(ROOT / "jobs" / "inverse_cable_cantilever.yaml")
    assert res["status"] == "converged"
    assert res["outputs"][ANGLE] == pytest.approx(np.radians(2.0), abs=1e-3)

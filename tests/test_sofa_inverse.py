"""SOFA fast path for inverse control (softsim.control.sofa_inverse): SoftRobots.Inverse's QP on coarse meshes."""

import time
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("yaml")

from softsim.backends.sofa.backend import import_sofa  # noqa: E402
from softsim.errors import SoftSimError  # noqa: E402

try:
    import_sofa()
except (SoftSimError, ImportError):
    pytest.skip("SOFA (SofaPython3) not available", allow_module_level=True)

from softsim.config.loader import apply_overrides, load_yaml  # noqa: E402
from softsim.control.inverse import DEFAULT_DAMPING, Equilibrium, _with_damping, run_job, solve_inverse  # noqa: E402
from softsim.control.sofa_inverse import solve_inverse_sofa  # noqa: E402
from softsim.runner import Simulation  # noqa: E402

ROOT = Path(__file__).parents[1]
CANTILEVER = ROOT / "scenarios" / "actuation" / "cable_cantilever.yaml"
ARM = ROOT / "scenarios" / "actuation" / "cable_arm.yaml"
COARSE = ["body.mesh.size=4mm", "simulation.timestep=2ms"]
ARM_COARSE = ["body.mesh.size=6mm"]
ARM_DAMPING = {"model": "rayleigh", "beta": "0.15 s"}


def _forward(config, overrides, tensions, outputs):
    """The ordinary SOFA backend (explicit cable force) held at ``tensions`` until the ``outputs`` are at rest."""
    raw = _with_damping(apply_overrides(load_yaml(config), overrides), DEFAULT_DAMPING)
    sim = Simulation(raw, backend="sofa", base_dir=config.parent)
    try:
        eq = Equilibrium(sim, outputs, {o: 1e-3 for o in outputs}, settle_window=0.2, max_settle=4.0)
        out = eq(tensions)
        assert eq.unsettled == 0
        return dict(zip(outputs, map(float, out)))
    finally:
        sim.close()


def test_qp_tension_applied_forward_reaches_the_target():
    """The QP's tension, applied as the ordinary backend's explicit cable force, puts the tip at the target
    (2 mm, within 0.05 mm: the explicit cable and the CableActuator are the same load, docs/roadmap.md M6)."""
    res = solve_inverse_sofa(CANTILEVER, {"tip_x": "2 mm"}, ["cable"], {"cable": (0, "0.7 N")}, overrides=COARSE)
    assert res.status == "converged" and res.unsettled == 0
    assert 0.3 < res.values["cable"] < 0.7
    reached = _forward(CANTILEVER, COARSE, res.values, ["tip_x"])["tip_x"]
    assert reached == pytest.approx(2e-3, abs=5e-5)
    assert res.outputs["tip_x"] == pytest.approx(reached, abs=5e-5)


def test_agrees_with_the_repeated_solve_and_is_faster():
    """Same model, same target (a 2 degree tip tilt): the QP tension is the repeated solve's to 2%, which is the
    1e-2 rad * 0.05 tolerance both stop at (0.4% in angle). Measured ~8 times faster; asserted > 2 (a shared CPU)."""
    target, bounds = {"bend_angle": "2 deg"}, {"cable": (0, "0.7 N")}
    qp = solve_inverse_sofa(CANTILEVER, target, ["cable"], bounds, overrides=COARSE)
    t0 = time.perf_counter()
    rs = solve_inverse(CANTILEVER, target, ["cable"], bounds, backend="sofa", overrides=COARSE, max_settle=3.0)
    t_rs = time.perf_counter() - t0
    assert qp.status == rs.status == "converged"
    assert qp.values["cable"] == pytest.approx(rs.values["cable"], rel=0.02)
    assert t_rs / qp.wall_time > 2.0


def test_target_beyond_the_bounds_gives_the_bounded_solution():
    """8 mm needs more than the 0.3 N allowed: the cable sits at its bound, the tip is where 0.3 N puts it (the
    ordinary backend's forward map) and the result is flagged."""
    res = solve_inverse_sofa(CANTILEVER, {"tip_x": "8 mm"}, ["cable"], {"cable": (0, "0.3 N")}, overrides=COARSE)
    assert res.status == "bounded_best" and not res.converged and res.at_bounds == ["cable"]
    assert res.values["cable"] == pytest.approx(0.3, rel=1e-3)
    reached = _forward(CANTILEVER, COARSE, {"cable": 0.3}, ["tip_x"])["tip_x"]
    assert res.outputs["tip_x"] == pytest.approx(reached, abs=5e-5)
    assert res.outputs["tip_x"] < 8e-3


def test_unsupported_targets_are_refused():
    from softsim.errors import ConfigError

    with pytest.raises(ConfigError, match="not effector outputs"):
        solve_inverse_sofa(CANTILEVER, {"cable_tension": 0.1}, ["cable"], {"cable": (0, 0.7)})


def test_job_file_and_method_selection():
    """``method: sofa-qp`` in the job and ``method='sofa-qp'`` on a repeated-solve job give the same QP answer."""
    a = run_job(ROOT / "jobs" / "inverse_cable_cantilever_sofa.yaml")
    b = run_job(ROOT / "jobs" / "inverse_cable_cantilever.yaml", "sofa-qp")
    assert a["status"] == b["status"] == "converged"
    assert a["values"]["cable"] == pytest.approx(b["values"]["cable"], rel=1e-6)


def test_two_cable_arm_recovers_the_tensions():
    """Tip x and z of the arm (jobs/inverse_cable_arm.yaml: the tip MuJoCo reaches at 0.45 / 0.15 N on the same 6 mm
    mesh) are the targets of one QP run, which recovers both tensions to 3% (measured 0.07% and 0.02%). The
    ordinary SOFA backend's explicit cable is not used as the reference here: on this arm it does not settle
    (tip x keeps oscillating by +-0.5 mm at 1-5 ms steps), docs/control.md."""
    bounds = {"cable_a": (0, "0.5 N"), "cable_b": (0, "0.5 N")}
    res = solve_inverse_sofa(ARM, {"tip_x": "3.994 mm", "tip_z": "118.515 mm"}, list(bounds), bounds,
                             overrides=ARM_COARSE, settle_damping=ARM_DAMPING, scales={"tip_z": "0.2 mm"},
                             tolerance=0.1, timestep="5 ms", settle_window="0.2 s", max_settle="4 s")
    assert res.status == "converged" and res.unsettled == 0
    assert res.values["cable_a"] == pytest.approx(0.45, rel=0.03)
    assert res.values["cable_b"] == pytest.approx(0.15, rel=0.03)


def test_arm_tip_settles_on_the_forward_backend():
    """The two-cable arm under constant 0.45 / 0.15 N: the tension-controlled cable's tangent stiffness is in the
    implicit system, so the tip settles (an explicit follower force swung it by +-0.5 mm indefinitely, 5 ms or 1 ms)
    at the QP path's tip (x 3.994 mm, z 118.515 mm on the 6 mm mesh, to 0.02 mm and 0.05 mm)."""
    import numpy as np

    raw = _with_damping(apply_overrides(load_yaml(ARM), [*ARM_COARSE, "simulation.timestep=5ms"]), ARM_DAMPING)
    sim = Simulation(raw, backend="sofa", base_dir=ARM.parent)
    try:
        sim.backend.set_control("cable_a", 0.45)
        sim.backend.set_control("cable_b", 0.15)
        tip, top = [], np.where(sim.backend.state().soft["arm"].vertices[:, 2] > 0.1199)[0]
        for _ in range(500):  # 2.5 s
            sim.backend.step(1)
            tip.append(sim.backend.state().soft["arm"].vertices[top].mean(axis=0))
        tip = np.array(tip)
        assert np.ptp(tip[-100:, 0]) < 2e-6 and np.ptp(tip[-100:, 2]) < 2e-6  # the last 0.5 s
        assert tip[-1, 0] == pytest.approx(3.994e-3, abs=2e-5)
        assert tip[-1, 2] == pytest.approx(118.515e-3, abs=5e-5)
    finally:
        sim.close()


BENDING = ROOT / "scenarios" / "inflation" / "bending.yaml"


@pytest.mark.slow
def test_pressure_round_trip_on_the_bending_actuator():
    """Pressure actuator: the tip the ordinary SOFA backend (explicit follower pressure) reaches at a held 4 kPa
    is the QP's target, and the QP gives 4 kPa back (measured 3999.0 Pa, 0.03%; asserted 1%). The SurfacePressureActuator
    bounds go in as ``pressure * dt`` and the pressure is ``lambda / dt``, as for the cables."""
    outs = ["tip_x"]
    reached = _forward(BENDING, ["simulation.timestep=2ms"], {"cavity": 4000.0}, outs)["tip_x"]
    assert reached < -2e-4                       # curls towards -x
    res = solve_inverse_sofa(BENDING, {"tip_x": reached}, ["cavity"], {"cavity": (0, "10 kPa")}, timestep="2 ms",
                             scales={"tip_x": "0.01 mm"})
    assert res.status == "converged" and res.unsettled == 0
    assert res.values["cavity"] == pytest.approx(4000.0, rel=0.01)
    assert res.outputs["tip_x"] == pytest.approx(reached, abs=2e-6)


def test_pressure_target_beyond_the_bounds_is_the_bounded_solution():
    """2 kPa cannot reach 1 mm: the cavity sits at its bound and the tip is where the explicit backend puts it at
    2 kPa (-0.123066 mm, measured with 0.1 ms steps; the QP's clamped run gives -0.123066 mm)."""
    res = solve_inverse_sofa(BENDING, {"tip_x": "-1 mm"}, ["cavity"], {"cavity": (0, "2 kPa")}, timestep="5 ms",
                             scales={"tip_x": "0.01 mm"}, max_corrections=2)
    assert res.status == "bounded_best" and res.at_bounds == ["cavity"]
    assert res.values["cavity"] == pytest.approx(2000.0, rel=1e-3)
    assert res.outputs["tip_x"] == pytest.approx(-1.23066e-4, abs=3e-6)


def test_pressure_angle_target_is_refused():
    from softsim.errors import ConfigError

    with pytest.raises(ConfigError, match="angle targets are not supported with pressure"):
        solve_inverse_sofa(BENDING, {"bend_angle": "1 deg"}, ["cavity"], {"cavity": (0, "10 kPa")})


@pytest.mark.slow
def test_pressure_job_file():
    """jobs/inverse_bending_sofa.yaml: the target is the tip the explicit backend reaches at 4 kPa."""
    out = run_job(ROOT / "jobs" / "inverse_bending_sofa.yaml")
    assert out["status"] == "converged" and out["values"]["cavity"] == pytest.approx(4000.0, rel=0.01)

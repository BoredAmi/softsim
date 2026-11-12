"""Closed-loop controller (softsim.control.controller): on an analytic plant with a mismatched model feedback removes the
steady-state error feedforward leaves, anti-windup at the bounds, MPC on a linear model, bad inputs raise ConfigError;
one slow MuJoCo closed-loop test."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

import softsim.control.surrogate as sg  # noqa: E402
from softsim.control.controller import Controller  # noqa: E402
from softsim.control.surrogate import Surrogate, design_points, fit_dmd, fit_model  # noqa: E402
from softsim.errors import ConfigError  # noqa: E402

ROOT = Path(__file__).parents[1]
DT = 0.02


def _map(x, gain=1.0):
    """Two inputs (N) -> tip_x (m), tip_z (m); mildly nonlinear and well conditioned."""
    a, b = x[..., 0], x[..., 1]
    return gain * np.stack([2e-2 * a - 1e-2 * b + 4e-3 * a * b, 1e-2 * a + 3e-2 * b - 3e-3 * a ** 2], axis=-1)


def _data(dynamics=None):
    lo, hi = np.zeros(2), np.full(2, 0.5)
    u = design_points("lhs", 40, 2, seed=1)
    model, _, val = fit_model(2 * u - 1, _map(lo + u * (hi - lo)), 40, "poly", 3, 4, 0, np.array([1e-3, 1e-3]),
                              ["tip_x", "tip_z"])
    data = {"format": sg.FORMAT, "inputs": [{"name": "a", "unit": "N", "lower": 0.0, "upper": 0.5},
                                            {"name": "b", "unit": "N", "lower": 0.0, "upper": 0.5}],
            "outputs": [{"name": "tip_x", "unit": "m"}, {"name": "tip_z", "unit": "m"}], "model": model,
            "validation": val}
    if dynamics:
        data["dynamics"] = dynamics
    return data


class Plant:
    """First-order lag (time constant tau) towards a static map that is 25 % stiffer than the surrogate's."""

    def __init__(self, gain=0.8, tau=0.05):
        self.gain, self.tau, self.y = gain, tau, np.zeros(2)

    def step(self, u, dt=DT, substeps=20):
        for _ in range(substeps):
            self.y = self.y + (dt / substeps) / self.tau * (_map(np.asarray(u), self.gain) - self.y)
        return self.y.copy()


def _run(ctl, plant, targets, steps_each=150):
    err = []
    for r in targets:
        for _ in range(steps_each):
            u = ctl.step(plant.y, r)
            plant.step(u)
        err.append(np.abs(plant.y - r))
    return err


def test_feedforward_alone_leaves_the_error_and_feedback_removes_it():
    sur = Surrogate(_data())
    r = _map(np.array([0.3, 0.2]))                    # reachable on the surrogate; the plant has 20 % less gain
    ff = _run(Controller(sur, dt=DT, mode="ff"), Plant(), [r])[0]
    fb = _run(Controller(sur, dt=DT, ki=4.0), Plant(), [r])[0]
    assert ff.max() > 1e-3                            # about 20 % of 8 mm
    assert fb.max() < 1e-5


def test_step_and_ramp_tracking_converges_with_the_dict_api():
    ctl = Controller(Surrogate(_data()), dt=DT, ki=5.0)
    plant = Plant()
    r0 = _map(np.array([0.2, 0.3]))
    for _ in range(200):
        u = ctl.step({"tip_x": plant.y[0], "tip_z": plant.y[1]}, {"tip_x": r0[0], "tip_z": r0[1]})
        plant.step(u)
    assert np.abs(plant.y - r0).max() < 1e-5
    lag = []
    for k in range(200):                              # slow ramp towards another reachable point
        r = r0 + (_map(np.array([0.4, 0.1])) - r0) * k / 199
        plant.step(ctl.step(plant.y, r))
        lag.append(np.abs(plant.y - r).max())
    assert max(lag[100:]) < 5e-4                      # ramp following error is small against ~1.5 cm of travel
    assert lag[-1] < 5e-4


def test_anti_windup_at_the_bounds():
    ctl = Controller(Surrogate(_data()), dt=DT, ki=10.0)
    plant = Plant()
    unreachable = np.array([0.5, 0.5])                # far beyond what 0.5 N can do
    for _ in range(400):
        u = ctl.step(plant.y, unreachable)
        plant.step(u)
        assert np.all(u >= -1e-12) and np.all(u <= 0.5 + 1e-12)
    assert ctl.info["saturated"]
    reachable = _map(np.array([0.1, 0.1]))
    n_back = None
    for k in range(300):
        plant.step(ctl.step(plant.y, reachable))
        if np.abs(plant.y - reachable).max() < 1e-4:
            n_back = k
            break
    assert n_back is not None and n_back < 150        # no wound-up integrator to unwind
    assert np.all(np.abs(ctl.v) <= 0.5 * 0.5 + 1e-12)


def test_reset_clears_the_integrator_and_bounds_are_tightened():
    ctl = Controller(Surrogate(_data()), dt=DT, bounds={"a": (0.0, 0.2)})
    plant = Plant()
    for _ in range(50):
        u = ctl.step(plant.y, _map(np.array([0.45, 0.3])))
        plant.step(u)
        assert u[0] <= 0.2 + 1e-12
    assert np.any(ctl.v != 0)
    ctl.reset()
    assert np.all(ctl.v == 0)


def _linear_dynamics():
    """DMD model of a discrete first-order plant y+ = 0.8 y + 0.2 gain (u - u0), fitted by fit_dmd from step responses."""
    G = np.array([[0.02, -0.01], [0.01, 0.03]])
    trajs, us = [], np.array([[0.4, 0.1], [0.1, 0.4], [0.3, 0.3]])
    for u in us:
        y, tr = np.zeros(2), []
        for _ in range(40):
            y = 0.8 * y + 0.2 * G @ u
            tr.append(y.copy())
        trajs.append(np.array(tr))
    m = fit_dmd(trajs, us, np.zeros(2), delay=1)
    return {"type": "dmd_control", "dt": DT, "delay": 1, "x0": [0.0, 0.0], "u0": [0.0, 0.0], "A": m["A"].tolist(),
            "B": m["B"].tolist(), "input_names": ["a", "b"], "output_names": ["tip_x", "tip_z"]}, G


def test_mpc_on_a_linear_plant_with_model_mismatch():
    dyn, G = _linear_dynamics()
    data = _data(dyn)
    ctl = Controller(Surrogate(data), dt=DT, mode="mpc", horizon=8)
    y, r = np.zeros(2), np.array([5e-3, 6e-3])
    for _ in range(80):
        u = ctl.step(y, r)
        y = 0.8 * y + 0.2 * (0.85 * G) @ u            # the plant is 15 % off the model
    assert np.abs(y - r).max() < 1e-5                 # offset-free: the disturbance estimate removes the mismatch
    assert np.all(u >= -1e-12) and np.all(u <= 0.5 + 1e-12)
    with pytest.raises(ConfigError, match="dynamics"):
        Controller(Surrogate(_data()), mode="mpc")
    with pytest.raises(ConfigError, match="sample period"):
        Controller(Surrogate(data), dt=0.05, mode="mpc")


def test_bad_inputs_raise_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Controller.from_json(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="JSON"):
        Controller.from_json(bad)
    bad.write_text(json.dumps({"format": sg.FORMAT, "inputs": []}))
    with pytest.raises(ConfigError, match="valid surrogate"):
        Controller.from_json(bad)
    bad.write_text(json.dumps([1, 2]))
    with pytest.raises(ConfigError):
        Controller.from_json(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_data()))
    ctl = Controller.from_json(good, dt=DT)
    with pytest.raises(ConfigError, match="needs 2 values"):
        ctl.step([0.0], [0.0, 0.0])
    with pytest.raises(ConfigError, match="lacks"):
        ctl.step({"tip_x": 0.0}, [0.0, 0.0])
    with pytest.raises(ConfigError, match="not finite"):
        ctl.step([0.0, np.nan], [0.0, 0.0])
    with pytest.raises(ConfigError, match="no output"):
        Controller.from_json(good, outputs=["nope"])
    with pytest.raises(ConfigError, match="unique"):
        Controller.from_json(good, outputs=["tip_x"])
    with pytest.raises(ConfigError, match="mode"):
        Controller.from_json(good, mode="pid")
    with pytest.raises(ConfigError, match="dt"):
        Controller.from_json(good, dt=0)


@pytest.mark.slow
def test_mujoco_cantilever_closed_loop_with_a_stiffer_beam(tmp_path):
    """Surrogate fitted on E = 500 kPa; the plant is 20 % stiffer: feedback removes the error feedforward leaves."""
    from softsim.control.inverse import build_simulation
    from softsim.control.surrogate import run_surrogate_job

    over = ["body.mesh.size=6mm", "simulation.timestep=1ms", "body.material.damping={model: rayleigh, beta: 0.01}"]
    job = tmp_path / "job.yaml"
    job.write_text(json.dumps({"config": str(ROOT / "scenarios/actuation/cable_cantilever.yaml"), "overrides": over[:1],
                               "inputs": {"cable": ["0 N", "0.7 N"]}, "outputs": ["bend_angle"], "samples": 6, "folds": 3,
                               "model": "poly", "degree": 3, "output": str(tmp_path / "s")}))
    run_surrogate_job(job)
    path = tmp_path / "s" / "surrogate.json"
    target = 0.5 * Surrogate.load(path).predict([0.7])[0, 0]
    final = {}
    for mode in ("ff", "integral"):
        sim = build_simulation(ROOT / "scenarios/actuation/cable_cantilever.yaml", over + ["body.material.elastic.youngs_modulus=600kPa"])
        ctl = Controller.from_json(path, dt=0.02, ki=6.0, mode=mode, scales={"bend_angle": 0.005})
        sim.reset()
        u = ctl.step([0.0], [target])
        for _ in range(60):
            sim.apply_controls({"cable": float(u[0])})
            sim.step(20)
            u = ctl.step([sim.metrics()["bend_angle"]], [target])
        final[mode] = abs(sim.metrics()["bend_angle"] - target)
    assert final["ff"] > 0.1 * target and final["integral"] < 0.1 * final["ff"]


def test_ros_controller_node_compiles_without_ros():
    """The ROS 2 wrapper cannot be imported here (no rclpy); at least parse it and check it only calls the API."""
    import ast

    src = (ROOT / "ros2/softsim_ros/softsim_ros/controller_node.py").read_text()
    tree = ast.parse(src)
    assert any(isinstance(n, ast.ClassDef) and n.name == "ControllerNode" for n in ast.walk(tree))
    for call in ("from_json", "step", "reset"):
        assert f".{call}(" in src
    assert "controller_node = softsim_ros.controller_node:main" in (ROOT / "ros2/softsim_ros/setup.py").read_text()

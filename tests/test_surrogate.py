"""Reduced-order surrogates (softsim.control.surrogate): an analytic function is recovered, the pipeline resumes, bad
jobs raise ConfigError; a tiny MuJoCo run and the full cantilever are the slow tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")

import softsim.control.surrogate as sg  # noqa: E402
from softsim.control.surrogate import (Surrogate, SurrogateJob, design_points, dmd_rollout, dmd_rollout_error,  # noqa: E402
                                       fit_dmd, fit_model, solve_on_surrogate)
from softsim.errors import ConfigError  # noqa: E402

ROOT = Path(__file__).parents[1]
CANTILEVER = ROOT / "scenarios" / "actuation" / "cable_cantilever.yaml"


def _f(x):
    """An analytic two-input map with a mild nonlinearity: tip_x [m], bend_angle [rad]."""
    a, b = x[..., 0], x[..., 1]
    return np.stack([1e-2 * a + 4e-3 * b - 2e-3 * a * b + 1e-3 * a ** 2, 0.1 * a - 0.05 * b + 0.02 * a ** 2], axis=-1)


def _analytic_surrogate(n=24, holdout=6):
    lo, hi = np.zeros(2), np.array([0.5, 1.0])
    u = design_points("lhs", n + holdout, 2, seed=1)
    X = lo + u * (hi - lo)
    Y = _f(X)
    Z = 2 * u - 1
    model, degree, val = fit_model(Z, Y, n, "poly", "auto", 4, 0, np.array([1e-3, 1e-2]), ["tip_x", "bend_angle"])
    data = {"format": sg.FORMAT, "inputs": [{"name": "a", "unit": "N", "lower": 0.0, "upper": 0.5},
                                            {"name": "b", "unit": "N", "lower": 0.0, "upper": 1.0}],
            "outputs": [{"name": "tip_x", "unit": "m"}, {"name": "bend_angle", "unit": "rad"}], "model": model,
            "validation": val}
    return Surrogate(data), degree, val


def test_analytic_function_is_recovered_with_held_out_error():
    sur, degree, val = _analytic_surrogate()
    assert degree == 2                                   # the true degree, picked by cross-validation
    assert val["holdout"]["tip_x"]["max"] < 1e-12 and val["cv"]["bend_angle"]["rmse"] < 1e-12
    x = np.random.default_rng(3).random((50, 2)) * [0.5, 1.0]
    assert np.allclose(sur.predict(x), _f(x), atol=1e-12)


def test_rbf_is_close_on_a_smooth_function_and_extrapolation_is_refused():
    lo, hi = np.zeros(2), np.array([0.5, 1.0])
    u = design_points("lhs", 30, 2, seed=2)
    Y = _f(lo + u * (hi - lo))
    model, _, val = fit_model(2 * u - 1, Y, 24, "rbf", "auto", 4, 0, np.array([1e-3, 1e-2]), ["tip_x", "bend_angle"])
    assert val["holdout"]["bend_angle"]["rmse_rel_range"] < 0.02
    sur = Surrogate({"format": sg.FORMAT, "inputs": [{"name": "a", "unit": "N", "lower": 0.0, "upper": 0.5},
                                                     {"name": "b", "unit": "N", "lower": 0.0, "upper": 1.0}],
                     "outputs": [{"name": "tip_x"}, {"name": "bend_angle"}], "model": model})
    with pytest.raises(ConfigError, match="outside"):
        sur.predict([0.6, 0.5])
    assert sur.predict([0.6, 0.5], extrapolate=True).shape == (1, 2)


def test_inverse_on_the_surrogate_recovers_the_inputs():
    sur, _, _ = _analytic_surrogate()
    true = np.array([0.31, 0.62])
    t = _f(true)
    res = solve_on_surrogate(sur, {"tip_x": t[0], "bend_angle": t[1]}, scales={"tip_x": 1e-4, "bend_angle": 1e-3})
    assert res["status"] == "converged"
    assert res["values"]["a"] == pytest.approx(0.31, abs=1e-6) and res["values"]["b"] == pytest.approx(0.62, abs=1e-6)


def test_unreachable_target_gives_the_bounded_best_and_underdetermined_is_refused():
    sur, _, _ = _analytic_surrogate()
    res = solve_on_surrogate(sur, {"tip_x": 1.0, "bend_angle": 5.0})
    assert res["status"] == "bounded_best" and res["at_bounds"]
    with pytest.raises(ConfigError, match="not be unique"):
        solve_on_surrogate(sur, {"tip_x": 0.001})
    with pytest.raises(ConfigError, match="no output"):
        solve_on_surrogate(sur, {"nope": 1.0, "tip_x": 0.0})


def test_dmd_recovers_a_linear_system_and_predicts_held_out_steps():
    A = np.array([[0.9]])
    B = np.array([[0.1, 0.05]])

    def traj(u, n=40):
        x, out = np.zeros(1), []
        for _ in range(n):
            x = A @ x + B @ u
            out.append(x.copy() + 2.0)        # rest value x0 = 2
        return np.array(out)

    U = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    model = fit_dmd([traj(u) for u in U], U, np.array([2.0]), delay=0)
    assert model["A"][0, 0] == pytest.approx(0.9, abs=1e-9) and np.allclose(model["B"], B, atol=1e-9)
    u = np.array([0.3, 0.9])
    assert np.allclose(dmd_rollout(model, u, 40), traj(u), atol=1e-9)
    assert dmd_rollout_error(model, [traj(u)], [u], ["x"])["x"]["rmse"] < 1e-9


# ----------------------------------------------------------------------------- the job, with a fake simulation
class _FakeSim:
    class config:                                            # noqa: N801
        class simulation:                                    # noqa: N801
            output_every, timestep = 1, 1e-3

    class scene:                                             # noqa: N801
        actuators = []

    def __init__(self):
        self.u = {}

    def reset(self): self.u = {}
    def close(self): pass
    def step(self): pass

    def apply_controls(self, v): self.u = dict(v)

    def metrics(self):
        a, b = self.u.get("cable_a", 0.0), self.u.get("cable_b", 0.0)
        return {"tip_x": 0.01 * a + 0.004 * b, "bend_angle": 0.1 * a - 0.05 * b + 0.02 * a * a}


class _FakeEq:
    count = 0

    def __init__(self, sim, outputs, *a, **k):
        self.sim, self.outputs, self.fixed, self.unsettled = sim, outputs, {"cable_a": 0.0, "cable_b": 0.0}, 0

    def __call__(self, values):
        type(self).count += 1
        self.sim.apply_controls(values)
        m = self.sim.metrics()
        return np.array([m[o] for o in self.outputs])


@pytest.fixture
def fake(monkeypatch):
    _FakeEq.count = 0
    monkeypatch.setattr(sg, "build_simulation", lambda *a, **k: _FakeSim())
    monkeypatch.setattr(sg, "Equilibrium", _FakeEq)
    monkeypatch.setattr(sg, "actuator_dimension", lambda sim, n: "force")


def _job(tmp_path, **over):
    spec = {"config": str(CANTILEVER), "inputs": {"cable_a": ["0 N", "0.5 N"], "cable_b": ["0 N", "0.5 N"]},
            "outputs": ["tip_x", "bend_angle"], "samples": 14, "holdout": 4, "folds": 4, "output": str(tmp_path / "out")}
    spec.update(over)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_job_fits_writes_json_and_resumes(fake, tmp_path):
    r1 = SurrogateJob(_job(tmp_path)).run()
    assert r1["samples"]["simulated_now"] == 18 and _FakeEq.count == 18
    assert r1["validation"]["bend_angle"]["max"] < 1e-9
    data = json.loads((tmp_path / "out" / "surrogate.json").read_text())
    for k in ("inputs", "outputs", "model", "validation", "scenario_hash", "code_hash"):
        assert k in data
    assert data["inputs"][0]["unit"] == "N" and data["outputs"][0]["unit"] == "m"
    sur = Surrogate.load(tmp_path / "out" / "surrogate.json")
    assert sur.predict_dict({"cable_a": 0.3, "cable_b": 0.2})["tip_x"] == pytest.approx(0.0038, abs=1e-9)
    # resume: nothing is simulated again; a truncated log simulates only what is missing
    r2 = SurrogateJob(_job(tmp_path)).run()
    assert r2["samples"]["simulated_now"] == 0 and _FakeEq.count == 18
    log = tmp_path / "out" / "samples.jsonl"
    log.write_text("\n".join(log.read_text().splitlines()[:10]) + "\n")
    assert SurrogateJob(_job(tmp_path)).run()["samples"]["simulated_now"] == 8
    # a changed override is another scenario: the log is not reused
    r4 = SurrogateJob(_job(tmp_path, overrides=["body.mesh.size=5mm"])).run()
    assert r4["samples"]["simulated_now"] == 18


def test_job_with_dynamics(fake, tmp_path):
    r = SurrogateJob(_job(tmp_path, dynamics={"steps": 3, "test": 2, "duration": 0.02, "delay": 1})).run()
    assert r["dynamics"]["holdout"]["tip_x"]["rmse"] < 1e-9


def test_bad_jobs_raise_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        SurrogateJob(tmp_path / "missing.yaml")
    for over, msg in [({"samples": 2}, "samples"), ({"model": "nn"}, "model"), ({"design": "sobol"}, "design"),
                      ({"colour": 1}, "unknown job keys"), ({"outputs": []}, "outputs"), ({"inputs": {}}, "inputs"),
                      ({"folds": 99}, "folds"), ({"dynamics": {"bogus": 1}}, "dynamics"), ({"degree": "x"}, "degree")]:
        with pytest.raises(ConfigError, match=msg):
            SurrogateJob(_job(tmp_path, **over))
    p = tmp_path / "nokeys.yaml"
    p.write_text("config: x.yaml\n")
    with pytest.raises(ConfigError, match="needs"):
        SurrogateJob(p)
    with pytest.raises(ConfigError, match="scenario file"):
        SurrogateJob(_job(tmp_path, config="nope.yaml"))


def test_bad_bounds_and_outputs_raise(fake, tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match="finite"):
        SurrogateJob(_job(tmp_path, inputs={"cable_a": ["0.5 N", "0.1 N"], "cable_b": ["0 N", "1 N"]})).run()
    with pytest.raises(ConfigError, match="unknown outputs"):
        SurrogateJob(_job(tmp_path, outputs=["nope"])).run()


def test_inverse_job_errors(tmp_path):
    from softsim.control.inverse import run_job

    job = tmp_path / "inv.yaml"
    job.write_text(yaml.safe_dump({"config": str(CANTILEVER), "targets": {"bend_angle": "2 deg"},
                                  "actuators": {"cable": [0, "0.7 N"]}}))
    with pytest.raises(ConfigError, match="needs --surrogate"):
        run_job(job, "surrogate")
    with pytest.raises(ConfigError, match="not found"):
        run_job(job, "surrogate", tmp_path / "nope.json")


# ---------------------------------------------------------------------------------------- MuJoCo, coarse mesh
@pytest.mark.slow
def test_mujoco_cantilever_surrogate_end_to_end(tmp_path):
    pytest.importorskip("mujoco")
    spec = {"config": str(CANTILEVER), "overrides": ["body.mesh.size=4mm"], "inputs": {"cable": ["0 N", "0.7 N"]},
            "outputs": ["bend_angle"], "samples": 5, "holdout": 2, "folds": 3, "degree": 2, "output": str(tmp_path / "o")}
    job = tmp_path / "job.yaml"
    job.write_text(yaml.safe_dump(spec))
    res = SurrogateJob(job).run()
    assert res["validation"]["bend_angle"]["rmse_rel_range"] < 0.02       # the map is nearly linear (EB)
    assert res["samples"]["unsettled"] == 0
    assert SurrogateJob(job).run()["samples"]["simulated_now"] == 0
    inv = tmp_path / "inv.yaml"
    inv.write_text(yaml.safe_dump({"config": str(CANTILEVER), "overrides": ["body.mesh.size=4mm"],
                                   "surrogate": str(tmp_path / "o" / "surrogate.json"), "verify": True,
                                   "targets": {"bend_angle": "2 deg"}, "actuators": {"cable": ["0 N", "0.7 N"]}}))
    from softsim.control.inverse import run_job

    out = run_job(inv)   # a `surrogate:` key selects the method
    assert out["method"] == "surrogate" and out["status"] == "converged"
    assert abs(out["verified"]["errors"]["bend_angle"]) < 0.1 * np.pi / 180      # true error under 0.1 degree

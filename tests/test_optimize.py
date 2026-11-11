"""Design optimisation (softsim.design.optimize), MuJoCo, coarse cable cantilever.

The cantilever (scenarios/actuation/cable_cantilever.yaml) at 6 mm mesh and 0.3 ms steps takes ~2 s per
simulation. Its tip angle grows monotonically with the cable offset d (a pure moment T*d), which gives the
known optima: maximising the angle under an upper bound on d ends at the bound, and matching a target angle
ends at the d a bisection on the same forward map finds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from softsim.config.loader import apply_overrides
from softsim.design import Optimization
from softsim.design import optimize as opt_module
from softsim.errors import ConfigError

pytest.importorskip("mujoco")

SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "actuation" / "cable_cantilever.yaml"
D_KEY = "actuators[0].path[0][0]"
TARGET = {"metric": "final_bend_angle", "target": 0.05}   # needs many steps: the optimum is interior
COARSE = ["body.mesh.size=6mm", "simulation.timestep=0.3ms"]


def job(tmp_path: Path, **changes) -> Path:
    spec = {
        "scenario": str(SCENARIO), "backend": "mujoco", "overrides": COARSE,
        "parameters": {D_KEY: {"initial": "2 mm", "bounds": ["1 mm", "4 mm"], "tie": ["actuators[0].path[1][0]"]}},
        "objective": {"metric": "final_bend_angle", "goal": "maximize"},
        "max_evaluations": 12, "output": "out",
    }
    spec.update(changes)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


@pytest.fixture
def counter(monkeypatch):
    """Counts the simulations the optimiser really runs."""
    calls = []
    real = opt_module._run_summary

    def counting(task):
        calls.append(task)
        return real(task)

    monkeypatch.setattr(opt_module, "_run_summary", counting)
    return calls


def angle(d: float, tmp_path: Path) -> float:
    """The forward map: tip angle of the coarse cantilever at cable offset d (a single simulation)."""
    opt = Optimization(job(tmp_path))
    opt._ensure([{D_KEY: d}])
    return opt._done[opt._key({D_KEY: d})]["final_bend_angle"]


def test_maximising_angle_ends_at_the_upper_bound(tmp_path, counter):
    """Angle = f(d) increases with d, so the maximum under d <= 4 mm is the bound itself."""
    report = Optimization(job(tmp_path)).run()
    assert report["best"][D_KEY] == pytest.approx(4e-3, rel=1e-3)
    assert D_KEY in report["at_bounds"]
    assert report["feasible"] is True
    # both path points moved together (tie) and the overrides apply the design
    applied = apply_overrides(yaml.safe_load(SCENARIO.read_text()), report["overrides"])
    assert [pt[0] for pt in applied["actuators"][0]["path"]] == pytest.approx([4e-3, 4e-3])
    assert applied["actuators"][0]["path"][1][2] == "55 mm"      # the rest of the path is untouched
    out = tmp_path / "out"
    assert (out / "report.yaml").is_file() and (out / "history.parquet").is_file()
    assert len((out / "best_overrides.txt").read_text().splitlines()) == 2


@pytest.mark.slow
def test_target_angle_matches_bisection_on_the_forward_map(tmp_path):
    """Matching a target angle finds the d that bisection on the same MuJoCo forward map finds."""
    target = 0.05
    lo, hi = 1e-3, 4e-3
    sub = tmp_path / "ref"
    sub.mkdir()
    assert angle(lo, sub) < target < angle(hi, sub)
    for _ in range(9):  # bracket 3 mm / 512 = 6 um
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if angle(mid, sub) < target else (lo, mid)
    d_ref = 0.5 * (lo + hi)
    report = Optimization(job(tmp_path, objective={"metric": "final_bend_angle", "target": target},
                              max_evaluations=40)).run()
    assert report["best"][D_KEY] == pytest.approx(d_ref, abs=2e-5)
    assert report["objective"]["value"] == pytest.approx(target, abs=2e-4)


def test_resume_does_not_resimulate_logged_points(tmp_path, counter):
    first = Optimization(job(tmp_path, objective=TARGET, max_evaluations=5)).run()
    n_first = len(counter)
    assert n_first == first["simulated"] and 3 <= n_first <= 5
    # continue with a larger budget: the first evaluations replay from the log
    counter.clear()
    second = Optimization(job(tmp_path, objective=TARGET, max_evaluations=8)).run()
    assert second["evaluations"] == 8
    assert len(counter) == second["simulated"] <= 8 - first["evaluations"]
    logged = [line for line in (tmp_path / "out" / "evaluations.jsonl").read_text().splitlines() if line]
    assert len(logged) == n_first + len(counter)       # no point is in the log twice
    # an identical rerun simulates nothing
    counter.clear()
    third = Optimization(job(tmp_path, objective=TARGET, max_evaluations=8)).run()
    assert counter == [] and third["simulated"] == 0
    assert third["best"] == second["best"]


def test_changed_scenario_overrides_do_not_reuse_the_log(tmp_path, counter):
    Optimization(job(tmp_path, max_evaluations=3)).run()
    counter.clear()
    Optimization(job(tmp_path, max_evaluations=3, overrides=[*COARSE, "simulation.duration=0.45"])).run()
    assert counter, "a different simulation must not be answered from the old log"


def test_infeasible_constraint_is_reported(tmp_path, counter):
    """No d reaches a 1 m cable stroke: the report says so and returns the least-violating design."""
    path = job(tmp_path, constraints=[{"metric": "cable_stroke", "min": "1 m"}], max_evaluations=6)
    report = Optimization(path).run()
    assert report["feasible"] is False
    assert "NO FEASIBLE DESIGN" in report["warning"]
    assert report["constraints"]["cable_stroke"]["violation"] > 0
    saved = yaml.safe_load((tmp_path / "out" / "report.yaml").read_text())
    assert saved["feasible"] is False


def test_feasible_best_respects_the_constraint(tmp_path, counter):
    """Maximise the angle with the angle itself capped: the best feasible design sits just under the cap."""
    cap = 0.06
    path = job(tmp_path, constraints=[{"metric": "final_bend_angle", "max": cap}], max_evaluations=25)
    report = Optimization(path).run()
    assert report["feasible"] is True
    assert report["objective"]["value"] <= cap
    assert report["objective"]["value"] > 0.9 * cap


@pytest.mark.parametrize("change, message", [
    ({"scenario": "nope.yaml"}, "scenario file not found"),
    ({"surprise": 1}, "unknown job keys"),
    ({"parameters": {}}, "non-empty 'parameters'"),
    ({"parameters": {D_KEY: {"initial": "2 mm"}}}, "'initial' and 'bounds' are required"),
    ({"parameters": {D_KEY: {"initial": "2 mm", "bounds": ["4 mm", "1 mm"]}}}, "must be below"),
    ({"parameters": {D_KEY: {"initial": "9 mm", "bounds": ["1 mm", "4 mm"]}}}, "outside bounds"),
    ({"parameters": {D_KEY: {"initial": "2 mm", "bounds": ["1 mm", "4 N"]}}}, "different dimensions"),
    ({"parameters": {D_KEY: {"initial": "2 mm", "bounds": ["1 mm", "4 mm"], "log": True, "extra": 1}}}, "unknown fields"),
    ({"parameters": {"actuators[0].path[0][9]": {"initial": 1, "bounds": [0, 2]}}}, "does not exist"),
    ({"parameters": {"body.geometri.size": {"initial": 1, "bounds": [0, 2]}}}, "does not exist"),
    ({"objective": {"metric": "final_bend_angle"}}, "goal must be"),
    ({"objective": {"goal": "maximize"}}, "needs an 'objective'"),
    ({"constraints": [{"metric": "cable_stroke"}]}, "give min and/or max"),
    ({"constraints": [{"metric": "cable_stroke", "min": "5 mm", "max": "1 mm"}]}, "above max"),
    ({"max_evaluations": 0}, "at least 1"),
])
def test_bad_job_files_raise_clear_config_errors(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        Optimization(job(tmp_path, **change))


def test_unknown_metric_names_the_available_ones(tmp_path):
    with pytest.raises(ConfigError, match="not in the scenario's summary.*final_bend_angle"):
        Optimization(job(tmp_path, objective={"metric": "tip_wobble", "goal": "maximize"}, max_evaluations=2)).run()


def test_missing_job_file(tmp_path):
    with pytest.raises(ConfigError, match="job file not found"):
        Optimization(tmp_path / "absent.yaml")

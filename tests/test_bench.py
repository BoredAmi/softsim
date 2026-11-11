"""softsim bench: entries load, truths evaluate, a real run passes or fails against its tolerance, results are
cached and reused, a curve is read on the rising part of its ramp, and the report shows every status."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from softsim import bench
from softsim.errors import ConfigError

ROOT = Path(__file__).parents[1]


def test_shipped_entries_load_and_their_truths_evaluate():
    entries = bench.load_entries(ROOT / "benchmarks")
    assert {"tension_test", "cable_cantilever", "tip_load_cantilever", "wang_finger"} <= {e.name for e in entries}
    for e in entries:
        assert e.config.is_file(), e.config
        if e.name == "tension_test":
            pytest.importorskip("jax")
        x, y = bench.truth_values(e)
        assert np.all(np.isfinite(y)) and (x is None) == (e.kind == "scalar")
    wang = next(e for e in entries if e.name == "wang_finger")
    x, y = bench.truth_values(wang)
    assert len(x) == 8 and y[0] == 5.0 and x[-1] == 40


def test_entry_validation(tmp_path):
    base = {"name": "x", "config": str(ROOT / "scenarios/actuation/tension_test.yaml"), "metric": {"summary": "a"},
            "truth": {"value": 1.0}, "runs": [{"backend": "mujoco", "tolerance": 0.1}]}
    for change, match in (({"tier": "slow"}, "tier"), ({"metric": {}}, "metric"),
                          ({"truth": {"value": 1, "analytic": {}}}, "truth")):
        (tmp_path / "e.yaml").write_text(yaml.safe_dump({**base, **change}))
        with pytest.raises(ConfigError, match=match):
            bench.load_entry(tmp_path / "e.yaml")


def test_curve_is_read_on_the_rising_part_and_scored_by_rms(tmp_path):
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"table": [[1.0, 10.0], [2.0, 20.0]]}))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({
        "name": "c", "config": str(ROOT / "scenarios/actuation/tension_test.yaml"),
        "metric": {"curve": {"x": "p", "y": "a", "x_scale": 1e-3, "y_scale": 10.0}},
        "truth": {"dataset": {"file": "data.yaml", "key": "table"}}, "runs": [{"backend": "mujoco", "tolerance": 1}]}))
    e = bench.load_entry(tmp_path / "e.yaml")
    # rises 0 -> 3 kPa, then held (the hold's repeated x must not be used), y = x[kPa]
    ts = pd.DataFrame({"p": [0, 1000, 2000, 3000, 3000, 3000], "a": [0.0, 1.0, 2.1, 3.0, 9.0, 9.0]})
    value = bench.measure(e, {}, ts)
    np.testing.assert_allclose(value, [10.0, 21.0])
    assert bench.error(e, value) == pytest.approx(np.sqrt((0 + 1.0) / 2))


def _entry(tmp_path, name, tolerance):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({
        "name": name, "config": str(ROOT / "scenarios/actuation/tension_test.yaml"), "quantity": "pull [N]",
        "metric": {"summary": "top_final_reaction_z"},
        "truth": {"analytic": {"function": "uniaxial_force", "args": {
            "material": {"density": 1070, "elastic": {"model": "neo_hookean", "youngs_modulus": 100000,
                                                      "poisson_ratio": 0.45}},
            "stretch": 1.1, "area": 1.0e-4}}, "source": "homogeneous"},
        "runs": [{"backend": "mujoco", "tolerance": tolerance,
                  "overrides": ["body.mesh.size=5mm", "simulation.timestep=0.2ms"]}]}))
    return bench.load_entry(path)


def test_real_run_pass_fail_cache_and_report(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    pytest.importorskip("jax")
    cache = tmp_path / "cache"
    good, strict = _entry(tmp_path, "good", 0.015), _entry(tmp_path, "strict", 1e-9)
    rec = bench.run_one(good, good.runs[0], cache)
    assert rec["status"] == "pass" and abs(rec["error"]) < 0.015 and rec["n_tets"] > 0
    assert bench.run_one(strict, strict.runs[0], cache)["status"] == "fail"

    # cached: a second call doesn't simulate
    import softsim.runner

    def boom(*a, **k):
        raise AssertionError("re-ran a cached case")

    monkeypatch.setattr(softsim.runner, "Simulation", boom)
    assert bench.run_one(good, good.runs[0], cache)["error"] == rec["error"]
    # a changed run invalidates the cache entry: reported stale until re-run
    good.runs[0].overrides.append("simulation.duration=250ms")
    assert bench.cached(good, good.runs[0], cache)["status"] == "stale"
    good.runs[0].overrides.pop()
    # a changed tolerance re-judges the cached result without re-running it
    good.runs[0].tolerance = 1e-9
    assert bench.run_one(good, good.runs[0], cache)["status"] == "fail"
    good.runs[0].tolerance = 0.015

    page = bench.report([good, strict], cache)
    assert "✅ pass" in page and "❌ fail" in page and "| good | pull [N]" in page


def test_unknown_backend_is_an_error_row_not_a_crash(tmp_path):
    e = _entry(tmp_path, "nobackend", 0.1)
    e.runs[0].backend = "no_such_backend"
    e.truth = {"value": 1.0}
    rec = bench.run_one(e, e.runs[0], tmp_path / "cache")
    assert rec["status"] == "error" and "no_such_backend" in rec["reason"]
    assert "error" in bench.report([e], tmp_path / "cache")

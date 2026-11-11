"""A run that fails partway keeps what it computed: ``softsim run`` saves the samples up to the failure, marked
incomplete with the error, and still reports the failure."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from softsim.cli import main
from softsim.errors import StabilityError

BEAM = Path(__file__).parents[1] / "scenarios" / "actuation" / "cable_cantilever.yaml"


def test_a_failed_run_saves_its_samples_so_far(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mujoco")
    from softsim.runner import Simulation

    calls = {"n": 0}
    step = Simulation.step

    def failing(self, n=None):
        calls["n"] += 1
        if calls["n"] > 20:
            raise StabilityError("diverged (test)")
        return step(self, n)

    monkeypatch.setattr(Simulation, "step", failing)
    out = tmp_path / "run"
    code = main(["run", str(BEAM), "--out", str(out), "--quiet", "--set", "body.mesh.size=5mm",
                 "--set", "simulation.duration=100ms"])
    assert code == 2 and "diverged (test)" in capsys.readouterr().err
    ts = pd.read_parquet(out / "timeseries.parquet")
    assert len(ts) == 21 and ts.t.iloc[-1] == pytest.approx(0.02)  # samples 0..20 at 1 kHz
    meta = yaml.safe_load((out / "summary.yaml").read_text())["metadata"]
    assert "diverged (test)" in meta["incomplete"]


def test_a_run_that_fails_before_any_sample_saves_nothing(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    from softsim.runner import Simulation

    def broken(self):
        raise StabilityError("no state (test)")

    monkeypatch.setattr(Simulation, "metrics", broken)
    monkeypatch.setattr("softsim.scenarios.inflation.InflationScenario.metrics", lambda self, state: broken(self))
    out = tmp_path / "run"
    assert main(["run", str(BEAM), "--out", str(out), "--quiet", "--set", "body.mesh.size=5mm",
                 "--set", "simulation.duration=10ms"]) == 2
    assert not out.exists()

import numpy as np
import pytest

from softsim.cli import main
from softsim.sweep import expand

from conftest import DROP_DIR


def test_expand():
    pts = expand({"a": [1, 2], "b": ["x", "y"]}, {"c": [10, 20], "d": [0.1, 0.2]})
    assert len(pts) == 8
    assert {"a": 1, "b": "x", "c": 10, "d": 0.1} in pts
    assert {"a": 1, "b": "x", "c": 10, "d": 0.2} not in pts  # zipped keys vary together
    assert expand() == [{}]


def test_small_sweep(tmp_path):
    pytest.importorskip("mujoco")
    from softsim.sweep import run_sweep

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(f"""
base: {DROP_DIR / 'cases' / 'w020.yaml'}
overrides: [specimen.mesh.size=0.006, simulation.duration=0.006, simulation.timestep=0.0002,
            output.save_model=false]
grid:
  specimen.material.elastic.youngs_modulus: [40000, 120000]
  weight.body: [czopek2026_w020, nonexistent_weight]
""")
    df = run_sweep(sweep, tmp_path / "out")
    assert len(df) == 4 and (tmp_path / "out" / "results.parquet").exists()
    ok = df[df["error"] == ""].sort_values("specimen.material.elastic.youngs_modulus")
    bad = df[df["error"] != ""]
    assert len(ok) == 2 and len(bad) == 2 and bad["error"].str.contains("nonexistent_weight").all()
    assert ok["peak_indentation"].iloc[0] > ok["peak_indentation"].iloc[1]  # softer -> deeper


def test_cli_check_and_lists(capsys):
    pytest.importorskip("mujoco")
    assert main(["materials"]) == 0
    assert "ecoflex_00_10" in capsys.readouterr().out
    assert main(["scenarios"]) == 0
    assert main(["check", str(DROP_DIR / "cases" / "w020.yaml")]) == 0
    out = capsys.readouterr().out
    assert "mujoco: OK" in out
    assert main(["check", str(DROP_DIR / "cases" / "w020.yaml"), "--set", "specimen.material=generic_gel",
                 "--set", "specimen.mesh.size=1"]) == 2  # mesh coarser than the body -> config error


def test_sweep_samples_material_uncertainty(tmp_path):
    pytest.importorskip("mujoco")
    from softsim.sweep import run_sweep

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(f"""
base: {DROP_DIR / 'cases' / 'w020.yaml'}
overrides:
  - specimen.mesh.size=0.006
  - simulation.duration=0.006
  - simulation.timestep=0.0002
  - output.save_model=false
  - "specimen.material={{preset: ecoflex_00_10, uncertainty: {{elastic.youngs_modulus: {{rel: 0.3}}}}}}"
grid:
  drop.height: [0.2, 0.38]
sample: {{n: 3, seed: 1}}
""")
    df = run_sweep(sweep, tmp_path / "out")
    assert len(df) == 6 and (df["error"] == "").all()
    e = df["sample.elastic.youngs_modulus"]
    assert e.nunique() == 3 and e.std() > 1e3
    # the stiffest draw indents least at either height
    for _, g in df.groupby("drop.height"):
        assert g.sort_values("sample.elastic.youngs_modulus")["peak_indentation"].is_monotonic_decreasing


def test_morris_design_and_indices():
    from softsim.sensitivity import morris_design, morris_indices

    d = morris_design(3, 20, levels=4, seed=2)
    steps = np.abs(np.diff(d, axis=1))
    assert np.all((steps > 0).sum(axis=2) == 1) and np.allclose(steps.max(axis=2), 4 / 6)  # one move of Delta
    assert d.min() >= 0 and d.max() <= 1
    y = 5 * d[..., 0] + d[..., 1] ** 2 + 0 * d[..., 2]  # linear, nonlinear, inert
    mu, sigma = morris_indices(d, y)
    assert mu[0] == pytest.approx(5.0) and sigma[0] == pytest.approx(0.0, abs=1e-12)
    assert 0 < mu[1] < 2 and sigma[1] > 0 and mu[2] == 0


def test_sensitivity_cli_ranks_modulus_above_density(capsys):
    pytest.importorskip("mujoco")
    from softsim.cli import main

    rc = main(["sensitivity", str(DROP_DIR / "cases" / "w050.yaml"),
               "--param", "specimen.material.elastic.youngs_modulus=40 kPa:120 kPa",
               "--param", "specimen.material.density=900:1200",
               "--metric", "peak_indentation", "--trajectories", "2",
               "--set", "specimen.mesh.size=0.006", "--set", "simulation.duration=0.006",
               "--set", "simulation.timestep=0.0002", "--set", "output.save_model=false"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = [ln for ln in out.splitlines() if "mu*" in ln and "specimen" in ln]
    assert "youngs_modulus" in lines[0]  # sorted by influence

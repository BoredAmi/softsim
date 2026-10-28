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
    assert main(["materials"]) == 0
    assert "ecoflex_00_10" in capsys.readouterr().out
    assert main(["scenarios"]) == 0
    assert main(["check", str(DROP_DIR / "cases" / "w020.yaml")]) == 0
    out = capsys.readouterr().out
    assert "mujoco: OK" in out
    assert main(["check", str(DROP_DIR / "cases" / "w020.yaml"), "--set", "specimen.material=generic_gel",
                 "--set", "specimen.mesh.size=1"]) == 2  # mesh coarser than the body -> config error

"""The packaged example configs: they mirror scenarios/, copy cleanly and load."""

from pathlib import Path

import pytest

from softsim.cli import main
from softsim.examples import EXAMPLES_DIR, copy_examples, example_files

REPO_SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def test_examples_match_repository_scenarios():
    if not REPO_SCENARIOS.is_dir():
        pytest.skip("not a source checkout")
    for rel in example_files():
        assert (REPO_SCENARIOS / rel).read_bytes() == (EXAMPLES_DIR / rel).read_bytes(), f"{rel} drifted from scenarios/"


def test_copy_and_refuse_to_overwrite(tmp_path):
    written = copy_examples(tmp_path / "ex")
    assert len(written) == len(example_files()) > 0
    with pytest.raises(FileExistsError):
        copy_examples(tmp_path / "ex")
    copy_examples(tmp_path / "ex", overwrite=True)


def test_copied_example_validates(tmp_path, capsys):
    assert main(["examples", "copy", str(tmp_path / "ex")]) == 0
    assert main(["params", str(tmp_path / "ex" / "drop_test" / "cases" / "w020.yaml")]) == 0
    assert main(["examples", "copy", str(tmp_path / "ex")]) == 1

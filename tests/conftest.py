"""Shared fixtures. Physics tests use coarse meshes and short runs to stay fast."""

from __future__ import annotations

from pathlib import Path

import pytest

from softsim.config.loader import apply_overrides, load_yaml

ROOT = Path(__file__).resolve().parents[1]
DROP_DIR = ROOT / "scenarios" / "drop_test"


@pytest.fixture
def drop_config() -> dict:
    """20 g drop test, coarse mesh, 15 ms: enough for first impact and rebound start."""
    raw = load_yaml(DROP_DIR / "cases" / "w020.yaml")
    return apply_overrides(raw, [
        "specimen.mesh.size=0.005",
        "simulation.duration=0.015",
        "simulation.timestep=0.0002",
        "contact.timeconst=0.0002",
        "output.save_model=false",
    ])


@pytest.fixture
def drop_base_dir() -> Path:
    return DROP_DIR / "cases"

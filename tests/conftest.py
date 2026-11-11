"""Shared fixtures. Physics tests use coarse meshes and short runs to stay fast."""

from __future__ import annotations

import importlib.util
import os
import re
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


# -- optional dependencies --------------------------------------------------------------------------------------
# A test that fails because an optional package (an extra, or SOFA) is really absent is reported as skipped, naming
# the package; with the package installed the same failure is a failure. This keeps `pip install softsim` plus
# pytest green without a skip guard in every test that builds a MuJoCo model or meshes with gmsh.

OPTIONAL = {"mujoco", "scipy", "jax", "jaxlib", "gmsh", "cv2", "matplotlib", "pypardiso", "PIL", "Sofa", "SofaRuntime",
            "SofaPython3"}
_NO_MODULE = re.compile(r"No module named '([\w.]+)'")
_HINTS = {"needs JAX": "jax", "needs the gmsh package": "gmsh", "SOFA_ROOT": "Sofa"}


def _absent(module: str) -> bool:
    if module == "Sofa" and not os.environ.get("SOFA_ROOT"):
        return True
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def _optional_cause(exc: BaseException | None) -> str | None:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        names = []
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            names.append(exc.name)
        names += _NO_MODULE.findall(str(exc))
        names += [m for hint, m in _HINTS.items() if hint in str(exc)]
        for name in names:
            top = name.split(".")[0]
            if top in OPTIONAL and _absent(top):
                return top
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or report.passed or call.excinfo.errisinstance(pytest.skip.Exception):
        return
    module = _optional_cause(call.excinfo.value)
    if module:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: needs the optional package {module!r}")

"""Scenario lookup by name via the ``softsim.scenarios`` entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points

from softsim.errors import ConfigError
from softsim.scenarios.base import Scenario

_BUILTIN = {"drop_test": "softsim.scenarios.drop_test:DropTestScenario"}


def available_scenarios() -> dict[str, str]:
    """Scenario name -> import target (built-ins plus installed entry points)."""
    found = dict(_BUILTIN)
    for ep in entry_points(group="softsim.scenarios"):
        found[ep.name] = ep.value
    return found


def get_scenario(name: str) -> type[Scenario]:
    """Scenario class for ``name``."""
    targets = available_scenarios()
    if name not in targets:
        raise ConfigError(f"unknown scenario {name!r}; available: {', '.join(sorted(targets))}")
    module, _, attr = targets[name].partition(":")
    return getattr(__import__(module, fromlist=[attr]), attr)

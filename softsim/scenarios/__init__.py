"""Scenarios: configurable experiments built from bodies, loads and outputs."""

from softsim.scenarios.base import OutputSettings, Scenario, ScenarioConfig
from softsim.scenarios.registry import available_scenarios, get_scenario

__all__ = ["OutputSettings", "Scenario", "ScenarioConfig", "available_scenarios", "get_scenario"]

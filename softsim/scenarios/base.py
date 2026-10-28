"""Scenario base class.

A scenario turns a validated YAML config into a :class:`~softsim.scene.Scene`,
computes per-sample metrics from the simulation state and summarises a run.
Adding a phenomenon means adding a Scenario subclass (and registering it under
the ``softsim.scenarios`` entry point) -- no core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd

from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.errors import ConfigError
from softsim.materials import load_material
from softsim.materials.models import Material
from softsim.scene import ContactSettings, GroundSettings, Scene, SimulationSettings


class OutputSettings(StrictModel):
    """Where and what to write when a run is saved."""

    directory: str = "runs"
    format: Literal["parquet", "csv"] = "parquet"
    save_vertices: bool = False
    save_model: bool = True


class ScenarioConfig(StrictModel):
    """Sections common to every scenario config file."""

    scenario: str
    name: str | None = None
    description: str = ""
    backend: str = "mujoco"
    simulation: SimulationSettings = SimulationSettings()
    contact: ContactSettings = ContactSettings()
    ground: GroundSettings = GroundSettings()
    output: OutputSettings = OutputSettings()
    material_paths: list[str] = []


class Scenario(ABC):
    """Base class for scenarios.

    Class attributes:
        name: Registry name, equal to the ``scenario:`` key in configs.
        config_model: Pydantic model (subclass of ScenarioConfig) for the file.
        actions: Action name -> description, for runtime actions (ROS services).
    """

    name: ClassVar[str]
    config_model: ClassVar[type[ScenarioConfig]]
    actions: ClassVar[dict[str, str]] = {}

    def __init__(self, config: ScenarioConfig, base_dir: Path | None = None):
        self.config = config
        self.base_dir = base_dir or Path.cwd()

    def load_material(self, ref: str | dict) -> Material:
        """Resolve a material reference relative to the config file."""
        extra = [self.base_dir / p for p in self.config.material_paths]
        return load_material(ref, extra=extra, base_dir=self.base_dir)

    @abstractmethod
    def build_scene(self) -> Scene:
        """Scene with initial conditions for this config."""

    @abstractmethod
    def metrics(self, state: SimState) -> dict[str, float]:
        """Scalar metrics for one output sample (column name -> value)."""

    def summarize(self, timeseries: pd.DataFrame) -> dict[str, float]:
        """Scalar summary of a whole run (e.g. peak values)."""
        return {}

    def estimates(self) -> dict[str, float]:
        """Rough closed-form predictions used by the stability checks."""
        return {}

    def material_key(self, body: str) -> str:
        """Dotted config key holding ``body``'s material (used when materials change at runtime)."""
        raise ConfigError(f"scenario {self.name!r} has no soft body {body!r} with a configurable material")

    def apply_action(self, action: str, raw_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Return a modified raw config implementing ``action``; the simulation is rebuilt from it."""
        raise ConfigError(f"scenario {self.name!r} has no action {action!r}; available: {list(self.actions)}")

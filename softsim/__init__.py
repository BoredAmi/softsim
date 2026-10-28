"""softsim: data-driven simulation of soft materials and soft robots.

The core package is ROS-agnostic. Typical use::

    from softsim import run_config
    result = run_config("scenarios/drop_test/cases/w020.yaml")
    print(result.timeseries[["t", "indentation"]])
"""

from softsim.errors import ConfigError, SoftSimError, UnsupportedFeatureError

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "RunResult",
    "Simulation",
    "SoftSimError",
    "UnsupportedFeatureError",
    "run_config",
]


def __getattr__(name: str):
    # Lazy so that light submodules (config, meshing, materials) import without physics backends.
    if name in ("RunResult", "Simulation", "run_config"):
        from softsim import runner

        return getattr(runner, name)
    raise AttributeError(f"module 'softsim' has no attribute {name!r}")

"""Backend lookup by name via the ``softsim.backends`` entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points

from softsim.backends.base import Backend
from softsim.errors import ConfigError

_BUILTIN = {
    "mujoco": "softsim.backends.mujoco:MujocoBackend",
    "sofa": "softsim.backends.sofa:SofaBackend",
}


def _load(target: str) -> type[Backend]:
    module, _, attr = target.partition(":")
    mod = __import__(module, fromlist=[attr])
    return getattr(mod, attr)


def available_backends() -> dict[str, str]:
    """Backend name -> import target (built-ins plus installed entry points)."""
    found = dict(_BUILTIN)
    for ep in entry_points(group="softsim.backends"):
        found[ep.name] = ep.value
    return found


def get_backend(name: str) -> type[Backend]:
    """Backend class for ``name``."""
    targets = available_backends()
    if name not in targets:
        raise ConfigError(f"unknown backend {name!r}; available: {', '.join(sorted(targets))}")
    try:
        return _load(targets[name])
    except ImportError as exc:
        raise ConfigError(f"backend {name!r} is not installed: {exc}") from None

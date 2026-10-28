"""YAML loading with inheritance and command-line style overrides.

Conventions:
    * ``extends: <path>`` (or a list of paths) at the top of a file merges the
      referenced file(s) first; paths are relative to the extending file.
      Mappings merge recursively, everything else is replaced.
    * Overrides use dotted keys, ``"weight.preset=w050"``; the value is parsed
      as YAML, so ``"drop.height=0.2"`` gives a float and ``"a=[1, 2]"`` a list.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from softsim.errors import ConfigError


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated with ``update`` (inputs are not modified)."""
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML mapping, resolving ``extends`` recursively.

    Raises:
        ConfigError: Missing file, invalid YAML, non-mapping root or cyclic ``extends``.
    """
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"cyclic 'extends' involving {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    parents = data.pop("extends", None)
    if parents is None:
        return data
    merged: dict[str, Any] = {}
    for parent in [parents] if isinstance(parents, str) else parents:
        merged = deep_merge(merged, load_yaml(path.parent / parent, _seen | {path}))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``"a.b.c=value"`` into ``(["a", "b", "c"], parsed_value)``."""
    if "=" not in text:
        raise ConfigError(f"override must look like key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    if not key.strip():
        raise ConfigError(f"empty key in override {text!r}")
    return key.strip().split("."), yaml.safe_load(raw)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides, creating intermediate mappings as needed."""
    out = copy.deepcopy(dict(config))
    for text in overrides:
        keys, value = parse_override(text)
        node = out
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return out


def set_dotted(config: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``config`` with ``key`` (dotted path) set to ``value``."""
    out = copy.deepcopy(dict(config))
    keys = key.split(".")
    node = out
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value
    return out

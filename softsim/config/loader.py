"""YAML loading with inheritance and command-line style overrides.

Conventions:
    * ``extends: <path>`` (or a list of paths) at the top of a file merges the
      referenced file(s) first; paths are relative to the extending file.
      Mappings merge recursively, everything else is replaced.
    * Overrides use dotted keys, ``"weight.preset=w050"``; the value is parsed
      as YAML, so ``"drop.height=0.2"`` gives a float and ``"a=[1, 2]"`` a list.
    * Descending into a string (a preset reference such as ``material:
      ecoflex_00_10``) turns it into ``{preset: <name>}`` first, so
      ``specimen.material.elastic.youngs_modulus=40e3`` overrides one field of
      the preset instead of discarding it.
    * A final key may index a list: ``specimen.material.elastic.mu[0]=2e4``.
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


def _split_index(key: str) -> tuple[str, int | None]:
    """``"mu[0]"`` -> ``("mu", 0)``; ``"mu"`` -> ``("mu", None)``."""
    if key.endswith("]") and "[" in key:
        name, idx = key[:-1].split("[", 1)
        return name, int(idx)
    return key, None


def _set_path(node: dict[str, Any], keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        child = node.get(k)
        if isinstance(child, str):
            child = {"preset": child}  # expand a preset reference instead of discarding it
        elif not isinstance(child, dict):
            child = {}
        node[k] = child
        node = child
    name, idx = _split_index(keys[-1])
    if idx is None:
        node[name] = value
        return
    items = list(node.get(name) or [])
    items.extend([None] * (idx + 1 - len(items)))
    items[idx] = value
    node[name] = items


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted ``key=value`` overrides, creating intermediate mappings as needed."""
    out = copy.deepcopy(dict(config))
    for text in overrides:
        keys, value = parse_override(text)
        _set_path(out, keys, value)
    return out


def set_dotted(config: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``config`` with ``key`` (dotted path) set to ``value``."""
    out = copy.deepcopy(dict(config))
    _set_path(out, key.split("."), value)
    return out

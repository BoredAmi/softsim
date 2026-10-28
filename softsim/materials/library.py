"""Material preset database.

Presets are YAML files named ``<name>.yaml``. Search order: directories in
``$SOFTSIM_MATERIAL_PATH`` (colon separated), extra directories passed by the
caller, then the built-in ``softsim/materials/presets``.

A material reference in a config can be:
    * a preset name: ``ecoflex_00_10``
    * a path to a YAML file: ``./my_material.yaml``
    * an inline mapping (full material), or
    * a mapping with ``preset: <name>`` plus fields to override, e.g.
      ``{preset: ecoflex_00_10, elastic: {youngs_modulus: 40 kPa}}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, get_args

import yaml

from softsim.config.loader import deep_merge, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.materials.models import DampingModel, ElasticModel, Material

BUILTIN_DIR = Path(__file__).parent / "presets"


def search_path(extra: Iterable[str | Path] = ()) -> list[Path]:
    """Directories searched for presets, highest priority first."""
    env = [Path(p) for p in os.environ.get("SOFTSIM_MATERIAL_PATH", "").split(os.pathsep) if p]
    return env + [Path(p) for p in extra] + [BUILTIN_DIR]


def list_presets(extra: Iterable[str | Path] = ()) -> list[str]:
    """Names of all presets visible on the search path."""
    names: set[str] = set()
    for d in search_path(extra):
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.yaml"))
    return sorted(names)


def _find_preset(name: str, extra: Iterable[str | Path]) -> Path:
    for d in search_path(extra):
        candidate = d / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    raise ConfigError(f"unknown material preset {name!r}; available: {', '.join(list_presets(extra))}")


def material_data(ref: str | Mapping[str, Any], extra: Iterable[str | Path] = (),
                  base_dir: Path | None = None) -> dict[str, Any]:
    """Resolve a material reference to a raw (unvalidated) mapping."""
    if isinstance(ref, str):
        path = Path(ref)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        if ref.endswith((".yaml", ".yml")):
            return load_yaml(path)
        return load_yaml(_find_preset(ref, extra))
    if isinstance(ref, Mapping):
        data = dict(ref)
        preset = data.pop("preset", None)
        if preset is None:
            return data
        base = material_data(preset, extra, base_dir)
        # Switching a model type keeps only the inherited parameters the new model
        # understands: stvk keeps E and nu from neo_hookean, mooney_rivlin keeps none.
        for section in ("elastic", "damping"):
            new, old = data.get(section), base.get(section)
            if isinstance(new, Mapping) and isinstance(old, Mapping) and "model" in new \
                    and new["model"] != old.get("model"):
                allowed = _model_fields(section, new["model"])
                base[section] = {k: v for k, v in old.items() if k in allowed and k != "model"}
        return deep_merge(base, data)
    raise ConfigError(f"material must be a preset name, file path or mapping, got {type(ref).__name__}")


def _model_fields(section: str, model: str) -> set[str]:
    """Field names of the ``elastic``/``damping`` model class called ``model`` (empty if unknown)."""
    union = {"elastic": ElasticModel, "damping": DampingModel}[section]
    for cls in get_args(get_args(union)[0]):
        if cls.model_fields["model"].default == model:
            return set(cls.model_fields)
    return set()


def load_material(ref: str | Mapping[str, Any], extra: Iterable[str | Path] = (),
                  base_dir: Path | None = None) -> Material:
    """Resolve and validate a material reference."""
    return validate(Material, material_data(ref, extra, base_dir), where=f"material {ref!r:.60}")


def save_material(material: Material, path: str | Path) -> Path:
    """Write a material as a preset YAML (SI units), e.g. after calibration."""
    path = Path(path)
    data = material.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path

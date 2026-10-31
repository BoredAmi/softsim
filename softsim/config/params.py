"""Every changeable value of a config or material: dotted key, value, unit, allowed range.

Keys are the dotted paths that overrides use, e.g. ``specimen.material.elastic.c10``
or ``specimen.material.damping.tau[1]``, so anything listed here can be set with
``--set key=value`` (CLI), ``overrides=[...]`` (Python) or a sweep grid.
``softsim params <config>`` prints them; :func:`find_param` resolves one key.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Iterator, Literal, Union, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from softsim.config.units import Unit

# card metadata, not physics: not listed as parameters
SKIPPED_FIELDS = frozenset({"provenance", "uncertainty", "references"})

Resolver = Callable[[str, Any], "BaseModel | None"]


@dataclass(frozen=True)
class Param:
    """One changeable value."""

    key: str
    value: Any
    unit: Unit | None = None
    bounds: str = ""
    choices: tuple[Any, ...] = ()

    def describe(self) -> str:
        """``value unit`` as stored (SI)."""
        if self.value is None:
            return "unset"
        text = f"{self.value:.6g}" if isinstance(self.value, float) else str(self.value)
        return f"{text} {self.unit.symbol}" if self.unit else text


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def _strip_optional(annotation: Any) -> Any:
    if _is_union(annotation):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _metadata(annotation: Any, extra: list[Any] = ()) -> tuple[Unit | None, list[Any], tuple[Any, ...]]:
    """Unit marker, numeric constraints and Literal choices found in ``annotation``
    (through Annotated, Optional and nested Field metadata)."""
    unit: Unit | None = None
    constraints: list[Any] = []
    choices: tuple[Any, ...] = ()

    def take(meta: Any) -> None:
        nonlocal unit
        if isinstance(meta, Unit):
            unit = meta
        elif isinstance(meta, FieldInfo):
            for m in meta.metadata:
                take(m)
        elif isinstance(meta, (Gt, Ge, Lt, Le)):
            constraints.append(meta)

    def visit(a: Any) -> None:
        nonlocal choices
        origin = get_origin(a)
        if origin is Annotated:
            base, *metas = get_args(a)
            visit(base)
            for m in metas:
                take(m)
        elif _is_union(a):
            for x in get_args(a):
                if x is not type(None):
                    visit(x)
        elif origin is Literal:
            choices = get_args(a)

    visit(annotation)
    for m in extra:
        take(m)
    return unit, constraints, choices


def _bounds_text(constraints: list[Any]) -> str:
    parts = []
    for c in constraints:
        if isinstance(c, Gt):
            parts.append(f"> {c.gt:g}")
        elif isinstance(c, Ge):
            parts.append(f">= {c.ge:g}")
        elif isinstance(c, Lt):
            parts.append(f"< {c.lt:g}")
        elif isinstance(c, Le):
            parts.append(f"<= {c.le:g}")
    return ", ".join(parts)


def _element_annotation(annotation: Any, index: int) -> Any:
    annotation = _strip_optional(annotation)
    while get_origin(annotation) is Annotated:
        annotation = _strip_optional(get_args(annotation)[0])
    args = get_args(annotation)
    if get_origin(annotation) is tuple and args:
        return args[0] if len(args) == 2 and args[1] is Ellipsis else args[min(index, len(args) - 1)]
    return args[0] if args else Any


def _model_choices(annotation: Any) -> tuple[str, ...]:
    """``model`` names of a (discriminated) union of models, for the discriminator field."""
    names = []
    for a in get_args(_strip_optional(annotation)) or ():
        while get_origin(a) is Annotated:
            a = get_args(a)[0]
        for member in (get_args(a) if _is_union(a) else (a,)):
            fields = getattr(member, "model_fields", {})
            if "model" in fields and fields["model"].default is not None:
                names.append(fields["model"].default)
    return tuple(names)


def _values(key: str, value: Any, annotation: Any, extra_meta: list[Any], resolve: Resolver | None,
            model_choices: tuple[str, ...] = ()) -> Iterator[Param]:
    if isinstance(value, BaseModel):
        yield from iter_params(value, key + ".", resolve, model_choices)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _values(f"{key}[{i}]", item, _element_annotation(annotation, i), [], resolve)
    elif isinstance(value, dict):
        return  # free-form mapping (e.g. an unresolved preset reference)
    else:
        unit, constraints, choices = _metadata(annotation, extra_meta)
        yield Param(key, value, unit, _bounds_text(constraints), choices)


def iter_params(model: BaseModel, prefix: str = "", resolve: Resolver | None = None,
                model_choices: tuple[str, ...] = ()) -> Iterator[Param]:
    """All changeable values of a validated pydantic model, depth first.

    ``resolve(key, value)`` may return a model to descend into instead of
    ``value`` (e.g. the material a preset name refers to).
    """
    for name, field in type(model).model_fields.items():
        if name in SKIPPED_FIELDS:
            continue
        value = getattr(model, name)
        key = f"{prefix}{name}"
        sub = resolve(key, value) if resolve else None
        if sub is not None:
            yield from iter_params(sub, key + ".", resolve)
            continue
        if name == "model" and model_choices:
            yield Param(key, value, choices=model_choices)
            continue
        yield from _values(key, value, field.annotation, list(field.metadata), resolve,
                           _model_choices(field.annotation))


def find_param(model: BaseModel, key: str) -> Param:
    """The parameter at dotted ``key`` (KeyError if there is none)."""
    for p in iter_params(model):
        if p.key == key:
            return p
    raise KeyError(key)


def config_params(config: str, overrides: list[str] | tuple[str, ...] = (),
                  backend: str | None = None) -> list[Param]:
    """Every changeable value of a scenario config file, with materials resolved."""
    from pathlib import Path

    from softsim.config.loader import apply_overrides, load_yaml
    from softsim.config.schema import validate
    from softsim.scenarios import get_scenario

    path = Path(config).resolve()
    raw = apply_overrides(load_yaml(path), overrides)
    if backend:
        raw["backend"] = backend
    scenario_cls = get_scenario(raw.get("scenario", ""))
    cfg = validate(scenario_cls.config_model, raw, where="config")
    scenario = scenario_cls(cfg, path.parent)

    def resolve(key: str, value: Any) -> BaseModel | None:
        if key.rsplit(".", 1)[-1] == "material" and isinstance(value, (str, dict)):
            return scenario.load_material(value)
        return None

    return list(iter_params(cfg, resolve=resolve))

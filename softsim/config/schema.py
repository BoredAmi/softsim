"""Shared pydantic base model and validation helper."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from softsim.errors import ConfigError

M = TypeVar("M", bound=BaseModel)


class StrictModel(BaseModel):
    """Base for all config models: unknown keys are errors (typos must not pass silently)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def validate(model: type[M], data: Any, where: str = "config") -> M:
    """Validate ``data`` against ``model``, raising ConfigError with readable paths."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError(f"invalid {where}:\n" + "\n".join(lines)) from None

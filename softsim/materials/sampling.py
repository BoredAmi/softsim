"""Draw material variants from a card's ``uncertainty`` (for sweeps and robustness studies)."""

from __future__ import annotations

import math

import numpy as np

from softsim.config.loader import get_dotted
from softsim.errors import ConfigError
from softsim.materials.models import Material, ParameterUncertainty

MAX_TRIES = 100


def _draw(value: float, u: ParameterUncertainty, rng: np.random.Generator) -> float:
    spread = u.rel * abs(value) if u.rel is not None else float(u.abs)
    if u.dist == "normal":
        return value + spread * rng.standard_normal()
    if u.dist == "uniform":
        return value + spread * rng.uniform(-1.0, 1.0)
    # lognormal: median = value, log-space sigma from the relative spread
    if value <= 0:
        raise ConfigError(f"lognormal uncertainty needs a positive value, got {value}")
    return value * math.exp(spread / value * rng.standard_normal())


def sample(material: Material, n: int, seed: int | None = None) -> list[Material]:
    """``n`` materials with every uncertain parameter drawn independently.

    Draws that make the card invalid (e.g. a Poisson ratio >= 0.5) are redrawn.
    The samples record their draw in ``provenance["sample"]``.

    Raises:
        ConfigError: The card has no ``uncertainty``, or valid draws are too rare.
    """
    if not material.uncertainty:
        raise ConfigError(f"material {material.name!r} has no uncertainty to sample from")
    rng = np.random.default_rng(seed)
    data = material.model_dump(mode="json", exclude_none=True)
    out = []
    for i in range(n):
        for _ in range(MAX_TRIES):
            draws = {k: _draw(float(get_dotted(data, k)), u, rng) for k, u in material.uncertainty.items()}
            try:
                m = material.with_overrides([f"{k}={v!r}" for k, v in draws.items()])
            except ConfigError:
                continue
            break
        else:
            raise ConfigError(f"no valid draw for sample {i} of {material.name!r} in {MAX_TRIES} tries")
        prov = {**m.provenance, "sample": {"index": i, "seed": seed, **draws}}
        out.append(m.model_copy(update={"provenance": prov}))
    return out

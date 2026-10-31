"""Reference finite-element solver (axisymmetric, locking-free, JAX tangents).

Optional: needs ``jax`` (``pip install -e ".[fem]"``).
"""

from __future__ import annotations

from softsim.errors import SoftSimError


def jax_setup():
    """Import JAX with float64 enabled; raise a clear error if it is missing."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SoftSimError("the reference FEM needs JAX: pip install -e '.[fem]'") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp

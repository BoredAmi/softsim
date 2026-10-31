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
    _enable_compilation_cache(jax)
    return jax, jnp


def _enable_compilation_cache(jax) -> None:
    """Keep compiled kernels on disk (``~/.cache/softsim/jax``, or ``$SOFTSIM_JAX_CACHE``; ``0`` turns it
    off): repeated runs on the same mesh, e.g. a calibration's evaluations, skip part of the compilation."""
    import os
    from pathlib import Path

    where = os.environ.get("SOFTSIM_JAX_CACHE", str(Path.home() / ".cache" / "softsim" / "jax"))
    if where in ("", "0") or jax.config.jax_compilation_cache_dir:
        return
    try:
        jax.config.update("jax_compilation_cache_dir", where)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except Exception:  # noqa: BLE001 - an older JAX without the persistent cache: compile as before
        pass

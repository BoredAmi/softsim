"""Homogeneous material tests: exact stresses of a material card under uniform deformation.

What a card does in the standard tests of a materials lab, without any mesh or
backend: uniaxial tension/compression, equibiaxial tension, pure (planar)
shear, simple shear, volumetric and confined compression, and any strain
history with the card's Prony series (relaxation, constant-rate loading, DMA).
Used to plot cards (``softsim material plot``), to fit cards to material-test
data, and as the ground truth for single-element backend checks.

The energies and the viscoelastic update are those of the reference FEM
(:mod:`softsim.fem.materials`, :mod:`softsim.fem.axisym`): isochoric law +
Simo-Taylor volumetric energy, Prony series on the isochoric part (instantaneous
moduli = the card's elastic constants). Each step minimises the incremental
potential over the free lateral stretch, so lateral faces are traction free.
Rayleigh and Kelvin-Voigt damping are numerical damping, not material models,
and are ignored here. Needs the ``fem`` extra (JAX).

Modes and their controlled quantity ``x``:

==============  =======================  ==============================  ========================
mode            deformation gradient     x                               ``stress``
==============  =======================  ==============================  ========================
uniaxial        diag(x, f, f)            stretch (< 1: compression)      nominal P_11
equibiaxial     diag(x, x, f)            in-plane stretch                nominal P_11
planar          diag(x, 1, f)            stretch (pure shear test)       nominal P_11
simple_shear    I + x e1 (x) e2          shear strain gamma              nominal P_12
volumetric      x^(1/3) I                volume ratio J                  mean Cauchy stress
confined        diag(x, 1, 1)            stretch (uniaxial strain)       nominal P_11
==============  =======================  ==============================  ========================
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from softsim.fem import jax_setup
from softsim.fem.materials import from_material, volumetric_energy
from softsim.materials.models import Material, PronySeries

jax, jnp = jax_setup()

Mode = Literal["uniaxial", "equibiaxial", "planar", "simple_shear", "volumetric", "confined"]
MODES: tuple[str, ...] = ("uniaxial", "equibiaxial", "planar", "simple_shear", "volumetric", "confined")
_FREE = {"uniaxial": -0.5, "equibiaxial": -2.0, "planar": -1.0}  # free lateral stretch, first guess x**p
_NEWTON = 40


def _deformation(mode: str, x, f):
    one = jnp.ones_like(x)
    if mode == "uniaxial":
        return jnp.diag(jnp.stack([x, f, f]))
    if mode == "equibiaxial":
        return jnp.diag(jnp.stack([x, x, f]))
    if mode == "planar":
        return jnp.diag(jnp.stack([x, one, f]))
    if mode == "simple_shear":
        z = jnp.zeros_like(x)
        return jnp.stack([jnp.stack([one, x, z]), jnp.stack([z, one, z]), jnp.stack([z, z, one])])
    if mode == "volumetric":
        j3 = x ** (1.0 / 3.0)
        return jnp.diag(jnp.stack([j3, j3, j3]))
    if mode == "confined":
        return jnp.diag(jnp.stack([x, one, one]))
    raise ValueError(f"unknown mode {mode!r}; one of {', '.join(MODES)}")


def _potential(law, kappa, f_grad, c_iso, b):
    """Incremental potential per reference volume: c W_iso(C) + 1/2 B:C + U(J)."""
    c = f_grad.T @ f_grad
    return c_iso * law.energy(c) + 0.5 * jnp.sum(b * c) + volumetric_energy(jnp.linalg.det(f_grad), kappa)


def _solve(mode, law, kappa, x, c_iso, b, s0):
    """Free stretch f = exp(s) minimising the potential (traction-free lateral faces)."""
    if mode not in _FREE:
        return s0

    def pi(s):
        return _potential(law, kappa, _deformation(mode, x, jnp.exp(s)), c_iso, b)

    grad = jax.grad(pi)
    hess = jax.grad(grad)

    def body(_, s):
        g, h = grad(s), hess(s)
        step = jnp.where(h > 0, g / h, jnp.sign(g) * 0.1)
        return s - jnp.clip(step, -0.5, 0.5)

    return jax.lax.fori_loop(0, _NEWTON, body, s0)


def _outputs(mode, law, kappa, x, c_iso, b, s):
    f_grad = _deformation(mode, x, jnp.exp(s))
    p = jax.grad(lambda fg: _potential(law, kappa, fg, c_iso, b))(f_grad)
    j = jnp.linalg.det(f_grad)
    sigma = p @ f_grad.T / j
    if mode == "simple_shear":
        nominal, cauchy = p[0, 1], sigma[0, 1]
    elif mode == "volumetric":
        nominal = cauchy = jnp.trace(sigma) / 3.0
    else:
        nominal, cauchy = p[0, 0], sigma[0, 0]
    return nominal, cauchy, jnp.exp(s)


def _iso_pk2(law, f_grad):
    return 2.0 * jax.grad(law.energy)(f_grad.T @ f_grad)


def _first_guess(mode: str, x):
    return _FREE.get(mode, 0.0) * jnp.log(x) if mode in _FREE else jnp.zeros_like(x)


def _curve_impl(mode, law, kappa, x, c_iso):
    zero = jnp.zeros((3, 3))

    def one(xi):
        s = _solve(mode, law, kappa, xi, c_iso, zero, _first_guess(mode, xi))
        return _outputs(mode, law, kappa, xi, c_iso, zero, s)

    return jax.vmap(one)(x)


_curve = jax.jit(_curve_impl, static_argnums=0)


def _history_impl(mode, law, kappa, g, tau, t, x):
    """Time stepping with the reference solver's Prony update (see AxisymSolver._viscous_terms)."""
    n = g.shape[0]
    dts = jnp.diff(t)

    def step(carry, inp):
        q, s_prev, s_free = carry
        dt, xi = inp
        r = dt / tau
        a = -jnp.expm1(-r) / r
        c_iso = 1.0 - jnp.sum(g * (1.0 - a))
        b = jnp.sum(jnp.exp(-r)[:, None, None] * q, axis=0) - jnp.sum(g * a) * s_prev
        s_free = _solve(mode, law, kappa, xi, c_iso, b, s_free)
        out = _outputs(mode, law, kappa, xi, c_iso, b, s_free)
        s_new = _iso_pk2(law, _deformation(mode, xi, jnp.exp(s_free)))
        q = jnp.exp(-r)[:, None, None] * q + (g * a)[:, None, None] * (s_new - s_prev)[None]
        return (q, s_new, s_free), out

    x0 = x[0]
    s0 = _solve(mode, law, kappa, x0, 1.0, jnp.zeros((3, 3)), _first_guess(mode, x0))
    s_iso0 = _iso_pk2(law, _deformation(mode, x0, jnp.exp(s0)))
    first = _outputs(mode, law, kappa, x0, 1.0, jnp.zeros((3, 3)), s0)
    _, rest = jax.lax.scan(step, (jnp.zeros((n, 3, 3)), s_iso0, s0), (dts, x[1:]))
    return tuple(jnp.concatenate([jnp.atleast_1d(a0), a]) for a0, a in zip(first, rest))


_history = jax.jit(_history_impl, static_argnums=0)


def _card(material: Material):
    if material.damping is not None and not isinstance(material.damping, PronySeries):
        material = material.model_copy(update={"damping": None})  # numerical damping: not a material response
    law, kappa, prony = from_material(material)
    g = jnp.asarray(prony.g if prony else (), dtype=float)
    tau = jnp.asarray(prony.tau if prony else (), dtype=float)
    return law, float(kappa), g, tau


def _frame(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in columns.items()})


def curve(material: Material, mode: Mode, x, response: Literal["instantaneous", "relaxed"] = "relaxed") -> pd.DataFrame:
    """Rate-independent stress-strain curve.

    ``response="relaxed"`` is the equilibrium (infinitely slow) curve, i.e. the
    long-term moduli of a Prony card; ``"instantaneous"`` the infinitely fast
    one. Identical for cards without Prony damping.

    Returns columns ``x``, ``stress`` (nominal, see the module table),
    ``cauchy`` and ``lateral`` (the free stretch, 1 where there is none).
    """
    law, kappa, g, _ = _card(material)
    c_iso = 1.0 - float(jnp.sum(g)) if response == "relaxed" else 1.0
    x = jnp.atleast_1d(jnp.asarray(x, dtype=float))
    nominal, cauchy, lateral = _curve(mode, law, kappa, x, c_iso)
    return _frame({"x": x, "stress": nominal, "cauchy": cauchy, "lateral": lateral})


def history(material: Material, mode: Mode, t, x) -> pd.DataFrame:
    """Stress under a prescribed history ``x(t)`` (starting from rest), with Prony viscoelasticity.

    Returns columns ``t``, ``x``, ``stress``, ``cauchy``, ``lateral``. The time
    steps are those of ``t``: resolve the fastest Prony time and the loading.
    """
    law, kappa, g, tau = _card(material)
    t = jnp.asarray(t, dtype=float)
    x = jnp.asarray(x, dtype=float)
    if t.shape != x.shape or t.ndim != 1 or len(t) < 2:
        raise ValueError("t and x must be 1-D arrays of the same length (>= 2)")
    nominal, cauchy, lateral = _history(mode, law, kappa, g, tau, t, x)
    return _frame({"t": t, "x": x, "stress": nominal, "cauchy": cauchy, "lateral": lateral})


def _rest(mode: str) -> float:
    return 0.0 if mode == "simple_shear" else 1.0


def relaxation(material: Material, strain: float, mode: Mode = "uniaxial", ramp: float = 1e-4,
               hold: float = 10.0, n: int = 400) -> pd.DataFrame:
    """Ramp to ``strain`` (stretch - 1, or shear strain) in ``ramp`` seconds, then hold.

    Time steps are linear during the ramp and logarithmic during the hold, so
    every Prony time between ``ramp`` and ``hold`` is resolved.
    """
    x0 = _rest(mode)
    t_ramp = np.linspace(0.0, ramp, 21)
    t_hold = ramp + np.geomspace(ramp / 10.0, hold, n)
    t = np.concatenate([t_ramp, t_hold])
    x = x0 + strain * np.minimum(t / ramp, 1.0)
    return history(material, mode, t, x)


def constant_rate(material: Material, mode: Mode, x_end: float, rate: float, n: int = 200) -> pd.DataFrame:
    """Load from rest to ``x_end`` at constant ``rate`` (dx/dt, e.g. strain rate in 1/s)."""
    x0 = _rest(mode)
    t = np.linspace(0.0, abs(x_end - x0) / rate, n)
    return history(material, mode, t, np.linspace(x0, x_end, n))


def dma(material: Material, frequencies, amplitude: float = 0.002, mode: Mode = "uniaxial", cycles: int = 6,
        steps_per_cycle: int = 64) -> pd.DataFrame:
    """Dynamic mechanical analysis: small sinusoidal strain at each frequency [Hz].

    Fits the last cycle's stress to a sin + b cos: storage = a / amplitude,
    loss = b / amplitude. Uniaxial gives E' and E'', simple shear G' and G''.
    """
    rows = []
    x0 = _rest(mode)
    for f in np.atleast_1d(frequencies):
        t = np.linspace(0.0, cycles / f, cycles * steps_per_cycle + 1)
        ph = 2.0 * math.pi * f * t
        d = history(material, mode, t, x0 + amplitude * np.sin(ph))
        last = slice(-steps_per_cycle - 1, None)
        basis = np.c_[np.sin(ph[last]), np.cos(ph[last]), np.ones(steps_per_cycle + 1)]
        a, b, _ = np.linalg.lstsq(basis, d.stress.to_numpy()[last], rcond=None)[0]
        rows.append({"frequency": f, "storage": a / amplitude, "loss": b / amplitude, "tan_delta": b / a})
    return pd.DataFrame(rows)


def dma_small_strain(material: Material, frequencies, mode: Literal["uniaxial", "simple_shear"] = "uniaxial") -> pd.DataFrame:
    """Closed-form small-strain storage and loss moduli of the card's Prony series.

    G*(w) = G0 (1 - sum g_i + sum g_i i w tau_i / (1 + i w tau_i)) on the shear
    part, the bulk modulus elastic; uniaxial E* = 9 K G* / (3 K + G*).
    """
    _, kappa, g, tau = _card(material)
    g, tau = np.asarray(g), np.asarray(tau)
    mu0 = material.elastic.initial_shear_modulus()
    w = 2.0 * math.pi * np.atleast_1d(np.asarray(frequencies, dtype=float))
    iwt = 1j * w[:, None] * tau[None, :]
    gstar = mu0 * (1.0 - g.sum() + (g[None, :] * iwt / (1.0 + iwt)).sum(axis=1))
    mstar = gstar if mode == "simple_shear" else 9.0 * kappa * gstar / (3.0 * kappa + gstar)
    return pd.DataFrame({"frequency": w / (2.0 * math.pi), "storage": mstar.real, "loss": mstar.imag,
                         "tan_delta": mstar.imag / mstar.real})

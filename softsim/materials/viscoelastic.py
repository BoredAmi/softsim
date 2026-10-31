"""Prony viscoelasticity on tetrahedral backends: Maxwell overstresses as explicit vertex forces.

A Prony card (``damping: {model: prony}``) splits into:

* the **long-term law**, simulated by the backend's own implicit elasticity
  (:func:`long_term_material`: the isochoric energy scaled by 1 - sum g,
  the bulk modulus unchanged);
* the **Maxwell-branch stresses** Q_i of every linear tetrahedron, which this
  module integrates and turns into vertex forces each step
  (:class:`PronyOverstress`).

The total stress is that of the reference FEM, S = (1 - sum g) S_iso(C) + sum Q_i,
with the same update Q_i <- e^(-dt/tau_i) Q_i + g_i a_i (S_iso(C) - S_iso(C_prev)),
a_i = (1 - e^(-dt/tau_i)) / (dt/tau_i), and the same isochoric energies
(:mod:`softsim.fem.materials`). The overstress lags one step (it is computed
from the configuration at the start of the step), so it is explicit: its
stiffness sum g_i mu0 must be resolved by the time step (checked by
``softsim.checks``). Needs the ``fem`` extra (JAX).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from softsim.errors import ConfigError, UnsupportedFeatureError
from softsim.materials.models import (
    ArrudaBoyce,
    Gent,
    Material,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    PronySeries,
    StableNeoHookean,
    Yeoh,
    isotropic_e_nu,
)
from softsim.scene import Scene


def prony_of(material: Material) -> PronySeries | None:
    return material.damping if isinstance(material.damping, PronySeries) else None


def scale_isochoric(elastic, c: float):
    """The elastic law with its isochoric (shear) part scaled by ``c``, the bulk modulus unchanged."""
    e = elastic
    if isinstance(e, (NeoHookean, StableNeoHookean)):
        young, nu = isotropic_e_nu({"shear_modulus": c * e.initial_shear_modulus(),
                                    "bulk_modulus": e.initial_bulk_modulus()})
        return e.model_copy(update={"youngs_modulus": young, "poisson_ratio": nu})
    if isinstance(e, MooneyRivlin):
        return e.model_copy(update={"c10": c * e.c10, "c01": c * e.c01})
    if isinstance(e, Yeoh):
        return e.model_copy(update={"c10": c * e.c10, "c20": c * e.c20, "c30": c * e.c30})
    if isinstance(e, Ogden):
        return e.model_copy(update={"mu": [c * m for m in e.mu]})
    if isinstance(e, (Gent, ArrudaBoyce)):
        return e.model_copy(update={"mu": c * e.mu})
    raise UnsupportedFeatureError("prony", [f"Prony viscoelasticity on {e.model!r} elasticity (it needs a law "
                                            "with an isochoric/volumetric split)"])


def long_term_material(material: Material) -> Material:
    """The card with its isochoric energy relaxed by 1 - sum g and no damping (the backend's native part)."""
    prony = prony_of(material)
    if prony is None:
        return material
    elastic = scale_isochoric(material.elastic, 1.0 - sum(prony.g))
    return material.model_copy(update={"elastic": elastic, "damping": None})


def prony_from_fractional(n: float, tau: float, band: tuple[float, float] = (0.1, 2000.0), reference: float = 0.06,
                          per_decade: int = 1) -> tuple[list[float], list[float], float]:
    """Prony moduli matching a fractional Kelvin-Voigt shear modulus mu0 (1 + (i w tau)^n) over ``band`` [Hz].

    Returns ``(a, taus, rms)``: branch moduli ``a`` relative to the storage modulus at the
    ``reference`` frequency (default 0.06 Hz, a tension test at ~0.4/s), so that
    G*(w) / G'(reference) ~ 1 + sum a_i (i w t_i / (1 + i w t_i) - its value at the reference).
    The branch times 1 / (2 pi f) are a grid with ``per_decade`` points over the band, so no
    branch is faster than the band's top (keep it below 1 / (2 pi dt) of the backends);
    ``rms`` is the fit's relative error over the band. Fractional models stiffen without
    bound; the Prony series follows them only inside the band and saturates above it.
    """
    from scipy.optimize import nnls

    def frac(f):
        return 1.0 + (1j * 2.0 * math.pi * np.asarray(f, float) * tau) ** n

    f = np.geomspace(*band, 60)
    w, w_ref = 2.0 * math.pi * f, 2.0 * math.pi * reference
    target = frac(f) / frac(reference).real
    lo, hi = 1.0 / (2.0 * math.pi * band[1]), 1.0 / (2.0 * math.pi * band[0])
    taus = np.geomspace(lo, hi, int(round(per_decade * math.log10(hi / lo))) + 1)
    branch = lambda ww: 1j * ww[:, None] * taus[None, :] / (1.0 + 1j * ww[:, None] * taus[None, :])  # noqa: E731
    basis = branch(w) - branch(np.array([w_ref])).real
    scale = np.abs(target)[:, None]
    a_mat = np.vstack([(basis.real / scale), (basis.imag / scale)])
    rhs = np.concatenate([(target.real - 1.0) / scale[:, 0], target.imag / scale[:, 0]])
    a, _ = nnls(a_mat, rhs)
    rms = float(np.sqrt(np.mean((a_mat @ a - rhs) ** 2)))
    keep = a > 1e-4
    return [float(x) for x in a[keep]], [float(x) for x in taus[keep]], rms


def with_fast_relaxation(material: Material, a: list[float], tau: list[float]) -> Material:
    """Add fast Prony branches (moduli ``a`` relative to the card's current instantaneous shear modulus,
    as from :func:`prony_from_fractional`) without changing the response at the reference rate: the
    instantaneous law stiffens by 1 + sum a, every g is rescaled, the long-term modulus is unchanged."""
    s = 1.0 + sum(a)
    old = prony_of(material)
    g = [x / s for x in a] + ([x / s for x in old.g] if old else [])
    t = list(tau) + (list(old.tau) if old else [])
    order = np.argsort(t)
    damping = {"model": "prony", "g": [float(g[i]) for i in order], "tau": [float(t[i]) for i in order]}
    data = material.model_dump(mode="json", exclude_none=True)
    data["elastic"] = scale_isochoric(material.elastic, s).model_dump(mode="json", exclude_none=True)
    data["damping"] = damping
    return type(material).model_validate(data)


def native_scene(scene: Scene) -> Scene:
    """``scene`` with every Prony material replaced by its long-term part (what a backend builds natively)."""
    if not any(prony_of(b.material) for b in scene.soft_bodies):
        return scene
    bodies = tuple(dataclasses.replace(b, material=long_term_material(b.material)) if prony_of(b.material) else b
                   for b in scene.soft_bodies)
    return dataclasses.replace(scene, soft_bodies=bodies)


def overstress_stiffness(material: Material) -> float:
    """Shear stiffness carried by the explicit Maxwell branches, sum g * mu0 [Pa]."""
    prony = prony_of(material)
    return sum(prony.g) * material.elastic.initial_shear_modulus() if prony else 0.0


def max_stable_timestep(material: Material, density: float, min_edge: float) -> float:
    """dt with omega * dt = 1 for the explicit overstress on the smallest element (omega ~ 2 c / h)."""
    k = overstress_stiffness(material)
    return math.inf if k <= 0 else min_edge / (2.0 * math.sqrt(k / density))


class PronyOverstress:
    """Maxwell-branch stresses of one soft body's linear tetrahedra.

    Args:
        material: The Prony card (its instantaneous law drives the branches).
        rest: Rest vertex positions (any rigid placement; e.g. the mesh in the body frame).
        tets: (m, 4) vertex indices.
    """

    def __init__(self, material: Material, rest: np.ndarray, tets: np.ndarray):
        try:
            from softsim.fem import jax_setup
            from softsim.fem.materials import from_material
        except ImportError:
            raise ConfigError("Prony viscoelasticity on mujoco/sofa needs the fem extra (pip install -e '.[fem]')") \
                from None
        jax, jnp = jax_setup()
        law, _, prony = from_material(material)
        if prony is None:
            raise ConfigError(f"material {material.name!r} has no Prony series")
        self.g = np.asarray(prony.g, float)
        self.tau = np.asarray(prony.tau, float)
        self.tets = np.asarray(tets, int)
        dm = self._edges(np.asarray(rest, float))
        self.dm_inv = np.linalg.inv(dm)
        self.volume = np.abs(np.linalg.det(dm)) / 6.0
        self.n_vertices = len(rest)
        self._s_iso = jax.jit(jax.vmap(lambda c: 2.0 * jax.grad(law.energy)(c)))
        self._jnp = jnp
        self.reset()

    def _edges(self, x: np.ndarray) -> np.ndarray:
        """(m, 3, 3) edge matrices with columns x1 - x0, x2 - x0, x3 - x0."""
        t = self.tets
        return np.stack([x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]], x[t[:, 3]] - x[t[:, 0]]], axis=-1)

    def reset(self) -> None:
        m = len(self.tets)
        self.q = np.zeros((len(self.g), m, 3, 3))
        self.s_prev = np.zeros((m, 3, 3))

    def forces(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """Advance the branches by ``dt`` to the configuration ``positions``; return (n, 3) vertex forces."""
        f_grad = self._edges(np.asarray(positions, float)) @ self.dm_inv
        c = np.transpose(f_grad, (0, 2, 1)) @ f_grad
        s_new = np.asarray(self._s_iso(self._jnp.asarray(c)))
        r = dt / self.tau
        a = -np.expm1(-r) / r
        self.q = np.exp(-r)[:, None, None, None] * self.q + (self.g * a)[:, None, None, None] * (s_new - self.s_prev)
        self.s_prev = s_new
        p = f_grad @ self.q.sum(axis=0)                                   # first Piola-Kirchhoff overstress
        h = -self.volume[:, None, None] * (p @ np.transpose(self.dm_inv, (0, 2, 1)))
        out = np.zeros((self.n_vertices, 3))
        for j in range(3):
            np.add.at(out, self.tets[:, j + 1], h[:, :, j])
        np.add.at(out, self.tets[:, 0], -h.sum(axis=2))
        return out

    @property
    def stored_stress(self) -> np.ndarray:
        """Current total overstress (second Piola-Kirchhoff) per tetrahedron."""
        return self.q.sum(axis=0)

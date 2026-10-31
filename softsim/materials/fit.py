"""Fit material cards to measured test curves, and project a card onto another law.

:func:`fit` adjusts one constitutive law so that the card's homogeneous-test
response (:mod:`softsim.materials.homogeneous`) matches measured nominal
stress-stretch curves, by bounded nonlinear least squares on residuals scaled
by each curve's largest stress. It reports:

* the fitted card;
* parameter uncertainties from the residual-scaled Jacobian. These cover the
  misfit to the given curves only, not specimen-to-specimen scatter;
* AIC/BIC for comparing laws;
* where the fitted law stays stable (stress increasing with stretch in
  uniaxial, equibiaxial and planar loading).

:func:`project` fits a law to another card's homogeneous curves, e.g. to run a
Yeoh card on a backend that only has (stable) Neo-Hookean. The projection
error is recorded in provenance, never applied silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from softsim.errors import SoftSimError
from softsim.experiments.material_tests import TestCurve
from softsim.materials.models import Material, PronySeries

PENALTY = 10.0  # residual of an invalid parameter set (per point, in units of the curve's peak stress)
MAX_POINTS = 150  # each curve is resampled to at most this many points


@dataclass(frozen=True)
class _P:
    """A fitted parameter: card key (under ``elastic.``), internal coordinate and first guess."""

    key: str
    kind: str                           # "log": value = exp(u); "lin": value = u * scale; "gt1log": 1 + exp(u)
    guess: Callable[[float], float]     # first guess from the small-strain shear modulus estimate
    scale: Callable[[float], float] = lambda mu: 1.0


def _ogden(n: int) -> list[_P]:
    alphas = (2.0,) if n == 1 else (1.5, 6.0)
    shares = (1.0,) if n == 1 else (0.95, 0.05)
    out = []
    for k, (a, s) in enumerate(zip(alphas, shares)):
        # parametrised by the term's shear contribution m_k = mu_k alpha_k / 2 > 0 (stable) and alpha_k
        out += [_P(f"ogden_m[{k}]", "log", lambda mu, s=s: s * mu), _P(f"alpha[{k}]", "lin", lambda mu, a=a: a)]
    return out


LAWS: dict[str, tuple[str, list[_P]]] = {
    "neo_hookean": ("stable_neo_hookean", [_P("shear_modulus", "log", lambda mu: mu)]),
    "mooney_rivlin": ("mooney_rivlin", [_P("c10", "lin", lambda mu: 0.4 * mu, lambda mu: mu),
                                        _P("c01", "lin", lambda mu: 0.1 * mu, lambda mu: mu)]),
    "yeoh": ("yeoh", [_P("c10", "log", lambda mu: 0.5 * mu),
                      _P("c20", "lin", lambda mu: 0.0, lambda mu: 0.01 * mu),
                      _P("c30", "lin", lambda mu: 0.0, lambda mu: 1e-4 * mu)]),
    "ogden1": ("ogden", _ogden(1)),
    "ogden2": ("ogden", _ogden(2)),
    "gent": ("gent", [_P("mu", "log", lambda mu: mu), _P("jm", "log", lambda mu: 30.0)]),
    "arruda_boyce": ("arruda_boyce", [_P("mu", "log", lambda mu: mu), _P("lambda_m", "gt1log", lambda mu: 4.0)]),
}


def _to_internal(p: _P, value: float, mu: float) -> float:
    if p.kind == "log":
        return math.log(value)
    if p.kind == "gt1log":
        return math.log(value - 1.0)
    return value / p.scale(mu)


def _to_value(p: _P, u: float, mu: float) -> float:
    if p.kind == "log":
        return math.exp(u)
    if p.kind == "gt1log":
        return 1.0 + math.exp(u)
    return u * p.scale(mu)


def _elastic_section(model: str, params: list[_P], values: Sequence[float], poisson_ratio: float) -> dict[str, Any]:
    v = dict(zip((p.key for p in params), values))
    if model == "stable_neo_hookean":
        return {"model": model, "shear_modulus": v["shear_modulus"], "poisson_ratio": poisson_ratio}
    if model == "ogden":
        n = sum(1 for k in v if k.startswith("alpha"))
        alpha = [v[f"alpha[{k}]"] for k in range(n)]
        mu = [2.0 * v[f"ogden_m[{k}]"] / a if a != 0 else math.inf for k, a in enumerate(alpha)]
        return {"model": model, "mu": mu, "alpha": alpha, "poisson_ratio": poisson_ratio}
    return {"model": model, **v, "poisson_ratio": poisson_ratio}


@dataclass
class FitResult:
    law: str
    material: Material
    values: dict[str, float]               # card keys (e.g. "elastic.c10") -> fitted value
    sigma: dict[str, float]                # one standard deviation, same keys
    nrmse: float                           # rms residual / peak stress, over all points
    max_error: float                       # largest |model - data| / peak stress
    aic: float
    bic: float
    n_points: int
    stable: bool
    stability: dict[str, float] = field(default_factory=dict)  # mode -> first stretch where stress decreases
    model_curves: list[tuple[TestCurve, np.ndarray]] = field(default_factory=list)


def _resample(c: TestCurve) -> TestCurve:
    if len(c.stretch) <= MAX_POINTS:
        return c
    x = np.linspace(c.stretch.min(), c.stretch.max(), MAX_POINTS)
    return TestCurve(x, np.interp(x, c.stretch, c.stress), None, c.material, c.mode, c.rate, dict(c.meta))


def _model_stress(card: Material, c: TestCurve) -> np.ndarray:
    from softsim.materials.homogeneous import constant_rate, curve

    if isinstance(card.damping, PronySeries) and c.rate:
        d = constant_rate(card, c.mode, float(c.stretch.max()), c.rate, n=max(len(c.stretch), 50))
        return np.interp(c.stretch, d.x, d.stress)
    return curve(card, c.mode, c.stretch, "relaxed").stress.to_numpy()


# lowest controlled stretch per mode whose largest principal stretch is still <= stretch_max
_MODE_FLOOR = {"uniaxial": lambda m: m**-2.0, "equibiaxial": lambda m: m**-0.5, "planar": lambda m: 1.0 / m}


def stability_limits(card: Material, stretch_max: float, stretch_min: float = 0.4) -> dict[str, float]:
    """First stretch (per mode) where the nominal stress stops increasing: Drucker instability.

    Each mode is checked from ``stretch_min`` (or higher, so that no principal
    stretch exceeds ``stretch_max``) to ``stretch_max``. Returns only the modes
    that become unstable there.
    """
    from softsim.materials.homogeneous import curve

    out = {}
    for mode, floor in _MODE_FLOOR.items():
        x = np.linspace(max(stretch_min, floor(stretch_max)), stretch_max, 200)
        s = curve(card, mode, x, "relaxed").stress.to_numpy()
        d = np.diff(s)
        # non-finite: beyond a limiting extensibility (Gent, Arruda-Boyce), which locks rather than softens
        bad = np.nonzero(np.isfinite(d) & (d <= 0))[0]
        if len(bad):
            out[mode] = float(x[bad[0]])
    return out


def fit(curves: Iterable[TestCurve], law: str, base: Material, poisson_ratio: float = 0.49,
        weights: Sequence[float] | None = None, stability_range: tuple[float, float] | None = None) -> FitResult:
    """Fit ``law`` (a key of :data:`LAWS`) to ``curves`` on top of ``base`` (density, damping, surface...).

    Args:
        curves: Measured nominal stress vs stretch, each with its ``mode``.
        law: neo_hookean, mooney_rivlin, yeoh, ogden1, ogden2, gent or arruda_boyce.
        base: Card providing everything but the elastic law.
        poisson_ratio: Fixed during the fit. Tension tests barely constrain it; 0.49 is
            nearly incompressible yet tractable for tetrahedral backends.
        weights: Per-curve weights (default 1).
        stability_range: Stretch range checked for Drucker stability (default 0.4 to the largest
            fitted stretch, at least 2: a law fitted to compression alone must not blow up in
            moderate tension).
    """
    from scipy.optimize import least_squares

    if law not in LAWS:
        raise SoftSimError(f"unknown law {law!r}; one of {', '.join(LAWS)}")
    model, params = LAWS[law]
    curves = [_resample(c) for c in curves]
    if not curves or any(len(c.stretch) < 3 for c in curves):
        raise SoftSimError("every curve needs at least 3 points in the fitted range")
    weights = list(weights) if weights is not None else [1.0] * len(curves)
    scales = [float(np.abs(c.stress).max()) for c in curves]
    mu = _shear_estimate(curves)
    u0 = np.array([_to_internal(p, p.guess(mu), mu) for p in params])

    def card_for(u) -> Material:
        values = [_to_value(p, ui, mu) for p, ui in zip(params, u)]
        data = base.model_dump(mode="json", exclude_none=True)
        data["elastic"] = _elastic_section(model, params, values, poisson_ratio)
        data.pop("uncertainty", None)
        return Material.model_validate(data)

    def residuals(u) -> np.ndarray:
        try:
            card = card_for(u)
            parts = [np.sqrt(w / len(c.stretch)) * (_model_stress(card, c) - c.stress) / s
                     for c, w, s in zip(curves, weights, scales)]
            r = np.concatenate(parts)
        except (SoftSimError, ValueError, OverflowError, ZeroDivisionError):
            return np.full(sum(len(c.stretch) for c in curves), PENALTY)
        return np.where(np.isfinite(r), r, PENALTY)

    res = least_squares(residuals, u0, method="trf", x_scale="jac", max_nfev=400 * len(u0))
    card = card_for(res.x)
    m, k = sum(len(c.stretch) for c in curves), len(params)

    # uncertainties: residual-scaled covariance in internal coordinates, mapped to card values
    s2 = 2.0 * res.cost / max(m - k, 1)
    cov_u = s2 * np.linalg.pinv(res.jac.T @ res.jac)
    keys, values, jv = _card_values(card, params, res.x, mu, model)
    cov_v = jv @ cov_u @ jv.T
    sigma = {key: float(math.sqrt(max(cov_v[i, i], 0.0))) for i, key in enumerate(keys)}

    pred = [_model_stress(card, c) for c in curves]
    rel = np.concatenate([(p - c.stress) / s for p, c, s in zip(pred, curves, scales)])
    rss = float(np.sum(rel**2))
    lo, hi = stability_range or (0.4, max(2.0, max(float(c.stretch.max()) for c in curves)))
    unstable = stability_limits(card, hi, lo)
    return FitResult(
        law=law, material=card, values=dict(zip(keys, values)), sigma=sigma,
        nrmse=float(math.sqrt(rss / m)), max_error=float(np.abs(rel).max()),
        aic=m * math.log(rss / m) + 2 * k, bic=m * math.log(rss / m) + k * math.log(m), n_points=m,
        stable=not unstable, stability=unstable, model_curves=list(zip(curves, pred)))


def _shear_estimate(curves: list[TestCurve]) -> float:
    """Small-strain shear modulus guess: E/3 from the initial slope of the first tension-type curve."""
    for c in curves:
        sel = np.abs(c.stretch - 1.0) < 0.2
        if sel.sum() >= 3:
            slope = np.polyfit(c.stretch[sel] - 1.0, c.stress[sel], 1)[0]
            if slope > 0:
                return float(slope / (2.0 if c.mode == "simple_shear" else 3.0))
    c = curves[0]
    i = int(np.argmax(np.abs(c.stretch - 1.0)))
    return float(abs(c.stress[i] / (c.stretch[i] - 1.0)) / 3.0) or 1e4


def _card_values(card: Material, params: list[_P], u: np.ndarray, mu: float, model: str):
    """Card keys, values and d(value)/d(u) (finite differences through the card mapping)."""

    def values_of(uu):
        vals = [_to_value(p, ui, mu) for p, ui in zip(params, uu)]
        sec = _elastic_section(model, params, vals, 0.49)
        sec.pop("model")
        sec.pop("poisson_ratio")
        out = {}
        for key, v in sec.items():
            if isinstance(v, list):
                out.update({f"elastic.{key}[{i}]": float(x) for i, x in enumerate(v)})
            else:
                out[f"elastic.{key}"] = float(v)
        if model == "stable_neo_hookean":  # stored as E (nu fixed)
            g = out.pop("elastic.shear_modulus")
            out["elastic.youngs_modulus"] = 2.0 * g * (1.0 + card.elastic.poisson_ratio)
        return out

    base = values_of(u)
    keys = list(base)
    jv = np.zeros((len(keys), len(u)))
    for j in range(len(u)):
        h = 1e-6 * max(1.0, abs(u[j]))
        up = u.copy()
        up[j] += h
        moved = values_of(up)
        jv[:, j] = [(moved[k] - base[k]) / h for k in keys]
    return keys, [base[k] for k in keys], jv


def fit_laws(curves: Sequence[TestCurve], laws: Iterable[str], base: Material, **kwargs) -> list[FitResult]:
    """:func:`fit` for several laws, best (lowest AIC) first; stable fits before unstable ones."""
    results = [fit(curves, law, base, **kwargs) for law in laws]
    return sorted(results, key=lambda r: (not r.stable, r.aic))


def with_fit_uncertainty(result: FitResult) -> Material:
    """The fitted card with its parameter uncertainties as ``uncertainty`` entries."""
    unc = {}
    for key, s in result.sigma.items():
        v = result.values[key]
        if s > 0 and math.isfinite(s):
            unc[key] = {"rel": s / abs(v)} if v > 0 else {"abs": s}
    data = result.material.model_dump(mode="json", exclude_none=True)
    data["uncertainty"] = unc
    return Material.model_validate(data)


def project(card: Material, law: str, strain_range: tuple[float, float] = (-0.4, 1.0),
            modes: Sequence[str] = ("uniaxial", "equibiaxial", "planar"), n: int = 60) -> FitResult:
    """Fit ``law`` to ``card``'s own homogeneous curves over ``strain_range`` (engineering strain).

    The result keeps everything but the elastic law (density, damping, surface,
    kind); its provenance records the source and the projection error. Equibiaxial
    and planar curves cover the tensile part of the range only. The instantaneous
    (elastic-constant) curves are matched, so a Prony series carries over unchanged.
    """
    from softsim.materials.homogeneous import curve

    curves = []
    for mode in modes:
        lo = strain_range[0] if mode == "uniaxial" else max(strain_range[0], 0.0)
        x = 1.0 + np.linspace(lo, strain_range[1], n)
        x = x[np.abs(x - 1.0) > 1e-6]
        curves.append(TestCurve(x, curve(card, mode, x, "instantaneous").stress.to_numpy(), mode=mode,
                                material=card.name))
    base = card.model_copy(update={"damping": None, "uncertainty": {}})
    result = fit(curves, law, base, poisson_ratio=card.elastic.initial_poisson_ratio(),
                 stability_range=(1.0 + strain_range[0], 1.0 + strain_range[1]))
    data = result.material.model_dump(mode="json", exclude_none=True)
    if card.damping is not None:
        data["damping"] = card.damping.model_dump(mode="json")
    data["name"] = f"{card.name}_as_{law}"
    data["provenance"] = {**card.provenance, "projected_from": card.name, "source_law": card.elastic.model,
                          "projection": {"law": law, "strain_range": list(strain_range), "modes": list(modes),
                                         "nrmse": result.nrmse, "max_error": result.max_error}}
    result.material = Material.model_validate(data)
    return result


DEFAULT_TAUS = tuple(float(t) for t in np.geomspace(0.3, 3000.0, 9))  # 2 per decade, 0.3 s to 50 min


def fit_relaxation(curves: Sequence[TestCurve], taus: Sequence[float] = DEFAULT_TAUS) -> tuple[list[float], list[float], float]:
    """Prony series (g, tau) from stress relaxation at constant strain: (g, tau, rms error).

    Each curve's stress, normalised by its first point (t = 0, the start of the
    hold), is fitted with 1 - sum g_i (1 - exp(-t / tau_i)) on the grid of
    ``taus`` by non-negative least squares; terms with g = 0 are dropped. This
    is the small-strain form; the isochoric Prony series of a nearly
    incompressible card relaxes the same way at constant strain. The g are
    relative to the stress at the start of the hold: relaxation before it
    (faster than the ramp) is not measured.
    """
    from scipy.optimize import nnls

    rows, rhs = [], []
    for c in curves:
        t = np.asarray(c.time, float)
        r = c.stress / c.stress[0]
        rows.append(-(1.0 - np.exp(-t[:, None] / np.asarray(taus)[None, :])))
        rhs.append(r - 1.0)
    a, b = np.vstack(rows), np.concatenate(rhs)
    g, _ = nnls(a, b)
    rms = float(np.sqrt(np.mean((a @ g - b) ** 2)))
    keep = g > 1e-4
    if g.sum() >= 1:
        raise SoftSimError(f"relaxation fit: sum g = {g.sum():.3f} >= 1")
    return [float(x) for x in g[keep]], [float(x) for x in np.asarray(taus)[keep]], rms

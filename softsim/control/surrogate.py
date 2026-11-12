"""Reduced-order surrogates of the equilibrium map, for real-time control.

``softsim surrogate jobs/surrogate_cable_cantilever.yaml`` samples the actuator inputs on a seeded design of
experiments (a Latin hypercube plus the corners of the box), brings each sample to equilibrium with the same
:class:`~softsim.control.inverse.Equilibrium` the inverse solver uses, records the chosen outputs, fits a smooth
model ``inputs -> outputs`` (polynomial least squares or a cubic RBF with a linear tail; numpy only) and writes it as
JSON. The file holds the coefficients, the names, SI units and bounds, the validation errors, a hash of the resolved
scenario and softsim's code hash, and is evaluated by :class:`Surrogate` or by ten lines of plain numpy
(``docs/surrogate.md``), so a ROS 2 node or a browser can run it.

Validation is held out, never the fit residual: ``cv`` is the k-fold cross-validation error over the fitting samples
and, when ``holdout`` extra points are sampled (a different seeded design), ``holdout`` is the error of the model fitted
without them. Errors are RMS and max in SI units and relative to the range of the output over the samples.

``softsim inverse job.yaml --method surrogate --surrogate file.json`` solves the inverse problem on the surrogate
(bounded least squares from several starts, milliseconds) and with ``verify: true`` runs one forward equilibrium at
the solution and reports the true error.

Every simulated sample is appended to ``<output>/samples.jsonl``, keyed by the resolved scenario, the overrides, the
settling options, the code hash and the sample's inputs: a stopped job resumes, a rerun replays the log. Optional
second model (``dynamics``): a linear DMD-with-control model ``x_{k+1} = A x_k + B u_k`` from step responses (see
:func:`fit_dmd`).

Limits: a surrogate inherits the errors of the backend that produced its samples (MuJoCo's locking among them); it is
valid inside the sampled box only (a prediction outside is refused by :meth:`Surrogate.predict` unless
``extrapolate=True``); a polynomial cannot represent a kink such as the fold of an unsigned angle at zero.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.stats import qmc

from softsim.config.loader import load_yaml
from softsim.config.units import parse_quantity
from softsim.control.inverse import (_SCALES, DEFAULT_DAMPING, Equilibrium, actuator_dimension,
                                     build_simulation, output_dimension)
from softsim.errors import ConfigError

FORMAT = 1
_UNIT = {"length": "m", "angle": "rad", "force": "N", "pressure": "Pa"}
_JOB_KEYS = {"config", "overrides", "backend", "inputs", "outputs", "samples", "design", "seed", "model", "degree",
             "folds", "holdout", "scales", "settle_damping", "settle_tol", "settle_window", "max_settle", "warm_start",
             "output", "dynamics"}
_DYN_KEYS = {"steps", "test", "duration", "sample_every", "delay"}


# ---------------------------------------------------------------------------------------------------------- model
def _exponents(dim: int, degree: int) -> np.ndarray:
    """All monomial exponents of total degree <= ``degree`` in ``dim`` variables, shape (terms, dim)."""
    return np.array([e for e in itertools.product(range(degree + 1), repeat=dim) if sum(e) <= degree], dtype=int)


def _poly_features(z: np.ndarray, expo: np.ndarray) -> np.ndarray:
    return np.prod(z[:, None, :] ** expo[None, :, :], axis=2)


def _rbf_features(z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(z[:, None, :] - centers[None, :, :], axis=2)
    return np.hstack([d ** 3, np.ones((len(z), 1)), z])


def _fit_poly(z: np.ndarray, y: np.ndarray, degree: int) -> dict[str, Any]:
    expo = _exponents(z.shape[1], degree)
    if len(expo) > len(z):
        raise ConfigError(f"degree {degree} has {len(expo)} terms but there are only {len(z)} samples")
    coef = np.linalg.lstsq(_poly_features(z, expo), y, rcond=None)[0]
    return {"type": "poly", "degree": degree, "exponents": expo.tolist(), "coefficients": coef.T.tolist()}


def _fit_rbf(z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n, d = z.shape
    phi = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2) ** 3
    tail = np.hstack([np.ones((n, 1)), z])
    a = np.block([[phi, tail], [tail.T, np.zeros((d + 1, d + 1))]])
    rhs = np.vstack([y, np.zeros((d + 1, y.shape[1]))])
    w = np.linalg.lstsq(a, rhs, rcond=None)[0]
    return {"type": "rbf", "kernel": "cubic", "centers": z.tolist(), "weights": w.T.tolist()}


def _eval_model(model: Mapping[str, Any], z: np.ndarray) -> np.ndarray:
    """Normalised inputs ``z`` (rows, in [-1, 1]) -> outputs (rows, n_out); numpy only."""
    if model["type"] == "poly":
        return _poly_features(z, np.array(model["exponents"], dtype=int)) @ np.array(model["coefficients"]).T
    return _rbf_features(z, np.array(model["centers"])) @ np.array(model["weights"]).T


def _fit(z: np.ndarray, y: np.ndarray, kind: str, degree: int) -> dict[str, Any]:
    return _fit_poly(z, y, degree) if kind == "poly" else _fit_rbf(z, y)


def _cv_predictions(z: np.ndarray, y: np.ndarray, kind: str, degree: int, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions at every sample (k-fold with a seeded shuffle)."""
    idx = np.random.default_rng(seed).permutation(len(z))
    pred = np.empty_like(y)
    for part in np.array_split(idx, folds):
        train = np.setdiff1d(idx, part)
        pred[part] = _eval_model(_fit(z[train], y[train], kind, degree), z[part])
    return pred


def _errors(pred: np.ndarray, truth: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, float]]:
    """RMS and max error per output, absolute (SI) and relative to the range of the truth."""
    out = {}
    for j, n in enumerate(names):
        e = pred[:, j] - truth[:, j]
        rng = float(np.ptp(truth[:, j])) or 1.0
        out[n] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "max": float(np.abs(e).max()),
                  "rmse_rel_range": float(np.sqrt(np.mean(e ** 2)) / rng), "max_rel_range": float(np.abs(e).max() / rng)}
    return out


def fit_model(Z: np.ndarray, Y: np.ndarray, n_fit: int, kind: str = "poly", degree: int | str = "auto", folds: int = 5,
              seed: int = 0, scale: np.ndarray | None = None, names: Sequence[str] | None = None
              ) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Fit ``Y(Z)`` (``Z`` normalised to [-1, 1]) and validate it; returns (model, degree, validation).

    The first ``n_fit`` rows fit the model (degree ``auto``: the lowest k-fold cross-validation error, scaled by
    ``scale``); the rest are held out and reported. The returned model is refitted on all rows.
    """
    names = list(names or [f"y{j}" for j in range(Y.shape[1])])
    scale = np.ones(Y.shape[1]) if scale is None else np.asarray(scale)
    zf, yf = Z[:n_fit], Y[:n_fit]
    cands = [degree] if degree != "auto" else [d for d in range(1, 6)
                                               if len(_exponents(Z.shape[1], d)) <= 0.7 * n_fit]
    if kind == "rbf":
        cands = [0]
    if not cands:
        raise ConfigError("too few samples for even a linear model")
    best = None
    for d in cands:
        try:
            pred = _cv_predictions(zf, yf, kind, d, min(folds, n_fit), seed)
        except ConfigError:
            continue
        score = float(np.max(np.sqrt(np.mean(((pred - yf) / scale) ** 2, axis=0))))
        if best is None or score < best[0]:
            best = (score, d, pred)
    if best is None:
        raise ConfigError("no model could be fitted with these samples")
    _, d, cv_pred = best
    model = _fit(zf, yf, kind, d)
    validation: dict[str, Any] = {"cv": {"folds": min(folds, n_fit), **_errors(cv_pred, yf, names)},
                                  "fit_residual": _errors(_eval_model(model, zf), yf, names)}
    holdout = len(Z) - n_fit
    if holdout:
        validation["holdout"] = {"n": holdout, **_errors(_eval_model(model, Z[n_fit:]), Y[n_fit:], names)}
        model = _fit(Z, Y, kind, d)   # the held-out points go back into the fit after being reported
    key = "holdout" if holdout else "cv"
    validation["summary"] = {"source": key, **{o: {k: validation[key][o][k] for k in ("rmse", "max", "rmse_rel_range")}
                                               for o in names}}
    return model, d, validation


class Surrogate:
    """A fitted map ``inputs -> outputs``, loaded from the JSON ``softsim surrogate`` writes.

    Evaluates with numpy only. ``predict`` takes values in SI (a dict by input name, or an array in input order).
    """

    def __init__(self, data: Mapping[str, Any]):
        if data.get("format") != FORMAT:
            raise ConfigError(f"unsupported surrogate format {data.get('format')!r}")
        self.data = dict(data)
        self.inputs = [i["name"] for i in data["inputs"]]
        self.outputs = [o["name"] for o in data["outputs"]]
        self.lower = np.array([i["lower"] for i in data["inputs"]])
        self.upper = np.array([i["upper"] for i in data["inputs"]])
        self.model = data["model"]

    @classmethod
    def load(cls, path: str | Path) -> "Surrogate":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"surrogate file not found: {path}")
        return cls(json.loads(path.read_text()))

    def _array(self, x: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(x, Mapping):
            missing = [n for n in self.inputs if n not in x]
            if missing:
                raise ConfigError(f"missing inputs {missing}; the surrogate takes {self.inputs}")
            x = [x[n] for n in self.inputs]
        return np.atleast_2d(np.asarray(x, dtype=float))

    def predict(self, x: Mapping[str, float] | Sequence[float] | np.ndarray, extrapolate: bool = False) -> np.ndarray:
        """Outputs (rows, n_out) at inputs ``x`` (SI). Outside the sampled box it raises unless ``extrapolate``."""
        a = self._array(x)
        if a.shape[1] != len(self.inputs):
            raise ConfigError(f"expected {len(self.inputs)} inputs {self.inputs}, got {a.shape[1]}")
        tol = 1e-9 * (self.upper - self.lower)
        if not extrapolate and (np.any(a < self.lower - tol) or np.any(a > self.upper + tol)):
            raise ConfigError(f"input outside the sampled box [{self.lower.tolist()}, {self.upper.tolist()}]")
        return _eval_model(self.model, 2 * (a - self.lower) / (self.upper - self.lower) - 1)

    def predict_dict(self, x: Mapping[str, float]) -> dict[str, float]:
        return dict(zip(self.outputs, map(float, self.predict(x)[0])))


def save_surrogate(data: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=1, default=float))


# ----------------------------------------------------------------------------------------------- the inverse
def solve_on_surrogate(sur: Surrogate, targets: Mapping[str, float], scales: Mapping[str, float] | None = None,
                       bounds: Mapping[str, tuple[float, float]] | None = None, fixed: Mapping[str, float] | None = None,
                       starts: int = 8, tolerance: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Bounded least squares on the surrogate (SI targets). Returns values, predicted outputs, residual, status.

    The residual is scaled like :func:`~softsim.control.inverse.solve_inverse`'s. ``starts`` seeded starting points
    (the box centre and random ones) guard against local minima; the best residual wins.
    """
    fixed = dict(fixed or {})
    free = [n for n in sur.inputs if n not in fixed]
    outs = list(targets)
    for o in outs:
        if o not in sur.outputs:
            raise ConfigError(f"surrogate has no output {o!r}; it has {sur.outputs}")
    if not free:
        raise ConfigError("every input is fixed: nothing to solve for")
    if len(outs) < len(free):
        raise ConfigError(f"{len(free)} inputs but only {len(outs)} targets: the solution would not be unique")
    ix = [sur.inputs.index(n) for n in free]
    lo, hi = sur.lower[ix].copy(), sur.upper[ix].copy()
    for n, (a, b) in (bounds or {}).items():
        if n in free:
            k = free.index(n)
            lo[k], hi[k] = max(lo[k], a), min(hi[k], b)
            if hi[k] <= lo[k]:
                raise ConfigError(f"bounds of {n!r} do not overlap the surrogate's box")
    oi = [sur.outputs.index(o) for o in outs]
    tgt = np.array([targets[o] for o in outs])
    sc = np.array([(scales or {}).get(o, _SCALES[output_dimension(o)]) for o in outs])
    base = np.array([fixed.get(n, 0.0) for n in sur.inputs])

    def full(v: np.ndarray) -> np.ndarray:
        x = base.copy()
        x[ix] = v
        return x

    def res(v: np.ndarray) -> np.ndarray:
        return (sur.predict(full(v), extrapolate=True)[0][oi] - tgt) / sc

    rng = np.random.default_rng(seed)
    starts_u = np.vstack([np.full(len(free), 0.5), rng.random((max(starts - 1, 0), len(free)))])
    best = None
    for u in starts_u:
        r = least_squares(res, lo + u * (hi - lo), bounds=(lo, hi), x_scale=hi - lo, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        worst = float(np.abs(r.fun).max())
        if best is None or worst < best[0]:
            best = (worst, r.x)
    worst, v = best
    values = dict(zip(free, map(float, v)))
    at_bounds = [n for k, n in enumerate(free) if v[k] <= lo[k] + 1e-3 * (hi[k] - lo[k])
                 or v[k] >= hi[k] - 1e-3 * (hi[k] - lo[k])]
    status = "converged" if worst <= tolerance else ("bounded_best" if at_bounds else "stalled")
    pred = sur.predict(full(v), extrapolate=True)[0]
    return {"status": status, "values": values, "predicted": dict(zip(outs, (float(pred[k]) for k in oi))),
            "residual": worst, "at_bounds": at_bounds}


def run_inverse_job(path: Path, job: dict[str, Any], surrogate: str | Path | None = None) -> dict[str, Any]:
    """``softsim inverse --method surrogate``: solve on the surrogate, optionally verify with one simulation.

    Job keys used: ``targets``, ``actuators`` (bounds, tightening the surrogate's box), ``fixed`` (inputs not solved
    for), ``tolerance``, ``scales``, ``surrogate`` (file, relative to the job), ``verify`` (run one forward equilibrium
    at the solution with the job's ``config`` and settling options) and the settling keys of ``softsim inverse``.
    """
    known = {"config", "overrides", "backend", "targets", "actuators", "tolerance", "scales", "surrogate", "verify", "fixed",
             "settle_damping", "settle_tol", "settle_window", "max_settle", "warm_start", "initial", "max_evaluations",
             "jacobian_step", "starts"}
    extra = set(job) - known
    if extra:
        raise ConfigError(f"{path}: unknown keys {sorted(extra)}")
    ref = surrogate or job.get("surrogate")
    if not ref:
        raise ConfigError(f"{path}: --method surrogate needs --surrogate FILE or a `surrogate:` key")
    spath = Path(ref) if surrogate else path.parent / ref
    sur = Surrogate.load(spath)
    t0 = time.perf_counter()
    targets = {o: parse_quantity(v, output_dimension(o)) for o, v in job["targets"].items()}
    scales = {o: parse_quantity(v, output_dimension(o)) for o, v in (job.get("scales") or {}).items()}
    fixed = {n: parse_quantity(v, _in_dim(sur, n)) for n, v in (job.get("fixed") or {}).items()}
    bounds = {n: tuple(parse_quantity(x, _in_dim(sur, n)) for x in b) for n, b in job["actuators"].items()}
    unknown = [n for n in {**bounds, **fixed} if n not in sur.inputs]
    if unknown:
        raise ConfigError(f"{path}: the surrogate has no inputs {unknown}; it has {sur.inputs}")
    unlisted = [n for n in sur.inputs if n not in bounds and n not in fixed]
    if unlisted:
        raise ConfigError(f"{path}: surrogate inputs {unlisted} are neither in `actuators` nor in `fixed`")
    sol = solve_on_surrogate(sur, targets, scales, bounds, fixed, int(job.get("starts", 8)), float(job.get("tolerance", 0.05)))
    solve_time = time.perf_counter() - t0
    out = {"method": "surrogate", "surrogate": str(spath), "status": sol["status"], "values": {**fixed, **sol["values"]},
           "predicted": sol["predicted"], "targets": targets, "residual": sol["residual"],
           "at_bounds": sol["at_bounds"], "solve_time": solve_time,
           "surrogate_validation": sur.data.get("validation", {}).get("summary")}
    if job.get("verify"):
        if "config" not in job:
            raise ConfigError(f"{path}: `verify` needs the job's `config`")
        t1 = time.perf_counter()
        cfg = (path.parent / job["config"]).resolve()
        check_scenario(sur, cfg, job.get("overrides", []), job.get("backend", "mujoco"))
        sim = build_simulation(cfg, job.get("overrides", []), job.get("backend", "mujoco"),
                               job.get("settle_damping", DEFAULT_DAMPING))
        try:
            outs = list(targets)
            sc = {o: scales.get(o, _SCALES[output_dimension(o)]) for o in outs}
            eq = Equilibrium(sim, outs, sc, job.get("settle_tol", 1e-3),
                             parse_quantity(job.get("settle_window", 0.1), "time"),
                             parse_quantity(job.get("max_settle", 1.0), "time"), job.get("warm_start", True))
            got = eq(out["values"])
            out["verified"] = {"outputs": dict(zip(outs, map(float, got))),
                               "errors": {o: float(g - targets[o]) for o, g in zip(outs, got)},
                               "surrogate_error": {o: float(sol["predicted"][o] - g) for o, g in zip(outs, got)},
                               "residual": float(max(abs(g - targets[o]) / sc[o] for o, g in zip(outs, got))),
                               "unsettled": eq.unsettled, "wall_time": time.perf_counter() - t1}
        finally:
            sim.close()
    return out


def _in_dim(sur: Surrogate, name: str) -> str:
    for i in sur.data["inputs"]:
        if i["name"] == name:
            return {"N": "force", "Pa": "pressure", "m": "length"}.get(i["unit"], "force")
    return "force"


def check_scenario(sur: Surrogate, config: Path, overrides: Sequence[str], backend: str) -> None:
    """Refuse to verify against a scenario other than the one the surrogate was fitted on."""
    want = sur.data.get("scenario_hash")
    have = scenario_hash(config, overrides, backend)
    if want and want != have:
        raise ConfigError(f"the surrogate was fitted on another scenario/overrides/backend (hash {want}, this {have})")


# ------------------------------------------------------------------------------------------------------ the job
def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def scenario_hash(config: Path, overrides: Sequence[str], backend: str) -> str:
    """Hash of the resolved scenario (``extends`` included), the overrides and the backend."""
    return _hash({"resolved": load_yaml(config), "overrides": list(overrides), "backend": backend})


def design_points(kind: str, n: int, dim: int, seed: int) -> np.ndarray:
    """Seeded design in the unit cube: ``lhs`` (the box corners when ``dim <= 4`` plus a Latin hypercube) or ``grid``."""
    if kind == "grid":
        per = int(np.floor(n ** (1.0 / dim) + 1e-9))
        if per < 2:
            raise ConfigError(f"a grid of {n} points in {dim} dimensions has fewer than 2 per axis")
        axes = [np.linspace(0, 1, per)] * dim
        return np.array(list(itertools.product(*axes)))
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=dim))) if dim <= 4 else np.zeros((0, dim))
    if n < len(corners) + 1:
        raise ConfigError(f"samples must be at least {len(corners) + 1} for {dim} inputs")
    lhs = qmc.LatinHypercube(d=dim, seed=seed).random(n - len(corners))
    return np.vstack([corners, lhs])


class SurrogateJob:
    """Runs a ``softsim surrogate`` job file; see the module docstring and ``jobs/surrogate_cable_cantilever.yaml``."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ConfigError(f"job file not found: {self.path}")
        spec = yaml.safe_load(self.path.read_text()) or {}
        if not isinstance(spec, dict):
            raise ConfigError("job file must be a mapping")
        extra = set(spec) - _JOB_KEYS
        if extra:
            raise ConfigError(f"unknown job keys: {sorted(extra)}; known: {sorted(_JOB_KEYS)}")
        for k in ("config", "inputs", "outputs"):
            if k not in spec:
                raise ConfigError(f"job needs {k!r}")
        self.spec = spec
        self.config = (self.path.parent / spec["config"]).resolve()
        if not self.config.is_file():
            raise ConfigError(f"scenario file not found: {self.config}")
        self.overrides = list(spec.get("overrides", []))
        self.backend = spec.get("backend", "mujoco")
        if not isinstance(spec["inputs"], dict) or not spec["inputs"]:
            raise ConfigError("`inputs` must map input names to [lower, upper]")
        if not isinstance(spec["outputs"], list) or not spec["outputs"]:
            raise ConfigError("`outputs` must be a non-empty list of output columns")
        self.names = list(spec["inputs"])
        self.out_names = [str(o) for o in spec["outputs"]]
        try:
            self.n = int(spec.get("samples", 20))
            self.seed = int(spec.get("seed", 0))
            self.folds = int(spec.get("folds", 5))
            self.holdout = int(spec.get("holdout", 0))
            self.degree = spec.get("degree", "auto")
            if self.degree != "auto":
                self.degree = int(self.degree)
        except (TypeError, ValueError):
            raise ConfigError("samples, seed, folds, holdout and degree must be integers (degree may be 'auto')") from None
        self.kind = spec.get("model", "poly")
        self.design = spec.get("design", "lhs")
        if self.kind not in ("poly", "rbf"):
            raise ConfigError(f"model must be 'poly' or 'rbf', not {self.kind!r}")
        if self.design not in ("lhs", "grid"):
            raise ConfigError(f"design must be 'lhs' or 'grid', not {self.design!r}")
        if self.n < 3 or self.folds < 2 or self.holdout < 0:
            raise ConfigError("need samples >= 3, folds >= 2, holdout >= 0")
        if self.folds > self.n:
            raise ConfigError("folds cannot exceed the number of samples")
        dyn = spec.get("dynamics")
        if dyn is not None and (not isinstance(dyn, dict) or set(dyn) - _DYN_KEYS):
            raise ConfigError(f"`dynamics` must be a mapping with keys from {sorted(_DYN_KEYS)}")
        self.output = (self.path.parent / spec.get("output", "surrogate_out")).resolve()
        self.sim = None

    # -- helpers
    def _settle(self) -> dict[str, Any]:
        s = self.spec
        return {"damping": s.get("settle_damping", DEFAULT_DAMPING), "tol": s.get("settle_tol", 1e-3),
                "window": s.get("settle_window", 0.1), "max": s.get("max_settle", 1.0), "warm": s.get("warm_start", True)}

    def _open(self):
        sim = build_simulation(self.config, self.overrides, self.backend, self.spec.get("settle_damping", DEFAULT_DAMPING))
        metrics = sim.metrics()
        bad = [o for o in self.out_names if o not in metrics]
        if bad:
            sim.close()
            raise ConfigError(f"unknown outputs {bad}; available: {sorted(metrics)}")
        return sim

    def _bounds(self, sim) -> tuple[np.ndarray, np.ndarray, list[str]]:
        lo, hi, units = [], [], []
        for n in self.names:
            dim = actuator_dimension(sim, n)
            b = self.spec["inputs"][n]
            if not isinstance(b, (list, tuple)) or len(b) != 2:
                raise ConfigError(f"input {n!r}: bounds must be [lower, upper]")
            a, c = (parse_quantity(x, dim) for x in b)
            if not (np.isfinite(a) and np.isfinite(c) and c > a):
                raise ConfigError(f"input {n!r}: bounds must be finite with upper > lower, got {b!r}")
            lo.append(a)
            hi.append(c)
            units.append({"force": "N", "length": "m", "pressure": "Pa"}[dim])
        return np.array(lo), np.array(hi), units

    @staticmethod
    def _skey(x: np.ndarray) -> str:
        return json.dumps([float(f"{v:.10g}") for v in x])

    def _load(self, log: Path) -> dict[str, list[float]]:
        done = {}
        if log.is_file():
            for line in log.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("key") == self.key:
                    done[self._skey(np.array(e["inputs"]))] = e["outputs"]
        return done

    # -- run
    def run(self) -> dict[str, Any]:
        from softsim.bench import code_hash

        t0 = time.perf_counter()
        sim = self._open()
        try:
            lo, hi, units = self._bounds(sim)
            settle = self._settle()
            self.key = _hash({"scenario": scenario_hash(self.config, self.overrides, self.backend), "code": code_hash(),
                              "settle": settle, "inputs": self.names, "outputs": self.out_names,
                              "scales": self.spec.get("scales")})
            self.output.mkdir(parents=True, exist_ok=True)
            log = self.output / "samples.jsonl"
            done = self._load(log)
            unit = np.vstack([design_points(self.design, self.n, len(self.names), self.seed),
                              *( [qmc.LatinHypercube(d=len(self.names), seed=self.seed + 1).random(self.holdout)]
                                 if self.holdout else [])])
            n_fit = len(unit) - self.holdout
            X = lo + unit * (hi - lo)
            dims = [output_dimension(o) for o in self.out_names]
            sc = {o: parse_quantity((self.spec.get("scales") or {}).get(o, _SCALES[d]), d)
                  for o, d in zip(self.out_names, dims)}
            eq = Equilibrium(sim, self.out_names, sc, settle["tol"], parse_quantity(settle["window"], "time"),
                             parse_quantity(settle["max"], "time"), settle["warm"],
                             fixed=None)
            Y = np.empty((len(X), len(self.out_names)))
            simulated = 0
            with log.open("a") as fh:
                for i, x in enumerate(X):
                    k = self._skey(x)
                    if k in done:
                        Y[i] = done[k]
                        continue
                    y = eq(dict(zip(self.names, map(float, x))))
                    Y[i] = y
                    simulated += 1
                    fh.write(json.dumps({"key": self.key, "inputs": x.tolist(), "outputs": y.tolist(),
                                         "unsettled": eq.unsettled}) + "\n")
                    fh.flush()
            sim_time = time.perf_counter() - t0
            data = self._fit(X, Y, n_fit, lo, hi, units, dims, sc)
            data["samples"] = {"n": len(X), "simulated_now": simulated, "unsettled": eq.unsettled, "design": self.design,
                               "seed": self.seed, "simulation_time": sim_time}
            if self.spec.get("dynamics"):
                data["dynamics"] = self._dynamics(sim, eq, lo, hi)
            data["wall_time"] = time.perf_counter() - t0
        finally:
            sim.close()
        save_surrogate(data, self.output / "surrogate.json")
        return {"surrogate": str(self.output / "surrogate.json"), "validation": data["validation"]["summary"],
                "cv": {o: v["rmse"] for o, v in data["validation"]["cv"].items() if isinstance(v, dict)},
                **{k: data[k] for k in ("samples", "wall_time", "model_summary")},
                **({"dynamics": data["dynamics"]["validation"]} if "dynamics" in data else {})}

    def _fit(self, X, Y, n_fit, lo, hi, units, dims, sc) -> dict[str, Any]:
        Z = 2 * (X - lo) / (hi - lo) - 1
        scale = np.array([sc[o] for o in self.out_names])
        model, degree, validation = fit_model(Z, Y, n_fit, self.kind, self.degree, self.folds, self.seed, scale,
                                              self.out_names)
        from softsim.bench import code_hash

        return {"format": FORMAT, "kind": "equilibrium_map",
                "inputs": [{"name": n, "unit": u, "lower": float(a), "upper": float(b)}
                           for n, u, a, b in zip(self.names, units, lo, hi)],
                "outputs": [{"name": o, "unit": _UNIT[d], "range": [float(Y[:, j].min()), float(Y[:, j].max())]}
                            for j, (o, d) in enumerate(zip(self.out_names, dims))],
                "model": model, "model_summary": {"type": self.kind, "degree": degree if self.kind == "poly" else None,
                                                  "terms": len(model.get("exponents", model.get("centers")))},
                "validation": validation, "scenario": str(self.config), "backend": self.backend, "overrides": self.overrides,
                "scenario_hash": scenario_hash(self.config, self.overrides, self.backend), "code_hash": code_hash(),
                "key": self.key, "normalisation": "z = 2 (x - lower) / (upper - lower) - 1"}

    def _dynamics(self, sim, eq, lo, hi) -> dict[str, Any]:
        """Linear dynamic surrogate from constant-input step responses; see :func:`fit_dmd`."""
        d = self.spec["dynamics"]
        try:
            n_train, n_test = int(d.get("steps", 4)), int(d.get("test", 2))
            duration = parse_quantity(d.get("duration", 0.3), "time")
            every, delay = int(d.get("sample_every", 1)), int(d.get("delay", 1))
        except (TypeError, ValueError):
            raise ConfigError("dynamics: steps, test, sample_every and delay must be integers, duration a time") from None
        if n_train < 1 or n_test < 1 or every < 1 or delay < 0:
            raise ConfigError("dynamics: need steps >= 1, test >= 1, sample_every >= 1, delay >= 0")
        dt = sim.config.simulation.output_every * sim.config.simulation.timestep * every
        n_steps = int(round(duration / sim.config.simulation.output_every / sim.config.simulation.timestep))
        U = lo + qmc.LatinHypercube(d=len(self.names), seed=self.seed + 2).random(n_train + n_test) * (hi - lo)
        trajectories = []
        for u in U:
            sim.reset()
            sim.apply_controls({**eq.fixed, **dict(zip(self.names, map(float, u)))})
            xs = []
            for k in range(n_steps):
                sim.step()
                if (k + 1) % every == 0:
                    m = sim.metrics()
                    xs.append([m[o] for o in self.out_names])
            trajectories.append(np.array(xs))
        sim.reset()
        sim.apply_controls(dict(eq.fixed))
        x0 = np.array([sim.metrics()[o] for o in self.out_names])
        u0 = np.array([eq.fixed.get(n, 0.0) for n in self.names])     # the inputs at rest; the model takes u - u0
        dU = U - u0
        model = fit_dmd(trajectories[:n_train], dU[:n_train], x0, delay)
        val = {"train": dmd_rollout_error(model, trajectories[:n_train], dU[:n_train], self.out_names),
               "holdout": dmd_rollout_error(model, trajectories[n_train:], dU[n_train:], self.out_names)}
        return {"type": "dmd_control", "form": "s_{k+1} = A s_k + B (u_k - u0); s_k = [x_k - x0, ..., x_{k-delay} - x0]; "
                "x = first block of s", "dt": dt, "delay": delay, "x0": x0.tolist(), "u0": u0.tolist(), "A": model["A"].tolist(),
                "B": model["B"].tolist(), "input_names": self.names, "output_names": self.out_names,
                "trajectories": {"train": n_train, "test": n_test, "duration": duration}, "validation": val}


def fit_dmd(trajectories: Sequence[np.ndarray], inputs: np.ndarray, x0: np.ndarray, delay: int = 1) -> dict[str, np.ndarray]:
    """DMD with control from step responses: least-squares ``s_{k+1} = A s_k + B u_k`` (``s`` = outputs minus the
    rest value ``x0``, with ``delay`` earlier samples stacked, which gives the model its velocity).

    ``trajectories[i]`` is the (samples, outputs) response from rest to the constant input ``inputs[i]``; every
    trajectory starts at rest, so ``s_0 = 0``. A linear model reproduces the response to a step of any size up to the
    nonlinearity of the body (small for a cable on a cantilever, larger near locking or contact).
    """
    p = trajectories[0].shape[1]
    rows_s, rows_n, rows_u = [], [], []
    for traj, u in zip(trajectories, inputs):
        s = _stack(np.vstack([np.zeros((1, p)), traj - x0]), delay)
        rows_s.append(s[:-1])
        rows_n.append(s[1:])
        rows_u.append(np.tile(u, (len(s) - 1, 1)))
    S, N, U = np.vstack(rows_s), np.vstack(rows_n), np.vstack(rows_u)
    # outputs differ in scale: fit in scaled coordinates so one output does not dominate
    scale = np.maximum(np.abs(np.vstack(trajectories) - x0).max(axis=0), 1e-12)
    w = np.tile(1.0 / scale, delay + 1)
    g = np.linalg.lstsq(np.hstack([S * w, U]), N * w, rcond=None)[0].T
    A, B = g[:, :S.shape[1]], g[:, S.shape[1]:]
    return {"A": A * w[None, :] / w[:, None], "B": B / w[:, None], "x0": x0, "delay": delay}


def _stack(x: np.ndarray, delay: int) -> np.ndarray:
    """Rows ``[x_k, x_{k-1}, ..., x_{k-delay}]`` (earlier than the start taken as the first row)."""
    rows = []
    for k in range(len(x)):
        rows.append(np.concatenate([x[max(k - j, 0)] for j in range(delay + 1)]))
    return np.array(rows)


def dmd_rollout(model: Mapping[str, Any], u: np.ndarray, steps: int) -> np.ndarray:
    """Predicted outputs (steps, p) for a constant input ``u`` from rest."""
    A, B = np.asarray(model["A"]), np.asarray(model["B"])
    p = A.shape[0] // (int(model["delay"]) + 1)
    s, out = np.zeros(A.shape[0]), []
    for _ in range(steps):
        s = A @ s + B @ np.asarray(u)
        out.append(s[:p] + np.asarray(model["x0"]))
    return np.array(out)


def dmd_rollout_error(model: Mapping[str, Any], trajectories: Sequence[np.ndarray], inputs: np.ndarray,
                      names: Sequence[str]) -> dict[str, dict[str, float]]:
    """Open-loop rollout error over whole trajectories (RMS and max, SI; relative to the output's travel)."""
    pred = np.vstack([dmd_rollout(model, u, len(t)) for t, u in zip(trajectories, inputs)])
    truth = np.vstack(trajectories)
    travel = np.ptp(truth, axis=0)
    err = pred - truth
    return {n: {"rmse": float(np.sqrt(np.mean(err[:, j] ** 2))), "max": float(np.abs(err[:, j]).max()),
                "rmse_rel_travel": float(np.sqrt(np.mean(err[:, j] ** 2)) / (travel[j] or 1.0))}
            for j, n in enumerate(names)}


def run_surrogate_job(path: str | Path) -> dict[str, Any]:
    return SurrogateJob(path).run()

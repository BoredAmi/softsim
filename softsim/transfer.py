"""Transfer a physical material card to a fast backend: fit its effective values against the reference.

    softsim transfer calibrations/transfer_inflation_mujoco.yaml

A transfer file names a scenario config, the backend and mesh to transfer to,
the parameters to fit and the output columns to match::

    config: ../scenarios/inflation/hollow_sphere.yaml
    backend: mujoco
    overrides: [body.mesh.size=3mm]            # applied to both runs
    backend_overrides: []                      # only to the backend's runs (e.g. its timestep)
    reference: {backend: fem, overrides: [simulation.timestep=2ms]}
    observables: {cavity_volume: 1.0}          # column: weight
    parameters:
      body.material.elastic.youngs_modulus: {initial: 30 kPa, bounds: [10 kPa, 100 kPa], log: true}
    checks:                                    # load cases not fitted: does the transfer hold?
      half_pressure: ["actuators[0].schedule=[[0, 0], [0.15, 1500], [0.25, 1500]]"]
    max_evaluations: 15
    output: ../runs/transfer/inflation_mujoco_3mm

The reference runs once per load case and is cached in the output directory,
keyed by its configuration. Each observable is compared as its change from the
first sample, interpolated onto the reference's times: error = RMS difference /
largest reference change (NRMSE); the cost is the weighted mean. Evaluations are
logged and reused, so an interrupted transfer resumes. The report gives the
fitted values, the raw card's error and the transferred card's error on every
check, and writes the effective card (``kind: effective``) with its provenance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from softsim.config.loader import apply_overrides, get_dotted, load_yaml, set_dotted
from softsim.errors import ConfigError
from softsim.experiments.calibrate import Parameter, _hash, _parse_parameters, _relative


def _nrmse(ref: pd.DataFrame, sim: pd.DataFrame, column: str) -> float:
    d_ref = ref[column].to_numpy() - ref[column].iloc[0]
    d_sim = np.interp(ref["t"].to_numpy(), sim["t"].to_numpy(), sim[column].to_numpy() - sim[column].iloc[0])
    scale = float(np.abs(d_ref).max())
    if scale == 0:
        raise ConfigError(f"observable {column!r} does not change in the reference run")
    return float(np.sqrt(np.mean((d_sim - d_ref) ** 2)) / scale)


class Transfer:
    """One transfer file; see module docstring."""

    card_kind = "effective"        # what the fitted cards are (a subclass fitting the reference to data: physical)
    card_suffix = "transfer"
    card_verb = "Transfer"
    fitted_by = "softsim transfer"
    card_note = ("Effective parameters: valid only with this backend, element type and mesh size, and fitted "
                 "jointly with the other cards of this transfer.")

    @property
    def ref_label(self) -> str:
        return f"the {self.ref_backend} reference"

    def report_extra(self) -> dict[str, Any]:
        """Entries a subclass adds to the report."""
        return {}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        spec = load_yaml(self.path)
        base = self.path.parent
        self.config = (base / spec["config"]).resolve()
        self.backend = spec["backend"]
        self.overrides = list(spec.get("overrides", []))
        self.backend_overrides = list(spec.get("backend_overrides", []))
        self._read_reference(spec)
        self.params: list[Parameter] = _parse_parameters(spec["parameters"])
        self.checks = {name: list(o) for name, o in (spec.get("checks") or {}).items()}
        self.max_evaluations = int(spec.get("max_evaluations", 20))
        self.output = (base / spec["output"]).resolve()
        self._log = self.output / "evaluations.jsonl"
        self.history: list[dict[str, Any]] = []
        if self._log.exists():
            self.history = [json.loads(line) for line in self._log.read_text().splitlines() if line.strip()]

    def _read_reference(self, spec: dict[str, Any]) -> None:
        """What the backend is compared with: here another backend's run of the same scenario."""
        ref = spec.get("reference", {})
        self.ref_backend = ref.get("backend", "fem")
        self.ref_overrides = list(ref.get("overrides", []))
        self.observables = {k: float(v) for k, v in spec["observables"].items()}

    # -- runs ------------------------------------------------------------------------------------

    def _raw(self, case: list[str]) -> dict[str, Any]:
        return apply_overrides(load_yaml(self.config), [*self.overrides, *case])

    def reference(self, case: list[str]) -> pd.DataFrame:
        """Reference timeseries for a load case (cached by configuration)."""
        from softsim.runner import run_config

        raw = apply_overrides(self._raw(case), [f"backend={self.ref_backend}", *self.ref_overrides])
        key = _hash(raw)
        cached = self.output / "reference" / f"{key}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)
        cached.parent.mkdir(parents=True, exist_ok=True)
        ts = run_config(self._with_base(raw)).timeseries
        ts.to_parquet(cached)
        return ts

    def _with_base(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw)
        raw.setdefault("material_paths", [])
        raw["material_paths"] = [str((self.config.parent / p).resolve()) for p in raw["material_paths"]]
        return raw

    def _simulate(self, case: list[str], values: dict[str, float]) -> pd.DataFrame:
        from softsim.runner import run_config

        raw = apply_overrides(self._raw(case), [f"backend={self.backend}", *self.backend_overrides])
        for k, v in values.items():
            raw = set_dotted(raw, k, v)
        return run_config(self._with_base(raw)).timeseries

    def errors(self, case: list[str], values: dict[str, float]) -> dict[str, float]:
        ref, sim = self.reference(case), self._simulate(case, values)
        out = {c: _nrmse(ref, sim, c) for c in self.observables}
        out["cost"] = sum(w * out[c] for c, w in self.observables.items()) / sum(self.observables.values())
        return out

    # -- fit -------------------------------------------------------------------------------------

    def evaluate(self, values: dict[str, float]) -> float:
        for r in self.history:
            if all(np.isclose(r[k], v, rtol=1e-9) for k, v in values.items()):
                return r["cost"]
        t0 = time.perf_counter()
        try:
            row = {**values, **self.errors([], values)}
        except Exception as exc:  # divergence etc.: a bad point, not a failed transfer
            row = {**values, "cost": 1e3, "error": str(exc)[:200]}
        row["wall_s"] = time.perf_counter() - t0
        self.history.append(row)
        self.output.mkdir(parents=True, exist_ok=True)
        with self._log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  eval {len(self.history) - 1:3d}: cost {row['cost']:.4f}  "
              + "  ".join(f"{k.split('.')[-1]}={v:.4g}" for k, v in values.items()), flush=True)
        return row["cost"]

    def run(self) -> dict[str, Any]:
        from scipy.optimize import minimize, minimize_scalar

        start = time.perf_counter()
        initial = {p.key: p.initial for p in self.params}
        self.evaluate(initial)
        remaining = max(self.max_evaluations - len(self.history), 1)
        if len(self.params) == 1:
            p = self.params[0]
            minimize_scalar(lambda u: self.evaluate({p.key: p.from_unit(u)}), method="bounded",
                            bounds=(p.to_unit(p.lower), p.to_unit(p.upper)),
                            options={"maxiter": remaining, "xatol": 0.003})
        else:
            x0 = np.array([p.to_unit(p.initial) for p in self.params])
            minimize(lambda u: self.evaluate({p.key: p.from_unit(x) for p, x in zip(self.params, u)}), x0,
                     method="Nelder-Mead",
                     options={"maxfev": remaining, "initial_simplex": np.vstack([x0] + [x0 + 0.3 * e for e in
                                                                                       np.eye(len(x0))]),
                              "xatol": 0.003, "fatol": 1e-4})
        fitted = [r for r in self.history if all(p.key in r for p in self.params)]
        best = min(fitted, key=lambda r: r["cost"])
        values = {p.key: float(best[p.key]) for p in self.params}
        return self._report(initial, values, best, time.perf_counter() - start)

    def _report(self, initial: dict[str, float], values: dict[str, float], best: dict[str, Any],
                wall: float) -> dict[str, Any]:
        from softsim.config.schema import validate
        from softsim.materials import load_material, save_material
        from softsim.scenarios.registry import get_scenario

        checks = {}
        for name, case in self.checks.items():
            checks[name] = {"raw_card": self.errors(case, initial)["cost"],
                            "transferred": self.errors(case, values)["cost"]}
        raw0 = self._raw([])
        report = {"backend": self.backend, "best": values, "cost": best["cost"],
                  "raw_card_cost": next(r["cost"] for r in self.history
                                        if all(np.isclose(r.get(k, np.nan), v) for k, v in initial.items())),
                  "per_observable": {c: best[c] for c in self.observables if c in best},
                  "checks": checks, "evaluations": len(self.history), "wall_time_s": wall, **self.report_extra()}
        # the effective cards: the body's material, and every other fitted material (a zone's), with the
        # fitted values
        cls = get_scenario(raw0["scenario"])
        scenario = cls(validate(cls.config_model, raw0), self.config.parent)
        key = scenario.material_key(scenario.build_scene().soft_bodies[0].name)
        raw = raw0
        for k, v in values.items():
            raw = set_dotted(raw, k, v)
        mesh = get_dotted(raw0, key.rsplit(".", 1)[0]).get("mesh", {})
        keys = [key] + sorted({k[:k.index(".material.") + len(".material")] for k in values
                               if ".material." in k} - {key})
        self.output.mkdir(parents=True, exist_ok=True)
        files = []
        for k in keys:
            material = scenario.load_material(get_dotted(raw, k))
            owner = "" if k == key else f"_{get_dotted(raw, k.rsplit('.', 1)[0]).get('name', k.split('.')[-2])}"
            card = material.model_copy(update={
                "kind": self.card_kind,
                "name": f"{material.name}{owner}_{self.backend}_{self.card_suffix}",
                "source": f"{self.card_verb} of {material.name} ({k}) to {self.backend} against {self.ref_label} "
                          f"({_relative(self.path)}).",
                "provenance": {"fitted_by": self.fitted_by, "transfer_file": _relative(self.path),
                               "config_key": k, "backend": self.backend,
                               "mesh_size": mesh.get("size", "scenario default"),
                               "element": "quadratic tetrahedra" if self.backend == "fem" else "linear tetrahedra",
                               "reference": self.ref_backend,
                               "observables": self.observables, "cost": best["cost"], "checks": checks,
                               "note": self.card_note},
            })
            files.append(str(save_material(card, self.output / f"{card.name}.yaml")))
        report["material_file"] = files[0]
        if len(files) > 1:
            report["material_files"] = files
        (self.output / "report.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(report, default=float)),
                                                                sort_keys=False))
        return report

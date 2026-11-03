"""Fit a scenario's parameters to measured data: any observable of any scenario against a table.

    softsim fit calibrations/pneunet_fit.yaml

The scenario-neutral counterpart of ``softsim calibrate`` (which tracks drop-test videos) and
``softsim transfer`` (which compares a fast backend with the reference): the thing compared with is a
measured table, registered with its source. A fit file is a transfer file (``config``, ``backend``,
``overrides``, ``parameters``, ``checks``, ``max_evaluations``, ``output``) whose ``reference`` is replaced by
``data``::

    data:
      source: {name: ..., doi: 10.17028/rd.lboro.5509528, licence: CC BY-NC 4.0, file: ../data/x.csv}
      kind: curve                         # curve: y against a simulated x (a quasi-static ramp), or series: y(t)
      filter: {tilt: 0}                   # keep the rows where these columns have these values
      x: {column: p_kPa, scale: 1000, sim: cavity_pressure}     # measured column (x scale) = simulated column
      y: {column: theta_deg, scale: 0.0174533, sim: bend_angle}  # for series: t: {column: t_s, scale: 1}
      relative_to: first                  # compare changes from the first sample (zero offsets), or none

``scale`` converts the measured column to the simulation's SI unit. A curve is compared with the rising part
of the simulated ramp (the simulated y at the measured x); a series at the measured times. The error is the
RMS difference over the largest measured change (NRMSE), as in a transfer; a measurement noisier than that
sets the floor the fit can reach, which the report states (the RMS scatter of the measured values about
their own mean at repeated x).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from softsim.errors import ConfigError
from softsim.transfer import Transfer


class Fit(Transfer):
    """One fit file; see module docstring."""

    card_suffix = "fit"
    card_verb = "Fit"
    fitted_by = "softsim fit"

    @property
    def card_kind(self) -> str:  # the locking-free reference describes the material; a linear-tet backend only itself
        return "physical" if self.backend == "fem" else "effective"

    @property
    def card_note(self) -> str:
        if self.card_kind == "physical":
            return ("Fitted to measured data with the locking-free reference; valid for the geometry and loading "
                    "range of the data (see the source).")
        return "Effective parameters fitted to measured data: valid only with this backend, element type and mesh size."

    @property
    def ref_label(self) -> str:
        return f"measured data ({self.source.get('name', 'unnamed')}, DOI {self.source.get('doi', 'none')})"

    def _read_reference(self, spec: dict[str, Any]) -> None:
        data = spec.get("data")
        if not data:
            raise ConfigError("a fit file needs a `data:` section (see softsim.experiments.fit)")
        self.source = dict(data.get("source", {}))
        for key in ("name", "doi", "licence", "file"):
            if key not in self.source:
                raise ConfigError(f"fit data.source needs {key!r}: a measurement is registered with where it came from")
        self.kind = data.get("kind", "curve")
        if self.kind not in ("curve", "series"):
            raise ConfigError(f"fit data.kind must be curve or series, got {self.kind!r}")
        self.relative_to = data.get("relative_to", "first")
        if self.relative_to not in ("first", "none"):
            raise ConfigError(f"fit data.relative_to must be first or none, got {self.relative_to!r}")
        need = ("x", "y") if self.kind == "curve" else ("t", "y")
        for key in need:
            if key not in data or "column" not in data[key] or (key != "t" and "sim" not in data[key]):
                raise ConfigError(f"fit data.{key} needs `column` and, except for t, `sim`")
        self.axes = {k: {"column": data[k]["column"], "scale": float(data[k].get("scale", 1.0)),
                         "sim": data[k].get("sim")} for k in need}
        table = pd.read_csv((self.path.parent / self.source["file"]).resolve())
        for column, value in (data.get("filter") or {}).items():
            table = table[np.isclose(table[column].to_numpy(float), float(value))]
        columns = [self.axes[k]["column"] for k in need]
        missing = [c for c in columns if c not in table]
        if missing:
            raise ConfigError(f"fit data file has no column {missing} (has {list(table.columns)})")
        table = table[columns].dropna()
        if len(table) < 3:
            raise ConfigError("fit data: fewer than 3 rows after filtering")
        self.table = pd.DataFrame({k: table[self.axes[k]["column"]].to_numpy(float) * self.axes[k]["scale"]
                                   for k in need}).sort_values(need[0]).reset_index(drop=True)
        self.ref_backend = "measured"
        self.ref_overrides = []
        self.observables = {self.axes["y"]["sim"]: 1.0}

    # -- comparison ------------------------------------------------------------------------------

    def simulated(self, sim: pd.DataFrame) -> np.ndarray:
        """The simulation's y at the measured x (curve) or t (series)."""
        y_name = self.axes["y"]["sim"]
        if self.kind == "series":
            return np.interp(self.table["t"].to_numpy(), sim["t"].to_numpy(), sim[y_name].to_numpy())
        x, y = sim[self.axes["x"]["sim"]].to_numpy(), sim[y_name].to_numpy()
        rise = slice(0, int(np.argmax(x)) + 1)  # the loading branch: the part of the run that goes up to the peak
        x, y = x[rise], y[rise]
        order = np.argsort(x, kind="stable")
        if len(np.unique(x)) < 3:
            raise ConfigError("fit: the simulated x column does not vary (a curve needs a ramp)")
        return np.interp(self.table["x"].to_numpy(), x[order], y[order])

    def noise_floor(self) -> float:
        """RMS scatter of the measured y at repeated x, over the largest measured change (0 without repeats)."""
        key = "x" if self.kind == "curve" else "t"
        groups = self.table.groupby(self.table[key].round(9))["y"]
        resid = (self.table["y"] - groups.transform("mean"))[groups.transform("count") > 1]
        scale = float(np.abs(self.table["y"] - self.table["y"].iloc[0]).max())
        return float(np.sqrt(np.mean(resid**2)) / scale) if len(resid) and scale else 0.0

    def errors(self, case: list[str], values: dict[str, float]) -> dict[str, float]:
        sim = self._simulate(case, values)
        y_sim, y = self.simulated(sim), self.table["y"].to_numpy()
        if self.relative_to == "first":
            first = float(y[0]) if self.kind == "series" else float(y[self.table["x"].to_numpy() == self.table["x"].min()].mean())
            y_sim0 = float(y_sim[0]) if self.kind == "series" else float(
                y_sim[self.table["x"].to_numpy() == self.table["x"].min()].mean())
            y, y_sim = y - first, y_sim - y_sim0
        scale = float(np.abs(y).max())
        if scale == 0:
            raise ConfigError("fit: the measured y does not change")
        name = self.axes["y"]["sim"]
        out = {name: float(np.sqrt(np.mean((y_sim - y) ** 2)) / scale)}
        out["cost"] = out[name]
        return out

    def report_extra(self) -> dict[str, Any]:
        return {"source": self.source, "rows": int(len(self.table)), "noise_floor": self.noise_floor()}

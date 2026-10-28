"""Compare drop-test simulations with tracked experiments.

Both sides are reduced to the same observable with the same rules:
    * contact = first time the weight's centre of mass departs from free fall
      by more than ``contact_threshold`` (0.1 mm ~ 0.5 px in the recordings);
    * ``indentation_rel`` = downward displacement of the centre of mass since
      contact, relative to the specimen base (removes support motion);
    * the comparison window is the first contact phase: from contact until the
      experimental curve returns below zero after its peak (capped).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

G = 9.81


@dataclass(frozen=True)
class ExperimentCase:
    """A tracked recording: time series (t since contact) and tracking metadata."""

    name: str
    data: pd.DataFrame
    meta: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path, case: str) -> "ExperimentCase":
        directory = Path(directory)
        path = directory / f"{case}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `softsim track` first")
        meta = yaml.safe_load((directory / f"{case}.yaml").read_text())
        return cls(case, pd.read_parquet(path), meta)

    def window(self, max_duration: float = 0.02) -> tuple[float, float]:
        """First contact phase [s]: contact until the curve is back below zero after the peak."""
        d = self.data[(self.data.t >= 0) & (self.data.t <= max_duration)].dropna(subset=["indentation_rel"])
        if d.empty:
            return 0.0, 0.0
        k = d.indentation_rel.idxmax()
        back = d[(d.t > d.t[k]) & (d.indentation_rel <= 0)]
        return 0.0, float(back.t.iloc[0]) if len(back) else float(d.t.iloc[-1])

    def curve(self, column: str = "indentation_rel") -> tuple[np.ndarray, np.ndarray]:
        d = self.data.dropna(subset=[column])
        return d.t.to_numpy(), d[column].to_numpy()

    def sim_overrides(self, gap: float = 0.002) -> list[str]:
        """Scenario overrides reproducing the measured impact (speed, horizontal offset,
        in-plane rotation rate). Out-of-plane tilt is not observable and left at zero."""
        v = self.meta["impact_speed"]
        v0 = math.sqrt(max(v**2 - 2 * G * gap, 0.0))
        omega_y = math.radians(self.meta.get("rotation_rate_at_contact_deg_s", 0.0))  # clockwise on screen = +y
        return [
            f"drop.gap={gap}",
            f"drop.initial_speed={v0}",
            f"drop.offset=[{self.meta['offset_at_contact_mm'] * 1e-3}, 0.0]",
            f"drop.angular_velocity=[0.0, {omega_y}, 0.0]",
        ]


def simulated_indentation(ts: pd.DataFrame, gravity: float = G, contact_threshold: float = 1e-4) -> pd.DataFrame:
    """Apply the experiment's contact rule to a drop-test simulation time series.

    Returns columns ``t`` (since contact) and ``indentation_rel`` [m].
    """
    t = ts["t"].to_numpy()
    z = ts["weight_com_z"].to_numpy()
    base = ts["specimen_min_z"].to_numpy()
    # free fall fitted (known g) on the samples before any contact force, skipping the
    # first sample (the state right after reset lags one step)
    touching = np.nonzero(ts["contact_force_z"].to_numpy() != 0)[0]
    stop = touching[0] if len(touching) else len(t)
    flight = np.arange(1, stop)
    if len(flight) < 2:
        raise ValueError("simulation needs at least two free-flight samples (increase drop.gap or output rate)")
    a = np.c_[np.ones(len(flight)), t[flight]]
    c0, v0 = np.linalg.lstsq(a, z[flight] + 0.5 * gravity * t[flight] ** 2, rcond=None)[0]
    free = c0 + v0 * t - 0.5 * gravity * t**2
    departed = np.nonzero(z - free > contact_threshold)[0]
    if len(departed) == 0:
        return pd.DataFrame({"t": t - t[-1], "indentation_rel": np.full_like(t, np.nan)})
    c = max(departed[0] - 1, 0)
    return pd.DataFrame({"t": t - t[c], "indentation_rel": (z[c] - z) + (base - base[c])})


def compare_case(sim: pd.DataFrame, exp: ExperimentCase, window: tuple[float, float] | None = None) -> dict[str, float]:
    """Error metrics between a simulated and an experimental indentation curve."""
    t0, t1 = window or exp.window()
    te, ye = exp.curve()
    sel = (te >= t0) & (te <= t1)
    te, ye = te[sel], ye[sel]
    ts, ys = sim["t"].to_numpy(), sim["indentation_rel"].to_numpy()
    if len(te) < 3 or not np.isfinite(ys).any() or ts[-1] < t1:
        return {"rmse": float("nan"), "nrmse": float("nan")}
    yi = np.interp(te, ts, ys)
    peak_e, peak_s = float(ye.max()), float(ys[(ts >= t0) & (ts <= t1)].max())
    rmse = float(np.sqrt(np.mean((yi - ye) ** 2)))
    sim_win = (ts >= t0) & (ts <= t1)
    return {
        "rmse": rmse,
        "nrmse": rmse / peak_e if peak_e > 0 else float("nan"),
        "peak_exp": peak_e,
        "peak_sim": peak_s,
        "t_peak_exp": float(te[np.argmax(ye)]),
        "t_peak_sim": float(ts[sim_win][np.argmax(ys[sim_win])]),
        "window_end": t1,
    }

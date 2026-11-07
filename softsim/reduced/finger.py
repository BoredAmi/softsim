"""A pneumatic finger reduced to a planar chain of links, its tables taken from a reference run.

Each of ``links`` links has a hinge and a slide. At pressure p the finger at rest (no load) has joint angles
``angles(p)`` [rad] and link lengths ``lengths(p)`` [m], read from the reference's free-bend ramp along the pad
layer: tables, not laws, because the pad stretches by a third at 40 kPa and the bend falls back after 35 kPa, which
neither a constant link length nor one torque law can follow. Under load each hinge pulls back with ``stiffness(p)``
[N m/rad], from static tip-load probes on the reference (tools/probe_finger.py, :func:`fit_stiffness`).

The tables are the whole model: :meth:`ReducedFinger.to_json` is read both by the Python model
(:mod:`softsim.reduced.mjcf`) and by the browser page, so the two cannot drift apart. The chain does not deform
locally around an object and has no chambers: it reproduces the finger's centreline and its compliance, not its walls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field

from softsim.config.schema import StrictModel
from softsim.config.units import Length


class ReducedFingerConfig(StrictModel):
    """How to cut the finger: ``links`` equal links along the layer ``layer`` (z range in the finger's frame: the pad
    above the teeth), stations averaged over ``station_width`` of rest x, tables every ``pressure_step`` [Pa]."""

    links: Annotated[int, Field(ge=2)] = 8
    layer: tuple[Length, Length] = (0.0009, 0.0031)
    station_width: Annotated[Length, Field(gt=0)] = 0.0032
    pressure_step: Annotated[float, Field(gt=0)] = 2500.0


@dataclass(frozen=True)
class ReducedFinger:
    """Tables over ``pressures`` (P,): ``angles`` (P, N) [rad], each link's direction change from the one before (the
    first from the finger's axis; negative curls towards the pads), ``lengths`` (P, N) [m], ``stiffness`` (P,)
    [N m/rad] per hinge. ``offset`` [m]: the layer's height above the pad's face."""

    pressures: np.ndarray
    angles: np.ndarray
    lengths: np.ndarray
    stiffness: np.ndarray
    offset: float = 0.002
    width: float = 0.024
    thickness: float = 0.010
    mass: float = 0.021

    @property
    def links(self) -> int:
        return self.angles.shape[1]

    def _at(self, table: np.ndarray, p: float) -> np.ndarray:
        return np.array([np.interp(p, self.pressures, c) for c in np.atleast_2d(table.T)])

    def at(self, p: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(angles, lengths, stiffness) at pressure ``p`` [Pa], interpolated, clamped to the table."""
        return self._at(self.angles, p), self._at(self.lengths, p), float(np.interp(p, self.pressures, self.stiffness))

    def centreline(self, p: float) -> np.ndarray:
        """Unloaded layer points (N + 1, 2) in the finger's (x, z) frame, base at the origin."""
        a, l, _ = self.at(p)
        heading = np.cumsum(a)
        return np.vstack([[0.0, 0.0], np.cumsum(l[:, None] * np.column_stack([np.cos(heading), np.sin(heading)]), axis=0)])

    def loaded(self, p: float, loads: np.ndarray, k: float | None = None) -> np.ndarray:
        """Layer points (N + 1, 2) in static balance at pressure ``p`` [Pa] under forces ``loads`` (N + 1,) [N] along
        +z at the stations, the finger's axis horizontal: each hinge's turn from its rest angle is the moment of the
        forces beyond it over the stiffness (``k`` [N m/rad], or the table's). Damped fixed point on the joint angles."""
        a, l, table = self.at(p)
        k = table if k is None else k
        q = a.copy()
        for _ in range(20000):
            heading = np.cumsum(q)
            x = np.concatenate([[0.0], np.cumsum(l * np.cos(heading))])
            moment = np.array([loads[i + 1:] @ (x[i + 1:] - x[i]) for i in range(len(a))])
            change = a + moment / k - q
            q = q + 0.1 * change
            if np.abs(change).max() < 1e-8:
                break
        else:
            raise RuntimeError(f"no static balance at {p} Pa")
        heading = np.cumsum(q)
        return np.vstack([[0.0, 0.0], np.cumsum(l[:, None] * np.column_stack([np.cos(heading), np.sin(heading)]), axis=0)])

    def to_json(self, path: str | Path) -> None:
        d = {k: np.asarray(getattr(self, k)).tolist() for k in ("pressures", "angles", "lengths", "stiffness")}
        d.update(offset=self.offset, width=self.width, thickness=self.thickness, mass=self.mass)
        Path(path).write_text(json.dumps(d))

    @classmethod
    def from_json(cls, path: str | Path) -> "ReducedFinger":
        d = json.loads(Path(path).read_text())
        return cls(**{k: np.array(d[k]) if k in ("pressures", "angles", "lengths", "stiffness") else d[k] for k in d})


def tip_load(n: int, force: float) -> np.ndarray:
    """Station forces (n + 1,) [N] of ``force`` on the last station."""
    return np.eye(n + 1)[n] * force


def pad_load(n: int, force: float) -> np.ndarray:
    """Station forces (n + 1,) [N] of ``force`` spread evenly along the finger."""
    w = np.ones(n + 1)
    w[[0, -1]] = 0.5
    return force * w / w.sum()


def probe_error(f: ReducedFinger, probe: dict, k: float | None = None, key: str = "tip") -> np.ndarray:
    """Tip displacement error [m], chain minus reference, one row per force of ``probe`` (a row of
    tools/probe_finger.py): both displacements are from the free inflated shape, so the free-bend error stays out."""
    load = {"tip": tip_load, "pads": pad_load}[key]
    free = f.centreline(probe["pressure"])[-1]
    ref0 = np.asarray(probe["free"])[-1]
    return np.array([f.loaded(probe["pressure"], load(f.links, F), k)[-1] - free - (np.asarray(s)[-1] - ref0)
                     for F, s in zip(probe["forces"], probe[key])])


def fit_stiffness(f: ReducedFinger, probes: list[dict]) -> np.ndarray:
    """The ``stiffness`` table (P,) for ``f``'s pressures: at each probe the hinge stiffness that best matches the
    reference's tip-load displacements, then linear in pressure between probes, clamped outside them."""
    from scipy.optimize import minimize_scalar

    ps = np.array([r["pressure"] for r in probes])
    ks = np.array([minimize_scalar(lambda k: float((probe_error(f, r, k) ** 2).sum()), bounds=(1e-3, 2.0), method="bounded",
                                   options=dict(xatol=1e-5)).x for r in probes])
    order = np.argsort(ps)
    return np.interp(f.pressures, ps[order], ks[order])


def layer_points(rest: np.ndarray, x: np.ndarray, length: float, cfg: ReducedFingerConfig) -> np.ndarray:
    """Station points (F, N + 1, 2) of frames ``x`` (F, V, 3): the mean (x, z) of the layer's vertices around each
    station. Averaging across the width and a little along the axis takes out the mesh's irregularity."""
    stations = np.linspace(0.0, length, cfg.links + 1)
    layer = (rest[:, 2] > cfg.layer[0]) & (rest[:, 2] < cfg.layer[1])
    out = np.empty((len(x), len(stations), 2))
    for i, s in enumerate(stations):
        near = layer & (np.abs(rest[:, 0] - s) < cfg.station_width / 2)
        if not near.any():
            raise ValueError(f"no layer vertices near x = {s} m; widen station_width")
        out[:, i] = x[:, near][:, :, [0, 2]].mean(axis=1)
    return out


def extract_tables(rest: np.ndarray, x: np.ndarray, pressures: np.ndarray, length: float,
                   cfg: ReducedFingerConfig = ReducedFingerConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pressures, angles, lengths) tables from a free-bend ramp: positions ``x`` (F, V, 3) at ``pressures`` (F,)
    [Pa], increasing, on a finger with rest vertices ``rest`` and axis along +x."""
    grid = np.arange(0.0, pressures.max() + 0.5 * cfg.pressure_step, cfg.pressure_step)
    frames = np.unique([int(np.argmin(np.abs(pressures - p))) for p in grid])
    seg = np.diff(layer_points(rest, x[frames], length, cfg), axis=1)
    heading = np.unwrap(np.arctan2(seg[..., 1], seg[..., 0]), axis=1)
    heading0 = heading[0]  # the layer is not exactly straight at rest; its rest shape is the zero of the joints
    angles = np.diff(heading - heading0, axis=1, prepend=0.0)
    return pressures[frames].astype(float), angles, np.linalg.norm(seg, axis=2)

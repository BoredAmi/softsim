"""Figure: the cable-bent cantilever on every backend against Euler-Bernoulli (docs/actuation.md).

    python tools/compare_cable.py    # the runs (once)
    python tools/plot_cable.py       # ~2 min: re-solves the reference and MuJoCo at 0.7 N for the shapes

Left: tip angle against cable tension over the whole (quasi-static) ramp, from
the runs of tools/compare_cable.py. Right: the deformed beam at 0.7 N, side view,
for the reference and MuJoCo (2 mm), on top of the Euler-Bernoulli arc (with
the cable's axial shortening). Writes docs/media/cable_cantilever.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_cable import B, CONFIG, D, E, L, OUT  # noqa: E402

T = 0.7
SERIES = [  # run, label, colour (categorical slots 1-4, fixed order), marker, line style
    ("fem", "fem reference, 2 mm", "#2a78d6", "o", "-"),
    ("mujoco_1mm", "MuJoCo, 1 mm", "#1baf7a", "^", "-"),
    ("mujoco", "MuJoCo, 2 mm", "#eb6834", "s", "-"),
    ("sofa", "SOFA, 2 mm", "#eda100", "D", ":"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _theory(tension: np.ndarray) -> np.ndarray:
    return tension * D * L / (E * (1 + 2 * tension / (E * B * B)) * B**4 / 12)


def _arc(x0: float, z0: np.ndarray, tension: float) -> np.ndarray:
    """Euler-Bernoulli position of material points (x0, z0): a circular arc below the anchor, straight
    above it, shortened by the axial strain T / (E A)."""
    kappa = _theory(tension) / L
    s = z0 * (1 - tension / (E * B * B))
    sa = L * (1 - tension / (E * B * B))
    phi = kappa * np.minimum(s, sa)
    c = np.stack([(1 - np.cos(phi)) / kappa, np.sin(phi) / kappa], axis=1)
    c += np.maximum(s - sa, 0)[:, None] * np.stack([np.sin(phi), np.cos(phi)], axis=1)
    return c + x0 * np.stack([np.cos(phi), -np.sin(phi)], axis=1)


def _shapes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rest and deformed vertices at 0.7 N: reference (quasi-static) and MuJoCo (end of the ramp run)."""
    from softsim.backends.fem import FemBackend
    from softsim.backends.mujoco import MujocoBackend
    from softsim.config.loader import load_yaml
    from softsim.config.schema import validate
    from softsim.fem.solid3d import Solid3DSolver
    from softsim.scenarios.inflation import InflationConfig, InflationScenario

    scenario = InflationScenario(validate(InflationConfig, load_yaml(CONFIG)), CONFIG.parent)
    scene = scenario.build_scene()
    rest = scene.soft_bodies[0].mesh.points
    fem = FemBackend()
    fem.build(scene)
    s = Solid3DSolver(fem._problem, dynamic=False)
    s.tensions[:] = [T]
    assert s.advance(1.0)
    n = len(rest)
    ref = s.p.mesh.nodes[:n] + s.y[:3 * n].reshape(-1, 3)[:n]
    mj = MujocoBackend()
    mj.build(scene)
    mj.step(4500)  # to 450 ms: 0.7 N, held
    return {"fem": (rest, ref), "mujoco": (rest, mj.state().soft["beam"].vertices)}


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.35, 1]})

    tension = np.linspace(0, T, 50)
    ax.plot(tension, np.degrees(_theory(tension)), color=INK, lw=2, ls="--", label="Euler–Bernoulli", zorder=5)
    for run, label, color, marker, ls in SERIES:
        path = OUT / run / "timeseries.parquet"
        if not path.exists():
            continue
        ts = pd.read_parquet(path)
        ramp = ts[ts.t <= 0.4]
        ax.plot(ramp.cable_tension, np.degrees(ramp.bend_angle), color=color, lw=2, ls=ls, marker=marker,
                markevery=40, ms=8, mec="white", mew=1.5, label=label)
    ax.set_xlabel("cable tension [N]")
    ax.set_ylabel("tip angle [°]")
    ax.set_title("Tip angle over the whole ramp", loc="left", color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlim(0, T * 1.02)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left")

    shapes = _shapes()
    z0 = np.linspace(0, 0.06, 121)
    for x0 in (-B / 2, B / 2):
        arc = _arc(x0, z0, T)
        bx.plot(np.full_like(z0, x0) * 1e3, z0 * 1e3, color=GRID, lw=2, zorder=1)
        bx.plot(arc[:, 0] * 1e3, arc[:, 1] * 1e3, color=INK, lw=1.5, ls="--", zorder=3,
                label="Euler–Bernoulli" if x0 > 0 else None)
    for run, label, color, marker in (("fem", "fem reference", "#2a78d6", "o"), ("mujoco", "MuJoCo, 2 mm", "#eb6834", "s")):
        rest, x = shapes[run]
        face = np.isclose(np.abs(rest[:, 0]), B / 2) & np.isclose(rest[:, 1], -B / 2)
        for side in (-1, 1):
            sel = face & (np.sign(rest[:, 0]) == side)
            order = np.argsort(rest[sel, 2])
            bx.plot(x[sel, 0][order] * 1e3, x[sel, 2][order] * 1e3, color=color, lw=0, marker=marker, ms=6,
                    mec="white", mew=0.8, label=label if side > 0 else None, zorder=4 if run == "fem" else 2)
    bx.set_aspect("equal")
    bx.set_xlim(-8, 16)
    bx.set_xlabel("x [mm]")
    bx.set_ylabel("z [mm]")
    bx.set_title(f"Side faces at {T} N (rest in grey)", loc="left", color=INK, fontsize=11)
    bx.grid(color=GRID, lw=0.8)
    bx.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)

    fig.suptitle("Cable-bent cantilever: 10 × 10 × 60 mm, E = 500 kPa, cable 3 mm off-axis", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    out = ROOT / "docs" / "media" / "cable_cantilever.png"
    fig.savefig(out, dpi=130, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()

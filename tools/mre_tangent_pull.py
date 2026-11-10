"""Pull the tip of an MRE strip wrapped on Bernat et al.'s ring magnet along the rim's tangent: a peel that cannot happen.

    python tools/mre_tangent_pull.py --out runs/mre/peel_4mm_clamped.csv        # ~25 min on a GPU
    python tools/mre_tangent_pull.py --plot runs/mre/peel_4mm_clamped.csv docs/media/mre_tangent_pull.png

The strip (scenarios/magnetic/mre_finger_peel.yaml, ``--width`` across it: nothing fitted) is clamped at its root, as on the
gripper, and starts wrapped half-way round the magnet's rim: straight from the root to the rim, 180 deg round it, the rest
straight on along the tangent (:func:`wrapped`). Held by its tip it settles; then the tip is pulled along that last straight
line.

**What it shows.** A free end peeled along the rim's tangent would need ``F = (T - E I / 2 rho) / rho`` by energy balance
(the pull does ``F rho`` per angle unrolled; the strip gives back the rolling torque ``T`` of tools/mre_rolling_torque.py
and releases the bending energy ``E I / 2 rho``, with ``rho = R + t / 2``): 0.119 N for the 4 mm strip. The finite-element
pull does not level off there: 0.064 N held, 0.63 N after 1 mm, 1.9 N after 6.2 mm and still rising, so no peel. The reason
is geometric: with the root clamped, the wrap lies between the root's tangent point and the tail's, and a pull along the
tail's own line leaves both where they are, so nothing unrolls and the strip is stretched instead (and, the curve's
flattening suggests, slides on the rim against friction over a growing length; vertices were not saved to show it).
A clamped finger peels only when the push lets its free end swing round, at a lever set by where and which way it is
pushed, so Bernat et al.'s measured plateau (Fig. 11) depends on their gauge's contact point, which is not given. (A strip
held by its tip alone, frictionless, is no test either: the magnet slid it sideways towards a rim edge and Newton failed.)

**Settling.** The wrapped start is not an equilibrium (the bend starts and ends abruptly), and with numerical damping alone
it still swung at 0.4 m/s after 20 ms. ``--prony`` gives the silicone a Maxwell branch (default g = 0.3, tau = 2 ms) with
the instantaneous modulus raised to keep the long-term one at the card's 0.81 MPa (softsim's moduli are instantaneous):
it settled to 0.007 m/s in 30 ms, and the slow pull sees the card.

**Width.** Bernat's 15 mm strip overhangs the 5 mm magnet by 5 mm a side; the magnet's poles pull those flaps in at
metres per second and each step took minutes, so this runs on a strip narrower than the magnet (4 mm by default).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SCENARIO = ROOT / "scenarios" / "magnetic" / "mre_finger_peel.yaml"
WRAP = np.pi  # rad round the rim at the start
E_CARD, THICKNESS, LENGTH = 0.81e6, 0.002, 0.060
INK, MUTED, GRID, FE, ANSYS_C = "#17262e", "#566a75", "#c3ced4", "#1c8c8a", "#c8571f"


def build(width: float, prony: tuple[float, float] | None):
    from softsim.backends.fem.backend import FemBackend
    from softsim.config.loader import load_yaml
    from softsim.runner import Simulation

    raw = load_yaml(SCENARIO)
    raw["body"]["geometry"]["size"] = ["2 mm", f"{width * 1e3:g} mm", "60 mm"]
    if prony is not None:
        g, tau = prony
        raw["body"]["material"]["elastic"]["matrix"]["youngs_modulus"] = E_CARD / (1 - g)
        raw["body"]["material"]["damping"] = {"model": "prony", "g": [g], "tau": [tau]}
    sim = Simulation(raw, [], base_dir=SCENARIO.parent)
    scene = sim.scenario.build_scene()
    fem = FemBackend()
    fem.build(scene)
    return scene, fem


def wrapped(nodes: np.ndarray, centre: np.ndarray, radius: float, thickness: float, wrap: float) -> np.ndarray:
    """Displacements taking the straight strip (x through its thickness, |x| <= t / 2, the face x = t / 2 towards the magnet;
    z along it from the root at 0) round a magnet of ``radius`` at ``centre`` (axis y) whose rim touches that face: straight
    up to the rim, ``wrap`` rad round it over the top (clockwise seen from +y), then straight on along the tangent there,
    bent about the mid-surface (which keeps its length)."""
    x, z = nodes[:, 0], nodes[:, 2]
    rho = radius + 0.5 * thickness
    u = z - centre[2]  # mid-surface length past the first contact
    r = radius + (0.5 * thickness - x)  # distance from the axis, the face on the rim
    theta = np.pi - np.clip(u, 0, rho * wrap) / rho
    on = u > 0
    past = u > rho * wrap
    end = np.pi - wrap
    tangent = np.array([np.sin(end), -np.cos(end)])  # (x, z) direction of increasing length at the end of the wrap
    beyond = np.maximum(u - rho * wrap, 0.0)
    nx = np.where(on, centre[0] + r * np.cos(theta) + beyond * tangent[0], x)
    nz = np.where(on, centre[2] + r * np.sin(theta) + beyond * tangent[1], z)
    return np.column_stack([nx, nodes[:, 1], nz]) - nodes


def closed_form(torque: float, width: float, radius: float = 0.010) -> float:
    """``F = (T - E I / 2 rho) / rho`` [N] for the scenario's strip of ``width``, given its rolling torque [N m]."""
    rho = radius + THICKNESS / 2
    return (torque - E_CARD * width * THICKNESS ** 3 / 12 / (2 * rho)) / rho


def run(out: str, width: float, settle: float, stroke: float, speed: float, dt: float, prony) -> None:
    from mre_rolling_torque import Strip

    from softsim.fem.solid3d import Solid3DSolver

    scene, fem = build(width, prony)
    magnet = scene.magnetic_field.magnets[0]
    chi = scene.soft_bodies[0].material.elastic.chi
    torque = Strip(THICKNESS, chi, 1e-3, width=width, edge=magnet.edge).torque()
    meta = {"width": width, "chi": chi, "edge": magnet.edge, "torque": torque, "closed_form": closed_form(torque, width),
            "prony": prony, "speed": speed, "settle": settle, "dt": dt}
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"rolling torque {torque * 1e3:.3f} mN m -> energy balance {meta['closed_form']:.4f} N", flush=True)

    centre, radius = np.asarray(magnet.position), magnet.outer_radius
    s = Solid3DSolver(fem._problem, numerical_damping=0.25)
    s.magnetic_field = scene.magnetic_field.parameters(0.0)
    nodes = s.p.mesh.nodes
    y0 = np.zeros(s.ndof)
    y0[:3 * s.n] = wrapped(nodes, centre, radius, THICKNESS, WRAP).ravel()
    s.y = y0.copy()
    fixed = np.flatnonzero(s.fixed)
    tip = [np.flatnonzero(np.isin(fixed, d)) for d in fem._prescribed[0][2]]  # per axis: slots of the tip's dofs
    direction = np.array([np.sin(np.pi - WRAP), 0.0, -np.cos(np.pi - WRAP)])  # along the last straight part

    def pull(t: float) -> float:
        """The tip's travel [m]: none while settling, then up to ``speed`` with a minimum-jerk velocity ramp
        (``v = speed (10 u^3 - 15 u^4 + 6 u^5)`` over 20 ms), then steady."""
        ramp, tt = 0.02, t - settle
        if tt <= 0:
            return 0.0
        if tt < ramp:
            u = tt / ramp
            return speed * ramp * (2.5 * u ** 4 - 3 * u ** 5 + u ** 6)
        return speed * (tt - 0.5 * ramp)

    def motion(t: float) -> np.ndarray:
        v = y0[fixed].copy()
        for k in range(3):
            v[tip[k]] += pull(t) * direction[k]
        return v

    s.fixed_motion = motion
    rows, wall = [], time.perf_counter()
    end = settle + 0.02 + stroke / speed
    top = np.isclose(nodes[:, 2], LENGTH)
    while s.t < end - 1e-12:
        if not s.advance(dt):
            print(f"Newton failed at t = {s.t * 1e3:.2f} ms", flush=True)
            break
        reaction = np.array([s.residual[fixed[tip[k]]].sum() for k in range(3)])
        x = nodes + s.y[:3 * s.n].reshape(-1, 3)
        rows.append({"t": s.t, "pull": pull(s.t), "force": float(reaction @ direction),  # the holder's pull on the strip
                     "vmax": float(np.abs(s.v[:3 * s.n]).max()), "tip_x": float(x[top, 0].mean()),
                     "tip_z": float(x[top, 2].mean())})
        if len(rows) % 40 == 0:
            r = rows[-1]
            print(f"t {r['t'] * 1e3:7.2f} ms  pull {r['pull'] * 1e3:5.2f} mm  force {r['force']:7.4f} N  vmax {r['vmax']:.3f} m/s"
                  f"  wall {time.perf_counter() - wall:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


def plot(csv: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(csv)
    meta = json.loads(Path(csv).with_suffix(".json").read_text())
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    p = d[d["pull"] > 0]
    ax.plot(p["pull"] * 1e3, p["force"], color=FE, linewidth=2,
            label=f"finite elements: root clamped, {meta['width'] * 1e3:g} mm strip, chi {meta['chi']:.2f}")
    ax.axhline(meta["closed_form"], color=ANSYS_C, linestyle="--", linewidth=1.2,
               label=f"a free end's steady peel, (T - EI / 2 rho) / rho = {meta['closed_form']:.3f} N (not reached)")
    ax.set_xlabel("tip pulled along the rim's tangent [mm]", color=INK)
    ax.set_ylabel("pull on the tip [N]", color=INK)
    ax.set_ylim(bottom=0)
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Pulling a clamped MRE strip's tip along the rim's tangent: it does not level off, so no peel", color=INK, fontsize=10)
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/mre/peel_4mm_clamped.csv")
    ap.add_argument("--width", type=float, default=4e-3, help="m across the strip (the magnet is 5 mm thick)")
    ap.add_argument("--settle", type=float, default=0.03, help="s with the tip held before the pull")
    ap.add_argument("--stroke", type=float, default=0.006, help="m pulled at steady speed")
    ap.add_argument("--speed", type=float, default=0.02, help="m/s, steady")
    ap.add_argument("--dt", type=float, default=2.5e-4, help="s per step")
    ap.add_argument("--prony", type=float, nargs=2, default=[0.3, 2e-3], metavar=("G", "TAU"),
                    help="Maxwell branch for settling; the long-term modulus stays the card's")
    ap.add_argument("--plot", nargs=2, metavar=("CSV", "PNG"), help="only plot a finished run")
    args = ap.parse_args()
    if args.plot:
        plot(*args.plot)
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run(args.out, args.width, args.settle, args.stroke, args.speed, args.dt, tuple(args.prony))


if __name__ == "__main__":
    main()

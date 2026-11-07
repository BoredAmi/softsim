"""Draw one finite-element grasp against the fast finger, and print the numbers that compare them.

    python tools/plot_fe_grasp.py runs/grasp/box_10g_2mm.npz --finger softsim/reduced/wang_finger.json \
        --out docs/media/gripper_fe_grasp.png

Top: the fingers and the box in the x-z plane at four moments, the touching pad nodes marked. Bottom: the box's height
above the table against the palm's rise, for both models. Reads the file tools/run_fe_grasp.py writes, and runs the same
grasp on the fast finger. The box is scenarios/grasp/box.yaml's; a node touches it within ``TOUCH`` of its side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS
from softsim.scenarios.grasp import is_held

HALF_X, HALF_Z = 0.016, 0.03  # m, the box
TOUCH = 3e-4  # m
LIFT, PRESSURE, MASS, FRICTION = 0.03, 25e3, 0.01, 0.8  # m, Pa, kg, -
OFF_TABLE = 5e-5  # m
MOMENTS = (0.7, 0.9, 0.95, 1.5)  # s
INK, MUTED, FE, FAST = "#17262e", "#566a75", "#1c8c8a", "#c8571f"


def touching(x: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Nodes on the sides of the box, (K,)."""
    return np.flatnonzero((np.abs(x[:, 0] - box[0]) - HALF_X < TOUCH) & (np.abs(x[:, 2] - box[2]) < HALF_Z + TOUCH) & (np.abs(x[:, 1] - box[1]) < 0.0125))


def fe(path: Path) -> dict:
    d = np.load(path)
    height = d["object"][:, 2] - HALF_Z
    settled = height[int(np.argmax(d["t"] >= 0.1))]  # it starts on a small clearance and lands
    off = int(np.argmax((height > settled + OFF_TABLE) & (d["t"] > 0.1)))
    before = int(np.argmax(d["t"] >= 0.7))
    return dict(d=d, height=height, palm=d["palm"], lift_off=d["palm"][off], rise=height[-1] - settled, grip_before=d["grip"][before],
                grip_hold=d["grip"][-20:].mean(), steady=np.ptp(height[-20:]), held=float(is_held(d["t"], height - settled)))


def fast(f: ReducedFinger) -> dict:
    """The same grasp on the fast finger, recording height against palm."""
    g = ReducedGripper(f, fingers=2, shape="box", size=OBJECTS["box"], mass=MASS, friction=FRICTION)
    for i in range(1500):
        g.step(PRESSURE * min(i / 600, 1))
    z0, grip = g.object_height, g.grip_force()
    palm, height = [], []
    for i in range(2000):
        g.step(PRESSURE, LIFT * min(i / 500, 1))
        palm.append(g.palm_rise)
        height.append(g.object_height - z0)
    palm, height = np.array(palm), np.array(height)
    return dict(palm=palm, height=height, lift_off=palm[int(np.argmax(height > OFF_TABLE))], rise=height[-1], grip_before=grip,
                grip_hold=g.grip_force(), steady=np.ptp(height[-200:]), held=float(is_held(np.arange(2000) * g.model.opt.timestep, height)))


def draw(a: dict, b: dict, out: Path) -> None:
    d = a["d"]
    fig = plt.figure(figsize=(10, 7.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1])
    axes = top.subplots(1, len(MOMENTS), sharey=True)
    surface = np.unique(d["tris"])
    for ax, moment in zip(axes, MOMENTS):
        i = int(np.argmin(np.abs(d["t"] - moment)))
        x, box = d["x"][i], d["object"][i]
        ax.scatter(x[surface, 0] * 1e3, x[surface, 2] * 1e3, s=0.4, color=FE, alpha=0.5, linewidths=0, rasterized=True)
        ax.add_patch(plt.Rectangle(((box[0] - HALF_X) * 1e3, (box[2] - HALF_Z) * 1e3), 2e3 * HALF_X, 2e3 * HALF_Z, fill=False, edgecolor=INK, linewidth=1.5))
        on = touching(x, box)
        ax.scatter(x[on, 0] * 1e3, x[on, 2] * 1e3, s=14, color=FAST, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set(xlim=(-50, 50), ylim=(-4, 180), aspect="equal", title=f"{moment:.2f} s, palm {d['palm'][i] * 1e3:.1f} mm", xlabel="x, mm")
    axes[0].set_ylabel("z, mm")
    top.suptitle("Finite-element grasp: orange marks the pad nodes touching the box", color=INK)
    ax = bottom.subplots()
    ax.plot(a["palm"] * 1e3, a["height"] * 1e3, color=FE, linewidth=2, label="finite element")
    ax.plot(b["palm"] * 1e3, b["height"] * 1e3, color=FAST, linewidth=2, label="fast finger")
    ax.plot([0, LIFT * 1e3], [0, LIFT * 1e3], color=MUTED, linewidth=1, linestyle="--", label="no slack")
    ax.set(xlabel="palm rise, mm", ylabel="box above the table, mm", xlim=(0, 31), ylim=(-1, 31))
    ax.grid(color="#c3ced4", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = fe(args.run), fast(ReducedFinger.from_json(args.finger))
    print(f"{'':28}{'FE':>9}{'fast':>9}")
    for label, key, scale in (("palm rise at lift-off, mm", "lift_off", 1e3), ("final rise, mm", "rise", 1e3), ("grip before the lift, N", "grip_before", 1),
                              ("grip while holding, N", "grip_hold", 1), ("height wobble at the end, mm", "steady", 1e3),
                              ("held (1 = up and staying)", "held", 1)):
        print(f"{label:28}{a[key] * scale:9.3f}{b[key] * scale:9.3f}")
    draw(a, b, args.out)
    print(args.out)


if __name__ == "__main__":
    main()

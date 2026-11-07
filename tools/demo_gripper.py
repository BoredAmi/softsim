"""The fast gripper on the desktop, for when no browser will do: sliders for pressure and lift, MuJoCo's viewer.

    python tools/demo_gripper.py --finger softsim/reduced/wang_finger.json --shape box --mass 10 --fingers 2
    python tools/demo_gripper.py --finger softsim/reduced/wang_finger.json --selftest    # no window

The same model as the page (docs/viewer/gripper.html): :class:`softsim.reduced.ReducedGripper`. Needs a display,
``mujoco.viewer`` and tkinter. The slider window shows grip force and object height.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS, run_grasp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finger", type=Path, required=True)
    ap.add_argument("--shape", choices=list(OBJECTS), default="box")
    ap.add_argument("--mass", type=float, default=10.0, help="g")
    ap.add_argument("--friction", type=float, default=0.8)
    ap.add_argument("--fingers", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    f = ReducedFinger.from_json(a.finger)
    if a.selftest:
        print(run_grasp(f, 30e3, a.mass / 1e3, a.friction, a.shape, a.fingers))
        return

    import tkinter as tk

    g = ReducedGripper(f, fingers=a.fingers, shape=a.shape, size=OBJECTS[a.shape], mass=a.mass / 1e3, friction=a.friction)
    win = tk.Tk()
    win.title("Soft gripper")
    pressure = tk.Scale(win, from_=0, to=40, resolution=0.5, orient="horizontal", length=320, label="Pressure [kPa]")
    lift = tk.Scale(win, from_=0, to=60, resolution=1, orient="horizontal", length=320, label="Lift [mm]")
    text = tk.Label(win, font=("monospace", 11))
    for w in (pressure, lift, text):
        w.pack()
    p, z = 0.0, 0.0
    with mujoco.viewer.launch_passive(g.model, g.data) as view:
        while view.is_running():
            t0 = time.perf_counter()
            for _ in range(16):
                p += 0.025 * (pressure.get() * 1e3 - p)  # smoothed over about 40 ms, so a slider jump is no step load
                z += 0.025 * (lift.get() * 1e-3 - z)
                g.step(p, z)
            view.sync()
            text.config(text=f"grip {g.grip_force():.2f} N per finger   object {g.object_height * 1e3:.0f} mm")
            win.update()
            time.sleep(max(0.0, 0.016 - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()

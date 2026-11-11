"""Interactive demo: drive a soft actuator with sliders and watch the MuJoCo body move.

    python tools/demo_sliders.py                              # PneuNet finger, pressure slider
    python tools/demo_sliders.py scenarios/actuation/cable_finger.yaml      # tendon finger, tension slider
    python tools/demo_sliders.py scenarios/actuation/cable_arm.yaml         # two cables: the S-shape
    python tools/demo_sliders.py --selftest                   # no window: steps the sliders' code path

One slider per actuator of the scenario (a pressure in kPa, a cable tension in N, a stroke in mm, a force in
mN), up to the largest value of its schedule. The physics is the scenario's own (``Simulation`` on the MuJoCo
backend, ``backend.set_control``); the MuJoCo viewer shows the flex body, orbit with the left mouse button.
Sliders move the control smoothly (a first-order lag, 40 ms) so a jump of the slider is not a step load.
Needs a display, ``mujoco.viewer`` and tkinter. The physics runs in real time or slower; the slider
window shows the simulated/wall time ratio and the body's ``bend_angle`` when the scenario reports one.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from softsim.actuation import CableActuator, PressureActuator
from softsim.runner import Simulation

ROOT = Path(__file__).resolve().parents[1]
LAG = 0.04  # s: slider smoothing time constant


def unit_of(a) -> tuple[str, float]:
    """(label unit, SI per unit) of an actuator's control."""
    if isinstance(a, PressureActuator):
        return "kPa", 1e3
    if isinstance(a, CableActuator):
        return ("mm", 1e-3) if a.stroke_controlled else ("N", 1.0)
    return "mN", 1e-3


def limit_of(a) -> float:
    """Slider range [SI]: the largest value the scenario's schedule reaches (a default if it has none)."""
    values = [abs(v) for _, v in a.schedule] or [abs(a.value(0.0))]
    return max(max(values), 1e-12)


class Driver:
    """Targets (SI) per actuator, smoothed into ``backend.set_control`` as the physics advances."""

    def __init__(self, sim: Simulation):
        self.sim, self.b = sim, sim.backend
        self.target = {a.name: 0.0 for a in sim.scene.actuators}
        self.value = dict(self.target)
        self.dt = self.b.model.opt.timestep

    def advance(self, wall_dt: float, max_steps: int = 400) -> int:
        """Step the physics for about ``wall_dt`` of simulated time (capped); returns the steps taken."""
        n = int(np.clip(wall_dt / self.dt, 1, max_steps))
        k = 1.0 - np.exp(-self.dt * 10 / LAG)  # lag updated every 10 steps
        for i in range(0, n, 10):
            for name, tgt in self.target.items():
                self.value[name] += k * (tgt - self.value[name])
                self.b.set_control(name, self.value[name])
            self.b.step(min(10, n - i))
        return n


def selftest(sim: Simulation) -> None:
    """Headless: ramp every slider to its limit, hold, and print the metrics (the window's code path)."""
    d = Driver(sim)
    for a in sim.scene.actuators:
        d.target[a.name] = limit_of(a)
    t0 = time.time()
    while sim.time < 0.5:
        d.advance(0.01)
    m = sim.metrics()
    print(f"simulated {sim.time:.3f} s in {time.time() - t0:.1f} s wall; targets {d.target}")
    print({k: round(v, 5) for k, v in m.items() if isinstance(v, float)})


def gui(sim: Simulation) -> None:
    import tkinter as tk

    import mujoco.viewer

    d = Driver(sim)
    root = tk.Tk()
    root.title(f"softsim sliders: {sim.config.name}")
    labels = {}
    for a in sim.scene.actuators:
        unit, si = unit_of(a)
        top = limit_of(a) / si
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=8, pady=4)
        text = tk.Label(frame, text=f"{a.name} [{unit}]", width=22, anchor="w")
        text.pack(side="left")
        scale = tk.Scale(frame, from_=0.0, to=top, resolution=top / 200, orient="horizontal", length=360,
                         command=lambda v, n=a.name, s=si: d.target.__setitem__(n, float(v) * s))
        scale.pack(side="left")
        labels[a.name] = scale
    status = tk.Label(root, text="", anchor="w", font=("monospace", 10))
    status.pack(fill="x", padx=8, pady=4)
    tk.Button(root, text="reset", command=lambda: [s.set(0) for s in labels.values()]).pack(pady=4)

    with mujoco.viewer.launch_passive(d.b.model, d.b.data) as viewer:
        last = time.time()
        sim_last = sim.time
        while viewer.is_running():
            now = time.time()
            wall = now - last
            last = now
            d.advance(min(wall, 0.05))
            viewer.sync()
            root.update()
            m = sim.metrics()
            ratio = (sim.time - sim_last) / max(wall, 1e-9)
            sim_last = sim.time
            bend = f"  bend {np.degrees(m['bend_angle']):5.1f} deg" if "bend_angle" in m else ""
            status.config(text=f"t = {sim.time:6.2f} s   sim/wall = {ratio:4.2f}{bend}")
    root.destroy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", default=str(ROOT / "scenarios/actuation/pneunet.yaml"))
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    sim = Simulation(args.config, overrides=args.set, backend="mujoco")
    selftest(sim) if args.selftest else gui(sim)


if __name__ == "__main__":
    main()

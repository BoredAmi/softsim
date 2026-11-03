"""Figure for the rotating-mount verification: docs/media/rotating_mount.png.

    python tools/plot_spin.py            # ~6 min: scenarios/actuation/spinning_cantilever.yaml on every backend

A round beam bends the same in every direction, so in the world its tip moves as without the spinning clamp.
Top row: the tip's trajectory in the world with the mount spinning (two turns in 0.5 s) and without it, on the
3D reference (the world frame, no fictitious forces, trapezoidal rule at 2.5 ms: 6.4% apart) and on MuJoCo and
SOFA (the mount's frame; 17-21% apart at 3 mm linear tetrahedra: the mesh's 4% stiffness anisotropy and implicit
Euler's frame-dependent damping, not the forces, which are checked exactly in tests/test_rotating_mount.py).
Bottom row: SOFA with a fictitious force removed or reversed, which is what a missing term or a wrong sign would
do (131%, 48% and 355% apart).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "actuation" / "spinning_cantilever.yaml"
OUT = ROOT / "docs" / "media"
STEP = {"fem": "2.5ms", "sofa": "1ms", "mujoco": "0.1ms"}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from softsim.backends.fem import FemBackend
    from softsim.errors import SoftSimError
    from softsim.runner import run_config
    from softsim.scene import MountFrame

    original = MountFrame.fictitious
    # the reference's spinning runs use the trapezoidal rule (undamped): the still run must too, or the default
    # Newmark's numerical damping decays its oscillation and the two differ by 150%
    FemBackend._new_solver = lambda self: self._solver_cls(self._problem, numerical_damping=0.0)

    def run(name, spin=True, broken=None):
        MountFrame.fictitious = broken or original
        try:
            ov = [f"backend={name}", f"simulation.timestep={STEP[name]}", "output.save_model=false"]
            return run_config(SCENARIO, ov + ([] if spin else ["body.mount_rotation=null"])).timeseries
        finally:
            MountFrame.fictitious = original

    def distance(a, b):
        n = min(len(a), len(b))
        return np.hypot(a.tip_x.to_numpy()[:n] - b.tip_x.to_numpy()[:n], a.tip_y.to_numpy()[:n] - b.tip_y.to_numpy()[:n]).max() / b.tip_x.max()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
    plains = {}
    for ax, name, title in zip(axes[0], ("fem", "sofa", "mujoco"), ("3D reference (world frame)", "SOFA (mount frame)", "MuJoCo (mount frame)")):
        try:
            plain, spun = run(name, spin=False), run(name)
        except (ImportError, SoftSimError) as exc:
            print(f"{name}: skipped ({exc})", file=sys.stderr)
            continue
        plains[name] = plain
        ax.plot(plain.t * 1e3, plain.tip_x * 1e3, "k-", lw=2.5, alpha=0.35, label="not spinning")
        ax.plot(spun.t * 1e3, spun.tip_x * 1e3, "C0-", label="spinning")
        ax.set_title(f"{title}: {100 * distance(spun, plain):.1f}% apart", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        print(f"{name}: spinning vs not, {100 * distance(spun, plain):.1f}% of the sag", flush=True)
    if "sofa" in plains:
        variants = {
            "no Coriolis force": lambda self, x, v: original(self, x, v) + 2.0 * np.cross(self.omega, v),
            "no centrifugal force": lambda self, x, v: original(self, x, v) + np.cross(self.omega, np.cross(self.omega, x - self.center)),
            "Coriolis force reversed": lambda self, x, v: original(self, x, v) + 4.0 * np.cross(self.omega, v),
        }
        for ax, (label, broken) in zip(axes[1], variants.items()):
            plain = plains["sofa"]
            spun = run("sofa", broken=broken)
            ax.plot(plain.t * 1e3, plain.tip_x * 1e3, "k-", lw=2.5, alpha=0.35, label="not spinning")
            ax.plot(spun.t * 1e3, spun.tip_x * 1e3, "C3-", label=label)
            ax.set_title(f"SOFA, {label}: {100 * distance(spun, plain):.0f}% apart", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlabel("time [ms]")
            print(f"sofa {label}: {100 * distance(spun, plain):.0f}%", flush=True)
    axes[0][0].set_ylabel("tip x in the world [mm]")
    axes[1][0].set_ylabel("tip x in the world [mm]")
    fig.suptitle("Round cantilever under horizontal gravity on a mount spinning about its own axis (two turns in 0.5 s)", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "rotating_mount.png", dpi=120)
    print(f"wrote {OUT / 'rotating_mount.png'}")


if __name__ == "__main__":
    main()

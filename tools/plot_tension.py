"""Figure for the prescribed-motion verification: docs/media/prescribed_tension.png.

    python tools/plot_tension.py            # ~2 min: three backends, a 30% roller tension test each

Runs scenarios/actuation/tension_test.yaml pulled to 30% in 400 ms on MuJoCo, SOFA (needs SOFA_ROOT)
and the 3D reference and plots the reaction against the stretch with the exact homogeneous solution
(neo-Hookean, ``softsim.materials.homogeneous``), and each backend's relative difference from it.
MuJoCo and SOFA run the stable neo-Hookean law, 0.9% stiffer at 10% (not locking: a uniform stretch
cannot lock). SOFA's reaction is one step (1 ms) old (SofaPython3 does not expose the end-of-step
forces), which against the current stretch is a force offset of stiffness x speed x 1 ms: -5% at the
start of the pull, where the force is small, and gone at the hold. Writes the table of the figure next to it as prescribed_tension.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "actuation" / "tension_test.yaml"
OUT = ROOT / "docs" / "media"
H, AREA, STRETCH = 0.02, 1e-4, 0.3
PATH = f"body.prescribed[1].path=[[0 ms, [null, null, 0]], [400 ms, [null, null, {STRETCH * H}]]]"
BACKENDS = {"mujoco": [], "sofa": ["simulation.timestep=1ms"], "fem": ["simulation.timestep=5ms"]}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from softsim.errors import SoftSimError
    from softsim.materials import load_material
    from softsim.materials.homogeneous import curve
    from softsim.runner import run_config

    card = load_material({"density": 1070, "elastic": {"model": "neo_hookean", "youngs_modulus": 100e3,
                                                       "poisson_ratio": 0.45}})
    lam = np.linspace(1.0, 1 + STRETCH, 61)[1:]
    exact = curve(card, "uniaxial", lam).stress.to_numpy() * AREA
    runs = {}
    for name, extra in BACKENDS.items():
        try:
            ts = run_config(SCENARIO, [f"backend={name}", PATH, "simulation.duration=0.4", *extra,
                                       "output.save_model=false"]).timeseries
        except (ImportError, SoftSimError) as exc:
            print(f"{name}: skipped ({exc})", file=sys.stderr)
            continue
        runs[name] = ts
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a.plot(lam, exact, "k-", lw=2.5, alpha=0.35, label="exact (homogeneous, neo-Hookean)")
    table = {"stretch": lam, "exact_N": exact}
    for i, (name, ts) in enumerate(runs.items()):
        stretch = 1 + ts.top_displacement_z.to_numpy() / H
        pull = ts.top_reaction_z.to_numpy()
        a.plot(stretch, pull, "-", color=f"C{i}", label=name)
        interp = np.interp(lam, stretch, pull)
        table[f"{name}_N"] = interp
        b.plot(lam, 100 * (interp / exact - 1), "-", color=f"C{i}", label=name)
    a.set_xlabel("stretch")
    a.set_ylabel("reaction of the pulled face [N]")
    a.set_title("10 x 10 x 20 mm bar on a roller, pulled in 400 ms")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    b.set_xlabel("stretch")
    b.set_ylabel("reaction / exact - 1 [%]")
    b.set_title("difference from exact (SOFA's reaction is one step old)", fontsize=9)
    b.axhline(0, color="k", lw=0.8)
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "prescribed_tension.png", dpi=130)
    pd.DataFrame(table).to_csv(OUT / "prescribed_tension.csv", index=False)
    print(pd.DataFrame(table).iloc[[9, 19, 39, 59]].to_string(index=False))


if __name__ == "__main__":
    main()

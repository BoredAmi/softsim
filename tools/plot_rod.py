"""Figure for the rod backend: docs/media/rod_backend.png.

    python tools/plot_rod.py            # ~2 min (SOFA_ROOT set; the 3D reference runs on the GPU if there is one)

Left: scenarios/actuation/cable_cantilever.yaml at four times its tension (2.8 N). The cable's pure moment T d
bends the beam to a circular arc of tilt T d L / EI = 1.11 rad (64 deg); the rod's centreline is on the arc of the
beam shortened by the cable's compression, L / (1 + N / EA); a beam linear in the rotation would stay at 60 mm.
Right: scenarios/actuation/cable_arm.yaml, quasi-static (stiffness damping), the tilt of the sections along the
arm on the rod, the 3D reference (2 mm quadratic tetrahedra) and linear piecewise Euler-Bernoulli. The rod follows the linear solution; the
reference is lower by the compression stiffening E_t = E (1 + 2 N / EA) of its Neo-Hookean material
(tests/test_cable_arm.py), not an error of the rod.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "docs" / "media"


def centreline(points: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Mean (x, z) of the vertices at each rest height: the section centres."""
    levels = np.unique(np.round(rest[:, 2], 9))
    return np.array([points[np.abs(rest[:, 2] - z) < 1e-9][:, [0, 2]].mean(axis=0) for z in levels])


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import test_cable_arm as arm_test
    import test_rod_backend as rod_test

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # -- curl --------------------------------------------------------------------------------------------
    b = rod_test._run("cable_cantilever", ["actuators[0].schedule=[[0, 0], [0.4, 2.8], [0.5, 2.8]]"])
    x = b.state().soft["beam"].vertices
    c = centreline(x, b.model.rest_points) * 1e3
    theta = 2.8 * 0.003 * 0.055 / (500e3 * 1e-8 / 12)
    length = 0.055 / (1 + 2.8 / 50)
    s = np.linspace(0, length, 100)
    arc = np.column_stack([length / theta * (1 - np.cos(theta * s / length)), length / theta * np.sin(theta * s / length)]) * 1e3
    tilt = rod_test._tilt(b, 0.06)
    tail = arc[-1] + np.array([np.sin(tilt), np.cos(tilt)]) * np.linspace(0, 5, 10)[:, None]
    ax1.plot([0, 0], [0, 60], color="0.75", lw=6, label="rest (60 mm)")
    ax1.plot(*np.vstack([arc, tail]).T, color="C1", lw=2, label=f"arc of the compressed beam, tilt {np.degrees(tilt):.0f}°")
    ax1.plot(c[:, 0], c[:, 1], "o", color="C0", ms=3.5, label="rod (BeamAdapter)")
    ax1.set(xlabel="x [mm]", ylabel="z [mm]", title="Cable cantilever at 2.8 N: large rotation", aspect="equal")
    ax1.legend(loc="lower left", fontsize=8)

    # -- arm: tilt of the cross-sections along the arm -------------------------------------------------------
    def tilts(points, rest):
        levels = np.unique(np.round(rest[:, 2], 9))
        return levels * 1e3, np.degrees([np.arctan(-np.polyfit(points[np.abs(rest[:, 2] - z) < 1e-9][:, 0],
                                                                 points[np.abs(rest[:, 2] - z) < 1e-9][:, 2], 1)[0])
                                         for z in levels])

    D = "{model: rayleigh, beta: 0.3}"
    b = rod_test._run("cable_arm", ["simulation.timestep=1ms", f"body.material.damping={D}", f"body.zones[0].material.damping={D}",
                                    f"body.zones[1].material.damping={D}", "simulation.duration=2"], until=1.5)
    z, th = tilts(b.state().soft["arm"].vertices, b.model.rest_points)
    e, bw = 114e3, 0.012
    inertia = bw**4 / 12
    th1 = 0.3 * 0.004 * 0.038 / (e * inertia)
    th2 = th1 - 0.15 * 0.004 * 0.036 / (e * inertia)
    zz = np.array([0, 38, 42, 78, 82, 120.0])
    ax2.plot(zz, np.degrees([0, th1, th1, th2, th2, th2]), "--", color="k", lw=1.5, label="linear piecewise Euler–Bernoulli")
    ax2.plot(z, th, "o", color="C0", ms=3.5, label="rod")
    try:
        from softsim.backends.fem import FemBackend
        from softsim.fem.solid3d import Solid3DSolver

        fem = FemBackend()
        fem.build(arm_test._arm(["body.mesh.size=2mm"]).build_scene())
        sol = Solid3DSolver(fem._problem, dynamic=False)
        sol.tensions[:] = [arm_test.TA, arm_test.TB]
        assert sol.advance(1.0)
        nodes = sol.p.mesh.nodes
        zr, tr = tilts(nodes + sol.y[:3 * sol.n].reshape(-1, 3), nodes)
        ax2.plot(zr, tr, "-", color="C3", lw=1.8, label="3D reference (2 mm quadratic tets)")
    except Exception as exc:  # noqa: BLE001 - the figure is still useful without the reference
        print("reference skipped:", exc)
    ax2.axvspan(38, 42, color="0.88"), ax2.axvspan(78, 82, color="0.88")
    ax2.set(xlabel="height z [mm] (grey: stiff disks)", ylabel="section tilt [deg]",
            title="Two-segment cable arm: the S (rod = linear E-B;\nreference ~8% lower: compression stiffening)")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "rod_backend.png", dpi=130)
    print("wrote", OUT / "rod_backend.png")


if __name__ == "__main__":
    main()

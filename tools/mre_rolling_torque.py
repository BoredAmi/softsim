"""The torque that rolls an MRE strip onto Bernat et al.'s ring magnet: softsim's soft-magnetic body against their ANSYS.

    python tools/mre_rolling_torque.py                    # ~10 s on a GPU: table for 2, 3 and 4 mm strips
    python tools/mre_rolling_torque.py --size 0.5e-3      # the same at half the element size (moves it 0.3%)
    python tools/mre_rolling_torque.py --plot docs/media/mre_rolling_torque.png   # + torque against angle, ~2 min

The strip (60 x 15 mm, ``t`` thick) is wrapped round the magnet's rim by a prescribed map (:func:`rolled`): its free end's
first ``(R + t / 2) theta`` on the rim, the rest leaving along the tangent, bent about its mid-surface as a rolled strip
is. The energy is the solver's own (``Solid3DSolver.magnetic_energy``, the code a simulation runs) in the ring magnet's
exact field, and the torque is the coenergy's slope ``T = -dE/dtheta``: for the table a straight-line fit over 20-250 deg,
where the ANSYS curve (datasets/bernat2024_mre_gripper.yaml, Fig. 3b) is flat at 5.15 mN m on average; for the plot,
central differences at each angle. Once the whole strip is on the rim, at ``L / (R + t / 2)``, rolling further changes
nothing and the torque is zero: 312, 299 and 286 deg for 2, 3 and 4 mm.

The susceptibility is not in the paper. The estimate here comes from the recipe alone: iron (7.87 g/cm^3) and silicone
(1.07-1.24 g/cm^3) at 1 : 1 by weight is 12-14% iron by volume, and Maxwell Garnett's rule makes that chi = 0.41-0.47
(softsim.materials.models.composite_susceptibility); each axis then takes the strip's own demagnetising factor. The torque
is nearly linear in chi, so the table also gives the chi that would match ANSYS.

What this leaves out: the strip's own field (ANSYS solves it; here the magnet's field is applied as if the strip were not
there, first order in chi), saturation (at the rim the strip reaches about half the composite's saturation
magnetisation), the brass screw in the hole, and the strip's shape off the rim (a straight tail here; unknown in ANSYS).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softsim.fem.magnetic import composite_susceptibility, field_parameters, ring, strip_magnet  # noqa: E402
from softsim.fem.materials import NeoHookeanIso  # noqa: E402
from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, Solid3DSolver  # noqa: E402
from softsim.meshing import structured_box  # noqa: E402

LENGTH, WIDTH = 0.060, 0.015
R_OUT, R_IN, THICK, BR = 0.010, 0.0021, 0.005, 1.23  # m, m, m, T: Table 1 (remanence: the middle of 1.21-1.25)
IRON, SILICONE = 7.87, (1.07, 1.24)  # g/cm^3
FIT = np.radians(np.arange(20, 251, 23))  # the flat part of the ANSYS curve
INK, MUTED, GRID = "#17262e", "#566a75", "#c3ced4"
COLOURS = {2e-3: "#1c8c8a", 3e-3: "#c8571f", 4e-3: "#6b4fa0"}


def volume_fraction(silicone: float) -> float:
    """Iron by volume at 1 : 1 by weight."""
    return (1 / IRON) / (1 / IRON + 1 / silicone)


def recipe_chis() -> list[float]:
    return [composite_susceptibility(volume_fraction(d)) for d in SILICONE]


def rolled(nodes: np.ndarray, theta: float, thickness: float) -> np.ndarray:
    """Displacements wrapping the strip (rest: x along its length from the free end at -L/2, y across, z through the
    thickness from 0) onto the rim of a magnet at the origin with its axis along y, bent about its mid-surface (which
    keeps its length, so the volume does too): the free end's first ``(R + t / 2) theta`` on the rim, going round
    clockwise seen from +y from the point (R, 0, 0), where the rest leaves straight up (+z)."""
    s = nodes[:, 0] + 0.5 * LENGTH
    zeta = nodes[:, 2]
    mid = R_OUT + 0.5 * thickness
    u = mid * theta - s  # mid-surface arc length from the take-off point back towards the free end
    on = u > 0
    r = R_OUT + zeta
    x = np.where(on, r * np.cos(u / mid), r)
    z = np.where(on, -r * np.sin(u / mid), -u)
    return np.column_stack([x, nodes[:, 1], z]) - nodes


class Strip:
    """A meshed strip of ``thickness`` and composite ``chi`` on the ring magnet; :meth:`energy` at a rolled angle. ``width``
    and the magnet's ``edge`` (softsim.magnetism; 0: the sharp, exact field) default to Bernat et al.'s strip, sharp."""

    def __init__(self, thickness: float, chi: float, size: float, width: float = WIDTH, edge: float = 0.0):
        lin = structured_box((LENGTH, width, thickness), size)
        self.mesh = QuadraticTetMesh.from_linear(lin.points, lin.tets)
        self.thickness = thickness
        problem = Problem3D(self.mesh, NeoHookeanIso(1e5), 1e7, 2000.0, gravity=(0, 0, 0),
                            magnetic=strip_magnet(chi, LENGTH, thickness, width))
        self.solver = Solid3DSolver(problem, dynamic=False)
        self.solver.magnetic_field = field_parameters(cylinders=ring((0, 0, 0), (0, 1, 0), BR, R_OUT, R_IN, THICK, edge))

    def energy(self, theta: float) -> float:
        s = self.solver
        return s.magnetic_energy(np.concatenate([rolled(self.mesh.nodes, theta, self.thickness).ravel(),
                                                 np.zeros(s.ndof - 3 * s.n)]))

    def torque(self) -> float:
        """Coenergy slope [N m] over :data:`FIT`."""
        return float(-np.polyfit(FIT, [self.energy(a) for a in FIT], 1)[0])

    def curve(self, degrees: np.ndarray, step: float = np.radians(1.0)) -> np.ndarray:
        """Torque [N m] at each angle, by central differences."""
        a = np.radians(degrees)
        return np.array([-(self.energy(x + step) - self.energy(x - step)) / (2 * step) for x in a])


def plot(path: str, size: float, angle: np.ndarray, ansys: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chis = recipe_chis()
    degrees = np.arange(5.0, 331.0, 5.0)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for t, colour in COLOURS.items():
        lo, hi = (Strip(t, c, size).curve(degrees) * 1e3 for c in chis)
        ax.fill_between(degrees, lo, hi, color=colour, alpha=0.25, linewidth=0)
        ax.plot(degrees, 0.5 * (lo + hi), color=colour, linewidth=2, label=f"softsim, {t * 1e3:g} mm strip")
        ax.axvline(np.degrees(LENGTH / (R_OUT + t / 2)), color=colour, linewidth=1, linestyle=":")
        print(f"  plotted {t * 1e3:g} mm", flush=True)
    ax.plot(angle, ansys, color=INK, linewidth=2.2, label="Bernat et al., ANSYS (Fig. 3b)")
    ax.set_xlabel("angle rolled onto the magnet [deg]", color=INK)
    ax.set_ylabel("rolling torque [mN m]", color=INK)
    ax.set_xlim(0, 330)
    ax.set_ylim(bottom=-0.2)
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Strip rolling onto a ring magnet: bands are chi {chis[0]:.2f}-{chis[1]:.2f} from the recipe alone; "
                 "dotted: strip fully rolled", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=float, default=1e-3, help="element size [m]")
    ap.add_argument("--thickness", type=float, nargs="+", default=[2e-3, 3e-3, 4e-3])
    ap.add_argument("--plot", help="also plot torque against angle for 2, 3 and 4 mm to this file")
    args = ap.parse_args()
    data = yaml.safe_load((ROOT / "datasets/bernat2024_mre_gripper.yaml").read_text())["rolling_torque"]
    angle, ansys = np.array(data["angle"]), np.array(data["torque"])
    plateau = ansys[(angle >= 20) & (angle <= 250)].mean() * 1e-3
    chis = recipe_chis()
    print(f"ANSYS (Fig. 3b), 20-250 deg: {plateau * 1e3:.2f} mN m; chi from the recipe: {chis[0]:.2f}-{chis[1]:.2f}; "
          f"mesh {args.size * 1e3:g} mm")
    print(f"| Strip | Torque, chi {chis[0]:.2f} | Torque, chi {chis[1]:.2f} | Ratio to ANSYS | chi matching ANSYS | Fully rolled |")
    print("| --- | --- | --- | --- | --- | --- |")
    for t in args.thickness:
        lo, hi = (Strip(t, c, args.size).torque() for c in chis)
        # nearly linear in chi (the demagnetising factors bend it): a secant through the two, then one more
        want = chis[0] + (plateau - lo) * (chis[1] - chis[0]) / (hi - lo)
        want += (plateau - Strip(t, want, args.size).torque()) * (chis[1] - chis[0]) / (hi - lo)
        print(f"| {t * 1e3:g} mm | {lo * 1e3:.2f} mN m | {hi * 1e3:.2f} mN m | {lo / plateau:.2f}-{hi / plateau:.2f} | "
              f"{want:.2f} | {np.degrees(LENGTH / (R_OUT + t / 2)):.0f} deg |", flush=True)
    if args.plot:
        plot(args.plot, args.size, angle, ansys)


if __name__ == "__main__":
    main()

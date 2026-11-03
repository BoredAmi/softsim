"""Animations of the actuation scenarios (docs/media/actuation_*.gif): see the bodies move.

    python tools/make_actuation_media.py                 # every case (MuJoCo ~15 min, reference ~15 min)
    python tools/make_actuation_media.py --only mount    # one case
    python tools/make_actuation_media.py --rerender      # redraw from the saved runs

Each case runs the scenario with vertex recording (``Simulation.run(record_vertices=True)``)
and draws the body's boundary from the positions softsim reports (world frame: a moving mount's
translation included), shaded, with its rest shape as a grey outline. Where a case lists two
backends they are drawn side by side at the same times. Runs are kept in runs/media/<case>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from PIL import Image  # noqa: E402

from softsim.runner import Simulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "scenarios" / "actuation"
RUNS = ROOT / "runs" / "media"
MEDIA = ROOT / "docs" / "media"

# name -> (config, title, [(label, backend, overrides)], value column, value unit/scale, view (elev, azim))
CASES = {
    "pneunet": (ACT / "pneunet_from_step.yaml", "PneuNet finger from a STEP file, 0-8 kPa",
                [("MuJoCo", "mujoco", [])], "cavity_pressure", ("kPa", 1e-3), (12, -80)),
    "stroke": (ACT / "cable_cantilever_stroke.yaml", "Cable reeled in by stroke, 0-3 mm",
               [("reference (fem)", "fem", ["simulation.timestep=2ms"]), ("MuJoCo", "mujoco", [])],
               "cable_command", ("mm", 1e3), (5, -90)),
    "arm": (ACT / "cable_arm.yaml", "Two-segment cable arm: A 0.45 N, B 0.15 N (S-shape)",
            [("reference (fem)", "fem", ["simulation.timestep=2ms", "body.mesh.size=3mm"]),
             ("rod (BeamAdapter)", "rod", ["simulation.timestep=1ms", "body.mesh.size=3mm"])],
            None, None, (5, -90)),
    "mount": (ACT / "moving_mount.yaml", "Moving mount: base slides 20 mm in 200 ms",
              [("reference (fem)", "fem", ["simulation.timestep=2ms"]), ("MuJoCo", "mujoco", [])],
              None, None, (5, -90)),
}
FRAME_DT = 0.01  # s between frames
# cases shot as a turntable: the camera swings 40 degrees while the motion plays, then circles the final pose once
ORBIT = {"pneunet": (22, -110), "arm": (22, -120)}  # elev, azim at the start
ORBIT_SWING, ORBIT_HOLD = 40.0, 72  # degrees during the motion, frames of the full turn after it


def run_case(name: str, rerender: bool) -> list[tuple[str, np.ndarray, np.ndarray, dict]]:
    config, _, runs, column, _, _ = CASES[name]
    out = []
    for label, backend, overrides in runs:
        path = RUNS / name / f"{backend}.npz"
        if not (rerender and path.exists()):
            sim = Simulation(config, ["output.save_model=false", *overrides], backend=backend)
            result = sim.run(record_vertices=True)
            ts = result.timeseries
            body = sim.scene.soft_bodies[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, t=ts["t"].to_numpy(), x=result.vertices[body.name],
                                tris=body.mesh.surface_triangles(), rest=body.mesh.points,
                                value=ts[column].to_numpy() if column else np.zeros(len(ts)),
                                bend=ts["bend_angle"].to_numpy() if "bend_angle" in ts else np.zeros(len(ts)))
            print(f"{name}/{backend}: {result.metadata['wall_time_s']:.0f} s", flush=True)
        d = np.load(path)
        out.append((label, d["t"], d["x"], {k: d[k] for k in ("tris", "rest", "value", "bend")}))
    return out


def _shade(tri: np.ndarray, rot: np.ndarray, base=(0.45, 0.72, 0.55)) -> np.ndarray:
    """Face colours lit from the camera's upper left, so the body stays lit as the camera orbits."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    light = np.array([-0.4, 0.5, 0.8]) / np.linalg.norm([-0.4, 0.5, 0.8])  # camera frame: right, up, towards
    k = 0.35 + 0.65 * np.clip((n @ rot.T) @ light, 0, 1)
    return np.clip(np.asarray(base)[None, :] * k[:, None], 0, 1)


def _view(elev: float, azim: float) -> np.ndarray:
    """Rows: screen right, screen up, towards the viewer (orthographic camera)."""
    e, a = np.radians(elev), np.radians(azim)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.cross([0.0, 0.0, 1.0], toward)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(toward, right), toward])


def _draw(ax, tri: np.ndarray, rot: np.ndarray, faces, edges, lw: float = 0.0) -> None:
    """Triangles (K, 3, 3) projected by ``rot``, back faces culled, far ones first (painter's)."""
    p = tri @ rot.T
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    keep = n > 0
    order = np.argsort(p[keep][:, :, 2].mean(axis=1))
    colors = faces[keep][order] if isinstance(faces, np.ndarray) else faces
    ax.add_collection(PolyCollection(p[keep][order][:, :, :2], facecolors=colors, edgecolors=edges, linewidths=lw))


def render(name: str, runs) -> Path:
    _, title, _, _, unit, (elev, azim) = CASES[name]
    orbit = ORBIT.get(name)
    if orbit:
        elev, azim = orbit
    t = runs[0][1]
    frames_t = np.arange(t[0], t[-1] + 1e-9, FRAME_DT)
    n_move = len(frames_t)
    azims = [azim + ORBIT_SWING * k / max(n_move - 1, 1) for k in range(n_move)] if orbit else [azim] * n_move
    if orbit:  # then one full turn on the final pose
        frames_t = np.concatenate([frames_t, np.full(ORBIT_HOLD, frames_t[-1])])
        azims += [azims[-1] + 360.0 * k / ORBIT_HOLD for k in range(ORBIT_HOLD)]
    pts = np.concatenate([r[2].reshape(-1, 3) for r in runs] + [r[3]["rest"] for r in runs])
    spans = np.array([[(pts @ _view(elev, a).T).min(axis=0), (pts @ _view(elev, a).T).max(axis=0)]
                      for a in (azims if not orbit else np.arange(0.0, 360.0, 30.0))])
    lo, hi = spans[:, 0].min(axis=0), spans[:, 1].max(axis=0)
    pad = 0.06 * (hi - lo).max()
    images = []
    for ft, az in zip(frames_t, azims):
        rot = _view(elev, az)
        fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 4.0), dpi=90, squeeze=False)
        for ax, (label, times, x, extra) in zip(axes[0], runs):
            i = int(np.argmin(np.abs(times - ft)))
            tris = extra["tris"]
            _draw(ax, extra["rest"][tris], rot, "none", (0.55, 0.55, 0.55, 0.25), lw=0.3)
            cur = x[i][tris]
            _draw(ax, cur, rot, _shade(cur, rot), "none")
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_aspect("equal")
            ax.set_axis_off()
            text = f"{label}\nt = {1e3 * times[i]:.0f} ms"
            if unit:
                text += f"   {extra['value'][i] * unit[1]:.2f} {unit[0]}"
            if np.any(extra["bend"]):
                text += f"   tip {np.degrees(extra['bend'][i]):.1f}°"
            ax.set_title(text, fontsize=9)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / f"actuation_{name}.gif"
    images = [im.quantize(colors=96) for im in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=list(CASES))
    ap.add_argument("--rerender", action="store_true", help="redraw from saved runs where they exist")
    args = ap.parse_args()
    for name in [args.only] if args.only else CASES:
        print("wrote", render(name, run_case(name, args.rerender)), flush=True)


if __name__ == "__main__":
    main()

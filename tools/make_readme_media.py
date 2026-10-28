"""Generate the README GIFs in docs/media/.

    python tools/make_readme_media.py                      # both GIFs
    python tools/make_readme_media.py --skip-experiment    # simulation-only GIF
    python tools/make_readme_media.py --set specimen.material=ecoflex_00_10_drop_fit --set specimen.mesh.size=0.005

1. ``drop_20g_sim_vs_experiment.gif``: high-speed camera frames of the 20 g
   drop (Czopek et al., doi:10.5281/zenodo.23185808, CC-BY-4.0) next to the
   MuJoCo simulation, aligned at first contact. Needs ``020gram.zip`` in
   ``--dataset-dir``.
2. ``drop_all_weights.gif``: the five dataset weights simulated side by side.

The simulation uses the scenario configs plus ``--set`` overrides; the caption
states the material. If tracking results exist (``softsim track``), the 20 g
simulation uses the measured impact (speed, offset, spin) and the experiment is
aligned at the tracked contact frame. Rendering
uses MuJoCo's offscreen renderer (set MUJOCO_GL=egl/osmesa on headless hosts).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from softsim.runner import Simulation

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "scenarios" / "drop_test" / "cases"
CAMERA_FPS = 2999.3  # measured from the overlay timestamps

# Common overrides: start 15 mm above the specimen (impact speed unchanged) so
# the weight is seen arriving, and use the default mesh/timestep.
SIM_OVERRIDES = ["drop.gap=0.015", "output.save_model=false"]
TRACKS = ROOT / "data" / "processed" / "czopek2026_ecoflex_drop"


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _label(img: Image.Image, lines: list[str], size: int = 13) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font(size)
    y = 4
    for line in lines:
        draw.text((6, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += size + 3
    return img


def render_simulation(case: str, times: np.ndarray, size: int, extra: list[str] = (),
                      lookat_z: float = 0.014, distance: float = 0.12) -> tuple[list[Image.Image], float]:
    """Render frames at ``times`` (seconds relative to first contact); also return the peak indentation."""
    sim = Simulation(CASES / f"{case}.yaml", [*SIM_OVERRIDES, *extra])
    model, data = sim.backend.model, sim.backend.data
    renderer = mujoco.Renderer(model, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, lookat_z)
    cam.distance = distance
    cam.elevation = -32.0  # steep enough that the floor fills the background
    cam.azimuth = 90.0
    weight = sim.config.weight.name
    dt = sim.config.simulation.timestep

    # pass 1: find first contact
    t_contact = None
    while t_contact is None and sim.time < 0.1:
        sim.step(1)
        c = sim.state().contact_between(sim.config.specimen.name, weight)
        if c is not None and c.force[2] > 0.01:
            t_contact = sim.time
    if t_contact is None:
        raise RuntimeError(f"{case}: no contact within 100 ms")

    # pass 2: render at the requested times
    sim.reset()
    frames, peak = [], 0.0
    for t in t_contact + times:
        if t < 0:
            t = 0.0
        n = int(round((t - sim.time) / dt))
        if n > 0:
            sim.step(n)
        peak = max(peak, sim.metrics()["indentation"])
        renderer.update_scene(data, cam)
        frames.append(Image.fromarray(renderer.render()))
    renderer.close()
    sim.close()
    return frames, peak


def experiment_frames(zip_path: Path, prefix: str, contact_frame: int, times: np.ndarray, size: int,
                      crop: tuple[int, int, int, int]) -> list[Image.Image]:
    """Camera frames at ``times`` relative to ``contact_frame``, cropped (x0, y0, x1, y1) and resized."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for t in times:
            idx = contact_frame + int(round(t * CAMERA_FPS))
            name = f"{prefix}/frame_{idx:06d}.tiff"
            if name not in names:
                raise FileNotFoundError(f"{name} not in {zip_path}")
            with z.open(name) as f:
                # median filter removes sensor noise, which otherwise bloats the GIF
                img = Image.open(f).convert("RGB").crop(crop).filter(ImageFilter.MedianFilter(5))
                img = img.resize((size, size), Image.LANCZOS)
            frames.append(img)
    return frames


def with_caption(img: Image.Image, text: str, height: int = 22) -> Image.Image:
    """``img`` with a dark caption bar underneath."""
    out = Image.new("RGB", (img.width, img.height + height), (20, 20, 20))
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((6, img.height + 4), text, fill=(235, 235, 235), font=_font(12))
    return out


def save_gif(frames: list[Image.Image], path: Path, frame_ms: int, colors: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [f.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=frame_ms, loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def material_caption(extra: list[str]) -> str:
    sim = Simulation(CASES / "w020.yaml", ["output.save_model=false", *extra])
    m = sim.scene.soft_bodies[0].material
    caption = f"{m.name}: {m.elastic.model}, E={m.elastic.initial_youngs_modulus() / 1e3:.0f} kPa"
    sim.close()
    if "UNCALIBRATED" in m.source.upper():
        return caption + " (uncalibrated)"
    return caption + (" (fitted)" if m.provenance.get("fitted_by") else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    p.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "czopek2026")
    p.add_argument("--contact-frame", type=int, default=None,
                   help="020gram.zip contact frame (default: from tracking results, else 216)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="scenario override applied to every simulation (repeatable)")
    p.add_argument("--skip-experiment", action="store_true")
    args = p.parse_args()

    caption = material_caption(args.overrides)
    impact: list[str] = []
    contact_frame = args.contact_frame or 216
    if (TRACKS / "w020.yaml").exists():
        from softsim.experiments.compare import ExperimentCase

        exp_case = ExperimentCase.load(TRACKS, "w020")
        impact = exp_case.sim_overrides(gap=0.015)
        contact_frame = args.contact_frame or exp_case.meta["contact_frame"]
    frame_ms = 80  # 0.75 ms of physics per frame -> ~107x slow motion
    size = 256

    if not args.skip_experiment:
        times = np.arange(-0.009, 0.042, 0.00075)
        zip_path = args.dataset_dir / "020gram.zip"
        exp = experiment_frames(zip_path, "020gram", contact_frame, times, size, crop=(180, 300, 620, 740))
        sim, peak = render_simulation("w020", times, size, extra=[*args.overrides, *impact])
        frames = []
        for t, a, b in zip(times, exp, sim):
            _label(a, ["experiment"])
            _label(b, ["simulation"])
            canvas = Image.new("RGB", (2 * size, size))
            canvas.paste(a, (0, 0))
            canvas.paste(b, (size, 0))
            frames.append(with_caption(canvas, f"20 g, t = {t * 1e3:+5.1f} ms | {caption}"))
        save_gif(frames, args.out / "drop_20g_sim_vs_experiment.gif", frame_ms)
        print(f"simulated peak indentation 20 g: {peak * 1e3:.2f} mm")

    times_all = np.arange(-0.006, 0.05, 0.00075)
    panel = 200
    panels = []
    for case, label in [("w020", "20 g"), ("w050", "50 g"), ("w100", "100 g"), ("w200", "200 g"), ("w500", "500 g")]:
        frames, peak = render_simulation(case, times_all, panel, extra=args.overrides, lookat_z=0.026, distance=0.17)
        panels.append([_label(f, [label, f"peak {peak * 1e3:.1f} mm"], 12) for f in frames])
        print(f"{case}: peak indentation {peak * 1e3:.2f} mm")
    frames = []
    for i, t in enumerate(times_all):
        canvas = Image.new("RGB", (panel * len(panels), panel))
        for j, frames_j in enumerate(panels):
            canvas.paste(frames_j[i], (panel * j, 0))
        frames.append(with_caption(canvas, f"t = {t * 1e3:+5.1f} ms after contact | {caption}"))
    save_gif(frames, args.out / "drop_all_weights.gif", frame_ms)


if __name__ == "__main__":
    main()

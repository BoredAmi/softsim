"""Camera vs simulation, side by side, for every drop weight: docs/media/compare_wXXX.gif.

    python tools/make_comparison_media.py              # all five weights, ~10 min
    python tools/make_comparison_media.py w050 w500    # some of them

Left: the high-speed recording (Czopek et al., doi:10.5281/zenodo.23185808,
CC-BY-4.0; needs the ``*gram.zip`` files in data/raw/czopek2026/), aligned at
the tracked contact frame. Right: MuJoCo with the whole-bounce model of the
bounce validation (first-contact fit + fitted tack and compliant support,
calibrations/ecoflex_00_10_drop_adhesion_support.yaml) and the recording's
measured impact (speed, offset, spin, in-plane tilt). The simulated camera
roughly matches the real one in scale and view angle.
About 100x slow motion through the impact, then 2 ms per frame.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_readme_media import (  # noqa: E402
    _label,
    _times,
    experiment_frames,
    render_simulation,
    save_gif,
    with_caption,
)

from softsim.experiments.calibrate import Calibration, best_overrides  # noqa: E402
from softsim.experiments.compare import ExperimentCase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "calibrations" / "ecoflex_00_10_drop_adhesion_support.yaml"
RAW = ROOT / "data" / "raw" / "czopek2026"
OUT = ROOT / "docs" / "media"
CASES = {"w020": "20 g", "w050": "50 g", "w100": "100 g", "w200": "200 g", "w500": "500 g"}
CROP = (60, 90, 740, 770)            # 680 px square (~128 mm): tall weights, hops and the specimen base
CAMERA = {"lookat_z": 0.045, "distance": 0.155, "elevation": -10.0}
SIZE = 300
END = 0.15                           # s after contact


def one(case: str, model: list[str]) -> Path:
    cal = Calibration(FIT)
    exp = ExperimentCase.load(cal.tracks, case)
    w = case[1:]
    frame0 = exp.meta["contact_frame"]
    with zipfile.ZipFile(RAW / f"{w}gram.zip") as z:
        n_frames = sum(1 for name in z.namelist() if name.endswith(".tiff"))
    end = min(END, (n_frames - frame0 - 2) / 2999.3)
    times = _times(-0.006, 0.015, end)
    cam = experiment_frames(RAW / f"{w}gram.zip", f"{w}gram", frame0, times, SIZE, CROP)
    sim, peak = render_simulation(case, times, SIZE, extra=[*model, *exp.sim_overrides(gap=0.015)], **CAMERA)
    frames = []
    for t, a, b in zip(times, cam, sim):
        _label(a, ["camera"])
        _label(b, ["MuJoCo: fit + tack + support"])
        canvas = Image.new("RGB", (2 * SIZE, SIZE))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (SIZE, 0))
        frames.append(with_caption(canvas, f"{CASES[case]}, t = {t * 1e3:+6.1f} ms after contact"))
    path = OUT / f"compare_{case}.gif"
    save_gif(frames, path, 80)
    print(f"{case}: simulated peak indentation {peak * 1e3:.2f} mm")
    return path


def main() -> None:
    cal = Calibration(FIT)
    model = [*cal.overrides, *best_overrides(yaml.safe_load((cal.output / "report.yaml").read_text()))]
    model = [o for o in model if not o.startswith(("simulation.duration", "output."))] + ["output.save_model=false"]
    for case in sys.argv[1:] or list(CASES):
        one(case, model)


if __name__ == "__main__":
    main()

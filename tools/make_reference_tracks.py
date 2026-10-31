"""Reference runs of a physical card, saved as tracks (for transferring the card to MuJoCo/SOFA).

    python tools/make_reference_tracks.py            # ~10 min on a laptop (2 cases, 3D reference)
    python tools/make_reference_tracks.py --calibration calibrations/ecoflex_00_10_drop_fem_joint.yaml --force

Runs the 20 and 50 g drops on the locking-free 3D reference (``--backend fem``)
with the physical Ecoflex 00-10 card and each recording's measured impact, and
writes them to runs/reference_tracks/ecoflex_00_10_physical/ in the tracked
format. ``calibrations/transfer_ecoflex_00_10_mujoco.yaml`` then fits MuJoCo to
them. Existing tracks are reused unless ``--force``.

``--calibration`` takes the card from a finished 3D calibration instead: its
overrides plus its best values (``<output>/report.yaml``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ("{name: ecoflex_00_10_physical, density: 1040, elastic: {model: neo_hookean, youngs_modulus: 143.3 kPa, "
        "poisson_ratio: 0.4995}, damping: {model: prony, g: [0.691], tau: [2 ms]}, friction: {sliding: 1.0}}")
OVERRIDES = ["backend=fem", f"specimen.material={CARD}", "specimen.mesh.size=0.005", "simulation.timestep=5e-5",
             "simulation.duration=0.011", "simulation.output_rate=5000", "output.save_model=false"]
OUT = ROOT / "runs" / "reference_tracks" / "ecoflex_00_10_physical"


def main() -> None:
    import yaml

    from softsim.experiments.calibrate import Calibration, best_overrides
    from softsim.experiments.datasets import load_dataset
    from softsim.experiments.reference_tracks import make_reference_tracks

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibration", help="take the card from this finished fem calibration")
    ap.add_argument("--out", type=Path, default=OUT, help=f"track directory (default {OUT.relative_to(ROOT)})")
    ap.add_argument("--force", action="store_true", help="re-run cases whose tracks exist")
    args = ap.parse_args()
    overrides, description = OVERRIDES, f"3D reference (fem backend), card {CARD}"
    if args.calibration:
        cal = Calibration(args.calibration)
        if dict(o.split("=", 1) for o in cal.overrides if "=" in o).get("backend") != "fem":
            raise SystemExit(f"{args.calibration} is not a calibration on the fem backend")
        report = yaml.safe_load((cal.output / "report.yaml").read_text())
        overrides = [*cal.overrides, *best_overrides(report), "simulation.output_rate=5000"]
        description = f"3D reference (fem backend), best card of {args.calibration} (cost {report['cost']:.4f})"
    ds = load_dataset(ROOT / "datasets" / "czopek2026_ecoflex_drop.yaml")
    for p in make_reference_tracks(ds, ROOT / "data" / "processed" / "czopek2026_ecoflex_drop", ["w020", "w050"],
                                   overrides, args.out, description, reuse=not args.force):
        print(p)


if __name__ == "__main__":
    main()

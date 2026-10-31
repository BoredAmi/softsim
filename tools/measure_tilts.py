"""Add the in-plane tilt at contact to tracked drop-test cases (without re-tracking).

    python tools/measure_tilts.py            # all cases of datasets/czopek2026_ecoflex_drop.yaml

Writes ``tilt_at_contact`` into each case's YAML in the tracks directory; only
``reliable`` tilts (clean, consistent side-edge fits: the dark weights) are
used by ``ExperimentCase.sim_overrides``. ``softsim track`` measures it too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "czopek2026_ecoflex_drop.yaml"
TRACKS = ROOT / "data" / "processed" / "czopek2026_ecoflex_drop"


def main() -> None:
    from softsim.experiments.datasets import load_dataset
    from softsim.experiments.video.tilt import measure_tilt

    ds = load_dataset(DATASET)
    for case in sys.argv[1:] or list(ds.cases):
        path = TRACKS / f"{case}.yaml"
        meta = yaml.safe_load(path.read_text())
        rot = pd.read_parquet(TRACKS / f"{case}.parquet").set_index("frame").rotation_deg
        contact = meta["contact_frame"]
        try:
            tilt = measure_tilt(ds, case, contact, rotation_since=lambda f: rot.get(contact, np.nan) - rot.get(f, np.nan))
        except ValueError as exc:
            tilt = {"reliable": False, "error": str(exc)}
        meta["tilt_at_contact"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tilt.items()}
        path.write_text(yaml.safe_dump(meta, sort_keys=False))
        print(case, meta["tilt_at_contact"])


if __name__ == "__main__":
    main()

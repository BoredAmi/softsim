"""Reference simulations written as tracks: the bridge from a physical card to backend parameters.

A run of the locking-free reference (backend ``fem``) with a physical material
card and a recording's measured impact is saved in the tracked-experiment format
(``<case>.parquet`` + ``<case>.yaml``, :class:`~softsim.experiments.compare.ExperimentCase`),
with the same contact rule and the recording's metadata. A normal calibration
against such tracks then finds the parameters with which another backend (MuJoCo,
SOFA) at a given mesh reproduces the reference. That is the backend's
*transfer* of the physical card, and its remaining error is the backend's
error bar.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from softsim.experiments.compare import ExperimentCase, simulated_indentation


def write_reference_track(timeseries: pd.DataFrame, experiment: ExperimentCase, out_dir: str | Path,
                          description: str) -> Path:
    """Save a reference run as the track of ``experiment.name`` in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curves = simulated_indentation(timeseries)
    curves.to_parquet(out / f"{experiment.name}.parquet")
    meta = {**experiment.meta, "source": description, "reference_of": experiment.name}
    (out / f"{experiment.name}.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return out / f"{experiment.name}.parquet"


def make_reference_tracks(dataset, tracks: str | Path, cases: list[str], overrides: list[str], out_dir: str | Path,
                          description: str, reuse: bool = True) -> list[Path]:
    """Run each case with ``overrides`` (e.g. ``backend=fem`` and a physical card) and its measured impact,
    and save the runs as tracks. Existing tracks in ``out_dir`` are kept when ``reuse``."""
    from softsim.runner import run_config

    paths = []
    for case in cases:
        path = Path(out_dir) / f"{case}.parquet"
        if reuse and path.exists():
            paths.append(path)
            continue
        exp = ExperimentCase.load(tracks, case)
        ts = run_config(dataset.config_path(case), [*overrides, *exp.sim_overrides()]).timeseries
        paths.append(write_reference_track(ts, exp, out_dir, description))
    return paths

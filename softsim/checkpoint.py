"""Checkpoints: a long run saves itself as it goes, and the same command goes on from where it stopped.

A reference grasp takes days. Without this a power cut loses all of it, and a run that stops short leaves nothing. A file
holds the solver's state (:meth:`softsim.backends.Backend.checkpoint`), the samples recorded so far, and a fingerprint of
what was being run, so it cannot be continued as a different problem by accident. The fingerprint covers the config and
the mesh, but not the ``output`` section or ``simulation.duration``: a finished run can be extended.

Writes are atomic, a new file replacing the old in one rename, so stopping at any moment leaves a whole checkpoint. The
format is a plain ``.npz`` with no pickles, so a checkpoint from someone else can be read without running their code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from softsim.errors import ConfigError

FORMAT = 1


@dataclass
class Saved:
    """One checkpoint. ``sample``: the last output sample recorded; ``wall_time`` [s]: spent so far, over all sessions;
    ``rows``: the metric rows; ``vertices``: per body, (samples, V, 3) if recorded; ``state``: the backend's arrays."""

    fingerprint: str
    sample: int
    wall_time: float
    rows: list[dict]
    vertices: dict[str, np.ndarray]
    state: dict[str, np.ndarray]


def fingerprint(config, scene, vertices: bool) -> str:
    """What makes two runs the same run: the config without output settings and duration, every mesh, and whether
    vertices are being recorded (rows without them cannot be finished with them)."""
    raw = config.model_dump(mode="json", exclude={"output"})
    raw["simulation"].pop("duration", None)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode())
    digest.update(str(bool(vertices)).encode())
    for body in scene.soft_bodies:
        digest.update(np.ascontiguousarray(body.mesh.points).tobytes())
        digest.update(np.ascontiguousarray(body.mesh.tets).tobytes())
    return digest.hexdigest()


def write_checkpoint(path: Path, saved: Saved) -> None:
    """Save ``saved`` at ``path``, whole or not at all: to a temporary file, flushed to disk, then renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = dict(format=FORMAT, fingerprint=saved.fingerprint, sample=saved.sample, wall_time=saved.wall_time, rows=saved.rows)
    arrays = {f"state/{k}": v for k, v in saved.state.items()} | {f"vertices/{k}": v for k, v in saved.vertices.items()}
    temporary = path.with_name(path.name + ".writing")
    with open(temporary, "wb") as f:
        np.savez(f, head=np.array(json.dumps(head, default=float)), **arrays)  # JSON keeps every double exactly
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_checkpoint(path: Path, expected: str) -> Saved:
    """Read a checkpoint, refusing one written by another format or for a different run (``expected``: its fingerprint)."""
    with np.load(path, allow_pickle=False) as d:
        head = json.loads(str(d["head"]))
        if head.get("format") != FORMAT:
            raise ConfigError(f"{path}: checkpoint format {head.get('format')}, this version reads {FORMAT}")
        if head["fingerprint"] != expected:
            raise ConfigError(f"{path} is a checkpoint of a different run: the config or the mesh has changed. "
                              "Delete it to start again, or restore what changed.")
        pick = lambda prefix: {k.removeprefix(prefix): d[k] for k in d.files if k.startswith(prefix)}
        return Saved(head["fingerprint"], head["sample"], head["wall_time"], head["rows"], pick("vertices/"), pick("state/"))

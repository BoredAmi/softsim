"""A small set of example scenario configs shipped with the package (``softsim examples copy DIR``).

The repository's ``scenarios/`` directory is not part of a pip install; these are the files needed to run the first
drop test, the cable and PneuNet actuation scenarios and the Gymnasium environments built on them. The layout mirrors
``scenarios/`` (``drop_test/``, ``actuation/``, ``envs/``) so relative ``extends:`` and ``scenario:`` paths keep working.
"""

from __future__ import annotations

import shutil
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent


def example_files() -> list[Path]:
    """Paths of the packaged example configs, relative to the examples directory."""
    return sorted(p.relative_to(EXAMPLES_DIR) for p in EXAMPLES_DIR.rglob("*.yaml"))


def copy_examples(dest: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Copy the examples into ``dest`` (created if needed); return the files written.

    Existing files are left alone unless ``overwrite`` is set; a conflict raises ``FileExistsError`` before anything is
    written.
    """
    dest = Path(dest)
    files = example_files()
    clashes = [rel for rel in files if (dest / rel).exists()]
    if clashes and not overwrite:
        raise FileExistsError(f"{dest / clashes[0]} exists (and {len(clashes) - 1} more); use --overwrite")
    written = []
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXAMPLES_DIR / rel, target)
        written.append(target)
    return written

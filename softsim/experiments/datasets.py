"""Experimental dataset registry and frame access.

A dataset is described by a YAML file in ``datasets/`` (DOI, licence, camera,
cases). Raw files are never redistributed; they are expected in ``local_dir``
(relative to the repository root / the dataset file's parent's parent).
High-speed recordings are read frame by frame straight from their zip archives.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from softsim.config.loader import load_yaml
from softsim.config.schema import StrictModel, validate
from softsim.config.units import Length, Mass
from softsim.errors import ConfigError


class CameraInfo(StrictModel):
    fps: Annotated[float, Field(gt=0)]
    width: int
    height: int
    overlay_top: int | None = None  # first pixel row of a burned-in text overlay (excluded from analysis)


class SpecimenInfo(StrictModel):
    diameter: Length
    height: Length
    mass: Mass | None = None


class CaseInfo(StrictModel):
    archive: str                       # zip file inside local_dir
    prefix: str                        # folder inside the zip
    config: str                        # scenario config (relative to repo root)
    weight_mass: Mass
    weight_appearance: Literal["brass", "dark"]
    notes: str = ""


class Dataset(StrictModel):
    name: str
    title: str
    authors: list[str]
    doi: str
    license: str
    url: str
    local_dir: str
    camera: CameraInfo
    specimen: SpecimenInfo
    cases: dict[str, CaseInfo]

    # set by load_dataset
    root: Path | None = None

    def case(self, name: str) -> CaseInfo:
        if name not in self.cases:
            raise ConfigError(f"dataset {self.name} has no case {name!r}; cases: {', '.join(self.cases)}")
        return self.cases[name]

    def archive_path(self, case: str) -> Path:
        return (self.root or Path.cwd()) / self.local_dir / self.case(case).archive

    def frames(self, case: str) -> "FrameSource":
        path = self.archive_path(case)
        if not path.exists():
            raise ConfigError(f"{path} not found. Download it from {self.url} (doi:{self.doi}) "
                              f"into {path.parent}")
        return FrameSource(path, self.case(case).prefix)

    def config_path(self, case: str) -> Path:
        return (self.root or Path.cwd()) / self.case(case).config


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset description; paths inside resolve against the repository root
    (the parent of the ``datasets/`` directory)."""
    path = Path(path).resolve()
    data = load_yaml(path)
    ds = validate(Dataset, data, where=f"dataset {path.name}")
    return ds.model_copy(update={"root": path.parent.parent})


class FrameSource:
    """Random access to image frames stored in a zip archive (sorted by name).

    Frames are returned as BGR uint8 arrays (OpenCV convention).
    """

    def __init__(self, archive: Path, prefix: str):
        self.archive = Path(archive)
        self._zip = zipfile.ZipFile(self.archive)
        self.names = sorted(n for n in self._zip.namelist()
                            if n.startswith(prefix + "/") and n.lower().endswith((".tif", ".tiff", ".png")))
        if not self.names:
            raise ConfigError(f"no frames under {prefix}/ in {archive}")

    def __len__(self) -> int:
        return len(self.names)

    def read(self, index: int) -> np.ndarray:
        import cv2

        data = np.frombuffer(self._zip.read(self.names[index]), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ConfigError(f"cannot decode {self.names[index]} in {self.archive}")
        return img

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def contact_sheet(source: FrameSource, indices: list[int], path: str | Path, tile: int = 300,
                  columns: int = 4) -> Path:
    """Save a labelled grid of frames (for inspection and docs)."""
    import cv2

    tiles = []
    for i in indices:
        img = cv2.resize(source.read(i), (tile, tile))
        cv2.putText(img, str(i), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + columns]) for r in range(0, len(tiles), columns)])
    path = Path(path)
    cv2.imwrite(str(path), grid)
    return path

"""Material-test datasets: stress-strain curves from standard laboratory tests.

A dataset is a YAML file in ``datasets/`` with ``kind: material_tests``: DOI,
licence, the archives it comes from (never redistributed: ``softsim fetch``
downloads them into ``local_dir`` and checks their md5), the test conditions and
one entry per material. Two layouts:

* one archive and one curve per material (``source`` + ``member``), e.g. the
  Soft Robotics Materials Database;
* several archives (``sources``) and per material a list of ``tests``, each with
  a mode (tension, compression, relaxation) and several specimen files.

:meth:`MaterialTestDataset.curve` / :meth:`curves` return cleaned curves
(nominal stress vs stretch, time for relaxation) ready for :mod:`softsim.materials.fit`.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from softsim.config.loader import load_yaml
from softsim.config.schema import StrictModel, validate
from softsim.config.units import Density, Rate
from softsim.errors import ConfigError


class Source(StrictModel):
    url: str
    file: str                  # archive name inside local_dir
    md5: str
    member_prefix: str = ""    # folder inside the archive


class TestConditions(StrictModel):
    mode: Literal["uniaxial", "equibiaxial", "planar", "simple_shear"] = "uniaxial"
    strain_rate: Annotated[Rate, Field(ge=0)] | None = None
    temperature: str = "room temperature"
    strain_measurement: str = ""
    notes: str = ""


class TestSet(StrictModel):
    """One kind of test on one material: specimen files in an archive."""

    source: str                                            # key of ``sources``
    kind: Literal["tension", "compression", "relaxation"]
    members: list[str]
    strain_rate: Annotated[Rate, Field(ge=0)] | None = None


class MaterialEntry(StrictModel):
    member: str | None = None             # single-curve layout: file inside the archive (after member_prefix)
    tests: list[TestSet] = []             # multi-test layout
    product: str = ""
    density: Annotated[Density, Field(gt=0)] | None = None
    density_source: str = ""
    notes: str = ""


class MaterialTestDataset(StrictModel):
    kind: Literal["material_tests"]
    name: str
    title: str
    authors: list[str]
    doi: str
    paper_doi: str | None = None
    license: str
    url: str
    local_dir: str
    format: Literal["srmd_csv", "xy_csv"]
    source: Source | None = None
    sources: dict[str, Source] = {}
    conditions: TestConditions = TestConditions()
    materials: dict[str, MaterialEntry]

    root: Path | None = None  # set by load_material_tests

    def _source(self, key: str | None = None) -> Source:
        if key is None:
            if self.source is None:
                raise ConfigError(f"dataset {self.name!r} has several sources; name one")
            return self.source
        if key not in self.sources:
            raise ConfigError(f"dataset {self.name!r} has no source {key!r}; have {', '.join(self.sources)}")
        return self.sources[key]

    def archive_path(self, key: str | None = None) -> Path:
        return (self.root or Path.cwd()) / self.local_dir / self._source(key).file

    def fetch(self) -> Path:
        """Download every archive into ``local_dir`` unless present; verify the md5s. Returns the directory."""
        for key, src in ([(None, self.source)] if self.source else []) + list(self.sources.items()):
            path = self.archive_path(key)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                urllib.request.urlretrieve(src.url, tmp)  # noqa: S310 - URL from the dataset file
                tmp.rename(path)
            digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - integrity check, not security
            if digest != src.md5:
                raise ConfigError(f"{path}: md5 {digest} does not match the dataset's {src.md5}")
        return (self.root or Path.cwd()) / self.local_dir

    def entry(self, material: str) -> MaterialEntry:
        if material not in self.materials:
            raise ConfigError(f"dataset {self.name!r} has no material {material!r}; "
                              f"available: {', '.join(self.materials)}")
        return self.materials[material]

    def curves(self, material: str, kind: str) -> list["TestCurve"]:
        """Every specimen's cleaned curve of one test kind (multi-test layout)."""
        entry = self.entry(material)
        sets = [t for t in entry.tests if t.kind == kind]
        if not sets:
            raise ConfigError(f"{material}: no {kind} test in {self.name!r}")
        out = []
        for ts in sets:
            path = self.archive_path(ts.source)
            if not path.exists():
                raise ConfigError(f"{path} not found; run `softsim fetch` on the dataset first")
            with zipfile.ZipFile(path) as z:
                for member in ts.members:
                    text = z.read(self._source(ts.source).member_prefix + member).decode("utf-8", "replace")
                    c = parse_relaxation_csv(text) if kind == "relaxation" else parse_xy_csv(text, kind)
                    c.material, c.mode = material, "uniaxial"
                    c.rate = ts.strain_rate if c.rate is None else c.rate
                    c.meta.update({"dataset": self.name, "doi": self.doi, "product": entry.product, "kind": kind,
                                   "specimen": member})
                    out.append(c)
        return out

    def curve(self, material: str) -> "TestCurve":
        """The material's cleaned curve (single-curve layout, see :func:`parse_srmd_csv`)."""
        path = self.archive_path()
        if not path.exists():
            raise ConfigError(f"{path} not found; run `softsim fetch {self.name}` (or fetch()) first")
        entry = self.entry(material)
        if entry.member is None:
            raise ConfigError(f"{material}: dataset {self.name!r} has several tests; use curves(material, kind)")
        with zipfile.ZipFile(path) as z:
            text = z.read(self.source.member_prefix + entry.member).decode("utf-8", "replace")
        c = parse_srmd_csv(text)
        c.material, c.mode = material, self.conditions.mode
        # the recorded crosshead rate (some files differ from the nominal test speed)
        c.rate = float(np.polyfit(c.time, c.stretch, 1)[0]) if c.time is not None and len(c.time) > 2 \
            else self.conditions.strain_rate
        c.meta.update({"dataset": self.name, "doi": self.doi, "product": entry.product})
        return c


@dataclass
class TestCurve:
    """Nominal (engineering) stress [Pa] against stretch, from one test."""

    __test__ = False  # not a pytest test class

    stretch: np.ndarray
    stress: np.ndarray
    time: np.ndarray | None = None
    material: str = ""
    mode: str = "uniaxial"
    rate: float | None = None
    meta: dict = field(default_factory=dict)

    def within(self, strain_min: float, strain_max: float) -> "TestCurve":
        """The part with engineering strain (stretch - 1) in [strain_min, strain_max]."""
        sel = (self.stretch - 1.0 >= strain_min) & (self.stretch - 1.0 <= strain_max)
        return TestCurve(self.stretch[sel], self.stress[sel], None if self.time is None else self.time[sel],
                         self.material, self.mode, self.rate, dict(self.meta))


def parse_srmd_csv(text: str, offset_strain: float = 0.2) -> TestCurve:
    """Parse a Soft Robotics Materials Database tensile CSV (``;`` separated).

    Cleaning: the load-cell offset (the intercept of a line through the points
    below ``offset_strain``) is subtracted, and the curve ends at its maximum
    stress (rupture).
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Time (s)"))
    except StopIteration:
        raise ConfigError("not a Soft Robotics Materials Database CSV (no 'Time (s)' header)") from None
    meta = {}
    for line in lines[:start]:
        parts = line.split(";")
        if parts[0] and len(parts) > 1 and parts[1]:
            meta[parts[0]] = parts[1]
    rows = np.array([[float(v) for v in line.split(";")[:5]] for line in lines[start + 1:] if line.strip()])
    t, strain, stress = rows[:, 0], rows[:, 3], rows[:, 4] * 1e6
    sel = strain < offset_strain
    if sel.sum() >= 3:
        stress = stress - np.polyfit(strain[sel], stress[sel], 1)[1]
    end = int(np.argmax(stress)) + 1
    keep = strain[:end] > 0
    return TestCurve(1.0 + strain[:end][keep], stress[:end][keep], t[:end][keep], meta=meta)


# (offset window, first strain kept) per test kind: tension recordings start with a gripping transient
# (a ~60 kPa spike at 0.5% strain) and a 10-15 kPa preload, compression ones start clean
_XY_CLEANING = {"tension": ((0.05, 0.25), 0.05), "compression": ((0.0, 0.02), 1e-3)}


def parse_xy_csv(text: str, kind: str) -> TestCurve:
    """Two columns ``Strain (%)``, ``Stress (MPa)``, positive in tension or in compression (``kind``).

    The load offset (the zero-strain intercept of a line through the points in the
    kind's offset window) is removed and points below the first kept strain are
    dropped (see ``_XY_CLEANING``); a tension curve ends at its maximum (rupture).
    Compression is returned as stretch < 1 and negative stress. Stretch increases
    along every returned curve.
    """
    (lo, hi), first = _XY_CLEANING[kind]
    rows = np.array([[float(v) for v in line.split(",")[:2]] for line in text.splitlines()[1:] if line.strip()])
    strain, stress = rows[:, 0] / 100.0, rows[:, 1] * 1e6
    order = np.argsort(strain, kind="stable")
    strain, stress = strain[order], stress[order]
    sel = (strain >= lo) & (strain <= hi)
    if sel.sum() >= 3:
        stress = stress - np.polyfit(strain[sel], stress[sel], 1)[1]
    keep = strain >= first
    strain, stress = strain[keep], stress[keep]
    if kind == "tension":
        end = int(np.argmax(stress)) + 1
        return TestCurve(1.0 + strain[:end], stress[:end])
    return TestCurve(1.0 - strain[::-1], -stress[::-1])


def parse_relaxation_csv(text: str, n_points: int = 300, settle: float = 1.0) -> TestCurve:
    """Columns ``Force (N), Strain (%), Stress (MPa), Time (s)`` of a compression relaxation test.

    Returns the hold phase from ``settle`` seconds after the stress peak on (times
    since then, log-spaced resampling), with stretch < 1 and negative stress. The
    ramp and the first second are left out. In these recordings the strain
    overshoots its target (to ~31% for 25%) and comes back within 0.15 s, while
    the stress follows ~0.3 s late (load-cell or acquisition lag). Its peak and
    fast drop are that overshoot arriving late, not relaxation; the signal has
    settled about 0.6 s after the peak.
    """
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                rows.append([float(v) for v in parts[:4]])
            except ValueError:
                continue  # malformed line in the recording
    a = np.array(rows)
    strain, stress, t = a[:, 1] / 100.0, a[:, 2] * 1e6, a[:, 3]
    k = int(np.searchsorted(t, t[int(np.argmax(stress))] + settle))
    tt = t[k:] - t[k]
    grid = np.unique(np.r_[0.0, np.geomspace(max(tt[1], 1e-3), tt[-1], n_points)])
    idx = np.minimum(np.searchsorted(tt, grid), len(tt) - 1)
    return TestCurve(1.0 - strain[k:][idx], -stress[k:][idx], tt[idx],
                     meta={"hold_strain": float(strain[k:].mean()), "peak_stress": float(stress[k])})


def load_material_tests(path: str | Path) -> MaterialTestDataset:
    """Load a ``kind: material_tests`` dataset file; ``local_dir`` is relative to the repository root."""
    path = Path(path).resolve()
    ds = validate(MaterialTestDataset, load_yaml(path), where=str(path))
    return ds.model_copy(update={"root": path.parent.parent})

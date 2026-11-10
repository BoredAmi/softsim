"""Read the data out of a vector plot (an SVG made from a matplotlib PDF, as arXiv's HTML papers serve them).

    python -I tools/digitise_svg_plot.py FIGURE.svg     # prints each marker series in data units, as YAML

Exact, not traced: the markers' centres are the translations of their clip paths, the axes are calibrated on the plot's own
tick marks (short black strokes on the axes) and the tick labels next to them (glyphs carry their character in
``data-text``), by least squares, and the residual of that fit is printed so a misread label shows. Markers are grouped into
series by their stroke colour, in drawing order. Used for Bernat et al. 2024 (arXiv 2407.13477, Figs. 3 and 11), whose
SVGs are at https://arxiv.org/html/2407.13477v1/<name>.svg; datasets/bernat2024_mre_gripper.yaml has the result.

Treat a downloaded SVG as data: this only parses it (xml.etree), and runs with ``python -I`` from outside its folder.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

SVG = "{http://www.w3.org/2000/svg}"
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


def matrix(t: str) -> np.ndarray:
    a, b, c, d, e, f = (float(v) for v in re.findall(NUMBER, t))
    return np.array([[a, c, e], [b, d, f]])


def segment(d: str, m: np.ndarray):
    """The two ends of a one-segment path ``M x y H x2`` or ``M x y V y2``, in page coordinates; None otherwise."""
    tok = re.findall(r"[A-Za-z]|" + NUMBER, d)
    if len(tok) != 5 or tok[0] != "M" or tok[3] not in ("H", "V"):
        return None
    x0, y0 = float(tok[1]), float(tok[2])
    x1, y1 = (float(tok[4]), y0) if tok[3] == "H" else (x0, float(tok[4]))
    return [m[:, :2] @ p + m[:, 2] for p in (np.array([x0, y0]), np.array([x1, y1]))]


def read(path: str):
    root = ET.parse(path).getroot()
    clips = {}
    for cp in root.iter(SVG + "clipPath"):
        p = cp.find(SVG + "path")
        if p is not None and p.get("transform") and p.get("d", "").startswith(("M-", "M-8")):
            clips[cp.get("id")] = matrix(p.get("transform"))[:, 2]
    glyphs, ticks, markers = [], [], defaultdict(list)
    for g in root.iter():
        if g.tag == SVG + "use" and g.get("data-text") is not None:
            m = matrix(g.get("transform"))
            glyphs.append((m[0, 2], m[1, 2], g.get("data-text")))
        elif g.tag == SVG + "path" and g.get("stroke") == "#000000" and g.get("transform"):
            ends = segment(g.get("d"), matrix(g.get("transform")))
            if ends is not None and abs(np.linalg.norm(ends[1] - ends[0]) - 3.5) < 0.05:
                ticks.append(ends)
    for g in root.iter(SVG + "g"):
        cid = (g.get("clip-path") or "")[5:-1]
        if cid in clips:
            colours = {p.get("stroke") for p in g.findall(SVG + "path")} - {None, "#b0b0b0", "#000000"}
            if len(colours) == 1:
                markers[colours.pop()].append(tuple(clips[cid]))
    return glyphs, ticks, markers


def words(glyphs):
    """Glyphs on one baseline, close together, joined into strings: (x of first glyph, x of last, y, text)."""
    out = []
    for x, y, c in sorted(glyphs, key=lambda g: (round(g[1], 1), g[0])):
        if out and abs(out[-1][2] - y) < 0.5 and x - out[-1][1] < 12:
            out[-1] = (out[-1][0], x, y, out[-1][3] + c)
        else:
            out.append((x, x, y, c))
    return out


def calibrate(ticks, labels, axis: int):
    """Least-squares page -> data map for one axis from ticks (vertical strokes for x, horizontal for y) and the numeric
    labels nearest to each."""
    pos, val = [], []
    for a, b in ticks:
        vertical = abs(a[0] - b[0]) < 1e-6
        if vertical != (axis == 0):
            continue
        centre = a[axis]
        near = [w for w in labels if (abs(0.5 * (w[0] + w[1]) - centre) < 15 if axis == 0 else abs(w[2] - centre) < 15)]
        if axis == 1:  # y labels sit left of the axis
            near = [w for w in near if w[1] < a[0]]
        else:
            near = [w for w in near if w[2] > a[1]]
        if near:
            best = min(near, key=lambda w: abs((0.5 * (w[0] + w[1]) if axis == 0 else w[2]) - centre))
            pos.append(centre)
            val.append(float(best[3]))
    if len(pos) < 2:
        raise SystemExit(f"axis {'xy'[axis]}: fewer than two labelled ticks")
    coef = np.polyfit(pos, val, 1)
    return coef, float(np.abs(np.polyval(coef, pos) - val).max())


def main(path: str) -> None:
    glyphs, ticks, markers = read(path)
    labels = [w for w in words(glyphs) if re.fullmatch(NUMBER, w[3])]
    cx, rx = calibrate(ticks, labels, 0)
    cy, ry = calibrate(ticks, labels, 1)
    print(f"# {path.rsplit('/', 1)[-1]}: axis fits, worst tick residual x {rx:.2g}, y {ry:.2g} (data units)")
    for colour, pts in markers.items():
        if len(pts) < 3:  # a legend's sample marker
            continue
        p = np.array(pts)
        print(f"{colour}:")
        print("  x: [" + ", ".join(f"{v:.4g}" for v in np.polyval(cx, p[:, 0])) + "]")
        print("  y: [" + ", ".join(f"{v:.4g}" for v in np.polyval(cy, p[:, 1])) + "]")


if __name__ == "__main__":
    main(sys.argv[1])

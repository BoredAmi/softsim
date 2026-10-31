"""In-plane tilt of a falling cylindrical weight from its silhouette's side edges.

The weight's axis, projected into the image, is parallel to the two straight
side edges of its silhouette. On frames well before contact (the weight clear
of the specimen and its shadow), the left and right boundary of the body are
fitted with lines x = a + b y over the body's full-width rows (which excludes
the knob and the end faces' ellipses); the tilt is the angle of their mean
slope, positive = clockwise on screen (top towards +x), the convention of the
tracker's rotation. The rotation the tracker measured from those frames to
contact is added. The out-of-plane tilt is not observable from one side view.
"""

from __future__ import annotations

import math

import numpy as np

from softsim.experiments.datasets import Dataset
from softsim.experiments.video.segment import SegmentationParams, largest_component, reference_background, weight_mask

FULL_WIDTH = 0.9   # body rows: at least this fraction of the widest row
TRIM = 0.2         # and not the outer 20 % of those rows at either end
RELIABLE_SPREAD = 3.0     # deg over the measured frames
RELIABLE_RESIDUAL = 2.0   # px rms of the edge fits


def silhouette_tilt(mask: np.ndarray) -> tuple[float, float]:
    """(tilt [deg], rms residual of the edge fits [px]) of a weight silhouette."""
    rows = np.nonzero(mask.any(axis=1))[0]
    lefts = np.array([np.nonzero(mask[y])[0].min() for y in rows])
    rights = np.array([np.nonzero(mask[y])[0].max() for y in rows])
    widths = rights - lefts
    body = rows[widths >= FULL_WIDTH * widths.max()]
    lo, hi = body.min(), body.max()
    keep = (rows >= lo + TRIM * (hi - lo)) & (rows <= hi - TRIM * (hi - lo)) & (widths >= FULL_WIDTH * widths.max())
    used, left, right = rows[keep].astype(float), lefts[keep], rights[keep]
    if len(used) < 5:
        raise ValueError("silhouette too small for an edge fit")
    slopes, res = [], []
    for edge in (np.asarray(left, float), np.asarray(right, float)):
        b, a = np.polyfit(used, edge, 1)
        slopes.append(b)
        res.append(np.sqrt(np.mean((a + b * used - edge) ** 2)))
    return -math.degrees(math.atan(float(np.mean(slopes)))), float(np.mean(res))


def measure_tilt(dataset: Dataset, case: str, contact_frame: int, n_frames: int = 10, gap: int = 15,
                 params: SegmentationParams | None = None, reference_frames: int = 20,
                 rotation_since=None) -> dict[str, float]:
    """In-plane tilt at contact: the median over ``n_frames`` frames ending ``gap`` frames before
    contact, plus ``rotation_since(frame)`` [deg] (the tracked rotation from that frame to contact).

    Returns tilt at contact [deg], the raw tilt at the measured frames, its spread and the edge-fit
    residual [px].
    """
    from softsim.experiments.video.track import _fill_holes

    p = params or SegmentationParams()
    info = dataset.case(case)
    overlay = dataset.camera.overlay_top or dataset.camera.height
    with dataset.frames(case) as frames:
        reference = reference_background([frames.read(i)[:overlay] for i in range(0, reference_frames, 2)])
        tilts, res = [], []
        last = contact_frame - gap
        for i in range(last - n_frames + 1, last + 1):
            img = frames.read(i)[:overlay]
            w = largest_component(weight_mask(img, reference, info.weight_appearance, p), p.min_area)
            if w is None:
                continue
            try:
                t, r = silhouette_tilt(_fill_holes(w))
            except ValueError:  # no clean body rows in this frame (reflections, knob, motion blur)
                continue
            tilts.append(t)
            res.append(r)
    if not tilts:
        raise ValueError(f"{case}: weight not found before contact")
    raw = float(np.median(tilts))
    turned = float(rotation_since(last)) if rotation_since is not None else 0.0
    spread, residual = float(np.ptp(tilts)), float(np.median(res))
    return {"tilt_deg": raw + turned, "tilt_measured_deg": raw, "rotation_to_contact_deg": turned,
            "tilt_spread_deg": spread, "edge_residual_px": residual, "frames": len(tilts),
            "reliable": bool(len(tilts) >= n_frames // 2 and spread < RELIABLE_SPREAD and residual < RELIABLE_RESIDUAL)}

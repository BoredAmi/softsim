"""Track a drop-test recording: weight trajectory, contact, indentation, bulge, lift-off.

Geometry (side view, camera slightly above the table):
    * horizontal scale ``s`` [px/mm] from the specimen's resting silhouette
      width (= its diameter, independent of camera elevation);
    * camera elevation ``e`` from the silhouette height, which for a cylinder
      seen from above is ``s * (H cos e + D sin e)``;
    * vertical displacements scale with ``s_v = s cos e``.
The weight is followed as a rigid body (x, y, in-plane rotation) by incremental
ECC alignment of its upper part, away from the specimen and its shadow; its
centre of mass is the quantity compared with simulation. The camera looks down,
so a falling weight also moves away from it and perspective lowers its apparent
acceleration (~3-4 m/s^2 instead of g in these recordings). A free quadratic
is therefore fitted locally (last frames before the approach) and the first
frame departing from it marks contact (+-1 frame). The fitted impact speed is a
check of the scale against the nominal 2.73 m/s.

Image convention: x right, y down. Reported physical quantities: z up,
``indentation`` positive downward since contact (like the simulation), angles
in degrees, positive = clockwise on screen.

The table the specimen rests on is not rigid in these recordings: next to the
specimen it moves by several mm from ~20 ms after impact. ``indentation_rel``
(weight centre of mass relative to the specimen base) removes that motion and
is the quantity to compare with a rigid-ground simulation; ``specimen_lift``
is the base height and mostly shows the table motion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from softsim.experiments.datasets import Dataset
from softsim.experiments.video.segment import (
    SegmentationParams,
    largest_component,
    reference_background,
    specimen_mask,
    weight_mask,
)
from softsim.io import save_table

G = 9.81


@dataclass
class TrackingParams:
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    reference_frames: int = 20          # frames (from the start) without the weight
    flash_fraction: float = 0.15        # ignore this lowest fraction of the specimen (casting flash)
    edge_margin_px: int = 15            # weight must be this far inside the image to be "in view"
    approach_clearance_mm: float = 6.0  # free-fall fit stops this far above the specimen (shadow effects)
    min_fit_frames: int = 12
    fit_window_frames: int = 25         # free-fall fit uses only the last frames before the approach:
                                        # perspective (camera looks down) distorts longer falls
    template_fraction: float = 0.65     # part of the weight (from the top) used for tracking
    edge_dilate_px: int = 7             # include the silhouette edge (gradients) in the tracking mask
    ecc_blur: float = 1.0
    ecc_iterations: int = 200
    ecc_eps: float = 1e-8               # small: per-frame rotations are ~1e-3 rad
    min_ecc: float = 0.8                # stop tracking below this correlation
    track_after_frames: int = 300       # track this many frames past the approach
    contact_threshold_px: float = 0.5   # deviation from free fall that counts as contact
    contact_consecutive: int = 3


@dataclass
class CaseTrack:
    """Tracking result for one recording."""

    case: str
    timeseries: pd.DataFrame
    meta: dict[str, Any]

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_table(self.timeseries, directory / self.case)
        (directory / f"{self.case}.yaml").write_text(yaml.safe_dump(_plain(self.meta), sort_keys=False))
        return directory


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def _row_widths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.nonzero(mask.any(axis=1))[0]
    widths = np.array([np.ptp(np.nonzero(mask[r])[0]) + 1 for r in rows])
    return rows, widths


def specimen_geometry(mask: np.ndarray, flash_fraction: float) -> dict[str, float]:
    """Silhouette measurements of the specimen mask in pixels."""
    rows, widths = _row_widths(mask)
    top, bottom = rows.min(), rows.max()
    band = rows <= bottom - flash_fraction * (bottom - top)
    cols = np.nonzero(mask[rows[band]].any(axis=0))[0]
    return {"top": float(top), "bottom": float(bottom), "width": float(widths[band].max()),
            "center_x": float((cols.min() + cols.max()) / 2)}


def camera_elevation(width_px: float, height_px: float, diameter: float, height: float) -> float:
    """Elevation [rad] solving height_px / s = H cos e + D sin e with s = width_px / D."""
    s = width_px / diameter
    amp, phase = math.hypot(height, diameter), math.atan2(height, diameter)
    ratio = min(1.0, height_px / (s * amp))
    return max(0.0, math.asin(ratio) - phase)


def _blob_stats(mask: np.ndarray) -> dict[str, float]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    ys, xs = np.nonzero(mask)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    major = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    angle = (major + 45.0) % 90.0 - 45.0  # axis closest to horizontal (flat discs) or vertical (tall weights)
    return {"area": float(m["m00"]), "cx": cx, "cy": cy, "top": float(ys.min()), "bottom": float(ys.max()),
            "left": float(xs.min()), "right": float(xs.max()), "angle_deg": angle}


def _gray(img: np.ndarray, blur: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), blur) if blur > 0 else g


def _rotation(theta: float) -> np.ndarray:
    c, s_ = math.cos(theta), math.sin(theta)
    return np.array([[c, -s_], [s_, c]])


def track_rigid(read, weight_seg, start: int, stop: int, com0: np.ndarray, height0: float,
                p: "TrackingParams") -> dict[int, tuple[np.ndarray, float, float]]:
    """Incremental rigid (x, y, theta) tracking of the weight with ECC, frame to frame.

    ``read(i)`` returns frame i, ``weight_seg(img)`` the filled weight mask. Each
    step aligns the weight's upper part in frame i-1 (the part above a line
    ``(upper_fraction - 0.5) * height0`` below the centre of mass, rotating with
    the weight, so the specimen and its shadow are excluded) to frame i.
    Returns ``{frame: (com_xy, theta_rad, ecc_correlation)}``; theta is the
    in-plane rotation since ``start`` (positive = clockwise on screen).
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p.ecc_iterations, p.ecc_eps)
    cut = (p.template_fraction - 0.5) * height0
    out = {start: (com0.copy(), 0.0, 1.0)}
    prev_img = read(start)
    com, theta, vel = com0.copy(), 0.0, np.zeros(2)
    for i in range(start + 1, stop):
        mask = weight_seg(prev_img).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((p.edge_dilate_px, p.edge_dilate_px), np.uint8))
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        up = np.array([math.sin(theta), -math.cos(theta)])
        mask[((xx - com[0]) * up[0] + (yy - com[1]) * up[1]) < -cut] = 0
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            break
        pad = 4 + int(np.ceil(np.abs(vel).max()))
        x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
        x1, y1 = min(xs.max() + pad + 1, mask.shape[1]), min(ys.max() + pad + 1, mask.shape[0])
        template = _gray(prev_img[y0:y1, x0:x1], p.ecc_blur)
        tmask = mask[y0:y1, x0:x1]
        img = read(i)
        # same-size input crop, shifted by the predicted (integer) motion (OpenCV ECC needs equal sizes)
        sx, sy = int(round(vel[0])), int(round(vel[1]))
        if x0 + sx < 0 or y0 + sy < 0 or x1 + sx > img.shape[1] or y1 + sy > img.shape[0]:
            break
        target = _gray(img[y0 + sy:y1 + sy, x0 + sx:x1 + sx], p.ecc_blur)
        warp = np.array([[1, 0, vel[0] - sx], [0, 1, vel[1] - sy]], np.float32)
        cc = None
        for blur in (0, 3):  # retry with extra smoothing if the first attempt diverges
            try:
                tgt = cv2.GaussianBlur(target, (0, 0), blur) if blur else target
                tpl = cv2.GaussianBlur(template, (0, 0), blur) if blur else template
                cc, w = cv2.findTransformECC(tpl, tgt, warp.copy(), cv2.MOTION_EUCLIDEAN, criteria, tmask, 1)
                break
            except cv2.error:
                continue
        if cc is None or cc < p.min_ecc:
            break
        new_com = w[:, :2] @ (com - [x0, y0]) + w[:, 2] + [x0 + sx, y0 + sy]
        vel = new_com - com
        com = new_com
        theta += math.atan2(w[1, 0], w[0, 0])
        out[i] = (com.copy(), theta, float(cc))
        prev_img = img
    return out


def track_case(dataset: Dataset, case: str, params: TrackingParams | None = None) -> CaseTrack:
    """Track one case of a drop-test dataset."""
    p = params or TrackingParams()
    info = dataset.case(case)
    cam = dataset.camera
    overlay = cam.overlay_top or cam.height
    D, H = dataset.specimen.diameter * 1e3, dataset.specimen.height * 1e3  # mm
    m = p.edge_margin_px

    # --- pass 1: per-frame segmentation ------------------------------------------
    rows: list[dict[str, float]] = []
    with dataset.frames(case) as frames:
        reference = reference_background([frames.read(i)[:overlay] for i in range(0, p.reference_frames, 2)])
        rest = largest_component(specimen_mask(reference, p.segmentation), p.segmentation.min_area)
        if rest is None:
            raise RuntimeError(f"{case}: specimen not found in reference frames")
        rest_geo = specimen_geometry(_fill_holes(rest), p.flash_fraction)
        for i in range(len(frames)):
            img = frames.read(i)[:overlay]
            row: dict[str, float] = {"frame": i}
            w = largest_component(weight_mask(img, reference, info.weight_appearance, p.segmentation),
                                  p.segmentation.min_area)
            if w is not None:
                stats = _blob_stats(_fill_holes(w))
                row.update({f"w_{k}": v for k, v in stats.items()})
                row["w_in_view"] = float(stats["top"] > m and stats["bottom"] < overlay - m
                                         and stats["left"] > m and stats["right"] < cam.width - m)
            sp = largest_component(specimen_mask(img, p.segmentation), p.segmentation.min_area)
            if sp is not None:
                geo = specimen_geometry(_fill_holes(sp), p.flash_fraction)
                row.update({f"s_{k}": v for k, v in geo.items()})
            rows.append(row)
        df = pd.DataFrame(rows)

        # --- scale and camera elevation from the resting specimen --------------------
        s = rest_geo["width"] / D                       # px per mm (horizontal)
        elevation = camera_elevation(rest_geo["width"], rest_geo["bottom"] - rest_geo["top"], D, H)
        s_v = s * math.cos(elevation)                   # px per mm of vertical motion
        fps = cam.fps
        t = df["frame"].to_numpy() / fps

        # --- first approach: weight in view and close to the specimen top ------------
        in_view = df["w_in_view"].fillna(0).to_numpy() > 0
        bottom = df["w_bottom"].to_numpy()
        near = in_view & (bottom >= rest_geo["top"] - p.approach_clearance_mm * s_v)
        if not near.any():
            raise RuntimeError(f"{case}: weight never approaches the specimen in view")
        approach = int(np.argmax(near))
        run_end = approach - 1
        run_start = run_end
        while run_start - 1 >= 0 and in_view[run_start - 1]:
            run_start -= 1
        run_start = max(run_start, run_end - p.fit_window_frames + 1)
        run = np.arange(run_start, run_end + 1)
        if len(run) < p.min_fit_frames:
            raise RuntimeError(f"{case}: only {len(run)} free-fall frames in view before the approach")

        # --- pass 2: rigid tracking of the weight from the start of the fit window ----
        overlay_crop = lambda i: frames.read(i)[:overlay]  # noqa: E731
        seg = lambda img: _fill_holes(largest_component(  # noqa: E731
            weight_mask(img, reference, info.weight_appearance, p.segmentation), p.segmentation.min_area))
        ref_mask = seg(overlay_crop(run_start))
        ys, xs = np.nonzero(ref_mask)
        com0 = np.array([xs.mean(), ys.mean()])
        last = min(len(df), approach + p.track_after_frames)
        tracked = track_rigid(overlay_crop, seg, run_start, last, com0, float(ys.max() - ys.min()), p)
        pos = np.full((len(df), 2), np.nan)
        theta = np.full(len(df), np.nan)
        ecc = np.full(len(df), np.nan)
        for i, (c_xy, th, cc) in tracked.items():
            pos[i], theta[i], ecc[i] = c_xy, th, cc
        last = max(tracked) + 1

    # --- local free-fall fit of the tracked centre of mass, then contact -------------
    # A free quadratic absorbs the apparent acceleration (g distorted by perspective),
    # so its short extrapolation to the contact is accurate; a fixed-g model is not.
    y_tr = pos[:, 1]
    coef = np.polyfit(t[run], y_tr[run], 2)
    model = np.polyval(coef, t)
    residual = y_tr - model
    contact = None
    for i in range(run_end + 1, last - p.contact_consecutive):
        window = residual[i:i + p.contact_consecutive]
        if np.all(np.isfinite(window)) and np.all(window < -p.contact_threshold_px):
            contact = i - 1  # last frame still on the parabola
            break
    if contact is None:
        raise RuntimeError(f"{case}: no departure from free fall found in the tracked window")
    t_c = t[contact]
    y_c = float(np.polyval(coef, t_c))
    v_c = float(np.polyval(np.polyder(coef), t_c))

    # --- physical time series ----------------------------------------------------------
    bottom_px = pd.Series(df["s_bottom"].to_numpy()).rolling(5, center=True, min_periods=1).median().to_numpy()
    lift = (rest_geo["bottom"] - bottom_px) / s_v * 1e-3
    lift_c = float(lift[contact])
    out = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "t": t - t_c,
        "indentation": (y_tr - y_c) / s_v * 1e-3,                             # m, + down since contact
        "rotation_deg": np.degrees(theta),                                      # in-plane, since fit start
        "track_ecc": ecc,
        "weight_x": (df["w_cx"] - rest_geo["center_x"]) / s * 1e-3,            # m, + right (blob centroid)
        "weight_centroid_z": -(df["w_cy"] - y_c) / s_v * 1e-3,                 # m, diagnostic
        "silhouette_angle_deg": df["w_angle_deg"],                            # raw, unreliable with knobs
        "weight_area_px": df["w_area"],
        "bulge_diameter": df["s_width"] / s * 1e-3,                             # m
        "specimen_lift": lift,                                                 # m, + up (base moves with the table)
        # weight centre-of-mass displacement relative to the specimen base: removes the
        # motion of the (compliant) table the specimen rests on
        "indentation_rel": (y_tr - y_c) / s_v * 1e-3 + (lift - lift_c),
        "free_fall_residual_px": residual,
    })
    pre = slice(max(run_end - 5, 0), run_end + 1)  # clean free-fall frames just before the approach
    omega = np.gradient(theta, t)
    meta = {
        "dataset": dataset.name,
        "doi": dataset.doi,
        "case": case,
        "frames": int(len(df)),
        "fps": fps,
        "scale_px_per_mm": s,
        "camera_elevation_deg": math.degrees(elevation),
        "vertical_scale_px_per_mm": s_v,
        "rest_specimen_px": rest_geo,
        "free_fall": {
            "frames": [int(run[0]), int(run[-1])],
            "rms_residual_px": float(np.sqrt(np.nanmean(residual[run] ** 2))),
            # apparent acceleration in the image, < g because of perspective (diagnostic)
            "apparent_accel": float(2 * coef[0] / s_v * 1e-3),
        },
        "tracked_frames": [int(min(tracked)), int(max(tracked))],
        "contact_frame": int(contact),
        "contact_uncertainty_frames": 1,
        "impact_speed": v_c / s_v * 1e-3,
        "impact_speed_nominal": math.sqrt(2 * G * 0.38),
        "rotation_rate_at_contact_deg_s": float(np.degrees(np.nanmedian(omega[pre]))),
        "offset_at_contact_mm": float(np.nanmedian(out["weight_x"].to_numpy()[pre]) * 1e3),
        "params": asdict(p),
    }
    return CaseTrack(case, out, meta)

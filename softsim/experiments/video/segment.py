"""Colour segmentation for green-screen drop-test recordings.

Strategy (measured on the Czopek et al. recordings):
    * the backdrop is saturated green with essentially no red (R ~ 0), and
      shadows on it stay red-free, only ~20-30 % darker;
    * the translucent silicone specimen has red > 0 and is bluish (B - R large);
    * brass weights have red > 0 and neutral/yellow colour (B - R small);
    * black weights are much darker than the backdrop behind them (< ~55 %).
A per-pixel reference background (median of early frames, before the weight
enters) makes the darkness test robust to the backdrop's lighting gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationParams:
    red_min: int = 40               # red channel above this: not backdrop/shadow
    specimen_blue_minus_red: int = 30
    brass_blue_minus_red_max: int = 25
    dark_ratio: float = 0.55        # value / reference value below this: dark object
    min_area: int = 150             # px; smaller blobs are noise
    open_kernel: int = 3


def reference_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of frames without the weight (BGR uint8)."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _clean(mask: np.ndarray, p: SegmentationParams) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.open_kernel, p.open_kernel))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)


def largest_component(mask: np.ndarray, min_area: int) -> np.ndarray | None:
    """Mask of the largest connected component, or None if smaller than ``min_area``."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return None
    return labels == best


def specimen_mask(frame: np.ndarray, p: SegmentationParams) -> np.ndarray:
    b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
    return _clean((r > p.red_min) & (b - r > p.specimen_blue_minus_red), p)


def weight_mask(frame: np.ndarray, reference: np.ndarray, appearance: str, p: SegmentationParams) -> np.ndarray:
    """Weight pixels for a ``brass`` or ``dark`` weight."""
    b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
    value = frame.max(axis=2).astype(np.float32)
    ref_value = np.maximum(reference.max(axis=2).astype(np.float32), 1.0)
    dark = value / ref_value < p.dark_ratio
    if appearance == "brass":
        lit = (r > p.red_min) & (b - r < p.brass_blue_minus_red_max)
        mask = lit | dark  # brass has dark reflections too
    else:
        light_text = (r > p.red_min) & (b - r < p.brass_blue_minus_red_max)  # printed labels
        mask = dark | light_text
    return _clean(mask, p)

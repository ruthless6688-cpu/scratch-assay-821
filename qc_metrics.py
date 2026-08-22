"""Scratch-wound quality-control and front roughness metrics.

The module is deliberately independent from image segmentation.  It accepts a
binary wound mask and/or a sampled centre-line width sequence, so it can be
used by both the prototype scripts and a future batch pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


@dataclass
class WidthQCResult:
    """制作质量判定结果；异常点只标记，不删除原始数据。"""

    valid_points: int
    mean_width: float
    median_width: float
    std_width: float
    cv: float
    endpoint_ratio: float
    anomaly_fraction: float
    label: str
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _robust_anomaly_mask(widths: np.ndarray, z_limit: float = 3.5) -> np.ndarray:
    med = float(np.median(widths))
    mad = float(np.median(np.abs(widths - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        return np.zeros(widths.shape, dtype=bool)
    return np.abs(widths - med) > z_limit * scale


def evaluate_width_quality(
    widths: Iterable[float],
    *,
    min_points: int = 12,
    max_cv: float = 0.25,
    max_endpoint_ratio: float = 1.8,
    max_anomaly_fraction: float = 0.10,
    smooth_window: int = 7,
) -> WidthQCResult:
    """Evaluate width uniformity and identify common fabrication defects.

    ``widths`` should already be sampled along the main centre line, with the
    natural tapered ends removed when possible.  A robust trend test catches
    linear gradients; first differences catch abrupt local changes and an
    autocorrelation test catches oscillatory (wavy) wounds.
    """

    x = np.asarray(list(widths), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    if n == 0:
        return WidthQCResult(0, np.nan, np.nan, np.nan, np.inf, np.nan, 1.0,
                             "invalid", False, ("no_valid_width_points",))

    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    cv = std / mean if mean > 0 else np.inf
    endpoint_ratio = float(max(x[0], x[-1]) / max(min(x[0], x[-1]), 1e-9))
    anomaly = _robust_anomaly_mask(x)
    anomaly_fraction = float(np.mean(anomaly))
    reasons: list[str] = []

    if n < min_points:
        reasons.append("too_few_points")
    if cv > max_cv:
        reasons.append("width_cv_exceeds_limit")
    if endpoint_ratio > max_endpoint_ratio:
        reasons.append("endpoint_gradient")
    if anomaly_fraction > max_anomaly_fraction:
        reasons.append("local_outliers")

    # Detect a sustained monotonic trend, independently of endpoint tapering.
    if n >= 5:
        t = np.arange(n, dtype=float)
        slope = np.polyfit(t, x, 1)[0]
        norm_slope = abs(slope) * (n - 1) / max(mean, 1e-9)
        if norm_slope > 0.35:
            reasons.append("linear_gradient")

        d = np.diff(x)
        if d.size >= 4:
            jump = np.abs(d) / max(median, 1e-9)
            if float(np.quantile(jump, 0.95)) > 0.35:
                reasons.append("local_step_change")

        # Remove slow trend before testing alternating oscillation.
        if smooth_window > 1 and n >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            trend = np.convolve(x, kernel, mode="same")
            residual = x - trend
            ac = np.corrcoef(residual[:-1], residual[1:])[0, 1]
            if np.isfinite(ac) and ac < -0.35 and float(np.std(residual)) / max(mean, 1e-9) > 0.08:
                reasons.append("oscillatory_width")

    passed = len(reasons) == 0
    label = "pass" if passed else "fail"
    return WidthQCResult(n, mean, median, std, cv, endpoint_ratio,
                         anomaly_fraction, label, passed, tuple(dict.fromkeys(reasons)))


def front_roughness(mask: np.ndarray, *, smoothing_sigma: float = 2.0) -> Dict[str, float]:
    """Return perimeter ratio R = L_real / L_smooth for a wound mask.

    The mask is cleaned only for component selection; the measured contour is
    kept unsmoothed.  ``R`` is therefore decoupled from centre-line width CV.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return {"roughness_R": np.nan, "perimeter_real": np.nan,
                "perimeter_smooth": np.nan, "area": 0.0}
    labels = measure.label(binary)
    props = measure.regionprops(labels)
    region = max(props, key=lambda p: p.area)
    component = labels == region.label
    contour = measure.find_contours(component.astype(float), 0.5)
    if not contour:
        return {"roughness_R": np.nan, "perimeter_real": np.nan,
                "perimeter_smooth": np.nan, "area": float(region.area)}
    real = max(contour, key=len)
    smooth_mask = ndi.gaussian_filter(component.astype(float), smoothing_sigma) > 0.5
    smooth_contours = measure.find_contours(smooth_mask.astype(float), 0.5)
    smooth = max(smooth_contours, key=len) if smooth_contours else real

    def perimeter(points: np.ndarray) -> float:
        closed = np.vstack([points, points[0]])
        return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

    p_real = perimeter(real)
    p_smooth = perimeter(smooth)
    return {"roughness_R": p_real / max(p_smooth, 1e-9),
            "perimeter_real": p_real, "perimeter_smooth": p_smooth,
            "area": float(region.area)}


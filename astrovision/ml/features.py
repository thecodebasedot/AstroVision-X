"""Feature extraction: turn a catalog into a matrix models can consume.

Two kinds of representation are built here.  The *tabular* features are the
measured physical quantities -- fluxes, radii, CAS, Gini/M20, Sersic index --
which are interpretable and go straight into the report.  The *image*
embeddings are learned or hand-crafted descriptors of the pixel data itself,
which capture structure no scalar summary does.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import (
    as_float_image,
    gaussian_filter,
    nan_to_finite,
    normalise_unit,
    radial_profile,
)
from ..core.types import Source, SourceCatalog

log = get_logger("ml.features")

#: Tabular feature names, in the order :func:`catalog_features` produces them.
FEATURE_NAMES: List[str] = [
    "log_flux", "snr", "peak_over_flux", "log_area",
    "semi_major", "axis_ratio", "ellipticity", "fwhm",
    "concentration", "asymmetry", "smoothness", "gini", "m20",
    "sersic_index", "log_effective_radius", "spiral_strength", "bar_strength",
    "arm_count", "kron_radius", "petrosian_radius",
    "surface_brightness", "log_r50", "r90_over_r50", "elongation",
]


def _safe_log(value: float, floor: float = 1e-6) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(np.log10(max(value, floor)))


def source_features(source: Source) -> np.ndarray:
    """The tabular feature vector for one source, with NaN for what is unknown."""
    p, m = source.photometry, source.morphology
    r50 = source.meta.get("r50", float("nan"))
    r90 = source.meta.get("r90", float("nan"))
    axis_ratio = (m.semi_minor / m.semi_major
                  if np.isfinite(m.semi_major) and m.semi_major > 0 else float("nan"))
    peak_ratio = (p.peak / p.flux
                  if np.isfinite(p.flux) and p.flux > 0 and np.isfinite(p.peak)
                  else float("nan"))
    r_ratio = (r90 / r50 if np.isfinite(r50) and r50 > 0 and np.isfinite(r90)
               else float("nan"))
    return np.array([
        _safe_log(p.flux), p.snr, peak_ratio, _safe_log(max(m.area_pixels, 1)),
        m.semi_major, axis_ratio, m.ellipticity, m.fwhm,
        m.concentration, m.asymmetry, m.smoothness, m.gini, m.m20,
        m.sersic_index, _safe_log(m.effective_radius), m.spiral_strength, m.bar_strength,
        float(m.arm_count), p.kron_radius, p.petrosian_radius,
        p.surface_brightness, _safe_log(r50), r_ratio, m.elongation,
    ], dtype=float)


def catalog_features(catalog: SourceCatalog) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix ``(N, D)`` plus the column names."""
    if len(catalog) == 0:
        return np.empty((0, len(FEATURE_NAMES))), list(FEATURE_NAMES)
    matrix = np.vstack([source_features(s) for s in catalog])
    # Infinities arise from ratios like elongation on a perfectly round
    # source; treat them as missing rather than letting them poison a scaler.
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, list(FEATURE_NAMES)


def feature_report(matrix: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-feature coverage and spread, for the provenance section of a report."""
    report: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(names):
        column = matrix[:, j] if matrix.size else np.array([])
        finite = column[np.isfinite(column)]
        report[name] = {
            "coverage": float(finite.size / max(len(column), 1)),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "spread": float(np.std(finite)) if finite.size else float("nan"),
        }
    return report


# --------------------------------------------------------------------------
# image descriptors
# --------------------------------------------------------------------------
def cutout_descriptor(cutout: np.ndarray, n_radial: int = 8,
                      n_scales: int = 3) -> np.ndarray:
    """A compact hand-crafted descriptor of a postage stamp.

    Combines a normalised radial light profile, multi-scale texture energy
    and low-order image moments.  It needs no training data, which makes it
    the fallback whenever a learned embedding is unavailable -- and it is
    still informative enough for nearest-neighbour lookups.
    """
    data = nan_to_finite(as_float_image(cutout), 0.0)
    if data.size == 0:
        return np.zeros(n_radial + n_scales + 6, dtype=float)

    total = float(np.clip(data, 0, None).sum())
    if total <= 0:
        return np.zeros(n_radial + n_scales + 6, dtype=float)
    normalised = data / total

    _, profile = radial_profile(normalised, nbins=n_radial)
    profile = nan_to_finite(profile, 0.0)
    peak = float(profile.max()) if profile.size else 0.0
    profile = profile / peak if peak > 0 else profile

    # Texture energy at several smoothing scales: the difference between an
    # image and its blurred self measures structure at that scale.
    texture = []
    for scale in range(1, n_scales + 1):
        blurred = gaussian_filter(normalised, 1.5 * scale)
        texture.append(float(np.abs(normalised - blurred).sum()))

    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    weights = np.clip(normalised, 0, None)
    mass = float(weights.sum())
    if mass > 0:
        cx = float((weights * xx).sum() / mass)
        cy = float((weights * yy).sum() / mass)
        dx, dy = xx - cx, yy - cy
        mxx = float((weights * dx * dx).sum() / mass)
        myy = float((weights * dy * dy).sum() / mass)
        mxy = float((weights * dx * dy).sum() / mass)
        scale = max(mxx + myy, 1e-9)
        moments = [mxx / scale, myy / scale, mxy / scale,
                   float(np.sqrt(scale)) / max(min(ny, nx), 1)]
    else:
        moments = [0.0, 0.0, 0.0, 0.0]

    positive = np.clip(normalised, 0, None).ravel()
    positive = positive[positive > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    concentration = float(np.sort(normalised.ravel())[::-1][:max(1, data.size // 20)].sum())

    return np.concatenate([profile, np.array(texture), np.array(moments),
                           np.array([entropy, concentration])])


def catalog_embeddings(catalog: SourceCatalog, image, size: int = 48,
                       normalise: bool = True) -> np.ndarray:
    """Hand-crafted embeddings for every source; also stored on each source."""
    if len(catalog) == 0:
        return np.empty((0, 0))
    vectors = []
    for source in catalog:
        cutout = image.cutout(source.x, source.y, size, subtract_background=True)
        vectors.append(cutout_descriptor(cutout))
    matrix = np.vstack(vectors)
    if normalise:
        matrix = normalise_unit(matrix, axis=1)
    for source, vector in zip(catalog, matrix):
        source.embedding = vector
    log.debug("computed %d image embeddings of dimension %d", *matrix.shape)
    return matrix


def combine_features(tabular: np.ndarray, embeddings: Optional[np.ndarray] = None,
                     embedding_weight: float = 0.5) -> np.ndarray:
    """Concatenate tabular features with (down-weighted) image embeddings."""
    if embeddings is None or embeddings.size == 0:
        return tabular
    if len(embeddings) != len(tabular):
        raise ValueError("tabular features and embeddings must have equal length")
    return np.hstack([tabular, float(embedding_weight) * embeddings])

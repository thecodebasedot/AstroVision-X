"""Population statistics for a field.

Individual measurements say what one object is; these say what the *field*
is -- how many objects there are per magnitude, how they are clustered on
the sky, and whether the catalog is complete enough for those statements to
mean anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.types import ObjectClass, SourceCatalog

log = get_logger("astrophysics.statistics")


def number_counts(magnitudes: Sequence[float], bin_width: float = 0.5,
                  area_sq_arcmin: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Differential number counts ``N(m)``, the basic descriptor of a field."""
    values = np.asarray([m for m in magnitudes if np.isfinite(m)], dtype=float)
    if values.size < 3:
        return {"magnitude": np.array([]), "counts": np.array([]),
                "log_counts": np.array([]), "density": np.array([])}
    low = np.floor(values.min() / bin_width) * bin_width
    high = np.ceil(values.max() / bin_width) * bin_width + bin_width
    edges = np.arange(low, high, bin_width)
    counts, _ = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        log_counts = np.where(counts > 0, np.log10(np.maximum(counts, 1)), np.nan)
    density = (counts / (area_sq_arcmin * bin_width)
               if area_sq_arcmin else counts / bin_width)
    return {"magnitude": centres, "counts": counts.astype(float),
            "log_counts": log_counts, "density": density,
            "bin_width": np.array([bin_width])}


def counts_slope(counts: Dict[str, np.ndarray],
                 magnitude_range: Optional[Tuple[float, float]] = None) -> float:
    """Slope of ``log N`` versus magnitude.

    A Euclidean, uniformly-filled universe gives 0.6.  Departures indicate
    either genuine structure or, far more often, incompleteness at the faint
    end -- which is why the turnover is used to estimate the limit below.
    """
    magnitude = counts.get("magnitude", np.array([]))
    log_counts = counts.get("log_counts", np.array([]))
    good = np.isfinite(magnitude) & np.isfinite(log_counts)
    if magnitude_range is not None:
        good &= (magnitude >= magnitude_range[0]) & (magnitude <= magnitude_range[1])
    if good.sum() < 3:
        return float("nan")
    return float(np.polyfit(magnitude[good], log_counts[good], 1)[0])


def completeness_limit(counts: Dict[str, np.ndarray]) -> float:
    """Magnitude at which the counts turn over -- the practical depth limit.

    Counts rise with magnitude until the survey starts missing objects, and
    then fall.  The peak is the honest place to stop quoting completeness.
    """
    magnitude = counts.get("magnitude", np.array([]))
    values = counts.get("counts", np.array([]))
    if magnitude.size < 4 or values.size < 4:
        return float("nan")
    peak = int(np.argmax(values))
    if peak >= len(magnitude) - 1:
        return float(magnitude[-1])
    return float(magnitude[peak])


def luminosity_function(magnitudes: Sequence[float], redshift: float,
                        area_sq_deg: float, cosmology=None,
                        bin_width: float = 0.5) -> Dict[str, np.ndarray]:
    """Number density per absolute magnitude in a comoving volume."""
    from .cosmology import DEFAULT_COSMOLOGY

    cosmology = cosmology or DEFAULT_COSMOLOGY
    mu = cosmology.distance_modulus(redshift)
    if not np.isfinite(mu):
        return {"absolute_magnitude": np.array([]), "phi": np.array([])}
    absolute = np.asarray([m - mu for m in magnitudes if np.isfinite(m)], dtype=float)
    if absolute.size < 3:
        return {"absolute_magnitude": np.array([]), "phi": np.array([])}
    volume = cosmology.comoving_volume(redshift, area_sq_deg)
    counts = number_counts(absolute, bin_width)
    phi = counts["counts"] / max(volume * bin_width, 1e-9)
    return {"absolute_magnitude": counts["magnitude"], "phi": phi,
            "volume_mpc3": np.array([volume])}


def two_point_correlation(positions: np.ndarray, bins: Optional[np.ndarray] = None,
                          n_random: int = 5000, random_state: int = 42,
                          field_shape: Optional[Tuple[float, float]] = None
                          ) -> Dict[str, np.ndarray]:
    """Angular two-point correlation via the Landy-Szalay estimator.

    Measures how much more likely two objects are to be found at a given
    separation than if they were scattered at random -- the standard way to
    detect clustering, and hence structure, in a field.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or len(points) < 10:
        return {"separation": np.array([]), "w": np.array([])}

    if field_shape is None:
        low = points.min(axis=0)
        high = points.max(axis=0)
    else:
        low = np.zeros(2)
        high = np.array([field_shape[1], field_shape[0]], dtype=float)
    extent = float(np.min(high - low))
    if extent <= 0:
        return {"separation": np.array([]), "w": np.array([])}
    if bins is None:
        bins = np.logspace(np.log10(max(extent / 200.0, 0.5)),
                           np.log10(extent / 3.0), 12)

    rng = np.random.default_rng(random_state)
    randoms = rng.uniform(low, high, size=(int(n_random), 2))

    def pair_counts(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
        if b is None:
            distance = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
            distance = distance[np.triu_indices(len(a), k=1)]
        else:
            distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).ravel()
        return np.histogram(distance, bins=bins)[0].astype(float)

    dd = pair_counts(points)
    rr = pair_counts(randoms)
    dr = pair_counts(points, randoms)

    n_d, n_r = len(points), len(randoms)
    norm_dd = dd / max(n_d * (n_d - 1) / 2.0, 1.0)
    norm_rr = rr / max(n_r * (n_r - 1) / 2.0, 1.0)
    norm_dr = dr / max(float(n_d * n_r), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(norm_rr > 0,
                     (norm_dd - 2 * norm_dr + norm_rr) / np.maximum(norm_rr, 1e-12),
                     np.nan)
    centres = np.sqrt(bins[:-1] * bins[1:])
    return {"separation": centres, "w": w, "dd": dd, "rr": rr, "dr": dr}


def nearest_neighbour_statistics(positions: np.ndarray) -> Dict[str, float]:
    """Clustering summary from nearest-neighbour distances.

    The ratio of the observed mean nearest-neighbour distance to the value
    expected for a random field (Clark & Evans) is below 1 for a clustered
    distribution and above 1 for a regular one.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or len(points) < 4:
        return {"mean_nn_distance": float("nan"), "clark_evans": float("nan"),
                "n": float(len(points))}
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    mean_nn = float(np.mean(nearest))

    span = points.max(axis=0) - points.min(axis=0)
    area = float(np.prod(np.maximum(span, 1e-9)))
    density = len(points) / max(area, 1e-9)
    expected = 0.5 / np.sqrt(max(density, 1e-12))
    return {
        "mean_nn_distance": mean_nn,
        "median_nn_distance": float(np.median(nearest)),
        "expected_random": float(expected),
        "clark_evans": float(mean_nn / expected) if expected > 0 else float("nan"),
        "density_per_sq_px": float(density),
        "n": float(len(points)),
    }


def field_statistics(catalog: SourceCatalog, shape: Tuple[int, int],
                     pixel_scale: float = 1.0) -> Dict[str, Any]:
    """Everything the report needs to say about the field as a population."""
    if len(catalog) == 0:
        return {"n_sources": 0}

    magnitudes = [s.photometry.magnitude for s in catalog]
    counts = number_counts(magnitudes, 0.5)
    area_sq_arcmin = (shape[0] * shape[1] * pixel_scale ** 2) / 3600.0
    positions = catalog.positions()

    galaxies = catalog.of_class(ObjectClass.GALAXY)
    stars = catalog.of_class(ObjectClass.STAR)
    statistics: Dict[str, Any] = {
        "n_sources": len(catalog),
        "class_counts": catalog.class_counts(),
        "area_sq_arcmin": float(area_sq_arcmin),
        "source_density_per_sq_arcmin": float(len(catalog) / max(area_sq_arcmin, 1e-9)),
        "counts_slope": counts_slope(counts),
        "completeness_limit": completeness_limit(counts),
        "magnitude_range": [
            float(np.nanmin(magnitudes)) if np.isfinite(magnitudes).any() else float("nan"),
            float(np.nanmax(magnitudes)) if np.isfinite(magnitudes).any() else float("nan"),
        ],
        "clustering": nearest_neighbour_statistics(positions),
        "star_galaxy_ratio": (float(len(stars) / len(galaxies))
                              if len(galaxies) else float("nan")),
    }

    if 30 <= len(catalog) <= 4000:
        # The estimator is O(N * N_random); above a few thousand sources it
        # dominates the run, and a subsample would answer a different question.
        correlation = two_point_correlation(
            positions, field_shape=shape,
            n_random=int(np.clip(20 * len(catalog), 1000, 8000)))
        if correlation["separation"].size:
            finite = np.isfinite(correlation["w"])
            statistics["two_point_correlation"] = {
                "separation_px": correlation["separation"][finite].tolist(),
                "w": correlation["w"][finite].tolist(),
            }

    morphologies: Dict[str, int] = {}
    for source in galaxies:
        key = source.morphology.label.value
        morphologies[key] = morphologies.get(key, 0) + 1
    if morphologies:
        statistics["morphology_counts"] = dict(
            sorted(morphologies.items(), key=lambda kv: -kv[1]))

    return statistics

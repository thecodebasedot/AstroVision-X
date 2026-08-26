"""Light-curve extraction from a multi-epoch image series."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import TimeSeriesConfig
from ..core.logging import get_logger
from ..core.types import LightCurve, Source, SourceCatalog
from ..io.image import AstroImage, ImageSeries
from ..photometry.aperture import aperture_photometry
from .features import variability_features, variability_score
from .periodogram import find_period

log = get_logger("timeseries.lightcurve")


def extract_light_curve(series: ImageSeries, x: float, y: float,
                        radius: float = 4.0, gain: float = 1.0,
                        annulus: Tuple[float, float] = (8.0, 14.0),
                        wcs=None, band: Optional[str] = None,
                        source_id: Optional[int] = None) -> LightCurve:
    """Measure one position across every epoch of a series.

    When the series carries a WCS the aperture is placed by sky coordinate
    rather than by pixel, so a small pointing drift between epochs does not
    smear the photometry.
    """
    times: List[float] = []
    fluxes: List[float] = []
    errors: List[float] = []
    reference_wcs = wcs if wcs is not None else series.reference.wcs
    ra = dec = None
    if reference_wcs is not None:
        ra, dec = reference_wcs.pixel_to_world(x, y)

    for index, image in enumerate(series):
        cx, cy = x, y
        if ra is not None and image.wcs is not None:
            px, py = image.wcs.world_to_pixel(ra, dec)
            cx, cy = float(px), float(py)
        result = aperture_photometry(
            image.subtracted(), (cx, cy), radius, rms=image.rms_map(),
            gain=float(image.header.get("GAIN", gain) or gain),
            local_background=True, annulus=annulus, mask=image.mask)
        times.append(float(image.mjd if image.mjd is not None else index))
        fluxes.append(float(result.flux))
        errors.append(float(result.flux_err))

    return LightCurve(np.array(times), np.array(fluxes), np.array(errors),
                      band=band or series.reference.band, source_id=source_id,
                      meta={"x": float(x), "y": float(y), "ra": ra, "dec": dec,
                            "aperture_radius": float(radius)})


class LightCurveAnalyzer:
    """Extracts and characterises light curves for a whole catalog.

    >>> from astrovision.simulate import SkySimulator, SkyConfig
    >>> sim = SkySimulator(SkyConfig(shape=(96, 96), n_stars=8, n_galaxies=1,
    ...                              n_nebulae=0, n_clusters=0, n_lenses=0,
    ...                              n_anomalies=0, seed=1))
    >>> series, _, _ = sim.generate_series(n_epochs=4, n_transients=0)
    >>> isinstance(series.times, np.ndarray)
    True
    """

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.report: Dict[str, float] = {}

    def run(self, series: ImageSeries, catalog: SourceCatalog,
            sources: Optional[Sequence[Source]] = None) -> Dict[int, LightCurve]:
        """Build light curves and attach variability scores to the catalog."""
        cfg = self.config
        if not cfg.enabled or len(series) < cfg.min_epochs:
            if len(series) < cfg.min_epochs:
                log.info("only %d epochs; need %d for time-series analysis",
                         len(series), cfg.min_epochs)
            return {}

        targets = list(sources if sources is not None else catalog)
        curves: Dict[int, LightCurve] = {}
        n_variable = 0

        for source in targets:
            curve = extract_light_curve(
                series, source.x, source.y, cfg.aperture_radius,
                source_id=source.id)
            clean = curve.clean()
            if len(clean) < cfg.min_epochs:
                continue
            curves[source.id] = curve

            features = variability_features(curve)
            score = variability_score(curve, cfg.variability_threshold)
            source.variability_score = float(score)
            source.meta["variability"] = features

            if cfg.period_search and score > 0.35 and len(clean) >= 6:
                period = find_period(curve, cfg.min_period, cfg.max_period,
                                     cfg.n_frequencies)
                source.meta["period"] = period
                if (np.isfinite(period["false_alarm_probability"]) and
                        period["false_alarm_probability"] < cfg.fap_threshold):
                    source.add_flag("periodic")
            if score > 0.5:
                source.add_flag("variable")
                n_variable += 1

        self.report = {
            "n_curves": len(curves),
            "n_epochs": len(series),
            "baseline": float(series.times[-1] - series.times[0]) if len(series) > 1 else 0.0,
            "n_variable": n_variable,
        }
        log.info("extracted %d light curves over %d epochs; %d flagged variable",
                 len(curves), len(series), n_variable)
        return curves


def classify_variable(curve: LightCurve, period: Optional[Dict[str, float]] = None
                      ) -> Tuple[str, float]:
    """A coarse variability class from light-curve shape.

    Distinguishes the broad families a survey pipeline must triage:
    periodic pulsators and eclipsing systems, eruptive transients, secular
    trends, and stochastic variability.  It is a triage label, not a
    classification anyone should publish without follow-up.
    """
    features = variability_features(curve)
    score = variability_score(curve)
    if score < 0.3:
        return "non_variable", 1.0 - score

    significant_period = (period is not None and np.isfinite(period.get("period", np.nan))
                          and period.get("false_alarm_probability", 1.0) < 0.01)
    skew = features["skewness"]
    eta = features["von_neumann_eta"]
    trend = features["linear_trend"]
    baseline = max(features["baseline"], 1e-9)
    mean = features["mean_flux"]

    if significant_period:
        # A sinusoid is symmetric; eclipses spend most of the time at maximum
        # and dip sharply, which shows up as a strongly negative skew.
        if np.isfinite(skew) and skew < -0.7:
            return "eclipsing", float(min(0.5 + abs(skew) / 4, 0.95))
        return "periodic_pulsator", float(min(0.5 + period.get("power", 0.0), 0.95))

    if np.isfinite(skew) and skew > 1.0 and np.isfinite(eta) and eta < 1.0:
        return "eruptive", float(min(0.4 + skew / 5, 0.9))
    if (np.isfinite(trend) and np.isfinite(mean) and abs(mean) > 1e-9 and
            abs(trend) * baseline > 0.25 * abs(mean)):
        return "secular_trend", 0.7
    return "stochastic", float(min(score, 0.8))

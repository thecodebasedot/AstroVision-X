"""Colours for simulated objects: a stellar locus, and galaxies off it.

Multi-band analysis is only testable if the simulated colours carry the
structure the analysis relies on.  That structure is the **stellar locus**:
stars are (to first order) blackbodies seen through the same filters, so
their colours are not scattered over the colour-colour plane but confined to
a one-parameter curve running from hot and blue to cool and red.  Galaxies
are integrated over a mixed stellar population and redshifted, so they sit
*off* that curve -- and the perpendicular distance from it is what makes
colour-based star/galaxy separation work at all.

The anchor colours below are the SDSS main-sequence locus, in the sense of
Covey et al. (2007): a run of median ``u-g, g-r, r-i, i-z`` along spectral
type.  They are approximate -- this is a simulator, not a synthetic
photometry package -- but the *shape* of the locus, its curvature, and the
tightness of stars around it against the scatter of galaxies are right, and
those are the properties every downstream test depends on.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger

log = get_logger("simulate.sed")

#: Bands the simulator knows colours for, blue to red.
BAND_ORDER = ("u", "g", "r", "i", "z")

#: Median stellar-locus colours at eight points along the main sequence,
#: from hot O/B stars (index 0) to cool M dwarfs (index 7).  Each row is
#: ``(u-g, g-r, r-i, i-z)``.
STELLAR_LOCUS = np.array([
    [0.05, -0.30, -0.14, -0.09],   # O/B
    [0.55, -0.10, -0.04, -0.03],   # A
    [1.00,  0.20,  0.06,  0.02],   # early F
    [1.25,  0.40,  0.13,  0.05],   # late F
    [1.55,  0.60,  0.22,  0.09],   # G
    [2.00,  0.90,  0.34,  0.15],   # K
    [2.45,  1.25,  0.53,  0.26],   # early M
    [2.80,  1.45,  0.95,  0.52],   # late M
], dtype=float)

#: Rest-frame colours of galaxies by Hubble type, ``(u-g, g-r, r-i, i-z)``.
#: An elliptical is an old red population; an irregular is star-forming and
#: blue.  These sit deliberately off the stellar locus.
GALAXY_COLOURS: Dict[str, Sequence[float]] = {
    "elliptical":    (1.95, 1.32, 0.58, 0.31),
    "lenticular":    (1.80, 1.15, 0.50, 0.27),
    "spiral":        (1.45, 0.75, 0.35, 0.19),
    "barred_spiral": (1.50, 0.80, 0.37, 0.20),
    "irregular":     (1.00, 0.35, 0.17, 0.09),
    "merger":        (1.20, 0.55, 0.26, 0.14),
}

#: Non-galaxy classes.  An HII-region nebula is dominated by emission lines,
#: which no blackbody or stellar population reproduces -- it is off the
#: locus in a direction of its own.
OTHER_COLOURS: Dict[str, Sequence[float]] = {
    "nebula":    (0.60, -0.15, 0.45, 0.10),
    "cluster":   (1.35, 0.55, 0.20, 0.10),
    "anomaly":   (1.10, 0.30, 0.55, 0.30),
    "lens_arc":  (0.85, 0.25, 0.12, 0.06),   # blue star-forming source galaxy
    "transient": (1.05, 0.30, 0.10, 0.05),   # young supernova photosphere
}

#: How far, in magnitudes, each class scatters about its mean colour.  Stars
#: are tight because the locus is nearly one-dimensional; galaxies are wide
#: because they span redshift and star-formation history.
COLOUR_SCATTER: Dict[str, float] = {
    "star": 0.035,
    "galaxy": 0.16,
    "nebula": 0.20,
    "cluster": 0.08,
    "anomaly": 0.30,
    "lens_arc": 0.14,
    "transient": 0.12,
}


def _colours_to_offsets(colours: Sequence[float],
                        bands: Sequence[str] = BAND_ORDER) -> Dict[str, float]:
    """Turn consecutive colour indices into magnitude offsets from ``r``.

    ``colours`` is ``(u-g, g-r, r-i, i-z)``.  Magnitudes are referenced to
    ``r`` because that is the simulator's default band, so an object's total
    flux is unchanged when only its colour changes.

    >>> offsets = _colours_to_offsets((1.0, 0.5, 0.2, 0.1))
    >>> round(offsets["g"] - offsets["r"], 6)
    0.5
    """
    values = np.asarray(colours, dtype=float)
    magnitudes = {"r": 0.0}
    magnitudes["i"] = -values[2]
    magnitudes["z"] = magnitudes["i"] - values[3]
    magnitudes["g"] = values[1]
    magnitudes["u"] = magnitudes["g"] + values[0]
    return {band: float(magnitudes[band]) for band in bands if band in magnitudes}


def stellar_colours(temperature_index: float) -> Dict[str, float]:
    """Colours of a star at ``temperature_index`` in ``[0, 1]``.

    0 is the hot blue end of the main sequence, 1 the cool red end.  The
    locus is interpolated linearly between the anchor spectral types, which
    keeps it continuous and, importantly, *curved* in colour-colour space --
    a straight line would make the locus fit trivially and the test
    meaningless.
    """
    position = float(np.clip(temperature_index, 0.0, 1.0)) * (len(STELLAR_LOCUS) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(STELLAR_LOCUS) - 1)
    weight = position - low
    colours = (1.0 - weight) * STELLAR_LOCUS[low] + weight * STELLAR_LOCUS[high]
    return _colours_to_offsets(colours)


def sed_colours(kind: str, bands: Sequence[str], redshift: float,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Magnitude offsets from ``r`` computed by integrating a real spectrum.

    The table-driven :func:`object_colours` is a lookup with a linear
    reddening term, which is fine for a fixed-redshift field and useless for
    testing photometric redshifts: a fit that inverts the same linear law
    that generated the data is measuring its own arithmetic.

    This path draws a spectrum with *continuous* age, dust and emission
    parameters and integrates it through the filter curves, so no simulated
    galaxy is exactly reproducible by the six discrete templates the fit
    searches -- which is the situation with real galaxies.
    """
    from ..photoz.templates import draw_template

    rng = rng or np.random.default_rng()
    template = draw_template(rng, "galaxy" if kind == "galaxy" else kind)
    magnitudes = template.magnitudes(float(redshift), tuple(bands))
    reference = magnitudes.get("r")
    if reference is None or not np.isfinite(reference):
        finite = [v for v in magnitudes.values() if np.isfinite(v)]
        reference = float(np.mean(finite)) if finite else 0.0
    offsets = {band: float(value - reference) for band, value in magnitudes.items()}
    truth = {"redshift": float(redshift), "template": template.name,
             "age_gyr": template.age_gyr, "dust": template.dust,
             "emission": template.emission}
    return offsets, truth


def object_colours(kind: str, morphology: str = "", *,
                   rng: Optional[np.random.Generator] = None,
                   redshift: float = 0.0) -> Dict[str, float]:
    """Magnitude offsets from ``r`` for one simulated object.

    ``kind`` is the simulator's truth category (star, galaxy, nebula,
    cluster, lens, anomaly, transient); ``morphology`` refines galaxies into
    Hubble types.  Scatter is drawn from ``rng`` when given.

    Redshift reddens: at these low redshifts the 4000 Angstrom break moves
    through the bands, so a first-order linear reddening of the blue colours
    is enough to put a galaxy population where a real one would sit.
    """
    rng = rng or np.random.default_rng()
    if kind == "star":
        # Cool stars vastly outnumber hot ones in any magnitude-limited
        # sample, so the temperature index is drawn skewed toward red.
        offsets = stellar_colours(float(rng.beta(2.2, 1.4)))
        scatter = COLOUR_SCATTER["star"]
    elif kind == "galaxy":
        base = GALAXY_COLOURS.get(morphology, GALAXY_COLOURS["spiral"])
        reddened = np.asarray(base, dtype=float) + redshift * np.array([1.4, 0.9, 0.4, 0.2])
        offsets = _colours_to_offsets(reddened)
        scatter = COLOUR_SCATTER["galaxy"]
    elif kind == "lens":
        # The lensed object is what the arcs are made of; the object the
        # detector centroids on is the *deflector*, an old red elliptical.
        # Its arcs get their own colour from ``object_colours("lens_arc")``.
        offsets = _colours_to_offsets(GALAXY_COLOURS["elliptical"])
        scatter = COLOUR_SCATTER["galaxy"]
    else:
        base = OTHER_COLOURS.get(kind, GALAXY_COLOURS["spiral"])
        offsets = _colours_to_offsets(base)
        scatter = COLOUR_SCATTER.get(kind, 0.15)

    if scatter > 0:
        # Scatter is applied per band, then re-referenced to r, so the r-band
        # flux -- and hence the object's detectability -- never changes.
        noise = {band: float(rng.normal(0.0, scatter)) for band in offsets}
        noise["r"] = 0.0
        offsets = {band: value + noise[band] for band, value in offsets.items()}
    return offsets


def flux_ratios(offsets: Mapping[str, float]) -> Dict[str, float]:
    """Convert magnitude offsets from ``r`` into multiplicative flux ratios.

    >>> round(flux_ratios({"r": 0.0, "g": 0.75})["g"], 4)
    0.5012
    """
    return {band: float(10.0 ** (-0.4 * value)) for band, value in offsets.items()}


def locus_distance(colour_x: float, colour_y: float,
                   pair: Sequence[str] = ("g", "r", "i")) -> float:
    """Perpendicular distance of one point from the stellar locus, in mag.

    ``colour_x`` is ``pair[0] - pair[1]`` and ``colour_y`` is
    ``pair[1] - pair[2]``.  Used by the tests to check that the simulator's
    stars really do lie on the locus its analysis code later fits.
    """
    index = {band: i for i, band in enumerate(BAND_ORDER)}
    try:
        first = index[pair[0]], index[pair[1]]
        second = index[pair[1]], index[pair[2]]
    except KeyError as exc:                                # pragma: no cover
        raise ValueError(f"unknown band in {pair}") from exc

    def colour_of(row: np.ndarray, lo: int, hi: int) -> float:
        return float(row[lo:hi].sum())

    samples = np.linspace(0.0, 1.0, 200)
    points = []
    for value in samples:
        position = value * (len(STELLAR_LOCUS) - 1)
        low = int(np.floor(position))
        high = min(low + 1, len(STELLAR_LOCUS) - 1)
        weight = position - low
        row = (1.0 - weight) * STELLAR_LOCUS[low] + weight * STELLAR_LOCUS[high]
        points.append((colour_of(row, *first), colour_of(row, *second)))
    curve = np.asarray(points, dtype=float)
    distances = np.hypot(curve[:, 0] - colour_x, curve[:, 1] - colour_y)
    return float(distances.min())

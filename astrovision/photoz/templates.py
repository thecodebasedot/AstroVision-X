"""Galaxy spectra, filter curves, and the integral that turns them into colours.

A photometric redshift is an inference from a handful of broad-band fluxes to
a distance.  It works because a galaxy spectrum is not featureless: an old
stellar population has a sharp drop shortward of 4000 Angstroms -- the
accumulated absorption of ionised metals in stellar atmospheres -- and a
star-forming one has strong emission lines.  As the galaxy recedes, those
features slide through the filters, and the pattern of fluxes changes in a way
that depends on redshift.

Everything here rests on doing that integral honestly:

.. math::

    f_b = \\frac{\\int f_\\nu(\\lambda / (1+z))\\, T_b(\\lambda)\\, d\\lambda / \\lambda}
               {\\int T_b(\\lambda)\\, d\\lambda / \\lambda}

The ``1/lambda`` weighting is the photon-counting convention, which is what a
CCD does -- it counts photons, not energy.  Using an energy-weighted integral
instead shifts every magnitude by a few hundredths, and, worse, shifts them by
*different* amounts in different filters, which is a colour error and
therefore a redshift error.

The template family is parameterised continuously in age, dust and emission
strength.  That matters for testing: a fit library built from a handful of
discrete templates will never contain the exact spectrum that produced a
simulated galaxy, which is exactly the situation with real galaxies and the
only way to measure a photo-z honestly rather than measure a lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger

log = get_logger("photoz.templates")

#: Rest-frame wavelength grid, Angstroms.  1000-25000 covers the ultraviolet
#: shortward of the Lyman break through the near infrared, which is the range
#: optical filters sample for galaxies out to z ~ 2.
WAVELENGTH = np.logspace(math.log10(900.0), math.log10(25000.0), 900)

#: Approximate SDSS filter centres and widths, Angstroms.  Real throughput
#: curves are not top hats, but the *positions* of the band edges are what
#: photo-z depends on -- they decide at which redshift the 4000 Angstrom break
#: crosses from one filter into the next.
FILTER_BANDS: Dict[str, Tuple[float, float]] = {
    "u": (3550.0, 600.0),
    "g": (4770.0, 1400.0),
    "r": (6230.0, 1400.0),
    "i": (7620.0, 1500.0),
    "z": (9130.0, 1200.0),
    "y": (10200.0, 1100.0),
}

#: Strong rest-frame emission lines: wavelength, and equivalent width in
#: Angstroms at unit emission strength.  Only the few that dominate broad-band
#: colours in a star-forming galaxy are included.
EMISSION_LINES: Sequence[Tuple[float, float]] = (
    (3727.0, 40.0),      # [O II]
    (4861.0, 20.0),      # H beta
    (4959.0, 20.0),      # [O III]
    (5007.0, 60.0),      # [O III]
    (6563.0, 90.0),      # H alpha
    (6584.0, 25.0),      # [N II]
)


def filter_curve(band: str, wavelength: Optional[np.ndarray] = None) -> np.ndarray:
    """Transmission of one filter on a wavelength grid.

    Modelled as a top hat with softened edges rather than a hard one: a
    genuinely square filter makes the synthetic colour a discontinuous
    function of redshift as a feature crosses the edge, and the resulting
    redshift posterior is full of spurious spikes.
    """
    if band not in FILTER_BANDS:
        raise KeyError(f"unknown filter {band!r}; known: {sorted(FILTER_BANDS)}")
    grid = WAVELENGTH if wavelength is None else np.asarray(wavelength, dtype=float)
    centre, width = FILTER_BANDS[band]
    edge = 0.06 * width
    left = 0.5 * (1.0 + np.tanh((grid - (centre - 0.5 * width)) / edge))
    right = 0.5 * (1.0 - np.tanh((grid - (centre + 0.5 * width)) / edge))
    return left * right


@dataclass
class SEDTemplate:
    """A rest-frame spectrum, in per-unit-frequency flux."""

    name: str
    flux: np.ndarray                       # on :data:`WAVELENGTH`
    age_gyr: float = float("nan")
    dust: float = float("nan")
    emission: float = float("nan")
    meta: Dict[str, Any] = field(default_factory=dict)

    def redshifted(self, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """Observed ``(wavelength, flux)`` at redshift ``z``.

        Only the wavelength stretch matters here: the overall ``1/(1+z)``
        dimming is a normalisation, and every colour is a *ratio* of fluxes,
        so it cancels exactly.  Leaving it out avoids pretending the
        templates carry an absolute scale they do not have.
        """
        return WAVELENGTH * (1.0 + float(z)), self.flux

    def magnitudes(self, z: float, bands: Sequence[str]) -> Dict[str, float]:
        """Synthetic AB-like magnitudes through each band at redshift ``z``."""
        observed, flux = self.redshifted(z)
        result: Dict[str, float] = {}
        for band in bands:
            transmission = filter_curve(band, observed)
            denominator = float(np.trapezoid(transmission / observed, observed))
            if denominator <= 0:
                result[band] = float("nan")
                continue
            numerator = float(np.trapezoid(flux * transmission / observed, observed))
            result[band] = (-2.5 * math.log10(numerator / denominator)
                            if numerator > 0 else float("nan"))
        return result

    def colours(self, z: float, bands: Sequence[str]) -> np.ndarray:
        """Consecutive colour indices at redshift ``z``."""
        magnitudes = self.magnitudes(z, bands)
        values = [magnitudes[band] for band in bands]
        return np.array([values[i] - values[i + 1] for i in range(len(values) - 1)])


def build_template(age_gyr: float = 5.0, dust: float = 0.2,
                   emission: float = 0.0, name: str = "") -> SEDTemplate:
    """A galaxy spectrum from three physical knobs.

    Not a stellar population synthesis code -- it is a caricature with the
    right features in the right places:

    * a **4000 Angstrom break** whose depth grows with age, since it comes
      from metal absorption in the atmospheres of cool stars that only
      dominate once the hot ones have died;
    * a **blue continuum** that steepens as age falls, standing in for the
      young massive stars;
    * **dust**, reddening as ``exp(-dust * (5500/lambda))``, a Calzetti-like
      slope that is close enough over the optical;
    * **emission lines** at fixed rest wavelengths, which is what makes a
      star-forming galaxy's colours jump as a line enters a filter.

    Those four behaviours are what a photo-z actually keys on.  A smoother
    caricature would make the problem easier than it is.
    """
    age = float(np.clip(age_gyr, 0.05, 13.0))
    wavelength = WAVELENGTH

    # A young population is blue: a power law whose index runs from about
    # -2.2 (very blue, in f_nu terms) to +1.5 for an old red population.
    slope = -2.2 + 3.7 * float(np.clip(math.log10(age / 0.05) / math.log10(260.0), 0, 1))
    continuum = (wavelength / 5500.0) ** slope

    # The 4000 A break: a step, softened over 150 A, deepening with age.
    depth = 0.15 + 0.55 * float(np.clip(math.log10(age / 0.05) / math.log10(260.0), 0, 1))
    step = 0.5 * (1.0 + np.tanh((wavelength - 4000.0) / 150.0))
    continuum = continuum * (1.0 - depth * (1.0 - step))

    # The Lyman break: nothing gets out shortward of 912 A, and the
    # intergalactic medium eats much of the flux below 1216 A.  At the
    # redshifts this library covers it rarely enters an optical filter, but
    # leaving it out would make high-redshift templates wrongly bright in u.
    continuum = continuum * 0.5 * (1.0 + np.tanh((wavelength - 1216.0) / 60.0))

    attenuation = np.exp(-float(max(dust, 0.0)) * (5500.0 / wavelength))
    spectrum = continuum * attenuation

    strength = float(max(emission, 0.0))
    if strength > 0:
        for line_wavelength, equivalent_width in EMISSION_LINES:
            index = int(np.argmin(np.abs(wavelength - line_wavelength)))
            local = float(spectrum[index])
            sigma = 12.0
            profile = np.exp(-0.5 * ((wavelength - line_wavelength) / sigma) ** 2)
            # Equivalent width is defined against the local continuum, so the
            # line's flux scales with it -- a line on a faint continuum is a
            # faint line, which is what stops emission from dominating the
            # red end of an old galaxy.
            spectrum = spectrum + (local * strength * equivalent_width
                                   * profile / (sigma * math.sqrt(2 * math.pi)))

    total = float(np.trapezoid(spectrum, wavelength))
    if total > 0:
        spectrum = spectrum / total
    label = name or f"age{age:.2f}_dust{dust:.2f}_em{emission:.2f}"
    return SEDTemplate(name=label, flux=spectrum, age_gyr=age, dust=float(dust),
                       emission=float(emission))


#: The discrete library a fit searches over: six spectral types spanning old
#: and red to young and line-dominated.  Deliberately coarse -- a fit library
#: that reproduced every simulated galaxy exactly would measure nothing.
LIBRARY_PARAMETERS: Sequence[Tuple[str, float, float, float]] = (
    ("elliptical",     10.0, 0.10, 0.0),
    ("lenticular",      6.0, 0.20, 0.05),
    ("early_spiral",    3.0, 0.35, 0.3),
    ("late_spiral",     1.2, 0.45, 0.8),
    ("irregular",       0.4, 0.30, 1.4),
    ("starburst",       0.1, 0.60, 2.2),
)


def standard_library() -> List[SEDTemplate]:
    """The six-template library used for fitting."""
    return [build_template(age, dust, emission, name)
            for name, age, dust, emission in LIBRARY_PARAMETERS]


def draw_template(rng: np.random.Generator,
                  kind: str = "galaxy") -> SEDTemplate:
    """A galaxy spectrum with continuously drawn parameters.

    Used by the simulator.  Because the parameters are continuous and the fit
    library is discrete, no simulated galaxy is ever exactly reproducible by
    the fit -- which is the situation with real galaxies, and the only way the
    measured photo-z scatter means anything.
    """
    if kind == "quasar":
        # A power law with strong lines: flat in f_nu and nothing like the
        # galaxy templates, which is why quasars are the classic photo-z
        # catastrophic failure.
        return build_template(age_gyr=0.05, dust=0.0,
                              emission=float(rng.uniform(1.5, 3.0)), name="quasar_like")
    age = float(10 ** rng.uniform(math.log10(0.1), math.log10(11.0)))
    dust = float(rng.uniform(0.0, 0.8))
    emission = float(rng.uniform(0.0, 2.0) * math.exp(-age / 2.0))
    return build_template(age, dust, emission)


def colour_grid(templates: Sequence[SEDTemplate], redshifts: Sequence[float],
                bands: Sequence[str]) -> np.ndarray:
    """Predicted colours for every template at every redshift.

    Shape ``(n_templates, n_redshifts, n_bands - 1)``.  Computed once and
    reused for every galaxy, which is what makes the fit cheap: the expensive
    part is the integral, and it does not depend on the data.
    """
    grid = np.empty((len(templates), len(redshifts), max(len(bands) - 1, 0)))
    for i, template in enumerate(templates):
        for j, z in enumerate(redshifts):
            grid[i, j] = template.colours(z, bands)
    return grid


def describe_break_crossings(bands: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Redshifts at which the 4000 A break sits inside each filter.

    The most useful diagnostic there is for a photo-z: outside these ranges
    the break is between filters and the redshift is poorly constrained, which
    is where the systematic errors live.

    >>> ranges = describe_break_crossings(["g", "r", "i"])
    >>> round(ranges["r"][0], 2), round(ranges["r"][1], 2)
    (0.38, 0.73)
    """
    ranges: Dict[str, Tuple[float, float]] = {}
    for band in bands:
        if band not in FILTER_BANDS:
            continue
        centre, width = FILTER_BANDS[band]
        low = (centre - 0.5 * width) / 4000.0 - 1.0
        high = (centre + 0.5 * width) / 4000.0 - 1.0
        ranges[band] = (float(max(low, 0.0)), float(high))
    return ranges

"""A long-slit spectrograph, simulated as a detector rather than a spectrum.

The point of simulating the 2-D frame instead of handing the analysis a
finished 1-D spectrum is that most of what goes wrong in spectroscopy happens
before the spectrum exists:

* the trace is **curved**, so a fixed row is the wrong row at the ends of the
  order, and an extraction that ignores the curvature loses flux
  wavelength-dependently -- which looks exactly like a spectral feature;
* the dispersion is **non-linear**, so a wavelength solution fitted as a
  straight line leaves residuals that grow toward the edges, which a redshift
  fit will happily absorb into the redshift;
* the sky is **brighter than the object** over most of the optical, and its
  emission lines are far stronger than anything in the target, so sky
  subtraction residuals sit exactly where the interesting lines are;
* the seeing sets the **spatial profile**, and an extraction that sums a fixed
  aperture instead of weighting by that profile throws away signal-to-noise
  that no later step can recover.

Every one of those is drawn here, which is what makes the extraction code
testable against the truth rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..spectra.templates import Spectrum1D

log = get_logger("simulate.spectrograph")

#: Night-sky emission lines: wavelength in Angstroms and relative strength.
#: The bright ones are what dominate a sky-subtraction residual, and the OH
#: forest redward of 7000 A is why redshift surveys work so much better below
#: it.
SKY_LINES: Sequence[Tuple[float, float]] = (
    (5577.3, 1.00),      # [O I] -- the brightest optical sky line
    (5889.9, 0.35),      # Na D
    (5895.9, 0.30),
    (6300.3, 0.55),      # [O I]
    (6363.8, 0.20),
    (6863.9, 0.30),      # O2 band
    (7276.4, 0.35),      # OH
    (7340.9, 0.30),
    (7401.7, 0.30),
    (7524.1, 0.40),
    (7794.1, 0.45),
    (7913.7, 0.50),
    (8344.6, 0.60),
    (8399.2, 0.65),
    (8827.1, 0.70),
    (8919.6, 0.75),
)

#: Arc-lamp lines, wavelength and relative brightness.  A helium-neon-argon
#: lamp: the standard optical comparison source.  Their *positions* are the
#: known quantity a wavelength solution is fitted to.
ARC_LINES: Sequence[Tuple[float, float]] = (
    (3888.6, 0.55), (4026.2, 0.30), (4471.5, 0.45), (4713.1, 0.20),
    (4921.9, 0.35), (5015.7, 0.50), (5460.7, 0.40), (5875.6, 0.90),
    (5944.8, 0.35), (6029.9, 0.30), (6143.1, 0.60), (6266.5, 0.35),
    (6402.2, 1.00), (6506.5, 0.55), (6678.1, 0.50), (6929.5, 0.45),
    (7032.4, 0.70), (7245.2, 0.40), (7438.9, 0.35), (7635.1, 0.85),
    (8006.2, 0.40), (8115.3, 0.95), (8377.6, 0.45), (8521.4, 0.50),
    (9122.9, 0.60), (9657.8, 0.40),
)


@dataclass
class SpectrographConfig:
    """Geometry, dispersion and noise of the simulated instrument."""

    n_columns: int = 1400                  # dispersion axis
    n_rows: int = 60                       # spatial axis, along the slit
    wavelength_start: float = 3700.0       # at column 0
    dispersion: float = 4.0                # Angstroms per column, linear term
    dispersion_quadratic: float = 3.0e-4   # A per column squared
    dispersion_cubic: float = -6.0e-8
    resolution: float = 5.0                # instrumental FWHM, Angstroms
    trace_centre: float = 30.0             # row of the trace at column 0
    trace_tilt: float = 4.0e-3             # rows per column
    trace_curvature: float = 2.2e-6        # rows per column squared
    seeing_sigma: float = 2.2              # spatial profile sigma, rows
    sky_level: float = 120.0               # counts per pixel in the continuum
    sky_line_scale: float = 900.0          # peak counts of the brightest line
    read_noise: float = 4.0
    gain: float = 1.0                      # electrons per count
    seed: int = 0

    def wavelength_at(self, column: np.ndarray) -> np.ndarray:
        """True wavelength of each column.

        Deliberately non-linear.  A real grating's dispersion changes across
        the detector, and pretending otherwise is the single commonest way a
        wavelength solution is quietly wrong at the ends.
        """
        c = np.asarray(column, dtype=float)
        return (self.wavelength_start + self.dispersion * c
                + self.dispersion_quadratic * c ** 2
                + self.dispersion_cubic * c ** 3)

    def trace_at(self, column: np.ndarray) -> np.ndarray:
        """Row of the spectrum's centre at each column."""
        c = np.asarray(column, dtype=float)
        return (self.trace_centre + self.trace_tilt * c
                + self.trace_curvature * c ** 2)


@dataclass
class SpectrographFrame:
    """One simulated exposure, with the truth needed to grade an extraction."""

    image: np.ndarray                       # rows x columns, counts
    variance: np.ndarray
    config: SpectrographConfig
    truth: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.image.shape

    def true_wavelength(self) -> np.ndarray:
        return self.config.wavelength_at(np.arange(self.image.shape[1]))

    def true_trace(self) -> np.ndarray:
        return self.config.trace_at(np.arange(self.image.shape[1]))


class SpectrographSimulator:
    """Renders long-slit frames: objects, sky, and arc lamps."""

    def __init__(self, config: Optional[SpectrographConfig] = None) -> None:
        self.config = config or SpectrographConfig()
        self.rng = np.random.default_rng(self.config.seed)

    # -- helpers -----------------------------------------------------------
    def _broadened(self, spectrum: Spectrum1D, wavelength: np.ndarray,
                   scale: float) -> np.ndarray:
        """Sample a template onto the detector's wavelengths, at its resolution.

        The convolution is done in wavelength space with the instrumental
        Gaussian, before sampling, because sampling first and smoothing after
        would smooth the *pixels* -- a different, and wrong, kernel wherever
        the dispersion changes.
        """
        cfg = self.config
        sigma = cfg.resolution / 2.3548
        source_grid = spectrum.wavelength
        step = float(np.median(np.diff(source_grid)))
        width = max(int(round(4.0 * sigma / max(step, 1e-6))), 1)
        offsets = np.arange(-width, width + 1) * step
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        smoothed = np.convolve(spectrum.flux, kernel, mode="same")
        sampled = np.interp(wavelength, source_grid, smoothed,
                            left=0.0, right=0.0)
        return sampled * float(scale)

    def _profile(self, rows: np.ndarray, centre: float, sigma: float) -> np.ndarray:
        """Normalised spatial profile down the slit at one column."""
        weights = np.exp(-0.5 * ((rows - centre) / sigma) ** 2)
        total = weights.sum()
        return weights / total if total > 0 else weights

    def _sky(self, wavelength: np.ndarray) -> np.ndarray:
        """Sky counts per pixel per row, continuum plus emission lines."""
        cfg = self.config
        sky = np.full_like(wavelength, cfg.sky_level)
        # Redward of 7000 A the OH forest lifts the continuum too.
        sky = sky * (1.0 + 0.6 / (1.0 + np.exp(-(wavelength - 7200.0) / 300.0)))
        sigma = cfg.resolution / 2.3548
        for centre, strength in SKY_LINES:
            sky = sky + (cfg.sky_line_scale * strength
                         * np.exp(-0.5 * ((wavelength - centre) / sigma) ** 2))
        return sky

    # -- frames ------------------------------------------------------------
    def object_frame(self, spectrum: Spectrum1D, redshift: float = 0.0,
                     total_counts: float = 3.0e5,
                     with_sky: bool = True,
                     cosmic_rays: int = 6) -> SpectrographFrame:
        """A frame containing one object on the slit, plus sky and noise.

        ``total_counts`` is the object's summed counts over the whole frame,
        so the signal-to-noise of the extracted spectrum is set by one
        number, which is what makes the tests below able to vary it.
        """
        cfg = self.config
        columns = np.arange(cfg.n_columns)
        rows = np.arange(cfg.n_rows)[:, None]
        wavelength = cfg.wavelength_at(columns)

        observed = spectrum.redshifted(redshift)
        profile_flux = self._broadened(observed, wavelength, 1.0)
        if profile_flux.sum() > 0:
            profile_flux = profile_flux * (total_counts / profile_flux.sum())

        centres = cfg.trace_at(columns)
        # Seeing worsens toward the blue, as atmospheric turbulence does; the
        # profile therefore is not the same width at every wavelength, which
        # is one more thing a fixed aperture gets wrong.
        widths = cfg.seeing_sigma * (wavelength / 5500.0) ** -0.2
        spatial = np.exp(-0.5 * ((rows - centres[None, :]) / widths[None, :]) ** 2)
        spatial = spatial / np.clip(spatial.sum(axis=0, keepdims=True), 1e-9, None)
        clean = spatial * profile_flux[None, :]

        sky = self._sky(wavelength)[None, :] if with_sky else np.zeros((1, len(columns)))
        total = clean + sky

        noisy = self.rng.poisson(np.clip(total * cfg.gain, 0, None)) / cfg.gain
        noisy = noisy + self.rng.normal(0.0, cfg.read_noise, total.shape)
        variance = np.clip(total, 0, None) / cfg.gain + cfg.read_noise ** 2

        hits: List[Tuple[int, int]] = []
        for _ in range(int(cosmic_rays)):
            row = int(self.rng.integers(0, cfg.n_rows))
            column = int(self.rng.integers(0, cfg.n_columns))
            noisy[row, column] += float(self.rng.uniform(2000.0, 9000.0))
            hits.append((row, column))

        truth = {
            "wavelength": wavelength,
            "trace": centres,
            "object_counts": profile_flux,
            "sky": sky[0] if with_sky else np.zeros(len(columns)),
            "redshift": float(redshift),
            "cosmic_rays": hits,
            "template": dict(spectrum.meta),
        }
        return SpectrographFrame(noisy, variance, cfg, truth)

    def arc_frame(self, exposure: float = 1.0) -> SpectrographFrame:
        """A comparison-lamp exposure: known lines, unknown solution.

        The lamp fills the slit, so the lines run the full height of the
        frame -- which is why an arc can be collapsed along the slit and a
        science frame cannot.
        """
        cfg = self.config
        columns = np.arange(cfg.n_columns)
        wavelength = cfg.wavelength_at(columns)
        sigma = cfg.resolution / 2.3548
        spectrum = np.full_like(wavelength, 20.0)
        for centre, strength in ARC_LINES:
            spectrum = spectrum + (4000.0 * strength * float(exposure)
                                   * np.exp(-0.5 * ((wavelength - centre) / sigma) ** 2))
        frame = np.repeat(spectrum[None, :], cfg.n_rows, axis=0)
        noisy = self.rng.poisson(np.clip(frame, 0, None)).astype(float)
        noisy = noisy + self.rng.normal(0.0, cfg.read_noise, frame.shape)
        variance = np.clip(frame, 0, None) + cfg.read_noise ** 2
        truth = {"wavelength": wavelength,
                 "lines": [c for c, _ in ARC_LINES]}
        return SpectrographFrame(noisy, variance, cfg, truth)

    def extracted(self, spectrum: Spectrum1D, redshift: float = 0.0,
                  snr: float = 20.0) -> Spectrum1D:
        """A 1-D spectrum straight to noise, skipping the detector.

        For tests that are about what happens *after* extraction, building
        the 2-D frame only adds a slow step and an unrelated failure mode.
        ``snr`` is the median signal-to-noise per pixel.
        """
        cfg = self.config
        wavelength = cfg.wavelength_at(np.arange(cfg.n_columns))
        flux = self._broadened(spectrum.redshifted(redshift), wavelength, 1.0)
        level = float(np.median(flux[flux > 0])) if np.any(flux > 0) else 1.0
        noise = level / max(float(snr), 1e-6)
        observed = flux + self.rng.normal(0.0, noise, flux.shape)
        return Spectrum1D(wavelength, observed, np.full_like(flux, noise),
                          meta={"redshift": float(redshift),
                                "template": dict(spectrum.meta)})

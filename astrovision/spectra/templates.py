"""Rest-frame spectra at spectroscopic resolution.

The photometric-redshift templates in :mod:`astrovision.photoz.templates` are
sampled at about 18 Angstroms per pixel at 5000 A, which is right for
integrating through a broad-band filter and useless here: every feature a
spectroscopic redshift keys on -- the Ca H and K doublet at 3934 and 3969, the
G band, Mg b, Na D, the Balmer series -- is narrower than one of those pixels.
So this module carries its own grid at 0.5 A, and its own features.

What the spectra need to have, for the code that consumes them to mean
anything:

* **Absorption lines of the right depth and width.**  A cross-correlation
  redshift is driven almost entirely by Ca H and K in an early-type galaxy;
  if those are painted too deep, the measured redshift error is optimistic in
  a way no amount of noise added afterwards will reveal.
* **Emission lines with realistic ratios.**  The diagnostic diagrams work on
  ratios, so a simulator that draws [O III]/H-beta and [N II]/H-alpha
  independently would let a classifier "succeed" on nonsense.  The ratios here
  come from an ionisation-parameter-like knob, so star-forming galaxies land
  on the star-forming locus and Seyferts land above it.
* **Broad features for supernovae.**  Type Ia is identified by Si II 6355 and
  the S II "W", Ib by He I, Ic by neither, II by hydrogen with a P-Cygni
  profile.  Those are the features a classifier must key on, so those are the
  features drawn -- at the right velocity widths, which are what makes them
  hard to confuse with a galaxy.

None of this is a spectral synthesis code.  It is the smallest set of
structures that makes the measurements below honest rather than circular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger

log = get_logger("spectra.templates")

#: Rest-frame grid, Angstroms.  0.5 A sampling over the optical: fine enough
#: that a 2 A resolution element is properly sampled, coarse enough that a
#: template library stays small.
REST_GRID = np.arange(3000.0, 10000.0, 0.5)

#: Rest wavelengths of the lines this package fits or measures, Angstroms.
#: Vacuum-to-air is ignored -- a 1.5 A offset common to every line is a
#: velocity zero point, not a redshift error, and nothing here measures
#: velocities to 100 km/s.
LINES: Dict[str, float] = {
    "[O II]": 3727.4,
    "Ca K": 3933.7,
    "Ca H": 3968.5,
    "H delta": 4101.7,
    "G band": 4304.4,
    "H gamma": 4340.5,
    "H beta": 4861.3,
    "[O III] 4959": 4958.9,
    "[O III] 5007": 5006.8,
    "Mg b": 5175.0,
    "Na D": 5892.5,
    "[O I] 6300": 6300.3,
    "H alpha": 6562.8,
    "[N II] 6548": 6548.0,
    "[N II] 6584": 6583.5,
    "[S II] 6716": 6716.4,
    "[S II] 6731": 6730.8,
}

#: The four lines the BPT diagram needs.  Kept separate because a spectrum
#: missing any one of them cannot be placed on the diagram at all, and the
#: code that says so needs to know which four to look for.
BPT_LINES = ("H beta", "[O III] 5007", "H alpha", "[N II] 6584")

#: Absorption features of an old stellar population: line, rest-frame
#: equivalent width in Angstroms at full strength, and Gaussian sigma.  The
#: widths are the intrinsic stellar-velocity-dispersion broadening of a giant
#: elliptical, around 200 km/s.
ABSORPTION: Sequence[Tuple[str, float, float]] = (
    ("Ca K", 14.0, 3.5),
    ("Ca H", 11.0, 3.5),
    ("H delta", 3.0, 3.0),
    ("G band", 6.0, 4.5),
    ("H gamma", 3.0, 3.0),
    ("H beta", 2.5, 3.2),
    ("Mg b", 5.0, 4.0),
    ("Na D", 4.0, 3.5),
)


def _gaussian(grid: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    """Unit-area Gaussian on ``grid``."""
    if sigma <= 0:
        return np.zeros_like(grid)
    return (np.exp(-0.5 * ((grid - centre) / sigma) ** 2)
            / (sigma * math.sqrt(2.0 * math.pi)))


def velocity_sigma(wavelength: float, velocity_km_s: float) -> float:
    """Gaussian sigma in Angstroms for a velocity width.

    Spectral features are broadened by motion, which is a *fractional*
    wavelength shift, so a line at 6563 A is physically wider in Angstroms
    than the same feature at 4861 A.  Drawing every line with one width in
    Angstroms would be drawing an instrument, not a galaxy.

    >>> round(velocity_sigma(5000.0, 300.0), 3)
    5.003
    """
    return float(wavelength) * float(velocity_km_s) / 299792.458


@dataclass
class Spectrum1D:
    """One extracted spectrum: wavelength, flux, and the error on the flux.

    The error array is not decoration.  Every measurement downstream -- a
    line flux, a redshift, a line ratio -- is only as meaningful as the noise
    estimate it was made against, and a spectrum that has lost its errors can
    still produce all of those numbers and none of their uncertainties.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None       # True where the pixel is unusable
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.wavelength.shape != self.flux.shape:
            raise ValueError("wavelength and flux must have the same shape")
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float)
            if self.error.shape != self.flux.shape:
                raise ValueError("error must match flux")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)

    def __len__(self) -> int:
        return int(self.wavelength.size)

    @property
    def good(self) -> np.ndarray:
        """Pixels that are finite and unmasked."""
        ok = np.isfinite(self.wavelength) & np.isfinite(self.flux)
        if self.mask is not None:
            ok &= ~self.mask
        return ok

    def snr(self) -> float:
        """Median signal-to-noise per pixel, or NaN without errors."""
        if self.error is None:
            return float("nan")
        ok = self.good & np.isfinite(self.error) & (self.error > 0)
        if not ok.any():
            return float("nan")
        return float(np.median(self.flux[ok] / self.error[ok]))

    def dispersion(self) -> float:
        """Median Angstroms per pixel."""
        if len(self) < 2:
            return float("nan")
        return float(np.median(np.diff(self.wavelength)))

    def slice(self, low: float, high: float) -> "Spectrum1D":
        """The part of the spectrum between two wavelengths."""
        keep = (self.wavelength >= float(low)) & (self.wavelength <= float(high))
        return Spectrum1D(
            self.wavelength[keep], self.flux[keep],
            None if self.error is None else self.error[keep],
            None if self.mask is None else self.mask[keep],
            dict(self.meta))

    def redshifted(self, z: float) -> "Spectrum1D":
        """The same spectrum as it would be observed at redshift ``z``.

        Wavelengths stretch by ``1 + z``.  The flux is left alone: every
        measurement here is a ratio or a position, and carrying a dimming
        factor would imply an absolute calibration these templates do not
        have.
        """
        return Spectrum1D(self.wavelength * (1.0 + float(z)), self.flux.copy(),
                          None if self.error is None else self.error.copy(),
                          None if self.mask is None else self.mask.copy(),
                          {**self.meta, "redshift": float(z)})

    def resample(self, grid: np.ndarray) -> "Spectrum1D":
        """Interpolate onto a new wavelength grid.

        Linear interpolation, not flux-conserving rebinning.  For the uses
        here -- cross-correlation and line fitting on a grid finer than the
        resolution element -- the difference is far below the noise, and a
        conserving rebin would need bin edges this class does not carry.
        """
        grid = np.asarray(grid, dtype=float)
        ok = self.good
        flux = np.interp(grid, self.wavelength[ok], self.flux[ok],
                         left=np.nan, right=np.nan)
        error = None
        if self.error is not None:
            error = np.interp(grid, self.wavelength[ok], self.error[ok],
                              left=np.nan, right=np.nan)
        return Spectrum1D(grid, flux, error, ~np.isfinite(flux), dict(self.meta))

    def to_dict(self) -> Dict[str, Any]:
        return {"n_pixels": len(self), "wavelength_range":
                [float(self.wavelength.min()), float(self.wavelength.max())]
                if len(self) else [float("nan"), float("nan")],
                "dispersion": self.dispersion(), "snr": self.snr(),
                "meta": dict(self.meta)}


def galaxy_spectrum(age_gyr: float = 8.0, emission: float = 0.0,
                    ionisation: float = 0.5, metallicity: float = 1.0,
                    velocity_dispersion: float = 200.0,
                    grid: Optional[np.ndarray] = None) -> Spectrum1D:
    """A galaxy at spectroscopic resolution.

    ``emission`` scales the line strengths; ``ionisation`` moves the line
    *ratios* along the sequence from a quiescent star-forming galaxy to a
    Seyfert.  Tying the ratios to one knob is what stops the simulator from
    drawing points that no physical galaxy occupies -- and therefore what
    makes a diagnostic-diagram test worth running.
    """
    wavelength = REST_GRID if grid is None else np.asarray(grid, dtype=float)
    age = float(np.clip(age_gyr, 0.05, 13.0))
    maturity = float(np.clip(math.log10(age / 0.05) / math.log10(260.0), 0.0, 1.0))

    # Continuum: blue and rising for a young population, red for an old one,
    # with the 4000 A break deepening as the hot stars die off.
    slope = -2.0 + 3.4 * maturity
    continuum = (wavelength / 5500.0) ** slope
    break_depth = 0.10 + 0.45 * maturity
    step = 0.5 * (1.0 + np.tanh((wavelength - 4000.0) / 120.0))
    continuum = continuum * (1.0 - break_depth * (1.0 - step))

    spectrum = continuum.copy()

    # Absorption.  Depth scales with the old population and with metallicity;
    # width with the velocity dispersion, which is why a cross-correlation
    # against a narrow-lined template loses signal on a giant elliptical.
    for name, equivalent_width, intrinsic in ABSORPTION:
        centre = LINES[name]
        sigma = math.hypot(intrinsic,
                           velocity_sigma(centre, velocity_dispersion))
        strength = equivalent_width * maturity * float(np.clip(metallicity, 0.2, 2.0))
        local = float(np.interp(centre, wavelength, continuum))
        spectrum = spectrum - local * strength * _gaussian(wavelength, centre, sigma)

    strength = float(max(emission, 0.0))
    if strength > 0:
        spectrum = spectrum + strength * _emission_lines(
            wavelength, continuum, ionisation, velocity_dispersion)

    spectrum = np.clip(spectrum, 1e-6, None)
    meta = {"kind": "galaxy", "age_gyr": age, "emission": strength,
            "ionisation": float(ionisation), "metallicity": float(metallicity),
            "velocity_dispersion": float(velocity_dispersion)}
    return Spectrum1D(wavelength, spectrum, meta=meta)


def _emission_lines(wavelength: np.ndarray, continuum: np.ndarray,
                    ionisation: float, velocity_dispersion: float) -> np.ndarray:
    """Nebular emission whose ratios follow one ionisation parameter.

    At ``ionisation`` 0 the ratios are those of a metal-rich star-forming
    galaxy -- strong H-alpha, weak [O III], [N II] comparable to H-alpha.  At
    1 they are those of a Seyfert: [O III]/H-beta above 10 and [N II]/H-alpha
    above 1.  A real galaxy is somewhere on that sequence, and so is every
    spectrum drawn here.
    """
    u = float(np.clip(ionisation, 0.0, 1.0))
    balmer = 2.86                       # H-alpha / H-beta, case B recombination
    o3_hb = 0.6 + 11.0 * u ** 2         # [O III] 5007 / H-beta
    n2_ha = 0.20 + 1.30 * u ** 1.5      # [N II] 6584 / H-alpha
    s2_ha = 0.20 + 0.45 * u
    o1_ha = 0.02 + 0.14 * u ** 2
    o2_hb = 2.2 - 1.0 * u

    # Widths: narrow-line-region gas is not as hot as the stars but does move,
    # and an AGN's narrow lines are broader than an H II region's.
    width_km_s = max(velocity_dispersion * (0.35 + 0.45 * u), 60.0)

    hbeta = 1.0
    strengths = {
        "H beta": hbeta,
        "H alpha": hbeta * balmer,
        "[O III] 5007": hbeta * o3_hb,
        "[O III] 4959": hbeta * o3_hb / 2.98,     # fixed by atomic physics
        "[N II] 6584": hbeta * balmer * n2_ha,
        "[N II] 6548": hbeta * balmer * n2_ha / 2.95,
        "[S II] 6716": hbeta * balmer * s2_ha * 0.55,
        "[S II] 6731": hbeta * balmer * s2_ha * 0.45,
        "[O I] 6300": hbeta * balmer * o1_ha,
        "[O II]": hbeta * o2_hb,
        "H gamma": hbeta * 0.47,
        "H delta": hbeta * 0.26,
    }
    total = np.zeros_like(wavelength)
    # Scale so H-beta has an equivalent width of about 10 A at unit strength,
    # measured against the local continuum, as a real one does.
    reference = float(np.interp(LINES["H beta"], wavelength, continuum))
    for name, relative in strengths.items():
        centre = LINES[name]
        sigma = velocity_sigma(centre, width_km_s)
        total = total + (reference * 10.0 * relative
                         * _gaussian(wavelength, centre, sigma))
    return total


def star_spectrum(temperature_class: str = "G",
                  grid: Optional[np.ndarray] = None) -> Spectrum1D:
    """A stellar spectrum, coarse but with the right lines in the right place.

    Stars matter here for one reason: they are the commonest thing a redshift
    code is handed by mistake, and a cross-correlation against galaxy
    templates will happily return a redshift for one.  Having them in the
    simulator is what lets that failure be measured instead of assumed away.
    """
    wavelength = REST_GRID if grid is None else np.asarray(grid, dtype=float)
    settings = {
        "O": (-3.0, (("H beta", 8.0, 6.0), ("H gamma", 7.0, 6.0),
                     ("H delta", 6.0, 6.0))),
        "A": (-1.5, (("H beta", 14.0, 5.0), ("H gamma", 13.0, 5.0),
                     ("H delta", 12.0, 5.0), ("Ca K", 2.0, 2.0))),
        "G": (0.4, (("Ca K", 12.0, 2.5), ("Ca H", 10.0, 2.5),
                    ("G band", 6.0, 4.0), ("Mg b", 4.0, 3.0),
                    ("Na D", 3.0, 2.5), ("H beta", 4.0, 3.0))),
        "M": (2.6, (("Na D", 8.0, 3.0), ("Ca K", 6.0, 3.0),
                    ("Mg b", 3.0, 3.0))),
    }
    slope, features = settings.get(temperature_class.upper(), settings["G"])
    continuum = (wavelength / 5500.0) ** slope
    spectrum = continuum.copy()
    for name, equivalent_width, sigma in features:
        centre = LINES[name]
        local = float(np.interp(centre, wavelength, continuum))
        spectrum = spectrum - local * equivalent_width * _gaussian(
            wavelength, centre, sigma)
    if temperature_class.upper() == "M":
        # TiO bands, the feature that actually identifies an M dwarf.
        for edge in (4955.0, 5450.0, 6160.0, 7050.0):
            spectrum = spectrum * (1.0 - 0.25 * np.exp(
                -0.5 * ((wavelength - edge - 90.0) / 70.0) ** 2))
    spectrum = np.clip(spectrum, 1e-6, None)
    return Spectrum1D(wavelength, spectrum,
                      meta={"kind": "star", "type": temperature_class.upper()})


def quasar_spectrum(grid: Optional[np.ndarray] = None) -> Spectrum1D:
    """A broad-line AGN: a power law with lines thousands of km/s wide."""
    wavelength = REST_GRID if grid is None else np.asarray(grid, dtype=float)
    continuum = (wavelength / 5500.0) ** (-1.5)
    spectrum = continuum.copy()
    broad = (("H beta", 60.0, 4000.0), ("H alpha", 200.0, 4000.0),
             ("H gamma", 25.0, 4000.0))
    for name, equivalent_width, velocity in broad:
        centre = LINES[name]
        local = float(np.interp(centre, wavelength, continuum))
        spectrum = spectrum + local * equivalent_width * _gaussian(
            wavelength, centre, velocity_sigma(centre, velocity))
    for name, equivalent_width in (("[O III] 5007", 30.0), ("[O III] 4959", 10.0),
                                   ("[O II]", 12.0)):
        centre = LINES[name]
        local = float(np.interp(centre, wavelength, continuum))
        spectrum = spectrum + local * equivalent_width * _gaussian(
            wavelength, centre, velocity_sigma(centre, 500.0))
    return Spectrum1D(wavelength, spectrum, meta={"kind": "quasar"})


#: Supernova spectral features by type: (rest wavelength of the absorption
#: minimum, depth as a fraction of the continuum, expansion velocity in km/s).
#: A P-Cygni profile is an absorption trough blueshifted from an emission
#: peak, so each entry is drawn as both.
SN_FEATURES: Dict[str, Sequence[Tuple[float, float, float]]] = {
    "Ia": ((6355.0, 0.55, 11000.0),      # Si II -- the defining feature
           (5972.0, 0.25, 10000.0),      # Si II
           (5454.0, 0.30, 10000.0),      # S II, the blue arm of the "W"
           (5640.0, 0.28, 10000.0),      # S II, the red arm
           (4130.0, 0.35, 12000.0),      # Si II
           (3860.0, 0.40, 13000.0)),     # Ca II H&K
    "Ib": ((5876.0, 0.45, 10000.0),      # He I -- the defining feature
           (6678.0, 0.25, 10000.0),      # He I
           (7065.0, 0.25, 10000.0),      # He I
           (4471.0, 0.30, 11000.0),      # He I
           (3860.0, 0.35, 12000.0)),     # Ca II
    "Ic": ((7774.0, 0.35, 12000.0),      # O I -- and no Si, no He
           (3860.0, 0.40, 13000.0),      # Ca II
           (5200.0, 0.25, 12000.0)),     # Fe II blend
    "II": ((6563.0, 0.60, 9000.0),       # H-alpha P-Cygni: the defining feature
           (4861.0, 0.40, 9000.0),       # H-beta
           (4341.0, 0.30, 9000.0),       # H-gamma
           (5876.0, 0.20, 8000.0)),      # He I, early only
}


def supernova_spectrum(sn_type: str = "Ia", phase_days: float = 0.0,
                       grid: Optional[np.ndarray] = None) -> Spectrum1D:
    """A supernova spectrum: a hot continuum under broad P-Cygni features.

    ``phase_days`` is days from maximum light.  It matters because the
    features move: the ejecta slow as the photosphere recedes, so an
    identification made with a template at the wrong phase is a
    cross-correlation against lines at the wrong wavelength.  Getting that
    dependence into the simulator is what makes the phase search in the
    classifier necessary rather than decorative.
    """
    wavelength = REST_GRID if grid is None else np.asarray(grid, dtype=float)
    phase = float(np.clip(phase_days, -15.0, 60.0))

    # A blackbody-ish continuum that cools with time.
    temperature = 12000.0 - 90.0 * max(phase, 0.0) - 200.0 * max(-phase, 0.0)
    temperature = float(np.clip(temperature, 4500.0, 15000.0))
    x = 1.4388e8 / (wavelength * temperature)          # h c / (lambda k T)
    continuum = 1.0 / (wavelength ** 5 * np.expm1(np.clip(x, 1e-6, 500.0)))
    continuum = continuum / continuum.max()

    spectrum = continuum.copy()
    # Velocities fall roughly as a power of time after explosion.
    slowdown = float(np.clip((20.0 + phase) / 20.0, 0.5, 3.0)) ** -0.25
    features = SN_FEATURES.get(sn_type, SN_FEATURES["Ia"])
    for rest, depth, velocity in features:
        speed = velocity * slowdown
        sigma = velocity_sigma(rest, 0.35 * speed)
        # Absorption, blueshifted by the expansion speed.
        trough = rest * (1.0 - speed / 299792.458)
        local = float(np.interp(trough, wavelength, continuum))
        spectrum = spectrum - (local * depth * sigma * math.sqrt(2 * math.pi)
                               * _gaussian(wavelength, trough, sigma))
        # The emission side of the P-Cygni profile, at rest, weaker.
        peak = float(np.interp(rest, wavelength, continuum))
        spectrum = spectrum + (peak * 0.35 * depth * 1.6 * sigma
                               * math.sqrt(2 * math.pi)
                               * _gaussian(wavelength, rest, 1.6 * sigma))
    spectrum = np.clip(spectrum, 1e-6, None)
    return Spectrum1D(wavelength, spectrum,
                      meta={"kind": "supernova", "type": sn_type,
                            "phase_days": phase})


#: The galaxy and star templates a redshift cross-correlation searches.  Kept
#: deliberately few and coarse: a library that contained every simulated
#: galaxy exactly would measure a lookup, not a redshift.
def standard_templates(grid: Optional[np.ndarray] = None) -> List[Spectrum1D]:
    """Templates for the redshift cross-correlation."""
    return [
        _named(galaxy_spectrum(10.0, emission=0.0, velocity_dispersion=250.0,
                               grid=grid), "early_type"),
        _named(galaxy_spectrum(4.0, emission=0.5, ionisation=0.25,
                               velocity_dispersion=170.0, grid=grid),
               "spiral"),
        _named(galaxy_spectrum(0.6, emission=2.0, ionisation=0.2,
                               velocity_dispersion=90.0, grid=grid),
               "starburst"),
        _named(quasar_spectrum(grid=grid), "quasar"),
        _named(star_spectrum("G", grid=grid), "star_G"),
        _named(star_spectrum("A", grid=grid), "star_A"),
        _named(star_spectrum("M", grid=grid), "star_M"),
    ]


def supernova_templates(grid: Optional[np.ndarray] = None,
                        phases: Sequence[float] = (-7.0, 0.0, 10.0, 25.0)
                        ) -> List[Spectrum1D]:
    """Supernova templates across type and phase.

    Both axes are searched, because the features move with phase; a library
    at maximum light alone would misidentify anything caught two weeks later.
    """
    templates: List[Spectrum1D] = []
    for sn_type in ("Ia", "Ib", "Ic", "II"):
        for phase in phases:
            spectrum = supernova_spectrum(sn_type, phase, grid=grid)
            templates.append(_named(spectrum, f"SN {sn_type} {phase:+.0f}d"))
    return templates


def _named(spectrum: Spectrum1D, name: str) -> Spectrum1D:
    spectrum.meta["name"] = name
    return spectrum

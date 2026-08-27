"""Synthetic sky generator.

The simulator renders physically-motivated fields -- stars with a seeing
PSF, Sersic galaxies with spiral arms and bars, nebulae, lensed arcs,
gradients, noise and detector artefacts -- together with a *truth table*.
That truth table is what lets the platform's detection, photometry,
morphology and transient stages be validated quantitatively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from ..core.numeric import convolve, gaussian_kernel
from ..io.image import AstroImage, ImageSeries
from ..io.wcs import SimpleWCS
from .profiles import (
    bar_pattern,
    einstein_arc,
    elliptical_radius,
    gaussian_psf,
    moffat_psf,
    sersic_profile,
    spiral_pattern,
    supersample,
)
from .sed import flux_ratios, object_colours, sed_colours

log = get_logger("simulate.sky")


@dataclass
class TruthObject:
    """Ground truth for one injected object."""

    id: int
    x: float
    y: float
    kind: str                      # star | galaxy | nebula | cluster | lens | transient
    flux: float
    morphology: str = "unresolved"
    r_eff: float = 0.0
    sersic_n: float = 0.0
    axis_ratio: float = 1.0
    position_angle: float = 0.0
    variable: bool = False
    period: float = 0.0
    amplitude: float = 0.0
    lensed: bool = False
    einstein_radius: float = 0.0
    anomalous: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "x": self.x, "y": self.y, "kind": self.kind,
            "flux": self.flux, "morphology": self.morphology, "r_eff": self.r_eff,
            "sersic_n": self.sersic_n, "axis_ratio": self.axis_ratio,
            "position_angle": self.position_angle, "variable": self.variable,
            "period": self.period, "amplitude": self.amplitude,
            "lensed": self.lensed, "einstein_radius": self.einstein_radius,
            "anomalous": self.anomalous, "meta": dict(self.meta),
        }


@dataclass
class SkyConfig:
    """Observing conditions and field content for :class:`SkySimulator`."""

    shape: Tuple[int, int] = (512, 512)
    seeing_fwhm: float = 3.2               # pixels
    psf: str = "moffat"                    # moffat | gaussian
    background: float = 120.0              # counts / pixel
    background_gradient: float = 0.06      # fractional across the field
    #: Fractional growth of the seeing width from the optical axis to the
    #: field corner.  0.2 means the corners are 20% blurrier than the centre,
    #: which is a mild but entirely typical amount for a wide-field camera.
    seeing_variation: float = 0.0
    optical_axis: Optional[Tuple[float, float]] = None   # defaults to the centre
    read_noise: float = 5.0
    gain: float = 2.0                      # e-/count, sets Poisson scaling
    saturation: float = 60_000.0
    n_stars: int = 220
    n_galaxies: int = 45
    n_nebulae: int = 2
    n_clusters: int = 1
    n_lenses: int = 1
    n_anomalies: int = 2
    star_flux_range: Tuple[float, float] = (200.0, 90_000.0)
    galaxy_flux_range: Tuple[float, float] = (900.0, 60_000.0)
    variable_fraction: float = 0.06
    cosmic_ray_rate: float = 4e-5          # fraction of pixels hit
    bad_column_count: int = 1
    pixel_scale: float = 0.4               # arcsec / pixel
    field_centre: Tuple[float, float] = (150.0, 2.2)   # ra, dec in degrees
    band: str = "r"
    zero_point: float = 25.0
    seed: int = 42


class SkySimulator:
    """Render synthetic astronomical fields with an accompanying truth table.

    >>> sim = SkySimulator(SkyConfig(shape=(128, 128), n_stars=20, n_galaxies=3))
    >>> image, truth = sim.generate()
    >>> image.shape
    (128, 128)
    """

    def __init__(self, config: Optional[SkyConfig] = None):
        self.config = config or SkyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        # Detector effects draw from their own stream so a field can be
        # re-rendered with identical sources but independent noise -- which
        # is exactly what a second filter of the same sky looks like.  It
        # defaults to the source stream, leaving single-band output
        # bit-for-bit as it was.
        self._noise_rng = self.rng
        self._next_id = 1

    # -- helpers -----------------------------------------------------------
    def _new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def local_fwhm(self, x: float, y: float) -> float:
        """Seeing width at a field position.

        Real optics degrade off-axis, so a PSF measured at the centre does
        not describe the corners.  The model here is the simple one that
        matches what a telescope actually does: the width grows with the
        square of the distance from the optical axis, since defocus and
        field curvature are both second-order in field angle.
        """
        cfg = self.config
        if not cfg.seeing_variation:
            return float(cfg.seeing_fwhm)
        ny, nx = cfg.shape
        ox = cfg.optical_axis[0] if cfg.optical_axis else (nx - 1) / 2.0
        oy = cfg.optical_axis[1] if cfg.optical_axis else (ny - 1) / 2.0
        half_diagonal = 0.5 * math.hypot(nx, ny)
        radius = math.hypot(float(x) - ox, float(y) - oy) / max(half_diagonal, 1e-9)
        return float(cfg.seeing_fwhm * (1.0 + cfg.seeing_variation * radius ** 2))

    def _psf_stamp(self, shape, centre, amplitude: float,
                   position: Optional[Tuple[float, float]] = None) -> np.ndarray:
        cfg = self.config
        fwhm = self.local_fwhm(*position) if position is not None else cfg.seeing_fwhm
        if cfg.psf == "gaussian":
            return gaussian_psf(shape, centre, fwhm, amplitude)
        return moffat_psf(shape, centre, fwhm, 3.2, amplitude)

    def _psf_kernel(self, size: int = 0,
                    position: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Normalised convolution kernel matching the configured seeing.

        Extended objects must be blurred by the *same* PSF as the stars,
        or the PSF model the pipeline measures from those stars will not
        describe the galaxies -- which quietly biases every profile fit.
        """
        cfg = self.config
        if not size:
            size = int(2 * np.ceil(4.0 * cfg.seeing_fwhm * (1.0 + cfg.seeing_variation)) + 1)
        size = max(5, int(size) | 1)
        centre = ((size - 1) / 2.0, (size - 1) / 2.0)
        kernel = self._psf_stamp((size, size), centre, 1.0, position)
        total = float(kernel.sum())
        return kernel / total if total > 0 else kernel

    def _stamp_bounds(self, x: float, y: float, half: int):
        """Return the image slice and the stamp-local centre for an object."""
        ny, nx = self.config.shape
        x0 = int(max(0, np.floor(x) - half))
        x1 = int(min(nx, np.ceil(x) + half + 1))
        y0 = int(max(0, np.floor(y) - half))
        y1 = int(min(ny, np.ceil(y) + half + 1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (slice(y0, y1), slice(x0, x1)), (x - x0, y - y0), (y1 - y0, x1 - x0)

    # -- object renderers --------------------------------------------------
    def add_star(self, canvas: np.ndarray, x: float, y: float, flux: float,
                 variable: bool = False) -> TruthObject:
        """Render a point source convolved with the seeing PSF."""
        half = int(np.ceil(6 * self.local_fwhm(x, y)))
        bounds = self._stamp_bounds(x, y, half)
        if bounds is not None:
            region, centre, shape = bounds
            stamp = self._psf_stamp(shape, centre, 1.0, position=(x, y))
            total = stamp.sum()
            if total > 0:
                canvas[region] += stamp * (flux / total)
        truth = TruthObject(self._new_id(), x, y, "star", flux, morphology="unresolved")
        if variable:
            truth.variable = True
            truth.period = float(self.rng.uniform(0.05, 12.0))
            truth.amplitude = float(self.rng.uniform(0.08, 0.6))
        return truth

    def add_mover(self, canvas: np.ndarray, x: float, y: float, flux: float,
                  trail_length: float = 0.0, trail_angle: float = 0.0) -> TruthObject:
        """Render a moving object, trailed by its motion during the exposure.

        A trailed source is not a stretched point source: it is the PSF
        *integrated along the track the object took while the shutter was
        open*.  Rendering it as a sum of PSF stamps along that path is
        therefore exact rather than an approximation, and it reproduces the
        property detection relies on -- a trail has the PSF's profile across
        its width and a flat top along its length, which no elongated galaxy
        does.

        ``trail_length`` is in pixels; below about half the seeing width the
        object is indistinguishable from a point source and one stamp is
        rendered instead of many.
        """
        length = float(max(trail_length, 0.0))
        angle = math.radians(float(trail_angle))
        if length < 0.5 * self.config.seeing_fwhm:
            self.add_star(canvas, x, y, flux)
            self._next_id -= 1
            return TruthObject(self._new_id(), x, y, "mover", flux,
                               morphology="unresolved",
                               meta={"trail_length": length, "trail_angle": trail_angle,
                                     "trailed": False})
        # One sample per half pixel keeps the trail smooth; fewer leaves it
        # visibly beaded, which would be an artefact of the simulator rather
        # than of the sky.
        n_samples = max(2, int(np.ceil(2.0 * length)))
        for step in range(n_samples):
            offset = (step / (n_samples - 1) - 0.5) * length
            self.add_star(canvas, x + offset * np.cos(angle), y + offset * np.sin(angle),
                          flux / n_samples)
            self._next_id -= 1
        return TruthObject(self._new_id(), x, y, "mover", flux, morphology="trailed",
                           meta={"trail_length": length, "trail_angle": trail_angle,
                                 "trailed": True})

    def add_galaxy(self, canvas: np.ndarray, x: float, y: float, flux: float,
                   morphology: Optional[str] = None) -> TruthObject:
        """Render a galaxy of the requested Hubble type.

        Each type is built from the components that actually define it, so
        the classes are structurally distinct rather than differently
        labelled: an elliptical is a single high-index spheroid; a
        lenticular is a prominent bulge inside a smooth, flattened
        exponential disc with no arms; a spiral is a disc-dominated system
        carrying an arm pattern; a barred spiral adds the bar; an irregular
        is a set of clumps with no ordered structure.
        """
        cfg = self.config
        rng = self.rng
        morphology = morphology or str(rng.choice(
            ["spiral", "barred_spiral", "elliptical", "lenticular", "irregular"],
            p=[0.34, 0.16, 0.28, 0.12, 0.10]))

        # Structural parameters per Hubble type.
        if morphology == "elliptical":
            sersic_n = float(rng.uniform(3.5, 6.0))
            r_eff = float(rng.uniform(3.5, 9.0))
            axis_ratio = float(rng.uniform(0.62, 1.0))
            bulge_fraction = 1.0
        elif morphology == "lenticular":
            sersic_n = float(rng.uniform(0.9, 1.3))     # the disc component
            r_eff = float(rng.uniform(5.0, 11.0))
            axis_ratio = float(rng.uniform(0.28, 0.55))  # S0 discs are flat
            bulge_fraction = float(rng.uniform(0.45, 0.65))
        elif morphology == "irregular":
            sersic_n = float(rng.uniform(0.5, 1.1))
            r_eff = float(rng.uniform(3.5, 9.0))
            axis_ratio = float(rng.uniform(0.4, 0.9))
            bulge_fraction = 0.0
        else:                                            # spiral / barred spiral
            sersic_n = float(rng.uniform(0.8, 1.4))
            r_eff = float(rng.uniform(5.0, 12.0))
            axis_ratio = float(rng.uniform(0.45, 0.95))
            bulge_fraction = float(rng.uniform(0.05, 0.25))
        pa = float(rng.uniform(0, 180))

        half = int(np.ceil(max(5.0 * r_eff, 3 * cfg.seeing_fwhm)))
        bounds = self._stamp_bounds(x, y, half)
        if bounds is None:
            return TruthObject(self._new_id(), x, y, "galaxy", flux, morphology)
        region, centre, shape = bounds

        # Supersample: a cuspy Sersic profile is badly biased if the
        # analytic form is merely point-sampled at pixel centres.
        disc = supersample(shape, centre, lambda s, c: sersic_profile(
            elliptical_radius(s, c, axis_ratio, pa), 1.0,
            r_eff * (s[0] / shape[0]), sersic_n), factor=3)
        disc = disc / max(disc.sum(), 1e-12)

        arms = 0
        if morphology in ("spiral", "barred_spiral"):
            arms = int(rng.choice([2, 2, 2, 3, 4]))
            disc = disc * spiral_pattern(
                shape, centre, r_eff, arms, float(rng.uniform(12, 32)),
                float(rng.uniform(0.4, 0.75)), axis_ratio, pa,
                float(rng.uniform(0, 2 * np.pi)))
        elif morphology == "irregular":
            for _ in range(int(rng.integers(3, 7))):
                ox = centre[0] + float(rng.normal(0, r_eff * 0.6))
                oy = centre[1] + float(rng.normal(0, r_eff * 0.6))
                disc = disc + gaussian_psf(shape, (ox, oy), r_eff * 0.5,
                                           float(rng.uniform(0.3, 1.0)) * disc.max())

        stamp = disc / max(disc.sum(), 1e-12) * (1.0 - bulge_fraction)

        if bulge_fraction > 0:
            bulge_r_eff = r_eff * (1.0 if morphology == "elliptical" else 0.25)
            bulge_q = 1.0 if morphology == "elliptical" else float(rng.uniform(0.75, 1.0))
            bulge_n = sersic_n if morphology == "elliptical" else 3.5
            bulge = supersample(shape, centre, lambda s, c: sersic_profile(
                elliptical_radius(s, c, bulge_q if morphology != "elliptical" else axis_ratio,
                                  pa), 1.0,
                bulge_r_eff * (s[0] / shape[0]), bulge_n), factor=3)
            stamp = stamp + bulge / max(bulge.sum(), 1e-12) * bulge_fraction

        if morphology == "barred_spiral":
            stamp = stamp + stamp.max() * bar_pattern(
                shape, centre, r_eff * 0.8, r_eff * 0.2, pa,
                float(rng.uniform(0.35, 0.7)))

        stamp = convolve(stamp, self._psf_kernel())
        total = stamp.sum()
        if total > 0:
            canvas[region] += stamp * (flux / total)

        return TruthObject(self._new_id(), x, y, "galaxy", flux, morphology,
                           r_eff=r_eff, sersic_n=sersic_n, axis_ratio=axis_ratio,
                           position_angle=pa,
                           meta={"arms": arms, "bulge_fraction": bulge_fraction})

    def add_nebula(self, canvas: np.ndarray, x: float, y: float,
                   flux: float) -> TruthObject:
        """Render a diffuse, filamentary emission region."""
        rng = self.rng
        scale = float(rng.uniform(14.0, 34.0))
        half = int(np.ceil(3.2 * scale))
        bounds = self._stamp_bounds(x, y, half)
        if bounds is None:
            return TruthObject(self._new_id(), x, y, "nebula", flux, "diffuse")
        region, centre, shape = bounds
        # Smoothed noise gives a filamentary texture; the envelope keeps it local.
        texture = convolve(rng.normal(0, 1, shape), gaussian_kernel(scale * 0.18))
        texture = np.clip(texture - texture.mean(), 0, None)
        radius = elliptical_radius(shape, centre, float(rng.uniform(0.5, 1.0)),
                                   float(rng.uniform(0, 180)))
        stamp = np.exp(-0.5 * (radius / scale) ** 2) * (0.45 + texture / (texture.max() or 1.0))
        stamp = convolve(stamp, self._psf_kernel())
        total = stamp.sum()
        if total > 0:
            canvas[region] += stamp * (flux / total)
        return TruthObject(self._new_id(), x, y, "nebula", flux, "diffuse",
                           r_eff=scale, meta={"extended": True})

    def add_star_cluster(self, canvas: np.ndarray, x: float, y: float,
                         flux: float, n_members: int = 40) -> TruthObject:
        """Render a compact, centrally-concentrated group of stars."""
        rng = self.rng
        radius = float(rng.uniform(8.0, 22.0))
        # King-like concentration: members are drawn from a truncated profile.
        for _ in range(int(n_members)):
            r = abs(float(rng.normal(0, radius * 0.45)))
            angle = float(rng.uniform(0, 2 * np.pi))
            member_flux = flux / n_members * float(rng.uniform(0.4, 2.4))
            self.add_star(canvas, x + r * np.cos(angle), y + r * np.sin(angle), member_flux)
            self._next_id -= 1  # members are not separate truth entries
        return TruthObject(self._new_id(), x, y, "cluster", flux, "cluster",
                           r_eff=radius, meta={"n_members": int(n_members)})

    def add_lens_system(self, canvas: np.ndarray, x: float, y: float,
                        flux: float, arc_scale: float = 1.0) -> TruthObject:
        """Render an elliptical deflector surrounded by lensed arcs.

        ``arc_scale`` brightens or dims the arcs relative to the deflector
        without touching either one's shape.  The deflector is an old red
        elliptical and the arcs are a lensed star-forming galaxy behind it,
        so in a multi-band field the two carry different colours -- which is
        the single most useful discriminator a real lens search has.
        """
        rng = self.rng
        theta_e = float(rng.uniform(7.0, 15.0))
        r_eff = theta_e * float(rng.uniform(0.45, 0.75))
        half = int(np.ceil(3.4 * theta_e))
        bounds = self._stamp_bounds(x, y, half)
        if bounds is None:
            return TruthObject(self._new_id(), x, y, "lens", flux, "elliptical")
        region, centre, shape = bounds

        pa = float(rng.uniform(0, 180))
        deflector_q = float(rng.uniform(0.7, 1.0))
        deflector = supersample(shape, centre, lambda s, c: sersic_profile(
            elliptical_radius(s, c, deflector_q, pa), 1.0,
            r_eff * (s[0] / shape[0]), 4.0), factor=3)
        deflector *= 0.72 * flux / max(deflector.sum(), 1e-9)

        arcs = np.zeros(shape, dtype=float)
        n_arcs = int(rng.integers(2, 5))
        base_pa = float(rng.uniform(0, 360))
        for k in range(n_arcs):
            arcs += einstein_arc(
                shape, centre, theta_e * float(rng.uniform(0.92, 1.08)),
                float(rng.uniform(1.0, 2.2)),
                float(rng.uniform(45, 110)),
                base_pa + 360.0 * k / n_arcs + float(rng.uniform(-18, 18)),
                float(rng.uniform(0.5, 1.0)))
        if arcs.sum() > 0:
            arcs *= 0.28 * flux * float(arc_scale) / arcs.sum()

        stamp = convolve(deflector + arcs, self._psf_kernel())
        canvas[region] += stamp
        return TruthObject(self._new_id(), x, y, "lens", flux, "elliptical",
                           r_eff=r_eff, lensed=True, einstein_radius=theta_e,
                           meta={"n_arcs": n_arcs})

    def add_anomaly(self, canvas: np.ndarray, x: float, y: float,
                    flux: float) -> TruthObject:
        """Render an object that matches no standard morphological class.

        These exist so the anomaly engine has a recoverable target: the
        shapes are deliberately outside the trained galaxy/star manifold.
        """
        rng = self.rng
        kind = str(rng.choice(["ring", "double_nucleus", "jet", "cross"]))
        scale = float(rng.uniform(7.0, 16.0))
        half = int(np.ceil(3.5 * scale))
        bounds = self._stamp_bounds(x, y, half)
        if bounds is None:
            return TruthObject(self._new_id(), x, y, "anomaly", flux, kind, anomalous=True)
        region, centre, shape = bounds
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        dx, dy = xx - centre[0], yy - centre[1]
        r = np.hypot(dx, dy)

        if kind == "ring":
            stamp = np.exp(-0.5 * ((r - scale) / (scale * 0.16)) ** 2)
        elif kind == "double_nucleus":
            offset = scale * 0.55
            angle = float(rng.uniform(0, np.pi))
            ox, oy = offset * np.cos(angle), offset * np.sin(angle)
            stamp = (np.exp(-0.5 * (((dx - ox) ** 2 + (dy - oy) ** 2) / (scale * 0.3) ** 2)) +
                     np.exp(-0.5 * (((dx + ox) ** 2 + (dy + oy) ** 2) / (scale * 0.3) ** 2)))
            stamp += 0.35 * np.exp(-0.5 * (r / (scale * 0.9)) ** 2)
        elif kind == "jet":
            angle = float(rng.uniform(0, np.pi))
            xr = dx * np.cos(angle) + dy * np.sin(angle)
            yr = -dx * np.sin(angle) + dy * np.cos(angle)
            stamp = np.exp(-0.5 * (r / (scale * 0.35)) ** 2)
            stamp += 0.6 * np.exp(-0.5 * ((xr - scale) / (scale * 0.8)) ** 2
                                  - 0.5 * (yr / (scale * 0.12)) ** 2)
        else:  # cross / X-shaped
            angle = float(rng.uniform(0, np.pi))
            xr = dx * np.cos(angle) + dy * np.sin(angle)
            yr = -dx * np.sin(angle) + dy * np.cos(angle)
            envelope = np.exp(-0.5 * (r / scale) ** 2)
            stamp = envelope * (np.exp(-0.5 * (yr / (scale * 0.12)) ** 2) +
                                np.exp(-0.5 * (xr / (scale * 0.12)) ** 2))

        stamp = convolve(stamp, self._psf_kernel())
        total = stamp.sum()
        if total > 0:
            canvas[region] += stamp * (flux / total)
        return TruthObject(self._new_id(), x, y, "anomaly", flux, kind,
                           r_eff=scale, anomalous=True, meta={"anomaly_kind": kind})

    # -- detector effects --------------------------------------------------
    def _background_map(self) -> np.ndarray:
        cfg = self.config
        ny, nx = cfg.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        gx = float(self._noise_rng.uniform(-1, 1)) * cfg.background_gradient
        gy = float(self._noise_rng.uniform(-1, 1)) * cfg.background_gradient
        plane = 1.0 + gx * (xx / max(nx - 1, 1) - 0.5) + gy * (yy / max(ny - 1, 1) - 0.5)
        return cfg.background * plane

    def _add_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Cosmic rays and bad columns -- the artefacts vetting must reject."""
        cfg = self.config
        rng = self._noise_rng
        ny, nx = cfg.shape
        record: Dict[str, Any] = {"cosmic_rays": [], "bad_columns": []}

        n_rays = int(cfg.cosmic_ray_rate * ny * nx)
        for _ in range(max(0, n_rays)):
            cx = int(rng.integers(2, nx - 2))
            cy = int(rng.integers(2, ny - 2))
            amplitude = float(rng.uniform(500, 20_000))
            length = int(rng.integers(1, 4))
            angle = float(rng.uniform(0, np.pi))
            for step in range(length):
                px = int(round(cx + step * np.cos(angle)))
                py = int(round(cy + step * np.sin(angle)))
                if 0 <= px < nx and 0 <= py < ny:
                    image[py, px] += amplitude
            record["cosmic_rays"].append({"x": cx, "y": cy, "amplitude": amplitude})

        for _ in range(max(0, int(cfg.bad_column_count))):
            col = int(rng.integers(0, nx))
            image[:, col] += float(rng.uniform(-0.4, 0.8)) * cfg.background
            record["bad_columns"].append(col)
        return record

    def _apply_noise(self, signal: np.ndarray) -> np.ndarray:
        """Poisson photon noise plus Gaussian read noise, then saturation."""
        cfg = self.config
        electrons = np.clip(signal, 0, None) * cfg.gain
        rng = self._noise_rng
        noisy = rng.poisson(np.clip(electrons, 0, 1e12)).astype(float) / cfg.gain
        noisy += rng.normal(0.0, cfg.read_noise, size=signal.shape)
        return np.clip(noisy, None, cfg.saturation)

    # -- top-level generation ---------------------------------------------
    def generate(self, inject: Optional[Sequence[TruthObject]] = None,
                 flux_scale: Optional[Mapping[int, float]] = None,
                 arc_scale: Optional[Mapping[int, float]] = None
                 ) -> Tuple[AstroImage, List[TruthObject]]:
        """Generate one field; returns the image and its truth table.

        ``flux_scale`` multiplies individual objects' fluxes, keyed by the
        truth id they are about to be given.  Because object ids are handed
        out in a fixed order, re-seeding the simulator and passing a scale
        map re-renders *the same field* at different brightnesses -- which is
        how :meth:`generate_multiband` produces several filters of one sky.
        ``arc_scale`` does the same for a lens system's arcs alone, letting
        them carry a colour of their own.
        """
        cfg = self.config
        rng = self.rng

        def scale_for(default: float = 1.0) -> float:
            # `_next_id` is the id the *next* object created will receive.
            return float(flux_scale.get(self._next_id, default)) if flux_scale else default
        ny, nx = cfg.shape
        margin = 6
        canvas = np.zeros(cfg.shape, dtype=float)
        truth: List[TruthObject] = []

        def position() -> Tuple[float, float]:
            return (float(rng.uniform(margin, nx - margin)),
                    float(rng.uniform(margin, ny - margin)))

        def log_flux(bounds: Tuple[float, float]) -> float:
            # A log-uniform draw approximates real number counts far better
            # than a uniform one: faint objects vastly outnumber bright ones.
            low, high = np.log10(bounds[0]), np.log10(bounds[1])
            return float(10 ** rng.uniform(low, high))

        for _ in range(cfg.n_galaxies):
            x, y = position()
            truth.append(self.add_galaxy(canvas, x, y,
                                         log_flux(cfg.galaxy_flux_range) * scale_for()))
        for _ in range(cfg.n_nebulae):
            x, y = position()
            truth.append(self.add_nebula(canvas, x, y, log_flux((5e3, 8e4)) * scale_for()))
        for _ in range(cfg.n_clusters):
            x, y = position()
            truth.append(self.add_star_cluster(canvas, x, y,
                                               log_flux((2e4, 1.2e5)) * scale_for()))
        for _ in range(cfg.n_lenses):
            x, y = position()
            arcs = float(arc_scale.get(self._next_id, 1.0)) if arc_scale else 1.0
            truth.append(self.add_lens_system(canvas, x, y,
                                              log_flux((2e4, 9e4)) * scale_for(),
                                              arc_scale=arcs))
        for _ in range(cfg.n_anomalies):
            x, y = position()
            truth.append(self.add_anomaly(canvas, x, y, log_flux((6e3, 5e4)) * scale_for()))
        for _ in range(cfg.n_stars):
            x, y = position()
            variable = bool(rng.random() < cfg.variable_fraction)
            truth.append(self.add_star(canvas, x, y,
                                       log_flux(cfg.star_flux_range) * scale_for(), variable))

        for obj in inject or []:
            if obj.kind == "star":
                self.add_star(canvas, obj.x, obj.y, obj.flux)
                self._next_id -= 1
            truth.append(obj)

        signal = canvas + self._background_map()
        artifacts = self._add_artifacts(signal)
        pixels = self._apply_noise(signal)

        wcs = SimpleWCS.tangent(cfg.field_centre[0], cfg.field_centre[1],
                                cfg.shape, cfg.pixel_scale)
        image = AstroImage(
            data=pixels,
            header={"OBJECT": "SYNTHETIC-FIELD", "FILTER": cfg.band,
                    "GAIN": cfg.gain, "SATURATE": cfg.saturation,
                    "MAGZP": cfg.zero_point, "SIMULATE": True},
            wcs=wcs, name="synthetic_field", band=cfg.band,
            exposure_time=300.0, mjd=59000.0,
            meta={"simulated": True, "artifacts": artifacts,
                  "seeing_fwhm": cfg.seeing_fwhm,
                  "truth_count": len(truth)},
        )
        log.info("generated %dx%d synthetic field with %d truth objects",
                 nx, ny, len(truth))
        return image, truth

    def generate_multiband(self, bands: Sequence[str] = ("g", "r", "i"),
                           seeing: Optional[Mapping[str, float]] = None,
                           background: Optional[Mapping[str, float]] = None,
                           zero_point: Optional[Mapping[str, float]] = None,
                           redshift: float = 0.15,
                           redshift_range: Optional[Tuple[float, float]] = None,
                           ) -> Tuple[Dict[str, AstroImage], List[TruthObject]]:
        """Render the same sky through several filters.

        Every band shows the *same objects in the same places*; what changes
        is each object's brightness, set by the colours in :mod:`sed`, and
        the observing conditions -- seeing, sky level and zero point are all
        band-dependent in reality and are here too.  Noise is drawn
        independently per band, because two exposures of one field are two
        separate realisations of the detector.

        Returns a band-keyed dict of images and one truth table, where each
        object carries ``meta["magnitudes"]`` (offsets from the ``r`` band)
        and ``meta["band_flux"]``.

        >>> sim = SkySimulator(SkyConfig(shape=(96, 96), n_stars=8, n_galaxies=2,
        ...                              n_nebulae=0, n_clusters=0, n_lenses=0,
        ...                              n_anomalies=0))
        >>> images, truth = sim.generate_multiband(("g", "r"))
        >>> sorted(images)
        ['g', 'r']
        """
        bands = tuple(dict.fromkeys(bands))
        if not bands:
            raise ValueError("at least one band is required")
        cfg = self.config
        seed = int(cfg.seed)

        # Pass one establishes the truth table.  Its image is discarded: the
        # bands are all rendered below under identical conditions, so that no
        # single filter is privileged by having been drawn first.
        self.rng = np.random.default_rng(seed)
        self._noise_rng = self.rng
        self._next_id = 1
        _, truth = self.generate()

        colour_rng = np.random.default_rng(seed + 104729)
        offsets: Dict[int, Dict[str, float]] = {}
        arc_offsets: Dict[int, Dict[str, float]] = {}
        for obj in truth:
            if redshift_range is not None and obj.kind == "galaxy":
                # Each galaxy gets its own redshift and its own drawn
                # spectrum, integrated through the filters.  This is the path
                # that makes a photometric-redshift test mean something.
                obj_z = float(colour_rng.uniform(*redshift_range))
                offsets[obj.id], sed_truth = sed_colours(
                    obj.kind, bands, obj_z, colour_rng)
                obj.meta.update(sed_truth)
            else:
                offsets[obj.id] = object_colours(
                    obj.kind, obj.morphology, rng=colour_rng,
                    redshift=redshift if obj.kind == "galaxy" else 0.0)
            obj.meta["magnitudes"] = dict(offsets[obj.id])
            if obj.kind == "lens":
                # The deflector is the elliptical the object's colour
                # describes; the arcs are a separate, bluer galaxy behind it.
                arc_offsets[obj.id] = object_colours("lens_arc", rng=colour_rng)
                obj.meta["arc_magnitudes"] = dict(arc_offsets[obj.id])

        images: Dict[str, AstroImage] = {}
        saved = (cfg.seeing_fwhm, cfg.background, cfg.zero_point, cfg.band)
        try:
            for index, band in enumerate(bands):
                ratios = {oid: flux_ratios(value).get(band, 1.0)
                          for oid, value in offsets.items()}
                arcs = {oid: (flux_ratios(value).get(band, 1.0) /
                              max(ratios.get(oid, 1.0), 1e-9))
                        for oid, value in arc_offsets.items()}
                cfg.seeing_fwhm = float((seeing or {}).get(band, saved[0]))
                cfg.background = float((background or {}).get(band, saved[1]))
                cfg.zero_point = float((zero_point or {}).get(band, saved[2]))
                cfg.band = band

                self.rng = np.random.default_rng(seed)
                self._noise_rng = np.random.default_rng(seed + 7919 * (index + 1))
                self._next_id = 1
                image, band_truth = self.generate(flux_scale=ratios, arc_scale=arcs)
                image.name = f"synthetic_{band}"
                image.meta["band"] = band
                images[band] = image
                for obj, rendered in zip(truth, band_truth):
                    obj.meta.setdefault("band_flux", {})[band] = float(rendered.flux)
        finally:
            cfg.seeing_fwhm, cfg.background, cfg.zero_point, cfg.band = saved
            self._noise_rng = self.rng

        log.info("generated %d-band field (%s) with %d truth objects",
                 len(bands), ", ".join(bands), len(truth))
        return images, truth

    def generate_series(self, n_epochs: int = 5, cadence: float = 2.0,
                        n_transients: int = 2, transient_kind: str = "supernova",
                        n_movers: int = 0,
                        mover_rate_range: Tuple[float, float] = (15.0, 90.0)
                        ) -> Tuple[ImageSeries, List[TruthObject], List[Dict[str, Any]]]:
        """Generate a multi-epoch series with injected transients.

        Returns ``(series, static_truth, transient_truth)``.  Every epoch
        shares the same static sky so difference imaging has a genuine
        template, while transients and variable stars change with time.

        ``n_movers`` injects solar-system objects, whose truth records are
        appended to the transient list with ``kind="mover"``.  Their rates
        are drawn from ``mover_rate_range`` in **arcseconds per hour**, which
        is the unit the sky works in: a main-belt asteroid near opposition
        moves at roughly 30 arcsec/hour and a near-Earth object far faster.

        The unit matters for the cadence too.  At 30 arcsec/hour an object
        crosses a 2-arcminute field in four hours, so a series taken two days
        apart never sees the same asteroid twice -- which is why asteroid
        linking is done *within a night*.  Pass a cadence of a fraction of a
        day when injecting movers.
        """
        cfg = self.config
        rng = self.rng
        base_seed = cfg.seed

        # Render the static sky once, without noise, so all epochs match.
        static_sim = SkySimulator(SkyConfig(**{**cfg.__dict__, "seed": base_seed}))
        static_canvas = np.zeros(cfg.shape, dtype=float)
        static_truth: List[TruthObject] = []
        ny, nx = cfg.shape
        margin = 8

        def position():
            return (float(static_sim.rng.uniform(margin, nx - margin)),
                    float(static_sim.rng.uniform(margin, ny - margin)))

        def log_flux(bounds):
            low, high = np.log10(bounds[0]), np.log10(bounds[1])
            return float(10 ** static_sim.rng.uniform(low, high))

        for _ in range(cfg.n_galaxies):
            static_truth.append(static_sim.add_galaxy(static_canvas, *position(),
                                                      log_flux(cfg.galaxy_flux_range)))
        for _ in range(cfg.n_lenses):
            static_truth.append(static_sim.add_lens_system(static_canvas, *position(),
                                                           log_flux((2e4, 9e4))))
        for _ in range(cfg.n_anomalies):
            static_truth.append(static_sim.add_anomaly(static_canvas, *position(),
                                                       log_flux((6e3, 5e4))))

        variable_stars: List[TruthObject] = []
        for _ in range(cfg.n_stars):
            x, y = position()
            flux = log_flux(cfg.star_flux_range)
            is_variable = bool(static_sim.rng.random() < cfg.variable_fraction)
            star = TruthObject(static_sim._new_id(), x, y, "star", flux,
                               morphology="unresolved")
            if is_variable:
                star.variable = True
                star.period = float(static_sim.rng.uniform(0.5, 8.0))
                star.amplitude = float(static_sim.rng.uniform(0.15, 0.55))
                variable_stars.append(star)
            else:
                static_sim.add_star(static_canvas, x, y, flux)
                static_sim._next_id -= 1
            static_truth.append(star)

        # Transients: choose hosts among galaxies where possible.
        galaxies = [t for t in static_truth if t.kind == "galaxy"]
        transients: List[Dict[str, Any]] = []
        for k in range(max(0, int(n_transients))):
            if galaxies and rng.random() < 0.75:
                host = galaxies[int(rng.integers(0, len(galaxies)))]
                offset = float(rng.uniform(1.5, max(2.0, host.r_eff * 1.6)))
                angle = float(rng.uniform(0, 2 * np.pi))
                tx = float(np.clip(host.x + offset * np.cos(angle), margin, nx - margin))
                ty = float(np.clip(host.y + offset * np.sin(angle), margin, ny - margin))
                host_id: Optional[int] = host.id
            else:
                tx, ty = position()
                host_id = None
            transients.append({
                "id": 10_000 + k, "x": tx, "y": ty, "kind": transient_kind,
                "peak_flux": float(10 ** rng.uniform(3.4, 4.7)),
                "peak_epoch": float(rng.uniform(0.2, 0.7) * (n_epochs - 1)),
                "rise": float(rng.uniform(0.8, 2.0)),
                "decay": float(rng.uniform(2.5, 6.0)),
                "host_truth_id": host_id,
            })

        # Movers: a start position, a constant sky velocity, and a trail set
        # by how far the object travels while the shutter is open.
        for k in range(max(0, int(n_movers))):
            rate = float(rng.uniform(*mover_rate_range))          # arcsec / hour
            heading = float(rng.uniform(0.0, 360.0))
            # arcsec/hour -> pixels/day: divide by the pixel scale, then
            # multiply by 24.  Dividing by 24 instead leaves an asteroid
            # moving half a pixel across a whole series, which looks exactly
            # like a stationary source and silently removes the thing being
            # tested.
            speed = rate / cfg.pixel_scale * 24.0                 # pixels / day
            # Start on the side the object is coming *from*, so it spends the
            # series crossing the field rather than leaving it at once.
            span = speed * cadence * (n_epochs - 1)
            start_x = float(np.clip(nx / 2 - 0.5 * span * np.cos(np.radians(heading)),
                                    margin, nx - margin))
            start_y = float(np.clip(ny / 2 - 0.5 * span * np.sin(np.radians(heading)),
                                    margin, ny - margin))
            transients.append({
                "id": 20_000 + k, "x": start_x, "y": start_y, "kind": "mover",
                "peak_flux": float(10 ** rng.uniform(3.6, 4.6)),
                "rate_arcsec_per_hour": rate,
                "heading_deg": heading,
                "vx": speed * float(np.cos(np.radians(heading))),
                "vy": speed * float(np.sin(np.radians(heading))),
                "trail_length": rate * (300.0 / 3600.0) / cfg.pixel_scale,
                "host_truth_id": None,
            })

        images: List[AstroImage] = []
        for epoch in range(int(n_epochs)):
            epoch_sim = SkySimulator(SkyConfig(**{**cfg.__dict__, "seed": base_seed + 1000 + epoch}))
            frame = static_canvas.copy()

            for star in variable_stars:
                phase = 2 * np.pi * (epoch * cadence) / max(star.period, 1e-3)
                scale = 1.0 + star.amplitude * np.sin(phase)
                epoch_sim.add_star(frame, star.x, star.y, star.flux * scale)
                epoch_sim._next_id -= 1

            for spec in transients:
                if spec["kind"] == "mover":
                    elapsed = epoch * cadence
                    mx = spec["x"] + spec["vx"] * elapsed
                    my = spec["y"] + spec["vy"] * elapsed
                    inside = (0 <= mx < nx) and (0 <= my < ny)
                    if inside:
                        epoch_sim.add_mover(frame, mx, my, spec["peak_flux"],
                                            spec["trail_length"], spec["heading_deg"])
                        epoch_sim._next_id -= 1
                    spec.setdefault("positions", []).append(
                        {"epoch": epoch, "time": elapsed, "x": float(mx), "y": float(my),
                         "inside": bool(inside)})
                    continue
                dt = epoch - spec["peak_epoch"]
                # Fast exponential rise, slower exponential decline.
                if dt < 0:
                    scale = float(np.exp(dt / max(spec["rise"], 1e-3)))
                else:
                    scale = float(np.exp(-dt / max(spec["decay"], 1e-3)))
                if scale > 1e-3:
                    epoch_sim.add_star(frame, spec["x"], spec["y"],
                                       spec["peak_flux"] * scale)
                    epoch_sim._next_id -= 1
                spec.setdefault("light_curve", []).append(
                    {"epoch": epoch, "time": epoch * cadence,
                     "flux": spec["peak_flux"] * scale})

            signal = frame + epoch_sim._background_map()
            artifacts = epoch_sim._add_artifacts(signal)
            pixels = epoch_sim._apply_noise(signal)
            images.append(AstroImage(
                data=pixels,
                header={"OBJECT": "SYNTHETIC-SERIES", "FILTER": cfg.band,
                        "GAIN": cfg.gain, "MAGZP": cfg.zero_point},
                wcs=SimpleWCS.tangent(cfg.field_centre[0], cfg.field_centre[1],
                                      cfg.shape, cfg.pixel_scale),
                name=f"epoch_{epoch:02d}", band=cfg.band,
                mjd=59000.0 + epoch * cadence, exposure_time=300.0,
                meta={"simulated": True, "epoch": epoch, "artifacts": artifacts},
            ))

        log.info("generated %d-epoch series with %d transients and %d variables",
                 len(images), len(transients), len(variable_stars))
        return ImageSeries(images, name="synthetic_series"), static_truth, transients


def quick_field(shape: Tuple[int, int] = (256, 256), seed: int = 7,
                **overrides) -> Tuple[AstroImage, List[TruthObject]]:
    """Small demo field with sensible defaults -- handy in tests and docs."""
    config = SkyConfig(shape=shape, seed=seed,
                       n_stars=overrides.pop("n_stars", 60),
                       n_galaxies=overrides.pop("n_galaxies", 12),
                       n_nebulae=overrides.pop("n_nebulae", 1),
                       n_clusters=overrides.pop("n_clusters", 1),
                       n_lenses=overrides.pop("n_lenses", 1),
                       n_anomalies=overrides.pop("n_anomalies", 1),
                       **overrides)
    return SkySimulator(config).generate()

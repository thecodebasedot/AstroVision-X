"""A mass model for a strong lens, and the fit that measures it.

Detecting arcs says a lens is probably there.  A *model* says how much mass is
inside the Einstein radius, how the mass is shaped, and where the background
galaxy really is -- which is the difference between a candidate and a
measurement.

The model is the standard one for an early-type galaxy: a **singular
isothermal ellipsoid** plus **external shear**.  Both parts earn their place.
An isothermal profile is what stellar dynamics and lensing independently find
for massive ellipticals, and it has the convenient property that its
deflection is nearly constant with radius.  External shear stands in for
everything else along the line of sight -- a neighbouring group, a filament --
and leaving it out does not make the model simpler, it makes it *wrong*: the
fit absorbs the missing shear into the ellipticity and reports a mass
distribution flatter than the real one.

Deflections follow Keeton (2001), which writes the singular case in closed
form.  The circular limit is delicate -- both components carry a
``1/sqrt(1-q^2)`` that diverges as the ellipsoid becomes round -- so the
round case is handled separately rather than approached numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger

log = get_logger("lensing.model")

#: Below this flattening the SIE formulae are numerically unstable and the
#: circular (isothermal sphere) limit is used instead.
ROUND_LIMIT = 1e-4

#: Azimuthal coverage, in degrees, below which external shear is not fitted.
#: Shear and ellipticity both stretch images; separating them needs to see
#: that stretch from more than one direction.
#:
#: The number is measured, not chosen.  Fitting ray-traced images of a lens
#: with a true axis ratio of 0.70 and a true shear of 0.036, at spans from
#: 140 to 267 degrees: above about 220 degrees the fit returns q within 0.06
#: and the shear within 0.05, while everything sampled between 140 and 165
#: degrees with the shear free collapsed to q ~ 0.2 with a shear of 0.4 --
#: the ellipsoid's own flattening reappearing as a fictitious tidal field.
#: Nothing between 165 and 223 degrees was reachable in that geometry, so the
#: threshold sits in the middle of the untested gap rather than at the edge
#: of the band that worked.
MIN_SHEAR_SPAN_DEG = 200.0

#: External shear above this is no longer a perturbation from neighbouring
#: structure; a fit that wants it may be absorbing something else.
MAX_PLAUSIBLE_SHEAR = 0.3

#: How much worse the shear-free fit must be before an implausibly large
#: shear is believed rather than dropped.  Measured on ray-traced images: when
#: the shear is spurious -- the ellipsoid's flattening fitted twice, which is
#: what poor azimuthal coverage produces -- removing it costs a factor 1.0 to
#: 1.7 in source-plane scatter, because it was buying almost nothing.  When
#: the lens really does sit in a strong tidal field (true shear 0.2 to 0.45),
#: removing it costs a factor of 16 to 30.  Nothing measured landed between,
#: so the two cases separate cleanly.
SHEAR_EVIDENCE_FACTOR = 4.0


def azimuthal_span(positions: np.ndarray, centre: Sequence[float]) -> float:
    """Angular coverage of a set of positions around a centre, in degrees.

    The largest gap is found and subtracted from a full turn, which is what
    makes this a *coverage* rather than a range: two arcs on opposite sides
    span the sky far better than the same number of points crowded into one.

    >>> import numpy as np
    >>> half = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    >>> round(azimuthal_span(half, (0.0, 0.0)))       # nothing below the axis
    180
    >>> ring = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -10.0]])
    >>> round(azimuthal_span(ring, (0.0, 0.0)))
    270
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    angles = np.sort(np.degrees(np.arctan2(points[:, 1] - centre[1],
                                           points[:, 0] - centre[0])) % 360.0)
    gaps = np.diff(np.concatenate([angles, angles[:1] + 360.0]))
    return float(360.0 - gaps.max())


def sis_deflection(dx: np.ndarray, dy: np.ndarray, b: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Deflection of a singular isothermal *sphere*: constant magnitude ``b``.

    >>> ax, ay = sis_deflection(np.array([3.0]), np.array([4.0]), 10.0)
    >>> round(float(np.hypot(ax, ay)[0]), 6)
    10.0
    """
    radius = np.hypot(dx, dy)
    safe = np.where(radius > 0, radius, 1.0)
    scale = float(b) / safe
    return np.where(radius > 0, scale * dx, 0.0), np.where(radius > 0, scale * dy, 0.0)


def sie_deflection(dx: np.ndarray, dy: np.ndarray, b: float, q: float,
                   pa_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Deflection of a singular isothermal ellipsoid, in the image plane.

    ``b`` is the Einstein radius, ``q`` the minor-to-major axis ratio, and
    ``pa_deg`` the major axis measured counter-clockwise from ``+x``.  The
    calculation is done in the ellipsoid's own frame and rotated back, which
    is what keeps the position angle meaningful rather than absorbed into the
    other parameters.
    """
    q = float(np.clip(q, 0.05, 1.0))
    if 1.0 - q < ROUND_LIMIT:
        return sis_deflection(dx, dy, b)

    angle = math.radians(float(pa_deg))
    cos, sin = math.cos(angle), math.sin(angle)
    x = cos * dx + sin * dy          # into the ellipsoid frame
    y = -sin * dx + cos * dy

    eccentricity = math.sqrt(1.0 - q * q)
    psi = np.sqrt(q * q * x * x + y * y)
    safe = np.where(psi > 0, psi, 1.0)
    prefactor = float(b) * math.sqrt(q) / eccentricity
    ax = prefactor * np.arctan(eccentricity * x / safe)
    ay = prefactor * np.arctanh(np.clip(eccentricity * y / safe, -0.999999, 0.999999))
    ax = np.where(psi > 0, ax, 0.0)
    ay = np.where(psi > 0, ay, 0.0)

    return cos * ax - sin * ay, sin * ax + cos * ay      # back to the sky frame


def shear_deflection(dx: np.ndarray, dy: np.ndarray, g1: float, g2: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Deflection from a uniform external shear.

    Shear is a *tidal* distortion: it stretches images without adding mass at
    the lens, so a fit that omits it has to explain the stretch with the
    ellipsoid and comes out flatter than the galaxy really is.
    """
    return float(g1) * dx + float(g2) * dy, float(g2) * dx - float(g1) * dy


@dataclass
class LensModel:
    """A singular isothermal ellipsoid plus external shear."""

    x0: float = 0.0                  # lens centre, pixels
    y0: float = 0.0
    theta_e: float = 10.0            # Einstein radius, pixels
    axis_ratio: float = 1.0
    position_angle: float = 0.0      # degrees CCW from +x
    shear1: float = 0.0
    shear2: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shear_magnitude(self) -> float:
        return float(math.hypot(self.shear1, self.shear2))

    @property
    def shear_angle(self) -> float:
        """Direction of the external shear, degrees CCW from ``+x``."""
        return float(math.degrees(0.5 * math.atan2(self.shear2, self.shear1)) % 180.0)

    @property
    def ellipticity(self) -> float:
        return float(1.0 - self.axis_ratio)

    def deflection(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Total deflection at image-plane positions ``(x, y)``."""
        dx = np.asarray(x, dtype=float) - self.x0
        dy = np.asarray(y, dtype=float) - self.y0
        ax, ay = sie_deflection(dx, dy, self.theta_e, self.axis_ratio,
                                self.position_angle)
        sx, sy = shear_deflection(dx, dy, self.shear1, self.shear2)
        return ax + sx, ay + sy

    def source_plane(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Where image-plane positions map to: the lens equation ``b = t - a``."""
        ax, ay = self.deflection(x, y)
        return np.asarray(x, dtype=float) - ax, np.asarray(y, dtype=float) - ay

    def magnification(self, x, y, step: float = 0.05) -> np.ndarray:
        """Signed magnification, from the Jacobian of the lens equation.

        Differenced numerically rather than differentiated by hand: the
        analytic Jacobian of an SIE plus shear is long enough that an error in
        it would be invisible, and the numerical version is exact to the step
        size on a map this smooth.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        bx_px, by_px = self.source_plane(x + step, y)
        bx_mx, by_mx = self.source_plane(x - step, y)
        bx_py, by_py = self.source_plane(x, y + step)
        bx_my, by_my = self.source_plane(x, y - step)
        a11 = (bx_px - bx_mx) / (2 * step)
        a21 = (by_px - by_mx) / (2 * step)
        a12 = (bx_py - bx_my) / (2 * step)
        a22 = (by_py - by_my) / (2 * step)
        determinant = a11 * a22 - a12 * a21
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(determinant) > 1e-12, 1.0 / determinant, np.inf)

    def critical_curve(self, shape: Tuple[int, int], step: float = 0.05) -> np.ndarray:
        """Boolean mask of where the magnification changes sign.

        The critical curve is where a lens formally magnifies infinitely, and
        images pile up along it -- which is why arcs sit near the Einstein
        radius rather than anywhere else.
        """
        ny, nx = int(shape[0]), int(shape[1])
        yy, xx = np.mgrid[0:ny, 0:nx]
        inverse = 1.0 / self.magnification(xx, yy, step)
        sign = np.sign(inverse)
        crossing = np.zeros((ny, nx), dtype=bool)
        crossing[:, :-1] |= sign[:, :-1] != sign[:, 1:]
        crossing[:-1, :] |= sign[:-1, :] != sign[1:, :]
        return crossing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x0": float(self.x0), "y0": float(self.y0),
            "theta_e": float(self.theta_e),
            "axis_ratio": float(self.axis_ratio),
            "ellipticity": self.ellipticity,
            "position_angle": float(self.position_angle),
            "shear1": float(self.shear1), "shear2": float(self.shear2),
            "shear_magnitude": self.shear_magnitude,
            "shear_angle": self.shear_angle,
            **{k: v for k, v in self.meta.items()},
        }


def ray_trace(shape: Tuple[int, int], model: LensModel,
              source_x: float, source_y: float, source_radius: float,
              source_flux: float = 1.0, sersic_n: float = 1.0,
              supersample: int = 3) -> np.ndarray:
    """Render what a lens does to a background galaxy.

    Ray shooting, which is the honest way round: for every image-plane pixel,
    solve the lens equation *forwards* to find where that ray came from in the
    source plane, and read the source's brightness there.  Multiple images,
    arcs and rings then appear because they are what the mapping does -- not
    because anything drew an arc.

    That matters for testing a lens fit.  Arcs painted at a chosen radius
    reward a fit for finding the radius they were painted at; arcs produced by
    a mass distribution test whether the fit can recover the mass.
    """
    ny, nx = int(shape[0]), int(shape[1])
    factor = max(1, int(supersample))
    yy, xx = np.mgrid[0:ny * factor, 0:nx * factor]
    # Sub-pixel sample centres, so the supersampled grid covers the same area.
    xx = (xx + 0.5) / factor - 0.5
    yy = (yy + 0.5) / factor - 0.5

    beta_x, beta_y = model.source_plane(xx, yy)
    radius = np.hypot(beta_x - float(source_x), beta_y - float(source_y))
    scale = max(float(source_radius), 0.3)
    n = float(np.clip(sersic_n, 0.3, 6.0))
    # Sersic b_n via the Ciotti-Bertin expansion, so a de Vaucouleurs source
    # really does have its half-light radius where it is asked for.
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
    brightness = np.exp(-bn * ((radius / scale) ** (1.0 / n) - 1.0))

    if factor > 1:
        brightness = brightness.reshape(ny, factor, nx, factor).mean(axis=(1, 3))
    total = float(brightness.sum())
    if total > 0:
        brightness = brightness * (float(source_flux) / total)
    return brightness


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
@dataclass
class LensFit:
    """A fitted model with the evidence for it."""

    model: Optional[LensModel] = None
    n_points: int = 0
    source_rms: float = float("nan")       # scatter of the mapped source, pixels
    image_rms: float = float("nan")        # residual back in the image plane
    theta_e_error: float = float("nan")
    axis_ratio_error: float = float("nan")
    shear_error: float = float("nan")
    source_x: float = float("nan")
    source_y: float = float("nan")
    azimuthal_span: float = float("nan")   # degrees of arc coverage
    succeeded: bool = False
    reason: str = ""
    flags: List[str] = field(default_factory=list)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "succeeded": bool(self.succeeded),
            "n_points": int(self.n_points),
            "source_rms": float(self.source_rms),
            "image_rms": float(self.image_rms),
            "theta_e_error": float(self.theta_e_error),
            "axis_ratio_error": float(self.axis_ratio_error),
            "shear_error": float(self.shear_error),
            "source_position": [float(self.source_x), float(self.source_y)],
            "azimuthal_span": float(self.azimuthal_span),
            "reason": self.reason,
            "flags": list(self.flags),
        }
        if self.model is not None:
            payload["model"] = self.model.to_dict()
        return payload


def source_plane_scatter(positions: np.ndarray, model: LensModel) -> float:
    """How far apart a model puts images that should share one source.

    This is the quantity the fit minimises, and its weakness is worth stating:
    a source-plane residual is *not* the image-plane residual an astronomer
    cares about, and it under-weights highly magnified images -- the very ones
    whose positions are measured best.  It is used because it needs no
    solution of the lens equation, and the image-plane residual is then
    computed once at the end as an honest check on what it gave.
    """
    beta_x, beta_y = model.source_plane(positions[:, 0], positions[:, 1])
    return float(np.sqrt(np.mean((beta_x - beta_x.mean()) ** 2
                                 + (beta_y - beta_y.mean()) ** 2)))


def image_plane_residual(positions: np.ndarray, model: LensModel,
                         source: Tuple[float, float],
                         search_radius: float = 6.0,
                         samples: int = 61) -> float:
    """Distance from each observed image to the nearest predicted one.

    Predicted images are found by shooting a local grid around each
    observation and taking the point that lands closest to the source -- which
    is a genuine solution of the lens equation, just a local one.  Reported in
    pixels, so it can be compared directly against the astrometric precision.
    """
    if positions.size == 0:
        return float("nan")
    residuals = []
    for x, y in positions:
        offsets = np.linspace(-float(search_radius), float(search_radius), int(samples))
        gx, gy = np.meshgrid(x + offsets, y + offsets)
        beta_x, beta_y = model.source_plane(gx, gy)
        distance = np.hypot(beta_x - source[0], beta_y - source[1])
        index = np.unravel_index(int(np.argmin(distance)), distance.shape)
        residuals.append(math.hypot(gx[index] - x, gy[index] - y))
    return float(np.sqrt(np.mean(np.square(residuals))))


#: Speed of light and Newton's constant in the units the mass formula uses:
#: c in km/s, G in Mpc (km/s)^2 / Msun.
C_KM_S = 299792.458
G_MPC_KMS2_MSUN = 4.30091e-9


def einstein_mass(theta_e_arcsec: float, z_lens: float, z_source: float,
                  cosmology=None) -> Dict[str, float]:
    """Projected mass inside the Einstein radius.

    The classic result, and the reason a lens is worth modelling at all:

    .. math:: M_E = \frac{c^2}{4G} \frac{D_L D_S}{D_{LS}} \theta_E^2

    It is a *projected* mass -- everything within a cylinder along the line
    of sight, not a sphere -- and it is nearly assumption-free, needing no
    knowledge of the mass profile.  That is what makes lensing masses trusted
    where dynamical ones are argued about.

    What it does need is both redshifts.  Without them the distance ratio has
    to be assumed, and an assumed ratio is where an order-of-magnitude error
    enters; the returned record says which redshifts were used so the number
    is never mistaken for one that was measured.
    """
    from ..astrophysics.cosmology import Cosmology

    cosmology = cosmology or Cosmology()
    result = {"theta_e_arcsec": float(theta_e_arcsec),
              "z_lens": float(z_lens), "z_source": float(z_source),
              "mass_solar": float("nan"), "log_mass_solar": float("nan"),
              "velocity_dispersion_km_s": float("nan"),
              "einstein_radius_kpc": float("nan")}
    if not np.isfinite(theta_e_arcsec) or theta_e_arcsec <= 0:
        return result
    if not (0 < z_lens < z_source):
        result["reason"] = "needs 0 < z_lens < z_source"
        return result

    d_l = float(cosmology.angular_diameter_distance(z_lens))
    d_s = float(cosmology.angular_diameter_distance(z_source))
    d_ls = float(cosmology.angular_diameter_distance_between(z_lens, z_source))
    if min(d_l, d_s, d_ls) <= 0:
        return result                                          # pragma: no cover

    theta = math.radians(float(theta_e_arcsec) / 3600.0)
    sigma_critical_factor = (C_KM_S ** 2 / (4.0 * G_MPC_KMS2_MSUN)) * (d_l * d_s / d_ls)
    mass = sigma_critical_factor * theta ** 2
    result["mass_solar"] = float(mass)
    result["log_mass_solar"] = float(math.log10(mass)) if mass > 0 else float("nan")
    result["einstein_radius_kpc"] = float(theta * d_l * 1000.0)
    # The isothermal-sphere velocity dispersion the same radius implies.
    ratio = theta / (4.0 * math.pi) * (d_s / d_ls)
    result["velocity_dispersion_km_s"] = float(C_KM_S * math.sqrt(max(ratio, 0.0)))
    return result


def arc_sample_points(arcs: Sequence[Any], centre: Sequence[float],
                      per_arc: int = 7) -> np.ndarray:
    """Positions along each detected arc, as constraints for the fit.

    An arc is an *extended* constraint and reducing it to one centroid throws
    most of it away -- the curvature along an arc is what pins the Einstein
    radius, and a single point carries none of it.  Points are laid out along
    the arc's own angular extent, since ``length`` is measured tangentially:
    the half-span in radians is simply ``length / (2 * radius)``.
    """
    points: List[Tuple[float, float]] = []
    for arc in arcs:
        ridge = np.asarray(getattr(arc, "points", np.zeros((0, 2))), dtype=float)
        if ridge.size:
            points.extend((float(x), float(y)) for x, y in ridge.reshape(-1, 2))
            continue
        # Only when an arc carries no ridge -- a ring found by the radial
        # scan, which has no pixels of its own -- fall back to laying points
        # along its nominal circle.  Those constrain the radius and nothing
        # else, which is the honest limit of what a ring scan measured.
        radius = float(getattr(arc, "radius", float("nan")))
        angle = float(getattr(arc, "angle", float("nan")))
        length = float(getattr(arc, "length", 0.0))
        if not (np.isfinite(radius) and np.isfinite(angle)) or radius <= 0:
            continue
        half_span = min(0.5 * max(length, 1.0) / radius, math.pi * 0.9)
        n = max(2, int(per_arc))
        for offset in np.linspace(-half_span, half_span, n):
            theta = math.radians(angle) + offset
            points.append((centre[0] + radius * math.cos(theta),
                           centre[1] + radius * math.sin(theta)))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _pack(model: LensModel) -> np.ndarray:
    return np.array([model.theta_e, model.axis_ratio, model.position_angle,
                     model.shear1, model.shear2, model.x0, model.y0], dtype=float)


def _unpack(values: np.ndarray) -> LensModel:
    return LensModel(theta_e=float(values[0]), axis_ratio=float(values[1]),
                     position_angle=float(values[2]), shear1=float(values[3]),
                     shear2=float(values[4]), x0=float(values[5]), y0=float(values[6]))


def fit_lens_model(positions: Sequence[Sequence[float]],
                   centre: Tuple[float, float],
                   theta_e_guess: float = float("nan"),
                   fit_shear: bool = True,
                   fit_centre: bool = False,
                   bootstrap: int = 24,
                   seed: int = 0) -> LensFit:
    """Fit an SIE (plus shear) so the arc positions share one source.

    Needs at least four positions for the ellipsoid alone and six with shear,
    and refuses below that rather than returning a model with more parameters
    than constraints -- which would fit perfectly and mean nothing.

    Errors come from a bootstrap over the positions.  A curvature estimate
    would be cheaper and would understate them badly here: the parameters are
    strongly degenerate (ellipticity against shear, most of all) and the
    positions are few.
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    fit = LensFit(n_points=len(points))

    # External shear is a *tidal* stretch, so telling it apart from the
    # ellipsoid's own flattening needs images at different azimuths, not just
    # more images.  A single arc sampled a hundred times still sees the shear
    # from one direction, and the fit then trades ellipticity against shear
    # freely -- measured on ray-traced systems, axis ratios came out anywhere
    # from 0.35 to 1.0 with shears up to 0.33 for a true 0.05.
    span = azimuthal_span(points, centre)
    if fit_shear and span < MIN_SHEAR_SPAN_DEG:
        fit.add_flag("shear_fixed_to_zero")
        fit.reason = (f"arcs span only {span:.0f} degrees around the lens; "
                      f"external shear needs {MIN_SHEAR_SPAN_DEG:.0f} to be "
                      "separable from the ellipticity and was held at zero")
        fit_shear = False

    n_free = 3 + (2 if fit_shear else 0) + (2 if fit_centre else 0)
    # Each position contributes two numbers but also costs the two unknowns of
    # the shared source position, so the constraint count is 2N - 2.
    if 2 * len(points) - 2 < n_free:
        fit.reason = (f"{len(points)} arc positions give {2 * len(points) - 2} "
                      f"constraints for {n_free} parameters; refusing to fit")
        return fit

    if not np.isfinite(theta_e_guess) or theta_e_guess <= 0:
        theta_e_guess = float(np.median(np.hypot(points[:, 0] - centre[0],
                                                 points[:, 1] - centre[1])))
    start = LensModel(x0=float(centre[0]), y0=float(centre[1]),
                      theta_e=max(theta_e_guess, 1.0), axis_ratio=0.85,
                      position_angle=0.0)

    best = _minimise(points, start, fit_shear, fit_centre)
    if fit_shear and best.shear_magnitude > MAX_PLAUSIBLE_SHEAR:
        # A shear this large is either real or an artefact, and the two look
        # identical in the fitted parameters -- both report a nearly round
        # lens in a violent tidal field.  What tells them apart is what the
        # shear is *buying*: drop it and refit, and see how much worse the fit
        # gets.  A spurious shear was standing in for the ellipsoid's own
        # flattening, so the shear-free model reproduces the images nearly as
        # well; a real one is holding the model together, and removing it
        # wrecks it.  Believe the shear only when the data insist on it.
        shear_free = _minimise(points, start, False, fit_centre)
        free_rms = source_plane_scatter(points, best)
        held_rms = source_plane_scatter(points, shear_free)
        if held_rms <= SHEAR_EVIDENCE_FACTOR * max(free_rms, 1e-9):
            best = shear_free
            fit_shear = False
            fit.add_flag("implausible_shear_refit_without_shear")
            fit.reason = (f"the fitted shear exceeded {MAX_PLAUSIBLE_SHEAR:.1f} and "
                          f"holding it at zero cost only a factor "
                          f"{held_rms / max(free_rms, 1e-9):.1f} in scatter, so it "
                          "was not external shear but the ellipsoid's own "
                          "flattening; refitted without it")
        else:
            fit.add_flag("implausible_shear")
            fit.reason = (f"the images require a shear of {best.shear_magnitude:.2f} "
                          f"-- removing it costs a factor "
                          f"{held_rms / max(free_rms, 1e-9):.0f} in scatter -- which "
                          "is beyond a normal external field and suggests a second "
                          "deflector this model does not have")
    fit.model = best
    fit.azimuthal_span = span
    fit.source_rms = source_plane_scatter(points, best)
    beta_x, beta_y = best.source_plane(points[:, 0], points[:, 1])
    fit.source_x, fit.source_y = float(beta_x.mean()), float(beta_y.mean())
    fit.image_rms = image_plane_residual(points, best, (fit.source_x, fit.source_y))
    fit.succeeded = True
    if not fit.reason:
        fit.reason = "fitted by source-plane minimisation"

    if bootstrap and len(points) >= 4:
        rng = np.random.default_rng(int(seed))
        samples = []
        for _ in range(int(bootstrap)):
            index = rng.integers(0, len(points), len(points))
            if len(np.unique(index)) < 3:
                continue
            trial = _minimise(points[index], best, fit_shear, fit_centre)
            samples.append([trial.theta_e, trial.axis_ratio, trial.shear_magnitude])
        if len(samples) >= 5:
            array = np.asarray(samples, dtype=float)
            fit.theta_e_error = float(np.std(array[:, 0]))
            fit.axis_ratio_error = float(np.std(array[:, 1]))
            fit.shear_error = float(np.std(array[:, 2]))

    if fit.source_rms > 0.3 * best.theta_e:
        fit.add_flag("poor_source_convergence")
    return fit


def _minimise(points: np.ndarray, start: LensModel, fit_shear: bool,
              fit_centre: bool, iterations: int = 400) -> LensModel:
    """Nelder-Mead on the source-plane scatter, without SciPy.

    A downhill simplex rather than a gradient method: the objective is cheap,
    the parameter count is small, and the surface has flat directions along
    the ellipticity-shear degeneracy where a gradient step goes nowhere
    useful.  SciPy is used when it is installed.
    """
    from ..core.backend import try_import

    scale = np.array([0.15 * max(start.theta_e, 1.0), 0.08, 15.0, 0.03, 0.03, 1.0, 1.0])
    mask = np.array([True, True, True, fit_shear, fit_shear, fit_centre, fit_centre])

    def objective(free: np.ndarray) -> float:
        values = _pack(start).copy()
        values[mask] = free
        values[1] = float(np.clip(values[1], 0.15, 1.0))
        values[0] = float(max(values[0], 0.5))
        if fit_shear:
            values[3] = float(np.clip(values[3], -0.4, 0.4))
            values[4] = float(np.clip(values[4], -0.4, 0.4))
        return source_plane_scatter(points, _unpack(values))

    initial = _pack(start)[mask]
    steps = scale[mask]

    def run(seed_point: np.ndarray) -> Tuple[np.ndarray, float]:
        optimize = try_import("scipy.optimize")
        if optimize is not None:
            # The initial simplex has to be supplied explicitly.  SciPy builds
            # its own by perturbing each coordinate 5%, which is *zero* for a
            # parameter starting at zero -- and both shear components and the
            # position angle start at zero here.  Left to itself the optimiser
            # therefore barely explores exactly the directions that matter,
            # and stopped seven times short of the true minimum on constraints
            # whose exact solution was known.
            simplex = np.vstack([seed_point] +
                                [seed_point + np.eye(len(seed_point))[i] * steps[i]
                                 for i in range(len(seed_point))])
            result = optimize.minimize(objective, seed_point, method="Nelder-Mead",
                                       options={"maxiter": iterations,
                                                "initial_simplex": simplex,
                                                "xatol": 1e-4, "fatol": 1e-6})
            return np.asarray(result.x, dtype=float), float(result.fun)
        found = _nelder_mead(objective, seed_point, steps, iterations)
        return found, objective(found)

    # The position angle is periodic and the objective is multimodal in it, so
    # a single start lands in whichever basin it happens to begin in.  Four
    # seeds a quarter-turn apart cost four cheap fits and remove that lottery.
    best_point, best_value = None, float("inf")
    pa_index = int(np.nonzero(mask[:3])[0].tolist().index(2)) if mask[2] else None
    for pa_seed in (0.0, 45.0, 90.0, 135.0):
        seed_point = initial.copy()
        if pa_index is not None:
            seed_point[pa_index] = pa_seed
        point, value = run(seed_point)
        if value < best_value:
            best_point, best_value = point, value
        if pa_index is None:
            break
    solution = best_point if best_point is not None else initial

    values = _pack(start).copy()
    values[mask] = solution
    values[1] = float(np.clip(values[1], 0.15, 1.0))
    values[0] = float(max(values[0], 0.5))
    values[2] = float(values[2] % 180.0)
    return _unpack(values)


def _nelder_mead(objective, start: np.ndarray, scale: np.ndarray,
                 iterations: int = 400, tolerance: float = 1e-6) -> np.ndarray:
    """Downhill simplex, so the fit works with NumPy alone."""
    n = len(start)
    simplex = [np.asarray(start, dtype=float)]
    for i in range(n):
        point = np.asarray(start, dtype=float).copy()
        point[i] += scale[i]
        simplex.append(point)
    values = [objective(p) for p in simplex]

    for _ in range(int(iterations)):
        order = np.argsort(values)
        simplex = [simplex[int(i)] for i in order]
        values = [values[int(i)] for i in order]
        if abs(values[-1] - values[0]) <= tolerance * (abs(values[0]) + tolerance):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = objective(reflected)
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            f_expanded = objective(expanded)
            simplex[-1], values[-1] = ((expanded, f_expanded)
                                       if f_expanded < f_reflected
                                       else (reflected, f_reflected))
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_contracted = objective(contracted)
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
    order = int(np.argmin(values))
    return simplex[order]

"""Error bars for morphology, by parametric bootstrap and by covariance.

Every morphological statistic in this package returns a number.  None of
them, on their own, says how well that number is known -- and a Sersic index
of 3.8 means something entirely different when it is 3.8 +- 0.2 than when it
is 3.8 +- 2.1.  In practice the second case is common: at moderate
signal-to-noise the Sersic index is one of the worst-determined quantities in
galaxy photometry, and quoting it bare invites conclusions the data cannot
support.

Two methods, because the statistics differ in kind:

* **Covariance** for the fitted parameters.  A least-squares fit has a
  natural error estimate in the curvature of its own chi-squared surface --
  the inverse of ``J^T W J`` at the minimum.  It is cheap, and it captures
  the parameter *correlations*, which for Sersic fits are severe: the index,
  the effective radius and the sky level trade off against each other almost
  freely, and the marginal error on ``n`` alone understates the problem.

* **Parametric bootstrap** for the non-parametric statistics.  Concentration,
  asymmetry, Gini and M20 are not fits and have no Jacobian, so their errors
  come from re-measuring the object on repeated noise realisations drawn at
  the image's own measured noise level.  This is the honest way round: it
  propagates exactly the noise the pixels actually have, through exactly the
  code that produced the measurement, including whatever the code does at
  its own edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging import get_logger
from .cas import asymmetry, concentration, smoothness
from .gini_m20 import gini_m20

log = get_logger("morphology.uncertainty")


@dataclass
class ParameterErrors:
    """Marginal errors and the correlations behind them."""

    names: Sequence[str]
    errors: np.ndarray
    correlation: np.ndarray
    scaled_by_chi2: bool = True
    reason: str = ""

    def of(self, name: str) -> float:
        """The marginal error on one parameter, or NaN if it was not fitted."""
        if name not in self.names:
            return float("nan")
        return float(self.errors[list(self.names).index(name)])

    def worst_correlation(self) -> Tuple[str, str, float]:
        """The most degenerate parameter pair, which is what to report.

        A fit whose worst pair correlates at 0.99 has, in effect, one
        parameter fewer than it claims, and the marginal errors on those two
        badly understate how poorly each is separately determined.
        """
        if self.correlation.size == 0 or len(self.names) < 2:
            return ("", "", 0.0)
        matrix = np.abs(np.asarray(self.correlation, dtype=float)).copy()
        np.fill_diagonal(matrix, 0.0)
        index = int(np.argmax(matrix))
        i, j = divmod(index, matrix.shape[1])
        return (self.names[i], self.names[j], float(self.correlation[i, j]))

    def to_dict(self) -> Dict[str, Any]:
        first, second, value = self.worst_correlation()
        return {
            "errors": {name: float(err) for name, err in zip(self.names, self.errors)},
            "worst_correlation": {"parameters": [first, second], "value": value},
            "scaled_by_chi2": bool(self.scaled_by_chi2),
            "reason": self.reason,
        }


def covariance_errors(residual_function: Callable[[np.ndarray], np.ndarray],
                      parameters: Sequence[float], names: Sequence[str],
                      n_data: int, step: float = 1e-4,
                      scale_by_chi2: bool = True) -> ParameterErrors:
    """Parameter errors from the numerical Jacobian at the solution.

    ``residual_function`` must return the *weighted* residual vector, so that
    the chi-squared is its sum of squares.  The Jacobian is taken by central
    differences with a step scaled to each parameter, which is stable enough
    here because the residual is smooth by construction.

    With ``scale_by_chi2`` the covariance is multiplied by the reduced
    chi-squared.  That is the right default for image fitting: the formal
    errors assume the model is correct and the noise estimate exact, and
    neither holds for a galaxy -- a real galaxy is not a Sersic profile.
    Scaling by the achieved chi-squared reports the errors the *fit* justifies
    rather than the ones the model would deserve if it were true.
    """
    theta = np.asarray(parameters, dtype=float)
    base = np.asarray(residual_function(theta), dtype=float)
    if base.size == 0 or not np.all(np.isfinite(base)):
        return ParameterErrors(names, np.full(len(theta), np.nan),
                               np.zeros((0, 0)), reason="residual is not finite")

    jacobian = np.zeros((base.size, theta.size), dtype=float)
    for index in range(theta.size):
        delta = float(step * max(abs(theta[index]), 1.0))
        forward, backward = theta.copy(), theta.copy()
        forward[index] += delta
        backward[index] -= delta
        high = np.asarray(residual_function(forward), dtype=float)
        low = np.asarray(residual_function(backward), dtype=float)
        if high.shape != base.shape or low.shape != base.shape:
            return ParameterErrors(names, np.full(len(theta), np.nan),
                                   np.zeros((0, 0)),
                                   reason="residual length changed under perturbation")
        jacobian[:, index] = (high - low) / (2.0 * delta)

    hessian = jacobian.T @ jacobian
    degrees = max(int(n_data) - theta.size, 1)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # A singular Hessian means the parameters are exactly degenerate --
        # which is information, not an error.  The pseudo-inverse gives the
        # errors along the directions that *are* constrained.
        covariance = np.linalg.pinv(hessian)
    if scale_by_chi2:
        covariance = covariance * float(np.sum(base ** 2)) / degrees

    variance = np.diag(covariance).copy()
    variance[variance < 0] = np.nan
    errors = np.sqrt(variance)
    with np.errstate(invalid="ignore", divide="ignore"):
        outer = np.outer(errors, errors)
        correlation = np.where(outer > 0, covariance / outer, 0.0)
    return ParameterErrors(names=list(names), errors=errors,
                           correlation=np.clip(correlation, -1.0, 1.0),
                           scaled_by_chi2=bool(scale_by_chi2),
                           reason="from the Jacobian at the solution")


@dataclass
class BootstrapErrors:
    """Spread of each statistic over repeated noise realisations."""

    n_samples: int
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    reason: str = ""

    def error(self, name: str) -> float:
        """Standard deviation of one statistic, or NaN if unmeasured."""
        sample = self.values.get(name)
        if sample is None or sample.size < 2:
            return float("nan")
        finite = sample[np.isfinite(sample)]
        return float(np.std(finite)) if finite.size >= 2 else float("nan")

    def bias(self, name: str, measured: float) -> float:
        """How far the resampled median sits from the original measurement.

        A large bias means the statistic is not merely noisy but *skewed* by
        noise -- true of asymmetry, which is built from absolute differences
        and so is pushed upward by noise no matter which way the noise goes.
        """
        sample = self.values.get(name)
        if sample is None or sample.size < 2 or not np.isfinite(measured):
            return float("nan")
        finite = sample[np.isfinite(sample)]
        return float(np.median(finite) - measured) if finite.size >= 2 else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": int(self.n_samples),
            "errors": {name: self.error(name) for name in self.values},
            "reason": self.reason,
        }


def bootstrap_morphology(cutout: np.ndarray, noise: float,
                         centre: Optional[Tuple[float, float]] = None,
                         mask: Optional[np.ndarray] = None,
                         n_samples: int = 24,
                         statistics: Sequence[str] = ("concentration", "asymmetry",
                                                      "smoothness", "gini", "m20"),
                         seed: int = 0) -> BootstrapErrors:
    """Re-measure the shape statistics on ``n_samples`` noise realisations.

    Noise is *added* to the observed cutout rather than to a noiseless model,
    which slightly overstates the errors -- the realisations carry the
    original noise plus a fresh draw.  The alternative needs a model of the
    galaxy, and a wrong model produces confidently wrong error bars; erring
    toward larger error bars is the safe direction, and the overstatement is
    a factor of sqrt(2) at worst.

    ``n_samples`` is small by default because these statistics are expensive
    and a standard deviation from 24 samples is already good to about 15% --
    far better than the order-of-magnitude uncertainty in the error bar's
    *meaning* that comes from anything else in the measurement.
    """
    data = np.asarray(cutout, dtype=float)
    if not np.isfinite(noise) or noise <= 0:
        return BootstrapErrors(n_samples=0, reason="no usable noise estimate")
    if data.size == 0:
        return BootstrapErrors(n_samples=0, reason="empty cutout")

    rng = np.random.default_rng(int(seed))
    samples: Dict[str, List[float]] = {name: [] for name in statistics}
    for _ in range(max(2, int(n_samples))):
        realisation = data + rng.normal(0.0, float(noise), size=data.shape)
        try:
            if "concentration" in samples:
                samples["concentration"].append(
                    float(concentration(realisation, centre)["concentration"]))
            if "asymmetry" in samples:
                samples["asymmetry"].append(
                    float(asymmetry(realisation, centre)["asymmetry"]))
            if "smoothness" in samples:
                samples["smoothness"].append(
                    float(smoothness(realisation, centre=centre, mask=mask)["smoothness"]))
            if "gini" in samples or "m20" in samples:
                shape = gini_m20(realisation, mask=mask)
                if "gini" in samples:
                    samples["gini"].append(float(shape["gini"]))
                if "m20" in samples:
                    samples["m20"].append(float(shape["m20"]))
        except (ValueError, FloatingPointError, IndexError, KeyError):
            # A realisation whose noise happens to leave no pixels above the
            # statistic's own threshold is a legitimate outcome, not a bug;
            # it simply contributes nothing to this sample.
            continue

    values = {name: np.asarray(sample, dtype=float) for name, sample in samples.items()}
    usable = min((v.size for v in values.values()), default=0)
    return BootstrapErrors(
        n_samples=int(usable), values=values,
        reason=f"parametric bootstrap at {noise:.4g} counts/pixel")


def annotate_uncertainty(source, bootstrap: BootstrapErrors,
                         parameters: Optional[ParameterErrors] = None) -> None:
    """Attach error bars to a source's morphology record.

    The errors go in ``meta["morphology_errors"]`` rather than onto the
    metrics themselves so that nothing downstream silently starts reading a
    different type -- and so that a consumer that ignores them is obviously
    ignoring them, rather than appearing not to need them.
    """
    record: Dict[str, Any] = {"bootstrap": bootstrap.to_dict()}
    biases = {
        "asymmetry": bootstrap.bias("asymmetry", source.morphology.asymmetry),
        "concentration": bootstrap.bias("concentration", source.morphology.concentration),
    }
    record["noise_bias"] = {k: float(v) for k, v in biases.items() if np.isfinite(v)}
    if parameters is not None:
        record["sersic"] = parameters.to_dict()
        first, second, value = parameters.worst_correlation()
        if abs(value) > 0.95 and first and second:
            source.add_flag("degenerate_sersic_fit")
            record["sersic"]["note"] = (
                f"{first} and {second} are {abs(value):.2f} correlated: the fit "
                "constrains their combination far better than either alone")
    source.meta["morphology_errors"] = record

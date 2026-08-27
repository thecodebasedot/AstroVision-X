"""Tying an image to the sky and to a magnitude system.

Two calibrations turn measurements into science:

* **Astrometric** -- the WCS that came with the file is a starting guess, not
  an answer.  Pointing models are imperfect, focal planes flex, and a
  telescope that reports its position to an arcsecond is doing well.
  :mod:`~astrovision.calibration.astrometry` refits the plate against a
  reference catalog and reports the residual, so the number quoted for a
  candidate's position has an error bar behind it.
* **Photometric** -- a zero point of 25 is a placeholder that makes fluxes
  into numbers.  :mod:`~astrovision.calibration.photometry` derives the real
  one from catalogued standards in the same frame, with the colour term that
  accounts for the filter not being exactly the reference filter.

Both are optional and both are honest about failing: with too few matched
standards they refuse rather than fitting noise, and say so.
"""

from .astrometry import (
    AstrometricSolution,
    match_to_reference,
    solve_plate,
)
from .photometry import (
    PhotometricSolution,
    apply_zero_point,
    solve_zero_point,
)

__all__ = [
    "AstrometricSolution", "solve_plate", "match_to_reference",
    "PhotometricSolution", "solve_zero_point", "apply_zero_point",
]

# Validation

Every performance number in this repository is measured against the
simulator's ground truth, not asserted. This document says exactly how, so the
numbers can be reproduced and disputed.

## Why a simulator

Real astronomical images have no ground truth. You cannot ask an image how many
stars are really in it, or what the true Sérsic index of a galaxy is. A
simulator gives both — and if it is built from the same physics the pipeline
tries to recover, then recovering it is a genuine test.

`astrovision.simulate` renders fields from the same functional forms the
morphology stage fits back out: Sérsic profiles, logarithmic spiral patterns,
stellar bars, Einstein arcs, a Moffat PSF, Poisson photon noise, Gaussian read
noise, a background gradient, cosmic rays and bad columns. Every injected
object is recorded in a truth table with its position, flux and structural
parameters.

Two things this does *not* prove: that the pipeline works on real data with
their instrument signatures, and that the simulator's assumptions match any
particular telescope. What it does prove is that each measurement recovers what
it claims to measure, under noise, and that no stage silently destroys signal.

## Reproducing the numbers

```bash
pytest tests/ -q                          # everything, about a minute
pytest tests/test_pipeline_stages.py -q   # detection, photometry, morphology
pytest tests/test_transients.py -q        # difference imaging, variability, lensing
```

The benchmarks quoted in the README come from the tests below plus the
multi-field sweeps described in each section.

---

## Detection

**Setup.** 384 × 384 fields, 100–120 stars, no galaxies, cosmic rays and bad
columns disabled, five seeds. Signal-to-noise computed per object as
`flux / (rms × √A)` with `A = 4π(FWHM/2.355)²`.

**Result.** At a 3.5 σ threshold, 93 % of sources above S/N 10 are recovered
within 3 px. Spurious detections — catalog entries with no truth object within
3 px — are 1–3 % of the catalog at 3.5 σ and below 1 % at 5 σ. Median
astrometric error is 0.15 px.

**Recovery by signal-to-noise**, 512 × 512 field, 200 stars:

| S/N | 3.0 σ | 3.5 σ | 5.0 σ |
| --- | --- | --- | --- |
| 3–5 | 4/8 | 1/8 | 0/8 |
| 5–10 | 20/26 | 19/26 | 8/26 |
| > 10 | 155/166 | 154/166 | 158/166 |

The missing objects above S/N 10 are blends: with 200 stars placed at random in
512², some pairs fall within a PSF width of each other.

Tests: `TestDetection::test_recovers_bright_isolated_stars`,
`test_spurious_rate_is_low`.

## Deblending

Two Gaussians separated by 16 px are split into two; a single Gaussian stays
one; three sources become three; a companion below the flux-contrast threshold
is correctly merged.

Tests: `TestDetection::test_deblends_two_overlapping_sources`,
`test_leaves_a_single_source_intact`.

## Point-spread function

**Setup.** 400 × 400 fields, five field compositions from star-only to
galaxy-dominated, four seeing values (2.5–5.5 px), three seeds — 60 fields.

**Result.** Median FWHM error 6 %, 90th percentile 15 %. The failures are
concentrated in the extreme case of 10 stars against 50 bright galaxies, where
too few point sources survive the stellar-locus cut; the pipeline emits a
warning there rather than proceeding silently.

| Field | 2.5 px | 2.8 px | 4.0 px | 5.5 px |
| --- | --- | --- | --- | --- |
| Stars only | +8.5 % | +5.4 % | +4.8 % | +1.6 % |
| 40 stars, 30 galaxies | +7.8 % | +5.3 % | +4.8 % | +2.9 % |
| 20 stars, 40 galaxies | +9.4 % | +6.0 % | +5.8 % | +1.2 % |
| Crowded mixed | +9.1 % | +5.6 % | +6.3 % | +2.3 % |

The consistent positive bias is expected: the isophotal second-moment width of
a Moffat profile sits slightly above its true FWHM.

Tests: `TestPreprocess::test_psf_fwhm_matches_the_seeing`,
`test_psf_rejects_galaxies`.

## Spatially varying PSF

**Setup.** 800 × 800 fields, 220 stars, seeing 3.0 px on the optical axis
growing quadratically with field radius. Star selection over 4 × 4 tiles, a
21-pixel stamp, quadratic in position.

**Result.**

| Injected variation | Stars used | Detected | Fitted / true corner-to-centre ratio |
| --- | --- | --- | --- |
| 0 % | 87 | no (correctly) | — |
| 20 % | 119 | no | — |
| 20 % | 231 | no | — |
| 40 % | 120 | yes | 1.310 / 1.360 |
| 40 % | 227 | yes | 1.379 / 1.371 |

The method detects and recovers variation of roughly 35% and above given
about 120 well-separated stars. At 20% it does not, even with 231 stars —
and it says so and falls back to one PSF rather than fitting noise. That
boundary is a property of the per-star photon noise in these simulations, not
a threshold anyone chose.

**The payoff.** Aperture corrections derived from the local PSF close the
centre-to-corner photometric gap:

| PSF model | Centre flux ratio | Corner flux ratio | Gap |
| --- | --- | --- | --- |
| One for the field | 1.0146 | 0.9929 | **2.2 %** |
| Position-dependent | 0.9959 | 0.9949 | **0.1 %** |

**Negative result.** Regional PSF matching in difference imaging was
implemented in three variants and made things worse every time — spurious
candidates rose from 18 to 45–114 on the same field. It was removed rather
than left behind a switch. See `docs/methods.md`.

Tests: `tests/test_varying_psf.py`.

## Photometry

**Aperture geometry.** Fractional-coverage areas match `πr²` to better than
1 % for radii 2–8 px.

**Flux recovery.** A synthetic Gaussian star of known total flux on a noisy
background is recovered to 0.5 % at the matched aperture.

**End-to-end.** Isolated stars brighter than 2000 counts, matched to truth
within 2 px, in 256 × 256 fields: median flux ratio 1.02–1.03 with 2–3 %
scatter. Before the PSF aperture correction the same measurement gave 0.95 —
the 5 % deficit is the Moffat wings outside the aperture.

**Concentration index.** PSF-convolved Sérsic profiles: `n = 1` gives
`C = 2.75` (textbook 2.7), `n = 4` gives `C = 4.9` (textbook 5.2, the shortfall
from truncating the integration at 70 px).

Tests: `TestPhotometry::*`.

## Morphology

**Sérsic index**, PSF-convolved profiles at S/N typical of a survey detection:

| True n | Recovered | True r_eff | Recovered |
| --- | --- | --- | --- |
| 0.7 | 0.63–0.64 | 7.0 | 6.8–7.0 |
| 1.0 | 0.82–0.95 | 6.0–8.0 | 5.9–7.9 |
| 2.5 | 2.32–2.51 | 8.0 | 6.9–7.7 |
| 4.0 | 4.13–5.37 | 7.0–10.0 | 6.3–13.7 |
| 6.0 | 7.20–7.24 | 6.0 | 4.7–5.1 |

Median relative error 11 %. High indices are the least well constrained, which
is the known behaviour of every Sérsic fitting code.

**Non-parametric statistics**, on rendered galaxies:

| Object | C | A | S | Gini | M₂₀ |
| --- | --- | --- | --- | --- | --- |
| Elliptical n = 4 | 4.26 | 0.000 | 0.319 | 0.620 | −2.20 |
| Lenticular n = 2.5 | 3.77 | 0.000 | 0.274 | 0.569 | −2.08 |
| Disc n = 1 | 2.76 | 0.000 | 0.203 | 0.487 | −1.78 |
| Merger (double) | 2.66 | 0.108 | 0.169 | 0.383 | −1.17 |

Gini for an elliptical (0.62) and a disc (0.49) match published values, and the
merger sits above the Lotz merger line while the others sit below it.

**Arms and bars.** Rendered 2-, 3- and 4-armed spirals at per-pixel noise 0, 3
and 10:

| Object | Detected arms | Significance | Bar? |
| --- | --- | --- | --- |
| Elliptical | 0 (0 at 2/3 noise levels) | 2.6–3.1 | no |
| 2-arm spiral | 2 at all noise levels | 9.6–9.9 | no |
| 3-arm spiral | 3 at all noise levels | 5.1–5.8 | no |
| 4-arm spiral | 4 at all noise levels | 3.5–4.0 | no |
| Barred disc | 0 | 8.6–10.1 | **yes** |

Recovered pitch angles are 18–35° for a true 20°. The bar is correctly
separated from arms by its constant mode phase.

**Classification**, five galaxy types on 400 × 400 fields, five seeds, 108
matched galaxies: **59 % exact**, **78 % at family level** (early / disc /
irregular / merger). Per-class recall: elliptical 79 %, irregular 93 %, spiral
53 %, lenticular 47 %, barred spiral 12 %. Barred spirals are almost all
classified as ordinary spirals — a sub-type confusion, correct at family level.

Tests: `TestMorphology::*`.

## Star/galaxy separation

**Setup.** 384 × 384 fields, 60–90 stars and 20–25 galaxies plus nebulae and
clusters, five seeds, 372 matched objects.

**Result.** 90 % correct for stars and galaxies; 87 % across all four classes.

| Signal-to-noise | Stars | Galaxies |
| --- | --- | --- |
| < 10 | 95 % | 43 % |
| 10–30 | 81 % | 100 % |
| > 30 | 94 % | 100 % |

Galaxies below S/N 10 are genuinely unresolved at this seeing, and the
classifier reports low confidence for them rather than a confident wrong
answer. Nebulae (1/6) and star clusters (4/12) remain weak.

Tests: `TestClassification::test_separates_stars_from_galaxies`.

## Multi-band colours

**Setup.** 260 × 260 three-band fields (g, r, i) with per-band seeing of
1.44", 1.28" and 1.36", five seeds. Colours are measured by forced
photometry in a 1.6" aperture at the r-band positions, after every band is
convolved to the worst seeing in the set. 264 matched objects above S/N 15.

**Result.** g−r bias **−0.009 mag**, scatter **0.060 mag**.

The bias depends on how the bands' seeing differs, and finding that out
required a controlled test:

| Seeing | Without aperture correction | With it |
| --- | --- | --- |
| All bands equal | +0.003 ± 0.010 | −0.016 ± 0.010 |
| g worst (1.56") | **+0.035** ± 0.007 | −0.001 ± 0.007 |
| r worst (1.56") | −0.034 ± 0.012 | −0.029 ± 0.012 |

Matching a Moffat PSF to a wider Moffat with a Gaussian kernel does not
reproduce the wings exactly, so a fixed aperture still catches slightly
different fractions per band. Correcting each band by the enclosed energy of
its own post-matching PSF removes the largest case entirely. Two bugs
surfaced here: the matched PSF *model* was carrying the unconvolved stamp
while claiming the widened FWHM, and colours were being recorded for sources
detected at 40σ in one band and 1σ in the other — the latter alone inflated
the star colour scatter from 0.10 to 0.37 mag.

**Limit.** Formal photometric errors are about **2.5× too small**: measured
per-axis colour scatter for stars is 0.083 mag where the formal errors say
0.033. They count photon and read noise but not sky estimation, blending, or
PSF-matching residuals. Nothing downstream uses them as if they were right.

Tests: `tests/test_multiband.py`.

## Colour-based star/galaxy separation

**Setup.** As above, 442 matched stars and galaxies over five seeds.

**Result.** Morphology alone **93.9 %**; with colour **93.7 %** — a
difference of one object, which is to say no difference.

That is the honest outcome and it is worth stating plainly: at this depth
the colour test carries almost no star/galaxy information. Its measured
separation (ROC area against the morphological labels) is 0.70, which the
field-calibrated weighting turns into a colour weight of 0.38. In a deeper
variant (lower sky, lower read noise) the same machinery gives 94.1 % → 95.1 %.

Getting to "no difference" took four corrections, each caught by measurement:

1. A one-sided sigmoid of the locus offset returned ≈0.85 for *everything*,
   including galaxies — an uninformative test that voted "star" for all of
   it. Replacing it with a likelihood ratio between two calibrated
   populations makes an uninformative test return 0.5 on its own. Before:
   94.1 % → 84.6 %.
2. The widths were taken from formal errors, which are 2.5× too small (above).
   Calibrating both populations' widths from the field itself, in
   signal-to-noise bins, fixed that.
3. The Rayleigh tail is far too thin for real data: a blended star lands five
   sigma off the locus and gets convicted with certainty. A 5 % outlier
   component in both hypotheses stopped that.
4. Sources fainter than the calibrated range were being judged against widths
   measured on brighter ones. They are now given no colour vote at all.

| Signal-to-noise bin | Star width | Galaxy width | Usable? |
| --- | --- | --- | --- |
| ~11 | 0.092 | 0.094 | no — noise dominates |
| ~46 | 0.050 | 0.108 | yes |
| ~166 | 0.028 | 0.083 | yes |

Tests: `TestStellarLocus`, particularly
`test_an_uninformative_test_returns_one_half` and
`test_colour_never_makes_classification_worse`.

## Astrometric calibration

**Setup.** 300 × 300 fields, five seeds. The header WCS is corrupted the way
a real pointing error corrupts one: shifted 6.5 and −4.2 px, rotated 0.35°,
and rescaled by 1.004. ~117 reference stars.

**Result.** Corner error **3.51" → 0.047"**, fit rms **0.110"** from 82
matched stars, in 5 of 5 fields. The solution recovers the injected rotation
to 0.05° and the scale to 1 part in 10⁴.

Mutual-nearest matching matters: one-sided matching assigns several
detections to one bright reference star in a crowded field, and those
duplicated pairs pull the fit. The solver refuses rather than returning a
plate solution from fewer than 8 stars.

## Photometric calibration

**Setup.** As above, standards drawn from the injected star fluxes.

**Result.** Zero point **24.987 ± 0.002** against a true 25.000, with 0.017
mag scatter from 50 standards.

The −0.013 mag offset is a **systematic six times the formal error** — it
comes from the residual aperture correction, not from the standards — and it
is exactly why the formal error on a zero point should not be quoted as its
accuracy.

## SIP distortion

**Setup.** A 2048 × 2048 tangent plane with second-order SIP coefficients
producing ~1" of distortion at the corners.

**Result.** Pixel → sky → pixel round-trips to **4 × 10⁻¹⁰ px** using the
iterative inverse, with no reverse coefficients needed. The header round-trip
preserves the coefficients and stamps the `-SIP` suffix on `CTYPE`.

## Known-object crossmatch

**Setup.** 260 × 260 field, reference catalog built from the injected star
positions, 1.5" match radius.

**Result.** 84 of 113 detections match a known object. Matched sources are
flagged, described in words ("variable star", "minor planet"), and demoted in
the follow-up ranking — an anomaly by a factor of 4, a transient barely at
all, since a supernova in a known galaxy is the normal case.

The report distinguishes three states that a naive implementation conflates:
*checked and matched*, *checked and not matched*, and *not checked* — the
last including a cone that returned zero references, which establishes
nothing about the field.

## Morphological uncertainty

**Setup.** Parametric bootstrap over 16–24 noise realisations at the image's
own measured noise.

**Result.** Errors scale with noise as they should; at 6× the noise every
statistic's error grows by roughly 6×. The bootstrap also measures a **+0.13
noise bias in asymmetry**, which is real: asymmetry is built from absolute
differences, so noise pushes it up whichever way it goes.

Sérsic parameter errors come from the curvature at the solution, scaled by
the achieved chi-squared. On simulator galaxies the median error on *n* is
**2.6**, against an actual median |n_fit − n_true| of 0.76 — the error bars
are conservative. The median |correlation| of the worst parameter pair is
**1.00**: these fits are effectively degenerate, which is the finding, and
sources carry a `degenerate_sersic_fit` flag rather than a bare index that
reads as well determined.

Fixing this exposed that `reduced_chi2` was not a reduced chi-squared at all —
the residual was unweighted, so the number was in counts squared.

## Probability calibration

**Setup.** A synthetic overconfident classifier (true log-odds multiplied by
2.2), a 4000-point held-out test set.

**Result.** Expected calibration error **0.112 → 0.027** with isotonic
regression on 1000 training points; Platt scaling recovers the overconfidence
slope to within 0.12 of its true value.

Two corrections were needed. Platt scaling belongs on the **log-odds** scale,
not on the probability — applied to probabilities directly it made the
calibration error worse. And Newton's method without a line search diverges
on near-separable validation sets: the fitted slope reached 3.8 × 10⁷,
producing a calibrator that returns only 0 and 1.

## Transient detection

**Setup.** Five multi-epoch series (five or six epochs, 2-day cadence),
160–300 px, 2–3 injected transients with exponential rise and decline, plus
variable stars at a 6 % rate.

**Result.** 12/14 injected transients recovered within 4 px, all classified as
supernova candidates with the correct host, and 2 spurious vetted candidates in
total. 10/18 variable stars also recovered as variability candidates.

Real and spurious candidates separate cleanly on the real/bogus score:

| Threshold | Real kept | Spurious kept |
| --- | --- | --- |
| 0.5 | 16/16 | 10/10 |
| 0.6 | 16/16 | 8/10 |
| 0.7 (default) | 16/16 | 2/10 |
| 0.8 | 16/16 | 0/10 |

**Vetting**, on synthetic stamps: a real point source scores 0.97, a faint one
0.82, a dipole 0.07, a cosmic ray 0.10, a satellite streak 0.32, pure noise
0.04, an over-subtraction residual 0.02.

Three bugs were found and fixed by this benchmark, each of which had silently
destroyed real signal:

1. Bad-column detection flagged 78 % of an image and erased injected
   transients (missing noise floor on the column-median test).
2. Flux scaling was biased 10 % high by a median-of-ratios estimator, leaving
   50 σ residuals at every bright star.
3. PSF matching to a PSF measured from three stars degraded the science image.

After the fixes, the maximum residual in a difference image dropped from 128 σ
to 33 σ — and the 33 σ residual is the transient itself.

Tests: `TestRealBogus::*`, `TestDifferenceImaging::*`, `TestTransientSearch::*`.

## Moving objects

**Setup.** 300 × 300 fields, five epochs at a 14-minute cadence (a
one-hour arc), two injected asteroids at 15–90 arcsec/hour plus one
supernova. Five seeds with movers and five without.

**Result.** **10/10** injected movers recovered, **0** spurious tracklets,
**10/10** supernovae still present as transient candidates. Rates recovered
to better than 0.1 arcsec/hour and headings to within a degree; track
residuals 0.19–0.24 px against an astrometric precision of 0.15 px.

44 transient candidates were reclassified as movers — detections that would
otherwise have been 44 single-epoch entries in a follow-up queue.

The five mover-free runs produced no tracklets at all, which is the control
that matters: a linker that finds asteroids in a field containing none is
finding its own tolerance.

| Cut on the track residual | Real kept | Spurious kept |
| --- | --- | --- |
| ≤ 0.4 px | 10/14 | 0/1 |
| ≤ 0.8 px | 12/14 | 0/1 |
| ≤ 1.5 px (raw) | 14/14 | 1/1 |
| ≤ 1.5 px (reduced by d.o.f.) | 14/14 | 0/1 |

The last row is the whole point. Reducing the residual by the degrees of
freedom is not a tuned threshold — it corrects a real unfairness, since a
three-point fit of four parameters cannot help but look tighter than a
five-point one.

Tests: `tests/test_moving.py`.

## Variability and periods

Irregularly sampled light curves, 40 epochs over 50 days:

| Curve | χ²/ν | Stetson J | η | Score | Recovered period |
| --- | --- | --- | --- | --- | --- |
| Constant | 0.9 | 0.05 | 1.97 | 0.08 | none (FAP 0.88) |
| Sinusoid P = 3.7 | 30.8 | 1.44 | 1.39 | 0.75 | **3.70** |
| Sinusoid P = 11 | 10.0 | 2.07 | 0.52 | 0.92 | **10.91** |
| Eruptive | 16.0 | 1.81 | 0.26 | 0.90 | none (aperiodic) |
| Linear trend | 51.2 | 5.94 | 0.05 | 1.00 | none (FAP 0.78) |

Six-class light-curve classification: rule-based 79 %, nearest-centroid 100 %,
LSTM 92 %, Transformer 82 % on held-out curves.

Tests: `TestVariability::*`.

## Novelty detection

**Isolation forest.** Injected outliers rank first in a 400-point Gaussian
cloud; scores stay in [0, 1]; the flagged fraction tracks the configured
contamination.

**Autoencoder.** Points off a 2-D manifold embedded in 6-D score 30–50× the
reconstruction error of points on it, for both the linear and the deep
implementation.

**End-to-end.** Injected anomalous morphologies (rings, double nuclei, jets,
X-shapes) in 400 × 400 fields land in the **top 11 % by score** (median), with
explanations naming the actual cause — high asymmetry, multiple nuclei, no
close analogue in the field.

Tests: `TestIsolationForest::*`, `TestAutoencoder::*`.

## Strong lensing

**Setup.** 320 × 320 fields with three injected lens systems each, five seeds,
14 matched deflectors.

**Result.** 8/14 recovered with 3 false positives. Einstein radii are recovered
to within 10 % when arcs are detected (13.0–13.7 px for a true 14.0).

Detection depends strongly on arc geometry, which is honest — that is also true
of real searches:

| Configuration | Arcs found | θ_E recovered | Ring detected |
| --- | --- | --- | --- |
| 2 arcs, 50° spans | 0 | — | no |
| 3 arcs, 50° spans | 1 | 13.5 | no |
| 3 arcs, 60° spans | 3 | 13.0 ± 0.2 | no |
| 4 arcs, 45° spans | 2 | 13.7 | no |
| 3 arcs, 70° spans | 1 | 13.2 | **yes** |
| 6 arcs, 70° spans | 0 | — | **yes** |
| Plain elliptical galaxy | **0** | — | **no** |

Wide arcs merge into a ring, which the azimuthal-baseline arc finder cannot
see; the radial ring scan catches those instead. A plain elliptical galaxy
produces no arcs and no ring, which is the property that matters most.

Tests: `TestLensing::*`.

## Photometric redshifts

**Setup.** 400 galaxies per configuration, each drawn from a spectrum with
continuous age, dust and emission parameters — so none of them is in the
six-template fit library. Metric is `Δz/(1+z)`; an outlier is `|Δz/(1+z)| >
0.15`; the scatter quoted is the robust spread of the non-outliers.

| Filters | Colour error | Bias | Scatter | Outliers |
| --- | --- | --- | --- | --- |
| g r i | 0.02 | −0.005 | 0.043 | **22.2 %** |
| g r i | 0.10 | −0.005 | 0.070 | 19.8 % |
| g r i z | 0.02 | −0.005 | 0.021 | 11.0 % |
| g r i z | 0.10 | −0.006 | 0.044 | 13.5 % |
| u g r i z | 0.02 | −0.003 | **0.015** | **2.8 %** |
| u g r i z | 0.10 | +0.001 | 0.034 | 4.0 % |

The filter count dominates everything else. Three filters give two colours
against three unknowns — redshift, spectral type and dust — and the outlier
rate reflects that rather than any deficiency in the fit.

The `reliable` flag earns its place: at three filters and 0.10 mag colour
errors it cuts the outlier rate from 19.8 % to 4.2 %, at the cost of keeping
30 % of the sample.

**End to end.** Through the full pipeline — detection, forced photometry,
colours, fit — on a 300 × 300 five-band field with 30 galaxies at redshifts
0.05–0.9: **bias 0.000, scatter 0.011, 9 % outliers**; on the reliable subset,
scatter 0.005 and no outliers. All 33 fitted galaxies then carry their own
photometric distance in the astrophysics layer instead of a field-wide
assumption.

Tests: `tests/test_photoz.py`.

## Cosmology

Distances agree with Astropy's `FlatLambdaCDM(H0=70, Om0=0.3)` to machine
precision across 0.1 ≤ z ≤ 5 — comoving, angular-diameter and luminosity
distances, distance modulus, angular scale, lookback time, and the
angular-diameter distance between two redshifts.

## Deep models

Trained from scratch on simulated data inside the test environment:

| Model | Training set | Result |
| --- | --- | --- |
| CenterNet detector | 28 images, 399 objects, 40 epochs | 97 % held-out recall |
| CNN stamp classifier | 266 stamps, 4 classes, 30 epochs | 85 % test accuracy |
| ViT stamp classifier | same | 49 % — transformers need far more data |
| U-Net segmenter | 16 images, 25 epochs | 97 % pixel accuracy |
| LSTM light curves | 240 curves, 6 classes | 92 % test accuracy |
| Transformer light curves | same | 82 % |

The ViT result is reported as measured. On a few hundred stamps a
convolutional inductive bias wins, and that is worth knowing before choosing a
backbone.

## What is not validated

- Real instrument signatures: fringing, scattered light, non-linearity,
  charge transfer inefficiency.
- The VizieR and SIMBAD backends against the live services. Their URL
  construction and response parsing are tested against recorded fixtures;
  the transport is not, because a test suite that needs the network to pass
  fails for reasons unrelated to the code.
- Real astrometric distortion beyond TAN and second-order SIP.
- Colours across genuinely different cameras, where the pixel grids and the
  filter throughputs both differ. The code handles it; the simulator renders
  every band on one grid, so the path is exercised but not stressed.
- Crowded stellar fields at globular-cluster densities.
- Any absolute photometric calibration against a standard system.
- Whether the morphological classifier agrees with human classifiers on real
  galaxies, which is the only test that would matter for that stage.

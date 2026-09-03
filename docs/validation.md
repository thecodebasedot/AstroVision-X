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
15 injected deflectors. Lensed images are produced by **ray tracing through a
mass model**, not painted at a chosen radius.

**Result.** 4/15 recovered with 9 false positives.

That is a large downgrade from the 8/14 with 3 false positives this document
reported previously, and the reason is worth stating plainly: **the old number
was measured against arcs that were drawn rather than lensed.** Painted arcs
are placed at a chosen radius with chosen spans and chosen brightness, and
the search was, in effect, being asked to find the thing that had been drawn
for it. Ray-traced arcs are whatever the mass model actually produces — often
a single faint image rather than a tidy pair, and frequently below the arc
finder's threshold. The search did not get worse; the test got honest.

Where the recall goes: of the eleven missed systems, most produce one arc too
faint or too short to clear the detection threshold. The false positives are
almost all single-arc detections around ordinary galaxies, scoring just over
the threshold — the score rewards arc multiplicity, and a single arc is weak
evidence being counted as some evidence.

Detection depends strongly on arc geometry, which is honest — that is also true
of real searches. The table below uses *painted* arcs, where span and count can
be set directly, which is what makes the dependence visible:

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

### The mass model

**Setup.** A singular isothermal ellipsoid with external shear, fitted to
image positions found by ray shooting through a known model — so the exact
answer is known and any failure is the fit's, not the data's.

**Result: exact constraints.** Given 17 images of one source spanning 245°,
the fit reaches the true model: θ_E 14.00 for a true 14.00, axis ratio 0.702
for a true 0.700, position angle 35.0 for a true 35.0, shear 0.036 for a true
0.036. Its source-plane scatter is 0.0374 px against the true model's own
0.0377, and the image-plane residual is 0.12 px.

Dropping the shear from that same fit pulls the axis ratio to 0.649 — the
ellipsoid flattening itself to reproduce a tidal stretch it is not allowed to
name. That is the measured reason external shear is in the model.

Getting there required fixing the optimiser rather than the physics. The
Nelder–Mead simplex was being built by perturbing each parameter by a fixed
*fraction* — which is zero for any parameter starting at zero, and both shear
components and the position angle start at zero. Three of the search
directions were therefore degenerate and the fit stopped short of the true
minimum on constraints whose exact solution was known, returning an axis ratio
of exactly 1.000 every time. Supplying the initial simplex explicitly, plus
restarts from four position angles because the objective is multimodal in it,
recovers the true model.

**Result: why the shear is gated.** The same true model (q = 0.70, shear =
0.036), fitted to images spanning different angles around the lens:

| Azimuthal span | Fitted q, no noise | Fitted q, 1 px noise | Fitted shear |
| --- | --- | --- | --- |
| 267° | 0.89 | 0.79 | 0.04–0.09 |
| 255° | 0.76 | 0.73 | 0.02–0.10 |
| 237° | 0.74 | 0.88 | 0.03–0.07 |
| 223° | 0.69 | 0.61 | 0.05–0.10 |
| 163° *(shear free)* | **0.48** | **0.25** | **0.11–0.43** |
| 158° *(shear free)* | **0.22** | **0.20** | **0.41–0.43** |
| 145° *(shear free)* | **0.24** | **0.22** | **0.36–0.50** |

Below about 165° the fit reports a nearly round lens in a violent tidal field —
the ellipsoid's own flattening counted twice. Above about 220° it is right.
Nothing in this geometry lands between, so the gate sits at **200°**, in the
middle of the untested gap rather than at the edge of the band that worked.
With the gate in place and the shear held at zero below it, every span in the
table returns an axis ratio between 0.59 and 0.82 for a true 0.70.

**Result: when a large shear is believed.** Removing a *spurious* shear costs
a factor 1.0–1.7 in source-plane scatter; removing a *real* one (true shear
0.2–0.45) costs a factor 16–30. The fit refits without the shear whenever the
free value exceeds 0.3 and keeps whichever model the data support, flagging
either outcome.

**Result: end to end.** Of the four recovered systems above, all four were
modelled:

| Quantity | Result |
| --- | --- |
| Einstein radius | 20 % median fractional error (9 %, 16 %, 24 %, 34 %) |
| Axis ratio | 0.13 median absolute error, true 0.76–0.97 |
| Image-plane residual | 3.8 px median |
| Implied mass | log M_E 12.6–13.1, at an assumed source redshift |

The Einstein radii from the model are less accurate than the 10 % quoted above
for the arc-fitting radius, and that is expected: the arc fit is handed a
radius by construction, while the model is fitted to a handful of ridge points
on one or two faint arcs and is solving for five parameters at once.

**Sanity check on the masses.** θ_E = 1.2″ with the deflector at z = 0.4 and
the source at z = 2.0 gives 2.8 × 10¹¹ M☉ inside an Einstein radius of 6.4 kpc,
implying 244 km/s — squarely the SLACS regime for an early-type galaxy lens.

Tests: `tests/test_lens_model.py`.

## Spectroscopy

**Setup.** Simulated long-slit frames: a curved trace, a cubic dispersion,
seeing that worsens to the blue, the night sky with its emission lines,
Poisson and read noise, and cosmic rays. Galaxy spectra carry absorption of
the right depth and emission lines whose *ratios* follow one ionisation
parameter, so a diagnostic test cannot succeed on unphysical input.

### Extraction

| Quantity | Result |
| --- | --- |
| Trace accuracy | 0.014 px rms, 0.019 px worst, against a trace curving 2 px |
| Sky subtraction | median residual 0.05 counts against a 131-count sky |
| Reported errors | within 6 % of the scatter of repeated exposures |
| Optimal vs best fixed aperture | +3 % in signal-to-noise |
| Optimal vs a reasonable aperture (±5 rows) | +16 % |
| Corrupted columns, 20 cosmic rays, boxcar | 13 of 11200 |
| Corrupted columns, profile weighting, no rejection | 25 |
| Corrupted columns, profile weighting with rejection | **0** |

The signal-to-noise gain from optimal extraction is small and the textbook
claim is easy to overstate: a Gaussian profile is forgiving, and a well-chosen
aperture already recovers most of what is available. The rejection is where
the value is.

### Wavelength calibration

All 26 arc lines are found and identified. Fitted at the polynomial order that
generated the data:

| Order | Fit residual | True wavelength error, inside the fitted range |
| --- | --- | --- |
| 1 | 4.76 Å | *refused* |
| 2 | 1.69 Å | *refused* |
| 3 | 0.52 Å | 0.10 Å rms, 0.38 Å worst |
| 4 | 0.50 Å | 0.16 Å rms, 0.67 Å worst |

The 0.5 Å residual floor is the centroiding error, about an eighth of a pixel,
not a failure of the fit. A **misidentification by one line leaves 3.7 Å**,
which is what sets the refusal threshold between them.

Identifying the lines from a linear first guess fails in a way that looks like
success: the residual settled at 3.4–3.9 Å at every order, including order 3.
The pairwise vote fixes it. The vote itself places only 16 of 26 lines, which
is not a defect — no linear solution fits a cubic dispersion — and is ample to
anchor the polynomial refinement that then picks up all 26.

Night-sky check on a correctly calibrated frame: offset −0.03 ± 0.03 Å with
0.10 Å scatter over 14 lines, and an injected 3 Å flexure is recovered as
2.96 Å. Handed the sky-*subtracted* spectrum instead the same check gives −0.4
with 2.8 Å of scatter — not wrong, but blind to anything under a pixel.

### Redshifts

Recovery on spectra at signal-to-noise 15, redshifts 0.02–0.55: **median
|Δz/(1+z)| = 7 × 10⁻⁵**, about 21 km/s, with no outliers in 18.

The interesting measurement is the reliability flag, since the flag is the
product and not the number:

| Median S/N per pixel | Reported reliable | Of which correct | Purity | Completeness |
| --- | --- | --- | --- | --- |
| 30 | 12/12 | 12 | 1.00 | 1.00 |
| 15 | 12/12 | 12 | 1.00 | 1.00 |
| 8 | 10/12 | 10 | 1.00 | 0.83 |
| 5 | 11/12 | 10 | 0.91 | 0.83 |
| 3 | 4/12 | 1 | **0.25** | 0.08 |
| pure noise | **0/20** | — | — | — |

**Below S/N ≈ 5 the reliability flag stops meaning anything**, and that is the
honest limit of this implementation. Above it, no pure-noise spectrum and no
wrong answer at S/N ≥ 8 survives the gates.

Both gates were placed from measurement, and the two do different jobs:

| Gate | What it removes | What it costs |
| --- | --- | --- |
| R ≥ 7 | 40/40 pure-noise spectra | 3 of 24 correct at S/N 3–8 |
| Peak ratio ≥ 1.3 | 63 % of wrong redshifts | 21 % of right ones |

R alone cannot do the second job: wrong redshifts at low S/N reached R = 24,
as strong as the right answers.

### Lines and diagnostics

Line ratios against the values the simulator drew, at S/N 30:

| Ratio | Ionisation 0.2 | Ionisation 0.6 |
| --- | --- | --- |
| [N II]/H-alpha | 0.315 vs 0.316 (**0 %**) | 0.790 vs 0.804 (−2 %) |
| [O III]/H-beta | +14 % | +5 % |

The residual [O III]/H-beta bias is stellar Balmer absorption not fully
recovered; before the absorption component was fitted it was +19 % at every
ionisation.

BPT classification across the ionisation sequence, against the region the
drawn ratios actually fall in: **7/7**, including the composite region and
both sides of the Seyfert/LINER division.

### Supernova typing

Types Ia, Ib, Ic and II at three phases each and three signal-to-noise levels,
36 spectra:

| | Result |
| --- | --- |
| Type claimed | 30/36 |
| Of those, correct | **30 (purity 1.00)** |
| Completeness | 0.83 |
| Wrong types claimed | **0** |

Every refusal is a Type Ic, which is exactly right: Ic is defined by what it
*lacks* — no hydrogen, no helium, no strong silicon — so it has the fewest
distinctive features and is genuinely the hardest to match. The classifier
refuses rather than guessing.

The quality statistic behaves differently here than for galaxy redshifts: it
is flat at R ≈ 7–9 from S/N 50 down to 5, and only collapses below 2. For
broad-featured spectra R measures how well the template *shape* matches, not
how good the data are.

Tests: `tests/test_spectra.py`.

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

## Active learning from human verdicts

**Setup.** A pool of 457 labelled stamps standing in for an unreviewed
queue, 201 held out, four classes. Each run starts from the same
randomly-drawn, stratified seed set of 20 — no strategy can select from a
model that does not exist yet — then adds 20 labels per round for four rounds.
Six repeats. The pool's true labels play the astronomer, which is an
idealisation worth naming: the oracle is instant, always right and always
decisive.

**Uncertainty sampling against random selection**, balanced accuracy:

| Labels | Random | Uncertainty | Advantage |
| --- | --- | --- | --- |
| 20 (seed set) | 0.422 ± 0.091 | 0.422 ± 0.091 | — |
| 40 | 0.576 ± 0.103 | 0.554 ± 0.075 | −0.022 |
| 60 | 0.660 ± 0.025 | 0.589 ± 0.088 | −0.070 |
| 80 | 0.737 ± 0.059 | 0.658 ± 0.077 | −0.078 |
| 100 | 0.674 ± 0.149 | 0.720 ± 0.064 | +0.045 |

**Uncertainty sampling lost at three of the four budgets.** An earlier
three-repeat run, with a different seed and a third strategy, found the same
pattern (−0.076, −0.010, +0.034, −0.021).

**Why**, from the composition of what got labelled by 100:

| Strategy | Star | Galaxy | Nebula | Cluster |
| --- | --- | --- | --- | --- |
| Random | 42 | 32 | 15 | 11 |
| Uncertainty | **58** | **22** | 12 | 8 |
| Balanced quotas | 51 | 26 | 13 | 10 |

The decision boundary is crowded with faint stars — numerous, and
individually uninformative — so uncertainty sampling bought more of the
majority class and starved the others.

**The obvious fix also failed.** Quotas per predicted class scored +0.044,
−0.028, −0.055, +0.008 against random, and its composition is barely more
balanced (51 stars against 42), because the quota is on the class the model
*predicts* and early on it predicts the majority class for almost everything.

**Diversity-aware selection**, in the three-repeat run, was worst of the three
(−0.065, −0.040, +0.014, −0.111) with the widest spreads.

So the default is random selection. Nothing tried here beat it, and it has the
property none of the others do: it samples the distribution the model will
actually meet.

Tests: `tests/test_active.py`.

## Learning without labels

**Setup.** 764 unlabelled stamps for contrastive pretraining, 210 labelled for
probing, 201 held out, four classes, a 32-dimensional embedding. Balanced
accuracy throughout. Every augmentation policy is run with three seeds,
because a single contrastive run varies by more than the differences being
compared.

**What the augmentations are worth:**

| Policy | Balanced accuracy | Star/galaxy recall |
| --- | --- | --- |
| Rotation and reflection only | 0.611 ± 0.041 | 0.737 ± 0.009 |
| No PSF blur | 0.720 ± 0.015 | 0.732 ± 0.034 |
| **Default (rotate, flip, shift, noise, blur, brightness)** | **0.755 ± 0.015** | **0.790 ± 0.012** |
| Default plus resized crop | 0.776 ± 0.014 | 0.784 ± 0.016 |

The photometric augmentations carry most of the value: keeping only the exact
sky symmetries costs 14 points, and dropping the PSF blur alone costs 3.5.

**The resized-crop prediction was wrong.** The argument for excluding it is
clean — a scale-changing crop destroys angular size, and angular size is what
separates an unresolved star from a resolved galaxy. Measured, it left
star/galaxy recall unchanged and slightly improved the overall score. The
default remains no crop on the physical argument, not on a measured
advantage, and the switch stays so the question can be reopened with larger
stamps or a wider crop range.

**What the unlabelled data buys**, linear probe on frozen pretrained features
against a network trained from scratch on the same labels:

| Target labels | Probe on pretrained features | From scratch | Advantage |
| --- | --- | --- | --- |
| 10 | 0.471 ± 0.065 | 0.298 ± 0.052 | **+0.173** |
| 25 | 0.605 ± 0.020 | 0.538 ± 0.104 | +0.067 |
| 50 | 0.682 ± 0.053 | 0.598 ± 0.081 | +0.084 |
| 100 | 0.764 ± 0.008 | 0.655 ± 0.146 | +0.109 |

The pretrained probe wins at every budget, and the **spread** is the part
worth noticing: at 100 labels it varies by ±0.008 across draws against ±0.146
from scratch. Pretrained features do not merely score better, they score
*reproducibly* — which is what matters when a real run happens once.

For reference, a probe on the **supervised** embedding trained on all 210
labels reaches 0.855, so self-supervision gets most of the way there from a
fraction of the labels and does not replace them.

**Anomaly ranking**, k-nearest-neighbour distance as the score, on a field
with 49 injected anomalies among 258 objects:

| Embedding | ROC area |
| --- | --- |
| Self-supervised | **0.648** |
| Supervised on four classes | 0.602 |

The self-supervised embedding ranks oddities slightly better, which is what
the argument predicts — a supervised embedding is trained to discard whatever
does not separate its four classes, and an anomaly is by definition outside
them. But 0.65 is a weak detector, and the honest reading is that neither
embedding alone finds anomalies well; this measures representations, not the
anomaly engine, which combines three detectors.

Tests: `tests/test_selfsupervised.py`.

## Explanations

**Setup.** The CNN stamp classifier trained on simulated fields, 40 test
stamps, four classes. Every explanation is scored against the model's own
behaviour rather than against how it looks.

**Saliency maps.** The deletion test erases the pixels a map calls important
and compares the class-score drop against erasing the same number at random.
The area between the two curves is the advantage; positive means the map found
what mattered.

| Method | Deletion advantage | Beats chance | Correlation with the light | Mass on the object |
| --- | --- | --- | --- | --- |
| Grad-CAM | +0.034 | 21/40 | −0.04 | 0.15 |
| Occlusion | **+0.234** | **37/40** | **+0.30** | **0.60** |

A uniform map would put 0.11 of its mass on the central sixteenth of the
stamp, where the object is by construction. **Grad-CAM at 0.15 is barely
distinguishable from uniform.** It is computed correctly — the gradient
weights equal the head weights to 1.2 × 10⁻¹⁰, which is an exact identity for
this architecture — so the failure is not a bug but the method's resolution: a
12 × 12 map on a 48-pixel stamp, one to four cells of which cover a compact
source.

Occlusion is therefore the default, with Grad-CAM kept for speed and
documented as the weaker map.

**The deletion test itself was wrong first.** Filling erased pixels with a
constant narrows the stamp's noise distribution, which the classifier's asinh
stretch renormalises against:

| Fill | Mean advantage | Beats chance |
| --- | --- | --- |
| Constant (the usual choice) | +0.109 | 32/40 |
| Noise from the stamp's own background | **+0.044** | **25/40** |

More than half the apparent effect was an artefact of the ablation. The
noise-preserving fill is the default, and the constant remains available so
the gap can be reproduced.

**Shapley attributions**, on a model with two informative features and six
that carry nothing:

| Quantity | Result |
| --- | --- |
| Both informative features in the top two | 12/12 objects |
| Mean absolute attribution, informative | 0.269 and 0.140 |
| Mean absolute attribution, worst noise feature | **0.003** |
| Additivity residual | 0.005 – 0.010 |

The estimate converges as one over the square root of the draws — max standard
error 0.049 at 50 permutations, 0.025 at 200, 0.013 at 800 — and `converged`
reports whether the tolerance was actually reached rather than assuming it.

**Retrieval**, on the classifier's embedding of 104 test stamps:

| Neighbours | Purity | Chance | Lift |
| --- | --- | --- | --- |
| 1 | 0.904 | 0.352 | 2.57 |
| 3 | 0.859 | 0.352 | 2.44 |
| 5 | 0.863 | 0.352 | 2.45 |
| 3, raw pixels instead of the embedding | 0.587 | 0.352 | 1.67 |

The raw-pixel row is the control that matters: the learned embedding retrieves
better than comparing pixels does, so the explanation is drawing on what the
model learned rather than on the images alone.

Tests: `tests/test_explain.py`.

## Moving a model to another instrument

**Setup.** Two simulated instruments, differing in everything that changes how
an object looks and nothing about what it is:

| | Source | Target |
| --- | --- | --- |
| PSF | Moffat, 3.0 px FWHM | Gaussian, 5.2 px FWHM |
| Background | 120 counts, 6 % gradient | 380 counts, 15 % gradient |
| Read noise | 5.0 | 9.0 |
| Gain | 2.0 | 1.2 |

378 training stamps from the source, 201 source test, 350 target pool, 216
target test, four classes. Balanced accuracy throughout, because the classes
run 4:1 and plain accuracy is mostly a statement about stars.

**The gap.**

| | Balanced accuracy | Per-class recall |
| --- | --- | --- |
| Source model on source test | **0.924** | galaxy 0.87, nebula 0.86, star 0.96, cluster 1.00 |
| Source model on target test | **0.697** | galaxy 0.88, nebula 0.53, star 0.75, cluster 0.62 |

A 23-point drop from an instrument change alone. The galaxies survive it — a
galaxy is extended at either seeing — and the classes defined by their
*profile* against the PSF do not: nebulae fall from 0.86 to 0.53, clusters
from 1.00 to 0.62.

**What it takes to recover**, fine-tuning the head on N target labels, each
budget drawn three times:

| Target labels | Fine-tuned | From scratch on the same labels | Transfer advantage |
| --- | --- | --- | --- |
| 12 | 0.716 | 0.250 | **+0.466** |
| 25 | 0.837 | 0.281 | +0.556 |
| 50 | 0.842 | 0.337 | +0.505 |
| 100 | 0.851 | 0.631 | +0.220 |
| 200 | 0.820 | 0.747 | **+0.073** |

The transfer advantage decays as labels accumulate, which is the expected
shape and the honest headline: pretraining is worth most exactly when labels
are scarce, and by 200 target labels it is nearly irrelevant.

**The spread matters more than the mean at small budgets.** Five independent
draws of 25 labels:

| Budget | Mean | Standard deviation | Range |
| --- | --- | --- | --- |
| 25 | 0.795 | 0.059 | 0.726 – 0.866 |
| 100 | 0.827 | 0.027 | 0.776 – 0.853 |

The single draw in the table above scored 0.837 at 25 labels, which would have
supported "25 labels recover 90 % of the source score". Three of the five
draws do not reach that. The study now repeats every budget and the threshold
must be cleared by the mean.

**Freezing the backbone is right at every budget tested**, which contradicts
the usual expectation that head-only tuning saturates:

| Target labels | Frozen backbone | Whole network |
| --- | --- | --- |
| 25 | 0.716 | 0.720 |
| 50 | **0.747** | 0.629 |
| 100 | **0.835** | 0.647 |
| 200 | **0.860** | 0.798 |

A few hundred examples cannot retrain a network without destroying the
features the source domain paid for.

**Not measured here:** any of this against real survey data. Nothing outside
the package registries was reachable from this environment, so the loaders are
exercised against files written in the same formats and the domain shift is
between two simulated instruments. The method and the shape of the answer are
real; a number for any particular survey is not.

Tests: `tests/test_transfer.py`.

## Survey products

The loader is tested on multi-extension files written in the layouts the
surveys use (`SCI`/`MASK`/`WEIGHT`, `SCI`/`VAR`, a bare primary HDU) and
checked for each thing it must get right:

| Convention | Check |
| --- | --- |
| Weight plane | σ = 1/√w to 1e-6; zero weight masked, not infinite |
| Data-quality plane | every set bit masked by default; `mask_bits` narrows it |
| Saturation | pixels above `SATURATE` masked and counted |
| Pixels in electrons | `BUNIT = electron` gives gain 1, with a note; not applied twice |
| Missing gain | assumed, and said so in the report |
| Variance and weight both present | variance wins |
| `PC` matrix with `CDELT` | rotation honoured; a `CD` matrix still wins when both exist |
| Survey noise plane through preprocessing | a 25-count σ plane survives as 25 ± 5 %, not the 3-count estimate |

The last two rows are regression tests for defects this work found: the WCS
reader silently dropped the rotation of every `PC`-form header, and the
preprocessor overwrote the survey's noise plane with its own estimate so the
photometer never saw it.

**Not measured:** any archive file. Nothing outside the package registries was
reachable, so the layouts are those of files written here.

Tests: `tests/test_survey.py`.

## Agreement with photutils and SEP

Three 512² fields (160 stars, 30 galaxies, no cosmic rays or bad columns so
the comparison is about photometry, not cleaning), the same 3.5 σ threshold,
the same 5-pixel aperture, catalogs matched by mutual nearest neighbour
within 2 pixels:

| Seed | Tool | Matched | Position agreement | Flux ratio (ours / theirs) |
| --- | --- | --- | --- | --- |
| 1 | photutils | 115 of 123 (93 %) | 0.06 px | 0.998 ± 0.011 |
| 1 | SEP | 112 of 120 (93 %) | 0.06 px | 0.997 ± 0.008 |
| 2 | photutils | 131 of 133 (98 %) | 0.08 px | 0.999 ± 0.011 |
| 2 | SEP | 127 of 128 (99 %) | 0.08 px | 0.997 ± 0.010 |
| 3 | photutils | 126 of 130 (97 %) | 0.07 px | 0.999 ± 0.014 |
| 3 | SEP | 124 of 128 (97 %) | 0.06 px | 0.998 ± 0.011 |

Where both codes detect an object they measure the same thing. Against the
simulator's truth all three codes recover the same flux fraction through a
5-pixel aperture (0.921–0.928, the aperture's enclosed energy) with the same
scatter, and recall of objects above 1500 counts is comparable (this package
0.88–0.94, photutils 0.88–0.92, SEP 0.87–0.92).

The one number that looked bad was the "spurious" count: 36–62 detections
per field with no truth object above 1500 counts within 2 px, against 2–12
for the other tools. Traced object by object on seed 1: of 169 detections,
162 sit on a real object of *some* brightness; 55 of those are faint real
objects (median 748 counts) below the cut that the other tools do not report
at this threshold; only about 3 are noise. This package's deblender runs
deeper than the others' defaults. That is a difference in where the threshold
bites, not in correctness, and the truth table is what says so.

What the benchmark did find was a performance defect. Timed on the same
512² fields:

| | Before | After |
| --- | --- | --- |
| Photometry, 169 sources | 17.6 s | 1.1 s |
| Preprocess + detect + photometry, 512² | 18.3–19.5 s | 2.0 s |
| One aperture on a 4096² frame | 1817 ms | 0.31 ms |
| photutils, whole field | 0.2–3.2 s | — |
| SEP, whole field | < 0.1 s | — |

Every aperture, annulus, growth-curve step and Petrosian ring was a
full-frame array, about two hundred of them per source. They now work on the
smallest rectangle containing the aperture; the results agree with the old
code to 4 × 10⁻¹⁶ relative, checked over circles, ellipses, masks, edge
positions and NaNs.

Tests: `tests/test_benchmark.py` (skipped where the tools are not installed).

## Tiled processing

Two 1024² fields (400 stars, 60 galaxies, two nebulae, two clusters),
processed whole and in tiles, catalogs matched within 2 px:

| Seed | Tiles / overlap | Whole | Tiled | Matched | Position | Flux ratio (tiled / whole) | Recall whole → tiled | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 × 328 / 96 | 427 | 430 | 426 | 0.002 px | 1.007 ± 0.017 | 0.913 → 0.913 | 191 → 23 MB |
| 1 | 9 × 419 / 128 | 427 | 428 | 425 | 0.001 px | 1.001 ± 0.016 | 0.913 → 0.910 | 191 → 37 MB |
| 2 | 16 × 328 / 96 | 462 | 462 | 454 | 0.002 px | 1.004 ± 0.036 | 0.925 → 0.937 | 191 → 24 MB |
| 2 | 9 × 419 / 128 | 462 | 459 | 455 | 0.002 px | 1.010 ± 0.028 | 0.925 → 0.928 | 191 → 38 MB |

Sources near a tile boundary are measured no differently from the rest
(flux ratio 1.001–1.010 within 20 px of a boundary, the same as the field).
The residual 0.1–1 % is the per-tile background, and the 3–8 sources found
on only one side are threshold cases near S/N 4, split evenly.

Three earlier states of the merge are recorded because each one looked
plausible and was wrong:

| Merge | Only in tiled | Flux ratio | Cause |
| --- | --- | --- | --- |
| Nearest neighbour within 2 px, remainder tiles allowed | 43 | 1.009 ± 0.032, 33 sources > 5 % off | truncated edge fragments 3–5 px from the whole-image centroid; a 160-px remainder strip 6 % off |
| Core membership, equal tiles, PSF per tile | 4 | 1.064 ± 0.088 | — but the *whole-image* run had no PSF and hence no aperture correction; the tiled one did |
| Core membership, equal tiles, shared PSF | 3–8 | 1.001–1.010 | as above |

Which PSF to use was decided against the truth on the same fields: stars
of 2000–60000 counts recover 1.006 and 0.999 of their true flux with the
shared PSF (tile-to-tile spread of the correction 0.4 %), 1.011 and 1.005
with a PSF per tile (spread 1.1 %), and 0.945 with no correction at all.

Tiled is slower on a frame small enough to do whole (28 s against 20 s):
the per-tile preprocessing repeats what the whole frame did once. It is not
a speed-up; it is the difference between a frame that runs and one that
does not fit.

Tests: `tests/test_tiles.py`.

## Reproducibility

The manifest records the configuration hash, package version, git revision
and dirty flag, Python and platform, the versions of NumPy, SciPy, Astropy,
scikit-learn, PyTorch, photutils and SEP, the seeds, and a checksum of every
input. Tests check that:

- the configuration hash ignores key order and sees a changed threshold;
- the catalog digest ignores source order and last-bit noise (1e-9 px) and
  sees a millipixel;
- a manifest round-trips through JSON with the same reproducibility key;
- `differences()` names a changed seed and a changed configuration;
- **two runs with the same reproducibility key give the same catalog digest**,
  through the full detect-and-measure stage;
- the pipeline attaches the manifest and the catalog digest to every report.

Tests: `tests/test_provenance.py`.

## Environments

The package promises that every optional dependency is a feature and never a
requirement. CI now runs the suite in each environment that promise covers:

| Job | Installed | Python |
| --- | --- | --- |
| numpy-only | NumPy | 3.11 |
| numpy-1.21 | NumPy 1.21.6, the declared floor | 3.9 |
| science | + SciPy, Astropy | 3.11 |
| ml | + scikit-learn | 3.11 |
| deep | + SciPy, Astropy, scikit-learn, PyTorch (CPU) | 3.11 |
| all | + SciPy, Astropy, scikit-learn, Matplotlib | 3.9, 3.11 |
| all+benchmark | + photutils, SEP | 3.12 |

Setting this up found that the NumPy-only job had been red on every push of
this branch. The failing test said a 6 σ threshold found 43 sources where
3 σ found 42; with SciPy the same field gives 44 and 48. The cause was one
word: SciPy's ``reflect`` edge mode repeats the edge sample
(``d c b a | a b c d``) and NumPy's ``reflect`` does not (``d c b | a b c d``,
which SciPy calls ``mirror``), and every fallback mapped the word to itself.
The interiors agreed exactly; only the edges differed. But the background
is estimated on a mesh — 3 × 3 cells for the test field — that has no
interior, so the NumPy-only background model differed everywhere, by up to
3.8 counts on a field with a 10-count sky rms, and six real sources fell
below threshold. The Gaussian filter fallback also truncated its kernel at
3 σ where SciPy uses 4 σ.

After the fix every fallback is compared with its SciPy original, on a
field with marked corners, in every edge mode:

| Primitive | Agreement |
| --- | --- |
| `convolve` (reflect, nearest, mirror, wrap, constant) | 1e-10 |
| `gaussian_filter` | 4e-15 |
| `median_filter` (3, 5, 7) | exact |
| `maximum_filter` (3, 5) | exact |
| `label`, `find_objects`, `binary_dilate` | exact |
| Preprocessed field: background, noise map, PSF FWHM | 1e-8 |

The lesson is the one the whole document keeps teaching: "the results are
identical" was a sentence in the architecture notes, not a test, for as
long as it was wrong.

A second thing the job logs showed was a stream of "I/O operation on
closed file" errors from the logger: the handler kept the ``sys.stderr`` it
was given at configuration, which under pytest is a capture buffer that is
closed after the first test. The handler now resolves the stream per
record.

Tests: `tests/test_fallbacks.py`; the matrix is `.github/workflows/ci.yml`.

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
- Any model against real survey images or crowd-sourced labels. The loaders
  for those formats exist and are tested against files, and the cost of an
  instrument change is measured between two simulated instruments — but no
  external data was reachable from the environment this was built in, so the
  transfer numbers are about the method, not about SDSS or ZTF.

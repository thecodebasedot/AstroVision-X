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
- Real astrometric distortion beyond a TAN projection.
- Crowded stellar fields at globular-cluster densities.
- Any absolute photometric calibration against a standard system.
- Whether the morphological classifier agrees with human classifiers on real
  galaxies, which is the only test that would matter for that stage.

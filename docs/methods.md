# Methods

What each measurement is, why it is done that way, and what it cannot tell you.

---

## Background estimation

The sky is not flat. A 2-D mesh of boxes is built, and in each the sky level is
estimated with the SExtractor mode, `2.5 × median − 1.5 × mean`, which is far
closer to the true sky than a mean in a crowded field because sources pull the
mean up and the median much less. The mesh is median-filtered and interpolated
back to full resolution, giving a background map and an RMS map. Every
threshold downstream is expressed in units of that RMS.

## Cosmic rays

Cosmic rays are sharper than the point-spread function — that is the whole
signal. LACosmic (van Dokkum 2001) exploits it: the image is subsampled 2×,
Laplacian-filtered and re-binned, which responds strongly to single-pixel
spikes and weakly to seeing-limited stars. Dividing that response by a "fine
structure" image built from median filters separates the two populations.

Getting this wrong destroys data. On simulated fields the implementation
recovers 6/6 injected cosmic rays while flagging zero bright stars.

## Bad columns

A defective column is bad along its *whole* length; a bright source is not.
Two conditions are required together: the column median must deviate from its
neighbours by more than the median's own uncertainty allows, and a majority of
the column's pixels must deviate in the same direction.

The single-condition version of this test was found, during validation, to flag
78 % of an image and to erase injected transients. The uncertainty floor
matters: on a flat profile the measured scatter collapses toward zero and
everything clears a 6 σ cut.

## The point-spread function

The PSF is measured empirically by stacking isolated, unsaturated stars. The
hard part is not the stacking but the selection: a field with bright galaxies
will happily supply "stars" that are small galaxies, and the resulting PSF is
too broad — which then biases every profile fit and every star/galaxy
separation that depends on it.

Candidates are therefore cut against the **stellar locus**: point sources share
one PSF, so their sizes cluster tightly while galaxies spread upward. Seeding
on the smallest candidates and iterating converges onto that locus even when
galaxies outnumber stars. The second-moment box is sized from the seeing
itself, because in poor seeing a fixed box saturates and stars stop being
distinguishable from small galaxies.

Median error across 60 simulated fields: 6 %. Where fewer than five point
sources survive the cut, the pipeline warns that the PSF is poorly constrained
rather than silently proceeding.

## Detection

Detection runs on a matched-filtered copy of the image: convolving with a
kernel matched to the source profile maximises signal-to-noise for point
sources. The threshold is scaled by the kernel's RMS so it stays at N σ.

Connected regions above threshold are labelled, then **deblended**. Following
SExtractor, each region is re-thresholded at exponentially spaced levels;
branches that split off and carry more than a contrast fraction of the total
flux become separate objects, and the remaining pixels are reassigned to
whichever branch dominates there.

## Photometry

Apertures use *fractional* pixel coverage — each boundary pixel is subdivided
and the covered fraction computed — because at the few-pixel radii typical of
astronomical sources a binary mask biases the flux by several percent.

An adaptive aperture is set from the **Kron radius**, the first moment of the
light: `2.5 × R₁` captures roughly 94 % of a galaxy's light regardless of
profile shape. The **Petrosian radius**, where the local surface brightness
falls to a fixed fraction of the mean inside it, is independent of distance and
of exposure depth, which is why surveys use it for sizes.

Real PSFs have wings, so any practical aperture loses a few percent. The
**aperture correction** — the enclosed energy of the measured PSF at the same
radius — removes that bias. Without it, isolated-star photometry came out 5 %
faint; with it, within 3 %.

## Morphology

**CAS** (Conselice 2003). *Concentration* `C = 5 log₁₀(r₈₀/r₂₀)` is about 5 for
a de Vaucouleurs bulge and 2.7 for an exponential disc. *Asymmetry* is the
residual after a 180° rotation, with the rotation centre minimised because the
result is very sensitive to mis-centring. *Smoothness* is the fraction of light
in high spatial frequencies, with the intrinsically sharp nucleus excluded.

**Gini and M₂₀** (Lotz 2004, 2008). Gini measures how unequally light is
distributed among an object's pixels; M₂₀ measures how far the brightest fifth
of the light sits from the centre. Neither assumes circular symmetry, which is
why together they catch mergers and double nuclei that CAS alone misses.

Both are measured over a **Petrosian-defined footprint** — the image smoothed
and thresholded at the surface brightness at the Petrosian radius — rather than
over the raw isophotal detection footprint. That definition scales with each
galaxy instead of with the depth of the exposure, which is what makes the
statistics comparable between objects.

**The Sérsic index.** Three things had to be right for the fit to work:

1. The model must be **convolved with the PSF** before comparison. Seeing
   flattens the central cusp, and fitting an unconvolved model gives every
   concentrated galaxy an index far too low (4.0 was recovered as 1.4).
2. The fit region must be **scaled to the object** — a few effective radii.
   Restricted to the bright isophote it loses the outer profile that
   constrains the index; extended over the whole cutout, the optimiser trades
   radius against index to absorb sky noise.
3. `r_eff` must be **pinned by the measured half-light radius**. The index and
   the radius are strongly degenerate, and `r_eff` *is* the half-light radius
   by definition, so the direct measurement fixes the degenerate direction
   without constraining the shape.

With all three: 11 % median error on the index across 1.0 ≤ n ≤ 6.0.

**Spiral arms and bars.** In log-polar coordinates a logarithmic spiral becomes
a straight ridge, and an azimuthal Fourier decomposition measures how many arms
there are. But amplitude alone does not identify arms: a flattened galaxy
leaves an `m = 2` residual whenever the assumed axis ratio is slightly wrong,
and a bar produces a large `m = 2` too.

What makes a spiral a spiral is that the mode's phase **winds** with radius —
`dφ/d ln r = m / tan(pitch)` is constant for a logarithmic spiral, while an
ellipse or a bar holds its phase fixed. Arms are confirmed only when the
amplitude clears a noise floor (estimated from the high-order modes, which
carry no two- to four-armed signal) *and* the phase winds coherently. The same
test, inverted, is what identifies a bar.

## Classification

Star/galaxy separation has a clean physical basis: a star is a point source, so
its light profile *is* the PSF. Every test is expressed as a ratio to the
measured PSF, so the same thresholds hold whatever the seeing was. The
half-light radius separates the populations most cleanly, followed by
peak-to-total flux, then isophotal width.

The thresholds sit between the two loci as measured on simulated fields. This
detail is the difference between 42 % and 90 % accuracy: an early version put
them on the wrong side of the stellar locus and classified most stars as
galaxies.

Nebulae and star clusters are defined *relative to the field's own galaxies* —
more diffuse, or brighter and granular — which keeps them independent of the
zero point and the exposure depth. They remain the weakest part of the
classifier, because they genuinely overlap galaxies in every measured statistic.

## Colours, and the stellar locus

Morphology answers "is this resolved?" and answers it well when the object is
bright. It stops working exactly where survey catalogs get interesting — near
the detection limit, where a small galaxy and a star are both a few pixels of
noise-dominated light and every size statistic collapses onto the same value.

Colour does not degrade the same way, because it is not a size measurement.
Stars are approximately blackbodies behind the same filters, so their colours
fall on a one-parameter curve, the **stellar locus**. Galaxies are integrated,
redshifted stellar populations and sit off it.

Three decisions carry that idea into something usable.

**The locus is fitted from the field itself**, not taken from a table. That
makes the test independent of the zero point, the filter's exact throughput,
and any reddening common to the field — all of which move the locus bodily
without changing the fact that stars lie on it. Morphological stellarity seeds
the fit, and the seed is deliberately the *unfused* value: a locus seeded by
its own output would confirm itself.

**The widths are fitted from the field too**, and for a harder reason. The
test compares how far a source sits off the locus against how far it could sit
off it by chance, so it needs that second number. Formal photometric errors
are not it — measured against truth they come out about 2.5 times too small,
because they count photon and read noise but not sky estimation, blending, or
the residual of matching one band's PSF to another. Both populations' widths
are therefore measured directly, in signal-to-noise bins, from the field's own
point-like and resolved sources.

**The test is a likelihood ratio, not a sigmoid.** This is the decision the
whole thing turns on. A sigmoid of the locus offset saturates near 1 for every
small offset, so a test with no discriminating power votes "star" for
everything — including the galaxies it exists to catch. Comparing two
hypotheses instead makes an uninformative test return 0.5 on its own, which
carries no weight when it is fused with morphology in log-odds. Both
hypotheses are mixed with a broad outlier component, because the Rayleigh tail
is far too thin for real data: a blended star lands five sigma out and would
otherwise be convicted with certainty.

Finally the fusion weight is not a constant. Each field measures how well its
own colour test separates the morphological classes, and weights it
accordingly — to zero when the answer is chance. A weak, noisy vote added at
full strength to a strong one costs accuracy in every field where the colours
happen to be poor, and no single constant can be right for both a shallow
two-band field and a deep five-band one.

## Forced photometry

A colour is a *difference of magnitudes measured the same way*, and almost
everything that goes wrong with colours goes wrong because that condition was
quietly broken. Three ways, three fixes:

- **Different apertures.** Detecting independently in each band gives each one
  its own centroid and Kron radius, so the two apertures sample different parts
  of the same galaxy. One aperture, defined once in the detection band, is
  applied at the same *sky* position everywhere.
- **Different seeing.** A fixed aperture catches more of a point source in good
  seeing than in bad, so a star observed in 1.0" and 1.6" acquires a colour it
  does not have. Every band is convolved to the worst PSF in the set — blurring
  is stable, sharpening is not. Each band is then corrected by the enclosed
  energy of its own *post-matching* PSF, because a Gaussian kernel applied to a
  Moffat leaves something that is not quite either.
- **Different pixel grids.** Apertures are specified in arcseconds and
  converted per band.

A colour is recorded only when both bands clear a signal-to-noise floor. A
source detected at 40σ in the red and 1σ in the blue does not have a faint blue
measurement; it has noise, and left in, such values dominate the scatter of any
colour cut. What it does have is a one-sided limit — it is at least that red —
and that is what gets stored.

## Calibration

The WCS that comes with a file is a starting guess. Pointing models are
imperfect, focal planes flex, and a telescope reporting its position to an
arcsecond is doing well. The plate solution matches detections to reference
stars under the current guess, fits, then re-matches with a radius drawn from
that fit's own residual and refits. Matching is *mutual*-nearest: one-sided
matching quietly assigns several detections to one bright reference star in a
crowded field, and those duplicated pairs pull the fit toward that one star.

Distortion is handled by SIP coefficients applied, per the convention, to the
offset from the reference pixel and *before* the linear matrix. Getting that
order wrong is the classic SIP bug: applied to absolute pixel coordinates it
leaves an error that grows across the detector and looks exactly like a bad
plate scale. The inverse is found by fixed-point iteration rather than by
requiring reverse coefficients, which many real headers do not carry.

A zero point is fitted as `m_catalog = m_instrumental + zp + k·colour`. The
colour term is not optional refinement: a filter is never exactly the reference
survey's filter, so the offset between systems depends on the shape of the
source's spectrum. Fitting only a constant leaves that dependence in the
residuals, where it becomes a systematic, colour-dependent error in every
magnitude. It is fitted only when the standards span enough colour to
constrain it — a slope fitted to points all at the same colour returns a large
coefficient with a small formal error, which is the worst possible pairing.

## Knowing what is already known

Everything this package calls a *candidate* rests on one unstated claim: that
the object is not already known. Without checking, an anomaly ranking is a
list of the field's oddest objects, which is a different thing and is mostly
catalogued variable stars, asteroids on their published ephemerides, and
galaxies someone measured in 1991.

One cone covering the whole image is fetched once and matched locally — the
other way round is one HTTP request per source, which is slow for the caller
and rude to a service that is free to use. The backend is pluggable and the
default does nothing, so a pipeline never silently stalls on an unreachable
service; a local reference file works with no network at all.

The report distinguishes three states a naive implementation conflates:
*checked and matched*, *checked and not matched*, and *not checked* — where
the last includes a cone that came back empty, which establishes nothing.
Matched sources keep every measurement and lose their claim to novelty: an
anomaly is demoted by a factor of four, a transient barely at all, because a
supernova going off in a known galaxy is the normal case rather than a reason
for suspicion.

## Uncertainty

A Sérsic index of 3.8 means something entirely different at ±0.2 than at ±2.1,
and at moderate signal-to-noise the second is the common case.

Fitted parameters get their errors from the curvature of the fit's own
chi-squared surface, scaled by the achieved chi-squared — the formal errors
assume the model is correct and the noise estimate exact, and a real galaxy is
not a Sérsic profile. What matters more than the marginal errors is the
*correlation*: these fits are effectively degenerate in `n` against `r_eff`,
so a source whose worst pair correlates above 0.95 carries a flag saying so.

The non-parametric statistics have no Jacobian, so their errors come from
re-measuring the object on repeated noise realisations at the image's own
measured noise. This also exposes bias, not just scatter: asymmetry is built
from absolute differences, so noise pushes it upward whichever way the noise
goes, by about 0.13 in a typical faint source.

Classifier confidences are numbers between 0 and 1, which does not make them
probabilities. Isotonic regression or Platt scaling — chosen by how much
labelled data there is, since isotonic overfits below a few hundred points —
maps them to values that mean what they say. Platt scaling belongs on the
log-odds scale; applied to probabilities directly it makes calibration worse.

## Difference imaging

Subtraction removes every constant source and leaves what changed. Getting it
clean is the whole problem: the epochs must be aligned to a fraction of a
pixel, matched in PSF, and scaled to a common flux system.

**Alignment** is done by star-pattern matching, which needs no prior alignment
and tolerates rotation: pairs of stars vote for a rotation and scale, a
histogram of residual offsets fixes the translation, and inliers are refined by
least squares. Cross-correlation with an upsampled-DFT peak refinement is the
fallback, accurate to about 0.15 px.

**PSF matching** always blurs the *sharper* epoch. Deconvolving the blurrier
one would amplify noise and ring around every bright star. When either PSF is
built from fewer than five stars, matching is skipped entirely — matching to a
wrong PSF is worse than not matching at all.

**Flux scaling** is a least-squares fit through the origin over pixels well
above the noise in both frames. The obvious alternative, a median of per-pixel
ratios, is biased upward by Jensen's inequality wherever the denominator is
noisy — which is exactly the faint end of any threshold cut. In validation that
bias reached 10 %, and a 10 % scale error leaves a 10 % residual at every star:
50 σ at a bright one, indistinguishable from a transient.

**Templates hold out the epoch being searched.** Including it would put a
fraction of any transient into the very template used to subtract it.

## Real/bogus vetting

Any difference image produces far more artefacts than transients. Nine features
are measured per candidate — peak significance, PSF correlation, dipole ratio,
negative fraction, sharpness, elongation, centroid offset, flux ratio, area —
and combined into a score.

The combination is a **weighted geometric mean**, not an arithmetic one.
Vetting is a veto: an object that fails one test decisively — a clean dipole, a
one-pixel cosmic ray — is bogus no matter how well it does on the rest, and an
arithmetic mean lets the other terms outvote that. With the geometric mean,
genuine transients score above 0.84 and artefacts below 0.72 on simulated data,
which is where the default threshold of 0.7 comes from.

## Variability

Each statistic attacks the same question — did this object really change? —
from a different angle. Reduced χ² tests the scatter against the quoted errors.
Stetson's J uses consecutive-epoch correlations: a real variable brightens and
fades coherently, so neighbouring epochs deviate in the same direction while
noise does not. The von Neumann ratio η falls below 2 when consecutive epochs
are correlated.

Periods come from **Lomb–Scargle**, which fits a sinusoid at each trial
frequency and so handles the irregular sampling that weather and scheduling
impose — an FFT cannot. False-alarm probabilities use the standard analytic
expression with an independent-frequency correction; a bootstrap is preferable
before publishing.

## Strong lensing

A lens candidate must be both a plausible deflector — massive, early-type,
concentrated — *and* surrounded by tangential arcs at a consistent radius.
Either condition alone produces mostly false positives.

The deflector's own light is removed by an azimuthal baseline at each radius. A
**low percentile, not the median**: several arcs can fill most of the azimuth
at the Einstein radius, and a median baseline then subtracts the very signal
being searched for. A genuinely complete Einstein ring defeats any azimuthal
baseline, and is found instead by a radial scan — a ring is a bump in the
azimuthally-averaged profile of a galaxy whose light otherwise falls
monotonically, and it is uniformly filled in azimuth.

Arcs must be *tangentially* elongated: a radial streak of the same shape is a
diffraction spike or a merging companion. Multiple arcs sharing one radius is
the strongest single signal, because a chance alignment of neighbours has no
reason to share one.

The implied velocity dispersion is computed for a singular isothermal sphere as
a sanity check — a candidate implying 600 km/s is not a galaxy-scale lens — but
without redshifts the distance ratio must be assumed, and the report says so.

## Novelty

Three detectors with different blind spots are combined: an isolation forest
works on raw feature geometry, an autoencoder on whether a low-dimensional
model can reproduce the object, and a k-nearest-neighbour distance on whether
the object has any analogues at all. Scores are rank-normalised, so the final
number reads as "more unusual than this fraction of the field".

Each record carries a written explanation of *why* the object stands out. An
outlier score measures dissimilarity from this field, not physical novelty, and
instrumental artefacts score highly too — which is why the recommended first
step is always visual inspection.

## What none of this can do

Nothing here confirms anything. A transient candidate needs an independent
epoch and, for a supernova, a classification spectrum. A lens candidate needs
colour information — lensed sources are typically bluer than their deflectors —
and ultimately spectroscopic redshifts for both. A morphological type from a
single band at survey depth is a useful prior, not a measurement. Physical
quantities derived from an assumed redshift are order-of-magnitude estimates,
and every assumption is listed in the report that quotes them.

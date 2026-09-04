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

## A PSF that changes across the field

One PSF for a whole frame is a convenient fiction. Optics degrade off-axis
and a wide-field camera is easily twenty percent blurrier in the corners.
Everything downstream inherits that error, and the aperture correction is the
clearest case: derived from a field-average PSF it is wrong at the centre and
wrong at the edges *in opposite directions*.

Each pixel of the PSF stamp is a low-order polynomial in field coordinates,
fitted across the frame's own stars. Every stamp pixel shares one design
matrix, so the whole thing is a single least-squares solve.

Four details decide whether it works, and each was forced by a measurement
that failed first.

**It validates itself.** A quadratic has six free parameters per stamp pixel
and will always fit its training stars better than a constant. Whether it is
*better* is answered by holding stars out — seven splits, and the varying
model must win in the worst of them, not on average. A single hold-out is not
enough: on a field whose PSF does not vary at all, one split claimed an 11.8%
improvement and the model invented a hundred-percent variation across the
frame.

**Star selection has to be regional.** The existing selector keeps the
sharpest sources, which is the right way to reject small galaxies when the
PSF is constant and exactly the wrong way when it is not — a star in a blurry
corner is broader than a galaxy at the sharp centre. Selecting within tiles
keeps the galaxy rejection while sampling the whole field. The tiles have to
be small enough that the PSF really is nearly constant inside one: on a field
with a real 40% variation, three tiles a side found nothing and four found it
cleanly.

**The comparison must be weighted by the PSF profile.** A 21-pixel stamp of a
3-pixel PSF is mostly empty sky, and an unweighted residual over all 441
pixels is dominated by wing noise. Unweighted, the metric could not tell the
varying model from the constant one on a field with a genuine 34% variation.
For the same reason the stamp is sized to 2.5 times the measured seeing
rather than left at a fixed 25 pixels — every extra ring of wing pixels adds
noise without adding signal.

**It never extrapolates.** Evaluation positions are clamped to the bounding
box of the stars the fit actually saw. A quadratic run past its last star
returned FWHMs twice the true value.

### What was tried and not shipped

Position-dependent PSF *matching* in difference imaging is the obvious next
step, and it was built and measured: tiled matching with hard tile edges,
then with smooth inverse-distance blending, then with the template's spatial
PSF refitted rather than inherited from one epoch. Every variant made the
spurious candidate count worse — from 18 to between 45 and 114 on the same
field. It is not shipped, and no configuration switch offers it.

The honest reading is that a matching kernel good enough to beat one global
kernel has to be derived from the difference itself, in the manner of a
proper image-subtraction basis, rather than assembled from per-tile PSF
models. That is a larger piece of work.

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
result is very sensitive to mis-centring, and with the asymmetry of the sky
subtracted, because noise contributes an absolute difference to every pixel
whichever way it goes: without that term the statistic ranks faint smooth
galaxies above bright disturbed ones, which the statmorph comparison in
`validation.md` caught as a rank correlation of −0.8. *Smoothness* is the
light in structure smaller than a quarter of the Petrosian radius — a boxcar
smoothing, positive residuals only, an annulus that excludes the sharp
nucleus, sky term subtracted — following Lotz et al. (2004). Both are
normalised as statmorph normalises them, so the numbers are comparable.

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

## Spectroscopy

A spectrum is where most of the statements in this package stop being
inferences and start being measurements — a redshift good to 20 km/s instead
of 4 %, a line ratio that names the ionising source, a supernova type. The
price is that far more can go wrong before the numbers exist at all.

### From a frame to a spectrum

The spectrum is not a row of the detector. It is a curved path, and summing a
fixed band of rows loses flux at the ends of the order — a smooth,
wavelength-dependent loss indistinguishable from a broad spectral feature. The
trace is centroided in binned column groups, then fitted with a low-order
polynomial: a per-column centroid is itself noisy, and following that noise is
as bad as ignoring the curvature.

The sky is brighter than the target over most of the optical and its emission
lines are far brighter, so it is estimated in each column separately from rows
away from the object, robustly, with a low-order fit along the slit.

Extraction is profile-weighted after Horne (1986), each pixel weighted by
profile over variance. What that buys is worth stating precisely rather than
by reputation: against the *best* fixed aperture it gains 3 % in
signal-to-noise, and against a merely reasonable one 16 %. A Gaussian profile
is forgiving, and a well-chosen aperture already recovers most of what is
there. The real gains are not having to know the right aperture in advance,
and cosmic-ray rejection — a pixel far from the profile-scaled model is a hit,
not a photon, and dropping it does not cost the column.

### Wavelength calibration

The dispersion is not linear, and a linear solution leaves several Angstroms
of systematic residual at the ends of the order — at 5000 Å that is a redshift
error larger than the statistical error of a good cross-correlation.

The harder problem is identifying the arc lines in the first place. Matching
peaks to their nearest predicted wavelength needs a first guess, and a linear
guess for a non-linear dispersion is wrong by tens of Angstroms mid-detector,
which is comparable to the line spacing. Lines then get identified one over,
the polynomial absorbs the error, and the residual settles at a few Angstroms
at *every* order — including the order that generated the data. So the
identification does not start from a guess. Every pair of detected peaks is
compared with every pair of catalogue lines and votes for the linear solution
it implies; the right answer collects votes from many independent pairs, a
wrong one collects a few by chance. This is the same idea as pattern-matching
an astrometric field: use the relative geometry, which is invariant, not the
absolute positions, which are what is unknown.

A solution is refused when its residual exceeds 0.3 of a resolution element.
That threshold is not taste: a correct identification lands at the centroiding
floor of about 0.5 Å here, and a one-line misidentification lands at 3.7 Å.

The arc is exposed at a different time and often a different telescope
position from the science frame, and flexure between them shifts the solution
bodily. The night-sky lines are *in* the science frame at known wavelengths,
so they measure that shift directly — the check that catches a calibration
which is internally perfect and externally wrong. It needs the sky spectrum,
not the sky-subtracted one: after subtraction only residuals remain, and a
residual's centroid describes how the subtraction failed.

### Redshifts

A spectroscopic redshift is a *pattern* match, not a measurement of one line.
Twenty absorption lines all shifted by the same fraction is an enormously
stronger statement than any one of them, and the cross-correlation is the
arithmetic that adds them up. Both spectra go onto a grid uniform in log
wavelength, where a redshift is a pure shift rather than a stretch, so one
correlation covers every redshift at once.

Three things decide whether the peak means anything.

**The continuum must exist before it can be divided out.** Every spectrograph
has ends where the throughput collapses, and a redshifted object often has no
flux at all over part of the detector. Dividing the remaining noise by a
continuum near zero does not produce a faint spectrum, it produces a loud one:
measured here, the normalised flux reached 2000 where real features reach 0.3,
and the correlation returned a confident and entirely wrong redshift. Pixels
without a continuum are masked.

**R must be measured the way Tonry & Davis defined it.** A true match produces
a *symmetric* correlation peak, because it is the autocorrelation of the
template's own features; noise produces a lopsided one. Comparing the peak
with the roughness of the background instead — the obvious simplification —
gives R = 18.7 on pure noise, because the correlation of noise with a smooth
template is itself smooth. Measured over the full overlap, the antisymmetric
construction separates the two cleanly: forty noise spectra gave R from 3.6 to
6.6, correct redshifts start at 5.6 with a median of 13.

**R is not enough by itself.** At low signal-to-noise, wrong redshifts reach
R = 24 — as strong a correlation as the right answers — because a catastrophic
failure is not a weak match, it is a confident match to the wrong feature.
What separates them is whether a rival explanation exists, so the winning peak
must also lead the best peak at a different redshift by 30 % in R. Emission
lines can overrule that: they are an independent identification, not another
peak in the same correlation.

The correlation measures a redshift, not a classification. A quasar here is
matched by the starburst template — its narrow [O III] lines correlate better
than its broad Balmer lines, at every continuum window tried — and the
redshift is right anyway. The exception is stars, which are searched only
within 3000 km/s of rest, because a star has a radial velocity and not a
redshift; without that restriction a G-star template won the fit for a galaxy
at z = 0.39, reporting the right redshift for entirely the wrong reason.

### Lines

Lines closer together than the widest profile the fit may give them are fitted
*together*, sharing a velocity width — which is also what the physics says,
since the same gas emits them. Getting that threshold from the instrumental
resolution instead put [N II] 6584 just outside H-alpha's group, 20.7 Å apart
against a 20 Å threshold; fitted alone it widened to the top of the velocity
search, absorbed H-alpha's flux, and the [N II]/H-alpha ratio came out
inverted — falling as the simulated ionisation rose.

A Balmer emission line sits inside the stellar absorption trough of the same
transition, and the trough is far wider than the line, so measuring the
emission against a smooth continuum measures emission minus absorption. That
error does not cancel in a ratio: [O III]/H-beta came out 19 % high at every
ionisation. A broad component under each Balmer line fixes it — but only where
it is identifiable. The trough is barely 1.5 times wider than the line it
holds, so with another emission line 20 Å away it trades off freely against
that neighbour; adding it under H-alpha moved [N II]/H-alpha from correct to
44 % high. It is fitted for the isolated Balmer lines and not for H-alpha.

A line that is not there still returns a fitted amplitude whose sign is
decided by noise. Below three sigma the fit reports an upper limit instead,
and a ratio built from one is labelled a limit rather than plotted as a point.

### What the lines mean

The BPT diagram separates gas ionised by young stars from gas ionised by an
accreting black hole using two ratios of *adjacent* lines, so reddening and
flux calibration cancel and the diagram works on uncalibrated data. Between
the empirical star-forming boundary and the theoretical maximum-starburst line
lies a real composite region where both contribute; reporting a composite as
one or the other is not a rounding error but a different physical claim, so it
is named. Past the asymptote of those curves every object is on the AGN side —
treating the curve as infinitely high there classifies the hardest-ionised
objects in a sample as star-forming.

Supernova typing matches against templates in both type and phase, because the
features move as the ejecta slow: a Type Ia a month after maximum looks nothing
like one a week before it. A type is reported only when the match is strong
*and* leads the best rival type by a margin; otherwise the answer is that the
spectrum does not choose, which is what a coin toss deserves instead of a
label.

A spectral match is a classification of a spectrum. Whether the object is a
supernova at all is settled by a light curve, a host association and a person,
and the record says so on every confident type.

## Photometric redshifts

A distance from four numbers. It works because a galaxy spectrum is not
featureless: an old stellar population drops sharply shortward of 4000
angstroms, a young one carries strong emission lines, and as the galaxy
recedes those features slide through the filters. The pattern of broad-band
fluxes therefore depends on redshift, and fitting a library of redshifted
spectra recovers it.

The integral that turns a spectrum into a magnitude is weighted by `1/λ`,
the photon-counting convention, because a CCD counts photons rather than
energy. The energy-weighted version shifts every magnitude by a few
hundredths — and shifts them by *different* amounts in different filters,
which is a colour error and therefore a redshift error.

Three things the arithmetic makes it easy to hide, and this implementation
does not.

**The posterior is often bimodal.** A red galaxy at low redshift and a blue
one higher up can produce the same colours, because a 4000 Å break sitting
between two filters looks much like a red continuum. Reporting only the peak
converts a known ambiguity into a confident wrong answer. Both peaks are
reported; a source whose second peak carries a quarter of the weight is
flagged ambiguous and is not called reliable.

**The posterior width is not the error.** It is the error *given the
template library*, and no six templates describe every galaxy. Measured
against simulated truth the width came out about three times too small, so a
floor is added in quadrature rather than the model being quoted as if it
were right.

**The estimate is the posterior mean, not the χ² minimum.** With a bimodal
posterior the minimum sits in whichever peak happens to be a hair deeper and
flips between them on noise.

The library also reports the redshifts at which the break actually lies
inside one of the filters. Outside that range the colours change slowly with
redshift and dust can imitate the continuum slope; that is where the errors
live, and it is a property of the filter set rather than of any algorithm.

### The filter count is the whole story

With `g, r, i` there are two colours and at least three unknowns — redshift,
spectral type, dust. The problem is underdetermined, and no care in the fit
changes that. What care does is make the failure visible: measured on 400
galaxies drawn from spectra the library does not contain, three filters give
a 22% catastrophic-outlier rate and five give 2.8%.

The templates the simulator draws from have *continuous* age, dust and
emission parameters while the fit library has six discrete entries, so no
simulated galaxy is ever exactly reproducible by the fit. That is the
situation with real galaxies, and it is the only way the measured scatter
means anything rather than measuring a lookup.

### What it replaces

Every distance-dependent quantity — physical size, absolute magnitude,
luminosity — previously inherited one assumed redshift for a whole field.
Each galaxy now carries its own, with its own error, and the report's
assumptions section says how many were measured rather than assumed. A
galaxy whose photo-z is unreliable still gets one, and everything derived
from it carries the flag that says so.

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

## Moving objects

A difference image finds an asteroid as readily as a supernova. What tells
them apart is that one of them is in a different place every time — and that
difference is destructive, not helpful, to a transient pipeline: the
position-based merging that consolidates a real transient into one candidate
*scatters* a mover into N separate single-epoch detections, each looking like
a marginal unconfirmed transient. One asteroid crossing five epochs becomes
five entries in a follow-up queue, none of them real.

Linking puts it back together. Every pair of detections from two epochs
implies a velocity; those within the searched rate range are extrapolated to
the other epochs, and whatever lands on the prediction joins the track.

Three decisions do the real work.

**Rates are in arcseconds per hour, not pixels per epoch.** A rate in pixels
says as much about the camera as about the object, and the cut that keeps the
search tractable is a physical one: a main-belt asteroid near opposition moves
at roughly 30 arcsec/hour. It also sets the cadence — at that rate an object
crosses a two-arcminute field in four hours, so a series taken two nights
apart never sees the same asteroid twice. Asteroid linking is a within-night
operation, and the simulator's units had to be right before any of this could
be tested. They were not, at first: a factor-of-24 slip left injected
asteroids crossing half a pixel per night, indistinguishable from stationary
sources.

**The residual is reduced by the degrees of freedom.** A tracklet fits four
parameters, so three detections leave two degrees of freedom and five leave
six — and a three-point track therefore has a *smaller* raw residual than a
five-point one for identical astrometric errors. Comparing raw residuals
rewards the shorter, weaker link. Dividing by `sqrt(1 - 2/n)` widened the
measured gap between real and chance tracklets from a factor of five to a
factor of seven and removed the only spurious tracklet in the validation set,
without tuning a threshold.

**The chance rate is estimated, not assumed.** Unrelated detections do line
up. The expected number of coincidental tracklets follows from the field's own
detection density, the matching tolerance and the number of epochs, and it is
reported per run and per tracklet. Twenty detections an epoch over three
epochs with a three-pixel tolerance yields about 2.7 expected false tracklets
— which is not a defect in the estimate but a fair description of what three
epochs buy.

### Trails

An object that moves while the shutter is open leaves the PSF smeared along
its track. That is a second handle, and its value lies in being *independent*:
linking works across exposures, a trail works within one. When both agree on
a direction, coincidence becomes a much worse explanation.

Measuring it correctly took two corrections. Comparing a source's
second-moment size against the fitted PSF FWHM is comparing two different
quantities — a Moffat's second moment far exceeds its FWHM, because the wings
carry weight a FWHM ignores — and the subtraction reported a multi-pixel trail
on a perfectly round star. The measurement is now the source's own major axis
against its minor, which cancels the profile shape and the seeing together.
And moments must be taken in a bounded window: unbounded, on a faint source,
clipping the negative half of the noise leaves a positive floor across every
pixel whose second moment is the *stamp's* size, and a round point source was
credited with a 41-pixel trail.

A trail is finally required to be an elongation, not merely an excess. A
noise-dominated source has two large but nearly equal axes whose quadrature
difference is still several pixels while the source is round to two percent.

### What a tracklet is not

A tracklet is not an object and certainly not a discovery. It is a set of
detections consistent with linear motion over a short arc. Turning that into
an object needs an orbit; turning it into a *new* object needs the Minor
Planet Center. The stage says so in its log line and the report repeats it.

Movers are demoted inside the transient list rather than deleted from it. The
tracklet is an interpretation of those detections, and an astronomer who
disagrees needs to see what was interpreted.

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

### The mass model

Detecting arcs says a lens may be there. Measuring the mass needs a model, and
the model used is a **singular isothermal ellipsoid with external shear** —
the standard first model for a galaxy-scale lens, because an isothermal
profile is what stellar dynamics and lensing independently find for early-type
galaxies, and because its deflection has a closed form (Keeton 2001) rather
than needing an integral at every pixel. As the lens becomes round the
ellipsoid formulae divide by `sqrt(1 - q²)`, so below a set flattening the
code switches to the exact isothermal *sphere* limit instead of computing
zero over zero.

External shear is the tidal field of everything else along the line of sight,
represented as a traceless stretch: it distorts images without adding mass.
It is nearly always needed — a lens with no neighbours is the exception — and
leaving it out makes the ellipsoid absorb the stretch, which measurably
flattens the fitted mass.

The fit minimises **source-plane scatter**: every arc position is mapped back
through the lens equation `β = θ − α(θ)`, and a correct model sends them all
to the same place. This is far cheaper than the image-plane alternative, which
would need the lens equation solved for every trial model. The price is that
source-plane scatter is weighted by magnification, so it is quoted alongside
the image-plane residual, which is in pixels and means what a reader expects
it to mean.

Two things about that fit are worth stating plainly, because both were
measured rather than assumed:

**Ellipticity and shear are degenerate, and coverage is what breaks the
degeneracy.** Both stretch images; separating them requires seeing the stretch
from more than one direction. Fitting ray-traced images spanning 140° to 267°
around the lens, everything above about 220° recovered the axis ratio to
within 0.06 and the shear to within 0.05, while everything between 140° and
165° with the shear free collapsed to an axis ratio near 0.2 with a shear of
0.4 — the lens's own flattening reappearing as a fictitious tidal field. So
the shear is only fitted when the arcs span at least 200° around the
deflector, and the report says when it was held at zero and why.

**A large fitted shear is tested, not assumed to be wrong.** When the free fit
still wants a shear beyond 0.3, the model is refitted without it and the two
are compared. If dropping the shear barely worsens the fit, the shear was
buying nothing and is dropped; if dropping it wrecks the fit, the shear is
real and is kept with a warning. In the measured cases the two are far apart:
a spurious shear costs a factor 1.0–1.7 in scatter to remove, a real one 16–30.

Errors come from a **bootstrap over the arc positions**, not from the
curvature of the objective. With few positions and strongly correlated
parameters, a curvature estimate understates the error badly — it describes a
parabola the likelihood does not have.

Finally, the fit refuses rather than overfits. Each position supplies two
numbers but also costs the two unknowns of the shared source position, so *N*
positions give `2N − 2` constraints; below the parameter count the routine
returns no model and records why. A candidate with one short arc stays a
candidate — it is simply one that cannot be weighed.

### From an Einstein radius to a mass

The projected mass inside the Einstein radius follows directly:

    M_E = (c² / 4G) · (D_L D_S / D_LS) · θ_E²

with no assumption about how the mass is distributed — that is the reason this
quantity, rather than a total mass, is the one lensing measures well. What it
does need is both redshifts. The deflector's may be photometric, from the
redshift stage; the source's essentially never is from imaging alone. It is
assumed, labelled `assumed` in the record, and the note that quotes the mass
says so and says that the mass scales as `θ_E²` and with the distance ratio,
which is where its error lives.

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

## Feeding an astronomer's decisions back in

Everything here produces a recommendation; a person then decides. That
decision is the most expensive data this project will ever have — an expert's
judgement on a specific object — and it used to be discarded the moment the
screen closed. It is now recorded, fed back as a label, and used to measure
whether choosing *which* objects to show makes the labelling effort go
further.

**A model's verdict is not a label.** The pipeline's `Verdict` is what it
recommended; a `HumanVerdict` is what a person concluded, and they are
separate types deliberately. Feeding the first back in as training data is
self-training: the model's own errors return as truth and the next model is
more confident about them. A verdict without a named reviewer is refused for
that reason, and "I am not sure" is kept as an answer rather than converted
into a class.

The log is append-only, because a review is a historical fact — an astronomer
looked at this object on this date and concluded this. When two reviewers
disagree, that is surfaced rather than resolved by majority: an object experts
split on is either genuinely ambiguous or badly presented, and both are more
useful to know than a vote. The most valuable diagnostic that falls out is the
model's agreement with the reviewers, measured on real decisions rather than a
held-out split — and specifically the objects the model called at over 0.9
confidence and a reviewer overruled, which is where a calibration problem
shows up first.

### Choosing what to show: a negative result

Uncertainty sampling — show the astronomer whatever the model is least sure
about — is the textbook answer, and on this problem it does not work. Over six
repeats at budgets from 20 to 100 labels it lost to plain random selection at
three of the four budgets and won at one, with overlapping spreads throughout.

The class composition explains it. The decision boundary is crowded with faint
*stars*, which are the majority class and individually uninformative, so
uncertainty sampling spent 58 of its 100 labels on them against random's 42,
while galaxies fell from 32 to 22. It bought more of what there was already
plenty of.

Quotas per predicted class were the obvious fix and also failed, for a reason
worth keeping: the quota is on what the model *predicts*, and early in the
loop the model predicts the majority class for nearly everything, so the quota
rebalances nothing.

The default is therefore random selection — unbiased about the distribution
the model will actually meet, simpler, and no worse than anything tried
against it. The other strategies remain with their numbers attached, because
this is one problem at one range of budgets, and a harder one with a rarer
target class is exactly where uncertainty sampling should be tried again.

One caveat belongs on all of it: the loop is measured with the pool's true
labels standing in for the astronomer. That oracle is instant, always right
and always decisive, and a real reviewer is none of those, so these curves are
the best case for every strategy alike.

### Where the decision is made

The log described above had no front door: a verdict could only be added
from Python. `vetting` is that door -- a page served on localhost by the
standard library, no framework and no network, that shows one candidate at
a time and turns a keystroke into a :class:`HumanVerdict`.

What it shows is chosen for a decision that takes seconds: the cutout under
an asinh stretch about the stamp's own sky, the background-subtracted
cutout beside it, the pipeline's verdict and the reasons and caveats it
attached, the evidence numbers behind the rank, the source's measurements,
and -- when the catalog database is given -- every detection of that
position across epochs, drawn as a light curve. The stretch and the PNG
are written by hand so the page runs where the package runs, on NumPy
alone.

What it enforces is the boundary. A verdict without a reviewer's name is
refused by the server, not merely discouraged by the page, because an
anonymous decision cannot be told from the model's output and training on
it would be self-training. Verdicts are appended, never overwritten; a
reviewer who changes their mind adds a second record, and two reviewers
who disagree are surfaced as a count on the page rather than averaged.
Each verdict carries what the model had said, so the log can answer the
question that matters for calibration: how often did a person overrule a
model that was over 0.9 confident?

## Learning without labels

Labels are the scarce resource. A survey produces millions of cutouts a night
and a few thousand will ever be looked at by a person, so a representation
learned without labels and adapted with a handful of them is worth more than a
better classifier trained on the labels that exist.

The method is contrastive: two augmented views of one stamp are pulled
together in the embedding while views of different stamps are pushed apart. No
label enters anywhere — the `fit` method takes stamps and nothing else, so a
run cannot quietly have used them. What the network learns is whatever
survives the augmentations, which makes the augmentation list the whole design
rather than a detail.

The rule for including one: it must change the *observation* and leave the
*object* alone — what a telescope could have done differently on another night
to the same source. Rotations and reflections are exact sky symmetries;
translation, added noise, mild extra PSF blur and brightness scaling are all
things a different exposure would have produced. Blur has to stay mild, since
blurring a star far enough makes it a galaxy, which is teaching the network
something false.

Measured over three seeds per policy, the photometric augmentations carry most
of the value: dropping everything except rotation and reflection costs 14
points of balanced accuracy, and dropping the PSF blur alone costs 3.5.

**One prediction of mine was wrong and the measurement says so.** The standard
SimCLR recipe uses a random resized crop, which teaches the network that scale
does not matter — and in a survey cutout angular size is exactly what
separates an unresolved star from a resolved galaxy. That argument is clean,
and it did not survive contact with the data: adding resized crops left star
and galaxy recall unchanged (0.784 ± 0.016 against 0.790 ± 0.012) and slightly
*improved* overall accuracy. The default is still no resized crop, on the
physical argument rather than on a measured advantage, and the switch is kept
so the question can be reopened at a larger stamp size or crop range. Deleting
the experiment would have been easier and would have left the docstring's
confident claim standing.

What self-supervision buys is measured the same way transfer was: against
training from scratch on the same labels, because "the probe reached 80 %" is
only interesting if 80 % was not available without the pretraining.

## Explaining a score

An astronomer will not act on a number a black box produced, and is right not
to. Every model here that scores an object has a matching explanation: the
stamp classifier gets a saliency map, the boosted trees get Shapley values,
and the anomaly ranking gets retrieval — "this looks like these three, and
here is how far from them it is".

Producing any of those is easy. Knowing whether they are *true* is the work,
because a saliency map is an image and an image is convincing whether or not
it describes the model. So each explanation is checked against the model's own
behaviour:

* a saliency map by **deletion** — erase the pixels it calls important and the
  class score must fall further than erasing the same number of random ones;
* Shapley values by **additivity** — they must sum, with the base rate, to the
  model's actual output;
* retrieval by **purity** — the neighbours must share the query's class more
  often than chance, where chance is the probability two random objects match,
  not one over the class count.

Three things those checks turned up.

**The deletion test was measuring itself.** Erasing pixels by setting them to
a constant — the obvious choice, and the usual one — narrows the stamp's noise
distribution, and the classifier's asinh stretch computes its softening from
exactly that distribution. A map that ranks background pixels highly therefore
changes the *stretch* rather than removing information, and scores an
advantage it has not earned. On the same maps, a constant fill reported a mean
advantage of 0.109 with 32 of 40 stamps beating chance; filling with noise
drawn from the stamp's own background gave 0.044 and 25 of 40. More than half
the effect was the test. The noise-preserving fill is now the default.

**Grad-CAM is nearly useless on this architecture, and the measurement says so
plainly.** It computes correctly — with global average pooling into a single
linear head the gradient weights equal the head weights exactly, and that
identity is checked to 1 part in 10¹⁰. But under the corrected deletion test
it beat chance on 21 of 40 stamps with an advantage of 0.03, its correlation
with where the object's light actually is came out at −0.04, and it placed
0.15 of its mass on the central sixteenth of the stamp where a *uniform* map
would place 0.11. The cause is structural: a 48-pixel stamp leaves a 12 × 12
map, one to four cells of which cover a compact source, and global average
pooling means the decision genuinely draws on the whole frame.

**Occlusion works, so it is the default.** Covering the image a patch at a
time and measuring the drop gives an advantage of 0.23, beats chance on 37 of
40, correlates 0.30 with the light and puts 0.60 of its mass on the object.
Part of that gap is expected — occlusion optimises the quantity the deletion
test measures — but the correlation and the concentration are not what it
optimises and it wins those too. It costs one forward pass per patch, which is
the honest trade: Grad-CAM remains available for when speed matters more than
fidelity.

Shapley values are estimated by permutation sampling rather than computed
exactly, because exact enumeration is factorial in the feature count. An
estimate without an error bar is a number pretending to be a fact, so the
standard error is returned per feature and a run that has not converged says
so. On a model with two informative features among six carriers of nothing,
both informative ones land in the top two attributions for every object
tested, and the noise features come out two orders of magnitude smaller.

Retrieval is reported against the *typical* separation in the same field,
because a distance alone means nothing — the same number is close in one
embedding and remote in another. The learned embedding retrieves same-class
neighbours 86 % of the time against a 35 % chance rate; comparing raw pixels
manages 59 %, which is what says the embedding is contributing rather than the
pixels.

## Training on data from somewhere else

Every model in this package has been trained on simulated fields, where the
label is exact, the stamp is clean, and the instrument is the one the model
will be used on. Real training data is none of those things, and the
differences are not details.

**Labels are votes.** A crowd-sourced catalogue gives the fraction of
volunteers who chose each answer. A stamp 98 % of them agreed on is not the
same training example as one they split 51/49 over, and treating them alike
teaches the model the disagreement. The winning fraction becomes a per-sample
weight, and anything below 60 % agreement is dropped — not as a hard example
to learn from, but as one the labellers themselves did not settle.

**Stamps have holes.** Chip gaps, saturated columns and masked cosmic rays
arrive as NaNs, and a NaN reaching an optimiser turns every weight downstream
into a NaN on the first backward pass, with the failure surfacing far from its
cause. Bad pixels are therefore handled at the door: a few are filled with the
local median — not zero, because a zero-filled hole is a dark patch and a
network will learn dark patches as a feature of whichever class had the most
chip gaps — and a stamp more than a quarter unusable is refused.

**Units are arbitrary.** Counts, nanomaggies, calibrated flux, varying between
files. The per-stamp asinh stretch already removes this, which is worth
stating for what it implies: the input is scale-free, so what a domain gap
consists of is optics and noise, not units.

**The instrument is different**, and this is the one that costs accuracy.

### What a change of instrument costs

The useful question is not whether a model transports between telescopes — it
does not — but how many labelled examples from the new one it takes to
recover. That question has an answer, and measuring it takes three legs, of
which the third is the one usually left out:

1. train on the source instrument, test on it — the number people quote;
2. train on the source, test on the target — the gap;
3. fine-tune on *N* target examples, **and train from scratch on the same
   *N***, and compare.

Without the third leg, "fine-tuning reached 84 %" is unfalsifiable: those *N*
examples alone might have got there, and the pretraining contributed nothing.

Measured between two simulated instruments — a sharp Moffat PSF over a quiet
background, against a blurrier Gaussian one on a sky three times brighter —
the gap is 23 points of balanced accuracy, and fine-tuning the head on 25
target labels recovers about half of it. The full numbers, with their spread,
are in `docs/validation.md`.

Two findings there are worth carrying here because both contradicted a
reasonable guess.

**The frozen backbone is not the limiting factor**, even at 200 labels. The
obvious expectation is that head-only fine-tuning saturates and unfreezing the
network does better once there is enough data. Measured, unfreezing is *worse*
at every budget from 50 labels up — 0.65 against 0.84 at 100 — because a few
hundred examples cannot retrain a whole network without destroying the
features the source domain paid for.

**A single draw is not a measurement.** One draw of 25 target labels scored
0.837, which would have supported a claim that 25 labels recover 90 % of the
source score. Five draws of the same size gave 0.795 ± 0.059, spanning 0.726
to 0.866 — and three of the five do not reach that threshold. At small
budgets the spread is the finding, so the study repeats every budget and
reports the spread rather than the first number it saw.

**No real survey data was used.** Nothing outside the package registries was
reachable from the environment this was written in, so the loaders are
exercised against files written in the same formats rather than against
archive files, and the domain shift is measured between two simulated
instruments. What that buys is the *method* and the shape of the answer; what
it does not buy is a number for any particular survey.

## Reading what a survey actually delivers

A survey image is not a FITS file with a picture in it. It is a science plane,
a data-quality plane whose bits mean different things in every pipeline, and a
weight or variance plane that is the survey's own statement of its noise —
and a header whose gain keyword may be `GAIN`, `EGAIN`, `ARCONG` or absent,
whose pixels may already be in electrons, and whose saturation level is the
one thing that must be believed. Getting any of these wrong is silent: a gain
applied twice halves every Poisson error, an ignored weight plane leaves the
photometer guessing at noise the survey had already measured, a missed
saturation limit puts a flat-topped star through the PSF fitter.

The loader (`io.survey`) reads the planes by their `EXTNAME`, converts a weight
plane to σ = 1/√w and masks zero weight rather than calling it infinite noise,
masks every set DQ bit unless told which bits matter, reads the saturation
level and masks above it, and recognises `BUNIT = electron` so the gain becomes
one. Each of those is a choice, and each is written into the load report so
the analysis can say what it assumed. The preprocessor then *combines* the
survey's noise plane with its own estimate — the larger of the two, pixel by
pixel — rather than overwriting it, which is what it did before this was
tested.

Two conventions the world-coordinate reader had wrong are worth naming. A
header written as a `PC` matrix with `CDELT` — the form most modern pipelines
emit — was being read as if it had no rotation, and every world coordinate
was off by the field's rotation angle. And a frame past a few tens of
megapixels is memory-mapped, so a 4 GB mosaic is opened rather than loaded.

None of this was tested on an archive file: nothing outside the package
registries was reachable from the environment it was written in. It is tested
on files written in the same layout, which proves the conventions are handled
and does not prove that any particular survey's are.

## Checking against codes that have been checked

photutils and SEP are what the community measures with, and they have twenty
years of comparison behind them. A new photometry code that has only ever been
compared with its own simulator has one obvious blind spot: the simulator and
the code were written by the same hands. So `validation.benchmark` runs both
tools on the same pixels, with the same threshold and the same aperture, and
matches the three catalogs to each other and — when the field is simulated —
to the truth.

The numbers are in `validation.md`. Where both codes detect an object they
agree on its position to a few hundredths of a pixel and on its flux to a few
tenths of a percent, which is what it means for two implementations of the
same measurement to be correct. Where they disagree is on *which* objects to
report near the threshold, and the truth table is what settles that: most of
what this package finds and the others do not are faint real objects, and a
handful are noise.

What the benchmark found that no simulator comparison had was that the
photometry stage was slow — a second per source on a survey frame, because
every aperture was a full-frame array — and that is the reason
`photometry.aperture` now works on cut-outs. The measurement is the same to
the last bit; the time is a thousandth.

## A frame too large to process

A survey frame is sixteen thousand pixels on a side, and the stages here were
written for a field that fits in memory several times over — a filtered
copy, a segmentation map, a background model, a noise map. On a gigapixel
they do not. So the frame is cut into tiles that do, each is processed as a
field of its own, and the catalogs are merged (`engine.tiles`).

Every mistake a tiling can make was made here first and then addressed by
name:

- **A thin remainder tile.** A 384-pixel tiling of a 1024-pixel frame left a
  160-pixel strip whose background mesh and PSF star count were unlike every
  other tile's; fluxes measured in it were 6 % off. The planner now fixes the
  tile count and stretches the tiles evenly, so every tile is the same size.
- **Truncated fragments at tile edges.** An object cut by a tile boundary is
  detected in that tile as a fragment whose centroid is pulled several pixels
  toward the edge — farther than any matching radius, so a nearest-neighbour
  merge kept both copies. The tiles' *cores* — each tile's region minus half
  the overlap on every interior side — partition the frame exactly, and a
  source is kept only from the tile whose core contains it. The fragment is
  outside its tile's core and is dropped without being matched at all. The
  matching radius is left with the one case it can handle: an object sitting
  on a core boundary, whose two centroids fall on either side of it.
- **A PSF per tile.** The aperture correction is one over the PSF's enclosed
  energy, and a PSF fitted from the fifteen stars a tile happens to hold
  differs from its neighbour's by a percent, so corrected fluxes stepped at
  every tile boundary. One PSF is built from a central tile and shared, and
  the measured tile-to-tile spread of the correction fell from 1.1 % to
  0.4 %. What a shared PSF cannot represent is a PSF that varies across a
  mosaic; the spatially varying fit runs on whole frames only, and the
  per-tile mode is there for the case where each tile is its own detector.
- **A background per tile.** This one is kept on purpose. A 16k frame has sky
  structure no single fit follows, and the overlap is what lets the merge
  ignore the small step between neighbouring tiles' estimates.

Positions come back in the frame's own pixels, every source records which
tile measured it and how far from that tile's edge, and the whole thing is
measured against the whole-image catalog on a frame small enough to do both
(`validation.md`). Peak memory is the size of one tile, not the frame.

## A catalog that remembers

A CSV per image answers "what was in this image". A survey asks what is at
*this position* in every image ever taken of it, and how *this object* has
behaved over the year. Those need one store across fields and epochs,
indexed by sky position, with a notion of an object that persists between
detections. That is `catalog.database`: SQLite, because it is in the
standard library and the package promises to run on NumPy alone; three
tables, fields, detections and objects; and a HEALPix index on every row.

**HEALPix, in NumPy.** The sphere is cut into twelve base faces and each
into nside² equal-area pixels. In the *nested* numbering a pixel's index
interleaves the bits of its position inside the face, so the four children
of a pixel at one resolution are four consecutive integers at the next and a
coarse pixel is a contiguous range of fine ones. That is what a database
index wants: a cone on the sky becomes a few `BETWEEN` ranges. The
implementation (`catalog.healpix`) follows the reference algorithms and is
checked pixel for pixel against healpy where healpy is installed, and
against HEALPix's own properties -- equal areas, hierarchy, round trips --
where it is not.

**Object identity is positional, and says so.** On ingest each detection
with sky coordinates is linked to an existing object within the match radius
(1.5 arcseconds by default) or founds a new one. The association is
vectorised: candidate pairs come from a sorted-pixel join over the
detection's own 26-arcsecond pixel and the pixels of a ring of points around
it, and the exact separation is computed only for candidates. What this
cannot know is whether two things at the same position are the same thing.
In a crowded overlap of two fields, different objects fall within 1.5
arcseconds of each other and are linked; an object whose measured position
wanders more than that -- a fast mover, a poor astrometric solution --
becomes several. The moving-object stage links tracklets on its own terms;
this store keeps a history per position, and a history is what a light
curve is.

**What is kept.** Each field row carries the run's manifest and
reproducibility key, so every detection traces to the code and
configuration that made it. Each detection carries the flat export schema
with NaN stored as NULL. Each object carries its founding position, its
detection count, first and last epoch and the bands it was seen in. The
measured cost of all this is in `validation.md`: ingest rates, cone-search
and history times at half a million rows.

## Writing down what produced the result

A catalog without its provenance is a number without a unit. The question
that arrives six months later is never "what did the pipeline find" but "could
I get it again, and if not, what changed", and the answer has to come from
something written down at the time.

Every run therefore carries a manifest (`core.provenance`): a content hash of
the configuration and the configuration itself, the package version and git
revision with a flag if the tree was dirty, the versions of every dependency
that does arithmetic — NumPy, SciPy, Astropy, scikit-learn, PyTorch — the
random seeds, and a checksum of each input file. A reproducibility key
summarises the parts that decide the result, so two manifests can be compared
in one line, and `differences()` says in words why two runs might not agree.
The run's own catalog is digested too, rounded to a millionth of a pixel so a
last-bit difference in a sum does not register as a failure while anything
larger does. The test suite asserts the property this exists for: two runs
with the same key produce the same digest.

What a manifest cannot do is make a result from uncommitted code reproducible;
it can only say so, and it does.

## Where the decision is made

Every stage of this package ends in a recommendation; a person makes the
decision. `vetting` is where. It serves one local page from the standard
library: a candidate at a time, its cutout and background-subtracted cutout,
the pipeline's verdict, its reasons and caveats, the numbers behind them,
and -- when the catalog database holds earlier epochs -- the object's
history. A verdict is one key: R for real, B for bogus, U for unsure, S to
skip. Each is appended to the active-learning log with the model's label and
confidence beside it, under the reviewer's name. A verdict without a name is
refused by the server, because an unattributed decision cannot be told from
the model's own output and training on it is self-training. The page shows
its own disclaimer: nothing on it is a confirmed detection, and the verdict
is the reviewer's.

## Speaking the community's formats

Alert brokers exchange Avro, and a candidate worth reporting goes to the
Transient Name Server as a specific JSON document. `alerts` does both, with
one constraint chosen on purpose.

**Avro without a dependency.** Reading an alert should not need a compiled
library on a machine that has only NumPy, so the Avro binary encoding and
the object-container format are implemented in plain Python: zig-zag
varints, little-endian floats, length-prefixed bytes, records as fields in
order, unions as index then value, blocks with a sync marker, ``null`` or
``deflate``. The reader is schema-driven -- it decodes whatever schema the
file embeds -- so a real ZTF or Rubin file is read in full. When fastavro is
installed it is used instead, and the tests write with each and read with
the other.

**One vocabulary out, three in.** Alerts this package writes use ZTF's
names (``objectId``, ``candid``, ``candidate.jd``, ``magpsf``, ``rb``,
``prv_candidates``, gzip-compressed FITS cutouts) so the community's filters
and plots read them unchanged; it is a documented subset, and fields the
package cannot measure are absent rather than invented. Coming in, the
packet reader understands that vocabulary, ZTF's own, and Rubin's
``diaSource`` spelling (``midpointMjdTai``, ``psfFlux`` in nanojansky,
``band``), and gives one :class:`AlertPacket` for all three.

**A report drafted, never sent.** The TNS draft has the bulk-report layout
-- position with error, group, source and instrument ids, reporter,
discovery date, type, the discovery photometry, the last non-detection --
built from the packet's history. It requires a reporter's name, marks
itself ``_draft``, lists in ``_todo`` everything a person still has to fill
in or check (the TNS ids, a missing non-detection, an unvetted candidate),
and there is no HTTP client in the module at all. A claim to the community
is made by a person under their own credentials; the package writes the
form.

## What none of this can do

Nothing here confirms anything. A transient candidate needs an independent
epoch and, for a supernova, a classification spectrum. A lens candidate needs
colour information — lensed sources are typically bluer than their deflectors —
and ultimately spectroscopic redshifts for both. A morphological type from a
single band at survey depth is a useful prior, not a measurement. Physical
quantities derived from an assumed redshift are order-of-magnitude estimates,
and every assumption is listed in the report that quotes them.

A fitted mass model is not an exception to that. It converts an Einstein
radius into a mass under an assumed profile and an assumed source redshift; it
does not establish that the object is a lens. That still takes colours, a
second look, and spectroscopy.

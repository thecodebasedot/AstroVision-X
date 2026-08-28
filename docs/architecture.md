# Architecture

AstroVision-X is a linear pipeline of independent stages. Each stage reads the
typed records the previous one wrote and adds to them; none reaches back into
raw pixels except through the shared `AstroImage`. That is what makes any
stage individually replaceable, testable and skippable.

## The data model

Three types carry everything between stages.

**`AstroImage`** — pixels plus what is known about them: the header, an
optional WCS, a bad-pixel mask, a per-pixel uncertainty map and the background
model. `subtracted()` and `rms_map()` are the two accessors every downstream
stage uses, so a stage never needs to know whether the background was
subtracted in place or modelled separately.

**`Source`** — one detected object. Detection creates it with a position, a
bounding box and a segmentation label; photometry fills its `Photometry`;
morphology fills its `MorphologyMetrics`; classification sets `object_class`.
Nothing is recomputed: each stage adds the fields it owns.

**`FieldAnalysis`** — the whole result: the catalog, the transient candidates,
the anomaly records, the lens candidates, the light curves, the statistics, the
provenance and any warnings. This is what the reporters render.

## Stage order, and why it is fixed

```
preprocess → detect → photometry → [calibration] → [multiband]
           → segmentation → morphology → classification → [crossmatch]
           → embeddings → anomaly → lensing
           → [transient → timeseries] → clustering → statistics → assistant
```

Stages in brackets run only when configured or when the data supports them.
The order is not arbitrary; each dependency is real.

- **Detection needs preprocessing** because a detection threshold is only
  meaningful relative to a background model and its noise.
- **Photometry needs the segmentation map** to mask neighbouring sources out of
  each aperture, and to know which pixels belong to which object.
- **Morphology needs photometry** because the Sérsic fit is pinned by the
  measured half-light radius and the fit region is scaled to the object's own
  size. Without that, the index–radius degeneracy makes the fit run away.
- **Calibration needs photometry** because a zero point is fitted to
  instrumental magnitudes, and it comes *before* the multi-band pass so that
  the colours are differences of calibrated magnitudes rather than of
  detector counts.
- **Forced photometry needs the detection band's apertures**, which is the
  entire point: one aperture, defined once, applied at the same sky position
  in every band. Detecting independently in each band gives each one its own
  centroid and its own Kron radius, and the difference of two such magnitudes
  is not a colour.
- **Classification needs morphology** because a confident morphological type is
  itself evidence that a source is a galaxy; it needs the multi-band pass
  because the stellar locus is fitted from colours, and it runs its rules
  *twice* — once on morphology alone to seed the locus fit, then again with
  colour folded in. A locus seeded by a colour-informed answer would be
  confirming its own conclusion.
- **The known-object crossmatch comes after classification** and before the
  anomaly ranking, because what it changes is the priority of a *candidate*:
  an outlier that is already catalogued is not a discovery.
- **Spectroscopy is a separate entry point, not a pipeline stage.** The
  imaging pipeline runs over a field; a spectrum is one object on one slit,
  and the two share templates and numerics but not a control flow. Inside it
  the order is forced: no redshift without a wavelength solution, no line
  ratios without a redshift, and each step records where it stopped.
- **The mass model runs inside the lensing stage, after the arcs are found**,
  because it needs positions along real arcs rather than the candidate's
  summary numbers. It is also allowed to fail without removing the candidate:
  detection and measurement are separate claims and are kept separate in the
  record.
- **The lensing search needs classification** because it only examines objects
  massive and early-type enough to lens; searching every source for arcs
  produces mostly false positives.
- **The assistant needs everything**, because its job is to rank across
  stages and explain the ranking.

The two time-domain stages run only when a multi-epoch series is supplied.
The multi-band stage runs only when other filters are passed to `run(bands=…)`.
Calibration and crossmatch run only when a reference-catalog backend is
configured; the default is `none`, and a run without one records that
*nothing in the field has been shown to be previously unknown* rather than
letting silence read as a clean check.

## Failure is contained

A stage that raises is recorded as `status="failed"` with its message, added to
`analysis.warnings`, and the pipeline continues. A partial catalog with an
honest account of what is missing is far more useful than no catalog. The
provenance block lists every stage with its status and wall-clock time, so a
report always shows what actually ran.

## Optional dependencies

NumPy is the only hard requirement. Everything else is reached through
`core.backend`, which returns `None` rather than raising on a missing import.
Three patterns are used, deliberately:

1. **Transparent acceleration.** `convolve`, `label`, `median_filter` and
   friends use SciPy when it is present and a NumPy implementation otherwise.
   The results are identical; only the speed differs.
2. **Transparent substitution.** FITS reads through Astropy when available and
   through a self-contained parser otherwise. Both directions are tested
   against each other.
3. **Feature gating.** The deep detector, the U-Net and the CNN classifiers
   need PyTorch and say so clearly when it is absent. The classical paths that
   cover the same ground continue to work.

Anomaly detection sits between (1) and (3): the autoencoder uses a non-linear
PyTorch model when it can and a PCA-equivalent linear one when it cannot, and
both are genuinely useful.

## Configuration

One `AstroVisionConfig` dataclass tree describes a run completely. It can be
built in Python, loaded from JSON or YAML, adjusted by a named preset, and
overridden per-key from the command line. It is serialised into every report,
so any result can be reproduced from the report alone.

## Extending it

Each stage is a class with a `run()` method and its own config section. To add
a detection backend, register it:

```python
from astrovision.detect.detector import DETECTORS, BaseDetector

@DETECTORS.register("my_backend")
class MyDetector(BaseDetector):
    def detect(self, image):
        ...
        return catalog, segmentation
```

then set `detection.backend = "my_backend"`. The same registry pattern is used
wherever a component is meant to be swappable.

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
preprocess → detect → photometry → segmentation → morphology
           → classification → embeddings → anomaly → lensing
           → [transient → timeseries] → clustering → statistics → assistant
```

The order is not arbitrary; each dependency is real.

- **Detection needs preprocessing** because a detection threshold is only
  meaningful relative to a background model and its noise.
- **Photometry needs the segmentation map** to mask neighbouring sources out of
  each aperture, and to know which pixels belong to which object.
- **Morphology needs photometry** because the Sérsic fit is pinned by the
  measured half-light radius and the fit region is scaled to the object's own
  size. Without that, the index–radius degeneracy makes the fit run away.
- **Classification needs morphology** because a confident morphological type is
  itself evidence that a source is a galaxy.
- **The lensing search needs classification** because it only examines objects
  massive and early-type enough to lens; searching every source for arcs
  produces mostly false positives.
- **The assistant needs everything**, because its job is to rank across
  stages and explain the ranking.

The two time-domain stages run only when a multi-epoch series is supplied.
Everything else runs on a single image.

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

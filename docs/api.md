# API reference

## Top level

```python
import astrovision

astrovision.analyze("image.fits")                  # FieldAnalysis
astrovision.analyze_series(["e1.fits", "e2.fits"]) # FieldAnalysis
astrovision.quick_field((512, 512))                # (AstroImage, [TruthObject])
astrovision.describe_capabilities()                # which backends are present
```

## The pipeline

```python
from astrovision import Pipeline, AstroVisionConfig

config = AstroVisionConfig().with_preset("deep_field")
config.detection.threshold_sigma = 2.5

pipeline = Pipeline(config)
analysis = pipeline.run(image, redshift=0.3)        # one field
analysis = pipeline.run_series(series)              # multi-epoch
```

`Pipeline.run(image, series=None, redshift=None, preprocess=True)` — pass
`preprocess=False` for an image that is already calibrated and
background-subtracted.

## Results

```python
analysis.catalog            # SourceCatalog
analysis.transients         # list[TransientCandidate]
analysis.anomalies          # list[AnomalyRecord]
analysis.lenses             # list[LensCandidate]
analysis.light_curves       # dict[int, LightCurve]
analysis.statistics         # field, stellar, photometry, physical, narrative
analysis.provenance         # config, stages, timings, capabilities, version
analysis.warnings           # anything that went wrong
analysis.summary()          # counts, at a glance
```

### Querying the catalog

```python
from astrovision import ObjectClass

catalog.of_class(ObjectClass.GALAXY)
catalog.filter(lambda s: s.photometry.snr > 20)
catalog.sorted_by(lambda s: s.anomaly_score)
catalog.brightest(10)
catalog.match(x=120.4, y=88.1, radius=3.0)
catalog.class_counts()
catalog.positions()                    # (N, 2) array
catalog.embeddings()                   # (N, D) array or None
```

### One source

```python
source.x, source.y, source.ra, source.dec
source.object_class, source.class_confidence, source.class_scores
source.photometry.flux, .magnitude, .snr, .kron_radius, .petrosian_radius
source.morphology.sersic_index, .concentration, .asymmetry, .gini, .m20
source.morphology.label, .arm_count, .bar_strength
source.anomaly_score, source.lens_score, source.variability_score
source.flags                           # edge, blended, saturated, variable, ...
source.meta                            # r50, apertures, sersic, spiral, physical
source.cutout(image.data, pad=4)
```

## Individual stages

Every stage runs standalone, in this order:

```python
from astrovision.preprocess import Preprocessor
from astrovision.detect import Detector
from astrovision.photometry import Photometer
from astrovision.segment import Segmenter
from astrovision.morphology import MorphologyAnalyzer
from astrovision.classify import Classifier
from astrovision.anomaly import AnomalyEngine
from astrovision.lensing import LensSearch
from astrovision.transient import TransientDetector
from astrovision.timeseries import LightCurveAnalyzer

clean = Preprocessor().run(image)
catalog, segmentation = Detector().detect(clean)
Photometer().run(clean, catalog, segmentation)
Segmenter().run(clean, catalog, segmentation)
MorphologyAnalyzer().run(clean, catalog, segmentation)
Classifier().run(clean, catalog)

anomalies = AnomalyEngine().run(catalog)
lenses = LensSearch().run(clean, catalog)
transients = TransientDetector().run(series, catalog)
curves = LightCurveAnalyzer().run(series, catalog)
```

## Several filters

```python
from astrovision.photometry import forced_photometry, measure_colours
from astrovision.classify import fit_stellar_locus, annotate_catalog

# `bands` maps filter name to a preprocessed AstroImage of the same sky.
report = forced_photometry(bands, catalog, detection_band="r",
                           aperture_arcsec=1.6, segmentation=segmentation)
measure_colours(catalog, [("g", "r"), ("r", "i")], min_snr=5.0)

source = catalog[0]
source.colour("g", "r")            # magnitudes, NaN if either band is missing
source.bands["g"].magnitude        # the per-band measurement
source.meta["colours"]["g-r"]      # only present when both bands cleared min_snr
source.meta["colour_limits"]       # one-sided limits where one band did not

locus = annotate_catalog(catalog)  # fits and applies the stellar locus
locus.separation                   # ROC area of the colour test on this field
locus.information_weight           # 0 when colour carries no information here
```

The pipeline does all of this when extra bands are passed to `run`:

```python
analysis = Pipeline(config).run(bands["r"], bands=bands, preprocess=False)
```

## Reference catalogs and calibration

```python
from astrovision.io.external import build_service, crossmatch_catalog
from astrovision.calibration import solve_plate, solve_zero_point
from astrovision.calibration.astrometry import apply_solution
from astrovision.calibration.photometry import apply_zero_point

service = build_service("vizier", cache_dir=".cache")   # or "local", path=...
report = crossmatch_catalog(catalog, service, radius_arcsec=2.0)
report.conclusive                  # False when nothing was actually checked

for source in catalog:
    if "known" in source.flags:
        print(source.meta["known_object"]["described_type"])

reference = service.query(ra, dec, radius_arcsec)
solution = solve_plate(catalog, reference, image.wcs, radius_arcsec=5.0)
if solution.succeeded:
    image.wcs = solution.wcs
    apply_solution(catalog, solution)
    print(f"{solution.rms_arcsec:.3f} arcsec from {solution.n_matched} stars")

zero_point = solve_zero_point(catalog, reference, band="r",
                              colour_pair=("g", "r"))
apply_zero_point(catalog, zero_point, "r")
```

## Uncertainty

```python
from astrovision.morphology import bootstrap_morphology
from astrovision.ml import fit_calibrator, calibration_report

errors = bootstrap_morphology(cutout, noise, centre=(x, y), n_samples=24)
errors.error("gini")                    # standard deviation over realisations
errors.bias("asymmetry", measured)      # how far noise pushes it

fit = source.meta["sersic"]
fit["errors"]["n"]                      # one-sigma marginal error
fit["worst_correlation"]                # usually n against r_eff, near 1

calibrator = fit_calibrator(scores, labels)     # isotonic or Platt, by data size
calibration_report(calibrator.transform(scores), labels)["usable_as_probability"]
```

Enable the bootstrap in the pipeline with `config.morphology.uncertainty = True`;
it is off by default because it costs `bootstrap_samples` times the shape
measurement.

## Reports

```python
from astrovision.report import generate_reports, render_text, render_json, render_html

generate_reports(analysis, "results/", formats=("text", "json", "html"),
                 title="Field 42", observer="A. Astronomer", image=image)

print(render_text(analysis))
payload = json.loads(render_json(analysis))
open("report.html", "w").write(render_html(analysis, image=image))
```

## Input and output

```python
from astrovision import AstroImage, ImageSeries, read_catalog, write_catalog
from astrovision.io import crossmatch

image = AstroImage.from_fits("field.fits")     # or .load() for npy/png/jpg
image.describe(); image.stats(); image.cutout(x, y, 64)
image.write("out.fits")

series = ImageSeries.from_paths(sorted(paths))
series.times, series.bands(), series.check_alignment(), series.stack("median")

write_catalog(catalog, "catalog.csv")          # csv, json, fits
catalog = read_catalog("catalog.csv")
matches = crossmatch(catalog_a, catalog_b, radius=2.0, use_world=True)
```

## Simulation

```python
from astrovision import SkySimulator, SkyConfig

sim = SkySimulator(SkyConfig(shape=(512, 512), n_stars=200, n_galaxies=40,
                             n_lenses=1, seeing_fwhm=3.2, seed=42))
image, truth = sim.generate()
series, static_truth, transients = sim.generate_series(
    n_epochs=6, cadence=2.0, n_transients=3)
```

## Training the deep models

```python
from astrovision.detect import DeepDetector
from astrovision.ml import StampClassifier
from astrovision.segment import UNetSegmenter
from astrovision.timeseries import SequenceClassifier

detector = DeepDetector(width=32).build()
detector.fit(images, annotations, epochs=40)
detector.save("detector.pt")

classifier = StampClassifier(backbone="cnn").build()
classifier.fit(stamps, labels, epochs=40)
classifier.annotate(catalog, image)
```

Point the configuration at the saved weights to use them:

```python
config.detection.backend = "dnn"
config.detection.model_path = "detector.pt"
config.classification.backend = "hybrid"
config.classification.model_path = "classifier.pt"
```

## Command line

```
astrovision analyze  IMAGE...   [-c CONFIG] [-p PRESET] [--set K=V] [-o DIR]
                                [--report text,json,html] [--threshold N] [-z Z]
astrovision series   IMAGE...   [same options] [--top N]
astrovision simulate            [--out PATH] [--size N] [--stars N] [--galaxies N]
                                [--epochs N] [--transients N] [--seed N]
astrovision inspect  IMAGE...   [--header]
astrovision config              [--out PATH] [--show] [--list-presets]
astrovision info
```

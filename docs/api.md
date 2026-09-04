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

## Lens mass models

```python
from astrovision.lensing import (LensModel, arc_sample_points, einstein_mass,
                                 fit_lens_model, ray_trace)

model = LensModel(x0=64.0, y0=64.0, theta_e=12.0, axis_ratio=0.7,
                  position_angle=35.0, shear1=0.03, shear2=-0.02)
model.deflection(x, y)                  # alpha, in pixels
model.source_plane(x, y)                # beta = theta - alpha
model.magnification(x, y)               # signed; diverges on the critical curve

fit = fit_lens_model(points, centre=(64.0, 64.0), theta_e_guess=12.0)
fit.succeeded, fit.reason               # a refusal explains itself
fit.model.axis_ratio, fit.model.shear_magnitude
fit.theta_e_error                       # bootstrap over the arc positions
fit.image_rms                           # residual in pixels
fit.flags                               # e.g. shear_fixed_to_zero

einstein_mass(1.2, z_lens=0.4, z_source=2.0)["mass_solar"]
```

Candidates from `LensSearch` carry the same results: `candidate.model`,
`candidate.model_theta_e_arcsec`, `candidate.model_axis_ratio` and
`candidate.mass`. A candidate whose arcs give fewer constraints than
parameters is still a candidate — `candidate.model["model"]` is `None` and a
note says why. Turn the fit off with `config.lensing.fit_model = False`.

The simulator can ray-trace a source through a model, which is how the fit is
tested against arcs it did not draw:

```python
arcs = ray_trace(shape, model, source_x, source_y, source_radius,
                 source_flux=1.0e4)
```

## Spectroscopy

```python
from astrovision.spectra import (analyse_frame, analyse_spectrum, classify_bpt,
                                 classify_supernova, extract_spectrum, fit_lines,
                                 fit_wavelength_solution, measure_redshift)

# A long-slit frame, an arc exposure and a line list
analysis = analyse_frame(image, variance, arc=arc_image, line_list=arc_lines,
                         sky_lines=sky_lines, resolution=5.0)
analysis.summary()                      # one readable line
analysis.stopped_at                     # "" when it ran to the end
analysis.redshift.z, analysis.redshift.reliable, analysis.redshift.r_statistic
analysis.bpt.classification             # star-forming / composite / Seyfert / LINER
analysis.lines["H alpha"].flux, analysis.lines["H alpha"].detected

# Or the pieces
spectrum, trace = extract_spectrum(image, variance, method="optimal")
solution = fit_wavelength_solution(arc_1d, arc_lines, order=3)
solution.succeeded, solution.rms, solution.reason
result = measure_redshift(calibrated)
lines = fit_lines(calibrated, result.z, resolution=5.0)
match = classify_supernova(calibrated, redshift=0.03)
match.sn_type, match.confident, match.caveat
```

A step that cannot be done is not done: without an arc the spectrum keeps a
column axis, and without a reliable redshift no lines are fitted — each would
otherwise produce numbers with nothing behind them. `analysis.stopped_at` says
which. Supernova typing is the exception and runs regardless, because a
supernova is not a galaxy and the galaxy correlation failing on one is the
expected outcome; pass the host `redshift` when it is known.

Simulated frames come from `astrovision.simulate.spectrograph`:

```python
from astrovision.simulate.spectrograph import (ARC_LINES, SKY_LINES,
                                               SpectrographConfig,
                                               SpectrographSimulator)
from astrovision.spectra import galaxy_spectrum, supernova_spectrum

simulator = SpectrographSimulator(SpectrographConfig(seed=1))
frame = simulator.object_frame(galaxy_spectrum(3.0, emission=1.2), redshift=0.12)
arc = simulator.arc_frame()
quick = simulator.extracted(supernova_spectrum("Ia", 0.0), redshift=0.02, snr=20)
```

## Human verdicts and active learning

```python
from astrovision.ml import (HumanVerdict, VerdictLog, compare_strategies,
                            review_queue, select_for_review, verdicts_to_labels)

queue = review_queue(catalog, probabilities, classes, n=20)   # random by default
# each entry carries model_label, model_confidence, runner_up, uncertainty

log = VerdictLog()
log.add(HumanVerdict(source_id=41, label="galaxy", reviewer="a.astronomer",
                     model_label="star", model_confidence=0.94,
                     note="faint disc visible"))
log.save("verdicts.json")

log.agreement_with_model()["confidently_wrong"]   # calibration problems, first
log.disagreements()                               # objects experts split on
training = verdicts_to_labels(log, dataset)       # confident verdicts only

compare_strategies(pool, test, classes,
                   strategies=("random", "uncertainty", "balanced"))
```

A verdict without a named reviewer is refused: an unattributed decision cannot
be told apart from the model's own output, and training on that is
self-training. `select_for_review` defaults to `random` because that is what
measured best — uncertainty sampling lost at three of four budgets and spent
its labels on the majority class. See `docs/validation.md`.

## Learning from unlabelled cutouts

```python
from astrovision.ml import (AugmentationPolicy, ContrastiveEncoder,
                            anomaly_ranking_quality, label_efficiency,
                            linear_probe)

encoder = ContrastiveEncoder(cutout=48, width=16)
encoder.fit(unlabelled_stamps, epochs=60)     # stamps only -- no labels taken
embeddings = encoder.embed(stamps)
classifier = encoder.to_classifier(classes)   # fresh head on learned features

# What the representation contains, as opposed to what it could be trained to do
linear_probe(encoder.embed(train.stamps), train_labels,
             encoder.embed(test.stamps), test_labels)["balanced_accuracy"]

# What the unlabelled data was worth, against the same labels from scratch
label_efficiency(encoder, labelled, test, budgets=(10, 25, 50, 100))

# Whether the embedding separates known oddities at all
anomaly_ranking_quality(embeddings, is_anomalous)["auc"]

AugmentationPolicy(resized_crop=True)          # off by default; see the docs
```

`fit` deliberately accepts no labels, so a run claiming to be unsupervised
cannot have used any. The augmentation policy is the design: keeping only
rotations and reflections costs 14 points of balanced accuracy against the
default set.

## Explaining a score

```python
from astrovision.ml import (deletion_curve, explain_catalog, explain_prediction,
                            explain_stamp, retrieval_purity, retrieve_similar)

# Which pixels the class score depended on
saliency = explain_stamp(classifier, stamp)          # occlusion, by default
saliency.heatmap, saliency.predicted_class, saliency.native_shape
explain_stamp(classifier, stamp, method="grad-cam")  # faster, measurably worse

# Is the map describing the model, or just looking convincing?
check = deletion_curve(classifier, stamp, saliency.heatmap)
check["advantage"], check["beats_chance"]            # positive means it earned it

# Which measured features moved this prediction
attribution = explain_prediction(gbdt, x, background=training_X,
                                 feature_names=names, n_samples=200)
attribution.explain()                                # a readable sentence
attribution.top(3), attribution.errors, attribution.converged
attribution.additivity_error()                       # must be small

# What an unusual object resembles
found = retrieve_similar(embeddings, index, n=3, labels=labels)
found.explain()                                      # "nearest ... 3.4x typical"
retrieval_purity(embeddings, labels)["lift"]         # > 1 means better than chance
explain_catalog(catalog, top=10)                     # attaches to the oddest sources
```

`explain_stamp` defaults to occlusion because that is what measured better:
on this classifier Grad-CAM beat chance on 21 of 40 stamps against occlusion's
37, and put no more of its mass on the object than a uniform map would. The
`deletion_curve` fill defaults to background noise rather than a constant —
a constant narrows the stamp's noise distribution, which the asinh stretch
renormalises against, and flatters every map. Both numbers are in
`docs/validation.md`.

## Training data and transfer

```python
from astrovision.ml import (class_balance_report, domain_study, evaluate,
                            fine_tune, freeze_backbone, load_fits_cutouts,
                            read_label_table, split_dataset, stamps_from_fields)

# Real survey cutouts: one FITS file per object, plus a table of labels
labels = read_label_table("labels.csv", id_column="objid",
                          vote_columns={"galaxy": "p_spiral", "star": "p_star"})
dataset = load_fits_cutouts("cutouts/", labels)
dataset.report()                     # what loaded, what was dropped and why
class_balance_report(dataset)        # the majority-class baseline to beat

# Or from simulated fields, one instrument per config factory
train = stamps_from_fields(lambda seed: SkyConfig(seed=seed, seeing_fwhm=3.0),
                           range(400, 420))
train, validation, test = split_dataset(train, (0.7, 0.15, 0.15), by_group="seed")

# Adapt a trained model to a new instrument
freeze_backbone(classifier)          # returns the parameter counts, so check them
result = fine_tune(classifier, target_train, target_validation, epochs=60)
evaluate(classifier, target_test)    # accuracy, per-class recall, confusion

study = domain_study(train_source_model, source_test, target_pool, target_test,
                     label_budgets=(12, 25, 50, 100), repeats=3)
study.summary()                      # gap, and the labels needed to close it
```

`split_dataset(..., by_group="seed")` keeps stamps from one field together:
they share its noise, PSF and background, so splitting them across train and
test measures memorisation. `domain_study` draws each budget `repeats` times
because a single draw of 25 labels varied by 0.14 in balanced accuracy here,
and it always trains from scratch on the same labels for comparison — without
that, a fine-tuning score says nothing about whether the pretraining helped.

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

## Survey products

```python
from astrovision.io import load_survey_image

image, report = load_survey_image("frame.fits.fz")     # SCI / MASK / WEIGHT (or VAR) planes
image.mask, image.uncertainty                          # from the DQ bits and 1/sqrt(weight)
report.gain, report.gain_source                        # "header", "assumed" or "pixels in electrons"
report.n_masked, report.n_saturated, report.notes      # every assumption the loader made
load_survey_image("frame.fits", mask_bits=1 | 4 | 16)  # honour only these DQ bits
```

The planes are found by their `EXTNAME` (`SCI`, `IMAGE`; `MASK`, `DQ`,
`FLAGS`; `WEIGHT`, `WHT`, `IVAR`; `VAR`, `VARIANCE`; `ERR`, `SIGMA`), the
science header is searched for gain, read noise, saturation, zero point,
exposure and filter under every common spelling, and pixels already in
electrons are recognised from `BUNIT` so the gain is not applied twice. A
variance or weight plane becomes `image.uncertainty` and is combined with the
preprocessor's own estimate rather than replaced by it; zero-weight pixels are
masked. Frames past sixteen million pixels are memory-mapped.

## Frames too large for memory

```python
from astrovision.engine.tiles import process_tiled, standard_stage, plan_tiles

result = process_tiled(image, standard_stage(config), tile=2048, overlap=128)
result.catalog                          # frame coordinates; each source knows its tile
result.n_duplicates_removed, result.peak_tile_pixels, result.per_tile

standard_stage(config, psf="shared")    # one PSF for every tile (default)
standard_stage(config, psf="per-tile")  # each tile fits its own, if it has the stars
for tile, sub in iter_tiles(image, tile=2048, overlap=128): ...
```

## Checking against other codes

```python
from astrovision.validation import benchmark_field, available_tools

available_tools()                                     # {"photutils": True, "sep": True}
for result in benchmark_field(clean, catalog, truth=truth, aperture_radius=5.0):
    print(result.summary())
    result.against_truth                              # recall, spurious, flux ratio per code
```

Needs `pip install -e ".[benchmark]"`. The comparison is on the same pixels,
the same threshold and the same aperture; the measured agreement is in
[`validation.md`](validation.md).

## A catalog across fields and epochs

```python
from astrovision.catalog import CatalogDB, ingest_analysis

with CatalogDB("survey.sqlite") as db:                 # SQLite: stdlib, NumPy-only
    report = ingest_analysis(db, analysis, image)      # or db.ingest(catalog, name=, band=, mjd=)
    report.n_matched, report.n_new_objects             # linked to known objects / founded new ones

    db.cone_search(150.1, 2.2, radius_arcsec=30)       # every detection, nearest first
    db.cone_search(150.1, 2.2, 30, table="objects")    # distinct sky objects instead
    db.history(object_id)                              # its detections across fields, in time order
    mjd, flux, err = db.light_curve(object_id, band="r")
    db.objects_with_history(min_detections=3)          # most-seen objects first
    db.field_catalog(field_id)                         # back as a SourceCatalog
    db.fields(); db.counts()
```

Every row carries a nested HEALPix index (`astrovision.catalog.healpix`,
pure NumPy, checked against healpy). Ingest links each detection to an
existing object within the match radius (1.5 arcseconds by default) or
founds a new one; a cone search is a handful of index ranges. The command
line does the same: `astrovision analyze image.fits --db survey.sqlite`,
then `astrovision db survey.sqlite cone RA DEC RADIUS`,
`astrovision db survey.sqlite history OBJECT`, `astrovision db survey.sqlite info`.

## Reproducing a run

```python
from astrovision.core.provenance import build_manifest, Manifest, same_result

manifest = build_manifest(config, inputs=["frame.fits"], seeds={"random_state": 42})
manifest.save("results/manifest.json")
manifest.reproducibility_key()          # hash of everything that decides the result
Manifest.load(path).differences(other)  # ["numpy 1.26.4 vs 2.0.1", "random seeds differ"]
same_result(catalog_a, catalog_b)       # identical measurements, to 1e-6
analysis.provenance["manifest"]         # the pipeline attaches one to every run
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

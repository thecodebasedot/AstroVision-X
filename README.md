# AstroVision-X

**Advanced Computer Vision & Machine Learning Framework for Astronomical and
Astrophysical Analysis**

AstroVision-X takes telescope imagery and produces a scientific report. Sources
are detected and deblended, measured photometrically and morphologically,
classified, searched for novelty and for gravitational lensing and — given
several epochs — differenced for transients and characterised in the time
domain. A research assistant then ranks what a human should look at first and
says why.

> **The platform reports candidates, not discoveries.** A transient candidate
> needs an independent epoch and, for a supernova, a spectrum. A lens candidate
> needs colours and redshifts. An anomaly is an object unlike the rest of *this
> field*, which is not the same as an object unlike anything known. Every
> finding is written to make that boundary visible.

---

## Quick start

```bash
pip install -e ".[all]"          # NumPy is the only hard dependency
astrovision info                 # show which optional backends are present
```

```bash
# Generate a synthetic field with a ground-truth table, then analyse it
astrovision simulate --out field.fits --size 512 --galaxies 40 --lenses 1
astrovision analyze field.fits --report text,html -o results/

# Search a multi-epoch series for transients
astrovision simulate --out night.fits --epochs 6 --transients 3
astrovision series night_epoch*.fits -o results/
```

```python
from astrovision import Pipeline, quick_field

image, truth = quick_field((512, 512))
analysis = Pipeline().run(image)

print(analysis.summary())
print(analysis.statistics["narrative"]["summary"])
for line in analysis.statistics["narrative"]["priority_text"][:3]:
    print(line)
```

---

## What it does

```
                      TELESCOPE / SURVEY DATA
                                │
                ┌───────────────┴───────────────┐
                │                               │
           2-D IMAGES                    TIME SERIES
                │                               │
                ▼                               ▼
        ┌───────────────┐              ┌────────────────┐
        │ PREPROCESSING │              │  REGISTRATION  │
        │ background    │              │  PSF matching  │
        │ cosmic rays   │              │  flux scaling  │
        │ PSF model     │              └───────┬────────┘
        └───────┬───────┘                      │
                ▼                              ▼
        ┌───────────────┐              ┌────────────────┐
        │   DETECTION   │              │  DIFFERENCE    │
        │ matched filter│              │    IMAGING     │
        │ deblending    │              └───────┬────────┘
        │ deep detector │                      │
        └───────┬───────┘                      ▼
                │                      ┌────────────────┐
                ▼                      │  REAL / BOGUS  │
        ┌───────────────┐              │    VETTING     │
        │ SEGMENTATION  │              └───────┬────────┘
        │ watershed     │                      │
        │ U-Net         │                      ▼
        │ galaxy parts  │              ┌────────────────┐
        └───────┬───────┘              │   TRANSIENT    │
                ▼                      │ CHARACTERISATION│
        ┌───────────────┐              │  light curves  │
        │  PHOTOMETRY   │              │  periodograms  │
        │ apertures     │              └───────┬────────┘
        │ Kron/Petrosian│                      │
        └───────┬───────┘                      │
                ▼                              │
        ┌───────────────┐                      │
        │  MORPHOLOGY   │                      │
        │ CAS, Gini/M20 │                      │
        │ Sérsic fit    │                      │
        │ arms and bars │                      │
        └───────┬───────┘                      │
                ▼                              │
        ┌───────────────┐                      │
        │CLASSIFICATION │                      │
        │ star / galaxy │                      │
        │ CNN, ViT      │                      │
        └───────┬───────┘                      │
                │                              │
      ┌─────────┼──────────┐                   │
      ▼         ▼          ▼                   │
  ┌───────┐ ┌───────┐ ┌────────┐               │
  │ANOMALY│ │LENSING│ │ASTRO-  │               │
  │isol.  │ │ arcs  │ │PHYSICS │               │
  │forest │ │ rings │ │ counts │               │
  │autoenc│ │ θ_E   │ │ cosmo  │               │
  └───┬───┘ └───┬───┘ └───┬────┘               │
      └─────────┼─────────┴────────────────────┘
                ▼
        ┌────────────────────┐
        │  RESEARCH ENGINE   │
        │  priority ranking  │
        │  narrative + why   │
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ SCIENTIFIC REPORT  │
        │  text / JSON / HTML│
        │  + source catalog  │
        └────────────────────┘
```

### Computer vision

| Task | Method |
| --- | --- |
| Source detection | Matched-filter thresholding on a 2-D background mesh |
| Deblending | Multi-threshold tree with flux-contrast pruning |
| Deep detection | Anchor-free CenterNet-style detector (PyTorch, optional) |
| Segmentation | Marker-controlled watershed; U-Net semantic labels |
| Galaxy decomposition | Nucleus / bulge / disc / outskirts from the curve of growth |
| Stamp classification | Residual CNN and a compact Vision Transformer |

### Measurement

| Quantity | Method |
| --- | --- |
| Flux | Sub-pixel apertures with a PSF aperture correction |
| Adaptive aperture | Kron and Petrosian radii from the curve of growth |
| Concentration | `C = 5 log₁₀(r₈₀/r₂₀)` |
| Asymmetry, smoothness | Conselice CAS, with the rotation centre minimised |
| Gini, M₂₀ | Lotz statistics over a Petrosian-defined footprint |
| Sérsic index | PSF-convolved 2-D fit, radius pinned by the measured `r₅₀` |
| Spiral arms | Fourier modes in log-polar space, confirmed by phase winding |
| Bars | The same `m = 2` mode, distinguished by *constant* phase |

### Discovery

| Search | Method |
| --- | --- |
| Transients | Hold-one-out templates, PSF matching, veto-style real/bogus vetting |
| Variability | Reduced χ², Stetson J, von Neumann η, Lomb–Scargle periods |
| Novelty | Isolation forest + autoencoder + k-NN isolation, rank-combined |
| Strong lensing | Tangential arcs at a shared radius; radial scan for full rings |
| Populations | Number counts, completeness turnover, Landy–Szalay clustering |

---

## Measured performance

Every number below is measured against the simulator's ground truth by the
test suite, on fields with realistic Poisson and read noise, cosmic rays and
bad columns. They are *not* claims about real survey data.

| Capability | Result |
| --- | --- |
| Detection recall (S/N > 10) | 93 % |
| Spurious detection rate | 1–3 % at 3.5 σ |
| Astrometric precision | 0.15 px median |
| Photometric accuracy (isolated stars) | within 3 %, 3 % scatter |
| PSF FWHM recovery | 6 % median error |
| Sérsic index recovery | 11 % median error |
| Star/galaxy separation | 90 % (100 % for galaxies at S/N > 10) |
| Galaxy morphology (5 classes) | 59 % exact, 78 % at family level |
| Transient recall | 12/14, with 2 spurious over five fields |
| Strong-lens recall | 8/14, with 3 false positives over five fields |
| CNN stamp classification | 85 % on a 266-stamp training set |
| LSTM light-curve classification | 92 % over six variability classes |

Known limits, stated plainly: nebula and star-cluster classification is weak
(they overlap galaxies in every measured statistic); the PSF is unreliable in
fields where galaxies outnumber stars several to one, and the pipeline warns
when that happens; and single-band lens searching cannot use the colour
information real searches rely on.

---

## Gallery

Every image below is produced by `examples/04_make_figures.py`, which runs the
real pipeline on a simulated field and plots what each stage returned. Nothing
is drawn by hand.

| | |
| --- | --- |
| ![Detections](figures/01_field_detections.png) | **Detection and classification.** The field before and after background subtraction, with every detection circled and coloured by class. |
| ![Stages](figures/02_pipeline_stages.png) | **Preprocessing.** Raw frame, the fitted background model, the subtracted image, and the deblended segmentation. |
| ![Morphology](figures/03_galaxy_morphology.png) | **Galaxy morphology.** Injected type against measured type, with Sérsic *n*, concentration, asymmetry, Gini/M20 and arm count. |
| ![Transients](figures/04_transient_discovery.png) | **Transient search.** Template, new epoch and difference, held on a single intensity scale so the residual is comparable to the source it came from. |
| ![Light curves](figures/05_light_curves.png) | **Light curves.** Photometry recovered from the epoch stack, against the injected curve. |
| ![Anomalies](figures/06_anomalies.png) | **Novelty search.** The highest-ranked outliers, each with the written reason it was flagged. |
| ![Lens](figures/07_lens_candidate.png) | **Lens candidate.** Deflector, the same cutout with smooth galaxy light removed, and the tangential arcs with a fitted Einstein radius. |
| ![Report](figures/08_html_report.png) | **The written report.** What the research assistant produces for a field, including the ranked follow-up list. |

---

## Installation

```bash
pip install -e .                 # core: NumPy only
pip install -e ".[science]"      # + SciPy, Astropy
pip install -e ".[ml]"           # + scikit-learn
pip install -e ".[deep]"         # + PyTorch
pip install -e ".[all]"          # everything except PyTorch
```

Optional dependencies enable *features*, never whole subsystems. Without
Astropy, FITS still reads and writes through a self-contained parser; without
SciPy, labelling, filtering and fitting fall back to NumPy; without PyTorch,
the deep detector, U-Net and CNN classifiers raise a clear error and the
classical paths carry on. Run `astrovision info` to see what is active.

---

## Configuration

```bash
astrovision analyze field.fits --preset deep_field \
    --set detection.threshold_sigma=2.5 \
    --set anomaly.contamination=0.01
```

Presets: `deep_field`, `wide_survey`, `transient_search`, `lens_search`,
`quicklook`. The full configuration is written into every report, so a result
can always be reproduced.

```python
from astrovision import AstroVisionConfig, Pipeline

config = AstroVisionConfig().with_preset("transient_search")
config.detection.threshold_sigma = 4.0
config.transient.real_bogus_threshold = 0.8      # purity over completeness
analysis = Pipeline(config).run_series(series)
```

---

## Working with real data

```python
from astrovision import AstroImage, ImageSeries, Pipeline

image = AstroImage.from_fits("survey_r.fits")          # WCS, band, MJD from the header
analysis = Pipeline().run(image, redshift=0.34)

series = ImageSeries.from_paths(sorted(glob.glob("epoch_*.fits")))
analysis = Pipeline().run_series(series)
```

The pipeline reads the zero point from `MAGZP`, the gain from `GAIN` and the
saturation level from `SATURATE` when present, and falls back to configured
values otherwise. Every assumption it had to make is listed in the report.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the stages fit together
- [`docs/methods.md`](docs/methods.md) — the science behind each measurement
- [`docs/validation.md`](docs/validation.md) — how each number above was measured
- [`docs/api.md`](docs/api.md) — the Python API
- [`examples/`](examples/) — runnable end-to-end scripts

---

## Development

```bash
pip install -e ".[all,dev]"
pytest                           # the full suite
pytest -m "not slow" -q          # quick run
```

## License

MIT — see [LICENSE](LICENSE).

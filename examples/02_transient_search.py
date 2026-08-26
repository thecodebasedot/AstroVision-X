#!/usr/bin/env python3
"""Search a multi-epoch series for transients.

    python examples/02_transient_search.py [--epochs 6] [--transients 3]

Injects supernova-like transients into a series, searches for them by
difference imaging, and reports which were recovered and how they were
classified.
"""

from __future__ import annotations

import argparse

import numpy as np

from astrovision import SkyConfig, SkySimulator, configure
from astrovision.classify import Classifier
from astrovision.detect import Detector
from astrovision.io.image import ImageSeries
from astrovision.morphology import MorphologyAnalyzer
from astrovision.photometry import Photometer
from astrovision.preprocess import Preprocessor
from astrovision.transient import TransientDetector, describe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--transients", type=int, default=3)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    configure("warning")

    print(f"1. Generating a {args.epochs}-epoch series with "
          f"{args.transients} injected transients ...")
    simulator = SkySimulator(SkyConfig(
        shape=(args.size, args.size), n_stars=70, n_galaxies=12, n_nebulae=0,
        n_clusters=0, n_lenses=0, n_anomalies=0, variable_fraction=0.06,
        seed=args.seed))
    series, static_truth, injected = simulator.generate_series(
        n_epochs=args.epochs, cadence=2.0, n_transients=args.transients)
    variables = [t for t in static_truth if t.variable]
    print(f"   {len(injected)} transients and {len(variables)} variable stars injected")
    for entry in injected:
        curve = [round(p["flux"]) for p in entry["light_curve"]]
        print(f"     ({entry['x']:6.1f}, {entry['y']:6.1f})  light curve {curve}")

    print("\n2. Preprocessing every epoch ...")
    preprocessor = Preprocessor()
    prepared = ImageSeries([preprocessor.run(image) for image in series],
                           name="example_series")

    print("3. Building the reference catalog from the deepest epoch ...")
    catalog, segmentation = Detector().detect(prepared.reference)
    Photometer().run(prepared.reference, catalog, segmentation)
    # Classify too, so host association can prefer galaxies over point sources.
    MorphologyAnalyzer().run(prepared.reference, catalog, segmentation)
    Classifier().run(prepared.reference, catalog)
    print(f"   {len(catalog)} sources: " +
          ", ".join(f"{v} {k}" for k, v in catalog.class_counts().items()))

    print("\n4. Difference imaging, one epoch at a time against a "
          "hold-one-out template ...")
    detector = TransientDetector()
    candidates = detector.run(prepared, catalog)
    vetted = [c for c in candidates if "bogus" not in c.flags]
    print(f"   {detector.report['n_raw_candidates']} residuals -> "
          f"{len(vetted)} pass vetting")
    print(f"   median subtraction quality "
          f"{detector.report['median_subtraction_quality']:.2f} (1.0 = noise-limited)")

    print("\n5. Did we find what was injected?")
    for entry in injected:
        distances = [np.hypot(c.x - entry["x"], c.y - entry["y"]) for c in vetted]
        if distances and min(distances) < 4.0:
            candidate = vetted[int(np.argmin(distances))]
            print(f"   FOUND    ({entry['x']:6.1f}, {entry['y']:6.1f}) -> "
                  f"#{candidate.id}, {candidate.significance:.0f} sigma, "
                  f"{candidate.classification}, {candidate.verdict.value}")
        else:
            print(f"   MISSED   ({entry['x']:6.1f}, {entry['y']:6.1f}) "
                  f"peak flux {entry['peak_flux']:.0f}")

    print("\n6. Every vetted candidate")
    for candidate in vetted:
        print("   " + describe(candidate, catalog))

    print("\nNote: these are candidates. A real search would require a second "
          "independent epoch and, for any supernova claim, a spectrum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

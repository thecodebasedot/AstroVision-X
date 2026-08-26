#!/usr/bin/env python3
"""Train the CNN stamp classifier on simulated data and use it in the pipeline.

    python examples/03_train_a_classifier.py [--epochs 30]

Requires PyTorch: pip install 'astrovision-x[deep]'
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter

import numpy as np

from astrovision import SkyConfig, SkySimulator, configure
from astrovision.core.backend import has
from astrovision.core.types import ObjectClass
from astrovision.preprocess import Preprocessor


def build_training_set(n_fields: int, first_seed: int):
    """Cut labelled postage stamps out of simulated fields."""
    stamps, labels = [], []
    kinds = {"star": ObjectClass.STAR, "galaxy": ObjectClass.GALAXY,
             "nebula": ObjectClass.NEBULA, "cluster": ObjectClass.STAR_CLUSTER}
    preprocessor = Preprocessor()
    for index in range(n_fields):
        simulator = SkySimulator(SkyConfig(
            shape=(200, 200), n_stars=25, n_galaxies=10, n_nebulae=1,
            n_clusters=1, n_lenses=0, n_anomalies=0, seed=first_seed + index))
        image, truth = simulator.generate()
        clean = preprocessor.run(image, estimate_psf=False)
        for entry in truth:
            label = kinds.get(entry.kind)
            if label is None or entry.flux < 1500:
                continue
            if not (24 < entry.x < 176 and 24 < entry.y < 176):
                continue
            stamps.append(clean.cutout(entry.x, entry.y, 48, subtract_background=True))
            labels.append(label)
    return stamps, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fields", type=int, default=14)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configure("warning")
    if not has("torch"):
        print("PyTorch is not installed. Install it with:")
        print("    pip install 'astrovision-x[deep]'")
        return 1

    from astrovision.ml import StampClassifier

    print("1. Building a training set from simulated fields ...")
    stamps, labels = build_training_set(args.fields, 400)
    test_stamps, test_labels = build_training_set(5, 700)
    print(f"   train {len(stamps)} stamps {dict(Counter(l.value for l in labels))}")
    print(f"   test  {len(test_stamps)} stamps")

    print(f"\n2. Training a residual CNN for {args.epochs} epochs ...")
    classes = [ObjectClass.STAR, ObjectClass.GALAXY,
               ObjectClass.NEBULA, ObjectClass.STAR_CLUSTER]
    classifier = StampClassifier(backbone="cnn", classes=classes,
                                 cutout=48, width=16)
    history = classifier.fit(stamps, labels, epochs=args.epochs,
                             batch_size=32, verbose=False)
    print(f"   loss {history[0]:.3f} -> {history[-1]:.3f}")

    print("\n3. Evaluating on held-out fields ...")
    predictions = classifier.predict(test_stamps)
    accuracy = float(np.mean([p == t for p, t in zip(predictions, test_labels)]))
    print(f"   overall accuracy {100 * accuracy:.0f}%")
    per_class: dict = {}
    for predicted, actual in zip(predictions, test_labels):
        correct, total = per_class.get(actual.value, (0, 0))
        per_class[actual.value] = (correct + int(predicted == actual), total + 1)
    for name, (correct, total) in sorted(per_class.items()):
        print(f"     {name:<14} {correct}/{total}")

    print(f"\n4. Embeddings: {classifier.embed(test_stamps[:4]).shape} "
          "(usable for similarity search and anomaly detection)")

    path = args.out or os.path.join(tempfile.mkdtemp(), "stamp_classifier.pt")
    classifier.save(path)
    print(f"\n5. Saved to {path}")
    print("   Use it in the pipeline with:")
    print("       config.classification.backend = 'hybrid'")
    print(f"       config.classification.model_path = '{path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

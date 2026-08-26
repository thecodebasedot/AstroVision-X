#!/usr/bin/env python3
"""Analyse a single field end to end and write a report.

    python examples/01_analyse_a_field.py [--out results/]

Generates a synthetic field, runs the full pipeline over it, and compares the
catalog against the simulator's truth table so the numbers can be checked
rather than taken on trust.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from astrovision import Pipeline, SkyConfig, SkySimulator, configure
from astrovision.report import generate_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/single_field")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    configure("info")

    print("1. Generating a synthetic field with ground truth ...")
    simulator = SkySimulator(SkyConfig(
        shape=(args.size, args.size), n_stars=90, n_galaxies=25, n_nebulae=1,
        n_clusters=1, n_lenses=2, n_anomalies=3, seed=args.seed))
    image, truth = simulator.generate()
    print(f"   injected {len(truth)} objects into a {args.size}x{args.size} field")

    print("\n2. Running the pipeline ...")
    analysis = Pipeline().run(image, redshift=0.15)

    print("\n3. Stage timings")
    for stage in analysis.provenance["stages"]:
        print(f"   {stage['name']:<16}{stage['status']:<9}{stage['seconds']:>7.2f}s")

    print("\n4. Checking the catalog against the truth table")
    positions = analysis.catalog.positions()
    by_kind: dict = {}
    for entry in truth:
        distance = (np.hypot(positions[:, 0] - entry.x, positions[:, 1] - entry.y).min()
                    if len(positions) else np.inf)
        found, total = by_kind.get(entry.kind, (0, 0))
        by_kind[entry.kind] = (found + int(distance < 3.0), total + 1)
    for kind, (found, total) in sorted(by_kind.items()):
        print(f"   {kind:<12} recovered {found:>3}/{total:<3} ({100 * found / total:.0f}%)")

    print("\n5. What the assistant says")
    narrative = analysis.statistics.get("narrative", {})
    print("  ", narrative.get("summary", ""))

    print("\n6. Top candidates for follow-up")
    for text in narrative.get("priority_text", [])[:3]:
        print(text)

    print("\n7. Recommended next steps")
    for index, action in enumerate(narrative.get("recommendations", []), start=1):
        print(f"   {index}. {action}")

    print(f"\n8. Writing reports to {args.out}/ ...")
    written = generate_reports(analysis, args.out, ("text", "json", "html"),
                               title="Example Field", image=image)
    for kind, path in sorted(written.items()):
        print(f"   {kind:<14} {path} ({os.path.getsize(path):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

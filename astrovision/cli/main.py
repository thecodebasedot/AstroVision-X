"""Command-line interface for AstroVision-X.

    astrovision analyze image.fits --report text,html
    astrovision series epoch*.fits --transients
    astrovision simulate --out field.fits --transients 3
    astrovision inspect image.fits
    astrovision config --preset deep_field --out run.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from ..core.backend import describe_capabilities
from ..core.config import PRESETS, AstroVisionConfig
from ..core.exceptions import AstroVisionError
from ..core.logging import configure, get_logger
from ..version import __version__

log = get_logger("cli")

BANNER = r"""
    _        _         __     ___     _              __  __
   / \   ___| |_ _ __ __\ \   / (_)___(_) ___  _ __   \ \/ /
  / _ \ / __| __| '__/ _ \ \ / /| / __| |/ _ \| '_ \   \  /
 / ___ \\__ \ |_| | | (_) \ V / | \__ \ | (_) | | | |  /  \
/_/   \_\___/\__|_|  \___/ \_/  |_|___/_|\___/|_| |_| /_/\_\
"""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_config(args: argparse.Namespace) -> AstroVisionConfig:
    """Build the configuration from a file, a preset and ``--set`` overrides."""
    config = (AstroVisionConfig.load(args.config) if getattr(args, "config", None)
              else AstroVisionConfig())
    if getattr(args, "preset", None):
        config.with_preset(args.preset)
    config.apply_overrides(getattr(args, "set", None))
    if getattr(args, "threshold", None) is not None:
        config.detection.threshold_sigma = float(args.threshold)
    if getattr(args, "output", None):
        config.report.output_dir = args.output
    if getattr(args, "report", None):
        config.report.formats = [f.strip() for f in args.report.split(",") if f.strip()]
    config.log_level = getattr(args, "log_level", config.log_level)
    return config


def _expand(patterns: Sequence[str]) -> List[str]:
    """Expand shell globs that the shell itself did not."""
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def _summarise(analysis) -> str:
    summary = analysis.summary()
    counts = ", ".join(f"{v} {k}" for k, v in summary["class_counts"].items())
    parts = [f"{summary['n_sources']} sources ({counts})"]
    if summary["n_transients"]:
        parts.append(f"{summary['n_transients']} transient candidate(s)")
    if summary["n_lens_candidates"]:
        parts.append(f"{summary['n_lens_candidates']} lens candidate(s)")
    if summary["n_anomalies"]:
        parts.append(f"{summary['n_anomalies']} anomalies ranked")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyse one image (or several, independently)."""
    from ..engine import Pipeline
    from ..io.image import AstroImage
    from ..report import generate_reports

    config = _load_config(args)
    paths = _expand(args.images)
    exit_code = 0

    for path in paths:
        if not os.path.exists(path):
            log.error("file not found: %s", path)
            exit_code = 1
            continue
        try:
            image = AstroImage.load(path)
        except AstroVisionError as exc:
            log.error("could not read %s: %s", path, exc)
            exit_code = 1
            continue

        log.info("analysing %s", path)
        analysis = Pipeline(config).run(image, redshift=args.redshift)

        output_dir = config.report.output_dir
        if len(paths) > 1:
            output_dir = os.path.join(output_dir,
                                      os.path.splitext(os.path.basename(path))[0])
        written = generate_reports(
            analysis, output_dir, config.report.formats,
            title=config.report.title, observer=config.report.observer,
            top_candidates=config.report.top_candidates,
            include_embeddings=config.report.include_embeddings,
            image=image)

        print(f"\n{path}: {_summarise(analysis)}")
        for kind, written_path in sorted(written.items()):
            print(f"  {kind:<14} {written_path}")
        if analysis.warnings:
            print("  warnings:")
            for warning in analysis.warnings:
                print(f"    - {warning}")
        if args.print_report and "text" in written:
            with open(written["text"], encoding="utf-8") as handle:
                print("\n" + handle.read())
    return exit_code


def cmd_series(args: argparse.Namespace) -> int:
    """Analyse a multi-epoch series and search it for transients."""
    from ..engine import Pipeline
    from ..io.image import ImageSeries
    from ..report import generate_reports

    config = _load_config(args)
    paths = _expand(args.images)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        log.error("file(s) not found: %s", ", ".join(missing))
        return 1
    if len(paths) < 2:
        log.error("a series needs at least two images; got %d", len(paths))
        return 1

    log.info("loading %d epochs", len(paths))
    series = ImageSeries.from_paths(paths, name=args.name or "series")
    for problem in series.check_alignment():
        log.warning("series consistency: %s", problem)

    analysis = Pipeline(config).run_series(series, redshift=args.redshift)
    written = generate_reports(
        analysis, config.report.output_dir, config.report.formats,
        title=config.report.title, observer=config.report.observer,
        top_candidates=config.report.top_candidates,
        image=series.reference)

    print(f"\n{len(paths)} epochs: {_summarise(analysis)}")
    vetted = [c for c in analysis.transients if "bogus" not in c.flags]
    if vetted:
        print("\nTransient candidates:")
        from ..transient.supernova import describe
        for candidate in vetted[:args.top]:
            print("  " + describe(candidate, analysis.catalog))
    for kind, path in sorted(written.items()):
        print(f"  {kind:<14} {path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Generate a synthetic field or series, with its truth table."""
    from ..simulate import SkyConfig, SkySimulator

    config = SkyConfig(
        shape=(args.size, args.size), seed=args.seed,
        n_stars=args.stars, n_galaxies=args.galaxies,
        n_nebulae=args.nebulae, n_clusters=args.clusters,
        n_lenses=args.lenses, n_anomalies=args.anomalies,
        seeing_fwhm=args.seeing, background=args.background)
    simulator = SkySimulator(config)

    output = args.out or "synthetic_field.fits"
    directory = os.path.dirname(os.path.abspath(output))
    os.makedirs(directory, exist_ok=True)
    base, extension = os.path.splitext(output)

    if args.epochs > 1:
        series, truth, transients = simulator.generate_series(
            n_epochs=args.epochs, cadence=args.cadence,
            n_transients=args.transients)
        written: List[str] = []
        for index, image in enumerate(series):
            path = f"{base}_epoch{index:02d}{extension or '.fits'}"
            image.write(path)
            written.append(path)
        truth_payload = {"static": [t.to_dict() for t in truth],
                         "transients": transients,
                         "epochs": written,
                         "cadence": args.cadence}
        print(f"wrote {len(written)} epochs with {len(transients)} injected transients")
        for path in written:
            print(f"  {path}")
    else:
        image, truth = simulator.generate()
        image.write(output)
        truth_payload = {"static": [t.to_dict() for t in truth]}
        print(f"wrote {output} ({image.shape[1]}x{image.shape[0]} px, "
              f"{len(truth)} truth objects)")

    truth_path = f"{base}_truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth_payload, handle, indent=2, default=float)
    print(f"  truth table   {truth_path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a summary of a FITS file without running the pipeline."""
    from ..io.fits import list_hdus
    from ..io.image import AstroImage

    for path in _expand(args.images):
        if not os.path.exists(path):
            log.error("file not found: %s", path)
            continue
        print(f"\n{path}")
        try:
            for hdu in list_hdus(path):
                print(f"  HDU {hdu['index']}: {hdu['name']} shape={hdu['shape']}")
        except Exception as exc:                    # noqa: BLE001 - informational
            print(f"  (could not list HDUs: {exc})")
        image = AstroImage.load(path)
        print("  " + image.describe().replace("\n", "\n  "))
        if args.header:
            print("  header:")
            for key, value in list(image.header.items())[:40]:
                print(f"    {key:<10} {value}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show or write a configuration file."""
    config = _load_config(args)
    if args.out:
        config.save(args.out)
        print(f"wrote configuration to {args.out}")
    elif args.show:
        print("\n".join(config.describe()))
    elif not args.list_presets:
        # Listing presets on its own should not also dump the whole config.
        print(json.dumps(config.to_dict(), indent=2))
    if args.list_presets:
        print("\nAvailable presets:")
        for name, values in PRESETS.items():
            print(f"  {name}")
            for key, value in values.items():
                print(f"      {key} = {value}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Report the installed version and which optional backends are present."""
    print(BANNER)
    print(f"AstroVision-X {__version__}")
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(f"NumPy {np.__version__}")
    print("\nOptional backends:")
    print(describe_capabilities())
    print("\nConfiguration presets: " + ", ".join(PRESETS))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrovision",
        description="Computer vision and machine learning for astronomical imagery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"AstroVision-X {__version__}")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="verbosity of the log stream (default: info)")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-c", "--config", help="configuration file (JSON or YAML)")
        sub.add_argument("-p", "--preset", choices=sorted(PRESETS),
                         help="named parameter preset")
        sub.add_argument("--set", action="append", metavar="KEY=VALUE",
                         help="override one option, e.g. detection.threshold_sigma=4")
        sub.add_argument("-o", "--output", help="output directory for reports")
        sub.add_argument("--report", help="comma-separated formats: text,json,html")
        sub.add_argument("--threshold", type=float,
                         help="detection threshold in sigma")
        sub.add_argument("-z", "--redshift", type=float,
                         help="assume this redshift when deriving physical quantities")

    analyze = subparsers.add_parser("analyze", help="analyse one or more images")
    analyze.add_argument("images", nargs="+", help="FITS or image files")
    analyze.add_argument("--print-report", action="store_true",
                         help="print the text report to stdout as well")
    add_common(analyze)
    analyze.set_defaults(func=cmd_analyze)

    series = subparsers.add_parser("series",
                                   help="analyse a multi-epoch series for transients")
    series.add_argument("images", nargs="+", help="epoch files, in any order")
    series.add_argument("--name", help="name for the series")
    series.add_argument("--top", type=int, default=10,
                        help="how many transient candidates to print")
    add_common(series)
    series.set_defaults(func=cmd_series)

    simulate = subparsers.add_parser("simulate",
                                     help="generate a synthetic field with truth")
    simulate.add_argument("--out", help="output FITS path")
    simulate.add_argument("--size", type=int, default=512, help="image size in pixels")
    simulate.add_argument("--stars", type=int, default=200)
    simulate.add_argument("--galaxies", type=int, default=40)
    simulate.add_argument("--nebulae", type=int, default=2)
    simulate.add_argument("--clusters", type=int, default=1)
    simulate.add_argument("--lenses", type=int, default=1)
    simulate.add_argument("--anomalies", type=int, default=2)
    simulate.add_argument("--epochs", type=int, default=1,
                          help="number of epochs; >1 writes a series")
    simulate.add_argument("--cadence", type=float, default=2.0,
                          help="days between epochs")
    simulate.add_argument("--transients", type=int, default=0,
                          help="transients to inject (series only)")
    simulate.add_argument("--seeing", type=float, default=3.2,
                          help="seeing FWHM in pixels")
    simulate.add_argument("--background", type=float, default=120.0)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.set_defaults(func=cmd_simulate)

    inspect = subparsers.add_parser("inspect", help="summarise a file without analysing")
    inspect.add_argument("images", nargs="+")
    inspect.add_argument("--header", action="store_true", help="print header keywords")
    inspect.set_defaults(func=cmd_inspect)

    config_parser = subparsers.add_parser("config", help="show or write configuration")
    config_parser.add_argument("--out", help="write the configuration here")
    config_parser.add_argument("--show", action="store_true",
                               help="print as flat key = value lines")
    config_parser.add_argument("--list-presets", action="store_true")
    add_common(config_parser)
    config_parser.set_defaults(func=cmd_config)

    info = subparsers.add_parser("info", help="show version and available backends")
    info.set_defaults(func=cmd_info)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``astrovision`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(getattr(args, "log_level", "info"))

    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except AstroVisionError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:                        # pragma: no cover - interactive
        log.warning("interrupted")
        return 130


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())

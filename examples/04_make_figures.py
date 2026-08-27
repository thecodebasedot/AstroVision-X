#!/usr/bin/env python3
"""Render figures showing what each stage of the pipeline actually produced.

    python examples/04_make_figures.py --out figures/

Runs the pipeline on simulated data and draws the result: detections over the
field, the stages side by side, measured galaxy morphology, a transient
recovered by difference imaging, its light curve, a lens candidate's arcs, and
the top novelty candidates.

Requires Matplotlib: pip install 'astrovision-x[viz]'
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from astrovision import SkyConfig, SkySimulator, configure
from astrovision.core.backend import has

# --- chart parameters -------------------------------------------------------
# Categorical hues, in fixed slot order, stepped for a dark surface and
# validated for all-pairs colour-vision separation. Never cycled: the fourth
# thing on a plot gets a shape, not a new hue.
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_MUTED = "#c3c2b7"
SERIES = {"star": "#3987e5", "galaxy": "#d95926", "other": "#199e70"}
HIGHLIGHT = "#e66767"          # status: the thing being pointed at
GRID = "#3a3a37"


def style(plt) -> None:
    """One dark chart surface for every figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_MUTED, "axes.edgecolor": GRID,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.5,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "600",
        "legend.frameon": True, "legend.labelcolor": INK_MUTED,
        "legend.facecolor": SURFACE, "legend.edgecolor": GRID,
        "legend.framealpha": 0.88,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "figure.dpi": 130,
    })


def show_image(axes, data, title="", stretch=True, limits=None):
    """Draw an astronomical image: one hue, light to dark, no rainbow.

    Pass ``limits`` to hold several panels on one scale -- a subtraction
    triplet is unreadable if each panel is stretched independently, because
    the residual then looks as bright as the source it came from.
    """
    from astrovision.preprocess.normalize import asinh_stretch, zscale

    if limits is not None:
        axes.imshow(data, origin="lower", cmap="gray", interpolation="nearest",
                    vmin=limits[0], vmax=limits[1])
    else:
        axes.imshow(asinh_stretch(data) if stretch else zscale(data),
                    origin="lower", cmap="gray", interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_edgecolor(GRID)
    if title:
        axes.set_title(title, color=INK, pad=6)
    return axes


# ---------------------------------------------------------------------------
def figure_field(plt, image, clean, catalog, path):
    """The whole field, with every detection marked by class."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    show_image(axes[0], image.data, "1. Raw field, as observed")
    show_image(axes[1], clean.subtracted(),
               f"2. Background-subtracted, {len(catalog)} sources detected")

    # Only three hues validate for all-pairs separation, so classes past the
    # third are folded onto the third hue and distinguished by line style --
    # identity is never carried by colour alone.
    styles = {"star": (SERIES["star"], "solid"),
              "galaxy": (SERIES["galaxy"], "solid"),
              "star_cluster": (SERIES["other"], "solid"),
              "nebula": (SERIES["other"], (0, (4, 2))),
              "artifact": (SERIES["other"], (0, (1, 2)))}
    default = (SERIES["other"], (0, (6, 3)))

    counts = {}
    for source in catalog:
        kind = source.object_class.value
        colour, dashes = styles.get(kind, default)
        radius = max(source.morphology.semi_major * 2.2, 3.5)
        axes[1].add_patch(plt.Circle((source.x, source.y), radius, fill=False,
                                     edgecolor=colour, linewidth=1.1,
                                     linestyle=dashes, alpha=0.9))
        counts[kind] = counts.get(kind, 0) + 1

    # Direct-label the brightest few rather than numbering every source.
    for source in catalog.brightest(4):
        axes[1].annotate(f"#{source.id}  m={source.photometry.magnitude:.1f}",
                         (source.x, source.y),
                         xytext=(6, 6), textcoords="offset points",
                         color=INK, fontsize=7.5)

    handles = []
    for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        colour, dashes = styles.get(kind, default)
        handles.append(plt.Line2D([], [], color=colour, linestyle=dashes,
                                  linewidth=1.8,
                                  label=f"{kind.replace('_', ' ')}  ({count})"))
    axes[1].legend(handles=handles, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.02), fontsize=8.5)

    figure.suptitle("AstroVision-X — detection and classification",
                    color=INK, fontsize=13, fontweight="600", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_stages(plt, image, clean, catalog, segmentation, path):
    """The same 120-pixel corner as it passes through each stage."""
    from matplotlib.colors import ListedColormap

    size = min(150, image.shape[0] // 2)
    # Centre the crop on the busiest part of the field.
    positions = catalog.positions()
    if len(positions):
        centre = np.median(positions, axis=0).astype(int)
    else:
        centre = np.array(image.shape[::-1]) // 2
    x0 = int(np.clip(centre[0] - size // 2, 0, image.shape[1] - size))
    y0 = int(np.clip(centre[1] - size // 2, 0, image.shape[0] - size))
    cut = (slice(y0, y0 + size), slice(x0, x0 + size))

    figure, axes = plt.subplots(1, 4, figsize=(14.5, 4.1))
    show_image(axes[0], image.data[cut], "Raw")
    background = clean.meta.get("background_model")
    if background is None:
        background = np.full_like(image.data, float(np.median(image.data)))
    show_image(axes[1], background[cut], "Background model", stretch=False)
    show_image(axes[2], clean.subtracted()[cut], "Subtracted")

    labels = segmentation[cut]
    rng = np.random.default_rng(0)
    palette = np.vstack([[0, 0, 0],
                         rng.uniform(0.25, 1.0, (max(int(labels.max()), 1), 3))])
    axes[3].imshow(ListedColormap(palette)(labels % len(palette)),
                   origin="lower", interpolation="nearest")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_title(f"Deblended segments ({int(labels.max())} here)", color=INK, pad=6)
    for spine in axes[3].spines.values():
        spine.set_edgecolor(GRID)

    figure.suptitle("One corner of the field, stage by stage",
                    color=INK, fontsize=13, fontweight="600", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_morphology(plt, clean, catalog, truth, path):
    """Galaxy cutouts beside the statistics measured from them."""
    positions = catalog.positions()
    chosen = []
    for entry in truth:
        if entry.kind != "galaxy" or len(chosen) >= 5:
            continue
        distance = np.hypot(positions[:, 0] - entry.x, positions[:, 1] - entry.y)
        if not len(distance) or distance.min() > 4:
            continue
        source = catalog[int(distance.argmin())]
        if source.morphology.label.value in ("unknown", "unresolved"):
            continue
        if source.morphology.area_pixels < 60:
            continue
        chosen.append((entry, source))
    if not chosen:
        return None

    figure, axes = plt.subplots(2, len(chosen), figsize=(2.9 * len(chosen), 6.6),
                                gridspec_kw={"height_ratios": [1.3, 1],
                                             "hspace": 0.18})
    axes = np.atleast_2d(axes)
    for column, (entry, source) in enumerate(chosen):
        size = int(np.clip(source.morphology.semi_major * 9, 40, 120))
        show_image(axes[0, column], clean.cutout(source.x, source.y, size, True))
        axes[0, column].set_title(
            f"injected: {entry.morphology.replace('_', ' ')}\n"
            f"measured: {source.morphology.label.value.replace('_', ' ')}",
            color=INK if entry.morphology == source.morphology.label.value else INK_MUTED,
            fontsize=9, pad=5)

        morphology = source.morphology
        rows = [("Sérsic n", morphology.sersic_index, entry.sersic_n),
                ("Concentration", morphology.concentration, None),
                ("Asymmetry", morphology.asymmetry, None),
                ("Gini", morphology.gini, None),
                ("M20", morphology.m20, None),
                ("arms", float(morphology.arm_count), None)]
        axes[1, column].axis("off")
        for row, (name, measured, true_value) in enumerate(rows):
            y = 0.86 - row * 0.155
            axes[1, column].text(0.02, y, name, color=INK_MUTED, fontsize=8.5,
                                 transform=axes[1, column].transAxes)
            text = "—" if not np.isfinite(measured) else (
                f"{measured:.0f}" if name == "arms" else f"{measured:.2f}")
            if true_value is not None and np.isfinite(true_value):
                text += f"   (true {true_value:.2f})"
            axes[1, column].text(0.98, y, text, color=INK, fontsize=8.5, ha="right",
                                 transform=axes[1, column].transAxes)

    figure.suptitle("Galaxy morphology: what was injected, and what was measured",
                    color=INK, fontsize=13, fontweight="600", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_transients(plt, series, candidates, injected, path):
    """Template, new epoch and difference, with the transient circled."""
    from astrovision.transient.difference import build_template, subtract

    real = [c for c in candidates if "bogus" not in c.flags
            and c.classification == "supernova_candidate"][:2]
    if not real:
        return None

    figure, axes = plt.subplots(len(real), 3, figsize=(11.0, 3.8 * len(real)))
    axes = np.atleast_2d(axes)
    for row, candidate in enumerate(real):
        epoch = int(candidate.epoch_index)
        science = series[epoch]
        template = build_template(series, "median", exclude=epoch)
        result = subtract(science, template)
        size = 70
        x, y = candidate.x, candidate.y

        def crop(data):
            half = size // 2
            y0 = int(np.clip(y - half, 0, data.shape[0] - size))
            x0 = int(np.clip(x - half, 0, data.shape[1] - size))
            return data[y0:y0 + size, x0:x0 + size], (x - x0, y - y0)

        template_cut, _ = crop(template.subtracted())
        science_cut, _ = crop(science.subtracted())
        difference_cut, local = crop(result.difference)

        # All three panels share the science epoch's scale, so the residual
        # can be compared with the source rather than re-stretched to match it.
        limits = (float(np.percentile(science_cut, 1)),
                  float(np.percentile(science_cut, 99.7)))
        show_image(axes[row, 0], template_cut,
                   f"Template — median of the other {len(series) - 1} epochs",
                   limits=limits)
        show_image(axes[row, 1], science_cut, f"Epoch {epoch}", limits=limits)
        show_image(axes[row, 2], difference_cut,
                   f"Difference — {candidate.significance:.0f}σ residual",
                   limits=limits)
        for column in range(3):
            axes[row, column].add_patch(plt.Circle(
                local, 9, fill=False, edgecolor=HIGHLIGHT,
                linewidth=1.6, alpha=0.95))
        axes[row, 2].text(
            0.04, 0.06,
            f"real/bogus {candidate.real_bogus:.2f}\n"
            f"{candidate.classification.replace('_', ' ')}\n"
            f"{candidate.verdict.value.replace('_', ' ')}",
            transform=axes[row, 2].transAxes, color=INK, fontsize=8,
            va="bottom", linespacing=1.5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=3))

    figure.suptitle("Transient discovery by difference imaging",
                    color=INK, fontsize=13, fontweight="600", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_light_curves(plt, candidates, injected, path):
    """Recovered light curves against what was injected."""
    real = [c for c in candidates
            if "bogus" not in c.flags and c.light_curve is not None][:3]
    if not real:
        return None

    figure, axes = plt.subplots(1, len(real), figsize=(4.6 * len(real), 4.0),
                                squeeze=False)
    for column, candidate in enumerate(real):
        panel = axes[0, column]
        curve = candidate.light_curve.clean()
        times = curve.times - curve.times[0]

        match = None
        for entry in injected:
            if np.hypot(candidate.x - entry["x"], candidate.y - entry["y"]) < 4:
                match = entry
                break

        if match is not None:
            truth_times = [p["time"] for p in match["light_curve"]]
            truth_flux = [p["flux"] for p in match["light_curve"]]
            panel.plot(truth_times, truth_flux, color=SERIES["other"],
                       linewidth=2.0, label="injected", zorder=2)
        panel.errorbar(times, curve.fluxes,
                       yerr=curve.errors if curve.errors is not None else None,
                       fmt="o", color=SERIES["star"], markersize=7,
                       markeredgecolor=SURFACE, markeredgewidth=2,
                       elinewidth=1.2, capsize=0, label="measured", zorder=3)

        peak = int(np.argmax(curve.fluxes))
        panel.annotate(f"peak {curve.fluxes[peak]:,.0f}",
                       (times[peak], curve.fluxes[peak]),
                       xytext=(8, 4), textcoords="offset points",
                       color=INK, fontsize=8)

        panel.set_xlabel("days since first epoch")
        if column == 0:
            panel.set_ylabel("flux (counts)")
        panel.set_title(f"Candidate #{candidate.id} — "
                        f"{candidate.classification.replace('_', ' ')}",
                        color=INK, pad=6)
        panel.grid(True, axis="y")
        panel.set_axisbelow(True)
        panel.legend(loc="upper left", fontsize=8)

    figure.suptitle("Light curves: measured against injected",
                    color=INK, fontsize=13, fontweight="600", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_lens(plt, clean, catalog, lenses, path):
    """A lens candidate: the galaxy, the residual, and the arcs found."""
    from astrovision.lensing.arcs import detect_arcs, subtract_smooth_light

    if not lenses:
        return None
    lens = lenses[0]
    source = catalog.by_id(lens.source_id)
    if source is None:
        return None

    reach = max(4.0 * max(source.morphology.semi_major, 2.0), 22.0)
    size = int(2 * np.ceil(reach) + 1)
    cutout = clean.cutout(source.x, source.y, size, subtract_background=True)
    centre = ((cutout.shape[1] - 1) / 2.0, (cutout.shape[0] - 1) / 2.0)
    residual = subtract_smooth_light(cutout, centre)
    arcs = detect_arcs(cutout, centre, float(source.meta.get("local_rms", 10.0)),
                       min_axis_ratio=2.0, max_width=7.0,
                       min_radius=max(3.0, 0.9 * source.morphology.semi_major))

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    show_image(axes[0], cutout, "Candidate deflector")
    show_image(axes[1], residual, "Smooth galaxy light removed")
    show_image(axes[2], residual, f"{len(arcs)} tangential arc(s)")

    if np.isfinite(lens.einstein_radius_px) and lens.einstein_radius_px > 0:
        axes[2].add_patch(plt.Circle(centre, lens.einstein_radius_px, fill=False,
                                     edgecolor=SERIES["other"], linewidth=1.2,
                                     linestyle="--", alpha=0.9,
                                     label="Einstein radius"))
    for arc in arcs:
        angle = np.deg2rad(arc.angle)
        axes[2].plot(centre[0] + arc.radius * np.cos(angle),
                     centre[1] + arc.radius * np.sin(angle),
                     marker="o", markersize=9, markerfacecolor="none",
                     markeredgecolor=HIGHLIGHT, markeredgewidth=1.8)
    axes[2].text(0.04, 0.05,
                 f"score {lens.score:.2f}\n"
                 f"θ_E = {lens.einstein_radius_arcsec:.2f}″\n"
                 f"ring {100 * lens.ring_completeness:.0f}% complete",
                 transform=axes[2].transAxes, color=INK, fontsize=8,
                 va="bottom", linespacing=1.5,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75, pad=3))
    handles = [plt.Line2D([], [], marker="o", markerfacecolor="none", linestyle="",
                          markeredgecolor=HIGHLIGHT, markersize=9,
                          markeredgewidth=1.4, label="detected arc"),
               plt.Line2D([], [], color=SERIES["other"], linestyle="--",
                          linewidth=1.2, label="Einstein radius")]
    axes[2].legend(handles=handles, loc="upper right", fontsize=8)

    figure.suptitle("Gravitational-lens candidate — candidate only, "
                    "confirmation needs colours and redshifts",
                    color=INK, fontsize=12, fontweight="600", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_anomalies(plt, clean, catalog, anomalies, path):
    """The top novelty candidates, with the reason each was flagged."""
    records = anomalies[:5]
    if not records:
        return None
    figure, axes = plt.subplots(1, len(records), figsize=(2.9 * len(records), 3.9))
    axes = np.atleast_1d(axes)
    for column, record in enumerate(records):
        source = catalog.by_id(record.source_id)
        if source is None:
            axes[column].axis("off")
            continue
        size = int(np.clip(max(source.morphology.semi_major * 8, 36), 36, 110))
        show_image(axes[column], clean.cutout(source.x, source.y, size, True))
        axes[column].set_title(
            f"#{record.rank}  score {record.score:.2f}\n"
            f"{record.novelty_type.replace('_', ' ')}",
            color=INK, fontsize=9, pad=5)
        # Split on the clause separator, not on "." -- that cut "A=0.44" to "A=0".
        reason = record.explanation.replace("Unusual because ", "").split(";")[0]
        reason = reason.split(". This is a candidate")[0].strip()
        axes[column].set_xlabel(_wrap(reason, 34), color=INK_MUTED, fontsize=7.5,
                                labelpad=6)

    figure.suptitle("Novelty candidates — unusual for this field, "
                    "which is not the same as new",
                    color=INK, fontsize=12, fontweight="600", y=1.0)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width)[:3])


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    if not has("matplotlib"):
        print("Matplotlib is not installed. Install it with:")
        print("    pip install 'astrovision-x[viz]'")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure("warning")
    style(plt)
    os.makedirs(args.out, exist_ok=True)
    written = []

    from astrovision import Pipeline
    from astrovision.io.image import ImageSeries
    from astrovision.preprocess import Preprocessor

    print("Running the pipeline on a single field ...")
    simulator = SkySimulator(SkyConfig(
        shape=(args.size, args.size), n_stars=80, n_galaxies=22, n_nebulae=1,
        n_clusters=1, n_lenses=2, n_anomalies=3, seed=args.seed))
    image, truth = simulator.generate()

    pipeline = Pipeline()
    analysis = pipeline.run(image, redshift=0.12)
    clean = pipeline.preprocessor.run(image)
    _, segmentation = pipeline.detector.detect(clean)

    written.append(figure_field(plt, image, clean, analysis.catalog,
                                os.path.join(args.out, "01_field_detections.png")))
    written.append(figure_stages(plt, image, clean, analysis.catalog, segmentation,
                                 os.path.join(args.out, "02_pipeline_stages.png")))
    written.append(figure_morphology(plt, clean, analysis.catalog, truth,
                                     os.path.join(args.out, "03_galaxy_morphology.png")))
    written.append(figure_anomalies(plt, clean, analysis.catalog, analysis.anomalies,
                                    os.path.join(args.out, "06_anomalies.png")))
    written.append(figure_lens(plt, clean, analysis.catalog, analysis.lenses,
                               os.path.join(args.out, "07_lens_candidate.png")))

    print("Running the transient search on a multi-epoch series ...")
    series_simulator = SkySimulator(SkyConfig(
        shape=(300, 300), n_stars=70, n_galaxies=12, n_nebulae=0, n_clusters=0,
        n_lenses=0, n_anomalies=0, variable_fraction=0.06, seed=5))
    raw_series, _, injected = series_simulator.generate_series(
        n_epochs=6, cadence=2.0, n_transients=3)
    preprocessor = Preprocessor()
    prepared = ImageSeries([preprocessor.run(epoch) for epoch in raw_series])

    from astrovision.detect import Detector
    from astrovision.photometry import Photometer
    from astrovision.transient import TransientDetector

    catalog, segmentation = Detector().detect(prepared.reference)
    Photometer().run(prepared.reference, catalog, segmentation)
    candidates = TransientDetector().run(prepared, catalog)

    written.append(figure_transients(plt, prepared, candidates, injected,
                                     os.path.join(args.out, "04_transient_discovery.png")))
    written.append(figure_light_curves(plt, candidates, injected,
                                       os.path.join(args.out, "05_light_curves.png")))

    print()
    for path in [p for p in written if p]:
        print(f"  {path}  ({os.path.getsize(path):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Declarative configuration for the whole analysis pipeline.

A run is fully described by an :class:`AstroVisionConfig`, which can be
built in Python, loaded from JSON/YAML, overridden from the command line
(``--set detection.threshold_sigma=4``), and serialised into the report so
a result is always reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

from .backend import try_import
from .exceptions import ConfigError


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort conversion of ``value`` to the annotated ``target_type``."""
    if value is None:
        return None
    origin = getattr(target_type, "__origin__", None)
    if origin is not None:                       # Optional[...] / List[...] / Tuple[...]
        args = [a for a in getattr(target_type, "__args__", ()) if a is not type(None)]
        if origin in (list, List):
            return [_coerce(v, args[0]) if args else v for v in value]
        if origin in (tuple, Tuple):
            return tuple(value)
        if origin is dict:
            return dict(value)
        return _coerce(value, args[0]) if args else value
    if target_type is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if target_type in (int, float, str):
        try:
            return target_type(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"cannot convert {value!r} to {target_type.__name__}") from exc
    return value


_HINT_CACHE: Dict[type, Dict[str, Any]] = {}


def _hints(cls: type) -> Dict[str, Any]:
    """Resolved type hints for a config section.

    ``from __future__ import annotations`` turns every annotation into a
    string, so dataclass ``field.type`` cannot be used directly; resolve
    and cache the real objects once per class.
    """
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = get_type_hints(cls, globalns=dict(globals()))
    return _HINT_CACHE[cls]


@dataclass
class _Section:
    """Base for config sections: dict round-tripping plus dotted updates."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        data = dict(data or {})
        hints = _hints(cls)
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                raise ConfigError(f"unknown option '{key}' in section '{cls.__name__}'")
            annotation = hints.get(key, Any)
            if is_dataclass(annotation) and isinstance(value, dict):
                kwargs[key] = annotation.from_dict(value)
            else:
                kwargs[key] = _coerce(value, annotation)
        return cls(**kwargs)

    def update(self, key_path: str, value: Any) -> None:
        """Set a nested option addressed as ``a.b.c``."""
        head, _, rest = key_path.partition(".")
        hints = _hints(type(self))
        if head not in {f.name for f in fields(self)}:
            raise ConfigError(f"unknown option '{head}' in '{type(self).__name__}'")
        current = getattr(self, head)
        if rest:
            if not isinstance(current, _Section):
                raise ConfigError(f"'{head}' is not a config section")
            current.update(rest, value)
        else:
            setattr(self, head, _coerce(value, hints.get(head, Any)))

    def describe(self, prefix: str = "") -> List[str]:
        """Flatten to ``key.path = value`` lines for logging and reports."""
        lines: List[str] = []
        for spec in fields(self):
            value = getattr(self, spec.name)
            key = f"{prefix}{spec.name}"
            if isinstance(value, _Section):
                lines.extend(value.describe(key + "."))
            else:
                lines.append(f"{key} = {value!r}")
        return lines


@dataclass
class PreprocessConfig(_Section):
    """Instrumental calibration and background handling."""

    subtract_background: bool = True
    background_box: int = 64
    background_filter: int = 3
    reject_cosmic_rays: bool = True
    mask_bad_columns: bool = True
    bad_column_sigma: float = 6.0
    cosmic_ray_sigma: float = 6.0
    cosmic_ray_contrast: float = 2.0
    normalize: str = "zscale"          # zscale | asinh | percentile | zscore | none
    mask_saturated: bool = True
    saturation_level: float = float("inf")
    smooth_sigma: float = 0.0


@dataclass
class DetectionConfig(_Section):
    """Source detection and deblending."""

    backend: str = "threshold"         # threshold | dnn
    threshold_sigma: float = 3.5
    min_area: int = 5
    max_area: int = 200_000
    filter_fwhm: float = 2.5
    deblend: bool = True
    deblend_levels: int = 32
    deblend_contrast: float = 0.005
    border_margin: int = 2
    max_sources: int = 50_000
    bbox_pad: float = 1.5
    model_path: Optional[str] = None
    dnn_score_threshold: float = 0.3
    dnn_nms_iou: float = 0.4


@dataclass
class SegmentationConfig(_Section):
    """Per-object segmentation and galaxy component decomposition."""

    enabled: bool = True
    backend: str = "classical"         # classical | unet
    watershed: bool = True
    decompose_galaxies: bool = True
    component_levels: int = 4
    model_path: Optional[str] = None
    cutout_size: int = 64


@dataclass
class PhotometryConfig(_Section):
    """Aperture and isophotal flux measurement."""

    aperture_radii: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0, 8.0, 12.0])
    primary_aperture: float = 5.0
    auto_aperture: bool = True
    kron_factor: float = 2.5
    petrosian_eta: float = 0.2
    zero_point: float = 25.0
    gain: float = 1.0
    pixel_scale: float = 1.0           # arcsec / pixel
    local_background: bool = True
    annulus_inner: float = 8.0
    annulus_outer: float = 14.0


@dataclass
class MorphologyConfig(_Section):
    """Non-parametric and parametric shape analysis."""

    enabled: bool = True
    compute_cas: bool = True
    compute_gini_m20: bool = True
    fit_sersic: bool = True
    detect_spiral_arms: bool = True
    polar_bins: int = 128
    min_area_for_morphology: int = 12
    uncertainty: bool = False          # bootstrap error bars; costs ~n_samples x
    bootstrap_samples: int = 24


@dataclass
class ClassificationConfig(_Section):
    """Object-class assignment."""

    backend: str = "hybrid"            # rules | ml | hybrid | cnn
    model_path: Optional[str] = None
    star_galaxy_threshold: float = 0.5
    min_confidence: float = 0.15
    use_embeddings: bool = True
    embedding_dim: int = 64
    use_colours: bool = True           # fold the stellar locus into the decision
    colour_weight: float = 0.8         # relative to the morphological answer


@dataclass
class MultiBandConfig(_Section):
    """Forced photometry across filters and the colours derived from it."""

    enabled: bool = True
    detection_band: Optional[str] = None    # defaults to the first band given
    aperture_arcsec: float = 1.6            # the colour aperture, not the total
    annulus_arcsec: Tuple[float, float] = (5.0, 9.0)
    homogenise_psf: bool = True
    target_fwhm_arcsec: Optional[float] = None
    use_kron: bool = False
    min_colour_snr: float = 5.0
    colour_pairs: Optional[List[Tuple[str, str]]] = None   # adjacent bands by default


@dataclass
class CalibrationConfig(_Section):
    """Astrometric and photometric calibration against reference standards.

    Both draw their reference catalog from the ``crossmatch`` backend, so
    configuring one service serves all three uses.
    """

    astrometry: bool = False               # refit the WCS
    photometry: bool = False               # refit the zero point
    match_radius_arcsec: float = 5.0       # generous: the header may be wrong
    min_matches: int = 8
    rounds: int = 3
    standard_radius_arcsec: float = 2.0
    min_standards: int = 5
    min_standard_snr: float = 20.0
    reference_band: Optional[str] = None
    colour_pair: Optional[Tuple[str, str]] = None


@dataclass
class CrossmatchConfig(_Section):
    """Checking sources against external catalogs of known objects."""

    backend: str = "none"                   # none | local | vizier | simbad
    path: Optional[str] = None              # for the local backend
    catalog: str = "I/355/gaiadr3"          # for VizieR
    radius_arcsec: float = 2.0
    max_field_radius_arcsec: float = 3600.0
    timeout: float = 20.0
    cache_dir: Optional[str] = None
    cache_max_age_days: float = 30.0


@dataclass
class AnomalyConfig(_Section):
    """Novelty / outlier discovery."""

    enabled: bool = True
    methods: List[str] = field(default_factory=lambda: ["isolation_forest", "autoencoder", "knn"])
    contamination: float = 0.02
    n_estimators: int = 128
    autoencoder_latent: int = 8
    autoencoder_epochs: int = 120
    knn_neighbours: int = 8
    top_k: int = 20
    random_state: int = 42


@dataclass
class TransientConfig(_Section):
    """Difference imaging and transient candidate vetting."""

    enabled: bool = True
    align: bool = True
    psf_match: bool = True
    detection_sigma: float = 5.0
    min_area: int = 3
    # Calibrated on simulated fields: genuine transients score above 0.84
    # and subtraction artefacts below 0.72, so 0.7 keeps every real
    # candidate while removing most of the artefacts.  Lower it for a
    # completeness-driven search, raise it for a purity-driven one.
    real_bogus_threshold: float = 0.7
    host_search_radius: float = 25.0
    max_candidates: int = 500
    reject_dipoles: bool = True
    dipole_threshold: float = 0.35


@dataclass
class TimeSeriesConfig(_Section):
    """Light-curve extraction and variability analysis."""

    enabled: bool = True
    aperture_radius: float = 4.0
    min_epochs: int = 3
    variability_threshold: float = 3.0
    period_search: bool = True
    min_period: float = 0.02
    max_period: float = 100.0
    n_frequencies: int = 2000
    fap_threshold: float = 0.01


@dataclass
class LensingConfig(_Section):
    """Strong gravitational-lens candidate search."""

    enabled: bool = True
    min_arc_length: float = 5.0
    max_arc_width: float = 7.0
    min_axis_ratio: float = 2.0
    ring_bins: int = 72
    score_threshold: float = 0.5
    search_radius_factor: float = 4.0


@dataclass
class ClusteringConfig(_Section):
    """Embedding-space clustering of the catalog."""

    enabled: bool = True
    method: str = "kmeans"             # kmeans | dbscan | hdbscan
    n_clusters: int = 8
    min_cluster_size: int = 5
    eps: float = 0.5
    random_state: int = 42


@dataclass
class CosmologyConfig(_Section):
    """Cosmological parameters used by the astrophysics layer."""

    H0: float = 70.0                   # km/s/Mpc
    Om0: float = 0.3
    Ode0: float = 0.7


@dataclass
class ReportConfig(_Section):
    """Scientific report generation."""

    formats: List[str] = field(default_factory=lambda: ["text", "json"])
    output_dir: str = "astrovision_output"
    top_candidates: int = 10
    include_embeddings: bool = False
    include_cutouts: bool = False
    title: str = "AstroVision-X Field Analysis"
    observer: str = ""


@dataclass
class AstroVisionConfig(_Section):
    """Root configuration object passed to :class:`~astrovision.engine.Pipeline`."""

    name: str = "astrovision-run"
    random_state: int = 42
    log_level: str = "info"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    multiband: MultiBandConfig = field(default_factory=MultiBandConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    crossmatch: CrossmatchConfig = field(default_factory=CrossmatchConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    transient: TransientConfig = field(default_factory=TransientConfig)
    timeseries: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    lensing: LensingConfig = field(default_factory=LensingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # -- I/O ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "AstroVisionConfig":
        """Load configuration from a ``.json``, ``.yaml`` or ``.yml`` file."""
        if not os.path.exists(path):
            raise ConfigError(f"configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if path.lower().endswith((".yaml", ".yml")):
            yaml = try_import("yaml")
            if yaml is None:
                raise ConfigError("PyYAML is required to read YAML configs")
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be a mapping")
        return cls.from_dict(data)

    def save(self, path: str) -> str:
        """Write the configuration to JSON or YAML, inferred from ``path``."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        data = self.to_dict()
        if path.lower().endswith((".yaml", ".yml")):
            yaml = try_import("yaml")
            if yaml is None:
                raise ConfigError("PyYAML is required to write YAML configs")
            text = yaml.safe_dump(data, sort_keys=False)
        else:
            text = json.dumps(data, indent=2)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def apply_overrides(self, overrides: Optional[List[str]]) -> "AstroVisionConfig":
        """Apply ``key.path=value`` strings, as produced by ``--set``."""
        for item in overrides or []:
            if "=" not in item:
                raise ConfigError(f"override must look like key=value, got '{item}'")
            key, _, raw = item.partition("=")
            self.update(key.strip(), _parse_scalar(raw.strip()))
        return self

    def with_preset(self, preset: str) -> "AstroVisionConfig":
        """Return this config adjusted by a named preset (see :data:`PRESETS`)."""
        if preset not in PRESETS:
            raise ConfigError(f"unknown preset '{preset}'; available: {', '.join(PRESETS)}")
        for key, value in PRESETS[preset].items():
            self.update(key, value)
        return self


def _parse_scalar(text: str) -> Any:
    """Parse a CLI scalar: JSON first, then bare strings."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


#: Named parameter bundles for common observing situations.
PRESETS: Dict[str, Dict[str, Any]] = {
    "deep_field": {
        "detection.threshold_sigma": 2.5,
        "detection.min_area": 8,
        "detection.deblend_contrast": 0.001,
        "morphology.fit_sersic": True,
        "anomaly.contamination": 0.01,
    },
    "wide_survey": {
        "detection.threshold_sigma": 4.0,
        "detection.min_area": 4,
        "segmentation.decompose_galaxies": False,
        "morphology.fit_sersic": False,
    },
    "transient_search": {
        "detection.threshold_sigma": 4.0,
        "transient.detection_sigma": 4.5,
        "timeseries.period_search": True,
        "lensing.enabled": False,
        "morphology.fit_sersic": False,
    },
    "lens_search": {
        "detection.threshold_sigma": 3.0,
        "lensing.enabled": True,
        "lensing.score_threshold": 0.35,
        "segmentation.decompose_galaxies": True,
    },
    "quicklook": {
        "segmentation.enabled": False,
        "morphology.fit_sersic": False,
        "morphology.detect_spiral_arms": False,
        "anomaly.enabled": False,
        "lensing.enabled": False,
    },
}


def default_config() -> AstroVisionConfig:
    """Convenience factory for a fresh default configuration."""
    return AstroVisionConfig()

"""The alert schema this package writes, in ZTF's vocabulary.

ZTF's alert schema is the one the community's tools know: ``objectId``,
``candid``, a ``candidate`` record with ``jd``, ``fid``, ``ra``, ``dec``,
``magpsf``, ``sigmapsf``, ``rb`` and ``drb``, a ``prv_candidates`` array of
earlier detections and upper limits, and three cutouts. Alerts written here
use those names and units, so a broker filter or a plotting script written
for ZTF reads them without translation. It is a *subset*, not the full
schema: ZTF's ``candidate`` has over a hundred fields tied to its own
pipeline (``ssdistnr``, ``sgscore1``, ...) that a general-purpose package
cannot honestly fill. Fields this package cannot measure are absent rather
than filled with placeholders.

Reading goes the other way: the reader decodes whatever schema a file
embeds, so a real ZTF or Rubin alert file is read in full, and
:mod:`packet` picks the fields it understands from either vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = "avx-1.0"

#: ZTF filter ids: 1 g, 2 r, 3 i. Other bands are carried by name in
#: ``candidate.filter`` and get ``fid`` 0.
ZTF_FID = {"g": 1, "r": 2, "i": 3}
FID_TO_BAND = {v: k for k, v in ZTF_FID.items()}

CANDIDATE_FIELDS = [
    {"name": "jd", "type": "double", "doc": "observation Julian date"},
    {"name": "fid", "type": "int", "doc": "ZTF filter id: 1 g, 2 r, 3 i, 0 other"},
    {"name": "filter", "type": "string", "doc": "band name"},
    {"name": "pid", "type": "long", "doc": "processing id of the epoch"},
    {"name": "candid", "type": ["null", "long"], "default": None,
     "doc": "candidate id; null for a non-detection"},
    {"name": "isdiffpos", "type": ["null", "string"], "default": None,
     "doc": "'t' if the difference is positive, 'f' if negative"},
    {"name": "ra", "type": ["null", "double"], "default": None},
    {"name": "dec", "type": ["null", "double"], "default": None},
    {"name": "magpsf", "type": ["null", "float"], "default": None, "doc": "AB magnitude"},
    {"name": "sigmapsf", "type": ["null", "float"], "default": None},
    {"name": "diffmaglim", "type": ["null", "float"], "default": None,
     "doc": "5-sigma limiting magnitude of the difference image"},
    {"name": "fluxpsf", "type": ["null", "float"], "default": None,
     "doc": "difference flux in counts (this package's addition)"},
    {"name": "sigmaflux", "type": ["null", "float"], "default": None},
    {"name": "rb", "type": ["null", "float"], "default": None,
     "doc": "real-bogus score, 1 real"},
    {"name": "drb", "type": ["null", "float"], "default": None,
     "doc": "deep-learning real-bogus score, when a model ran"},
    {"name": "classtar", "type": ["null", "float"], "default": None,
     "doc": "star-galaxy score of the nearest host"},
    {"name": "distnr", "type": ["null", "float"], "default": None,
     "doc": "distance to the nearest reference source, arcsec"},
    {"name": "magnr", "type": ["null", "float"], "default": None,
     "doc": "magnitude of that reference source"},
    {"name": "fwhm", "type": ["null", "float"], "default": None, "doc": "pixels"},
    {"name": "field", "type": ["null", "int"], "default": None},
    {"name": "programid", "type": "int", "default": 1},
    {"name": "nbad", "type": ["null", "int"], "default": None},
]

CUTOUT_SCHEMA: Dict[str, Any] = {
    "type": "record", "name": "cutout", "namespace": "astrovision.alert",
    "fields": [
        {"name": "fileName", "type": "string"},
        {"name": "stampData", "type": "bytes",
         "doc": "gzip-compressed FITS when Astropy is present, else raw "
                "little-endian float32 pixels; the format is in stampFormat"},
        {"name": "stampFormat", "type": "string", "default": "fits.gz"},
        {"name": "width", "type": "int", "default": 0},
        {"name": "height", "type": "int", "default": 0},
    ],
}

ALERT_SCHEMA: Dict[str, Any] = {
    "type": "record", "name": "alert", "namespace": "astrovision.alert",
    "doc": "AstroVision-X alert in ZTF vocabulary (subset). A candidate the "
           "pipeline ranked; nothing here is a confirmed detection.",
    "fields": [
        {"name": "schemavsn", "type": "string", "default": SCHEMA_VERSION},
        {"name": "publisher", "type": "string", "default": "astrovision-x"},
        {"name": "objectId", "type": "string"},
        {"name": "candid", "type": "long"},
        {"name": "candidate", "type": {"type": "record", "name": "candidate",
                                       "namespace": "astrovision.alert",
                                       "fields": CANDIDATE_FIELDS}},
        {"name": "prv_candidates", "type": ["null", {"type": "array", "items": "candidate"}],
         "default": None},
        {"name": "cutoutScience", "type": ["null", CUTOUT_SCHEMA], "default": None},
        {"name": "cutoutTemplate", "type": ["null", "cutout"], "default": None},
        {"name": "cutoutDifference", "type": ["null", "cutout"], "default": None},
        {"name": "classification", "type": ["null", "string"], "default": None,
         "doc": "the pipeline's tentative class (supernova, variable_star, mover, ...)"},
        {"name": "verdict", "type": ["null", "string"], "default": None,
         "doc": "the pipeline's recommendation: not_interesting ... high_priority"},
        {"name": "human_verdict", "type": ["null", "string"], "default": None,
         "doc": "a reviewer's decision, when one has been recorded, as 'label by name'"},
        {"name": "provenance", "type": ["null", {"type": "map", "values": "string"}],
         "default": None, "doc": "reproducibility key, config hash, code revision"},
    ],
}

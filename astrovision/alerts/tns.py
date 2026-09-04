"""A Transient Name Server report, drafted and never sent.

The TNS bulk-report format is a JSON document with an ``at_report`` block
per object: position with its error, the reporting group, the discovery
data source, the reporter's name, the discovery date, the object type, the
photometry that constitutes the discovery, and the last non-detection
before it. This module builds that document from an :class:`AlertPacket`
and writes it to a file.

It does not submit it. There is no HTTP client here and no API key field,
on purpose: a report to the TNS is a claim to the community that a person
stands behind, so the person makes it. The draft carries ``_draft: true``,
a ``_not_submitted`` note, and the reporter's name that was required to
build it -- an anonymous draft is refused for the same reason an anonymous
verdict is.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .packet import AlertPacket, Detection

#: TNS "AT type" ids: 1 is "Other", the honest default for an unclassified
#: candidate. A person picks a specific type when the evidence supports it.
AT_TYPE_OTHER = 1
AT_TYPES = {"other": 1, "supernova": 3, "nova": 4, "agn": 5, "tde": 120, "fbot": 130}

#: TNS filter ids for the bands this package names; others are reported as
#: "Other" (id 0) with the band name in the comment.
FILTER_IDS = {"u": 20, "g": 21, "r": 22, "i": 23, "z": 24, "clear": 1, "other": 0}


def _mjd_to_datetime(mjd: float) -> str:
    """``YYYY-MM-DD HH:MM:SS`` UTC from an MJD (TNS wants this string)."""
    seconds = (float(mjd) - 40587.0) * 86400.0            # 40587 = 1970-01-01
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(seconds))


def _photometry(detection: Detection, instrument_id: int, observer: str) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "obsdate": _mjd_to_datetime(detection.mjd),
        "flux": detection.mag, "flux_error": detection.mag_err,
        "limiting_flux": detection.limiting_mag,
        "flux_units": "1",                                  # 1 = ABMag in TNS's table
        "filter_value": FILTER_IDS.get(detection.band, FILTER_IDS["other"]),
        "instrument_value": int(instrument_id),
        "exptime": None, "observer": observer,
        "comments": f"band {detection.band}",
    }
    return entry


def draft_tns_report(packet: AlertPacket, reporter: str, reporting_group_id: int = 0,
                     data_source_id: int = 0, instrument_id: int = 0,
                     at_type: str = "other", internal_name: Optional[str] = None,
                     remarks: str = "", position_error_arcsec: float = 0.5,
                     host_name: Optional[str] = None,
                     host_redshift: Optional[float] = None) -> Dict[str, Any]:
    """A TNS bulk-report document for one packet, marked as a draft.

    ``reporter`` is required. ``reporting_group_id``, ``data_source_id`` and
    ``instrument_id`` are the TNS's own integer ids for the group, survey and
    instrument; zero means "fill this in", and the draft says so.
    """
    if not str(reporter).strip():
        raise ValueError("a TNS report needs a reporter's name; nothing is drafted anonymously")
    if not (packet.ra == packet.ra and packet.dec == packet.dec):       # NaN check
        raise ValueError("the packet has no sky position; a report cannot be drafted")
    at_type_id = AT_TYPES.get(str(at_type).lower(), AT_TYPE_OTHER)
    todo: List[str] = []
    for name, value in (("reporting_group_id", reporting_group_id),
                        ("discovery_data_source_id", data_source_id),
                        ("instrument_id", instrument_id)):
        if int(value) == 0:
            todo.append(f"{name} is 0: set your TNS {name.replace('_', ' ')}")

    detections = packet.detections()
    discovery = Detection(mjd=packet.mjd, band=packet.band, mag=packet.mag,
                          mag_err=packet.mag_err, flux=packet.flux, flux_err=packet.flux_err,
                          limiting_mag=packet.limiting_mag, ra=packet.ra, dec=packet.dec)
    if discovery.mag is None and not detections:
        todo.append("no magnitude on the discovery epoch or in the history")
    photometry = {"photometry_group": {}}
    ordered = sorted(detections + [discovery], key=lambda d: d.mjd)
    for index, detection in enumerate(ordered):
        photometry["photometry_group"][str(index)] = _photometry(detection, instrument_id,
                                                                 reporter)
    limit = packet.last_non_detection_before()
    if limit is not None:
        non_detection: Dict[str, Any] = {
            "obsdate": _mjd_to_datetime(limit.mjd), "limiting_flux": limit.limiting_mag,
            "flux_units": "1", "filter_value": FILTER_IDS.get(limit.band, 0),
            "instrument_value": int(instrument_id), "exptime": None,
            "observer": reporter, "comments": f"band {limit.band}",
        }
    else:
        non_detection = {"archiveid": "0", "archival_remarks": "no earlier non-detection "
                         "in the alert history; add archival limits by hand"}
        todo.append("no non-detection before discovery in the history")

    remarks_parts = [remarks] if remarks else []
    if packet.classification:
        remarks_parts.append(f"pipeline class {packet.classification} (tentative)")
    if packet.real_bogus is not None:
        remarks_parts.append(f"real-bogus {packet.real_bogus:.2f}")
    if packet.human_verdict:
        remarks_parts.append(f"vetted: {packet.human_verdict}")
    else:
        todo.append("no human verdict recorded for this candidate; vet it before reporting")
    if packet.provenance:
        remarks_parts.append("provenance " + ", ".join(f"{k}={v}" for k, v in packet.provenance.items()))

    report = {
        "at_report": {
            "0": {
                "ra": {"value": float(packet.ra), "error": float(position_error_arcsec),
                       "units": "arcsec"},
                "dec": {"value": float(packet.dec), "error": float(position_error_arcsec),
                        "units": "arcsec"},
                "reporting_group_id": int(reporting_group_id),
                "discovery_data_source_id": int(data_source_id),
                "reporter": str(reporter).strip(),
                "discovery_datetime": _mjd_to_datetime(ordered[0].mjd),
                "at_type": at_type_id,
                "host_name": host_name or "",
                "host_redshift": host_redshift,
                "transient_redshift": None,
                "internal_name": internal_name or packet.object_id,
                "remarks": "; ".join(remarks_parts),
                "photometry": photometry,
                "non_detection": non_detection,
            }
        },
        "_draft": True,
        "_not_submitted": "This document was drafted by AstroVision-X and has not been sent. "
                          "A person reviews it, fills in the ids marked below, and submits it "
                          "through the TNS bulk-report interface under their own credentials.",
        "_todo": todo,
        "_source": {"object_id": packet.object_id, "candid": packet.candid,
                    "format": packet.source_format, "publisher": packet.publisher},
        "_generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return report


def write_tns_draft(report: Dict[str, Any], path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return path


__all__ = ["draft_tns_report", "write_tns_draft", "AT_TYPES", "FILTER_IDS"]

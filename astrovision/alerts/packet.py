"""One alert, whatever vocabulary it arrived in.

An :class:`AlertPacket` is the package's own view of an alert: a position,
an epoch, a band, a brightness, a real-bogus score, the earlier detections
and limits, the cutouts, and what the pipeline and any reviewer concluded.
It is built from this package's transient candidates, written in the ZTF
vocabulary of :mod:`schema`, and read back from that vocabulary, from real
ZTF alerts, and from Rubin-style ``diaSource`` alerts, which spell the same
things differently (``midpointMjdTai`` for the epoch, ``psfFlux`` in
nanojansky for the brightness, ``band`` for the filter).

Cutouts travel as gzip-compressed FITS, the way ZTF sends them; the FITS
is written by the package's own writer so no Astropy is needed at either
end.
"""

from __future__ import annotations

import math

import gzip
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.logging import get_logger
from ..io.fits import read_fits, write_fits
from .schema import FID_TO_BAND, SCHEMA_VERSION, ZTF_FID

log = get_logger("alerts.packet")

MJD_OFFSET = 2400000.5


def mjd_to_jd(mjd: float) -> float:
    return float(mjd) + MJD_OFFSET


def jd_to_mjd(jd: float) -> float:
    return float(jd) - MJD_OFFSET


def flux_to_mag(flux: float, zero_point: float) -> Optional[float]:
    if flux is None or not np.isfinite(flux) or flux <= 0:
        return None
    return float(zero_point - 2.5 * np.log10(flux))


def flux_err_to_mag_err(flux: float, err: float) -> Optional[float]:
    if flux is None or err is None or not np.isfinite(flux) or flux <= 0 or not np.isfinite(err):
        return None
    return float(1.0857 * err / flux)


def nanojansky_to_ab(flux_njy: float) -> Optional[float]:
    """AB magnitude from a flux in nanojansky (Rubin's unit)."""
    if flux_njy is None or not np.isfinite(flux_njy) or flux_njy <= 0:
        return None
    return float(31.4 - 2.5 * np.log10(flux_njy))


# -- cutouts -------------------------------------------------------------------
def encode_stamp(stamp: np.ndarray, name: str = "cutout.fits.gz") -> Dict[str, Any]:
    """A cutout record: gzip-compressed FITS bytes, as ZTF sends them."""
    array = np.asarray(stamp, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stamp.fits")
        write_fits(path, array, {"STAMP": "astrovision-x cutout"})
        with open(path, "rb") as handle:
            raw = handle.read()
    return {"fileName": name, "stampData": gzip.compress(raw, 6), "stampFormat": "fits.gz",
            "width": int(array.shape[1]), "height": int(array.shape[0])}


def decode_stamp(record: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Pixels from a cutout record; None when there is none."""
    if not record or not record.get("stampData"):
        return None
    data = bytes(record["stampData"])
    fmt = record.get("stampFormat", "fits.gz")
    if fmt == "f4le":
        width, height = int(record.get("width", 0)), int(record.get("height", 0))
        return np.frombuffer(data, dtype="<f4").reshape(height, width).astype(float)
    try:
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stamp.fits")
            with open(path, "wb") as handle:
                handle.write(data)
            pixels, _ = read_fits(path)
    except Exception as exc:
        # A stamp that is not what it claims must not take the packet down
        # with it: the light curve and the scores are still worth reading.
        log.debug("cutout %r could not be decoded (%s)", record.get("fileName"), exc)
        return None
    return np.asarray(pixels, dtype=float)


# -- the packet ----------------------------------------------------------------
@dataclass
class Detection:
    """One epoch of an object: a measurement or an upper limit."""

    mjd: float
    band: str
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    flux: Optional[float] = None
    flux_err: Optional[float] = None
    limiting_mag: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    candid: Optional[int] = None
    is_positive: Optional[bool] = None
    #: True for a forced-photometry point (ZTF ``fp_hists``): a flux
    #: measured at the position whether or not anything was detected there.
    forced: bool = False

    @property
    def is_detection(self) -> bool:
        return self.mag is not None or (self.flux is not None and self.flux_err is not None
                                        and self.flux_err > 0 and self.flux / self.flux_err >= 5)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AlertPacket:
    """What this package knows about one alert, in one vocabulary."""

    object_id: str
    candid: int
    ra: float
    dec: float
    mjd: float
    band: str
    mag: Optional[float] = None
    mag_err: Optional[float] = None
    flux: Optional[float] = None
    flux_err: Optional[float] = None
    limiting_mag: Optional[float] = None
    real_bogus: Optional[float] = None
    deep_real_bogus: Optional[float] = None
    is_positive: Optional[bool] = None
    fwhm: Optional[float] = None
    host_distance_arcsec: Optional[float] = None
    host_mag: Optional[float] = None
    host_star_score: Optional[float] = None
    classification: Optional[str] = None
    verdict: Optional[str] = None
    human_verdict: Optional[str] = None
    history: List[Detection] = field(default_factory=list)
    cutout_science: Optional[np.ndarray] = None
    cutout_template: Optional[np.ndarray] = None
    cutout_difference: Optional[np.ndarray] = None
    provenance: Dict[str, str] = field(default_factory=dict)
    publisher: str = "astrovision-x"
    schema_version: str = SCHEMA_VERSION
    source_format: str = "astrovision"        # astrovision | ztf | rubin
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_transient(cls, candidate, series=None, image=None, zero_point: float = 25.0,
                       provenance: Optional[Dict[str, Any]] = None,
                       human_verdict: Optional[str] = None, stamp_size: int = 63,
                       object_prefix: str = "AVX") -> "AlertPacket":
        """Build a packet from a :class:`TransientCandidate`.

        ``series`` supplies epochs and bands (and the reference image for the
        template cutout); ``image`` alone supplies a science cutout. The
        light curve on the candidate becomes ``history``.
        """
        times = getattr(series, "times", None)
        epoch = int(getattr(candidate, "epoch_index", 0) or 0)
        mjd = None
        if times is not None and len(times) > epoch:
            mjd = float(times[epoch])
        if mjd is None or not np.isfinite(mjd):
            mjd = float(candidate.meta.get("mjd", getattr(image, "mjd", 0.0) or 0.0))
        band = "clear"
        images = getattr(series, "images", None)
        if images is not None and len(images) > epoch and getattr(images[epoch], "band", None):
            band = str(images[epoch].band)
        elif getattr(image, "band", None):
            band = str(image.band)
        ra = float(candidate.ra) if candidate.ra is not None else float("nan")
        dec = float(candidate.dec) if candidate.dec is not None else float("nan")
        if (not np.isfinite(ra)) and image is not None and getattr(image, "wcs", None) is not None:
            world = image.wcs.pixel_to_world(candidate.x, candidate.y)
            ra, dec = float(np.asarray(world[0]).ravel()[0]), float(np.asarray(world[1]).ravel()[0])
        delta = float(candidate.delta_flux)
        candid = int(candidate.meta.get("candid", 0) or (int(abs(mjd) * 1e5) * 1000 + int(candidate.id)))
        object_id = f"{object_prefix}{int(candidate.id):06d}" if not candidate.meta.get("object_id") \
            else str(candidate.meta["object_id"])
        packet = cls(
            object_id=object_id, candid=candid, ra=ra, dec=dec, mjd=mjd, band=band,
            mag=flux_to_mag(abs(delta), zero_point) if delta != 0 else None,
            flux=delta, is_positive=delta >= 0,
            real_bogus=float(candidate.real_bogus) if np.isfinite(candidate.real_bogus) else None,
            classification=str(candidate.classification) if candidate.classification else None,
            verdict=candidate.verdict.value if hasattr(candidate.verdict, "value") else str(candidate.verdict),
            human_verdict=human_verdict,
            host_distance_arcsec=(float(candidate.host_offset)
                                  if np.isfinite(candidate.host_offset) else None),
            provenance={k: str(v) for k, v in (provenance or {}).items()},
            extra={"significance": float(candidate.significance),
                   "confidence": float(candidate.confidence),
                   "flags": list(candidate.flags), "source_id": candidate.id})
        if candidate.delta_magnitude is not None and np.isfinite(candidate.delta_magnitude):
            packet.extra["delta_magnitude"] = float(candidate.delta_magnitude)
        curve = getattr(candidate, "light_curve", None)
        if curve is not None:
            errors = curve.errors if curve.errors is not None else np.full(curve.times.size, np.nan)
            for t, f, e in zip(curve.times, curve.fluxes, errors):
                # An epoch below 5 sigma is an upper limit, not a magnitude:
                # reporting a 2-sigma flux as a detection would put a spurious
                # pre-discovery point into every TNS draft built from this.
                significant = not np.isfinite(e) or e <= 0 or (f / e) >= 5.0
                packet.history.append(Detection(
                    mjd=float(t), band=str(curve.band or band),
                    flux=float(f), flux_err=None if not np.isfinite(e) else float(e),
                    mag=flux_to_mag(f, zero_point) if significant else None,
                    mag_err=(flux_err_to_mag_err(f, e) if significant and np.isfinite(e)
                             else None),
                    limiting_mag=(None if significant or not np.isfinite(e) or e <= 0
                                  else flux_to_mag(5.0 * float(e), zero_point))))
        science = image
        if science is None and images is not None and len(images) > epoch:
            science = images[epoch]
        if science is not None:
            packet.cutout_science = science.cutout(candidate.x, candidate.y, stamp_size)
        reference = getattr(series, "reference", None)
        if reference is not None:
            packet.cutout_template = reference.cutout(candidate.x, candidate.y, stamp_size)
        difference = candidate.meta.get("difference_stamp")
        if difference is not None:
            packet.cutout_difference = np.asarray(difference, dtype=float)
        return packet

    # -- ZTF vocabulary --------------------------------------------------------
    def _candidate_record(self) -> Dict[str, Any]:
        return {
            "jd": mjd_to_jd(self.mjd), "fid": ZTF_FID.get(self.band, 0), "filter": self.band,
            "pid": int(abs(self.mjd) * 1e5), "candid": int(self.candid),
            "isdiffpos": None if self.is_positive is None else ("t" if self.is_positive else "f"),
            "ra": self.ra, "dec": self.dec, "magpsf": self.mag, "sigmapsf": self.mag_err,
            "diffmaglim": self.limiting_mag, "fluxpsf": self.flux, "sigmaflux": self.flux_err,
            "rb": self.real_bogus, "drb": self.deep_real_bogus,
            "classtar": self.host_star_score, "distnr": self.host_distance_arcsec,
            "magnr": self.host_mag, "fwhm": self.fwhm, "field": None, "programid": 1,
            "nbad": None,
        }

    def to_record(self) -> Dict[str, Any]:
        """The alert as a record of :data:`schema.ALERT_SCHEMA`."""
        prv = []
        for d in self.history:
            prv.append({
                "jd": mjd_to_jd(d.mjd), "fid": ZTF_FID.get(d.band, 0), "filter": d.band,
                "pid": int(abs(d.mjd) * 1e5), "candid": d.candid,
                "isdiffpos": None if d.is_positive is None else ("t" if d.is_positive else "f"),
                "ra": d.ra, "dec": d.dec, "magpsf": d.mag, "sigmapsf": d.mag_err,
                "diffmaglim": d.limiting_mag, "fluxpsf": d.flux, "sigmaflux": d.flux_err,
                "rb": None, "drb": None, "classtar": None, "distnr": None, "magnr": None,
                "fwhm": None, "field": None, "programid": 1, "nbad": None})
        return {
            "schemavsn": self.schema_version, "publisher": self.publisher,
            "objectId": self.object_id, "candid": int(self.candid),
            "candidate": self._candidate_record(),
            "prv_candidates": prv or None,
            "cutoutScience": None if self.cutout_science is None
            else encode_stamp(self.cutout_science, "science.fits.gz"),
            "cutoutTemplate": None if self.cutout_template is None
            else encode_stamp(self.cutout_template, "template.fits.gz"),
            "cutoutDifference": None if self.cutout_difference is None
            else encode_stamp(self.cutout_difference, "difference.fits.gz"),
            "classification": self.classification, "verdict": self.verdict,
            "human_verdict": self.human_verdict,
            "provenance": dict(self.provenance) or None,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "AlertPacket":
        """Read a packet from a ZTF-vocabulary record (ours or ZTF's own)
        or a Rubin-style ``diaSource`` record."""
        if "diaSource" in record:
            return cls._from_rubin(record)
        cand = record.get("candidate") or {}
        band = cand.get("filter") or FID_TO_BAND.get(int(cand.get("fid", 0) or 0), "clear")
        prov = record.get("provenance") or {}
        packet = cls(
            object_id=str(record.get("objectId", "")), candid=int(record.get("candid", 0) or 0),
            ra=float(cand.get("ra") if cand.get("ra") is not None else np.nan),
            dec=float(cand.get("dec") if cand.get("dec") is not None else np.nan),
            mjd=jd_to_mjd(float(cand.get("jd", MJD_OFFSET))), band=str(band),
            mag=cand.get("magpsf"), mag_err=cand.get("sigmapsf"),
            flux=cand.get("fluxpsf"), flux_err=cand.get("sigmaflux"),
            limiting_mag=cand.get("diffmaglim"), real_bogus=cand.get("rb"),
            deep_real_bogus=cand.get("drb"),
            is_positive=None if cand.get("isdiffpos") is None
            else str(cand.get("isdiffpos")).lower() in ("t", "1", "true"),
            fwhm=cand.get("fwhm"), host_distance_arcsec=cand.get("distnr"),
            host_mag=cand.get("magnr"), host_star_score=cand.get("classtar"),
            classification=record.get("classification"), verdict=record.get("verdict"),
            human_verdict=record.get("human_verdict"),
            provenance={k: str(v) for k, v in prov.items()},
            publisher=str(record.get("publisher", "")),
            schema_version=str(record.get("schemavsn", "")),
            source_format="astrovision" if str(record.get("publisher", "")).startswith("astrovision")
            else "ztf")
        for prv in record.get("prv_candidates") or []:
            if prv is None:
                continue
            pband = prv.get("filter") or FID_TO_BAND.get(int(prv.get("fid", 0) or 0), "clear")
            packet.history.append(Detection(
                mjd=jd_to_mjd(float(prv.get("jd", MJD_OFFSET))), band=str(pband),
                mag=prv.get("magpsf"), mag_err=prv.get("sigmapsf"),
                flux=prv.get("fluxpsf"), flux_err=prv.get("sigmaflux"),
                limiting_mag=prv.get("diffmaglim"), ra=prv.get("ra"), dec=prv.get("dec"),
                candid=prv.get("candid"),
                is_positive=None if prv.get("isdiffpos") is None
                else str(prv.get("isdiffpos")).lower() in ("t", "1", "true")))
        # ZTF's forced photometry (schema 4.x): difference fluxes in DN at
        # the object's position for the last month of epochs.  Kept as flux
        # points with the epoch's zero point applied where it is given, so a
        # light curve can be drawn from a single packet.
        for forced in record.get("fp_hists") or []:
            if forced is None or forced.get("forcediffimflux") is None:
                continue
            flux = float(forced["forcediffimflux"])
            err = forced.get("forcediffimfluxunc")
            err = None if err is None else float(err)
            zero_point = forced.get("magzpsci")
            mag = mag_err = None
            if zero_point is not None and flux > 0 and err and flux / err >= 3.0:
                mag = float(zero_point) - 2.5 * math.log10(flux)
                mag_err = flux_err_to_mag_err(flux, err)
            fband = FID_TO_BAND.get(int(forced.get("fid", 0) or 0), "clear")
            packet.history.append(Detection(
                mjd=jd_to_mjd(float(forced.get("jd", MJD_OFFSET))), band=str(fband),
                mag=mag, mag_err=mag_err, flux=flux, flux_err=err,
                limiting_mag=forced.get("diffmaglim"), ra=forced.get("ranr"),
                dec=forced.get("decnr"), is_positive=flux > 0, forced=True))
        packet.cutout_science = decode_stamp(record.get("cutoutScience"))
        packet.cutout_template = decode_stamp(record.get("cutoutTemplate"))
        packet.cutout_difference = decode_stamp(record.get("cutoutDifference"))
        return packet

    @classmethod
    def _from_rubin(cls, record: Dict[str, Any]) -> "AlertPacket":
        src = record.get("diaSource") or {}
        flux, err = src.get("psfFlux"), src.get("psfFluxErr")
        packet = cls(
            object_id=str(src.get("diaObjectId", record.get("alertId", ""))),
            candid=int(src.get("diaSourceId", record.get("alertId", 0)) or 0),
            ra=float(src.get("ra", np.nan)), dec=float(src.get("dec", src.get("decl", np.nan))),
            mjd=float(src.get("midpointMjdTai", src.get("midPointTai", 0.0)) or 0.0),
            band=str(src.get("band", src.get("filterName", "clear"))),
            mag=nanojansky_to_ab(flux), flux=flux, flux_err=err,
            mag_err=(flux_err_to_mag_err(flux, err) if flux is not None and err is not None
                     else None),
            real_bogus=src.get("reliability"), fwhm=None,
            is_positive=None if flux is None else flux >= 0,
            publisher="rubin", schema_version=str(record.get("schemavsn", "")),
            source_format="rubin",
            extra={"snr": src.get("snr"), "diaObjectId": src.get("diaObjectId")})
        for prv in record.get("prvDiaSources") or []:
            if prv is None:
                continue
            f, e = prv.get("psfFlux"), prv.get("psfFluxErr")
            packet.history.append(Detection(
                mjd=float(prv.get("midpointMjdTai", prv.get("midPointTai", 0.0)) or 0.0),
                band=str(prv.get("band", prv.get("filterName", "clear"))),
                mag=nanojansky_to_ab(f), flux=f, flux_err=e,
                mag_err=flux_err_to_mag_err(f, e) if f is not None and e is not None else None,
                ra=prv.get("ra"), dec=prv.get("dec", prv.get("decl")),
                candid=prv.get("diaSourceId")))
        for limit in record.get("prvDiaForcedSources") or []:
            if limit is None:
                continue
            f, e = limit.get("psfFlux"), limit.get("psfFluxErr")
            packet.history.append(Detection(
                mjd=float(limit.get("midpointMjdTai", limit.get("midPointTai", 0.0)) or 0.0),
                band=str(limit.get("band", limit.get("filterName", "clear"))),
                flux=f, flux_err=e))
        # ZTF's forced photometry (schema 4.x): difference fluxes in DN at
        # the object's position for the last month of epochs.  Kept as flux
        # points with the epoch's zero point applied where it is given, so a
        # light curve can be drawn from a single packet.
        for forced in record.get("fp_hists") or []:
            if forced is None or forced.get("forcediffimflux") is None:
                continue
            flux = float(forced["forcediffimflux"])
            err = forced.get("forcediffimfluxunc")
            err = None if err is None else float(err)
            zero_point = forced.get("magzpsci")
            mag = mag_err = None
            if zero_point is not None and flux > 0 and err and flux / err >= 3.0:
                mag = float(zero_point) - 2.5 * math.log10(flux)
                mag_err = flux_err_to_mag_err(flux, err)
            fband = FID_TO_BAND.get(int(forced.get("fid", 0) or 0), "clear")
            packet.history.append(Detection(
                mjd=jd_to_mjd(float(forced.get("jd", MJD_OFFSET))), band=str(fband),
                mag=mag, mag_err=mag_err, flux=flux, flux_err=err,
                limiting_mag=forced.get("diffmaglim"), ra=forced.get("ranr"),
                dec=forced.get("decnr"), is_positive=flux > 0, forced=True))
        packet.cutout_science = decode_stamp(record.get("cutoutScience"))
        packet.cutout_template = decode_stamp(record.get("cutoutTemplate"))
        packet.cutout_difference = decode_stamp(record.get("cutoutDifference"))
        return packet

    # -- summaries -------------------------------------------------------------
    def detections(self) -> List[Detection]:
        return [d for d in self.history if d.is_detection]

    def last_non_detection_before(self, mjd: Optional[float] = None) -> Optional[Detection]:
        """The latest earlier epoch with a limit and no detection."""
        cutoff = self.mjd if mjd is None else mjd
        limits = [d for d in self.history
                  if d.mjd < cutoff and not d.is_detection and d.limiting_mag is not None]
        return max(limits, key=lambda d: d.mjd) if limits else None

    def summary(self) -> str:
        mag = "—" if self.mag is None else f"{self.mag:.2f}"
        rb = "—" if self.real_bogus is None else f"{self.real_bogus:.2f}"
        return (f"{self.object_id} candid {self.candid} ({self.source_format}) "
                f"ra {self.ra:.5f} dec {self.dec:.5f} mjd {self.mjd:.4f} {self.band} "
                f"mag {mag} rb {rb} history {len(self.history)} "
                f"class {self.classification or '—'} verdict {self.verdict or '—'}")

    def to_dict(self) -> Dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items()
                   if not k.startswith("cutout_")}
        payload["history"] = [d.to_dict() for d in self.history]
        payload["has_cutouts"] = {k: getattr(self, f"cutout_{k}") is not None
                                  for k in ("science", "template", "difference")}
        return payload


def packets_from_analysis(analysis, series=None, image=None, zero_point: float = 25.0,
                          verdict_log=None, include_bogus: bool = False) -> List[AlertPacket]:
    """One packet per transient candidate of a :class:`FieldAnalysis`.

    A reviewer's decision from ``verdict_log`` (a
    :class:`~astrovision.ml.active.VerdictLog`) is attached as
    ``human_verdict`` when one exists for the candidate.
    """
    provenance = {}
    prov = getattr(analysis, "provenance", {}) or {}
    manifest = prov.get("manifest") or {}
    for key, value in (("reproducibility_key", prov.get("reproducibility_key")),
                       ("config_hash", manifest.get("config_hash")),
                       ("revision", (manifest.get("git") or {}).get("revision")),
                       ("package_version", manifest.get("package_version"))):
        if value:
            provenance[key] = str(value)
    latest = verdict_log.latest() if verdict_log is not None else {}
    packets = []
    for candidate in getattr(analysis, "transients", []):
        if not include_bogus and "bogus" in candidate.flags:
            continue
        human = None
        for key in (candidate.host_source_id, -int(candidate.id)):
            if key is not None and int(key) in latest:
                record = latest[int(key)]
                if not record.kind or record.kind == "transient":
                    human = f"{record.label} by {record.reviewer}"
                    break
        packets.append(AlertPacket.from_transient(candidate, series=series, image=image,
                                                  zero_point=zero_point, provenance=provenance,
                                                  human_verdict=human))
    return packets


__all__ = ["AlertPacket", "Detection", "packets_from_analysis", "encode_stamp", "decode_stamp",
           "mjd_to_jd", "jd_to_mjd", "flux_to_mag", "nanojansky_to_ab"]

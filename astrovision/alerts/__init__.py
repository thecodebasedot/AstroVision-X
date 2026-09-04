"""Alerts in the community's formats: Avro packets in, Avro packets out,
TNS reports drafted for a person to send."""

from .avro import read_avro, schema_of, write_avro
from .packet import AlertPacket, Detection, decode_stamp, encode_stamp, packets_from_analysis
from .schema import ALERT_SCHEMA, SCHEMA_VERSION
from .tns import draft_tns_report, write_tns_draft


def write_alerts(path: str, packets, codec: str = "deflate") -> int:
    """Write packets as an Avro alert file in the ZTF vocabulary."""
    return write_avro(path, ALERT_SCHEMA, [p.to_record() for p in packets], codec=codec)


def read_alerts(path: str):
    """``(schema, packets)`` from any alert file this package understands."""
    schema, records = read_avro(path)
    return schema, [AlertPacket.from_record(r) for r in records]


__all__ = ["AlertPacket", "Detection", "ALERT_SCHEMA", "SCHEMA_VERSION", "write_alerts",
           "read_alerts", "read_avro", "write_avro", "schema_of", "packets_from_analysis",
           "draft_tns_report", "write_tns_draft", "encode_stamp", "decode_stamp"]

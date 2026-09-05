"""Avro, in the standard library.

Alert brokers speak Avro: ZTF has since 2018, Rubin will. Reading one alert
should not need a compiled dependency, so this is a schema-driven Avro
binary codec and object-container reader/writer in plain Python -- the
same choice the package made for FITS. The Avro specification is short and
this follows it: zig-zag varints for ints and longs, little-endian IEEE
floats, length-prefixed bytes and strings, records as their fields in
order, unions as an index then the value, arrays and maps as counted
blocks ending in a zero. Containers are the ``Obj\\x01`` magic, a metadata
map holding the schema and the codec, a 16-byte sync marker, and blocks of
``(count, size, data, sync)`` compressed with ``null`` or ``deflate``.

When ``fastavro`` is installed it is used for reading and writing, and the
test suite checks that every file this module writes is read back
identically by fastavro and vice versa. When it is not, this module is the
implementation, and the same tests run against itself.
"""

from __future__ import annotations

import io
import json
import os
import struct
import zlib
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Tuple

from ..core.backend import try_import

MAGIC = b"Obj\x01"
_PRIMITIVES = {"null", "boolean", "int", "long", "float", "double", "bytes", "string"}


# -- schema handling -----------------------------------------------------------
def _named_types(schema: Any, registry: Dict[str, Any]) -> None:
    """Collect named records/enums/fixed so later references resolve."""
    if isinstance(schema, dict):
        kind = schema.get("type")
        if kind in ("record", "enum", "fixed") and "name" in schema:
            registry[schema["name"]] = schema
            if "namespace" in schema:
                registry[f"{schema['namespace']}.{schema['name']}"] = schema
        if kind == "record":
            for fld in schema.get("fields", []):
                _named_types(fld.get("type"), registry)
        elif kind == "array":
            _named_types(schema.get("items"), registry)
        elif kind == "map":
            _named_types(schema.get("values"), registry)
    elif isinstance(schema, list):
        for branch in schema:
            _named_types(branch, registry)


def _resolve(schema: Any, registry: Dict[str, Any]) -> Any:
    if isinstance(schema, str) and schema not in _PRIMITIVES:
        if schema not in registry:
            raise ValueError(f"unknown Avro type {schema!r}")
        return registry[schema]
    return schema


def _type_of(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    if isinstance(schema, list):
        return "union"
    return schema["type"]


# -- binary encoding -----------------------------------------------------------
def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 63)


def _write_long(out: io.BytesIO, value: int) -> None:
    n = _zigzag(int(value)) & 0xFFFFFFFFFFFFFFFF
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.write(bytes((byte | 0x80,)))
        else:
            out.write(bytes((byte,)))
            return


def _read_long(data: bytes, pos: int) -> Tuple[int, int]:
    shift, result = 0, 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return (result >> 1) ^ -(result & 1), pos


def _matches(value: Any, schema: Any, registry: Dict[str, Any]) -> bool:
    """Which union branch a Python value belongs to."""
    schema = _resolve(schema, registry)
    kind = _type_of(schema)
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind in ("int", "long"):
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in ("float", "double"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind in ("bytes", "fixed"):
        return isinstance(value, (bytes, bytearray))
    if kind == "array":
        return isinstance(value, (list, tuple))
    if kind == "map":
        return isinstance(value, dict)
    if kind == "record":
        return isinstance(value, dict)
    if kind == "enum":
        return isinstance(value, str) and value in schema.get("symbols", [])
    return False


def encode(value: Any, schema: Any, registry: Dict[str, Any], out: io.BytesIO) -> None:
    schema = _resolve(schema, registry)
    kind = _type_of(schema)
    if kind == "null":
        return
    if kind == "boolean":
        out.write(b"\x01" if value else b"\x00")
    elif kind in ("int", "long"):
        _write_long(out, int(value))
    elif kind == "float":
        out.write(struct.pack("<f", float(value)))
    elif kind == "double":
        out.write(struct.pack("<d", float(value)))
    elif kind == "bytes":
        data = bytes(value)
        _write_long(out, len(data))
        out.write(data)
    elif kind == "string":
        data = str(value).encode("utf-8")
        _write_long(out, len(data))
        out.write(data)
    elif kind == "fixed":
        data = bytes(value)
        if len(data) != int(schema["size"]):
            raise ValueError(f"fixed {schema.get('name')} needs {schema['size']} bytes")
        out.write(data)
    elif kind == "enum":
        out.write(b"")
        _write_long(out, schema["symbols"].index(value))
    elif kind == "union":
        for index, branch in enumerate(schema):
            if _matches(value, branch, registry):
                _write_long(out, index)
                encode(value, branch, registry, out)
                return
        raise ValueError(f"value {value!r} matches no branch of union {schema}")
    elif kind == "array":
        items = list(value)
        if items:
            _write_long(out, len(items))
            for item in items:
                encode(item, schema["items"], registry, out)
        _write_long(out, 0)
    elif kind == "map":
        entries = dict(value)
        if entries:
            _write_long(out, len(entries))
            for key, item in entries.items():
                encode(str(key), "string", registry, out)
                encode(item, schema["values"], registry, out)
        _write_long(out, 0)
    elif kind == "record":
        for fld in schema["fields"]:
            name = fld["name"]
            if name in value:
                item = value[name]
            elif "default" in fld:
                item = fld["default"]
            else:
                raise ValueError(f"record {schema.get('name')} is missing field {name!r}")
            encode(item, fld["type"], registry, out)
    else:
        raise ValueError(f"unsupported Avro type {kind!r}")


def decode(data: bytes, pos: int, schema: Any, registry: Dict[str, Any]) -> Tuple[Any, int]:
    schema = _resolve(schema, registry)
    kind = _type_of(schema)
    if kind == "null":
        return None, pos
    if kind == "boolean":
        return data[pos] != 0, pos + 1
    if kind in ("int", "long"):
        return _read_long(data, pos)
    if kind == "float":
        return struct.unpack("<f", data[pos:pos + 4])[0], pos + 4
    if kind == "double":
        return struct.unpack("<d", data[pos:pos + 8])[0], pos + 8
    if kind == "bytes":
        length, pos = _read_long(data, pos)
        return bytes(data[pos:pos + length]), pos + length
    if kind == "string":
        length, pos = _read_long(data, pos)
        return data[pos:pos + length].decode("utf-8"), pos + length
    if kind == "fixed":
        size = int(schema["size"])
        return bytes(data[pos:pos + size]), pos + size
    if kind == "enum":
        index, pos = _read_long(data, pos)
        return schema["symbols"][index], pos
    if kind == "union":
        index, pos = _read_long(data, pos)
        return decode(data, pos, schema[index], registry)
    if kind == "array":
        items: List[Any] = []
        while True:
            count, pos = _read_long(data, pos)
            if count == 0:
                return items, pos
            if count < 0:
                count = -count
                _, pos = _read_long(data, pos)          # byte size of the block
            for _ in range(count):
                item, pos = decode(data, pos, schema["items"], registry)
                items.append(item)
    if kind == "map":
        entries: Dict[str, Any] = {}
        while True:
            count, pos = _read_long(data, pos)
            if count == 0:
                return entries, pos
            if count < 0:
                count = -count
                _, pos = _read_long(data, pos)
            for _ in range(count):
                key, pos = decode(data, pos, "string", registry)
                entries[key], pos = decode(data, pos, schema["values"], registry)
    if kind == "record":
        record: Dict[str, Any] = {}
        for fld in schema["fields"]:
            record[fld["name"]], pos = decode(data, pos, fld["type"], registry)
        return record, pos
    raise ValueError(f"unsupported Avro type {kind!r}")


# -- object container files ----------------------------------------------------
def _sync_marker() -> bytes:
    return os.urandom(16)


def write_container(handle: BinaryIO, schema: Dict[str, Any], records: Iterable[Dict[str, Any]],
                    codec: str = "deflate", block_size: int = 64) -> int:
    """Write records as an Avro object container; returns the record count."""
    if codec not in ("null", "deflate"):
        raise ValueError("codec must be 'null' or 'deflate'")
    registry: Dict[str, Any] = {}
    _named_types(schema, registry)
    sync = _sync_marker()
    header = io.BytesIO()
    header.write(MAGIC)
    encode({"avro.schema": json.dumps(schema).encode("utf-8"), "avro.codec": codec.encode()},
           {"type": "map", "values": "bytes"}, registry, header)
    header.write(sync)
    handle.write(header.getvalue())

    total = 0
    batch = io.BytesIO()
    count = 0

    def flush() -> None:
        nonlocal batch, count
        if count == 0:
            return
        payload = batch.getvalue()
        if codec == "deflate":
            compressor = zlib.compressobj(6, zlib.DEFLATED, -15)      # raw deflate
            payload = compressor.compress(payload) + compressor.flush()
        block = io.BytesIO()
        _write_long(block, count)
        _write_long(block, len(payload))
        block.write(payload)
        block.write(sync)
        handle.write(block.getvalue())
        batch, count = io.BytesIO(), 0

    for record in records:
        encode(record, schema, registry, batch)
        count += 1
        total += 1
        if count >= block_size:
            flush()
    flush()
    return total


def read_container(handle: BinaryIO) -> Tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """``(schema, records)`` from an Avro object container."""
    data = handle.read()
    if not data.startswith(MAGIC):
        raise ValueError("not an Avro object container")
    registry: Dict[str, Any] = {}
    meta, pos = decode(data, 4, {"type": "map", "values": "bytes"}, registry)
    schema = json.loads(meta["avro.schema"].decode("utf-8"))
    codec = meta.get("avro.codec", b"null").decode("utf-8")
    if codec not in ("null", "deflate"):
        raise ValueError(f"unsupported Avro codec {codec!r}")
    _named_types(schema, registry)
    sync = data[pos:pos + 16]
    pos += 16

    def records() -> Iterator[Dict[str, Any]]:
        cursor = pos
        while cursor < len(data):
            count, cursor = _read_long(data, cursor)
            size, cursor = _read_long(data, cursor)
            payload = data[cursor:cursor + size]
            cursor += size
            if data[cursor:cursor + 16] != sync:
                raise ValueError("Avro sync marker mismatch: the file is corrupt")
            cursor += 16
            if codec == "deflate":
                payload = zlib.decompress(payload, -15)
            inner = 0
            for _ in range(count):
                record, inner = decode(payload, inner, schema, registry)
                yield record

    return schema, records()


# -- the public face, with fastavro when it is there ---------------------------
def write_avro(path: str, schema: Dict[str, Any], records: Iterable[Dict[str, Any]],
               codec: str = "deflate", prefer_fastavro: bool = True) -> int:
    """Write an Avro file; fastavro when installed, this module otherwise."""
    records = list(records)
    fastavro = try_import("fastavro") if prefer_fastavro else None
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if fastavro is not None:
        parsed = fastavro.parse_schema(schema)
        with open(path, "wb") as handle:
            fastavro.writer(handle, parsed, records, codec=codec)
        return len(records)
    with open(path, "wb") as handle:
        return write_container(handle, schema, records, codec=codec)


def read_avro(path: str, prefer_fastavro: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(schema, records)`` from an Avro file, with fastavro when installed."""
    fastavro = try_import("fastavro") if prefer_fastavro else None
    if fastavro is not None:
        with open(path, "rb") as handle:
            reader = fastavro.reader(handle)
            schema = reader.writer_schema
            records = [dict(r) for r in reader]
        if isinstance(schema, dict) and "__fastavro_parsed" in schema:
            schema = _strip_fastavro(schema)
        return schema, records
    with open(path, "rb") as handle:
        schema, records = read_container(handle)
        return schema, list(records)


def _strip_fastavro(schema: Any) -> Any:
    """fastavro annotates parsed schemas; give back plain Avro JSON."""
    if isinstance(schema, dict):
        return {k: _strip_fastavro(v) for k, v in schema.items()
                if not k.startswith("__")}
    if isinstance(schema, list):
        return [_strip_fastavro(v) for v in schema]
    return schema


def schema_of(path: str) -> Dict[str, Any]:
    """The schema embedded in a file, without decoding a single record."""
    with open(path, "rb") as handle:
        data = handle.read(1 << 20)
    if not data.startswith(MAGIC):
        raise ValueError("not an Avro object container")
    meta, _ = decode(data, 4, {"type": "map", "values": "bytes"}, {})
    return json.loads(meta["avro.schema"].decode("utf-8"))


def parse_schema_names(schema: Any) -> List[str]:
    """Field names of a record schema, for summaries."""
    if isinstance(schema, dict) and schema.get("type") == "record":
        return [f["name"] for f in schema.get("fields", [])]
    return []


__all__ = ["encode", "decode", "write_container", "read_container", "write_avro",
           "read_avro", "schema_of", "parse_schema_names"]

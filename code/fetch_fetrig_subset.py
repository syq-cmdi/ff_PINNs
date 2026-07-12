#!/usr/bin/env python3
"""Fetch a small, reproducible subset from the 5.4 GB FETRIG ZIP archive.

The DaRUS archive supports HTTP byte ranges.  This script reads the ZIP64
central directory and downloads only the gas-free Keyence traces used in the
paper (three downstream stations and six liquid Reynolds numbers), together
with the data-manager scripts and README.  The full archive is never stored.

Source: https://doi.org/10.18419/DARUS-4998 (CC BY 4.0)
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import re
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


DATAFILE_ID = 379564
DOI = "10.18419/DARUS-4998"
DEFAULT_URL = f"https://darus.uni-stuttgart.de/api/access/datafile/{DATAFILE_ID}"
TRACE_RE = re.compile(
    r"/Y_(100|350|600)/X_50/Keyence Data(?: [23])?/"
    r"ReL(500|620|740|860|980|1100)_ReG0\.csv$"
)
ANCILLARY_RE = re.compile(
    r"/(?:Keyence_Data_Manager\.m|LabView_Data_Manager\.m|README\.md)$"
)


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    method: int
    local_offset: int
    crc32: int
    flags: int


class RangeReader:
    def __init__(self, url: str, retries: int = 5) -> None:
        self.url = url
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "KS-PINN-reproducibility/2026 (+research use)",
            }
        )

    def get(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range {start}-{end}")
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 180),
                    allow_redirects=True,
                )
                response.raise_for_status()
                payload = response.content
                expected = end - start + 1
                if response.status_code != 206 or len(payload) != expected:
                    raise RuntimeError(
                        f"range request returned status={response.status_code}, "
                        f"bytes={len(payload)}, expected={expected}"
                    )
                return payload
            except (requests.RequestException, RuntimeError) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"failed to fetch byte range {start}-{end}") from error

    def size(self) -> int:
        response = self.session.get(
            self.url,
            headers={"Range": "bytes=0-0"},
            timeout=(30, 180),
            allow_redirects=True,
        )
        response.raise_for_status()
        match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match:
            raise RuntimeError("server did not expose a byte-range Content-Range")
        return int(match.group(1))


def zip64_central_directory(reader: RangeReader) -> tuple[int, int, int]:
    total_size = reader.size()
    tail_size = min(total_size, 256 * 1024)
    tail_start = total_size - tail_size
    tail = reader.get(tail_start, total_size - 1)
    signature = b"PK\x06\x06"
    index = tail.rfind(signature)
    if index < 0:
        raise RuntimeError("ZIP64 end-of-central-directory record not found")
    record = tail[index : index + 56]
    if len(record) < 56:
        raise RuntimeError("truncated ZIP64 end-of-central-directory record")
    entries = struct.unpack_from("<Q", record, 32)[0]
    central_size = struct.unpack_from("<Q", record, 40)[0]
    central_offset = struct.unpack_from("<Q", record, 48)[0]
    return central_offset, central_size, entries


def _zip64_values(extra: bytes) -> list[int]:
    pos = 0
    while pos + 4 <= len(extra):
        tag, length = struct.unpack_from("<HH", extra, pos)
        pos += 4
        body = extra[pos : pos + length]
        pos += length
        if tag == 0x0001:
            if len(body) % 8:
                raise RuntimeError("malformed ZIP64 extended-information field")
            return list(struct.unpack(f"<{len(body) // 8}Q", body))
    return []


def parse_central_directory(blob: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    pos = 0
    while pos < len(blob):
        if blob[pos : pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"bad central-directory signature at byte {pos}")
        flags, method = struct.unpack_from("<HH", blob, pos + 8)
        crc32, csize32, usize32 = struct.unpack_from("<III", blob, pos + 16)
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", blob, pos + 28)
        local32 = struct.unpack_from("<I", blob, pos + 42)[0]
        name_start = pos + 46
        name_bytes = blob[name_start : name_start + name_len]
        extra = blob[name_start + name_len : name_start + name_len + extra_len]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)

        values = iter(_zip64_values(extra))
        usize = next(values) if usize32 == 0xFFFFFFFF else usize32
        csize = next(values) if csize32 == 0xFFFFFFFF else csize32
        local_offset = next(values) if local32 == 0xFFFFFFFF else local32
        entries.append(
            ZipEntry(name, csize, usize, method, local_offset, crc32, flags)
        )
        pos = name_start + name_len + extra_len + comment_len
    return entries


def extract_entry(reader: RangeReader, entry: ZipEntry) -> bytes:
    header = reader.get(entry.local_offset, entry.local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {entry.name}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    data_start = entry.local_offset + 30 + name_len + extra_len
    compressed = reader.get(data_start, data_start + entry.compressed_size - 1)
    if entry.method == 0:
        raw = compressed
    elif entry.method == 8:
        raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported compression method {entry.method}")
    if len(raw) != entry.uncompressed_size:
        raise RuntimeError(f"uncompressed-size mismatch for {entry.name}")
    if (binascii.crc32(raw) & 0xFFFFFFFF) != entry.crc32:
        raise RuntimeError(f"CRC-32 mismatch for {entry.name}")
    return raw


def output_name(entry: ZipEntry) -> Path:
    trace = TRACE_RE.search(entry.name)
    if trace:
        y_mm, re_l = trace.groups()
        return Path(f"Y_{y_mm}") / f"ReL{re_l}_ReG0.csv"
    return Path("ancillary") / Path(entry.name).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experimental/fetrig"),
    )
    args = parser.parse_args()

    reader = RangeReader(args.url)
    central_offset, central_size, archive_entries = zip64_central_directory(reader)
    central = reader.get(central_offset, central_offset + central_size - 1)
    entries = parse_central_directory(central)
    if len(entries) != archive_entries:
        raise RuntimeError(
            f"central directory has {len(entries)} entries; expected {archive_entries}"
        )
    selected = [
        entry
        for entry in entries
        if TRACE_RE.search(entry.name) or ANCILLARY_RE.search(entry.name)
    ]
    trace_count = sum(bool(TRACE_RE.search(entry.name)) for entry in selected)
    if trace_count != 18:
        raise RuntimeError(f"expected 18 gas-free Keyence traces, found {trace_count}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []
    for entry in sorted(selected, key=lambda item: item.name):
        raw = extract_entry(reader, entry)
        destination = args.output / output_name(entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        manifest_entries.append(
            {
                **asdict(entry),
                "output": str(destination.relative_to(args.output)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        print(f"extracted {destination} ({len(raw):,} bytes)")

    manifest = {
        "source_doi": DOI,
        "source_url": args.url,
        "datafile_id": DATAFILE_ID,
        "license": "CC BY 4.0",
        "archive_entries": archive_entries,
        "central_directory_offset": central_offset,
        "central_directory_size": central_size,
        "selection": (
            "ReG=0 Keyence film-thickness traces at Y=100, 350, and 600 mm; "
            "ReL=500, 620, 740, 860, 980, and 1100"
        ),
        "files": manifest_entries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

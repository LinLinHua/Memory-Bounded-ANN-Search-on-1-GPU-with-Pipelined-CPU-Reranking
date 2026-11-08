#!/usr/bin/env python3
"""Download the DEEP1B 10M prefix and public queries without fetching all 1B rows.

Uses only Python's standard library. Range responses are checked before writing;
interrupted downloads resume from .part files. Existing final files are validated.
"""

import argparse
import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
import urllib.request


SOURCE = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/"
CHUNK = 32 * 1024 * 1024
FILES = (
    ("base.1B.fbin", "base.10M.fbin", 10_000_000, 96),
    ("query.public.10K.fbin", "query.public.10K.fbin", 10_000, 96),
)


def fetch_range(url, start, end, expected_total=None):
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}",
                              "Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=120) as response:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                     response.headers.get("Content-Range", ""))
                if response.status != 206 or not match:
                    raise ValueError("Server did not honor HTTP Range; refusing full download")
                first, last, total = map(int, match.groups())
                if (first, last) != (start, end):
                    raise ValueError("Unexpected response byte range")
                if expected_total is not None and total != expected_total:
                    raise ValueError("Source size changed or does not match its fbin header")
                data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete or oversized range response")
                return data, total
        except Exception as error:
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}: {error}", flush=True)
            time.sleep(2 ** attempt)


def download(output_dir, remote, local, rows, dimensions):
    path = output_dir / local
    if path.exists():
        print(f"Existing {path}; will validate it", flush=True)
        return
    url = SOURCE + remote
    source_header, total = fetch_range(url, 0, 7)
    source_rows, source_dimensions = struct.unpack("<ii", source_header)
    if source_rows < rows or source_dimensions != dimensions:
        raise ValueError(f"Unexpected source header: {source_rows} x {source_dimensions}")
    if total != 8 + source_rows * dimensions * 4:
        raise ValueError("Source file size does not match header")
    expected_size = 8 + rows * dimensions * 4
    header = struct.pack("<ii", rows, dimensions)
    partial = path.with_name(path.name + ".part")
    if not partial.exists():
        with partial.open("xb") as stream:
            stream.write(header)
    with partial.open("r+b") as stream:
        if stream.read(8) != header:
            raise ValueError(f"Unexpected partial file header: {partial}")
        offset = stream.seek(0, os.SEEK_END)
        if offset < 8 or offset > expected_size:
            raise ValueError(f"Invalid partial file size: {partial}")
        # Keep four requests in flight, but append in order so the partial file
        # always represents a contiguous prefix and can resume after interruption.
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = deque()
            requested = offset
            while offset < expected_size:
                while len(pending) < 4 and requested < expected_size:
                    end = min(requested + CHUNK, expected_size) - 1
                    pending.append(executor.submit(fetch_range, url, requested, end, total))
                    requested = end + 1
                data, _ = pending.popleft().result()
                stream.write(data)
                stream.flush()
                offset += len(data)
                print(f"{local}: {offset:,}/{expected_size:,} bytes "
                      f"({offset / expected_size:.1%})", flush=True)
        os.fsync(stream.fileno())
    os.chmod(partial, 0o664)
    os.replace(partial, path)


def validate(path, rows, dimensions):
    expected_size = 8 + rows * dimensions * 4
    if path.stat().st_size != expected_size:
        raise ValueError(f"Wrong file size: {path}")
    digest = hashlib.sha256()
    minimum, maximum = math.inf, -math.inf
    with path.open("rb") as stream:
        header = stream.read(8)
        if struct.unpack("<ii", header) != (rows, dimensions):
            raise ValueError(f"Wrong fbin header: {path}")
        digest.update(header)
        count = 0
        while True:
            data = stream.read(CHUNK)
            if not data:
                break
            digest.update(data)
            values = array.array("f")
            values.frombytes(data)
            if sys.byteorder != "little":
                values.byteswap()
            if not all(map(math.isfinite, values)):
                raise ValueError(f"NaN or infinity found in {path}")
            minimum = min(minimum, min(values))
            maximum = max(maximum, max(values))
            count += len(values)
    if count != rows * dimensions:
        raise ValueError(f"Unexpected float count in {path}")
    result = {"file": path.name, "rows": rows, "dimensions": dimensions,
              "dtype": "little-endian float32", "bytes": expected_size,
              "sha256": digest.hexdigest(), "all_finite": True,
              "min": minimum, "max": maximum}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/deep10m"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "selection": "first 10,000,000 base rows, original order",
                "checksum_note": "Locally computed SHA-256, not a published source checksum",
                "files": []}
    for remote, local, rows, dimensions in FILES:
        if not args.verify_only:
            download(args.output_dir, remote, local, rows, dimensions)
        result = validate(args.output_dir / local, rows, dimensions)
        result["source_url"] = SOURCE + remote
        manifest["files"].append(result)
    path = args.output_dir / "manifest.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(temporary, 0o664)
    os.replace(temporary, path)
    print(f"Ready: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

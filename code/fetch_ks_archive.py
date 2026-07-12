#!/usr/bin/env python3
"""Fetch the pinned MIT-licensed KS archive and verify its SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "13ce051750dbc2c9afa3b105b09f373c524ef272"
URL = f"https://raw.githubusercontent.com/maziarraissi/PINNs/{COMMIT}/main/Data/KS.mat"
SHA256 = "a47286bb83b79d1ac0d46fe818693a627df411556166afe454386d88ad5d1ac7"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "code" / "KS_raissi.mat")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        observed = digest(output)
        if observed != SHA256:
            raise RuntimeError(
                f"existing file has unexpected SHA-256 {observed}; refusing to overwrite {output}"
            )
        print(f"verified existing {output} ({SHA256})")
        return

    temporary = output.with_suffix(output.suffix + ".download")
    request = urllib.request.Request(URL, headers={"User-Agent": "ff-PINNs-reproducibility/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        observed = digest(temporary)
        if observed != SHA256:
            raise RuntimeError(f"download SHA-256 mismatch: expected {SHA256}, observed {observed}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"downloaded and verified {output} ({SHA256})")


if __name__ == "__main__":
    main()

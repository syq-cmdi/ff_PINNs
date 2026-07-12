#!/usr/bin/env python3
"""Build or verify the strict code-only release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
FORBIDDEN_PARTS = {"data", "figures", "output", "paper", "results", "submission"}
IGNORED_PARTS = {".git", ".venv", "__pycache__", "reproduced", "tmp", "venv"}
FORBIDDEN_SUFFIXES = {
    ".docx",
    ".mat",
    ".npz",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def released_files() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or IGNORED_PARTS.intersection(relative.parts)
            or path == MANIFEST
        ):
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            raise RuntimeError(f"forbidden code-only path: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"forbidden code-only file: {relative.as_posix()}")
        paths.append(path)
    return paths


def build() -> dict[str, object]:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in released_files()
    ]
    return {
        "schema": "ff_pinns_code_only_manifest_v1",
        "release_policy": (
            "source code and support metadata only; no manuscripts, submission "
            "documents, data, results, figures, checkpoints, or model weights"
        ),
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    observed = build()
    if args.check:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if observed != expected:
            raise SystemExit("RELEASE_MANIFEST.json does not match the code-only tree")
        print(f"verified code-only manifest: {observed['file_count']} files")
        return
    MANIFEST.write_text(
        json.dumps(observed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote code-only manifest: {observed['file_count']} files")


if __name__ == "__main__":
    main()

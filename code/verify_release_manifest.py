#!/usr/bin/env python3
"""Verify every allow-listed file in a public release SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


IGNORED_PARTS = {".git", ".venv", "__pycache__", "reproduced", "tmp", "venv"}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("RELEASE_MANIFEST.json"))
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest files must be a non-empty list")

    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        name = str(record["path"])
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or name in seen:
            errors.append(f"unsafe or duplicate path: {name}")
            continue
        seen.add(name)
        path = root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or non-regular file: {name}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"size mismatch: {name}")
        if sha256(path) != record["sha256"]:
            errors.append(f"SHA-256 mismatch: {name}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != manifest_path.name
        and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
    }
    missing = sorted(seen - actual)
    extra = sorted(actual - seen)
    errors.extend(f"manifest entry missing from tree: {name}" for name in missing)
    errors.extend(f"unlisted file in release tree: {name}" for name in extra)

    report = {
        "status": "passed" if not errors else "failed",
        "file_count": len(records),
        "actual_file_count": len(actual),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

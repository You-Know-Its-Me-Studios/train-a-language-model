"""Verify course download files using Python's standard library."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest", type=Path,
                        default=Path("DOWNLOAD-MANIFEST.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported download manifest format.")
    failures = []
    files = manifest["files"]
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"Missing or invalid path: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            failures.append(f"Size mismatch: {relative}")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print(f"Verified {len(files)} supplied files.")
    print("Download verification passed.")


if __name__ == "__main__":
    main()

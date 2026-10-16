import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prepared"
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
LICENSE_URL = "https://cdla.dev/sharing-1-0/"
SEED = 42
MARKER = b"<|endoftext|>"
SOURCES = {
    "train": {
        "bytes": 25_000_000,
        "sha256": "4f58be27e7c321948886b8d10bbfe270df428609a847569ecb6ce8ea7f472caa",
    },
    "valid": {
        "bytes": 22_502_601,
        "sha256": "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
    },
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_url(name):
    return f"{BASE}/resolve/{REVISION}/TinyStoriesV2-GPT4-{name}.txt"


def download(name):
    spec = SOURCES[name]
    path = RAW / f"{name}.txt"
    if path.exists():
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise RuntimeError(f"Cached file failed verification: {path}")
        print(f"Verified cached source: {name}", flush=True)
        return path

    temporary = path.with_suffix(".txt.part")
    request = Request(source_url(name), headers={
        "Range": f"bytes=0-{spec['bytes'] - 1}",
        "Accept-Encoding": "identity",
        "User-Agent": "LocalLM-Course/1.0",
    })
    print(f"Downloading {name}: {spec['bytes']:,} bytes", flush=True)
    try:
        with urlopen(request, timeout=60) as response:
            if response.status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status: {response.status}")
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith("bytes 0-"):
                    raise RuntimeError("The server returned the wrong byte range.")
            remaining = spec["bytes"]
            with temporary.open("wb") as stream:
                while remaining:
                    block = response.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError(f"Download ended early: {name}")
                    stream.write(block)
                    remaining -= len(block)
        if digest(temporary) != spec["sha256"]:
            raise RuntimeError(f"Downloaded file failed SHA256 verification: {name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified downloaded source: {name}", flush=True)
    return path


def read_unique(path, seen, is_prefix):
    pieces = path.read_bytes().split(MARKER)
    audit = {"empty": 0, "invalid_text": 0, "duplicates": 0,
             "discarded_tail_bytes": 0, "kept": 0}
    if is_prefix:
        audit["discarded_tail_bytes"] = len(pieces.pop())
    stories = []
    for piece in pieces:
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError:
            audit["invalid_text"] += 1
            continue
        if "\ufffd" in text:
            audit["invalid_text"] += 1
            continue
        text = " ".join(text.split())
        if not text:
            audit["empty"] += 1
            continue
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in seen:
            audit["duplicates"] += 1
            continue
        seen.add(key)
        stories.append(text)
    audit["kept"] = len(stories)
    return stories, audit


def write_split(name, stories):
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for text in stories:
            stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return {"stories": len(stories), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for name in ("train", "val", "test"):
        path = OUT / f"{name}.jsonl"
        expected = manifest["splits"][name]
        if digest(path) != expected["sha256"]:
            raise RuntimeError(f"Prepared file failed SHA256 verification: {name}")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                text = json.loads(line)["text"]
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Invalid story in {name}")
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if key in seen:
                    raise RuntimeError(f"Duplicate story found in {name}")
                seen.add(key)
                count += 1
        if count != expected["stories"]:
            raise RuntimeError(f"Story count changed: {name}")
        print(f"{name}: {count:,} stories verified")
    print("Dataset verification passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if OUT.exists():
        raise RuntimeError("Prepared folder already exists. Use --verify, or preserve it before rebuilding.")
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {name: download(name) for name in SOURCES}
    seen = set()
    train, train_audit = read_unique(paths["train"], seen, is_prefix=True)
    held, held_audit = read_unique(paths["valid"], seen, is_prefix=False)
    if not train or len(held) < 7000:
        raise RuntimeError("Not enough clean stories to create the course splits.")
    random.Random(SEED).shuffle(held)
    splits = {"train": train, "val": held[:5000], "test": held[5000:7000]}
    OUT.mkdir()
    manifest = {
        "dataset": "TinyStories V2 GPT4",
        "source": BASE,
        "revision": REVISION,
        "license": "CDLA-Sharing-1.0",
        "license_url": LICENSE_URL,
        "sampling": "Training prefix; complete upstream validation file.",
        "cleaning": "Strict UTF-8; reject replacement characters; collapse whitespace.",
        "deduplication": "SHA256 of normalized text across all retained stories.",
        "seed": SEED,
        "python_version": sys.version,
        "preparation_script_sha256": digest(Path(__file__)),
        "source_files": {
            name: {"url": source_url(name), **spec}
            for name, spec in SOURCES.items()
        },
        "audit": {"train": train_audit, "held_out": held_audit},
        "unused_held_out_stories": len(held) - 7000,
        "splits": {name: write_split(name, stories)
                   for name, stories in splits.items()},
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    verify()
    print(f"Saved dataset: {OUT}")


if __name__ == "__main__":
    main()

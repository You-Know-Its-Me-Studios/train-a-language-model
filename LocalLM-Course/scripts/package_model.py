import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default="notes/selected-model.json"
    )
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    selection_path = project_path(args.selection)
    selection = read_json(selection_path)
    checkpoint = project_path(selection["checkpoint_path"])
    expected_hash = selection["checkpoint_sha256"].lower()
    if digest(checkpoint) != expected_hash:
        raise RuntimeError("The selected checkpoint has changed.")
    run = checkpoint.parent.parent
    validation_path = project_path(selection["validation_report"])
    samples_path = project_path(selection["samples_report"])
    test_path = project_path(args.test_report)
    validation = read_json(validation_path)
    samples = read_json(samples_path)
    test = read_json(test_path)
    for report in (validation, samples, test):
        if report["checkpoint_sha256"] != expected_hash:
            raise RuntimeError("Report belongs to another checkpoint.")
    if validation["split"] != "val" or test["split"] != "test":
        raise RuntimeError("Expected validation and test reports.")
    if validation["protocol"] != test["protocol"]:
        raise RuntimeError("Evaluation protocols differ.")
    if samples["protocol"] != "fixed-story-prompts-v1":
        raise RuntimeError("Expected the fixed-prompt sample report.")
    identity = validation["training_fingerprint"]
    if digest(run / "tokenizer.json") != identity["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer differs from the evaluated run.")
    if digest(ROOT / "scripts/model.py") != (
        identity["sources"]["model.py"]
    ):
        raise RuntimeError("Model source differs from the saved run.")
    bundle = project_path(args.output_dir).resolve()
    archive = Path(str(bundle) + ".zip")
    checksum = Path(str(archive) + ".sha256")
    if any(path.exists() for path in (bundle, archive, checksum)):
        raise FileExistsError("Choose an unused package folder name.")
    files = {
        "checkpoints/model.pt": checkpoint,
        "tokenizer.json": run / "tokenizer.json",
        "README.md": ROOT / "model-README.md",
        "records/selection.json": selection_path,
        "records/validation.json": validation_path,
        "records/samples.json": samples_path,
        "records/test.json": test_path,
        "records/training-config.json": run / "config.json",
        "records/run.json": run / "run.json",
        "records/token-manifest.json": run / "manifest.json",
    }
    for name in ("model.py", "inference.py", "generate_text.py"):
        files[f"scripts/{name}"] = ROOT / "scripts" / name
    for name in ("cpu", "tokenizer", "model"):
        filename = f"requirements-{name}.txt"
        files[filename] = ROOT / filename
    for name, expected in identity["sources"].items():
        if Path(name).name != name:
            raise ValueError("Invalid training source filename.")
        source = run / "sources" / name
        if digest(source) != expected:
            raise RuntimeError(f"Training source changed: {name}")
        files[f"records/training-sources/{name}"] = source
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    bundle.mkdir(parents=True)
    for relative, source in files.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if digest(bundle / "checkpoints/model.pt") != expected_hash:
        raise RuntimeError("The copied checkpoint failed verification.")
    installed = "\n".join(sorted(
        f"{item.metadata['Name']}=={item.version}"
        for item in distributions()
    ))
    (bundle / "records/inference-environment.txt").write_text(
        "Python: " + sys.version + "\n\n" + installed,
        encoding="utf-8",
    )
    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": expected_hash,
        "files": {
            path.relative_to(bundle).as_posix(): digest(path)
            for path in sorted(bundle.rglob("*")) if path.is_file()
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    shutil.make_archive(
        str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name
    )
    archive_hash = digest(archive)
    checksum.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="ascii"
    )
    print("Package folder:", bundle)
    print("Archive:", archive)
    print("SHA-256:", archive_hash)
    print("Checksum file:", checksum)


if __name__ == "__main__":
    main()
